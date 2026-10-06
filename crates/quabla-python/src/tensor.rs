use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyEllipsis, PyMemoryView, PySlice, PySliceMethods, PyTuple};
use quabla_core::tensor_ir::{
    EagerKernel, EagerOperand, HostTensorStorage, TensorComparison, TensorDType, UnaryMathKind,
};
use std::borrow::Cow;
use std::ffi::c_int;
#[cfg(test)]
use std::sync::Arc;

use crate::composite::{self, Primitives};
use crate::dtype::PyDType;
use crate::interop;
use crate::tensor_trace::{eager_traced_binary, TraceTensor, TracedBinary};

/// The result of an eager `Tensor` operation. It is traced when the other
/// operand is a tracer: the tensor is then captured as a constant of the
/// tracer's graph, as a `jax.jit` closure constant is.
#[derive(IntoPyObject)]
pub enum EagerOrTraced {
    Eager(PyTensor),
    Traced(TraceTensor),
    /// `NotImplemented` from a binary operator, so that Python tries the
    /// reflected method of an operand this type does not know.
    NotImplemented(Py<PyAny>),
}

/// Whether a binary operator can handle `operand` itself: a `Tensor`, or a
/// tracer that captures this tensor.
fn is_array_operand(operand: &Bound<'_, PyAny>) -> bool {
    operand.is_instance_of::<PyTensor>() || operand.is_instance_of::<TraceTensor>()
}

/// Wraps an eager result, mapping its error to `ValueError`.
fn eager(result: Result<PyTensor, String>) -> PyResult<EagerOrTraced> {
    result
        .map(EagerOrTraced::Eager)
        .map_err(PyValueError::new_err)
}

/// Runs `op` with a traced `operand` (see [`eager_traced_binary`]).
fn traced(
    tensor: &PyTensor,
    operand: &Bound<'_, PyAny>,
    op: TracedBinary,
    tensor_first: bool,
) -> Option<PyResult<EagerOrTraced>> {
    eager_traced_binary(tensor, operand, op, tensor_first)
        .map(|result| result.map(EagerOrTraced::Traced))
}

/// Eager host tensor with physical storage matching its logical dtype.
/// Every op evaluates the Tensor IR nodes that a trace of it records with the
/// core's CPU evaluator, so eager results, dtypes, weak types, and errors are
/// those of CPU `jit`. Python scalars are weak operands, and a `bool` tensor
/// holds `0.0`/`1.0` and follows the Tensor IR promotion rule.
#[pyclass(name = "Tensor", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTensor {
    shape: Vec<usize>,
    data: HostTensorStorage,
    dtype: TensorDType,
    /// JAX-style weak type: set only when a `bool` tensor meets a Python
    /// scalar (`mask * 2.0`), so the result still adopts a later strong
    /// operand's dtype. Elementwise arithmetic propagates it.
    weak: bool,
}

#[pyclass(name = "TensorView", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTensorView {
    data: HostTensorStorage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    dtype: TensorDType,
}

#[derive(Clone, Copy, Debug)]
pub enum TensorIndex {
    Integer(usize),
    Slice { start: usize, stop: usize },
}

/// A parsed `__getitem__` key, applied in three steps: the integer and
/// contiguous-slice `indices`, then one gather per strided slice (`(axis,
/// positions)` on the result of the first step, positions relative to the
/// slice start), then a reshape to `shape` when the key inserts new axes.
/// Strided slices and `None` reuse the differentiable slice, gather, and
/// reshape nodes, so every key form works eagerly and under every transform.
pub struct IndexPlan {
    pub indices: Vec<TensorIndex>,
    pub gathers: Vec<(usize, Vec<usize>)>,
    pub shape: Option<Vec<usize>>,
}

/// Parses an index key with NumPy basic-indexing rules: integers (negative
/// ones count from the end) drop their axis, non-empty slices of any step
/// keep it, `None` inserts a unit axis, one `...` stands for as many full
/// slices as needed, and axes left unindexed at the end are kept whole.
pub fn parse_index_plan(index: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<IndexPlan> {
    let rank = shape.len();
    let items = if let Ok(tuple) = index.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else {
        vec![index.clone()]
    };
    let ellipsis = PyEllipsis::get(index.py());
    let is_ellipsis = |item: &Bound<'_, PyAny>| item.is(ellipsis);
    if items.iter().filter(|item| is_ellipsis(item)).count() > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis ('...')",
        ));
    }
    let consumed = items
        .iter()
        .filter(|item| !item.is_none() && !is_ellipsis(item))
        .count();
    if consumed > rank {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor of rank {rank}: got {consumed}"
        )));
    }
    let mut indices = Vec::with_capacity(consumed);
    let mut gathers = Vec::new();
    // Extents of the final result, with the unit axes of `None` included.
    let mut output_shape = Vec::with_capacity(rank + items.len());
    let mut new_axes = false;
    // The next input axis, and the axis count of the first step's result.
    let mut axis = 0;
    let mut kept = 0;
    for item in items {
        if item.is_none() {
            output_shape.push(1);
            new_axes = true;
            continue;
        }
        if is_ellipsis(&item) {
            for _ in 0..rank - consumed {
                indices.push(TensorIndex::Slice {
                    start: 0,
                    stop: shape[axis],
                });
                output_shape.push(shape[axis]);
                axis += 1;
                kept += 1;
            }
            continue;
        }
        let extent = shape[axis];
        if let Ok(value) = item.extract::<isize>() {
            let normalized = if value < 0 {
                extent as isize + value
            } else {
                value
            };
            let value = usize::try_from(normalized)
                .ok()
                .filter(|value| *value < extent)
                .ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "index {value} is out of bounds for axis {axis} with extent {extent}"
                    ))
                })?;
            indices.push(TensorIndex::Integer(value));
            axis += 1;
            continue;
        }
        if let Ok(slice) = item.cast::<PySlice>() {
            let range = slice.indices(extent as isize)?;
            if range.slicelength == 0 {
                return Err(PyValueError::new_err(
                    "Tensor indexing currently rejects empty slices",
                ));
            }
            let length = range.slicelength;
            let first = range.start;
            let last = first + (length as isize - 1) * range.step;
            let (low, high) = (first.min(last), first.max(last) + 1);
            indices.push(TensorIndex::Slice {
                start: low as usize,
                stop: high as usize,
            });
            if range.step != 1 {
                // The contiguous slice spans every selected element; the
                // gather then picks them in order, reversed for a negative step.
                gathers.push((
                    kept,
                    (0..length as isize)
                        .map(|k| (first + k * range.step - low) as usize)
                        .collect(),
                ));
            }
            output_shape.push(length);
            axis += 1;
            kept += 1;
            continue;
        }
        return Err(PyTypeError::new_err(
            "Tensor indexing supports integers, slices, None, and ... only",
        ));
    }
    output_shape.extend_from_slice(&shape[axis..]);
    Ok(IndexPlan {
        indices,
        gathers,
        shape: new_axes.then_some(output_shape),
    })
}

/// Parses the arguments of `reshape`: the extents as separate ints, one int,
/// or one sequence of ints, as in NumPy. One extent may be `-1`; it is
/// inferred from `size`, the element count of the reshaped array.
pub fn parse_reshape_args(args: &Bound<'_, PyTuple>, size: usize) -> PyResult<Vec<usize>> {
    let invalid = || PyTypeError::new_err("reshape expects ints or a single sequence of ints");
    let extents = match args.len() {
        0 => return Err(invalid()),
        1 => {
            let arg = args.get_item(0)?;
            match arg.extract::<isize>() {
                Ok(extent) => vec![extent],
                Err(_) => arg.extract::<Vec<isize>>().map_err(|_| invalid())?,
            }
        }
        _ => args.extract::<Vec<isize>>().map_err(|_| invalid())?,
    };
    let mut inferred = None;
    let mut known = 1usize;
    for (axis, &extent) in extents.iter().enumerate() {
        if extent == -1 {
            if inferred.replace(axis).is_some() {
                return Err(PyValueError::new_err(
                    "reshape can only infer one dimension (-1)",
                ));
            }
        } else {
            let extent = usize::try_from(extent)
                .map_err(|_| PyValueError::new_err(format!("negative reshape extent {extent}")))?;
            known = known.saturating_mul(extent);
        }
    }
    let mut shape = extents
        .iter()
        .map(|&extent| extent.max(0) as usize)
        .collect::<Vec<_>>();
    if let Some(axis) = inferred {
        if known == 0 || !size.is_multiple_of(known) {
            return Err(PyValueError::new_err(format!(
                "cannot reshape an array of size {size} into shape {extents:?}"
            )));
        }
        shape[axis] = size / known;
    }
    Ok(shape)
}

pub fn parse_axis_indices(indices: &Bound<'_, PyAny>, axis_extent: usize) -> PyResult<Vec<usize>> {
    let indices = indices
        .extract::<Vec<isize>>()
        .map_err(|_| PyTypeError::new_err("indices must be a sequence of integers"))?;
    if indices.is_empty() {
        return Err(PyValueError::new_err(
            "gather/scatter indices must not be empty",
        ));
    }
    indices
        .into_iter()
        .map(|index| {
            let normalized = if index < 0 {
                axis_extent as isize + index
            } else {
                index
            };
            usize::try_from(normalized)
                .ok()
                .filter(|index| *index < axis_extent)
                .ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "index {index} is out of bounds for axis extent {axis_extent}"
                    ))
                })
        })
        .collect()
}

/// Same wording as the Tensor IR builder error for arithmetic on `bool`.
pub(crate) fn bool_operation_error(op: &str) -> String {
    format!(
        "{op} is not defined for bool tensors; use logical_and/logical_or/logical_not \
         (& | ~) or convert explicitly with astype"
    )
}

/// A Python number operand of a tensor op. `Tensor` defines `__float__`
/// (for `float(t)`), so a bare `extract::<f64>()` would also accept a
/// single-element `Tensor` and turn it into a weak scalar, dropping its
/// dtype; operand parsing keeps rejecting tensors there as before.
pub(crate) fn extract_scalar(value: &Bound<'_, PyAny>) -> Option<f64> {
    if value.is_instance_of::<PyTensor>() {
        return None;
    }
    value.extract::<f64>().ok()
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.contains(&0) {
        return Err("tensor extents must be greater than zero".to_string());
    }

    shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| "tensor element count overflows usize".to_string())
    })
}

fn next_random_key(key: &mut u64) -> u64 {
    *key = key.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *key;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn uniform_open_unit(key: &mut u64) -> f64 {
    let bits = next_random_key(key) >> 11;
    (bits as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
}

fn standard_normal(key: &mut u64) -> f64 {
    let radius = (-2.0 * uniform_open_unit(key).ln()).sqrt();
    let angle = std::f64::consts::TAU * uniform_open_unit(key);
    radius * angle.cos()
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for axis in (1..shape.len()).rev() {
        strides[axis - 1] = strides[axis] * shape[axis];
    }

    strides
}

fn normalize_axis(axis: isize, rank: usize) -> Result<usize, String> {
    let rank = isize::try_from(rank).map_err(|_| "tensor rank exceeds isize".to_string())?;
    let normalized = if axis < 0 { rank + axis } else { axis };
    if normalized < 0 || normalized >= rank {
        return Err(format!("axis {axis} is out of bounds for rank {rank}"));
    }

    Ok(normalized as usize)
}

fn extract_reduction_axes(axis: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<isize>>> {
    let Some(axis) = axis else {
        return Ok(None);
    };
    if axis.is_none() {
        return Ok(None);
    }
    if let Ok(axis) = axis.extract::<isize>() {
        return Ok(Some(vec![axis]));
    }
    axis.extract::<Vec<isize>>().map(Some).map_err(|_| {
        PyTypeError::new_err("axis must be None, an integer, or a sequence of integers")
    })
}

fn view_offset(flat_index: usize, shape: &[usize], strides: &[usize], offset: usize) -> usize {
    let mut remaining = flat_index;
    let mut physical_offset = offset;

    for axis in (0..shape.len()).rev() {
        let coordinate = remaining % shape[axis];
        remaining /= shape[axis];
        physical_offset += coordinate * strides[axis];
    }

    physical_offset
}

fn sliced_layout(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    axis: usize,
    start: usize,
    length: usize,
    step: usize,
) -> Result<(Vec<usize>, Vec<usize>, usize), String> {
    if axis >= shape.len() {
        return Err(format!(
            "slice axis {axis} is out of bounds for rank {}",
            shape.len()
        ));
    }
    if length == 0 {
        return Err("slice length must be greater than zero".to_string());
    }
    if step == 0 {
        return Err("slice step must be greater than zero".to_string());
    }

    let last = length
        .checked_sub(1)
        .and_then(|count| count.checked_mul(step))
        .and_then(|distance| start.checked_add(distance))
        .ok_or_else(|| "slice bounds overflow usize".to_string())?;
    if start >= shape[axis] || last >= shape[axis] {
        return Err(format!(
            "slice start {start}, length {length}, step {step} is out of bounds for axis {axis} with extent {}",
            shape[axis]
        ));
    }

    let new_offset = offset
        .checked_add(
            start
                .checked_mul(strides[axis])
                .ok_or_else(|| "slice offset overflows usize".to_string())?,
        )
        .ok_or_else(|| "slice offset overflows usize".to_string())?;
    let mut new_shape = shape.to_vec();
    let mut new_strides = strides.to_vec();
    new_shape[axis] = length;
    new_strides[axis] = new_strides[axis]
        .checked_mul(step)
        .ok_or_else(|| "slice stride overflows usize".to_string())?;

    Ok((new_shape, new_strides, new_offset))
}

impl PyTensor {
    pub fn from_shape_data(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if data.len() != expected {
            return Err(format!(
                "tensor data length {} does not match shape {:?} with {expected} elements",
                data.len(),
                shape
            ));
        }

        Ok(Self {
            shape,
            data: HostTensorStorage::from_f64(data, TensorDType::F64),
            dtype: TensorDType::F64,
            weak: false,
        })
    }

    /// A weak `f64` scalar, the eager form of a Python scalar operand.
    pub fn weak_scalar(value: f64) -> Self {
        Self {
            shape: vec![],
            data: HostTensorStorage::from_f64(vec![value], TensorDType::F64),
            dtype: TensorDType::F64,
            weak: true,
        }
    }

    /// Builds a tensor of `dtype`, rounding each value to that dtype.
    pub fn from_shape_data_typed(
        shape: Vec<usize>,
        data: Vec<f64>,
        dtype: TensorDType,
    ) -> Result<Self, String> {
        Self::from_shape_data(shape, data).map(|tensor| tensor.typed(dtype))
    }

    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Whether this tensor is weakly typed (see the `weak` field).
    pub fn is_weak(&self) -> bool {
        self.weak
    }

    /// The tensor with weak type `weak` (see the `weak` field).
    pub(crate) fn with_weak(mut self, weak: bool) -> Self {
        self.weak = weak;
        self
    }

    pub fn shape_data(&self) -> (&[usize], Cow<'_, [f64]>) {
        (&self.shape, self.data.to_f64())
    }

    /// This tensor as a borrowed operand of a direct kernel.
    fn operand(&self) -> EagerOperand<'_> {
        EagerOperand::Array {
            shape: &self.shape,
            storage: &self.data,
        }
    }

    pub fn to_dynamic_tensor(&self) -> Result<quabla_core::tensor_ir::DynamicTensor, String> {
        quabla_core::tensor_ir::DynamicTensor::from_storage(self.shape.clone(), self.data.clone())
    }

    pub fn from_dynamic_tensor(
        tensor: quabla_core::tensor_ir::DynamicTensor,
    ) -> Result<Self, String> {
        let (shape, data, dtype) = tensor.into_parts();
        Ok(Self {
            shape,
            data,
            dtype,
            weak: false,
        })
    }

    /// Rounds the values to `dtype` and records it as a strong dtype (exact
    /// for widening; nonzero and `NaN` become `1.0` for `bool`).
    fn typed(mut self, dtype: TensorDType) -> Self {
        self.data = self.data.into_dtype(dtype);
        self.dtype = dtype;
        self.weak = false;
        self
    }

    fn ensure_not_bool(&self, op: &str) -> Result<(), String> {
        if self.dtype == TensorDType::Bool {
            return Err(bool_operation_error(op));
        }
        Ok(())
    }

    /// Python truthiness: a single-element `bool` tensor converts like
    /// NumPy; float tensors keep the default object truthiness (always true).
    pub fn truthiness(&self) -> Result<bool, String> {
        if self.dtype != TensorDType::Bool {
            return Ok(true);
        }
        match self.data.to_f64().as_ref() {
            [value] => Ok(*value != 0.0),
            _ => Err(format!(
                "the truth value of a bool tensor with shape {:?} is ambiguous; use .any() or .all()",
                self.shape
            )),
        }
    }

    // Eager ops are adapters over the core: an op is the graph that a trace
    // of it records (the `TraceTensor` builder methods), evaluated at once by
    // the CPU evaluator (`TraceTensor::evaluate_eager`), so eager and `jit`
    // results, dtypes, weak types, and errors cannot diverge. An op whose
    // graph is a single node over operands that need no promotion cast calls
    // that node's kernel directly instead (`EagerKernel`), which skips the
    // graph's fixed cost of a few hundred nanoseconds per node, and the
    // composites of `crate::composite` run each of their primitives so.

    /// The traced op `build` evaluated on `operands`.
    fn evaluated<const N: usize>(
        operands: [&Self; N],
        build: impl FnOnce([TraceTensor; N]) -> Result<TraceTensor, String>,
    ) -> Result<Self, String> {
        TraceTensor::evaluate_eager(operands, build)
    }

    /// `kernel` over `operands`, or `None` when the kernel rejects them, so
    /// that the caller's graph route runs the op and reports the builder's
    /// error. The caller ensures that the graph of the op would be exactly
    /// this node: the operands already have the dtypes promotion gives them,
    /// so no cast is recorded. `weak` is the weak type of the operands that
    /// determine the node dtype; a `Compare` node is never weak.
    fn direct<const N: usize>(
        kernel: EagerKernel,
        operands: [EagerOperand<'_>; N],
        weak: bool,
    ) -> Option<Self> {
        let value = kernel.evaluate(&operands).ok()?;
        let weak = weak && !matches!(kernel, EagerKernel::Compare(_));
        Self::from_dynamic_tensor(value)
            .ok()
            .map(|tensor| tensor.with_weak(weak))
    }

    /// The binary op `op` with a tensor operand; `kernel` is its node's
    /// kernel when it has a direct path. Operands of one dtype record no
    /// promotion cast.
    fn binary_op(
        &self,
        rhs: &Self,
        kernel: Option<EagerKernel>,
        op: TracedBinary,
    ) -> Result<Self, String> {
        if let Some(kernel) = kernel.filter(|_| self.dtype == rhs.dtype) {
            let operands = [self.operand(), rhs.operand()];
            if let Some(output) = Self::direct(kernel, operands, self.weak && rhs.weak) {
                return Ok(output);
            }
        }
        Self::evaluated([self, rhs], |[lhs, rhs]| op.apply(&lhs, &rhs))
    }

    /// `self op value`, or `value op self` when `scalar_first`, with a
    /// Python number, which is a weak scalar; `build` records the op. A
    /// floating operand that is strong (or weak `f64`, the only weak float)
    /// casts the scalar to its dtype, which the direct path does by
    /// building the scalar in that dtype.
    fn scalar_op(
        &self,
        value: f64,
        kernel: Option<EagerKernel>,
        scalar_first: bool,
        build: impl FnOnce(&TraceTensor) -> Result<TraceTensor, String>,
    ) -> Result<Self, String> {
        let castable = self.dtype.is_floating() && (!self.weak || self.dtype == TensorDType::F64);
        if let Some(kernel) = kernel.filter(|_| castable) {
            let scalar = EagerOperand::Scalar(self.dtype.round(value), self.dtype);
            let tensor = self.operand();
            let operands = if scalar_first {
                [scalar, tensor]
            } else {
                [tensor, scalar]
            };
            if let Some(output) = Self::direct(kernel, operands, self.weak) {
                return Ok(output);
            }
        }
        Self::evaluated([self], |[tensor]| build(&tensor))
    }

    /// The `TraceTensor::binary` op `op` with a Python number.
    fn scalar_arithmetic(
        &self,
        value: f64,
        op: &'static str,
        scalar_first: bool,
    ) -> Result<Self, String> {
        self.scalar_op(value, arithmetic_kernel(op), scalar_first, |tensor| {
            if scalar_first {
                tensor.scalar_left_binary(value, op)
            } else {
                tensor.scalar_binary(value, op)
            }
        })
    }

    /// A one-operand op with the direct path `kernel` for floating tensors.
    fn unary(
        &self,
        kernel: EagerKernel,
        build: impl FnOnce(&TraceTensor) -> Result<TraceTensor, String>,
    ) -> Result<Self, String> {
        if self.dtype.is_floating() {
            if let Some(output) = Self::direct(kernel, [self.operand()], self.weak) {
                return Ok(output);
            }
        }
        Self::evaluated([self], |[tensor]| build(&tensor))
    }

    pub fn try_add(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(rhs, Some(EagerKernel::Add), TracedBinary::Arithmetic("add"))
    }

    pub fn try_add_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(rhs, "add", false)
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(rhs, Some(EagerKernel::Sub), TracedBinary::Arithmetic("sub"))
    }

    pub fn try_sub_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(rhs, "sub", false)
    }

    pub fn try_scalar_sub(&self, lhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(lhs, "sub", true)
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(rhs, Some(EagerKernel::Mul), TracedBinary::Arithmetic("mul"))
    }

    pub fn try_mul_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(rhs, "mul", false)
    }

    // Division follows IEEE 754 (±inf, NaN), as traced CPU, CUDA and MLX
    // execution do.
    pub fn try_div(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(rhs, Some(EagerKernel::Div), TracedBinary::Arithmetic("div"))
    }

    pub fn try_div_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(rhs, "div", false)
    }

    pub fn try_scalar_div(&self, lhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(lhs, "div", true)
    }

    /// `-x` as `x * -1`, so `-0.0` stays signed; `bool` is rejected.
    pub fn try_negative(&self) -> Result<Self, String> {
        if self.dtype.is_floating() {
            return self.try_mul_scalar(-1.0);
        }
        Self::evaluated([self], |[tensor]| tensor.negative_tensor())
    }

    /// The legacy `gt` mask: `x > y` as 0/1 values of the promoted dtype.
    pub fn try_gt(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(
            rhs,
            Some(EagerKernel::Greater),
            TracedBinary::Arithmetic("greater"),
        )
    }

    pub fn try_gt_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.scalar_arithmetic(rhs, "greater", false)
    }

    /// IEEE comparison producing a strong `bool` tensor.
    pub fn try_compare(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String> {
        self.binary_op(
            rhs,
            Some(EagerKernel::Compare(kind)),
            TracedBinary::Compare(kind),
        )
    }

    /// Compares with a weak Python scalar, which adopts the tensor dtype.
    pub fn try_compare_scalar(&self, rhs: f64, kind: TensorComparison) -> Result<Self, String> {
        self.scalar_op(rhs, Some(EagerKernel::Compare(kind)), false, |tensor| {
            tensor.compare_scalar(rhs, kind)
        })
    }

    /// `lhs && rhs` and `lhs || rhs` of two `bool` tensors, which the graph
    /// records as `where(lhs, rhs, lhs)` and `where(lhs, lhs, rhs)`.
    pub fn try_logical(&self, rhs: &Self, and: bool) -> Result<Self, String> {
        if self.dtype == TensorDType::Bool && rhs.dtype == TensorDType::Bool {
            let weak = self.weak && rhs.weak;
            let (lhs, rhs) = (self.operand(), rhs.operand());
            let operands = if and {
                [lhs, rhs, lhs]
            } else {
                [lhs, lhs, rhs]
            };
            if let Some(output) = Self::direct(EagerKernel::Where, operands, weak) {
                return Ok(output);
            }
        }
        Self::evaluated([self, rhs], |[lhs, rhs]| lhs.logical_tensor(&rhs, and))
    }

    pub fn try_logical_not(&self) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.logical_not_tensor())
    }

    /// `isnan` (`nan`) or `isfinite` of a float tensor as a `bool` tensor.
    /// The graph records `isnan(x)` as the single node `x != x`.
    pub fn try_classify(&self, nan: bool) -> Result<Self, String> {
        if nan && self.dtype.is_floating() {
            let kernel = EagerKernel::Compare(TensorComparison::NotEqual);
            if let Some(output) = Self::direct(kernel, [self.operand(), self.operand()], false) {
                return Ok(output);
            }
        }
        Self::evaluated([self], |[tensor]| tensor.classify_tensor(nan))
    }

    pub fn try_any_all(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        any: bool,
    ) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| {
            tensor.any_all_tensor(axes, keepdims, any)
        })
    }

    /// `where(isnan(x) | (x > y), x, y)`: ties select `y`, and NaN in either
    /// operand propagates.
    pub fn try_maximum(&self, rhs: &Self) -> Result<Self, String> {
        composite::maximum(self, rhs)
    }

    pub fn try_maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        composite::maximum_scalar(self, rhs)
    }

    /// `where(isnan(x) | (y > x), x, y)`: ties select `y`, and NaN in either
    /// operand propagates.
    pub fn try_minimum(&self, rhs: &Self) -> Result<Self, String> {
        composite::minimum(self, rhs)
    }

    pub fn try_minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        composite::minimum_scalar(self, rhs)
    }

    /// `where(mask, on_true, on_false)`. Values of one dtype record no
    /// promotion cast; neither does a weak scalar value (a Python number)
    /// meeting a strong float, whose cast the direct path applies by
    /// rounding the scalar to that dtype.
    pub fn try_where(mask: &Self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        let values = if on_true.dtype == on_false.dtype {
            Some((
                on_true.operand(),
                on_false.operand(),
                on_true.weak && on_false.weak,
            ))
        } else if let Some(scalar) = on_false.scalar_cast_to(on_true) {
            Some((on_true.operand(), scalar, false))
        } else {
            on_true
                .scalar_cast_to(on_false)
                .map(|scalar| (scalar, on_false.operand(), false))
        };
        if let Some((on_true, on_false, weak)) = values {
            let operands = [mask.operand(), on_true, on_false];
            if let Some(output) = Self::direct(EagerKernel::Where, operands, weak) {
                return Ok(output);
            }
        }
        Self::evaluated([mask, on_true, on_false], |[mask, on_true, on_false]| {
            mask.where_tensor(&on_true, &on_false)
        })
    }

    /// This tensor as the promotion cast to `other`'s dtype gives it, when
    /// this is a weak rank-0 `f64` (a Python number) and `other` a strong
    /// float of another dtype.
    fn scalar_cast_to(&self, other: &Self) -> Option<EagerOperand<'static>> {
        (self.weak
            && self.shape.is_empty()
            && self.dtype == TensorDType::F64
            && !other.weak
            && other.dtype.is_floating())
        .then(|| EagerOperand::Scalar(other.dtype.round(self.data.get(0)), other.dtype))
    }

    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(rhs, Some(EagerKernel::Matmul), TracedBinary::Matmul)
    }

    pub fn try_solve(&self, rhs: &Self) -> Result<Self, String> {
        self.binary_op(rhs, None, TracedBinary::Solve)
    }

    pub fn try_solve_triangular(
        &self,
        rhs: &Self,
        lower: bool,
        transpose: bool,
    ) -> Result<Self, String> {
        self.binary_op(
            rhs,
            None,
            TracedBinary::SolveTriangular { lower, transpose },
        )
    }

    pub fn try_linalg(&self, kind: &str) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.linalg_tensor(kind))
    }

    /// The traced `cholesky` factor of each matrix, with the strict input
    /// validation that eager Cholesky documents and the traced op omits (it
    /// reads the lower triangle and returns NaN for an indefinite matrix):
    /// an asymmetric matrix is rejected before, and a non-positive pivot
    /// (a factor diagonal entry that is not positive) after, the
    /// factorization.
    pub fn try_cholesky(&self) -> Result<Self, String> {
        let rank = self.shape.len();
        let square = rank >= 2 && self.shape[rank - 2] == self.shape[rank - 1];
        if !square || !self.dtype.is_floating() {
            return Self::evaluated([self], |[tensor]| tensor.cholesky_tensor());
        }
        let n = self.shape[rank - 1];
        let values = self.data.to_f64();
        for matrix in values.chunks_exact(n * n) {
            for row in 0..n {
                for column in 0..row {
                    let (value, mirrored) = (matrix[row * n + column], matrix[column * n + row]);
                    let tolerance = 1e-12 * value.abs().max(mirrored.abs()).max(1.0);
                    if (value - mirrored).abs() > tolerance {
                        return Err("cholesky requires a symmetric coefficient matrix".to_string());
                    }
                }
            }
        }
        let factor = Self::evaluated([self], |[tensor]| tensor.cholesky_tensor())?;
        let positive = factor
            .data
            .to_f64()
            .chunks_exact(n * n)
            .all(|matrix| (0..n).all(|index| matrix[index * n + index] > 0.0));
        if !positive {
            return Err("cholesky requires a positive-definite coefficient matrix".to_string());
        }
        Ok(factor)
    }

    /// A view of the same storage with another shape (the `Reshape` node's
    /// value), keeping the dtype and weak type.
    pub fn try_reshape(&self, shape: Vec<usize>) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if expected != self.data.len() {
            return Err(format!(
                "cannot reshape tensor with {} elements to shape {:?}",
                self.data.len(),
                shape
            ));
        }

        Ok(Self {
            shape,
            data: self.data.clone(),
            dtype: self.dtype,
            weak: self.weak,
        })
    }

    pub fn try_broadcast_to(&self, shape: Vec<usize>) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.broadcast_to_tensor(shape))
    }

    pub fn try_transpose(&self, axes: Option<Vec<isize>>) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.transpose_tensor(axes))
    }

    pub fn try_sum_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.reduction(axes, keepdims, false)
    }

    pub fn try_mean_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.reduction(axes, keepdims, true)
    }

    /// `sum` or `mean` over `axes`. Every axis or one axis is a single
    /// `Sum`/`Mean` node (plus a reshape for `keepdims`), which runs
    /// directly; several axes reduce one at a time in the graph.
    fn reduction(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        mean: bool,
    ) -> Result<Self, String> {
        let rank = self.shape.len();
        let single = match axes.as_deref() {
            None => Some(None),
            Some(&[axis]) => normalize_axis(axis, rank).ok().map(Some),
            Some(_) => None,
        };
        if let (Some(axis), true) = (single, self.dtype.is_floating()) {
            let kernel = match (axis, mean) {
                (None, false) => EagerKernel::Sum,
                (None, true) => EagerKernel::Mean,
                (Some(axis), false) => EagerKernel::SumAxis(axis),
                (Some(axis), true) => EagerKernel::MeanAxis(axis),
            };
            if let Some(output) = Self::direct(kernel, [self.operand()], self.weak) {
                if !keepdims {
                    return Ok(output);
                }
                let shape = match axis {
                    None => vec![1; rank],
                    Some(axis) => {
                        let mut shape = output.shape.clone();
                        shape.insert(axis, 1);
                        shape
                    }
                };
                return output.try_reshape(shape);
            }
        }
        Self::evaluated([self], |[tensor]| {
            tensor.reduce_axes_tensor(axes, keepdims, mean)
        })
    }

    pub fn try_norm(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.norm_tensor(axes, keepdims))
    }

    pub fn try_prod_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.prod_axes_tensor(axes, keepdims))
    }

    pub fn try_max_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| {
            tensor.extrema_axes_tensor(axes, keepdims, true)
        })
    }

    pub fn try_min_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| {
            tensor.extrema_axes_tensor(axes, keepdims, false)
        })
    }

    pub fn try_tanh(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Tanh, TraceTensor::tanh_tensor)
    }

    pub fn try_exp(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Exp, TraceTensor::exp_tensor)
    }

    pub fn try_log(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Log, TraceTensor::log_tensor)
    }

    pub fn try_log1p(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Log1p, TraceTensor::log1p_tensor)
    }

    pub fn try_expm1(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Expm1, TraceTensor::expm1_tensor)
    }

    pub fn try_erf(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Erf, TraceTensor::erf_tensor)
    }

    pub fn try_erfc(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Erfc, TraceTensor::erfc_tensor)
    }

    pub fn try_sqrt(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Sqrt, TraceTensor::sqrt_tensor)
    }

    pub fn try_sin(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Sin, TraceTensor::sin_tensor)
    }

    pub fn try_cos(&self) -> Result<Self, String> {
        self.unary(EagerKernel::Cos, TraceTensor::cos_tensor)
    }

    pub fn try_unary_math(&self, kind: UnaryMathKind) -> Result<Self, String> {
        self.unary(EagerKernel::UnaryMath(kind), |tensor| {
            tensor.unary_math_tensor(kind)
        })
    }

    fn unary_math(&self, kind: UnaryMathKind) -> PyResult<Self> {
        self.try_unary_math(kind).map_err(PyValueError::new_err)
    }

    /// Elementwise `atan2(self, x)` with broadcasting and the promotion of
    /// the other binary ops.
    pub fn try_atan2(&self, x: &Self) -> Result<Self, String> {
        self.binary_op(x, None, TracedBinary::Arithmetic("atan2"))
    }

    /// `atan2(self, x)` for a Python number `x`, a weak scalar.
    pub fn try_atan2_scalar(&self, x: f64) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.float_scalar_binary(x, "atan2"))
    }

    /// Elementwise C `fmod(self, y)` (the sign of `self`) with broadcasting
    /// and the promotion of the other binary ops.
    pub fn try_fmod(&self, y: &Self) -> Result<Self, String> {
        self.binary_op(y, None, TracedBinary::Arithmetic("fmod"))
    }

    /// `fmod(self, y)` for a Python number `y`, a weak scalar.
    pub fn try_fmod_scalar(&self, y: f64) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.float_scalar_binary(y, "fmod"))
    }

    /// Inclusive prefix sums along `axis` (over the flattened tensor when
    /// `None`, like NumPy), from the last entry when `reverse`.
    pub fn try_cumsum(&self, axis: Option<isize>, reverse: bool) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.cumsum_tensor(axis, reverse))
    }

    pub fn try_relu(&self) -> Result<Self, String> {
        composite::relu(self)
    }

    pub fn try_abs(&self) -> Result<Self, String> {
        composite::abs(self)
    }

    pub fn try_sigmoid(&self) -> Result<Self, String> {
        composite::sigmoid(self)
    }

    pub fn try_softplus(&self) -> Result<Self, String> {
        composite::softplus(self)
    }

    pub fn try_triangular(&self, lower: bool) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.triangular_tensor(lower))
    }

    pub fn try_powi(&self, exponent: u32) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.powi_tensor(exponent))
    }

    /// `self ** exponent` for a Python number other than a non-negative
    /// int, through the elementwise `pow` op with a weak scalar.
    pub fn try_powf(&self, exponent: f64) -> Result<Self, String> {
        self.scalar_arithmetic(exponent, "pow", false)
    }

    /// Elementwise `self ** exponent` of two tensors with broadcasting and
    /// the promotion of the other binary ops.
    pub fn try_pow(&self, exponent: &Self) -> Result<Self, String> {
        self.binary_op(exponent, None, TracedBinary::Arithmetic("pow"))
    }

    /// `base ** self` for a Python number `base`, a weak scalar.
    pub fn try_scalar_pow(&self, base: f64) -> Result<Self, String> {
        self.scalar_arithmetic(base, "pow", true)
    }

    pub fn try_concat(tensors: &[PyTensor], axis: usize) -> Result<Self, String> {
        if tensors.is_empty() {
            return Err("concat requires at least one tensor".to_string());
        }
        TraceTensor::evaluate_eager_all(&tensors.iter().collect::<Vec<_>>(), |tensors| {
            TraceTensor::try_concat(&tensors, axis)
        })
    }

    pub fn try_stack(tensors: &[PyTensor], axis: isize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "stack requires at least one tensor".to_string())?;
        if tensors.iter().any(|tensor| tensor.shape != first.shape) {
            return Err("stack requires tensors with identical shapes".to_string());
        }
        TraceTensor::evaluate_eager_all(&tensors.iter().collect::<Vec<_>>(), |tensors| {
            TraceTensor::try_stack(&tensors, axis)
        })
    }

    pub fn try_slice(
        &self,
        axis: usize,
        start: usize,
        length: usize,
        step: usize,
    ) -> Result<PyTensorView, String> {
        let strides = contiguous_strides(&self.shape);
        let (shape, strides, offset) =
            sliced_layout(&self.shape, &strides, 0, axis, start, length, step)?;

        Ok(PyTensorView {
            data: self.data.clone(),
            shape,
            strides,
            offset,
            dtype: self.dtype,
        })
    }

    /// The integer and contiguous-slice indices of a `__getitem__` key: the
    /// values the traced `Slice` and `Reshape` nodes select, copied once
    /// after the slice layouts are composed. The weak type is kept, as by
    /// those nodes.
    pub fn try_index(&self, indices: &[TensorIndex]) -> Result<Self, String> {
        if indices.is_empty() {
            return Ok(self.clone());
        }
        // Compose slice layouts before copying, so intermediate axes do not
        // materialize arrays that the next index immediately discards.
        let mut output = PyTensorView {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
            dtype: self.dtype,
        };
        let mut axis = 0;
        for index in indices {
            match *index {
                TensorIndex::Slice { start, stop } => {
                    output = output.try_slice(axis, start, stop - start, 1)?;
                    axis += 1;
                }
                TensorIndex::Integer(index) => {
                    output = output.try_slice(axis, index, 1, 1)?;
                    output.shape.remove(axis);
                    output.strides.remove(axis);
                }
            }
        }
        Ok(output.materialize()?.with_weak(self.weak))
    }

    pub fn try_gather(&self, indices: &[usize], axis: isize) -> Result<Self, String> {
        Self::evaluated([self], |[tensor]| tensor.gather_tensor(indices, axis))
    }

    pub fn try_scatter_add(
        &self,
        indices: &[usize],
        updates: &Self,
        axis: isize,
    ) -> Result<Self, String> {
        Self::evaluated([self, updates], |[tensor, updates]| {
            tensor.scatter_add_tensor(indices, &updates, axis)
        })
    }
}

/// The kernel of a `TraceTensor::binary` op that has a direct path.
fn arithmetic_kernel(op: &str) -> Option<EagerKernel> {
    Some(match op {
        "add" => EagerKernel::Add,
        "sub" => EagerKernel::Sub,
        "mul" => EagerKernel::Mul,
        "div" => EagerKernel::Div,
        "greater" => EagerKernel::Greater,
        _ => return None,
    })
}

/// The primitives of the composites, each evaluated at once.
impl Primitives for PyTensor {
    fn dtype(&self) -> Result<TensorDType, String> {
        Ok(self.dtype)
    }

    fn binary(&self, rhs: &Self, op: &'static str) -> Result<Self, String> {
        self.binary_op(rhs, arithmetic_kernel(op), TracedBinary::Arithmetic(op))
    }

    fn scalar_binary(&self, value: f64, op: &'static str) -> Result<Self, String> {
        self.scalar_arithmetic(value, op, false)
    }

    fn scalar_left_binary(&self, value: f64, op: &'static str) -> Result<Self, String> {
        self.scalar_arithmetic(value, op, true)
    }

    fn compare(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String> {
        self.try_compare(rhs, kind)
    }

    fn compare_scalar(&self, value: f64, kind: TensorComparison) -> Result<Self, String> {
        self.try_compare_scalar(value, kind)
    }

    fn isnan(&self) -> Result<Self, String> {
        self.try_classify(true)
    }

    fn logical_or(&self, rhs: &Self) -> Result<Self, String> {
        self.try_logical(rhs, false)
    }

    fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        Self::try_where(self, on_true, on_false)
    }

    fn scalar(&self, value: f64) -> Result<Self, String> {
        Ok(Self::weak_scalar(value))
    }

    fn exp(&self) -> Result<Self, String> {
        self.try_exp()
    }

    fn log1p(&self) -> Result<Self, String> {
        self.try_log1p()
    }
}

#[pymethods]
impl PyTensor {
    /// `dtype` defaults to `quabla.float64`; `quabla.float32` rounds `data`.
    #[new]
    #[pyo3(signature = (shape, data, dtype = None))]
    fn py_new(shape: Vec<usize>, data: Vec<f64>, dtype: Option<PyDType>) -> PyResult<Self> {
        let dtype = dtype.map_or(TensorDType::F64, |dtype| dtype.dtype);
        Self::from_shape_data_typed(shape, data, dtype).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let size = element_count(&shape).map_err(PyValueError::new_err)?;

        Ok(Self {
            shape,
            data: HostTensorStorage::from_f64(vec![0.0; size], TensorDType::F64),
            dtype: TensorDType::F64,
            weak: false,
        })
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> PyResult<Self> {
        Self::full(shape, 1.0)
    }

    #[staticmethod]
    fn full(shape: Vec<usize>, value: f64) -> PyResult<Self> {
        let size = element_count(&shape).map_err(PyValueError::new_err)?;
        Ok(Self {
            shape,
            data: HostTensorStorage::from_f64(vec![value; size], TensorDType::F64),
            dtype: TensorDType::F64,
            weak: false,
        })
    }

    #[staticmethod]
    fn split_key(key: u64, count: usize) -> PyResult<Vec<u64>> {
        if count == 0 {
            return Err(PyValueError::new_err("split_key count must be positive"));
        }
        let mut state = key;
        Ok((0..count).map(|_| next_random_key(&mut state)).collect())
    }

    #[staticmethod]
    #[pyo3(signature = (shape, key, mean = 0.0, stddev = 1.0))]
    fn random_normal(shape: Vec<usize>, key: u64, mean: f64, stddev: f64) -> PyResult<Self> {
        if !(mean.is_finite() && stddev.is_finite() && stddev >= 0.0) {
            return Err(PyValueError::new_err(
                "random_normal mean must be finite and stddev must be finite and non-negative",
            ));
        }
        let count = element_count(&shape).map_err(PyValueError::new_err)?;
        let mut state = key;
        let data = (0..count)
            .map(|_| mean + stddev * standard_normal(&mut state))
            .collect();
        Self::from_shape_data(shape, data).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn glorot_normal(shape: Vec<usize>, key: u64) -> PyResult<Self> {
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "glorot_normal requires a rank-2 shape, got {:?}",
                shape
            )));
        }
        let fan_sum = shape[0]
            .checked_add(shape[1])
            .ok_or_else(|| PyValueError::new_err("glorot_normal fan sum overflows usize"))?;
        let stddev = (2.0 / fan_sum as f64).sqrt();
        Self::random_normal(shape, key, 0.0, stddev)
    }

    /// Uniform samples of `dtype` (default `float64`) in `[minval, maxval)`
    /// from the SplitMix64 stream seeded by `key`, one stream output per
    /// element: its top 53 bits give `k * 2^-53` in `[0, 1)`, scaled to the
    /// interval. Rounding the scaled value, or narrowing it to `float32`,
    /// can land on `maxval` (or below `minval`), so each sample is clamped
    /// to the representable values of `dtype` inside the interval.
    #[staticmethod]
    #[pyo3(signature = (shape, key, minval = 0.0, maxval = 1.0, dtype = None))]
    fn random_uniform(
        shape: Vec<usize>,
        key: u64,
        minval: f64,
        maxval: f64,
        dtype: Option<PyDType>,
    ) -> PyResult<Self> {
        let dtype = dtype.map_or(TensorDType::F64, |dtype| dtype.dtype);
        let width = maxval - minval;
        if !(minval < maxval && width.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "random_uniform requires finite minval < maxval, got [{minval}, {maxval})"
            )));
        }
        let (low, high) = match dtype {
            TensorDType::F64 => (minval, maxval.next_down()),
            TensorDType::F32 => {
                let mut low = minval as f32;
                if f64::from(low) < minval {
                    low = low.next_up();
                }
                let mut high = maxval as f32;
                if f64::from(high) >= maxval {
                    high = high.next_down();
                }
                if low > high {
                    return Err(PyValueError::new_err(format!(
                        "no float32 value lies in [{minval}, {maxval})"
                    )));
                }
                (f64::from(low), f64::from(high))
            }
            TensorDType::Bool => {
                return Err(PyValueError::new_err(
                    "random_uniform requires a floating-point dtype",
                ))
            }
        };
        let count = element_count(&shape).map_err(PyValueError::new_err)?;
        let mut state = key;
        let data = (0..count)
            .map(|_| {
                let unit = (next_random_key(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                let value = minval + width * unit;
                let value = if dtype == TensorDType::F32 {
                    f64::from(value as f32)
                } else {
                    value
                };
                value.clamp(low, high)
            })
            .collect();
        Self::from_shape_data_typed(shape, data, dtype).map_err(PyValueError::new_err)
    }

    /// A key derived from `key` and the integer `data` by SplitMix64 mixing,
    /// distinct from the keys `split_key(key, count)` returns.
    #[staticmethod]
    fn fold_in_key(key: u64, data: u64) -> u64 {
        let mut data_state = data;
        let mut state = key ^ next_random_key(&mut data_state);
        next_random_key(&mut state)
    }

    #[staticmethod]
    #[pyo3(signature = (start, stop, step = 1.0))]
    fn arange(start: f64, stop: f64, step: f64) -> PyResult<Self> {
        if !(start.is_finite() && stop.is_finite() && step.is_finite()) {
            return Err(PyValueError::new_err(
                "arange start, stop, and step must be finite",
            ));
        }
        if step == 0.0 {
            return Err(PyValueError::new_err("arange step must not be zero"));
        }
        if (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Err(PyValueError::new_err(
                "arange start, stop, and step do not define a non-empty range",
            ));
        }
        let mut data = Vec::new();
        let mut value = start;
        if step > 0.0 {
            while value < stop {
                data.push(value);
                value += step;
            }
        } else {
            while value > stop {
                data.push(value);
                value += step;
            }
        }
        Self::from_shape_data(vec![data.len()], data).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn linspace(start: f64, stop: f64, num: usize) -> PyResult<Self> {
        if !(start.is_finite() && stop.is_finite()) {
            return Err(PyValueError::new_err(
                "linspace start and stop must be finite",
            ));
        }
        if num == 0 {
            return Err(PyValueError::new_err("linspace num must be positive"));
        }
        if num == 1 {
            return Self::from_shape_data(vec![1], vec![start]).map_err(PyValueError::new_err);
        }
        let denominator = (num - 1) as f64;
        let data = (0..num)
            .map(|index| start + (stop - start) * index as f64 / denominator)
            .collect();
        Self::from_shape_data(vec![num], data).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    #[pyo3(signature = (rows, cols = None))]
    fn eye(rows: usize, cols: Option<usize>) -> PyResult<Self> {
        if rows == 0 {
            return Err(PyValueError::new_err("eye rows must be positive"));
        }
        let cols = cols.unwrap_or(rows);
        if cols == 0 {
            return Err(PyValueError::new_err("eye cols must be positive"));
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| PyValueError::new_err("eye element count overflows usize"))?;
        let mut data = vec![0.0; count];
        for diagonal in 0..rows.min(cols) {
            data[diagonal * cols + diagonal] = 1.0;
        }
        Self::from_shape_data(vec![rows, cols], data).map_err(PyValueError::new_err)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter(dtype)]
    fn py_dtype(&self) -> PyDType {
        self.dtype.into()
    }

    /// Converts to `dtype` with round-to-nearest-even (exact for widening).
    fn astype(&self, dtype: PyDType) -> Self {
        self.clone().typed(dtype.dtype)
    }

    /// Pure host SGD leaf update with one final rounding and owned output.
    fn _sgd_update(&self, gradient: &Self, rate: f64) -> PyResult<Self> {
        self.ensure_not_bool("sgd").map_err(PyValueError::new_err)?;
        gradient
            .ensure_not_bool("sgd")
            .map_err(PyValueError::new_err)?;
        if self.shape != gradient.shape {
            return Err(PyValueError::new_err(
                "gradient shape does not match parameter",
            ));
        }
        // Keep F64 multiplication/subtraction and round only the final parameter.
        let values = self
            .data
            .iter()
            .zip(gradient.data.iter())
            .map(|(parameter, gradient)| parameter - gradient * rate)
            .collect();
        Self::from_shape_data_typed(self.shape.clone(), values, self.dtype)
            .map_err(PyValueError::new_err)
    }

    /// Pure host Adam leaf update with independent parameter and f64 moment
    /// buffers. Keep the eager expression's operation order before rounding
    /// the parameter once to its original dtype.
    ///
    /// A nonzero `weight_decay` gives AdamW's decoupled decay (Loshchilov and
    /// Hutter): the step becomes `learning_rate * (delta + weight_decay *
    /// parameter)` with the pre-update parameter, as in optax's `adamw`.
    /// Zero keeps the Adam expression exactly, including for infinite
    /// parameters, where `0 * inf` would otherwise introduce a NaN.
    #[pyo3(signature = (gradient, first, second, hyperparameters, weight_decay = 0.0))]
    fn _adam_update(
        &self,
        gradient: &Self,
        first: &Self,
        second: &Self,
        hyperparameters: (f64, f64, f64, f64, f64, f64),
        weight_decay: f64,
    ) -> PyResult<(Self, Self, Self)> {
        for tensor in [self, gradient, first, second] {
            if tensor.dtype == TensorDType::Bool {
                return Err(PyTypeError::new_err(
                    "Adam update requires floating-point Tensors",
                ));
            }
            if tensor.shape != self.shape {
                return Err(PyValueError::new_err(
                    "Adam update shapes must match parameter shape",
                ));
            }
        }
        let (learning_rate, b1, b2, eps, correction1, correction2) = hyperparameters;
        if correction1 == 0.0 || correction2 == 0.0 {
            return Err(PyValueError::new_err(
                "division by zero scalar is not supported",
            ));
        }
        let mut parameters = Vec::with_capacity(self.data.len());
        let mut moments = Vec::with_capacity(self.data.len());
        let mut variances = Vec::with_capacity(self.data.len());
        for (((parameter, gradient), m), v) in self
            .data
            .iter()
            .zip(gradient.data.iter())
            .zip(first.data.iter())
            .zip(second.data.iter())
        {
            let m = m * b1 + gradient * (1.0 - b1);
            let v = v * b2 + (gradient * gradient) * (1.0 - b2);
            let denominator = (v / correction2).sqrt() + eps;
            if denominator == 0.0 {
                return Err(PyValueError::new_err(
                    "failed to evaluate tensor /: division by zero is not supported",
                ));
            }
            let mut delta = (m / correction1) / denominator;
            if weight_decay != 0.0 {
                delta += weight_decay * parameter;
            }
            parameters.push(self.dtype.round(parameter - delta * learning_rate));
            moments.push(m);
            variances.push(v);
        }
        let result = |data, dtype| Self {
            shape: self.shape.clone(),
            data: HostTensorStorage::from_f64(data, dtype),
            dtype,
            weak: false,
        };
        Ok((
            result(parameters, self.dtype),
            result(moments, TensorDType::F64),
            result(variances, TensorDType::F64),
        ))
    }

    fn to_flat_list(&self) -> Vec<f64> {
        self.data.to_f64().into_owned()
    }

    /// Backend of `quabla.array` for Python scalars and nested lists/tuples
    /// (see `interop::tensor_from_nested`).
    #[staticmethod]
    #[pyo3(name = "_from_nested", signature = (value, dtype = None))]
    fn from_nested(value: &Bound<'_, PyAny>, dtype: Option<PyDType>) -> PyResult<Self> {
        interop::tensor_from_nested(value, dtype.map(|dtype| dtype.dtype))
    }

    /// Backend of `quabla.array` for buffer-protocol objects; `None` when
    /// `value` has no buffer (see `interop::tensor_from_buffer`).
    #[staticmethod]
    #[pyo3(name = "_from_buffer", signature = (value, dtype = None))]
    fn from_buffer(value: &Bound<'_, PyAny>, dtype: Option<PyDType>) -> PyResult<Option<Self>> {
        interop::tensor_from_buffer(value, dtype.map(|dtype| dtype.dtype))
    }

    /// Nested Python lists (a bare scalar for rank 0); `bool` tensors give
    /// Python `bool`s.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        interop::nested_list(py, &self.shape, &self.data.to_f64(), self.dtype)
    }

    /// The single element as a Python `float` (or `bool` for `bool`
    /// tensors); any shape with exactly one element is accepted.
    fn item<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        interop::single_item(py, &self.shape, &self.data.to_f64(), self.dtype)
    }

    fn __float__(&self) -> PyResult<f64> {
        match self.data.to_f64().as_ref() {
            [value] => Ok(*value),
            _ => Err(PyTypeError::new_err(format!(
                "only single-element tensors can be converted to a Python float, got shape {:?}",
                self.shape
            ))),
        }
    }

    /// A read-only NumPy array sharing physical storage for every dtype. Imports NumPy
    /// on first use; `numpy.array(tensor)` gives a writable copy.
    fn numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let numpy = slf.py().import("numpy")?;
        numpy.call_method1("asarray", (PyMemoryView::from(slf.as_any())?,))
    }

    /// NumPy's array protocol, for NumPy 1.x (`dtype` only) and 2.x
    /// (`copy`). Without a dtype change the result is the read-only export
    /// (`copy=True` copies it into a writable array); a dtype change gives
    /// a fresh writable array, and with `copy=False` raises `ValueError` as
    /// NumPy does. Implemented with `asarray` and `astype` only, because
    /// NumPy 1.x's `asarray` has no `copy` keyword.
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let numpy = slf.py().import("numpy")?;
        let export = numpy.call_method1("asarray", (PyMemoryView::from(slf.as_any())?,))?;
        if let Some(dtype) = dtype {
            let dtype = numpy.call_method1("dtype", (dtype,))?;
            if !export.getattr("dtype")?.eq(&dtype)? {
                if copy == Some(false) {
                    return Err(PyValueError::new_err(format!(
                        "cannot convert a {} tensor to {dtype} without a copy (copy=False)",
                        PyDType::from(slf.borrow().dtype).__repr__()
                    )));
                }
                return export.call_method1("astype", (dtype,));
            }
        }
        if copy == Some(true) {
            export.call_method0("copy")
        } else {
            Ok(export)
        }
    }

    /// Read-only buffer export (see `interop::fill_buffer`).
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let export = {
            let tensor = slf.borrow();
            interop::BufferExport::new(&tensor.shape, &tensor.data, tensor.dtype)
        };
        // SAFETY: CPython calls `bf_getbuffer` with a `Py_buffer` it owns
        // for the duration of the export, and `slf` is the exporter, which
        // is what `fill_buffer` requires.
        unsafe { interop::fill_buffer(view, flags, export, slf.into_any()) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        // SAFETY: CPython calls `bf_releasebuffer` once for each view that
        // `__getbuffer__` filled successfully, which is what
        // `release_buffer` requires.
        unsafe { interop::release_buffer(view) }
    }

    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let shape = parse_reshape_args(shape, self.data.len())?;
        self.try_reshape(shape).map_err(PyValueError::new_err)
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.try_broadcast_to(shape).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axes = None))]
    fn transpose(&self, axes: Option<Vec<isize>>) -> PyResult<Self> {
        self.try_transpose(axes).map_err(PyValueError::new_err)
    }

    /// The tensor with its axes reversed, as `transpose()`.
    #[getter(T)]
    fn reversed_axes(&self) -> PyResult<Self> {
        self.try_transpose(None).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn sum(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_sum_axes(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn mean(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_mean_axes(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn norm(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_norm(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn prod(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_prod_axes(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn max(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_max_axes(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn min(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_min_axes(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (indices, axis = 0))]
    fn gather(&self, indices: &Bound<'_, PyAny>, axis: isize) -> PyResult<Self> {
        let axis = normalize_axis(axis, self.shape.len()).map_err(PyValueError::new_err)?;
        let indices = parse_axis_indices(indices, self.shape[axis])?;
        self.try_gather(&indices, axis as isize)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (indices, updates, axis = 0))]
    fn scatter_add(
        &self,
        indices: &Bound<'_, PyAny>,
        updates: &Self,
        axis: isize,
    ) -> PyResult<Self> {
        let axis = normalize_axis(axis, self.shape.len()).map_err(PyValueError::new_err)?;
        let indices = parse_axis_indices(indices, self.shape[axis])?;
        self.try_scatter_add(&indices, updates, axis as isize)
            .map_err(PyValueError::new_err)
    }

    fn tanh(&self) -> PyResult<Self> {
        self.try_tanh().map_err(PyValueError::new_err)
    }

    fn exp(&self) -> PyResult<Self> {
        self.try_exp().map_err(PyValueError::new_err)
    }

    fn log(&self) -> PyResult<Self> {
        self.try_log().map_err(PyValueError::new_err)
    }

    fn log1p(&self) -> PyResult<Self> {
        self.try_log1p().map_err(PyValueError::new_err)
    }

    fn expm1(&self) -> PyResult<Self> {
        self.try_expm1().map_err(PyValueError::new_err)
    }

    fn erf(&self) -> PyResult<Self> {
        self.try_erf().map_err(PyValueError::new_err)
    }

    fn erfc(&self) -> PyResult<Self> {
        self.try_erfc().map_err(PyValueError::new_err)
    }

    /// `atan2(self, x)` of a tensor or Python number `x`.
    fn atan2(&self, x: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, x, TracedBinary::Arithmetic("atan2"), true) {
            return result;
        }
        if let Ok(x) = x.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_atan2(&x));
        }
        if let Ok(x) = x.extract::<f64>() {
            return eager(self.try_atan2_scalar(x));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    /// C `fmod(self, y)` of a tensor or Python number `y`: the remainder of
    /// the quotient truncated toward zero, with the sign of `self`.
    fn fmod(&self, y: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, y, TracedBinary::Arithmetic("fmod"), true) {
            return result;
        }
        if let Ok(y) = y.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_fmod(&y));
        }
        if let Ok(y) = y.extract::<f64>() {
            return eager(self.try_fmod_scalar(y));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn tan(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Tan)
    }

    fn arcsin(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arcsin)
    }

    fn arccos(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arccos)
    }

    fn arctan(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arctan)
    }

    fn sinh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Sinh)
    }

    fn cosh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Cosh)
    }

    fn arcsinh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arcsinh)
    }

    fn arccosh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arccosh)
    }

    fn arctanh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arctanh)
    }

    fn log2(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Log2)
    }

    fn log10(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Log10)
    }

    fn cbrt(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Cbrt)
    }

    fn floor(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Floor)
    }

    fn ceil(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Ceil)
    }

    fn round(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Round)
    }

    /// The value itself: outside a transform nothing is differentiated.
    /// Traced code records a `stop_gradient` node with a zero derivative.
    fn stop_gradient(&self) -> Self {
        self.clone()
    }

    #[pyo3(signature = (axis = None, reverse = false))]
    fn cumsum(&self, axis: Option<isize>, reverse: bool) -> PyResult<Self> {
        self.try_cumsum(axis, reverse)
            .map_err(PyValueError::new_err)
    }

    fn sqrt(&self) -> PyResult<Self> {
        self.try_sqrt().map_err(PyValueError::new_err)
    }

    fn relu(&self) -> PyResult<Self> {
        self.try_relu().map_err(PyValueError::new_err)
    }

    fn abs(&self) -> PyResult<Self> {
        self.try_abs().map_err(PyValueError::new_err)
    }

    fn sigmoid(&self) -> PyResult<Self> {
        self.try_sigmoid().map_err(PyValueError::new_err)
    }

    fn softplus(&self) -> PyResult<Self> {
        self.try_softplus().map_err(PyValueError::new_err)
    }

    fn tril(&self) -> PyResult<Self> {
        self.try_triangular(true).map_err(PyValueError::new_err)
    }

    fn triu(&self) -> PyResult<Self> {
        self.try_triangular(false).map_err(PyValueError::new_err)
    }

    fn sin(&self) -> PyResult<Self> {
        self.try_sin().map_err(PyValueError::new_err)
    }

    fn cos(&self) -> PyResult<Self> {
        self.try_cos().map_err(PyValueError::new_err)
    }

    fn powi(&self, exponent: u32) -> PyResult<Self> {
        self.try_powi(exponent).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis, start, length, step = 1))]
    fn slice(
        &self,
        axis: usize,
        start: usize,
        length: usize,
        step: usize,
    ) -> PyResult<PyTensorView> {
        self.try_slice(axis, start, length, step)
            .map_err(PyValueError::new_err)
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let plan = parse_index_plan(index, &self.shape)?;
        let mut output = self
            .try_index(&plan.indices)
            .map_err(PyValueError::new_err)?;
        for (axis, positions) in &plan.gathers {
            output = output
                .try_gather(positions, *axis as isize)
                .map_err(PyValueError::new_err)?;
        }
        match plan.shape {
            Some(shape) => output.try_reshape(shape).map_err(PyValueError::new_err),
            None => Ok(output),
        }
    }

    // With a tracer operand, arithmetic, comparisons, `maximum`/`minimum`,
    // `**`, `@`, `solve`, and the logical ops are traced, with this tensor
    // captured as a constant; the tracer check comes first because a tracer
    // refuses the `float()` conversion that the scalar branch attempts.
    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Arithmetic("add"), true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_add(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_add_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.__add__(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, lhs, TracedBinary::Arithmetic("add"), false) {
            return result;
        }
        self.__add__(lhs)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Arithmetic("sub"), true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_sub(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_sub_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.__sub__(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, lhs, TracedBinary::Arithmetic("sub"), false) {
            return result;
        }
        if let Ok(lhs) = lhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(lhs.try_sub(self));
        }
        if let Ok(lhs) = lhs.extract::<f64>() {
            return eager(self.try_scalar_sub(lhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Arithmetic("mul"), true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_mul(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_mul_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.__mul__(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, lhs, TracedBinary::Arithmetic("mul"), false) {
            return result;
        }
        self.__mul__(lhs)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Arithmetic("div"), true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_div(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_div_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar divisor",
        ))
    }

    fn div(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.__truediv__(rhs)
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, lhs, TracedBinary::Arithmetic("div"), false) {
            return result;
        }
        if let Ok(lhs) = lhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(lhs.try_div(self));
        }
        if let Ok(lhs) = lhs.extract::<f64>() {
            return eager(self.try_scalar_div(lhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar dividend",
        ))
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.try_negative().map_err(PyValueError::new_err)
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<EagerOrTraced> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for Tensor power",
            ));
        }
        if let Some(result) = traced(self, exponent, TracedBinary::Arithmetic("pow"), true) {
            return result;
        }
        if let Ok(exponent) = exponent.extract::<u32>() {
            return eager(self.try_powi(exponent));
        }
        // A tensor exponent, including a single-element one, is an elementwise
        // operand that broadcasts, as in NumPy and in traced code; it is never
        // unwrapped into a scalar.
        if let Ok(exponent) = exponent.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_pow(&exponent));
        }
        let exponent = extract_scalar(exponent)
            .ok_or_else(|| PyTypeError::new_err("expected a Tensor or numeric scalar exponent"))?;
        eager(self.try_powf(exponent))
    }

    /// `c ** x` for a Python number `c`.
    fn __rpow__(
        &self,
        base: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for Tensor power",
            ));
        }
        let base = extract_scalar(base)
            .ok_or_else(|| PyTypeError::new_err("expected a numeric scalar base"))?;
        self.try_scalar_pow(base).map_err(PyValueError::new_err)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Arithmetic("greater"), true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_gt(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_gt_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    /// Elementwise `self > rhs` as a `bool` tensor (unlike `gt`, which keeps
    /// returning a 0/1 float mask).
    fn greater(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.compare_operand(rhs, TensorComparison::Greater)
    }

    fn greater_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.compare_operand(rhs, TensorComparison::GreaterEqual)
    }

    fn less(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.compare_operand(rhs, TensorComparison::Less)
    }

    fn less_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.compare_operand(rhs, TensorComparison::LessEqual)
    }

    fn equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.compare_operand(rhs, TensorComparison::Equal)
    }

    fn not_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.compare_operand(rhs, TensorComparison::NotEqual)
    }

    // `==`/`!=` are deliberately not overloaded: tensors keep identity
    // equality and stay hashable; use `equal`/`not_equal` for elementwise.
    fn __gt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.greater(rhs)
    }

    fn __ge__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.greater_equal(rhs)
    }

    fn __lt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.less(rhs)
    }

    fn __le__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.less_equal(rhs)
    }

    /// Defining ordering operators installs a rich comparison, which drops
    /// Python's default identity hash; restore it so tensors stay usable as
    /// dictionary keys.
    fn __hash__(slf: &Bound<'_, Self>) -> isize {
        slf.as_ptr() as isize
    }

    fn logical_and(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.logical_operand(rhs, true)
    }

    fn logical_or(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        self.logical_operand(rhs, false)
    }

    fn logical_not(&self) -> PyResult<Self> {
        self.try_logical_not().map_err(PyValueError::new_err)
    }

    fn __and__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if !is_array_operand(rhs) {
            return Ok(EagerOrTraced::NotImplemented(rhs.py().NotImplemented()));
        }
        self.logical_and(rhs)
    }

    fn __or__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if !is_array_operand(rhs) {
            return Ok(EagerOrTraced::NotImplemented(rhs.py().NotImplemented()));
        }
        self.logical_or(rhs)
    }

    fn __invert__(&self) -> PyResult<Self> {
        self.logical_not()
    }

    fn isnan(&self) -> PyResult<Self> {
        self.try_classify(true).map_err(PyValueError::new_err)
    }

    fn isfinite(&self) -> PyResult<Self> {
        self.try_classify(false).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn any(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_any_all(extract_reduction_axes(axis)?, keepdims, true)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn all(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.try_any_all(extract_reduction_axes(axis)?, keepdims, false)
            .map_err(PyValueError::new_err)
    }

    fn __bool__(&self) -> PyResult<bool> {
        self.truthiness().map_err(PyValueError::new_err)
    }

    fn maximum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Maximum, true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_maximum(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_maximum_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn minimum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Minimum, true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_minimum(&rhs));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_minimum_scalar(rhs));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn __matmul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if !is_array_operand(rhs) {
            return Ok(EagerOrTraced::NotImplemented(rhs.py().NotImplemented()));
        }
        self.matmul(rhs)
    }

    fn matmul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Matmul, true) {
            return result;
        }
        eager(self.try_matmul(&self.tensor_operand(rhs, "matmul")?))
    }

    fn solve(&self, rhs: &Bound<'_, PyAny>) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Solve, true) {
            return result;
        }
        eager(self.try_solve(&self.tensor_operand(rhs, "solve")?))
    }

    #[pyo3(signature = (rhs, lower = true, transpose = false))]
    fn solve_triangular(
        &self,
        rhs: &Bound<'_, PyAny>,
        lower: bool,
        transpose: bool,
    ) -> PyResult<EagerOrTraced> {
        let op = TracedBinary::SolveTriangular { lower, transpose };
        if let Some(result) = traced(self, rhs, op, true) {
            return result;
        }
        eager(self.try_solve_triangular(
            &self.tensor_operand(rhs, "solve_triangular")?,
            lower,
            transpose,
        ))
    }

    fn cholesky(&self) -> PyResult<Self> {
        self.try_cholesky().map_err(PyValueError::new_err)
    }

    /// One output of a dense decomposition of each matrix (`kind` is a
    /// `quabla_core` `LinalgKind` name); the backing op of `quabla.linalg`.
    fn _linalg(&self, kind: &str) -> PyResult<Self> {
        self.try_linalg(kind).map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        crate::repr::tensor_repr(&self.shape, self.dtype, |index| self.data.get(index))
    }
}

impl PyTensor {
    fn compare_operand(
        &self,
        rhs: &Bound<'_, PyAny>,
        kind: TensorComparison,
    ) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Compare(kind), true) {
            return result;
        }
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return eager(self.try_compare(&rhs, kind));
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return eager(self.try_compare_scalar(rhs, kind));
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn logical_operand(&self, rhs: &Bound<'_, PyAny>, and: bool) -> PyResult<EagerOrTraced> {
        if let Some(result) = traced(self, rhs, TracedBinary::Logical { and }, true) {
            return result;
        }
        let name = if and { "logical_and" } else { "logical_or" };
        eager(self.try_logical(&self.tensor_operand(rhs, name)?, and))
    }

    /// The eager `Tensor` operand of a method without a scalar form; the
    /// `TypeError` matches the one argument extraction used to raise.
    fn tensor_operand(&self, operand: &Bound<'_, PyAny>, op: &str) -> PyResult<PyTensor> {
        operand
            .extract::<PyRef<'_, PyTensor>>()
            .map(|tensor| tensor.clone())
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "{op} expects a Tensor operand, got {}",
                    operand
                        .get_type()
                        .name()
                        .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string())
                ))
            })
    }
}

impl PyTensorView {
    fn try_slice(
        &self,
        axis: usize,
        start: usize,
        length: usize,
        step: usize,
    ) -> Result<Self, String> {
        let (shape, strides, offset) = sliced_layout(
            &self.shape,
            &self.strides,
            self.offset,
            axis,
            start,
            length,
            step,
        )?;

        Ok(Self {
            data: self.data.clone(),
            shape,
            strides,
            offset,
            dtype: self.dtype,
        })
    }

    fn to_flat_vec(&self) -> Result<Vec<f64>, String> {
        let count = element_count(&self.shape)?;
        Ok((0..count)
            .map(|index| {
                self.data
                    .get(view_offset(index, &self.shape, &self.strides, self.offset))
            })
            .collect())
    }

    pub(crate) fn materialize(&self) -> Result<PyTensor, String> {
        PyTensor::from_shape_data_typed(self.shape.clone(), self.to_flat_vec()?, self.dtype)
    }
}

#[pymethods]
impl PyTensorView {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    #[getter]
    fn offset(&self) -> usize {
        self.offset
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn to_flat_list(&self) -> PyResult<Vec<f64>> {
        self.to_flat_vec().map_err(PyValueError::new_err)
    }

    fn to_tensor(&self) -> PyResult<PyTensor> {
        self.materialize().map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis, start, length, step = 1))]
    fn slice(&self, axis: usize, start: usize, length: usize, step: usize) -> PyResult<Self> {
        self.try_slice(axis, start, length, step)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorView(shape={:?}, strides={:?})",
            self.shape, self.strides
        )
    }
}

#[cfg(test)]
fn storage_address(storage: &HostTensorStorage) -> *const () {
    match storage {
        HostTensorStorage::F64(values) => values.as_ptr().cast(),
        HostTensorStorage::F32(values) => values.as_ptr().cast(),
        HostTensorStorage::Bool(values) => values.as_ptr().cast(),
    }
}

#[cfg(test)]
fn storage_ptr_eq(lhs: &HostTensorStorage, rhs: &HostTensorStorage) -> bool {
    match (lhs, rhs) {
        (HostTensorStorage::F64(lhs), HostTensorStorage::F64(rhs)) => Arc::ptr_eq(lhs, rhs),
        (HostTensorStorage::F32(lhs), HostTensorStorage::F32(rhs)) => Arc::ptr_eq(lhs, rhs),
        (HostTensorStorage::Bool(lhs), HostTensorStorage::Bool(rhs)) => Arc::ptr_eq(lhs, rhs),
        _ => false,
    }
}

#[cfg(test)]
fn storage_capacity(storage: &HostTensorStorage) -> usize {
    match storage {
        HostTensorStorage::F64(values) => values.capacity(),
        HostTensorStorage::F32(values) => values.capacity(),
        HostTensorStorage::Bool(values) => values.capacity(),
    }
}

#[cfg(test)]
mod indexing_tests {
    use super::*;

    fn previous_index(input: &PyTensor, indices: &[TensorIndex]) -> Result<PyTensor, String> {
        let mut output = input.clone();
        let mut axis = 0;
        for index in indices {
            match *index {
                TensorIndex::Slice { start, stop } => {
                    output = output
                        .try_slice(axis, start, stop - start, 1)?
                        .materialize()?;
                    axis += 1;
                }
                TensorIndex::Integer(index) => {
                    output = output.try_slice(axis, index, 1, 1)?.materialize()?;
                    let mut shape = output.shape.clone();
                    shape.remove(axis);
                    output = output.try_reshape(shape)?;
                }
            }
        }
        Ok(output.with_weak(input.weak))
    }

    fn assert_same(actual: &PyTensor, expected: &PyTensor) {
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.dtype, expected.dtype);
        assert_eq!(actual.weak, expected.weak);
        assert_eq!(actual.data.len(), expected.data.len());
        for (actual, expected) in actual.data.iter().zip(expected.data.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    fn fixture(dtype: TensorDType) -> PyTensor {
        let mut input = PyTensor::from_shape_data_typed(
            vec![2, 3, 4],
            (0..24)
                .map(|i| match i {
                    0 => -0.0,
                    1 => f64::from_bits(0x7ff0_0000_0000_0001),
                    2 => f64::INFINITY,
                    _ => (i as f64 - 7.0) / 13.0,
                })
                .collect(),
            dtype,
        )
        .unwrap();
        input.weak = dtype != TensorDType::Bool;
        input
    }

    #[test]
    fn gather_preserves_order_duplicates_rounding_and_errors() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let input = fixture(dtype);
            let original = input.clone();
            for axis in 0..3 {
                let indices = [input.shape[axis] - 1, 0, input.shape[axis] - 1];
                let slices = indices
                    .iter()
                    .map(|&index| {
                        let slice = input.try_slice(axis, index, 1, 1)?.materialize()?;
                        Ok(slice.with_weak(input.weak))
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .unwrap();
                let expected = PyTensor::try_concat(&slices, axis).unwrap();
                for normalized in [axis as isize, axis as isize - 3] {
                    let actual = input.try_gather(&indices, normalized).unwrap();
                    assert_same(&actual, &expected);
                    assert!(!storage_ptr_eq(&actual.data, &input.data));
                }
            }
            assert_same(&input, &original);
            assert_eq!(
                input.try_gather(&[], 0).unwrap_err(),
                "gather indices must not be empty"
            );
            assert_eq!(
                input.try_gather(&[3], 1).unwrap_err(),
                "gather index is out of bounds for axis 1 with extent 3"
            );
            assert!(input.try_gather(&[0], 3).is_err());
        }
        assert!(PyTensor::weak_scalar(2.0).try_gather(&[0], 0).is_err());
    }

    #[test]
    fn composed_index_preserves_dropped_axes_scalar_results_and_empty_index() {
        use TensorIndex::{Integer as I, Slice as S};
        let selections = [
            vec![],
            vec![S { start: 1, stop: 2 }],
            vec![S { start: 0, stop: 2 }, I(1), S { start: 1, stop: 4 }],
            vec![I(1), S { start: 0, stop: 3 }, I(2)],
            vec![I(1), I(2), I(3)],
            vec![
                S { start: 0, stop: 2 },
                S { start: 0, stop: 3 },
                S { start: 0, stop: 4 },
            ],
        ];
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let input = fixture(dtype);
            for indices in &selections {
                let actual = input.try_index(indices).unwrap();
                assert_same(&actual, &previous_index(&input, indices).unwrap());
                assert_eq!(
                    storage_ptr_eq(&actual.data, &input.data),
                    indices.is_empty()
                );
            }
            for indices in [
                vec![I(2)],
                vec![S { start: 1, stop: 1 }],
                vec![I(0), I(0), I(0), I(0)],
            ] {
                assert_eq!(
                    input.try_index(&indices).unwrap_err(),
                    previous_index(&input, &indices).unwrap_err()
                );
            }
        }
        let scalar = PyTensor::weak_scalar(-0.0);
        assert_same(&scalar.try_index(&[]).unwrap(), &scalar);
    }
}

#[cfg(test)]
mod adam_tests {
    use super::*;

    #[test]
    fn pure_adam_allocates_three_independent_buffers_and_preserves_aliases() {
        let input = PyTensor::from_shape_data(vec![2], vec![1.0, -2.0]).unwrap();
        let (parameters, moments, variances) = input
            ._adam_update(
                &input,
                &input,
                &input,
                (0.1, 0.5, 0.75, 0.01, 0.5, 0.25),
                0.0,
            )
            .unwrap();
        for output in [&parameters, &moments, &variances] {
            assert!(!storage_ptr_eq(&input.data, &output.data));
            assert_eq!(storage_capacity(&output.data), input.data.len());
        }
        assert!(!storage_ptr_eq(&parameters.data, &moments.data));
        assert!(!storage_ptr_eq(&parameters.data, &variances.data));
        assert!(!storage_ptr_eq(&moments.data, &variances.data));
        assert_eq!(input.data.to_f64().as_ref(), &[1.0, -2.0]);
        assert_eq!(moments.data.to_f64().as_ref(), &[1.0, -2.0]);
        assert_eq!(variances.data.to_f64().as_ref(), &[1.0, -0.5]);
        assert!(parameters.data.get(1).is_nan());
    }

    #[test]
    fn same_shape_binary_preserves_rounding_and_broadcast_fallback() {
        let shape = vec![1, 1, 2, 1, 1, 1];
        let lhs = PyTensor::from_shape_data_typed(shape.clone(), vec![1.0, 2.0], TensorDType::F32)
            .unwrap();
        let rhs = PyTensor::from_shape_data_typed(shape.clone(), vec![0.1, 0.1], TensorDType::F32)
            .unwrap();
        let zipped = lhs.try_add(&rhs).unwrap();
        let scalar = PyTensor::from_shape_data_typed(vec![], vec![0.1], TensorDType::F32).unwrap();
        let broadcast = lhs.try_add(&scalar).unwrap();
        assert_eq!(zipped.shape, shape);
        assert_eq!(zipped.data, broadcast.data);
        assert_eq!(zipped.dtype, TensorDType::F32);
    }
}

#[cfg(test)]
mod owned_output_tests {
    use super::*;

    #[test]
    fn native_storage_is_dtype_sized_and_shape_views_share_it() {
        for (dtype, itemsize) in [
            (TensorDType::F64, 8),
            (TensorDType::F32, 4),
            (TensorDType::Bool, 1),
        ] {
            let tensor =
                PyTensor::from_shape_data_typed(vec![16, 16], vec![0.1; 256], dtype).unwrap();
            assert_eq!(storage_capacity(&tensor.data) * itemsize, 256 * itemsize);
            assert_eq!(tensor.data.dtype(), dtype);
            let reshaped = tensor.try_reshape(vec![256]).unwrap();
            assert!(storage_ptr_eq(&tensor.data, &reshaped.data));
            let dynamic = tensor.to_dynamic_tensor().unwrap();
            assert!(storage_ptr_eq(&tensor.data, dynamic.storage()));
            let output = PyTensor::from_dynamic_tensor(dynamic).unwrap();
            assert!(storage_ptr_eq(&tensor.data, &output.data));
            let cast = tensor.clone().typed(TensorDType::F64);
            assert_eq!(
                cast.data.iter().collect::<Vec<_>>(),
                tensor.data.iter().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn fused_sgd_preserves_f64_intermediate_bits_and_input_aliases() {
        for dtype in [TensorDType::F64, TensorDType::F32] {
            let parameter = PyTensor::from_shape_data_typed(
                vec![6],
                vec![-0.0, 1.0000001, 1e16, f64::INFINITY, f64::NAN, -2.0],
                dtype,
            )
            .unwrap();
            let gradient = PyTensor::from_shape_data_typed(
                vec![6],
                vec![0.0, 0.12345678, 1.0, 1.0, 2.0, -3.0],
                dtype,
            )
            .unwrap();
            let snapshot = parameter.data.iter().map(f64::to_bits).collect::<Vec<_>>();
            let alias = parameter.clone();
            for rate in [0.0, 0.123456789, -1.0] {
                let output = parameter._sgd_update(&gradient, rate).unwrap();
                let expected = parameter
                    .data
                    .iter()
                    .zip(gradient.data.iter())
                    .map(|(parameter, gradient)| dtype.round(parameter - gradient * rate).to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(
                    output.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                    expected
                );
                assert!(!storage_ptr_eq(&output.data, &parameter.data));
                assert!(!output.weak);
            }
            assert_eq!(
                alias.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                snapshot
            );
        }
    }

    #[test]
    fn conversion_preserves_storage_address_bits_dtype_and_strength() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let tensor = quabla_core::tensor_ir::DynamicTensor::with_dtype(
                vec![2, 2],
                vec![-0.0, f64::from_bits(0x7ff0_0000_0000_0001), 0.1, -2.0],
                dtype,
            )
            .unwrap();
            let address = storage_address(tensor.storage());
            let bits = tensor
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            let output = PyTensor::from_dynamic_tensor(tensor).unwrap();
            assert_eq!(storage_address(&output.data), address);
            assert_eq!(output.shape, [2, 2]);
            assert_eq!(output.dtype, dtype);
            assert!(!output.weak);
            assert_eq!(
                output.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bits
            );
        }
    }
}

#[cfg(test)]
mod delegation_tests {
    use super::*;

    /// An eager op and the traced op it must equal.
    type Pair = (
        fn(&PyTensor) -> Result<PyTensor, String>,
        fn(&TraceTensor) -> Result<TraceTensor, String>,
    );

    /// Special values: signed zeros, subnormals, infinities, NaNs of both
    /// signs and a payload, rounding and domain edges.
    fn values(count: usize, offset: usize) -> Vec<f64> {
        let special = [
            0.0,
            -0.0,
            1e-310,
            -1e-30,
            0.5,
            -1.5,
            2.5,
            1e30,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfff8_0000_0000_1234),
            0.9999999,
            1.0,
            -88.5,
        ];
        (0..count)
            .map(|index| special[(index * 5 + offset) % special.len()])
            .collect()
    }

    fn tensor(shape: &[usize], dtype: TensorDType, offset: usize) -> PyTensor {
        let count = shape.iter().product();
        PyTensor::from_shape_data_typed(shape.to_vec(), values(count, offset), dtype).unwrap()
    }

    fn assert_same(actual: &PyTensor, expected: &PyTensor, context: &str) {
        assert_eq!(actual.shape, expected.shape, "{context}");
        assert_eq!(actual.dtype, expected.dtype, "{context}");
        assert_eq!(actual.weak, expected.weak, "{context}");
        let bits = |tensor: &PyTensor| tensor.data.iter().map(f64::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(actual), bits(expected), "{context}");
    }

    /// Every op with a direct path gives the value, dtype, and weak type of
    /// its traced graph evaluated by the CPU evaluator, for strong and weak
    /// operands of both float dtypes and for the promotion cases that the
    /// direct path leaves to the graph.
    #[test]
    fn direct_paths_equal_the_traced_graph() {
        let mut operands = Vec::new();
        for dtype in [TensorDType::F32, TensorDType::F64] {
            operands.push((tensor(&[3, 4], dtype, 0), tensor(&[3, 4], dtype, 7)));
            operands.push((tensor(&[3, 4], dtype, 2), tensor(&[], dtype, 11)));
            operands.push((tensor(&[], dtype, 12), tensor(&[3, 4], dtype, 3)));
            operands.push((tensor(&[3, 1], dtype, 1), tensor(&[1, 4], dtype, 5)));
        }
        // A weak `f64` (from `bool * 2.0`) meeting strong and weak operands.
        let mask = tensor(&[3, 4], TensorDType::Bool, 4);
        let weak = mask.try_mul_scalar(2.0).unwrap();
        assert!(weak.weak);
        operands.push((weak.clone(), tensor(&[3, 4], TensorDType::F32, 6)));
        operands.push((weak.clone(), weak.clone()));
        operands.push((weak.clone(), tensor(&[3, 4], TensorDType::F64, 9)));
        operands.push((mask.clone(), tensor(&[3, 4], TensorDType::F32, 6)));
        operands.push((mask.clone(), mask.clone()));
        for (lhs, rhs) in &operands {
            let context = format!("{:?}/{} {:?}/{}", lhs.dtype, lhs.weak, rhs.dtype, rhs.weak);
            for op in ["add", "sub", "mul", "div", "greater"] {
                let graph = PyTensor::evaluated([lhs, rhs], |[x, y]| x.binary(&y, op));
                let direct = Primitives::binary(lhs, rhs, op);
                match (direct, graph) {
                    (Ok(direct), Ok(graph)) => {
                        assert_same(&direct, &graph, &format!("{op} {context}"))
                    }
                    (Err(direct), Err(graph)) => assert_eq!(direct, graph),
                    pair => panic!("{op} {context}: {pair:?}"),
                }
            }
            for kind in [TensorComparison::Less, TensorComparison::NotEqual] {
                let graph = PyTensor::evaluated([lhs, rhs], |[x, y]| x.compare_tensor(&y, kind));
                match (lhs.try_compare(rhs, kind), graph) {
                    (Ok(direct), Ok(graph)) => assert_same(&direct, &graph, &context),
                    (Err(direct), Err(graph)) => assert_eq!(direct, graph),
                    pair => panic!("{kind:?} {context}: {pair:?}"),
                }
            }
            let graph = PyTensor::evaluated([&mask, lhs, rhs], |[m, x, y]| m.where_tensor(&x, &y));
            match (PyTensor::try_where(&mask, lhs, rhs), graph) {
                (Ok(direct), Ok(graph)) => assert_same(&direct, &graph, &context),
                (Err(direct), Err(graph)) => assert_eq!(direct, graph),
                pair => panic!("where {context}: {pair:?}"),
            }
            if lhs.dtype == TensorDType::F32 || lhs.dtype == TensorDType::F64 {
                for (value, scalar_first) in [(0.1, false), (-0.0, true), (f64::NAN, false)] {
                    for op in ["add", "sub", "mul", "div", "greater"] {
                        let graph = PyTensor::evaluated([lhs], |[x]| {
                            if scalar_first {
                                x.scalar_left_binary(value, op)
                            } else {
                                x.scalar_binary(value, op)
                            }
                        })
                        .unwrap();
                        let direct = lhs.scalar_arithmetic(value, op, scalar_first).unwrap();
                        assert_same(&direct, &graph, &format!("{op} {value} {context}"));
                    }
                    let scalar = PyTensor::weak_scalar(value);
                    let graph = PyTensor::evaluated([&mask, lhs, &scalar], |[m, x, y]| {
                        m.where_tensor(&x, &y)
                    })
                    .unwrap();
                    let direct = PyTensor::try_where(&mask, lhs, &scalar).unwrap();
                    assert_same(&direct, &graph, &format!("where scalar {context}"));
                }
            }
        }
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let input = tensor(&[2, 3, 4], dtype, 0);
            let unary: [Pair; 6] = [
                (PyTensor::try_exp, TraceTensor::exp_tensor),
                (PyTensor::try_log1p, TraceTensor::log1p_tensor),
                (PyTensor::try_sqrt, TraceTensor::sqrt_tensor),
                (PyTensor::try_erfc, TraceTensor::erfc_tensor),
                (PyTensor::try_tanh, TraceTensor::tanh_tensor),
                (PyTensor::try_negative, TraceTensor::negative_tensor),
            ];
            for (direct, build) in unary {
                let graph = PyTensor::evaluated([&input], |[x]| build(&x)).unwrap();
                assert_same(&direct(&input).unwrap(), &graph, "unary");
            }
            let graph = PyTensor::evaluated([&input], |[x]| x.classify_tensor(true)).unwrap();
            assert_same(&input.try_classify(true).unwrap(), &graph, "isnan");
            for (axes, keepdims) in [
                (None, false),
                (None, true),
                (Some(vec![1]), false),
                (Some(vec![-1]), true),
                (Some(vec![0, 2]), false),
            ] {
                for mean in [false, true] {
                    let graph = PyTensor::evaluated([&input], |[x]| {
                        x.reduce_axes_tensor(axes.clone(), keepdims, mean)
                    })
                    .unwrap();
                    let direct = input.reduction(axes.clone(), keepdims, mean).unwrap();
                    assert_same(&direct, &graph, &format!("{axes:?} {keepdims} {mean}"));
                }
            }
            let lhs = tensor(&[4, 5], dtype, 1);
            let rhs = tensor(&[5, 3], dtype, 2);
            let graph = PyTensor::evaluated([&lhs, &rhs], |[x, y]| x.matmul_tensor(&y)).unwrap();
            assert_same(&lhs.try_matmul(&rhs).unwrap(), &graph, "matmul");
            let masks = [
                tensor(&[3, 4], TensorDType::Bool, 0),
                tensor(&[3, 4], TensorDType::Bool, 3),
            ];
            for and in [false, true] {
                let graph =
                    PyTensor::evaluated([&masks[0], &masks[1]], |[x, y]| x.logical_tensor(&y, and))
                        .unwrap();
                assert_same(
                    &masks[0].try_logical(&masks[1], and).unwrap(),
                    &graph,
                    "logical",
                );
            }
        }
    }

    /// A composite evaluated primitive by primitive equals its whole traced
    /// graph, including the signed zeros and NaN signs that the former
    /// hand-written eager versions changed.
    #[test]
    fn composites_equal_their_traced_graph() {
        let composites: [Pair; 4] = [
            (composite::abs, TraceTensor::abs_tensor),
            (composite::relu, TraceTensor::relu_tensor),
            (composite::sigmoid, TraceTensor::sigmoid_tensor),
            (composite::softplus, TraceTensor::softplus_tensor),
        ];
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let input = tensor(&[4, 4], dtype, 0);
            let other = tensor(&[4, 4], dtype, 9);
            for (direct, build) in composites {
                let graph = PyTensor::evaluated([&input], |[x]| build(&x)).unwrap();
                assert_same(&direct(&input).unwrap(), &graph, "composite");
            }
            for value in [0.0, -0.0, 1.5, f64::NAN] {
                let graph = PyTensor::evaluated([&input], |[x]| x.maximum_scalar(value)).unwrap();
                assert_same(&input.try_maximum_scalar(value).unwrap(), &graph, "maximum");
                let graph = PyTensor::evaluated([&input], |[x]| x.minimum_scalar(value)).unwrap();
                assert_same(&input.try_minimum_scalar(value).unwrap(), &graph, "minimum");
            }
            let graph =
                PyTensor::evaluated([&input, &other], |[x, y]| x.maximum_tensor(&y)).unwrap();
            assert_same(
                &input.try_maximum(&other).unwrap(),
                &graph,
                "maximum tensor",
            );
            let graph =
                PyTensor::evaluated([&input, &other], |[x, y]| x.minimum_tensor(&y)).unwrap();
            assert_same(
                &input.try_minimum(&other).unwrap(),
                &graph,
                "minimum tensor",
            );
            let zeros = PyTensor::from_shape_data_typed(vec![2], vec![0.0, -0.0], dtype).unwrap();
            let absolute = zeros.try_abs().unwrap();
            assert!(absolute.data.iter().all(|value| value.to_bits() == 0));
        }
        let mask = tensor(&[4], TensorDType::Bool, 0);
        assert_eq!(mask.try_abs().unwrap_err(), bool_operation_error("abs"));
    }

    #[test]
    fn scalar_broadcast_preserves_operand_order_and_shape_validation() {
        let vector = PyTensor::from_shape_data(vec![2], vec![2.0, 4.0]).unwrap();
        let scalar = PyTensor::from_shape_data(vec![1, 1], vec![8.0]).unwrap();
        let left = scalar.try_sub(&vector).unwrap();
        let right = vector.try_sub(&scalar).unwrap();
        assert_eq!(left.shape, [1, 2]);
        assert_eq!(left.data.to_f64().as_ref(), [6.0, 4.0]);
        assert_eq!(right.data.to_f64().as_ref(), [-6.0, -4.0]);
        let incompatible = PyTensor::from_shape_data(vec![3], vec![1.0; 3]).unwrap();
        assert!(vector
            .try_add(&incompatible)
            .unwrap_err()
            .contains("cannot broadcast"));
    }
}
