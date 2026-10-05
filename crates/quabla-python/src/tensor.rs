use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyMemoryView, PySlice, PySliceMethods, PyTuple};
use quabla_core::tensor_ir::{HostTensorStorage, TensorComparison, TensorDType};
use std::borrow::Cow;
use std::ffi::c_int;
use std::sync::Arc;

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
/// Every eager op rounds its `f64` intermediate result the way
/// the CPU Tensor IR backend rounds an `f32` node. Python scalars are weak:
/// they are rounded to the tensor dtype before the op. A `bool` tensor holds
/// `0.0`/`1.0` and follows the Tensor IR promotion rule.
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

pub fn parse_tensor_indices(
    index: &Bound<'_, PyAny>,
    rank: usize,
    shape: &[usize],
) -> PyResult<Vec<TensorIndex>> {
    let items = if let Ok(tuple) = index.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else {
        vec![index.clone()]
    };
    if items.len() > rank {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor of rank {rank}: got {}",
            items.len()
        )));
    }
    let mut result = Vec::with_capacity(items.len());
    let mut axis = 0;
    for item in items {
        let extent = *shape.get(axis).ok_or_else(|| {
            PyIndexError::new_err(format!("too many indices for tensor of rank {rank}"))
        })?;
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
            result.push(TensorIndex::Integer(value));
            axis += 1;
            continue;
        }
        if let Ok(slice) = item.cast::<PySlice>() {
            let indices = slice.indices(extent as isize)?;
            if indices.step != 1 {
                return Err(PyValueError::new_err(
                    "Tensor indexing currently requires slice step == 1; use Tensor.slice for eager strided views",
                ));
            }
            if indices.slicelength == 0 {
                return Err(PyValueError::new_err(
                    "Tensor indexing currently rejects empty slices",
                ));
            }
            result.push(TensorIndex::Slice {
                start: indices.start as usize,
                stop: indices.stop as usize,
            });
            axis += 1;
            continue;
        }
        return Err(PyTypeError::new_err(
            "Tensor indexing supports integers and contiguous slices only",
        ));
    }
    Ok(result)
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

/// Whether a max/min reduction replaces its running `current` with the later
/// `value`. It mirrors the traced pairwise `maximum(current, value)`: ties
/// take the later element, and a NaN is kept once seen and taken when it
/// arrives, so NaN propagates from any position like NumPy.
fn extrema_replaces(maximum: bool, current: f64, value: f64) -> bool {
    if current.is_nan() {
        return false;
    }
    value.is_nan() || (maximum && value >= current) || (!maximum && value <= current)
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

fn normalize_permutation(axes: Option<Vec<isize>>, rank: usize) -> Result<Vec<usize>, String> {
    let axes = axes.unwrap_or_else(|| (0..rank).rev().map(|axis| axis as isize).collect());
    if axes.len() != rank {
        return Err(format!(
            "transpose axes must have length {rank}, got {}",
            axes.len()
        ));
    }

    let axes = axes
        .into_iter()
        .map(|axis| normalize_axis(axis, rank))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = vec![false; rank];
    for axis in &axes {
        if std::mem::replace(&mut seen[*axis], true) {
            return Err(format!(
                "transpose axes {:?} are not a permutation of 0..{rank}",
                axes
            ));
        }
    }

    Ok(axes)
}

fn normalize_reduction_axes(axes: Vec<isize>, rank: usize) -> Result<Vec<usize>, String> {
    let mut axes = axes
        .into_iter()
        .map(|axis| normalize_axis(axis, rank))
        .collect::<Result<Vec<_>, _>>()?;
    axes.sort_unstable();
    if axes.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("reduction axes must be unique".to_string());
    }
    Ok(axes)
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

fn matmul_float_block<T: Copy + Into<f64>>(
    lhs: &[T],
    rhs: &[T],
    output: &mut [f64],
    [rows, inner, columns]: [usize; 3],
) {
    for row in 0..rows {
        let output = &mut output[row * columns..(row + 1) * columns];
        for k in 0..inner {
            let lhs = lhs[row * inner + k].into();
            let rhs = &rhs[k * columns..(k + 1) * columns];
            // Accumulate in F64 in the original increasing-inner order.
            for (output, &rhs) in output.iter_mut().zip(rhs) {
                *output += lhs * rhs.into();
            }
        }
    }
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);

    for offset in 0..rank {
        let lhs_extent = lhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let rhs_extent = rhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let extent = if lhs_extent == rhs_extent {
            lhs_extent
        } else if lhs_extent == 1 {
            rhs_extent
        } else if rhs_extent == 1 {
            lhs_extent
        } else {
            return Err(format!(
                "cannot broadcast tensor shapes {:?} and {:?}",
                lhs, rhs
            ));
        };
        shape.push(extent);
    }

    shape.reverse();
    Ok(shape)
}

// Advance row-major broadcast offsets with stack-only operand metadata. Only
// wrapped axes propagate a carry, so ordinary elements avoid full rank decoding.
// Inlining specializes the one-to-three operand loop and removes per-element calls.
#[inline(always)]
fn advance_broadcast_offsets<const N: usize>(
    mut next: usize,
    shape: &[usize],
    operands: [(&[usize], &[usize]); N],
    offsets: &mut [usize; N],
) {
    for axis in (0..shape.len()).rev() {
        let wrapped = next.is_multiple_of(shape[axis]);
        for (offset, (input_shape, strides)) in offsets.iter_mut().zip(operands) {
            let rank_offset = shape.len() - input_shape.len();
            if axis >= rank_offset && input_shape[axis - rank_offset] != 1 {
                let step = strides[axis - rank_offset];
                if wrapped {
                    *offset -= (shape[axis] - 1) * step;
                } else {
                    *offset += step;
                }
            }
        }
        if !wrapped {
            break;
        }
        next /= shape[axis];
    }
}

fn broadcast_offset(
    output_index: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let mut remaining = output_index;
    let mut offset = 0;
    let rank_offset = output_shape.len() - input_shape.len();

    for axis in (0..output_shape.len()).rev() {
        let coordinate = remaining % output_shape[axis];
        remaining /= output_shape[axis];

        if axis >= rank_offset {
            let input_axis = axis - rank_offset;
            if input_shape[input_axis] != 1 {
                offset += coordinate * input_strides[input_axis];
            }
        }
    }

    offset
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

    pub fn shape_data(&self) -> (&[usize], Cow<'_, [f64]>) {
        (&self.shape, self.data.to_f64())
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

    /// Strict promotion for two tensors: dtypes must match exactly.
    fn result_dtype(&self, rhs: &Self, op: &str) -> Result<TensorDType, String> {
        if self.dtype != rhs.dtype {
            return Err(format!(
                "tensor {op} operands have mismatched dtypes {} and {}; \
                 convert one of them explicitly with astype",
                self.dtype, rhs.dtype
            ));
        }
        Ok(self.dtype)
    }

    /// Result dtype and weakness under the Tensor IR promotion rule: strong
    /// floats must match, weak floats adopt the strong dtype, and `bool`
    /// joins a float operand as 0/1 values of its dtype and weakness. Two
    /// `bool` operands are accepted only when `allow_bool` (comparisons,
    /// `where`, concatenation).
    fn promotion(
        tensors: &[&Self],
        op: &str,
        allow_bool: bool,
    ) -> Result<(TensorDType, bool), String> {
        let mut result: Option<(TensorDType, bool)> = None;
        for tensor in tensors {
            if tensor.dtype == TensorDType::Bool {
                continue;
            }
            result = Some(match result {
                None => (tensor.dtype, tensor.weak),
                Some((dtype, weak)) if dtype == tensor.dtype => (dtype, weak && tensor.weak),
                Some((_, true)) if !tensor.weak => (tensor.dtype, false),
                Some((dtype, false)) if tensor.weak => (dtype, false),
                Some((dtype, _)) => {
                    return Err(format!(
                        "tensor {op} operands have mismatched dtypes {dtype} and {}; \
                         convert one of them explicitly with astype",
                        tensor.dtype
                    ))
                }
            });
        }
        match result {
            Some(result) => Ok(result),
            None if allow_bool => Ok((TensorDType::Bool, false)),
            None => Err(bool_operation_error(op)),
        }
    }

    /// Operands converted to their promoted dtype, or `None` when they
    /// already share it; callers retry the op on the converted pair.
    fn promoted_pair(
        &self,
        rhs: &Self,
        op: &str,
        allow_bool: bool,
    ) -> Result<Option<(Self, Self)>, String> {
        let (dtype, _) = Self::promotion(&[self, rhs], op, allow_bool)?;
        if self.dtype == dtype && rhs.dtype == dtype {
            return Ok(None);
        }
        Ok(Some((self.converted(dtype), rhs.converted(dtype))))
    }

    fn converted(&self, dtype: TensorDType) -> Self {
        if self.dtype == dtype {
            self.clone()
        } else {
            self.clone().typed(dtype)
        }
    }

    /// A `bool` tensor enters scalar arithmetic as weak `f64` 0/1 values.
    fn arithmetic_base(&self) -> Self {
        if self.dtype == TensorDType::Bool {
            let mut base = self.clone().typed(TensorDType::F64);
            base.weak = true;
            base
        } else {
            self.clone()
        }
    }

    fn ensure_not_bool(&self, op: &str) -> Result<(), String> {
        if self.dtype == TensorDType::Bool {
            return Err(bool_operation_error(op));
        }
        Ok(())
    }

    fn ensure_bool(&self, op: &str) -> Result<(), String> {
        if self.dtype != TensorDType::Bool {
            return Err(format!(
                "{op} requires bool operands, got dtype {}; \
                 build a mask with a comparison or convert explicitly with astype",
                self.dtype
            ));
        }
        Ok(())
    }

    /// Broadcasts two same-dtype operands through `f`, rounding each result
    /// to `dtype`.
    fn zip_broadcast(
        &self,
        rhs: &Self,
        op: &str,
        dtype: TensorDType,
        f: impl Fn(f64, f64) -> Result<f64, String>,
    ) -> Result<Self, String> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let output_size = element_count(&shape)?;
        let mut data = Vec::with_capacity(output_size);

        if self.shape == rhs.shape {
            for (lhs, rhs) in self.data.iter().zip(rhs.data.iter()) {
                let value =
                    f(lhs, rhs).map_err(|err| format!("failed to evaluate tensor {op}: {err}"))?;
                data.push(dtype.round(value));
            }
        } else if self.data.len() == 1 && rhs.data.len() == output_size {
            for rhs in rhs.data.iter() {
                let value = f(self.data.get(0), rhs)
                    .map_err(|err| format!("failed to evaluate tensor {op}: {err}"))?;
                data.push(dtype.round(value));
            }
        } else if rhs.data.len() == 1 && self.data.len() == output_size {
            for lhs in self.data.iter() {
                let value = f(lhs, rhs.data.get(0))
                    .map_err(|err| format!("failed to evaluate tensor {op}: {err}"))?;
                data.push(dtype.round(value));
            }
        } else {
            let lhs_strides = contiguous_strides(&self.shape);
            let rhs_strides = contiguous_strides(&rhs.shape);
            let mut offsets = [0; 2];
            for index in 0..output_size {
                let value = f(self.data.get(offsets[0]), rhs.data.get(offsets[1]))
                    .map_err(|err| format!("failed to evaluate tensor {op}: {err}"))?;
                data.push(dtype.round(value));
                if index + 1 < output_size {
                    advance_broadcast_offsets(
                        index + 1,
                        &shape,
                        [(&self.shape, &lhs_strides), (&rhs.shape, &rhs_strides)],
                        &mut offsets,
                    );
                }
            }
        }

        Ok(Self {
            shape,
            data: HostTensorStorage::from_f64(data, dtype),
            dtype,
            weak: false,
        })
    }

    fn try_elementwise(
        &self,
        rhs: &Self,
        op: &str,
        f: impl Fn(f64, f64) -> Result<f64, String>,
    ) -> Result<Self, String> {
        let (dtype, weak) = Self::promotion(&[self, rhs], op, false)?;
        let lhs = self.converted(dtype);
        let rhs = rhs.converted(dtype);
        let mut output = lhs.zip_broadcast(&rhs, op, dtype, f)?;
        output.weak = weak;
        Ok(output)
    }

    fn try_map(&self, f: impl Fn(f64) -> f64) -> Result<Self, String> {
        // Keep F64 arithmetic per lane, but write directly to dtype-sized output.
        let data = match self.dtype {
            TensorDType::F64 => HostTensorStorage::F64(Arc::new(self.data.iter().map(f).collect())),
            TensorDType::F32 => {
                HostTensorStorage::F32(Arc::new(self.data.iter().map(|v| f(v) as f32).collect()))
            }
            TensorDType::Bool => HostTensorStorage::Bool(Arc::new(
                self.data.iter().map(|v| u8::from(f(v) != 0.0)).collect(),
            )),
        };
        Ok(Self {
            shape: self.shape.clone(),
            data,
            dtype: self.dtype,
            weak: self.weak,
        })
    }

    /// Elementwise float math; `bool` tensors must be converted explicitly.
    fn try_unary(&self, op: &str, f: impl Fn(f64) -> f64) -> Result<Self, String> {
        self.ensure_not_bool(op)?;
        self.try_map(f)
    }

    /// IEEE comparison producing a strong `bool` tensor.
    pub fn try_compare(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String> {
        let (dtype, _) = Self::promotion(&[self, rhs], kind.name(), true)?;
        self.converted(dtype).zip_broadcast(
            &rhs.converted(dtype),
            kind.name(),
            TensorDType::Bool,
            |lhs, rhs| Ok(f64::from(kind.evaluate(lhs, rhs))),
        )
    }

    /// Compares with a weak Python scalar, which adopts the tensor dtype.
    pub fn try_compare_scalar(&self, rhs: f64, kind: TensorComparison) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        Self::from_shape_data_typed(
            base.shape.clone(),
            base.data
                .iter()
                .map(|lhs| f64::from(kind.evaluate(lhs, rhs)))
                .collect(),
            TensorDType::Bool,
        )
    }

    pub fn try_logical(&self, rhs: &Self, and: bool) -> Result<Self, String> {
        let op = if and { "logical_and" } else { "logical_or" };
        self.ensure_bool(op)?;
        rhs.ensure_bool(op)?;
        self.zip_broadcast(rhs, op, TensorDType::Bool, |lhs, rhs| {
            let (lhs, rhs) = (lhs != 0.0, rhs != 0.0);
            Ok(f64::from(if and { lhs && rhs } else { lhs || rhs }))
        })
    }

    pub fn try_logical_not(&self) -> Result<Self, String> {
        self.ensure_bool("logical_not")?;
        self.try_map(|value| f64::from(value == 0.0))
    }

    /// `isnan`/`isfinite` of a float tensor as a `bool` tensor.
    pub fn try_classify(&self, op: &str, test: impl Fn(f64) -> bool) -> Result<Self, String> {
        self.ensure_not_bool(op)?;
        Self::from_shape_data_typed(
            self.shape.clone(),
            self.data
                .iter()
                .map(|value| f64::from(test(value)))
                .collect(),
            TensorDType::Bool,
        )
    }

    /// `any`/`all` as max/min reductions of the 0/1 values.
    pub fn try_any_all(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        any: bool,
    ) -> Result<Self, String> {
        self.ensure_bool(if any { "any" } else { "all" })?;
        self.try_extrema_axes(axes, keepdims, any)
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

    pub fn try_add(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "+", |lhs, rhs| Ok(lhs + rhs))
    }

    pub fn try_add_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| lhs + rhs)
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "-", |lhs, rhs| Ok(lhs - rhs))
    }

    pub fn try_sub_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| lhs - rhs)
    }

    pub fn try_scalar_sub(&self, lhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let lhs = base.dtype.round(lhs);
        base.try_map(|rhs| lhs - rhs)
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "*", |lhs, rhs| Ok(lhs * rhs))
    }

    pub fn try_mul_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| lhs * rhs)
    }

    pub fn try_div(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "/", |lhs, rhs| {
            if rhs == 0.0 {
                return Err("division by zero is not supported".to_string());
            }

            Ok(lhs / rhs)
        })
    }

    pub fn try_div_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        if rhs == 0.0 {
            return Err("division by zero scalar is not supported".to_string());
        }
        base.try_map(|lhs| lhs / rhs)
    }

    pub fn try_scalar_div(&self, lhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let lhs = base.dtype.round(lhs);
        if base.data.iter().any(|value| value == 0.0) {
            return Err("division by zero is not supported".to_string());
        }
        base.try_map(|rhs| lhs / rhs)
    }

    pub fn try_gt(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "gt", |lhs, rhs| Ok(if lhs > rhs { 1.0 } else { 0.0 }))
    }

    pub fn try_gt_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| if lhs > rhs { 1.0 } else { 0.0 })
    }

    /// `where(isnan(x) | (x > y), x, y)` like the traced form: ties select
    /// `y`, and NaN in either operand propagates (a NaN `y` fails `>`).
    pub fn try_maximum(&self, rhs: &Self) -> Result<Self, String> {
        let mask = self.try_elementwise(rhs, "gt", |lhs, rhs| {
            Ok(f64::from(lhs.is_nan() || lhs > rhs))
        })?;
        Self::try_where(&mask, self, rhs)
    }

    pub fn try_maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| if lhs.is_nan() || lhs > rhs { lhs } else { rhs })
    }

    /// `where(isnan(x) | (y > x), x, y)` like the traced form: ties select
    /// `y`, and NaN in either operand propagates.
    pub fn try_minimum(&self, rhs: &Self) -> Result<Self, String> {
        let mask = rhs.try_elementwise(self, "gt", |rhs, lhs| {
            Ok(f64::from(lhs.is_nan() || rhs > lhs))
        })?;
        Self::try_where(&mask, self, rhs)
    }

    pub fn try_minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| if lhs.is_nan() || rhs > lhs { lhs } else { rhs })
    }

    pub fn try_where(mask: &Self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        // The mask is only tested for non-zero (bool or legacy float mask) and does not take part
        // in dtype unification.
        if let Some((on_true, on_false)) = on_true.promoted_pair(on_false, "where", true)? {
            return Self::try_where(mask, &on_true, &on_false);
        }
        let dtype = on_true.result_dtype(on_false, "where")?;
        let value_shape = broadcast_shape(&on_true.shape, &on_false.shape)?;
        let shape = broadcast_shape(&mask.shape, &value_shape)?;
        let count = element_count(&shape)?;
        let mask_strides = contiguous_strides(&mask.shape);
        let true_strides = contiguous_strides(&on_true.shape);
        let false_strides = contiguous_strides(&on_false.shape);
        let mut data = Vec::with_capacity(count);

        if mask.shape == shape && on_true.shape == shape && on_false.shape == shape {
            for ((mask, on_true), on_false) in mask
                .data
                .iter()
                .zip(on_true.data.iter())
                .zip(on_false.data.iter())
            {
                data.push(if mask != 0.0 { on_true } else { on_false });
            }
            return Self::from_shape_data_typed(shape, data, dtype);
        }

        let mut offsets = [0; 3];
        for index in 0..count {
            data.push(if mask.data.get(offsets[0]) != 0.0 {
                on_true.data.get(offsets[1])
            } else {
                on_false.data.get(offsets[2])
            });
            if index + 1 < count {
                advance_broadcast_offsets(
                    index + 1,
                    &shape,
                    [
                        (&mask.shape, &mask_strides),
                        (&on_true.shape, &true_strides),
                        (&on_false.shape, &false_strides),
                    ],
                    &mut offsets,
                );
            }
        }

        Self::from_shape_data_typed(shape, data, dtype)
    }

    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, String> {
        if let Some((lhs, rhs)) = self.promoted_pair(rhs, "matmul", false)? {
            return lhs.try_matmul(&rhs);
        }
        let dtype = self.result_dtype(rhs, "matmul")?;
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            return Err(format!(
                "matmul requires tensors with at least two dimensions, got {:?} and {:?}",
                self.shape, rhs.shape
            ));
        }

        let lhs_rows = self.shape[self.shape.len() - 2];
        let lhs_inner = self.shape[self.shape.len() - 1];
        let rhs_inner = rhs.shape[rhs.shape.len() - 2];
        let rhs_cols = rhs.shape[rhs.shape.len() - 1];
        if lhs_inner != rhs_inner {
            return Err(format!(
                "cannot matmul tensor shapes {:?} and {:?}: inner dimensions {lhs_inner} and {rhs_inner} differ",
                self.shape, rhs.shape
            ));
        }

        let lhs_batch_shape = &self.shape[..self.shape.len() - 2];
        let rhs_batch_shape = &rhs.shape[..rhs.shape.len() - 2];
        let batch_shape = broadcast_shape(lhs_batch_shape, rhs_batch_shape)?;
        let batch_count = element_count(&batch_shape)?;
        let lhs_batch_strides = contiguous_strides(lhs_batch_shape);
        let rhs_batch_strides = contiguous_strides(rhs_batch_shape);

        let mut shape = batch_shape;
        shape.push(lhs_rows);
        shape.push(rhs_cols);
        let mut data = vec![0.0; element_count(&shape)?];

        for batch_index in 0..batch_count {
            let lhs_batch = broadcast_offset(
                batch_index,
                &shape[..shape.len() - 2],
                lhs_batch_shape,
                &lhs_batch_strides,
            );
            let rhs_batch = broadcast_offset(
                batch_index,
                &shape[..shape.len() - 2],
                rhs_batch_shape,
                &rhs_batch_strides,
            );

            let lhs_start = lhs_batch * lhs_rows * lhs_inner;
            let rhs_start = rhs_batch * rhs_inner * rhs_cols;
            let output_start = batch_index * lhs_rows * rhs_cols;
            let output = &mut data[output_start..output_start + lhs_rows * rhs_cols];
            let dimensions = [lhs_rows, lhs_inner, rhs_cols];
            match (&self.data, &rhs.data) {
                (HostTensorStorage::F64(lhs), HostTensorStorage::F64(rhs)) => {
                    matmul_float_block(&lhs[lhs_start..], &rhs[rhs_start..], output, dimensions);
                }
                (HostTensorStorage::F32(lhs), HostTensorStorage::F32(rhs)) => {
                    matmul_float_block(&lhs[lhs_start..], &rhs[rhs_start..], output, dimensions);
                }
                _ => unreachable!("matmul operands have validated matching float dtypes"),
            }
        }

        Self::from_shape_data_typed(shape, data, dtype)
    }

    pub fn try_solve(&self, rhs: &Self) -> Result<Self, String> {
        if let Some((lhs, rhs)) = self.promoted_pair(rhs, "solve", false)? {
            return lhs.try_solve(&rhs);
        }
        let dtype = self.result_dtype(rhs, "solve")?;
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(format!(
                "solve requires rank-2 matrix and right-hand side tensors, got {:?} and {:?}",
                self.shape, rhs.shape
            ));
        }
        let n = self.shape[0];
        if n != self.shape[1] || n != rhs.shape[0] {
            return Err(format!(
                "solve requires coefficient shape {:?} and right-hand side shape {:?} to have compatible rows",
                self.shape, rhs.shape
            ));
        }
        if let Some(result) = self.finite_triangular_solution(rhs) {
            return Self::from_shape_data_typed(rhs.shape.clone(), result, dtype);
        }
        self.solve_lu(rhs, dtype)
    }

    fn solve_lu(&self, rhs: &Self, dtype: TensorDType) -> Result<Self, String> {
        let n = self.shape[0];
        let columns = rhs.shape[1];
        let mut factor = self.data.to_f64().into_owned();
        let mut result = rhs.data.to_f64().into_owned();
        for pivot in 0..n {
            let pivot_row = (pivot..n)
                .max_by(|&left, &right| {
                    factor[left * n + pivot]
                        .abs()
                        .total_cmp(&factor[right * n + pivot].abs())
                })
                .expect("pivot range is non-empty");
            if factor[pivot_row * n + pivot] == 0.0 {
                return Err("solve requires a non-singular coefficient matrix".to_string());
            }
            if pivot_row != pivot {
                for column in 0..n {
                    factor.swap(pivot * n + column, pivot_row * n + column);
                }
                for column in 0..columns {
                    result.swap(pivot * columns + column, pivot_row * columns + column);
                }
            }
            let diagonal = factor[pivot * n + pivot];
            for row in pivot + 1..n {
                let multiplier = factor[row * n + pivot] / diagonal;
                factor[row * n + pivot] = multiplier;
                for column in pivot + 1..n {
                    factor[row * n + column] -= multiplier * factor[pivot * n + column];
                }
                for column in 0..columns {
                    result[row * columns + column] -= multiplier * result[pivot * columns + column];
                }
            }
        }
        for row in (0..n).rev() {
            for column in 0..columns {
                let mut value = result[row * columns + column];
                for inner in row + 1..n {
                    value -= factor[row * n + inner] * result[inner * columns + column];
                }
                result[row * columns + column] = value / factor[row * n + row];
            }
        }
        Self::from_shape_data_typed(rhs.shape.clone(), result, dtype)
    }

    // Exact triangular structure permits substitution without a factor matrix.
    // Exceptional values keep the pivoted solver's existing error/IEEE behavior.
    fn finite_triangular_solution(&self, rhs: &Self) -> Option<Vec<f64>> {
        if self
            .data
            .iter()
            .chain(rhs.data.iter())
            .any(|value| !value.is_finite())
        {
            return None;
        }
        let n = self.shape[0];
        let columns = rhs.shape[1];
        let mut lower = true;
        let mut upper = true;
        for row in 0..n {
            if self.data.get(row * n + row) == 0.0 {
                return None;
            }
            for column in 0..row {
                upper &= self.data.get(row * n + column) == 0.0;
                lower &= self.data.get(column * n + row) == 0.0;
            }
        }
        if !lower && !upper {
            return None;
        }
        let mut result = rhs.data.to_f64().into_owned();
        for step in 0..n {
            let row = if lower { step } else { n - 1 - step };
            let diagonal = self.data.get(row * n + row);
            let dependencies = if lower { 0..row } else { row + 1..n };
            for column in 0..columns {
                let mut value = result[row * columns + column];
                for inner in dependencies.clone() {
                    value -= self.data.get(row * n + inner) * result[inner * columns + column];
                    if !value.is_finite() {
                        return None;
                    }
                }
                let solved = value / diagonal;
                if !solved.is_finite() {
                    return None;
                }
                result[row * columns + column] = solved;
            }
        }
        Some(result)
    }

    pub fn try_solve_triangular(
        &self,
        rhs: &Self,
        lower: bool,
        transpose: bool,
    ) -> Result<Self, String> {
        let matrix = self.try_triangular(lower)?;
        let matrix = if transpose {
            matrix.try_transpose(None)?
        } else {
            matrix
        };
        matrix.try_solve(rhs)
    }

    pub fn try_cholesky(&self) -> Result<Self, String> {
        self.ensure_not_bool("cholesky")?;
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(format!(
                "cholesky requires a square rank-2 tensor, got {:?}",
                self.shape
            ));
        }
        let n = self.shape[0];
        let mut factor = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..=row {
                let symmetric = self.data.get(column * n + row);
                let value = self.data.get(row * n + column);
                let tolerance = 1e-12 * value.abs().max(symmetric.abs()).max(1.0);
                if (value - symmetric).abs() > tolerance {
                    return Err("cholesky requires a symmetric coefficient matrix".to_string());
                }
                let mut reduced = value;
                for inner in 0..column {
                    reduced -= factor[row * n + inner] * factor[column * n + inner];
                }
                if row == column {
                    if reduced.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                        return Err(
                            "cholesky requires a positive-definite coefficient matrix".to_string()
                        );
                    }
                    factor[row * n + column] = reduced.sqrt();
                } else {
                    factor[row * n + column] = reduced / factor[column * n + column];
                }
            }
        }
        Self::from_shape_data_typed(self.shape.clone(), factor, self.dtype)
    }

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
        if broadcast_shape(&self.shape, &shape)? != shape {
            return Err(format!(
                "cannot broadcast tensor shape {:?} to {:?}",
                self.shape, shape
            ));
        }
        let count = element_count(&shape)?;
        let strides = contiguous_strides(&self.shape);
        let mut data = Vec::with_capacity(count);
        let mut offsets = [0];
        for index in 0..count {
            data.push(self.data.get(offsets[0]));
            if index + 1 < count {
                advance_broadcast_offsets(
                    index + 1,
                    &shape,
                    [(&self.shape, &strides)],
                    &mut offsets,
                );
            }
        }
        Self::from_shape_data_typed(shape, data, self.dtype)
    }

    pub fn try_transpose(&self, axes: Option<Vec<isize>>) -> Result<Self, String> {
        let axes = normalize_permutation(axes, self.shape.len())?;
        let shape = axes
            .iter()
            .map(|axis| self.shape[*axis])
            .collect::<Vec<_>>();
        let input_strides = contiguous_strides(&self.shape);
        let mut data = vec![0.0; self.data.len()];

        for (output_index, output_value) in data.iter_mut().enumerate() {
            let mut remaining = output_index;
            let mut input_index = 0;
            for output_axis in (0..shape.len()).rev() {
                let coordinate = remaining % shape[output_axis];
                remaining /= shape[output_axis];
                input_index += coordinate * input_strides[axes[output_axis]];
            }
            *output_value = self.data.get(input_index);
        }

        Self::from_shape_data_typed(shape, data, self.dtype)
    }

    fn try_reduce(&self, axis: Option<isize>, scale: f64) -> Result<Self, String> {
        self.ensure_not_bool(if scale == 1.0 { "sum" } else { "mean" })?;
        let Some(axis) = axis else {
            return Self::from_shape_data_typed(
                vec![],
                vec![self.data.iter().sum::<f64>() * scale],
                self.dtype,
            );
        };
        let axis = normalize_axis(axis, self.shape.len())?;
        let mut shape = self.shape.clone();
        shape.remove(axis);
        let mut data = vec![0.0; element_count(&shape)?];
        let inner = element_count(&self.shape[axis + 1..])?;
        let outer = element_count(&self.shape[..axis])?;
        let extent = self.shape[axis];
        // Preserve each output's source order and scale before accumulation.
        for block in 0..outer {
            for reduced in 0..extent {
                let source = (block * extent + reduced) * inner;
                let output = block * inner;
                for offset in 0..inner {
                    data[output + offset] += self.data.get(source + offset) * scale;
                }
            }
        }

        Self::from_shape_data_typed(shape, data, self.dtype)
    }

    pub fn try_sum(&self, axis: Option<isize>) -> Result<Self, String> {
        self.try_reduce(axis, 1.0)
    }

    pub fn try_mean(&self, axis: Option<isize>) -> Result<Self, String> {
        let scale = match axis {
            Some(axis) => 1.0 / self.shape[normalize_axis(axis, self.shape.len())?] as f64,
            None => 1.0 / self.data.len() as f64,
        };
        self.try_reduce(axis, scale)
    }

    pub fn try_sum_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.try_reduce_axes(axes, keepdims, false)
    }

    pub fn try_mean_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.try_reduce_axes(axes, keepdims, true)
    }

    pub fn try_norm(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.ensure_not_bool("norm")?;
        self.try_powi(2)?.try_sum_axes(axes, keepdims)?.try_sqrt()
    }

    pub fn try_max_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.ensure_not_bool("max")?;
        self.try_extrema_axes(axes, keepdims, true)
    }

    pub fn try_min_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.ensure_not_bool("min")?;
        self.try_extrema_axes(axes, keepdims, false)
    }

    fn try_reduce_axes(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        mean: bool,
    ) -> Result<Self, String> {
        let Some(axes) = axes else {
            let reduced = if mean {
                self.try_mean(None)?
            } else {
                self.try_sum(None)?
            };
            return if keepdims {
                reduced.try_reshape(vec![1; self.shape.len()])
            } else {
                Ok(reduced)
            };
        };
        let mut axes = normalize_reduction_axes(axes, self.shape.len())?;
        axes.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        let mut reduced = self.clone();
        for axis in axes {
            reduced = if mean {
                reduced.try_mean(Some(axis as isize))?
            } else {
                reduced.try_sum(Some(axis as isize))?
            };
            if keepdims {
                let mut shape = reduced.shape.clone();
                shape.insert(axis, 1);
                reduced = reduced.try_reshape(shape)?;
            }
        }
        Ok(reduced)
    }

    fn try_extrema_axes(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        maximum: bool,
    ) -> Result<Self, String> {
        let Some(axes) = axes else {
            let reduced = self.try_extrema(None, maximum)?;
            return if keepdims {
                reduced.try_reshape(vec![1; self.shape.len()])
            } else {
                Ok(reduced)
            };
        };
        let mut axes = normalize_reduction_axes(axes, self.shape.len())?;
        axes.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        let mut reduced = self.clone();
        for axis in axes {
            reduced = reduced.try_extrema(Some(axis as isize), maximum)?;
            if keepdims {
                let mut shape = reduced.shape.clone();
                shape.insert(axis, 1);
                reduced = reduced.try_reshape(shape)?;
            }
        }
        Ok(reduced)
    }

    fn try_extrema(&self, axis: Option<isize>, maximum: bool) -> Result<Self, String> {
        if self.data.is_empty() {
            return Err("max/min reduction requires at least one tensor element".to_string());
        }
        let Some(axis) = axis else {
            let value = self.data.iter().fold(self.data.get(0), |current, value| {
                if extrema_replaces(maximum, current, value) {
                    value
                } else {
                    current
                }
            });
            return Self::from_shape_data_typed(vec![], vec![value], self.dtype);
        };
        let axis = normalize_axis(axis, self.shape.len())?;
        if self.shape[axis] == 0 {
            return Err("max/min reduction requires a non-empty reduced axis".to_string());
        }
        let mut shape = self.shape.clone();
        shape.remove(axis);
        let output_strides = contiguous_strides(&shape);
        let mut data = vec![None; element_count(&shape)?];

        for (source_index, value) in self.data.iter().enumerate() {
            let mut remaining = source_index;
            let mut output_index = 0;
            for source_axis in (0..self.shape.len()).rev() {
                let coordinate = remaining % self.shape[source_axis];
                remaining /= self.shape[source_axis];
                if source_axis != axis {
                    let output_axis = if source_axis < axis {
                        source_axis
                    } else {
                        source_axis - 1
                    };
                    output_index += coordinate * output_strides[output_axis];
                }
            }
            match data[output_index] {
                Some(current) if !extrema_replaces(maximum, current, value) => {}
                _ => data[output_index] = Some(value),
            }
        }

        Self::from_shape_data_typed(
            shape,
            data.into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "max/min reduction produced an empty output".to_string())?,
            self.dtype,
        )
    }

    pub fn try_tanh(&self) -> Result<Self, String> {
        self.try_unary("tanh", f64::tanh)
    }

    pub fn try_exp(&self) -> Result<Self, String> {
        self.try_unary("exp", f64::exp)
    }

    pub fn try_log(&self) -> Result<Self, String> {
        self.try_unary("log", f64::ln)
    }

    pub fn try_log1p(&self) -> Result<Self, String> {
        self.try_unary("log1p", f64::ln_1p)
    }

    pub fn try_sqrt(&self) -> Result<Self, String> {
        self.try_unary("sqrt", f64::sqrt)
    }

    pub fn try_relu(&self) -> Result<Self, String> {
        self.ensure_not_bool("relu")?;
        self.try_maximum_scalar(0.0)
    }

    #[allow(clippy::neg_multiply)] // Multiplication preserves the old NaN quieting behavior.
    pub fn try_abs(&self) -> Result<Self, String> {
        self.ensure_not_bool("abs")?;
        if self.data.iter().any(f64::is_nan) {
            // LLVM may change NaN signs/payloads when fusing the multiply.
            // Preserve the established kernels for exceptional inputs.
            let mask = self.try_gt_scalar(0.0)?;
            let negative = self.try_mul_scalar(-1.0)?;
            return Self::try_where(&mask, self, &negative);
        }
        // Preserve the former where(x > 0, x, x * -1) semantics, including
        // signed zero, NaNs, and the intermediate multiplication rounding.
        // The final where result is strongly typed even for weak inputs.
        Ok(Self {
            shape: self.shape.clone(),
            data: HostTensorStorage::from_f64(
                self.data
                    .iter()
                    .map(|value| {
                        let selected = if value > 0.0 {
                            value
                        } else {
                            self.dtype.round(value * -1.0)
                        };
                        self.dtype.round(selected)
                    })
                    .collect(),
                self.dtype,
            ),
            dtype: self.dtype,
            weak: false,
        })
    }

    /// The traced `sigmoid` expression `where(x > 0, 1 / (1 + z), z / (1 + z))`
    /// with `z = exp(-|x|)`, rounded to the dtype after every op exactly as
    /// per-node CPU execution does, so eager and CPU `jit` agree bitwise.
    pub fn try_sigmoid(&self) -> Result<Self, String> {
        let round = |value: f64| self.dtype.round(value);
        self.try_unary("sigmoid", |value| {
            let decay = round((-value.abs()).exp());
            let denominator = round(decay + 1.0);
            if value > 0.0 {
                1.0 / denominator
            } else {
                decay / denominator
            }
        })
    }

    /// The traced `softplus` expression `maximum(x, 0) + log1p(exp(-|x|))`,
    /// rounded to the dtype after every op like `try_sigmoid`.
    pub fn try_softplus(&self) -> Result<Self, String> {
        let round = |value: f64| self.dtype.round(value);
        self.try_unary("softplus", |value| {
            let linear = if value.is_nan() || value > 0.0 {
                value
            } else {
                0.0
            };
            linear + round(round((-value.abs()).exp()).ln_1p())
        })
    }

    pub fn try_triangular(&self, lower: bool) -> Result<Self, String> {
        self.ensure_not_bool(if lower { "tril" } else { "triu" })?;
        if self.shape.len() < 2 {
            return Err(format!(
                "triangular projection requires at least rank two, got {:?}",
                self.shape
            ));
        }
        let columns = self.shape[self.shape.len() - 1];
        let rows = self.shape[self.shape.len() - 2];
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let row = (index / columns) % rows;
                let column = index % columns;
                if (lower && row >= column) || (!lower && row <= column) {
                    value
                } else {
                    0.0
                }
            })
            .collect();
        Self::from_shape_data_typed(self.shape.clone(), data, self.dtype)
    }

    pub fn try_sin(&self) -> Result<Self, String> {
        self.try_unary("sin", f64::sin)
    }

    pub fn try_cos(&self) -> Result<Self, String> {
        self.try_unary("cos", f64::cos)
    }

    pub fn try_powi(&self, exponent: u32) -> Result<Self, String> {
        self.try_unary("powi", |value| value.powf(exponent as f64))
    }

    pub fn try_powf(&self, exponent: f64) -> Result<Self, String> {
        self.try_unary("pow", |value| value.powf(exponent))
    }

    /// Elementwise `self ** exponent` of two tensors with broadcasting and
    /// the dtype promotion of the other binary ops (strict between tensors).
    /// `bool` operands are rejected, as by the scalar-exponent form.
    pub fn try_pow(&self, exponent: &Self) -> Result<Self, String> {
        self.ensure_not_bool("pow")?;
        exponent.ensure_not_bool("pow")?;
        self.try_elementwise(exponent, "**", |base, exponent| Ok(base.powf(exponent)))
    }

    /// `base ** self` for a Python number `base`, which is a weak scalar
    /// rounded to the tensor dtype like the other scalar operands.
    pub fn try_scalar_pow(&self, base: f64) -> Result<Self, String> {
        self.ensure_not_bool("pow")?;
        let base = self.dtype.round(base);
        self.try_map(|exponent| base.powf(exponent))
    }

    pub fn try_concat(tensors: &[PyTensor], axis: usize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "concat requires at least one tensor".to_string())?;
        let (dtype, _) = Self::promotion(&tensors.iter().collect::<Vec<_>>(), "concat", true)?;
        if tensors.iter().any(|tensor| tensor.dtype != dtype) {
            let converted = tensors
                .iter()
                .map(|tensor| tensor.converted(dtype))
                .collect::<Vec<_>>();
            return Self::try_concat(&converted, axis);
        }
        if axis >= first.shape.len() {
            return Err(format!(
                "concat axis {axis} is out of bounds for rank {}",
                first.shape.len()
            ));
        }
        let mut shape = first.shape.clone();
        let mut axis_extent = 0usize;
        for tensor in tensors {
            if tensor.shape.len() != shape.len() {
                return Err(format!(
                    "concat requires tensors with the same rank, got {:?} and {:?}",
                    shape, tensor.shape
                ));
            }
            for (dimension, (&expected, &actual)) in shape.iter().zip(&tensor.shape).enumerate() {
                if dimension != axis && expected != actual {
                    return Err(format!(
                        "cannot concatenate shapes {:?} and {:?} along axis {axis}",
                        shape, tensor.shape
                    ));
                }
            }
            axis_extent = axis_extent
                .checked_add(tensor.shape[axis])
                .ok_or_else(|| "concat axis extent overflows usize".to_string())?;
        }
        shape[axis] = axis_extent;

        let outer = first.shape[..axis].iter().product::<usize>();
        let inner = first.shape[axis + 1..].iter().product::<usize>();
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for outer_index in 0..outer {
            for tensor in tensors {
                let block = tensor.shape[axis]
                    .checked_mul(inner)
                    .ok_or_else(|| "concat block size overflows usize".to_string())?;
                let start = outer_index
                    .checked_mul(block)
                    .ok_or_else(|| "concat offset overflows usize".to_string())?;
                data.extend((start..start + block).map(|index| tensor.data.get(index)));
            }
        }
        Self::from_shape_data_typed(shape, data, first.dtype)
    }

    pub fn try_stack(tensors: &[PyTensor], axis: isize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "stack requires at least one tensor".to_string())?;
        if tensors.iter().any(|tensor| tensor.shape != first.shape) {
            return Err("stack requires tensors with identical shapes".to_string());
        }
        let rank = first.shape.len() + 1;
        let axis = normalize_axis(axis, rank)?;
        let reshaped = tensors
            .iter()
            .map(|tensor| {
                let mut shape = tensor.shape.clone();
                shape.insert(axis, 1);
                tensor.try_reshape(shape)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_concat(&reshaped, axis)
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
        output.materialize()
    }

    pub fn try_gather(&self, indices: &[usize], axis: isize) -> Result<Self, String> {
        let axis = normalize_axis(axis, self.shape.len())?;
        if indices.is_empty() {
            return Err("gather indices must not be empty".to_string());
        }
        if indices.iter().any(|index| *index >= self.shape[axis]) {
            return Err(format!(
                "gather index is out of bounds for axis {axis} with extent {}",
                self.shape[axis]
            ));
        }
        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        let mut data = Vec::with_capacity(element_count(&shape)?);
        let outer = self.shape[..axis].iter().product::<usize>();
        let inner = self.shape[axis + 1..].iter().product::<usize>();
        // Each selected element along the axis owns one contiguous inner
        // block; copying it directly avoids retaining all sliced tensors.
        for outer_index in 0..outer {
            for &index in indices {
                let start = (outer_index * self.shape[axis] + index) * inner;
                data.extend((start..start + inner).map(|index| self.data.get(index)));
            }
        }
        Self::from_shape_data_typed(shape, data, self.dtype)
    }

    pub fn try_scatter_add(
        &self,
        indices: &[usize],
        updates: &Self,
        axis: isize,
    ) -> Result<Self, String> {
        if let Some((base, updates)) = self.promoted_pair(updates, "scatter_add", false)? {
            return base.try_scatter_add(indices, &updates, axis);
        }
        let dtype = self.result_dtype(updates, "scatter_add")?;
        let axis = normalize_axis(axis, self.shape.len())?;
        if indices.is_empty() {
            return Err("scatter indices must not be empty".to_string());
        }
        if indices.iter().any(|index| *index >= self.shape[axis]) {
            return Err(format!(
                "scatter index is out of bounds for axis {axis} with extent {}",
                self.shape[axis]
            ));
        }
        if updates.shape.len() != self.shape.len()
            || updates.shape.iter().enumerate().any(|(dimension, extent)| {
                if dimension == axis {
                    *extent != indices.len()
                } else {
                    *extent != self.shape[dimension]
                }
            })
        {
            return Err(format!(
                "scatter updates shape {:?} is incompatible with base shape {:?}, axis {axis}, and {} indices",
                updates.shape,
                self.shape,
                indices.len()
            ));
        }
        let strides = contiguous_strides(&self.shape);
        let mut data = self.data.to_f64().into_owned();
        for (source_index, value) in updates.data.iter().enumerate() {
            let mut remaining = source_index;
            let mut destination_index = 0;
            for dimension in (0..updates.shape.len()).rev() {
                let coordinate = remaining % updates.shape[dimension];
                remaining /= updates.shape[dimension];
                let coordinate = if dimension == axis {
                    indices[coordinate]
                } else {
                    coordinate
                };
                destination_index += coordinate * strides[dimension];
            }
            data[destination_index] += value;
        }
        Self::from_shape_data_typed(self.shape.clone(), data, dtype)
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
    fn _adam_update(
        &self,
        gradient: &Self,
        first: &Self,
        second: &Self,
        hyperparameters: (f64, f64, f64, f64, f64, f64),
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
            let delta = (m / correction1) / denominator;
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

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.try_reshape(shape).map_err(PyValueError::new_err)
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.try_broadcast_to(shape).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axes = None))]
    fn transpose(&self, axes: Option<Vec<isize>>) -> PyResult<Self> {
        self.try_transpose(axes).map_err(PyValueError::new_err)
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
        let indices = parse_tensor_indices(index, self.shape.len(), &self.shape)?;
        self.try_index(&indices).map_err(PyValueError::new_err)
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
        self.ensure_not_bool("negative")
            .and_then(|_| self.try_mul_scalar(-1.0))
            .map_err(PyValueError::new_err)
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
        self.try_classify("isnan", f64::is_nan)
            .map_err(PyValueError::new_err)
    }

    fn isfinite(&self) -> PyResult<Self> {
        self.try_classify("isfinite", f64::is_finite)
            .map_err(PyValueError::new_err)
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

    fn __repr__(&self) -> String {
        match self.dtype {
            TensorDType::F64 => format!("Tensor(shape={:?})", self.shape),
            dtype => format!(
                "Tensor(shape={:?}, dtype={})",
                self.shape,
                PyDType::from(dtype).__repr__()
            ),
        }
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
        Ok(output)
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
                    .map(|&index| input.try_slice(axis, index, 1, 1)?.materialize())
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
mod triangular_solve_tests {
    use super::*;

    #[test]
    fn eager_triangular_solve_preserves_projection_transpose_and_rounding() -> Result<(), String> {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let original = PyTensor::from_shape_data_typed(
                vec![3, 3],
                vec![2.0, 0.7, -0.2, 3.5, 4.0, 0.3, -0.6, 0.4, 5.0],
                dtype,
            )?;
            let rhs = PyTensor::from_shape_data_typed(
                vec![3, 2],
                vec![0.3, -1.2, 2.1, 0.4, -0.1, 4.3],
                dtype,
            )?;
            for lower in [false, true] {
                for transpose in [false, true] {
                    let projected = original.try_triangular(lower)?;
                    let matrix = if transpose {
                        projected.try_transpose(None)?
                    } else {
                        projected
                    };
                    let scratch = matrix.finite_triangular_solution(&rhs).unwrap();
                    let actual = original.try_solve_triangular(&rhs, lower, transpose)?;
                    let previous = matrix.solve_lu(&rhs, dtype)?;
                    let rounded =
                        PyTensor::from_shape_data_typed(rhs.shape.clone(), scratch, dtype)?;
                    assert_eq!(actual.dtype, dtype);
                    assert_eq!(
                        actual.data.to_f64().as_ref(),
                        rounded.data.to_f64().as_ref()
                    );
                    let tolerance = if dtype == TensorDType::F32 {
                        1e-6
                    } else {
                        1e-12
                    };
                    for (actual, old) in actual.data.iter().zip(previous.data.iter()) {
                        assert!((actual - old).abs() < tolerance * old.abs().max(1.0));
                    }
                    for (residual, expected) in
                        matrix.try_matmul(&actual)?.data.iter().zip(rhs.data.iter())
                    {
                        assert!((residual - expected).abs() < tolerance * expected.abs().max(1.0));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn eager_triangular_solve_keeps_dtype_shape_and_singular_errors() -> Result<(), String> {
        let matrix = PyTensor::from_shape_data_typed(
            vec![2, 2],
            vec![2.0, 0.0, 1.0, 3.0],
            TensorDType::F32,
        )?;
        let rhs = PyTensor::from_shape_data_typed(vec![2, 1], vec![1.0, 2.0], TensorDType::F64)?;
        assert!(matrix
            .try_solve_triangular(&rhs, true, false)
            .unwrap_err()
            .contains("mismatched dtypes"));
        let rhs = rhs.converted(TensorDType::F32);
        let singular = PyTensor::from_shape_data_typed(
            vec![2, 2],
            vec![0.0, 0.0, 1.0, 3.0],
            TensorDType::F32,
        )?;
        assert_eq!(
            singular
                .try_solve_triangular(&rhs, true, false)
                .unwrap_err(),
            "solve requires a non-singular coefficient matrix"
        );
        let vector = PyTensor::from_shape_data_typed(vec![2], vec![1.0, 2.0], TensorDType::F32)?;
        assert!(matrix
            .try_solve_triangular(&vector, true, false)
            .unwrap_err()
            .contains("rank-2"));
        Ok(())
    }
}

#[cfg(test)]
mod abs_tests {
    use super::*;

    fn previous_abs(input: &PyTensor) -> PyTensor {
        let mask = input.try_gt_scalar(0.0).unwrap();
        let negative = input.try_mul_scalar(-1.0).unwrap();
        PyTensor::try_where(&mask, input, &negative).unwrap()
    }

    #[test]
    fn abs_preserves_ieee_bits_and_dtype_for_both_float_types() {
        let values = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff8_0000_0000_1234),
            f64::from_bits(0x7ff0_0000_0000_1234),
            f64::from_bits(0xfff0_0000_0000_1234),
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::from(f32::from_bits(1)),
            -f64::from(f32::from_bits(1)),
            1.000_000_06,
            -1.000_000_06,
            f64::MAX,
            -f64::MAX,
        ];
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let input = PyTensor::from_shape_data_typed(vec![3, 6], values.clone(), dtype).unwrap();
            let expected = previous_abs(&input);
            let actual = input.try_abs().unwrap();
            assert_eq!(actual.shape, input.shape);
            assert_eq!(actual.dtype, dtype);
            assert_eq!(storage_capacity(&actual.data), input.data.len());
            assert!(!storage_ptr_eq(&actual.data, &input.data));
            for (actual, expected) in actual.data.iter().zip(expected.data.iter()) {
                assert_eq!(actual.to_bits(), expected.to_bits(), "dtype {dtype}");
            }
            assert_eq!(actual.data.get(0).to_bits(), (-0.0_f64).to_bits());
            assert_eq!(actual.data.get(1).to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn abs_preserves_strong_result_of_weak_inputs_and_bool_error() {
        let input = PyTensor::weak_scalar(-2.0);
        let expected = previous_abs(&input);
        let actual = input.try_abs().unwrap();
        assert_eq!(actual.weak, expected.weak);
        assert!(!actual.weak);
        assert_eq!(actual.data.to_f64().as_ref(), &[2.0]);
        assert!(input.weak);
        assert_eq!(input.data.to_f64().as_ref(), &[-2.0]);

        let input =
            PyTensor::from_shape_data_typed(vec![2], vec![0.0, 1.0], TensorDType::Bool).unwrap();
        assert_eq!(input.try_abs().unwrap_err(), bool_operation_error("abs"));
    }
}

#[cfg(test)]
mod matmul_tests {
    use super::*;

    #[test]
    fn contiguous_batched_matmul_preserves_accumulation_and_rounding() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let lhs = PyTensor::from_shape_data_typed(
                vec![3, 5, 37],
                (0..555)
                    .map(|i| match i % 6 {
                        0 => 1e16,
                        1 => 1.0,
                        2 => -1e16,
                        _ => (i as f64 - 71.0) / 13.0,
                    })
                    .collect(),
                dtype,
            )
            .unwrap();
            let rhs = PyTensor::from_shape_data_typed(
                vec![1, 37, 29],
                (0..1073).map(|i| (i as f64 % 17.0 - 8.0) / 7.0).collect(),
                dtype,
            )
            .unwrap();
            let actual = lhs.try_matmul(&rhs).unwrap();
            assert_eq!(actual.shape, [3, 5, 29]);
            assert_eq!(actual.dtype, dtype);
            for batch in 0..3 {
                for row in 0..5 {
                    for col in 0..29 {
                        let mut expected = 0.0;
                        for inner in 0..37 {
                            expected += lhs.data.get(batch * 185 + row * 37 + inner)
                                * rhs.data.get(inner * 29 + col);
                        }
                        assert_eq!(
                            actual.data.get(batch * 145 + row * 29 + col).to_bits(),
                            dtype.round(expected).to_bits(),
                            "dtype {dtype}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod adam_tests {
    use super::*;

    #[test]
    fn pure_adam_allocates_three_independent_buffers_and_preserves_aliases() {
        let input = PyTensor::from_shape_data(vec![2], vec![1.0, -2.0]).unwrap();
        let (parameters, moments, variances) = input
            ._adam_update(&input, &input, &input, (0.1, 0.5, 0.75, 0.01, 0.5, 0.25))
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
mod contiguous_reduction_tests {
    use super::*;

    #[test]
    fn axis_reduction_preserves_coordinate_reference_bits() {
        for shape in [vec![2, 3, 5], vec![1, 2, 1, 3], vec![6]] {
            for dtype in [TensorDType::F32, TensorDType::F64] {
                let values = (0..element_count(&shape).unwrap())
                    .map(|i| match i % 8 {
                        0 => 1e16,
                        1 => 1.0,
                        2 => -1e16,
                        3 => -0.0,
                        4 => f64::INFINITY,
                        5 => f64::NEG_INFINITY,
                        6 => f64::from_bits(0x7ff8_0000_0000_1234),
                        _ => 0.125,
                    })
                    .collect();
                let input = PyTensor::from_shape_data_typed(shape.clone(), values, dtype).unwrap();
                for (axis, &extent) in shape.iter().enumerate() {
                    for scale in [1.0, 1.0 / extent as f64] {
                        let mut output_shape = shape.clone();
                        output_shape.remove(axis);
                        let strides = contiguous_strides(&output_shape);
                        let mut expected = vec![0.0; element_count(&output_shape).unwrap()];
                        for (index, value) in input.data.iter().enumerate() {
                            let mut remaining = index;
                            let mut output_index = 0;
                            for source_axis in (0..shape.len()).rev() {
                                let coordinate = remaining % shape[source_axis];
                                remaining /= shape[source_axis];
                                if source_axis != axis {
                                    let output_axis = if source_axis < axis {
                                        source_axis
                                    } else {
                                        source_axis - 1
                                    };
                                    output_index += coordinate * strides[output_axis];
                                }
                            }
                            expected[output_index] += value * scale;
                        }
                        let actual = input.try_reduce(Some(axis as isize), scale).unwrap();
                        assert_eq!(actual.shape, output_shape);
                        assert_eq!(actual.dtype, dtype);
                        for (actual, expected) in actual.data.iter().zip(expected) {
                            assert_eq!(actual.to_bits(), dtype.round(expected).to_bits());
                        }
                    }
                }
            }
        }
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
mod broadcast_carry_tests {
    use super::*;

    fn reference_zip(
        lhs: &PyTensor,
        rhs: &PyTensor,
        dtype: TensorDType,
        f: impl Fn(f64, f64) -> Result<f64, String>,
    ) -> Result<PyTensor, String> {
        let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
        let lhs_strides = contiguous_strides(&lhs.shape);
        let rhs_strides = contiguous_strides(&rhs.shape);
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for index in 0..element_count(&shape)? {
            let lhs_index = broadcast_offset(index, &shape, &lhs.shape, &lhs_strides);
            let rhs_index = broadcast_offset(index, &shape, &rhs.shape, &rhs_strides);
            let value = f(lhs.data.get(lhs_index), rhs.data.get(rhs_index))
                .map_err(|err| format!("failed to evaluate tensor test: {err}"))?;
            data.push(dtype.round(value));
        }
        Ok(PyTensor {
            shape,
            data: HostTensorStorage::from_f64(data, dtype),
            dtype,
            weak: false,
        })
    }

    fn fixture(shape: Vec<usize>, dtype: TensorDType) -> PyTensor {
        let count = element_count(&shape).unwrap();
        PyTensor::from_shape_data_typed(
            shape,
            (0..count)
                .map(|index| match index % 7 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f64::INFINITY,
                    3 => f64::NAN,
                    _ => index as f64 * 0.13 - 0.7,
                })
                .collect(),
            dtype,
        )
        .unwrap()
    }

    #[test]
    fn carried_offsets_preserve_output_bits_and_first_error() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            for (lhs_shape, rhs_shape) in [
                (vec![3, 1], vec![1, 5]),
                (vec![2, 1, 4, 1], vec![3, 1, 5]),
                (vec![2, 1, 1, 1, 3, 1], vec![1, 4]),
                (vec![2, 3], vec![3]),
                (vec![2, 3], vec![4]),
            ] {
                let lhs = fixture(lhs_shape, dtype);
                let rhs = fixture(rhs_shape, dtype);
                for checked in [false, true] {
                    let operation = |left, right| {
                        if checked && right == 0.0 {
                            Err("zero divisor".into())
                        } else {
                            Ok(left - right)
                        }
                    };
                    match (
                        lhs.zip_broadcast(&rhs, "test", dtype, operation),
                        reference_zip(&lhs, &rhs, dtype, operation),
                    ) {
                        (Ok(actual), Ok(expected)) => {
                            assert_eq!(actual.shape, expected.shape);
                            assert_eq!(
                                actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                                expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
                            );
                        }
                        (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                        pair => panic!("different broadcast outcomes: {pair:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn direct_unary_storage_preserves_dtype_bits_and_weakness() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let source = fixture(vec![7], dtype);
            let operation = |value: f64| -value;
            let expected = PyTensor::from_shape_data_typed(
                source.shape.clone(),
                source.data.iter().map(operation).collect(),
                dtype,
            )
            .unwrap();
            let actual = source.try_map(operation).unwrap();
            assert_eq!(actual.dtype, expected.dtype);
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(
                actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
            );
        }
        let scalar = PyTensor::weak_scalar(-0.0);
        let result = scalar.try_map(|value| -value).unwrap();
        assert!(result.weak);
        assert_eq!(result.data.get(0).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    #[ignore = "complete native broadcast release timing probe"]
    fn profile_broadcast_carry() {
        use std::time::Instant;
        for (lhs_shape, rhs_shape) in [
            (vec![4, 1], vec![1, 4]),
            (vec![256, 1], vec![1, 256]),
            (vec![16, 1, 16, 1], vec![1, 16, 1, 16]),
            (vec![4, 1, 4, 1, 4, 1, 4, 1], vec![1, 4, 1, 4, 1, 4, 1, 4]),
        ] {
            let rank = lhs_shape.len();
            let elements =
                element_count(&broadcast_shape(&lhs_shape, &rhs_shape).unwrap()).unwrap();
            let lhs = fixture(lhs_shape, TensorDType::F32);
            let rhs = fixture(rhs_shape, TensorDType::F32);
            let modes = if std::env::var_os("QUABLA_BROADCAST_CARRY_FIRST").is_some() {
                [false, true]
            } else {
                [true, false]
            };
            for reference in modes {
                let mut times = Vec::new();
                for _ in 0..15 {
                    let start = Instant::now();
                    let output = if reference {
                        reference_zip(&lhs, &rhs, TensorDType::F32, |left, right| Ok(left - right))
                    } else {
                        lhs.zip_broadcast(&rhs, "test", TensorDType::F32, |left, right| {
                            Ok(left - right)
                        })
                    }
                    .unwrap();
                    std::hint::black_box(output);
                    times.push(start.elapsed().as_nanos());
                }
                times.sort();
                println!("{{\"rank\":{rank},\"elements\":{elements},\"reference\":{reference},\"median_ns\":{}}}", times[7]);
            }
        }
    }
    fn reference_where(
        mask: &PyTensor,
        on_true: &PyTensor,
        on_false: &PyTensor,
    ) -> Result<PyTensor, String> {
        let shape = broadcast_shape(
            &mask.shape,
            &broadcast_shape(&on_true.shape, &on_false.shape)?,
        )?;
        let strides = [
            contiguous_strides(&mask.shape),
            contiguous_strides(&on_true.shape),
            contiguous_strides(&on_false.shape),
        ];
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for index in 0..element_count(&shape)? {
            let m = broadcast_offset(index, &shape, &mask.shape, &strides[0]);
            let t = broadcast_offset(index, &shape, &on_true.shape, &strides[1]);
            let f = broadcast_offset(index, &shape, &on_false.shape, &strides[2]);
            data.push(if mask.data.get(m) != 0.0 {
                on_true.data.get(t)
            } else {
                on_false.data.get(f)
            });
        }
        PyTensor::from_shape_data_typed(shape, data, on_true.dtype)
    }

    fn reference_broadcast(input: &PyTensor, target: &[usize]) -> Result<PyTensor, String> {
        if broadcast_shape(&input.shape, target)? != target {
            return Err(format!(
                "cannot broadcast tensor shape {:?} to {:?}",
                input.shape, target
            ));
        }
        let strides = contiguous_strides(&input.shape);
        let data = (0..element_count(target)?)
            .map(|index| {
                input
                    .data
                    .get(broadcast_offset(index, target, &input.shape, &strides))
            })
            .collect();
        PyTensor::from_shape_data_typed(target.to_vec(), data, input.dtype)
    }

    #[test]
    fn selection_and_unary_broadcast_preserve_bits_dtype_and_errors() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let mask = PyTensor::from_shape_data_typed(
                vec![2, 1, 1, 1],
                vec![0.0, 1.0],
                TensorDType::Bool,
            )
            .unwrap();
            let on_true = fixture(vec![1, 3, 1, 4], dtype);
            let on_false = fixture(vec![2, 1, 5, 1], dtype);
            let actual = PyTensor::try_where(&mask, &on_true, &on_false).unwrap();
            let expected = reference_where(&mask, &on_true, &on_false).unwrap();
            assert_eq!(actual.dtype, expected.dtype);
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(
                actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
            );
            let input = fixture(vec![3, 1, 4], dtype);
            for target in [vec![2, 3, 5, 4], vec![3, 5, 4], vec![2, 3, 5, 7]] {
                match (
                    input.try_broadcast_to(target.clone()),
                    reference_broadcast(&input, &target),
                ) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(actual.dtype, expected.dtype);
                        assert_eq!(actual.shape, expected.shape);
                        assert_eq!(
                            actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                            expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
                        );
                    }
                    (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                    pair => panic!("different broadcast outcomes: {pair:?}"),
                }
            }
            let mask = fixture(vec![7], TensorDType::Bool);
            assert_eq!(
                PyTensor::try_where(&mask, &on_true, &on_false).unwrap_err(),
                reference_where(&mask, &on_true, &on_false).unwrap_err()
            );
        }
        // A legacy NaN mask is nonzero; unselected NaNs remain unobserved.
        let mask = fixture(vec![4, 1], TensorDType::F64);
        let on_true = fixture(vec![1, 7], TensorDType::F64);
        let on_false = fixture(vec![4, 1], TensorDType::F64);
        let actual = PyTensor::try_where(&mask, &on_true, &on_false).unwrap();
        let expected = reference_where(&mask, &on_true, &on_false).unwrap();
        assert_eq!(
            actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
            expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "complete where and unary broadcast release timing probe"]
    fn profile_selection_carry() {
        use std::time::Instant;
        let mask = fixture(vec![16, 1, 1, 1], TensorDType::Bool);
        let on_true = fixture(vec![1, 16, 16, 1], TensorDType::F32);
        let on_false = fixture(vec![1, 1, 1, 16], TensorDType::F32);
        let input = fixture(vec![1, 16, 1, 16], TensorDType::F32);
        let target = vec![16; 4];
        let modes = if std::env::var_os("QUABLA_BROADCAST_CARRY_FIRST").is_some() {
            [false, true]
        } else {
            [true, false]
        };
        for operation in ["where", "broadcast"] {
            for old in modes {
                let mut times = Vec::new();
                for _ in 0..15 {
                    let start = Instant::now();
                    let result = match (operation, old) {
                        ("where", true) => reference_where(&mask, &on_true, &on_false),
                        ("where", false) => PyTensor::try_where(&mask, &on_true, &on_false),
                        (_, true) => reference_broadcast(&input, &target),
                        (_, false) => input.try_broadcast_to(target.clone()),
                    }
                    .unwrap();
                    std::hint::black_box(result);
                    times.push(start.elapsed().as_nanos());
                }
                times.sort();
                println!(
                    "{{\"operation\":\"{operation}\",\"reference\":{old},\"median_ns\":{}}}",
                    times[7]
                );
            }
        }
    }
}
