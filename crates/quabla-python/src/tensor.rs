use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyMemoryView, PySlice, PySliceMethods, PyTuple};
use quabla_core::tensor_ir::{TensorComparison, TensorDType};
use std::ffi::c_int;
use std::sync::Arc;

use crate::dtype::PyDType;
use crate::interop;

/// Eager host tensor. Storage is `f64`; a `float32` tensor holds only values
/// rounded to `f32`, and every eager op on it rounds its `f64` result the way
/// the CPU Tensor IR backend rounds an `f32` node. Python scalars are weak:
/// they are rounded to the tensor dtype before the op. A `bool` tensor holds
/// `0.0`/`1.0` and follows the Tensor IR promotion rule.
#[pyclass(name = "Tensor", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTensor {
    shape: Vec<usize>,
    data: Arc<Vec<f64>>,
    dtype: TensorDType,
    /// JAX-style weak type: set only when a `bool` tensor meets a Python
    /// scalar (`mask * 2.0`), so the result still adopts a later strong
    /// operand's dtype. Elementwise arithmetic propagates it.
    weak: bool,
}

#[pyclass(name = "TensorView", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTensorView {
    data: Arc<Vec<f64>>,
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
            data: Arc::new(data),
            dtype: TensorDType::F64,
            weak: false,
        })
    }

    /// A weak `f64` scalar, the eager form of a Python scalar operand.
    pub fn weak_scalar(value: f64) -> Self {
        Self {
            shape: vec![],
            data: Arc::new(vec![value]),
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

    pub fn shape_data(&self) -> (&[usize], &[f64]) {
        (&self.shape, self.data.as_ref())
    }

    pub fn to_dynamic_tensor(&self) -> Result<quabla_core::tensor_ir::DynamicTensor, String> {
        quabla_core::tensor_ir::DynamicTensor::with_dtype(
            self.shape.clone(),
            self.data.as_ref().clone(),
            self.dtype,
        )
    }

    pub fn from_dynamic_tensor(
        tensor: quabla_core::tensor_ir::DynamicTensor,
    ) -> Result<Self, String> {
        Self::from_shape_data_typed(
            tensor.shape().to_vec(),
            tensor.data().to_vec(),
            tensor.dtype(),
        )
    }

    /// Rounds the values to `dtype` and records it as a strong dtype (exact
    /// for widening; nonzero and `NaN` become `1.0` for `bool`).
    fn typed(mut self, dtype: TensorDType) -> Self {
        if dtype != TensorDType::F64 {
            for value in Arc::make_mut(&mut self.data) {
                *value = dtype.round(*value);
            }
        }
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
        let lhs_strides = contiguous_strides(&self.shape);
        let rhs_strides = contiguous_strides(&rhs.shape);
        let mut data = Vec::with_capacity(output_size);

        for index in 0..output_size {
            let lhs_index = broadcast_offset(index, &shape, &self.shape, &lhs_strides);
            let rhs_index = broadcast_offset(index, &shape, &rhs.shape, &rhs_strides);
            let value = f(self.data[lhs_index], rhs.data[rhs_index])
                .map_err(|err| format!("failed to evaluate tensor {op}: {err}"))?;
            data.push(dtype.round(value));
        }

        Ok(Self {
            shape,
            data: Arc::new(data),
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
        let mut output = Self::from_shape_data_typed(
            self.shape.clone(),
            self.data.iter().copied().map(f).collect(),
            self.dtype,
        )?;
        output.weak = self.weak;
        Ok(output)
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
                .map(|lhs| f64::from(kind.evaluate(*lhs, rhs)))
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
                .map(|value| f64::from(test(*value)))
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
        match self.data.as_slice() {
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
        if base.data.contains(&0.0) {
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

    pub fn try_maximum(&self, rhs: &Self) -> Result<Self, String> {
        let mask = self.try_gt(rhs)?;
        Self::try_where(&mask, self, rhs)
    }

    pub fn try_maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| if lhs > rhs { lhs } else { rhs })
    }

    pub fn try_minimum(&self, rhs: &Self) -> Result<Self, String> {
        let mask = rhs.try_gt(self)?;
        Self::try_where(&mask, self, rhs)
    }

    pub fn try_minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let base = self.arithmetic_base();
        let rhs = base.dtype.round(rhs);
        base.try_map(|lhs| if rhs > lhs { lhs } else { rhs })
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

        for index in 0..count {
            let mask_index = broadcast_offset(index, &shape, &mask.shape, &mask_strides);
            let true_index = broadcast_offset(index, &shape, &on_true.shape, &true_strides);
            let false_index = broadcast_offset(index, &shape, &on_false.shape, &false_strides);
            data.push(if mask.data[mask_index] != 0.0 {
                on_true.data[true_index]
            } else {
                on_false.data[false_index]
            });
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

            for row in 0..lhs_rows {
                for col in 0..rhs_cols {
                    let mut value = 0.0;
                    for inner in 0..lhs_inner {
                        let lhs_index = lhs_batch * lhs_rows * lhs_inner + row * lhs_inner + inner;
                        let rhs_index = rhs_batch * rhs_inner * rhs_cols + inner * rhs_cols + col;
                        value += self.data[lhs_index] * rhs.data[rhs_index];
                    }

                    data[batch_index * lhs_rows * rhs_cols + row * rhs_cols + col] = value;
                }
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
        let columns = rhs.shape[1];
        let mut factor = self.data.as_ref().clone();
        let mut result = rhs.data.as_ref().clone();
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
                let symmetric = self.data[column * n + row];
                let value = self.data[row * n + column];
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
        let data = (0..count)
            .map(|index| self.data[broadcast_offset(index, &shape, &self.shape, &strides)])
            .collect();
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
            *output_value = self.data[input_index];
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
        let output_strides = contiguous_strides(&shape);
        let mut data = vec![0.0; element_count(&shape)?];

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
            data[output_index] += value * scale;
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
            let value = self
                .data
                .iter()
                .copied()
                .fold(self.data[0], |current, value| {
                    if (maximum && value >= current) || (!maximum && value <= current) {
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

        for (source_index, value) in self.data.iter().copied().enumerate() {
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
                Some(current)
                    if !((maximum && value >= current) || (!maximum && value <= current)) => {}
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

    pub fn try_sqrt(&self) -> Result<Self, String> {
        self.try_unary("sqrt", f64::sqrt)
    }

    pub fn try_relu(&self) -> Result<Self, String> {
        self.ensure_not_bool("relu")?;
        self.try_maximum_scalar(0.0)
    }

    pub fn try_abs(&self) -> Result<Self, String> {
        self.ensure_not_bool("abs")?;
        let mask = self.try_gt_scalar(0.0)?;
        let negative = self.try_mul_scalar(-1.0)?;
        Self::try_where(&mask, self, &negative)
    }

    pub fn try_sigmoid(&self) -> Result<Self, String> {
        self.try_unary("sigmoid", |value| 1.0 / (1.0 + (-value).exp()))
    }

    pub fn try_softplus(&self) -> Result<Self, String> {
        self.try_unary("softplus", |value| {
            value.max(0.0) + (-value.abs()).exp().ln_1p()
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
                    *value
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
                data.extend_from_slice(&tensor.data[start..start + block]);
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
        let mut output = self.clone();
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
        let gathered = indices
            .iter()
            .map(|index| self.try_slice(axis, *index, 1, 1)?.materialize())
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_concat(&gathered, axis)
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
        let mut data = self.data.as_ref().clone();
        for (source_index, value) in updates.data.iter().copied().enumerate() {
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
            data: Arc::new(vec![0.0; size]),
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
            data: Arc::new(vec![value; size]),
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

    fn to_flat_list(&self) -> Vec<f64> {
        self.data.as_ref().clone()
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
        interop::nested_list(py, &self.shape, &self.data, self.dtype)
    }

    /// The single element as a Python `float` (or `bool` for `bool`
    /// tensors); any shape with exactly one element is accepted.
    fn item<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        interop::single_item(py, &self.shape, &self.data, self.dtype)
    }

    fn __float__(&self) -> PyResult<f64> {
        match self.data.as_slice() {
            [value] => Ok(*value),
            _ => Err(PyTypeError::new_err(format!(
                "only single-element tensors can be converted to a Python float, got shape {:?}",
                self.shape
            ))),
        }
    }

    /// A read-only NumPy array over the buffer export: zero-copy for
    /// `float64`, a converted copy for `float32` and `bool`. Imports NumPy
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

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_add(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_add_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(lhs)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_sub(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_sub_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__sub__(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lhs) = lhs.extract::<PyRef<'_, PyTensor>>() {
            return lhs.try_sub(self).map_err(PyValueError::new_err);
        }
        if let Ok(lhs) = lhs.extract::<f64>() {
            return self.try_scalar_sub(lhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_mul(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_mul_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(lhs)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_div(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_div_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar divisor",
        ))
    }

    fn div(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__truediv__(rhs)
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(lhs) = lhs.extract::<PyRef<'_, PyTensor>>() {
            return lhs.try_div(self).map_err(PyValueError::new_err);
        }
        if let Ok(lhs) = lhs.extract::<f64>() {
            return self.try_scalar_div(lhs).map_err(PyValueError::new_err);
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
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for Tensor power",
            ));
        }
        if let Ok(exponent) = exponent.extract::<u32>() {
            return self.try_powi(exponent).map_err(PyValueError::new_err);
        }
        // A tensor exponent, including a single-element one, is an elementwise
        // operand that broadcasts, as in NumPy and in traced code; it is never
        // unwrapped into a scalar.
        if let Ok(exponent) = exponent.extract::<PyRef<'_, PyTensor>>() {
            return self.try_pow(&exponent).map_err(PyValueError::new_err);
        }
        let exponent = extract_scalar(exponent)
            .ok_or_else(|| PyTypeError::new_err("expected a Tensor or numeric scalar exponent"))?;
        self.try_powf(exponent).map_err(PyValueError::new_err)
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

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_gt(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_gt_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    /// Elementwise `self > rhs` as a `bool` tensor (unlike `gt`, which keeps
    /// returning a 0/1 float mask).
    fn greater(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare_operand(rhs, TensorComparison::Greater)
    }

    fn greater_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare_operand(rhs, TensorComparison::GreaterEqual)
    }

    fn less(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare_operand(rhs, TensorComparison::Less)
    }

    fn less_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare_operand(rhs, TensorComparison::LessEqual)
    }

    fn equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare_operand(rhs, TensorComparison::Equal)
    }

    fn not_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare_operand(rhs, TensorComparison::NotEqual)
    }

    // `==`/`!=` are deliberately not overloaded: tensors keep identity
    // equality and stay hashable; use `equal`/`not_equal` for elementwise.
    fn __gt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.greater(rhs)
    }

    fn __ge__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.greater_equal(rhs)
    }

    fn __lt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.less(rhs)
    }

    fn __le__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.less_equal(rhs)
    }

    /// Defining ordering operators installs a rich comparison, which drops
    /// Python's default identity hash; restore it so tensors stay usable as
    /// dictionary keys.
    fn __hash__(slf: &Bound<'_, Self>) -> isize {
        slf.as_ptr() as isize
    }

    fn logical_and(&self, rhs: &Self) -> PyResult<Self> {
        self.try_logical(rhs, true).map_err(PyValueError::new_err)
    }

    fn logical_or(&self, rhs: &Self) -> PyResult<Self> {
        self.try_logical(rhs, false).map_err(PyValueError::new_err)
    }

    fn logical_not(&self) -> PyResult<Self> {
        self.try_logical_not().map_err(PyValueError::new_err)
    }

    fn __and__(&self, rhs: &Self) -> PyResult<Self> {
        self.logical_and(rhs)
    }

    fn __or__(&self, rhs: &Self) -> PyResult<Self> {
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

    fn maximum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_maximum(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_maximum_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn minimum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_minimum(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.try_minimum_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
    }

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_matmul(rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.__matmul__(rhs)
    }

    fn solve(&self, rhs: &Self) -> PyResult<Self> {
        self.try_solve(rhs).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (rhs, lower = true, transpose = false))]
    fn solve_triangular(&self, rhs: &Self, lower: bool, transpose: bool) -> PyResult<Self> {
        self.try_solve_triangular(rhs, lower, transpose)
            .map_err(PyValueError::new_err)
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
    fn compare_operand(&self, rhs: &Bound<'_, PyAny>, kind: TensorComparison) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, PyTensor>>() {
            return self.try_compare(&rhs, kind).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self
                .try_compare_scalar(rhs, kind)
                .map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected Tensor or numeric scalar operand",
        ))
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
            .map(|index| self.data[view_offset(index, &self.shape, &self.strides, self.offset)])
            .collect())
    }

    fn materialize(&self) -> Result<PyTensor, String> {
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
