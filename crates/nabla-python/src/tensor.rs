use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySlice, PySliceMethods, PyTuple};
use std::sync::Arc;

#[pyclass(name = "Tensor", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTensor {
    shape: Vec<usize>,
    data: Arc<Vec<f64>>,
}

#[pyclass(name = "TensorView", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTensorView {
    data: Arc<Vec<f64>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
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
        })
    }

    pub fn shape_data(&self) -> (&[usize], &[f64]) {
        (&self.shape, self.data.as_ref())
    }

    pub fn to_dynamic_tensor(&self) -> Result<nabla_core::tensor_ir::DynamicTensor, String> {
        nabla_core::tensor_ir::DynamicTensor::new(self.shape.clone(), self.data.as_ref().clone())
    }

    pub fn from_dynamic_tensor(
        tensor: nabla_core::tensor_ir::DynamicTensor,
    ) -> Result<Self, String> {
        Self::from_shape_data(tensor.shape().to_vec(), tensor.data().to_vec())
    }

    fn try_elementwise(
        &self,
        rhs: &Self,
        op: &str,
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
            data.push(value);
        }

        Ok(Self {
            shape,
            data: Arc::new(data),
        })
    }

    fn try_map(&self, f: impl Fn(f64) -> f64) -> Result<Self, String> {
        Self::from_shape_data(
            self.shape.clone(),
            self.data.iter().copied().map(f).collect(),
        )
    }

    pub fn try_add(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "+", |lhs, rhs| Ok(lhs + rhs))
    }

    pub fn try_add_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_map(|lhs| lhs + rhs)
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "-", |lhs, rhs| Ok(lhs - rhs))
    }

    pub fn try_sub_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_map(|lhs| lhs - rhs)
    }

    pub fn try_scalar_sub(&self, lhs: f64) -> Result<Self, String> {
        self.try_map(|rhs| lhs - rhs)
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "*", |lhs, rhs| Ok(lhs * rhs))
    }

    pub fn try_mul_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_map(|lhs| lhs * rhs)
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
        if rhs == 0.0 {
            return Err("division by zero scalar is not supported".to_string());
        }
        self.try_map(|lhs| lhs / rhs)
    }

    pub fn try_scalar_div(&self, lhs: f64) -> Result<Self, String> {
        if self.data.contains(&0.0) {
            return Err("division by zero is not supported".to_string());
        }
        self.try_map(|rhs| lhs / rhs)
    }

    pub fn try_gt(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "gt", |lhs, rhs| Ok(if lhs > rhs { 1.0 } else { 0.0 }))
    }

    pub fn try_gt_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_map(|lhs| if lhs > rhs { 1.0 } else { 0.0 })
    }

    pub fn try_maximum(&self, rhs: &Self) -> Result<Self, String> {
        let mask = self.try_gt(rhs)?;
        Self::try_where(&mask, self, rhs)
    }

    pub fn try_maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_map(|lhs| if lhs > rhs { lhs } else { rhs })
    }

    pub fn try_minimum(&self, rhs: &Self) -> Result<Self, String> {
        let mask = rhs.try_gt(self)?;
        Self::try_where(&mask, self, rhs)
    }

    pub fn try_minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_map(|lhs| if rhs > lhs { lhs } else { rhs })
    }

    pub fn try_where(mask: &Self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
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

        Self::from_shape_data(shape, data)
    }

    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, String> {
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

        Ok(Self {
            shape,
            data: Arc::new(data),
        })
    }

    pub fn try_solve(&self, rhs: &Self) -> Result<Self, String> {
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
        Ok(Self {
            shape: rhs.shape.clone(),
            data: Arc::new(result),
        })
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
        Self::from_shape_data(shape, data)
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

        Self::from_shape_data(shape, data)
    }

    fn try_reduce(&self, axis: Option<isize>, scale: f64) -> Result<Self, String> {
        let Some(axis) = axis else {
            return Self::from_shape_data(vec![], vec![self.data.iter().sum::<f64>() * scale]);
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

        Self::from_shape_data(shape, data)
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
        self.try_powi(2)?.try_sum_axes(axes, keepdims)?.try_sqrt()
    }

    pub fn try_max_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.try_extrema_axes(axes, keepdims, true)
    }

    pub fn try_min_axes(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
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
            return Self::from_shape_data(vec![], vec![value]);
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

        Self::from_shape_data(
            shape,
            data.into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "max/min reduction produced an empty output".to_string())?,
        )
    }

    pub fn try_tanh(&self) -> Result<Self, String> {
        self.try_map(f64::tanh)
    }

    pub fn try_exp(&self) -> Result<Self, String> {
        self.try_map(f64::exp)
    }

    pub fn try_log(&self) -> Result<Self, String> {
        self.try_map(f64::ln)
    }

    pub fn try_sqrt(&self) -> Result<Self, String> {
        self.try_map(f64::sqrt)
    }

    pub fn try_relu(&self) -> Result<Self, String> {
        self.try_maximum_scalar(0.0)
    }

    pub fn try_abs(&self) -> Result<Self, String> {
        let mask = self.try_gt_scalar(0.0)?;
        let negative = self.try_mul_scalar(-1.0)?;
        Self::try_where(&mask, self, &negative)
    }

    pub fn try_sigmoid(&self) -> Result<Self, String> {
        self.try_map(|value| 1.0 / (1.0 + (-value).exp()))
    }

    pub fn try_softplus(&self) -> Result<Self, String> {
        self.try_map(|value| value.max(0.0) + (-value.abs()).exp().ln_1p())
    }

    pub fn try_triangular(&self, lower: bool) -> Result<Self, String> {
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
        Self::from_shape_data(self.shape.clone(), data)
    }

    pub fn try_sin(&self) -> Result<Self, String> {
        self.try_map(f64::sin)
    }

    pub fn try_cos(&self) -> Result<Self, String> {
        self.try_map(f64::cos)
    }

    pub fn try_powi(&self, exponent: u32) -> Result<Self, String> {
        self.try_map(|value| value.powf(exponent as f64))
    }

    pub fn try_powf(&self, exponent: f64) -> Result<Self, String> {
        self.try_map(|value| value.powf(exponent))
    }

    pub fn try_concat(tensors: &[PyTensor], axis: usize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "concat requires at least one tensor".to_string())?;
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
        Self::from_shape_data(shape, data)
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
        Self::from_shape_data(self.shape.clone(), data)
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn py_new(shape: Vec<usize>, data: Vec<f64>) -> PyResult<Self> {
        Self::from_shape_data(shape, data).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let size = element_count(&shape).map_err(PyValueError::new_err)?;

        Ok(Self {
            shape,
            data: Arc::new(vec![0.0; size]),
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

    fn to_flat_list(&self) -> Vec<f64> {
        self.data.as_ref().clone()
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
        self.try_mul_scalar(-1.0).map_err(PyValueError::new_err)
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
        let exponent = exponent
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err("expected numeric scalar exponent"))?;
        self.try_powf(exponent).map_err(PyValueError::new_err)
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

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?})", self.shape)
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
        })
    }

    fn to_flat_vec(&self) -> Result<Vec<f64>, String> {
        let count = element_count(&self.shape)?;
        Ok((0..count)
            .map(|index| self.data[view_offset(index, &self.shape, &self.strides, self.offset)])
            .collect())
    }

    fn materialize(&self) -> Result<PyTensor, String> {
        PyTensor::from_shape_data(self.shape.clone(), self.to_flat_vec()?)
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
