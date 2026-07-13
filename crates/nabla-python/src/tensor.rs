use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for axis in (1..shape.len()).rev() {
        strides[axis - 1] = strides[axis] * shape[axis];
    }

    strides
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

    pub fn try_add(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "+", |lhs, rhs| Ok(lhs + rhs))
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "-", |lhs, rhs| Ok(lhs - rhs))
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "*", |lhs, rhs| Ok(lhs * rhs))
    }

    pub fn try_div(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise(rhs, "/", |lhs, rhs| {
            if rhs == 0.0 {
                return Err("division by zero is not supported".to_string());
            }

            Ok(lhs / rhs)
        })
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

    fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_add(rhs).map_err(PyValueError::new_err)
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        self.__add__(rhs)
    }

    fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_sub(rhs).map_err(PyValueError::new_err)
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        self.__sub__(rhs)
    }

    fn __mul__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_mul(rhs).map_err(PyValueError::new_err)
    }

    fn mul(&self, rhs: &Self) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    fn __truediv__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_div(rhs).map_err(PyValueError::new_err)
    }

    fn div(&self, rhs: &Self) -> PyResult<Self> {
        self.__truediv__(rhs)
    }

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_matmul(rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.__matmul__(rhs)
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
        Ok(PyTensor {
            shape: self.shape.clone(),
            data: Arc::new(self.to_flat_vec().map_err(PyValueError::new_err)?),
        })
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
