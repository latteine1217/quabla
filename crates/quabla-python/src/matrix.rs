use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass(name = "Matrix", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

fn broadcast_extent(lhs: usize, rhs: usize) -> Option<usize> {
    if lhs == rhs {
        Some(lhs)
    } else if lhs == 1 {
        Some(rhs)
    } else if rhs == 1 {
        Some(lhs)
    } else {
        None
    }
}

impl PyMatrix {
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self, String> {
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);

        if rows.iter().any(|row| row.len() != col_count) {
            return Err("ragged matrix rows are not supported".to_string());
        }

        let mut data = Vec::with_capacity(row_count * col_count);
        for row in rows {
            data.extend(row);
        }

        Ok(Self {
            rows: row_count,
            cols: col_count,
            data,
        })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|row| {
                let start = row * self.cols;
                let end = start + self.cols;
                self.data[start..end].to_vec()
            })
            .collect()
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn sum(&self) -> Self {
        Self::filled(1, 1, self.data.iter().sum())
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Self, String> {
        match axis {
            0 => {
                let mut data = vec![0.0; self.cols];
                for row in 0..self.rows {
                    for (col, value) in data.iter_mut().enumerate() {
                        *value += self.data[row * self.cols + col];
                    }
                }

                Ok(Self {
                    rows: 1,
                    cols: self.cols,
                    data,
                })
            }
            1 => {
                let data = (0..self.rows)
                    .map(|row| {
                        let start = row * self.cols;
                        let end = start + self.cols;
                        self.data[start..end].iter().sum()
                    })
                    .collect();

                Ok(Self {
                    rows: self.rows,
                    cols: 1,
                    data,
                })
            }
            _ => Err(format!("axis must be 0 or 1, got {axis}")),
        }
    }

    pub fn mean(&self) -> Self {
        Self::filled(1, 1, self.data.iter().sum::<f64>() / self.data.len() as f64)
    }

    pub fn mean_axis(&self, axis: usize) -> Result<Self, String> {
        let divisor = match axis {
            0 => self.rows as f64,
            1 => self.cols as f64,
            _ => return Err(format!("axis must be 0 or 1, got {axis}")),
        };

        self.sum_axis(axis)?.try_div_scalar(divisor)
    }

    pub fn elementwise_tanh(&self) -> Self {
        let data = self.data.iter().map(|value| value.tanh()).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_exp(&self) -> Self {
        let data = self.data.iter().map(|value| value.exp()).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_log(&self) -> Self {
        let data = self.data.iter().map(|value| value.ln()).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_sqrt(&self) -> Self {
        let data = self.data.iter().map(|value| value.sqrt()).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_powf(&self, exponent: f64) -> Self {
        let data = self.data.iter().map(|value| value.powf(exponent)).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_reciprocal(&self) -> Self {
        let data = self.data.iter().map(|value| 1.0 / value).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_sin(&self) -> Self {
        let data = self.data.iter().map(|value| value.sin()).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn elementwise_cos(&self) -> Self {
        let data = self.data.iter().map(|value| value.cos()).collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0; self.rows * self.cols];

        for row in 0..self.rows {
            for col in 0..self.cols {
                data[col * self.rows + row] = self.data[row * self.cols + col];
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    pub fn try_reshape(&self, rows: usize, cols: usize) -> Result<Self, String> {
        if rows * cols != self.data.len() {
            return Err(format!(
                "cannot reshape matrix with {} elements to ({rows}, {cols})",
                self.data.len()
            ));
        }

        Ok(Self {
            rows,
            cols,
            data: self.data.clone(),
        })
    }

    pub fn try_concat(matrices: &[Self], axis: usize) -> Result<Self, String> {
        let first = matrices
            .first()
            .ok_or_else(|| "concat requires at least one matrix".to_string())?;

        match axis {
            0 => {
                if matrices.iter().any(|matrix| matrix.cols != first.cols) {
                    return Err("concat axis 0 requires matching column counts".to_string());
                }

                let rows = matrices.iter().map(|matrix| matrix.rows).sum();
                let cols = first.cols;
                let mut data = Vec::with_capacity(rows * cols);
                for matrix in matrices {
                    data.extend_from_slice(matrix.data());
                }

                Ok(Self { rows, cols, data })
            }
            1 => {
                if matrices.iter().any(|matrix| matrix.rows != first.rows) {
                    return Err("concat axis 1 requires matching row counts".to_string());
                }

                let rows = first.rows;
                let cols = matrices.iter().map(|matrix| matrix.cols).sum();
                let mut data = Vec::with_capacity(rows * cols);
                for row in 0..rows {
                    for matrix in matrices {
                        let start = row * matrix.cols;
                        let end = start + matrix.cols;
                        data.extend_from_slice(&matrix.data[start..end]);
                    }
                }

                Ok(Self { rows, cols, data })
            }
            _ => Err(format!("axis must be 0 or 1, got {axis}")),
        }
    }

    pub fn try_slice_rows(&self, start: usize, rows: usize) -> Result<Self, String> {
        let end = start + rows;
        if end > self.rows {
            return Err(format!(
                "row slice [{start}, {end}) is out of bounds for {} rows",
                self.rows
            ));
        }

        let mut data = Vec::with_capacity(rows * self.cols);
        for row in start..end {
            let row_start = row * self.cols;
            let row_end = row_start + self.cols;
            data.extend_from_slice(&self.data[row_start..row_end]);
        }

        Ok(Self {
            rows,
            cols: self.cols,
            data,
        })
    }

    pub fn try_slice_cols(&self, start: usize, cols: usize) -> Result<Self, String> {
        let end = start + cols;
        if end > self.cols {
            return Err(format!(
                "column slice [{start}, {end}) is out of bounds for {} columns",
                self.cols
            ));
        }

        let mut data = Vec::with_capacity(self.rows * cols);
        for row in 0..self.rows {
            let row_start = row * self.cols;
            data.extend_from_slice(&self.data[row_start + start..row_start + end]);
        }

        Ok(Self {
            rows: self.rows,
            cols,
            data,
        })
    }

    fn elementwise_broadcast_shape(&self, rhs: &Self, op: &str) -> Result<(usize, usize), String> {
        let rows = broadcast_extent(self.rows, rhs.rows).ok_or_else(|| {
            format!(
                "incompatible {op} shapes: ({}, {}) {op} ({}, {})",
                self.rows, self.cols, rhs.rows, rhs.cols
            )
        })?;
        let cols = broadcast_extent(self.cols, rhs.cols).ok_or_else(|| {
            format!(
                "incompatible {op} shapes: ({}, {}) {op} ({}, {})",
                self.rows, self.cols, rhs.rows, rhs.cols
            )
        })?;

        Ok((rows, cols))
    }

    fn broadcast_value_at(&self, row: usize, col: usize) -> f64 {
        let row = if self.rows == 1 { 0 } else { row };
        let col = if self.cols == 1 { 0 } else { col };

        self.data[row * self.cols + col]
    }

    fn try_elementwise_broadcast(
        &self,
        rhs: &Self,
        op: &str,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Self, String> {
        let (rows, cols) = self.elementwise_broadcast_shape(rhs, op)?;
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                data.push(f(
                    self.broadcast_value_at(row, col),
                    rhs.broadcast_value_at(row, col),
                ));
            }
        }

        Ok(Self { rows, cols, data })
    }

    pub fn try_add(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise_broadcast(rhs, "+", |lhs, rhs| lhs + rhs)
    }

    pub fn try_add_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_add(&Self::filled(self.rows, self.cols, rhs))
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise_broadcast(rhs, "-", |lhs, rhs| lhs - rhs)
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise_broadcast(rhs, "*", |lhs, rhs| lhs * rhs)
    }

    pub fn try_div(&self, rhs: &Self) -> Result<Self, String> {
        let (rows, cols) = self.elementwise_broadcast_shape(rhs, "/")?;
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                let rhs_value = rhs.broadcast_value_at(row, col);
                if rhs_value == 0.0 {
                    return Err("division by zero is not supported".to_string());
                }

                data.push(self.broadcast_value_at(row, col) / rhs_value);
            }
        }

        Ok(Self { rows, cols, data })
    }

    pub fn try_gt(&self, rhs: &Self) -> Result<Self, String> {
        self.try_elementwise_broadcast(rhs, "gt", |lhs, rhs| if lhs > rhs { 1.0 } else { 0.0 })
    }

    pub fn try_where(mask: &Self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        let (value_rows, value_cols) = on_true.elementwise_broadcast_shape(on_false, "where")?;
        let value_shape = Self::filled(value_rows, value_cols, 0.0);
        let (rows, cols) = mask.elementwise_broadcast_shape(&value_shape, "where")?;
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                if mask.broadcast_value_at(row, col) != 0.0 {
                    data.push(on_true.broadcast_value_at(row, col));
                } else {
                    data.push(on_false.broadcast_value_at(row, col));
                }
            }
        }

        Ok(Self { rows, cols, data })
    }

    pub fn try_mul_scalar(&self, rhs: f64) -> Result<Self, String> {
        self.try_mul(&Self::filled(self.rows, self.cols, rhs))
    }

    pub fn try_div_scalar(&self, rhs: f64) -> Result<Self, String> {
        if rhs == 0.0 {
            return Err("division by zero scalar is not supported".to_string());
        }

        self.try_mul_scalar(1.0 / rhs)
    }

    pub fn try_powi(&self, exponent: u32) -> Result<Self, String> {
        if exponent == 0 {
            return Ok(Self::filled(self.rows, self.cols, 1.0));
        }

        let mut output = self.clone();
        for _ in 1..exponent {
            output = output.try_mul(self)?;
        }

        Ok(output)
    }

    pub fn try_powf(&self, exponent: f64) -> Result<Self, String> {
        Ok(self.elementwise_powf(exponent))
    }

    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, String> {
        if self.cols != rhs.rows {
            return Err(format!(
                "incompatible matmul shapes: ({}, {}) x ({}, {})",
                self.rows, self.cols, rhs.rows, rhs.cols
            ));
        }

        let mut data = vec![0.0; self.rows * rhs.cols];

        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let mut sum = 0.0;

                for inner in 0..self.cols {
                    sum += self.data[row * self.cols + inner] * rhs.data[inner * rhs.cols + col];
                }

                data[row * rhs.cols + col] = sum;
            }
        }

        Ok(Self {
            rows: self.rows,
            cols: rhs.cols,
            data,
        })
    }
}

fn matrix_or_scalar_operand(lhs: &PyMatrix, rhs: &Bound<'_, PyAny>) -> PyResult<PyMatrix> {
    if let Ok(rhs) = rhs.extract::<PyRef<'_, PyMatrix>>() {
        return Ok(rhs.clone());
    }

    if let Ok(value) = rhs.extract::<f64>() {
        return Ok(PyMatrix::filled(lhs.rows, lhs.cols, value));
    }

    Err(PyTypeError::new_err(
        "expected Matrix or numeric scalar operand",
    ))
}

#[pymethods]
impl PyMatrix {
    #[new]
    fn py_new(rows: Vec<Vec<f64>>) -> PyResult<Self> {
        Self::from_rows(rows).map_err(PyValueError::new_err)
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dims()
    }

    fn to_list(&self) -> Vec<Vec<f64>> {
        self.to_rows()
    }

    #[pyo3(name = "sum")]
    #[pyo3(signature = (axis=None))]
    fn py_sum(&self, axis: Option<usize>) -> PyResult<Self> {
        match axis {
            Some(axis) => self.sum_axis(axis).map_err(PyValueError::new_err),
            None => Ok(PyMatrix::sum(self)),
        }
    }

    #[pyo3(name = "mean")]
    #[pyo3(signature = (axis=None))]
    fn py_mean(&self, axis: Option<usize>) -> PyResult<Self> {
        match axis {
            Some(axis) => self.mean_axis(axis).map_err(PyValueError::new_err),
            None => Ok(PyMatrix::mean(self)),
        }
    }

    fn tanh(&self) -> Self {
        PyMatrix::elementwise_tanh(self)
    }

    fn exp(&self) -> Self {
        PyMatrix::elementwise_exp(self)
    }

    fn log(&self) -> Self {
        PyMatrix::elementwise_log(self)
    }

    fn sqrt(&self) -> Self {
        PyMatrix::elementwise_sqrt(self)
    }

    fn sin(&self) -> Self {
        PyMatrix::elementwise_sin(self)
    }

    fn cos(&self) -> Self {
        PyMatrix::elementwise_cos(self)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = matrix_or_scalar_operand(self, rhs)?;
        self.try_gt(&rhs).map_err(PyValueError::new_err)
    }

    #[getter(T)]
    fn py_t(&self) -> Self {
        self.transpose()
    }

    #[pyo3(name = "transpose")]
    fn py_transpose(&self) -> Self {
        PyMatrix::transpose(self)
    }

    fn reshape(&self, rows: usize, cols: usize) -> PyResult<Self> {
        self.try_reshape(rows, cols).map_err(PyValueError::new_err)
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = matrix_or_scalar_operand(self, rhs)?;
        self.try_add(&rhs).map_err(PyValueError::new_err)
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(lhs)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = matrix_or_scalar_operand(self, rhs)?;
        self.try_sub(&rhs).map_err(PyValueError::new_err)
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__sub__(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = lhs.extract::<f64>() {
            return PyMatrix::filled(self.rows, self.cols, value)
                .try_sub(self)
                .map_err(PyValueError::new_err);
        }

        Err(PyTypeError::new_err("expected numeric scalar operand"))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = matrix_or_scalar_operand(self, rhs)?;
        self.try_mul(&rhs).map_err(PyValueError::new_err)
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(lhs)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.try_mul_scalar(-1.0).map_err(PyValueError::new_err)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = rhs.extract::<f64>() {
            return self.try_div_scalar(value).map_err(PyValueError::new_err);
        }

        let rhs = rhs
            .extract::<PyRef<'_, PyMatrix>>()
            .map_err(|_| PyTypeError::new_err("expected Matrix or numeric scalar divisor"))?;

        self.try_div(&rhs).map_err(PyValueError::new_err)
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for Matrix power",
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

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_matmul(rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.try_matmul(rhs).map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!("Matrix(shape=({}, {}))", self.rows, self.cols)
    }
}
