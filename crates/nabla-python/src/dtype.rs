use nabla_core::tensor_ir::TensorDType;
use pyo3::prelude::*;

/// Element type object exposed as `nabla.float32` and `nabla.float64`.
///
/// Objects compare and hash by their dtype, so they can be used as dictionary
/// keys and compared with `==`. `str()` gives the short IR spelling
/// (`"f32"`/`"f64"`) used by `kernel_ir()` records.
#[pyclass(name = "dtype", module = "nabla", frozen, eq, hash, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyDType {
    pub(crate) dtype: TensorDType,
}

impl From<TensorDType> for PyDType {
    fn from(dtype: TensorDType) -> Self {
        Self { dtype }
    }
}

#[pymethods]
impl PyDType {
    #[getter]
    fn name(&self) -> &'static str {
        match self.dtype {
            TensorDType::F32 => "float32",
            TensorDType::F64 => "float64",
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("nabla.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.dtype.to_string()
    }
}
