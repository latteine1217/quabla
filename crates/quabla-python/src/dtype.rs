// PyO3's `#[pyclass(from_py_object)]` expansion clones the extracted value,
// and `PyDType` is `Copy`, so clippy >= 1.99 reports `clone_on_copy` inside
// generated code that this module cannot change.
#![allow(clippy::clone_on_copy)]

use pyo3::prelude::*;
use quabla_core::tensor_ir::TensorDType;

/// Element type object exposed as `quabla.float32`, `quabla.float64`, and
/// `quabla.bool_`.
///
/// Objects compare and hash by their dtype, so they can be used as dictionary
/// keys and compared with `==`. `str()` gives the short IR spelling
/// (`"f32"`/`"f64"`/`"bool"`) used by `kernel_ir()` records.
#[pyclass(name = "dtype", module = "quabla", frozen, eq, hash, from_py_object)]
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
    pub(crate) fn name(&self) -> &'static str {
        match self.dtype {
            TensorDType::F32 => "float32",
            TensorDType::F64 => "float64",
            TensorDType::Bool => "bool",
        }
    }

    /// The module attribute holding this dtype (`bool_` avoids shadowing the
    /// Python builtin, as in NumPy).
    pub(crate) fn __repr__(&self) -> String {
        match self.dtype {
            TensorDType::Bool => "quabla.bool_".to_string(),
            _ => format!("quabla.{}", self.name()),
        }
    }

    fn __str__(&self) -> String {
        self.dtype.to_string()
    }
}
