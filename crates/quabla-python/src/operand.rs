//! The operands of tensor methods, classified once for the eager `Tensor`
//! and the tracer `TraceTensor`.
//!
//! Both classes take the same operands (a tracer, an eager `Tensor` or
//! `TensorView`, or a Python number) and report an unsupported one with the
//! same `TypeError` text, so that eager code and the `jit` of the same code
//! accept and reject alike. Each class still decides what an operand
//! becomes: eager code computes on a `Tensor` and traces against a tracer,
//! traced code captures an eager array as a constant.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::tensor::{extract_scalar, PyTensor, PyTensorView};
use crate::tensor_trace::TraceTensor;

/// The `TypeError` text of an operand that is neither an array nor a
/// Python number.
const OPERAND_ERROR: &str = "expected a Tensor, TraceTensor, or numeric scalar operand";

/// One operand of a tensor method.
pub(crate) enum Operand<'py> {
    Tracer(PyRef<'py, TraceTensor>),
    Tensor(PyRef<'py, PyTensor>),
    View(PyRef<'py, PyTensorView>),
    Scalar(f64),
}

impl<'py> Operand<'py> {
    /// An array or a Python number; `None` for anything else.
    pub(crate) fn parse(value: &Bound<'py, PyAny>) -> Option<Self> {
        Self::parse_array(value).or_else(|| extract_scalar(value).map(Self::Scalar))
    }

    /// An array (a tracer, `Tensor`, or `TensorView`); `None` for anything
    /// else, Python numbers included.
    pub(crate) fn parse_array(value: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(tracer) = value.extract::<PyRef<'py, TraceTensor>>() {
            return Some(Self::Tracer(tracer));
        }
        if let Ok(tensor) = value.extract::<PyRef<'py, PyTensor>>() {
            return Some(Self::Tensor(tensor));
        }
        value
            .extract::<PyRef<'py, PyTensorView>>()
            .ok()
            .map(Self::View)
    }
}

/// Whether a binary operator handles `value` itself rather than returning
/// `NotImplemented`: an array, never a Python number.
pub(crate) fn is_array(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<TraceTensor>()
        || value.is_instance_of::<PyTensor>()
        || value.is_instance_of::<PyTensorView>()
}

/// The `TypeError` of an operand that [`Operand::parse`] rejects.
pub(crate) fn operand_error() -> PyErr {
    PyTypeError::new_err(OPERAND_ERROR)
}

/// The `TypeError` of an operand of `op` that must be an array.
pub(crate) fn array_operand_error(op: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string());
    PyTypeError::new_err(format!(
        "{op} expects a Tensor or TraceTensor operand, got {type_name}"
    ))
}

/// The `TypeError` of the three-argument form `pow(x, y, modulo)`.
pub(crate) fn modulo_error() -> PyErr {
    PyTypeError::new_err("modulo argument is not supported for tensor power")
}
