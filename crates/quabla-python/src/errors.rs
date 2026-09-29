//! Raising the v0.2 exception classes (docs/api_v0_2_design.md, D16) from Rust.
//!
//! The classes live in `quabla/_errors.py`, so Python and Rust raise the same
//! class objects; each subclasses the builtin exception that the same failure
//! raised before v0.2, so existing `except TypeError` handlers still match.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::tensor::{PyTensor, PyTensorView};
use crate::tensor_trace::TraceTensor;

/// Why an eager `Tensor` and a traced value cannot meet in one operation.
/// Constant capture is in no v0.2 slice, so the message says how to pass the
/// array in instead.
const MIXED_OPERANDS: &str = "cannot combine an eager Tensor with a traced value: traced \
     functions do not capture arrays as constants yet; pass the array as an argument of the \
     transformed function (it becomes a graph input), or use a Python scalar";

/// A `quabla.TracerError` carrying `message`.
///
/// `TracerError` is a `TypeError`, which is also the fallback when the
/// pure-Python package cannot be imported: only a broken installation loads
/// the extension without it, and the builtin keeps the documented type.
pub(crate) fn tracer_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    let message = message.into();
    let class = py
        .import("quabla._errors")
        .and_then(|module| module.getattr("TracerError"))
        .ok()
        .and_then(|class| class.cast_into::<PyType>().ok());
    match class {
        Some(class) => PyErr::from_type(class, message),
        None => PyTypeError::new_err(message),
    }
}

/// The `TracerError` of a traced value that Python code needs as a concrete
/// value (`float(x)`, `np.asarray(x)`, `x.item()`, ...).
pub(crate) fn concrete_value_error(py: Python<'_>, what: &str) -> PyErr {
    tracer_error(
        py,
        format!(
            "{what} needs a concrete value, but a TraceTensor is symbolic inside a traced \
             function; return it from the function and convert the result, or keep computing \
             with quabla operations"
        ),
    )
}

/// The error of an eager `Tensor` operation whose `operand` is unusable: a
/// tracer is a `TracerError` that explains the mix, anything else keeps
/// `message` as a `TypeError`.
pub(crate) fn eager_operand_error(operand: &Bound<'_, PyAny>, message: &str) -> PyErr {
    if operand.is_instance_of::<TraceTensor>() {
        tracer_error(operand.py(), MIXED_OPERANDS)
    } else {
        PyTypeError::new_err(message.to_string())
    }
}

/// The error of a `TraceTensor` operation whose `operand` is unusable: an
/// eager `Tensor` or `TensorView` is a `TracerError` that explains the mix,
/// anything else keeps `message` as a `TypeError`.
pub(crate) fn traced_operand_error(operand: &Bound<'_, PyAny>, message: &str) -> PyErr {
    if operand.is_instance_of::<PyTensor>() || operand.is_instance_of::<PyTensorView>() {
        tracer_error(operand.py(), MIXED_OPERANDS)
    } else {
        PyTypeError::new_err(message.to_string())
    }
}

/// The error of a module-level function whose `operands` fit none of its
/// forms: a mix of tracers and eager arrays is a `TracerError`, anything else
/// keeps `message` as a `TypeError`.
pub(crate) fn operands_error(operands: &[&Bound<'_, PyAny>], message: &str) -> PyErr {
    let traced = operands
        .iter()
        .any(|operand| operand.is_instance_of::<TraceTensor>());
    let eager = operands.iter().any(|operand| {
        operand.is_instance_of::<PyTensor>() || operand.is_instance_of::<PyTensorView>()
    });
    match operands.first() {
        Some(first) if traced && eager => tracer_error(first.py(), MIXED_OPERANDS),
        _ => PyTypeError::new_err(message.to_string()),
    }
}
