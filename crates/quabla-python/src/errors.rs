//! Raising the v0.2 exception classes (docs/api_v0_2_design.md, D16) from Rust.
//!
//! The classes live in `quabla/_errors.py`, so Python and Rust raise the same
//! class objects; each subclasses the builtin exception that the same failure
//! raised before v0.2, so existing `except TypeError` handlers still match.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use quabla_core::compiler::QuablaCompileError;

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

/// Marks a failure at the foreign-trace check, preserving its public class.
/// Region adapters add context from this category, never from its message.
fn mark_foreign_trace(py: Python<'_>, error: PyErr) -> PyErr {
    let category = py
        .import("quabla._errors")
        .and_then(|module| module.getattr("_BridgeFailure"))
        .and_then(|class| class.getattr("FOREIGN_TRACE"));
    // As with `tracer_error`, a broken installation keeps the builtin error.
    let Ok(category) = category else {
        return error;
    };
    match error.value(py).setattr("_quabla_failure", category) {
        Ok(()) => error,
        Err(classification_error) => classification_error,
    }
}

pub(crate) fn foreign_tracer_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    mark_foreign_trace(py, tracer_error(py, message))
}

pub(crate) fn foreign_region_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    mark_foreign_trace(py, PyValueError::new_err(message.into()))
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

/// A `quabla.UnsupportedOperationError` carrying `message` and `op`.
///
/// The class is a `ValueError`, which is also the fallback when the
/// pure-Python package cannot be imported (see [`tracer_error`]).
pub(crate) fn unsupported_operation_error(
    py: Python<'_>,
    message: impl Into<String>,
    op: &str,
) -> PyErr {
    let message = message.into();
    let error = py
        .import("quabla._errors")
        .and_then(|module| module.getattr("UnsupportedOperationError"))
        .and_then(|class| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("op", op)?;
            class.call((message.as_str(),), Some(&kwargs))
        });
    match error {
        Ok(error) => PyErr::from_value(error),
        Err(_) => PyValueError::new_err(message),
    }
}

/// A typed device lowering rejection; no message substring classification.
pub(crate) fn device_operation_error(
    py: Python<'_>,
    message: impl Into<String>,
    op: &str,
    device: &str,
) -> PyErr {
    let message = message.into();
    let error = py
        .import("quabla._errors")
        .and_then(|module| module.getattr("UnsupportedOperationError"))
        .and_then(|class| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("op", op)?;
            kwargs.set_item("device", device)?;
            class.call((message.as_str(),), Some(&kwargs))
        });
    match error {
        Ok(error) => PyErr::from_value(error),
        Err(_) => PyValueError::new_err(message),
    }
}

/// The distributed adapter keeps invalid programs and backend execution
/// failures as `ValueError`; only unavailable execution gets device metadata.
pub(crate) fn data_parallel_compile_error(py: Python<'_>, error: QuablaCompileError) -> PyErr {
    match error {
        QuablaCompileError::Unavailable(message) => {
            device_operation_error(py, message, "distributed.value_and_grad", "cuda")
        }
        QuablaCompileError::Unsupported { op, message } => {
            device_operation_error(py, message, &op, "cuda")
        }
        QuablaCompileError::InvalidProgram(message) | QuablaCompileError::Backend(message) => {
            PyValueError::new_err(message)
        }
    }
}
