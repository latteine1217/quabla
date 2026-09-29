//! Raising the v0.2 exception classes (docs/api_v0_2_design.md, D16) from Rust.
//!
//! The classes live in `quabla/_errors.py`, so Python and Rust raise the same
//! class objects; each subclasses the builtin exception that the same failure
//! raised before v0.2, so existing `except TypeError` handlers still match.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyType;

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
