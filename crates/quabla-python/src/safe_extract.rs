//! Bounded sequence extraction before PyO3 can reserve an unchecked length.

use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Checks a native vector layout and returns allocation failure as a Python error.
pub(crate) fn buffer<T>(count: usize) -> PyResult<Vec<T>> {
    let capacity = isize::MAX as usize / std::mem::size_of::<T>().max(1);
    if count > capacity {
        return Err(PyValueError::new_err(
            "sequence length exceeds addressable memory",
        ));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|error| {
        PyMemoryError::new_err(format!(
            "cannot allocate sequence of {count} elements: {error}"
        ))
    })?;
    Ok(output)
}

pub(crate) fn vec<'py, T: FromPyObjectOwned<'py>>(value: &Bound<'py, PyAny>) -> PyResult<Vec<T>> {
    if value.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("expected a sequence, not str"));
    }
    // SAFETY: the Bound keeps a live object in the attached interpreter.
    // CPython only queries its sequence slots; this matches PyO3's Vec check
    // without requiring registration with collections.abc.Sequence.
    if unsafe { pyo3::ffi::PySequence_Check(value.as_ptr()) } == 0 {
        return Err(PyTypeError::new_err("expected a sequence"));
    }
    let count = match value.len() {
        Ok(count) => Some(count),
        // PyO3 also accepts the finite __getitem__ protocol without __len__.
        Err(error) if error.is_instance_of::<PyTypeError>(value.py()) => None,
        Err(error) => return Err(error),
    };
    let mut output = buffer(count.unwrap_or(0))?;
    for item in value.try_iter()? {
        let item = item?;
        if count.is_some_and(|count| output.len() == count) {
            return Err(PyValueError::new_err(
                "sequence length changed during conversion",
            ));
        }
        if output.len() == output.capacity() {
            output.try_reserve(1).map_err(|error| {
                PyMemoryError::new_err(format!("cannot grow sequence: {error}"))
            })?;
        }
        output.push(item.extract::<T>().map_err(Into::into)?);
    }
    if count.is_some_and(|count| output.len() != count) {
        return Err(PyValueError::new_err(
            "sequence length changed during conversion",
        ));
    }
    Ok(output)
}

pub(crate) fn optional_vec<'py, T: FromPyObjectOwned<'py>>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Vec<T>>> {
    if value.is_none() {
        Ok(None)
    } else {
        vec(value).map(Some)
    }
}
