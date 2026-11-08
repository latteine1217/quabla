//! Python protocol hooks that `Tensor` and `TraceTensor` share: sequence
//! length and iteration over the leading axis, `round`, `format`, `%`, `//`,
//! and the keywords NumPy passes when it delegates a reduction to a method.
//!
//! `%`, `//`, and `round(x, ndigits)` call the functions of
//! `quabla._ops`, so eager and traced operands, weak Python scalars, and the
//! derivatives follow the same definitions as `quabla.mod`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList};

use crate::tensor::PyTensor;
use crate::tensor_trace::TraceTensor;

/// `len(x)`: the extent of the leading axis, as NumPy reports it.
pub(crate) fn length(shape: &[usize]) -> PyResult<usize> {
    shape
        .first()
        .copied()
        .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
}

/// `iter(x)`: the slices along the leading axis. Without `__iter__`, Python
/// falls back to calling `__getitem__(0), __getitem__(1), ...` until an
/// `IndexError`, so a 0-d tensor iterated as an empty sequence and
/// `sum(x)` silently gave 0; NumPy raises `TypeError` for it.
pub(crate) fn iterate<'py>(
    value: &Bound<'py, PyAny>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyIterator>> {
    let extent = shape
        .first()
        .copied()
        .ok_or_else(|| PyTypeError::new_err("iteration over a 0-d tensor"))?;
    let items = PyList::empty(value.py());
    for index in 0..extent {
        items.append(value.get_item(index)?)?;
    }
    items.as_any().try_iter()
}

/// `value in x`: whether any element equals `value`, as NumPy and JAX
/// define it. Without `__contains__`, Python iterated and compared each
/// slice with `==`, which is identity for tensors, so it was always `False`.
/// A traced `x` raises the `TracerError` of its `bool()`.
pub(crate) fn contains(tensor: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    tensor
        .call_method1("equal", (value,))?
        .call_method0("any")?
        .is_truthy()
}

/// `format(x, spec)`: the `str` of the tensor for an empty spec, and the
/// spec applied to the element of a 0-d tensor otherwise, as NumPy does; a
/// traced value raises the `TracerError` of `item()`.
pub(crate) fn format<'py>(
    value: &Bound<'py, PyAny>,
    rank: usize,
    spec: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if spec.is_empty() {
        return Ok(value.str()?.into_any());
    }
    if rank != 0 {
        return Err(PyTypeError::new_err(format!(
            "unsupported format string passed to {}.__format__; only a 0-d tensor takes a format spec",
            value.get_type().name()?
        )));
    }
    value
        .call_method0("item")?
        .call_method1("__format__", (spec,))
}

/// `round(x, ndigits)` through `quabla._ops._round_digits`.
pub(crate) fn round<'py>(
    value: &Bound<'py, PyAny>,
    ndigits: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    value
        .py()
        .import("quabla._ops")?
        .getattr("_round_digits")?
        .call1((value, ndigits))
}

/// `x1 <op> x2` through the `quabla._ops` function `name`, where `other` is
/// the operand that is not this tensor. An operand that is not an array or
/// a number returns `NotImplemented`, so Python can try its reflected
/// method or raise its own `TypeError`.
pub(crate) fn binary<'py>(
    name: &str,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let operand = other.is_instance_of::<PyTensor>()
        || other.is_instance_of::<TraceTensor>()
        || other.extract::<f64>().is_ok();
    if !operand {
        return Ok(other.py().NotImplemented().into_bound(other.py()));
    }
    x1.py()
        .import("quabla._ops")?
        .getattr(name)?
        .call1((x1, x2))
}

/// Accepts the `dtype` and `out` keywords that NumPy passes, as `None`,
/// when `np.sum(x)` and the other reductions delegate to the method of the
/// same name; any other value is rejected rather than ignored.
pub(crate) fn numpy_reduction_keywords(
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if out.is_some_and(|out| !out.is_none()) {
        return Err(PyTypeError::new_err(
            "out= is not supported: tensors are immutable, so use the returned tensor",
        ));
    }
    if dtype.is_some_and(|dtype| !dtype.is_none()) {
        return Err(PyTypeError::new_err(
            "dtype= is not supported: convert the tensor with astype before reducing",
        ));
    }
    Ok(())
}
