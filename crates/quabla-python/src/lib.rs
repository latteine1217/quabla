// Every unsafe block and impl must document its soundness invariant.
#![warn(clippy::undocumented_unsafe_blocks)]

use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};
use quabla_core::tensor_ir::TensorDType;

mod compiler;
mod composite;
mod dtype;
mod errors;
mod interop;
mod repr;
mod tensor;
mod tensor_trace;

pub use compiler::{PyQuablaCompiler, PyQuablaExecutable, PyQuablaProgram};
pub use dtype::PyDType;
pub use tensor::{PyTensor, PyTensorView};
pub use tensor_trace::{
    InlineBinding, StagedExecutable, TensorCudaAdamOptimizer, TensorMlxAdamOptimizer,
    TensorTraceGraph, TraceTensor,
};

#[pyfunction(name = "where")]
fn py_where(
    py: Python<'_>,
    mask: &Bound<'_, PyAny>,
    on_true: &Bound<'_, PyAny>,
    on_false: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Tensor and TraceTensor branches may be Python scalars (weakly typed, adopting the other
    // branch's dtype).
    let eager_branch = |value: &Bound<'_, PyAny>| -> Option<PyTensor> {
        if let Ok(tensor) = value.extract::<PyRef<'_, PyTensor>>() {
            return Some(tensor.clone());
        }
        tensor::extract_scalar(value).map(PyTensor::weak_scalar)
    };
    if let (Ok(mask), Some(on_true), Some(on_false)) = (
        mask.extract::<PyRef<'_, PyTensor>>(),
        eager_branch(on_true),
        eager_branch(on_false),
    ) {
        let output = PyTensor::try_where(&mask, &on_true, &on_false)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    // A traced operand anywhere traces the selection: eager arrays become constants of its
    // graph and scalar branches stay weak.
    let tracer = [mask, on_true, on_false]
        .into_iter()
        .find_map(|value| value.extract::<TraceTensor>().ok());
    if let Some(tracer) = tracer {
        let traced_branch = |value: &Bound<'_, PyAny>| -> Option<PyResult<TraceTensor>> {
            tracer.traced_operand(value).or_else(|| {
                tensor::extract_scalar(value).map(|value| {
                    tracer
                        .scalar_tensor(value)
                        .map_err(pyo3::exceptions::PyValueError::new_err)
                })
            })
        };
        if let (Some(mask), Some(on_true), Some(on_false)) = (
            tracer.traced_operand(mask),
            traced_branch(on_true),
            traced_branch(on_false),
        ) {
            let output = mask?
                .where_tensor(&on_true?, &on_false?)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            return Ok(output.into_pyobject(py)?.into_any().unbind());
        }
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "where expects a Tensor or TraceTensor mask with Tensor, TraceTensor, or numeric \
         scalar branches",
    ))
}

fn is_tensor(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyTensor>() || value.is_instance_of::<TraceTensor>()
}

/// Module-level comparison: `quabla.greater(a, b)` is `a.greater(b)`, and a
/// scalar left operand uses the reflected method (`b.less(a)`).
fn compare_function(
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    name: &str,
    reflected: &str,
) -> PyResult<Py<PyAny>> {
    if is_tensor(lhs) {
        return Ok(lhs.call_method1(name, (rhs,))?.unbind());
    }
    if is_tensor(rhs) {
        return Ok(rhs.call_method1(reflected, (lhs,))?.unbind());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} expects a Tensor or TraceTensor operand"
    )))
}

fn tensor_method<'py>(
    value: &Bound<'py, PyAny>,
    name: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
) -> PyResult<Py<PyAny>> {
    if !is_tensor(value) {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} expects a Tensor or TraceTensor operand"
        )));
    }
    Ok(value.call_method1(name, args)?.unbind())
}

#[pyfunction(name = "greater")]
fn py_greater(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    compare_function(lhs, rhs, "greater", "less")
}

#[pyfunction(name = "greater_equal")]
fn py_greater_equal(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    compare_function(lhs, rhs, "greater_equal", "less_equal")
}

#[pyfunction(name = "less")]
fn py_less(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    compare_function(lhs, rhs, "less", "greater")
}

#[pyfunction(name = "less_equal")]
fn py_less_equal(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    compare_function(lhs, rhs, "less_equal", "greater_equal")
}

#[pyfunction(name = "equal")]
fn py_equal(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    compare_function(lhs, rhs, "equal", "equal")
}

#[pyfunction(name = "not_equal")]
fn py_not_equal(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    compare_function(lhs, rhs, "not_equal", "not_equal")
}

#[pyfunction(name = "logical_and")]
fn py_logical_and(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    tensor_method(lhs, "logical_and", (rhs,))
}

#[pyfunction(name = "logical_or")]
fn py_logical_or(lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    tensor_method(lhs, "logical_or", (rhs,))
}

#[pyfunction(name = "logical_not")]
fn py_logical_not(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    tensor_method(value, "logical_not", ())
}

#[pyfunction(name = "isnan")]
fn py_isnan(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    tensor_method(value, "isnan", ())
}

#[pyfunction(name = "isfinite")]
fn py_isfinite(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    tensor_method(value, "isfinite", ())
}

#[pyfunction(name = "concat")]
#[pyo3(signature = (matrices, axis))]
fn py_concat(py: Python<'_>, matrices: &Bound<'_, PySequence>, axis: usize) -> PyResult<Py<PyAny>> {
    let len = matrices.len()?;

    let mut eager_tensors = Vec::with_capacity(len);
    let mut all_tensors = true;
    for index in 0..len {
        let item = matrices.get_item(index)?;
        match item.extract::<PyRef<'_, PyTensor>>() {
            Ok(tensor) => eager_tensors.push(tensor.clone()),
            Err(_) => {
                all_tensors = false;
                break;
            }
        }
    }

    if all_tensors {
        let output = PyTensor::try_concat(&eager_tensors, axis)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    if let Some(trace_tensors) = traced_sequence(matrices)? {
        let output = TraceTensor::try_concat(&trace_tensors, axis)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "concat expects a sequence containing only Tensor or TraceTensor operands",
    ))
}

#[pyfunction(name = "stack")]
#[pyo3(signature = (tensors, axis=0))]
fn py_stack(py: Python<'_>, tensors: &Bound<'_, PySequence>, axis: isize) -> PyResult<Py<PyAny>> {
    let len = tensors.len()?;
    let mut eager = Vec::with_capacity(len);
    let mut all_eager = true;
    for index in 0..len {
        match tensors.get_item(index)?.extract::<PyRef<'_, PyTensor>>() {
            Ok(tensor) => eager.push(tensor.clone()),
            Err(_) => {
                all_eager = false;
                break;
            }
        }
    }
    if all_eager {
        let output =
            PyTensor::try_stack(&eager, axis).map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    let traced = traced_sequence(tensors)?.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("stack expects only Tensor or TraceTensor operands")
    })?;
    let output =
        TraceTensor::try_stack(&traced, axis).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.into_pyobject(py)?.into_any().unbind())
}

/// The items of a sequence holding at least one tracer as values of that
/// tracer's graph, with eager `Tensor` and `TensorView` items captured as
/// constants. `None` when there is no tracer or another kind of item.
fn traced_sequence(items: &Bound<'_, PySequence>) -> PyResult<Option<Vec<TraceTensor>>> {
    let items = (0..items.len()?)
        .map(|index| items.get_item(index))
        .collect::<PyResult<Vec<_>>>()?;
    let Some(tracer) = items
        .iter()
        .find_map(|item| item.extract::<TraceTensor>().ok())
    else {
        return Ok(None);
    };
    items
        .iter()
        .map(|item| tracer.traced_operand(item))
        .collect::<Option<PyResult<Vec<_>>>>()
        .transpose()
}

#[pyfunction(name = "einsum")]
fn py_einsum(
    py: Python<'_>,
    equation: &str,
    operands: &Bound<'_, PySequence>,
) -> PyResult<Py<PyAny>> {
    let normalized = equation
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    if !matches!(normalized.as_str(), "ij,jk->ik" | "...ij,...jk->...ik") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "einsum currently supports only 'ij,jk->ik' and '...ij,...jk->...ik'",
        ));
    }
    if operands.len()? != 2 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "einsum matrix multiplication requires exactly two operands",
        ));
    }
    let lhs = operands.get_item(0)?;
    let rhs = operands.get_item(1)?;

    if let (Ok(lhs), Ok(rhs)) = (
        lhs.extract::<PyRef<'_, PyTensor>>(),
        rhs.extract::<PyRef<'_, PyTensor>>(),
    ) {
        let output = lhs
            .try_matmul(&rhs)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }
    let tracer = [&lhs, &rhs]
        .into_iter()
        .find_map(|value| value.extract::<TraceTensor>().ok());
    if let Some(tracer) = tracer {
        if let (Some(lhs), Some(rhs)) = (tracer.traced_operand(&lhs), tracer.traced_operand(&rhs)) {
            let output = lhs?
                .try_matmul(&rhs?)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            return Ok(output.into_pyobject(py)?.into_any().unbind());
        }
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "einsum expects two Tensor or TraceTensor operands",
    ))
}

// The extension is the private submodule `quabla._quabla`; the pure-Python
// package in `python/quabla/` re-exports its names as the public `quabla`.
// The Rust library keeps the crate name `quabla` so the crate's own tests
// and `use quabla::...` paths are unaffected.
#[pymodule(name = "_quabla")]
fn quabla(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Set as a plain attribute: `add` would also append it to `__all__`,
    // and a star import must not rebind the importer's `__version__`.
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyQuablaCompiler>()?;
    m.add_class::<PyQuablaProgram>()?;
    m.add_class::<PyQuablaExecutable>()?;
    m.add_class::<PyDType>()?;
    m.add("float32", PyDType::from(TensorDType::F32))?;
    m.add("float64", PyDType::from(TensorDType::F64))?;
    m.add("bool_", PyDType::from(TensorDType::Bool))?;
    for function in [
        wrap_pyfunction!(py_greater, m)?,
        wrap_pyfunction!(py_greater_equal, m)?,
        wrap_pyfunction!(py_less, m)?,
        wrap_pyfunction!(py_less_equal, m)?,
        wrap_pyfunction!(py_equal, m)?,
        wrap_pyfunction!(py_not_equal, m)?,
        wrap_pyfunction!(py_logical_and, m)?,
        wrap_pyfunction!(py_logical_or, m)?,
        wrap_pyfunction!(py_logical_not, m)?,
        wrap_pyfunction!(py_isnan, m)?,
        wrap_pyfunction!(py_isfinite, m)?,
    ] {
        m.add_function(function)?;
    }
    m.add_class::<PyTensor>()?;
    m.add_class::<PyTensorView>()?;
    m.add_class::<TensorTraceGraph>()?;
    m.add_class::<TraceTensor>()?;
    m.add_class::<TensorCudaAdamOptimizer>()?;
    m.add_class::<TensorMlxAdamOptimizer>()?;
    // The region builders and the data-parallel constructor behind the
    // core control flow and `distributed.value_and_grad`.
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_cond, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_fori_loop, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_fori_loop_region, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_while_loop_region, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_scan_region, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_scan, m)?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_data_parallel_cuda_fn,
        m
    )?)?;
    // `Trainer`'s private executors: plain attributes, so they stay out of
    // `__all__` and the public type stub.
    m.setattr(
        "_cuda_trainer_optimizer",
        wrap_pyfunction!(tensor_trace::cuda_trainer_optimizer, m)?,
    )?;
    m.setattr(
        "_mlx_trainer_optimizer",
        wrap_pyfunction!(tensor_trace::mlx_trainer_optimizer, m)?,
    )?;
    m.add_function(wrap_pyfunction!(py_where, m)?)?;
    m.add_function(wrap_pyfunction!(py_concat, m)?)?;
    m.add_function(wrap_pyfunction!(py_stack, m)?)?;
    m.add_function(wrap_pyfunction!(py_einsum, m)?)?;
    Ok(())
}
