use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

mod matrix;
mod trace;

pub use matrix::PyMatrix;
pub use trace::{
    CpuExecutionPlan, GradFunction, GradScalarFunction, GradScalarTransform, IrAttrValue,
    JacobianFunction, JacobiansFunction, JitFunction, JitTransform, JvpFunction, TraceGraph,
    TraceMatrix, TraceResult, ValueAndGradFunction, VjpFunction,
};

#[pyfunction(name = "where")]
fn py_where(
    py: Python<'_>,
    mask: &Bound<'_, PyAny>,
    on_true: &Bound<'_, PyAny>,
    on_false: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if let (Ok(mask), Ok(on_true), Ok(on_false)) = (
        mask.extract::<PyRef<'_, PyMatrix>>(),
        on_true.extract::<PyRef<'_, PyMatrix>>(),
        on_false.extract::<PyRef<'_, PyMatrix>>(),
    ) {
        let output = PyMatrix::try_where(&mask, &on_true, &on_false)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    if let (Ok(mask), Ok(on_true), Ok(on_false)) = (
        mask.extract::<PyRef<'_, TraceMatrix>>(),
        on_true.extract::<PyRef<'_, TraceMatrix>>(),
        on_false.extract::<PyRef<'_, TraceMatrix>>(),
    ) {
        let output = TraceMatrix::try_where(&mask, &on_true, &on_false)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "where expects either three Matrix operands or three TraceMatrix operands",
    ))
}

#[pyfunction(name = "concat")]
#[pyo3(signature = (matrices, axis))]
fn py_concat(py: Python<'_>, matrices: &Bound<'_, PySequence>, axis: usize) -> PyResult<Py<PyAny>> {
    let len = matrices.len()?;

    let mut eager_matrices = Vec::with_capacity(len);
    let mut all_eager = true;
    for index in 0..len {
        let item = matrices.get_item(index)?;
        match item.extract::<PyRef<'_, PyMatrix>>() {
            Ok(matrix) => eager_matrices.push(matrix.clone()),
            Err(_) => {
                all_eager = false;
                break;
            }
        }
    }

    if all_eager {
        let output = PyMatrix::try_concat(&eager_matrices, axis)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    let mut trace_matrices = Vec::with_capacity(len);
    for index in 0..len {
        let item = matrices.get_item(index)?;
        let matrix = item.extract::<PyRef<'_, TraceMatrix>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "concat expects a sequence containing only Matrix or only TraceMatrix operands",
            )
        })?;
        trace_matrices.push(matrix.clone());
    }

    let output = TraceMatrix::try_concat(&trace_matrices, axis)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.into_pyobject(py)?.into_any().unbind())
}

#[pymodule]
fn nabla(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    m.add_class::<TraceGraph>()?;
    m.add_class::<TraceMatrix>()?;
    m.add_class::<TraceResult>()?;
    m.add_class::<CpuExecutionPlan>()?;
    m.add_class::<GradFunction>()?;
    m.add_class::<GradScalarFunction>()?;
    m.add_class::<GradScalarTransform>()?;
    m.add_class::<ValueAndGradFunction>()?;
    m.add_class::<VjpFunction>()?;
    m.add_class::<JacobianFunction>()?;
    m.add_class::<JacobiansFunction>()?;
    m.add_class::<JvpFunction>()?;
    m.add_class::<JitFunction>()?;
    m.add_class::<JitTransform>()?;
    m.add_function(wrap_pyfunction!(trace::trace, m)?)?;
    m.add_function(wrap_pyfunction!(trace::grad, m)?)?;
    m.add_function(wrap_pyfunction!(trace::grad_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::grad_scalar_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::value_and_grad_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::vjp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::jacobian_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::jacobians_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::jvp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(trace::grad_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(trace::jit, m)?)?;
    m.add_function(wrap_pyfunction!(py_where, m)?)?;
    m.add_function(wrap_pyfunction!(py_concat, m)?)?;
    Ok(())
}
