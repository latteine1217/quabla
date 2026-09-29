// Every unsafe block and impl must document its soundness invariant.
#![warn(clippy::undocumented_unsafe_blocks)]

use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};
use quabla_core::tensor_ir::TensorDType;

mod compiler;
mod dtype;
mod errors;
mod interop;
mod matrix;
mod optim;
mod tensor;
mod tensor_trace;
mod trace;

pub use compiler::{PyQuablaCompiler, PyQuablaExecutable, PyQuablaProgram};
pub use dtype::PyDType;
pub use matrix::PyMatrix;
pub use optim::{sum_gradients, PyAdam};
pub use tensor::{PyTensor, PyTensorView};
pub use tensor_trace::{
    InlineBinding, StagedExecutable, TensorBatchCudaJitFunction,
    TensorBatchCudaValueAndGradFunction, TensorBatchJitFunction,
    TensorBatchMlxValueAndGradFunction, TensorBatchValueAndGradFunction, TensorCondFunction,
    TensorCondJvpFunction, TensorCondValueAndGradFunction, TensorCpuExecutionPlan,
    TensorCudaAdamOptimizer, TensorCudaExecutionPlan, TensorCudaValueAndGradFunction,
    TensorGradScalarFunction, TensorHessianScalarFunction, TensorHvpScalarFunction,
    TensorJacobianFunction, TensorJitFunction, TensorJvpFunction, TensorMlxAdamOptimizer,
    TensorMlxExecutionPlan, TensorMlxValueAndGradFunction, TensorTraceGraph, TensorTraceResult,
    TensorValueAndGradFunction, TensorVjpFunction, TensorVmapCudaFunction,
    TensorVmapCudaHvpScalarFunction, TensorVmapCudaJvpFunction, TensorVmapCudaVjpFunction,
    TensorVmapFunction, TensorVmapHvpScalarFunction, TensorVmapJvpFunction, TensorVmapMlxFunction,
    TensorVmapMlxJvpFunction, TensorVmapMlxVjpFunction, TensorVmapVjpFunction, TraceTensor,
};
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

    if let (Ok(mask), Ok(on_true), Ok(on_false)) = (
        mask.extract::<PyRef<'_, TraceMatrix>>(),
        on_true.extract::<PyRef<'_, TraceMatrix>>(),
        on_false.extract::<PyRef<'_, TraceMatrix>>(),
    ) {
        let output = TraceMatrix::try_where(&mask, &on_true, &on_false)
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
        "where expects three Matrix or TraceMatrix operands, or a Tensor or TraceTensor mask \
         with Tensor, TraceTensor, or numeric scalar branches",
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

    let mut trace_matrices = Vec::with_capacity(len);
    for index in 0..len {
        let item = matrices.get_item(index)?;
        let matrix = item.extract::<PyRef<'_, TraceMatrix>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "concat expects a sequence containing only Matrix, Tensor, TraceMatrix, or TraceTensor operands",
            )
        })?;
        trace_matrices.push(matrix.clone());
    }

    let output = TraceMatrix::try_concat(&trace_matrices, axis)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.into_pyobject(py)?.into_any().unbind())
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
    m.add_class::<PyQuablaCompiler>()?;
    m.add_class::<PyQuablaProgram>()?;
    m.add_class::<PyQuablaExecutable>()?;
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyAdam>()?;
    m.add_function(wrap_pyfunction!(optim::sum_gradients, m)?)?;
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
    m.add_class::<TensorTraceResult>()?;
    m.add_class::<TensorCpuExecutionPlan>()?;
    m.add_class::<TensorCudaExecutionPlan>()?;
    m.add_class::<TensorMlxExecutionPlan>()?;
    m.add_class::<TensorCudaAdamOptimizer>()?;
    m.add_class::<TensorMlxAdamOptimizer>()?;
    m.add_class::<TensorGradScalarFunction>()?;
    m.add_class::<TensorValueAndGradFunction>()?;
    m.add_class::<TensorHessianScalarFunction>()?;
    m.add_class::<TensorHvpScalarFunction>()?;
    m.add_class::<TensorJitFunction>()?;
    m.add_class::<TensorCondFunction>()?;
    m.add_class::<TensorCondJvpFunction>()?;
    m.add_class::<TensorCondValueAndGradFunction>()?;
    m.add_class::<TensorBatchJitFunction>()?;
    m.add_class::<TensorBatchValueAndGradFunction>()?;
    m.add_class::<TensorBatchMlxValueAndGradFunction>()?;
    m.add_class::<TensorBatchCudaJitFunction>()?;
    m.add_class::<TensorBatchCudaValueAndGradFunction>()?;
    m.add_class::<TensorCudaValueAndGradFunction>()?;
    m.add_class::<TensorMlxValueAndGradFunction>()?;
    m.add_class::<TensorVjpFunction>()?;
    m.add_class::<TensorJvpFunction>()?;
    m.add_class::<TensorJacobianFunction>()?;
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
    m.add_function(wrap_pyfunction!(tensor_trace::trace_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_grad_scalar_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_value_and_grad_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_hessian_scalar_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_hvp_scalar_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_jit_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_cond, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_cond_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_cond_jvp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_cond_value_and_grad_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_jit_batch_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_jit_batch_cuda_fn, m)?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_batch_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_batch_cuda_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_batch_mlx_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_fori_loop, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_fori_loop_region, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_scan_region, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_scan, m)?)?;
    m.add_class::<TensorVmapFunction>()?;
    m.add_class::<TensorVmapHvpScalarFunction>()?;
    m.add_class::<TensorVmapCudaFunction>()?;
    m.add_class::<TensorVmapCudaHvpScalarFunction>()?;
    m.add_class::<TensorVmapCudaJvpFunction>()?;
    m.add_class::<TensorVmapMlxFunction>()?;
    m.add_class::<TensorVmapMlxJvpFunction>()?;
    m.add_class::<TensorVmapMlxVjpFunction>()?;
    m.add_class::<TensorVmapVjpFunction>()?;
    m.add_class::<TensorVmapJvpFunction>()?;
    m.add_class::<TensorVmapCudaVjpFunction>()?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_jit_cuda_fn, m)?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_cuda_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_data_parallel_cuda_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_value_and_grad_mlx_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::mlx_adam_loss_optimizer, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_cuda_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_mlx_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_vjp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_vjp_mlx_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_vjp_cuda_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_jvp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_jvp_mlx_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vmap_jvp_cuda_fn, m)?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_vmap_hvp_scalar_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        tensor_trace::tensor_vmap_hvp_scalar_cuda_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_vjp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_jvp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_jacobian_fn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::cuda_adam_step, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::cuda_adam_optimizer, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::cuda_adam_vjp_optimizer, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_trace::cuda_adam_loss_optimizer, m)?)?;
    m.add_function(wrap_pyfunction!(py_where, m)?)?;
    m.add_function(wrap_pyfunction!(py_concat, m)?)?;
    m.add_function(wrap_pyfunction!(py_stack, m)?)?;
    m.add_function(wrap_pyfunction!(py_einsum, m)?)?;
    Ok(())
}
