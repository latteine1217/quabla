use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

mod matrix;
mod optim;
mod tensor;
mod tensor_trace;
mod trace;

pub use matrix::PyMatrix;
pub use optim::{sum_gradients, PyAdam};
pub use tensor::{PyTensor, PyTensorView};
pub use tensor_trace::{
    TensorBatchCudaJitFunction, TensorBatchCudaValueAndGradFunction, TensorBatchJitFunction,
    TensorBatchMlxValueAndGradFunction, TensorBatchValueAndGradFunction, TensorCondFunction,
    TensorCondJvpFunction, TensorCondValueAndGradFunction, TensorCpuExecutionPlan,
    TensorCudaAdamOptimizer, TensorCudaExecutionPlan, TensorCudaValueAndGradFunction,
    TensorGradScalarFunction, TensorHessianScalarFunction, TensorHvpScalarFunction,
    TensorJacobianFunction, TensorJitFunction, TensorJvpFunction, TensorMlxAdamOptimizer,
    TensorMlxExecutionPlan, TensorMlxValueAndGradFunction, TensorTraceGraph, TensorTraceResult,
    TensorValueAndGradFunction, TensorVjpFunction, TensorVmapCudaFunction,
    TensorVmapCudaJvpFunction, TensorVmapCudaVjpFunction, TensorVmapFunction,
    TensorVmapJvpFunction, TensorVmapMlxFunction, TensorVmapMlxJvpFunction,
    TensorVmapMlxVjpFunction, TensorVmapVjpFunction, TraceTensor,
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

    if let (Ok(mask), Ok(on_true), Ok(on_false)) = (
        mask.extract::<PyRef<'_, PyTensor>>(),
        on_true.extract::<PyRef<'_, PyTensor>>(),
        on_false.extract::<PyRef<'_, PyTensor>>(),
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

    if let (Ok(mask), Ok(on_true), Ok(on_false)) = (
        mask.extract::<PyRef<'_, TraceTensor>>(),
        on_true.extract::<PyRef<'_, TraceTensor>>(),
        on_false.extract::<PyRef<'_, TraceTensor>>(),
    ) {
        let output = mask
            .where_tensor(&on_true, &on_false)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "where expects three Matrix, Tensor, TraceMatrix, or TraceTensor operands",
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

    let mut trace_tensors = Vec::with_capacity(len);
    let mut all_trace_tensors = true;
    for index in 0..len {
        let item = matrices.get_item(index)?;
        match item.extract::<PyRef<'_, TraceTensor>>() {
            Ok(tensor) => trace_tensors.push(tensor.clone()),
            Err(_) => {
                all_trace_tensors = false;
                break;
            }
        }
    }

    if all_trace_tensors {
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

    let mut traced = Vec::with_capacity(len);
    for index in 0..len {
        let tensor = tensors
            .get_item(index)?
            .extract::<PyRef<'_, TraceTensor>>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "stack expects only Tensor or TraceTensor operands",
                )
            })?;
        traced.push(tensor.clone());
    }
    let output =
        TraceTensor::try_stack(&traced, axis).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(output.into_pyobject(py)?.into_any().unbind())
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
    if let (Ok(lhs), Ok(rhs)) = (
        lhs.extract::<PyRef<'_, TraceTensor>>(),
        rhs.extract::<PyRef<'_, TraceTensor>>(),
    ) {
        let output = lhs
            .try_matmul(&rhs)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(output.into_pyobject(py)?.into_any().unbind());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "einsum expects two Tensor or two TraceTensor operands",
    ))
}

#[pymodule]
fn nabla(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyAdam>()?;
    m.add_function(wrap_pyfunction!(optim::sum_gradients, m)?)?;
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
    m.add_function(wrap_pyfunction!(tensor_trace::tensor_scan, m)?)?;
    m.add_class::<TensorVmapFunction>()?;
    m.add_class::<TensorVmapCudaFunction>()?;
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
