//! Python-facing compiler facade for the rank-N Tensor IR path.
//!
//! Existing transform helpers remain compatibility APIs. New code should use
//! `Compiler -> Program -> Executable` to make tracing, transformation, and
//! backend compilation one explicit lifecycle.

use std::collections::BTreeMap;

use nabla_core::tensor_ir::MlxBackend;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::tensor::PyTensor;
use crate::tensor_trace::{
    extract_tensor_map, trace_tensor, TensorCpuExecutionPlan, TensorCudaExecutionPlan,
    TensorInputSpec, TensorMlxExecutionPlan, TensorTraceResult,
};

#[pyclass(name = "Compiler", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PyNablaCompiler;

#[pyclass(name = "Program", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyNablaProgram {
    traced: TensorTraceResult,
}

#[pyclass(name = "Executable", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyNablaExecutable {
    inner: PyExecutable,
}

#[derive(Clone, Debug)]
enum PyExecutable {
    Cpu(TensorCpuExecutionPlan),
    Cuda(TensorCudaExecutionPlan),
    Mlx(TensorMlxExecutionPlan),
}

#[derive(Clone, Copy, Debug)]
enum PyExecutableTarget {
    Cpu,
    Cuda { device_ordinal: usize },
    Mlx,
}

fn parse_target(target: &str, device_ordinal: usize) -> PyResult<PyExecutableTarget> {
    match target {
        "cpu" => Ok(PyExecutableTarget::Cpu),
        "cuda" => Ok(PyExecutableTarget::Cuda { device_ordinal }),
        "mlx" => Ok(PyExecutableTarget::Mlx),
        _ => Err(PyValueError::new_err(
            "target must be one of 'cpu', 'cuda', or 'mlx'",
        )),
    }
}

fn capability(target: &str) -> PyResult<bool> {
    match target {
        "cpu" => Ok(true),
        "cuda" => Ok(cfg!(all(feature = "cuda", target_os = "linux"))),
        "mlx" => Ok(cfg!(all(feature = "mlx", target_os = "macos"))),
        _ => Err(PyValueError::new_err(
            "target must be one of 'cpu', 'cuda', or 'mlx'",
        )),
    }
}

#[pymethods]
impl PyNablaCompiler {
    #[new]
    fn new() -> Self {
        Self
    }

    fn trace(
        &self,
        py: Python<'_>,
        function: &Bound<'_, PyAny>,
        input_specs: Vec<TensorInputSpec>,
    ) -> PyResult<PyNablaProgram> {
        trace_tensor(py, function, input_specs).map(|traced| PyNablaProgram { traced })
    }

    fn capability(&self, target: &str) -> PyResult<bool> {
        capability(target)
    }

    fn capabilities(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        for target in ["cpu", "cuda", "mlx"] {
            result.set_item(target, capability(target)?)?;
        }
        Ok(result.unbind())
    }
}

#[pymethods]
impl PyNablaProgram {
    #[getter]
    fn output_shape(&self) -> PyResult<Vec<usize>> {
        self.traced
            .graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .node_shape(self.traced.output.node_id)
            .map_err(PyValueError::new_err)
    }

    fn lower_text(&self) -> PyResult<String> {
        self.traced
            .graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))
            .map(|ir| ir.lower_text())
    }

    fn jvp(&self, input_name: &str) -> PyResult<Self> {
        self.traced
            .symbolic_jvp_result(input_name)
            .map(|traced| Self { traced })
            .map_err(PyValueError::new_err)
    }

    fn vjp(&self, cotangent_name: &str) -> PyResult<BTreeMap<String, Self>> {
        self.traced
            .symbolic_vjp_results(cotangent_name)
            .map(|results| {
                results
                    .into_iter()
                    .map(|(name, traced)| (name, Self { traced }))
                    .collect()
            })
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (target = "cpu", device_ordinal = 0))]
    fn compile(&self, target: &str, device_ordinal: usize) -> PyResult<PyNablaExecutable> {
        let inner = match parse_target(target, device_ordinal)? {
            PyExecutableTarget::Cpu => PyExecutable::Cpu(
                self.traced
                    .graph
                    .compile_cpu_plan(self.traced.output.node_id)
                    .map_err(PyValueError::new_err)?,
            ),
            PyExecutableTarget::Cuda { device_ordinal } => PyExecutable::Cuda(
                self.traced
                    .graph
                    .compile_cuda_plan(self.traced.output.node_id, device_ordinal)
                    .map_err(PyValueError::new_err)?,
            ),
            PyExecutableTarget::Mlx => PyExecutable::Mlx(
                self.traced
                    .graph
                    .compile_mlx_plan(self.traced.output.node_id)
                    .map_err(PyValueError::new_err)?,
            ),
        };
        Ok(PyNablaExecutable { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Program(output_shape={:?})",
            self.output_shape().unwrap_or_default()
        )
    }
}

#[pymethods]
impl PyNablaExecutable {
    #[getter]
    fn target(&self) -> &'static str {
        match self.inner {
            PyExecutable::Cpu(_) => "cpu",
            PyExecutable::Cuda(_) => "cuda",
            PyExecutable::Mlx(_) => "mlx",
        }
    }

    #[getter]
    fn node_count(&self) -> usize {
        match &self.inner {
            PyExecutable::Cpu(plan) => plan.plan.node_count(),
            PyExecutable::Cuda(plan) => plan.plan.node_count(),
            PyExecutable::Mlx(plan) => plan.plan.node_count(),
        }
    }

    fn evaluate(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let inputs = extract_tensor_map(inputs)?;
        let value = match &self.inner {
            PyExecutable::Cpu(plan) => plan.plan.evaluate(&inputs),
            PyExecutable::Cuda(plan) => plan.plan.execute(&inputs),
            PyExecutable::Mlx(plan) => {
                let retained = plan
                    .retained_inputs
                    .lock()
                    .map_err(|_| PyValueError::new_err("MLX retained input lock is poisoned"))?;
                MlxBackend
                    .execute_many_with_state(
                        &plan.plan,
                        &[plan.plan.output_node_id()],
                        &inputs,
                        &retained,
                    )
                    .and_then(|mut values| {
                        values
                            .pop()
                            .ok_or_else(|| "MLX execution produced no output".to_string())
                    })
            }
        }
        .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    fn __call__(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        self.evaluate(inputs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Executable(target={}, node_count={})",
            self.target(),
            self.node_count()
        )
    }
}
