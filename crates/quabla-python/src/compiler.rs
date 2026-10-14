//! Python-facing compiler facade for the rank-N Tensor IR path:
//! `Compiler -> Program -> Executable` makes tracing, transformation, and
//! backend compilation one explicit lifecycle.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use quabla_core::compiler::{
    QuablaCompileError, QuablaCompiler, QuablaExecutable, QuablaMultiOutputExecutable, QuablaTarget,
};

use crate::tensor::PyTensor;
use crate::tensor_trace::{
    extract_tensor_map, trace_tensor, TensorInputSpec, TensorTraceGraph, TensorTraceResult,
    TraceTensor,
};

#[pyclass(name = "Compiler", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PyQuablaCompiler;

#[pyclass(name = "Program", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyQuablaProgram {
    inner: PyProgram,
}

#[derive(Clone, Debug)]
enum PyProgram {
    Single(TensorTraceResult),
    Many {
        graph: TensorTraceGraph,
        outputs: Vec<TraceTensor>,
    },
}

#[pyclass(name = "Executable", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyQuablaExecutable {
    inner: PyExecutable,
}

#[derive(Clone, Debug)]
enum PyExecutable {
    Single(QuablaExecutable),
    Many(QuablaMultiOutputExecutable),
}

impl PyQuablaProgram {
    pub(crate) fn from_outputs(
        graph: TensorTraceGraph,
        outputs: Vec<TraceTensor>,
    ) -> PyResult<Self> {
        {
            let ir = graph
                .ir
                .lock()
                .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
            for output in &outputs {
                ir.node_shape(output.node_id)
                    .map_err(PyValueError::new_err)?;
            }
        }
        Ok(Self {
            inner: PyProgram::Many { graph, outputs },
        })
    }

    fn single(traced: TensorTraceResult) -> Self {
        Self {
            inner: PyProgram::Single(traced),
        }
    }

    fn single_trace(&self, py: Python<'_>, op: &str) -> PyResult<&TensorTraceResult> {
        match &self.inner {
            PyProgram::Single(traced) => Ok(traced),
            PyProgram::Many { .. } => Err(crate::errors::unsupported_operation_error(
                py, "Program transforms require a single-output Compiler.trace program; use Python transforms before lowering a multi-output function", op,
            )),
        }
    }
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
impl PyQuablaCompiler {
    #[new]
    fn new() -> Self {
        Self
    }

    fn trace(
        &self,
        py: Python<'_>,
        function: &Bound<'_, PyAny>,
        input_specs: Vec<TensorInputSpec>,
    ) -> PyResult<PyQuablaProgram> {
        trace_tensor(py, function, input_specs).map(PyQuablaProgram::single)
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
impl PyQuablaProgram {
    #[getter]
    fn output_shape(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        let traced = self.single_trace(py, "output_shape")?;
        traced
            .graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .node_shape(traced.output.node_id)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn output_shapes(&self) -> PyResult<Vec<Vec<usize>>> {
        let (graph, outputs) = match &self.inner {
            PyProgram::Single(traced) => (&traced.graph, vec![traced.output.node_id]),
            PyProgram::Many { graph, outputs } => {
                (graph, outputs.iter().map(|output| output.node_id).collect())
            }
        };
        let ir = graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
        outputs
            .into_iter()
            .map(|output| ir.node_shape(output).map_err(PyValueError::new_err))
            .collect()
    }

    fn lower_text(&self) -> PyResult<String> {
        let graph = match &self.inner {
            PyProgram::Single(traced) => &traced.graph,
            PyProgram::Many { graph, .. } => graph,
        };
        let text = graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))
            .map(|ir| ir.lower_text())?;
        match &self.inner {
            PyProgram::Single(_) => Ok(text),
            PyProgram::Many { .. } => Ok(format!("{text}\noutputs: {:?}\n", self.output_shapes()?)),
        }
    }

    /// The program in the StableHLO inspection subset (single-output
    /// programs only); not an execution path.
    fn stablehlo_text(&self, py: Python<'_>) -> PyResult<String> {
        let traced = self.single_trace(py, "stablehlo_text")?;
        traced
            .graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .stablehlo_text(traced.output.node_id)
            .map_err(PyValueError::new_err)
    }

    fn jvp(&self, py: Python<'_>, input_name: &str) -> PyResult<Self> {
        self.single_trace(py, "jvp")?
            .symbolic_jvp_result(input_name)
            .map(Self::single)
            .map_err(PyValueError::new_err)
    }

    fn vjp(&self, py: Python<'_>, cotangent_name: &str) -> PyResult<BTreeMap<String, Self>> {
        self.single_trace(py, "vjp")?
            .symbolic_vjp_results(cotangent_name)
            .map(|results| {
                results
                    .into_iter()
                    .map(|(name, traced)| (name, Self::single(traced)))
                    .collect()
            })
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (target = "cpu", device_ordinal = 0))]
    fn compile(
        &self,
        py: Python<'_>,
        target: &str,
        device_ordinal: usize,
    ) -> PyResult<PyQuablaExecutable> {
        let target = match parse_target(target, device_ordinal)? {
            PyExecutableTarget::Cpu => QuablaTarget::Cpu,
            PyExecutableTarget::Cuda { device_ordinal } => QuablaTarget::Cuda { device_ordinal },
            PyExecutableTarget::Mlx => QuablaTarget::Mlx,
        };
        let traced = match &self.inner {
            PyProgram::Single(traced) => traced,
            PyProgram::Many { graph, outputs } => {
                if outputs.is_empty() {
                    return Err(crate::errors::unsupported_operation_error(
                        py, "an inspection Program with no array outputs has no native executable; use Lowered.compile() to assemble constant outputs", "compile",
                    ));
                }
                let program = graph
                    .clone()
                    .into_multi_output_program(
                        outputs.iter().map(|output| output.node_id).collect(),
                    )
                    .map_err(PyValueError::new_err)?;
                let executable = QuablaCompiler
                    .compile_many_checked(&program, target)
                    .map_err(|error| match error {
                        QuablaCompileError::Unavailable(message) => {
                            crate::errors::device_operation_error(
                                py,
                                message,
                                "compile",
                                target.name(),
                            )
                        }
                        QuablaCompileError::Unsupported { op, message } => {
                            crate::errors::device_operation_error(py, message, &op, target.name())
                        }
                        QuablaCompileError::InvalidProgram(message)
                        | QuablaCompileError::Backend(message) => PyValueError::new_err(message),
                    })?;
                return Ok(PyQuablaExecutable {
                    inner: PyExecutable::Many(executable),
                });
            }
        };
        // Without the build check: a target missing from the build keeps its
        // backend's own error (CUDA when compiling, MLX on first execution).
        let executable = traced
            .graph
            .compile_executable(traced.output.node_id, target)
            .map_err(PyValueError::new_err)?;
        Ok(PyQuablaExecutable {
            inner: PyExecutable::Single(executable),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        match &self.inner {
            PyProgram::Single(_) => format!(
                "Program(output_shape={:?})",
                self.output_shape(py).unwrap_or_default()
            ),
            PyProgram::Many { .. } => format!(
                "Program(output_shapes={:?})",
                self.output_shapes().unwrap_or_default()
            ),
        }
    }
}

#[pymethods]
impl PyQuablaExecutable {
    #[getter]
    fn target(&self) -> &'static str {
        match &self.inner {
            PyExecutable::Single(executable) => executable.target().name(),
            PyExecutable::Many(plan) => plan.target().name(),
        }
    }

    #[getter]
    fn node_count(&self) -> usize {
        match &self.inner {
            PyExecutable::Single(executable) => executable.plan().node_count(),
            PyExecutable::Many(plan) => plan.plan().node_count(),
        }
    }

    fn evaluate(&self, py: Python<'_>, inputs: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let inputs = extract_tensor_map(inputs)?;
        match &self.inner {
            PyExecutable::Single(executable) => {
                let value = executable.execute(&inputs).map_err(PyValueError::new_err)?;
                let value = PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?;
                Ok(Py::new(py, value)?.into_any())
            }
            PyExecutable::Many(plan) => {
                let values = plan
                    .execute(&inputs)
                    .map_err(PyValueError::new_err)?
                    .into_iter()
                    .map(PyTensor::from_dynamic_tensor)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(PyValueError::new_err)?;
                Ok(PyList::new(py, values)?.into_any().unbind())
            }
        }
    }

    fn __call__(&self, py: Python<'_>, inputs: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        self.evaluate(py, inputs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Executable(target={}, node_count={})",
            self.target(),
            self.node_count()
        )
    }
}
