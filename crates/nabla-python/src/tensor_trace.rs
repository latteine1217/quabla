use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nabla_core::tensor_ir::{
    CudaBackend, CudaExecutionPlan, DynamicTensor, MlxBackend, TensorBackend, TensorExecutionPlan,
    TensorIr, TensorNodeId,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyString, PyTuple};

use crate::tensor::PyTensor;

#[pyclass(name = "TensorTraceGraph", skip_from_py_object)]
#[derive(Clone, Debug, Default)]
pub struct TensorTraceGraph {
    ir: Arc<Mutex<TensorIr>>,
}

#[pyclass(name = "TraceTensor", from_py_object)]
#[derive(Clone, Debug)]
pub struct TraceTensor {
    graph: TensorTraceGraph,
    node_id: TensorNodeId,
    shape: Vec<usize>,
}

#[pyclass(name = "TensorTraceResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorTraceResult {
    graph: TensorTraceGraph,
    output: TraceTensor,
}

#[pyclass(name = "TensorCpuExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorCpuExecutionPlan {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorCudaExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorCudaExecutionPlan {
    plan: CudaExecutionPlan,
}

#[pyclass(name = "TensorMlxExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorMlxExecutionPlan {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorCudaAdamOptimizer", skip_from_py_object)]
pub struct TensorCudaAdamOptimizer {
    parameter_plans: Vec<(String, TensorCudaExecutionPlan)>,
    shared_plan: Option<SharedCudaAdamPlan>,
    parameter_names: BTreeSet<String>,
    inputs: BTreeMap<String, DynamicTensor>,
    retained_inputs: BTreeSet<String>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

#[derive(Clone, Debug)]
struct SharedCudaAdamPlan {
    plan: TensorCudaExecutionPlan,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
}

#[pyclass(name = "TensorGradScalarFunction", skip_from_py_object)]
pub struct TensorGradScalarFunction {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorValueAndGradFunction", skip_from_py_object)]
pub struct TensorValueAndGradFunction {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorHessianScalarFunction", skip_from_py_object)]
pub struct TensorHessianScalarFunction {
    plan: TensorExecutionPlan,
    input_name: String,
}

#[pyclass(name = "TensorHvpScalarFunction", skip_from_py_object)]
pub struct TensorHvpScalarFunction {
    plan: TensorExecutionPlan,
    input_name: String,
}

#[pyclass(name = "TensorJitFunction", skip_from_py_object)]
pub struct TensorJitFunction {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorVjpFunction", skip_from_py_object)]
pub struct TensorVjpFunction {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorJvpFunction", skip_from_py_object)]
pub struct TensorJvpFunction {
    plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorJacobianFunction", skip_from_py_object)]
pub struct TensorJacobianFunction {
    plan: TensorExecutionPlan,
    input_name: String,
}

impl TensorTraceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&self, name: &str, shape: Vec<usize>) -> Result<TraceTensor, String> {
        let mut ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.input(name, shape)?;
        let shape = ir.node_shape(node_id)?;
        Ok(TraceTensor {
            graph: self.clone(),
            node_id,
            shape,
        })
    }

    fn existing_input(&self, name: &str) -> Result<TraceTensor, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.input_node_id(name)?;
        let shape = ir.node_shape(node_id)?;
        Ok(TraceTensor {
            graph: self.clone(),
            node_id,
            shape,
        })
    }

    fn evaluate_tensor(
        &self,
        output_node_id: TensorNodeId,
        inputs: BTreeMap<String, DynamicTensor>,
    ) -> Result<PyTensor, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        PyTensor::from_dynamic_tensor(ir.evaluate(output_node_id, &inputs)?)
    }

    fn evaluate_vjp_tensor(
        &self,
        output_node_id: TensorNodeId,
        inputs: BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<BTreeMap<String, PyTensor>, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        ir.vjp(output_node_id, &inputs, output_cotangent)?
            .into_iter()
            .map(|(name, tensor)| {
                PyTensor::from_dynamic_tensor(tensor).map(|tensor| (name, tensor))
            })
            .collect()
    }

    fn evaluate_jvp_tensor(
        &self,
        output_node_id: TensorNodeId,
        inputs: BTreeMap<String, DynamicTensor>,
        input_tangents: BTreeMap<String, DynamicTensor>,
    ) -> Result<(PyTensor, PyTensor), String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let (value, tangent) = ir.jvp(output_node_id, &inputs, &input_tangents)?;
        Ok((
            PyTensor::from_dynamic_tensor(value)?,
            PyTensor::from_dynamic_tensor(tangent)?,
        ))
    }

    fn hessian_scalar_tensor(
        &self,
        output_node_id: TensorNodeId,
        input_name: &str,
        inputs: BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        ir.hessian_scalar(output_node_id, input_name, &inputs)
    }

    fn hvp_scalar_tensor(
        &self,
        output_node_id: TensorNodeId,
        input_name: &str,
        inputs: BTreeMap<String, DynamicTensor>,
        input_tangent: DynamicTensor,
    ) -> Result<PyTensor, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        PyTensor::from_dynamic_tensor(ir.hvp_scalar(
            output_node_id,
            input_name,
            &inputs,
            input_tangent,
        )?)
    }

    fn lower_text_ir(&self) -> Result<String, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        Ok(ir.lower_text())
    }

    fn compile_cpu_plan(
        &self,
        output_node_id: TensorNodeId,
    ) -> Result<TensorCpuExecutionPlan, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        Ok(TensorCpuExecutionPlan {
            plan: ir.compile_cpu(output_node_id)?,
        })
    }

    fn compile_cuda_plan(
        &self,
        output_node_id: TensorNodeId,
        device_ordinal: usize,
    ) -> Result<TensorCudaExecutionPlan, String> {
        let plan = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .compile_cpu(output_node_id)?;
        CudaBackend::new(device_ordinal)
            .compile(plan)
            .map(|plan| TensorCudaExecutionPlan { plan })
    }

    fn compile_mlx_plan(
        &self,
        output_node_id: TensorNodeId,
    ) -> Result<TensorMlxExecutionPlan, String> {
        let plan = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .compile_cpu(output_node_id)?;
        Ok(TensorMlxExecutionPlan { plan })
    }

    fn compile_cuda_multi_plan(
        &self,
        output_node_ids: &[TensorNodeId],
        device_ordinal: usize,
    ) -> Result<(TensorCudaExecutionPlan, Vec<TensorNodeId>), String> {
        let (plan, output_node_ids) = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .compile_cpu_many(output_node_ids)?;
        CudaBackend::new(device_ordinal)
            .compile(plan)
            .map(|plan| (TensorCudaExecutionPlan { plan }, output_node_ids))
    }
}

impl TraceTensor {
    fn same_graph(&self, rhs: &Self) -> Result<(), String> {
        if Arc::ptr_eq(&self.graph.ir, &rhs.graph.ir) {
            Ok(())
        } else {
            Err("cannot combine TraceTensor values from different graphs".to_string())
        }
    }

    pub fn try_concat(tensors: &[Self], axis: usize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "concat requires at least one TraceTensor".to_string())?;
        for tensor in &tensors[1..] {
            first.same_graph(tensor)?;
        }
        let mut ir = first
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.concat(
            tensors.iter().map(|tensor| tensor.node_id).collect(),
            axis as isize,
        )?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: first.graph.clone(),
            node_id,
            shape,
        })
    }

    pub fn try_stack(tensors: &[Self], axis: isize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "stack requires at least one TraceTensor".to_string())?;
        if tensors.iter().any(|tensor| tensor.shape != first.shape) {
            return Err("stack requires TraceTensor values with identical shapes".to_string());
        }
        let rank = first.shape.len() + 1;
        let normalized_axis = if axis < 0 { axis + rank as isize } else { axis };
        let axis = usize::try_from(normalized_axis)
            .ok()
            .filter(|axis| *axis < rank)
            .ok_or_else(|| format!("axis {axis} is out of bounds for rank {rank}"))?;
        let reshaped = tensors
            .iter()
            .map(|tensor| {
                let mut shape = tensor.shape.clone();
                shape.insert(axis, 1);
                tensor.reshape_tensor(shape)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_concat(&reshaped, axis)
    }

    fn binary(&self, rhs: &Self, op: &str) -> Result<Self, String> {
        self.same_graph(rhs)?;
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = match op {
            "add" => ir.add(self.node_id, rhs.node_id)?,
            "sub" => ir.sub(self.node_id, rhs.node_id)?,
            "div" => ir.div(self.node_id, rhs.node_id)?,
            "mul" => ir.mul(self.node_id, rhs.node_id)?,
            "greater" => ir.greater(self.node_id, rhs.node_id)?,
            _ => return Err(format!("unsupported trace tensor binary op {op}")),
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn scalar_binary(&self, value: f64, op: &str) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let scalar = ir.scalar_constant(value);
        let node_id = match op {
            "add" => ir.add(self.node_id, scalar)?,
            "sub" => ir.sub(self.node_id, scalar)?,
            "div" => ir.div(self.node_id, scalar)?,
            "mul" => ir.mul(self.node_id, scalar)?,
            "greater" => ir.greater(self.node_id, scalar)?,
            _ => return Err(format!("unsupported trace tensor scalar op {op}")),
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn scalar_left_binary(&self, value: f64, op: &str) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let scalar = ir.scalar_constant(value);
        let node_id = match op {
            "add" => ir.add(scalar, self.node_id)?,
            "sub" => ir.sub(scalar, self.node_id)?,
            "div" => ir.div(scalar, self.node_id)?,
            "mul" => ir.mul(scalar, self.node_id)?,
            "greater" => ir.greater(scalar, self.node_id)?,
            _ => return Err(format!("unsupported left scalar trace op {op}")),
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn sum_tensor(&self, axis: Option<isize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = match axis {
            Some(axis) => ir.sum_axis(self.node_id, axis)?,
            None => ir.sum(self.node_id)?,
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn matmul_tensor(&self, rhs: &Self) -> Result<Self, String> {
        self.same_graph(rhs)?;
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.matmul(self.node_id, rhs.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    pub fn where_tensor(&self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        self.same_graph(on_true)?;
        self.same_graph(on_false)?;
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.where_select(self.node_id, on_true.node_id, on_false.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn tanh_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.tanh(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn exp_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.exp(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn reshape_tensor(&self, shape: Vec<usize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.reshape(self.node_id, shape)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn slice_tensor(&self, axis: isize, start: usize, stop: usize) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.slice_axis(self.node_id, axis, start, stop)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn broadcast_to_tensor(&self, shape: Vec<usize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.broadcast_to(self.node_id, shape)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn mean_tensor(&self, axis: Option<isize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = match axis {
            Some(axis) => ir.mean_axis(self.node_id, axis)?,
            None => ir.mean(self.node_id)?,
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn sin_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.sin(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn cos_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.cos(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn powi_tensor(&self, exponent: u32) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.powi(self.node_id, exponent)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn transpose_tensor(&self, axes: Option<Vec<isize>>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.transpose(self.node_id, axes)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn log_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.log(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }

    fn sqrt_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let log = ir.log(self.node_id)?;
        let half = ir.scalar_constant(0.5);
        let scaled = ir.mul(log, half)?;
        let node_id = ir.exp(scaled)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self {
            graph: self.graph.clone(),
            node_id,
            shape,
        })
    }
}

impl TensorTraceResult {
    fn new(graph: TensorTraceGraph, output: TraceTensor) -> Self {
        Self { graph, output }
    }

    fn symbolic_jvp_result(&self, input_name: &str) -> Result<Self, String> {
        let ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let transformed = ir.symbolic_jvp(self.output.node_id, input_name)?;
        let graph = TensorTraceGraph {
            ir: Arc::new(Mutex::new(transformed.graph)),
        };
        let shape = graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .node_shape(transformed.tangent)?;
        Ok(Self::new(
            graph.clone(),
            TraceTensor {
                graph,
                node_id: transformed.tangent,
                shape,
            },
        ))
    }

    fn symbolic_vjp_results(&self, cotangent_name: &str) -> Result<BTreeMap<String, Self>, String> {
        let ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let transformed = ir.symbolic_vjp(self.output.node_id, cotangent_name)?;
        let graph = TensorTraceGraph {
            ir: Arc::new(Mutex::new(transformed.graph)),
        };
        let ir = graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        transformed
            .gradients
            .into_iter()
            .map(|(name, node_id)| {
                let shape = ir.node_shape(node_id)?;
                Ok((
                    name,
                    Self::new(
                        graph.clone(),
                        TraceTensor {
                            graph: graph.clone(),
                            node_id,
                            shape,
                        },
                    ),
                ))
            })
            .collect()
    }
}

#[pymethods]
impl TensorTraceGraph {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    #[pyo3(signature = (name, shape = None))]
    fn input(&self, name: &str, shape: Option<Vec<usize>>) -> PyResult<TraceTensor> {
        match shape {
            Some(shape) => self.add_input(name, shape),
            None => self.existing_input(name),
        }
        .map_err(PyValueError::new_err)
    }

    fn lower_text(&self) -> PyResult<String> {
        self.lower_text_ir().map_err(PyValueError::new_err)
    }

    fn evaluate(
        &self,
        output_node_id: TensorNodeId,
        inputs: &Bound<'_, PyDict>,
    ) -> PyResult<PyTensor> {
        self.evaluate_tensor(output_node_id, extract_tensor_map(inputs)?)
            .map_err(PyValueError::new_err)
    }

    fn evaluate_vjp(
        &self,
        output_node_id: TensorNodeId,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<BTreeMap<String, PyTensor>> {
        self.evaluate_vjp_tensor(
            output_node_id,
            extract_tensor_map(inputs)?,
            output_cotangent
                .to_dynamic_tensor()
                .map_err(PyValueError::new_err)?,
        )
        .map_err(PyValueError::new_err)
    }

    fn evaluate_jvp(
        &self,
        output_node_id: TensorNodeId,
        inputs: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        self.evaluate_jvp_tensor(
            output_node_id,
            extract_tensor_map(inputs)?,
            extract_tensor_map(input_tangents)?,
        )
        .map_err(PyValueError::new_err)
    }

    fn hessian_scalar(
        &self,
        output_node_id: TensorNodeId,
        input_name: &str,
        inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.hessian_scalar_tensor(output_node_id, input_name, extract_tensor_map(inputs)?)
            .map_err(PyValueError::new_err)
    }

    fn hvp_scalar(
        &self,
        output_node_id: TensorNodeId,
        input_name: &str,
        inputs: &Bound<'_, PyDict>,
        input_tangent: &PyTensor,
    ) -> PyResult<PyTensor> {
        self.hvp_scalar_tensor(
            output_node_id,
            input_name,
            extract_tensor_map(inputs)?,
            input_tangent
                .to_dynamic_tensor()
                .map_err(PyValueError::new_err)?,
        )
        .map_err(PyValueError::new_err)
    }

    fn compile_cpu(&self, output_node_id: TensorNodeId) -> PyResult<TensorCpuExecutionPlan> {
        self.compile_cpu_plan(output_node_id)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (output_node_id, device_ordinal = 0))]
    fn compile_cuda(
        &self,
        output_node_id: TensorNodeId,
        device_ordinal: usize,
    ) -> PyResult<TensorCudaExecutionPlan> {
        self.compile_cuda_plan(output_node_id, device_ordinal)
            .map_err(PyValueError::new_err)
    }

    fn compile_mlx(&self, output_node_id: TensorNodeId) -> PyResult<TensorMlxExecutionPlan> {
        self.compile_mlx_plan(output_node_id)
            .map_err(PyValueError::new_err)
    }

    fn evaluate_value_and_vjp(
        &self,
        output_node_id: TensorNodeId,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(inputs)?;
        let output_cotangent = output_cotangent
            .to_dynamic_tensor()
            .map_err(PyValueError::new_err)?;
        let value = self
            .evaluate_tensor(output_node_id, inputs.clone())
            .map_err(PyValueError::new_err)?;
        let gradients = self
            .evaluate_vjp_tensor(output_node_id, inputs, output_cotangent)
            .map_err(PyValueError::new_err)?;
        Ok((value, gradients))
    }
}

#[pymethods]
impl TraceTensor {
    #[getter]
    fn node_id(&self) -> TensorNodeId {
        self.node_id
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "add")
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "sub")
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__sub__(rhs)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "mul")
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "div")
    }

    fn div(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__truediv__(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_scalar_left_operand(self, lhs, "add")
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_scalar_left_operand(self, lhs, "sub")
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_scalar_left_operand(self, lhs, "mul")
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_scalar_left_operand(self, lhs, "div")
    }

    #[pyo3(signature = (axis = None))]
    fn sum(&self, axis: Option<isize>) -> PyResult<Self> {
        self.sum_tensor(axis).map_err(PyValueError::new_err)
    }

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.matmul_tensor(rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.__matmul__(rhs)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "greater")
    }

    fn where_select(&self, on_true: &Self, on_false: &Self) -> PyResult<Self> {
        self.where_tensor(on_true, on_false)
            .map_err(PyValueError::new_err)
    }

    fn tanh(&self) -> PyResult<Self> {
        self.tanh_tensor().map_err(PyValueError::new_err)
    }

    fn exp(&self) -> PyResult<Self> {
        self.exp_tensor().map_err(PyValueError::new_err)
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.reshape_tensor(shape).map_err(PyValueError::new_err)
    }

    fn slice(&self, axis: isize, start: usize, stop: usize) -> PyResult<Self> {
        self.slice_tensor(axis, start, stop)
            .map_err(PyValueError::new_err)
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.broadcast_to_tensor(shape)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None))]
    fn mean(&self, axis: Option<isize>) -> PyResult<Self> {
        self.mean_tensor(axis).map_err(PyValueError::new_err)
    }

    fn sin(&self) -> PyResult<Self> {
        self.sin_tensor().map_err(PyValueError::new_err)
    }

    fn cos(&self) -> PyResult<Self> {
        self.cos_tensor().map_err(PyValueError::new_err)
    }

    fn powi(&self, exponent: u32) -> PyResult<Self> {
        self.powi_tensor(exponent).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axes = None))]
    fn transpose(&self, axes: Option<Vec<isize>>) -> PyResult<Self> {
        self.transpose_tensor(axes).map_err(PyValueError::new_err)
    }

    fn log(&self) -> PyResult<Self> {
        self.log_tensor().map_err(PyValueError::new_err)
    }

    fn sqrt(&self) -> PyResult<Self> {
        self.sqrt_tensor().map_err(PyValueError::new_err)
    }

    fn compile_cpu(&self) -> PyResult<TensorCpuExecutionPlan> {
        self.graph
            .compile_cpu_plan(self.node_id)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TraceTensor(node_id={}, shape={:?})",
            self.node_id, self.shape
        )
    }

    #[pyo3(signature = (device_ordinal = 0))]
    fn compile_cuda(&self, device_ordinal: usize) -> PyResult<TensorCudaExecutionPlan> {
        self.graph
            .compile_cuda_plan(self.node_id, device_ordinal)
            .map_err(PyValueError::new_err)
    }

    fn compile_mlx(&self) -> PyResult<TensorMlxExecutionPlan> {
        self.graph
            .compile_mlx_plan(self.node_id)
            .map_err(PyValueError::new_err)
    }

    fn symbolic_jvp(&self, input_name: &str) -> PyResult<TensorTraceResult> {
        TensorTraceResult::new(self.graph.clone(), self.clone())
            .symbolic_jvp_result(input_name)
            .map_err(PyValueError::new_err)
    }

    fn symbolic_vjp(&self, cotangent_name: &str) -> PyResult<BTreeMap<String, TensorTraceResult>> {
        TensorTraceResult::new(self.graph.clone(), self.clone())
            .symbolic_vjp_results(cotangent_name)
            .map_err(PyValueError::new_err)
    }
}

fn trace_tensor_or_scalar_operand(
    lhs: &TraceTensor,
    rhs: &Bound<'_, PyAny>,
    op: &str,
) -> PyResult<TraceTensor> {
    if let Ok(rhs) = rhs.extract::<PyRef<'_, TraceTensor>>() {
        return lhs.binary(&rhs, op).map_err(PyValueError::new_err);
    }
    if let Ok(value) = rhs.extract::<f64>() {
        return lhs.scalar_binary(value, op).map_err(PyValueError::new_err);
    }
    Err(PyTypeError::new_err(
        "expected a TraceTensor or numeric scalar operand",
    ))
}

fn trace_scalar_left_operand(
    rhs: &TraceTensor,
    lhs: &Bound<'_, PyAny>,
    op: &str,
) -> PyResult<TraceTensor> {
    let value = lhs.extract::<f64>().map_err(|_| {
        PyTypeError::new_err("expected a numeric scalar as the left TraceTensor operand")
    })?;
    rhs.scalar_left_binary(value, op)
        .map_err(PyValueError::new_err)
}

#[pymethods]
impl TensorTraceResult {
    #[getter]
    fn graph(&self) -> TensorTraceGraph {
        self.graph.clone()
    }

    #[getter]
    fn output(&self) -> TraceTensor {
        self.output.clone()
    }

    fn symbolic_jvp(&self, input_name: &str) -> PyResult<Self> {
        self.symbolic_jvp_result(input_name)
            .map_err(PyValueError::new_err)
    }

    fn symbolic_vjp(&self, cotangent_name: &str) -> PyResult<BTreeMap<String, Self>> {
        self.symbolic_vjp_results(cotangent_name)
            .map_err(PyValueError::new_err)
    }

    fn compile_cpu(&self) -> PyResult<TensorCpuExecutionPlan> {
        self.graph
            .compile_cpu_plan(self.output.node_id)
            .map_err(PyValueError::new_err)
    }

    fn compile_mlx(&self) -> PyResult<TensorMlxExecutionPlan> {
        self.graph
            .compile_mlx_plan(self.output.node_id)
            .map_err(PyValueError::new_err)
    }
}

#[pymethods]
impl TensorCpuExecutionPlan {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    #[getter]
    fn output_shape(&self) -> PyResult<Vec<usize>> {
        self.plan.output_shape().map_err(PyValueError::new_err)
    }

    fn lower_text(&self) -> String {
        self.plan.lower_text()
    }

    fn kernel_ir(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let program = self.plan.kernel_ir();
        let nodes = PyList::empty(py);
        for node in program.nodes {
            let item = PyDict::new(py);
            item.set_item("id", node.id)?;
            item.set_item("op", node.op)?;
            item.set_item("shape", node.shape)?;
            item.set_item("inputs", node.inputs)?;
            if let Some(name) = node.name {
                item.set_item("name", name)?;
            }
            nodes.append(item)?;
        }
        Ok(nodes.into())
    }

    fn validate_kernel_ir(&self) -> PyResult<()> {
        self.plan
            .kernel_ir()
            .validate()
            .map_err(PyValueError::new_err)
    }

    fn evaluate(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = self
            .plan
            .evaluate(&extract_tensor_map(inputs)?)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    fn evaluate_vjp(
        &self,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(inputs)?;
        let output_cotangent = output_cotangent
            .to_dynamic_tensor()
            .map_err(PyValueError::new_err)?;
        let (value, gradients) = self
            .plan
            .value_and_vjp(&inputs, output_cotangent)
            .map_err(PyValueError::new_err)?;
        let gradients = gradients
            .into_iter()
            .map(|(name, tensor)| {
                PyTensor::from_dynamic_tensor(tensor).map(|tensor| (name, tensor))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok((
            PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
            gradients,
        ))
    }

    fn evaluate_value_and_vjp(
        &self,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        self.evaluate_vjp(inputs, output_cotangent)
    }

    fn evaluate_jvp(
        &self,
        inputs: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (value, tangent) = self
            .plan
            .jvp(
                &extract_tensor_map(inputs)?,
                &extract_tensor_map(input_tangents)?,
            )
            .map_err(PyValueError::new_err)?;
        Ok((
            PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
            PyTensor::from_dynamic_tensor(tangent).map_err(PyValueError::new_err)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorCpuExecutionPlan(node_count={})",
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorMlxExecutionPlan {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        "mlx"
    }

    fn evaluate(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = MlxBackend
            .execute(&self.plan, &extract_tensor_map(inputs)?)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    fn __call__(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        self.evaluate(inputs)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorMlxExecutionPlan(node_count={})",
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorCudaExecutionPlan {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    #[getter]
    fn device_ordinal(&self) -> usize {
        self.plan.device_ordinal()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        if self.plan.uses_cublas() {
            "cublas"
        } else {
            "nvrtc"
        }
    }

    #[getter]
    fn device_buffer_count(&self) -> PyResult<usize> {
        self.plan
            .device_buffer_count()
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn fused_matmul_bias_tanh(&self) -> bool {
        self.plan.uses_fused_matmul_bias_tanh()
    }

    fn evaluate(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = self
            .plan
            .execute(&extract_tensor_map(inputs)?)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    fn __call__(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        self.evaluate(inputs)
    }

    #[pyo3(signature = (inputs, retained_input_names = None))]
    fn evaluate_device(
        &self,
        inputs: &Bound<'_, PyDict>,
        retained_input_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let retained_inputs = retained_input_names
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        self.plan
            .execute_retaining_without_output(&extract_tensor_map(inputs)?, &retained_inputs)
            .map_err(PyValueError::new_err)
    }

    fn synchronize(&self) -> PyResult<()> {
        self.plan.synchronize().map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (inputs, iterations, retained_input_names = None))]
    fn benchmark_device(
        &self,
        inputs: &Bound<'_, PyDict>,
        iterations: usize,
        retained_input_names: Option<Vec<String>>,
    ) -> PyResult<f64> {
        if iterations == 0 {
            return Err(PyValueError::new_err(
                "CUDA device benchmark iterations must be positive",
            ));
        }
        let inputs = extract_tensor_map(inputs)?;
        let retained_inputs = retained_input_names
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        self.plan
            .execute_retaining_without_output(&inputs, &retained_inputs)
            .map_err(PyValueError::new_err)?;
        self.plan.synchronize().map_err(PyValueError::new_err)?;
        let start = Instant::now();
        for _ in 0..iterations {
            self.plan
                .execute_retaining_without_output(&inputs, &retained_inputs)
                .map_err(PyValueError::new_err)?;
        }
        self.plan.synchronize().map_err(PyValueError::new_err)?;
        Ok(start.elapsed().as_secs_f64())
    }

    #[pyo3(signature = (inputs, parameter_name, learning_rate, retained_input_names = None))]
    fn sgd_step(
        &self,
        inputs: &Bound<'_, PyDict>,
        parameter_name: &str,
        learning_rate: f32,
        retained_input_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let mut retained = retained_input_names
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        retained.insert(parameter_name.to_string());
        self.plan
            .execute_retaining(&extract_tensor_map(inputs)?, &retained)
            .map_err(PyValueError::new_err)?;
        self.plan
            .sgd_step_input_from_output(parameter_name, learning_rate)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (inputs, parameter_name, learning_rate, retained_input_names = None, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8))]
    #[allow(clippy::too_many_arguments)]
    fn adam_step(
        &self,
        inputs: &Bound<'_, PyDict>,
        parameter_name: &str,
        learning_rate: f32,
        retained_input_names: Option<Vec<String>>,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> PyResult<()> {
        let mut retained = retained_input_names
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        retained.insert(parameter_name.to_string());
        self.plan
            .execute_retaining(&extract_tensor_map(inputs)?, &retained)
            .map_err(PyValueError::new_err)?;
        self.plan
            .adam_step_input_from_output(parameter_name, learning_rate, beta1, beta2, epsilon)
            .map_err(PyValueError::new_err)
    }

    fn retained_input(&self, name: &str) -> PyResult<PyTensor> {
        let value = self
            .plan
            .retained_input_to_host(name)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (source_name, target, target_name = None))]
    fn sync_retained_input_to(
        &self,
        source_name: &str,
        target: PyRef<'_, TensorCudaExecutionPlan>,
        target_name: Option<&str>,
    ) -> PyResult<()> {
        self.plan
            .sync_retained_input_to(
                source_name,
                &target.plan,
                target_name.unwrap_or(source_name),
            )
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorCudaExecutionPlan(node_count={}, device_ordinal={}, backend={})",
            self.plan.node_count(),
            self.plan.device_ordinal(),
            if self.plan.uses_cublas() {
                "cublas"
            } else {
                "nvrtc"
            }
        )
    }
}

#[pymethods]
impl TensorGradScalarFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, PyTensor>> {
        let inputs = extract_tensor_map(values)?;
        self.plan
            .vjp(
                &inputs,
                DynamicTensor::filled(vec![], 1.0).map_err(PyValueError::new_err)?,
            )
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|(name, tensor)| {
                PyTensor::from_dynamic_tensor(tensor).map(|tensor| (name, tensor))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorGradScalarFunction(node_count={})",
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorValueAndGradFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let (value, gradients) = self
            .plan
            .value_and_vjp(
                &extract_tensor_map(values)?,
                DynamicTensor::filled(vec![], 1.0).map_err(PyValueError::new_err)?,
            )
            .map_err(PyValueError::new_err)?;
        let gradients = gradients
            .into_iter()
            .map(|(name, tensor)| {
                PyTensor::from_dynamic_tensor(tensor).map(|tensor| (name, tensor))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok((
            PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
            gradients,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorValueAndGradFunction(node_count={})",
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorHessianScalarFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<Vec<Vec<f64>>> {
        self.plan
            .hessian_scalar(&self.input_name, &extract_tensor_map(values)?)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorHessianScalarFunction(input_name={:?}, node_count={})",
            self.input_name,
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorHvpScalarFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>, input_tangent: &PyTensor) -> PyResult<PyTensor> {
        let tangent = input_tangent
            .to_dynamic_tensor()
            .map_err(PyValueError::new_err)?;
        let value = self
            .plan
            .hvp_scalar(&self.input_name, &extract_tensor_map(values)?, tangent)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorHvpScalarFunction(input_name={:?}, node_count={})",
            self.input_name,
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorJitFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = self
            .plan
            .evaluate(&extract_tensor_map(values)?)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!("TensorJitFunction(node_count={})", self.plan.node_count())
    }
}

#[pymethods]
impl TensorVjpFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(values)?;
        let output_cotangent = output_cotangent
            .to_dynamic_tensor()
            .map_err(PyValueError::new_err)?;
        let value = self.plan.evaluate(&inputs).map_err(PyValueError::new_err)?;
        let gradients = self
            .plan
            .vjp(&inputs, output_cotangent)
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|(name, tensor)| {
                PyTensor::from_dynamic_tensor(tensor).map(|tensor| (name, tensor))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok((
            PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
            gradients,
        ))
    }

    fn __repr__(&self) -> String {
        format!("TensorVjpFunction(node_count={})", self.plan.node_count())
    }
}

#[pymethods]
impl TensorJvpFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (value, tangent) = self
            .plan
            .jvp(
                &extract_tensor_map(values)?,
                &extract_tensor_map(input_tangents)?,
            )
            .map_err(PyValueError::new_err)?;
        Ok((
            PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
            PyTensor::from_dynamic_tensor(tangent).map_err(PyValueError::new_err)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!("TensorJvpFunction(node_count={})", self.plan.node_count())
    }
}

#[pymethods]
impl TensorJacobianFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<Vec<Vec<f64>>> {
        let inputs = extract_tensor_map(values)?;
        let selected = inputs
            .get(&self.input_name)
            .ok_or_else(|| PyValueError::new_err(format!("missing input {:?}", self.input_name)))?;
        let input_count = selected.data().len();
        let output_count = self
            .plan
            .output_shape()
            .map_err(PyValueError::new_err)?
            .iter()
            .product::<usize>();
        let mut jacobian = vec![vec![0.0; input_count]; output_count];
        let zero_tangents = inputs
            .iter()
            .map(|(name, value)| {
                DynamicTensor::filled(value.shape().to_vec(), 0.0).map(|zero| (name.clone(), zero))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)?;

        for column in 0..input_count {
            let mut tangents = zero_tangents.clone();
            let mut direction = vec![0.0; input_count];
            direction[column] = 1.0;
            tangents.insert(
                self.input_name.clone(),
                DynamicTensor::new(selected.shape().to_vec(), direction)
                    .map_err(PyValueError::new_err)?,
            );
            let (_, tangent) = self
                .plan
                .jvp(&inputs, &tangents)
                .map_err(PyValueError::new_err)?;
            for (row, value) in tangent.data().iter().enumerate() {
                jacobian[row][column] = *value;
            }
        }
        Ok(jacobian)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorJacobianFunction(input_name={:?}, node_count={})",
            self.input_name,
            self.plan.node_count()
        )
    }
}

pub fn trace_tensor_python_function(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorTraceResult> {
    let graph = TensorTraceGraph::new();
    let mut inputs = Vec::with_capacity(input_specs.len());
    for (name, shape) in input_specs {
        inputs.push(
            graph
                .add_input(&name, shape)
                .map_err(PyValueError::new_err)?,
        );
    }
    let args = PyTuple::new(py, inputs)?;
    let output: TraceTensor = function
        .call1(args)?
        .extract()
        .map_err(|_| PyTypeError::new_err("trace_tensor function must return a TraceTensor"))?;
    if !Arc::ptr_eq(&graph.ir, &output.graph.ir) {
        return Err(PyValueError::new_err(
            "trace_tensor function returned a tensor from a different graph",
        ));
    }
    Ok(TensorTraceResult::new(graph, output))
}

#[pyfunction]
pub fn trace_tensor(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorTraceResult> {
    trace_tensor_python_function(py, function, input_specs)
}

#[pyfunction]
pub fn tensor_grad_scalar_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorGradScalarFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    if !traced.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "tensor_grad_scalar_fn requires a scalar output, got shape {:?}",
            traced.output.shape
        )));
    }
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorGradScalarFunction { plan })
}

#[pyfunction]
pub fn tensor_value_and_grad_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorValueAndGradFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    if !traced.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_fn requires a scalar output, got shape {:?}",
            traced.output.shape
        )));
    }
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorValueAndGradFunction { plan })
}

#[pyfunction]
pub fn tensor_hessian_scalar_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    input_name: String,
) -> PyResult<TensorHessianScalarFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    if !traced.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "tensor_hessian_scalar_fn requires a scalar output, got shape {:?}",
            traced.output.shape
        )));
    }
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    plan.input_shape(&input_name)
        .map_err(PyValueError::new_err)?;
    Ok(TensorHessianScalarFunction { plan, input_name })
}

#[pyfunction]
pub fn tensor_hvp_scalar_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    input_name: String,
) -> PyResult<TensorHvpScalarFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    if !traced.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "tensor_hvp_scalar_fn requires a scalar output, got shape {:?}",
            traced.output.shape
        )));
    }
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    plan.input_shape(&input_name)
        .map_err(PyValueError::new_err)?;
    Ok(TensorHvpScalarFunction { plan, input_name })
}

#[pyfunction]
pub fn tensor_jit_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorJitFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorJitFunction { plan })
}

fn batched_input_specs(
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
) -> PyResult<Vec<(String, Vec<usize>)>> {
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "tensor_vmap_fn requires batch_size > 0",
        ));
    }

    Ok(input_specs
        .into_iter()
        .map(|(name, shape)| {
            let mut batched_shape = Vec::with_capacity(shape.len() + 1);
            batched_shape.push(batch_size);
            batched_shape.extend(shape);
            (name, batched_shape)
        })
        .collect())
}

/// Trace a fixed-size axis-0 vectorized function into one reusable CPU plan.
///
/// Every argument is mapped over its leading dimension. The supplied input
/// shapes describe one example; the compiled plan expects `[batch_size, ..shape]`
/// inputs and returns an output with the same leading batch dimension.
#[pyfunction]
pub fn tensor_vmap_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
) -> PyResult<TensorJitFunction> {
    tensor_jit_fn(py, function, batched_input_specs(input_specs, batch_size)?)
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, device_ordinal = 0))]
pub fn tensor_jit_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    device_ordinal: usize,
) -> PyResult<TensorCudaExecutionPlan> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    traced
        .graph
        .compile_cuda_plan(traced.output.node_id, device_ordinal)
        .map_err(PyValueError::new_err)
}

/// Trace a fixed-size axis-0 vectorized function into one reusable CUDA plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, device_ordinal = 0))]
pub fn tensor_vmap_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    device_ordinal: usize,
) -> PyResult<TensorCudaExecutionPlan> {
    tensor_jit_cuda_fn(
        py,
        function,
        batched_input_specs(input_specs, batch_size)?,
        device_ordinal,
    )
}

/// Trace a fixed-size axis-0 vectorized function into one reusable MLX plan.
#[pyfunction]
pub fn tensor_vmap_mlx_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
) -> PyResult<TensorMlxExecutionPlan> {
    let traced =
        trace_tensor_python_function(py, function, batched_input_specs(input_specs, batch_size)?)?;
    traced
        .graph
        .compile_mlx_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn tensor_vjp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorVjpFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorVjpFunction { plan })
}

#[pyfunction]
pub fn tensor_jvp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
) -> PyResult<TensorJvpFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorJvpFunction { plan })
}

#[pyfunction]
pub fn tensor_jacobian_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    input_name: String,
) -> PyResult<TensorJacobianFunction> {
    if !input_specs.iter().any(|(name, _)| name == &input_name) {
        return Err(PyValueError::new_err(format!(
            "input {:?} is not declared in input_specs",
            input_name
        )));
    }
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorJacobianFunction { plan, input_name })
}

fn extract_cuda_parameter_plans(
    plans: &Bound<'_, PyDict>,
) -> PyResult<Vec<(String, TensorCudaExecutionPlan)>> {
    if plans.is_empty() {
        return Err(PyValueError::new_err(
            "cuda_adam_step requires at least one parameter gradient plan",
        ));
    }
    let mut parameter_plans = Vec::with_capacity(plans.len());
    for (key, value) in plans.iter() {
        let parameter_name = key.cast::<PyString>()?.to_str()?.to_string();
        let plan = value
            .extract::<PyRef<'_, TensorCudaExecutionPlan>>()
            .map_err(|_| {
                PyTypeError::new_err(
                "cuda_adam_step plans must map parameter names to TensorCudaExecutionPlan values",
            )
            })?;
        parameter_plans.push((parameter_name, plan.clone()));
    }
    Ok(parameter_plans)
}

fn retained_cuda_inputs(
    parameter_plans: &[(String, TensorCudaExecutionPlan)],
    retained_input_names: Option<Vec<String>>,
) -> BTreeSet<String> {
    let mut retained = retained_input_names
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeSet<_>>();
    retained.extend(parameter_plans.iter().map(|(name, _)| name.clone()));
    retained
}

fn extract_cuda_vjp_outputs(
    gradients: &Bound<'_, PyDict>,
) -> PyResult<(TensorTraceGraph, Vec<(String, TensorNodeId)>)> {
    if gradients.is_empty() {
        return Err(PyValueError::new_err(
            "cuda_adam_vjp_optimizer requires at least one symbolic VJP result",
        ));
    }
    let mut graph: Option<TensorTraceGraph> = None;
    let mut outputs = Vec::with_capacity(gradients.len());
    for (key, value) in gradients.iter() {
        let parameter_name = key.cast::<PyString>()?.to_str()?.to_string();
        let result = value.extract::<PyRef<'_, TensorTraceResult>>().map_err(|_| {
            PyTypeError::new_err(
                "cuda_adam_vjp_optimizer gradients must map parameter names to TensorTraceResult values",
            )
        })?;
        if let Some(existing) = &graph {
            if !Arc::ptr_eq(&existing.ir, &result.graph.ir) {
                return Err(PyValueError::new_err(
                    "cuda_adam_vjp_optimizer gradients must originate from one symbolic VJP graph",
                ));
            }
        } else {
            graph = Some(result.graph.clone());
        }
        outputs.push((parameter_name, result.output.node_id));
    }
    Ok((
        graph.ok_or_else(|| PyValueError::new_err("missing symbolic VJP graph"))?,
        outputs,
    ))
}

fn execute_cuda_adam_step(
    parameter_plans: &[(String, TensorCudaExecutionPlan)],
    values: &BTreeMap<String, DynamicTensor>,
    retained: &BTreeSet<String>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> Result<(), String> {
    for (parameter_name, plan) in parameter_plans {
        plan.plan.execute_retaining(values, retained)?;
        plan.plan.adam_step_input_from_output(
            parameter_name,
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )?;
    }
    for (source_name, source_plan) in parameter_plans {
        for (_, target_plan) in parameter_plans {
            source_plan
                .plan
                .sync_retained_input_to(source_name, &target_plan.plan, source_name)?;
        }
    }
    Ok(())
}

fn execute_shared_cuda_adam_step(
    shared_plan: &SharedCudaAdamPlan,
    values: &BTreeMap<String, DynamicTensor>,
    retained: &BTreeSet<String>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> Result<(), String> {
    shared_plan
        .plan
        .plan
        .execute_retaining_without_output(values, retained)?;
    for (parameter_name, gradient_node_id) in &shared_plan.gradient_node_ids {
        shared_plan.plan.plan.adam_step_input_from_node(
            parameter_name,
            *gradient_node_id,
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )?;
    }
    Ok(())
}

#[pymethods]
impl TensorCudaAdamOptimizer {
    #[pyo3(signature = (inputs = None))]
    fn step(&mut self, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut retained_inputs = self.retained_inputs.clone();
        if let Some(inputs) = inputs {
            for (name, tensor) in extract_tensor_map(inputs)? {
                self.inputs.insert(name.clone(), tensor);
                if !self.parameter_names.contains(&name) {
                    retained_inputs.remove(&name);
                }
            }
        }
        retained_inputs.extend(self.parameter_names.iter().cloned());
        let result = if let Some(shared_plan) = &self.shared_plan {
            execute_shared_cuda_adam_step(
                shared_plan,
                &self.inputs,
                &retained_inputs,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
            )
        } else {
            execute_cuda_adam_step(
                &self.parameter_plans,
                &self.inputs,
                &retained_inputs,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
            )
        };
        result.map_err(PyValueError::new_err)
    }

    fn parameters(&self) -> PyResult<BTreeMap<String, PyTensor>> {
        if let Some(shared_plan) = &self.shared_plan {
            return self
                .parameter_names
                .iter()
                .map(|name| {
                    let value = shared_plan
                        .plan
                        .plan
                        .retained_input_to_host(name)
                        .map_err(PyValueError::new_err)?;
                    Ok((
                        name.clone(),
                        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
                    ))
                })
                .collect();
        }
        self.parameter_plans
            .iter()
            .map(|(name, plan)| {
                let value = plan
                    .plan
                    .retained_input_to_host(name)
                    .map_err(PyValueError::new_err)?;
                Ok((
                    name.clone(),
                    PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
                ))
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorCudaAdamOptimizer(parameter_count={}, retained_input_count={})",
            self.parameter_names.len(),
            self.retained_inputs.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (plans, inputs, learning_rate, retained_input_names = None, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_adam_optimizer(
    plans: &Bound<'_, PyDict>,
    inputs: &Bound<'_, PyDict>,
    learning_rate: f32,
    retained_input_names: Option<Vec<String>>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> PyResult<TensorCudaAdamOptimizer> {
    let parameter_plans = extract_cuda_parameter_plans(plans)?;
    let parameter_names = parameter_plans
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    Ok(TensorCudaAdamOptimizer {
        retained_inputs: retained_cuda_inputs(&parameter_plans, retained_input_names),
        parameter_plans,
        shared_plan: None,
        parameter_names,
        inputs: extract_tensor_map(inputs)?,
        learning_rate,
        beta1,
        beta2,
        epsilon,
    })
}

#[pyfunction]
#[pyo3(signature = (gradients, inputs, learning_rate, retained_input_names = None, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, device_ordinal = 0))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_adam_vjp_optimizer(
    gradients: &Bound<'_, PyDict>,
    inputs: &Bound<'_, PyDict>,
    learning_rate: f32,
    retained_input_names: Option<Vec<String>>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    device_ordinal: usize,
) -> PyResult<TensorCudaAdamOptimizer> {
    let (graph, outputs) = extract_cuda_vjp_outputs(gradients)?;
    let parameter_names = outputs
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    let output_node_ids = outputs
        .iter()
        .map(|(_, node_id)| *node_id)
        .collect::<Vec<_>>();
    let (plan, remapped_output_node_ids) = graph
        .compile_cuda_multi_plan(&output_node_ids, device_ordinal)
        .map_err(PyValueError::new_err)?;
    let gradient_node_ids = outputs
        .into_iter()
        .map(|(name, _)| name)
        .zip(remapped_output_node_ids)
        .collect::<BTreeMap<_, _>>();
    let mut retained_inputs = retained_input_names
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeSet<_>>();
    retained_inputs.extend(parameter_names.iter().cloned());
    Ok(TensorCudaAdamOptimizer {
        parameter_plans: Vec::new(),
        shared_plan: Some(SharedCudaAdamPlan {
            plan,
            gradient_node_ids,
        }),
        parameter_names,
        inputs: extract_tensor_map(inputs)?,
        retained_inputs,
        learning_rate,
        beta1,
        beta2,
        epsilon,
    })
}

#[pyfunction]
#[pyo3(signature = (plans, inputs, learning_rate, retained_input_names = None, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_adam_step(
    plans: &Bound<'_, PyDict>,
    inputs: &Bound<'_, PyDict>,
    learning_rate: f32,
    retained_input_names: Option<Vec<String>>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> PyResult<()> {
    let parameter_plans = extract_cuda_parameter_plans(plans)?;
    let retained = retained_cuda_inputs(&parameter_plans, retained_input_names);
    let values = extract_tensor_map(inputs)?;
    execute_cuda_adam_step(
        &parameter_plans,
        &values,
        &retained,
        learning_rate,
        beta1,
        beta2,
        epsilon,
    )
    .map_err(PyValueError::new_err)
}

fn extract_tensor_map(inputs: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, DynamicTensor>> {
    let mut tensors = BTreeMap::new();
    for (key, value) in inputs.iter() {
        let key = key.cast::<PyString>()?.to_str()?.to_string();
        let tensor = value.extract::<PyRef<'_, PyTensor>>()?;
        tensors.insert(
            key,
            tensor.to_dynamic_tensor().map_err(PyValueError::new_err)?,
        );
    }
    Ok(tensors)
}
