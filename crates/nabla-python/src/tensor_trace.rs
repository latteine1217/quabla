use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nabla_core::tensor_ir::{DynamicTensor, TensorExecutionPlan, TensorIr, TensorNodeId};
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

#[pyclass(name = "TensorGradScalarFunction", skip_from_py_object)]
pub struct TensorGradScalarFunction {
    plan: TensorExecutionPlan,
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
}

impl TraceTensor {
    fn same_graph(&self, rhs: &Self) -> Result<(), String> {
        if Arc::ptr_eq(&self.graph.ir, &rhs.graph.ir) {
            Ok(())
        } else {
            Err("cannot combine TraceTensor values from different graphs".to_string())
        }
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

    fn tanh(&self) -> PyResult<Self> {
        self.tanh_tensor().map_err(PyValueError::new_err)
    }

    fn exp(&self) -> PyResult<Self> {
        self.exp_tensor().map_err(PyValueError::new_err)
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.reshape_tensor(shape).map_err(PyValueError::new_err)
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

    fn compile_cpu(&self) -> PyResult<TensorCpuExecutionPlan> {
        self.graph
            .compile_cpu_plan(self.output.node_id)
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
