use std::cell::RefCell;
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

use crate::tensor::{parse_axis_indices, parse_tensor_indices, PyTensor, TensorIndex};

fn normalize_reduction_axes(axes: Vec<isize>, rank: usize) -> Result<Vec<usize>, String> {
    let rank = isize::try_from(rank).map_err(|_| "tensor rank exceeds isize".to_string())?;
    let mut axes = axes
        .into_iter()
        .map(|axis| {
            let normalized = if axis < 0 { rank + axis } else { axis };
            if !(0..rank).contains(&normalized) {
                return Err(format!("axis {axis} is out of bounds for rank {rank}"));
            }
            Ok(normalized as usize)
        })
        .collect::<Result<Vec<_>, _>>()?;
    axes.sort_unstable();
    if axes.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("reduction axes must be unique".to_string());
    }
    Ok(axes)
}

fn extract_reduction_axes(axis: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<isize>>> {
    let Some(axis) = axis else {
        return Ok(None);
    };
    if axis.is_none() {
        return Ok(None);
    }
    if let Ok(axis) = axis.extract::<isize>() {
        return Ok(Some(vec![axis]));
    }
    axis.extract::<Vec<isize>>().map(Some).map_err(|_| {
        PyTypeError::new_err("axis must be None, an integer, or a sequence of integers")
    })
}

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
    // vmap canonicalizes the mapped dimension to axis zero while tracing.
    // Normal traces have no mapped axis.
    batch_axis: Option<usize>,
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
    shared_plan_includes_loss: bool,
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

#[pyclass(name = "TensorBatchJitFunction", unsendable)]
pub struct TensorBatchJitFunction {
    function: Py<PyAny>,
    input_names: Vec<String>,
    input_axes: Vec<Option<isize>>,
    max_specializations: usize,
    static_shapes: RefCell<Option<Vec<Vec<usize>>>>,
    plans: RefCell<BTreeMap<usize, TensorExecutionPlan>>,
}

#[derive(Clone, Debug)]
struct VmapSignature {
    input_names: Vec<String>,
    in_axes: Vec<Option<usize>>,
    out_axis: usize,
}

#[pyclass(name = "TensorVmapFunction", skip_from_py_object)]
pub struct TensorVmapFunction {
    plan: TensorExecutionPlan,
    signature: VmapSignature,
}

#[pyclass(name = "TensorVmapCudaFunction", skip_from_py_object)]
pub struct TensorVmapCudaFunction {
    plan: TensorCudaExecutionPlan,
    signature: VmapSignature,
}

#[pyclass(name = "TensorVmapMlxFunction", skip_from_py_object)]
pub struct TensorVmapMlxFunction {
    plan: TensorMlxExecutionPlan,
    signature: VmapSignature,
}

#[pyclass(name = "TensorVmapVjpFunction", skip_from_py_object)]
pub struct TensorVmapVjpFunction {
    plan: TensorExecutionPlan,
    signature: VmapSignature,
}

#[pyclass(name = "TensorVmapJvpFunction", skip_from_py_object)]
pub struct TensorVmapJvpFunction {
    plan: TensorExecutionPlan,
    signature: VmapSignature,
}

#[pyclass(name = "TensorVmapCudaVjpFunction", skip_from_py_object)]
pub struct TensorVmapCudaVjpFunction {
    plan: TensorCudaExecutionPlan,
    output_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
    signature: VmapSignature,
    cotangent_name: String,
}

#[pyclass(name = "TensorCudaValueAndGradFunction", skip_from_py_object)]
pub struct TensorCudaValueAndGradFunction {
    plan: TensorCudaExecutionPlan,
    loss_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
    cotangent_name: String,
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
            batch_axis: None,
        })
    }

    fn add_batched_input(&self, name: &str, shape: Vec<usize>) -> Result<TraceTensor, String> {
        let mut tensor = self.add_input(name, shape)?;
        tensor.batch_axis = Some(0);
        Ok(tensor)
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
            batch_axis: None,
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
    fn from_node(
        graph: TensorTraceGraph,
        node_id: TensorNodeId,
        shape: Vec<usize>,
        batch_axis: Option<usize>,
    ) -> Self {
        Self {
            graph,
            node_id,
            shape,
            batch_axis,
        }
    }

    fn merged_batch_axis(tensors: &[&Self]) -> Result<Option<usize>, String> {
        let mut batch_axis = None;
        for tensor in tensors {
            match (batch_axis, tensor.batch_axis) {
                (None, axis) => batch_axis = axis,
                (Some(lhs), Some(rhs)) if lhs == rhs => {}
                (Some(_), Some(_)) => {
                    return Err("cannot combine vmap tensors with different batch axes".to_string())
                }
                (Some(_), None) => {}
            }
        }
        Ok(batch_axis)
    }

    fn example_axis(&self, axis: isize) -> Result<isize, String> {
        let rank = self.shape.len() - usize::from(self.batch_axis.is_some());
        let normalized = if axis < 0 { axis + rank as isize } else { axis };
        if !(0..rank as isize).contains(&normalized) {
            return Err(format!("axis {axis} is out of bounds for rank {rank}"));
        }
        Ok(normalized + self.batch_axis.map_or(0, |_| 1) as isize)
    }

    fn example_shape(&self, shape: Vec<usize>) -> Vec<usize> {
        match self.batch_axis {
            Some(axis) => {
                debug_assert_eq!(axis, 0);
                let mut batched = Vec::with_capacity(shape.len() + 1);
                batched.push(self.shape[axis]);
                batched.extend(shape);
                batched
            }
            None => shape,
        }
    }

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
        let batch_axis = Self::merged_batch_axis(&tensors.iter().collect::<Vec<_>>())?;
        let batch_extent = tensors
            .iter()
            .find_map(|tensor| tensor.batch_axis.map(|axis| tensor.shape[axis]));
        let mut ir = first
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let axis = if batch_axis.is_some() { axis + 1 } else { axis };
        let inputs = tensors
            .iter()
            .map(|tensor| match (batch_extent, tensor.batch_axis) {
                (Some(batch_extent), None) => {
                    let mut shape = Vec::with_capacity(tensor.shape.len() + 1);
                    shape.push(batch_extent);
                    shape.extend(&tensor.shape);
                    ir.broadcast_to(tensor.node_id, shape)
                }
                _ => Ok(tensor.node_id),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let node_id = ir.concat(inputs, axis as isize)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            first.graph.clone(),
            node_id,
            shape,
            batch_axis,
        ))
    }

    pub fn try_stack(tensors: &[Self], axis: isize) -> Result<Self, String> {
        let first = tensors
            .first()
            .ok_or_else(|| "stack requires at least one TraceTensor".to_string())?;
        if tensors.iter().any(|tensor| tensor.shape != first.shape) {
            return Err("stack requires TraceTensor values with identical shapes".to_string());
        }
        let example_rank = first.shape.len() - usize::from(first.batch_axis.is_some());
        let rank = example_rank + 1;
        let normalized_axis = if axis < 0 { axis + rank as isize } else { axis };
        let axis = usize::try_from(normalized_axis)
            .ok()
            .filter(|axis| *axis < rank)
            .ok_or_else(|| format!("axis {axis} is out of bounds for rank {rank}"))?;
        let reshaped = tensors
            .iter()
            .map(|tensor| {
                let mut shape = tensor.shape[usize::from(tensor.batch_axis.is_some())..].to_vec();
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
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            Self::merged_batch_axis(&[self, rhs])?,
        ))
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
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn scalar_tensor(&self, value: f64) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.scalar_constant(value);
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(self.graph.clone(), node_id, shape, None))
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
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn sum_tensor(&self, axis: Option<isize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = match (self.batch_axis, axis) {
            (Some(_), None) => {
                let mut node_id = self.node_id;
                for axis in (1..self.shape.len()).rev() {
                    node_id = ir.sum_axis(node_id, axis as isize)?;
                }
                node_id
            }
            (_, Some(axis)) => ir.sum_axis(self.node_id, self.example_axis(axis)?)?,
            (None, None) => ir.sum(self.node_id)?,
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
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
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            Self::merged_batch_axis(&[self, rhs])?,
        ))
    }

    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, String> {
        self.matmul_tensor(rhs)
    }

    fn solve_tensor(&self, rhs: &Self) -> Result<Self, String> {
        self.same_graph(rhs)?;
        if self.batch_axis.is_some() || rhs.batch_axis.is_some() {
            return Err("solve does not yet support vmap-batched tensors".to_string());
        }
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.solve(self.node_id, rhs.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(self.graph.clone(), node_id, shape, None))
    }

    pub fn try_solve(&self, rhs: &Self) -> Result<Self, String> {
        self.solve_tensor(rhs)
    }

    fn solve_triangular_tensor(
        &self,
        rhs: &Self,
        lower: bool,
        transpose: bool,
    ) -> Result<Self, String> {
        let matrix = self.triangular_tensor(lower)?;
        let matrix = if transpose {
            matrix.transpose_tensor(None)?
        } else {
            matrix
        };
        matrix.solve_tensor(rhs)
    }

    #[allow(clippy::needless_range_loop)]
    fn cholesky_tensor(&self) -> Result<Self, String> {
        if self.batch_axis.is_some() || self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(format!(
                "cholesky reference tracing requires a square unbatched rank-2 tensor, got {:?}",
                self.shape
            ));
        }
        let n = self.shape[0];
        let mut rows: Vec<Vec<Self>> = Vec::with_capacity(n);
        for row in 0..n {
            let mut current = Vec::with_capacity(n);
            for column in 0..n {
                if column > row {
                    current.push(self.scalar_tensor(0.0)?);
                    continue;
                }
                let mut reduced =
                    self.index_tensor(&[TensorIndex::Integer(row), TensorIndex::Integer(column)])?;
                for inner in 0..column {
                    let column_value = if row == column {
                        &current[inner]
                    } else {
                        &rows[column][inner]
                    };
                    let product = current[inner].binary(column_value, "mul")?;
                    reduced = reduced.binary(&product, "sub")?;
                }
                let value = if row == column {
                    reduced.sqrt_tensor()?
                } else {
                    reduced.binary(&rows[column][column], "div")?
                };
                current.push(value);
            }
            rows.push(current);
        }
        let rows = rows
            .into_iter()
            .map(|row| {
                let entries = row
                    .into_iter()
                    .map(|value| value.reshape_tensor(vec![1]))
                    .collect::<Result<Vec<_>, _>>()?;
                Self::try_concat(&entries, 0)?.reshape_tensor(vec![1, n])
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_concat(&rows, 0)
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
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            Self::merged_batch_axis(&[self, on_true, on_false])?,
        ))
    }

    fn maximum_tensor(&self, rhs: &Self) -> Result<Self, String> {
        let mask = self.binary(rhs, "greater")?;
        mask.where_tensor(self, rhs)
    }

    fn maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let mask = self.scalar_binary(rhs, "greater")?;
        let rhs = self.scalar_tensor(rhs)?;
        mask.where_tensor(self, &rhs)
    }

    fn minimum_tensor(&self, rhs: &Self) -> Result<Self, String> {
        let mask = rhs.binary(self, "greater")?;
        mask.where_tensor(self, rhs)
    }

    fn minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let mask = self.scalar_left_binary(rhs, "greater")?;
        let rhs = self.scalar_tensor(rhs)?;
        mask.where_tensor(self, &rhs)
    }

    fn tanh_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.tanh(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn relu_tensor(&self) -> Result<Self, String> {
        self.maximum_scalar(0.0)
    }

    fn abs_tensor(&self) -> Result<Self, String> {
        let mask = self.scalar_binary(0.0, "greater")?;
        let negative = self.scalar_left_binary(0.0, "sub")?;
        mask.where_tensor(self, &negative)
    }

    fn sigmoid_tensor(&self) -> Result<Self, String> {
        self.scalar_binary(-1.0, "mul")?
            .exp_tensor()?
            .scalar_binary(1.0, "add")?
            .scalar_left_binary(1.0, "div")
    }

    fn softplus_tensor(&self) -> Result<Self, String> {
        let linear = self.maximum_scalar(0.0)?;
        let correction = self
            .abs_tensor()?
            .scalar_binary(-1.0, "mul")?
            .exp_tensor()?
            .scalar_binary(1.0, "add")?
            .log_tensor()?;
        linear.binary(&correction, "add")
    }

    fn triangular_tensor(&self, lower: bool) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.triangular(self.node_id, lower)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn exp_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.exp(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn reshape_tensor(&self, shape: Vec<usize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.reshape(self.node_id, self.example_shape(shape))?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn slice_tensor(&self, axis: isize, start: usize, stop: usize) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.slice_axis(self.node_id, self.example_axis(axis)?, start, stop)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn pad_slice_tensor(
        &self,
        shape: Vec<usize>,
        axis: isize,
        start: usize,
    ) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.pad_slice(
            self.node_id,
            self.example_shape(shape),
            usize::try_from(self.example_axis(axis)?)
                .map_err(|_| "normalized tensor axis is negative".to_string())?,
            start,
        )?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn gather_tensor(&self, indices: &[usize], axis: isize) -> Result<Self, String> {
        let actual_axis = usize::try_from(self.example_axis(axis)?)
            .map_err(|_| "normalized tensor axis is negative".to_string())?;
        if indices.is_empty() {
            return Err("gather indices must not be empty".to_string());
        }
        if indices
            .iter()
            .any(|index| *index >= self.shape[actual_axis])
        {
            return Err(format!(
                "gather index is out of bounds for axis {axis} with extent {}",
                self.shape[actual_axis]
            ));
        }
        let gathered = indices
            .iter()
            .map(|index| self.slice_tensor(axis, *index, index + 1))
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_concat(
            &gathered,
            actual_axis - usize::from(self.batch_axis.is_some()),
        )
    }

    fn scatter_add_tensor(
        &self,
        indices: &[usize],
        updates: &Self,
        axis: isize,
    ) -> Result<Self, String> {
        self.same_graph(updates)?;
        let actual_axis = usize::try_from(self.example_axis(axis)?)
            .map_err(|_| "normalized tensor axis is negative".to_string())?;
        if indices.is_empty() {
            return Err("scatter indices must not be empty".to_string());
        }
        if indices
            .iter()
            .any(|index| *index >= self.shape[actual_axis])
        {
            return Err(format!(
                "scatter index is out of bounds for axis {axis} with extent {}",
                self.shape[actual_axis]
            ));
        }
        let mut expected_shape = self.shape.clone();
        expected_shape[actual_axis] = indices.len();
        if updates.shape != expected_shape || updates.batch_axis != self.batch_axis {
            return Err(format!(
                "scatter updates shape {:?} is incompatible with base shape {:?}, axis {axis}, and {} indices",
                updates.shape,
                self.shape,
                indices.len()
            ));
        }
        let batch_offset = usize::from(self.batch_axis.is_some());
        let mut output = self.clone();
        for (update_index, destination) in indices.iter().copied().enumerate() {
            let update = updates.slice_tensor(axis, update_index, update_index + 1)?;
            let padded =
                update.pad_slice_tensor(self.shape[batch_offset..].to_vec(), axis, destination)?;
            output = output.binary(&padded, "add")?;
        }
        Ok(output)
    }

    fn index_tensor(&self, indices: &[TensorIndex]) -> Result<Self, String> {
        let mut output = self.clone();
        let mut axis = 0;
        for index in indices {
            match *index {
                TensorIndex::Slice { start, stop } => {
                    output = output.slice_tensor(axis as isize, start, stop)?;
                    axis += 1;
                }
                TensorIndex::Integer(index) => {
                    output = output.slice_tensor(axis as isize, index, index + 1)?;
                    let batch_offset = usize::from(output.batch_axis.is_some());
                    let mut shape = output.shape[batch_offset..].to_vec();
                    shape.remove(axis);
                    output = output.reshape_tensor(shape)?;
                }
            }
        }
        Ok(output)
    }

    fn broadcast_to_tensor(&self, shape: Vec<usize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.broadcast_to(self.node_id, self.example_shape(shape))?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn mean_tensor(&self, axis: Option<isize>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = match (self.batch_axis, axis) {
            (Some(_), None) => {
                let mut node_id = self.node_id;
                for axis in (1..self.shape.len()).rev() {
                    node_id = ir.mean_axis(node_id, axis as isize)?;
                }
                node_id
            }
            (_, Some(axis)) => ir.mean_axis(self.node_id, self.example_axis(axis)?)?,
            (None, None) => ir.mean(self.node_id)?,
        };
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn reduce_axes_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        mean: bool,
    ) -> Result<Self, String> {
        let example_rank = self.shape.len() - usize::from(self.batch_axis.is_some());
        let Some(axes) = axes else {
            let reduced = if mean {
                self.mean_tensor(None)?
            } else {
                self.sum_tensor(None)?
            };
            return if keepdims {
                reduced.reshape_tensor(vec![1; example_rank])
            } else {
                Ok(reduced)
            };
        };
        let mut axes = normalize_reduction_axes(axes, example_rank)?;
        axes.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        let mut reduced = self.clone();
        for axis in axes {
            reduced = if mean {
                reduced.mean_tensor(Some(axis as isize))?
            } else {
                reduced.sum_tensor(Some(axis as isize))?
            };
            if keepdims {
                let batch_offset = usize::from(reduced.batch_axis.is_some());
                let mut shape = reduced.shape[batch_offset..].to_vec();
                shape.insert(axis, 1);
                reduced = reduced.reshape_tensor(shape)?;
            }
        }
        Ok(reduced)
    }

    fn norm_tensor(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.powi_tensor(2)?
            .reduce_axes_tensor(axes, keepdims, false)?
            .sqrt_tensor()
    }

    fn extrema_axes_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        maximum: bool,
    ) -> Result<Self, String> {
        let example_rank = self.shape.len() - usize::from(self.batch_axis.is_some());
        let Some(axes) = axes else {
            let mut reduced = self.clone();
            for _ in 0..example_rank {
                reduced = reduced.extrema_axis_tensor(-1, maximum)?;
            }
            return if keepdims {
                reduced.reshape_tensor(vec![1; example_rank])
            } else {
                Ok(reduced)
            };
        };
        let mut axes = normalize_reduction_axes(axes, example_rank)?;
        axes.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        let mut reduced = self.clone();
        for axis in axes {
            reduced = reduced.extrema_axis_tensor(axis as isize, maximum)?;
            if keepdims {
                let batch_offset = usize::from(reduced.batch_axis.is_some());
                let mut shape = reduced.shape[batch_offset..].to_vec();
                shape.insert(axis, 1);
                reduced = reduced.reshape_tensor(shape)?;
            }
        }
        Ok(reduced)
    }

    fn extrema_axis_tensor(&self, axis: isize, maximum: bool) -> Result<Self, String> {
        let batch_offset = usize::from(self.batch_axis.is_some());
        let actual_axis = usize::try_from(self.example_axis(axis)?)
            .map_err(|_| "normalized tensor axis is negative".to_string())?;
        let axis = actual_axis - batch_offset;
        let axis_extent = self.shape[actual_axis];
        if axis_extent == 0 {
            return Err("max/min reduction requires a non-empty reduced axis".to_string());
        }
        let remove_axis = |tensor: Self| {
            let mut shape = tensor.shape[batch_offset..].to_vec();
            shape.remove(axis);
            tensor.reshape_tensor(shape)
        };
        let mut reduced = remove_axis(self.slice_tensor(axis as isize, 0, 1)?)?;
        for index in 1..axis_extent {
            let candidate = remove_axis(self.slice_tensor(axis as isize, index, index + 1)?)?;
            reduced = if maximum {
                reduced.maximum_tensor(&candidate)?
            } else {
                reduced.minimum_tensor(&candidate)?
            };
        }
        Ok(reduced)
    }

    fn sin_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.sin(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn cos_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.cos(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn powi_tensor(&self, exponent: u32) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.powi(self.node_id, exponent)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn transpose_tensor(&self, axes: Option<Vec<isize>>) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let axes = match (self.batch_axis, axes) {
            (Some(_), Some(axes)) => Some(
                axes.into_iter()
                    .map(|axis| self.example_axis(axis))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            (Some(_), None) => Some(
                std::iter::once(0)
                    .chain((1..self.shape.len()).rev().map(|axis| axis as isize))
                    .collect(),
            ),
            (None, axes) => axes,
        };
        let node_id = ir.transpose(self.node_id, axes)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn log_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.log(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn sqrt_tensor(&self) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.sqrt(self.node_id)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
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
                batch_axis: self.output.batch_axis,
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
                            batch_axis: self.output.batch_axis,
                        },
                    ),
                ))
            })
            .collect()
    }

    fn symbolic_vjp_graph(
        &self,
        cotangent_name: &str,
    ) -> Result<
        (
            TensorTraceGraph,
            TensorNodeId,
            BTreeMap<String, TensorNodeId>,
        ),
        String,
    > {
        let ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let transformed = ir.symbolic_vjp(self.output.node_id, cotangent_name)?;
        Ok((
            TensorTraceGraph {
                ir: Arc::new(Mutex::new(transformed.graph)),
            },
            transformed.value,
            transformed.gradients,
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

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn sum(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.reduce_axes_tensor(extract_reduction_axes(axis)?, keepdims, false)
            .map_err(PyValueError::new_err)
    }

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.matmul_tensor(rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.__matmul__(rhs)
    }

    fn solve(&self, rhs: &Self) -> PyResult<Self> {
        self.solve_tensor(rhs).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (rhs, lower = true, transpose = false))]
    fn solve_triangular(&self, rhs: &Self, lower: bool, transpose: bool) -> PyResult<Self> {
        self.solve_triangular_tensor(rhs, lower, transpose)
            .map_err(PyValueError::new_err)
    }

    fn cholesky(&self) -> PyResult<Self> {
        self.cholesky_tensor().map_err(PyValueError::new_err)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "greater")
    }

    fn maximum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, TraceTensor>>() {
            return self.maximum_tensor(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.maximum_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected TraceTensor or numeric scalar operand",
        ))
    }

    fn minimum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = rhs.extract::<PyRef<'_, TraceTensor>>() {
            return self.minimum_tensor(&rhs).map_err(PyValueError::new_err);
        }
        if let Ok(rhs) = rhs.extract::<f64>() {
            return self.minimum_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected TraceTensor or numeric scalar operand",
        ))
    }

    fn where_select(&self, on_true: &Self, on_false: &Self) -> PyResult<Self> {
        self.where_tensor(on_true, on_false)
            .map_err(PyValueError::new_err)
    }

    fn tanh(&self) -> PyResult<Self> {
        self.tanh_tensor().map_err(PyValueError::new_err)
    }

    fn relu(&self) -> PyResult<Self> {
        self.relu_tensor().map_err(PyValueError::new_err)
    }

    fn abs(&self) -> PyResult<Self> {
        self.abs_tensor().map_err(PyValueError::new_err)
    }

    fn sigmoid(&self) -> PyResult<Self> {
        self.sigmoid_tensor().map_err(PyValueError::new_err)
    }

    fn softplus(&self) -> PyResult<Self> {
        self.softplus_tensor().map_err(PyValueError::new_err)
    }

    fn tril(&self) -> PyResult<Self> {
        self.triangular_tensor(true).map_err(PyValueError::new_err)
    }

    fn triu(&self) -> PyResult<Self> {
        self.triangular_tensor(false).map_err(PyValueError::new_err)
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

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch_offset = usize::from(self.batch_axis.is_some());
        let shape = &self.shape[batch_offset..];
        let indices = parse_tensor_indices(index, shape.len(), shape)?;
        self.index_tensor(&indices).map_err(PyValueError::new_err)
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.broadcast_to_tensor(shape)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn mean(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.reduce_axes_tensor(extract_reduction_axes(axis)?, keepdims, true)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn norm(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.norm_tensor(extract_reduction_axes(axis)?, keepdims)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn max(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.extrema_axes_tensor(extract_reduction_axes(axis)?, keepdims, true)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn min(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.extrema_axes_tensor(extract_reduction_axes(axis)?, keepdims, false)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (indices, axis = 0))]
    fn gather(&self, indices: &Bound<'_, PyAny>, axis: isize) -> PyResult<Self> {
        let actual_axis = usize::try_from(self.example_axis(axis).map_err(PyValueError::new_err)?)
            .map_err(|_| PyValueError::new_err("normalized tensor axis is negative"))?;
        let indices = parse_axis_indices(indices, self.shape[actual_axis])?;
        self.gather_tensor(&indices, axis)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (indices, updates, axis = 0))]
    fn scatter_add(
        &self,
        indices: &Bound<'_, PyAny>,
        updates: &Self,
        axis: isize,
    ) -> PyResult<Self> {
        let actual_axis = usize::try_from(self.example_axis(axis).map_err(PyValueError::new_err)?)
            .map_err(|_| PyValueError::new_err("normalized tensor axis is negative"))?;
        let indices = parse_axis_indices(indices, self.shape[actual_axis])?;
        self.scatter_add_tensor(&indices, updates, axis)
            .map_err(PyValueError::new_err)
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
            item.set_item("dtype", node.dtype)?;
            item.set_item("layout", node.layout)?;
            item.set_item("placement", node.placement)?;
            item.set_item("effect", node.effect)?;
            item.set_item("alias_of", node.alias_of)?;
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

    fn buffer_plan(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let buffer_plan = self.plan.buffer_plan().map_err(PyValueError::new_err)?;
        let result = PyDict::new(py);
        let slots = PyList::empty(py);
        for slot in buffer_plan.slots {
            let item = PyDict::new(py);
            item.set_item("id", slot.id)?;
            item.set_item("element_count", slot.element_count)?;
            slots.append(item)?;
        }
        result.set_item("slots", slots)?;
        result.set_item("node_slots", buffer_plan.node_slots)?;
        result.set_item("node_aliases", buffer_plan.node_aliases)?;
        result.set_item("output_backing_node_id", buffer_plan.output_backing_node_id)?;
        Ok(result.into())
    }

    fn fusion_regions(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let regions = PyList::empty(py);
        for region in self.plan.fusion_regions() {
            let item = PyDict::new(py);
            item.set_item("output_node_id", region.output_node_id)?;
            item.set_item("node_ids", region.node_ids)?;
            item.set_item("input_node_ids", region.input_node_ids)?;
            regions.append(item)?;
        }
        Ok(regions.into())
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
impl TensorVmapMlxFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.plan.node_count()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        "mlx"
    }

    fn __call__(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = MlxBackend
            .execute(&self.plan.plan, &vmap_inputs(&self.signature, inputs)?)
            .map_err(PyValueError::new_err)?;
        vmap_output(&self.signature, value)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapMlxFunction(node_count={})",
            self.plan.plan.node_count()
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

    #[getter]
    fn fused_region_count(&self) -> usize {
        self.plan.fused_region_count()
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
impl TensorVmapCudaFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.plan.node_count()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        if self.plan.plan.uses_cublas() {
            "cublas"
        } else {
            "nvrtc"
        }
    }

    fn __call__(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = self
            .plan
            .plan
            .execute(&vmap_inputs(&self.signature, inputs)?)
            .map_err(PyValueError::new_err)?;
        vmap_output(&self.signature, value)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapCudaFunction(node_count={})",
            self.plan.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorVmapCudaVjpFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.plan.node_count()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        if self.plan.plan.uses_cublas() {
            "cublas"
        } else {
            "nvrtc"
        }
    }

    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let mut inputs = vmap_inputs(&self.signature, values)?;
        let cotangent = move_axis(
            output_cotangent
                .to_dynamic_tensor()
                .map_err(PyValueError::new_err)?,
            self.signature.out_axis,
            0,
        )
        .map_err(PyValueError::new_err)?;
        inputs.insert(self.cotangent_name.clone(), cotangent);
        self.plan
            .plan
            .execute_retaining_without_output(&inputs, &BTreeSet::new())
            .map_err(PyValueError::new_err)?;
        let output = self
            .plan
            .plan
            .computed_node_to_host(self.output_node_id)
            .map_err(PyValueError::new_err)?;
        let gradients = self
            .gradient_node_ids
            .iter()
            .map(|(name, node_id)| {
                self.plan
                    .plan
                    .computed_node_to_host(*node_id)
                    .map(|gradient| (name.clone(), gradient))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok((
            vmap_output(&self.signature, output)?,
            vmap_gradient_outputs(&self.signature, gradients)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapCudaVjpFunction(node_count={})",
            self.plan.plan.node_count()
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
impl TensorBatchJitFunction {
    fn __call__(&self, py: Python<'_>, values: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let inputs = extract_tensor_map(values)?;
        if inputs.len() != self.input_names.len()
            || self
                .input_names
                .iter()
                .any(|name| !inputs.contains_key(name))
        {
            return Err(PyValueError::new_err(format!(
                "tensor_jit_batch_fn expects exactly inputs {:?}",
                self.input_names
            )));
        }
        let mut batch_size = None;
        let mut static_shapes = Vec::with_capacity(self.input_names.len());
        let mut input_specs = Vec::with_capacity(self.input_names.len());
        for ((name, axis), tensor) in self
            .input_names
            .iter()
            .zip(&self.input_axes)
            .zip(self.input_names.iter().map(|name| &inputs[name]))
        {
            let shape = tensor.shape().to_vec();
            let static_shape = if let Some(axis) = axis {
                let normalized = normalize_batch_axis(*axis, shape.len(), name)?;
                let extent = shape[normalized];
                match batch_size {
                    Some(existing) if existing != extent => {
                        return Err(PyValueError::new_err(format!(
                            "mapped input {name:?} has batch size {extent}, expected {existing}"
                        )));
                    }
                    None => batch_size = Some(extent),
                    _ => {}
                }
                shape
                    .iter()
                    .enumerate()
                    .filter_map(|(index, extent)| (index != normalized).then_some(*extent))
                    .collect()
            } else {
                shape.clone()
            };
            static_shapes.push(static_shape);
            input_specs.push((name.clone(), shape));
        }
        let batch_size = batch_size.ok_or_else(|| {
            PyValueError::new_err("tensor_jit_batch_fn requires at least one mapped input axis")
        })?;
        let mut expected_shapes = self.static_shapes.borrow_mut();
        if let Some(expected) = expected_shapes.as_ref() {
            if expected != &static_shapes {
                return Err(PyValueError::new_err(format!(
                    "tensor_jit_batch_fn only specializes mapped batch axes; expected non-batch shapes {expected:?}, got {static_shapes:?}"
                )));
            }
        } else {
            *expected_shapes = Some(static_shapes);
        }
        drop(expected_shapes);

        let cached_plan = self.plans.borrow().get(&batch_size).cloned();
        let plan = if let Some(plan) = cached_plan {
            plan.clone()
        } else {
            if self.plans.borrow().len() >= self.max_specializations {
                return Err(PyValueError::new_err(format!(
                    "tensor_jit_batch_fn reached max_specializations={} before batch size {batch_size}",
                    self.max_specializations
                )));
            }
            let traced = trace_tensor_python_function(py, self.function.bind(py), input_specs)?;
            let plan = traced
                .graph
                .compile_cpu_plan(traced.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan;
            self.plans.borrow_mut().insert(batch_size, plan.clone());
            plan
        };
        PyTensor::from_dynamic_tensor(plan.evaluate(&inputs).map_err(PyValueError::new_err)?)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn specialization_count(&self) -> usize {
        self.plans.borrow().len()
    }

    #[getter]
    fn max_specializations(&self) -> usize {
        self.max_specializations
    }
}

#[pymethods]
impl TensorVmapFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    fn __call__(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let value = self
            .plan
            .evaluate(&vmap_inputs(&self.signature, inputs)?)
            .map_err(PyValueError::new_err)?;
        vmap_output(&self.signature, value)
    }

    fn __repr__(&self) -> String {
        format!("TensorVmapFunction(node_count={})", self.plan.node_count())
    }
}

#[pymethods]
impl TensorVmapVjpFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = vmap_inputs(&self.signature, values)?;
        let output_cotangent = move_axis(
            output_cotangent
                .to_dynamic_tensor()
                .map_err(PyValueError::new_err)?,
            self.signature.out_axis,
            0,
        )
        .map_err(PyValueError::new_err)?;
        let (value, gradients) = self
            .plan
            .value_and_vjp(&inputs, output_cotangent)
            .map_err(PyValueError::new_err)?;
        let gradients = vmap_gradient_outputs(&self.signature, gradients)?;
        Ok((vmap_output(&self.signature, value)?, gradients))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapVjpFunction(node_count={})",
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorVmapJvpFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (value, tangent) = self
            .plan
            .jvp(
                &vmap_inputs(&self.signature, values)?,
                &vmap_inputs(&self.signature, input_tangents)?,
            )
            .map_err(PyValueError::new_err)?;
        Ok((
            vmap_output(&self.signature, value)?,
            vmap_output(&self.signature, tangent)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapJvpFunction(node_count={})",
            self.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorCudaValueAndGradFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let mut inputs = extract_tensor_map(values)?;
        inputs.insert(
            self.cotangent_name.clone(),
            DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
        );
        self.plan
            .plan
            .execute_retaining_without_output(&inputs, &BTreeSet::new())
            .map_err(PyValueError::new_err)?;
        let loss = self
            .plan
            .plan
            .computed_node_to_host(self.loss_node_id)
            .map_err(PyValueError::new_err)?;
        let gradients = self
            .gradient_node_ids
            .iter()
            .map(|(name, node_id)| {
                self.plan
                    .plan
                    .computed_node_to_host(*node_id)
                    .and_then(PyTensor::from_dynamic_tensor)
                    .map(|tensor| (name.clone(), tensor))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok((
            PyTensor::from_dynamic_tensor(loss).map_err(PyValueError::new_err)?,
            gradients,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorCudaValueAndGradFunction(parameter_count={}, node_count={})",
            self.gradient_node_ids.len(),
            self.plan.plan.node_count()
        )
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

fn normalize_vmap_axis(axis: isize, rank: usize, label: &str) -> PyResult<usize> {
    let normalized = if axis < 0 { axis + rank as isize } else { axis };
    usize::try_from(normalized)
        .ok()
        .filter(|axis| *axis < rank)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{label} axis {axis} is out of bounds for rank {rank}"
            ))
        })
}

fn make_vmap_signature(
    input_specs: &[(String, Vec<usize>)],
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
    output_rank: usize,
) -> PyResult<VmapSignature> {
    let in_axes = in_axes.unwrap_or_else(|| vec![Some(0); input_specs.len()]);
    if in_axes.len() != input_specs.len() {
        return Err(PyValueError::new_err(format!(
            "in_axes has length {}, but function has {} inputs",
            in_axes.len(),
            input_specs.len()
        )));
    }
    let in_axes = input_specs
        .iter()
        .zip(in_axes)
        .map(|((name, shape), axis)| match axis {
            Some(axis) => normalize_vmap_axis(axis, shape.len() + 1, name).map(Some),
            None => Ok(None),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(VmapSignature {
        input_names: input_specs.iter().map(|(name, _)| name.clone()).collect(),
        in_axes,
        out_axis: normalize_vmap_axis(out_axis, output_rank, "out_axes")?,
    })
}

fn trace_tensor_vmap_python_function(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<(TensorTraceResult, VmapSignature)> {
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "tensor_vmap_fn requires batch_size > 0",
        ));
    }
    let input_axes = in_axes
        .clone()
        .unwrap_or_else(|| vec![Some(0); input_specs.len()]);
    if input_axes.len() != input_specs.len() {
        return Err(PyValueError::new_err(format!(
            "in_axes has length {}, but function has {} inputs",
            input_axes.len(),
            input_specs.len()
        )));
    }
    let graph = TensorTraceGraph::new();
    let mut inputs = Vec::with_capacity(input_specs.len());
    for ((name, shape), axis) in input_specs.iter().zip(&input_axes) {
        let tensor = if axis.is_some() {
            let mut batched_shape = Vec::with_capacity(shape.len() + 1);
            batched_shape.push(batch_size);
            batched_shape.extend(shape.iter().copied());
            graph.add_batched_input(name, batched_shape)
        } else {
            graph.add_input(name, shape.clone())
        }
        .map_err(PyValueError::new_err)?;
        inputs.push(tensor);
    }
    let args = PyTuple::new(py, inputs)?;
    let output: TraceTensor = function
        .call1(args)?
        .extract()
        .map_err(|_| PyTypeError::new_err("tensor_vmap_fn function must return a TraceTensor"))?;
    if !Arc::ptr_eq(&graph.ir, &output.graph.ir) {
        return Err(PyValueError::new_err(
            "tensor_vmap_fn function returned a tensor from a different graph",
        ));
    }
    let output = if output.batch_axis.is_none() {
        let mut shape = Vec::with_capacity(output.shape.len() + 1);
        shape.push(batch_size);
        shape.extend(&output.shape);
        let node_id = graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .broadcast_to(output.node_id, shape.clone())
            .map_err(PyValueError::new_err)?;
        TraceTensor::from_node(graph.clone(), node_id, shape, Some(0))
    } else {
        output
    };
    let signature = make_vmap_signature(&input_specs, in_axes, out_axis, output.shape.len())?;
    Ok((TensorTraceResult::new(graph, output), signature))
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

#[pyfunction]
#[pyo3(signature = (function, input_names, in_axes = None, batch_axis = 0, max_specializations = 4))]
pub fn tensor_jit_batch_fn(
    function: Py<PyAny>,
    input_names: Vec<String>,
    in_axes: Option<Vec<Option<isize>>>,
    batch_axis: isize,
    max_specializations: usize,
) -> PyResult<TensorBatchJitFunction> {
    if input_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_jit_batch_fn requires at least one input name",
        ));
    }
    if max_specializations == 0 {
        return Err(PyValueError::new_err(
            "tensor_jit_batch_fn max_specializations must be positive",
        ));
    }
    let input_axes = in_axes.unwrap_or_else(|| vec![Some(batch_axis); input_names.len()]);
    if input_axes.len() != input_names.len() {
        return Err(PyValueError::new_err(format!(
            "tensor_jit_batch_fn in_axes has length {}, expected {}",
            input_axes.len(),
            input_names.len()
        )));
    }
    if !input_axes.iter().any(Option::is_some) {
        return Err(PyValueError::new_err(
            "tensor_jit_batch_fn requires at least one mapped input axis",
        ));
    }
    Ok(TensorBatchJitFunction {
        function,
        input_names,
        input_axes,
        max_specializations,
        static_shapes: RefCell::new(None),
        plans: RefCell::new(BTreeMap::new()),
    })
}

fn normalize_batch_axis(axis: isize, rank: usize, name: &str) -> PyResult<usize> {
    let rank = isize::try_from(rank)
        .map_err(|_| PyValueError::new_err(format!("input {name:?} rank exceeds isize")))?;
    let normalized = if axis < 0 { rank + axis } else { axis };
    if !(0..rank).contains(&normalized) {
        return Err(PyValueError::new_err(format!(
            "batch axis {axis} is out of bounds for input {name:?} rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

fn move_axis(
    tensor: DynamicTensor,
    source: usize,
    destination: usize,
) -> Result<DynamicTensor, String> {
    if source == destination {
        return Ok(tensor);
    }
    let rank = tensor.shape().len();
    if source >= rank || destination >= rank {
        return Err(format!(
            "cannot move axis {source} to {destination} for rank {rank}"
        ));
    }
    let mut axes = (0..rank).collect::<Vec<_>>();
    let axis = axes.remove(source);
    axes.insert(destination, axis);
    tensor.permute(&axes)
}

fn vmap_inputs(
    signature: &VmapSignature,
    values: &Bound<'_, PyDict>,
) -> PyResult<BTreeMap<String, DynamicTensor>> {
    let mut inputs = extract_tensor_map(values)?;
    for (name, axis) in signature.input_names.iter().zip(&signature.in_axes) {
        let Some(axis) = axis else { continue };
        let tensor = inputs
            .remove(name)
            .ok_or_else(|| PyValueError::new_err(format!("missing input {name:?}")))?;
        inputs.insert(
            name.clone(),
            move_axis(tensor, *axis, 0).map_err(PyValueError::new_err)?,
        );
    }
    Ok(inputs)
}

fn vmap_output(signature: &VmapSignature, output: DynamicTensor) -> PyResult<PyTensor> {
    let output = move_axis(output, 0, signature.out_axis).map_err(PyValueError::new_err)?;
    PyTensor::from_dynamic_tensor(output).map_err(PyValueError::new_err)
}

fn vmap_gradient_outputs(
    signature: &VmapSignature,
    gradients: BTreeMap<String, DynamicTensor>,
) -> PyResult<BTreeMap<String, PyTensor>> {
    gradients
        .into_iter()
        .map(|(name, gradient)| {
            let axis = signature
                .input_names
                .iter()
                .position(|candidate| candidate == &name)
                .and_then(|index| signature.in_axes[index]);
            let gradient = match axis {
                Some(axis) => move_axis(gradient, 0, axis).map_err(PyValueError::new_err)?,
                None => gradient,
            };
            PyTensor::from_dynamic_tensor(gradient)
                .map(|gradient| (name, gradient))
                .map_err(PyValueError::new_err)
        })
        .collect()
}

/// Trace a fixed-size vectorized function into one reusable CPU plan.
///
/// `input_specs` describes one example. Mapped inputs are canonicalized to a
/// leading batch axis during tracing; `in_axes` and `out_axis` only select the
/// public layout at invocation boundaries.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorVmapFunction { plan, signature })
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

const CUDA_LOSS_COTANGENT_NAME: &str = "__nabla_loss_cotangent";

fn compile_cuda_scalar_value_and_grad(
    loss: &TensorTraceResult,
    parameter_names: Vec<String>,
    device_ordinal: usize,
) -> PyResult<(
    TensorCudaExecutionPlan,
    TensorNodeId,
    BTreeMap<String, TensorNodeId>,
)> {
    if parameter_names.is_empty() {
        return Err(PyValueError::new_err(
            "CUDA value-and-grad requires at least one parameter name",
        ));
    }
    if !loss.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "CUDA value-and-grad requires a scalar output, got shape {:?}",
            loss.output.shape
        )));
    }
    let (graph, loss_node_id, gradients) = loss
        .symbolic_vjp_graph(CUDA_LOSS_COTANGENT_NAME)
        .map_err(PyValueError::new_err)?;
    let mut output_node_ids = vec![loss_node_id];
    let mut requested_names = Vec::with_capacity(parameter_names.len());
    let mut seen = BTreeSet::new();
    for parameter_name in parameter_names {
        if !seen.insert(parameter_name.clone()) {
            return Err(PyValueError::new_err(format!(
                "duplicate parameter name {parameter_name:?}"
            )));
        }
        let gradient_node_id = gradients.get(&parameter_name).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "parameter {parameter_name:?} is not declared in the loss trace"
            ))
        })?;
        requested_names.push(parameter_name);
        output_node_ids.push(gradient_node_id);
    }
    let (plan, remapped_outputs) = graph
        .compile_cuda_multi_plan(&output_node_ids, device_ordinal)
        .map_err(PyValueError::new_err)?;
    let loss_node_id = remapped_outputs[0];
    let gradient_node_ids = requested_names
        .into_iter()
        .zip(remapped_outputs.into_iter().skip(1))
        .collect();
    Ok((plan, loss_node_id, gradient_node_ids))
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, parameter_names, device_ordinal = 0))]
pub fn tensor_value_and_grad_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    parameter_names: Vec<String>,
    device_ordinal: usize,
) -> PyResult<TensorCudaValueAndGradFunction> {
    if input_specs
        .iter()
        .any(|(name, _)| name == CUDA_LOSS_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_cuda_fn reserves input name {CUDA_LOSS_COTANGENT_NAME:?}"
        )));
    }
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let (plan, loss_node_id, gradient_node_ids) =
        compile_cuda_scalar_value_and_grad(&traced, parameter_names, device_ordinal)?;
    Ok(TensorCudaValueAndGradFunction {
        plan,
        loss_node_id,
        gradient_node_ids,
        cotangent_name: CUDA_LOSS_COTANGENT_NAME.to_string(),
    })
}

/// Trace a fixed-size vectorized function into one reusable CUDA plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0, device_ordinal = 0))]
pub fn tensor_vmap_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
    device_ordinal: usize,
) -> PyResult<TensorVmapCudaFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let plan = traced
        .graph
        .compile_cuda_plan(traced.output.node_id, device_ordinal)
        .map_err(PyValueError::new_err)?;
    Ok(TensorVmapCudaFunction { plan, signature })
}

/// Trace a fixed-size vectorized function into one reusable MLX plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_mlx_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapMlxFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let plan = traced
        .graph
        .compile_mlx_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?;
    Ok(TensorVmapMlxFunction { plan, signature })
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_vjp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapVjpFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorVmapVjpFunction { plan, signature })
}

const CUDA_VMAP_COTANGENT_NAME: &str = "__nabla_vmap_cotangent";

#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0, device_ordinal = 0))]
pub fn tensor_vmap_vjp_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
    device_ordinal: usize,
) -> PyResult<TensorVmapCudaVjpFunction> {
    if input_specs
        .iter()
        .any(|(name, _)| name == CUDA_VMAP_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_vmap_vjp_cuda_fn reserves input name {CUDA_VMAP_COTANGENT_NAME:?}"
        )));
    }
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let (graph, value_node_id, gradients) = traced
        .symbolic_vjp_graph(CUDA_VMAP_COTANGENT_NAME)
        .map_err(PyValueError::new_err)?;
    let mut output_node_ids = vec![value_node_id];
    let mut gradient_names = Vec::with_capacity(signature.input_names.len());
    for name in &signature.input_names {
        let node_id = gradients.get(name).copied().ok_or_else(|| {
            PyValueError::new_err(format!("vmap VJP input {name:?} is absent from the trace"))
        })?;
        gradient_names.push(name.clone());
        output_node_ids.push(node_id);
    }
    let (plan, output_node_ids) = graph
        .compile_cuda_multi_plan(&output_node_ids, device_ordinal)
        .map_err(PyValueError::new_err)?;
    let output_node_id = output_node_ids[0];
    let gradient_node_ids = gradient_names
        .into_iter()
        .zip(output_node_ids.into_iter().skip(1))
        .collect();
    Ok(TensorVmapCudaVjpFunction {
        plan,
        output_node_id,
        gradient_node_ids,
        signature,
        cotangent_name: CUDA_VMAP_COTANGENT_NAME.to_string(),
    })
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_jvp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, Vec<usize>)>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapJvpFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorVmapJvpFunction { plan, signature })
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

    #[getter]
    fn device_buffer_count(&self) -> PyResult<usize> {
        if let Some(shared_plan) = &self.shared_plan {
            return shared_plan
                .plan
                .plan
                .device_buffer_count()
                .map_err(PyValueError::new_err);
        }
        self.parameter_plans
            .iter()
            .try_fold(0usize, |count, (_, plan)| {
                plan.plan.device_buffer_count().and_then(|plan_count| {
                    count
                        .checked_add(plan_count)
                        .ok_or_else(|| "CUDA device buffer count overflow".to_string())
                })
            })
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (inputs = None))]
    fn loss(&mut self, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
        if !self.shared_plan_includes_loss {
            return Err(PyValueError::new_err(
                "loss is available only on optimizers created by cuda_adam_loss_optimizer",
            ));
        }
        let shared_plan = self.shared_plan.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "loss is available only on optimizers created by cuda_adam_loss_optimizer",
            )
        })?;
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
        let loss = shared_plan
            .plan
            .plan
            .execute_retaining(&self.inputs, &retained_inputs)
            .map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(loss).map_err(PyValueError::new_err)
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
        shared_plan_includes_loss: false,
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
        shared_plan_includes_loss: false,
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
#[pyo3(signature = (loss, parameter_names, inputs, learning_rate, retained_input_names = None, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, device_ordinal = 0))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_adam_loss_optimizer(
    loss: &Bound<'_, PyAny>,
    parameter_names: Vec<String>,
    inputs: &Bound<'_, PyDict>,
    learning_rate: f32,
    retained_input_names: Option<Vec<String>>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    device_ordinal: usize,
) -> PyResult<TensorCudaAdamOptimizer> {
    let loss = if let Ok(loss) = loss.extract::<PyRef<'_, TensorTraceResult>>() {
        loss.clone()
    } else if let Ok(loss) = loss.extract::<PyRef<'_, TraceTensor>>() {
        TensorTraceResult::new(loss.graph.clone(), loss.clone())
    } else {
        return Err(PyTypeError::new_err(
            "cuda_adam_loss_optimizer loss must be a TraceTensor or TensorTraceResult",
        ));
    };
    let (plan, _, gradient_node_ids) =
        compile_cuda_scalar_value_and_grad(&loss, parameter_names, device_ordinal)?;
    let parameter_names = gradient_node_ids.keys().cloned().collect::<BTreeSet<_>>();
    let mut values = extract_tensor_map(inputs)?;
    if values.contains_key(CUDA_LOSS_COTANGENT_NAME) {
        return Err(PyValueError::new_err(format!(
            "cuda_adam_loss_optimizer reserves input name {CUDA_LOSS_COTANGENT_NAME:?}"
        )));
    }
    values.insert(
        CUDA_LOSS_COTANGENT_NAME.to_string(),
        DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
    );
    let mut retained_inputs = retained_input_names
        .map(|names| names.into_iter().collect::<BTreeSet<_>>())
        .unwrap_or_else(|| values.keys().cloned().collect());
    retained_inputs.insert(CUDA_LOSS_COTANGENT_NAME.to_string());
    retained_inputs.extend(parameter_names.iter().cloned());
    Ok(TensorCudaAdamOptimizer {
        parameter_plans: Vec::new(),
        shared_plan: Some(SharedCudaAdamPlan {
            plan,
            gradient_node_ids,
        }),
        shared_plan_includes_loss: true,
        parameter_names,
        inputs: values,
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
