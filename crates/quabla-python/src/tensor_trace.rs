use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use quabla_core::tensor_ir::{
    BatchingError, CudaBackend, CudaDataParallelExecutionPlan, CudaDataParallelTiming,
    CudaExecutionPlan, DynamicTensor, MlxAdamPlan, MlxBackend, MlxRetainedInputs,
    SymbolicCotangent, TensorBackend, TensorComparison, TensorCondExecutionPlan, TensorDType,
    TensorExecutionPlan, TensorForiExecutionPlan, TensorIr, TensorNodeId, TensorRegion,
    TensorReplicaReduction, TensorScanExecutionPlan,
};
use quabla_core::{
    QuablaCompiler, QuablaExecutable, QuablaMultiOutputExecutable, QuablaMultiOutputProgram,
    QuablaTarget,
};

use crate::dtype::PyDType;
use crate::errors::{concrete_value_error, tracer_error, unsupported_operation_error};
use crate::tensor::bool_operation_error;
use crate::tensor::{
    extract_scalar, parse_axis_indices, parse_index_plan, parse_reshape_args, PyTensor,
    PyTensorView, TensorIndex,
};

/// One traced input declaration: `(name, shape)` for `float64`, or
/// `(name, shape, dtype)` with a `quabla.float32`/`quabla.float64` object.
#[derive(Clone, Debug)]
pub struct TensorInputSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
}

impl TensorInputSpec {
    fn new(name: String, shape: Vec<usize>, dtype: TensorDType) -> Self {
        Self { name, shape, dtype }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for TensorInputSpec {
    type Error = PyErr;

    fn extract(spec: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok((name, shape)) = spec.extract::<(String, Vec<usize>)>() {
            return Ok(Self::new(name, shape, TensorDType::F64));
        }
        let (name, shape, dtype) = spec
            .extract::<(String, Vec<usize>, PyDType)>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "tensor input spec must be (name, shape) or (name, shape, dtype) with a quabla dtype",
                )
            })?;
        Ok(Self::new(name, shape, dtype.dtype))
    }
}

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
    pub(crate) ir: Arc<Mutex<TensorIr>>,
}

#[pyclass(name = "TraceTensor", from_py_object)]
#[derive(Clone, Debug)]
pub struct TraceTensor {
    graph: TensorTraceGraph,
    pub(crate) node_id: TensorNodeId,
    shape: Vec<usize>,
    // vmap canonicalizes the mapped dimension to axis zero while tracing.
    // Normal traces have no mapped axis.
    batch_axis: Option<usize>,
}

#[pyclass(name = "TensorTraceResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorTraceResult {
    pub(crate) graph: TensorTraceGraph,
    pub(crate) output: TraceTensor,
}

#[pyclass(name = "TensorCpuExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorCpuExecutionPlan {
    pub(crate) plan: TensorExecutionPlan,
}

#[pyclass(name = "TensorCudaExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorCudaExecutionPlan {
    pub(crate) plan: CudaExecutionPlan,
}

#[pyclass(name = "TensorMlxExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TensorMlxExecutionPlan {
    pub(crate) plan: TensorExecutionPlan,
    pub(crate) retained_inputs: Arc<Mutex<MlxRetainedInputs>>,
}

#[pyclass(name = "TensorCudaAdamOptimizer", skip_from_py_object)]
pub struct TensorCudaAdamOptimizer {
    parameter_plans: Vec<(String, TensorCudaExecutionPlan)>,
    shared_plan: Option<SharedCudaAdamPlan>,
    shared_plan_includes_loss: bool,
    parameter_names: BTreeSet<String>,
    // Parameters the loss does not depend on. The compiled plan prunes them,
    // so they have no device buffer; their gradient is zero and Adam leaves
    // them unchanged, so their initial host values are returned as is.
    frozen_parameters: BTreeMap<String, DynamicTensor>,
    inputs: BTreeMap<String, DynamicTensor>,
    retained_inputs: BTreeSet<String>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

#[pyclass(name = "TensorMlxAdamOptimizer", unsendable, skip_from_py_object)]
pub struct TensorMlxAdamOptimizer {
    plan: MlxAdamPlan,
    parameter_names: BTreeSet<String>,
    inputs: BTreeMap<String, DynamicTensor>,
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

/// A pair of frozen CPU branch regions selected by a host boolean predicate.
///
/// This is deliberately a function-level control-flow boundary: it preserves
/// lazy branch execution while the Tensor IR gains nested region nodes.
#[pyclass(name = "TensorCondFunction", skip_from_py_object)]
pub struct TensorCondFunction {
    plan: TensorCondExecutionPlan,
}

#[pyclass(name = "TensorCondValueAndGradFunction", skip_from_py_object)]
pub struct TensorCondValueAndGradFunction {
    plan: TensorCondExecutionPlan,
}

#[pyclass(name = "TensorCondJvpFunction", skip_from_py_object)]
pub struct TensorCondJvpFunction {
    plan: TensorCondExecutionPlan,
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

#[pyclass(name = "TensorBatchValueAndGradFunction", unsendable)]
pub struct TensorBatchValueAndGradFunction {
    function: Py<PyAny>,
    input_names: Vec<String>,
    input_axes: Vec<Option<isize>>,
    max_specializations: usize,
    static_shapes: RefCell<Option<Vec<Vec<usize>>>>,
    plans: RefCell<BTreeMap<usize, TensorExecutionPlan>>,
}

struct MlxBatchValueAndGradPlan {
    plan: TensorMlxExecutionPlan,
    loss_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
}

struct CudaBatchValueAndGradPlan {
    plan: TensorCudaExecutionPlan,
    loss_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
}

type BatchSpecializationSignature = (usize, Vec<Vec<usize>>, Vec<TensorInputSpec>);

#[pyclass(name = "TensorBatchMlxValueAndGradFunction", unsendable)]
pub struct TensorBatchMlxValueAndGradFunction {
    function: Py<PyAny>,
    input_names: Vec<String>,
    parameter_names: Vec<String>,
    input_axes: Vec<Option<isize>>,
    max_specializations: usize,
    static_shapes: RefCell<Option<Vec<Vec<usize>>>>,
    plans: RefCell<BTreeMap<usize, MlxBatchValueAndGradPlan>>,
}

#[pyclass(name = "TensorBatchCudaJitFunction", unsendable)]
pub struct TensorBatchCudaJitFunction {
    function: Py<PyAny>,
    input_names: Vec<String>,
    input_axes: Vec<Option<isize>>,
    max_specializations: usize,
    device_ordinal: usize,
    static_shapes: RefCell<Option<Vec<Vec<usize>>>>,
    plans: RefCell<BTreeMap<usize, TensorCudaExecutionPlan>>,
}

#[pyclass(name = "TensorBatchCudaValueAndGradFunction", unsendable)]
pub struct TensorBatchCudaValueAndGradFunction {
    function: Py<PyAny>,
    input_names: Vec<String>,
    parameter_names: Vec<String>,
    input_axes: Vec<Option<isize>>,
    max_specializations: usize,
    device_ordinal: usize,
    static_shapes: RefCell<Option<Vec<Vec<usize>>>>,
    plans: RefCell<BTreeMap<usize, CudaBatchValueAndGradPlan>>,
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

#[pyclass(name = "TensorVmapMlxVjpFunction", skip_from_py_object)]
pub struct TensorVmapMlxVjpFunction {
    plan: TensorMlxExecutionPlan,
    output_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
    signature: VmapSignature,
    cotangent_name: String,
}

#[pyclass(name = "TensorVmapMlxJvpFunction", skip_from_py_object)]
pub struct TensorVmapMlxJvpFunction {
    plan: TensorMlxExecutionPlan,
    value_node_id: TensorNodeId,
    tangent_node_id: TensorNodeId,
    signature: VmapSignature,
    tangent_names: BTreeMap<String, String>,
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

#[pyclass(name = "TensorVmapCudaJvpFunction", skip_from_py_object)]
pub struct TensorVmapCudaJvpFunction {
    plan: TensorCudaExecutionPlan,
    value_node_id: TensorNodeId,
    tangent_node_id: TensorNodeId,
    signature: VmapSignature,
    tangent_names: BTreeMap<String, String>,
}

/// A per-example Hessian-vector product. The traced function must produce one
/// scalar for every mapped example; internally Quabla differentiates their sum.
#[pyclass(name = "TensorVmapHvpScalarFunction", skip_from_py_object)]
pub struct TensorVmapHvpScalarFunction {
    plan: TensorExecutionPlan,
    signature: VmapSignature,
    input_name: String,
    input_axis: usize,
    cotangent_name: String,
    tangent_name: String,
}

/// CUDA counterpart of [`TensorVmapHvpScalarFunction`].
#[pyclass(name = "TensorVmapCudaHvpScalarFunction", skip_from_py_object)]
pub struct TensorVmapCudaHvpScalarFunction {
    plan: TensorCudaExecutionPlan,
    signature: VmapSignature,
    input_name: String,
    input_axis: usize,
    cotangent_name: String,
    tangent_name: String,
}

#[pyclass(name = "TensorCudaValueAndGradFunction", skip_from_py_object)]
pub struct TensorCudaValueAndGradFunction {
    plan: TensorCudaExecutionPlan,
    loss_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
    cotangent_name: String,
}

#[pyclass(
    name = "TensorCudaDataParallelValueAndGradFunction",
    skip_from_py_object
)]
pub struct TensorCudaDataParallelValueAndGradFunction {
    plan: CudaDataParallelExecutionPlan,
    input_shapes: BTreeMap<String, Vec<usize>>,
    mapped_input_names: BTreeSet<String>,
    loss_node_id: TensorNodeId,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
    cotangent_name: String,
    reduction: TensorReplicaReduction,
    last_timing: Arc<Mutex<Option<CudaDataParallelTiming>>>,
}

#[pyclass(name = "TensorMlxValueAndGradFunction", skip_from_py_object)]
pub struct TensorMlxValueAndGradFunction {
    plan: TensorMlxExecutionPlan,
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

    pub fn add_input(
        &self,
        name: &str,
        shape: Vec<usize>,
        dtype: TensorDType,
    ) -> Result<TraceTensor, String> {
        let mut ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.input_typed(name, shape, dtype)?;
        let shape = ir.node_shape(node_id)?;
        Ok(TraceTensor {
            graph: self.clone(),
            node_id,
            shape,
            batch_axis: None,
        })
    }

    fn add_batched_input(
        &self,
        name: &str,
        shape: Vec<usize>,
        dtype: TensorDType,
    ) -> Result<TraceTensor, String> {
        let mut tensor = self.add_input(name, shape, dtype)?;
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

    fn stablehlo_text_ir(&self, output_node_id: TensorNodeId) -> Result<String, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        ir.stablehlo_text(output_node_id)
    }

    /// Compiles one traced output through the core compiler facade.
    ///
    /// Single-output compilation for function-specific helpers and
    /// `Program.compile` funnels through here, so `QuablaCompiler` owns program
    /// construction, freezing, and target selection for every target. The
    /// facade owns an immutable program, while tracing keeps a shared mutable
    /// graph, so each compilation snapshots the traced IR once. It skips the
    /// facade's build-availability check so these entrypoints keep their
    /// pre-facade errors in builds without the target's backend.
    fn compile_executable(
        &self,
        output_node_id: TensorNodeId,
        target: QuablaTarget,
    ) -> Result<QuablaExecutable, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .clone();
        let compiler = QuablaCompiler;
        compiler.compile_without_build_check(&compiler.program(ir, output_node_id)?, target)
    }

    pub(crate) fn compile_cpu_plan(
        &self,
        output_node_id: TensorNodeId,
    ) -> Result<TensorCpuExecutionPlan, String> {
        match self.compile_executable(output_node_id, QuablaTarget::Cpu)? {
            QuablaExecutable::Cpu(plan) => Ok(TensorCpuExecutionPlan { plan }),
            executable => Err(unexpected_executable(QuablaTarget::Cpu, &executable)),
        }
    }

    pub(crate) fn compile_cuda_plan(
        &self,
        output_node_id: TensorNodeId,
        device_ordinal: usize,
    ) -> Result<TensorCudaExecutionPlan, String> {
        let target = QuablaTarget::Cuda { device_ordinal };
        cuda_execution_plan(self.compile_executable(output_node_id, target)?, target)
    }

    pub(crate) fn compile_mlx_plan(
        &self,
        output_node_id: TensorNodeId,
    ) -> Result<TensorMlxExecutionPlan, String> {
        mlx_execution_plan(self.compile_executable(output_node_id, QuablaTarget::Mlx)?)
    }

    /// Compiles several traced outputs as one ordered facade program.
    ///
    /// Value-and-gradient and primal/tangent helpers funnel through here, so
    /// `QuablaCompiler` owns freezing, frozen output ids, and target selection
    /// for them as well. The handle is consumed: a graph built only for this
    /// compilation, such as a symbolic transform result, moves into the
    /// program, while a graph still shared with Python traces is snapshotted
    /// like single-output compilation. It skips the build-availability check
    /// for the same compatibility reason as [`Self::compile_executable`].
    fn compile_multi_output_executable(
        self,
        output_node_ids: Vec<TensorNodeId>,
        target: QuablaTarget,
    ) -> Result<QuablaMultiOutputExecutable, String> {
        QuablaCompiler.compile_many_without_build_check(
            &self.into_multi_output_program(output_node_ids)?,
            target,
        )
    }

    /// Builds the ordered facade program behind
    /// [`Self::compile_multi_output_executable`]; the CUDA data-parallel
    /// helper freezes it directly because it targets a replica set.
    pub(crate) fn into_multi_output_program(
        self,
        output_node_ids: Vec<TensorNodeId>,
    ) -> Result<QuablaMultiOutputProgram, String> {
        let ir = match Arc::try_unwrap(self.ir) {
            Ok(ir) => ir
                .into_inner()
                .map_err(|_| "tensor trace graph lock is poisoned".to_string())?,
            Err(ir) => ir
                .lock()
                .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
                .clone(),
        };
        QuablaMultiOutputProgram::new(ir, output_node_ids)
    }
}

/// A reverse-mode transform staged by [`TensorTraceGraph::symbolic_vjp_many`]:
/// the transformed graph, the retained tracers in request order, and one
/// gradient tracer per non-`bool` input, keyed by input name.
pub type StagedVjp = (
    TensorTraceGraph,
    Vec<TraceTensor>,
    BTreeMap<String, TraceTensor>,
);

/// A forward-mode transform staged by [`TensorTraceGraph::symbolic_jvp_many`]:
/// the transformed graph and the primal and tangent tracer of each output.
pub type StagedJvp = (TensorTraceGraph, Vec<TraceTensor>, Vec<TraceTensor>);

/// Multi-output staging for the v0.2 Python transforms (`quabla.grad`,
/// `quabla.jit`, ...). The Python layer traces a function into a graph,
/// flattens its pytree result into tracers, and composes these transforms on
/// the graph; every result is a fresh graph, so tracers of the source graph
/// stay valid for further transforms of the same trace.
impl TensorTraceGraph {
    fn ensure_owns(&self, tensors: &[&TraceTensor]) -> Result<(), String> {
        if tensors
            .iter()
            .all(|tensor| Arc::ptr_eq(&self.ir, &tensor.graph.ir))
        {
            Ok(())
        } else {
            Err(
                "a traced output belongs to a different graph; a tracer escaped from \
                 another trace"
                    .to_string(),
            )
        }
    }

    /// Wraps nodes of a freshly transformed graph as unbatched tracers.
    fn tracers(&self, node_ids: &[TensorNodeId]) -> Result<Vec<TraceTensor>, String> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        node_ids
            .iter()
            .map(|&node_id| {
                Ok(TraceTensor::from_node(
                    self.clone(),
                    node_id,
                    ir.node_shape(node_id)?,
                    None,
                ))
            })
            .collect()
    }

    /// Reverse mode over several seeded outputs, retaining `retained`
    /// (for example the value and auxiliary outputs) from the same graph.
    pub fn symbolic_vjp_many(
        &self,
        outputs: &[(TraceTensor, SymbolicCotangent)],
        retained: &[TraceTensor],
    ) -> Result<StagedVjp, String> {
        self.ensure_owns(
            &outputs
                .iter()
                .map(|(output, _)| output)
                .chain(retained)
                .collect::<Vec<_>>(),
        )?;
        let seeds = outputs
            .iter()
            .map(|(output, seed)| (output.node_id, seed.clone()))
            .collect::<Vec<_>>();
        let transformed = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .symbolic_vjp_many(&seeds)?;
        let retained = retained
            .iter()
            .map(|tensor| transformed.primals[tensor.node_id])
            .collect::<Vec<_>>();
        let (names, gradients): (Vec<_>, Vec<_>) = transformed.gradients.into_iter().unzip();
        let graph = TensorTraceGraph {
            ir: Arc::new(Mutex::new(transformed.graph)),
        };
        let retained = graph.tracers(&retained)?;
        let gradients = names.into_iter().zip(graph.tracers(&gradients)?).collect();
        Ok((graph, retained, gradients))
    }

    /// Forward mode over several outputs with runtime tangent inputs:
    /// `tangent_inputs` maps input names to new tangent input names, and
    /// omitted inputs get a zero tangent.
    pub fn symbolic_jvp_many(
        &self,
        outputs: &[TraceTensor],
        tangent_inputs: &BTreeMap<String, String>,
    ) -> Result<StagedJvp, String> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())?;
        let output_ids = outputs
            .iter()
            .map(|output| output.node_id)
            .collect::<Vec<_>>();
        let transformed = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .symbolic_jvp_many_with_tangent_inputs(&output_ids, tangent_inputs)?;
        let graph = TensorTraceGraph {
            ir: Arc::new(Mutex::new(transformed.graph)),
        };
        let values = graph.tracers(&transformed.values)?;
        let tangents = graph.tracers(&transformed.tangents)?;
        Ok((graph, values, tangents))
    }

    /// Freezes `outputs` into one CPU program whose inputs are bound by
    /// position in `input_names` order.
    pub fn compile_cpu_many(
        &self,
        outputs: &[TraceTensor],
        input_names: Vec<String>,
    ) -> Result<StagedExecutable, String> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())?;
        let program = self
            .clone()
            .into_multi_output_program(outputs.iter().map(|output| output.node_id).collect())?;
        Ok(StagedExecutable {
            executable: QuablaCompiler.compile_many(&program, QuablaTarget::Cpu)?,
            input_names,
        })
    }

    /// Splices `outputs` of this staged graph into the graph of the traced
    /// bindings (see [`TensorIr::inline`]) and returns them as tracers there;
    /// when the bindings come from several traces in progress, that is the
    /// innermost one, and tracers of the others are lifted into it as
    /// capture inputs. `bindings[i]` binds input `input_names[i]`; a scalar
    /// binding becomes a constant of that input's dtype. The tracers of the
    /// `tensor_vmap_*` helpers are rejected; `quabla.vmap` batches through
    /// [`Self::inline_batched_into`] instead.
    pub fn inline_into(
        &self,
        input_names: &[String],
        bindings: &[InlineBinding],
        outputs: &[TraceTensor],
    ) -> Result<Vec<TraceTensor>, String> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())?;
        if input_names.len() != bindings.len() {
            return Err(format!(
                "{} callee inputs with {} bindings",
                input_names.len(),
                bindings.len()
            ));
        }
        let traced = bindings
            .iter()
            .filter_map(|binding| match binding {
                InlineBinding::Traced(tensor) => Some(tensor),
                InlineBinding::Scalar(_) | InlineBinding::Constant(_) => None,
            })
            .collect::<Vec<_>>();
        // Bindings from several traces in progress (explicit arguments of an
        // inner trace and tracers captured from an outer one) splice into
        // the innermost of them, where the outer ones become capture inputs;
        // constant bindings alone splice into the innermost trace in progress.
        let target = match traced.first() {
            Some(first) => first.graph.clone(),
            None => TRACE_FRAMES
                .with(|frames| frames.borrow().last().map(|frame| frame.graph.clone()))
                .ok_or_else(|| "inlining needs at least one traced binding".to_string())?,
        };
        let traced = TraceTensor::lifted_to_common_graph(&traced)?
            .unwrap_or_else(|| traced.into_iter().cloned().collect());
        let target = traced.first().map_or(target, |tensor| tensor.graph.clone());
        if traced.iter().any(|tensor| tensor.batch_axis.is_some()) {
            return Err("a batched (vmap) tracer cannot be bound to an inlined graph".to_string());
        }
        if Arc::ptr_eq(&self.ir, &target.ir) {
            return Err("a staged graph cannot be inlined into itself".to_string());
        }
        let callee = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let mut ir = target
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let mut traced = traced.iter();
        let mut bound = BTreeMap::new();
        for (name, binding) in input_names.iter().zip(bindings) {
            let node_id = match binding {
                InlineBinding::Traced(_) => {
                    traced
                        .next()
                        .expect("one lifted tracer per traced binding")
                        .node_id
                }
                InlineBinding::Scalar(value) => {
                    let constant = ir.scalar_constant(*value);
                    let dtype = callee.node_dtype(callee.input_node_id(name)?)?;
                    if dtype == TensorDType::F64 {
                        constant
                    } else {
                        ir.cast(constant, dtype)?
                    }
                }
                InlineBinding::Constant(value) => ir.constant(value.clone(), false),
            };
            if bound.insert(name.clone(), node_id).is_some() {
                return Err(format!("callee input {name:?} is bound twice"));
            }
        }
        let output_ids = outputs
            .iter()
            .map(|output| output.node_id)
            .collect::<Vec<_>>();
        let spliced = ir.inline(&callee, &bound, &output_ids)?;
        spliced
            .into_iter()
            .map(|node_id| {
                Ok(TraceTensor::from_node(
                    target.clone(),
                    node_id,
                    ir.node_shape(node_id)?,
                    None,
                ))
            })
            .collect()
    }
}

impl TensorTraceGraph {
    /// Splices `outputs` of this staged per-example graph into the graph of
    /// the traced bindings, vectorized over `batch_size` examples (see
    /// [`TensorIr::inline_batched`]); this is how `quabla.vmap` stages its
    /// function. `bindings[i]` binds input `input_names[i]` and is mapped
    /// (leading batch axis) when `mapped[i]` is set. Returns each output as
    /// a tracer of the enclosing graph with its mapped flag.
    pub fn inline_batched_into(
        &self,
        input_names: &[String],
        bindings: &[TraceTensor],
        mapped: &[bool],
        batch_size: usize,
        outputs: &[TraceTensor],
    ) -> Result<Vec<(TraceTensor, bool)>, BatchingError> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())?;
        if input_names.len() != bindings.len() || input_names.len() != mapped.len() {
            return Err(BatchingError::Invalid(format!(
                "{} callee inputs with {} bindings and {} mapped flags",
                input_names.len(),
                bindings.len(),
                mapped.len()
            )));
        }
        let target = bindings
            .first()
            .map(|tensor| tensor.graph.clone())
            .ok_or_else(|| "vmap needs at least one traced binding".to_string())?;
        target
            .ensure_owns(&bindings.iter().collect::<Vec<_>>())
            .map_err(|_| "the bindings of a vmap splice belong to different traces".to_string())?;
        if bindings.iter().any(|tensor| tensor.batch_axis.is_some()) {
            return Err(BatchingError::Invalid(
                "a batched tracer of a tensor_vmap_* helper cannot be bound to a vmap splice"
                    .to_string(),
            ));
        }
        if Arc::ptr_eq(&self.ir, &target.ir) {
            return Err(BatchingError::Invalid(
                "a staged graph cannot be inlined into itself".to_string(),
            ));
        }
        let callee = self
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let mut ir = target
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let mut bound = BTreeMap::new();
        for ((name, binding), is_mapped) in input_names.iter().zip(bindings).zip(mapped) {
            if bound
                .insert(name.clone(), (binding.node_id, *is_mapped))
                .is_some()
            {
                return Err(BatchingError::Invalid(format!(
                    "callee input {name:?} is bound twice"
                )));
            }
        }
        let output_ids = outputs
            .iter()
            .map(|output| output.node_id)
            .collect::<Vec<_>>();
        let spliced = ir.inline_batched(&callee, &bound, batch_size, &output_ids)?;
        spliced
            .into_iter()
            .map(|(node_id, is_mapped)| {
                Ok((
                    TraceTensor::from_node(target.clone(), node_id, ir.node_shape(node_id)?, None),
                    is_mapped,
                ))
            })
            .collect()
    }
}

/// A value bound to a callee input by [`TensorTraceGraph::inline_into`]: a
/// tracer of the enclosing trace or a Python number.
#[derive(Clone, Debug)]
pub enum InlineBinding {
    Traced(TraceTensor),
    Scalar(f64),
    /// An eager array, bound as a strong constant of the enclosing trace like
    /// the graph input it replaces.
    Constant(DynamicTensor),
}

/// A multi-output program compiled for the v0.2 transforms. Inputs are
/// positional, in the order of [`Self::input_names`], so a call does not
/// build a name-keyed dict in Python; outputs follow program order.
///
/// The class is deliberately not registered in the extension module: it is
/// an implementation detail of `quabla._transforms`.
#[pyclass(name = "StagedExecutable", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct StagedExecutable {
    executable: QuablaMultiOutputExecutable,
    input_names: Vec<String>,
}

impl StagedExecutable {
    pub fn execute(&self, inputs: Vec<DynamicTensor>) -> Result<Vec<DynamicTensor>, String> {
        if inputs.len() != self.input_names.len() {
            return Err(format!(
                "staged program expects {} inputs, got {}",
                self.input_names.len(),
                inputs.len()
            ));
        }
        self.executable
            .execute(&self.input_names.iter().cloned().zip(inputs).collect())
    }
}

/// Unwraps a facade CUDA executable into the bridge plan whose device
/// buffers retained-input and Adam executors reuse across calls.
fn cuda_execution_plan(
    executable: QuablaExecutable,
    target: QuablaTarget,
) -> Result<TensorCudaExecutionPlan, String> {
    match executable {
        QuablaExecutable::Cuda(plan) => Ok(TensorCudaExecutionPlan { plan }),
        executable => Err(unexpected_executable(target, &executable)),
    }
}

/// Wraps a facade MLX executable in the bridge plan that owns retained
/// input arrays, which stay outside `QuablaExecutable`.
fn mlx_execution_plan(executable: QuablaExecutable) -> Result<TensorMlxExecutionPlan, String> {
    match executable {
        QuablaExecutable::Mlx(plan) => Ok(TensorMlxExecutionPlan {
            plan,
            retained_inputs: Arc::new(Mutex::new(MlxRetainedInputs::empty())),
        }),
        executable => Err(unexpected_executable(QuablaTarget::Mlx, &executable)),
    }
}

/// Reports a facade executable whose backend differs from the requested
/// target; this is an invariant violation, never a fallback.
fn unexpected_executable(requested: QuablaTarget, executable: &QuablaExecutable) -> String {
    format!(
        "compiler facade returned a {} executable for a {} target",
        executable.target().name(),
        requested.name()
    )
}

/// Name prefix of the inputs that stand for captured tracers (the
/// `__quabla_` prefix is reserved, design 3.11).
const CAPTURE_PREFIX: &str = "__quabla_capture/";

/// The message of a tracer that meets a graph it cannot be lifted into.
const UNLIFTABLE_TRACER: &str =
    "a TraceTensor of another trace was combined with this trace, but that trace is not an \
     enclosing quabla transform that is still tracing: the tracer escaped the function that \
     created it (through a global, a container, or an attribute) and is used after its trace \
     ended, or it meets a tensor_* helper or region trace, which cannot capture it; return the \
     value from the transformed function, or pass it as an argument";

/// A trace of a v0.2 transform in progress (`quabla._transforms._trace`).
///
/// A function staged by an inner transform may close over tracers of the
/// traces that enclose it (`grad(lambda x: f(x, w))` with a traced `w`). Such
/// a tracer is lifted into the inner graph as an input, recorded here with
/// its source, so the inner transform treats it as a constant and the
/// inlined graph binds the input back to the source; outer transforms then
/// differentiate through it (slice S4b).
struct TraceFrame {
    graph: TensorTraceGraph,
    captures: Vec<CapturedTracer>,
}

struct CapturedTracer {
    name: String,
    source: TraceTensor,
    input: TraceTensor,
}

thread_local! {
    /// The traces in progress on this thread, outermost first. Staging runs
    /// the user function synchronously on the calling thread, so a trace and
    /// the traces nested in it share one stack.
    static TRACE_FRAMES: RefCell<Vec<TraceFrame>> = const { RefCell::new(Vec::new()) };
}

impl TraceFrame {
    /// The input of this frame's graph standing for `tensor`, a tracer of an
    /// enclosing frame; one input per source node.
    fn capture(&mut self, tensor: &TraceTensor) -> Result<TraceTensor, String> {
        if let Some(captured) = self.captures.iter().find(|captured| {
            Arc::ptr_eq(&captured.source.graph.ir, &tensor.graph.ir)
                && captured.source.node_id == tensor.node_id
        }) {
            return Ok(captured.input.clone());
        }
        if tensor.batch_axis.is_some() {
            return Err(
                "a batched tracer of a tensor_vmap_* helper cannot be captured by a nested \
                 transform; batch with quabla.vmap instead"
                    .to_string(),
            );
        }
        let name = format!("{CAPTURE_PREFIX}{}", self.captures.len());
        let input = self
            .graph
            .add_input(&name, tensor.shape.clone(), tensor.dtype()?)?;
        self.captures.push(CapturedTracer {
            name,
            source: tensor.clone(),
            input: input.clone(),
        });
        Ok(input)
    }
}

fn frame_position(frames: &[TraceFrame], graph: &TensorTraceGraph) -> Option<usize> {
    frames
        .iter()
        .position(|frame| Arc::ptr_eq(&frame.graph.ir, &graph.ir))
}

impl TraceTensor {
    fn shares_graph(tensors: &[&Self]) -> bool {
        tensors
            .windows(2)
            .all(|pair| Arc::ptr_eq(&pair[0].graph.ir, &pair[1].graph.ir))
    }

    /// Checks that tracers of several graphs can meet: each of their graphs
    /// is a trace in progress, so the outer ones lift into the innermost.
    /// Operand extraction calls this to raise `TracerError` for the others.
    pub(crate) fn ensure_liftable(tensors: &[&Self]) -> Result<(), String> {
        if Self::shares_graph(tensors) {
            return Ok(());
        }
        TRACE_FRAMES.with(|frames| {
            let frames = frames.borrow();
            if tensors
                .iter()
                .all(|tensor| frame_position(&frames, &tensor.graph).is_some())
            {
                Ok(())
            } else {
                Err(UNLIFTABLE_TRACER.to_string())
            }
        })
    }

    /// `tensors` moved into one graph when they come from several traces in
    /// progress: the innermost of their graphs keeps its tracers, and every
    /// other tracer becomes a capture input of it (see [`TraceFrame`]).
    /// `None` when they already share a graph.
    fn lifted_to_common_graph(tensors: &[&Self]) -> Result<Option<Vec<Self>>, String> {
        if Self::shares_graph(tensors) {
            return Ok(None);
        }
        TRACE_FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            let positions = tensors
                .iter()
                .map(|tensor| {
                    frame_position(&frames, &tensor.graph)
                        .ok_or_else(|| UNLIFTABLE_TRACER.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let home = positions.iter().copied().max().unwrap_or_default();
            tensors
                .iter()
                .zip(positions)
                .map(|(tensor, position)| {
                    if position == home {
                        Ok((*tensor).clone())
                    } else {
                        frames[home].capture(tensor)
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        })
    }
}

impl TensorTraceGraph {
    /// Marks this graph as a trace in progress (see [`TraceFrame`]).
    pub fn begin_trace(&self) -> Result<(), String> {
        TRACE_FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            if frame_position(&frames, self).is_some() {
                return Err("this graph is already being traced".to_string());
            }
            frames.push(TraceFrame {
                graph: self.clone(),
                captures: Vec::new(),
            });
            Ok(())
        })
    }

    /// Ends the trace of this graph, which must be the innermost one, and
    /// returns its capture inputs as `(input name, captured tracer)` in
    /// capture order.
    pub fn end_trace(&self) -> Result<Vec<(String, TraceTensor)>, String> {
        TRACE_FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            match frames.last() {
                Some(frame) if Arc::ptr_eq(&frame.graph.ir, &self.ir) => {}
                _ => return Err("this graph is not the innermost trace in progress".to_string()),
            }
            let frame = frames.pop().expect("the innermost frame was just checked");
            Ok(frame
                .captures
                .into_iter()
                .map(|captured| (captured.name, captured.source))
                .collect())
        })
    }

    /// `tensor` as a tracer of this graph, a trace in progress: itself, or a
    /// capture input when it belongs to an enclosing trace.
    pub fn lift(&self, tensor: &TraceTensor) -> Result<TraceTensor, String> {
        if Arc::ptr_eq(&self.ir, &tensor.graph.ir) {
            return Ok(tensor.clone());
        }
        TRACE_FRAMES.with(|frames| {
            let mut frames = frames.borrow_mut();
            match (
                frame_position(&frames, self),
                frame_position(&frames, &tensor.graph),
            ) {
                (Some(home), Some(source)) if source < home => frames[home].capture(tensor),
                _ => Err(UNLIFTABLE_TRACER.to_string()),
            }
        })
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

    pub(crate) fn dtype(&self) -> Result<TensorDType, String> {
        self.graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .node_dtype(self.node_id)
    }

    fn astype_tensor(&self, dtype: TensorDType) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.cast(self.node_id, dtype)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            self.shape.clone(),
            self.batch_axis,
        ))
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
        if let Some(lifted) = Self::lifted_to_common_graph(&tensors.iter().collect::<Vec<_>>())? {
            return Self::try_concat(&lifted, axis);
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
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].binary(&lifted[1], op);
        }
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
            "pow" => ir.pow(self.node_id, rhs.node_id)?,
            "atan2" => ir.atan2(self.node_id, rhs.node_id)?,
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
            "pow" => ir.pow(self.node_id, scalar)?,
            "atan2" => ir.atan2(self.node_id, scalar)?,
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

    /// Embeds `value` as an array constant of this tracer's graph. Like a
    /// `jax.jit` closure constant, its data is fixed when the function is
    /// traced; it carries no vmap batch axis, so it broadcasts against the
    /// examples of a batched operand like an unmapped value.
    fn constant_tensor(&self, value: DynamicTensor, weak: bool) -> Result<Self, String> {
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.constant(value, weak);
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(self.graph.clone(), node_id, shape, None))
    }

    /// Captures an eager tensor as a constant of this tracer's graph, keeping
    /// its dtype and (rare) weak type, so eager and traced code promote alike.
    pub(crate) fn capture(&self, eager: &PyTensor) -> Result<Self, String> {
        self.constant_tensor(eager.to_dynamic_tensor()?, eager.is_weak())
    }

    /// `operand` as a value of this tracer's graph: a tracer as is, and an
    /// eager `Tensor` or `TensorView` captured as a constant. Anything else is
    /// `None`, for the caller to try a Python number or raise its `TypeError`.
    pub(crate) fn traced_operand(&self, operand: &Bound<'_, PyAny>) -> Option<PyResult<Self>> {
        if let Ok(tracer) = operand.extract::<PyRef<'_, TraceTensor>>() {
            // A tracer of an enclosing trace is lifted when the op is built;
            // any other tracer of another graph escaped its trace.
            return Some(
                Self::ensure_liftable(&[self, &tracer])
                    .map(|()| tracer.clone())
                    .map_err(|message| tracer_error(operand.py(), message)),
            );
        }
        let captured = if let Ok(tensor) = operand.extract::<PyRef<'_, PyTensor>>() {
            self.capture(&tensor)
        } else if let Ok(view) = operand.extract::<PyRef<'_, PyTensorView>>() {
            view.materialize().and_then(|tensor| self.capture(&tensor))
        } else {
            return None;
        };
        Some(captured.map_err(PyValueError::new_err))
    }

    /// [`Self::traced_operand`] for an operand that must be an array.
    fn array_operand(&self, operand: &Bound<'_, PyAny>, op: &str) -> PyResult<Self> {
        self.traced_operand(operand).unwrap_or_else(|| {
            Err(PyTypeError::new_err(format!(
                "{op} expects a TraceTensor or Tensor operand"
            )))
        })
    }

    pub(crate) fn scalar_tensor(&self, value: f64) -> Result<Self, String> {
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
            "pow" => ir.pow(scalar, self.node_id)?,
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
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].matmul_tensor(&lifted[1]);
        }
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
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].solve_tensor(&lifted[1]);
        }
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

    fn cholesky_tensor(&self) -> Result<Self, String> {
        if self.batch_axis.is_some() || self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(format!(
                "cholesky reference tracing requires a square unbatched rank-2 tensor, got {:?}",
                self.shape
            ));
        }
        self.apply(&[], |ir| ir.cholesky(self.node_id))
    }

    pub fn where_tensor(&self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, on_true, on_false])? {
            return lifted[0].where_tensor(&lifted[1], &lifted[2]);
        }
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

    /// Appends one node built from `self` and `others` (same graph), keeping
    /// the merged vmap batch axis.
    fn apply(
        &self,
        others: &[&Self],
        build: impl FnOnce(&mut TensorIr) -> Result<TensorNodeId, String>,
    ) -> Result<Self, String> {
        for other in others {
            self.same_graph(other)?;
        }
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = build(&mut ir)?;
        let shape = ir.node_shape(node_id)?;
        let tensors = std::iter::once(self)
            .chain(others.iter().copied())
            .collect::<Vec<_>>();
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            Self::merged_batch_axis(&tensors)?,
        ))
    }

    fn ensure_not_bool(&self, op: &str) -> Result<(), String> {
        if self.dtype()? == TensorDType::Bool {
            return Err(bool_operation_error(op));
        }
        Ok(())
    }

    fn compare_tensor(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String> {
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].compare_tensor(&lifted[1], kind);
        }
        self.apply(&[rhs], |ir| ir.compare(self.node_id, rhs.node_id, kind))
    }

    fn compare_scalar(&self, value: f64, kind: TensorComparison) -> Result<Self, String> {
        self.apply(&[], |ir| {
            let scalar = ir.scalar_constant(value);
            ir.compare(self.node_id, scalar, kind)
        })
    }

    fn logical_tensor(&self, rhs: &Self, and: bool) -> Result<Self, String> {
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].logical_tensor(&lifted[1], and);
        }
        self.apply(&[rhs], |ir| {
            if and {
                ir.logical_and(self.node_id, rhs.node_id)
            } else {
                ir.logical_or(self.node_id, rhs.node_id)
            }
        })
    }

    fn logical_not_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.logical_not(self.node_id))
    }

    fn classify_tensor(&self, nan: bool) -> Result<Self, String> {
        self.apply(&[], |ir| {
            if nan {
                ir.isnan(self.node_id)
            } else {
                ir.isfinite(self.node_id)
            }
        })
    }

    /// `any`/`all` with the multi-axis, keepdims and vmap handling of `sum`:
    /// `sum(cast(b)) > 0` and `sum(cast(!b)) == 0`, as `TensorIr::any`/`all`.
    fn any_all_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        any: bool,
    ) -> Result<Self, String> {
        let dtype = self.dtype()?;
        if dtype != TensorDType::Bool {
            return Err(format!(
                "{} requires bool operands, got dtype {dtype}; \
                 build a mask with a comparison or convert explicitly with astype",
                if any { "any" } else { "all" }
            ));
        }
        let counted = if any {
            self.clone()
        } else {
            self.logical_not_tensor()?
        };
        let counts = counted
            .astype_tensor(TensorDType::F64)?
            .reduce_axes_tensor(axes, keepdims, false)?;
        let kind = if any {
            TensorComparison::Greater
        } else {
            TensorComparison::Equal
        };
        counts.compare_scalar(0.0, kind)
    }

    /// `isnan(self) | ordered`, the `maximum`/`minimum` mask for a float
    /// `self`: a NaN `self` selects itself, and a NaN right operand fails the
    /// ordered comparison and is selected, so NaN propagates from either side
    /// like NumPy and JAX. A `bool` left operand cannot be NaN; it keeps the
    /// legacy `greater` mask so its promotion and errors stay unchanged.
    fn nan_or(&self, ordered: Self) -> Result<Self, String> {
        self.classify_tensor(true)?.logical_tensor(&ordered, false)
    }

    /// `where(isnan(x) | (x > y), x, y)`: ties select `y`, and NaN in either
    /// operand propagates.
    fn maximum_tensor(&self, rhs: &Self) -> Result<Self, String> {
        if !self.dtype()?.is_floating() {
            let mask = self.binary(rhs, "greater")?;
            return mask.where_tensor(self, rhs);
        }
        let mask = self.nan_or(self.compare_tensor(rhs, TensorComparison::Greater)?)?;
        mask.where_tensor(self, rhs)
    }

    fn maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let mask = if self.dtype()?.is_floating() {
            self.nan_or(self.compare_scalar(rhs, TensorComparison::Greater)?)?
        } else {
            self.scalar_binary(rhs, "greater")?
        };
        let rhs = self.scalar_tensor(rhs)?;
        mask.where_tensor(self, &rhs)
    }

    /// `where(isnan(x) | (y > x), x, y)`: ties select `y`, and NaN in either
    /// operand propagates.
    fn minimum_tensor(&self, rhs: &Self) -> Result<Self, String> {
        if !self.dtype()?.is_floating() {
            let mask = rhs.binary(self, "greater")?;
            return mask.where_tensor(self, rhs);
        }
        let mask = self.nan_or(rhs.compare_tensor(self, TensorComparison::Greater)?)?;
        mask.where_tensor(self, rhs)
    }

    fn minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        let mask = if self.dtype()?.is_floating() {
            self.nan_or(self.compare_scalar(rhs, TensorComparison::Less)?)?
        } else {
            self.scalar_left_binary(rhs, "greater")?
        };
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
        self.ensure_not_bool("relu")?;
        self.maximum_scalar(0.0)
    }

    fn abs_tensor(&self) -> Result<Self, String> {
        self.ensure_not_bool("abs")?;
        let mask = self.scalar_binary(0.0, "greater")?;
        let negative = self.scalar_left_binary(0.0, "sub")?;
        mask.where_tensor(self, &negative)
    }

    /// `exp(-|x|)`, which never overflows. `abs` takes its `0 - x` branch at
    /// zero, so the derivative there is that of `exp(x)`.
    fn exp_negative_abs(&self) -> Result<Self, String> {
        self.abs_tensor()?.scalar_binary(-1.0, "mul")?.exp_tensor()
    }

    /// `where(x > 0, 1 / (1 + z), z / (1 + z))` with `z = exp(-|x|)`. Both
    /// branches stay finite for every finite `x`, so the unselected branch
    /// cannot form `0 * inf` in the gradient (the textbook
    /// `1 / (1 + exp(-x))` overflows `exp` for large negative `x`). Zero
    /// takes the `z / (1 + z)` branch, which matches the branch `abs` takes.
    /// `PyTensor::try_sigmoid` evaluates the same per-op rounded expression.
    fn sigmoid_tensor(&self) -> Result<Self, String> {
        self.ensure_not_bool("sigmoid")?;
        let decay = self.exp_negative_abs()?;
        let denominator = decay.scalar_binary(1.0, "add")?;
        let positive = denominator.scalar_left_binary(1.0, "div")?;
        let negative = decay.binary(&denominator, "div")?;
        self.compare_scalar(0.0, TensorComparison::Greater)?
            .where_tensor(&positive, &negative)
    }

    /// `maximum(x, 0) + log1p(exp(-|x|))`; `log1p` keeps the correction
    /// accurate where `exp(-|x|)` is below the dtype epsilon.
    /// `PyTensor::try_softplus` evaluates the same per-op rounded expression.
    fn softplus_tensor(&self) -> Result<Self, String> {
        self.ensure_not_bool("softplus")?;
        let linear = self.maximum_scalar(0.0)?;
        let correction = self.exp_negative_abs()?.log1p_tensor()?;
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
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.gather(self.node_id, indices.into(), actual_axis)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
    }

    fn scatter_add_tensor(
        &self,
        indices: &[usize],
        updates: &Self,
        axis: isize,
    ) -> Result<Self, String> {
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, updates])? {
            return lifted[0].scatter_add_tensor(indices, &lifted[1], axis);
        }
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
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let node_id = ir.scatter_add(self.node_id, updates.node_id, indices.into(), actual_axis)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            self.batch_axis,
        ))
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

    // `max|x| * sqrt(sum((x / max|x|)^2))`: scaling keeps the sum of squares
    // from overflowing or underflowing when |x| is outside the square root of
    // the dtype's range. A zero, infinite or NaN scale falls back to 1, which
    // keeps those inputs' unscaled results. The scale's derivative terms
    // cancel, so gradients stay `x / norm` up to rounding.
    fn norm_tensor(&self, axes: Option<Vec<isize>>, keepdims: bool) -> Result<Self, String> {
        self.ensure_not_bool("norm")?;
        let largest = self
            .abs_tensor()?
            .extrema_axes_tensor(axes.clone(), true, true)?;
        let usable = largest
            .compare_scalar(0.0, TensorComparison::Greater)?
            .logical_tensor(&largest.classify_tensor(false)?, true)?;
        let scale = usable.where_tensor(&largest, &largest.scalar_tensor(1.0)?)?;
        let reduced = self
            .binary(&scale, "div")?
            .powi_tensor(2)?
            .reduce_axes_tensor(axes, keepdims, false)?
            .sqrt_tensor()?;
        let batch_offset = usize::from(reduced.batch_axis.is_some());
        let scale = scale.reshape_tensor(reduced.shape[batch_offset..].to_vec())?;
        reduced.binary(&scale, "mul")
    }

    fn extrema_axes_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        maximum: bool,
    ) -> Result<Self, String> {
        self.ensure_not_bool(if maximum { "max" } else { "min" })?;
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

    fn log1p_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.log1p(self.node_id))
    }

    fn expm1_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.expm1(self.node_id))
    }

    fn erf_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.erf(self.node_id))
    }

    fn stop_gradient_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.stop_gradient(self.node_id))
    }

    /// Inclusive prefix sums along the example `axis`, or over the flattened
    /// example when `axis` is `None` (NumPy's convention).
    fn cumsum_tensor(&self, axis: Option<isize>, reverse: bool) -> Result<Self, String> {
        self.ensure_not_bool("cumsum")?;
        let source = match axis {
            Some(_) => self.clone(),
            None => {
                let example = &self.shape[usize::from(self.batch_axis.is_some())..];
                self.reshape_tensor(vec![example.iter().product()])?
            }
        };
        let axis = source.example_axis(axis.unwrap_or(0))?;
        source.apply(&[], |ir| ir.cumsum(source.node_id, axis, reverse))
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

    pub(crate) fn symbolic_jvp_result(&self, input_name: &str) -> Result<Self, String> {
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

    pub(crate) fn symbolic_vjp_results(
        &self,
        cotangent_name: &str,
    ) -> Result<BTreeMap<String, Self>, String> {
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

    /// Adds an input when `shape` is given (`dtype` defaults to float64), or
    /// returns the existing input `name` otherwise.
    #[pyo3(signature = (name, shape = None, dtype = None))]
    fn input(
        &self,
        name: &str,
        shape: Option<Vec<usize>>,
        dtype: Option<PyDType>,
    ) -> PyResult<TraceTensor> {
        match (shape, dtype) {
            (Some(shape), dtype) => self.add_input(
                name,
                shape,
                dtype.map_or(TensorDType::F64, |dtype| dtype.dtype),
            ),
            (None, None) => self.existing_input(name),
            (None, Some(_)) => Err("an existing input cannot be retyped; use astype".to_string()),
        }
        .map_err(PyValueError::new_err)
    }

    fn lower_text(&self) -> PyResult<String> {
        self.lower_text_ir().map_err(PyValueError::new_err)
    }

    fn stablehlo_text(&self, output_node_id: TensorNodeId) -> PyResult<String> {
        self.stablehlo_text_ir(output_node_id)
            .map_err(PyValueError::new_err)
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

    /// `symbolic_vjp_many` for `quabla._transforms`: a `None` cotangent name
    /// seeds that output with ones.
    #[pyo3(name = "_symbolic_vjp")]
    fn py_symbolic_vjp(
        &self,
        outputs: Vec<TraceTensor>,
        cotangent_names: Vec<Option<String>>,
        retained: Vec<TraceTensor>,
    ) -> PyResult<StagedVjp> {
        if outputs.len() != cotangent_names.len() {
            return Err(PyValueError::new_err(format!(
                "{} differentiated outputs with {} cotangent names",
                outputs.len(),
                cotangent_names.len()
            )));
        }
        let outputs = outputs
            .into_iter()
            .zip(cotangent_names)
            .map(|(output, name)| {
                (
                    output,
                    name.map_or(SymbolicCotangent::Ones, SymbolicCotangent::Input),
                )
            })
            .collect::<Vec<_>>();
        self.symbolic_vjp_many(&outputs, &retained)
            .map_err(PyValueError::new_err)
    }

    /// `symbolic_jvp_many` for `quabla._transforms`.
    #[pyo3(name = "_symbolic_jvp")]
    fn py_symbolic_jvp(
        &self,
        outputs: Vec<TraceTensor>,
        tangent_inputs: BTreeMap<String, String>,
    ) -> PyResult<StagedJvp> {
        self.symbolic_jvp_many(&outputs, &tangent_inputs)
            .map_err(PyValueError::new_err)
    }

    /// `compile_cpu_many` for `quabla._transforms`.
    #[pyo3(name = "_compile_cpu")]
    fn py_compile_cpu(
        &self,
        outputs: Vec<TraceTensor>,
        input_names: Vec<String>,
    ) -> PyResult<StagedExecutable> {
        self.compile_cpu_many(&outputs, input_names)
            .map_err(PyValueError::new_err)
    }

    /// Compiles the ordered v0.2 output program for an explicitly selected backend.
    #[pyo3(name = "_compile", signature = (outputs, input_names, target = "cpu", device_ordinal = 0))]
    fn py_compile_target(
        &self,
        py: Python<'_>,
        outputs: Vec<TraceTensor>,
        input_names: Vec<String>,
        target: &str,
        device_ordinal: usize,
    ) -> PyResult<StagedExecutable> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())
            .map_err(PyValueError::new_err)?;
        let target = match target {
            "cpu" => QuablaTarget::Cpu,
            "cuda" => QuablaTarget::Cuda { device_ordinal },
            "mlx" => QuablaTarget::Mlx,
            _ => return Err(PyValueError::new_err("unknown compilation target")),
        };
        let program = self
            .clone()
            .into_multi_output_program(outputs.iter().map(|output| output.node_id).collect())
            .map_err(PyValueError::new_err)?;
        Ok(StagedExecutable {
            executable: QuablaCompiler
                .compile_many_checked(&program, target)
                .map_err(|error| {
                    use quabla_core::compiler::QuablaCompileError;
                    match error {
                        QuablaCompileError::Unavailable(message) => {
                            crate::errors::device_operation_error(py, message, "jit", target.name())
                        }
                        QuablaCompileError::Unsupported { op, message } => {
                            crate::errors::device_operation_error(py, message, &op, target.name())
                        }
                        QuablaCompileError::InvalidProgram(message)
                        | QuablaCompileError::Backend(message) => PyValueError::new_err(message),
                    }
                })?,
            input_names,
        })
    }

    #[pyo3(name = "_as_program")]
    fn py_as_program(
        &self,
        outputs: Vec<TraceTensor>,
    ) -> PyResult<crate::compiler::PyQuablaProgram> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())
            .map_err(PyValueError::new_err)?;
        crate::compiler::PyQuablaProgram::from_outputs(self.clone(), outputs)
    }

    /// `inline_into` for `quabla._transforms`: each binding is a tracer of
    /// the enclosing trace, a Python number, or an eager array.
    #[pyo3(name = "_inline")]
    fn py_inline(
        &self,
        py: Python<'_>,
        input_names: Vec<String>,
        bindings: Vec<Bound<'_, PyAny>>,
        outputs: Vec<TraceTensor>,
    ) -> PyResult<Vec<TraceTensor>> {
        let bindings = bindings
            .iter()
            .map(|binding| {
                if let Ok(tensor) = binding.extract::<TraceTensor>() {
                    Ok(InlineBinding::Traced(tensor))
                } else if let Ok(tensor) = binding.extract::<PyRef<'_, PyTensor>>() {
                    // Checked before numbers: a single-element Tensor also
                    // converts to f64, which would drop its dtype.
                    Ok(InlineBinding::Constant(
                        tensor.to_dynamic_tensor().map_err(PyValueError::new_err)?,
                    ))
                } else if binding.is_instance_of::<PyFloat>() || binding.is_instance_of::<PyInt>() {
                    Ok(InlineBinding::Scalar(binding.extract::<f64>()?))
                } else {
                    Err(PyTypeError::new_err(format!(
                        "an inline binding must be a TraceTensor, a Tensor, or a Python number, \
                         got {}",
                        binding.get_type().name()?
                    )))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let traced = bindings
            .iter()
            .filter_map(|binding| match binding {
                InlineBinding::Traced(tensor) => Some(tensor),
                InlineBinding::Scalar(_) | InlineBinding::Constant(_) => None,
            })
            .collect::<Vec<_>>();
        TraceTensor::ensure_liftable(&traced).map_err(|message| tracer_error(py, message))?;
        self.inline_into(&input_names, &bindings, &outputs)
            .map_err(PyValueError::new_err)
    }

    /// Marks this graph as a trace in progress of a v0.2 transform, so
    /// tracers of the traces enclosing it are lifted into it as capture
    /// inputs (slice S4b); `_end_trace` must follow.
    #[pyo3(name = "_begin_trace")]
    fn py_begin_trace(&self) -> PyResult<()> {
        self.begin_trace().map_err(PyValueError::new_err)
    }

    /// Ends the innermost trace in progress, this graph, and returns its
    /// capture inputs as `(input name, captured tracer)` in capture order.
    #[pyo3(name = "_end_trace")]
    fn py_end_trace(&self) -> PyResult<Vec<(String, TraceTensor)>> {
        self.end_trace().map_err(PyValueError::new_err)
    }

    /// A result of the traced function as a tracer of this graph: a tracer
    /// of an enclosing trace becomes a capture input, and a tracer of any
    /// other graph raises `TracerError`.
    #[pyo3(name = "_lift")]
    fn py_lift(&self, py: Python<'_>, tensor: TraceTensor) -> PyResult<TraceTensor> {
        self.lift(&tensor)
            .map_err(|message| tracer_error(py, message))
    }

    /// Whether a trace of a v0.2 transform is in progress on this thread.
    #[staticmethod]
    #[pyo3(name = "_tracing")]
    fn py_tracing() -> bool {
        TRACE_FRAMES.with(|frames| !frames.borrow().is_empty())
    }

    /// `inline_batched_into` for `quabla._transforms` (`quabla.vmap`): a
    /// mapped op without a batching rule raises `UnsupportedOperationError`.
    #[pyo3(name = "_inline_batched")]
    fn py_inline_batched(
        &self,
        py: Python<'_>,
        input_names: Vec<String>,
        bindings: Vec<TraceTensor>,
        mapped: Vec<bool>,
        batch_size: usize,
        outputs: Vec<TraceTensor>,
    ) -> PyResult<Vec<(TraceTensor, bool)>> {
        self.inline_batched_into(&input_names, &bindings, &mapped, batch_size, &outputs)
            .map_err(|error| match &error {
                BatchingError::Unsupported { op } => {
                    unsupported_operation_error(py, error.to_string(), op)
                }
                BatchingError::Invalid(message) => PyValueError::new_err(message.clone()),
            })
    }

    /// An eager tensor as a constant of this graph, keeping its dtype and
    /// weak type (the constants of `quabla.vmap` and `quabla.jacobian`).
    #[pyo3(name = "_constant")]
    fn py_constant(&self, value: PyRef<'_, PyTensor>) -> PyResult<TraceTensor> {
        let dynamic = value.to_dynamic_tensor().map_err(PyValueError::new_err)?;
        let mut ir = self
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
        let node_id = ir.constant(dynamic, value.is_weak());
        let shape = ir.node_shape(node_id).map_err(PyValueError::new_err)?;
        Ok(TraceTensor::from_node(self.clone(), node_id, shape, None))
    }

    /// Number of nodes in the graph, including nodes no output depends on.
    #[getter(_node_count)]
    fn py_node_count(&self) -> PyResult<usize> {
        Ok(self
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .node_count())
    }

    #[getter(_has_f32_nodes)]
    fn py_has_f32_nodes(&self) -> PyResult<bool> {
        let ir = self
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
        for node_id in 0..ir.node_count() {
            if ir.node_dtype(node_id).map_err(PyValueError::new_err)? == TensorDType::F32 {
                return Ok(true);
            }
        }
        Ok(false)
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
impl StagedExecutable {
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Number of nodes in the frozen plan (after DCE and structural CSE).
    #[getter]
    fn node_count(&self) -> usize {
        self.executable.plan().node_count()
    }

    fn __call__(&self, inputs: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<PyTensor>> {
        let inputs = inputs
            .iter()
            .map(|tensor| tensor.to_dynamic_tensor())
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        self.execute(inputs)
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(PyTensor::from_dynamic_tensor)
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "StagedExecutable(inputs={}, outputs={})",
            self.input_names.len(),
            self.executable.output_node_ids().len()
        )
    }
}

#[pymethods]
impl TraceTensor {
    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        Err(tracer_error(
            py,
            "TraceTensor cannot drive Python control flow; use quabla.where for elementwise selection or an explicit control-flow primitive",
        ))
    }

    // A tracer has no value while its function is traced. Without these
    // methods `np.asarray(x)` would silently wrap the tracer in an object
    // array, so every conversion to concrete data raises `TracerError`.
    fn __float__(&self, py: Python<'_>) -> PyResult<f64> {
        Err(concrete_value_error(py, "float()"))
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<i64> {
        Err(concrete_value_error(py, "int()"))
    }

    fn __index__(&self, py: Python<'_>) -> PyResult<isize> {
        Err(concrete_value_error(py, "an integer index"))
    }

    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (dtype, copy);
        Err(concrete_value_error(py, "conversion to a NumPy array"))
    }

    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(concrete_value_error(py, "numpy()"))
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(concrete_value_error(py, "tolist()"))
    }

    fn item(&self, py: Python<'_>) -> PyResult<f64> {
        Err(concrete_value_error(py, "item()"))
    }

    #[getter]
    fn node_id(&self) -> TensorNodeId {
        self.node_id
    }

    /// Whether the tracer carries a `vmap` batch axis.
    #[getter(_batched)]
    fn py_batched(&self) -> bool {
        self.batch_axis.is_some()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter(dtype)]
    fn py_dtype(&self) -> PyResult<PyDType> {
        self.dtype()
            .map(PyDType::from)
            .map_err(PyValueError::new_err)
    }

    /// Traces an explicit dtype conversion (see `TensorIr::cast`).
    fn astype(&self, dtype: PyDType) -> PyResult<Self> {
        self.astype_tensor(dtype.dtype)
            .map_err(PyValueError::new_err)
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

    /// `-x` as `x * -1`, like the eager `Tensor` (so `-0.0` stays signed).
    fn __neg__(&self) -> PyResult<Self> {
        if self.dtype().map_err(PyValueError::new_err)? == TensorDType::Bool {
            return Err(PyValueError::new_err(bool_operation_error("negative")));
        }
        self.scalar_binary(-1.0, "mul")
            .map_err(PyValueError::new_err)
    }

    /// `x ** n` for a non-negative Python int `n` lowers to `powi`, which
    /// is exact and cheaper than a general power (like JAX's
    /// `integer_pow`). Every other exponent, including an integer-valued
    /// float, a negative int, and a `TraceTensor`, lowers to the
    /// differentiable elementwise `pow` op; a Python number is a weak scalar
    /// that adopts the dtype of `x`.
    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for TraceTensor power",
            ));
        }
        if let Ok(exponent) = exponent.extract::<u32>() {
            return self.powi(exponent);
        }
        trace_tensor_or_scalar_operand(self, exponent, "pow")
    }

    /// `c ** x` for a Python number `c`, through the `pow` op.
    fn __rpow__(
        &self,
        base: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for TraceTensor power",
            ));
        }
        trace_scalar_left_operand(self, base, "pow")
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn sum(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.reduce_axes_tensor(extract_reduction_axes(axis)?, keepdims, false)
            .map_err(PyValueError::new_err)
    }

    fn __matmul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.traced_operand(rhs).ok_or_else(|| {
            PyTypeError::new_err("expected a TraceTensor or Tensor matmul operand")
        })??;
        self.matmul_tensor(&rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__matmul__(rhs)
    }

    /// `a @ x` for an operand `a` whose `__matmul__` declined the tracer,
    /// such as a `TensorView`, which is captured as a constant.
    fn __rmatmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = self.traced_operand(lhs).ok_or_else(|| {
            PyTypeError::new_err("expected a TraceTensor or Tensor matmul operand")
        })??;
        lhs.matmul_tensor(self).map_err(PyValueError::new_err)
    }

    fn solve(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.array_operand(rhs, "solve")?;
        self.solve_tensor(&rhs).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (rhs, lower = true, transpose = false))]
    fn solve_triangular(
        &self,
        rhs: &Bound<'_, PyAny>,
        lower: bool,
        transpose: bool,
    ) -> PyResult<Self> {
        let rhs = self.array_operand(rhs, "solve_triangular")?;
        self.solve_triangular_tensor(&rhs, lower, transpose)
            .map_err(PyValueError::new_err)
    }

    fn cholesky(&self) -> PyResult<Self> {
        self.cholesky_tensor().map_err(PyValueError::new_err)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_tensor_or_scalar_operand(self, rhs, "greater")
    }

    /// Elementwise `self > rhs` as a `bool` tensor (unlike `gt`, which keeps
    /// returning a 0/1 float mask).
    fn greater(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_compare_operand(self, rhs, TensorComparison::Greater)
    }

    fn greater_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_compare_operand(self, rhs, TensorComparison::GreaterEqual)
    }

    fn less(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_compare_operand(self, rhs, TensorComparison::Less)
    }

    fn less_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_compare_operand(self, rhs, TensorComparison::LessEqual)
    }

    fn equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_compare_operand(self, rhs, TensorComparison::Equal)
    }

    fn not_equal(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        trace_compare_operand(self, rhs, TensorComparison::NotEqual)
    }

    // `==`/`!=` stay identity comparisons (see `Tensor`).
    fn __gt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.greater(rhs)
    }

    fn __ge__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.greater_equal(rhs)
    }

    fn __lt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.less(rhs)
    }

    fn __le__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.less_equal(rhs)
    }

    /// Defining ordering operators installs a rich comparison, which drops
    /// Python's default identity hash; restore it so tensors stay usable as
    /// dictionary keys.
    fn __hash__(slf: &Bound<'_, Self>) -> isize {
        slf.as_ptr() as isize
    }

    fn logical_and(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.array_operand(rhs, "logical_and")?;
        self.logical_tensor(&rhs, true)
            .map_err(PyValueError::new_err)
    }

    fn logical_or(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.array_operand(rhs, "logical_or")?;
        self.logical_tensor(&rhs, false)
            .map_err(PyValueError::new_err)
    }

    fn logical_not(&self) -> PyResult<Self> {
        self.logical_not_tensor().map_err(PyValueError::new_err)
    }

    fn __and__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.logical_and(rhs)
    }

    fn __or__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.logical_or(rhs)
    }

    fn __rand__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = self.array_operand(lhs, "logical_and")?;
        lhs.logical_tensor(self, true)
            .map_err(PyValueError::new_err)
    }

    fn __ror__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = self.array_operand(lhs, "logical_or")?;
        lhs.logical_tensor(self, false)
            .map_err(PyValueError::new_err)
    }

    fn __invert__(&self) -> PyResult<Self> {
        self.logical_not()
    }

    fn isnan(&self) -> PyResult<Self> {
        self.classify_tensor(true).map_err(PyValueError::new_err)
    }

    fn isfinite(&self) -> PyResult<Self> {
        self.classify_tensor(false).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn any(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.any_all_tensor(extract_reduction_axes(axis)?, keepdims, true)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn all(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.any_all_tensor(extract_reduction_axes(axis)?, keepdims, false)
            .map_err(PyValueError::new_err)
    }

    fn maximum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(rhs) = self.traced_operand(rhs) {
            return self.maximum_tensor(&rhs?).map_err(PyValueError::new_err);
        }
        if let Some(rhs) = extract_scalar(rhs) {
            return self.maximum_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected a TraceTensor, Tensor, or numeric scalar operand",
        ))
    }

    fn minimum(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(rhs) = self.traced_operand(rhs) {
            return self.minimum_tensor(&rhs?).map_err(PyValueError::new_err);
        }
        if let Some(rhs) = extract_scalar(rhs) {
            return self.minimum_scalar(rhs).map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected a TraceTensor, Tensor, or numeric scalar operand",
        ))
    }

    fn where_select(
        &self,
        on_true: &Bound<'_, PyAny>,
        on_false: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let on_true = self.array_operand(on_true, "where")?;
        let on_false = self.array_operand(on_false, "where")?;
        self.where_tensor(&on_true, &on_false)
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

    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let batch_offset = usize::from(self.batch_axis.is_some());
        let size = self.shape[batch_offset..].iter().product();
        let shape = parse_reshape_args(shape, size)?;
        self.reshape_tensor(shape).map_err(PyValueError::new_err)
    }

    fn slice(&self, axis: isize, start: usize, stop: usize) -> PyResult<Self> {
        self.slice_tensor(axis, start, stop)
            .map_err(PyValueError::new_err)
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch_offset = usize::from(self.batch_axis.is_some());
        let shape = &self.shape[batch_offset..];
        let plan = parse_index_plan(index, shape)?;
        let mut output = self
            .index_tensor(&plan.indices)
            .map_err(PyValueError::new_err)?;
        for (axis, positions) in &plan.gathers {
            output = output
                .gather_tensor(positions, *axis as isize)
                .map_err(PyValueError::new_err)?;
        }
        match plan.shape {
            Some(shape) => output.reshape_tensor(shape).map_err(PyValueError::new_err),
            None => Ok(output),
        }
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
        updates: &Bound<'_, PyAny>,
        axis: isize,
    ) -> PyResult<Self> {
        let actual_axis = usize::try_from(self.example_axis(axis).map_err(PyValueError::new_err)?)
            .map_err(|_| PyValueError::new_err("normalized tensor axis is negative"))?;
        let indices = parse_axis_indices(indices, self.shape[actual_axis])?;
        let updates = self.array_operand(updates, "scatter_add")?;
        self.scatter_add_tensor(&indices, &updates, axis)
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

    /// The traced tensor with its axes reversed, as `transpose()`.
    #[getter(T)]
    fn reversed_axes(&self) -> PyResult<Self> {
        self.transpose_tensor(None).map_err(PyValueError::new_err)
    }

    fn log(&self) -> PyResult<Self> {
        self.log_tensor().map_err(PyValueError::new_err)
    }

    fn log1p(&self) -> PyResult<Self> {
        self.log1p_tensor().map_err(PyValueError::new_err)
    }

    fn expm1(&self) -> PyResult<Self> {
        self.expm1_tensor().map_err(PyValueError::new_err)
    }

    fn erf(&self) -> PyResult<Self> {
        self.erf_tensor().map_err(PyValueError::new_err)
    }

    /// `atan2(self, x)` of an array or Python number `x`.
    fn atan2(&self, x: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(x) = self.traced_operand(x) {
            return self.binary(&x?, "atan2").map_err(PyValueError::new_err);
        }
        if let Some(x) = extract_scalar(x) {
            self.ensure_not_bool("atan2")
                .map_err(PyValueError::new_err)?;
            return self
                .scalar_binary(x, "atan2")
                .map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected a TraceTensor, Tensor, or numeric scalar operand",
        ))
    }

    /// The value with a zero derivative in every transform.
    fn stop_gradient(&self) -> PyResult<Self> {
        self.stop_gradient_tensor().map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis = None, reverse = false))]
    fn cumsum(&self, axis: Option<isize>, reverse: bool) -> PyResult<Self> {
        self.cumsum_tensor(axis, reverse)
            .map_err(PyValueError::new_err)
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

/// A binary operation of two traced values, which an eager `Tensor` method
/// forwards to its traced operand after capturing itself as a constant.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TracedBinary {
    /// An op name of [`TraceTensor::binary`].
    Arithmetic(&'static str),
    Compare(TensorComparison),
    Maximum,
    Minimum,
    Matmul,
    Solve,
    SolveTriangular {
        lower: bool,
        transpose: bool,
    },
    Logical {
        and: bool,
    },
}

impl TracedBinary {
    fn apply(self, lhs: &TraceTensor, rhs: &TraceTensor) -> Result<TraceTensor, String> {
        match self {
            Self::Arithmetic(op) => lhs.binary(rhs, op),
            Self::Compare(kind) => lhs.compare_tensor(rhs, kind),
            Self::Maximum => lhs.maximum_tensor(rhs),
            Self::Minimum => lhs.minimum_tensor(rhs),
            Self::Matmul => lhs.matmul_tensor(rhs),
            Self::Solve => lhs.solve_tensor(rhs),
            Self::SolveTriangular { lower, transpose } => {
                lhs.solve_triangular_tensor(rhs, lower, transpose)
            }
            Self::Logical { and } => lhs.logical_tensor(rhs, and),
        }
    }
}

/// `eager op tracer` (or `tracer op eager` when `eager_first` is false) for
/// an eager `Tensor` method whose `operand` is a tracer: the tensor becomes a
/// constant of the tracer's graph. `None` when `operand` is not a tracer.
pub(crate) fn eager_traced_binary(
    eager: &PyTensor,
    operand: &Bound<'_, PyAny>,
    op: TracedBinary,
    eager_first: bool,
) -> Option<PyResult<TraceTensor>> {
    let tracer = operand.extract::<PyRef<'_, TraceTensor>>().ok()?;
    let result = tracer.capture(eager).and_then(|constant| {
        if eager_first {
            op.apply(&constant, &tracer)
        } else {
            op.apply(&tracer, &constant)
        }
    });
    Some(result.map_err(PyValueError::new_err))
}

fn trace_tensor_or_scalar_operand(
    lhs: &TraceTensor,
    rhs: &Bound<'_, PyAny>,
    op: &str,
) -> PyResult<TraceTensor> {
    if let Some(rhs) = lhs.traced_operand(rhs) {
        return lhs.binary(&rhs?, op).map_err(PyValueError::new_err);
    }
    if let Some(value) = extract_scalar(rhs) {
        return lhs.scalar_binary(value, op).map_err(PyValueError::new_err);
    }
    Err(PyTypeError::new_err(
        "expected a TraceTensor, Tensor, or numeric scalar operand",
    ))
}

fn trace_input_is_bool(graph: &TensorTraceGraph, name: &str) -> PyResult<bool> {
    let ir = graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
    Ok(ir
        .input_node_id(name)
        .and_then(|node_id| ir.node_dtype(node_id))
        .is_ok_and(|dtype| dtype == TensorDType::Bool))
}

/// Symbolic VJP gradient maps omit `bool` inputs; report a request for one
/// as such instead of as a missing input.
fn ensure_differentiable_trace_input(graph: &TensorTraceGraph, name: &str) -> PyResult<()> {
    graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
        .ensure_differentiable_input(name)
        .map_err(PyValueError::new_err)
}

fn trace_compare_operand(
    lhs: &TraceTensor,
    rhs: &Bound<'_, PyAny>,
    kind: TensorComparison,
) -> PyResult<TraceTensor> {
    if let Some(rhs) = lhs.traced_operand(rhs) {
        return lhs
            .compare_tensor(&rhs?, kind)
            .map_err(PyValueError::new_err);
    }
    if let Some(value) = extract_scalar(rhs) {
        return lhs
            .compare_scalar(value, kind)
            .map_err(PyValueError::new_err);
    }
    Err(PyTypeError::new_err(
        "expected a TraceTensor, Tensor, or numeric scalar operand",
    ))
}

/// A reflected op (`lhs op rhs` with a traced `rhs`), reached when `lhs`
/// declined it: a Python number, or an eager array captured as a constant.
fn trace_scalar_left_operand(
    rhs: &TraceTensor,
    lhs: &Bound<'_, PyAny>,
    op: &str,
) -> PyResult<TraceTensor> {
    if let Some(lhs) = rhs.traced_operand(lhs) {
        return lhs?.binary(rhs, op).map_err(PyValueError::new_err);
    }
    let value = extract_scalar(lhs).ok_or_else(|| {
        PyTypeError::new_err("expected a Tensor or numeric scalar as the left TraceTensor operand")
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
            item.set_item("dtype", node.dtype.to_string())?;
            item.set_item("layout", node.layout)?;
            item.set_item("placement", node.placement.to_string())?;
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

    #[pyo3(signature = (inputs, mapped_input_names, shard_count, reduction = "mean"))]
    fn value_and_grad_data_parallel(
        &self,
        inputs: &Bound<'_, PyDict>,
        mapped_input_names: Vec<String>,
        shard_count: usize,
        reduction: &str,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let reduction = match reduction {
            "sum" => TensorReplicaReduction::Sum,
            "mean" => TensorReplicaReduction::Mean,
            _ => {
                return Err(PyValueError::new_err(
                    "data-parallel reduction must be 'sum' or 'mean'",
                ))
            }
        };
        let (value, gradients) = self
            .plan
            .value_and_vjp_data_parallel(
                &extract_tensor_map(inputs)?,
                DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
                &mapped_input_names.into_iter().collect(),
                shard_count,
                reduction,
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
        let retained_inputs = self
            .retained_inputs
            .lock()
            .map_err(|_| PyValueError::new_err("MLX retained input lock is poisoned"))?;
        let value = MlxBackend
            .execute_many_with_state(
                &self.plan,
                &[self.plan.output_node_id()],
                &extract_tensor_map(inputs)?,
                &retained_inputs,
            )
            .map_err(PyValueError::new_err)?;
        let value = value
            .into_iter()
            .next()
            .ok_or_else(|| PyValueError::new_err("MLX execution produced no output"))?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
    }

    /// Evaluate a fixed MLX plan and synchronize its GPU output without a host readback.
    fn evaluate_device(&self, inputs: &Bound<'_, PyDict>) -> PyResult<()> {
        let retained_inputs = self
            .retained_inputs
            .lock()
            .map_err(|_| PyValueError::new_err("MLX retained input lock is poisoned"))?;
        MlxBackend
            .execute_without_output_with_state(
                &self.plan,
                &extract_tensor_map(inputs)?,
                &retained_inputs,
            )
            .map_err(PyValueError::new_err)
    }

    /// Upload selected bindings once and reuse their MLX arrays on later calls.
    fn retain_inputs(&self, inputs: &Bound<'_, PyDict>, input_names: Vec<String>) -> PyResult<()> {
        let inputs = extract_tensor_map(inputs)?;
        let mut retained = self
            .retained_inputs
            .lock()
            .map_err(|_| PyValueError::new_err("MLX retained input lock is poisoned"))?;
        for name in input_names {
            if self.plan.input_shape(&name).is_err() {
                return Err(PyValueError::new_err(format!(
                    "cannot retain {name:?}: it is not a plan input"
                )));
            }
            let input = inputs.get(&name).ok_or_else(|| {
                PyValueError::new_err(format!("cannot retain missing input {name:?}"))
            })?;
            retained
                .replace(name, input)
                .map_err(PyValueError::new_err)?;
        }
        Ok(())
    }

    #[getter]
    fn retained_input_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .retained_inputs
            .lock()
            .map_err(|_| PyValueError::new_err("MLX retained input lock is poisoned"))?
            .names()
            .map(str::to_string)
            .collect())
    }

    fn clear_retained_inputs(&self) -> PyResult<()> {
        self.retained_inputs
            .lock()
            .map_err(|_| PyValueError::new_err("MLX retained input lock is poisoned"))?
            .clear();
        Ok(())
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
impl TensorVmapMlxVjpFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.plan.node_count()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        "mlx"
    }

    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        output_cotangent: &PyTensor,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let mut inputs = vmap_inputs(&self.signature, values)?;
        inputs.insert(
            self.cotangent_name.clone(),
            move_axis(
                output_cotangent
                    .to_dynamic_tensor()
                    .map_err(PyValueError::new_err)?,
                self.signature.out_axis,
                0,
            )
            .map_err(PyValueError::new_err)?,
        );
        let mut output_node_ids = Vec::with_capacity(self.gradient_node_ids.len() + 1);
        output_node_ids.push(self.output_node_id);
        output_node_ids.extend(self.gradient_node_ids.values().copied());
        let outputs = MlxBackend
            .execute_many(&self.plan.plan, &output_node_ids, &inputs)
            .map_err(PyValueError::new_err)?;
        let mut outputs = outputs.into_iter();
        let value = outputs
            .next()
            .ok_or_else(|| PyValueError::new_err("MLX vmap VJP produced no primal output"))?;
        let gradients = self
            .gradient_node_ids
            .keys()
            .cloned()
            .zip(outputs)
            .collect::<BTreeMap<_, _>>();
        Ok((
            vmap_output(&self.signature, value)?,
            vmap_gradient_outputs(&self.signature, gradients)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapMlxVjpFunction(parameter_count={}, node_count={})",
            self.gradient_node_ids.len(),
            self.plan.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorVmapMlxJvpFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.plan.node_count()
    }

    #[getter]
    fn backend(&self) -> &'static str {
        "mlx"
    }

    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let mut inputs = vmap_inputs(&self.signature, values)?;
        let mut tangents = vmap_inputs(&self.signature, input_tangents)?;
        for (input_name, tangent_name) in &self.tangent_names {
            let tangent = tangents.remove(input_name).ok_or_else(|| {
                PyValueError::new_err(format!("missing input tangent {input_name:?}"))
            })?;
            inputs.insert(tangent_name.clone(), tangent);
        }
        let outputs = MlxBackend
            .execute_many(
                &self.plan.plan,
                &[self.value_node_id, self.tangent_node_id],
                &inputs,
            )
            .map_err(PyValueError::new_err)?;
        let mut outputs = outputs.into_iter();
        let value = outputs
            .next()
            .ok_or_else(|| PyValueError::new_err("MLX vmap JVP produced no primal output"))?;
        let tangent = outputs
            .next()
            .ok_or_else(|| PyValueError::new_err("MLX vmap JVP produced no tangent output"))?;
        Ok((
            vmap_output(&self.signature, value)?,
            vmap_output(&self.signature, tangent)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapMlxJvpFunction(node_count={})",
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
impl TensorVmapCudaJvpFunction {
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
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let mut inputs = vmap_inputs(&self.signature, values)?;
        let mut tangents = vmap_inputs(&self.signature, input_tangents)?;
        for (input_name, tangent_name) in &self.tangent_names {
            let tangent = tangents.remove(input_name).ok_or_else(|| {
                PyValueError::new_err(format!("missing input tangent {input_name:?}"))
            })?;
            inputs.insert(tangent_name.clone(), tangent);
        }
        self.plan
            .plan
            .execute_retaining_without_output(&inputs, &BTreeSet::new())
            .map_err(PyValueError::new_err)?;
        let value = self
            .plan
            .plan
            .computed_node_to_host(self.value_node_id)
            .map_err(PyValueError::new_err)?;
        let tangent = self
            .plan
            .plan
            .computed_node_to_host(self.tangent_node_id)
            .map_err(PyValueError::new_err)?;
        Ok((
            vmap_output(&self.signature, value)?,
            vmap_output(&self.signature, tangent)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapCudaJvpFunction(node_count={})",
            self.plan.plan.node_count()
        )
    }
}

#[pymethods]
impl TensorVmapCudaHvpScalarFunction {
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

    fn __call__(&self, values: &Bound<'_, PyDict>, input_tangent: &PyTensor) -> PyResult<PyTensor> {
        let inputs = vmap_hvp_inputs(
            &self.signature,
            values,
            &self.cotangent_name,
            &self.tangent_name,
            self.input_axis,
            input_tangent,
        )?;
        let value = self
            .plan
            .plan
            .execute(&inputs)
            .map_err(PyValueError::new_err)?;
        vmap_hvp_output(value, self.input_axis)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapCudaHvpScalarFunction(input_name={:?}, node_count={})",
            self.input_name,
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
impl TensorCondFunction {
    fn __call__(&self, predicate: bool, values: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        PyTensor::from_dynamic_tensor(
            self.plan
                .evaluate(predicate, &extract_tensor_map(values)?)
                .map_err(PyValueError::new_err)?,
        )
        .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorCondFunction(true_node_count={}, false_node_count={})",
            self.plan.true_node_count(),
            self.plan.false_node_count()
        )
    }
}

#[pymethods]
impl TensorCondValueAndGradFunction {
    fn __call__(
        &self,
        predicate: bool,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let (value, gradients) = self
            .plan
            .value_and_vjp(
                predicate,
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
            "TensorCondValueAndGradFunction(true_node_count={}, false_node_count={})",
            self.plan.true_node_count(),
            self.plan.false_node_count()
        )
    }
}

#[pymethods]
impl TensorCondJvpFunction {
    fn __call__(
        &self,
        predicate: bool,
        values: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (value, tangent) = self
            .plan
            .jvp(
                predicate,
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
        format!(
            "TensorCondJvpFunction(true_node_count={}, false_node_count={})",
            self.plan.true_node_count(),
            self.plan.false_node_count()
        )
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
            // Batch specialization keys the cache by batch size, so inputs are always traced as
            // f64; f32 values widen losslessly.
            input_specs.push(TensorInputSpec::new(name.clone(), shape, TensorDType::F64));
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
impl TensorBatchValueAndGradFunction {
    fn __call__(
        &self,
        py: Python<'_>,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(values)?;
        if inputs.len() != self.input_names.len()
            || self
                .input_names
                .iter()
                .any(|name| !inputs.contains_key(name))
        {
            return Err(PyValueError::new_err(format!(
                "tensor_value_and_grad_batch_fn expects exactly inputs {:?}",
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
            // Batch specialization keys the cache by batch size, so inputs are always traced as
            // f64; f32 values widen losslessly.
            input_specs.push(TensorInputSpec::new(name.clone(), shape, TensorDType::F64));
        }
        let batch_size = batch_size.ok_or_else(|| {
            PyValueError::new_err(
                "tensor_value_and_grad_batch_fn requires at least one mapped input axis",
            )
        })?;
        let mut expected_shapes = self.static_shapes.borrow_mut();
        if let Some(expected) = expected_shapes.as_ref() {
            if expected != &static_shapes {
                return Err(PyValueError::new_err(format!(
                    "tensor_value_and_grad_batch_fn only specializes mapped batch axes; expected non-batch shapes {expected:?}, got {static_shapes:?}"
                )));
            }
        } else {
            *expected_shapes = Some(static_shapes);
        }
        drop(expected_shapes);

        let cached_plan = self.plans.borrow().get(&batch_size).cloned();
        let plan = if let Some(plan) = cached_plan {
            plan
        } else {
            if self.plans.borrow().len() >= self.max_specializations {
                return Err(PyValueError::new_err(format!(
                    "tensor_value_and_grad_batch_fn reached max_specializations={} before batch size {batch_size}",
                    self.max_specializations
                )));
            }
            let traced = trace_tensor_python_function(py, self.function.bind(py), input_specs)?;
            if !traced.output.shape.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "tensor_value_and_grad_batch_fn requires a scalar output, got shape {:?}",
                    traced.output.shape
                )));
            }
            let plan = traced
                .graph
                .compile_cpu_plan(traced.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan;
            self.plans.borrow_mut().insert(batch_size, plan.clone());
            plan
        };
        let (value, gradients) = plan
            .value_and_vjp(
                &inputs,
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
impl TensorBatchMlxValueAndGradFunction {
    fn __call__(
        &self,
        py: Python<'_>,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(values)?;
        if inputs.len() != self.input_names.len()
            || self
                .input_names
                .iter()
                .any(|name| !inputs.contains_key(name))
        {
            return Err(PyValueError::new_err(format!(
                "tensor_value_and_grad_batch_mlx_fn expects exactly inputs {:?}",
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
            // Batch specialization keys the cache by batch size, so inputs are always traced as
            // f64; f32 values widen losslessly.
            input_specs.push(TensorInputSpec::new(name.clone(), shape, TensorDType::F64));
        }
        let batch_size = batch_size.ok_or_else(|| {
            PyValueError::new_err(
                "tensor_value_and_grad_batch_mlx_fn requires at least one mapped input axis",
            )
        })?;
        let mut expected_shapes = self.static_shapes.borrow_mut();
        if let Some(expected) = expected_shapes.as_ref() {
            if expected != &static_shapes {
                return Err(PyValueError::new_err(format!(
                    "tensor_value_and_grad_batch_mlx_fn only specializes mapped batch axes; expected non-batch shapes {expected:?}, got {static_shapes:?}"
                )));
            }
        } else {
            *expected_shapes = Some(static_shapes);
        }
        drop(expected_shapes);

        if !self.plans.borrow().contains_key(&batch_size) {
            if self.plans.borrow().len() >= self.max_specializations {
                return Err(PyValueError::new_err(format!(
                    "tensor_value_and_grad_batch_mlx_fn reached max_specializations={} before batch size {batch_size}",
                    self.max_specializations
                )));
            }
            let traced = trace_tensor_python_function(py, self.function.bind(py), input_specs)?;
            let (plan, loss_node_id, gradient_node_ids) =
                compile_mlx_scalar_value_and_grad(&traced, self.parameter_names.clone())?;
            self.plans.borrow_mut().insert(
                batch_size,
                MlxBatchValueAndGradPlan {
                    plan,
                    loss_node_id,
                    gradient_node_ids,
                },
            );
        }
        let plan = self.plans.borrow();
        let plan = plan.get(&batch_size).ok_or_else(|| {
            PyValueError::new_err("MLX batch value-and-grad specialization was not cached")
        })?;
        execute_mlx_value_and_grad_plan(plan, inputs)
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
impl TensorBatchCudaJitFunction {
    fn __call__(&self, py: Python<'_>, values: &Bound<'_, PyDict>) -> PyResult<PyTensor> {
        let inputs = extract_tensor_map(values)?;
        let (batch_size, static_shapes, input_specs) = batch_specialization_signature(
            &inputs,
            &self.input_names,
            &self.input_axes,
            "tensor_jit_batch_cuda_fn",
        )?;
        validate_batch_static_shapes(
            &self.static_shapes,
            static_shapes,
            "tensor_jit_batch_cuda_fn",
        )?;
        if !self.plans.borrow().contains_key(&batch_size) {
            if self.plans.borrow().len() >= self.max_specializations {
                return Err(PyValueError::new_err(format!(
                    "tensor_jit_batch_cuda_fn reached max_specializations={} before batch size {batch_size}",
                    self.max_specializations
                )));
            }
            let traced = trace_tensor_python_function(py, self.function.bind(py), input_specs)?;
            let plan = traced
                .graph
                .compile_cuda_plan(traced.output.node_id, self.device_ordinal)
                .map_err(PyValueError::new_err)?;
            self.plans.borrow_mut().insert(batch_size, plan);
        }
        let plans = self.plans.borrow();
        let plan = plans
            .get(&batch_size)
            .ok_or_else(|| PyValueError::new_err("CUDA batch JIT specialization was not cached"))?;
        let value = plan.plan.execute(&inputs).map_err(PyValueError::new_err)?;
        PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)
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
impl TensorBatchCudaValueAndGradFunction {
    fn __call__(
        &self,
        py: Python<'_>,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(values)?;
        let (batch_size, static_shapes, input_specs) = batch_specialization_signature(
            &inputs,
            &self.input_names,
            &self.input_axes,
            "tensor_value_and_grad_batch_cuda_fn",
        )?;
        validate_batch_static_shapes(
            &self.static_shapes,
            static_shapes,
            "tensor_value_and_grad_batch_cuda_fn",
        )?;
        if !self.plans.borrow().contains_key(&batch_size) {
            if self.plans.borrow().len() >= self.max_specializations {
                return Err(PyValueError::new_err(format!(
                    "tensor_value_and_grad_batch_cuda_fn reached max_specializations={} before batch size {batch_size}",
                    self.max_specializations
                )));
            }
            let traced = trace_tensor_python_function(py, self.function.bind(py), input_specs)?;
            let (plan, loss_node_id, gradient_node_ids) = compile_cuda_scalar_value_and_grad(
                &traced,
                self.parameter_names.clone(),
                self.device_ordinal,
            )?;
            self.plans.borrow_mut().insert(
                batch_size,
                CudaBatchValueAndGradPlan {
                    plan,
                    loss_node_id,
                    gradient_node_ids,
                },
            );
        }
        let plans = self.plans.borrow();
        let plan = plans.get(&batch_size).ok_or_else(|| {
            PyValueError::new_err("CUDA batch value-and-grad specialization was not cached")
        })?;
        execute_cuda_value_and_grad_plan(plan, inputs)
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

fn batch_specialization_signature(
    inputs: &BTreeMap<String, DynamicTensor>,
    input_names: &[String],
    input_axes: &[Option<isize>],
    function_name: &str,
) -> PyResult<BatchSpecializationSignature> {
    if inputs.len() != input_names.len()
        || input_names.iter().any(|name| !inputs.contains_key(name))
    {
        return Err(PyValueError::new_err(format!(
            "{function_name} expects exactly inputs {input_names:?}"
        )));
    }
    let mut batch_size = None;
    let mut static_shapes = Vec::with_capacity(input_names.len());
    let mut input_specs = Vec::with_capacity(input_names.len());
    for ((name, axis), tensor) in input_names
        .iter()
        .zip(input_axes)
        .zip(input_names.iter().map(|name| &inputs[name]))
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
        // Batch specialization keys the cache by batch size, so inputs are always traced as f64;
        // f32 values widen losslessly.
        input_specs.push(TensorInputSpec::new(name.clone(), shape, TensorDType::F64));
    }
    let batch_size = batch_size.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{function_name} requires at least one mapped input axis"
        ))
    })?;
    Ok((batch_size, static_shapes, input_specs))
}

fn validate_batch_static_shapes(
    expected_shapes: &RefCell<Option<Vec<Vec<usize>>>>,
    static_shapes: Vec<Vec<usize>>,
    function_name: &str,
) -> PyResult<()> {
    let mut expected = expected_shapes.borrow_mut();
    if let Some(expected) = expected.as_ref() {
        if expected != &static_shapes {
            return Err(PyValueError::new_err(format!(
                "{function_name} only specializes mapped batch axes; expected non-batch shapes {expected:?}, got {static_shapes:?}"
            )));
        }
    } else {
        *expected = Some(static_shapes);
    }
    Ok(())
}

fn execute_cuda_value_and_grad_plan(
    plan: &CudaBatchValueAndGradPlan,
    mut inputs: BTreeMap<String, DynamicTensor>,
) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
    inputs.insert(
        CUDA_LOSS_COTANGENT_NAME.to_string(),
        DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
    );
    plan.plan
        .plan
        .execute_retaining_without_output(&inputs, &BTreeSet::new())
        .map_err(PyValueError::new_err)?;
    let loss = plan
        .plan
        .plan
        .computed_node_to_host(plan.loss_node_id)
        .map_err(PyValueError::new_err)?;
    let gradients = plan
        .gradient_node_ids
        .iter()
        .map(|(name, node_id)| {
            plan.plan
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
impl TensorVmapHvpScalarFunction {
    #[getter]
    fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    fn __call__(&self, values: &Bound<'_, PyDict>, input_tangent: &PyTensor) -> PyResult<PyTensor> {
        let inputs = vmap_hvp_inputs(
            &self.signature,
            values,
            &self.cotangent_name,
            &self.tangent_name,
            self.input_axis,
            input_tangent,
        )?;
        let value = self.plan.evaluate(&inputs).map_err(PyValueError::new_err)?;
        vmap_hvp_output(value, self.input_axis)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVmapHvpScalarFunction(input_name={:?}, node_count={})",
            self.input_name,
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
impl TensorMlxValueAndGradFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let mut inputs = extract_tensor_map(values)?;
        inputs.insert(
            self.cotangent_name.clone(),
            DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
        );
        let mut output_node_ids = Vec::with_capacity(self.gradient_node_ids.len() + 1);
        output_node_ids.push(self.loss_node_id);
        output_node_ids.extend(self.gradient_node_ids.values().copied());
        let outputs = MlxBackend
            .execute_many(&self.plan.plan, &output_node_ids, &inputs)
            .map_err(PyValueError::new_err)?;
        let mut outputs = outputs.into_iter();
        let loss = outputs
            .next()
            .ok_or_else(|| PyValueError::new_err("MLX value-and-grad produced no loss"))?;
        let gradients = self
            .gradient_node_ids
            .keys()
            .cloned()
            .zip(outputs)
            .map(|(name, gradient)| {
                PyTensor::from_dynamic_tensor(gradient).map(|tensor| (name, tensor))
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
            "TensorMlxValueAndGradFunction(parameter_count={}, node_count={})",
            self.gradient_node_ids.len(),
            self.plan.plan.node_count()
        )
    }
}

fn execute_mlx_value_and_grad_plan(
    plan: &MlxBatchValueAndGradPlan,
    mut inputs: BTreeMap<String, DynamicTensor>,
) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
    inputs.insert(
        MLX_LOSS_COTANGENT_NAME.to_string(),
        DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
    );
    let mut output_node_ids = Vec::with_capacity(plan.gradient_node_ids.len() + 1);
    output_node_ids.push(plan.loss_node_id);
    output_node_ids.extend(plan.gradient_node_ids.values().copied());
    let outputs = MlxBackend
        .execute_many(&plan.plan.plan, &output_node_ids, &inputs)
        .map_err(PyValueError::new_err)?;
    let mut outputs = outputs.into_iter();
    let loss = outputs
        .next()
        .ok_or_else(|| PyValueError::new_err("MLX value-and-grad produced no loss"))?;
    let gradients = plan
        .gradient_node_ids
        .keys()
        .cloned()
        .zip(outputs)
        .map(|(name, gradient)| {
            PyTensor::from_dynamic_tensor(gradient).map(|tensor| (name, tensor))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(PyValueError::new_err)?;
    Ok((
        PyTensor::from_dynamic_tensor(loss).map_err(PyValueError::new_err)?,
        gradients,
    ))
}

#[pymethods]
impl TensorMlxAdamOptimizer {
    #[pyo3(signature = (inputs = None))]
    fn step(&mut self, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(inputs) = inputs {
            self.inputs.extend(extract_tensor_map(inputs)?);
        }
        self.plan.step(&self.inputs).map_err(PyValueError::new_err)
    }

    /// Learning rate of the next step; host schedules set it between steps.
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.plan.learning_rate()
    }

    #[setter]
    fn set_learning_rate(&mut self, learning_rate: f32) -> PyResult<()> {
        self.plan
            .set_learning_rate(learning_rate)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (inputs = None))]
    fn loss(&mut self, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
        if let Some(inputs) = inputs {
            self.inputs.extend(extract_tensor_map(inputs)?);
        }
        self.plan
            .loss(&self.inputs)
            .and_then(|value| {
                PyTensor::from_dynamic_tensor(value).map_err(|error| error.to_string())
            })
            .map_err(PyValueError::new_err)
    }

    fn parameters(&self) -> PyResult<BTreeMap<String, PyTensor>> {
        self.parameter_names
            .iter()
            .map(|name| {
                self.plan
                    .parameter(name)
                    .and_then(|value| {
                        PyTensor::from_dynamic_tensor(value).map_err(|error| error.to_string())
                    })
                    .map(|value| (name.clone(), value))
                    .map_err(PyValueError::new_err)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorMlxAdamOptimizer(parameter_count={})",
            self.parameter_names.len()
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
    input_specs: Vec<TensorInputSpec>,
) -> PyResult<TensorTraceResult> {
    let graph = TensorTraceGraph::new();
    let mut inputs = Vec::with_capacity(input_specs.len());
    for spec in input_specs {
        inputs.push(
            graph
                .add_input(&spec.name, spec.shape, spec.dtype)
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

/// Compiles a traced region, keeping the trace for `retain_region_inputs`.
fn compile_cpu_region(
    traced: &TensorTraceResult,
) -> Result<(&TensorTraceResult, TensorExecutionPlan), String> {
    Ok((
        traced,
        traced.graph.compile_cpu_plan(traced.output.node_id)?.plan,
    ))
}

fn region_input_names(plan: &TensorExecutionPlan) -> Vec<String> {
    TensorRegion::new(plan.clone())
        .captures()
        .keys()
        .cloned()
        .collect()
}

/// Recompiles a region whose output also references `names`, or returns the
/// compiled plan unchanged when nothing is missing.
fn retain_region_inputs(
    (traced, plan): (&TensorTraceResult, TensorExecutionPlan),
    names: &[String],
) -> Result<TensorExecutionPlan, String> {
    if names.is_empty() {
        return Ok(plan);
    }
    let output = traced
        .graph
        .ir
        .lock()
        .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
        .retain_inputs(names, traced.output.node_id)?;
    Ok(traced.graph.compile_cpu_plan(output)?.plan)
}

/// Traces a lazy scalar conditional into the parent Tensor IR.
///
/// Branches are separate region traces. They can only depend on the explicit
/// `operands`, which become ordered capture bindings in the parent graph.
#[pyfunction]
pub fn tensor_cond(
    py: Python<'_>,
    predicate: TraceTensor,
    on_true: &Bound<'_, PyAny>,
    on_false: &Bound<'_, PyAny>,
    operands: Vec<TraceTensor>,
) -> PyResult<TraceTensor> {
    if !predicate.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "tensor_cond predicate must be scalar, got shape {:?}",
            predicate.shape
        )));
    }
    if predicate.batch_axis.is_some() || operands.iter().any(|operand| operand.batch_axis.is_some())
    {
        return Err(PyValueError::new_err(
            "tensor_cond does not yet support vmapped predicates or operands",
        ));
    }
    for operand in &operands {
        predicate
            .same_graph(operand)
            .map_err(PyValueError::new_err)?;
    }

    let input_specs = operands
        .iter()
        .enumerate()
        .map(|(index, operand)| {
            Ok(TensorInputSpec::new(
                format!("__quabla_cond_capture_{index}"),
                operand.shape.clone(),
                operand.dtype().map_err(PyValueError::new_err)?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let on_true = trace_tensor_python_function(py, on_true, input_specs.clone())?;
    let on_false = trace_tensor_python_function(py, on_false, input_specs)?;
    let on_true = compile_cpu_region(&on_true).map_err(PyValueError::new_err)?;
    let on_false = compile_cpu_region(&on_false).map_err(PyValueError::new_err)?;
    // A branch may ignore an operand the other branch reads; retaining it
    // gives both regions the same capture interface without reading it.
    let missing_from = |plan: &TensorExecutionPlan, other: &TensorExecutionPlan| {
        region_input_names(other)
            .into_iter()
            .filter(|name| plan.input_shape(name).is_err())
            .collect::<Vec<_>>()
    };
    let true_missing = missing_from(&on_true.1, &on_false.1);
    let false_missing = missing_from(&on_false.1, &on_true.1);
    let branches = TensorCondExecutionPlan::new(
        retain_region_inputs(on_true, &true_missing).map_err(PyValueError::new_err)?,
        retain_region_inputs(on_false, &false_missing).map_err(PyValueError::new_err)?,
    )
    .map_err(PyValueError::new_err)?;
    let captures = branches
        .captures()
        .keys()
        .map(|name| {
            let index = name
                .strip_prefix("__quabla_cond_capture_")
                .ok_or_else(|| {
                    PyValueError::new_err("conditional branch capture is not an operand")
                })?
                .parse::<usize>()
                .map_err(|_| {
                    PyValueError::new_err("conditional branch capture index is invalid")
                })?;
            let operand = operands.get(index).ok_or_else(|| {
                PyValueError::new_err("conditional branch capture index is out of range")
            })?;
            Ok((name.clone(), operand.node_id))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let graph = predicate.graph.clone();
    let mut ir = graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
    let node_id = ir
        .cond_with_captures(predicate.node_id, branches, captures)
        .map_err(PyValueError::new_err)?;
    let shape = ir.node_shape(node_id).map_err(PyValueError::new_err)?;
    drop(ir);
    Ok(TraceTensor::from_node(graph, node_id, shape, None))
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
    input_specs: &[TensorInputSpec],
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
        .map(|(spec, axis)| match axis {
            Some(axis) => normalize_vmap_axis(axis, spec.shape.len() + 1, &spec.name).map(Some),
            None => Ok(None),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(VmapSignature {
        input_names: input_specs.iter().map(|spec| spec.name.clone()).collect(),
        in_axes,
        out_axis: normalize_vmap_axis(out_axis, output_rank, "out_axes")?,
    })
}

fn trace_tensor_vmap_python_function(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
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
    for (spec, axis) in input_specs.iter().zip(&input_axes) {
        let tensor = if axis.is_some() {
            let mut batched_shape = Vec::with_capacity(spec.shape.len() + 1);
            batched_shape.push(batch_size);
            batched_shape.extend(spec.shape.iter().copied());
            graph.add_batched_input(&spec.name, batched_shape, spec.dtype)
        } else {
            graph.add_input(&spec.name, spec.shape.clone(), spec.dtype)
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
    input_specs: Vec<TensorInputSpec>,
) -> PyResult<TensorTraceResult> {
    trace_tensor_python_function(py, function, input_specs)
}

/// Statically unrolls a differentiable loop into the current Tensor IR.
///
/// The bounds are host integers, so this does not model data-dependent loop
/// control flow. The carry must retain its graph and shape on every iteration.
#[pyfunction]
pub fn tensor_fori_loop(
    py: Python<'_>,
    lower: usize,
    upper: usize,
    body: &Bound<'_, PyAny>,
    init: TraceTensor,
) -> PyResult<TraceTensor> {
    if upper < lower {
        return Err(PyValueError::new_err(format!(
            "tensor_fori_loop requires upper >= lower, got {upper} < {lower}"
        )));
    }
    let expected_graph = init.graph.clone();
    let expected_shape = init.shape.clone();
    let mut carry = init;
    for index in lower..upper {
        let argument = Py::new(py, carry.clone())?;
        let output = body.call1((index, argument))?;
        let next = output
            .extract::<PyRef<'_, TraceTensor>>()
            .map_err(|_| PyTypeError::new_err("tensor_fori_loop body must return a TraceTensor"))?;
        if !Arc::ptr_eq(&expected_graph.ir, &next.graph.ir) {
            return Err(PyValueError::new_err(
                "tensor_fori_loop body returned a TraceTensor from a different graph",
            ));
        }
        if next.shape != expected_shape {
            return Err(PyValueError::new_err(format!(
                "tensor_fori_loop body changed carry shape from {expected_shape:?} to {:?}",
                next.shape
            )));
        }
        carry = next.clone();
    }
    Ok(carry)
}

/// Traces a fixed-bounds loop body once into a runtime Tensor IR region.
///
/// `body` receives `(index, carry, *operands)`, where `index` is a scalar
/// TraceTensor. All external values must be provided as explicit operands.
#[pyfunction]
pub fn tensor_fori_loop_region(
    py: Python<'_>,
    lower: usize,
    upper: usize,
    body: &Bound<'_, PyAny>,
    init: TraceTensor,
    operands: Vec<TraceTensor>,
) -> PyResult<TraceTensor> {
    if upper < lower {
        return Err(PyValueError::new_err(format!(
            "tensor_fori_loop_region requires upper >= lower, got {upper} < {lower}"
        )));
    }
    if init.batch_axis.is_none() && operands.iter().any(|operand| operand.batch_axis.is_some()) {
        return Err(PyValueError::new_err(
            "tensor_fori_loop_region requires a vmapped carry when an operand is vmapped",
        ));
    }
    if init.batch_axis.is_some_and(|axis| axis != 0)
        || operands
            .iter()
            .any(|operand| operand.batch_axis.is_some_and(|axis| axis != 0))
    {
        return Err(PyValueError::new_err(
            "tensor_fori_loop_region only supports canonical batch axis zero",
        ));
    }
    if let Some(batch_size) = init.batch_axis.map(|_| init.shape[0]) {
        for operand in &operands {
            if operand.batch_axis.is_some() && operand.shape.first() != Some(&batch_size) {
                return Err(PyValueError::new_err(
                    "tensor_fori_loop_region vmapped operands must share the carry batch size",
                ));
            }
        }
    }
    for operand in &operands {
        init.same_graph(operand).map_err(PyValueError::new_err)?;
    }

    let body_graph = TensorTraceGraph::new();
    // The loop index adopts the carry dtype and behaves like a weak scalar: loops over an f32 carry
    // stay f32.
    let carry_dtype = init.dtype().map_err(PyValueError::new_err)?;
    let carry = if init.batch_axis.is_some() {
        body_graph.add_batched_input("__quabla_fori_carry", init.shape.clone(), carry_dtype)
    } else {
        body_graph.add_input("__quabla_fori_carry", init.shape.clone(), carry_dtype)
    }
    .map_err(PyValueError::new_err)?;
    let index = body_graph
        .add_input("__quabla_fori_index", vec![], carry_dtype)
        .map_err(PyValueError::new_err)?;
    let captures = operands
        .iter()
        .enumerate()
        .map(|(index, operand)| {
            let dtype = operand.dtype().map_err(PyValueError::new_err)?;
            if operand.batch_axis.is_some() {
                body_graph.add_batched_input(
                    &format!("__quabla_fori_capture_{index}"),
                    operand.shape.clone(),
                    dtype,
                )
            } else {
                body_graph.add_input(
                    &format!("__quabla_fori_capture_{index}"),
                    operand.shape.clone(),
                    dtype,
                )
            }
            .map_err(PyValueError::new_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut arguments = vec![index, carry];
    arguments.extend(captures);
    let output: TraceTensor = body
        .call1(PyTuple::new(py, arguments)?)?
        .extract()
        .map_err(|_| {
            PyTypeError::new_err("tensor_fori_loop_region body must return a TraceTensor")
        })?;
    if !Arc::ptr_eq(&body_graph.ir, &output.graph.ir) {
        return Err(PyValueError::new_err(
            "tensor_fori_loop_region body returned a TraceTensor from a different graph",
        ));
    }
    if output.batch_axis != init.batch_axis {
        return Err(PyValueError::new_err(
            "tensor_fori_loop_region body output must preserve the carry batch axis",
        ));
    }
    if output.shape != init.shape {
        return Err(PyValueError::new_err(format!(
            "tensor_fori_loop_region body changed carry shape from {:?} to {:?}",
            init.shape, output.shape
        )));
    }
    let loop_plan = TensorForiExecutionPlan::new(
        lower,
        upper,
        body_graph
            .compile_cpu_plan(output.node_id)
            .map_err(PyValueError::new_err)?
            .plan,
        "__quabla_fori_carry",
        "__quabla_fori_index",
    )
    .map_err(PyValueError::new_err)?;
    let parent_graph = init.graph.clone();
    // The compiled body drops operands it never reads, so bind only the
    // captures the body still declares; an ignored operand is legal.
    let captures = operands
        .iter()
        .enumerate()
        .map(|(index, operand)| (format!("__quabla_fori_capture_{index}"), operand.node_id))
        .filter(|(name, _)| loop_plan.external_captures().contains_key(name))
        .collect();
    let mut ir = parent_graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
    let node_id = ir
        .fori(init.node_id, loop_plan, captures)
        .map_err(PyValueError::new_err)?;
    let shape = ir.node_shape(node_id).map_err(PyValueError::new_err)?;
    drop(ir);
    Ok(TraceTensor::from_node(
        parent_graph,
        node_id,
        shape,
        init.batch_axis,
    ))
}

/// Traces a fixed-bounds carry/output scan body once into a runtime Tensor IR
/// region. `body` receives `(index, carry, *operands)` and returns
/// `(next_carry, output)`.
#[pyfunction]
pub fn tensor_scan_region(
    py: Python<'_>,
    lower: usize,
    upper: usize,
    body: &Bound<'_, PyAny>,
    init: TraceTensor,
    operands: Vec<TraceTensor>,
) -> PyResult<Py<PyTuple>> {
    if upper <= lower {
        return Err(PyValueError::new_err(format!(
            "tensor_scan_region requires upper > lower, got {upper} <= {lower}"
        )));
    }
    if init.batch_axis.is_none() && operands.iter().any(|operand| operand.batch_axis.is_some()) {
        return Err(PyValueError::new_err(
            "tensor_scan_region requires a vmapped carry when an operand is vmapped",
        ));
    }
    if init.batch_axis.is_some_and(|axis| axis != 0)
        || operands
            .iter()
            .any(|operand| operand.batch_axis.is_some_and(|axis| axis != 0))
    {
        return Err(PyValueError::new_err(
            "tensor_scan_region only supports canonical batch axis zero",
        ));
    }
    if let Some(batch_size) = init.batch_axis.map(|_| init.shape[0]) {
        for operand in &operands {
            if operand.batch_axis.is_some() && operand.shape.first() != Some(&batch_size) {
                return Err(PyValueError::new_err(
                    "tensor_scan_region vmapped operands must share the carry batch size",
                ));
            }
        }
    }
    for operand in &operands {
        init.same_graph(operand).map_err(PyValueError::new_err)?;
    }

    let body_graph = TensorTraceGraph::new();
    // The loop index adopts the carry dtype and behaves like a weak scalar: loops over an f32 carry
    // stay f32.
    let carry_dtype = init.dtype().map_err(PyValueError::new_err)?;
    let carry = if init.batch_axis.is_some() {
        body_graph.add_batched_input("__quabla_scan_carry", init.shape.clone(), carry_dtype)
    } else {
        body_graph.add_input("__quabla_scan_carry", init.shape.clone(), carry_dtype)
    }
    .map_err(PyValueError::new_err)?;
    let index = body_graph
        .add_input("__quabla_scan_index", vec![], carry_dtype)
        .map_err(PyValueError::new_err)?;
    let body_captures = operands
        .iter()
        .enumerate()
        .map(|(index, operand)| {
            let dtype = operand.dtype().map_err(PyValueError::new_err)?;
            if operand.batch_axis.is_some() {
                body_graph.add_batched_input(
                    &format!("__quabla_scan_capture_{index}"),
                    operand.shape.clone(),
                    dtype,
                )
            } else {
                body_graph.add_input(
                    &format!("__quabla_scan_capture_{index}"),
                    operand.shape.clone(),
                    dtype,
                )
            }
            .map_err(PyValueError::new_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut arguments = vec![index, carry];
    arguments.extend(body_captures);
    let result = body.call1(PyTuple::new(py, arguments)?)?;
    let result = result.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err("tensor_scan_region body must return a (carry, output) tuple")
    })?;
    if result.len() != 2 {
        return Err(PyTypeError::new_err(
            "tensor_scan_region body must return exactly two values",
        ));
    }
    let next = result
        .get_item(0)?
        .extract::<PyRef<'_, TraceTensor>>()
        .map_err(|_| PyTypeError::new_err("tensor_scan_region carry must be a TraceTensor"))?;
    let output = result
        .get_item(1)?
        .extract::<PyRef<'_, TraceTensor>>()
        .map_err(|_| PyTypeError::new_err("tensor_scan_region output must be a TraceTensor"))?;
    if !Arc::ptr_eq(&body_graph.ir, &next.graph.ir)
        || !Arc::ptr_eq(&body_graph.ir, &output.graph.ir)
    {
        return Err(PyValueError::new_err(
            "tensor_scan_region body returned a TraceTensor from a different graph",
        ));
    }
    if next.batch_axis != init.batch_axis || output.batch_axis != init.batch_axis {
        return Err(PyValueError::new_err(
            "tensor_scan_region body results must preserve the carry batch axis",
        ));
    }
    if next.shape != init.shape {
        return Err(PyValueError::new_err(format!(
            "tensor_scan_region body changed carry shape from {:?} to {:?}",
            init.shape, next.shape
        )));
    }
    let body_plan = body_graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
        .compile_cpu_many(&[next.node_id, output.node_id])
        .map_err(PyValueError::new_err)?
        .0;
    let scan_plan = TensorScanExecutionPlan::new(
        lower,
        upper,
        body_plan,
        "__quabla_scan_carry",
        "__quabla_scan_index",
    )
    .map_err(PyValueError::new_err)?;
    let parent_graph = init.graph.clone();
    // The compiled body drops operands it never reads, so bind only the
    // captures the body still declares; an ignored operand is legal.
    let captures = operands
        .iter()
        .enumerate()
        .map(|(index, operand)| (format!("__quabla_scan_capture_{index}"), operand.node_id))
        .filter(|(name, _)| scan_plan.external_captures().contains_key(name))
        .collect();
    let mut ir = parent_graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
    let (carry_node_id, scan_output_node_id) = ir
        .scan(init.node_id, scan_plan, captures)
        .map_err(PyValueError::new_err)?;
    let carry_shape = ir
        .node_shape(carry_node_id)
        .map_err(PyValueError::new_err)?;
    let output_node_id = if init.batch_axis.is_some() {
        let scan_shape = ir
            .node_shape(scan_output_node_id)
            .map_err(PyValueError::new_err)?;
        let axes = std::iter::once(1)
            .chain(std::iter::once(0))
            .chain(2..scan_shape.len())
            .map(|axis| axis as isize)
            .collect();
        ir.transpose(scan_output_node_id, Some(axes))
            .map_err(PyValueError::new_err)?
    } else {
        scan_output_node_id
    };
    let output_shape = ir
        .node_shape(output_node_id)
        .map_err(PyValueError::new_err)?;
    drop(ir);
    Ok(PyTuple::new(
        py,
        [
            TraceTensor::from_node(
                parent_graph.clone(),
                carry_node_id,
                carry_shape,
                init.batch_axis,
            ),
            TraceTensor::from_node(parent_graph, output_node_id, output_shape, init.batch_axis),
        ],
    )?
    .unbind())
}

/// Statically unrolls a carry/output scan into the current Tensor IR.
#[pyfunction]
pub fn tensor_scan(
    py: Python<'_>,
    length: usize,
    body: &Bound<'_, PyAny>,
    init: TraceTensor,
) -> PyResult<Py<PyTuple>> {
    if length == 0 {
        return Err(PyValueError::new_err("tensor_scan length must be positive"));
    }
    let graph = init.graph.clone();
    let carry_shape = init.shape.clone();
    let mut carry = init;
    let mut outputs = Vec::with_capacity(length);
    let mut output_shape = None;
    for index in 0..length {
        let argument = Py::new(py, carry.clone())?;
        let result = body.call1((index, argument))?;
        let result = result.cast::<PyTuple>().map_err(|_| {
            PyTypeError::new_err("tensor_scan body must return a (carry, output) tuple")
        })?;
        if result.len() != 2 {
            return Err(PyTypeError::new_err(
                "tensor_scan body must return exactly two values",
            ));
        }
        let next = result
            .get_item(0)?
            .extract::<PyRef<'_, TraceTensor>>()
            .map_err(|_| PyTypeError::new_err("tensor_scan carry must be a TraceTensor"))?;
        let output = result
            .get_item(1)?
            .extract::<PyRef<'_, TraceTensor>>()
            .map_err(|_| PyTypeError::new_err("tensor_scan output must be a TraceTensor"))?;
        if !Arc::ptr_eq(&graph.ir, &next.graph.ir) || !Arc::ptr_eq(&graph.ir, &output.graph.ir) {
            return Err(PyValueError::new_err(
                "tensor_scan body returned a TraceTensor from a different graph",
            ));
        }
        if next.shape != carry_shape {
            return Err(PyValueError::new_err(format!(
                "tensor_scan body changed carry shape from {carry_shape:?} to {:?}",
                next.shape
            )));
        }
        if let Some(expected) = &output_shape {
            if &output.shape != expected {
                return Err(PyValueError::new_err(format!(
                    "tensor_scan body changed output shape from {expected:?} to {:?}",
                    output.shape
                )));
            }
        } else {
            output_shape = Some(output.shape.clone());
        }
        carry = next.clone();
        outputs.push(output.clone());
    }
    let stacked = TraceTensor::try_stack(&outputs, 0).map_err(PyValueError::new_err)?;
    Ok(PyTuple::new(py, [carry, stacked])?.unbind())
}

#[pyfunction]
pub fn tensor_grad_scalar_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
) -> PyResult<TensorJitFunction> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let plan = traced
        .graph
        .compile_cpu_plan(traced.output.node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorJitFunction { plan })
}

/// Freezes two CPU branch regions and dispatches only the selected branch.
///
/// `predicate` is supplied as a host boolean when invoking the returned
/// callable. This avoids evaluating an inactive branch, unlike elementwise
/// `where`; nested Tensor IR regions and device-resident predicates remain a
/// later control-flow lowering step.
#[pyfunction]
pub fn tensor_cond_fn(
    py: Python<'_>,
    on_true: &Bound<'_, PyAny>,
    on_false: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
) -> PyResult<TensorCondFunction> {
    let on_true = trace_tensor_python_function(py, on_true, input_specs.clone())?;
    let on_false = trace_tensor_python_function(py, on_false, input_specs)?;
    if on_true.output.shape != on_false.output.shape {
        return Err(PyValueError::new_err(format!(
            "tensor_cond_fn branch output shapes differ: {:?} versus {:?}",
            on_true.output.shape, on_false.output.shape
        )));
    }
    Ok(TensorCondFunction {
        plan: TensorCondExecutionPlan::new(
            on_true
                .graph
                .compile_cpu_plan(on_true.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan,
            on_false
                .graph
                .compile_cpu_plan(on_false.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan,
        )
        .map_err(PyValueError::new_err)?,
    })
}

/// Freezes two scalar-loss CPU branch regions with branch-selected VJP.
#[pyfunction]
pub fn tensor_cond_value_and_grad_fn(
    py: Python<'_>,
    on_true: &Bound<'_, PyAny>,
    on_false: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
) -> PyResult<TensorCondValueAndGradFunction> {
    let on_true = trace_tensor_python_function(py, on_true, input_specs.clone())?;
    let on_false = trace_tensor_python_function(py, on_false, input_specs)?;
    if !on_true.output.shape.is_empty() || !on_false.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "tensor_cond_value_and_grad_fn requires scalar branch outputs, got {:?} and {:?}",
            on_true.output.shape, on_false.output.shape
        )));
    }
    Ok(TensorCondValueAndGradFunction {
        plan: TensorCondExecutionPlan::new(
            on_true
                .graph
                .compile_cpu_plan(on_true.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan,
            on_false
                .graph
                .compile_cpu_plan(on_false.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan,
        )
        .map_err(PyValueError::new_err)?,
    })
}

/// Freezes two CPU branch regions with branch-selected JVP execution.
#[pyfunction]
pub fn tensor_cond_jvp_fn(
    py: Python<'_>,
    on_true: &Bound<'_, PyAny>,
    on_false: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
) -> PyResult<TensorCondJvpFunction> {
    let on_true = trace_tensor_python_function(py, on_true, input_specs.clone())?;
    let on_false = trace_tensor_python_function(py, on_false, input_specs)?;
    if on_true.output.shape != on_false.output.shape {
        return Err(PyValueError::new_err(format!(
            "tensor_cond_jvp_fn branch output shapes differ: {:?} versus {:?}",
            on_true.output.shape, on_false.output.shape
        )));
    }
    Ok(TensorCondJvpFunction {
        plan: TensorCondExecutionPlan::new(
            on_true
                .graph
                .compile_cpu_plan(on_true.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan,
            on_false
                .graph
                .compile_cpu_plan(on_false.output.node_id)
                .map_err(PyValueError::new_err)?
                .plan,
        )
        .map_err(PyValueError::new_err)?,
    })
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

/// Creates a bounded-specialization scalar value-and-gradient callable.
///
/// Each observed mapped batch size receives one frozen CPU plan. Non-batch
/// dimensions and unmapped inputs must remain fixed across calls.
#[pyfunction]
#[pyo3(signature = (function, input_names, in_axes = None, batch_axis = 0, max_specializations = 4))]
pub fn tensor_value_and_grad_batch_fn(
    function: Py<PyAny>,
    input_names: Vec<String>,
    in_axes: Option<Vec<Option<isize>>>,
    batch_axis: isize,
    max_specializations: usize,
) -> PyResult<TensorBatchValueAndGradFunction> {
    if input_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_fn requires at least one input name",
        ));
    }
    if max_specializations == 0 {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_fn max_specializations must be positive",
        ));
    }
    let input_axes = in_axes.unwrap_or_else(|| vec![Some(batch_axis); input_names.len()]);
    if input_axes.len() != input_names.len() {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_batch_fn in_axes has length {}, expected {}",
            input_axes.len(),
            input_names.len()
        )));
    }
    if !input_axes.iter().any(Option::is_some) {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_fn requires at least one mapped input axis",
        ));
    }
    Ok(TensorBatchValueAndGradFunction {
        function,
        input_names,
        input_axes,
        max_specializations,
        static_shapes: RefCell::new(None),
        plans: RefCell::new(BTreeMap::new()),
    })
}

/// Creates a bounded-specialization MLX scalar value-and-gradient callable.
///
/// Every allowed batch size owns one MLX multi-output plan. Results are copied
/// to host only because this diagnostic callable returns the loss and gradients.
#[pyfunction]
#[pyo3(signature = (function, input_names, parameter_names, in_axes = None, batch_axis = 0, max_specializations = 4))]
pub fn tensor_value_and_grad_batch_mlx_fn(
    function: Py<PyAny>,
    input_names: Vec<String>,
    parameter_names: Vec<String>,
    in_axes: Option<Vec<Option<isize>>>,
    batch_axis: isize,
    max_specializations: usize,
) -> PyResult<TensorBatchMlxValueAndGradFunction> {
    if input_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_mlx_fn requires at least one input name",
        ));
    }
    if parameter_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_mlx_fn requires at least one parameter name",
        ));
    }
    if parameter_names
        .iter()
        .any(|name| name == MLX_LOSS_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_batch_mlx_fn reserves parameter name {MLX_LOSS_COTANGENT_NAME:?}"
        )));
    }
    if max_specializations == 0 {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_mlx_fn max_specializations must be positive",
        ));
    }
    let input_axes = in_axes.unwrap_or_else(|| vec![Some(batch_axis); input_names.len()]);
    if input_axes.len() != input_names.len() {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_batch_mlx_fn in_axes has length {}, expected {}",
            input_axes.len(),
            input_names.len()
        )));
    }
    if !input_axes.iter().any(Option::is_some) {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_mlx_fn requires at least one mapped input axis",
        ));
    }
    Ok(TensorBatchMlxValueAndGradFunction {
        function,
        input_names,
        parameter_names,
        input_axes,
        max_specializations,
        static_shapes: RefCell::new(None),
        plans: RefCell::new(BTreeMap::new()),
    })
}

/// Creates a bounded-specialization CUDA JIT callable.
///
/// Each observed mapped batch size owns one CUDA plan. Runtime execution still
/// requires the CUDA driver and NVRTC library on Linux.
#[pyfunction]
#[pyo3(signature = (function, input_names, in_axes = None, batch_axis = 0, max_specializations = 4, device_ordinal = 0))]
pub fn tensor_jit_batch_cuda_fn(
    function: Py<PyAny>,
    input_names: Vec<String>,
    in_axes: Option<Vec<Option<isize>>>,
    batch_axis: isize,
    max_specializations: usize,
    device_ordinal: usize,
) -> PyResult<TensorBatchCudaJitFunction> {
    if input_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_jit_batch_cuda_fn requires at least one input name",
        ));
    }
    if max_specializations == 0 {
        return Err(PyValueError::new_err(
            "tensor_jit_batch_cuda_fn max_specializations must be positive",
        ));
    }
    let input_axes = in_axes.unwrap_or_else(|| vec![Some(batch_axis); input_names.len()]);
    if input_axes.len() != input_names.len() {
        return Err(PyValueError::new_err(format!(
            "tensor_jit_batch_cuda_fn in_axes has length {}, expected {}",
            input_axes.len(),
            input_names.len()
        )));
    }
    if !input_axes.iter().any(Option::is_some) {
        return Err(PyValueError::new_err(
            "tensor_jit_batch_cuda_fn requires at least one mapped input axis",
        ));
    }
    Ok(TensorBatchCudaJitFunction {
        function,
        input_names,
        input_axes,
        max_specializations,
        device_ordinal,
        static_shapes: RefCell::new(None),
        plans: RefCell::new(BTreeMap::new()),
    })
}

/// Creates a bounded-specialization CUDA scalar value-and-gradient callable.
#[pyfunction]
#[pyo3(signature = (function, input_names, parameter_names, in_axes = None, batch_axis = 0, max_specializations = 4, device_ordinal = 0))]
pub fn tensor_value_and_grad_batch_cuda_fn(
    function: Py<PyAny>,
    input_names: Vec<String>,
    parameter_names: Vec<String>,
    in_axes: Option<Vec<Option<isize>>>,
    batch_axis: isize,
    max_specializations: usize,
    device_ordinal: usize,
) -> PyResult<TensorBatchCudaValueAndGradFunction> {
    if input_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_cuda_fn requires at least one input name",
        ));
    }
    if input_names
        .iter()
        .any(|name| name == CUDA_LOSS_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_batch_cuda_fn reserves input name {CUDA_LOSS_COTANGENT_NAME:?}"
        )));
    }
    if parameter_names.is_empty() {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_cuda_fn requires at least one parameter name",
        ));
    }
    if max_specializations == 0 {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_cuda_fn max_specializations must be positive",
        ));
    }
    let input_axes = in_axes.unwrap_or_else(|| vec![Some(batch_axis); input_names.len()]);
    if input_axes.len() != input_names.len() {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_batch_cuda_fn in_axes has length {}, expected {}",
            input_axes.len(),
            input_names.len()
        )));
    }
    if !input_axes.iter().any(Option::is_some) {
        return Err(PyValueError::new_err(
            "tensor_value_and_grad_batch_cuda_fn requires at least one mapped input axis",
        ));
    }
    Ok(TensorBatchCudaValueAndGradFunction {
        function,
        input_names,
        parameter_names,
        input_axes,
        max_specializations,
        device_ordinal,
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

fn vmap_hvp_inputs(
    signature: &VmapSignature,
    values: &Bound<'_, PyDict>,
    cotangent_name: &str,
    tangent_name: &str,
    input_axis: usize,
    input_tangent: &PyTensor,
) -> PyResult<BTreeMap<String, DynamicTensor>> {
    let mut inputs = vmap_inputs(signature, values)?;
    let tangent = move_axis(
        input_tangent
            .to_dynamic_tensor()
            .map_err(PyValueError::new_err)?,
        input_axis,
        0,
    )
    .map_err(PyValueError::new_err)?;
    inputs.insert(
        cotangent_name.to_string(),
        DynamicTensor::filled(vec![], 1.0).map_err(PyValueError::new_err)?,
    );
    inputs.insert(tangent_name.to_string(), tangent);
    Ok(inputs)
}

fn vmap_hvp_output(output: DynamicTensor, input_axis: usize) -> PyResult<PyTensor> {
    let output = move_axis(output, 0, input_axis).map_err(PyValueError::new_err)?;
    PyTensor::from_dynamic_tensor(output).map_err(PyValueError::new_err)
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
    device_ordinal: usize,
) -> PyResult<TensorCudaExecutionPlan> {
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    traced
        .graph
        .compile_cuda_plan(traced.output.node_id, device_ordinal)
        .map_err(PyValueError::new_err)
}

const CUDA_LOSS_COTANGENT_NAME: &str = "__quabla_loss_cotangent";
const MLX_LOSS_COTANGENT_NAME: &str = "__quabla_mlx_loss_cotangent";

fn compile_cuda_scalar_value_and_grad(
    loss: &TensorTraceResult,
    parameter_names: Vec<String>,
    device_ordinal: usize,
) -> PyResult<(
    TensorCudaExecutionPlan,
    TensorNodeId,
    BTreeMap<String, TensorNodeId>,
)> {
    let (program, parameter_names) =
        build_cuda_scalar_value_and_grad_program(loss, parameter_names)?;
    let target = QuablaTarget::Cuda { device_ordinal };
    let executable = QuablaCompiler
        .compile_many_without_build_check(&program, target)
        .map_err(PyValueError::new_err)?;
    let (loss_node_id, gradient_node_ids) =
        value_and_named_outputs(executable.output_node_ids(), parameter_names);
    let plan =
        cuda_execution_plan(executable.into_executable(), target).map_err(PyValueError::new_err)?;
    Ok((plan, loss_node_id, gradient_node_ids))
}

/// Splits the frozen output ids of a program ordered as a value followed by
/// one gradient per name, as the value-and-gradient and vmap VJP helpers
/// build them.
fn value_and_named_outputs(
    frozen_output_node_ids: &[TensorNodeId],
    names: Vec<String>,
) -> (TensorNodeId, BTreeMap<String, TensorNodeId>) {
    (
        frozen_output_node_ids[0],
        names
            .into_iter()
            .zip(frozen_output_node_ids[1..].iter().copied())
            .collect(),
    )
}

/// Returns the loss followed by each requested gradient as one ordered
/// program, together with the requested names in output order.
fn build_cuda_scalar_value_and_grad_program(
    loss: &TensorTraceResult,
    parameter_names: Vec<String>,
) -> PyResult<(QuablaMultiOutputProgram, Vec<String>)> {
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
        ensure_differentiable_trace_input(&loss.graph, &parameter_name)?;
        let gradient_node_id = gradients.get(&parameter_name).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "parameter {parameter_name:?} is not declared in the loss trace"
            ))
        })?;
        requested_names.push(parameter_name);
        output_node_ids.push(gradient_node_id);
    }
    let program = graph
        .into_multi_output_program(output_node_ids)
        .map_err(PyValueError::new_err)?;
    Ok((program, requested_names))
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, parameter_names, device_ordinal = 0))]
pub fn tensor_value_and_grad_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    parameter_names: Vec<String>,
    device_ordinal: usize,
) -> PyResult<TensorCudaValueAndGradFunction> {
    if input_specs
        .iter()
        .any(|spec| spec.name == CUDA_LOSS_COTANGENT_NAME)
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

/// Creates a fixed-shape, single-node CUDA data-parallel value-and-gradient
/// callable. Mapped inputs are split along axis zero; requested parameters must
/// be replicated and are therefore the only gradients all-reduced by NCCL.
#[pyfunction]
#[pyo3(signature = (function, input_specs, parameter_names, mapped_input_names, device_ordinals, reduction = "mean"))]
pub fn tensor_value_and_grad_data_parallel_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    parameter_names: Vec<String>,
    mapped_input_names: Vec<String>,
    device_ordinals: Vec<usize>,
    reduction: &str,
) -> PyResult<TensorCudaDataParallelValueAndGradFunction> {
    if input_specs
        .iter()
        .any(|spec| spec.name == CUDA_LOSS_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_data_parallel_cuda_fn reserves input name {CUDA_LOSS_COTANGENT_NAME:?}"
        )));
    }
    if device_ordinals.len() < 2 {
        return Err(PyValueError::new_err(
            "CUDA data-parallel value-and-grad requires at least two device ordinals",
        ));
    }
    let reduction = match reduction {
        "sum" => TensorReplicaReduction::Sum,
        "mean" => TensorReplicaReduction::Mean,
        _ => {
            return Err(PyValueError::new_err(
                "CUDA data-parallel reduction must be 'sum' or 'mean'",
            ))
        }
    };
    let mapped_input_names = mapped_input_names.into_iter().collect::<BTreeSet<_>>();
    if mapped_input_names.is_empty() {
        return Err(PyValueError::new_err(
            "CUDA data-parallel value-and-grad requires at least one mapped input",
        ));
    }
    if parameter_names
        .iter()
        .any(|name| mapped_input_names.contains(name))
    {
        return Err(PyValueError::new_err(
            "CUDA data-parallel value-and-grad only supports replicated parameter gradients; mapped inputs cannot be requested parameters",
        ));
    }

    let mut input_shapes = BTreeMap::new();
    for TensorInputSpec { name, shape, .. } in &input_specs {
        if input_shapes.insert(name.clone(), shape.clone()).is_some() {
            return Err(PyValueError::new_err(format!(
                "duplicate CUDA data-parallel input name {name:?}"
            )));
        }
    }
    if input_shapes.len() != input_specs.len()
        || mapped_input_names
            .iter()
            .any(|name| !input_shapes.contains_key(name))
    {
        return Err(PyValueError::new_err(format!(
            "CUDA data-parallel mapped inputs must be declared in input_specs: {mapped_input_names:?}"
        )));
    }

    let mut shard_input_specs = input_specs;
    for TensorInputSpec { name, shape, .. } in &mut shard_input_specs {
        if !mapped_input_names.contains(name) {
            continue;
        }
        let batch_extent = shape.first_mut().ok_or_else(|| {
            PyValueError::new_err(format!(
                "CUDA data-parallel mapped input {name:?} must have rank at least one"
            ))
        })?;
        if *batch_extent % device_ordinals.len() != 0 {
            return Err(PyValueError::new_err(format!(
                "CUDA data-parallel mapped input {name:?} batch extent {batch_extent} is not divisible by {} replicas",
                device_ordinals.len()
            )));
        }
        *batch_extent /= device_ordinals.len();
    }

    let traced = trace_tensor_python_function(py, function, shard_input_specs)?;
    let (program, parameter_names) =
        build_cuda_scalar_value_and_grad_program(&traced, parameter_names)?;
    let plan = program.freeze().map_err(PyValueError::new_err)?;
    let (loss_node_id, gradient_node_ids) =
        value_and_named_outputs(plan.output_node_ids(), parameter_names);
    let plan = CudaBackend::new(device_ordinals[0])
        .compile_data_parallel(plan, device_ordinals)
        .map_err(PyValueError::new_err)?;
    Ok(TensorCudaDataParallelValueAndGradFunction {
        plan,
        input_shapes,
        mapped_input_names,
        loss_node_id,
        gradient_node_ids,
        cotangent_name: CUDA_LOSS_COTANGENT_NAME.to_string(),
        reduction,
        last_timing: Arc::new(Mutex::new(None)),
    })
}

/// Copies each mapped element once across all replicas. Replicated tensors
/// still have independent storage per replica; the full input map stays owned
/// by the caller. Mapped shapes and divisibility are checked at compilation.
fn build_data_parallel_host_inputs(
    inputs: &BTreeMap<String, DynamicTensor>,
    input_shapes: &BTreeMap<String, Vec<usize>>,
    mapped_input_names: &BTreeSet<String>,
    replica_count: usize,
    cotangent_name: &str,
) -> Result<Vec<BTreeMap<String, DynamicTensor>>, String> {
    if inputs.len() != input_shapes.len()
        || input_shapes.iter().any(|(name, shape)| {
            inputs
                .get(name)
                .is_none_or(|input| input.shape() != shape.as_slice())
        })
    {
        return Err(format!(
            "CUDA data-parallel value-and-grad expects exactly input shapes {input_shapes:?}"
        ));
    }
    if replica_count == 0 {
        return Err("CUDA data-parallel plan has no compiled replicas".to_string());
    }
    let mut replica_inputs = Vec::with_capacity(replica_count);
    for replica in 0..replica_count {
        let mut shard_inputs = BTreeMap::new();
        for (name, input) in inputs {
            let value = if mapped_input_names.contains(name) {
                let shard_extent = input.shape()[0] / replica_count;
                input.slice_axis(0, replica * shard_extent, shard_extent)?
            } else {
                input.clone()
            };
            shard_inputs.insert(name.clone(), value);
        }
        shard_inputs.insert(
            cotangent_name.to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        );
        replica_inputs.push(shard_inputs);
    }
    Ok(replica_inputs)
}

#[pymethods]
impl TensorCudaDataParallelValueAndGradFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyTensor, BTreeMap<String, PyTensor>)> {
        let inputs = extract_tensor_map(values)?;
        let replica_inputs = build_data_parallel_host_inputs(
            &inputs,
            &self.input_shapes,
            &self.mapped_input_names,
            self.plan.replica_count(),
            &self.cotangent_name,
        )
        .map_err(PyValueError::new_err)?;
        let result = self
            .plan
            .execute_replicas(&replica_inputs, self.reduction)
            .map_err(PyValueError::new_err)?;
        let positions = self
            .plan
            .output_node_ids()
            .iter()
            .enumerate()
            .map(|(position, node_id)| (*node_id, position))
            .collect::<BTreeMap<_, _>>();
        let loss_position = positions.get(&self.loss_node_id).copied().ok_or_else(|| {
            PyValueError::new_err("CUDA data-parallel loss output was not retained")
        })?;
        let loss =
            result.outputs.get(loss_position).cloned().ok_or_else(|| {
                PyValueError::new_err("CUDA data-parallel loss output is missing")
            })?;
        let gradients = self
            .gradient_node_ids
            .iter()
            .map(|(name, node_id)| -> PyResult<(String, PyTensor)> {
                let position = positions.get(node_id).copied().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "CUDA data-parallel gradient output for {name:?} was not retained"
                    ))
                })?;
                let gradient = result.outputs.get(position).cloned().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "CUDA data-parallel gradient output for {name:?} is missing"
                    ))
                })?;
                PyTensor::from_dynamic_tensor(gradient)
                    .map(|tensor| (name.clone(), tensor))
                    .map_err(PyValueError::new_err)
            })
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        *self
            .last_timing
            .lock()
            .map_err(|_| PyValueError::new_err("CUDA data-parallel timing lock is poisoned"))? =
            Some(result.timing);
        Ok((
            PyTensor::from_dynamic_tensor(loss).map_err(PyValueError::new_err)?,
            gradients,
        ))
    }

    #[getter]
    fn replica_count(&self) -> usize {
        self.plan.replica_count()
    }

    #[getter]
    fn device_ordinals(&self) -> Vec<usize> {
        self.plan.device_ordinals()
    }

    #[getter]
    fn last_timing(&self) -> PyResult<Option<BTreeMap<String, f64>>> {
        Ok(self
            .last_timing
            .lock()
            .map_err(|_| PyValueError::new_err("CUDA data-parallel timing lock is poisoned"))?
            .map(|timing| {
                BTreeMap::from([
                    (
                        "replica_enqueue".to_string(),
                        timing.replica_enqueue.as_secs_f64(),
                    ),
                    ("collective".to_string(), timing.collective.as_secs_f64()),
                    (
                        "output_readback".to_string(),
                        timing.output_readback.as_secs_f64(),
                    ),
                ])
            }))
    }
}

fn compile_mlx_scalar_value_and_grad(
    loss: &TensorTraceResult,
    parameter_names: Vec<String>,
) -> PyResult<(
    TensorMlxExecutionPlan,
    TensorNodeId,
    BTreeMap<String, TensorNodeId>,
)> {
    if parameter_names.is_empty() {
        return Err(PyValueError::new_err(
            "MLX value-and-grad requires at least one parameter name",
        ));
    }
    if !loss.output.shape.is_empty() {
        return Err(PyValueError::new_err(format!(
            "MLX value-and-grad requires a scalar output, got shape {:?}",
            loss.output.shape
        )));
    }
    let (graph, loss_node_id, gradients) = loss
        .symbolic_vjp_graph(MLX_LOSS_COTANGENT_NAME)
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
        ensure_differentiable_trace_input(&loss.graph, &parameter_name)?;
        let gradient_node_id = gradients.get(&parameter_name).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "parameter {parameter_name:?} is not declared in the loss trace"
            ))
        })?;
        requested_names.push(parameter_name);
        output_node_ids.push(gradient_node_id);
    }
    let executable = graph
        .compile_multi_output_executable(output_node_ids, QuablaTarget::Mlx)
        .map_err(PyValueError::new_err)?;
    let (loss_node_id, gradient_node_ids) =
        value_and_named_outputs(executable.output_node_ids(), requested_names);
    let plan = mlx_execution_plan(executable.into_executable()).map_err(PyValueError::new_err)?;
    Ok((plan, loss_node_id, gradient_node_ids))
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, parameter_names))]
pub fn tensor_value_and_grad_mlx_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    parameter_names: Vec<String>,
) -> PyResult<TensorMlxValueAndGradFunction> {
    if input_specs
        .iter()
        .any(|spec| spec.name == MLX_LOSS_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_value_and_grad_mlx_fn reserves input name {MLX_LOSS_COTANGENT_NAME:?}"
        )));
    }
    let traced = trace_tensor_python_function(py, function, input_specs)?;
    let (plan, loss_node_id, gradient_node_ids) =
        compile_mlx_scalar_value_and_grad(&traced, parameter_names)?;
    Ok(TensorMlxValueAndGradFunction {
        plan,
        loss_node_id,
        gradient_node_ids,
        cotangent_name: MLX_LOSS_COTANGENT_NAME.to_string(),
    })
}

#[pyfunction]
#[pyo3(signature = (loss, parameter_names, inputs, learning_rate, retained_input_names = None, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8))]
#[allow(clippy::too_many_arguments)]
pub fn mlx_adam_loss_optimizer(
    loss: &Bound<'_, PyAny>,
    parameter_names: Vec<String>,
    inputs: &Bound<'_, PyDict>,
    learning_rate: f32,
    retained_input_names: Option<Vec<String>>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> PyResult<TensorMlxAdamOptimizer> {
    let loss = if let Ok(loss) = loss.extract::<PyRef<'_, TensorTraceResult>>() {
        loss.clone()
    } else if let Ok(loss) = loss.extract::<PyRef<'_, TraceTensor>>() {
        TensorTraceResult::new(loss.graph.clone(), loss.clone())
    } else {
        return Err(PyTypeError::new_err(
            "mlx_adam_loss_optimizer loss must be a TraceTensor or TensorTraceResult",
        ));
    };
    let (plan, loss_node_id, gradient_node_ids) =
        compile_mlx_scalar_value_and_grad(&loss, parameter_names)?;
    let parameter_names = gradient_node_ids.keys().cloned().collect::<BTreeSet<_>>();
    let mut values = extract_tensor_map(inputs)?;
    if values.contains_key(MLX_LOSS_COTANGENT_NAME) {
        return Err(PyValueError::new_err(format!(
            "mlx_adam_loss_optimizer reserves input name {MLX_LOSS_COTANGENT_NAME:?}"
        )));
    }
    values.insert(
        MLX_LOSS_COTANGENT_NAME.to_string(),
        DynamicTensor::new(vec![], vec![1.0]).map_err(PyValueError::new_err)?,
    );
    let mut retained_inputs = retained_input_names
        .unwrap_or_else(|| values.keys().cloned().collect())
        .into_iter()
        .collect::<BTreeSet<_>>();
    retained_inputs.insert(MLX_LOSS_COTANGENT_NAME.to_string());
    retained_inputs.extend(parameter_names.iter().cloned());
    let plan = MlxAdamPlan::new(
        plan.plan,
        loss_node_id,
        gradient_node_ids,
        &values,
        retained_inputs,
        learning_rate,
        beta1,
        beta2,
        epsilon,
    )
    .map_err(PyValueError::new_err)?;
    Ok(TensorMlxAdamOptimizer {
        plan,
        parameter_names,
        inputs: values,
    })
}

/// Trace a fixed-size vectorized function into one reusable CUDA plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0, device_ordinal = 0))]
pub fn tensor_vmap_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
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

/// Trace a fixed-size vectorized VJP into one reusable MLX multi-output plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_vjp_mlx_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapMlxVjpFunction> {
    if input_specs
        .iter()
        .any(|spec| spec.name == MLX_LOSS_COTANGENT_NAME)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_vmap_vjp_mlx_fn reserves input name {MLX_LOSS_COTANGENT_NAME:?}"
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
    let (graph, output_node_id, gradients) = traced
        .symbolic_vjp_graph(MLX_LOSS_COTANGENT_NAME)
        .map_err(PyValueError::new_err)?;
    let mut output_node_ids = vec![output_node_id];
    let mut gradient_names = Vec::with_capacity(signature.input_names.len());
    for name in &signature.input_names {
        // bool inputs are not differentiable: the VJP gradient table excludes them and they are
        // skipped in the result.
        if trace_input_is_bool(&traced.graph, name)? {
            continue;
        }
        let node_id = gradients.get(name).copied().ok_or_else(|| {
            PyValueError::new_err(format!("vmap VJP input {name:?} is absent from the trace"))
        })?;
        gradient_names.push(name.clone());
        output_node_ids.push(node_id);
    }
    let executable = graph
        .compile_multi_output_executable(output_node_ids, QuablaTarget::Mlx)
        .map_err(PyValueError::new_err)?;
    let (output_node_id, gradient_node_ids) =
        value_and_named_outputs(executable.output_node_ids(), gradient_names);
    let plan = mlx_execution_plan(executable.into_executable()).map_err(PyValueError::new_err)?;
    Ok(TensorVmapMlxVjpFunction {
        plan,
        output_node_id,
        gradient_node_ids,
        signature,
        cotangent_name: MLX_LOSS_COTANGENT_NAME.to_string(),
    })
}

const MLX_VMAP_TANGENT_PREFIX: &str = "__quabla_mlx_vmap_tangent_";
const CUDA_VMAP_TANGENT_PREFIX: &str = "__quabla_cuda_vmap_tangent_";

/// Trace a fixed-size vectorized JVP into one reusable MLX multi-output plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_jvp_mlx_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapMlxJvpFunction> {
    let input_names = input_specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<BTreeSet<_>>();
    // bool inputs have no tangent, so no tangent input is created.
    let tangent_names = input_specs
        .iter()
        .filter(|spec| spec.dtype.is_floating())
        .map(|spec| {
            (
                spec.name.clone(),
                format!("{MLX_VMAP_TANGENT_PREFIX}{}", spec.name),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if tangent_names
        .values()
        .any(|tangent_name| input_names.contains(tangent_name))
    {
        return Err(PyValueError::new_err(format!(
            "tensor_vmap_jvp_mlx_fn reserves input names beginning with {MLX_VMAP_TANGENT_PREFIX:?}"
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
    let transformed = traced
        .graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
        .symbolic_jvp_with_tangent_inputs(traced.output.node_id, &tangent_names)
        .map_err(PyValueError::new_err)?;
    let graph = TensorTraceGraph {
        ir: Arc::new(Mutex::new(transformed.graph)),
    };
    let executable = graph
        .compile_multi_output_executable(
            vec![transformed.value, transformed.tangent],
            QuablaTarget::Mlx,
        )
        .map_err(PyValueError::new_err)?;
    let output_node_ids = executable.output_node_ids().to_vec();
    let plan = mlx_execution_plan(executable.into_executable()).map_err(PyValueError::new_err)?;
    Ok(TensorVmapMlxJvpFunction {
        plan,
        value_node_id: output_node_ids[0],
        tangent_node_id: output_node_ids[1],
        signature,
        tangent_names,
    })
}

/// Trace a fixed-size vectorized JVP into one reusable CUDA union plan.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0, device_ordinal = 0))]
pub fn tensor_vmap_jvp_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
    device_ordinal: usize,
) -> PyResult<TensorVmapCudaJvpFunction> {
    let input_names = input_specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<BTreeSet<_>>();
    // bool inputs have no tangent, so no tangent input is created.
    let tangent_names = input_specs
        .iter()
        .filter(|spec| spec.dtype.is_floating())
        .map(|spec| {
            (
                spec.name.clone(),
                format!("{CUDA_VMAP_TANGENT_PREFIX}{}", spec.name),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if tangent_names
        .values()
        .any(|tangent_name| input_names.contains(tangent_name))
    {
        return Err(PyValueError::new_err(format!(
            "tensor_vmap_jvp_cuda_fn reserves input names beginning with {CUDA_VMAP_TANGENT_PREFIX:?}"
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
    let transformed = traced
        .graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
        .symbolic_jvp_with_tangent_inputs(traced.output.node_id, &tangent_names)
        .map_err(PyValueError::new_err)?;
    let graph = TensorTraceGraph {
        ir: Arc::new(Mutex::new(transformed.graph)),
    };
    let target = QuablaTarget::Cuda { device_ordinal };
    let executable = graph
        .compile_multi_output_executable(vec![transformed.value, transformed.tangent], target)
        .map_err(PyValueError::new_err)?;
    let output_node_ids = executable.output_node_ids().to_vec();
    let plan =
        cuda_execution_plan(executable.into_executable(), target).map_err(PyValueError::new_err)?;
    Ok(TensorVmapCudaJvpFunction {
        plan,
        value_node_id: output_node_ids[0],
        tangent_node_id: output_node_ids[1],
        signature,
        tangent_names,
    })
}

const VMAP_HVP_COTANGENT_PREFIX: &str = "__quabla_vmap_hvp_cotangent";
const VMAP_HVP_TANGENT_PREFIX: &str = "__quabla_vmap_hvp_tangent_";

fn build_vmap_hvp_scalar_graph(
    traced: TensorTraceResult,
    signature: &VmapSignature,
    batch_size: usize,
    input_name: &str,
) -> PyResult<(TensorTraceGraph, TensorNodeId, usize, String, String)> {
    if traced.output.shape != vec![batch_size] || traced.output.batch_axis != Some(0) {
        return Err(PyValueError::new_err(format!(
            "tensor_vmap_hvp_scalar_fn requires one scalar output per mapped example, got shape {:?}",
            traced.output.shape
        )));
    }
    let input_index = signature
        .input_names
        .iter()
        .position(|name| name == input_name)
        .ok_or_else(|| {
            PyValueError::new_err(format!("input {input_name:?} is absent from the trace"))
        })?;
    ensure_differentiable_trace_input(&traced.graph, input_name)?;
    let input_axis = signature.in_axes[input_index].ok_or_else(|| {
        PyValueError::new_err(format!(
            "tensor_vmap_hvp_scalar_fn requires input {input_name:?} to be mapped"
        ))
    })?;
    let cotangent_name = VMAP_HVP_COTANGENT_PREFIX.to_string();
    let tangent_name = format!("{VMAP_HVP_TANGENT_PREFIX}{input_name}");
    if signature
        .input_names
        .iter()
        .any(|name| name == &cotangent_name || name == &tangent_name)
    {
        return Err(PyValueError::new_err(format!(
            "tensor_vmap_hvp_scalar_fn reserves input names {cotangent_name:?} and {tangent_name:?}"
        )));
    }

    let mut ir = traced
        .graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
    let scalar_loss = ir
        .sum(traced.output.node_id)
        .map_err(PyValueError::new_err)?;
    let vjp = ir
        .symbolic_vjp(scalar_loss, &cotangent_name)
        .map_err(PyValueError::new_err)?;
    drop(ir);
    let gradient = vjp.gradients.get(input_name).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "vmap HVP input {input_name:?} is absent from the VJP"
        ))
    })?;
    let directional = vjp
        .graph
        .symbolic_jvp_with_tangent_inputs(
            gradient,
            &BTreeMap::from([(input_name.to_string(), tangent_name.clone())]),
        )
        .map_err(PyValueError::new_err)?;
    Ok((
        TensorTraceGraph {
            ir: Arc::new(Mutex::new(directional.graph)),
        },
        directional.tangent,
        input_axis,
        cotangent_name,
        tangent_name,
    ))
}

/// Trace a vectorized per-example scalar loss and compile its HVP with respect
/// to one mapped input. The returned callable accepts the public-layout values
/// and a same-shaped tangent for `input_name`.
#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, input_name, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_hvp_scalar_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    batch_size: usize,
    input_name: String,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
) -> PyResult<TensorVmapHvpScalarFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let (graph, output_node_id, input_axis, cotangent_name, tangent_name) =
        build_vmap_hvp_scalar_graph(traced, &signature, batch_size, &input_name)?;
    let plan = graph
        .compile_cpu_plan(output_node_id)
        .map_err(PyValueError::new_err)?
        .plan;
    Ok(TensorVmapHvpScalarFunction {
        plan,
        signature,
        input_name,
        input_axis,
        cotangent_name,
        tangent_name,
    })
}

/// CUDA variant of [`tensor_vmap_hvp_scalar_fn`]. It emits the existing
/// structural VJP-JVP IR, so Scan and Fori regions remain in the CUDA plan.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (function, input_specs, batch_size, input_name, in_axes = None, out_axis = 0, device_ordinal = 0))]
pub fn tensor_vmap_hvp_scalar_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    batch_size: usize,
    input_name: String,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
    device_ordinal: usize,
) -> PyResult<TensorVmapCudaHvpScalarFunction> {
    let (traced, signature) = trace_tensor_vmap_python_function(
        py,
        function,
        input_specs,
        batch_size,
        in_axes,
        out_axis,
    )?;
    let (graph, output_node_id, input_axis, cotangent_name, tangent_name) =
        build_vmap_hvp_scalar_graph(traced, &signature, batch_size, &input_name)?;
    let plan = graph
        .compile_cuda_plan(output_node_id, device_ordinal)
        .map_err(PyValueError::new_err)?;
    Ok(TensorVmapCudaHvpScalarFunction {
        plan,
        signature,
        input_name,
        input_axis,
        cotangent_name,
        tangent_name,
    })
}

#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0))]
pub fn tensor_vmap_vjp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
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

const CUDA_VMAP_COTANGENT_NAME: &str = "__quabla_vmap_cotangent";

#[pyfunction]
#[pyo3(signature = (function, input_specs, batch_size, in_axes = None, out_axis = 0, device_ordinal = 0))]
pub fn tensor_vmap_vjp_cuda_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<TensorInputSpec>,
    batch_size: usize,
    in_axes: Option<Vec<Option<isize>>>,
    out_axis: isize,
    device_ordinal: usize,
) -> PyResult<TensorVmapCudaVjpFunction> {
    if input_specs
        .iter()
        .any(|spec| spec.name == CUDA_VMAP_COTANGENT_NAME)
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
        // bool inputs are not differentiable: the VJP gradient table excludes them and they are
        // skipped in the result.
        if trace_input_is_bool(&traced.graph, name)? {
            continue;
        }
        let node_id = gradients.get(name).copied().ok_or_else(|| {
            PyValueError::new_err(format!("vmap VJP input {name:?} is absent from the trace"))
        })?;
        gradient_names.push(name.clone());
        output_node_ids.push(node_id);
    }
    let target = QuablaTarget::Cuda { device_ordinal };
    let executable = graph
        .compile_multi_output_executable(output_node_ids, target)
        .map_err(PyValueError::new_err)?;
    let (output_node_id, gradient_node_ids) =
        value_and_named_outputs(executable.output_node_ids(), gradient_names);
    let plan =
        cuda_execution_plan(executable.into_executable(), target).map_err(PyValueError::new_err)?;
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
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
    input_specs: Vec<TensorInputSpec>,
    input_name: String,
) -> PyResult<TensorJacobianFunction> {
    if !input_specs.iter().any(|spec| spec.name == input_name) {
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

    /// Learning rate of the next step; host schedules set it between steps.
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    #[setter]
    fn set_learning_rate(&mut self, learning_rate: f32) -> PyResult<()> {
        if !(learning_rate.is_finite() && learning_rate >= 0.0) {
            return Err(PyValueError::new_err(
                "CUDA Adam learning_rate must be finite and nonnegative",
            ));
        }
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn parameters(&self) -> PyResult<BTreeMap<String, PyTensor>> {
        let frozen = self.frozen_parameters.iter().map(|(name, value)| {
            Ok((
                name.clone(),
                PyTensor::from_dynamic_tensor(value.clone()).map_err(PyValueError::new_err)?,
            ))
        });
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
                .chain(frozen)
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
            .execute_primary_retaining(&self.inputs, &retained_inputs)
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
        frozen_parameters: BTreeMap::new(),
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
    let target = QuablaTarget::Cuda { device_ordinal };
    let executable = graph
        .compile_multi_output_executable(output_node_ids, target)
        .map_err(PyValueError::new_err)?;
    let gradient_node_ids = outputs
        .into_iter()
        .map(|(name, _)| name)
        .zip(executable.output_node_ids().iter().copied())
        .collect::<BTreeMap<_, _>>();
    let plan =
        cuda_execution_plan(executable.into_executable(), target).map_err(PyValueError::new_err)?;
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
        frozen_parameters: BTreeMap::new(),
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
    let (plan, _, mut gradient_node_ids) =
        compile_cuda_scalar_value_and_grad(&loss, parameter_names, device_ordinal)?;
    let mut values = extract_tensor_map(inputs)?;
    let frozen_names = gradient_node_ids
        .keys()
        .filter(|name| plan.plan.plan().input_shape(name).is_err())
        .cloned()
        .collect::<Vec<_>>();
    let mut frozen_parameters = BTreeMap::new();
    for name in frozen_names {
        gradient_node_ids.remove(&name);
        let value = values.get(&name).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "cuda_adam_loss_optimizer parameter {name:?} has no value"
            ))
        })?;
        frozen_parameters.insert(name, value);
    }
    let parameter_names = gradient_node_ids.keys().cloned().collect::<BTreeSet<_>>();
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
        frozen_parameters,
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

pub(crate) fn extract_tensor_map(
    inputs: &Bound<'_, PyDict>,
) -> PyResult<BTreeMap<String, DynamicTensor>> {
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

#[cfg(test)]
mod data_parallel_host_input_tests {
    use super::*;

    #[test]
    fn mapped_leaves_cover_batch_once_and_preserve_dtype() {
        let inputs = BTreeMap::from([
            (
                "x".to_string(),
                DynamicTensor::with_dtype(
                    vec![12, 2],
                    (0..24).map(f64::from).collect(),
                    TensorDType::F32,
                )
                .unwrap(),
            ),
            (
                "y".to_string(),
                DynamicTensor::new(vec![12], (100..112).map(f64::from).collect()).unwrap(),
            ),
            (
                "weights".to_string(),
                DynamicTensor::with_dtype(vec![2], vec![0.5, 1.5], TensorDType::F32).unwrap(),
            ),
            (
                "seed".to_string(),
                DynamicTensor::new(vec![], vec![7.0]).unwrap(),
            ),
        ]);
        let shapes = inputs
            .iter()
            .map(|(name, input)| (name.clone(), input.shape().to_vec()))
            .collect();
        let mapped = BTreeSet::from(["x".to_string(), "y".to_string()]);
        for replica_count in [1, 2, 3, 4, 6, 12] {
            let shards =
                build_data_parallel_host_inputs(&inputs, &shapes, &mapped, replica_count, "seed")
                    .unwrap();
            assert_eq!(shards.len(), replica_count);
            for name in &mapped {
                let original = &inputs[name];
                let reconstructed = shards
                    .iter()
                    .flat_map(|shard| shard[name].data().into_owned().into_iter())
                    .collect::<Vec<_>>();
                assert_eq!(reconstructed, original.data().as_ref());
                for (replica, shard) in shards.iter().enumerate() {
                    let value = &shard[name];
                    assert_eq!(value.dtype(), original.dtype());
                    assert_eq!(value.shape()[0], original.shape()[0] / replica_count);
                    assert_eq!(&value.shape()[1..], &original.shape()[1..]);
                    let extent = original.data().len() / replica_count;
                    assert_eq!(
                        value.data(),
                        &original.data()[replica * extent..(replica + 1) * extent]
                    );
                }
            }
            for shard in &shards {
                assert_eq!(shard.len(), inputs.len());
                assert_eq!(shard["weights"].data(), inputs["weights"].data());
                assert_eq!(shard["weights"].dtype(), TensorDType::F32);
                assert_eq!(shard["weights"].shape(), &[2]);
                assert_ne!(
                    shard["weights"].data().as_ptr(),
                    inputs["weights"].data().as_ptr()
                );
                assert_eq!(shard["seed"].shape(), &[] as &[usize]);
                assert_eq!(shard["seed"].data().as_ref(), &[1.0]);
                assert_eq!(shard["seed"].dtype(), TensorDType::F64);
            }
            assert_eq!(inputs["seed"].data().as_ref(), &[7.0]);
        }
    }

    #[test]
    fn runtime_validation_keeps_exact_shape_errors_and_precedence() {
        let shapes = BTreeMap::from([("x".to_string(), vec![4, 2])]);
        let mapped = BTreeSet::from(["x".to_string()]);
        let expected =
            format!("CUDA data-parallel value-and-grad expects exactly input shapes {shapes:?}");
        for inputs in [
            BTreeMap::new(),
            BTreeMap::from([(
                "x".to_string(),
                DynamicTensor::filled(vec![4], 0.0).unwrap(),
            )]),
            BTreeMap::from([(
                "wrong".to_string(),
                DynamicTensor::filled(vec![4, 2], 0.0).unwrap(),
            )]),
            BTreeMap::from([
                (
                    "x".to_string(),
                    DynamicTensor::filled(vec![4, 2], 0.0).unwrap(),
                ),
                (
                    "extra".to_string(),
                    DynamicTensor::filled(vec![], 0.0).unwrap(),
                ),
            ]),
        ] {
            for replicas in [0, 2] {
                assert_eq!(
                    build_data_parallel_host_inputs(&inputs, &shapes, &mapped, replicas, "seed")
                        .unwrap_err(),
                    expected,
                );
            }
        }
        let valid = BTreeMap::from([(
            "x".to_string(),
            DynamicTensor::filled(vec![4, 2], 0.0).unwrap(),
        )]);
        assert_eq!(
            build_data_parallel_host_inputs(&valid, &shapes, &mapped, 0, "seed").unwrap_err(),
            "CUDA data-parallel plan has no compiled replicas",
        );
    }
}
