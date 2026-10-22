use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFloat, PyInt, PyString, PyTuple};
use quabla_core::tensor_ir::{
    AdamCoefficients, BatchingError, CudaBackend, CudaDataParallelExecutionPlan,
    CudaDataParallelTiming, CudaExecutionPlan, DeviceOptimizerConfig, DeviceUpdateRule,
    DynamicTensor, F64Arith, FusedKernel, MlxAdamPlan, SymbolicCotangent, TensorComparison,
    TensorCondExecutionPlan, TensorCustomRule, TensorCustomTangent, TensorDType,
    TensorExecutionPlan, TensorExtremum, TensorForiExecutionPlan, TensorIr, TensorNodeId,
    TensorRegion, TensorReplicaReduction, TensorScanExecutionPlan, TensorWhileExecutionPlan,
    UnaryMathKind,
};
use quabla_core::{
    QuablaCompiler, QuablaExecutable, QuablaMultiOutputExecutable, QuablaMultiOutputProgram,
    QuablaPrecision, QuablaTarget,
};

use crate::composite::{self, Primitives};
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

#[derive(Clone, Debug)]
pub struct TensorTraceResult {
    pub(crate) graph: TensorTraceGraph,
    pub(crate) output: TraceTensor,
}

#[pyclass(name = "TensorCudaAdamOptimizer", skip_from_py_object)]
pub struct TensorCudaAdamOptimizer {
    shared_plan: SharedCudaAdamPlan,
    parameter_names: BTreeSet<String>,
    // Parameters the loss does not depend on. The compiled plan prunes them,
    // so they have no device buffer; their gradient is zero and Adam leaves
    // them unchanged, so their initial host values are returned as is.
    frozen_parameters: BTreeMap<String, DynamicTensor>,
    inputs: BTreeMap<String, DynamicTensor>,
    retained_inputs: BTreeSet<String>,
    config: DeviceOptimizerConfig,
}

#[pyclass(name = "TensorMlxAdamOptimizer", unsendable, skip_from_py_object)]
pub struct TensorMlxAdamOptimizer {
    plan: MlxAdamPlan,
    parameter_names: BTreeSet<String>,
    inputs: BTreeMap<String, DynamicTensor>,
}

#[derive(Clone, Debug)]
struct SharedCudaAdamPlan {
    plan: CudaTrainingPlan,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
}

/// The value-and-gradient plan of a device trainer in its element type:
/// float32 for a float32 loss, float64 (parameters and optimizer moments in
/// double on the device) for a float64 one.
#[derive(Clone, Debug)]
enum CudaTrainingPlan {
    Float32(CudaExecutionPlan),
    Float64(CudaExecutionPlan<f64>),
}

macro_rules! with_cuda_training_plan {
    ($plan:expr, $inner:ident => $body:expr) => {
        match $plan {
            CudaTrainingPlan::Float32($inner) => $body,
            CudaTrainingPlan::Float64($inner) => $body,
        }
    };
}

impl CudaTrainingPlan {
    fn plan(&self) -> &TensorExecutionPlan {
        with_cuda_training_plan!(self, plan => plan.plan())
    }

    fn step(
        &self,
        gradient_node_ids: &BTreeMap<String, TensorNodeId>,
        values: &BTreeMap<String, DynamicTensor>,
        retained: &BTreeSet<String>,
        config: &DeviceOptimizerConfig,
    ) -> Result<(), String> {
        with_cuda_training_plan!(self, plan => {
            plan.execute_retaining_without_output(values, retained)?;
            plan.optimizer_step(gradient_node_ids, config)
        })
    }

    fn retained_input_to_host(&self, name: &str) -> Result<DynamicTensor, String> {
        with_cuda_training_plan!(self, plan => plan.retained_input_to_host(name))
    }

    fn nonfinite_clip_seen(&self) -> Result<bool, String> {
        with_cuda_training_plan!(self, plan => plan.nonfinite_clip_seen())
    }

    fn device_buffer_count(&self) -> Result<usize, String> {
        with_cuda_training_plan!(self, plan => plan.device_buffer_count())
    }

    fn execute_primary_retaining(
        &self,
        values: &BTreeMap<String, DynamicTensor>,
        retained: &BTreeSet<String>,
    ) -> Result<DynamicTensor, String> {
        with_cuda_training_plan!(self, plan => plan.execute_primary_retaining(values, retained))
    }
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

    /// Compiles one traced output through the core compiler facade.
    ///
    /// Single-output compilation for function-specific helpers and
    /// `Program.compile` funnels through here, so `QuablaCompiler` owns program
    /// construction, freezing, and target selection for every target. The
    /// facade owns an immutable program, while tracing keeps a shared mutable
    /// graph, so each compilation snapshots the traced IR once. It skips the
    /// facade's build-availability check so these entrypoints keep their
    /// pre-facade errors in builds without the target's backend.
    pub(crate) fn compile_executable(
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

    /// Compiles the body of a control-flow region, keeping the custom rule
    /// nodes its derivatives need (see [`TensorIr::compile_region`]).
    pub(crate) fn compile_region_plan(
        &self,
        output_node_id: TensorNodeId,
    ) -> Result<TensorExecutionPlan, String> {
        self.ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
            .compile_region(output_node_id)
    }

    /// Builds an ordered facade program of several traced outputs. The handle
    /// is consumed: a graph built only for this compilation, such as a
    /// symbolic transform result, moves into the program, while a graph still
    /// shared with Python traces is snapshotted. The MLX trainer compiles it
    /// through the checked facade; the CUDA data-parallel helper freezes it
    /// directly because it targets a replica set.
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
        self.inline_into_with(input_names, bindings, outputs, |_, _, spliced| Ok(spliced))
    }

    /// [`Self::inline_into`], with `finish(ir, bound, spliced)` mapping the
    /// spliced output ids before they become tracers, where `bound[i]` is
    /// the node bound to `input_names[i]` (a custom rule call wraps the
    /// outputs in `Custom` nodes over those operands).
    fn inline_into_with(
        &self,
        input_names: &[String],
        bindings: &[InlineBinding],
        outputs: &[TraceTensor],
        finish: impl FnOnce(
            &mut TensorIr,
            &[TensorNodeId],
            Vec<TensorNodeId>,
        ) -> Result<Vec<TensorNodeId>, String>,
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
        let mut bound_in_order = Vec::with_capacity(bindings.len());
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
            bound_in_order.push(node_id);
        }
        let output_ids = outputs
            .iter()
            .map(|output| output.node_id)
            .collect::<Vec<_>>();
        let spliced = ir.inline(&callee, &bound, &output_ids)?;
        let spliced = finish(&mut ir, &bound_in_order, spliced)?;
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

/// A frozen custom differentiation rule (`quabla.custom_vjp`,
/// `quabla.custom_jvp`, `quabla.checkpoint`), built by
/// `TensorTraceGraph._custom_rule`. Like `StagedExecutable`, the class is an
/// implementation detail of `quabla._transforms` and is not registered.
#[pyclass(name = "CustomRule", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCustomRule {
    rule: Arc<TensorCustomRule>,
}

#[pymethods]
impl PyCustomRule {
    fn __repr__(&self) -> String {
        format!("CustomRule({})", self.rule.name())
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

    pub(crate) fn binary(&self, rhs: &Self, op: &str) -> Result<Self, String> {
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
            "fmod" => ir.fmod(self.node_id, rhs.node_id)?,
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

    pub(crate) fn scalar_binary(&self, value: f64, op: &str) -> Result<Self, String> {
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
            "fmod" => ir.fmod(self.node_id, scalar)?,
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

    /// Evaluates the traced op `build` on eager operands at once: each
    /// operand becomes a constant of a fresh graph, `build` appends the nodes
    /// a trace of the op records, and the CPU evaluator runs them node by
    /// node, which a `jit` plan, fused elementwise regions included, matches
    /// bit for bit. Eager results therefore equal traced ones, and errors are
    /// the builder's. The result keeps the weak type of its node.
    pub(crate) fn evaluate_eager<const N: usize>(
        operands: [&PyTensor; N],
        build: impl FnOnce([Self; N]) -> Result<Self, String>,
    ) -> Result<PyTensor, String> {
        Self::evaluate_eager_all(&operands, |tracers| {
            build(
                tracers
                    .try_into()
                    .map_err(|_| "eager operand count mismatch".to_string())?,
            )
        })
    }

    /// [`Self::evaluate_eager`] for any number of operands.
    pub(crate) fn evaluate_eager_all(
        operands: &[&PyTensor],
        build: impl FnOnce(Vec<Self>) -> Result<Self, String>,
    ) -> Result<PyTensor, String> {
        let graph = TensorTraceGraph::new();
        let tracers = operands
            .iter()
            .map(|operand| {
                let value = operand.to_dynamic_tensor()?;
                let shape = value.shape().to_vec();
                let node_id = graph
                    .ir
                    .lock()
                    .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
                    .constant(value, operand.is_weak());
                Ok(Self::from_node(graph.clone(), node_id, shape, None))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let output = build(tracers)?;
        let ir = graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let value = ir.evaluate(output.node_id, &BTreeMap::new())?;
        Ok(PyTensor::from_dynamic_tensor(value)?.with_weak(ir.node_weak(output.node_id)?))
    }

    /// [`Self::evaluate_eager_all`] for an op composed of elementwise
    /// primitives, which also returns its graph compiled to one fused kernel
    /// over the operands (`None` when the CPU evaluator would not run the
    /// graph as one fused region) and the weak type of the result. The
    /// kernel computes the graph's value bit for bit, so a later call with
    /// operands of the same shapes, dtypes, and weak types can run it
    /// without building the graph.
    pub(crate) fn evaluate_eager_fused(
        operands: &[&PyTensor],
        build: impl FnOnce(Vec<Self>) -> Result<Self, String>,
    ) -> Result<(PyTensor, Option<FusedKernel>, bool), String> {
        let graph = TensorTraceGraph::new();
        let tracers = operands
            .iter()
            .map(|operand| {
                let value = operand.to_dynamic_tensor()?;
                let shape = value.shape().to_vec();
                let node_id = graph
                    .ir
                    .lock()
                    .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
                    .constant(value, operand.is_weak());
                Ok(Self::from_node(graph.clone(), node_id, shape, None))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let operand_nodes = tracers
            .iter()
            .map(|tracer| tracer.node_id)
            .collect::<Vec<_>>();
        let output = build(tracers)?;
        let ir = graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let weak = ir.node_weak(output.node_id)?;
        let kernel = ir.fused_kernel(output.node_id, &operand_nodes);
        let fused = match &kernel {
            Some(kernel) => kernel.evaluate(
                &operands
                    .iter()
                    .map(|operand| operand.operand())
                    .collect::<Vec<_>>(),
            )?,
            None => None,
        };
        let value = match fused {
            Some(value) => value,
            None => ir.evaluate(output.node_id, &BTreeMap::new())?,
        };
        Ok((
            PyTensor::from_dynamic_tensor(value)?.with_weak(weak),
            kernel,
            weak,
        ))
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

    /// `-x` as `x * -1`, so `-0.0` stays signed; `bool` is rejected.
    pub(crate) fn negative_tensor(&self) -> Result<Self, String> {
        self.ensure_not_bool("negative")?;
        self.scalar_binary(-1.0, "mul")
    }

    /// [`Self::scalar_binary`] for an op without a `bool` form (`atan2`,
    /// `fmod`): a `bool` operand would otherwise join the weak scalar as
    /// 0/1 values.
    pub(crate) fn float_scalar_binary(&self, value: f64, op: &str) -> Result<Self, String> {
        self.ensure_not_bool(op)?;
        self.scalar_binary(value, op)
    }

    pub(crate) fn scalar_left_binary(&self, value: f64, op: &str) -> Result<Self, String> {
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

    pub(crate) fn sum_tensor(&self, axis: Option<isize>) -> Result<Self, String> {
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

    pub(crate) fn matmul_tensor(&self, rhs: &Self) -> Result<Self, String> {
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

    pub(crate) fn solve_tensor(&self, rhs: &Self) -> Result<Self, String> {
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].solve_tensor(&lifted[1]);
        }
        let batch_axis = Self::merged_batch_axis(&[self, rhs])?;
        let mut ir = self
            .graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        // The batch axis leads; `Solve` batches over equal leading axes, so an
        // unbatched operand is broadcast to the batch.
        let mut operands = [
            (self.node_id, self.batch_axis),
            (rhs.node_id, rhs.batch_axis),
        ];
        if let Some(axis) = batch_axis {
            let size = if self.batch_axis.is_some() {
                self.shape[axis]
            } else {
                rhs.shape[axis]
            };
            for (node_id, operand_axis) in &mut operands {
                if operand_axis.is_none() {
                    let mut shape = vec![size];
                    shape.extend(ir.node_shape(*node_id)?);
                    *node_id = ir.broadcast_to(*node_id, shape)?;
                }
            }
        }
        let node_id = ir.solve(operands[0].0, operands[1].0)?;
        let shape = ir.node_shape(node_id)?;
        Ok(Self::from_node(
            self.graph.clone(),
            node_id,
            shape,
            batch_axis,
        ))
    }

    pub fn try_solve(&self, rhs: &Self) -> Result<Self, String> {
        self.solve_tensor(rhs)
    }

    pub(crate) fn solve_triangular_tensor(
        &self,
        rhs: &Self,
        lower: bool,
        transpose: bool,
    ) -> Result<Self, String> {
        let matrix = self.triangular_tensor(lower)?;
        let matrix = if transpose {
            // Swap only the matrix axes; leading batch axes stay in place.
            let rank = matrix.shape.len() - usize::from(matrix.batch_axis.is_some());
            let mut axes = (0..rank as isize).collect::<Vec<_>>();
            axes.swap(rank - 2, rank - 1);
            matrix.transpose_tensor(Some(axes))?
        } else {
            matrix
        };
        matrix.solve_tensor(rhs)
    }

    pub(crate) fn cholesky_tensor(&self) -> Result<Self, String> {
        // The IR factors every matrix of the leading axes, including a
        // leading vmap batch axis.
        let rank = self.shape.len();
        if rank < 2 || self.shape[rank - 2] != self.shape[rank - 1] {
            return Err(format!(
                "cholesky requires a stack of square matrices [..., n, n], got {:?}",
                self.shape
            ));
        }
        self.apply(&[], |ir| ir.cholesky(self.node_id))
    }

    pub(crate) fn linalg_tensor(&self, kind: &str) -> Result<Self, String> {
        let kind = quabla_core::tensor_ir::LinalgKind::from_name(kind)?;
        self.apply(&[], |ir| ir.linalg(self.node_id, kind))
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

    pub(crate) fn compare_tensor(
        &self,
        rhs: &Self,
        kind: TensorComparison,
    ) -> Result<Self, String> {
        if let Some(lifted) = Self::lifted_to_common_graph(&[self, rhs])? {
            return lifted[0].compare_tensor(&lifted[1], kind);
        }
        self.apply(&[rhs], |ir| ir.compare(self.node_id, rhs.node_id, kind))
    }

    pub(crate) fn compare_scalar(
        &self,
        value: f64,
        kind: TensorComparison,
    ) -> Result<Self, String> {
        self.apply(&[], |ir| {
            let scalar = ir.scalar_constant(value);
            ir.compare(self.node_id, scalar, kind)
        })
    }

    pub(crate) fn logical_tensor(&self, rhs: &Self, and: bool) -> Result<Self, String> {
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

    pub(crate) fn logical_not_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.logical_not(self.node_id))
    }

    pub(crate) fn classify_tensor(&self, nan: bool) -> Result<Self, String> {
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
    pub(crate) fn any_all_tensor(
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

    pub(crate) fn maximum_tensor(&self, rhs: &Self) -> Result<Self, String> {
        composite::maximum(self, rhs)
    }

    pub(crate) fn maximum_scalar(&self, rhs: f64) -> Result<Self, String> {
        composite::maximum_scalar(self, rhs)
    }

    pub(crate) fn minimum_tensor(&self, rhs: &Self) -> Result<Self, String> {
        composite::minimum(self, rhs)
    }

    pub(crate) fn minimum_scalar(&self, rhs: f64) -> Result<Self, String> {
        composite::minimum_scalar(self, rhs)
    }

    pub(crate) fn tanh_tensor(&self) -> Result<Self, String> {
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

    pub(crate) fn relu_tensor(&self) -> Result<Self, String> {
        composite::relu(self)
    }

    pub(crate) fn abs_tensor(&self) -> Result<Self, String> {
        composite::abs(self)
    }

    pub(crate) fn sigmoid_tensor(&self) -> Result<Self, String> {
        composite::sigmoid(self)
    }

    pub(crate) fn softplus_tensor(&self) -> Result<Self, String> {
        composite::softplus(self)
    }

    pub(crate) fn triangular_tensor(&self, lower: bool) -> Result<Self, String> {
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

    pub(crate) fn exp_tensor(&self) -> Result<Self, String> {
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

    pub(crate) fn reshape_tensor(&self, shape: Vec<usize>) -> Result<Self, String> {
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

    pub(crate) fn gather_tensor(&self, indices: &[usize], axis: isize) -> Result<Self, String> {
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

    pub(crate) fn scatter_add_tensor(
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

    pub(crate) fn broadcast_to_tensor(&self, shape: Vec<usize>) -> Result<Self, String> {
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

    pub(crate) fn reduce_axes_tensor(
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

    /// The product over the example `axes` (every example axis when `None`)
    /// as the pairwise `mul` tree of `TensorIr::prod_axis`.
    pub(crate) fn prod_axes_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
    ) -> Result<Self, String> {
        self.ensure_not_bool("prod")?;
        let batch_offset = usize::from(self.batch_axis.is_some());
        let example_rank = self.shape.len() - batch_offset;
        let product_along = |source: &Self, axis: isize| {
            let axis = source.example_axis(axis)?;
            source.apply(&[], |ir| ir.prod_axis(source.node_id, axis))
        };
        let Some(axes) = axes else {
            let flat = self.reshape_tensor(vec![self.shape[batch_offset..].iter().product()])?;
            let product = product_along(&flat, 0)?;
            return if keepdims {
                product.reshape_tensor(vec![1; example_rank])
            } else {
                Ok(product)
            };
        };
        let mut axes = normalize_reduction_axes(axes, example_rank)?;
        axes.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        let mut reduced = self.clone();
        for axis in axes {
            reduced = product_along(&reduced, axis as isize)?;
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
    pub(crate) fn norm_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
    ) -> Result<Self, String> {
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

    /// The max or min over the example `axes` (every example axis when
    /// `None`) as one `ExtremumAxis` node: the reduced axes are moved last
    /// and flattened into one, so the graph has O(1) nodes in the reduced
    /// size and ties across all reduced axes share the derivative equally,
    /// as in JAX. An empty `axes` returns `self`.
    pub(crate) fn extrema_axes_tensor(
        &self,
        axes: Option<Vec<isize>>,
        keepdims: bool,
        maximum: bool,
    ) -> Result<Self, String> {
        self.ensure_not_bool(if maximum { "max" } else { "min" })?;
        let batch_offset = usize::from(self.batch_axis.is_some());
        let example_shape = self.shape[batch_offset..].to_vec();
        let axes = match axes {
            Some(axes) => normalize_reduction_axes(axes, example_shape.len())?,
            None => (0..example_shape.len()).collect(),
        };
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let kept = (0..example_shape.len())
            .filter(|axis| !axes.contains(axis))
            .collect::<Vec<_>>();
        let order = kept.iter().chain(&axes).copied().collect::<Vec<_>>();
        let mut source = self.clone();
        if order
            .iter()
            .enumerate()
            .any(|(position, axis)| position != *axis)
        {
            source =
                source.transpose_tensor(Some(order.iter().map(|axis| *axis as isize).collect()))?;
        }
        if axes.len() > 1 {
            let mut flat_shape = kept
                .iter()
                .map(|axis| example_shape[*axis])
                .collect::<Vec<_>>();
            flat_shape.push(axes.iter().map(|axis| example_shape[*axis]).product());
            source = source.reshape_tensor(flat_shape)?;
        }
        let kind = if maximum {
            TensorExtremum::Max
        } else {
            TensorExtremum::Min
        };
        let reduced = source.apply(&[], |ir| ir.extremum_axis(source.node_id, -1, kind))?;
        if !keepdims {
            return Ok(reduced);
        }
        let mut shape = example_shape;
        for axis in axes {
            shape[axis] = 1;
        }
        reduced.reshape_tensor(shape)
    }

    pub(crate) fn sin_tensor(&self) -> Result<Self, String> {
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

    pub(crate) fn cos_tensor(&self) -> Result<Self, String> {
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

    pub(crate) fn powi_tensor(&self, exponent: u32) -> Result<Self, String> {
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

    pub(crate) fn transpose_tensor(&self, axes: Option<Vec<isize>>) -> Result<Self, String> {
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

    pub(crate) fn log_tensor(&self) -> Result<Self, String> {
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

    pub(crate) fn log1p_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.log1p(self.node_id))
    }

    pub(crate) fn expm1_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.expm1(self.node_id))
    }

    pub(crate) fn erf_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.erf(self.node_id))
    }

    pub(crate) fn erfc_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.erfc(self.node_id))
    }

    pub(crate) fn unary_math_tensor(&self, kind: UnaryMathKind) -> Result<Self, String> {
        self.apply(&[], |ir| ir.unary_math(self.node_id, kind))
    }

    fn unary_math(&self, kind: UnaryMathKind) -> PyResult<Self> {
        self.unary_math_tensor(kind).map_err(PyValueError::new_err)
    }

    fn stop_gradient_tensor(&self) -> Result<Self, String> {
        self.apply(&[], |ir| ir.stop_gradient(self.node_id))
    }

    /// Inclusive prefix sums along the example `axis`, or over the flattened
    /// example when `axis` is `None` (NumPy's convention).
    pub(crate) fn cumsum_tensor(&self, axis: Option<isize>, reverse: bool) -> Result<Self, String> {
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

    pub(crate) fn sqrt_tensor(&self) -> Result<Self, String> {
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

/// The primitives of the composites as graph nodes.
impl Primitives for TraceTensor {
    fn dtype(&self) -> Result<TensorDType, String> {
        TraceTensor::dtype(self)
    }

    fn binary(&self, rhs: &Self, op: &'static str) -> Result<Self, String> {
        TraceTensor::binary(self, rhs, op)
    }

    fn scalar_binary(&self, value: f64, op: &'static str) -> Result<Self, String> {
        TraceTensor::scalar_binary(self, value, op)
    }

    fn scalar_left_binary(&self, value: f64, op: &'static str) -> Result<Self, String> {
        TraceTensor::scalar_left_binary(self, value, op)
    }

    fn compare(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String> {
        self.compare_tensor(rhs, kind)
    }

    fn compare_scalar(&self, value: f64, kind: TensorComparison) -> Result<Self, String> {
        TraceTensor::compare_scalar(self, value, kind)
    }

    fn isnan(&self) -> Result<Self, String> {
        self.classify_tensor(true)
    }

    fn logical_or(&self, rhs: &Self) -> Result<Self, String> {
        self.logical_tensor(rhs, false)
    }

    fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        self.where_tensor(on_true, on_false)
    }

    fn scalar(&self, value: f64) -> Result<Self, String> {
        self.scalar_tensor(value)
    }

    fn exp(&self) -> Result<Self, String> {
        self.exp_tensor()
    }

    fn log1p(&self) -> Result<Self, String> {
        self.log1p_tensor()
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
    /// Every node runs in its own dtype; `precision="float64"` only states
    /// that request (see `QuablaPrecision`) and is rejected on MLX.
    #[pyo3(
        name = "_compile",
        signature = (outputs, input_names, target = "cpu", device_ordinal = 0, precision = None)
    )]
    fn py_compile_target(
        &self,
        py: Python<'_>,
        outputs: Vec<TraceTensor>,
        input_names: Vec<String>,
        target: &str,
        device_ordinal: usize,
        precision: Option<&str>,
    ) -> PyResult<StagedExecutable> {
        self.ensure_owns(&outputs.iter().collect::<Vec<_>>())
            .map_err(PyValueError::new_err)?;
        let target = match target {
            "cpu" => QuablaTarget::Cpu,
            "cuda" => QuablaTarget::Cuda { device_ordinal },
            "mlx" => QuablaTarget::Mlx,
            _ => return Err(PyValueError::new_err("unknown compilation target")),
        };
        let precision = match precision {
            None => QuablaPrecision::Default,
            Some("float64") => QuablaPrecision::Float64,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "precision must be None or \"float64\", got {other:?}"
                )))
            }
        };
        let program = self
            .clone()
            .into_multi_output_program(outputs.iter().map(|output| output.node_id).collect())
            .map_err(PyValueError::new_err)?;
        Ok(StagedExecutable {
            executable: QuablaCompiler
                .compile_many_checked_with_precision(&program, target, precision)
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

    /// A call of a function with a custom differentiation rule: inlines
    /// `outputs` of this staged primal graph like `_inline`, then wraps each
    /// spliced output in a `Custom` node of `rule` whose operands are the
    /// bindings, in `input_names` order (the rule's operand order).
    #[pyo3(name = "_custom_call")]
    fn py_custom_call(
        &self,
        py: Python<'_>,
        rule: PyRef<'_, PyCustomRule>,
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
                    Ok(InlineBinding::Constant(
                        tensor.to_dynamic_tensor().map_err(PyValueError::new_err)?,
                    ))
                } else {
                    Err(PyTypeError::new_err(format!(
                        "a custom rule operand must be a TraceTensor or a Tensor, got {}",
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
        let rule = rule.rule.clone();
        self.inline_into_with(&input_names, &bindings, &outputs, |ir, bound, spliced| {
            ir.custom(rule, &spliced, bound)
        })
        .map_err(PyValueError::new_err)
    }

    /// Freezes a custom differentiation rule from staged graphs (see
    /// `TensorCustomRule`): `forward_outputs` are tracers of `forward`,
    /// `backward_outputs` (one per operand, `None` for no cotangent) of
    /// `backward`, and `tangent_outputs` of `tangent`, if given, whose
    /// inputs `output_names` (one per output, or none) read the call's
    /// outputs. `prefer_reverse` (default: no tangent graph) asks `jacobian`
    /// and `hessian` for reverse mode.
    #[staticmethod]
    #[pyo3(name = "_custom_rule")]
    #[pyo3(signature = (
        name,
        forward,
        operand_names,
        output_count,
        forward_outputs,
        backward,
        residual_names,
        cotangent_names,
        backward_outputs,
        tangent,
        tangent_names,
        tangent_outputs,
        rematerialize,
        *,
        output_names = Vec::new(),
        prefer_reverse = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_custom_rule(
        name: String,
        forward: PyRef<'_, TensorTraceGraph>,
        operand_names: Vec<String>,
        output_count: usize,
        forward_outputs: Vec<TraceTensor>,
        backward: PyRef<'_, TensorTraceGraph>,
        residual_names: Vec<String>,
        cotangent_names: Vec<String>,
        backward_outputs: Vec<Option<TraceTensor>>,
        tangent: Option<PyRef<'_, TensorTraceGraph>>,
        tangent_names: Vec<Option<String>>,
        tangent_outputs: Vec<TraceTensor>,
        rematerialize: bool,
        output_names: Vec<String>,
        prefer_reverse: Option<bool>,
    ) -> PyResult<PyCustomRule> {
        let snapshot = |graph: &TensorTraceGraph, tensors: &[&TraceTensor]| {
            graph.ensure_owns(tensors)?;
            Ok::<_, String>(
                graph
                    .ir
                    .lock()
                    .map_err(|_| "tensor trace graph lock is poisoned".to_string())?
                    .clone(),
            )
        };
        let rule = (|| {
            let forward_ir = snapshot(&forward, &forward_outputs.iter().collect::<Vec<_>>())?;
            let backward_ir = snapshot(
                &backward,
                &backward_outputs.iter().flatten().collect::<Vec<_>>(),
            )?;
            let tangent = match tangent {
                Some(graph) => Some(
                    TensorCustomTangent::new(
                        snapshot(&graph, &tangent_outputs.iter().collect::<Vec<_>>())?,
                        tangent_names,
                        tangent_outputs
                            .iter()
                            .map(|output| output.node_id)
                            .collect(),
                    )
                    .with_output_inputs(output_names),
                ),
                None => None,
            };
            let rule = TensorCustomRule::new(
                name,
                forward_ir,
                operand_names,
                output_count,
                forward_outputs
                    .iter()
                    .map(|output| output.node_id)
                    .collect(),
                backward_ir,
                residual_names,
                cotangent_names,
                backward_outputs
                    .iter()
                    .map(|output| output.as_ref().map(|output| output.node_id))
                    .collect(),
                tangent,
                rematerialize,
            )?;
            Ok::<_, String>(match prefer_reverse {
                Some(prefer_reverse) => rule.with_prefer_reverse(prefer_reverse),
                None => rule,
            })
        })()
        .map_err(PyValueError::new_err)?;
        Ok(PyCustomRule {
            rule: Arc::new(rule),
        })
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

    /// Whether forward mode through the graph fails on a `custom_vjp` call.
    #[getter(_has_reverse_only_custom_rule)]
    fn py_has_reverse_only_custom_rule(&self) -> PyResult<bool> {
        Ok(self
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .has_reverse_only_custom_rule())
    }

    /// Whether `jacobian` and `hessian` use reverse mode for the graph
    /// because it calls a `custom_vjp` function (see
    /// `TensorIr::prefers_reverse_mode`).
    #[getter(_prefers_reverse_mode)]
    fn py_prefers_reverse_mode(&self) -> PyResult<bool> {
        Ok(self
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?
            .prefers_reverse_mode())
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

    fn __neg__(&self) -> PyResult<Self> {
        self.negative_tensor().map_err(PyValueError::new_err)
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

    #[pyo3(signature = (axis = None, keepdims = false))]
    fn prod(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<Self> {
        self.prod_axes_tensor(extract_reduction_axes(axis)?, keepdims)
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

    /// One output of a dense decomposition of each matrix (`kind` is a
    /// `quabla_core` `LinalgKind` name); the backing op of `quabla.linalg`.
    fn _linalg(&self, kind: &str) -> PyResult<Self> {
        self.linalg_tensor(kind).map_err(PyValueError::new_err)
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

    fn erfc(&self) -> PyResult<Self> {
        self.erfc_tensor().map_err(PyValueError::new_err)
    }

    /// `atan2(self, x)` of an array or Python number `x`.
    fn atan2(&self, x: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(x) = self.traced_operand(x) {
            return self.binary(&x?, "atan2").map_err(PyValueError::new_err);
        }
        if let Some(x) = extract_scalar(x) {
            return self
                .float_scalar_binary(x, "atan2")
                .map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected a TraceTensor, Tensor, or numeric scalar operand",
        ))
    }

    /// C `fmod(self, y)` of an array or Python number `y`: the remainder of
    /// the quotient truncated toward zero, with the sign of `self`.
    fn fmod(&self, y: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(y) = self.traced_operand(y) {
            return self.binary(&y?, "fmod").map_err(PyValueError::new_err);
        }
        if let Some(y) = extract_scalar(y) {
            return self
                .float_scalar_binary(y, "fmod")
                .map_err(PyValueError::new_err);
        }
        Err(PyTypeError::new_err(
            "expected a TraceTensor, Tensor, or numeric scalar operand",
        ))
    }

    fn tan(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Tan)
    }

    fn arcsin(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arcsin)
    }

    fn arccos(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arccos)
    }

    fn arctan(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arctan)
    }

    fn sinh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Sinh)
    }

    fn cosh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Cosh)
    }

    fn arcsinh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arcsinh)
    }

    fn arccosh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arccosh)
    }

    fn arctanh(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Arctanh)
    }

    fn log2(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Log2)
    }

    fn log10(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Log10)
    }

    fn cbrt(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Cbrt)
    }

    fn floor(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Floor)
    }

    fn ceil(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Ceil)
    }

    fn round(&self) -> PyResult<Self> {
        self.unary_math(UnaryMathKind::Round)
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

    /// Shape and dtype only: a traced value has no data to show.
    fn __repr__(&self) -> String {
        match self.dtype() {
            Ok(dtype) => format!(
                "TraceTensor(node_id={}, shape={:?}, dtype={})",
                self.node_id,
                self.shape,
                PyDType::from(dtype).name()
            ),
            // A poisoned graph lock must not make `repr` raise.
            Err(_) => format!(
                "TraceTensor(node_id={}, shape={:?})",
                self.node_id, self.shape
            ),
        }
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
    pub(crate) fn apply(self, lhs: &TraceTensor, rhs: &TraceTensor) -> Result<TraceTensor, String> {
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
        traced.graph.compile_region_plan(traced.output.node_id)?,
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
    traced.graph.compile_region_plan(output)
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
        compile_loop_body(&body_graph, output.node_id, "__quabla_fori_carry")
            .map_err(PyValueError::new_err)?,
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

/// Compiles a single-output loop body whose output is the next carry.
///
/// A body whose result does not read the carry is valid, as in JAX, but a
/// compiled region drops the inputs it never reads, and every loop plan binds
/// its carry by name. Such a body is recompiled as `where(false, carry,
/// next)`: the select returns `next` bit for bit (NaN payloads and signed
/// zeros included), never reads the carry's values, and gives the carry a
/// zero derivative, while keeping the carry an input of the region.
fn compile_loop_body(
    graph: &TensorTraceGraph,
    next: TensorNodeId,
    carry_name: &str,
) -> Result<TensorExecutionPlan, String> {
    let plan = graph.compile_region_plan(next)?;
    if region_input_names(&plan)
        .iter()
        .any(|name| name == carry_name)
    {
        return Ok(plan);
    }
    let retained = {
        let mut ir = graph
            .ir
            .lock()
            .map_err(|_| "tensor trace graph lock is poisoned".to_string())?;
        let carry = ir.input_node_id(carry_name)?;
        let never = ir.scalar_constant(0.0);
        ir.where_select(never, carry, next)?
    };
    graph.compile_region_plan(retained)
}

/// Traces one `while_loop` region over the carry and operand inputs.
fn trace_while_region(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    init: &TraceTensor,
    operands: &[TraceTensor],
    label: &str,
) -> PyResult<(TensorTraceGraph, TraceTensor)> {
    let graph = TensorTraceGraph::new();
    let mut arguments = vec![graph
        .add_input(
            "__quabla_while_carry",
            init.shape.clone(),
            init.dtype().map_err(PyValueError::new_err)?,
        )
        .map_err(PyValueError::new_err)?];
    for (index, operand) in operands.iter().enumerate() {
        arguments.push(
            graph
                .add_input(
                    &format!("__quabla_while_capture_{index}"),
                    operand.shape.clone(),
                    operand.dtype().map_err(PyValueError::new_err)?,
                )
                .map_err(PyValueError::new_err)?,
        );
    }
    let output: TraceTensor = function
        .call1(PyTuple::new(py, arguments)?)?
        .extract()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "tensor_while_loop_region {label} must return a TraceTensor"
            ))
        })?;
    if !Arc::ptr_eq(&graph.ir, &output.graph.ir) {
        return Err(PyValueError::new_err(format!(
            "tensor_while_loop_region {label} returned a TraceTensor from a different graph"
        )));
    }
    Ok((graph, output))
}

/// Traces a data-dependent loop into one runtime Tensor IR region.
///
/// `cond` and `body` both receive `(carry, *operands)`; `cond` returns a
/// scalar bool and `body` the next carry. All external values must be
/// provided as explicit operands; either region may ignore some of them.
#[pyfunction]
pub fn tensor_while_loop_region(
    py: Python<'_>,
    cond: &Bound<'_, PyAny>,
    body: &Bound<'_, PyAny>,
    init: TraceTensor,
    operands: Vec<TraceTensor>,
) -> PyResult<TraceTensor> {
    if init.batch_axis.is_some() || operands.iter().any(|operand| operand.batch_axis.is_some()) {
        return Err(PyValueError::new_err(
            "the tensor_vmap_* helpers cannot batch a while_loop; quabla.vmap batches it, \
             stopping each batch element at its own trip count",
        ));
    }
    for operand in &operands {
        init.same_graph(operand).map_err(PyValueError::new_err)?;
    }
    let (predicate_graph, predicate) = trace_while_region(py, cond, &init, &operands, "cond_fun")?;
    if !predicate.shape.is_empty()
        || predicate.dtype().map_err(PyValueError::new_err)? != TensorDType::Bool
    {
        return Err(PyTypeError::new_err(format!(
            "while_loop cond_fun must return a scalar bool, got shape {:?} and dtype {}",
            predicate.shape,
            predicate.dtype().map_err(PyValueError::new_err)?
        )));
    }
    let (body_graph, output) = trace_while_region(py, body, &init, &operands, "body_fun")?;
    if output.shape != init.shape {
        return Err(PyValueError::new_err(format!(
            "tensor_while_loop_region body changed carry shape from {:?} to {:?}",
            init.shape, output.shape
        )));
    }
    let loop_plan = TensorWhileExecutionPlan::new(
        predicate_graph
            .compile_region_plan(predicate.node_id)
            .map_err(PyValueError::new_err)?,
        compile_loop_body(&body_graph, output.node_id, "__quabla_while_carry")
            .map_err(PyValueError::new_err)?,
        "__quabla_while_carry",
    )
    .map_err(PyValueError::new_err)?;
    // Compiled regions drop operands they never read; bind only the captures
    // that at least one region still declares.
    let captures = operands
        .iter()
        .enumerate()
        .map(|(index, operand)| (format!("__quabla_while_capture_{index}"), operand.node_id))
        .filter(|(name, _)| loop_plan.external_captures().contains_key(name))
        .collect();
    let parent_graph = init.graph.clone();
    let mut ir = parent_graph
        .ir
        .lock()
        .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
    let node_id = ir
        .while_loop(init.node_id, loop_plan, captures)
        .map_err(PyValueError::new_err)?;
    let shape = ir.node_shape(node_id).map_err(PyValueError::new_err)?;
    drop(ir);
    Ok(TraceTensor::from_node(parent_graph, node_id, shape, None))
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
    let body_plan = {
        let mut ir = body_graph
            .ir
            .lock()
            .map_err(|_| PyValueError::new_err("tensor trace graph lock is poisoned"))?;
        let mut plan = ir
            .compile_region_many(&[next.node_id, output.node_id])
            .map_err(PyValueError::new_err)?
            .0;
        // A body that reads its carry in neither result keeps it as an input
        // through `where(false, carry, next)`, as `compile_loop_body` does.
        if !region_input_names(&plan)
            .iter()
            .any(|name| name == "__quabla_scan_carry")
        {
            let carry = ir
                .input_node_id("__quabla_scan_carry")
                .map_err(PyValueError::new_err)?;
            let never = ir.scalar_constant(0.0);
            let retained = ir
                .where_select(never, carry, next.node_id)
                .map_err(PyValueError::new_err)?;
            plan = ir
                .compile_region_many(&[retained, output.node_id])
                .map_err(PyValueError::new_err)?
                .0;
        }
        plan
    };
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

const CUDA_LOSS_COTANGENT_NAME: &str = "__quabla_loss_cotangent";
const MLX_LOSS_COTANGENT_NAME: &str = "__quabla_mlx_loss_cotangent";

/// Compiles the scalar loss and its parameter gradients into one CUDA plan
/// through the checked facade, as `jit` compiles: a float64 loss gets `f64`
/// device buffers and a float32 one `f32`.
fn compile_cuda_scalar_value_and_grad_with_precision(
    py: Python<'_>,
    loss: &TensorTraceResult,
    parameter_names: Vec<String>,
    device_ordinal: usize,
    precision: QuablaPrecision,
) -> PyResult<(CudaTrainingPlan, BTreeMap<String, TensorNodeId>)> {
    use quabla_core::compiler::QuablaCompileError;

    let (program, parameter_names) =
        build_cuda_scalar_value_and_grad_program(loss, parameter_names)?;
    let target = QuablaTarget::Cuda { device_ordinal };
    let executable = QuablaCompiler
        .compile_many_checked_with_precision(&program, target, precision)
        .map_err(|error| match error {
            QuablaCompileError::Unavailable(message) => {
                crate::errors::device_operation_error(py, message, "Trainer", target.name())
            }
            QuablaCompileError::Unsupported { op, message } => {
                crate::errors::device_operation_error(py, message, &op, target.name())
            }
            QuablaCompileError::InvalidProgram(message) | QuablaCompileError::Backend(message) => {
                PyValueError::new_err(message)
            }
        })?;
    let (_, gradient_node_ids) =
        value_and_named_outputs(executable.output_node_ids(), parameter_names);
    let plan = match executable.into_executable() {
        QuablaExecutable::Cuda(plan) => CudaTrainingPlan::Float32(plan),
        QuablaExecutable::CudaFloat64(plan) => CudaTrainingPlan::Float64(plan),
        executable => {
            return Err(PyValueError::new_err(unexpected_executable(
                target,
                &executable,
            )))
        }
    };
    Ok((plan, gradient_node_ids))
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
    // The NCCL replicas run `f32` plans; a float64 loss is rejected rather
    // than converted. A build without data parallelism reports that first.
    if cfg!(all(feature = "cuda-nccl", target_os = "linux"))
        && plan
            .has_float64_values()
            .map_err(|(_, message)| PyValueError::new_err(message))?
    {
        return Err(crate::errors::device_operation_error(
            py,
            "CUDA data-parallel execution supports float32 programs only; cast the \
             parameters and data to float32 with astype"
                .to_string(),
            "float64",
            "cuda",
        ));
    }
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
    py: Python<'_>,
    loss: &TensorTraceResult,
    parameter_names: Vec<String>,
) -> PyResult<(
    TensorExecutionPlan,
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
    // The checked facade, as `jit` compiles: it validates the MLX lowering
    // and rejects a float64 loss, which MLX cannot run.
    use quabla_core::compiler::QuablaCompileError;
    let program = graph
        .into_multi_output_program(output_node_ids)
        .map_err(PyValueError::new_err)?;
    let executable = QuablaCompiler
        .compile_many_checked(&program, QuablaTarget::Mlx)
        .map_err(|error| match error {
            QuablaCompileError::Unavailable(message) => {
                crate::errors::device_operation_error(py, message, "Trainer", "mlx")
            }
            QuablaCompileError::Unsupported { op, message } => {
                crate::errors::device_operation_error(py, message, &op, "mlx")
            }
            QuablaCompileError::InvalidProgram(message) | QuablaCompileError::Backend(message) => {
                PyValueError::new_err(message)
            }
        })?;
    let (loss_node_id, gradient_node_ids) =
        value_and_named_outputs(executable.output_node_ids(), requested_names);
    let plan = match executable.into_executable() {
        QuablaExecutable::Mlx(plan) => plan,
        executable => {
            return Err(PyValueError::new_err(unexpected_executable(
                QuablaTarget::Mlx,
                &executable,
            )))
        }
    };
    Ok((plan, loss_node_id, gradient_node_ids))
}

/// The executor behind `quabla.optim.Trainer(..., device="mlx")`, with the
/// full device optimizer configuration (see `_cuda_trainer_optimizer`). MLX
/// has no float64, so there is no `precision`; `Trainer` rejects it first.
/// It is private: `Trainer` is the public entry.
#[pyfunction]
#[pyo3(name = "_mlx_trainer_optimizer", signature = (loss, parameter_names, inputs, retained_input_names, optimizer, learning_rate, beta1, beta2, epsilon, weight_decay, clip_norm))]
#[allow(clippy::too_many_arguments)]
pub fn mlx_trainer_optimizer(
    loss: &Bound<'_, PyAny>,
    parameter_names: Vec<String>,
    inputs: &Bound<'_, PyDict>,
    retained_input_names: Vec<String>,
    optimizer: &str,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    clip_norm: Option<f64>,
) -> PyResult<TensorMlxAdamOptimizer> {
    let config = trainer_optimizer_config(
        optimizer,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        clip_norm,
    )?;
    build_mlx_loss_optimizer(
        "Trainer",
        loss,
        parameter_names,
        inputs,
        Some(retained_input_names),
        |plan, loss_node_id, gradient_node_ids, values, retained_inputs| {
            MlxAdamPlan::with_config(
                plan,
                loss_node_id,
                gradient_node_ids,
                values,
                retained_inputs,
                config,
            )
        },
    )
}

/// Compiles the scalar loss and its parameter gradients into one MLX plan
/// and hands it to `make_plan` with the bound inputs and the retained input
/// names. `function` names the caller in error messages.
fn build_mlx_loss_optimizer(
    function: &str,
    loss: &Bound<'_, PyAny>,
    parameter_names: Vec<String>,
    inputs: &Bound<'_, PyDict>,
    retained_input_names: Option<Vec<String>>,
    make_plan: impl FnOnce(
        TensorExecutionPlan,
        usize,
        BTreeMap<String, usize>,
        &BTreeMap<String, DynamicTensor>,
        BTreeSet<String>,
    ) -> Result<MlxAdamPlan, String>,
) -> PyResult<TensorMlxAdamOptimizer> {
    let loss = loss
        .extract::<PyRef<'_, TraceTensor>>()
        .map_err(|_| PyTypeError::new_err(format!("{function} loss must be a TraceTensor")))?;
    let function_py = inputs.py();
    let loss = TensorTraceResult::new(loss.graph.clone(), loss.clone());
    let (plan, loss_node_id, gradient_node_ids) =
        compile_mlx_scalar_value_and_grad(function_py, &loss, parameter_names)?;
    let parameter_names = gradient_node_ids.keys().cloned().collect::<BTreeSet<_>>();
    let mut values = extract_tensor_map(inputs)?;
    if values.contains_key(MLX_LOSS_COTANGENT_NAME) {
        return Err(PyValueError::new_err(format!(
            "{function} reserves input name {MLX_LOSS_COTANGENT_NAME:?}"
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
    let plan = make_plan(
        plan,
        loss_node_id,
        gradient_node_ids,
        &values,
        retained_inputs,
    )
    .map_err(PyValueError::new_err)?;
    Ok(TensorMlxAdamOptimizer {
        plan,
        parameter_names,
        inputs: values,
    })
}

impl TensorCudaAdamOptimizer {
    /// Applies AdamW's decoupled decay to the parameters the loss ignores.
    ///
    /// Their gradient and moments are zero, so the shared canonical rule,
    /// evaluated here in float64 and rounded once exactly as the CPU trainer
    /// does, reduces to `p - (0 + weight_decay * p) * learning_rate` (the
    /// `0 +` keeps its signed-zero result). The update count does not matter:
    /// a zero moment over any positive bias correction is `+0`. Plain Adam
    /// and SGD leave them unchanged.
    fn decay_frozen_parameters(&mut self) -> Result<(), String> {
        let DeviceUpdateRule::Adam {
            beta1,
            beta2,
            epsilon,
            weight_decay,
        } = self.config.rule
        else {
            return Ok(());
        };
        if weight_decay == 0.0 {
            return Ok(());
        }
        let coefficients = AdamCoefficients::new(
            self.config.learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            1,
        );
        let mut arith = F64Arith::default();
        for value in self.frozen_parameters.values_mut() {
            let decayed = value
                .data()
                .iter()
                .map(|parameter| arith.adam(&coefficients, *parameter, 0.0, 0.0, 0.0).0)
                .collect();
            *value = DynamicTensor::from_storage(
                value.shape().to_vec(),
                quabla_core::tensor_ir::HostTensorStorage::from_f64(decayed, value.dtype()),
            )?;
        }
        Ok(())
    }
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
        self.shared_plan
            .plan
            .step(
                &self.shared_plan.gradient_node_ids,
                &self.inputs,
                &retained_inputs,
                &self.config,
            )
            .map_err(PyValueError::new_err)?;
        self.decay_frozen_parameters()
            .map_err(PyValueError::new_err)
    }

    /// Learning rate of the next step; host schedules set it between steps.
    #[getter]
    fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    #[setter]
    fn set_learning_rate(&mut self, learning_rate: f64) -> PyResult<()> {
        if !(learning_rate.is_finite() && learning_rate >= 0.0) {
            return Err(PyValueError::new_err(
                "CUDA Adam learning_rate must be finite and nonnegative",
            ));
        }
        self.config.learning_rate = learning_rate;
        Ok(())
    }

    fn parameters(&self) -> PyResult<BTreeMap<String, PyTensor>> {
        // A clipped step with a NaN or infinite global norm turns every
        // gradient into NaN on the CPU, also the zero gradients of the
        // parameters the loss ignores, which then stay NaN.
        let poisoned = self.config.clip_norm.is_some()
            && !self.frozen_parameters.is_empty()
            && self
                .shared_plan
                .plan
                .nonfinite_clip_seen()
                .map_err(PyValueError::new_err)?;
        let frozen = self.frozen_parameters.iter().map(|(name, value)| {
            let value = if poisoned {
                DynamicTensor::from_storage(
                    value.shape().to_vec(),
                    quabla_core::tensor_ir::HostTensorStorage::from_f64(
                        vec![f64::NAN; value.data().len()],
                        value.dtype(),
                    ),
                )
                .map_err(PyValueError::new_err)?
            } else {
                value.clone()
            };
            Ok((
                name.clone(),
                PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
            ))
        });
        self.parameter_names
            .iter()
            .map(|name| {
                let value = self
                    .shared_plan
                    .plan
                    .retained_input_to_host(name)
                    .map_err(PyValueError::new_err)?;
                Ok((
                    name.clone(),
                    PyTensor::from_dynamic_tensor(value).map_err(PyValueError::new_err)?,
                ))
            })
            .chain(frozen)
            .collect()
    }

    #[getter]
    fn device_buffer_count(&self) -> PyResult<usize> {
        self.shared_plan
            .plan
            .device_buffer_count()
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (inputs = None))]
    fn loss(&mut self, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
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
        let loss = self
            .shared_plan
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

/// The executor behind `quabla.optim.Trainer(..., device="cuda")`.
///
/// It takes the full device optimizer configuration (`optimizer` is
/// `"adam"` or `"sgd"`; a nonzero `weight_decay` makes Adam AdamW) and
/// `jit`'s `precision`, and keeps the hyperparameters in float64. It is
/// private: `Trainer` is the public entry.
#[pyfunction]
#[pyo3(name = "_cuda_trainer_optimizer", signature = (loss, parameter_names, inputs, retained_input_names, optimizer, learning_rate, beta1, beta2, epsilon, weight_decay, clip_norm, precision, device_ordinal))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_trainer_optimizer(
    py: Python<'_>,
    loss: &Bound<'_, PyAny>,
    parameter_names: Vec<String>,
    inputs: &Bound<'_, PyDict>,
    retained_input_names: Vec<String>,
    optimizer: &str,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    clip_norm: Option<f64>,
    precision: Option<&str>,
    device_ordinal: usize,
) -> PyResult<TensorCudaAdamOptimizer> {
    let config = trainer_optimizer_config(
        optimizer,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        clip_norm,
    )?;
    let precision = match precision {
        None => QuablaPrecision::Default,
        Some("float64") => QuablaPrecision::Float64,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "precision must be None or \"float64\", got {other:?}"
            )))
        }
    };
    build_cuda_loss_optimizer(
        py,
        "Trainer",
        loss,
        parameter_names,
        inputs,
        Some(retained_input_names),
        config,
        device_ordinal,
        precision,
    )
}

/// The validated device configuration of a `Trainer` optimizer.
pub(crate) fn trainer_optimizer_config(
    optimizer: &str,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    clip_norm: Option<f64>,
) -> PyResult<DeviceOptimizerConfig> {
    let rule = match optimizer {
        "adam" => DeviceUpdateRule::Adam {
            beta1,
            beta2,
            epsilon,
            weight_decay,
        },
        "sgd" => DeviceUpdateRule::Sgd,
        other => {
            return Err(PyValueError::new_err(format!(
                "device optimizer must be \"adam\" or \"sgd\", got {other:?}"
            )))
        }
    };
    let config = DeviceOptimizerConfig {
        rule,
        learning_rate,
        clip_norm,
    };
    config.validate().map_err(PyValueError::new_err)?;
    Ok(config)
}

/// Compiles the scalar loss and its parameter gradients into one shared
/// CUDA plan and wraps it with the optimizer state. `function` names the
/// caller in error messages.
#[allow(clippy::too_many_arguments)]
fn build_cuda_loss_optimizer(
    py: Python<'_>,
    function: &str,
    loss: &Bound<'_, PyAny>,
    parameter_names: Vec<String>,
    inputs: &Bound<'_, PyDict>,
    retained_input_names: Option<Vec<String>>,
    config: DeviceOptimizerConfig,
    device_ordinal: usize,
    precision: QuablaPrecision,
) -> PyResult<TensorCudaAdamOptimizer> {
    let loss = loss
        .extract::<PyRef<'_, TraceTensor>>()
        .map_err(|_| PyTypeError::new_err(format!("{function} loss must be a TraceTensor")))?;
    let loss = TensorTraceResult::new(loss.graph.clone(), loss.clone());
    let (plan, mut gradient_node_ids) = compile_cuda_scalar_value_and_grad_with_precision(
        py,
        &loss,
        parameter_names,
        device_ordinal,
        precision,
    )?;
    let mut values = extract_tensor_map(inputs)?;
    let frozen_names = gradient_node_ids
        .keys()
        .filter(|name| plan.plan().input_shape(name).is_err())
        .cloned()
        .collect::<Vec<_>>();
    let mut frozen_parameters = BTreeMap::new();
    for name in frozen_names {
        gradient_node_ids.remove(&name);
        let value = values.get(&name).cloned().ok_or_else(|| {
            PyValueError::new_err(format!("{function} parameter {name:?} has no value"))
        })?;
        frozen_parameters.insert(name, value);
    }
    let parameter_names = gradient_node_ids.keys().cloned().collect::<BTreeSet<_>>();
    if values.contains_key(CUDA_LOSS_COTANGENT_NAME) {
        return Err(PyValueError::new_err(format!(
            "{function} reserves input name {CUDA_LOSS_COTANGENT_NAME:?}"
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
        shared_plan: SharedCudaAdamPlan {
            plan,
            gradient_node_ids,
        },
        parameter_names,
        frozen_parameters,
        inputs: values,
        retained_inputs,
        config,
    })
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
