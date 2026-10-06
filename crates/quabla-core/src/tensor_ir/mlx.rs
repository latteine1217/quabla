// mlx-rs 0.32 deprecates the `*_device(..., stream)` op variants in favour of scoping ops with
// `mlx_rs::with_stream`. The deprecated variants are thin wrappers over that scope, so this
// module keeps passing the GPU stream explicitly without a behaviour change.
// TODO: build ops inside `with_stream` scopes and remove this allow.
#![allow(deprecated)]

#[path = "mlx_cholesky.rs"]
mod cholesky_backend;
#[path = "mlx_scatter.rs"]
mod scatter_backend;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use mlx_rs::{ops, transforms, Array, Dtype, StreamOrDevice};

use super::{
    sqrt_derivative_coefficient, DynamicTensor, TensorBackend, TensorComparison, TensorConstant,
    TensorDType, TensorDeviceBackend, TensorExecutionPlan, TensorForiExecutionPlan, TensorOp,
    UnaryMathKind,
};

/// Apple MLX backend for the supported rank-N Tensor IR primitives.
///
/// MLX uses unified memory on Apple silicon. This backend keeps intermediate
/// arrays on MLX's GPU stream and only materializes the final value for Quabla's
/// host-facing `DynamicTensor` result.
///
/// Thread safety: all MLX graph construction, evaluation, and readback are serialized by a
/// process-wide lock, so the backend can be called from multiple threads.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxBackend;

/// Serializes all access to MLX from this module.
///
/// The lock was introduced for MLX 0.25, which encoded GPU work into a process-wide, unguarded
/// default GPU stream: two threads evaluating concurrently interleaved encoding into the same
/// Metal command buffer, triggering a Metal assertion abort or a hang. MLX 0.32.2 (mlx-rs 0.32)
/// gives each thread its own default stream and command encoder, but only supports multiple
/// threads for independent computations; retained parameters and Adam state are shared across
/// calls that may run on different threads, so evaluation stays serialized.
static MLX_EXECUTION_LOCK: Mutex<()> = Mutex::new(());

/// Acquires the MLX process lock.
///
/// std `Mutex` is not reentrant: acquire it only once at public entry points, internal helpers
/// always assume it is held, and a public function may delegate to exactly one other lock-acquiring
/// public function.
fn mlx_execution_guard() -> MutexGuard<'static, ()> {
    // The lock guards no Rust data; poisoning only means a previous holder panicked, so keep
    // serializing.
    MLX_EXECUTION_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Named MLX arrays retained across executions of a fixed Tensor IR plan.
///
/// This owns long-lived parameters and, later, optimizer state. Dynamic batch
/// inputs remain ordinary host bindings and are uploaded for each execution.
#[derive(Clone, Debug, Default)]
pub struct MlxRetainedInputs {
    values: BTreeMap<String, Array>,
}

#[derive(Debug)]
struct MlxAdamState {
    first_moment: Array,
    second_moment: Array,
    step: u64,
}

#[derive(Debug)]
struct MlxForiVjpEvaluation {
    carry_gradient: Array,
    external_gradients: BTreeMap<String, Array>,
}

#[derive(Debug)]
struct MlxForiVjpJvpEvaluation {
    gradients: BTreeMap<String, Array>,
}

#[derive(Debug)]
struct MlxScanVjpEvaluation {
    carry_gradient: Array,
    external_gradients: BTreeMap<String, Array>,
}

#[derive(Debug)]
struct MlxScanVjpJvpEvaluation {
    gradients: BTreeMap<String, Array>,
}

/// A scalar-loss MLX training plan with retained parameters and Adam state.
///
/// `step` only evaluates gradients, updated parameters, and moment buffers on
/// MLX's GPU stream. Host materialization is limited to explicit diagnostics.
#[derive(Debug)]
pub struct MlxAdamPlan {
    plan: TensorExecutionPlan,
    loss_node_id: usize,
    forward_loss_plan: OnceLock<Result<TensorExecutionPlan, String>>,
    gradient_node_ids: BTreeMap<String, usize>,
    // Parameter shape and dtype from the initial values. A parameter the loss
    // does not depend on is pruned from the plan, so its layout cannot be
    // read from the plan's input nodes.
    parameter_layouts: BTreeMap<String, (Vec<usize>, TensorDType)>,
    retained_inputs: MlxRetainedInputs,
    adam: BTreeMap<String, MlxAdamState>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl MlxRetainedInputs {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn upload(
        inputs: &BTreeMap<String, DynamicTensor>,
        names: impl IntoIterator<Item = String>,
    ) -> Result<Self, String> {
        let _guard = mlx_execution_guard();
        let mut values = BTreeMap::new();
        for name in names {
            let input = inputs
                .get(&name)
                .ok_or_else(|| format!("cannot retain missing input {name:?}"))?;
            values.insert(name, mlx_array_from_dynamic(input)?);
        }
        Ok(Self { values })
    }

    pub fn replace(&mut self, name: String, value: &DynamicTensor) -> Result<(), String> {
        let _guard = mlx_execution_guard();
        self.values.insert(name, mlx_array_from_dynamic(value)?);
        Ok(())
    }

    pub fn arrays(&self) -> &BTreeMap<String, Array> {
        &self.values
    }

    pub fn get(&self, name: &str) -> Option<&Array> {
        self.values.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }
}

impl MlxAdamPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plan: TensorExecutionPlan,
        loss_node_id: usize,
        gradient_node_ids: BTreeMap<String, usize>,
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_input_names: impl IntoIterator<Item = String>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self, String> {
        if gradient_node_ids.is_empty() {
            return Err("MLX Adam requires at least one parameter gradient".to_string());
        }
        // A zero learning rate is valid: learning-rate schedules start or end
        // at zero, and a zero step still advances the moments.
        if !(learning_rate.is_finite()
            && learning_rate >= 0.0
            && epsilon.is_finite()
            && epsilon > 0.0
            && beta1.is_finite()
            && (0.0..1.0).contains(&beta1)
            && beta2.is_finite()
            && (0.0..1.0).contains(&beta2))
        {
            return Err(
                "MLX Adam requires a finite nonnegative learning_rate, a positive finite epsilon, and beta1/beta2 in [0, 1)"
                    .to_string(),
            );
        }
        let mut names = retained_input_names.into_iter().collect::<Vec<_>>();
        names.extend(gradient_node_ids.keys().cloned());
        names.sort();
        names.dedup();
        let retained_inputs = MlxRetainedInputs::upload(inputs, names)?;
        let mut parameter_layouts = BTreeMap::new();
        for parameter_name in gradient_node_ids.keys() {
            let initial = inputs.get(parameter_name).ok_or_else(|| {
                format!("MLX Adam parameter {parameter_name:?} has no initial value")
            })?;
            let parameter = retained_inputs
                .values
                .get(parameter_name)
                .ok_or_else(|| format!("MLX Adam did not retain parameter {parameter_name:?}"))?;
            if parameter.shape() != mlx_shape(initial.shape())?.as_slice() {
                return Err(format!(
                    "MLX Adam parameter {parameter_name:?} has an unexpected retained shape"
                ));
            }
            parameter_layouts.insert(
                parameter_name.clone(),
                (initial.shape().to_vec(), initial.dtype()),
            );
        }
        Ok(Self {
            plan,
            loss_node_id,
            forward_loss_plan: OnceLock::new(),
            gradient_node_ids,
            parameter_layouts,
            retained_inputs,
            adam: BTreeMap::new(),
            learning_rate,
            beta1,
            beta2,
            epsilon,
        })
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Replace the learning rate used by later steps, as a host schedule does
    /// between steps. Zero is accepted because schedules such as warmup and
    /// cosine decay reach it; moments and bias corrections are unaffected.
    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<(), String> {
        if !(learning_rate.is_finite() && learning_rate >= 0.0) {
            return Err("MLX Adam learning_rate must be finite and nonnegative".to_string());
        }
        self.learning_rate = learning_rate;
        Ok(())
    }

    pub fn step(&mut self, inputs: &BTreeMap<String, DynamicTensor>) -> Result<(), String> {
        let _guard = mlx_execution_guard();
        let parameter_names = self.gradient_node_ids.keys().cloned().collect::<Vec<_>>();
        let gradient_node_ids = parameter_names
            .iter()
            .map(|name| {
                self.gradient_node_ids
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("MLX Adam gradient for {name:?} is missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let gradients = MlxBackend.execute_arrays_with_retained(
            &self.plan,
            &gradient_node_ids,
            inputs,
            self.retained_inputs.arrays(),
        )?;
        let stream = StreamOrDevice::gpu();
        for (parameter_name, gradient) in parameter_names.iter().zip(gradients) {
            let parameter = self
                .retained_inputs
                .values
                .get(parameter_name)
                .ok_or_else(|| {
                    format!("MLX Adam retained parameter {parameter_name:?} is missing")
                })?
                .clone();
            if parameter.shape() != gradient.shape() {
                return Err(format!(
                    "MLX Adam parameter {parameter_name:?} and gradient shapes differ"
                ));
            }
            if !self.adam.contains_key(parameter_name) {
                let shape = parameter.shape();
                let first_moment = Array::zeros_device::<f32>(shape, &stream)
                    .map_err(|error| format!("MLX Adam first moment allocation failed: {error}"))?;
                let second_moment =
                    Array::zeros_device::<f32>(shape, &stream).map_err(|error| {
                        format!("MLX Adam second moment allocation failed: {error}")
                    })?;
                self.adam.insert(
                    parameter_name.clone(),
                    MlxAdamState {
                        first_moment,
                        second_moment,
                        step: 0,
                    },
                );
            }
            let state = self
                .adam
                .get_mut(parameter_name)
                .ok_or_else(|| format!("MLX Adam state for {parameter_name:?} is missing"))?;
            state.step = state
                .step
                .checked_add(1)
                .ok_or_else(|| "MLX Adam step counter overflow".to_string())?;
            let one_minus_beta1 = Array::from_f32(1.0 - self.beta1);
            let one_minus_beta2 = Array::from_f32(1.0 - self.beta2);
            let first_moment = state
                .first_moment
                .multiply_device(Array::from_f32(self.beta1), &stream)
                .and_then(|value| {
                    gradient
                        .multiply_device(&one_minus_beta1, &stream)
                        .and_then(|scaled_gradient| value.add_device(&scaled_gradient, &stream))
                })
                .map_err(|error| format!("MLX Adam first moment update failed: {error}"))?;
            let second_moment = state
                .second_moment
                .multiply_device(Array::from_f32(self.beta2), &stream)
                .and_then(|value| {
                    gradient
                        .multiply_device(&gradient, &stream)
                        .and_then(|squared_gradient| {
                            squared_gradient.multiply_device(&one_minus_beta2, &stream)
                        })
                        .and_then(|scaled_gradient| value.add_device(&scaled_gradient, &stream))
                })
                .map_err(|error| format!("MLX Adam second moment update failed: {error}"))?;
            let correction1 = Array::from_f32(1.0 - self.beta1.powf(state.step as f32));
            let correction2 = Array::from_f32(1.0 - self.beta2.powf(state.step as f32));
            let update = first_moment
                .divide_device(&correction1, &stream)
                .and_then(|first| {
                    second_moment
                        .divide_device(&correction2, &stream)
                        .and_then(|second| second.sqrt_device(&stream))
                        .and_then(|denominator| {
                            denominator.add_device(Array::from_f32(self.epsilon), &stream)
                        })
                        .and_then(|denominator| first.divide_device(&denominator, &stream))
                })
                .and_then(|normalized| {
                    normalized.multiply_device(Array::from_f32(self.learning_rate), &stream)
                })
                .map_err(|error| format!("MLX Adam update failed: {error}"))?;
            let updated_parameter = parameter
                .subtract_device(&update, &stream)
                .map_err(|error| format!("MLX Adam parameter update failed: {error}"))?;
            state.first_moment = first_moment;
            state.second_moment = second_moment;
            self.retained_inputs
                .values
                .insert(parameter_name.clone(), updated_parameter);
        }
        let mut values = self.retained_inputs.values.values().collect::<Vec<_>>();
        for state in self.adam.values() {
            values.push(&state.first_moment);
            values.push(&state.second_moment);
        }
        transforms::eval(values)
            .map_err(|error| format!("MLX Adam state evaluation failed: {error}"))
    }

    pub fn loss(&self, inputs: &BTreeMap<String, DynamicTensor>) -> Result<DynamicTensor, String> {
        // Cache only graph metadata; bind current device parameters and the latest batch.
        let forward = self
            .forward_loss_plan
            .get_or_init(|| self.plan.as_ir().compile_cpu(self.loss_node_id))
            .as_ref()
            .map_err(Clone::clone)?;
        MlxBackend
            .execute_many_with_retained(
                forward,
                &[forward.output_node_id],
                inputs,
                self.retained_inputs.arrays(),
            )
            .and_then(|mut values| {
                values
                    .pop()
                    .ok_or_else(|| "MLX Adam loss evaluation produced no output".to_string())
            })
    }

    pub fn parameter(&self, name: &str) -> Result<DynamicTensor, String> {
        let _guard = mlx_execution_guard();
        let (shape, dtype) = self
            .parameter_layouts
            .get(name)
            .ok_or_else(|| format!("MLX Adam parameter {name:?} is not a parameter"))?;
        let parameter = self
            .retained_inputs
            .values
            .get(name)
            .ok_or_else(|| format!("MLX Adam parameter {name:?} is not retained"))?;
        let parameter = mlx_row_major(parameter, &StreamOrDevice::gpu())?;
        DynamicTensor::from_storage(
            shape.clone(),
            super::HostTensorStorage::from_f32(
                mlx_host_values(&parameter)
                    .map_err(|error| format!("MLX Adam parameter {name:?} {error}"))?,
                *dtype,
            ),
        )
    }
}

impl MlxBackend {
    /// Executes one compiled graph and materializes each requested output.
    ///
    /// All requested outputs share the same MLX value table, so symbolic VJP
    /// graphs can return their primal loss and several gradients without
    /// rebuilding the graph or recomputing its common prefix.
    pub fn execute_many(
        &self,
        plan: &TensorExecutionPlan,
        output_node_ids: &[usize],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        self.execute_many_with_retained(plan, output_node_ids, inputs, &BTreeMap::new())
    }

    /// Executes a graph to completion without materializing its output on the host.
    ///
    /// Dynamic host inputs are still uploaded for this call. The requested output
    /// is evaluated on the MLX GPU stream and then discarded, which makes this a
    /// useful boundary for forward-execution timing without diagnostic readback.
    pub fn execute_without_output(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(), String> {
        let _guard = mlx_execution_guard();
        self.execute_arrays_with_retained(plan, &[plan.output_node_id], inputs, &BTreeMap::new())
            .map(|_| ())
    }

    /// Executes a graph to completion using retained input arrays without host readback.
    pub fn execute_without_output_with_state(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
        state: &MlxRetainedInputs,
    ) -> Result<(), String> {
        let _guard = mlx_execution_guard();
        self.execute_arrays_with_retained(plan, &[plan.output_node_id], inputs, state.arrays())
            .map(|_| ())
    }

    /// Executes a graph while reusing named MLX input arrays.
    ///
    /// Retained arrays take precedence over same-named host inputs. Callers
    /// must create them from the matching Tensor IR input shapes.
    pub fn execute_many_with_retained(
        &self,
        plan: &TensorExecutionPlan,
        output_node_ids: &[usize],
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeMap<String, Array>,
    ) -> Result<Vec<DynamicTensor>, String> {
        let _guard = mlx_execution_guard();
        let stream = StreamOrDevice::gpu();
        // Row-major copies are built before the single evaluation so that they run in the same
        // GPU submission as the graph; an output that is already row-major shares its buffer.
        let outputs = self
            .lower_arrays_with_retained(plan, output_node_ids, inputs, retained_inputs)?
            .iter()
            .map(|output| mlx_row_major(output, &stream))
            .collect::<Result<Vec<_>, _>>()?;
        transforms::eval(&outputs)
            .map_err(|error| format!("MLX output evaluation failed: {error}"))?;
        output_node_ids
            .iter()
            .zip(&outputs)
            .map(|(node_id, output)| {
                let node = plan
                    .nodes
                    .get(*node_id)
                    .ok_or_else(|| format!("MLX output node {node_id} is missing"))?;
                // Readback is f32; tag it with the node's logical dtype (an f64 node holds the
                // result of f32-demoted execution).
                DynamicTensor::from_storage(
                    node.shape.clone(),
                    super::HostTensorStorage::from_f32(
                        mlx_host_values(output)
                            .map_err(|error| format!("MLX output node {node_id} {error}"))?,
                        node.dtype,
                    ),
                )
            })
            .collect()
    }

    /// Executes a graph and returns evaluated MLX arrays without host readback.
    fn execute_arrays_with_retained(
        &self,
        plan: &TensorExecutionPlan,
        output_node_ids: &[usize],
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeMap<String, Array>,
    ) -> Result<Vec<Array>, String> {
        let outputs =
            self.lower_arrays_with_retained(plan, output_node_ids, inputs, retained_inputs)?;
        transforms::eval(&outputs)
            .map_err(|error| format!("MLX output evaluation failed: {error}"))?;
        Ok(outputs)
    }

    /// Builds the MLX arrays of the requested outputs without evaluating them.
    ///
    /// Region nodes still synchronize internally (a `Cond` reads its predicate back and loops
    /// evaluate their carries), but the returned arrays are lazy until the caller evaluates them.
    fn lower_arrays_with_retained(
        &self,
        plan: &TensorExecutionPlan,
        output_node_ids: &[usize],
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeMap<String, Array>,
    ) -> Result<Vec<Array>, String> {
        if output_node_ids.is_empty() {
            return Err("MLX execution requires at least one output node".to_string());
        }
        let stream = StreamOrDevice::gpu();
        let mut values = Vec::with_capacity(plan.nodes.len());
        let mut fori_vjp_cache: HashMap<usize, MlxForiVjpEvaluation> = HashMap::new();
        let mut fori_vjp_jvp_cache: HashMap<usize, MlxForiVjpJvpEvaluation> = HashMap::new();
        let mut scan_cache: HashMap<usize, (Array, Array)> = HashMap::new();
        let mut scan_vjp_cache: HashMap<usize, MlxScanVjpEvaluation> = HashMap::new();
        let mut scan_vjp_jvp_cache: HashMap<usize, MlxScanVjpJvpEvaluation> = HashMap::new();

        for (node_id, node) in plan.nodes.iter().enumerate() {
            // Every array in this backend is f32; any logical dtype that cannot be demoted to f32
            // must be rejected explicitly.
            if TensorDeviceBackend::Mlx.execution_dtype(node.dtype)? != TensorDType::F32 {
                return Err(format!(
                    "MLX backend cannot execute node {node_id} of dtype {}",
                    node.dtype
                ));
            }
            let value = match &node.op {
                TensorOp::Input { name } => {
                    if let Some(input) = retained_inputs.get(name) {
                        Ok(input.clone())
                    } else {
                        let input = inputs
                            .get(name)
                            .ok_or_else(|| format!("missing input {name:?}"))?;
                        if input.shape() != node.shape {
                            return Err(format!(
                                "input {name:?} has shape {:?}, expected {:?}",
                                input.shape(),
                                node.shape
                            ));
                        }
                        let shape = mlx_shape(&node.shape)?;
                        let data = input
                            .storage()
                            .iter()
                            .map(|value| value as f32)
                            .collect::<Vec<_>>();
                        Ok(Array::from_slice(&data, &shape))
                    }
                }
                TensorOp::ScalarConstant { value } if value.is_finite() => {
                    Ok(Array::from_f32(*value as f32))
                }
                TensorOp::ScalarConstant { .. } => {
                    return Err("MLX backend does not support non-finite constants".to_string())
                }
                TensorOp::Constant { value } => value.mlx_array(),
                // Source and target both execute as f32 (see the check above), so cast is the
                // identity.
                TensorOp::Cast { input } => Ok(mlx_value(&values, *input)?.clone()),
                TensorOp::Add { lhs, rhs } => mlx_value(&values, *lhs)?
                    .add_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Sub { lhs, rhs } => mlx_value(&values, *lhs)?
                    .subtract_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Div { lhs, rhs } => mlx_value(&values, *lhs)?
                    .divide_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Mul { lhs, rhs } => mlx_value(&values, *lhs)?
                    .multiply_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                // IR greater is a 0/1 float mask (matching the CPU); MLX comparisons produce bool,
                // which must be converted back to f32 or readback, accumulation, and Cond
                // predicates would hit a dtype mismatch.
                TensorOp::Greater { lhs, rhs } => mlx_value(&values, *lhs)?
                    .gt_device(mlx_value(&values, *rhs)?, &stream)
                    .and_then(|mask| mask.as_dtype_device(Dtype::Float32, &stream))
                    .map_err(|error| error.to_string()),
                // Bool nodes likewise cross node boundaries as f32 0/1
                // (`execution_dtype(Bool) = f32`).
                TensorOp::Compare { lhs, rhs, kind } => {
                    let lhs = mlx_value(&values, *lhs)?;
                    let rhs = mlx_value(&values, *rhs)?;
                    match kind {
                        TensorComparison::Greater => lhs.gt_device(rhs, &stream),
                        TensorComparison::GreaterEqual => lhs.ge_device(rhs, &stream),
                        TensorComparison::Less => lhs.lt_device(rhs, &stream),
                        TensorComparison::LessEqual => lhs.le_device(rhs, &stream),
                        TensorComparison::Equal => lhs.eq_device(rhs, &stream),
                        TensorComparison::NotEqual => lhs.ne_device(rhs, &stream),
                    }
                    .and_then(|mask| mask.as_dtype_device(Dtype::Float32, &stream))
                    .map_err(|error| error.to_string())
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => ops::r#where_device(
                    mlx_value(&values, *condition)?,
                    mlx_value(&values, *on_true)?,
                    mlx_value(&values, *on_false)?,
                    &stream,
                )
                .map_err(|error| error.to_string()),
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => {
                    // Host synchronization point: the scalar predicate is read back once and only
                    // the selected branch runs on the GPU stream. Computing both branches with
                    // `where` is avoided so NaN/Inf from the unselected branch cannot leak into
                    // values or gradients.
                    let predicate = mlx_scalar_predicate(mlx_value(&values, *predicate)?)?;
                    let branch_inputs = captures
                        .iter()
                        .map(|(name, node_id)| {
                            mlx_value(&values, *node_id).map(|value| (name.clone(), value.clone()))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    mlx_execute_plan_output(self, branches.selected(predicate), &branch_inputs)
                }
                TensorOp::While {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    // Host synchronization point per iteration: the scalar predicate is read back
                    // (like `Cond`) and decides whether the body region is dispatched again.
                    let mut region_inputs = captures
                        .iter()
                        .map(|(name, node_id)| {
                            mlx_value(&values, *node_id).map(|value| (name.clone(), value.clone()))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    let carry_name = loop_plan.carry_name().to_string();
                    region_inputs.insert(carry_name.clone(), mlx_value(&values, *carry)?.clone());
                    loop {
                        let predicate =
                            mlx_execute_plan_output(self, loop_plan.predicate_plan(), &region_inputs)?;
                        if !mlx_scalar_predicate(&predicate)? {
                            break;
                        }
                        let next =
                            mlx_execute_plan_output(self, loop_plan.body_plan(), &region_inputs)?;
                        region_inputs.insert(carry_name.clone(), next);
                    }
                    region_inputs
                        .remove(&carry_name)
                        .ok_or_else(|| "MLX While carry is missing".to_string())
                }
                TensorOp::Fori { .. } => {
                    let TensorOp::Fori {
                        carry,
                        loop_plan,
                        captures,
                    } = &node.op
                    else {
                        unreachable!();
                    };
                    let mut carry = mlx_value(&values, *carry)?.clone();
                    let external_captures = captures
                        .iter()
                        .map(|(name, node_id)| {
                            mlx_value(&values, *node_id).map(|value| (name.clone(), value.clone()))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    for index in loop_plan.lower..loop_plan.upper {
                        let mut body_inputs = external_captures.clone();
                        body_inputs.insert(loop_plan.carry_name.clone(), carry);
                        body_inputs.insert(
                            loop_plan.index_name.clone(),
                            mlx_array_from_dynamic(&DynamicTensor::filled(
                                vec![], index as f64,
                            )?)?,
                        );
                        let mut outputs = self.execute_arrays_with_retained(
                            &loop_plan.body.plan,
                            &[loop_plan.body.plan.output_node_id],
                            &BTreeMap::new(),
                            &body_inputs,
                        )?;
                        carry = outputs.pop().ok_or_else(|| {
                            "MLX Fori body execution produced no carry output".to_string()
                        })?;
                    }
                    Ok(carry)
                }
                TensorOp::ForiJvp {
                    carry,
                    carry_tangent,
                    loop_plan,
                    captures,
                    tangent_captures,
                } => {
                    let captures = captures
                        .iter()
                        .map(|(name, node_id)| {
                            mlx_value(&values, *node_id)
                                .map(|value| (name.clone(), value.clone()))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    let tangents = tangent_captures
                        .iter()
                        .map(|(name, node_id)| {
                            mlx_value(&values, *node_id)
                                .map(|value| (name.clone(), value.clone()))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    mlx_fori_jvp(
                        self,
                        loop_plan,
                        mlx_value(&values, *carry)?.clone(),
                        mlx_value(&values, *carry_tangent)?.clone(),
                        &captures,
                        &tangents,
                    )
                }
                TensorOp::ForiVjp {
                    carry,
                    output_cotangent,
                    loop_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !fori_vjp_cache.contains_key(group) {
                        let captures = captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let evaluation = mlx_fori_value_and_vjp(
                            self,
                            loop_plan,
                            mlx_value(&values, *carry)?.clone(),
                            &captures,
                            mlx_value(&values, *output_cotangent)?.clone(),
                        )?;
                        fori_vjp_cache.insert(*group, evaluation);
                    }
                    let cached = fori_vjp_cache.get(group).ok_or_else(|| {
                        format!("MLX Fori VJP group {group} was not cached after evaluation")
                    })?;
                    match target {
                        super::TensorForiVjpTarget::Carry => Ok(cached.carry_gradient.clone()),
                        super::TensorForiVjpTarget::External(name) => cached
                            .external_gradients
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("MLX Fori VJP has no gradient for {name:?}")),
                    }
                }
                TensorOp::ForiVjpJvp {
                    carry,
                    carry_tangent,
                    output_cotangent,
                    output_cotangent_tangent,
                    plan,
                    captures,
                    tangent_captures,
                    target,
                    group,
                } => {
                    if !fori_vjp_jvp_cache.contains_key(group) {
                        let captures = captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let tangents = tangent_captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let gradients = mlx_fori_vjp_jvp(
                            self,
                            plan,
                            mlx_value(&values, *carry)?.clone(),
                            mlx_value(&values, *carry_tangent)?.clone(),
                            &captures,
                            &tangents,
                            mlx_value(&values, *output_cotangent)?.clone(),
                            mlx_value(&values, *output_cotangent_tangent)?.clone(),
                        )?;
                        fori_vjp_jvp_cache.insert(*group, MlxForiVjpJvpEvaluation { gradients });
                    }
                    let cached = fori_vjp_jvp_cache.get(group).ok_or_else(|| {
                        format!("MLX Fori VJP JVP group {group} was not cached after evaluation")
                    })?;
                    let name = match target {
                        super::TensorForiVjpTarget::Carry => &plan.loop_plan.carry_name,
                        super::TensorForiVjpTarget::External(name) => name,
                    };
                    cached
                        .gradients
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("MLX Fori VJP JVP has no gradient for {name:?}"))
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !scan_cache.contains_key(group) {
                        let external_captures = captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let mut carry = mlx_value(&values, *carry)?.clone();
                        let mut outputs = Vec::with_capacity(scan_plan.upper - scan_plan.lower);
                        for index in scan_plan.lower..scan_plan.upper {
                            let mut body_inputs = external_captures.clone();
                            body_inputs.insert(scan_plan.carry_name.clone(), carry);
                            body_inputs.insert(
                                scan_plan.index_name.clone(),
                                mlx_array_from_dynamic(&DynamicTensor::filled(
                                    vec![], index as f64,
                                )?)?,
                            );
                            let values = self.execute_arrays_with_retained(
                                &scan_plan.body.plan,
                                &scan_plan.body.plan.output_node_ids,
                                &BTreeMap::new(),
                                &body_inputs,
                            )?;
                            carry = values
                                .first()
                                .cloned()
                                .ok_or_else(|| "MLX Scan body has no carry output".to_string())?;
                            let output = values
                                .get(1)
                                .cloned()
                                .ok_or_else(|| "MLX Scan body has no output value".to_string())?;
                            outputs.push(output);
                        }
                        let output_shape = scan_plan.output_shape()?;
                        let step_shape = std::iter::once(1_i32)
                            .chain(mlx_shape(&output_shape[1..])?)
                            .collect::<Vec<_>>();
                        let outputs = outputs
                            .iter()
                            .map(|output| {
                                output
                                    .reshape_device(&step_shape, &stream)
                                    .map_err(|error| error.to_string())
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let stacked = ops::concatenate_axis_device(&outputs, 0, &stream)
                            .map_err(|error| error.to_string())?;
                        scan_cache.insert(*group, (carry, stacked));
                    }
                    let (carry, outputs) = scan_cache.get(group).ok_or_else(|| {
                        format!("MLX Scan group {group} was not cached after evaluation")
                    })?;
                    Ok(match target {
                        super::TensorScanTarget::Carry => carry.clone(),
                        super::TensorScanTarget::Outputs => outputs.clone(),
                    })
                }
                TensorOp::ScanVjp {
                    carry,
                    final_carry_cotangent,
                    output_cotangent,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !scan_vjp_cache.contains_key(group) {
                        let captures = captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let evaluation = mlx_scan_value_and_vjp(
                            self,
                            scan_plan,
                            mlx_value(&values, *carry)?.clone(),
                            &captures,
                            mlx_value(&values, *final_carry_cotangent)?.clone(),
                            mlx_value(&values, *output_cotangent)?.clone(),
                        )?;
                        scan_vjp_cache.insert(*group, evaluation);
                    }
                    let cached = scan_vjp_cache.get(group).ok_or_else(|| {
                        format!("MLX Scan VJP group {group} was not cached after evaluation")
                    })?;
                    match target {
                        super::TensorScanVjpTarget::Carry => Ok(cached.carry_gradient.clone()),
                        super::TensorScanVjpTarget::External(name) => cached
                            .external_gradients
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("MLX Scan VJP has no gradient for {name:?}")),
                    }
                }
                TensorOp::ScanVjpJvp {
                    carry,
                    carry_tangent,
                    final_carry_cotangent,
                    final_carry_cotangent_tangent,
                    output_cotangent,
                    output_cotangent_tangent,
                    plan,
                    captures,
                    tangent_captures,
                    target,
                    group,
                } => {
                    if !scan_vjp_jvp_cache.contains_key(group) {
                        let captures = captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let tangents = tangent_captures
                            .iter()
                            .map(|(name, node_id)| {
                                mlx_value(&values, *node_id)
                                    .map(|value| (name.clone(), value.clone()))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?;
                        let gradients = mlx_scan_vjp_jvp(
                            self,
                            plan,
                            mlx_value(&values, *carry)?.clone(),
                            mlx_value(&values, *carry_tangent)?.clone(),
                            &captures,
                            &tangents,
                            mlx_value(&values, *final_carry_cotangent)?.clone(),
                            mlx_value(&values, *final_carry_cotangent_tangent)?.clone(),
                            mlx_value(&values, *output_cotangent)?.clone(),
                            mlx_value(&values, *output_cotangent_tangent)?.clone(),
                        )?;
                        scan_vjp_jvp_cache
                            .insert(*group, MlxScanVjpJvpEvaluation { gradients });
                    }
                    let cached = scan_vjp_jvp_cache.get(group).ok_or_else(|| {
                        format!("MLX Scan VJP JVP group {group} was not cached after evaluation")
                    })?;
                    let name = match target {
                        super::TensorScanVjpTarget::Carry => &plan.scan_plan.carry_name,
                        super::TensorScanVjpTarget::External(name) => name,
                    };
                    cached.gradients.get(name).cloned().ok_or_else(|| {
                        format!("MLX Scan VJP JVP has no gradient for {name:?}")
                    })
                }
                TensorOp::Tanh { input } => ops::tanh_device(mlx_value(&values, *input)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Exp { input } => mlx_value(&values, *input)?
                    .exp_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Sqrt { input } => mlx_value(&values, *input)?
                    .sqrt_device(&stream)
                    .map_err(|error| error.to_string()),
                // Matches the CPU `sqrt_derivative_value`: zero at +-0, NaN for every negative
                // input (`-inf` included, where `power` alone is not NaN), and NaN propagated
                // through `power`.
                TensorOp::SqrtDerivative { input, order } => {
                    let input = mlx_value(&values, *input)?;
                    let zero = Array::from_f32(0.0);
                    let nan = Array::from_f32(f32::NAN);
                    let exponent = Array::from_f32(0.5 - *order as f32);
                    let coefficient = Array::from_f32(sqrt_derivative_coefficient(*order) as f32);
                    let power = input
                        .power_device(&exponent, &stream)
                        .map_err(|error| error.to_string())?;
                    let scaled = power
                        .multiply_device(&coefficient, &stream)
                        .map_err(|error| error.to_string())?;
                    let negative = input
                        .lt_device(&zero, &stream)
                        .map_err(|error| error.to_string())?;
                    let domain = ops::r#where_device(&negative, &nan, &scaled, &stream)
                        .map_err(|error| error.to_string())?;
                    let origin = input
                        .eq_device(&zero, &stream)
                        .map_err(|error| error.to_string())?;
                    ops::r#where_device(&origin, &zero, &domain, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Sin { input } => mlx_value(&values, *input)?
                    .sin_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Cos { input } => mlx_value(&values, *input)?
                    .cos_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Log { input } => mlx_value(&values, *input)?
                    .log_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Log1p { input } => mlx_value(&values, *input)?
                    .log1p_device(&stream)
                    .map_err(|error| error.to_string()),
                // MLX's own expm1 is off by up to several hundred float32 ulp, and
                // exp(x) - 1 cancels near zero. Kahan's (u - 1) * x / log(u) with
                // u = exp(x) keeps both within a few ulp: it is used for |x| < 0.5,
                // where exp(x) - 1 loses precision, and returns x where u rounds to
                // 1. Elsewhere exp(x) - 1 is accurate and handles +-inf, overflow
                // and NaN.
                TensorOp::Expm1 { input } => {
                    let input = mlx_value(&values, *input)?;
                    let fail = |error: mlx_rs::error::Exception| error.to_string();
                    let one = Array::from_f32(1.0);
                    let half = Array::from_f32(0.5);
                    let two = Array::from_f32(2.0);
                    let zero = Array::from_f32(0.0);
                    let exp = input.exp_device(&stream).map_err(fail)?;
                    let shifted = exp.subtract_device(&one, &stream).map_err(fail)?;
                    let small = input
                        .abs_device(&stream)
                        .and_then(|magnitude| magnitude.lt_device(&half, &stream))
                        .map_err(fail)?;
                    let exact = shifted.eq_device(&zero, &stream).map_err(fail)?;
                    let log = ops::r#where_device(&small, &exp, &two, &stream)
                        .and_then(|base| base.log_device(&stream))
                        .map_err(fail)?;
                    let safe_log =
                        ops::r#where_device(&exact, &one, &log, &stream).map_err(fail)?;
                    let kahan = shifted
                        .multiply_device(input, &stream)
                        .and_then(|product| product.divide_device(&safe_log, &stream))
                        .map_err(fail)?;
                    let near_zero =
                        ops::r#where_device(&exact, input, &kahan, &stream).map_err(fail)?;
                    ops::r#where_device(&small, &near_zero, &shifted, &stream).map_err(fail)
                }
                TensorOp::Erf { input } => ops::erf_device(mlx_value(&values, *input)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Erfc { input } => {
                    mlx_erfc(mlx_value(&values, *input)?, &stream).map_err(|error| error.to_string())
                }
                TensorOp::Atan2 { y, x } => {
                    ops::atan2_device(mlx_value(&values, *y)?, mlx_value(&values, *x)?, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::UnaryMath { input, kind } => {
                    mlx_unary_math(mlx_value(&values, *input)?, *kind, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Fmod { x, y } => {
                    mlx_fmod(mlx_value(&values, *x)?, mlx_value(&values, *y)?, &stream)
                        .map_err(|error| error.to_string())
                }
                // AD happens on the IR before lowering, so the value is all
                // that remains of a stop_gradient or a custom rule node (plan
                // compilation already aliases the latter to its value).
                TensorOp::StopGradient { input } | TensorOp::Custom { value: input, .. } => {
                    Ok(mlx_value(&values, *input)?.clone())
                }
                // MLX scans in parallel, so float32 prefix sums may differ
                // from the CPU's sequential rounded adds in the last bits.
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => mlx_value(&values, *input)?
                    .cumsum_device(
                        i32::try_from(*axis)
                            .map_err(|_| "MLX cumsum axis exceeds i32".to_string())?,
                        *reverse,
                        true,
                        &stream,
                    )
                    .map_err(|error| error.to_string()),
                TensorOp::Powi { input, exponent } => {
                    let exponent = Array::from_f32(*exponent as f32);
                    mlx_value(&values, *input)?
                        .power_device(&exponent, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Pow { base, exponent } => mlx_value(&values, *base)?
                    .power_device(mlx_value(&values, *exponent)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Matmul { lhs, rhs } => mlx_value(&values, *lhs)?
                    .matmul_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Cholesky { input } => cholesky_backend::evaluate(
                    &[mlx_value(&values, *input)?], &node.shape, None, &stream,
                ),
                TensorOp::CholeskyAd { inputs, kind } => {
                    let operands = inputs.iter().map(|input| mlx_value(&values, *input)).collect::<Result<Vec<_>, _>>()?;
                    cholesky_backend::evaluate(&operands, &node.shape, Some(*kind), &stream)
                }
                TensorOp::Solve { .. } => {
                    return Err(
                        "MLX GPU backend does not yet support solve: MLX linalg::solve only accepts a CPU stream"
                            .to_string(),
                    )
                }
                TensorOp::Linalg { kind, .. } => {
                    return Err(format!(
                        "MLX GPU backend does not support {}: MLX's LU, eigh, QR, and SVD \
                         factorizations only accept a CPU stream",
                        kind.name()
                    ))
                }
                TensorOp::Triangular { input, lower } => {
                    let input = mlx_value(&values, *input)?;
                    if *lower {
                        ops::tril_device(input, None, &stream)
                    } else {
                        ops::triu_device(input, None, &stream)
                    }
                    .map_err(|error| error.to_string())
                }
                TensorOp::Sum { input } => mlx_value(&values, *input)?
                    .sum_device(None, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Mean { input } => mlx_value(&values, *input)?
                    .mean_device(None, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::SumAxis { input, axis } => mlx_value(&values, *input)?
                    .sum_axis_device(
                        i32::try_from(*axis).map_err(|_| "MLX sum axis exceeds i32".to_string())?,
                        None,
                        &stream,
                    )
                    .map_err(|error| error.to_string()),
                TensorOp::MeanAxis { input, axis } => mlx_value(&values, *input)?
                    .mean_axis_device(
                        i32::try_from(*axis)
                            .map_err(|_| "MLX mean axis exceeds i32".to_string())?,
                        None,
                        &stream,
                    )
                    .map_err(|error| error.to_string()),
                TensorOp::Reshape { input } => mlx_value(&values, *input)?
                    .reshape_device(&mlx_shape(&node.shape)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Transpose { input, axes } => {
                    let axes = axes
                        .iter()
                        .map(|axis| {
                            i32::try_from(*axis)
                                .map_err(|_| "MLX transpose axis exceeds i32".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    mlx_value(&values, *input)?
                        .transpose_axes_device(&axes, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Concat { inputs, axis } => {
                    let axis = i32::try_from(*axis)
                        .map_err(|_| "MLX concat axis exceeds i32".to_string())?;
                    let arrays = inputs
                        .iter()
                        .map(|input| mlx_value(&values, *input))
                        .collect::<Result<Vec<_>, _>>()?;
                    ops::concatenate_axis_device(&arrays, axis, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Broadcast { input } => ops::broadcast_to_device(
                    mlx_value(&values, *input)?,
                    &mlx_shape(&node.shape)?,
                    &stream,
                )
                .map_err(|error| error.to_string()),
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => {
                    let start = i32::try_from(*start)
                        .map_err(|_| "MLX slice start exceeds i32".to_string())?;
                    let length = i32::try_from(*length)
                        .map_err(|_| "MLX slice length exceeds i32".to_string())?;
                    let indices = (start
                        ..start
                            .checked_add(length)
                            .ok_or_else(|| "MLX slice index range overflows i32".to_string())?)
                        .collect::<Vec<_>>();
                    let indices = Array::from_slice(&indices, &[length]);
                    mlx_value(&values, *input)?
                        .take_axis_device(
                            &indices,
                            i32::try_from(*axis)
                                .map_err(|_| "MLX slice axis exceeds i32".to_string())?,
                            &stream,
                        )
                        .map_err(|error| error.to_string())
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let rank = node.shape.len();
                    let mut widths = vec![(0_i32, 0_i32); rank];
                    let before = i32::try_from(*start)
                        .map_err(|_| "MLX pad start exceeds i32".to_string())?;
                    let after = node.shape[*axis]
                        .checked_sub(start + plan.nodes[*input].shape[*axis])
                        .ok_or_else(|| "MLX pad extent underflows".to_string())?;
                    widths[*axis] = (
                        before,
                        i32::try_from(after)
                            .map_err(|_| "MLX pad extent exceeds i32".to_string())?,
                    );
                    ops::pad_device(
                        mlx_value(&values, *input)?,
                        widths.as_slice(),
                        None,
                        None,
                        &stream,
                    )
                    .map_err(|error| error.to_string())
                }
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => {
                    let indices = indices
                        .iter()
                        .map(|index| {
                            i32::try_from(*index)
                                .map_err(|_| "MLX gather index exceeds i32".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let count = i32::try_from(indices.len())
                        .map_err(|_| "MLX gather index count exceeds i32".to_string())?;
                    mlx_value(&values, *input)?
                        .take_axis_device(
                            Array::from_slice(&indices, &[count]),
                            i32::try_from(*axis)
                                .map_err(|_| "MLX gather axis exceeds i32".to_string())?,
                            &stream,
                        )
                        .map_err(|error| error.to_string())
                }
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => scatter_backend::evaluate(
                    mlx_value(&values, *base)?,
                    mlx_value(&values, *updates)?,
                    indices,
                    *axis,
                    &node.shape,
                    &stream,
                ),
            }
            .map_err(|error| {
                format!(
                    "MLX node {node_id} {} failed: {error}",
                    mlx_op_name(&node.op)
                )
            })?;
            values.push(value);
        }

        output_node_ids
            .iter()
            .map(|node_id| mlx_value(&values, *node_id).cloned())
            .collect()
    }

    /// Executes a graph using the backend-owned retained input state.
    pub fn execute_many_with_state(
        &self,
        plan: &TensorExecutionPlan,
        output_node_ids: &[usize],
        inputs: &BTreeMap<String, DynamicTensor>,
        state: &MlxRetainedInputs,
    ) -> Result<Vec<DynamicTensor>, String> {
        self.execute_many_with_retained(plan, output_node_ids, inputs, state.arrays())
    }
}

impl TensorBackend for MlxBackend {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.execute_many(plan, &[plan.output_node_id], inputs)
            .and_then(|mut outputs| {
                outputs
                    .pop()
                    .ok_or_else(|| "MLX execution produced no output".to_string())
            })
    }
}

fn mlx_fori_value_and_vjp(
    backend: &MlxBackend,
    loop_plan: &super::TensorForiExecutionPlan,
    initial_carry: Array,
    external_captures: &BTreeMap<String, Array>,
    output_cotangent: Array,
) -> Result<MlxForiVjpEvaluation, String> {
    let vjp = loop_plan.mlx_vjp.as_ref().ok_or_else(|| {
        "MLX Fori VJP requires a body with supported symbolic reverse lowering".to_string()
    })?;
    let stream = StreamOrDevice::gpu();
    let steps = loop_plan.upper - loop_plan.lower;
    let evaluate = |offset, carry| {
        let index = loop_plan.lower + offset;
        let mut body_inputs = external_captures.clone();
        body_inputs.insert(loop_plan.carry_name.clone(), carry);
        body_inputs.insert(
            loop_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(vec![], index as f64)?)?,
        );
        let mut outputs = backend.execute_arrays_with_retained(
            &loop_plan.body.plan,
            &[loop_plan.body.plan.output_node_id],
            &BTreeMap::new(),
            &body_inputs,
        )?;
        let carry = outputs
            .pop()
            .ok_or_else(|| "MLX Fori forward body produced no carry output".to_string())?;
        Ok(carry)
    };
    let mut checkpoints = None;
    let mut full_tape = None;
    if let Some(block) = super::TensorCarryCheckpoints::<Array>::block_size(steps) {
        let (_, tape) =
            super::TensorCarryCheckpoints::forward(initial_carry, steps, block, evaluate)?;
        checkpoints = Some(tape);
    } else {
        let mut carry = initial_carry;
        let mut carries = Vec::with_capacity(steps);
        for offset in 0..steps {
            carries.push(carry.clone());
            carry = evaluate(offset, carry)?;
        }
        full_tape = Some(carries);
    }

    let mut carry_gradient = output_cotangent;
    let mut external_gradients = loop_plan
        .external_captures
        .iter()
        .map(|(name, shape)| {
            Array::zeros_device::<f32>(&mlx_shape(shape)?, &stream)
                .map_err(|error| error.to_string())
                .map(|value| (name.clone(), value))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let output_ids = vjp.gradient_node_ids.values().copied().collect::<Vec<_>>();
    loop {
        let block = if let Some(checkpoints) = &mut checkpoints {
            checkpoints.pop_block(evaluate)?
        } else {
            full_tape.take().map(|states| (0, states))
        };
        let Some((start, carries)) = block else {
            break;
        };
        for (local_offset, carry) in carries.into_iter().enumerate().rev() {
            let offset = start + local_offset;
            let mut body_inputs = external_captures.clone();
            body_inputs.insert(loop_plan.carry_name.clone(), carry);
            body_inputs.insert(
                loop_plan.index_name.clone(),
                mlx_array_from_dynamic(&DynamicTensor::filled(
                    vec![],
                    (loop_plan.lower + offset) as f64,
                )?)?,
            );
            body_inputs.insert(vjp.cotangent_name.clone(), carry_gradient);
            let gradients = backend.execute_arrays_with_retained(
                &vjp.plan,
                &output_ids,
                &BTreeMap::new(),
                &body_inputs,
            )?;
            let gradients = vjp
                .gradient_node_ids
                .keys()
                .cloned()
                .zip(gradients)
                .collect::<BTreeMap<_, _>>();
            carry_gradient = gradients
                .get(&loop_plan.carry_name)
                .cloned()
                .ok_or_else(|| "MLX Fori VJP body has no carry gradient".to_string())?;
            for name in loop_plan.external_captures.keys() {
                let contribution = gradients
                    .get(name)
                    .ok_or_else(|| format!("MLX Fori VJP has no gradient for capture {name:?}"))?;
                let accumulated = external_gradients
                    .get_mut(name)
                    .ok_or_else(|| format!("MLX Fori VJP gradient {name:?} is missing"))?;
                *accumulated = accumulated
                    .add_device(contribution, &stream)
                    .map_err(|error| error.to_string())?;
            }
        }
        if checkpoints.is_some() {
            // Bound lazy accumulation history to the replay block without changing sums.
            transforms::eval(std::iter::once(&carry_gradient).chain(external_gradients.values()))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(MlxForiVjpEvaluation {
        carry_gradient,
        external_gradients,
    })
}

#[allow(clippy::too_many_arguments)]
fn mlx_fori_jvp(
    backend: &MlxBackend,
    loop_plan: &TensorForiExecutionPlan,
    initial_carry: Array,
    initial_tangent: Array,
    external_captures: &BTreeMap<String, Array>,
    external_tangents: &BTreeMap<String, Array>,
) -> Result<Array, String> {
    let body = loop_plan.body.plan.as_ir();
    let mut tangent_names = BTreeMap::new();
    for (index, name) in loop_plan.body.captures.keys().enumerate() {
        if name == &loop_plan.index_name {
            continue;
        }
        let mut tangent_name = format!("__quabla_mlx_fori_jvp_tangent_{index}");
        while loop_plan.body.captures.contains_key(&tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &tangent_name)
        {
            tangent_name.push('_');
        }
        tangent_names.insert(name.clone(), tangent_name);
    }
    let forward = body.symbolic_jvp_with_seed(
        loop_plan.body.plan.output_node_id,
        |graph, name, value, shape| {
            tangent_names
                .get(name)
                .map(|tangent_name| {
                    let dtype = graph.node_dtype(value)?;
                    graph.input_typed(tangent_name.clone(), shape.to_vec(), dtype)
                })
                .transpose()
        },
    )?;
    let (forward_plan, outputs) = forward
        .graph
        .compile_cpu_many(&[forward.value, forward.tangent])?;
    let mut carry = initial_carry;
    let mut carry_tangent = initial_tangent;
    for index in loop_plan.lower..loop_plan.upper {
        let mut inputs = external_captures.clone();
        inputs.insert(loop_plan.carry_name.clone(), carry);
        inputs.insert(
            loop_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(vec![], index as f64)?)?,
        );
        for (name, tangent_name) in &tangent_names {
            let tangent = if name == &loop_plan.carry_name {
                carry_tangent.clone()
            } else {
                external_tangents
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("MLX Fori JVP lacks tangent capture {name:?}"))?
            };
            inputs.insert(tangent_name.clone(), tangent);
        }
        let outputs = backend.execute_arrays_with_retained(
            &forward_plan,
            &outputs,
            &BTreeMap::new(),
            &inputs,
        )?;
        carry = outputs
            .first()
            .cloned()
            .ok_or_else(|| "MLX Fori JVP forward body has no primal output".to_string())?;
        carry_tangent = outputs
            .get(1)
            .cloned()
            .ok_or_else(|| "MLX Fori JVP forward body has no tangent output".to_string())?;
    }
    Ok(carry_tangent)
}

#[allow(clippy::too_many_arguments)]
fn mlx_fori_vjp_jvp(
    backend: &MlxBackend,
    plan: &super::TensorForiVjpJvpExecutionPlan,
    initial_carry: Array,
    initial_tangent: Array,
    external_captures: &BTreeMap<String, Array>,
    external_tangents: &BTreeMap<String, Array>,
    output_cotangent: Array,
    output_cotangent_tangent: Array,
) -> Result<BTreeMap<String, Array>, String> {
    let stream = StreamOrDevice::gpu();
    let loop_plan = &plan.loop_plan;
    let steps = loop_plan.upper - loop_plan.lower;
    let evaluate = |offset, (carry, tangent): (Array, Array)| {
        let mut inputs = external_captures.clone();
        inputs.insert(loop_plan.carry_name.clone(), carry);
        inputs.insert(
            loop_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(
                vec![],
                (loop_plan.lower + offset) as f64,
            )?)?,
        );
        for (name, tangent_name) in &plan.mlx_forward_jvp.tangent_names {
            let value = if name == &loop_plan.carry_name {
                tangent.clone()
            } else {
                external_tangents
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("MLX Fori JVP lacks tangent capture {name:?}"))?
            };
            inputs.insert(tangent_name.clone(), value);
        }
        let values = backend.execute_arrays_with_retained(
            &plan.mlx_forward_jvp.plan,
            &[
                plan.mlx_forward_jvp.value_node_id,
                plan.mlx_forward_jvp.tangent_node_id,
            ],
            &BTreeMap::new(),
            &inputs,
        )?;
        let carry = values
            .first()
            .cloned()
            .ok_or_else(|| "MLX Fori JVP forward body has no primal output".to_string())?;
        let tangent = values
            .get(1)
            .cloned()
            .ok_or_else(|| "MLX Fori JVP forward body has no tangent output".to_string())?;
        Ok((carry, tangent))
    };
    let mut checkpoints = None;
    let mut full_tape = None;
    let initial = (initial_carry, initial_tangent);
    if let Some(block) = super::TensorCarryCheckpoints::<Array>::block_size(steps) {
        let (_, tape) = super::TensorCarryCheckpoints::forward(initial, steps, block, evaluate)?;
        checkpoints = Some(tape);
    } else {
        let mut state = initial;
        let mut states = Vec::with_capacity(steps);
        for offset in 0..steps {
            states.push(state.clone());
            state = evaluate(offset, state)?;
        }
        full_tape = Some(states);
    }

    let mut carry_cotangent = output_cotangent;
    let mut carry_cotangent_tangent = output_cotangent_tangent;
    let mut gradients = loop_plan
        .external_captures
        .iter()
        .map(|(name, shape)| {
            Array::zeros_device::<f32>(&mlx_shape(shape)?, &stream)
                .map_err(|error| error.to_string())
                .map(|value| (name.clone(), value))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    loop {
        let block = if let Some(checkpoints) = &mut checkpoints {
            checkpoints.pop_block(evaluate)?
        } else {
            full_tape.take().map(|states| (0, states))
        };
        let Some((start, states)) = block else {
            break;
        };
        for (local_offset, (carry, tangent)) in states.into_iter().enumerate().rev() {
            let offset = start + local_offset;
            let mut inputs = external_captures.clone();
            inputs.insert(loop_plan.carry_name.clone(), carry);
            inputs.insert(
                loop_plan.index_name.clone(),
                mlx_array_from_dynamic(&DynamicTensor::filled(
                    vec![],
                    (loop_plan.lower + offset) as f64,
                )?)?,
            );
            let body_gradients = mlx_region_vjp_gradients(
                backend,
                loop_plan.mlx_vjp.as_ref().ok_or_else(|| {
                    "MLX Fori VJP JVP requires a body with supported symbolic reverse lowering"
                        .to_string()
                })?,
                &inputs,
                carry_cotangent.clone(),
            )?;
            let mut jvp_inputs = inputs;
            jvp_inputs.insert(plan.cotangent_name.clone(), carry_cotangent);
            for (name, tangent_name) in &plan.tangent_names {
                let tangent = if name == &loop_plan.carry_name {
                    tangent.clone()
                } else if name == &plan.cotangent_name {
                    carry_cotangent_tangent.clone()
                } else {
                    external_tangents
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("MLX Fori VJP JVP lacks tangent capture {name:?}"))?
                };
                jvp_inputs.insert(tangent_name.clone(), tangent);
            }
            let next_carry_cotangent_tangent = mlx_execute_plan_output(
                backend,
                plan.gradient_tangent_plans
                    .get(&loop_plan.carry_name)
                    .ok_or_else(|| "MLX Fori VJP JVP has no carry gradient plan".to_string())?,
                &jvp_inputs,
            )?;
            for name in loop_plan.external_captures.keys() {
                let contribution = mlx_execute_plan_output(
                    backend,
                    plan.gradient_tangent_plans.get(name).ok_or_else(|| {
                        format!("MLX Fori VJP JVP has no gradient plan for {name:?}")
                    })?,
                    &jvp_inputs,
                )?;
                let accumulated = gradients
                    .get_mut(name)
                    .ok_or_else(|| format!("MLX Fori VJP JVP gradient {name:?} is missing"))?;
                *accumulated = accumulated
                    .add_device(&contribution, &stream)
                    .map_err(|error| error.to_string())?;
            }
            carry_cotangent = body_gradients
                .get(&loop_plan.carry_name)
                .cloned()
                .ok_or_else(|| "MLX Fori VJP JVP body has no carry gradient".to_string())?;
            carry_cotangent_tangent = next_carry_cotangent_tangent;
        }
        if checkpoints.is_some() {
            transforms::eval(
                [&carry_cotangent, &carry_cotangent_tangent]
                    .into_iter()
                    .chain(gradients.values()),
            )
            .map_err(|error| error.to_string())?;
        }
    }
    gradients.insert(loop_plan.carry_name.clone(), carry_cotangent_tangent);
    Ok(gradients)
}

fn mlx_execute_plan_output(
    backend: &MlxBackend,
    plan: &super::TensorExecutionPlan,
    inputs: &BTreeMap<String, Array>,
) -> Result<Array, String> {
    backend
        .execute_arrays_with_retained(plan, &[plan.output_node_id], &BTreeMap::new(), inputs)
        .and_then(|mut outputs| {
            outputs
                .pop()
                .ok_or_else(|| "MLX execution plan produced no output".to_string())
        })
}

#[allow(clippy::too_many_arguments)]
fn mlx_scan_vjp_jvp(
    backend: &MlxBackend,
    plan: &super::TensorScanVjpJvpExecutionPlan,
    initial_carry: Array,
    initial_tangent: Array,
    external_captures: &BTreeMap<String, Array>,
    external_tangents: &BTreeMap<String, Array>,
    final_carry_cotangent: Array,
    final_carry_cotangent_tangent: Array,
    output_cotangent: Array,
    output_cotangent_tangent: Array,
) -> Result<BTreeMap<String, Array>, String> {
    let stream = StreamOrDevice::gpu();
    let scan_plan = &plan.scan_plan;
    let steps = scan_plan.upper - scan_plan.lower;
    let evaluate = |offset, (carry, tangent): (Array, Array)| {
        let mut inputs = external_captures.clone();
        inputs.insert(scan_plan.carry_name.clone(), carry);
        inputs.insert(
            scan_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(
                vec![],
                (scan_plan.lower + offset) as f64,
            )?)?,
        );
        for (name, tangent_name) in &plan.mlx_forward_jvp.tangent_names {
            let value = if name == &scan_plan.carry_name {
                tangent.clone()
            } else {
                external_tangents
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("MLX Scan JVP lacks tangent capture {name:?}"))?
            };
            inputs.insert(tangent_name.clone(), value);
        }
        let values = backend.execute_arrays_with_retained(
            &plan.mlx_forward_jvp.plan,
            &[
                plan.mlx_forward_jvp.value_node_id,
                plan.mlx_forward_jvp.tangent_node_id,
            ],
            &BTreeMap::new(),
            &inputs,
        )?;
        let carry = values
            .first()
            .cloned()
            .ok_or_else(|| "MLX Scan JVP forward body has no primal output".to_string())?;
        let tangent = values
            .get(1)
            .cloned()
            .ok_or_else(|| "MLX Scan JVP forward body has no tangent output".to_string())?;
        Ok((carry, tangent))
    };
    let mut checkpoints = None;
    let mut full_tape = None;
    let initial = (initial_carry, initial_tangent);
    if let Some(block) = super::TensorCarryCheckpoints::<Array>::block_size(steps) {
        let (_, tape) = super::TensorCarryCheckpoints::forward(initial, steps, block, evaluate)?;
        checkpoints = Some(tape);
    } else {
        let mut state = initial;
        let mut states = Vec::with_capacity(steps);
        for offset in 0..steps {
            states.push(state.clone());
            state = evaluate(offset, state)?;
        }
        full_tape = Some(states);
    }

    let vjp = scan_plan.mlx_vjp.as_ref().ok_or_else(|| {
        "MLX Scan VJP JVP requires body outputs with supported symbolic reverse lowering"
            .to_string()
    })?;
    let output_step_shape = scan_plan.body.output_shapes()[1].clone();
    let mut carry_cotangent = final_carry_cotangent;
    let mut carry_cotangent_tangent = final_carry_cotangent_tangent;
    let mut gradients = scan_plan
        .external_captures
        .iter()
        .map(|(name, shape)| {
            Array::zeros_device::<f32>(&mlx_shape(shape)?, &stream)
                .map_err(|error| error.to_string())
                .map(|value| (name.clone(), value))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    loop {
        let block = if let Some(checkpoints) = &mut checkpoints {
            checkpoints.pop_block(evaluate)?
        } else {
            full_tape.take().map(|states| (0, states))
        };
        let Some((start, states)) = block else {
            break;
        };
        for (local_offset, (carry, tangent)) in states.into_iter().enumerate().rev() {
            let offset = start + local_offset;
            let mut inputs = external_captures.clone();
            inputs.insert(scan_plan.carry_name.clone(), carry);
            inputs.insert(
                scan_plan.index_name.clone(),
                mlx_array_from_dynamic(&DynamicTensor::filled(
                    vec![],
                    (scan_plan.lower + offset) as f64,
                )?)?,
            );
            let output_step_cotangent = mlx_scan_output_cotangent_at(
                &output_cotangent,
                offset,
                &output_step_shape,
                &stream,
            )?;
            let output_step_cotangent_tangent = mlx_scan_output_cotangent_at(
                &output_cotangent_tangent,
                offset,
                &output_step_shape,
                &stream,
            )?;
            let carry_gradients =
                mlx_region_vjp_gradients(backend, &vjp.carry, &inputs, carry_cotangent.clone())?;
            let output_gradients = mlx_region_vjp_gradients(
                backend,
                &vjp.output,
                &inputs,
                output_step_cotangent.clone(),
            )?;
            let mut jvp_inputs = inputs;
            jvp_inputs.insert(plan.carry_cotangent_name.clone(), carry_cotangent);
            jvp_inputs.insert(plan.output_cotangent_name.clone(), output_step_cotangent);
            for (name, tangent_name) in &plan.tangent_names {
                let tangent = if name == &scan_plan.carry_name {
                    tangent.clone()
                } else if name == &plan.carry_cotangent_name {
                    carry_cotangent_tangent.clone()
                } else if name == &plan.output_cotangent_name {
                    output_step_cotangent_tangent.clone()
                } else {
                    external_tangents
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("MLX Scan VJP JVP lacks tangent capture {name:?}"))?
                };
                jvp_inputs.insert(tangent_name.clone(), tangent);
            }
            let directional_gradient = |name: &str| -> Result<Array, String> {
                let carry_term = mlx_execute_plan_output(
                    backend,
                    plan.carry_gradient_tangent_plans.get(name).ok_or_else(|| {
                        format!("MLX Scan VJP JVP has no carry gradient plan for {name:?}")
                    })?,
                    &jvp_inputs,
                )?;
                let output_term = mlx_execute_plan_output(
                    backend,
                    plan.output_gradient_tangent_plans
                        .get(name)
                        .ok_or_else(|| {
                            format!("MLX Scan VJP JVP has no output gradient plan for {name:?}")
                        })?,
                    &jvp_inputs,
                )?;
                carry_term
                    .add_device(&output_term, &stream)
                    .map_err(|error| error.to_string())
            };
            carry_cotangent_tangent = directional_gradient(&scan_plan.carry_name)?;
            for name in scan_plan.external_captures.keys() {
                let contribution = directional_gradient(name)?;
                let accumulated = gradients
                    .get_mut(name)
                    .ok_or_else(|| format!("MLX Scan VJP JVP gradient {name:?} is missing"))?;
                *accumulated = accumulated
                    .add_device(&contribution, &stream)
                    .map_err(|error| error.to_string())?;
            }
            carry_cotangent = carry_gradients
                .get(&scan_plan.carry_name)
                .ok_or_else(|| "MLX Scan carry VJP has no carry gradient".to_string())?
                .add_device(
                    output_gradients
                        .get(&scan_plan.carry_name)
                        .ok_or_else(|| "MLX Scan output VJP has no carry gradient".to_string())?,
                    &stream,
                )
                .map_err(|error| error.to_string())?;
        }
        if checkpoints.is_some() {
            transforms::eval(
                [&carry_cotangent, &carry_cotangent_tangent]
                    .into_iter()
                    .chain(gradients.values()),
            )
            .map_err(|error| error.to_string())?;
        }
    }
    gradients.insert(scan_plan.carry_name.clone(), carry_cotangent_tangent);
    Ok(gradients)
}

fn mlx_scan_value_and_vjp(
    backend: &MlxBackend,
    scan_plan: &super::TensorScanExecutionPlan,
    initial_carry: Array,
    external_captures: &BTreeMap<String, Array>,
    final_carry_cotangent: Array,
    output_cotangent: Array,
) -> Result<MlxScanVjpEvaluation, String> {
    let vjp = scan_plan.mlx_vjp.as_ref().ok_or_else(|| {
        "MLX Scan VJP requires body outputs with supported symbolic reverse lowering".to_string()
    })?;
    let stream = StreamOrDevice::gpu();
    let steps = scan_plan.upper - scan_plan.lower;
    let evaluate = |offset, carry| {
        let index = scan_plan.lower + offset;
        let mut body_inputs = external_captures.clone();
        body_inputs.insert(scan_plan.carry_name.clone(), carry);
        body_inputs.insert(
            scan_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(vec![], index as f64)?)?,
        );
        let values = backend.execute_arrays_with_retained(
            &scan_plan.body.plan,
            &scan_plan.body.plan.output_node_ids,
            &BTreeMap::new(),
            &body_inputs,
        )?;
        let carry = values
            .first()
            .cloned()
            .ok_or_else(|| "MLX Scan forward body has no carry output".to_string())?;
        Ok(carry)
    };
    let mut checkpoints = None;
    let mut full_tape = None;
    if let Some(block) = super::TensorCarryCheckpoints::<Array>::block_size(steps) {
        let (_, tape) =
            super::TensorCarryCheckpoints::forward(initial_carry, steps, block, evaluate)?;
        checkpoints = Some(tape);
    } else {
        let mut carry = initial_carry;
        let mut carries = Vec::with_capacity(steps);
        for offset in 0..steps {
            carries.push(carry.clone());
            carry = evaluate(offset, carry)?;
        }
        full_tape = Some(carries);
    }

    let mut carry_gradient = final_carry_cotangent;
    let mut external_gradients = scan_plan
        .external_captures
        .iter()
        .map(|(name, shape)| {
            Array::zeros_device::<f32>(&mlx_shape(shape)?, &stream)
                .map_err(|error| error.to_string())
                .map(|value| (name.clone(), value))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let output_step_shape = scan_plan.body.output_shapes()[1].clone();
    loop {
        let block = if let Some(checkpoints) = &mut checkpoints {
            checkpoints.pop_block(evaluate)?
        } else {
            full_tape.take().map(|states| (0, states))
        };
        let Some((start, carries)) = block else {
            break;
        };
        for (local_offset, carry) in carries.into_iter().enumerate().rev() {
            let offset = start + local_offset;
            let mut body_inputs = external_captures.clone();
            body_inputs.insert(scan_plan.carry_name.clone(), carry);
            body_inputs.insert(
                scan_plan.index_name.clone(),
                mlx_array_from_dynamic(&DynamicTensor::filled(
                    vec![],
                    (scan_plan.lower + offset) as f64,
                )?)?,
            );
            let output_gradient = mlx_scan_output_cotangent_at(
                &output_cotangent,
                offset,
                &output_step_shape,
                &stream,
            )?;
            let carry_gradients =
                mlx_region_vjp_gradients(backend, &vjp.carry, &body_inputs, carry_gradient)?;
            let output_gradients =
                mlx_region_vjp_gradients(backend, &vjp.output, &body_inputs, output_gradient)?;
            let carry_from_carry = carry_gradients
                .get(&scan_plan.carry_name)
                .ok_or_else(|| "MLX Scan carry VJP has no carry gradient".to_string())?;
            let carry_from_output = output_gradients
                .get(&scan_plan.carry_name)
                .ok_or_else(|| "MLX Scan output VJP has no carry gradient".to_string())?;
            carry_gradient = carry_from_carry
                .add_device(carry_from_output, &stream)
                .map_err(|error| error.to_string())?;
            for name in scan_plan.external_captures.keys() {
                let carry_contribution = carry_gradients
                    .get(name)
                    .ok_or_else(|| format!("MLX Scan carry VJP has no gradient for {name:?}"))?;
                let output_contribution = output_gradients
                    .get(name)
                    .ok_or_else(|| format!("MLX Scan output VJP has no gradient for {name:?}"))?;
                let contribution = carry_contribution
                    .add_device(output_contribution, &stream)
                    .map_err(|error| error.to_string())?;
                let accumulated = external_gradients
                    .get_mut(name)
                    .ok_or_else(|| format!("MLX Scan VJP gradient {name:?} is missing"))?;
                *accumulated = accumulated
                    .add_device(&contribution, &stream)
                    .map_err(|error| error.to_string())?;
            }
        }
        if checkpoints.is_some() {
            // Bound lazy accumulation history to the replay block without changing sums.
            transforms::eval(std::iter::once(&carry_gradient).chain(external_gradients.values()))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(MlxScanVjpEvaluation {
        carry_gradient,
        external_gradients,
    })
}

fn mlx_region_vjp_gradients(
    backend: &MlxBackend,
    vjp: &super::TensorMlxVjpPlan,
    body_inputs: &BTreeMap<String, Array>,
    cotangent: Array,
) -> Result<BTreeMap<String, Array>, String> {
    let mut inputs = body_inputs.clone();
    inputs.insert(vjp.cotangent_name.clone(), cotangent);
    let output_ids = vjp.gradient_node_ids.values().copied().collect::<Vec<_>>();
    let values =
        backend.execute_arrays_with_retained(&vjp.plan, &output_ids, &BTreeMap::new(), &inputs)?;
    Ok(vjp.gradient_node_ids.keys().cloned().zip(values).collect())
}

fn mlx_scan_output_cotangent_at(
    output_cotangent: &Array,
    offset: usize,
    step_shape: &[usize],
    stream: &StreamOrDevice,
) -> Result<Array, String> {
    let offset = i32::try_from(offset)
        .map_err(|_| "MLX Scan output cotangent offset exceeds i32".to_string())?;
    let indices = Array::from_slice(&[offset], &[1]);
    output_cotangent
        .take_axis_device(&indices, 0, stream)
        .map_err(|error| error.to_string())?
        .reshape_device(&mlx_shape(step_shape)?, stream)
        .map_err(|error| error.to_string())
}

fn mlx_shape(shape: &[usize]) -> Result<Vec<i32>, String> {
    shape
        .iter()
        .map(|extent| {
            i32::try_from(*extent).map_err(|_| "MLX shape extent exceeds i32".to_string())
        })
        .collect()
}

impl TensorConstant {
    /// The MLX array of this constant: created and evaluated by the first MLX
    /// execution that needs it, then shared by every later execution of any
    /// plan holding the constant, so its data crosses to MLX once. The caller
    /// holds the MLX execution lock, which also serializes this cache.
    fn mlx_array(&self) -> Result<Array, String> {
        let mut cached = self
            .0
            .mlx_array
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(array) = cached.as_ref() {
            return Ok(array.clone());
        }
        let array = mlx_array_from_dynamic(self.value())?;
        transforms::eval([&array])
            .map_err(|error| format!("MLX constant upload failed: {error}"))?;
        *cached = Some(array.clone());
        Ok(array)
    }
}

fn mlx_array_from_dynamic(input: &DynamicTensor) -> Result<Array, String> {
    let shape = mlx_shape(input.shape())?;
    let data = input.storage().to_f32();
    let stream = StreamOrDevice::gpu();
    let host_value = Array::from_slice(&data, &shape);
    let zeros = Array::zeros_device::<f32>(&shape, &stream).map_err(|error| error.to_string())?;
    host_value
        .add_device(&zeros, &stream)
        .map_err(|error| error.to_string())
}

/// Returns `array` with a row-major layout, as host readback requires.
///
/// Transposes, broadcasts, and axis-moving gathers are strided views in MLX, and elementwise
/// kernels keep a column-major input layout for their output, so a valid program can produce an
/// array whose storage order differs from its logical row-major order. `contiguous` copies such
/// an array and shares the buffer of one that is already row-major.
fn mlx_row_major(array: &Array, stream: &StreamOrDevice) -> Result<Array, String> {
    mlx_rs::with_stream(stream.as_ref(), || array.contiguous())
        .map_err(|error| format!("MLX row-major copy failed: {error}"))
}

/// Copies an evaluated row-major f32 array to the host.
///
/// Callers pass the result of [`mlx_row_major`]; `try_as_slice` still rejects any other layout
/// instead of returning elements in storage order.
fn mlx_host_values(array: &Array) -> Result<Vec<f32>, String> {
    array
        .try_as_slice::<f32>()
        .map(|values| values.to_vec())
        .map_err(|error| format!("readback failed: {error}"))
}

/// Reads back the scalar predicate of a `Cond` (evaluating and synchronizing the GPU stream).
///
/// Same semantics as the CPU `tensor_scalar_predicate`: non-finite values are rejected and non-zero
/// is true.
fn mlx_scalar_predicate(predicate: &Array) -> Result<bool, String> {
    let value = predicate
        .try_item_cast::<f32>()
        .map_err(|error| format!("MLX Cond predicate readback failed: {error}"))?;
    if !value.is_finite() {
        return Err("conditional predicate must be finite".to_string());
    }
    Ok(value != 0.0)
}

/// Coefficients `c0..c9` of the Chebyshev-fitted `erfc` approximation of
/// Press et al., Numerical Recipes (2nd ed., section 6.2):
/// `erfc(z) ~= t exp(-z^2 + c0 + t (c1 + t (c2 + ... + t c9)))` with
/// `t = 1 / (1 + z / 2)` for `z >= 0`, relative error below `1.2e-7` for
/// every `z`.
const MLX_ERFC_COEFFICIENTS: [f32; 10] = [
    -1.265_512_2,
    1.000_023_7,
    0.374_091_96,
    0.096_784_18,
    -0.186_288_06,
    0.278_868_07,
    -1.135_204,
    1.488_515_9,
    -0.822_152_23,
    0.170_872_77,
];

/// `erfc(x)` on MLX, which has no `erfc` of its own; `1 - erf(x)` would lose
/// all relative accuracy for large `x`. Evaluates the approximation of
/// [`MLX_ERFC_COEFFICIENTS`] at `z = min(|x|, 16)` (`erfc(16)` underflows
/// `f32`, and the clamp keeps `inf` out of the arithmetic) and reflects
/// `erfc(x) = 2 - erfc(-x)` for negative `x`. The Gaussian factor is split
/// as `exp(-h^2) exp(-(z - h)(z + h))` with `h = floor(16 z) / 16`, whose
/// square is exact in `f32`, so the rounding of `z^2` (relative error
/// `eps z^2`) does not reach the result.
fn mlx_erfc(input: &Array, stream: &StreamOrDevice) -> Result<Array, mlx_rs::error::Exception> {
    let scalar = Array::from_f32;
    let z = ops::minimum_device(input.abs_device(stream)?, scalar(16.0), stream)?;
    let t = z
        .multiply_device(scalar(0.5), stream)?
        .add_device(scalar(1.0), stream)?
        .reciprocal_device(stream)?;
    let (&last, rest) = MLX_ERFC_COEFFICIENTS
        .split_last()
        .expect("the coefficient table is non-empty");
    let mut polynomial = scalar(last);
    for &coefficient in rest.iter().rev() {
        polynomial = polynomial
            .multiply_device(&t, stream)?
            .add_device(scalar(coefficient), stream)?;
    }
    let head = z
        .multiply_device(scalar(16.0), stream)?
        .floor_device(stream)?
        .multiply_device(scalar(0.0625), stream)?;
    let head_decay = head
        .square_device(stream)?
        .negative_device(stream)?
        .exp_device(stream)?;
    let tail = z
        .subtract_device(&head, stream)?
        .multiply_device(z.add_device(&head, stream)?, stream)?;
    let tail_decay = polynomial
        .subtract_device(&tail, stream)?
        .exp_device(stream)?;
    let positive = t
        .multiply_device(&head_decay, stream)?
        .multiply_device(&tail_decay, stream)?;
    let reflected = scalar(2.0).subtract_device(&positive, stream)?;
    let negative = input.lt_device(scalar(0.0), stream)?;
    ops::r#where_device(&negative, &reflected, &positive, stream)
}

/// `kind(input)` with MLX's own op for every kind except `cbrt`, which MLX
/// lacks (see [`mlx_cbrt`]). The MLX kernels are compiled without fast math
/// and call Metal's `precise` transcendental functions; `round` is Metal's
/// `rint`, halfway cases to even.
fn mlx_unary_math(
    input: &Array,
    kind: UnaryMathKind,
    stream: &StreamOrDevice,
) -> Result<Array, mlx_rs::error::Exception> {
    match kind {
        UnaryMathKind::Tan => ops::tan_device(input, stream),
        UnaryMathKind::Arcsin => ops::asin_device(input, stream),
        UnaryMathKind::Arccos => ops::acos_device(input, stream),
        UnaryMathKind::Arctan => ops::atan_device(input, stream),
        UnaryMathKind::Sinh => ops::sinh_device(input, stream),
        UnaryMathKind::Cosh => ops::cosh_device(input, stream),
        UnaryMathKind::Arcsinh => ops::asinh_device(input, stream),
        UnaryMathKind::Arccosh => ops::acosh_device(input, stream),
        UnaryMathKind::Arctanh => ops::atanh_device(input, stream),
        UnaryMathKind::Log2 => input.log2_device(stream),
        UnaryMathKind::Log10 => input.log10_device(stream),
        UnaryMathKind::Cbrt => mlx_cbrt(input, stream),
        UnaryMathKind::Floor => input.floor_device(stream),
        UnaryMathKind::Ceil => ops::ceil_device(input, stream),
        UnaryMathKind::Round => ops::round_device(input, 0, stream),
    }
}

/// The real cube root on MLX, which has no `cbrt`: `y = |x|^(1/3)` from
/// `power` (whose error is amplified by the rounding of the exponent `1/3`
/// to `f32`, by up to `ln|x| * 2^-25` relative) followed by one Newton step
/// `y - (y - |x| / y^2) / 3` for `y^3 = |x|`, which squares that error away
/// and leaves the rounding of the step, about one float32 ulp. The sign of
/// `x` is restored afterwards. Zeros, infinities and NaN, where the Newton
/// step would form `0 / 0` or `inf / inf`, return `x` itself, as `cbrt`
/// does. Metal flushes float32 subnormals to zero (MLX's `sqrt` and `log2`
/// see them as zeros too), so a subnormal `x` counts as a zero: it returns
/// `x * 1`, the flushed signed zero, not the subnormal itself.
fn mlx_cbrt(input: &Array, stream: &StreamOrDevice) -> Result<Array, mlx_rs::error::Exception> {
    let scalar = Array::from_f32;
    let magnitude = input.abs_device(stream)?;
    let root = magnitude.power_device(scalar(1.0 / 3.0), stream)?;
    let quotient = magnitude.divide_device(root.square_device(stream)?, stream)?;
    let step = root
        .subtract_device(&quotient, stream)?
        .divide_device(scalar(3.0), stream)?;
    let refined = root.subtract_device(&step, stream)?;
    let signed = ops::r#where_device(
        input.lt_device(scalar(0.0), stream)?,
        refined.negative_device(stream)?,
        &refined,
        stream,
    )?;
    let regular = ops::logical_and_device(
        magnitude.gt_device(scalar(0.0), stream)?,
        magnitude.lt_device(scalar(f32::INFINITY), stream)?,
        stream,
    )?;
    let flushed = input.multiply_device(scalar(1.0), stream)?;
    ops::r#where_device(&regular, &signed, &flushed, stream)
}

/// C `fmod(x, y)` on MLX, which only has the floor-mod `remainder` (sign of
/// `y`). `remainder` computes Metal's exact `fmod` and adds `y` when the
/// signs of that remainder and `y` differ, which would round; for `|x|` and
/// `|y|` the signs never differ, so `remainder(|x|, |y|)` is the exact
/// `fmod(|x|, |y|)`, and `fmod(x, y)` is it with the sign of `x`. A zero
/// result takes the sign of `x` through `x * 0`, so `fmod(-0, y)` and
/// `fmod(-2, 1)` are `-0` as in C.
fn mlx_fmod(
    x: &Array,
    y: &Array,
    stream: &StreamOrDevice,
) -> Result<Array, mlx_rs::error::Exception> {
    let scalar = Array::from_f32;
    let magnitude = x
        .abs_device(stream)?
        .remainder_device(y.abs_device(stream)?, stream)?;
    let signed = ops::r#where_device(
        x.lt_device(scalar(0.0), stream)?,
        magnitude.negative_device(stream)?,
        &magnitude,
        stream,
    )?;
    ops::r#where_device(
        magnitude.eq_device(scalar(0.0), stream)?,
        x.multiply_device(scalar(0.0), stream)?,
        &signed,
        stream,
    )
}

fn mlx_value(values: &[Array], node_id: usize) -> Result<&Array, String> {
    values
        .get(node_id)
        .ok_or_else(|| format!("MLX operand node {node_id} is missing"))
}

fn mlx_op_name(op: &TensorOp) -> &'static str {
    match op {
        TensorOp::Input { .. } => "input",
        TensorOp::ScalarConstant { .. } => "constant",
        TensorOp::Constant { .. } => "tensor_constant",
        TensorOp::Cast { .. } => "cast",
        TensorOp::Add { .. } => "add",
        TensorOp::Sub { .. } => "sub",
        TensorOp::Div { .. } => "div",
        TensorOp::Mul { .. } => "mul",
        TensorOp::Greater { .. } => "greater",
        TensorOp::Compare { kind, .. } => kind.name(),
        TensorOp::Where { .. } => "where",
        TensorOp::Cond { .. } => "cond",
        TensorOp::While { .. } => "while",
        TensorOp::Fori { .. } => "fori",
        TensorOp::ForiJvp { .. } => "fori_jvp",
        TensorOp::ForiVjp { .. } => "fori_vjp",
        TensorOp::ForiVjpJvp { .. } => "fori_vjp_jvp",
        TensorOp::Scan { .. } => "scan",
        TensorOp::ScanVjp { .. } => "scan_vjp",
        TensorOp::ScanVjpJvp { .. } => "scan_vjp_jvp",
        TensorOp::Sum { .. } => "sum",
        TensorOp::SumAxis { .. } => "sum_axis",
        TensorOp::Matmul { .. } => "matmul",
        TensorOp::Solve { .. } => "solve",
        TensorOp::Linalg { kind, .. } => kind.name(),
        TensorOp::Cholesky { .. } => "cholesky",
        TensorOp::CholeskyAd { .. } => "cholesky_ad",
        TensorOp::Triangular { .. } => "triangular",
        TensorOp::Tanh { .. } => "tanh",
        TensorOp::Exp { .. } => "exp",
        TensorOp::Sqrt { .. } => "sqrt",
        TensorOp::SqrtDerivative { .. } => "sqrt_derivative",
        TensorOp::Reshape { .. } => "reshape",
        TensorOp::Mean { .. } => "mean",
        TensorOp::MeanAxis { .. } => "mean_axis",
        TensorOp::Sin { .. } => "sin",
        TensorOp::Cos { .. } => "cos",
        TensorOp::Powi { .. } => "powi",
        TensorOp::Pow { .. } => "pow",
        TensorOp::Transpose { .. } => "transpose",
        TensorOp::Log { .. } => "log",
        TensorOp::Log1p { .. } => "log1p",
        TensorOp::Expm1 { .. } => "expm1",
        TensorOp::Erf { .. } => "erf",
        TensorOp::Erfc { .. } => "erfc",
        TensorOp::Atan2 { .. } => "atan2",
        TensorOp::UnaryMath { kind, .. } => kind.name(),
        TensorOp::Fmod { .. } => "fmod",
        TensorOp::StopGradient { .. } => "stop_gradient",
        TensorOp::Custom { .. } => "custom",
        TensorOp::CumSum { .. } => "cumsum",
        TensorOp::Concat { .. } => "concat",
        TensorOp::Slice { .. } => "slice",
        TensorOp::PadSlice { .. } => "pad_slice",
        TensorOp::Gather { .. } => "gather",
        TensorOp::ScatterAdd { .. } => "scatter_add",
        TensorOp::Broadcast { .. } => "broadcast",
    }
}

#[cfg(test)]
mod retained_loss_tests {
    use super::*;
    use crate::tensor_ir::{SymbolicCotangent, TensorIr};

    fn fixture() -> Result<(MlxAdamPlan, BTreeMap<String, DynamicTensor>), String> {
        let mut graph = TensorIr::new();
        let parameter = graph.input_typed("parameter", vec![2048], TensorDType::F32)?;
        let batch = graph.input_typed("batch", vec![2048], TensorDType::F32)?;
        let mut value = graph.sub(parameter, batch)?;
        for _ in 0..32 {
            value = graph.sin(value)?;
        }
        let square = graph.mul(value, value)?;
        let loss = graph.sum(square)?;
        let reverse = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
        let (shared, outputs) = reverse
            .graph
            .compile_cpu_many(&[reverse.primals[loss], reverse.gradients["parameter"]])?;
        let inputs = BTreeMap::from([
            ("parameter".into(), DynamicTensor::filled(vec![2048], 0.5)?),
            ("batch".into(), DynamicTensor::filled(vec![2048], 0.125)?),
        ]);
        let optimizer = MlxAdamPlan::new(
            shared,
            outputs[0],
            BTreeMap::from([("parameter".into(), outputs[1])]),
            &inputs,
            ["parameter".into()],
            0.01,
            0.9,
            0.999,
            1e-8,
        )?;
        Ok((optimizer, inputs))
    }

    #[test]
    fn loss_prunes_backward_nodes_and_keeps_latest_state_and_batch() -> Result<(), String> {
        let (mut optimizer, mut inputs) = fixture()?;
        assert!(optimizer.forward_loss_plan.get().is_none());
        for step in 0..4 {
            inputs.insert(
                "batch".into(),
                DynamicTensor::filled(vec![2048], step as f64 * 0.01)?,
            );
            optimizer.step(&inputs)?;
            let parameter = optimizer.parameter("parameter")?;
            let counter = optimizer.adam["parameter"].step;
            let actual = optimizer.loss(&inputs)?;
            let expected = MlxBackend
                .execute_many_with_retained(
                    &optimizer.plan,
                    &[optimizer.loss_node_id],
                    &inputs,
                    optimizer.retained_inputs.arrays(),
                )?
                .remove(0);
            assert_eq!(
                actual
                    .storage()
                    .iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                expected
                    .storage()
                    .iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>()
            );
            assert_eq!(optimizer.adam["parameter"].step, counter);
            assert_eq!(
                optimizer.parameter("parameter")?.storage(),
                parameter.storage()
            );
        }
        let forward = optimizer.forward_loss_plan.get().unwrap().as_ref().unwrap();
        assert!(forward.node_count() < optimizer.plan.node_count());
        Ok(())
    }

    #[test]
    fn learning_rate_setter_applies_to_later_steps_only() -> Result<(), String> {
        let (mut optimizer, inputs) = fixture()?;
        assert_eq!(optimizer.learning_rate(), 0.01);
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(optimizer.set_learning_rate(invalid).is_err());
        }
        assert_eq!(optimizer.learning_rate(), 0.01);
        // A zero rate still advances the moments and the bias-correction
        // counter, but leaves the parameter unchanged.
        optimizer.set_learning_rate(0.0)?;
        let initial = optimizer.parameter("parameter")?;
        optimizer.step(&inputs)?;
        assert_eq!(optimizer.adam["parameter"].step, 1);
        assert_eq!(
            optimizer.parameter("parameter")?.storage(),
            initial.storage()
        );
        optimizer.set_learning_rate(0.01)?;
        optimizer.step(&inputs)?;
        assert_ne!(
            optimizer.parameter("parameter")?.storage(),
            initial.storage()
        );
        Ok(())
    }

    #[test]
    #[ignore = "isolated MLX retained loss profiling"]
    fn profile_forward_only_loss() -> Result<(), String> {
        let (optimizer, inputs) = fixture()?;
        optimizer.loss(&inputs)?;
        let mut before = Vec::new();
        let mut after = Vec::new();
        for round in 0..12 {
            for mode in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = std::time::Instant::now();
                for _ in 0..5 {
                    if mode {
                        optimizer.loss(&inputs)?;
                    } else {
                        MlxBackend.execute_many_with_retained(
                            &optimizer.plan,
                            &[optimizer.loss_node_id],
                            &inputs,
                            optimizer.retained_inputs.arrays(),
                        )?;
                    }
                }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
                if mode {
                    after.push(ms);
                } else {
                    before.push(ms);
                }
            }
        }
        before.sort_by(f64::total_cmp);
        after.sort_by(f64::total_cmp);
        println!("before_median_ms={} after_median_ms={} before_samples={before:?} after_samples={after:?}", (before[5]+before[6])/2.0, (after[5]+after[6])/2.0);
        Ok(())
    }
}

#[cfg(test)]
mod loop_window_profile_tests {
    use super::*;
    use crate::tensor_ir::TensorIr;

    #[test]
    #[ignore = "isolated bounded lazy-loop time and device-memory measurement"]
    fn profile_pure_loop_windows() -> Result<(), String> {
        let _guard = mlx_execution_guard();
        for (lanes, length, depth) in [(256, 16, 1), (256, 128, 8), (65536, 16, 1), (65536, 128, 8)]
        {
            let mut graph = TensorIr::new();
            let mut output = graph.input_typed("carry", vec![lanes], TensorDType::F32)?;
            for _ in 0..depth {
                output = graph.sin(output)?;
            }
            let plan = graph.compile_cpu(output)?;
            let initial = mlx_array_from_dynamic(&DynamicTensor::filled(vec![lanes], 0.125)?)?;
            transforms::eval([&initial]).map_err(|e| e.to_string())?;
            let mut reference = None;
            for round in 0..4 {
                let windows = if round % 2 == 0 {
                    [1, 4, 16]
                } else {
                    [16, 4, 1]
                };
                for window in windows {
                    if let Ok(selected) = std::env::var("QUABLA_MLX_LOOP_WINDOW") {
                        if selected.parse::<usize>().map_err(|e| e.to_string())? != window {
                            continue;
                        }
                    }
                    mlx_rs::memory::reset_peak_memory().map_err(|e| e.to_string())?;
                    let baseline = mlx_rs::memory::active_memory().map_err(|e| e.to_string())?;
                    let mut carry = initial.clone();
                    let start = std::time::Instant::now();
                    for iteration in 0..length {
                        carry = MlxBackend
                            .lower_arrays_with_retained(
                                &plan,
                                &[plan.output_node_id],
                                &BTreeMap::new(),
                                &BTreeMap::from([("carry".into(), carry)]),
                            )?
                            .remove(0);
                        if (iteration + 1) % window == 0 || iteration + 1 == length {
                            transforms::eval([&carry]).map_err(|e| e.to_string())?;
                        }
                    }
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let extra_peak = mlx_rs::memory::peak_memory()
                        .map_err(|e| e.to_string())?
                        .saturating_sub(baseline);
                    let actual = mlx_host_values(&carry)?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    if let Some(expected) = &reference {
                        assert_eq!(&actual, expected);
                    } else {
                        reference = Some(actual);
                    }
                    println!("lanes={lanes} length={length} depth={depth} round={round} window={window} elapsed_ms={ms} extra_device_peak_bytes={extra_peak}");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "mlx_checkpoint_tests.rs"]
mod checkpoint_tests;

#[cfg(test)]
#[path = "mlx_cholesky_tests.rs"]
mod cholesky_tests;
