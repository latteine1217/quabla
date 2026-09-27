use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard, PoisonError};

use mlx_rs::{ops, transforms, Array, Dtype, StreamOrDevice};

use super::{
    sqrt_derivative_coefficient, DynamicTensor, TensorBackend, TensorComparison, TensorDType,
    TensorDeviceBackend, TensorExecutionPlan, TensorForiExecutionPlan, TensorOp,
};

/// Apple MLX backend for the supported rank-N Tensor IR primitives.
///
/// MLX uses unified memory on Apple silicon. This backend keeps intermediate
/// arrays on MLX's GPU stream and only materializes the final value for Nabla's
/// host-facing `DynamicTensor` result.
///
/// 執行緒安全：所有 MLX 圖建構、求值與回讀都經由行程層級的鎖序列化，
/// 因此可從多條執行緒同時呼叫；GPU 工作本來就排在同一條預設 stream 上。
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxBackend;

/// 序列化本模組對 MLX 的所有存取。
///
/// 所依賴的 MLX 0.25 在呼叫端執行緒上把 GPU 工作編碼進預設 GPU stream，
/// 而該 stream 的 Metal command buffer / encoder 為全行程共用且無鎖保護；
/// 兩條執行緒同時 eval 會交錯編碼同一個 command buffer，觸發 Metal
/// assertion 中止或卡死。
static MLX_EXECUTION_LOCK: Mutex<()> = Mutex::new(());

/// 取得 MLX 行程鎖。
///
/// std `Mutex` 不可重入：只在公開進入點取得一次，內部 helper 一律假設已持有，
/// 公開函式之間也只能委派給恰好一個會取鎖的公開函式。
fn mlx_execution_guard() -> MutexGuard<'static, ()> {
    // 鎖不保護任何 Rust 資料，poison 只表示先前持鎖者 panic；照常序列化即可。
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
    gradient_node_ids: BTreeMap<String, usize>,
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
        if !(learning_rate.is_finite()
            && learning_rate > 0.0
            && epsilon.is_finite()
            && epsilon > 0.0
            && beta1.is_finite()
            && (0.0..1.0).contains(&beta1)
            && beta2.is_finite()
            && (0.0..1.0).contains(&beta2))
        {
            return Err(
                "MLX Adam requires positive finite learning_rate and epsilon, plus beta1/beta2 in [0, 1)"
                    .to_string(),
            );
        }
        let mut names = retained_input_names.into_iter().collect::<Vec<_>>();
        names.extend(gradient_node_ids.keys().cloned());
        names.sort();
        names.dedup();
        let retained_inputs = MlxRetainedInputs::upload(inputs, names)?;
        for parameter_name in gradient_node_ids.keys() {
            let node = plan
                .nodes
                .iter()
                .find(|node| matches!(&node.op, TensorOp::Input { name } if name == parameter_name))
                .ok_or_else(|| {
                    format!("MLX Adam parameter {parameter_name:?} is not a plan input")
                })?;
            let parameter = retained_inputs
                .values
                .get(parameter_name)
                .ok_or_else(|| format!("MLX Adam did not retain parameter {parameter_name:?}"))?;
            if parameter.shape() != mlx_shape(&node.shape)?.as_slice() {
                return Err(format!(
                    "MLX Adam parameter {parameter_name:?} has an unexpected retained shape"
                ));
            }
        }
        Ok(Self {
            plan,
            loss_node_id,
            gradient_node_ids,
            retained_inputs,
            adam: BTreeMap::new(),
            learning_rate,
            beta1,
            beta2,
            epsilon,
        })
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
        MlxBackend
            .execute_many_with_retained(
                &self.plan,
                &[self.loss_node_id],
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
        let node = self
            .plan
            .nodes
            .iter()
            .find(|node| matches!(&node.op, TensorOp::Input { name: input_name } if input_name == name))
            .ok_or_else(|| format!("MLX Adam parameter {name:?} is not a plan input"))?;
        let parameter = self
            .retained_inputs
            .values
            .get(name)
            .ok_or_else(|| format!("MLX Adam parameter {name:?} is not retained"))?;
        DynamicTensor::with_dtype(
            node.shape.clone(),
            parameter
                .as_slice::<f32>()
                .iter()
                .copied()
                .map(f64::from)
                .collect(),
            node.dtype,
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
        let outputs =
            self.execute_arrays_with_retained(plan, output_node_ids, inputs, retained_inputs)?;
        output_node_ids
            .iter()
            .zip(outputs)
            .map(|(node_id, output)| {
                let node = plan
                    .nodes
                    .get(*node_id)
                    .ok_or_else(|| format!("MLX output node {node_id} is missing"))?;
                // 讀回值為 f32；依節點邏輯 dtype 標記（f64 節點即 f32 降階執行的結果）。
                DynamicTensor::with_dtype(
                    node.shape.clone(),
                    output
                        .as_slice::<f32>()
                        .iter()
                        .copied()
                        .map(f64::from)
                        .collect(),
                    node.dtype,
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
            // 本後端所有陣列皆為 f32；任何無法降階為 f32 的邏輯 dtype 必須明確拒絕。
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
                            .data()
                            .iter()
                            .map(|value| *value as f32)
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
                // 來源與目標都以 f32 執行（見上方檢查），cast 為恆等。
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
                // IR 的 greater 是 0/1 浮點遮罩（與 CPU 一致）；MLX 比較產生 bool，
                // 必須轉回 f32，否則回讀、累加與 Cond 謂詞都會遇到 dtype 不符。
                TensorOp::Greater { lhs, rhs } => mlx_value(&values, *lhs)?
                    .gt_device(mlx_value(&values, *rhs)?, &stream)
                    .and_then(|mask| mask.as_dtype_device(Dtype::Float32, &stream))
                    .map_err(|error| error.to_string()),
                // Bool 節點同樣以 f32 0/1 跨越節點邊界（execution_dtype(Bool) = f32）。
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
                    // 主機同步邊界：純量謂詞讀回一次，只在 GPU stream 上執行被選分支。
                    // 不以 where 同時計算兩分支，避免未選分支的 NaN/Inf 滲入值與梯度。
                    let predicate = mlx_scalar_predicate(mlx_value(&values, *predicate)?)?;
                    let branch_inputs = captures
                        .iter()
                        .map(|(name, node_id)| {
                            mlx_value(&values, *node_id).map(|value| (name.clone(), value.clone()))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    mlx_execute_plan_output(self, branches.selected(predicate), &branch_inputs)
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
                TensorOp::SqrtDerivative { input, order } => {
                    let input = mlx_value(&values, *input)?;
                    let zero = Array::from_f32(0.0);
                    let exponent = Array::from_f32(0.5 - *order as f32);
                    let coefficient = Array::from_f32(sqrt_derivative_coefficient(*order) as f32);
                    let power = input
                        .power_device(&exponent, &stream)
                        .map_err(|error| error.to_string())?;
                    let scaled = power
                        .multiply_device(&coefficient, &stream)
                        .map_err(|error| error.to_string())?;
                    let positive = input
                        .gt_device(&zero, &stream)
                        .map_err(|error| error.to_string())?;
                    ops::r#where_device(&positive, &scaled, &zero, &stream)
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
                TensorOp::Powi { input, exponent } => {
                    let exponent = Array::from_f32(*exponent as f32);
                    mlx_value(&values, *input)?
                        .power_device(&exponent, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Matmul { lhs, rhs } => mlx_value(&values, *lhs)?
                    .matmul_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Solve { .. } => {
                    return Err(
                        "MLX GPU backend does not yet support solve: MLX linalg::solve only accepts a CPU stream"
                            .to_string(),
                    )
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
            }
            .map_err(|error| {
                format!(
                    "MLX node {node_id} {} failed: {error}",
                    mlx_op_name(&node.op)
                )
            })?;
            values.push(value);
        }

        let outputs = output_node_ids
            .iter()
            .map(|node_id| mlx_value(&values, *node_id))
            .collect::<Result<Vec<_>, _>>()?;
        transforms::eval(outputs.iter().copied())
            .map_err(|error| format!("MLX output evaluation failed: {error}"))?;
        Ok(outputs.into_iter().cloned().collect())
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
    let mut carries = Vec::with_capacity(loop_plan.upper - loop_plan.lower + 1);
    let mut carry = initial_carry;
    carries.push(carry.clone());
    for index in loop_plan.lower..loop_plan.upper {
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
        carry = outputs
            .pop()
            .ok_or_else(|| "MLX Fori forward body produced no carry output".to_string())?;
        carries.push(carry.clone());
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
    for (offset, carry) in carries[..carries.len() - 1]
        .iter()
        .cloned()
        .enumerate()
        .rev()
    {
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
        let mut tangent_name = format!("__nabla_mlx_fori_jvp_tangent_{index}");
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
    let mut carries = vec![initial_carry];
    let mut carry_tangents = vec![initial_tangent];
    for index in loop_plan.lower..loop_plan.upper {
        let mut inputs = external_captures.clone();
        inputs.insert(
            loop_plan.carry_name.clone(),
            carries
                .last()
                .cloned()
                .ok_or_else(|| "MLX Fori JVP carry tape is empty".to_string())?,
        );
        inputs.insert(
            loop_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(vec![], index as f64)?)?,
        );
        for (name, tangent_name) in &plan.mlx_forward_jvp.tangent_names {
            let tangent = if name == &loop_plan.carry_name {
                carry_tangents
                    .last()
                    .cloned()
                    .ok_or_else(|| "MLX Fori JVP tangent tape is empty".to_string())?
            } else {
                external_tangents
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("MLX Fori JVP lacks tangent capture {name:?}"))?
            };
            inputs.insert(tangent_name.clone(), tangent);
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
        let next = values
            .first()
            .cloned()
            .ok_or_else(|| "MLX Fori JVP forward body has no primal output".to_string())?;
        let tangent = values
            .get(1)
            .cloned()
            .ok_or_else(|| "MLX Fori JVP forward body has no tangent output".to_string())?;
        carries.push(next);
        carry_tangents.push(tangent);
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
    for offset in (0..loop_plan.upper - loop_plan.lower).rev() {
        let mut inputs = external_captures.clone();
        inputs.insert(loop_plan.carry_name.clone(), carries[offset].clone());
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
                carry_tangents[offset].clone()
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
                plan.gradient_tangent_plans
                    .get(name)
                    .ok_or_else(|| format!("MLX Fori VJP JVP has no gradient plan for {name:?}"))?,
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
    let mut carries = vec![initial_carry];
    let mut carry_tangents = vec![initial_tangent];
    for index in scan_plan.lower..scan_plan.upper {
        let mut inputs = external_captures.clone();
        inputs.insert(
            scan_plan.carry_name.clone(),
            carries
                .last()
                .cloned()
                .ok_or_else(|| "MLX Scan JVP carry tape is empty".to_string())?,
        );
        inputs.insert(
            scan_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(vec![], index as f64)?)?,
        );
        for (name, tangent_name) in &plan.mlx_forward_jvp.tangent_names {
            let tangent = if name == &scan_plan.carry_name {
                carry_tangents
                    .last()
                    .cloned()
                    .ok_or_else(|| "MLX Scan JVP tangent tape is empty".to_string())?
            } else {
                external_tangents
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("MLX Scan JVP lacks tangent capture {name:?}"))?
            };
            inputs.insert(tangent_name.clone(), tangent);
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
        carries.push(
            values
                .first()
                .cloned()
                .ok_or_else(|| "MLX Scan JVP forward body has no primal output".to_string())?,
        );
        carry_tangents.push(
            values
                .get(1)
                .cloned()
                .ok_or_else(|| "MLX Scan JVP forward body has no tangent output".to_string())?,
        );
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
    for offset in (0..scan_plan.upper - scan_plan.lower).rev() {
        let mut inputs = external_captures.clone();
        inputs.insert(scan_plan.carry_name.clone(), carries[offset].clone());
        inputs.insert(
            scan_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(
                vec![],
                (scan_plan.lower + offset) as f64,
            )?)?,
        );
        let output_step_cotangent =
            mlx_scan_output_cotangent_at(&output_cotangent, offset, &output_step_shape, &stream)?;
        let output_step_cotangent_tangent = mlx_scan_output_cotangent_at(
            &output_cotangent_tangent,
            offset,
            &output_step_shape,
            &stream,
        )?;
        let carry_gradients =
            mlx_region_vjp_gradients(backend, &vjp.carry, &inputs, carry_cotangent.clone())?;
        let output_gradients =
            mlx_region_vjp_gradients(backend, &vjp.output, &inputs, output_step_cotangent.clone())?;
        let mut jvp_inputs = inputs;
        jvp_inputs.insert(plan.carry_cotangent_name.clone(), carry_cotangent);
        jvp_inputs.insert(plan.output_cotangent_name.clone(), output_step_cotangent);
        for (name, tangent_name) in &plan.tangent_names {
            let tangent = if name == &scan_plan.carry_name {
                carry_tangents[offset].clone()
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
    let mut carries = Vec::with_capacity(scan_plan.upper - scan_plan.lower + 1);
    let mut carry = initial_carry;
    carries.push(carry.clone());
    for index in scan_plan.lower..scan_plan.upper {
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
        carry = values
            .first()
            .cloned()
            .ok_or_else(|| "MLX Scan forward body has no carry output".to_string())?;
        carries.push(carry.clone());
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
    for (offset, carry) in carries[..carries.len() - 1]
        .iter()
        .cloned()
        .enumerate()
        .rev()
    {
        let mut body_inputs = external_captures.clone();
        body_inputs.insert(scan_plan.carry_name.clone(), carry);
        body_inputs.insert(
            scan_plan.index_name.clone(),
            mlx_array_from_dynamic(&DynamicTensor::filled(
                vec![],
                (scan_plan.lower + offset) as f64,
            )?)?,
        );
        let output_gradient =
            mlx_scan_output_cotangent_at(&output_cotangent, offset, &output_step_shape, &stream)?;
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

fn mlx_array_from_dynamic(input: &DynamicTensor) -> Result<Array, String> {
    let shape = mlx_shape(input.shape())?;
    let data = input
        .data()
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let stream = StreamOrDevice::gpu();
    let host_value = Array::from_slice(&data, &shape);
    let zeros = Array::zeros_device::<f32>(&shape, &stream).map_err(|error| error.to_string())?;
    host_value
        .add_device(&zeros, &stream)
        .map_err(|error| error.to_string())
}

/// 讀回 `Cond` 的純量謂詞（評估並同步 GPU stream）。
///
/// 語意與 CPU `tensor_scalar_predicate` 一致：非有限值拒絕，非零為真。
fn mlx_scalar_predicate(predicate: &Array) -> Result<bool, String> {
    let value = predicate
        .try_item::<f32>()
        .map_err(|error| format!("MLX Cond predicate readback failed: {error}"))?;
    if !value.is_finite() {
        return Err("conditional predicate must be finite".to_string());
    }
    Ok(value != 0.0)
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
        TensorOp::Cast { .. } => "cast",
        TensorOp::Add { .. } => "add",
        TensorOp::Sub { .. } => "sub",
        TensorOp::Div { .. } => "div",
        TensorOp::Mul { .. } => "mul",
        TensorOp::Greater { .. } => "greater",
        TensorOp::Compare { kind, .. } => kind.name(),
        TensorOp::Where { .. } => "where",
        TensorOp::Cond { .. } => "cond",
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
        TensorOp::Transpose { .. } => "transpose",
        TensorOp::Log { .. } => "log",
        TensorOp::Concat { .. } => "concat",
        TensorOp::Slice { .. } => "slice",
        TensorOp::PadSlice { .. } => "pad_slice",
        TensorOp::Broadcast { .. } => "broadcast",
    }
}
