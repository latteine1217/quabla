use std::collections::{BTreeMap, BTreeSet};

use std::sync::{Arc, Mutex};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cusolver::{safe::DnHandle, sys as cusolver_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use super::{
    contiguous_strides, element_count, sqrt_derivative_coefficient, tensor_op_inputs,
    DynamicTensor, TensorBackend, TensorExecutionPlan, TensorOp,
};

const CUDA_MATMUL_TILE: usize = 32;
const CUDA_MATMUL_TILE_U64: u64 = 32;
const CUDA_MATMUL_BLOCK: u32 = 16;
const CUDA_REDUCTION_BLOCK: u32 = 256;

/// NVIDIA CUDA backend for fused rank-N elementwise and rank-two matmul plans.
///
/// The backend compiles a plan-specialized CUDA C kernel with NVRTC. It is
/// intentionally narrow until reductions, GEMM, and reverse transforms have
/// GPU lowering that preserves the current TensorIr semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend {
    device_ordinal: usize,
}

/// Immutable CUDA lowering with a retained CUDA context and loaded NVRTC module.
///
/// Intermediate buffers are recycled after their final consumer. Retained inputs,
/// the plan output, and optimizer state keep stable device allocations.
#[derive(Clone, Debug)]
pub struct CudaExecutionPlan {
    plan: TensorExecutionPlan,
    fused_elementwise: bool,
    matmul_bias_tanh: Option<CudaMatmulBiasTanhEpilogue>,
    device_ordinal: usize,
    context: Arc<CudaContext>,
    module: Arc<CudaModule>,
    blas: Option<Arc<Mutex<CudaBlas>>>,
    solver: Option<Arc<Mutex<DnHandle>>>,
    state: Arc<Mutex<CudaExecutionState>>,
}

#[derive(Clone, Copy, Debug)]
struct CudaMatmulBiasTanhEpilogue {
    lhs: usize,
    rhs: usize,
    bias: usize,
    rows: usize,
    inner: usize,
    cols: usize,
}

#[derive(Debug, Default)]
struct CudaExecutionState {
    values: Vec<Option<CudaSlice<f32>>>,
    free_buffers: BTreeMap<usize, Vec<CudaSlice<f32>>>,
    adam: BTreeMap<String, CudaAdamState>,
}

#[derive(Debug)]
struct CudaAdamState {
    first_moment: CudaSlice<f32>,
    second_moment: CudaSlice<f32>,
    step: u64,
}

impl CudaBackend {
    pub fn new(device_ordinal: usize) -> Self {
        Self { device_ordinal }
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn compile(&self, plan: TensorExecutionPlan) -> Result<CudaExecutionPlan, String> {
        let matmul_bias_tanh = cuda_matmul_bias_tanh_epilogue(&plan);
        let fused_candidate = plan.uses_fused_elementwise_kernel()
            && !matches!(&plan.nodes[plan.output_node_id].op, TensorOp::Input { .. });
        let (source, fused_elementwise) = if fused_candidate {
            match plan.cuda_source() {
                Ok(source) => (source, true),
                Err(_) => (cuda_program_source(&plan)?, false),
            }
        } else {
            (cuda_program_source(&plan)?, false)
        };
        let mut source = source;
        if matmul_bias_tanh.is_some() {
            source.push_str(CUDA_MATMUL_BIAS_TANH_SOURCE);
        }
        let context = CudaContext::new(self.device_ordinal)
            .map_err(|error| format!("failed to create CUDA context: {error:?}"))?;
        let ptx = compile_ptx(source).map_err(|error| {
            format!("failed to compile CUDA device program with NVRTC: {error:?}")
        })?;
        let module = context
            .load_module(ptx)
            .map_err(|error| format!("failed to load CUDA device program: {error:?}"))?;
        let blas = cuda_blas(context.default_stream())?;
        let solver = cuda_solver(context.default_stream())?;
        Ok(CudaExecutionPlan {
            plan,
            fused_elementwise,
            matmul_bias_tanh,
            device_ordinal: self.device_ordinal,
            context,
            module,
            blas,
            solver,
            state: Arc::new(Mutex::new(CudaExecutionState::default())),
        })
    }
}

impl CudaExecutionPlan {
    pub fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn uses_cublas(&self) -> bool {
        self.matmul_bias_tanh.is_none() && self.blas.is_some()
    }

    pub fn uses_fused_matmul_bias_tanh(&self) -> bool {
        self.matmul_bias_tanh.is_some()
    }

    /// Number of allocated device buffers currently owned by this plan,
    /// including reusable temporary buffers and retained values.
    pub fn device_buffer_count(&self) -> Result<usize, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        Ok(state.values.iter().flatten().count()
            + state.free_buffers.values().map(Vec::len).sum::<usize>())
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.context
            .default_stream()
            .synchronize()
            .map_err(|error| format!("failed to synchronize CUDA execution plan: {error:?}"))
    }

    pub fn execute(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.execute_retaining(inputs, &BTreeSet::new())
    }

    pub fn execute_retaining(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeSet<String>,
    ) -> Result<DynamicTensor, String> {
        self.execute_retaining_inner(inputs, retained_inputs, true)?
            .ok_or_else(|| "CUDA execution did not materialize an output".to_string())
    }

    /// Executes a plan while retaining device buffers without copying any
    /// output to the host. Optimizer paths consume gradient nodes in place.
    pub fn execute_retaining_without_output(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeSet<String>,
    ) -> Result<(), String> {
        self.execute_retaining_inner(inputs, retained_inputs, false)
            .map(|_| ())
    }

    fn execute_retaining_inner(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeSet<String>,
        copy_output: bool,
    ) -> Result<Option<DynamicTensor>, String> {
        let stream = self.context.default_stream();
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let CudaExecutionState {
            values,
            free_buffers,
            ..
        } = &mut *state;
        if let Some(epilogue) = self.matmul_bias_tanh {
            execute_cuda_matmul_bias_tanh_program(
                &self.plan,
                inputs,
                &stream,
                &self.module,
                values,
                free_buffers,
                retained_inputs,
                copy_output,
                epilogue,
            )
        } else if self.fused_elementwise {
            execute_cuda_fused_elementwise_program(
                &self.plan,
                inputs,
                &stream,
                &self.module,
                values,
                free_buffers,
                retained_inputs,
                copy_output,
            )
        } else {
            execute_cuda_device_program(
                &self.plan,
                inputs,
                CudaProgramRuntime {
                    stream: &stream,
                    module: &self.module,
                    blas: self.blas.as_ref(),
                    solver: self.solver.as_ref(),
                },
                values,
                free_buffers,
                retained_inputs,
                copy_output,
            )
        }
    }

    pub fn sgd_step_input_from_output(
        &self,
        parameter_name: &str,
        learning_rate: f32,
    ) -> Result<(), String> {
        if !(learning_rate.is_finite() && learning_rate > 0.0) {
            return Err("CUDA SGD learning rate must be finite and positive".to_string());
        }
        let parameter_node_id = input_node_id(&self.plan, parameter_name)?;
        let output_node_id = self.plan.output_node_id;
        if parameter_node_id >= output_node_id {
            return Err(
                "CUDA SGD requires a computed gradient output after its parameter input"
                    .to_string(),
            );
        }
        let stream = self.context.default_stream();
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let (before_output, output_and_after) = state.values.split_at_mut(output_node_id);
        let parameter = cuda_value_mut(before_output, parameter_node_id)?;
        let gradient = output_and_after
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(|| "CUDA SGD gradient output is missing".to_string())?;
        if parameter.len() != gradient.len() {
            return Err(format!(
                "CUDA SGD parameter {parameter_name:?} has {} elements, gradient has {}",
                parameter.len(),
                gradient.len()
            ));
        }
        let count = u64::try_from(parameter.len())
            .map_err(|_| "CUDA SGD parameter count exceeds u64".to_string())?;
        let launch_count = u32::try_from(parameter.len())
            .map_err(|_| "CUDA SGD parameter count exceeds u32 launch size".to_string())?;
        let kernel = self
            .module
            .load_function("nabla_sgd")
            .map_err(|error| format!("failed to load CUDA SGD kernel: {error:?}"))?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(parameter);
        launch.arg(gradient);
        launch.arg(&learning_rate);
        launch.arg(&count);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(launch_count))
                .map_err(|error| format!("failed to launch CUDA SGD kernel: {error:?}"))?;
        }
        Ok(())
    }

    pub fn adam_step_input_from_output(
        &self,
        parameter_name: &str,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.adam_step_input_from_node(
            parameter_name,
            self.plan.output_node_id,
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )
    }

    /// Updates a retained parameter from any computed gradient node.
    pub fn adam_step_input_from_node(
        &self,
        parameter_name: &str,
        gradient_node_id: usize,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<(), String> {
        if !(learning_rate.is_finite() && learning_rate > 0.0)
            || !(beta1.is_finite() && (0.0..1.0).contains(&beta1))
            || !(beta2.is_finite() && (0.0..1.0).contains(&beta2))
            || !(epsilon.is_finite() && epsilon > 0.0)
        {
            return Err("CUDA Adam requires positive finite learning_rate and epsilon, plus beta1/beta2 in [0, 1)".to_string());
        }
        let parameter_node_id = input_node_id(&self.plan, parameter_name)?;
        if parameter_node_id >= gradient_node_id {
            return Err(
                "CUDA Adam requires a computed gradient node after its parameter input".to_string(),
            );
        }
        let stream = self.context.default_stream();
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let count = state
            .values
            .get(parameter_node_id)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("CUDA input {parameter_name:?} has not been initialized"))?
            .len();
        if state
            .values
            .get(gradient_node_id)
            .and_then(Option::as_ref)
            .map(CudaSlice::len)
            != Some(count)
        {
            return Err("CUDA Adam parameter and gradient shapes must match".to_string());
        }
        if !state.adam.contains_key(parameter_name) {
            let first_moment = stream
                .alloc_zeros::<f32>(count)
                .map_err(|error| format!("failed to allocate CUDA Adam first moment: {error:?}"))?;
            let second_moment = stream.alloc_zeros::<f32>(count).map_err(|error| {
                format!("failed to allocate CUDA Adam second moment: {error:?}")
            })?;
            state.adam.insert(
                parameter_name.to_string(),
                CudaAdamState {
                    first_moment,
                    second_moment,
                    step: 0,
                },
            );
        }
        let CudaExecutionState {
            values,
            adam: adam_states,
            ..
        } = &mut *state;
        let (before_output, output_and_after) = values.split_at_mut(gradient_node_id);
        let parameter = cuda_value_mut(before_output, parameter_node_id)?;
        let gradient = output_and_after
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(|| "CUDA Adam gradient node is missing".to_string())?;
        let adam = adam_states
            .get_mut(parameter_name)
            .expect("CUDA Adam state was initialized");
        adam.step = adam
            .step
            .checked_add(1)
            .ok_or_else(|| "CUDA Adam step counter overflow".to_string())?;
        let correction1 = 1.0 - beta1.powf(adam.step as f32);
        let correction2 = 1.0 - beta2.powf(adam.step as f32);
        let count_u64 = u64::try_from(count)
            .map_err(|_| "CUDA Adam parameter count exceeds u64".to_string())?;
        let launch_count = u32::try_from(count)
            .map_err(|_| "CUDA Adam parameter count exceeds u32 launch size".to_string())?;
        let kernel = self
            .module
            .load_function("nabla_adam")
            .map_err(|error| format!("failed to load CUDA Adam kernel: {error:?}"))?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(parameter);
        launch.arg(gradient);
        launch.arg(&mut adam.first_moment);
        launch.arg(&mut adam.second_moment);
        launch.arg(&learning_rate);
        launch.arg(&beta1);
        launch.arg(&beta2);
        launch.arg(&epsilon);
        launch.arg(&correction1);
        launch.arg(&correction2);
        launch.arg(&count_u64);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(launch_count))
                .map_err(|error| format!("failed to launch CUDA Adam kernel: {error:?}"))?;
        }
        Ok(())
    }

    pub fn retained_input_to_host(&self, name: &str) -> Result<DynamicTensor, String> {
        let node_id = input_node_id(&self.plan, name)?;
        let shape = self.plan.nodes[node_id].shape.clone();
        let stream = self.context.default_stream();
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let buffer = state
            .values
            .get(node_id)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("CUDA input {name:?} has not been initialized"))?;
        let data = stream
            .clone_dtoh(buffer)
            .map_err(|error| format!("failed to copy CUDA input {name:?} to host: {error:?}"))?;
        DynamicTensor::new(shape, data.into_iter().map(f64::from).collect())
    }

    /// Materializes an already-evaluated node from a multi-output plan.
    pub fn computed_node_to_host(&self, node_id: usize) -> Result<DynamicTensor, String> {
        let node = self
            .plan
            .nodes
            .get(node_id)
            .ok_or_else(|| format!("CUDA node {node_id} does not exist"))?;
        let stream = self.context.default_stream();
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let buffer = state
            .values
            .get(node_id)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("CUDA node {node_id} has not been evaluated"))?;
        let data = stream
            .clone_dtoh(buffer)
            .map_err(|error| format!("failed to copy CUDA node {node_id} to host: {error:?}"))?;
        DynamicTensor::new(
            node.shape.clone(),
            data.into_iter().map(f64::from).collect(),
        )
    }

    /// Copies a retained input directly into another CUDA plan's retained input.
    ///
    /// Call this after every participating gradient plan has evaluated and
    /// updated its own parameter, so all plans observe the same parameter state
    /// on the next iteration.
    pub fn sync_retained_input_to(
        &self,
        source_name: &str,
        target: &CudaExecutionPlan,
        target_name: &str,
    ) -> Result<(), String> {
        if Arc::ptr_eq(&self.state, &target.state) && source_name == target_name {
            return Ok(());
        }
        let source_node_id = input_node_id(&self.plan, source_name)?;
        let target_node_id = input_node_id(&target.plan, target_name)?;
        if self.plan.nodes[source_node_id].shape != target.plan.nodes[target_node_id].shape {
            return Err(format!(
                "CUDA input {source_name:?} shape {:?} does not match target input {target_name:?} shape {:?}",
                self.plan.nodes[source_node_id].shape, target.plan.nodes[target_node_id].shape
            ));
        }
        let source_state = self
            .state
            .lock()
            .map_err(|_| "CUDA source execution plan state lock is poisoned".to_string())?;
        let source = source_state
            .values
            .get(source_node_id)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("CUDA source input {source_name:?} has not been initialized"))?;
        let mut target_state = target
            .state
            .lock()
            .map_err(|_| "CUDA target execution plan state lock is poisoned".to_string())?;
        let destination = target_state
            .values
            .get_mut(target_node_id)
            .and_then(Option::as_mut)
            .ok_or_else(|| format!("CUDA target input {target_name:?} has not been initialized"))?;
        if source.context() != destination.context() {
            return Err(
                "CUDA retained-input synchronization requires the same device context".to_string(),
            );
        }
        target
            .context
            .default_stream()
            .memcpy_dtod(source, destination)
            .map_err(|error| {
                format!(
                    "failed to synchronize CUDA input {source_name:?} to {target_name:?}: {error:?}"
                )
            })
    }
}

impl TensorBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        if let Some((lhs, rhs)) = direct_rank_two_matmul_inputs(plan)? {
            return self.execute_rank_two_matmul(plan, inputs, lhs, rhs);
        }
        if !plan.uses_fused_elementwise_kernel() {
            return self.execute_device_program(plan, inputs);
        }

        let source = plan.cuda_source()?;
        let output_shape = plan.output_shape()?;
        let count = element_count(&output_shape)?;
        let launch_count = u32::try_from(count)
            .map_err(|_| "CUDA elementwise launch exceeds u32 element count".to_string())?;
        let input_nodes = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                TensorOp::Input { name } => Some((name.as_str(), node.shape.as_slice())),
                _ => None,
            })
            .collect::<Vec<_>>();

        for (name, shape) in &input_nodes {
            let input = inputs
                .get(*name)
                .ok_or_else(|| format!("missing input {name:?}"))?;
            if input.shape() != *shape {
                return Err(format!(
                    "input {name:?} has shape {:?}, expected {:?}",
                    input.shape(),
                    shape
                ));
            }
        }

        let context = CudaContext::new(self.device_ordinal)
            .map_err(|error| format!("failed to create CUDA context: {error:?}"))?;
        let stream = context.default_stream();
        let ptx = compile_ptx(source).map_err(|error| {
            format!("failed to compile CUDA elementwise kernel with NVRTC: {error:?}")
        })?;
        let module = context
            .load_module(ptx)
            .map_err(|error| format!("failed to load CUDA elementwise module: {error:?}"))?;
        let kernel = module
            .load_function("nabla_fused_elementwise")
            .map_err(|error| format!("failed to load CUDA elementwise kernel: {error:?}"))?;

        let host_inputs = input_nodes
            .iter()
            .map(|(name, _)| {
                inputs[*name]
                    .data()
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let device_inputs = host_inputs
            .iter()
            .zip(input_nodes.iter())
            .map(|(input, (name, _))| {
                stream
                    .clone_htod(input)
                    .map_err(|error| format!("failed to copy input {name:?} to CUDA: {error:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut device_output = stream
            .alloc_zeros::<f32>(count)
            .map_err(|error| format!("failed to allocate CUDA output: {error:?}"))?;
        let device_count = count as u64;

        let mut launch = stream.launch_builder(&kernel);
        for input in &device_inputs {
            launch.arg(input);
        }
        launch.arg(&mut device_output);
        launch.arg(&device_count);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(launch_count))
                .map_err(|error| format!("failed to launch CUDA elementwise kernel: {error:?}"))?;
        }
        let data = stream
            .clone_dtoh(&device_output)
            .map_err(|error| format!("failed to copy CUDA output to host: {error:?}"))?;
        DynamicTensor::new(output_shape, data.into_iter().map(f64::from).collect())
    }
}

impl CudaBackend {
    fn execute_device_program(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.compile(plan.clone())?.execute(inputs)
    }
}

struct CudaProgramRuntime<'a> {
    stream: &'a Arc<CudaStream>,
    module: &'a Arc<CudaModule>,
    blas: Option<&'a Arc<Mutex<CudaBlas>>>,
    solver: Option<&'a Arc<Mutex<DnHandle>>>,
}

fn cuda_remaining_use_counts(plan: &TensorExecutionPlan) -> Vec<usize> {
    let mut counts = vec![0; plan.nodes.len()];
    for node in &plan.nodes {
        for input in tensor_op_inputs(&node.op) {
            counts[input] += 1;
        }
    }
    counts
}

fn cuda_matmul_bias_tanh_epilogue(
    plan: &TensorExecutionPlan,
) -> Option<CudaMatmulBiasTanhEpilogue> {
    let TensorOp::Tanh { input: add } = plan.nodes.get(plan.output_node_id)?.op else {
        return None;
    };
    let TensorOp::Add { lhs, rhs } = plan.nodes.get(add)?.op else {
        return None;
    };
    let (matmul, bias) = match (&plan.nodes.get(lhs)?.op, &plan.nodes.get(rhs)?.op) {
        (TensorOp::Matmul { .. }, _) => (lhs, rhs),
        (_, TensorOp::Matmul { .. }) => (rhs, lhs),
        _ => return None,
    };
    let TensorOp::Matmul { lhs, rhs } = plan.nodes.get(matmul)?.op else {
        return None;
    };
    let shape = &plan.nodes[plan.output_node_id].shape;
    let lhs_shape = &plan.nodes[lhs].shape;
    let rhs_shape = &plan.nodes[rhs].shape;
    if shape.len() != 2
        || lhs_shape.len() != 2
        || rhs_shape.len() != 2
        || plan.nodes[add].shape != *shape
        || plan.nodes[matmul].shape != *shape
        || plan.nodes[bias].shape != [1, shape[1]]
    {
        return None;
    }
    Some(CudaMatmulBiasTanhEpilogue {
        lhs,
        rhs,
        bias,
        rows: shape[0],
        inner: lhs_shape[1],
        cols: shape[1],
    })
}

fn take_cuda_buffer(
    stream: &Arc<CudaStream>,
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    count: usize,
    node_id: usize,
) -> Result<CudaSlice<f32>, String> {
    if let Some(buffer) = free_buffers
        .get_mut(&count)
        .and_then(|buffers| buffers.pop())
    {
        return Ok(buffer);
    }
    stream
        .alloc_zeros::<f32>(count)
        .map_err(|error| format!("failed to allocate CUDA node {node_id}: {error:?}"))
}

fn upload_cuda_input(
    stream: &Arc<CudaStream>,
    slot: &mut Option<CudaSlice<f32>>,
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    host: &[f32],
    name: &str,
) -> Result<(), String> {
    if slot.is_none() {
        *slot = Some(take_cuda_buffer(
            stream,
            free_buffers,
            host.len(),
            usize::MAX,
        )?);
    }
    stream
        .memcpy_htod(
            host,
            slot.as_mut()
                .expect("CUDA input buffer was allocated above"),
        )
        .map_err(|error| format!("failed to update CUDA input {name:?}: {error:?}"))
}

fn release_cuda_value(
    values: &mut [Option<CudaSlice<f32>>],
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    node_id: usize,
) -> Result<(), String> {
    let slot = values
        .get_mut(node_id)
        .ok_or_else(|| format!("CUDA node {node_id} is missing its buffer slot"))?;
    if let Some(buffer) = slot.take() {
        free_buffers.entry(buffer.len()).or_default().push(buffer);
    }
    Ok(())
}

fn release_dead_cuda_values(
    plan: &TensorExecutionPlan,
    node_id: usize,
    values: &mut [Option<CudaSlice<f32>>],
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    retained_inputs: &BTreeSet<String>,
    remaining_uses: &mut [usize],
) -> Result<(), String> {
    for input_id in tensor_op_inputs(&plan.nodes[node_id].op) {
        let remaining = remaining_uses
            .get_mut(input_id)
            .ok_or_else(|| format!("CUDA input node {input_id} does not exist"))?;
        *remaining = remaining
            .checked_sub(1)
            .ok_or_else(|| format!("CUDA input node {input_id} has an invalid use count"))?;
        if *remaining != 0 || input_id == plan.output_node_id {
            continue;
        }
        if matches!(
            &plan.nodes[input_id].op,
            TensorOp::Input { name } if retained_inputs.contains(name)
        ) {
            continue;
        }
        release_cuda_value(values, free_buffers, input_id)?;
    }
    Ok(())
}

fn execute_cuda_device_program(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
    runtime: CudaProgramRuntime<'_>,
    values: &mut Vec<Option<CudaSlice<f32>>>,
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    retained_inputs: &BTreeSet<String>,
    copy_output: bool,
) -> Result<Option<DynamicTensor>, String> {
    let CudaProgramRuntime {
        stream,
        module,
        blas,
        solver,
    } = runtime;
    validate_cuda_program_inputs(plan, inputs)?;
    if values.len() != plan.nodes.len() {
        *values = std::iter::repeat_with(|| None)
            .take(plan.nodes.len())
            .collect();
    }
    let mut remaining_uses = cuda_remaining_use_counts(plan);

    for (node_id, node) in plan.nodes.iter().enumerate() {
        let count = element_count(&node.shape)?;
        let launch_count = u32::try_from(count)
            .map_err(|_| format!("CUDA node {node_id} launch exceeds u32 element count"))?;
        match &node.op {
            TensorOp::Input { name } => {
                let host = inputs[name]
                    .data()
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                let slot = values
                    .get_mut(node_id)
                    .ok_or_else(|| format!("CUDA input node {node_id} is missing its buffer"))?;
                if retained_inputs.contains(name) && slot.is_some() {
                    continue;
                }
                upload_cuda_input(stream, slot, free_buffers, &host, name)?;
                continue;
            }
            TensorOp::Solve { matrix, rhs } => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after
                    .first_mut()
                    .ok_or_else(|| format!("CUDA node {node_id} is missing its buffer"))?;
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                let output = slot
                    .as_mut()
                    .ok_or_else(|| format!("CUDA node {node_id} buffer was not allocated"))?;
                let solver = solver.ok_or_else(|| {
                    "CUDA solve requires CUSOLVER, but libcusolver could not be loaded".to_string()
                })?;
                launch_cusolver_rank_two_solve(
                    stream,
                    module,
                    solver,
                    cuda_value(before, *matrix)?,
                    cuda_value(before, *rhs)?,
                    output,
                    plan.nodes[*matrix].shape[0],
                    plan.nodes[*rhs].shape[1],
                )?;
            }
            TensorOp::ScalarConstant { .. }
            | TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Div { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Where { .. }
            | TensorOp::Tanh { .. }
            | TensorOp::Exp { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Sin { .. }
            | TensorOp::Cos { .. }
            | TensorOp::Powi { .. }
            | TensorOp::Log { .. }
            | TensorOp::Matmul { .. }
            | TensorOp::Sum { .. }
            | TensorOp::Mean { .. }
            | TensorOp::SumAxis { .. }
            | TensorOp::MeanAxis { .. }
            | TensorOp::Transpose { .. }
            | TensorOp::Concat { .. }
            | TensorOp::Slice { .. }
            | TensorOp::PadSlice { .. }
            | TensorOp::Broadcast { .. } => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after
                    .first_mut()
                    .ok_or_else(|| format!("CUDA node {node_id} is missing its buffer"))?;
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                let output = slot
                    .as_mut()
                    .ok_or_else(|| format!("CUDA node {node_id} buffer was not allocated"))?;
                let kernel = module
                    .load_function(&cuda_node_function_name(node_id))
                    .map_err(|error| {
                        format!("failed to load CUDA node {node_id} kernel: {error:?}")
                    })?;
                launch_cuda_node(
                    stream,
                    CudaNodeLaunch {
                        kernel: &kernel,
                        output,
                        values: before,
                        op: &node.op,
                        plan,
                        count,
                        launch_count,
                        output_shape: &node.shape,
                        blas,
                    },
                )?;
                release_dead_cuda_values(
                    plan,
                    node_id,
                    values,
                    free_buffers,
                    retained_inputs,
                    &mut remaining_uses,
                )?;
                continue;
            }
            TensorOp::Reshape { input } => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let input = cuda_value(before, *input)?;
                let slot = current_and_after
                    .first_mut()
                    .ok_or_else(|| format!("CUDA reshape node {node_id} is missing its buffer"))?;
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                stream
                    .memcpy_dtod(input, slot.as_mut().expect("allocated above"))
                    .map_err(|error| {
                        format!("failed to copy CUDA reshape node {node_id}: {error:?}")
                    })?;
                release_dead_cuda_values(
                    plan,
                    node_id,
                    values,
                    free_buffers,
                    retained_inputs,
                    &mut remaining_uses,
                )?;
                continue;
            }
        };
    }

    if !copy_output {
        return Ok(None);
    }
    let output_shape = plan.output_shape()?;
    let output = values
        .get(plan.output_node_id)
        .and_then(Option::as_ref)
        .ok_or_else(|| "CUDA device program output is missing".to_string())?;
    let data = stream
        .clone_dtoh(output)
        .map_err(|error| format!("failed to copy CUDA program output to host: {error:?}"))?;
    DynamicTensor::new(output_shape, data.into_iter().map(f64::from).collect()).map(Some)
}

#[allow(clippy::too_many_arguments)]
fn execute_cuda_fused_elementwise_program(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    values: &mut Vec<Option<CudaSlice<f32>>>,
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    retained_inputs: &BTreeSet<String>,
    copy_output: bool,
) -> Result<Option<DynamicTensor>, String> {
    validate_cuda_program_inputs(plan, inputs)?;
    if values.len() != plan.nodes.len() {
        *values = std::iter::repeat_with(|| None)
            .take(plan.nodes.len())
            .collect();
    }

    for (node_id, node) in plan.nodes.iter().enumerate() {
        let TensorOp::Input { name } = &node.op else {
            continue;
        };
        let host = inputs[name]
            .data()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let slot = values
            .get_mut(node_id)
            .ok_or_else(|| format!("CUDA input node {node_id} is missing its buffer"))?;
        if retained_inputs.contains(name) && slot.is_some() {
            continue;
        }
        upload_cuda_input(stream, slot, free_buffers, &host, name)?;
    }

    let output_node_id = plan.output_node_id;
    let output_shape = plan.output_shape()?;
    let count = element_count(&output_shape)?;
    let launch_count = u32::try_from(count)
        .map_err(|_| "CUDA fused elementwise launch exceeds u32 element count".to_string())?;
    let data = {
        let (before_output, output_and_after) = values.split_at_mut(output_node_id);
        let output = output_and_after
            .first_mut()
            .ok_or_else(|| "CUDA fused elementwise output slot is missing".to_string())?;
        if output.is_none() {
            *output = Some(take_cuda_buffer(
                stream,
                free_buffers,
                count,
                output_node_id,
            )?);
        }
        let output = output
            .as_mut()
            .ok_or_else(|| "CUDA fused elementwise output was not allocated".to_string())?;
        let kernel = module
            .load_function("nabla_fused_elementwise")
            .map_err(|error| format!("failed to load CUDA fused elementwise kernel: {error:?}"))?;
        let mut launch = stream.launch_builder(&kernel);
        for (node_id, node) in plan.nodes.iter().enumerate() {
            if matches!(&node.op, TensorOp::Input { .. }) {
                launch.arg(cuda_value(before_output, node_id)?);
            }
        }
        let device_count = u64::try_from(count)
            .map_err(|_| "CUDA fused elementwise count exceeds u64".to_string())?;
        launch.arg(&mut *output);
        launch.arg(&device_count);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(launch_count))
                .map_err(|error| {
                    format!("failed to launch CUDA fused elementwise kernel: {error:?}")
                })?;
        }
        if copy_output {
            Some(
                stream.clone_dtoh(output).map_err(|error| {
                    format!("failed to copy CUDA fused output to host: {error:?}")
                })?,
            )
        } else {
            None
        }
    };
    for (node_id, node) in plan.nodes.iter().enumerate() {
        if matches!(&node.op, TensorOp::Input { name } if !retained_inputs.contains(name)) {
            release_cuda_value(values, free_buffers, node_id)?;
        }
    }
    data.map(|data| DynamicTensor::new(output_shape, data.into_iter().map(f64::from).collect()))
        .transpose()
}

#[allow(clippy::too_many_arguments)]
fn execute_cuda_matmul_bias_tanh_program(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    values: &mut Vec<Option<CudaSlice<f32>>>,
    free_buffers: &mut BTreeMap<usize, Vec<CudaSlice<f32>>>,
    retained_inputs: &BTreeSet<String>,
    copy_output: bool,
    epilogue: CudaMatmulBiasTanhEpilogue,
) -> Result<Option<DynamicTensor>, String> {
    validate_cuda_program_inputs(plan, inputs)?;
    if values.len() != plan.nodes.len() {
        *values = std::iter::repeat_with(|| None)
            .take(plan.nodes.len())
            .collect();
    }
    for (node_id, node) in plan.nodes.iter().enumerate() {
        let TensorOp::Input { name } = &node.op else {
            continue;
        };
        let slot = &mut values[node_id];
        if retained_inputs.contains(name) && slot.is_some() {
            continue;
        }
        let host = inputs[name]
            .data()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        upload_cuda_input(stream, slot, free_buffers, &host, name)?;
    }
    let output_node_id = plan.output_node_id;
    let count = element_count(&plan.output_shape()?)?;
    let data = {
        let (before, current) = values.split_at_mut(output_node_id);
        let output = current
            .first_mut()
            .ok_or_else(|| "CUDA epilogue output slot is missing".to_string())?;
        if output.is_none() {
            *output = Some(take_cuda_buffer(
                stream,
                free_buffers,
                count,
                output_node_id,
            )?);
        }
        let output = output
            .as_mut()
            .expect("CUDA epilogue output was allocated above");
        let kernel = module
            .load_function("nabla_matmul_bias_tanh")
            .map_err(|error| format!("failed to load CUDA matmul epilogue kernel: {error:?}"))?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(cuda_value(before, epilogue.lhs)?);
        launch.arg(cuda_value(before, epilogue.rhs)?);
        launch.arg(cuda_value(before, epilogue.bias)?);
        launch.arg(&mut *output);
        let rows = epilogue.rows as u64;
        let inner = epilogue.inner as u64;
        let cols = epilogue.cols as u64;
        launch.arg(&rows);
        launch.arg(&inner);
        launch.arg(&cols);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(
                    u32::try_from(count)
                        .map_err(|_| "CUDA epilogue launch exceeds u32".to_string())?,
                ))
                .map_err(|error| format!("failed to launch CUDA matmul epilogue: {error:?}"))?;
        }
        if copy_output {
            Some(
                stream
                    .clone_dtoh(output)
                    .map_err(|error| format!("failed to copy CUDA epilogue output: {error:?}"))?,
            )
        } else {
            None
        }
    };
    for (node_id, node) in plan.nodes.iter().enumerate() {
        if matches!(&node.op, TensorOp::Input { name } if !retained_inputs.contains(name)) {
            release_cuda_value(values, free_buffers, node_id)?;
        }
    }
    data.map(|data| {
        DynamicTensor::new(
            plan.output_shape()?,
            data.into_iter().map(f64::from).collect(),
        )
    })
    .transpose()
}

impl CudaBackend {
    fn execute_rank_two_matmul(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
        lhs_name: &str,
        rhs_name: &str,
    ) -> Result<DynamicTensor, String> {
        let lhs = inputs
            .get(lhs_name)
            .ok_or_else(|| format!("missing input {lhs_name:?}"))?;
        let rhs = inputs
            .get(rhs_name)
            .ok_or_else(|| format!("missing input {rhs_name:?}"))?;
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            return Err("CUDA direct matmul requires rank-two input tensors".to_string());
        }
        if lhs_shape[1] != rhs_shape[0] {
            return Err(format!(
                "CUDA matmul inner dimensions {} and {} differ",
                lhs_shape[1], rhs_shape[0]
            ));
        }
        let output_shape = plan.output_shape()?;
        let rows = lhs_shape[0];
        let inner = lhs_shape[1];
        let cols = rhs_shape[1];
        if output_shape != [rows, cols] {
            return Err(format!(
                "CUDA direct matmul output shape {:?} does not match [{rows}, {cols}]",
                output_shape
            ));
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| "CUDA matmul output element count overflows usize".to_string())?;
        let grid_x = u32::try_from(cols.div_ceil(CUDA_MATMUL_TILE))
            .map_err(|_| "CUDA matmul grid width exceeds u32".to_string())?;
        let grid_y = u32::try_from(rows.div_ceil(CUDA_MATMUL_TILE))
            .map_err(|_| "CUDA matmul grid height exceeds u32".to_string())?;
        let rows = u64::try_from(rows).map_err(|_| "CUDA matmul rows exceed u64".to_string())?;
        let inner =
            u64::try_from(inner).map_err(|_| "CUDA matmul inner size exceeds u64".to_string())?;
        let cols = u64::try_from(cols).map_err(|_| "CUDA matmul columns exceed u64".to_string())?;
        let lhs_host = lhs
            .data()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs
            .data()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

        let context = CudaContext::new(self.device_ordinal)
            .map_err(|error| format!("failed to create CUDA context: {error:?}"))?;
        let stream = context.default_stream();
        let ptx = compile_ptx(CUDA_RANK_TWO_MATMUL_SOURCE).map_err(|error| {
            format!("failed to compile CUDA matmul kernel with NVRTC: {error:?}")
        })?;
        let module = context
            .load_module(ptx)
            .map_err(|error| format!("failed to load CUDA matmul module: {error:?}"))?;
        let kernel = module
            .load_function("nabla_rank_two_matmul")
            .map_err(|error| format!("failed to load CUDA matmul kernel: {error:?}"))?;
        let lhs_device = stream
            .clone_htod(&lhs_host)
            .map_err(|error| format!("failed to copy CUDA matmul lhs: {error:?}"))?;
        let rhs_device = stream
            .clone_htod(&rhs_host)
            .map_err(|error| format!("failed to copy CUDA matmul rhs: {error:?}"))?;
        let mut output_device = stream
            .alloc_zeros::<f32>(count)
            .map_err(|error| format!("failed to allocate CUDA matmul output: {error:?}"))?;

        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&lhs_device);
        launch.arg(&rhs_device);
        launch.arg(&mut output_device);
        launch.arg(&rows);
        launch.arg(&inner);
        launch.arg(&cols);
        unsafe {
            launch
                .launch(LaunchConfig {
                    grid_dim: (grid_x, grid_y, 1),
                    block_dim: (CUDA_MATMUL_BLOCK, CUDA_MATMUL_BLOCK, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| format!("failed to launch CUDA matmul kernel: {error:?}"))?;
        }
        let data = stream
            .clone_dtoh(&output_device)
            .map_err(|error| format!("failed to copy CUDA matmul output to host: {error:?}"))?;
        DynamicTensor::new(output_shape, data.into_iter().map(f64::from).collect())
    }
}

fn validate_cuda_program_inputs(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<(), String> {
    for node in &plan.nodes {
        if let TensorOp::Input { name } = &node.op {
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
        }
    }
    Ok(())
}

struct CudaNodeLaunch<'a> {
    kernel: &'a CudaFunction,
    output: &'a mut CudaSlice<f32>,
    values: &'a [Option<CudaSlice<f32>>],
    op: &'a TensorOp,
    plan: &'a TensorExecutionPlan,
    count: usize,
    launch_count: u32,
    output_shape: &'a [usize],
    blas: Option<&'a Arc<Mutex<CudaBlas>>>,
}

fn launch_cuda_node(stream: &Arc<CudaStream>, request: CudaNodeLaunch<'_>) -> Result<(), String> {
    let CudaNodeLaunch {
        kernel,
        output,
        values,
        op,
        plan,
        count,
        launch_count,
        output_shape,
        blas,
    } = request;
    if let TensorOp::Matmul { lhs, rhs } = op {
        if let Some(blas) = blas {
            if use_cublas_rank_two_matmul(plan, *lhs, *rhs, output_shape) {
                return launch_cublas_rank_two_matmul(
                    blas,
                    cuda_value(values, *lhs)?,
                    cuda_value(values, *rhs)?,
                    output,
                    output_shape[0],
                    plan.nodes[*lhs].shape[1],
                    output_shape[1],
                );
            }
            if let Some(dimensions) =
                cublas_batched_matmul_dimensions(plan, *lhs, *rhs, output_shape)
            {
                return launch_cublas_batched_matmul(
                    blas,
                    cuda_value(values, *lhs)?,
                    cuda_value(values, *rhs)?,
                    output,
                    dimensions,
                );
            }
        }
    }
    let count = count as u64;
    let dimensions = match op {
        TensorOp::Matmul { lhs, .. } => Some([
            output_shape[output_shape.len() - 2] as u64,
            plan.nodes[*lhs].shape[plan.nodes[*lhs].shape.len() - 1] as u64,
            output_shape[output_shape.len() - 1] as u64,
        ]),
        TensorOp::Sum { input } | TensorOp::Mean { input } => {
            Some([cuda_value(values, *input)?.len() as u64, 0, 0])
        }
        _ => None,
    };
    if matches!(op, TensorOp::Sum { .. } | TensorOp::Mean { .. }) {
        stream
            .memcpy_htod(&[0.0f32], output)
            .map_err(|error| format!("failed to clear CUDA reduction output: {error:?}"))?;
    }
    let mut launch = stream.launch_builder(kernel);
    match op {
        TensorOp::ScalarConstant { .. } => {
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Div { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Greater { lhs, rhs } => {
            launch.arg(cuda_value(values, *lhs)?);
            launch.arg(cuda_value(values, *rhs)?);
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => {
            launch.arg(cuda_value(values, *condition)?);
            launch.arg(cuda_value(values, *on_true)?);
            launch.arg(cuda_value(values, *on_false)?);
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Tanh { input }
        | TensorOp::Exp { input }
        | TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Sin { input }
        | TensorOp::Cos { input }
        | TensorOp::Powi { input, .. }
        | TensorOp::Log { input }
        | TensorOp::Transpose { input, .. } => {
            launch.arg(cuda_value(values, *input)?);
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Concat { inputs, .. } => {
            for input in inputs {
                launch.arg(cuda_value(values, *input)?);
            }
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Slice { input, .. } | TensorOp::PadSlice { input, .. } => {
            launch.arg(cuda_value(values, *input)?);
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Broadcast { input } => {
            launch.arg(cuda_value(values, *input)?);
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::Matmul { lhs, rhs } => {
            let dimensions = dimensions
                .as_ref()
                .ok_or_else(|| "CUDA matmul launch dimensions are missing".to_string())?;
            launch.arg(cuda_value(values, *lhs)?);
            launch.arg(cuda_value(values, *rhs)?);
            launch.arg(output);
            launch.arg(&dimensions[0]);
            launch.arg(&dimensions[1]);
            launch.arg(&dimensions[2]);
        }
        TensorOp::Sum { input } | TensorOp::Mean { input } => {
            let dimensions = dimensions
                .as_ref()
                .ok_or_else(|| "CUDA reduction launch dimensions are missing".to_string())?;
            launch.arg(cuda_value(values, *input)?);
            launch.arg(output);
            launch.arg(&dimensions[0]);
        }
        TensorOp::SumAxis { input, .. } | TensorOp::MeanAxis { input, .. } => {
            launch.arg(cuda_value(values, *input)?);
            launch.arg(output);
            launch.arg(&count);
        }
        _ => {
            return Err(format!(
                "invalid CUDA device-program op {}",
                cuda_op_name(op)
            ))
        }
    }
    let config = match op {
        TensorOp::Matmul { lhs, rhs }
            if use_tiled_rank_two_matmul(plan, *lhs, *rhs, output_shape) =>
        {
            let dimensions = dimensions
                .as_ref()
                .ok_or_else(|| "CUDA matmul launch dimensions are missing".to_string())?;
            LaunchConfig {
                grid_dim: (
                    u32::try_from(dimensions[2].div_ceil(CUDA_MATMUL_TILE_U64))
                        .map_err(|_| "CUDA matmul grid width exceeds u32".to_string())?,
                    u32::try_from(dimensions[0].div_ceil(CUDA_MATMUL_TILE_U64))
                        .map_err(|_| "CUDA matmul grid height exceeds u32".to_string())?,
                    1,
                ),
                block_dim: (CUDA_MATMUL_BLOCK, CUDA_MATMUL_BLOCK, 1),
                shared_mem_bytes: 0,
            }
        }
        TensorOp::Sum { .. } | TensorOp::Mean { .. } => {
            let dimensions = dimensions
                .as_ref()
                .ok_or_else(|| "CUDA reduction launch dimensions are missing".to_string())?;
            LaunchConfig {
                grid_dim: (
                    u32::try_from(dimensions[0].div_ceil(CUDA_REDUCTION_BLOCK as u64))
                        .map_err(|_| "CUDA reduction grid size exceeds u32".to_string())?,
                    1,
                    1,
                ),
                block_dim: (CUDA_REDUCTION_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            }
        }
        _ => LaunchConfig::for_num_elems(launch_count),
    };
    unsafe {
        launch.launch(config).map_err(|error| {
            format!("failed to launch CUDA node {}: {error:?}", cuda_op_name(op))
        })?;
    }
    Ok(())
}

fn cuda_value(
    values: &[Option<CudaSlice<f32>>],
    node_id: usize,
) -> Result<&CudaSlice<f32>, String> {
    values
        .get(node_id)
        .and_then(Option::as_ref)
        .ok_or_else(|| format!("CUDA operand node {node_id} is missing its buffer"))
}

fn use_tiled_rank_two_matmul(
    plan: &TensorExecutionPlan,
    lhs: usize,
    rhs: usize,
    output_shape: &[usize],
) -> bool {
    let lhs_shape = &plan.nodes[lhs].shape;
    let rhs_shape = &plan.nodes[rhs].shape;
    lhs_shape.len() == 2
        && rhs_shape.len() == 2
        && output_shape.len() == 2
        && lhs_shape[0] >= CUDA_MATMUL_TILE
        && lhs_shape[1] >= CUDA_MATMUL_TILE
        && rhs_shape[1] >= CUDA_MATMUL_TILE
}

fn use_cublas_rank_two_matmul(
    plan: &TensorExecutionPlan,
    lhs: usize,
    rhs: usize,
    output_shape: &[usize],
) -> bool {
    let lhs_shape = &plan.nodes[lhs].shape;
    let rhs_shape = &plan.nodes[rhs].shape;
    lhs_shape.len() == 2 && rhs_shape.len() == 2 && output_shape.len() == 2
}

fn cuda_blas(stream: Arc<CudaStream>) -> Result<Option<Arc<Mutex<CudaBlas>>>, String> {
    let library_name = libloading::library_filename("cublas");
    let library = unsafe { libloading::Library::new(&library_name) };
    if library.is_err() {
        return Ok(None);
    }
    drop(library);
    CudaBlas::new(stream)
        .map(|blas| Some(Arc::new(Mutex::new(blas))))
        .map_err(|error| format!("failed to initialize cuBLAS from {library_name:?}: {error:?}"))
}

fn cuda_solver(stream: Arc<CudaStream>) -> Result<Option<Arc<Mutex<DnHandle>>>, String> {
    let library_name = libloading::library_filename("cusolver");
    let library = unsafe { libloading::Library::new(&library_name) };
    if library.is_err() {
        return Ok(None);
    }
    drop(library);
    DnHandle::new(stream)
        .map(|solver| Some(Arc::new(Mutex::new(solver))))
        .map_err(|error| format!("failed to initialize CUSOLVER from {library_name:?}: {error:?}"))
}

fn launch_cuda_transpose_copy(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    columns: usize,
) -> Result<(), String> {
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| "CUDA transpose element count overflows usize".to_string())?;
    let count =
        u32::try_from(count).map_err(|_| "CUDA transpose element count exceeds u32".to_string())?;
    let rows = u64::try_from(rows).map_err(|_| "CUDA transpose rows exceed u64".to_string())?;
    let columns =
        u64::try_from(columns).map_err(|_| "CUDA transpose columns exceed u64".to_string())?;
    let kernel = module
        .load_function("nabla_transpose_copy")
        .map_err(|error| format!("failed to load CUDA transpose kernel: {error:?}"))?;
    let mut launch = stream.launch_builder(&kernel);
    launch.arg(input);
    launch.arg(output);
    launch.arg(&rows);
    launch.arg(&columns);
    unsafe {
        launch
            .launch(LaunchConfig::for_num_elems(count))
            .map_err(|error| format!("failed to launch CUDA transpose kernel: {error:?}"))?;
    }
    Ok(())
}

fn launch_cusolver_rank_two_solve(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    solver: &Arc<Mutex<DnHandle>>,
    matrix: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
    rhs_columns: usize,
) -> Result<(), String> {
    let n_i32 = i32::try_from(n).map_err(|_| "CUSOLVER solve dimension exceeds i32".to_string())?;
    let rhs_columns_i32 = i32::try_from(rhs_columns)
        .map_err(|_| "CUSOLVER solve right-hand-side columns exceed i32".to_string())?;
    let mut factor = unsafe { stream.alloc::<f32>(n * n) }
        .map_err(|error| format!("failed to allocate CUSOLVER factor buffer: {error:?}"))?;
    let mut column_rhs = unsafe { stream.alloc::<f32>(n * rhs_columns) }.map_err(|error| {
        format!("failed to allocate CUSOLVER right-hand-side buffer: {error:?}")
    })?;
    launch_cuda_transpose_copy(stream, module, matrix, &mut factor, n, n)?;
    launch_cuda_transpose_copy(stream, module, rhs, &mut column_rhs, n, rhs_columns)?;
    let mut pivots = unsafe { stream.alloc::<i32>(n) }
        .map_err(|error| format!("failed to allocate CUSOLVER pivot buffer: {error:?}"))?;
    let mut info = stream
        .alloc_zeros::<i32>(1)
        .map_err(|error| format!("failed to allocate CUSOLVER status buffer: {error:?}"))?;
    let solver = solver
        .lock()
        .map_err(|_| "CUSOLVER handle lock is poisoned".to_string())?;
    {
        let (factor_ptr, _factor_read) = factor.device_ptr_mut(stream);
        let (rhs_ptr, _rhs_read) = column_rhs.device_ptr_mut(stream);
        let (pivot_ptr, _pivot_read) = pivots.device_ptr_mut(stream);
        let (info_ptr, _info_read) = info.device_ptr_mut(stream);
        let mut workspace_elements = 0_i32;
        unsafe {
            cusolver_sys::cusolverDnSgetrf_bufferSize(
                solver.cu(),
                n_i32,
                n_i32,
                factor_ptr as *mut f32,
                n_i32,
                &mut workspace_elements,
            )
            .result()
            .map_err(|error| format!("CUSOLVER Sgetrf workspace query failed: {error:?}"))?;
        }
        let mut workspace = unsafe { stream.alloc::<f32>(workspace_elements as usize) }
            .map_err(|error| format!("failed to allocate CUSOLVER workspace: {error:?}"))?;
        let (workspace_ptr, _workspace_read) = workspace.device_ptr_mut(stream);
        unsafe {
            cusolver_sys::cusolverDnSgetrf(
                solver.cu(),
                n_i32,
                n_i32,
                factor_ptr as *mut f32,
                n_i32,
                workspace_ptr as *mut f32,
                pivot_ptr as *mut i32,
                info_ptr as *mut i32,
            )
            .result()
            .map_err(|error| format!("CUSOLVER Sgetrf failed: {error:?}"))?;
            cusolver_sys::cusolverDnSgetrs(
                solver.cu(),
                cusolver_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                rhs_columns_i32,
                factor_ptr as *const f32,
                n_i32,
                pivot_ptr as *const i32,
                rhs_ptr as *mut f32,
                n_i32,
                info_ptr as *mut i32,
            )
            .result()
            .map_err(|error| format!("CUSOLVER Sgetrs failed: {error:?}"))?;
        }
    }
    let mut host_info = [0_i32; 1];
    stream
        .memcpy_dtoh(&info, &mut host_info)
        .map_err(|error| format!("failed to read CUSOLVER status: {error:?}"))?;
    if host_info[0] != 0 {
        return Err(format!(
            "CUSOLVER solve failed with devInfo={}",
            host_info[0]
        ));
    }
    launch_cuda_transpose_copy(stream, module, &column_rhs, output, rhs_columns, n)
}

fn launch_cublas_rank_two_matmul(
    blas: &Arc<Mutex<CudaBlas>>,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    let rows = i32::try_from(rows).map_err(|_| "cuBLAS matmul rows exceed i32".to_string())?;
    let inner =
        i32::try_from(inner).map_err(|_| "cuBLAS matmul inner size exceeds i32".to_string())?;
    let cols = i32::try_from(cols).map_err(|_| "cuBLAS matmul columns exceed i32".to_string())?;
    let config = GemmConfig {
        // Row-major C = A * B is column-major C^T = B^T * A^T.
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: cols,
        n: rows,
        k: inner,
        alpha: 1.0_f32,
        lda: cols,
        ldb: inner,
        beta: 0.0_f32,
        ldc: cols,
    };
    let blas = blas
        .lock()
        .map_err(|_| "cuBLAS handle lock is poisoned".to_string())?;
    unsafe { blas.gemm(config, rhs, lhs, output) }
        .map_err(|error| format!("cuBLAS SGEMM failed: {error:?}"))
}

#[derive(Clone, Copy, Debug)]
struct CublasBatchedMatmulDimensions {
    rows: usize,
    inner: usize,
    cols: usize,
    batch_count: usize,
    lhs_stride: usize,
    rhs_stride: usize,
    output_stride: usize,
}

fn cublas_batched_matmul_dimensions(
    plan: &TensorExecutionPlan,
    lhs: usize,
    rhs: usize,
    output_shape: &[usize],
) -> Option<CublasBatchedMatmulDimensions> {
    let lhs_shape = &plan.nodes[lhs].shape;
    let rhs_shape = &plan.nodes[rhs].shape;
    if lhs_shape.len() < 2 || rhs_shape.len() < 2 || output_shape.len() < 3 {
        return None;
    }
    let batch_shape = &output_shape[..output_shape.len() - 2];
    let batch_count = element_count(batch_shape).ok()?;
    let rows = *lhs_shape.get(lhs_shape.len() - 2)?;
    let inner = *lhs_shape.last()?;
    let cols = *rhs_shape.last()?;
    let lhs_stride = cublas_batch_stride(lhs_shape, batch_shape)?;
    let rhs_stride = cublas_batch_stride(rhs_shape, batch_shape)?;
    let output_stride = rows.checked_mul(cols)?;
    Some(CublasBatchedMatmulDimensions {
        rows,
        inner,
        cols,
        batch_count,
        lhs_stride,
        rhs_stride,
        output_stride,
    })
}

fn cublas_batch_stride(input_shape: &[usize], output_batch_shape: &[usize]) -> Option<usize> {
    let input_batch_shape = &input_shape[..input_shape.len() - 2];
    let output_offset = output_batch_shape
        .len()
        .checked_sub(input_batch_shape.len())?;
    let mut all_broadcast = true;
    for (&input_extent, &output_extent) in input_batch_shape
        .iter()
        .zip(output_batch_shape[output_offset..].iter())
    {
        if input_extent != output_extent && input_extent != 1 {
            return None;
        }
        all_broadcast &= input_extent == 1;
    }
    if all_broadcast {
        return Some(0);
    }
    if input_batch_shape != output_batch_shape {
        return None;
    }
    input_shape[input_shape.len() - 2].checked_mul(*input_shape.last()?)
}

fn launch_cublas_batched_matmul(
    blas: &Arc<Mutex<CudaBlas>>,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    dimensions: CublasBatchedMatmulDimensions,
) -> Result<(), String> {
    let rows = i32::try_from(dimensions.rows)
        .map_err(|_| "cuBLAS batched matmul rows exceed i32".to_string())?;
    let inner = i32::try_from(dimensions.inner)
        .map_err(|_| "cuBLAS batched matmul inner size exceeds i32".to_string())?;
    let cols = i32::try_from(dimensions.cols)
        .map_err(|_| "cuBLAS batched matmul columns exceed i32".to_string())?;
    let batch_size = i32::try_from(dimensions.batch_count)
        .map_err(|_| "cuBLAS batched matmul batch count exceeds i32".to_string())?;
    let config = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: cols,
            n: rows,
            k: inner,
            alpha: 1.0_f32,
            lda: cols,
            ldb: inner,
            beta: 0.0_f32,
            ldc: cols,
        },
        batch_size,
        stride_a: i64::try_from(dimensions.rhs_stride)
            .map_err(|_| "cuBLAS rhs stride exceeds i64".to_string())?,
        stride_b: i64::try_from(dimensions.lhs_stride)
            .map_err(|_| "cuBLAS lhs stride exceeds i64".to_string())?,
        stride_c: i64::try_from(dimensions.output_stride)
            .map_err(|_| "cuBLAS output stride exceeds i64".to_string())?,
    };
    let blas = blas
        .lock()
        .map_err(|_| "cuBLAS handle lock is poisoned".to_string())?;
    unsafe { blas.gemm_strided_batched(config, rhs, lhs, output) }
        .map_err(|error| format!("cuBLAS strided-batched SGEMM failed: {error:?}"))
}

fn cuda_value_mut(
    values: &mut [Option<CudaSlice<f32>>],
    node_id: usize,
) -> Result<&mut CudaSlice<f32>, String> {
    values
        .get_mut(node_id)
        .and_then(Option::as_mut)
        .ok_or_else(|| format!("CUDA operand node {node_id} is missing its buffer"))
}

fn input_node_id(plan: &TensorExecutionPlan, name: &str) -> Result<usize, String> {
    plan.nodes
        .iter()
        .position(
            |node| matches!(&node.op, TensorOp::Input { name: input_name } if input_name == name),
        )
        .ok_or_else(|| format!("CUDA plan has no input {name:?}"))
}

fn cuda_matmul_batch_offset_source(
    prefix: &str,
    output_batch_shape: &[usize],
    input_shape: &[usize],
) -> Result<String, String> {
    if input_shape.len() < 2 {
        return Err(format!(
            "CUDA matmul {prefix} input must have rank at least two"
        ));
    }
    let input_batch_shape = &input_shape[..input_shape.len() - 2];
    let input_batch_strides = contiguous_strides(input_batch_shape);
    let rank_offset = output_batch_shape
        .len()
        .checked_sub(input_batch_shape.len())
        .ok_or_else(|| format!("CUDA matmul {prefix} batch rank exceeds output batch rank"))?;
    let mut source = format!("unsigned long long {prefix}_remaining = batch_index; unsigned long long {prefix}_batch = 0ULL;");
    for output_axis in (0..output_batch_shape.len()).rev() {
        source.push_str(&format!(
            " unsigned long long {prefix}_coordinate_{output_axis} = {prefix}_remaining % {}ULL; {prefix}_remaining /= {}ULL;",
            output_batch_shape[output_axis], output_batch_shape[output_axis]
        ));
        if output_axis >= rank_offset {
            let input_axis = output_axis - rank_offset;
            if input_batch_shape[input_axis] != 1 {
                source.push_str(&format!(
                    " {prefix}_batch += {prefix}_coordinate_{output_axis} * {}ULL;",
                    input_batch_strides[input_axis]
                ));
            }
        }
    }
    Ok(source)
}

fn cuda_program_source(plan: &TensorExecutionPlan) -> Result<String, String> {
    let mut source = String::from(
        "__device__ __forceinline__ float nabla_powi(float base, unsigned int exponent) {\n\
    float result = 1.0f;\n\
    while (exponent != 0U) { if ((exponent & 1U) != 0U) result *= base; base *= base; exponent >>= 1U; }\n\
    return result;\n}\n\
extern \"C\" __global__ void nabla_sgd(float* parameter, const float* gradient, float learning_rate, unsigned long long count) {\n\
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (index < count) parameter[index] -= learning_rate * gradient[index];\n}\n",
    );
    source.push_str(
        "extern \"C\" __global__ void nabla_adam(float* parameter, const float* gradient, float* first_moment, float* second_moment, float learning_rate, float beta1, float beta2, float epsilon, float correction1, float correction2, unsigned long long count) {\n\
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (index < count) {\n\
        float gradient_value = gradient[index];\n\
        float first = beta1 * first_moment[index] + (1.0f - beta1) * gradient_value;\n\
        float second = beta2 * second_moment[index] + (1.0f - beta2) * gradient_value * gradient_value;\n\
        first_moment[index] = first;\n\
        second_moment[index] = second;\n\
        parameter[index] -= learning_rate * (first / correction1) / (sqrtf(second / correction2) + epsilon);\n\
    }\n}\n",
    );
    source.push_str(
        "extern \"C\" __global__ void nabla_transpose_copy(const float* input, float* output, unsigned long long rows, unsigned long long columns) {\n\\
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
    unsigned long long count = rows * columns;\n\\
    if (index < count) { unsigned long long row = index / columns; unsigned long long column = index % columns; output[column * rows + row] = input[index]; }\n}\n",
    );
    for (node_id, node) in plan.nodes.iter().enumerate() {
        let function = cuda_node_function_name(node_id);
        let count = element_count(&node.shape)?;
        let kernel = match &node.op {
            TensorOp::Input { .. } | TensorOp::Reshape { .. } => continue,
            TensorOp::ScalarConstant { value } if value.is_finite() => format!(
                "extern \"C\" __global__ void {function}(float* out, unsigned long long count) {{\n\
                    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                    if (index < count) out[index] = {};\n}}\n",
                cuda_float_literal(*value)
            ),
            TensorOp::ScalarConstant { .. } => {
                return Err("CUDA device program does not support non-finite constants".to_string())
            }
            TensorOp::Add { lhs, rhs }
            | TensorOp::Sub { lhs, rhs }
            | TensorOp::Div { lhs, rhs }
            | TensorOp::Mul { lhs, rhs }
            | TensorOp::Greater { lhs, rhs } => {
                let lhs_offset = cuda_offset_expression(&node.shape, &plan.nodes[*lhs].shape);
                let rhs_offset = cuda_offset_expression(&node.shape, &plan.nodes[*rhs].shape);
                let expression = match &node.op {
                    TensorOp::Add { .. } => format!("lhs[{lhs_offset}] + rhs[{rhs_offset}]"),
                    TensorOp::Sub { .. } => format!("lhs[{lhs_offset}] - rhs[{rhs_offset}]"),
                    TensorOp::Div { .. } => format!("lhs[{lhs_offset}] / rhs[{rhs_offset}]"),
                    TensorOp::Mul { .. } => format!("lhs[{lhs_offset}] * rhs[{rhs_offset}]"),
                    TensorOp::Greater { .. } => {
                        format!("lhs[{lhs_offset}] > rhs[{rhs_offset}] ? 1.0f : 0.0f")
                    }
                    _ => unreachable!(),
                };
                format!(
                    "extern \"C\" __global__ void {function}(const float* lhs, const float* rhs, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) out[index] = {expression};\n}}\n"
                )
            }
            TensorOp::Where { condition, on_true, on_false } => {
                let condition_offset = cuda_offset_expression(&node.shape, &plan.nodes[*condition].shape);
                let true_offset = cuda_offset_expression(&node.shape, &plan.nodes[*on_true].shape);
                let false_offset = cuda_offset_expression(&node.shape, &plan.nodes[*on_false].shape);
                format!(
                    "extern \"C\" __global__ void {function}(const float* condition, const float* on_true, const float* on_false, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) out[index] = condition[{condition_offset}] != 0.0f ? on_true[{true_offset}] : on_false[{false_offset}];\n}}\n"
                )
            }
            TensorOp::Tanh { input }
            | TensorOp::Exp { input }
            | TensorOp::Sqrt { input }
            | TensorOp::SqrtDerivative { input, .. }
            | TensorOp::Sin { input }
            | TensorOp::Cos { input }
            | TensorOp::Powi { input, .. }
            | TensorOp::Log { input } => {
                let expression = match &node.op {
                    TensorOp::Tanh { .. } => "tanhf(input[index])".to_string(),
                    TensorOp::Exp { .. } => "expf(input[index])".to_string(),
                    TensorOp::Sqrt { .. } => "sqrtf(input[index])".to_string(),
                    TensorOp::SqrtDerivative { order, .. } => {
                        let coefficient = cuda_float_literal(sqrt_derivative_coefficient(*order));
                        let exponent = cuda_float_literal(0.5 - *order as f64);
                        format!(
                            "input[index] == 0.0f ? 0.0f : {coefficient} * powf(input[index], {exponent})"
                        )
                    }
                    TensorOp::Sin { .. } => "sinf(input[index])".to_string(),
                    TensorOp::Cos { .. } => "cosf(input[index])".to_string(),
                    TensorOp::Powi { exponent, .. } => {
                        format!("nabla_powi(input[index], {exponent}U)")
                    }
                    TensorOp::Log { .. } => "logf(input[index])".to_string(),
                    _ => unreachable!(),
                };
                let input_count = element_count(&plan.nodes[*input].shape)?;
                if input_count != count {
                    return Err(format!(
                        "CUDA unary node {node_id} changes element count without reshape"
                    ));
                }
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) out[index] = {expression};\n}}\n"
                )
            }
            TensorOp::Matmul { lhs, rhs } => {
                let lhs_shape = &plan.nodes[*lhs].shape;
                let rhs_shape = &plan.nodes[*rhs].shape;
                if use_tiled_rank_two_matmul(plan, *lhs, *rhs, &node.shape) {
                    cuda_register_tiled_rank_two_matmul_source(&function)
                } else {
                let batch_shape = &node.shape[..node.shape.len() - 2];
                let lhs_batch = cuda_matmul_batch_offset_source("lhs", batch_shape, lhs_shape)?;
                let rhs_batch = cuda_matmul_batch_offset_source("rhs", batch_shape, rhs_shape)?;
                format!(
                    "extern \"C\" __global__ void {function}(const float* lhs, const float* rhs, float* out, unsigned long long rows, unsigned long long inner, unsigned long long cols) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index >= {count}ULL) return;\n\
                        unsigned long long matrix_size = rows * cols; unsigned long long batch_index = index / matrix_size;\n\
                        unsigned long long row = (index % matrix_size) / cols; unsigned long long col = index % cols;\n\
                        {lhs_batch}\n\
                        {rhs_batch}\n\
                        float value = 0.0f;\n\
                        for (unsigned long long k = 0; k < inner; ++k) value += lhs[lhs_batch * rows * inner + row * inner + k] * rhs[rhs_batch * inner * cols + k * cols + col];\n\
                        out[index] = value;\n}}\n"
                )
                }
            }
            TensorOp::Solve { .. } => String::new(),
            TensorOp::Sum { .. } | TensorOp::Mean { .. } => {
                let scale = if matches!(&node.op, TensorOp::Mean { .. }) {
                    " / (float)count".to_string()
                } else {
                    String::new()
                };
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        __shared__ float partial[256];\n\
                        unsigned int thread = threadIdx.x;\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + thread;\n\
                        partial[thread] = index < count ? input[index] : 0.0f;\n\
                        __syncthreads();\n\
                        for (unsigned int stride = 128U; stride > 0U; stride >>= 1U) {{\n\
                            if (thread < stride) partial[thread] += partial[thread + stride];\n\
                            __syncthreads();\n\
                        }}\n\
                        if (thread == 0U) atomicAdd(out, partial[0]{scale});\n}}\n"
                )
            }
            TensorOp::SumAxis { input, axis } | TensorOp::MeanAxis { input, axis } => {
                let input_shape = &plan.nodes[*input].shape;
                if input_shape.is_empty() || *axis >= input_shape.len() {
                    return Err(format!("CUDA axis reduction node {node_id} has an invalid axis"));
                }
                let expected_shape = input_shape
                    .iter()
                    .enumerate()
                    .filter_map(|(input_axis, extent)| (input_axis != *axis).then_some(*extent))
                    .collect::<Vec<_>>();
                if node.shape != expected_shape {
                    return Err(format!(
                        "CUDA axis reduction node {node_id} has shape {:?}, expected {:?}",
                        node.shape, expected_shape
                    ));
                }
                let input_strides = contiguous_strides(input_shape);
                let base_terms = (0..input_shape.len())
                    .filter(|input_axis| *input_axis != *axis)
                    .map(|input_axis| {
                        let output_axis = if input_axis < *axis {
                            input_axis
                        } else {
                            input_axis - 1
                        };
                        let output_stride = node.shape[output_axis + 1..].iter().product::<usize>();
                        format!(
                            "((index / {output_stride}ULL) % {}ULL) * {}ULL",
                            node.shape[output_axis], input_strides[input_axis]
                        )
                    })
                    .collect::<Vec<_>>();
                let base = if base_terms.is_empty() {
                    "0ULL".to_string()
                } else {
                    base_terms.join(" + ")
                };
                let scale = if matches!(&node.op, TensorOp::MeanAxis { .. }) {
                    format!(" / {}.0f", input_shape[*axis])
                } else {
                    String::new()
                };
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index >= count) return;\n\
                        unsigned long long base = {base}; float value = 0.0f;\n\
                        for (unsigned long long k = 0; k < {}ULL; ++k) value += input[base + k * {}ULL];\n\
                        out[index] = value{scale};\n}}\n",
                    input_shape[*axis], input_strides[*axis]
                )
            }
            TensorOp::Broadcast { input } => {
                let offset = cuda_offset_expression(&node.shape, &plan.nodes[*input].shape);
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) out[index] = input[{offset}];\n}}\n"
                )
            }
            TensorOp::Concat { inputs, axis } => {
                let inner = element_count(&node.shape[*axis + 1..])?;
                let parameters = inputs
                    .iter()
                    .enumerate()
                    .map(|(index, _)| format!("const float* input_{index}"))
                    .chain(["float* out".to_string(), "unsigned long long count".to_string()])
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut start = 0usize;
                let mut branches = String::new();
                for (index, input) in inputs.iter().enumerate() {
                    let width = plan.nodes[*input].shape[*axis];
                    let condition = if index == 0 { "if" } else { "else if" };
                    branches.push_str(&format!(
                        "{condition} (coordinate < {}ULL) out[index] = input_{index}[(outer_index * {}ULL + (coordinate - {}ULL)) * {}ULL + inner_index];\n",
                        start + width,
                        width,
                        start,
                        inner
                    ));
                    start += width;
                }
                format!(
                    "extern \"C\" __global__ void {function}({parameters}) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index >= count) return;\n\
                        unsigned long long outer_index = index / {}ULL;\n\
                        unsigned long long remainder = index % {}ULL;\n\
                        unsigned long long coordinate = remainder / {}ULL;\n\
                        unsigned long long inner_index = remainder % {}ULL;\n\
                        {branches}}}\n",
                    node.shape[*axis] * inner,
                    node.shape[*axis] * inner,
                    inner,
                    inner,
                )
            }
            TensorOp::Slice { input, axis, start, length } => {
                let input_extent = plan.nodes[*input].shape[*axis];
                let inner = element_count(&node.shape[*axis + 1..])?;
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) {{ unsigned long long outer = index / {}ULL; unsigned long long rem = index % {}ULL; out[index] = input[(outer * {}ULL + {}ULL) * {}ULL + rem]; }}\n}}\n",
                    length * inner,
                    length * inner,
                    input_extent,
                    start,
                    inner,
                )
            }
            TensorOp::PadSlice { input, axis, start } => {
                let input_extent = plan.nodes[*input].shape[*axis];
                let output_extent = node.shape[*axis];
                let inner = element_count(&node.shape[*axis + 1..])?;
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) {{ unsigned long long outer = index / {}ULL; unsigned long long rem = index % {}ULL; unsigned long long coordinate = rem / {}ULL; unsigned long long inner_index = rem % {}ULL; out[index] = coordinate >= {}ULL && coordinate < {}ULL ? input[(outer * {}ULL + coordinate - {}ULL) * {}ULL + inner_index] : 0.0f; }}\n}}\n",
                    output_extent * inner,
                    output_extent * inner,
                    inner,
                    inner,
                    start,
                    start + input_extent,
                    input_extent,
                    start,
                    inner,
                )
            }
            TensorOp::Transpose { input, axes } => {
                let input_shape = &plan.nodes[*input].shape;
                if axes.len() != input_shape.len() || node.shape.len() != input_shape.len() {
                    return Err(format!("CUDA transpose node {node_id} has incompatible rank"));
                }
                let mut seen = vec![false; axes.len()];
                for (output_axis, input_axis) in axes.iter().copied().enumerate() {
                    if input_axis >= input_shape.len() || seen[input_axis] {
                        return Err(format!(
                            "CUDA transpose node {node_id} has invalid axes {axes:?}"
                        ));
                    }
                    seen[input_axis] = true;
                    if node.shape[output_axis] != input_shape[input_axis] {
                        return Err(format!(
                            "CUDA transpose node {node_id} has shape {:?}, expected axes {axes:?} of {:?}",
                            node.shape, input_shape
                        ));
                    }
                }
                let input_strides = contiguous_strides(input_shape);
                let terms = axes
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(output_axis, input_axis)| {
                        let output_stride = node.shape[output_axis + 1..]
                            .iter()
                            .product::<usize>();
                        format!(
                            "((index / {output_stride}ULL) % {}ULL) * {}ULL",
                            node.shape[output_axis], input_strides[input_axis]
                        )
                    })
                    .collect::<Vec<_>>();
                let input_offset = if terms.is_empty() {
                    "0ULL".to_string()
                } else {
                    terms.join(" + ")
                };
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
                        if (index < count) out[index] = input[{input_offset}];\n}}\n"
                )
            }
        };
        source.push_str(&kernel);
    }
    Ok(source)
}

fn cuda_node_function_name(node_id: usize) -> String {
    format!("nabla_node_{node_id}")
}

fn cuda_offset_expression(output_shape: &[usize], input_shape: &[usize]) -> String {
    let input_strides = contiguous_strides(input_shape);
    let rank_offset = output_shape.len() - input_shape.len();
    let terms = (0..output_shape.len())
        .filter(|axis| *axis >= rank_offset && input_shape[*axis - rank_offset] != 1)
        .map(|axis| {
            let output_stride = output_shape[axis + 1..].iter().product::<usize>();
            format!(
                "((index / {output_stride}ULL) % {}ULL) * {}ULL",
                output_shape[axis],
                input_strides[axis - rank_offset]
            )
        })
        .collect::<Vec<_>>();
    if terms.is_empty() {
        "0ULL".to_string()
    } else {
        terms.join(" + ")
    }
}

fn cuda_float_literal(value: f64) -> String {
    let value = value as f32;
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value:?}f")
    }
}

fn cuda_op_name(op: &TensorOp) -> &'static str {
    match op {
        TensorOp::Input { .. } => "input",
        TensorOp::ScalarConstant { .. } => "constant",
        TensorOp::Add { .. } => "add",
        TensorOp::Sub { .. } => "sub",
        TensorOp::Div { .. } => "div",
        TensorOp::Mul { .. } => "mul",
        TensorOp::Greater { .. } => "greater",
        TensorOp::Where { .. } => "where",
        TensorOp::Sum { .. } => "sum",
        TensorOp::SumAxis { .. } => "sum_axis",
        TensorOp::Matmul { .. } => "matmul",
        TensorOp::Solve { .. } => "solve",
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

fn direct_rank_two_matmul_inputs(
    plan: &TensorExecutionPlan,
) -> Result<Option<(&str, &str)>, String> {
    let output = plan
        .nodes
        .get(plan.output_node_id)
        .ok_or_else(|| "execution plan output node does not exist".to_string())?;
    let TensorOp::Matmul { lhs, rhs } = output.op else {
        return Ok(None);
    };
    let TensorOp::Input { name: lhs_name } = &plan.nodes[lhs].op else {
        return Ok(None);
    };
    let TensorOp::Input { name: rhs_name } = &plan.nodes[rhs].op else {
        return Ok(None);
    };
    if plan.nodes[lhs].shape.len() != 2
        || plan.nodes[rhs].shape.len() != 2
        || output.shape.len() != 2
    {
        return Ok(None);
    }
    Ok(Some((lhs_name, rhs_name)))
}

const CUDA_RANK_TWO_MATMUL_SOURCE: &str = r#"
constexpr unsigned int NABLA_TILE = 32;
constexpr unsigned int NABLA_BLOCK = 16;

extern "C" __global__ void nabla_rank_two_matmul(
    const float* lhs,
    const float* rhs,
    float* output,
    unsigned long long rows,
    unsigned long long inner,
    unsigned long long cols
) {
    __shared__ float lhs_tile[NABLA_TILE][NABLA_TILE];
    __shared__ float rhs_tile[NABLA_TILE][NABLA_TILE];
    const unsigned long long row = (unsigned long long)blockIdx.y * NABLA_TILE + threadIdx.y;
    const unsigned long long col = (unsigned long long)blockIdx.x * NABLA_TILE + threadIdx.x;
    float value_00 = 0.0f;
    float value_01 = 0.0f;
    float value_10 = 0.0f;
    float value_11 = 0.0f;
    const unsigned long long tile_count = (inner + NABLA_TILE - 1) / NABLA_TILE;
    for (unsigned long long tile = 0; tile < tile_count; ++tile) {
        const unsigned long long lhs_col = tile * NABLA_TILE + threadIdx.x;
        const unsigned long long rhs_row = tile * NABLA_TILE + threadIdx.y;
        lhs_tile[threadIdx.y][threadIdx.x] = row < rows && lhs_col < inner
            ? lhs[row * inner + lhs_col] : 0.0f;
        lhs_tile[threadIdx.y][threadIdx.x + NABLA_BLOCK] = row < rows && lhs_col + NABLA_BLOCK < inner
            ? lhs[row * inner + lhs_col + NABLA_BLOCK] : 0.0f;
        lhs_tile[threadIdx.y + NABLA_BLOCK][threadIdx.x] = row + NABLA_BLOCK < rows && lhs_col < inner
            ? lhs[(row + NABLA_BLOCK) * inner + lhs_col] : 0.0f;
        lhs_tile[threadIdx.y + NABLA_BLOCK][threadIdx.x + NABLA_BLOCK] = row + NABLA_BLOCK < rows && lhs_col + NABLA_BLOCK < inner
            ? lhs[(row + NABLA_BLOCK) * inner + lhs_col + NABLA_BLOCK] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x] = rhs_row < inner && col < cols
            ? rhs[rhs_row * cols + col] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x + NABLA_BLOCK] = rhs_row < inner && col + NABLA_BLOCK < cols
            ? rhs[rhs_row * cols + col + NABLA_BLOCK] : 0.0f;
        rhs_tile[threadIdx.y + NABLA_BLOCK][threadIdx.x] = rhs_row + NABLA_BLOCK < inner && col < cols
            ? rhs[(rhs_row + NABLA_BLOCK) * cols + col] : 0.0f;
        rhs_tile[threadIdx.y + NABLA_BLOCK][threadIdx.x + NABLA_BLOCK] = rhs_row + NABLA_BLOCK < inner && col + NABLA_BLOCK < cols
            ? rhs[(rhs_row + NABLA_BLOCK) * cols + col + NABLA_BLOCK] : 0.0f;
        __syncthreads();
        for (unsigned int k = 0; k < NABLA_TILE; ++k) {
            float lhs_top = lhs_tile[threadIdx.y][k];
            float lhs_bottom = lhs_tile[threadIdx.y + NABLA_BLOCK][k];
            float rhs_left = rhs_tile[k][threadIdx.x];
            float rhs_right = rhs_tile[k][threadIdx.x + NABLA_BLOCK];
            value_00 += lhs_top * rhs_left;
            value_01 += lhs_top * rhs_right;
            value_10 += lhs_bottom * rhs_left;
            value_11 += lhs_bottom * rhs_right;
        }
        __syncthreads();
    }
    if (row < rows && col < cols) output[row * cols + col] = value_00;
    if (row < rows && col + NABLA_BLOCK < cols) output[row * cols + col + NABLA_BLOCK] = value_01;
    if (row + NABLA_BLOCK < rows && col < cols) output[(row + NABLA_BLOCK) * cols + col] = value_10;
    if (row + NABLA_BLOCK < rows && col + NABLA_BLOCK < cols) output[(row + NABLA_BLOCK) * cols + col + NABLA_BLOCK] = value_11;
}
"#;

const CUDA_MATMUL_BIAS_TANH_SOURCE: &str = r#"
extern "C" __global__ void nabla_matmul_bias_tanh(
    const float* lhs, const float* rhs, const float* bias, float* out,
    unsigned long long rows, unsigned long long inner, unsigned long long cols
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = rows * cols;
    if (index >= count) return;
    unsigned long long row = index / cols;
    unsigned long long col = index % cols;
    float value = 0.0f;
    for (unsigned long long k = 0; k < inner; ++k) value += lhs[row * inner + k] * rhs[k * cols + col];
    out[index] = tanhf(value + bias[col]);
}
"#;

fn cuda_register_tiled_rank_two_matmul_source(function: &str) -> String {
    CUDA_REGISTER_TILED_MATMUL_TEMPLATE.replace("NABLA_MATMUL_FUNCTION", function)
}

const CUDA_REGISTER_TILED_MATMUL_TEMPLATE: &str = r#"
extern "C" __global__ void NABLA_MATMUL_FUNCTION(
    const float* lhs,
    const float* rhs,
    float* out,
    unsigned long long rows,
    unsigned long long inner,
    unsigned long long cols
) {
    __shared__ float lhs_tile[32][32];
    __shared__ float rhs_tile[32][32];
    const unsigned long long row = (unsigned long long)blockIdx.y * 32ULL + threadIdx.y;
    const unsigned long long col = (unsigned long long)blockIdx.x * 32ULL + threadIdx.x;
    float value_00 = 0.0f;
    float value_01 = 0.0f;
    float value_10 = 0.0f;
    float value_11 = 0.0f;
    const unsigned long long tile_count = (inner + 31ULL) / 32ULL;
    for (unsigned long long tile = 0; tile < tile_count; ++tile) {
        const unsigned long long lhs_col = tile * 32ULL + threadIdx.x;
        const unsigned long long rhs_row = tile * 32ULL + threadIdx.y;
        lhs_tile[threadIdx.y][threadIdx.x] = row < rows && lhs_col < inner
            ? lhs[row * inner + lhs_col] : 0.0f;
        lhs_tile[threadIdx.y][threadIdx.x + 16] = row < rows && lhs_col + 16ULL < inner
            ? lhs[row * inner + lhs_col + 16ULL] : 0.0f;
        lhs_tile[threadIdx.y + 16][threadIdx.x] = row + 16ULL < rows && lhs_col < inner
            ? lhs[(row + 16ULL) * inner + lhs_col] : 0.0f;
        lhs_tile[threadIdx.y + 16][threadIdx.x + 16] = row + 16ULL < rows && lhs_col + 16ULL < inner
            ? lhs[(row + 16ULL) * inner + lhs_col + 16ULL] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x] = rhs_row < inner && col < cols
            ? rhs[rhs_row * cols + col] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x + 16] = rhs_row < inner && col + 16ULL < cols
            ? rhs[rhs_row * cols + col + 16ULL] : 0.0f;
        rhs_tile[threadIdx.y + 16][threadIdx.x] = rhs_row + 16ULL < inner && col < cols
            ? rhs[(rhs_row + 16ULL) * cols + col] : 0.0f;
        rhs_tile[threadIdx.y + 16][threadIdx.x + 16] = rhs_row + 16ULL < inner && col + 16ULL < cols
            ? rhs[(rhs_row + 16ULL) * cols + col + 16ULL] : 0.0f;
        __syncthreads();
        for (unsigned int k = 0; k < 32U; ++k) {
            float lhs_top = lhs_tile[threadIdx.y][k];
            float lhs_bottom = lhs_tile[threadIdx.y + 16][k];
            float rhs_left = rhs_tile[k][threadIdx.x];
            float rhs_right = rhs_tile[k][threadIdx.x + 16];
            value_00 += lhs_top * rhs_left;
            value_01 += lhs_top * rhs_right;
            value_10 += lhs_bottom * rhs_left;
            value_11 += lhs_bottom * rhs_right;
        }
        __syncthreads();
    }
    if (row < rows && col < cols) out[row * cols + col] = value_00;
    if (row < rows && col + 16ULL < cols) out[row * cols + col + 16ULL] = value_01;
    if (row + 16ULL < rows && col < cols) out[(row + 16ULL) * cols + col] = value_10;
    if (row + 16ULL < rows && col + 16ULL < cols) out[(row + 16ULL) * cols + col + 16ULL] = value_11;
}
"#;
