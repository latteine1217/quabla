//! Fused device optimizer steps over the retained parameters of one plan.
//!
//! After a value-and-gradient execution has retained the parameters and
//! computed every gradient, [`CudaExecutionPlan::optimizer_step`] updates all
//! parameters in place with one SGD or Adam/AdamW kernel per parameter. With
//! `clip_norm`, the gradients are first scaled by the global-norm factor of
//! `quabla.optim.clip_by_global_norm`, which is reduced on the device and
//! read by the update kernels from device memory: nothing crosses to the
//! host. Like every generated kernel, the source below is written in single
//! precision and rewritten to double for float64 plans, so parameters and
//! moments keep the plan's element type.
//!
//! The global norm follows the host definition: `m * sqrt(sum((g / m)^2))`
//! with `m` the largest `|g|` over all gradients, accumulated in double.
//! Dividing by `m` bounds every term by one, so the sum cannot overflow even
//! for float64 gradients near the largest finite value, and tiny gradients do
//! not underflow to a zero norm. Both passes reduce in a fixed order (per-block
//! partials, then one block over the partials), so a step is deterministic.

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use super::super::{DeviceOptimizerConfig, DeviceUpdateRule, TensorNodeId};
use super::{cuda_value_mut, input_node_id, CudaAdamState, CudaExecutionPlan, CudaReal};

/// Threads per block of the global-norm kernels; the kernels' shared arrays
/// have exactly this many entries and the tree reduction needs a power of two.
const GLOBAL_NORM_THREADS: u32 = 256;
/// Upper bound on the partial blocks of one gradient; larger gradients are
/// covered by grid-stride loops, which keeps the final single-block pass short.
const GLOBAL_NORM_MAX_BLOCKS: usize = 256;

pub(super) const CUDA_TRAINING_SOURCE: &str = r#"
__device__ __forceinline__ double quabla_global_norm_combine(double total, double value, int square) {
    if (square) return total + value;
    // A NaN maximum stays NaN, as the host norm propagates it.
    return (value > total || value != value) ? value : total;
}

extern "C" __global__ void quabla_global_norm_partials(
    const float* gradient, unsigned long long count, const double* scalars,
    double* partials, unsigned long long offset, int square
) {
    __shared__ double cache[256];
    double scale = 1.0;
    if (square) {
        double peak = scalars[0];
        if (isfinite(peak) && peak > 0.0) scale = peak;
    }
    double total = 0.0;
    for (unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += (unsigned long long)gridDim.x * blockDim.x) {
        double value = fabs((double)gradient[index]) / scale;
        total = quabla_global_norm_combine(total, square ? value * value : value, square);
    }
    cache[threadIdx.x] = total;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            cache[threadIdx.x] = quabla_global_norm_combine(
                cache[threadIdx.x], cache[threadIdx.x + stride], square);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[offset + blockIdx.x] = cache[0];
}

extern "C" __global__ void quabla_global_norm_reduce(
    const double* partials, unsigned long long count, double* scalars, double max_norm, int square
) {
    __shared__ double cache[256];
    double total = 0.0;
    for (unsigned long long index = threadIdx.x; index < count; index += blockDim.x) {
        total = quabla_global_norm_combine(total, partials[index], square);
    }
    cache[threadIdx.x] = total;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            cache[threadIdx.x] = quabla_global_norm_combine(
                cache[threadIdx.x], cache[threadIdx.x + stride], square);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    if (!square) {
        scalars[0] = cache[0];
        return;
    }
    double peak = scalars[0];
    double scale = isfinite(peak) && peak > 0.0 ? peak : 1.0;
    double norm = scale * sqrt(cache[0]);
    // As clip_by_global_norm: gradients at or below the bound are unchanged
    // and a non-finite norm turns every gradient into NaN. The sticky flag
    // records that for parameters the plan pruned (see `nonfinite_clip_seen`).
    scalars[1] = norm <= max_norm ? 1.0 : (isfinite(norm) ? max_norm / norm : nan(""));
    if (!isfinite(norm)) scalars[2] = 1.0;
}

extern "C" __global__ void quabla_device_sgd(
    float* parameter, const float* gradient, float learning_rate,
    const double* scalars, int clipped, unsigned long long count
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float gradient_value = gradient[index];
    // Clipping scales in double and rounds once, as clip_by_global_norm.
    if (clipped) gradient_value = (float)((double)gradient_value * scalars[1]);
    parameter[index] = parameter[index] - gradient_value * learning_rate;
}

extern "C" __global__ void quabla_device_adam(
    float* parameter, const float* gradient, float* first_moment, float* second_moment,
    float learning_rate, float beta1, float beta2, float one_minus_beta1, float one_minus_beta2,
    float epsilon, float correction1, float correction2, float weight_decay,
    const double* scalars, int clipped, unsigned long long count
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float gradient_value = gradient[index];
    if (clipped) gradient_value = (float)((double)gradient_value * scalars[1]);
    // The host Adam/AdamW expression, in its operation order. `1 - beta` and
    // the corrections arrive formed in float64 from the float64 betas: in a
    // float32 plan `1.0f - beta2` of the rounded 0.999 is 1.3e-5 relative off.
    float first = first_moment[index] * beta1 + gradient_value * one_minus_beta1;
    float second = second_moment[index] * beta2 + (gradient_value * gradient_value) * one_minus_beta2;
    float delta = (first / correction1) / (sqrtf(second / correction2) + epsilon);
    float value = parameter[index];
    if (weight_decay != 0.0f) delta += weight_decay * value;
    first_moment[index] = first;
    second_moment[index] = second;
    parameter[index] = value - delta * learning_rate;
}
"#;

/// Device scratch of the global-norm reduction, reused across steps.
#[derive(Debug, Default)]
pub(super) struct CudaGlobalNormScratch {
    /// `[max |g|, clip factor, non-finite norm seen]`; the update kernels
    /// read the factor, and the last entry is sticky across steps.
    scalars: Option<CudaSlice<f64>>,
    partials: Option<CudaSlice<f64>>,
}

/// One parameter of a step: name, parameter and gradient node ids, length.
type StepTarget<'a> = (&'a str, usize, TensorNodeId, usize);

fn global_norm_blocks(count: usize) -> usize {
    count
        .div_ceil(GLOBAL_NORM_THREADS as usize)
        .min(GLOBAL_NORM_MAX_BLOCKS)
}

fn reduction_config(blocks: usize) -> Result<LaunchConfig, String> {
    let blocks = u32::try_from(blocks)
        .map_err(|_| "CUDA global-norm block count exceeds u32".to_string())?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (GLOBAL_NORM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

impl<T: CudaReal> CudaExecutionPlan<T> {
    /// Applies one fused optimizer update to every parameter in `gradients`
    /// (parameter input name to its computed gradient node), after an
    /// execution that retained the parameters and computed the gradients.
    ///
    /// Adam moments live in the plan's element type and are created at the
    /// first step; each parameter counts its own updates for the bias
    /// corrections, which are computed in float64 on the host. A rejected
    /// configuration or a missing buffer fails before any update.
    pub fn optimizer_step(
        &self,
        gradients: &BTreeMap<String, TensorNodeId>,
        config: &DeviceOptimizerConfig,
    ) -> Result<(), String> {
        config.validate()?;
        let stream = self.context.default_stream();
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let mut targets: Vec<StepTarget<'_>> = Vec::with_capacity(gradients.len());
        for (name, &gradient_node_id) in gradients {
            let parameter_node_id = input_node_id(&self.plan, name)?;
            if parameter_node_id >= gradient_node_id {
                return Err(format!(
                    "CUDA optimizer parameter {name:?} needs a computed gradient node after its input"
                ));
            }
            let count = state
                .values
                .get(parameter_node_id)
                .and_then(Option::as_ref)
                .ok_or_else(|| format!("CUDA input {name:?} has not been initialized"))?
                .len();
            if state
                .values
                .get(gradient_node_id)
                .and_then(Option::as_ref)
                .map(CudaSlice::len)
                != Some(count)
            {
                return Err(format!(
                    "CUDA optimizer parameter {name:?} and its gradient have different sizes"
                ));
            }
            targets.push((name.as_str(), parameter_node_id, gradient_node_id, count));
        }
        let state = &mut *state;
        if state.global_norm.scalars.is_none() {
            state.global_norm.scalars = Some(stream.alloc_zeros::<f64>(3).map_err(|error| {
                format!("failed to allocate CUDA optimizer scalars: {error:?}")
            })?);
        }
        let clipped: i32 = match config.clip_norm {
            Some(clip_norm) => {
                self.global_norm_factor(
                    &stream,
                    &state.values,
                    &targets,
                    &mut state.global_norm,
                    clip_norm,
                )?;
                1
            }
            None => 0,
        };
        let scalars = state
            .global_norm
            .scalars
            .as_ref()
            .expect("CUDA optimizer scalars were allocated");
        let learning_rate = T::from_f64(config.learning_rate);
        for &(name, parameter_node_id, gradient_node_id, count) in &targets {
            if count == 0 {
                continue;
            }
            let count_u64 = u64::try_from(count)
                .map_err(|_| "CUDA optimizer parameter count exceeds u64".to_string())?;
            let launch_count = u32::try_from(count).map_err(|_| {
                "CUDA optimizer parameter count exceeds u32 launch size".to_string()
            })?;
            let (before_gradient, gradient_and_after) = state.values.split_at_mut(gradient_node_id);
            let parameter = cuda_value_mut(before_gradient, parameter_node_id)?;
            let gradient = gradient_and_after
                .first()
                .and_then(Option::as_ref)
                .ok_or_else(|| format!("CUDA optimizer gradient of {name:?} is missing"))?;
            match config.rule {
                DeviceUpdateRule::Sgd => {
                    let kernel = self
                        .module
                        .load_function("quabla_device_sgd")
                        .map_err(|error| format!("failed to load CUDA SGD kernel: {error:?}"))?;
                    let mut launch = stream.launch_builder(&kernel);
                    launch.arg(parameter);
                    launch.arg(gradient);
                    launch.arg(&learning_rate);
                    launch.arg(scalars);
                    launch.arg(&clipped);
                    launch.arg(&count_u64);
                    // SAFETY: the arguments match the six parameters of `quabla_device_sgd` in
                    // `CUDA_TRAINING_SOURCE` in order and type (`float` is `T` after the
                    // float64 rewrite); `parameter` and `gradient` hold `count` elements
                    // (checked above), `scalars` holds three, the kernel guards
                    // `index < count`, and only the mutably passed `parameter` is written.
                    unsafe {
                        launch
                            .launch(LaunchConfig::for_num_elems(launch_count))
                            .map_err(|error| {
                                format!("failed to launch CUDA SGD kernel: {error:?}")
                            })?;
                    }
                }
                DeviceUpdateRule::Adam {
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                } => {
                    if !state.adam.contains_key(name) {
                        let first_moment = stream.alloc_zeros::<T>(count).map_err(|error| {
                            format!("failed to allocate CUDA Adam first moment: {error:?}")
                        })?;
                        let second_moment = stream.alloc_zeros::<T>(count).map_err(|error| {
                            format!("failed to allocate CUDA Adam second moment: {error:?}")
                        })?;
                        state.adam.insert(
                            name.to_string(),
                            CudaAdamState {
                                first_moment,
                                second_moment,
                                step: 0,
                            },
                        );
                    }
                    let adam = state
                        .adam
                        .get_mut(name)
                        .expect("CUDA Adam state was initialized");
                    adam.step = adam
                        .step
                        .checked_add(1)
                        .ok_or_else(|| "CUDA Adam step counter overflow".to_string())?;
                    let (correction1, correction2) =
                        DeviceOptimizerConfig::adam_corrections(beta1, beta2, adam.step);
                    let hyperparameters = [
                        beta1,
                        beta2,
                        1.0 - beta1,
                        1.0 - beta2,
                        epsilon,
                        correction1,
                        correction2,
                        weight_decay,
                    ]
                    .map(T::from_f64);
                    let [beta1, beta2, one_minus_beta1, one_minus_beta2, epsilon, correction1, correction2, weight_decay] =
                        hyperparameters;
                    let kernel = self
                        .module
                        .load_function("quabla_device_adam")
                        .map_err(|error| format!("failed to load CUDA Adam kernel: {error:?}"))?;
                    let mut launch = stream.launch_builder(&kernel);
                    launch.arg(parameter);
                    launch.arg(gradient);
                    launch.arg(&mut adam.first_moment);
                    launch.arg(&mut adam.second_moment);
                    launch.arg(&learning_rate);
                    launch.arg(&beta1);
                    launch.arg(&beta2);
                    launch.arg(&one_minus_beta1);
                    launch.arg(&one_minus_beta2);
                    launch.arg(&epsilon);
                    launch.arg(&correction1);
                    launch.arg(&correction2);
                    launch.arg(&weight_decay);
                    launch.arg(scalars);
                    launch.arg(&clipped);
                    launch.arg(&count_u64);
                    // SAFETY: the arguments match the sixteen parameters of `quabla_device_adam`
                    // in `CUDA_TRAINING_SOURCE` in order and type (`float` is `T` after the
                    // float64 rewrite); the parameter, gradient, and both moments hold `count`
                    // elements (the moments are allocated from the parameter length, which the
                    // plan's input shape fixes), `scalars` holds three, the kernel guards
                    // `index < count`, and only the mutably passed buffers are written.
                    unsafe {
                        launch
                            .launch(LaunchConfig::for_num_elems(launch_count))
                            .map_err(|error| {
                                format!("failed to launch CUDA Adam kernel: {error:?}")
                            })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a clipped step of [`Self::optimizer_step`] has met a NaN or
    /// infinite global norm. Such a step turns every gradient into NaN, also
    /// the zero gradients of parameters the plan pruned because the loss
    /// ignores them; their owner reads this flag (one scalar, no gradient)
    /// to reproduce that on its host copies.
    pub fn nonfinite_clip_seen(&self) -> Result<bool, String> {
        let stream = self.context.default_stream();
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let Some(scalars) = state.global_norm.scalars.as_ref() else {
            return Ok(false);
        };
        let values = stream
            .clone_dtoh(scalars)
            .map_err(|error| format!("failed to read CUDA optimizer scalars: {error:?}"))?;
        Ok(values[2] != 0.0)
    }

    /// Writes the clip factor `min(1, clip_norm / norm)` of all target
    /// gradients to `scratch.scalars[1]` (NaN for a non-finite norm) in two
    /// passes: the largest magnitude, then the sum of squares scaled by it.
    fn global_norm_factor(
        &self,
        stream: &Arc<CudaStream>,
        values: &[Option<CudaSlice<T>>],
        targets: &[StepTarget<'_>],
        scratch: &mut CudaGlobalNormScratch,
        clip_norm: f64,
    ) -> Result<(), String> {
        let blocks = targets
            .iter()
            .map(|&(_, _, _, count)| global_norm_blocks(count))
            .collect::<Vec<_>>();
        let partial_count = blocks.iter().sum::<usize>();
        // The reduction always reads at least one partial; empty gradients add none.
        let needed = partial_count.max(1);
        if scratch
            .partials
            .as_ref()
            .is_none_or(|partials| partials.len() < needed)
        {
            scratch.partials = Some(stream.alloc_zeros::<f64>(needed).map_err(|error| {
                format!("failed to allocate CUDA global-norm partials: {error:?}")
            })?);
        }
        let (Some(scalars), Some(partials)) = (scratch.scalars.as_mut(), scratch.partials.as_mut())
        else {
            unreachable!("CUDA global-norm scratch was allocated");
        };
        let partial_count_u64 = u64::try_from(partial_count)
            .map_err(|_| "CUDA global-norm partial count exceeds u64".to_string())?;
        let partials_kernel = self
            .module
            .load_function("quabla_global_norm_partials")
            .map_err(|error| format!("failed to load CUDA global-norm kernel: {error:?}"))?;
        let reduce_kernel = self
            .module
            .load_function("quabla_global_norm_reduce")
            .map_err(|error| format!("failed to load CUDA global-norm kernel: {error:?}"))?;
        for square in [0i32, 1] {
            let mut offset = 0u64;
            for (&(name, _, gradient_node_id, count), &block_count) in targets.iter().zip(&blocks) {
                if block_count == 0 {
                    continue;
                }
                let gradient = values
                    .get(gradient_node_id)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("CUDA optimizer gradient of {name:?} is missing"))?;
                let count_u64 = u64::try_from(count)
                    .map_err(|_| "CUDA optimizer parameter count exceeds u64".to_string())?;
                let mut launch = stream.launch_builder(&partials_kernel);
                launch.arg(gradient);
                launch.arg(&count_u64);
                launch.arg(&*scalars);
                launch.arg(&mut *partials);
                launch.arg(&offset);
                launch.arg(&square);
                // SAFETY: the arguments match the six parameters of `quabla_global_norm_partials`
                // in `CUDA_TRAINING_SOURCE` in order and type; `gradient` holds `count` elements
                // and the grid-stride loop guards `index < count`; the launch has `block_count`
                // blocks of `GLOBAL_NORM_THREADS` threads (the kernel's shared array size) and
                // block `b` writes only `partials[offset + b]`, inside the `partial_count`
                // entries allocated above; `scalars` holds three elements.
                unsafe {
                    launch
                        .launch(reduction_config(block_count)?)
                        .map_err(|error| {
                            format!("failed to launch CUDA global-norm kernel: {error:?}")
                        })?;
                }
                offset += block_count as u64;
            }
            let mut launch = stream.launch_builder(&reduce_kernel);
            launch.arg(&*partials);
            launch.arg(&partial_count_u64);
            launch.arg(&mut *scalars);
            launch.arg(&clip_norm);
            launch.arg(&square);
            // SAFETY: the arguments match the five parameters of `quabla_global_norm_reduce` in
            // `CUDA_TRAINING_SOURCE` in order and type; the kernel reads the first
            // `partial_count` entries of `partials`, all written by the pass above, runs as one
            // block of `GLOBAL_NORM_THREADS` threads, and writes only `scalars[0]`,
            // `scalars[1]`, or `scalars[2]` of its three elements.
            unsafe {
                launch.launch(reduction_config(1)?).map_err(|error| {
                    format!("failed to launch CUDA global-norm kernel: {error:?}")
                })?;
            }
        }
        Ok(())
    }
}
