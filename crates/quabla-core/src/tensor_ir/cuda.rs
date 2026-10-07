use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard};
use std::time::Duration;

#[cfg(feature = "cuda-nccl")]
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, GemmConfig, StridedBatchedConfig};
use cudarc::cusolver::{safe::DnHandle, sys as cusolver_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

#[cfg(feature = "cuda-nccl")]
use cudarc::nccl::{group_end, group_start, Comm as NcclComm, ReduceOp as NcclReduceOp};

use super::region::{evaluator_region_plans, RegionPlan};
use super::{
    contiguous_strides, cuda_elementwise_formula, cuda_scalar_literal, element_count,
    tensor_op_inputs, DynamicTensor, LinalgKind, RegionKind, RegionNode, TensorBackend,
    TensorDType, TensorDeviceBackend, TensorExecutionPlan, TensorExtremum, TensorForiExecutionPlan,
    TensorForiVjpJvpExecutionPlan, TensorForiVjpTarget, TensorFusionRegion, TensorNodeId, TensorOp,
    TensorReplicaReduction, TensorScanExecutionPlan, TensorScanTarget,
    TensorScanVjpJvpExecutionPlan, TensorScanVjpTarget, TensorShardingPlan, UnaryMathKind,
};

#[cfg(test)]
thread_local! {
    static CUDA_LOOP_TEST_TAPE_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const CUDA_MATMUL_TILE: usize = 32;
const CUDA_MATMUL_TILE_U64: u64 = 32;
const CUDA_MATMUL_BLOCK: u32 = 16;
const CUDA_REDUCTION_BLOCK: u32 = 256;
/// Threads per block of a one-thread-per-element launch the kernel can run
/// with; see [`cuda_elementwise_launch`].
const CUDA_ELEMENTWISE_BLOCK: u32 = 1024;

#[path = "cuda_cholesky.rs"]
mod cholesky_backend;
#[path = "cuda_decompositions.rs"]
mod decompositions;
#[path = "cuda_host_loop.rs"]
mod host_loop;
#[path = "cuda_optimizer.rs"]
mod optimizer;
#[path = "cuda_real.rs"]
mod real;

pub use real::CudaReal;

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
pub struct CudaExecutionPlan<T: CudaReal = f32> {
    plan: TensorExecutionPlan,
    fused_elementwise: bool,
    fused_region_count: usize,
    matmul_bias_tanh: Option<CudaMatmulBiasTanhEpilogue>,
    device_ordinal: usize,
    context: Arc<CudaContext>,
    module: Arc<CudaModule>,
    blas: Option<Arc<Mutex<CudaLibraryHandle<CudaBlas>>>>,
    solver: Option<Arc<Mutex<CudaSolver<T>>>>,
    state: Arc<Mutex<CudaExecutionState<T>>>,
    cond_branches: BTreeMap<TensorNodeId, CudaCondBranches<T>>,
    /// Loop nodes whose body is not fused-lowerable, keyed by the first node of
    /// their result group; see `host_loop`.
    host_loops: BTreeMap<TensorNodeId, host_loop::CudaHostLoop<T>>,
    /// Host-to-device copies of array constants made by this plan's
    /// executions; see [`CudaExecutionPlan::constant_upload_count`].
    constant_uploads: Arc<AtomicUsize>,
    /// Forward-only executable for querying a shared value-and-gradient plan.
    primary_plan: Arc<Mutex<Option<CudaExecutionPlan<T>>>>,
}

/// Device plans for the two regions of one `Cond` node.
///
/// Both regions are compiled ahead of time in the parent's CUDA context so
/// captures and the selected result stay device buffers; only the scalar
/// predicate crosses to the host at execution time.
#[derive(Clone, Debug)]
struct CudaCondBranches<T: CudaReal> {
    on_true: CudaExecutionPlan<T>,
    on_false: CudaExecutionPlan<T>,
}

/// A single-node NCCL data-parallel executable.
///
/// Each replica owns a complete CUDA compilation on one explicit device. The
/// caller supplies already-sharded inputs, then this plan all-reduces retained
/// outputs in place. This keeps input partitioning and the collective
/// boundary explicit: it never copies a shard through the host to another GPU.
#[derive(Clone, Debug)]
pub struct CudaDataParallelExecutionPlan {
    replicas: Vec<CudaExecutionPlan>,
    output_node_ids: Vec<TensorNodeId>,
    /// Ordered collectives bound from a `TensorShardingPlan`; `None` keeps the
    /// original contract where the caller selects one reduction per call.
    #[cfg(feature = "cuda-nccl")]
    collectives: Option<Vec<(TensorNodeId, TensorReplicaReduction)>>,
    /// Rank-ordered NCCL communicators created once at compile time and
    /// shared by clones. `ncclCommInitAll` costs far more than a small
    /// all-reduce, so creating it per call dominated every training step.
    /// The mutex also serializes invocations: NCCL forbids concurrent use of
    /// a communicator, and clones share the replicas' retained buffers.
    #[cfg(feature = "cuda-nccl")]
    communicators: Arc<Mutex<CudaNcclCommunicators>>,
}

/// Rank-ordered NCCL communicators of one data-parallel plan.
///
/// `Comm` wraps a raw `ncclComm_t` and is therefore not `Send`. The handle is
/// not bound to its creating thread; NCCL only forbids concurrent use, which
/// the owning `Mutex` enforces. Each `Comm` holds an `Arc` of its replica's
/// stream and context, so those outlive the `ncclCommAbort` issued on drop.
///
/// `None` means a previous invocation failed. A failure can leave a
/// collective launched on some ranks but not others, which would block the
/// next collective or synchronization on those streams, so the failed call
/// drops (aborts) the communicators and the next call creates new ones.
/// Successful invocations synchronize every replica and failed ones abort
/// their communicators, so no collective is in flight when the last clone
/// is dropped.
#[cfg(feature = "cuda-nccl")]
#[derive(Debug)]
struct CudaNcclCommunicators(Option<Vec<NcclComm>>);

// SAFETY: see `CudaNcclCommunicators`; all access goes through its `Mutex`.
#[cfg(feature = "cuda-nccl")]
unsafe impl Send for CudaNcclCommunicators {}

/// Host-observed boundaries for one data-parallel invocation.
///
/// `replica_enqueue` measures submission of replica work, not GPU kernel
/// runtime. `collective` includes NCCL submission and device synchronization;
/// `output_readback` measures the final rank-zero diagnostics transfer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaDataParallelTiming {
    pub replica_enqueue: Duration,
    pub collective: Duration,
    pub output_readback: Duration,
}

#[derive(Clone, Debug)]
pub struct CudaDataParallelResult {
    pub outputs: Vec<DynamicTensor>,
    pub timing: CudaDataParallelTiming,
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
struct CudaExecutionState<T: CudaReal> {
    values: Vec<Option<CudaSlice<T>>>,
    free_buffers: CudaBufferPool<T>,
    adam: BTreeMap<String, CudaAdamState<T>>,
    global_norm: optimizer::CudaGlobalNormScratch,
}

#[derive(Debug, Default)]
struct CudaBufferPool<T: CudaReal> {
    buffers: BTreeMap<usize, Vec<CudaSlice<T>>>,
    elements: usize,
    budget: usize,
    /// Set while a host-driven loop records a region into a CUDA graph: the
    /// pool then neither allocates (an allocation would become a graph node)
    /// nor frees (the graph keeps referring to every buffer it recorded).
    capturing: bool,
}

impl<T: CudaReal> CudaBufferPool<T> {
    fn take(&mut self, count: usize) -> Option<CudaSlice<T>> {
        let values = self.buffers.get_mut(&count)?;
        let buffer = values.pop()?;
        self.elements -= count;
        if values.is_empty() {
            self.buffers.remove(&count);
        }
        Some(buffer)
    }

    fn values(&self) -> impl Iterator<Item = &Vec<CudaSlice<T>>> {
        self.buffers.values()
    }

    fn recycle(&mut self, buffer: CudaSlice<T>) {
        let count = buffer.len();
        if self.capturing {
            self.elements += count;
            self.buffers.entry(count).or_default().push(buffer);
            return;
        }
        if count > self.budget {
            return;
        }
        // Prefer reusable large allocations; never retain more than one live frontier.
        while self.elements > self.budget - count {
            let Some((&smallest, _)) = self.buffers.first_key_value() else {
                break;
            };
            if smallest >= count {
                return;
            }
            let values = self
                .buffers
                .get_mut(&smallest)
                .expect("key was found above");
            values.pop();
            self.elements -= smallest;
            if values.is_empty() {
                self.buffers.remove(&smallest);
            }
        }
        self.elements += count;
        self.buffers.entry(count).or_default().push(buffer);
    }
}

fn cuda_pool_frontier_budget(plan: &TensorExecutionPlan) -> Result<usize, String> {
    let counts = plan
        .nodes
        .iter()
        .map(|node| element_count(&node.shape))
        .collect::<Result<Vec<_>, _>>()?;
    let mut uses = cuda_remaining_use_counts(plan);
    let outputs = plan
        .output_node_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let pinned = |id: usize| {
        outputs.contains(&id)
            || matches!(
                plan.nodes[id].op,
                TensorOp::Input { .. } | TensorOp::Constant { .. }
            )
    };
    let mut live = 0usize;
    let mut peak = 0usize;
    for (id, node) in plan.nodes.iter().enumerate() {
        live = live.saturating_add(counts[id]);
        peak = peak.max(live);
        for input in tensor_op_inputs(&node.op) {
            uses[input] -= 1;
            if uses[input] == 0 && !pinned(input) {
                live -= counts[input];
            }
        }
        if uses[id] == 0 && !pinned(id) {
            live -= counts[id];
        }
    }
    Ok(peak)
}

#[cfg(test)]
mod pool_profile_tests {
    use super::*;
    use crate::tensor_ir::TensorIr;
    use std::time::Instant;

    #[test]
    fn profile_shrinking_shape_pool() -> Result<(), String> {
        if std::env::var_os("QUABLA_CUDA_PROFILE").is_none() {
            return Ok(());
        }
        let mut graph = TensorIr::new();
        let input = graph.input_typed("x", vec![65536], TensorDType::F32)?;
        let mut value = input;
        for depth in 1..=24 {
            value = graph.sin(value)?;
            value = graph.slice_axis(value, 0, 0, 65536 - depth * 1024)?;
        }
        let plan = CudaBackend::new(0).compile(graph.compile_cpu(value)?)?;
        let inputs = BTreeMap::from([("x".into(), DynamicTensor::filled(vec![65536], 0.125)?)]);
        let mut expected = 0.125f64;
        for _ in 0..24 {
            expected = expected.sin();
        }
        let mut samples = Vec::new();
        for _ in 0..15 {
            let start = Instant::now();
            let output = plan.execute(&inputs)?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            assert!((output.storage().get(0) - expected).abs() < 1e-5);
        }
        samples.sort_by(f64::total_cmp);
        let state = plan
            .state
            .lock()
            .map_err(|_| "CUDA state lock was poisoned")?;
        let pooled_elements = state
            .free_buffers
            .values()
            .flatten()
            .map(CudaSlice::len)
            .sum::<usize>();
        let retained_elements = state
            .values
            .iter()
            .flatten()
            .map(CudaSlice::len)
            .sum::<usize>();
        println!("{{\"case\":\"cuda_pool_shrinking\",\"pooled_elements\":{pooled_elements},\"retained_elements\":{retained_elements},\"median_ms\":{},\"samples_ms\":{:?}}}", samples[7], samples);
        Ok(())
    }
}

#[derive(Debug)]
struct CudaScanCache<T: CudaReal> {
    carry: Option<CudaSlice<T>>,
    outputs: Option<CudaSlice<T>>,
}

/// Device buffers produced together by one structural Fori reverse group.
/// Keeping the sibling results here lets the first group node launch the
/// shared reverse kernel while later nodes simply claim their result buffer.
#[derive(Debug, Default)]
struct CudaForiVjpCache<T: CudaReal> {
    results: BTreeMap<TensorNodeId, CudaSlice<T>>,
}

/// Device buffers produced together by one structural Scan reverse group.
#[derive(Debug, Default)]
struct CudaScanVjpCache<T: CudaReal> {
    results: BTreeMap<TensorNodeId, CudaSlice<T>>,
}

/// Device buffers produced together by one structural Scan forward-over-reverse group.
#[derive(Debug, Default)]
struct CudaScanVjpJvpCache<T: CudaReal> {
    results: BTreeMap<TensorNodeId, CudaSlice<T>>,
}

#[derive(Debug)]
struct CudaAdamState<T: CudaReal> {
    first_moment: CudaSlice<T>,
    second_moment: CudaSlice<T>,
    step: u64,
}

impl CudaBackend {
    pub fn new(device_ordinal: usize) -> Self {
        Self { device_ordinal }
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Compiles one identical retained-output plan per explicitly selected GPU
    /// and creates the plan's NCCL communicators once, for its lifetime.
    ///
    /// This API is available in all CUDA builds so callers get a deterministic
    /// feature error instead of an implicit single-device fallback. Actual
    /// collective execution requires the optional `cuda-nccl` feature.
    #[cfg(feature = "cuda-nccl")]
    pub fn compile_data_parallel(
        &self,
        plan: TensorExecutionPlan,
        device_ordinals: Vec<usize>,
    ) -> Result<CudaDataParallelExecutionPlan, String> {
        validate_cuda_data_parallel_devices(&device_ordinals)?;
        let output_node_ids = plan.output_node_ids().to_vec();
        let replicas = device_ordinals
            .into_iter()
            .map(|ordinal| CudaBackend::new(ordinal).compile(plan.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let communicators = create_nccl_communicators(&replicas)?;
        Ok(CudaDataParallelExecutionPlan {
            replicas,
            output_node_ids,
            collectives: None,
            communicators: Arc::new(Mutex::new(CudaNcclCommunicators(Some(communicators)))),
        })
    }

    #[cfg(not(feature = "cuda-nccl"))]
    pub fn compile_data_parallel(
        &self,
        _plan: TensorExecutionPlan,
        _device_ordinals: Vec<usize>,
    ) -> Result<CudaDataParallelExecutionPlan, String> {
        Err("CUDA data-parallel execution requires the optional cuda-nccl feature".to_string())
    }

    /// Compiles the replica-local program of a sharding schedule derived from
    /// `plan` and binds its all-reduces, in schedule order, to NCCL.
    ///
    /// `plan` keeps global shapes; replicas run its axis-zero shard
    /// specialization. Schedules outside the first data-parallel subset are
    /// rejected by [`TensorExecutionPlan::cuda_data_parallel_program`].
    #[cfg(feature = "cuda-nccl")]
    pub fn compile_data_parallel_sharded(
        &self,
        plan: TensorExecutionPlan,
        sharding: &TensorShardingPlan,
        device_ordinals: Vec<usize>,
    ) -> Result<CudaDataParallelExecutionPlan, String> {
        let program = plan.cuda_data_parallel_program(sharding, &device_ordinals)?;
        let mut compiled = self.compile_data_parallel(program.replica_plan, device_ordinals)?;
        compiled.collectives = Some(program.collectives);
        Ok(compiled)
    }

    #[cfg(not(feature = "cuda-nccl"))]
    pub fn compile_data_parallel_sharded(
        &self,
        _plan: TensorExecutionPlan,
        _sharding: &TensorShardingPlan,
        _device_ordinals: Vec<usize>,
    ) -> Result<CudaDataParallelExecutionPlan, String> {
        Err("CUDA data-parallel execution requires the optional cuda-nccl feature".to_string())
    }

    /// Evaluates every retained output of one multi-output Tensor IR plan.
    /// CUDA lowering must keep all requested outputs alive in the shared value
    /// table, which is required by structured multi-result regions such as Scan.
    pub fn execute_many(
        &self,
        plan: &TensorExecutionPlan,
        output_node_ids: &[TensorNodeId],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        if output_node_ids != plan.output_node_ids() {
            return Err(
                "CUDA multi-output execution requires the plan's complete retained output list"
                    .to_string(),
            );
        }
        self.compile(plan.clone())?.execute_many(inputs)
    }

    pub fn compile(&self, plan: TensorExecutionPlan) -> Result<CudaExecutionPlan, String> {
        self.compile_in_context(plan, None)
    }

    /// Compiles `plan` with `f64` device buffers: every logical `float64` node
    /// executes in double precision (the opt-in `precision="float64"` lowering).
    ///
    /// The plan must not contain `float32` nodes: one plan has one floating
    /// element type, and running them in double would silently change their
    /// rounding (see [`validate_cuda_float64_plan`]).
    pub fn compile_float64(
        &self,
        plan: TensorExecutionPlan,
    ) -> Result<CudaExecutionPlan<f64>, String> {
        validate_cuda_float64_plan(&plan).map_err(|(_, message)| message)?;
        self.compile_in_context(plan, None)
    }

    /// When `region_context` is `Some`, compiles a `Cond` branch region: it shares the parent
    /// plan's CUDA context and always uses the per-node device program so captures bind as device
    /// buffers.
    fn compile_in_context<T: CudaReal>(
        &self,
        plan: TensorExecutionPlan,
        region_context: Option<&Arc<CudaContext>>,
    ) -> Result<CudaExecutionPlan<T>, String> {
        validate_cuda_plan(&plan).map_err(|(_, message)| message)?;
        ensure_cuda_driver_available()?;
        ensure_nvrtc_runtime_available()?;
        // The fused epilogue binds only uploaded inputs and never runs `Cond` first; region plans
        // need the per-node program.
        let has_cond = plan.nodes.iter().any(|node| {
            matches!(
                node.op,
                TensorOp::Region(RegionNode {
                    kind: RegionKind::Cond { .. },
                    ..
                })
            )
        });
        let matmul_bias_tanh = if region_context.is_none() && !has_cond {
            cuda_matmul_bias_tanh_epilogue(&plan)
        } else {
            None
        };
        let fusion_regions = plan.fusion_regions();
        let host_driven = host_loop::cuda_host_driven_nodes(&plan);
        let fused_candidate = region_context.is_none()
            && plan.uses_fused_elementwise_kernel()
            && !matches!(&plan.nodes[plan.output_node_id].op, TensorOp::Input { .. });
        let (source, fused_elementwise) = if fused_candidate {
            match plan.cuda_source() {
                Ok(source) => (source, true),
                Err(_) => (cuda_program_source(&plan, &host_driven)?, false),
            }
        } else {
            (cuda_program_source(&plan, &host_driven)?, false)
        };
        let mut source = source;
        if fused_elementwise {
            source.push_str(optimizer::cuda_optimizer_source());
        }
        if matmul_bias_tanh.is_some() {
            source.push_str(CUDA_MATMUL_BIAS_TANH_SOURCE);
        }
        for region in &fusion_regions {
            source.push_str(&plan.cuda_fusion_region_source(region)?);
        }
        let context = match region_context {
            Some(context) => context.clone(),
            None => {
                let context = CudaContext::new(self.device_ordinal)
                    .map_err(|error| format!("failed to create CUDA context: {error:?}"))?;
                // SAFETY: called before this context allocates anything, so no buffer of it
                // carries events. A plan issues all of its work on the context's default
                // stream; the only other stream, a host-driven loop's graph capture stream
                // (`host_loop`), records work without executing it, and the recorded graphs
                // are launched on the default stream. No buffer is therefore used by two
                // executing streams, which is the synchronization event tracking provides.
                // Without events, buffers can also be read inside a stream capture, where a
                // wait on an event recorded outside the capture is an error.
                unsafe { context.disable_event_tracking() };
                context
            }
        };
        let ptx = compile_ptx(T::program_source(source)?).map_err(|error| {
            format!("failed to compile CUDA device program with NVRTC: {error:?}")
        })?;
        let module = context
            .load_module(ptx)
            .map_err(|error| format!("failed to load CUDA device program: {error:?}"))?;
        let blas = cuda_blas(context.default_stream())?;
        let solver = cuda_solver(context.default_stream())?;
        let mut cond_branches = BTreeMap::new();
        for (node_id, node) in plan.nodes.iter().enumerate() {
            let TensorOp::Region(RegionNode {
                kind: RegionKind::Cond { branches, .. },
                ..
            }) = &node.op
            else {
                continue;
            };
            let compile_region = |region: &TensorExecutionPlan| {
                self.compile_in_context(region.clone(), Some(&context))
                    .map_err(|error| {
                        format!("CUDA Cond node {node_id} region cannot lower: {error}")
                    })
            };
            cond_branches.insert(
                node_id,
                CudaCondBranches {
                    on_true: compile_region(&branches.on_true.plan)?,
                    on_false: compile_region(&branches.on_false.plan)?,
                },
            );
        }
        let mut host_loops = BTreeMap::new();
        for node_id in &host_driven {
            if host_loop::cuda_loop_group_members(&plan, *node_id).first() == Some(node_id) {
                host_loops.insert(
                    *node_id,
                    host_loop::CudaHostLoop::compile(
                        self,
                        &context,
                        *node_id,
                        &plan.nodes[*node_id].op,
                    )?,
                );
            }
        }
        let pool_budget = cuda_pool_frontier_budget(&plan)?;
        Ok(CudaExecutionPlan {
            plan,
            fused_elementwise,
            fused_region_count: fusion_regions.len(),
            matmul_bias_tanh,
            device_ordinal: self.device_ordinal,
            context,
            module,
            blas,
            solver,
            state: Arc::new(Mutex::new(CudaExecutionState {
                free_buffers: CudaBufferPool {
                    budget: pool_budget,
                    ..CudaBufferPool::default()
                },
                ..CudaExecutionState::default()
            })),
            cond_branches,
            host_loops,
            constant_uploads: Arc::new(AtomicUsize::new(0)),
            primary_plan: Arc::new(Mutex::new(None)),
        })
    }
}

impl CudaDataParallelExecutionPlan {
    pub fn replica_count(&self) -> usize {
        self.replicas.len()
    }

    pub fn device_ordinals(&self) -> Vec<usize> {
        self.replicas
            .iter()
            .map(CudaExecutionPlan::device_ordinal)
            .collect()
    }

    pub fn output_node_ids(&self) -> &[TensorNodeId] {
        &self.output_node_ids
    }

    /// Runs every replica, then all-reduces with the caller's `reduction`.
    ///
    /// A plan bound to a sharding schedule accepts this call only when every
    /// scheduled collective uses `reduction`; the schedule is still executed.
    #[cfg(feature = "cuda-nccl")]
    pub fn execute_replicas(
        &self,
        replica_inputs: &[BTreeMap<String, DynamicTensor>],
        reduction: TensorReplicaReduction,
    ) -> Result<CudaDataParallelResult, String> {
        let collectives = match &self.collectives {
            None => self
                .output_node_ids
                .iter()
                .map(|node_id| (*node_id, reduction))
                .collect(),
            Some(scheduled) => {
                if let Some((node_id, planned)) =
                    scheduled.iter().find(|(_, planned)| *planned != reduction)
                {
                    return Err(format!(
                        "CUDA data-parallel sharding schedule reduces node {node_id} with {planned:?}, but execute_replicas requested {reduction:?}; use execute_sharded"
                    ));
                }
                scheduled.clone()
            }
        };
        self.execute_collectives(replica_inputs, &collectives)
    }

    /// Runs every replica, then applies the bound sharding schedule in order.
    #[cfg(feature = "cuda-nccl")]
    pub fn execute_sharded(
        &self,
        replica_inputs: &[BTreeMap<String, DynamicTensor>],
    ) -> Result<CudaDataParallelResult, String> {
        let collectives = self.collectives.as_ref().ok_or_else(|| {
            "CUDA data-parallel plan was compiled without a TensorShardingPlan; use execute_replicas with an explicit reduction"
                .to_string()
        })?;
        self.execute_collectives(replica_inputs, collectives)
    }

    #[cfg(feature = "cuda-nccl")]
    fn execute_collectives(
        &self,
        replica_inputs: &[BTreeMap<String, DynamicTensor>],
        collectives: &[(TensorNodeId, TensorReplicaReduction)],
    ) -> Result<CudaDataParallelResult, String> {
        if replica_inputs.len() != self.replicas.len() {
            return Err(format!(
                "CUDA data-parallel plan has {} replicas but received {} input maps",
                self.replicas.len(),
                replica_inputs.len()
            ));
        }
        // Held for the whole invocation so concurrent callers cannot
        // interleave replica work with another call's collectives.
        let mut guard = self
            .communicators
            .lock()
            .map_err(|_| "CUDA data-parallel NCCL communicator lock is poisoned".to_string())?;
        let communicators = match guard.0.take() {
            Some(communicators) => communicators,
            None => create_nccl_communicators(&self.replicas)
                .map_err(|error| format!("failed to recreate NCCL communicators: {error}"))?,
        };
        match self.run_collectives(replica_inputs, collectives, &communicators) {
            Ok(result) => {
                guard.0 = Some(communicators);
                Ok(result)
            }
            Err(error) => {
                // Aborts any collective this call left partially launched.
                drop(communicators);
                Err(format!(
                    "{error}; the plan's NCCL communicators were aborted and the next call recreates them"
                ))
            }
        }
    }

    /// Enqueues every replica, all-reduces, synchronizes, and reads rank zero.
    #[cfg(feature = "cuda-nccl")]
    fn run_collectives(
        &self,
        replica_inputs: &[BTreeMap<String, DynamicTensor>],
        collectives: &[(TensorNodeId, TensorReplicaReduction)],
        communicators: &[NcclComm],
    ) -> Result<CudaDataParallelResult, String> {
        let enqueue_start = Instant::now();
        for (replica, inputs) in self.replicas.iter().zip(replica_inputs) {
            replica.execute_retaining_without_output(inputs, &BTreeSet::new())?;
        }
        let replica_enqueue = enqueue_start.elapsed();

        let collective_start = Instant::now();
        self.all_reduce_retained_outputs(communicators, collectives)?;
        for replica in &self.replicas {
            replica.synchronize()?;
        }
        let collective = collective_start.elapsed();

        let readback_start = Instant::now();
        let rank_zero = self
            .replicas
            .first()
            .ok_or_else(|| "CUDA data-parallel plan has no replicas".to_string())?;
        let outputs = self
            .output_node_ids
            .iter()
            .map(|node_id| rank_zero.computed_node_to_host(*node_id))
            .collect::<Result<Vec<_>, _>>()?;
        let output_readback = readback_start.elapsed();

        Ok(CudaDataParallelResult {
            outputs,
            timing: CudaDataParallelTiming {
                replica_enqueue,
                collective,
                output_readback,
            },
        })
    }

    #[cfg(not(feature = "cuda-nccl"))]
    pub fn execute_replicas(
        &self,
        _replica_inputs: &[BTreeMap<String, DynamicTensor>],
        _reduction: TensorReplicaReduction,
    ) -> Result<CudaDataParallelResult, String> {
        Err("CUDA data-parallel execution requires the optional cuda-nccl feature".to_string())
    }

    #[cfg(not(feature = "cuda-nccl"))]
    pub fn execute_sharded(
        &self,
        _replica_inputs: &[BTreeMap<String, DynamicTensor>],
    ) -> Result<CudaDataParallelResult, String> {
        Err("CUDA data-parallel execution requires the optional cuda-nccl feature".to_string())
    }

    #[cfg(feature = "cuda-nccl")]
    fn all_reduce_retained_outputs(
        &self,
        communicators: &[NcclComm],
        collectives: &[(TensorNodeId, TensorReplicaReduction)],
    ) -> Result<(), String> {
        for (node_id, reduction) in collectives {
            let operation = match reduction {
                TensorReplicaReduction::Sum => NcclReduceOp::Sum,
                TensorReplicaReduction::Mean => NcclReduceOp::Avg,
            };
            let expected_len = self.replicas[0]
                .plan
                .nodes
                .get(*node_id)
                .ok_or_else(|| format!("CUDA data-parallel output node {node_id} does not exist"))?
                .shape
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                .ok_or_else(|| {
                    format!("CUDA data-parallel output node {node_id} size overflows usize")
                })?;
            // One thread drives every rank, so NCCL requires group semantics:
            // an ungrouped per-rank call may block waiting for the other ranks.
            group_start().map_err(|error| {
                format!("NCCL group start failed for node {node_id}: {error:?}")
            })?;
            let enqueued =
                self.enqueue_all_reduce(communicators, *node_id, &operation, expected_len);
            let ended = group_end()
                .map_err(|error| format!("NCCL group end failed for node {node_id}: {error:?}"));
            enqueued?;
            ended?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda-nccl")]
    fn enqueue_all_reduce(
        &self,
        communicators: &[NcclComm],
        node_id: TensorNodeId,
        operation: &NcclReduceOp,
        expected_len: usize,
    ) -> Result<(), String> {
        for (rank, (communicator, replica)) in communicators.iter().zip(&self.replicas).enumerate()
        {
            let mut state = replica
                .state
                .lock()
                .map_err(|_| format!("CUDA data-parallel replica {rank} state lock is poisoned"))?;
            let buffer = state
                .values
                .get_mut(node_id)
                .and_then(Option::as_mut)
                .ok_or_else(|| {
                    format!(
                        "CUDA data-parallel replica {rank} output node {node_id} was not retained"
                    )
                })?;
            if buffer.len() != expected_len {
                return Err(format!(
                    "CUDA data-parallel replica {rank} output node {node_id} has {} elements, expected {expected_len}",
                    buffer.len()
                ));
            }
            communicator
                .all_reduce_in_place(buffer, operation)
                .map_err(|error| {
                    format!(
                        "NCCL all-reduce failed for output node {node_id} on replica {rank}: {error:?}"
                    )
                })?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda-nccl")]
fn create_nccl_communicators(replicas: &[CudaExecutionPlan]) -> Result<Vec<NcclComm>, String> {
    NcclComm::from_devices(
        replicas
            .iter()
            .map(|replica| replica.context.default_stream())
            .collect(),
    )
    .map_err(|error| format!("failed to initialize NCCL communicators: {error:?}"))
}

/// Pure lowering validation shared by v0.1 compilation and the typed v0.2 facade.
///
/// Every plan CUDA runs for `plan` ([`cuda_region_plans`]) must execute in `float` and have
/// only finite scalar constants, which no CUDA program can spell.
pub(super) fn validate_cuda_plan(plan: &TensorExecutionPlan) -> Result<(), (String, String)> {
    plan.try_for_each_plan(&cuda_region_plans, &mut |plan| {
        ensure_cuda_f32_execution(plan).map_err(|message| ("dtype".into(), message))?;
        cuda_reject_non_finite_constants(plan)
    })
}

/// The plans CUDA runs for region node `node_id` of `plan`, in the plan walk of
/// [`validate_cuda_plan`]: both branches of a `Cond`, which compile as region programs; the
/// traced body of a loop that the fused per-lane kernel lowers, whose symbolic derivative the
/// kernel inlines expression by expression and `cuda_fused_loop_lowering` checks itself; and
/// for a loop that runs host-driven instead, the region programs that `host_loop` builds
/// from it (its body JVP or VJP among them), built here as they will be at compilation.
fn cuda_region_plans<'p>(
    plan: &'p TensorExecutionPlan,
    node_id: TensorNodeId,
    region: &'p RegionNode,
) -> Result<Vec<RegionPlan<'p>>, (String, String)> {
    let traced = || {
        region
            .regions()
            .into_iter()
            .map(RegionPlan::traced)
            .collect::<Vec<_>>()
    };
    if !region.is_loop() {
        return Ok(traced());
    }
    let op = region.label();
    if let RegionKind::ScanVjpJvp { group, .. } = &region.kind {
        cuda_scan_vjp_jvp_group(plan, *group).map_err(|error| {
            (
                op.to_string(),
                format!("CUDA {op} node {node_id} has invalid group bindings: {error}"),
            )
        })?;
    }
    let node_op = &plan.nodes[node_id].op;
    let Some(Err(error)) = cuda_fused_loop_lowering(node_op) else {
        return Ok(traced());
    };
    // A loop the fused per-lane kernel cannot lower runs as a host-driven region loop when
    // each of its regions lowers on its own.
    let regions = host_loop::cuda_host_loop_regions(node_op).map_err(|region_error| {
        (
            op.to_string(),
            format!(
                "CUDA {op} node {node_id} cannot lower to a device loop: {error}; its \
                 host-driven region loop cannot lower either: {region_error}"
            ),
        )
    })?;
    Ok(regions
        .into_iter()
        .map(|region| RegionPlan {
            plan: Cow::Owned(region),
            derived: Some("host-driven loop region"),
        })
        .collect())
}

/// Rejects a non-finite scalar constant, which neither the per-node device program nor a fused
/// kernel can spell as a CUDA literal.
fn cuda_reject_non_finite_constants(plan: &TensorExecutionPlan) -> Result<(), (String, String)> {
    for (node_id, node) in plan.nodes.iter().enumerate() {
        if let TensorOp::ScalarConstant { value } = &node.op {
            if !value.is_finite() {
                return Err((
                    "constant".into(),
                    format!(
                        "CUDA backend does not support non-finite constants (node {node_id} is \
                         {value}); use a finite bound instead, or run the function on the CPU"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Whether the fused per-lane kernel lowers a loop node, with the reason
/// when it cannot; `None` for a node that is not a loop.
pub(super) fn cuda_fused_loop_lowering(op: &TensorOp) -> Option<Result<(), String>> {
    Some(match op {
        // A data-dependent trip count has no fused kernel; `While` always runs as a
        // host-driven region loop.
        TensorOp::Region(RegionNode {
            kind: RegionKind::While { .. },
            ..
        }) => Err("while_loop has a data-dependent trip count".to_string()),
        TensorOp::Region(RegionNode {
            kind: RegionKind::Fori { loop_plan, .. },
            ..
        }) => cuda_fori_body_is_lowerable(loop_plan),
        TensorOp::Region(RegionNode {
            kind: RegionKind::ForiJvp { loop_plan, .. },
            ..
        }) => cuda_fori_jvp_is_lowerable(loop_plan),
        TensorOp::Region(RegionNode {
            kind: RegionKind::ForiVjp {
                loop_plan, target, ..
            },
            ..
        }) => cuda_fori_vjp_plan(loop_plan, target).map(|_| ()),
        TensorOp::Region(RegionNode {
            kind: RegionKind::ForiVjpJvp { plan, .. },
            ..
        }) => cuda_fori_vjp_jvp_is_lowerable(plan),
        TensorOp::Region(RegionNode {
            kind: RegionKind::Scan { scan_plan, .. },
            ..
        }) => cuda_scan_body_is_lowerable(scan_plan),
        TensorOp::Region(RegionNode {
            kind: RegionKind::ScanVjp {
                scan_plan, target, ..
            },
            ..
        }) => cuda_scan_vjp_plans(scan_plan, target).map(|_| ()),
        TensorOp::Region(RegionNode {
            kind: RegionKind::ScanVjpJvp { plan, .. },
            ..
        }) => cuda_scan_vjp_jvp_is_lowerable(plan),
        _ => return None,
    })
}

/// Rejects a `float32` node anywhere in a plan compiled with `f64` buffers.
///
/// A plan has a single floating element type. A `float32` node inside a double
/// plan would skip the per-operation `f32` rounding the CPU reference applies,
/// so such a program is refused instead of silently running at another
/// precision. Bool nodes stay `0`/`1` values of the plan's element type.
///
/// The walk covers every region plan and the symbolic derivative plans of each
/// loop ([`evaluator_region_plans`], built here if still lazy): a fused loop kernel
/// and a host-driven loop's region programs differentiate the same body with
/// the same symbolic transforms, splicing in the same custom rule graphs, so
/// those plans hold every node, and so every dtype, that the CUDA loop
/// programs are built from.
pub(super) fn validate_cuda_float64_plan(
    plan: &TensorExecutionPlan,
) -> Result<(), (String, String)> {
    plan.try_for_each_plan(&evaluator_region_plans, &mut |plan| {
        for (node_id, node) in plan.nodes.iter().enumerate() {
            if node.dtype == TensorDType::F32 {
                return Err((
                    "float32".into(),
                    format!(
                        "CUDA precision=\"float64\" cannot execute float32 node {node_id} ({}): \
                         the float64 lowering runs every floating node in double; cast the \
                         value to float64 or use the default precision",
                        cuda_op_name(&node.op)
                    ),
                ));
            }
        }
        Ok(())
    })
}

#[cfg(feature = "cuda-nccl")]
fn validate_cuda_data_parallel_devices(device_ordinals: &[usize]) -> Result<(), String> {
    if device_ordinals.len() < 2 {
        return Err(
            "CUDA data-parallel execution requires at least two device ordinals".to_string(),
        );
    }
    let mut unique = BTreeSet::new();
    if device_ordinals
        .iter()
        .any(|ordinal| !unique.insert(*ordinal))
    {
        return Err("CUDA data-parallel device ordinals must be unique".to_string());
    }
    ensure_cuda_driver_available()?;
    // SAFETY: probing for the NCCL library only dlopens fixed sonames and drops the handles,
    // as `ensure_cuda_driver_available` does for the driver.
    if !unsafe { cudarc::nccl::sys::is_culib_present() } {
        return Err(
            "NCCL library is unavailable. Install NCCL and add the directory that provides libnccl.so to LD_LIBRARY_PATH".to_string(),
        );
    }
    let available = CudaContext::device_count()
        .map_err(|error| format!("failed to query CUDA device count: {error:?}"))?;
    let available = usize::try_from(available)
        .map_err(|_| "CUDA driver reported a negative device count".to_string())?;
    if let Some(ordinal) = device_ordinals
        .iter()
        .copied()
        .find(|ordinal| *ordinal >= available)
    {
        return Err(format!(
            "CUDA data-parallel device ordinal {ordinal} is unavailable; CUDA reports {available} device(s)"
        ));
    }
    Ok(())
}

/// cudarc loads the CUDA driver lazily and panics when no candidate library loads. Builds with
/// the CUDA feature, such as the published Linux wheels, also run on hosts without an NVIDIA
/// driver, so every entry point that creates a context checks first and fails with an error.
fn ensure_cuda_driver_available() -> Result<(), String> {
    // SAFETY: cudarc's probe dlopens the same fixed driver sonames that `culib` loads and drops
    // each handle immediately; it resolves no symbols and runs only the library's initializers.
    if unsafe { cudarc::driver::sys::is_culib_present() } {
        Ok(())
    } else {
        Err("CUDA driver library is unavailable. Install an NVIDIA driver that provides libcuda.so to use device=\"cuda\"".to_string())
    }
}

fn ensure_nvrtc_runtime_available() -> Result<(), String> {
    const LIBRARY_NAMES: [&str; 6] = [
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "libnvrtc.so.10",
        "libnvrtc.so.9",
    ];

    let mut errors = Vec::with_capacity(LIBRARY_NAMES.len());
    for name in LIBRARY_NAMES {
        // SAFETY: loading a fixed CUDA toolkit soname through the system dynamic linker only runs
        // that library's own initializers, which cudarc runs anyway when it loads the same library;
        // the handle is dropped immediately and no symbol is resolved from it.
        match unsafe { libloading::Library::new(name) } {
            Ok(library) => {
                drop(library);
                return Ok(());
            }
            // libloading 0.9 renders only "dlopen failed" through Display and exposes the
            // dynamic linker's message through Error::source, so append the source to keep
            // the reason (for example a missing file) in the diagnostic.
            Err(error) => match std::error::Error::source(&error) {
                Some(source) => errors.push(format!("{name}: {error}: {source}")),
                None => errors.push(format!("{name}: {error}")),
            },
        }
    }

    Err(format!(
        "CUDA NVRTC runtime is unavailable. Install the Linux CUDA toolkit that provides libnvrtc.so and add its library directory to LD_LIBRARY_PATH. Attempted: {}",
        errors.join("; ")
    ))
}

impl<T: CudaReal> CudaExecutionPlan<T> {
    pub fn plan(&self) -> &TensorExecutionPlan {
        &self.plan
    }

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

    pub fn fused_region_count(&self) -> usize {
        self.fused_region_count
    }

    /// Number of allocated device buffers currently owned by this plan,
    /// including reusable temporary buffers and retained values.
    pub fn device_buffer_count(&self) -> Result<usize, String> {
        let primary = self
            .primary_plan
            .lock()
            .map_err(|_| "CUDA primary plan cache lock is poisoned".to_string())?;
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        let primary_count = primary
            .as_ref()
            .map(CudaExecutionPlan::device_buffer_count)
            .transpose()?
            .unwrap_or(0);
        Ok(state.values.iter().flatten().count()
            + state.free_buffers.values().map(Vec::len).sum::<usize>()
            + primary_count)
    }

    /// Number of array constants this plan has copied to the device.
    ///
    /// The first execution uploads each constant of the plan into a device
    /// buffer that stays read-only and is never recycled, and every later
    /// execution reads that buffer, so the count stops growing after the first
    /// call. `Cond` region plans keep their own counts.
    pub fn constant_upload_count(&self) -> usize {
        self.constant_uploads.load(Ordering::Relaxed)
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

    /// Materializes every output preserved by a shared Tensor IR compilation.
    pub fn execute_many(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        if self.plan.output_node_ids().len() == 1 {
            return self.execute(inputs).map(|value| vec![value]);
        }
        self.execute_retaining_inner(inputs, &BTreeSet::new(), false)?;
        let stream = self.context.default_stream();
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA execution plan state lock is poisoned".to_string())?;
        self.plan
            .output_node_ids()
            .iter()
            .map(|output_node_id| {
                let buffer = state
                    .values
                    .get(*output_node_id)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        format!("CUDA multi-output node {output_node_id} was not evaluated")
                    })?;
                let data = stream.clone_dtoh(buffer).map_err(|error| {
                    format!("failed to copy CUDA multi-output node {output_node_id}: {error:?}")
                })?;
                cuda_host_tensor(&self.plan, *output_node_id, data)
            })
            .collect()
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

    /// Evaluates only the primary output while updating the original retained
    /// input bindings. A later optimizer step therefore sees the latest batch,
    /// and the forward query reads device-resident updated parameters.
    pub fn execute_primary_retaining(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeSet<String>,
    ) -> Result<DynamicTensor, String> {
        if self.plan.output_node_ids().len() == 1 {
            return self.execute_retaining(inputs, retained_inputs);
        }
        validate_cuda_program_inputs(&self.plan, inputs)?;
        let mut cached = self
            .primary_plan
            .lock()
            .map_err(|_| "CUDA primary plan cache lock is poisoned".to_string())?;
        if cached.is_none() {
            let forward = self.plan.as_ir().compile_cpu(self.plan.output_node_id)?;
            *cached = Some(
                CudaBackend::new(self.device_ordinal)
                    .compile_in_context(forward, Some(&self.context))?,
            );
        }
        let forward = cached.as_ref().expect("primary plan was compiled above");
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
        if values.len() != self.plan.nodes.len() {
            *values = std::iter::repeat_with(|| None)
                .take(self.plan.nodes.len())
                .collect();
        }
        for (node_id, node) in self.plan.nodes.iter().enumerate() {
            let TensorOp::Input { name } = &node.op else {
                continue;
            };
            if retained_inputs.contains(name) && values[node_id].is_some() {
                continue;
            }
            let host = T::host_values(inputs[name].storage());
            upload_cuda_input(&stream, &mut values[node_id], free_buffers, &host, name)?;
        }
        let captures = forward
            .plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                TensorOp::Input { name } => Some(name),
                _ => None,
            })
            .map(|name| {
                let node_id = input_node_id(&self.plan, name)?;
                Ok((name.clone(), cuda_value(values, node_id)?))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let output = stream
            .alloc_zeros::<T>(element_count(
                &forward.plan.nodes[forward.plan.output_node_id].shape,
            )?)
            .map_err(|error| format!("failed to allocate CUDA primary output: {error:?}"))?;
        let output = forward.execute_region(&captures, output)?;
        let data = stream
            .clone_dtoh(&output)
            .map_err(|error| format!("failed to copy CUDA primary output: {error:?}"))?;
        cuda_host_tensor(&forward.plan, forward.plan.output_node_id, data)
    }

    /// Executes this `Cond` region with parent device buffers as captures and
    /// hands its output buffer to the caller without a host copy.
    ///
    /// `output` comes from the parent's buffer pool and is written in place, so
    /// repeated executions recycle one buffer instead of growing either pool.
    fn execute_region(
        &self,
        captures: &BTreeMap<String, &CudaSlice<T>>,
        output: CudaSlice<T>,
    ) -> Result<CudaSlice<T>, String> {
        let stream = self.context.default_stream();
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CUDA Cond region state lock is poisoned".to_string())?;
        let CudaExecutionState {
            values,
            free_buffers,
            ..
        } = &mut *state;
        if values.len() != self.plan.nodes.len() {
            *values = std::iter::repeat_with(|| None)
                .take(self.plan.nodes.len())
                .collect();
        }
        recycle_cuda_computed_values(&self.plan, values, free_buffers);
        // A region that returns an array constant keeps the constant's resident buffer and copies
        // it into `output`; every other region writes its result into `output` directly.
        let output_is_constant = matches!(
            self.plan.nodes[self.plan.output_node_id].op,
            TensorOp::Constant { .. }
        );
        let mut output = Some(output);
        if !output_is_constant {
            values[self.plan.output_node_id] = output.take();
        }
        execute_cuda_device_program(
            &self.plan,
            &BTreeMap::new(),
            CudaProgramRuntime {
                stream: &stream,
                module: &self.module,
                blas: self.blas.as_ref(),
                solver: self.solver.as_ref(),
                cond_branches: &self.cond_branches,
                host_loops: &self.host_loops,
                region_captures: captures,
                prebound_inputs: false,
                constant_uploads: &self.constant_uploads,
            },
            values,
            free_buffers,
            &BTreeSet::new(),
            false,
        )?;
        match output {
            Some(mut output) => {
                let constant = cuda_value(values, self.plan.output_node_id)?;
                stream.memcpy_dtod(constant, &mut output).map_err(|error| {
                    format!("failed to copy the CUDA Cond region constant output: {error:?}")
                })?;
                Ok(output)
            }
            None => values
                .get_mut(self.plan.output_node_id)
                .and_then(Option::take)
                .ok_or_else(|| "CUDA Cond region did not produce its output".to_string()),
        }
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
        recycle_cuda_computed_values(&self.plan, values, free_buffers);
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
                    cond_branches: &self.cond_branches,
                    host_loops: &self.host_loops,
                    region_captures: &BTreeMap::new(),
                    prebound_inputs: false,
                    constant_uploads: &self.constant_uploads,
                },
                values,
                free_buffers,
                retained_inputs,
                copy_output,
            )
        }
    }
}

/// The fused SGD and Adam kernels take `float` hyperparameters and update
/// `float` buffers, so device training steps exist only for the default
/// single-precision lowering.
impl CudaExecutionPlan<f32> {
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
            .load_function("quabla_sgd")
            .map_err(|error| format!("failed to load CUDA SGD kernel: {error:?}"))?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(parameter);
        launch.arg(gradient);
        launch.arg(&learning_rate);
        launch.arg(&count);
        // SAFETY: the arguments match `quabla_sgd(float*, const float*, float, unsigned long long)`
        // generated by `legacy_update_kernels` in `cuda_optimizer.rs`; `parameter` and `gradient`
        // both hold `count` elements (checked above), the kernel guards `index < count`, and only
        // the mutably passed `parameter` is written.
        unsafe {
            launch
                .launch(cuda_elementwise_launch(&kernel, launch_count)?)
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
        // A zero learning rate is valid: learning-rate schedules start or end
        // at zero, and a zero step still advances the moments.
        if !(learning_rate.is_finite() && learning_rate >= 0.0)
            || !(beta1.is_finite() && (0.0..1.0).contains(&beta1))
            || !(beta2.is_finite() && (0.0..1.0).contains(&beta2))
            || !(epsilon.is_finite() && epsilon > 0.0)
        {
            return Err("CUDA Adam requires a finite nonnegative learning_rate, a positive finite epsilon, and beta1/beta2 in [0, 1)".to_string());
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
        // v0.1 formed the corrections in float32; kept until v1.0.
        let (correction1, correction2) =
            super::AdamCoefficients::v01_float32_corrections(beta1, beta2, adam.step);
        let count_u64 = u64::try_from(count)
            .map_err(|_| "CUDA Adam parameter count exceeds u64".to_string())?;
        let launch_count = u32::try_from(count)
            .map_err(|_| "CUDA Adam parameter count exceeds u32 launch size".to_string())?;
        let kernel = self
            .module
            .load_function("quabla_adam")
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
        // SAFETY: the arguments match the eleven parameters of `quabla_adam`, generated by
        // `legacy_update_kernels` in `cuda_optimizer.rs`, in order and type; the parameter,
        // gradient, and both moment buffers hold `count` elements (the moments are allocated from
        // the parameter length, which is fixed by the plan's input shape), the kernel guards
        // `index < count`, and only the mutably passed buffers are written.
        unsafe {
            launch
                .launch(cuda_elementwise_launch(&kernel, launch_count)?)
                .map_err(|error| format!("failed to launch CUDA Adam kernel: {error:?}"))?;
        }
        Ok(())
    }
}

impl<T: CudaReal> CudaExecutionPlan<T> {
    pub fn retained_input_to_host(&self, name: &str) -> Result<DynamicTensor, String> {
        let node_id = input_node_id(&self.plan, name)?;
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
        cuda_host_tensor(&self.plan, node_id, data)
    }

    /// Materializes an already-evaluated node from a multi-output plan.
    pub fn computed_node_to_host(&self, node_id: usize) -> Result<DynamicTensor, String> {
        self.plan
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
        cuda_host_tensor(&self.plan, node_id, data)
    }

    /// Copies a retained input directly into another CUDA plan's retained input.
    ///
    /// Call this after every participating gradient plan has evaluated and
    /// updated its own parameter, so all plans observe the same parameter state
    /// on the next iteration.
    pub fn sync_retained_input_to(
        &self,
        source_name: &str,
        target: &CudaExecutionPlan<T>,
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
        ensure_cuda_driver_available()?;
        ensure_nvrtc_runtime_available()?;
        ensure_cuda_f32_execution(plan)?;
        if let Some((lhs, rhs)) = direct_rank_two_matmul_inputs(plan)? {
            return self.execute_rank_two_matmul::<f32>(plan, inputs, lhs, rhs);
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
            .load_function("quabla_fused_elementwise")
            .map_err(|error| format!("failed to load CUDA elementwise kernel: {error:?}"))?;

        let host_inputs = input_nodes
            .iter()
            .map(|(name, _)| f32::host_values(inputs[*name].storage()))
            .collect::<Vec<_>>();
        let device_inputs = host_inputs
            .iter()
            .zip(input_nodes.iter())
            .map(|(input, (name, _))| {
                stream
                    .clone_htod(input.as_ref())
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
        // SAFETY: `plan.cuda_source()` declares one `const float*` per `Input` node in node order,
        // then `float* output` and the `unsigned long long` count, matching the pushes above;
        // inputs were checked against their node shapes and are read through broadcast offsets into
        // the output shape, `device_output` holds `count` elements, and the kernel guards
        // `index < count`.
        unsafe {
            launch
                .launch(cuda_elementwise_launch(&kernel, launch_count)?)
                .map_err(|error| format!("failed to launch CUDA elementwise kernel: {error:?}"))?;
        }
        let data = stream
            .clone_dtoh(&device_output)
            .map_err(|error| format!("failed to copy CUDA output to host: {error:?}"))?;
        cuda_host_tensor(plan, plan.output_node_id, data)
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

struct CudaProgramRuntime<'a, T: CudaReal> {
    stream: &'a Arc<CudaStream>,
    module: &'a Arc<CudaModule>,
    blas: Option<&'a Arc<Mutex<CudaLibraryHandle<CudaBlas>>>>,
    solver: Option<&'a Arc<Mutex<CudaSolver<T>>>>,
    cond_branches: &'a BTreeMap<TensorNodeId, CudaCondBranches<T>>,
    host_loops: &'a BTreeMap<TensorNodeId, host_loop::CudaHostLoop<T>>,
    /// When non-empty, this program is a `Cond` region: every input is bound to a parent-plan
    /// device buffer.
    region_captures: &'a BTreeMap<String, &'a CudaSlice<T>>,
    /// The program is a host-driven loop region whose input buffers the loop has already
    /// written into the value table (see `host_loop`); inputs are neither uploaded nor copied.
    prebound_inputs: bool,
    /// Counts the array-constant uploads of the plan that owns `values`.
    constant_uploads: &'a AtomicUsize,
}

fn cuda_remaining_use_counts(plan: &TensorExecutionPlan) -> Vec<usize> {
    let mut counts = vec![0; plan.nodes.len()];
    for node in plan.nodes.iter() {
        for input in tensor_op_inputs(&node.op) {
            counts[input] += 1;
        }
    }
    counts
}

fn cuda_matmul_bias_tanh_epilogue(
    plan: &TensorExecutionPlan,
) -> Option<CudaMatmulBiasTanhEpilogue> {
    let TensorOp::UnaryMath {
        input: add,
        kind: UnaryMathKind::Tanh,
    } = plan.nodes.get(plan.output_node_id)?.op
    else {
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
    // The epilogue uploads only plan inputs and writes only the primary output: computed operands
    // have no device buffer and other outputs would not be evaluated, so both cases must use the
    // per-node program.
    let is_input = |node_id: usize| matches!(plan.nodes[node_id].op, TensorOp::Input { .. });
    if plan.output_node_ids().len() != 1 || !is_input(lhs) || !is_input(rhs) || !is_input(bias) {
        return None;
    }
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

fn take_cuda_buffer<T: CudaReal>(
    stream: &Arc<CudaStream>,
    free_buffers: &mut CudaBufferPool<T>,
    count: usize,
    node_id: usize,
) -> Result<CudaSlice<T>, String> {
    if let Some(buffer) = free_buffers.take(count) {
        return Ok(buffer);
    }
    if free_buffers.capturing {
        return Err(format!(
            "CUDA node {node_id} needs a new buffer while its loop region is recorded as a graph"
        ));
    }
    stream
        .alloc_zeros::<T>(count)
        .map_err(|error| format!("failed to allocate CUDA node {node_id}: {error:?}"))
}

fn upload_cuda_input<T: CudaReal>(
    stream: &Arc<CudaStream>,
    slot: &mut Option<CudaSlice<T>>,
    free_buffers: &mut CudaBufferPool<T>,
    host: &[T],
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

fn release_cuda_value<T: CudaReal>(
    values: &mut [Option<CudaSlice<T>>],
    free_buffers: &mut CudaBufferPool<T>,
    node_id: usize,
) -> Result<(), String> {
    let slot = values
        .get_mut(node_id)
        .ok_or_else(|| format!("CUDA node {node_id} is missing its buffer slot"))?;
    if let Some(buffer) = slot.take() {
        free_buffers.recycle(buffer);
    }
    Ok(())
}

/// Returns every computed-node buffer that a previous execution left in
/// `values` to the pool, so each execution starts from the value table the
/// first one saw.
///
/// Output nodes keep their buffers after a run for host readback, optimizer
/// steps, and collectives. Structural Scan and loop-gradient groups place a
/// result computed by a sibling node into an empty slot, so a slot still
/// holding the previous run's output would otherwise be rejected. Input slots
/// stay: retained inputs hold device-resident parameters, and the upload
/// overwrites every other input. Array-constant slots stay too: they hold the
/// read-only copy that the first execution uploaded.
fn recycle_cuda_computed_values<T: CudaReal>(
    plan: &TensorExecutionPlan,
    values: &mut [Option<CudaSlice<T>>],
    free_buffers: &mut CudaBufferPool<T>,
) {
    for (slot, node) in values.iter_mut().zip(plan.nodes.iter()) {
        if matches!(node.op, TensorOp::Input { .. } | TensorOp::Constant { .. }) {
            continue;
        }
        if let Some(buffer) = slot.take() {
            free_buffers.recycle(buffer);
        }
    }
}

fn release_dead_cuda_values<T: CudaReal>(
    plan: &TensorExecutionPlan,
    node_id: usize,
    values: &mut [Option<CudaSlice<T>>],
    free_buffers: &mut CudaBufferPool<T>,
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
        if *remaining != 0 || plan.output_node_ids().contains(&input_id) {
            continue;
        }
        match &plan.nodes[input_id].op {
            TensorOp::Input { name } if retained_inputs.contains(name) => continue,
            // An array constant's buffer is read-only and reused by every execution.
            TensorOp::Constant { .. } => continue,
            _ => {}
        }
        release_cuda_value(values, free_buffers, input_id)?;
    }
    Ok(())
}

fn execute_cuda_device_program<T: CudaReal>(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
    runtime: CudaProgramRuntime<'_, T>,
    values: &mut Vec<Option<CudaSlice<T>>>,
    free_buffers: &mut CudaBufferPool<T>,
    retained_inputs: &BTreeSet<String>,
    copy_output: bool,
) -> Result<Option<DynamicTensor>, String> {
    let CudaProgramRuntime {
        stream,
        module,
        blas,
        solver,
        cond_branches,
        host_loops,
        region_captures,
        prebound_inputs,
        constant_uploads,
    } = runtime;
    // A host-driven loop validates every binding when it writes it (`host_loop`).
    if !prebound_inputs {
        if region_captures.is_empty() {
            validate_cuda_program_inputs(plan, inputs)?;
        } else {
            validate_cuda_region_captures(plan, region_captures)?;
        }
    }
    if values.len() != plan.nodes.len() {
        *values = std::iter::repeat_with(|| None)
            .take(plan.nodes.len())
            .collect();
    }
    let mut remaining_uses = cuda_remaining_use_counts(plan);
    let mut fusion_regions = BTreeMap::new();
    let mut fusion_interior_nodes = BTreeSet::new();
    let mut fori_vjp_cache = BTreeMap::<usize, CudaForiVjpCache<T>>::new();
    let mut scan_cache = BTreeMap::<usize, CudaScanCache<T>>::new();
    let mut scan_vjp_cache = BTreeMap::<usize, CudaScanVjpCache<T>>::new();
    let mut scan_vjp_jvp_cache = BTreeMap::<usize, CudaScanVjpJvpCache<T>>::new();
    let mut host_loop_results = BTreeMap::<TensorNodeId, CudaSlice<T>>::new();
    for region in plan.fusion_regions() {
        for node_id in &region.node_ids {
            if *node_id != region.output_node_id {
                fusion_interior_nodes.insert(*node_id);
            }
        }
        fusion_regions.insert(region.output_node_id, region);
    }

    for (node_id, node) in plan.nodes.iter().enumerate() {
        if fusion_interior_nodes.contains(&node_id) {
            continue;
        }
        if let Some(region) = fusion_regions.get(&node_id) {
            execute_cuda_fusion_region(plan, region, stream, module, values, free_buffers)?;
            for fused_node_id in &region.node_ids {
                release_dead_cuda_values(
                    plan,
                    *fused_node_id,
                    values,
                    free_buffers,
                    retained_inputs,
                    &mut remaining_uses,
                )?;
            }
            continue;
        }
        let count = element_count(&node.shape)?;
        let launch_count = u32::try_from(count)
            .map_err(|_| format!("CUDA node {node_id} launch exceeds u32 element count"))?;
        // Host-driven loops: the first node of a group runs the loop and parks its siblings'
        // results until those nodes are reached.
        let host_result = match host_loop_results.remove(&node_id) {
            Some(result) => Some(result),
            None => match host_loops.get(&node_id) {
                Some(host) => {
                    let mut result = None;
                    for (member, buffer) in host.execute(plan, node_id, values, stream)? {
                        if member == node_id {
                            result = Some(buffer);
                        } else {
                            host_loop_results.insert(member, buffer);
                        }
                    }
                    Some(result.ok_or_else(|| {
                        format!("CUDA host-driven loop node {node_id} produced no result")
                    })?)
                }
                None => None,
            },
        };
        if let Some(result) = host_result {
            let slot = values
                .get_mut(node_id)
                .ok_or_else(|| format!("CUDA loop node {node_id} is missing its buffer slot"))?;
            if let Some(stale) = slot.replace(result) {
                free_buffers.recycle(stale);
            }
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
        match &node.op {
            TensorOp::Input { name } => {
                if prebound_inputs {
                    if values.get(node_id).is_none_or(Option::is_none) {
                        return Err(format!("CUDA loop region input {name:?} is not bound"));
                    }
                    continue;
                }
                if let Some(capture) = region_captures.get(name) {
                    // Region captures are copied on the device; the parent plan keeps ownership of
                    // the original buffer.
                    let slot = values.get_mut(node_id).ok_or_else(|| {
                        format!("CUDA region input node {node_id} is missing its buffer")
                    })?;
                    if slot.is_none() {
                        *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                    }
                    stream
                        .memcpy_dtod(*capture, slot.as_mut().expect("allocated above"))
                        .map_err(|error| {
                            format!("failed to bind CUDA region capture {name:?}: {error:?}")
                        })?;
                    continue;
                }
                let slot = values
                    .get_mut(node_id)
                    .ok_or_else(|| format!("CUDA input node {node_id} is missing its buffer"))?;
                if retained_inputs.contains(name) && slot.is_some() {
                    continue;
                }
                let host = T::host_values(inputs[name].storage());
                upload_cuda_input(stream, slot, free_buffers, &host, name)?;
                continue;
            }
            TensorOp::Constant { value } => {
                // Uploaded by the first execution only: the slot survives recycling and dead-value
                // release, and no kernel writes to it.
                let slot = values
                    .get_mut(node_id)
                    .ok_or_else(|| format!("CUDA constant node {node_id} is missing its buffer"))?;
                if slot.is_none() {
                    let host = T::host_values(value.value().storage());
                    *slot = Some(stream.clone_htod(host.as_ref()).map_err(|error| {
                        format!("failed to copy CUDA constant node {node_id}: {error:?}")
                    })?);
                    constant_uploads.fetch_add(1, Ordering::Relaxed);
                }
                continue;
            }
            TensorOp::Linalg { input, kind } => {
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
                    format!(
                        "CUDA {} requires CUSOLVER, but libcusolver could not be loaded",
                        kind.name()
                    )
                })?;
                let input_shape = &plan.nodes[*input].shape;
                let rank = input_shape.len();
                let batch = element_count(&input_shape[..rank - 2])?;
                if decompositions::is_decomposition(*kind) {
                    let stack = decompositions::MatrixStack {
                        batch,
                        m: input_shape[rank - 2],
                        n: input_shape[rank - 1],
                    };
                    let matrix = cuda_value(before, *input)?;
                    decompositions::launch(stream, module, solver, *kind, matrix, output, stack)?;
                } else {
                    launch_cusolver_linalg(
                        stream,
                        module,
                        solver,
                        *kind,
                        cuda_value(before, *input)?,
                        output,
                        batch,
                        input_shape[rank - 1],
                    )?;
                }
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
                let rhs_shape = &plan.nodes[*rhs].shape;
                let rank = rhs_shape.len();
                launch_cusolver_solve(
                    stream,
                    module,
                    solver,
                    cuda_value(before, *matrix)?,
                    cuda_value(before, *rhs)?,
                    output,
                    element_count(&rhs_shape[..rank - 2])?,
                    rhs_shape[rank - 2],
                    rhs_shape[rank - 1],
                )?;
            }
            // `validate_cuda_plan` rejects While before compilation.
            TensorOp::Region(RegionNode {
                kind: RegionKind::While { .. },
                ..
            }) => {
                return Err(format!(
                    "CUDA While node {node_id} reached execution without validation"
                ))
            }
            TensorOp::Region(RegionNode {
                kind: RegionKind::Cond { predicate, .. },
                captures,
                ..
            }) => {
                let branches = cond_branches
                    .get(&node_id)
                    .ok_or_else(|| format!("CUDA Cond node {node_id} has no compiled regions"))?;
                let (before, current_and_after) = values.split_at_mut(node_id);
                // Host synchronization point: clone_dtoh waits for the stream and reads back one
                // f32 predicate; afterwards only the selected branch's kernels are launched, so the
                // other branch is never evaluated on device.
                let predicate =
                    stream
                        .clone_dtoh(cuda_value(before, *predicate)?)
                        .map_err(|error| {
                            format!("failed to read CUDA Cond node {node_id} predicate: {error:?}")
                        })?;
                let region = if cuda_scalar_predicate(&predicate)? {
                    &branches.on_true
                } else {
                    &branches.on_false
                };
                let captures = captures
                    .iter()
                    .map(|(name, capture)| {
                        cuda_value(before, *capture).map(|value| (name.clone(), value))
                    })
                    .collect::<Result<BTreeMap<_, _>, _>>()?;
                let slot = current_and_after
                    .first_mut()
                    .ok_or_else(|| format!("CUDA Cond node {node_id} is missing its buffer"))?;
                let output = match slot.take() {
                    Some(buffer) => buffer,
                    None => take_cuda_buffer(stream, free_buffers, count, node_id)?,
                };
                *slot = Some(region.execute_region(&captures, output)?);
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
            TensorOp::Region(RegionNode {
                kind: RegionKind::Fori { carry, .. },
                captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Fori node {node_id} is missing its buffer slot")
                })?;
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                let output = slot
                    .as_mut()
                    .ok_or_else(|| format!("CUDA Fori node {node_id} buffer was not allocated"))?;
                let kernel = module
                    .load_function(&cuda_node_function_name(node_id))
                    .map_err(|error| {
                        format!("failed to load CUDA Fori node {node_id} kernel: {error:?}")
                    })?;
                launch_cuda_fori_node(
                    stream,
                    &kernel,
                    output,
                    before,
                    *carry,
                    captures,
                    launch_count,
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
            TensorOp::Region(RegionNode {
                kind:
                    RegionKind::ForiJvp {
                        carry,
                        carry_tangent,
                        ..
                    },
                captures,
                tangent_captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Fori JVP node {node_id} is missing its buffer slot")
                })?;
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                let output = slot.as_mut().ok_or_else(|| {
                    format!("CUDA Fori JVP node {node_id} buffer was not allocated")
                })?;
                let kernel = module
                    .load_function(&cuda_node_function_name(node_id))
                    .map_err(|error| {
                        format!("failed to load CUDA Fori JVP node {node_id} kernel: {error:?}")
                    })?;
                launch_cuda_fori_jvp_node(
                    stream,
                    &kernel,
                    output,
                    before,
                    *carry,
                    *carry_tangent,
                    captures,
                    tangent_captures,
                    launch_count,
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
            TensorOp::Region(RegionNode {
                kind:
                    RegionKind::ForiVjp {
                        carry,
                        output_cotangent,
                        loop_plan,
                        target: _,
                        group,
                    },
                captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Fori VJP node {node_id} is missing its buffer slot")
                })?;
                let carry_count = element_count(&loop_plan.carry_shape()?)?;
                if let Some(cached) = fori_vjp_cache.get_mut(group) {
                    let output = cached.results.remove(&node_id).ok_or_else(|| {
                        format!(
                            "CUDA Fori VJP group {group} has no cached result for node {node_id}"
                        )
                    })?;
                    if slot.is_some() {
                        return Err(format!(
                            "CUDA Fori VJP node {node_id} unexpectedly owns a result buffer before cache reuse"
                        ));
                    }
                    *slot = Some(output);
                    if cached.results.is_empty() {
                        fori_vjp_cache.remove(group);
                    }
                } else {
                    let members = cuda_fori_vjp_group(plan, *group)?;
                    if members.first().map(|(member_id, _)| *member_id) != Some(node_id) {
                        return Err(format!(
                            "CUDA Fori VJP group {group} reached node {node_id} before its producer"
                        ));
                    }
                    let tape_count = cuda_loop_tape_states(
                        loop_plan.lower,
                        loop_plan.upper,
                        cuda_loop_checkpoint_block(
                            loop_plan.lower,
                            loop_plan.upper,
                            &loop_plan.carry_shape()?,
                            loop_plan.external_captures(),
                            None,
                            true,
                        ),
                    )
                    .and_then(|states| states.checked_mul(carry_count))
                    .ok_or_else(|| "CUDA Fori VJP carry tape size overflowed usize".to_string())?;
                    let mut tape = take_cuda_buffer(stream, free_buffers, tape_count, node_id)?;
                    #[cfg(test)]
                    CUDA_LOOP_TEST_TAPE_BYTES
                        .with(|value| value.set(tape.len() * std::mem::size_of::<T>()));
                    let carry_launch_count = u32::try_from(carry_count).map_err(|_| {
                        format!("CUDA Fori VJP node {node_id} launch exceeds u32 element count")
                    })?;
                    let mut outputs = Vec::with_capacity(members.len());
                    for (member_id, member_target) in &members {
                        let member_count = element_count(&plan.nodes[*member_id].shape)?;
                        let mut output =
                            take_cuda_buffer(stream, free_buffers, member_count, *member_id)?;
                        // A capture that broadcasts into the carry reduces into its output
                        // with atomicAdd, and a pooled buffer still holds an earlier value, so
                        // every lane of a capture gradient is cleared; carry-shaped captures are
                        // overwritten anyway, which keeps this independent of the kernel's
                        // shape test.
                        if matches!(member_target, TensorForiVjpTarget::External(_)) {
                            stream.memset_zeros(&mut output).map_err(|error| {
                                format!("failed to clear CUDA reduced Fori VJP output: {error:?}")
                            })?;
                        }
                        outputs.push(output);
                    }
                    let kernel = module
                        .load_function(&cuda_node_function_name(node_id))
                        .map_err(|error| {
                            format!("failed to load CUDA Fori VJP node {node_id} kernel: {error:?}")
                        })?;
                    if outputs.len() == 1 {
                        launch_cuda_fori_vjp_node(
                            stream,
                            &kernel,
                            outputs
                                .first_mut()
                                .ok_or_else(|| "CUDA Fori VJP group has no output".to_string())?,
                            &mut tape,
                            before,
                            *carry,
                            *output_cotangent,
                            captures,
                            carry_launch_count,
                        )?;
                    } else {
                        launch_cuda_fori_vjp_group(
                            stream,
                            &kernel,
                            &mut outputs,
                            &mut tape,
                            before,
                            *carry,
                            *output_cotangent,
                            captures,
                            carry_launch_count,
                        )?;
                    }
                    free_buffers.recycle(tape);
                    let mut cached = CudaForiVjpCache::default();
                    for ((member_id, _), output) in members.into_iter().zip(outputs) {
                        if member_id == node_id {
                            *slot = Some(output);
                        } else {
                            cached.results.insert(member_id, output);
                        }
                    }
                    if !cached.results.is_empty() {
                        fori_vjp_cache.insert(*group, cached);
                    }
                }
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
            TensorOp::Region(RegionNode {
                kind:
                    RegionKind::ForiVjpJvp {
                        carry,
                        carry_tangent,
                        output_cotangent,
                        output_cotangent_tangent,
                        plan: fori_plan,
                        ..
                    },
                captures,
                tangent_captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Fori VJP JVP node {node_id} is missing its buffer slot")
                })?;
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                let output = slot.as_mut().ok_or_else(|| {
                    format!("CUDA Fori VJP JVP node {node_id} buffer was not allocated")
                })?;
                let carry_count = element_count(&fori_plan.loop_plan.carry_shape()?)?;
                let tape_count = cuda_loop_tape_states(
                    fori_plan.loop_plan.lower,
                    fori_plan.loop_plan.upper,
                    cuda_loop_checkpoint_block(
                        fori_plan.loop_plan.lower,
                        fori_plan.loop_plan.upper,
                        &fori_plan.loop_plan.carry_shape()?,
                        fori_plan.loop_plan.external_captures(),
                        None,
                        true,
                    ),
                )
                .and_then(|states| states.checked_mul(carry_count))
                .ok_or_else(|| "CUDA Fori VJP JVP carry tape size overflowed usize".to_string())?;
                let mut carry_tape = take_cuda_buffer(stream, free_buffers, tape_count, node_id)?;
                let mut carry_tangent_tape =
                    take_cuda_buffer(stream, free_buffers, tape_count, node_id)?;
                #[cfg(test)]
                CUDA_LOOP_TEST_TAPE_BYTES.with(|value| {
                    value.set(
                        (carry_tape.len() + carry_tangent_tape.len()) * std::mem::size_of::<T>(),
                    )
                });
                let launch_count = u32::try_from(carry_count).map_err(|_| {
                    format!("CUDA Fori VJP JVP node {node_id} launch exceeds u32 element count")
                })?;
                let kernel = module
                    .load_function(&cuda_node_function_name(node_id))
                    .map_err(|error| {
                        format!("failed to load CUDA Fori VJP JVP node {node_id} kernel: {error:?}")
                    })?;
                launch_cuda_fori_vjp_jvp_node(
                    stream,
                    &kernel,
                    output,
                    &mut carry_tape,
                    &mut carry_tangent_tape,
                    before,
                    *carry,
                    *carry_tangent,
                    *output_cotangent,
                    *output_cotangent_tangent,
                    captures,
                    tangent_captures,
                    launch_count,
                )?;
                free_buffers.recycle(carry_tape);
                free_buffers.recycle(carry_tangent_tape);
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
            TensorOp::Region(RegionNode {
                kind:
                    RegionKind::Scan {
                        carry,
                        scan_plan,
                        target,
                        group,
                    },
                captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Scan node {node_id} is missing its buffer slot")
                })?;
                if let Some(cached) = scan_cache.get_mut(group) {
                    let cached_value = match target {
                        TensorScanTarget::Carry => cached.carry.take(),
                        TensorScanTarget::Outputs => cached.outputs.take(),
                    }
                    .ok_or_else(|| {
                        format!("CUDA Scan group {group} has no cached {target:?} result")
                    })?;
                    if slot.is_some() {
                        return Err(format!(
                            "CUDA Scan node {node_id} unexpectedly owns a result buffer before cache reuse"
                        ));
                    }
                    *slot = Some(cached_value);
                    if cached.carry.is_none() && cached.outputs.is_none() {
                        scan_cache.remove(group);
                    }
                } else {
                    let carry_count = element_count(&scan_plan.carry_shape()?)?;
                    let output_shape = scan_plan.output_shape()?;
                    let output_count = element_count(&output_shape)?;
                    let output_step_count = element_count(&scan_plan.body.output_shapes()[1])?;
                    let launch_count =
                        u32::try_from(carry_count.max(output_step_count)).map_err(|_| {
                            format!("CUDA Scan node {node_id} launch exceeds u32 element count")
                        })?;
                    if slot.is_none() {
                        *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                    }
                    let auxiliary_count = match target {
                        TensorScanTarget::Carry => output_count,
                        TensorScanTarget::Outputs => carry_count,
                    };
                    let mut auxiliary =
                        take_cuda_buffer(stream, free_buffers, auxiliary_count, node_id)?;
                    let kernel = module
                        .load_function(&cuda_node_function_name(node_id))
                        .map_err(|error| {
                            format!("failed to load CUDA Scan node {node_id} kernel: {error:?}")
                        })?;
                    match target {
                        TensorScanTarget::Carry => {
                            let output = slot.as_mut().ok_or_else(|| {
                                format!("CUDA Scan node {node_id} buffer was not allocated")
                            })?;
                            launch_cuda_scan_node(
                                stream,
                                &kernel,
                                output,
                                &mut auxiliary,
                                before,
                                *carry,
                                captures,
                                output_step_count,
                                launch_count,
                            )?;
                            scan_cache.insert(
                                *group,
                                CudaScanCache {
                                    carry: None,
                                    outputs: Some(auxiliary),
                                },
                            );
                        }
                        TensorScanTarget::Outputs => {
                            let output = slot.as_mut().ok_or_else(|| {
                                format!("CUDA Scan node {node_id} buffer was not allocated")
                            })?;
                            launch_cuda_scan_node(
                                stream,
                                &kernel,
                                &mut auxiliary,
                                output,
                                before,
                                *carry,
                                captures,
                                output_step_count,
                                launch_count,
                            )?;
                            scan_cache.insert(
                                *group,
                                CudaScanCache {
                                    carry: Some(auxiliary),
                                    outputs: None,
                                },
                            );
                        }
                    }
                }
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
            TensorOp::Region(RegionNode {
                kind:
                    RegionKind::ScanVjp {
                        carry,
                        final_carry_cotangent,
                        output_cotangent,
                        scan_plan,
                        target: _,
                        group,
                    },
                captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Scan VJP node {node_id} is missing its buffer slot")
                })?;
                let carry_count = element_count(&scan_plan.carry_shape()?)?;
                let output_count = element_count(
                    &scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]].shape,
                )?;
                if let Some(cached) = scan_vjp_cache.get_mut(group) {
                    let output = cached.results.remove(&node_id).ok_or_else(|| {
                        format!(
                            "CUDA Scan VJP group {group} has no cached result for node {node_id}"
                        )
                    })?;
                    if slot.is_some() {
                        return Err(format!(
                            "CUDA Scan VJP node {node_id} unexpectedly owns a result buffer before cache reuse"
                        ));
                    }
                    *slot = Some(output);
                    if cached.results.is_empty() {
                        scan_vjp_cache.remove(group);
                    }
                } else {
                    let members = cuda_scan_vjp_group(plan, *group)?;
                    if members.first().map(|(member_id, _)| *member_id) != Some(node_id) {
                        return Err(format!(
                            "CUDA Scan VJP group {group} reached node {node_id} before its producer"
                        ));
                    }
                    let tape_count = cuda_loop_tape_states(
                        scan_plan.lower,
                        scan_plan.upper,
                        cuda_loop_checkpoint_block(
                            scan_plan.lower,
                            scan_plan.upper,
                            &scan_plan.carry_shape()?,
                            scan_plan.external_captures(),
                            Some(
                                &scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]]
                                    .shape,
                            ),
                            !cuda_scan_uses_packed_halves(scan_plan),
                        ),
                    )
                    .and_then(|states| states.checked_mul(carry_count))
                    .ok_or_else(|| "CUDA Scan VJP carry tape size overflowed usize".to_string())?;
                    let mut tape = take_cuda_buffer(stream, free_buffers, tape_count, node_id)?;
                    #[cfg(test)]
                    CUDA_LOOP_TEST_TAPE_BYTES
                        .with(|value| value.set(tape.len() * std::mem::size_of::<T>()));
                    let launch_count = u32::try_from(carry_count).map_err(|_| {
                        format!("CUDA Scan VJP node {node_id} launch exceeds u32 element count")
                    })?;
                    let mut outputs = Vec::with_capacity(members.len());
                    for (member_id, member_target) in &members {
                        let member_count = element_count(&plan.nodes[*member_id].shape)?;
                        let mut output =
                            take_cuda_buffer(stream, free_buffers, member_count, *member_id)?;
                        // A capture that broadcasts into the carry reduces into its output
                        // with atomicAdd, and a pooled buffer still holds an earlier value, so
                        // every lane of a capture gradient is cleared; carry-shaped captures are
                        // overwritten anyway, which keeps this independent of the kernel's
                        // shape test.
                        if matches!(member_target, TensorScanVjpTarget::External(_)) {
                            stream.memset_zeros(&mut output).map_err(|error| {
                                format!("failed to clear CUDA reduced Scan VJP output: {error:?}")
                            })?;
                        }
                        outputs.push(output);
                    }
                    let kernel = module
                        .load_function(&cuda_node_function_name(node_id))
                        .map_err(|error| {
                            format!("failed to load CUDA Scan VJP node {node_id} kernel: {error:?}")
                        })?;
                    if outputs.len() == 1 {
                        launch_cuda_scan_vjp_node(
                            stream,
                            &kernel,
                            outputs
                                .first_mut()
                                .ok_or_else(|| "CUDA Scan VJP group has no output".to_string())?,
                            &mut tape,
                            before,
                            *carry,
                            *final_carry_cotangent,
                            *output_cotangent,
                            captures,
                            launch_count,
                            output_count as u64,
                        )?;
                    } else {
                        launch_cuda_scan_vjp_group(
                            stream,
                            &kernel,
                            &mut outputs,
                            &mut tape,
                            before,
                            *carry,
                            *final_carry_cotangent,
                            *output_cotangent,
                            captures,
                            launch_count,
                            output_count as u64,
                        )?;
                    }
                    free_buffers.recycle(tape);
                    let mut cached = CudaScanVjpCache::default();
                    for ((member_id, _), output) in members.into_iter().zip(outputs) {
                        if member_id == node_id {
                            *slot = Some(output);
                        } else {
                            cached.results.insert(member_id, output);
                        }
                    }
                    if !cached.results.is_empty() {
                        scan_vjp_cache.insert(*group, cached);
                    }
                }
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
            TensorOp::Region(RegionNode {
                kind:
                    RegionKind::ScanVjpJvp {
                        carry,
                        carry_tangent,
                        final_carry_cotangent,
                        final_carry_cotangent_tangent,
                        output_cotangent,
                        output_cotangent_tangent,
                        plan: scan_hvp,
                        group,
                        ..
                    },
                captures,
                tangent_captures,
                ..
            }) => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = current_and_after.first_mut().ok_or_else(|| {
                    format!("CUDA Scan VJP JVP node {node_id} is missing its buffer slot")
                })?;
                let carry_count = element_count(&scan_hvp.scan_plan.carry_shape()?)?;
                if let Some(cached) = scan_vjp_jvp_cache.get_mut(group) {
                    let output = cached.results.remove(&node_id).ok_or_else(|| {
                        format!("CUDA Scan VJP JVP group {group} has no cached result for node {node_id}")
                    })?;
                    if slot.is_some() {
                        return Err(format!("CUDA Scan VJP JVP node {node_id} unexpectedly owns a result buffer before cache reuse"));
                    }
                    *slot = Some(output);
                    if cached.results.is_empty() {
                        scan_vjp_jvp_cache.remove(group);
                    }
                } else {
                    let members = cuda_scan_vjp_jvp_group(plan, *group)?;
                    if members.first().map(|(member_id, _)| *member_id) != Some(node_id) {
                        return Err(format!("CUDA Scan VJP JVP group {group} reached node {node_id} before its producer"));
                    }
                    let output_count = element_count(
                        &scan_hvp.scan_plan.body.plan.nodes
                            [scan_hvp.scan_plan.body.plan.output_node_ids[1]]
                            .shape,
                    )?;
                    let tape_count = cuda_loop_tape_states(
                        scan_hvp.scan_plan.lower,
                        scan_hvp.scan_plan.upper,
                        cuda_loop_checkpoint_block(
                            scan_hvp.scan_plan.lower,
                            scan_hvp.scan_plan.upper,
                            &scan_hvp.scan_plan.carry_shape()?,
                            scan_hvp.scan_plan.external_captures(),
                            Some(
                                &scan_hvp.scan_plan.body.plan.nodes
                                    [scan_hvp.scan_plan.body.plan.output_node_ids[1]]
                                    .shape,
                            ),
                            !cuda_scan_uses_packed_halves(&scan_hvp.scan_plan),
                        ),
                    )
                    .and_then(|states| states.checked_mul(carry_count))
                    .ok_or_else(|| {
                        "CUDA Scan VJP JVP carry tape size overflowed usize".to_string()
                    })?;
                    let mut carry_tape =
                        take_cuda_buffer(stream, free_buffers, tape_count, node_id)?;
                    let mut tangent_tape =
                        take_cuda_buffer(stream, free_buffers, tape_count, node_id)?;
                    #[cfg(test)]
                    CUDA_LOOP_TEST_TAPE_BYTES.with(|value| {
                        value
                            .set((carry_tape.len() + tangent_tape.len()) * std::mem::size_of::<T>())
                    });
                    let launch_count = u32::try_from(carry_count).map_err(|_| {
                        format!("CUDA Scan VJP JVP node {node_id} launch exceeds u32 element count")
                    })?;
                    let mut outputs = Vec::with_capacity(members.len());
                    for (member_id, member_target) in &members {
                        let member_count = element_count(&plan.nodes[*member_id].shape)?;
                        let mut output =
                            take_cuda_buffer(stream, free_buffers, member_count, *member_id)?;
                        // A capture that broadcasts into the carry reduces into its output
                        // with atomicAdd, and a pooled buffer still holds an earlier value, so
                        // every lane of a capture gradient is cleared; carry-shaped captures are
                        // overwritten anyway, which keeps this independent of the kernel's
                        // shape test.
                        if matches!(member_target, TensorScanVjpTarget::External(_)) {
                            stream.memset_zeros(&mut output).map_err(|error| {
                                format!(
                                    "failed to clear CUDA reduced Scan VJP JVP output: {error:?}"
                                )
                            })?;
                        }
                        outputs.push(output);
                    }
                    let kernel = module
                        .load_function(&cuda_node_function_name(node_id))
                        .map_err(|error| {
                            format!(
                                "failed to load CUDA Scan VJP JVP node {node_id} kernel: {error:?}"
                            )
                        })?;
                    launch_cuda_scan_vjp_jvp_group(
                        stream,
                        &kernel,
                        &mut outputs,
                        &mut carry_tape,
                        &mut tangent_tape,
                        before,
                        *carry,
                        *carry_tangent,
                        *final_carry_cotangent,
                        *final_carry_cotangent_tangent,
                        *output_cotangent,
                        *output_cotangent_tangent,
                        captures,
                        tangent_captures,
                        launch_count,
                        output_count as u64,
                    )?;
                    free_buffers.recycle(carry_tape);
                    free_buffers.recycle(tangent_tape);
                    let mut cached = CudaScanVjpJvpCache::default();
                    for ((member_id, _), output) in members.into_iter().zip(outputs) {
                        if member_id == node_id {
                            *slot = Some(output);
                        } else {
                            cached.results.insert(member_id, output);
                        }
                    }
                    if !cached.results.is_empty() {
                        scan_vjp_jvp_cache.insert(*group, cached);
                    }
                }
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
            TensorOp::CholeskyAd {
                inputs: operands,
                kind,
            } => {
                let (before, current_and_after) = values.split_at_mut(node_id);
                let slot = &mut current_and_after[0];
                if slot.is_none() {
                    *slot = Some(take_cuda_buffer(stream, free_buffers, count, node_id)?);
                }
                let scratch_count = count
                    .checked_mul(cholesky_backend::scratch_lanes(*kind))
                    .ok_or_else(|| {
                        "CUDA Cholesky derivative scratch size overflowed usize".to_string()
                    })?;
                let mut scratch = take_cuda_buffer(stream, free_buffers, scratch_count, node_id)?;
                let kernel = module
                    .load_function(&cuda_node_function_name(node_id))
                    .map_err(|error| {
                        format!("failed to load CUDA Cholesky derivative kernel: {error:?}")
                    })?;
                let mut launch = stream.launch_builder(&kernel);
                for operand in operands {
                    launch.arg(cuda_value(before, *operand)?);
                }
                launch.arg(slot.as_mut().expect("allocated above"));
                launch.arg(&mut scratch);
                let config = cuda_fixed_block_launch(
                    &kernel,
                    LaunchConfig {
                        grid_dim: (
                            u32::try_from(
                                node.shape[..node.shape.len() - 2].iter().product::<usize>(),
                            )
                            .map_err(|_| "CUDA Cholesky batch count exceeds u32".to_string())?,
                            1,
                            1,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    "Cholesky derivative",
                )?;
                // SAFETY: generated derivative kernels accept the ordered operand pointers,
                // n*n output lanes, and the exact checked Jet workspace allocation above.
                // One block owns each matrix's factorization and column-ordered reverse
                // traversal, synchronizing its threads between dependent phases.
                unsafe { launch.launch(config) }.map_err(|error| {
                    format!("failed to launch CUDA Cholesky derivative kernel: {error:?}")
                })?;
                free_buffers.recycle(scratch);
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
            TensorOp::ScalarConstant { .. }
            | TensorOp::Cholesky { .. }
            | TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Div { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. }
            | TensorOp::Where { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Powi { .. }
            | TensorOp::UnaryMath { .. }
            | TensorOp::BinaryMath { .. }
            | TensorOp::CumSum { .. }
            | TensorOp::Triangular { .. }
            | TensorOp::Matmul { .. }
            | TensorOp::Sum { .. }
            | TensorOp::Mean { .. }
            | TensorOp::SumAxis { .. }
            | TensorOp::MeanAxis { .. }
            | TensorOp::ExtremumAxis { .. }
            | TensorOp::Transpose { .. }
            | TensorOp::Concat { .. }
            | TensorOp::Slice { .. }
            | TensorOp::PadSlice { .. }
            | TensorOp::Gather { .. }
            | TensorOp::ScatterAdd { .. }
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
            // Like reshape, cast and stop_gradient only copy the float buffer: both sides share
            // the same execution dtype, and AD has already run on the IR. Plan compilation
            // aliases a custom rule node to its value, so one only reaches here unaliased.
            TensorOp::Reshape { input }
            | TensorOp::Cast { input }
            | TensorOp::StopGradient { input }
            | TensorOp::Custom { value: input, .. } => {
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
    let output = values
        .get(plan.output_node_id)
        .and_then(Option::as_ref)
        .ok_or_else(|| "CUDA device program output is missing".to_string())?;
    let data = stream
        .clone_dtoh(output)
        .map_err(|error| format!("failed to copy CUDA program output to host: {error:?}"))?;
    cuda_host_tensor(plan, plan.output_node_id, data).map(Some)
}

fn execute_cuda_fusion_region<T: CudaReal>(
    plan: &TensorExecutionPlan,
    region: &TensorFusionRegion,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    values: &mut [Option<CudaSlice<T>>],
    free_buffers: &mut CudaBufferPool<T>,
) -> Result<(), String> {
    let output_node_id = region.output_node_id;
    let output_node = plan
        .nodes
        .get(output_node_id)
        .ok_or_else(|| format!("CUDA fusion output node {output_node_id} is missing"))?;
    let count = element_count(&output_node.shape)?;
    let launch_count = u32::try_from(count).map_err(|_| {
        format!("CUDA fusion node {output_node_id} launch exceeds u32 element count")
    })?;
    let (before_output, output_and_after) = values.split_at_mut(output_node_id);
    let output = output_and_after
        .first_mut()
        .ok_or_else(|| format!("CUDA fusion output node {output_node_id} is missing its buffer"))?;
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
        .ok_or_else(|| format!("CUDA fusion output node {output_node_id} was not allocated"))?;
    let kernel = module
        .load_function(&format!("quabla_fused_region_{output_node_id}"))
        .map_err(|error| format!("failed to load CUDA fusion region kernel: {error:?}"))?;
    let mut launch = stream.launch_builder(&kernel);
    for input_node_id in &region.input_node_ids {
        launch.arg(cuda_value(before_output, *input_node_id)?);
    }
    let count = u64::try_from(count)
        .map_err(|_| format!("CUDA fusion node {output_node_id} count exceeds u64"))?;
    launch.arg(&mut *output);
    launch.arg(&count);
    // SAFETY: `cuda_fusion_region_source` declares one `const float*` per `region.input_node_ids`
    // entry in the same order, then the output pointer and count; each leaf is a node buffer sized
    // by its node shape and read through broadcast offsets, the output buffer holds `count`
    // elements, and the kernel guards `index < count`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(&kernel, launch_count)?)
            .map_err(|error| format!("failed to launch CUDA fusion region: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_cuda_fused_elementwise_program<T: CudaReal>(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    values: &mut Vec<Option<CudaSlice<T>>>,
    free_buffers: &mut CudaBufferPool<T>,
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
        let slot = values
            .get_mut(node_id)
            .ok_or_else(|| format!("CUDA input node {node_id} is missing its buffer"))?;
        if retained_inputs.contains(name) && slot.is_some() {
            continue;
        }
        let host = T::host_values(inputs[name].storage());
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
            .load_function("quabla_fused_elementwise")
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
        // SAFETY: same contract as `quabla_fused_elementwise` in `CudaBackend::execute`: one
        // `const float*` per `Input` node in node order, then the output and count; input buffers
        // were uploaded from shape-checked host tensors, the output holds `count` elements, and the
        // kernel guards `index < count`.
        unsafe {
            launch
                .launch(cuda_elementwise_launch(&kernel, launch_count)?)
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
    data.map(|data| cuda_host_tensor(plan, plan.output_node_id, data))
        .transpose()
}

#[allow(clippy::too_many_arguments)]
fn execute_cuda_matmul_bias_tanh_program<T: CudaReal>(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    values: &mut Vec<Option<CudaSlice<T>>>,
    free_buffers: &mut CudaBufferPool<T>,
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
        let host = T::host_values(inputs[name].storage());
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
            .load_function("quabla_matmul_bias_tanh")
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
        // SAFETY: the arguments match `quabla_matmul_bias_tanh` in `CUDA_MATMUL_BIAS_TANH_SOURCE`;
        // `cuda_matmul_bias_tanh_epilogue` only matches rank-2 `lhs [rows, inner]`,
        // `rhs [inner, cols]`, `bias [1, cols]` inputs and a `[rows, cols]` output, which is the
        // buffer size and the bound the kernel guards.
        unsafe {
            launch
                .launch(cuda_elementwise_launch(
                    &kernel,
                    u32::try_from(count)
                        .map_err(|_| "CUDA epilogue launch exceeds u32".to_string())?,
                )?)
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
    data.map(|data| cuda_host_tensor(plan, plan.output_node_id, data))
        .transpose()
}

impl CudaBackend {
    fn execute_rank_two_matmul<T: CudaReal>(
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
        let lhs_host = T::host_values(lhs.storage());
        let rhs_host = T::host_values(rhs.storage());

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
            .load_function("quabla_rank_two_matmul")
            .map_err(|error| format!("failed to load CUDA matmul kernel: {error:?}"))?;
        let lhs_device = stream
            .clone_htod(lhs_host.as_ref())
            .map_err(|error| format!("failed to copy CUDA matmul lhs: {error:?}"))?;
        let rhs_device = stream
            .clone_htod(rhs_host.as_ref())
            .map_err(|error| format!("failed to copy CUDA matmul rhs: {error:?}"))?;
        let mut output_device = stream
            .alloc_zeros::<T>(count)
            .map_err(|error| format!("failed to allocate CUDA matmul output: {error:?}"))?;

        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&lhs_device);
        launch.arg(&rhs_device);
        launch.arg(&mut output_device);
        launch.arg(&rows);
        launch.arg(&inner);
        launch.arg(&cols);
        // SAFETY: the arguments match `quabla_rank_two_matmul` in `CUDA_RANK_TWO_MATMUL_SOURCE`;
        // the host buffers hold `rows * inner` and `inner * cols` elements (`DynamicTensor`
        // enforces data/shape agreement), the output holds `rows * cols`, the 16x16 block matches
        // the kernel's `QUABLA_BLOCK` and 32-element shared tiles, and every load and store is
        // bounds-checked against `rows`, `inner`, and `cols`.
        unsafe {
            launch
                .launch(cuda_fixed_block_launch(
                    &kernel,
                    LaunchConfig {
                        grid_dim: (grid_x, grid_y, 1),
                        block_dim: (CUDA_MATMUL_BLOCK, CUDA_MATMUL_BLOCK, 1),
                        shared_mem_bytes: 0,
                    },
                    "matmul",
                )?)
                .map_err(|error| format!("failed to launch CUDA matmul kernel: {error:?}"))?;
        }
        let data = stream
            .clone_dtoh(&output_device)
            .map_err(|error| format!("failed to copy CUDA matmul output to host: {error:?}"))?;
        cuda_host_tensor(plan, plan.output_node_id, data)
    }
}

/// Wraps a device readback as a host tensor of the node's logical dtype (in
/// the default lowering an `f64` node carries the result of its `f32` lowering).
fn cuda_host_tensor<T: CudaReal>(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
    data: Vec<T>,
) -> Result<DynamicTensor, String> {
    let node = plan
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA readback node {node_id} is missing"))?;
    DynamicTensor::from_storage(node.shape.clone(), T::host_storage(data, node.dtype))
}

/// Every CUDA buffer and kernel is `float`; a logical dtype must lower to
/// `f32` (see `TensorDeviceBackend::execution_dtype`) or be rejected here.
fn ensure_cuda_f32_execution(plan: &TensorExecutionPlan) -> Result<(), String> {
    for (node_id, node) in plan.nodes.iter().enumerate() {
        if TensorDeviceBackend::Cuda.execution_dtype(node.dtype)? != TensorDType::F32 {
            return Err(format!(
                "CUDA backend cannot execute node {node_id} of dtype {}",
                node.dtype
            ));
        }
    }
    Ok(())
}

fn validate_cuda_program_inputs(
    plan: &TensorExecutionPlan,
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<(), String> {
    for node in plan.nodes.iter() {
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

/// Every input of a region program must have a parent-plan device buffer with a matching element
/// count.
fn validate_cuda_region_captures<T: CudaReal>(
    plan: &TensorExecutionPlan,
    captures: &BTreeMap<String, &CudaSlice<T>>,
) -> Result<(), String> {
    for node in plan.nodes.iter() {
        if let TensorOp::Input { name } = &node.op {
            let capture = captures
                .get(name)
                .ok_or_else(|| format!("CUDA Cond region capture {name:?} is not bound"))?;
            if capture.len() != element_count(&node.shape)? {
                return Err(format!(
                    "CUDA Cond region capture {name:?} has {} elements, expected shape {:?}",
                    capture.len(),
                    node.shape
                ));
            }
        }
    }
    Ok(())
}

/// Same as the CPU `tensor_scalar_predicate`: non-finite values are rejected and non-zero is true.
fn cuda_scalar_predicate<T: CudaReal>(value: &[T]) -> Result<bool, String> {
    let [value] = value else {
        return Err(format!(
            "conditional predicate must be scalar, got {} elements",
            value.len()
        ));
    };
    let value = value.to_f64();
    if !value.is_finite() {
        return Err("conditional predicate must be finite".to_string());
    }
    Ok(value != 0.0)
}

/// The most threads per block `kernel` can be launched with on its device.
///
/// `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` is derived from the compiled
/// kernel's register and shared-memory use and the device's per-block limits.
/// It falls below 1024 for kernels that need many registers per thread: a
/// block holds at most 64K registers on current devices, so 1024 threads fit
/// only at up to 64 registers each, and a long fused chain of `double` math
/// functions needs more. A launch with more threads than this fails with
/// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
fn cuda_kernel_thread_limit(kernel: &CudaFunction) -> Result<u32, String> {
    let limit = kernel
        .max_threads_per_block()
        .map_err(|error| format!("failed to query the CUDA kernel thread limit: {error:?}"))?;
    u32::try_from(limit)
        .ok()
        .filter(|limit| *limit > 0)
        .ok_or_else(|| format!("CUDA kernel reports an invalid thread limit of {limit}"))
}

/// A one-dimensional launch of one thread per element for `count` elements.
///
/// It serves kernels whose thread `blockIdx.x * blockDim.x + threadIdx.x`
/// computes its element alone and returns past the count. Each element is
/// then the same arithmetic whatever the block size (only the interleaving
/// of the loop VJP kernels' atomic additions follows the hardware schedule,
/// in any case), so the block only has to be launchable: it is
/// `CUDA_ELEMENTWISE_BLOCK` whenever the kernel allows that, and otherwise
/// the kernel's own limit.
fn cuda_elementwise_launch(kernel: &CudaFunction, count: u32) -> Result<LaunchConfig, String> {
    let block = CUDA_ELEMENTWISE_BLOCK.min(cuda_kernel_thread_limit(kernel)?);
    Ok(LaunchConfig {
        grid_dim: (count.div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Checks a launch whose block shape the kernel fixes (shared-memory tiles,
/// or a reduction tree whose summation order follows the block) against the
/// kernel's thread limit, so a kernel that cannot run with that block fails
/// with the limit named instead of `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
fn cuda_fixed_block_launch(
    kernel: &CudaFunction,
    config: LaunchConfig,
    name: &str,
) -> Result<LaunchConfig, String> {
    let (x, y, z) = config.block_dim;
    let threads = x * y * z;
    let limit = cuda_kernel_thread_limit(kernel)?;
    if threads > limit {
        let registers = kernel
            .num_regs()
            .map_or_else(|_| "an unknown number of".to_string(), |r| r.to_string());
        return Err(format!(
            "CUDA {name} kernel needs {threads} threads per block, but this device can launch \
             it with at most {limit} (CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK; the kernel uses \
             {registers} registers per thread)"
        ));
    }
    Ok(config)
}

struct CudaNodeLaunch<'a, T: CudaReal> {
    kernel: &'a CudaFunction,
    output: &'a mut CudaSlice<T>,
    values: &'a [Option<CudaSlice<T>>],
    op: &'a TensorOp,
    plan: &'a TensorExecutionPlan,
    count: usize,
    launch_count: u32,
    output_shape: &'a [usize],
    blas: Option<&'a Arc<Mutex<CudaLibraryHandle<CudaBlas>>>>,
}

fn launch_cuda_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    request: CudaNodeLaunch<'_, T>,
) -> Result<(), String> {
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
        // All-zero bits are +0.0 in both precisions. A device memset needs no host
        // staging buffer, and a CUDA graph can record it (a copy from a temporary host
        // value could not be replayed).
        stream
            .memset_zeros(output)
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
        | TensorOp::Greater { lhs, rhs }
        | TensorOp::Compare { lhs, rhs, .. }
        | TensorOp::BinaryMath { lhs, rhs, .. } => {
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
        TensorOp::Cholesky { input }
        | TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Powi { input, .. }
        | TensorOp::UnaryMath { input, .. }
        | TensorOp::CumSum { input, .. }
        | TensorOp::Transpose { input, .. }
        | TensorOp::Triangular { input, .. } => {
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
        TensorOp::Slice { input, .. }
        | TensorOp::PadSlice { input, .. }
        | TensorOp::Gather { input, .. } => {
            launch.arg(cuda_value(values, *input)?);
            launch.arg(output);
            launch.arg(&count);
        }
        TensorOp::ScatterAdd { base, updates, .. } => {
            launch.arg(cuda_value(values, *base)?);
            launch.arg(cuda_value(values, *updates)?);
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
        TensorOp::SumAxis { input, .. }
        | TensorOp::MeanAxis { input, .. }
        | TensorOp::ExtremumAxis { input, .. } => {
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
    // Reductions, tiled matmul, and Cholesky fix their block shape; every other
    // node kernel computes one element per thread.
    let fixed_block = match op {
        TensorOp::Matmul { lhs, rhs }
            if use_tiled_rank_two_matmul(plan, *lhs, *rhs, output_shape) =>
        {
            let dimensions = dimensions
                .as_ref()
                .ok_or_else(|| "CUDA matmul launch dimensions are missing".to_string())?;
            Some(LaunchConfig {
                grid_dim: (
                    u32::try_from(dimensions[2].div_ceil(CUDA_MATMUL_TILE_U64))
                        .map_err(|_| "CUDA matmul grid width exceeds u32".to_string())?,
                    u32::try_from(dimensions[0].div_ceil(CUDA_MATMUL_TILE_U64))
                        .map_err(|_| "CUDA matmul grid height exceeds u32".to_string())?,
                    1,
                ),
                block_dim: (CUDA_MATMUL_BLOCK, CUDA_MATMUL_BLOCK, 1),
                shared_mem_bytes: 0,
            })
        }
        TensorOp::Sum { .. } | TensorOp::Mean { .. } => {
            let dimensions = dimensions
                .as_ref()
                .ok_or_else(|| "CUDA reduction launch dimensions are missing".to_string())?;
            Some(LaunchConfig {
                grid_dim: (
                    u32::try_from(dimensions[0].div_ceil(CUDA_REDUCTION_BLOCK as u64))
                        .map_err(|_| "CUDA reduction grid size exceeds u32".to_string())?,
                    1,
                    1,
                ),
                block_dim: (CUDA_REDUCTION_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        TensorOp::Cholesky { .. } => Some(LaunchConfig {
            grid_dim: (
                u32::try_from(
                    output_shape[..output_shape.len() - 2]
                        .iter()
                        .product::<usize>(),
                )
                .map_err(|_| "CUDA Cholesky batch count exceeds u32".to_string())?,
                1,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }),
        TensorOp::SumAxis { input, axis }
        | TensorOp::MeanAxis { input, axis }
        | TensorOp::ExtremumAxis { input, axis, .. }
            if plan.nodes[*input].shape[*axis] >= CUDA_REDUCTION_BLOCK as usize =>
        {
            Some(LaunchConfig {
                grid_dim: (launch_count, 1, 1),
                block_dim: (CUDA_REDUCTION_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        _ => None,
    };
    let config = match fixed_block {
        Some(config) => cuda_fixed_block_launch(kernel, config, cuda_op_name(op))?,
        None => cuda_elementwise_launch(kernel, launch_count)?,
    };
    // SAFETY: the argument pushes above follow the per-op signatures emitted by
    // `cuda_program_source` for this node, and the launch shape matches the emitted variant: the
    // 16x16 tiled grid only when `use_tiled_rank_two_matmul` also selected the tiled kernel,
    // `CUDA_REDUCTION_BLOCK` (256) threads for each reduction block (one block per
    // output for long-axis reductions), one block per matrix for Cholesky, and a flat grid
    // otherwise. Operands are node buffers sized by their node shapes, the output holds `count`
    // elements, and every kernel bounds its index by the count or dimensions it receives, except
    // the Cholesky kernel: it replaces `count` with the per-matrix size and offsets both buffers
    // by `blockIdx.x` matrices, which stays in bounds because the grid above launches exactly
    // one block per matrix of the batch.
    unsafe {
        launch.launch(config).map_err(|error| {
            format!("failed to launch CUDA node {}: {error:?}", cuda_op_name(op))
        })?;
    }
    Ok(())
}

fn launch_cuda_fori_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    output: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    launch_count: u32,
) -> Result<(), String> {
    let count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(output);
    launch.arg(&count);
    // SAFETY: the arguments follow the parameter list emitted by `cuda_fori_node_kernel_source`
    // (captures in order, then `initial_carry`, `out`, then the `unsigned long long` counts); every
    // operand is a node buffer of `element_count(node.shape)` elements read through offsets derived
    // from those shapes, and the kernel returns for `index >= count`. Read-only operands are passed
    // by shared reference and written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| format!("failed to launch CUDA Fori device loop: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_fori_jvp_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    output: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    carry_tangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    tangent_captures: &[(String, TensorNodeId)],
    launch_count: u32,
) -> Result<(), String> {
    let count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    for (_, capture) in tangent_captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, carry_tangent)?);
    launch.arg(output);
    launch.arg(&count);
    // SAFETY: the arguments follow the parameter list emitted by `cuda_fori_jvp_node_kernel_source`
    // (captures in order, then tangent captures, `initial_carry`, `initial_carry_tangent`, `out`,
    // then the `unsigned long long` counts); every operand is a node buffer of
    // `element_count(node.shape)` elements read through offsets derived from those shapes, and the
    // kernel returns for `index >= count`. Read-only operands are passed by shared reference and
    // written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| format!("failed to launch CUDA Fori JVP device loop: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_fori_vjp_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    output: &mut CudaSlice<T>,
    tape: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    launch_count: u32,
) -> Result<(), String> {
    let count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, output_cotangent)?);
    launch.arg(tape);
    launch.arg(output);
    launch.arg(&count);
    // SAFETY: the arguments follow the parameter list emitted by `cuda_fori_vjp_node_kernel_source`
    // (captures in order, then `initial_carry`, `output_cotangent`, `carry_tape`, `out`, then the
    // `unsigned long long` counts); every operand is a node buffer of `element_count(node.shape)`
    // elements read through offsets derived from those shapes, the tape holds
    // `(upper - lower + 1) * count` elements so every `(step - lower + 1) * count + index` write is
    // in bounds and is written before it is read, and the kernel returns for `index >= count`.
    // Read-only operands are passed by shared reference and written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| format!("failed to launch CUDA Fori VJP device loop: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_fori_vjp_group<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    outputs: &mut [CudaSlice<T>],
    tape: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    launch_count: u32,
) -> Result<(), String> {
    let count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, output_cotangent)?);
    launch.arg(tape);
    for output in outputs {
        launch.arg(output);
    }
    launch.arg(&count);
    // SAFETY: the arguments follow the parameter list emitted by
    // `cuda_fori_vjp_group_kernel_source` (captures in order, then `initial_carry`,
    // `output_cotangent`, `carry_tape`, one `out_i` per group member, then the `unsigned long long`
    // counts); every operand is a node buffer of `element_count(node.shape)` elements read through
    // offsets derived from those shapes, the tape holds `(upper - lower + 1) * count` elements and
    // is written before it is read, and the kernel returns for `index >= count`. Read-only operands
    // are passed by shared reference and written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| {
                format!("failed to launch grouped CUDA Fori VJP device loop: {error:?}")
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_fori_vjp_jvp_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    output: &mut CudaSlice<T>,
    carry_tape: &mut CudaSlice<T>,
    carry_tangent_tape: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    carry_tangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    output_cotangent_tangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    tangent_captures: &[(String, TensorNodeId)],
    launch_count: u32,
) -> Result<(), String> {
    let count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    for (_, capture) in tangent_captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, carry_tangent)?);
    launch.arg(cuda_value(values, output_cotangent)?);
    launch.arg(cuda_value(values, output_cotangent_tangent)?);
    launch.arg(carry_tape);
    launch.arg(carry_tangent_tape);
    launch.arg(output);
    launch.arg(&count);
    // SAFETY: the arguments follow the parameter list emitted by
    // `cuda_fori_vjp_jvp_node_kernel_source` (captures in order, then tangent captures, the carry,
    // carry tangent, output cotangent and its tangent, both tapes, `out`, then the
    // `unsigned long long` counts); every operand is a node buffer of `element_count(node.shape)`
    // elements read through offsets derived from those shapes, both tapes hold
    // `(upper - lower + 1) * count` elements and are written before they are read, and the kernel
    // returns for `index >= count`. Read-only operands are passed by shared reference and written
    // buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| {
                format!("failed to launch CUDA Fori VJP JVP device loop: {error:?}")
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_scan_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    final_carry: &mut CudaSlice<T>,
    outputs: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    output_count: usize,
    launch_count: u32,
) -> Result<(), String> {
    let carry_count = final_carry.len() as u64;
    let output_count = u64::try_from(output_count)
        .map_err(|_| "CUDA Scan output count exceeds u64".to_string())?;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(final_carry);
    launch.arg(outputs);
    launch.arg(&carry_count);
    launch.arg(&output_count);
    // SAFETY: the arguments follow the parameter list emitted by `cuda_scan_node_kernel_source`
    // (captures in order, then `initial_carry`, `final_carry`, `outputs`, then the
    // `unsigned long long` counts); every operand is a node buffer of `element_count(node.shape)`
    // elements read through offsets derived from those shapes, `final_carry` holds `carry_count`
    // elements and `outputs` holds `(upper - lower) * output_count`, and the kernel returns for
    // `index >= carry_count && index >= output_count`. Read-only operands are passed by shared
    // reference and written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| format!("failed to launch CUDA Scan device loop: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_scan_vjp_node<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    output: &mut CudaSlice<T>,
    tape: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    final_carry_cotangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    launch_count: u32,
    output_count: u64,
) -> Result<(), String> {
    let carry_count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, final_carry_cotangent)?);
    launch.arg(cuda_value(values, output_cotangent)?);
    launch.arg(tape);
    launch.arg(output);
    launch.arg(&carry_count);
    launch.arg(&output_count);
    // SAFETY: the arguments follow the parameter list emitted by `cuda_scan_vjp_node_kernel_source`
    // (captures in order, then `initial_carry`, `final_carry_cotangent`, `output_cotangent`,
    // `carry_tape`, `out`, then the `unsigned long long` counts); every operand is a node buffer of
    // `element_count(node.shape)` elements read through offsets derived from those shapes, the tape
    // holds `(upper - lower + 1) * carry_count` elements and is written before it is read, and the
    // kernel returns for `index >= carry_count`. Read-only operands are passed by shared reference
    // and written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| format!("failed to launch CUDA Scan VJP device loop: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_scan_vjp_group<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    outputs: &mut [CudaSlice<T>],
    tape: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    final_carry_cotangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    launch_count: u32,
    output_count: u64,
) -> Result<(), String> {
    let carry_count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, capture) in captures {
        launch.arg(cuda_value(values, *capture)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, final_carry_cotangent)?);
    launch.arg(cuda_value(values, output_cotangent)?);
    launch.arg(tape);
    for output in outputs {
        launch.arg(output);
    }
    launch.arg(&carry_count);
    launch.arg(&output_count);
    // SAFETY: the arguments follow the parameter list emitted by
    // `cuda_scan_vjp_group_kernel_source` (captures in order, then `initial_carry`,
    // `final_carry_cotangent`, `output_cotangent`, `carry_tape`, one `out_i` per group member, then
    // the `unsigned long long` counts); every operand is a node buffer of
    // `element_count(node.shape)` elements read through offsets derived from those shapes, the tape
    // holds `(upper - lower + 1) * carry_count` elements and is written before it is read,
    // `output_cotangent` holds `(upper - lower) * output_count` elements and is read only at
    // `(step - lower) * output_count + lane` with `lane < output_count`, and the kernel returns for
    // `index >= carry_count`. Read-only operands are passed by shared reference and written
    // buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| {
                format!("failed to launch grouped CUDA Scan VJP device loop: {error:?}")
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cuda_scan_vjp_jvp_group<T: CudaReal>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    outputs: &mut [CudaSlice<T>],
    carry_tape: &mut CudaSlice<T>,
    carry_tangent_tape: &mut CudaSlice<T>,
    values: &[Option<CudaSlice<T>>],
    carry: TensorNodeId,
    carry_tangent: TensorNodeId,
    final_carry_cotangent: TensorNodeId,
    final_carry_cotangent_tangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    output_cotangent_tangent: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    tangent_captures: &[(String, TensorNodeId)],
    launch_count: u32,
    output_count: u64,
) -> Result<(), String> {
    let carry_count = cuda_value(values, carry)?.len() as u64;
    let mut launch = stream.launch_builder(kernel);
    for (_, value) in captures {
        launch.arg(cuda_value(values, *value)?);
    }
    for (_, value) in tangent_captures {
        launch.arg(cuda_value(values, *value)?);
    }
    launch.arg(cuda_value(values, carry)?);
    launch.arg(cuda_value(values, carry_tangent)?);
    launch.arg(cuda_value(values, final_carry_cotangent)?);
    launch.arg(cuda_value(values, final_carry_cotangent_tangent)?);
    launch.arg(cuda_value(values, output_cotangent)?);
    launch.arg(cuda_value(values, output_cotangent_tangent)?);
    launch.arg(carry_tape);
    launch.arg(carry_tangent_tape);
    for output in outputs {
        launch.arg(output);
    }
    launch.arg(&carry_count);
    launch.arg(&output_count);
    // SAFETY: the arguments follow the parameter list emitted by
    // `cuda_scan_vjp_jvp_group_kernel_source` (captures in order, then tangent captures, the carry,
    // carry tangent, both final-carry cotangents, both output cotangents, both tapes, one `out_i`
    // per group member, then the `unsigned long long` counts); every operand is a node buffer of
    // `element_count(node.shape)` elements read through offsets derived from those shapes, both
    // tapes hold `(upper - lower + 1) * carry_count` elements and are written before they are read,
    // and the kernel returns for `index >= carry_count`. Read-only operands are passed by shared
    // reference and written buffers by `&mut`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(kernel, launch_count)?)
            .map_err(|error| {
                format!("failed to launch grouped CUDA Scan VJP JVP device loop: {error:?}")
            })?;
    }
    Ok(())
}

fn cuda_value<T: CudaReal>(
    values: &[Option<CudaSlice<T>>],
    node_id: usize,
) -> Result<&CudaSlice<T>, String> {
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

/// Orders CUDA graph recordings (`host_loop::capture_region`) against the
/// destruction of cuBLAS and cuSOLVER handles.
///
/// Destroying either handle synchronizes the device (documented for
/// `cublasDestroy`; observed for `cusolverDnDestroy`), and synchronizing a
/// device or context while one of its streams is captured is invalid in every
/// capture mode: the driver invalidates that capture. Every plan of a process
/// shares the device's primary context, so a plan dropped on one thread would
/// otherwise abort a region recording on another, which then silently falls
/// back to eager runs. Recordings hold the gate shared, handle destruction
/// exclusively. Creating handles, allocating, freeing, and work on other
/// streams leave a capture of the non-blocking capture stream valid and are not
/// gated.
static CUDA_GRAPH_RECORDING_GATE: RwLock<()> = RwLock::new(());

/// Held while a region is recorded; see `CUDA_GRAPH_RECORDING_GATE`.
fn cuda_graph_recording_guard() -> RwLockReadGuard<'static, ()> {
    // The gate guards no data, so a panic while it was held leaves nothing inconsistent.
    CUDA_GRAPH_RECORDING_GATE
        .read()
        .unwrap_or_else(PoisonError::into_inner)
}

/// A cuBLAS or cuSOLVER handle that is destroyed outside every CUDA graph
/// recording (`CUDA_GRAPH_RECORDING_GATE`).
#[derive(Debug)]
struct CudaLibraryHandle<H>(ManuallyDrop<H>);

impl<H> CudaLibraryHandle<H> {
    fn new(handle: H) -> Self {
        Self(ManuallyDrop::new(handle))
    }
}

impl<H> Deref for CudaLibraryHandle<H> {
    type Target = H;

    fn deref(&self) -> &H {
        &self.0
    }
}

impl<H> DerefMut for CudaLibraryHandle<H> {
    fn deref_mut(&mut self) -> &mut H {
        &mut self.0
    }
}

impl<H> Drop for CudaLibraryHandle<H> {
    fn drop(&mut self) {
        let _gate = CUDA_GRAPH_RECORDING_GATE
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `drop` runs once, and the handle is never used after it.
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

fn cuda_blas(
    stream: Arc<CudaStream>,
) -> Result<Option<Arc<Mutex<CudaLibraryHandle<CudaBlas>>>>, String> {
    if !cuda_library_available("cublas") {
        return Ok(None);
    }
    CudaBlas::new(stream)
        .map(|blas| Some(Arc::new(Mutex::new(CudaLibraryHandle::new(blas)))))
        .map_err(|error| format!("failed to initialize cuBLAS: {error:?}"))
}

#[derive(Debug)]
struct CudaSolver<T: CudaReal> {
    handle: CudaLibraryHandle<DnHandle>,
    /// Keyed by `(batch, n, rhs_columns)`.
    workspaces: BTreeMap<(usize, usize, usize), CudaSolveWorkspace<T>>,
}

#[derive(Debug)]
struct CudaSolveWorkspace<T: CudaReal> {
    factor: CudaSlice<T>,
    column_rhs: CudaSlice<T>,
    pivots: CudaSlice<i32>,
    info: CudaSlice<i32>,
    scratch: CudaSlice<T>,
}

fn cuda_solver<T: CudaReal>(
    stream: Arc<CudaStream>,
) -> Result<Option<Arc<Mutex<CudaSolver<T>>>>, String> {
    if !cuda_library_available("cusolver") {
        return Ok(None);
    }
    DnHandle::new(stream)
        .map(|handle| {
            Some(Arc::new(Mutex::new(CudaSolver {
                handle: CudaLibraryHandle::new(handle),
                workspaces: BTreeMap::new(),
            })))
        })
        .map_err(|error| format!("failed to initialize CUSOLVER: {error:?}"))
}

fn cuda_library_available(name: &str) -> bool {
    [
        format!("lib{name}.so"),
        format!("lib{name}.so.13"),
        format!("lib{name}.so.12"),
        format!("lib{name}.so.11"),
    ]
    .iter()
    // SAFETY: loading a fixed CUDA toolkit soname through the system dynamic linker only runs that
    // library's own initializers, which cudarc runs anyway when it loads the same library; the
    // handle is dropped immediately and no symbol is resolved from it.
    .any(|candidate| unsafe { libloading::Library::new(candidate).is_ok() })
}

/// Transposes each of `batch` contiguous row-major `rows x columns` matrices.
fn launch_cuda_transpose_copy<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch: usize,
    rows: usize,
    columns: usize,
) -> Result<(), String> {
    let count = rows
        .checked_mul(columns)
        .and_then(|count| count.checked_mul(batch))
        .ok_or_else(|| "CUDA transpose element count overflows usize".to_string())?;
    let count =
        u32::try_from(count).map_err(|_| "CUDA transpose element count exceeds u32".to_string())?;
    let batch = u64::try_from(batch).map_err(|_| "CUDA transpose batch exceeds u64".to_string())?;
    let rows = u64::try_from(rows).map_err(|_| "CUDA transpose rows exceed u64".to_string())?;
    let columns =
        u64::try_from(columns).map_err(|_| "CUDA transpose columns exceed u64".to_string())?;
    let kernel = module
        .load_function("quabla_transpose_copy")
        .map_err(|error| format!("failed to load CUDA transpose kernel: {error:?}"))?;
    let mut launch = stream.launch_builder(&kernel);
    launch.arg(input);
    launch.arg(output);
    launch.arg(&batch);
    launch.arg(&rows);
    launch.arg(&columns);
    // SAFETY: the arguments match `quabla_transpose_copy(const float*, float*, unsigned long long,
    // unsigned long long, unsigned long long)`; callers pass `input` and `output` buffers of
    // `batch * rows * columns` elements, and the kernel guards `index < batch * rows * columns`.
    unsafe {
        launch
            .launch(cuda_elementwise_launch(&kernel, count)?)
            .map_err(|error| format!("failed to launch CUDA transpose kernel: {error:?}"))?;
    }
    Ok(())
}

/// Solves `batch` row-major systems `A_b X_b = B_b` (`A_b` is `n x n`, `B_b`
/// is `n x rhs_columns`, both contiguous per batch element) with one LU
/// factorization (`getrf`, partial pivoting) and one `getrs` per element.
/// cuSOLVER's dense API has no strided-batched LU, and the cuBLAS batched
/// LU targets many tiny matrices through pointer arrays; issuing the
/// per-element calls on one stream keeps a single code path for every `n`
/// while the transposes into column-major order stay one launch each.
#[allow(clippy::too_many_arguments)]
fn launch_cusolver_solve<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    solver: &Arc<Mutex<CudaSolver<T>>>,
    matrix: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch: usize,
    n: usize,
    rhs_columns: usize,
) -> Result<(), String> {
    let n_i32 = i32::try_from(n).map_err(|_| "CUSOLVER solve dimension exceeds i32".to_string())?;
    let rhs_columns_i32 = i32::try_from(rhs_columns)
        .map_err(|_| "CUSOLVER solve right-hand-side columns exceed i32".to_string())?;
    let factor_count = n
        .checked_mul(n)
        .and_then(|count| count.checked_mul(batch))
        .ok_or_else(|| "CUSOLVER solve factor size overflows usize".to_string())?;
    let rhs_count = n
        .checked_mul(rhs_columns)
        .and_then(|count| count.checked_mul(batch))
        .ok_or_else(|| "CUSOLVER solve right-hand-side size overflows usize".to_string())?;
    let mut solver = solver
        .lock()
        .map_err(|_| "CUSOLVER handle lock is poisoned".to_string())?;
    let CudaSolver { handle, workspaces } = &mut *solver;
    let key = (batch, n, rhs_columns);
    // A plan may visit many Solve shapes; retaining each factorization workspace
    // accumulates their device storage. Keep only the current shape for reuse.
    if !workspaces.contains_key(&key) {
        workspaces.clear();
    }
    if let std::collections::btree_map::Entry::Vacant(entry) = workspaces.entry(key) {
        let mut factor = stream
            .alloc_zeros::<T>(factor_count)
            .map_err(|error| format!("failed to allocate CUSOLVER factor buffer: {error:?}"))?;
        let mut workspace_elements = 0_i32;
        {
            let (factor_ptr, _factor_read) = factor.device_ptr_mut(stream);
            // SAFETY: the live handle owns this stream, factor holds at least n*n floats,
            // and workspace_elements is a valid host output location.
            unsafe {
                (T::GETRF_BUFFER_SIZE)(
                    handle.cu(),
                    n_i32,
                    n_i32,
                    factor_ptr as *mut T,
                    n_i32,
                    &mut workspace_elements,
                )
                .result()
                .map_err(|error| format!("CUSOLVER Sgetrf workspace query failed: {error:?}"))?;
            }
        }
        let workspace_elements = usize::try_from(workspace_elements)
            .map_err(|_| "CUSOLVER returned a negative workspace size".to_string())?;
        entry.insert(CudaSolveWorkspace {
            factor,
            column_rhs: stream.alloc_zeros::<T>(rhs_count).map_err(|error| {
                format!("failed to allocate CUSOLVER right-hand-side buffer: {error:?}")
            })?,
            pivots: stream
                .alloc_zeros::<i32>(n * batch)
                .map_err(|error| format!("failed to allocate CUSOLVER pivot buffer: {error:?}"))?,
            info: stream
                .alloc_zeros::<i32>(batch)
                .map_err(|error| format!("failed to allocate CUSOLVER status buffer: {error:?}"))?,
            scratch: stream
                .alloc_zeros::<T>(workspace_elements)
                .map_err(|error| format!("failed to allocate CUSOLVER workspace: {error:?}"))?,
        });
    }
    let CudaSolveWorkspace {
        factor,
        column_rhs,
        pivots,
        info,
        scratch,
    } = workspaces
        .get_mut(&key)
        .expect("solver workspace was initialized above");
    launch_cuda_transpose_copy(stream, module, matrix, factor, batch, n, n)?;
    launch_cuda_transpose_copy(stream, module, rhs, column_rhs, batch, n, rhs_columns)?;
    stream
        .memcpy_htod(&vec![0_i32; batch], info)
        .map_err(|error| format!("failed to reset CUSOLVER status: {error:?}"))?;
    {
        let (factor_ptr, _factor_read) = factor.device_ptr_mut(stream);
        let (rhs_ptr, _rhs_read) = column_rhs.device_ptr_mut(stream);
        let (pivot_ptr, _pivot_read) = pivots.device_ptr_mut(stream);
        let (info_ptr, _info_read) = info.device_ptr_mut(stream);
        let (workspace_ptr, _workspace_read) = scratch.device_ptr_mut(stream);
        let element = std::mem::size_of::<T>() as u64;
        let int = std::mem::size_of::<i32>() as u64;
        for index in 0..batch as u64 {
            let factor_ptr = factor_ptr + index * (n * n) as u64 * element;
            let rhs_ptr = rhs_ptr + index * (n * rhs_columns) as u64 * element;
            let pivot_ptr = pivot_ptr + index * n as u64 * int;
            let info_ptr = info_ptr + index * int;
            // SAFETY: every pointer is offset by whole batch elements inside a buffer of `batch`
            // such elements (factor `n * n` column-major with `lda = n`, right-hand side
            // `n * rhs_columns` column-major with `ldb = n`, `n` pivots, one info word), the
            // workspace holds the queried size and is reused sequentially on one stream, and the
            // `device_ptr_mut` guards stay alive across the loop. The handle is bound to the
            // context's default stream, the same stream that ordered the transpose copies, so the
            // inputs are written before they are read.
            unsafe {
                (T::GETRF)(
                    handle.cu(),
                    n_i32,
                    n_i32,
                    factor_ptr as *mut T,
                    n_i32,
                    workspace_ptr as *mut T,
                    pivot_ptr as *mut i32,
                    info_ptr as *mut i32,
                )
                .result()
                .map_err(|error| format!("CUSOLVER Sgetrf failed: {error:?}"))?;
                (T::GETRS)(
                    handle.cu(),
                    cusolver_sys::cublasOperation_t::CUBLAS_OP_N,
                    n_i32,
                    rhs_columns_i32,
                    factor_ptr as *const T,
                    n_i32,
                    pivot_ptr as *const i32,
                    rhs_ptr as *mut T,
                    n_i32,
                    info_ptr as *mut i32,
                )
                .result()
                .map_err(|error| format!("CUSOLVER Sgetrs failed: {error:?}"))?;
            }
        }
    }
    let mut host_info = vec![0_i32; batch];
    stream
        .memcpy_dtoh(info, &mut host_info)
        .map_err(|error| format!("failed to read CUSOLVER status: {error:?}"))?;
    if let Some((index, status)) = host_info
        .iter()
        .enumerate()
        .find(|(_, status)| **status != 0)
    {
        return Err(if batch == 1 {
            format!("CUSOLVER solve failed with devInfo={status}")
        } else {
            format!("CUSOLVER solve failed with devInfo={status} for batch element {index}")
        });
    }
    launch_cuda_transpose_copy(stream, module, column_rhs, output, batch, rhs_columns, n)
}

/// Device kernels of the `Linalg` lowering (see [`launch_cusolver_linalg`]).
/// `quabla_lu_slogdet` reads one LU factor per thread: `det = (-1)^swaps *
/// prod(u_ii)`, with the log-magnitude summed in double precision; a zero
/// pivot gives `(0, -inf)` and a non-finite one `(NaN, NaN)`, as on the CPU.
/// `quabla_symmetric_part` forms `(A + A^T) / 2`. `quabla_eigenvector_signs`
/// flips each eigenvector column so its largest-magnitude component (the
/// lowest index on ties) is positive, the CPU convention.
const CUDA_LINALG_SOURCE: &str = r#"
extern "C" __global__ void quabla_lu_slogdet(const float* factor, const int* pivots, float* out, unsigned long long batch, unsigned long long n, int want_sign) {
    unsigned long long b = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float* lu = factor + b * n * n;
    double sign = 1.0;
    double log_abs = 0.0;
    int singular = 0;
    int invalid = 0;
    for (unsigned long long i = 0; i < n; ++i) {
        float u = lu[i * n + i];
        if (!isfinite(u)) invalid = 1;
        else if (u == 0.0f) singular = 1;
        else { if (u < 0.0f) sign = -sign; log_abs += log(fabs((double)u)); }
        if (pivots[b * n + i] != (int)(i + 1)) sign = -sign;
    }
    float result;
    if (invalid) result = __int_as_float(0x7fc00000);
    else if (singular) result = want_sign ? 0.0f : -__int_as_float(0x7f800000);
    else result = want_sign ? (float)sign : (float)log_abs;
    out[b] = result;
}

extern "C" __global__ void quabla_symmetric_part(const float* input, float* output, unsigned long long batch, unsigned long long n) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long size = n * n;
    if (index >= batch * size) return;
    unsigned long long base = index - index % size;
    unsigned long long row = (index % size) / n;
    unsigned long long column = index % n;
    output[index] = 0.5f * (input[index] + input[base + column * n + row]);
}

extern "C" __global__ void quabla_eigenvector_signs(float* vectors, unsigned long long batch, unsigned long long n) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * n) return;
    float* matrix = vectors + (index / n) * n * n;
    unsigned long long column = index % n;
    unsigned long long largest = 0;
    for (unsigned long long row = 1; row < n; ++row) {
        if (fabsf(matrix[row * n + column]) > fabsf(matrix[largest * n + column])) largest = row;
    }
    if (matrix[largest * n + column] < 0.0f) {
        for (unsigned long long row = 0; row < n; ++row) matrix[row * n + column] = -matrix[row * n + column];
    }
}
"#;

/// Loads a `CUDA_LINALG_SOURCE` kernel of `module`.
fn linalg_kernel(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, String> {
    module
        .load_function(name)
        .map_err(|error| format!("failed to load CUDA kernel {name}: {error:?}"))
}

/// A one-dimensional launch of `kernel` over `count` independent elements.
fn linalg_launch_config(kernel: &CudaFunction, count: usize) -> Result<LaunchConfig, String> {
    let count =
        u32::try_from(count).map_err(|_| "CUDA linalg thread count exceeds u32".to_string())?;
    cuda_elementwise_launch(kernel, count.max(1))
}

/// Evaluates one [`LinalgKind`] output for `batch` row-major `n x n`
/// matrices with cuSOLVER, one call per batch element on the handle's stream
/// (as in [`launch_cusolver_solve`]).
///
/// The determinant kinds factor each matrix with `getrf` (partial
/// pivoting). Because `det(A^T) = det(A)`, the row-major buffer is factored
/// as is, without a transpose; `getrf` reports an exactly singular factor
/// through `info > 0` and still completes it, so a zero pivot reaches
/// `quabla_lu_slogdet` as `(0, -inf)` instead of an error.
///
/// The eigen kinds symmetrize the input, then call `syevd` (divide and
/// conquer, eigenvalues ascending). The symmetric row-major matrix equals its
/// column-major self, and the column-major eigenvector matrix transposed is
/// the row-major matrix whose columns are the eigenvectors.
#[allow(clippy::too_many_arguments)]
fn launch_cusolver_linalg<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    solver: &Arc<Mutex<CudaSolver<T>>>,
    kind: LinalgKind,
    matrix: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch: usize,
    n: usize,
) -> Result<(), String> {
    let name = kind.name();
    let n_i32 = i32::try_from(n).map_err(|_| format!("CUSOLVER {name} dimension exceeds i32"))?;
    let matrix_count = n
        .checked_mul(n)
        .and_then(|count| count.checked_mul(batch))
        .ok_or_else(|| format!("CUSOLVER {name} matrix size overflows usize"))?;
    let solver = solver
        .lock()
        .map_err(|_| "CUSOLVER handle lock is poisoned".to_string())?;
    let handle = &solver.handle;
    let alloc_real = |count: usize| {
        stream
            .alloc_zeros::<T>(count)
            .map_err(|error| format!("failed to allocate CUSOLVER {name} buffer: {error:?}"))
    };
    let alloc_i32 = |count: usize| {
        stream
            .alloc_zeros::<i32>(count)
            .map_err(|error| format!("failed to allocate CUSOLVER {name} buffer: {error:?}"))
    };
    let mut factor = alloc_real(matrix_count)?;
    let mut info = alloc_i32(batch)?;
    let element = std::mem::size_of::<T>() as u64;
    let int = std::mem::size_of::<i32>() as u64;
    let (batch_u64, n_u64) = (batch as u64, n as u64);
    let lu = matches!(kind, LinalgKind::DetSign | LinalgKind::LogAbsDet);
    let mut pivots = alloc_i32(if lu { batch * n } else { 1 })?;
    let mut values = alloc_real(if lu { 1 } else { batch * n })?;
    if lu {
        stream
            .memcpy_dtod(matrix, &mut factor)
            .map_err(|error| format!("failed to copy CUDA {name} input: {error:?}"))?;
    } else {
        let kernel = linalg_kernel(module, "quabla_symmetric_part")?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(matrix);
        launch.arg(&mut factor);
        launch.arg(&batch_u64);
        launch.arg(&n_u64);
        // SAFETY: the arguments match `quabla_symmetric_part(const float*, float*, unsigned long
        // long, unsigned long long)`; `matrix` and `factor` hold `batch * n * n` floats and the
        // kernel guards its index against that count.
        unsafe { launch.launch(linalg_launch_config(&kernel, matrix_count)?) }
            .map_err(|error| format!("failed to launch CUDA symmetrization: {error:?}"))?;
    }
    let mut workspace_elements = 0_i32;
    {
        let (factor_ptr, _factor_guard) = factor.device_ptr_mut(stream);
        let (values_ptr, _values_guard) = values.device_ptr_mut(stream);
        // SAFETY: the live handle owns this stream, `factor` holds at least one `n x n`
        // column-major matrix with `lda = n`, `values` at least `n` floats for `syevd`, and
        // `workspace_elements` is a valid host output location.
        unsafe {
            if lu {
                (T::GETRF_BUFFER_SIZE)(
                    handle.cu(),
                    n_i32,
                    n_i32,
                    factor_ptr as *mut T,
                    n_i32,
                    &mut workspace_elements,
                )
            } else {
                (T::SYEVD_BUFFER_SIZE)(
                    handle.cu(),
                    cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
                    cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    n_i32,
                    factor_ptr as *const T,
                    n_i32,
                    values_ptr as *const T,
                    &mut workspace_elements,
                )
            }
            .result()
            .map_err(|error| format!("CUSOLVER {name} workspace query failed: {error:?}"))?;
        }
    }
    let workspace_elements = usize::try_from(workspace_elements)
        .map_err(|_| "CUSOLVER returned a negative workspace size".to_string())?;
    let mut scratch = alloc_real(workspace_elements.max(1))?;
    {
        let (factor_ptr, _factor_guard) = factor.device_ptr_mut(stream);
        let (values_ptr, _values_guard) = values.device_ptr_mut(stream);
        let (pivot_ptr, _pivot_guard) = pivots.device_ptr_mut(stream);
        let (info_ptr, _info_guard) = info.device_ptr_mut(stream);
        let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(stream);
        let workspace_i32 = i32::try_from(workspace_elements)
            .map_err(|_| "CUSOLVER workspace size exceeds i32".to_string())?;
        for index in 0..batch as u64 {
            let factor_ptr = factor_ptr + index * (n * n) as u64 * element;
            let info_ptr = info_ptr + index * int;
            // SAFETY: each pointer is offset by whole batch elements inside a buffer of `batch`
            // elements (an `n x n` matrix with `lda = n`, `n` pivots or eigenvalues, one info
            // word); the workspace holds the queried size and is reused sequentially on the
            // handle's stream, which also ordered the copy or symmetrization that wrote `factor`;
            // the `device_ptr_mut` guards outlive the loop.
            unsafe {
                if lu {
                    (T::GETRF)(
                        handle.cu(),
                        n_i32,
                        n_i32,
                        factor_ptr as *mut T,
                        n_i32,
                        scratch_ptr as *mut T,
                        (pivot_ptr + index * n as u64 * int) as *mut i32,
                        info_ptr as *mut i32,
                    )
                } else {
                    (T::SYEVD)(
                        handle.cu(),
                        cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
                        cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                        n_i32,
                        factor_ptr as *mut T,
                        n_i32,
                        (values_ptr + index * n as u64 * element) as *mut T,
                        scratch_ptr as *mut T,
                        workspace_i32,
                        info_ptr as *mut i32,
                    )
                }
                .result()
                .map_err(|error| format!("CUSOLVER {name} failed: {error:?}"))?;
            }
        }
    }
    let mut host_info = vec![0_i32; batch];
    stream
        .memcpy_dtoh(&info, &mut host_info)
        .map_err(|error| format!("failed to read CUSOLVER {name} status: {error:?}"))?;
    // `getrf` reports a zero pivot with `info > 0`, which the slogdet kernel
    // handles; any negative `info` is an invalid argument, and a positive
    // `syevd` `info` means the eigensolver did not converge.
    if let Some((index, status)) = host_info
        .iter()
        .enumerate()
        .find(|(_, status)| **status < 0 || (!lu && **status > 0))
    {
        return Err(format!(
            "CUSOLVER {name} failed with devInfo={status} for batch element {index}"
        ));
    }
    match kind {
        LinalgKind::DetSign | LinalgKind::LogAbsDet => {
            let want_sign = i32::from(kind == LinalgKind::DetSign);
            let kernel = linalg_kernel(module, "quabla_lu_slogdet")?;
            let mut launch = stream.launch_builder(&kernel);
            launch.arg(&factor);
            launch.arg(&pivots);
            launch.arg(output);
            launch.arg(&batch_u64);
            launch.arg(&n_u64);
            launch.arg(&want_sign);
            // SAFETY: the arguments match `quabla_lu_slogdet(const float*, const int*, float*,
            // unsigned long long, unsigned long long, int)`; `factor` holds `batch` LU factors of
            // `n * n` floats, `pivots` `batch * n` ints, `output` `batch` floats, and the kernel
            // guards its index against `batch`.
            unsafe { launch.launch(linalg_launch_config(&kernel, batch)?) }
                .map(|_| ())
                .map_err(|error| format!("failed to launch CUDA slogdet kernel: {error:?}"))
        }
        LinalgKind::EighValues => stream
            .memcpy_dtod(&values, output)
            .map_err(|error| format!("failed to copy CUDA eigenvalues: {error:?}")),
        LinalgKind::EighVectors => {
            launch_cuda_transpose_copy(stream, module, &factor, output, batch, n, n)?;
            let kernel = linalg_kernel(module, "quabla_eigenvector_signs")?;
            let mut launch = stream.launch_builder(&kernel);
            launch.arg(output);
            launch.arg(&batch_u64);
            launch.arg(&n_u64);
            // SAFETY: the arguments match `quabla_eigenvector_signs(float*, unsigned long long,
            // unsigned long long)`; `output` holds `batch` row-major `n x n` matrices, and each
            // thread (guarded against `batch * n`) touches only its own column.
            unsafe { launch.launch(linalg_launch_config(&kernel, batch * n)?) }
                .map(|_| ())
                .map_err(|error| {
                    format!("failed to launch CUDA eigenvector sign kernel: {error:?}")
                })
        }
        _ => Err(format!(
            "{name} is lowered by the CUDA QR and SVD path, not the LU and eigen path"
        )),
    }
}

fn launch_cublas_rank_two_matmul<T: CudaReal>(
    blas: &Arc<Mutex<CudaLibraryHandle<CudaBlas>>>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
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
        alpha: T::from_f64(1.0),
        lda: cols,
        ldb: inner,
        beta: T::from_f64(0.0),
        ldc: cols,
    };
    let blas = blas
        .lock()
        .map_err(|_| "cuBLAS handle lock is poisoned".to_string())?;
    // SAFETY: the row-major `[rows, inner] x [inner, cols]` product is issued as column-major
    // `C^T = B^T * A^T`, so `rhs` is `cols x inner` with `lda = cols`, `lhs` is `inner x rows` with
    // `ldb = inner`, and `output` is `cols x rows` with `ldc = cols`; the node buffers hold exactly
    // those element counts because `matmul_shape` fixed the rank-2 shapes, and the handle is bound
    // to the stream that produced them.
    unsafe { T::gemm(&blas, config, rhs, lhs, output) }
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

fn launch_cublas_batched_matmul<T: CudaReal>(
    blas: &Arc<Mutex<CudaLibraryHandle<CudaBlas>>>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
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
            alpha: T::from_f64(1.0),
            lda: cols,
            ldb: inner,
            beta: T::from_f64(0.0),
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
    // SAFETY: each batch uses the rank-2 layout of `launch_cublas_rank_two_matmul`;
    // `cublas_batch_stride` only accepts operands whose batch shape equals the output batch shape
    // (stride = matrix size) or is all ones (stride 0), so batch `batch_count - 1` stays inside
    // every buffer, and the output holds `batch_count * rows * cols` elements.
    unsafe { T::gemm_strided_batched(&blas, config, rhs, lhs, output) }
        .map_err(|error| format!("cuBLAS strided-batched SGEMM failed: {error:?}"))
}

fn cuda_value_mut<T: CudaReal>(
    values: &mut [Option<CudaSlice<T>>],
    node_id: usize,
) -> Result<&mut CudaSlice<T>, String> {
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

fn cuda_fori_body_is_lowerable(loop_plan: &TensorForiExecutionPlan) -> Result<(), String> {
    let body = &loop_plan.body.plan;
    ensure_cuda_f32_execution(body)?;
    let carry_shape = loop_plan.carry_shape()?;
    if body.output_shape()? != carry_shape {
        return Err("body output shape does not match carry shape".to_string());
    }
    for (node_id, node) in body.nodes.iter().enumerate() {
        match &node.op {
            TensorOp::Input { name } => {
                if !loop_plan.body.captures().contains_key(name) {
                    return Err(format!("body input {name:?} is not a declared capture"));
                }
            }
            TensorOp::ScalarConstant { value } if value.is_finite() => {}
            TensorOp::Constant { .. } => return Err(cuda_loop_constant_error(node_id)),
            TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Div { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. }
            | TensorOp::Where { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Powi { .. }
            | TensorOp::StopGradient { .. }
            | TensorOp::Custom { .. }
            | TensorOp::Broadcast { .. }
            | TensorOp::Cast { .. } => {}
            TensorOp::UnaryMath { kind, .. } if kind.cuda_loop_lowerable() => {}
            TensorOp::BinaryMath { kind, .. } if kind.cuda_loop_lowerable() => {}
            TensorOp::Reshape { input } if body.nodes[*input].shape == node.shape => {}
            _ => {
                return Err(format!(
                    "body node {node_id} uses unsupported {} operation",
                    cuda_op_name(&node.op)
                ))
            }
        }
    }
    Ok(())
}

/// Fused device loops compile their body into one elementwise expression whose
/// arrays are all captures of the parent plan; an array constant in the body
/// has no binding there, so such a loop runs as a host-driven region loop,
/// whose body program uploads the constant once.
fn cuda_loop_constant_error(node_id: usize) -> String {
    format!(
        "body node {node_id} is an array constant; fused CUDA device loops read arrays only \
         through captures"
    )
}

fn cuda_scan_body_is_lowerable(scan_plan: &TensorScanExecutionPlan) -> Result<(), String> {
    ensure_cuda_f32_execution(&scan_plan.body.plan)?;
    let carry_shape = scan_plan.carry_shape()?;
    let output_shape = scan_plan
        .body
        .output_shapes()
        .get(1)
        .ok_or_else(|| "Scan body output is missing".to_string())?;
    let output_matches_carry_lanes = element_count(output_shape)? == element_count(&carry_shape)?;
    if !output_matches_carry_lanes && !cuda_shapes_broadcastable(output_shape, &carry_shape) {
        return Err(format!(
            "Scan body output shape {output_shape:?} must broadcast the carry shape {carry_shape:?}"
        ));
    }
    let packed_pair = cuda_scan_is_packed_pair_layout(scan_plan)?;
    // With equal-length lanes and every capture holding 1 or lane_count elements, all leaves read
    // by flat lane index, so only an element-count-preserving reshape is an identity lane mapping.
    let lane_count = element_count(&carry_shape)?;
    let mut identity_reshape_lanes = output_matches_carry_lanes.then_some(lane_count);
    for shape in scan_plan.external_captures().values() {
        let capture_count = element_count(shape)?;
        if capture_count != 1 && capture_count != lane_count {
            identity_reshape_lanes = None;
        }
    }
    for (name, shape) in scan_plan.external_captures() {
        if !cuda_shapes_broadcastable(&carry_shape, shape) {
            return Err(format!(
                "Scan capture {name:?} shape {shape:?} cannot broadcast to carry shape {carry_shape:?}"
            ));
        }
        if element_count(output_shape)? != element_count(shape)?
            && !cuda_shapes_broadcastable(output_shape, shape)
        {
            return Err(format!(
                "Scan capture {name:?} shape {shape:?} cannot broadcast to output shape {output_shape:?}"
            ));
        }
    }
    for (node_id, node) in scan_plan.body.plan.nodes.iter().enumerate() {
        let node_element_count = element_count(&node.shape)?;
        match &node.op {
            TensorOp::Input { name } => {
                if !scan_plan.body.captures().contains_key(name) {
                    return Err(format!(
                        "Scan body input {name:?} is not a declared capture"
                    ));
                }
            }
            TensorOp::ScalarConstant { value } if value.is_finite() => {}
            TensorOp::Constant { .. } => return Err(cuda_loop_constant_error(node_id)),
            TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Div { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. }
            | TensorOp::Where { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Powi { .. }
            | TensorOp::StopGradient { .. }
            | TensorOp::Custom { .. }
            | TensorOp::Broadcast { .. }
            | TensorOp::Cast { .. } => {}
            TensorOp::UnaryMath { kind, .. } if kind.cuda_loop_lowerable() => {}
            TensorOp::BinaryMath { kind, .. } if kind.cuda_loop_lowerable() => {}
            TensorOp::Reshape { input }
                if cuda_shapes_match_without_leading_units(
                    &scan_plan.body.plan.nodes[*input].shape,
                    &node.shape,
                ) || identity_reshape_lanes == Some(node_element_count) => {}
            TensorOp::Reshape { input } => {
                return Err(format!(
                    "Scan body node {node_id} reshape {:?} -> {:?} changes the per-lane broadcast layout; CUDA Scan lowers only reshapes that add or drop leading unit axes, or equal-lane reshapes whose captures are scalar or carry-sized",
                    scan_plan.body.plan.nodes[*input].shape, node.shape
                ))
            }
            TensorOp::Slice {
                input,
                axis,
                start: _,
                length,
            } if packed_pair
                && *axis == 0
                && scan_plan.body.plan.nodes[*input].shape == carry_shape
                && *length == 1
                && node.shape.len() == carry_shape.len()
                && node.shape[0] == 1
                && node.shape[1..] == carry_shape[1..] => {}
            TensorOp::Concat { inputs, axis }
                if packed_pair
                    && *axis == 0
                    && node.shape.first() == Some(&2)
                    && node_element_count
                        == element_count(&node.shape[1..])?
                            .checked_mul(2)
                            .ok_or_else(|| {
                                "CUDA Scan packed concat element count overflows usize".to_string()
                            })?
                    && inputs.iter().all(|input| {
                        let shape = &scan_plan.body.plan.nodes[*input].shape;
                        shape.first() == Some(&1)
                            && element_count(&shape[1..])
                                .and_then(|count| count.checked_mul(2).ok_or_else(|| {
                                    "CUDA Scan packed concat element count overflows usize"
                                        .to_string()
                                }))
                                .is_ok_and(|count| count == node_element_count)
                    }) => {}
            _ => {
                return Err(format!(
                    "Scan body node {node_id} uses unsupported {} operation with shape {:?}: {:?}",
                    cuda_op_name(&node.op), node.shape, node.op
                ))
            }
        }
    }
    Ok(())
}

/// Symbolic JVP represents one Scan carry as `[primal, tangent, ...]` with a
/// leading extent of two. A tangent lane reads its primal partner, so a body
/// that slices or concatenates this layout is lowered with both halves held
/// in registers by every thread (see `cuda_scan_packed_node_kernel_source`).
/// This is deliberately not a general strided/indexed Scan lowering.
fn cuda_scan_is_packed_pair_layout(scan_plan: &TensorScanExecutionPlan) -> Result<bool, String> {
    let carry_shape = scan_plan.carry_shape()?;
    Ok(carry_shape.first() == Some(&2))
}

/// Returns true when the body addresses packed-pair halves through slice or
/// concat nodes, which the lane-local loop cannot evaluate.
fn cuda_scan_uses_packed_halves(scan_plan: &TensorScanExecutionPlan) -> bool {
    scan_plan
        .body
        .plan
        .nodes
        .iter()
        .any(|node| matches!(node.op, TensorOp::Slice { .. } | TensorOp::Concat { .. }))
}

/// Flat stride of the packed carry axis inside `reference_shape`. Equal-count
/// references map lanes to carry lanes by identity; broadcast references keep
/// the carry axes trailing-aligned.
fn cuda_scan_packed_stride(
    reference_shape: &[usize],
    carry_shape: &[usize],
) -> Result<usize, String> {
    let reference_count = element_count(reference_shape)?;
    if reference_count == element_count(carry_shape)? {
        return Ok(reference_count / 2);
    }
    let axis = reference_shape
        .len()
        .checked_sub(carry_shape.len())
        .ok_or_else(|| "CUDA Scan packed reference has lower rank than the carry".to_string())?;
    element_count(&reference_shape[axis + 1..])
}

/// Reference lane at the same position as `index` but inside packed `half`.
fn cuda_scan_packed_half_lane(half: usize, stride: usize) -> String {
    format!(
        "(index - ((index / {stride}ULL) % 2ULL) * {stride}ULL + {}ULL)",
        half * stride
    )
}

/// One flat scalar block per expression keeps shared region DAGs linear in
/// emitted statements. Pure operand computations preserve their original
/// arithmetic; `where` still selects its value with a ternary.
#[derive(Default)]
struct CudaScalarExpressionBuilder {
    references: std::cell::RefCell<BTreeMap<CudaScalarReferenceKey, String>>,
    statements: std::cell::RefCell<Vec<String>>,
}

type CudaScalarReferenceKey = (TensorNodeId, Option<(usize, usize)>);

impl CudaScalarExpressionBuilder {
    fn emit(
        &self,
        node_id: TensorNodeId,
        half: Option<(usize, usize)>,
        expression: String,
    ) -> String {
        let reference = format!("quabla_body_value_{}", self.statements.borrow().len());
        self.statements
            .borrow_mut()
            .push(format!("const float {reference} = {expression};"));
        self.references
            .borrow_mut()
            .insert((node_id, half), reference.clone());
        reference
    }

    fn finish(&self, reference: &str) -> String {
        format!(
            "([&]() -> float {{ {} return {reference}; }}())",
            self.statements.borrow().join("\n")
        )
    }
}

/// The shared CUDA formula (`cuda_elementwise_formula`) of `op` in a fused
/// loop body or loop derivative body, operands spelled by `operand`;
/// `None` for the ops the loop generators resolve themselves and for the
/// math kinds fused loops do not admit (`cuda_loop_lowerable`).
fn cuda_loop_formula(
    op: &TensorOp,
    operand: impl Fn(TensorNodeId) -> Result<String, String>,
) -> Result<Option<String>, String> {
    let admitted = match op {
        TensorOp::UnaryMath { kind, .. } => kind.cuda_loop_lowerable(),
        TensorOp::BinaryMath { kind, .. } => kind.cuda_loop_lowerable(),
        _ => true,
    };
    if !admitted {
        return Ok(None);
    }
    cuda_elementwise_formula(op, |_, id| operand(id))
}

fn cuda_fori_body_expression(
    loop_plan: &TensorForiExecutionPlan,
    node_id: TensorNodeId,
    capture_parameters: &BTreeMap<String, usize>,
    carry_shape: &[usize],
) -> Result<String, String> {
    let builder = CudaScalarExpressionBuilder::default();
    let reference = cuda_fori_body_expression_inner(
        loop_plan,
        node_id,
        capture_parameters,
        carry_shape,
        &builder,
    )?;
    Ok(builder.finish(&reference))
}

fn cuda_fori_body_expression_inner(
    loop_plan: &TensorForiExecutionPlan,
    node_id: TensorNodeId,
    capture_parameters: &BTreeMap<String, usize>,
    carry_shape: &[usize],
    builder: &CudaScalarExpressionBuilder,
) -> Result<String, String> {
    if let Some(reference) = builder.references.borrow().get(&(node_id, None)) {
        return Ok(reference.clone());
    }
    let body = &loop_plan.body.plan;
    let node = body
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA Fori body node {node_id} is missing"))?;
    let child = |child_id| {
        cuda_fori_body_expression_inner(
            loop_plan,
            child_id,
            capture_parameters,
            carry_shape,
            builder,
        )
    };
    let expression = match &node.op {
        TensorOp::Input { name } if name == &loop_plan.carry_name => Ok("carry".to_string()),
        TensorOp::Input { name } if name == &loop_plan.index_name => Ok("loop_index".to_string()),
        TensorOp::Input { name } => {
            let parameter = capture_parameters.get(name).ok_or_else(|| {
                format!("CUDA Fori body input {name:?} has no parent capture binding")
            })?;
            let offset = cuda_offset_expression(carry_shape, &node.shape);
            Ok(format!("capture_{parameter}[{offset}]"))
        }
        TensorOp::ScalarConstant { value } => Ok(cuda_scalar_literal(*value)),
        // Cast source and target both execute as float, so the cast is the identity; AD has
        // already run, so stop_gradient and a custom rule node are the identity too.
        TensorOp::Broadcast { input }
        | TensorOp::Reshape { input }
        | TensorOp::Cast { input }
        | TensorOp::StopGradient { input }
        | TensorOp::Custom { value: input, .. } => child(*input),
        op => cuda_loop_formula(op, child)?.ok_or_else(|| {
            format!(
                "CUDA Fori body {} is not elementwise-lowerable",
                cuda_op_name(op)
            )
        }),
    }?;
    Ok(builder.emit(node_id, None, expression))
}

fn cuda_scan_body_expression(
    scan_plan: &TensorScanExecutionPlan,
    node_id: TensorNodeId,
    capture_parameters: &BTreeMap<String, usize>,
    carry_shape: &[usize],
) -> Result<String, String> {
    cuda_scan_body_expression_with_reference(
        scan_plan,
        node_id,
        capture_parameters,
        carry_shape,
        "carry",
    )
}

fn cuda_scan_body_expression_with_reference(
    scan_plan: &TensorScanExecutionPlan,
    node_id: TensorNodeId,
    capture_parameters: &BTreeMap<String, usize>,
    reference_shape: &[usize],
    carry_expression: &str,
) -> Result<String, String> {
    cuda_scan_body_expression_in_half(
        scan_plan,
        node_id,
        capture_parameters,
        reference_shape,
        carry_expression,
        None,
    )
}

/// `packed_half = Some((half, stride))` evaluates the node inside one half of
/// a packed-pair layout: the carry reads register `{carry_expression}_{half}`,
/// captures read the reference lane at the same position in that half, and
/// slice/concat select their half statically. Without it, slice and concat are
/// rejected because a lane-local register cannot read its partner half.
fn cuda_scan_body_expression_in_half(
    scan_plan: &TensorScanExecutionPlan,
    node_id: TensorNodeId,
    capture_parameters: &BTreeMap<String, usize>,
    reference_shape: &[usize],
    carry_expression: &str,
    packed_half: Option<(usize, usize)>,
) -> Result<String, String> {
    let builder = CudaScalarExpressionBuilder::default();
    let reference = cuda_scan_body_expression_in_half_inner(
        scan_plan,
        node_id,
        capture_parameters,
        reference_shape,
        carry_expression,
        packed_half,
        &builder,
    )?;
    Ok(builder.finish(&reference))
}

fn cuda_scan_body_expression_in_half_inner(
    scan_plan: &TensorScanExecutionPlan,
    node_id: TensorNodeId,
    capture_parameters: &BTreeMap<String, usize>,
    reference_shape: &[usize],
    carry_expression: &str,
    packed_half: Option<(usize, usize)>,
    builder: &CudaScalarExpressionBuilder,
) -> Result<String, String> {
    if let Some(reference) = builder.references.borrow().get(&(node_id, packed_half)) {
        return Ok(reference.clone());
    }
    let body = &scan_plan.body.plan;
    let node = body
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA Scan body node {node_id} is missing"))?;
    let in_half = |child_id, packed_half| {
        cuda_scan_body_expression_in_half_inner(
            scan_plan,
            child_id,
            capture_parameters,
            reference_shape,
            carry_expression,
            packed_half,
            builder,
        )
    };
    let child = |child_id| in_half(child_id, packed_half);
    let expression = match &node.op {
        TensorOp::Input { name } if name == &scan_plan.carry_name => Ok(match packed_half {
            Some((half, _)) => format!("{carry_expression}_{half}"),
            None => carry_expression.to_string(),
        }),
        TensorOp::Input { name } if name == &scan_plan.index_name => Ok("loop_index".to_string()),
        TensorOp::Input { name } => {
            let parameter = capture_parameters.get(name).ok_or_else(|| {
                format!("CUDA Scan body input {name:?} has no parent capture binding")
            })?;
            let offset = match packed_half {
                Some((half, stride)) => cuda_offset_expression_with_index(
                    reference_shape,
                    &node.shape,
                    &cuda_scan_packed_half_lane(half, stride),
                ),
                None => cuda_offset_expression(reference_shape, &node.shape),
            };
            Ok(format!("capture_{parameter}[{offset}]"))
        }
        TensorOp::ScalarConstant { value } => Ok(cuda_scalar_literal(*value)),
        // Cast source and target both execute as float, so the cast is the identity; AD has
        // already run, so stop_gradient and a custom rule node are the identity too.
        TensorOp::Broadcast { input }
        | TensorOp::Reshape { input }
        | TensorOp::Cast { input }
        | TensorOp::StopGradient { input }
        | TensorOp::Custom { value: input, .. } => child(*input),
        // The lowerability check restricts slices to one packed half of a
        // carry-shaped value, so the slice reads that half at the same lane.
        TensorOp::Slice { input, start, .. } => {
            let (_, stride) = packed_half.ok_or_else(|| {
                "CUDA Scan packed-pair slice requires half-aware lowering".to_string()
            })?;
            in_half(*input, Some((*start, stride)))
        }
        TensorOp::Concat { inputs, axis } => {
            let (half, stride) = packed_half.ok_or_else(|| {
                "CUDA Scan packed-pair concat requires half-aware lowering".to_string()
            })?;
            let aligned = element_count(reference_shape)? == element_count(&node.shape)?
                || (cuda_shapes_broadcastable(reference_shape, &node.shape)
                    && reference_shape[reference_shape.len() - node.shape.len()] == 2);
            if *axis != 0
                || inputs.len() != 2
                || !aligned
                || cuda_scan_packed_stride(reference_shape, &node.shape)? != stride
            {
                return Err(format!(
                    "CUDA Scan concat {:?} is not aligned with the packed carry axis of reference {reference_shape:?}",
                    node.shape
                ));
            }
            in_half(inputs[half], packed_half)
        }
        op => cuda_loop_formula(op, child)?.ok_or_else(|| {
            format!(
                "CUDA Scan body {} is not elementwise-lowerable",
                cuda_op_name(op)
            )
        }),
    }?;
    Ok(builder.emit(node_id, packed_half, expression))
}

#[derive(Clone, Debug)]
enum CudaElementwiseInput {
    Scalar(String),
    Buffer(usize),
}

fn cuda_elementwise_plan_expression(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
    inputs: &BTreeMap<String, CudaElementwiseInput>,
    reference_shape: &[usize],
) -> Result<String, String> {
    cuda_elementwise_plan_expression_with_index(plan, node_id, inputs, reference_shape, "index")
}

fn cuda_elementwise_plan_expression_with_index(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
    inputs: &BTreeMap<String, CudaElementwiseInput>,
    reference_shape: &[usize],
    index_expression: &str,
) -> Result<String, String> {
    let builder = CudaScalarExpressionBuilder::default();
    let reference = cuda_elementwise_plan_expression_with_index_inner(
        plan,
        node_id,
        inputs,
        reference_shape,
        index_expression,
        &builder,
    )?;
    Ok(builder.finish(&reference))
}

fn cuda_elementwise_plan_expression_with_index_inner(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
    inputs: &BTreeMap<String, CudaElementwiseInput>,
    reference_shape: &[usize],
    index_expression: &str,
    builder: &CudaScalarExpressionBuilder,
) -> Result<String, String> {
    if let Some(reference) = builder.references.borrow().get(&(node_id, None)) {
        return Ok(reference.clone());
    }
    let node = plan
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA elementwise plan node {node_id} is missing"))?;
    let child = |child_id| {
        cuda_elementwise_plan_expression_with_index_inner(
            plan,
            child_id,
            inputs,
            reference_shape,
            index_expression,
            builder,
        )
    };
    let expression = match &node.op {
        TensorOp::Input { name } => match inputs.get(name) {
            Some(CudaElementwiseInput::Scalar(expression)) => Ok(expression.clone()),
            Some(CudaElementwiseInput::Buffer(parameter)) => Ok(format!(
                "capture_{parameter}[{}]",
                cuda_offset_expression_with_index(reference_shape, &node.shape, index_expression)
            )),
            None => Err(format!(
                "CUDA elementwise plan input {name:?} has no binding"
            )),
        },
        TensorOp::ScalarConstant { value } if value.is_finite() => Ok(cuda_scalar_literal(*value)),
        // Cast source and target both execute as float, so the cast is the identity; AD has
        // already run, so stop_gradient and a custom rule node are the identity too.
        TensorOp::Broadcast { input }
        | TensorOp::Reshape { input }
        | TensorOp::Cast { input }
        | TensorOp::StopGradient { input }
        | TensorOp::Custom { value: input, .. } => child(*input),
        op => cuda_loop_formula(op, child)?.ok_or_else(|| {
            format!(
                "CUDA Fori VJP body uses unsupported {} operation",
                cuda_op_name(op)
            )
        }),
    }?;
    Ok(builder.emit(node_id, None, expression))
}

fn cuda_fori_vjp_plan(
    loop_plan: &TensorForiExecutionPlan,
    target: &TensorForiVjpTarget,
) -> Result<TensorExecutionPlan, String> {
    cuda_fori_body_is_lowerable(loop_plan)?;
    let carry_shape = loop_plan.carry_shape()?;
    for (name, shape) in loop_plan.external_captures() {
        if !cuda_shapes_broadcastable(&carry_shape, shape) {
            return Err(format!(
                "capture {name:?} shape {shape:?} cannot broadcast to carry shape {carry_shape:?}"
            ));
        }
    }
    let cotangent_name = "__quabla_cuda_fori_vjp_cotangent";
    if loop_plan.body.captures().contains_key(cotangent_name) {
        return Err(
            "CUDA Fori VJP internal cotangent name collides with a body capture".to_string(),
        );
    }
    let body = loop_plan.body.plan.as_ir();
    let transformed = body.symbolic_vjp(loop_plan.body.plan.output_node_id, cotangent_name)?;
    let gradient_name = match target {
        TensorForiVjpTarget::Carry => &loop_plan.carry_name,
        TensorForiVjpTarget::External(name) => name,
    };
    let gradient = transformed
        .gradients
        .get(gradient_name)
        .ok_or_else(|| format!("CUDA Fori VJP body has no gradient for {gradient_name:?}"))?;
    let plan = transformed.graph.compile_cpu(*gradient)?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        loop_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        loop_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    inputs.insert(
        cotangent_name.to_string(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    for (index, name) in loop_plan.external_captures().keys().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    let target_shape = match target {
        TensorForiVjpTarget::Carry => carry_shape.clone(),
        TensorForiVjpTarget::External(name) => loop_plan
            .external_captures()
            .get(name)
            .cloned()
            .ok_or_else(|| format!("CUDA Fori VJP has no capture {name:?}"))?,
    };
    let expression_node = cuda_vjp_elementwise_output_node(&plan, &target_shape, &carry_shape)?;
    cuda_elementwise_plan_expression(&plan, expression_node, &inputs, &carry_shape)?;
    Ok(plan)
}

fn cuda_fori_jvp_tangent_names(loop_plan: &TensorForiExecutionPlan) -> BTreeMap<String, String> {
    let mut tangent_names = BTreeMap::new();
    for (index, name) in loop_plan.body.captures.keys().enumerate() {
        if name == &loop_plan.index_name {
            continue;
        }
        let mut tangent_name = format!("__quabla_cuda_fori_jvp_tangent_{index}");
        while loop_plan.body.captures.contains_key(&tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &tangent_name)
        {
            tangent_name.push('_');
        }
        tangent_names.insert(name.clone(), tangent_name);
    }
    tangent_names
}

fn cuda_fori_jvp_expressions(
    loop_plan: &TensorForiExecutionPlan,
    captures: &[(String, TensorNodeId)],
) -> Result<(String, String), String> {
    cuda_fori_body_is_lowerable(loop_plan)?;
    let carry_shape = loop_plan.carry_shape()?;
    let tangent_names = cuda_fori_jvp_tangent_names(loop_plan);
    let body = loop_plan.body.plan.as_ir();
    let forward =
        body.symbolic_jvp_with_tangent_inputs(loop_plan.body.plan.output_node_id, &tangent_names)?;
    let (forward_plan, outputs) = forward
        .graph
        .compile_cpu_many(&[forward.value, forward.tangent])?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        loop_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        loop_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    for (name, tangent_name) in &tangent_names {
        let input = if name == &loop_plan.carry_name {
            CudaElementwiseInput::Scalar("carry_tangent".to_string())
        } else {
            let capture_index = captures
                .iter()
                .position(|(candidate, _)| candidate == name)
                .ok_or_else(|| format!("CUDA Fori JVP has no capture {name:?}"))?;
            CudaElementwiseInput::Buffer(captures.len() + capture_index)
        };
        inputs.insert(tangent_name.clone(), input);
    }
    Ok((
        cuda_elementwise_plan_expression(&forward_plan, outputs[0], &inputs, &carry_shape)?,
        cuda_elementwise_plan_expression(&forward_plan, outputs[1], &inputs, &carry_shape)?,
    ))
}

fn cuda_fori_jvp_is_lowerable(loop_plan: &TensorForiExecutionPlan) -> Result<(), String> {
    let captures = loop_plan
        .external_captures()
        .keys()
        .cloned()
        .map(|name| (name, 0))
        .collect::<Vec<_>>();
    cuda_fori_jvp_expressions(loop_plan, &captures).map(|_| ())
}

fn cuda_fori_vjp_jvp_is_lowerable(plan: &TensorForiVjpJvpExecutionPlan) -> Result<(), String> {
    let loop_plan = &plan.loop_plan;
    cuda_fori_body_is_lowerable(loop_plan)?;
    let carry_shape = loop_plan.carry_shape()?;
    for (name, shape) in loop_plan.external_captures() {
        if shape != &carry_shape {
            return Err(format!(
                "CUDA Fori VJP JVP capture {name:?} shape {shape:?} must match carry shape {carry_shape:?}"
            ));
        }
    }
    let mut forward_tangent_names = plan.tangent_names.clone();
    forward_tangent_names.remove(&plan.cotangent_name);
    let body = loop_plan.body.plan.as_ir();
    let forward = body.symbolic_jvp_with_tangent_inputs(
        loop_plan.body.plan.output_node_id,
        &forward_tangent_names,
    )?;
    let (forward_plan, outputs) = forward
        .graph
        .compile_cpu_many(&[forward.value, forward.tangent])?;
    let mut forward_inputs = BTreeMap::new();
    forward_inputs.insert(
        loop_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    forward_inputs.insert(
        loop_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    for (index, name) in loop_plan.external_captures().keys().enumerate() {
        forward_inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    for (name, tangent_name) in &forward_tangent_names {
        let input = if name == &loop_plan.carry_name {
            CudaElementwiseInput::Scalar("carry_tangent".to_string())
        } else {
            let capture_index = loop_plan
                .external_captures()
                .keys()
                .position(|candidate| candidate == name)
                .ok_or_else(|| format!("CUDA Fori VJP JVP has no capture {name:?}"))?;
            CudaElementwiseInput::Buffer(loop_plan.external_captures().len() + capture_index)
        };
        forward_inputs.insert(tangent_name.clone(), input);
    }
    for output in outputs {
        cuda_elementwise_plan_expression(&forward_plan, output, &forward_inputs, &carry_shape)?;
    }
    let mut reverse_inputs = forward_inputs;
    reverse_inputs.insert(
        plan.cotangent_name.clone(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    reverse_inputs.insert(
        "__quabla_cuda_fori_vjp_cotangent".to_string(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    let cotangent_tangent = plan
        .tangent_names
        .get(&plan.cotangent_name)
        .ok_or_else(|| "CUDA Fori VJP JVP has no cotangent tangent name".to_string())?;
    reverse_inputs.insert(
        cotangent_tangent.clone(),
        CudaElementwiseInput::Scalar("cotangent_tangent".to_string()),
    );
    for gradient in plan.gradient_tangent_plans.values() {
        cuda_elementwise_plan_expression(
            gradient,
            gradient.output_node_id,
            &reverse_inputs,
            &carry_shape,
        )?;
    }
    Ok(())
}

fn cuda_fori_vjp_group(
    plan: &TensorExecutionPlan,
    group: usize,
) -> Result<Vec<(TensorNodeId, &TensorForiVjpTarget)>, String> {
    let mut members = Vec::new();
    let mut signature = None;
    for (node_id, node) in plan.nodes.iter().enumerate() {
        let TensorOp::Region(RegionNode {
            kind:
                RegionKind::ForiVjp {
                    carry,
                    output_cotangent,
                    loop_plan: _,
                    target,
                    group: node_group,
                },
            captures,
            ..
        }) = &node.op
        else {
            continue;
        };
        if *node_group != group {
            continue;
        }
        let current_signature = (*carry, *output_cotangent, captures);
        if let Some((expected_carry, expected_cotangent, expected_captures)) = signature.as_ref() {
            if *expected_carry != *carry
                || *expected_cotangent != *output_cotangent
                || *expected_captures != captures
            {
                return Err(format!(
                    "CUDA Fori VJP group {group} mixes incompatible loop bindings"
                ));
            }
        } else {
            signature = Some(current_signature);
        }
        members.push((node_id, target));
    }
    if members.is_empty() {
        return Err(format!("CUDA Fori VJP group {group} has no result nodes"));
    }
    Ok(members)
}

fn cuda_vjp_elementwise_output_node(
    plan: &TensorExecutionPlan,
    target_shape: &[usize],
    element_shape: &[usize],
) -> Result<TensorNodeId, String> {
    if target_shape == element_shape {
        return Ok(plan.output_node_id);
    }
    if !cuda_shapes_broadcastable(element_shape, target_shape) {
        return Err(format!(
            "CUDA VJP target shape {target_shape:?} cannot broadcast to element shape {element_shape:?}"
        ));
    }
    let mut node_id = plan.output_node_id;
    loop {
        let node = plan
            .nodes
            .get(node_id)
            .ok_or_else(|| format!("CUDA VJP output node {node_id} is missing"))?;
        match &node.op {
            TensorOp::Reshape { input }
            | TensorOp::SumAxis { input, .. }
            | TensorOp::Sum { input } => {
                node_id = *input;
            }
            _ => break,
        }
    }
    if element_count(&plan.nodes[node_id].shape)? == element_count(element_shape)? {
        return Ok(node_id);
    }
    if plan.nodes[node_id].shape != element_shape {
        return Err(format!(
            "CUDA broadcast VJP reduction did not expose an elementwise contribution of shape {element_shape:?}; stopped at {:?} with shape {:?}",
            plan.nodes[node_id].op,
            plan.nodes[node_id].shape,
        ));
    }
    Ok(node_id)
}

fn cuda_scan_vjp_plan(
    scan_plan: &TensorScanExecutionPlan,
    target: &TensorScanVjpTarget,
    body_output_index: usize,
    cotangent_name: &str,
) -> Result<TensorExecutionPlan, String> {
    cuda_scan_body_is_lowerable(scan_plan)?;
    let carry_shape = scan_plan.carry_shape()?;
    for (name, shape) in scan_plan.external_captures() {
        if !cuda_shapes_broadcastable(&carry_shape, shape) {
            return Err(format!(
                "capture {name:?} shape {shape:?} cannot broadcast to carry shape {carry_shape:?}"
            ));
        }
    }
    if scan_plan.body.captures().contains_key(cotangent_name) {
        return Err(
            "CUDA Scan VJP internal cotangent name collides with a body capture".to_string(),
        );
    }
    let output_node_id = *scan_plan
        .body
        .plan
        .output_node_ids
        .get(body_output_index)
        .ok_or_else(|| format!("CUDA Scan VJP body output {body_output_index} is missing"))?;
    let (vjp_output_node_id, element_shape) = if body_output_index == 1 {
        if let Some(input) = cuda_scan_direct_broadcast_output_input(scan_plan)? {
            (input, carry_shape.clone())
        } else {
            (
                output_node_id,
                scan_plan.body.plan.nodes[output_node_id].shape.clone(),
            )
        }
    } else {
        (
            output_node_id,
            scan_plan.body.plan.nodes[output_node_id].shape.clone(),
        )
    };
    let body = scan_plan.body.plan.as_ir();
    let transformed = body.symbolic_vjp(vjp_output_node_id, cotangent_name)?;
    let gradient_name = match target {
        TensorScanVjpTarget::Carry => &scan_plan.carry_name,
        TensorScanVjpTarget::External(name) => name,
    };
    let gradient = transformed
        .gradients
        .get(gradient_name)
        .ok_or_else(|| format!("CUDA Scan VJP body has no gradient for {gradient_name:?}"))?;
    let plan = transformed.graph.compile_cpu(*gradient)?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        scan_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        scan_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    inputs.insert(
        cotangent_name.to_string(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    for (index, name) in scan_plan.external_captures().keys().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    let target_shape = match target {
        TensorScanVjpTarget::Carry => carry_shape.clone(),
        TensorScanVjpTarget::External(name) => scan_plan
            .external_captures()
            .get(name)
            .cloned()
            .ok_or_else(|| format!("CUDA Scan VJP has no capture {name:?}"))?,
    };
    let expression_node = cuda_vjp_elementwise_output_node(&plan, &target_shape, &element_shape)?;
    cuda_elementwise_plan_expression(&plan, expression_node, &inputs, &element_shape)?;
    Ok(plan)
}

/// Returns the carry-shaped source node when an unequal-lane Scan output is a
/// direct broadcast. Its cotangent can be reduced per carry lane before VJP.
fn cuda_scan_direct_broadcast_output_input(
    scan_plan: &TensorScanExecutionPlan,
) -> Result<Option<TensorNodeId>, String> {
    let carry_shape = scan_plan.carry_shape()?;
    let output_node_id = scan_plan.body.plan.output_node_ids[1];
    let output_shape = &scan_plan.body.plan.nodes[output_node_id].shape;
    if element_count(output_shape)? == element_count(&carry_shape)? {
        return Ok(None);
    }
    match &scan_plan.body.plan.nodes[output_node_id].op {
        TensorOp::Broadcast { input } if scan_plan.body.plan.nodes[*input].shape == carry_shape => {
            Ok(Some(*input))
        }
        _ => Err(
            "CUDA Scan VJP with unequal output and carry lanes requires a direct broadcast from the carry shape"
                .to_string(),
        ),
    }
}

fn cuda_scan_vjp_plans(
    scan_plan: &TensorScanExecutionPlan,
    target: &TensorScanVjpTarget,
) -> Result<(TensorExecutionPlan, TensorExecutionPlan), String> {
    Ok((
        cuda_scan_vjp_plan(
            scan_plan,
            target,
            0,
            "__quabla_cuda_scan_vjp_carry_cotangent",
        )?,
        cuda_scan_vjp_plan(
            scan_plan,
            target,
            1,
            "__quabla_cuda_scan_vjp_output_cotangent",
        )?,
    ))
}

fn cuda_scan_vjp_jvp_is_lowerable(plan: &TensorScanVjpJvpExecutionPlan) -> Result<(), String> {
    let scan_plan = &plan.scan_plan;
    cuda_scan_body_is_lowerable(scan_plan)?;
    let carry_shape = scan_plan.carry_shape()?;
    let output_shape = &scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]].shape;
    let direct_broadcast_output = cuda_scan_direct_broadcast_output_input(scan_plan)?.is_some();
    if element_count(output_shape)? != element_count(&carry_shape)? && !direct_broadcast_output {
        return Err("CUDA Scan VJP JVP unequal output and carry lanes require a direct broadcast from the carry shape".to_string());
    }
    for (name, shape) in scan_plan.external_captures() {
        if !cuda_shapes_broadcastable(&carry_shape, shape) {
            return Err(format!(
                "CUDA Scan VJP JVP capture {name:?} shape {shape:?} cannot broadcast to carry shape {carry_shape:?}"
            ));
        }
    }
    let forward_tangent_names = plan
        .tangent_names
        .iter()
        .filter(|(name, _)| scan_plan.body.captures.contains_key(*name))
        .map(|(name, tangent)| (name.clone(), tangent.clone()))
        .collect::<BTreeMap<_, _>>();
    let body = scan_plan.body.plan.as_ir();
    let forward = body.symbolic_jvp_with_tangent_inputs(
        scan_plan.body.plan.output_node_ids[0],
        &forward_tangent_names,
    )?;
    let (forward_plan, outputs) = forward
        .graph
        .compile_cpu_many(&[forward.value, forward.tangent])?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        scan_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        scan_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    for (index, name) in scan_plan.external_captures().keys().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    for (name, tangent_name) in &forward_tangent_names {
        let input = if name == &scan_plan.carry_name {
            CudaElementwiseInput::Scalar("carry_tangent".to_string())
        } else {
            let capture_index = scan_plan
                .external_captures()
                .keys()
                .position(|candidate| candidate == name)
                .ok_or_else(|| format!("CUDA Scan VJP JVP has no capture {name:?}"))?;
            CudaElementwiseInput::Buffer(scan_plan.external_captures().len() + capture_index)
        };
        inputs.insert(tangent_name.clone(), input);
    }
    for output in outputs {
        cuda_elementwise_plan_expression(&forward_plan, output, &inputs, &carry_shape)?;
    }
    let mut reverse_inputs = inputs;
    reverse_inputs.insert(
        plan.carry_cotangent_name.clone(),
        CudaElementwiseInput::Scalar("carry_cotangent".to_string()),
    );
    reverse_inputs.insert(
        plan.output_cotangent_name.clone(),
        CudaElementwiseInput::Scalar("output_cotangent_step".to_string()),
    );
    reverse_inputs.insert(
        "__quabla_cuda_scan_vjp_carry_cotangent".to_string(),
        CudaElementwiseInput::Scalar("carry_cotangent".to_string()),
    );
    reverse_inputs.insert(
        "__quabla_cuda_scan_vjp_output_cotangent".to_string(),
        CudaElementwiseInput::Scalar("output_cotangent_step".to_string()),
    );
    for (name, tangent_name) in &plan.tangent_names {
        if name == &plan.carry_cotangent_name {
            reverse_inputs.insert(
                tangent_name.clone(),
                CudaElementwiseInput::Scalar("carry_cotangent_tangent".to_string()),
            );
        } else if name == &plan.output_cotangent_name {
            reverse_inputs.insert(
                tangent_name.clone(),
                CudaElementwiseInput::Scalar("output_cotangent_tangent_step".to_string()),
            );
        }
    }
    for (name, gradient) in &plan.carry_gradient_tangent_plans {
        let target_shape = if name == &scan_plan.carry_name {
            carry_shape.clone()
        } else {
            scan_plan
                .external_captures()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("CUDA Scan VJP JVP has no capture {name:?}"))?
        };
        cuda_elementwise_plan_expression(
            gradient,
            cuda_vjp_elementwise_output_node(gradient, &target_shape, &carry_shape)?,
            &reverse_inputs,
            &carry_shape,
        )?;
    }
    if direct_broadcast_output {
        for target in std::iter::once(TensorScanVjpTarget::Carry).chain(
            scan_plan
                .external_captures()
                .keys()
                .cloned()
                .map(TensorScanVjpTarget::External),
        ) {
            let directional =
                cuda_scan_vjp_jvp_direct_broadcast_output_directional_plan(plan, &target)?;
            let target_shape = match &target {
                TensorScanVjpTarget::Carry => carry_shape.clone(),
                TensorScanVjpTarget::External(name) => scan_plan
                    .external_captures()
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("CUDA Scan VJP JVP has no capture {name:?}"))?,
            };
            cuda_elementwise_plan_expression(
                &directional,
                cuda_vjp_elementwise_output_node(&directional, &target_shape, &carry_shape)?,
                &reverse_inputs,
                &carry_shape,
            )?;
        }
    } else {
        for (name, gradient) in &plan.output_gradient_tangent_plans {
            let target_shape = if name == &scan_plan.carry_name {
                carry_shape.clone()
            } else {
                scan_plan
                    .external_captures()
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("CUDA Scan VJP JVP has no capture {name:?}"))?
            };
            cuda_elementwise_plan_expression(
                gradient,
                cuda_vjp_elementwise_output_node(gradient, &target_shape, &carry_shape)?,
                &reverse_inputs,
                &carry_shape,
            )?;
        }
    }
    Ok(())
}

fn cuda_scan_vjp_jvp_direct_broadcast_output_directional_plan(
    plan: &TensorScanVjpJvpExecutionPlan,
    target: &TensorScanVjpTarget,
) -> Result<TensorExecutionPlan, String> {
    let scan_plan = &plan.scan_plan;
    let output_node = cuda_scan_direct_broadcast_output_input(scan_plan)?.ok_or_else(|| {
        "CUDA Scan VJP JVP direct-broadcast directional plan requires unequal output lanes"
            .to_string()
    })?;
    let gradient_name = match target {
        TensorScanVjpTarget::Carry => &scan_plan.carry_name,
        TensorScanVjpTarget::External(name) => name,
    };
    let output_vjp = scan_plan
        .body
        .plan
        .as_ir()
        .symbolic_vjp(output_node, &plan.output_cotangent_name)?;
    let gradient = *output_vjp.gradients.get(gradient_name).ok_or_else(|| {
        format!("CUDA Scan VJP JVP body has no output gradient for {gradient_name:?}")
    })?;
    let tangent_names = plan
        .tangent_names
        .iter()
        .filter(|(name, _)| {
            scan_plan.body.captures.contains_key(*name) || *name == &plan.output_cotangent_name
        })
        .map(|(name, tangent)| (name.clone(), tangent.clone()))
        .collect::<BTreeMap<_, _>>();
    let directional = output_vjp
        .graph
        .symbolic_jvp_with_tangent_inputs(gradient, &tangent_names)?;
    directional.graph.compile_cpu(directional.tangent)
}

fn cuda_scan_vjp_group(
    plan: &TensorExecutionPlan,
    group: usize,
) -> Result<Vec<(TensorNodeId, &TensorScanVjpTarget)>, String> {
    let mut members = Vec::new();
    let mut signature = None;
    for (node_id, node) in plan.nodes.iter().enumerate() {
        let TensorOp::Region(RegionNode {
            kind:
                RegionKind::ScanVjp {
                    carry,
                    final_carry_cotangent,
                    output_cotangent,
                    scan_plan: _,
                    target,
                    group: node_group,
                },
            captures,
            ..
        }) = &node.op
        else {
            continue;
        };
        if *node_group != group {
            continue;
        }
        let current_signature = (*carry, *final_carry_cotangent, *output_cotangent, captures);
        if let Some((expected_carry, expected_final, expected_output, expected_captures)) =
            signature.as_ref()
        {
            if *expected_carry != *carry
                || *expected_final != *final_carry_cotangent
                || *expected_output != *output_cotangent
                || *expected_captures != captures
            {
                return Err(format!(
                    "CUDA Scan VJP group {group} mixes incompatible scan bindings"
                ));
            }
        } else {
            signature = Some(current_signature);
        }
        members.push((node_id, target));
    }
    if members.is_empty() {
        return Err(format!("CUDA Scan VJP group {group} has no result nodes"));
    }
    Ok(members)
}

fn cuda_scan_vjp_jvp_group(
    plan: &TensorExecutionPlan,
    group: usize,
) -> Result<Vec<(TensorNodeId, &TensorScanVjpTarget)>, String> {
    let mut members = Vec::new();
    let mut signature = None;
    for (node_id, node) in plan.nodes.iter().enumerate() {
        let TensorOp::Region(RegionNode {
            kind:
                RegionKind::ScanVjpJvp {
                    carry,
                    carry_tangent,
                    final_carry_cotangent,
                    final_carry_cotangent_tangent,
                    output_cotangent,
                    output_cotangent_tangent,
                    target,
                    group: node_group,
                    ..
                },
            captures,
            tangent_captures,
            ..
        }) = &node.op
        else {
            continue;
        };
        if *node_group != group {
            continue;
        }
        let current_signature = (
            *carry,
            *carry_tangent,
            *final_carry_cotangent,
            *final_carry_cotangent_tangent,
            *output_cotangent,
            *output_cotangent_tangent,
            captures,
            tangent_captures,
        );
        if let Some(expected) = signature.as_ref() {
            if *expected != current_signature {
                return Err(format!(
                    "CUDA Scan VJP JVP group {group} mixes incompatible bindings"
                ));
            }
        } else {
            signature = Some(current_signature);
        }
        members.push((node_id, target));
    }
    if members.is_empty() {
        return Err(format!(
            "CUDA Scan VJP JVP group {group} has no result nodes"
        ));
    }
    Ok(members)
}

/// Replay independent lanes once per reverse block. Broadcast capture reductions
/// retain their original tape and launch timing because atomic accumulation order
/// is not part of the approved numerical changes.
fn cuda_loop_checkpoint_block(
    lower: usize,
    upper: usize,
    carry_shape: &[usize],
    captures: &BTreeMap<String, Vec<usize>>,
    output_shape: Option<&[usize]>,
    independent_lanes: bool,
) -> Option<usize> {
    if !independent_lanes
        || captures.values().any(|shape| shape != carry_shape)
        || output_shape.is_some_and(|shape| shape != carry_shape)
    {
        return None;
    }
    super::TensorCarryCheckpoints::<DynamicTensor>::block_size(upper.checked_sub(lower)?)
}

fn cuda_loop_tape_states(lower: usize, upper: usize, block: Option<usize>) -> Option<usize> {
    let steps = upper.checked_sub(lower)?;
    match block {
        Some(block) => steps.div_ceil(block).checked_add(block),
        None => steps.checked_add(1),
    }
}

fn cuda_loop_tape_source(
    lower: usize,
    upper: usize,
    count: &str,
    next: &str,
    next_tangent: Option<&str>,
    block: Option<usize>,
) -> (String, String) {
    let update = match next_tangent {
        Some(tangent) => format!(
            "float checkpoint_next = {next}; float checkpoint_next_tangent = {tangent}; carry = checkpoint_next; carry_tangent = checkpoint_next_tangent;"
        ),
        None => format!("carry = {next};"),
    };
    let store = |position: &str| {
        let carry = format!("carry_tape[({position}) * {count} + index] = carry;");
        match next_tangent {
            Some(_) => format!(
                "{carry} carry_tangent_tape[({position}) * {count} + index] = carry_tangent;"
            ),
            None => carry,
        }
    };
    let load = |position: &str| {
        let carry = format!("carry = carry_tape[({position}) * {count} + index];");
        match next_tangent {
            Some(_) => format!(
                "{carry} carry_tangent = carry_tangent_tape[({position}) * {count} + index];"
            ),
            None => carry,
        }
    };
    let initial = store("0ULL");
    match block {
        None => (
            format!("{initial}\nfor (unsigned long long step = {lower}ULL; step < {upper}ULL; ++step) {{ float loop_index = (float)step; {update} {} }}\n", store(&format!("step - {lower}ULL + 1ULL"))),
            load(&format!("step - {lower}ULL")),
        ),
        Some(block) => {
            let checkpoints = (upper - lower).div_ceil(block);
            let forward = format!(
                "{initial}\nfor (unsigned long long step = {lower}ULL; step < {upper}ULL; ++step) {{ float loop_index = (float)step; {update} unsigned long long offset = step - {lower}ULL + 1ULL; if (offset % {block}ULL == 0ULL && step + 1ULL < {upper}ULL) {{ {} }} }}\n",
                store(&format!("offset / {block}ULL"))
            );
            let reverse = format!(
                "unsigned long long block_start = (step - {lower}ULL) / {block}ULL * {block}ULL + {lower}ULL; unsigned long long block_end = {upper}ULL; if ({upper}ULL - block_start > {block}ULL) block_end = block_start + {block}ULL; if (step + 1ULL == block_end) {{ {} {} for (unsigned long long replay = block_start; replay + 1ULL < block_end; ++replay) {{ float loop_index = (float)replay; {update} {} }} }} {}",
                load(&format!("(block_start - {lower}ULL) / {block}ULL")),
                store(&format!("{checkpoints}ULL")),
                store(&format!("{checkpoints}ULL + replay - block_start + 1ULL")),
                load(&format!("{checkpoints}ULL + step - block_start")),
            );
            (forward, reverse)
        }
    }
}

fn cuda_fori_vjp_node_kernel_source(
    node_id: TensorNodeId,
    loop_plan: &TensorForiExecutionPlan,
    captures: &[(String, TensorNodeId)],
    target: &TensorForiVjpTarget,
) -> Result<String, String> {
    let gradient_plan = cuda_fori_vjp_plan(loop_plan, target)?;
    let carry_gradient_plan = match target {
        TensorForiVjpTarget::Carry => None,
        TensorForiVjpTarget::External(_) => {
            Some(cuda_fori_vjp_plan(loop_plan, &TensorForiVjpTarget::Carry)?)
        }
    };
    let carry_shape = loop_plan.carry_shape()?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        loop_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        loop_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    inputs.insert(
        "__quabla_cuda_fori_vjp_cotangent".to_string(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    let forward_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let forward_expression = cuda_fori_body_expression(
        loop_plan,
        loop_plan.body.plan.output_node_id,
        &forward_parameters,
        &carry_shape,
    )?;
    let gradient_expression = cuda_elementwise_plan_expression(
        &gradient_plan,
        cuda_vjp_elementwise_output_node(
            &gradient_plan,
            &match target {
                TensorForiVjpTarget::Carry => carry_shape.clone(),
                TensorForiVjpTarget::External(name) => loop_plan
                    .external_captures()
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("CUDA Fori VJP has no capture {name:?}"))?,
            },
            &carry_shape,
        )?,
        &inputs,
        &carry_shape,
    )?;
    let carry_gradient_expression = carry_gradient_plan
        .as_ref()
        .map(|plan| {
            cuda_elementwise_plan_expression(plan, plan.output_node_id, &inputs, &carry_shape)
        })
        .transpose()?;
    let reverse_update = match (target, carry_gradient_expression) {
        (TensorForiVjpTarget::Carry, None) => "cotangent = contribution;".to_string(),
        (TensorForiVjpTarget::External(_), Some(carry_gradient)) => format!(
            "gradient += contribution;\n\\
                cotangent = {carry_gradient};"
        ),
        _ => return Err("CUDA Fori VJP failed to construct its carry reverse update".to_string()),
    };
    let target_write = match target {
        TensorForiVjpTarget::Carry => "out[index] = cotangent;".to_string(),
        TensorForiVjpTarget::External(name) => {
            let target_shape = loop_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("CUDA Fori VJP has no capture {name:?}"))?;
            if target_shape == &carry_shape {
                "out[index] = gradient;".to_string()
            } else {
                format!(
                    "atomicAdd(out + {}, gradient);",
                    cuda_offset_expression(&carry_shape, target_shape)
                )
            }
        }
    };
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "const float* output_cotangent".to_string(),
            "float* carry_tape".to_string(),
            "float* out".to_string(),
            "unsigned long long count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let lower = loop_plan.lower;
    let upper = loop_plan.upper;
    let function = cuda_node_function_name(node_id);
    let block = cuda_loop_checkpoint_block(
        lower,
        upper,
        &carry_shape,
        loop_plan.external_captures(),
        None,
        true,
    );
    let (tape_forward, tape_reverse) =
        cuda_loop_tape_source(lower, upper, "count", &forward_expression, None, block);
    Ok(format!(
        "extern \"C\" __global__ void {function}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= count) return;\n\\
            float carry = initial_carry[index];\n\\
            {tape_forward}\n\
            float cotangent = output_cotangent[index];\n\\
            float gradient = 0.0f;\n\\
            for (unsigned long long reverse = {upper}ULL; reverse > {lower}ULL; --reverse) {{\n\\
                unsigned long long step = reverse - 1ULL;\n\\
                float loop_index = (float)step;\n\\
                {tape_reverse}\n\\
                float contribution = {gradient_expression};\n\\
                {reverse_update}\n\\
            }}\n\\
            {target_write}\n}}\n"))
}
fn cuda_fori_vjp_group_kernel_source(
    node_id: TensorNodeId,
    loop_plan: &TensorForiExecutionPlan,
    captures: &[(String, TensorNodeId)],
    targets: &[&TensorForiVjpTarget],
) -> Result<String, String> {
    let carry_shape = loop_plan.carry_shape()?;
    let carry_gradient_plan = cuda_fori_vjp_plan(loop_plan, &TensorForiVjpTarget::Carry)?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        loop_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        loop_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    inputs.insert(
        "__quabla_cuda_fori_vjp_cotangent".to_string(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    let forward_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let forward_expression = cuda_fori_body_expression(
        loop_plan,
        loop_plan.body.plan.output_node_id,
        &forward_parameters,
        &carry_shape,
    )?;
    let carry_gradient_expression = cuda_elementwise_plan_expression(
        &carry_gradient_plan,
        carry_gradient_plan.output_node_id,
        &inputs,
        &carry_shape,
    )?;

    let mut gradient_declarations = String::new();
    let mut contribution_updates = String::new();
    let mut target_writes = String::new();
    for (target_index, target) in targets.iter().enumerate() {
        match target {
            TensorForiVjpTarget::Carry => {
                target_writes.push_str(&format!("out_{target_index}[index] = cotangent;\n"));
            }
            TensorForiVjpTarget::External(name) => {
                let target_shape = loop_plan
                    .external_captures()
                    .get(name)
                    .ok_or_else(|| format!("CUDA Fori VJP has no capture {name:?}"))?;
                let gradient_plan = cuda_fori_vjp_plan(loop_plan, target)?;
                let expression_node =
                    cuda_vjp_elementwise_output_node(&gradient_plan, target_shape, &carry_shape)?;
                let gradient_expression = cuda_elementwise_plan_expression(
                    &gradient_plan,
                    expression_node,
                    &inputs,
                    &carry_shape,
                )?;
                gradient_declarations.push_str(&format!("float gradient_{target_index} = 0.0f;\n"));
                contribution_updates.push_str(&format!(
                    "gradient_{target_index} += {gradient_expression};\n"
                ));
                if target_shape == &carry_shape {
                    target_writes.push_str(&format!(
                        "out_{target_index}[index] = gradient_{target_index};\n"
                    ));
                } else {
                    target_writes.push_str(&format!(
                        "atomicAdd(out_{target_index} + {}, gradient_{target_index});\n",
                        cuda_offset_expression(&carry_shape, target_shape)
                    ));
                }
            }
        }
    }
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "const float* output_cotangent".to_string(),
            "float* carry_tape".to_string(),
        ])
        .chain(
            targets
                .iter()
                .enumerate()
                .map(|(index, _)| format!("float* out_{index}")),
        )
        .chain(["unsigned long long count".to_string()])
        .collect::<Vec<_>>()
        .join(", ");
    let lower = loop_plan.lower;
    let upper = loop_plan.upper;
    let function = cuda_node_function_name(node_id);
    let block = cuda_loop_checkpoint_block(
        lower,
        upper,
        &carry_shape,
        loop_plan.external_captures(),
        None,
        true,
    );
    let (tape_forward, tape_reverse) =
        cuda_loop_tape_source(lower, upper, "count", &forward_expression, None, block);
    Ok(format!(
        "extern \"C\" __global__ void {function}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= count) return;\n\\
            float carry = initial_carry[index];\n\\
            {tape_forward}\n\
            float cotangent = output_cotangent[index];\n\\
            {gradient_declarations}\
            for (unsigned long long reverse = {upper}ULL; reverse > {lower}ULL; --reverse) {{\n\\
                unsigned long long step = reverse - 1ULL;\n\\
                float loop_index = (float)step;\n\\
                {tape_reverse}\n\\
                {contribution_updates}\
                cotangent = {carry_gradient_expression};\n\\
            }}\n\\
            {target_writes}\
        }}\n"))
}
fn cuda_fori_vjp_jvp_node_kernel_source(
    node_id: TensorNodeId,
    plan: &TensorForiVjpJvpExecutionPlan,
    captures: &[(String, TensorNodeId)],
    target: &TensorForiVjpTarget,
) -> Result<String, String> {
    cuda_fori_vjp_jvp_is_lowerable(plan)?;
    let loop_plan = &plan.loop_plan;
    let carry_shape = loop_plan.carry_shape()?;
    let body = loop_plan.body.plan.as_ir();
    let mut forward_tangent_names = plan.tangent_names.clone();
    forward_tangent_names.remove(&plan.cotangent_name);
    let forward = body.symbolic_jvp_with_tangent_inputs(
        loop_plan.body.plan.output_node_id,
        &forward_tangent_names,
    )?;
    let (forward_plan, forward_outputs) = forward
        .graph
        .compile_cpu_many(&[forward.value, forward.tangent])?;
    let mut forward_inputs = BTreeMap::new();
    forward_inputs.insert(
        loop_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    forward_inputs.insert(
        loop_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        forward_inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    for (name, tangent_name) in &forward_tangent_names {
        let input = if name == &loop_plan.carry_name {
            CudaElementwiseInput::Scalar("carry_tangent".to_string())
        } else {
            let capture_index = captures
                .iter()
                .position(|(candidate, _)| candidate == name)
                .ok_or_else(|| format!("CUDA Fori VJP JVP has no capture {name:?}"))?;
            CudaElementwiseInput::Buffer(captures.len() + capture_index)
        };
        forward_inputs.insert(tangent_name.clone(), input);
    }
    let forward_expression = cuda_elementwise_plan_expression(
        &forward_plan,
        forward_outputs[0],
        &forward_inputs,
        &carry_shape,
    )?;
    let forward_tangent_expression = cuda_elementwise_plan_expression(
        &forward_plan,
        forward_outputs[1],
        &forward_inputs,
        &carry_shape,
    )?;

    let mut reverse_inputs = forward_inputs;
    reverse_inputs.insert(
        plan.cotangent_name.clone(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    reverse_inputs.insert(
        "__quabla_cuda_fori_vjp_cotangent".to_string(),
        CudaElementwiseInput::Scalar("cotangent".to_string()),
    );
    let cotangent_tangent_name = plan
        .tangent_names
        .get(&plan.cotangent_name)
        .ok_or_else(|| "CUDA Fori VJP JVP has no cotangent tangent input".to_string())?;
    reverse_inputs.insert(
        cotangent_tangent_name.clone(),
        CudaElementwiseInput::Scalar("cotangent_tangent".to_string()),
    );
    let carry_vjp = cuda_fori_vjp_plan(loop_plan, &TensorForiVjpTarget::Carry)?;
    let carry_cotangent_expression = cuda_elementwise_plan_expression(
        &carry_vjp,
        carry_vjp.output_node_id,
        &reverse_inputs,
        &carry_shape,
    )?;
    let target_name = match target {
        TensorForiVjpTarget::Carry => &loop_plan.carry_name,
        TensorForiVjpTarget::External(name) => name,
    };
    let target_plan = plan
        .gradient_tangent_plans
        .get(target_name)
        .ok_or_else(|| {
            format!("CUDA Fori VJP JVP has no directional gradient plan for {target_name:?}")
        })?;
    let target_expression = cuda_elementwise_plan_expression(
        target_plan,
        target_plan.output_node_id,
        &reverse_inputs,
        &carry_shape,
    )?;
    let carry_tangent_plan = plan
        .gradient_tangent_plans
        .get(&loop_plan.carry_name)
        .ok_or_else(|| "CUDA Fori VJP JVP has no carry directional gradient plan".to_string())?;
    let carry_tangent_expression = cuda_elementwise_plan_expression(
        carry_tangent_plan,
        carry_tangent_plan.output_node_id,
        &reverse_inputs,
        &carry_shape,
    )?;
    let target_write = match target {
        TensorForiVjpTarget::Carry => "out[index] = cotangent_tangent;".to_string(),
        TensorForiVjpTarget::External(_) => "out[index] = gradient;".to_string(),
    };
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain(
            captures
                .iter()
                .enumerate()
                .map(|(index, _)| format!("const float* capture_{}", captures.len() + index)),
        )
        .chain([
            "const float* initial_carry".to_string(),
            "const float* initial_carry_tangent".to_string(),
            "const float* output_cotangent".to_string(),
            "const float* output_cotangent_tangent".to_string(),
            "float* carry_tape".to_string(),
            "float* carry_tangent_tape".to_string(),
            "float* out".to_string(),
            "unsigned long long count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let lower = loop_plan.lower;
    let upper = loop_plan.upper;
    let function = cuda_node_function_name(node_id);
    let block = cuda_loop_checkpoint_block(
        lower,
        upper,
        &carry_shape,
        loop_plan.external_captures(),
        None,
        true,
    );
    let (tape_forward, tape_reverse) = cuda_loop_tape_source(
        lower,
        upper,
        "count",
        &forward_expression,
        Some(&forward_tangent_expression),
        block,
    );
    Ok(format!(
        "extern \"C\" __global__ void {function}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= count) return;\n\\
            float carry = initial_carry[index];\n\\
            float carry_tangent = initial_carry_tangent[index];\n\\
            {tape_forward}\n\
            float cotangent = output_cotangent[index];\n\\
            float cotangent_tangent = output_cotangent_tangent[index];\n\\
            float gradient = 0.0f;\n\\
            for (unsigned long long reverse = {upper}ULL; reverse > {lower}ULL; --reverse) {{\n\\
                unsigned long long step = reverse - 1ULL;\n\\
                float loop_index = (float)step;\n\\
                {tape_reverse}\n\\
                gradient += {target_expression};\n\\
                float next_cotangent = {carry_cotangent_expression};\n\\
                float next_cotangent_tangent = {carry_tangent_expression};\n\\
                cotangent = next_cotangent;\n\\
                cotangent_tangent = next_cotangent_tangent;\n\\
            }}\n\\
            {target_write}\n\\
        }}\n"))
}
fn cuda_fori_jvp_node_kernel_source(
    node_id: TensorNodeId,
    loop_plan: &TensorForiExecutionPlan,
    captures: &[(String, TensorNodeId)],
) -> Result<String, String> {
    let (primal_expression, tangent_expression) = cuda_fori_jvp_expressions(loop_plan, captures)?;
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain(
            captures
                .iter()
                .enumerate()
                .map(|(index, _)| format!("const float* capture_{}", captures.len() + index)),
        )
        .chain([
            "const float* initial_carry".to_string(),
            "const float* initial_carry_tangent".to_string(),
            "float* out".to_string(),
            "unsigned long long count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "extern \"C\" __global__ void {}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= count) return;\n\\
            float carry = initial_carry[index];\n\\
            float carry_tangent = initial_carry_tangent[index];\n\\
            for (unsigned long long step = {}ULL; step < {}ULL; ++step) {{\n\\
                float loop_index = (float)step;\n\\
                float next_carry = {primal_expression};\n\\
                float next_carry_tangent = {tangent_expression};\n\\
                carry = next_carry;\n\\
                carry_tangent = next_carry_tangent;\n\\
            }}\n\\
            out[index] = carry_tangent;\n}}\n",
        cuda_node_function_name(node_id), loop_plan.lower, loop_plan.upper
    ))
}

fn cuda_fori_node_kernel_source(
    node_id: TensorNodeId,
    loop_plan: &TensorForiExecutionPlan,
    captures: &[(String, TensorNodeId)],
) -> Result<String, String> {
    cuda_fori_body_is_lowerable(loop_plan)?;
    let carry_shape = loop_plan.carry_shape()?;
    let capture_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let expression = cuda_fori_body_expression(
        loop_plan,
        loop_plan.body.plan.output_node_id,
        &capture_parameters,
        &carry_shape,
    )?;
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "float* out".to_string(),
            "unsigned long long count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "extern \"C\" __global__ void {}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= count) return;\n\\
            float carry = initial_carry[index];\n\\
            for (unsigned long long step = {}ULL; step < {}ULL; ++step) {{\n\\
                float loop_index = (float)step;\n\\
                carry = {expression};\n\\
            }}\n\\
            out[index] = carry;\n}}\n",
        cuda_node_function_name(node_id), loop_plan.lower, loop_plan.upper
    ))
}

fn cuda_scan_node_kernel_source(
    node_id: TensorNodeId,
    scan_plan: &TensorScanExecutionPlan,
    captures: &[(String, TensorNodeId)],
) -> Result<String, String> {
    cuda_scan_body_is_lowerable(scan_plan)?;
    if cuda_scan_uses_packed_halves(scan_plan) {
        return cuda_scan_packed_node_kernel_source(node_id, scan_plan, captures);
    }
    let carry_shape = scan_plan.carry_shape()?;
    let output_step_shape = scan_plan.body.output_shapes()[1].clone();
    let capture_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let next_expression = cuda_scan_body_expression(
        scan_plan,
        scan_plan.body.plan.output_node_ids[0],
        &capture_parameters,
        &carry_shape,
    )?;
    let output_next_expression = cuda_scan_body_expression_with_reference(
        scan_plan,
        scan_plan.body.plan.output_node_ids[0],
        &capture_parameters,
        &output_step_shape,
        "output_carry",
    )?;
    let output_expression = cuda_scan_body_expression_with_reference(
        scan_plan,
        scan_plan.body.plan.output_node_ids[1],
        &capture_parameters,
        &output_step_shape,
        "output_carry",
    )?;
    let output_initial_offset = cuda_offset_expression(&output_step_shape, &carry_shape);
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "float* final_carry".to_string(),
            "float* outputs".to_string(),
            "unsigned long long carry_count".to_string(),
            "unsigned long long output_count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "extern \"C\" __global__ void {}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= carry_count && index >= output_count) return;\n\\
            float carry = index < carry_count ? initial_carry[index] : 0.0f;\n\\
            float output_carry = index < output_count ? initial_carry[{output_initial_offset}] : 0.0f;\n\\
            for (unsigned long long step = {}ULL; step < {}ULL; ++step) {{\n\\
                float loop_index = (float)step;\n\\
                if (index < carry_count) {{ float next_carry = {next_expression}; carry = next_carry; }}\n\\
                if (index < output_count) {{ float scan_output = {output_expression}; outputs[(step - {}ULL) * output_count + index] = scan_output; output_carry = {output_next_expression}; }}\n\\
            }}\n\\
            if (index < carry_count) final_carry[index] = carry;\n}}\n",
        cuda_node_function_name(node_id),
        scan_plan.lower,
        scan_plan.upper,
        scan_plan.lower,
    ))
}

/// Packed-pair Scan kernel: every thread keeps both halves of its carry lane
/// (and of its mapped output lane) in registers and advances them together,
/// so a half may read its partner, e.g. a nonlinear JVP tangent reading the
/// primal. Each thread still writes only its own final-carry and output lane.
fn cuda_scan_packed_node_kernel_source(
    node_id: TensorNodeId,
    scan_plan: &TensorScanExecutionPlan,
    captures: &[(String, TensorNodeId)],
) -> Result<String, String> {
    let carry_shape = scan_plan.carry_shape()?;
    let output_step_shape = scan_plan.body.output_shapes()[1].clone();
    let carry_stride = cuda_scan_packed_stride(&carry_shape, &carry_shape)?;
    let output_stride = cuda_scan_packed_stride(&output_step_shape, &carry_shape)?;
    let capture_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let [carry_node, output_node] = [0, 1].map(|index| scan_plan.body.plan.output_node_ids[index]);
    let expression = |node, reference: &[usize], carry: &str, half, stride| {
        cuda_scan_body_expression_in_half(
            scan_plan,
            node,
            &capture_parameters,
            reference,
            carry,
            Some((half, stride)),
        )
    };
    let mut initial = String::new();
    let mut carry_step = String::new();
    let mut output_step = String::new();
    for half in 0..2 {
        let carry_lane = cuda_scan_packed_half_lane(half, carry_stride);
        let output_lane = cuda_scan_packed_half_lane(half, output_stride);
        let output_initial_offset =
            cuda_offset_expression_with_index(&output_step_shape, &carry_shape, &output_lane);
        initial.push_str(&format!(
            "float carry_{half} = index < carry_count ? initial_carry[{carry_lane}] : 0.0f;\n\\
            float output_carry_{half} = index < output_count ? initial_carry[{output_initial_offset}] : 0.0f;\n"
        ));
        carry_step.push_str(&format!(
            "float next_carry_{half} = {};\n",
            expression(carry_node, &carry_shape, "carry", half, carry_stride)?
        ));
        output_step.push_str(&format!(
            "float next_output_carry_{half} = {};\n",
            expression(
                carry_node,
                &output_step_shape,
                "output_carry",
                half,
                output_stride
            )?
        ));
    }
    let output_expression = format!(
        "(((index / {output_stride}ULL) % 2ULL) == 0ULL ? {} : {})",
        expression(
            output_node,
            &output_step_shape,
            "output_carry",
            0,
            output_stride
        )?,
        expression(
            output_node,
            &output_step_shape,
            "output_carry",
            1,
            output_stride
        )?,
    );
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "float* final_carry".to_string(),
            "float* outputs".to_string(),
            "unsigned long long carry_count".to_string(),
            "unsigned long long output_count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "extern \"C\" __global__ void {}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= carry_count && index >= output_count) return;\n\\
            {initial}\
            for (unsigned long long step = {}ULL; step < {}ULL; ++step) {{\n\\
                float loop_index = (float)step;\n\\
                if (index < carry_count) {{\n{carry_step}carry_0 = next_carry_0; carry_1 = next_carry_1; }}\n\\
                if (index < output_count) {{ float scan_output = {output_expression}; outputs[(step - {}ULL) * output_count + index] = scan_output;\n{output_step}output_carry_0 = next_output_carry_0; output_carry_1 = next_output_carry_1; }}\n\\
            }}\n\\
            if (index < carry_count) final_carry[index] = ((index / {carry_stride}ULL) % 2ULL) == 0ULL ? carry_0 : carry_1;\n}}\n",
        cuda_node_function_name(node_id),
        scan_plan.lower,
        scan_plan.upper,
        scan_plan.lower,
    ))
}

fn cuda_scan_vjp_node_kernel_source(
    node_id: TensorNodeId,
    scan_plan: &TensorScanExecutionPlan,
    captures: &[(String, TensorNodeId)],
    target: &TensorScanVjpTarget,
) -> Result<String, String> {
    let (carry_target_plan, output_target_plan) = cuda_scan_vjp_plans(scan_plan, target)?;
    let carry_reverse_plans = match target {
        TensorScanVjpTarget::Carry => None,
        TensorScanVjpTarget::External(_) => {
            Some(cuda_scan_vjp_plans(scan_plan, &TensorScanVjpTarget::Carry)?)
        }
    };
    let carry_shape = scan_plan.carry_shape()?;
    let output_step_shape = scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]]
        .shape
        .clone();
    let unequal_output_lanes = element_count(&output_step_shape)? != element_count(&carry_shape)?;
    let target_shape = match target {
        TensorScanVjpTarget::Carry => carry_shape.clone(),
        TensorScanVjpTarget::External(name) => scan_plan
            .external_captures()
            .get(name)
            .cloned()
            .ok_or_else(|| format!("CUDA Scan VJP has no capture {name:?}"))?,
    };
    let capture_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let forward_next_expression = cuda_scan_body_expression(
        scan_plan,
        scan_plan.body.plan.output_node_ids[0],
        &capture_parameters,
        &carry_shape,
    )?;
    let mut carry_inputs = BTreeMap::new();
    carry_inputs.insert(
        scan_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    carry_inputs.insert(
        scan_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    carry_inputs.insert(
        "__quabla_cuda_scan_vjp_carry_cotangent".to_string(),
        CudaElementwiseInput::Scalar("carry_cotangent".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        carry_inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    let mut output_inputs = carry_inputs.clone();
    let output_carry_offset =
        cuda_offset_expression_with_index(&output_step_shape, &carry_shape, "output_index");
    if !unequal_output_lanes {
        output_inputs.insert(
            scan_plan.carry_name.clone(),
            CudaElementwiseInput::Scalar(if output_step_shape == carry_shape {
                "carry".to_string()
            } else {
                format!(
                    "carry_tape[(step - {}ULL) * carry_count + {output_carry_offset}]",
                    scan_plan.lower
                )
            }),
        );
    }
    output_inputs.remove("__quabla_cuda_scan_vjp_carry_cotangent");
    output_inputs.insert(
        "__quabla_cuda_scan_vjp_output_cotangent".to_string(),
        CudaElementwiseInput::Scalar(if unequal_output_lanes {
            "output_cotangent_aggregate".to_string()
        } else {
            "output_cotangent_step".to_string()
        }),
    );
    let output_reference_shape = if unequal_output_lanes {
        &carry_shape
    } else {
        &output_step_shape
    };
    let output_index_expression = if unequal_output_lanes {
        "index"
    } else {
        "output_index"
    };
    let carry_target_expression = cuda_elementwise_plan_expression(
        &carry_target_plan,
        cuda_vjp_elementwise_output_node(&carry_target_plan, &target_shape, &carry_shape)?,
        &carry_inputs,
        &carry_shape,
    )?;
    let output_target_expression = cuda_elementwise_plan_expression_with_index(
        &output_target_plan,
        cuda_vjp_elementwise_output_node(
            &output_target_plan,
            &target_shape,
            output_reference_shape,
        )?,
        &output_inputs,
        output_reference_shape,
        output_index_expression,
    )?;
    let (carry_reverse_expression, output_reverse_expression) =
        if let Some((carry_plan, output_plan)) = carry_reverse_plans.as_ref() {
            (
                cuda_elementwise_plan_expression(
                    carry_plan,
                    carry_plan.output_node_id,
                    &carry_inputs,
                    &carry_shape,
                )?,
                cuda_elementwise_plan_expression_with_index(
                    output_plan,
                    cuda_vjp_elementwise_output_node(
                        output_plan,
                        &carry_shape,
                        output_reference_shape,
                    )?,
                    &output_inputs,
                    output_reference_shape,
                    output_index_expression,
                )?,
            )
        } else {
            (String::new(), String::new())
        };
    let reverse_update = match target {
        TensorScanVjpTarget::Carry => "carry_cotangent = contribution;".to_string(),
        TensorScanVjpTarget::External(_) => format!(
            "gradient += contribution;\n\\
                carry_cotangent = {carry_reverse_expression} + output_reverse_contribution;"
        ),
    };
    let target_write = match target {
        TensorScanVjpTarget::Carry => "out[index] = carry_cotangent;".to_string(),
        TensorScanVjpTarget::External(name) => {
            let target_shape = scan_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("CUDA Scan VJP has no capture {name:?}"))?;
            if target_shape == &carry_shape {
                "out[index] = gradient;".to_string()
            } else {
                format!(
                    "atomicAdd(out + {}, gradient);",
                    cuda_offset_expression(&carry_shape, target_shape)
                )
            }
        }
    };
    let output_reverse_update = if output_reverse_expression.is_empty() {
        String::new()
    } else {
        format!("output_reverse_contribution += {output_reverse_expression};")
    };
    let (output_loop_update, output_post_loop_update) = if unequal_output_lanes {
        (
            String::new(),
            format!(
                "output_contribution = {output_target_expression};\n\\
                {}",
                if output_reverse_expression.is_empty() {
                    String::new()
                } else {
                    format!("output_reverse_contribution = {output_reverse_expression};")
                }
            ),
        )
    } else {
        (
            format!(
                "output_contribution += {output_target_expression};\n\\
                {output_reverse_update}"
            ),
            String::new(),
        )
    };
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "const float* final_carry_cotangent".to_string(),
            "const float* output_cotangent".to_string(),
            "float* carry_tape".to_string(),
            "float* out".to_string(),
            "unsigned long long carry_count".to_string(),
            "unsigned long long output_count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let lower = scan_plan.lower;
    let upper = scan_plan.upper;
    let function = cuda_node_function_name(node_id);
    let block = cuda_loop_checkpoint_block(
        lower,
        upper,
        &carry_shape,
        scan_plan.external_captures(),
        Some(&scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]].shape),
        !cuda_scan_uses_packed_halves(scan_plan),
    );
    let (tape_forward, tape_reverse) = cuda_loop_tape_source(
        lower,
        upper,
        "carry_count",
        &forward_next_expression,
        None,
        block,
    );
    Ok(format!(
        "extern \"C\" __global__ void {function}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= carry_count) return;\n\\
            float carry = initial_carry[index];\n\\
            {tape_forward}\n\
            float carry_cotangent = final_carry_cotangent[index];\n\\
            float gradient = 0.0f;\n\\
            for (unsigned long long reverse = {upper}ULL; reverse > {lower}ULL; --reverse) {{\n\\
                unsigned long long step = reverse - 1ULL;\n\\
                float loop_index = (float)step;\n\\
                {tape_reverse}\n\\
                float output_contribution = 0.0f;\n\\
                float output_reverse_contribution = 0.0f;\n\\
                float output_cotangent_aggregate = 0.0f;\n\\
                for (unsigned long long output_index = 0ULL; output_index < output_count; ++output_index) {{\n\\
                    if ({output_carry_offset} == index) {{\n\\
                        float output_cotangent_step = output_cotangent[(step - {lower}ULL) * output_count + output_index];\n\\
                        output_cotangent_aggregate += output_cotangent_step;\n\\
                        {output_loop_update}\n\\
                    }}\n\\
                }}\n\\
                {output_post_loop_update}\n\\
                float contribution = ({carry_target_expression} + output_contribution);\n\\
                {reverse_update}\n\\
            }}\n\\
            {target_write}\n}}\n"))
}
fn cuda_scan_vjp_group_kernel_source(
    node_id: TensorNodeId,
    scan_plan: &TensorScanExecutionPlan,
    captures: &[(String, TensorNodeId)],
    targets: &[&TensorScanVjpTarget],
) -> Result<String, String> {
    let carry_shape = scan_plan.carry_shape()?;
    let (carry_reverse_plan, output_reverse_plan) =
        cuda_scan_vjp_plans(scan_plan, &TensorScanVjpTarget::Carry)?;
    let capture_parameters = captures
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let forward_next_expression = cuda_scan_body_expression(
        scan_plan,
        scan_plan.body.plan.output_node_ids[0],
        &capture_parameters,
        &carry_shape,
    )?;
    let mut carry_inputs = BTreeMap::new();
    carry_inputs.insert(
        scan_plan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    carry_inputs.insert(
        scan_plan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    carry_inputs.insert(
        "__quabla_cuda_scan_vjp_carry_cotangent".to_string(),
        CudaElementwiseInput::Scalar("carry_cotangent".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        carry_inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    let mut output_inputs = carry_inputs.clone();
    output_inputs.remove("__quabla_cuda_scan_vjp_carry_cotangent");
    output_inputs.insert(
        "__quabla_cuda_scan_vjp_output_cotangent".to_string(),
        CudaElementwiseInput::Scalar("output_cotangent_step".to_string()),
    );
    let carry_reverse_expression = cuda_elementwise_plan_expression(
        &carry_reverse_plan,
        carry_reverse_plan.output_node_id,
        &carry_inputs,
        &carry_shape,
    )?;
    let output_reverse_expression = cuda_elementwise_plan_expression(
        &output_reverse_plan,
        output_reverse_plan.output_node_id,
        &output_inputs,
        &carry_shape,
    )?;
    let output_step_shape =
        &scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]].shape;
    // With unequal lanes `cuda_scan_vjp_plans` differentiates the carry-shaped source of the
    // direct output broadcast (it rejects any other unequal-lane output), so each carry lane
    // consumes the sum of the step's output cotangents over the output lanes it broadcasts into,
    // as in `cuda_scan_vjp_node_kernel_source`. Equal lane counts map output lanes 1:1.
    let output_cotangent_setup = if element_count(output_step_shape)?
        != element_count(&carry_shape)?
    {
        let carry_offset =
            cuda_offset_expression_with_index(output_step_shape, &carry_shape, "output_index");
        format!(
            "float output_cotangent_step = 0.0f;\n\\
                for (unsigned long long output_index = 0ULL; output_index < output_count; ++output_index) {{\n\\
                    if ({carry_offset} == index) {{\n\\
                        output_cotangent_step += output_cotangent[(step - {}ULL) * output_count + output_index];\n\\
                    }}\n\\
                }}\n",
            scan_plan.lower
        )
    } else {
        format!(
            "float output_cotangent_step = output_cotangent[(step - {}ULL) * output_count + index];\n",
            scan_plan.lower
        )
    };

    let mut gradient_declarations = String::new();
    let mut contribution_updates = String::new();
    let mut target_writes = String::new();
    for (target_index, target) in targets.iter().enumerate() {
        match target {
            TensorScanVjpTarget::Carry => {
                target_writes.push_str(&format!("out_{target_index}[index] = carry_cotangent;\n"));
            }
            TensorScanVjpTarget::External(name) => {
                let target_shape = scan_plan
                    .external_captures()
                    .get(name)
                    .ok_or_else(|| format!("CUDA Scan VJP has no capture {name:?}"))?;
                let (carry_target_plan, output_target_plan) =
                    cuda_scan_vjp_plans(scan_plan, target)?;
                let carry_expression = cuda_elementwise_plan_expression(
                    &carry_target_plan,
                    cuda_vjp_elementwise_output_node(
                        &carry_target_plan,
                        target_shape,
                        &carry_shape,
                    )?,
                    &carry_inputs,
                    &carry_shape,
                )?;
                let output_expression = cuda_elementwise_plan_expression(
                    &output_target_plan,
                    cuda_vjp_elementwise_output_node(
                        &output_target_plan,
                        target_shape,
                        &carry_shape,
                    )?,
                    &output_inputs,
                    &carry_shape,
                )?;
                gradient_declarations.push_str(&format!("float gradient_{target_index} = 0.0f;\n"));
                contribution_updates.push_str(&format!(
                    "gradient_{target_index} += ({carry_expression} + {output_expression});\n"
                ));
                if target_shape == &carry_shape {
                    target_writes.push_str(&format!(
                        "out_{target_index}[index] = gradient_{target_index};\n"
                    ));
                } else {
                    target_writes.push_str(&format!(
                        "atomicAdd(out_{target_index} + {}, gradient_{target_index});\n",
                        cuda_offset_expression(&carry_shape, target_shape)
                    ));
                }
            }
        }
    }
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(index, _)| format!("const float* capture_{index}"))
        .chain([
            "const float* initial_carry".to_string(),
            "const float* final_carry_cotangent".to_string(),
            "const float* output_cotangent".to_string(),
            "float* carry_tape".to_string(),
        ])
        .chain(
            targets
                .iter()
                .enumerate()
                .map(|(index, _)| format!("float* out_{index}")),
        )
        .chain([
            "unsigned long long carry_count".to_string(),
            "unsigned long long output_count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let lower = scan_plan.lower;
    let upper = scan_plan.upper;
    let function = cuda_node_function_name(node_id);
    let block = cuda_loop_checkpoint_block(
        lower,
        upper,
        &carry_shape,
        scan_plan.external_captures(),
        Some(&scan_plan.body.plan.nodes[scan_plan.body.plan.output_node_ids[1]].shape),
        !cuda_scan_uses_packed_halves(scan_plan),
    );
    let (tape_forward, tape_reverse) = cuda_loop_tape_source(
        lower,
        upper,
        "carry_count",
        &forward_next_expression,
        None,
        block,
    );
    Ok(format!(
        "extern \"C\" __global__ void {function}({parameters}) {{\n\\
            unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
            if (index >= carry_count) return;\n\\
            float carry = initial_carry[index];\n\\
            {tape_forward}\n\
            float carry_cotangent = final_carry_cotangent[index];\n\\
            {gradient_declarations}\
            for (unsigned long long reverse = {upper}ULL; reverse > {lower}ULL; --reverse) {{\n\\
                unsigned long long step = reverse - 1ULL;\n\\
                float loop_index = (float)step;\n\\
                {tape_reverse}\n\\
                {output_cotangent_setup}\
                {contribution_updates}\
                carry_cotangent = ({carry_reverse_expression} + {output_reverse_expression});\n\\
            }}\n\\
            {target_writes}\
        }}\n"))
}
fn cuda_scan_vjp_jvp_group_kernel_source(
    node_id: TensorNodeId,
    plan: &TensorScanVjpJvpExecutionPlan,
    captures: &[(String, TensorNodeId)],
    targets: &[&TensorScanVjpTarget],
) -> Result<String, String> {
    if targets.is_empty() {
        return Err("CUDA Scan VJP JVP group has no targets".to_string());
    }
    cuda_scan_vjp_jvp_is_lowerable(plan)?;
    let scan = &plan.scan_plan;
    let shape = scan.carry_shape()?;
    let output_shape = scan.body.plan.nodes[scan.body.plan.output_node_ids[1]]
        .shape
        .clone();
    let direct_broadcast_output = cuda_scan_direct_broadcast_output_input(scan)?.is_some();
    let forward_tangents = plan
        .tangent_names
        .iter()
        .filter(|(name, _)| scan.body.captures.contains_key(*name))
        .map(|(name, tangent)| (name.clone(), tangent.clone()))
        .collect::<BTreeMap<_, _>>();
    let forward = scan
        .body
        .plan
        .as_ir()
        .symbolic_jvp_with_tangent_inputs(scan.body.plan.output_node_ids[0], &forward_tangents)?;
    let (forward_plan, forward_outputs) = forward
        .graph
        .compile_cpu_many(&[forward.value, forward.tangent])?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        scan.carry_name.clone(),
        CudaElementwiseInput::Scalar("carry".to_string()),
    );
    inputs.insert(
        scan.index_name.clone(),
        CudaElementwiseInput::Scalar("loop_index".to_string()),
    );
    for (index, (name, _)) in captures.iter().enumerate() {
        inputs.insert(name.clone(), CudaElementwiseInput::Buffer(index));
    }
    for (name, tangent_name) in &forward_tangents {
        let binding = if name == &scan.carry_name {
            CudaElementwiseInput::Scalar("carry_tangent".to_string())
        } else {
            let index = captures
                .iter()
                .position(|(candidate, _)| candidate == name)
                .ok_or_else(|| format!("CUDA Scan VJP JVP has no capture {name:?}"))?;
            CudaElementwiseInput::Buffer(captures.len() + index)
        };
        inputs.insert(tangent_name.clone(), binding);
    }
    let next =
        cuda_elementwise_plan_expression(&forward_plan, forward_outputs[0], &inputs, &shape)?;
    let next_tangent =
        cuda_elementwise_plan_expression(&forward_plan, forward_outputs[1], &inputs, &shape)?;
    let mut reverse = inputs;
    reverse.insert(
        plan.carry_cotangent_name.clone(),
        CudaElementwiseInput::Scalar("carry_cotangent".to_string()),
    );
    reverse.insert(
        plan.output_cotangent_name.clone(),
        CudaElementwiseInput::Scalar(if direct_broadcast_output {
            "output_cotangent_aggregate".to_string()
        } else {
            "output_cotangent_step".to_string()
        }),
    );
    reverse.insert(
        "__quabla_cuda_scan_vjp_carry_cotangent".to_string(),
        CudaElementwiseInput::Scalar("carry_cotangent".to_string()),
    );
    reverse.insert(
        "__quabla_cuda_scan_vjp_output_cotangent".to_string(),
        CudaElementwiseInput::Scalar(if direct_broadcast_output {
            "output_cotangent_aggregate".to_string()
        } else {
            "output_cotangent_step".to_string()
        }),
    );
    for (name, tangent_name) in &plan.tangent_names {
        if name == &plan.carry_cotangent_name {
            reverse.insert(
                tangent_name.clone(),
                CudaElementwiseInput::Scalar("carry_cotangent_tangent".to_string()),
            );
        }
        if name == &plan.output_cotangent_name {
            reverse.insert(
                tangent_name.clone(),
                CudaElementwiseInput::Scalar(if direct_broadcast_output {
                    "output_cotangent_tangent_aggregate".to_string()
                } else {
                    "output_cotangent_tangent_step".to_string()
                }),
            );
        }
    }
    let (carry_vjp, output_vjp) = cuda_scan_vjp_plans(scan, &TensorScanVjpTarget::Carry)?;
    let primal_carry =
        cuda_elementwise_plan_expression(&carry_vjp, carry_vjp.output_node_id, &reverse, &shape)?;
    let primal_output =
        cuda_elementwise_plan_expression(&output_vjp, output_vjp.output_node_id, &reverse, &shape)?;
    let mut gradient_declarations = String::new();
    let mut gradient_updates = String::new();
    let mut target_writes = String::new();
    for (target_index, target) in targets.iter().enumerate() {
        let name = match *target {
            TensorScanVjpTarget::Carry => &scan.carry_name,
            TensorScanVjpTarget::External(name) => name,
        };
        let target_shape = match *target {
            TensorScanVjpTarget::Carry => shape.clone(),
            TensorScanVjpTarget::External(name) => scan
                .external_captures()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("CUDA Scan VJP JVP has no capture {name:?}"))?,
        };
        let carry_directional = plan.carry_gradient_tangent_plans.get(name).ok_or_else(|| {
            format!("CUDA Scan VJP JVP has no carry directional plan for {name:?}")
        })?;
        let output_directional = if direct_broadcast_output {
            cuda_scan_vjp_jvp_direct_broadcast_output_directional_plan(plan, target)?
        } else {
            plan.output_gradient_tangent_plans
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    format!("CUDA Scan VJP JVP has no output directional plan for {name:?}")
                })?
        };
        let directional = format!(
            "({} + {})",
            cuda_elementwise_plan_expression(
                carry_directional,
                cuda_vjp_elementwise_output_node(carry_directional, &target_shape, &shape)?,
                &reverse,
                &shape
            )?,
            cuda_elementwise_plan_expression(
                &output_directional,
                cuda_vjp_elementwise_output_node(&output_directional, &target_shape, &shape)?,
                &reverse,
                &shape
            )?
        );
        gradient_declarations.push_str(&format!("float gradient_{target_index} = 0.0f; "));
        gradient_updates.push_str(&format!("gradient_{target_index} += {directional}; "));
        match *target {
            TensorScanVjpTarget::Carry => target_writes.push_str(&format!(
                "out_{target_index}[index] = carry_cotangent_tangent; "
            )),
            TensorScanVjpTarget::External(_) if target_shape == shape => {
                target_writes.push_str(&format!(
                    "out_{target_index}[index] = gradient_{target_index}; "
                ));
            }
            TensorScanVjpTarget::External(_) => target_writes.push_str(&format!(
                "atomicAdd(out_{target_index} + {}, gradient_{target_index}); ",
                cuda_offset_expression(&shape, &target_shape)
            )),
        }
    }
    let carry_directional = plan
        .carry_gradient_tangent_plans
        .get(&scan.carry_name)
        .ok_or_else(|| "CUDA Scan VJP JVP has no carry directional carry plan".to_string())?;
    let output_directional = if direct_broadcast_output {
        cuda_scan_vjp_jvp_direct_broadcast_output_directional_plan(
            plan,
            &TensorScanVjpTarget::Carry,
        )?
    } else {
        plan.output_gradient_tangent_plans
            .get(&scan.carry_name)
            .cloned()
            .ok_or_else(|| "CUDA Scan VJP JVP has no output directional carry plan".to_string())?
    };
    let next_cotangent_tangent = format!(
        "({} + {})",
        cuda_elementwise_plan_expression(
            carry_directional,
            carry_directional.output_node_id,
            &reverse,
            &shape
        )?,
        cuda_elementwise_plan_expression(
            &output_directional,
            output_directional.output_node_id,
            &reverse,
            &shape
        )?
    );
    let reverse_output_setup = if direct_broadcast_output {
        let carry_offset = cuda_offset_expression_with_index(&output_shape, &shape, "output_index");
        format!(
            "float output_cotangent_aggregate = 0.0f; float output_cotangent_tangent_aggregate = 0.0f; for (unsigned long long output_index = 0ULL; output_index < output_count; ++output_index) {{ if ({carry_offset} == index) {{ output_cotangent_aggregate += output_cotangent[(step - {}ULL) * output_count + output_index]; output_cotangent_tangent_aggregate += output_cotangent_tangent[(step - {}ULL) * output_count + output_index]; }} }}",
            scan.lower, scan.lower
        )
    } else {
        format!(
            "float output_cotangent_step = output_cotangent[(step - {}ULL) * carry_count + index]; float output_cotangent_tangent_step = output_cotangent_tangent[(step - {}ULL) * carry_count + index];",
            scan.lower, scan.lower
        )
    };
    let parameters = captures
        .iter()
        .enumerate()
        .map(|(i, _)| format!("const float* capture_{i}"))
        .chain(
            captures
                .iter()
                .enumerate()
                .map(|(i, _)| format!("const float* capture_{}", captures.len() + i)),
        )
        .chain([
            "const float* initial_carry".to_string(),
            "const float* initial_carry_tangent".to_string(),
            "const float* final_carry_cotangent".to_string(),
            "const float* final_carry_cotangent_tangent".to_string(),
            "const float* output_cotangent".to_string(),
            "const float* output_cotangent_tangent".to_string(),
            "float* carry_tape".to_string(),
            "float* carry_tangent_tape".to_string(),
        ])
        .chain(
            targets
                .iter()
                .enumerate()
                .map(|(index, _)| format!("float* out_{index}")),
        )
        .chain([
            "unsigned long long carry_count".to_string(),
            "unsigned long long output_count".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let lower = scan.lower;
    let upper = scan.upper;
    let function = cuda_node_function_name(node_id);
    let block = cuda_loop_checkpoint_block(
        lower,
        upper,
        &shape,
        scan.external_captures(),
        Some(&scan.body.plan.nodes[scan.body.plan.output_node_ids[1]].shape),
        !cuda_scan_uses_packed_halves(scan),
    );
    let (tape_forward, tape_reverse) = cuda_loop_tape_source(
        lower,
        upper,
        "carry_count",
        &next,
        Some(&next_tangent),
        block,
    );
    Ok(format!("extern \"C\" __global__ void {function}({parameters}) {{\n\
        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; if (index >= carry_count) return;\n\
        float carry = initial_carry[index]; float carry_tangent = initial_carry_tangent[index]; {tape_forward}\n\
            float carry_cotangent = final_carry_cotangent[index]; float carry_cotangent_tangent = final_carry_cotangent_tangent[index]; {gradient_declarations}\n\
        for (unsigned long long reverse_step = {upper}ULL; reverse_step > {lower}ULL; --reverse_step) {{ unsigned long long step = reverse_step - 1ULL; float loop_index = (float)step; {tape_reverse} {reverse_output_setup} {gradient_updates} float nc = ({primal_carry} + {primal_output}); float nct = {next_cotangent_tangent}; carry_cotangent = nc; carry_cotangent_tangent = nct; }}\n\
        {target_writes}\n}}\n"))
}
/// `host_driven` lists loop nodes that run as host-driven region loops; they
/// launch their regions' own programs, so no node kernel is emitted for them.
/// The shared CUDA formula (`cuda_elementwise_formula`) of one per-node
/// kernel, whose operand `position` is the buffer read `operands[position]`.
fn cuda_node_formula(
    node_id: TensorNodeId,
    op: &TensorOp,
    operands: &[String],
) -> Result<String, String> {
    cuda_elementwise_formula(op, |position, _| {
        operands
            .get(position)
            .cloned()
            .ok_or_else(|| format!("CUDA node {node_id} reads a missing operand {position}"))
    })?
    .ok_or_else(|| format!("CUDA node {node_id} has no elementwise formula"))
}

fn cuda_program_source(
    plan: &TensorExecutionPlan,
    host_driven: &BTreeSet<TensorNodeId>,
) -> Result<String, String> {
    let mut source = String::from(
        "__device__ __forceinline__ float quabla_powi(float base, unsigned int exponent) {\n\
    float result = 1.0f;\n\
    while (exponent != 0U) { if ((exponent & 1U) != 0U) result *= base; base *= base; exponent >>= 1U; }\n\
    return result;\n}\n",
    );
    source.push_str(optimizer::cuda_optimizer_source());
    source.push_str(
        "extern \"C\" __global__ void quabla_transpose_copy(const float* input, float* output, unsigned long long batch, unsigned long long rows, unsigned long long columns) {\n\\
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
    unsigned long long size = rows * columns;\n\\
    if (index < batch * size) { unsigned long long base = index - index % size; unsigned long long local = index % size; unsigned long long row = local / columns; unsigned long long column = local % columns; output[base + column * rows + row] = input[index]; }\n}\n",
    );
    source.push_str(CUDA_LINALG_SOURCE);
    source.push_str(decompositions::CUDA_DECOMPOSITION_SOURCE);
    for (node_id, node) in plan.nodes.iter().enumerate() {
        if host_driven.contains(&node_id) {
            continue;
        }
        let function = cuda_node_function_name(node_id);
        let count = element_count(&node.shape)?;
        let kernel = match &node.op {
            TensorOp::Input { .. }
            | TensorOp::Constant { .. }
            | TensorOp::Reshape { .. }
            | TensorOp::Cast { .. }
            | TensorOp::StopGradient { .. }
            | TensorOp::Custom { .. } => continue,
            TensorOp::ScalarConstant { value } if value.is_finite() => format!(
                "extern \"C\" __global__ void {function}(float* out, unsigned long long count) {{\n\
                    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                    if (index < count) out[index] = {};\n}}\n",
                cuda_scalar_literal(*value)
            ),
            TensorOp::ScalarConstant { .. } => {
                return Err("CUDA device program does not support non-finite constants".to_string())
            }
            TensorOp::Triangular { input: _, lower } => {
                let rows = node.shape[node.shape.len() - 2];
                let columns = node.shape[node.shape.len() - 1];
                let matrix_size = rows * columns;
                let predicate = if *lower { "row >= column" } else { "row <= column" };
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\\
                        if (index < count) {{ unsigned long long local = index % {matrix_size}ULL; unsigned long long row = local / {columns}ULL; unsigned long long column = local % {columns}ULL; out[index] = ({predicate}) ? input[index] : 0.0f; }}\n}}\n"
                )
            }
            TensorOp::Add { lhs, rhs }
            | TensorOp::Sub { lhs, rhs }
            | TensorOp::Div { lhs, rhs }
            | TensorOp::Mul { lhs, rhs }
            | TensorOp::Greater { lhs, rhs }
            | TensorOp::Compare { lhs, rhs, .. }
            | TensorOp::BinaryMath { lhs, rhs, .. } => {
                let lhs_offset = cuda_offset_expression(&node.shape, &plan.nodes[*lhs].shape);
                let rhs_offset = cuda_offset_expression(&node.shape, &plan.nodes[*rhs].shape);
                let operands = [format!("lhs[{lhs_offset}]"), format!("rhs[{rhs_offset}]")];
                let expression = cuda_node_formula(node_id, &node.op, &operands)?;
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
                let operands = [
                    format!("condition[{condition_offset}]"),
                    format!("on_true[{true_offset}]"),
                    format!("on_false[{false_offset}]"),
                ];
                let expression = cuda_node_formula(node_id, &node.op, &operands)?;
                format!(
                    "extern \"C\" __global__ void {function}(const float* condition, const float* on_true, const float* on_false, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) out[index] = {expression};\n}}\n"
                )
            }
            TensorOp::Sqrt { input }
            | TensorOp::SqrtDerivative { input, .. }
            | TensorOp::Powi { input, .. }
            | TensorOp::UnaryMath { input, .. } => {
                let expression =
                    cuda_node_formula(node_id, &node.op, &["input[index]".to_string()])?;
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
            TensorOp::Cholesky { .. } => cholesky_backend::primal_source(&function, *node.shape.last().expect("matrix shape")),
            TensorOp::CholeskyAd { kind, .. } => cholesky_backend::ad_source(&function, *node.shape.last().expect("matrix shape"), *kind),
            TensorOp::Solve { .. }
            | TensorOp::Linalg { .. }
            | TensorOp::Region(RegionNode { kind: RegionKind::Cond { .. }, .. })
            | TensorOp::Region(RegionNode { kind: RegionKind::While { .. }, .. }) => {
                String::new()
            }
            TensorOp::Region(RegionNode { kind: RegionKind::Fori { loop_plan, .. }, captures, .. }) => cuda_fori_node_kernel_source(node_id, loop_plan, captures)?,
            TensorOp::Region(RegionNode { kind: RegionKind::ForiJvp { loop_plan, .. }, captures, .. }) => cuda_fori_jvp_node_kernel_source(node_id, loop_plan, captures)?,
            TensorOp::Region(RegionNode { kind: RegionKind::ForiVjp { loop_plan, group, .. }, captures, .. }) => {
                let members = cuda_fori_vjp_group(plan, *group)?;
                if members.first().map(|(member_id, _)| *member_id) != Some(node_id) {
                    continue;
                }
                let targets = members
                    .iter()
                    .map(|(_, target)| *target)
                    .collect::<Vec<_>>();
                if targets.len() == 1 {
                    cuda_fori_vjp_node_kernel_source(node_id, loop_plan, captures, targets[0])?
                } else {
                    cuda_fori_vjp_group_kernel_source(node_id, loop_plan, captures, &targets)?
                }
            }
            TensorOp::Region(RegionNode { kind: RegionKind::ForiVjpJvp { plan, target, .. }, captures, .. }) => cuda_fori_vjp_jvp_node_kernel_source(node_id, plan, captures, target)?,
            TensorOp::Region(RegionNode { kind: RegionKind::Scan { scan_plan, .. }, captures, .. }) => cuda_scan_node_kernel_source(node_id, scan_plan, captures)?,
            TensorOp::Region(RegionNode { kind: RegionKind::ScanVjp { scan_plan, group, .. }, captures, .. }) => {
                let members = cuda_scan_vjp_group(plan, *group)?;
                if members.first().map(|(member_id, _)| *member_id) != Some(node_id) {
                    continue;
                }
                let targets = members
                    .iter()
                    .map(|(_, target)| *target)
                    .collect::<Vec<_>>();
                if targets.len() == 1 {
                    cuda_scan_vjp_node_kernel_source(node_id, scan_plan, captures, targets[0])?
                } else {
                    cuda_scan_vjp_group_kernel_source(node_id, scan_plan, captures, &targets)?
                }
            }
            TensorOp::Region(RegionNode { kind: RegionKind::ScanVjpJvp { plan: scan_hvp, group, .. }, captures, .. }) => {
                let members = cuda_scan_vjp_jvp_group(plan, *group)?;
                if members.first().map(|(member_id, _)| *member_id) != Some(node_id) {
                    continue;
                }
                let targets = members
                    .iter()
                    .map(|(_, target)| *target)
                    .collect::<Vec<_>>();
                cuda_scan_vjp_jvp_group_kernel_source(node_id, scan_hvp, captures, &targets)?
            }
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
            // One thread scans one line along `axis` with `__fadd_rn`, so each running sum
            // is the rounded float add of the CPU reference, in the same order.
            TensorOp::CumSum {
                input,
                axis,
                reverse,
            } => {
                if plan.nodes[*input].shape != node.shape || *axis >= node.shape.len() {
                    return Err(format!("CUDA cumsum node {node_id} has an invalid shape or axis"));
                }
                let extent = node.shape[*axis];
                let inner = element_count(&node.shape[*axis + 1..])?;
                let (first, step) = if *reverse {
                    (format!("{}ULL * {inner}ULL", extent - 1), format!("-{inner}LL"))
                } else {
                    ("0ULL".to_string(), format!("{inner}LL"))
                };
                format!(
                    "extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long line = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (line >= count / {extent}ULL) return;\n\
                        long long position = (long long)((line / {inner}ULL) * {extent}ULL * {inner}ULL + line % {inner}ULL + {first});\n\
                        float running = input[position];\n\
                        out[position] = running;\n\
                        for (unsigned long long k = 1ULL; k < {extent}ULL; ++k) {{ position += {step}; running = __fadd_rn(running, input[position]); out[position] = running; }}\n}}\n"
                )
            }
            TensorOp::ExtremumAxis { input, axis, kind } => {
                let input_shape = &plan.nodes[*input].shape;
                let base = cuda_axis_reduction_base(node_id, input_shape, &node.shape, *axis)?;
                cuda_extremum_axis_kernel_source(
                    &function,
                    &base,
                    input_shape[*axis],
                    contiguous_strides(input_shape)[*axis],
                    *kind,
                )
            }
            TensorOp::SumAxis { input, axis } | TensorOp::MeanAxis { input, axis } => {
                let input_shape = &plan.nodes[*input].shape;
                let base = cuda_axis_reduction_base(node_id, input_shape, &node.shape, *axis)?;
                let input_strides = contiguous_strides(input_shape);
                let scale = if matches!(&node.op, TensorOp::MeanAxis { .. }) {
                    format!(" / {}.0f", input_shape[*axis])
                } else {
                    String::new()
                };
                if input_shape[*axis] >= CUDA_REDUCTION_BLOCK as usize {
                    // A block owns one output: no partial-sum allocation or atomic races.
                    // Wide accumulation limits cancellation error; extreme magnitudes keep
                    // the original serial overflow behavior before narrowing the result.
                    let extent = input_shape[*axis];
                    let stride = input_strides[*axis];
                    let divisor = if matches!(&node.op, TensorOp::MeanAxis { .. }) {
                        format!(" / (double){extent}ULL")
                    } else {
                        String::new()
                    };
                    format!(
                        r#"extern "C" __global__ void {function}(const float* input, float* out, unsigned long long count) {{
    unsigned long long index = blockIdx.x;
    if (index >= count) return;
    unsigned int thread = threadIdx.x;
    unsigned long long base = {base};
    __shared__ double partial[256];
    __shared__ unsigned int extreme;
    if (thread == 0U) extreme = 0U;
    __syncthreads();
    double value = 0.0;
    for (unsigned long long k = thread; k < {extent}ULL; k += 256ULL) {{
        float lane = input[base + k * {stride}ULL];
        if (isfinite(lane) && fabsf(lane) > 3.4028234663852886e38f / (float){extent}ULL) atomicExch(&extreme, 1U);
        value += (double)lane;
    }}
    partial[thread] = value;
    __syncthreads();
    if (extreme) {{
        if (thread == 0U) {{
            float serial = 0.0f;
            for (unsigned long long k = 0; k < {extent}ULL; ++k) serial += input[base + k * {stride}ULL];
            out[index] = serial{scale};
        }}
        return;
    }}
    for (unsigned int stride = 128U; stride > 0U; stride >>= 1U) {{
        if (thread < stride) partial[thread] += partial[thread + stride];
        __syncthreads();
    }}
    if (thread == 0U) out[index] = (float)(partial[0]{divisor});
}}
"#
                    )
                } else {
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
            // Index tables are baked into the module as device constants, like
            // the static offsets of slice and concat kernels.
            TensorOp::Gather {
                input,
                indices,
                axis,
            } => {
                let input_extent = plan.nodes[*input].shape[*axis];
                let inner = element_count(&node.shape[*axis + 1..])?;
                format!(
                    "__device__ const unsigned int {function}_indices[{}] = {{{}}};\n\
                    extern \"C\" __global__ void {function}(const float* input, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index < count) {{ unsigned long long outer = index / {}ULL; unsigned long long rem = index % {}ULL; unsigned long long position = rem / {inner}ULL; unsigned long long lane = rem % {inner}ULL; out[index] = input[(outer * {input_extent}ULL + {function}_indices[position]) * {inner}ULL + lane]; }}\n}}\n",
                    indices.len(),
                    cuda_index_table(indices)?,
                    indices.len() * inner,
                    indices.len() * inner,
                )
            }
            // One thread per output element adds its contributions in update
            // order (destination-grouped offsets and sources), so duplicates
            // round exactly like the CPU reference's sequential adds.
            TensorOp::ScatterAdd { indices, axis, .. } => {
                let extent = node.shape[*axis];
                let inner = element_count(&node.shape[*axis + 1..])?;
                let (offsets, sources) = cuda_scatter_tables(indices, extent);
                format!(
                    "__device__ const unsigned int {function}_offsets[{}] = {{{}}};\n\
                    __device__ const unsigned int {function}_sources[{}] = {{{}}};\n\
                    extern \"C\" __global__ void {function}(const float* base, const float* updates, float* out, unsigned long long count) {{\n\
                        unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
                        if (index >= count) return;\n\
                        unsigned long long outer = index / {}ULL; unsigned long long row = (index / {inner}ULL) % {extent}ULL; unsigned long long lane = index % {inner}ULL;\n\
                        float value = base[index];\n\
                        for (unsigned int k = {function}_offsets[row]; k < {function}_offsets[row + 1ULL]; ++k) value = value + updates[(outer * {}ULL + {function}_sources[k]) * {inner}ULL + lane];\n\
                        out[index] = value;\n}}\n",
                    offsets.len(),
                    cuda_index_table(&offsets)?,
                    sources.len(),
                    cuda_index_table(&sources)?,
                    extent * inner,
                    indices.len(),
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
    format!("quabla_node_{node_id}")
}

/// The input offset of the first entry of the line that output element
/// `index` of an axis reduction reduces, after checking that the output
/// shape is the input shape without `axis`.
fn cuda_axis_reduction_base(
    node_id: TensorNodeId,
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
) -> Result<String, String> {
    if input_shape.is_empty() || axis >= input_shape.len() {
        return Err(format!(
            "CUDA axis reduction node {node_id} has an invalid axis"
        ));
    }
    let expected_shape = input_shape
        .iter()
        .enumerate()
        .filter_map(|(input_axis, extent)| (input_axis != axis).then_some(*extent))
        .collect::<Vec<_>>();
    if output_shape != expected_shape {
        return Err(format!(
            "CUDA axis reduction node {node_id} has shape {output_shape:?}, expected {expected_shape:?}"
        ));
    }
    let input_strides = contiguous_strides(input_shape);
    let base_terms = (0..input_shape.len())
        .filter(|input_axis| *input_axis != axis)
        .map(|input_axis| {
            let output_axis = if input_axis < axis {
                input_axis
            } else {
                input_axis - 1
            };
            let output_stride = output_shape[output_axis + 1..].iter().product::<usize>();
            format!(
                "((index / {output_stride}ULL) % {}ULL) * {}ULL",
                output_shape[output_axis], input_strides[input_axis]
            )
        })
        .collect::<Vec<_>>();
    Ok(if base_terms.is_empty() {
        "0ULL".to_string()
    } else {
        base_terms.join(" + ")
    })
}

/// The kernel of `TensorOp::ExtremumAxis`, shaped like the sum-axis kernel:
/// one thread folds one line of fewer than `CUDA_REDUCTION_BLOCK` entries,
/// and a block of `CUDA_REDUCTION_BLOCK` threads owns each longer line, each
/// thread folding a strided share before a shared-memory tree combines them.
/// The fold (`TensorExtremum::cuda_fold`) is order independent, so both
/// shapes return the CPU value bit for bit, deterministically, with no
/// atomics; the double-precision rewrite keeps `isnan` and `signbit`, which
/// are overloaded for `double`.
fn cuda_extremum_axis_kernel_source(
    function: &str,
    base: &str,
    extent: usize,
    stride: usize,
    kind: TensorExtremum,
) -> String {
    let fold_lane = kind.cuda_fold("value", "lane");
    if extent >= CUDA_REDUCTION_BLOCK as usize {
        let fold_partial = kind.cuda_fold("value", "partial[thread + width]");
        // `extent >= 256`, so every thread starts from an entry of its own.
        format!(
            r#"extern "C" __global__ void {function}(const float* input, float* out, unsigned long long count) {{
    unsigned long long index = blockIdx.x;
    if (index >= count) return;
    unsigned int thread = threadIdx.x;
    unsigned long long base = {base};
    __shared__ float partial[256];
    float value = input[base + (unsigned long long)thread * {stride}ULL];
    for (unsigned long long k = thread + 256ULL; k < {extent}ULL; k += 256ULL) {{
        float lane = input[base + k * {stride}ULL];
        {fold_lane}
    }}
    partial[thread] = value;
    __syncthreads();
    for (unsigned int width = 128U; width > 0U; width >>= 1U) {{
        if (thread < width) {{
            {fold_partial}
            partial[thread] = value;
        }}
        __syncthreads();
    }}
    if (thread == 0U) out[index] = partial[0];
}}
"#
        )
    } else {
        format!(
            r#"extern "C" __global__ void {function}(const float* input, float* out, unsigned long long count) {{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned long long base = {base};
    float value = input[base];
    for (unsigned long long k = 1ULL; k < {extent}ULL; ++k) {{
        float lane = input[base + k * {stride}ULL];
        {fold_lane}
    }}
    out[index] = value;
}}
"#
        )
    }
}

fn cuda_offset_expression(output_shape: &[usize], input_shape: &[usize]) -> String {
    cuda_offset_expression_with_index(output_shape, input_shape, "index")
}

fn cuda_offset_expression_with_index(
    output_shape: &[usize],
    input_shape: &[usize],
    index_expression: &str,
) -> String {
    let input_strides = contiguous_strides(input_shape);
    // A contiguous reshape preserves the flat element order.
    if output_shape.iter().product::<usize>() == input_shape.iter().product::<usize>() {
        return index_expression.to_string();
    }
    let rank_offset = output_shape.len() - input_shape.len();
    let terms = (0..output_shape.len())
        .filter(|axis| *axis >= rank_offset && input_shape[*axis - rank_offset] != 1)
        .map(|axis| {
            let output_stride = output_shape[axis + 1..].iter().product::<usize>();
            format!(
                "(({index_expression} / {output_stride}ULL) % {}ULL) * {}ULL",
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

/// Two shapes that differ only in leading unit axes have the same offset mapping under
/// trailing-aligned broadcasting.
fn cuda_shapes_match_without_leading_units(lhs: &[usize], rhs: &[usize]) -> bool {
    let strip = |shape: &[usize]| -> Vec<usize> {
        shape
            .iter()
            .copied()
            .skip_while(|extent| *extent == 1)
            .collect()
    };
    strip(lhs) == strip(rhs)
}

fn cuda_shapes_broadcastable(output_shape: &[usize], input_shape: &[usize]) -> bool {
    if input_shape.len() > output_shape.len() {
        return false;
    }
    let rank_offset = output_shape.len() - input_shape.len();
    output_shape[rank_offset..]
        .iter()
        .zip(input_shape)
        .all(|(output_extent, input_extent)| *input_extent == 1 || input_extent == output_extent)
}

/// A comma-separated `unsigned int` initializer for a baked index table.
fn cuda_index_table(values: &[usize]) -> Result<String, String> {
    let mut table = String::with_capacity(values.len() * 6);
    for (position, value) in values.iter().enumerate() {
        let value = u32::try_from(*value)
            .map_err(|_| format!("CUDA index table entry {value} exceeds u32"))?;
        if position > 0 {
            table.push(',');
        }
        table.push_str(&value.to_string());
    }
    Ok(table)
}

/// Scatter contributions grouped by destination: `sources[offsets[row]..
/// offsets[row + 1]]` are the update positions that add into `row`, in
/// increasing order.
fn cuda_scatter_tables(indices: &[usize], extent: usize) -> (Vec<usize>, Vec<usize>) {
    let mut offsets = vec![0; extent + 1];
    for index in indices {
        offsets[index + 1] += 1;
    }
    for row in 0..extent {
        offsets[row + 1] += offsets[row];
    }
    let mut cursor = offsets[..extent].to_vec();
    let mut sources = vec![0; indices.len()];
    for (source, index) in indices.iter().enumerate() {
        sources[cursor[*index]] = source;
        cursor[*index] += 1;
    }
    (offsets, sources)
}

fn cuda_op_name(op: &TensorOp) -> &'static str {
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
        TensorOp::Region(region) => region.name(),
        TensorOp::Sum { .. } => "sum",
        TensorOp::SumAxis { .. } => "sum_axis",
        TensorOp::ExtremumAxis { kind, .. } => match kind {
            TensorExtremum::Max => "max_axis",
            TensorExtremum::Min => "min_axis",
        },
        TensorOp::Matmul { .. } => "matmul",
        TensorOp::Solve { .. } => "solve",
        TensorOp::Linalg { kind, .. } => kind.name(),
        TensorOp::Cholesky { .. } => "cholesky",
        TensorOp::CholeskyAd { .. } => "cholesky_ad",
        TensorOp::Triangular { .. } => "triangular",
        TensorOp::Sqrt { .. } => "sqrt",
        TensorOp::SqrtDerivative { .. } => "sqrt_derivative",
        TensorOp::Reshape { .. } => "reshape",
        TensorOp::Mean { .. } => "mean",
        TensorOp::MeanAxis { .. } => "mean_axis",
        TensorOp::Powi { .. } => "powi",
        TensorOp::BinaryMath { kind, .. } => kind.name(),
        TensorOp::Transpose { .. } => "transpose",
        TensorOp::UnaryMath { kind, .. } => kind.name(),
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
constexpr unsigned int QUABLA_TILE = 32;
constexpr unsigned int QUABLA_BLOCK = 16;

extern "C" __global__ void quabla_rank_two_matmul(
    const float* lhs,
    const float* rhs,
    float* output,
    unsigned long long rows,
    unsigned long long inner,
    unsigned long long cols
) {
    __shared__ float lhs_tile[QUABLA_TILE][QUABLA_TILE];
    __shared__ float rhs_tile[QUABLA_TILE][QUABLA_TILE];
    const unsigned long long row = (unsigned long long)blockIdx.y * QUABLA_TILE + threadIdx.y;
    const unsigned long long col = (unsigned long long)blockIdx.x * QUABLA_TILE + threadIdx.x;
    float value_00 = 0.0f;
    float value_01 = 0.0f;
    float value_10 = 0.0f;
    float value_11 = 0.0f;
    const unsigned long long tile_count = (inner + QUABLA_TILE - 1) / QUABLA_TILE;
    for (unsigned long long tile = 0; tile < tile_count; ++tile) {
        const unsigned long long lhs_col = tile * QUABLA_TILE + threadIdx.x;
        const unsigned long long rhs_row = tile * QUABLA_TILE + threadIdx.y;
        lhs_tile[threadIdx.y][threadIdx.x] = row < rows && lhs_col < inner
            ? lhs[row * inner + lhs_col] : 0.0f;
        lhs_tile[threadIdx.y][threadIdx.x + QUABLA_BLOCK] = row < rows && lhs_col + QUABLA_BLOCK < inner
            ? lhs[row * inner + lhs_col + QUABLA_BLOCK] : 0.0f;
        lhs_tile[threadIdx.y + QUABLA_BLOCK][threadIdx.x] = row + QUABLA_BLOCK < rows && lhs_col < inner
            ? lhs[(row + QUABLA_BLOCK) * inner + lhs_col] : 0.0f;
        lhs_tile[threadIdx.y + QUABLA_BLOCK][threadIdx.x + QUABLA_BLOCK] = row + QUABLA_BLOCK < rows && lhs_col + QUABLA_BLOCK < inner
            ? lhs[(row + QUABLA_BLOCK) * inner + lhs_col + QUABLA_BLOCK] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x] = rhs_row < inner && col < cols
            ? rhs[rhs_row * cols + col] : 0.0f;
        rhs_tile[threadIdx.y][threadIdx.x + QUABLA_BLOCK] = rhs_row < inner && col + QUABLA_BLOCK < cols
            ? rhs[rhs_row * cols + col + QUABLA_BLOCK] : 0.0f;
        rhs_tile[threadIdx.y + QUABLA_BLOCK][threadIdx.x] = rhs_row + QUABLA_BLOCK < inner && col < cols
            ? rhs[(rhs_row + QUABLA_BLOCK) * cols + col] : 0.0f;
        rhs_tile[threadIdx.y + QUABLA_BLOCK][threadIdx.x + QUABLA_BLOCK] = rhs_row + QUABLA_BLOCK < inner && col + QUABLA_BLOCK < cols
            ? rhs[(rhs_row + QUABLA_BLOCK) * cols + col + QUABLA_BLOCK] : 0.0f;
        __syncthreads();
        for (unsigned int k = 0; k < QUABLA_TILE; ++k) {
            float lhs_top = lhs_tile[threadIdx.y][k];
            float lhs_bottom = lhs_tile[threadIdx.y + QUABLA_BLOCK][k];
            float rhs_left = rhs_tile[k][threadIdx.x];
            float rhs_right = rhs_tile[k][threadIdx.x + QUABLA_BLOCK];
            value_00 += lhs_top * rhs_left;
            value_01 += lhs_top * rhs_right;
            value_10 += lhs_bottom * rhs_left;
            value_11 += lhs_bottom * rhs_right;
        }
        __syncthreads();
    }
    if (row < rows && col < cols) output[row * cols + col] = value_00;
    if (row < rows && col + QUABLA_BLOCK < cols) output[row * cols + col + QUABLA_BLOCK] = value_01;
    if (row + QUABLA_BLOCK < rows && col < cols) output[(row + QUABLA_BLOCK) * cols + col] = value_10;
    if (row + QUABLA_BLOCK < rows && col + QUABLA_BLOCK < cols) output[(row + QUABLA_BLOCK) * cols + col + QUABLA_BLOCK] = value_11;
}
"#;

const CUDA_MATMUL_BIAS_TANH_SOURCE: &str = r#"
extern "C" __global__ void quabla_matmul_bias_tanh(
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
    CUDA_REGISTER_TILED_MATMUL_TEMPLATE.replace("QUABLA_MATMUL_FUNCTION", function)
}

const CUDA_REGISTER_TILED_MATMUL_TEMPLATE: &str = r#"
extern "C" __global__ void QUABLA_MATMUL_FUNCTION(
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

#[cfg(test)]
mod region_codegen_tests {
    use super::*;
    use crate::tensor_ir::TensorIr;

    #[test]
    fn shared_fori_and_scan_body_sources_are_linear() -> Result<(), String> {
        for depth in [6, 18, 40] {
            let mut body = TensorIr::new();
            let carry = body.input("carry", vec![8])?;
            let index = body.input("index", vec![])?;
            let mut next = carry;
            for _ in 0..depth {
                next = body.add(next, next)?;
            }
            next = body.add(next, index)?;
            let fori =
                TensorForiExecutionPlan::new(0, 2, body.compile_cpu(next)?, "carry", "index")?;
            let source = cuda_fori_body_expression(
                &fori,
                fori.body.plan.output_node_id,
                &BTreeMap::new(),
                &[8],
            )?;
            assert_eq!(
                source.matches("const float quabla_body_value_").count(),
                fori.body.plan.node_count()
            );
            assert_eq!(source.matches("[&]").count(), 1);
            assert!(source.len() < 256 * fori.body.plan.node_count());
            let (plan, _) = body.compile_cpu_many(&[next, next])?;
            let scan = TensorScanExecutionPlan::new(0, 2, plan, "carry", "index")?;
            let source = cuda_scan_body_expression(
                &scan,
                scan.body.plan.output_node_ids[0],
                &BTreeMap::new(),
                &[8],
            )?;
            assert_eq!(
                source.matches("const float quabla_body_value_").count(),
                scan.body.plan.node_count()
            );
            assert_eq!(source.matches("[&]").count(), 1);
            assert!(source.len() < 256 * scan.body.plan.node_count());
            let input_bindings = BTreeMap::from([
                ("carry".into(), CudaElementwiseInput::Scalar("carry".into())),
                (
                    "index".into(),
                    CudaElementwiseInput::Scalar("loop_index".into()),
                ),
            ]);
            let source = cuda_elementwise_plan_expression(
                &fori.body.plan,
                fori.body.plan.output_node_id,
                &input_bindings,
                &[8],
            )?;
            assert_eq!(
                source.matches("const float quabla_body_value_").count(),
                fori.body.plan.node_count()
            );
            assert!(source.len() < 256 * fori.body.plan.node_count());
        }
        Ok(())
    }

    #[test]
    fn packed_scan_keeps_distinct_half_references() -> Result<(), String> {
        let mut body = TensorIr::new();
        let carry = body.input("carry", vec![2, 8])?;
        let index = body.input("index", vec![])?;
        let first = body.slice_axis(carry, 0, 0, 1)?;
        let second = body.slice_axis(carry, 0, 1, 2)?;
        let combined = body.add(first, second)?;
        let combined = body.add(combined, index)?;
        let packed = body.concat(vec![combined, second], 0)?;
        let (plan, _) = body.compile_cpu_many(&[packed, packed])?;
        let scan = TensorScanExecutionPlan::new(0, 2, plan, "carry", "index")?;
        let source = cuda_scan_body_expression_in_half(
            &scan,
            scan.body.plan.output_node_ids[0],
            &BTreeMap::new(),
            &[2, 8],
            "carry",
            Some((0, 8)),
        )?;
        assert!(source.contains("= carry_0;"));
        assert!(source.contains("= carry_1;"));
        assert_eq!(source.matches("[&]").count(), 1);
        assert!(source.len() < 512 * scan.body.plan.node_count());
        Ok(())
    }
}

#[cfg(test)]
mod solver_workspace_profile_tests {
    use super::*;
    use crate::tensor_ir::TensorIr;
    use std::hash::{Hash, Hasher};
    use std::time::Instant;

    #[test]
    fn profile_solver_workspace_shape_changes() -> Result<(), String> {
        if std::env::var_os("QUABLA_CUDA_PROFILE").is_none() {
            return Ok(());
        }
        let mut graph = TensorIr::new();
        let matrix = graph.input("matrix", vec![8, 8])?;
        let rhs = graph.input("rhs", vec![8, 1])?;
        let output = graph.solve(matrix, rhs)?;
        let plan = CudaBackend::new(0).compile(graph.compile_cpu(output)?)?;
        let solver = plan.solver.as_ref().ok_or("CUSOLVER is unavailable")?;
        let stream = plan.context.default_stream();
        let unbounded_baseline = std::env::var_os("QUABLA_CUDA_SOLVER_UNBOUNDED_PROFILE").is_some();
        let mut samples = Vec::new();
        let mut output_checksums = Vec::new();
        let mut retained_bytes = Vec::new();
        let mut entries = Vec::new();
        let mut current_bytes = Vec::new();
        // Revisit the largest shape after smaller shapes, then repeat it to
        // distinguish bounded eviction from unchanged same-shape reuse.
        for (n, columns) in [(192, 4), (64, 2), (8, 1), (64, 4), (192, 4), (192, 4)] {
            let mut matrix = vec![0.01_f32; n * n];
            for row in 0..n {
                matrix[row * n + row] = 2.0;
            }
            let rhs = vec![1.0_f32; n * columns];
            let matrix = stream
                .clone_htod(&matrix)
                .map_err(|error| format!("failed to upload test matrix: {error:?}"))?;
            let rhs = stream
                .clone_htod(&rhs)
                .map_err(|error| format!("failed to upload test RHS: {error:?}"))?;
            let mut output = stream
                .alloc_zeros::<f32>(n * columns)
                .map_err(|error| format!("failed to allocate test output: {error:?}"))?;
            let start = Instant::now();
            launch_cusolver_solve(
                &stream,
                &plan.module,
                solver,
                &matrix,
                &rhs,
                &mut output,
                1,
                n,
                columns,
            )?;
            let actual = stream
                .clone_dtoh(&output)
                .map_err(|error| format!("failed to read test solution: {error:?}"))?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            let expected = 1.0 / (2.0 + (n - 1) as f64 * 0.01);
            let mut checksum = std::collections::hash_map::DefaultHasher::new();
            for value in actual {
                value.to_bits().hash(&mut checksum);
                assert!((f64::from(value) - expected).abs() < 1e-5);
            }
            output_checksums.push(checksum.finish());
            let solver = solver.lock().map_err(|_| "solver lock was poisoned")?;
            let bytes = |workspace: &CudaSolveWorkspace<f32>| {
                4 * (workspace.factor.len()
                    + workspace.column_rhs.len()
                    + workspace.pivots.len()
                    + workspace.info.len()
                    + workspace.scratch.len())
            };
            let current = solver
                .workspaces
                .get(&(1, n, columns))
                .ok_or("current solver workspace is missing")?;
            current_bytes.push(bytes(current));
            entries.push(solver.workspaces.len());
            retained_bytes.push(solver.workspaces.values().map(bytes).sum::<usize>());
            if !unbounded_baseline {
                assert_eq!(solver.workspaces.len(), 1);
                assert_eq!(*retained_bytes.last().unwrap(), bytes(current));
            }
        }
        println!(
            "{{\"case\":\"cuda_solver_shape_changes\",\"entries\":{entries:?},\"retained_bytes\":{retained_bytes:?},\"current_bytes\":{current_bytes:?},\"samples_ms\":{samples:?},\"output_checksums\":{output_checksums:?}}}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod loop_checkpoint_tests {
    use super::*;
    use crate::tensor_ir::{TensorIr, LOOP_CHECKPOINT_TEST_FULL_TAPE};
    use std::time::Instant;

    fn plan(
        scan: bool,
        directional: bool,
        steps: usize,
        n: usize,
        lower: usize,
    ) -> Result<(TensorExecutionPlan, BTreeMap<String, DynamicTensor>), String> {
        let mut body = TensorIr::new();
        let carry = body.input_typed("carry", vec![n], TensorDType::F32)?;
        let index = body.input_typed("index", vec![], TensorDType::F32)?;
        let scale = body.input_typed("scale", vec![n], TensorDType::F32)?;
        let scaled = body.mul(carry, scale)?;
        let scaled = if lower == 3 {
            let delta = body.scalar_constant(0.00001);
            let delta = body.mul(index, delta)?;
            body.add(scaled, delta)?
        } else {
            scaled
        };
        let next = body.tanh(scaled)?;
        let mut graph = TensorIr::new();
        let initial = graph.input_typed("initial", vec![n], TensorDType::F32)?;
        let scale = graph.input_typed("scale", vec![n], TensorDType::F32)?;
        let captures = vec![("scale".to_string(), scale)];
        let loss = if scan {
            let body = body.compile_cpu_many(&[next, next])?.0;
            let scan_plan =
                TensorScanExecutionPlan::new(lower, lower + steps, body, "carry", "index")?;
            let (final_carry, outputs) = graph.scan(initial, scan_plan, captures)?;
            let output_sum = graph.sum(outputs)?;
            let final_sum = graph.sum(final_carry)?;
            graph.add(final_sum, output_sum)?
        } else {
            let loop_plan = TensorForiExecutionPlan::new(
                lower,
                lower + steps,
                body.compile_cpu(next)?,
                "carry",
                "index",
            )?;
            let output = graph.fori(initial, loop_plan, captures)?;
            graph.sum(output)?
        };
        let vjp = graph.symbolic_vjp(loss, "seed")?;
        let targets = [vjp.gradients["initial"], vjp.gradients["scale"]];
        let plan = if directional {
            let jvp = vjp.graph.symbolic_jvp_many_with_tangent_inputs(
                &targets,
                &BTreeMap::from([("scale".into(), "scale_tangent".into())]),
            )?;
            jvp.graph.compile_cpu_many(&jvp.tangents)?.0
        } else {
            vjp.graph.compile_cpu_many(&targets)?.0
        };
        let tensor = |value| DynamicTensor::with_dtype(vec![n], vec![value; n], TensorDType::F32);
        let mut inputs = BTreeMap::from([
            (
                "initial".into(),
                DynamicTensor::with_dtype(
                    vec![n],
                    (0..n).map(|i| ((i % 29) as f64 - 14.0) * 0.007).collect(),
                    TensorDType::F32,
                )?,
            ),
            (
                "scale".into(),
                DynamicTensor::with_dtype(
                    vec![n],
                    (0..n).map(|i| 0.97 + (i % 23) as f64 * 0.001).collect(),
                    TensorDType::F32,
                )?,
            ),
            (
                "seed".into(),
                DynamicTensor::with_dtype(vec![], vec![1.0], TensorDType::F32)?,
            ),
        ]);
        if directional {
            inputs.insert("scale_tangent".into(), tensor(0.02)?);
        }
        Ok((plan, inputs))
    }

    fn run(
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
        full: bool,
    ) -> Result<Vec<DynamicTensor>, String> {
        LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(full));
        let result = CudaBackend::new(0)
            .compile(plan.clone())
            .and_then(|plan| plan.execute_many(inputs));
        LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(false));
        result
    }

    #[test]
    fn checkpoint_vjp_and_directional_match_full_tape_on_device() -> Result<(), String> {
        if !crate::test_support::Gate::Cuda.enabled() {
            return Ok(());
        }
        for scan in [false, true] {
            for directional in [false, true] {
                for steps in [1, 8, 17, 64, 512] {
                    let (plan, inputs) = plan(scan, directional, steps, 257, 3)?;
                    if steps == 17 {
                        for output in &plan.output_node_ids {
                            let single = plan.as_ir().compile_cpu(*output)?;
                            let old = run(&single, &inputs, true)?;
                            let new = run(&single, &inputs, false)?;
                            assert!(
                                old[0]
                                    .data()
                                    .iter()
                                    .zip(new[0].data().iter())
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "single scan={scan} directional={directional}"
                            );
                        }
                    }
                    let old = run(&plan, &inputs, true)?;
                    let new = run(&plan, &inputs, false)?;
                    assert_eq!(old.len(), new.len());
                    for (old, new) in old.iter().zip(&new) {
                        assert_eq!(old.shape(), new.shape());
                        assert!(
                            old.data()
                                .iter()
                                .zip(new.data().iter())
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "scan={scan} directional={directional} steps={steps}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn checkpoint_device_bounds_and_input_errors_match_full_tape() -> Result<(), String> {
        if !crate::test_support::Gate::Cuda.enabled() {
            return Ok(());
        }
        for scan in [false, true] {
            for directional in [false, true] {
                let (plan, inputs) = plan(scan, directional, 17, 5, usize::MAX - 17)?;
                let old = run(&plan, &inputs, true)?;
                let new = run(&plan, &inputs, false)?;
                for (old, new) in old.iter().zip(&new) {
                    assert!(
                        old.data()
                            .iter()
                            .zip(new.data().iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "upper usizeMAX scan={scan} directional={directional}"
                    );
                }
                let mut missing = inputs.clone();
                missing.remove("scale");
                assert_eq!(
                    run(&plan, &missing, true).unwrap_err(),
                    run(&plan, &missing, false).unwrap_err()
                );
                let mut wrong_shape = inputs.clone();
                wrong_shape.insert(
                    "scale".into(),
                    DynamicTensor::with_dtype(vec![4], vec![0.99; 4], TensorDType::F32)?,
                );
                assert_eq!(
                    run(&plan, &wrong_shape, true).unwrap_err(),
                    run(&plan, &wrong_shape, false).unwrap_err()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn checkpoint_layout_preserves_broadcast_fallback() {
        let shape = [257];
        let captures = BTreeMap::from([("scale".into(), vec![257])]);
        assert_eq!(
            cuda_loop_checkpoint_block(3, 67, &shape, &captures, Some(&shape), true),
            Some(8)
        );
        assert_eq!(cuda_loop_tape_states(3, 67, Some(8)), Some(16));
        assert_eq!(
            cuda_loop_checkpoint_block(3, 67, &shape, &captures, Some(&shape), false),
            None
        );
        let (_, reverse) =
            cuda_loop_tape_source(usize::MAX - 64, usize::MAX, "count", "carry", None, Some(8));
        assert!(reverse.contains(&format!("if ({}ULL - block_start > 8ULL)", usize::MAX)));
        assert_eq!(
            cuda_loop_checkpoint_block(
                3,
                67,
                &shape,
                &BTreeMap::from([("scale".into(), vec![])]),
                None,
                true
            ),
            None
        );
        assert_eq!(
            cuda_loop_checkpoint_block(3, 67, &shape, &captures, Some(&[1, 257]), true),
            None
        );
    }

    #[test]
    fn profile_complete_checkpoint_vjp_on_device() -> Result<(), String> {
        if std::env::var_os("QUABLA_CUDA_CHECKPOINT_PROFILE").is_none() {
            return Ok(());
        }
        let n = 8192;
        for scan in [false, true] {
            for directional in [false, true] {
                for steps in [64, 512] {
                    let (plan, inputs) = plan(scan, directional, steps, n, 3)?;
                    let mut expected_bits: Option<Vec<Vec<u64>>> = None;
                    for full in if std::env::var_os("QUABLA_CHECKPOINT_FIRST").is_some() {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(full));
                        let device = CudaBackend::new(0).compile(plan.clone())?;
                        let warm = device.execute_many(&inputs)?;
                        let bits = warm
                            .iter()
                            .map(|output| {
                                output
                                    .data()
                                    .iter()
                                    .map(|value| value.to_bits())
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        if let Some(expected) = &expected_bits {
                            assert_eq!(
                                &bits, expected,
                                "profile scan={scan} directional={directional} steps={steps}"
                            );
                        } else {
                            expected_bits = Some(bits);
                        }
                        let mut times = Vec::new();
                        for _ in 0..7 {
                            let start = Instant::now();
                            let output = device.execute_many(&inputs)?;
                            times.push(start.elapsed().as_nanos());
                            std::hint::black_box(output);
                        }
                        times.sort_unstable();
                        let block = if full {
                            None
                        } else {
                            super::super::TensorCarryCheckpoints::<DynamicTensor>::block_size(steps)
                        };
                        let states =
                            cuda_loop_tape_states(3, 3 + steps, block).ok_or("invalid tape")?;
                        assert_eq!(
                            CUDA_LOOP_TEST_TAPE_BYTES.with(|value| value.get()),
                            states * n * 4 * if directional { 2 } else { 1 }
                        );
                        println!("{{\"scan\":{scan},\"directional\":{directional},\"steps\":{steps},\"full\":{full},\"tape_bytes\":{},\"median_ns\":{}}}",CUDA_LOOP_TEST_TAPE_BYTES.with(|value| value.get()),times[3]);
                        LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(false));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod host_loop_graph_tests {
    use super::host_loop::{
        CUDA_LOOP_GRAPHS_DISABLED, CUDA_LOOP_GRAPH_LAUNCHES, CUDA_LOOP_GRAPH_RECORDING_ERRORS,
    };
    use super::*;
    use crate::tensor_ir::{TensorComparison, TensorIr, TensorWhileExecutionPlan};

    const LANES: usize = 6;

    /// `roll(c) * scale + rate * sum(c)`, plus `reshape(c, [2, 3]) @ weight`
    /// with `matmul` (a cuBLAS product): the roll and the reduction make a loop
    /// body host-driven.
    fn rotate(
        body: &mut TensorIr,
        carry: TensorNodeId,
        matmul: bool,
    ) -> Result<TensorNodeId, String> {
        let scale = body.input("scale", vec![LANES])?;
        let rate = body.input("rate", vec![])?;
        let head = body.slice_axis(carry, 0, 0, 1)?;
        let tail = body.slice_axis(carry, 0, 1, LANES)?;
        let rolled = body.concat(vec![tail, head], 0)?;
        let scaled = body.mul(rolled, scale)?;
        let total = body.sum(carry)?;
        let drift = body.mul(total, rate)?;
        let next = body.add(scaled, drift)?;
        if !matmul {
            return Ok(next);
        }
        let weight = body.input("weight", vec![3, 3])?;
        let matrix = body.reshape(carry, vec![2, 3])?;
        let mixed = body.matmul(matrix, weight)?;
        let mixed = body.reshape(mixed, vec![LANES])?;
        body.add(next, mixed)
    }

    /// `tanh(rotate(c) + 0.001 * i)`.
    fn rotate_step(body: &mut TensorIr, matmul: bool) -> Result<TensorNodeId, String> {
        let carry = body.input("carry", vec![LANES])?;
        let index = body.input("index", vec![])?;
        let next = rotate(body, carry, matmul)?;
        let step = body.scalar_constant(0.001);
        let step = body.mul(index, step)?;
        let next = body.add(next, step)?;
        body.tanh(next)
    }

    fn square_sum(graph: &mut TensorIr, value: TensorNodeId) -> Result<TensorNodeId, String> {
        let squared = graph.mul(value, value)?;
        graph.sum(squared)
    }

    /// Loss of a Fori or Scan loop of `steps` iterations over the graph inputs
    /// `initial`, `scale`, `rate` and, with `matmul`, `weight`.
    fn loop_loss(
        scan: bool,
        matmul: bool,
        steps: usize,
    ) -> Result<(TensorIr, TensorNodeId), String> {
        let mut body = TensorIr::new();
        let next = rotate_step(&mut body, matmul)?;
        let mut graph = TensorIr::new();
        let initial = graph.input("initial", vec![LANES])?;
        let scale = graph.input("scale", vec![LANES])?;
        let rate = graph.input("rate", vec![])?;
        let mut captures = vec![("scale".to_string(), scale), ("rate".to_string(), rate)];
        if matmul {
            captures.push(("weight".to_string(), graph.input("weight", vec![3, 3])?));
        }
        let loss = if scan {
            // A two-lane step output, so carry and output lanes differ.
            let head = body.slice_axis(next, 0, 0, 2)?;
            let tail = body.slice_axis(next, 0, 1, 3)?;
            let output = body.mul(head, tail)?;
            let plan = TensorScanExecutionPlan::new(
                2,
                2 + steps,
                body.compile_cpu_many(&[next, output])?.0,
                "carry",
                "index",
            )?;
            let (carry, outputs) = graph.scan(initial, plan, captures)?;
            let carry = square_sum(&mut graph, carry)?;
            let outputs = square_sum(&mut graph, outputs)?;
            graph.add(carry, outputs)?
        } else {
            let plan = TensorForiExecutionPlan::new(
                2,
                2 + steps,
                body.compile_cpu(next)?,
                "carry",
                "index",
            )?;
            let output = graph.fori(initial, plan, captures)?;
            square_sum(&mut graph, output)?
        };
        Ok((graph, loss))
    }

    /// `[counter, carry...]` stepped by the rotation while `counter < limit`.
    fn while_output() -> Result<(TensorIr, TensorNodeId), String> {
        let mut predicate = TensorIr::new();
        let carry = predicate.input("carry", vec![LANES + 1])?;
        let limit = predicate.input("limit", vec![])?;
        let counter = predicate.slice_axis(carry, 0, 0, 1)?;
        let counter = predicate.reshape(counter, vec![])?;
        let keep_going = predicate.compare(counter, limit, TensorComparison::Less)?;
        let mut body = TensorIr::new();
        let carry = body.input("carry", vec![LANES + 1])?;
        let counter = body.slice_axis(carry, 0, 0, 1)?;
        let one = body.scalar_constant(1.0);
        let counter = body.add(counter, one)?;
        let state = body.slice_axis(carry, 0, 1, LANES + 1)?;
        let state = rotate(&mut body, state, false)?;
        let state = body.tanh(state)?;
        let next = body.concat(vec![counter, state], 0)?;
        let plan = TensorWhileExecutionPlan::new(
            predicate.compile_cpu(keep_going)?,
            body.compile_cpu(next)?,
            "carry",
        )?;
        let mut graph = TensorIr::new();
        let initial = graph.input("counted", vec![LANES + 1])?;
        let mut captures = Vec::new();
        for (name, shape) in [("scale", vec![LANES]), ("rate", vec![]), ("limit", vec![])] {
            captures.push((name.to_string(), graph.input(name, shape)?));
        }
        let output = graph.while_loop(initial, plan, captures)?;
        Ok((graph, output))
    }

    fn tangents(matmul: bool) -> BTreeMap<String, String> {
        ["initial", "scale", "rate", "weight"]
            .into_iter()
            .take(if matmul { 4 } else { 3 })
            .map(|name| (name.to_string(), format!("{name}_tangent")))
            .collect()
    }

    /// Programs of every host-driven loop kind, with and without a cuBLAS
    /// product in the body: the loss (`Fori`/`Scan`), its forward derivative
    /// (`ForiJvp`/a packed `Scan`), its gradient (`ForiVjp`/`ScanVjp`), the
    /// Hessian-vector product (`ForiVjpJvp`/`ScanVjpJvp`), and a `While` loop
    /// with its forward derivative.
    fn programs(steps: usize) -> Result<Vec<(String, TensorExecutionPlan)>, String> {
        let mut programs = Vec::new();
        for (scan, matmul) in [(false, false), (true, false), (false, true), (true, true)] {
            let kind = match (scan, matmul) {
                (false, false) => "fori",
                (true, false) => "scan",
                (false, true) => "fori matmul",
                (true, true) => "scan matmul",
            };
            let tangents = tangents(matmul);
            let (graph, loss) = loop_loss(scan, matmul, steps)?;
            programs.push((format!("{kind} value"), graph.compile_cpu(loss)?));
            let forward = graph.symbolic_jvp_with_tangent_inputs(loss, &tangents)?;
            programs.push((
                format!("{kind} jvp"),
                forward.graph.compile_cpu(forward.tangent)?,
            ));
            let vjp = graph.symbolic_vjp(loss, "seed")?;
            let gradients = tangents
                .keys()
                .map(|name| vjp.gradients[name])
                .collect::<Vec<_>>();
            programs.push((
                format!("{kind} vjp"),
                vjp.graph.compile_cpu_many(&gradients)?.0,
            ));
            let hvp = vjp
                .graph
                .symbolic_jvp_many_with_tangent_inputs(&gradients, &tangents)?;
            programs.push((
                format!("{kind} hvp"),
                hvp.graph.compile_cpu_many(&hvp.tangents)?.0,
            ));
        }
        let (graph, output) = while_output()?;
        programs.push(("while value".to_string(), graph.compile_cpu(output)?));
        let forward = graph.symbolic_jvp_with_tangent_inputs(
            output,
            &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
        )?;
        programs.push((
            "while jvp".to_string(),
            forward.graph.compile_cpu(forward.tangent)?,
        ));
        Ok(programs)
    }

    fn inputs(steps: usize) -> Result<BTreeMap<String, DynamicTensor>, String> {
        let lanes = |values: Vec<f64>| DynamicTensor::new(vec![LANES], values);
        let scalar = |value: f64| DynamicTensor::new(vec![], vec![value]);
        let initial = (0..LANES)
            .map(|i| (i as f64 - 2.5) * 0.3)
            .collect::<Vec<_>>();
        let mut counted = vec![0.0];
        counted.extend(&initial);
        Ok(BTreeMap::from([
            ("initial".to_string(), lanes(initial)?),
            (
                "counted".to_string(),
                DynamicTensor::new(vec![LANES + 1], counted)?,
            ),
            (
                "scale".to_string(),
                lanes((0..LANES).map(|i| 0.9 + i as f64 * 0.04).collect())?,
            ),
            ("rate".to_string(), scalar(-0.07)?),
            ("limit".to_string(), scalar(steps as f64)?),
            ("seed".to_string(), scalar(1.0)?),
            (
                "initial_tangent".to_string(),
                lanes((0..LANES).map(|i| 0.5 - i as f64 * 0.2).collect())?,
            ),
            (
                "scale_tangent".to_string(),
                lanes((0..LANES).map(|i| (i % 3) as f64 - 1.0).collect())?,
            ),
            ("rate_tangent".to_string(), scalar(0.75)?),
            (
                "weight".to_string(),
                DynamicTensor::new(
                    vec![3, 3],
                    (0..9).map(|i| ((i * 7) % 9) as f64 * 0.05 - 0.2).collect(),
                )?,
            ),
            (
                "weight_tangent".to_string(),
                DynamicTensor::new(vec![3, 3], (0..9).map(|i| 0.1 * i as f64).collect())?,
            ),
        ]))
    }

    fn bits(outputs: &[DynamicTensor]) -> Vec<Vec<u64>> {
        outputs
            .iter()
            .map(|output| output.data().iter().map(|value| value.to_bits()).collect())
            .collect()
    }

    /// Runs `program` with graphs disabled and then enabled, each compiled
    /// once and executed twice (the second execution reuses the regions'
    /// value tables), and returns both results and the enabled run's graph
    /// launches.
    fn run_both<P>(
        program: &TensorExecutionPlan,
        compile: impl Fn(TensorExecutionPlan) -> Result<P, String>,
        execute: impl Fn(&P) -> Result<Vec<DynamicTensor>, String>,
    ) -> Result<(Vec<DynamicTensor>, Vec<DynamicTensor>, usize), String> {
        let mut results = Vec::new();
        for disabled in [true, false] {
            CUDA_LOOP_GRAPHS_DISABLED.with(|value| value.set(disabled));
            CUDA_LOOP_GRAPH_LAUNCHES.with(|value| value.set(0));
            CUDA_LOOP_GRAPH_RECORDING_ERRORS.with(|errors| errors.borrow_mut().clear());
            let result = compile(program.clone()).and_then(|plan| {
                execute(&plan)?;
                execute(&plan)
            });
            CUDA_LOOP_GRAPHS_DISABLED.with(|value| value.set(false));
            results.push(result?);
        }
        let launches = CUDA_LOOP_GRAPH_LAUNCHES.with(|value| value.get());
        let enabled = results.pop().expect("two runs");
        let eager = results.pop().expect("two runs");
        Ok((eager, enabled, launches))
    }

    /// The errors of the region recordings that failed on this thread since
    /// the last call.
    fn recording_errors() -> Vec<String> {
        CUDA_LOOP_GRAPH_RECORDING_ERRORS.with(|errors| std::mem::take(&mut *errors.borrow_mut()))
    }

    fn assert_close(name: &str, actual: &[DynamicTensor], expected: &[DynamicTensor], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.shape(), expected.shape(), "{name}");
            for (a, e) in actual.data().iter().zip(expected.data().iter()) {
                assert!(
                    (a - e).abs() <= tol * e.abs().max(1.0),
                    "{name}: {a} vs {e} (tolerance {tol})"
                );
            }
        }
    }

    /// Graph replays launch the eager per-node sequence unchanged, so every
    /// host-driven loop kind gives bit-identical results with and without
    /// them, in both precisions; both also match the CPU reference.
    #[test]
    fn host_loop_graph_replays_match_eager_runs_bit_for_bit() -> Result<(), String> {
        if !crate::test_support::Gate::Cuda.enabled() {
            return Ok(());
        }
        // One step never records; three record at the last step; 17 and 64
        // replay, 64 through checkpoint blocks in the reverse passes.
        for steps in [1, 2, 3, 17, 64] {
            let inputs = inputs(steps)?;
            for (name, program) in programs(steps)? {
                let cpu = program
                    .output_node_ids()
                    .iter()
                    .map(|output| program.as_ir().compile_cpu(*output)?.evaluate(&inputs))
                    .collect::<Result<Vec<_>, String>>()?;
                let backend = CudaBackend::new(0);
                let (eager, graph, launches) = run_both(
                    &program,
                    |plan| backend.compile(plan),
                    |plan| plan.execute_many(&inputs),
                )?;
                assert_eq!(bits(&eager), bits(&graph), "{name} float32 steps={steps}");
                assert_close(&format!("{name} float32 steps={steps}"), &graph, &cpu, 1e-4);
                let (eager, graph, launches64) = run_both(
                    &program,
                    |plan| backend.compile_float64(plan),
                    |plan| plan.execute_many(&inputs),
                )?;
                assert_eq!(bits(&eager), bits(&graph), "{name} float64 steps={steps}");
                assert_close(
                    &format!("{name} float64 steps={steps}"),
                    &graph,
                    &cpu,
                    1e-11,
                );
                let errors = recording_errors();
                assert!(errors.is_empty(), "{name} steps={steps}: {errors:?}");
                if steps >= 17 {
                    assert!(launches > 0 && launches64 > 0, "{name} replayed no graph");
                } else if steps == 1 {
                    // At most two runs per region (a `While` predicate runs twice).
                    assert_eq!((launches, launches64), (0, 0), "{name} steps={steps}");
                }
            }
        }
        Ok(())
    }

    /// Every elementwise math kind runs as one NVRTC kernel of the per-node
    /// program, so a host-driven loop body that uses it records and replays
    /// CUDA graphs, bit-identical to the eager launches.
    #[test]
    fn host_loop_graphs_admit_every_elementwise_math_kind() -> Result<(), String> {
        if !crate::test_support::Gate::Cuda.enabled() {
            return Ok(());
        }
        let steps = 17;
        let inputs = inputs(steps)?;
        let backend = CudaBackend::new(0);
        let kinds = UnaryMathKind::ALL
            .into_iter()
            .map(|kind| (kind.name(), Some(kind), None))
            .chain(
                crate::tensor_ir::BinaryMathKind::ALL
                    .into_iter()
                    .map(|kind| (kind.name(), None, Some(kind))),
            );
        for (name, unary, binary) in kinds {
            // `rotate(c) + 0.25 kind(0.25 c + 0.5) + 0.001 i`, with the
            // argument of `arccosh` moved into its domain.
            let mut body = TensorIr::new();
            let carry = body.input("carry", vec![LANES])?;
            let index = body.input("index", vec![])?;
            let rotated = rotate(&mut body, carry, false)?;
            let quarter = body.scalar_constant(0.25);
            let half = body.scalar_constant(0.5);
            let scaled = body.mul(carry, quarter)?;
            let mut argument = body.add(scaled, half)?;
            let value = match (unary, binary) {
                (Some(kind), _) => {
                    if kind == UnaryMathKind::Arccosh {
                        let one = body.scalar_constant(1.0);
                        argument = body.add(argument, one)?;
                    }
                    body.unary_math(argument, kind)?
                }
                (_, Some(kind)) => {
                    let operand = body.scalar_constant(0.75);
                    body.binary_math(argument, operand, kind)?
                }
                _ => unreachable!("every entry has a kind"),
            };
            let term = body.mul(value, quarter)?;
            let next = body.add(rotated, term)?;
            let step = body.scalar_constant(0.001);
            let step = body.mul(index, step)?;
            let next = body.add(next, step)?;
            let plan =
                TensorForiExecutionPlan::new(0, steps, body.compile_cpu(next)?, "carry", "index")?;
            let mut graph = TensorIr::new();
            let initial = graph.input("initial", vec![LANES])?;
            let captures = vec![
                ("scale".to_string(), graph.input("scale", vec![LANES])?),
                ("rate".to_string(), graph.input("rate", vec![])?),
            ];
            let output = graph.fori(initial, plan, captures)?;
            let program = graph.compile_cpu(output)?;
            let (eager, replayed, launches) = run_both(
                &program,
                |plan| backend.compile(plan),
                |plan| plan.execute_many(&inputs),
            )?;
            assert_eq!(bits(&eager), bits(&replayed), "{name}");
            let errors = recording_errors();
            assert!(errors.is_empty(), "{name}: {errors:?}");
            assert!(launches > 0, "{name} replayed no graph");
        }
        Ok(())
    }

    /// Destroying a cuBLAS or cuSOLVER handle synchronizes the device, which
    /// invalidates a stream capture in progress anywhere in the context, and
    /// every plan of the process shares the device's primary context. One
    /// thread records the region of a host-driven loop (with a cuBLAS product)
    /// on every execution while another creates and drops library handles, as
    /// plans compiled and dropped by other threads do; every recording must
    /// succeed and replay bit-identically to the eager runs.
    #[test]
    fn host_loop_graph_recordings_survive_concurrent_library_handle_drops() -> Result<(), String> {
        if !crate::test_support::Gate::Cuda.enabled() {
            return Ok(());
        }
        const EXECUTIONS: usize = 40;
        let steps = 17;
        let inputs = inputs(steps)?;
        let (graph, loss) = loop_loss(false, true, steps)?;
        let program = graph.compile_cpu(loss)?;
        let backend = CudaBackend::new(0);
        CUDA_LOOP_GRAPHS_DISABLED.with(|value| value.set(true));
        let eager = backend
            .compile(program.clone())
            .and_then(|plan| plan.execute_many(&inputs));
        CUDA_LOOP_GRAPHS_DISABLED.with(|value| value.set(false));
        let eager = eager?;
        let plan = backend.compile(program)?;
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let churn = scope.spawn(|| -> Result<usize, String> {
                let context = CudaContext::new(0)
                    .map_err(|error| format!("failed to create CUDA context: {error:?}"))?;
                let mut drops = 0;
                while !done.load(Ordering::Relaxed) {
                    let blas = cuda_blas(context.default_stream())?;
                    let solver = cuda_solver::<f32>(context.default_stream())?;
                    drop((blas, solver));
                    drops += 1;
                }
                Ok(drops)
            });
            // Failures are returned rather than asserted here, so that the churn thread is
            // always stopped and joined.
            let recorded = (|| -> Result<(), String> {
                for execution in 0..EXECUTIONS {
                    CUDA_LOOP_GRAPH_LAUNCHES.with(|value| value.set(0));
                    let replayed = plan.execute_many(&inputs)?;
                    let errors = recording_errors();
                    if !errors.is_empty() {
                        return Err(format!(
                            "execution {execution} recorded no graph: {errors:?}"
                        ));
                    }
                    if CUDA_LOOP_GRAPH_LAUNCHES.with(|value| value.get()) == 0 {
                        return Err(format!("execution {execution} replayed no graph"));
                    }
                    if bits(&eager) != bits(&replayed) {
                        return Err(format!("execution {execution} differs from the eager runs"));
                    }
                }
                Ok(())
            })();
            done.store(true, Ordering::Relaxed);
            let drops = churn
                .join()
                .map_err(|_| "the handle churn thread panicked".to_string())??;
            recorded?;
            assert!(drops > 0, "no handle was dropped during the recordings");
            Ok(())
        })
    }
}

#[cfg(test)]
mod region_batching_tests {
    //! Loops batched by `vmap` (`region_batching.rs`) on the device: an
    //! elementwise body stays eligible for the fused per-lane loop kernels
    //! with its larger `[B, n]` carry, also when an unmapped `[n]` capture
    //! broadcasts against it; the checkpointed reverse passes replay the
    //! batched carries bit for bit like the complete tape; and a body that is
    //! not elementwise runs host-driven.
    use super::*;
    use crate::tensor_ir::{SymbolicCotangent, TensorIr, LOOP_CHECKPOINT_TEST_FULL_TAPE};

    const BATCH: usize = 3;
    const LANES: usize = 33;

    /// A loss over a fori loop and a scan whose bodies are elementwise, or
    /// rotate the carry when `rotate` (not lane-local, so host-driven).
    fn loss_graph(steps: usize, rotate: bool) -> Result<(TensorIr, TensorNodeId), String> {
        let mut body = TensorIr::new();
        let carry = body.input_typed("carry", vec![LANES], TensorDType::F32)?;
        let index = body.input_typed("index", vec![], TensorDType::F32)?;
        let scale = body.input_typed("scale", vec![LANES], TensorDType::F32)?;
        let carry_term = if rotate {
            let head = body.slice_axis(carry, 0, 0, 1)?;
            let tail = body.slice_axis(carry, 0, 1, LANES)?;
            body.concat(vec![tail, head], 0)?
        } else {
            carry
        };
        let scaled = body.mul(carry_term, scale)?;
        let delta = body.scalar_constant(0.001);
        let delta = body.mul(index, delta)?;
        let shifted = body.add(scaled, delta)?;
        let next = body.tanh(shifted)?;
        let fori_plan =
            TensorForiExecutionPlan::new(0, steps, body.compile_cpu(next)?, "carry", "index")?;
        let scan_plan = TensorScanExecutionPlan::new(
            0,
            steps,
            body.compile_cpu_many(&[next, next])?.0,
            "carry",
            "index",
        )?;
        let mut graph = TensorIr::new();
        let initial = graph.input_typed("initial", vec![LANES], TensorDType::F32)?;
        let scale = graph.input_typed("scale", vec![LANES], TensorDType::F32)?;
        let looped = graph.fori(initial, fori_plan, vec![("scale".to_string(), scale)])?;
        let (final_carry, outputs) =
            graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)])?;
        let mut total = graph.sum(looped)?;
        for value in [final_carry, outputs] {
            let squared = graph.mul(value, value)?;
            let sum = graph.sum(squared)?;
            total = graph.add(total, sum)?;
        }
        Ok((graph, total))
    }

    /// The batched gradients and their directional derivatives (forward over
    /// reverse), with `initial` and the tangents mapped and `scale` mapped
    /// when `scale_mapped`.
    fn batched_plan(
        steps: usize,
        rotate: bool,
        scale_mapped: bool,
    ) -> Result<(TensorExecutionPlan, BTreeMap<String, DynamicTensor>), String> {
        let (graph, total) = loss_graph(steps, rotate)?;
        let reverse = graph.symbolic_vjp_many(&[(total, SymbolicCotangent::Ones)])?;
        let gradients = [reverse.gradients["initial"], reverse.gradients["scale"]];
        let tangents = BTreeMap::from([
            ("initial".to_string(), "initial_tangent".to_string()),
            ("scale".to_string(), "scale_tangent".to_string()),
        ]);
        let hvp = reverse
            .graph
            .symbolic_jvp_many_with_tangent_inputs(&gradients, &tangents)?;
        let mut outputs = hvp.values.clone();
        outputs.extend(&hvp.tangents);
        let mut batched = TensorIr::new();
        let mut bindings = BTreeMap::new();
        let mut inputs = BTreeMap::new();
        for (salt, name) in ["initial", "scale", "initial_tangent", "scale_tangent"]
            .into_iter()
            .enumerate()
        {
            let mapped = name != "scale" || scale_mapped;
            let shape = if mapped {
                vec![BATCH, LANES]
            } else {
                vec![LANES]
            };
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|i| match name {
                    "scale" => 0.9 + ((i * 7 + salt) % 23) as f64 * 0.005,
                    _ => ((i * 5 + salt * 3) % 29) as f64 * 0.01 - 0.14,
                })
                .collect();
            let node = batched.input_typed(name, shape.clone(), TensorDType::F32)?;
            bindings.insert(name.to_string(), (node, mapped));
            inputs.insert(
                name.to_string(),
                DynamicTensor::with_dtype(shape, values, TensorDType::F32)?,
            );
        }
        let results = batched
            .inline_batched(&hvp.graph, &bindings, BATCH, &outputs)
            .map_err(|error| error.to_string())?;
        let ids = results.iter().map(|(node, _)| *node).collect::<Vec<_>>();
        Ok((batched.compile_cpu_many(&ids)?.0, inputs))
    }

    fn run(
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
        full_tape: bool,
    ) -> Result<Vec<DynamicTensor>, String> {
        LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(full_tape));
        let result = CudaBackend::new(0)
            .compile(plan.clone())
            .and_then(|plan| plan.execute_many(inputs));
        LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(false));
        result
    }

    #[test]
    fn batched_loops_fuse_checkpoint_and_match_the_cpu_on_device() -> Result<(), String> {
        if !crate::test_support::Gate::Cuda.enabled() {
            return Ok(());
        }
        for rotate in [false, true] {
            for scale_mapped in [false, true] {
                // 17 and 64 steps take the checkpointed reverse pass.
                for steps in [3, 17, 64] {
                    let label =
                        format!("rotate={rotate} scale_mapped={scale_mapped} steps={steps}");
                    let (plan, inputs) = batched_plan(steps, rotate, scale_mapped)?;
                    let loops = plan
                        .nodes
                        .iter()
                        .filter(|node| {
                            matches!(
                                node.op,
                                TensorOp::Region(RegionNode {
                                    kind: RegionKind::ForiVjp { .. },
                                    ..
                                }) | TensorOp::Region(RegionNode {
                                    kind: RegionKind::ForiVjpJvp { .. },
                                    ..
                                }) | TensorOp::Region(RegionNode {
                                    kind: RegionKind::ScanVjp { .. },
                                    ..
                                }) | TensorOp::Region(RegionNode {
                                    kind: RegionKind::ScanVjpJvp { .. },
                                    ..
                                })
                            )
                        })
                        .count();
                    assert!(loops >= 4, "{label}: the batched plan keeps its loop nodes");
                    let host_driven = host_loop::cuda_host_driven_nodes(&plan);
                    let reasons = host_driven
                        .iter()
                        .map(|id| {
                            (
                                cuda_op_name(&plan.nodes[*id].op),
                                host_loop::cuda_loop_fused_error(&plan, *id),
                            )
                        })
                        .collect::<Vec<_>>();
                    // The forward-mode Scan inside the directional derivative
                    // packs its carry as `[primal, tangent]`; batched, that is
                    // `[B, 2, n]`, which the packed-pair kernel (pair axis
                    // leading) does not lower, so it alone runs host-driven.
                    let batched_packed_scan = |id: &TensorNodeId| {
                        matches!(&plan.nodes[*id].op, TensorOp::Region(RegionNode { kind: RegionKind::Scan { scan_plan, .. }, .. })
                            if cuda_scan_uses_packed_halves(scan_plan)
                                && scan_plan.carry_shape().is_ok_and(|shape| shape.get(1) == Some(&2)))
                    };
                    if rotate {
                        assert!(
                            host_driven.iter().any(|id| !batched_packed_scan(id)),
                            "{label}"
                        );
                    } else {
                        assert!(
                            host_driven.iter().all(batched_packed_scan),
                            "{label}: {reasons:?}"
                        );
                    }
                    let cpu = plan.evaluate_many(&inputs)?;
                    let checkpointed = run(&plan, &inputs, false)?;
                    let full = run(&plan, &inputs, true)?;
                    assert_eq!(checkpointed.len(), cpu.len(), "{label}");
                    for ((device, tape), cpu) in checkpointed.iter().zip(&full).zip(&cpu) {
                        assert_eq!(device.shape(), cpu.shape(), "{label}");
                        assert!(
                            device
                                .data()
                                .iter()
                                .zip(tape.data().iter())
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{label}: checkpointed and full-tape results differ"
                        );
                        for (device, cpu) in device.data().iter().zip(cpu.data().iter()) {
                            assert!(
                                (device - cpu).abs() <= 1e-4 * (1.0 + cpu.abs()),
                                "{label}: {device} != {cpu}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
