use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

mod cholesky;
#[cfg(test)]
mod cholesky_tests;
pub use cholesky::CholeskyAdKind;
mod linalg;
pub use linalg::{evaluate_eager as evaluate_linalg, LinalgKind};
mod custom;
pub use custom::{TensorCustomRule, TensorCustomTangent};
mod device_optimizer;
pub use device_optimizer::{DeviceOptimizerConfig, DeviceUpdateRule};
mod elementwise;
pub use elementwise::{BinaryMathKind, UnaryMathKind};
mod extremum;
pub use extremum::TensorExtremum;
mod host_storage;
pub use host_storage::HostTensorStorage;
mod region_batching;
#[cfg(test)]
mod region_batching_tests;

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda;

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use cuda::{
    CudaBackend, CudaDataParallelExecutionPlan, CudaDataParallelResult, CudaDataParallelTiming,
    CudaExecutionPlan, CudaReal,
};

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use mlx::{MlxAdamPlan, MlxBackend, MlxRetainedInputs};

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxBackend;

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
#[derive(Clone, Debug, Default)]
pub struct MlxRetainedInputs;

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
impl MlxRetainedInputs {
    pub fn empty() -> Self {
        Self
    }

    pub fn replace(&mut self, _name: String, _value: &DynamicTensor) -> Result<(), String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn clear(&mut self) {}

    pub fn names(&self) -> impl Iterator<Item = &str> {
        std::iter::empty()
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
#[derive(Clone, Debug, Default)]
pub struct MlxAdamPlan;

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
impl MlxBackend {
    pub fn execute_many(
        &self,
        _plan: &TensorExecutionPlan,
        _output_node_ids: &[TensorNodeId],
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn execute_without_output(
        &self,
        _plan: &TensorExecutionPlan,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(), String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn execute_without_output_with_state(
        &self,
        _plan: &TensorExecutionPlan,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _state: &MlxRetainedInputs,
    ) -> Result<(), String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn execute_many_with_state(
        &self,
        _plan: &TensorExecutionPlan,
        _output_node_ids: &[TensorNodeId],
        _inputs: &BTreeMap<String, DynamicTensor>,
        _state: &MlxRetainedInputs,
    ) -> Result<Vec<DynamicTensor>, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
impl MlxAdamPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _plan: TensorExecutionPlan,
        _loss_node_id: usize,
        _gradient_node_ids: BTreeMap<String, usize>,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _retained_input_names: impl IntoIterator<Item = String>,
        _learning_rate: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
    ) -> Result<Self, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn with_config(
        _plan: TensorExecutionPlan,
        _loss_node_id: usize,
        _gradient_node_ids: BTreeMap<String, usize>,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _retained_input_names: impl IntoIterator<Item = String>,
        _config: DeviceOptimizerConfig,
    ) -> Result<Self, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    // `new` always fails without MLX, so no stub plan exists to report a rate.
    pub fn learning_rate(&self) -> f32 {
        0.0
    }

    pub fn set_learning_rate(&mut self, _learning_rate: f32) -> Result<(), String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn step(&mut self, _inputs: &BTreeMap<String, DynamicTensor>) -> Result<(), String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn loss(&self, _inputs: &BTreeMap<String, DynamicTensor>) -> Result<DynamicTensor, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }

    pub fn parameter(&self, _name: &str) -> Result<DynamicTensor, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }
}

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
impl TensorBackend for MlxBackend {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn execute(
        &self,
        _plan: &TensorExecutionPlan,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        Err("MLX backend is unavailable: build Quabla on macOS with --features mlx".to_string())
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend {
    device_ordinal: usize,
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
/// `T` is the device element type of the CUDA build's plan (`f32`, or `f64`
/// for the `precision="float64"` lowering); this build never constructs one.
#[derive(Clone, Debug)]
pub struct CudaExecutionPlan<T = f32> {
    plan: TensorExecutionPlan,
    device_ordinal: usize,
    element: std::marker::PhantomData<T>,
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Clone, Debug, Default)]
pub struct CudaDataParallelExecutionPlan;

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaDataParallelTiming {
    pub replica_enqueue: std::time::Duration,
    pub collective: std::time::Duration,
    pub output_readback: std::time::Duration,
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Clone, Debug, Default)]
pub struct CudaDataParallelResult {
    pub outputs: Vec<DynamicTensor>,
    pub timing: CudaDataParallelTiming,
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl CudaBackend {
    pub fn new(device_ordinal: usize) -> Self {
        Self { device_ordinal }
    }

    pub fn compile(&self, plan: TensorExecutionPlan) -> Result<CudaExecutionPlan, String> {
        let _ = plan;
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn compile_float64(
        &self,
        plan: TensorExecutionPlan,
    ) -> Result<CudaExecutionPlan<f64>, String> {
        let _ = plan;
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn compile_data_parallel(
        &self,
        _plan: TensorExecutionPlan,
        _device_ordinals: Vec<usize>,
    ) -> Result<CudaDataParallelExecutionPlan, String> {
        Err(format!(
            "CUDA data-parallel backend is unavailable for device {}: build Quabla on Linux with --features cuda-nccl",
            self.device_ordinal
        ))
    }

    pub fn compile_data_parallel_sharded(
        &self,
        _plan: TensorExecutionPlan,
        _sharding: &TensorShardingPlan,
        _device_ordinals: Vec<usize>,
    ) -> Result<CudaDataParallelExecutionPlan, String> {
        Err(format!(
            "CUDA data-parallel backend is unavailable for device {}: build Quabla on Linux with --features cuda-nccl",
            self.device_ordinal
        ))
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl CudaDataParallelExecutionPlan {
    pub fn replica_count(&self) -> usize {
        0
    }

    pub fn device_ordinals(&self) -> Vec<usize> {
        Vec::new()
    }

    pub fn output_node_ids(&self) -> &[TensorNodeId] {
        &[]
    }

    pub fn execute_replicas(
        &self,
        _replica_inputs: &[BTreeMap<String, DynamicTensor>],
        _reduction: TensorReplicaReduction,
    ) -> Result<CudaDataParallelResult, String> {
        Err("CUDA data-parallel backend is unavailable: build Quabla on Linux with --features cuda-nccl".to_string())
    }

    pub fn execute_sharded(
        &self,
        _replica_inputs: &[BTreeMap<String, DynamicTensor>],
    ) -> Result<CudaDataParallelResult, String> {
        Err("CUDA data-parallel backend is unavailable: build Quabla on Linux with --features cuda-nccl".to_string())
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl<T> CudaExecutionPlan<T> {
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
        false
    }

    pub fn uses_fused_matmul_bias_tanh(&self) -> bool {
        false
    }

    pub fn fused_region_count(&self) -> usize {
        0
    }

    pub fn device_buffer_count(&self) -> Result<usize, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn synchronize(&self) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn execute(
        &self,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn execute_retaining(
        &self,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _retained_inputs: &BTreeSet<String>,
    ) -> Result<DynamicTensor, String> {
        self.execute(_inputs)
    }

    pub fn execute_many(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        self.execute(inputs).map(|value| vec![value])
    }

    pub fn execute_primary_retaining(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        retained_inputs: &BTreeSet<String>,
    ) -> Result<DynamicTensor, String> {
        self.execute_retaining(inputs, retained_inputs)
    }

    pub fn sgd_step_input_from_output(
        &self,
        _parameter_name: &str,
        _learning_rate: f32,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn execute_retaining_without_output(
        &self,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _retained_inputs: &BTreeSet<String>,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn adam_step_input_from_output(
        &self,
        _parameter_name: &str,
        _learning_rate: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn adam_step_input_from_node(
        &self,
        _parameter_name: &str,
        _gradient_node_id: usize,
        _learning_rate: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn optimizer_step(
        &self,
        _gradients: &BTreeMap<String, TensorNodeId>,
        _config: &DeviceOptimizerConfig,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn nonfinite_clip_seen(&self) -> Result<bool, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn retained_input_to_host(&self, _name: &str) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn computed_node_to_host(&self, _node_id: usize) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn sync_retained_input_to(
        &self,
        _source_name: &str,
        _target: &Self,
        _target_name: &str,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl TensorBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn execute(
        &self,
        _plan: &TensorExecutionPlan,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Quabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }
}

pub type TensorNodeId = usize;

/// Immutable host tensor with storage matching its logical element type.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicTensor {
    shape: Vec<usize>,
    data: HostTensorStorage,
    dtype: TensorDType,
}

/// The value of a [`TensorOp::Constant`]: an array baked into a graph, such
/// as an eager array that a traced function captured.
///
/// The data is shared, so cloning a graph, a plan, or a region never copies
/// the elements, and `Debug` prints only the shape and dtype because captured
/// arrays can be large. Plan compilation deduplicates constants by value
/// (bitwise, see [`Self::same_bits`]), not by identity.
#[derive(Clone)]
struct TensorConstant(Arc<TensorConstantData>);

struct TensorConstantData {
    value: DynamicTensor,
    /// The MLX copy of `value`, created by the first MLX execution of any plan
    /// holding this constant and reused by every later one. It is only
    /// accessed while the MLX execution lock is held.
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    mlx_array: std::sync::Mutex<Option<mlx_rs::Array>>,
}

impl TensorConstant {
    fn new(value: DynamicTensor) -> Self {
        Self(Arc::new(TensorConstantData {
            value,
            #[cfg(all(feature = "mlx", target_os = "macos"))]
            mlx_array: std::sync::Mutex::new(None),
        }))
    }

    fn value(&self) -> &DynamicTensor {
        &self.0.value
    }

    /// Whether both constants hold the same shape, dtype, and element bit
    /// patterns. Bitwise equality keeps `-0.0` apart from `0.0` and never
    /// merges values that some consumer could tell apart; two NaNs merge only
    /// when their payloads match.
    fn same_bits(&self, other: &Self) -> bool {
        let (lhs, rhs) = (self.value(), other.value());
        Arc::ptr_eq(&self.0, &other.0)
            || (lhs.dtype == rhs.dtype
                && lhs.shape == rhs.shape
                && lhs
                    .data
                    .iter()
                    .zip(rhs.data.iter())
                    .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits()))
    }

    /// A hash consistent with [`Self::same_bits`].
    fn bits_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let value = self.value();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.dtype.hash(&mut hasher);
        value.shape.hash(&mut hasher);
        for element in value.data.iter() {
            element.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl std::fmt::Debug for TensorConstant {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "TensorConstant({})",
            format_tensor_type(&self.value().shape, self.value().dtype)
        )
    }
}

#[derive(Clone, Debug)]
enum TensorOp {
    Input {
        name: String,
    },
    /// A weak or strong scalar known at graph-construction time (Python
    /// scalars and AD-internal constants).
    ScalarConstant {
        value: f64,
    },
    /// An array known at graph-construction time; the node's shape and dtype
    /// are those of the value. It has no inputs, a zero tangent, and receives
    /// no cotangent.
    Constant {
        value: TensorConstant,
    },
    /// Element type conversion; the target dtype is the node's dtype.
    Cast {
        input: TensorNodeId,
    },
    Add {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Sub {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Div {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Mul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Greater {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    /// IEEE comparison with a `Bool` result. Unlike the legacy `Greater`
    /// float mask, its operands are promoted like arithmetic operands.
    Compare {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        kind: TensorComparison,
    },
    Where {
        condition: TensorNodeId,
        on_true: TensorNodeId,
        on_false: TensorNodeId,
    },
    /// A scalar-predicate lazy branch. The regions own their input captures
    /// and are evaluated only after the predicate has been materialized.
    Cond {
        predicate: TensorNodeId,
        branches: TensorCondExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    },
    Fori {
        carry: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    },
    /// Tangent result of a fixed-bound `Fori`. The matching primal remains an
    /// ordinary `Fori`, avoiding the packed slice/concat carry representation.
    ForiJvp {
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        tangent_captures: Vec<(String, TensorNodeId)>,
    },
    /// One selected result of a shared fixed-bound `Fori` reverse pass.
    ForiVjp {
        carry: TensorNodeId,
        output_cotangent: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        target: TensorForiVjpTarget,
        group: usize,
    },
    /// One selected directional derivative of a shared fixed-bound `Fori`
    /// reverse pass. This is the structural forward-over-reverse rule used by
    /// compiled HVP and Hessian transforms.
    ForiVjpJvp {
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        output_cotangent: TensorNodeId,
        output_cotangent_tangent: TensorNodeId,
        plan: TensorForiVjpJvpExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        tangent_captures: Vec<(String, TensorNodeId)>,
        target: TensorForiVjpTarget,
        group: usize,
    },
    /// A data-dependent loop: the body runs while the predicate region
    /// returns true. Only the final carry is produced.
    While {
        carry: TensorNodeId,
        loop_plan: TensorWhileExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    },
    /// One selected result of a shared fixed-bound `Scan` execution.
    Scan {
        carry: TensorNodeId,
        scan_plan: TensorScanExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        target: TensorScanTarget,
        group: usize,
    },
    /// One selected result of a shared fixed-bound `Scan` reverse pass.
    ScanVjp {
        carry: TensorNodeId,
        final_carry_cotangent: TensorNodeId,
        output_cotangent: TensorNodeId,
        scan_plan: TensorScanExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        target: TensorScanVjpTarget,
        group: usize,
    },
    /// One selected directional derivative of a shared fixed-bound `Scan`
    /// reverse pass. Both the final-carry and stacked-output cotangents are
    /// differentiated together, preserving the joint reverse semantics.
    ScanVjpJvp {
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        final_carry_cotangent: TensorNodeId,
        final_carry_cotangent_tangent: TensorNodeId,
        output_cotangent: TensorNodeId,
        output_cotangent_tangent: TensorNodeId,
        plan: TensorScanVjpJvpExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        tangent_captures: Vec<(String, TensorNodeId)>,
        target: TensorScanVjpTarget,
        group: usize,
    },
    Sum {
        input: TensorNodeId,
    },
    SumAxis {
        input: TensorNodeId,
        axis: usize,
    },
    /// The maximum or minimum of `input` along `axis`, which is removed.
    /// NaN propagates and the sign of zero is ordered (`-0 < +0`, IEEE 754-2019
    /// `maximum`), so the result does not depend on the reduction order; ties
    /// share the derivative equally (JAX's chooser rule). See `extremum.rs`.
    ExtremumAxis {
        input: TensorNodeId,
        axis: usize,
        kind: TensorExtremum,
    },
    Matmul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Solve {
        matrix: TensorNodeId,
        rhs: TensorNodeId,
    },
    Cholesky {
        input: TensorNodeId,
    },
    CholeskyAd {
        inputs: Vec<TensorNodeId>,
        kind: CholeskyAdKind,
    },
    Triangular {
        input: TensorNodeId,
        lower: bool,
    },
    /// One output of a dense decomposition of every square matrix of the
    /// leading batch axes; see [`LinalgKind`]. The derivative rules are
    /// written with IR ops (`solve`, `matmul`, and the `Linalg` outputs
    /// themselves), so every derivative order is again an ordinary graph.
    Linalg {
        input: TensorNodeId,
        kind: LinalgKind,
    },
    Sqrt {
        input: TensorNodeId,
    },
    SqrtDerivative {
        input: TensorNodeId,
        order: u32,
    },
    Reshape {
        input: TensorNodeId,
    },
    Mean {
        input: TensorNodeId,
    },
    MeanAxis {
        input: TensorNodeId,
        axis: usize,
    },
    Powi {
        input: TensorNodeId,
        exponent: u32,
    },
    Transpose {
        input: TensorNodeId,
        axes: Vec<usize>,
    },
    /// Elementwise `kind(input)` for the math functions of
    /// [`UnaryMathKind`] (`exp`, `log`, `sin`, `tanh`, `erf`, the inverse
    /// trigonometric and hyperbolic functions, `cbrt`, `floor`, ...), with
    /// IEEE semantics: NaN outside the domain, never an error. Every rule of
    /// a kind (value, derivatives, CUDA and MLX lowering, folding) lives in
    /// `elementwise.rs`.
    UnaryMath {
        input: TensorNodeId,
        kind: UnaryMathKind,
    },
    /// Elementwise `kind(lhs, rhs)` with broadcasting for the math
    /// functions of [`BinaryMathKind`] (`pow`, `atan2`, `fmod`), whose rules
    /// live in `elementwise.rs` like those of `UnaryMath`.
    BinaryMath {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        kind: BinaryMathKind,
    },
    /// The identity on values whose derivative is zero in every AD mode: its
    /// JVP tangent is zero and its VJP sends no cotangent to `input`.
    /// Backends execute it as the identity.
    StopGradient {
        input: TensorNodeId,
    },
    /// Inclusive prefix sum along `axis`, from the last entry when
    /// `reverse`. Each running sum rounds to the node dtype, so a float32
    /// scan is the sequential chain of rounded float32 adds.
    CumSum {
        input: TensorNodeId,
        axis: usize,
        reverse: bool,
    },
    Concat {
        inputs: Vec<TensorNodeId>,
        axis: usize,
    },
    Slice {
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
    },
    PadSlice {
        input: TensorNodeId,
        axis: usize,
        start: usize,
    },
    /// The entries of `input` at `indices` along `axis`, in index order;
    /// a repeated index repeats its entry. One node replaces a slice per
    /// index, so traced gathers and their gradients stay O(1) in nodes.
    Gather {
        input: TensorNodeId,
        indices: Arc<[usize]>,
        axis: usize,
    },
    /// `base` with entry `j` of `updates` along `axis` added to entry
    /// `indices[j]`. Contributions to one destination accumulate in
    /// increasing `j`, and every sum rounds to the node dtype, exactly as a
    /// chain of one padded add per index would.
    ScatterAdd {
        base: TensorNodeId,
        updates: TensorNodeId,
        indices: Arc<[usize]>,
        axis: usize,
    },
    Broadcast {
        input: TensorNodeId,
    },
    /// Output `output` of a call of a function with a custom differentiation
    /// rule (`custom_vjp`, `custom_jvp`, `checkpoint`): the identity on its
    /// primal `value`, which the symbolic transforms differentiate through
    /// `rule` with respect to `operands` instead of through the primal
    /// computation. The outputs of one call share `group`, the id of the
    /// first of them, as `Scan` results do. Plan compilation aliases the
    /// node to `value`, so backends never execute it; see `custom.rs`.
    Custom {
        value: TensorNodeId,
        operands: Vec<TensorNodeId>,
        rule: Arc<TensorCustomRule>,
        output: usize,
        group: usize,
    },
}

#[derive(Clone, Debug)]
enum TensorForiVjpTarget {
    Carry,
    External(String),
}

#[derive(Clone, Debug)]
struct TensorForiVjpEvaluation {
    carry_gradient: DynamicTensor,
    external_gradients: BTreeMap<String, DynamicTensor>,
}

#[derive(Clone, Debug)]
struct TensorForiVjpJvpEvaluation {
    gradients: BTreeMap<String, DynamicTensor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorScanTarget {
    Carry,
    Outputs,
}

#[derive(Clone, Debug)]
struct TensorScanEvaluation {
    carry: DynamicTensor,
    outputs: DynamicTensor,
}

#[derive(Clone, Debug)]
struct TensorScanJvpEvaluation {
    carry_tangent: DynamicTensor,
    output_tangent: DynamicTensor,
}

#[derive(Clone, Debug)]
enum TensorScanVjpTarget {
    Carry,
    External(String),
}

#[derive(Clone, Debug)]
struct TensorScanVjpEvaluation {
    carry_gradient: DynamicTensor,
    external_gradients: BTreeMap<String, DynamicTensor>,
}

#[derive(Clone, Debug)]
struct TensorScanVjpJvpEvaluation {
    gradients: BTreeMap<String, DynamicTensor>,
}

struct TensorScanVjpBindings {
    carry: TensorNodeId,
    final_carry_cotangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: Vec<(String, TensorNodeId)>,
}

struct TensorScanVjpJvpBindings {
    carry: TensorNodeId,
    carry_tangent: TensorNodeId,
    final_carry_cotangent: TensorNodeId,
    final_carry_cotangent_tangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    output_cotangent_tangent: TensorNodeId,
    captures: Vec<(String, TensorNodeId)>,
    tangent_captures: Vec<(String, TensorNodeId)>,
}

struct TensorForiVjpJvpBindings {
    carry: TensorNodeId,
    carry_tangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    output_cotangent_tangent: TensorNodeId,
    captures: Vec<(String, TensorNodeId)>,
    tangent_captures: Vec<(String, TensorNodeId)>,
}

type SymbolicScanPair = ((TensorNodeId, TensorNodeId), (TensorNodeId, TensorNodeId));

/// `weak` follows JAX's weak_type: a node derived only from Python scalars or
/// AD-internal constants. When a weak node meets a strong node of another
/// dtype, graph construction inserts an explicit `Cast` of the weak operand,
/// so every non-mask operand of a built node already has the node's dtype.
#[derive(Clone, Debug)]
struct TensorNode {
    op: TensorOp,
    shape: Vec<usize>,
    dtype: TensorDType,
    weak: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TensorIr {
    nodes: Arc<Vec<TensorNode>>,
    input_nodes: Arc<HashMap<String, TensorNodeId>>,
}

#[derive(Clone, Debug)]
pub struct SymbolicJvp {
    pub graph: TensorIr,
    pub value: TensorNodeId,
    pub tangent: TensorNodeId,
}

/// A forward-mode transform retaining multiple requested primal/tangent pairs
/// from one shared transformed Tensor IR graph.
#[derive(Clone, Debug)]
pub struct SymbolicJvpMany {
    pub graph: TensorIr,
    pub values: Vec<TensorNodeId>,
    pub tangents: Vec<TensorNodeId>,
}

#[derive(Clone, Debug)]
pub struct SymbolicVjp {
    pub graph: TensorIr,
    pub value: TensorNodeId,
    pub cotangent: TensorNodeId,
    pub gradients: BTreeMap<String, TensorNodeId>,
}

/// The cotangent seed of one differentiated output of
/// [`TensorIr::symbolic_vjp_many`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicCotangent {
    /// A new graph input with this name, shaped and typed like the output,
    /// bound by the caller at execution time (`vjp`).
    Input(String),
    /// A constant tensor of ones shaped and typed like the output, so the
    /// transformed graph has no extra inputs (`grad` of a scalar loss).
    Ones,
}

/// A reverse-mode transform of several seeded outputs that shares one
/// transformed graph with the rebuilt primal of every source node.
#[derive(Clone, Debug)]
pub struct SymbolicVjpMany {
    pub graph: TensorIr,
    /// The rebuilt primal of each source node, indexed by source node id, so
    /// a caller can retain values that are not differentiated (auxiliary
    /// outputs) without evaluating the source graph a second time.
    pub primals: Vec<TensorNodeId>,
    /// The seed node of each differentiated output, in request order.
    pub cotangents: Vec<TensorNodeId>,
    /// The summed VJP of every non-`bool` source input, keyed by input name.
    pub gradients: BTreeMap<String, TensorNodeId>,
}

/// Why [`TensorIr::inline_batched`] could not splice a callee.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchingError {
    /// A node that depends on a mapped input has no batching rule; `op` is
    /// its IR op name.
    Unsupported { op: &'static str },
    /// Invalid bindings, outputs, or batch size.
    Invalid(String),
}

impl std::fmt::Display for BatchingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported { op } => write!(
                formatter,
                "vmap cannot batch a {op} node that depends on a mapped argument: it has no \
                 batching rule"
            ),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl From<String> for BatchingError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

#[derive(Clone, Debug)]
pub struct TensorExecutionPlan {
    nodes: Arc<Vec<TensorNode>>,
    input_nodes: Arc<HashMap<String, TensorNodeId>>,
    output_node_id: TensorNodeId,
    output_node_ids: Vec<TensorNodeId>,
    fused_elementwise_output: bool,
}

/// A frozen, explicitly captured branch region.
///
/// Regions retain the normal Tensor IR input contract, but are kept separate
/// from their parent until a future nested `Cond` node owns them. This makes
/// lazy branch evaluation available without treating `where` as control flow.
#[derive(Clone, Debug)]
pub struct TensorRegion {
    plan: TensorExecutionPlan,
    captures: BTreeMap<String, Vec<usize>>,
}

/// A frozen region retaining every output from one shared Tensor IR plan.
///
/// This is the multi-result counterpart of `TensorRegion`; it is used by
/// structured loops that must carry state and reverse accumulators together.
#[derive(Clone, Debug)]
pub struct TensorMultiRegion {
    plan: TensorExecutionPlan,
    captures: BTreeMap<String, Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

/// Two shape-compatible CPU branch regions selected by a host boolean.
///
/// `TensorOp::Cond` materializes its scalar predicate before selecting a
/// region. MLX and CUDA read a device predicate back once and execute only the
/// selected region on the device; CUDA compiles both regions ahead of time and
/// rejects `Cond` inside fused device loop bodies.
#[derive(Clone, Debug)]
pub struct TensorCondExecutionPlan {
    on_true: TensorRegion,
    on_false: TensorRegion,
}

/// A frozen, fixed-bound loop body with an explicit carry capture and an
/// optional scalar index capture.
///
/// This is the region-level execution contract used by a future `Fori` IR
/// node. It is intentionally separate from host-static graph unrolling.
#[derive(Clone, Debug)]
pub struct TensorForiExecutionPlan {
    lower: usize,
    upper: usize,
    body: TensorRegion,
    carry_name: String,
    index_name: String,
    external_captures: BTreeMap<String, Vec<usize>>,
    #[cfg(feature = "mlx")]
    mlx_vjp: Option<TensorMlxVjpPlan>,
}

/// A loop with a traced termination predicate (`while_loop`).
///
/// Both regions read the same named inputs: the carry plus the external
/// captures. The predicate region returns a scalar `Bool`; the body region
/// returns the next carry. Every backend evaluates the predicate, reads it
/// back to the host, and runs the body only while it is true, so the trip
/// count is data dependent and no fixed-length tape exists. Reverse-mode
/// differentiation is therefore rejected; forward mode runs the same loop
/// over a packed `(primal, tangent)` carry.
#[derive(Clone, Debug)]
pub struct TensorWhileExecutionPlan {
    predicate: TensorRegion,
    body: TensorRegion,
    carry_name: String,
    external_captures: BTreeMap<String, Vec<usize>>,
}

/// Precompiled body reverse plan used by device backends that execute fixed
/// loop iterations through host-side dispatch.
#[derive(Clone, Debug)]
#[cfg(feature = "mlx")]
struct TensorMlxVjpPlan {
    plan: TensorExecutionPlan,
    cotangent_name: String,
    gradient_node_ids: BTreeMap<String, TensorNodeId>,
}

/// The forward carry sequence for one fixed-bound `Fori` invocation.
///
/// Entry zero is the initial carry and entry `n` is the carry after `n`
/// iterations. The tape is intentionally opaque so reverse lowering can own
/// its storage representation without exposing mutable tensor buffers.
#[derive(Clone, Debug)]
pub struct TensorForiTape {
    carries: Vec<DynamicTensor>,
}

/// A fixed-bound region loop with several independently shaped carries.
///
/// The body has one output per carry, in `carry_names` order. It is not yet a
/// parent Tensor IR node; the initial use is reverse-loop lowering where the
/// carry contains a cotangent plus capture-gradient accumulators.
#[derive(Clone, Debug)]
pub struct TensorForiMultiExecutionPlan {
    lower: usize,
    upper: usize,
    body: TensorMultiRegion,
    carry_names: Vec<String>,
    index_name: String,
    external_captures: BTreeMap<String, Vec<usize>>,
}

/// A fixed-bound carry/output region scan.
///
/// The body returns `(next_carry, output)` and runs once per index. Outputs
/// have one leading time axis in the returned tensor.
#[derive(Clone, Debug)]
pub struct TensorScanExecutionPlan {
    lower: usize,
    upper: usize,
    body: TensorMultiRegion,
    carry_name: String,
    index_name: String,
    external_captures: BTreeMap<String, Vec<usize>>,
    #[cfg(feature = "mlx")]
    mlx_vjp: Option<TensorScanMlxVjpPlan>,
}

#[derive(Clone, Debug)]
#[cfg(feature = "mlx")]
struct TensorScanMlxVjpPlan {
    carry: TensorMlxVjpPlan,
    output: TensorMlxVjpPlan,
}

/// Compiled forward-over-reverse transform for one fixed-bound `Fori` VJP.
///
/// The plan records primal and directional carry tapes, then runs the body VJP
/// backward while evaluating compiled JVPs of each requested body gradient.
#[derive(Clone, Debug)]
pub struct TensorForiVjpJvpExecutionPlan {
    loop_plan: TensorForiExecutionPlan,
    /// The prefix of the plan's internal input names, kept so a batched
    /// loop plan can be transformed again with the same names.
    namespace: String,
    cotangent_name: String,
    tangent_names: BTreeMap<String, String>,
    gradient_tangent_plans: BTreeMap<String, TensorExecutionPlan>,
    #[cfg(feature = "mlx")]
    mlx_forward_jvp: TensorForiMlxForwardJvpPlan,
}

/// Compiled forward-over-reverse transform for one fixed-bound `Scan` VJP.
///
/// The two body outputs have independent cotangent shapes, so the plan keeps
/// separate reverse transforms and adds their directional contributions in the
/// shared reverse recurrence.
#[derive(Clone, Debug)]
pub struct TensorScanVjpJvpExecutionPlan {
    scan_plan: TensorScanExecutionPlan,
    /// The prefix of the plan's internal input names, kept so a batched
    /// scan plan can be transformed again with the same names.
    namespace: String,
    carry_cotangent_name: String,
    output_cotangent_name: String,
    tangent_names: BTreeMap<String, String>,
    carry_gradient_tangent_plans: BTreeMap<String, TensorExecutionPlan>,
    output_gradient_tangent_plans: BTreeMap<String, TensorExecutionPlan>,
    #[cfg(feature = "mlx")]
    mlx_forward_jvp: TensorScanMlxForwardJvpPlan,
}

#[derive(Clone, Debug)]
#[cfg(feature = "mlx")]
struct TensorScanMlxForwardJvpPlan {
    plan: TensorExecutionPlan,
    value_node_id: TensorNodeId,
    tangent_node_id: TensorNodeId,
    tangent_names: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
#[cfg(feature = "mlx")]
struct TensorForiMlxForwardJvpPlan {
    plan: TensorExecutionPlan,
    value_node_id: TensorNodeId,
    tangent_node_id: TensorNodeId,
    tangent_names: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct TensorScanTape {
    carries: Vec<DynamicTensor>,
}

type TensorScanCheckpointVjpResult = (
    DynamicTensor,
    Option<DynamicTensor>,
    DynamicTensor,
    BTreeMap<String, DynamicTensor>,
);

pub type TensorScanJvpResult = (
    (DynamicTensor, DynamicTensor),
    (DynamicTensor, DynamicTensor),
);

pub type TensorForiMultiVjpResult = (
    Vec<DynamicTensor>,
    Vec<DynamicTensor>,
    BTreeMap<String, DynamicTensor>,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorDeviceBackend {
    Cpu,
    Cuda,
    Mlx,
}

impl std::fmt::Display for TensorDeviceBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Mlx => "mlx",
        })
    }
}

/// One addressable device in a backend-specific process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorDeviceId {
    pub backend: TensorDeviceBackend,
    pub ordinal: usize,
}

impl std::fmt::Display for TensorDeviceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:{}", self.backend, self.ordinal)
    }
}

/// A static logical mesh over uniquely addressed devices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDeviceMesh {
    pub devices: Vec<TensorDeviceId>,
    pub axis_names: Vec<String>,
    pub shape: Vec<usize>,
}

impl TensorDeviceMesh {
    pub fn validate(&self) -> Result<(), String> {
        if self.axis_names.is_empty() || self.shape.is_empty() {
            return Err("tensor device mesh requires at least one axis".to_string());
        }
        if self.axis_names.len() != self.shape.len() {
            return Err(format!(
                "tensor device mesh has {} axis names but {} shape extents",
                self.axis_names.len(),
                self.shape.len()
            ));
        }
        if self.axis_names.iter().any(|axis| axis.is_empty()) {
            return Err("tensor device mesh axis names must be non-empty".to_string());
        }
        let mut axis_names = HashSet::new();
        if self
            .axis_names
            .iter()
            .any(|axis| !axis_names.insert(axis.as_str()))
        {
            return Err("tensor device mesh axis names must be unique".to_string());
        }
        let expected_devices = self.shape.iter().try_fold(1usize, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| "tensor device mesh shape product overflows usize".to_string())
        })?;
        if self.shape.contains(&0) {
            return Err("tensor device mesh extents must be positive".to_string());
        }
        if self.devices.len() != expected_devices {
            return Err(format!(
                "tensor device mesh shape requires {expected_devices} devices, got {}",
                self.devices.len()
            ));
        }
        let mut devices = HashSet::new();
        if self.devices.iter().any(|device| !devices.insert(*device)) {
            return Err("tensor device mesh devices must be unique".to_string());
        }
        Ok(())
    }

    fn axis_extent(&self, axis_name: &str) -> Result<usize, String> {
        self.axis_names
            .iter()
            .position(|candidate| candidate == axis_name)
            .map(|index| self.shape[index])
            .ok_or_else(|| format!("tensor device mesh has no axis {axis_name:?}"))
    }
}

/// How one logical tensor is partitioned over a device mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorPartitionSpec {
    Replicated,
    Sharded {
        tensor_axis: usize,
        mesh_axis: String,
    },
}

/// How replica-local scalar losses combine into one global scalar loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorReplicaReduction {
    Sum,
    Mean,
}

/// Typed device placement metadata carried by compiler IR, not by eager values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TensorPlacement {
    #[default]
    Unplaced,
    SingleDevice(TensorDeviceId),
    Mesh {
        mesh: TensorDeviceMesh,
        partition: TensorPartitionSpec,
    },
}

impl std::fmt::Display for TensorPlacement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unplaced => formatter.write_str("unplaced"),
            Self::SingleDevice(device) => write!(formatter, "device[{device}]"),
            Self::Mesh { mesh, partition } => {
                let devices = mesh
                    .devices
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                match partition {
                    TensorPartitionSpec::Replicated => {
                        write!(formatter, "mesh[{devices}]{{replicated}}")
                    }
                    TensorPartitionSpec::Sharded {
                        tensor_axis,
                        mesh_axis,
                    } => write!(
                        formatter,
                        "mesh[{devices}]{{sharded:tensor_axis={tensor_axis},mesh_axis={mesh_axis}}}"
                    ),
                }
            }
        }
    }
}

impl TensorPlacement {
    fn validate_for_shape(&self, shape: &[usize]) -> Result<(), String> {
        let Self::Mesh { mesh, partition } = self else {
            return Ok(());
        };
        mesh.validate()?;
        let TensorPartitionSpec::Sharded {
            tensor_axis,
            mesh_axis,
        } = partition
        else {
            return Ok(());
        };
        let extent = shape.get(*tensor_axis).ok_or_else(|| {
            format!(
                "sharded tensor axis {tensor_axis} is out of bounds for rank {}",
                shape.len()
            )
        })?;
        let mesh_extent = mesh.axis_extent(mesh_axis)?;
        if extent % mesh_extent != 0 {
            return Err(format!(
                "tensor extent {extent} on axis {tensor_axis} is not divisible by mesh axis {mesh_axis:?} extent {mesh_extent}"
            ));
        }
        Ok(())
    }
}

/// Logical element type carried by compiler IR nodes and host tensors.
///
/// Host storage stays `f64`; an `F32` value is the correctly rounded `f32` of
/// its `f64` computation. CUDA and MLX map each logical dtype to an execution
/// dtype through [`TensorDeviceBackend::execution_dtype`].
///
/// `Bool` is stored as `0.0`/`1.0` and produced by comparisons; it is not an
/// arithmetic dtype (see [`TensorIr::compare`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorDType {
    F32,
    F64,
    Bool,
}

impl TensorDType {
    /// Rounds an `f64` value to the nearest value representable in this dtype.
    ///
    /// For `Bool` this is the float-to-bool conversion: every non-zero value,
    /// including `NaN`, becomes `1.0` (NumPy `astype(bool)` semantics).
    pub fn round(self, value: f64) -> f64 {
        match self {
            // `as f32` is IEEE round-to-nearest-even and overflows to ±inf, matching device f32.
            Self::F32 => value as f32 as f64,
            Self::F64 => value,
            Self::Bool => f64::from(value != 0.0),
        }
    }

    pub fn is_floating(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

impl std::fmt::Display for TensorDType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bool => "bool",
        })
    }
}

/// Elementwise comparison kind of [`TensorIr::compare`]. Comparisons follow
/// IEEE semantics: any comparison with `NaN` is false except `NotEqual`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorComparison {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
}

impl TensorComparison {
    pub fn name(self) -> &'static str {
        match self {
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
        }
    }

    /// C spelling shared by the CUDA code generators.
    fn operator(self) -> &'static str {
        match self {
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        }
    }

    /// The IEEE comparison itself; host tensors store the result as 0/1.
    pub fn evaluate(self, lhs: f64, rhs: f64) -> bool {
        match self {
            Self::Greater => lhs > rhs,
            Self::GreaterEqual => lhs >= rhs,
            Self::Less => lhs < rhs,
            Self::LessEqual => lhs <= rhs,
            Self::Equal => lhs == rhs,
            Self::NotEqual => lhs != rhs,
        }
    }
}

/// Precision boundary between a logical Tensor IR program and one backend's
/// execution representation. Conversion is a lowering responsibility, never
/// an implicit eager-tensor mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorBackendPrecision {
    pub logical: TensorDType,
    pub execution: TensorDType,
}

impl TensorDeviceBackend {
    /// Default-precision contract for `f64` programs, kept for callers that
    /// predate typed IR; equivalent to `execution_dtype(TensorDType::F64)`.
    pub const fn precision(self) -> TensorBackendPrecision {
        TensorBackendPrecision {
            logical: TensorDType::F64,
            execution: match self {
                Self::Cpu => TensorDType::F64,
                Self::Cuda | Self::Mlx => TensorDType::F32,
            },
        }
    }

    /// Maps a logical IR dtype to the element type this backend executes.
    ///
    /// CUDA and MLX lower `F64` programs to `f32` kernels (native device `f64`
    /// is not implemented), execute `F32` natively, and hold `Bool` as `f32`
    /// `0`/`1`; the CPU executes every dtype as its logical type.
    pub fn execution_dtype(self, logical: TensorDType) -> Result<TensorDType, String> {
        Ok(match (self, logical) {
            (Self::Cpu, dtype) => dtype,
            // Bool executes as f32 0/1 values; real bit/u8 storage is post-D4 work.
            (Self::Cuda | Self::Mlx, TensorDType::F64 | TensorDType::F32 | TensorDType::Bool) => {
                TensorDType::F32
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorKernelNode {
    pub id: TensorNodeId,
    pub op: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub layout: String,
    pub placement: TensorPlacement,
    pub effect: String,
    pub alias_of: Option<TensorNodeId>,
    pub inputs: Vec<TensorNodeId>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorKernelProgram {
    pub nodes: Vec<TensorKernelNode>,
    pub output_node_id: TensorNodeId,
}

/// One collective required after a replica-local node in a sharded program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAllReduce {
    pub node_id: TensorNodeId,
    pub mesh: TensorDeviceMesh,
    pub mesh_axis: String,
    pub reduction: TensorReplicaReduction,
}

/// Backend-neutral sharding schedule for a frozen tensor plan.
///
/// The kernel program describes local computation. `all_reduces` records the
/// collective boundaries a CUDA/NCCL or another distributed backend must lower
/// before consuming the affected values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShardingPlan {
    pub program: TensorKernelProgram,
    pub all_reduces: Vec<TensorAllReduce>,
}

/// Replica-local lowering of a [`TensorShardingPlan`] for single-node 1-D data
/// parallelism.
///
/// `replica_plan` is the source plan specialized to axis-zero shard shapes with
/// unchanged node ids. `collectives` keeps the order of
/// [`TensorShardingPlan::all_reduces`]; a backend applies them after the whole
/// replica program, which is only equivalent to the schedule because every
/// entry reduces a retained output that no other node consumes.
#[derive(Clone, Debug)]
pub struct TensorDataParallelProgram {
    pub replica_plan: TensorExecutionPlan,
    pub collectives: Vec<(TensorNodeId, TensorReplicaReduction)>,
}

/// One reusable temporary allocation in a backend-neutral execution plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorBufferSlot {
    pub id: usize,
    pub element_count: usize,
}

/// Static liveness-based allocation contract for a frozen tensor plan.
///
/// Inputs are externally bound and array constants are resident for the
/// plan's lifetime, so neither has a slot. `reshape` nodes
/// carry an alias instead of allocating storage. All other values use an
/// exact-size temporary slot that may be reused after its final consumer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorBufferPlan {
    pub slots: Vec<TensorBufferSlot>,
    pub node_slots: Vec<Option<usize>>,
    pub node_aliases: Vec<Option<TensorNodeId>>,
    pub output_backing_node_id: TensorNodeId,
}

/// A maximal elementwise region whose leaves must be materialized first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorFusionRegion {
    pub output_node_id: TensorNodeId,
    pub node_ids: Vec<TensorNodeId>,
    pub input_node_ids: Vec<TensorNodeId>,
}

impl TensorKernelProgram {
    pub fn validate(&self) -> Result<(), String> {
        for (position, node) in self.nodes.iter().enumerate() {
            if node.id != position {
                return Err(format!(
                    "kernel node id {} does not match its position {position}",
                    node.id
                ));
            }
            if node.shape.contains(&0) {
                return Err(format!("kernel node {} has a zero tensor extent", node.id));
            }
            if !matches!(
                node.dtype,
                TensorDType::F64 | TensorDType::F32 | TensorDType::Bool
            ) {
                return Err(format!(
                    "kernel node {} has unsupported dtype {:?}",
                    node.id, node.dtype
                ));
            }
            let expected_layout = if node.shape.is_empty() {
                "scalar"
            } else {
                "row_major_contiguous"
            };
            if node.layout != expected_layout {
                return Err(format!(
                    "kernel node {} has layout {:?}, expected {expected_layout}",
                    node.id, node.layout
                ));
            }
            node.placement
                .validate_for_shape(&node.shape)
                .map_err(|error| {
                    format!("kernel node {} has invalid placement: {error}", node.id)
                })?;
            let expected_effect = if node.op == "input" { "input" } else { "pure" };
            if node.effect != expected_effect {
                return Err(format!(
                    "kernel node {} has effect {:?}, expected {expected_effect}",
                    node.id, node.effect
                ));
            }
            let expected_alias = if node.op == "reshape" {
                node.inputs.first().copied()
            } else {
                None
            };
            if node.alias_of != expected_alias {
                return Err(format!(
                    "kernel node {} has alias {:?}, expected {:?}",
                    node.id, node.alias_of, expected_alias
                ));
            }
            for input in &node.inputs {
                if *input >= position {
                    return Err(format!(
                        "kernel node {} references non-dominating input {input}",
                        node.id
                    ));
                }
            }
            if node.op == "input" && node.name.is_none() {
                return Err(format!("input kernel node {} has no name", node.id));
            }
        }
        if self.output_node_id >= self.nodes.len() {
            return Err(format!(
                "kernel output node {} does not exist",
                self.output_node_id
            ));
        }
        Ok(())
    }
}

impl TensorShardingPlan {
    pub fn validate(&self) -> Result<(), String> {
        self.program.validate()?;
        let mut seen_nodes = HashSet::new();
        for collective in &self.all_reduces {
            if !seen_nodes.insert(collective.node_id) {
                return Err(format!(
                    "sharding plan has multiple all-reduces for node {}",
                    collective.node_id
                ));
            }
            collective.mesh.validate()?;
            collective.mesh.axis_extent(&collective.mesh_axis)?;
            let node = self.program.nodes.get(collective.node_id).ok_or_else(|| {
                format!(
                    "sharding plan all-reduce node {} does not exist",
                    collective.node_id
                )
            })?;
            let expected = TensorPlacement::Mesh {
                mesh: collective.mesh.clone(),
                partition: TensorPartitionSpec::Replicated,
            };
            if node.placement != expected {
                return Err(format!(
                    "sharding plan all-reduce node {} must become replicated on its mesh",
                    collective.node_id
                ));
            }
        }
        Ok(())
    }
}

pub trait TensorBackend {
    fn name(&self) -> &'static str;

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl TensorBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        plan.execute_cpu(inputs)
    }
}

#[derive(Clone)]
struct MixedTangent {
    value: DynamicTensor,
    first: DynamicTensor,
    second: DynamicTensor,
    mixed: DynamicTensor,
}

// Completed lower columns record historical multipliers and do not follow later
// row swaps. RHS replay preserves solve_lu's swap/subtraction arithmetic order.
struct SolveReplayPlan {
    factor: Vec<f64>,
    pivots: Vec<usize>,
    n: usize,
}

impl SolveReplayPlan {
    fn for_finite_dense(matrix: &DynamicTensor) -> Option<Self> {
        // Batched stacks take the general per-matrix path.
        if matrix.shape.len() != 2 {
            return None;
        }
        let n = matrix.shape[0];
        if matrix.data.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut lower = true;
        let mut upper = true;
        for row in 0..n {
            for column in 0..row {
                upper &= matrix.data.get(row * n + column) == 0.0;
                lower &= matrix.data.get(column * n + row) == 0.0;
            }
        }
        if lower || upper {
            return None;
        }
        let mut factor = matrix.data.to_vec();
        let mut pivots = Vec::with_capacity(n);
        for pivot in 0..n {
            let pivot_row = (pivot..n).max_by(|&left, &right| {
                factor[left * n + pivot]
                    .abs()
                    .total_cmp(&factor[right * n + pivot].abs())
            })?;
            if factor[pivot_row * n + pivot] == 0.0 {
                return None;
            }
            pivots.push(pivot_row);
            for column in pivot..n {
                factor.swap(pivot * n + column, pivot_row * n + column);
            }
            let diagonal = factor[pivot * n + pivot];
            for row in pivot + 1..n {
                let multiplier = factor[row * n + pivot] / diagonal;
                factor[row * n + pivot] = multiplier;
                for column in pivot + 1..n {
                    factor[row * n + column] -= multiplier * factor[pivot * n + column];
                }
            }
        }
        factor
            .iter()
            .all(|value| value.is_finite())
            .then_some(Self { factor, pivots, n })
    }

    fn solve_finite(&self, rhs: &DynamicTensor) -> Option<DynamicTensor> {
        if rhs.data.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let columns = rhs.shape[1];
        let n = self.n;
        let mut result = rhs.data.to_vec();
        for (pivot, &pivot_row) in self.pivots.iter().enumerate() {
            for column in 0..columns {
                result.swap(pivot * columns + column, pivot_row * columns + column);
            }
            for row in pivot + 1..n {
                let multiplier = self.factor[row * n + pivot];
                for column in 0..columns {
                    result[row * columns + column] -= multiplier * result[pivot * columns + column];
                }
            }
        }
        for row in (0..n).rev() {
            let diagonal = self.factor[row * n + row];
            for column in 0..columns {
                let mut value = result[row * columns + column];
                for inner in row + 1..n {
                    value -= self.factor[row * n + inner] * result[inner * columns + column];
                }
                result[row * columns + column] = value / diagonal;
            }
        }
        if result.iter().any(|value| !value.is_finite()) {
            return None;
        }
        // Replay preserves the validated RHS shape and element count.
        Some(DynamicTensor {
            shape: rhs.shape.clone(),
            data: HostTensorStorage::from_f64(result, TensorDType::F64),
            dtype: TensorDType::F64,
        })
    }
}

impl MixedTangent {
    fn solve_reusing_finite_factor(&self, rhs: &Self) -> Result<Option<Self>, String> {
        solve_shape(&self.value.shape, &rhs.value.shape)?;
        let Some(plan) = SolveReplayPlan::for_finite_dense(&self.value) else {
            return Ok(None);
        };
        let Some(value) = plan.solve_finite(&rhs.value) else {
            return Ok(None);
        };
        let first_rhs = rhs.first.sub(&self.first.matmul(&value)?)?;
        solve_shape(&self.value.shape, &first_rhs.shape)?;
        let Some(first) = plan.solve_finite(&first_rhs) else {
            return Ok(None);
        };
        let second_rhs = rhs.second.sub(&self.second.matmul(&value)?)?;
        solve_shape(&self.value.shape, &second_rhs.shape)?;
        let Some(second) = plan.solve_finite(&second_rhs) else {
            return Ok(None);
        };
        let mixed_rhs = rhs
            .mixed
            .sub(&self.mixed.matmul(&value)?)?
            .sub(&self.first.matmul(&second)?)?
            .sub(&self.second.matmul(&first)?)?;
        solve_shape(&self.value.shape, &mixed_rhs.shape)?;
        Ok(plan.solve_finite(&mixed_rhs).map(|mixed| Self {
            value,
            first,
            second,
            mixed,
        }))
    }
}

fn matmul_host_float_block<L: Copy + Into<f64>, R: Copy + Into<f64>>(
    lhs: &[L],
    rhs: &[R],
    output: &mut [f64],
    [rows, inner, columns]: [usize; 3],
) {
    for row in 0..rows {
        let output = &mut output[row * columns..(row + 1) * columns];
        for k in 0..inner {
            let lhs = lhs[row * inner + k].into();
            let rhs = &rhs[k * columns..(k + 1) * columns];
            // Preserve F64 accumulation and increasing-inner arithmetic order.
            for (output, &rhs) in output.iter_mut().zip(rhs) {
                *output += lhs * rhs.into();
            }
        }
    }
}

impl DynamicTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if data.len() != expected {
            return Err(format!(
                "tensor data length {} does not match shape {:?} with {expected} elements",
                data.len(),
                shape
            ));
        }

        Ok(Self {
            shape,
            data: HostTensorStorage::from_f64(data, TensorDType::F64),
            dtype: TensorDType::F64,
        })
    }

    /// Builds a tensor of `dtype`, rounding each value to that dtype.
    pub fn with_dtype(
        shape: Vec<usize>,
        data: Vec<f64>,
        dtype: TensorDType,
    ) -> Result<Self, String> {
        Self::new(shape, data).map(|tensor| tensor.into_dtype(dtype))
    }

    pub fn filled(shape: Vec<usize>, value: f64) -> Result<Self, String> {
        let count = element_count(&shape)?;
        Ok(Self {
            shape,
            data: HostTensorStorage::from_f64(vec![value; count], TensorDType::F64),
            dtype: TensorDType::F64,
        })
    }

    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Consumes the tensor, transferring its shape, storage, and dtype without copying.
    pub fn into_parts(self) -> (Vec<usize>, HostTensorStorage, TensorDType) {
        (self.shape, self.data, self.dtype)
    }

    /// Converts to `dtype` with round-to-nearest-even (exact for widening).
    pub fn astype(&self, dtype: TensorDType) -> Self {
        self.clone().into_dtype(dtype)
    }

    fn into_dtype(mut self, dtype: TensorDType) -> Self {
        self.data = self.data.into_dtype(dtype);
        self.dtype = dtype;
        self
    }

    fn one_hot(shape: Vec<usize>, index: usize) -> Result<Self, String> {
        let count = element_count(&shape)?;
        if index >= count {
            return Err(format!(
                "one-hot index {index} is out of bounds for {count} elements"
            ));
        }
        let mut data = vec![0.0; count];
        data[index] = 1.0;
        Self::new(shape, data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrows F64 storage; other dtypes widen only for this accessor's lifetime.
    pub fn data(&self) -> Cow<'_, [f64]> {
        self.data.to_f64()
    }

    pub fn storage(&self) -> &HostTensorStorage {
        &self.data
    }

    pub fn from_storage(shape: Vec<usize>, data: HostTensorStorage) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if data.len() != expected {
            return Err(format!(
                "tensor data length {} does not match shape {:?} with {expected} elements",
                data.len(),
                shape
            ));
        }
        let dtype = data.dtype();
        Ok(Self { shape, data, dtype })
    }

    fn add_assign(&mut self, rhs: &Self) -> Result<(), String> {
        if self.shape == rhs.shape
            && !self.data.iter().any(|value| value.is_nan())
            && !rhs.data.iter().any(|value| value.is_nan())
        {
            if let HostTensorStorage::F64(values) = &mut self.data {
                // NaNs retain the original kernel's payload selection on each architecture.
                // Copy shared storage so caller-visible seeds and sibling aliases stay immutable.
                for (value, contribution) in Arc::make_mut(values).iter_mut().zip(rhs.data.iter()) {
                    *value += contribution;
                }
                return Ok(());
            }
        }
        *self = self.add(rhs)?;
        Ok(())
    }

    fn add(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs + rhs)
    }

    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs * rhs)
    }

    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs - rhs)
    }

    fn neg(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| -value).collect(),
        )
    }

    // Division and log follow IEEE 754 (±inf, NaN) like the devices and
    // eager tensors, so a `where` guard can mask an invalid branch.
    fn div(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs / rhs)
    }

    fn greater(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| f64::from(lhs > rhs))
    }

    fn compare(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| f64::from(kind.evaluate(lhs, rhs)))
    }

    fn where_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        let shape = broadcast_shape(
            &broadcast_shape(&self.shape, &on_true.shape)?,
            &on_false.shape,
        )?;
        let count = element_count(&shape)?;
        let condition_strides = contiguous_strides(&self.shape);
        let true_strides = contiguous_strides(&on_true.shape);
        let false_strides = contiguous_strides(&on_false.shape);
        let mut data = Vec::with_capacity(count);

        let mut offsets = [0; 3];
        for index in 0..count {
            data.push(if self.data.get(offsets[0]) != 0.0 {
                on_true.data.get(offsets[1])
            } else {
                on_false.data.get(offsets[2])
            });
            if index + 1 < count {
                advance_broadcast_offsets(
                    index + 1,
                    &shape,
                    [
                        (&self.shape, &condition_strides),
                        (&on_true.shape, &true_strides),
                        (&on_false.shape, &false_strides),
                    ],
                    &mut offsets,
                );
            }
        }

        Self::new(shape, data)
    }

    fn scale(&self, factor: f64) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value * factor).collect(),
        )
    }

    fn sum_all(&self) -> Result<Self, String> {
        Self::new(vec![], vec![self.data.iter().sum()])
    }

    fn reduce_axis(&self, axis: usize, scale: f64) -> Result<Self, String> {
        let output_shape = reduced_shape(&self.shape, axis)?;
        let mut data = vec![0.0; element_count(&output_shape)?];
        let inner = element_count(&self.shape[axis + 1..])?;
        let reduced = self.shape[axis];

        // Contiguous blocks visit source values in the original linear order,
        // retaining +0 initialization and multiplication before each addition.
        for (outer, output_block) in data.chunks_exact_mut(inner).enumerate() {
            for row in 0..reduced {
                let source_start = (outer * reduced + row) * inner;
                for (column, output) in output_block.iter_mut().enumerate() {
                    *output += self.data.get(source_start + column) * scale;
                }
            }
        }

        Self::new(output_shape, data)
    }

    fn expand_reduced_axis(&self, target_shape: &[usize], axis: usize) -> Result<Self, String> {
        let expected_shape = reduced_shape(target_shape, axis)?;
        if self.shape != expected_shape {
            return Err(format!(
                "cannot expand reduced tensor shape {:?} along axis {axis} to {:?}",
                self.shape, target_shape
            ));
        }

        let source_strides = contiguous_strides(&self.shape);
        let mut data = Vec::with_capacity(element_count(target_shape)?);
        for target_index in 0..element_count(target_shape)? {
            let mut remaining = target_index;
            let mut source_index = 0;
            for target_axis in (0..target_shape.len()).rev() {
                let coordinate = remaining % target_shape[target_axis];
                remaining /= target_shape[target_axis];
                if target_axis != axis {
                    let source_axis = if target_axis < axis {
                        target_axis
                    } else {
                        target_axis - 1
                    };
                    source_index += coordinate * source_strides[source_axis];
                }
            }
            data.push(self.data.get(source_index));
        }

        Self::new(target_shape.to_vec(), data)
    }

    fn sqrt(&self) -> Result<Self, String> {
        self.sqrt_derivative(0)
    }

    fn sqrt_derivative(&self, order: u32) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .map(|value| sqrt_derivative_value(value, order))
                .collect(),
        )
    }

    fn reshape(&self, shape: Vec<usize>) -> Result<Self, String> {
        if element_count(&shape)? != self.data.len() {
            return Err(format!(
                "cannot reshape tensor with {} elements to shape {:?}",
                self.data.len(),
                shape
            ));
        }
        Self::from_storage(shape, self.data.clone())
    }

    fn concat(inputs: &[&Self], axis: usize) -> Result<Self, String> {
        let input_shapes = inputs
            .iter()
            .map(|input| input.shape.as_slice())
            .collect::<Vec<_>>();
        let shape = concat_shape(&input_shapes, axis)?;
        let mut data = Vec::with_capacity(element_count(&shape)?);
        let outer = element_count(&shape[..axis])?;
        let inner = element_count(&shape[axis + 1..])?;
        for outer_index in 0..outer {
            for input in inputs {
                let start = outer_index * input.shape[axis] * inner;
                let end = start + input.shape[axis] * inner;
                data.extend((start..end).map(|index| input.data.get(index)));
            }
        }
        Self::new(shape, data)
    }

    /// Returns a contiguous slice along one axis without exposing the eager
    /// storage layout. Data-parallel callers use this to build explicit host
    /// shards before transferring each shard to its assigned device.
    pub fn slice_axis(&self, axis: usize, start: usize, length: usize) -> Result<Self, String> {
        if axis >= self.shape.len()
            || start
                .checked_add(length)
                .is_none_or(|end| end > self.shape[axis])
        {
            return Err(format!(
                "invalid slice [{start}..{}) on axis {axis} for shape {:?}",
                start + length,
                self.shape
            ));
        }
        let mut shape = self.shape.clone();
        shape[axis] = length;
        let outer = element_count(&self.shape[..axis])?;
        let inner = element_count(&self.shape[axis + 1..])?;
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for outer_index in 0..outer {
            let source_start = (outer_index * self.shape[axis] + start) * inner;
            let source_end = source_start + length * inner;
            data.extend((source_start..source_end).map(|index| self.data.get(index)));
        }
        Self::new(shape, data).map(|slice| slice.into_dtype(self.dtype))
    }

    fn pad_slice(&self, output_shape: &[usize], axis: usize, start: usize) -> Result<Self, String> {
        if axis >= output_shape.len() || self.shape.len() != output_shape.len() {
            return Err(format!(
                "cannot pad tensor shape {:?} into {:?} along axis {axis}",
                self.shape, output_shape
            ));
        }
        for (input_extent, output_extent) in self.shape.iter().zip(output_shape) {
            if input_extent != output_extent && input_extent != &self.shape[axis] {
                return Err(format!(
                    "cannot pad tensor shape {:?} into {:?} along axis {axis}",
                    self.shape, output_shape
                ));
            }
        }
        if self
            .shape
            .iter()
            .enumerate()
            .any(|(index, extent)| index != axis && *extent != output_shape[index])
            || start
                .checked_add(self.shape[axis])
                .is_none_or(|end| end > output_shape[axis])
        {
            return Err(format!(
                "cannot pad tensor shape {:?} into {:?} along axis {axis}",
                self.shape, output_shape
            ));
        }
        let outer = element_count(&output_shape[..axis])?;
        let inner = element_count(&output_shape[axis + 1..])?;
        let mut data = vec![0.0; element_count(output_shape)?];
        for outer_index in 0..outer {
            let source_start = outer_index * self.shape[axis] * inner;
            let destination_start = (outer_index * output_shape[axis] + start) * inner;
            let width = self.shape[axis] * inner;
            for (offset, value) in data[destination_start..destination_start + width]
                .iter_mut()
                .enumerate()
            {
                *value = self.data.get(source_start + offset);
            }
        }
        Self::new(output_shape.to_vec(), data)
    }

    /// Copies the entries at `indices` along `axis`; see [`TensorOp::Gather`].
    fn gather_axis(&self, indices: &[usize], axis: usize) -> Result<Self, String> {
        let extent = *self
            .shape
            .get(axis)
            .ok_or_else(|| format!("gather axis {axis} is out of bounds for {:?}", self.shape))?;
        if let Some(index) = indices.iter().find(|index| **index >= extent) {
            return Err(format!(
                "gather index {index} is out of bounds for axis {axis} with extent {extent}"
            ));
        }
        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        let outer = element_count(&self.shape[..axis])?;
        let inner = element_count(&self.shape[axis + 1..])?;
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for outer_index in 0..outer {
            for index in indices {
                let start = (outer_index * extent + index) * inner;
                data.extend((start..start + inner).map(|position| self.data.get(position)));
            }
        }
        Self::new(shape, data).map(|gathered| gathered.into_dtype(self.dtype))
    }

    /// `self` with `updates` added at `indices` along `axis`, rounding every
    /// sum to `rounding` (`F64` for derivative arithmetic). Contributions to
    /// one destination accumulate in increasing update order, or decreasing
    /// order with `reverse`; see [`TensorOp::ScatterAdd`].
    fn scatter_add_axis(
        &self,
        updates: &Self,
        indices: &[usize],
        axis: usize,
        rounding: TensorDType,
        reverse: bool,
    ) -> Result<Self, String> {
        let extent = *self
            .shape
            .get(axis)
            .ok_or_else(|| format!("scatter axis {axis} is out of bounds for {:?}", self.shape))?;
        let mut expected = self.shape.clone();
        expected[axis] = indices.len();
        if updates.shape != expected || indices.iter().any(|index| *index >= extent) {
            return Err(format!(
                "cannot scatter updates of shape {:?} into {:?} along axis {axis} at {} indices",
                updates.shape,
                self.shape,
                indices.len()
            ));
        }
        let outer = element_count(&self.shape[..axis])?;
        let inner = element_count(&self.shape[axis + 1..])?;
        let mut data = self.data.to_vec();
        let count = indices.len();
        for outer_index in 0..outer {
            for step in 0..count {
                let source = if reverse { count - 1 - step } else { step };
                let destination = (outer_index * extent + indices[source]) * inner;
                let source = (outer_index * count + source) * inner;
                for offset in 0..inner {
                    let slot = &mut data[destination + offset];
                    *slot = rounding.round(*slot + updates.data.get(source + offset));
                }
            }
        }
        Self::new(self.shape.clone(), data)
    }

    fn mean_all(&self) -> Result<Self, String> {
        self.sum_all()?.scale(1.0 / self.data.len() as f64)
    }

    fn powi(&self, exponent: u32) -> Result<Self, String> {
        let exponent = i32::try_from(exponent)
            .map_err(|_| "powi exponent must fit in a signed 32-bit integer".to_string())?;
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.powi(exponent)).collect(),
        )
    }

    fn map_f64(&self, f: impl Fn(f64) -> f64) -> Result<Self, String> {
        Self::new(self.shape.clone(), self.data.iter().map(f).collect())
    }

    /// Inclusive prefix sums along `axis` (from the last entry when
    /// `reverse`), rounding every running sum to `rounding` (`F64` for
    /// derivative arithmetic); see [`TensorOp::CumSum`].
    fn cumsum_axis(
        &self,
        axis: usize,
        reverse: bool,
        rounding: TensorDType,
    ) -> Result<Self, String> {
        let extent = *self
            .shape
            .get(axis)
            .ok_or_else(|| format!("cumsum axis {axis} is out of bounds for {:?}", self.shape))?;
        let outer = element_count(&self.shape[..axis])?;
        let inner = element_count(&self.shape[axis + 1..])?;
        let mut data = self.data.to_vec();
        for outer_index in 0..outer {
            for lane in 0..inner {
                let position = |step: usize| {
                    let row = if reverse { extent - 1 - step } else { step };
                    (outer_index * extent + row) * inner + lane
                };
                let mut running = rounding.round(data[position(0)]);
                data[position(0)] = running;
                for step in 1..extent {
                    running = rounding.round(running + data[position(step)]);
                    data[position(step)] = running;
                }
            }
        }
        Self::new(self.shape.clone(), data)
    }

    fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        let shape = matmul_shape(&self.shape, &rhs.shape)?;
        let lhs_rows = self.shape[self.shape.len() - 2];
        let lhs_inner = self.shape[self.shape.len() - 1];
        let rhs_cols = rhs.shape[rhs.shape.len() - 1];
        let lhs_batch_shape = &self.shape[..self.shape.len() - 2];
        let rhs_batch_shape = &rhs.shape[..rhs.shape.len() - 2];
        let batch_shape = &shape[..shape.len() - 2];
        let batch_count = element_count(batch_shape)?;
        let lhs_batch_strides = contiguous_strides(lhs_batch_shape);
        let rhs_batch_strides = contiguous_strides(rhs_batch_shape);
        let mut data = vec![0.0; element_count(&shape)?];

        for batch_index in 0..batch_count {
            let lhs_batch = broadcast_offset(
                batch_index,
                batch_shape,
                lhs_batch_shape,
                &lhs_batch_strides,
            );
            let rhs_batch = broadcast_offset(
                batch_index,
                batch_shape,
                rhs_batch_shape,
                &rhs_batch_strides,
            );
            let lhs_start = lhs_batch * lhs_rows * lhs_inner;
            let rhs_start = rhs_batch * lhs_inner * rhs_cols;
            let output_start = batch_index * lhs_rows * rhs_cols;
            let output = &mut data[output_start..output_start + lhs_rows * rhs_cols];
            let dimensions = [lhs_rows, lhs_inner, rhs_cols];
            match (&self.data, &rhs.data) {
                (HostTensorStorage::F64(lhs), HostTensorStorage::F64(rhs)) => {
                    matmul_host_float_block(
                        &lhs[lhs_start..],
                        &rhs[rhs_start..],
                        output,
                        dimensions,
                    )
                }
                (HostTensorStorage::F32(lhs), HostTensorStorage::F32(rhs)) => {
                    matmul_host_float_block(
                        &lhs[lhs_start..],
                        &rhs[rhs_start..],
                        output,
                        dimensions,
                    )
                }
                (HostTensorStorage::F32(lhs), HostTensorStorage::F64(rhs)) => {
                    matmul_host_float_block(
                        &lhs[lhs_start..],
                        &rhs[rhs_start..],
                        output,
                        dimensions,
                    )
                }
                (HostTensorStorage::F64(lhs), HostTensorStorage::F32(rhs)) => {
                    matmul_host_float_block(
                        &lhs[lhs_start..],
                        &rhs[rhs_start..],
                        output,
                        dimensions,
                    )
                }
                _ => {
                    for row in 0..lhs_rows {
                        let lhs_row_start = lhs_batch * lhs_rows * lhs_inner + row * lhs_inner;
                        let output_row_start = batch_index * lhs_rows * rhs_cols + row * rhs_cols;
                        for inner in 0..lhs_inner {
                            let lhs_value = self.data.get(lhs_row_start + inner);
                            let rhs_row_start = rhs_batch * lhs_inner * rhs_cols + inner * rhs_cols;
                            for col in 0..rhs_cols {
                                data[output_row_start + col] +=
                                    lhs_value * rhs.data.get(rhs_row_start + col);
                            }
                        }
                    }
                }
            }
        }

        Self::new(shape, data)
    }

    /// Solves `self @ x == rhs` for every matrix of the leading batch axes
    /// (see [`solve_shape`]); each matrix is solved independently, so one
    /// batch element's pivots never affect another's.
    fn solve(&self, rhs: &Self) -> Result<Self, String> {
        solve_shape(&self.shape, &rhs.shape)?;
        if self.shape.len() == 2 {
            return self.solve_matrix(rhs);
        }
        let rank = self.shape.len();
        let (n, columns) = (self.shape[rank - 1], rhs.shape[rank - 1]);
        let batch = element_count(&self.shape[..rank - 2])?;
        let (matrices, rhs_data) = (self.data(), rhs.data());
        let mut output = Vec::with_capacity(rhs_data.len());
        for index in 0..batch {
            let matrix = Self::new(
                vec![n, n],
                matrices[index * n * n..(index + 1) * n * n].to_vec(),
            )?;
            let block = Self::new(
                vec![n, columns],
                rhs_data[index * n * columns..(index + 1) * n * columns].to_vec(),
            )?;
            output.extend(matrix.solve_matrix(&block)?.data().iter());
        }
        Self::new(rhs.shape.clone(), output)
    }

    fn solve_matrix(&self, rhs: &Self) -> Result<Self, String> {
        if let Some(result) = self.finite_triangular_solution(rhs) {
            return Self::new(rhs.shape.clone(), result);
        }
        self.solve_lu(rhs)
    }

    fn solve_lu(&self, rhs: &Self) -> Result<Self, String> {
        let n = self.shape[0];
        let columns = rhs.shape[1];
        let mut factor = self.data.to_vec();
        let mut result = rhs.data.to_vec();

        for pivot in 0..n {
            let pivot_row = (pivot..n)
                .max_by(|&left, &right| {
                    factor[left * n + pivot]
                        .abs()
                        .total_cmp(&factor[right * n + pivot].abs())
                })
                .expect("pivot range is non-empty");
            if factor[pivot_row * n + pivot] == 0.0 {
                return Err("solve requires a non-singular coefficient matrix".to_string());
            }
            if pivot_row != pivot {
                for column in 0..n {
                    factor.swap(pivot * n + column, pivot_row * n + column);
                }
                for column in 0..columns {
                    result.swap(pivot * columns + column, pivot_row * columns + column);
                }
            }
            let diagonal = factor[pivot * n + pivot];
            for row in pivot + 1..n {
                let multiplier = factor[row * n + pivot] / diagonal;
                factor[row * n + pivot] = multiplier;
                for column in pivot + 1..n {
                    factor[row * n + column] -= multiplier * factor[pivot * n + column];
                }
                for column in 0..columns {
                    result[row * columns + column] -= multiplier * result[pivot * columns + column];
                }
            }
        }
        for row in (0..n).rev() {
            let diagonal = factor[row * n + row];
            for column in 0..columns {
                let mut value = result[row * columns + column];
                for inner in row + 1..n {
                    value -= factor[row * n + inner] * result[inner * columns + column];
                }
                result[row * columns + column] = value / diagonal;
            }
        }
        Self::new(rhs.shape.clone(), result)
    }

    // Exact triangular structure permits substitution without a factor matrix.
    // Exceptional values keep the pivoted solver's existing error/IEEE behavior.
    fn finite_triangular_solution(&self, rhs: &Self) -> Option<Vec<f64>> {
        if self
            .data
            .iter()
            .chain(rhs.data.iter())
            .any(|value| !value.is_finite())
        {
            return None;
        }
        let n = self.shape[0];
        let columns = rhs.shape[1];
        let mut lower = true;
        let mut upper = true;
        for row in 0..n {
            if self.data.get(row * n + row) == 0.0 {
                return None;
            }
            for column in 0..row {
                upper &= self.data.get(row * n + column) == 0.0;
                lower &= self.data.get(column * n + row) == 0.0;
            }
        }
        if !lower && !upper {
            return None;
        }
        let mut result = rhs.data.to_vec();
        for step in 0..n {
            let row = if lower { step } else { n - 1 - step };
            let diagonal = self.data.get(row * n + row);
            let dependencies = if lower { 0..row } else { row + 1..n };
            for column in 0..columns {
                let mut value = result[row * columns + column];
                for inner in dependencies.clone() {
                    value -= self.data.get(row * n + inner) * result[inner * columns + column];
                    if !value.is_finite() {
                        return None;
                    }
                }
                let solved = value / diagonal;
                if !solved.is_finite() {
                    return None;
                }
                result[row * columns + column] = solved;
            }
        }
        Some(result)
    }

    fn cholesky(&self) -> Result<Self, String> {
        let n = self.shape[self.shape.len() - 1];
        let mut output = Vec::with_capacity(self.data.len());
        for input in self.data().chunks_exact(n * n) {
            let mut values = vec![0.0; n * n];
            for row in 0..n {
                for column in 0..=row {
                    let mut reduced = input[row * n + column];
                    for inner in 0..column {
                        let product = self
                            .dtype
                            .round(values[row * n + inner] * values[column * n + inner]);
                        reduced = self.dtype.round(reduced - product);
                    }
                    values[row * n + column] = self.dtype.round(if row == column {
                        sqrt_derivative_value(reduced, 0)
                    } else {
                        reduced / values[column * n + column]
                    });
                }
            }
            output.extend(values);
        }
        Self::with_dtype(self.shape.clone(), output, self.dtype)
    }

    fn triangular(&self, lower: bool) -> Result<Self, String> {
        if self.shape.len() < 2 {
            return Err(format!(
                "triangular projection requires at least rank two, got {:?}",
                self.shape
            ));
        }
        let rows = self.shape[self.shape.len() - 2];
        let columns = self.shape[self.shape.len() - 1];
        let matrix_size = rows
            .checked_mul(columns)
            .ok_or_else(|| "triangular matrix size overflows usize".to_string())?;
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let index = index % matrix_size;
                let row = index / columns;
                let column = index % columns;
                if (lower && row >= column) || (!lower && row <= column) {
                    value
                } else {
                    0.0
                }
            })
            .collect();
        Self::new(self.shape.clone(), data)
    }

    fn transpose_last_two(&self) -> Result<Self, String> {
        if self.shape.len() < 2 {
            return Err(format!(
                "transpose_last_two requires a tensor with at least two dimensions, got {:?}",
                self.shape
            ));
        }

        let rows = self.shape[self.shape.len() - 2];
        let cols = self.shape[self.shape.len() - 1];
        let batch_count = element_count(&self.shape[..self.shape.len() - 2])?;
        let mut shape = self.shape.clone();
        let last_axis = shape.len() - 1;
        shape.swap(last_axis - 1, last_axis);
        let mut data = vec![0.0; self.data.len()];
        for batch in 0..batch_count {
            for row in 0..rows {
                for col in 0..cols {
                    data[batch * cols * rows + col * rows + row] =
                        self.data.get(batch * rows * cols + row * cols + col);
                }
            }
        }

        Self::new(shape, data)
    }

    fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
        validate_permutation(axes, self.shape.len())?;
        let output_shape = axes
            .iter()
            .map(|axis| self.shape[*axis])
            .collect::<Vec<_>>();
        let input_strides = contiguous_strides(&self.shape);
        let mut data = vec![0.0; self.data.len()];

        for (output_index, output_value) in data.iter_mut().enumerate() {
            let mut remaining = output_index;
            let mut input_index = 0;
            for output_axis in (0..output_shape.len()).rev() {
                let coordinate = remaining % output_shape[output_axis];
                remaining /= output_shape[output_axis];
                input_index += coordinate * input_strides[axes[output_axis]];
            }
            *output_value = self.data.get(input_index);
        }

        Self::new(output_shape, data)
    }

    /// Returns a materialized permutation of the tensor axes.
    ///
    /// This is intentionally a small public boundary for transform wrappers;
    /// compiled execution plans remain responsible for backend lowering.
    pub fn permute(&self, axes: &[usize]) -> Result<Self, String> {
        self.transpose(axes)
            .map(|permuted| permuted.into_dtype(self.dtype))
    }

    fn broadcast_to_shape(&self, target_shape: &[usize]) -> Result<Self, String> {
        if broadcast_shape(&self.shape, target_shape)? != target_shape {
            return Err(format!(
                "cannot broadcast tensor shape {:?} to {:?}",
                self.shape, target_shape
            ));
        }

        let count = element_count(target_shape)?;
        let strides = contiguous_strides(&self.shape);
        let mut data = Vec::with_capacity(count);
        let mut offsets = [0];
        for index in 0..count {
            data.push(self.data.get(offsets[0]));
            if index + 1 < count {
                advance_broadcast_offsets(
                    index + 1,
                    target_shape,
                    [(&self.shape, &strides)],
                    &mut offsets,
                );
            }
        }
        Self::new(target_shape.to_vec(), data)
    }

    fn elementwise(&self, rhs: &Self, f: impl Fn(f64, f64) -> f64) -> Result<Self, String> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let count = element_count(&shape)?;
        if self.shape == rhs.shape {
            let data = self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(lhs, rhs)| f(lhs, rhs))
                .collect();
            return Self::new(shape, data);
        }
        // A one-element operand broadcasts without changing the other operand's
        // linear order, including additional leading singleton dimensions.
        if self.data.len() == 1 && rhs.data.len() == count {
            let data = rhs
                .data
                .iter()
                .map(|rhs| f(self.data.get(0), rhs))
                .collect();
            return Self::new(shape, data);
        }
        if rhs.data.len() == 1 && self.data.len() == count {
            let data = self
                .data
                .iter()
                .map(|lhs| f(lhs, rhs.data.get(0)))
                .collect();
            return Self::new(shape, data);
        }
        let lhs_strides = contiguous_strides(&self.shape);
        let rhs_strides = contiguous_strides(&rhs.shape);
        let mut data = Vec::with_capacity(count);

        let mut offsets = [0; 2];
        for index in 0..count {
            data.push(f(self.data.get(offsets[0]), rhs.data.get(offsets[1])));
            if index + 1 < count {
                advance_broadcast_offsets(
                    index + 1,
                    &shape,
                    [(&self.shape, &lhs_strides), (&rhs.shape, &rhs_strides)],
                    &mut offsets,
                );
            }
        }

        Self::new(shape, data)
    }

    fn reduce_to_shape(&self, target_shape: &[usize]) -> Result<Self, String> {
        if target_shape.len() > self.shape.len() {
            return Err(format!(
                "cannot reduce tensor shape {:?} to higher-rank shape {:?}",
                self.shape, target_shape
            ));
        }

        let rank_offset = self.shape.len() - target_shape.len();
        for (source_extent, target_extent) in self.shape[rank_offset..].iter().zip(target_shape) {
            if source_extent != target_extent && *target_extent != 1 {
                return Err(format!(
                    "cannot reduce tensor shape {:?} to {:?}",
                    self.shape, target_shape
                ));
            }
        }

        let target_count = element_count(target_shape)?;
        let target_strides = contiguous_strides(target_shape);
        let mut data = vec![0.0; target_count];

        for (index, value) in self.data.iter().enumerate() {
            let mut remaining = index;
            let mut target_index = 0;
            for axis in (0..self.shape.len()).rev() {
                let coordinate = remaining % self.shape[axis];
                remaining /= self.shape[axis];
                if axis >= rank_offset {
                    let target_axis = axis - rank_offset;
                    if target_shape[target_axis] != 1 {
                        target_index += coordinate * target_strides[target_axis];
                    }
                }
            }
            data[target_index] += value;
        }

        Self::new(target_shape.to_vec(), data)
    }
}

impl TensorIr {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of nodes in the graph, including nodes no output depends on;
    /// compiled plans drop those and common pure subexpressions.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Splices the part of `callee` that `outputs` depend on into this graph
    /// and returns the ids of the spliced outputs, in order.
    ///
    /// This is how a transformed function is called inside another traced
    /// function: the callee is staged in its own graph, then every callee
    /// `Input` reached from `outputs` is replaced by the node that `bindings`
    /// names for it, and the other reached nodes are appended with their
    /// shape, dtype, and weak flag unchanged. A binding must have the shape
    /// and dtype of its input; a weak binding keeps its weakness, while the
    /// spliced nodes keep the flags the callee derived from a strong input.
    /// Callee nodes that no output depends on are not copied, and nodes equal
    /// to existing ones are not merged here: plan compilation removes common
    /// pure subexpressions, as for any traced graph.
    ///
    /// Region nodes (`Cond`, `Fori`, `Scan`, and their derivative nodes)
    /// clone their compiled region plans and rebind their captures like any
    /// other operand. Nodes that share one execution through a `group` id
    /// receive a new id, the id of the first spliced node of the group, as
    /// [`Self::scan`] assigns them. Every group id of a graph is below its
    /// node count, so the spliced groups never merge with existing or later
    /// groups, and two splices of one callee stay separate executions.
    pub fn inline(
        &mut self,
        callee: &TensorIr,
        bindings: &BTreeMap<String, TensorNodeId>,
        outputs: &[TensorNodeId],
    ) -> Result<Vec<TensorNodeId>, String> {
        let mut callee_inputs = HashMap::new();
        for (id, node) in callee.nodes.iter().enumerate() {
            if let TensorOp::Input { name } = &node.op {
                callee_inputs.insert(name.as_str(), id);
            }
        }
        for (name, binding) in bindings {
            let input = callee_inputs
                .get(name.as_str())
                .map(|id| &callee.nodes[*id])
                .ok_or_else(|| format!("inline binds {name:?}, which is not a callee input"))?;
            let bound = self.node(*binding)?;
            if bound.shape != input.shape || bound.dtype != input.dtype {
                return Err(format!(
                    "inline binds callee input {name:?} ({}{:?}) to node {binding} ({}{:?}); \
                     the shape and dtype must match",
                    input.dtype, input.shape, bound.dtype, bound.shape
                ));
            }
        }

        let reachable = callee.reachable_from(outputs)?;
        let mut remap = HashMap::new();
        let mut groups = HashMap::new();
        for (id, node) in callee.nodes.iter().enumerate() {
            if !reachable[id] {
                continue;
            }
            let spliced = if let TensorOp::Input { name } = &node.op {
                *bindings
                    .get(name)
                    .ok_or_else(|| format!("inline leaves callee input {name:?} unbound"))?
            } else {
                self.push_spliced(node, &remap, &mut groups)?
            };
            remap.insert(id, spliced);
        }
        Ok(outputs.iter().map(|output| remap[output]).collect())
    }

    /// Splices the part of `callee` that `outputs` depend on into this graph
    /// like [`Self::inline`], vectorized over `batch_size` examples: this is
    /// the lowering of `vmap`.
    ///
    /// `callee` computes one example. Each binding is `(node, mapped)`: an
    /// unmapped binding has the shape of its callee input and is shared by
    /// every example, while a mapped binding has shape
    /// `[batch_size, *input shape]` and holds one example per index of its
    /// leading axis. A spliced node that depends on a mapped binding is
    /// mapped, with the batch as its leading axis; the other nodes are copied
    /// as `inline` copies them, so work that does not depend on the mapped
    /// arguments is done once, not per example. Returns `(node, mapped)` per
    /// output.
    ///
    /// Batching rules, with the batch axis always leading: elementwise ops,
    /// `where`, `matmul`, and `broadcast` first insert unit axes after the
    /// batch axis of a mapped operand whose example rank is below the node's,
    /// because broadcasting aligns trailing axes and would otherwise align
    /// the batch axis of a `[B]` operand with an example axis of the other
    /// operand; ops with an axis shift it by one; full reductions reduce the
    /// flattened example axes; `concat` broadcasts its unmapped operands over
    /// the batch. The result is an ordinary graph, so it can be batched again
    /// (nested `vmap`), differentiated, and inlined. `solve` broadcasts an
    /// unmapped operand over the batch like `concat`. Loop region nodes
    /// (`fori`, `scan`, and their JVP, VJP, and forward-over-reverse nodes,
    /// and `while`) batch their body regions recursively (see
    /// `region_batching.rs`); a loop result that does not depend on a mapped
    /// operand stays unmapped, a scan's stacked outputs are `[B, T, *y]`, and
    /// a `while` with a mapped predicate runs until every example is done,
    /// freezing finished examples' carries. A `cond` batches both branch
    /// regions under an unmapped predicate and becomes a per-example `where`
    /// of both branches under a mapped one. Unmapped region nodes are copied
    /// unchanged.
    pub fn inline_batched(
        &mut self,
        callee: &TensorIr,
        bindings: &BTreeMap<String, (TensorNodeId, bool)>,
        batch_size: usize,
        outputs: &[TensorNodeId],
    ) -> Result<Vec<(TensorNodeId, bool)>, BatchingError> {
        if batch_size == 0 {
            return Err(BatchingError::Invalid(
                "vmap requires a batch size above zero".to_string(),
            ));
        }
        let mut callee_inputs = HashMap::new();
        for (id, node) in callee.nodes.iter().enumerate() {
            if let TensorOp::Input { name } = &node.op {
                callee_inputs.insert(name.as_str(), id);
            }
        }
        for (name, (binding, mapped)) in bindings {
            let input = callee_inputs
                .get(name.as_str())
                .map(|id| &callee.nodes[*id])
                .ok_or_else(|| format!("inline binds {name:?}, which is not a callee input"))?;
            let bound = self.node(*binding)?;
            let expected = if *mapped {
                batched_shape(batch_size, &input.shape)
            } else {
                input.shape.clone()
            };
            if bound.shape != expected || bound.dtype != input.dtype {
                return Err(BatchingError::Invalid(format!(
                    "vmap binds callee input {name:?} ({}{:?}, {}) to node {binding} ({}{:?}); \
                     a mapped binding must have shape [{batch_size}, *input shape] and an \
                     unmapped one the input shape, both with the input dtype",
                    input.dtype,
                    input.shape,
                    if *mapped { "mapped" } else { "unmapped" },
                    bound.dtype,
                    bound.shape
                )));
            }
        }

        let reachable = callee.reachable_from(outputs)?;
        let mut remap = HashMap::new();
        let mut mapped = vec![false; callee.nodes.len()];
        let mut groups = HashMap::new();
        let mut custom_groups = HashMap::new();
        let mut loop_groups = HashMap::new();
        for (id, node) in callee.nodes.iter().enumerate() {
            if !reachable[id] {
                continue;
            }
            let spliced = if let TensorOp::Input { name } = &node.op {
                let (binding, is_mapped) = *bindings
                    .get(name)
                    .ok_or_else(|| format!("inline leaves callee input {name:?} unbound"))?;
                mapped[id] = is_mapped;
                binding
            } else if tensor_op_inputs(&node.op)
                .iter()
                .any(|input| mapped[*input])
            {
                let (spliced, is_mapped) = self.push_batched(
                    callee,
                    node,
                    (&remap, &mapped),
                    batch_size,
                    (&mut custom_groups, &mut loop_groups),
                )?;
                mapped[id] = is_mapped;
                spliced
            } else {
                self.push_spliced(node, &remap, &mut groups)?
            };
            remap.insert(id, spliced);
        }
        Ok(outputs
            .iter()
            .map(|output| (remap[output], mapped[*output]))
            .collect())
    }

    /// The callee nodes that `outputs` depend on, by callee node id.
    fn reachable_from(&self, outputs: &[TensorNodeId]) -> Result<Vec<bool>, String> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut pending = outputs.to_vec();
        while let Some(id) = pending.pop() {
            let node = self.node(id)?;
            if !std::mem::replace(&mut reachable[id], true) {
                pending.extend(tensor_op_inputs(&node.op));
            }
        }
        Ok(reachable)
    }

    /// Appends a copy of a callee node with its operands remapped; a group id
    /// becomes the id of the group's first spliced node (see [`Self::inline`]).
    fn push_spliced(
        &mut self,
        node: &TensorNode,
        remap: &HashMap<TensorNodeId, TensorNodeId>,
        groups: &mut HashMap<usize, usize>,
    ) -> Result<TensorNodeId, String> {
        let mut op = remap_tensor_op(&node.op, &|id| remap.get(&id).copied())?;
        if let Some(group) = tensor_op_group_mut(&mut op) {
            *group = *groups.entry(*group).or_insert(self.nodes.len());
        }
        Ok(self.push_node(op, node.shape.clone(), node.dtype, node.weak))
    }

    /// Appends the batched form of a callee node with at least one mapped
    /// operand (see [`Self::inline_batched`]) and returns it with whether it
    /// is mapped; the node's dtype and weak flag are kept, and its shape gains
    /// the leading batch axis. Only a region node can stay unmapped.
    ///
    /// A `Custom` node is batched with its rule (`TensorCustomRule::batched`),
    /// once per call: `custom_groups` maps a callee group id to the batched
    /// group id and rule. Loop region nodes are batched by
    /// `push_batched_loop`, once per multi-result group (`loop_groups`).
    fn push_batched(
        &mut self,
        callee: &TensorIr,
        node: &TensorNode,
        (remap, mapped): (&HashMap<TensorNodeId, TensorNodeId>, &[bool]),
        batch_size: usize,
        (custom_groups, loop_groups): (
            &mut HashMap<usize, (usize, Arc<TensorCustomRule>)>,
            &mut region_batching::LoopBatchGroups,
        ),
    ) -> Result<(TensorNodeId, bool), BatchingError> {
        let target = |operand: TensorNodeId| {
            remap
                .get(&operand)
                .copied()
                .ok_or_else(|| format!("node {operand} is missing from the vmap remap"))
        };
        let rank = node.shape.len();
        let op = match &node.op {
            TensorOp::Cast { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Powi { .. }
            | TensorOp::UnaryMath { .. }
            | TensorOp::StopGradient { .. }
            | TensorOp::Triangular { .. }
            | TensorOp::Linalg { .. }
            | TensorOp::Reshape { .. } => remap_tensor_op(&node.op, &|id| remap.get(&id).copied())?,
            TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Div { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. }
            | TensorOp::BinaryMath { .. }
            | TensorOp::Where { .. }
            | TensorOp::Matmul { .. }
            | TensorOp::Broadcast { .. } => {
                let mut local = HashMap::new();
                for operand in tensor_op_inputs(&node.op) {
                    let spliced = target(operand)?;
                    let spliced = if mapped[operand] {
                        self.pad_batched(spliced, rank)?
                    } else {
                        spliced
                    };
                    local.insert(operand, spliced);
                }
                remap_tensor_op(&node.op, &|id| local.get(&id).copied())?
            }
            TensorOp::Sum { input } | TensorOp::Mean { input } => {
                let count = element_count(&callee.node(*input)?.shape)?;
                let spliced = target(*input)?;
                let source = self.node(spliced)?;
                let (dtype, weak) = (source.dtype, source.weak);
                let flat = self.push_node(
                    TensorOp::Reshape { input: spliced },
                    vec![batch_size, count],
                    dtype,
                    weak,
                );
                match node.op {
                    TensorOp::Sum { .. } => TensorOp::SumAxis {
                        input: flat,
                        axis: 1,
                    },
                    _ => TensorOp::MeanAxis {
                        input: flat,
                        axis: 1,
                    },
                }
            }
            TensorOp::SumAxis { input, axis } => TensorOp::SumAxis {
                input: target(*input)?,
                axis: axis + 1,
            },
            TensorOp::ExtremumAxis { input, axis, kind } => TensorOp::ExtremumAxis {
                input: target(*input)?,
                axis: axis + 1,
                kind: *kind,
            },
            TensorOp::MeanAxis { input, axis } => TensorOp::MeanAxis {
                input: target(*input)?,
                axis: axis + 1,
            },
            TensorOp::CumSum {
                input,
                axis,
                reverse,
            } => TensorOp::CumSum {
                input: target(*input)?,
                axis: axis + 1,
                reverse: *reverse,
            },
            TensorOp::Transpose { input, axes } => TensorOp::Transpose {
                input: target(*input)?,
                axes: std::iter::once(0)
                    .chain(axes.iter().map(|axis| axis + 1))
                    .collect(),
            },
            TensorOp::Concat { inputs, axis } => {
                let mut spliced_inputs = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let spliced = target(*input)?;
                    spliced_inputs.push(if mapped[*input] {
                        spliced
                    } else {
                        let source = self.node(spliced)?;
                        let (shape, dtype, weak) = (
                            batched_shape(batch_size, &source.shape),
                            source.dtype,
                            source.weak,
                        );
                        self.push_node(TensorOp::Broadcast { input: spliced }, shape, dtype, weak)
                    });
                }
                TensorOp::Concat {
                    inputs: spliced_inputs,
                    axis: axis + 1,
                }
            }
            TensorOp::Slice {
                input,
                axis,
                start,
                length,
            } => TensorOp::Slice {
                input: target(*input)?,
                axis: axis + 1,
                start: *start,
                length: *length,
            },
            TensorOp::PadSlice { input, axis, start } => TensorOp::PadSlice {
                input: target(*input)?,
                axis: axis + 1,
                start: *start,
            },
            TensorOp::Gather {
                input,
                indices,
                axis,
            } => TensorOp::Gather {
                input: target(*input)?,
                indices: indices.clone(),
                axis: axis + 1,
            },
            TensorOp::ScatterAdd {
                base,
                updates,
                indices,
                axis,
            } => {
                // Both operands need the batch axis; an unmapped one is the
                // same for every example.
                let mut operands = [*base, *updates];
                for operand in &mut operands {
                    let spliced = target(*operand)?;
                    *operand = if mapped[*operand] {
                        spliced
                    } else {
                        let source = self.node(spliced)?;
                        let (shape, dtype, weak) = (
                            batched_shape(batch_size, &source.shape),
                            source.dtype,
                            source.weak,
                        );
                        self.push_node(TensorOp::Broadcast { input: spliced }, shape, dtype, weak)
                    };
                }
                TensorOp::ScatterAdd {
                    base: operands[0],
                    updates: operands[1],
                    indices: indices.clone(),
                    axis: axis + 1,
                }
            }
            TensorOp::Cholesky { input } => TensorOp::Cholesky {
                input: target(*input)?,
            },
            TensorOp::CholeskyAd { inputs, kind } => {
                let mut arguments = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let argument = target(*input)?;
                    arguments.push(if mapped[*input] {
                        argument
                    } else {
                        let source = self.node(argument)?;
                        self.push_node(
                            TensorOp::Broadcast { input: argument },
                            batched_shape(batch_size, &source.shape),
                            source.dtype,
                            source.weak,
                        )
                    });
                }
                TensorOp::CholeskyAd {
                    inputs: arguments,
                    kind: *kind,
                }
            }
            TensorOp::Custom {
                value,
                operands,
                rule,
                output,
                group,
            } => {
                // Every output of a batched call is batched, so the primal
                // value is broadcast when it does not depend on the batch.
                let mut spliced = target(*value)?;
                if !mapped[*value] {
                    spliced = self.broadcast_to(spliced, batched_shape(batch_size, &node.shape))?;
                }
                let (batched_group, batched_rule) = match custom_groups.get(group) {
                    Some(existing) => existing.clone(),
                    None => {
                        let operand_mapped = operands
                            .iter()
                            .map(|operand| mapped[*operand])
                            .collect::<Vec<_>>();
                        let entry = (
                            self.nodes.len(),
                            Arc::new(rule.batched(&operand_mapped, batch_size)?),
                        );
                        custom_groups.insert(*group, entry.clone());
                        entry
                    }
                };
                TensorOp::Custom {
                    value: spliced,
                    operands: operands
                        .iter()
                        .map(|operand| target(*operand))
                        .collect::<Result<_, _>>()?,
                    rule: batched_rule,
                    output: *output,
                    group: batched_group,
                }
            }
            TensorOp::Solve { matrix, rhs } => {
                // `Solve` batches over equal leading axes; an unmapped
                // operand is the same system or right-hand side for every
                // example.
                let mut operands = [*matrix, *rhs];
                for operand in &mut operands {
                    let spliced = target(*operand)?;
                    *operand = if mapped[*operand] {
                        spliced
                    } else {
                        let source = self.node(spliced)?;
                        let (shape, dtype, weak) = (
                            batched_shape(batch_size, &source.shape),
                            source.dtype,
                            source.weak,
                        );
                        self.push_node(TensorOp::Broadcast { input: spliced }, shape, dtype, weak)
                    };
                }
                TensorOp::Solve {
                    matrix: operands[0],
                    rhs: operands[1],
                }
            }
            TensorOp::Fori { .. }
            | TensorOp::ForiJvp { .. }
            | TensorOp::ForiVjp { .. }
            | TensorOp::ForiVjpJvp { .. }
            | TensorOp::Scan { .. }
            | TensorOp::ScanVjp { .. }
            | TensorOp::ScanVjpJvp { .. }
            | TensorOp::While { .. } => {
                return self.push_batched_loop(node, (remap, mapped), batch_size, loop_groups)
            }
            TensorOp::Cond { .. } => {
                return self.push_batched_cond(node, (remap, mapped), batch_size)
            }
            TensorOp::Input { .. }
            | TensorOp::ScalarConstant { .. }
            | TensorOp::Constant { .. } => {
                return Err(BatchingError::Invalid(format!(
                    "{} has no operand and cannot depend on a mapped input",
                    tensor_op_name(&node.op)
                )))
            }
        };
        Ok((
            self.push_node(
                op,
                batched_shape(batch_size, &node.shape),
                node.dtype,
                node.weak,
            ),
            true,
        ))
    }

    /// `node` (mapped, shape `[B, *example]`) with unit axes inserted after
    /// the batch axis up to example rank `rank`, so that broadcasting against
    /// an operand of that rank aligns example axes with example axes.
    fn pad_batched(&mut self, node: TensorNodeId, rank: usize) -> Result<TensorNodeId, String> {
        let source = self.node(node)?;
        let example_rank = source.shape.len() - 1;
        if example_rank >= rank {
            return Ok(node);
        }
        let mut shape = Vec::with_capacity(rank + 1);
        shape.push(source.shape[0]);
        shape.extend(std::iter::repeat_n(1, rank - example_rank));
        shape.extend_from_slice(&source.shape[1..]);
        let (dtype, weak) = (source.dtype, source.weak);
        Ok(self.push_node(TensorOp::Reshape { input: node }, shape, dtype, weak))
    }

    pub fn symbolic_jvp(
        &self,
        output: TensorNodeId,
        differentiated_input: &str,
    ) -> Result<SymbolicJvp, String> {
        self.ensure_differentiable_input(differentiated_input)?;
        let mut found_input = false;
        let transformed = self.symbolic_jvp_with_seed(output, |transformed, name, value, _| {
            if name == differentiated_input {
                found_input = true;
                Ok(Some(symbolic_full_like(transformed, 1.0, value)?))
            } else {
                Ok(None)
            }
        })?;
        if !found_input {
            return Err(format!("input {differentiated_input:?} does not exist"));
        }
        Ok(transformed)
    }

    /// Emits a forward-mode transform with explicit tangent input tensors.
    ///
    /// `tangent_inputs` maps original input names to same-shaped tangent input
    /// names. Inputs omitted from the map receive a zero tangent. This keeps
    /// runtime JVP directions in the transformable IR so a backend can execute
    /// the primal and directional result in one compiled plan.
    pub fn symbolic_jvp_with_tangent_inputs(
        &self,
        output: TensorNodeId,
        tangent_inputs: &BTreeMap<String, String>,
    ) -> Result<SymbolicJvp, String> {
        if tangent_inputs.is_empty() {
            return Err("symbolic JVP requires at least one tangent input".to_string());
        }
        let original_inputs = self
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                TensorOp::Input { name } => Some(name.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let mut tangent_names = BTreeSet::new();
        for (input_name, tangent_name) in tangent_inputs {
            if !original_inputs.contains(input_name) {
                return Err(format!("input {input_name:?} does not exist"));
            }
            self.ensure_differentiable_input(input_name)?;
            if original_inputs.contains(tangent_name) {
                return Err(format!(
                    "tangent input name {tangent_name:?} conflicts with an existing input"
                ));
            }
            if !tangent_names.insert(tangent_name) {
                return Err(format!("duplicate tangent input name {tangent_name:?}"));
            }
        }
        self.symbolic_jvp_with_seed(output, |transformed, name, value, shape| {
            tangent_inputs
                .get(name)
                .map(|tangent_name| {
                    let dtype = transformed.node_dtype(value)?;
                    transformed.input_typed(tangent_name.clone(), shape.to_vec(), dtype)
                })
                .transpose()
        })
    }

    /// Emits a forward-mode transform retaining every requested output pair.
    ///
    /// The returned values and tangents preserve the order of `outputs` and
    /// share one transformed graph. Keeping sibling outputs live is required
    /// for structural higher-order transforms to lower a shared backend group.
    pub fn symbolic_jvp_many_with_tangent_inputs(
        &self,
        outputs: &[TensorNodeId],
        tangent_inputs: &BTreeMap<String, String>,
    ) -> Result<SymbolicJvpMany, String> {
        if tangent_inputs.is_empty() {
            return Err("symbolic JVP requires at least one tangent input".to_string());
        }
        let original_inputs = self
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                TensorOp::Input { name } => Some(name.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let mut tangent_names = BTreeSet::new();
        for (input_name, tangent_name) in tangent_inputs {
            if !original_inputs.contains(input_name) {
                return Err(format!("input {input_name:?} does not exist"));
            }
            self.ensure_differentiable_input(input_name)?;
            if original_inputs.contains(tangent_name) {
                return Err(format!(
                    "tangent input name {tangent_name:?} conflicts with an existing input"
                ));
            }
            if !tangent_names.insert(tangent_name) {
                return Err(format!("duplicate tangent input name {tangent_name:?}"));
            }
        }
        self.symbolic_jvp_many_with_seed(outputs, |transformed, name, value, shape| {
            tangent_inputs
                .get(name)
                .map(|tangent_name| {
                    let dtype = transformed.node_dtype(value)?;
                    transformed.input_typed(tangent_name.clone(), shape.to_vec(), dtype)
                })
                .transpose()
        })
    }

    fn symbolic_jvp_with_seed<F>(
        &self,
        output: TensorNodeId,
        input_tangent: F,
    ) -> Result<SymbolicJvp, String>
    where
        F: FnMut(
            &mut TensorIr,
            &str,
            TensorNodeId,
            &[usize],
        ) -> Result<Option<TensorNodeId>, String>,
    {
        let SymbolicJvpMany {
            graph,
            mut values,
            mut tangents,
        } = self.symbolic_jvp_many_with_seed(&[output], input_tangent)?;
        let value = values
            .pop()
            .ok_or_else(|| "symbolic JVP transform has no requested value output".to_string())?;
        let tangent = tangents
            .pop()
            .ok_or_else(|| "symbolic JVP transform has no requested tangent output".to_string())?;
        Ok(SymbolicJvp {
            graph,
            value,
            tangent,
        })
    }

    fn symbolic_jvp_many_with_seed<F>(
        &self,
        outputs: &[TensorNodeId],
        mut input_tangent: F,
    ) -> Result<SymbolicJvpMany, String>
    where
        F: FnMut(
            &mut TensorIr,
            &str,
            TensorNodeId,
            &[usize],
        ) -> Result<Option<TensorNodeId>, String>,
    {
        if let Some((expanded, mapped)) = self.expand_cholesky_for_ad()? {
            let outputs = outputs
                .iter()
                .map(|output| {
                    mapped
                        .get(*output)
                        .copied()
                        .ok_or_else(|| format!("output node {output} does not exist"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return expanded.symbolic_jvp_many_with_seed(&outputs, input_tangent);
        }
        if outputs.is_empty() {
            return Err("symbolic JVP requires at least one output".to_string());
        }
        for output in outputs {
            self.node(*output)?;
        }
        let reachable = tensor_output_reachability(&self.nodes, outputs);
        let mut transformed = TensorIr::new();
        let mut pairs = Vec::with_capacity(self.nodes.len());
        let mut scan_jvp_results = HashMap::new();
        let mut fori_vjp_jvp_groups = HashMap::new();
        let mut scan_vjp_jvp_groups = HashMap::new();
        // Whether a node depends on a seeded tangent input: a custom rule is
        // applied only to calls whose tangent is not known to be zero, so a
        // `custom_vjp` call outside the differentiated path does not fail.
        let mut active = vec![false; self.nodes.len()];
        let custom_members = self.custom_groups();
        let mut custom_pairs =
            HashMap::<usize, HashMap<usize, (TensorNodeId, TensorNodeId)>>::new();

        for (node_index, node) in self.nodes.iter().enumerate() {
            if !reachable[node_index] && !matches!(node.op, TensorOp::Input { .. }) {
                pairs.push((usize::MAX, usize::MAX));
                continue;
            }
            if !matches!(node.op, TensorOp::Input { .. }) {
                active[node_index] = tensor_op_inputs(&node.op)
                    .iter()
                    .any(|input| active[*input]);
            }
            let pair = match &node.op {
                TensorOp::Input { name } => {
                    let value =
                        transformed.input_typed(name.clone(), node.shape.clone(), node.dtype)?;
                    // Bool inputs have no tangent; callers requesting one were rejected at the
                    // entry point.
                    let tangent = if node.dtype == TensorDType::Bool {
                        symbolic_zero_tangent(&mut transformed, &node.shape)?
                    } else {
                        match input_tangent(&mut transformed, name, value, &node.shape)? {
                            Some(tangent) => {
                                active[node_index] = true;
                                tangent
                            }
                            None => symbolic_zero_like(&mut transformed, value)?,
                        }
                    };
                    (value, tangent)
                }
                TensorOp::ScalarConstant { value } => (
                    transformed.constant_like(*value, node.dtype, node.weak),
                    transformed.scalar_constant(0.0),
                ),
                TensorOp::Constant { value } => {
                    let primal = transformed.push_node(
                        TensorOp::Constant {
                            value: value.clone(),
                        },
                        node.shape.clone(),
                        node.dtype,
                        node.weak,
                    );
                    let tangent = if node.dtype == TensorDType::Bool {
                        symbolic_zero_tangent(&mut transformed, &node.shape)?
                    } else {
                        symbolic_zero_like(&mut transformed, primal)?
                    };
                    (primal, tangent)
                }
                // d cast(x) = cast(dx): the tangent has the same dtype as the primal.
                TensorOp::Cast { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.cast(value, node.dtype)?,
                        transformed.cast(tangent, node.dtype)?,
                    )
                }
                TensorOp::Add { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    (
                        transformed.add(lhs_value, rhs_value)?,
                        transformed.add(lhs_tangent, rhs_tangent)?,
                    )
                }
                TensorOp::Sub { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    (
                        transformed.sub(lhs_value, rhs_value)?,
                        transformed.sub(lhs_tangent, rhs_tangent)?,
                    )
                }
                TensorOp::Div { lhs, rhs } => {
                    // d(l/r) = (dl - q dr) / r with q = l/r; unlike the quotient
                    // rule over r*r, it cannot overflow or underflow when |r|
                    // is outside the square root of the dtype's range.
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let quotient = transformed.div(lhs_value, rhs_value)?;
                    let scaled = transformed.mul(quotient, rhs_tangent)?;
                    let numerator = transformed.sub(lhs_tangent, scaled)?;
                    (quotient, transformed.div(numerator, rhs_value)?)
                }
                TensorOp::Mul { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let left_term = transformed.mul(lhs_tangent, rhs_value)?;
                    let right_term = transformed.mul(lhs_value, rhs_tangent)?;
                    (
                        transformed.mul(lhs_value, rhs_value)?,
                        transformed.add(left_term, right_term)?,
                    )
                }
                TensorOp::Greater { lhs, rhs } => {
                    let (lhs_value, _) = pairs[*lhs];
                    let (rhs_value, _) = pairs[*rhs];
                    let value = transformed.greater(lhs_value, rhs_value)?;
                    // Shaped like the mask, so a reshape of it (a mapped
                    // predicate broadcast by `vmap`) reshapes its tangent too.
                    let tangent = symbolic_zero_tangent(&mut transformed, &node.shape)?;
                    (value, tangent)
                }
                // The tangent of a Bool result is a weak f64 zero shaped like the value, so data
                // movement such as reshape/concat and Bool->float casts propagate zero tangents as
                // usual.
                TensorOp::Compare { lhs, rhs, kind } => {
                    let (lhs_value, _) = pairs[*lhs];
                    let (rhs_value, _) = pairs[*rhs];
                    (
                        transformed.compare(lhs_value, rhs_value, *kind)?,
                        symbolic_zero_tangent(&mut transformed, &node.shape)?,
                    )
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let (condition_value, _) = pairs[*condition];
                    let (true_value, true_tangent) = pairs[*on_true];
                    let (false_value, false_tangent) = pairs[*on_false];
                    (
                        transformed.where_select(condition_value, true_value, false_value)?,
                        transformed.where_select(condition_value, true_tangent, false_tangent)?,
                    )
                }
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => symbolic_jvp_cond(
                    &mut transformed,
                    pairs[*predicate].0,
                    branches,
                    captures,
                    &pairs,
                    &format!("__quabla_cond_jvp_{node_index}"),
                )?,
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => symbolic_jvp_fori(
                    &mut transformed,
                    loop_plan,
                    *carry,
                    captures,
                    &pairs,
                    &format!("__quabla_fori_jvp_{node_index}"),
                )?,
                TensorOp::While {
                    carry,
                    loop_plan,
                    captures,
                } => symbolic_jvp_while(
                    &mut transformed,
                    loop_plan,
                    *carry,
                    captures,
                    &pairs,
                    &format!("__quabla_while_jvp_{node_index}"),
                )?,
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "symbolic JVP through a Fori JVP result is not implemented".to_string()
                    );
                }
                TensorOp::ForiVjp {
                    carry,
                    output_cotangent,
                    loop_plan,
                    captures,
                    target,
                    group,
                } => symbolic_jvp_fori_vjp(
                    &mut transformed,
                    SymbolicJvpForiVjpContext {
                        loop_plan,
                        carry: *carry,
                        output_cotangent: *output_cotangent,
                        captures,
                        target,
                        source_group: *group,
                    },
                    &pairs,
                    &mut fori_vjp_jvp_groups,
                    &format!("__quabla_fori_vjp_jvp_{node_index}"),
                )?,
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "symbolic JVP through a Fori VJP JVP result is not implemented".to_string(),
                    );
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !scan_jvp_results.contains_key(group) {
                        let result = symbolic_jvp_scan(
                            &mut transformed,
                            scan_plan,
                            *carry,
                            captures,
                            &pairs,
                            &format!("__quabla_scan_jvp_{node_index}"),
                        )?;
                        scan_jvp_results.insert(*group, result);
                    }
                    let result = scan_jvp_results.get(group).ok_or_else(|| {
                        format!("symbolic Scan JVP group {group} was not constructed")
                    })?;
                    match target {
                        TensorScanTarget::Carry => result.0,
                        TensorScanTarget::Outputs => result.1,
                    }
                }
                TensorOp::ScanVjp {
                    carry,
                    final_carry_cotangent,
                    output_cotangent,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => symbolic_jvp_scan_vjp(
                    &mut transformed,
                    SymbolicJvpScanVjpContext {
                        scan_plan,
                        carry: *carry,
                        final_carry_cotangent: *final_carry_cotangent,
                        output_cotangent: *output_cotangent,
                        captures,
                        target,
                        source_group: *group,
                    },
                    &pairs,
                    &mut scan_vjp_jvp_groups,
                    &format!("__quabla_scan_vjp_jvp_{node_index}"),
                )?,
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "symbolic JVP through a Scan VJP JVP result is not implemented".to_string(),
                    );
                }
                TensorOp::Sqrt { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.sqrt(input_value)?;
                    let derivative = transformed.sqrt_derivative(input_value, 1)?;
                    (value, transformed.mul(input_tangent, derivative)?)
                }
                TensorOp::SqrtDerivative { input, order } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.sqrt_derivative(input_value, *order)?;
                    let next_order = order
                        .checked_add(1)
                        .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                    let derivative = transformed.sqrt_derivative(input_value, next_order)?;
                    (value, transformed.mul(input_tangent, derivative)?)
                }
                TensorOp::UnaryMath { input, kind } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let (value, tangent) =
                        transformed.unary_math_jvp(*kind, input_value, input_tangent)?;
                    let tangent = match tangent {
                        Some(tangent) => tangent,
                        None => symbolic_zero_like(&mut transformed, value)?,
                    };
                    (value, tangent)
                }
                TensorOp::StopGradient { input } => {
                    let (input_value, _) = pairs[*input];
                    let value = transformed.stop_gradient(input_value)?;
                    let tangent = if node.dtype == TensorDType::Bool {
                        symbolic_zero_tangent(&mut transformed, &node.shape)?
                    } else {
                        symbolic_zero_like(&mut transformed, value)?
                    };
                    (value, tangent)
                }
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => {
                    let (input_value, input_tangent) = pairs[*input];
                    (
                        transformed.cumsum(input_value, *axis as isize, *reverse)?,
                        transformed.cumsum(input_tangent, *axis as isize, *reverse)?,
                    )
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let left_term = transformed.matmul(lhs_tangent, rhs_value)?;
                    let right_term = transformed.matmul(lhs_value, rhs_tangent)?;
                    (
                        transformed.matmul(lhs_value, rhs_value)?,
                        transformed.add(left_term, right_term)?,
                    )
                }
                TensorOp::Cholesky { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.cholesky(value)?,
                        transformed.cholesky_ad(vec![value, tangent], CholeskyAdKind::Jvp)?,
                    )
                }
                TensorOp::CholeskyAd { inputs, kind } => {
                    let values = inputs.iter().map(|id| pairs[*id].0).collect::<Vec<_>>();
                    let tangent = match kind {
                        CholeskyAdKind::Jvp => transformed.cholesky_ad(
                            vec![values[0], values[1], pairs[inputs[0]].1, pairs[inputs[1]].1],
                            CholeskyAdKind::Mixed,
                        )?,
                        CholeskyAdKind::Vjp => transformed.cholesky_ad(
                            vec![values[0], pairs[inputs[0]].1, values[1], pairs[inputs[1]].1],
                            CholeskyAdKind::VjpJvp,
                        )?,
                        _ => unreachable!("higher Cholesky derivatives expanded"),
                    };
                    (transformed.cholesky_ad(values, *kind)?, tangent)
                }
                TensorOp::Solve { matrix, rhs } => {
                    let (matrix_value, matrix_tangent) = pairs[*matrix];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let value = transformed.solve(matrix_value, rhs_value)?;
                    let correction = transformed.matmul(matrix_tangent, value)?;
                    let adjusted_rhs = transformed.sub(rhs_tangent, correction)?;
                    let tangent = transformed.solve(matrix_value, adjusted_rhs)?;
                    (value, tangent)
                }
                TensorOp::Triangular { input, lower } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.triangular(value, *lower)?,
                        transformed.triangular(tangent, *lower)?,
                    )
                }
                TensorOp::Linalg { input, kind } => {
                    let (matrix, tangent) = pairs[*input];
                    let value = transformed.linalg(matrix, *kind)?;
                    let tangent = transformed.linalg_jvp(matrix, value, tangent, *kind)?;
                    (value, tangent)
                }
                TensorOp::Sum { input } => {
                    let (value, tangent) = pairs[*input];
                    (transformed.sum(value)?, transformed.sum(tangent)?)
                }
                TensorOp::Mean { input } => {
                    let (value, tangent) = pairs[*input];
                    (transformed.mean(value)?, transformed.mean(tangent)?)
                }
                TensorOp::SumAxis { input, axis } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.sum_axis(value, *axis as isize)?,
                        transformed.sum_axis(tangent, *axis as isize)?,
                    )
                }
                TensorOp::ExtremumAxis { input, axis, kind } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.extremum_axis(input_value, *axis as isize, *kind)?;
                    let tangent =
                        transformed.extremum_axis_jvp(input_value, value, input_tangent, *axis)?;
                    (value, tangent)
                }
                TensorOp::MeanAxis { input, axis } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.mean_axis(value, *axis as isize)?,
                        transformed.mean_axis(tangent, *axis as isize)?,
                    )
                }
                TensorOp::Reshape { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.reshape(value, node.shape.clone())?,
                        transformed.reshape(tangent, node.shape.clone())?,
                    )
                }
                TensorOp::Transpose { input, axes } => {
                    let (value, tangent) = pairs[*input];
                    let axes: Vec<isize> = axes.iter().map(|axis| *axis as isize).collect();
                    (
                        transformed.transpose(value, Some(axes.clone()))?,
                        transformed.transpose(tangent, Some(axes))?,
                    )
                }
                TensorOp::Powi { input, exponent } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.powi(input_value, *exponent)?;
                    let tangent = if *exponent == 0 {
                        symbolic_zero_like(&mut transformed, input_value)?
                    } else {
                        let coefficient = transformed.scalar_constant(*exponent as f64);
                        let lower_power = transformed.powi(input_value, *exponent - 1)?;
                        let derivative = transformed.mul(coefficient, lower_power)?;
                        transformed.mul(input_tangent, derivative)?
                    };
                    (value, tangent)
                }
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    transformed.binary_math_jvp(*kind, pairs[*lhs], pairs[*rhs])?
                }
                TensorOp::Concat { inputs, axis } => {
                    let values = inputs.iter().map(|input| pairs[*input].0).collect();
                    let tangents = inputs.iter().map(|input| pairs[*input].1).collect();
                    (
                        transformed.concat(values, *axis as isize)?,
                        transformed.concat(tangents, *axis as isize)?,
                    )
                }
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.slice(value, *axis, *start, *length)?,
                        transformed.slice(tangent, *axis, *start, *length)?,
                    )
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.pad_slice(value, node.shape.clone(), *axis, *start)?,
                        transformed.pad_slice(tangent, node.shape.clone(), *axis, *start)?,
                    )
                }
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.gather(value, indices.clone(), *axis)?,
                        transformed.gather(tangent, indices.clone(), *axis)?,
                    )
                }
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => {
                    let (base_value, base_tangent) = pairs[*base];
                    let (update_value, update_tangent) = pairs[*updates];
                    (
                        transformed.scatter_add(
                            base_value,
                            update_value,
                            indices.clone(),
                            *axis,
                        )?,
                        transformed.scatter_add(
                            base_tangent,
                            update_tangent,
                            indices.clone(),
                            *axis,
                        )?,
                    )
                }
                TensorOp::Broadcast { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.broadcast_to(value, node.shape.clone())?,
                        transformed.broadcast_to(tangent, node.shape.clone())?,
                    )
                }
                // The whole call is rebuilt at its first reachable output:
                // the primal stays a `Custom` node, so a later reverse-mode
                // transform still applies the rule, and the tangents come
                // from the rule's tangent graph.
                TensorOp::Custom {
                    operands,
                    rule,
                    output,
                    group,
                    ..
                } => {
                    if !custom_pairs.contains_key(group) {
                        let members = custom_members
                            .get(group)
                            .ok_or_else(|| format!("custom group {group} has no members"))?
                            .iter()
                            .filter(|(_, member)| reachable[*member])
                            .map(|&(member_output, member)| match &self.nodes[member].op {
                                TensorOp::Custom { value, .. } => {
                                    Ok((member_output, pairs[*value].0))
                                }
                                _ => Err(format!(
                                    "custom group {group} member {member} is not custom"
                                )),
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                        let primal_operands = operands
                            .iter()
                            .map(|operand| pairs[*operand].0)
                            .collect::<Vec<_>>();
                        let ids = transformed.push_custom_group(
                            rule.clone(),
                            &members,
                            &primal_operands,
                        )?;
                        let tangents = if operands.iter().any(|operand| active[*operand]) {
                            let operand_tangents = operands
                                .iter()
                                .map(|operand| pairs[*operand].1)
                                .collect::<Vec<_>>();
                            Some(rule.splice_tangent(
                                &mut transformed,
                                &primal_operands,
                                &operand_tangents,
                            )?)
                        } else {
                            None
                        };
                        let mut rebuilt = HashMap::new();
                        for ((member_output, _), id) in members.iter().zip(ids) {
                            let tangent = match &tangents {
                                Some(tangents) => tangents[*member_output],
                                None => symbolic_zero_like(&mut transformed, id)?,
                            };
                            rebuilt.insert(*member_output, (id, tangent));
                        }
                        custom_pairs.insert(*group, rebuilt);
                    }
                    custom_pairs[group][output]
                }
            };
            if transformed.node(pair.0)?.dtype != node.dtype {
                return Err(format!(
                    "symbolic JVP rebuilt node {node_index} with dtype {}, expected {}",
                    transformed.node(pair.0)?.dtype,
                    node.dtype
                ));
            }
            pairs.push(pair);
        }

        let (values, tangents) = outputs.iter().map(|output| pairs[*output]).unzip();
        Ok(SymbolicJvpMany {
            graph: transformed,
            values,
            tangents,
        })
    }

    /// Emits a reverse-mode transformable TensorIr graph.
    ///
    /// The returned graph takes every original input plus `cotangent_name`.
    /// Its `gradients` nodes are the corresponding VJP outputs, so a backend
    /// can compile primal and reverse graphs without returning to the CPU AD
    /// evaluator.
    pub fn symbolic_vjp(
        &self,
        output: TensorNodeId,
        cotangent_name: &str,
    ) -> Result<SymbolicVjp, String> {
        let transformed = self.symbolic_vjp_many_impl(
            &[(output, SymbolicCotangent::Input(cotangent_name.to_string()))],
            true,
        )?;
        Ok(SymbolicVjp {
            value: transformed.primals[output],
            cotangent: transformed.cotangents[0],
            graph: transformed.graph,
            gradients: transformed.gradients,
        })
    }

    /// Emits one reverse-mode graph for several seeded outputs.
    ///
    /// The gradients are the sum of the VJPs of every output with its seed,
    /// which is the VJP of the tuple of outputs; an output listed twice
    /// accumulates both seeds. Every source node is replayed, so
    /// [`SymbolicVjpMany::primals`] also maps nodes the outputs do not depend
    /// on. With [`SymbolicCotangent::Input`] seeds and a single output this is
    /// exactly [`Self::symbolic_vjp`].
    pub fn symbolic_vjp_many(
        &self,
        outputs: &[(TensorNodeId, SymbolicCotangent)],
    ) -> Result<SymbolicVjpMany, String> {
        self.symbolic_vjp_many_impl(outputs, false)
    }

    fn symbolic_vjp_many_impl(
        &self,
        outputs: &[(TensorNodeId, SymbolicCotangent)],
        prune_dead_primals: bool,
    ) -> Result<SymbolicVjpMany, String> {
        if let Some((expanded, mapped)) = self.expand_cholesky_for_ad()? {
            let outputs = outputs
                .iter()
                .map(|(output, seed)| {
                    mapped
                        .get(*output)
                        .copied()
                        .map(|id| (id, seed.clone()))
                        .ok_or_else(|| format!("output node {output} does not exist"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut result = expanded.symbolic_vjp_many_impl(&outputs, prune_dead_primals)?;
            result.primals = mapped.iter().map(|id| result.primals[*id]).collect();
            return Ok(result);
        }
        if outputs.is_empty() {
            return Err("symbolic VJP requires at least one output".to_string());
        }
        let mut seed_names = BTreeSet::new();
        for (output, seed) in outputs {
            ensure_differentiable_output(self.node(*output)?)?;
            let SymbolicCotangent::Input(cotangent_name) = seed else {
                continue;
            };
            if self
                .nodes
                .iter()
                .any(|node| matches!(&node.op, TensorOp::Input { name } if name == cotangent_name))
            {
                return Err(format!(
                    "cotangent input name {cotangent_name:?} conflicts with an existing input"
                ));
            }
            if !seed_names.insert(cotangent_name) {
                return Err(format!("duplicate cotangent input name {cotangent_name:?}"));
            }
        }

        let requested = outputs
            .iter()
            .map(|(output, _)| *output)
            .collect::<Vec<_>>();
        let reachable = tensor_output_reachability(&self.nodes, &requested);
        let mut transformed = TensorIr::new();
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut scan_values = HashMap::new();
        // The forward graph of each custom call (primal outputs, then
        // residuals), spliced at its first reachable output.
        let mut custom_forward = HashMap::<usize, Vec<TensorNodeId>>::new();
        for (node_index, node) in self.nodes.iter().enumerate() {
            if prune_dead_primals
                && !reachable[node_index]
                && !matches!(node.op, TensorOp::Input { .. })
            {
                values.push(usize::MAX);
                continue;
            }
            let value = match &node.op {
                TensorOp::Input { name } => {
                    transformed.input_typed(name.clone(), node.shape.clone(), node.dtype)?
                }
                TensorOp::ScalarConstant { value } => {
                    transformed.constant_like(*value, node.dtype, node.weak)
                }
                TensorOp::Constant { .. } => transformed.push_node(
                    node.op.clone(),
                    node.shape.clone(),
                    node.dtype,
                    node.weak,
                ),
                TensorOp::Cast { input } => transformed.cast(values[*input], node.dtype)?,
                TensorOp::Add { lhs, rhs } => transformed.add(values[*lhs], values[*rhs])?,
                TensorOp::Sub { lhs, rhs } => transformed.sub(values[*lhs], values[*rhs])?,
                TensorOp::Div { lhs, rhs } => transformed.div(values[*lhs], values[*rhs])?,
                TensorOp::Mul { lhs, rhs } => transformed.mul(values[*lhs], values[*rhs])?,
                TensorOp::Greater { lhs, rhs } => {
                    transformed.greater(values[*lhs], values[*rhs])?
                }
                TensorOp::Compare { lhs, rhs, kind } => {
                    transformed.compare(values[*lhs], values[*rhs], *kind)?
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => transformed.where_select(
                    values[*condition],
                    values[*on_true],
                    values[*on_false],
                )?,
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => symbolic_clone_cond(
                    &mut transformed,
                    values[*predicate],
                    branches,
                    captures,
                    &values,
                )?,
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => symbolic_clone_fori(&mut transformed, *carry, loop_plan, captures, &values)?,
                TensorOp::While {
                    carry,
                    loop_plan,
                    captures,
                } => symbolic_clone_while(&mut transformed, *carry, loop_plan, captures, &values)?,
                TensorOp::ForiVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP result is not implemented".to_string()
                    );
                }
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori JVP result is not implemented".to_string()
                    );
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP JVP result is not implemented".to_string(),
                    );
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !scan_values.contains_key(group) {
                        let carry = values.get(*carry).copied().ok_or_else(|| {
                            format!("scan carry node {carry} has no symbolic value")
                        })?;
                        let captures = captures
                            .iter()
                            .map(|(name, node_id)| {
                                values
                                    .get(*node_id)
                                    .map(|value| (name.clone(), *value))
                                    .ok_or_else(|| {
                                        format!("scan capture node {node_id} has no symbolic value")
                                    })
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                        scan_values.insert(
                            *group,
                            transformed.scan(carry, scan_plan.clone(), captures)?,
                        );
                    }
                    let result = scan_values.get(group).ok_or_else(|| {
                        format!("symbolic Scan group {group} was not constructed")
                    })?;
                    match target {
                        TensorScanTarget::Carry => result.0,
                        TensorScanTarget::Outputs => result.1,
                    }
                }
                TensorOp::ScanVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP result is not implemented".to_string()
                    );
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP JVP result is not implemented".to_string(),
                    );
                }
                TensorOp::Sum { input } => transformed.sum(values[*input])?,
                TensorOp::SumAxis { input, axis } => {
                    transformed.sum_axis(values[*input], *axis as isize)?
                }
                TensorOp::ExtremumAxis { input, axis, kind } => {
                    transformed.extremum_axis(values[*input], *axis as isize, *kind)?
                }
                TensorOp::Matmul { lhs, rhs } => transformed.matmul(values[*lhs], values[*rhs])?,
                TensorOp::Cholesky { input } => transformed.cholesky(values[*input])?,
                TensorOp::CholeskyAd { inputs, kind } => {
                    transformed.cholesky_ad(inputs.iter().map(|id| values[*id]).collect(), *kind)?
                }
                TensorOp::Solve { matrix, rhs } => {
                    transformed.solve(values[*matrix], values[*rhs])?
                }
                TensorOp::Linalg { input, kind } => transformed.linalg(values[*input], *kind)?,
                TensorOp::Triangular { input, lower } => {
                    transformed.triangular(values[*input], *lower)?
                }
                TensorOp::Sqrt { input } => transformed.sqrt(values[*input])?,
                TensorOp::SqrtDerivative { input, order } => {
                    transformed.sqrt_derivative(values[*input], *order)?
                }
                TensorOp::Reshape { input } => {
                    transformed.reshape(values[*input], node.shape.clone())?
                }
                TensorOp::Mean { input } => transformed.mean(values[*input])?,
                TensorOp::MeanAxis { input, axis } => {
                    transformed.mean_axis(values[*input], *axis as isize)?
                }
                TensorOp::Powi { input, exponent } => {
                    transformed.powi(values[*input], *exponent)?
                }
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    transformed.binary_math(values[*lhs], values[*rhs], *kind)?
                }
                TensorOp::Transpose { input, axes } => transformed.transpose(
                    values[*input],
                    Some(axes.iter().map(|axis| *axis as isize).collect()),
                )?,
                TensorOp::UnaryMath { input, kind } => {
                    transformed.unary_math(values[*input], *kind)?
                }
                TensorOp::StopGradient { input } => transformed.stop_gradient(values[*input])?,
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => transformed.cumsum(values[*input], *axis as isize, *reverse)?,
                TensorOp::Concat { inputs, axis } => transformed.concat(
                    inputs.iter().map(|input| values[*input]).collect(),
                    *axis as isize,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => transformed.slice(values[*input], *axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => {
                    transformed.pad_slice(values[*input], node.shape.clone(), *axis, *start)?
                }
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => transformed.gather(values[*input], indices.clone(), *axis)?,
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => transformed.scatter_add(
                    values[*base],
                    values[*updates],
                    indices.clone(),
                    *axis,
                )?,
                TensorOp::Broadcast { input } => {
                    transformed.broadcast_to(values[*input], node.shape.clone())?
                }
                // Under reverse mode the primal comes from the rule's forward
                // graph, which also computes the residuals, as in JAX.
                TensorOp::Custom {
                    operands,
                    rule,
                    output,
                    group,
                    ..
                } => {
                    if !custom_forward.contains_key(group) {
                        let operands = operands
                            .iter()
                            .map(|operand| values[*operand])
                            .collect::<Vec<_>>();
                        custom_forward
                            .insert(*group, rule.splice_forward(&mut transformed, &operands)?);
                    }
                    custom_forward[group][*output]
                }
            };
            debug_assert_eq!(values.len(), node_index);
            if transformed.node(value)?.dtype != node.dtype {
                return Err(format!(
                    "symbolic VJP rebuilt node {node_index} with dtype {}, expected {}",
                    transformed.node(value)?.dtype,
                    node.dtype
                ));
            }
            values.push(value);
        }

        let mut cotangents = vec![None; self.nodes.len()];
        let mut seeds = Vec::with_capacity(outputs.len());
        for (output, seed) in outputs {
            let output_node = self.node(*output)?;
            let seed = match seed {
                SymbolicCotangent::Input(cotangent_name) => transformed.input_typed(
                    cotangent_name.clone(),
                    output_node.shape.clone(),
                    output_node.dtype,
                )?,
                // A strong constant, typed like the output as the input seed is.
                SymbolicCotangent::Ones => {
                    let one = transformed.constant_like(1.0, output_node.dtype, false);
                    if output_node.shape.is_empty() {
                        one
                    } else {
                        transformed.broadcast_to(one, output_node.shape.clone())?
                    }
                }
            };
            symbolic_accumulate(&mut transformed, &mut cotangents, *output, seed)?;
            seeds.push(seed);
        }
        let mut processed_scan_groups = HashSet::new();
        let custom_members = self.custom_groups();
        let mut processed_custom_groups = HashSet::new();

        for node_id in (0..self.nodes.len()).rev() {
            let Some(upstream) = cotangents[node_id] else {
                continue;
            };
            let node = self.node(node_id)?;
            match &node.op {
                TensorOp::Input { .. }
                | TensorOp::ScalarConstant { .. }
                | TensorOp::Constant { .. }
                | TensorOp::Greater { .. }
                | TensorOp::Compare { .. } => {}
                // The cast VJP converts the cotangent back to the source dtype; Bool sources are
                // not differentiable and get no gradient.
                TensorOp::Cast { input } => {
                    let source_dtype = self.node(*input)?.dtype;
                    if source_dtype != TensorDType::Bool {
                        let contribution = transformed.cast(upstream, source_dtype)?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                    }
                }
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => symbolic_vjp_cond(
                    &mut transformed,
                    values[*predicate],
                    branches,
                    SymbolicVjpCondContext {
                        captures,
                        values: &values,
                        upstream,
                    },
                    &mut cotangents,
                    node_id,
                )?,
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => symbolic_vjp_fori(
                    &mut transformed,
                    loop_plan,
                    SymbolicVjpForiContext {
                        original_carry: *carry,
                        carry: values[*carry],
                        captures,
                        values: &values,
                        upstream,
                    },
                    &mut cotangents,
                    node_id,
                )?,
                TensorOp::While { .. } => {
                    return Err(WHILE_LOOP_REVERSE_MODE_ERROR.to_string());
                }
                TensorOp::ForiVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP result is not implemented".to_string()
                    );
                }
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori JVP result is not implemented".to_string()
                    );
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP JVP result is not implemented".to_string(),
                    );
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    group,
                    ..
                } => {
                    if processed_scan_groups.insert(*group) {
                        let mut carry_upstream = None;
                        let mut output_upstream = None;
                        for (scan_node_id, scan_node) in self.nodes.iter().enumerate() {
                            let TensorOp::Scan {
                                target,
                                group: candidate_group,
                                ..
                            } = &scan_node.op
                            else {
                                continue;
                            };
                            if candidate_group != group {
                                continue;
                            }
                            let group_upstream = match cotangents[scan_node_id] {
                                Some(cotangent) => cotangent,
                                None => {
                                    // An unrequested sibling may have no rebuilt primal. Its
                                    // zero seed needs only shape/type, never the primal data.
                                    let zero = transformed.constant_like(
                                        0.0,
                                        scan_node.dtype,
                                        scan_node.weak,
                                    );
                                    transformed.broadcast_to(zero, scan_node.shape.clone())?
                                }
                            };
                            match target {
                                TensorScanTarget::Carry => {
                                    if carry_upstream.replace(group_upstream).is_some() {
                                        return Err(format!(
                                            "symbolic Scan group {group} has multiple carry results"
                                        ));
                                    }
                                }
                                TensorScanTarget::Outputs => {
                                    if output_upstream.replace(group_upstream).is_some() {
                                        return Err(format!(
                                            "symbolic Scan group {group} has multiple output results"
                                        ));
                                    }
                                }
                            }
                        }
                        // Dead-code elimination drops a sibling whose value is
                        // unused, e.g. the stacked outputs of a scan whose region
                        // returns only the final carry. A discarded result
                        // contributes an exact zero cotangent, typed as the scan
                        // body would type that result.
                        let final_carry_upstream = match carry_upstream {
                            Some(upstream) => upstream,
                            None => {
                                let dtype =
                                    scan_plan.body.plan.input_dtype(&scan_plan.carry_name)?;
                                let zero = transformed.constant_like(0.0, dtype, false);
                                transformed.broadcast_to(zero, scan_plan.carry_shape()?)?
                            }
                        };
                        let output_upstream = match output_upstream {
                            Some(upstream) => upstream,
                            None => {
                                let dtype = scan_plan
                                    .body
                                    .plan
                                    .node_dtype(scan_plan.body.plan.output_node_ids[1])?;
                                let zero = transformed.constant_like(0.0, dtype, false);
                                transformed.broadcast_to(zero, scan_plan.output_shape()?)?
                            }
                        };
                        symbolic_vjp_scan(
                            &mut transformed,
                            scan_plan,
                            SymbolicVjpScanContext {
                                original_carry: *carry,
                                carry: values[*carry],
                                captures,
                                values: &values,
                                final_carry_upstream,
                                output_upstream,
                            },
                            &mut cotangents,
                            *group,
                        )?;
                    }
                }
                TensorOp::ScanVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP result is not implemented".to_string()
                    );
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP JVP result is not implemented".to_string(),
                    );
                }
                TensorOp::Add { lhs, rhs } => {
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Sub { lhs, rhs } => {
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let zero = transformed.scalar_constant(0.0);
                    let negated = transformed.sub(zero, upstream)?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        negated,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Div { lhs, rhs } => {
                    let lhs_value = values[*lhs];
                    let rhs_value = values[*rhs];
                    // d/dr (l/r) = -(1/r)(l/r): no r*r term that could
                    // overflow or underflow.
                    let lhs_contribution = transformed.div(upstream, rhs_value)?;
                    let quotient = transformed.div(lhs_value, rhs_value)?;
                    let product = transformed.mul(lhs_contribution, quotient)?;
                    let zero = transformed.scalar_constant(0.0);
                    let rhs_contribution = transformed.sub(zero, product)?;
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        lhs_contribution,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        rhs_contribution,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_contribution = transformed.mul(upstream, values[*rhs])?;
                    let rhs_contribution = transformed.mul(upstream, values[*lhs])?;
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        lhs_contribution,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        rhs_contribution,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let zero = transformed.scalar_constant(0.0);
                    let true_contribution =
                        transformed.where_select(values[*condition], upstream, zero)?;
                    let false_contribution =
                        transformed.where_select(values[*condition], zero, upstream)?;
                    let true_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        true_contribution,
                        &node.shape,
                        &self.node(*on_true)?.shape,
                    )?;
                    let false_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        false_contribution,
                        &node.shape,
                        &self.node(*on_false)?.shape,
                    )?;
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        *on_true,
                        true_contribution,
                    )?;
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        *on_false,
                        false_contribution,
                    )?;
                }
                TensorOp::Sum { input } => {
                    let contribution =
                        symbolic_broadcast_like(&mut transformed, upstream, values[*input])?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Broadcast { input } => {
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::SumAxis { input, axis } => {
                    let contribution = symbolic_expand_reduced_axis(
                        &mut transformed,
                        upstream,
                        values[*input],
                        *axis,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::ExtremumAxis { input, axis, .. } => {
                    let contribution = transformed.extremum_axis_vjp(
                        values[*input],
                        values[node_id],
                        upstream,
                        *axis,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_transposed = symbolic_transpose_last_two(
                        &mut transformed,
                        values[*lhs],
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_transposed = symbolic_transpose_last_two(
                        &mut transformed,
                        values[*rhs],
                        &self.node(*rhs)?.shape,
                    )?;
                    let lhs_contribution = transformed.matmul(upstream, rhs_transposed)?;
                    let rhs_contribution = transformed.matmul(lhs_transposed, upstream)?;
                    let lhs_shape = transformed.node_shape(lhs_contribution)?;
                    let rhs_shape = transformed.node_shape(rhs_contribution)?;
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        lhs_contribution,
                        &lhs_shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        rhs_contribution,
                        &rhs_shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Cholesky { input } => {
                    let contribution = transformed
                        .cholesky_ad(vec![values[*input], upstream], CholeskyAdKind::Vjp)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::CholeskyAd { inputs, kind } => {
                    let zero = symbolic_full_like(&mut transformed, 0.0, upstream)?;
                    let (input_gradient, other_gradient) = match kind {
                        CholeskyAdKind::Jvp => (
                            transformed.cholesky_ad(
                                vec![values[inputs[0]], values[inputs[1]], upstream, zero],
                                CholeskyAdKind::VjpJvp,
                            )?,
                            transformed.cholesky_ad(
                                vec![values[inputs[0]], upstream],
                                CholeskyAdKind::Vjp,
                            )?,
                        ),
                        CholeskyAdKind::Vjp => (
                            transformed.cholesky_ad(
                                vec![values[inputs[0]], upstream, values[inputs[1]], zero],
                                CholeskyAdKind::VjpJvp,
                            )?,
                            transformed.cholesky_ad(
                                vec![values[inputs[0]], upstream],
                                CholeskyAdKind::Jvp,
                            )?,
                        ),
                        _ => unreachable!("higher Cholesky derivatives expanded"),
                    };
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        inputs[0],
                        input_gradient,
                    )?;
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        inputs[1],
                        other_gradient,
                    )?;
                }
                TensorOp::Solve { matrix, rhs } => {
                    let matrix_shape = self.node(*matrix)?.shape.clone();
                    let mut axes = (0..matrix_shape.len())
                        .map(|axis| axis as isize)
                        .collect::<Vec<_>>();
                    axes.swap(matrix_shape.len() - 2, matrix_shape.len() - 1);
                    let transposed = transformed.transpose(values[*matrix], Some(axes))?;
                    let rhs_contribution = transformed.solve(transposed, upstream)?;
                    let value_transposed = symbolic_transpose_last_two(
                        &mut transformed,
                        values[node_id],
                        &node.shape,
                    )?;
                    let matrix_contribution =
                        transformed.matmul(rhs_contribution, value_transposed)?;
                    let zero = transformed.scalar_constant(0.0);
                    let matrix_contribution = transformed.sub(zero, matrix_contribution)?;
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        *matrix,
                        matrix_contribution,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Triangular { input, lower } => {
                    let contribution = transformed.triangular(upstream, *lower)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Linalg { input, kind } => {
                    if let Some(contribution) =
                        transformed.linalg_vjp(values[*input], values[node_id], upstream, *kind)?
                    {
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                    }
                }
                TensorOp::Sqrt { input } => {
                    let derivative = transformed.sqrt_derivative(values[*input], 1)?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::SqrtDerivative { input, order } => {
                    let next_order = order
                        .checked_add(1)
                        .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                    let derivative = transformed.sqrt_derivative(values[*input], next_order)?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Reshape { input } => {
                    let contribution =
                        transformed.reshape(upstream, self.node(*input)?.shape.clone())?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Mean { input } => {
                    let contribution =
                        symbolic_broadcast_like(&mut transformed, upstream, values[*input])?;
                    let scale = transformed
                        .scalar_constant(1.0 / element_count(&self.node(*input)?.shape)? as f64);
                    let contribution = transformed.mul(contribution, scale)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::MeanAxis { input, axis } => {
                    let contribution = symbolic_expand_reduced_axis(
                        &mut transformed,
                        upstream,
                        values[*input],
                        *axis,
                    )?;
                    let scale =
                        transformed.scalar_constant(1.0 / self.node(*input)?.shape[*axis] as f64);
                    let contribution = transformed.mul(contribution, scale)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent != 0 {
                        let coefficient = transformed.scalar_constant(*exponent as f64);
                        let lower_power = transformed.powi(values[*input], *exponent - 1)?;
                        let derivative = transformed.mul(coefficient, lower_power)?;
                        let contribution = transformed.mul(upstream, derivative)?;
                        let contribution = symbolic_reduce_to_shape(
                            &mut transformed,
                            contribution,
                            &node.shape,
                            &self.node(*input)?.shape,
                        )?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                    }
                }
                // A scalar constant operand receives no cotangent, so its
                // partial is not emitted.
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    let operands = [*lhs, *rhs];
                    let differentiable =
                        operands.map(|operand| self.scalar_constant_value(operand).is_none());
                    transformed.binary_math_vjp(
                        *kind,
                        (values[*lhs], values[*rhs], values[node_id]),
                        upstream,
                        differentiable,
                        |transformed, index, contribution| {
                            let contribution = symbolic_reduce_to_shape(
                                transformed,
                                contribution,
                                &node.shape,
                                &self.node(operands[index])?.shape,
                            )?;
                            symbolic_accumulate(
                                transformed,
                                &mut cotangents,
                                operands[index],
                                contribution,
                            )
                        },
                    )?;
                }
                TensorOp::Transpose { input, axes } => {
                    let inverse = inverse_permutation(axes)?;
                    let contribution = transformed.transpose(
                        upstream,
                        Some(inverse.into_iter().map(|axis| axis as isize).collect()),
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::UnaryMath { input, kind } => {
                    let chain = transformed.unary_math_chain(
                        *kind,
                        values[*input],
                        Some(values[node_id]),
                    )?;
                    if let Some(contribution) = transformed.apply_unary_chain(chain, upstream)? {
                        let contribution = symbolic_reduce_to_shape(
                            &mut transformed,
                            contribution,
                            &node.shape,
                            &self.node(*input)?.shape,
                        )?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                    }
                }
                // The value passes through, but no cotangent does.
                TensorOp::StopGradient { .. } => {}
                // The transpose of an inclusive prefix sum is the inclusive
                // prefix sum in the opposite direction.
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => {
                    let contribution = transformed.cumsum(upstream, *axis as isize, !*reverse)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Concat { inputs, axis } => {
                    let mut start = 0;
                    for input in inputs {
                        let input_shape = &self.node(*input)?.shape;
                        let contribution =
                            transformed.slice(upstream, *axis, start, input_shape[*axis])?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                        start += input_shape[*axis];
                    }
                }
                TensorOp::Slice {
                    input, axis, start, ..
                } => {
                    let contribution = transformed.pad_slice(
                        upstream,
                        self.node(*input)?.shape.clone(),
                        *axis,
                        *start,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let contribution = transformed.slice(
                        upstream,
                        *axis,
                        *start,
                        self.node(*input)?.shape[*axis],
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => {
                    let base = match cotangents[*input] {
                        Some(existing) => existing,
                        None => {
                            let dtype = transformed.node(upstream)?.dtype;
                            let zero = transformed.constant_like(0.0, dtype, false);
                            transformed.broadcast_to(zero, self.node(*input)?.shape.clone())?
                        }
                    };
                    cotangents[*input] =
                        Some(transformed.gather_cotangent(base, upstream, indices, *axis)?);
                }
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => {
                    let contribution = transformed.gather(upstream, indices.clone(), *axis)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *base, upstream)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *updates, contribution)?;
                }
                // The backward graph runs once per call, at the first output
                // reached in reverse order. Every consumer of the call's
                // outputs comes after all of them, so their cotangents are
                // complete here; an output without one gets zeros.
                TensorOp::Custom {
                    operands,
                    rule,
                    group,
                    ..
                } => {
                    if processed_custom_groups.insert(*group) {
                        let forward = custom_forward
                            .get(group)
                            .ok_or_else(|| format!("custom group {group} has no forward values"))?
                            .clone();
                        let mut output_cotangents = forward[..rule.output_count()]
                            .iter()
                            .map(|_| None)
                            .collect::<Vec<_>>();
                        for (output, member) in custom_members.get(group).into_iter().flatten() {
                            output_cotangents[*output] = cotangents[*member];
                        }
                        let output_cotangents = output_cotangents
                            .into_iter()
                            .zip(&forward)
                            .map(|(cotangent, value)| match cotangent {
                                Some(cotangent) => Ok(cotangent),
                                None => symbolic_zero_like(&mut transformed, *value),
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                        let contributions = rule.splice_backward(
                            &mut transformed,
                            &forward[rule.output_count()..],
                            &output_cotangents,
                        )?;
                        for (operand, contribution) in operands.iter().zip(contributions) {
                            if let Some(contribution) = contribution {
                                symbolic_accumulate(
                                    &mut transformed,
                                    &mut cotangents,
                                    *operand,
                                    contribution,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        let mut gradients = BTreeMap::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let TensorOp::Input { name } = &node.op {
                // Bool inputs are not differentiable and do not appear in the gradient table.
                if node.dtype == TensorDType::Bool {
                    continue;
                }
                let gradient = match cotangents[node_id] {
                    Some(cotangent) => cotangent,
                    None => symbolic_zero_like(&mut transformed, values[node_id])?,
                };
                gradients.insert(name.clone(), gradient);
            }
        }
        Ok(SymbolicVjpMany {
            graph: transformed,
            primals: values,
            cotangents: seeds,
            gradients,
        })
    }

    /// Adds an `f64` input; see [`TensorIr::input_typed`] for other dtypes.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, String> {
        self.input_typed(name, shape, TensorDType::F64)
    }

    /// Adds a strong input of `dtype`. Host values bound to it are rounded to
    /// `dtype` at execution, like JAX jit arguments.
    pub fn input_typed(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: TensorDType,
    ) -> Result<TensorNodeId, String> {
        element_count(&shape)?;
        let name = name.into();
        if self.input_nodes.contains_key(&name) {
            return Err(format!("input {name:?} already exists"));
        }

        Ok(self.push_node(TensorOp::Input { name }, shape, dtype, false))
    }

    /// Adds a weak `f64` scalar that adopts the dtype of a strong operand it is
    /// combined with (Python scalars and AD-internal constants).
    pub fn scalar_constant(&mut self, value: f64) -> TensorNodeId {
        self.push_node(
            TensorOp::ScalarConstant { value },
            vec![],
            TensorDType::F64,
            true,
        )
    }

    /// Adds an array constant with the shape and dtype of `value`.
    ///
    /// A constant is strong unless `weak` is set: an eager array keeps its
    /// dtype when it meets a traced value, so a strong `f64` constant combined
    /// with an `f32` operand is a dtype error that asks for `astype`. `weak`
    /// preserves the weak type of a value that already has one (an eager
    /// `bool` mask combined with a Python scalar). The data is shared, never
    /// copied, by later transforms and compiled plans; it has a zero tangent
    /// and receives no cotangent.
    pub fn constant(&mut self, value: DynamicTensor, weak: bool) -> TensorNodeId {
        let (shape, dtype) = (value.shape.clone(), value.dtype);
        self.push_node(
            TensorOp::Constant {
                value: TensorConstant::new(value),
            },
            shape,
            dtype,
            weak,
        )
    }

    /// Recreates a constant with the dtype and weakness of an existing node,
    /// for transforms that rebuild a graph node by node.
    fn constant_like(&mut self, value: f64, dtype: TensorDType, weak: bool) -> TensorNodeId {
        self.push_node(TensorOp::ScalarConstant { value }, vec![], dtype, weak)
    }

    /// Converts `input` to `dtype`. Same-dtype casts are kept here and removed
    /// when a plan is compiled; lossy round trips are never folded.
    ///
    /// A float-to-`Bool` conversion is emitted as `not_equal(input, 0)`, so
    /// `NaN` becomes true and no `Cast` node ever targets `Bool` from a float;
    /// `Cast` therefore stays an identity on CUDA and MLX, where `Bool` is
    /// held as `f32` `0`/`1`.
    pub fn cast(
        &mut self,
        input: TensorNodeId,
        dtype: TensorDType,
    ) -> Result<TensorNodeId, String> {
        let source = self.node(input)?;
        if dtype == TensorDType::Bool && source.dtype != TensorDType::Bool {
            let zero = self.scalar_constant(0.0);
            return self.compare(input, zero, TensorComparison::NotEqual);
        }
        let shape = source.shape.clone();
        Ok(self.push_node(TensorOp::Cast { input }, shape, dtype, false))
    }

    pub fn node_dtype(&self, id: TensorNodeId) -> Result<TensorDType, String> {
        Ok(self.node(id)?.dtype)
    }

    /// Rejects differentiation with respect to a `Bool` input: bool values
    /// have no tangent, and VJP gradient maps omit them.
    pub fn ensure_differentiable_input(&self, name: &str) -> Result<(), String> {
        let is_bool = self.nodes.iter().any(|node| {
            node.dtype == TensorDType::Bool
                && matches!(&node.op, TensorOp::Input { name: candidate } if candidate == name)
        });
        if is_bool {
            return Err(bool_input_derivative_error(name));
        }
        Ok(())
    }

    pub fn add(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary("add", lhs, rhs, |lhs, rhs| TensorOp::Add { lhs, rhs })
    }

    pub fn mul(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary("mul", lhs, rhs, |lhs, rhs| TensorOp::Mul { lhs, rhs })
    }

    pub fn sub(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary("sub", lhs, rhs, |lhs, rhs| TensorOp::Sub { lhs, rhs })
    }

    pub fn div(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary("div", lhs, rhs, |lhs, rhs| TensorOp::Div { lhs, rhs })
    }

    /// Elementwise `lhs > rhs` as a 0/1 mask in the operands' dtype.
    pub fn greater(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        self.binary("greater", lhs, rhs, |lhs, rhs| TensorOp::Greater {
            lhs,
            rhs,
        })
    }

    /// Elementwise IEEE comparison producing `Bool`. Operands follow the
    /// arithmetic promotion rule (strict floats, weak scalars, `Bool` promoted
    /// to the other operand's float dtype); two `Bool` operands compare as 0/1.
    pub fn compare(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        kind: TensorComparison,
    ) -> Result<TensorNodeId, String> {
        let shape = broadcast_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape)?;
        let [lhs, rhs] = self.coerce_operands(kind.name(), [lhs, rhs])?;
        Ok(self.push_node(
            TensorOp::Compare { lhs, rhs, kind },
            shape,
            TensorDType::Bool,
            false,
        ))
    }

    /// Elementwise `lhs && rhs` of two `Bool` tensors, lowered as
    /// `where(lhs, rhs, lhs)` so every backend reuses its select kernel.
    pub fn logical_and(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        self.ensure_bool_operands("logical_and", &[lhs, rhs])?;
        self.where_select(lhs, rhs, lhs)
    }

    /// Elementwise `lhs || rhs` of two `Bool` tensors, lowered as
    /// `where(lhs, lhs, rhs)`.
    pub fn logical_or(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        self.ensure_bool_operands("logical_or", &[lhs, rhs])?;
        self.where_select(lhs, lhs, rhs)
    }

    /// Elementwise negation of a `Bool` tensor, lowered as `equal(input, false)`.
    pub fn logical_not(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.ensure_bool_operands("logical_not", &[input])?;
        let false_value = self.push_node(
            TensorOp::ScalarConstant { value: 0.0 },
            vec![],
            TensorDType::Bool,
            false,
        );
        self.compare(input, false_value, TensorComparison::Equal)
    }

    /// Elementwise `NaN` test, lowered as the IEEE identity `input != input`.
    pub fn isnan(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.ensure_float_operand("isnan", input)?;
        self.compare(input, input, TensorComparison::NotEqual)
    }

    /// Elementwise finiteness test, lowered as `input - input == 0`: the
    /// difference is `0` for finite values and `NaN` for `±inf` and `NaN`.
    pub fn isfinite(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.ensure_float_operand("isfinite", input)?;
        let difference = self.sub(input, input)?;
        let zero = self.scalar_constant(0.0);
        self.compare(difference, zero, TensorComparison::Equal)
    }

    /// True when any element of a `Bool` tensor is true (scalar `Bool`).
    ///
    /// Lowered as `sum(cast(input, f64)) > 0`: a sum of non-negative 0/1
    /// values is zero only when every term is zero, even when a device
    /// accumulates in `f32`, so no dedicated reduction kernel is needed.
    pub fn any(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.bool_count_reduction("any", input, None, false)
    }

    pub fn any_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        self.bool_count_reduction("any", input, Some(axis), false)
    }

    /// True when every element of a `Bool` tensor is true (scalar `Bool`),
    /// lowered as `sum(cast(logical_not(input), f64)) == 0`.
    pub fn all(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.bool_count_reduction("all", input, None, true)
    }

    pub fn all_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        self.bool_count_reduction("all", input, Some(axis), true)
    }

    fn bool_count_reduction(
        &mut self,
        op_name: &str,
        input: TensorNodeId,
        axis: Option<isize>,
        all: bool,
    ) -> Result<TensorNodeId, String> {
        self.ensure_bool_operands(op_name, &[input])?;
        let counted = if all { self.logical_not(input)? } else { input };
        let counted = self.cast(counted, TensorDType::F64)?;
        let count = match axis {
            Some(axis) => self.sum_axis(counted, axis)?,
            None => self.sum(counted)?,
        };
        let zero = self.scalar_constant(0.0);
        let kind = if all {
            TensorComparison::Equal
        } else {
            TensorComparison::Greater
        };
        self.compare(count, zero, kind)
    }

    fn ensure_bool_operands(&self, op_name: &str, operands: &[TensorNodeId]) -> Result<(), String> {
        for operand in operands {
            let dtype = self.node(*operand)?.dtype;
            if dtype != TensorDType::Bool {
                return Err(format!(
                    "{op_name} requires bool operands, got dtype {dtype}; \
                     build a mask with a comparison or convert explicitly with astype"
                ));
            }
        }
        Ok(())
    }

    fn ensure_float_operand(&self, op_name: &str, operand: TensorNodeId) -> Result<(), String> {
        let dtype = self.node(operand)?.dtype;
        if !dtype.is_floating() {
            return Err(format!(
                "{op_name} requires a floating operand, got dtype {dtype}"
            ));
        }
        Ok(())
    }

    pub fn where_select(
        &mut self,
        condition: TensorNodeId,
        on_true: TensorNodeId,
        on_false: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let condition_shape = self.node(condition)?.shape.clone();
        let true_shape = self.node(on_true)?.shape.clone();
        let false_shape = self.node(on_false)?.shape.clone();
        let shape = broadcast_shape(
            &broadcast_shape(&condition_shape, &true_shape)?,
            &false_shape,
        )?;
        // The condition is only tested for non-zero and does not take part in dtype unification;
        // both branch values must share a dtype.
        let [on_true, on_false] = self.coerce_operands("where", [on_true, on_false])?;
        self.push_derived(
            TensorOp::Where {
                condition,
                on_true,
                on_false,
            },
            shape,
        )
    }

    /// Adds a lazy scalar conditional whose branch regions are frozen CPU
    /// plans. Region inputs are explicit named captures validated by
    /// `TensorCondExecutionPlan::new`.
    pub fn cond(
        &mut self,
        predicate: TensorNodeId,
        branches: TensorCondExecutionPlan,
    ) -> Result<TensorNodeId, String> {
        let captures = branches
            .captures()
            .keys()
            .map(|name| {
                self.input_node_id(name)
                    .map(|node_id| (name.clone(), node_id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.cond_with_captures(predicate, branches, captures)
    }

    /// Adds a lazy conditional with explicit ordered bindings from region input
    /// names to parent graph values. This is the primitive used by future
    /// traced nested control flow; `cond` is the input-name convenience form.
    pub fn cond_with_captures(
        &mut self,
        predicate: TensorNodeId,
        branches: TensorCondExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    ) -> Result<TensorNodeId, String> {
        if !self.node(predicate)?.shape.is_empty() {
            return Err(format!(
                "conditional predicate must be scalar, got shape {:?}",
                self.node(predicate)?.shape
            ));
        }
        if captures.len() != branches.captures().len() {
            return Err(
                "conditional captures must bind every branch capture exactly once".to_string(),
            );
        }
        let mut seen = BTreeSet::new();
        for (name, capture) in &captures {
            if !seen.insert(name.as_str()) {
                return Err(format!(
                    "conditional capture {name:?} is bound more than once"
                ));
            }
            let expected_shape = branches
                .captures()
                .get(name)
                .ok_or_else(|| format!("conditional binds unknown branch capture {name:?}"))?;
            if self.node(*capture)?.shape != *expected_shape {
                return Err(format!(
                    "conditional capture {name:?} has shape {:?}, expected {:?}",
                    self.node(*capture)?.shape,
                    expected_shape
                ));
            }
            self.check_region_binding_dtype("conditional", name, *capture, &branches.on_true.plan)?;
        }
        let shape = branches.output_shape()?;
        let dtype = branches.on_true.plan.output_dtype()?;
        Ok(self.push_node(
            TensorOp::Cond {
                predicate,
                branches,
                captures,
            },
            shape,
            dtype,
            false,
        ))
    }

    /// Region captures are strict: a parent value bound to a region input
    /// must already have that input's dtype.
    fn check_region_binding_dtype(
        &self,
        context: &str,
        name: &str,
        value: TensorNodeId,
        region: &TensorExecutionPlan,
    ) -> Result<(), String> {
        let expected = region.input_dtype(name)?;
        let actual = self.node(value)?.dtype;
        if actual != expected {
            return Err(format!(
                "{context} binds {name:?} of dtype {actual} to a region input of dtype {expected}; \
                 convert it explicitly with astype"
            ));
        }
        Ok(())
    }

    /// Adds a fixed-bound loop whose body owns an explicit carry/index region.
    /// External region captures are bound to already-dominating parent nodes.
    pub fn fori(
        &mut self,
        carry: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    ) -> Result<TensorNodeId, String> {
        let carry_shape = self.node(carry)?.shape.clone();
        if carry_shape != loop_plan.carry_shape()? {
            return Err(format!(
                "fori carry shape {:?} does not match loop body carry shape {:?}",
                carry_shape,
                loop_plan.carry_shape()?
            ));
        }
        if captures.len() != loop_plan.external_captures().len() {
            return Err(
                "fori loop captures must bind every external body capture exactly once".to_string(),
            );
        }
        let mut seen = BTreeSet::new();
        for (name, capture) in &captures {
            if !seen.insert(name.as_str()) {
                return Err(format!(
                    "fori loop capture {name:?} is bound more than once"
                ));
            }
            let expected_shape = loop_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("fori loop binds unknown external capture {name:?}"))?;
            if self.node(*capture)?.shape != *expected_shape {
                return Err(format!(
                    "fori loop capture {name:?} has shape {:?}, expected {:?}",
                    self.node(*capture)?.shape,
                    expected_shape
                ));
            }
            self.check_region_binding_dtype("fori loop", name, *capture, &loop_plan.body.plan)?;
        }
        self.check_region_binding_dtype(
            "fori loop",
            &loop_plan.carry_name,
            carry,
            &loop_plan.body.plan,
        )?;
        let dtype = loop_plan.body.plan.input_dtype(&loop_plan.carry_name)?;
        Ok(self.push_node(
            TensorOp::Fori {
                carry,
                loop_plan,
                captures,
            },
            carry_shape,
            dtype,
            false,
        ))
    }

    /// Adds a data-dependent loop whose predicate and body regions share the
    /// carry and external captures. External captures bind to parent nodes.
    pub fn while_loop(
        &mut self,
        carry: TensorNodeId,
        loop_plan: TensorWhileExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    ) -> Result<TensorNodeId, String> {
        let carry_shape = self.node(carry)?.shape.clone();
        if carry_shape != loop_plan.carry_shape()? {
            return Err(format!(
                "while loop carry shape {:?} does not match loop body carry shape {:?}",
                carry_shape,
                loop_plan.carry_shape()?
            ));
        }
        if captures.len() != loop_plan.external_captures().len() {
            return Err(
                "while loop captures must bind every external region capture exactly once"
                    .to_string(),
            );
        }
        let mut seen = BTreeSet::new();
        for (name, capture) in &captures {
            if !seen.insert(name.as_str()) {
                return Err(format!(
                    "while loop capture {name:?} is bound more than once"
                ));
            }
            let expected_shape = loop_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("while loop binds unknown external capture {name:?}"))?;
            if self.node(*capture)?.shape != *expected_shape {
                return Err(format!(
                    "while loop capture {name:?} has shape {:?}, expected {:?}",
                    self.node(*capture)?.shape,
                    expected_shape
                ));
            }
            self.check_region_binding_dtype(
                "while loop",
                name,
                *capture,
                loop_plan.region_of(name),
            )?;
        }
        self.check_region_binding_dtype(
            "while loop",
            &loop_plan.carry_name,
            carry,
            &loop_plan.body.plan,
        )?;
        let dtype = loop_plan.body.plan.input_dtype(&loop_plan.carry_name)?;
        Ok(self.push_node(
            TensorOp::While {
                carry,
                loop_plan,
                captures,
            },
            carry_shape,
            dtype,
            false,
        ))
    }

    /// Adds the tangent result for a fixed-bound loop. The primal result is
    /// represented by a separate `Fori` node so backends can lower both loop
    /// states without materializing a packed carry tensor.
    fn fori_jvp(
        &mut self,
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        tangent_captures: Vec<(String, TensorNodeId)>,
    ) -> Result<TensorNodeId, String> {
        let carry_shape = self.node(carry)?.shape.clone();
        if self.node(carry_tangent)?.shape != carry_shape {
            return Err(format!(
                "fori JVP carry tangent shape {:?} does not match carry shape {:?}",
                self.node(carry_tangent)?.shape,
                carry_shape
            ));
        }
        if carry_shape != loop_plan.carry_shape()? {
            return Err(format!(
                "fori JVP carry shape {:?} does not match loop body carry shape {:?}",
                carry_shape,
                loop_plan.carry_shape()?
            ));
        }
        let expected_captures = loop_plan.external_captures();
        for (kind, bound_captures) in [("primal", &captures), ("tangent", &tangent_captures)] {
            if bound_captures.len() != expected_captures.len() {
                return Err(format!(
                    "fori JVP {kind} captures must bind every external body capture exactly once"
                ));
            }
            let mut seen = BTreeSet::new();
            for (name, capture) in bound_captures {
                if !seen.insert(name.as_str()) {
                    return Err(format!(
                        "fori JVP {kind} capture {name:?} is bound more than once"
                    ));
                }
                let expected_shape = expected_captures
                    .get(name)
                    .ok_or_else(|| format!("fori JVP binds unknown external capture {name:?}"))?;
                if self.node(*capture)?.shape != *expected_shape {
                    return Err(format!(
                        "fori JVP {kind} capture {name:?} has shape {:?}, expected {:?}",
                        self.node(*capture)?.shape,
                        expected_shape
                    ));
                }
            }
        }
        let dtype = loop_plan.body.plan.input_dtype(&loop_plan.carry_name)?;
        Ok(self.push_node(
            TensorOp::ForiJvp {
                carry,
                carry_tangent,
                loop_plan,
                captures,
                tangent_captures,
            },
            carry_shape,
            dtype,
            false,
        ))
    }

    /// Adds a fixed-bound scan whose body returns `(next_carry, output)`.
    /// Both returned node ids share one execution group so evaluating a plan
    /// that consumes both results runs the scan body only once.
    pub fn scan(
        &mut self,
        carry: TensorNodeId,
        scan_plan: TensorScanExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
    ) -> Result<(TensorNodeId, TensorNodeId), String> {
        let carry_shape = self.node(carry)?.shape.clone();
        if carry_shape != scan_plan.carry_shape()? {
            return Err(format!(
                "scan carry shape {:?} does not match scan body carry shape {:?}",
                carry_shape,
                scan_plan.carry_shape()?
            ));
        }
        if captures.len() != scan_plan.external_captures().len() {
            return Err(
                "scan captures must bind every external body capture exactly once".to_string(),
            );
        }
        let mut seen = BTreeSet::new();
        for (name, capture) in &captures {
            if !seen.insert(name.as_str()) {
                return Err(format!("scan capture {name:?} is bound more than once"));
            }
            let expected_shape = scan_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("scan binds unknown external capture {name:?}"))?;
            if self.node(*capture)?.shape != *expected_shape {
                return Err(format!(
                    "scan capture {name:?} has shape {:?}, expected {:?}",
                    self.node(*capture)?.shape,
                    expected_shape
                ));
            }
            self.check_region_binding_dtype("scan", name, *capture, &scan_plan.body.plan)?;
        }
        self.check_region_binding_dtype(
            "scan",
            &scan_plan.carry_name,
            carry,
            &scan_plan.body.plan,
        )?;

        let output_shape = scan_plan.output_shape()?;
        let carry_dtype = scan_plan.body.plan.input_dtype(&scan_plan.carry_name)?;
        let output_dtype = scan_plan
            .body
            .plan
            .node_dtype(scan_plan.body.plan.output_node_ids[1])?;
        let group = self.nodes.len();
        let carry_id = self.push_node(
            TensorOp::Scan {
                carry,
                scan_plan: scan_plan.clone(),
                captures: captures.clone(),
                target: TensorScanTarget::Carry,
                group,
            },
            carry_shape,
            carry_dtype,
            false,
        );
        let outputs_id = self.push_node(
            TensorOp::Scan {
                carry,
                scan_plan,
                captures,
                target: TensorScanTarget::Outputs,
                group,
            },
            output_shape,
            output_dtype,
            false,
        );
        Ok((carry_id, outputs_id))
    }

    fn fori_vjp(
        &mut self,
        carry: TensorNodeId,
        output_cotangent: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        captures: Vec<(String, TensorNodeId)>,
        target: TensorForiVjpTarget,
        group: usize,
    ) -> Result<TensorNodeId, String> {
        let carry_shape = self.node(carry)?.shape.clone();
        if carry_shape != loop_plan.carry_shape()? {
            return Err("fori VJP carry shape does not match loop body carry shape".to_string());
        }
        if self.node(output_cotangent)?.shape != carry_shape {
            return Err("fori VJP output cotangent shape does not match carry shape".to_string());
        }
        if captures.len() != loop_plan.external_captures().len() {
            return Err(
                "fori VJP captures must bind every external body capture exactly once".to_string(),
            );
        }
        let mut seen = BTreeSet::new();
        for (name, capture) in &captures {
            if !seen.insert(name.as_str()) {
                return Err(format!("fori VJP capture {name:?} is bound more than once"));
            }
            let expected_shape = loop_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("fori VJP binds unknown external capture {name:?}"))?;
            if self.node(*capture)?.shape != *expected_shape {
                return Err(format!(
                    "fori VJP capture {name:?} has shape {:?}, expected {:?}",
                    self.node(*capture)?.shape,
                    expected_shape
                ));
            }
        }
        let shape = match &target {
            TensorForiVjpTarget::Carry => carry_shape,
            TensorForiVjpTarget::External(name) => loop_plan
                .external_captures()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("fori VJP has no external capture {name:?}"))?,
        };
        let dtype = loop_plan.body.plan.input_dtype(match &target {
            TensorForiVjpTarget::Carry => &loop_plan.carry_name,
            TensorForiVjpTarget::External(name) => name,
        })?;
        Ok(self.push_node(
            TensorOp::ForiVjp {
                carry,
                output_cotangent,
                loop_plan,
                captures,
                target,
                group,
            },
            shape,
            dtype,
            false,
        ))
    }

    fn fori_vjp_jvp(
        &mut self,
        plan: TensorForiVjpJvpExecutionPlan,
        bindings: TensorForiVjpJvpBindings,
        target: TensorForiVjpTarget,
        group: usize,
    ) -> Result<TensorNodeId, String> {
        let TensorForiVjpJvpBindings {
            carry,
            carry_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures,
            tangent_captures,
        } = bindings;
        let loop_plan = &plan.loop_plan;
        let carry_shape = loop_plan.carry_shape()?;
        for (label, node_id) in [
            ("carry", carry),
            ("carry tangent", carry_tangent),
            ("output cotangent", output_cotangent),
            ("output cotangent tangent", output_cotangent_tangent),
        ] {
            if self.node(node_id)?.shape != carry_shape {
                return Err(format!(
                    "fori VJP JVP {label} shape does not match loop body carry shape"
                ));
            }
        }
        for (label, bound) in [
            ("capture", &captures),
            ("capture tangent", &tangent_captures),
        ] {
            if bound.len() != loop_plan.external_captures().len() {
                return Err(format!(
                    "fori VJP JVP {label}s must bind every external body capture exactly once"
                ));
            }
            let mut seen = BTreeSet::new();
            for (name, node_id) in bound {
                if !seen.insert(name.as_str()) {
                    return Err(format!(
                        "fori VJP JVP {label} {name:?} is bound more than once"
                    ));
                }
                let expected_shape = loop_plan.external_captures().get(name).ok_or_else(|| {
                    format!("fori VJP JVP binds unknown external capture {name:?}")
                })?;
                if self.node(*node_id)?.shape != *expected_shape {
                    return Err(format!(
                        "fori VJP JVP {label} {name:?} has shape {:?}, expected {:?}",
                        self.node(*node_id)?.shape,
                        expected_shape
                    ));
                }
            }
        }
        let shape = match &target {
            TensorForiVjpTarget::Carry => carry_shape,
            TensorForiVjpTarget::External(name) => loop_plan
                .external_captures()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("fori VJP JVP has no external capture {name:?}"))?,
        };
        let dtype = loop_plan.body.plan.input_dtype(match &target {
            TensorForiVjpTarget::Carry => &loop_plan.carry_name,
            TensorForiVjpTarget::External(name) => name,
        })?;
        Ok(self.push_node(
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
            },
            shape,
            dtype,
            false,
        ))
    }

    fn scan_vjp(
        &mut self,
        scan_plan: TensorScanExecutionPlan,
        bindings: TensorScanVjpBindings,
        target: TensorScanVjpTarget,
        group: usize,
    ) -> Result<TensorNodeId, String> {
        let TensorScanVjpBindings {
            carry,
            final_carry_cotangent,
            output_cotangent,
            captures,
        } = bindings;
        let carry_shape = self.node(carry)?.shape.clone();
        if carry_shape != scan_plan.carry_shape()? {
            return Err("scan VJP carry shape does not match scan body carry shape".to_string());
        }
        if self.node(final_carry_cotangent)?.shape != carry_shape {
            return Err(
                "scan VJP final carry cotangent shape does not match carry shape".to_string(),
            );
        }
        if self.node(output_cotangent)?.shape != scan_plan.output_shape()? {
            return Err(
                "scan VJP output cotangent shape does not match scan output shape".to_string(),
            );
        }
        if captures.len() != scan_plan.external_captures().len() {
            return Err(
                "scan VJP captures must bind every external body capture exactly once".to_string(),
            );
        }
        let mut seen = BTreeSet::new();
        for (name, capture) in &captures {
            if !seen.insert(name.as_str()) {
                return Err(format!("scan VJP capture {name:?} is bound more than once"));
            }
            let expected_shape = scan_plan
                .external_captures()
                .get(name)
                .ok_or_else(|| format!("scan VJP binds unknown external capture {name:?}"))?;
            if self.node(*capture)?.shape != *expected_shape {
                return Err(format!(
                    "scan VJP capture {name:?} has shape {:?}, expected {:?}",
                    self.node(*capture)?.shape,
                    expected_shape
                ));
            }
        }
        let shape = match &target {
            TensorScanVjpTarget::Carry => carry_shape,
            TensorScanVjpTarget::External(name) => scan_plan
                .external_captures()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("scan VJP has no external capture {name:?}"))?,
        };
        let dtype = scan_plan.body.plan.input_dtype(match &target {
            TensorScanVjpTarget::Carry => &scan_plan.carry_name,
            TensorScanVjpTarget::External(name) => name,
        })?;
        Ok(self.push_node(
            TensorOp::ScanVjp {
                carry,
                final_carry_cotangent,
                output_cotangent,
                scan_plan,
                captures,
                target,
                group,
            },
            shape,
            dtype,
            false,
        ))
    }

    fn scan_vjp_jvp(
        &mut self,
        plan: TensorScanVjpJvpExecutionPlan,
        bindings: TensorScanVjpJvpBindings,
        target: TensorScanVjpTarget,
        group: usize,
    ) -> Result<TensorNodeId, String> {
        let TensorScanVjpJvpBindings {
            carry,
            carry_tangent,
            final_carry_cotangent,
            final_carry_cotangent_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures,
            tangent_captures,
        } = bindings;
        let scan_plan = &plan.scan_plan;
        let carry_shape = scan_plan.carry_shape()?;
        let output_shape = scan_plan.output_shape()?;
        for (label, node_id, expected_shape) in [
            ("carry", carry, &carry_shape),
            ("carry tangent", carry_tangent, &carry_shape),
            ("final carry cotangent", final_carry_cotangent, &carry_shape),
            (
                "final carry cotangent tangent",
                final_carry_cotangent_tangent,
                &carry_shape,
            ),
            ("output cotangent", output_cotangent, &output_shape),
            (
                "output cotangent tangent",
                output_cotangent_tangent,
                &output_shape,
            ),
        ] {
            if self.node(node_id)?.shape != *expected_shape {
                return Err(format!(
                    "scan VJP JVP {label} shape does not match the Scan result shape"
                ));
            }
        }
        for (label, bound) in [
            ("capture", &captures),
            ("capture tangent", &tangent_captures),
        ] {
            if bound.len() != scan_plan.external_captures().len() {
                return Err(format!(
                    "scan VJP JVP {label}s must bind every external body capture exactly once"
                ));
            }
            let mut seen = BTreeSet::new();
            for (name, node_id) in bound {
                if !seen.insert(name.as_str()) {
                    return Err(format!(
                        "scan VJP JVP {label} {name:?} is bound more than once"
                    ));
                }
                let expected_shape = scan_plan.external_captures().get(name).ok_or_else(|| {
                    format!("scan VJP JVP binds unknown external capture {name:?}")
                })?;
                if self.node(*node_id)?.shape != *expected_shape {
                    return Err(format!(
                        "scan VJP JVP {label} {name:?} has shape {:?}, expected {:?}",
                        self.node(*node_id)?.shape,
                        expected_shape
                    ));
                }
            }
        }
        let shape = match &target {
            TensorScanVjpTarget::Carry => carry_shape,
            TensorScanVjpTarget::External(name) => scan_plan
                .external_captures()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("scan VJP JVP has no external capture {name:?}"))?,
        };
        let dtype = scan_plan.body.plan.input_dtype(match &target {
            TensorScanVjpTarget::Carry => &scan_plan.carry_name,
            TensorScanVjpTarget::External(name) => name,
        })?;
        Ok(self.push_node(
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
            },
            shape,
            dtype,
            false,
        ))
    }

    pub fn sum(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.node(input)?;
        self.push_derived(TensorOp::Sum { input }, vec![])
    }

    pub fn sum_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axis = normalize_axis(axis, input_shape.len())?;
        self.push_derived(
            TensorOp::SumAxis { input, axis },
            reduced_shape(&input_shape, axis)?,
        )
    }

    pub fn matmul(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = matmul_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape)?;
        let [lhs, rhs] = self.coerce_operands("matmul", [lhs, rhs])?;
        self.push_derived(TensorOp::Matmul { lhs, rhs }, shape)
    }

    pub fn solve(
        &mut self,
        matrix: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let shape = solve_shape(&self.node(matrix)?.shape, &self.node(rhs)?.shape)?;
        let [matrix, rhs] = self.coerce_operands("solve", [matrix, rhs])?;
        self.push_derived(TensorOp::Solve { matrix, rhs }, shape)
    }

    /// One output of a dense decomposition of each matrix of `input`
    /// (`[..., n, n]`, floating point); see [`LinalgKind`].
    pub fn linalg(
        &mut self,
        input: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<TensorNodeId, String> {
        let source = self.node(input)?;
        if !source.dtype.is_floating() {
            return Err(format!("{} requires a floating-point tensor", kind.name()));
        }
        let shape = kind.output_shape(&source.shape)?;
        self.push_derived(TensorOp::Linalg { input, kind }, shape)
    }

    /// `x` with its last two axes swapped.
    fn matrix_transpose(&mut self, x: TensorNodeId) -> Result<TensorNodeId, String> {
        let rank = self.node(x)?.shape.len();
        let mut axes = (0..rank as isize).collect::<Vec<_>>();
        axes.swap(rank - 2, rank - 1);
        self.transpose(x, Some(axes))
    }

    /// `(x + x^T) / 2` over the last two axes.
    fn symmetric_part(&mut self, x: TensorNodeId) -> Result<TensorNodeId, String> {
        let transposed = self.matrix_transpose(x)?;
        let sum = self.add(x, transposed)?;
        let half = self.scalar_constant(0.5);
        self.mul(sum, half)
    }

    /// The diagonals `[..., n]` of a square stack `[..., n, n]`: the two
    /// triangular projections keep only the diagonal, so the row sums are it.
    fn matrix_diagonal(&mut self, x: TensorNodeId) -> Result<TensorNodeId, String> {
        let lower = self.triangular(x, true)?;
        let diagonal = self.triangular(lower, false)?;
        self.sum_axis(diagonal, -1)
    }

    /// An `n x n` identity constant with `like`'s dtype.
    fn identity_like(&mut self, n: usize, like: TensorNodeId) -> Result<TensorNodeId, String> {
        let dtype = self.node(like)?.dtype;
        let mut data = vec![0.0; n * n];
        for index in 0..n {
            data[index * n + index] = 1.0;
        }
        let identity = DynamicTensor::with_dtype(vec![n, n], data, dtype)?;
        Ok(self.constant(identity, false))
    }

    /// `F` of the eigenvector derivative: `F_ij = 1 / (w_j - w_i)` for
    /// `i != j` and `0` on the diagonal, for eigenvalues `w` (`[..., n]`).
    /// Adding the identity before the reciprocal keeps the diagonal finite;
    /// a repeated eigenvalue makes an off-diagonal entry infinite, so the
    /// eigenvector derivative is then inf or NaN, as in JAX.
    fn eigh_gap_reciprocals(&mut self, values: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(values)?.shape.clone();
        let n = *shape.last().ok_or("eigenvalues have rank at least one")?;
        let batch = &shape[..shape.len() - 1];
        let row = self.reshape(values, [batch, &[1, n]].concat())?;
        let column = self.reshape(values, [batch, &[n, 1]].concat())?;
        let gaps = self.sub(row, column)?;
        let identity = self.identity_like(n, values)?;
        let shifted = self.add(gaps, identity)?;
        let one = self.scalar_constant(1.0);
        let reciprocals = self.div(one, shifted)?;
        self.sub(reciprocals, identity)
    }

    /// The tangent of `linalg(matrix, kind)` (whose value is `value`) along
    /// `tangent`:
    ///
    /// - `log|det|`: `tr(A^-1 dA)`, the diagonal sum of `solve(A, dA)`; no
    ///   inverse is formed.
    /// - eigenvalues: `diag(V^T dS V)`; eigenvectors: `V (F o (V^T dS V))`,
    ///   with `dS` the symmetric part of `dA` (the decomposition reads the
    ///   symmetric part of its input) and `F` from [`Self::eigh_gap_reciprocals`].
    /// - the determinant sign is piecewise constant: zero.
    fn linalg_jvp(
        &mut self,
        matrix: TensorNodeId,
        value: TensorNodeId,
        tangent: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<TensorNodeId, String> {
        match kind {
            LinalgKind::DetSign => symbolic_zero_like(self, value),
            LinalgKind::LogAbsDet => {
                let solved = self.solve(matrix, tangent)?;
                let diagonal = self.matrix_diagonal(solved)?;
                self.sum_axis(diagonal, -1)
            }
            LinalgKind::EighValues | LinalgKind::EighVectors => {
                let vectors = if kind == LinalgKind::EighVectors {
                    value
                } else {
                    self.linalg(matrix, LinalgKind::EighVectors)?
                };
                let symmetric = self.symmetric_part(tangent)?;
                let transposed = self.matrix_transpose(vectors)?;
                let rotated = self.matmul(transposed, symmetric)?;
                let rotated = self.matmul(rotated, vectors)?;
                if kind == LinalgKind::EighValues {
                    return self.matrix_diagonal(rotated);
                }
                let values = self.linalg(matrix, LinalgKind::EighValues)?;
                let gaps = self.eigh_gap_reciprocals(values)?;
                let scaled = self.mul(gaps, rotated)?;
                self.matmul(vectors, scaled)
            }
            LinalgKind::QrQ | LinalgKind::QrR | LinalgKind::QrQComplete => {
                self.qr_jvp(matrix, tangent, kind)
            }
            LinalgKind::SvdU
            | LinalgKind::SvdS
            | LinalgKind::SvdVh
            | LinalgKind::SvdUFull
            | LinalgKind::SvdVhFull => self.svd_jvp(matrix, tangent, kind),
        }
    }

    /// The cotangent of `matrix` for `linalg(matrix, kind)` (value `value`)
    /// receiving `cotangent`, the transpose of [`Self::linalg_jvp`]:
    ///
    /// - `log|det|`: `g A^-T`, computed as `solve(A^T, g I)`.
    /// - eigenvalues: `V diag(g) V^T`; eigenvectors: the symmetric part of
    ///   `V (F o (V^T G)) V^T`.
    /// - the determinant sign: no contribution (`None`).
    fn linalg_vjp(
        &mut self,
        matrix: TensorNodeId,
        value: TensorNodeId,
        cotangent: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<Option<TensorNodeId>, String> {
        let shape = self.node(matrix)?.shape.clone();
        let n = shape[shape.len() - 1];
        let batch = &shape[..shape.len() - 2];
        Ok(Some(match kind {
            LinalgKind::DetSign => return Ok(None),
            LinalgKind::LogAbsDet => {
                let identity = self.identity_like(n, matrix)?;
                let scale = self.reshape(cotangent, [batch, &[1, 1]].concat())?;
                let scaled = self.mul(identity, scale)?;
                let transposed = self.matrix_transpose(matrix)?;
                self.solve(transposed, scaled)?
            }
            LinalgKind::EighValues => {
                let vectors = self.linalg(matrix, LinalgKind::EighVectors)?;
                let scale = self.reshape(cotangent, [batch, &[1, n]].concat())?;
                let scaled = self.mul(vectors, scale)?;
                let transposed = self.matrix_transpose(vectors)?;
                self.matmul(scaled, transposed)?
            }
            LinalgKind::EighVectors => {
                let values = self.linalg(matrix, LinalgKind::EighValues)?;
                let gaps = self.eigh_gap_reciprocals(values)?;
                let transposed = self.matrix_transpose(value)?;
                let rotated = self.matmul(transposed, cotangent)?;
                let scaled = self.mul(gaps, rotated)?;
                let product = self.matmul(value, scaled)?;
                let product = self.matmul(product, transposed)?;
                self.symmetric_part(product)?
            }
            LinalgKind::QrQ | LinalgKind::QrR | LinalgKind::QrQComplete => {
                self.qr_vjp(matrix, cotangent, kind)?
            }
            LinalgKind::SvdU
            | LinalgKind::SvdS
            | LinalgKind::SvdVh
            | LinalgKind::SvdUFull
            | LinalgKind::SvdVhFull => self.svd_vjp(matrix, cotangent, kind)?,
        }))
    }

    /// `(m, n)` of the matrix stack `x` (`[..., m, n]`).
    fn matrix_extents(&self, x: TensorNodeId) -> Result<(usize, usize), String> {
        let shape = &self.node(x)?.shape;
        let rank = shape.len();
        if rank < 2 {
            return Err(format!("expected a stack of matrices, got shape {shape:?}"));
        }
        Ok((shape[rank - 2], shape[rank - 1]))
    }

    /// Columns `start..start + length` of the matrix stack `x`.
    fn matrix_columns(
        &mut self,
        x: TensorNodeId,
        start: usize,
        length: usize,
    ) -> Result<TensorNodeId, String> {
        let rank = self.node(x)?.shape.len();
        self.slice(x, rank - 1, start, length)
    }

    /// `x` with everything on and above the diagonal set to zero.
    fn strictly_lower(&mut self, x: TensorNodeId) -> Result<TensorNodeId, String> {
        let upper = self.triangular(x, false)?;
        self.sub(x, upper)
    }

    /// `x R^-1` for an upper-triangular `R`, as `solve(R^T, x^T)^T`.
    fn solve_right_upper(
        &mut self,
        x: TensorNodeId,
        r: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let r_transposed = self.matrix_transpose(r)?;
        let x_transposed = self.matrix_transpose(x)?;
        let solved = self.solve(r_transposed, x_transposed)?;
        self.matrix_transpose(solved)
    }

    /// `x R^-T` for an upper-triangular `R`, as `solve(R, x^T)^T`.
    fn solve_right_upper_transposed(
        &mut self,
        x: TensorNodeId,
        r: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let x_transposed = self.matrix_transpose(x)?;
        let solved = self.solve(r, x_transposed)?;
        self.matrix_transpose(solved)
    }

    /// Zeros of `shape` with `like`'s dtype.
    fn zeros_like_dtype(
        &mut self,
        shape: Vec<usize>,
        like: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let dtype = self.node(like)?.dtype;
        let zero = self.constant_like(0.0, dtype, false);
        self.broadcast_to(zero, shape)
    }

    /// The error of a derivative through the basis completion of a complete
    /// QR or full SVD, which is not unique.
    fn completion_derivative_error(kind: LinalgKind) -> String {
        format!(
            "{} is not differentiable: the extra orthonormal columns that complete the basis are \
             not unique; use the reduced decomposition",
            kind.name()
        )
    }

    /// The tangents `(dQ, dR)` of the QR factorization `A = Q R` of a
    /// matrix stack with at least as many rows as columns and full column
    /// rank, along `dA` (Walter, Lehmann & Lamour 2012; the rule of JAX's
    /// `qr_jvp_rule`): with `B = dA R^-1`, `C = Q^T B`, and the
    /// skew-symmetric `W = L - L^T` from the strictly lower triangle `L` of
    /// `C`, `dQ = Q (W - C) + B` and `dR = (C - W) R`. `R^-1` is applied
    /// with `solve`, never formed.
    fn qr_tall_jvp(
        &mut self,
        q: TensorNodeId,
        r: TensorNodeId,
        direction: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), String> {
        let b = self.solve_right_upper(direction, r)?;
        let q_transposed = self.matrix_transpose(q)?;
        let c = self.matmul(q_transposed, b)?;
        let lower = self.strictly_lower(c)?;
        let lower_transposed = self.matrix_transpose(lower)?;
        let skew = self.sub(lower, lower_transposed)?;
        let skew_minus_c = self.sub(skew, c)?;
        let rotated = self.matmul(q, skew_minus_c)?;
        let dq = self.add(rotated, b)?;
        let c_minus_skew = self.sub(c, skew)?;
        let dr = self.matmul(c_minus_skew, r)?;
        Ok((dq, dr))
    }

    /// The tangent of one QR output. A wide `A = [X | Y]` (`m < n`) has
    /// `Q R_X = X` and `R = [R_X | Q^T Y]`, so `X` takes the tall rule and
    /// `dR_Y = dQ^T Y + Q^T dY`; `X` must then have full rank.
    fn qr_jvp(
        &mut self,
        matrix: TensorNodeId,
        tangent: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<TensorNodeId, String> {
        let (m, n) = self.matrix_extents(matrix)?;
        if kind == LinalgKind::QrQComplete && m > n {
            return Err(Self::completion_derivative_error(kind));
        }
        let q = self.linalg(matrix, LinalgKind::QrQ)?;
        let r = self.linalg(matrix, LinalgKind::QrR)?;
        if m >= n {
            let (dq, dr) = self.qr_tall_jvp(q, r, tangent)?;
            return Ok(if kind == LinalgKind::QrR { dr } else { dq });
        }
        let r_square = self.matrix_columns(r, 0, m)?;
        let tangent_square = self.matrix_columns(tangent, 0, m)?;
        let (dq, dr_square) = self.qr_tall_jvp(q, r_square, tangent_square)?;
        if kind != LinalgKind::QrR {
            return Ok(dq);
        }
        let rest = self.matrix_columns(matrix, m, n - m)?;
        let tangent_rest = self.matrix_columns(tangent, m, n - m)?;
        let dq_transposed = self.matrix_transpose(dq)?;
        let q_transposed = self.matrix_transpose(q)?;
        let moved = self.matmul(dq_transposed, rest)?;
        let direct = self.matmul(q_transposed, tangent_rest)?;
        let dr_rest = self.add(moved, direct)?;
        let rank = self.node(matrix)?.shape.len();
        self.concat(vec![dr_square, dr_rest], rank as isize - 1)
    }

    /// The cotangent of a tall full-rank `A` for cotangents `G_Q` and `G_R`
    /// of its QR factors, the transpose of [`Self::qr_tall_jvp`]: with
    /// `M = Q^T G_Q - G_R R^T` and `N = L(M - M^T) - M` (`L` the strictly
    /// lower triangle), `G_A = (G_Q + Q N) R^-T`.
    fn qr_tall_vjp(
        &mut self,
        q: TensorNodeId,
        r: TensorNodeId,
        q_cotangent: Option<TensorNodeId>,
        r_cotangent: Option<TensorNodeId>,
    ) -> Result<TensorNodeId, String> {
        let q_transposed = self.matrix_transpose(q)?;
        let from_q = match q_cotangent {
            Some(cotangent) => Some(self.matmul(q_transposed, cotangent)?),
            None => None,
        };
        let from_r = match r_cotangent {
            Some(cotangent) => {
                let r_transposed = self.matrix_transpose(r)?;
                Some(self.matmul(cotangent, r_transposed)?)
            }
            None => None,
        };
        let mixed = match (from_q, from_r) {
            (Some(from_q), Some(from_r)) => self.sub(from_q, from_r)?,
            (Some(from_q), None) => from_q,
            (None, Some(from_r)) => {
                let zero = self.scalar_constant(0.0);
                self.sub(zero, from_r)?
            }
            (None, None) => return Err("QR cotangent requires at least one output".into()),
        };
        let mixed_transposed = self.matrix_transpose(mixed)?;
        let skew = self.sub(mixed, mixed_transposed)?;
        let lower = self.strictly_lower(skew)?;
        let correction = self.sub(lower, mixed)?;
        let rotated = self.matmul(q, correction)?;
        let total = match q_cotangent {
            Some(cotangent) => self.add(cotangent, rotated)?,
            None => rotated,
        };
        self.solve_right_upper_transposed(total, r)
    }

    /// The cotangent of `A` for one QR output, the transpose of
    /// [`Self::qr_jvp`]. For a wide `A = [X | Y]`, a cotangent `[G_X | G_Y]`
    /// of `R` sends `Q G_Y` to `Y` and `Y G_Y^T` to `Q`.
    fn qr_vjp(
        &mut self,
        matrix: TensorNodeId,
        cotangent: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<TensorNodeId, String> {
        let (m, n) = self.matrix_extents(matrix)?;
        if kind == LinalgKind::QrQComplete && m > n {
            return Err(Self::completion_derivative_error(kind));
        }
        let q = self.linalg(matrix, LinalgKind::QrQ)?;
        let r = self.linalg(matrix, LinalgKind::QrR)?;
        let is_r = kind == LinalgKind::QrR;
        if m >= n {
            return if is_r {
                self.qr_tall_vjp(q, r, None, Some(cotangent))
            } else {
                self.qr_tall_vjp(q, r, Some(cotangent), None)
            };
        }
        let rank = self.node(matrix)?.shape.len();
        let r_square = self.matrix_columns(r, 0, m)?;
        let (square, rest) = if is_r {
            let square_cotangent = self.matrix_columns(cotangent, 0, m)?;
            let rest_cotangent = self.matrix_columns(cotangent, m, n - m)?;
            let rest = self.matrix_columns(matrix, m, n - m)?;
            let rest_cotangent_transposed = self.matrix_transpose(rest_cotangent)?;
            let q_cotangent = self.matmul(rest, rest_cotangent_transposed)?;
            let square =
                self.qr_tall_vjp(q, r_square, Some(q_cotangent), Some(square_cotangent))?;
            (square, self.matmul(q, rest_cotangent)?)
        } else {
            let square = self.qr_tall_vjp(q, r_square, Some(cotangent), None)?;
            let mut shape = self.node(matrix)?.shape.clone();
            shape[rank - 1] = n - m;
            (square, self.zeros_like_dtype(shape, matrix)?)
        };
        self.concat(vec![square, rest], rank as isize - 1)
    }

    /// `F` of the singular vector derivatives: `F_ij = 1 / (s_j^2 - s_i^2)`,
    /// and `0` where `s_i == s_j` (the diagonal, and repeated singular
    /// values), as JAX's `svd_jvp_rule` masks it.
    fn svd_gap_reciprocals(&mut self, values: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(values)?.shape.clone();
        let k = *shape
            .last()
            .ok_or("singular values have rank at least one")?;
        let batch = &shape[..shape.len() - 1];
        let row = self.reshape(values, [batch, &[1, k]].concat())?;
        let column = self.reshape(values, [batch, &[k, 1]].concat())?;
        let sum = self.add(row, column)?;
        let difference = self.sub(row, column)?;
        let gaps = self.mul(sum, difference)?;
        let zero = self.scalar_constant(0.0);
        let equal = self.compare(gaps, zero, TensorComparison::Equal)?;
        let dtype = self.node(values)?.dtype;
        let mask = self.cast(equal, dtype)?;
        let shifted = self.add(gaps, mask)?;
        let one = self.scalar_constant(1.0);
        let reciprocals = self.div(one, shifted)?;
        self.sub(reciprocals, mask)
    }

    /// The reduced SVD factors `(U, s, Vh)` of `matrix` as `Linalg` nodes.
    fn svd_factors(
        &mut self,
        matrix: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId, TensorNodeId), String> {
        Ok((
            self.linalg(matrix, LinalgKind::SvdU)?,
            self.linalg(matrix, LinalgKind::SvdS)?,
            self.linalg(matrix, LinalgKind::SvdVh)?,
        ))
    }

    /// `s` (`[..., k]`) as a row `[..., 1, k]` and a column `[..., k, 1]`.
    fn row_and_column(
        &mut self,
        values: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), String> {
        let shape = self.node(values)?.shape.clone();
        let k = shape[shape.len() - 1];
        let batch = &shape[..shape.len() - 1];
        Ok((
            self.reshape(values, [batch, &[1, k]].concat())?,
            self.reshape(values, [batch, &[k, 1]].concat())?,
        ))
    }

    /// The tangent of one SVD output along `dA`, JAX's `svd_jvp_rule` for
    /// real matrices: with `dS = U^T dA V` and `S = diag(s)`,
    ///
    /// - `ds = diag(dS)`;
    /// - `dU = U (F o (dS S + S dS^T)) + (I - U U^T) dA V S^-1`, the last
    ///   term only for `m > n`;
    /// - `dV = V (F o (S dS + dS^T S)) + (I - V V^T) dA^T U S^-1`, the last
    ///   term only for `m < n`, and `dVh = dV^T`;
    ///
    /// with `F` from [`Self::svd_gap_reciprocals`]. The singular vector
    /// tangents are exact for distinct nonzero singular values; a repeated
    /// value gets the masked `F` (finite but not a derivative) and a zero
    /// value makes the `S^-1` term infinite or NaN, as in JAX.
    fn svd_jvp(
        &mut self,
        matrix: TensorNodeId,
        tangent: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<TensorNodeId, String> {
        let (m, n) = self.matrix_extents(matrix)?;
        let kind = match kind {
            LinalgKind::SvdUFull if m <= n => LinalgKind::SvdU,
            LinalgKind::SvdVhFull if n <= m => LinalgKind::SvdVh,
            LinalgKind::SvdUFull | LinalgKind::SvdVhFull => {
                return Err(Self::completion_derivative_error(kind))
            }
            kind => kind,
        };
        let (u, s, vh) = self.svd_factors(matrix)?;
        let v = self.matrix_transpose(vh)?;
        let u_transposed = self.matrix_transpose(u)?;
        let projected = self.matmul(u_transposed, tangent)?;
        let rotated = self.matmul(projected, v)?;
        if kind == LinalgKind::SvdS {
            return self.matrix_diagonal(rotated);
        }
        let gaps = self.svd_gap_reciprocals(s)?;
        let (row, column) = self.row_and_column(s)?;
        if kind == LinalgKind::SvdU {
            let scaled = self.mul(rotated, row)?;
            let scaled_transposed = self.matrix_transpose(scaled)?;
            let symmetric = self.add(scaled, scaled_transposed)?;
            let inner = self.mul(gaps, symmetric)?;
            let du = self.matmul(u, inner)?;
            if m <= n {
                return Ok(du);
            }
            let applied = self.matmul(tangent, v)?;
            let inside = self.matmul(u_transposed, applied)?;
            let inside = self.matmul(u, inside)?;
            let outside = self.sub(applied, inside)?;
            let outside = self.div(outside, row)?;
            return self.add(du, outside);
        }
        let scaled = self.mul(column, rotated)?;
        let scaled_transposed = self.matrix_transpose(scaled)?;
        let symmetric = self.add(scaled, scaled_transposed)?;
        let inner = self.mul(gaps, symmetric)?;
        let mut dv = self.matmul(v, inner)?;
        if m < n {
            let tangent_transposed = self.matrix_transpose(tangent)?;
            let applied = self.matmul(tangent_transposed, u)?;
            let inside = self.matmul(vh, applied)?;
            let inside = self.matmul(v, inside)?;
            let outside = self.sub(applied, inside)?;
            let outside = self.div(outside, row)?;
            dv = self.add(dv, outside)?;
        }
        self.matrix_transpose(dv)
    }

    /// The cotangent of `A` for one SVD output, the transpose of
    /// [`Self::svd_jvp`]:
    ///
    /// - `s`: `U diag(g) Vh`;
    /// - `U`: with `J = F o (U^T G)`, `U (J + J^T) S Vh`, plus
    ///   `(I - U U^T) G S^-1 Vh` for `m > n`;
    /// - `Vh`: with `G_V = G^T` and `K = F o (V^T G_V)`, `U S (K + K^T) Vh`,
    ///   plus `U S^-1 G_V^T (I - V V^T)` for `m < n`.
    fn svd_vjp(
        &mut self,
        matrix: TensorNodeId,
        cotangent: TensorNodeId,
        kind: LinalgKind,
    ) -> Result<TensorNodeId, String> {
        let (m, n) = self.matrix_extents(matrix)?;
        let kind = match kind {
            LinalgKind::SvdUFull if m <= n => LinalgKind::SvdU,
            LinalgKind::SvdVhFull if n <= m => LinalgKind::SvdVh,
            LinalgKind::SvdUFull | LinalgKind::SvdVhFull => {
                return Err(Self::completion_derivative_error(kind))
            }
            kind => kind,
        };
        let (u, s, vh) = self.svd_factors(matrix)?;
        if kind == LinalgKind::SvdS {
            let (row, _) = self.row_and_column(cotangent)?;
            let scaled = self.mul(u, row)?;
            return self.matmul(scaled, vh);
        }
        let gaps = self.svd_gap_reciprocals(s)?;
        let (row, column) = self.row_and_column(s)?;
        if kind == LinalgKind::SvdU {
            let u_transposed = self.matrix_transpose(u)?;
            let projected = self.matmul(u_transposed, cotangent)?;
            let inner = self.mul(gaps, projected)?;
            let inner_transposed = self.matrix_transpose(inner)?;
            let symmetric = self.add(inner, inner_transposed)?;
            let scaled = self.mul(symmetric, row)?;
            let mut core = self.matmul(u, scaled)?;
            if m > n {
                let inside = self.matmul(u, projected)?;
                let outside = self.sub(cotangent, inside)?;
                let outside = self.div(outside, row)?;
                core = self.add(core, outside)?;
            }
            return self.matmul(core, vh);
        }
        let v = self.matrix_transpose(vh)?;
        let v_cotangent = self.matrix_transpose(cotangent)?;
        let projected = self.matmul(vh, v_cotangent)?;
        let inner = self.mul(gaps, projected)?;
        let inner_transposed = self.matrix_transpose(inner)?;
        let symmetric = self.add(inner, inner_transposed)?;
        let scaled = self.mul(column, symmetric)?;
        let scaled = self.matmul(scaled, vh)?;
        let mut gradient = self.matmul(u, scaled)?;
        if m < n {
            let inside = self.matmul(v, projected)?;
            let outside = self.sub(v_cotangent, inside)?;
            let outside = self.matrix_transpose(outside)?;
            let scaled_u = self.div(u, row)?;
            let outside = self.matmul(scaled_u, outside)?;
            gradient = self.add(gradient, outside)?;
        }
        Ok(gradient)
    }

    /// Compact staged lower-triangle factorization. Validation follows the
    /// reference staged recurrence, which differs from the eager SPD contract.
    pub fn cholesky(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let source = self.node(input)?;
        let shape = source.shape.clone();
        if shape.len() < 2 || shape[shape.len() - 2] != shape[shape.len() - 1] {
            return Err(format!("cholesky reference tracing requires a square unbatched rank-2 tensor, got {shape:?}"));
        }
        if !source.dtype.is_floating() {
            return Err("cholesky requires a floating-point tensor".into());
        }
        self.push_derived(TensorOp::Cholesky { input }, shape)
    }

    fn cholesky_ad(
        &mut self,
        inputs: Vec<TensorNodeId>,
        kind: CholeskyAdKind,
    ) -> Result<TensorNodeId, String> {
        let expected = if matches!(kind, CholeskyAdKind::Jvp | CholeskyAdKind::Vjp) {
            2
        } else {
            4
        };
        if inputs.len() != expected {
            return Err("invalid Cholesky derivative arity".into());
        }
        let shape = self.node(inputs[0])?.shape.clone();
        for input in &inputs {
            if self.node(*input)?.shape != shape {
                return Err("Cholesky derivative operand shape mismatch".into());
            }
        }
        let dtype = self.node(inputs[0])?.dtype;
        Ok(self.push_node(TensorOp::CholeskyAd { inputs, kind }, shape, dtype, false))
    }

    // Preserve the established scalar derivative and exceptional-value rules
    // until a compact differentiable device factorization is available.
    #[allow(clippy::needless_range_loop)] // The recurrence indexes earlier triangular rows.
    fn expanded_cholesky(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let n = shape[shape.len() - 1];
        if shape.len() > 2 {
            let count = element_count(&shape[..shape.len() - 2])?;
            let flattened = self.reshape(input, vec![count, n, n])?;
            let mut outputs = Vec::with_capacity(count);
            for batch in 0..count {
                let matrix = self.slice(flattened, 0, batch, 1)?;
                let matrix = self.reshape(matrix, vec![n, n])?;
                let factor = self.expanded_cholesky(matrix)?;
                outputs.push(self.reshape(factor, vec![1, n, n])?);
            }
            let stacked = self.concat(outputs, 0)?;
            return self.reshape(stacked, shape);
        }
        let mut rows: Vec<Vec<TensorNodeId>> = Vec::with_capacity(n);
        for row in 0..n {
            let mut current = Vec::with_capacity(n);
            for column in 0..n {
                if column > row {
                    current.push(self.scalar_constant(0.0));
                    continue;
                }
                let entry = self.slice(input, 0, row, 1)?;
                let entry = self.slice(entry, 1, column, 1)?;
                let mut reduced = self.reshape(entry, vec![])?;
                for inner in 0..column {
                    let other = if row == column {
                        current[inner]
                    } else {
                        rows[column][inner]
                    };
                    let product = self.mul(current[inner], other)?;
                    reduced = self.sub(reduced, product)?;
                }
                let value = if row == column {
                    self.sqrt(reduced)?
                } else {
                    self.div(reduced, rows[column][column])?
                };
                current.push(value);
            }
            rows.push(current);
        }
        let mut output_rows = Vec::with_capacity(n);
        for row in rows {
            let entries = row
                .into_iter()
                .map(|entry| self.reshape(entry, vec![1]))
                .collect::<Result<Vec<_>, _>>()?;
            let row = self.concat(entries, 0)?;
            output_rows.push(self.reshape(row, vec![1, n])?);
        }
        self.concat(output_rows, 0)
    }

    fn expanded_cholesky_ad(
        &mut self,
        inputs: &[TensorNodeId],
        kind: CholeskyAdKind,
    ) -> Result<TensorNodeId, String> {
        let shape = self.node(inputs[0])?.shape.clone();
        let dtype = self.node(inputs[0])?.dtype;
        let mut reference = TensorIr::new();
        let argument = reference.input_typed("matrix", shape, dtype)?;
        let factor = reference.expanded_cholesky(argument)?;
        let mut bindings = BTreeMap::from([("matrix".to_string(), inputs[0])]);
        let (graph, output) = match kind {
            CholeskyAdKind::Jvp | CholeskyAdKind::Mixed => {
                let first = reference.symbolic_jvp_with_tangent_inputs(
                    factor,
                    &BTreeMap::from([("matrix".into(), "direction".into())]),
                )?;
                bindings.insert("direction".into(), inputs[1]);
                if kind == CholeskyAdKind::Jvp {
                    (first.graph, first.tangent)
                } else {
                    let mixed = first.graph.symbolic_jvp_with_tangent_inputs(
                        first.tangent,
                        &BTreeMap::from([
                            ("matrix".into(), "second".into()),
                            ("direction".into(), "mixed".into()),
                        ]),
                    )?;
                    bindings.insert("second".into(), inputs[2]);
                    bindings.insert("mixed".into(), inputs[3]);
                    (mixed.graph, mixed.tangent)
                }
            }
            CholeskyAdKind::Vjp | CholeskyAdKind::VjpJvp => {
                let reverse = reference.symbolic_vjp(factor, "cotangent")?;
                let gradient = reverse.gradients["matrix"];
                if kind == CholeskyAdKind::Vjp {
                    bindings.insert("cotangent".into(), inputs[1]);
                    (reverse.graph, gradient)
                } else {
                    let direction = reverse.graph.symbolic_jvp_with_tangent_inputs(
                        gradient,
                        &BTreeMap::from([
                            ("matrix".into(), "direction".into()),
                            ("cotangent".into(), "cotangent_direction".into()),
                        ]),
                    )?;
                    bindings.insert("direction".into(), inputs[1]);
                    bindings.insert("cotangent".into(), inputs[2]);
                    bindings.insert("cotangent_direction".into(), inputs[3]);
                    (direction.graph, direction.tangent)
                }
            }
        };
        Ok(self.inline(&graph, &bindings, &[output])?[0])
    }

    fn expand_cholesky_for_ad(&self) -> Result<Option<(Self, Vec<TensorNodeId>)>, String> {
        // Native jets cover first and second order in either float dtype; a
        // derivative of a second-order node (third order) uses the expansion.
        if !self.nodes.iter().any(|node| {
            matches!(
                node.op,
                TensorOp::CholeskyAd {
                    kind: CholeskyAdKind::Mixed | CholeskyAdKind::VjpJvp,
                    ..
                }
            )
        }) {
            return Ok(None);
        }
        self.expand_all_cholesky().map(Some)
    }

    fn expand_all_cholesky(&self) -> Result<(Self, Vec<TensorNodeId>), String> {
        let mut graph = Self::new();
        let mut mapped = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.iter() {
            let id = if let TensorOp::Cholesky { input } = node.op {
                graph.expanded_cholesky(mapped[input])?
            } else if let TensorOp::CholeskyAd { inputs, kind } = &node.op {
                let inputs = inputs.iter().map(|id| mapped[*id]).collect::<Vec<_>>();
                graph.expanded_cholesky_ad(&inputs, *kind)?
            } else {
                let op = remap_tensor_op(&node.op, &|id| mapped.get(id).copied())?;
                graph.push_node(op, node.shape.clone(), node.dtype, node.weak)
            };
            mapped.push(id);
        }
        Ok((graph, mapped))
    }

    pub fn triangular(&mut self, input: TensorNodeId, lower: bool) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        if shape.len() < 2 {
            return Err(format!(
                "triangular projection requires at least rank two, got {:?}",
                shape
            ));
        }
        self.push_derived(TensorOp::Triangular { input, lower }, shape)
    }

    pub fn tanh(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Tanh)
    }

    pub fn exp(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Exp)
    }

    pub fn sqrt(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Sqrt { input }, shape)
    }

    fn sqrt_derivative(&mut self, input: TensorNodeId, order: u32) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::SqrtDerivative { input, order }, shape)
    }

    pub fn reshape(
        &mut self,
        input: TensorNodeId,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if element_count(&shape)? != element_count(&input_shape)? {
            return Err(format!(
                "cannot reshape tensor shape {:?} to {:?}",
                input_shape, shape
            ));
        }
        self.push_derived(TensorOp::Reshape { input }, shape)
    }

    pub fn mean(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.node(input)?;
        self.push_derived(TensorOp::Mean { input }, vec![])
    }

    pub fn mean_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axis = normalize_axis(axis, input_shape.len())?;
        self.push_derived(
            TensorOp::MeanAxis { input, axis },
            reduced_shape(&input_shape, axis)?,
        )
    }

    pub fn sin(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Sin)
    }

    pub fn cos(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Cos)
    }

    pub fn powi(&mut self, input: TensorNodeId, exponent: u32) -> Result<TensorNodeId, String> {
        if exponent > i32::MAX as u32 {
            return Err("powi exponent must fit in a signed 32-bit integer".to_string());
        }
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Powi { input, exponent }, shape)
    }

    /// The value of `id` when it is a scalar constant, possibly behind the
    /// casts that promotion inserts for weak scalars, rounded like the cast
    /// chain. The `pow` rules use it to skip the derivative of a constant
    /// operand and to fold the constants of the derivative expressions.
    fn scalar_constant_value(&self, id: TensorNodeId) -> Option<f64> {
        let node = self.nodes.get(id)?;
        match node.op {
            TensorOp::ScalarConstant { value } if node.shape.is_empty() => Some(value),
            TensorOp::Cast { input } => Some(node.dtype.round(self.scalar_constant_value(input)?)),
            _ => None,
        }
    }

    pub fn transpose(
        &mut self,
        input: TensorNodeId,
        axes: Option<Vec<isize>>,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axes = normalize_permutation(axes, input_shape.len())?;
        let shape = axes.iter().map(|axis| input_shape[*axis]).collect();
        self.push_derived(TensorOp::Transpose { input, axes }, shape)
    }

    pub fn log(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Log)
    }

    /// Elementwise `log1p`; see [`UnaryMathKind::Log1p`].
    pub fn log1p(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Log1p)
    }

    /// Elementwise `expm1`; see [`UnaryMathKind::Expm1`].
    pub fn expm1(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Expm1)
    }

    /// Elementwise `erf`; see [`UnaryMathKind::Erf`].
    pub fn erf(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Erf)
    }

    /// Elementwise `erfc`; see [`UnaryMathKind::Erfc`].
    pub fn erfc(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.unary_math(input, UnaryMathKind::Erfc)
    }

    /// `input` with a zero derivative; see [`TensorOp::StopGradient`]. `Bool`
    /// inputs are accepted, as for other data movement.
    pub fn stop_gradient(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::StopGradient { input }, shape)
    }

    /// Inclusive prefix sum along `axis` (negative axes count from the end);
    /// see [`TensorOp::CumSum`].
    pub fn cumsum(
        &mut self,
        input: TensorNodeId,
        axis: isize,
        reverse: bool,
    ) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let axis = normalize_axis(axis, shape.len())?;
        self.push_derived(
            TensorOp::CumSum {
                input,
                axis,
                reverse,
            },
            shape,
        )
    }

    /// The product of the entries of `input` along `axis` (negative axes
    /// count from the end), which is removed from the shape.
    ///
    /// The product is a pairwise tree of ordinary `mul` nodes: each level
    /// multiplies entry `k` with entry `k + h` of the `2h` leading entries and
    /// carries an odd last entry over unchanged, so an extent `n` takes
    /// `ceil(log2 n)` levels and every result is rounded at most that many
    /// times. No division and no logarithm is involved, so the derivatives
    /// are those of the multiplications: `d prod / d x_i` is the product of
    /// the other entries, exactly zero away from a single zero entry and zero
    /// everywhere for two or more zeros, and every derivative order is again
    /// a graph of `mul`, `slice`, and `concat` nodes on every backend. This
    /// is also the shape of JAX's `reduce_prod` derivative rule.
    pub fn prod_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        let source = self.node(input)?;
        if source.dtype == TensorDType::Bool {
            return Err(
                "prod is not defined for bool tensors; convert explicitly with astype".to_string(),
            );
        }
        let mut shape = source.shape.clone();
        let axis = normalize_axis(axis, shape.len())?;
        let mut extent = shape[axis];
        if extent == 0 {
            return Err("prod requires a non-empty reduced axis".to_string());
        }
        let mut current = input;
        while extent > 1 {
            let half = extent / 2;
            let lower = self.slice(current, axis, 0, half)?;
            let upper = self.slice(current, axis, half, half)?;
            let mut next = self.mul(lower, upper)?;
            if extent % 2 == 1 {
                let carried = self.slice(current, axis, 2 * half, 1)?;
                next = self.concat(vec![next, carried], axis as isize)?;
            }
            current = next;
            extent = half + extent % 2;
        }
        shape.remove(axis);
        self.reshape(current, shape)
    }

    pub fn concat(
        &mut self,
        inputs: Vec<TensorNodeId>,
        axis: isize,
    ) -> Result<TensorNodeId, String> {
        if inputs.is_empty() {
            return Err("concat requires at least one input".to_string());
        }
        let shapes = inputs
            .iter()
            .map(|input| self.node(*input).map(|node| node.shape.as_slice()))
            .collect::<Result<Vec<_>, _>>()?;
        let axis = normalize_axis(axis, shapes[0].len())?;
        let shape = concat_shape(&shapes, axis)?;
        let mut inputs = inputs;
        self.coerce_operand_slice("concat", &mut inputs)?;
        self.push_derived(TensorOp::Concat { inputs, axis }, shape)
    }

    pub fn broadcast_to(
        &mut self,
        input: TensorNodeId,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if broadcast_shape(&input_shape, &shape)? != shape {
            return Err(format!(
                "cannot broadcast tensor shape {input_shape:?} to {shape:?}"
            ));
        }
        self.push_derived(TensorOp::Broadcast { input }, shape)
    }

    fn slice(
        &mut self,
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if axis >= input_shape.len()
            || start
                .checked_add(length)
                .is_none_or(|end| end > input_shape[axis])
        {
            return Err(format!(
                "invalid slice [{start}..{}) on axis {axis} for shape {input_shape:?}",
                start + length
            ));
        }
        let mut shape = input_shape;
        shape[axis] = length;
        self.push_derived(
            TensorOp::Slice {
                input,
                axis,
                start,
                length,
            },
            shape,
        )
    }

    pub fn slice_axis(
        &mut self,
        input: TensorNodeId,
        axis: isize,
        start: usize,
        stop: usize,
    ) -> Result<TensorNodeId, String> {
        let rank = self.node(input)?.shape.len();
        let axis = normalize_axis(axis, rank)?;
        let length = stop.checked_sub(start).ok_or_else(|| {
            format!("slice stop {stop} must be greater than or equal to start {start}")
        })?;
        self.slice(input, axis, start, length)
    }

    pub fn pad_slice(
        &mut self,
        input: TensorNodeId,
        output_shape: Vec<usize>,
        axis: usize,
        start: usize,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if input_shape.len() != output_shape.len()
            || axis >= output_shape.len()
            || input_shape
                .iter()
                .enumerate()
                .any(|(index, extent)| index != axis && *extent != output_shape[index])
            || start
                .checked_add(input_shape[axis])
                .is_none_or(|end| end > output_shape[axis])
        {
            return Err(format!(
                "cannot pad tensor shape {input_shape:?} into {output_shape:?} along axis {axis}"
            ));
        }
        self.push_derived(TensorOp::PadSlice { input, axis, start }, output_shape)
    }

    /// The entries of `input` at `indices` along `axis` (already
    /// normalized); see [`TensorOp::Gather`].
    pub fn gather(
        &mut self,
        input: TensorNodeId,
        indices: Arc<[usize]>,
        axis: usize,
    ) -> Result<TensorNodeId, String> {
        let mut shape = self.node(input)?.shape.clone();
        let extent = *shape
            .get(axis)
            .ok_or_else(|| format!("gather axis {axis} is out of bounds for shape {shape:?}"))?;
        if indices.is_empty() {
            return Err("gather indices must not be empty".to_string());
        }
        if let Some(index) = indices.iter().find(|index| **index >= extent) {
            return Err(format!(
                "gather index {index} is out of bounds for axis {axis} with extent {extent}"
            ));
        }
        shape[axis] = indices.len();
        self.push_derived(
            TensorOp::Gather {
                input,
                indices,
                axis,
            },
            shape,
        )
    }

    /// `base` plus `updates` scattered to `indices` along `axis` (already
    /// normalized); the operands are promoted like `add`. See
    /// [`TensorOp::ScatterAdd`].
    pub fn scatter_add(
        &mut self,
        base: TensorNodeId,
        updates: TensorNodeId,
        indices: Arc<[usize]>,
        axis: usize,
    ) -> Result<TensorNodeId, String> {
        let shape = self.node(base)?.shape.clone();
        let update_shape = &self.node(updates)?.shape;
        let extent = *shape
            .get(axis)
            .ok_or_else(|| format!("scatter axis {axis} is out of bounds for shape {shape:?}"))?;
        if indices.is_empty() {
            return Err("scatter indices must not be empty".to_string());
        }
        if let Some(index) = indices.iter().find(|index| **index >= extent) {
            return Err(format!(
                "scatter index {index} is out of bounds for axis {axis} with extent {extent}"
            ));
        }
        let mut expected = shape.clone();
        expected[axis] = indices.len();
        if *update_shape != expected {
            return Err(format!(
                "scatter updates shape {update_shape:?} is incompatible with base shape \
                 {shape:?}, axis {axis}, and {} indices",
                indices.len()
            ));
        }
        let [base, updates] = self.coerce_operands("scatter_add", [base, updates])?;
        self.push_derived(
            TensorOp::ScatterAdd {
                base,
                updates,
                indices,
                axis,
            },
            shape,
        )
    }

    /// `base`, the cotangent accumulated so far for the input of
    /// `gather(input, indices, axis)` (zeros if there is none yet), plus the
    /// gather's cotangent `upstream`. Reverse mode used to visit one slice per
    /// index, from the last index to the first, adding each into the running
    /// sum; scattering repeated entries in decreasing index order into that
    /// same running sum reproduces those sums bit for bit.
    fn gather_cotangent(
        &mut self,
        base: TensorNodeId,
        upstream: TensorNodeId,
        indices: &Arc<[usize]>,
        axis: usize,
    ) -> Result<TensorNodeId, String> {
        let mut seen = HashSet::new();
        if indices.iter().all(|index| seen.insert(*index)) {
            return self.scatter_add(base, upstream, indices.clone(), axis);
        }
        let positions = (0..indices.len()).rev().collect::<Arc<[usize]>>();
        let reversed = self.gather(upstream, positions, axis)?;
        let reversed_indices = indices.iter().rev().copied().collect::<Arc<[usize]>>();
        self.scatter_add(base, reversed, reversed_indices, axis)
    }

    pub fn evaluate(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.node(output)?;
        let values = self.evaluate_all(inputs)?;
        values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))
    }

    pub fn node_shape(&self, id: TensorNodeId) -> Result<Vec<usize>, String> {
        Ok(self.node(id)?.shape.clone())
    }

    pub fn input_node_id(&self, name: &str) -> Result<TensorNodeId, String> {
        self.nodes
            .iter()
            .position(
                |node| matches!(&node.op, TensorOp::Input { name: candidate } if candidate == name),
            )
            .ok_or_else(|| format!("input {name:?} does not exist"))
    }

    /// Returns `output` plus a zero term that references each named input
    /// without reading its values, so compiling the result keeps those inputs
    /// live. Region builders use it to give branches one capture interface.
    pub fn retain_inputs(
        &mut self,
        input_names: &[String],
        output: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        symbolic_retain_region_inputs(self, input_names, output)
    }

    pub fn compile_cpu(&self, output: TensorNodeId) -> Result<TensorExecutionPlan, String> {
        let (plan, _) = self.compile_cpu_many(&[output])?;
        Ok(plan)
    }

    /// Freezes the union of several output-reachable subgraphs into one plan.
    ///
    /// The returned node ids correspond to `outputs` in order after DCE and
    /// structural CSE. Backends can execute the shared prefix once and consume
    /// each retained output directly, which is required for coherent
    /// multi-parameter reverse-mode updates.
    pub fn compile_cpu_many(
        &self,
        outputs: &[TensorNodeId],
    ) -> Result<(TensorExecutionPlan, Vec<TensorNodeId>), String> {
        if outputs.is_empty() {
            return Err("execution plan requires at least one output".to_string());
        }
        for output in outputs {
            self.node(*output)?;
        }
        let reachable = tensor_output_reachability(&self.nodes, outputs);
        let reachable_count = reachable.iter().filter(|keep| **keep).count();
        // Node ids are dense and strictly below the allocated node count.
        let mut remap = TensorNodeRemap::new(self.nodes.len(), reachable_count);
        let mut cse_nodes = HashMap::new();
        // Array constants by `TensorConstant::bits_hash`; candidates in one bucket are compared
        // bitwise, so a hash collision never merges different values.
        let mut constant_nodes = HashMap::<u64, Vec<TensorNodeId>>::new();
        let mut nodes = Vec::with_capacity(reachable_count);
        let mut source_uses = vec![0usize; self.nodes.len()];
        for &output in outputs {
            source_uses[output] += 1;
        }
        for (node_id, &keep) in reachable.iter().enumerate() {
            if !keep {
                continue;
            }
            for input in tensor_op_inputs(&self.nodes[node_id].op) {
                source_uses[input] += 1;
            }
        }
        let mut pending_uses = vec![0usize; self.nodes.len()];
        let mut retained = vec![false; self.nodes.len()];
        for (old_id, node) in self.nodes.iter().enumerate() {
            if !reachable[old_id] {
                continue;
            }
            let source_inputs = tensor_op_inputs(&node.op);
            let compiled_id = (|| -> Result<TensorNodeId, String> {
                let mut op = remap_tensor_op(&node.op, &|id| remap.get(id))?;
                if let Some(alias) =
                    canonicalize_tensor_op(&mut op, &node.shape, node.dtype, &nodes)?
                {
                    return Ok(alias);
                }
                if let Some(value) = fold_scalar_constant_op(&op, &nodes) {
                    // Folded values are rounded to the node dtype, matching per-node execution
                    // semantics.
                    op = TensorOp::ScalarConstant {
                        value: node.dtype.round(value),
                    };
                } else if let Some(value) = fold_tensor_constant_op(&op, &nodes) {
                    op = TensorOp::Constant {
                        value: TensorConstant::new(value.into_dtype(node.dtype)),
                    };
                }
                if let TensorOp::Constant { value } = &op {
                    let bucket = constant_nodes.entry(value.bits_hash()).or_default();
                    let existing = bucket.iter().copied().find(|id| {
                    let existing: &TensorNode = &nodes[*id];
                    existing.weak == node.weak
                        && matches!(&existing.op, TensorOp::Constant { value: other } if other.same_bits(value))
                });
                    if let Some(existing_id) = existing {
                        return Ok(existing_id);
                    }
                    bucket.push(nodes.len());
                }
                if let Some(key) = pure_tensor_op_cse_key(&op, &node.shape) {
                    // dtype and weakness are part of a value's identity: cast(x, f32) must not be
                    // merged with x.
                    let key = (key, node.dtype, node.weak);
                    if let Some(existing_id) = cse_nodes.get(&key) {
                        return Ok(*existing_id);
                    }
                    cse_nodes.insert(key, nodes.len());
                }
                // A surviving operation pins its operands for execution. Folded constants
                // have no runtime dependencies and may release their temporary inputs.
                for input in tensor_op_inputs(&op) {
                    retained[input] = true;
                }
                let compiled_id = nodes.len();
                nodes.push(TensorNode {
                    op,
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    weak: node.weak,
                });
                Ok(compiled_id)
            })()?;
            remap.insert(old_id, compiled_id);
            pending_uses[compiled_id] += source_uses[old_id];
            for input in source_inputs {
                let id = remap
                    .get(input)
                    .expect("operand remap was validated while compiling the node");
                pending_uses[id] -= 1;
                release_folded_constant(&mut nodes, id, pending_uses[id], retained[id]);
            }
            release_folded_constant(
                &mut nodes,
                compiled_id,
                pending_uses[compiled_id],
                retained[compiled_id],
            );
        }
        let output_node_ids = outputs
            .iter()
            .map(|output| {
                remap
                    .get(*output)
                    .ok_or_else(|| format!("output node {output} is not reachable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Folding and CSE can make nodes retained by the source-graph DCE
        // unreachable. Compact the finalized plan so backends never allocate
        // buffers or launch work for those obsolete intermediates.
        let (nodes, output_node_ids) = prune_unreachable_tensor_nodes(nodes, &output_node_ids)?;
        let output_node_id = *output_node_ids
            .first()
            .ok_or_else(|| "execution plan requires at least one output".to_string())?;
        let fused_elementwise_output =
            output_node_ids.len() == 1 && is_fusable_elementwise_subgraph(&nodes, output_node_id);
        Ok((
            TensorExecutionPlan {
                input_nodes: Arc::new(tensor_input_nodes(&nodes)),
                nodes: Arc::new(nodes),
                output_node_id,
                output_node_ids: output_node_ids.clone(),
                fused_elementwise_output,
            },
            output_node_ids,
        ))
    }

    pub fn vjp(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        self.value_and_vjp(output, inputs, output_cotangent)
            .map(|(_, gradients)| gradients)
    }

    pub fn value_and_vjp(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        let (mut values, gradients) =
            self.value_and_vjp_many(&[(output, output_cotangent)], inputs)?;
        let value = values
            .pop()
            .ok_or_else(|| "single-output VJP produced no value".to_string())?;
        Ok((value, gradients))
    }

    /// Evaluates several outputs and backpropagates their cotangents together.
    ///
    /// Shared subgraphs are visited once, so structured multi-carry loops do
    /// not repeat the body reverse pass for every carry component.
    pub fn value_and_vjp_many(
        &self,
        outputs: &[(TensorNodeId, DynamicTensor)],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(Vec<DynamicTensor>, BTreeMap<String, DynamicTensor>), String> {
        if let Some((expanded, mapped)) = self.expand_cholesky_for_ad()? {
            let outputs = outputs
                .iter()
                .map(|(output, seed)| {
                    mapped
                        .get(*output)
                        .copied()
                        .map(|id| (id, seed.clone()))
                        .ok_or_else(|| format!("output node {output} does not exist"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return expanded.value_and_vjp_many(&outputs, inputs);
        }
        if outputs.is_empty() {
            return Err("multi-output VJP requires at least one output".to_string());
        }
        let retained = tensor_reverse_retained_primals(&self.nodes, outputs);
        let mut values =
            Self::evaluate_tensor_nodes_with_outputs(&self.nodes, inputs, Some(&retained))?;
        let output_values = outputs
            .iter()
            .map(|(output, cotangent)| {
                let output_node = self.node(*output)?;
                ensure_differentiable_output(output_node)?;
                if cotangent.shape != output_node.shape {
                    return Err(format!(
                        "output cotangent shape {:?} does not match output shape {:?}",
                        cotangent.shape, output_node.shape
                    ));
                }
                values
                    .get(*output)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| format!("output node {output} has no value"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut cotangents = vec![None; self.nodes.len()];
        for (output, cotangent) in outputs {
            accumulate(&mut cotangents[*output], cotangent.clone())?;
        }
        let mut processed_scan_groups = HashSet::new();

        for node_id in (0..self.nodes.len()).rev() {
            // Input cotangents are the returned gradients. Other cotangents have no
            // remaining consumers once their reverse rule has run.
            if matches!(self.nodes[node_id].op, TensorOp::Input { .. }) {
                values[node_id] = None;
                continue;
            }
            let cotangent = match cotangents[node_id].take() {
                Some(value) => value,
                None => {
                    values[node_id] = None;
                    continue;
                }
            };
            match &self.nodes[node_id].op {
                TensorOp::Input { .. }
                | TensorOp::ScalarConstant { .. }
                | TensorOp::Constant { .. } => {}
                TensorOp::Cast { input } => {
                    // Bool sources are not differentiable: a mask converted to float returns no
                    // gradient.
                    let source_dtype = self.node(*input)?.dtype;
                    if source_dtype != TensorDType::Bool {
                        accumulate(&mut cotangents[*input], cotangent.astype(source_dtype))?;
                    }
                }
                TensorOp::Add { lhs, rhs } => {
                    let lhs_contribution = cotangent.reduce_to_shape(&self.node(*lhs)?.shape)?;
                    let rhs_contribution = cotangent.reduce_to_shape(&self.node(*rhs)?.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Sub { lhs, rhs } => {
                    let lhs_contribution = cotangent.reduce_to_shape(&self.node(*lhs)?.shape)?;
                    let rhs_contribution =
                        cotangent.neg()?.reduce_to_shape(&self.node(*rhs)?.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Div { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    // As in the symbolic rule: g/r and -(g/r)(l/r), no r*r.
                    let scaled = cotangent.div(rhs_value)?;
                    let rhs_contribution = scaled
                        .mul(&lhs_value.div(rhs_value)?)?
                        .neg()?
                        .reduce_to_shape(&rhs_value.shape)?;
                    let lhs_contribution = scaled.reduce_to_shape(&lhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let lhs_contribution = cotangent
                        .mul(rhs_value)?
                        .reduce_to_shape(&lhs_value.shape)?;
                    let rhs_contribution = cotangent
                        .mul(lhs_value)?
                        .reduce_to_shape(&rhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Greater { .. } | TensorOp::Compare { .. } => {}
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let condition_value = values
                        .get(*condition)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {condition} has no evaluated value"))?;
                    let zero = DynamicTensor::filled(vec![], 0.0)?;
                    let true_contribution = condition_value
                        .where_select(&cotangent, &zero)?
                        .reduce_to_shape(&self.node(*on_true)?.shape)?;
                    let false_contribution = condition_value
                        .where_select(&zero, &cotangent)?
                        .reduce_to_shape(&self.node(*on_false)?.shape)?;
                    accumulate(&mut cotangents[*on_true], true_contribution)?;
                    accumulate(&mut cotangents[*on_false], false_contribution)?;
                }
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => {
                    let predicate = tensor_scalar_predicate(
                        values
                            .get(*predicate)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {predicate} has no evaluated value"))?,
                    )?;
                    let branch_inputs = tensor_forward_capture_values(captures, &values)?;
                    let (_, branch_gradients) =
                        branches.value_and_vjp(predicate, &branch_inputs, cotangent)?;
                    for (name, gradient) in branch_gradients {
                        let (_, capture) = captures
                            .iter()
                            .find(|(capture_name, _)| capture_name == &name)
                            .ok_or_else(|| {
                                format!("conditional gradient {name:?} has no parent capture")
                            })?;
                        accumulate(&mut cotangents[*capture], gradient)?;
                    }
                }
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let external = tensor_forward_capture_values(captures, &values)?;
                    let (_, carry_gradient, external_gradients) = loop_plan.value_and_vjp(
                        values
                            .get(*carry)
                            .and_then(Option::as_ref)
                            .cloned()
                            .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                        &external,
                        cotangent,
                    )?;
                    accumulate(&mut cotangents[*carry], carry_gradient)?;
                    for (name, gradient) in external_gradients {
                        let (_, capture) = captures
                            .iter()
                            .find(|(capture_name, _)| capture_name == &name)
                            .ok_or_else(|| {
                                format!("fori gradient {name:?} has no parent capture")
                            })?;
                        accumulate(&mut cotangents[*capture], gradient)?;
                    }
                }
                TensorOp::While { .. } => {
                    return Err(WHILE_LOOP_REVERSE_MODE_ERROR.to_string());
                }
                TensorOp::ForiVjp { .. } => {
                    return Err(
                        "direct VJP through a Fori VJP result is not implemented".to_string()
                    )
                }
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "direct VJP through a Fori JVP result is not implemented".to_string()
                    )
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "direct VJP through a Fori VJP JVP result is not implemented".to_string(),
                    )
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    group,
                    ..
                } => {
                    if processed_scan_groups.insert(*group) {
                        let mut current_cotangent = Some(cotangent);
                        let mut carry_cotangent = None;
                        let mut output_cotangent = None;
                        for (scan_node_id, scan_node) in self.nodes.iter().enumerate() {
                            let TensorOp::Scan {
                                target,
                                group: candidate_group,
                                ..
                            } = &scan_node.op
                            else {
                                continue;
                            };
                            if candidate_group != group {
                                continue;
                            }
                            let group_cotangent = if scan_node_id == node_id {
                                current_cotangent.take()
                            } else {
                                cotangents[scan_node_id].take()
                            };
                            let group_cotangent = match group_cotangent {
                                Some(value) => value,
                                None => DynamicTensor::filled(scan_node.shape.clone(), 0.0)?,
                            };
                            match target {
                                TensorScanTarget::Carry => {
                                    if carry_cotangent.replace(group_cotangent).is_some() {
                                        return Err(format!(
                                            "scan group {group} has multiple carry results"
                                        ));
                                    }
                                }
                                TensorScanTarget::Outputs => {
                                    if output_cotangent.replace(group_cotangent).is_some() {
                                        return Err(format!(
                                            "scan group {group} has multiple output results"
                                        ));
                                    }
                                }
                            }
                        }
                        let carry_cotangent = match carry_cotangent {
                            Some(cotangent) => cotangent,
                            None => DynamicTensor::filled(scan_plan.carry_shape()?, 0.0)?,
                        };
                        let output_cotangent = match output_cotangent {
                            Some(cotangent) => cotangent,
                            None => DynamicTensor::filled(scan_plan.output_shape()?, 0.0)?,
                        };
                        let external_inputs = tensor_forward_capture_values(captures, &values)?;
                        let (carry_gradient, external_gradients) = scan_plan.vjp(
                            values
                                .get(*carry)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            &external_inputs,
                            carry_cotangent,
                            output_cotangent,
                        )?;
                        accumulate(&mut cotangents[*carry], carry_gradient)?;
                        for (name, gradient) in external_gradients {
                            let (_, capture) = captures
                                .iter()
                                .find(|(capture_name, _)| capture_name == &name)
                                .ok_or_else(|| {
                                    format!("scan gradient {name:?} has no parent capture")
                                })?;
                            accumulate(&mut cotangents[*capture], gradient)?;
                        }
                    }
                }
                TensorOp::ScanVjp { .. } => {
                    return Err(
                        "direct VJP through a Scan VJP result is not implemented".to_string()
                    )
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "direct VJP through a Scan VJP JVP result is not implemented".to_string(),
                    )
                }
                TensorOp::Sum { input } => {
                    let contribution = cotangent.broadcast_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::SumAxis { input, axis } => {
                    let contribution =
                        cotangent.expand_reduced_axis(&self.node(*input)?.shape, *axis)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::ExtremumAxis { input, axis, .. } => {
                    let input_value = values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let output_value = values
                        .get(node_id)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    let contribution =
                        input_value.extremum_axis_cotangent(output_value, &cotangent, *axis)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let lhs_contribution = cotangent
                        .matmul(&rhs_value.transpose_last_two()?)?
                        .reduce_to_shape(&lhs_value.shape)?;
                    let rhs_contribution = lhs_value
                        .transpose_last_two()?
                        .matmul(&cotangent)?
                        .reduce_to_shape(&rhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Cholesky { input } => {
                    let value = values[*input]
                        .as_ref()
                        .ok_or_else(|| format!("node {input} has no value"))?;
                    accumulate(
                        &mut cotangents[*input],
                        cholesky::evaluate(CholeskyAdKind::Vjp, &[value, &cotangent])?,
                    )?;
                }
                TensorOp::CholeskyAd { inputs, kind } => {
                    let value = values[inputs[0]].as_ref().ok_or("missing Cholesky value")?;
                    let other = values[inputs[1]]
                        .as_ref()
                        .ok_or("missing Cholesky derivative operand")?;
                    let zero = DynamicTensor::filled(self.nodes[node_id].shape.clone(), 0.0)?;
                    let (first, second) = match kind {
                        CholeskyAdKind::Jvp => (
                            cholesky::evaluate(
                                CholeskyAdKind::VjpJvp,
                                &[value, other, &cotangent, &zero],
                            )?,
                            cholesky::evaluate(CholeskyAdKind::Vjp, &[value, &cotangent])?,
                        ),
                        CholeskyAdKind::Vjp => (
                            cholesky::evaluate(
                                CholeskyAdKind::VjpJvp,
                                &[value, &cotangent, other, &zero],
                            )?,
                            cholesky::evaluate(CholeskyAdKind::Jvp, &[value, &cotangent])?,
                        ),
                        _ => unreachable!("higher Cholesky derivatives expanded"),
                    };
                    accumulate(&mut cotangents[inputs[0]], first)?;
                    accumulate(&mut cotangents[inputs[1]], second)?;
                }
                TensorOp::Solve { matrix, rhs } => {
                    let matrix_value = values
                        .get(*matrix)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {matrix} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let output_value = values
                        .get(node_id)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    let transposed = matrix_value.transpose_last_two()?;
                    let rhs_contribution = transposed.solve(&cotangent)?;
                    let matrix_contribution = rhs_contribution
                        .matmul(&output_value.transpose_last_two()?)?
                        .mul(&DynamicTensor::filled(vec![], -1.0)?)?;
                    accumulate(&mut cotangents[*matrix], matrix_contribution)?;
                    let _ = rhs_value;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Triangular { input, lower } => {
                    accumulate(&mut cotangents[*input], cotangent.triangular(*lower)?)?;
                }
                TensorOp::Linalg { input, kind } => {
                    let matrix = values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = evaluate_linalg_derivative(
                        *kind,
                        LinalgDerivative::Vjp,
                        &[matrix, &cotangent],
                    )?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Sqrt { input } => {
                    let input_value = values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.sqrt_derivative(1)?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::SqrtDerivative { input, order } => {
                    let input_value = values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let next_order = order
                        .checked_add(1)
                        .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                    let contribution = cotangent
                        .mul(&input_value.sqrt_derivative(next_order)?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Reshape { input } => {
                    let contribution = cotangent.reshape(self.node(*input)?.shape.clone())?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Mean { input } => {
                    let input_shape = self.node(*input)?.shape.clone();
                    let input_count = element_count(&input_shape)?;
                    let contribution = cotangent
                        .broadcast_to_shape(&input_shape)?
                        .scale(1.0 / input_count as f64)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::MeanAxis { input, axis } => {
                    let input_shape = self.node(*input)?.shape.clone();
                    let contribution = cotangent
                        .expand_reduced_axis(&input_shape, *axis)?
                        .scale(1.0 / input_shape[*axis] as f64)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent == 0 {
                        continue;
                    }
                    let input_value = values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.powi(*exponent - 1)?)?
                        .scale(*exponent as f64)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    let lhs_value = values
                        .get(*lhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    // Constant operands are skipped, as in the symbolic rule.
                    for (index, operand) in [*lhs, *rhs].into_iter().enumerate() {
                        if self.scalar_constant_value(operand).is_some() {
                            continue;
                        }
                        let shape = &self.node(operand)?.shape;
                        let contribution =
                            match kind.numeric_partial(index, lhs_value, rhs_value)? {
                                Some(partial) => cotangent.mul(&partial)?.reduce_to_shape(shape)?,
                                None => cotangent.reduce_to_shape(shape)?,
                            };
                        accumulate(&mut cotangents[operand], contribution)?;
                    }
                }
                TensorOp::Transpose { input, axes } => {
                    let contribution = cotangent.transpose(&inverse_permutation(axes)?)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::UnaryMath { input, kind } => {
                    let input_value = values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let output_value = values
                        .get(node_id)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    if let Some(contribution) =
                        kind.numeric_chain(&cotangent, input_value, output_value)?
                    {
                        accumulate(
                            &mut cotangents[*input],
                            contribution.reduce_to_shape(&self.node(*input)?.shape)?,
                        )?;
                    }
                }
                TensorOp::StopGradient { .. } => {}
                TensorOp::Custom { rule, .. } => {
                    return Err(custom_rule_numeric_error(rule));
                }
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => accumulate(
                    &mut cotangents[*input],
                    cotangent.cumsum_axis(*axis, !*reverse, TensorDType::F64)?,
                )?,
                TensorOp::Concat { inputs, axis } => {
                    let mut start = 0;
                    for input in inputs {
                        let input_shape = &self.node(*input)?.shape;
                        accumulate(
                            &mut cotangents[*input],
                            cotangent.slice_axis(*axis, start, input_shape[*axis])?,
                        )?;
                        start += input_shape[*axis];
                    }
                }
                TensorOp::Slice {
                    input, axis, start, ..
                } => accumulate(
                    &mut cotangents[*input],
                    cotangent.pad_slice(&self.node(*input)?.shape, *axis, *start)?,
                )?,
                TensorOp::PadSlice { input, axis, start } => accumulate(
                    &mut cotangents[*input],
                    cotangent.slice_axis(*axis, *start, self.node(*input)?.shape[*axis])?,
                )?,
                // Scattered into the running sum in decreasing index order, as
                // in `TensorIr::gather_cotangent`.
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => {
                    let base = match cotangents[*input].take() {
                        Some(existing) => existing,
                        None => DynamicTensor::filled(self.node(*input)?.shape.clone(), 0.0)?,
                    };
                    cotangents[*input] = Some(base.scatter_add_axis(
                        &cotangent,
                        indices,
                        *axis,
                        TensorDType::F64,
                        true,
                    )?);
                }
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => {
                    accumulate(
                        &mut cotangents[*updates],
                        cotangent.gather_axis(indices, *axis)?,
                    )?;
                    accumulate(&mut cotangents[*base], cotangent)?;
                }
                TensorOp::Broadcast { input } => accumulate(
                    &mut cotangents[*input],
                    cotangent.reduce_to_shape(&self.node(*input)?.shape)?,
                )?,
            }
            values[node_id] = None;
        }

        // Eager backpropagation does derivative arithmetic in f64 and rounds by dtype only at casts
        // and input gradients; per-node dtype semantics are provided by the symbolic VJP.
        let mut gradients = BTreeMap::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let TensorOp::Input { name } = &node.op {
                // Bool inputs are not differentiable and do not appear in the gradient table
                // (matching the symbolic VJP).
                if node.dtype == TensorDType::Bool {
                    continue;
                }
                let gradient = match cotangents[node_id].take() {
                    Some(value) => value,
                    None => DynamicTensor::filled(node.shape.clone(), 0.0)?,
                };
                gradients.insert(name.clone(), gradient.into_dtype(node.dtype));
            }
        }
        Ok((output_values, gradients))
    }

    pub fn jvp(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.jvp_many(&[output], inputs, input_tangents)?
            .pop()
            .ok_or_else(|| format!("output node {output} has no value"))
    }

    // Direct input solves can share one finite LU factor without retaining factors for
    // an entire graph. Replay keeps the existing pivot/subtraction order and rounds
    // the primal before forming the tangent correction, including for F32 outputs.
    fn direct_solve_jvp(
        &self,
        outputs: &[TensorNodeId],
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Option<(DynamicTensor, DynamicTensor)>, String> {
        let Some((&output, [])) = outputs.split_first() else {
            return Ok(None);
        };
        if output + 1 != self.nodes.len() {
            return Ok(None);
        }
        let node = &self.nodes[output];
        let TensorOp::Solve { matrix, rhs } = node.op else {
            return Ok(None);
        };
        let prefix = &self.nodes[..output];
        if prefix
            .iter()
            .any(|node| !matches!(node.op, TensorOp::Input { .. }))
        {
            return Ok(None);
        }
        // Validate every primal before any tangent, as the general JVP does.
        let values =
            Self::evaluate_tensor_nodes_with_outputs(prefix, inputs, Some(&[matrix, rhs]))?;
        let matrix_value = values[matrix].as_ref().expect("retained solve matrix");
        let rhs_value = values[rhs].as_ref().expect("retained solve RHS");
        solve_shape(&matrix_value.shape, &rhs_value.shape)?;
        let Some(plan) = SolveReplayPlan::for_finite_dense(matrix_value) else {
            return Ok(None);
        };
        let Some(value) = plan.solve_finite(rhs_value) else {
            return Ok(None);
        };
        let value = value.into_dtype(node.dtype);
        let mut matrix_tangent = None;
        let mut rhs_tangent = None;
        for (id, input_node) in prefix.iter().enumerate() {
            let TensorOp::Input { name } = &input_node.op else {
                unreachable!("input-only prefix was checked above");
            };
            if input_node.dtype == TensorDType::Bool {
                if input_tangents
                    .get(name)
                    .is_some_and(|tangent| tangent.data.iter().any(|value| value != 0.0))
                {
                    return Err(bool_input_derivative_error(name));
                }
                continue;
            }
            let tangent = input_tangents
                .get(name)
                .ok_or_else(|| format!("missing input tangent {name:?}"))?;
            if tangent.shape != input_node.shape {
                return Err(format!(
                    "input tangent {name:?} has shape {:?}, expected {:?}",
                    tangent.shape, input_node.shape
                ));
            }
            if id == matrix {
                matrix_tangent = Some(tangent.astype(input_node.dtype));
            }
            if id == rhs {
                rhs_tangent = Some(tangent.astype(input_node.dtype));
            }
        }
        let correction = rhs_tangent.expect("floating solve RHS").sub(
            &matrix_tangent
                .expect("floating solve matrix")
                .matmul(&value)?,
        )?;
        let Some(tangent) = plan.solve_finite(&correction) else {
            return Ok(None);
        };
        Ok(Some((value, tangent.into_dtype(node.dtype))))
    }

    fn jvp_many(
        &self,
        outputs: &[TensorNodeId],
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<(DynamicTensor, DynamicTensor)>, String> {
        for &output in outputs {
            self.node(output)?;
        }
        if let Some((expanded, mapped)) = self.expand_cholesky_for_ad()? {
            let outputs = outputs
                .iter()
                .map(|output| mapped[*output])
                .collect::<Vec<_>>();
            return expanded.jvp_many(&outputs, inputs, input_tangents);
        }
        if let Some(result) = self.direct_solve_jvp(outputs, inputs, input_tangents)? {
            return Ok(vec![result]);
        }
        let values = self.evaluate_all(inputs)?;
        let mut tangents: Vec<DynamicTensor> = Vec::with_capacity(self.nodes.len());
        let mut scan_jvp_cache: HashMap<usize, TensorScanJvpEvaluation> = HashMap::new();

        for (node_id, node) in self.nodes.iter().enumerate() {
            let tangent = match &node.op {
                // A Bool input's tangent may only be omitted or zero (callers often supply a
                // tangent for every input, e.g. vmap JVP); a non-zero direction amounts to
                // differentiating with respect to a bool.
                TensorOp::Input { name } if node.dtype == TensorDType::Bool => {
                    if input_tangents
                        .get(name)
                        .is_some_and(|tangent| tangent.data.iter().any(|value| value != 0.0))
                    {
                        return Err(bool_input_derivative_error(name));
                    }
                    DynamicTensor::filled(node.shape.clone(), 0.0)?
                }
                TensorOp::Input { name } => {
                    let input_tangent = input_tangents
                        .get(name)
                        .ok_or_else(|| format!("missing input tangent {name:?}"))?;
                    if input_tangent.shape != node.shape {
                        return Err(format!(
                            "input tangent {name:?} has shape {:?}, expected {:?}",
                            input_tangent.shape, node.shape
                        ));
                    }
                    input_tangent.astype(node.dtype)
                }
                TensorOp::ScalarConstant { .. } => DynamicTensor::filled(vec![], 0.0)?,
                TensorOp::Constant { .. } => DynamicTensor::filled(node.shape.clone(), 0.0)?,
                TensorOp::Cast { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .astype(node.dtype),
                TensorOp::Add { lhs, rhs } => tangents
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?
                    .add(
                        tangents
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?,
                    )?,
                TensorOp::Sub { lhs, rhs } => tangents
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?
                    .sub(
                        tangents
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?,
                    )?,
                TensorOp::Div { lhs, rhs } => {
                    let lhs_tangent = tangents
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    lhs_tangent
                        .sub(&lhs_value.div(rhs_value)?.mul(rhs_tangent)?)?
                        .div(rhs_value)?
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_tangent = tangents
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    lhs_tangent
                        .mul(rhs_value)?
                        .add(&lhs_value.mul(rhs_tangent)?)?
                }
                TensorOp::Greater { .. } | TensorOp::Compare { .. } => {
                    DynamicTensor::filled(node.shape.clone(), 0.0)?
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => values
                    .get(*condition)
                    .ok_or_else(|| format!("node {condition} has no evaluated value"))?
                    .where_select(
                        tangents
                            .get(*on_true)
                            .ok_or_else(|| format!("node {on_true} has no evaluated tangent"))?,
                        tangents
                            .get(*on_false)
                            .ok_or_else(|| format!("node {on_false} has no evaluated tangent"))?,
                    )?,
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => {
                    let predicate = tensor_scalar_predicate(
                        values
                            .get(*predicate)
                            .ok_or_else(|| format!("node {predicate} has no evaluated value"))?,
                    )?;
                    let branch_inputs = tensor_cond_capture_values(captures, &values)?;
                    let branch_tangents = tensor_cond_capture_values(captures, &tangents)?;
                    branches.jvp(predicate, &branch_inputs, &branch_tangents)?.1
                }
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let external_inputs = tensor_fori_capture_values(captures, &values)?;
                    let external_tangents = tensor_fori_capture_values(captures, &tangents)?;
                    loop_plan
                        .jvp(
                            values
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            tangents
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated tangent"))?,
                            &external_inputs,
                            &external_tangents,
                        )?
                        .1
                }
                TensorOp::While {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let external_inputs = tensor_fori_capture_values(captures, &values)?;
                    let external_tangents = tensor_fori_capture_values(captures, &tangents)?;
                    loop_plan
                        .jvp(
                            values
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            tangents
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated tangent"))?,
                            &external_inputs,
                            &external_tangents,
                        )?
                        .1
                }
                TensorOp::ForiVjp { .. } => {
                    return Err("JVP through a Fori VJP result is not implemented".to_string())
                }
                TensorOp::ForiJvp { .. } => {
                    return Err("JVP through a Fori JVP result is not implemented".to_string())
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err("JVP through a Fori VJP JVP result is not implemented".to_string())
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !scan_jvp_cache.contains_key(group) {
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let external_tangents = tensor_fori_capture_values(captures, &tangents)?;
                        let (_, (carry_tangent, output_tangent)) =
                            scan_plan.jvp(
                                values.get(*carry).cloned().ok_or_else(|| {
                                    format!("node {carry} has no evaluated value")
                                })?,
                                tangents.get(*carry).cloned().ok_or_else(|| {
                                    format!("node {carry} has no evaluated tangent")
                                })?,
                                &external_inputs,
                                &external_tangents,
                            )?;
                        scan_jvp_cache.insert(
                            *group,
                            TensorScanJvpEvaluation {
                                carry_tangent,
                                output_tangent,
                            },
                        );
                    }
                    let cached = scan_jvp_cache.get(group).ok_or_else(|| {
                        format!("scan JVP group {group} was not cached after evaluation")
                    })?;
                    match target {
                        TensorScanTarget::Carry => cached.carry_tangent.clone(),
                        TensorScanTarget::Outputs => cached.output_tangent.clone(),
                    }
                }
                TensorOp::ScanVjp { .. } => {
                    return Err("JVP through a Scan VJP result is not implemented".to_string())
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err("JVP through a Scan VJP JVP result is not implemented".to_string())
                }
                TensorOp::Sum { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .sum_all()?,
                TensorOp::SumAxis { input, axis } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .reduce_axis(*axis, 1.0)?,
                TensorOp::ExtremumAxis { input, axis, .. } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .extremum_axis_tangent(
                        values
                            .get(node_id)
                            .ok_or_else(|| format!("node {node_id} has no evaluated value"))?,
                        tangents
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated tangent"))?,
                        *axis,
                    )?,
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_tangent = tangents
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    lhs_tangent
                        .matmul(rhs_value)?
                        .add(&lhs_value.matmul(rhs_tangent)?)?
                }
                TensorOp::Cholesky { input } => {
                    cholesky::evaluate(CholeskyAdKind::Jvp, &[&values[*input], &tangents[*input]])?
                }
                TensorOp::CholeskyAd { inputs, kind } => match kind {
                    CholeskyAdKind::Jvp => cholesky::evaluate(
                        CholeskyAdKind::Mixed,
                        &[
                            &values[inputs[0]],
                            &values[inputs[1]],
                            &tangents[inputs[0]],
                            &tangents[inputs[1]],
                        ],
                    )?,
                    CholeskyAdKind::Vjp => cholesky::evaluate(
                        CholeskyAdKind::VjpJvp,
                        &[
                            &values[inputs[0]],
                            &tangents[inputs[0]],
                            &values[inputs[1]],
                            &tangents[inputs[1]],
                        ],
                    )?,
                    _ => unreachable!("higher Cholesky derivatives expanded"),
                },
                TensorOp::Solve { matrix, rhs } => {
                    let matrix_value = values
                        .get(*matrix)
                        .ok_or_else(|| format!("node {matrix} has no evaluated value"))?;
                    let matrix_tangent = tangents
                        .get(*matrix)
                        .ok_or_else(|| format!("node {matrix} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    matrix_value.solve(&rhs_tangent.sub(&matrix_tangent.matmul(output_value)?)?)?
                }
                TensorOp::Linalg { input, kind } => evaluate_linalg_derivative(
                    *kind,
                    LinalgDerivative::Jvp,
                    &[&values[*input], &tangents[*input]],
                )?,
                TensorOp::Triangular { input, lower } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .triangular(*lower)?,
                TensorOp::Sqrt { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.sqrt_derivative(1)?)?
                }
                TensorOp::SqrtDerivative { input, order } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let next_order = order
                        .checked_add(1)
                        .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                    input_tangent.mul(&input_value.sqrt_derivative(next_order)?)?
                }
                TensorOp::Reshape { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .reshape(node.shape.clone())?,
                TensorOp::Mean { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .mean_all()?,
                TensorOp::MeanAxis { input, axis } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .reduce_axis(*axis, 1.0 / self.node(*input)?.shape[*axis] as f64)?,
                TensorOp::Powi { input, exponent } => {
                    if *exponent == 0 {
                        DynamicTensor::filled(node.shape.clone(), 0.0)?
                    } else {
                        let input_tangent = tangents
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                        let input_value = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        input_tangent
                            .mul(&input_value.powi(*exponent - 1)?)?
                            .scale(*exponent as f64)?
                    }
                }
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    // Constant operands are skipped, as in the symbolic rule.
                    let mut tangent = DynamicTensor::filled(node.shape.clone(), 0.0)?;
                    for (index, operand) in [*lhs, *rhs].into_iter().enumerate() {
                        if self.scalar_constant_value(operand).is_some() {
                            continue;
                        }
                        let operand_tangent = tangents
                            .get(operand)
                            .ok_or_else(|| format!("node {operand} has no evaluated tangent"))?;
                        tangent = match kind.numeric_partial(index, lhs_value, rhs_value)? {
                            Some(partial) => tangent.add(&operand_tangent.mul(&partial)?)?,
                            None => tangent.add(operand_tangent)?,
                        };
                    }
                    tangent
                }
                TensorOp::Transpose { input, axes } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .transpose(axes)?,
                TensorOp::UnaryMath { input, kind } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    match kind.numeric_chain(input_tangent, input_value, output_value)? {
                        Some(tangent) => tangent,
                        None => DynamicTensor::filled(node.shape.clone(), 0.0)?,
                    }
                }
                TensorOp::StopGradient { .. } => DynamicTensor::filled(node.shape.clone(), 0.0)?,
                TensorOp::Custom { rule, .. } => {
                    return Err(custom_rule_numeric_error(rule));
                }
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .cumsum_axis(*axis, *reverse, TensorDType::F64)?,
                TensorOp::Concat { inputs, axis } => DynamicTensor::concat(
                    &inputs
                        .iter()
                        .map(|input| {
                            tangents
                                .get(*input)
                                .ok_or_else(|| format!("node {input} has no evaluated tangent"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    *axis,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .slice_axis(*axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .pad_slice(&node.shape, *axis, *start)?,
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .gather_axis(indices, *axis)?,
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => tangents
                    .get(*base)
                    .ok_or_else(|| format!("node {base} has no evaluated tangent"))?
                    .scatter_add_axis(
                        tangents
                            .get(*updates)
                            .ok_or_else(|| format!("node {updates} has no evaluated tangent"))?,
                        indices,
                        *axis,
                        TensorDType::F64,
                        false,
                    )?,
                TensorOp::Broadcast { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .broadcast_to_shape(&node.shape)?,
            };
            tangents.push(tangent);
        }

        outputs
            .iter()
            .map(|&output| {
                let value = values
                    .get(output)
                    .cloned()
                    .ok_or_else(|| format!("output node {output} has no value"))?;
                // Derivative arithmetic runs in f64; round only each requested output tangent.
                // Bool outputs retain the weak f64 zero used by the single-output JVP.
                let tangent_dtype = match self.node(output)?.dtype {
                    TensorDType::Bool => TensorDType::F64,
                    dtype => dtype,
                };
                let tangent = tangents
                    .get(output)
                    .cloned()
                    .ok_or_else(|| format!("output node {output} has no tangent"))?
                    .into_dtype(tangent_dtype);
                Ok((value, tangent))
            })
            .collect()
    }

    pub fn hessian_scalar(
        &self,
        output: TensorNodeId,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.ensure_differentiable_input(input_name)?;
        if !self.node(output)?.shape.is_empty() {
            return Err(format!(
                "hessian_scalar requires a scalar output, got shape {:?}",
                self.node(output)?.shape
            ));
        }
        if self.nodes.iter().any(|node| {
            matches!(
                node.op,
                TensorOp::Cond { .. } | TensorOp::Fori { .. } | TensorOp::Scan { .. }
            )
        }) {
            return self.symbolic_hessian_scalar_through_regions(output, input_name, inputs);
        }
        let input_shape = self
            .nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name } if name == input_name => Some(node.shape.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("input {input_name:?} does not exist"))?;
        let input_count = element_count(&input_shape)?;
        let mut hessian = vec![vec![0.0; input_count]; input_count];

        if input_count > 1 && self.supports_finite_symbolic_second_order(inputs, 1.0)? {
            let (graph, tangent_output, cotangent_name, tangent_name) =
                self.scalar_hvp_graph(output, input_name)?;
            let mut transformed_inputs = inputs.clone();
            transformed_inputs.insert(cotangent_name, DynamicTensor::filled(vec![], 1.0)?);
            let mut finite = true;
            for column in 0..input_count {
                transformed_inputs.insert(
                    tangent_name.clone(),
                    DynamicTensor::one_hot(input_shape.clone(), column)?,
                );
                let value = graph.evaluate(tangent_output, &transformed_inputs)?;
                if value.data.iter().any(|entry| !entry.is_finite()) {
                    finite = false;
                    break;
                }
                for (row, entries) in hessian.iter_mut().enumerate() {
                    entries[column] = value.data.get(row);
                }
            }
            if finite {
                return Ok(hessian);
            }
        }

        for (row, hessian_row) in hessian.iter_mut().enumerate() {
            for (col, entry) in hessian_row.iter_mut().enumerate() {
                let first_tangents = BTreeMap::from([(
                    input_name.to_string(),
                    DynamicTensor::one_hot(input_shape.clone(), row)?,
                )]);
                let second_tangents = BTreeMap::from([(
                    input_name.to_string(),
                    DynamicTensor::one_hot(input_shape.clone(), col)?,
                )]);
                let result =
                    self.evaluate_mixed(output, inputs, &first_tangents, &second_tangents)?;
                *entry = result.mixed.data.get(0);
            }
        }

        Ok(hessian)
    }

    pub fn hvp_scalar(
        &self,
        output: TensorNodeId,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangent: DynamicTensor,
    ) -> Result<DynamicTensor, String> {
        self.ensure_differentiable_input(input_name)?;
        if !self.node(output)?.shape.is_empty() {
            return Err(format!(
                "hvp_scalar requires a scalar output, got shape {:?}",
                self.node(output)?.shape
            ));
        }
        let input_shape = self
            .nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name } if name == input_name => Some(node.shape.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("input {input_name:?} does not exist"))?;
        if input_tangent.shape != input_shape {
            return Err(format!(
                "input tangent shape {:?} does not match input {input_name:?} shape {:?}",
                input_tangent.shape, input_shape
            ));
        }

        if self.nodes.iter().any(|node| {
            matches!(
                node.op,
                TensorOp::Cond { .. } | TensorOp::Fori { .. } | TensorOp::Scan { .. }
            )
        }) {
            return self.symbolic_hvp_scalar_through_regions(
                output,
                input_name,
                inputs,
                input_tangent,
            );
        }

        let input_count = element_count(&input_shape)?;
        if input_count > 1
            && input_tangent.dtype == TensorDType::F64
            && input_tangent.data.iter().all(|value| value.is_finite())
            && self.supports_finite_symbolic_second_order(
                inputs,
                input_tangent
                    .data
                    .iter()
                    .fold(1.0_f64, |bound, value| bound.max(value.abs())),
            )?
        {
            let result = self.symbolic_hvp_scalar_through_regions(
                output,
                input_name,
                inputs,
                input_tangent.clone(),
            )?;
            if result.data.iter().all(|value| value.is_finite()) {
                return Ok(result);
            }
        }

        let second_tangents = BTreeMap::from([(input_name.to_string(), input_tangent)]);
        let mut data = Vec::with_capacity(input_count);
        for index in 0..input_count {
            let first_tangents = BTreeMap::from([(
                input_name.to_string(),
                DynamicTensor::one_hot(input_shape.clone(), index)?,
            )]);
            let result = self.evaluate_mixed(output, inputs, &first_tangents, &second_tangents)?;
            data.push(result.mixed.data.get(0));
        }
        DynamicTensor::new(input_shape, data)
    }

    fn symbolic_hessian_scalar_through_regions(
        &self,
        output: TensorNodeId,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let input_shape = self
            .input_node_id(input_name)
            .and_then(|node_id| self.node_shape(node_id))?;
        let input_count = element_count(&input_shape)?;
        let columns = (0..input_count)
            .map(|column| {
                let tangent = DynamicTensor::one_hot(input_shape.clone(), column)?;
                self.symbolic_hvp_scalar_through_regions(output, input_name, inputs, tangent)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut hessian = vec![vec![0.0; input_count]; input_count];
        for (row, hessian_row) in hessian.iter_mut().enumerate() {
            *hessian_row = columns.iter().map(|column| column.data.get(row)).collect();
        }
        Ok(hessian)
    }

    fn symbolic_hvp_scalar_through_regions(
        &self,
        output: TensorNodeId,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangent: DynamicTensor,
    ) -> Result<DynamicTensor, String> {
        let (graph, tangent_output, cotangent_name, tangent_name) =
            self.scalar_hvp_graph(output, input_name)?;
        let mut transformed_inputs = inputs.clone();
        transformed_inputs.insert(cotangent_name, DynamicTensor::filled(vec![], 1.0)?);
        transformed_inputs.insert(tangent_name, input_tangent);
        graph.evaluate(tangent_output, &transformed_inputs)
    }

    // F32 rounding and the explicit singular/discontinuous derivative conventions
    // retain the mixed-dual route. Smooth finite F64 graphs can share one
    // forward-over-reverse graph instead of evaluating every coordinate pair.
    fn supports_finite_symbolic_second_order(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        direction_bound: f64,
    ) -> Result<bool, String> {
        if self.nodes.iter().any(|node| {
            node.dtype != TensorDType::F64
                || matches!(
                    node.op,
                    TensorOp::UnaryMath { kind, .. } if kind.second_order_bounds(0.0).is_none()
                )
                || !matches!(
                    node.op,
                    TensorOp::Input { .. }
                        | TensorOp::ScalarConstant { .. }
                        | TensorOp::Constant { .. }
                        | TensorOp::Add { .. }
                        | TensorOp::Sub { .. }
                        | TensorOp::Mul { .. }
                        | TensorOp::Powi { .. }
                        | TensorOp::UnaryMath { .. }
                        | TensorOp::Matmul { .. }
                        | TensorOp::Sum { .. }
                        | TensorOp::SumAxis { .. }
                        | TensorOp::Mean { .. }
                        | TensorOp::MeanAxis { .. }
                        | TensorOp::Reshape { .. }
                        | TensorOp::Transpose { .. }
                )
        }) || inputs.values().any(|value| {
            value.dtype != TensorDType::F64 || value.data.iter().any(|entry| !entry.is_finite())
        }) {
            return Ok(false);
        }
        let values = self.evaluate_all(inputs)?;
        let mut bounds: Vec<(f64, f64, f64)> = Vec::with_capacity(self.nodes.len());
        // Absolute element bounds for primal, first and mixed derivatives also
        // guard intermediate arithmetic that a symbolic zero could eliminate.
        // All inputs are conservatively assigned a unit first-derivative bound.
        // Keep conservative headroom for the four-term mixed product sum and
        // rounding differences between the two derivative evaluation orders.
        let safe = |bound: f64| bound.is_finite() && bound < f64::MAX / 16.0;
        for (id, node) in self.nodes.iter().enumerate() {
            if values[id].data.iter().any(|value| !value.is_finite()) {
                return Ok(false);
            }
            let magnitude = values[id]
                .data
                .iter()
                .fold(0.0_f64, |bound, value| bound.max(value.abs()));
            let unary = |input: TensorNodeId, first: f64, second: f64| {
                let (_, linear, mixed) = bounds[input];
                // The mixed-dual rule forms this product before applying even
                // a zero second derivative (e.g. powi(1) or saturated tanh).
                let product = linear * linear;
                (first * linear, first * mixed + second * product, product)
            };
            let (linear, mixed, intermediate) = match &node.op {
                TensorOp::Input { .. } => (1.0, 0.0, 0.0),
                TensorOp::ScalarConstant { .. } | TensorOp::Constant { .. } => (0.0, 0.0, 0.0),
                TensorOp::Add { lhs, rhs } | TensorOp::Sub { lhs, rhs } => {
                    let (_, ll, hl) = bounds[*lhs];
                    let (_, lr, hr) = bounds[*rhs];
                    (ll + lr, hl + hr, 0.0)
                }
                TensorOp::Mul { lhs, rhs } | TensorOp::Matmul { lhs, rhs } => {
                    let (ml, ll, hl) = bounds[*lhs];
                    let (mr, lr, hr) = bounds[*rhs];
                    let factor = if matches!(node.op, TensorOp::Matmul { .. }) {
                        *self.nodes[*lhs]
                            .shape
                            .last()
                            .expect("matmul has rank at least two") as f64
                    } else {
                        1.0
                    };
                    (
                        factor * (mr * ll + ml * lr),
                        factor * (mr * hl + ml * hr + 2.0 * ll * lr),
                        0.0,
                    )
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent == 0 {
                        (0.0, 0.0, 0.0)
                    } else {
                        let magnitude = bounds[*input].0;
                        let first = *exponent as f64 * magnitude.powi((*exponent - 1) as i32);
                        let second = if *exponent == 1 {
                            0.0
                        } else {
                            *exponent as f64
                                * (*exponent - 1) as f64
                                * magnitude.powi((*exponent - 2) as i32)
                        };
                        if !safe(first) || !safe(second) {
                            return Ok(false);
                        }
                        unary(*input, first, second)
                    }
                }
                TensorOp::UnaryMath { input, kind } => match kind.second_order_bounds(magnitude) {
                    Some((first, second)) => unary(*input, first, second),
                    None => return Ok(false),
                },
                TensorOp::Sum { input } | TensorOp::Mean { input } => {
                    let (_, linear, mixed) = bounds[*input];
                    let count = values[*input].data.len() as f64;
                    (count * linear, count * mixed, 0.0)
                }
                TensorOp::SumAxis { input, axis } | TensorOp::MeanAxis { input, axis } => {
                    let (_, linear, mixed) = bounds[*input];
                    let count = self.nodes[*input].shape[*axis] as f64;
                    (count * linear, count * mixed, 0.0)
                }
                TensorOp::Reshape { input } | TensorOp::Transpose { input, .. } => {
                    let (_, linear, mixed) = bounds[*input];
                    (linear, mixed, 0.0)
                }
                _ => return Ok(false),
            };
            if !safe(magnitude)
                || !safe(linear * direction_bound)
                || !safe(mixed * direction_bound)
                || !safe(intermediate * direction_bound)
            {
                return Ok(false);
            }
            bounds.push((magnitude, linear, mixed));
        }
        Ok(true)
    }

    fn scalar_hvp_graph(
        &self,
        output: TensorNodeId,
        input_name: &str,
    ) -> Result<(TensorIr, TensorNodeId, String, String), String> {
        let cotangent_name = fresh_tensor_input_name(self, "__quabla_hvp_cotangent");
        let vjp = self.symbolic_vjp(output, &cotangent_name)?;
        let gradient = *vjp
            .gradients
            .get(input_name)
            .ok_or_else(|| format!("input {input_name:?} does not exist"))?;
        let tangent_name = fresh_tensor_input_name(&vjp.graph, "__quabla_hvp_tangent");
        let directional = vjp.graph.symbolic_jvp_with_tangent_inputs(
            gradient,
            &BTreeMap::from([(input_name.to_string(), tangent_name.clone())]),
        )?;
        Ok((
            directional.graph,
            directional.tangent,
            cotangent_name,
            tangent_name,
        ))
    }

    pub fn lower_text(&self) -> String {
        self.nodes
            .iter()
            .enumerate()
            .map(|(id, node)| match &node.op {
                TensorOp::Input { name } => {
                    format!("%{id} = input[name={name}] : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::ScalarConstant { value } => {
                    format!(
                        "%{id} = constant[value={value}] : {}",
                        format_tensor_type(&node.shape, node.dtype)
                    )
                }
                // The elements are not printed: a captured array can be large, and the text is
                // for reading the graph structure.
                TensorOp::Constant { .. } => format!(
                    "%{id} = constant[dense] : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Cast { input } => format!(
                    "%{id} = cast(%{input}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Add { lhs, rhs } => format!(
                    "%{id} = add(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ScanVjp { target, group, .. } => format!(
                    "%{id} = scan_vjp(group={group}, target={target:?}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ScanVjpJvp { target, group, .. } => format!(
                    "%{id} = scan_vjp_jvp(group={group}, target={target:?}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => format!(
                    "%{id} = cond(%{predicate}, captures={captures:?}, true_nodes={}, false_nodes={}) : {}",
                    branches.true_node_count(),
                    branches.false_node_count(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::While {
                    carry,
                    loop_plan,
                    captures,
                } => format!(
                    "%{id} = while(carry=%{carry}, captures={captures:?}, predicate_nodes={}, body_nodes={}) : {}",
                    loop_plan.predicate.plan.node_count(),
                    loop_plan.body.plan.node_count(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => format!(
                    "%{id} = fori(carry=%{carry}, lower={}, upper={}, captures={captures:?}, body_nodes={}) : {}",
                    loop_plan.lower,
                    loop_plan.upper,
                    loop_plan.body.plan.node_count(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ForiJvp {
                    carry,
                    carry_tangent,
                    loop_plan,
                    captures,
                    tangent_captures,
                } => format!(
                    "%{id} = fori_jvp(carry=%{carry}, carry_tangent=%{carry_tangent}, lower={}, upper={}, captures={captures:?}, tangent_captures={tangent_captures:?}, body_nodes={}) : {}",
                    loop_plan.lower,
                    loop_plan.upper,
                    loop_plan.body.plan.node_count(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ForiVjp { target, group, .. } => format!(
                    "%{id} = fori_vjp(group={group}, target={target:?}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ForiVjpJvp { target, group, .. } => format!(
                    "%{id} = fori_vjp_jvp(group={group}, target={target:?}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Scan {
                    scan_plan,
                    captures,
                    target,
                    group,
                    ..
                } => format!(
                    "%{id} = scan(group={group}, target={target:?}, lower={}, upper={}, captures={captures:?}, body_nodes={}) : {}",
                    scan_plan.lower,
                    scan_plan.upper,
                    scan_plan.body.plan.node_count(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Sub { lhs, rhs } => format!(
                    "%{id} = sub(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Div { lhs, rhs } => format!(
                    "%{id} = div(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Mul { lhs, rhs } => format!(
                    "%{id} = mul(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Greater { lhs, rhs } => format!(
                    "%{id} = greater(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Compare { lhs, rhs, kind } => format!(
                    "%{id} = compare(%{lhs}, %{rhs}, kind={}) : {}",
                    kind.name(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => format!(
                    "%{id} = where(%{condition}, %{on_true}, %{on_false}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Sum { input } => {
                    format!("%{id} = sum(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::SumAxis { input, axis } => format!(
                    "%{id} = sum(%{input}, axis={axis}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ExtremumAxis { input, axis, kind } => format!(
                    "%{id} = {}(%{input}, axis={axis}) : {}",
                    kind.name(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Matmul { lhs, rhs } => format!(
                    "%{id} = matmul(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::CholeskyAd { inputs, kind } => format!("%{id} = cholesky_{kind:?}({inputs:?}) : {}", format_tensor_type(&node.shape, node.dtype)),
                TensorOp::Cholesky { input } => format!("%{id} = cholesky(%{input}) : {}", format_tensor_type(&node.shape, node.dtype)),
                TensorOp::Solve { matrix, rhs } => format!(
                    "%{id} = solve(%{matrix}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Linalg { input, kind } => format!(
                    "%{id} = {}(%{input}) : {}",
                    kind.name(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Triangular { input, lower } => format!(
                    "%{id} = {}(%{input}) : {}",
                    if *lower { "tril" } else { "triu" },
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Sqrt { input } => {
                    format!("%{id} = sqrt(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::SqrtDerivative { input, order } => format!(
                    "%{id} = sqrt_derivative(%{input}, order={order}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Reshape { input } => {
                    format!("%{id} = reshape(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::Mean { input } => {
                    format!("%{id} = mean(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::MeanAxis { input, axis } => format!(
                    "%{id} = mean(%{input}, axis={axis}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Powi { input, exponent } => format!(
                    "%{id} = powi(%{input}, {exponent}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::BinaryMath { lhs, rhs, kind } => format!(
                    "%{id} = {}(%{lhs}, %{rhs}) : {}",
                    kind.name(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Transpose { input, axes } => format!(
                    "%{id} = transpose(%{input}, axes={axes:?}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::UnaryMath { input, kind } => format!(
                    "%{id} = {}(%{input}) : {}",
                    kind.name(),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::StopGradient { input } => format!(
                    "%{id} = stop_gradient(%{input}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Custom {
                    value,
                    operands,
                    rule,
                    output,
                    group,
                } => format!(
                    "%{id} = custom(%{value}) {{rule = {:?}, output = {output}, group = {group}, \
                     operands = [{}]}} : {}",
                    rule.name(),
                    operands
                        .iter()
                        .map(|operand| format!("%{operand}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => format!(
                    "%{id} = cumsum(%{input}, axis={axis}, reverse={reverse}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Concat { inputs, axis } => format!(
                    "%{id} = concat({}) axis={axis} : {}",
                    inputs
                        .iter()
                        .map(|input| format!("%{input}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => format!(
                    "%{id} = slice(%{input}, axis={axis}, start={start}, length={length}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::PadSlice { input, axis, start } => format!(
                    "%{id} = pad_slice(%{input}, axis={axis}, start={start}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => format!(
                    "%{id} = gather(%{input}, axis={axis}, indices={}) : {}",
                    format_index_list(indices),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => format!(
                    "%{id} = scatter_add(%{base}, %{updates}, axis={axis}, indices={}) : {}",
                    format_index_list(indices),
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Broadcast { input } => format!(
                    "%{id} = broadcast(%{input}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Exports a deliberately restricted, static-shape StableHLO module.
    ///
    /// This is an interoperability probe for the pure elementwise subset, not
    /// a fallback execution backend. Unsupported operations fail rather than
    /// being silently rewritten with different semantics.
    pub fn stablehlo_text(&self, output: TensorNodeId) -> Result<String, String> {
        self.node(output)?;
        let tensor_type = |node: &TensorNode| format_tensor_type(&node.shape, node.dtype);
        let inputs = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| match &node.op {
                TensorOp::Input { name } => Some((id, name, node)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let arguments = inputs
            .iter()
            .enumerate()
            .map(|(position, (_, name, node))| {
                format!("%arg{position}: {} // {name}", tensor_type(node))
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut values = BTreeMap::new();
        for (position, (id, _, _)) in inputs.iter().enumerate() {
            values.insert(*id, format!("%arg{position}"));
        }
        let mut body = Vec::new();
        for (id, node) in self.nodes.iter().enumerate() {
            let value = match &node.op {
                TensorOp::Input { .. } => continue,
                TensorOp::Add { lhs, rhs } => format!(
                    "%v{id} = stablehlo.add {}, {} : {}",
                    values[lhs],
                    values[rhs],
                    tensor_type(node)
                ),
                TensorOp::Mul { lhs, rhs } => format!(
                    "%v{id} = stablehlo.multiply {}, {} : {}",
                    values[lhs],
                    values[rhs],
                    tensor_type(node)
                ),
                TensorOp::BinaryMath { lhs, rhs, kind } if kind.stablehlo_name().is_some() => {
                    format!(
                        "%v{id} = {} {}, {} : {}",
                        kind.stablehlo_name().unwrap_or_default(),
                        values[lhs],
                        values[rhs],
                        tensor_type(node)
                    )
                }
                TensorOp::UnaryMath { input, kind } if kind.stablehlo_name().is_some() => format!(
                    "%v{id} = {} {} : {}",
                    kind.stablehlo_name().unwrap_or_default(),
                    values[input],
                    tensor_type(node)
                ),
                TensorOp::Cast { input } => format!(
                    "%v{id} = stablehlo.convert {} : ({}) -> {}",
                    values[input],
                    tensor_type(self.node(*input)?),
                    tensor_type(node)
                ),
                _ => {
                    return Err(format!(
                        "StableHLO export does not yet support {}",
                        tensor_op_name(&node.op)
                    ))
                }
            };
            values.insert(id, format!("%v{id}"));
            body.push(value);
        }
        let result = values
            .get(&output)
            .ok_or_else(|| format!("StableHLO output node {output} has no value"))?;
        body.push(format!(
            "return {result} : {}",
            tensor_type(self.node(output)?)
        ));
        Ok(format!(
            "module {{\n  func.func @main({arguments}) -> {} {{\n    {}\n  }}\n}}\n",
            tensor_type(self.node(output)?),
            body.join("\n    ")
        ))
    }

    fn binary(
        &mut self,
        name: &str,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        op: impl FnOnce(TensorNodeId, TensorNodeId) -> TensorOp,
    ) -> Result<TensorNodeId, String> {
        let shape = broadcast_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape)?;
        let [lhs, rhs] = self.coerce_operands(name, [lhs, rhs])?;
        self.push_derived(op(lhs, rhs), shape)
    }

    /// Applies the promotion rule (strict tensors, weak scalars) to operands
    /// that must share one dtype.
    ///
    /// Strong float operands of different dtypes are rejected; weak operands
    /// whose dtype differs from the strong one receive an explicit `Cast`.
    /// `Bool` operands mixed with float operands are cast to 0/1 values of
    /// the strong float dtype, or to a weak `f64` when every float operand is
    /// weak (so `mask * 2.0` still adopts a later `f32` operand). Operands
    /// that are all `Bool` are left unchanged for the op to accept or reject.
    /// Nothing is appended to the graph when the operands are rejected.
    fn coerce_operands<const N: usize>(
        &mut self,
        op_name: &str,
        mut operands: [TensorNodeId; N],
    ) -> Result<[TensorNodeId; N], String> {
        self.coerce_operand_slice(op_name, &mut operands)?;
        Ok(operands)
    }

    fn coerce_operand_slice(
        &mut self,
        op_name: &str,
        operands: &mut [TensorNodeId],
    ) -> Result<(), String> {
        let mut strong: Option<TensorDType> = None;
        let mut has_bool = false;
        let mut has_float = false;
        for operand in operands.iter() {
            let node = self.node(*operand)?;
            if node.dtype == TensorDType::Bool {
                has_bool = true;
                continue;
            }
            has_float = true;
            if node.weak {
                continue;
            }
            match strong {
                Some(dtype) if dtype != node.dtype => {
                    return Err(format!(
                        "{op_name} operands have mismatched dtypes {dtype} and {}; \
                         convert one of them explicitly with astype",
                        node.dtype
                    ))
                }
                _ => strong = Some(node.dtype),
            }
        }
        if has_bool && has_float {
            // Bool mixed with float is treated as 0/1 float; with only weak scalars the result
            // stays weak (default f64).
            let (target, weak) = strong.map_or((TensorDType::F64, true), |dtype| (dtype, false));
            for operand in operands.iter_mut() {
                let node = self.node(*operand)?;
                if node.dtype == TensorDType::Bool {
                    let shape = node.shape.clone();
                    *operand =
                        self.push_node(TensorOp::Cast { input: *operand }, shape, target, weak);
                }
            }
        }
        let Some(target) = strong else {
            // With only weak scalars, each operand keeps its dtype (all default f64).
            return Ok(());
        };
        for operand in operands.iter_mut() {
            if self.node(*operand)?.dtype != target {
                *operand = self.cast(*operand, target)?;
            }
        }
        Ok(())
    }

    fn push_node(
        &mut self,
        op: TensorOp,
        shape: Vec<usize>,
        dtype: TensorDType,
        weak: bool,
    ) -> TensorNodeId {
        let id = self.nodes.len();
        if let TensorOp::Input { name } = &op {
            Arc::make_mut(&mut self.input_nodes).insert(name.clone(), id);
        }
        Arc::make_mut(&mut self.nodes).push(TensorNode {
            op,
            shape,
            dtype,
            weak,
        });
        id
    }

    /// Appends an op whose dtype is that of its value operands (already
    /// coerced to one dtype); it is weak only when every value operand is.
    fn push_derived(&mut self, op: TensorOp, shape: Vec<usize>) -> Result<TensorNodeId, String> {
        let operands = tensor_value_operands(&op);
        let first = *operands.first().ok_or_else(|| {
            format!(
                "{} has no value operand to derive a dtype from",
                tensor_op_name(&op)
            )
        })?;
        let dtype = self.node(first)?.dtype;
        // Bool only allows data movement and select; arithmetic, reductions and math functions need
        // an explicit cast first.
        if dtype == TensorDType::Bool
            && !matches!(
                op,
                TensorOp::Where { .. }
                    | TensorOp::Reshape { .. }
                    | TensorOp::Transpose { .. }
                    | TensorOp::Concat { .. }
                    | TensorOp::Slice { .. }
                    | TensorOp::PadSlice { .. }
                    | TensorOp::Gather { .. }
                    | TensorOp::StopGradient { .. }
                    | TensorOp::Broadcast { .. }
                    | TensorOp::Custom { .. }
            )
        {
            return Err(format!(
                "{} is not defined for bool tensors; use logical_and/logical_or/logical_not \
                 (& | ~) or convert explicitly with astype",
                tensor_op_name(&op)
            ));
        }
        let mut weak = true;
        for operand in &operands {
            let node = self.node(*operand)?;
            if node.dtype != dtype {
                return Err(format!(
                    "{} operands have mismatched dtypes {dtype} and {}; \
                     convert one of them explicitly with astype",
                    tensor_op_name(&op),
                    node.dtype
                ));
            }
            weak &= node.weak;
        }
        Ok(self.push_node(op, shape, dtype, weak))
    }

    fn evaluate_all(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        Self::evaluate_tensor_nodes(&self.nodes, inputs)
    }

    fn evaluate_tensor_nodes(
        nodes: &[TensorNode],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        Self::evaluate_tensor_nodes_with_outputs(nodes, inputs, None)?
            .into_iter()
            .enumerate()
            .map(|(id, value)| value.ok_or_else(|| format!("node {id} has no evaluated value")))
            .collect()
    }

    // Retention roots cover forward outputs or the primals required by reverse rules.
    fn evaluate_tensor_nodes_with_outputs(
        nodes: &[TensorNode],
        inputs: &BTreeMap<String, DynamicTensor>,
        outputs: Option<&[TensorNodeId]>,
    ) -> Result<Vec<Option<DynamicTensor>>, String> {
        let mut values: Vec<Option<DynamicTensor>> = Vec::with_capacity(nodes.len());
        let mut last_uses = outputs.map(|outputs| tensor_forward_last_uses(nodes, outputs));
        let mut fori_vjp_cache: HashMap<usize, TensorForiVjpEvaluation> = HashMap::new();
        let mut fori_vjp_jvp_cache: HashMap<usize, TensorForiVjpJvpEvaluation> = HashMap::new();
        let mut scan_cache: HashMap<usize, TensorScanEvaluation> = HashMap::new();
        let mut scan_vjp_cache: HashMap<usize, TensorScanVjpEvaluation> = HashMap::new();
        let mut scan_vjp_jvp_cache: HashMap<usize, TensorScanVjpJvpEvaluation> = HashMap::new();

        for (node_id, node) in nodes.iter().enumerate() {
            let reused = match reuse_forward_unary(node, &mut values, last_uses.as_ref())? {
                Some(value) => Some(value),
                None => reuse_forward_binary(node, &mut values, last_uses.as_ref())?,
            };
            let value = match &node.op {
                _ if reused.is_some() => reused.expect("reused value was checked above"),
                TensorOp::Input { name } => {
                    let input = inputs
                        .get(name)
                        .ok_or_else(|| format!("missing input {name:?}"))?;
                    if input.shape != node.shape {
                        return Err(format!(
                            "input {name:?} has shape {:?}, expected {:?}",
                            input.shape, node.shape
                        ));
                    }
                    input.clone()
                }
                TensorOp::ScalarConstant { value } => DynamicTensor::filled(vec![], *value)?,
                TensorOp::Constant { value } => value.value().clone(),
                TensorOp::Cast { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .clone(),
                TensorOp::Add { lhs, rhs } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .add(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Sub { lhs, rhs } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .sub(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Div { lhs, rhs } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .div(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Mul { lhs, rhs } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .mul(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Greater { lhs, rhs } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .greater(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Compare { lhs, rhs, kind } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .compare(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                        *kind,
                    )?,
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => values
                    .get(*condition)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {condition} has no evaluated value"))?
                    .where_select(
                        values
                            .get(*on_true)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {on_true} has no evaluated value"))?,
                        values
                            .get(*on_false)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {on_false} has no evaluated value"))?,
                    )?,
                TensorOp::Cond {
                    predicate,
                    branches,
                    captures,
                } => {
                    let predicate = tensor_scalar_predicate(
                        values
                            .get(*predicate)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {predicate} has no evaluated value"))?,
                    )?;
                    let branch_inputs = tensor_forward_capture_values(captures, &values)?;
                    branches.evaluate(predicate, &branch_inputs)?
                }
                TensorOp::While {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let external_inputs = tensor_forward_capture_values(captures, &values)?;
                    loop_plan.evaluate(
                        values
                            .get(*carry)
                            .and_then(Option::as_ref)
                            .cloned()
                            .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                        &external_inputs,
                    )?
                }
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let external_inputs = tensor_forward_capture_values(captures, &values)?;
                    loop_plan.evaluate(
                        values
                            .get(*carry)
                            .and_then(Option::as_ref)
                            .cloned()
                            .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                        &external_inputs,
                    )?
                }
                TensorOp::ForiJvp {
                    carry,
                    carry_tangent,
                    loop_plan,
                    captures,
                    tangent_captures,
                } => {
                    let external_inputs = tensor_forward_capture_values(captures, &values)?;
                    let external_tangents =
                        tensor_forward_capture_values(tangent_captures, &values)?;
                    loop_plan
                        .jvp(
                            values
                                .get(*carry)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            values
                                .get(*carry_tangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("node {carry_tangent} has no evaluated value")
                                })?,
                            &external_inputs,
                            &external_tangents,
                        )?
                        .1
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
                        let external_inputs = tensor_forward_capture_values(captures, &values)?;
                        let (_, carry_gradient, external_gradients) = loop_plan.value_and_vjp(
                            values
                                .get(*carry)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            &external_inputs,
                            values
                                .get(*output_cotangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("node {output_cotangent} has no evaluated value")
                                })?,
                        )?;
                        fori_vjp_cache.insert(
                            *group,
                            TensorForiVjpEvaluation {
                                carry_gradient,
                                external_gradients,
                            },
                        );
                    }
                    let cached = fori_vjp_cache.get(group).ok_or_else(|| {
                        format!("fori VJP group {group} was not cached after evaluation")
                    })?;
                    match target {
                        TensorForiVjpTarget::Carry => cached.carry_gradient.clone(),
                        TensorForiVjpTarget::External(name) => cached
                            .external_gradients
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("fori VJP has no gradient for {name:?}"))?,
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
                        let external_inputs = tensor_forward_capture_values(captures, &values)?;
                        let external_tangents =
                            tensor_forward_capture_values(tangent_captures, &values)?;
                        let gradients = plan.jvp(
                            values
                                .get(*carry)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            values
                                .get(*carry_tangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("node {carry_tangent} has no evaluated value")
                                })?,
                            &external_inputs,
                            &external_tangents,
                            values
                                .get(*output_cotangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("node {output_cotangent} has no evaluated value")
                                })?,
                            values
                                .get(*output_cotangent_tangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!(
                                        "node {output_cotangent_tangent} has no evaluated value"
                                    )
                                })?,
                        )?;
                        fori_vjp_jvp_cache.insert(*group, TensorForiVjpJvpEvaluation { gradients });
                    }
                    let cached = fori_vjp_jvp_cache.get(group).ok_or_else(|| {
                        format!("fori VJP JVP group {group} was not cached after evaluation")
                    })?;
                    let name = match target {
                        TensorForiVjpTarget::Carry => &plan.loop_plan.carry_name,
                        TensorForiVjpTarget::External(name) => name,
                    };
                    cached
                        .gradients
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("fori VJP JVP has no gradient for {name:?}"))?
                }
                TensorOp::Scan {
                    carry,
                    scan_plan,
                    captures,
                    target,
                    group,
                } => {
                    if !scan_cache.contains_key(group) {
                        let external_inputs = tensor_forward_capture_values(captures, &values)?;
                        let (carry, outputs) = scan_plan.evaluate(
                            values
                                .get(*carry)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            &external_inputs,
                        )?;
                        scan_cache.insert(*group, TensorScanEvaluation { carry, outputs });
                    }
                    let cached = scan_cache.get(group).ok_or_else(|| {
                        format!("scan group {group} was not cached after evaluation")
                    })?;
                    match target {
                        TensorScanTarget::Carry => cached.carry.clone(),
                        TensorScanTarget::Outputs => cached.outputs.clone(),
                    }
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
                        let external_inputs = tensor_forward_capture_values(captures, &values)?;
                        let (carry_gradient, external_gradients) = scan_plan.vjp(
                            values
                                .get(*carry)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            &external_inputs,
                            values
                                .get(*final_carry_cotangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("node {final_carry_cotangent} has no evaluated value")
                                })?,
                            values
                                .get(*output_cotangent)
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("node {output_cotangent} has no evaluated value")
                                })?,
                        )?;
                        scan_vjp_cache.insert(
                            *group,
                            TensorScanVjpEvaluation {
                                carry_gradient,
                                external_gradients,
                            },
                        );
                    }
                    let cached = scan_vjp_cache.get(group).ok_or_else(|| {
                        format!("scan VJP group {group} was not cached after evaluation")
                    })?;
                    match target {
                        TensorScanVjpTarget::Carry => cached.carry_gradient.clone(),
                        TensorScanVjpTarget::External(name) => cached
                            .external_gradients
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("scan VJP has no gradient for {name:?}"))?,
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
                        let external_inputs = tensor_forward_capture_values(captures, &values)?;
                        let external_tangents =
                            tensor_forward_capture_values(tangent_captures, &values)?;
                        let gradients = plan.jvp(
                                values.get(*carry).and_then(Option::as_ref).cloned().ok_or_else(|| {
                                    format!("node {carry} has no evaluated value")
                                })?,
                                values.get(*carry_tangent).and_then(Option::as_ref).cloned().ok_or_else(|| {
                                    format!("node {carry_tangent} has no evaluated value")
                                })?,
                                &external_inputs,
                                &external_tangents,
                                values.get(*final_carry_cotangent).and_then(Option::as_ref).cloned().ok_or_else(|| {
                                    format!(
                                        "node {final_carry_cotangent} has no evaluated value"
                                    )
                                })?,
                                values
                                    .get(*final_carry_cotangent_tangent).and_then(Option::as_ref)
                                    .cloned()
                                    .ok_or_else(|| {
                                        format!(
                                            "node {final_carry_cotangent_tangent} has no evaluated value"
                                        )
                                    })?,
                                values.get(*output_cotangent).and_then(Option::as_ref).cloned().ok_or_else(|| {
                                    format!("node {output_cotangent} has no evaluated value")
                                })?,
                                values
                                    .get(*output_cotangent_tangent).and_then(Option::as_ref)
                                    .cloned()
                                    .ok_or_else(|| {
                                        format!(
                                            "node {output_cotangent_tangent} has no evaluated value"
                                        )
                                    })?,
                            )?;
                        scan_vjp_jvp_cache.insert(*group, TensorScanVjpJvpEvaluation { gradients });
                    }
                    let cached = scan_vjp_jvp_cache.get(group).ok_or_else(|| {
                        format!("Scan VJP JVP group {group} was not cached after evaluation")
                    })?;
                    let name = match target {
                        TensorScanVjpTarget::Carry => &plan.scan_plan.carry_name,
                        TensorScanVjpTarget::External(name) => name,
                    };
                    cached
                        .gradients
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("Scan VJP JVP has no gradient for {name:?}"))?
                }
                TensorOp::Sum { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sum_all()?,
                TensorOp::SumAxis { input, axis } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_axis(*axis, 1.0)?,
                TensorOp::ExtremumAxis { input, axis, kind } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_extremum_axis(*axis, *kind)?,
                TensorOp::Matmul { lhs, rhs } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .matmul(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::CholeskyAd { inputs, kind } => {
                    let operands = inputs
                        .iter()
                        .map(|id| {
                            values[*id]
                                .as_ref()
                                .ok_or_else(|| format!("node {id} has no value"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    cholesky::evaluate_symbolic(*kind, &operands)?
                }
                TensorOp::Cholesky { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .cholesky()?,
                TensorOp::Solve { matrix, rhs } => values
                    .get(*matrix)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {matrix} has no evaluated value"))?
                    .solve(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Triangular { input, lower } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .triangular(*lower)?,
                TensorOp::Linalg { input, kind } => linalg::evaluate(
                    *kind,
                    values
                        .get(*input)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?,
                )?,
                TensorOp::Sqrt { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sqrt()?,
                TensorOp::SqrtDerivative { input, order } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sqrt_derivative(*order)?,
                TensorOp::Reshape { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reshape(node.shape.clone())?,
                TensorOp::Mean { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .mean_all()?,
                TensorOp::MeanAxis { input, axis } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_axis(*axis, 1.0 / nodes[*input].shape[*axis] as f64)?,
                TensorOp::Powi { input, exponent } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .powi(*exponent)?,
                TensorOp::BinaryMath { lhs, rhs, kind } => values
                    .get(*lhs)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .elementwise(
                        values
                            .get(*rhs)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                        kind.function(),
                    )?,
                TensorOp::Transpose { input, axes } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .transpose(axes)?,
                TensorOp::UnaryMath { input, kind } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .map_f64(|x| kind.evaluate(x))?,
                TensorOp::StopGradient { input } | TensorOp::Custom { value: input, .. } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .clone(),
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .cumsum_axis(*axis, *reverse, node.dtype)?,
                TensorOp::Concat { inputs, axis } => DynamicTensor::concat(
                    &inputs
                        .iter()
                        .map(|input| {
                            values
                                .get(*input)
                                .and_then(Option::as_ref)
                                .ok_or_else(|| format!("node {input} has no evaluated value"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    *axis,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .slice_axis(*axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .pad_slice(&node.shape, *axis, *start)?,
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .gather_axis(indices, *axis)?,
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => values
                    .get(*base)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {base} has no evaluated value"))?
                    .scatter_add_axis(
                        values
                            .get(*updates)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| format!("node {updates} has no evaluated value"))?,
                        indices,
                        *axis,
                        node.dtype,
                        false,
                    )?,
                TensorOp::Broadcast { input } => values
                    .get(*input)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .broadcast_to_shape(&node.shape)?,
            };
            // Each node computes in f64 and rounds to the node dtype: inputs are rounded
            // automatically, casts convert, and f32 nodes get the correctly rounded f64 result
            // (`+ - * / sqrt` are bit-identical to IEEE f32).
            values.push(Some(value.into_dtype(node.dtype)));
            #[cfg(test)]
            if last_uses.is_some() {
                let elements = values.iter().flatten().map(|value| value.data.len()).sum();
                CPU_FORWARD_PEAK_ELEMENTS.with(|peak| peak.set(peak.get().max(elements)));
            }
            if let Some(last_uses) = &mut last_uses {
                for input in tensor_op_inputs(&node.op) {
                    last_uses.remaining[input] -= 1;
                    if last_uses.remaining[input] == 0 && !last_uses.retained[input] {
                        values[input] = None;
                    }
                }
                if last_uses.remaining[node_id] == 0 && !last_uses.retained[node_id] {
                    values[node_id] = None;
                }
                // Region siblings share one evaluation, but its cached tensors need not outlive
                // the final sibling. Node values above protect any requested output separately.
                if let Some(group) = tensor_op_group(&node.op) {
                    if last_uses.group_last_nodes.get(&group) == Some(&node_id) {
                        fori_vjp_cache.remove(&group);
                        fori_vjp_jvp_cache.remove(&group);
                        scan_cache.remove(&group);
                        scan_vjp_cache.remove(&group);
                        scan_vjp_jvp_cache.remove(&group);
                    }
                }
            }
        }
        Ok(values)
    }

    fn evaluate_mixed(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        first_tangents: &BTreeMap<String, DynamicTensor>,
        second_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<MixedTangent, String> {
        self.evaluate_mixed_with_input_mixed(
            output,
            inputs,
            first_tangents,
            second_tangents,
            &BTreeMap::new(),
        )
    }

    fn evaluate_mixed_with_input_mixed(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        first_tangents: &BTreeMap<String, DynamicTensor>,
        second_tangents: &BTreeMap<String, DynamicTensor>,
        input_mixed: &BTreeMap<String, DynamicTensor>,
    ) -> Result<MixedTangent, String> {
        self.node(output)?;
        if self
            .nodes
            .iter()
            .any(|node| matches!(node.op, TensorOp::CholeskyAd { .. }))
        {
            let (expanded, mapped) = self.expand_all_cholesky()?;
            return expanded.evaluate_mixed_with_input_mixed(
                mapped[output],
                inputs,
                first_tangents,
                second_tangents,
                input_mixed,
            );
        }
        if let Some((expanded, mapped)) = self.expand_cholesky_for_ad()? {
            return expanded.evaluate_mixed_with_input_mixed(
                mapped[output],
                inputs,
                first_tangents,
                second_tangents,
                input_mixed,
            );
        }
        let mut values: Vec<MixedTangent> = Vec::with_capacity(self.nodes.len());

        for node in self.nodes.iter() {
            let value = match &node.op {
                TensorOp::Input { name } => {
                    let value = input_value(inputs, name, &node.shape)?;
                    MixedTangent {
                        first: input_tangent_or_zero(first_tangents, name, &node.shape)?,
                        second: input_tangent_or_zero(second_tangents, name, &node.shape)?,
                        mixed: input_tangent_or_zero(input_mixed, name, &node.shape)?,
                        value,
                    }
                }
                TensorOp::UnaryMath { input, kind } => kind.numeric_mixed(
                    values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?,
                    &node.shape,
                )?,
                TensorOp::Custom { rule, .. } => {
                    return Err(custom_rule_numeric_error(rule));
                }
                TensorOp::StopGradient { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let zero = DynamicTensor::filled(node.shape.clone(), 0.0)?;
                    MixedTangent {
                        value: input.value.clone(),
                        first: zero.clone(),
                        second: zero.clone(),
                        mixed: zero,
                    }
                }
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let scan = |tensor: &DynamicTensor, rounding| {
                        tensor.cumsum_axis(*axis, *reverse, rounding)
                    };
                    MixedTangent {
                        value: scan(&input.value, node.dtype)?,
                        first: scan(&input.first, TensorDType::F64)?,
                        second: scan(&input.second, TensorDType::F64)?,
                        mixed: scan(&input.mixed, TensorDType::F64)?,
                    }
                }
                TensorOp::Reshape { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.reshape(node.shape.clone())?,
                        first: input.first.reshape(node.shape.clone())?,
                        second: input.second.reshape(node.shape.clone())?,
                        mixed: input.mixed.reshape(node.shape.clone())?,
                    }
                }
                TensorOp::Mean { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.mean_all()?,
                        first: input.first.mean_all()?,
                        second: input.second.mean_all()?,
                        mixed: input.mixed.mean_all()?,
                    }
                }
                TensorOp::MeanAxis { input, axis } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let scale = 1.0 / input.value.shape[*axis] as f64;
                    MixedTangent {
                        value: input.value.reduce_axis(*axis, scale)?,
                        first: input.first.reduce_axis(*axis, scale)?,
                        second: input.second.reduce_axis(*axis, scale)?,
                        mixed: input.mixed.reduce_axis(*axis, scale)?,
                    }
                }
                TensorOp::Powi { input, exponent } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let value = input.value.powi(*exponent)?;
                    if *exponent == 0 {
                        MixedTangent {
                            value,
                            first: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                            second: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                            mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                        }
                    } else {
                        let first_derivative =
                            input.value.powi(*exponent - 1)?.scale(*exponent as f64)?;
                        let second_derivative = if *exponent < 2 {
                            DynamicTensor::filled(node.shape.clone(), 0.0)?
                        } else {
                            input
                                .value
                                .powi(*exponent - 2)?
                                .scale((*exponent as f64) * ((*exponent - 1) as f64))?
                        };
                        MixedTangent {
                            first: input.first.mul(&first_derivative)?,
                            second: input.second.mul(&first_derivative)?,
                            mixed: input
                                .mixed
                                .mul(&first_derivative)?
                                .add(&input.first.mul(&input.second)?.mul(&second_derivative)?)?,
                            value,
                        }
                    }
                }
                TensorOp::Transpose { input, axes } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.transpose(axes)?,
                        first: input.first.transpose(axes)?,
                        second: input.second.transpose(axes)?,
                        mixed: input.mixed.transpose(axes)?,
                    }
                }
                TensorOp::Sqrt { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let first_derivative = input.value.sqrt_derivative(1)?;
                    let second_derivative = input.value.sqrt_derivative(2)?;
                    MixedTangent {
                        value: input.value.sqrt()?,
                        first: input.first.mul(&first_derivative)?,
                        second: input.second.mul(&first_derivative)?,
                        mixed: input
                            .mixed
                            .mul(&first_derivative)?
                            .add(&input.first.mul(&input.second)?.mul(&second_derivative)?)?,
                    }
                }
                TensorOp::SqrtDerivative { input, order } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let first_order = order
                        .checked_add(1)
                        .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                    let second_order = order
                        .checked_add(2)
                        .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                    let first_derivative = input.value.sqrt_derivative(first_order)?;
                    let second_derivative = input.value.sqrt_derivative(second_order)?;
                    MixedTangent {
                        value: input.value.sqrt_derivative(*order)?,
                        first: input.first.mul(&first_derivative)?,
                        second: input.second.mul(&first_derivative)?,
                        mixed: input
                            .mixed
                            .mul(&first_derivative)?
                            .add(&input.first.mul(&input.second)?.mul(&second_derivative)?)?,
                    }
                }
                TensorOp::ScalarConstant { value } => MixedTangent {
                    value: DynamicTensor::filled(vec![], *value)?,
                    first: DynamicTensor::filled(vec![], 0.0)?,
                    second: DynamicTensor::filled(vec![], 0.0)?,
                    mixed: DynamicTensor::filled(vec![], 0.0)?,
                },
                TensorOp::Constant { value } => MixedTangent {
                    value: value.value().clone(),
                    first: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                    second: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                    mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                },
                TensorOp::Cast { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.astype(node.dtype),
                        first: input.first.astype(node.dtype),
                        second: input.second.astype(node.dtype),
                        mixed: input.mixed.astype(node.dtype),
                    }
                }
                TensorOp::Add { lhs, rhs } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    MixedTangent {
                        value: lhs.value.add(&rhs.value)?,
                        first: lhs.first.add(&rhs.first)?,
                        second: lhs.second.add(&rhs.second)?,
                        mixed: lhs.mixed.add(&rhs.mixed)?,
                    }
                }
                TensorOp::Sub { lhs, rhs } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    MixedTangent {
                        value: lhs.value.sub(&rhs.value)?,
                        first: lhs.first.sub(&rhs.first)?,
                        second: lhs.second.sub(&rhs.second)?,
                        mixed: lhs.mixed.sub(&rhs.mixed)?,
                    }
                }
                TensorOp::Div { lhs, rhs } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    // Differentiating q r = l twice gives every tangent as a
                    // residual divided by r, with no powers of r.
                    let quotient = lhs.value.div(&rhs.value)?;
                    let first = lhs
                        .first
                        .sub(&quotient.mul(&rhs.first)?)?
                        .div(&rhs.value)?;
                    let second = lhs
                        .second
                        .sub(&quotient.mul(&rhs.second)?)?
                        .div(&rhs.value)?;
                    let mixed = lhs
                        .mixed
                        .sub(&first.mul(&rhs.second)?)?
                        .sub(&second.mul(&rhs.first)?)?
                        .sub(&quotient.mul(&rhs.mixed)?)?
                        .div(&rhs.value)?;
                    MixedTangent {
                        value: quotient,
                        first,
                        second,
                        mixed,
                    }
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    MixedTangent {
                        value: lhs.value.mul(&rhs.value)?,
                        first: lhs
                            .first
                            .mul(&rhs.value)?
                            .add(&lhs.value.mul(&rhs.first)?)?,
                        second: lhs
                            .second
                            .mul(&rhs.value)?
                            .add(&lhs.value.mul(&rhs.second)?)?,
                        mixed: lhs
                            .mixed
                            .mul(&rhs.value)?
                            .add(&lhs.first.mul(&rhs.second)?)?
                            .add(&lhs.second.mul(&rhs.first)?)?
                            .add(&lhs.value.mul(&rhs.mixed)?)?,
                    }
                }
                // Second-order forward rule matching the composed symbolic rules:
                // the partials of the non-constant operands, see `numeric_mixed`.
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    let operand = |id: TensorNodeId| {
                        values
                            .get(id)
                            .ok_or_else(|| format!("node {id} has no evaluated value"))
                    };
                    kind.numeric_mixed(
                        [operand(*lhs)?, operand(*rhs)?],
                        [*lhs, *rhs].map(|id| self.scalar_constant_value(id).is_none()),
                        &node.shape,
                    )?
                }
                TensorOp::Greater { lhs, rhs } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    MixedTangent {
                        value: lhs.value.greater(&rhs.value)?,
                        first: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                        second: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                        mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                    }
                }
                TensorOp::Compare { lhs, rhs, kind } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    MixedTangent {
                        value: lhs.value.compare(&rhs.value, *kind)?,
                        first: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                        second: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                        mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                    }
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let condition = values
                        .get(*condition)
                        .ok_or_else(|| format!("node {condition} has no evaluated value"))?;
                    let on_true = values
                        .get(*on_true)
                        .ok_or_else(|| format!("node {on_true} has no evaluated value"))?;
                    let on_false = values
                        .get(*on_false)
                        .ok_or_else(|| format!("node {on_false} has no evaluated value"))?;
                    MixedTangent {
                        value: condition
                            .value
                            .where_select(&on_true.value, &on_false.value)?,
                        first: condition
                            .value
                            .where_select(&on_true.first, &on_false.first)?,
                        second: condition
                            .value
                            .where_select(&on_true.second, &on_false.second)?,
                        mixed: condition
                            .value
                            .where_select(&on_true.mixed, &on_false.mixed)?,
                    }
                }
                TensorOp::While { .. } => return Err(WHILE_LOOP_REVERSE_MODE_ERROR.to_string()),
                TensorOp::Cond { .. } => return Err(
                    "mixed second-order differentiation through Cond regions is not implemented"
                        .to_string(),
                ),
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let initial_carry = values
                        .get(*carry)
                        .cloned()
                        .ok_or_else(|| format!("node {carry} has no evaluated mixed value"))?;
                    let mut external_values = BTreeMap::new();
                    let mut external_first = BTreeMap::new();
                    let mut external_second = BTreeMap::new();
                    let mut external_mixed = BTreeMap::new();
                    for (name, capture) in captures {
                        let capture = values.get(*capture).ok_or_else(|| {
                            format!("node {capture} has no evaluated mixed value")
                        })?;
                        external_values.insert(name.clone(), capture.value.clone());
                        external_first.insert(name.clone(), capture.first.clone());
                        external_second.insert(name.clone(), capture.second.clone());
                        external_mixed.insert(name.clone(), capture.mixed.clone());
                    }
                    let mut carry = initial_carry;
                    for index in loop_plan.lower..loop_plan.upper {
                        let inputs = loop_plan.body_inputs(
                            carry.value.clone(),
                            index,
                            &external_values,
                        )?;
                        let first = loop_plan.body_tangents(
                            carry.first.clone(),
                            &external_first,
                        )?;
                        let second = loop_plan.body_tangents(
                            carry.second.clone(),
                            &external_second,
                        )?;
                        let mixed = loop_plan.body_tangents(carry.mixed.clone(), &external_mixed)?;
                        carry = loop_plan
                            .body
                            .plan
                            .as_ir()
                            .evaluate_mixed_with_input_mixed(
                                loop_plan.body.plan.output_node_id,
                                &inputs,
                                &first,
                                &second,
                                &mixed,
                            )?;
                    }
                    carry
                }
                TensorOp::ForiVjp { .. } => return Err(
                    "mixed second-order differentiation through Fori VJP results is not implemented"
                        .to_string(),
                ),
                TensorOp::ForiJvp { .. } => return Err(
                    "mixed differentiation through Fori JVP results is not implemented".to_string(),
                ),
                TensorOp::ForiVjpJvp { .. } => return Err(
                    "mixed differentiation through Fori VJP JVP results is not implemented"
                        .to_string(),
                ),
                TensorOp::Scan { .. } => return Err(
                    "mixed second-order differentiation through Scan regions is not implemented"
                        .to_string(),
                ),
                TensorOp::ScanVjp { .. } => return Err(
                    "mixed second-order differentiation through Scan VJP results is not implemented"
                        .to_string(),
                ),
                TensorOp::ScanVjpJvp { .. } => return Err(
                    "mixed differentiation through Scan VJP JVP results is not implemented"
                        .to_string(),
                ),
                TensorOp::Sum { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.sum_all()?,
                        first: input.first.sum_all()?,
                        second: input.second.sum_all()?,
                        mixed: input.mixed.sum_all()?,
                    }
                }
                TensorOp::SumAxis { input, axis } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.reduce_axis(*axis, 1.0)?,
                        first: input.first.reduce_axis(*axis, 1.0)?,
                        second: input.second.reduce_axis(*axis, 1.0)?,
                        mixed: input.mixed.reduce_axis(*axis, 1.0)?,
                    }
                }
                // Piecewise linear: every derivative, including the mixed
                // second-order term, is the chooser average of the input's.
                TensorOp::ExtremumAxis { input, axis, kind } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let value = input.value.reduce_extremum_axis(*axis, *kind)?;
                    let choose = |tangent: &DynamicTensor| {
                        input.value.extremum_axis_tangent(&value, tangent, *axis)
                    };
                    MixedTangent {
                        first: choose(&input.first)?,
                        second: choose(&input.second)?,
                        mixed: choose(&input.mixed)?,
                        value,
                    }
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    MixedTangent {
                        value: lhs.value.matmul(&rhs.value)?,
                        first: lhs
                            .first
                            .matmul(&rhs.value)?
                            .add(&lhs.value.matmul(&rhs.first)?)?,
                        second: lhs
                            .second
                            .matmul(&rhs.value)?
                            .add(&lhs.value.matmul(&rhs.second)?)?,
                        mixed: lhs
                            .mixed
                            .matmul(&rhs.value)?
                            .add(&lhs.first.matmul(&rhs.second)?)?
                            .add(&lhs.second.matmul(&rhs.first)?)?
                            .add(&lhs.value.matmul(&rhs.mixed)?)?,
                    }
                }
                TensorOp::Cholesky { input } => {
                    let input = &values[*input];
                    MixedTangent { value: input.value.cholesky()?,
                        first: cholesky::evaluate(CholeskyAdKind::Jvp, &[&input.value, &input.first])?,
                        second: cholesky::evaluate(CholeskyAdKind::Jvp, &[&input.value, &input.second])?,
                        mixed: cholesky::evaluate(CholeskyAdKind::Mixed, &[&input.value, &input.first, &input.second, &input.mixed])? }
                }
                TensorOp::CholeskyAd { .. } => unreachable!("Cholesky AD expanded before mixed evaluation"),
                TensorOp::Solve { matrix, rhs } => {
                    let matrix = values
                        .get(*matrix)
                        .ok_or_else(|| format!("node {matrix} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    if let Some(result) = matrix.solve_reusing_finite_factor(rhs)? {
                        result
                    } else {
                        let value = matrix.value.solve(&rhs.value)?;
                        let first = matrix
                            .value
                            .solve(&rhs.first.sub(&matrix.first.matmul(&value)?)?)?;
                        let second = matrix
                            .value
                            .solve(&rhs.second.sub(&matrix.second.matmul(&value)?)?)?;
                        let mixed_rhs = rhs
                            .mixed
                            .sub(&matrix.mixed.matmul(&value)?)?
                            .sub(&matrix.first.matmul(&second)?)?
                            .sub(&matrix.second.matmul(&first)?)?;
                        MixedTangent {
                            value,
                            first,
                            second,
                            mixed: matrix.value.solve(&mixed_rhs)?,
                        }
                    }
                }
                TensorOp::Linalg { input, kind } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let jvp = |direction: &DynamicTensor| {
                        evaluate_linalg_derivative(
                            *kind,
                            LinalgDerivative::Jvp,
                            &[&input.value, direction],
                        )
                    };
                    MixedTangent {
                        value: linalg::evaluate(*kind, &input.value)?,
                        first: jvp(&input.first)?,
                        second: jvp(&input.second)?,
                        mixed: evaluate_linalg_derivative(
                            *kind,
                            LinalgDerivative::Mixed,
                            &[&input.value, &input.first, &input.second, &input.mixed],
                        )?,
                    }
                }
                TensorOp::Triangular { input, lower } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.triangular(*lower)?,
                        first: input.first.triangular(*lower)?,
                        second: input.second.triangular(*lower)?,
                        mixed: input.mixed.triangular(*lower)?,
                    }
                }
                TensorOp::Concat { inputs, axis } => {
                    let parts = |select: fn(&MixedTangent) -> &DynamicTensor| {
                        inputs
                            .iter()
                            .map(|input| {
                                values
                                    .get(*input)
                                    .map(select)
                                    .ok_or_else(|| format!("node {input} has no evaluated value"))
                            })
                            .collect::<Result<Vec<_>, String>>()
                    };
                    MixedTangent {
                        value: DynamicTensor::concat(&parts(|value| &value.value)?, *axis)?,
                        first: DynamicTensor::concat(&parts(|value| &value.first)?, *axis)?,
                        second: DynamicTensor::concat(&parts(|value| &value.second)?, *axis)?,
                        mixed: DynamicTensor::concat(&parts(|value| &value.mixed)?, *axis)?,
                    }
                }
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.slice_axis(*axis, *start, *length)?,
                        first: input.first.slice_axis(*axis, *start, *length)?,
                        second: input.second.slice_axis(*axis, *start, *length)?,
                        mixed: input.mixed.slice_axis(*axis, *start, *length)?,
                    }
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.pad_slice(&node.shape, *axis, *start)?,
                        first: input.first.pad_slice(&node.shape, *axis, *start)?,
                        second: input.second.pad_slice(&node.shape, *axis, *start)?,
                        mixed: input.mixed.pad_slice(&node.shape, *axis, *start)?,
                    }
                }
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.gather_axis(indices, *axis)?,
                        first: input.first.gather_axis(indices, *axis)?,
                        second: input.second.gather_axis(indices, *axis)?,
                        mixed: input.mixed.gather_axis(indices, *axis)?,
                    }
                }
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => {
                    let base = values
                        .get(*base)
                        .ok_or_else(|| format!("node {base} has no evaluated value"))?;
                    let updates = values
                        .get(*updates)
                        .ok_or_else(|| format!("node {updates} has no evaluated value"))?;
                    // The primal rounds every sum to the node dtype; derivative
                    // components stay f64, as for a chain of adds.
                    let scatter = |base: &DynamicTensor, updates: &DynamicTensor, rounding| {
                        base.scatter_add_axis(updates, indices, *axis, rounding, false)
                    };
                    MixedTangent {
                        value: scatter(&base.value, &updates.value, node.dtype)?,
                        first: scatter(&base.first, &updates.first, TensorDType::F64)?,
                        second: scatter(&base.second, &updates.second, TensorDType::F64)?,
                        mixed: scatter(&base.mixed, &updates.mixed, TensorDType::F64)?,
                    }
                }
                TensorOp::Broadcast { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    MixedTangent {
                        value: input.value.broadcast_to_shape(&node.shape)?,
                        first: input.first.broadcast_to_shape(&node.shape)?,
                        second: input.second.broadcast_to_shape(&node.shape)?,
                        mixed: input.mixed.broadcast_to_shape(&node.shape)?,
                    }
                }
            };
            // Primal values are rounded to the node dtype (matching evaluate); derivative
            // components stay f64.
            values.push(MixedTangent {
                value: value.value.into_dtype(node.dtype),
                ..value
            });
        }

        values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))
    }

    fn node(&self, id: TensorNodeId) -> Result<&TensorNode, String> {
        self.nodes
            .get(id)
            .ok_or_else(|| format!("node {id} does not exist"))
    }
}

/// The product of an eager tensor along `axis`, evaluated through the graph
/// of [`TensorIr::prod_axis`], so eager arrays and traced CPU programs round
/// the product identically.
pub fn evaluate_prod_axis(input: &DynamicTensor, axis: isize) -> Result<DynamicTensor, String> {
    let mut graph = TensorIr::new();
    let argument = graph.input_typed("input", input.shape.clone(), input.dtype)?;
    let output = graph.prod_axis(argument, axis)?;
    graph.evaluate(
        output,
        &BTreeMap::from([("input".to_string(), input.clone())]),
    )
}

/// A derivative of a `Linalg` node for the numeric evaluators.
#[derive(Clone, Copy)]
enum LinalgDerivative {
    /// Operands `[matrix, direction]`.
    Jvp,
    /// Operands `[matrix, cotangent]`.
    Vjp,
    /// Operands `[matrix, first, second, mixed]`: the second-order term of
    /// a jet, `D^2 f[first, second] + D f[mixed]`.
    Mixed,
}

/// Evaluates a derivative of `linalg(matrix, kind)` by building the symbolic
/// rule of [`TensorIr::linalg_jvp`] or [`TensorIr::linalg_vjp`] on a small
/// reference graph, so the numeric and symbolic transforms share one rule.
fn evaluate_linalg_derivative(
    kind: LinalgKind,
    derivative: LinalgDerivative,
    operands: &[&DynamicTensor],
) -> Result<DynamicTensor, String> {
    let matrix = operands[0];
    let dtype = matrix.dtype;
    let mut reference = TensorIr::new();
    let argument = reference.input_typed("matrix", matrix.shape.clone(), dtype)?;
    let output = reference.linalg(argument, kind)?;
    let mut inputs = BTreeMap::from([("matrix".to_string(), matrix.clone())]);
    let mut bind = |name: &str, operand: &DynamicTensor| {
        inputs.insert(name.to_string(), operand.astype(dtype));
    };
    let (graph, result) = match derivative {
        LinalgDerivative::Vjp => {
            let reverse = reference.symbolic_vjp(output, "cotangent")?;
            bind("cotangent", operands[1]);
            let gradient = reverse.gradients["matrix"];
            (reverse.graph, gradient)
        }
        LinalgDerivative::Jvp | LinalgDerivative::Mixed => {
            let first = reference.symbolic_jvp_with_tangent_inputs(
                output,
                &BTreeMap::from([("matrix".into(), "direction".into())]),
            )?;
            bind("direction", operands[1]);
            if matches!(derivative, LinalgDerivative::Jvp) {
                (first.graph, first.tangent)
            } else {
                let mixed = first.graph.symbolic_jvp_with_tangent_inputs(
                    first.tangent,
                    &BTreeMap::from([
                        ("matrix".into(), "second".into()),
                        ("direction".into(), "mixed".into()),
                    ]),
                )?;
                bind("second", operands[2]);
                bind("mixed", operands[3]);
                (mixed.graph, mixed.tangent)
            }
        }
    };
    graph.evaluate(result, &inputs)
}

fn symbolic_accumulate(
    graph: &mut TensorIr,
    cotangents: &mut [Option<TensorNodeId>],
    target: TensorNodeId,
    contribution: TensorNodeId,
) -> Result<(), String> {
    cotangents[target] = Some(match cotangents[target] {
        Some(existing) => graph.add(existing, contribution)?,
        None => contribution,
    });
    Ok(())
}

fn symbolic_jvp_cond(
    transformed: &mut TensorIr,
    predicate: TensorNodeId,
    branches: &TensorCondExecutionPlan,
    captures: &[(String, TensorNodeId)],
    parent_pairs: &[(TensorNodeId, TensorNodeId)],
    namespace: &str,
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let tangent_names = symbolic_cond_tangent_names(branches, namespace);
    let (true_value, true_tangent) = symbolic_jvp_region(&branches.on_true.plan, &tangent_names)?;
    let (false_value, false_tangent) =
        symbolic_jvp_region(&branches.on_false.plan, &tangent_names)?;
    let value_branches = TensorCondExecutionPlan::new(true_value, false_value)?;
    let tangent_branches = TensorCondExecutionPlan::new(true_tangent, false_tangent)?;
    let value = transformed.cond_with_captures(
        predicate,
        value_branches.clone(),
        symbolic_jvp_cond_captures(&value_branches, captures, parent_pairs, &tangent_names)?,
    )?;
    let tangent = transformed.cond_with_captures(
        predicate,
        tangent_branches.clone(),
        symbolic_jvp_cond_captures(&tangent_branches, captures, parent_pairs, &tangent_names)?,
    )?;
    Ok((value, tangent))
}

fn symbolic_jvp_fori(
    transformed: &mut TensorIr,
    loop_plan: &TensorForiExecutionPlan,
    carry: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    parent_pairs: &[(TensorNodeId, TensorNodeId)],
    _namespace: &str,
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let (initial_value, initial_tangent) = *parent_pairs
        .get(carry)
        .ok_or_else(|| format!("fori carry node {carry} has no symbolic JVP pair"))?;
    let mut primal_captures = Vec::with_capacity(captures.len());
    let mut tangent_captures = Vec::with_capacity(captures.len());
    for (name, parent) in captures {
        let (value, tangent) = *parent_pairs
            .get(*parent)
            .ok_or_else(|| format!("fori capture node {parent} has no symbolic JVP pair"))?;
        primal_captures.push((name.clone(), value));
        tangent_captures.push((name.clone(), tangent));
    }
    let value = transformed.fori(initial_value, loop_plan.clone(), primal_captures.clone())?;
    let tangent = transformed.fori_jvp(
        initial_value,
        initial_tangent,
        loop_plan.clone(),
        primal_captures,
        tangent_captures,
    )?;
    Ok((value, tangent))
}

/// Forward mode through a while loop is another while loop over the packed
/// `[primal, tangent]` carry: the predicate reads only the primal half, so the
/// tangent follows exactly the primal iterations (as in JAX).
fn symbolic_jvp_while(
    transformed: &mut TensorIr,
    loop_plan: &TensorWhileExecutionPlan,
    carry: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    parent_pairs: &[(TensorNodeId, TensorNodeId)],
    namespace: &str,
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let carry_shape = loop_plan.carry_shape()?;
    let (augmented_plan, tangent_names) = symbolic_jvp_while_plan(loop_plan, namespace)?;
    let (initial_value, initial_tangent) = *parent_pairs
        .get(carry)
        .ok_or_else(|| format!("while carry node {carry} has no symbolic JVP pair"))?;
    let packed_initial =
        symbolic_pack_tensor_pair(transformed, initial_value, initial_tangent, &carry_shape)?;
    let parent_captures = captures.iter().cloned().collect::<BTreeMap<_, _>>();
    let augmented_captures = augmented_plan
        .external_captures()
        .keys()
        .map(|name| {
            if let Some(parent) = parent_captures.get(name) {
                return parent_pairs
                    .get(*parent)
                    .map(|pair| (name.clone(), pair.0))
                    .ok_or_else(|| {
                        format!("while capture node {parent} has no symbolic JVP value")
                    });
            }
            let source = tangent_names
                .iter()
                .find_map(|(source, tangent)| (tangent == name).then_some(source))
                .ok_or_else(|| format!("symbolic While JVP has unknown capture {name:?}"))?;
            let parent = parent_captures.get(source).ok_or_else(|| {
                format!("symbolic While JVP tangent capture {source:?} has no parent binding")
            })?;
            parent_pairs
                .get(*parent)
                .map(|pair| (name.clone(), pair.1))
                .ok_or_else(|| format!("while capture node {parent} has no symbolic JVP tangent"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let packed = transformed.while_loop(packed_initial, augmented_plan, augmented_captures)?;
    symbolic_unpack_tensor_pair(transformed, packed, &carry_shape)
}

fn symbolic_jvp_while_plan(
    loop_plan: &TensorWhileExecutionPlan,
    namespace: &str,
) -> Result<(TensorWhileExecutionPlan, BTreeMap<String, String>), String> {
    let body_plan = &loop_plan.body.plan;
    let predicate_plan = &loop_plan.predicate.plan;
    let mut captures = loop_plan.external_captures.clone();
    captures.insert(loop_plan.carry_name.clone(), loop_plan.carry_shape()?);
    let input_dtype = |name: &str| {
        body_plan
            .input_dtype(name)
            .or_else(|_| predicate_plan.input_dtype(name))
    };
    let mut tangent_names = BTreeMap::new();
    for (index, name) in captures.keys().enumerate() {
        let mut tangent_name = format!("{namespace}_tangent_{index}");
        while captures.contains_key(&tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &tangent_name)
        {
            tangent_name.push('_');
        }
        tangent_names.insert(name.clone(), tangent_name);
    }
    let body = body_plan.as_ir();
    let body_transform =
        body.symbolic_jvp_with_seed(body_plan.output_node_id, |graph, name, value, shape| {
            tangent_names
                .get(name)
                .map(|tangent_name| {
                    let dtype = graph.node_dtype(value)?;
                    graph.input_typed(tangent_name.clone(), shape.to_vec(), dtype)
                })
                .transpose()
        })?;

    let carry_shape = loop_plan.carry_shape()?;
    let mut packed_carry_shape = vec![2];
    packed_carry_shape.extend_from_slice(&carry_shape);
    let packed_name = format!("{namespace}_carry");
    let mut augmented = TensorIr::new();
    let packed_carry = augmented.input_typed(
        packed_name.clone(),
        packed_carry_shape,
        input_dtype(&loop_plan.carry_name)?,
    )?;
    let (primal_carry, tangent_carry) =
        symbolic_unpack_tensor_pair(&mut augmented, packed_carry, &carry_shape)?;
    let mut replacements = BTreeMap::new();
    for (name, shape) in &captures {
        let replacement = if name == &loop_plan.carry_name {
            primal_carry
        } else {
            augmented.input_typed(name.clone(), shape.clone(), input_dtype(name)?)?
        };
        replacements.insert(name.clone(), replacement);
    }
    for (source, tangent_name) in &tangent_names {
        let replacement = if source == &loop_plan.carry_name {
            tangent_carry
        } else {
            augmented.input_typed(
                tangent_name.clone(),
                captures[source].clone(),
                input_dtype(source)?,
            )?
        };
        replacements.insert(tangent_name.clone(), replacement);
    }
    let next_value = symbolic_clone_with_input_replacements(
        &mut augmented,
        &body_transform.graph,
        body_transform.value,
        &replacements,
    )?;
    let next_tangent = symbolic_clone_with_input_replacements(
        &mut augmented,
        &body_transform.graph,
        body_transform.tangent,
        &replacements,
    )?;
    let next_packed =
        symbolic_pack_tensor_pair(&mut augmented, next_value, next_tangent, &carry_shape)?;
    let predicate_ir = predicate_plan.as_ir();
    let keep_going = symbolic_clone_with_input_replacements(
        &mut augmented,
        &predicate_ir,
        predicate_plan.output_node_id,
        &replacements,
    )?;
    Ok((
        TensorWhileExecutionPlan::new(
            augmented.compile_cpu(keep_going)?,
            augmented.compile_cpu(next_packed)?,
            packed_name,
        )?,
        tangent_names,
    ))
}

fn symbolic_jvp_scan(
    transformed: &mut TensorIr,
    scan_plan: &TensorScanExecutionPlan,
    carry: TensorNodeId,
    captures: &[(String, TensorNodeId)],
    parent_pairs: &[(TensorNodeId, TensorNodeId)],
    namespace: &str,
) -> Result<SymbolicScanPair, String> {
    let carry_shape = scan_plan.carry_shape()?;
    let output_shape = scan_plan.output_shape()?;
    let (augmented_plan, tangent_names) = symbolic_jvp_scan_plan(scan_plan, namespace)?;
    let (initial_value, initial_tangent) = *parent_pairs
        .get(carry)
        .ok_or_else(|| format!("scan carry node {carry} has no symbolic JVP pair"))?;
    let packed_initial =
        symbolic_pack_tensor_pair(transformed, initial_value, initial_tangent, &carry_shape)?;
    let parent_captures = captures.iter().cloned().collect::<BTreeMap<_, _>>();
    let augmented_captures = augmented_plan
        .external_captures()
        .keys()
        .map(|name| {
            if let Some(parent) = parent_captures.get(name) {
                return parent_pairs
                    .get(*parent)
                    .map(|pair| (name.clone(), pair.0))
                    .ok_or_else(|| {
                        format!("scan capture node {parent} has no symbolic JVP value")
                    });
            }
            let source = tangent_names
                .iter()
                .find_map(|(source, tangent)| (tangent == name).then_some(source))
                .ok_or_else(|| format!("symbolic Scan JVP has unknown capture {name:?}"))?;
            let parent = parent_captures.get(source).ok_or_else(|| {
                format!("symbolic Scan JVP tangent capture {source:?} has no parent binding")
            })?;
            parent_pairs
                .get(*parent)
                .map(|pair| (name.clone(), pair.1))
                .ok_or_else(|| format!("scan capture node {parent} has no symbolic JVP tangent"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (packed_carry, packed_outputs) =
        transformed.scan(packed_initial, augmented_plan, augmented_captures)?;
    let carry_pair = symbolic_unpack_tensor_pair(transformed, packed_carry, &carry_shape)?;
    let output_pair =
        symbolic_unpack_tensor_pair_axis(transformed, packed_outputs, 1, &output_shape)?;
    Ok((carry_pair, output_pair))
}

fn symbolic_jvp_scan_plan(
    scan_plan: &TensorScanExecutionPlan,
    namespace: &str,
) -> Result<(TensorScanExecutionPlan, BTreeMap<String, String>), String> {
    let body_plan = &scan_plan.body.plan;
    let body = body_plan.as_ir();
    let captures = scan_plan.body.captures.clone();
    let mut tangent_names = BTreeMap::new();
    for (index, name) in captures.keys().enumerate() {
        if name == &scan_plan.index_name {
            continue;
        }
        let mut tangent_name = format!("{namespace}_tangent_{index}");
        while captures.contains_key(&tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &tangent_name)
        {
            tangent_name.push('_');
        }
        tangent_names.insert(name.clone(), tangent_name);
    }
    let transform_output = |output| {
        body.symbolic_jvp_with_seed(output, |graph, name, value, shape| {
            tangent_names
                .get(name)
                .map(|tangent_name| {
                    let dtype = graph.node_dtype(value)?;
                    graph.input_typed(tangent_name.clone(), shape.to_vec(), dtype)
                })
                .transpose()
        })
    };
    let carry_transform = transform_output(body_plan.output_node_ids[0])?;
    let output_transform = transform_output(body_plan.output_node_ids[1])?;

    let carry_shape = scan_plan.carry_shape()?;
    let output_shape = scan_plan
        .body
        .output_shapes()
        .get(1)
        .cloned()
        .ok_or_else(|| "scan body output is missing".to_string())?;
    let mut packed_carry_shape = vec![2];
    packed_carry_shape.extend_from_slice(&carry_shape);
    let mut augmented = TensorIr::new();
    let packed_carry = augmented.input_typed(
        "__quabla_scan_jvp_carry",
        packed_carry_shape,
        body_plan.input_dtype(&scan_plan.carry_name)?,
    )?;
    let (primal_carry, tangent_carry) =
        symbolic_unpack_tensor_pair(&mut augmented, packed_carry, &carry_shape)?;
    let mut replacements = BTreeMap::new();
    for (name, shape) in &captures {
        let replacement = if name == &scan_plan.carry_name {
            primal_carry
        } else if name == &scan_plan.index_name {
            augmented.input_typed(name.clone(), vec![], body_plan.input_dtype(name)?)?
        } else {
            augmented.input_typed(name.clone(), shape.clone(), body_plan.input_dtype(name)?)?
        };
        replacements.insert(name.clone(), replacement);
    }
    for (source, tangent_name) in &tangent_names {
        let replacement = if source == &scan_plan.carry_name {
            tangent_carry
        } else {
            augmented.input_typed(
                tangent_name.clone(),
                captures
                    .get(source)
                    .cloned()
                    .ok_or_else(|| format!("scan body capture {source:?} is missing"))?,
                body_plan.input_dtype(source)?,
            )?
        };
        replacements.insert(tangent_name.clone(), replacement);
    }
    let carry_value = symbolic_clone_with_input_replacements(
        &mut augmented,
        &carry_transform.graph,
        carry_transform.value,
        &replacements,
    )?;
    let carry_tangent = symbolic_clone_with_input_replacements(
        &mut augmented,
        &carry_transform.graph,
        carry_transform.tangent,
        &replacements,
    )?;
    let output_value = symbolic_clone_with_input_replacements(
        &mut augmented,
        &output_transform.graph,
        output_transform.value,
        &replacements,
    )?;
    let output_tangent = symbolic_clone_with_input_replacements(
        &mut augmented,
        &output_transform.graph,
        output_transform.tangent,
        &replacements,
    )?;
    let packed_carry =
        symbolic_pack_tensor_pair(&mut augmented, carry_value, carry_tangent, &carry_shape)?;
    let packed_output =
        symbolic_pack_tensor_pair(&mut augmented, output_value, output_tangent, &output_shape)?;
    let (body_plan, _) = augmented.compile_cpu_many(&[packed_carry, packed_output])?;
    Ok((
        TensorScanExecutionPlan::new(
            scan_plan.lower,
            scan_plan.upper,
            body_plan,
            "__quabla_scan_jvp_carry",
            scan_plan.index_name.clone(),
        )?,
        tangent_names,
    ))
}

fn symbolic_clone_with_input_replacements(
    destination: &mut TensorIr,
    source: &TensorIr,
    output: TensorNodeId,
    replacements: &BTreeMap<String, TensorNodeId>,
) -> Result<TensorNodeId, String> {
    source.node(output)?;
    let mut remap = HashMap::new();
    for (node_id, node) in source.nodes.iter().enumerate() {
        let mapped = match &node.op {
            TensorOp::Input { name } => replacements.get(name).copied().ok_or_else(|| {
                format!("symbolic Fori JVP is missing input replacement {name:?}")
            })?,
            _ => {
                let op = remap_tensor_op(&node.op, &|id| remap.get(&id).copied())?;
                destination.push_node(op, node.shape.clone(), node.dtype, node.weak)
            }
        };
        remap.insert(node_id, mapped);
    }
    remap
        .get(&output)
        .copied()
        .ok_or_else(|| format!("symbolic Fori JVP output node {output} is missing"))
}

fn symbolic_pack_tensor_pair(
    graph: &mut TensorIr,
    value: TensorNodeId,
    tangent: TensorNodeId,
    shape: &[usize],
) -> Result<TensorNodeId, String> {
    let mut component_shape = Vec::with_capacity(shape.len() + 1);
    component_shape.push(1);
    component_shape.extend_from_slice(shape);
    let value = graph.reshape(value, component_shape.clone())?;
    let tangent = graph.reshape(tangent, component_shape)?;
    graph.concat(vec![value, tangent], 0)
}

fn symbolic_unpack_tensor_pair(
    graph: &mut TensorIr,
    pair: TensorNodeId,
    shape: &[usize],
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let value_slice = graph.slice(pair, 0, 0, 1)?;
    let value = graph.reshape(value_slice, shape.to_vec())?;
    let tangent_slice = graph.slice(pair, 0, 1, 1)?;
    let tangent = graph.reshape(tangent_slice, shape.to_vec())?;
    Ok((value, tangent))
}

fn symbolic_unpack_tensor_pair_axis(
    graph: &mut TensorIr,
    pair: TensorNodeId,
    axis: usize,
    shape: &[usize],
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let mut component_shape = shape.to_vec();
    component_shape.insert(axis, 1);
    let value_slice = graph.slice(pair, axis, 0, 1)?;
    let value = graph.reshape(value_slice, shape.to_vec())?;
    let tangent_slice = graph.slice(pair, axis, 1, 1)?;
    let tangent = graph.reshape(tangent_slice, shape.to_vec())?;
    // Retaining this construction documents the required packed layout and
    // validates the inserted singleton dimension at trace time.
    if graph.node(value_slice)?.shape != component_shape {
        return Err("symbolic tensor pair slice has an invalid component shape".to_string());
    }
    Ok((value, tangent))
}

fn symbolic_jvp_region(
    plan: &TensorExecutionPlan,
    tangent_names: &BTreeMap<String, String>,
) -> Result<(TensorExecutionPlan, TensorExecutionPlan), String> {
    let ir = plan.as_ir();
    let transformed = if tangent_names.is_empty() {
        ir.symbolic_jvp_with_seed(plan.output_node_id, |_, _, _, _| Ok(None))?
    } else {
        ir.symbolic_jvp_with_tangent_inputs(plan.output_node_id, tangent_names)?
    };
    let mut graph = transformed.graph;
    let input_names = symbolic_region_input_names(&graph);
    let value = symbolic_retain_region_inputs(&mut graph, &input_names, transformed.value)?;
    let tangent = symbolic_retain_region_inputs(&mut graph, &input_names, transformed.tangent)?;
    Ok((graph.compile_cpu(value)?, graph.compile_cpu(tangent)?))
}

fn symbolic_cond_tangent_names(
    branches: &TensorCondExecutionPlan,
    namespace: &str,
) -> BTreeMap<String, String> {
    let captures = branches.captures();
    captures
        .keys()
        // Bool captures have no tangent, so no tangent input is created.
        .filter(|name| branches.on_true.plan.input_dtype(name).ok() != Some(TensorDType::Bool))
        .enumerate()
        .map(|(index, name)| {
            let mut tangent_name = format!("{namespace}_tangent_{index}");
            while captures.contains_key(&tangent_name) {
                tangent_name.push('_');
            }
            (name.clone(), tangent_name)
        })
        .collect()
}

fn symbolic_jvp_cond_captures(
    transformed_branches: &TensorCondExecutionPlan,
    captures: &[(String, TensorNodeId)],
    parent_pairs: &[(TensorNodeId, TensorNodeId)],
    tangent_names: &BTreeMap<String, String>,
) -> Result<Vec<(String, TensorNodeId)>, String> {
    let parent_captures = captures.iter().cloned().collect::<BTreeMap<_, _>>();
    transformed_branches
        .captures()
        .keys()
        .map(|name| {
            if let Some(parent) = parent_captures.get(name) {
                return parent_pairs
                    .get(*parent)
                    .map(|pair| (name.clone(), pair.0))
                    .ok_or_else(|| {
                        format!("conditional capture node {parent} has no symbolic JVP value")
                    });
            }
            let source = tangent_names
                .iter()
                .find_map(|(source, tangent)| (tangent == name).then_some(source))
                .ok_or_else(|| format!("symbolic JVP region has unknown capture {name:?}"))?;
            let parent = parent_captures.get(source).ok_or_else(|| {
                format!("symbolic JVP tangent capture {source:?} has no parent binding")
            })?;
            parent_pairs
                .get(*parent)
                .map(|pair| (name.clone(), pair.1))
                .ok_or_else(|| {
                    format!("conditional capture node {parent} has no symbolic JVP tangent")
                })
        })
        .collect()
}

fn symbolic_clone_cond(
    transformed: &mut TensorIr,
    predicate: TensorNodeId,
    branches: &TensorCondExecutionPlan,
    captures: &[(String, TensorNodeId)],
    values: &[TensorNodeId],
) -> Result<TensorNodeId, String> {
    let captures = captures
        .iter()
        .map(|(name, node_id)| {
            values
                .get(*node_id)
                .map(|value| (name.clone(), *value))
                .ok_or_else(|| format!("conditional capture node {node_id} has no symbolic value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    transformed.cond_with_captures(predicate, branches.clone(), captures)
}

fn symbolic_clone_while(
    transformed: &mut TensorIr,
    carry: TensorNodeId,
    loop_plan: &TensorWhileExecutionPlan,
    captures: &[(String, TensorNodeId)],
    values: &[TensorNodeId],
) -> Result<TensorNodeId, String> {
    let carry = values
        .get(carry)
        .copied()
        .ok_or_else(|| format!("while carry node {carry} has no symbolic value"))?;
    let captures = captures
        .iter()
        .map(|(name, node_id)| {
            values
                .get(*node_id)
                .map(|value| (name.clone(), *value))
                .ok_or_else(|| format!("while capture node {node_id} has no symbolic value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    transformed.while_loop(carry, loop_plan.clone(), captures)
}

fn symbolic_clone_fori(
    transformed: &mut TensorIr,
    carry: TensorNodeId,
    loop_plan: &TensorForiExecutionPlan,
    captures: &[(String, TensorNodeId)],
    values: &[TensorNodeId],
) -> Result<TensorNodeId, String> {
    let carry = values
        .get(carry)
        .copied()
        .ok_or_else(|| format!("fori carry node {carry} has no symbolic value"))?;
    let captures = captures
        .iter()
        .map(|(name, node_id)| {
            values
                .get(*node_id)
                .map(|value| (name.clone(), *value))
                .ok_or_else(|| format!("fori capture node {node_id} has no symbolic value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    transformed.fori(carry, loop_plan.clone(), captures)
}

struct SymbolicJvpForiVjpContext<'a> {
    loop_plan: &'a TensorForiExecutionPlan,
    carry: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: &'a [(String, TensorNodeId)],
    target: &'a TensorForiVjpTarget,
    source_group: usize,
}

fn symbolic_jvp_fori_vjp(
    transformed: &mut TensorIr,
    context: SymbolicJvpForiVjpContext<'_>,
    pairs: &[(TensorNodeId, TensorNodeId)],
    groups: &mut HashMap<usize, (usize, usize, TensorForiVjpJvpExecutionPlan)>,
    namespace: &str,
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let (primal_group, directional_group, plan) =
        if let Some(group) = groups.get(&context.source_group) {
            group.clone()
        } else {
            let plan = TensorForiVjpJvpExecutionPlan::new(context.loop_plan.clone(), namespace)?;
            let primal_group = transformed.nodes.len();
            let directional_group = primal_group
                .checked_add(1)
                .ok_or_else(|| "Fori VJP JVP group identifier overflows usize".to_string())?;
            groups.insert(
                context.source_group,
                (primal_group, directional_group, plan.clone()),
            );
            (primal_group, directional_group, plan)
        };
    let pair = |node_id: TensorNodeId, label: &str| {
        pairs
            .get(node_id)
            .copied()
            .ok_or_else(|| format!("Fori VJP {label} node {node_id} has no symbolic pair"))
    };
    let (carry, carry_tangent) = pair(context.carry, "carry")?;
    let (output_cotangent, output_cotangent_tangent) =
        pair(context.output_cotangent, "output cotangent")?;
    let mut primal_captures = Vec::with_capacity(context.captures.len());
    let mut tangent_captures = Vec::with_capacity(context.captures.len());
    for (name, node_id) in context.captures {
        let (value, tangent) = pair(*node_id, "capture")?;
        primal_captures.push((name.clone(), value));
        tangent_captures.push((name.clone(), tangent));
    }
    let value = transformed.fori_vjp(
        carry,
        output_cotangent,
        context.loop_plan.clone(),
        primal_captures.clone(),
        context.target.clone(),
        primal_group,
    )?;
    let tangent = transformed.fori_vjp_jvp(
        plan,
        TensorForiVjpJvpBindings {
            carry,
            carry_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures: primal_captures,
            tangent_captures,
        },
        context.target.clone(),
        directional_group,
    )?;
    Ok((value, tangent))
}

struct SymbolicJvpScanVjpContext<'a> {
    scan_plan: &'a TensorScanExecutionPlan,
    carry: TensorNodeId,
    final_carry_cotangent: TensorNodeId,
    output_cotangent: TensorNodeId,
    captures: &'a [(String, TensorNodeId)],
    target: &'a TensorScanVjpTarget,
    source_group: usize,
}

fn symbolic_jvp_scan_vjp(
    transformed: &mut TensorIr,
    context: SymbolicJvpScanVjpContext<'_>,
    pairs: &[(TensorNodeId, TensorNodeId)],
    groups: &mut HashMap<usize, (usize, usize, TensorScanVjpJvpExecutionPlan)>,
    namespace: &str,
) -> Result<(TensorNodeId, TensorNodeId), String> {
    let (primal_group, directional_group, plan) =
        if let Some(group) = groups.get(&context.source_group) {
            group.clone()
        } else {
            let plan = TensorScanVjpJvpExecutionPlan::new(context.scan_plan.clone(), namespace)?;
            let primal_group = transformed.nodes.len();
            let directional_group = primal_group
                .checked_add(1)
                .ok_or_else(|| "Scan VJP JVP group identifier overflows usize".to_string())?;
            groups.insert(
                context.source_group,
                (primal_group, directional_group, plan.clone()),
            );
            (primal_group, directional_group, plan)
        };
    let pair = |node_id: TensorNodeId, label: &str| {
        pairs
            .get(node_id)
            .copied()
            .ok_or_else(|| format!("Scan VJP {label} node {node_id} has no symbolic pair"))
    };
    let (carry, carry_tangent) = pair(context.carry, "carry")?;
    let (final_carry_cotangent, final_carry_cotangent_tangent) =
        pair(context.final_carry_cotangent, "final carry cotangent")?;
    let (output_cotangent, output_cotangent_tangent) =
        pair(context.output_cotangent, "output cotangent")?;
    let mut primal_captures = Vec::with_capacity(context.captures.len());
    let mut tangent_captures = Vec::with_capacity(context.captures.len());
    for (name, node_id) in context.captures {
        let (value, tangent) = pair(*node_id, "capture")?;
        primal_captures.push((name.clone(), value));
        tangent_captures.push((name.clone(), tangent));
    }
    let value = transformed.scan_vjp(
        context.scan_plan.clone(),
        TensorScanVjpBindings {
            carry,
            final_carry_cotangent,
            output_cotangent,
            captures: primal_captures.clone(),
        },
        context.target.clone(),
        primal_group,
    )?;
    let tangent = transformed.scan_vjp_jvp(
        plan,
        TensorScanVjpJvpBindings {
            carry,
            carry_tangent,
            final_carry_cotangent,
            final_carry_cotangent_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures: primal_captures,
            tangent_captures,
        },
        context.target.clone(),
        directional_group,
    )?;
    Ok((value, tangent))
}

struct SymbolicVjpForiContext<'a> {
    original_carry: TensorNodeId,
    carry: TensorNodeId,
    captures: &'a [(String, TensorNodeId)],
    values: &'a [TensorNodeId],
    upstream: TensorNodeId,
}

fn symbolic_vjp_fori(
    transformed: &mut TensorIr,
    loop_plan: &TensorForiExecutionPlan,
    context: SymbolicVjpForiContext<'_>,
    cotangents: &mut [Option<TensorNodeId>],
    group: usize,
) -> Result<(), String> {
    let transformed_captures = context
        .captures
        .iter()
        .map(|(name, node_id)| {
            context
                .values
                .get(*node_id)
                .map(|value| (name.clone(), *value))
                .ok_or_else(|| format!("fori capture node {node_id} has no symbolic value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let carry_gradient = transformed.fori_vjp(
        context.carry,
        context.upstream,
        loop_plan.clone(),
        transformed_captures.clone(),
        TensorForiVjpTarget::Carry,
        group,
    )?;
    symbolic_accumulate(
        transformed,
        cotangents,
        context.original_carry,
        carry_gradient,
    )?;
    for (name, parent_node_id) in context.captures {
        let gradient = transformed.fori_vjp(
            context.carry,
            context.upstream,
            loop_plan.clone(),
            transformed_captures.clone(),
            TensorForiVjpTarget::External(name.clone()),
            group,
        )?;
        symbolic_accumulate(transformed, cotangents, *parent_node_id, gradient)?;
    }
    Ok(())
}

struct SymbolicVjpScanContext<'a> {
    original_carry: TensorNodeId,
    carry: TensorNodeId,
    captures: &'a [(String, TensorNodeId)],
    values: &'a [TensorNodeId],
    final_carry_upstream: TensorNodeId,
    output_upstream: TensorNodeId,
}

fn symbolic_vjp_scan(
    transformed: &mut TensorIr,
    scan_plan: &TensorScanExecutionPlan,
    context: SymbolicVjpScanContext<'_>,
    cotangents: &mut [Option<TensorNodeId>],
    group: usize,
) -> Result<(), String> {
    let transformed_captures = context
        .captures
        .iter()
        .map(|(name, node_id)| {
            context
                .values
                .get(*node_id)
                .map(|value| (name.clone(), *value))
                .ok_or_else(|| format!("scan capture node {node_id} has no symbolic value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let carry_gradient = transformed.scan_vjp(
        scan_plan.clone(),
        TensorScanVjpBindings {
            carry: context.carry,
            final_carry_cotangent: context.final_carry_upstream,
            output_cotangent: context.output_upstream,
            captures: transformed_captures.clone(),
        },
        TensorScanVjpTarget::Carry,
        group,
    )?;
    symbolic_accumulate(
        transformed,
        cotangents,
        context.original_carry,
        carry_gradient,
    )?;
    for (name, parent_node_id) in context.captures {
        let gradient = transformed.scan_vjp(
            scan_plan.clone(),
            TensorScanVjpBindings {
                carry: context.carry,
                final_carry_cotangent: context.final_carry_upstream,
                output_cotangent: context.output_upstream,
                captures: transformed_captures.clone(),
            },
            TensorScanVjpTarget::External(name.clone()),
            group,
        )?;
        symbolic_accumulate(transformed, cotangents, *parent_node_id, gradient)?;
    }
    Ok(())
}

struct SymbolicVjpCondContext<'a> {
    captures: &'a [(String, TensorNodeId)],
    values: &'a [TensorNodeId],
    upstream: TensorNodeId,
}

fn symbolic_vjp_cond(
    transformed: &mut TensorIr,
    predicate: TensorNodeId,
    branches: &TensorCondExecutionPlan,
    context: SymbolicVjpCondContext<'_>,
    cotangents: &mut [Option<TensorNodeId>],
    node_id: TensorNodeId,
) -> Result<(), String> {
    let cotangent_name = symbolic_cond_vjp_cotangent_name(branches, node_id);
    let true_gradients = symbolic_vjp_region(&branches.on_true.plan, &cotangent_name)?;
    let false_gradients = symbolic_vjp_region(&branches.on_false.plan, &cotangent_name)?;
    for (capture_name, parent_node_id) in context.captures {
        // Bool captures are not differentiable, so the region VJP produces no gradient for them.
        if branches.on_true.plan.input_dtype(capture_name)? == TensorDType::Bool {
            continue;
        }
        let on_true = true_gradients.get(capture_name).ok_or_else(|| {
            format!("true conditional region has no gradient for capture {capture_name:?}")
        })?;
        let on_false = false_gradients.get(capture_name).ok_or_else(|| {
            format!("false conditional region has no gradient for capture {capture_name:?}")
        })?;
        let gradient_branches = TensorCondExecutionPlan::new(on_true.clone(), on_false.clone())?;
        let gradient = transformed.cond_with_captures(
            predicate,
            gradient_branches.clone(),
            symbolic_vjp_cond_captures(
                &gradient_branches,
                context.captures,
                context.values,
                context.upstream,
                &cotangent_name,
            )?,
        )?;
        symbolic_accumulate(transformed, cotangents, *parent_node_id, gradient)?;
    }
    Ok(())
}

fn symbolic_vjp_region(
    plan: &TensorExecutionPlan,
    cotangent_name: &str,
) -> Result<BTreeMap<String, TensorExecutionPlan>, String> {
    let transformed = plan
        .as_ir()
        .symbolic_vjp(plan.output_node_id, cotangent_name)?;
    let mut graph = transformed.graph;
    let input_names = symbolic_region_input_names(&graph);
    transformed
        .gradients
        .into_iter()
        .map(|(name, mut output)| {
            output = symbolic_retain_region_inputs(&mut graph, &input_names, output)?;
            graph.compile_cpu(output).map(|plan| (name, plan))
        })
        .collect()
}

fn symbolic_region_input_names(graph: &TensorIr) -> Vec<String> {
    graph
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            TensorOp::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn fresh_tensor_input_name(graph: &TensorIr, prefix: &str) -> String {
    let existing = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            TensorOp::Input { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut name = prefix.to_string();
    while existing.contains(name.as_str()) {
        name.push('_');
    }
    name
}

fn symbolic_retain_region_inputs(
    graph: &mut TensorIr,
    input_names: &[String],
    mut output: TensorNodeId,
) -> Result<TensorNodeId, String> {
    // A region's capture interface must not depend on which branch or AD
    // output happens to use an input after DCE.
    //
    // The retaining term is `where(0, input, 0)`: the constant-false mask
    // references the input without reading its values, so a `NaN`/`inf`
    // capture cannot leak into the output as `input - input` would.
    for input_name in input_names {
        let input = graph.input_node_id(input_name)?;
        let zero = graph.scalar_constant(0.0);
        let zero = graph.where_select(zero, input, zero)?;
        let mut scalar_zero = graph.sum(zero)?;
        // Regions may mix dtypes (e.g. an f32 carry with an f64 capture), so the zero term is cast
        // to the output dtype.
        let output_dtype = graph.node_dtype(output)?;
        if graph.node_dtype(scalar_zero)? != output_dtype {
            scalar_zero = graph.cast(scalar_zero, output_dtype)?;
        }
        let zero_like_output = symbolic_broadcast_like(graph, scalar_zero, output)?;
        output = graph.add(output, zero_like_output)?;
    }
    Ok(output)
}

fn symbolic_cond_vjp_cotangent_name(
    branches: &TensorCondExecutionPlan,
    node_id: TensorNodeId,
) -> String {
    let captures = branches.captures();
    let mut name = format!("__quabla_cond_vjp_cotangent_{node_id}");
    while captures.contains_key(&name) {
        name.push('_');
    }
    name
}

fn symbolic_vjp_cond_captures(
    transformed_branches: &TensorCondExecutionPlan,
    captures: &[(String, TensorNodeId)],
    values: &[TensorNodeId],
    upstream: TensorNodeId,
    cotangent_name: &str,
) -> Result<Vec<(String, TensorNodeId)>, String> {
    let parent_captures = captures.iter().cloned().collect::<BTreeMap<_, _>>();
    transformed_branches
        .captures()
        .keys()
        .map(|name| {
            if name == cotangent_name {
                return Ok((name.clone(), upstream));
            }
            let parent = parent_captures
                .get(name)
                .ok_or_else(|| format!("symbolic VJP region has unknown capture {name:?}"))?;
            let value = values.get(*parent).ok_or_else(|| {
                format!("conditional capture node {parent} has no symbolic value")
            })?;
            Ok((name.clone(), *value))
        })
        .collect()
}

/// A tensor filled with `value`, with the shape, dtype and weakness of
/// `target`: a scalar constant broadcast to `target`'s shape (no broadcast
/// for a scalar `target`).
///
/// Neither the result nor any of its derivatives reads the values of
/// `target`, so it stays exact where `target` is infinite or NaN; zeros
/// built as `target - target` would be NaN there.
fn symbolic_full_like(
    graph: &mut TensorIr,
    value: f64,
    target: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let target_node = graph.node(target)?;
    let (shape, dtype, weak) = (
        target_node.shape.clone(),
        target_node.dtype,
        target_node.weak,
    );
    let constant = graph.constant_like(value, dtype, weak);
    if shape.is_empty() {
        Ok(constant)
    } else {
        graph.broadcast_to(constant, shape)
    }
}

/// Zeros shaped and typed like `target` (see [`symbolic_full_like`]): the
/// tangent or cotangent that an AD rule knows to be exactly zero.
fn symbolic_zero_like(graph: &mut TensorIr, target: TensorNodeId) -> Result<TensorNodeId, String> {
    symbolic_full_like(graph, 0.0, target)
}

/// Weak `f64` zero of `shape`: the tangent of a `Bool` value.
fn symbolic_zero_tangent(graph: &mut TensorIr, shape: &[usize]) -> Result<TensorNodeId, String> {
    let zero = graph.scalar_constant(0.0);
    if shape.is_empty() {
        Ok(zero)
    } else {
        graph.broadcast_to(zero, shape.to_vec())
    }
}

/// Broadcasts `value` to the shape of `target`, promoted with the dtype of
/// `target` as `value * ones_like(target)`.
///
/// The ones come from [`symbolic_full_like`], so neither the result nor any
/// of its derivatives reads the values of `target`, and forward-over-reverse
/// Hessians of a loss with an infinite or NaN entry stay finite.
fn symbolic_broadcast_like(
    graph: &mut TensorIr,
    value: TensorNodeId,
    target: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let ones = symbolic_full_like(graph, 1.0, target)?;
    graph.mul(value, ones)
}

fn symbolic_expand_reduced_axis(
    graph: &mut TensorIr,
    value: TensorNodeId,
    target: TensorNodeId,
    axis: usize,
) -> Result<TensorNodeId, String> {
    let target_shape = graph.node_shape(target)?;
    if axis >= target_shape.len() {
        return Err(format!(
            "cannot expand symbolic reduced axis {axis} for shape {target_shape:?}"
        ));
    }
    let mut expanded_shape = target_shape.clone();
    expanded_shape[axis] = 1;
    let value_shape = graph.node_shape(value)?;
    let expected_shape = reduced_shape(&target_shape, axis)?;
    if value_shape != expected_shape {
        return Err(format!(
            "cannot expand symbolic reduced tensor shape {value_shape:?} along axis {axis} to {target_shape:?}"
        ));
    }
    let expanded = graph.reshape(value, expanded_shape)?;
    graph.broadcast_to(expanded, target_shape)
}

fn symbolic_reduce_to_shape(
    graph: &mut TensorIr,
    value: TensorNodeId,
    source_shape: &[usize],
    target_shape: &[usize],
) -> Result<TensorNodeId, String> {
    if target_shape.len() > source_shape.len() {
        return Err(format!(
            "cannot symbolically reduce tensor shape {source_shape:?} to higher-rank shape {target_shape:?}"
        ));
    }
    let rank_offset = source_shape.len() - target_shape.len();
    for (source_extent, target_extent) in source_shape[rank_offset..].iter().zip(target_shape) {
        if source_extent != target_extent && *target_extent != 1 {
            return Err(format!(
                "cannot symbolically reduce tensor shape {source_shape:?} to {target_shape:?}"
            ));
        }
    }

    let mut reduced = value;
    for axis in (0..source_shape.len()).rev() {
        let must_reduce =
            axis < rank_offset || target_shape[axis - rank_offset] == 1 && source_shape[axis] != 1;
        if must_reduce {
            reduced = graph.sum_axis(reduced, axis as isize)?;
        }
    }
    graph.reshape(reduced, target_shape.to_vec())
}

fn symbolic_transpose_last_two(
    graph: &mut TensorIr,
    value: TensorNodeId,
    shape: &[usize],
) -> Result<TensorNodeId, String> {
    if shape.len() < 2 {
        return Err(format!(
            "symbolic VJP matmul requires rank-two operands, got shape {shape:?}"
        ));
    }
    let mut axes = (0..shape.len()).collect::<Vec<_>>();
    let last = axes.len() - 1;
    axes.swap(last - 1, last);
    graph.transpose(
        value,
        Some(axes.into_iter().map(|axis| axis as isize).collect()),
    )
}

impl TensorRegion {
    pub fn new(plan: TensorExecutionPlan) -> Self {
        let captures = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                TensorOp::Input { name } => Some((name.clone(), node.shape.clone())),
                _ => None,
            })
            .collect();
        Self { plan, captures }
    }

    pub fn captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.captures
    }

    pub fn output_shape(&self) -> Result<Vec<usize>, String> {
        self.plan.output_shape()
    }
}

impl TensorMultiRegion {
    pub fn new(plan: TensorExecutionPlan) -> Result<Self, String> {
        if plan.output_node_ids.is_empty() {
            return Err("multi-result region requires at least one output".to_string());
        }
        let captures = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                TensorOp::Input { name } => Some((name.clone(), node.shape.clone())),
                _ => None,
            })
            .collect();
        let output_shapes = plan
            .output_node_ids
            .iter()
            .map(|output| {
                plan.nodes
                    .get(*output)
                    .map(|node| node.shape.clone())
                    .ok_or_else(|| format!("multi-result region output node {output} is missing"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            plan,
            captures,
            output_shapes,
        })
    }

    pub fn captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.captures
    }

    pub fn output_shapes(&self) -> &[Vec<usize>] {
        &self.output_shapes
    }

    pub fn evaluate(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        self.plan.evaluate_many(inputs)
    }
}

impl TensorCondExecutionPlan {
    pub fn new(
        on_true: TensorExecutionPlan,
        on_false: TensorExecutionPlan,
    ) -> Result<Self, String> {
        let on_true = TensorRegion::new(on_true);
        let on_false = TensorRegion::new(on_false);
        if on_true.output_shape()? != on_false.output_shape()? {
            return Err(format!(
                "conditional branch output shapes differ: {:?} versus {:?}",
                on_true.output_shape()?,
                on_false.output_shape()?
            ));
        }
        if on_true.captures != on_false.captures {
            return Err(
                "conditional branches must capture identical named inputs and shapes".to_string(),
            );
        }
        if !on_true.plan.output_dtype()?.is_floating() {
            return Err(format!(
                "conditional branches must return a floating dtype, got {}; \
                 convert bool results explicitly with astype",
                on_true.plan.output_dtype()?
            ));
        }
        if on_true.plan.output_dtype()? != on_false.plan.output_dtype()? {
            return Err(format!(
                "conditional branch output dtypes differ: {} versus {}",
                on_true.plan.output_dtype()?,
                on_false.plan.output_dtype()?
            ));
        }
        for name in on_true.captures.keys() {
            if on_true.plan.input_dtype(name)? != on_false.plan.input_dtype(name)? {
                return Err(format!(
                    "conditional branches capture {name:?} with different dtypes"
                ));
            }
        }
        Ok(Self { on_true, on_false })
    }

    fn selected(&self, predicate: bool) -> &TensorExecutionPlan {
        if predicate {
            &self.on_true.plan
        } else {
            &self.on_false.plan
        }
    }

    pub fn captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.on_true.captures
    }

    pub fn output_shape(&self) -> Result<Vec<usize>, String> {
        self.on_true.output_shape()
    }

    pub fn evaluate(
        &self,
        predicate: bool,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.selected(predicate).evaluate(inputs)
    }

    pub fn jvp(
        &self,
        predicate: bool,
        inputs: &BTreeMap<String, DynamicTensor>,
        tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.selected(predicate).jvp(inputs, tangents)
    }

    pub fn value_and_vjp(
        &self,
        predicate: bool,
        inputs: &BTreeMap<String, DynamicTensor>,
        cotangent: DynamicTensor,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        self.selected(predicate).value_and_vjp(inputs, cotangent)
    }

    pub fn true_node_count(&self) -> usize {
        self.on_true.plan.node_count()
    }

    pub fn false_node_count(&self) -> usize {
        self.on_false.plan.node_count()
    }
}

#[cfg(feature = "mlx")]
fn build_tensor_fori_mlx_vjp_plan(
    body_plan: &TensorExecutionPlan,
    captures: &BTreeMap<String, Vec<usize>>,
    output_node_id: TensorNodeId,
    carry_name: &str,
    external_captures: &BTreeMap<String, Vec<usize>>,
) -> Option<TensorMlxVjpPlan> {
    let mut cotangent_name = "__quabla_mlx_fori_cotangent".to_string();
    while captures.contains_key(&cotangent_name) {
        cotangent_name.push('_');
    }
    let symbolic = body_plan
        .as_ir()
        .symbolic_vjp(output_node_id, &cotangent_name)
        .ok()?;
    let mut names = vec![carry_name.to_string()];
    names.extend(external_captures.keys().cloned());
    let output_node_ids = names
        .iter()
        .map(|name| symbolic.gradients.get(name).copied())
        .collect::<Option<Vec<_>>>()?;
    let (plan, output_node_ids) = symbolic.graph.compile_cpu_many(&output_node_ids).ok()?;
    let gradient_node_ids = names.into_iter().zip(output_node_ids).collect();
    Some(TensorMlxVjpPlan {
        plan,
        cotangent_name,
        gradient_node_ids,
    })
}

// VJP consumers need block boundaries rather than the public complete carry tape.
// Rematerialize each block in forward order, then consume it in reverse order.
#[cfg(test)]
thread_local! {
    // Tests compare replay with the complete-tape fallback on the same long loop.
    static LOOP_CHECKPOINT_TEST_FULL_TAPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct TensorCarryCheckpoints<T = DynamicTensor> {
    carries: Vec<T>,
    block: usize,
    steps: usize,
}

impl<T: Clone> TensorCarryCheckpoints<T> {
    fn block_size(steps: usize) -> Option<usize> {
        #[cfg(test)]
        if LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.get()) {
            return None;
        }
        if steps == 0 {
            return None;
        }
        let root = steps.isqrt();
        let block = root + usize::from(root * root < steps);
        // Include both checkpoint and block entries conservatively. Short loops
        // retain the complete tape when separate workspace would not shrink it.
        steps
            .div_ceil(block)
            .checked_add(block)
            .filter(|&states| states < steps)?;
        Some(block)
    }

    fn forward(
        initial: T,
        steps: usize,
        block: usize,
        mut evaluate: impl FnMut(usize, T) -> Result<T, String>,
    ) -> Result<(T, Self), String> {
        let mut checkpoints = Vec::with_capacity(steps.div_ceil(block));
        let mut carry = initial;
        checkpoints.push(carry.clone());
        for offset in 0..steps {
            carry = evaluate(offset, carry)?;
            if (offset + 1).is_multiple_of(block) && offset + 1 < steps {
                checkpoints.push(carry.clone());
            }
        }
        Ok((
            carry,
            Self {
                carries: checkpoints,
                block,
                steps,
            },
        ))
    }

    fn pop_block(
        &mut self,
        mut evaluate: impl FnMut(usize, T) -> Result<T, String>,
    ) -> Result<Option<(usize, Vec<T>)>, String> {
        let Some(mut carry) = self.carries.pop() else {
            return Ok(None);
        };
        let start = self.carries.len() * self.block;
        let length = self.block.min(self.steps - start);
        let mut carries = Vec::with_capacity(length);
        carries.push(carry.clone());
        for offset in start..start + length - 1 {
            carry = evaluate(offset, carry)?;
            carries.push(carry.clone());
        }
        Ok(Some((start, carries)))
    }
}

impl TensorForiExecutionPlan {
    pub fn new(
        lower: usize,
        upper: usize,
        body: TensorExecutionPlan,
        carry_name: impl Into<String>,
        index_name: impl Into<String>,
    ) -> Result<Self, String> {
        if upper < lower {
            return Err(format!(
                "fori loop requires upper >= lower, got {upper} < {lower}"
            ));
        }
        let carry_name = carry_name.into();
        let index_name = index_name.into();
        if carry_name == index_name {
            return Err("fori loop carry and index captures must have different names".to_string());
        }
        let body = TensorRegion::new(body);
        let carry_shape = body
            .captures
            .get(&carry_name)
            .ok_or_else(|| format!("fori loop body does not capture carry {carry_name:?}"))?;
        if body.output_shape()? != *carry_shape {
            return Err(format!(
                "fori loop body output shape {:?} does not match carry shape {:?}",
                body.output_shape()?,
                carry_shape
            ));
        }
        check_loop_region_inputs(&body.plan, "fori loop")?;
        check_region_output_dtype(
            &body.plan,
            body.plan.output_node_id,
            &carry_name,
            "fori loop",
        )?;
        if let Some(index_shape) = body.captures.get(&index_name) {
            if !index_shape.is_empty() {
                return Err(format!(
                    "fori loop index capture must be scalar, got shape {index_shape:?}"
                ));
            }
        }
        let external_captures = body
            .captures
            .iter()
            .filter(|(name, _)| *name != &carry_name && *name != &index_name)
            .map(|(name, shape)| (name.clone(), shape.clone()))
            .collect::<BTreeMap<_, _>>();
        #[cfg(feature = "mlx")]
        let mlx_vjp = build_tensor_fori_mlx_vjp_plan(
            &body.plan,
            &body.captures,
            body.plan.output_node_id,
            &carry_name,
            &external_captures,
        );
        Ok(Self {
            lower,
            upper,
            body,
            carry_name,
            index_name,
            external_captures,
            #[cfg(feature = "mlx")]
            mlx_vjp,
        })
    }

    pub fn carry_shape(&self) -> Result<Vec<usize>, String> {
        self.body
            .captures
            .get(&self.carry_name)
            .cloned()
            .ok_or_else(|| "fori loop carry capture is missing".to_string())
    }

    pub fn external_captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.external_captures
    }

    pub fn evaluate(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.validate_external_inputs(external_inputs)?;
        if initial_carry.shape != self.carry_shape()? {
            return Err(format!(
                "fori initial carry shape {:?} does not match {:?}",
                initial_carry.shape,
                self.carry_shape()?
            ));
        }
        let mut carry = initial_carry;
        for index in self.lower..self.upper {
            let inputs = self.body_inputs(carry, index, external_inputs)?;
            carry = self.body.plan.evaluate(&inputs)?;
        }
        Ok(carry)
    }

    /// Executes the forward loop once and returns both the final carry and its
    /// fixed-bound carry tape for reverse-mode consumers.
    pub fn evaluate_with_tape(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, TensorForiTape), String> {
        self.validate_external_inputs(external_inputs)?;
        if initial_carry.shape != self.carry_shape()? {
            return Err(format!(
                "fori initial carry shape {:?} does not match {:?}",
                initial_carry.shape,
                self.carry_shape()?
            ));
        }
        let mut carries = Vec::with_capacity(self.upper - self.lower + 1);
        let mut carry = initial_carry;
        carries.push(carry.clone());
        for index in self.lower..self.upper {
            let inputs = self.body_inputs(carry.clone(), index, external_inputs)?;
            carry = self.body.plan.evaluate(&inputs)?;
            carries.push(carry.clone());
        }
        Ok((carry, TensorForiTape { carries }))
    }

    pub fn jvp(
        &self,
        initial_carry: DynamicTensor,
        initial_tangent: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        external_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.validate_external_inputs(external_inputs)?;
        self.validate_external_inputs(external_tangents)?;
        let carry_shape = self.carry_shape()?;
        if initial_carry.shape != carry_shape || initial_tangent.shape != carry_shape {
            return Err(format!(
                "fori carry and tangent must both have shape {carry_shape:?}"
            ));
        }
        let mut carry = initial_carry;
        let mut tangent = initial_tangent;
        for index in self.lower..self.upper {
            let inputs = self.body_inputs(carry, index, external_inputs)?;
            let tangents = self.body_tangents(tangent, external_tangents)?;
            let (next_carry, next_tangent) = self.body.plan.jvp(&inputs, &tangents)?;
            carry = next_carry;
            tangent = next_tangent;
        }
        Ok((carry, tangent))
    }

    pub fn value_and_vjp(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<
        (
            DynamicTensor,
            DynamicTensor,
            BTreeMap<String, DynamicTensor>,
        ),
        String,
    > {
        self.validate_external_inputs(external_inputs)?;
        let carry_shape = self.carry_shape()?;
        if initial_carry.shape != carry_shape || output_cotangent.shape != carry_shape {
            return Err(format!(
                "fori carry and output cotangent must both have shape {carry_shape:?}"
            ));
        }
        let steps = self.upper - self.lower;
        let mut checkpoints = None;
        let mut full_tape = None;
        let output = if let Some(block) = TensorCarryCheckpoints::<DynamicTensor>::block_size(steps)
        {
            let (output, tape) =
                TensorCarryCheckpoints::forward(initial_carry, steps, block, |offset, carry| {
                    self.body.plan.evaluate(&self.body_inputs(
                        carry,
                        self.lower + offset,
                        external_inputs,
                    )?)
                })?;
            checkpoints = Some(tape);
            output
        } else {
            let (output, tape) = self.evaluate_with_tape(initial_carry, external_inputs)?;
            full_tape = Some(tape);
            output
        };
        let mut carry_cotangent = output_cotangent;
        let mut external_gradients = self
            .external_captures
            .iter()
            .map(|(name, shape)| {
                DynamicTensor::filled(shape.clone(), 0.0).map(|value| (name.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        loop {
            let block = if let Some(checkpoints) = &mut checkpoints {
                checkpoints.pop_block(|offset, carry| {
                    self.body.plan.evaluate(&self.body_inputs(
                        carry,
                        self.lower + offset,
                        external_inputs,
                    )?)
                })?
            } else {
                full_tape.take().map(|mut tape| {
                    tape.carries.pop();
                    (0, tape.carries)
                })
            };
            let Some((start, carries)) = block else {
                break;
            };
            for (local_offset, carry) in carries.into_iter().enumerate().rev() {
                let offset = start + local_offset;
                let index = self.lower + offset;
                let (_, gradients) = self.body.plan.value_and_vjp(
                    &self.body_inputs(carry, index, external_inputs)?,
                    carry_cotangent,
                )?;
                carry_cotangent = gradients
                    .get(&self.carry_name)
                    .cloned()
                    .ok_or_else(|| "fori loop body did not return a carry gradient".to_string())?;
                for name in self.external_captures.keys() {
                    let contribution = gradients.get(name).ok_or_else(|| {
                        format!("fori loop body did not return a gradient for capture {name:?}")
                    })?;
                    let accumulated = external_gradients.get_mut(name).ok_or_else(|| {
                        format!("fori loop external gradient {name:?} is missing")
                    })?;
                    accumulated.add_assign(contribution)?;
                }
            }
        }
        Ok((output, carry_cotangent, external_gradients))
    }

    fn validate_external_inputs(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(), String> {
        for (name, shape) in &self.external_captures {
            let value = inputs
                .get(name)
                .ok_or_else(|| format!("missing fori loop external capture {name:?}"))?;
            if value.shape != *shape {
                return Err(format!(
                    "fori loop external capture {name:?} has shape {:?}, expected {:?}",
                    value.shape, shape
                ));
            }
        }
        Ok(())
    }

    fn body_inputs(
        &self,
        carry: DynamicTensor,
        index: usize,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        let index = DynamicTensor::new(vec![], vec![index as f64])?;
        let mut inputs = external_inputs.clone();
        inputs.insert(self.carry_name.clone(), carry);
        inputs.insert(self.index_name.clone(), index);
        Ok(inputs)
    }

    fn body_tangents(
        &self,
        carry_tangent: DynamicTensor,
        external_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        let mut tangents = external_tangents.clone();
        tangents.insert(self.carry_name.clone(), carry_tangent);
        tangents.insert(self.index_name.clone(), DynamicTensor::filled(vec![], 0.0)?);
        Ok(tangents)
    }
}

impl TensorWhileExecutionPlan {
    /// Builds a while loop from a predicate region and a body region over the
    /// same named inputs. Either region may ignore an input the other reads,
    /// so the external captures are the union of both regions' inputs.
    pub fn new(
        predicate: TensorExecutionPlan,
        body: TensorExecutionPlan,
        carry_name: impl Into<String>,
    ) -> Result<Self, String> {
        let carry_name = carry_name.into();
        let predicate = TensorRegion::new(predicate);
        let body = TensorRegion::new(body);
        let carry_shape = body
            .captures
            .get(&carry_name)
            .ok_or_else(|| format!("while loop body does not capture carry {carry_name:?}"))?;
        if body.output_shape()? != *carry_shape {
            return Err(format!(
                "while loop body output shape {:?} does not match carry shape {:?}",
                body.output_shape()?,
                carry_shape
            ));
        }
        check_loop_region_inputs(&body.plan, "while loop")?;
        check_loop_region_inputs(&predicate.plan, "while loop predicate")?;
        check_region_output_dtype(
            &body.plan,
            body.plan.output_node_id,
            &carry_name,
            "while loop",
        )?;
        if !predicate.output_shape()?.is_empty()
            || predicate.plan.output_dtype()? != TensorDType::Bool
        {
            return Err(format!(
                "while loop predicate must return a scalar bool, got shape {:?} and dtype {}",
                predicate.output_shape()?,
                predicate.plan.output_dtype()?
            ));
        }
        let mut external_captures = BTreeMap::new();
        for region in [&body, &predicate] {
            for (name, shape) in &region.captures {
                if let Some(other) = body.captures.get(name) {
                    if other != shape
                        || body.plan.input_dtype(name)? != region.plan.input_dtype(name)?
                    {
                        return Err(format!(
                            "while loop predicate and body capture {name:?} with different \
                             shapes or dtypes"
                        ));
                    }
                }
                if name != &carry_name {
                    external_captures.insert(name.clone(), shape.clone());
                }
            }
        }
        Ok(Self {
            predicate,
            body,
            carry_name,
            external_captures,
        })
    }

    pub fn carry_shape(&self) -> Result<Vec<usize>, String> {
        self.body
            .captures
            .get(&self.carry_name)
            .cloned()
            .ok_or_else(|| "while loop carry capture is missing".to_string())
    }

    pub fn carry_name(&self) -> &str {
        &self.carry_name
    }

    pub fn external_captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.external_captures
    }

    /// The region that declares `name`; the body wins when both read it.
    fn region_of(&self, name: &str) -> &TensorExecutionPlan {
        if self.body.captures.contains_key(name) {
            &self.body.plan
        } else {
            &self.predicate.plan
        }
    }

    pub fn predicate_plan(&self) -> &TensorExecutionPlan {
        &self.predicate.plan
    }

    pub fn body_plan(&self) -> &TensorExecutionPlan {
        &self.body.plan
    }

    fn region_inputs(
        &self,
        carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        for (name, shape) in &self.external_captures {
            let value = external_inputs
                .get(name)
                .ok_or_else(|| format!("missing while loop external capture {name:?}"))?;
            if value.shape != *shape {
                return Err(format!(
                    "while loop external capture {name:?} has shape {:?}, expected {:?}",
                    value.shape, shape
                ));
            }
        }
        if carry.shape != self.carry_shape()? {
            return Err(format!(
                "while loop carry shape {:?} does not match {:?}",
                carry.shape,
                self.carry_shape()?
            ));
        }
        let mut inputs = external_inputs.clone();
        inputs.insert(self.carry_name.clone(), carry);
        Ok(inputs)
    }

    fn keep_going(&self, inputs: &BTreeMap<String, DynamicTensor>) -> Result<bool, String> {
        tensor_scalar_predicate(&self.predicate.plan.evaluate(inputs)?)
    }

    pub fn evaluate(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        let mut inputs = self.region_inputs(initial_carry, external_inputs)?;
        while self.keep_going(&inputs)? {
            let next = self.body.plan.evaluate(&inputs)?;
            inputs.insert(self.carry_name.clone(), next);
        }
        inputs
            .remove(&self.carry_name)
            .ok_or_else(|| "while loop carry is missing".to_string())
    }

    /// Forward-mode derivative: the predicate only reads primal values, so
    /// the tangent follows the same iterations as the primal carry.
    pub fn jvp(
        &self,
        initial_carry: DynamicTensor,
        initial_tangent: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        external_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        if initial_tangent.shape != initial_carry.shape {
            return Err("while loop carry tangent shape does not match the carry".to_string());
        }
        let mut inputs = self.region_inputs(initial_carry, external_inputs)?;
        let mut tangents = external_tangents.clone();
        tangents.insert(self.carry_name.clone(), initial_tangent);
        while self.keep_going(&inputs)? {
            let (next, next_tangent) = self.body.plan.jvp(&inputs, &tangents)?;
            inputs.insert(self.carry_name.clone(), next);
            tangents.insert(self.carry_name.clone(), next_tangent);
        }
        let value = inputs
            .remove(&self.carry_name)
            .ok_or_else(|| "while loop carry is missing".to_string())?;
        let tangent = tangents
            .remove(&self.carry_name)
            .ok_or_else(|| "while loop carry tangent is missing".to_string())?;
        Ok((value, tangent))
    }
}

/// Reverse mode needs the per-iteration carries of a fixed trip count; a
/// traced predicate gives neither, matching JAX's `while_loop` restriction.
const WHILE_LOOP_REVERSE_MODE_ERROR: &str =
    "reverse-mode differentiation through while_loop is not supported because its trip count is \
     data dependent; use forward mode (jvp) or rewrite it as a bounded fori_loop whose \
     body masks finished iterations with where";

impl TensorForiVjpJvpExecutionPlan {
    pub fn new(loop_plan: TensorForiExecutionPlan, namespace: &str) -> Result<Self, String> {
        let body_plan = &loop_plan.body.plan;
        let body = body_plan.as_ir();
        let cotangent_name = format!("{namespace}_cotangent");
        let body_vjp = body.symbolic_vjp(body_plan.output_node_id, &cotangent_name)?;
        let mut tangent_names = BTreeMap::new();
        for (index, name) in loop_plan.body.captures.keys().enumerate() {
            if name == &loop_plan.index_name {
                continue;
            }
            let mut tangent_name = format!("{namespace}_tangent_{index}");
            while loop_plan.body.captures.contains_key(&tangent_name)
                || tangent_name == cotangent_name
                || tangent_names
                    .values()
                    .any(|candidate| candidate == &tangent_name)
            {
                tangent_name.push('_');
            }
            tangent_names.insert(name.clone(), tangent_name);
        }
        #[cfg(feature = "mlx")]
        let mlx_forward_jvp = {
            let transformed =
                body.symbolic_jvp_with_tangent_inputs(body_plan.output_node_id, &tangent_names)?;
            let (plan, output_node_ids) = transformed
                .graph
                .compile_cpu_many(&[transformed.value, transformed.tangent])?;
            TensorForiMlxForwardJvpPlan {
                plan,
                value_node_id: output_node_ids[0],
                tangent_node_id: output_node_ids[1],
                tangent_names: tangent_names.clone(),
            }
        };
        let mut jvp_tangent_names = tangent_names.clone();
        let mut cotangent_tangent_name = format!("{namespace}_cotangent_tangent");
        while loop_plan
            .body
            .captures
            .contains_key(&cotangent_tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &cotangent_tangent_name)
            || cotangent_tangent_name == cotangent_name
        {
            cotangent_tangent_name.push('_');
        }
        jvp_tangent_names.insert(cotangent_name.clone(), cotangent_tangent_name);
        let mut gradient_tangent_plans = BTreeMap::new();
        for name in std::iter::once(&loop_plan.carry_name).chain(loop_plan.external_captures.keys())
        {
            let gradient = body_vjp
                .gradients
                .get(name)
                .copied()
                .ok_or_else(|| format!("Fori body VJP has no gradient for capture {name:?}"))?;
            let transformed = body_vjp
                .graph
                .symbolic_jvp_with_tangent_inputs(gradient, &jvp_tangent_names)?;
            gradient_tangent_plans.insert(
                name.clone(),
                transformed.graph.compile_cpu(transformed.tangent)?,
            );
        }
        Ok(Self {
            loop_plan,
            namespace: namespace.to_string(),
            cotangent_name,
            tangent_names: jvp_tangent_names,
            gradient_tangent_plans,
            #[cfg(feature = "mlx")]
            mlx_forward_jvp,
        })
    }

    pub fn jvp(
        &self,
        initial_carry: DynamicTensor,
        initial_tangent: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        external_tangents: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
        output_cotangent_tangent: DynamicTensor,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        self.loop_plan.validate_external_inputs(external_inputs)?;
        self.loop_plan.validate_external_inputs(external_tangents)?;
        let carry_shape = self.loop_plan.carry_shape()?;
        if initial_carry.shape != carry_shape
            || initial_tangent.shape != carry_shape
            || output_cotangent.shape != carry_shape
            || output_cotangent_tangent.shape != carry_shape
        {
            return Err(
                "Fori VJP JVP carry or cotangent shape does not match carry shape".to_string(),
            );
        }
        let steps = self.loop_plan.upper - self.loop_plan.lower;
        let evaluate = |offset, (carry, tangent)| {
            let inputs = self.loop_plan.body_inputs(
                carry,
                self.loop_plan.lower + offset,
                external_inputs,
            )?;
            let tangents = self.loop_plan.body_tangents(tangent, external_tangents)?;
            self.loop_plan.body.plan.as_ir().jvp(
                self.loop_plan.body.plan.output_node_id,
                &inputs,
                &tangents,
            )
        };
        let mut checkpoints = None;
        let mut full_tape = None;
        let initial = (initial_carry, initial_tangent);
        if let Some(block) = TensorCarryCheckpoints::<DynamicTensor>::block_size(steps) {
            let (_, tape) = TensorCarryCheckpoints::forward(initial, steps, block, evaluate)?;
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
        let mut gradients = self
            .loop_plan
            .external_captures
            .iter()
            .map(|(name, shape)| {
                DynamicTensor::filled(shape.clone(), 0.0).map(|value| (name.clone(), value))
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
                let inputs = self.loop_plan.body_inputs(
                    carry,
                    self.loop_plan.lower + offset,
                    external_inputs,
                )?;
                let body_gradients = self
                    .loop_plan
                    .body
                    .plan
                    .value_and_vjp(&inputs, carry_cotangent.clone())?
                    .1;
                let mut jvp_inputs = inputs;
                jvp_inputs.insert(self.cotangent_name.clone(), carry_cotangent.clone());
                let mut jvp_tangents = BTreeMap::new();
                jvp_tangents.insert(
                    self.tangent_names
                        .get(&self.loop_plan.carry_name)
                        .cloned()
                        .ok_or_else(|| "Fori VJP JVP has no carry tangent input".to_string())?,
                    tangent,
                );
                for (name, tangent_name) in &self.tangent_names {
                    if name == &self.loop_plan.carry_name || name == &self.cotangent_name {
                        continue;
                    }
                    if let Some(tangent) = external_tangents.get(name) {
                        jvp_tangents.insert(tangent_name.clone(), tangent.clone());
                    }
                }
                let cotangent_tangent_name = self
                    .tangent_names
                    .get(&self.cotangent_name)
                    .ok_or_else(|| "Fori VJP JVP has no cotangent tangent input".to_string())?;
                jvp_tangents.insert(
                    cotangent_tangent_name.clone(),
                    carry_cotangent_tangent.clone(),
                );
                jvp_inputs.extend(jvp_tangents);
                let next_carry_cotangent_tangent = self
                    .gradient_tangent_plans
                    .get(&self.loop_plan.carry_name)
                    .ok_or_else(|| "Fori VJP JVP has no carry gradient plan".to_string())?
                    .evaluate(&jvp_inputs)?;
                for name in self.loop_plan.external_captures.keys() {
                    let contribution = self
                        .gradient_tangent_plans
                        .get(name)
                        .ok_or_else(|| format!("Fori VJP JVP has no gradient plan for {name:?}"))?
                        .evaluate(&jvp_inputs)?;
                    let accumulated = gradients
                        .get_mut(name)
                        .ok_or_else(|| format!("Fori VJP JVP gradient {name:?} is missing"))?;
                    accumulated.add_assign(&contribution)?;
                }
                carry_cotangent = body_gradients
                    .get(&self.loop_plan.carry_name)
                    .cloned()
                    .ok_or_else(|| "Fori body VJP has no carry gradient".to_string())?;
                carry_cotangent_tangent = next_carry_cotangent_tangent;
            }
        }
        gradients.insert(self.loop_plan.carry_name.clone(), carry_cotangent_tangent);
        Ok(gradients)
    }
}

impl TensorScanVjpJvpExecutionPlan {
    pub fn new(scan_plan: TensorScanExecutionPlan, namespace: &str) -> Result<Self, String> {
        let body_plan = &scan_plan.body.plan;
        let body = body_plan.as_ir();
        let carry_cotangent_name = format!("{namespace}_carry_cotangent");
        let output_cotangent_name = format!("{namespace}_output_cotangent");
        let mut tangent_names = BTreeMap::new();
        for (index, name) in scan_plan.body.captures.keys().enumerate() {
            if name == &scan_plan.index_name {
                continue;
            }
            let mut tangent_name = format!("{namespace}_tangent_{index}");
            while scan_plan.body.captures.contains_key(&tangent_name)
                || tangent_name == carry_cotangent_name
                || tangent_name == output_cotangent_name
                || tangent_names
                    .values()
                    .any(|candidate| candidate == &tangent_name)
            {
                tangent_name.push('_');
            }
            tangent_names.insert(name.clone(), tangent_name);
        }
        let mut carry_cotangent_tangent_name = format!("{namespace}_carry_cotangent_tangent");
        while scan_plan
            .body
            .captures
            .contains_key(&carry_cotangent_tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &carry_cotangent_tangent_name)
        {
            carry_cotangent_tangent_name.push('_');
        }
        let mut output_cotangent_tangent_name = format!("{namespace}_output_cotangent_tangent");
        while scan_plan
            .body
            .captures
            .contains_key(&output_cotangent_tangent_name)
            || tangent_names
                .values()
                .any(|candidate| candidate == &output_cotangent_tangent_name)
            || output_cotangent_tangent_name == carry_cotangent_tangent_name
        {
            output_cotangent_tangent_name.push('_');
        }
        tangent_names.insert(carry_cotangent_name.clone(), carry_cotangent_tangent_name);
        tangent_names.insert(output_cotangent_name.clone(), output_cotangent_tangent_name);
        let carry_vjp = body.symbolic_vjp(body_plan.output_node_ids[0], &carry_cotangent_name)?;
        let output_vjp = body.symbolic_vjp(body_plan.output_node_ids[1], &output_cotangent_name)?;
        let compile_gradient_tangents = |vjp: SymbolicVjp, active_cotangent: &str| {
            let active_tangent_names = tangent_names
                .iter()
                .filter(|(name, _)| {
                    scan_plan.body.captures.contains_key(*name) || name.as_str() == active_cotangent
                })
                .map(|(name, tangent_name)| (name.clone(), tangent_name.clone()))
                .collect::<BTreeMap<_, _>>();
            std::iter::once(&scan_plan.carry_name)
                .chain(scan_plan.external_captures.keys())
                .map(|name| {
                    let gradient = vjp.gradients.get(name).copied().ok_or_else(|| {
                        format!("Scan body VJP has no gradient for capture {name:?}")
                    })?;
                    let transformed = vjp
                        .graph
                        .symbolic_jvp_with_tangent_inputs(gradient, &active_tangent_names)?;
                    Ok((
                        name.clone(),
                        transformed.graph.compile_cpu(transformed.tangent)?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, String>>()
        };
        let carry_gradient_tangent_plans =
            compile_gradient_tangents(carry_vjp, &carry_cotangent_name)?;
        let output_gradient_tangent_plans =
            compile_gradient_tangents(output_vjp, &output_cotangent_name)?;
        #[cfg(feature = "mlx")]
        let mlx_forward_jvp = {
            let forward_tangent_names = tangent_names
                .iter()
                .filter(|(name, _)| scan_plan.body.captures.contains_key(*name))
                .map(|(name, tangent_name)| (name.clone(), tangent_name.clone()))
                .collect::<BTreeMap<_, _>>();
            let transformed = body.symbolic_jvp_with_tangent_inputs(
                body_plan.output_node_ids[0],
                &forward_tangent_names,
            )?;
            let (plan, output_node_ids) = transformed
                .graph
                .compile_cpu_many(&[transformed.value, transformed.tangent])?;
            TensorScanMlxForwardJvpPlan {
                plan,
                value_node_id: output_node_ids[0],
                tangent_node_id: output_node_ids[1],
                tangent_names: forward_tangent_names,
            }
        };
        Ok(Self {
            scan_plan,
            namespace: namespace.to_string(),
            carry_cotangent_name,
            output_cotangent_name,
            tangent_names,
            carry_gradient_tangent_plans,
            output_gradient_tangent_plans,
            #[cfg(feature = "mlx")]
            mlx_forward_jvp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jvp(
        &self,
        initial_carry: DynamicTensor,
        initial_tangent: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        external_tangents: &BTreeMap<String, DynamicTensor>,
        final_carry_cotangent: DynamicTensor,
        final_carry_cotangent_tangent: DynamicTensor,
        output_cotangent: DynamicTensor,
        output_cotangent_tangent: DynamicTensor,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        self.scan_plan
            .validate(initial_carry.clone(), external_inputs)?;
        self.scan_plan
            .validate(initial_tangent.clone(), external_tangents)?;
        let carry_shape = self.scan_plan.carry_shape()?;
        let output_shape = self.scan_plan.output_shape()?;
        if initial_tangent.shape != carry_shape
            || final_carry_cotangent.shape != carry_shape
            || final_carry_cotangent_tangent.shape != carry_shape
            || output_cotangent.shape != output_shape
            || output_cotangent_tangent.shape != output_shape
        {
            return Err(
                "Scan VJP JVP input shapes do not match the Scan result shapes".to_string(),
            );
        }
        let steps = self.scan_plan.upper - self.scan_plan.lower;
        let evaluate = |offset, (carry, tangent)| {
            let inputs =
                self.scan_plan
                    .inputs(carry, self.scan_plan.lower + offset, external_inputs)?;
            let tangents = self.scan_plan.tangents(tangent, external_tangents)?;
            self.scan_plan.body.plan.as_ir().jvp(
                self.scan_plan.body.plan.output_node_ids[0],
                &inputs,
                &tangents,
            )
        };
        let mut checkpoints = None;
        let mut full_tape = None;
        let initial = (initial_carry, initial_tangent);
        if let Some(block) = TensorCarryCheckpoints::<DynamicTensor>::block_size(steps) {
            let (_, tape) = TensorCarryCheckpoints::forward(initial, steps, block, evaluate)?;
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

        let output_step_shape = self.scan_plan.body.output_shapes()[1].clone();
        let mut carry_cotangent = final_carry_cotangent;
        let mut carry_cotangent_tangent = final_carry_cotangent_tangent;
        let mut gradients = self
            .scan_plan
            .external_captures
            .iter()
            .map(|(name, shape)| {
                DynamicTensor::filled(shape.clone(), 0.0).map(|value| (name.clone(), value))
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
                let inputs =
                    self.scan_plan
                        .inputs(carry, self.scan_plan.lower + offset, external_inputs)?;
                let output_step_cotangent = output_cotangent
                    .slice_axis(0, offset, 1)?
                    .reshape(output_step_shape.clone())?;
                let output_step_cotangent_tangent = output_cotangent_tangent
                    .slice_axis(0, offset, 1)?
                    .reshape(output_step_shape.clone())?;
                let (_, primal_gradients) = self.scan_plan.body.plan.value_and_vjp_many(
                    &inputs,
                    vec![carry_cotangent.clone(), output_step_cotangent.clone()],
                )?;
                let mut jvp_inputs = inputs;
                jvp_inputs.insert(self.carry_cotangent_name.clone(), carry_cotangent.clone());
                jvp_inputs.insert(self.output_cotangent_name.clone(), output_step_cotangent);
                let mut jvp_tangents = BTreeMap::new();
                jvp_tangents.insert(
                    self.tangent_names
                        .get(&self.scan_plan.carry_name)
                        .cloned()
                        .ok_or_else(|| "Scan VJP JVP has no carry tangent input".to_string())?,
                    tangent,
                );
                for name in self.scan_plan.external_captures.keys() {
                    if let Some(tangent) = external_tangents.get(name) {
                        let tangent_name = self.tangent_names.get(name).ok_or_else(|| {
                            format!("Scan VJP JVP has no tangent input for {name:?}")
                        })?;
                        jvp_tangents.insert(tangent_name.clone(), tangent.clone());
                    }
                }
                jvp_tangents.insert(
                    self.tangent_names
                        .get(&self.carry_cotangent_name)
                        .cloned()
                        .ok_or_else(|| {
                            "Scan VJP JVP has no carry cotangent tangent input".to_string()
                        })?,
                    carry_cotangent_tangent.clone(),
                );
                jvp_tangents.insert(
                    self.tangent_names
                        .get(&self.output_cotangent_name)
                        .cloned()
                        .ok_or_else(|| {
                            "Scan VJP JVP has no output cotangent tangent input".to_string()
                        })?,
                    output_step_cotangent_tangent,
                );
                jvp_inputs.extend(jvp_tangents);
                let directional_gradient = |name: &str| -> Result<DynamicTensor, String> {
                    let carry_term = self
                        .carry_gradient_tangent_plans
                        .get(name)
                        .ok_or_else(|| {
                            format!("Scan VJP JVP has no carry gradient plan for {name:?}")
                        })?
                        .evaluate(&jvp_inputs)?;
                    let output_term = self
                        .output_gradient_tangent_plans
                        .get(name)
                        .ok_or_else(|| {
                            format!("Scan VJP JVP has no output gradient plan for {name:?}")
                        })?
                        .evaluate(&jvp_inputs)?;
                    carry_term.add(&output_term)
                };
                carry_cotangent_tangent = directional_gradient(&self.scan_plan.carry_name)?;
                for name in self.scan_plan.external_captures.keys() {
                    let contribution = directional_gradient(name)?;
                    let accumulated = gradients
                        .get_mut(name)
                        .ok_or_else(|| format!("Scan VJP JVP gradient {name:?} is missing"))?;
                    accumulated.add_assign(&contribution)?;
                }
                carry_cotangent = primal_gradients
                    .get(&self.scan_plan.carry_name)
                    .cloned()
                    .ok_or_else(|| "Scan body VJP has no carry gradient".to_string())?;
            }
        }
        gradients.insert(self.scan_plan.carry_name.clone(), carry_cotangent_tangent);
        Ok(gradients)
    }
}

impl TensorScanExecutionPlan {
    pub fn new(
        lower: usize,
        upper: usize,
        body: TensorExecutionPlan,
        carry_name: impl Into<String>,
        index_name: impl Into<String>,
    ) -> Result<Self, String> {
        if upper < lower {
            return Err(format!(
                "scan requires upper >= lower, got {upper} < {lower}"
            ));
        }
        let carry_name = carry_name.into();
        let index_name = index_name.into();
        if carry_name == index_name {
            return Err("scan carry and index captures must differ".to_string());
        }
        let body = TensorMultiRegion::new(body)?;
        if body.output_shapes().len() != 2 {
            return Err("scan body must return exactly carry and output".to_string());
        }
        let carry_shape = body
            .captures()
            .get(&carry_name)
            .ok_or_else(|| format!("scan body does not capture carry {carry_name:?}"))?;
        if body.output_shapes()[0] != *carry_shape {
            return Err("scan body next carry shape does not match carry shape".to_string());
        }
        check_loop_region_inputs(&body.plan, "scan")?;
        check_region_output_dtype(
            &body.plan,
            body.plan.output_node_ids[0],
            &carry_name,
            "scan",
        )?;
        let output_dtype = body.plan.node_dtype(body.plan.output_node_ids[1])?;
        if !output_dtype.is_floating() {
            return Err(format!(
                "scan outputs must have a floating dtype, got {output_dtype}; \
                 convert bool values explicitly with astype"
            ));
        }
        if let Some(index_shape) = body.captures().get(&index_name) {
            if !index_shape.is_empty() {
                return Err("scan index capture must be scalar".to_string());
            }
        }
        let external_captures = body
            .captures()
            .iter()
            .filter(|(name, _)| *name != &carry_name && *name != &index_name)
            .map(|(name, shape)| (name.clone(), shape.clone()))
            .collect::<BTreeMap<_, _>>();
        #[cfg(feature = "mlx")]
        let mlx_vjp = match (
            build_tensor_fori_mlx_vjp_plan(
                &body.plan,
                body.captures(),
                body.plan.output_node_ids[0],
                &carry_name,
                &external_captures,
            ),
            build_tensor_fori_mlx_vjp_plan(
                &body.plan,
                body.captures(),
                body.plan.output_node_ids[1],
                &carry_name,
                &external_captures,
            ),
        ) {
            (Some(carry), Some(output)) => Some(TensorScanMlxVjpPlan { carry, output }),
            _ => None,
        };
        Ok(Self {
            lower,
            upper,
            body,
            carry_name,
            index_name,
            external_captures,
            #[cfg(feature = "mlx")]
            mlx_vjp,
        })
    }

    pub fn carry_shape(&self) -> Result<Vec<usize>, String> {
        self.body
            .captures()
            .get(&self.carry_name)
            .cloned()
            .ok_or_else(|| "scan carry capture is missing".to_string())
    }

    pub fn output_shape(&self) -> Result<Vec<usize>, String> {
        let mut shape = vec![self.upper - self.lower];
        shape.extend_from_slice(
            self.body
                .output_shapes()
                .get(1)
                .ok_or_else(|| "scan body output is missing".to_string())?,
        );
        Ok(shape)
    }

    pub fn external_captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.external_captures
    }

    pub fn evaluate(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.validate(initial_carry.clone(), external_inputs)?;
        let mut carry = initial_carry;
        let mut outputs = Vec::with_capacity(self.upper - self.lower);
        for index in self.lower..self.upper {
            let mut values = self
                .body
                .evaluate(&self.inputs(carry, index, external_inputs)?)?
                .into_iter();
            carry = values
                .next()
                .ok_or_else(|| "scan body has no carry".to_string())?;
            outputs.push(
                values
                    .next()
                    .ok_or_else(|| "scan body has no output".to_string())?,
            );
        }
        Ok((carry, stack_scan_outputs(outputs)?))
    }

    pub fn evaluate_with_tape(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor, TensorScanTape), String> {
        self.validate(initial_carry.clone(), external_inputs)?;
        let mut carry = initial_carry;
        let mut outputs = Vec::with_capacity(self.upper - self.lower);
        let mut carries = vec![carry.clone()];
        for index in self.lower..self.upper {
            let values = self
                .body
                .evaluate(&self.inputs(carry, index, external_inputs)?)?;
            carry = values[0].clone();
            outputs.push(values[1].clone());
            carries.push(carry.clone());
        }
        Ok((
            carry,
            stack_scan_outputs(outputs)?,
            TensorScanTape { carries },
        ))
    }

    pub fn jvp(
        &self,
        initial_carry: DynamicTensor,
        initial_tangent: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        external_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<TensorScanJvpResult, String> {
        self.validate(initial_carry.clone(), external_inputs)?;
        self.validate(initial_tangent.clone(), external_tangents)?;
        let mut carry = initial_carry;
        let mut tangent = initial_tangent;
        let mut outputs = Vec::with_capacity(self.upper - self.lower);
        let mut output_tangents = Vec::with_capacity(self.upper - self.lower);
        for index in self.lower..self.upper {
            let inputs = self.inputs(carry, index, external_inputs)?;
            let tangents = self.tangents(tangent, external_tangents)?;
            let body = self.body.plan.as_ir();
            // Both outputs share the primal and numeric tangent traversal, including nested scans.
            let mut results = body
                .jvp_many(&self.body.plan.output_node_ids, &inputs, &tangents)?
                .into_iter();
            let (next_carry, next_tangent) = results
                .next()
                .ok_or_else(|| "scan body has no carry".to_string())?;
            let (output, output_tangent) = results
                .next()
                .ok_or_else(|| "scan body has no output".to_string())?;
            carry = next_carry;
            tangent = next_tangent;
            outputs.push(output);
            output_tangents.push(output_tangent);
        }
        Ok((
            (carry, stack_scan_outputs(outputs)?),
            (tangent, stack_scan_outputs(output_tangents)?),
        ))
    }

    pub fn value_and_vjp(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        final_carry_cotangent: DynamicTensor,
        output_cotangent: DynamicTensor,
    ) -> Result<
        (
            DynamicTensor,
            DynamicTensor,
            DynamicTensor,
            BTreeMap<String, DynamicTensor>,
        ),
        String,
    > {
        let (carry, outputs, gradient, captures) = self.value_and_vjp_internal(
            initial_carry,
            external_inputs,
            final_carry_cotangent,
            output_cotangent,
            true,
        )?;
        Ok((
            carry,
            outputs.expect("requested Scan outputs"),
            gradient,
            captures,
        ))
    }

    // Gradient-only region consumers need no stacked primal scan output.
    fn vjp(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        final_carry_cotangent: DynamicTensor,
        output_cotangent: DynamicTensor,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        let (_, _, gradient, captures) = self.value_and_vjp_internal(
            initial_carry,
            external_inputs,
            final_carry_cotangent,
            output_cotangent,
            false,
        )?;
        Ok((gradient, captures))
    }

    fn value_and_vjp_internal(
        &self,
        initial_carry: DynamicTensor,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        final_carry_cotangent: DynamicTensor,
        output_cotangent: DynamicTensor,
        keep_outputs: bool,
    ) -> Result<TensorScanCheckpointVjpResult, String> {
        self.validate(initial_carry.clone(), external_inputs)?;
        let steps = self.upper - self.lower;
        if steps == 0 {
            return Err("scan requires at least one output".to_string());
        }
        let mut outputs = if keep_outputs {
            Vec::with_capacity(steps)
        } else {
            Vec::new()
        };
        let mut checkpoints = None;
        let mut full_tape = None;
        let final_carry = if let Some(block) =
            TensorCarryCheckpoints::<DynamicTensor>::block_size(steps)
        {
            let (carry, tape) =
                TensorCarryCheckpoints::forward(initial_carry, steps, block, |offset, carry| {
                    let values = self.body.evaluate(&self.inputs(
                        carry,
                        self.lower + offset,
                        external_inputs,
                    )?)?;
                    if keep_outputs {
                        outputs.push(values[1].clone());
                    }
                    Ok(values[0].clone())
                })?;
            checkpoints = Some(tape);
            carry
        } else {
            let mut carry = initial_carry;
            let mut carries = vec![carry.clone()];
            for index in self.lower..self.upper {
                let values = self
                    .body
                    .evaluate(&self.inputs(carry, index, external_inputs)?)?;
                carry = values[0].clone();
                if keep_outputs {
                    outputs.push(values[1].clone());
                }
                carries.push(carry.clone());
            }
            full_tape = Some(carries);
            carry
        };
        let outputs = if keep_outputs {
            Some(stack_scan_outputs(outputs)?)
        } else {
            None
        };
        if final_carry_cotangent.shape != final_carry.shape
            || output_cotangent.shape != self.output_shape()?
        {
            return Err("scan cotangent shapes do not match outputs".to_string());
        }
        let mut carry_cotangent = final_carry_cotangent;
        let mut external_gradients = self
            .external_captures
            .iter()
            .map(|(name, shape)| {
                DynamicTensor::filled(shape.clone(), 0.0).map(|value| (name.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let output_shape = self.body.output_shapes()[1].clone();
        loop {
            let block = if let Some(checkpoints) = &mut checkpoints {
                checkpoints.pop_block(|offset, carry| {
                    let values = self.body.evaluate(&self.inputs(
                        carry,
                        self.lower + offset,
                        external_inputs,
                    )?)?;
                    Ok(values[0].clone())
                })?
            } else {
                full_tape.take().map(|mut carries| {
                    carries.pop();
                    (0, carries)
                })
            };
            let Some((start, carries)) = block else {
                break;
            };
            for (local_offset, carry) in carries.into_iter().enumerate().rev() {
                let offset = start + local_offset;
                let output_gradient = output_cotangent
                    .slice_axis(0, offset, 1)?
                    .reshape(output_shape.clone())?;
                let inputs = self.inputs(carry, self.lower + offset, external_inputs)?;
                let (_, gradients) = self
                    .body
                    .plan
                    .value_and_vjp_many(&inputs, vec![carry_cotangent, output_gradient])?;
                carry_cotangent = gradients
                    .get(&self.carry_name)
                    .cloned()
                    .ok_or_else(|| "scan body has no carry gradient".to_string())?;
                for name in self.external_captures.keys() {
                    let contribution = gradients
                        .get(name)
                        .ok_or_else(|| format!("scan body has no gradient for {name:?}"))?;
                    let accumulated = external_gradients
                        .get_mut(name)
                        .ok_or_else(|| format!("scan gradient {name:?} is missing"))?;
                    accumulated.add_assign(contribution)?;
                }
            }
        }
        Ok((final_carry, outputs, carry_cotangent, external_gradients))
    }

    fn validate(
        &self,
        carry: DynamicTensor,
        external: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(), String> {
        if carry.shape
            != *self
                .body
                .captures()
                .get(&self.carry_name)
                .ok_or_else(|| "scan carry capture is missing".to_string())?
        {
            return Err("scan initial carry shape does not match body carry shape".to_string());
        }
        for (name, shape) in &self.external_captures {
            if external.get(name).map(|value| &value.shape) != Some(shape) {
                return Err(format!(
                    "scan external capture {name:?} has an invalid shape"
                ));
            }
        }
        Ok(())
    }

    fn inputs(
        &self,
        carry: DynamicTensor,
        index: usize,
        external: &BTreeMap<String, DynamicTensor>,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        let mut inputs = external.clone();
        inputs.insert(self.carry_name.clone(), carry);
        inputs.insert(
            self.index_name.clone(),
            DynamicTensor::new(vec![], vec![index as f64])?,
        );
        Ok(inputs)
    }

    fn tangents(
        &self,
        carry_tangent: DynamicTensor,
        external_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        let mut tangents = external_tangents.clone();
        tangents.insert(self.carry_name.clone(), carry_tangent);
        tangents.insert(self.index_name.clone(), DynamicTensor::filled(vec![], 0.0)?);
        Ok(tangents)
    }
}

fn stack_scan_outputs(outputs: Vec<DynamicTensor>) -> Result<DynamicTensor, String> {
    let first = outputs
        .first()
        .ok_or_else(|| "scan requires at least one output".to_string())?;
    let mut shape = vec![1];
    shape.extend_from_slice(&first.shape);
    let reshaped = outputs
        .iter()
        .map(|value| value.reshape(shape.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    DynamicTensor::concat(&reshaped.iter().collect::<Vec<_>>(), 0)
}

impl TensorForiMultiExecutionPlan {
    pub fn new(
        lower: usize,
        upper: usize,
        body: TensorExecutionPlan,
        carry_names: Vec<String>,
        index_name: impl Into<String>,
    ) -> Result<Self, String> {
        if upper < lower {
            return Err(format!(
                "multi-carry fori loop requires upper >= lower, got {upper} < {lower}"
            ));
        }
        if carry_names.is_empty() {
            return Err("multi-carry fori loop requires at least one carry".to_string());
        }
        let index_name = index_name.into();
        let mut names = BTreeSet::new();
        for name in &carry_names {
            if name == &index_name || !names.insert(name.as_str()) {
                return Err(
                    "multi-carry fori names must be distinct from the index and each other"
                        .to_string(),
                );
            }
        }
        let body = TensorMultiRegion::new(body)?;
        if body.output_shapes().len() != carry_names.len() {
            return Err(format!(
                "multi-carry fori body has {} outputs, expected {}",
                body.output_shapes().len(),
                carry_names.len()
            ));
        }
        for (name, output_shape) in carry_names.iter().zip(body.output_shapes()) {
            let carry_shape = body
                .captures()
                .get(name)
                .ok_or_else(|| format!("multi-carry fori body does not capture carry {name:?}"))?;
            if carry_shape != output_shape {
                return Err(format!(
                    "multi-carry fori output for {name:?} has shape {output_shape:?}, expected {carry_shape:?}"
                ));
            }
        }
        check_loop_region_inputs(&body.plan, "multi-carry fori")?;
        for (name, output) in carry_names.iter().zip(&body.plan.output_node_ids) {
            check_region_output_dtype(&body.plan, *output, name, "multi-carry fori")?;
        }
        let index_shape = body.captures().get(&index_name).ok_or_else(|| {
            format!("multi-carry fori body does not capture index {index_name:?}")
        })?;
        if !index_shape.is_empty() {
            return Err(format!(
                "multi-carry fori index capture must be scalar, got shape {index_shape:?}"
            ));
        }
        let external_captures = body
            .captures()
            .iter()
            .filter(|(name, _)| *name != &index_name && !carry_names.contains(*name))
            .map(|(name, shape)| (name.clone(), shape.clone()))
            .collect();
        Ok(Self {
            lower,
            upper,
            body,
            carry_names,
            index_name,
            external_captures,
        })
    }

    pub fn carry_names(&self) -> &[String] {
        &self.carry_names
    }

    pub fn external_captures(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.external_captures
    }

    pub fn evaluate(
        &self,
        initial_carries: Vec<DynamicTensor>,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        if initial_carries.len() != self.carry_names.len() {
            return Err(format!(
                "multi-carry fori received {} carries, expected {}",
                initial_carries.len(),
                self.carry_names.len()
            ));
        }
        self.validate_external_inputs(external_inputs)?;
        for (carry, name) in initial_carries.iter().zip(&self.carry_names) {
            let shape = self
                .body
                .captures()
                .get(name)
                .ok_or_else(|| format!("multi-carry fori carry {name:?} is missing"))?;
            if &carry.shape != shape {
                return Err(format!(
                    "multi-carry fori initial carry {name:?} has shape {:?}, expected {shape:?}",
                    carry.shape
                ));
            }
        }
        let mut carries = initial_carries;
        for index in self.lower..self.upper {
            let mut inputs = external_inputs.clone();
            inputs.insert(
                self.index_name.clone(),
                DynamicTensor::new(vec![], vec![index as f64])?,
            );
            for (name, carry) in self.carry_names.iter().zip(&carries) {
                inputs.insert(name.clone(), carry.clone());
            }
            carries = self.body.evaluate(&inputs)?;
        }
        Ok(carries)
    }

    /// Executes a multi-carry loop and reverses all body outputs together.
    ///
    /// This is the reverse-loop primitive needed by symbolic `Fori` VJP. The
    /// body plan receives one joint reverse pass per iteration rather than one
    /// pass per carry, preserving shared-subgraph work.
    pub fn value_and_vjp(
        &self,
        initial_carries: Vec<DynamicTensor>,
        external_inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangents: Vec<DynamicTensor>,
    ) -> Result<TensorForiMultiVjpResult, String> {
        if output_cotangents.len() != self.carry_names.len() {
            return Err(format!(
                "multi-carry fori received {} output cotangents, expected {}",
                output_cotangents.len(),
                self.carry_names.len()
            ));
        }
        if initial_carries.len() != self.carry_names.len() {
            return Err(format!(
                "multi-carry fori received {} carries, expected {}",
                initial_carries.len(),
                self.carry_names.len()
            ));
        }
        self.validate_external_inputs(external_inputs)?;
        for ((carry, cotangent), name) in initial_carries
            .iter()
            .zip(&output_cotangents)
            .zip(&self.carry_names)
        {
            let shape = self
                .body
                .captures()
                .get(name)
                .ok_or_else(|| format!("multi-carry fori carry {name:?} is missing"))?;
            if &carry.shape != shape || &cotangent.shape != shape {
                return Err(format!(
                    "multi-carry fori carry and cotangent for {name:?} must have shape {shape:?}"
                ));
            }
        }

        let mut tape = Vec::with_capacity(self.upper - self.lower + 1);
        let mut carries = initial_carries;
        tape.push(carries.clone());
        for index in self.lower..self.upper {
            carries = self
                .body
                .evaluate(&self.body_inputs(carries, index, external_inputs)?)?;
            tape.push(carries.clone());
        }
        let outputs = carries;
        let mut carry_cotangents = output_cotangents;
        let mut external_gradients = self
            .external_captures
            .iter()
            .map(|(name, shape)| {
                DynamicTensor::filled(shape.clone(), 0.0).map(|value| (name.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        for offset in (0..self.upper - self.lower).rev() {
            let index = self.lower + offset;
            let (_, gradients) = self.body.plan.value_and_vjp_many(
                &self.body_inputs(tape[offset].clone(), index, external_inputs)?,
                carry_cotangents,
            )?;
            carry_cotangents = self
                .carry_names
                .iter()
                .map(|name| {
                    gradients
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("multi-carry body has no gradient for {name:?}"))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for name in self.external_captures.keys() {
                let gradient = gradients.get(name).ok_or_else(|| {
                    format!("multi-carry body has no gradient for external capture {name:?}")
                })?;
                let slot = external_gradients.get_mut(name).ok_or_else(|| {
                    format!("multi-carry external gradient slot {name:?} is missing")
                })?;
                *slot = slot.add(gradient)?;
            }
        }
        Ok((outputs, carry_cotangents, external_gradients))
    }

    fn body_inputs(
        &self,
        carries: Vec<DynamicTensor>,
        index: usize,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        let mut inputs = external_inputs.clone();
        inputs.insert(
            self.index_name.clone(),
            DynamicTensor::new(vec![], vec![index as f64])?,
        );
        for (name, carry) in self.carry_names.iter().zip(carries) {
            inputs.insert(name.clone(), carry);
        }
        Ok(inputs)
    }

    fn validate_external_inputs(
        &self,
        external_inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(), String> {
        if external_inputs.len() != self.external_captures.len() {
            return Err(
                "multi-carry fori external inputs must match body captures exactly".to_string(),
            );
        }
        for (name, expected_shape) in &self.external_captures {
            let value = external_inputs
                .get(name)
                .ok_or_else(|| format!("missing multi-carry fori external input {name:?}"))?;
            if &value.shape != expected_shape {
                return Err(format!(
                    "multi-carry fori external input {name:?} has shape {:?}, expected {expected_shape:?}",
                    value.shape
                ));
            }
        }
        Ok(())
    }
}

impl TensorForiTape {
    /// Number of stored carries, including the initial and final states.
    pub fn len(&self) -> usize {
        self.carries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.carries.is_empty()
    }

    /// Returns the carry after `iteration` loop steps.
    pub fn carry_at(&self, iteration: usize) -> Result<&DynamicTensor, String> {
        self.carries
            .get(iteration)
            .ok_or_else(|| format!("fori tape has no carry at iteration {iteration}"))
    }
}

impl TensorScanTape {
    /// Number of stored carries, including the initial and final states.
    pub fn len(&self) -> usize {
        self.carries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.carries.is_empty()
    }

    /// Returns the carry after `iteration` scan steps.
    pub fn carry_at(&self, iteration: usize) -> Result<&DynamicTensor, String> {
        self.carries
            .get(iteration)
            .ok_or_else(|| format!("scan tape has no carry at iteration {iteration}"))
    }
}

impl TensorExecutionPlan {
    /// Checks CUDA device-loop forms before driver/NVRTC initialization.
    pub fn validate_cuda(&self) -> Result<(), (String, String)> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            cuda::validate_cuda_plan(self)
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            Err((
                "cuda".into(),
                "CUDA target is unavailable in this build".into(),
            ))
        }
    }

    /// Checks the `precision="float64"` CUDA lowering, which gives the plan
    /// `f64` device buffers: no node, nested regions included, may be `float32`.
    pub fn validate_cuda_float64(&self) -> Result<(), (String, String)> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            cuda::validate_cuda_float64_plan(self)
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            Err((
                "cuda".into(),
                "CUDA target is unavailable in this build".into(),
            ))
        }
    }

    /// Whether any node of this plan (outside nested regions) is `float64`.
    pub fn has_float64_nodes(&self) -> bool {
        self.nodes.iter().any(|node| node.dtype == TensorDType::F64)
    }

    /// Checks lazy MLX lowering without allocating arrays or executing a branch.
    /// The compatibility helpers keep their historical execution-time errors.
    pub fn validate_mlx(&self) -> Result<(), (String, String)> {
        for node in self.nodes.iter() {
            if node
                .shape
                .iter()
                .any(|extent| i32::try_from(*extent).is_err())
            {
                return Err(("shape".into(), "MLX shape extent exceeds i32".into()));
            }
            match &node.op {
                TensorOp::ScalarConstant { value } if !value.is_finite() => {
                    return Err((
                        "constant".into(),
                        "MLX backend does not support non-finite constants".into(),
                    ))
                }
                TensorOp::Cond { branches, .. } => {
                    branches.on_true.plan.validate_mlx()?;
                    branches.on_false.plan.validate_mlx()?;
                }
                TensorOp::While { loop_plan, .. } => {
                    loop_plan.predicate.plan.validate_mlx()?;
                    loop_plan.body.plan.validate_mlx()?;
                }
                TensorOp::Fori { loop_plan, .. }
                | TensorOp::ForiJvp { loop_plan, .. }
                | TensorOp::ForiVjp { loop_plan, .. } => loop_plan.body.plan.validate_mlx()?,
                TensorOp::ForiVjpJvp { plan, .. } => {
                    plan.loop_plan.body.plan.validate_mlx()?;
                    for gradient in plan.gradient_tangent_plans.values() {
                        gradient.validate_mlx()?;
                    }
                }
                TensorOp::Scan { scan_plan, .. } | TensorOp::ScanVjp { scan_plan, .. } => {
                    scan_plan.body.plan.validate_mlx()?
                }
                TensorOp::ScanVjpJvp { plan, .. } => {
                    plan.scan_plan.body.plan.validate_mlx()?;
                    for gradient in plan
                        .carry_gradient_tangent_plans
                        .values()
                        .chain(plan.output_gradient_tangent_plans.values())
                    {
                        gradient.validate_mlx()?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn output_node_id(&self) -> TensorNodeId {
        self.output_node_id
    }

    /// Every output preserved by a multi-output compilation.
    pub fn output_node_ids(&self) -> &[TensorNodeId] {
        &self.output_node_ids
    }

    pub fn uses_fused_elementwise_kernel(&self) -> bool {
        self.fused_elementwise_output
    }

    pub fn output_shape(&self) -> Result<Vec<usize>, String> {
        self.nodes
            .get(self.output_node_id)
            .map(|node| node.shape.clone())
            .ok_or_else(|| "execution plan output node does not exist".to_string())
    }

    pub fn input_shape(&self, name: &str) -> Result<Vec<usize>, String> {
        self.input_nodes
            .get(name)
            .map(|id| self.nodes[*id].shape.clone())
            .ok_or_else(|| format!("execution plan input {name:?} does not exist"))
    }

    pub fn input_dtype(&self, name: &str) -> Result<TensorDType, String> {
        self.input_nodes
            .get(name)
            .map(|id| self.nodes[*id].dtype)
            .ok_or_else(|| format!("execution plan input {name:?} does not exist"))
    }

    pub fn node_dtype(&self, id: TensorNodeId) -> Result<TensorDType, String> {
        self.nodes
            .get(id)
            .map(|node| node.dtype)
            .ok_or_else(|| format!("execution plan node {id} does not exist"))
    }

    pub fn output_dtype(&self) -> Result<TensorDType, String> {
        self.node_dtype(self.output_node_id)
    }

    pub fn lower_text(&self) -> String {
        self.as_ir().lower_text()
    }

    pub fn kernel_ir(&self) -> TensorKernelProgram {
        TensorKernelProgram {
            nodes: self
                .nodes
                .iter()
                .enumerate()
                .map(|(id, node)| TensorKernelNode {
                    id,
                    op: tensor_op_name(&node.op).to_string(),
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    layout: if node.shape.is_empty() {
                        "scalar".to_string()
                    } else {
                        "row_major_contiguous".to_string()
                    },
                    placement: TensorPlacement::Unplaced,
                    effect: if matches!(node.op, TensorOp::Input { .. }) {
                        "input".to_string()
                    } else {
                        "pure".to_string()
                    },
                    alias_of: match node.op {
                        TensorOp::Reshape { input } => Some(input),
                        _ => None,
                    },
                    inputs: tensor_op_inputs(&node.op),
                    name: match &node.op {
                        TensorOp::Input { name } => Some(name.clone()),
                        _ => None,
                    },
                })
                .collect(),
            output_node_id: self.output_node_id,
        }
    }

    /// Applies typed placement metadata to selected nodes of a frozen plan.
    ///
    /// This changes compiler metadata only. Backend execution still rejects
    /// distributed placement until a sharded backend lowering is available.
    pub fn kernel_ir_with_placements(
        &self,
        placements: &BTreeMap<TensorNodeId, TensorPlacement>,
    ) -> Result<TensorKernelProgram, String> {
        let mut program = self.kernel_ir();
        for (node_id, placement) in placements {
            let node = program
                .nodes
                .get_mut(*node_id)
                .ok_or_else(|| format!("kernel placement node {node_id} does not exist"))?;
            node.placement = placement.clone();
        }
        program.validate()?;
        Ok(program)
    }

    /// Propagates explicit input placements through operations with statically
    /// defined local semantics.
    ///
    /// Unlike [`Self::kernel_ir_with_placements`], this rejects operations that
    /// would require a collective or a redistribution. It is therefore a
    /// backend-neutral compiler check, not a request to insert implicit copies.
    pub fn kernel_ir_with_placement_propagation(
        &self,
        placements: &BTreeMap<TensorNodeId, TensorPlacement>,
    ) -> Result<TensorKernelProgram, String> {
        self.kernel_ir_with_placement_mode(placements, TensorPlacementCollectiveMode::Reject)
            .map(|(program, _)| program)
    }

    /// Builds a backend-neutral local program plus every all-reduce required
    /// by its placement contract.
    ///
    /// This does not execute a collective or select a transport library. The
    /// returned schedule is the input contract for a future NCCL lowering.
    pub fn sharding_plan(
        &self,
        placements: &BTreeMap<TensorNodeId, TensorPlacement>,
    ) -> Result<TensorShardingPlan, String> {
        let (program, all_reduces) =
            self.kernel_ir_with_placement_mode(placements, TensorPlacementCollectiveMode::Plan)?;
        let plan = TensorShardingPlan {
            program,
            all_reduces,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Lowers a sharding schedule derived from this plan to the first CUDA
    /// data-parallel subset: one 1-D CUDA mesh whose devices are
    /// `device_ordinals` in rank order, inputs sharded only on axis zero with
    /// one shared batch extent, and all-reduces only on unconsumed retained
    /// outputs. Schedules outside that subset would need a mid-graph
    /// collective, a gather, or a redistribution, so they are rejected.
    pub fn cuda_data_parallel_program(
        &self,
        sharding: &TensorShardingPlan,
        device_ordinals: &[usize],
    ) -> Result<TensorDataParallelProgram, String> {
        // Re-deriving the schedule proves the node ids, placements, and
        // collectives all belong to this plan and obey propagation rules.
        let explicit = sharding
            .program
            .nodes
            .iter()
            .filter(|node| node.placement != TensorPlacement::Unplaced)
            .map(|node| (node.id, node.placement.clone()))
            .collect::<BTreeMap<_, _>>();
        match self.sharding_plan(&explicit) {
            Ok(derived) if derived == *sharding => {}
            Ok(_) => {
                return Err(
                    "sharding plan was not derived from this execution plan; recompute it with sharding_plan"
                        .to_string(),
                )
            }
            Err(error) => {
                return Err(format!(
                    "sharding plan does not match this execution plan: {error}"
                ))
            }
        }

        let mesh = &sharding
            .all_reduces
            .first()
            .ok_or_else(|| {
                "sharding plan has no all-reduce; CUDA data-parallel execution requires at least one collective"
                    .to_string()
            })?
            .mesh;
        if mesh.shape.len() != 1 {
            return Err(format!(
                "CUDA data-parallel execution supports only a 1-D mesh, got axes {:?} with shape {:?}",
                mesh.axis_names, mesh.shape
            ));
        }
        let mesh_ordinals = mesh
            .devices
            .iter()
            .map(|device| (device.backend == TensorDeviceBackend::Cuda).then_some(device.ordinal))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                format!(
                    "CUDA data-parallel mesh must contain only CUDA devices, got {}",
                    TensorPlacement::Mesh {
                        mesh: mesh.clone(),
                        partition: TensorPartitionSpec::Replicated,
                    }
                )
            })?;
        if mesh_ordinals != device_ordinals {
            return Err(format!(
                "sharding mesh CUDA ordinals {mesh_ordinals:?} do not match data-parallel device ordinals {device_ordinals:?} in rank order"
            ));
        }
        let mesh_extent = mesh.shape[0];

        let mut sharded_inputs = BTreeSet::new();
        let mut batch_extent = None;
        for (placed, node) in sharding.program.nodes.iter().zip(self.nodes.iter()) {
            let partition = match &placed.placement {
                TensorPlacement::Unplaced => continue,
                TensorPlacement::Mesh {
                    mesh: node_mesh,
                    partition,
                } if node_mesh == mesh => partition,
                other => {
                    return Err(format!(
                        "kernel node {} has placement {other} outside the data-parallel mesh; explicit redistribution is required",
                        placed.id
                    ))
                }
            };
            let TensorPartitionSpec::Sharded { tensor_axis, .. } = partition else {
                continue;
            };
            let TensorOp::Input { name } = &node.op else {
                continue;
            };
            if *tensor_axis != 0 {
                return Err(format!(
                    "CUDA data-parallel input {name:?} is sharded on axis {tensor_axis}; only axis-zero batch sharding is supported"
                ));
            }
            let extent = node.shape[0];
            if let Some(expected) = batch_extent.filter(|expected| *expected != extent) {
                return Err(format!(
                    "CUDA data-parallel input {name:?} has batch extent {extent}, expected {expected}; sharded inputs must share one axis-zero extent"
                ));
            }
            batch_extent = Some(extent);
            sharded_inputs.insert(name.clone());
        }
        let batch_extent = batch_extent.ok_or_else(|| {
            "sharding plan has no axis-zero sharded input for CUDA data-parallel execution"
                .to_string()
        })?;

        let consumed = self
            .nodes
            .iter()
            .flat_map(|node| tensor_op_inputs(&node.op))
            .collect::<HashSet<_>>();
        let mut collectives = Vec::with_capacity(sharding.all_reduces.len());
        for collective in &sharding.all_reduces {
            let node_id = collective.node_id;
            if !self.output_node_ids.contains(&node_id) || consumed.contains(&node_id) {
                return Err(format!(
                    "sharding plan all-reduces node {node_id} before a consumer; CUDA data-parallel execution only all-reduces unconsumed retained outputs, so this mid-graph collective is unsupported"
                ));
            }
            collectives.push((node_id, collective.reduction));
        }
        if let Some(output) = self
            .output_node_ids
            .iter()
            .find(|output| !collectives.iter().any(|(node_id, _)| node_id == *output))
        {
            return Err(format!(
                "retained output {output} has no all-reduce; CUDA data-parallel execution returns rank-zero values only after a collective"
            ));
        }

        let replica = self
            .specialize_mapped_axis_zero(&sharded_inputs, batch_extent / mesh_extent)
            .map_err(|error| {
                format!("CUDA data-parallel replica specialization failed: {error}")
            })?;
        if replica.nodes.len() != self.nodes.len() {
            return Err(format!(
                "CUDA data-parallel replica specialization produced {} nodes, expected {}",
                replica.nodes.len(),
                self.nodes.len()
            ));
        }
        for ((local, node), placed) in replica
            .nodes
            .iter()
            .zip(self.nodes.iter())
            .zip(&sharding.program.nodes)
        {
            if tensor_op_inputs(&local.op) != tensor_op_inputs(&node.op) {
                return Err(format!(
                    "CUDA data-parallel replica specialization changed the inputs of node {}",
                    placed.id
                ));
            }
            let mut expected = node.shape.clone();
            if let Some(axis) = sharded_tensor_axis(&placed.placement) {
                expected[axis] /= mesh_extent;
            }
            if local.shape != expected {
                return Err(format!(
                    "kernel node {} has replica-local shape {:?}, but placement {} requires {expected:?}; explicit redistribution is required",
                    placed.id, local.shape, placed.placement
                ));
            }
        }
        Ok(TensorDataParallelProgram {
            replica_plan: TensorExecutionPlan {
                nodes: replica.nodes,
                input_nodes: replica.input_nodes,
                output_node_id: self.output_node_id,
                output_node_ids: self.output_node_ids.clone(),
                fused_elementwise_output: self.fused_elementwise_output,
            },
            collectives,
        })
    }

    fn kernel_ir_with_placement_mode(
        &self,
        placements: &BTreeMap<TensorNodeId, TensorPlacement>,
        collective_mode: TensorPlacementCollectiveMode,
    ) -> Result<(TensorKernelProgram, Vec<TensorAllReduce>), String> {
        for node_id in placements.keys() {
            if *node_id >= self.nodes.len() {
                return Err(format!("kernel placement node {node_id} does not exist"));
            }
        }

        let mut program = self.kernel_ir();
        let mut all_reduces = Vec::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            let explicit = placements
                .get(&node_id)
                .cloned()
                .unwrap_or(TensorPlacement::Unplaced);
            let inferred = infer_tensor_placement(
                node_id,
                node,
                &program.nodes,
                collective_mode,
                &mut all_reduces,
            )?;
            let placement = reconcile_tensor_placement(node_id, explicit, inferred)?;
            placement
                .validate_for_shape(&node.shape)
                .map_err(|error| format!("kernel node {node_id} has invalid placement: {error}"))?;
            program.nodes[node_id].placement = placement;
        }
        program.validate()?;
        Ok((program, all_reduces))
    }

    pub fn buffer_plan(&self) -> Result<TensorBufferPlan, String> {
        build_tensor_buffer_plan(&self.nodes, &self.output_node_ids)
    }

    /// Returns maximal elementwise regions that backends may lower as one
    /// kernel after materializing each listed input node.
    pub fn fusion_regions(&self) -> Vec<TensorFusionRegion> {
        build_tensor_fusion_regions(&self.nodes)
            .into_iter()
            .filter(|region| {
                region.node_ids.iter().all(|node_id| {
                    *node_id == region.output_node_id || !self.output_node_ids.contains(node_id)
                })
            })
            .collect()
    }

    /// Lowers a fused rank-N elementwise plan to one CUDA C kernel.
    ///
    /// The generated kernel deliberately excludes operations whose current CPU
    /// semantics need host-side validation or separate lowering: reductions,
    /// matmul, reshape, transpose, division, and log.
    pub fn cuda_source(&self) -> Result<String, String> {
        if !self.fused_elementwise_output {
            return Err(
                "CUDA lowering currently requires a pure fused elementwise output graph"
                    .to_string(),
            );
        }

        let output = self
            .nodes
            .get(self.output_node_id)
            .ok_or_else(|| "execution plan output node does not exist".to_string())?;
        let input_nodes = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| match &node.op {
                TensorOp::Input { name } => Some((id, name.as_str(), node.shape.as_slice())),
                _ => None,
            })
            .collect::<Vec<_>>();
        let expression = cuda_expression(&self.nodes, self.output_node_id)?;
        let parameters = input_nodes
            .iter()
            .map(|(id, _, _)| format!("const float* input_{id}"))
            .chain([
                "float* output".to_string(),
                "unsigned long long count".to_string(),
            ])
            .collect::<Vec<_>>()
            .join(", ");
        let offsets = input_nodes
            .iter()
            .map(|(id, _, shape)| cuda_broadcast_offset_function(*id, &output.shape, shape))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(
            "__device__ __forceinline__ float quabla_powi(float base, unsigned int exponent) {{\n\
    float result = 1.0f;\n\
    while (exponent != 0U) {{\n\
        if ((exponent & 1U) != 0U) result *= base;\n\
        base *= base;\n\
        exponent >>= 1U;\n\
    }}\n\
    return result;\n\
}}\n\
{offsets}\n\
extern \"C\" __global__ void quabla_fused_elementwise({parameters}) {{\n\
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (index < count) {{\n{expression}    }}\n\
}}\n"
        ))
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn cuda_fusion_region_source(&self, region: &TensorFusionRegion) -> Result<String, String> {
        let output = self
            .nodes
            .get(region.output_node_id)
            .ok_or_else(|| "CUDA fusion region output node does not exist".to_string())?;
        let leaves = region
            .input_node_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let expression = cuda_region_expression(
            &self.nodes,
            region.output_node_id,
            &leaves,
            region.output_node_id,
        )?;
        let parameters = region
            .input_node_ids
            .iter()
            .map(|id| format!("const float* input_{id}"))
            .chain([
                "float* output".to_string(),
                "unsigned long long count".to_string(),
            ])
            .collect::<Vec<_>>()
            .join(", ");
        let offsets = region
            .input_node_ids
            .iter()
            .map(|id| {
                let input = self.nodes.get(*id).ok_or_else(|| {
                    format!("CUDA fusion region references missing input node {id}")
                })?;
                Ok(cuda_broadcast_offset_function_named(
                    &cuda_fusion_region_offset_function_name(region.output_node_id, *id),
                    &output.shape,
                    &input.shape,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?
            .join("\n");
        let function = cuda_fusion_region_function_name(region.output_node_id);
        Ok(format!(
            "{offsets}\nextern \"C\" __global__ void {function}({parameters}) {{\n\
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (index < count) {{\n{expression}    }}\n\
}}\n"
        ))
    }

    pub fn evaluate(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        CpuBackend.execute(self, inputs)
    }

    /// Evaluates every retained output from a shared multi-output plan.
    pub fn evaluate_many(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        let mut values = TensorIr::evaluate_tensor_nodes_with_outputs(
            &self.nodes,
            inputs,
            Some(&self.output_node_ids),
        )?;
        let mut output_uses = vec![0usize; self.nodes.len()];
        for output in &self.output_node_ids {
            output_uses[*output] += 1;
        }
        self.output_node_ids
            .iter()
            .map(|output| {
                output_uses[*output] -= 1;
                let value = values
                    .get_mut(*output)
                    .ok_or_else(|| format!("output node {output} has no value"))?;
                if output_uses[*output] == 0 {
                    value.take()
                } else {
                    value.clone()
                }
                .ok_or_else(|| format!("output node {output} has no value"))
            })
            .collect()
    }

    fn execute_cpu(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        // The CPU runs every plan node by node, including elementwise plans
        // that CUDA fuses: whole-buffer kernels round each `float32` operation
        // to `f32` and avoid a per-element graph walk.
        TensorIr::evaluate_tensor_nodes_with_outputs(
            &self.nodes,
            inputs,
            Some(&[self.output_node_id]),
        )?
        .get_mut(self.output_node_id)
        .and_then(Option::take)
        .ok_or_else(|| format!("output node {} has no value", self.output_node_id))
    }

    pub fn vjp(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        self.as_ir()
            .vjp(self.output_node_id, inputs, output_cotangent)
    }

    pub fn value_and_vjp(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        self.as_ir()
            .value_and_vjp(self.output_node_id, inputs, output_cotangent)
    }

    /// Reverse-mode evaluation for every retained output in plan order.
    pub fn value_and_vjp_many(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangents: Vec<DynamicTensor>,
    ) -> Result<(Vec<DynamicTensor>, BTreeMap<String, DynamicTensor>), String> {
        if output_cotangents.len() != self.output_node_ids.len() {
            return Err(format!(
                "multi-output plan received {} cotangents, expected {}",
                output_cotangents.len(),
                self.output_node_ids.len()
            ));
        }
        let outputs = self
            .output_node_ids
            .iter()
            .copied()
            .zip(output_cotangents)
            .collect::<Vec<_>>();
        self.as_ir().value_and_vjp_many(&outputs, inputs)
    }

    /// Deterministic CPU reference for axis-zero data-parallel scalar losses.
    ///
    /// `mapped_input_names` are split evenly along axis zero. All other inputs
    /// are replicated. The caller must declare whether replica-local scalar
    /// losses combine by sum or mean; this determines both primal and VJP
    /// scaling. This is a correctness oracle for future collective lowering,
    /// not a parallel execution backend.
    pub fn value_and_vjp_data_parallel(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
        mapped_input_names: &BTreeSet<String>,
        shard_count: usize,
        reduction: TensorReplicaReduction,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        if shard_count == 0 {
            return Err("data-parallel shard_count must be positive".to_string());
        }
        if mapped_input_names.is_empty() {
            return Err("data-parallel execution requires at least one mapped input".to_string());
        }
        if !self.output_shape()?.is_empty() {
            return Err(format!(
                "data-parallel reference requires a scalar loss, got shape {:?}",
                self.output_shape()?
            ));
        }
        if !output_cotangent.shape.is_empty() {
            return Err(format!(
                "data-parallel scalar loss requires a scalar cotangent, got shape {:?}",
                output_cotangent.shape
            ));
        }
        let mut batch_extent = None;
        for name in mapped_input_names {
            let input = inputs
                .get(name)
                .ok_or_else(|| format!("data-parallel mapped input {name:?} is missing"))?;
            let extent = *input.shape.first().ok_or_else(|| {
                format!("data-parallel mapped input {name:?} must have rank at least one")
            })?;
            if extent == 0 {
                return Err(format!(
                    "data-parallel mapped input {name:?} has an empty axis zero"
                ));
            }
            if let Some(expected) = batch_extent {
                if extent != expected {
                    return Err(format!(
                        "data-parallel mapped input {name:?} has batch extent {extent}, expected {expected}"
                    ));
                }
            } else {
                batch_extent = Some(extent);
            }
        }
        let batch_extent = batch_extent.ok_or_else(|| {
            "data-parallel execution could not determine a mapped batch extent".to_string()
        })?;
        if batch_extent % shard_count != 0 {
            return Err(format!(
                "data-parallel batch extent {batch_extent} is not divisible by shard_count {shard_count}"
            ));
        }
        let shard_extent = batch_extent / shard_count;
        let shard_plan = self
            .specialize_mapped_axis_zero(mapped_input_names, shard_extent)?
            .compile_cpu(self.output_node_id)?;
        let scale = match reduction {
            TensorReplicaReduction::Sum => 1.0,
            TensorReplicaReduction::Mean => 1.0 / shard_count as f64,
        };
        let local_cotangent = output_cotangent.scale(scale)?;
        let mut output: Option<DynamicTensor> = None;
        let mut replicated_gradients = BTreeMap::<String, DynamicTensor>::new();
        let mut mapped_gradients = BTreeMap::<String, Vec<DynamicTensor>>::new();

        for shard in 0..shard_count {
            let start = shard * shard_extent;
            let mut shard_inputs = inputs.clone();
            for name in mapped_input_names {
                let input = inputs
                    .get(name)
                    .ok_or_else(|| format!("data-parallel mapped input {name:?} is missing"))?;
                shard_inputs.insert(name.clone(), input.slice_axis(0, start, shard_extent)?);
            }
            let (local_output, local_gradients) =
                shard_plan.value_and_vjp(&shard_inputs, local_cotangent.clone())?;
            output = Some(match output {
                Some(existing) => existing.add(&local_output.scale(scale)?)?,
                None => local_output.scale(scale)?,
            });
            for (name, gradient) in local_gradients {
                if mapped_input_names.contains(&name) {
                    mapped_gradients.entry(name).or_default().push(gradient);
                } else if let Some(existing) = replicated_gradients.get_mut(&name) {
                    *existing = existing.add(&gradient)?;
                } else {
                    replicated_gradients.insert(name, gradient);
                }
            }
        }
        for (name, shards) in mapped_gradients {
            let references = shards.iter().collect::<Vec<_>>();
            replicated_gradients.insert(name, DynamicTensor::concat(&references, 0)?);
        }
        output
            .ok_or_else(|| "data-parallel reference produced no shards".to_string())
            .map(|value| (value, replicated_gradients))
    }

    pub fn jvp(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.as_ir()
            .jvp(self.output_node_id, inputs, input_tangents)
    }

    pub fn hessian_scalar(
        &self,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.as_ir()
            .hessian_scalar(self.output_node_id, input_name, inputs)
    }

    pub fn hvp_scalar(
        &self,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangent: DynamicTensor,
    ) -> Result<DynamicTensor, String> {
        self.as_ir()
            .hvp_scalar(self.output_node_id, input_name, inputs, input_tangent)
    }

    fn as_ir(&self) -> TensorIr {
        TensorIr {
            nodes: self.nodes.clone(),
            input_nodes: self.input_nodes.clone(),
        }
    }

    fn specialize_mapped_axis_zero(
        &self,
        mapped_input_names: &BTreeSet<String>,
        shard_extent: usize,
    ) -> Result<TensorIr, String> {
        let mut specialized = TensorIr::new();
        let mut remap = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.iter() {
            let mapped = |node_id: TensorNodeId| {
                remap.get(node_id).copied().ok_or_else(|| {
                    format!("data-parallel specialization operand {node_id} is missing")
                })
            };
            let node_id = match &node.op {
                TensorOp::Input { name } => {
                    let mut shape = node.shape.clone();
                    if mapped_input_names.contains(name) {
                        let extent = shape.first_mut().ok_or_else(|| {
                            format!(
                                "data-parallel mapped input {name:?} must have rank at least one"
                            )
                        })?;
                        *extent = shard_extent;
                    }
                    specialized.input_typed(name.clone(), shape, node.dtype)?
                }
                TensorOp::ScalarConstant { value } => {
                    specialized.constant_like(*value, node.dtype, node.weak)
                }
                // Constants are replicated: only mapped inputs are sharded.
                TensorOp::Constant { .. } => specialized.push_node(
                    node.op.clone(),
                    node.shape.clone(),
                    node.dtype,
                    node.weak,
                ),
                TensorOp::Cast { input } => specialized.cast(mapped(*input)?, node.dtype)?,
                TensorOp::Add { lhs, rhs } => specialized.add(mapped(*lhs)?, mapped(*rhs)?)?,
                TensorOp::Sub { lhs, rhs } => specialized.sub(mapped(*lhs)?, mapped(*rhs)?)?,
                TensorOp::Div { lhs, rhs } => specialized.div(mapped(*lhs)?, mapped(*rhs)?)?,
                TensorOp::Mul { lhs, rhs } => specialized.mul(mapped(*lhs)?, mapped(*rhs)?)?,
                TensorOp::Greater { lhs, rhs } => {
                    specialized.greater(mapped(*lhs)?, mapped(*rhs)?)?
                }
                TensorOp::Compare { lhs, rhs, kind } => {
                    specialized.compare(mapped(*lhs)?, mapped(*rhs)?, *kind)?
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => specialized.where_select(
                    mapped(*condition)?,
                    mapped(*on_true)?,
                    mapped(*on_false)?,
                )?,
                TensorOp::Sum { input } => specialized.sum(mapped(*input)?)?,
                TensorOp::SumAxis { input, axis } => {
                    specialized.sum_axis(mapped(*input)?, *axis as isize)?
                }
                TensorOp::ExtremumAxis { input, axis, kind } => {
                    specialized.extremum_axis(mapped(*input)?, *axis as isize, *kind)?
                }
                TensorOp::Matmul { lhs, rhs } => {
                    specialized.matmul(mapped(*lhs)?, mapped(*rhs)?)?
                }
                TensorOp::CholeskyAd { inputs, kind } => specialized.cholesky_ad(
                    inputs
                        .iter()
                        .map(|id| mapped(*id))
                        .collect::<Result<Vec<_>, _>>()?,
                    *kind,
                )?,
                TensorOp::Cholesky { input } => specialized.cholesky(mapped(*input)?)?,
                TensorOp::Solve { matrix, rhs } => {
                    specialized.solve(mapped(*matrix)?, mapped(*rhs)?)?
                }
                TensorOp::Linalg { input, kind } => specialized.linalg(mapped(*input)?, *kind)?,
                TensorOp::Triangular { input, lower } => {
                    specialized.triangular(mapped(*input)?, *lower)?
                }
                TensorOp::Sqrt { input } => specialized.sqrt(mapped(*input)?)?,
                TensorOp::SqrtDerivative { input, order } => {
                    specialized.sqrt_derivative(mapped(*input)?, *order)?
                }
                TensorOp::Reshape { input } => {
                    specialized.reshape(mapped(*input)?, node.shape.clone())?
                }
                TensorOp::Mean { input } => specialized.mean(mapped(*input)?)?,
                TensorOp::MeanAxis { input, axis } => {
                    specialized.mean_axis(mapped(*input)?, *axis as isize)?
                }
                TensorOp::Powi { input, exponent } => {
                    specialized.powi(mapped(*input)?, *exponent)?
                }
                TensorOp::BinaryMath { lhs, rhs, kind } => {
                    specialized.binary_math(mapped(*lhs)?, mapped(*rhs)?, *kind)?
                }
                TensorOp::Transpose { input, axes } => specialized.transpose(
                    mapped(*input)?,
                    Some(axes.iter().map(|axis| *axis as isize).collect()),
                )?,
                TensorOp::UnaryMath { input, kind } => {
                    specialized.unary_math(mapped(*input)?, *kind)?
                }
                TensorOp::StopGradient { input } => specialized.stop_gradient(mapped(*input)?)?,
                TensorOp::CumSum {
                    input,
                    axis,
                    reverse,
                } => specialized.cumsum(mapped(*input)?, *axis as isize, *reverse)?,
                TensorOp::Concat { inputs, axis } => specialized.concat(
                    inputs
                        .iter()
                        .map(|input| mapped(*input))
                        .collect::<Result<Vec<_>, _>>()?,
                    *axis as isize,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => specialized.slice(mapped(*input)?, *axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => {
                    specialized.pad_slice(mapped(*input)?, node.shape.clone(), *axis, *start)?
                }
                TensorOp::Gather {
                    input,
                    indices,
                    axis,
                } => specialized.gather(mapped(*input)?, indices.clone(), *axis)?,
                TensorOp::ScatterAdd {
                    base,
                    updates,
                    indices,
                    axis,
                } => specialized.scatter_add(
                    mapped(*base)?,
                    mapped(*updates)?,
                    indices.clone(),
                    *axis,
                )?,
                TensorOp::Broadcast { input } => {
                    specialized.broadcast_to(mapped(*input)?, node.shape.clone())?
                }
                TensorOp::Cond { .. } => {
                    return Err(
                        "batch specialization does not yet transform Cond regions".to_string()
                    )
                }
                TensorOp::While { .. } => {
                    return Err(
                        "batch specialization does not yet transform While regions".to_string()
                    )
                }
                TensorOp::Custom { .. } => {
                    return Err(
                        "batch specialization applies to compiled plans, which have no custom \
                         rule nodes"
                            .to_string(),
                    )
                }
                TensorOp::Fori { .. } => {
                    return Err(
                        "batch specialization does not yet transform Fori regions".to_string()
                    )
                }
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "batch specialization does not yet transform Fori JVP regions".to_string(),
                    )
                }
                TensorOp::ForiVjp { .. } => {
                    return Err(
                        "batch specialization does not yet transform Fori VJP regions".to_string(),
                    )
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "batch specialization does not yet transform Fori VJP JVP regions"
                            .to_string(),
                    )
                }
                TensorOp::Scan { .. } => {
                    return Err(
                        "batch specialization does not yet transform Scan regions".to_string()
                    )
                }
                TensorOp::ScanVjp { .. } => {
                    return Err(
                        "batch specialization does not yet transform Scan VJP regions".to_string(),
                    )
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "batch specialization does not yet transform Scan VJP JVP regions"
                            .to_string(),
                    )
                }
            };
            if specialized.node(node_id)?.dtype != node.dtype {
                return Err(format!(
                    "data-parallel specialization changed node dtype from {} to {}",
                    node.dtype,
                    specialized.node(node_id)?.dtype
                ));
            }
            remap.push(node_id);
        }
        Ok(specialized)
    }
}

fn cuda_expression(nodes: &[TensorNode], node_id: TensorNodeId) -> Result<String, String> {
    cuda_dag_statements(nodes, node_id, None)
}

fn cuda_dag_statements(
    nodes: &[TensorNode],
    root: TensorNodeId,
    region_leaves: Option<&HashSet<TensorNodeId>>,
) -> Result<String, String> {
    let mut reachable = vec![false; nodes.len()];
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        let node = nodes
            .get(id)
            .ok_or_else(|| format!("CUDA lowering references missing node {id}"))?;
        if reachable[id] {
            continue;
        }
        reachable[id] = true;
        if !region_leaves.is_some_and(|leaves| leaves.contains(&id)) {
            pending.extend(tensor_op_inputs(&node.op));
        }
    }
    let reference = |id| format!("quabla_value_{id}");
    let mut source = String::new();
    for (id, node) in nodes.iter().enumerate() {
        if !reachable[id] {
            continue;
        }
        let expression = if region_leaves.is_some_and(|leaves| leaves.contains(&id)) {
            format!("input_{id}[quabla_region_{root}_offset_{id}(index)]")
        } else {
            if region_leaves.is_some()
                && matches!(node.op, TensorOp::ScalarConstant { value } if !value.is_finite())
            {
                return Err(
                    "CUDA fusion region does not support non-finite scalar constants".to_string(),
                );
            }
            if region_leaves.is_some()
                && !is_fusable_elementwise_compute_op(&node.op)
                && !matches!(node.op, TensorOp::ScalarConstant { .. })
            {
                return Err(format!(
                    "CUDA fusion region cannot inline {} node {id}",
                    tensor_op_name(&node.op)
                ));
            }
            cuda_scalar_expression(nodes, id, |child| Ok(reference(child)))?
        };
        // These CUDA elementwise operations are pure and have no checked-domain
        // errors. Where still selects one value with a ternary, so NaNs in its
        // unused operand do not propagate. SSA also avoids recursive device
        // calls or exponentially expanded expressions in the CUDA compiler.
        source.push_str(&format!(
            "        const float quabla_value_{id} = {expression};\n"
        ));
    }
    source.push_str(&format!("        output[index] = {};\n", reference(root)));
    Ok(source)
}

fn cuda_scalar_expression(
    nodes: &[TensorNode],
    node_id: TensorNodeId,
    child: impl Fn(TensorNodeId) -> Result<String, String>,
) -> Result<String, String> {
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA lowering references missing node {node_id}"))?;
    match &node.op {
        TensorOp::Input { .. } => Ok(format!("input_{node_id}[quabla_offset_{node_id}(index)]")),
        TensorOp::ScalarConstant { value } if value.is_finite() => Ok(cuda_scalar_literal(*value)),
        TensorOp::ScalarConstant { .. } => {
            Err("CUDA lowering does not support non-finite scalar constants".to_string())
        }
        // `is_fusable_elementwise_subgraph` keeps array constants out of whole-plan kernels; the
        // per-node program binds them as device buffers.
        TensorOp::Constant { .. } => {
            Err("the fused CUDA kernel does not bind array constants".to_string())
        }
        // f32 and f64 both execute as float on CUDA (TensorDeviceBackend::execution_dtype), so cast
        // is the identity.
        TensorOp::Cast { input } => child(*input),
        TensorOp::Add { lhs, rhs } => Ok(format!("({} + {})", child(*lhs)?, child(*rhs)?)),
        TensorOp::Sub { lhs, rhs } => Ok(format!("({} - {})", child(*lhs)?, child(*rhs)?)),
        TensorOp::Mul { lhs, rhs } => Ok(format!("({} * {})", child(*lhs)?, child(*rhs)?)),
        TensorOp::Greater { lhs, rhs } => Ok(format!(
            "(({} > {}) ? 1.0f : 0.0f)",
            child(*lhs)?,
            child(*rhs)?
        )),
        TensorOp::Compare { lhs, rhs, kind } => Ok(format!(
            "(({} {} {}) ? 1.0f : 0.0f)",
            child(*lhs)?,
            kind.operator(),
            child(*rhs)?
        )),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => Ok(format!(
            "(({} != 0.0f) ? {} : {})",
            child(*condition)?,
            child(*on_true)?,
            child(*on_false)?
        )),
        TensorOp::Sqrt { input } => Ok(format!("sqrtf({})", child(*input)?)),
        TensorOp::SqrtDerivative { input, order } => {
            Ok(cuda_sqrt_derivative_expression(&child(*input)?, *order))
        }
        TensorOp::Powi { input, exponent } => {
            Ok(format!("quabla_powi({}, {}U)", child(*input)?, exponent))
        }
        TensorOp::BinaryMath { lhs, rhs, kind } => Ok(format!(
            "{}({}, {})",
            kind.cuda_function(),
            child(*lhs)?,
            child(*rhs)?
        )),
        TensorOp::UnaryMath { input, kind } => {
            Ok(format!("{}({})", kind.cuda_function(), child(*input)?))
        }
        TensorOp::StopGradient { input } => child(*input),
        TensorOp::Div { .. } => {
            Err("CUDA loop-body lowering does not yet support div or log".to_string())
        }
        TensorOp::Sum { .. }
        | TensorOp::CumSum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::ExtremumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Solve { .. }
        | TensorOp::Linalg { .. }
        | TensorOp::Cholesky { .. }
        | TensorOp::CholeskyAd { .. }
        | TensorOp::Triangular { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Gather { .. }
        | TensorOp::ScatterAdd { .. }
        | TensorOp::Broadcast { .. }
        | TensorOp::Cond { .. }
        | TensorOp::While { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. }
        | TensorOp::Custom { .. } => Err(format!(
            "CUDA lowering does not yet support {}",
            tensor_op_name(&node.op)
        )),
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_region_expression(
    nodes: &[TensorNode],
    node_id: TensorNodeId,
    leaves: &HashSet<TensorNodeId>,
    _region_output_node_id: TensorNodeId,
) -> Result<String, String> {
    cuda_dag_statements(nodes, node_id, Some(leaves))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_fusion_region_function_name(output_node_id: TensorNodeId) -> String {
    format!("quabla_fused_region_{output_node_id}")
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_fusion_region_offset_function_name(
    region_output_node_id: TensorNodeId,
    input_node_id: TensorNodeId,
) -> String {
    format!("quabla_region_{region_output_node_id}_offset_{input_node_id}")
}

fn cuda_broadcast_offset_function(
    node_id: TensorNodeId,
    output_shape: &[usize],
    input_shape: &[usize],
) -> String {
    cuda_broadcast_offset_function_named(
        &format!("quabla_offset_{node_id}"),
        output_shape,
        input_shape,
    )
}

fn cuda_broadcast_offset_function_named(
    function_name: &str,
    output_shape: &[usize],
    input_shape: &[usize],
) -> String {
    let input_strides = contiguous_strides(input_shape);
    let rank_offset = output_shape.len() - input_shape.len();
    let terms = (0..output_shape.len())
        .filter_map(|axis| {
            if axis < rank_offset {
                return None;
            }
            let input_axis = axis - rank_offset;
            (input_shape[input_axis] != 1).then(|| {
                let output_stride = output_shape[axis + 1..].iter().product::<usize>();
                format!(
                    "((index / {output_stride}ULL) % {}ULL) * {}ULL",
                    output_shape[axis], input_strides[input_axis]
                )
            })
        })
        .collect::<Vec<_>>();
    let offset = if terms.is_empty() {
        "0ULL".to_string()
    } else {
        terms.join(" + ")
    };
    format!(
        "__device__ __forceinline__ unsigned long long {function_name}(unsigned long long index) {{ return {offset}; }}"
    )
}

/// A CUDA `float` literal that rounds `value` like `value as f32`, spelled
/// with the shortest round-trip `f64` digits: a value exact in `f32` keeps an
/// `f` suffix, any other is cast to `float`. A program retyped to `double` for
/// `precision="float64"` drops the suffix and the cast and so keeps every bit
/// of the `f64` constant (the shortest `f32` digits would not: `2^-12` would
/// become `0.00024414062`).
fn cuda_scalar_literal(value: f64) -> String {
    if f64::from(value as f32) == value {
        format!("{value:?}f")
    } else {
        format!("((float){value:?})")
    }
}

fn is_fusable_elementwise_subgraph(nodes: &[TensorNode], node_id: TensorNodeId) -> bool {
    if node_id >= nodes.len() {
        return false;
    }
    let mut eligible = vec![false; node_id + 1];
    for (id, node) in nodes.iter().enumerate().take(node_id + 1) {
        eligible[id] = match &node.op {
            TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } => true,
            op if is_fusable_elementwise_compute_op(op) => tensor_op_inputs(op)
                .into_iter()
                .all(|input| input < id && eligible[input]),
            // Array constants and non-elementwise operators use the per-node program.
            _ => false,
        };
    }
    eligible[node_id]
}

/// Whether whole-plan and region fusion inline `op` into one CUDA
/// elementwise kernel; an elementwise math op answers through its kind.
fn is_fusable_elementwise_compute_op(op: &TensorOp) -> bool {
    match op {
        TensorOp::UnaryMath { kind, .. } => kind.cuda_fusable(),
        TensorOp::BinaryMath { kind, .. } => kind.cuda_fusable(),
        _ => matches!(
            op,
            TensorOp::Add { .. }
                | TensorOp::Sub { .. }
                | TensorOp::Mul { .. }
                | TensorOp::Greater { .. }
                | TensorOp::Compare { .. }
                | TensorOp::Where { .. }
                | TensorOp::Sqrt { .. }
                | TensorOp::SqrtDerivative { .. }
                | TensorOp::Powi { .. }
                | TensorOp::Cast { .. }
        ),
    }
}

/// The error of the numeric (CPU evaluator) AD paths, which do not apply
/// custom rules; only the symbolic transforms behind `quabla.grad`,
/// `quabla.jvp`, and the other v0.2 transforms do.
fn custom_rule_numeric_error(rule: &TensorCustomRule) -> String {
    format!(
        "{} has a custom differentiation rule, which only the symbolic transforms \
         (quabla.grad, quabla.vjp, quabla.jvp, ...) apply",
        rule.name()
    )
}

fn tensor_op_inputs(op: &TensorOp) -> Vec<TensorNodeId> {
    match op {
        TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } | TensorOp::Constant { .. } => {
            Vec::new()
        }
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Div { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Greater { lhs, rhs }
        | TensorOp::Compare { lhs, rhs, .. }
        | TensorOp::Matmul { lhs, rhs } => {
            vec![*lhs, *rhs]
        }
        TensorOp::BinaryMath { lhs, rhs, .. } => vec![*lhs, *rhs],
        TensorOp::Solve { matrix, rhs } => vec![*matrix, *rhs],
        TensorOp::CholeskyAd { inputs, .. } => inputs.clone(),
        TensorOp::Cholesky { input }
        | TensorOp::Triangular { input, .. }
        | TensorOp::Linalg { input, .. } => vec![*input],
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => vec![*condition, *on_true, *on_false],
        TensorOp::Cond {
            predicate,
            captures,
            ..
        } => std::iter::once(*predicate)
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::While {
            carry, captures, ..
        } => std::iter::once(*carry)
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::Fori {
            carry, captures, ..
        } => std::iter::once(*carry)
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::ForiJvp {
            carry,
            carry_tangent,
            captures,
            tangent_captures,
            ..
        } => std::iter::once(*carry)
            .chain(std::iter::once(*carry_tangent))
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .chain(tangent_captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::ForiVjp {
            carry,
            output_cotangent,
            captures,
            ..
        } => std::iter::once(*carry)
            .chain(std::iter::once(*output_cotangent))
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::ForiVjpJvp {
            carry,
            carry_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures,
            tangent_captures,
            ..
        } => std::iter::once(*carry)
            .chain(std::iter::once(*carry_tangent))
            .chain(std::iter::once(*output_cotangent))
            .chain(std::iter::once(*output_cotangent_tangent))
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .chain(tangent_captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::Scan {
            carry, captures, ..
        } => std::iter::once(*carry)
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::ScanVjp {
            carry,
            final_carry_cotangent,
            output_cotangent,
            captures,
            ..
        } => std::iter::once(*carry)
            .chain(std::iter::once(*final_carry_cotangent))
            .chain(std::iter::once(*output_cotangent))
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::ScanVjpJvp {
            carry,
            carry_tangent,
            final_carry_cotangent,
            final_carry_cotangent_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures,
            tangent_captures,
            ..
        } => std::iter::once(*carry)
            .chain(std::iter::once(*carry_tangent))
            .chain(std::iter::once(*final_carry_cotangent))
            .chain(std::iter::once(*final_carry_cotangent_tangent))
            .chain(std::iter::once(*output_cotangent))
            .chain(std::iter::once(*output_cotangent_tangent))
            .chain(captures.iter().map(|(_, node_id)| *node_id))
            .chain(tangent_captures.iter().map(|(_, node_id)| *node_id))
            .collect(),
        TensorOp::Sum { input }
        | TensorOp::SumAxis { input, .. }
        | TensorOp::ExtremumAxis { input, .. }
        | TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Reshape { input }
        | TensorOp::Mean { input }
        | TensorOp::MeanAxis { input, .. }
        | TensorOp::Powi { input, .. }
        | TensorOp::Transpose { input, .. }
        | TensorOp::UnaryMath { input, .. }
        | TensorOp::StopGradient { input }
        | TensorOp::CumSum { input, .. }
        | TensorOp::Slice { input, .. }
        | TensorOp::PadSlice { input, .. }
        | TensorOp::Gather { input, .. }
        | TensorOp::Broadcast { input }
        | TensorOp::Cast { input } => {
            vec![*input]
        }
        TensorOp::ScatterAdd { base, updates, .. } => vec![*base, *updates],
        TensorOp::Concat { inputs, .. } => inputs.clone(),
        // The operands are inputs for reachability and inlining, so a rule
        // can still be applied to them after the node is spliced.
        TensorOp::Custom {
            value, operands, ..
        } => std::iter::once(*value)
            .chain(operands.iter().copied())
            .collect(),
    }
}

/// Operands whose dtype an ordinary op's result inherits. A `where` mask
/// is only tested for non-zero values and therefore does not participate.
fn tensor_value_operands(op: &TensorOp) -> Vec<TensorNodeId> {
    match op {
        TensorOp::Where {
            on_true, on_false, ..
        } => vec![*on_true, *on_false],
        TensorOp::Custom { value, .. } => vec![*value],
        _ => tensor_op_inputs(op),
    }
}

/// `Bool` values have no tangent or cotangent; differentiating one is an error
/// rather than a silent zero.
fn ensure_differentiable_output(node: &TensorNode) -> Result<(), String> {
    if node.dtype == TensorDType::Bool {
        return Err(
            "cannot differentiate a bool output; convert it explicitly with astype".to_string(),
        );
    }
    Ok(())
}

fn bool_input_derivative_error(name: &str) -> String {
    format!("cannot differentiate with respect to bool input {name:?}; bool values have no tangent")
}

/// Loop-region AD transforms pair every body input with a tangent or
/// gradient, so D2 keeps `Bool` values out of loop regions entirely.
fn check_loop_region_inputs(plan: &TensorExecutionPlan, context: &str) -> Result<(), String> {
    for node in plan.nodes.iter() {
        if let TensorOp::Input { name } = &node.op {
            if !node.dtype.is_floating() {
                return Err(format!(
                    "{context} input {name:?} has dtype {}; loop regions accept only floating \
                     values, so convert masks explicitly with astype",
                    node.dtype
                ));
            }
        }
    }
    Ok(())
}

/// Loop regions feed each body output back into its carry input, so both
/// must share one dtype.
fn check_region_output_dtype(
    plan: &TensorExecutionPlan,
    output: TensorNodeId,
    carry_name: &str,
    context: &str,
) -> Result<(), String> {
    let output_dtype = plan.node_dtype(output)?;
    let carry_dtype = plan.input_dtype(carry_name)?;
    if output_dtype != carry_dtype {
        return Err(format!(
            "{context} body returns dtype {output_dtype} for carry {carry_name:?} of dtype {carry_dtype}; \
             convert it explicitly with astype"
        ));
    }
    Ok(())
}

fn tensor_scalar_predicate(value: &DynamicTensor) -> Result<bool, String> {
    if !value.shape.is_empty() {
        return Err(format!(
            "conditional predicate must be scalar, got shape {:?}",
            value.shape
        ));
    }
    let scalar = value
        .data
        .first()
        .ok_or_else(|| "scalar conditional predicate has no value".to_string())?;
    if !scalar.is_finite() {
        return Err("conditional predicate must be finite".to_string());
    }
    Ok(scalar != 0.0)
}

#[cfg(test)]
thread_local! {
    // Counts simultaneously retained interpreter-node elements, independent of allocator RSS.
    static CPU_FORWARD_PEAK_ELEMENTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct TensorForwardLastUses {
    remaining: Vec<usize>,
    retained: Vec<bool>,
    group_last_nodes: HashMap<usize, TensorNodeId>,
}

fn reuse_forward_unary(
    node: &TensorNode,
    values: &mut [Option<DynamicTensor>],
    last_uses: Option<&TensorForwardLastUses>,
) -> Result<Option<DynamicTensor>, String> {
    let Some(last_uses) = last_uses else {
        return Ok(None);
    };
    type UnaryOperation = Option<fn(f64) -> f64>;
    let (input, operation): (usize, UnaryOperation) = match node.op {
        TensorOp::UnaryMath { input, kind } => (input, Some(kind.function())),
        TensorOp::Reshape { input } | TensorOp::StopGradient { input } => (input, None),
        _ => return Ok(None),
    };
    if last_uses.remaining.get(input) != Some(&1) || last_uses.retained.get(input) != Some(&false) {
        return Ok(None);
    }
    let source = values
        .get_mut(input)
        .and_then(Option::as_mut)
        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
    if operation.is_none() && element_count(&node.shape)? != source.data.len() {
        return Err(format!(
            "cannot reshape tensor with {} elements to shape {:?}",
            source.data.len(),
            node.shape
        ));
    }
    // Only the final, unretained use moves storage; reverse tapes never enter this path.
    let mut value = values[input].take().expect("source was validated above");
    if let Some(operation) = operation {
        match &mut value.data {
            HostTensorStorage::F64(data) => {
                for element in Arc::make_mut(data) {
                    *element = operation(*element);
                }
            }
            HostTensorStorage::F32(data) => {
                // Each lane keeps the F64 intermediate and the original final F32 rounding.
                for element in Arc::make_mut(data) {
                    *element = operation(f64::from(*element)) as f32;
                }
            }
            HostTensorStorage::Bool(_) => {
                value.data = HostTensorStorage::from_f64(
                    value.data.iter().map(operation).collect(),
                    TensorDType::F64,
                );
                value.dtype = TensorDType::F64;
            }
        }
    }
    value.shape = node.shape.clone();
    Ok(Some(value))
}

fn reuse_forward_binary(
    node: &TensorNode,
    values: &mut [Option<DynamicTensor>],
    last_uses: Option<&TensorForwardLastUses>,
) -> Result<Option<DynamicTensor>, String> {
    let Some(last_uses) = last_uses else {
        return Ok(None);
    };
    let (lhs, rhs, operation): (usize, usize, fn(f64, f64) -> f64) = match node.op {
        TensorOp::Add { lhs, rhs } => (lhs, rhs, |a, b| a + b),
        TensorOp::Sub { lhs, rhs } => (lhs, rhs, |a, b| a - b),
        TensorOp::Mul { lhs, rhs } => (lhs, rhs, |a, b| a * b),
        TensorOp::Div { lhs, rhs } => (lhs, rhs, |a, b| a / b),
        _ => return Ok(None),
    };
    let uses = if lhs == rhs { 2 } else { 1 };
    if last_uses.remaining.get(lhs) != Some(&uses) || last_uses.retained.get(lhs) != Some(&false) {
        return Ok(None);
    }
    let left = values
        .get(lhs)
        .and_then(Option::as_ref)
        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
    let right = values
        .get(rhs)
        .and_then(Option::as_ref)
        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
    if left.shape != node.shape
        || !(right.shape == node.shape || right.shape.is_empty())
        || left.dtype == TensorDType::Bool
    {
        return Ok(None);
    }
    let scalar = right.shape.is_empty();
    let right = (lhs != rhs).then(|| right.data.clone());
    let mut value = values[lhs].take().expect("operand was validated above");
    let lane = |index: usize, left: f64| {
        let right = right
            .as_ref()
            .map_or(left, |data| data.get(if scalar { 0 } else { index }));
        operation(left, right)
    };
    match &mut value.data {
        HostTensorStorage::F64(data) => {
            for (index, value) in Arc::make_mut(data).iter_mut().enumerate() {
                *value = lane(index, *value);
            }
        }
        HostTensorStorage::F32(data) => {
            for (index, value) in Arc::make_mut(data).iter_mut().enumerate() {
                *value = lane(index, f64::from(*value)) as f32;
            }
        }
        HostTensorStorage::Bool(_) => unreachable!("Bool operands use the original evaluator"),
    }
    Ok(Some(value))
}

#[cfg(test)]
mod forward_storage_reuse_tests {
    use super::*;

    #[test]
    fn graph_snapshots_share_metadata_until_mutation_and_keep_name_index() {
        let mut graph = TensorIr::new();
        let input = graph.input_typed("x", vec![2], TensorDType::F32).unwrap();
        let snapshot = graph.clone();
        assert!(Arc::ptr_eq(&graph.nodes, &snapshot.nodes));
        graph.sin(input).unwrap();
        assert!(!Arc::ptr_eq(&graph.nodes, &snapshot.nodes));
        assert_eq!(snapshot.node_count(), 1);
        assert_eq!(
            graph.input("x", vec![2]).unwrap_err(),
            "input \"x\" already exists"
        );
        let plan = snapshot.compile_cpu(input).unwrap();
        let cloned = plan.clone();
        assert!(Arc::ptr_eq(&plan.nodes, &cloned.nodes));
        let thawed = plan.as_ir();
        assert!(Arc::ptr_eq(&plan.nodes, &thawed.nodes));
        assert_eq!(plan.input_shape("x").unwrap(), vec![2]);
        assert_eq!(plan.input_dtype("x").unwrap(), TensorDType::F32);
        assert_eq!(
            plan.input_shape("missing").unwrap_err(),
            "execution plan input \"missing\" does not exist"
        );
    }

    #[test]
    fn last_use_reuses_owned_storage_and_preserves_node_rounding() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let original = DynamicTensor::with_dtype(
                vec![4],
                vec![-0.0, 0.125, f64::INFINITY, f64::NAN],
                dtype,
            )
            .unwrap();
            let mut ir = TensorIr::new();
            let input = ir.input_typed("x", vec![4], dtype).unwrap();
            let output = ir.sin(input).unwrap();
            let expected = original.map_f64(f64::sin).unwrap().into_dtype(dtype);
            let mut values = vec![Some(original)];
            let address = match &values[0].as_ref().unwrap().data {
                HostTensorStorage::F64(data) => data.as_ptr() as usize,
                HostTensorStorage::F32(data) => data.as_ptr() as usize,
                HostTensorStorage::Bool(_) => unreachable!(),
            };
            let uses = tensor_forward_last_uses(&ir.nodes, &[output]);
            let actual = reuse_forward_unary(&ir.nodes[output], &mut values, Some(&uses))
                .unwrap()
                .unwrap()
                .into_dtype(dtype);
            let actual_address = match &actual.data {
                HostTensorStorage::F64(data) => data.as_ptr() as usize,
                HostTensorStorage::F32(data) => data.as_ptr() as usize,
                HostTensorStorage::Bool(_) => unreachable!(),
            };
            assert_eq!(actual_address, address);
            assert!(values[0].is_none());
            assert_eq!(actual.dtype, dtype);
            for (actual, expected) in actual.data.iter().zip(expected.data.iter()) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn reshape_shares_typed_storage_and_unary_protects_external_alias() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let tensor =
                DynamicTensor::with_dtype(vec![4], vec![0.0, 1.0, 2.0, 3.0], dtype).unwrap();
            let reshaped = tensor.reshape(vec![2, 2]).unwrap();
            let shared = match (&tensor.data, &reshaped.data) {
                (HostTensorStorage::F64(a), HostTensorStorage::F64(b)) => Arc::ptr_eq(a, b),
                (HostTensorStorage::F32(a), HostTensorStorage::F32(b)) => Arc::ptr_eq(a, b),
                (HostTensorStorage::Bool(a), HostTensorStorage::Bool(b)) => Arc::ptr_eq(a, b),
                _ => false,
            };
            assert!(shared);
            assert_eq!(reshaped.dtype, dtype);
        }
        let tensor = DynamicTensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let alias = tensor.clone();
        let mut ir = TensorIr::new();
        let input = ir.input("x", vec![2]).unwrap();
        let output = ir.sin(input).unwrap();
        let uses = tensor_forward_last_uses(&ir.nodes, &[output]);
        let mut values = vec![Some(tensor)];
        let result = reuse_forward_unary(&ir.nodes[output], &mut values, Some(&uses))
            .unwrap()
            .unwrap();
        assert_eq!(alias.data().as_ref(), [1.0, 2.0]);
        assert_eq!(result.data().as_ref(), [1.0_f64.sin(), 2.0_f64.sin()]);
    }

    #[test]
    fn retained_outputs_and_reverse_tapes_keep_source_storage() {
        let mut ir = TensorIr::new();
        let input = ir.input("x", vec![2]).unwrap();
        let output = ir.sin(input).unwrap();
        let mut values = vec![Some(DynamicTensor::new(vec![2], vec![1.0, 2.0]).unwrap())];
        let uses = tensor_forward_last_uses(&ir.nodes, &[input, output]);
        assert!(
            reuse_forward_unary(&ir.nodes[output], &mut values, Some(&uses))
                .unwrap()
                .is_none()
        );
        assert!(reuse_forward_unary(&ir.nodes[output], &mut values, None)
            .unwrap()
            .is_none());
        assert_eq!(values[0].as_ref().unwrap().data().as_ref(), [1.0, 2.0]);
    }
}

fn tensor_forward_last_uses(
    nodes: &[TensorNode],
    outputs: &[TensorNodeId],
) -> TensorForwardLastUses {
    let mut remaining = vec![0; nodes.len()];
    let mut retained = vec![false; nodes.len()];
    let mut group_last_nodes = HashMap::new();
    for (id, node) in nodes.iter().enumerate() {
        for input in tensor_op_inputs(&node.op) {
            remaining[input] += 1;
        }
        if let Some(group) = tensor_op_group(&node.op) {
            group_last_nodes.insert(group, id);
        }
    }
    for output in outputs {
        if let Some(retained) = retained.get_mut(*output) {
            *retained = true;
        }
    }
    TensorForwardLastUses {
        remaining,
        retained,
        group_last_nodes,
    }
}

fn tensor_op_group(op: &TensorOp) -> Option<usize> {
    match op {
        TensorOp::ForiVjp { group, .. }
        | TensorOp::ForiVjpJvp { group, .. }
        | TensorOp::Scan { group, .. }
        | TensorOp::ScanVjp { group, .. }
        | TensorOp::ScanVjpJvp { group, .. } => Some(*group),
        _ => None,
    }
}

fn tensor_forward_capture_values(
    captures: &[(String, TensorNodeId)],
    values: &[Option<DynamicTensor>],
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    captures
        .iter()
        .map(|(name, node_id)| {
            values
                .get(*node_id)
                .and_then(Option::as_ref)
                .cloned()
                .map(|value| (name.clone(), value))
                .ok_or_else(|| format!("conditional capture node {node_id} has no evaluated value"))
        })
        .collect()
}

fn tensor_cond_capture_values(
    captures: &[(String, TensorNodeId)],
    values: &[DynamicTensor],
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    captures
        .iter()
        .map(|(name, node_id)| {
            values
                .get(*node_id)
                .cloned()
                .map(|value| (name.clone(), value))
                .ok_or_else(|| format!("conditional capture node {node_id} has no evaluated value"))
        })
        .collect()
}

fn tensor_fori_capture_values(
    captures: &[(String, TensorNodeId)],
    values: &[DynamicTensor],
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    tensor_cond_capture_values(captures, values)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorPlacementCollectiveMode {
    Reject,
    Plan,
}

fn infer_tensor_placement(
    node_id: TensorNodeId,
    node: &TensorNode,
    kernel_nodes: &[TensorKernelNode],
    collective_mode: TensorPlacementCollectiveMode,
    all_reduces: &mut Vec<TensorAllReduce>,
) -> Result<TensorPlacement, String> {
    let input_node = |input_id: TensorNodeId| -> Result<&TensorKernelNode, String> {
        kernel_nodes
            .get(input_id)
            .ok_or_else(|| format!("kernel node {node_id} references missing input {input_id}"))
    };
    let unary = |input_id| input_node(input_id).map(|node| node.placement.clone());
    let merge =
        |input_ids: &[TensorNodeId]| merge_tensor_placements(node_id, input_ids, kernel_nodes);

    match &node.op {
        TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } | TensorOp::Constant { .. } => {
            Ok(TensorPlacement::Unplaced)
        }
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Div { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Greater { lhs, rhs }
        | TensorOp::Compare { lhs, rhs, .. } => merge(&[*lhs, *rhs]),
        TensorOp::BinaryMath { lhs, rhs, .. } => merge(&[*lhs, *rhs]),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => merge(&[*condition, *on_true, *on_false]),
        TensorOp::Cond { .. }
        | TensorOp::While { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. }
        | TensorOp::Custom { .. } => Err(format!(
            "kernel node {node_id} contains {} regions; placement propagation requires explicit region lowering",
            tensor_op_name(&node.op)
        )),
        TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Powi { input, .. }
        | TensorOp::UnaryMath { input, .. }
        | TensorOp::StopGradient { input }
        | TensorOp::Cast { input } => unary(*input),
        TensorOp::CumSum { input, axis, .. } => {
            let placement = unary(*input)?;
            if sharded_tensor_axis(&placement) == Some(*axis) {
                return Err(format!(
                    "kernel node {node_id} scans along a sharded axis; explicit redistribution is required"
                ));
            }
            Ok(placement)
        }
        // There is no max/min replica reduction: a sharded reduced axis needs explicit
        // redistribution, as the per-element slices of the former traced chain did. Another
        // sharded axis keeps its shard and shifts down past the removed axis.
        TensorOp::ExtremumAxis { input, axis, .. } => match unary(*input)? {
            TensorPlacement::Mesh {
                mesh,
                partition:
                    TensorPartitionSpec::Sharded {
                        tensor_axis,
                        mesh_axis,
                    },
            } => {
                if tensor_axis == *axis {
                    return Err(format!(
                        "kernel node {node_id} reduces a max/min along a sharded axis; explicit redistribution is required"
                    ));
                }
                Ok(TensorPlacement::Mesh {
                    mesh,
                    partition: TensorPartitionSpec::Sharded {
                        tensor_axis: tensor_axis - usize::from(tensor_axis > *axis),
                        mesh_axis,
                    },
                })
            }
            placement => Ok(placement),
        },
        TensorOp::Sum { input } => {
            let placement = unary(*input)?;
            plan_full_reduction_placement(
                node_id,
                placement,
                TensorReplicaReduction::Sum,
                collective_mode,
                all_reduces,
            )
        }
        TensorOp::Mean { input } => {
            let placement = unary(*input)?;
            plan_full_reduction_placement(
                node_id,
                placement,
                TensorReplicaReduction::Mean,
                collective_mode,
                all_reduces,
            )
        }
        TensorOp::SumAxis { input, axis } => {
            let placement = unary(*input)?;
            plan_reduced_axis_placement(
                node_id,
                placement,
                *axis,
                TensorReplicaReduction::Sum,
                collective_mode,
                all_reduces,
            )
        }
        TensorOp::MeanAxis { input, axis } => {
            let placement = unary(*input)?;
            plan_reduced_axis_placement(
                node_id,
                placement,
                *axis,
                TensorReplicaReduction::Mean,
                collective_mode,
                all_reduces,
            )
        }
        TensorOp::Matmul { lhs, rhs } | TensorOp::Solve { matrix: lhs, rhs } => {
            let placement = merge(&[*lhs, *rhs])?;
            reject_sharded_placement(node_id, &placement, "matmul/solve lowering")?;
            Ok(placement)
        }
        TensorOp::CholeskyAd { inputs, .. } => { let placement = merge(inputs)?; reject_sharded_placement(node_id, &placement, "cholesky AD lowering")?; Ok(placement) },
        TensorOp::Cholesky { input } | TensorOp::Triangular { input, .. } => {
            let placement = unary(*input)?;
            reject_sharded_placement(node_id, &placement, "triangular lowering")?;
            Ok(placement)
        }
        TensorOp::Linalg { input, .. } => {
            let placement = unary(*input)?;
            reject_sharded_placement(node_id, &placement, "linalg lowering")?;
            Ok(placement)
        }
        TensorOp::Reshape { input } => {
            let placement = unary(*input)?;
            if is_sharded_placement(&placement) && input_node(*input)?.shape != node.shape {
                return Err(format!(
                    "kernel node {node_id} reshape changes a sharded tensor shape; explicit redistribution is required"
                ));
            }
            Ok(placement)
        }
        TensorOp::Transpose { input, axes } => {
            let placement = unary(*input)?;
            remap_transpose_placement(node_id, placement, axes)
        }
        TensorOp::Concat { inputs, axis } => {
            let placement = merge(inputs)?;
            // Local shard concatenation interleaves operands instead of
            // producing a contiguous shard of the global result.
            if sharded_tensor_axis(&placement) == Some(*axis) {
                return Err(format!(
                    "kernel node {node_id} concatenates along a sharded axis; explicit redistribution is required"
                ));
            }
            Ok(placement)
        }
        TensorOp::Slice { input, axis, .. }
        | TensorOp::PadSlice { input, axis, .. }
        | TensorOp::Gather { input, axis, .. } => {
            let placement = unary(*input)?;
            if sharded_tensor_axis(&placement) == Some(*axis) {
                return Err(format!(
                    "kernel node {node_id} slices along a sharded axis; explicit redistribution is required"
                ));
            }
            Ok(placement)
        }
        TensorOp::ScatterAdd {
            base,
            updates,
            axis,
            ..
        } => {
            let placement = merge(&[*base, *updates])?;
            if sharded_tensor_axis(&placement) == Some(*axis) {
                return Err(format!(
                    "kernel node {node_id} scatters along a sharded axis; explicit redistribution is required"
                ));
            }
            Ok(placement)
        }
        TensorOp::Broadcast { input } => {
            let placement = unary(*input)?;
            remap_broadcast_placement(
                node_id,
                placement,
                input_node(*input)?.shape.len(),
                node.shape.len(),
            )
        }
    }
}

fn merge_tensor_placements(
    node_id: TensorNodeId,
    input_ids: &[TensorNodeId],
    nodes: &[TensorKernelNode],
) -> Result<TensorPlacement, String> {
    let mut merged = TensorPlacement::Unplaced;
    let mut saw_unplaced_non_scalar = false;
    for input_id in input_ids {
        let input = nodes
            .get(*input_id)
            .ok_or_else(|| format!("kernel node {node_id} references missing input {input_id}"))?;
        let candidate = &input.placement;
        if *candidate == TensorPlacement::Unplaced {
            if input.shape.is_empty() {
                continue;
            }
            if merged != TensorPlacement::Unplaced {
                return Err(format!(
                    "kernel node {node_id} mixes placed input {input_id} with unplaced non-scalar input; explicit placement is required"
                ));
            }
            saw_unplaced_non_scalar = true;
            continue;
        }
        if saw_unplaced_non_scalar {
            return Err(format!(
                "kernel node {node_id} mixes placed input {input_id} with an unplaced non-scalar input; explicit placement is required"
            ));
        }
        if merged == TensorPlacement::Unplaced {
            merged = candidate.clone();
        } else {
            merged = compatible_tensor_placement(node_id, &merged, candidate)?;
        }
    }
    Ok(merged)
}

fn compatible_tensor_placement(
    node_id: TensorNodeId,
    left: &TensorPlacement,
    right: &TensorPlacement,
) -> Result<TensorPlacement, String> {
    if left == right {
        return Ok(left.clone());
    }
    match (left, right) {
        (
            TensorPlacement::Mesh {
                mesh: left_mesh,
                partition: TensorPartitionSpec::Replicated,
            },
            TensorPlacement::Mesh {
                mesh: right_mesh,
                partition: right_partition,
            },
        ) if left_mesh == right_mesh => Ok(TensorPlacement::Mesh {
            mesh: right_mesh.clone(),
            partition: right_partition.clone(),
        }),
        (
            TensorPlacement::Mesh {
                mesh: left_mesh,
                partition: left_partition,
            },
            TensorPlacement::Mesh {
                mesh: right_mesh,
                partition: TensorPartitionSpec::Replicated,
            },
        ) if left_mesh == right_mesh => Ok(TensorPlacement::Mesh {
            mesh: left_mesh.clone(),
            partition: left_partition.clone(),
        }),
        _ => Err(format!(
            "kernel node {node_id} has incompatible input placements {left} and {right}; explicit redistribution is required"
        )),
    }
}

fn reconcile_tensor_placement(
    node_id: TensorNodeId,
    explicit: TensorPlacement,
    inferred: TensorPlacement,
) -> Result<TensorPlacement, String> {
    if explicit == TensorPlacement::Unplaced {
        return Ok(inferred);
    }
    if inferred == TensorPlacement::Unplaced || explicit == inferred {
        return Ok(explicit);
    }
    Err(format!(
        "kernel node {node_id} explicitly requests placement {explicit}, but propagation inferred {inferred}"
    ))
}

fn is_sharded_placement(placement: &TensorPlacement) -> bool {
    matches!(
        placement,
        TensorPlacement::Mesh {
            partition: TensorPartitionSpec::Sharded { .. },
            ..
        }
    )
}

fn sharded_tensor_axis(placement: &TensorPlacement) -> Option<usize> {
    match placement {
        TensorPlacement::Mesh {
            partition: TensorPartitionSpec::Sharded { tensor_axis, .. },
            ..
        } => Some(*tensor_axis),
        _ => None,
    }
}

fn reject_sharded_placement(
    node_id: TensorNodeId,
    placement: &TensorPlacement,
    operation: &str,
) -> Result<(), String> {
    if is_sharded_placement(placement) {
        return Err(format!(
            "kernel node {node_id} requires a collective or redistribution for {operation}"
        ));
    }
    Ok(())
}

fn plan_full_reduction_placement(
    node_id: TensorNodeId,
    placement: TensorPlacement,
    reduction: TensorReplicaReduction,
    collective_mode: TensorPlacementCollectiveMode,
    all_reduces: &mut Vec<TensorAllReduce>,
) -> Result<TensorPlacement, String> {
    let TensorPlacement::Mesh { mesh, partition } = placement else {
        return Ok(placement);
    };
    let TensorPartitionSpec::Sharded {
        tensor_axis,
        mesh_axis,
    } = partition
    else {
        return Ok(TensorPlacement::Mesh { mesh, partition });
    };
    let _ = tensor_axis;
    plan_all_reduce(
        node_id,
        mesh,
        mesh_axis,
        reduction,
        collective_mode,
        all_reduces,
    )
}

fn plan_reduced_axis_placement(
    node_id: TensorNodeId,
    placement: TensorPlacement,
    reduced_axis: usize,
    reduction: TensorReplicaReduction,
    collective_mode: TensorPlacementCollectiveMode,
    all_reduces: &mut Vec<TensorAllReduce>,
) -> Result<TensorPlacement, String> {
    let TensorPlacement::Mesh { mesh, partition } = placement else {
        return Ok(placement);
    };
    let TensorPartitionSpec::Sharded {
        tensor_axis,
        mesh_axis,
    } = partition
    else {
        return Ok(TensorPlacement::Mesh { mesh, partition });
    };
    if tensor_axis == reduced_axis {
        return plan_all_reduce(
            node_id,
            mesh,
            mesh_axis,
            reduction,
            collective_mode,
            all_reduces,
        );
    }
    Ok(TensorPlacement::Mesh {
        mesh,
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: if tensor_axis > reduced_axis {
                tensor_axis - 1
            } else {
                tensor_axis
            },
            mesh_axis,
        },
    })
}

fn plan_all_reduce(
    node_id: TensorNodeId,
    mesh: TensorDeviceMesh,
    mesh_axis: String,
    reduction: TensorReplicaReduction,
    collective_mode: TensorPlacementCollectiveMode,
    all_reduces: &mut Vec<TensorAllReduce>,
) -> Result<TensorPlacement, String> {
    if collective_mode == TensorPlacementCollectiveMode::Reject {
        return Err(format!(
            "kernel node {node_id} reduces a sharded axis; an all-reduce is required"
        ));
    }
    all_reduces.push(TensorAllReduce {
        node_id,
        mesh: mesh.clone(),
        mesh_axis,
        reduction,
    });
    Ok(TensorPlacement::Mesh {
        mesh,
        partition: TensorPartitionSpec::Replicated,
    })
}

fn remap_transpose_placement(
    node_id: TensorNodeId,
    placement: TensorPlacement,
    axes: &[usize],
) -> Result<TensorPlacement, String> {
    let TensorPlacement::Mesh { mesh, partition } = placement else {
        return Ok(placement);
    };
    let TensorPartitionSpec::Sharded {
        tensor_axis,
        mesh_axis,
    } = partition
    else {
        return Ok(TensorPlacement::Mesh { mesh, partition });
    };
    let output_axis = axes
        .iter()
        .position(|axis| *axis == tensor_axis)
        .ok_or_else(|| {
            format!("kernel node {node_id} transpose does not map sharded axis {tensor_axis}")
        })?;
    Ok(TensorPlacement::Mesh {
        mesh,
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: output_axis,
            mesh_axis,
        },
    })
}

fn remap_broadcast_placement(
    node_id: TensorNodeId,
    placement: TensorPlacement,
    input_rank: usize,
    output_rank: usize,
) -> Result<TensorPlacement, String> {
    let TensorPlacement::Mesh { mesh, partition } = placement else {
        return Ok(placement);
    };
    let TensorPartitionSpec::Sharded {
        tensor_axis,
        mesh_axis,
    } = partition
    else {
        return Ok(TensorPlacement::Mesh { mesh, partition });
    };
    let rank_offset = output_rank.checked_sub(input_rank).ok_or_else(|| {
        format!("kernel node {node_id} broadcast lowers rank from {input_rank} to {output_rank}")
    })?;
    Ok(TensorPlacement::Mesh {
        mesh,
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: tensor_axis + rank_offset,
            mesh_axis,
        },
    })
}

fn tensor_op_name(op: &TensorOp) -> &'static str {
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
        TensorOp::Fori { .. } => "fori",
        TensorOp::While { .. } => "while",
        TensorOp::ForiJvp { .. } => "fori_jvp",
        TensorOp::ForiVjp { .. } => "fori_vjp",
        TensorOp::ForiVjpJvp { .. } => "fori_vjp_jvp",
        TensorOp::Scan { .. } => "scan",
        TensorOp::ScanVjp { .. } => "scan_vjp",
        TensorOp::ScanVjpJvp { .. } => "scan_vjp_jvp",
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
        TensorOp::CumSum { .. } => "cumsum",
        TensorOp::Concat { .. } => "concat",
        TensorOp::Slice { .. } => "slice",
        TensorOp::PadSlice { .. } => "pad_slice",
        TensorOp::Gather { .. } => "gather",
        TensorOp::ScatterAdd { .. } => "scatter_add",
        TensorOp::Broadcast { .. } => "broadcast",
        TensorOp::Custom { .. } => "custom",
    }
}

fn fold_scalar_constant_op(op: &TensorOp, nodes: &[TensorNode]) -> Option<f64> {
    let scalar = |node_id: TensorNodeId| {
        nodes.get(node_id).and_then(|node| match node {
            TensorNode {
                op: TensorOp::ScalarConstant { value },
                shape,
                ..
            } if shape.is_empty() => Some(*value),
            _ => None,
        })
    };
    match op {
        TensorOp::Add { lhs, rhs } => Some(scalar(*lhs)? + scalar(*rhs)?),
        TensorOp::Sub { lhs, rhs } => Some(scalar(*lhs)? - scalar(*rhs)?),
        TensorOp::Mul { lhs, rhs } => Some(scalar(*lhs)? * scalar(*rhs)?),
        TensorOp::Div { lhs, rhs } => {
            let denominator = scalar(*rhs)?;
            if denominator == 0.0 {
                None
            } else {
                Some(scalar(*lhs)? / denominator)
            }
        }
        TensorOp::Greater { lhs, rhs } => Some(f64::from(scalar(*lhs)? > scalar(*rhs)?)),
        TensorOp::Compare { lhs, rhs, kind } => {
            Some(f64::from(kind.evaluate(scalar(*lhs)?, scalar(*rhs)?)))
        }
        // The caller rounds to the target dtype, so f64->f32->f64 keeps the f32 rounding error
        // instead of cancelling it.
        TensorOp::Cast { input } => scalar(*input),
        TensorOp::Powi { input, exponent } => {
            let exponent = i32::try_from(*exponent).ok()?;
            Some(scalar(*input)?.powi(exponent))
        }
        TensorOp::BinaryMath { lhs, rhs, kind } => {
            Some(kind.evaluate(scalar(*lhs)?, scalar(*rhs)?))
        }
        TensorOp::UnaryMath { input, kind } => kind.fold_scalar(scalar(*input)?),
        _ => None,
    }
}

/// Folds an elementwise op whose operands are all constants, at least one of
/// them an array constant: the array counterpart of `fold_scalar_constant_op`,
/// covering the same ops. It evaluates with the CPU evaluator's own kernels on
/// operands rounded to their dtypes, so the caller only rounds the result to
/// the node dtype to get what per-node execution computes. An op that the
/// evaluator rejects (a zero divisor, a non-positive logarithm) stays unfolded
/// and still fails at execution.
fn fold_tensor_constant_op(op: &TensorOp, nodes: &[TensorNode]) -> Option<DynamicTensor> {
    let operands = tensor_op_inputs(op);
    let is_array = |id: &TensorNodeId| {
        matches!(
            nodes.get(*id).map(|node| &node.op),
            Some(TensorOp::Constant { .. })
        )
    };
    if !operands.iter().any(is_array) {
        return None;
    }
    let constant = |id: TensorNodeId| -> Option<std::borrow::Cow<'_, DynamicTensor>> {
        let node = nodes.get(id)?;
        match &node.op {
            TensorOp::ScalarConstant { value } if node.shape.is_empty() => {
                Some(std::borrow::Cow::Owned(
                    DynamicTensor::filled(vec![], *value)
                        .ok()?
                        .into_dtype(node.dtype),
                ))
            }
            TensorOp::Constant { value } => Some(std::borrow::Cow::Borrowed(value.value())),
            _ => None,
        }
    };
    match op {
        TensorOp::Add { lhs, rhs } => constant(*lhs)?.add(&*constant(*rhs)?).ok(),
        TensorOp::Sub { lhs, rhs } => constant(*lhs)?.sub(&*constant(*rhs)?).ok(),
        TensorOp::Mul { lhs, rhs } => constant(*lhs)?.mul(&*constant(*rhs)?).ok(),
        TensorOp::Div { lhs, rhs } => constant(*lhs)?.div(&*constant(*rhs)?).ok(),
        TensorOp::Greater { lhs, rhs } => constant(*lhs)?.greater(&*constant(*rhs)?).ok(),
        TensorOp::Compare { lhs, rhs, kind } => {
            constant(*lhs)?.compare(&*constant(*rhs)?, *kind).ok()
        }
        TensorOp::Cast { input } => Some(constant(*input)?.into_owned()),
        TensorOp::Powi { input, exponent } => constant(*input)?.powi(*exponent).ok(),
        TensorOp::BinaryMath { lhs, rhs, kind } => constant(*lhs)?
            .elementwise(&*constant(*rhs)?, kind.function())
            .ok(),
        TensorOp::UnaryMath { input, kind } => constant(*input)?.map_f64(kind.function()).ok(),
        _ => None,
    }
}

fn canonicalize_tensor_op(
    op: &mut TensorOp,
    output_shape: &[usize],
    output_dtype: TensorDType,
    nodes: &[TensorNode],
) -> Result<Option<TensorNodeId>, String> {
    match op {
        // Only same-dtype strong casts are removed; a lossy round trip (f64->f32->f64) keeps both
        // casts. A cast from a weak source makes the node strong; aliasing would change later
        // promotion semantics, so it is kept.
        TensorOp::Cast { input } => {
            let source = nodes
                .get(*input)
                .ok_or_else(|| format!("cast source node {input} is missing"))?;
            Ok((source.dtype == output_dtype && !source.weak).then_some(*input))
        }
        TensorOp::Reshape { input } => loop {
            let source = nodes
                .get(*input)
                .ok_or_else(|| format!("reshape source node {input} is missing"))?;
            if source.shape == output_shape {
                return Ok(Some(*input));
            }
            match source.op {
                TensorOp::Reshape { input: parent } => *input = parent,
                _ => return Ok(None),
            }
        },
        TensorOp::Broadcast { input } => {
            let source = nodes
                .get(*input)
                .ok_or_else(|| format!("broadcast source node {input} is missing"))?;
            Ok((source.shape == output_shape).then_some(*input))
        }
        // The identity on its value once the symbolic transforms are done.
        TensorOp::Custom { value, .. } => Ok(Some(*value)),
        _ => Ok(None),
    }
}

fn build_tensor_fusion_regions(nodes: &[TensorNode]) -> Vec<TensorFusionRegion> {
    let mut users = vec![Vec::new(); nodes.len()];
    for (node_id, node) in nodes.iter().enumerate() {
        for input in tensor_op_inputs(&node.op) {
            if let Some(input_users) = users.get_mut(input) {
                // Nodes are visited in order, so duplicate operands can only
                // repeat the most recently recorded user.
                if input_users.last() != Some(&node_id) {
                    input_users.push(node_id);
                }
            }
        }
    }

    let mut regions = Vec::new();
    for (root, node) in nodes.iter().enumerate() {
        if !is_fusable_elementwise_compute_op(&node.op)
            || users[root]
                .iter()
                .any(|user| is_fusable_elementwise_compute_op(&nodes[*user].op))
        {
            continue;
        }
        let mut region_nodes = HashSet::new();
        let mut inputs = HashSet::new();
        collect_fusion_region_nodes(nodes, &users, root, root, &mut region_nodes, &mut inputs);
        if region_nodes.len() < 2 {
            continue;
        }
        let mut node_ids = region_nodes.into_iter().collect::<Vec<_>>();
        node_ids.sort_unstable();
        let mut input_node_ids = inputs.into_iter().collect::<Vec<_>>();
        input_node_ids.sort_unstable();
        regions.push(TensorFusionRegion {
            output_node_id: root,
            node_ids,
            input_node_ids,
        });
    }
    regions
}

fn collect_fusion_region_nodes(
    nodes: &[TensorNode],
    users: &[Vec<TensorNodeId>],
    root: TensorNodeId,
    node_id: TensorNodeId,
    region_nodes: &mut HashSet<TensorNodeId>,
    inputs: &mut HashSet<TensorNodeId>,
) {
    let Some(node) = nodes.get(node_id) else {
        return;
    };
    if matches!(node.op, TensorOp::ScalarConstant { .. }) {
        return;
    }
    if !is_fusable_elementwise_compute_op(&node.op)
        || (node_id != root
            && (users[node_id].len() != 1
                || users[node_id]
                    .iter()
                    .any(|user| !is_fusable_elementwise_compute_op(&nodes[*user].op))))
    {
        inputs.insert(node_id);
        return;
    }
    if !region_nodes.insert(node_id) {
        return;
    }
    for input in tensor_op_inputs(&node.op) {
        collect_fusion_region_nodes(nodes, users, root, input, region_nodes, inputs);
    }
}

// IDs remain stable until final DCE; retired constants are unreachable tombstones.
fn release_folded_constant(nodes: &mut [TensorNode], id: usize, uses: usize, retained: bool) {
    if uses == 0 && !retained && matches!(nodes[id].op, TensorOp::Constant { .. }) {
        nodes[id].op = TensorOp::ScalarConstant { value: 0.0 };
    }
}

// Reverse rules for these operations need only cotangents and static node metadata.
// Every other rule conservatively retains its own primal and every operand.
fn tensor_reverse_retained_primals(
    nodes: &[TensorNode],
    outputs: &[(TensorNodeId, DynamicTensor)],
) -> Vec<TensorNodeId> {
    let roots = outputs
        .iter()
        .map(|(id, _)| *id)
        .filter(|id| *id < nodes.len())
        .collect::<Vec<_>>();
    let active = tensor_output_reachability(nodes, &roots);
    let mut retained = vec![false; nodes.len()];
    for &id in &roots {
        retained[id] = true;
    }
    for (id, node) in nodes.iter().enumerate() {
        if !active[id]
            || matches!(
                node.op,
                TensorOp::Input { .. }
                    | TensorOp::Constant { .. }
                    | TensorOp::ScalarConstant { .. }
                    | TensorOp::Cast { .. }
                    | TensorOp::Add { .. }
                    | TensorOp::Sub { .. }
                    | TensorOp::Greater { .. }
                    | TensorOp::Compare { .. }
                    | TensorOp::Sum { .. }
                    | TensorOp::SumAxis { .. }
                    | TensorOp::Mean { .. }
                    | TensorOp::MeanAxis { .. }
                    | TensorOp::Triangular { .. }
                    | TensorOp::Reshape { .. }
                    | TensorOp::Transpose { .. }
                    | TensorOp::Concat { .. }
                    | TensorOp::Slice { .. }
                    | TensorOp::PadSlice { .. }
                    | TensorOp::Gather { .. }
                    | TensorOp::ScatterAdd { .. }
                    | TensorOp::StopGradient { .. }
                    | TensorOp::CumSum { .. }
                    | TensorOp::Broadcast { .. }
            )
        {
            continue;
        }
        retained[id] = true;
        for input in tensor_op_inputs(&node.op) {
            retained[input] = true;
        }
    }
    retained
        .into_iter()
        .enumerate()
        .filter_map(|(id, keep)| keep.then_some(id))
        .collect()
}

fn tensor_output_reachability(nodes: &[TensorNode], outputs: &[TensorNodeId]) -> Vec<bool> {
    let mut reachable = vec![false; nodes.len()];
    let mut pending = outputs.to_vec();
    while let Some(id) = pending.pop() {
        if reachable[id] {
            continue;
        }
        reachable[id] = true;
        pending.extend(tensor_op_inputs(&nodes[id].op));
    }
    reachable
}

fn prune_unreachable_tensor_nodes(
    nodes: Vec<TensorNode>,
    outputs: &[TensorNodeId],
) -> Result<(Vec<TensorNode>, Vec<TensorNodeId>), String> {
    for &output in outputs {
        if output >= nodes.len() {
            return Err(format!("execution plan output node {output} is missing"));
        }
    }
    let reachable = tensor_output_reachability(&nodes, outputs);
    let reachable_count = reachable.iter().filter(|keep| **keep).count();
    let mut remap = TensorNodeRemap::new(nodes.len(), reachable_count);
    let mut compacted = Vec::with_capacity(reachable_count);
    for (old_id, node) in nodes.iter().enumerate() {
        if !reachable[old_id] {
            continue;
        }
        remap.insert(old_id, compacted.len());
        compacted.push(TensorNode {
            op: remap_tensor_op(&node.op, &|id| remap.get(id))?,
            shape: node.shape.clone(),
            dtype: node.dtype,
            weak: node.weak,
        });
    }
    let remapped_outputs = outputs
        .iter()
        .map(|output| {
            remap
                .get(*output)
                .ok_or_else(|| format!("execution plan output node {output} is unreachable"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((compacted, remapped_outputs))
}

fn build_tensor_buffer_plan(
    nodes: &[TensorNode],
    output_node_ids: &[TensorNodeId],
) -> Result<TensorBufferPlan, String> {
    let output_node_id = *output_node_ids
        .first()
        .ok_or_else(|| "tensor buffer plan has no output".to_string())?;
    let output_backing_node_id = tensor_storage_root(nodes, output_node_id)?;
    let retained_roots = output_node_ids
        .iter()
        .map(|output| tensor_storage_root(nodes, *output))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut remaining_uses = vec![0usize; nodes.len()];
    for node in nodes {
        for input in tensor_op_inputs(&node.op) {
            let root = tensor_storage_root(nodes, input)?;
            remaining_uses[root] = remaining_uses[root]
                .checked_add(1)
                .ok_or_else(|| "tensor buffer use count overflows usize".to_string())?;
        }
    }

    let mut free_slots = BTreeMap::<usize, Vec<usize>>::new();
    let mut slots = Vec::new();
    let mut node_slots = vec![None; nodes.len()];
    let mut node_aliases = vec![None; nodes.len()];
    for (node_id, node) in nodes.iter().enumerate() {
        if let TensorOp::Reshape { input } = &node.op {
            let root = tensor_storage_root(nodes, *input)?;
            node_slots[node_id] = node_slots[root];
            node_aliases[node_id] = Some(*input);
        } else if !matches!(node.op, TensorOp::Input { .. } | TensorOp::Constant { .. }) {
            // Inputs and array constants are bound storage, not planned slots: a constant is
            // read-only for the plan's lifetime, so its storage is never recycled.
            let count = element_count(&node.shape)?;
            let slot = free_slots
                .get_mut(&count)
                .and_then(|available| available.pop())
                .unwrap_or_else(|| {
                    let id = slots.len();
                    slots.push(TensorBufferSlot {
                        id,
                        element_count: count,
                    });
                    id
                });
            node_slots[node_id] = Some(slot);
        }

        for input in tensor_op_inputs(&node.op) {
            let root = tensor_storage_root(nodes, input)?;
            let uses = remaining_uses
                .get_mut(root)
                .ok_or_else(|| format!("tensor buffer input node {root} is missing"))?;
            *uses = uses
                .checked_sub(1)
                .ok_or_else(|| format!("tensor buffer input node {root} has invalid use count"))?;
            if *uses == 0 && !retained_roots.contains(&root) {
                if let Some(slot) = node_slots[root] {
                    let count = slots
                        .get(slot)
                        .ok_or_else(|| format!("tensor buffer slot {slot} is missing"))?
                        .element_count;
                    free_slots.entry(count).or_default().push(slot);
                }
            }
        }
    }
    Ok(TensorBufferPlan {
        slots,
        node_slots,
        node_aliases,
        output_backing_node_id,
    })
}

fn tensor_storage_root(
    nodes: &[TensorNode],
    node_id: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let mut root = node_id;
    loop {
        match &nodes
            .get(root)
            .ok_or_else(|| format!("tensor buffer node {root} is missing"))?
            .op
        {
            TensorOp::Reshape { input } => root = *input,
            _ => return Ok(root),
        }
    }
}

fn tensor_input_nodes(nodes: &[TensorNode]) -> HashMap<String, TensorNodeId> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(id, node)| {
            if let TensorOp::Input { name } = &node.op {
                Some((name.clone(), id))
            } else {
                None
            }
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PureTensorOpKey {
    operation: std::mem::Discriminant<TensorOp>,
    words: Box<[u64]>,
}

fn pure_tensor_op_cse_key(op: &TensorOp, shape: &[usize]) -> Option<PureTensorOpKey> {
    let mut arguments = [0; 4];
    let mut list = None;
    match op {
        TensorOp::Input { .. }
        | TensorOp::Constant { .. }
        | TensorOp::CholeskyAd { .. }
        | TensorOp::Cond { .. }
        | TensorOp::While { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. }
        | TensorOp::Custom { .. } => return None,
        TensorOp::ScalarConstant { value } => arguments[0] = value.to_bits(),
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Div { lhs, rhs }
        | TensorOp::Greater { lhs, rhs }
        | TensorOp::Matmul { lhs, rhs }
        | TensorOp::Solve { matrix: lhs, rhs } => {
            arguments[0] = *lhs as u64;
            arguments[1] = *rhs as u64;
        }
        TensorOp::BinaryMath { lhs, rhs, kind } => {
            arguments[0] = *lhs as u64;
            arguments[1] = *rhs as u64;
            arguments[2] = *kind as u64;
        }
        TensorOp::Compare { lhs, rhs, kind } => {
            arguments[0] = *lhs as u64;
            arguments[1] = *rhs as u64;
            arguments[2] = *kind as u64;
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => {
            arguments[0] = *condition as u64;
            arguments[1] = *on_true as u64;
            arguments[2] = *on_false as u64;
        }
        TensorOp::Cast { input }
        | TensorOp::Sum { input }
        | TensorOp::Sqrt { input }
        | TensorOp::Reshape { input }
        | TensorOp::Mean { input }
        | TensorOp::StopGradient { input }
        | TensorOp::Broadcast { input }
        | TensorOp::Cholesky { input } => arguments[0] = *input as u64,
        TensorOp::CumSum {
            input,
            axis,
            reverse,
        } => {
            arguments[0] = *input as u64;
            arguments[1] = *axis as u64;
            arguments[2] = u64::from(*reverse);
        }
        TensorOp::SumAxis { input, axis } | TensorOp::MeanAxis { input, axis } => {
            arguments[0] = *input as u64;
            arguments[1] = *axis as u64;
        }
        TensorOp::ExtremumAxis { input, axis, kind } => {
            arguments[0] = *input as u64;
            arguments[1] = *axis as u64;
            arguments[2] = *kind as u64;
        }
        TensorOp::Triangular { input, lower } => {
            arguments[0] = *input as u64;
            arguments[1] = u64::from(*lower);
        }
        TensorOp::Linalg { input, kind } => {
            arguments[0] = *input as u64;
            arguments[1] = *kind as u64;
        }
        TensorOp::UnaryMath { input, kind } => {
            arguments[0] = *input as u64;
            arguments[1] = *kind as u64;
        }
        TensorOp::SqrtDerivative { input, order }
        | TensorOp::Powi {
            input,
            exponent: order,
        } => {
            arguments[0] = *input as u64;
            arguments[1] = *order as u64;
        }
        TensorOp::Transpose { input, axes } => {
            arguments[0] = *input as u64;
            list = Some(axes.as_slice());
        }
        TensorOp::Concat { inputs, axis } => {
            arguments[0] = *axis as u64;
            list = Some(inputs.as_slice());
        }
        TensorOp::Slice {
            input,
            axis,
            start,
            length,
        } => {
            arguments = [*input as u64, *axis as u64, *start as u64, *length as u64];
        }
        TensorOp::PadSlice { input, axis, start } => {
            arguments[0] = *input as u64;
            arguments[1] = *axis as u64;
            arguments[2] = *start as u64;
        }
        TensorOp::Gather {
            input,
            indices,
            axis,
        } => {
            arguments[0] = *input as u64;
            arguments[1] = *axis as u64;
            list = Some(indices.as_ref());
        }
        TensorOp::ScatterAdd {
            base,
            updates,
            indices,
            axis,
        } => {
            arguments[0] = *base as u64;
            arguments[1] = *updates as u64;
            arguments[2] = *axis as u64;
            list = Some(indices.as_ref());
        }
    }
    let argument_count = match op {
        TensorOp::Slice { .. } => 4,
        TensorOp::Compare { .. }
        | TensorOp::BinaryMath { .. }
        | TensorOp::Where { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::ScatterAdd { .. }
        | TensorOp::ExtremumAxis { .. }
        | TensorOp::CumSum { .. } => 3,
        TensorOp::Gather { .. }
        | TensorOp::Add { .. }
        | TensorOp::Sub { .. }
        | TensorOp::Mul { .. }
        | TensorOp::Div { .. }
        | TensorOp::Greater { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Solve { .. }
        | TensorOp::Cholesky { .. }
        | TensorOp::CholeskyAd { .. }
        | TensorOp::UnaryMath { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Triangular { .. }
        | TensorOp::Linalg { .. }
        | TensorOp::SqrtDerivative { .. }
        | TensorOp::Powi { .. } => 2,
        _ => 1,
    };
    // Validated operand ids determine output shapes except for shape-changing ops
    // whose target shape is not encoded in the op. Avoid duplicating inferred shapes.
    let shape = if matches!(
        op,
        TensorOp::Reshape { .. } | TensorOp::Broadcast { .. } | TensorOp::PadSlice { .. }
    ) {
        shape
    } else {
        &[]
    };
    // Variable operand lists carry their length so they cannot collide with shape words.
    let list_words = list.map_or(0, |values| values.len() + 1);
    let mut words = Vec::with_capacity(argument_count + list_words + shape.len());
    words.extend_from_slice(&arguments[..argument_count]);
    if let Some(values) = list {
        words.push(values.len() as u64);
        words.extend(values.iter().map(|value| *value as u64));
    }
    words.extend(shape.iter().map(|value| *value as u64));
    Some(PureTensorOpKey {
        operation: std::mem::discriminant(op),
        words: words.into_boxed_slice(),
    })
}

/// The execution-group id of a node that shares one region execution with
/// its sibling results, if the op has one.
fn tensor_op_group_mut(op: &mut TensorOp) -> Option<&mut usize> {
    match op {
        TensorOp::ForiVjp { group, .. }
        | TensorOp::ForiVjpJvp { group, .. }
        | TensorOp::Scan { group, .. }
        | TensorOp::ScanVjp { group, .. }
        | TensorOp::ScanVjpJvp { group, .. }
        | TensorOp::Custom { group, .. } => Some(group),
        _ => None,
    }
}

// Dense indexing removes hash work without allocating a full table for sparse outputs.
enum TensorNodeRemap {
    Dense(Vec<TensorNodeId>),
    Sparse(HashMap<TensorNodeId, TensorNodeId>),
}

impl TensorNodeRemap {
    fn new(total: usize, reachable: usize) -> Self {
        if total <= reachable.saturating_mul(2) {
            Self::Dense(vec![usize::MAX; total])
        } else {
            Self::Sparse(HashMap::with_capacity(reachable))
        }
    }

    fn insert(&mut self, source: TensorNodeId, target: TensorNodeId) {
        match self {
            Self::Dense(values) => values[source] = target,
            Self::Sparse(values) => {
                values.insert(source, target);
            }
        }
    }

    fn get(&self, source: TensorNodeId) -> Option<TensorNodeId> {
        match self {
            Self::Dense(values) => values.get(source).copied().filter(|id| *id != usize::MAX),
            Self::Sparse(values) => values.get(&source).copied(),
        }
    }
}

fn remap_tensor_op(
    op: &TensorOp,
    remap: &impl Fn(TensorNodeId) -> Option<TensorNodeId>,
) -> Result<TensorOp, String> {
    let remap_node = |node_id: TensorNodeId| {
        remap(node_id).ok_or_else(|| format!("node {node_id} is missing from execution plan remap"))
    };
    match op {
        TensorOp::Input { name } => Ok(TensorOp::Input { name: name.clone() }),
        TensorOp::ScalarConstant { value } => Ok(TensorOp::ScalarConstant { value: *value }),
        TensorOp::Constant { value } => Ok(TensorOp::Constant {
            value: value.clone(),
        }),
        TensorOp::Cast { input } => Ok(TensorOp::Cast {
            input: remap_node(*input)?,
        }),
        TensorOp::Add { lhs, rhs } => Ok(TensorOp::Add {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Sub { lhs, rhs } => Ok(TensorOp::Sub {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Div { lhs, rhs } => Ok(TensorOp::Div {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Mul { lhs, rhs } => Ok(TensorOp::Mul {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Greater { lhs, rhs } => Ok(TensorOp::Greater {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Compare { lhs, rhs, kind } => Ok(TensorOp::Compare {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
            kind: *kind,
        }),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => Ok(TensorOp::Where {
            condition: remap_node(*condition)?,
            on_true: remap_node(*on_true)?,
            on_false: remap_node(*on_false)?,
        }),
        TensorOp::Cond {
            predicate,
            branches,
            captures,
        } => Ok(TensorOp::Cond {
            predicate: remap_node(*predicate)?,
            branches: branches.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
        }),
        TensorOp::ForiJvp {
            carry,
            carry_tangent,
            loop_plan,
            captures,
            tangent_captures,
        } => Ok(TensorOp::ForiJvp {
            carry: remap_node(*carry)?,
            carry_tangent: remap_node(*carry_tangent)?,
            loop_plan: loop_plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            tangent_captures: tangent_captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
        }),
        TensorOp::ForiVjp {
            carry,
            output_cotangent,
            loop_plan,
            captures,
            target,
            group,
        } => Ok(TensorOp::ForiVjp {
            carry: remap_node(*carry)?,
            output_cotangent: remap_node(*output_cotangent)?,
            loop_plan: loop_plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            target: target.clone(),
            group: *group,
        }),
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
        } => Ok(TensorOp::ScanVjpJvp {
            carry: remap_node(*carry)?,
            carry_tangent: remap_node(*carry_tangent)?,
            final_carry_cotangent: remap_node(*final_carry_cotangent)?,
            final_carry_cotangent_tangent: remap_node(*final_carry_cotangent_tangent)?,
            output_cotangent: remap_node(*output_cotangent)?,
            output_cotangent_tangent: remap_node(*output_cotangent_tangent)?,
            plan: plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            tangent_captures: tangent_captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            target: target.clone(),
            group: *group,
        }),
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
        } => Ok(TensorOp::ForiVjpJvp {
            carry: remap_node(*carry)?,
            carry_tangent: remap_node(*carry_tangent)?,
            output_cotangent: remap_node(*output_cotangent)?,
            output_cotangent_tangent: remap_node(*output_cotangent_tangent)?,
            plan: plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            tangent_captures: tangent_captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            target: target.clone(),
            group: *group,
        }),
        TensorOp::Fori {
            carry,
            loop_plan,
            captures,
        } => Ok(TensorOp::Fori {
            carry: remap_node(*carry)?,
            loop_plan: loop_plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
        }),
        TensorOp::While {
            carry,
            loop_plan,
            captures,
        } => Ok(TensorOp::While {
            carry: remap_node(*carry)?,
            loop_plan: loop_plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
        }),
        TensorOp::Scan {
            carry,
            scan_plan,
            captures,
            target,
            group,
        } => Ok(TensorOp::Scan {
            carry: remap_node(*carry)?,
            scan_plan: scan_plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            target: *target,
            group: *group,
        }),
        TensorOp::ScanVjp {
            carry,
            final_carry_cotangent,
            output_cotangent,
            scan_plan,
            captures,
            target,
            group,
        } => Ok(TensorOp::ScanVjp {
            carry: remap_node(*carry)?,
            final_carry_cotangent: remap_node(*final_carry_cotangent)?,
            output_cotangent: remap_node(*output_cotangent)?,
            scan_plan: scan_plan.clone(),
            captures: captures
                .iter()
                .map(|(name, node_id)| Ok((name.clone(), remap_node(*node_id)?)))
                .collect::<Result<Vec<_>, String>>()?,
            target: target.clone(),
            group: *group,
        }),
        TensorOp::Sum { input } => Ok(TensorOp::Sum {
            input: remap_node(*input)?,
        }),
        TensorOp::SumAxis { input, axis } => Ok(TensorOp::SumAxis {
            input: remap_node(*input)?,
            axis: *axis,
        }),
        TensorOp::ExtremumAxis { input, axis, kind } => Ok(TensorOp::ExtremumAxis {
            input: remap_node(*input)?,
            axis: *axis,
            kind: *kind,
        }),
        TensorOp::Matmul { lhs, rhs } => Ok(TensorOp::Matmul {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Solve { matrix, rhs } => Ok(TensorOp::Solve {
            matrix: remap_node(*matrix)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Linalg { input, kind } => Ok(TensorOp::Linalg {
            input: remap_node(*input)?,
            kind: *kind,
        }),
        TensorOp::Cholesky { input } => Ok(TensorOp::Cholesky {
            input: remap_node(*input)?,
        }),
        TensorOp::CholeskyAd { inputs, kind } => Ok(TensorOp::CholeskyAd {
            inputs: inputs
                .iter()
                .map(|id| remap_node(*id))
                .collect::<Result<Vec<_>, _>>()?,
            kind: *kind,
        }),
        TensorOp::Triangular { input, lower } => Ok(TensorOp::Triangular {
            input: remap_node(*input)?,
            lower: *lower,
        }),
        TensorOp::Sqrt { input } => Ok(TensorOp::Sqrt {
            input: remap_node(*input)?,
        }),
        TensorOp::SqrtDerivative { input, order } => Ok(TensorOp::SqrtDerivative {
            input: remap_node(*input)?,
            order: *order,
        }),
        TensorOp::Reshape { input } => Ok(TensorOp::Reshape {
            input: remap_node(*input)?,
        }),
        TensorOp::Mean { input } => Ok(TensorOp::Mean {
            input: remap_node(*input)?,
        }),
        TensorOp::MeanAxis { input, axis } => Ok(TensorOp::MeanAxis {
            input: remap_node(*input)?,
            axis: *axis,
        }),
        TensorOp::Powi { input, exponent } => Ok(TensorOp::Powi {
            input: remap_node(*input)?,
            exponent: *exponent,
        }),
        TensorOp::BinaryMath { lhs, rhs, kind } => Ok(TensorOp::BinaryMath {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
            kind: *kind,
        }),
        TensorOp::Transpose { input, axes } => Ok(TensorOp::Transpose {
            input: remap_node(*input)?,
            axes: axes.clone(),
        }),
        TensorOp::UnaryMath { input, kind } => Ok(TensorOp::UnaryMath {
            input: remap_node(*input)?,
            kind: *kind,
        }),
        TensorOp::StopGradient { input } => Ok(TensorOp::StopGradient {
            input: remap_node(*input)?,
        }),
        TensorOp::CumSum {
            input,
            axis,
            reverse,
        } => Ok(TensorOp::CumSum {
            input: remap_node(*input)?,
            axis: *axis,
            reverse: *reverse,
        }),
        TensorOp::Concat { inputs, axis } => Ok(TensorOp::Concat {
            inputs: inputs
                .iter()
                .map(|input| remap_node(*input))
                .collect::<Result<Vec<_>, _>>()?,
            axis: *axis,
        }),
        TensorOp::Slice {
            input,
            axis,
            start,
            length,
        } => Ok(TensorOp::Slice {
            input: remap_node(*input)?,
            axis: *axis,
            start: *start,
            length: *length,
        }),
        TensorOp::PadSlice { input, axis, start } => Ok(TensorOp::PadSlice {
            input: remap_node(*input)?,
            axis: *axis,
            start: *start,
        }),
        TensorOp::Gather {
            input,
            indices,
            axis,
        } => Ok(TensorOp::Gather {
            input: remap_node(*input)?,
            indices: indices.clone(),
            axis: *axis,
        }),
        TensorOp::ScatterAdd {
            base,
            updates,
            indices,
            axis,
        } => Ok(TensorOp::ScatterAdd {
            base: remap_node(*base)?,
            updates: remap_node(*updates)?,
            indices: indices.clone(),
            axis: *axis,
        }),
        TensorOp::Broadcast { input } => Ok(TensorOp::Broadcast {
            input: remap_node(*input)?,
        }),
        TensorOp::Custom {
            value,
            operands,
            rule,
            output,
            group,
        } => Ok(TensorOp::Custom {
            value: remap_node(*value)?,
            operands: operands
                .iter()
                .map(|operand| remap_node(*operand))
                .collect::<Result<_, _>>()?,
            rule: rule.clone(),
            output: *output,
            group: *group,
        }),
    }
}

fn accumulate(slot: &mut Option<DynamicTensor>, contribution: DynamicTensor) -> Result<(), String> {
    match slot {
        Some(existing) => {
            existing.add_assign(&contribution)?;
        }
        None => *slot = Some(contribution),
    }
    Ok(())
}

fn input_value(
    inputs: &BTreeMap<String, DynamicTensor>,
    name: &str,
    shape: &[usize],
) -> Result<DynamicTensor, String> {
    let input = inputs
        .get(name)
        .ok_or_else(|| format!("missing input {name:?}"))?;
    if input.shape != shape {
        return Err(format!(
            "input {name:?} has shape {:?}, expected {:?}",
            input.shape, shape
        ));
    }
    Ok(input.clone())
}

fn input_tangent_or_zero(
    tangents: &BTreeMap<String, DynamicTensor>,
    name: &str,
    shape: &[usize],
) -> Result<DynamicTensor, String> {
    match tangents.get(name) {
        Some(tangent) => {
            if tangent.shape != shape {
                return Err(format!(
                    "input tangent {name:?} has shape {:?}, expected {:?}",
                    tangent.shape, shape
                ));
            }
            Ok(tangent.clone())
        }
        None => DynamicTensor::filled(shape.to_vec(), 0.0),
    }
}

/// The `order`-th derivative of `sqrt` (order 0 is `sqrt` itself), the CPU
/// reference that the CUDA and MLX lowerings reproduce.
///
/// Every order is `0` at `+-0`, the zero subgradient that keeps higher-order
/// AD away from `0 * inf` at the origin. A negative input, `-inf` included, is
/// outside the domain and gives NaN as IEEE `sqrt` does; the explicit branch
/// is needed because `powf(-inf, y)` is `inf` or `0` rather than NaN. NaN
/// propagates through `powf`.
fn sqrt_derivative_value(value: f64, order: u32) -> f64 {
    if value == 0.0 {
        0.0
    } else if value < 0.0 {
        f64::NAN
    } else {
        sqrt_derivative_coefficient(order) * value.powf(0.5 - order as f64)
    }
}

/// CUDA C for [`sqrt_derivative_value`] applied to the `float` expression
/// `input`. `powf` alone would give `inf` or `0` at `-inf`, so negative
/// inputs select a quiet NaN explicitly.
fn cuda_sqrt_derivative_expression(input: &str, order: u32) -> String {
    let coefficient = cuda_scalar_literal(sqrt_derivative_coefficient(order));
    let exponent = cuda_scalar_literal(0.5 - order as f64);
    format!(
        "(({input} == 0.0f) ? 0.0f : (({input} < 0.0f) ? __int_as_float(0x7fc00000) : \
         ({coefficient} * powf({input}, {exponent}))))"
    )
}

fn sqrt_derivative_coefficient(order: u32) -> f64 {
    (0..order).fold(1.0, |coefficient, index| coefficient * (0.5 - index as f64))
}

/// `[batch_size, *shape]`: the shape of a mapped node under `vmap`.
fn batched_shape(batch_size: usize, shape: &[usize]) -> Vec<usize> {
    std::iter::once(batch_size)
        .chain(shape.iter().copied())
        .collect()
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.contains(&0) {
        return Err("tensor extents must be greater than zero".to_string());
    }
    shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| "tensor element count overflows usize".to_string())
    })
}

fn concat_shape(shapes: &[&[usize]], axis: usize) -> Result<Vec<usize>, String> {
    let Some(first) = shapes.first() else {
        return Err("concat requires at least one input".to_string());
    };
    if axis >= first.len() {
        return Err(format!(
            "axis {axis} is out of bounds for rank {}",
            first.len()
        ));
    }
    let mut shape = first.to_vec();
    for candidate in &shapes[1..] {
        if candidate.len() != shape.len()
            || candidate
                .iter()
                .enumerate()
                .any(|(index, extent)| index != axis && *extent != shape[index])
        {
            return Err(format!(
                "cannot concatenate shapes {:?} and {:?} along axis {axis}",
                first, candidate
            ));
        }
        shape[axis] = shape[axis]
            .checked_add(candidate[axis])
            .ok_or_else(|| "concatenated tensor extent overflows usize".to_string())?;
    }
    Ok(shape)
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (1..shape.len()).rev() {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    strides
}

fn normalize_axis(axis: isize, rank: usize) -> Result<usize, String> {
    let normalized = if axis < 0 {
        axis.checked_add(rank as isize)
            .ok_or_else(|| format!("axis {axis} is out of bounds for rank {rank}"))?
    } else {
        axis
    };
    usize::try_from(normalized)
        .ok()
        .filter(|axis| *axis < rank)
        .ok_or_else(|| format!("axis {axis} is out of bounds for rank {rank}"))
}

fn reduced_shape(shape: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    if axis >= shape.len() {
        return Err(format!(
            "axis {axis} is out of bounds for tensor shape {:?}",
            shape
        ));
    }
    let mut reduced = shape.to_vec();
    reduced.remove(axis);
    Ok(reduced)
}

fn normalize_permutation(axes: Option<Vec<isize>>, rank: usize) -> Result<Vec<usize>, String> {
    let axes = axes.unwrap_or_else(|| (0..rank).rev().map(|axis| axis as isize).collect());
    if axes.len() != rank {
        return Err(format!(
            "transpose axes must have length {rank}, got {}",
            axes.len()
        ));
    }
    let axes = axes
        .into_iter()
        .map(|axis| normalize_axis(axis, rank))
        .collect::<Result<Vec<_>, _>>()?;
    validate_permutation(&axes, rank)?;
    Ok(axes)
}

fn validate_permutation(axes: &[usize], rank: usize) -> Result<(), String> {
    if axes.len() != rank {
        return Err(format!(
            "transpose axes must have length {rank}, got {}",
            axes.len()
        ));
    }
    let mut seen = vec![false; rank];
    for axis in axes {
        if *axis >= rank || std::mem::replace(&mut seen[*axis], true) {
            return Err(format!(
                "transpose axes {:?} are not a permutation of 0..{rank}",
                axes
            ));
        }
    }
    Ok(())
}

fn inverse_permutation(axes: &[usize]) -> Result<Vec<usize>, String> {
    validate_permutation(axes, axes.len())?;
    let mut inverse = vec![0; axes.len()];
    for (output_axis, input_axis) in axes.iter().enumerate() {
        inverse[*input_axis] = output_axis;
    }
    Ok(inverse)
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_extent = lhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let rhs_extent = rhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let extent = if lhs_extent == rhs_extent {
            lhs_extent
        } else if lhs_extent == 1 {
            rhs_extent
        } else if rhs_extent == 1 {
            lhs_extent
        } else {
            return Err(format!(
                "cannot broadcast tensor shapes {:?} and {:?}",
                lhs, rhs
            ));
        };
        shape.push(extent);
    }
    shape.reverse();
    Ok(shape)
}

fn matmul_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    if lhs.len() < 2 || rhs.len() < 2 {
        return Err(format!(
            "matmul requires tensors with at least two dimensions, got {:?} and {:?}",
            lhs, rhs
        ));
    }
    let lhs_inner = lhs[lhs.len() - 1];
    let rhs_inner = rhs[rhs.len() - 2];
    if lhs_inner != rhs_inner {
        return Err(format!(
            "cannot matmul tensor shapes {:?} and {:?}: inner dimensions {lhs_inner} and {rhs_inner} differ",
            lhs, rhs
        ));
    }

    let mut shape = broadcast_shape(&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2])?;
    shape.push(lhs[lhs.len() - 2]);
    shape.push(rhs[rhs.len() - 1]);
    Ok(shape)
}

/// The result shape of `Solve`: a coefficient stack `[..., n, n]` and a
/// right-hand-side stack `[..., n, k]` with the same leading batch axes give
/// `[..., n, k]`. Batch axes do not broadcast in the IR, so every backend and
/// derivative rule sees one matrix per right-hand-side block; front ends
/// broadcast explicitly with `broadcast`, whose VJP sums the expanded axes.
fn solve_shape(matrix: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    let rank = matrix.len();
    if rank < 2 || rhs.len() != rank {
        return Err(format!(
            "solve requires matrix and right-hand side tensors of the same rank, at least two, \
             got {:?} and {:?}",
            matrix, rhs
        ));
    }
    if matrix[rank - 2] != matrix[rank - 1] {
        return Err(format!(
            "solve requires square coefficient matrices, got {:?}",
            matrix
        ));
    }
    if matrix[rank - 2] != rhs[rank - 2] {
        return Err(format!(
            "solve requires matrix shape {:?} and right-hand side shape {:?} to agree on rows",
            matrix, rhs
        ));
    }
    if matrix[..rank - 2] != rhs[..rank - 2] {
        return Err(format!(
            "solve requires matrix shape {:?} and right-hand side shape {:?} to have the same \
             batch axes",
            matrix, rhs
        ));
    }
    Ok(rhs.to_vec())
}

// Advance row-major broadcast offsets with stack-only operand metadata. Only
// wrapped axes propagate a carry, so ordinary elements avoid full rank decoding.
// Inlining specializes the one-to-three operand loop and removes per-element calls.
#[inline(always)]
fn advance_broadcast_offsets<const N: usize>(
    mut next: usize,
    shape: &[usize],
    operands: [(&[usize], &[usize]); N],
    offsets: &mut [usize; N],
) {
    for axis in (0..shape.len()).rev() {
        let wrapped = next.is_multiple_of(shape[axis]);
        for (offset, (input_shape, strides)) in offsets.iter_mut().zip(operands) {
            let rank_offset = shape.len() - input_shape.len();
            if axis >= rank_offset && input_shape[axis - rank_offset] != 1 {
                let step = strides[axis - rank_offset];
                if wrapped {
                    *offset -= (shape[axis] - 1) * step;
                } else {
                    *offset += step;
                }
            }
        }
        if !wrapped {
            break;
        }
        next /= shape[axis];
    }
}

fn broadcast_offset(
    output_index: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let mut remaining = output_index;
    let mut offset = 0;
    let rank_offset = output_shape.len() - input_shape.len();
    for axis in (0..output_shape.len()).rev() {
        let coordinate = remaining % output_shape[axis];
        remaining /= output_shape[axis];
        if axis >= rank_offset {
            let input_axis = axis - rank_offset;
            if input_shape[input_axis] != 1 {
                offset += coordinate * input_strides[input_axis];
            }
        }
    }
    offset
}

/// MLIR-style tensor type; `Bool` prints as `i1` like StableHLO predicates.
/// An index list for IR text, abbreviated after a few entries so that large
/// gathers keep one readable line per node.
fn format_index_list(indices: &[usize]) -> String {
    const SHOWN: usize = 8;
    let shown = indices
        .iter()
        .take(SHOWN)
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if indices.len() > SHOWN {
        format!("[{shown}, ...] ({} total)", indices.len())
    } else {
        format!("[{shown}]")
    }
}

fn format_tensor_type(shape: &[usize], dtype: TensorDType) -> String {
    let dtype = match dtype {
        TensorDType::Bool => "i1".to_string(),
        dtype => dtype.to_string(),
    };
    if shape.is_empty() {
        return format!("tensor<{dtype}>");
    }
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x");
    format!("tensor<{dimensions}x{dtype}>")
}

#[cfg(test)]
mod triangular_solve_tests {
    use super::*;

    #[test]
    fn triangular_substitution_matches_pivoted_lu_and_residuals() -> Result<(), String> {
        for n in [1, 3, 17] {
            let mut data = vec![0.0; n * n];
            for row in 0..n {
                for column in 0..n {
                    data[row * n + column] = if row == column {
                        2.0 + row as f64
                    } else {
                        ((row * 7 + column * 3) % 11) as f64 / 7.0 - 0.5
                    };
                }
            }
            // Force partial pivoting in the old lower-triangular route.
            if n > 1 {
                data[n] = 3.5;
            }
            let original = DynamicTensor::new(vec![n, n], data)?;
            let expected = DynamicTensor::new(
                vec![n, 3],
                (0..n * 3).map(|index| index as f64 / 17.0 - 1.0).collect(),
            )?;
            for lower in [false, true] {
                for transpose in [false, true] {
                    let projected = original.triangular(lower)?;
                    let matrix = if transpose {
                        projected.transpose_last_two()?
                    } else {
                        projected
                    };
                    let rhs = matrix.matmul(&expected)?;
                    assert!(matrix.finite_triangular_solution(&rhs).is_some());
                    let result = matrix.solve(&rhs)?;
                    let previous = matrix.solve_lu(&rhs)?;
                    for (actual, old) in result.data.iter().zip(previous.data.iter()) {
                        assert!((actual - old).abs() < 1e-11 * old.abs().max(1.0));
                    }
                    for (actual, expected) in result.data.iter().zip(expected.data.iter()) {
                        assert!((actual - expected).abs() < 1e-11 * expected.abs().max(1.0));
                    }
                    for (actual, rhs) in matrix.matmul(&result)?.data.iter().zip(rhs.data.iter()) {
                        assert!((actual - rhs).abs() < 1e-11 * rhs.abs().max(1.0));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn triangular_substitution_preserves_exceptional_lu_fallbacks() -> Result<(), String> {
        let cases = [
            (vec![2.0, 1.0, 1.0, 2.0], vec![1.0, 2.0]),
            (vec![0.0, 0.0, 1.0, 2.0], vec![1.0, 2.0]),
            (vec![1.0, 0.0, f64::NAN, 2.0], vec![1.0, 2.0]),
            (vec![1.0, 0.0, 1.0, 2.0], vec![f64::INFINITY, 2.0]),
            (vec![1e-320, 0.0, 1.0, 2.0], vec![f64::MAX, 2.0]),
        ];
        for (matrix, rhs) in cases {
            let matrix = DynamicTensor::new(vec![2, 2], matrix)?;
            let rhs = DynamicTensor::new(vec![2, 1], rhs)?;
            assert!(matrix.finite_triangular_solution(&rhs).is_none());
            match (matrix.solve(&rhs), matrix.solve_lu(&rhs)) {
                (Ok(actual), Ok(old)) => {
                    for (actual, old) in actual.data.iter().zip(old.data.iter()) {
                        assert!(
                            actual.to_bits() == old.to_bits() || (actual.is_nan() && old.is_nan())
                        );
                    }
                }
                (Err(actual), Err(old)) => assert_eq!(actual, old),
                _ => panic!("fallback changed LU success/error behavior"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod cpu_memory_regressions {
    use super::*;

    #[test]
    fn shared_elementwise_dag_evaluates_each_node_once() -> Result<(), String> {
        let mut graph = TensorIr::new();
        let x = graph.input("x", vec![2])?;
        let mut output = x;
        // A recursive tree walk would perform more than one trillion child visits.
        for _ in 0..40 {
            output = graph.add(output, output)?;
        }
        let plan = graph.compile_cpu(output)?;
        assert!(plan.uses_fused_elementwise_kernel());
        let inputs = BTreeMap::from([("x".into(), DynamicTensor::new(vec![2], vec![1.0, -2.0])?)]);
        assert_eq!(
            plan.evaluate(&inputs)?.data().as_ref(),
            &[2.0f64.powi(40), -2.0f64.powi(41)]
        );
        Ok(())
    }

    #[test]
    fn frozen_forward_chain_retains_only_the_live_frontier() -> Result<(), String> {
        const N: usize = 128;
        let mut graph = TensorIr::new();
        let x = graph.input("x", vec![N])?;
        let early = graph.sin(x)?;
        let mut output = early;
        for _ in 0..100 {
            output = graph.sin(output)?;
        }
        let loss = graph.sum(output)?;
        let (plan, _) = graph.compile_cpu_many(&[early, output, loss, early])?;
        let inputs = BTreeMap::from([("x".into(), DynamicTensor::filled(vec![N], 0.2)?)]);
        CPU_FORWARD_PEAK_ELEMENTS.with(|peak| peak.set(0));
        let values = plan.evaluate_many(&inputs)?;
        // Unary last-use execution moves the predecessor buffer. The peak is the two
        // requested vectors plus their scalar reduction, independent of chain depth.
        let peak = CPU_FORWARD_PEAK_ELEMENTS.with(|peak| peak.get());
        assert_eq!(peak, 2 * N + 1);
        assert_eq!(values[0], values[3]);
        assert_eq!(values[2].data()[0], values[1].data().iter().sum::<f64>());
        let retained = TensorIr::evaluate_tensor_nodes_with_outputs(
            &plan.nodes,
            &inputs,
            Some(&plan.output_node_ids),
        )?;
        assert_eq!(retained.iter().flatten().count(), 3);
        // Reverse consumers still obtain the full tape from the unchanged entry point.
        assert_eq!(
            TensorIr::evaluate_tensor_nodes(&plan.nodes, &inputs)?.len(),
            plan.node_count()
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "solve_replay_tests.rs"]
mod solve_replay_tests;

#[cfg(test)]
#[path = "cpu_kernel_indexing_tests.rs"]
mod cpu_kernel_indexing_tests;

#[cfg(test)]
mod structural_key_tests {
    use super::*;

    #[test]
    fn keys_preserve_scalar_bits_operation_shape_and_variable_boundaries() {
        let key = |op, shape: &[usize]| pure_tensor_op_cse_key(&op, shape).unwrap();
        assert_ne!(
            key(TensorOp::ScalarConstant { value: 0.0 }, &[]),
            key(TensorOp::ScalarConstant { value: -0.0 }, &[])
        );
        assert_ne!(
            key(
                TensorOp::ScalarConstant {
                    value: f64::from_bits(0x7ff8000000000001)
                },
                &[]
            ),
            key(
                TensorOp::ScalarConstant {
                    value: f64::from_bits(0x7ff8000000000002)
                },
                &[]
            )
        );
        assert_ne!(
            key(
                TensorOp::UnaryMath {
                    input: 1,
                    kind: UnaryMathKind::Sin
                },
                &[2]
            ),
            key(
                TensorOp::UnaryMath {
                    input: 1,
                    kind: UnaryMathKind::Cos
                },
                &[2]
            )
        );
        assert_ne!(
            key(TensorOp::Reshape { input: 1 }, &[2, 3]),
            key(TensorOp::Reshape { input: 1 }, &[6])
        );
        assert_ne!(
            key(
                TensorOp::Concat {
                    inputs: vec![1],
                    axis: 0
                },
                &[2, 3]
            ),
            key(
                TensorOp::Concat {
                    inputs: vec![1, 2],
                    axis: 0
                },
                &[3]
            )
        );
        assert_eq!(
            key(
                TensorOp::Slice {
                    input: 1,
                    axis: 0,
                    start: 0,
                    length: 2
                },
                &[2]
            ),
            key(
                TensorOp::Slice {
                    input: 1,
                    axis: 0,
                    start: 0,
                    length: 2
                },
                &[2]
            )
        );
        assert!(std::mem::size_of::<PureTensorOpKey>() <= 32);
    }
}

#[cfg(test)]
mod adaptive_remap_tests {
    use super::*;
    #[test]
    fn sparse_source_graphs_do_not_allocate_dense_id_tables() {
        let mut sparse = TensorNodeRemap::new(1_000_000, 2);
        assert!(matches!(sparse, TensorNodeRemap::Sparse(_)));
        sparse.insert(999_999, 1);
        assert_eq!(sparse.get(999_999), Some(1));
        assert_eq!(sparse.get(0), None);
        let mut dense = TensorNodeRemap::new(8, 6);
        assert!(matches!(dense, TensorNodeRemap::Dense(_)));
        dense.insert(3, 0);
        assert_eq!(dense.get(3), Some(0));
        assert_eq!(dense.get(4), None);
    }
}

#[cfg(test)]
mod binary_reuse_tests {
    use super::*;

    #[test]
    fn reused_binary_buffers_match_reference_bits_and_preserve_aliases() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            for scalar in [false, true] {
                for kind in ["add", "sub", "mul", "div"] {
                    let original = DynamicTensor::with_dtype(
                        vec![4],
                        vec![-0.0, 0.1, f64::NAN, f64::INFINITY],
                        dtype,
                    )
                    .unwrap();
                    let right = if scalar {
                        DynamicTensor::with_dtype(vec![], vec![2.0], dtype).unwrap()
                    } else {
                        DynamicTensor::with_dtype(
                            vec![4],
                            vec![1.0, -1.0, 3.0, f64::INFINITY],
                            dtype,
                        )
                        .unwrap()
                    };
                    let expected = match kind {
                        "add" => original.add(&right),
                        "sub" => original.sub(&right),
                        "mul" => original.mul(&right),
                        _ => original.div(&right),
                    }
                    .unwrap()
                    .into_dtype(dtype);
                    let alias = original.clone();
                    let mut graph = TensorIr::new();
                    let lhs = graph.input_typed("lhs", vec![4], dtype).unwrap();
                    let rhs = graph
                        .input_typed("rhs", right.shape.clone(), dtype)
                        .unwrap();
                    let output = match kind {
                        "add" => graph.add(lhs, rhs),
                        "sub" => graph.sub(lhs, rhs),
                        "mul" => graph.mul(lhs, rhs),
                        _ => graph.div(lhs, rhs),
                    }
                    .unwrap();
                    let uses = tensor_forward_last_uses(&graph.nodes, &[output]);
                    let mut values = vec![Some(original), Some(right)];
                    let actual =
                        reuse_forward_binary(&graph.nodes[output], &mut values, Some(&uses))
                            .unwrap()
                            .unwrap()
                            .into_dtype(dtype);
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
                    assert_eq!(alias.storage().get(0).to_bits(), (-0.0f64).to_bits());
                    assert_eq!(alias.storage().get(1).to_bits(), dtype.round(0.1).to_bits());
                }
            }
            let mut graph = TensorIr::new();
            let input = graph.input_typed("x", vec![2], dtype).unwrap();
            let output = graph.mul(input, input).unwrap();
            let mut values = vec![Some(
                DynamicTensor::with_dtype(vec![2], vec![-0.0, 0.1], dtype).unwrap(),
            )];
            let expected = values[0]
                .as_ref()
                .unwrap()
                .mul(values[0].as_ref().unwrap())
                .unwrap()
                .into_dtype(dtype);
            let uses = tensor_forward_last_uses(&graph.nodes, &[output]);
            let actual = reuse_forward_binary(&graph.nodes[output], &mut values, Some(&uses))
                .unwrap()
                .unwrap()
                .into_dtype(dtype);
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
            let division = graph.div(input, input).unwrap();
            let mut values = vec![Some(
                DynamicTensor::with_dtype(vec![2], vec![-0.0, 0.1], dtype).unwrap(),
            )];
            let uses = tensor_forward_last_uses(&graph.nodes, &[division]);
            assert!(
                reuse_forward_binary(&graph.nodes[division], &mut values, Some(&uses))
                    .unwrap()
                    .is_none()
            );
            assert!(values[0].is_some());
        }
    }
}

#[cfg(test)]
mod broadcast_carry_tests {
    use super::*;

    fn reference(lhs: &DynamicTensor, rhs: &DynamicTensor) -> Result<DynamicTensor, String> {
        let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
        let count = element_count(&shape)?;
        let lhs_strides = contiguous_strides(&lhs.shape);
        let rhs_strides = contiguous_strides(&rhs.shape);
        let mut data = Vec::with_capacity(count);
        for index in 0..count {
            let left = broadcast_offset(index, &shape, &lhs.shape, &lhs_strides);
            let right = broadcast_offset(index, &shape, &rhs.shape, &rhs_strides);
            data.push(lhs.data.get(left) - rhs.data.get(right));
        }
        DynamicTensor::new(shape, data)
    }

    fn fixture(shape: Vec<usize>, dtype: TensorDType) -> DynamicTensor {
        let count = element_count(&shape).unwrap();
        DynamicTensor::with_dtype(
            shape,
            (0..count)
                .map(|index| match index % 7 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f64::INFINITY,
                    3 => f64::NAN,
                    _ => index as f64 * 0.13 - 0.7,
                })
                .collect(),
            dtype,
        )
        .unwrap()
    }

    #[test]
    fn carried_elementwise_preserves_f64_arithmetic_and_shapes() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            for (lhs_shape, rhs_shape) in [
                (vec![3, 1], vec![1, 5]),
                (vec![2, 1, 4, 1], vec![3, 1, 5]),
                (vec![2, 1, 1, 1, 3, 1], vec![1, 4]),
                (vec![2, 3], vec![3]),
                (vec![2, 3], vec![4]),
            ] {
                let lhs = fixture(lhs_shape, dtype);
                let rhs = fixture(rhs_shape, dtype);
                match (lhs.sub(&rhs), reference(&lhs, &rhs)) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(actual.shape, expected.shape);
                        assert_eq!(actual.dtype, TensorDType::F64);
                        assert_eq!(
                            actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                            expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
                        );
                    }
                    (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                    pair => panic!("different broadcast outcomes: {pair:?}"),
                }
            }
        }
    }

    #[test]
    #[ignore = "complete core broadcast release timing probe"]
    fn profile_broadcast_carry() {
        use std::time::Instant;
        for (lhs_shape, rhs_shape) in [
            (vec![4, 1], vec![1, 4]),
            (vec![256, 1], vec![1, 256]),
            (vec![16, 1, 16, 1], vec![1, 16, 1, 16]),
            (vec![4, 1, 4, 1, 4, 1, 4, 1], vec![1, 4, 1, 4, 1, 4, 1, 4]),
        ] {
            let rank = lhs_shape.len();
            let elements =
                element_count(&broadcast_shape(&lhs_shape, &rhs_shape).unwrap()).unwrap();
            let lhs = fixture(lhs_shape, TensorDType::F32);
            let rhs = fixture(rhs_shape, TensorDType::F32);
            let modes = if std::env::var_os("QUABLA_BROADCAST_CARRY_FIRST").is_some() {
                [false, true]
            } else {
                [true, false]
            };
            for old in modes {
                let mut times = Vec::new();
                for _ in 0..15 {
                    let start = Instant::now();
                    let result = if old {
                        reference(&lhs, &rhs)
                    } else {
                        lhs.sub(&rhs)
                    }
                    .unwrap();
                    std::hint::black_box(result);
                    times.push(start.elapsed().as_nanos());
                }
                times.sort();
                println!("{{\"rank\":{rank},\"elements\":{elements},\"reference\":{old},\"median_ns\":{}}}", times[7]);
            }
        }
    }
    fn reference_where(
        mask: &DynamicTensor,
        on_true: &DynamicTensor,
        on_false: &DynamicTensor,
    ) -> Result<DynamicTensor, String> {
        let shape = broadcast_shape(
            &broadcast_shape(&mask.shape, &on_true.shape)?,
            &on_false.shape,
        )?;
        let strides = [
            contiguous_strides(&mask.shape),
            contiguous_strides(&on_true.shape),
            contiguous_strides(&on_false.shape),
        ];
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for index in 0..element_count(&shape)? {
            let m = broadcast_offset(index, &shape, &mask.shape, &strides[0]);
            let t = broadcast_offset(index, &shape, &on_true.shape, &strides[1]);
            let f = broadcast_offset(index, &shape, &on_false.shape, &strides[2]);
            data.push(if mask.data.get(m) != 0.0 {
                on_true.data.get(t)
            } else {
                on_false.data.get(f)
            });
        }
        DynamicTensor::new(shape, data)
    }

    fn reference_broadcast(
        input: &DynamicTensor,
        target: &[usize],
    ) -> Result<DynamicTensor, String> {
        if broadcast_shape(&input.shape, target)? != target {
            return Err(format!(
                "cannot broadcast tensor shape {:?} to {:?}",
                input.shape, target
            ));
        }
        let strides = contiguous_strides(&input.shape);
        let data = (0..element_count(target)?)
            .map(|index| {
                input
                    .data
                    .get(broadcast_offset(index, target, &input.shape, &strides))
            })
            .collect();
        DynamicTensor::new(target.to_vec(), data)
    }

    #[test]
    fn selection_and_unary_broadcast_preserve_bits_dtype_and_errors() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let mask =
                DynamicTensor::with_dtype(vec![2, 1, 1, 1], vec![0.0, 1.0], TensorDType::Bool)
                    .unwrap();
            let on_true = fixture(vec![1, 3, 1, 4], dtype);
            let on_false = fixture(vec![2, 1, 5, 1], dtype);
            let actual = mask.where_select(&on_true, &on_false).unwrap();
            let expected = reference_where(&mask, &on_true, &on_false).unwrap();
            assert_eq!(actual.dtype, expected.dtype);
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(
                actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
            );
            let input = fixture(vec![3, 1, 4], dtype);
            for target in [vec![2, 3, 5, 4], vec![3, 5, 4], vec![2, 3, 5, 7]] {
                match (
                    input.broadcast_to_shape(&target),
                    reference_broadcast(&input, &target),
                ) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(actual.dtype, expected.dtype);
                        assert_eq!(actual.shape, expected.shape);
                        assert_eq!(
                            actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                            expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
                        );
                    }
                    (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                    pair => panic!("different broadcast outcomes: {pair:?}"),
                }
            }
            let mask = fixture(vec![7], TensorDType::Bool);
            assert_eq!(
                mask.where_select(&on_true, &on_false).unwrap_err(),
                reference_where(&mask, &on_true, &on_false).unwrap_err()
            );
        }
        // A legacy NaN mask is nonzero; unselected NaNs remain unobserved.
        let mask = fixture(vec![4, 1], TensorDType::F64);
        let on_true = fixture(vec![1, 7], TensorDType::F64);
        let on_false = fixture(vec![4, 1], TensorDType::F64);
        let actual = mask.where_select(&on_true, &on_false).unwrap();
        let expected = reference_where(&mask, &on_true, &on_false).unwrap();
        assert_eq!(
            actual.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
            expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "complete where and unary broadcast release timing probe"]
    fn profile_selection_carry() {
        use std::time::Instant;
        let mask = fixture(vec![16, 1, 1, 1], TensorDType::Bool);
        let on_true = fixture(vec![1, 16, 16, 1], TensorDType::F32);
        let on_false = fixture(vec![1, 1, 1, 16], TensorDType::F32);
        let input = fixture(vec![1, 16, 1, 16], TensorDType::F32);
        let target = vec![16; 4];
        let modes = if std::env::var_os("QUABLA_BROADCAST_CARRY_FIRST").is_some() {
            [false, true]
        } else {
            [true, false]
        };
        for operation in ["where", "broadcast"] {
            for old in modes {
                let mut times = Vec::new();
                for _ in 0..15 {
                    let start = Instant::now();
                    let result = match (operation, old) {
                        ("where", true) => reference_where(&mask, &on_true, &on_false),
                        ("where", false) => mask.where_select(&on_true, &on_false),
                        (_, true) => reference_broadcast(&input, &target),
                        (_, false) => input.broadcast_to_shape(&target),
                    }
                    .unwrap();
                    std::hint::black_box(result);
                    times.push(start.elapsed().as_nanos());
                }
                times.sort();
                println!(
                    "{{\"operation\":\"{operation}\",\"reference\":{old},\"median_ns\":{}}}",
                    times[7]
                );
            }
        }
    }
}

#[cfg(test)]
mod gradient_accumulation_reuse_tests {
    use super::*;

    #[test]
    fn accumulation_preserves_bits_dtype_broadcast_and_shared_seeds() -> Result<(), String> {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            for shape in [vec![], vec![1], vec![4]] {
                let mut value = DynamicTensor::with_dtype(
                    vec![4],
                    vec![-0.0, 0.5, f64::INFINITY, f64::NAN],
                    dtype,
                )?;
                let seed_alias = value.clone();
                let contribution = DynamicTensor::with_dtype(
                    shape.clone(),
                    vec![0.25; element_count(&shape)?],
                    TensorDType::F32,
                )?;
                let expected = value.add(&contribution)?;
                let seed_bits = value.data.iter().map(f64::to_bits).collect::<Vec<_>>();
                value.add_assign(&contribution)?;
                assert_eq!(value.dtype, expected.dtype);
                assert_eq!(value.shape, expected.shape);
                assert_eq!(
                    value.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                    expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
                );
                assert_eq!(
                    seed_alias.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
                    seed_bits
                );
            }
        }
        let mut value = DynamicTensor::new(
            vec![4],
            vec![
                f64::from_bits(0x7ff0000000000001),
                f64::from_bits(0xfff8000000000005),
                -0.0,
                0.0,
            ],
        )?;
        let contribution = DynamicTensor::new(
            vec![4],
            vec![
                f64::from_bits(0x7ff8000000000007),
                f64::from_bits(0x7ff0000000000003),
                0.0,
                -0.0,
            ],
        )?;
        let expected = value.add(&contribution)?;
        value.add_assign(&contribution)?;
        assert_eq!(
            value.data.iter().map(f64::to_bits).collect::<Vec<_>>(),
            expected.data.iter().map(f64::to_bits).collect::<Vec<_>>()
        );
        let mut value = DynamicTensor::filled(vec![4], 1.0)?;
        let HostTensorStorage::F64(data) = &value.data else {
            unreachable!()
        };
        let pointer = data.as_ptr();
        value.add_assign(&DynamicTensor::filled(vec![4], 0.5)?)?;
        let HostTensorStorage::F64(data) = &value.data else {
            unreachable!()
        };
        assert_eq!(pointer, data.as_ptr());
        let previous = value.clone();
        let invalid = DynamicTensor::filled(vec![3], 1.0)?;
        let error = value.add(&invalid).unwrap_err();
        assert_eq!(value.add_assign(&invalid).unwrap_err(), error);
        assert_eq!(value.data(), previous.data());
        Ok(())
    }
}
