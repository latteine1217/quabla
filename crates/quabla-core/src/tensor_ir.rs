use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda;

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use cuda::{
    CudaBackend, CudaDataParallelExecutionPlan, CudaDataParallelResult, CudaDataParallelTiming,
    CudaExecutionPlan,
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
#[derive(Clone, Debug)]
pub struct CudaExecutionPlan {
    plan: TensorExecutionPlan,
    device_ordinal: usize,
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
impl CudaExecutionPlan {
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

/// Host tensor value. Storage is always `f64`; `dtype` records the logical
/// element type, and an `F32` tensor only holds values exactly representable
/// in `f32` (rounded to nearest-even on construction).
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
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
                    .zip(&rhs.data)
                    .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits()))
    }

    /// A hash consistent with [`Self::same_bits`].
    fn bits_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let value = self.value();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.dtype.hash(&mut hasher);
        value.shape.hash(&mut hasher);
        for element in &value.data {
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
    Matmul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Solve {
        matrix: TensorNodeId,
        rhs: TensorNodeId,
    },
    Triangular {
        input: TensorNodeId,
        lower: bool,
    },
    Tanh {
        input: TensorNodeId,
    },
    Exp {
        input: TensorNodeId,
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
    Sin {
        input: TensorNodeId,
    },
    Cos {
        input: TensorNodeId,
    },
    Powi {
        input: TensorNodeId,
        exponent: u32,
    },
    /// Elementwise `base ** exponent` with `f64::powf` semantics and
    /// broadcasting. Its derivative conventions at `base <= 0` are documented
    /// on `DynamicTensor::pow_base_derivative`.
    Pow {
        base: TensorNodeId,
        exponent: TensorNodeId,
    },
    Transpose {
        input: TensorNodeId,
        axes: Vec<usize>,
    },
    Log {
        input: TensorNodeId,
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
    Broadcast {
        input: TensorNodeId,
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
    nodes: Vec<TensorNode>,
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
    /// its IR op name (`solve`, or a `cond`/`fori`/`scan` region node).
    Unsupported { op: &'static str },
    /// Invalid bindings, outputs, or batch size.
    Invalid(String),
}

impl std::fmt::Display for BatchingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported { op: "solve" } => formatter.write_str(
                "vmap cannot batch solve: it takes rank-2 operands only, so an operand that \
                 depends on a mapped argument has no batching rule",
            ),
            Self::Unsupported { op } => write!(
                formatter,
                "vmap cannot batch a {op} region node: a cond, fori, or scan region whose \
                 operands depend on a mapped argument has no batching rule yet (the \
                 tensor_vmap_* helpers trace fori and scan bodies batched)"
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
    nodes: Vec<TensorNode>,
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
            data,
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
            data: vec![value; count],
            dtype: TensorDType::F64,
        })
    }

    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Converts to `dtype` with round-to-nearest-even (exact for widening).
    pub fn astype(&self, dtype: TensorDType) -> Self {
        self.clone().into_dtype(dtype)
    }

    fn into_dtype(mut self, dtype: TensorDType) -> Self {
        if dtype != TensorDType::F64 {
            for value in &mut self.data {
                *value = dtype.round(*value);
            }
        }
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

    pub fn data(&self) -> &[f64] {
        &self.data
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

    fn div(&self, rhs: &Self) -> Result<Self, String> {
        if rhs.data.contains(&0.0) {
            return Err("division by zero is not supported".to_string());
        }
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

        for index in 0..count {
            let condition_index = broadcast_offset(index, &shape, &self.shape, &condition_strides);
            let true_index = broadcast_offset(index, &shape, &on_true.shape, &true_strides);
            let false_index = broadcast_offset(index, &shape, &on_false.shape, &false_strides);
            data.push(if self.data[condition_index] != 0.0 {
                on_true.data[true_index]
            } else {
                on_false.data[false_index]
            });
        }

        Self::new(shape, data)
    }

    fn reciprocal(&self) -> Result<Self, String> {
        if self.data.contains(&0.0) {
            return Err("division by zero is not supported".to_string());
        }
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| 1.0 / value).collect(),
        )
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
        let output_strides = contiguous_strides(&output_shape);
        let mut data = vec![0.0; element_count(&output_shape)?];

        for (source_index, value) in self.data.iter().enumerate() {
            let mut remaining = source_index;
            let mut output_index = 0;
            for source_axis in (0..self.shape.len()).rev() {
                let coordinate = remaining % self.shape[source_axis];
                remaining /= self.shape[source_axis];
                if source_axis != axis {
                    let output_axis = if source_axis < axis {
                        source_axis
                    } else {
                        source_axis - 1
                    };
                    output_index += coordinate * output_strides[output_axis];
                }
            }
            data[output_index] += value * scale;
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
            data.push(self.data[source_index]);
        }

        Self::new(target_shape.to_vec(), data)
    }

    fn tanh(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.tanh()).collect(),
        )
    }

    fn exp(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.exp()).collect(),
        )
    }

    fn sqrt(&self) -> Result<Self, String> {
        self.sqrt_derivative(0)
    }

    fn sqrt_derivative(&self, order: u32) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .map(|value| sqrt_derivative_value(*value, order))
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
        Self::new(shape, self.data.clone())
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
                data.extend_from_slice(&input.data[start..end]);
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
            data.extend_from_slice(&self.data[source_start..source_end]);
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
            data[destination_start..destination_start + width]
                .copy_from_slice(&self.data[source_start..source_start + width]);
        }
        Self::new(output_shape.to_vec(), data)
    }

    fn mean_all(&self) -> Result<Self, String> {
        self.sum_all()?.scale(1.0 / self.data.len() as f64)
    }

    fn sin(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.sin()).collect(),
        )
    }

    fn cos(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.cos()).collect(),
        )
    }

    fn powi(&self, exponent: u32) -> Result<Self, String> {
        let exponent = i32::try_from(exponent)
            .map_err(|_| "powi exponent must fit in a signed 32-bit integer".to_string())?;
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.powi(exponent)).collect(),
        )
    }

    /// Elementwise `self ** exponent` with `f64::powf` semantics.
    fn pow(&self, exponent: &Self) -> Result<Self, String> {
        self.elementwise(exponent, f64::powf)
    }

    /// `d pow(x, y) / dx` under the conventions of `pow_base_derivative`.
    fn pow_base_derivative(&self, exponent: &Self) -> Result<Self, String> {
        self.elementwise(exponent, pow_base_derivative)
    }

    /// `d pow(x, y) / dy` under the conventions of `pow_exponent_derivative`.
    fn pow_exponent_derivative(&self, exponent: &Self) -> Result<Self, String> {
        self.elementwise(exponent, pow_exponent_derivative)
    }

    fn log(&self) -> Result<Self, String> {
        if self.data.iter().any(|value| *value <= 0.0) {
            return Err("log requires strictly positive tensor values".to_string());
        }
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.ln()).collect(),
        )
    }

    fn tanh_derivative_from_output(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| 1.0 - value * value).collect(),
        )
    }

    fn tanh_second_derivative_from_output(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .map(|value| -2.0 * value * (1.0 - value * value))
                .collect(),
        )
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
            for row in 0..lhs_rows {
                let lhs_row_start = lhs_batch * lhs_rows * lhs_inner + row * lhs_inner;
                let output_row_start = batch_index * lhs_rows * rhs_cols + row * rhs_cols;
                for inner in 0..lhs_inner {
                    let lhs_value = self.data[lhs_row_start + inner];
                    let rhs_row_start = rhs_batch * lhs_inner * rhs_cols + inner * rhs_cols;
                    for col in 0..rhs_cols {
                        data[output_row_start + col] += lhs_value * rhs.data[rhs_row_start + col];
                    }
                }
            }
        }

        Self::new(shape, data)
    }

    fn solve(&self, rhs: &Self) -> Result<Self, String> {
        solve_shape(&self.shape, &rhs.shape)?;
        let n = self.shape[0];
        let columns = rhs.shape[1];
        let mut factor = self.data.clone();
        let mut result = rhs.data.clone();

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
            .chunks_exact(matrix_size)
            .flat_map(|matrix| {
                matrix.iter().enumerate().map(move |(index, value)| {
                    let row = index / columns;
                    let column = index % columns;
                    if (lower && row >= column) || (!lower && row <= column) {
                        *value
                    } else {
                        0.0
                    }
                })
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
                        self.data[batch * rows * cols + row * cols + col];
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
            *output_value = self.data[input_index];
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
        let data = (0..count)
            .map(|index| self.data[broadcast_offset(index, target_shape, &self.shape, &strides)])
            .collect();
        Self::new(target_shape.to_vec(), data)
    }

    fn elementwise(&self, rhs: &Self, f: impl Fn(f64, f64) -> f64) -> Result<Self, String> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let count = element_count(&shape)?;
        let lhs_strides = contiguous_strides(&self.shape);
        let rhs_strides = contiguous_strides(&rhs.shape);
        let mut data = Vec::with_capacity(count);

        for index in 0..count {
            let lhs_index = broadcast_offset(index, &shape, &self.shape, &lhs_strides);
            let rhs_index = broadcast_offset(index, &shape, &rhs.shape, &rhs_strides);
            data.push(f(self.data[lhs_index], rhs.data[rhs_index]));
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
    /// (nested `vmap`), differentiated, and inlined. A mapped `solve` or
    /// region node (`cond`, `fori`, `scan`, and their derivative nodes) is
    /// [`BatchingError::Unsupported`]; unmapped ones are copied unchanged.
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
                mapped[id] = true;
                self.push_batched(callee, node, &remap, &mapped, batch_size)?
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
        let mut op = remap_tensor_op(&node.op, remap)?;
        if let Some(group) = tensor_op_group_mut(&mut op) {
            *group = *groups.entry(*group).or_insert(self.nodes.len());
        }
        Ok(self.push_node(op, node.shape.clone(), node.dtype, node.weak))
    }

    /// Appends the batched form of a callee node with at least one mapped
    /// operand (see [`Self::inline_batched`]); the node's dtype and weak
    /// flag are kept, and its shape gains the leading batch axis.
    fn push_batched(
        &mut self,
        callee: &TensorIr,
        node: &TensorNode,
        remap: &HashMap<TensorNodeId, TensorNodeId>,
        mapped: &[bool],
        batch_size: usize,
    ) -> Result<TensorNodeId, BatchingError> {
        let target = |operand: TensorNodeId| {
            remap
                .get(&operand)
                .copied()
                .ok_or_else(|| format!("node {operand} is missing from the vmap remap"))
        };
        let rank = node.shape.len();
        let op = match &node.op {
            TensorOp::Cast { .. }
            | TensorOp::Tanh { .. }
            | TensorOp::Exp { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Sin { .. }
            | TensorOp::Cos { .. }
            | TensorOp::Powi { .. }
            | TensorOp::Log { .. }
            | TensorOp::Triangular { .. }
            | TensorOp::Reshape { .. } => remap_tensor_op(&node.op, remap)?,
            TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Div { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. }
            | TensorOp::Pow { .. }
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
                remap_tensor_op(&node.op, &local)?
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
            TensorOp::MeanAxis { input, axis } => TensorOp::MeanAxis {
                input: target(*input)?,
                axis: axis + 1,
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
            TensorOp::Solve { .. }
            | TensorOp::Cond { .. }
            | TensorOp::Fori { .. }
            | TensorOp::ForiJvp { .. }
            | TensorOp::ForiVjp { .. }
            | TensorOp::ForiVjpJvp { .. }
            | TensorOp::Scan { .. }
            | TensorOp::ScanVjp { .. }
            | TensorOp::ScanVjpJvp { .. } => {
                return Err(BatchingError::Unsupported {
                    op: tensor_op_name(&node.op),
                })
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
        Ok(self.push_node(
            op,
            batched_shape(batch_size, &node.shape),
            node.dtype,
            node.weak,
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
        if outputs.is_empty() {
            return Err("symbolic JVP requires at least one output".to_string());
        }
        for output in outputs {
            self.node(*output)?;
        }
        let mut transformed = TensorIr::new();
        let mut pairs = Vec::with_capacity(self.nodes.len());
        let mut scan_jvp_results = HashMap::new();
        let mut fori_vjp_jvp_groups = HashMap::new();
        let mut scan_vjp_jvp_groups = HashMap::new();

        for (node_index, node) in self.nodes.iter().enumerate() {
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
                            Some(tangent) => tangent,
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
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let left_term = transformed.mul(lhs_tangent, rhs_value)?;
                    let right_term = transformed.mul(lhs_value, rhs_tangent)?;
                    let numerator = transformed.sub(left_term, right_term)?;
                    let denominator = transformed.mul(rhs_value, rhs_value)?;
                    (
                        transformed.div(lhs_value, rhs_value)?,
                        transformed.div(numerator, denominator)?,
                    )
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
                    let tangent = transformed.scalar_constant(0.0);
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
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "symbolic JVP through a Fori JVP result is not implemented".to_string()
                    )
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
                    )
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
                    )
                }
                TensorOp::Tanh { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.tanh(input_value)?;
                    let one = transformed.scalar_constant(1.0);
                    let squared = transformed.mul(value, value)?;
                    let derivative = transformed.sub(one, squared)?;
                    (value, transformed.mul(input_tangent, derivative)?)
                }
                TensorOp::Exp { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.exp(input_value)?;
                    (value, transformed.mul(input_tangent, value)?)
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
                TensorOp::Sin { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.sin(input_value)?;
                    let derivative = transformed.cos(input_value)?;
                    (value, transformed.mul(input_tangent, derivative)?)
                }
                TensorOp::Cos { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.cos(input_value)?;
                    let derivative = transformed.sin(input_value)?;
                    let tangent = transformed.mul(input_tangent, derivative)?;
                    let zero = transformed.scalar_constant(0.0);
                    (value, transformed.sub(zero, tangent)?)
                }
                TensorOp::Log { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    (
                        transformed.log(input_value)?,
                        transformed.div(input_tangent, input_value)?,
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
                TensorOp::Pow { base, exponent } => {
                    let (base_value, base_tangent) = pairs[*base];
                    let (exponent_value, exponent_tangent) = pairs[*exponent];
                    (
                        transformed.pow(base_value, exponent_value)?,
                        transformed.pow_tangent(
                            base_value,
                            base_tangent,
                            exponent_value,
                            exponent_tangent,
                        )?,
                    )
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
                TensorOp::Broadcast { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.broadcast_to(value, node.shape.clone())?,
                        transformed.broadcast_to(tangent, node.shape.clone())?,
                    )
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
        let transformed = self
            .symbolic_vjp_many(&[(output, SymbolicCotangent::Input(cotangent_name.to_string()))])?;
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

        let mut transformed = TensorIr::new();
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut scan_values = HashMap::new();
        for (node_index, node) in self.nodes.iter().enumerate() {
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
                TensorOp::ForiVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP result is not implemented".to_string()
                    )
                }
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori JVP result is not implemented".to_string()
                    )
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP JVP result is not implemented".to_string(),
                    )
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
                    )
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP JVP result is not implemented".to_string(),
                    )
                }
                TensorOp::Sum { input } => transformed.sum(values[*input])?,
                TensorOp::SumAxis { input, axis } => {
                    transformed.sum_axis(values[*input], *axis as isize)?
                }
                TensorOp::Matmul { lhs, rhs } => transformed.matmul(values[*lhs], values[*rhs])?,
                TensorOp::Solve { matrix, rhs } => {
                    transformed.solve(values[*matrix], values[*rhs])?
                }
                TensorOp::Triangular { input, lower } => {
                    transformed.triangular(values[*input], *lower)?
                }
                TensorOp::Tanh { input } => transformed.tanh(values[*input])?,
                TensorOp::Exp { input } => transformed.exp(values[*input])?,
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
                TensorOp::Sin { input } => transformed.sin(values[*input])?,
                TensorOp::Cos { input } => transformed.cos(values[*input])?,
                TensorOp::Powi { input, exponent } => {
                    transformed.powi(values[*input], *exponent)?
                }
                TensorOp::Pow { base, exponent } => {
                    transformed.pow(values[*base], values[*exponent])?
                }
                TensorOp::Transpose { input, axes } => transformed.transpose(
                    values[*input],
                    Some(axes.iter().map(|axis| *axis as isize).collect()),
                )?,
                TensorOp::Log { input } => transformed.log(values[*input])?,
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
                TensorOp::Broadcast { input } => {
                    transformed.broadcast_to(values[*input], node.shape.clone())?
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
                TensorOp::ForiVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP result is not implemented".to_string()
                    )
                }
                TensorOp::ForiJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori JVP result is not implemented".to_string()
                    )
                }
                TensorOp::ForiVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Fori VJP JVP result is not implemented".to_string(),
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
                                None => symbolic_zero_like(&mut transformed, values[scan_node_id])?,
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
                        symbolic_vjp_scan(
                            &mut transformed,
                            scan_plan,
                            SymbolicVjpScanContext {
                                original_carry: *carry,
                                carry: values[*carry],
                                captures,
                                values: &values,
                                final_carry_upstream: carry_upstream.ok_or_else(|| {
                                    format!("symbolic Scan group {group} has no carry result")
                                })?,
                                output_upstream: output_upstream.ok_or_else(|| {
                                    format!("symbolic Scan group {group} has no output result")
                                })?,
                            },
                            &mut cotangents,
                            *group,
                        )?;
                    }
                }
                TensorOp::ScanVjp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP result is not implemented".to_string()
                    )
                }
                TensorOp::ScanVjpJvp { .. } => {
                    return Err(
                        "symbolic VJP through a Scan VJP JVP result is not implemented".to_string(),
                    )
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
                    let lhs_contribution = transformed.div(upstream, rhs_value)?;
                    let rhs_squared = transformed.mul(rhs_value, rhs_value)?;
                    let numerator = transformed.mul(upstream, lhs_value)?;
                    let quotient = transformed.div(numerator, rhs_squared)?;
                    let zero = transformed.scalar_constant(0.0);
                    let rhs_contribution = transformed.sub(zero, quotient)?;
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
                TensorOp::Tanh { input } => {
                    let one = transformed.scalar_constant(1.0);
                    let squared = transformed.mul(values[node_id], values[node_id])?;
                    let derivative = transformed.sub(one, squared)?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Exp { input } => {
                    let contribution = transformed.mul(upstream, values[node_id])?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
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
                TensorOp::Sin { input } => {
                    let derivative = transformed.cos(values[*input])?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Cos { input } => {
                    let derivative = transformed.sin(values[*input])?;
                    let product = transformed.mul(upstream, derivative)?;
                    let zero = transformed.scalar_constant(0.0);
                    let contribution = transformed.sub(zero, product)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
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
                // A constant operand receives no cotangent, so its derivative expression is not
                // emitted at all.
                TensorOp::Pow { base, exponent } => {
                    let base_value = values[*base];
                    let exponent_value = values[*exponent];
                    let operands = [
                        (*base, self.scalar_constant_value(*base).is_none(), true),
                        (
                            *exponent,
                            self.scalar_constant_value(*exponent).is_none(),
                            false,
                        ),
                    ];
                    for (operand, differentiable, is_base) in operands {
                        if !differentiable {
                            continue;
                        }
                        let derivative = if is_base {
                            transformed.pow_base_derivative(base_value, exponent_value)?
                        } else {
                            transformed.pow_exponent_derivative(base_value, exponent_value)?
                        };
                        let contribution = transformed.mul(upstream, derivative)?;
                        let contribution = symbolic_reduce_to_shape(
                            &mut transformed,
                            contribution,
                            &node.shape,
                            &self.node(operand)?.shape,
                        )?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            operand,
                            contribution,
                        )?;
                    }
                }
                TensorOp::Transpose { input, axes } => {
                    let inverse = inverse_permutation(axes)?;
                    let contribution = transformed.transpose(
                        upstream,
                        Some(inverse.into_iter().map(|axis| axis as isize).collect()),
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Log { input } => {
                    let contribution = transformed.div(upstream, values[*input])?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
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
        if self.nodes.iter().any(
            |node| matches!(node.op, TensorOp::Input { name: ref existing } if existing == &name),
        ) {
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
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Tanh { input }, shape)
    }

    pub fn exp(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Exp { input }, shape)
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
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Sin { input }, shape)
    }

    pub fn cos(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Cos { input }, shape)
    }

    pub fn powi(&mut self, input: TensorNodeId, exponent: u32) -> Result<TensorNodeId, String> {
        if exponent > i32::MAX as u32 {
            return Err("powi exponent must fit in a signed 32-bit integer".to_string());
        }
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Powi { input, exponent }, shape)
    }

    /// Elementwise `base ** exponent` with `f64::powf` semantics: NaN for a
    /// negative base with a non-integer exponent, `0 ** 0 == 1`, and IEEE
    /// results for infinities and NaN. Operands broadcast and promote like
    /// arithmetic operands, so a weak scalar exponent adopts the base dtype.
    /// `Bool` operands are rejected, as by the unary math ops, instead of
    /// being promoted to 0/1.
    pub fn pow(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        for operand in [base, exponent] {
            if self.node(operand)?.dtype == TensorDType::Bool {
                return Err(
                    "pow is not defined for bool tensors; use logical_and/logical_or/logical_not \
                     (& | ~) or convert explicitly with astype"
                        .to_string(),
                );
            }
        }
        self.binary("pow", base, exponent, |base, exponent| TensorOp::Pow {
            base,
            exponent,
        })
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

    /// `exponent - 1`, folded to a constant of the same dtype and weakness
    /// when the exponent is a constant, so nested derivatives of a constant
    /// exponent keep recognizing it.
    fn pow_lowered_exponent(&mut self, exponent: TensorNodeId) -> Result<TensorNodeId, String> {
        if let Some(value) = self.scalar_constant_value(exponent) {
            let node = self.node(exponent)?;
            let (dtype, weak) = (node.dtype, node.weak);
            return Ok(self.constant_like(dtype.round(value - 1.0), dtype, weak));
        }
        let one = self.scalar_constant(1.0);
        self.sub(exponent, one)
    }

    /// Symbolic `d pow(x, y) / dx` with the conventions of the free function
    /// `pow_base_derivative`: `where(m, 0, y * pow(where(m, 1, x), y - 1))`
    /// with `m = (x == 0) & (y < 1)`. Masking the inner base keeps `0 * inf`
    /// out of the reverse pass through this expression, so Hessians stay
    /// finite at the origin. A constant exponent decides `y < 1` statically.
    fn pow_base_derivative(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let zero = self.scalar_constant(0.0);
        let one = self.scalar_constant(1.0);
        let singular = match self.scalar_constant_value(exponent) {
            // No point is singular unless `y < 1` (a NaN exponent is never singular).
            Some(value) if !pow_base_is_singular(0.0, value) => None,
            Some(_) => Some(self.compare(base, zero, TensorComparison::Equal)?),
            None => {
                let at_origin = self.compare(base, zero, TensorComparison::Equal)?;
                let below_one = self.compare(exponent, one, TensorComparison::Less)?;
                Some(self.logical_and(at_origin, below_one)?)
            }
        };
        let safe_base = match singular {
            Some(mask) => self.where_select(mask, one, base)?,
            None => base,
        };
        let lowered = self.pow_lowered_exponent(exponent)?;
        let power = self.pow(safe_base, lowered)?;
        let derivative = self.mul(exponent, power)?;
        match singular {
            Some(mask) => self.where_select(mask, zero, derivative),
            None => Ok(derivative),
        }
    }

    /// Symbolic `d pow(x, y) / dy` with the conventions of the free function
    /// `pow_exponent_derivative`: `pow(s, y) * log(s)` with
    /// `s = where(x <= 0, 1, x)`, which is exactly `0` for `x <= 0` and keeps
    /// the checked-domain `log` away from non-positive values.
    fn pow_exponent_derivative(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        if let Some(value) = self.scalar_constant_value(base) {
            if value <= 0.0 {
                return Ok(self.scalar_constant(0.0));
            }
            // A positive constant base reuses the primal `pow(base, y)` (CSE)
            // and folds its logarithm.
            let node = self.node(base)?;
            let (dtype, weak) = (node.dtype, node.weak);
            let log = self.constant_like(dtype.round(value.ln()), dtype, weak);
            let power = self.pow(base, exponent)?;
            return self.mul(power, log);
        }
        let zero = self.scalar_constant(0.0);
        let one = self.scalar_constant(1.0);
        let non_positive = self.compare(base, zero, TensorComparison::LessEqual)?;
        let safe_base = self.where_select(non_positive, one, base)?;
        let power = self.pow(safe_base, exponent)?;
        let log = self.log(safe_base)?;
        self.mul(power, log)
    }

    /// Tangent of `pow(x, y)`; the term of an operand whose tangent is the
    /// constant zero (a constant operand) is left out rather than multiplied
    /// by a derivative that may be non-finite.
    fn pow_tangent(
        &mut self,
        base: TensorNodeId,
        base_tangent: TensorNodeId,
        exponent: TensorNodeId,
        exponent_tangent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let mut tangent = None;
        if self.scalar_constant_value(base_tangent) != Some(0.0) {
            let derivative = self.pow_base_derivative(base, exponent)?;
            tangent = Some(self.mul(base_tangent, derivative)?);
        }
        if self.scalar_constant_value(exponent_tangent) != Some(0.0) {
            let derivative = self.pow_exponent_derivative(base, exponent)?;
            let term = self.mul(exponent_tangent, derivative)?;
            tangent = Some(match tangent {
                Some(tangent) => self.add(tangent, term)?,
                None => term,
            });
        }
        Ok(match tangent {
            Some(tangent) => tangent,
            None => self.scalar_constant(0.0),
        })
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
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::Log { input }, shape)
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
        let mut reachable = HashSet::new();
        let mut pending = outputs.to_vec();
        while let Some(node_id) = pending.pop() {
            if !reachable.insert(node_id) {
                continue;
            }
            pending.extend(tensor_op_inputs(&self.node(node_id)?.op));
        }

        let mut remap = HashMap::new();
        let mut cse_nodes = HashMap::new();
        // Array constants by `TensorConstant::bits_hash`; candidates in one bucket are compared
        // bitwise, so a hash collision never merges different values.
        let mut constant_nodes = HashMap::<u64, Vec<TensorNodeId>>::new();
        let mut nodes = Vec::with_capacity(reachable.len());
        for (old_id, node) in self.nodes.iter().enumerate() {
            if !reachable.contains(&old_id) {
                continue;
            }
            let mut op = remap_tensor_op(&node.op, &remap)?;
            if let Some(alias) = canonicalize_tensor_op(&mut op, &node.shape, node.dtype, &nodes)? {
                remap.insert(old_id, alias);
                continue;
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
                    remap.insert(old_id, existing_id);
                    continue;
                }
                bucket.push(nodes.len());
            }
            if let Some(key) = pure_tensor_op_cse_key(&op, &node.shape) {
                // dtype and weakness are part of a value's identity: cast(x, f32) must not be
                // merged with x.
                let key = format!("{key}:{}:{}", node.dtype, node.weak);
                if let Some(existing_id) = cse_nodes.get(&key) {
                    remap.insert(old_id, *existing_id);
                    continue;
                }
                cse_nodes.insert(key, nodes.len());
            }
            remap.insert(old_id, nodes.len());
            nodes.push(TensorNode {
                op,
                shape: node.shape.clone(),
                dtype: node.dtype,
                weak: node.weak,
            });
        }
        let output_node_ids = outputs
            .iter()
            .map(|output| {
                remap
                    .get(output)
                    .copied()
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
                nodes,
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
        if outputs.is_empty() {
            return Err("multi-output VJP requires at least one output".to_string());
        }
        let values = self.evaluate_all(inputs)?;
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
            let cotangent = match cotangents[node_id].clone() {
                Some(value) => value,
                None => continue,
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
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let reciprocal = rhs_value.reciprocal()?;
                    let reciprocal_squared = reciprocal.mul(&reciprocal)?;
                    let lhs_contribution = cotangent
                        .mul(&reciprocal)?
                        .reduce_to_shape(&lhs_value.shape)?;
                    let rhs_contribution = cotangent
                        .mul(lhs_value)?
                        .mul(&reciprocal_squared)?
                        .neg()?
                        .reduce_to_shape(&rhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
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
                            .ok_or_else(|| format!("node {predicate} has no evaluated value"))?,
                    )?;
                    let branch_inputs = tensor_cond_capture_values(captures, &values)?;
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
                    let external = tensor_fori_capture_values(captures, &values)?;
                    let (_, carry_gradient, external_gradients) = loop_plan.value_and_vjp(
                        values
                            .get(*carry)
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
                            let group_cotangent = cotangents[scan_node_id]
                                .clone()
                                .unwrap_or(DynamicTensor::filled(scan_node.shape.clone(), 0.0)?);
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
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let (_, _, carry_gradient, external_gradients) = scan_plan.value_and_vjp(
                            values
                                .get(*carry)
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
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
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
                TensorOp::Solve { matrix, rhs } => {
                    let matrix_value = values
                        .get(*matrix)
                        .ok_or_else(|| format!("node {matrix} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let output_value = values
                        .get(node_id)
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
                TensorOp::Tanh { input } => {
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&output_value.tanh_derivative_from_output()?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Exp { input } => {
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(output_value)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Sqrt { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.sqrt_derivative(1)?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::SqrtDerivative { input, order } => {
                    let input_value = values
                        .get(*input)
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
                TensorOp::Sin { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.cos()?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Cos { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.sin()?)?
                        .neg()?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent == 0 {
                        continue;
                    }
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.powi(*exponent - 1)?)?
                        .scale(*exponent as f64)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Pow { base, exponent } => {
                    let base_value = values
                        .get(*base)
                        .ok_or_else(|| format!("node {base} has no evaluated value"))?;
                    let exponent_value = values
                        .get(*exponent)
                        .ok_or_else(|| format!("node {exponent} has no evaluated value"))?;
                    // Constant operands are skipped, as in the symbolic rule.
                    if self.scalar_constant_value(*base).is_none() {
                        let contribution = cotangent
                            .mul(&base_value.pow_base_derivative(exponent_value)?)?
                            .reduce_to_shape(&base_value.shape)?;
                        accumulate(&mut cotangents[*base], contribution)?;
                    }
                    if self.scalar_constant_value(*exponent).is_none() {
                        let contribution = cotangent
                            .mul(&base_value.pow_exponent_derivative(exponent_value)?)?
                            .reduce_to_shape(&exponent_value.shape)?;
                        accumulate(&mut cotangents[*exponent], contribution)?;
                    }
                }
                TensorOp::Transpose { input, axes } => {
                    let contribution = cotangent.transpose(&inverse_permutation(axes)?)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Log { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.reciprocal()?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
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
                TensorOp::Broadcast { input } => accumulate(
                    &mut cotangents[*input],
                    cotangent.reduce_to_shape(&self.node(*input)?.shape)?,
                )?,
            }
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
                let gradient = match cotangents[node_id].clone() {
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
        self.node(output)?;
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
                        .is_some_and(|tangent| tangent.data.iter().any(|value| *value != 0.0))
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
                    let reciprocal = rhs_value.reciprocal()?;
                    lhs_tangent.mul(&reciprocal)?.sub(
                        &lhs_value
                            .mul(rhs_tangent)?
                            .mul(&reciprocal.mul(&reciprocal)?)?,
                    )?
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
                TensorOp::Triangular { input, lower } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .triangular(*lower)?,
                TensorOp::Tanh { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    input_tangent.mul(&output_value.tanh_derivative_from_output()?)?
                }
                TensorOp::Exp { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    input_tangent.mul(output_value)?
                }
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
                TensorOp::Sin { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.cos()?)?
                }
                TensorOp::Cos { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.sin()?)?.neg()?
                }
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
                TensorOp::Pow { base, exponent } => {
                    let base_value = values
                        .get(*base)
                        .ok_or_else(|| format!("node {base} has no evaluated value"))?;
                    let exponent_value = values
                        .get(*exponent)
                        .ok_or_else(|| format!("node {exponent} has no evaluated value"))?;
                    // Constant operands are skipped, as in the symbolic rule.
                    let mut tangent = DynamicTensor::filled(node.shape.clone(), 0.0)?;
                    if self.scalar_constant_value(*base).is_none() {
                        let base_tangent = tangents
                            .get(*base)
                            .ok_or_else(|| format!("node {base} has no evaluated tangent"))?;
                        tangent = tangent.add(
                            &base_tangent.mul(&base_value.pow_base_derivative(exponent_value)?)?,
                        )?;
                    }
                    if self.scalar_constant_value(*exponent).is_none() {
                        let exponent_tangent = tangents
                            .get(*exponent)
                            .ok_or_else(|| format!("node {exponent} has no evaluated tangent"))?;
                        tangent = tangent.add(
                            &exponent_tangent
                                .mul(&base_value.pow_exponent_derivative(exponent_value)?)?,
                        )?;
                    }
                    tangent
                }
                TensorOp::Transpose { input, axes } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .transpose(axes)?,
                TensorOp::Log { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.reciprocal()?)?
                }
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
                TensorOp::Broadcast { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .broadcast_to_shape(&node.shape)?,
            };
            tangents.push(tangent);
        }

        let value = values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))?;
        // As in the VJP: derivative arithmetic runs in f64 and output tangents are rounded to the
        // output dtype; the tangent of a Bool output is an f64 zero (matching the weak f64 zero of
        // the symbolic JVP).
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
                *entry = result.mixed.data[0];
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
        let second_tangents = BTreeMap::from([(input_name.to_string(), input_tangent)]);
        let mut data = Vec::with_capacity(input_count);
        for index in 0..input_count {
            let first_tangents = BTreeMap::from([(
                input_name.to_string(),
                DynamicTensor::one_hot(input_shape.clone(), index)?,
            )]);
            let result = self.evaluate_mixed(output, inputs, &first_tangents, &second_tangents)?;
            data.push(result.mixed.data[0]);
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
            *hessian_row = columns.iter().map(|column| column.data[row]).collect();
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
        let mut transformed_inputs = inputs.clone();
        transformed_inputs.insert(cotangent_name, DynamicTensor::filled(vec![], 1.0)?);
        transformed_inputs.insert(tangent_name, input_tangent);
        directional
            .graph
            .evaluate(directional.tangent, &transformed_inputs)
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
                TensorOp::Matmul { lhs, rhs } => format!(
                    "%{id} = matmul(%{lhs}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Solve { matrix, rhs } => format!(
                    "%{id} = solve(%{matrix}, %{rhs}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Triangular { input, lower } => format!(
                    "%{id} = {}(%{input}) : {}",
                    if *lower { "tril" } else { "triu" },
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Tanh { input } => {
                    format!("%{id} = tanh(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::Exp { input } => {
                    format!("%{id} = exp(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
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
                TensorOp::Sin { input } => {
                    format!("%{id} = sin(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::Cos { input } => {
                    format!("%{id} = cos(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
                TensorOp::Powi { input, exponent } => format!(
                    "%{id} = powi(%{input}, {exponent}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Pow { base, exponent } => format!(
                    "%{id} = pow(%{base}, %{exponent}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Transpose { input, axes } => format!(
                    "%{id} = transpose(%{input}, axes={axes:?}) : {}",
                    format_tensor_type(&node.shape, node.dtype)
                ),
                TensorOp::Log { input } => {
                    format!("%{id} = log(%{input}) : {}", format_tensor_type(&node.shape, node.dtype))
                }
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
                TensorOp::Pow { base, exponent } => format!(
                    "%v{id} = stablehlo.power {}, {} : {}",
                    values[base],
                    values[exponent],
                    tensor_type(node)
                ),
                TensorOp::Tanh { input } => format!(
                    "%v{id} = stablehlo.tanh {} : {}",
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
        self.nodes.push(TensorNode {
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
                    | TensorOp::Broadcast { .. }
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
        let mut values: Vec<DynamicTensor> = Vec::with_capacity(nodes.len());
        let mut fori_vjp_cache: HashMap<usize, TensorForiVjpEvaluation> = HashMap::new();
        let mut fori_vjp_jvp_cache: HashMap<usize, TensorForiVjpJvpEvaluation> = HashMap::new();
        let mut scan_cache: HashMap<usize, TensorScanEvaluation> = HashMap::new();
        let mut scan_vjp_cache: HashMap<usize, TensorScanVjpEvaluation> = HashMap::new();
        let mut scan_vjp_jvp_cache: HashMap<usize, TensorScanVjpJvpEvaluation> = HashMap::new();

        for node in nodes {
            let value = match &node.op {
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
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .clone(),
                TensorOp::Add { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .add(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Sub { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .sub(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Div { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .div(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Mul { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .mul(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Greater { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .greater(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Compare { lhs, rhs, kind } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .compare(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                        *kind,
                    )?,
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => values
                    .get(*condition)
                    .ok_or_else(|| format!("node {condition} has no evaluated value"))?
                    .where_select(
                        values
                            .get(*on_true)
                            .ok_or_else(|| format!("node {on_true} has no evaluated value"))?,
                        values
                            .get(*on_false)
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
                            .ok_or_else(|| format!("node {predicate} has no evaluated value"))?,
                    )?;
                    let branch_inputs = tensor_cond_capture_values(captures, &values)?;
                    branches.evaluate(predicate, &branch_inputs)?
                }
                TensorOp::Fori {
                    carry,
                    loop_plan,
                    captures,
                } => {
                    let external_inputs = tensor_fori_capture_values(captures, &values)?;
                    loop_plan.evaluate(
                        values
                            .get(*carry)
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
                    let external_inputs = tensor_fori_capture_values(captures, &values)?;
                    let external_tangents = tensor_fori_capture_values(tangent_captures, &values)?;
                    loop_plan
                        .jvp(
                            values
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            values.get(*carry_tangent).cloned().ok_or_else(|| {
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
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let (_, carry_gradient, external_gradients) = loop_plan.value_and_vjp(
                            values
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            &external_inputs,
                            values.get(*output_cotangent).cloned().ok_or_else(|| {
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
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let external_tangents =
                            tensor_fori_capture_values(tangent_captures, &values)?;
                        let gradients =
                            plan.jvp(
                                values.get(*carry).cloned().ok_or_else(|| {
                                    format!("node {carry} has no evaluated value")
                                })?,
                                values.get(*carry_tangent).cloned().ok_or_else(|| {
                                    format!("node {carry_tangent} has no evaluated value")
                                })?,
                                &external_inputs,
                                &external_tangents,
                                values.get(*output_cotangent).cloned().ok_or_else(|| {
                                    format!("node {output_cotangent} has no evaluated value")
                                })?,
                                values.get(*output_cotangent_tangent).cloned().ok_or_else(
                                    || {
                                        format!(
                                            "node {output_cotangent_tangent} has no evaluated value"
                                        )
                                    },
                                )?,
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
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let (carry, outputs) = scan_plan.evaluate(
                            values
                                .get(*carry)
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
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let (_, _, carry_gradient, external_gradients) = scan_plan.value_and_vjp(
                            values
                                .get(*carry)
                                .cloned()
                                .ok_or_else(|| format!("node {carry} has no evaluated value"))?,
                            &external_inputs,
                            values.get(*final_carry_cotangent).cloned().ok_or_else(|| {
                                format!("node {final_carry_cotangent} has no evaluated value")
                            })?,
                            values.get(*output_cotangent).cloned().ok_or_else(|| {
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
                        let external_inputs = tensor_fori_capture_values(captures, &values)?;
                        let external_tangents =
                            tensor_fori_capture_values(tangent_captures, &values)?;
                        let gradients = plan.jvp(
                                values.get(*carry).cloned().ok_or_else(|| {
                                    format!("node {carry} has no evaluated value")
                                })?,
                                values.get(*carry_tangent).cloned().ok_or_else(|| {
                                    format!("node {carry_tangent} has no evaluated value")
                                })?,
                                &external_inputs,
                                &external_tangents,
                                values.get(*final_carry_cotangent).cloned().ok_or_else(|| {
                                    format!(
                                        "node {final_carry_cotangent} has no evaluated value"
                                    )
                                })?,
                                values
                                    .get(*final_carry_cotangent_tangent)
                                    .cloned()
                                    .ok_or_else(|| {
                                        format!(
                                            "node {final_carry_cotangent_tangent} has no evaluated value"
                                        )
                                    })?,
                                values.get(*output_cotangent).cloned().ok_or_else(|| {
                                    format!("node {output_cotangent} has no evaluated value")
                                })?,
                                values
                                    .get(*output_cotangent_tangent)
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
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sum_all()?,
                TensorOp::SumAxis { input, axis } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_axis(*axis, 1.0)?,
                TensorOp::Matmul { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .matmul(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Solve { matrix, rhs } => values
                    .get(*matrix)
                    .ok_or_else(|| format!("node {matrix} has no evaluated value"))?
                    .solve(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Triangular { input, lower } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .triangular(*lower)?,
                TensorOp::Tanh { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .tanh()?,
                TensorOp::Exp { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .exp()?,
                TensorOp::Sqrt { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sqrt()?,
                TensorOp::SqrtDerivative { input, order } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sqrt_derivative(*order)?,
                TensorOp::Reshape { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reshape(node.shape.clone())?,
                TensorOp::Mean { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .mean_all()?,
                TensorOp::MeanAxis { input, axis } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_axis(*axis, 1.0 / nodes[*input].shape[*axis] as f64)?,
                TensorOp::Sin { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sin()?,
                TensorOp::Cos { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .cos()?,
                TensorOp::Powi { input, exponent } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .powi(*exponent)?,
                TensorOp::Pow { base, exponent } => values
                    .get(*base)
                    .ok_or_else(|| format!("node {base} has no evaluated value"))?
                    .pow(
                        values
                            .get(*exponent)
                            .ok_or_else(|| format!("node {exponent} has no evaluated value"))?,
                    )?,
                TensorOp::Transpose { input, axes } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .transpose(axes)?,
                TensorOp::Log { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .log()?,
                TensorOp::Concat { inputs, axis } => DynamicTensor::concat(
                    &inputs
                        .iter()
                        .map(|input| {
                            values
                                .get(*input)
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
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .slice_axis(*axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .pad_slice(&node.shape, *axis, *start)?,
                TensorOp::Broadcast { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .broadcast_to_shape(&node.shape)?,
            };
            // Each node computes in f64 and rounds to the node dtype: inputs are rounded
            // automatically, casts convert, and f32 nodes get the correctly rounded f64 result
            // (`+ - * / sqrt` are bit-identical to IEEE f32).
            values.push(value.into_dtype(node.dtype));
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
        let mut values: Vec<MixedTangent> = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
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
                TensorOp::Log { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let reciprocal = input.value.reciprocal()?;
                    let reciprocal_squared = reciprocal.mul(&reciprocal)?;
                    MixedTangent {
                        value: input.value.log()?,
                        first: input.first.mul(&reciprocal)?,
                        second: input.second.mul(&reciprocal)?,
                        mixed: input
                            .mixed
                            .mul(&reciprocal)?
                            .sub(&input.first.mul(&input.second)?.mul(&reciprocal_squared)?)?,
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
                TensorOp::Sin { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let value = input.value.sin()?;
                    let cosine = input.value.cos()?;
                    MixedTangent {
                        first: input.first.mul(&cosine)?,
                        second: input.second.mul(&cosine)?,
                        mixed: input
                            .mixed
                            .mul(&cosine)?
                            .sub(&input.first.mul(&input.second)?.mul(&value)?)?,
                        value,
                    }
                }
                TensorOp::Cos { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let value = input.value.cos()?;
                    let sine = input.value.sin()?;
                    MixedTangent {
                        first: input.first.mul(&sine)?.neg()?,
                        second: input.second.mul(&sine)?.neg()?,
                        mixed: input
                            .mixed
                            .mul(&sine)?
                            .neg()?
                            .sub(&input.first.mul(&input.second)?.mul(&value)?)?,
                        value,
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
                TensorOp::Exp { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let value = input.value.exp()?;
                    MixedTangent {
                        first: input.first.mul(&value)?,
                        second: input.second.mul(&value)?,
                        mixed: input
                            .mixed
                            .add(&input.first.mul(&input.second)?)?
                            .mul(&value)?,
                        value,
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
                    let reciprocal = rhs.value.reciprocal()?;
                    let reciprocal_squared = reciprocal.mul(&reciprocal)?;
                    let reciprocal_cubed = reciprocal_squared.mul(&reciprocal)?;
                    MixedTangent {
                        value: lhs.value.mul(&reciprocal)?,
                        first: lhs
                            .first
                            .mul(&reciprocal)?
                            .sub(&lhs.value.mul(&rhs.first)?.mul(&reciprocal_squared)?)?,
                        second: lhs
                            .second
                            .mul(&reciprocal)?
                            .sub(&lhs.value.mul(&rhs.second)?.mul(&reciprocal_squared)?)?,
                        mixed: lhs
                            .mixed
                            .mul(&reciprocal)?
                            .sub(&lhs.first.mul(&rhs.second)?.mul(&reciprocal_squared)?)?
                            .sub(&lhs.second.mul(&rhs.first)?.mul(&reciprocal_squared)?)?
                            .sub(&lhs.value.mul(&rhs.mixed)?.mul(&reciprocal_squared)?)?
                            .add(
                                &lhs.value
                                    .mul(&rhs.first)?
                                    .mul(&rhs.second)?
                                    .mul(&reciprocal_cubed)?
                                    .scale(2.0)?,
                            )?,
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
                // Second-order forward rule matching the composed symbolic rules: `first` and
                // `second` use the first partials, and `mixed` adds `sum_ij d_i d_j pow` over the
                // non-constant operands, with the mixed partials in composition order.
                TensorOp::Pow { base, exponent } => {
                    let base_tangent = values
                        .get(*base)
                        .ok_or_else(|| format!("node {base} has no evaluated value"))?;
                    let exponent_tangent = values
                        .get(*exponent)
                        .ok_or_else(|| format!("node {exponent} has no evaluated value"))?;
                    let (x, y) = (&base_tangent.value, &exponent_tangent.value);
                    let operands = [
                        (base_tangent, self.scalar_constant_value(*base).is_none()),
                        (exponent_tangent, self.scalar_constant_value(*exponent).is_none()),
                    ];
                    let derivatives = [x.pow_base_derivative(y)?, x.pow_exponent_derivative(y)?];
                    let mut first = DynamicTensor::filled(node.shape.clone(), 0.0)?;
                    let mut second = first.clone();
                    let mut mixed = first.clone();
                    for (index, (operand, varies)) in operands.iter().enumerate() {
                        if !varies {
                            continue;
                        }
                        let derivative = &derivatives[index];
                        first = first.add(&operand.first.mul(derivative)?)?;
                        second = second.add(&operand.second.mul(derivative)?)?;
                        mixed = mixed.add(&operand.mixed.mul(derivative)?)?;
                        for (other_index, (other, other_varies)) in operands.iter().enumerate() {
                            if !other_varies {
                                continue;
                            }
                            let partial = x.elementwise(y, |base, exponent| {
                                pow_second_derivatives(base, exponent)[2 * index + other_index]
                            })?;
                            mixed = mixed.add(&operand.first.mul(&other.second)?.mul(&partial)?)?;
                        }
                    }
                    MixedTangent {
                        value: x.pow(y)?,
                        first,
                        second,
                        mixed,
                    }
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
                TensorOp::Solve { matrix, rhs } => {
                    let matrix = values
                        .get(*matrix)
                        .ok_or_else(|| format!("node {matrix} has no evaluated value"))?;
                    let rhs = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
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
                TensorOp::Tanh { input } => {
                    let input = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let value = input.value.tanh()?;
                    let derivative = value.tanh_derivative_from_output()?;
                    let second_derivative = value.tanh_second_derivative_from_output()?;
                    MixedTangent {
                        first: input.first.mul(&derivative)?,
                        second: input.second.mul(&derivative)?,
                        mixed: input
                            .mixed
                            .mul(&derivative)?
                            .add(&input.first.mul(&input.second)?.mul(&second_derivative)?)?,
                        value,
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
                let op = remap_tensor_op(&node.op, &remap)?;
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
        self.evaluate_with_tape(initial_carry, external_inputs)
            .map(|(output, _)| output)
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
        let (output, tape) = self.evaluate_with_tape(initial_carry, external_inputs)?;
        let mut carry_cotangent = output_cotangent;
        let mut external_gradients = self
            .external_captures
            .iter()
            .map(|(name, shape)| {
                DynamicTensor::filled(shape.clone(), 0.0).map(|value| (name.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        for (offset, carry) in tape.carries[..tape.carries.len() - 1]
            .iter()
            .cloned()
            .enumerate()
            .rev()
        {
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
                let accumulated = external_gradients
                    .get_mut(name)
                    .ok_or_else(|| format!("fori loop external gradient {name:?} is missing"))?;
                *accumulated = accumulated.add(contribution)?;
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
        let mut carries = vec![initial_carry.clone()];
        let mut carry_tangents = vec![initial_tangent.clone()];
        let mut carry = initial_carry;
        let mut carry_tangent = initial_tangent;
        for index in self.loop_plan.lower..self.loop_plan.upper {
            let inputs = self
                .loop_plan
                .body_inputs(carry.clone(), index, external_inputs)?;
            let tangents = self
                .loop_plan
                .body_tangents(carry_tangent.clone(), external_tangents)?;
            let (next, next_tangent) = self.loop_plan.body.plan.as_ir().jvp(
                self.loop_plan.body.plan.output_node_id,
                &inputs,
                &tangents,
            )?;
            carry = next;
            carry_tangent = next_tangent;
            carries.push(carry.clone());
            carry_tangents.push(carry_tangent.clone());
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
        for offset in (0..self.loop_plan.upper - self.loop_plan.lower).rev() {
            let inputs = self.loop_plan.body_inputs(
                carries[offset].clone(),
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
                carry_tangents[offset].clone(),
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
                *accumulated = accumulated.add(&contribution)?;
            }
            carry_cotangent = body_gradients
                .get(&self.loop_plan.carry_name)
                .cloned()
                .ok_or_else(|| "Fori body VJP has no carry gradient".to_string())?;
            carry_cotangent_tangent = next_carry_cotangent_tangent;
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
        let mut carries = vec![initial_carry.clone()];
        let mut carry_tangents = vec![initial_tangent.clone()];
        let mut carry = initial_carry;
        let mut carry_tangent = initial_tangent;
        for index in self.scan_plan.lower..self.scan_plan.upper {
            let inputs = self.scan_plan.inputs(carry, index, external_inputs)?;
            let tangents = self.scan_plan.tangents(carry_tangent, external_tangents)?;
            let (next_carry, next_tangent) = self.scan_plan.body.plan.as_ir().jvp(
                self.scan_plan.body.plan.output_node_ids[0],
                &inputs,
                &tangents,
            )?;
            carry = next_carry;
            carry_tangent = next_tangent;
            carries.push(carry.clone());
            carry_tangents.push(carry_tangent.clone());
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
        for offset in (0..self.scan_plan.upper - self.scan_plan.lower).rev() {
            let inputs = self.scan_plan.inputs(
                carries[offset].clone(),
                self.scan_plan.lower + offset,
                external_inputs,
            )?;
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
                carry_tangents[offset].clone(),
            );
            for name in self.scan_plan.external_captures.keys() {
                if let Some(tangent) = external_tangents.get(name) {
                    let tangent_name = self
                        .tangent_names
                        .get(name)
                        .ok_or_else(|| format!("Scan VJP JVP has no tangent input for {name:?}"))?;
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
                    .ok_or_else(|| format!("Scan VJP JVP has no carry gradient plan for {name:?}"))?
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
                *accumulated = accumulated.add(&contribution)?;
            }
            carry_cotangent = primal_gradients
                .get(&self.scan_plan.carry_name)
                .cloned()
                .ok_or_else(|| "Scan body VJP has no carry gradient".to_string())?;
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
        self.evaluate_with_tape(initial_carry, external_inputs)
            .map(|(carry, outputs, _)| (carry, outputs))
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
            let (next_carry, next_tangent) =
                body.jvp(self.body.plan.output_node_ids[0], &inputs, &tangents)?;
            let (output, output_tangent) =
                body.jvp(self.body.plan.output_node_ids[1], &inputs, &tangents)?;
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
        let (final_carry, outputs, tape) =
            self.evaluate_with_tape(initial_carry, external_inputs)?;
        if final_carry_cotangent.shape != final_carry.shape
            || output_cotangent.shape != outputs.shape
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
        for (offset, carry) in tape.carries[..tape.carries.len() - 1]
            .iter()
            .cloned()
            .enumerate()
            .rev()
        {
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
                *accumulated = accumulated.add(contribution)?;
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
        self.nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name: candidate } if candidate == name => {
                    Some(node.shape.clone())
                }
                _ => None,
            })
            .ok_or_else(|| format!("execution plan input {name:?} does not exist"))
    }

    pub fn input_dtype(&self, name: &str) -> Result<TensorDType, String> {
        self.nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name: candidate } if candidate == name => Some(node.dtype),
                _ => None,
            })
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
        for (placed, node) in sharding.program.nodes.iter().zip(&self.nodes) {
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
            .zip(&self.nodes)
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
        build_tensor_buffer_plan(&self.nodes, self.output_node_id)
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
    if (index < count) output[index] = {expression};\n\
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
    if (index < count) output[index] = {expression};\n\
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
        if self.fused_elementwise_output {
            return self.evaluate(inputs).map(|value| vec![value]);
        }
        let values = TensorIr::evaluate_tensor_nodes(&self.nodes, inputs)?;
        self.output_node_ids
            .iter()
            .map(|output| {
                values
                    .get(*output)
                    .cloned()
                    .ok_or_else(|| format!("output node {output} has no value"))
            })
            .collect()
    }

    fn execute_cpu(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        if self.fused_elementwise_output {
            return evaluate_fused_elementwise(&self.nodes, self.output_node_id, inputs);
        }
        TensorIr::evaluate_tensor_nodes(&self.nodes, inputs)?
            .get(self.output_node_id)
            .cloned()
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
        }
    }

    fn specialize_mapped_axis_zero(
        &self,
        mapped_input_names: &BTreeSet<String>,
        shard_extent: usize,
    ) -> Result<TensorIr, String> {
        let mut specialized = TensorIr::new();
        let mut remap = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
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
                TensorOp::Matmul { lhs, rhs } => {
                    specialized.matmul(mapped(*lhs)?, mapped(*rhs)?)?
                }
                TensorOp::Solve { matrix, rhs } => {
                    specialized.solve(mapped(*matrix)?, mapped(*rhs)?)?
                }
                TensorOp::Triangular { input, lower } => {
                    specialized.triangular(mapped(*input)?, *lower)?
                }
                TensorOp::Tanh { input } => specialized.tanh(mapped(*input)?)?,
                TensorOp::Exp { input } => specialized.exp(mapped(*input)?)?,
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
                TensorOp::Sin { input } => specialized.sin(mapped(*input)?)?,
                TensorOp::Cos { input } => specialized.cos(mapped(*input)?)?,
                TensorOp::Powi { input, exponent } => {
                    specialized.powi(mapped(*input)?, *exponent)?
                }
                TensorOp::Pow { base, exponent } => {
                    specialized.pow(mapped(*base)?, mapped(*exponent)?)?
                }
                TensorOp::Transpose { input, axes } => specialized.transpose(
                    mapped(*input)?,
                    Some(axes.iter().map(|axis| *axis as isize).collect()),
                )?,
                TensorOp::Log { input } => specialized.log(mapped(*input)?)?,
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
                TensorOp::Broadcast { input } => {
                    specialized.broadcast_to(mapped(*input)?, node.shape.clone())?
                }
                TensorOp::Cond { .. } => {
                    return Err(
                        "batch specialization does not yet transform Cond regions".to_string()
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
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA lowering references missing node {node_id}"))?;
    let child = |child_id| cuda_expression(nodes, child_id);
    match &node.op {
        TensorOp::Input { .. } => Ok(format!("input_{node_id}[quabla_offset_{node_id}(index)]")),
        TensorOp::ScalarConstant { value } if value.is_finite() => Ok(cuda_scalar_literal(*value)),
        TensorOp::ScalarConstant { .. } => Err(
            "CUDA lowering does not support non-finite scalar constants".to_string(),
        ),
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
        TensorOp::Tanh { input } => Ok(format!("tanhf({})", child(*input)?)),
        TensorOp::Exp { input } => Ok(format!("expf({})", child(*input)?)),
        TensorOp::Sqrt { input } => Ok(format!("sqrtf({})", child(*input)?)),
        TensorOp::SqrtDerivative { input, order } => {
            Ok(cuda_sqrt_derivative_expression(&child(*input)?, *order))
        }
        TensorOp::Sin { input } => Ok(format!("sinf({})", child(*input)?)),
        TensorOp::Cos { input } => Ok(format!("cosf({})", child(*input)?)),
        TensorOp::Powi { input, exponent } => Ok(format!(
            "quabla_powi({}, {}U)",
            child(*input)?,
            exponent
        )),
        TensorOp::Pow { base, exponent } => {
            Ok(format!("powf({}, {})", child(*base)?, child(*exponent)?))
        }
        TensorOp::Div { .. } | TensorOp::Log { .. } => Err(
            "CUDA lowering does not yet support div or log because their CPU execution has checked domain semantics"
                .to_string(),
        ),
        TensorOp::Sum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Solve { .. }
        | TensorOp::Triangular { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Broadcast { .. }
        | TensorOp::Cond { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. } => Err(format!(
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
    region_output_node_id: TensorNodeId,
) -> Result<String, String> {
    if leaves.contains(&node_id) {
        return Ok(format!(
            "input_{node_id}[{}(index)]",
            cuda_fusion_region_offset_function_name(region_output_node_id, node_id)
        ));
    }
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA fusion region references missing node {node_id}"))?;
    let child = |child_id| cuda_region_expression(nodes, child_id, leaves, region_output_node_id);
    match &node.op {
        TensorOp::ScalarConstant { value } if value.is_finite() => Ok(cuda_scalar_literal(*value)),
        TensorOp::ScalarConstant { .. } => {
            Err("CUDA fusion region does not support non-finite scalar constants".to_string())
        }
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
        TensorOp::Tanh { input } => Ok(format!("tanhf({})", child(*input)?)),
        TensorOp::Exp { input } => Ok(format!("expf({})", child(*input)?)),
        TensorOp::Sqrt { input } => Ok(format!("sqrtf({})", child(*input)?)),
        TensorOp::SqrtDerivative { input, order } => {
            Ok(cuda_sqrt_derivative_expression(&child(*input)?, *order))
        }
        TensorOp::Sin { input } => Ok(format!("sinf({})", child(*input)?)),
        TensorOp::Cos { input } => Ok(format!("cosf({})", child(*input)?)),
        TensorOp::Powi { input, exponent } => {
            Ok(format!("quabla_powi({}, {}U)", child(*input)?, exponent))
        }
        TensorOp::Pow { base, exponent } => {
            Ok(format!("powf({}, {})", child(*base)?, child(*exponent)?))
        }
        _ => Err(format!(
            "CUDA fusion region cannot inline {} node {node_id}",
            tensor_op_name(&node.op)
        )),
    }
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

fn cuda_scalar_literal(value: f64) -> String {
    let value = value as f32;
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value:?}f")
    }
}

fn is_fusable_elementwise_subgraph(nodes: &[TensorNode], node_id: TensorNodeId) -> bool {
    let Some(node) = nodes.get(node_id) else {
        return false;
    };
    match &node.op {
        TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } => true,
        // A whole-plan kernel binds only plan inputs; a plan with an array constant runs the
        // per-node program, where fusion regions read the constant's buffer.
        TensorOp::Constant { .. } => false,
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Greater { lhs, rhs }
        | TensorOp::Compare { lhs, rhs, .. }
        | TensorOp::Pow {
            base: lhs,
            exponent: rhs,
        } => {
            is_fusable_elementwise_subgraph(nodes, *lhs)
                && is_fusable_elementwise_subgraph(nodes, *rhs)
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => {
            is_fusable_elementwise_subgraph(nodes, *condition)
                && is_fusable_elementwise_subgraph(nodes, *on_true)
                && is_fusable_elementwise_subgraph(nodes, *on_false)
        }
        TensorOp::Tanh { input }
        | TensorOp::Exp { input }
        | TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Sin { input }
        | TensorOp::Cos { input }
        | TensorOp::Powi { input, .. }
        | TensorOp::Cast { input } => is_fusable_elementwise_subgraph(nodes, *input),
        TensorOp::Sum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Solve { .. }
        | TensorOp::Triangular { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Div { .. }
        | TensorOp::Log { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Broadcast { .. }
        | TensorOp::Cond { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. } => false,
    }
}

fn is_fusable_elementwise_compute_op(op: &TensorOp) -> bool {
    matches!(
        op,
        TensorOp::Add { .. }
            | TensorOp::Sub { .. }
            | TensorOp::Mul { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. }
            | TensorOp::Where { .. }
            | TensorOp::Tanh { .. }
            | TensorOp::Exp { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Sin { .. }
            | TensorOp::Cos { .. }
            | TensorOp::Powi { .. }
            | TensorOp::Pow { .. }
            | TensorOp::Cast { .. }
    )
}

fn evaluate_fused_elementwise(
    nodes: &[TensorNode],
    output_node_id: TensorNodeId,
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<DynamicTensor, String> {
    let output = nodes
        .get(output_node_id)
        .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;
    for node in nodes {
        if let TensorOp::Input { name } = &node.op {
            let input = inputs
                .get(name)
                .ok_or_else(|| format!("missing input {name:?}"))?;
            if input.shape != node.shape {
                return Err(format!(
                    "input {name:?} has shape {:?}, expected {:?}",
                    input.shape, node.shape
                ));
            }
        }
    }

    let count = element_count(&output.shape)?;
    let mut data = Vec::with_capacity(count);
    for index in 0..count {
        data.push(evaluate_fused_element(
            nodes,
            output_node_id,
            index,
            &output.shape,
            inputs,
        )?);
    }
    DynamicTensor::new(output.shape.clone(), data).map(|value| value.into_dtype(output.dtype))
}

fn evaluate_fused_element(
    nodes: &[TensorNode],
    node_id: TensorNodeId,
    output_index: usize,
    output_shape: &[usize],
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<f64, String> {
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("node {node_id} does not exist"))?;
    let child =
        |child_id| evaluate_fused_element(nodes, child_id, output_index, output_shape, inputs);
    let value = match &node.op {
        TensorOp::Input { name } => {
            let input = inputs
                .get(name)
                .ok_or_else(|| format!("missing input {name:?}"))?;
            let strides = contiguous_strides(&node.shape);
            Ok(input.data[broadcast_offset(output_index, output_shape, &node.shape, &strides)])
        }
        TensorOp::ScalarConstant { value } => Ok(*value),
        TensorOp::Constant { .. } => {
            Err("the fused elementwise evaluator does not read array constants".to_string())
        }
        TensorOp::Cast { input } => child(*input),
        TensorOp::Add { lhs, rhs } => Ok(child(*lhs)? + child(*rhs)?),
        TensorOp::Sub { lhs, rhs } => Ok(child(*lhs)? - child(*rhs)?),
        TensorOp::Div { lhs, rhs } => {
            let denominator = child(*rhs)?;
            if denominator == 0.0 {
                return Err("division by zero is not supported".to_string());
            }
            Ok(child(*lhs)? / denominator)
        }
        TensorOp::Mul { lhs, rhs } => Ok(child(*lhs)? * child(*rhs)?),
        TensorOp::Greater { lhs, rhs } => Ok(f64::from(child(*lhs)? > child(*rhs)?)),
        TensorOp::Compare { lhs, rhs, kind } => {
            Ok(f64::from(kind.evaluate(child(*lhs)?, child(*rhs)?)))
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => {
            if child(*condition)? != 0.0 {
                child(*on_true)
            } else {
                child(*on_false)
            }
        }
        TensorOp::Tanh { input } => Ok(child(*input)?.tanh()),
        TensorOp::Exp { input } => Ok(child(*input)?.exp()),
        TensorOp::Sqrt { input } => {
            let value = child(*input)?;
            Ok(value.sqrt())
        }
        TensorOp::SqrtDerivative { input, order } => {
            Ok(sqrt_derivative_value(child(*input)?, *order))
        }
        TensorOp::Sin { input } => Ok(child(*input)?.sin()),
        TensorOp::Cos { input } => Ok(child(*input)?.cos()),
        TensorOp::Powi { input, exponent } => {
            let exponent = i32::try_from(*exponent)
                .map_err(|_| "powi exponent must fit in a signed 32-bit integer".to_string())?;
            Ok(child(*input)?.powi(exponent))
        }
        TensorOp::Pow { base, exponent } => Ok(child(*base)?.powf(child(*exponent)?)),
        TensorOp::Log { input } => {
            let value = child(*input)?;
            if value <= 0.0 {
                return Err("log requires strictly positive tensor values".to_string());
            }
            Ok(value.ln())
        }
        TensorOp::Sum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Solve { .. }
        | TensorOp::Triangular { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Broadcast { .. }
        | TensorOp::Cond { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. } => Err(format!(
            "node {node_id} is not supported by the fused elementwise evaluator"
        )),
    };
    // Same as the per-node interpreter: every fused node rounds to its own dtype.
    value.map(|value| node.dtype.round(value))
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
        TensorOp::Pow { base, exponent } => vec![*base, *exponent],
        TensorOp::Solve { matrix, rhs } => vec![*matrix, *rhs],
        TensorOp::Triangular { input, .. } => vec![*input],
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
        | TensorOp::Tanh { input }
        | TensorOp::Exp { input }
        | TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Reshape { input }
        | TensorOp::Mean { input }
        | TensorOp::MeanAxis { input, .. }
        | TensorOp::Sin { input }
        | TensorOp::Cos { input }
        | TensorOp::Powi { input, .. }
        | TensorOp::Transpose { input, .. }
        | TensorOp::Log { input }
        | TensorOp::Slice { input, .. }
        | TensorOp::PadSlice { input, .. }
        | TensorOp::Broadcast { input }
        | TensorOp::Cast { input } => {
            vec![*input]
        }
        TensorOp::Concat { inputs, .. } => inputs.clone(),
    }
}

/// Operands whose dtype an ordinary op's result inherits. A `where` mask
/// is only tested for non-zero values and therefore does not participate.
fn tensor_value_operands(op: &TensorOp) -> Vec<TensorNodeId> {
    match op {
        TensorOp::Where {
            on_true, on_false, ..
        } => vec![*on_true, *on_false],
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
    for node in &plan.nodes {
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
    let scalar = *value
        .data
        .first()
        .ok_or_else(|| "scalar conditional predicate has no value".to_string())?;
    if !scalar.is_finite() {
        return Err("conditional predicate must be finite".to_string());
    }
    Ok(scalar != 0.0)
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
        TensorOp::Pow { base, exponent } => merge(&[*base, *exponent]),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => merge(&[*condition, *on_true, *on_false]),
        TensorOp::Cond { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. } => Err(format!(
            "kernel node {node_id} contains {} regions; placement propagation requires explicit region lowering",
            tensor_op_name(&node.op)
        )),
        TensorOp::Tanh { input }
        | TensorOp::Exp { input }
        | TensorOp::Sqrt { input }
        | TensorOp::SqrtDerivative { input, .. }
        | TensorOp::Sin { input }
        | TensorOp::Cos { input }
        | TensorOp::Powi { input, .. }
        | TensorOp::Log { input }
        | TensorOp::Cast { input } => unary(*input),
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
        TensorOp::Triangular { input, .. } => {
            let placement = unary(*input)?;
            reject_sharded_placement(node_id, &placement, "triangular lowering")?;
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
        TensorOp::Slice { input, axis, .. } | TensorOp::PadSlice { input, axis, .. } => {
            let placement = unary(*input)?;
            if sharded_tensor_axis(&placement) == Some(*axis) {
                return Err(format!(
                    "kernel node {node_id} slices along a sharded axis; explicit redistribution is required"
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
        TensorOp::Pow { .. } => "pow",
        TensorOp::Transpose { .. } => "transpose",
        TensorOp::Log { .. } => "log",
        TensorOp::Concat { .. } => "concat",
        TensorOp::Slice { .. } => "slice",
        TensorOp::PadSlice { .. } => "pad_slice",
        TensorOp::Broadcast { .. } => "broadcast",
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
        TensorOp::Tanh { input } => Some(scalar(*input)?.tanh()),
        TensorOp::Exp { input } => Some(scalar(*input)?.exp()),
        TensorOp::Sin { input } => Some(scalar(*input)?.sin()),
        TensorOp::Cos { input } => Some(scalar(*input)?.cos()),
        TensorOp::Powi { input, exponent } => {
            let exponent = i32::try_from(*exponent).ok()?;
            Some(scalar(*input)?.powi(exponent))
        }
        TensorOp::Pow { base, exponent } => Some(scalar(*base)?.powf(scalar(*exponent)?)),
        TensorOp::Log { input } => {
            let value = scalar(*input)?;
            (value > 0.0).then(|| value.ln())
        }
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
        TensorOp::Tanh { input } => constant(*input)?.tanh().ok(),
        TensorOp::Exp { input } => constant(*input)?.exp().ok(),
        TensorOp::Sin { input } => constant(*input)?.sin().ok(),
        TensorOp::Cos { input } => constant(*input)?.cos().ok(),
        TensorOp::Powi { input, exponent } => constant(*input)?.powi(*exponent).ok(),
        TensorOp::Pow { base, exponent } => constant(*base)?.pow(&*constant(*exponent)?).ok(),
        TensorOp::Log { input } => constant(*input)?.log().ok(),
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
        _ => Ok(None),
    }
}

fn build_tensor_fusion_regions(nodes: &[TensorNode]) -> Vec<TensorFusionRegion> {
    let mut users = vec![Vec::new(); nodes.len()];
    for (node_id, node) in nodes.iter().enumerate() {
        for input in tensor_op_inputs(&node.op) {
            if let Some(input_users) = users.get_mut(input) {
                if !input_users.contains(&node_id) {
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

fn prune_unreachable_tensor_nodes(
    nodes: Vec<TensorNode>,
    outputs: &[TensorNodeId],
) -> Result<(Vec<TensorNode>, Vec<TensorNodeId>), String> {
    let mut reachable = HashSet::new();
    let mut pending = outputs.to_vec();
    while let Some(node_id) = pending.pop() {
        if !reachable.insert(node_id) {
            continue;
        }
        let node = nodes
            .get(node_id)
            .ok_or_else(|| format!("execution plan output node {node_id} is missing"))?;
        pending.extend(tensor_op_inputs(&node.op));
    }

    let mut remap = HashMap::new();
    let mut compacted = Vec::with_capacity(reachable.len());
    for (old_id, node) in nodes.iter().enumerate() {
        if !reachable.contains(&old_id) {
            continue;
        }
        remap.insert(old_id, compacted.len());
        compacted.push(TensorNode {
            op: remap_tensor_op(&node.op, &remap)?,
            shape: node.shape.clone(),
            dtype: node.dtype,
            weak: node.weak,
        });
    }
    let remapped_outputs = outputs
        .iter()
        .map(|output| {
            remap
                .get(output)
                .copied()
                .ok_or_else(|| format!("execution plan output node {output} is unreachable"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((compacted, remapped_outputs))
}

fn build_tensor_buffer_plan(
    nodes: &[TensorNode],
    output_node_id: TensorNodeId,
) -> Result<TensorBufferPlan, String> {
    let output_backing_node_id = tensor_storage_root(nodes, output_node_id)?;
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
            if *uses == 0 && root != output_backing_node_id {
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

fn pure_tensor_op_cse_key(op: &TensorOp, shape: &[usize]) -> Option<String> {
    let key = match op {
        TensorOp::Input { .. } => return None,
        TensorOp::ScalarConstant { value } => format!("constant:{value}:{shape:?}"),
        // Array constants are deduplicated by value in `compile_cpu_many`, without formatting
        // their elements into a key.
        TensorOp::Constant { .. } => return None,
        TensorOp::Cast { input } => format!("cast:{input}:{shape:?}"),
        TensorOp::Add { lhs, rhs } => format!("add:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Sub { lhs, rhs } => format!("sub:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Div { lhs, rhs } => format!("div:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Mul { lhs, rhs } => format!("mul:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Greater { lhs, rhs } => format!("greater:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Compare { lhs, rhs, kind } => {
            format!("compare:{}:{lhs}:{rhs}:{shape:?}", kind.name())
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => format!("where:{condition}:{on_true}:{on_false}:{shape:?}"),
        // Region identity is intentionally not CSE'd: control-flow owns
        // executable region plans, not only scalar operands.
        TensorOp::Cond { .. }
        | TensorOp::Fori { .. }
        | TensorOp::ForiJvp { .. }
        | TensorOp::ForiVjp { .. }
        | TensorOp::ForiVjpJvp { .. }
        | TensorOp::Scan { .. }
        | TensorOp::ScanVjp { .. }
        | TensorOp::ScanVjpJvp { .. } => return None,
        TensorOp::Sum { input } => format!("sum:{input}:{shape:?}"),
        TensorOp::SumAxis { input, axis } => format!("sum_axis:{input}:{axis}:{shape:?}"),
        TensorOp::Matmul { lhs, rhs } => format!("matmul:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Solve { matrix, rhs } => format!("solve:{matrix}:{rhs}:{shape:?}"),
        TensorOp::Triangular { input, lower } => format!("triangular:{input}:{lower}:{shape:?}"),
        TensorOp::Tanh { input } => format!("tanh:{input}:{shape:?}"),
        TensorOp::Exp { input } => format!("exp:{input}:{shape:?}"),
        TensorOp::Sqrt { input } => format!("sqrt:{input}:{shape:?}"),
        TensorOp::SqrtDerivative { input, order } => {
            format!("sqrt_derivative:{input}:{order}:{shape:?}")
        }
        TensorOp::Reshape { input } => format!("reshape:{input}:{shape:?}"),
        TensorOp::Mean { input } => format!("mean:{input}:{shape:?}"),
        TensorOp::MeanAxis { input, axis } => format!("mean_axis:{input}:{axis}:{shape:?}"),
        TensorOp::Sin { input } => format!("sin:{input}:{shape:?}"),
        TensorOp::Cos { input } => format!("cos:{input}:{shape:?}"),
        TensorOp::Powi { input, exponent } => format!("powi:{input}:{exponent}:{shape:?}"),
        TensorOp::Pow { base, exponent } => format!("pow:{base}:{exponent}:{shape:?}"),
        TensorOp::Transpose { input, axes } => format!("transpose:{input}:{axes:?}:{shape:?}"),
        TensorOp::Log { input } => format!("log:{input}:{shape:?}"),
        TensorOp::Concat { inputs, axis } => format!("concat:{inputs:?}:{axis}:{shape:?}"),
        TensorOp::Slice {
            input,
            axis,
            start,
            length,
        } => {
            format!("slice:{input}:{axis}:{start}:{length}:{shape:?}")
        }
        TensorOp::PadSlice { input, axis, start } => {
            format!("pad_slice:{input}:{axis}:{start}:{shape:?}")
        }
        TensorOp::Broadcast { input } => format!("broadcast:{input}:{shape:?}"),
    };
    Some(key)
}

/// The execution-group id of a node that shares one region execution with
/// its sibling results, if the op has one.
fn tensor_op_group_mut(op: &mut TensorOp) -> Option<&mut usize> {
    match op {
        TensorOp::ForiVjp { group, .. }
        | TensorOp::ForiVjpJvp { group, .. }
        | TensorOp::Scan { group, .. }
        | TensorOp::ScanVjp { group, .. }
        | TensorOp::ScanVjpJvp { group, .. } => Some(group),
        _ => None,
    }
}

fn remap_tensor_op(
    op: &TensorOp,
    remap: &HashMap<TensorNodeId, TensorNodeId>,
) -> Result<TensorOp, String> {
    let remap_node = |node_id: TensorNodeId| {
        remap
            .get(&node_id)
            .copied()
            .ok_or_else(|| format!("node {node_id} is missing from execution plan remap"))
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
        TensorOp::Matmul { lhs, rhs } => Ok(TensorOp::Matmul {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Solve { matrix, rhs } => Ok(TensorOp::Solve {
            matrix: remap_node(*matrix)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Triangular { input, lower } => Ok(TensorOp::Triangular {
            input: remap_node(*input)?,
            lower: *lower,
        }),
        TensorOp::Tanh { input } => Ok(TensorOp::Tanh {
            input: remap_node(*input)?,
        }),
        TensorOp::Exp { input } => Ok(TensorOp::Exp {
            input: remap_node(*input)?,
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
        TensorOp::Sin { input } => Ok(TensorOp::Sin {
            input: remap_node(*input)?,
        }),
        TensorOp::Cos { input } => Ok(TensorOp::Cos {
            input: remap_node(*input)?,
        }),
        TensorOp::Powi { input, exponent } => Ok(TensorOp::Powi {
            input: remap_node(*input)?,
            exponent: *exponent,
        }),
        TensorOp::Pow { base, exponent } => Ok(TensorOp::Pow {
            base: remap_node(*base)?,
            exponent: remap_node(*exponent)?,
        }),
        TensorOp::Transpose { input, axes } => Ok(TensorOp::Transpose {
            input: remap_node(*input)?,
            axes: axes.clone(),
        }),
        TensorOp::Log { input } => Ok(TensorOp::Log {
            input: remap_node(*input)?,
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
        TensorOp::Broadcast { input } => Ok(TensorOp::Broadcast {
            input: remap_node(*input)?,
        }),
    }
}

fn accumulate(slot: &mut Option<DynamicTensor>, contribution: DynamicTensor) -> Result<(), String> {
    match slot {
        Some(existing) => {
            *existing = existing.add(&contribution)?;
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

/// `d pow(x, y) / dx = y * x^(y-1)`, defined as `0` where `x == 0` and `y < 1`.
///
/// Those are the points where the derivative is infinite (`0 < y < 1`) or the
/// value itself is (`y < 0`), or where the formula is `0 * inf` (`y == 0`).
/// The zero subgradient matches `sqrt` at the origin (`SqrtDerivative`), so
/// `pow(x, 0.5)` and `sqrt(x)` agree there. For `y >= 1` the finite limit is
/// kept (`1` for `y == 1`, `0` above), so `x ** 2.0` keeps the derivative
/// `2x` and the second derivative `2` at the origin. A negative base with an
/// integer-valued exponent follows `powf` and stays finite; with a
/// non-integer exponent the value and this derivative are NaN.
///
/// The symbolic rule (`TensorIr::pow_base_derivative`) evaluates the same
/// expression with the singular points masked out of the inner power, so its
/// own derivatives stay finite there instead of producing `0 * inf`.
fn pow_base_derivative(base: f64, exponent: f64) -> f64 {
    if pow_base_is_singular(base, exponent) {
        0.0
    } else {
        exponent * base.powf(exponent - 1.0)
    }
}

fn pow_base_is_singular(base: f64, exponent: f64) -> bool {
    base == 0.0 && exponent < 1.0
}

/// `d pow(x, y) / dy = x^y * ln(x)` for `x > 0`, defined as `0` for `x <= 0`.
///
/// At `x == 0` the formula is `0 * -inf` or `inf * -inf`; the power is
/// piecewise constant along `y` there (`inf`, `1` at `y == 0`, then `0`), so
/// zero is its derivative everywhere except at the jump. For `x < 0` the real power exists only at
/// integer exponents, so no derivative in `y` exists; unlike JAX, which
/// returns NaN there, the gradient is `0`, keeping the backward pass finite
/// for a learnable exponent at an integer value with negative bases (the
/// value itself is NaN at every non-integer exponent). A NaN base propagates.
fn pow_exponent_derivative(base: f64, exponent: f64) -> f64 {
    if base <= 0.0 {
        0.0
    } else {
        base.powf(exponent) * base.ln()
    }
}

/// Second partial derivatives of `pow` as the symbolic rules compose them:
/// `(d/dx d/dx, d/dy d/dx, d/dx d/dy, d/dy d/dy)`. Each zero region of a
/// first derivative is also a zero region of its derivatives, so the two
/// mixed partials may differ at `x <= 0`, where the conventions apply.
fn pow_second_derivatives(base: f64, exponent: f64) -> [f64; 4] {
    let (base_base, base_exponent) = if pow_base_is_singular(base, exponent) {
        (0.0, 0.0)
    } else {
        (
            exponent * pow_base_derivative(base, exponent - 1.0),
            base.powf(exponent - 1.0) + exponent * pow_exponent_derivative(base, exponent - 1.0),
        )
    };
    let (exponent_base, exponent_exponent) = if base <= 0.0 {
        (0.0, 0.0)
    } else {
        let log = base.ln();
        (
            pow_base_derivative(base, exponent) * log + base.powf(exponent) / base,
            base.powf(exponent) * log * log,
        )
    };
    [base_base, base_exponent, exponent_base, exponent_exponent]
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

fn solve_shape(matrix: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    if matrix.len() != 2 || rhs.len() != 2 {
        return Err(format!(
            "solve requires rank-2 matrix and right-hand side tensors, got {:?} and {:?}",
            matrix, rhs
        ));
    }
    if matrix[0] != matrix[1] {
        return Err(format!(
            "solve requires a square coefficient matrix, got {:?}",
            matrix
        ));
    }
    if matrix[0] != rhs[0] {
        return Err(format!(
            "solve requires matrix shape {:?} and right-hand side shape {:?} to agree on rows",
            matrix, rhs
        ));
    }
    Ok(rhs.to_vec())
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
