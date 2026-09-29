//! Stable compiler-facing facade over the Tensor IR and execution backends.
//!
//! The tensor IR remains the implementation layer for tracing and AD.  This
//! module owns the public lifecycle: build a program, apply a transform,
//! freeze it, select a backend, then execute it.  Backend-specific execution
//! plans intentionally remain implementation details behind [`QuablaExecutable`].

use std::collections::BTreeMap;

use crate::tensor_ir::{
    CudaBackend, CudaExecutionPlan, DynamicTensor, MlxBackend, SymbolicJvp, SymbolicVjp,
    TensorBackend, TensorDeviceBackend, TensorExecutionPlan, TensorIr, TensorNodeId,
};

/// Backend requested when compiling a frozen Quabla program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuablaTarget {
    Cpu,
    Cuda { device_ordinal: usize },
    Mlx,
}

impl QuablaTarget {
    pub const fn backend(self) -> TensorDeviceBackend {
        match self {
            Self::Cpu => TensorDeviceBackend::Cpu,
            Self::Cuda { .. } => TensorDeviceBackend::Cuda,
            Self::Mlx => TensorDeviceBackend::Mlx,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda { .. } => "cuda",
            Self::Mlx => "mlx",
        }
    }

    /// Whether this build can attempt to compile for this target.
    ///
    /// This deliberately reports build availability only. A particular program
    /// can still be rejected when its operations have no verified lowering.
    pub const fn is_built(self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda { .. } => cfg!(all(feature = "cuda", target_os = "linux")),
            Self::Mlx => cfg!(all(feature = "mlx", target_os = "macos")),
        }
    }
}

/// Static compiler capability information for one target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuablaCapability {
    pub target: QuablaTarget,
    pub built: bool,
    pub logical_backend: TensorDeviceBackend,
}

/// A traceable Tensor IR program with one selected result.
#[derive(Clone, Debug)]
pub struct QuablaProgram {
    ir: TensorIr,
    output: TensorNodeId,
}

impl QuablaProgram {
    pub fn new(ir: TensorIr, output: TensorNodeId) -> Result<Self, String> {
        ir.node_shape(output)?;
        Ok(Self { ir, output })
    }

    pub fn output_node_id(&self) -> TensorNodeId {
        self.output
    }

    pub fn output_shape(&self) -> Result<Vec<usize>, String> {
        self.ir.node_shape(self.output)
    }

    pub fn ir(&self) -> &TensorIr {
        &self.ir
    }

    pub fn lower_text(&self) -> String {
        self.ir.lower_text()
    }

    pub fn freeze(&self) -> Result<TensorExecutionPlan, String> {
        self.ir.compile_cpu(self.output)
    }

    pub fn jvp(
        &self,
        tangent_inputs: &BTreeMap<String, String>,
    ) -> Result<QuablaJvpProgram, String> {
        let SymbolicJvp {
            graph,
            value,
            tangent,
        } = self
            .ir
            .symbolic_jvp_with_tangent_inputs(self.output, tangent_inputs)?;
        Ok(QuablaJvpProgram {
            program: Self::new(graph, value)?,
            tangent,
        })
    }

    pub fn vjp(&self, cotangent_name: &str) -> Result<QuablaVjpProgram, String> {
        let SymbolicVjp {
            graph,
            value,
            cotangent,
            gradients,
        } = self.ir.symbolic_vjp(self.output, cotangent_name)?;
        Ok(QuablaVjpProgram {
            program: Self::new(graph, value)?,
            cotangent,
            gradients,
        })
    }
}

/// A Tensor IR program with an ordered list of results frozen into one plan.
///
/// Backends evaluate the shared prefix once, which value-and-gradient and
/// primal/tangent helpers rely on. Freezing prunes and deduplicates nodes, so
/// the frozen node ids differ from [`Self::output_node_ids`];
/// [`QuablaMultiOutputExecutable::output_node_ids`] reports them in the same
/// order. Symbolic transforms stay on the single-output [`QuablaProgram`].
#[derive(Clone, Debug)]
pub struct QuablaMultiOutputProgram {
    ir: TensorIr,
    outputs: Vec<TensorNodeId>,
}

impl QuablaMultiOutputProgram {
    pub fn new(ir: TensorIr, outputs: Vec<TensorNodeId>) -> Result<Self, String> {
        if outputs.is_empty() {
            return Err("execution plan requires at least one output".to_string());
        }
        for output in &outputs {
            ir.node_shape(*output)?;
        }
        Ok(Self { ir, outputs })
    }

    /// Source-IR node ids of the results, in program order.
    pub fn output_node_ids(&self) -> &[TensorNodeId] {
        &self.outputs
    }

    pub fn ir(&self) -> &TensorIr {
        &self.ir
    }

    /// Freezes every result into one plan whose `output_node_ids()` follow
    /// program order.
    pub fn freeze(&self) -> Result<TensorExecutionPlan, String> {
        self.ir
            .compile_cpu_many(&self.outputs)
            .map(|(plan, _)| plan)
    }
}

/// A forward-mode transformed program. The primal is the embedded program's
/// output; `tangent_node_id` is evaluated from the same transformed IR.
#[derive(Clone, Debug)]
pub struct QuablaJvpProgram {
    program: QuablaProgram,
    tangent: TensorNodeId,
}

impl QuablaJvpProgram {
    pub fn program(&self) -> &QuablaProgram {
        &self.program
    }

    pub fn tangent_node_id(&self) -> TensorNodeId {
        self.tangent
    }

    pub fn freeze(&self) -> Result<(TensorExecutionPlan, TensorNodeId), String> {
        let (plan, outputs) = self
            .program
            .ir
            .compile_cpu_many(&[self.program.output, self.tangent])?;
        Ok((plan, outputs[1]))
    }
}

/// A reverse-mode transformed program with named gradient outputs.
#[derive(Clone, Debug)]
pub struct QuablaVjpProgram {
    program: QuablaProgram,
    cotangent: TensorNodeId,
    gradients: BTreeMap<String, TensorNodeId>,
}

impl QuablaVjpProgram {
    pub fn program(&self) -> &QuablaProgram {
        &self.program
    }

    pub fn cotangent_node_id(&self) -> TensorNodeId {
        self.cotangent
    }

    pub fn gradient_node_ids(&self) -> &BTreeMap<String, TensorNodeId> {
        &self.gradients
    }
}

/// One explicit backend compilation of a frozen program.
#[derive(Clone, Debug)]
pub enum QuablaExecutable {
    Cpu(TensorExecutionPlan),
    Cuda(CudaExecutionPlan),
    Mlx(TensorExecutionPlan),
}

impl QuablaExecutable {
    pub fn target(&self) -> QuablaTarget {
        match self {
            Self::Cpu(_) => QuablaTarget::Cpu,
            Self::Cuda(plan) => QuablaTarget::Cuda {
                device_ordinal: plan.device_ordinal(),
            },
            Self::Mlx(_) => QuablaTarget::Mlx,
        }
    }

    pub fn plan(&self) -> &TensorExecutionPlan {
        match self {
            Self::Cpu(plan) | Self::Mlx(plan) => plan,
            Self::Cuda(plan) => plan.plan(),
        }
    }

    pub fn execute(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        match self {
            Self::Cpu(plan) => plan.evaluate(inputs),
            Self::Cuda(plan) => plan.execute(inputs),
            Self::Mlx(plan) => MlxBackend.execute(plan, inputs),
        }
    }
}

/// One backend compilation of a frozen multi-output program.
///
/// The facade owns the source-to-frozen node remapping: position `i` of
/// [`Self::output_node_ids`] and of [`Self::execute`] is program output `i`.
#[derive(Clone, Debug)]
pub struct QuablaMultiOutputExecutable {
    executable: QuablaExecutable,
}

impl QuablaMultiOutputExecutable {
    pub fn target(&self) -> QuablaTarget {
        self.executable.target()
    }

    /// Frozen node ids of the program outputs, in program order.
    pub fn output_node_ids(&self) -> &[TensorNodeId] {
        self.executable.plan().output_node_ids()
    }

    /// The frozen plan, after dead-code elimination and structural CSE.
    pub fn plan(&self) -> &TensorExecutionPlan {
        self.executable.plan()
    }

    /// Executes the plan once and returns every output in program order.
    pub fn execute(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        match &self.executable {
            QuablaExecutable::Cpu(plan) => plan.evaluate_many(inputs),
            QuablaExecutable::Cuda(plan) => plan.execute_many(inputs),
            QuablaExecutable::Mlx(plan) => {
                MlxBackend.execute_many(plan, plan.output_node_ids(), inputs)
            }
        }
    }

    /// Returns the backend plan for executors that keep state across calls.
    ///
    /// Retained MLX input arrays, device-resident CUDA buffers, and Adam
    /// state live in those executors, not in the facade; they address
    /// results through [`Self::output_node_ids`].
    pub fn into_executable(self) -> QuablaExecutable {
        self.executable
    }
}

/// Stateless entrypoint for constructing, transforming, and compiling programs.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuablaCompiler;

impl QuablaCompiler {
    pub fn program(&self, ir: TensorIr, output: TensorNodeId) -> Result<QuablaProgram, String> {
        QuablaProgram::new(ir, output)
    }

    pub const fn capability(&self, target: QuablaTarget) -> QuablaCapability {
        QuablaCapability {
            target,
            built: target.is_built(),
            logical_backend: target.backend(),
        }
    }

    /// Freezes `program` and selects the backend for `target`, rejecting a
    /// target this build does not include before any lowering.
    pub fn compile(
        &self,
        program: &QuablaProgram,
        target: QuablaTarget,
    ) -> Result<QuablaExecutable, String> {
        ensure_built(target)?;
        self.compile_without_build_check(program, target)
    }

    /// Same lowering as [`Self::compile`] without the build-availability
    /// check: an unbuilt backend rejects the program itself, CUDA at compile
    /// time and MLX on first execution, with its own build instructions.
    ///
    /// Only the Python compatibility helpers use this, because their public
    /// error timing and messages predate the facade. New callers use
    /// [`Self::compile`].
    pub fn compile_without_build_check(
        &self,
        program: &QuablaProgram,
        target: QuablaTarget,
    ) -> Result<QuablaExecutable, String> {
        lower(program.freeze()?, target)
    }

    /// Multi-output counterpart of [`Self::compile`]; one lowering path serves
    /// both program kinds.
    pub fn compile_many(
        &self,
        program: &QuablaMultiOutputProgram,
        target: QuablaTarget,
    ) -> Result<QuablaMultiOutputExecutable, String> {
        ensure_built(target)?;
        self.compile_many_without_build_check(program, target)
    }

    /// Multi-output counterpart of [`Self::compile_without_build_check`], with
    /// the same restriction to Python compatibility helpers.
    pub fn compile_many_without_build_check(
        &self,
        program: &QuablaMultiOutputProgram,
        target: QuablaTarget,
    ) -> Result<QuablaMultiOutputExecutable, String> {
        lower(program.freeze()?, target)
            .map(|executable| QuablaMultiOutputExecutable { executable })
    }
}

fn ensure_built(target: QuablaTarget) -> Result<(), String> {
    if target.is_built() {
        Ok(())
    } else {
        Err(format!(
            "{} target is unavailable in this build",
            target.name()
        ))
    }
}

fn lower(plan: TensorExecutionPlan, target: QuablaTarget) -> Result<QuablaExecutable, String> {
    match target {
        QuablaTarget::Cpu => Ok(QuablaExecutable::Cpu(plan)),
        QuablaTarget::Cuda { device_ordinal } => CudaBackend::new(device_ordinal)
            .compile(plan)
            .map(QuablaExecutable::Cuda),
        QuablaTarget::Mlx => Ok(QuablaExecutable::Mlx(plan)),
    }
}
