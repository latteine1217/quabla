//! The fused CPU evaluator for elementwise subgraphs.
//!
//! The CPU evaluator runs a plan node by node: each node computes its values
//! in `f64`, rounds them to the node dtype, and stores them in a buffer of its
//! own, so a chain of elementwise nodes makes one pass over memory and at
//! least one allocation per node. Plan compilation instead groups maximal
//! elementwise subgraphs into regions ([`CpuFusionPlan`]) and compiles each
//! into a flat list of register instructions ([`FusedProgram`]), each
//! compiled in turn to a closure ([`Step`]) with its operation, rounding,
//! and operand kinds fixed. A program runs tile by tile: a step processes up
//! to [`TILE`] elements, so its dispatch is one indirect call per tile and
//! its inner loop is a plain slice loop that the compiler vectorizes. Only
//! the region's root is stored.
//!
//! Every instruction applies the rule of its node in `f64` (the kind's
//! `function()`, the arithmetic operator, the comparison, the evaluator's
//! `sqrt` and `powi`) and rounds the result to the node dtype exactly as
//! storing it does (`float32`: `as f32`; `bool`: nonzero is true), so a fused
//! region computes the same operations on the same operands in the same order
//! as node-by-node evaluation, and its values agree bit for bit. Rust never
//! contracts a product and a sum into a fused multiply-add, so keeping one
//! instruction per node keeps every rounding. The one result IEEE 754 leaves
//! open is the NaN of a sum or product of two NaNs, which the compiled
//! kernels pick by operand position after the compiler may have swapped the
//! operands: a run that meets one reports it, and the evaluator then runs
//! node by node (see [`binary_lanes`]).
//!
//! A region is grown backwards from a root, the last node of the region in
//! plan order: an operand joins when it is elementwise, has the root's shape
//! (or a single element, computed once per run), is not a retained output,
//! and every one of its users is already in the region. Operands are visited
//! in decreasing node order, so all users of a candidate are decided first.
//! Every value of a region except the root is then used only inside it, so
//! the root is the region's only output and no edge can leave the region and
//! come back, and the evaluator runs the whole region at the root's position.

use std::cell::{Cell, RefCell};
use std::collections::BinaryHeap;
use std::sync::Arc;

use super::{
    advance_broadcast_offsets, contiguous_strides, sqrt_derivative_value, tensor_op_inputs,
    BinaryMathKind, DynamicTensor, EagerOperand, HostTensorStorage, TensorComparison, TensorDType,
    TensorIr, TensorNode, TensorNodeId, TensorOp, UnaryMathKind,
};

/// Elements per instruction dispatch. 256 `f64` lanes are 2 KiB per
/// register, so a program's registers stay in the L1 cache while the per-tile
/// dispatch is a small fraction of the arithmetic.
const TILE: usize = 256;

/// How the CPU evaluator runs each node of a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FusionRole {
    /// Evaluated by itself.
    Node,
    /// Computed inside the region of a later root; its value is never stored.
    Member,
    /// The root of region `index`, which runs at this node's position.
    Root(usize),
}

/// The fused elementwise regions of a plan, found once at plan compilation.
#[derive(Clone, Debug, Default)]
pub(super) struct CpuFusionPlan {
    roles: Vec<FusionRole>,
    regions: Vec<FusedRegion>,
}

#[derive(Clone, Debug)]
struct FusedRegion {
    program: FusedProgram,
    /// The operands of every node of the region, root included, in node
    /// order: the uses the evaluator's last-use bookkeeping releases once the
    /// region has run.
    consumed: Vec<TensorNodeId>,
}

/// What the evaluator does at one node of a plan with a fusion plan.
pub(super) enum FusionStep<'a> {
    Node,
    Skip,
    Region {
        program: &'a FusedProgram,
        consumed: &'a [TensorNodeId],
    },
}

impl CpuFusionPlan {
    /// The regions of `nodes` whose interior values are not among `outputs`.
    pub(super) fn new(nodes: &[TensorNode], outputs: &[TensorNodeId]) -> Self {
        let live = live_nodes(nodes, outputs);
        let mut users = vec![Vec::new(); nodes.len()];
        for (node_id, node) in nodes.iter().enumerate() {
            if !live[node_id] {
                continue;
            }
            for input in tensor_op_inputs(&node.op) {
                // Nodes are visited in order, so a repeated operand can only
                // repeat the most recently recorded user.
                if users[input].last() != Some(&node_id) {
                    users[input].push(node_id);
                }
            }
        }
        let mut retained = vec![false; nodes.len()];
        for output in outputs {
            retained[*output] = true;
        }
        let mut roles = vec![FusionRole::Node; nodes.len()];
        let mut regions = Vec::new();
        // `member[id] == stamp` marks the members of the region being grown,
        // and `queued[id] == stamp` the operands already queued for it.
        let mut member = vec![usize::MAX; nodes.len()];
        let mut queued = vec![usize::MAX; nodes.len()];
        for root in (0..nodes.len()).rev() {
            if !live[root] || roles[root] != FusionRole::Node || !is_fusable(&nodes[root]) {
                continue;
            }
            let shape = &nodes[root].shape;
            let mut members = vec![root];
            member[root] = root;
            let mut candidates = BinaryHeap::new();
            for input in tensor_op_inputs(&nodes[root].op) {
                if queued[input] != root {
                    queued[input] = root;
                    candidates.push(input);
                }
            }
            while let Some(candidate) = candidates.pop() {
                let node = &nodes[candidate];
                let joins = roles[candidate] == FusionRole::Node
                    && !retained[candidate]
                    && users[candidate].iter().all(|user| member[*user] == root)
                    && (matches!(node.op, TensorOp::ScalarConstant { .. })
                        || (is_fusable(node)
                            && (node.shape == *shape || element_count_of(&node.shape) == 1)));
                if !joins {
                    continue;
                }
                member[candidate] = root;
                members.push(candidate);
                for input in tensor_op_inputs(&node.op) {
                    if queued[input] != root {
                        queued[input] = root;
                        candidates.push(input);
                    }
                }
            }
            members.sort_unstable();
            let computed = members
                .iter()
                .filter(|id| !matches!(nodes[**id].op, TensorOp::ScalarConstant { .. }))
                .count();
            // A lone node gains nothing from a program; it keeps its kernel.
            if computed < 2 {
                continue;
            }
            let Some(program) = FusedProgram::compile(nodes, &members) else {
                continue;
            };
            let consumed = members
                .iter()
                .flat_map(|id| tensor_op_inputs(&nodes[*id].op))
                .collect();
            for id in &members {
                roles[*id] = FusionRole::Member;
            }
            roles[root] = FusionRole::Root(regions.len());
            regions.push(FusedRegion { program, consumed });
        }
        Self { roles, regions }
    }

    pub(super) fn step(&self, node_id: TensorNodeId) -> FusionStep<'_> {
        match self.roles.get(node_id) {
            Some(FusionRole::Member) => FusionStep::Skip,
            Some(FusionRole::Root(index)) => {
                let region = &self.regions[*index];
                FusionStep::Region {
                    program: &region.program,
                    consumed: &region.consumed,
                }
            }
            _ => FusionStep::Node,
        }
    }
}

/// Whether the CPU evaluator's value of `node` is elementwise in its
/// operands, so a region can compute it per element.
fn is_fusable(node: &TensorNode) -> bool {
    match &node.op {
        TensorOp::Add { .. }
        | TensorOp::Sub { .. }
        | TensorOp::Mul { .. }
        | TensorOp::Div { .. }
        | TensorOp::Greater { .. }
        | TensorOp::Compare { .. }
        | TensorOp::Where { .. }
        | TensorOp::Sqrt { .. }
        | TensorOp::SqrtDerivative { .. }
        | TensorOp::UnaryMath { .. }
        | TensorOp::BinaryMath { .. }
        | TensorOp::Cast { .. }
        | TensorOp::StopGradient { .. }
        | TensorOp::Broadcast { .. } => true,
        // The evaluator rejects an exponent beyond `i32`; that node keeps
        // its kernel and its error.
        TensorOp::Powi { exponent, .. } => i32::try_from(*exponent).is_ok(),
        _ => false,
    }
}

fn element_count_of(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The nodes that `outputs` depend on.
fn live_nodes(nodes: &[TensorNode], outputs: &[TensorNodeId]) -> Vec<bool> {
    let mut live = vec![false; nodes.len()];
    for output in outputs {
        live[*output] = true;
    }
    for node_id in (0..nodes.len()).rev() {
        if live[node_id] {
            for input in tensor_op_inputs(&nodes[node_id].op) {
                live[input] = true;
            }
        }
    }
    live
}

/// The rounding of an `f64` result to a node dtype, as storing it rounds.
trait Rounding {
    /// Whether a NaN stays a NaN.
    const KEEPS_NAN: bool = true;

    fn round(value: f64) -> f64;
}

struct RoundF64;
struct RoundF32;
struct RoundBool;

impl Rounding for RoundF64 {
    #[inline(always)]
    fn round(value: f64) -> f64 {
        value
    }
}

impl Rounding for RoundF32 {
    /// The `float32` value, read back as `f64`. A NaN is quieted explicitly:
    /// storing quiets a signaling NaN, but the compiler may fold the widening
    /// and narrowing pair that a program forms around a selected `float32`
    /// input into nothing.
    #[inline(always)]
    fn round(value: f64) -> f64 {
        f64::from(quiet_f32(value as f32))
    }
}

impl Rounding for RoundBool {
    const KEEPS_NAN: bool = false;

    #[inline(always)]
    fn round(value: f64) -> f64 {
        flag(value != 0.0)
    }
}

/// `f64::from(value)`, written with integer operations that the compiler
/// vectorizes better than the conversion of a `bool`.
#[inline(always)]
fn flag(value: bool) -> f64 {
    f64::from_bits(u64::from(value) * 1.0f64.to_bits())
}

/// `value` with the quiet bit set if it is a NaN (branch-free, so that
/// rounding loops vectorize).
#[inline(always)]
fn quiet_f32(value: f32) -> f32 {
    f32::from_bits(value.to_bits() | (u32::from(value.is_nan()) << 22))
}

/// `value` rounded to `dtype`.
fn round(dtype: TensorDType, value: f64) -> f64 {
    match dtype {
        TensorDType::F64 => RoundF64::round(value),
        TensorDType::F32 => RoundF32::round(value),
        TensorDType::Bool => RoundBool::round(value),
    }
}

/// Evaluates `$body` with the type `$rounding` bound to the rounding of
/// `$dtype`, so that each dtype gets its own loop.
macro_rules! with_rounding {
    ($dtype:expr, $rounding:ident => $body:expr) => {
        match $dtype {
            TensorDType::F64 => {
                type $rounding = RoundF64;
                $body
            }
            TensorDType::F32 => {
                type $rounding = RoundF32;
                $body
            }
            TensorDType::Bool => {
                type $rounding = RoundBool;
                $body
            }
        }
    };
}

/// A one-operand instruction.
#[derive(Clone, Copy, Debug)]
enum Unary {
    /// The operand rounded to the instruction dtype: a narrowing `Cast`.
    Round,
    Math(UnaryMathKind),
    /// The evaluator's `sqrt` (order 0) and its derivatives.
    SqrtDerivative(u32),
    Powi(i32),
}

/// A two-operand instruction.
#[derive(Clone, Copy, Debug)]
enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    /// The legacy `gt` mask: 0/1 values of the instruction dtype.
    Greater,
    Compare(TensorComparison),
    Math(BinaryMathKind),
}

/// Where an instruction reads an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// A tile register.
    Register(usize),
    /// A contiguous `float64` input, read in place.
    Input(usize),
    /// A uniform: one value for every lane.
    Uniform(usize),
}

/// Where an instruction writes: the body writes tile registers, the
/// prologue writes uniforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Register(usize),
    Uniform(usize),
}

#[derive(Clone, Debug)]
enum Instruction {
    /// The tile of input `input` (one element for a uniform target), read as
    /// the evaluator reads stored values.
    Load { input: usize, destination: Target },
    Unary {
        op: Unary,
        source: Source,
        destination: Target,
        dtype: TensorDType,
    },
    Binary {
        op: Binary,
        lhs: Source,
        rhs: Source,
        destination: Target,
        dtype: TensorDType,
    },
    /// `where`, without a rounding: both values already have the node dtype,
    /// as every value of a program holds a value of its node's dtype.
    Select {
        condition: Source,
        on_true: Source,
        on_false: Source,
        destination: Target,
    },
}

impl Instruction {
    fn destination(&self) -> Target {
        match self {
            Self::Load { destination, .. }
            | Self::Unary { destination, .. }
            | Self::Binary { destination, .. }
            | Self::Select { destination, .. } => *destination,
        }
    }

    fn sources(&self) -> Vec<Source> {
        match self {
            Self::Load { .. } => Vec::new(),
            Self::Unary { source, .. } => vec![*source],
            Self::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Select {
                condition,
                on_true,
                on_false,
                ..
            } => vec![*condition, *on_true, *on_false],
        }
    }

    /// Replaces every tile register `r` with `physical[r]`.
    fn rename(&mut self, physical: &[usize]) {
        let source = |source: &mut Source| {
            if let Source::Register(register) = source {
                *register = physical[*register];
            }
        };
        match self {
            Self::Load { .. } => {}
            Self::Unary { source: input, .. } => source(input),
            Self::Binary { lhs, rhs, .. } => {
                source(lhs);
                source(rhs);
            }
            Self::Select {
                condition,
                on_true,
                on_false,
                ..
            } => {
                source(condition);
                source(on_true);
                source(on_false);
            }
        }
        let (Self::Load { destination, .. }
        | Self::Unary { destination, .. }
        | Self::Binary { destination, .. }
        | Self::Select { destination, .. }) = self;
        if let Target::Register(register) = destination {
            *register = physical[*register];
        }
    }
}

/// How the tiles of an input are read.
#[derive(Clone, Debug)]
enum Access {
    /// The input has one element, or the region's element count: lanes
    /// `start..start + len`.
    Contiguous,
    /// The input broadcasts to the region shape with these strides.
    Broadcast { strides: Vec<usize> },
}

#[derive(Clone, Debug)]
struct InputSlot {
    node: TensorNodeId,
    shape: Vec<usize>,
    dtype: TensorDType,
    access: Access,
}

/// One fused region compiled to steps over uniforms (one value each) and
/// tile registers, which a run keeps in one frame: the uniforms first, then
/// `TILE` lanes per register. The prologue computes the uniforms that depend
/// on inputs once per run; uniforms that depend on constants only are
/// computed at compilation, by the same steps, into `uniforms`. The body
/// runs once per tile.
#[derive(Clone, Debug)]
pub(super) struct FusedProgram {
    shape: Vec<usize>,
    /// The element count of `shape`.
    count: usize,
    dtype: TensorDType,
    inputs: Vec<InputSlot>,
    uniforms: Vec<f64>,
    registers: usize,
    output: Source,
    prologue: Vec<Step>,
    body: Vec<Step>,
}

/// Builds a program with one virtual tile register per value;
/// [`Self::finish`] maps them onto as few as their live ranges need.
struct ProgramBuilder<'a> {
    nodes: &'a [TensorNode],
    count: usize,
    inputs: Vec<InputSlot>,
    /// Each uniform's value when it is known at compilation.
    uniforms: Vec<Option<f64>>,
    registers: usize,
    prologue: Vec<Instruction>,
    body: Vec<Instruction>,
    values: Vec<Option<Source>>,
    /// Whether a known uniform summed or multiplied two NaNs.
    nan_pair: bool,
}

impl ProgramBuilder<'_> {
    fn uniform(&mut self, value: Option<f64>) -> usize {
        self.uniforms.push(value);
        self.uniforms.len() - 1
    }

    /// The value of operand `node_id`: a member's computed value, or a read
    /// of an input outside the region.
    fn operand(&mut self, node_id: TensorNodeId) -> Source {
        if let Some(value) = self.values[node_id] {
            return value;
        }
        let node = &self.nodes[node_id];
        let input = self.inputs.len();
        let elements = element_count_of(&node.shape);
        let contiguous = elements == 1 || elements == self.count;
        self.inputs.push(InputSlot {
            node: node_id,
            shape: node.shape.clone(),
            dtype: node.dtype,
            access: if contiguous {
                Access::Contiguous
            } else {
                Access::Broadcast {
                    strides: contiguous_strides(&node.shape),
                }
            },
        });
        let value = if elements == 1 {
            let slot = self.uniform(None);
            self.prologue.push(Instruction::Load {
                input,
                destination: Target::Uniform(slot),
            });
            Source::Uniform(slot)
        } else if contiguous && node.dtype == TensorDType::F64 {
            Source::Input(input)
        } else {
            self.registers += 1;
            let destination = self.registers - 1;
            self.body.push(Instruction::Load {
                input,
                destination: Target::Register(destination),
            });
            Source::Register(destination)
        };
        self.values[node_id] = Some(value);
        value
    }

    /// Emits the instruction `build` over `operands`: in the body when an
    /// operand varies per lane; otherwise as a uniform, computed now when
    /// every operand is known and in the prologue when one depends on an
    /// input.
    fn emit<const N: usize>(
        &mut self,
        operands: [Source; N],
        build: impl FnOnce(Target) -> Instruction,
    ) -> Source {
        if operands
            .iter()
            .any(|operand| !matches!(operand, Source::Uniform(_)))
        {
            self.registers += 1;
            let destination = self.registers - 1;
            self.body.push(build(Target::Register(destination)));
            return Source::Register(destination);
        }
        let known = operands.iter().all(|operand| match operand {
            Source::Uniform(slot) => self.uniforms[*slot].is_some(),
            _ => false,
        });
        let destination = self.uniform(None);
        let instruction = build(Target::Uniform(destination));
        if !known {
            self.prologue.push(instruction);
            return Source::Uniform(destination);
        }
        // Every operand is known: run the instruction now, as a run would,
        // on a frame of the known uniforms.
        let mut frame = self
            .uniforms
            .iter()
            .map(|value| value.unwrap_or(0.0))
            .collect::<Vec<_>>();
        let step = Step::compile(&instruction, &[], frame.len());
        let nan_pair = Cell::new(false);
        step.run(&Run::new(&mut frame, &[], &[], &[], &nan_pair, 0, 1));
        // Two NaN constants summed or multiplied: only the evaluator's own
        // kernel knows its NaN (see `binary_lanes`), so the region is left
        // to it.
        self.nan_pair |= nan_pair.get();
        self.uniforms[destination] = Some(frame[destination]);
        Source::Uniform(destination)
    }

    fn node(&mut self, node_id: TensorNodeId) {
        let node = &self.nodes[node_id];
        let dtype = node.dtype;
        let value = match &node.op {
            TensorOp::ScalarConstant { value } => {
                Source::Uniform(self.uniform(Some(round(dtype, *value))))
            }
            // The evaluator passes these values through unchanged.
            TensorOp::StopGradient { input } | TensorOp::Broadcast { input } => {
                self.operand(*input)
            }
            TensorOp::Cast { input } => {
                let source_dtype = self.nodes[*input].dtype;
                let operand = self.operand(*input);
                // Every `bool` and `float32` value is a value of the wider
                // dtypes, so only narrowing casts round.
                if source_dtype == dtype
                    || dtype == TensorDType::F64
                    || source_dtype == TensorDType::Bool
                {
                    operand
                } else {
                    self.unary(Unary::Round, operand, dtype)
                }
            }
            TensorOp::Sqrt { input } => {
                let operand = self.operand(*input);
                self.unary(Unary::SqrtDerivative(0), operand, dtype)
            }
            TensorOp::SqrtDerivative { input, order } => {
                let operand = self.operand(*input);
                self.unary(Unary::SqrtDerivative(*order), operand, dtype)
            }
            TensorOp::Powi { input, exponent } => {
                let operand = self.operand(*input);
                let exponent = i32::try_from(*exponent).expect("checked by is_fusable");
                self.unary(Unary::Powi(exponent), operand, dtype)
            }
            TensorOp::UnaryMath { input, kind } => {
                let operand = self.operand(*input);
                self.unary(Unary::Math(*kind), operand, dtype)
            }
            TensorOp::Add { lhs, rhs } => self.binary(Binary::Add, *lhs, *rhs, dtype),
            TensorOp::Sub { lhs, rhs } => self.binary(Binary::Sub, *lhs, *rhs, dtype),
            TensorOp::Mul { lhs, rhs } => self.binary(Binary::Mul, *lhs, *rhs, dtype),
            TensorOp::Div { lhs, rhs } => self.binary(Binary::Div, *lhs, *rhs, dtype),
            TensorOp::Greater { lhs, rhs } => self.binary(Binary::Greater, *lhs, *rhs, dtype),
            TensorOp::Compare { lhs, rhs, kind } => {
                self.binary(Binary::Compare(*kind), *lhs, *rhs, dtype)
            }
            TensorOp::BinaryMath { lhs, rhs, kind } => {
                self.binary(Binary::Math(*kind), *lhs, *rhs, dtype)
            }
            TensorOp::Where {
                condition,
                on_true,
                on_false,
            } => {
                let operands = [
                    self.operand(*condition),
                    self.operand(*on_true),
                    self.operand(*on_false),
                ];
                let [condition, on_true, on_false] = operands;
                self.emit(operands, |destination| Instruction::Select {
                    condition,
                    on_true,
                    on_false,
                    destination,
                })
            }
            op => unreachable!("{op:?} is not fusable"),
        };
        self.values[node_id] = Some(value);
    }

    fn unary(&mut self, op: Unary, source: Source, dtype: TensorDType) -> Source {
        self.emit([source], |destination| Instruction::Unary {
            op,
            source,
            destination,
            dtype,
        })
    }

    fn binary(
        &mut self,
        op: Binary,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        dtype: TensorDType,
    ) -> Source {
        let (lhs, rhs) = (self.operand(lhs), self.operand(rhs));
        self.emit([lhs, rhs], |destination| Instruction::Binary {
            op,
            lhs,
            rhs,
            destination,
            dtype,
        })
    }

    /// Maps the body's virtual tile registers onto physical ones: a register
    /// is free again after the last instruction that reads it, but only once
    /// that instruction's destination is chosen, so no instruction writes a
    /// register it reads. Then compiles every instruction to its step.
    fn finish(mut self, output: Source, root: &TensorNode) -> FusedProgram {
        let mut last_use = vec![None; self.registers];
        for (position, instruction) in self.body.iter().enumerate() {
            for source in instruction.sources() {
                if let Source::Register(register) = source {
                    last_use[register] = Some(position);
                }
            }
        }
        if let Source::Register(register) = output {
            last_use[register] = Some(self.body.len());
        }
        let mut physical = vec![usize::MAX; self.registers];
        let mut count = 0;
        let mut free = Vec::new();
        for position in 0..self.body.len() {
            if let Target::Register(destination) = self.body[position].destination() {
                physical[destination] = free.pop().unwrap_or_else(|| {
                    count += 1;
                    count - 1
                });
            }
            let sources = self.body[position].sources();
            for (index, source) in sources.iter().enumerate() {
                if let Source::Register(register) = *source {
                    // A register read twice by one instruction is freed once.
                    if last_use[register] == Some(position) && !sources[..index].contains(source) {
                        free.push(physical[register]);
                    }
                }
            }
        }
        for instruction in &mut self.body {
            instruction.rename(&physical);
        }
        let output = match output {
            Source::Register(register) => Source::Register(physical[register]),
            source => source,
        };
        let uniforms = self.uniforms.len();
        let compile = |instructions: &[Instruction]| {
            instructions
                .iter()
                .map(|instruction| Step::compile(instruction, &self.inputs, uniforms))
                .collect()
        };
        FusedProgram {
            shape: root.shape.clone(),
            count: self.count,
            dtype: root.dtype,
            uniforms: self
                .uniforms
                .iter()
                .map(|value| value.unwrap_or(0.0))
                .collect(),
            registers: count,
            prologue: compile(&self.prologue),
            body: compile(&self.body),
            inputs: self.inputs,
            output,
        }
    }
}

impl FusedProgram {
    /// Compiles the region of `members` (ascending node ids, the last one the
    /// root). `None` when the root's value would be an input passed through
    /// unchanged (the evaluator keeps such a value's bits as stored, while a
    /// program rounds what it stores), or when constants sum or multiply two
    /// NaNs (see [`binary_lanes`]).
    fn compile(nodes: &[TensorNode], members: &[TensorNodeId]) -> Option<Self> {
        let root = *members.last()?;
        let mut builder = ProgramBuilder {
            nodes,
            count: element_count_of(&nodes[root].shape),
            inputs: Vec::new(),
            uniforms: Vec::new(),
            registers: 0,
            prologue: Vec::new(),
            body: Vec::new(),
            values: vec![None; root + 1],
            nan_pair: false,
        };
        for member in members {
            builder.node(*member);
        }
        if builder.nan_pair {
            return None;
        }
        let output = builder.values[root]?;
        let loaded = |instructions: &[Instruction], target: Target| {
            instructions.iter().any(|instruction| {
                instruction.destination() == target
                    && matches!(instruction, Instruction::Load { .. })
            })
        };
        let input = match output {
            Source::Input(_) => true,
            Source::Register(register) => loaded(&builder.body, Target::Register(register)),
            Source::Uniform(slot) => loaded(&builder.prologue, Target::Uniform(slot)),
        };
        if input && passes_through(nodes, root) {
            return None;
        }
        Some(builder.finish(output, &nodes[root]))
    }

    /// The region's value over its inputs, read from the evaluator's values,
    /// or `None` when a sum or product met two NaN operands (see
    /// [`binary_lanes`]); the caller then evaluates node by node.
    pub(super) fn evaluate(
        &self,
        values: &[Option<DynamicTensor>],
    ) -> Result<Option<DynamicTensor>, String> {
        for slot in &self.inputs {
            if values.get(slot.node).and_then(Option::as_ref).is_none() {
                return Err(format!("node {} has no evaluated value", slot.node));
            }
        }
        let value = self.run(&|input| {
            let value = values[self.inputs[input].node]
                .as_ref()
                .expect("checked above");
            (value.shape.as_slice(), &value.data)
        })?;
        #[cfg(test)]
        if value.is_none() {
            tests::FALLBACKS.with(|count| count.set(count.get() + 1));
        }
        Ok(value)
    }

    fn run<'a>(&self, input: &dyn Fn(usize) -> Input<'a>) -> Result<Option<DynamicTensor>, String> {
        // A few inputs are resolved on the stack; a large region allocates.
        let mut inline = [None; 4];
        let mut heap = Vec::new();
        let resolved = if self.inputs.len() <= inline.len() {
            &mut inline[..self.inputs.len()]
        } else {
            heap.resize(self.inputs.len(), None);
            &mut heap[..]
        };
        for (index, (slot, resolved)) in self.inputs.iter().zip(resolved.iter_mut()).enumerate() {
            let (shape, storage) = input(index);
            if slot.shape != shape || slot.dtype != storage.dtype() {
                return Err(format!(
                    "fused region input node {} has shape {shape:?} and dtype {}, expected \
                     {:?} and {}",
                    slot.node,
                    storage.dtype(),
                    slot.shape,
                    slot.dtype
                ));
            }
            *resolved = Some((shape, storage));
        }
        let inputs = &*resolved;
        let count = self.count;
        // The read position of each broadcast input, kept across tiles.
        let cursors = if self
            .inputs
            .iter()
            .any(|slot| matches!(slot.access, Access::Broadcast { .. }))
        {
            vec![Cell::new(0); self.inputs.len()]
        } else {
            Vec::new()
        };
        let uniforms = self.uniforms.len();
        let nan_pair = Cell::new(false);
        SCRATCH.with(|scratch| {
            // A nested run (none exists today) would find the scratch taken
            // and allocate its own. Stale lanes need no clearing: every
            // instruction writes its destination before a later one reads it.
            let mut frame = scratch.take();
            let len = uniforms + self.registers * TILE;
            if frame.len() < len {
                frame.resize(len, 0.0);
            }
            frame[..uniforms].copy_from_slice(&self.uniforms);
            let mut output = Output::new(self.dtype, count);
            let prologue = Run::new(&mut frame, inputs, &cursors, &self.shape, &nan_pair, 0, 1);
            for step in &self.prologue {
                step.run(&prologue);
            }
            match self.output {
                _ if nan_pair.get() => {}
                Source::Uniform(slot) => output.fill(frame[slot], count),
                source => {
                    let source = Arg::new(source, uniforms);
                    let mut start = 0;
                    while start < count && !nan_pair.get() {
                        let len = TILE.min(count - start);
                        let tile = Run::new(
                            &mut frame,
                            inputs,
                            &cursors,
                            &self.shape,
                            &nan_pair,
                            start,
                            len,
                        );
                        for step in &self.body {
                            step.run(&tile);
                        }
                        output.extend(tile.tile(source));
                        start += len;
                    }
                }
            }
            scratch.replace(frame);
            Ok((!nan_pair.get()).then(|| DynamicTensor {
                shape: self.shape.clone(),
                data: output.finish(),
                dtype: self.dtype,
            }))
        })
    }
}

/// The shape and storage of a program input.
type Input<'a> = (&'a [usize], &'a HostTensorStorage);

/// Whether `node_id` passes the value of its operand through unchanged in
/// the evaluator: a `stop_gradient`, or a cast to the same dtype.
fn passes_through(nodes: &[TensorNode], node_id: TensorNodeId) -> bool {
    match &nodes[node_id].op {
        TensorOp::StopGradient { .. } => true,
        TensorOp::Cast { input } => nodes[*input].dtype == nodes[node_id].dtype,
        _ => false,
    }
}

thread_local! {
    /// The frame of the running program, kept between runs so that a small
    /// evaluation allocates only its output.
    static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// An operand as a step reads it: frame lanes at an offset, an input read in
/// place, or a uniform at a frame offset.
#[derive(Clone, Copy, Debug)]
enum Arg {
    Frame(usize),
    Input(usize),
    Uniform(usize),
}

impl Arg {
    /// `source` in a frame whose first `uniforms` entries are the uniforms.
    fn new(source: Source, uniforms: usize) -> Self {
        match source {
            Source::Register(register) => Self::Frame(uniforms + register * TILE),
            Source::Input(input) => Self::Input(input),
            Source::Uniform(slot) => Self::Uniform(slot),
        }
    }
}

/// One tile (or, for the prologue, one element) of a running program.
struct Run<'r> {
    frame: *mut f64,
    frame_len: usize,
    inputs: &'r [Option<Input<'r>>],
    cursors: &'r [Cell<usize>],
    shape: &'r [usize],
    /// Set when a sum or product met two NaN operands.
    nan_pair: &'r Cell<bool>,
    start: usize,
    len: usize,
}

impl<'r> Run<'r> {
    fn new(
        frame: &'r mut [f64],
        inputs: &'r [Option<Input<'r>>],
        cursors: &'r [Cell<usize>],
        shape: &'r [usize],
        nan_pair: &'r Cell<bool>,
        start: usize,
        len: usize,
    ) -> Self {
        Self {
            frame: frame.as_mut_ptr(),
            frame_len: frame.len(),
            inputs,
            cursors,
            shape,
            nan_pair,
            start,
            len,
        }
    }

    fn storage(&self, input: usize) -> &'r HostTensorStorage {
        self.inputs[input]
            .expect("inputs are resolved before a run")
            .1
    }

    /// The lanes of a tile operand.
    #[inline(always)]
    fn tile(&self, arg: Arg) -> &'r [f64] {
        match arg {
            Arg::Frame(offset) => {
                assert!(offset + self.len <= self.frame_len);
                // SAFETY: the lanes lie inside the frame (asserted above),
                // which the run borrows for `'r`. Register allocation never
                // makes a step's destination one of its sources, so these
                // lanes do not overlap the lanes a step writes.
                unsafe { std::slice::from_raw_parts(self.frame.add(offset), self.len) }
            }
            Arg::Input(input) => match self.storage(input) {
                HostTensorStorage::F64(values) => &values[self.start..self.start + self.len],
                _ => unreachable!("only float64 inputs are read in place"),
            },
            Arg::Uniform(_) => unreachable!("a uniform is not a tile"),
        }
    }

    #[inline(always)]
    fn uniform(&self, offset: usize) -> f64 {
        assert!(offset < self.frame_len);
        // SAFETY: in bounds (asserted above); read before the step writes.
        unsafe { *self.frame.add(offset) }
    }

    /// The lanes a step writes, at frame offset `offset`.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    fn target(&self, offset: usize) -> &'r mut [f64] {
        assert!(offset + self.len <= self.frame_len);
        // SAFETY: inside the frame (asserted above), which the run borrows
        // mutably for `'r`. A step takes its destination once and its
        // sources are other registers, inputs, or uniforms read by value, so
        // nothing else refers to these lanes while the step writes them.
        unsafe { std::slice::from_raw_parts_mut(self.frame.add(offset), self.len) }
    }
}

/// An operand kind fixed when a step is compiled, so that the step's loop is
/// specialized for it.
trait Fetch: Copy + Send + Sync + 'static {
    type Lanes<'r>: Lanes;
    fn fetch<'r>(self, run: &Run<'r>) -> Self::Lanes<'r>;
}

#[derive(Clone, Copy)]
struct TileAt(Arg);

#[derive(Clone, Copy)]
struct UniformAt(usize);

impl Fetch for TileAt {
    type Lanes<'r> = &'r [f64];

    #[inline(always)]
    fn fetch<'r>(self, run: &Run<'r>) -> &'r [f64] {
        run.tile(self.0)
    }
}

impl Fetch for UniformAt {
    type Lanes<'r> = f64;

    #[inline(always)]
    fn fetch<'r>(self, run: &Run<'r>) -> f64 {
        run.uniform(self.0)
    }
}

/// Binds `$name` to the fetch of `$arg` (a tile or a uniform) and evaluates
/// `$body` once per kind.
macro_rules! with_fetch {
    ($arg:expr, $name:ident => $body:expr) => {
        match $arg {
            Arg::Uniform(offset) => {
                let $name = UniformAt(offset);
                $body
            }
            tile => {
                let $name = TileAt(tile);
                $body
            }
        }
    };
}

/// An instruction compiled to a closure over its operands' frame offsets,
/// with its operation, rounding, and operand kinds fixed: running it is one
/// indirect call and one loop over the lanes.
#[derive(Clone)]
struct Step(Arc<dyn Fn(&Run<'_>) + Send + Sync>);

impl std::fmt::Debug for Step {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Step")
    }
}

impl Step {
    fn new(step: impl Fn(&Run<'_>) + Send + Sync + 'static) -> Self {
        Self(Arc::new(step))
    }

    #[inline(always)]
    fn run(&self, run: &Run<'_>) {
        (self.0)(run)
    }

    /// The step of `instruction` in a frame with `uniforms` uniforms.
    fn compile(instruction: &Instruction, inputs: &[InputSlot], uniforms: usize) -> Self {
        let target = |destination: Target| match destination {
            Target::Register(register) => uniforms + register * TILE,
            Target::Uniform(slot) => slot,
        };
        let arg = |source: Source| Arg::new(source, uniforms);
        match instruction {
            Instruction::Load { input, destination } => {
                let input = *input;
                match (destination, &inputs[input].access) {
                    (Target::Uniform(slot), _) => {
                        let slot = *slot;
                        Self::new(move |run| run.target(slot)[0] = run.storage(input).get(0))
                    }
                    (Target::Register(_), Access::Contiguous) => {
                        let target = target(*destination);
                        Self::new(move |run| {
                            load_contiguous(run.storage(input), run.start, run.target(target))
                        })
                    }
                    (Target::Register(_), Access::Broadcast { strides }) => {
                        let target = target(*destination);
                        let (shape, strides) = (inputs[input].shape.clone(), strides.clone());
                        Self::new(move |run| {
                            load_broadcast(
                                run.storage(input),
                                run.shape,
                                &shape,
                                &strides,
                                run.start,
                                &run.cursors[input],
                                run.target(target),
                            )
                        })
                    }
                }
            }
            Instruction::Unary {
                op,
                source,
                destination,
                dtype,
            } => {
                let (source, target) = (arg(*source), target(*destination));
                with_rounding!(*dtype, R => match *op {
                    Unary::Round => unary_step::<R>(source, target, |x| x),
                    Unary::Math(kind) => unary_step::<R>(source, target, kind.function()),
                    Unary::SqrtDerivative(order) => {
                        unary_step::<R>(source, target, move |x| sqrt_derivative_value(x, order))
                    }
                    Unary::Powi(exponent) => {
                        unary_step::<R>(source, target, move |x| x.powi(exponent))
                    }
                })
            }
            Instruction::Binary {
                op,
                lhs,
                rhs,
                destination,
                dtype,
            } => {
                let (lhs, rhs, target) = (arg(*lhs), arg(*rhs), target(*destination));
                // A comparison gives 0 or 1, which every dtype holds exactly,
                // so it skips the rounding. One step per comparison, a
                // closure each, so that no lane branches on the kind.
                macro_rules! compare {
                    ($kind:ident) => {
                        binary_step::<RoundF64>(
                            lhs,
                            rhs,
                            target,
                            |x, y| flag(TensorComparison::$kind.evaluate(x, y)),
                            false,
                        )
                    };
                }
                match *op {
                    Binary::Greater => {
                        return binary_step::<RoundF64>(lhs, rhs, target, |x, y| flag(x > y), false)
                    }
                    Binary::Compare(kind) => {
                        return match kind {
                            TensorComparison::Greater => compare!(Greater),
                            TensorComparison::GreaterEqual => compare!(GreaterEqual),
                            TensorComparison::Less => compare!(Less),
                            TensorComparison::LessEqual => compare!(LessEqual),
                            TensorComparison::Equal => compare!(Equal),
                            TensorComparison::NotEqual => compare!(NotEqual),
                        }
                    }
                    _ => {}
                }
                with_rounding!(*dtype, R => match *op {
                    Binary::Add => binary_step::<R>(lhs, rhs, target, |x, y| x + y, true),
                    Binary::Sub => binary_step::<R>(lhs, rhs, target, |x, y| x - y, false),
                    Binary::Mul => binary_step::<R>(lhs, rhs, target, |x, y| x * y, true),
                    Binary::Div => binary_step::<R>(lhs, rhs, target, |x, y| x / y, false),
                    Binary::Math(kind) => {
                        binary_step::<R>(lhs, rhs, target, kind.function(), false)
                    }
                    Binary::Greater | Binary::Compare(_) => unreachable!("compiled above"),
                })
            }
            Instruction::Select {
                condition,
                on_true,
                on_false,
                destination,
            } => {
                let target = target(*destination);
                with_fetch!(arg(*condition), condition => {
                    with_fetch!(arg(*on_true), on_true => {
                        with_fetch!(arg(*on_false), on_false => Self::new(move |run| {
                            select_lanes(
                                condition.fetch(run),
                                on_true.fetch(run),
                                on_false.fetch(run),
                                run.target(target),
                            )
                        }))
                    })
                })
            }
        }
    }
}

fn unary_step<R: Rounding + 'static>(
    source: Arg,
    target: usize,
    function: impl Fn(f64) -> f64 + Copy + Send + Sync + 'static,
) -> Step {
    with_fetch!(source, x => Step::new(move |run| {
        unary_lanes::<R>(x.fetch(run), run.target(target), function)
    }))
}

/// A binary step; a commutative one sets the run's `nan_pair` flag when a
/// lane has two NaN operands (see [`binary_lanes`]).
fn binary_step<R: Rounding + 'static>(
    lhs: Arg,
    rhs: Arg,
    target: usize,
    function: impl Fn(f64, f64) -> f64 + Copy + Send + Sync + 'static,
    commutative: bool,
) -> Step {
    with_fetch!(lhs, x => with_fetch!(rhs, y => Step::new(move |run| {
        let (x, y) = (x.fetch(run), y.fetch(run));
        if binary_lanes::<R>(x, y, run.target(target), function, commutative) {
            run.nan_pair.set(true);
        }
    })))
}

/// The lanes of a tile operand: a slice, or one value for every lane.
trait Lanes: Copy {
    fn lane(self, index: usize) -> f64;

    /// Panics unless a slice has exactly `len` lanes, which lets the loops
    /// below index it without bounds checks and vectorize.
    fn check(self, len: usize);
}

impl Lanes for &[f64] {
    #[inline(always)]
    fn lane(self, index: usize) -> f64 {
        self[index]
    }

    #[inline(always)]
    fn check(self, len: usize) {
        assert_eq!(self.len(), len);
    }
}

impl Lanes for f64 {
    #[inline(always)]
    fn lane(self, _: usize) -> f64 {
        self
    }

    #[inline(always)]
    fn check(self, _: usize) {}
}

#[inline(always)]
fn unary_lanes<R: Rounding>(x: impl Lanes, target: &mut [f64], function: impl Fn(f64) -> f64) {
    x.check(target.len());
    for (index, lane) in target.iter_mut().enumerate() {
        *lane = R::round(function(x.lane(index)));
    }
}

/// `function` lane by lane. A commutative operation (`commutative`) also
/// reports, through the return value, whether some lane had two NaN
/// operands. IEEE 754 leaves the NaN of such a sum or product to the
/// hardware, which picks one operand's NaN by its position in the
/// instruction, and the compiler may swap the operands of a commutative
/// operation, differently in different loops: the evaluator's own kernels
/// do (its loop over a scalar plus an array returns the array's NaN on
/// AArch64), and so does this loop for a tile times a uniform. Which NaN the
/// evaluator returns is therefore a property of its compiled kernels, which
/// the caller reproduces by evaluating node by node. With at most one NaN
/// operand the result is that NaN, quieted, in either order.
#[inline(always)]
fn binary_lanes<R: Rounding>(
    x: impl Lanes,
    y: impl Lanes,
    target: &mut [f64],
    function: impl Fn(f64, f64) -> f64,
    commutative: bool,
) -> bool {
    x.check(target.len());
    y.check(target.len());
    for (index, lane) in target.iter_mut().enumerate() {
        *lane = R::round(function(x.lane(index), y.lane(index)));
    }
    if !commutative {
        return false;
    }
    // Two NaN operands give a NaN result (unless rounded to `bool`), which
    // is rare: one branch-free pass over the result rules them out.
    let nan = |any: u8, value: &f64| any | u8::from(value.is_nan());
    if R::KEEPS_NAN && target.iter().fold(0, nan) == 0 {
        return false;
    }
    (0..target.len()).fold(false, |any, index| {
        any | (x.lane(index).is_nan() & y.lane(index).is_nan())
    })
}

/// `where`: a nonzero condition, NaN included, selects `on_true`. The values
/// already hold values of the node dtype, which its rounding would keep.
#[inline(always)]
fn select_lanes(
    condition: impl Lanes,
    on_true: impl Lanes,
    on_false: impl Lanes,
    target: &mut [f64],
) {
    condition.check(target.len());
    on_true.check(target.len());
    on_false.check(target.len());
    for (index, lane) in target.iter_mut().enumerate() {
        // A bitwise select, which vectorizes better than a branch and moves
        // the selected bits unchanged, as the branch would.
        let mask = 0u64.wrapping_sub(u64::from(condition.lane(index) != 0.0));
        let bits =
            (on_true.lane(index).to_bits() & mask) | (on_false.lane(index).to_bits() & !mask);
        *lane = f64::from_bits(bits);
    }
}

fn load_contiguous(storage: &HostTensorStorage, start: usize, target: &mut [f64]) {
    let end = start + target.len();
    match storage {
        HostTensorStorage::F64(values) => target.copy_from_slice(&values[start..end]),
        HostTensorStorage::F32(values) => {
            for (lane, value) in target.iter_mut().zip(&values[start..end]) {
                *lane = f64::from(*value);
            }
        }
        HostTensorStorage::Bool(values) => {
            for (lane, value) in target.iter_mut().zip(&values[start..end]) {
                *lane = f64::from(*value != 0);
            }
        }
    }
}

/// Lanes `start..` of an input broadcast to `shape`, read at the offsets the
/// evaluator's broadcasting kernels visit; `offset` carries the position from
/// one tile to the next.
fn load_broadcast(
    storage: &HostTensorStorage,
    shape: &[usize],
    input_shape: &[usize],
    strides: &[usize],
    start: usize,
    offset: &Cell<usize>,
    target: &mut [f64],
) {
    let count = element_count_of(shape);
    let mut offsets = [offset.get()];
    for (lane, index) in target.iter_mut().zip(start..) {
        *lane = storage.get(offsets[0]);
        if index + 1 < count {
            advance_broadcast_offsets(index + 1, shape, [(input_shape, strides)], &mut offsets);
        }
    }
    offset.set(offsets[0]);
}

/// The region's output storage, filled tile by tile.
enum Output {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bool(Vec<u8>),
}

impl Output {
    fn new(dtype: TensorDType, count: usize) -> Self {
        match dtype {
            TensorDType::F64 => Self::F64(Vec::with_capacity(count)),
            TensorDType::F32 => Self::F32(Vec::with_capacity(count)),
            TensorDType::Bool => Self::Bool(Vec::with_capacity(count)),
        }
    }

    fn extend(&mut self, tile: &[f64]) {
        match self {
            Self::F64(values) => values.extend_from_slice(tile),
            Self::F32(values) => values.extend(tile.iter().map(|value| quiet_f32(*value as f32))),
            Self::Bool(values) => values.extend(tile.iter().map(|value| u8::from(*value != 0.0))),
        }
    }

    /// `count` copies of a uniform value.
    fn fill(&mut self, value: f64, count: usize) {
        match self {
            Self::F64(values) => values.resize(count, value),
            Self::F32(values) => values.resize(count, quiet_f32(value as f32)),
            Self::Bool(values) => values.resize(count, u8::from(value != 0.0)),
        }
    }

    fn finish(self) -> HostTensorStorage {
        match self {
            Self::F64(values) => HostTensorStorage::F64(Arc::new(values)),
            Self::F32(values) => HostTensorStorage::F32(Arc::new(values)),
            Self::Bool(values) => HostTensorStorage::Bool(Arc::new(values)),
        }
    }
}

/// A whole graph compiled to one fused region over operand nodes, for eager
/// ops built from several primitives: [`TensorIr::fused_kernel`] compiles
/// the graph of one call, and later calls with operands of the same shapes
/// and dtypes run it without building a graph.
#[derive(Clone, Debug)]
pub struct FusedKernel {
    program: FusedProgram,
    /// The operand index of each program input.
    operands: Vec<usize>,
}

impl FusedKernel {
    /// The kernel's value over `operands`, which must be arrays with the
    /// shapes and dtypes of the operand nodes the kernel was compiled with;
    /// it equals the CPU evaluator's value of the graph bit for bit. `None`
    /// when a sum or product met two NaN operands, whose NaN only the
    /// evaluator's kernels decide: the caller then evaluates the graph.
    pub fn evaluate(&self, operands: &[EagerOperand<'_>]) -> Result<Option<DynamicTensor>, String> {
        for operand in &self.operands {
            if !matches!(operands.get(*operand), Some(EagerOperand::Array { .. })) {
                return Err(format!("fused kernel operand {operand} is not an array"));
            }
        }
        self.program
            .run(&|input| match operands[self.operands[input]] {
                EagerOperand::Array { shape, storage } => (shape, storage),
                EagerOperand::Scalar(..) => unreachable!("checked above"),
            })
    }
}

impl TensorIr {
    /// The graph of `output` as one fused kernel over the `operands` nodes
    /// (usually constants holding an eager call's arguments), or `None` when
    /// the evaluator would not run it as a single fused region: a node that
    /// is not elementwise, a value used outside the region, or a lone node.
    pub fn fused_kernel(
        &self,
        output: TensorNodeId,
        operands: &[TensorNodeId],
    ) -> Option<FusedKernel> {
        self.nodes.get(output)?;
        let plan = CpuFusionPlan::new(&self.nodes, &[output]);
        let FusionStep::Region { program, .. } = plan.step(output) else {
            return None;
        };
        // Every live node is in the region (scalar constants included) or
        // is an operand.
        let live = live_nodes(&self.nodes, &[output]);
        let complete = live.iter().enumerate().all(|(node_id, live)| {
            !live || plan.roles[node_id] != FusionRole::Node || operands.contains(&node_id)
        });
        if !complete {
            return None;
        }
        let operands = program
            .inputs
            .iter()
            .map(|slot| operands.iter().position(|operand| *operand == slot.node))
            .collect::<Option<Vec<_>>>()?;
        Some(FusedKernel {
            program: program.clone(),
            operands,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::CPU_FORWARD_PEAK_ELEMENTS;
    use super::*;
    use std::collections::BTreeMap;

    /// Special values: signed zeros, subnormals, huge and tiny magnitudes,
    /// infinities, NaNs of both signs with payloads, and domain edges.
    const SPECIAL: [f64; 24] = [
        0.0,
        -0.0,
        5e-324,
        -1e-310,
        1e-45,
        0.5,
        -1.5,
        2.5,
        1.0,
        -1.0,
        0.9999999,
        std::f64::consts::FRAC_PI_2,
        88.7,
        -745.5,
        710.0,
        1e30,
        -1e300,
        3.5e38,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8_0000_0000_1234),
        f64::from_bits(0xfff0_0000_0000_0042),
    ];

    fn values(count: usize, offset: usize) -> Vec<f64> {
        (0..count)
            .map(|index| SPECIAL[(index * 7 + offset) % SPECIAL.len()])
            .collect()
    }

    fn tensor(shape: &[usize], dtype: TensorDType, offset: usize) -> DynamicTensor {
        let count = shape.iter().product();
        DynamicTensor::with_dtype(shape.to_vec(), values(count, offset), dtype).unwrap()
    }

    /// Every pair of special values, both orders, so arithmetic meets two
    /// NaN operands with different payloads and signs.
    fn pairs(dtype: TensorDType) -> (DynamicTensor, DynamicTensor) {
        let (lhs, rhs): (Vec<f64>, Vec<f64>) = SPECIAL
            .iter()
            .flat_map(|lhs| SPECIAL.iter().map(move |rhs| (*lhs, *rhs)))
            .unzip();
        let count = lhs.len();
        (
            DynamicTensor::with_dtype(vec![count], lhs, dtype).unwrap(),
            DynamicTensor::with_dtype(vec![count], rhs, dtype).unwrap(),
        )
    }

    fn bits(tensor: &DynamicTensor) -> Vec<u64> {
        match &tensor.data {
            HostTensorStorage::F32(values) => values
                .iter()
                .map(|value| u64::from(value.to_bits()))
                .collect(),
            data => data.iter().map(f64::to_bits).collect(),
        }
    }

    /// Compiles `outputs` and checks that the fused plan returns the values,
    /// shapes, and dtypes of node-by-node evaluation of the same plan bit for
    /// bit. Returns the number of fused regions.
    fn check(
        ir: &TensorIr,
        outputs: &[TensorNodeId],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> usize {
        let (plan, _) = ir.compile_cpu_many(outputs).unwrap();
        let fused = plan.evaluate_many(inputs).unwrap();
        let reference = TensorIr::evaluate_tensor_nodes_with_outputs(
            &plan.nodes,
            inputs,
            Some(&plan.output_node_ids),
        )
        .unwrap();
        for (output, actual) in plan.output_node_ids.iter().zip(&fused) {
            let expected = reference[*output].as_ref().unwrap();
            assert_eq!(actual.shape, expected.shape, "{}", plan.lower_text());
            assert_eq!(actual.dtype, expected.dtype, "{}", plan.lower_text());
            assert_eq!(bits(actual), bits(expected), "{}", plan.lower_text());
        }
        assert_eq!(bits(&plan.evaluate(inputs).unwrap()), bits(&fused[0]));
        plan.cpu_fusion.regions.len()
    }

    thread_local! {
        /// The fused runs that fell back to node by node on this thread.
        pub(super) static FALLBACKS: Cell<usize> = const { Cell::new(0) };
    }

    /// [`check`], requiring that every fused run kept its program.
    fn check_fused(
        ir: &TensorIr,
        outputs: &[TensorNodeId],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> usize {
        let before = FALLBACKS.with(Cell::get);
        let regions = check(ir, outputs, inputs);
        assert_eq!(
            FALLBACKS.with(Cell::get),
            before,
            "fell back: {}",
            ir.lower_text()
        );
        regions
    }

    /// The pairs of special values that are not NaN, so that no sum or
    /// product of a chain meets two NaN operands unless the chain makes them.
    fn ordered_pairs(dtype: TensorDType) -> (DynamicTensor, DynamicTensor) {
        let finite = SPECIAL
            .iter()
            .copied()
            .filter(|value| !value.is_nan())
            .collect::<Vec<_>>();
        let (lhs, rhs): (Vec<f64>, Vec<f64>) = finite
            .iter()
            .flat_map(|lhs| finite.iter().map(move |rhs| (*lhs, *rhs)))
            .unzip();
        let count = lhs.len();
        (
            DynamicTensor::with_dtype(vec![count], lhs, dtype).unwrap(),
            DynamicTensor::with_dtype(vec![count], rhs, dtype).unwrap(),
        )
    }

    type Build = fn(&mut TensorIr, TensorNodeId, TensorNodeId) -> Result<TensorNodeId, String>;

    /// Chains that end in every elementwise op, each fed by computed values
    /// so that the op runs inside a region.
    fn chains() -> Vec<(&'static str, Build)> {
        vec![
            ("add", |ir, x, y| {
                let (a, b) = (ir.mul(x, y)?, ir.sub(y, x)?);
                ir.add(a, b)
            }),
            ("div", |ir, x, y| {
                let (a, b) = (ir.add(x, y)?, ir.mul(y, x)?);
                ir.div(a, b)
            }),
            ("greater", |ir, x, y| {
                let a = ir.sub(x, y)?;
                let mask = ir.greater(a, y)?;
                ir.mul(mask, x)
            }),
            ("compare", |ir, x, y| {
                let a = ir.div(x, y)?;
                let mut mask = ir.compare(a, x, TensorComparison::Less)?;
                for kind in [
                    TensorComparison::Greater,
                    TensorComparison::GreaterEqual,
                    TensorComparison::LessEqual,
                    TensorComparison::Equal,
                    TensorComparison::NotEqual,
                ] {
                    let next = ir.compare(a, y, kind)?;
                    mask = ir.logical_or(mask, next)?;
                }
                let zero = ir.scalar_constant(-0.0);
                ir.where_select(mask, a, zero)
            }),
            ("where", |ir, x, y| {
                let (a, b) = (ir.add(x, y)?, ir.sub(x, y)?);
                let condition = ir.mul(x, y)?;
                ir.where_select(condition, a, b)
            }),
            ("where_inputs", |ir, x, y| {
                // Selected inputs reach the output unchanged but rounded.
                let condition = ir.greater(x, y)?;
                let selected = ir.where_select(condition, x, y)?;
                ir.stop_gradient(selected)
            }),
            ("sqrt", |ir, x, y| {
                let a = ir.mul(x, y)?;
                let root = ir.sqrt(a)?;
                let first = ir.sqrt_derivative(a, 1)?;
                let second = ir.sqrt_derivative(x, 2)?;
                let sum = ir.add(root, first)?;
                ir.add(sum, second)
            }),
            ("powi", |ir, x, y| {
                let a = ir.add(x, y)?;
                let cube = ir.powi(a, 3)?;
                let one = ir.powi(a, 0)?;
                ir.sub(cube, one)
            }),
            ("cast", |ir, x, y| {
                let a = ir.mul(x, y)?;
                let wide = ir.cast(a, TensorDType::F64)?;
                let narrow = ir.cast(wide, TensorDType::F32)?;
                let mask = ir.cast(narrow, TensorDType::Bool)?;
                let back = ir.cast(mask, TensorDType::F64)?;
                let sum = ir.add(wide, back)?;
                ir.cast(sum, TensorDType::F32)
            }),
            ("scalars", |ir, x, y| {
                let (half, big) = (ir.scalar_constant(0.5), ir.scalar_constant(1e300));
                let a = ir.mul(x, half)?;
                let b = ir.sub(big, y)?;
                let c = ir.div(a, b)?;
                let tiny = ir.scalar_constant(1e-320);
                ir.add(c, tiny)
            }),
            ("dag", |ir, x, y| {
                // A shared value with several users inside one region.
                let e = ir.exp(x)?;
                let square = ir.mul(e, e)?;
                let ratio = ir.div(e, y)?;
                let sum = ir.add(square, ratio)?;
                ir.sub(sum, e)
            }),
        ]
    }

    #[test]
    fn fused_chains_match_node_by_node_bit_for_bit() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let (lhs, rhs) = pairs(dtype);
            let shape = lhs.shape.clone();
            let inputs = BTreeMap::from([("x".to_string(), lhs), ("y".to_string(), rhs)]);
            let typed = |ir: &mut TensorIr| {
                (
                    ir.input_typed("x", shape.clone(), dtype).unwrap(),
                    ir.input_typed("y", shape.clone(), dtype).unwrap(),
                )
            };
            let (lhs, rhs) = ordered_pairs(dtype);
            let ordered_shape = lhs.shape.clone();
            let ordered = BTreeMap::from([("x".to_string(), lhs), ("y".to_string(), rhs)]);
            for (name, build) in chains() {
                if name == "cast" && dtype == TensorDType::F64 {
                    continue;
                }
                let mut ir = TensorIr::new();
                let (x, y) = typed(&mut ir);
                let output = build(&mut ir, x, y).unwrap();
                assert!(check(&ir, &[output], &inputs) > 0, "{name} {dtype}");
                // Without NaN inputs the fused programs run to the end, over
                // signed zeros, subnormals, infinities, and the NaNs that
                // the chains make, except that `sqrt` of a negative and its
                // derivative are two NaNs that the chain adds.
                let mut ir = TensorIr::new();
                let x = ir.input_typed("x", ordered_shape.clone(), dtype).unwrap();
                let y = ir.input_typed("y", ordered_shape.clone(), dtype).unwrap();
                let output = build(&mut ir, x, y).unwrap();
                if name == "sqrt" {
                    check(&ir, &[output], &ordered);
                } else {
                    assert!(check_fused(&ir, &[output], &ordered) > 0, "{name} {dtype}");
                }
            }
            for kind in UnaryMathKind::ALL {
                let mut ir = TensorIr::new();
                let (x, y) = typed(&mut ir);
                let a = ir.div(x, y).unwrap();
                let value = ir.unary_math(a, kind).unwrap();
                let output = ir.mul(value, y).unwrap();
                assert_eq!(check(&ir, &[output], &inputs), 1, "{kind:?}");
                let mut ir = TensorIr::new();
                let x = ir.input_typed("x", ordered_shape.clone(), dtype).unwrap();
                let y = ir.input_typed("y", ordered_shape.clone(), dtype).unwrap();
                let a = ir.div(x, y).unwrap();
                let value = ir.unary_math(a, kind).unwrap();
                let output = ir.mul(value, y).unwrap();
                assert_eq!(check_fused(&ir, &[output], &ordered), 1, "{kind:?}");
            }
            for kind in BinaryMathKind::ALL {
                let mut ir = TensorIr::new();
                let (x, y) = typed(&mut ir);
                let a = ir.sub(x, y).unwrap();
                let value = ir.binary_math(a, y, kind).unwrap();
                let reversed = ir.binary_math(y, value, kind).unwrap();
                let output = ir.add(value, reversed).unwrap();
                assert_eq!(check(&ir, &[output], &inputs), 1, "{kind:?}");
            }
        }
    }

    /// A sum or product of two NaNs takes the NaN that the evaluator's
    /// kernels pick, also where one operand is a uniform (a single-element
    /// input) and the loop is vectorized: such a run falls back to node by
    /// node. A run without NaN pairs keeps the fused program.
    #[test]
    fn nan_pairs_fall_back_to_node_by_node() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let mut ir = TensorIr::new();
            let x = ir.constant(tensor(&[6], dtype, 20), false);
            let s = ir.constant(tensor(&[], dtype, 21), false);
            let product = ir.mul(x, s).unwrap();
            let output = ir.add(product, x).unwrap();
            let kernel = ir.fused_kernel(output, &[x, s]).unwrap();
            let nan = tensor(&[6], dtype, 20);
            let scalar = tensor(&[], dtype, 21);
            let finite = tensor(&[6], dtype, 5);
            let operands = |x| [EagerOperand::array(x), EagerOperand::array(&scalar)];
            assert!(kernel.evaluate(&operands(&nan)).unwrap().is_none());
            assert!(kernel.evaluate(&operands(&finite)).unwrap().is_some());
            // Two NaN constants: the region is left to the evaluator.
            let mut ir = TensorIr::new();
            let x = ir.constant(tensor(&[6], dtype, 5), false);
            let nan = ir.scalar_constant(f64::NAN);
            let negative = ir.scalar_constant(-f64::NAN);
            let sum = ir.add(nan, negative).unwrap();
            let output = ir.mul(x, sum).unwrap();
            let output = ir.sin(output).unwrap();
            assert!(ir.fused_kernel(output, &[x]).is_none());
            for scalar in SPECIAL.iter().filter(|value| value.is_nan()) {
                let inputs = BTreeMap::from([
                    ("x".to_string(), tensor(&[2 * SPECIAL.len()], dtype, 0)),
                    (
                        "s".to_string(),
                        DynamicTensor::with_dtype(vec![], vec![*scalar], dtype).unwrap(),
                    ),
                ]);
                let mut ir = TensorIr::new();
                let x = ir.input_typed("x", vec![2 * SPECIAL.len()], dtype).unwrap();
                let s = ir.input_typed("s", vec![], dtype).unwrap();
                let products = [ir.mul(x, s).unwrap(), ir.mul(s, x).unwrap()];
                let sums = [ir.add(x, s).unwrap(), ir.add(s, x).unwrap()];
                let left = ir.sub(products[0], sums[1]).unwrap();
                let right = ir.sub(sums[0], products[1]).unwrap();
                let output = ir.div(left, right).unwrap();
                assert_eq!(check(&ir, &[output], &inputs), 1);
            }
        }
    }

    /// A signaling `float32` NaN (from a buffer import) is quieted wherever
    /// the evaluator rounds a value, selected or computed, and kept where it
    /// passes a value through.
    #[test]
    fn signaling_nan_inputs_round_like_the_evaluator() {
        let data = vec![
            f32::from_bits(0xff80_0123),
            f32::from_bits(0x7f80_0001),
            1.5,
            -0.0,
        ];
        let signaling =
            DynamicTensor::from_storage(vec![4], HostTensorStorage::F32(Arc::new(data))).unwrap();
        let inputs = BTreeMap::from([
            ("x".to_string(), signaling),
            ("y".to_string(), tensor(&[4], TensorDType::F32, 3)),
        ]);
        let builds: Vec<Build> = vec![
            |ir, x, y| {
                let condition = ir.greater(y, x)?;
                ir.where_select(condition, x, y)
            },
            |ir, x, y| {
                let condition = ir.greater(y, x)?;
                let passed = ir.stop_gradient(x)?;
                let selected = ir.where_select(condition, passed, y)?;
                ir.stop_gradient(selected)
            },
            |ir, x, _| {
                let passed = ir.stop_gradient(x)?;
                ir.stop_gradient(passed)
            },
            |ir, x, y| {
                let wide = ir.cast(x, TensorDType::F64)?;
                let other = ir.cast(y, TensorDType::F64)?;
                let condition = ir.greater(other, wide)?;
                let selected = ir.where_select(condition, wide, other)?;
                ir.cast(selected, TensorDType::F32)
            },
        ];
        for build in builds {
            let mut ir = TensorIr::new();
            let x = ir.input_typed("x", vec![4], TensorDType::F32).unwrap();
            let y = ir.input_typed("y", vec![4], TensorDType::F32).unwrap();
            let output = build(&mut ir, x, y).unwrap();
            check(&ir, &[output], &inputs);
        }
    }

    #[test]
    fn broadcast_and_scalar_operands_match_node_by_node() {
        let shapes: [(&[usize], &[usize]); 6] = [
            (&[4, 1], &[1, 6]),
            (&[6], &[4, 6]),
            (&[], &[4, 6]),
            (&[3, 1, 5], &[2, 5]),
            (&[1], &[300, 3]),
            (&[2, 1, 1], &[1, 129, 3]),
        ];
        // NaN-free operands (offsets that skip the NaNs) keep the fused
        // program; operands with NaNs may fall back.
        let finite = |shape: &[usize], dtype, offset: usize| {
            let count: usize = shape.iter().product();
            let values = (0..count)
                .map(|index| SPECIAL[(index * 7 + offset) % 20])
                .collect();
            DynamicTensor::with_dtype(shape.to_vec(), values, dtype).unwrap()
        };
        for dtype in [TensorDType::F32, TensorDType::F64] {
            for (lhs, rhs) in shapes {
                let mut ir = TensorIr::new();
                let x = ir.input_typed("x", lhs.to_vec(), dtype).unwrap();
                let y = ir.input_typed("y", rhs.to_vec(), dtype).unwrap();
                let product = ir.mul(x, y).unwrap();
                let sine = ir.sin(product).unwrap();
                let shifted = ir.sub(sine, x).unwrap();
                let condition = ir.greater(y, x).unwrap();
                let selected = ir.where_select(condition, shifted, y).unwrap();
                let shape = ir.node_shape(selected).unwrap();
                let wide = ir.broadcast_to(x, shape).unwrap();
                let output = ir.add(selected, wide).unwrap();
                let inputs = BTreeMap::from([
                    ("x".to_string(), tensor(lhs, dtype, 1)),
                    ("y".to_string(), tensor(rhs, dtype, 4)),
                ]);
                assert_eq!(check(&ir, &[output], &inputs), 1, "{lhs:?} {rhs:?}");
                let inputs = BTreeMap::from([
                    ("x".to_string(), finite(lhs, dtype, 1)),
                    ("y".to_string(), finite(rhs, dtype, 4)),
                ]);
                assert_eq!(check_fused(&ir, &[output], &inputs), 1, "{lhs:?} {rhs:?}");
            }
        }
    }

    /// Values used outside a region, retained outputs, and operands of
    /// non-elementwise nodes bound regions; a lone node keeps its kernel.
    #[test]
    fn region_boundaries_keep_shared_and_retained_values() {
        let inputs = BTreeMap::from([("x".to_string(), tensor(&[3, 4], TensorDType::F64, 2))]);
        let mut ir = TensorIr::new();
        let x = ir.input("x", vec![3, 4]).unwrap();
        let a = ir.exp(x).unwrap();
        let b = ir.sin(a).unwrap();
        let c = ir.mul(b, a).unwrap();
        let total = ir.sum(c).unwrap();
        let scaled = ir.mul(b, total).unwrap();
        let d = ir.tanh(scaled).unwrap();
        let output = ir.add(d, b).unwrap();
        // `b` is used by both sides of the reduction, so only
        // `tanh(b * total) + b` fuses; a retained `d` cannot be interior.
        assert_eq!(check(&ir, &[output], &inputs), 1);
        assert_eq!(check(&ir, &[output, a], &inputs), 1);
        assert_eq!(check(&ir, &[output, c, d], &inputs), 1);

        // A value shared by two regions is stored once and read by both.
        let mut ir = TensorIr::new();
        let x = ir.input("x", vec![3, 4]).unwrap();
        let s = ir.sin(x).unwrap();
        let shared = ir.mul(s, x).unwrap();
        let p = ir.exp(shared).unwrap();
        let p = ir.mul(p, x).unwrap();
        let q = ir.cos(shared).unwrap();
        let q = ir.add(q, x).unwrap();
        assert_eq!(check(&ir, &[p, q], &inputs), 3);

        let mut ir = TensorIr::new();
        let x = ir.input("x", vec![3, 4]).unwrap();
        let output = ir.exp(x).unwrap();
        assert_eq!(check(&ir, &[output], &inputs), 0);
    }

    /// A fused region stores only its root: the peak of live values is the
    /// inputs and the output, where node by node also holds intermediates
    /// that several nodes read (`t`, `t * t`, and `1 + t` next to `x`).
    #[test]
    fn fused_region_stores_only_its_root() {
        const N: usize = 1000;
        let mut ir = TensorIr::new();
        let x = ir.input("x", vec![N]).unwrap();
        let y = ir.input("y", vec![N]).unwrap();
        let product = ir.mul(x, y).unwrap();
        let t = ir.exp(product).unwrap();
        let square = ir.mul(t, t).unwrap();
        let one = ir.scalar_constant(1.0);
        let shifted = ir.add(one, t).unwrap();
        let ratio = ir.div(t, shifted).unwrap();
        let mut value = ir.add(square, ratio).unwrap();
        for _ in 0..20 {
            let sine = ir.sin(value).unwrap();
            value = ir.add(sine, x).unwrap();
        }
        let plan = ir.compile_cpu(value).unwrap();
        let finite = |offset| {
            let values = (0..N).map(|index| SPECIAL[(index * 7 + offset) % 20]);
            DynamicTensor::new(vec![N], values.collect()).unwrap()
        };
        let inputs = BTreeMap::from([("x".to_string(), finite(0)), ("y".to_string(), finite(5))]);
        let peak = |fusion: Option<&CpuFusionPlan>| {
            CPU_FORWARD_PEAK_ELEMENTS.with(|peak| peak.set(0));
            let output = [plan.output_node_id];
            TensorIr::evaluate_tensor_nodes_fused(&plan.nodes, &inputs, Some(&output), fusion)
                .unwrap();
            CPU_FORWARD_PEAK_ELEMENTS.with(|peak| peak.get())
        };
        let before = FALLBACKS.with(Cell::get);
        assert_eq!(peak(Some(&plan.cpu_fusion)), 3 * N);
        assert_eq!(FALLBACKS.with(Cell::get), before);
        assert_eq!(peak(None), 4 * N + 1);
    }

    /// The `relu` graph of an eager call, over a constant operand.
    fn relu_graph(operand: &DynamicTensor) -> (TensorIr, TensorNodeId, TensorNodeId) {
        let mut ir = TensorIr::new();
        let x = ir.constant(operand.clone(), false);
        let zero = ir.scalar_constant(0.0);
        let nan = ir.isnan(x).unwrap();
        let positive = ir.compare(x, zero, TensorComparison::Greater).unwrap();
        let mask = ir.logical_or(nan, positive).unwrap();
        let zero = ir.scalar_constant(0.0);
        let output = ir.where_select(mask, x, zero).unwrap();
        (ir, x, output)
    }

    #[test]
    fn fused_kernel_runs_a_whole_graph_over_its_operands() {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let (ir, x, output) = relu_graph(&tensor(&[5, 3], dtype, 2));
            let kernel = ir.fused_kernel(output, &[x]).unwrap();
            for offset in [2, 7] {
                let operand = tensor(&[5, 3], dtype, offset);
                let (graph, _, output) = relu_graph(&operand);
                let expected = graph.evaluate(output, &BTreeMap::new()).unwrap();
                let actual = kernel
                    .evaluate(&[EagerOperand::array(&operand)])
                    .unwrap()
                    .unwrap();
                assert_eq!(actual.dtype, expected.dtype);
                assert_eq!(bits(&actual), bits(&expected));
            }
            let transposed = tensor(&[3, 5], dtype, 0);
            assert!(kernel
                .evaluate(&[EagerOperand::array(&transposed)])
                .is_err());
        }
        // A reduction is not elementwise; a lone node is not worth a kernel.
        let mut ir = TensorIr::new();
        let x = ir.constant(tensor(&[4], TensorDType::F64, 0), false);
        let e = ir.exp(x).unwrap();
        let total = ir.sum(e).unwrap();
        let output = ir.mul(e, total).unwrap();
        assert!(ir.fused_kernel(output, &[x]).is_none());
        assert!(ir.fused_kernel(e, &[x]).is_none());
    }
}
