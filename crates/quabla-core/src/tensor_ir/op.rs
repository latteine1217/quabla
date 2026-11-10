//! The answers each [`TensorOp`] gives to the questions the generic passes
//! of a tensor program ask about it, so the passes never match on the op
//! themselves. Each answer is one exhaustive match: adding an op makes the
//! compiler list every question it has to answer here.
//!
//! - its name in kernel IR and errors ([`TensorOp::name`]);
//! - its operands, in a fixed order ([`TensorOp::inputs`] and
//!   [`TensorOp::inputs_mut`], which list the same nodes), from which
//!   rewriting them ([`TensorOp::remap`]) and the dtype operands
//!   ([`TensorOp::value_operands`]) follow.
//!
//! The program text (`TensorIr::lower_text`) keeps its own exhaustive match,
//! because its format is public and pinned by golden baselines. Lowering an
//! op stays in each backend, whose exhaustive matches keep the bit-for-bit
//! concerns of that backend in one place.

use super::{TensorExtremum, TensorNodeId, TensorOp};

impl TensorOp {
    /// The nodes the op reads. Reachability, liveness, fusion, and inlining
    /// all walk this list. A custom op lists its value before its operands:
    /// the operands count as inputs so a rule can still be applied to them
    /// after the node is spliced.
    pub(super) fn inputs(&self) -> Vec<TensorNodeId> {
        match self {
            Self::Input { .. } | Self::ScalarConstant { .. } | Self::Constant { .. } => Vec::new(),
            Self::Add { lhs, rhs }
            | Self::Sub { lhs, rhs }
            | Self::Div { lhs, rhs }
            | Self::Mul { lhs, rhs }
            | Self::Greater { lhs, rhs }
            | Self::Compare { lhs, rhs, .. }
            | Self::Matmul { lhs, rhs }
            | Self::BinaryMath { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Solve { matrix, rhs } => vec![*matrix, *rhs],
            Self::Where {
                condition,
                on_true,
                on_false,
            } => vec![*condition, *on_true, *on_false],
            Self::ScatterAdd { base, updates, .. } => vec![*base, *updates],
            Self::CholeskyAd { inputs, .. } | Self::Concat { inputs, .. } => inputs.clone(),
            Self::Region(region) => region.operands(),
            Self::Custom {
                value, operands, ..
            } => std::iter::once(*value)
                .chain(operands.iter().copied())
                .collect(),
            Self::Cast { input }
            | Self::Sum { input }
            | Self::SumAxis { input, .. }
            | Self::ExtremumAxis { input, .. }
            | Self::Cholesky { input }
            | Self::Triangular { input, .. }
            | Self::Linalg { input, .. }
            | Self::Sqrt { input }
            | Self::SqrtDerivative { input, .. }
            | Self::Reshape { input }
            | Self::Mean { input }
            | Self::MeanAxis { input, .. }
            | Self::Powi { input, .. }
            | Self::Transpose { input, .. }
            | Self::UnaryMath { input, .. }
            | Self::StopGradient { input }
            | Self::CumSum { input, .. }
            | Self::Slice { input, .. }
            | Self::PadSlice { input, .. }
            | Self::Gather { input, .. }
            | Self::Broadcast { input } => vec![*input],
        }
    }

    /// The same nodes as [`Self::inputs`], in the same order, for rewriting.
    pub(super) fn inputs_mut(&mut self) -> Vec<&mut TensorNodeId> {
        match self {
            Self::Input { .. } | Self::ScalarConstant { .. } | Self::Constant { .. } => Vec::new(),
            Self::Add { lhs, rhs }
            | Self::Sub { lhs, rhs }
            | Self::Div { lhs, rhs }
            | Self::Mul { lhs, rhs }
            | Self::Greater { lhs, rhs }
            | Self::Compare { lhs, rhs, .. }
            | Self::Matmul { lhs, rhs }
            | Self::BinaryMath { lhs, rhs, .. } => vec![lhs, rhs],
            Self::Solve { matrix, rhs } => vec![matrix, rhs],
            Self::Where {
                condition,
                on_true,
                on_false,
            } => vec![condition, on_true, on_false],
            Self::ScatterAdd { base, updates, .. } => vec![base, updates],
            Self::CholeskyAd { inputs, .. } | Self::Concat { inputs, .. } => {
                inputs.iter_mut().collect()
            }
            Self::Region(region) => region.operands_mut(),
            Self::Custom {
                value, operands, ..
            } => std::iter::once(value).chain(operands.iter_mut()).collect(),
            Self::Cast { input }
            | Self::Sum { input }
            | Self::SumAxis { input, .. }
            | Self::ExtremumAxis { input, .. }
            | Self::Cholesky { input }
            | Self::Triangular { input, .. }
            | Self::Linalg { input, .. }
            | Self::Sqrt { input }
            | Self::SqrtDerivative { input, .. }
            | Self::Reshape { input }
            | Self::Mean { input }
            | Self::MeanAxis { input, .. }
            | Self::Powi { input, .. }
            | Self::Transpose { input, .. }
            | Self::UnaryMath { input, .. }
            | Self::StopGradient { input }
            | Self::CumSum { input, .. }
            | Self::Slice { input, .. }
            | Self::PadSlice { input, .. }
            | Self::Gather { input, .. }
            | Self::Broadcast { input } => vec![input],
        }
    }

    /// A copy of the op reading `remap(input)` for each input. A missing
    /// input is an error naming the first one, in [`Self::inputs`] order.
    pub(super) fn remap(
        &self,
        remap: &impl Fn(TensorNodeId) -> Option<TensorNodeId>,
    ) -> Result<Self, String> {
        let mut op = self.clone();
        // An input missing from `inputs_mut` would keep its stale id.
        debug_assert_eq!(
            op.inputs_mut()
                .into_iter()
                .map(|id| *id)
                .collect::<Vec<_>>(),
            self.inputs(),
            "inputs and inputs_mut of {} list different nodes",
            self.name()
        );
        for input in op.inputs_mut() {
            *input = remap(*input)
                .ok_or_else(|| format!("node {input} is missing from execution plan remap"))?;
        }
        Ok(op)
    }

    /// Operands whose dtype an ordinary op's result inherits. A `where` mask
    /// is only tested for non-zero values and therefore does not participate.
    pub(super) fn value_operands(&self) -> Vec<TensorNodeId> {
        match self {
            Self::Where {
                on_true, on_false, ..
            } => vec![*on_true, *on_false],
            Self::Custom { value, .. } => vec![*value],
            _ => self.inputs(),
        }
    }
    /// The op's name in kernel IR (`TensorKernelNode::op`), in validation
    /// and lowering errors of every backend, and in batching errors. A
    /// compare, linalg, unary math, or binary math op is named by its kind,
    /// and a region by its region kind.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Input { .. } => "input",
            Self::ScalarConstant { .. } => "constant",
            Self::Constant { .. } => "tensor_constant",
            Self::Cast { .. } => "cast",
            Self::Add { .. } => "add",
            Self::Sub { .. } => "sub",
            Self::Div { .. } => "div",
            Self::Mul { .. } => "mul",
            Self::Greater { .. } => "greater",
            Self::Compare { kind, .. } => kind.name(),
            Self::Where { .. } => "where",
            Self::Region(region) => region.name(),
            Self::Sum { .. } => "sum",
            Self::SumAxis { .. } => "sum_axis",
            Self::ExtremumAxis { kind, .. } => match kind {
                TensorExtremum::Max => "max_axis",
                TensorExtremum::Min => "min_axis",
            },
            Self::Matmul { .. } => "matmul",
            Self::Solve { .. } => "solve",
            Self::Linalg { kind, .. } => kind.name(),
            Self::Cholesky { .. } => "cholesky",
            Self::CholeskyAd { .. } => "cholesky_ad",
            Self::Triangular { .. } => "triangular",
            Self::Sqrt { .. } => "sqrt",
            Self::SqrtDerivative { .. } => "sqrt_derivative",
            Self::Reshape { .. } => "reshape",
            Self::Mean { .. } => "mean",
            Self::MeanAxis { .. } => "mean_axis",
            Self::Powi { .. } => "powi",
            Self::BinaryMath { kind, .. } => kind.name(),
            Self::Transpose { .. } => "transpose",
            Self::UnaryMath { kind, .. } => kind.name(),
            Self::StopGradient { .. } => "stop_gradient",
            Self::CumSum { .. } => "cumsum",
            Self::Concat { .. } => "concat",
            Self::Slice { .. } => "slice",
            Self::PadSlice { .. } => "pad_slice",
            Self::Gather { .. } => "gather",
            Self::ScatterAdd { .. } => "scatter_add",
            Self::Broadcast { .. } => "broadcast",
            Self::Custom { .. } => "custom",
        }
    }
}
