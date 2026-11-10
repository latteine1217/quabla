//! The answers each [`TensorOp`] gives to the questions the generic passes
//! of a tensor program ask about it, so the passes never match on the op
//! themselves. Each answer is one exhaustive match: adding an op makes the
//! compiler list every question it has to answer here.
//!
//! - its name in kernel IR, plan text, and errors ([`TensorOp::name`]).
//!
//! Lowering an op stays in each backend, whose exhaustive matches keep the
//! bit-for-bit concerns of that backend in one place.

use super::{TensorExtremum, TensorOp};

impl TensorOp {
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
