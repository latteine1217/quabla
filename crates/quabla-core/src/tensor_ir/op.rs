//! The answers each [`TensorOp`] gives to the questions the generic passes
//! of a tensor program ask about it, so the passes never match on the op
//! themselves. Each answer is one exhaustive match: adding an op makes the
//! compiler list every question it has to answer here.
//!
//! - its name in kernel IR and errors ([`TensorOp::name`]);
//! - its operands, in a fixed order ([`TensorOp::inputs`] and
//!   [`TensorOp::inputs_mut`], which list the same nodes), from which
//!   rewriting them ([`TensorOp::remap`]) and the dtype operands
//!   ([`TensorOp::value_operands`]) follow;
//! - its identity for common-subexpression elimination
//!   ([`TensorOp::cse_key`]);
//! - how its builder rebuilds it on new operands
//!   (`TensorIr::rebuild_like`, in [`rebuild`]).
//!
//! The program text (`TensorIr::lower_text`) keeps its own exhaustive match,
//! because its format is public and pinned by golden baselines. Lowering an
//! op stays in each backend, whose exhaustive matches keep the bit-for-bit
//! concerns of that backend in one place.

mod rebuild;

use super::{TensorExtremum, TensorNodeId, TensorOp};

/// The structural identity of a pure op: two ops with equal keys (and equal
/// dtype and weakness, which plan compilation adds) compute the same value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct PureTensorOpKey {
    operation: std::mem::Discriminant<TensorOp>,
    words: Box<[u64]>,
}

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

    /// The CSE key of the op with result `shape`, or `None` for an op that
    /// is never merged: inputs, array constants (deduplicated by their bits
    /// instead), Cholesky derivatives, regions, and custom ops.
    pub(super) fn cse_key(&self, shape: &[usize]) -> Option<PureTensorOpKey> {
        let id = |node: &TensorNodeId| *node as u64;
        // The fixed words of each op, how many of them are used, and its
        // variable-length list, if any.
        let (arguments, count, list): ([u64; 4], usize, Option<&[usize]>) = match self {
            Self::Input { .. }
            | Self::Constant { .. }
            | Self::CholeskyAd { .. }
            | Self::Region(_)
            | Self::Custom { .. } => return None,
            Self::ScalarConstant { value } => ([value.to_bits(), 0, 0, 0], 1, None),
            Self::Add { lhs, rhs }
            | Self::Sub { lhs, rhs }
            | Self::Mul { lhs, rhs }
            | Self::Div { lhs, rhs }
            | Self::Greater { lhs, rhs }
            | Self::Matmul { lhs, rhs }
            | Self::Solve { matrix: lhs, rhs } => ([id(lhs), id(rhs), 0, 0], 2, None),
            Self::BinaryMath { lhs, rhs, kind } => ([id(lhs), id(rhs), *kind as u64, 0], 3, None),
            Self::Compare { lhs, rhs, kind } => ([id(lhs), id(rhs), *kind as u64, 0], 3, None),
            Self::Where {
                condition,
                on_true,
                on_false,
            } => ([id(condition), id(on_true), id(on_false), 0], 3, None),
            Self::Cast { input }
            | Self::Sum { input }
            | Self::Sqrt { input }
            | Self::Reshape { input }
            | Self::Mean { input }
            | Self::StopGradient { input }
            | Self::Broadcast { input }
            | Self::Cholesky { input } => ([id(input), 0, 0, 0], 1, None),
            Self::CumSum {
                input,
                axis,
                reverse,
            } => ([id(input), *axis as u64, u64::from(*reverse), 0], 3, None),
            Self::SumAxis { input, axis } | Self::MeanAxis { input, axis } => {
                ([id(input), *axis as u64, 0, 0], 2, None)
            }
            Self::ExtremumAxis { input, axis, kind } => {
                ([id(input), *axis as u64, *kind as u64, 0], 3, None)
            }
            Self::Triangular { input, lower } => ([id(input), u64::from(*lower), 0, 0], 2, None),
            Self::Linalg { input, kind } => ([id(input), *kind as u64, 0, 0], 2, None),
            Self::UnaryMath { input, kind } => ([id(input), *kind as u64, 0, 0], 2, None),
            Self::SqrtDerivative { input, order }
            | Self::Powi {
                input,
                exponent: order,
            } => ([id(input), u64::from(*order), 0, 0], 2, None),
            Self::Transpose { input, axes } => ([id(input), 0, 0, 0], 1, Some(axes.as_slice())),
            Self::Concat { inputs, axis } => ([*axis as u64, 0, 0, 0], 1, Some(inputs.as_slice())),
            Self::Slice {
                input,
                axis,
                start,
                length,
            } => (
                [id(input), *axis as u64, *start as u64, *length as u64],
                4,
                None,
            ),
            Self::PadSlice { input, axis, start } => {
                ([id(input), *axis as u64, *start as u64, 0], 3, None)
            }
            Self::Gather {
                input,
                indices,
                axis,
            } => ([id(input), *axis as u64, 0, 0], 2, Some(indices.as_ref())),
            Self::ScatterAdd {
                base,
                updates,
                indices,
                axis,
            } => (
                [id(base), id(updates), *axis as u64, 0],
                3,
                Some(indices.as_ref()),
            ),
        };
        // Validated operand ids determine output shapes except for shape-changing ops
        // whose target shape is not encoded in the op. Avoid duplicating inferred shapes.
        let shape = if matches!(
            self,
            Self::Reshape { .. } | Self::Broadcast { .. } | Self::PadSlice { .. }
        ) {
            shape
        } else {
            &[]
        };
        // Variable operand lists carry their length so they cannot collide with shape words.
        let list_words = list.map_or(0, |values| values.len() + 1);
        let mut words = Vec::with_capacity(count + list_words + shape.len());
        words.extend_from_slice(&arguments[..count]);
        if let Some(values) = list {
            words.push(values.len() as u64);
            words.extend(values.iter().map(|value| *value as u64));
        }
        words.extend(shape.iter().map(|value| *value as u64));
        Some(PureTensorOpKey {
            operation: std::mem::discriminant(self),
            words: words.into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{TensorOp, UnaryMathKind};
    use super::PureTensorOpKey;

    #[test]
    fn keys_preserve_scalar_bits_operation_shape_and_variable_boundaries() {
        let key = |op: TensorOp, shape: &[usize]| op.cse_key(shape).unwrap();
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
