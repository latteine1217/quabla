//! Rebuilding an op on new operands through its public builder, which
//! re-infers the result shape from the operands. The reverse-mode primal
//! replay and the data-parallel shard specialization both copy a program
//! this way.

use super::super::{TensorIr, TensorNode, TensorNodeId, TensorOp};

impl TensorIr {
    /// Appends `node`'s op reading `operand(input)` for each input and
    /// returns the new node. An op the builder cannot rebuild on its own (an
    /// input, a region, or a custom call, whose copies depend on the
    /// transform) is an error; each transform handles those first.
    pub(in crate::tensor_ir) fn rebuild_like(
        &mut self,
        node: &TensorNode,
        operand: impl Fn(TensorNodeId) -> Result<TensorNodeId, String>,
    ) -> Result<TensorNodeId, String> {
        let operands = |inputs: &[TensorNodeId]| {
            inputs
                .iter()
                .map(|input| operand(*input))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(match &node.op {
            TensorOp::Input { .. } | TensorOp::Region(_) | TensorOp::Custom { .. } => {
                return Err(format!(
                    "{} nodes are rebuilt by the transform that copies them",
                    node.op.name()
                ))
            }
            TensorOp::ScalarConstant { value } => self.constant_like(*value, node.dtype, node.weak),
            TensorOp::Constant { .. } => {
                self.push_node(node.op.clone(), node.shape.clone(), node.dtype, node.weak)
            }
            TensorOp::Cast { input } => {
                self.rebuild_cast(operand(*input)?, node.dtype, node.weak)?
            }
            TensorOp::Add { lhs, rhs } => self.add(operand(*lhs)?, operand(*rhs)?)?,
            TensorOp::Sub { lhs, rhs } => self.sub(operand(*lhs)?, operand(*rhs)?)?,
            TensorOp::Div { lhs, rhs } => self.div(operand(*lhs)?, operand(*rhs)?)?,
            TensorOp::Mul { lhs, rhs } => self.mul(operand(*lhs)?, operand(*rhs)?)?,
            TensorOp::Greater { lhs, rhs } => self.greater(operand(*lhs)?, operand(*rhs)?)?,
            TensorOp::Compare { lhs, rhs, kind } => {
                self.compare(operand(*lhs)?, operand(*rhs)?, *kind)?
            }
            TensorOp::Where {
                condition,
                on_true,
                on_false,
            } => self.where_select(
                operand(*condition)?,
                operand(*on_true)?,
                operand(*on_false)?,
            )?,
            TensorOp::Sum { input } => self.sum(operand(*input)?)?,
            TensorOp::SumAxis { input, axis } => self.sum_axis(operand(*input)?, *axis as isize)?,
            TensorOp::ExtremumAxis { input, axis, kind } => {
                self.extremum_axis(operand(*input)?, *axis as isize, *kind)?
            }
            TensorOp::Matmul { lhs, rhs } => self.matmul(operand(*lhs)?, operand(*rhs)?)?,
            TensorOp::Cholesky { input } => self.cholesky(operand(*input)?)?,
            TensorOp::CholeskyAd { inputs, kind } => self.cholesky_ad(operands(inputs)?, *kind)?,
            TensorOp::Solve { matrix, rhs } => self.solve(operand(*matrix)?, operand(*rhs)?)?,
            TensorOp::Linalg { input, kind } => self.linalg(operand(*input)?, *kind)?,
            TensorOp::Triangular { input, lower } => self.triangular(operand(*input)?, *lower)?,
            TensorOp::Sqrt { input } => self.sqrt(operand(*input)?)?,
            TensorOp::SqrtDerivative { input, order } => {
                self.sqrt_derivative(operand(*input)?, *order)?
            }
            TensorOp::Reshape { input } => self.reshape(operand(*input)?, node.shape.clone())?,
            TensorOp::Mean { input } => self.mean(operand(*input)?)?,
            TensorOp::MeanAxis { input, axis } => {
                self.mean_axis(operand(*input)?, *axis as isize)?
            }
            TensorOp::Powi { input, exponent } => self.powi(operand(*input)?, *exponent)?,
            TensorOp::BinaryMath { lhs, rhs, kind } => {
                self.binary_math(operand(*lhs)?, operand(*rhs)?, *kind)?
            }
            TensorOp::Transpose { input, axes } => self.transpose(
                operand(*input)?,
                Some(axes.iter().map(|axis| *axis as isize).collect()),
            )?,
            TensorOp::UnaryMath { input, kind } => self.unary_math(operand(*input)?, *kind)?,
            TensorOp::StopGradient { input } => self.stop_gradient(operand(*input)?)?,
            TensorOp::CumSum {
                input,
                axis,
                reverse,
            } => self.cumsum(operand(*input)?, *axis as isize, *reverse)?,
            TensorOp::Concat { inputs, axis } => self.concat(operands(inputs)?, *axis as isize)?,
            TensorOp::Slice {
                input,
                axis,
                start,
                length,
            } => self.slice(operand(*input)?, *axis, *start, *length)?,
            TensorOp::PadSlice { input, axis, start } => {
                self.pad_slice(operand(*input)?, node.shape.clone(), *axis, *start)?
            }
            TensorOp::Gather {
                input,
                indices,
                axis,
            } => self.gather(operand(*input)?, indices.clone(), *axis)?,
            TensorOp::ScatterAdd {
                base,
                updates,
                indices,
                axis,
            } => self.scatter_add(operand(*base)?, operand(*updates)?, indices.clone(), *axis)?,
            TensorOp::Broadcast { input } => {
                self.broadcast_to(operand(*input)?, node.shape.clone())?
            }
        })
    }
}
