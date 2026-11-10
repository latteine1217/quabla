//! The reverse-mode rule of each op: the contributions the cotangent of its
//! result makes to the cotangents of its inputs, emitted into the VJP graph.
//! Each rule emits its nodes in a fixed order, which the program text of
//! every VJP graph records.

use super::super::{
    element_count, inverse_permutation, symbolic_accumulate, symbolic_broadcast_like,
    symbolic_expand_reduced_axis, symbolic_full_like, symbolic_reduce_to_shape,
    symbolic_transpose_last_two, CholeskyAdKind, TensorDType, TensorIr, TensorNodeId, TensorOp,
};

impl TensorIr {
    /// Accumulates the contributions of `upstream`, the cotangent of node
    /// `node_id` of this source program, into `cotangents` of its inputs.
    /// `values` maps source nodes to their rebuilt primal values in
    /// `transformed`. An input, a region, or a custom call is an error: its
    /// rule depends on the state of the VJP pass, which handles those first.
    pub(in crate::tensor_ir) fn push_vjp_contributions(
        &self,
        node_id: TensorNodeId,
        upstream: TensorNodeId,
        transformed: &mut TensorIr,
        values: &[TensorNodeId],
        cotangents: &mut [Option<TensorNodeId>],
    ) -> Result<(), String> {
        let node = self.node(node_id)?;
        match &node.op {
            TensorOp::Input { .. } | TensorOp::Region(_) | TensorOp::Custom { .. } => {
                return Err(format!(
                    "{} nodes are differentiated by the VJP pass",
                    node.op.name()
                ))
            }
            TensorOp::ScalarConstant { .. }
            | TensorOp::Constant { .. }
            | TensorOp::Greater { .. }
            | TensorOp::Compare { .. } => {}
            // The cast VJP converts the cotangent back to the source dtype; Bool sources are
            // not differentiable and get no gradient.
            TensorOp::Cast { input } => {
                let source_dtype = self.node(*input)?.dtype;
                if source_dtype != TensorDType::Bool {
                    let contribution = transformed.cast(upstream, source_dtype)?;
                    symbolic_accumulate(transformed, cotangents, *input, contribution)?;
                }
            }
            TensorOp::Add { lhs, rhs } => {
                let lhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    upstream,
                    &node.shape,
                    &self.node(*lhs)?.shape,
                )?;
                let rhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    upstream,
                    &node.shape,
                    &self.node(*rhs)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *lhs, lhs_contribution)?;
                symbolic_accumulate(transformed, cotangents, *rhs, rhs_contribution)?;
            }
            TensorOp::Sub { lhs, rhs } => {
                let lhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    upstream,
                    &node.shape,
                    &self.node(*lhs)?.shape,
                )?;
                let zero = transformed.scalar_constant(0.0);
                let negated = transformed.sub(zero, upstream)?;
                let rhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    negated,
                    &node.shape,
                    &self.node(*rhs)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *lhs, lhs_contribution)?;
                symbolic_accumulate(transformed, cotangents, *rhs, rhs_contribution)?;
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
                    transformed,
                    lhs_contribution,
                    &node.shape,
                    &self.node(*lhs)?.shape,
                )?;
                let rhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    rhs_contribution,
                    &node.shape,
                    &self.node(*rhs)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *lhs, lhs_contribution)?;
                symbolic_accumulate(transformed, cotangents, *rhs, rhs_contribution)?;
            }
            TensorOp::Mul { lhs, rhs } => {
                let lhs_contribution = transformed.mul(upstream, values[*rhs])?;
                let rhs_contribution = transformed.mul(upstream, values[*lhs])?;
                let lhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    lhs_contribution,
                    &node.shape,
                    &self.node(*lhs)?.shape,
                )?;
                let rhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    rhs_contribution,
                    &node.shape,
                    &self.node(*rhs)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *lhs, lhs_contribution)?;
                symbolic_accumulate(transformed, cotangents, *rhs, rhs_contribution)?;
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
                    transformed,
                    true_contribution,
                    &node.shape,
                    &self.node(*on_true)?.shape,
                )?;
                let false_contribution = symbolic_reduce_to_shape(
                    transformed,
                    false_contribution,
                    &node.shape,
                    &self.node(*on_false)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *on_true, true_contribution)?;
                symbolic_accumulate(transformed, cotangents, *on_false, false_contribution)?;
            }
            TensorOp::Sum { input } => {
                let contribution = symbolic_broadcast_like(transformed, upstream, values[*input])?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Broadcast { input } => {
                let contribution = symbolic_reduce_to_shape(
                    transformed,
                    upstream,
                    &node.shape,
                    &self.node(*input)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::SumAxis { input, axis } => {
                let contribution =
                    symbolic_expand_reduced_axis(transformed, upstream, values[*input], *axis)?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::ExtremumAxis { input, axis, .. } => {
                let contribution = transformed.extremum_axis_vjp(
                    values[*input],
                    values[node_id],
                    upstream,
                    *axis,
                )?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Matmul { lhs, rhs } => {
                let lhs_transposed = symbolic_transpose_last_two(
                    transformed,
                    values[*lhs],
                    &self.node(*lhs)?.shape,
                )?;
                let rhs_transposed = symbolic_transpose_last_two(
                    transformed,
                    values[*rhs],
                    &self.node(*rhs)?.shape,
                )?;
                let lhs_contribution = transformed.matmul(upstream, rhs_transposed)?;
                let rhs_contribution = transformed.matmul(lhs_transposed, upstream)?;
                let lhs_shape = transformed.node_shape(lhs_contribution)?;
                let rhs_shape = transformed.node_shape(rhs_contribution)?;
                let lhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    lhs_contribution,
                    &lhs_shape,
                    &self.node(*lhs)?.shape,
                )?;
                let rhs_contribution = symbolic_reduce_to_shape(
                    transformed,
                    rhs_contribution,
                    &rhs_shape,
                    &self.node(*rhs)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *lhs, lhs_contribution)?;
                symbolic_accumulate(transformed, cotangents, *rhs, rhs_contribution)?;
            }
            TensorOp::Cholesky { input } => {
                let contribution =
                    transformed.cholesky_ad(vec![values[*input], upstream], CholeskyAdKind::Vjp)?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::CholeskyAd { inputs, kind } => {
                let zero = symbolic_full_like(transformed, 0.0, upstream)?;
                let (input_gradient, other_gradient) = match kind {
                    CholeskyAdKind::Jvp => (
                        transformed.cholesky_ad(
                            vec![values[inputs[0]], values[inputs[1]], upstream, zero],
                            CholeskyAdKind::VjpJvp,
                        )?,
                        transformed
                            .cholesky_ad(vec![values[inputs[0]], upstream], CholeskyAdKind::Vjp)?,
                    ),
                    CholeskyAdKind::Vjp => (
                        transformed.cholesky_ad(
                            vec![values[inputs[0]], upstream, values[inputs[1]], zero],
                            CholeskyAdKind::VjpJvp,
                        )?,
                        transformed
                            .cholesky_ad(vec![values[inputs[0]], upstream], CholeskyAdKind::Jvp)?,
                    ),
                    _ => unreachable!("higher Cholesky derivatives expanded"),
                };
                symbolic_accumulate(transformed, cotangents, inputs[0], input_gradient)?;
                symbolic_accumulate(transformed, cotangents, inputs[1], other_gradient)?;
            }
            TensorOp::Solve { matrix, rhs } => {
                let matrix_shape = self.node(*matrix)?.shape.clone();
                let mut axes = (0..matrix_shape.len())
                    .map(|axis| axis as isize)
                    .collect::<Vec<_>>();
                axes.swap(matrix_shape.len() - 2, matrix_shape.len() - 1);
                let transposed = transformed.transpose(values[*matrix], Some(axes))?;
                let rhs_contribution = transformed.solve(transposed, upstream)?;
                let value_transposed =
                    symbolic_transpose_last_two(transformed, values[node_id], &node.shape)?;
                let matrix_contribution = transformed.matmul(rhs_contribution, value_transposed)?;
                let zero = transformed.scalar_constant(0.0);
                let matrix_contribution = transformed.sub(zero, matrix_contribution)?;
                symbolic_accumulate(transformed, cotangents, *matrix, matrix_contribution)?;
                symbolic_accumulate(transformed, cotangents, *rhs, rhs_contribution)?;
            }
            TensorOp::Triangular { input, lower } => {
                let contribution = transformed.triangular(upstream, *lower)?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Linalg { input, kind } => {
                if let Some(contribution) =
                    transformed.linalg_vjp(values[*input], values[node_id], upstream, *kind)?
                {
                    symbolic_accumulate(transformed, cotangents, *input, contribution)?;
                }
            }
            TensorOp::Sqrt { input } => {
                let derivative = transformed.sqrt_derivative(values[*input], 1)?;
                let contribution = transformed.mul(upstream, derivative)?;
                let contribution = symbolic_reduce_to_shape(
                    transformed,
                    contribution,
                    &node.shape,
                    &self.node(*input)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::SqrtDerivative { input, order } => {
                let next_order = order
                    .checked_add(1)
                    .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                let derivative = transformed.sqrt_derivative(values[*input], next_order)?;
                let contribution = transformed.mul(upstream, derivative)?;
                let contribution = symbolic_reduce_to_shape(
                    transformed,
                    contribution,
                    &node.shape,
                    &self.node(*input)?.shape,
                )?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Reshape { input } => {
                let contribution =
                    transformed.reshape(upstream, self.node(*input)?.shape.clone())?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Mean { input } => {
                let contribution = symbolic_broadcast_like(transformed, upstream, values[*input])?;
                let scale = transformed
                    .scalar_constant(1.0 / element_count(&self.node(*input)?.shape)? as f64);
                let contribution = transformed.mul(contribution, scale)?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::MeanAxis { input, axis } => {
                let contribution =
                    symbolic_expand_reduced_axis(transformed, upstream, values[*input], *axis)?;
                let scale =
                    transformed.scalar_constant(1.0 / self.node(*input)?.shape[*axis] as f64);
                let contribution = transformed.mul(contribution, scale)?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Powi { input, exponent } => {
                if *exponent != 0 {
                    let coefficient = transformed.scalar_constant(*exponent as f64);
                    let lower_power = transformed.powi(values[*input], *exponent - 1)?;
                    let derivative = transformed.mul(coefficient, lower_power)?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(transformed, cotangents, *input, contribution)?;
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
                        symbolic_accumulate(transformed, cotangents, operands[index], contribution)
                    },
                )?;
            }
            TensorOp::Transpose { input, axes } => {
                let inverse = inverse_permutation(axes)?;
                let contribution = transformed.transpose(
                    upstream,
                    Some(inverse.into_iter().map(|axis| axis as isize).collect()),
                )?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::UnaryMath { input, kind } => {
                let chain =
                    transformed.unary_math_chain(*kind, values[*input], Some(values[node_id]))?;
                if let Some(contribution) = transformed.apply_unary_chain(chain, upstream)? {
                    let contribution = symbolic_reduce_to_shape(
                        transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(transformed, cotangents, *input, contribution)?;
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
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::Concat { inputs, axis } => {
                let mut start = 0;
                for input in inputs {
                    let input_shape = &self.node(*input)?.shape;
                    let contribution =
                        transformed.slice(upstream, *axis, start, input_shape[*axis])?;
                    symbolic_accumulate(transformed, cotangents, *input, contribution)?;
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
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
            }
            TensorOp::PadSlice { input, axis, start } => {
                let contribution =
                    transformed.slice(upstream, *axis, *start, self.node(*input)?.shape[*axis])?;
                symbolic_accumulate(transformed, cotangents, *input, contribution)?;
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
                symbolic_accumulate(transformed, cotangents, *base, upstream)?;
                symbolic_accumulate(transformed, cotangents, *updates, contribution)?;
            }
        }
        Ok(())
    }
}
