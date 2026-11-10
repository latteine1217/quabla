//! The forward-mode rule of each op: the rebuilt value and its tangent,
//! emitted into the JVP graph from the (value, tangent) pairs of the op's
//! inputs. Each rule emits its nodes in a fixed order, which the program
//! text of every JVP graph records.

use super::super::{
    symbolic_zero_like, symbolic_zero_tangent, CholeskyAdKind, TensorDType, TensorIr, TensorNode,
    TensorNodeId, TensorOp,
};

impl TensorIr {
    /// Appends the value and tangent of `node`, whose inputs map to the
    /// (value, tangent) pairs in `pairs`. An input, a region, or a custom
    /// call is an error: its tangent depends on the state of the JVP pass,
    /// which handles those first.
    pub(in crate::tensor_ir) fn push_jvp_pair(
        &mut self,
        node: &TensorNode,
        pairs: &[(TensorNodeId, TensorNodeId)],
    ) -> Result<(TensorNodeId, TensorNodeId), String> {
        Ok(match &node.op {
            TensorOp::Input { .. } | TensorOp::Region(_) | TensorOp::Custom { .. } => {
                return Err(format!(
                    "{} nodes are differentiated by the JVP pass",
                    node.op.name()
                ))
            }
            // The zero tangent keeps the constant's dtype and weakness: a
            // strong `f32` constant (such as the ones seed of `grad`) can
            // be bound to a region input, which requires an exact dtype.
            TensorOp::ScalarConstant { value } => (
                self.constant_like(*value, node.dtype, node.weak),
                self.constant_like(0.0, node.dtype, node.weak),
            ),
            TensorOp::Constant { value } => {
                let primal = self.push_node(
                    TensorOp::Constant {
                        value: value.clone(),
                    },
                    node.shape.clone(),
                    node.dtype,
                    node.weak,
                );
                let tangent = if node.dtype == TensorDType::Bool {
                    symbolic_zero_tangent(self, &node.shape)?
                } else {
                    symbolic_zero_like(self, primal)?
                };
                (primal, tangent)
            }
            // d cast(x) = cast(dx): the tangent has the same dtype as the primal.
            TensorOp::Cast { input } => {
                let (value, tangent) = pairs[*input];
                (
                    self.rebuild_cast(value, node.dtype, node.weak)?,
                    self.rebuild_cast(tangent, node.dtype, node.weak)?,
                )
            }
            TensorOp::Add { lhs, rhs } => {
                let (lhs_value, lhs_tangent) = pairs[*lhs];
                let (rhs_value, rhs_tangent) = pairs[*rhs];
                (
                    self.add(lhs_value, rhs_value)?,
                    self.add(lhs_tangent, rhs_tangent)?,
                )
            }
            TensorOp::Sub { lhs, rhs } => {
                let (lhs_value, lhs_tangent) = pairs[*lhs];
                let (rhs_value, rhs_tangent) = pairs[*rhs];
                (
                    self.sub(lhs_value, rhs_value)?,
                    self.sub(lhs_tangent, rhs_tangent)?,
                )
            }
            TensorOp::Div { lhs, rhs } => {
                // d(l/r) = (dl - q dr) / r with q = l/r; unlike the quotient
                // rule over r*r, it cannot overflow or underflow when |r|
                // is outside the square root of the dtype's range.
                let (lhs_value, lhs_tangent) = pairs[*lhs];
                let (rhs_value, rhs_tangent) = pairs[*rhs];
                let quotient = self.div(lhs_value, rhs_value)?;
                let scaled = self.mul(quotient, rhs_tangent)?;
                let numerator = self.sub(lhs_tangent, scaled)?;
                (quotient, self.div(numerator, rhs_value)?)
            }
            TensorOp::Mul { lhs, rhs } => {
                let (lhs_value, lhs_tangent) = pairs[*lhs];
                let (rhs_value, rhs_tangent) = pairs[*rhs];
                let left_term = self.mul(lhs_tangent, rhs_value)?;
                let right_term = self.mul(lhs_value, rhs_tangent)?;
                (
                    self.mul(lhs_value, rhs_value)?,
                    self.add(left_term, right_term)?,
                )
            }
            TensorOp::Greater { lhs, rhs } => {
                let (lhs_value, _) = pairs[*lhs];
                let (rhs_value, _) = pairs[*rhs];
                let value = self.greater(lhs_value, rhs_value)?;
                // Shaped like the mask, so a reshape of it (a mapped
                // predicate broadcast by `vmap`) reshapes its tangent too.
                let tangent = symbolic_zero_tangent(self, &node.shape)?;
                (value, tangent)
            }
            // The tangent of a Bool result is a weak f64 zero shaped like the value, so data
            // movement such as reshape/concat and Bool->float casts propagate zero tangents as
            // usual.
            TensorOp::Compare { lhs, rhs, kind } => {
                let (lhs_value, _) = pairs[*lhs];
                let (rhs_value, _) = pairs[*rhs];
                (
                    self.compare(lhs_value, rhs_value, *kind)?,
                    symbolic_zero_tangent(self, &node.shape)?,
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
                    self.where_select(condition_value, true_value, false_value)?,
                    self.where_select(condition_value, true_tangent, false_tangent)?,
                )
            }
            TensorOp::Sqrt { input } => {
                let (input_value, input_tangent) = pairs[*input];
                let value = self.sqrt(input_value)?;
                let derivative = self.sqrt_derivative(input_value, 1)?;
                (value, self.mul(input_tangent, derivative)?)
            }
            TensorOp::SqrtDerivative { input, order } => {
                let (input_value, input_tangent) = pairs[*input];
                let value = self.sqrt_derivative(input_value, *order)?;
                let next_order = order
                    .checked_add(1)
                    .ok_or_else(|| "sqrt derivative order overflows u32".to_string())?;
                let derivative = self.sqrt_derivative(input_value, next_order)?;
                (value, self.mul(input_tangent, derivative)?)
            }
            TensorOp::UnaryMath { input, kind } => {
                let (input_value, input_tangent) = pairs[*input];
                let (value, tangent) = self.unary_math_jvp(*kind, input_value, input_tangent)?;
                let tangent = match tangent {
                    Some(tangent) => tangent,
                    None => symbolic_zero_like(self, value)?,
                };
                (value, tangent)
            }
            TensorOp::StopGradient { input } => {
                let (input_value, _) = pairs[*input];
                let value = self.stop_gradient(input_value)?;
                let tangent = if node.dtype == TensorDType::Bool {
                    symbolic_zero_tangent(self, &node.shape)?
                } else {
                    symbolic_zero_like(self, value)?
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
                    self.cumsum(input_value, *axis as isize, *reverse)?,
                    self.cumsum(input_tangent, *axis as isize, *reverse)?,
                )
            }
            TensorOp::Matmul { lhs, rhs } => {
                let (lhs_value, lhs_tangent) = pairs[*lhs];
                let (rhs_value, rhs_tangent) = pairs[*rhs];
                let left_term = self.matmul(lhs_tangent, rhs_value)?;
                let right_term = self.matmul(lhs_value, rhs_tangent)?;
                (
                    self.matmul(lhs_value, rhs_value)?,
                    self.add(left_term, right_term)?,
                )
            }
            TensorOp::Cholesky { input } => {
                let (value, tangent) = pairs[*input];
                (
                    self.cholesky(value)?,
                    self.cholesky_ad(vec![value, tangent], CholeskyAdKind::Jvp)?,
                )
            }
            TensorOp::CholeskyAd { inputs, kind } => {
                let values = inputs.iter().map(|id| pairs[*id].0).collect::<Vec<_>>();
                let tangent = match kind {
                    CholeskyAdKind::Jvp => self.cholesky_ad(
                        vec![values[0], values[1], pairs[inputs[0]].1, pairs[inputs[1]].1],
                        CholeskyAdKind::Mixed,
                    )?,
                    CholeskyAdKind::Vjp => self.cholesky_ad(
                        vec![values[0], pairs[inputs[0]].1, values[1], pairs[inputs[1]].1],
                        CholeskyAdKind::VjpJvp,
                    )?,
                    _ => unreachable!("higher Cholesky derivatives expanded"),
                };
                (self.cholesky_ad(values, *kind)?, tangent)
            }
            TensorOp::Solve { matrix, rhs } => {
                let (matrix_value, matrix_tangent) = pairs[*matrix];
                let (rhs_value, rhs_tangent) = pairs[*rhs];
                let value = self.solve(matrix_value, rhs_value)?;
                let correction = self.matmul(matrix_tangent, value)?;
                let adjusted_rhs = self.sub(rhs_tangent, correction)?;
                let tangent = self.solve(matrix_value, adjusted_rhs)?;
                (value, tangent)
            }
            TensorOp::Triangular { input, lower } => {
                let (value, tangent) = pairs[*input];
                (
                    self.triangular(value, *lower)?,
                    self.triangular(tangent, *lower)?,
                )
            }
            TensorOp::Linalg { input, kind } => {
                let (matrix, tangent) = pairs[*input];
                let value = self.linalg(matrix, *kind)?;
                let tangent = self.linalg_jvp(matrix, value, tangent, *kind)?;
                (value, tangent)
            }
            TensorOp::Sum { input } => {
                let (value, tangent) = pairs[*input];
                (self.sum(value)?, self.sum(tangent)?)
            }
            TensorOp::Mean { input } => {
                let (value, tangent) = pairs[*input];
                (self.mean(value)?, self.mean(tangent)?)
            }
            TensorOp::SumAxis { input, axis } => {
                let (value, tangent) = pairs[*input];
                (
                    self.sum_axis(value, *axis as isize)?,
                    self.sum_axis(tangent, *axis as isize)?,
                )
            }
            TensorOp::ExtremumAxis { input, axis, kind } => {
                let (input_value, input_tangent) = pairs[*input];
                let value = self.extremum_axis(input_value, *axis as isize, *kind)?;
                let tangent = self.extremum_axis_jvp(input_value, value, input_tangent, *axis)?;
                (value, tangent)
            }
            TensorOp::MeanAxis { input, axis } => {
                let (value, tangent) = pairs[*input];
                (
                    self.mean_axis(value, *axis as isize)?,
                    self.mean_axis(tangent, *axis as isize)?,
                )
            }
            TensorOp::Reshape { input } => {
                let (value, tangent) = pairs[*input];
                (
                    self.reshape(value, node.shape.clone())?,
                    self.reshape(tangent, node.shape.clone())?,
                )
            }
            TensorOp::Transpose { input, axes } => {
                let (value, tangent) = pairs[*input];
                let axes: Vec<isize> = axes.iter().map(|axis| *axis as isize).collect();
                (
                    self.transpose(value, Some(axes.clone()))?,
                    self.transpose(tangent, Some(axes))?,
                )
            }
            TensorOp::Powi { input, exponent } => {
                let (input_value, input_tangent) = pairs[*input];
                let value = self.powi(input_value, *exponent)?;
                let tangent = if *exponent == 0 {
                    symbolic_zero_like(self, input_value)?
                } else {
                    let coefficient = self.scalar_constant(*exponent as f64);
                    let lower_power = self.powi(input_value, *exponent - 1)?;
                    let derivative = self.mul(coefficient, lower_power)?;
                    self.mul(input_tangent, derivative)?
                };
                (value, tangent)
            }
            TensorOp::BinaryMath { lhs, rhs, kind } => {
                self.binary_math_jvp(*kind, pairs[*lhs], pairs[*rhs])?
            }
            TensorOp::Concat { inputs, axis } => {
                let values = inputs.iter().map(|input| pairs[*input].0).collect();
                let tangents = inputs.iter().map(|input| pairs[*input].1).collect();
                (
                    self.concat(values, *axis as isize)?,
                    self.concat(tangents, *axis as isize)?,
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
                    self.slice(value, *axis, *start, *length)?,
                    self.slice(tangent, *axis, *start, *length)?,
                )
            }
            TensorOp::PadSlice { input, axis, start } => {
                let (value, tangent) = pairs[*input];
                (
                    self.pad_slice(value, node.shape.clone(), *axis, *start)?,
                    self.pad_slice(tangent, node.shape.clone(), *axis, *start)?,
                )
            }
            TensorOp::Gather {
                input,
                indices,
                axis,
            } => {
                let (value, tangent) = pairs[*input];
                (
                    self.gather(value, indices.clone(), *axis)?,
                    self.gather(tangent, indices.clone(), *axis)?,
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
                    self.scatter_add(base_value, update_value, indices.clone(), *axis)?,
                    self.scatter_add(base_tangent, update_tangent, indices.clone(), *axis)?,
                )
            }
            TensorOp::Broadcast { input } => {
                let (value, tangent) = pairs[*input];
                (
                    self.broadcast_to(value, node.shape.clone())?,
                    self.broadcast_to(tangent, node.shape.clone())?,
                )
            }
        })
    }
}
