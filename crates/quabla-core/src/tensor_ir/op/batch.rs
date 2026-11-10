//! The batching rule of each op under `vmap`: the op that computes a batch
//! of its results from operands with a leading batch axis. Each rule emits
//! the nodes it needs (a flattening reshape, a broadcast of an unmapped
//! operand) in operand order, which the program text of every batched
//! program records.

use std::collections::HashMap;

use super::super::{
    batched_shape, element_count, BatchingError, TensorIr, TensorNode, TensorNodeId, TensorOp,
};

impl TensorIr {
    /// The batched op of callee `node`, reading the copies of its operands
    /// in `remap`; `mapped` tells which callee nodes carry the batch axis.
    /// Nodes an op needs before it (see the module docs) are appended here.
    /// A region or a custom call is an error: it is batched once per group,
    /// which the caller tracks.
    pub(in crate::tensor_ir) fn batched_op(
        &mut self,
        callee: &TensorIr,
        node: &TensorNode,
        (remap, mapped): (&HashMap<TensorNodeId, TensorNodeId>, &[bool]),
        batch_size: usize,
    ) -> Result<TensorOp, BatchingError> {
        let target = |operand: TensorNodeId| {
            remap
                .get(&operand)
                .copied()
                .ok_or_else(|| format!("node {operand} is missing from the vmap remap"))
        };
        // The copy of `operand` with the batch axis: an unmapped operand is
        // the same for every example and is broadcast to the batch.
        let batched_operand = |graph: &mut TensorIr, operand: TensorNodeId| {
            let spliced = target(operand)?;
            if mapped[operand] {
                return Ok::<_, String>(spliced);
            }
            let source = graph.node(spliced)?;
            let (shape, dtype, weak) = (
                batched_shape(batch_size, &source.shape),
                source.dtype,
                source.weak,
            );
            Ok(graph.push_node(TensorOp::Broadcast { input: spliced }, shape, dtype, weak))
        };
        let rank = node.shape.len();
        Ok(match &node.op {
            TensorOp::Cast { .. }
            | TensorOp::Sqrt { .. }
            | TensorOp::SqrtDerivative { .. }
            | TensorOp::Powi { .. }
            | TensorOp::UnaryMath { .. }
            | TensorOp::StopGradient { .. }
            | TensorOp::Triangular { .. }
            | TensorOp::Linalg { .. }
            | TensorOp::Reshape { .. } => node.op.remap(&|id| remap.get(&id).copied())?,
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
                for operand in node.op.inputs() {
                    let spliced = target(operand)?;
                    let spliced = if mapped[operand] {
                        self.pad_batched(spliced, rank)?
                    } else {
                        spliced
                    };
                    local.insert(operand, spliced);
                }
                node.op.remap(&|id| local.get(&id).copied())?
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
            TensorOp::Concat { inputs, axis } => TensorOp::Concat {
                inputs: inputs
                    .iter()
                    .map(|input| batched_operand(self, *input))
                    .collect::<Result<_, _>>()?,
                axis: axis + 1,
            },
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
            // Both operands need the batch axis.
            TensorOp::ScatterAdd {
                base,
                updates,
                indices,
                axis,
            } => TensorOp::ScatterAdd {
                base: batched_operand(self, *base)?,
                updates: batched_operand(self, *updates)?,
                indices: indices.clone(),
                axis: axis + 1,
            },
            TensorOp::Cholesky { input } => TensorOp::Cholesky {
                input: target(*input)?,
            },
            TensorOp::CholeskyAd { inputs, kind } => TensorOp::CholeskyAd {
                inputs: inputs
                    .iter()
                    .map(|input| batched_operand(self, *input))
                    .collect::<Result<_, _>>()?,
                kind: *kind,
            },
            // `Solve` batches over equal leading axes; an unmapped operand is
            // the same system or right-hand side for every example.
            TensorOp::Solve { matrix, rhs } => TensorOp::Solve {
                matrix: batched_operand(self, *matrix)?,
                rhs: batched_operand(self, *rhs)?,
            },
            TensorOp::Input { .. }
            | TensorOp::ScalarConstant { .. }
            | TensorOp::Constant { .. } => {
                return Err(BatchingError::Invalid(format!(
                    "{} has no operand and cannot depend on a mapped input",
                    node.op.name()
                )))
            }
            TensorOp::Region(_) | TensorOp::Custom { .. } => {
                return Err(BatchingError::Invalid(format!(
                    "{} nodes are batched once per group by inline_batched",
                    node.op.name()
                )))
            }
        })
    }
}
