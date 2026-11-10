//! How a device placement flows through each op. An op answers with one
//! of a few placement rules; sharding propagation (`infer_tensor_placement`)
//! interprets them against the placements of the op's inputs.

use super::super::{TensorOp, TensorReplicaReduction};

/// The placement rule of an op. A rule that reads "the input" applies to an
/// op with one input; "the inputs" are merged in [`TensorOp::inputs`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::tensor_ir) enum PlacementRule<'a> {
    /// No inputs: the node is placed only by an explicit annotation.
    Unplaced,
    /// A region or custom call, whose placement needs explicit lowering.
    Unsupported,
    /// The placement of the input, unchanged.
    Follow,
    /// The merged placement of the inputs.
    Merge,
    /// The input's (or with `merge`, the inputs') placement, rejected when
    /// sharded along `axis`, which the op `action`s along: local shards would
    /// not form a contiguous shard of the global result.
    LocalAlongAxis {
        merge: bool,
        axis: usize,
        action: &'static str,
    },
    /// A max or min along `axis`: there is no replica reduction for it, so a
    /// shard along `axis` is rejected and a shard along a later axis shifts
    /// down past the removed one.
    ExtremumAxis { axis: usize },
    /// A reduction to a scalar, all-reduced across replicas when sharded.
    FullReduction(TensorReplicaReduction),
    /// A reduction along `axis`, all-reduced when sharded along it.
    AxisReduction {
        axis: usize,
        reduction: TensorReplicaReduction,
    },
    /// The input's (or with `merge`, the inputs') placement, rejected when
    /// sharded at all, since `lowering` cannot split the op.
    Unsharded { merge: bool, lowering: &'static str },
    /// The input's placement, rejected when sharded and the shape changes.
    Reshape,
    /// The input's placement with its sharded axis permuted by `axes`.
    Transpose(&'a [usize]),
    /// The input's placement with its sharded axis shifted by the new
    /// leading axes.
    Broadcast,
}

impl TensorOp {
    /// How a device placement flows through the op.
    pub(in crate::tensor_ir) fn placement_rule(&self) -> PlacementRule<'_> {
        use TensorReplicaReduction::{Mean, Sum};
        match self {
            Self::Input { .. } | Self::ScalarConstant { .. } | Self::Constant { .. } => {
                PlacementRule::Unplaced
            }
            Self::Region(_) | Self::Custom { .. } => PlacementRule::Unsupported,
            Self::Sqrt { .. }
            | Self::SqrtDerivative { .. }
            | Self::Powi { .. }
            | Self::UnaryMath { .. }
            | Self::StopGradient { .. }
            | Self::Cast { .. } => PlacementRule::Follow,
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Div { .. }
            | Self::Mul { .. }
            | Self::Greater { .. }
            | Self::Compare { .. }
            | Self::BinaryMath { .. }
            | Self::Where { .. } => PlacementRule::Merge,
            Self::CumSum { axis, .. } => PlacementRule::LocalAlongAxis {
                merge: false,
                axis: *axis,
                action: "scans",
            },
            Self::Concat { axis, .. } => PlacementRule::LocalAlongAxis {
                merge: true,
                axis: *axis,
                action: "concatenates",
            },
            Self::Slice { axis, .. } | Self::PadSlice { axis, .. } | Self::Gather { axis, .. } => {
                PlacementRule::LocalAlongAxis {
                    merge: false,
                    axis: *axis,
                    action: "slices",
                }
            }
            Self::ScatterAdd { axis, .. } => PlacementRule::LocalAlongAxis {
                merge: true,
                axis: *axis,
                action: "scatters",
            },
            Self::ExtremumAxis { axis, .. } => PlacementRule::ExtremumAxis { axis: *axis },
            Self::Sum { .. } => PlacementRule::FullReduction(Sum),
            Self::Mean { .. } => PlacementRule::FullReduction(Mean),
            Self::SumAxis { axis, .. } => PlacementRule::AxisReduction {
                axis: *axis,
                reduction: Sum,
            },
            Self::MeanAxis { axis, .. } => PlacementRule::AxisReduction {
                axis: *axis,
                reduction: Mean,
            },
            Self::Matmul { .. } | Self::Solve { .. } => PlacementRule::Unsharded {
                merge: true,
                lowering: "matmul/solve lowering",
            },
            Self::CholeskyAd { .. } => PlacementRule::Unsharded {
                merge: true,
                lowering: "cholesky AD lowering",
            },
            Self::Cholesky { .. } | Self::Triangular { .. } => PlacementRule::Unsharded {
                merge: false,
                lowering: "triangular lowering",
            },
            Self::Linalg { .. } => PlacementRule::Unsharded {
                merge: false,
                lowering: "linalg lowering",
            },
            Self::Reshape { .. } => PlacementRule::Reshape,
            Self::Transpose { axes, .. } => PlacementRule::Transpose(axes),
            Self::Broadcast { .. } => PlacementRule::Broadcast,
        }
    }
}
