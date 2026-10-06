//! What every control-flow region node has in common.
//!
//! A region node (`Cond`, `While`, `Fori`, `Scan`, and the derivative nodes
//! of the fixed-bound loops) does not compute its result from its operands
//! with one primitive. It binds its operands by name to the inputs of one or
//! more frozen body regions and runs those bodies. Its operands are a few
//! fixed slots (the predicate, or the carry with its tangent and
//! cotangents), then the named captures, then the named tangent captures.
//!
//! [`RegionView`] answers the questions that do not depend on the kind of
//! region: the operands, the group, the op names, and the frozen plans. The
//! derivative rules, the evaluators, and the batching rules stay
//! kind-specific.

use super::{TensorExecutionPlan, TensorNodeId, TensorOp};

/// Matches every region variant of [`TensorOp`].
macro_rules! region_op {
    () => {
        TensorOp::Cond { .. }
            | TensorOp::While { .. }
            | TensorOp::Fori { .. }
            | TensorOp::ForiJvp { .. }
            | TensorOp::ForiVjp { .. }
            | TensorOp::ForiVjpJvp { .. }
            | TensorOp::Scan { .. }
            | TensorOp::ScanVjp { .. }
            | TensorOp::ScanVjpJvp { .. }
    };
}
pub(super) use region_op;

/// A borrowed view of a region node.
#[derive(Clone, Copy)]
pub(super) struct RegionView<'a> {
    op: &'a TensorOp,
}

impl<'a> RegionView<'a> {
    /// The view of `op`, or `None` when it is not a region node.
    pub(super) fn of(op: &'a TensorOp) -> Option<Self> {
        matches!(op, region_op!()).then_some(Self { op })
    }

    /// The view of `op`, which a `region_op!()` pattern has matched.
    pub(super) fn expect(op: &'a TensorOp) -> Self {
        Self::of(op).expect("region_op!() matched a region op")
    }

    /// The IR op name, as `lower_text` and error messages print it.
    pub(super) fn name(self) -> &'static str {
        match self.op {
            TensorOp::Cond { .. } => "cond",
            TensorOp::While { .. } => "while",
            TensorOp::Fori { .. } => "fori",
            TensorOp::ForiJvp { .. } => "fori_jvp",
            TensorOp::ForiVjp { .. } => "fori_vjp",
            TensorOp::ForiVjpJvp { .. } => "fori_vjp_jvp",
            TensorOp::Scan { .. } => "scan",
            TensorOp::ScanVjp { .. } => "scan_vjp",
            TensorOp::ScanVjpJvp { .. } => "scan_vjp_jvp",
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// The node kind as prose ("Fori VJP JVP"), for diagnostics.
    pub(super) fn label(self) -> &'static str {
        match self.op {
            TensorOp::Cond { .. } => "Cond",
            TensorOp::While { .. } => "While",
            TensorOp::Fori { .. } => "Fori",
            TensorOp::ForiJvp { .. } => "Fori JVP",
            TensorOp::ForiVjp { .. } => "Fori VJP",
            TensorOp::ForiVjpJvp { .. } => "Fori VJP JVP",
            TensorOp::Scan { .. } => "Scan",
            TensorOp::ScanVjp { .. } => "Scan VJP",
            TensorOp::ScanVjpJvp { .. } => "Scan VJP JVP",
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// Whether the node is a loop (every kind but `Cond`).
    #[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
    pub(super) fn is_loop(self) -> bool {
        !matches!(self.op, TensorOp::Cond { .. })
    }

    /// The fixed operand slots in role order: the predicate of a `Cond`;
    /// the carry of a loop, followed in a forward mode by its tangent and in
    /// a reverse mode by the cotangents, each VJP slot followed by its
    /// tangent in forward-over-reverse.
    fn slots(self) -> Vec<TensorNodeId> {
        match *self.op {
            TensorOp::Cond { predicate, .. } => vec![predicate],
            TensorOp::While { carry, .. }
            | TensorOp::Fori { carry, .. }
            | TensorOp::Scan { carry, .. } => vec![carry],
            TensorOp::ForiJvp {
                carry,
                carry_tangent,
                ..
            } => vec![carry, carry_tangent],
            TensorOp::ForiVjp {
                carry,
                output_cotangent,
                ..
            } => vec![carry, output_cotangent],
            TensorOp::ForiVjpJvp {
                carry,
                carry_tangent,
                output_cotangent,
                output_cotangent_tangent,
                ..
            } => vec![
                carry,
                carry_tangent,
                output_cotangent,
                output_cotangent_tangent,
            ],
            TensorOp::ScanVjp {
                carry,
                final_carry_cotangent,
                output_cotangent,
                ..
            } => vec![carry, final_carry_cotangent, output_cotangent],
            TensorOp::ScanVjpJvp {
                carry,
                carry_tangent,
                final_carry_cotangent,
                final_carry_cotangent_tangent,
                output_cotangent,
                output_cotangent_tangent,
                ..
            } => vec![
                carry,
                carry_tangent,
                final_carry_cotangent,
                final_carry_cotangent_tangent,
                output_cotangent,
                output_cotangent_tangent,
            ],
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// The named operands bound to the body inputs of the same name.
    pub(super) fn captures(self) -> &'a [(String, TensorNodeId)] {
        match self.op {
            TensorOp::Cond { captures, .. }
            | TensorOp::While { captures, .. }
            | TensorOp::Fori { captures, .. }
            | TensorOp::ForiJvp { captures, .. }
            | TensorOp::ForiVjp { captures, .. }
            | TensorOp::ForiVjpJvp { captures, .. }
            | TensorOp::Scan { captures, .. }
            | TensorOp::ScanVjp { captures, .. }
            | TensorOp::ScanVjpJvp { captures, .. } => captures,
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// The tangents of the captures, for the forward-mode kinds; empty for
    /// the others.
    pub(super) fn tangent_captures(self) -> &'a [(String, TensorNodeId)] {
        match self.op {
            TensorOp::ForiJvp {
                tangent_captures, ..
            }
            | TensorOp::ForiVjpJvp {
                tangent_captures, ..
            }
            | TensorOp::ScanVjpJvp {
                tangent_captures, ..
            } => tangent_captures,
            _ => &[],
        }
    }

    /// Every operand: the slots, then the captures, then the tangent
    /// captures.
    pub(super) fn operands(self) -> Vec<TensorNodeId> {
        let mut operands = self.slots();
        operands.extend(self.captures().iter().map(|(_, node_id)| *node_id));
        operands.extend(self.tangent_captures().iter().map(|(_, node_id)| *node_id));
        operands
    }

    /// The execution group of a node whose sibling nodes share one region
    /// execution (one result each): the node id of the group's first
    /// member. `None` for the single-result kinds.
    pub(super) fn group(self) -> Option<usize> {
        match self.op {
            TensorOp::ForiVjp { group, .. }
            | TensorOp::ForiVjpJvp { group, .. }
            | TensorOp::Scan { group, .. }
            | TensorOp::ScanVjp { group, .. }
            | TensorOp::ScanVjpJvp { group, .. } => Some(*group),
            _ => None,
        }
    }

    /// The traced body regions, in execution order: both branches of a
    /// `Cond`, the predicate and body of a `While`, the body of a loop.
    pub(super) fn regions(self) -> Vec<&'a TensorExecutionPlan> {
        match self.op {
            TensorOp::Cond { branches, .. } => {
                vec![&branches.on_true.plan, &branches.on_false.plan]
            }
            TensorOp::While { loop_plan, .. } => {
                vec![&loop_plan.predicate.plan, &loop_plan.body.plan]
            }
            TensorOp::Fori { loop_plan, .. }
            | TensorOp::ForiJvp { loop_plan, .. }
            | TensorOp::ForiVjp { loop_plan, .. } => vec![&loop_plan.body.plan],
            TensorOp::ForiVjpJvp { plan, .. } => vec![&plan.loop_plan.body.plan],
            TensorOp::Scan { scan_plan, .. } | TensorOp::ScanVjp { scan_plan, .. } => {
                vec![&scan_plan.body.plan]
            }
            TensorOp::ScanVjpJvp { plan, .. } => vec![&plan.scan_plan.body.plan],
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// The per-gradient tangent plans that a forward-over-reverse node
    /// compiles from its body when it is built; empty for the other kinds.
    /// The derivative plans that loops compile lazily on first use
    /// (`TensorLoopDerivatives`) and the carry JVP of `ScanVjpJvp` are not
    /// included.
    pub(super) fn derived_regions(self) -> Vec<&'a TensorExecutionPlan> {
        match self.op {
            TensorOp::ForiVjpJvp { plan, .. } => plan.gradient_tangent_plans.values().collect(),
            TensorOp::ScanVjpJvp { plan, .. } => plan
                .carry_gradient_tangent_plans
                .values()
                .chain(plan.output_gradient_tangent_plans.values())
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Mutable references to every operand of a region `op`, in
/// [`RegionView::operands`] order; `None` when it is not a region node.
pub(super) fn region_operands_mut(op: &mut TensorOp) -> Option<Vec<&mut TensorNodeId>> {
    let (slots, captures, tangent_captures): (Vec<&mut TensorNodeId>, _, _) = match op {
        TensorOp::Cond {
            predicate,
            captures,
            ..
        } => (vec![predicate], captures, None),
        TensorOp::While {
            carry, captures, ..
        }
        | TensorOp::Fori {
            carry, captures, ..
        }
        | TensorOp::Scan {
            carry, captures, ..
        } => (vec![carry], captures, None),
        TensorOp::ForiJvp {
            carry,
            carry_tangent,
            captures,
            tangent_captures,
            ..
        } => (vec![carry, carry_tangent], captures, Some(tangent_captures)),
        TensorOp::ForiVjp {
            carry,
            output_cotangent,
            captures,
            ..
        } => (vec![carry, output_cotangent], captures, None),
        TensorOp::ForiVjpJvp {
            carry,
            carry_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures,
            tangent_captures,
            ..
        } => (
            vec![
                carry,
                carry_tangent,
                output_cotangent,
                output_cotangent_tangent,
            ],
            captures,
            Some(tangent_captures),
        ),
        TensorOp::ScanVjp {
            carry,
            final_carry_cotangent,
            output_cotangent,
            captures,
            ..
        } => (
            vec![carry, final_carry_cotangent, output_cotangent],
            captures,
            None,
        ),
        TensorOp::ScanVjpJvp {
            carry,
            carry_tangent,
            final_carry_cotangent,
            final_carry_cotangent_tangent,
            output_cotangent,
            output_cotangent_tangent,
            captures,
            tangent_captures,
            ..
        } => (
            vec![
                carry,
                carry_tangent,
                final_carry_cotangent,
                final_carry_cotangent_tangent,
                output_cotangent,
                output_cotangent_tangent,
            ],
            captures,
            Some(tangent_captures),
        ),
        _ => return None,
    };
    let mut operands = slots;
    operands.extend(captures.iter_mut().map(|(_, node_id)| node_id));
    operands.extend(
        tangent_captures
            .into_iter()
            .flatten()
            .map(|(_, node_id)| node_id),
    );
    Some(operands)
}

/// The group of a multi-result node, mutably: a region execution group
/// ([`RegionView::group`]) or the call group of a `Custom` node. Both are the
/// node id of the group's first member, so splicing renumbers them alike.
/// Only region groups are executed; plan compilation aliases `Custom`
/// nodes, so the evaluators' group caches never see a call group.
pub(super) fn group_mut(op: &mut TensorOp) -> Option<&mut usize> {
    match op {
        TensorOp::ForiVjp { group, .. }
        | TensorOp::ForiVjpJvp { group, .. }
        | TensorOp::Scan { group, .. }
        | TensorOp::ScanVjp { group, .. }
        | TensorOp::ScanVjpJvp { group, .. }
        | TensorOp::Custom { group, .. } => Some(group),
        _ => None,
    }
}
