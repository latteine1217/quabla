//! What every control-flow region node has in common.
//!
//! A region node (`Cond`, `While`, `Fori`, `Scan`, and the derivative nodes
//! of the fixed-bound loops) does not compute its result from its operands
//! with one primitive. It binds its operands by name to the inputs of one or
//! more frozen body regions and runs those bodies. Its operands are a few
//! fixed slots (the predicate, or the carry with its tangent and
//! cotangents), then the named captures, then the named tangent captures.
//!
//! [`RegionNode`] holds what every kind shares (the captures) beside the
//! kind-specific slots, plans, and result selection of [`RegionKind`], and
//! answers the questions that do not depend on the kind: the operands, the
//! group, the op names, and the frozen plans. The derivative rules, the
//! evaluators, and the batching rules match on the kind.

use super::{
    TensorCondExecutionPlan, TensorExecutionPlan, TensorForiExecutionPlan,
    TensorForiVjpJvpExecutionPlan, TensorForiVjpTarget, TensorNodeId, TensorOp,
    TensorScanExecutionPlan, TensorScanTarget, TensorScanVjpJvpExecutionPlan, TensorScanVjpTarget,
    TensorWhileExecutionPlan,
};

/// Matches every region node of [`TensorOp`].
macro_rules! region_op {
    () => {
        TensorOp::Region(_)
    };
}
pub(super) use region_op;

/// A control-flow region node: its kind, with the kind's operand slots,
/// plans, and result selection, and the named operands it binds to the body
/// inputs of the same name.
#[derive(Clone, Debug)]
pub(super) struct RegionNode {
    pub(super) kind: RegionKind,
    /// The named operands bound to the body inputs of the same name.
    pub(super) captures: Vec<(String, TensorNodeId)>,
    /// The tangents of the captures, for the forward-mode kinds (`ForiJvp`,
    /// `ForiVjpJvp`, `ScanVjpJvp`); empty for the others.
    pub(super) tangent_captures: Vec<(String, TensorNodeId)>,
}

/// The kind of a [`RegionNode`] with its operand slots (in role order: the
/// predicate, or the carry followed in a forward mode by its tangent and in
/// a reverse mode by the cotangents, each followed by its tangent in
/// forward-over-reverse), its frozen plan, and, for the kinds whose sibling
/// nodes share one execution, the selected result and the group.
#[derive(Clone, Debug)]
pub(super) enum RegionKind {
    /// A scalar-predicate lazy branch. The regions own their input captures
    /// and are evaluated only after the predicate has been materialized.
    Cond {
        predicate: TensorNodeId,
        branches: TensorCondExecutionPlan,
    },
    /// A data-dependent loop: the body runs while the predicate region
    /// returns true. Only the final carry is produced.
    While {
        carry: TensorNodeId,
        loop_plan: TensorWhileExecutionPlan,
    },
    /// A fixed-bound loop; only the final carry is produced.
    Fori {
        carry: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
    },
    /// Tangent result of a fixed-bound `Fori`. The matching primal remains an
    /// ordinary `Fori`, avoiding the packed slice/concat carry representation.
    ForiJvp {
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
    },
    /// One selected result of a shared fixed-bound `Fori` reverse pass.
    ForiVjp {
        carry: TensorNodeId,
        output_cotangent: TensorNodeId,
        loop_plan: TensorForiExecutionPlan,
        target: TensorForiVjpTarget,
        group: usize,
    },
    /// One selected directional derivative of a shared fixed-bound `Fori`
    /// reverse pass. This is the structural forward-over-reverse rule used by
    /// compiled HVP and Hessian transforms.
    ForiVjpJvp {
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        output_cotangent: TensorNodeId,
        output_cotangent_tangent: TensorNodeId,
        plan: TensorForiVjpJvpExecutionPlan,
        target: TensorForiVjpTarget,
        group: usize,
    },
    /// One selected result of a shared fixed-bound `Scan` execution.
    Scan {
        carry: TensorNodeId,
        scan_plan: TensorScanExecutionPlan,
        target: TensorScanTarget,
        group: usize,
    },
    /// One selected result of a shared fixed-bound `Scan` reverse pass.
    ScanVjp {
        carry: TensorNodeId,
        final_carry_cotangent: TensorNodeId,
        output_cotangent: TensorNodeId,
        scan_plan: TensorScanExecutionPlan,
        target: TensorScanVjpTarget,
        group: usize,
    },
    /// One selected directional derivative of a shared fixed-bound `Scan`
    /// reverse pass. Both the final-carry and stacked-output cotangents are
    /// differentiated together, preserving the joint reverse semantics.
    ScanVjpJvp {
        carry: TensorNodeId,
        carry_tangent: TensorNodeId,
        final_carry_cotangent: TensorNodeId,
        final_carry_cotangent_tangent: TensorNodeId,
        output_cotangent: TensorNodeId,
        output_cotangent_tangent: TensorNodeId,
        plan: TensorScanVjpJvpExecutionPlan,
        target: TensorScanVjpTarget,
        group: usize,
    },
}

impl RegionNode {
    /// The IR op name, as `lower_text` and error messages print it.
    pub(super) fn name(&self) -> &'static str {
        match self.kind {
            RegionKind::Cond { .. } => "cond",
            RegionKind::While { .. } => "while",
            RegionKind::Fori { .. } => "fori",
            RegionKind::ForiJvp { .. } => "fori_jvp",
            RegionKind::ForiVjp { .. } => "fori_vjp",
            RegionKind::ForiVjpJvp { .. } => "fori_vjp_jvp",
            RegionKind::Scan { .. } => "scan",
            RegionKind::ScanVjp { .. } => "scan_vjp",
            RegionKind::ScanVjpJvp { .. } => "scan_vjp_jvp",
        }
    }

    /// The node kind as prose ("Fori VJP JVP"), for diagnostics.
    pub(super) fn label(&self) -> &'static str {
        match self.kind {
            RegionKind::Cond { .. } => "Cond",
            RegionKind::While { .. } => "While",
            RegionKind::Fori { .. } => "Fori",
            RegionKind::ForiJvp { .. } => "Fori JVP",
            RegionKind::ForiVjp { .. } => "Fori VJP",
            RegionKind::ForiVjpJvp { .. } => "Fori VJP JVP",
            RegionKind::Scan { .. } => "Scan",
            RegionKind::ScanVjp { .. } => "Scan VJP",
            RegionKind::ScanVjpJvp { .. } => "Scan VJP JVP",
        }
    }

    /// The fixed operand slots in role order.
    fn slots(&self) -> Vec<TensorNodeId> {
        match self.kind {
            RegionKind::Cond { predicate, .. } => vec![predicate],
            RegionKind::While { carry, .. } => vec![carry],
            RegionKind::Fori { carry, .. } => vec![carry],
            RegionKind::ForiJvp {
                carry,
                carry_tangent,
                ..
            } => vec![carry, carry_tangent],
            RegionKind::ForiVjp {
                carry,
                output_cotangent,
                ..
            } => vec![carry, output_cotangent],
            RegionKind::ForiVjpJvp {
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
            RegionKind::Scan { carry, .. } => vec![carry],
            RegionKind::ScanVjp {
                carry,
                final_carry_cotangent,
                output_cotangent,
                ..
            } => vec![carry, final_carry_cotangent, output_cotangent],
            RegionKind::ScanVjpJvp {
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
        }
    }

    /// Every operand: the slots, then the captures, then the tangent
    /// captures.
    pub(super) fn operands(&self) -> Vec<TensorNodeId> {
        let mut operands = self.slots();
        operands.extend(self.captures.iter().map(|(_, node_id)| *node_id));
        operands.extend(self.tangent_captures.iter().map(|(_, node_id)| *node_id));
        operands
    }

    /// Mutable references to every operand, in [`Self::operands`] order.
    pub(super) fn operands_mut(&mut self) -> Vec<&mut TensorNodeId> {
        let mut operands: Vec<&mut TensorNodeId> = match &mut self.kind {
            RegionKind::Cond { predicate, .. } => vec![predicate],
            RegionKind::While { carry, .. } => vec![carry],
            RegionKind::Fori { carry, .. } => vec![carry],
            RegionKind::ForiJvp {
                carry,
                carry_tangent,
                ..
            } => vec![carry, carry_tangent],
            RegionKind::ForiVjp {
                carry,
                output_cotangent,
                ..
            } => vec![carry, output_cotangent],
            RegionKind::ForiVjpJvp {
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
            RegionKind::Scan { carry, .. } => vec![carry],
            RegionKind::ScanVjp {
                carry,
                final_carry_cotangent,
                output_cotangent,
                ..
            } => vec![carry, final_carry_cotangent, output_cotangent],
            RegionKind::ScanVjpJvp {
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
        };
        operands.extend(self.captures.iter_mut().map(|(_, node_id)| node_id));
        operands.extend(self.tangent_captures.iter_mut().map(|(_, node_id)| node_id));
        operands
    }

    /// The execution group of a node whose sibling nodes share one region
    /// execution (one result each): the node id of the group's first
    /// member. `None` for the single-result kinds.
    pub(super) fn group(&self) -> Option<usize> {
        match self.kind {
            RegionKind::ForiVjp { group, .. } => Some(group),
            RegionKind::ForiVjpJvp { group, .. } => Some(group),
            RegionKind::Scan { group, .. } => Some(group),
            RegionKind::ScanVjp { group, .. } => Some(group),
            RegionKind::ScanVjpJvp { group, .. } => Some(group),
            _ => None,
        }
    }

    /// The group, mutably; see [`Self::group`].
    pub(super) fn group_mut(&mut self) -> Option<&mut usize> {
        match &mut self.kind {
            RegionKind::ForiVjp { group, .. } => Some(group),
            RegionKind::ForiVjpJvp { group, .. } => Some(group),
            RegionKind::Scan { group, .. } => Some(group),
            RegionKind::ScanVjp { group, .. } => Some(group),
            RegionKind::ScanVjpJvp { group, .. } => Some(group),
            _ => None,
        }
    }

    /// The traced body regions, in execution order: both branches of a
    /// `Cond`, the predicate and body of a `While`, the body of a loop.
    pub(super) fn regions(&self) -> Vec<&TensorExecutionPlan> {
        match &self.kind {
            RegionKind::Cond { branches, .. } => {
                vec![&branches.on_true.plan, &branches.on_false.plan]
            }
            RegionKind::While { loop_plan, .. } => {
                vec![&loop_plan.predicate.plan, &loop_plan.body.plan]
            }
            RegionKind::Fori { loop_plan, .. } => vec![&loop_plan.body.plan],
            RegionKind::ForiJvp { loop_plan, .. } => vec![&loop_plan.body.plan],
            RegionKind::ForiVjp { loop_plan, .. } => vec![&loop_plan.body.plan],
            RegionKind::ForiVjpJvp { plan, .. } => vec![&plan.loop_plan.body.plan],
            RegionKind::Scan { scan_plan, .. } => vec![&scan_plan.body.plan],
            RegionKind::ScanVjp { scan_plan, .. } => vec![&scan_plan.body.plan],
            RegionKind::ScanVjpJvp { plan, .. } => vec![&plan.scan_plan.body.plan],
        }
    }

    /// The per-gradient tangent plans that a forward-over-reverse node
    /// compiles from its body when it is built; empty for the other kinds.
    /// The derivative plans that loops compile lazily on first use
    /// (`TensorLoopDerivatives`) and the carry JVP of `ScanVjpJvp` are not
    /// included.
    pub(super) fn derived_regions(&self) -> Vec<&TensorExecutionPlan> {
        match &self.kind {
            RegionKind::ForiVjpJvp { plan, .. } => plan.gradient_tangent_plans.values().collect(),
            RegionKind::ScanVjpJvp { plan, .. } => plan
                .carry_gradient_tangent_plans
                .values()
                .chain(plan.output_gradient_tangent_plans.values())
                .collect(),
            _ => Vec::new(),
        }
    }
}

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
            TensorOp::Region(region) => region.name(),
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// The node kind as prose ("Fori VJP JVP"), for diagnostics.
    pub(super) fn label(self) -> &'static str {
        match self.op {
            TensorOp::Region(region) => region.label(),
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// Whether the node is a loop (every kind but `Cond`).
    #[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
    pub(super) fn is_loop(self) -> bool {
        self.name() != "cond"
    }

    /// The named operands bound to the body inputs of the same name.
    #[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
    pub(super) fn captures(self) -> &'a [(String, TensorNodeId)] {
        match self.op {
            TensorOp::Region(region) => &region.captures,
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// Every operand: the slots, then the captures, then the tangent
    /// captures.
    pub(super) fn operands(self) -> Vec<TensorNodeId> {
        match self.op {
            TensorOp::Region(region) => region.operands(),
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// The execution group of a node whose sibling nodes share one region
    /// execution (one result each): the node id of the group's first
    /// member. `None` for the single-result kinds.
    pub(super) fn group(self) -> Option<usize> {
        match self.op {
            TensorOp::Region(region) => region.group(),
            _ => None,
        }
    }

    /// The traced body regions, in execution order: both branches of a
    /// `Cond`, the predicate and body of a `While`, the body of a loop.
    pub(super) fn regions(self) -> Vec<&'a TensorExecutionPlan> {
        match self.op {
            TensorOp::Region(region) => region.regions(),
            _ => unreachable!("RegionView holds a region op"),
        }
    }

    /// See [`RegionNode::derived_regions`].
    pub(super) fn derived_regions(self) -> Vec<&'a TensorExecutionPlan> {
        match self.op {
            TensorOp::Region(region) => region.derived_regions(),
            _ => Vec::new(),
        }
    }
}

/// Mutable references to every operand of a region `op`, in
/// [`RegionView::operands`] order; `None` when it is not a region node.
pub(super) fn region_operands_mut(op: &mut TensorOp) -> Option<Vec<&mut TensorNodeId>> {
    match op {
        TensorOp::Region(region) => Some(region.operands_mut()),
        _ => None,
    }
}

/// The group of a multi-result node, mutably: a region execution group
/// ([`RegionNode::group`]) or the call group of a `Custom` node. Both are the
/// node id of the group's first member, so splicing renumbers them alike.
/// Only region groups are executed; plan compilation aliases `Custom`
/// nodes, so the evaluators' group caches never see a call group.
pub(super) fn group_mut(op: &mut TensorOp) -> Option<&mut usize> {
    match op {
        TensorOp::Custom { group, .. } => Some(group),
        TensorOp::Region(region) => region.group_mut(),
        _ => None,
    }
}
