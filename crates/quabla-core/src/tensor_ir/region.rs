//! The control-flow region node, `TensorOp::Region`.
//!
//! A region node (`cond`, `while_loop`, `fori_loop`, `scan`, and the
//! derivative nodes of the fixed-bound loops) does not compute its result
//! from its operands with one primitive. It binds its operands by name to
//! the inputs of one or more frozen body regions and runs those bodies. Its
//! operands are a few fixed slots (the predicate, or the carry with its
//! tangent and cotangents), then the named captures, then the named tangent
//! captures.
//!
//! [`RegionNode`] holds what every kind shares (the captures) beside the
//! kind-specific slots, plans, and result selection of [`RegionKind`], and
//! answers the questions that do not depend on the kind: the operands, the
//! group, the op names, and the frozen plans. The derivative rules, the
//! evaluators, and the batching rules match on the kind.

use super::{
    TensorCondExecutionPlan, TensorExecutionPlan, TensorForiExecutionPlan,
    TensorForiVjpJvpExecutionPlan, TensorForiVjpTarget, TensorNodeId, TensorScanExecutionPlan,
    TensorScanTarget, TensorScanVjpJvpExecutionPlan, TensorScanVjpTarget, TensorWhileExecutionPlan,
};

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

    /// Whether the node is a loop (every kind but `Cond`).
    #[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
    pub(super) fn is_loop(&self) -> bool {
        !matches!(self.kind, RegionKind::Cond { .. })
    }

    /// Every operand: the slots in role order, then the captures, then the
    /// tangent captures.
    pub(super) fn operands(&self) -> Vec<TensorNodeId> {
        let mut operands = match self.kind {
            RegionKind::Cond { predicate, .. } => vec![predicate],
            RegionKind::While { carry, .. }
            | RegionKind::Fori { carry, .. }
            | RegionKind::Scan { carry, .. } => vec![carry],
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
        operands.extend(self.captures.iter().map(|(_, node_id)| *node_id));
        operands.extend(self.tangent_captures.iter().map(|(_, node_id)| *node_id));
        operands
    }

    /// Mutable references to every operand, in [`Self::operands`] order.
    pub(super) fn operands_mut(&mut self) -> Vec<&mut TensorNodeId> {
        let mut operands = match &mut self.kind {
            RegionKind::Cond { predicate, .. } => vec![predicate],
            RegionKind::While { carry, .. }
            | RegionKind::Fori { carry, .. }
            | RegionKind::Scan { carry, .. } => vec![carry],
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
            RegionKind::ForiVjp { group, .. }
            | RegionKind::ForiVjpJvp { group, .. }
            | RegionKind::Scan { group, .. }
            | RegionKind::ScanVjp { group, .. }
            | RegionKind::ScanVjpJvp { group, .. } => Some(group),
            RegionKind::Cond { .. }
            | RegionKind::While { .. }
            | RegionKind::Fori { .. }
            | RegionKind::ForiJvp { .. } => None,
        }
    }

    /// The group, mutably; see [`Self::group`].
    pub(super) fn group_mut(&mut self) -> Option<&mut usize> {
        match &mut self.kind {
            RegionKind::ForiVjp { group, .. }
            | RegionKind::ForiVjpJvp { group, .. }
            | RegionKind::Scan { group, .. }
            | RegionKind::ScanVjp { group, .. }
            | RegionKind::ScanVjpJvp { group, .. } => Some(group),
            RegionKind::Cond { .. }
            | RegionKind::While { .. }
            | RegionKind::Fori { .. }
            | RegionKind::ForiJvp { .. } => None,
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
            RegionKind::Fori { loop_plan, .. }
            | RegionKind::ForiJvp { loop_plan, .. }
            | RegionKind::ForiVjp { loop_plan, .. } => vec![&loop_plan.body.plan],
            RegionKind::ForiVjpJvp { plan, .. } => vec![&plan.loop_plan.body.plan],
            RegionKind::Scan { scan_plan, .. } | RegionKind::ScanVjp { scan_plan, .. } => {
                vec![&scan_plan.body.plan]
            }
            RegionKind::ScanVjpJvp { plan, .. } => vec![&plan.scan_plan.body.plan],
        }
    }

    /// The `lower_text` line of the node with id `id` and tensor type `ty`.
    /// The per-kind formats are part of the printed IR and stay fixed.
    pub(super) fn lower_text(&self, id: TensorNodeId, ty: &str) -> String {
        let captures = &self.captures;
        let tangent_captures = &self.tangent_captures;
        match &self.kind {
            RegionKind::Cond {
                predicate,
                branches,
            } => format!(
                "%{id} = cond(%{predicate}, captures={captures:?}, true_nodes={}, \
                 false_nodes={}) : {ty}",
                branches.true_node_count(),
                branches.false_node_count(),
            ),
            RegionKind::While { carry, loop_plan } => format!(
                "%{id} = while(carry=%{carry}, captures={captures:?}, predicate_nodes={}, \
                 body_nodes={}) : {ty}",
                loop_plan.predicate.plan.node_count(),
                loop_plan.body.plan.node_count(),
            ),
            RegionKind::Fori { carry, loop_plan } => format!(
                "%{id} = fori(carry=%{carry}, lower={}, upper={}, captures={captures:?}, \
                 body_nodes={}) : {ty}",
                loop_plan.lower,
                loop_plan.upper,
                loop_plan.body.plan.node_count(),
            ),
            RegionKind::ForiJvp {
                carry,
                carry_tangent,
                loop_plan,
            } => format!(
                "%{id} = fori_jvp(carry=%{carry}, carry_tangent=%{carry_tangent}, lower={}, \
                 upper={}, captures={captures:?}, tangent_captures={tangent_captures:?}, \
                 body_nodes={}) : {ty}",
                loop_plan.lower,
                loop_plan.upper,
                loop_plan.body.plan.node_count(),
            ),
            RegionKind::ForiVjp { target, group, .. } => {
                format!("%{id} = fori_vjp(group={group}, target={target:?}) : {ty}")
            }
            RegionKind::ForiVjpJvp { target, group, .. } => {
                format!("%{id} = fori_vjp_jvp(group={group}, target={target:?}) : {ty}")
            }
            RegionKind::Scan {
                scan_plan,
                target,
                group,
                ..
            } => format!(
                "%{id} = scan(group={group}, target={target:?}, lower={}, upper={}, \
                 captures={captures:?}, body_nodes={}) : {ty}",
                scan_plan.lower,
                scan_plan.upper,
                scan_plan.body.plan.node_count(),
            ),
            RegionKind::ScanVjp { target, group, .. } => {
                format!("%{id} = scan_vjp(group={group}, target={target:?}) : {ty}")
            }
            RegionKind::ScanVjpJvp { target, group, .. } => {
                format!("%{id} = scan_vjp_jvp(group={group}, target={target:?}) : {ty}")
            }
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
            RegionKind::Cond { .. }
            | RegionKind::While { .. }
            | RegionKind::Fori { .. }
            | RegionKind::ForiJvp { .. }
            | RegionKind::ForiVjp { .. }
            | RegionKind::Scan { .. }
            | RegionKind::ScanVjp { .. } => Vec::new(),
        }
    }
}
