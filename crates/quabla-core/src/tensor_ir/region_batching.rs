//! Batching rules for region nodes: `vmap` of structured control flow.
//!
//! [`TensorIr::inline_batched`] batches every node whose operands depend on a
//! mapped argument. A region node does not compute its result from its
//! operands with one primitive: it binds them by name to the inputs of one or
//! more compiled body regions (`TensorRegion`, `TensorMultiRegion`) and runs
//! those bodies. Every region rule therefore follows the same four steps:
//!
//! 1. Decide which body inputs are mapped. A body input is mapped when an
//!    operand bound to it is mapped. Forward-mode nodes (`ForiJvp`) evaluate
//!    the primal and the tangent of an input with the same body, so the input
//!    is mapped when either of them is, and the other is broadcast.
//! 2. Batch the body with [`batch_region_plan`]. A body is an ordinary graph,
//!    so it is batched by `inline_batched` itself with exactly those inputs
//!    mapped (shape `[B, *shape]`) and the others unchanged; the result says
//!    which body outputs are mapped. Region nodes inside the body recurse
//!    through the same rules, so nested loops batch from the inside out.
//! 3. For a loop, close the carry under the body with [`batch_loop_body`]. A
//!    carry whose next value is mapped (it depends on a mapped capture) must
//!    be mapped from the first iteration, so the carry is marked mapped and
//!    the body batched again: the fixed point over carry batchedness that
//!    JAX computes for `scan` and `while_loop`. The loop carries one array,
//!    so it needs at most two passes. A mapped carry whose next value does
//!    not depend on a mapped input is broadcast inside the body, so the carry
//!    keeps one shape across iterations.
//! 4. Rebuild the node from the batched body with the ordinary constructors
//!    (`TensorForiExecutionPlan::new`, `TensorForiVjpJvpExecutionPlan::new`,
//!    ...), so every plan derived from the body (its VJP, its forward-over-
//!    reverse plans, the MLX reverse plans) is derived again from the batched
//!    body, and broadcast exactly the operands whose body input is mapped
//!    while the operand is not. Unmapped operands of unmapped inputs stay
//!    unbatched, so loop-invariant work is not repeated per example.
//!
//! Reverse-mode nodes (`ForiVjp`, `ForiVjpJvp`, `ScanVjp`, `ScanVjpJvp`) map
//! every body input once any operand is mapped. Their results are per-example
//! gradients with respect to the inputs, and the VJP of a batched body with
//! respect to an unmapped input would be the sum of the examples' gradients.
//! The nodes of one reverse pass share a group id (one execution computes all
//! of them), so the batched members are built together, share their batched
//! operands and plan, and get one new group id ([`LoopBatchGroups`]).
//!
//! Layout follows the rest of `inline_batched`: the batch axis leads every
//! mapped value. A scan stacks its per-step outputs on a new leading time
//! axis, so the batched body's `[B, *y]` steps stack to `[T, B, *y]`; the
//! rule moves the batch axis to the front (`[B, T, *y]`) and moves it back
//! (`[T, B, *y]`) on a stacked output cotangent before a batched reverse scan
//! consumes it. `Fori` and `Scan` trip counts are static, so no example
//! needs masking.
//!
//! `Cond` has no carry and no derivative node kinds of its own (its JVP and
//! VJP are again `Cond` nodes over derived branch regions), so one rule,
//! [`TensorIr::push_batched_cond`], covers it. Under an unmapped predicate it
//! batches both branches with [`batch_region_plan`] over the same mapped
//! inputs and forces the output mapped when either branch maps it; under a
//! mapped predicate it evaluates both branches batched and selects per
//! example with `where`, as JAX does.
//!
//! `While` closes its carry with [`batch_loop_body`] and batches the
//! predicate region over the resulting mapped inputs. An unmapped predicate
//! keeps one trip count for the batch; a mapped one is reduced with `any`,
//! and the body freezes every example whose own predicate is false with a
//! per-example select, as JAX does ([`batch_while_plan`]). `While` is
//! forward-mode only, and its JVP is another `While` over a packed
//! `(primal, tangent)` carry, so this one rule batches every while-derived
//! node.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{
    batched_shape, BatchingError, RegionKind, RegionNode, TensorComparison,
    TensorCondExecutionPlan, TensorExecutionPlan, TensorForiExecutionPlan,
    TensorForiVjpJvpBindings, TensorForiVjpJvpExecutionPlan, TensorForiVjpTarget, TensorIr,
    TensorNode, TensorNodeId, TensorOp, TensorScanExecutionPlan, TensorScanTarget,
    TensorScanVjpBindings, TensorScanVjpJvpBindings, TensorScanVjpJvpExecutionPlan,
    TensorScanVjpTarget, TensorWhileExecutionPlan,
};

/// Which result of a batched loop group a member node selects.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum LoopTarget {
    Carry,
    Outputs,
    External(String),
}

/// The batched members of each multi-result loop group of one
/// `inline_batched` call, keyed by the op name and the callee group id (group
/// ids are only unique per op kind): `(node, mapped)` per member target.
pub(super) type LoopBatchGroups =
    HashMap<(&'static str, usize), HashMap<LoopTarget, (TensorNodeId, bool)>>;

/// A loop body batched by [`batch_loop_body`].
struct BatchedBody {
    plan: TensorExecutionPlan,
    /// The body inputs that have the leading batch axis.
    mapped_inputs: BTreeSet<String>,
    /// Whether each body output has the leading batch axis.
    mapped_outputs: Vec<bool>,
}

impl BatchedBody {
    fn maps(&self, name: &str) -> bool {
        self.mapped_inputs.contains(name)
    }
}

/// `region` vectorized over `batch_size` examples with the inputs named in
/// `mapped_inputs` mapped: a mapped input has shape `[batch_size, *shape]`,
/// the others keep their shape. The region keeps its input names, so the
/// node that owns it binds the same operands. Output `i` is broadcast over
/// the batch when `force_mapped[i]` is set and it does not depend on a mapped
/// input. Returns the batched region and whether each output is mapped.
pub(super) fn batch_region_plan(
    region: &TensorExecutionPlan,
    mapped_inputs: &BTreeSet<String>,
    batch_size: usize,
    force_mapped: &[bool],
) -> Result<(TensorExecutionPlan, Vec<bool>), BatchingError> {
    let callee = region.as_ir();
    let mut graph = TensorIr::new();
    let mut bindings = BTreeMap::new();
    for node in callee.nodes.iter() {
        if let TensorOp::Input { name } = &node.op {
            let mapped = mapped_inputs.contains(name);
            let shape = if mapped {
                batched_shape(batch_size, &node.shape)
            } else {
                node.shape.clone()
            };
            let input = graph.input_typed(name.clone(), shape, node.dtype)?;
            bindings.insert(name.clone(), (input, mapped));
        }
    }
    let spliced = graph.inline_batched(&callee, &bindings, batch_size, &region.output_node_ids)?;
    let mut outputs = Vec::with_capacity(spliced.len());
    let mut mapped_outputs = Vec::with_capacity(spliced.len());
    for (index, (mut output, mut mapped)) in spliced.into_iter().enumerate() {
        if !mapped && force_mapped.get(index).copied().unwrap_or(false) {
            let shape = batched_shape(batch_size, &graph.node(output)?.shape);
            output = graph.broadcast_to(output, shape)?;
            mapped = true;
        }
        outputs.push(output);
        mapped_outputs.push(mapped);
    }
    let (plan, _) = graph.compile_cpu_many(&outputs)?;
    // Every input of a compiled region is reachable from its outputs, and a
    // batched node depends on the batched forms of the same operands, so the
    // region interface survives batching; check it rather than bind a
    // different capture set.
    if plan.input_nodes.len() != region.input_nodes.len()
        || region
            .input_nodes
            .keys()
            .any(|name| !plan.input_nodes.contains_key(name))
    {
        return Err(BatchingError::Invalid(
            "vmap changed the inputs of a region body while batching it".to_string(),
        ));
    }
    Ok((plan, mapped_outputs))
}

/// A loop body (output 0 is the next carry) batched with the inputs in
/// `mapped_inputs` mapped, closed under the carry fixed point: when the next
/// carry depends on a mapped input, the carry input `carry_name` is mapped
/// too and the body batched again. A mapped carry always yields a mapped next
/// carry (broadcast if needed); `map_all_outputs` also forces every other
/// output mapped (the reverse-mode rules, whose cotangents are batched).
fn batch_loop_body(
    body: &TensorExecutionPlan,
    carry_name: &str,
    mut mapped_inputs: BTreeSet<String>,
    batch_size: usize,
    map_all_outputs: bool,
) -> Result<BatchedBody, BatchingError> {
    loop {
        let carry_mapped = mapped_inputs.contains(carry_name);
        let mut force = vec![map_all_outputs; body.output_node_ids.len()];
        force[0] = carry_mapped;
        let (plan, mapped_outputs) = batch_region_plan(body, &mapped_inputs, batch_size, &force)?;
        if mapped_outputs[0] && !carry_mapped {
            mapped_inputs.insert(carry_name.to_string());
            continue;
        }
        return Ok(BatchedBody {
            plan,
            mapped_inputs,
            mapped_outputs,
        });
    }
}

/// Every input of a loop body except the scalar loop index, which is the same
/// for every example.
fn body_inputs_except(body: &TensorExecutionPlan, index_name: &str) -> BTreeSet<String> {
    body.input_nodes
        .keys()
        .filter(|name| name.as_str() != index_name)
        .cloned()
        .collect()
}

/// A `while_loop` batched with the inputs in `mapped_inputs` mapped, with
/// its batched body, whose mapped inputs are the loop's (carry included).
///
/// The body is closed under the carry fixed point by [`batch_loop_body`] and
/// the predicate is batched over the resulting mapped inputs. An unmapped
/// predicate reads only unmapped values, so every example has the same trip
/// count and the loop is the batched body under the unchanged predicate. A
/// mapped predicate gives each example its own trip count; as `jax.vmap` of
/// `lax.while_loop` does, the batched loop then runs while any example's
/// predicate holds and selects `where(pred, body(carry), carry)` per example
/// ([`masked_while_plan`]). That select makes the next carry depend on the
/// predicate, so a mapped predicate maps the carry: the second fixed point,
/// after which the predicate stays mapped (mapping more inputs never unmaps
/// a value), so this needs at most one more pass.
fn batch_while_plan(
    loop_plan: &TensorWhileExecutionPlan,
    mut mapped_inputs: BTreeSet<String>,
    batch_size: usize,
) -> Result<(TensorWhileExecutionPlan, BatchedBody), BatchingError> {
    let carry_name = loop_plan.carry_name();
    loop {
        let body = batch_loop_body(
            loop_plan.body_plan(),
            carry_name,
            mapped_inputs,
            batch_size,
            false,
        )?;
        let (predicate, predicate_mapped) = batch_region_plan(
            loop_plan.predicate_plan(),
            &body.mapped_inputs,
            batch_size,
            &[false],
        )?;
        if predicate_mapped[0] && !body.maps(carry_name) {
            mapped_inputs = body.mapped_inputs;
            mapped_inputs.insert(carry_name.to_string());
            continue;
        }
        let plan = if predicate_mapped[0] {
            masked_while_plan(&predicate, &body.plan, carry_name, batch_size)?
        } else {
            TensorWhileExecutionPlan::new(predicate, body.plan.clone(), carry_name)?
        };
        return Ok((plan, body));
    }
}

/// The loop of a batched while with a mapped predicate (`predicate` returns
/// `[B]` bools, `body` the mapped next carry `[B, *carry]`): the new
/// predicate is `any(predicate)`, and the new body recomputes the per-example
/// predicate and selects `where(predicate, body(carry), carry)`. An example
/// whose predicate is false keeps its carry bit for bit while the others
/// iterate, so its result is the one an unbatched loop stops at; whatever
/// the body computes for it meanwhile (NaN or inf from a converged state,
/// such as `0 / 0`) is discarded by the select and never reaches the carry
/// or, through the select's tangent rule, a forward-mode tangent. Recomputing
/// the predicate in the body costs one more predicate evaluation per
/// iteration, as in JAX; the predicate stays a scalar region, so every
/// backend still reads back one flag per iteration.
fn masked_while_plan(
    predicate: &TensorExecutionPlan,
    body: &TensorExecutionPlan,
    carry_name: &str,
    batch_size: usize,
) -> Result<TensorWhileExecutionPlan, BatchingError> {
    let mut graph = TensorIr::new();
    let mut inputs = BTreeMap::new();
    for region in [predicate, body] {
        for (name, &id) in region.input_nodes.iter() {
            if !inputs.contains_key(name) {
                let node = &region.nodes[id];
                let input = graph.input_typed(name.clone(), node.shape.clone(), node.dtype)?;
                inputs.insert(name.clone(), input);
            }
        }
    }
    let splice = |graph: &mut TensorIr, region: &TensorExecutionPlan| {
        let bindings = region
            .input_nodes
            .keys()
            .map(|name| (name.clone(), inputs[name]))
            .collect::<BTreeMap<_, _>>();
        graph
            .inline(&region.as_ir(), &bindings, &[region.output_node_id])
            .map(|spliced| spliced[0])
    };
    let running = splice(&mut graph, predicate)?;
    let next = splice(&mut graph, body)?;
    let carry = *inputs.get(carry_name).ok_or_else(|| {
        BatchingError::Invalid(format!("batched while body lost its carry {carry_name:?}"))
    })?;
    // `[B]` -> `[B, 1, ...]`: broadcasting aligns trailing axes, so the batch
    // axis of the mask must be made to lead explicitly.
    let mut mask_shape = vec![1; graph.node(carry)?.shape.len()];
    mask_shape[0] = batch_size;
    let mask = graph.reshape(running, mask_shape)?;
    let selected = graph.where_select(mask, next, carry)?;
    let any_running = graph.any(running)?;
    Ok(TensorWhileExecutionPlan::new(
        graph.compile_cpu(any_running)?,
        graph.compile_cpu(selected)?,
        carry_name,
    )?)
}

impl TensorForiExecutionPlan {
    /// This loop with its body replaced by a batched form of it.
    fn with_body(&self, body: TensorExecutionPlan) -> Result<Self, String> {
        Self::new(
            self.lower,
            self.upper,
            body,
            self.carry_name.clone(),
            self.index_name.clone(),
        )
    }
}

impl TensorScanExecutionPlan {
    /// This scan with its body replaced by a batched form of it.
    fn with_body(&self, body: TensorExecutionPlan) -> Result<Self, String> {
        Self::new(
            self.lower,
            self.upper,
            body,
            self.carry_name.clone(),
            self.index_name.clone(),
        )
    }
}

impl TensorIr {
    /// Appends the batched form of a loop region node (`Fori`, `Scan`, and
    /// their derivative nodes) with at least one mapped operand; see the
    /// module documentation. Returns the node and whether it is mapped: a
    /// loop result that does not depend on a mapped operand stays unmapped.
    pub(super) fn push_batched_loop(
        &mut self,
        node: &TensorNode,
        (remap, mapped): (&HashMap<TensorNodeId, TensorNodeId>, &[bool]),
        batch_size: usize,
        groups: &mut LoopBatchGroups,
    ) -> Result<(TensorNodeId, bool), BatchingError> {
        let operand = |id: TensorNodeId| -> Result<(TensorNodeId, bool), BatchingError> {
            remap
                .get(&id)
                .map(|spliced| (*spliced, mapped[id]))
                .ok_or_else(|| {
                    BatchingError::Invalid(format!("node {id} is missing from the vmap remap"))
                })
        };
        let mapped_names = |bound: &[(String, TensorNodeId)]| {
            bound
                .iter()
                .filter(|(_, id)| mapped[*id])
                .map(|(name, _)| name.clone())
                .collect::<BTreeSet<_>>()
        };
        match &node.op {
            TensorOp::Fori {
                carry,
                loop_plan,
                captures,
            } => {
                let mut inputs = mapped_names(captures);
                if mapped[*carry] {
                    inputs.insert(loop_plan.carry_name.clone());
                }
                let body = batch_loop_body(
                    &loop_plan.body.plan,
                    &loop_plan.carry_name,
                    inputs,
                    batch_size,
                    false,
                )?;
                let carry_mapped = body.maps(&loop_plan.carry_name);
                let carry = self.bind_batched(operand(*carry)?, carry_mapped, batch_size)?;
                let captures = self.bind_batched_captures(captures, &body, &operand, batch_size)?;
                let plan = loop_plan.with_body(body.plan)?;
                Ok((self.fori(carry, plan, captures)?, carry_mapped))
            }
            TensorOp::ForiJvp {
                carry,
                carry_tangent,
                loop_plan,
                captures,
                tangent_captures,
            } => {
                // One body evaluates both the primal and the tangent of each
                // input, so an input is mapped when either of them is.
                let mut inputs = mapped_names(captures);
                inputs.extend(mapped_names(tangent_captures));
                if mapped[*carry] || mapped[*carry_tangent] {
                    inputs.insert(loop_plan.carry_name.clone());
                }
                let body = batch_loop_body(
                    &loop_plan.body.plan,
                    &loop_plan.carry_name,
                    inputs,
                    batch_size,
                    false,
                )?;
                let carry_mapped = body.maps(&loop_plan.carry_name);
                let carry = self.bind_batched(operand(*carry)?, carry_mapped, batch_size)?;
                let carry_tangent =
                    self.bind_batched(operand(*carry_tangent)?, carry_mapped, batch_size)?;
                let captures = self.bind_batched_captures(captures, &body, &operand, batch_size)?;
                let tangent_captures =
                    self.bind_batched_captures(tangent_captures, &body, &operand, batch_size)?;
                let plan = loop_plan.with_body(body.plan)?;
                let tangent =
                    self.fori_jvp(carry, carry_tangent, plan, captures, tangent_captures)?;
                Ok((tangent, carry_mapped))
            }
            TensorOp::ForiVjp {
                carry,
                output_cotangent,
                loop_plan,
                captures,
                target,
                group,
            } => self.batched_group_member(
                groups,
                ("fori_vjp", *group),
                fori_target(target),
                |graph| {
                    let body = batch_loop_body(
                        &loop_plan.body.plan,
                        &loop_plan.carry_name,
                        body_inputs_except(&loop_plan.body.plan, &loop_plan.index_name),
                        batch_size,
                        true,
                    )?;
                    let carry = graph.bind_batched(operand(*carry)?, true, batch_size)?;
                    let cotangent =
                        graph.bind_batched(operand(*output_cotangent)?, true, batch_size)?;
                    let captures =
                        graph.bind_batched_captures(captures, &body, &operand, batch_size)?;
                    let plan = loop_plan.with_body(body.plan)?;
                    let batched_group = graph.nodes.len();
                    let mut members = HashMap::new();
                    for target in fori_targets(&captures) {
                        let member = graph.fori_vjp(
                            carry,
                            cotangent,
                            plan.clone(),
                            captures.clone(),
                            target.clone(),
                            batched_group,
                        )?;
                        members.insert(fori_target(&target), (member, true));
                    }
                    Ok(members)
                },
            ),
            TensorOp::ForiVjpJvp {
                carry,
                carry_tangent,
                output_cotangent,
                output_cotangent_tangent,
                plan,
                captures,
                tangent_captures,
                target,
                group,
            } => self.batched_group_member(
                groups,
                ("fori_vjp_jvp", *group),
                fori_target(target),
                |graph| {
                    let loop_plan = &plan.loop_plan;
                    let body = batch_loop_body(
                        &loop_plan.body.plan,
                        &loop_plan.carry_name,
                        body_inputs_except(&loop_plan.body.plan, &loop_plan.index_name),
                        batch_size,
                        true,
                    )?;
                    let mut lift = |id| graph.bind_batched(operand(id)?, true, batch_size);
                    let (carry, carry_tangent) = (lift(*carry)?, lift(*carry_tangent)?);
                    let (output_cotangent, output_cotangent_tangent) =
                        (lift(*output_cotangent)?, lift(*output_cotangent_tangent)?);
                    let captures =
                        graph.bind_batched_captures(captures, &body, &operand, batch_size)?;
                    let tangent_captures = graph.bind_batched_captures(
                        tangent_captures,
                        &body,
                        &operand,
                        batch_size,
                    )?;
                    let batched_plan = TensorForiVjpJvpExecutionPlan::new(
                        loop_plan.with_body(body.plan)?,
                        &plan.namespace,
                    )?;
                    let batched_group = graph.nodes.len();
                    let mut members = HashMap::new();
                    for target in fori_targets(&captures) {
                        let member = graph.fori_vjp_jvp(
                            batched_plan.clone(),
                            TensorForiVjpJvpBindings {
                                carry,
                                carry_tangent,
                                output_cotangent,
                                output_cotangent_tangent,
                                captures: captures.clone(),
                                tangent_captures: tangent_captures.clone(),
                            },
                            target.clone(),
                            batched_group,
                        )?;
                        members.insert(fori_target(&target), (member, true));
                    }
                    Ok(members)
                },
            ),
            TensorOp::Scan {
                carry,
                scan_plan,
                captures,
                target,
                group,
            } => {
                let target = match target {
                    TensorScanTarget::Carry => LoopTarget::Carry,
                    TensorScanTarget::Outputs => LoopTarget::Outputs,
                };
                self.batched_group_member(groups, ("scan", *group), target, |graph| {
                    let mut inputs = mapped_names(captures);
                    if mapped[*carry] {
                        inputs.insert(scan_plan.carry_name.clone());
                    }
                    let body = batch_loop_body(
                        &scan_plan.body.plan,
                        &scan_plan.carry_name,
                        inputs,
                        batch_size,
                        false,
                    )?;
                    let (carry_mapped, outputs_mapped) =
                        (body.maps(&scan_plan.carry_name), body.mapped_outputs[1]);
                    let carry = graph.bind_batched(operand(*carry)?, carry_mapped, batch_size)?;
                    let captures =
                        graph.bind_batched_captures(captures, &body, &operand, batch_size)?;
                    let plan = scan_plan.with_body(body.plan)?;
                    let (final_carry, mut outputs) = graph.scan(carry, plan, captures)?;
                    if outputs_mapped {
                        // [T, B, *y] -> [B, T, *y]: the batch axis leads.
                        outputs = graph.swap_leading_axes(outputs)?;
                    }
                    Ok(HashMap::from([
                        (LoopTarget::Carry, (final_carry, carry_mapped)),
                        (LoopTarget::Outputs, (outputs, outputs_mapped)),
                    ]))
                })
            }
            TensorOp::ScanVjp {
                carry,
                final_carry_cotangent,
                output_cotangent,
                scan_plan,
                captures,
                target,
                group,
            } => self.batched_group_member(
                groups,
                ("scan_vjp", *group),
                scan_vjp_target(target),
                |graph| {
                    let body = batch_loop_body(
                        &scan_plan.body.plan,
                        &scan_plan.carry_name,
                        body_inputs_except(&scan_plan.body.plan, &scan_plan.index_name),
                        batch_size,
                        true,
                    )?;
                    let carry = graph.bind_batched(operand(*carry)?, true, batch_size)?;
                    let final_carry_cotangent =
                        graph.bind_batched(operand(*final_carry_cotangent)?, true, batch_size)?;
                    let output_cotangent =
                        graph.bind_stacked_cotangent(operand(*output_cotangent)?, batch_size)?;
                    let captures =
                        graph.bind_batched_captures(captures, &body, &operand, batch_size)?;
                    let plan = scan_plan.with_body(body.plan)?;
                    let batched_group = graph.nodes.len();
                    let mut members = HashMap::new();
                    for target in scan_vjp_targets(&captures) {
                        let member = graph.scan_vjp(
                            plan.clone(),
                            TensorScanVjpBindings {
                                carry,
                                final_carry_cotangent,
                                output_cotangent,
                                captures: captures.clone(),
                            },
                            target.clone(),
                            batched_group,
                        )?;
                        members.insert(scan_vjp_target(&target), (member, true));
                    }
                    Ok(members)
                },
            ),
            TensorOp::ScanVjpJvp {
                carry,
                carry_tangent,
                final_carry_cotangent,
                final_carry_cotangent_tangent,
                output_cotangent,
                output_cotangent_tangent,
                plan,
                captures,
                tangent_captures,
                target,
                group,
            } => self.batched_group_member(
                groups,
                ("scan_vjp_jvp", *group),
                scan_vjp_target(target),
                |graph| {
                    let scan_plan = &plan.scan_plan;
                    let body = batch_loop_body(
                        &scan_plan.body.plan,
                        &scan_plan.carry_name,
                        body_inputs_except(&scan_plan.body.plan, &scan_plan.index_name),
                        batch_size,
                        true,
                    )?;
                    let mut lift = |id| graph.bind_batched(operand(id)?, true, batch_size);
                    let (carry, carry_tangent) = (lift(*carry)?, lift(*carry_tangent)?);
                    let final_carry_cotangent = lift(*final_carry_cotangent)?;
                    let final_carry_cotangent_tangent = lift(*final_carry_cotangent_tangent)?;
                    let output_cotangent =
                        graph.bind_stacked_cotangent(operand(*output_cotangent)?, batch_size)?;
                    let output_cotangent_tangent = graph
                        .bind_stacked_cotangent(operand(*output_cotangent_tangent)?, batch_size)?;
                    let captures =
                        graph.bind_batched_captures(captures, &body, &operand, batch_size)?;
                    let tangent_captures = graph.bind_batched_captures(
                        tangent_captures,
                        &body,
                        &operand,
                        batch_size,
                    )?;
                    let batched_plan = TensorScanVjpJvpExecutionPlan::new(
                        scan_plan.with_body(body.plan)?,
                        &plan.namespace,
                    )?;
                    let batched_group = graph.nodes.len();
                    let mut members = HashMap::new();
                    for target in scan_vjp_targets(&captures) {
                        let member = graph.scan_vjp_jvp(
                            batched_plan.clone(),
                            TensorScanVjpJvpBindings {
                                carry,
                                carry_tangent,
                                final_carry_cotangent,
                                final_carry_cotangent_tangent,
                                output_cotangent,
                                output_cotangent_tangent,
                                captures: captures.clone(),
                                tangent_captures: tangent_captures.clone(),
                            },
                            target.clone(),
                            batched_group,
                        )?;
                        members.insert(scan_vjp_target(&target), (member, true));
                    }
                    Ok(members)
                },
            ),
            TensorOp::Region(RegionNode {
                kind: RegionKind::While { carry, loop_plan },
                captures,
                ..
            }) => {
                let mut inputs = mapped_names(captures);
                if mapped[*carry] {
                    inputs.insert(loop_plan.carry_name().to_string());
                }
                let (plan, body) = batch_while_plan(loop_plan, inputs, batch_size)?;
                let carry_mapped = body.maps(loop_plan.carry_name());
                let carry = self.bind_batched(operand(*carry)?, carry_mapped, batch_size)?;
                let captures = self.bind_batched_captures(captures, &body, &operand, batch_size)?;
                Ok((self.while_loop(carry, plan, captures)?, carry_mapped))
            }
            _ => Err(BatchingError::Invalid(format!(
                "{} is not a loop region node",
                super::tensor_op_name(&node.op)
            ))),
        }
    }

    /// The member `target` of the batched loop group `key`, building every
    /// member of the group with `build` the first time one is requested, so
    /// the members share one batched plan, one set of batched operands, and
    /// one group id (one execution computes all of them).
    fn batched_group_member(
        &mut self,
        groups: &mut LoopBatchGroups,
        key: (&'static str, usize),
        target: LoopTarget,
        build: impl FnOnce(
            &mut Self,
        ) -> Result<HashMap<LoopTarget, (TensorNodeId, bool)>, BatchingError>,
    ) -> Result<(TensorNodeId, bool), BatchingError> {
        let members = match groups.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(build(self)?),
        };
        members.get(&target).copied().ok_or_else(|| {
            BatchingError::Invalid(format!("batched {} group has no member {target:?}", key.0))
        })
    }

    /// The spliced operand `(node, mapped)` bound to a body input that is
    /// mapped when `want_mapped`: an unmapped operand of a mapped input is
    /// broadcast over the batch.
    fn bind_batched(
        &mut self,
        (node, mapped): (TensorNodeId, bool),
        want_mapped: bool,
        batch_size: usize,
    ) -> Result<TensorNodeId, BatchingError> {
        match (mapped, want_mapped) {
            (false, true) => {
                let shape = batched_shape(batch_size, &self.node(node)?.shape);
                Ok(self.broadcast_to(node, shape)?)
            }
            (true, false) => Err(BatchingError::Invalid(
                "vmap bound a mapped operand to an unmapped region input".to_string(),
            )),
            _ => Ok(node),
        }
    }

    /// Named region captures bound to the batched body: each operand is
    /// broadcast when its body input is mapped and it is not.
    fn bind_batched_captures(
        &mut self,
        captures: &[(String, TensorNodeId)],
        body: &BatchedBody,
        operand: &impl Fn(TensorNodeId) -> Result<(TensorNodeId, bool), BatchingError>,
        batch_size: usize,
    ) -> Result<Vec<(String, TensorNodeId)>, BatchingError> {
        captures
            .iter()
            .map(|(name, id)| {
                let bound = self.bind_batched(operand(*id)?, body.maps(name), batch_size)?;
                Ok((name.clone(), bound))
            })
            .collect()
    }

    /// A stacked scan cotangent (`[T, *y]` unmapped, `[B, T, *y]` mapped) in
    /// the time-major layout `[T, B, *y]` of a batched scan's outputs.
    fn bind_stacked_cotangent(
        &mut self,
        operand: (TensorNodeId, bool),
        batch_size: usize,
    ) -> Result<TensorNodeId, BatchingError> {
        let batch_major = self.bind_batched(operand, true, batch_size)?;
        Ok(self.swap_leading_axes(batch_major)?)
    }

    /// `node` with its first two axes exchanged.
    fn swap_leading_axes(&mut self, node: TensorNodeId) -> Result<TensorNodeId, String> {
        let source = self.node(node)?;
        let rank = source.shape.len();
        if rank < 2 {
            return Err(format!(
                "cannot exchange the leading axes of a rank-{rank} tensor"
            ));
        }
        let mut shape = source.shape.clone();
        shape.swap(0, 1);
        let (dtype, weak) = (source.dtype, source.weak);
        let axes = [1, 0].into_iter().chain(2..rank).collect();
        Ok(self.push_node(
            TensorOp::Transpose { input: node, axes },
            shape,
            dtype,
            weak,
        ))
    }
}

impl TensorIr {
    /// Appends the batched form of a `Cond` node with at least one mapped
    /// operand (see the module documentation) and returns it with whether it
    /// is mapped.
    ///
    /// Under an unmapped predicate every example takes the same branch, so
    /// the node stays one lazy `Cond` that reads its predicate once. Both
    /// branch regions are batched over the inputs bound to mapped operands,
    /// which are bound unchanged; the output is mapped when either branch
    /// maps it, and the branch that does not is batched again with its
    /// output broadcast, so both regions keep one output shape and the cond
    /// one batchedness. Under a mapped predicate the examples may take
    /// different branches; see [`Self::push_selected_cond`].
    pub(super) fn push_batched_cond(
        &mut self,
        node: &TensorNode,
        (remap, mapped): (&HashMap<TensorNodeId, TensorNodeId>, &[bool]),
        batch_size: usize,
    ) -> Result<(TensorNodeId, bool), BatchingError> {
        let TensorOp::Cond {
            predicate,
            branches,
            captures,
        } = &node.op
        else {
            return Err(BatchingError::Invalid(format!(
                "{} is not a cond region node",
                super::tensor_op_name(&node.op)
            )));
        };
        let operand = |id: TensorNodeId| -> Result<(TensorNodeId, bool), BatchingError> {
            remap
                .get(&id)
                .map(|spliced| (*spliced, mapped[id]))
                .ok_or_else(|| {
                    BatchingError::Invalid(format!("node {id} is missing from the vmap remap"))
                })
        };
        let captures = captures
            .iter()
            .map(|(name, id)| Ok((name.clone(), operand(*id)?)))
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let (predicate, predicate_mapped) = operand(*predicate)?;
        if predicate_mapped {
            let selected =
                self.push_selected_cond(node, predicate, branches, &captures, batch_size)?;
            return Ok((selected, true));
        }
        let inputs = captures
            .iter()
            .filter(|(_, (_, mapped))| *mapped)
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();
        let batch = |region: &TensorExecutionPlan, force: bool| {
            batch_region_plan(region, &inputs, batch_size, &[force])
        };
        let (mut on_true, true_mapped) = batch(&branches.on_true.plan, false)?;
        let (mut on_false, false_mapped) = batch(&branches.on_false.plan, false)?;
        let output_mapped = true_mapped[0] || false_mapped[0];
        if output_mapped && !true_mapped[0] {
            on_true = batch(&branches.on_true.plan, true)?.0;
        }
        if output_mapped && !false_mapped[0] {
            on_false = batch(&branches.on_false.plan, true)?.0;
        }
        let captures = captures
            .into_iter()
            .map(|(name, (node, _))| (name, node))
            .collect();
        let branches = TensorCondExecutionPlan::new(on_true, on_false)?;
        let batched = self.cond_with_captures(predicate, branches, captures)?;
        Ok((batched, output_mapped))
    }

    /// A cond with a mapped `[B]` predicate as `jax.vmap` lowers it: both
    /// branches are spliced into this graph batched, and
    /// `where(predicate, on_true, on_false)`, the predicate broadcast over
    /// the output's trailing axes, takes each example's result from the
    /// branch its predicate selects. No predicate is read back to the host.
    ///
    /// Both branches now run for every example. That costs the work of both,
    /// and a branch that would raise or produce NaN or inf on the inputs of
    /// the examples that take the other branch (`log` of a negative value,
    /// say) now executes on them; the select keeps those values out of the
    /// result. Derivatives need one more step. `where` gives the unselected
    /// branch a zero cotangent, but that branch's own derivative can be
    /// infinite or NaN there (`log` at zero, `sqrt` below zero), and
    /// `0 * inf` is NaN, which would be added to the selected branch's
    /// gradient. So a mapped floating operand `x` enters the true branch as
    /// `where(predicate, x, stop_gradient(x))` and the false branch as
    /// `where(predicate, stop_gradient(x), x)`: the same values, but each
    /// branch's cotangent for `x` is selected by the example's predicate
    /// again after that branch's derivative, which drops a NaN of the
    /// unselected branch instead of adding it. Every derivative of a `where`
    /// is a `where` with the same condition, so higher orders stay clean, and
    /// forward mode is clean already because the output select comes after
    /// both tangents. An unmapped operand is shared by every example and its
    /// gradient sums the contributions of both branches over the batch, so a
    /// non-finite derivative of the unselected branch with respect to it
    /// still propagates, as in JAX; masking it would need a copy of it per
    /// example.
    ///
    /// A floating predicate selects by `!= 0`, as `where` does; an unbatched
    /// cond rejects a non-finite predicate, which cannot raise per example
    /// here, so it makes that example's output NaN instead.
    fn push_selected_cond(
        &mut self,
        node: &TensorNode,
        predicate: TensorNodeId,
        branches: &TensorCondExecutionPlan,
        captures: &[(String, (TensorNodeId, bool))],
        batch_size: usize,
    ) -> Result<TensorNodeId, BatchingError> {
        let rank = node.shape.len();
        let mut outputs = Vec::with_capacity(2);
        for (region, taken) in [
            (&branches.on_true.plan, true),
            (&branches.on_false.plan, false),
        ] {
            let mut bindings = BTreeMap::new();
            for (name, (value, mapped)) in captures {
                let source = self.node(*value)?;
                let value = if *mapped && source.dtype.is_floating() {
                    let (shape, dtype, weak) = (source.shape.clone(), source.dtype, source.weak);
                    let condition = self.pad_batched(predicate, shape.len() - 1)?;
                    let frozen = self.push_node(
                        TensorOp::StopGradient { input: *value },
                        shape.clone(),
                        dtype,
                        weak,
                    );
                    let (on_true, on_false) = if taken {
                        (*value, frozen)
                    } else {
                        (frozen, *value)
                    };
                    self.push_node(
                        TensorOp::Where {
                            condition,
                            on_true,
                            on_false,
                        },
                        shape,
                        dtype,
                        weak,
                    )
                } else {
                    *value
                };
                bindings.insert(name.clone(), (value, *mapped));
            }
            let spliced = self.inline_batched(
                &region.as_ir(),
                &bindings,
                batch_size,
                &[region.output_node_id],
            )?;
            outputs.push(self.bind_batched(spliced[0], true, batch_size)?);
        }
        let condition = self.pad_batched(predicate, rank)?;
        let selected = self.push_node(
            TensorOp::Where {
                condition,
                on_true: outputs[0],
                on_false: outputs[1],
            },
            batched_shape(batch_size, &node.shape),
            node.dtype,
            node.weak,
        );
        if !self.node(predicate)?.dtype.is_floating() {
            return Ok(selected);
        }
        // `p * 0 == 0` holds exactly for a finite `p`.
        let zero = self.scalar_constant(0.0);
        let probe = self.mul(predicate, zero)?;
        let finite = self.compare(probe, zero, TensorComparison::Equal)?;
        let finite = self.pad_batched(finite, rank)?;
        let nan = self.scalar_constant(f64::NAN);
        Ok(self.where_select(finite, selected, nan)?)
    }
}

fn fori_target(target: &TensorForiVjpTarget) -> LoopTarget {
    match target {
        TensorForiVjpTarget::Carry => LoopTarget::Carry,
        TensorForiVjpTarget::External(name) => LoopTarget::External(name.clone()),
    }
}

/// The carry target followed by one target per capture: every member of a
/// reverse-pass group.
fn fori_targets(captures: &[(String, TensorNodeId)]) -> Vec<TensorForiVjpTarget> {
    std::iter::once(TensorForiVjpTarget::Carry)
        .chain(
            captures
                .iter()
                .map(|(name, _)| TensorForiVjpTarget::External(name.clone())),
        )
        .collect()
}

fn scan_vjp_target(target: &TensorScanVjpTarget) -> LoopTarget {
    match target {
        TensorScanVjpTarget::Carry => LoopTarget::Carry,
        TensorScanVjpTarget::External(name) => LoopTarget::External(name.clone()),
    }
}

/// The carry target followed by one target per capture: every member of a
/// reverse-pass group.
fn scan_vjp_targets(captures: &[(String, TensorNodeId)]) -> Vec<TensorScanVjpTarget> {
    std::iter::once(TensorScanVjpTarget::Carry)
        .chain(
            captures
                .iter()
                .map(|(name, _)| TensorScanVjpTarget::External(name.clone())),
        )
        .collect()
}
