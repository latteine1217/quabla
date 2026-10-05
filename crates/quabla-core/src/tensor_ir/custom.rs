//! Custom differentiation rules: the `TensorOp::Custom` node behind
//! `quabla.custom_vjp`, `quabla.custom_jvp`, and `quabla.checkpoint`.
//!
//! A function with a custom rule is staged twice into the enclosing graph:
//! its primal graph is spliced in as ordinary nodes, and each primal output
//! is wrapped in a `Custom` node that names the rule and the operands the
//! rule differentiates with respect to. The node is the identity on its
//! primal value, so plan compilation aliases it away and no backend ever
//! executes it. Only the symbolic transforms look through it:
//!
//! - reverse mode splices the rule's forward graph (primal outputs and
//!   residuals) in place of the primal value, then its backward graph, which
//!   maps the residuals and the output cotangents to operand cotangents;
//! - forward mode splices the rule's tangent graph, or fails for a rule
//!   without one (a `custom_vjp` function, as in JAX);
//! - batching (`vmap`) batches the rule graphs along with the primal.
//!
//! Every spliced rule graph is ordinary IR, so a reverse-mode graph through
//! a custom rule can be differentiated again (the backward graph is
//! differentiated like any other code) and compiled for every backend.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{batched_shape, BatchingError, TensorDType, TensorIr, TensorNodeId, TensorOp};

/// The forward-mode part of a [`TensorCustomRule`]: a graph taking the
/// operand inputs of the rule and one tangent input per differentiable
/// operand (`tangent_names[i]` for operand `i`, `None` for a `bool`
/// operand), with one tangent output per rule output.
#[derive(Clone, Debug)]
pub struct TensorCustomTangent {
    graph: TensorIr,
    tangent_names: Vec<Option<String>>,
    outputs: Vec<TensorNodeId>,
}

impl TensorCustomTangent {
    pub fn new(
        graph: TensorIr,
        tangent_names: Vec<Option<String>>,
        outputs: Vec<TensorNodeId>,
    ) -> Self {
        Self {
            graph,
            tangent_names,
            outputs,
        }
    }
}

/// A custom differentiation rule shared by the `Custom` nodes of one call.
///
/// - `forward` takes the operands as inputs named `operand_names` and
///   returns `output_count` primal outputs followed by the residuals.
/// - `backward` takes the residuals (`residual_names`) and one cotangent per
///   output (`cotangent_names`) and returns one cotangent per operand, or
///   `None` for an operand that receives none.
/// - `tangent` is the forward-mode rule, absent for `custom_vjp`.
/// - `rematerialize` marks a `checkpoint` rule, whose residuals are the
///   operands and whose backward graph recomputes the primal: the residuals
///   are bound through a multiplication by one, which plan compilation does
///   not merge with the forward computation, so the recomputed intermediates
///   are not the stored forward ones.
#[derive(Clone, Debug)]
pub struct TensorCustomRule {
    name: String,
    operand_names: Vec<String>,
    output_count: usize,
    forward: TensorIr,
    forward_outputs: Vec<TensorNodeId>,
    residual_names: Vec<String>,
    cotangent_names: Vec<String>,
    backward: TensorIr,
    backward_outputs: Vec<Option<TensorNodeId>>,
    tangent: Option<TensorCustomTangent>,
    rematerialize: bool,
}

type Aval = (Vec<usize>, TensorDType);

fn aval(graph: &TensorIr, id: TensorNodeId) -> Result<Aval, String> {
    let node = graph.node(id)?;
    Ok((node.shape.clone(), node.dtype))
}

/// Checks that `name`, if `graph` has such an input, has the aval `expected`.
fn check_input(graph: &TensorIr, name: &str, expected: &Aval, what: &str) -> Result<(), String> {
    let Some(&id) = graph.input_nodes.get(name) else {
        return Ok(());
    };
    let actual = aval(graph, id)?;
    if &actual != expected {
        return Err(format!(
            "custom rule {what} input {name:?} is {}{:?}, expected {}{:?}",
            actual.1, actual.0, expected.1, expected.0
        ));
    }
    Ok(())
}

/// Splices `outputs` of `callee` into `graph`, binding each callee input in
/// `names` that the callee has to the node at the same position in `nodes`.
fn splice(
    graph: &mut TensorIr,
    callee: &TensorIr,
    names: &[&str],
    nodes: &[TensorNodeId],
    outputs: &[TensorNodeId],
) -> Result<Vec<TensorNodeId>, String> {
    let bindings = names
        .iter()
        .zip(nodes)
        .filter(|(name, _)| callee.input_nodes.contains_key(**name))
        .map(|(name, node)| (name.to_string(), *node))
        .collect::<BTreeMap<_, _>>();
    graph.inline(callee, &bindings, outputs)
}

/// [`splice`] vectorized over `batch_size` (see [`TensorIr::inline_batched`]).
fn splice_batched(
    graph: &mut TensorIr,
    callee: &TensorIr,
    bindings: Vec<(&str, TensorNodeId, bool)>,
    batch_size: usize,
    outputs: &[TensorNodeId],
) -> Result<Vec<(TensorNodeId, bool)>, BatchingError> {
    let bindings = bindings
        .into_iter()
        .filter(|(name, _, _)| callee.input_nodes.contains_key(*name))
        .map(|(name, node, mapped)| (name.to_string(), (node, mapped)))
        .collect::<BTreeMap<_, _>>();
    graph.inline_batched(callee, &bindings, batch_size, outputs)
}

/// `node` (shape `[B, ...]` when `mapped`) with the batch axis the rule's
/// convention asks for: broadcast over the batch when `want_mapped` and the
/// value is shared, summed over the batch when the value is per example but
/// a shared result is wanted (the cotangent of an operand shared by every
/// example is the sum of the per-example cotangents).
fn conform_batch(
    graph: &mut TensorIr,
    node: TensorNodeId,
    mapped: bool,
    want_mapped: bool,
    batch_size: usize,
) -> Result<TensorNodeId, String> {
    match (mapped, want_mapped) {
        (false, true) => {
            let shape = batched_shape(batch_size, &graph.node(node)?.shape);
            graph.broadcast_to(node, shape)
        }
        (true, false) => graph.sum_axis(node, 0),
        _ => Ok(node),
    }
}

impl TensorCustomRule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        forward: TensorIr,
        operand_names: Vec<String>,
        output_count: usize,
        forward_outputs: Vec<TensorNodeId>,
        backward: TensorIr,
        residual_names: Vec<String>,
        cotangent_names: Vec<String>,
        backward_outputs: Vec<Option<TensorNodeId>>,
        tangent: Option<TensorCustomTangent>,
        rematerialize: bool,
    ) -> Result<Self, String> {
        let rule = Self {
            name,
            operand_names,
            output_count,
            forward,
            forward_outputs,
            residual_names,
            cotangent_names,
            backward,
            backward_outputs,
            tangent,
            rematerialize,
        };
        rule.validate()?;
        Ok(rule)
    }

    fn validate(&self) -> Result<(), String> {
        let name = &self.name;
        let operands = self.operand_avals()?;
        if self.output_count == 0 {
            return Err(format!("custom rule {name} has no outputs"));
        }
        if self.forward_outputs.len() != self.output_count + self.residual_names.len() {
            return Err(format!(
                "custom rule {name} has {} forward outputs for {} outputs and {} residuals",
                self.forward_outputs.len(),
                self.output_count,
                self.residual_names.len()
            ));
        }
        let outputs = self.output_avals()?;
        if let Some((index, _)) = outputs
            .iter()
            .enumerate()
            .find(|(_, (_, dtype))| !dtype.is_floating())
        {
            return Err(format!(
                "custom rule {name} output {index} is not floating-point; a custom \
                 differentiation rule needs floating-point outputs"
            ));
        }
        if self.cotangent_names.len() != self.output_count {
            return Err(format!(
                "custom rule {name} has {} cotangent inputs for {} outputs",
                self.cotangent_names.len(),
                self.output_count
            ));
        }
        for (residual, id) in self
            .residual_names
            .iter()
            .zip(&self.forward_outputs[self.output_count..])
        {
            let expected = aval(&self.forward, *id)?;
            check_input(&self.backward, residual, &expected, "backward residual")?;
        }
        for (cotangent, expected) in self.cotangent_names.iter().zip(&outputs) {
            check_input(&self.backward, cotangent, expected, "backward cotangent")?;
        }
        if self.backward_outputs.len() != operands.len() {
            return Err(format!(
                "custom rule {name} has {} backward outputs for {} operands",
                self.backward_outputs.len(),
                operands.len()
            ));
        }
        for (index, (output, expected)) in self.backward_outputs.iter().zip(&operands).enumerate() {
            let Some(output) = output else {
                continue;
            };
            let actual = aval(&self.backward, *output)?;
            if &actual != expected || !expected.1.is_floating() {
                return Err(format!(
                    "custom rule {name} returns a {}{:?} cotangent for operand {index}, \
                     which is {}{:?}",
                    actual.1, actual.0, expected.1, expected.0
                ));
            }
        }
        if let Some(tangent) = &self.tangent {
            if tangent.tangent_names.len() != operands.len()
                || tangent.outputs.len() != self.output_count
            {
                return Err(format!(
                    "custom rule {name} has a tangent graph with {} tangent inputs and {} \
                     outputs for {} operands and {} outputs",
                    tangent.tangent_names.len(),
                    tangent.outputs.len(),
                    operands.len(),
                    self.output_count
                ));
            }
            for (operand, expected) in self.operand_names.iter().zip(&operands) {
                check_input(&tangent.graph, operand, expected, "tangent primal")?;
            }
            for (tangent_name, expected) in tangent.tangent_names.iter().zip(&operands) {
                if let Some(tangent_name) = tangent_name {
                    check_input(&tangent.graph, tangent_name, expected, "tangent")?;
                }
            }
            for (index, (output, expected)) in tangent.outputs.iter().zip(&outputs).enumerate() {
                let actual = aval(&tangent.graph, *output)?;
                if &actual != expected {
                    return Err(format!(
                        "custom rule {name} returns a {}{:?} tangent for output {index}, \
                         which is {}{:?}",
                        actual.1, actual.0, expected.1, expected.0
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// The shape and dtype of each operand, from the forward graph inputs.
    fn operand_avals(&self) -> Result<Vec<Aval>, String> {
        self.operand_names
            .iter()
            .map(|name| {
                let id = self.forward.input_nodes.get(name).ok_or_else(|| {
                    format!(
                        "custom rule {} has no forward input for operand {name:?}",
                        self.name
                    )
                })?;
                aval(&self.forward, *id)
            })
            .collect()
    }

    fn output_avals(&self) -> Result<Vec<Aval>, String> {
        self.forward_outputs[..self.output_count]
            .iter()
            .map(|id| aval(&self.forward, *id))
            .collect()
    }

    fn operand_name_refs(&self) -> Vec<&str> {
        self.operand_names.iter().map(String::as_str).collect()
    }

    /// Checks `operands` and the primal `values` of a new call against the
    /// rule's operand and output avals.
    pub(super) fn check_call(
        &self,
        graph: &TensorIr,
        values: &[TensorNodeId],
        operands: &[TensorNodeId],
    ) -> Result<(), String> {
        let name = &self.name;
        let expected_operands = self.operand_avals()?;
        if operands.len() != expected_operands.len() || values.len() != self.output_count {
            return Err(format!(
                "custom rule {name} takes {} operands and {} outputs, got {} and {}",
                expected_operands.len(),
                self.output_count,
                operands.len(),
                values.len()
            ));
        }
        for (index, (operand, expected)) in operands.iter().zip(&expected_operands).enumerate() {
            let actual = aval(graph, *operand)?;
            if &actual != expected {
                return Err(format!(
                    "custom rule {name} operand {index} is {}{:?}, expected {}{:?}",
                    actual.1, actual.0, expected.1, expected.0
                ));
            }
        }
        for (index, (value, expected)) in values.iter().zip(self.output_avals()?).enumerate() {
            let actual = aval(graph, *value)?;
            if actual != expected {
                return Err(format!(
                    "custom rule {name} output {index} is {}{:?}, but its rule computes \
                     {}{:?}",
                    actual.1, actual.0, expected.1, expected.0
                ));
            }
        }
        Ok(())
    }

    /// Splices the forward graph at `operands`: the primal outputs followed
    /// by the residuals.
    pub(super) fn splice_forward(
        &self,
        graph: &mut TensorIr,
        operands: &[TensorNodeId],
    ) -> Result<Vec<TensorNodeId>, String> {
        splice(
            graph,
            &self.forward,
            &self.operand_name_refs(),
            operands,
            &self.forward_outputs,
        )
    }

    /// Splices the backward graph: one cotangent per operand, `None` where
    /// the rule returns none.
    pub(super) fn splice_backward(
        &self,
        graph: &mut TensorIr,
        residuals: &[TensorNodeId],
        cotangents: &[TensorNodeId],
    ) -> Result<Vec<Option<TensorNodeId>>, String> {
        let mut bound = Vec::with_capacity(residuals.len() + cotangents.len());
        for residual in residuals {
            bound.push(
                if self.rematerialize && graph.node(*residual)?.dtype.is_floating() {
                    let one = graph.scalar_constant(1.0);
                    graph.mul(*residual, one)?
                } else {
                    *residual
                },
            );
        }
        bound.extend_from_slice(cotangents);
        let names = self
            .residual_names
            .iter()
            .chain(&self.cotangent_names)
            .map(String::as_str)
            .collect::<Vec<_>>();
        let wanted = self
            .backward_outputs
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut spliced = splice(graph, &self.backward, &names, &bound, &wanted)?.into_iter();
        Ok(self
            .backward_outputs
            .iter()
            .map(|output| output.and_then(|_| spliced.next()))
            .collect())
    }

    /// Splices the tangent graph: one tangent per output. `tangents[i]` is
    /// the tangent of operand `i`.
    pub(super) fn splice_tangent(
        &self,
        graph: &mut TensorIr,
        operands: &[TensorNodeId],
        tangents: &[TensorNodeId],
    ) -> Result<Vec<TensorNodeId>, String> {
        let tangent = self.tangent.as_ref().ok_or_else(|| {
            format!(
                "forward-mode differentiation (jvp, or jacobian in forward mode) of {} is not \
                 supported: it defines only a reverse-mode rule; use quabla.custom_jvp, or a \
                 reverse-mode transform",
                self.name
            )
        })?;
        let mut names = self.operand_name_refs();
        let mut bound = operands.to_vec();
        for (name, node) in tangent.tangent_names.iter().zip(tangents) {
            if let Some(name) = name {
                names.push(name);
                bound.push(*node);
            }
        }
        splice(graph, &tangent.graph, &names, &bound, &tangent.outputs)
    }

    /// The rule of a call batched over `batch_size` examples, where operand
    /// `i` carries a leading batch axis when `operand_mapped[i]`. Every
    /// output of the batched rule is batched; operand cotangents keep the
    /// batching of their operand.
    pub(super) fn batched(
        &self,
        operand_mapped: &[bool],
        batch_size: usize,
    ) -> Result<Self, BatchingError> {
        let operands = self.operand_avals()?;
        let outputs = self.output_avals()?;
        let mapped_shape = |shape: &[usize], mapped: bool| {
            if mapped {
                batched_shape(batch_size, shape)
            } else {
                shape.to_vec()
            }
        };

        let mut forward = TensorIr::new();
        let mut bindings = Vec::with_capacity(operands.len());
        for ((name, (shape, dtype)), &mapped) in
            self.operand_names.iter().zip(&operands).zip(operand_mapped)
        {
            let input = forward.input_typed(name.clone(), mapped_shape(shape, mapped), *dtype)?;
            bindings.push((name.as_str(), input, mapped));
        }
        let spliced = splice_batched(
            &mut forward,
            &self.forward,
            bindings,
            batch_size,
            &self.forward_outputs,
        )?;
        let mut forward_outputs = Vec::with_capacity(spliced.len());
        for (index, &(node, mapped)) in spliced.iter().enumerate() {
            forward_outputs.push(if index < self.output_count {
                conform_batch(&mut forward, node, mapped, true, batch_size)?
            } else {
                node
            });
        }

        let mut backward = TensorIr::new();
        let mut bindings = Vec::new();
        for (name, (&node, &(_, mapped))) in self.residual_names.iter().zip(
            forward_outputs[self.output_count..]
                .iter()
                .zip(&spliced[self.output_count..]),
        ) {
            let (shape, dtype) = aval(&forward, node)?;
            let input = backward.input_typed(name.clone(), shape, dtype)?;
            bindings.push((name.as_str(), input, mapped));
        }
        for (name, (shape, dtype)) in self.cotangent_names.iter().zip(&outputs) {
            let input =
                backward.input_typed(name.clone(), batched_shape(batch_size, shape), *dtype)?;
            bindings.push((name.as_str(), input, true));
        }
        let wanted = self
            .backward_outputs
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut spliced =
            splice_batched(&mut backward, &self.backward, bindings, batch_size, &wanted)?
                .into_iter();
        let mut backward_outputs = Vec::with_capacity(self.backward_outputs.len());
        for (output, &want_mapped) in self.backward_outputs.iter().zip(operand_mapped) {
            backward_outputs.push(match output {
                Some(_) => {
                    let (node, mapped) = spliced
                        .next()
                        .ok_or_else(|| "custom rule backward output is missing".to_string())?;
                    Some(conform_batch(
                        &mut backward,
                        node,
                        mapped,
                        want_mapped,
                        batch_size,
                    )?)
                }
                None => None,
            });
        }

        let tangent = match &self.tangent {
            None => None,
            Some(tangent) => {
                let mut graph = TensorIr::new();
                let mut bindings = Vec::new();
                for ((name, (shape, dtype)), &mapped) in
                    self.operand_names.iter().zip(&operands).zip(operand_mapped)
                {
                    let input =
                        graph.input_typed(name.clone(), mapped_shape(shape, mapped), *dtype)?;
                    bindings.push((name.as_str(), input, mapped));
                }
                for ((name, (shape, dtype)), &mapped) in tangent
                    .tangent_names
                    .iter()
                    .zip(&operands)
                    .zip(operand_mapped)
                {
                    if let Some(name) = name {
                        let input =
                            graph.input_typed(name.clone(), mapped_shape(shape, mapped), *dtype)?;
                        bindings.push((name.as_str(), input, mapped));
                    }
                }
                let spliced = splice_batched(
                    &mut graph,
                    &tangent.graph,
                    bindings,
                    batch_size,
                    &tangent.outputs,
                )?;
                let mut outputs = Vec::with_capacity(spliced.len());
                for (node, mapped) in spliced {
                    outputs.push(conform_batch(&mut graph, node, mapped, true, batch_size)?);
                }
                Some(TensorCustomTangent {
                    graph,
                    tangent_names: tangent.tangent_names.clone(),
                    outputs,
                })
            }
        };

        Ok(Self::new(
            self.name.clone(),
            forward,
            self.operand_names.clone(),
            self.output_count,
            forward_outputs,
            backward,
            self.residual_names.clone(),
            self.cotangent_names.clone(),
            backward_outputs,
            tangent,
            self.rematerialize,
        )?)
    }
}

impl TensorIr {
    /// Wraps the primal outputs `values` of a call of a function with the
    /// custom differentiation rule `rule` at `operands`, returning one
    /// `Custom` node per output (see the module notes). The values must be
    /// the outputs the rule's forward graph computes, with the same shapes
    /// and dtypes, and the operands must match the rule's operand inputs.
    pub fn custom(
        &mut self,
        rule: Arc<TensorCustomRule>,
        values: &[TensorNodeId],
        operands: &[TensorNodeId],
    ) -> Result<Vec<TensorNodeId>, String> {
        rule.check_call(self, values, operands)?;
        let members = values.iter().copied().enumerate().collect::<Vec<_>>();
        self.push_custom_group(rule, &members, operands)
    }

    /// Pushes the `Custom` nodes `(output index, primal value)` of one call
    /// consecutively; the group id is the id of the first.
    ///
    /// Every consumer of a call's outputs comes after all of them, which the
    /// reverse-mode rule relies on: when it reaches the call, the cotangents
    /// of every output are complete.
    pub(super) fn push_custom_group(
        &mut self,
        rule: Arc<TensorCustomRule>,
        members: &[(usize, TensorNodeId)],
        operands: &[TensorNodeId],
    ) -> Result<Vec<TensorNodeId>, String> {
        let group = self.nodes.len();
        let mut ids = Vec::with_capacity(members.len());
        for &(output, value) in members {
            let source = self.node(value)?;
            let (shape, dtype, weak) = (source.shape.clone(), source.dtype, source.weak);
            ids.push(self.push_node(
                TensorOp::Custom {
                    value,
                    operands: operands.to_vec(),
                    rule: rule.clone(),
                    output,
                    group,
                },
                shape,
                dtype,
                weak,
            ));
        }
        Ok(ids)
    }

    /// The `Custom` nodes of each call, `(output index, node id)` by group.
    pub(super) fn custom_groups(&self) -> BTreeMap<usize, Vec<(usize, TensorNodeId)>> {
        let mut groups = BTreeMap::<usize, Vec<_>>::new();
        for (id, node) in self.nodes.iter().enumerate() {
            if let TensorOp::Custom { output, group, .. } = &node.op {
                groups.entry(*group).or_default().push((*output, id));
            }
        }
        groups
    }

    /// Whether a `Custom` node has a rule without a tangent graph (a
    /// `custom_vjp` call), so forward mode through it fails; `jacobian`
    /// uses reverse mode for such graphs.
    pub fn has_reverse_only_custom_rule(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| matches!(&node.op, TensorOp::Custom { rule, .. } if rule.tangent.is_none()))
    }

    /// The name of the rule of the first `Custom` node that `outputs`
    /// depend on, if any. Control-flow regions reject such bodies, since a
    /// compiled region no longer carries the rule.
    pub fn custom_rule_name(&self, outputs: &[TensorNodeId]) -> Result<Option<String>, String> {
        let reachable = self.reachable_from(outputs)?;
        Ok(self
            .nodes
            .iter()
            .zip(reachable)
            .find_map(|(node, reachable)| match &node.op {
                TensorOp::Custom { rule, .. } if reachable => Some(rule.name().to_string()),
                _ => None,
            }))
    }
}
