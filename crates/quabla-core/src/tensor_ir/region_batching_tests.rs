//! Structural checks of the loop batching rules (`region_batching.rs`): which
//! operands and body inputs get the batch axis, and the carry fixed point.
//! The numerical checks against per-example loops are in
//! `tests/tensor_ir.rs`.

use super::*;

const BATCH: usize = 4;

/// `carry -> sin(carry * shift) + 0.1 * index` over `[0, 3)`.
fn fori_plan() -> Result<TensorForiExecutionPlan, String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![3])?;
    let shift = body.input("shift", vec![3])?;
    let index = body.input("index", vec![])?;
    let product = body.mul(carry, shift)?;
    let wave = body.sin(product)?;
    let tenth = body.scalar_constant(0.1);
    let offset = body.mul(index, tenth)?;
    let next = body.add(wave, offset)?;
    TensorForiExecutionPlan::new(0, 3, body.compile_cpu(next)?, "carry", "index")
}

/// `carry -> tanh(carry * scale + 0.1 * index)` with per-step output
/// `next * w`, over `[0, 5)`: `w` reaches the outputs but not the carry.
fn scan_plan() -> Result<TensorScanExecutionPlan, String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![3])?;
    let scale = body.input("scale", vec![3])?;
    let w = body.input("w", vec![3])?;
    let index = body.input("index", vec![])?;
    let product = body.mul(carry, scale)?;
    let tenth = body.scalar_constant(0.1);
    let offset = body.mul(index, tenth)?;
    let shifted = body.add(product, offset)?;
    let next = body.tanh(shifted)?;
    let output = body.mul(next, w)?;
    let (plan, _) = body.compile_cpu_many(&[next, output])?;
    TensorScanExecutionPlan::new(0, 5, plan, "carry", "index")
}

/// Binds every callee input of `inputs` (`name -> mapped`) to a new input of
/// `graph` with the matching (batched) shape.
fn bind(
    graph: &mut TensorIr,
    callee: &TensorIr,
    inputs: &[(&str, bool)],
) -> Result<BTreeMap<String, (TensorNodeId, bool)>, String> {
    let mut bindings = BTreeMap::new();
    for (name, mapped) in inputs {
        let shape = callee.node_shape(callee.input_node_id(name)?)?;
        let shape = if *mapped {
            batched_shape(BATCH, &shape)
        } else {
            shape
        };
        bindings.insert(name.to_string(), (graph.input(*name, shape)?, *mapped));
    }
    Ok(bindings)
}

fn batched(
    callee: &TensorIr,
    inputs: &[(&str, bool)],
    outputs: &[TensorNodeId],
) -> Result<(TensorIr, Vec<(TensorNodeId, bool)>), String> {
    let mut graph = TensorIr::new();
    let bindings = bind(&mut graph, callee, inputs)?;
    let results = graph
        .inline_batched(callee, &bindings, BATCH, outputs)
        .map_err(|error| error.to_string())?;
    Ok((graph, results))
}

fn is_broadcast_of(graph: &TensorIr, node: TensorNodeId, source: TensorNodeId) -> bool {
    matches!(graph.nodes[node].op, TensorOp::Broadcast { input } if input == source)
}

#[test]
fn a_mapped_carry_leaves_unmapped_captures_unbatched() -> Result<(), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let looped = callee.fori(x, fori_plan()?, vec![("shift".to_string(), s)])?;
    let (graph, results) = batched(&callee, &[("x", true), ("s", false)], &[looped])?;
    let (node, mapped) = results[0];
    assert!(mapped);
    assert_eq!(graph.nodes[node].shape, vec![BATCH, 3]);
    let TensorOp::Fori {
        carry,
        loop_plan,
        captures,
    } = &graph.nodes[node].op
    else {
        panic!("expected a batched fori node");
    };
    // The unmapped capture is bound as is, without a broadcast.
    assert_eq!(*carry, graph.input_node_id("x")?);
    assert_eq!(captures[0].1, graph.input_node_id("s")?);
    let body = &loop_plan.body.plan;
    assert_eq!(body.input_shape("carry")?, vec![BATCH, 3]);
    assert_eq!(body.input_shape("shift")?, vec![3]);
    assert_eq!(body.input_shape("index")?, Vec::<usize>::new());
    Ok(())
}

#[test]
fn a_mapped_capture_maps_the_carry_by_the_fixed_point() -> Result<(), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let looped = callee.fori(x, fori_plan()?, vec![("shift".to_string(), s)])?;
    let (graph, results) = batched(&callee, &[("x", false), ("s", true)], &[looped])?;
    let (node, mapped) = results[0];
    assert!(mapped);
    let TensorOp::Fori {
        carry,
        loop_plan,
        captures,
    } = &graph.nodes[node].op
    else {
        panic!("expected a batched fori node");
    };
    // The next carry depends on the mapped shift, so the carry is mapped
    // from the first iteration: the unmapped initial carry is broadcast.
    assert!(is_broadcast_of(&graph, *carry, graph.input_node_id("x")?));
    assert_eq!(captures[0].1, graph.input_node_id("s")?);
    let body = &loop_plan.body.plan;
    assert_eq!(body.input_shape("carry")?, vec![BATCH, 3]);
    assert_eq!(body.input_shape("shift")?, vec![BATCH, 3]);
    Ok(())
}

#[test]
fn a_scan_capture_that_only_feeds_outputs_leaves_the_carry_unmapped() -> Result<(), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let scale = callee.input("scale", vec![3])?;
    let w = callee.input("w", vec![3])?;
    let (carry, outputs) = callee.scan(
        x,
        scan_plan()?,
        vec![("scale".to_string(), scale), ("w".to_string(), w)],
    )?;
    let (graph, results) = batched(
        &callee,
        &[("x", false), ("scale", false), ("w", true)],
        &[carry, outputs],
    )?;
    let [(final_carry, carry_mapped), (stacked, outputs_mapped)] = results[..] else {
        panic!("expected two results");
    };
    assert!(!carry_mapped && outputs_mapped);
    assert_eq!(graph.nodes[final_carry].shape, vec![3]);
    // The body's [B, 3] steps stack to [T, B, 3]; the batch axis is moved to
    // the front.
    assert_eq!(graph.nodes[stacked].shape, vec![BATCH, 5, 3]);
    let TensorOp::Transpose { input, axes } = &graph.nodes[stacked].op else {
        panic!("expected the stacked outputs to be transposed");
    };
    assert_eq!(axes, &vec![1, 0, 2]);
    assert_eq!(graph.nodes[*input].shape, vec![5, BATCH, 3]);
    let TensorOp::Scan {
        scan_plan, group, ..
    } = &graph.nodes[final_carry].op
    else {
        panic!("expected a batched scan node");
    };
    // Both results come from one batched scan.
    assert!(
        matches!(&graph.nodes[*input].op, TensorOp::Scan { group: other, .. } if other == group)
    );
    let body = &scan_plan.body.plan;
    assert_eq!(body.input_shape("carry")?, vec![3]);
    assert_eq!(body.input_shape("scale")?, vec![3]);
    assert_eq!(body.input_shape("w")?, vec![BATCH, 3]);
    Ok(())
}

#[test]
fn reverse_pass_members_share_one_batched_group_with_every_input_mapped() -> Result<(), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let looped = callee.fori(x, fori_plan()?, vec![("shift".to_string(), s)])?;
    let loss = callee.sum(looped)?;
    let reverse = callee.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    let outputs = [reverse.gradients["x"], reverse.gradients["s"]];
    // Only x is mapped, yet the gradient with respect to the unmapped shift
    // differs per example, so the reverse body maps every input.
    let (graph, results) = batched(&reverse.graph, &[("x", true), ("s", false)], &outputs)?;
    assert!(results.iter().all(|(_, mapped)| *mapped));
    let members = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            TensorOp::ForiVjp {
                carry,
                output_cotangent,
                loop_plan,
                captures,
                group,
                ..
            } => Some((
                *carry,
                *output_cotangent,
                captures.clone(),
                *group,
                loop_plan,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(members.len(), 2, "one carry and one shift member");
    let (carry, cotangent, captures, group, loop_plan) = &members[0];
    for (other_carry, other_cotangent, other_captures, other_group, _) in &members[1..] {
        assert_eq!(
            (other_carry, other_cotangent, other_captures, other_group),
            (carry, cotangent, captures, group)
        );
    }
    assert!(is_broadcast_of(
        &graph,
        captures[0].1,
        graph.input_node_id("s")?
    ));
    let body = &loop_plan.body.plan;
    assert_eq!(body.input_shape("carry")?, vec![BATCH, 3]);
    assert_eq!(body.input_shape("shift")?, vec![BATCH, 3]);
    Ok(())
}

#[test]
fn nested_loops_batch_the_inner_region_inside_the_outer_body() -> Result<(), String> {
    // Outer body: carry -> fori(carry, shift = outer shift) * 0.5.
    let mut outer_body = TensorIr::new();
    let carry = outer_body.input("carry", vec![3])?;
    let shift = outer_body.input("shift", vec![3])?;
    let inner = outer_body.fori(carry, fori_plan()?, vec![("shift".to_string(), shift)])?;
    let half = outer_body.scalar_constant(0.5);
    let next = outer_body.mul(inner, half)?;
    let outer_plan =
        TensorForiExecutionPlan::new(0, 2, outer_body.compile_cpu(next)?, "carry", "index")?;
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let looped = callee.fori(x, outer_plan, vec![("shift".to_string(), s)])?;
    let (graph, results) = batched(&callee, &[("x", false), ("s", true)], &[looped])?;
    assert!(results[0].1);
    let TensorOp::Fori { loop_plan, .. } = &graph.nodes[results[0].0].op else {
        panic!("expected a batched outer fori node");
    };
    let inner = loop_plan
        .body
        .plan
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            TensorOp::Fori { loop_plan, .. } => Some(loop_plan),
            _ => None,
        })
        .expect("the outer body keeps its inner fori");
    assert_eq!(inner.body.plan.input_shape("carry")?, vec![BATCH, 3]);
    assert_eq!(inner.body.plan.input_shape("shift")?, vec![BATCH, 3]);
    Ok(())
}

/// `while sum(carry) > limit: carry *= scale`, or with `limit_only` the
/// predicate `limit < 0`, which reads only the limit.
fn while_plan(limit_only: bool) -> Result<TensorWhileExecutionPlan, String> {
    let mut predicate = TensorIr::new();
    let limit = predicate.input("limit", vec![])?;
    let running = if limit_only {
        let zero = predicate.scalar_constant(0.0);
        predicate.compare(limit, zero, TensorComparison::Less)?
    } else {
        let carry = predicate.input("carry", vec![3])?;
        let total = predicate.sum(carry)?;
        predicate.compare(total, limit, TensorComparison::Greater)?
    };
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![3])?;
    let scale = body.input("scale", vec![3])?;
    let next = body.mul(carry, scale)?;
    TensorWhileExecutionPlan::new(
        predicate.compile_cpu(running)?,
        body.compile_cpu(next)?,
        "carry",
    )
}

/// `while_plan(limit_only)` over inputs `x` (carry), `s` (scale), and `l`
/// (limit) batched with `mapped` inputs mapped: the batched graph, its
/// `While` node, and whether that node is mapped.
fn batched_while(
    limit_only: bool,
    mapped: &[(&str, bool)],
) -> Result<(TensorIr, TensorNodeId, bool), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let l = callee.input("l", vec![])?;
    let looped = callee.while_loop(
        x,
        while_plan(limit_only)?,
        vec![("scale".to_string(), s), ("limit".to_string(), l)],
    )?;
    let (graph, results) = batched(&callee, mapped, &[looped])?;
    Ok((graph, results[0].0, results[0].1))
}

/// The loop plan, carry operand, and captures of the `While` node `node`.
fn while_parts(
    graph: &TensorIr,
    node: TensorNodeId,
) -> (
    &TensorWhileExecutionPlan,
    TensorNodeId,
    &[(String, TensorNodeId)],
) {
    let TensorOp::While {
        loop_plan,
        carry,
        captures,
    } = &graph.nodes[node].op
    else {
        panic!("expected a batched while node");
    };
    (loop_plan, *carry, captures)
}

fn has_where(plan: &TensorExecutionPlan) -> bool {
    plan.nodes
        .iter()
        .any(|node| matches!(node.op, TensorOp::Where { .. }))
}

#[test]
fn a_while_with_an_unmapped_predicate_keeps_one_trip_count() -> Result<(), String> {
    // A mapped carry and scale, but the predicate reads only the unmapped
    // limit: the predicate region is unchanged and the body is not masked.
    let mapped = [("x", true), ("s", true), ("l", false)];
    let (graph, node, result_mapped) = batched_while(true, &mapped)?;
    let (plan, _, captures) = while_parts(&graph, node);
    assert!(result_mapped);
    assert_eq!(plan.carry_shape()?, vec![BATCH, 3]);
    assert!(plan.predicate_plan().output_shape()?.is_empty());
    assert!(!plan.predicate_plan().input_nodes.contains_key("carry"));
    assert!(!has_where(plan.body_plan()));
    let limit = captures.iter().find(|(name, _)| name == "limit").unwrap().1;
    assert!(
        graph.nodes[limit].shape.is_empty(),
        "the limit stays unbatched"
    );
    Ok(())
}

#[test]
fn a_mapped_predicate_maps_the_carry_and_masks_the_body() -> Result<(), String> {
    // Only the limit is mapped: the predicate is per example, so the carry is
    // mapped (broadcast from the unmapped initial carry), the predicate is
    // `any` over the batch, and the body selects per example.
    let mapped = [("x", false), ("s", false), ("l", true)];
    let (graph, node, result_mapped) = batched_while(false, &mapped)?;
    let (plan, carry, captures) = while_parts(&graph, node);
    assert!(result_mapped);
    assert_eq!(plan.carry_shape()?, vec![BATCH, 3]);
    assert!(plan.predicate_plan().output_shape()?.is_empty());
    assert!(has_where(plan.body_plan()));
    // The body recomputes the per-example predicate, so it reads the limit.
    assert_eq!(plan.body_plan().input_shape("limit")?, vec![BATCH]);
    let scale = captures.iter().find(|(name, _)| name == "scale").unwrap().1;
    assert_eq!(
        graph.nodes[scale].shape,
        vec![3],
        "the scale stays unbatched"
    );
    assert!(matches!(graph.nodes[carry].op, TensorOp::Broadcast { .. }));
    Ok(())
}

#[test]
fn a_masked_while_keeps_finished_examples_and_stops_when_all_are_done() -> Result<(), String> {
    // Halving `[4, 4, 4]` until its sum is at most the limit: examples with
    // limits 100, 6, 1, and 12 stop after 0, 1, 4, and 0 iterations.
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let l = callee.input("l", vec![])?;
    let looped = callee.while_loop(
        x,
        while_plan(false)?,
        vec![("scale".to_string(), s), ("limit".to_string(), l)],
    )?;
    let mut graph = TensorIr::new();
    let bindings = bind(
        &mut graph,
        &callee,
        &[("x", false), ("s", false), ("l", true)],
    )?;
    let batched = graph
        .inline_batched(&callee, &bindings, BATCH, &[looped])
        .map_err(|error| error.to_string())?;
    let inputs = BTreeMap::from([
        ("x".to_string(), DynamicTensor::new(vec![3], vec![4.0; 3])?),
        ("s".to_string(), DynamicTensor::new(vec![3], vec![0.5; 3])?),
        (
            "l".to_string(),
            DynamicTensor::new(vec![BATCH], vec![100.0, 6.0, 1.0, 12.0])?,
        ),
    ]);
    let result = graph.evaluate(batched[0].0, &inputs)?;
    let expected = [4.0, 2.0, 0.25, 4.0]
        .iter()
        .flat_map(|value| [*value; 3])
        .collect::<Vec<_>>();
    assert_eq!(result.data().to_vec(), expected);
    Ok(())
}

/// Two branch regions over the captures `value`, `scale`, and `w` (all
/// `[3]`): the final carry of [`scan_plan`] from `value`, in which `w` reaches
/// only the discarded outputs, and `value * w * scale`.
fn scan_carry_branches() -> Result<TensorCondExecutionPlan, String> {
    let mut on_true = TensorIr::new();
    let value = on_true.input("value", vec![3])?;
    let scale = on_true.input("scale", vec![3])?;
    let w = on_true.input("w", vec![3])?;
    let (carry, _) = on_true.scan(
        value,
        scan_plan()?,
        vec![("scale".to_string(), scale), ("w".to_string(), w)],
    )?;
    let mut on_false = TensorIr::new();
    let value = on_false.input("value", vec![3])?;
    let scale = on_false.input("scale", vec![3])?;
    let w = on_false.input("w", vec![3])?;
    let weighted = on_false.mul(value, w)?;
    let product = on_false.mul(weighted, scale)?;
    TensorCondExecutionPlan::new(on_true.compile_cpu(carry)?, on_false.compile_cpu(product)?)
}

/// `cond(p > 0, scan_carry_branches)` over the callee inputs `x`, `s`, `w`,
/// and the scalar `p`.
fn cond_callee() -> Result<(TensorIr, TensorNodeId), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![3])?;
    let s = callee.input("s", vec![3])?;
    let w = callee.input("w", vec![3])?;
    let p = callee.input("p", vec![])?;
    let zero = callee.scalar_constant(0.0);
    let predicate = callee.greater(p, zero)?;
    let branched = callee.cond_with_captures(
        predicate,
        scan_carry_branches()?,
        vec![
            ("value".to_string(), x),
            ("scale".to_string(), s),
            ("w".to_string(), w),
        ],
    )?;
    Ok((callee, branched))
}

#[test]
fn an_unmapped_predicate_keeps_one_cond_with_one_output_batchedness() -> Result<(), String> {
    let (callee, branched) = cond_callee()?;
    let (graph, results) = batched(
        &callee,
        &[("x", false), ("s", false), ("w", true), ("p", false)],
        &[branched],
    )?;
    let (node, mapped) = results[0];
    assert!(mapped);
    assert_eq!(graph.nodes[node].shape, vec![BATCH, 3]);
    let TensorOp::Cond {
        branches, captures, ..
    } = &graph.nodes[node].op
    else {
        panic!("an unmapped predicate keeps the lazy cond");
    };
    // Captures are bound as they are: only w is mapped.
    for (name, input) in [("value", "x"), ("scale", "s"), ("w", "w")] {
        let bound = captures
            .iter()
            .find(|(capture, _)| capture == name)
            .ok_or_else(|| format!("capture {name} is missing"))?;
        assert_eq!(bound.1, graph.input_node_id(input)?, "{name}");
    }
    for region in [&branches.on_true.plan, &branches.on_false.plan] {
        assert_eq!(region.input_shape("value")?, vec![3]);
        assert_eq!(region.input_shape("scale")?, vec![3]);
        assert_eq!(region.input_shape("w")?, vec![BATCH, 3]);
    }
    assert_eq!(branches.output_shape()?, vec![BATCH, 3]);
    // w does not reach the scan carry, so the true branch alone would stay
    // unmapped; it is broadcast to the false branch's batchedness.
    let on_true = &branches.on_true.plan;
    assert!(matches!(
        on_true.nodes[on_true.output_node_id].op,
        TensorOp::Broadcast { .. }
    ));
    Ok(())
}

#[test]
fn a_mapped_predicate_selects_between_both_batched_branches() -> Result<(), String> {
    let (callee, branched) = cond_callee()?;
    let (graph, results) = batched(
        &callee,
        &[("x", true), ("s", false), ("w", false), ("p", true)],
        &[branched],
    )?;
    let (node, mapped) = results[0];
    assert!(mapped);
    assert_eq!(graph.nodes[node].shape, vec![BATCH, 3]);
    let TensorOp::Where { condition, .. } = &graph.nodes[node].op else {
        panic!("a mapped predicate selects with where");
    };
    // The [B] predicate is broadcast over the output's trailing axis.
    assert_eq!(graph.nodes[*condition].shape, vec![BATCH, 1]);
    assert!(graph
        .nodes
        .iter()
        .all(|node| !matches!(node.op, TensorOp::Cond { .. })));
    // The mapped x enters each branch once behind a gradient mask; the
    // unmapped s and w are bound as they are.
    let frozen = |name: &str| -> Result<usize, String> {
        let input = graph.input_node_id(name)?;
        Ok(graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op, TensorOp::StopGradient { input: source } if source == input))
            .count())
    };
    assert_eq!(frozen("x")?, 2);
    assert_eq!(frozen("s")? + frozen("w")?, 0);
    Ok(())
}
