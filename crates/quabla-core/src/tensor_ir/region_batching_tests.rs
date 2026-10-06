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

#[test]
fn cond_and_while_regions_report_that_they_have_no_rule_yet() -> Result<(), String> {
    let mut on_true = TensorIr::new();
    let value = on_true.input("value", vec![])?;
    let squared = on_true.mul(value, value)?;
    let mut on_false = TensorIr::new();
    let value = on_false.input("value", vec![])?;
    let negated = on_false.sub(value, value)?;
    let branches = TensorCondExecutionPlan::new(
        on_true.compile_cpu(squared)?,
        on_false.compile_cpu(negated)?,
    )?;
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![])?;
    let zero = callee.scalar_constant(0.0);
    let predicate = callee.greater(x, zero)?;
    let branched =
        callee.cond_with_captures(predicate, branches, vec![("value".to_string(), x)])?;
    let mut graph = TensorIr::new();
    let bindings = bind(&mut graph, &callee, &[("x", true)])?;
    let error = graph
        .inline_batched(&callee, &bindings, BATCH, &[branched])
        .expect_err("cond has no batching rule yet");
    assert_eq!(error, BatchingError::Unsupported { op: "cond" });
    assert!(error.to_string().contains("where"), "{error}");
    Ok(())
}
