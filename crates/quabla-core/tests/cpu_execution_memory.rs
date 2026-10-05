use std::collections::BTreeMap;

use quabla_core::tensor_ir::{
    DynamicTensor, TensorDType, TensorForiExecutionPlan, TensorFusionRegion, TensorIr,
    TensorScanExecutionPlan,
};

#[test]
fn fusion_user_deduplication_preserves_duplicate_operands_and_wide_fanout() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    let shared = graph.sin(x)?;
    let mut outputs = Vec::new();
    for index in 0..128 {
        let branch = graph.powi(shared, index + 2)?;
        outputs.push(graph.add(branch, branch)?);
    }
    let (plan, _) = graph.compile_cpu_many(&outputs)?;
    let regions = plan.fusion_regions();
    assert_eq!(regions.len(), outputs.len());
    for (index, region) in regions.iter().enumerate() {
        let branch = 2 + 2 * index;
        assert_eq!(
            region,
            &TensorFusionRegion {
                output_node_id: branch + 1,
                node_ids: vec![branch, branch + 1],
                input_node_ids: vec![1],
            }
        );
    }
    Ok(())
}

#[test]
fn numerical_vjp_retains_input_seeds_and_accumulates_duplicate_operands() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    let _unused = graph.input("unused", vec![2])?;
    let square = graph.mul(x, x)?;
    let doubled = graph.add(square, square)?;
    let inputs = BTreeMap::from([
        ("x".into(), DynamicTensor::new(vec![2], vec![2.0, -3.0])?),
        ("unused".into(), DynamicTensor::filled(vec![2], 7.0)?),
    ]);
    let seed = DynamicTensor::filled(vec![2], 1.0)?;
    let (values, gradients) = graph.value_and_vjp_many(
        &[(doubled, seed.clone()), (doubled, seed.clone()), (x, seed)],
        &inputs,
    )?;
    assert_eq!(values[0].data().as_ref(), &[8.0, 18.0]);
    assert_eq!(values[0], values[1]);
    assert_eq!(values[2], inputs["x"]);
    assert_eq!(gradients["x"].data().as_ref(), &[17.0, -23.0]);
    assert_eq!(gradients["unused"].data().as_ref(), &[0.0, 0.0]);
    Ok(())
}

#[test]
fn numerical_vjp_moves_joint_scan_seeds_without_losing_siblings() -> Result<(), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![])?;
    let index = body.input("index", vec![])?;
    let scale = body.input("scale", vec![])?;
    let scaled = body.mul(carry, scale)?;
    let next = body.add(scaled, index)?;
    let scan = TensorScanExecutionPlan::new(
        0,
        3,
        body.compile_cpu_many(&[next, next])?.0,
        "carry",
        "index",
    )?;
    let mut graph = TensorIr::new();
    let initial = graph.input("initial", vec![])?;
    let scale = graph.input("scale", vec![])?;
    let (carry, outputs) = graph.scan(initial, scan, vec![("scale".into(), scale)])?;
    let inputs = BTreeMap::from([
        ("initial".into(), DynamicTensor::filled(vec![], 1.0)?),
        ("scale".into(), DynamicTensor::filled(vec![], 2.0)?),
    ]);
    let (_, gradients) = graph.value_and_vjp_many(
        &[
            (carry, DynamicTensor::filled(vec![], 1.0)?),
            (outputs, DynamicTensor::filled(vec![3], 1.0)?),
            (carry, DynamicTensor::filled(vec![], 1.0)?),
        ],
        &inputs,
    )?;
    assert_eq!(gradients["initial"].data().as_ref(), &[30.0]);
    assert_eq!(gradients["scale"].data().as_ref(), &[44.0]);
    Ok(())
}

#[test]
fn fused_memo_keeps_lazy_where_broadcasting_and_per_node_rounding() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![2, 1], TensorDType::F32)?;
    let y = graph.input_typed("y", vec![1, 3], TensorDType::F32)?;
    let sum = graph.add(x, y)?;
    let doubled = graph.add(sum, sum)?;
    let invalid = graph.sqrt(doubled)?;
    let yes = graph.scalar_constant(1.0);
    let output = graph.where_select(yes, doubled, invalid)?;
    let plan = graph.compile_cpu(output)?;
    assert!(plan.uses_fused_elementwise_kernel());
    let inputs = BTreeMap::from([
        ("x".into(), DynamicTensor::new(vec![2, 1], vec![0.1, 1.2])?),
        (
            "y".into(),
            DynamicTensor::new(vec![1, 3], vec![0.2, -0.3, 3.4])?,
        ),
    ]);
    let value = plan.evaluate(&inputs)?;
    assert_eq!(value.shape(), &[2, 3]);
    assert_eq!(value.dtype(), TensorDType::F32);
    let mut expected = Vec::new();
    for x in [0.1_f32, 1.2] {
        for y in [0.2_f32, -0.3, 3.4] {
            let sum = (f64::from(x) + f64::from(y)) as f32;
            expected.push(f64::from((f64::from(sum) + f64::from(sum)) as f32));
        }
    }
    assert_eq!(value.data().as_ref(), expected);
    let no = graph.scalar_constant(0.0);
    let bad_output = graph.where_select(no, doubled, invalid)?;
    let bad_values = graph.compile_cpu(bad_output)?.evaluate(&inputs)?;
    assert!(bad_values.data()[1].is_nan());
    let mut missing = inputs.clone();
    missing.remove("y");
    assert_eq!(plan.evaluate(&missing).unwrap_err(), "missing input \"y\"");
    Ok(())
}

#[test]
fn forward_outputs_and_buffer_slots_protect_early_and_reshape_values() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2, 2])?;
    let early = graph.sin(x)?;
    let alias = graph.reshape(early, vec![4])?;
    let successor = graph.cos(early)?;
    let last = graph.sin(successor)?;
    let total = graph.sum(last)?;
    // The second output is consumed later and the last output aliases its storage root.
    let (plan, ids) = graph.compile_cpu_many(&[total, early, last, early, alias])?;
    let inputs = BTreeMap::from([(
        "x".into(),
        DynamicTensor::new(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])?,
    )]);
    let values = plan.evaluate_many(&inputs)?;
    assert_eq!(values[0], graph.evaluate(total, &inputs)?);
    assert_eq!(values[1], graph.evaluate(early, &inputs)?);
    assert_eq!(values[2], graph.evaluate(last, &inputs)?);
    assert_eq!(values[1], values[3]);
    assert_eq!(values[4].shape(), &[4]);
    assert_eq!(values[4].data().as_ref(), values[1].data().as_ref());
    assert_eq!(plan.evaluate(&inputs)?, values[0]);
    let buffers = plan.buffer_plan()?;
    assert_eq!(buffers.node_slots[ids[1]], buffers.node_slots[ids[4]]);
    assert_ne!(buffers.node_slots[ids[1]], buffers.node_slots[ids[2]]);
    assert!(buffers.node_aliases[ids[4]].is_some());
    Ok(())
}

#[test]
fn forward_loop_results_match_taped_paths_and_validation() -> Result<(), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![3])?;
    let index = body.input("index", vec![])?;
    let scale = body.input("scale", vec![])?;
    let scaled = body.mul(carry, scale)?;
    let next = body.add(scaled, index)?;
    let output = body.sin(next)?;
    let fori = TensorForiExecutionPlan::new(0, 7, body.compile_cpu(next)?, "carry", "index")?;
    let scan = TensorScanExecutionPlan::new(
        0,
        7,
        body.compile_cpu_many(&[next, output])?.0,
        "carry",
        "index",
    )?;
    let captures = BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 1.1)?)]);
    let initial = DynamicTensor::new(vec![3], vec![0.1, 0.2, -0.3])?;
    let forward = fori.evaluate(initial.clone(), &captures)?;
    let (taped, _) = fori.evaluate_with_tape(initial.clone(), &captures)?;
    assert_eq!(forward, taped);
    let forward = scan.evaluate(initial.clone(), &captures)?;
    let (carry, outputs, _) = scan.evaluate_with_tape(initial.clone(), &captures)?;
    assert_eq!(forward, (carry, outputs));
    assert_eq!(forward.1.shape(), &[7, 3]);
    let empty_fori = TensorForiExecutionPlan::new(4, 4, body.compile_cpu(next)?, "carry", "index")?;
    assert_eq!(empty_fori.evaluate(initial.clone(), &captures)?, initial);
    let empty_scan = TensorScanExecutionPlan::new(
        4,
        4,
        body.compile_cpu_many(&[next, output])?.0,
        "carry",
        "index",
    )?;
    assert_eq!(
        empty_scan.evaluate(initial.clone(), &captures).unwrap_err(),
        empty_scan
            .evaluate_with_tape(initial.clone(), &captures)
            .unwrap_err()
    );
    let bad = DynamicTensor::filled(vec![2], 0.0)?;
    assert_eq!(
        fori.evaluate(bad.clone(), &captures).unwrap_err(),
        fori.evaluate_with_tape(bad, &captures).unwrap_err()
    );
    assert_eq!(
        scan.evaluate(initial.clone(), &BTreeMap::new())
            .unwrap_err(),
        scan.evaluate_with_tape(initial, &BTreeMap::new())
            .unwrap_err()
    );
    Ok(())
}

#[test]
fn grouped_scan_results_survive_interleaved_consumers() -> Result<(), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![2])?;
    let two = body.scalar_constant(2.0);
    let next = body.mul(carry, two)?;
    let output = body.sin(next)?;
    let scan = TensorScanExecutionPlan::new(
        0,
        3,
        body.compile_cpu_many(&[next, output])?.0,
        "carry",
        "index",
    )?;
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    let (carry, outputs) = graph.scan(x, scan.clone(), vec![])?;
    let loss = graph.sum(outputs)?;
    let reshaped = graph.reshape(outputs, vec![6])?;
    let carry_loss = graph.sum(carry)?;
    let (plan, _) = graph.compile_cpu_many(&[loss, carry, carry_loss, outputs, reshaped, carry])?;
    let initial = DynamicTensor::new(vec![2], vec![0.25, -0.5])?;
    let expected = scan.evaluate(initial.clone(), &BTreeMap::new())?;
    let values = plan.evaluate_many(&BTreeMap::from([("x".into(), initial)]))?;
    assert_eq!(values[1], expected.0);
    assert_eq!(values[3], expected.1);
    assert_eq!(values[3].data().as_ref(), values[4].data().as_ref());
    assert_eq!(values[1], values[5]);
    assert_eq!(values[0].data()[0], values[3].data().iter().sum::<f64>());
    Ok(())
}
