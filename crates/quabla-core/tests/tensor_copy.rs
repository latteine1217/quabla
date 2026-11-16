use std::collections::BTreeMap;
use std::sync::Arc;

use quabla_core::tensor_ir::{
    DynamicTensor, SymbolicCotangent, TensorCustomRule, TensorDType, TensorIr,
};

#[test]
fn vjp_retains_primal_metadata_bits_and_compiled_program() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![2], TensorDType::F32)?;
    let mask = graph.input_typed("mask", vec![2], TensorDType::Bool)?;
    graph.input("unused", vec![])?;
    let two = graph.scalar_constant(2.0);
    let four = graph.mul(two, two)?;
    let scaled = graph.mul(x, four)?;
    let zero = graph.scalar_constant(-0.0);
    let selected = graph.where_select(mask, scaled, zero)?;
    let loss = graph.sum(selected)?;
    graph.broadcast_to(zero, vec![2])?;
    graph.constant(DynamicTensor::new(vec![2], vec![-0.0, f64::NAN])?, false);

    let nodes = (0..graph.node_count()).collect::<Vec<_>>();
    let vjp = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    assert_eq!(vjp.primals.len(), nodes.len());
    for (source, copied) in nodes.iter().zip(&vjp.primals) {
        assert_eq!(graph.node_shape(*source)?, vjp.graph.node_shape(*copied)?);
        assert_eq!(graph.node_dtype(*source)?, vjp.graph.node_dtype(*copied)?);
        assert_eq!(graph.node_weak(*source)?, vjp.graph.node_weak(*copied)?);
    }
    let (source, _) = graph.compile_cpu_many(&nodes)?;
    let (copied, _) = vjp.graph.compile_cpu_many(&vjp.primals)?;
    assert_eq!(source.lower_text(), copied.lower_text());
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::with_dtype(vec![2], vec![1.25, -2.0], TensorDType::F32)?,
        ),
        (
            "mask".to_string(),
            DynamicTensor::with_dtype(vec![2], vec![1.0, 0.0], TensorDType::Bool)?,
        ),
        ("unused".to_string(), DynamicTensor::new(vec![], vec![7.0])?),
    ]);
    for (expected, actual) in source
        .evaluate_many(&inputs)?
        .iter()
        .zip(copied.evaluate_many(&inputs)?)
    {
        let expected = expected
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let actual = actual
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(expected, actual);
    }
    assert_eq!(
        vjp.graph
            .evaluate(vjp.gradients["x"], &inputs)?
            .data()
            .as_ref(),
        &[4.0, 0.0]
    );
    Ok(())
}

#[test]
fn copied_custom_calls_keep_residuals_and_higher_derivatives_separate() -> Result<(), String> {
    let mut forward = TensorIr::new();
    let x = forward.input("x", vec![2])?;
    let square = forward.mul(x, x)?;
    let mut backward = TensorIr::new();
    let residual = backward.input("residual", vec![2])?;
    let cotangent = backward.input("g", vec![2])?;
    let three = backward.scalar_constant(3.0);
    let scaled = backward.mul(residual, three)?;
    let gradient = backward.mul(scaled, cotangent)?;
    // A deliberately different derivative proves the custom rule survives copying.
    let rule = Arc::new(TensorCustomRule::new(
        "three_x".to_string(),
        forward,
        vec!["x".to_string()],
        1,
        vec![square, x],
        backward,
        vec!["residual".to_string()],
        vec!["g".to_string()],
        vec![Some(gradient)],
        None,
        false,
    )?);
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![2])?;
    let square = callee.mul(x, x)?;
    let output = callee.custom(rule, &[square], &[x])?[0];
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    let y = graph.input("y", vec![2])?;
    let first = graph.inline(&callee, &BTreeMap::from([("x".to_string(), x)]), &[output])?[0];
    let second = graph.inline(&callee, &BTreeMap::from([("x".to_string(), y)]), &[output])?[0];
    let sum = graph.add(first, second)?;
    let loss = graph.sum(sum)?;
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![2], vec![2.0, -1.0])?,
        ),
        (
            "y".to_string(),
            DynamicTensor::new(vec![2], vec![3.0, 4.0])?,
        ),
    ]);
    let mut vjp = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    assert_eq!(
        vjp.graph
            .evaluate(vjp.primals[loss], &inputs)?
            .data()
            .as_ref(),
        &[30.0]
    );
    assert_eq!(
        vjp.graph
            .evaluate(vjp.gradients["x"], &inputs)?
            .data()
            .as_ref(),
        &[6.0, -3.0]
    );
    assert_eq!(
        vjp.graph
            .evaluate(vjp.gradients["y"], &inputs)?
            .data()
            .as_ref(),
        &[9.0, 12.0]
    );
    let gradient_sum = vjp.graph.sum(vjp.gradients["x"])?;
    let second = vjp
        .graph
        .symbolic_vjp_many(&[(gradient_sum, SymbolicCotangent::Ones)])?;
    assert_eq!(
        second
            .graph
            .evaluate(second.gradients["x"], &inputs)?
            .data()
            .as_ref(),
        &[3.0, 3.0]
    );
    assert_eq!(
        second
            .graph
            .evaluate(second.gradients["y"], &inputs)?
            .data()
            .as_ref(),
        &[0.0, 0.0]
    );
    Ok(())
}
