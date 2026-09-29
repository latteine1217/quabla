use std::collections::BTreeMap;

use quabla::TensorTraceGraph;
use quabla_core::tensor_ir::{DynamicTensor, SymbolicCotangent, TensorDType};

fn matrix(rows: usize, columns: usize, data: &[f64]) -> Result<DynamicTensor, String> {
    DynamicTensor::new(vec![rows, columns], data.to_vec())
}

/// The staged reverse transform behind `quabla.vjp`: one program returns the
/// retained primal and every gradient for a runtime cotangent input.
#[test]
fn staged_vjp_compiles_value_and_gradients_into_one_program() -> Result<(), String> {
    let graph = TensorTraceGraph::new();
    let a = graph.add_input("a", vec![2, 3], TensorDType::F64)?;
    let b = graph.add_input("b", vec![3, 2], TensorDType::F64)?;
    let product = a.try_matmul(&b)?;
    let (transformed, retained, gradients) = graph.symbolic_vjp_many(
        &[(
            product.clone(),
            SymbolicCotangent::Input("product_bar".to_string()),
        )],
        &[product],
    )?;
    assert_eq!(
        gradients.keys().collect::<Vec<_>>(),
        vec!["a", "b"],
        "one gradient per input"
    );
    let executable = transformed.compile_cpu_many(
        &[
            retained[0].clone(),
            gradients["a"].clone(),
            gradients["b"].clone(),
        ],
        vec!["a".into(), "b".into(), "product_bar".into()],
    )?;
    let outputs = executable.execute(vec![
        matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?,
        matrix(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?,
        matrix(2, 2, &[1.0, 0.5, -1.0, 2.0])?,
    ])?;
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].data(), &[58.0, 64.0, 139.0, 154.0]);
    assert_eq!(outputs[1].data(), &[11.0, 14.0, 17.0, 9.0, 11.0, 13.0]);
    assert_eq!(outputs[2].data(), &[-3.0, 8.5, -3.0, 11.0, -3.0, 13.5]);

    let error = executable
        .execute(vec![matrix(2, 3, &[0.0; 6])?])
        .expect_err("a missing positional input must be rejected");
    assert!(error.contains("expects 3 inputs, got 1"), "{error}");
    Ok(())
}

/// The staged forward transform behind `quabla.jvp`, with a ones-seeded
/// reverse transform nested under it, as `quabla.hessian` composes them.
#[test]
fn staged_jvp_over_a_ones_seeded_vjp_gives_hessian_columns() -> Result<(), String> {
    let graph = TensorTraceGraph::new();
    let x = graph.add_input("x", vec![1, 2], TensorDType::F32)?;
    let gram = graph.add_input("g", vec![2, 2], TensorDType::F32)?;
    let column = graph.add_input("xt", vec![2, 1], TensorDType::F32)?;
    // loss = x g xt: d loss / d x = (g xt)^T, whose derivative along xt is (g xt_dot)^T.
    let loss = x.try_matmul(&gram)?.try_matmul(&column)?;
    let (reverse, _, gradients) =
        graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)], &[])?;
    let (forward, values, tangents) = reverse.symbolic_jvp_many(
        &[gradients["x"].clone()],
        &BTreeMap::from([("xt".to_string(), "xt_dot".to_string())]),
    )?;
    let executable = forward.compile_cpu_many(
        &[values[0].clone(), tangents[0].clone()],
        vec!["x".into(), "g".into(), "xt".into(), "xt_dot".into()],
    )?;
    let outputs = executable.execute(vec![
        DynamicTensor::with_dtype(vec![1, 2], vec![1.0, 2.0], TensorDType::F32)?,
        DynamicTensor::with_dtype(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], TensorDType::F32)?,
        DynamicTensor::with_dtype(vec![2, 1], vec![5.0, 6.0], TensorDType::F32)?,
        DynamicTensor::with_dtype(vec![2, 1], vec![0.0, 1.0], TensorDType::F32)?,
    ])?;
    assert_eq!(outputs[0].data(), &[17.0, 39.0]);
    assert_eq!(outputs[0].dtype(), TensorDType::F32);
    assert_eq!(outputs[1].data(), &[2.0, 4.0]);
    assert_eq!(outputs[1].dtype(), TensorDType::F32);
    Ok(())
}

#[test]
fn staging_rejects_tracers_of_another_graph() -> Result<(), String> {
    let graph = TensorTraceGraph::new();
    let other = TensorTraceGraph::new();
    let x = graph.add_input("x", vec![2], TensorDType::F64)?;
    let stray = other.add_input("x", vec![2], TensorDType::F64)?;
    for error in [
        graph
            .compile_cpu_many(&[x.clone(), stray.clone()], vec!["x".into()])
            .map(|_| ())
            .expect_err("a foreign output must be rejected"),
        graph
            .symbolic_vjp_many(
                &[(x.clone(), SymbolicCotangent::Ones)],
                std::slice::from_ref(&stray),
            )
            .map(|_| ())
            .expect_err("a foreign retained tracer must be rejected"),
        graph
            .symbolic_jvp_many(
                &[stray],
                &BTreeMap::from([("x".to_string(), "x_dot".to_string())]),
            )
            .map(|_| ())
            .expect_err("a foreign JVP output must be rejected"),
    ] {
        assert!(error.contains("different graph"), "{error}");
    }
    Ok(())
}
