use std::collections::BTreeMap;

use quabla::{InlineBinding, TensorTraceGraph, TraceTensor};
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

/// The splice behind `quabla.vmap`: a per-example graph batched into the
/// caller's graph, with mapped and unmapped bindings and flags per output.
#[test]
fn inline_batched_splices_a_per_example_graph_over_the_batch() -> Result<(), String> {
    let callee = TensorTraceGraph::new();
    let row = callee.add_input("row", vec![1, 2], TensorDType::F64)?;
    let weight = callee.add_input("weight", vec![2, 1], TensorDType::F64)?;
    let product = row.try_matmul(&weight)?;
    let shared = weight.try_matmul(&row)?.try_matmul(&weight)?;
    let unmapped = TraceTensor::try_concat(&[weight.clone(), weight.clone()], 0)?;

    let graph = TensorTraceGraph::new();
    let rows = graph.add_input("rows", vec![3, 1, 2], TensorDType::F64)?;
    let weight_node = graph.add_input("weight", vec![2, 1], TensorDType::F64)?;
    let spliced = callee
        .inline_batched_into(
            &["row".to_string(), "weight".to_string()],
            &[rows, weight_node],
            &[true, false],
            3,
            &[product, shared, unmapped],
        )
        .map_err(|error| error.to_string())?;
    assert_eq!(
        spliced
            .iter()
            .map(|(_, mapped)| *mapped)
            .collect::<Vec<_>>(),
        vec![true, true, false]
    );
    let outputs = spliced
        .into_iter()
        .map(|(tensor, _)| tensor)
        .collect::<Vec<_>>();
    let executable = graph.compile_cpu_many(&outputs, vec!["rows".into(), "weight".into()])?;
    let values = executable.execute(vec![
        DynamicTensor::new(vec![3, 1, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5])?,
        matrix(2, 1, &[0.5, -2.0])?,
    ])?;
    assert_eq!(values[0].shape(), &[3, 1, 1]);
    assert_eq!(values[0].data(), &[-3.5, -6.5, -1.5]);
    assert_eq!(values[1].shape(), &[3, 2, 1]);
    // weight @ (row @ weight) per example: the row products scale weight.
    assert_eq!(values[1].data(), &[-1.75, 7.0, -3.25, 13.0, -0.75, 3.0]);
    assert_eq!(values[2].shape(), &[4, 1]);
    Ok(())
}

#[test]
fn inline_batched_reports_unbatchable_ops_as_unsupported() -> Result<(), String> {
    use quabla_core::tensor_ir::BatchingError;
    let callee = TensorTraceGraph::new();
    let matrix_input = callee.add_input("matrix", vec![2, 2], TensorDType::F64)?;
    let rhs = callee.add_input("rhs", vec![2, 1], TensorDType::F64)?;
    let solved = matrix_input.try_solve(&rhs)?;
    let graph = TensorTraceGraph::new();
    let matrices = graph.add_input("matrices", vec![4, 2, 2], TensorDType::F64)?;
    let rhs_node = graph.add_input("rhs", vec![2, 1], TensorDType::F64)?;
    let error = callee
        .inline_batched_into(
            &["matrix".to_string(), "rhs".to_string()],
            &[matrices, rhs_node.clone()],
            &[true, false],
            4,
            std::slice::from_ref(&solved),
        )
        .map(|_| ())
        .expect_err("a mapped solve must be rejected");
    assert_eq!(error, BatchingError::Unsupported { op: "solve" });
    let other = TensorTraceGraph::new();
    let stray = other.add_input("matrix", vec![2, 2], TensorDType::F64)?;
    let error = callee
        .inline_batched_into(
            &["matrix".to_string(), "rhs".to_string()],
            &[stray, rhs_node],
            &[false, false],
            4,
            &[solved],
        )
        .map(|_| ())
        .expect_err("bindings of two traces must be rejected");
    assert!(error.to_string().contains("different traces"), "{error}");
    Ok(())
}

/// Closed-over tracers (slice S4b): while an inner trace is in progress, a
/// tracer of the enclosing trace that meets it becomes one capture input of
/// the inner graph, and inlining binds that input back to the tracer.
#[test]
fn tracers_of_an_enclosing_trace_lift_into_the_inner_trace() -> Result<(), String> {
    let outer = TensorTraceGraph::new();
    let w = outer.add_input("w", vec![1, 1], TensorDType::F64)?;
    outer.begin_trace()?;
    let inner = TensorTraceGraph::new();
    let x = inner.add_input("x", vec![1, 1], TensorDType::F64)?;
    inner.begin_trace()?;
    // Either operand order lifts `w` into the inner graph, once.
    let product = x.try_matmul(&w)?.try_matmul(&w.try_matmul(&x)?)?;
    let returned = inner.lift(&w)?;
    let error = outer
        .end_trace()
        .expect_err("only the innermost trace can end");
    assert!(error.contains("innermost"), "{error}");
    let captures = inner.end_trace()?;
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].0, "__quabla_capture/0");

    // x * w * (w * x) with x = 2 bound as a constant and w from the outer
    // trace, plus the lifted w returned as is.
    let spliced = inner.inline_into(
        &["x".to_string(), captures[0].0.clone()],
        &[
            InlineBinding::Constant(matrix(1, 1, &[2.0])?),
            InlineBinding::Traced(captures[0].1.clone()),
        ],
        &[product, returned],
    )?;
    outer.end_trace()?;
    let executable = outer.compile_cpu_many(&spliced, vec!["w".into()])?;
    let values = executable.execute(vec![matrix(1, 1, &[3.0])?])?;
    assert_eq!(values[0].data(), &[36.0]);
    assert_eq!(values[1].data(), &[3.0]);
    Ok(())
}

/// A tracer whose trace ended, or one of a graph that is not a trace in
/// progress, cannot be lifted: the error names the escape.
#[test]
fn tracers_of_finished_traces_are_not_lifted() -> Result<(), String> {
    let finished = TensorTraceGraph::new();
    let stale = finished.add_input("a", vec![2], TensorDType::F64)?;
    finished.begin_trace()?;
    finished.end_trace()?;
    let current = TensorTraceGraph::new();
    let x = current.add_input("x", vec![2], TensorDType::F64)?;
    current.begin_trace()?;
    let errors = [
        TraceTensor::try_concat(&[x.clone(), stale.clone()], 0)
            .map(|_| ())
            .expect_err("a finished trace's tracer must not be lifted"),
        current
            .lift(&stale)
            .map(|_| ())
            .expect_err("a finished trace's result must not be lifted"),
    ];
    current.end_trace()?;
    for error in errors {
        assert!(error.contains("escaped"), "{error}");
    }
    Ok(())
}
