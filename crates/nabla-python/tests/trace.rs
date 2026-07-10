use std::collections::HashMap;

use nabla::{IrAttrValue, PyMatrix, TraceGraph};

#[test]
fn trace_graph_records_inputs_and_matmul_nodes() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;
    let b = graph.add_input("b", (3, 2))?;

    let c = a.try_matmul(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.id(), 2);
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 3)",
            "1 input b shape=(3, 2)",
            "2 matmul inputs=[0, 1] shape=(2, 2)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir.len(), 3);
    assert_eq!(ir[0].id, 0);
    assert_eq!(ir[0].op, "input");
    assert_eq!(ir[0].shape, (2, 3));
    assert_eq!(ir[0].inputs, Vec::<usize>::new());
    assert_eq!(ir[0].name.as_deref(), Some("a"));
    assert_eq!(ir[2].id, 2);
    assert_eq!(ir[2].op, "matmul");
    assert_eq!(ir[2].shape, (2, 2));
    assert_eq!(ir[2].inputs, vec![0, 1]);
    assert_eq!(ir[2].name, None);

    let vjp = graph.vjp_ir_nodes(c.id())?;
    assert_eq!(vjp.len(), 5);
    assert_eq!(vjp[0].id, 0);
    assert_eq!(vjp[0].op, "cotangent_seed");
    assert_eq!(vjp[0].shape, (2, 2));
    assert_eq!(vjp[0].inputs, Vec::<usize>::new());
    assert_eq!(vjp[0].target, Some(2));
    assert_eq!(vjp[1].op, "transpose");
    assert_eq!(vjp[1].shape, (2, 3));
    assert_eq!(vjp[1].inputs, vec![1]);
    assert_eq!(vjp[2].op, "matmul");
    assert_eq!(vjp[2].shape, (2, 3));
    assert_eq!(vjp[2].inputs, vec![0, 1]);
    assert_eq!(vjp[2].target, Some(0));
    assert_eq!(vjp[3].op, "transpose");
    assert_eq!(vjp[3].shape, (3, 2));
    assert_eq!(vjp[3].inputs, vec![0]);
    assert_eq!(vjp[4].op, "matmul");
    assert_eq!(vjp[4].shape, (3, 2));
    assert_eq!(vjp[4].inputs, vec![3, 0]);
    assert_eq!(vjp[4].target, Some(1));

    Ok(())
}

#[test]
fn trace_graph_evaluates_matmul_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;
    let b = graph.add_input("b", (3, 2))?;
    let c = a.try_matmul(&b)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?,
    );
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0, 0.5], vec![-1.0, 2.0]])?;

    let gradients = graph.evaluate_vjp_cpu(c.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![11.0, 14.0, 17.0], vec![9.0, 11.0, 13.0]]
    );
    assert_eq!(
        gradients["b"].to_rows(),
        vec![vec![-3.0, 8.5], vec![-3.0, 11.0], vec![-3.0, 13.5]]
    );

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_add_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let b = graph.add_input("b", (2, 2))?;

    let c = a.try_add(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.id(), 2);
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 2)",
            "1 input b shape=(2, 2)",
            "2 add inputs=[0, 1] shape=(2, 2)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir[2].op, "add");
    assert_eq!(ir[2].shape, (2, 2));
    assert_eq!(ir[2].inputs, vec![0, 1]);

    let vjp = graph.vjp_ir_nodes(c.id())?;
    assert_eq!(vjp.len(), 3);
    assert_eq!(vjp[0].op, "cotangent_seed");
    assert_eq!(vjp[0].target, Some(2));
    assert_eq!(vjp[1].op, "identity");
    assert_eq!(vjp[1].target, Some(0));
    assert_eq!(vjp[2].op, "identity");
    assert_eq!(vjp[2].target, Some(1));

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?,
    );
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![0.5, 1.5], vec![2.5, 3.5]])?;

    let gradients = graph.evaluate_vjp_cpu(c.id(), inputs, &output_cotangent)?;

    assert_eq!(gradients["a"].to_rows(), output_cotangent.to_rows());
    assert_eq!(gradients["b"].to_rows(), output_cotangent.to_rows());

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_sub_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let b = graph.add_input("b", (2, 2))?;

    let c = a.try_sub(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.id(), 2);
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 2)",
            "1 input b shape=(2, 2)",
            "2 sub inputs=[0, 1] shape=(2, 2)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir[2].op, "sub");
    assert_eq!(ir[2].shape, (2, 2));
    assert_eq!(ir[2].inputs, vec![0, 1]);

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?,
    );
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![0.5, 1.5], vec![2.5, 3.5]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;

    let gradients = graph.evaluate_vjp_cpu(c.id(), inputs, &output_cotangent)?;

    assert_eq!(gradients["a"].to_rows(), output_cotangent.to_rows());
    assert_eq!(
        gradients["b"].to_rows(),
        vec![vec![-2.0, -3.0], vec![-4.0, -5.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_mul_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let b = graph.add_input("b", (2, 2))?;

    let c = a.try_mul(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.id(), 2);
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 2)",
            "1 input b shape=(2, 2)",
            "2 mul inputs=[0, 1] shape=(2, 2)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir[2].op, "mul");
    assert_eq!(ir[2].shape, (2, 2));
    assert_eq!(ir[2].inputs, vec![0, 1]);

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?,
    );
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![0.5, 1.5], vec![2.5, 3.5]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;

    let gradients = graph.evaluate_vjp_cpu(c.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![1.0, 4.5], vec![10.0, 17.5]]
    );
    assert_eq!(
        gradients["b"].to_rows(),
        vec![vec![2.0, 6.0], vec![12.0, 20.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_where_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;
    let zero = a.try_mul_scalar(0.0)?;
    let mask = a.try_gt(&zero)?;
    let leak = a.try_mul_scalar(0.1)?;
    let output = nabla::TraceMatrix::try_where(&mask, &a, &leak)?;
    let loss = output.try_sum(None)?;

    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 3)",
            "1 scalar_constant 0 shape=(2, 3)",
            "2 mul inputs=[0, 1] shape=(2, 3)",
            "3 gt inputs=[0, 2] shape=(2, 3)",
            "4 scalar_constant 0.1 shape=(2, 3)",
            "5 mul inputs=[0, 4] shape=(2, 3)",
            "6 where inputs=[3, 0, 5] shape=(2, 3)",
            "7 sum axis=None inputs=[6] shape=(1, 1)",
        ]
    );

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![-1.0, 0.5, 2.0], vec![3.0, -4.0, 5.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0]])?;

    let gradients = graph.evaluate_vjp_cpu(loss.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![0.1, 1.0, 1.0], vec![1.0, 0.1, 1.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_sum_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;

    let loss = a.try_sum(None)?;

    assert_eq!(loss.dims(), (1, 1));
    assert_eq!(loss.id(), 1);
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 2)",
            "1 sum axis=None inputs=[0] shape=(1, 1)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir[1].op, "sum");
    assert_eq!(ir[1].shape, (1, 1));
    assert_eq!(ir[1].inputs, vec![0]);
    assert!(ir[1].attrs.is_empty());

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![2.0]])?;

    let gradients = graph.evaluate_vjp_cpu(loss.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![2.0, 2.0], vec![2.0, 2.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_axis_mean_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;

    let reduced = a.try_mean(Some(0))?;
    let loss = reduced.try_sum(None)?;

    assert_eq!(reduced.dims(), (1, 3));
    assert_eq!(loss.dims(), (1, 1));
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(2, 3)",
            "1 mean axis=Some(0) inputs=[0] shape=(1, 3)",
            "2 sum axis=None inputs=[1] shape=(1, 1)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir[1].attrs.get("axis"), Some(&IrAttrValue::Int(0)));
    assert!(ir[2].attrs.is_empty());

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0]])?;

    let gradients = graph.evaluate_vjp_cpu(loss.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]]
    );

    Ok(())
}

#[test]
fn trace_graph_records_and_evaluates_concat_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (1, 2))?;
    let b = graph.add_input("b", (2, 2))?;

    let joined = nabla::TraceMatrix::try_concat(&[a, b], 0)?;

    assert_eq!(joined.dims(), (3, 2));
    assert_eq!(
        graph.describe_nodes()?,
        vec![
            "0 input a shape=(1, 2)",
            "1 input b shape=(2, 2)",
            "2 concat axis=0 inputs=[0, 1] shape=(3, 2)",
        ]
    );
    let ir = graph.ir_nodes()?;
    assert_eq!(ir[2].attrs.get("axis"), Some(&IrAttrValue::Int(0)));

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), PyMatrix::from_rows(vec![vec![1.0, 2.0]])?);
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![3.0, 4.0], vec![5.0, 6.0]])?,
    );
    let output_cotangent =
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])?;

    let gradients = graph.evaluate_vjp_cpu(joined.id(), inputs, &output_cotangent)?;

    assert_eq!(gradients["a"].to_rows(), vec![vec![1.0, 2.0]]);
    assert_eq!(
        gradients["b"].to_rows(),
        vec![vec![3.0, 4.0], vec![5.0, 6.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_evaluates_square_sum_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let square = a.try_mul(&a)?;
    let loss = square.try_sum(None)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0]])?;

    let gradients = graph.evaluate_vjp_cpu(loss.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![2.0, 4.0], vec![6.0, 8.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_evaluates_value_and_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let square = a.try_mul(&a)?;
    let loss = square.try_sum(None)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0]])?;

    let (value, gradients) =
        graph.evaluate_value_and_vjp_cpu(loss.id(), inputs, &output_cotangent)?;

    assert_eq!(value.to_rows(), vec![vec![30.0]]);
    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![2.0, 4.0], vec![6.0, 8.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_evaluates_squared_error_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let pred = graph.add_input("pred", (2, 2))?;
    let target = graph.add_input("target", (2, 2))?;
    let error = pred.try_sub(&target)?;
    let loss = error.try_mul(&error)?.try_sum(None)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        "pred".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 3.0], vec![2.0, 5.0]])?,
    );
    inputs.insert(
        "target".to_string(),
        PyMatrix::from_rows(vec![vec![0.5, 1.0], vec![3.0, 1.5]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0]])?;

    let gradients = graph.evaluate_vjp_cpu(loss.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["pred"].to_rows(),
        vec![vec![1.0, 4.0], vec![-2.0, 7.0]]
    );
    assert_eq!(
        gradients["target"].to_rows(),
        vec![vec![-1.0, -4.0], vec![2.0, -7.0]]
    );

    Ok(())
}

#[test]
fn trace_graph_evaluates_composed_add_matmul_vjp_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;
    let b = graph.add_input("b", (3, 2))?;
    let bias = graph.add_input("bias", (2, 2))?;
    let product = a.try_matmul(&b)?;
    let output = product.try_add(&bias)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?,
    );
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]])?,
    );
    inputs.insert(
        "bias".to_string(),
        PyMatrix::from_rows(vec![vec![0.1, 0.2], vec![0.3, 0.4]])?,
    );
    let output_cotangent = PyMatrix::from_rows(vec![vec![1.0, 0.5], vec![-1.0, 2.0]])?;

    let gradients = graph.evaluate_vjp_cpu(output.id(), inputs, &output_cotangent)?;

    assert_eq!(
        gradients["a"].to_rows(),
        vec![vec![11.0, 14.0, 17.0], vec![9.0, 11.0, 13.0]]
    );
    assert_eq!(
        gradients["b"].to_rows(),
        vec![vec![-3.0, 8.5], vec![-3.0, 11.0], vec![-3.0, 13.5]]
    );
    assert_eq!(gradients["bias"].to_rows(), output_cotangent.to_rows());

    Ok(())
}

#[test]
fn trace_graph_evaluates_composed_primal_on_cpu() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;
    let b = graph.add_input("b", (3, 2))?;
    let bias = graph.add_input("bias", (2, 2))?;
    let output = a.try_matmul(&b)?.try_add(&bias)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?,
    );
    inputs.insert(
        "b".to_string(),
        PyMatrix::from_rows(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]])?,
    );
    inputs.insert(
        "bias".to_string(),
        PyMatrix::from_rows(vec![vec![0.1, 0.2], vec![0.3, 0.4]])?,
    );

    let value = graph.evaluate_cpu(output.id(), inputs)?;

    assert_eq!(value.dims(), (2, 2));
    assert_eq!(value.to_rows(), vec![vec![58.1, 64.2], vec![139.3, 154.4]]);

    Ok(())
}

#[test]
fn trace_graph_rejects_incompatible_matmul_shapes() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 3))?;
    let b = graph.add_input("b", (4, 2))?;

    let err = a
        .try_matmul(&b)
        .err()
        .ok_or_else(|| "expected incompatible trace matmul to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn trace_graph_rejects_incompatible_add_shapes() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let b = graph.add_input("b", (2, 3))?;

    let err = a
        .try_add(&b)
        .err()
        .ok_or_else(|| "expected incompatible trace add to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn trace_graph_rejects_incompatible_sub_shapes() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let b = graph.add_input("b", (2, 3))?;

    let err = a
        .try_sub(&b)
        .err()
        .ok_or_else(|| "expected incompatible trace sub to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn trace_graph_rejects_incompatible_mul_shapes() -> Result<(), String> {
    let graph = TraceGraph::new();
    let a = graph.add_input("a", (2, 2))?;
    let b = graph.add_input("b", (2, 3))?;

    let err = a
        .try_mul(&b)
        .err()
        .ok_or_else(|| "expected incompatible trace mul to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn trace_graph_rejects_cross_graph_matmul() -> Result<(), String> {
    let left_graph = TraceGraph::new();
    let right_graph = TraceGraph::new();
    let a = left_graph.add_input("a", (2, 3))?;
    let b = right_graph.add_input("b", (3, 2))?;

    let err = a
        .try_matmul(&b)
        .err()
        .ok_or_else(|| "expected cross-graph matmul to fail".to_string())?;

    assert!(err.contains("same trace graph"));

    Ok(())
}
