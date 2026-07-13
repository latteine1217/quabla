use std::collections::BTreeMap;

use nabla_core::tensor_ir::{CpuBackend, DynamicTensor, TensorBackend, TensorIr};

macro_rules! must {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                assert!(false, "unexpected error: {error}");
                return;
            }
        }
    };
}

#[test]
fn tensor_ir_evaluates_broadcasted_expression_and_vjp() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1, 3]));
    let y = must!(graph.input("y", vec![1, 4, 1]));
    let product = must!(graph.mul(x, y));
    let output = must!(graph.add(product, x));

    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(
                vec![2, 1, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::new(
                vec![1, 4, 1],
                vec![10.0, 20.0, 30.0, 40.0],
            )),
        ),
    ]);

    let value = must!(graph.evaluate(output, &inputs));
    assert_eq!(value.shape(), &[2, 4, 3]);
    assert_eq!(
        value.data(),
        &[
            11.0, 22.0, 33.0, 21.0, 42.0, 63.0, 31.0, 62.0, 93.0, 41.0, 82.0, 123.0, 44.0, 55.0,
            66.0, 84.0, 105.0, 126.0, 124.0, 155.0, 186.0, 164.0, 205.0, 246.0
        ]
    );

    let cotangent = must!(DynamicTensor::filled(vec![2, 4, 3], 1.0));
    let gradients = must!(graph.vjp(output, &inputs, cotangent));
    assert_eq!(gradients["x"].data(), &[104.0; 6]);
    assert_eq!(gradients["y"].data(), &[21.0; 4]);
    assert!(graph.lower_text().contains("tensor<2x4x3xf64>"));

    let tangents = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::filled(vec![2, 1, 3], 1.0)),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::filled(vec![1, 4, 1], 1.0)),
        ),
    ]);
    let (jvp_value, jvp_tangent) = must!(graph.jvp(output, &inputs, &tangents));
    assert_eq!(jvp_value, value);
    assert_eq!(jvp_tangent.shape(), &[2, 4, 3]);
    assert_eq!(
        jvp_tangent.data(),
        &[
            12.0, 13.0, 14.0, 22.0, 23.0, 24.0, 32.0, 33.0, 34.0, 42.0, 43.0, 44.0, 15.0, 16.0,
            17.0, 25.0, 26.0, 27.0, 35.0, 36.0, 37.0, 45.0, 46.0, 47.0
        ]
    );
}

#[test]
fn tensor_ir_sum_builds_a_scalar_loss_with_vjp_and_jvp() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 2]));
    let squared = must!(graph.mul(x, x));
    let loss = must!(graph.sum(squared));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
    )]);

    let value = must!(graph.evaluate(loss, &inputs));
    assert_eq!(value.shape(), &[]);
    assert_eq!(value.data(), &[30.0]);

    let gradients = must!(graph.vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert_eq!(gradients["x"].data(), &[2.0, 4.0, 6.0, 8.0]);

    let tangents = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::filled(vec![2, 2], 1.0)),
    )]);
    let (_, tangent) = must!(graph.jvp(loss, &inputs, &tangents));
    assert_eq!(tangent.shape(), &[]);
    assert_eq!(tangent.data(), &[20.0]);
}

#[test]
fn tensor_ir_where_routes_gradients_without_differentiating_the_condition() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4]));
    let zero = graph.scalar_constant(0.0);
    let condition = must!(graph.greater(x, zero));
    let squared = must!(graph.mul(x, x));
    let triple = graph.scalar_constant(3.0);
    let scaled = must!(graph.mul(x, triple));
    let selected = must!(graph.where_select(condition, squared, scaled));
    let loss = must!(graph.sum(selected));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![4], vec![-2.0, -1.0, 0.0, 2.0])),
    )]);

    assert_eq!(
        must!(graph.evaluate(selected, &inputs)).data(),
        &[-6.0, -3.0, 0.0, 4.0]
    );
    let gradients = must!(graph.vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert_eq!(gradients["x"].data(), &[3.0, 3.0, 3.0, 4.0]);
    let (_, tangent) = must!(graph.jvp(
        loss,
        &inputs,
        &BTreeMap::from([("x".to_string(), must!(DynamicTensor::filled(vec![4], 1.0)))]),
    ));
    assert_eq!(tangent.data(), &[13.0]);

    let transformed = must!(graph.symbolic_jvp(loss, "x"));
    assert_eq!(
        must!(transformed.graph.evaluate(transformed.tangent, &inputs)).data(),
        &[13.0]
    );
}

#[test]
fn cpu_plan_fuses_pure_elementwise_broadcast_and_mask_chains() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let bias = must!(graph.input("bias", vec![1, 3]));
    let zero = graph.scalar_constant(0.0);
    let condition = must!(graph.greater(x, zero));
    let shifted = must!(graph.add(x, bias));
    let activated = must!(graph.tanh(shifted));
    let output = must!(graph.where_select(condition, activated, zero));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![-1.0, 2.0])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1, 3], vec![0.0, 1.0, -1.0])),
        ),
    ]);

    assert!(plan.uses_fused_elementwise_kernel());
    assert_eq!(
        must!(plan.evaluate(&inputs)),
        must!(graph.evaluate(output, &inputs))
    );
}

#[test]
fn cpu_backend_executes_a_frozen_tensor_plan() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 2]));
    let output = must!(graph.add(x, x));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
    )]);

    let backend = CpuBackend;
    let value = must!(backend.execute(&plan, &inputs));

    assert_eq!(backend.name(), "cpu");
    assert_eq!(value.data(), &[2.0, 4.0, 6.0, 8.0]);
    let (value, gradients) =
        must!(plan.value_and_vjp(&inputs, must!(DynamicTensor::filled(vec![2, 2], 1.0)),));
    assert_eq!(value.data(), &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(gradients["x"].data(), &[2.0, 2.0, 2.0, 2.0]);
    let kernel = must!(graph.compile_cpu(output)).kernel_ir();
    must!(kernel.validate());
}

#[test]
fn tensor_ir_axis_mean_and_transpose_preserve_ad_layout() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let mean = must!(graph.mean_axis(x, 0));
    let output = must!(graph.transpose(mean, None));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )),
    )]);

    let value = must!(graph.evaluate(output, &inputs));
    assert_eq!(value.shape(), &[3]);
    assert_eq!(value.data(), &[2.5, 3.5, 4.5]);

    let gradients = must!(graph.vjp(
        output,
        &inputs,
        must!(DynamicTensor::new(vec![3], vec![2.0, 3.0, 4.0])),
    ));
    assert_eq!(gradients["x"].data(), &[1.0, 1.5, 2.0, 1.0, 1.5, 2.0]);
}

#[test]
fn symbolic_jvp_preserves_parameter_vjp_for_second_coordinate_derivative() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let weight = must!(graph.input("weight", vec![1]));
    let product = must!(graph.mul(x, weight));
    let output = must!(graph.tanh(product));

    let first = must!(graph.symbolic_jvp(output, "x"));
    let mut second = must!(first.graph.symbolic_jvp(first.tangent, "x"));
    let loss = must!(second.graph.mul(second.tangent, second.tangent));
    let loss = must!(second.graph.sum(loss));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.3])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![1.2])),
        ),
    ]);

    let gradients =
        must!(second
            .graph
            .vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert!(gradients["weight"].data()[0].abs() > 1e-8);
}

#[test]
fn symbolic_jvp_supports_batched_matmul_mlp_nodes() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let weight = must!(graph.input("weight", vec![1, 1]));
    let product = must!(graph.matmul(x, weight));
    let output = must!(graph.tanh(product));
    let mut transformed = must!(graph.symbolic_jvp(output, "x"));
    let loss = must!(transformed.graph.sum(transformed.tangent));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![0.1, 0.2])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1, 1], vec![1.5])),
        ),
    ]);

    let gradients =
        must!(transformed
            .graph
            .vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert!(gradients["weight"].data()[0].abs() > 1e-8);
}

#[test]
fn symbolic_jvp_supports_sine_second_derivative_and_parameter_vjp() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let weight = must!(graph.input("weight", vec![1]));
    let product = must!(graph.mul(x, weight));
    let output = must!(graph.sin(product));
    let first = must!(graph.symbolic_jvp(output, "x"));
    let mut second = must!(first.graph.symbolic_jvp(first.tangent, "x"));
    let loss = must!(second.graph.mul(second.tangent, second.tangent));
    let loss = must!(second.graph.sum(loss));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.4])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![1.3])),
        ),
    ]);
    let gradients =
        must!(second
            .graph
            .vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert!(gradients["weight"].data()[0].abs() > 1e-8);
}

#[test]
fn symbolic_jvp_preserves_transpose_reshape_and_axis_reduction() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 2]));
    let transposed = must!(graph.transpose(x, Some(vec![1, 0])));
    let reshaped = must!(graph.reshape(transposed, vec![4]));
    let output = must!(graph.mean_axis(reshaped, 0));
    let transformed = must!(graph.symbolic_jvp(output, "x"));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
    )]);

    let tangent = must!(transformed.graph.evaluate(transformed.tangent, &inputs));
    assert_eq!(tangent.shape(), &[]);
    assert_eq!(tangent.data(), &[1.0]);
}

#[test]
fn symbolic_jvp_supports_division_and_parameter_vjp() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let weight = must!(graph.input("weight", vec![1]));
    let output = must!(graph.div(x, weight));
    let mut transformed = must!(graph.symbolic_jvp(output, "x"));
    let loss = must!(transformed
        .graph
        .mul(transformed.tangent, transformed.tangent));
    let loss = must!(transformed.graph.sum(loss));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.4])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![2.0])),
        ),
    ]);

    let tangent = must!(transformed.graph.evaluate(transformed.tangent, &inputs));
    assert_eq!(tangent.data(), &[0.5]);
    let gradients =
        must!(transformed
            .graph
            .vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert!((gradients["weight"].data()[0] + 0.25).abs() < 1e-12);
}

#[test]
fn symbolic_jvp_supports_log_and_parameter_vjp() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let weight = must!(graph.input("weight", vec![1]));
    let sum = must!(graph.add(x, weight));
    let output = must!(graph.log(sum));
    let mut transformed = must!(graph.symbolic_jvp(output, "x"));
    let loss = must!(transformed
        .graph
        .mul(transformed.tangent, transformed.tangent));
    let loss = must!(transformed.graph.sum(loss));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1], vec![2.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![3.0])),
        ),
    ]);

    let tangent = must!(transformed.graph.evaluate(transformed.tangent, &inputs));
    assert_eq!(tangent.data(), &[0.2]);
    let gradients =
        must!(transformed
            .graph
            .vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert!((gradients["weight"].data()[0] + 2.0 / 125.0).abs() < 1e-12);
}
