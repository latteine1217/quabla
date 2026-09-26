use std::collections::{BTreeMap, BTreeSet};

use nabla_core::tensor_ir::{
    CpuBackend, DynamicTensor, TensorBackend, TensorBufferSlot, TensorCondExecutionPlan,
    TensorDType, TensorDeviceBackend, TensorDeviceId, TensorDeviceMesh, TensorForiExecutionPlan,
    TensorForiMultiExecutionPlan, TensorForiVjpJvpExecutionPlan, TensorFusionRegion, TensorIr,
    TensorPartitionSpec, TensorPlacement, TensorReplicaReduction, TensorScanExecutionPlan,
};
use nabla_core::{NablaCompiler, NablaTarget};

#[cfg(all(feature = "cuda", target_os = "linux"))]
use nabla_core::tensor_ir::CudaBackend;

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
use nabla_core::tensor_ir::CudaDataParallelExecutionPlan;

#[cfg(all(feature = "mlx", target_os = "macos"))]
use nabla_core::tensor_ir::MlxBackend;

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
fn backend_precision_contract_keeps_cpu_reference_and_gpu_execution_explicit() {
    assert_eq!(
        TensorDeviceBackend::Cpu.precision(),
        nabla_core::tensor_ir::TensorBackendPrecision {
            logical: TensorDType::F64,
            execution: TensorDType::F64,
        }
    );
    for backend in [TensorDeviceBackend::Cuda, TensorDeviceBackend::Mlx] {
        assert_eq!(backend.precision().logical, TensorDType::F64);
        assert_eq!(backend.precision().execution, TensorDType::F32);
    }
}

#[test]
fn compiler_facade_owns_program_transform_and_cpu_execution_lifecycle() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let squared = must!(graph.mul(x, x));
    let compiler = NablaCompiler;
    let program = must!(compiler.program(graph, squared));

    assert_eq!(program.output_shape(), Ok(vec![2]));
    assert!(program.lower_text().contains("mul"));
    assert!(compiler.capability(NablaTarget::Cpu).built);

    let executable = must!(compiler.compile(&program, NablaTarget::Cpu));
    assert_eq!(executable.target(), NablaTarget::Cpu);
    assert_eq!(
        must!(executable.execute(&BTreeMap::from([(
            "x".to_string(),
            must!(DynamicTensor::new(vec![2], vec![2.0, 3.0])),
        )]))),
        must!(DynamicTensor::new(vec![2], vec![4.0, 9.0]))
    );

    let jvp = must!(program.jvp(&BTreeMap::from([(
        "x".to_string(),
        "x_tangent".to_string(),
    )])));
    let (jvp_plan, tangent) = must!(jvp.freeze());
    assert_eq!(jvp_plan.output_node_ids().len(), 2);
    assert_ne!(jvp_plan.output_node_id(), tangent);

    let vjp = must!(program.vjp("cotangent"));
    assert_eq!(
        vjp.cotangent_node_id(),
        vjp.program().ir().input_node_id("cotangent").unwrap()
    );
    assert!(vjp.gradient_node_ids().contains_key("x"));
}

#[test]
fn stablehlo_export_preserves_a_pure_elementwise_static_graph() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let y = must!(graph.input("y", vec![2]));
    let sum = must!(graph.add(x, y));
    let output = must!(graph.tanh(sum));
    let text = must!(graph.stablehlo_text(output));
    assert!(text.contains("func.func @main(%arg0: tensor<2xf64> // x, %arg1: tensor<2xf64> // y)"));
    assert!(text.contains("stablehlo.add %arg0, %arg1 : tensor<2xf64>"));
    assert!(text.contains("stablehlo.tanh %v2 : tensor<2xf64>"));
}

#[test]
fn stablehlo_export_rejects_operations_without_a_verified_lowering() {
    let mut graph = TensorIr::new();
    let input = must!(graph.input("x", vec![2]));
    let output = must!(graph.exp(input));

    let error = graph
        .stablehlo_text(output)
        .expect_err("unimplemented StableHLO operations must fail explicitly");
    assert!(error.contains("does not yet support exp"));
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
    assert_eq!(value.shape(), &[] as &[usize]);
    assert_eq!(value.data(), &[30.0]);

    let gradients = must!(graph.vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert_eq!(gradients["x"].data(), &[2.0, 4.0, 6.0, 8.0]);

    let tangents = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::filled(vec![2, 2], 1.0)),
    )]);
    let (_, tangent) = must!(graph.jvp(loss, &inputs, &tangents));
    assert_eq!(tangent.shape(), &[] as &[usize]);
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
fn tensor_ir_cond_executes_only_the_selected_cpu_region_and_routes_ad() {
    let mut on_true = TensorIr::new();
    let true_x = must!(on_true.input("captured", vec![]));
    let true_output = must!(on_true.mul(true_x, true_x));

    let mut on_false = TensorIr::new();
    let false_x = must!(on_false.input("captured", vec![]));
    let three = on_false.scalar_constant(3.0);
    let false_output = must!(on_false.mul(false_x, three));

    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let x = must!(graph.input("x", vec![]));
    let captured = must!(graph.add(x, x));
    let conditional = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("captured".to_string(), captured)],
    ));
    let plan = must!(graph.compile_cpu(conditional));

    let true_inputs = BTreeMap::from([
        (
            "predicate".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    assert_eq!(must!(plan.evaluate(&true_inputs)).data(), &[16.0]);
    let (_, tangent) = must!(plan.jvp(
        &true_inputs,
        &BTreeMap::from([
            (
                "predicate".to_string(),
                must!(DynamicTensor::new(vec![], vec![7.0])),
            ),
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0]))
            ),
        ]),
    ));
    assert_eq!(tangent.data(), &[16.0]);
    let gradients = must!(plan.vjp(&true_inputs, must!(DynamicTensor::new(vec![], vec![1.0]))));
    assert_eq!(gradients["x"].data(), &[16.0]);
    assert_eq!(gradients["predicate"].data(), &[0.0]);

    let false_inputs = BTreeMap::from([
        (
            "predicate".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.0])),
        ),
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    assert_eq!(must!(plan.evaluate(&false_inputs)).data(), &[12.0]);
}

#[test]
fn tensor_ir_cond_does_not_evaluate_an_inactive_failing_branch() {
    let mut on_true = TensorIr::new();
    let true_x = must!(on_true.input("x", vec![]));

    let mut on_false = TensorIr::new();
    let false_x = must!(on_false.input("x", vec![]));
    let zero = on_false.scalar_constant(0.0);
    let false_output = must!(on_false.div(false_x, zero));

    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_x)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    must!(graph.input("x", vec![]));
    let output = must!(graph.cond(predicate, branches));
    let inputs = BTreeMap::from([
        (
            "predicate".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    assert_eq!(must!(graph.evaluate(output, &inputs)).data(), &[2.0]);
}

#[test]
fn symbolic_jvp_transforms_cond_regions_with_explicit_captures() {
    let mut on_true = TensorIr::new();
    let true_capture = must!(on_true.input("captured", vec![]));
    let true_output = must!(on_true.mul(true_capture, true_capture));

    let mut on_false = TensorIr::new();
    let false_capture = must!(on_false.input("captured", vec![]));
    let three = on_false.scalar_constant(3.0);
    let false_output = must!(on_false.mul(false_capture, three));

    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let x = must!(graph.input("x", vec![]));
    let capture = must!(graph.add(x, x));
    let output = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("captured".to_string(), capture)],
    ));
    let transformed = must!(graph.symbolic_jvp(output, "x"));

    for (predicate, expected_value, expected_tangent) in [(1.0, 16.0, 16.0), (0.0, 12.0, 6.0)] {
        let inputs = BTreeMap::from([
            (
                "predicate".to_string(),
                must!(DynamicTensor::new(vec![], vec![predicate])),
            ),
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![], vec![2.0])),
            ),
        ]);
        assert_eq!(
            must!(transformed.graph.evaluate(transformed.value, &inputs)).data(),
            &[expected_value]
        );
        assert_eq!(
            must!(transformed.graph.evaluate(transformed.tangent, &inputs)).data(),
            &[expected_tangent]
        );
    }
}

#[test]
fn symbolic_vjp_transforms_cond_regions_with_explicit_captures() {
    let mut on_true = TensorIr::new();
    let true_capture = must!(on_true.input("captured", vec![]));
    let true_output = must!(on_true.mul(true_capture, true_capture));

    let mut on_false = TensorIr::new();
    let false_capture = must!(on_false.input("captured", vec![]));
    let three = on_false.scalar_constant(3.0);
    let false_output = must!(on_false.mul(false_capture, three));

    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let x = must!(graph.input("x", vec![]));
    let capture = must!(graph.add(x, x));
    let output = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("captured".to_string(), capture)],
    ));
    let transformed = must!(graph.symbolic_vjp(output, "seed"));

    for (predicate, expected_value, expected_gradient) in [(1.0, 16.0, 16.0), (0.0, 12.0, 6.0)] {
        let inputs = BTreeMap::from([
            (
                "predicate".to_string(),
                must!(DynamicTensor::new(vec![], vec![predicate])),
            ),
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![], vec![2.0])),
            ),
            (
                "seed".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ),
        ]);
        assert_eq!(
            must!(transformed.graph.evaluate(transformed.value, &inputs)).data(),
            &[expected_value]
        );
        assert_eq!(
            must!(transformed
                .graph
                .evaluate(transformed.gradients["x"], &inputs))
            .data(),
            &[expected_gradient]
        );
        assert_eq!(
            must!(transformed
                .graph
                .evaluate(transformed.gradients["predicate"], &inputs))
            .data(),
            &[0.0]
        );
    }
}

#[test]
fn hessian_and_hvp_support_cond_regions_through_symbolic_ad() {
    let mut on_true = TensorIr::new();
    let true_capture = must!(on_true.input("captured", vec![]));
    let true_output = must!(on_true.mul(true_capture, true_capture));

    let mut on_false = TensorIr::new();
    let false_capture = must!(on_false.input("captured", vec![]));
    let three = on_false.scalar_constant(3.0);
    let false_output = must!(on_false.mul(false_capture, three));

    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let x = must!(graph.input("x", vec![]));
    let capture = must!(graph.add(x, x));
    let output = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("captured".to_string(), capture)],
    ));

    for (predicate, expected) in [(1.0, 8.0), (0.0, 0.0)] {
        let inputs = BTreeMap::from([
            (
                "predicate".to_string(),
                must!(DynamicTensor::new(vec![], vec![predicate])),
            ),
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![], vec![2.0])),
            ),
        ]);
        assert_eq!(
            must!(graph.hessian_scalar(output, "x", &inputs)),
            vec![vec![expected]]
        );
        assert_eq!(
            must!(graph.hvp_scalar(
                output,
                "x",
                &inputs,
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ))
            .data(),
            &[expected]
        );
    }
}

#[test]
fn fori_region_executes_and_differentiates_without_static_unrolling() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let external = BTreeMap::from([(
        "scale".to_string(),
        must!(DynamicTensor::new(vec![], vec![2.0])),
    )]);
    assert_eq!(
        must!(loop_plan.evaluate(must!(DynamicTensor::new(vec![], vec![1.0])), &external)).data(),
        &[12.0]
    );
    let (_, tangent) = must!(loop_plan.jvp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &external,
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.0])),
        )]),
    ));
    assert_eq!(tangent.data(), &[8.0]);
    let (value, initial_gradient, external_gradients) = must!(loop_plan.value_and_vjp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &external,
        must!(DynamicTensor::new(vec![], vec![1.0])),
    ));
    assert_eq!(value.data(), &[12.0]);
    assert_eq!(initial_gradient.data(), &[8.0]);
    assert_eq!(external_gradients["scale"].data(), &[13.0]);

    let (_, tape) = must!(
        loop_plan.evaluate_with_tape(must!(DynamicTensor::new(vec![], vec![1.0])), &external,)
    );
    assert_eq!(tape.len(), 4);
    assert_eq!(must!(tape.carry_at(0)).data(), &[1.0]);
    assert_eq!(must!(tape.carry_at(1)).data(), &[2.0]);
    assert_eq!(must!(tape.carry_at(2)).data(), &[5.0]);
    assert_eq!(must!(tape.carry_at(3)).data(), &[12.0]);
}

#[test]
fn fori_region_allows_an_unused_index_input() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let output = must!(body.add(carry, scale));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));

    assert_eq!(
        must!(loop_plan.evaluate(
            must!(DynamicTensor::new(vec![], vec![1.0])),
            &BTreeMap::from([(
                "scale".to_string(),
                must!(DynamicTensor::new(vec![], vec![2.0])),
            )]),
        ))
        .data(),
        &[7.0]
    );
}

#[test]
fn multi_carry_fori_region_preserves_ordered_outputs_and_external_captures() {
    let mut body = TensorIr::new();
    let position = must!(body.input("position", vec![]));
    let energy = must!(body.input("energy", vec![1]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![1]));
    let next_position = must!(body.add(position, index));
    let next_energy = must!(body.mul(energy, scale));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next_position, next_energy]));
    let loop_plan = must!(TensorForiMultiExecutionPlan::new(
        0,
        2,
        body_plan,
        vec!["position".to_string(), "energy".to_string()],
        "index",
    ));
    assert_eq!(loop_plan.carry_names(), &["position", "energy"]);
    let carries = must!(loop_plan.evaluate(
        vec![
            must!(DynamicTensor::new(vec![], vec![1.0])),
            must!(DynamicTensor::new(vec![1], vec![2.0])),
        ],
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![1], vec![3.0])),
        )]),
    ));
    assert_eq!(carries[0].data(), &[2.0]);
    assert_eq!(carries[1].data(), &[18.0]);

    let (_, initial_gradients, external_gradients) = must!(loop_plan.value_and_vjp(
        vec![
            must!(DynamicTensor::new(vec![], vec![1.0])),
            must!(DynamicTensor::new(vec![1], vec![2.0])),
        ],
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![1], vec![3.0])),
        )]),
        vec![
            must!(DynamicTensor::new(vec![], vec![1.0])),
            must!(DynamicTensor::new(vec![1], vec![1.0])),
        ],
    ));
    assert_eq!(initial_gradients[0].data(), &[1.0]);
    assert_eq!(initial_gradients[1].data(), &[9.0]);
    assert_eq!(external_gradients["scale"].data(), &[12.0]);
}

#[test]
fn scan_region_stacks_fixed_shape_outputs_without_unrolling_the_body_plan() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let (plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan = must!(TensorScanExecutionPlan::new(0, 3, plan, "carry", "index"));
    let (carry, outputs) = must!(scan.evaluate(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
    ));
    assert_eq!(carry.data(), &[12.0]);
    assert_eq!(outputs.shape(), &[3]);
    assert_eq!(outputs.data(), &[2.0, 5.0, 12.0]);
    let (_, _, tape) = must!(scan.evaluate_with_tape(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
    ));
    assert_eq!(tape.len(), 4);
    assert_eq!(must!(tape.carry_at(2)).data(), &[5.0]);
    let ((carry, outputs), (carry_tangent, output_tangents)) = must!(scan.jvp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        must!(DynamicTensor::new(vec![], vec![0.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        )]),
    ));
    assert_eq!(carry.data(), &[12.0]);
    assert_eq!(outputs.data(), &[2.0, 5.0, 12.0]);
    assert_eq!(carry_tangent.data(), &[13.0]);
    assert_eq!(output_tangents.data(), &[1.0, 4.0, 13.0]);
    let (_, _, initial_gradient, gradients) = must!(scan.value_and_vjp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
        must!(DynamicTensor::new(vec![], vec![0.0])),
        must!(DynamicTensor::new(vec![3], vec![1.0, 1.0, 1.0])),
    ));
    assert_eq!(initial_gradient.data(), &[14.0]);
    assert_eq!(gradients["scale"].data(), &[18.0]);
}

#[test]
fn scan_region_integrates_with_parent_ir_execution_jvp_and_vjp() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[next, next])).0,
        "carry",
        "index",
    ));

    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let output_sum = must!(graph.sum(outputs));
    let total = must!(graph.add(final_carry, output_sum));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);

    assert_eq!(must!(graph.evaluate(final_carry, &inputs)).data(), &[12.0]);
    assert_eq!(
        must!(graph.evaluate(outputs, &inputs)).data(),
        &[2.0, 5.0, 12.0]
    );
    assert_eq!(must!(graph.evaluate(total, &inputs)).data(), &[31.0]);
    assert!(graph.lower_text().contains("scan(group="));
    let (_, tangent) = must!(graph.jvp(
        total,
        &inputs,
        &BTreeMap::from([
            (
                "initial".to_string(),
                must!(DynamicTensor::new(vec![], vec![0.0])),
            ),
            (
                "scale".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ),
        ]),
    ));
    assert_eq!(tangent.data(), &[31.0]);
    let (_, gradients) =
        must!(graph.value_and_vjp(total, &inputs, must!(DynamicTensor::new(vec![], vec![1.0])),));
    assert_eq!(gradients["initial"].data(), &[22.0]);
    assert_eq!(gradients["scale"].data(), &[31.0]);

    let (_, final_carry_gradients) = must!(must!(graph.compile_cpu(final_carry))
        .value_and_vjp(&inputs, must!(DynamicTensor::new(vec![], vec![1.0])),));
    assert_eq!(final_carry_gradients["initial"].data(), &[8.0]);
    assert_eq!(final_carry_gradients["scale"].data(), &[13.0]);
    let (_, output_gradients) = must!(must!(graph.compile_cpu(outputs)).value_and_vjp(
        &inputs,
        must!(DynamicTensor::new(vec![3], vec![1.0, 1.0, 1.0])),
    ));
    assert_eq!(output_gradients["initial"].data(), &[14.0]);
    assert_eq!(output_gradients["scale"].data(), &[18.0]);
}

#[test]
fn symbolic_jvp_transforms_scan_region_without_unrolling_parent_loop() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[next, next])).0,
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let output_sum = must!(graph.sum(outputs));
    let total = must!(graph.add(final_carry, output_sum));
    let base_inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);

    let initial_jvp = must!(graph.symbolic_jvp(total, "initial"));
    assert!(initial_jvp.graph.lower_text().contains("scan(group="));
    assert_eq!(
        must!(initial_jvp
            .graph
            .compile_cpu(initial_jvp.tangent)
            .and_then(|plan| plan.evaluate(&base_inputs)))
        .data(),
        &[22.0]
    );
    let scale_jvp = must!(graph.symbolic_jvp_with_tangent_inputs(
        total,
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
    ));
    let mut inputs = base_inputs;
    inputs.insert(
        "scale_tangent".to_string(),
        must!(DynamicTensor::new(vec![], vec![1.0])),
    );
    assert_eq!(
        must!(scale_jvp
            .graph
            .compile_cpu(scale_jvp.tangent)
            .and_then(|plan| plan.evaluate(&inputs)))
        .data(),
        &[31.0]
    );
}

#[test]
fn symbolic_vjp_transforms_scan_region_with_joint_carry_and_output_cotangents() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[next, next])).0,
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let output_sum = must!(graph.sum(outputs));
    let total = must!(graph.add(final_carry, output_sum));
    let symbolic = must!(graph.symbolic_vjp(total, "seed"));
    assert!(symbolic.graph.lower_text().contains("scan_vjp(group="));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    assert_eq!(
        must!(symbolic
            .graph
            .compile_cpu(symbolic.gradients["initial"])
            .and_then(|plan| plan.evaluate(&inputs)))
        .data(),
        &[22.0]
    );
    assert_eq!(
        must!(symbolic
            .graph
            .compile_cpu(symbolic.gradients["scale"])
            .and_then(|plan| plan.evaluate(&inputs)))
        .data(),
        &[31.0]
    );
}

#[test]
fn symbolic_jvp_many_retains_scan_vjp_jvp_sibling_targets() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[next, next])).0,
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)]));
    let output_sum = must!(graph.sum(outputs));
    let total = must!(graph.add(final_carry, output_sum));
    let vjp = must!(graph.symbolic_vjp(total, "seed"));
    let directional = must!(vjp.graph.symbolic_jvp_many_with_tangent_inputs(
        &[vjp.gradients["initial"], vjp.gradients["scale"]],
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
    ));
    assert_eq!(
        directional
            .graph
            .lower_text()
            .matches("scan_vjp_jvp(group=")
            .count(),
        2
    );
    let (plan, output_ids) = must!(directional.graph.compile_cpu_many(&directional.tangents));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let outputs = must!(plan.evaluate_many(&inputs));
    assert_eq!(output_ids.len(), 2);
    assert_eq!(outputs[0].data(), &[29.0]);
    assert_eq!(outputs[1].data(), &[26.0]);

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if std::env::var_os("NABLA_CUDA_TEST").is_some() {
        let cuda = must!(CudaBackend::default().execute_many(&plan, &output_ids, &inputs));
        for (actual, expected) in cuda.iter().zip(&outputs) {
            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.data().iter().zip(expected.data()) {
                assert!((actual - expected).abs() < 3e-5);
            }
        }
    }
}

#[test]
fn hessian_scalar_propagates_through_nonlinear_scan_vjp_jvp() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let squared = must!(body.mul(carry, carry));
    let scaled = must!(body.mul(squared, scale));
    let next = must!(body.add(scaled, index));
    let output = must!(body.mul(next, scale));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[next, output])).0,
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let output_sum = must!(graph.sum(outputs));
    let loss = must!(graph.add(final_carry, output_sum));
    let evaluate_loss = |scale: f64| {
        graph
            .evaluate(
                loss,
                &BTreeMap::from([
                    (
                        "initial".to_string(),
                        DynamicTensor::new(vec![], vec![0.4])
                            .expect("scalar initial tensor is valid"),
                    ),
                    (
                        "scale".to_string(),
                        DynamicTensor::new(vec![], vec![scale])
                            .expect("scalar scale tensor is valid"),
                    ),
                ]),
            )
            .expect("nonlinear Scan loss evaluates")
            .data()[0]
    };
    let scale = 0.8;
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.4])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![scale])),
        ),
    ]);
    let hvp = must!(graph.hvp_scalar(
        loss,
        "scale",
        &inputs,
        must!(DynamicTensor::new(vec![], vec![1.0])),
    ));
    let step = 1e-4;
    let finite_difference = (evaluate_loss(scale + step) - 2.0 * evaluate_loss(scale)
        + evaluate_loss(scale - step))
        / (step * step);
    assert!((hvp.data()[0] - finite_difference).abs() < 2e-5);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_nonlinear_scan_forward_over_reverse_hvp() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let squared = must!(body.mul(carry, carry));
    let scaled = must!(body.mul(squared, scale));
    let next = must!(body.add(scaled, index));
    let output = must!(body.mul(next, scale));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[next, output])).0,
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let output_sum = must!(graph.sum(outputs));
    let loss = must!(graph.add(final_carry, output_sum));
    let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
    let directional = must!(vjp.graph.symbolic_jvp_with_tangent_inputs(
        vjp.gradients["scale"],
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
    ));
    assert!(directional
        .graph
        .lower_text()
        .contains("scan_vjp_jvp(group="));
    let plan = must!(directional.graph.compile_cpu(directional.tangent));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.4])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.8])),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate(&inputs));
    let mlx = must!(MlxBackend.execute(&plan, &inputs));
    assert_eq!(mlx.shape(), cpu.shape());
    assert!((mlx.data()[0] - cpu.data()[0]).abs() < 2e-5);
}

#[test]
fn symbolic_jvp_of_scalar_vjp_gradient_propagates_cotangent_direction() {
    let mut graph = TensorIr::new();
    let carry = must!(graph.input("carry", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.mul(carry, scale));
    let vjp = must!(graph.symbolic_vjp(output, "cotangent"));
    let jvp = must!(vjp.graph.symbolic_jvp_with_tangent_inputs(
        vjp.gradients["carry"],
        &BTreeMap::from([
            ("scale".to_string(), "scale_tangent".to_string()),
            ("cotangent".to_string(), "cotangent_tangent".to_string()),
        ]),
    ));
    let inputs = BTreeMap::from([
        (
            "carry".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "cotangent_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let value = must!(must!(jvp.graph.compile_cpu(jvp.tangent)).evaluate(&inputs));
    assert_eq!(value.data(), &[4.0]);
}

#[test]
fn scan_region_allows_an_unused_index_input() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let next = must!(body.add(carry, scale));
    let (plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan = must!(TensorScanExecutionPlan::new(0, 3, plan, "carry", "index"));

    let (carry, outputs) = must!(scan.evaluate(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
    ));
    assert_eq!(carry.data(), &[7.0]);
    assert_eq!(outputs.data(), &[3.0, 5.0, 7.0]);
}

#[test]
fn fori_vjp_jvp_plan_matches_exact_loop_gradient_direction() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index"
    ));
    let plan = must!(TensorForiVjpJvpExecutionPlan::new(plan, "__debug"));
    let tangents = must!(plan.jvp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        must!(DynamicTensor::new(vec![], vec![0.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        )]),
        must!(DynamicTensor::new(vec![], vec![1.0])),
        must!(DynamicTensor::new(vec![], vec![0.0])),
    ));
    assert_eq!(tangents["carry"].data(), &[12.0]);
    assert_eq!(tangents["scale"].data(), &[12.0]);
}

#[test]
fn fori_region_integrates_with_parent_ir_execution_and_ad() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));

    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)],));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);

    assert_eq!(must!(graph.evaluate(output, &inputs)).data(), &[12.0]);
    assert!(graph.lower_text().contains("fori(carry="));
    assert_eq!(
        must!(must!(graph.compile_cpu(output)).evaluate(&inputs)).data(),
        &[12.0]
    );
    let (_, tangent) = must!(graph.jvp(
        output,
        &inputs,
        &BTreeMap::from([
            (
                "initial".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ),
            (
                "scale".to_string(),
                must!(DynamicTensor::new(vec![], vec![0.0])),
            ),
        ]),
    ));
    assert_eq!(tangent.data(), &[8.0]);
    let (_, gradients) = must!(graph.value_and_vjp(
        output,
        &inputs,
        must!(DynamicTensor::new(vec![], vec![1.0])),
    ));
    assert_eq!(gradients["initial"].data(), &[8.0]);
    assert_eq!(gradients["scale"].data(), &[13.0]);
}

#[test]
fn symbolic_jvp_transforms_fori_region_without_unrolling_parent_loop() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    let symbolic = must!(graph.symbolic_jvp(output, "initial"));
    assert!(symbolic.graph.lower_text().contains("fori(carry="));
    assert!(symbolic.graph.lower_text().contains("fori_jvp(carry="));
    assert!(!symbolic.graph.lower_text().contains("slice("));
    let tangent = must!(symbolic
        .graph
        .compile_cpu(symbolic.tangent)
        .and_then(|plan| plan.evaluate(&inputs)));
    assert_eq!(tangent.data(), &[8.0]);

    let symbolic = must!(graph.symbolic_jvp_with_tangent_inputs(
        output,
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string(),)]),
    ));
    let tangent_inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let tangent = must!(symbolic
        .graph
        .compile_cpu(symbolic.tangent)
        .and_then(|plan| plan.evaluate(&tangent_inputs)));
    assert_eq!(tangent.data(), &[13.0]);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_structural_fori_jvp_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let symbolic = must!(graph.symbolic_jvp_with_tangent_inputs(
        output,
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
    ));
    let inputs = BTreeMap::from([
        ("initial".to_string(), must!(DynamicTensor::new(vec![], vec![1.0]))),
        ("scale".to_string(), must!(DynamicTensor::new(vec![], vec![2.0]))),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let plan = must!(symbolic.graph.compile_cpu(symbolic.tangent));
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    assert!((cuda.data()[0] - cpu.data()[0]).abs() < 3e-5);
    assert!((cuda.data()[0] - 13.0).abs() < 3e-5);
}

#[test]
fn symbolic_vjp_transforms_fori_region_with_shared_reverse_loop_results() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let transformed = must!(graph.symbolic_vjp(output, "seed"));
    assert!(transformed.graph.lower_text().contains("fori_vjp(group="));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let initial_gradient = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["initial"])
        .and_then(|plan| plan.evaluate(&inputs)));
    let scale_gradient = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["scale"])
        .and_then(|plan| plan.evaluate(&inputs)));
    assert_eq!(initial_gradient.data(), &[8.0]);
    assert_eq!(scale_gradient.data(), &[13.0]);
    let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
        transformed.gradients["initial"],
        transformed.gradients["scale"],
    ]));
    assert_eq!(
        must!(plan.evaluate_many(&inputs))
            .iter()
            .map(|value| value.data().to_vec())
            .collect::<Vec<_>>(),
        vec![vec![8.0], vec![13.0]]
    );
    assert_eq!(output_ids.len(), 2);
}

#[test]
fn hessian_and_hvp_propagate_exactly_through_fori_regions() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);

    assert_eq!(
        must!(graph.hessian_scalar(output, "scale", &inputs)),
        vec![vec![12.0]]
    );
    assert_eq!(
        must!(graph.hvp_scalar(
            output,
            "scale",
            &inputs,
            must!(DynamicTensor::new(vec![], vec![3.0])),
        ))
        .data(),
        &[36.0]
    );
}

#[test]
fn symbolic_hvp_through_fori_lowers_to_structural_fori_vjp_jvp() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let vjp = must!(graph.symbolic_vjp(output, "cotangent"));
    let directional = must!(vjp.graph.symbolic_jvp_with_tangent_inputs(
        vjp.gradients["scale"],
        &BTreeMap::from([(String::from("scale"), String::from("scale_tangent"))]),
    ));
    assert!(directional.graph.lower_text().contains("fori_vjp_jvp"));
    let value = must!(directional.graph.evaluate(
        directional.tangent,
        &BTreeMap::from([
            (
                "initial".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ),
            (
                "scale".to_string(),
                must!(DynamicTensor::new(vec![], vec![2.0])),
            ),
            (
                "cotangent".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ),
            (
                "scale_tangent".to_string(),
                must!(DynamicTensor::new(vec![], vec![1.0])),
            ),
        ]),
    ));
    assert_eq!(value.data(), &[12.0]);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_structural_fori_hvp_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let vjp = must!(graph.symbolic_vjp(output, "cotangent"));
    let directional = must!(vjp.graph.symbolic_jvp_with_tangent_inputs(
        vjp.gradients["scale"],
        &BTreeMap::from([(String::from("scale"), String::from("scale_tangent"))]),
    ));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let plan = must!(directional.graph.compile_cpu(directional.tangent));
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    assert!(
        (cuda.data()[0] - cpu.data()[0]).abs() < 3e-5,
        "CUDA Fori HVP mismatch: {} vs {}",
        cuda.data()[0],
        cpu.data()[0]
    );
    assert!((cuda.data()[0] - 12.0).abs() < 3e-5);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_rejects_cond_regions_before_device_lowering() {
    let mut on_true = TensorIr::new();
    let true_output = on_true.scalar_constant(1.0);
    let mut on_false = TensorIr::new();
    let false_output = on_false.scalar_constant(2.0);
    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let output = must!(graph.cond(predicate, branches));
    let error = CudaBackend::new(0)
        .compile(must!(graph.compile_cpu(output)))
        .expect_err("CUDA must reject Cond until device-predicate lowering exists");
    assert!(error.contains("does not yet support Cond regions"));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_elementwise_fixed_fori_regions_on_device() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let increment = must!(body.mul(index, scale));
    let output = must!(body.add(carry, increment));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    assert!((cuda.data()[0] - cpu.data()[0]).abs() < 1e-5);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_fixed_fori_vjp_with_device_resident_carry_tape_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![3]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let output = must!(body.tanh(shifted));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![3]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let transformed = must!(graph.symbolic_vjp(output, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.8, 1.1, 0.6])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![3], vec![1.0, -0.5, 0.25])),
        ),
    ]);
    for name in ["initial", "scale"] {
        let plan = must!(transformed.graph.compile_cpu(transformed.gradients[name]));
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
            assert!(
                (actual - expected).abs() < 2e-5,
                "CUDA Fori VJP mismatch for {name}: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_fuses_grouped_fori_vjp_targets_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![3]));
    let bias = must!(body.input("bias", vec![3]));
    let scaled = must!(body.mul(carry, scale));
    let biased = must!(body.add(scaled, bias));
    let shifted = must!(body.add(biased, index));
    let output = must!(body.tanh(shifted));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![3]));
    let bias = must!(graph.input("bias", vec![3]));
    let output = must!(graph.fori(
        initial,
        loop_plan,
        vec![("scale".to_string(), scale), ("bias".to_string(), bias)],
    ));
    let transformed = must!(graph.symbolic_vjp(output, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.8, 1.1, 0.6])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.1, -0.2, 0.05])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![3], vec![1.0, -0.5, 0.25])),
        ),
    ]);
    let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
        transformed.gradients["initial"],
        transformed.gradients["scale"],
        transformed.gradients["bias"],
    ]));
    let cpu = must!(plan.evaluate_many(&inputs));
    let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
    for (actual, expected) in cuda.iter().zip(cpu.iter()) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 3e-5,
                "grouped CUDA Fori VJP mismatch: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_reduces_scalar_fori_capture_vjp_on_device_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let output = must!(body.tanh(shifted));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let loss = must!(graph.sum(output));
    let transformed = must!(graph.symbolic_vjp(loss, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.8])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["scale"]));
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    assert!((cuda.data()[0] - cpu.data()[0]).abs() < 3e-5);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_reduces_broadcast_fori_capture_vjp_on_device_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![2, 3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![1, 3]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let output = must!(body.tanh(shifted));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![2, 3]));
    let scale = must!(graph.input("scale", vec![1, 3]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let loss = must!(graph.sum(output));
    let transformed = must!(graph.symbolic_vjp(loss, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3],
                vec![0.2, -0.1, 0.3, -0.4, 0.5, 0.1],
            )),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![1, 3], vec![0.8, 1.1, 0.6])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["scale"]));
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 5e-5);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_fixed_scan_regions_with_shared_device_results_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![3]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let next = must!(body.tanh(shifted));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![3]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let (plan, output_ids) = must!(graph.compile_cpu_many(&[final_carry, outputs]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.8, 1.1, 0.6])),
        ),
    ]);
    let cpu = must!(plan.evaluate_many(&inputs));
    let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
    for (actual, expected) in cuda.iter().zip(cpu) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 2e-5);
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_fixed_scan_vjp_with_device_resident_carry_tape_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![3]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let next = must!(body.tanh(shifted));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![3]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let final_sum = must!(graph.sum(final_carry));
    let output_sum = must!(graph.sum(outputs));
    let loss = must!(graph.add(final_sum, output_sum));
    let transformed = must!(graph.symbolic_vjp(loss, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.8, 1.1, 0.6])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    for name in ["initial", "scale"] {
        let plan = must!(transformed.graph.compile_cpu(transformed.gradients[name]));
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
            assert!(
                (actual - expected).abs() < 3e-5,
                "CUDA Scan VJP mismatch for {name}: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_fuses_grouped_scan_vjp_targets_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![3]));
    let bias = must!(body.input("bias", vec![3]));
    let scaled = must!(body.mul(carry, scale));
    let biased = must!(body.add(scaled, bias));
    let shifted = must!(body.add(biased, index));
    let next = must!(body.tanh(shifted));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![3]));
    let bias = must!(graph.input("bias", vec![3]));
    let (final_carry, outputs) = must!(graph.scan(
        initial,
        scan_plan,
        vec![("scale".to_string(), scale), ("bias".to_string(), bias)],
    ));
    let final_sum = must!(graph.sum(final_carry));
    let output_sum = must!(graph.sum(outputs));
    let loss = must!(graph.add(final_sum, output_sum));
    let transformed = must!(graph.symbolic_vjp(loss, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.8, 1.1, 0.6])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.1, -0.2, 0.05])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
        transformed.gradients["initial"],
        transformed.gradients["scale"],
        transformed.gradients["bias"],
    ]));
    let cpu = must!(plan.evaluate_many(&inputs));
    let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
    for (actual, expected) in cuda.iter().zip(cpu.iter()) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 4e-5,
                "grouped CUDA Scan VJP mismatch: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_reduces_scalar_scan_capture_vjp_on_device_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let next = must!(body.tanh(shifted));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![3]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let final_sum = must!(graph.sum(final_carry));
    let output_sum = must!(graph.sum(outputs));
    let loss = must!(graph.add(final_sum, output_sum));
    let transformed = must!(graph.symbolic_vjp(loss, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.2, -0.1, 0.3])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.8])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["scale"]));
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    assert!((cuda.data()[0] - cpu.data()[0]).abs() < 4e-5);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_reduces_broadcast_scan_capture_vjp_on_device_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![2, 3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![1, 3]));
    let scaled = must!(body.mul(carry, scale));
    let shifted = must!(body.add(scaled, index));
    let next = must!(body.tanh(shifted));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![2, 3]));
    let scale = must!(graph.input("scale", vec![1, 3]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let final_sum = must!(graph.sum(final_carry));
    let output_sum = must!(graph.sum(outputs));
    let loss = must!(graph.add(final_sum, output_sum));
    let transformed = must!(graph.symbolic_vjp(loss, "seed"));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3],
                vec![0.2, -0.1, 0.3, -0.4, 0.5, 0.1],
            )),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![1, 3], vec![0.8, 1.1, 0.6])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["scale"]));
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 6e-5);
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_rejects_cond_regions_before_device_lowering() {
    let mut on_true = TensorIr::new();
    let true_output = on_true.scalar_constant(1.0);
    let mut on_false = TensorIr::new();
    let false_output = on_false.scalar_constant(2.0);
    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let output = must!(graph.cond(predicate, branches));
    let inputs = BTreeMap::from([(
        "predicate".to_string(),
        must!(DynamicTensor::new(vec![], vec![1.0])),
    )]);
    let error = MlxBackend
        .execute(&must!(graph.compile_cpu(output)), &inputs)
        .expect_err("MLX must reject Cond until device-predicate lowering exists");
    assert!(error.contains("does not yet support Cond regions"));
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_fixed_fori_regions_with_device_resident_carry() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let increment = must!(body.mul(index, scale));
    let output = must!(body.add(carry, increment));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate(&inputs));
    let mlx = must!(MlxBackend.execute(&plan, &inputs));
    assert_eq!(mlx.shape(), cpu.shape());
    assert!((mlx.data()[0] - cpu.data()[0]).abs() < 1e-6);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_fixed_fori_symbolic_vjp_with_device_resident_tape() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let transformed = must!(graph.symbolic_vjp(output, "seed"));
    let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
        transformed.value,
        transformed.gradients["initial"],
        transformed.gradients["scale"],
    ]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate_many(&inputs));
    let mlx = must!(MlxBackend.execute_many(&plan, &output_ids, &inputs));
    for (actual, expected) in mlx.iter().zip(cpu) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_fixed_fori_forward_over_reverse_hvp() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let output = must!(body.add(scaled, index));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let vjp = must!(graph.symbolic_vjp(output, "cotangent"));
    let directional = must!(vjp.graph.symbolic_jvp_with_tangent_inputs(
        vjp.gradients["scale"],
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
    ));
    let plan = must!(directional.graph.compile_cpu(directional.tangent));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate(&inputs));
    let mlx = must!(MlxBackend.execute(&plan, &inputs));
    assert_eq!(mlx.shape(), cpu.shape());
    assert!((mlx.data()[0] - cpu.data()[0]).abs() < 1e-5);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_fixed_scan_regions_with_device_resident_carry() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index"
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)],));
    let (plan, output_ids) = must!(graph.compile_cpu_many(&[final_carry, outputs]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate_many(&inputs));
    let mlx = must!(MlxBackend.execute_many(&plan, &output_ids, &inputs));
    for (actual, expected) in mlx.iter().zip(cpu) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_fixed_scan_symbolic_vjp_with_device_resident_tape() {
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let scaled = must!(body.mul(carry, scale));
    let next = must!(body.add(scaled, index));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, next]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index"
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)]));
    let output_sum = must!(graph.sum(outputs));
    let total = must!(graph.add(final_carry, output_sum));
    let transformed = must!(graph.symbolic_vjp(total, "seed"));
    let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
        transformed.value,
        transformed.gradients["initial"],
        transformed.gradients["scale"],
    ]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    let cpu = must!(plan.evaluate_many(&inputs));
    let mlx = must!(MlxBackend.execute_many(&plan, &output_ids, &inputs));
    for (actual, expected) in mlx.iter().zip(cpu) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn symbolic_vjp_matches_direct_vjp_for_broadcasted_scalar_loss() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let bias = must!(graph.input("bias", vec![1, 3]));
    let shifted = must!(graph.add(x, bias));
    let activation = must!(graph.tanh(shifted));
    let loss = must!(graph.sum(activation));
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
    let cotangent = must!(DynamicTensor::filled(vec![], 1.0));
    let direct = must!(graph.vjp(loss, &inputs, cotangent.clone()));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let mut transformed_inputs = inputs;
    transformed_inputs.insert("loss_cotangent".to_string(), cotangent);

    for (name, gradient) in &transformed.gradients {
        assert_eq!(
            must!(transformed.graph.evaluate(*gradient, &transformed_inputs)),
            direct[name]
        );
    }
}

#[test]
fn symbolic_vjp_matches_direct_vjp_for_matmul_and_mean_axis() {
    let mut graph = TensorIr::new();
    let inputs_node = must!(graph.input("inputs", vec![2, 2]));
    let weights = must!(graph.input("weights", vec![2, 3]));
    let linear = must!(graph.matmul(inputs_node, weights));
    let activations = must!(graph.tanh(linear));
    let loss = must!(graph.mean_axis(activations, 0));
    let inputs = BTreeMap::from([
        (
            "inputs".to_string(),
            must!(DynamicTensor::new(vec![2, 2], vec![1.0, -2.0, 0.5, 3.0])),
        ),
        (
            "weights".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3],
                vec![0.2, -0.4, 0.6, -0.1, 0.3, 0.5],
            )),
        ),
    ]);
    let cotangent = must!(DynamicTensor::new(vec![3], vec![1.0, -0.5, 2.0]));
    let direct = must!(graph.vjp(loss, &inputs, cotangent.clone()));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let mut transformed_inputs = inputs;
    transformed_inputs.insert("loss_cotangent".to_string(), cotangent);

    for (name, gradient) in &transformed.gradients {
        let symbolic = must!(transformed.graph.evaluate(*gradient, &transformed_inputs));
        assert_eq!(symbolic.shape(), direct[name].shape());
        for (actual, expected) in symbolic.data().iter().zip(direct[name].data()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
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

    let source = must!(plan.cuda_source());
    assert!(source.contains("nabla_fused_elementwise"));
    assert!(source.contains("input_0[nabla_offset_0(index)]"));
    assert!(source.contains("input_1[nabla_offset_1(index)]"));
    assert!(source.contains("tanhf("));
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

fn two_cuda_mesh() -> TensorDeviceMesh {
    TensorDeviceMesh {
        devices: vec![
            TensorDeviceId {
                backend: TensorDeviceBackend::Cuda,
                ordinal: 0,
            },
            TensorDeviceId {
                backend: TensorDeviceBackend::Cuda,
                ordinal: 1,
            },
        ],
        axis_names: vec!["data".to_string()],
        shape: vec![2],
    }
}

#[test]
fn kernel_ir_accepts_valid_typed_mesh_sharding() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let output = must!(graph.add(x, x));
    let plan = must!(graph.compile_cpu(output));
    let placements = BTreeMap::from([(
        x,
        TensorPlacement::Mesh {
            mesh: two_cuda_mesh(),
            partition: TensorPartitionSpec::Sharded {
                tensor_axis: 0,
                mesh_axis: "data".to_string(),
            },
        },
    )]);

    let program = must!(plan.kernel_ir_with_placements(&placements));
    assert_eq!(
        program.nodes[x].placement.to_string(),
        "mesh[cuda:0,cuda:1]{sharded:tensor_axis=0,mesh_axis=data}"
    );
    must!(program.validate());
}

#[test]
fn kernel_ir_rejects_invalid_typed_mesh_sharding() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3, 2]));
    let plan = must!(graph.compile_cpu(x));
    let non_divisible = BTreeMap::from([(
        x,
        TensorPlacement::Mesh {
            mesh: two_cuda_mesh(),
            partition: TensorPartitionSpec::Sharded {
                tensor_axis: 0,
                mesh_axis: "data".to_string(),
            },
        },
    )]);
    let error = plan
        .kernel_ir_with_placements(&non_divisible)
        .expect_err("non-divisible shard must be rejected");
    assert!(error.contains("not divisible"));

    let unknown_axis = BTreeMap::from([(
        x,
        TensorPlacement::Mesh {
            mesh: two_cuda_mesh(),
            partition: TensorPartitionSpec::Sharded {
                tensor_axis: 0,
                mesh_axis: "model".to_string(),
            },
        },
    )]);
    let error = plan
        .kernel_ir_with_placements(&unknown_axis)
        .expect_err("unknown mesh axis must be rejected");
    assert!(error.contains("no axis \"model\""));

    let duplicate_device = TensorDeviceMesh {
        devices: vec![
            TensorDeviceId {
                backend: TensorDeviceBackend::Cuda,
                ordinal: 0,
            },
            TensorDeviceId {
                backend: TensorDeviceBackend::Cuda,
                ordinal: 0,
            },
        ],
        axis_names: vec!["data".to_string()],
        shape: vec![2],
    };
    let duplicate = BTreeMap::from([(
        x,
        TensorPlacement::Mesh {
            mesh: duplicate_device,
            partition: TensorPartitionSpec::Replicated,
        },
    )]);
    let error = plan
        .kernel_ir_with_placements(&duplicate)
        .expect_err("duplicate mesh device must be rejected");
    assert!(error.contains("devices must be unique"));

    let missing = BTreeMap::from([(99usize, TensorPlacement::Unplaced)]);
    let error = plan
        .kernel_ir_with_placements(&missing)
        .expect_err("missing placement node must be rejected");
    assert!(error.contains("does not exist"));
}

#[test]
fn kernel_ir_propagates_sharding_through_local_elementwise_and_transpose_ops() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let offset = graph.scalar_constant(1.0);
    let shifted = must!(graph.add(x, offset));
    let activated = must!(graph.tanh(shifted));
    let output = must!(graph.transpose(activated, Some(vec![1, 0])));
    let plan = must!(graph.compile_cpu(output));
    let placement = TensorPlacement::Mesh {
        mesh: two_cuda_mesh(),
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: 0,
            mesh_axis: "data".to_string(),
        },
    };

    let program =
        must!(plan.kernel_ir_with_placement_propagation(&BTreeMap::from([(x, placement,)])));

    assert_eq!(program.nodes[shifted].placement, program.nodes[x].placement);
    assert_eq!(
        program.nodes[activated].placement,
        program.nodes[x].placement
    );
    assert_eq!(
        program.nodes[output].placement,
        TensorPlacement::Mesh {
            mesh: two_cuda_mesh(),
            partition: TensorPartitionSpec::Sharded {
                tensor_axis: 1,
                mesh_axis: "data".to_string(),
            },
        }
    );
}

#[test]
fn kernel_ir_placement_propagation_rejects_collectives_and_implicit_redistribution() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let y = must!(graph.input("y", vec![4, 2]));
    let reduced = must!(graph.sum_axis(x, 0));
    let mixed = must!(graph.add(y, x));
    let reduction_plan = must!(graph.compile_cpu(reduced));
    let mixed_plan = must!(graph.compile_cpu(mixed));
    let sharded_x = TensorPlacement::Mesh {
        mesh: two_cuda_mesh(),
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: 0,
            mesh_axis: "data".to_string(),
        },
    };

    let error = reduction_plan
        .kernel_ir_with_placement_propagation(&BTreeMap::from([(x, sharded_x.clone())]))
        .expect_err("reduction over a sharded axis needs an all-reduce");
    assert!(error.contains("all-reduce"));

    let error = mixed_plan
        .kernel_ir_with_placement_propagation(&BTreeMap::from([(x, sharded_x)]))
        .expect_err("placed and unplaced tensors require explicit redistribution");
    assert!(error.contains("unplaced non-scalar"));
}

#[test]
fn sharding_plan_records_all_reduce_and_replication_for_sharded_mean_axis() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let output = must!(graph.mean_axis(x, 0));
    let plan = must!(graph.compile_cpu(output));
    let mesh = two_cuda_mesh();
    let sharding = must!(plan.sharding_plan(&BTreeMap::from([(
        x,
        TensorPlacement::Mesh {
            mesh: mesh.clone(),
            partition: TensorPartitionSpec::Sharded {
                tensor_axis: 0,
                mesh_axis: "data".to_string(),
            },
        },
    )])));

    assert_eq!(
        sharding.program.nodes[output].placement,
        TensorPlacement::Mesh {
            mesh: mesh.clone(),
            partition: TensorPartitionSpec::Replicated,
        }
    );
    assert_eq!(sharding.all_reduces.len(), 1);
    assert_eq!(sharding.all_reduces[0].node_id, output);
    assert_eq!(sharding.all_reduces[0].mesh, mesh);
    assert_eq!(sharding.all_reduces[0].mesh_axis, "data");
    assert_eq!(
        sharding.all_reduces[0].reduction,
        TensorReplicaReduction::Mean
    );
    must!(sharding.validate());
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[test]
fn cuda_data_parallel_validates_replica_device_contract_before_lowering() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let plan = must!(graph.compile_cpu(x));
    let backend = CudaBackend::new(0);

    let error = backend
        .compile_data_parallel(plan.clone(), vec![0])
        .expect_err("one CUDA device cannot form a data-parallel plan");
    assert!(error.contains("at least two"));

    let error = backend
        .compile_data_parallel(plan.clone(), vec![0, 0])
        .expect_err("duplicate CUDA device must be rejected");
    assert!(error.contains("must be unique"));

    let error = backend
        .compile_data_parallel(plan, vec![0, 1])
        .expect_err("the single-GPU test host must reject an unavailable second ordinal");
    assert!(error.contains("CUDA reports"));

    let _: fn(&CudaDataParallelExecutionPlan) -> usize =
        CudaDataParallelExecutionPlan::replica_count;
}

#[test]
fn data_parallel_vjp_matches_single_device_mean_loss() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 1]));
    let target = must!(graph.input("target", vec![4, 1]));
    let weight = must!(graph.input("weight", vec![1, 1]));
    let prediction = must!(graph.mul(x, weight));
    let error = must!(graph.sub(prediction, target));
    let squared_error = must!(graph.mul(error, error));
    let loss = must!(graph.mean(squared_error));
    let plan = must!(graph.compile_cpu(loss));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![4, 1], vec![-2.0, -1.0, 1.0, 2.0])),
        ),
        (
            "target".to_string(),
            must!(DynamicTensor::new(vec![4, 1], vec![-3.0, -1.0, 3.0, 5.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1, 1], vec![0.5])),
        ),
    ]);
    let cotangent = must!(DynamicTensor::new(vec![], vec![1.0]));
    let (single_value, single_gradients) = must!(plan.value_and_vjp(&inputs, cotangent.clone()));
    let mapped = BTreeSet::from(["x".to_string(), "target".to_string()]);
    let (parallel_value, parallel_gradients) = must!(plan.value_and_vjp_data_parallel(
        &inputs,
        cotangent,
        &mapped,
        2,
        TensorReplicaReduction::Mean,
    ));

    assert_eq!(parallel_value, single_value);
    assert_eq!(parallel_gradients, single_gradients);
}

#[test]
fn data_parallel_vjp_rejects_non_divisible_batch() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3, 1]));
    let loss = must!(graph.mean(x));
    let plan = must!(graph.compile_cpu(loss));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![3, 1], vec![1.0, 2.0, 3.0])),
    )]);
    let error = plan
        .value_and_vjp_data_parallel(
            &inputs,
            must!(DynamicTensor::new(vec![], vec![1.0])),
            &BTreeSet::from(["x".to_string()]),
            2,
            TensorReplicaReduction::Mean,
        )
        .expect_err("non-divisible batch must be rejected");
    assert!(error.contains("not divisible"));
}

#[test]
fn compile_cpu_folds_scalar_constant_subgraphs() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let two = graph.scalar_constant(2.0);
    let three = graph.scalar_constant(3.0);
    let five = must!(graph.add(two, three));
    let output = must!(graph.mul(x, five));
    let plan = must!(graph.compile_cpu(output));
    assert_eq!(plan.node_count(), 3);
    assert!(plan.lower_text().contains("constant[value=5]"));
    assert!(!plan.lower_text().contains("constant[value=2]"));
    assert!(!plan.lower_text().contains("constant[value=3]"));
    assert!(!plan.lower_text().contains("add(%"));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![1], vec![4.0])),
    )]);
    assert_eq!(must!(CpuBackend.execute(&plan, &inputs)).data(), &[20.0]);
}

#[test]
fn compile_cpu_canonicalizes_identity_broadcast_and_reshape_chains() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 2]));
    let same_shape = must!(graph.broadcast_to(x, vec![2, 2]));
    let flattened = must!(graph.reshape(same_shape, vec![4]));
    let output = must!(graph.reshape(flattened, vec![2, 2]));
    let plan = must!(graph.compile_cpu(output));

    assert_eq!(plan.node_count(), 1);
    assert_eq!(plan.lower_text(), "%0 = input[name=x] : tensor<2x2xf64>");
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
    )]);
    assert_eq!(
        must!(CpuBackend.execute(&plan, &inputs)).data(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn compile_cpu_composes_non_identity_reshape_chains() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 2]));
    let flattened = must!(graph.reshape(x, vec![4]));
    let output = must!(graph.reshape(flattened, vec![1, 4]));
    let plan = must!(graph.compile_cpu(output));

    assert_eq!(plan.node_count(), 2);
    assert_eq!(plan.lower_text().matches("reshape(").count(), 1);
    assert!(plan.lower_text().contains("reshape(%0)"));
}

#[test]
fn fusion_regions_isolate_an_elementwise_tail_after_matmul() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let weight = must!(graph.input("weight", vec![3, 4]));
    let bias = must!(graph.input("bias", vec![1, 4]));
    let product = must!(graph.matmul(x, weight));
    let shifted = must!(graph.add(product, bias));
    let activated = must!(graph.tanh(shifted));
    let output = must!(graph.sin(activated));

    let regions = must!(graph.compile_cpu(output)).fusion_regions();
    assert_eq!(
        regions,
        vec![TensorFusionRegion {
            output_node_id: 6,
            node_ids: vec![4, 5, 6],
            input_node_ids: vec![2, 3],
        }]
    );
}

#[test]
fn fusion_regions_materialize_values_with_non_elementwise_consumers() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let weight = must!(graph.input("weight", vec![3, 4]));
    let activated = must!(graph.tanh(x));
    let _product = must!(graph.matmul(activated, weight));
    let cosine = must!(graph.cos(activated));
    let output = must!(graph.sin(cosine));

    let (plan, _) = must!(graph.compile_cpu_many(&[_product, output]));
    let regions = plan.fusion_regions();
    assert_eq!(
        regions,
        vec![TensorFusionRegion {
            output_node_id: 5,
            node_ids: vec![4, 5],
            input_node_ids: vec![2],
        }]
    );
}

#[test]
fn fusion_regions_do_not_duplicate_a_value_shared_by_two_elementwise_tails() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let shared = must!(graph.tanh(x));
    let left = must!(graph.sin(shared));
    let right = must!(graph.cos(shared));
    let left_loss = must!(graph.sum(left));

    let (plan, _) = must!(graph.compile_cpu_many(&[left_loss, right]));
    assert!(plan.fusion_regions().is_empty());
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_fusion_region_source_materializes_matmul_and_bias_leaves() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let weight = must!(graph.input("weight", vec![3, 4]));
    let bias = must!(graph.input("bias", vec![1, 4]));
    let product = must!(graph.matmul(x, weight));
    let shifted = must!(graph.add(product, bias));
    let activated = must!(graph.tanh(shifted));
    let output = must!(graph.sin(activated));
    let plan = must!(graph.compile_cpu(output));
    let region = plan.fusion_regions().pop().expect("expected fusion region");
    let source = must!(plan.cuda_fusion_region_source(&region));

    assert!(source.contains("nabla_fused_region_6"));
    assert!(source.contains("const float* input_2"));
    assert!(source.contains("const float* input_3"));
    assert!(source.contains("tanhf("));
    assert!(source.contains("sinf("));
}

#[test]
fn buffer_plan_reuses_temporary_slots_after_their_final_use() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let y = must!(graph.input("y", vec![2]));
    let z = must!(graph.input("z", vec![2]));
    let q = must!(graph.input("q", vec![2]));
    let r = must!(graph.input("r", vec![2]));
    let first = must!(graph.add(x, y));
    let second = must!(graph.add(first, z));
    let third = must!(graph.add(second, q));
    let output = must!(graph.add(third, r));

    let buffers = must!(must!(graph.compile_cpu(output)).buffer_plan());
    assert_eq!(buffers.slots.len(), 2);
    assert_eq!(
        buffers.slots,
        vec![
            TensorBufferSlot {
                id: 0,
                element_count: 2,
            },
            TensorBufferSlot {
                id: 1,
                element_count: 2,
            },
        ]
    );
    assert_eq!(
        buffers.node_slots,
        vec![
            None,
            None,
            None,
            None,
            None,
            Some(0),
            Some(1),
            Some(0),
            Some(1)
        ]
    );
}

#[test]
fn buffer_plan_preserves_reshape_as_an_input_alias() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 2]));
    let output = must!(graph.reshape(x, vec![4]));

    let buffers = must!(must!(graph.compile_cpu(output)).buffer_plan());
    assert!(buffers.slots.is_empty());
    assert_eq!(buffers.node_slots, vec![None, None]);
    assert_eq!(buffers.node_aliases, vec![None, Some(0)]);
    assert_eq!(buffers.output_backing_node_id, 0);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_fused_elementwise_plan_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let bias = must!(graph.input("bias", vec![1, 3]));
    let shifted = must!(graph.add(x, bias));
    let output = must!(graph.tanh(shifted));
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

    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_execution_plan_reuses_buffers_for_updated_inputs_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let bias = must!(graph.input("bias", vec![1, 2]));
    let shifted = must!(graph.add(x, bias));
    let output = must!(graph.tanh(shifted));
    let plan = must!(graph.compile_cpu(output));
    let cuda = must!(CudaBackend::new(0).compile(plan.clone()));
    let first_inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![-1.0, 2.0])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![0.0, 1.0])),
        ),
    ]);
    let second_inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![0.5, -0.25])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![0.2, -0.4])),
        ),
    ]);

    for inputs in [&first_inputs, &second_inputs] {
        let expected = must!(plan.evaluate(inputs));
        let actual = must!(cuda.execute(inputs));
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_execution_plan_updates_a_retained_parameter_with_device_sgd_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let weight = must!(graph.input("weight", vec![1]));
    let target = must!(graph.input("target", vec![1]));
    let prediction = must!(graph.mul(x, weight));
    let error = must!(graph.sub(prediction, target));
    let squared_error = must!(graph.mul(error, error));
    let loss = must!(graph.sum(squared_error));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let gradient_plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["weight"]));
    let cuda = must!(CudaBackend::new(0).compile(gradient_plan));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1], vec![2.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.0])),
        ),
        (
            "target".to_string(),
            must!(DynamicTensor::new(vec![1], vec![6.0])),
        ),
        (
            "loss_cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let retained = std::collections::BTreeSet::from(["weight".to_string()]);

    for _ in 0..20 {
        must!(cuda.execute_retaining(&inputs, &retained));
        must!(cuda.sgd_step_input_from_output("weight", 0.1));
    }

    let weight = must!(cuda.retained_input_to_host("weight"));
    assert!((weight.data()[0] - 3.0).abs() < 1e-5);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_execution_plan_updates_a_retained_parameter_with_device_adam_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let weight = must!(graph.input("weight", vec![1]));
    let target = must!(graph.input("target", vec![1]));
    let prediction = must!(graph.mul(x, weight));
    let error = must!(graph.sub(prediction, target));
    let squared_error = must!(graph.mul(error, error));
    let loss = must!(graph.sum(squared_error));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["weight"]));
    let cuda = must!(CudaBackend::new(0).compile(plan));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1], vec![2.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.0])),
        ),
        (
            "target".to_string(),
            must!(DynamicTensor::new(vec![1], vec![6.0])),
        ),
        (
            "loss_cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let retained = std::collections::BTreeSet::from(["weight".to_string()]);

    for _ in 0..2 {
        must!(cuda.execute_retaining(&inputs, &retained));
        must!(cuda.adam_step_input_from_output("weight", 0.1, 0.9, 0.999, 1e-8));
    }

    let weight = must!(cuda.retained_input_to_host("weight"));
    assert!(weight.data()[0] > 0.19 && weight.data()[0] < 0.21);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_execution_plans_synchronize_multi_parameter_device_sgd_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let weight = must!(graph.input("weight", vec![1]));
    let bias = must!(graph.input("bias", vec![1]));
    let target = must!(graph.input("target", vec![2]));
    let weighted = must!(graph.mul(x, weight));
    let linear = must!(graph.add(weighted, bias));
    let error = must!(graph.sub(linear, target));
    let squared_error = must!(graph.mul(error, error));
    let loss = must!(graph.sum(squared_error));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let weight_plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["weight"]));
    let bias_plan = must!(transformed.graph.compile_cpu(transformed.gradients["bias"]));
    let weight_cuda = must!(CudaBackend::new(0).compile(weight_plan));
    let bias_cuda = must!(CudaBackend::new(0).compile(bias_plan));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2], vec![-1.0, 1.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.0])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.0])),
        ),
        (
            "target".to_string(),
            must!(DynamicTensor::new(vec![2], vec![-1.0, 3.0])),
        ),
        (
            "loss_cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let retained = std::collections::BTreeSet::from(["weight".to_string(), "bias".to_string()]);

    for _ in 0..100 {
        must!(weight_cuda.execute_retaining(&inputs, &retained));
        must!(bias_cuda.execute_retaining(&inputs, &retained));
        must!(weight_cuda.sgd_step_input_from_output("weight", 0.05));
        must!(bias_cuda.sgd_step_input_from_output("bias", 0.05));
        must!(weight_cuda.sync_retained_input_to("weight", &bias_cuda, "weight"));
        must!(bias_cuda.sync_retained_input_to("bias", &weight_cuda, "bias"));
    }

    let weight = must!(weight_cuda.retained_input_to_host("weight"));
    let bias = must!(bias_cuda.retained_input_to_host("bias"));
    assert!((weight.data()[0] - 2.0).abs() < 1e-4);
    assert!((bias.data()[0] - 1.0).abs() < 1e-4);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_execution_plans_synchronize_multi_parameter_device_adam_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let weight = must!(graph.input("weight", vec![1]));
    let bias = must!(graph.input("bias", vec![1]));
    let target = must!(graph.input("target", vec![2]));
    let weighted = must!(graph.mul(x, weight));
    let prediction = must!(graph.add(weighted, bias));
    let error = must!(graph.sub(prediction, target));
    let squared_error = must!(graph.mul(error, error));
    let loss = must!(graph.sum(squared_error));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let weight_plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["weight"]));
    let bias_plan = must!(transformed.graph.compile_cpu(transformed.gradients["bias"]));
    let weight_cuda = must!(CudaBackend::new(0).compile(weight_plan));
    let bias_cuda = must!(CudaBackend::new(0).compile(bias_plan));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2], vec![-1.0, 1.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.0])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1], vec![0.0])),
        ),
        (
            "target".to_string(),
            must!(DynamicTensor::new(vec![2], vec![-1.0, 3.0])),
        ),
        (
            "loss_cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let retained = std::collections::BTreeSet::from([
        "x".to_string(),
        "weight".to_string(),
        "bias".to_string(),
        "target".to_string(),
        "loss_cotangent".to_string(),
    ]);

    for _ in 0..200 {
        must!(weight_cuda.execute_retaining(&inputs, &retained));
        must!(bias_cuda.execute_retaining(&inputs, &retained));
        must!(weight_cuda.adam_step_input_from_output("weight", 0.05, 0.9, 0.999, 1e-8));
        must!(bias_cuda.adam_step_input_from_output("bias", 0.05, 0.9, 0.999, 1e-8));
        must!(weight_cuda.sync_retained_input_to("weight", &bias_cuda, "weight"));
        must!(bias_cuda.sync_retained_input_to("bias", &weight_cuda, "bias"));
    }

    let weight = must!(weight_cuda.retained_input_to_host("weight"));
    let bias = must!(bias_cuda.retained_input_to_host("bias"));
    assert!((weight.data()[0] - 2.0).abs() < 1e-3);
    assert!((bias.data()[0] - 1.0).abs() < 1e-3);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_direct_rank_two_matmul_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let lhs = must!(graph.input("lhs", vec![3, 2]));
    let rhs = must!(graph.input("rhs", vec![2, 4]));
    let output = must!(graph.matmul(lhs, rhs));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "lhs".to_string(),
            must!(DynamicTensor::new(
                vec![3, 2],
                vec![1.0, -2.0, 0.5, 3.0, -1.0, 4.0]
            )),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(
                vec![2, 4],
                vec![0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.5, -0.7],
            )),
        ),
    ]);

    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_tiled_rank_two_matmul_inside_generic_plan_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let lhs = must!(graph.input("lhs", vec![32, 32]));
    let rhs = must!(graph.input("rhs", vec![32, 32]));
    let matmul = must!(graph.matmul(lhs, rhs));
    let output = must!(graph.tanh(matmul));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "lhs".to_string(),
            must!(DynamicTensor::new(
                vec![32, 32],
                (0..32 * 32)
                    .map(|index| (index % 13) as f64 * 0.05 - 0.3)
                    .collect(),
            )),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(
                vec![32, 32],
                (0..32 * 32)
                    .map(|index| (index % 17) as f64 * -0.04 + 0.28)
                    .collect(),
            )),
        ),
    ]);

    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_broadcast_batched_matmul_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let lhs = must!(graph.input("lhs", vec![2, 3, 2]));
    let rhs = must!(graph.input("rhs", vec![1, 2, 4]));
    let output = must!(graph.matmul(lhs, rhs));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "lhs".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3, 2],
                vec![1.0, -2.0, 0.5, 3.0, -1.0, 4.0, 2.0, 1.0, -3.0, 0.25, 0.75, -2.0,],
            )),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(
                vec![1, 2, 4],
                vec![0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.5, -0.7],
            )),
        ),
    ]);

    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), &[2, 3, 4]);
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_broadcast_batched_matmul_vjp_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let lhs = must!(graph.input("lhs", vec![2, 3, 2]));
    let rhs = must!(graph.input("rhs", vec![1, 2, 4]));
    let output = must!(graph.matmul(lhs, rhs));
    let loss = must!(graph.sum(output));
    let inputs = BTreeMap::from([
        (
            "lhs".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3, 2],
                vec![1.0, -2.0, 0.5, 3.0, -1.0, 4.0, 2.0, 1.0, -3.0, 0.25, 0.75, -2.0,],
            )),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(
                vec![1, 2, 4],
                vec![0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.5, -0.7],
            )),
        ),
    ]);
    let cotangent = must!(DynamicTensor::filled(vec![], 1.0));
    let cpu = must!(graph.vjp(loss, &inputs, cotangent.clone()));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let mut cuda_inputs = inputs;
    cuda_inputs.insert("loss_cotangent".to_string(), cotangent);

    for name in ["lhs", "rhs"] {
        let plan = must!(transformed.graph.compile_cpu(transformed.gradients[name]));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &cuda_inputs));
        assert_eq!(cuda.shape(), cpu[name].shape());
        for (actual, expected) in cuda.data().iter().zip(cpu[name].data()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_broadcast_batched_matmul_trains_with_retained_adam_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let lhs = must!(graph.input("lhs", vec![2, 1, 1]));
    let weight = must!(graph.input("weight", vec![1, 1, 1]));
    let target = must!(graph.input("target", vec![2, 1, 1]));
    let prediction = must!(graph.matmul(lhs, weight));
    let error = must!(graph.sub(prediction, target));
    let squared_error = must!(graph.mul(error, error));
    let loss = must!(graph.mean(squared_error));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let plan = must!(transformed
        .graph
        .compile_cpu(transformed.gradients["weight"]));
    let cuda = must!(CudaBackend::new(0).compile(plan));
    let inputs = BTreeMap::from([
        (
            "lhs".to_string(),
            must!(DynamicTensor::new(vec![2, 1, 1], vec![1.0, 2.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1, 1, 1], vec![0.0])),
        ),
        (
            "target".to_string(),
            must!(DynamicTensor::new(vec![2, 1, 1], vec![2.0, 4.0])),
        ),
        (
            "loss_cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let retained = std::collections::BTreeSet::from([
        "lhs".to_string(),
        "target".to_string(),
        "loss_cotangent".to_string(),
        "weight".to_string(),
    ]);

    for _ in 0..150 {
        must!(cuda.execute_retaining(&inputs, &retained));
        must!(cuda.adam_step_input_from_output("weight", 0.05, 0.9, 0.999, 1e-8));
    }

    let weight = must!(cuda.retained_input_to_host("weight"));
    assert!((weight.data()[0] - 2.0).abs() < 1e-3);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_two_layer_mlp_scalar_loss_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3, 1]));
    let weight_one = must!(graph.input("weight_one", vec![1, 4]));
    let bias_one = must!(graph.input("bias_one", vec![1, 4]));
    let weight_two = must!(graph.input("weight_two", vec![4, 1]));
    let hidden_linear = must!(graph.matmul(x, weight_one));
    let hidden_shifted = must!(graph.add(hidden_linear, bias_one));
    let hidden = must!(graph.tanh(hidden_shifted));
    let output = must!(graph.matmul(hidden, weight_two));
    let squared = must!(graph.mul(output, output));
    let loss = must!(graph.sum(squared));
    let plan = must!(graph.compile_cpu(loss));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![3, 1], vec![-1.0, 0.5, 2.0])),
        ),
        (
            "weight_one".to_string(),
            must!(DynamicTensor::new(vec![1, 4], vec![0.2, -0.4, 0.6, 0.8])),
        ),
        (
            "bias_one".to_string(),
            must!(DynamicTensor::new(vec![1, 4], vec![0.1, 0.3, -0.2, 0.5])),
        ),
        (
            "weight_two".to_string(),
            must!(DynamicTensor::new(vec![4, 1], vec![0.7, -0.5, 0.9, 0.2])),
        ),
    ]);

    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    assert!((cuda.data()[0] - cpu.data()[0]).abs() < 1e-5);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_symbolic_vjp_for_broadcast_bias_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3, 1]));
    let weight = must!(graph.input("weight", vec![1, 2]));
    let bias = must!(graph.input("bias", vec![1, 2]));
    let linear = must!(graph.matmul(x, weight));
    let shifted = must!(graph.add(linear, bias));
    let output = must!(graph.tanh(shifted));
    let squared = must!(graph.mul(output, output));
    let loss = must!(graph.sum(squared));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![3, 1], vec![-1.0, 0.5, 2.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![0.2, -0.4])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![0.1, 0.3])),
        ),
    ]);
    let cotangent = must!(DynamicTensor::filled(vec![], 1.0));
    let direct = must!(graph.vjp(loss, &inputs, cotangent.clone()));
    let transformed = must!(graph.symbolic_vjp(loss, "loss_cotangent"));
    let mut transformed_inputs = inputs;
    transformed_inputs.insert("loss_cotangent".to_string(), cotangent);

    for name in ["weight", "bias"] {
        let gradient = transformed.gradients[name];
        let plan = must!(transformed.graph.compile_cpu(gradient));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &transformed_inputs));
        assert_eq!(cuda.shape(), direct[name].shape());
        for (actual, expected) in cuda.data().iter().zip(direct[name].data()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "gradient mismatch for {name}"
            );
        }
    }
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
fn symbolic_jvp_with_tangent_inputs_uses_runtime_directions() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let weight = must!(graph.input("weight", vec![2]));
    let output = must!(graph.mul(x, weight));
    let tangents = BTreeMap::from([
        ("x".to_string(), "x_tangent".to_string()),
        ("weight".to_string(), "weight_tangent".to_string()),
    ]);
    let transformed = must!(graph.symbolic_jvp_with_tangent_inputs(output, &tangents));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2], vec![2.0, -3.0])),
        ),
        (
            "weight".to_string(),
            must!(DynamicTensor::new(vec![2], vec![4.0, 5.0])),
        ),
        (
            "x_tangent".to_string(),
            must!(DynamicTensor::new(vec![2], vec![0.5, 2.0])),
        ),
        (
            "weight_tangent".to_string(),
            must!(DynamicTensor::new(vec![2], vec![1.5, -0.25])),
        ),
    ]);

    let value = must!(transformed.graph.evaluate(transformed.value, &inputs));
    let tangent = must!(transformed.graph.evaluate(transformed.tangent, &inputs));
    assert_eq!(value.data(), &[8.0, -15.0]);
    assert_eq!(tangent.data(), &[5.0, 10.75]);
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
    assert_eq!(tangent.shape(), &[] as &[usize]);
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

#[test]
fn solve_evaluates_and_differentiates() {
    let mut graph = TensorIr::new();
    let matrix = must!(graph.input("matrix", vec![2, 2]));
    let rhs = must!(graph.input("rhs", vec![2, 1]));
    let output = must!(graph.solve(matrix, rhs));
    let inputs = BTreeMap::from([
        (
            "matrix".to_string(),
            must!(DynamicTensor::new(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![9.0, 8.0])),
        ),
    ]);
    let value = must!(graph.evaluate(output, &inputs));
    assert_eq!(value.data(), &[2.0, 3.0]);

    let tangents = BTreeMap::from([
        (
            "matrix".to_string(),
            must!(DynamicTensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 0.0])),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![0.0, 1.0])),
        ),
    ]);
    let (_, tangent) = must!(graph.jvp(output, &inputs, &tangents));
    for (actual, expected) in tangent.data().iter().zip([-1.0, 1.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }

    let upstream = must!(DynamicTensor::new(vec![2, 1], vec![1.0, 1.0]));
    let gradients = must!(graph.vjp(output, &inputs, upstream));
    for (actual, expected) in gradients["rhs"].data().iter().zip([0.2, 0.4]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in gradients["matrix"]
        .data()
        .iter()
        .zip([-0.4, -0.6, -0.8, -1.2])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn solve_rejects_unsupported_shapes_and_singular_matrices() {
    let mut graph = TensorIr::new();
    let matrix = must!(graph.input("matrix", vec![2, 3]));
    let rhs = must!(graph.input("rhs", vec![2, 1]));
    assert!(graph.solve(matrix, rhs).is_err());

    let mut graph = TensorIr::new();
    let matrix = must!(graph.input("matrix", vec![2, 2]));
    let rhs = must!(graph.input("rhs", vec![2, 1]));
    let output = must!(graph.solve(matrix, rhs));
    let inputs = BTreeMap::from([
        (
            "matrix".to_string(),
            must!(DynamicTensor::new(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0])),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![1.0, 2.0])),
        ),
    ]);
    assert!(graph.evaluate(output, &inputs).is_err());
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_rank_two_solve_when_enabled() {
    if std::env::var_os("NABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let matrix = must!(graph.input("matrix", vec![2, 2]));
    let rhs = must!(graph.input("rhs", vec![2, 1]));
    let output = must!(graph.solve(matrix, rhs));
    let inputs = BTreeMap::from([
        (
            "matrix".to_string(),
            must!(DynamicTensor::new(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![9.0, 8.0])),
        ),
    ]);
    let plan = must!(graph.compile_cpu(output));
    let cpu = must!(CpuBackend.execute(&plan, &inputs));
    let cuda = must!(CudaBackend::default().execute(&plan, &inputs));
    assert_eq!(cuda.shape(), cpu.shape());
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_matches_cpu_for_concat_broadcast_and_tanh() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1, 2]));
    let y = must!(graph.input("y", vec![1, 2]));
    let bias = must!(graph.input("bias", vec![1, 2]));
    let joined = must!(graph.concat(vec![x, y], 0));
    let bias = must!(graph.broadcast_to(bias, vec![2, 2]));
    let shifted = must!(graph.add(joined, bias));
    let zero = graph.scalar_constant(0.0);
    let mask = must!(graph.greater(shifted, zero));
    let tanh = must!(graph.tanh(shifted));
    let activated = must!(graph.where_select(mask, tanh, shifted));
    let reduced = must!(graph.sum_axis(activated, 0));
    let output = must!(graph.sum(reduced));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![-1.0, 2.0])),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![0.5, -0.25])),
        ),
        (
            "bias".to_string(),
            must!(DynamicTensor::new(vec![1, 2], vec![0.25, -0.5])),
        ),
    ]);
    let plan = must!(graph.compile_cpu(output));
    let cpu = must!(CpuBackend.execute(&plan, &inputs));
    let mlx = must!(MlxBackend.execute(&plan, &inputs));
    assert_eq!(mlx.shape(), cpu.shape());
    for (actual, expected) in mlx.data().iter().zip(cpu.data()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "actual={actual}, expected={expected}"
        );
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_serializes_concurrent_execution_across_threads() {
    // MLX 的預設 GPU stream 為全行程共用；若後端未序列化，多執行緒同時 eval
    // 會觸發 Metal "uncommitted encoder" assertion 或卡死。Fori 每步都會 eval，
    // 可放大交錯機率。
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let increment = must!(body.mul(index, scale));
    let output = must!(body.add(carry, increment));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        16,
        must!(body.compile_cpu(output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let scale = must!(graph.input("scale", vec![]));
    let output = must!(graph.fori(initial, loop_plan, vec![("scale".to_string(), scale)]));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    let expected = must!(plan.evaluate(&inputs)).data()[0];
    let results = std::thread::scope(|scope| {
        let workers = (0..4)
            .map(|_| {
                scope.spawn(|| -> Result<(), String> {
                    for _ in 0..16 {
                        let actual = MlxBackend.execute(&plan, &inputs)?.data()[0];
                        if (actual - expected).abs() >= 1e-5 {
                            return Err(format!("actual={actual}, expected={expected}"));
                        }
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("MLX worker thread panicked"))
            .collect::<Vec<_>>()
    });
    for result in results {
        must!(result);
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_rejects_solve_until_a_gpu_implementation_exists() {
    let mut graph = TensorIr::new();
    let matrix = must!(graph.input("matrix", vec![2, 2]));
    let rhs = must!(graph.input("rhs", vec![2, 1]));
    let output = must!(graph.solve(matrix, rhs));
    let inputs = BTreeMap::from([
        (
            "matrix".to_string(),
            must!(DynamicTensor::new(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])),
        ),
        (
            "rhs".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![9.0, 8.0])),
        ),
    ]);
    let plan = must!(graph.compile_cpu(output));
    let error = MlxBackend
        .execute(&plan, &inputs)
        .expect_err("MLX solve must not fall back to CPU");
    assert!(error.contains("does not yet support solve"));
}

#[test]
fn compile_cpu_many_preserves_every_requested_output_node() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let value = must!(graph.tanh(x));
    let tangent = must!(graph.mul(value, value));
    let (plan, outputs) = must!(graph.compile_cpu_many(&[value, tangent]));

    assert_eq!(outputs.len(), 2);
    assert_eq!(plan.output_node_ids(), outputs.as_slice());
    assert_ne!(outputs[0], outputs[1]);
}
