use std::collections::{BTreeMap, BTreeSet};

use quabla_core::tensor_ir::{
    CpuBackend, DynamicTensor, TensorBackend, TensorBufferSlot, TensorCondExecutionPlan,
    TensorDType, TensorDeviceBackend, TensorDeviceId, TensorDeviceMesh, TensorExecutionPlan,
    TensorForiExecutionPlan, TensorForiMultiExecutionPlan, TensorForiVjpJvpExecutionPlan,
    TensorFusionRegion, TensorIr, TensorNodeId, TensorPartitionSpec, TensorPlacement,
    TensorReplicaReduction, TensorScanExecutionPlan, TensorShardingPlan,
};
use quabla_core::{QuablaCompiler, QuablaMultiOutputProgram, QuablaTarget};

#[cfg(all(feature = "cuda", target_os = "linux"))]
use quabla_core::tensor_ir::CudaBackend;

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
use quabla_core::tensor_ir::CudaDataParallelExecutionPlan;

#[cfg(all(feature = "mlx", target_os = "macos"))]
use quabla_core::tensor_ir::{MlxBackend, MlxRetainedInputs};

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
        quabla_core::tensor_ir::TensorBackendPrecision {
            logical: TensorDType::F64,
            execution: TensorDType::F64,
        }
    );
    for backend in [TensorDeviceBackend::Cuda, TensorDeviceBackend::Mlx] {
        assert_eq!(backend.precision().logical, TensorDType::F64);
        assert_eq!(backend.precision().execution, TensorDType::F32);
        assert_eq!(
            backend.execution_dtype(TensorDType::F64),
            Ok(TensorDType::F32)
        );
        assert_eq!(
            backend.execution_dtype(TensorDType::F32),
            Ok(TensorDType::F32)
        );
    }
    for dtype in [TensorDType::F32, TensorDType::F64] {
        assert_eq!(TensorDeviceBackend::Cpu.execution_dtype(dtype), Ok(dtype));
    }
}

#[test]
fn compiler_facade_owns_program_transform_and_cpu_execution_lifecycle() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let squared = must!(graph.mul(x, x));
    let compiler = QuablaCompiler;
    let program = must!(compiler.program(graph, squared));

    assert_eq!(program.output_shape(), Ok(vec![2]));
    assert!(program.lower_text().contains("mul"));
    assert!(compiler.capability(QuablaTarget::Cpu).built);

    let executable = must!(compiler.compile(&program, QuablaTarget::Cpu));
    assert_eq!(executable.target(), QuablaTarget::Cpu);
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
fn compiler_facade_rejects_unbuilt_targets_up_front() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let doubled = must!(graph.add(x, x));
    let compiler = QuablaCompiler;
    let program = must!(compiler.program(graph, doubled));

    for target in [QuablaTarget::Cuda { device_ordinal: 0 }, QuablaTarget::Mlx] {
        if target.is_built() {
            continue;
        }
        let error = compiler
            .compile(&program, target)
            .expect_err("the facade must reject a target missing from this build");
        assert_eq!(
            error,
            format!("{} target is unavailable in this build", target.name())
        );
    }
}

#[test]
fn compiler_build_check_bypass_leaves_unbuilt_errors_to_the_backend() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let doubled = must!(graph.add(x, x));
    let compiler = QuablaCompiler;
    let program = must!(compiler.program(graph, doubled));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2], vec![1.0, 2.0])),
    )]);

    // The Python compatibility helpers keep their pre-facade error contract.
    let cuda = QuablaTarget::Cuda { device_ordinal: 0 };
    if !cuda.is_built() {
        let error = compiler
            .compile_without_build_check(&program, cuda)
            .expect_err("an unbuilt CUDA backend must reject compilation");
        assert!(error.contains("build Quabla on Linux with --features cuda"));
    }
    if !QuablaTarget::Mlx.is_built() {
        let executable = must!(compiler.compile_without_build_check(&program, QuablaTarget::Mlx));
        let error = executable
            .execute(&inputs)
            .expect_err("an unbuilt MLX backend must reject execution");
        assert!(error.contains("--features mlx"));
    }
}

#[test]
fn compiler_facade_multi_output_program_keeps_order_and_owns_frozen_ids() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    // A pruned node before the outputs shifts every frozen id away from its
    // source id, and a duplicate tanh is aliased by structural CSE.
    let _unused = must!(graph.add(x, x));
    let hidden = must!(graph.tanh(x));
    let duplicate = must!(graph.tanh(x));
    let squared = must!(graph.mul(hidden, hidden));
    let loss = must!(graph.sum(squared));
    let outputs = vec![loss, duplicate, hidden];
    let compiler = QuablaCompiler;
    let program = must!(QuablaMultiOutputProgram::new(
        graph.clone(),
        outputs.clone()
    ));
    assert_eq!(program.output_node_ids(), outputs.as_slice());

    let executable = must!(compiler.compile_many(&program, QuablaTarget::Cpu));
    assert_eq!(executable.target(), QuablaTarget::Cpu);
    let frozen = executable.output_node_ids().to_vec();
    assert_eq!(frozen.len(), outputs.len());
    assert_eq!(frozen[1], frozen[2], "CSE aliases the duplicate output");
    assert_ne!(frozen, outputs, "freezing remaps pruned source ids");

    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2], vec![0.5, -1.25])),
    )]);
    let values = must!(executable.execute(&inputs));
    assert_eq!(values.len(), outputs.len());
    for (value, output) in values.iter().zip(&outputs) {
        let single = must!(compiler.compile(
            &must!(compiler.program(graph.clone(), *output)),
            QuablaTarget::Cpu
        ));
        assert_eq!(*value, must!(single.execute(&inputs)));
    }

    assert_eq!(
        QuablaMultiOutputProgram::new(graph.clone(), Vec::new()).unwrap_err(),
        "execution plan requires at least one output"
    );
    assert_eq!(
        QuablaMultiOutputProgram::new(graph, vec![loss, 999]).unwrap_err(),
        "node 999 does not exist"
    );
}

#[test]
fn compiler_facade_multi_output_build_check_matches_single_output() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let doubled = must!(graph.add(x, x));
    let squared = must!(graph.mul(x, x));
    let compiler = QuablaCompiler;
    let program = must!(QuablaMultiOutputProgram::new(graph, vec![doubled, squared]));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2], vec![1.0, 2.0])),
    )]);

    for target in [QuablaTarget::Cuda { device_ordinal: 0 }, QuablaTarget::Mlx] {
        if target.is_built() {
            continue;
        }
        let error = compiler
            .compile_many(&program, target)
            .expect_err("the facade must reject a target missing from this build");
        assert_eq!(
            error,
            format!("{} target is unavailable in this build", target.name())
        );
    }
    let cuda = QuablaTarget::Cuda { device_ordinal: 0 };
    if !cuda.is_built() {
        let error = compiler
            .compile_many_without_build_check(&program, cuda)
            .expect_err("an unbuilt CUDA backend must reject compilation");
        assert!(error.contains("build Quabla on Linux with --features cuda"));
    }
    if !QuablaTarget::Mlx.is_built() {
        let executable =
            must!(compiler.compile_many_without_build_check(&program, QuablaTarget::Mlx));
        let error = executable
            .execute(&inputs)
            .expect_err("an unbuilt MLX backend must reject execution");
        assert!(error.contains("--features mlx"));
    }
}

/// A scalar value with two named gradients, frozen as one ordered program,
/// mirroring the value-and-gradient Python helpers.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn multi_output_value_and_grad_program(
) -> Result<(QuablaMultiOutputProgram, BTreeMap<String, DynamicTensor>), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![3, 2])?;
    let weight = graph.input("weight", vec![2, 1])?;
    let hidden = graph.matmul(x, weight)?;
    let activated = graph.tanh(hidden)?;
    let loss = graph.sum(activated)?;
    let vjp = graph.symbolic_vjp(loss, "cotangent")?;
    let outputs = vec![vjp.value, vjp.gradients["weight"], vjp.gradients["x"]];
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![3, 2], vec![0.5, -1.0, 0.25, 0.75, -0.5, 1.5])?,
        ),
        (
            "weight".to_string(),
            DynamicTensor::new(vec![2, 1], vec![0.3, -0.7])?,
        ),
        (
            "cotangent".to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        ),
    ]);
    Ok((QuablaMultiOutputProgram::new(vjp.graph, outputs)?, inputs))
}

#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn assert_multi_output_parity(target: QuablaTarget) {
    let (program, inputs) = must!(multi_output_value_and_grad_program());
    let compiler = QuablaCompiler;
    let cpu = must!(compiler.compile_many(&program, QuablaTarget::Cpu));
    let device = must!(compiler.compile_many(&program, target));
    assert_eq!(device.target(), target);
    assert_eq!(device.output_node_ids(), cpu.output_node_ids());
    let expected = must!(cpu.execute(&inputs));
    let actual = must!(device.execute(&inputs));
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
    }
}

/// f32 MLP value and gradients for device parity against the CPU f32
/// reference; `dtype` selects the logical dtype of every input.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn mlp_value_and_grad_program(
    dtype: TensorDType,
) -> Result<(QuablaMultiOutputProgram, BTreeMap<String, DynamicTensor>), String> {
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![8, 3], dtype)?;
    let w1 = graph.input_typed("w1", vec![3, 16], dtype)?;
    let b1 = graph.input_typed("b1", vec![1, 16], dtype)?;
    let w2 = graph.input_typed("w2", vec![16, 1], dtype)?;
    let b2 = graph.input_typed("b2", vec![1, 1], dtype)?;
    let hidden = graph.matmul(x, w1)?;
    let hidden = graph.add(hidden, b1)?;
    let hidden = graph.tanh(hidden)?;
    let output = graph.matmul(hidden, w2)?;
    let output = graph.add(output, b2)?;
    let scale = graph.scalar_constant(0.5);
    let scaled = graph.mul(output, scale)?;
    let squared = graph.powi(scaled, 2)?;
    let loss = graph.mean(squared)?;
    let vjp = graph.symbolic_vjp(loss, "cotangent")?;
    let outputs = vec![
        vjp.value,
        vjp.gradients["w1"],
        vjp.gradients["b1"],
        vjp.gradients["w2"],
        vjp.gradients["b2"],
    ];
    // Deterministic non-integer inputs so that f32 rounding actually happens at every node.
    let values = |count: usize, seed: f64| -> Vec<f64> {
        (0..count)
            .map(|index| ((index as f64 + 1.0) * seed).sin() * 0.9)
            .collect()
    };
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![8, 3], values(24, 0.37))?,
        ),
        (
            "w1".to_string(),
            DynamicTensor::new(vec![3, 16], values(48, 1.13))?,
        ),
        (
            "b1".to_string(),
            DynamicTensor::new(vec![1, 16], values(16, 0.71))?,
        ),
        (
            "w2".to_string(),
            DynamicTensor::new(vec![16, 1], values(16, 2.03))?,
        ),
        ("b2".to_string(), DynamicTensor::new(vec![1, 1], vec![0.1])?),
        (
            "cotangent".to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        ),
    ]);
    Ok((QuablaMultiOutputProgram::new(vjp.graph, outputs)?, inputs))
}

/// Largest `|actual - expected| / max(1, |expected|)` over all outputs.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn max_scaled_error(actual: &[DynamicTensor], expected: &[DynamicTensor]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .flat_map(|(actual, expected)| {
            assert_eq!(actual.shape(), expected.shape());
            actual
                .data()
                .iter()
                .zip(expected.data())
                .map(|(actual, expected)| (actual - expected).abs() / expected.abs().max(1.0))
                .collect::<Vec<_>>()
        })
        .fold(0.0, f64::max)
}

/// f32 programs run natively on the device, so the CPU f32 reference sees the
/// same rounded inputs and only per-op rounding, transcendental accuracy and
/// reduction order differ: a few f32 ulps (6e-8 each at unit scale) per op over
/// this ~10-op-deep graph stay below 1e-6, ten times tighter than the existing
/// f64-reference 1e-5 contract.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn assert_f32_mlp_parity(target: QuablaTarget) {
    let compiler = QuablaCompiler;
    let (program, inputs) = must!(mlp_value_and_grad_program(TensorDType::F32));
    let cpu = must!(must!(compiler.compile_many(&program, QuablaTarget::Cpu)).execute(&inputs));
    let device = must!(must!(compiler.compile_many(&program, target)).execute(&inputs));
    assert!(cpu
        .iter()
        .chain(&device)
        .all(|value| value.dtype() == TensorDType::F32));
    let (reference_program, _) = must!(mlp_value_and_grad_program(TensorDType::F64));
    let reference =
        must!(must!(compiler.compile_many(&reference_program, QuablaTarget::Cpu)).execute(&inputs));
    let f32_error = max_scaled_error(&device, &cpu);
    let f64_error = max_scaled_error(&device, &reference);
    println!("{target:?} f32 MLP: error vs CPU f32 {f32_error:e}, vs CPU f64 {f64_error:e}");
    assert!(f32_error <= 1e-6, "device vs CPU f32 error {f32_error:e}");
    assert!(f64_error <= 1e-5, "device vs CPU f64 error {f64_error:e}");

    // cast is an f32 identity on device; the f64->f32->f64 round trip matches the CPU bitwise and
    // keeps the dtype tag.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let single = must!(graph.cast(x, TensorDType::F32));
    let doubled = must!(graph.add(single, single));
    let round_trip = must!(graph.cast(doubled, TensorDType::F64));
    let program = must!(QuablaMultiOutputProgram::new(
        graph,
        vec![doubled, round_trip]
    ));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![3], vec![0.1, -2.5e-3, 7.0 / 3.0])),
    )]);
    let cpu = must!(must!(compiler.compile_many(&program, QuablaTarget::Cpu)).execute(&inputs));
    let device = must!(must!(compiler.compile_many(&program, target)).execute(&inputs));
    for (actual, expected) in device.iter().zip(&cpu) {
        assert_eq!(actual.dtype(), expected.dtype());
        assert_eq!(actual.data(), expected.data());
    }
    assert_eq!(device[0].dtype(), TensorDType::F32);
    assert_eq!(device[1].dtype(), TensorDType::F64);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_f32_mlp_value_and_gradients_match_the_cpu_f32_reference() {
    assert_f32_mlp_parity(QuablaTarget::Mlx);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_f32_mlp_value_and_gradients_match_the_cpu_f32_reference_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    assert_f32_mlp_parity(QuablaTarget::Cuda { device_ordinal: 0 });
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn compiler_facade_multi_output_mlx_matches_cpu() {
    assert_multi_output_parity(QuablaTarget::Mlx);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn compiler_facade_multi_output_cuda_matches_cpu_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    assert_multi_output_parity(QuablaTarget::Cuda { device_ordinal: 0 });
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_some() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
fn cuda_backend_executes_only_the_selected_cond_region_with_ad_parity_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    must!(assert_log_guard_cond_matches_cpu("CUDA", |plan, inputs| {
        CudaBackend::new(0).compile(plan.clone())?.execute(inputs)
    }));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_reuses_cond_regions_and_rejects_non_finite_predicates_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    let (graph, loss) = must!(log_guard_cond_graph());
    let compiled = must!(CudaBackend::new(0).compile(must!(graph.compile_cpu(loss))));
    // Alternate both branches on one compiled plan to check that region buffers stay correct and do
    // not accumulate after reuse.
    let mut buffer_counts = Vec::new();
    for (x, expected) in [
        (2.0, 6.0 * 2.0_f64.ln()),
        (-1.0, 14.0),
        (3.0, 6.0 * 3.0_f64.ln()),
        (-2.0, 14.0),
    ] {
        let inputs = BTreeMap::from([
            ("x".to_string(), must!(DynamicTensor::new(vec![], vec![x]))),
            (
                "y".to_string(),
                must!(DynamicTensor::new(vec![3], vec![1.0, 2.0, 3.0])),
            ),
        ]);
        let value = must!(compiled.execute(&inputs));
        assert!(
            (value.data()[0] - expected).abs() < 1e-4,
            "CUDA Cond at x={x}: {} versus {expected}",
            value.data()[0]
        );
        buffer_counts.push(must!(compiled.device_buffer_count()));
    }
    assert_eq!(buffer_counts[1], buffer_counts[3], "{buffer_counts:?}");

    let mut on_true = TensorIr::new();
    let true_x = must!(on_true.input("x", vec![]));
    let mut on_false = TensorIr::new();
    let false_x = must!(on_false.input("x", vec![]));
    let two = on_false.scalar_constant(2.0);
    let false_output = must!(on_false.mul(false_x, two));
    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_x)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    must!(graph.input("x", vec![]));
    let output = must!(graph.cond(predicate, branches));
    let compiled = must!(CudaBackend::new(0).compile(must!(graph.compile_cpu(output))));
    for (predicate, expected) in [(1.0, 3.0), (0.0, 6.0)] {
        let inputs = BTreeMap::from([
            (
                "predicate".to_string(),
                must!(DynamicTensor::new(vec![], vec![predicate])),
            ),
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![], vec![3.0])),
            ),
        ]);
        assert_eq!(must!(compiled.execute(&inputs)).data(), &[expected]);
    }
    let inputs = BTreeMap::from([
        (
            "predicate".to_string(),
            must!(DynamicTensor::new(vec![], vec![f64::NAN])),
        ),
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![], vec![3.0])),
        ),
    ]);
    let error = compiled
        .execute(&inputs)
        .expect_err("CUDA must reject a non-finite Cond predicate like CPU");
    assert!(error.contains("conditional predicate must be finite"));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_rejects_cond_inside_fused_fori_and_scan_bodies() {
    let mut on_true = TensorIr::new();
    let true_carry = must!(on_true.input("carry", vec![]));
    let mut on_false = TensorIr::new();
    let false_carry = must!(on_false.input("carry", vec![]));
    let two = on_false.scalar_constant(2.0);
    let false_output = must!(on_false.mul(false_carry, two));
    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_carry)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    must!(body.input("index", vec![]));
    let body_output = must!(body.cond(carry, branches));

    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(body_output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let output = must!(graph.fori(initial, loop_plan, Vec::new()));
    let error = CudaBackend::new(0)
        .compile(must!(graph.compile_cpu(output)))
        .expect_err("CUDA must reject Cond inside a fused Fori body");
    assert!(
        error.contains("Fori node") && error.contains("cannot lower to a device loop"),
        "unexpected error: {error}"
    );
    assert!(error.contains("cond"), "unexpected error: {error}");

    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu_many(&[body_output, body_output])).0,
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let (carry, _) = must!(graph.scan(initial, scan_plan, Vec::new()));
    let error = CudaBackend::new(0)
        .compile(must!(graph.compile_cpu(carry)))
        .expect_err("CUDA must reject Cond inside a fused Scan body");
    assert!(
        error.contains("Scan node") && error.contains("cannot lower to a device loop"),
        "unexpected error: {error}"
    );
    assert!(error.contains("cond"), "unexpected error: {error}");
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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

/// Builds `sum(final_carry) + sum(outputs * outputs)` for a Scan whose per-step
/// output either broadcasts the carry `[2, 1]` to `[2, 3]` (unequal lanes) or
/// scales the carry `[2, 3]` elementwise (equal lanes).
///
/// Squaring the outputs makes the output cotangent differ per step and lane, so
/// a grouped kernel that reads it with the wrong stride or skips the
/// per-carry-lane reduction cannot match the CPU reference by accident.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_group_scan_loss(
    broadcast_output: bool,
    nonlinear: bool,
) -> Result<(TensorIr, TensorNodeId, BTreeMap<String, DynamicTensor>), String> {
    let (carry_shape, bias_shape) = if broadcast_output {
        (vec![2, 1], vec![1, 1])
    } else {
        (vec![2, 3], vec![1, 3])
    };
    let mut body = TensorIr::new();
    let carry = body.input("carry", carry_shape.clone())?;
    let index = body.input("index", vec![])?;
    let scale = body.input("scale", carry_shape.clone())?;
    let bias = body.input("bias", bias_shape.clone())?;
    let scaled = body.mul(carry, scale)?;
    let biased = body.add(scaled, bias)?;
    let activated = if nonlinear {
        body.tanh(biased)?
    } else {
        biased
    };
    let next = body.add(activated, index)?;
    let output = if broadcast_output {
        body.broadcast_to(next, vec![2, 3])?
    } else {
        body.mul(next, scale)?
    };
    let (body_plan, _) = body.compile_cpu_many(&[next, output])?;
    let scan_plan = TensorScanExecutionPlan::new(0, 3, body_plan, "carry", "index")?;
    let mut graph = TensorIr::new();
    let initial = graph.input("initial", carry_shape.clone())?;
    let scale = graph.input("scale", carry_shape.clone())?;
    let bias = graph.input("bias", bias_shape.clone())?;
    let (final_carry, outputs) = graph.scan(
        initial,
        scan_plan,
        vec![("scale".to_string(), scale), ("bias".to_string(), bias)],
    )?;
    let carry_sum = graph.sum(final_carry)?;
    let squared = graph.mul(outputs, outputs)?;
    let output_sum = graph.sum(squared)?;
    let loss = graph.add(carry_sum, output_sum)?;
    let lanes = |shape: &[usize], values: &[f64]| {
        DynamicTensor::new(
            shape.to_vec(),
            values[..shape.iter().product::<usize>()].to_vec(),
        )
    };
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            lanes(&carry_shape, &[0.4, -0.6, 0.3, -0.2, 0.5, 0.1])?,
        ),
        (
            "scale".to_string(),
            lanes(&carry_shape, &[0.8, 1.1, 0.6, -0.7, 0.9, 1.2])?,
        ),
        ("bias".to_string(), lanes(&bias_shape, &[0.1, -0.2, 0.05])?),
        (
            "cotangent".to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        ),
    ]);
    Ok((graph, loss, inputs))
}

/// Asserts elementwise `|cuda - cpu| <= tolerance * max(1, |cpu|)` for every
/// output of a multi-output plan.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn assert_cuda_outputs_match_cpu(
    label: &str,
    cuda: &[DynamicTensor],
    cpu: &[DynamicTensor],
    tolerance: f64,
) {
    assert_eq!(cuda.len(), cpu.len(), "{label}: output count differs");
    for (output, (actual, expected)) in cuda.iter().zip(cpu).enumerate() {
        assert_eq!(actual.shape(), expected.shape(), "{label}: output {output}");
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "{label}: output {output} CUDA {actual} vs CPU {expected}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_grouped_scan_vjp_matches_cpu_for_broadcast_and_equal_lane_outputs_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    for (broadcast_output, nonlinear) in
        [(true, false), (true, true), (false, false), (false, true)]
    {
        let label = format!("broadcast_output={broadcast_output} nonlinear={nonlinear}");
        let (graph, loss, inputs) = must!(cuda_group_scan_loss(broadcast_output, nonlinear));
        let transformed = must!(graph.symbolic_vjp(loss, "cotangent"));
        let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
            transformed.gradients["initial"],
            transformed.gradients["scale"],
            transformed.gradients["bias"],
        ]));
        // All three targets share one Scan VJP group, so CUDA emits the grouped kernel.
        assert_eq!(
            transformed
                .graph
                .lower_text()
                .matches("scan_vjp(group=")
                .count(),
            3,
            "{label}"
        );
        let cpu = must!(plan.evaluate_many(&inputs));
        let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
        assert_cuda_outputs_match_cpu(&label, &cuda, &cpu, 2e-5);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_grouped_scan_vjp_jvp_matches_cpu_for_broadcast_and_equal_lane_outputs_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    for (broadcast_output, nonlinear) in
        [(true, false), (true, true), (false, false), (false, true)]
    {
        let label = format!("broadcast_output={broadcast_output} nonlinear={nonlinear}");
        let (graph, loss, mut inputs) = must!(cuda_group_scan_loss(broadcast_output, nonlinear));
        let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
        // One scalar that reads every target keeps all three Scan VJP JVP members in one group.
        let mut directional_graph = vjp.graph;
        let initial_sum = must!(directional_graph.sum(vjp.gradients["initial"]));
        let scale_sum = must!(directional_graph.sum(vjp.gradients["scale"]));
        let bias_sum = must!(directional_graph.sum(vjp.gradients["bias"]));
        let partial = must!(directional_graph.add(initial_sum, scale_sum));
        let combined_sum = must!(directional_graph.add(partial, bias_sum));
        let directional = must!(directional_graph.symbolic_jvp_with_tangent_inputs(
            combined_sum,
            &BTreeMap::from([
                ("initial".to_string(), "initial_tangent".to_string()),
                ("scale".to_string(), "scale_tangent".to_string()),
            ]),
        ));
        assert_eq!(
            directional
                .graph
                .lower_text()
                .matches("scan_vjp_jvp(group=")
                .count(),
            3,
            "{label}"
        );
        let tangent = |name: &str, values: &[f64]| {
            let shape = inputs[name].shape().to_vec();
            DynamicTensor::new(
                shape.clone(),
                values[..shape.iter().product::<usize>()].to_vec(),
            )
        };
        let initial_tangent = must!(tangent("initial", &[0.3, -0.5, 0.2, 0.7, -0.1, 0.4]));
        let scale_tangent = must!(tangent("scale", &[-0.2, 0.6, 0.1, 0.3, -0.4, 0.5]));
        inputs.insert("initial_tangent".to_string(), initial_tangent);
        inputs.insert("scale_tangent".to_string(), scale_tangent);
        let plan = must!(directional.graph.compile_cpu(directional.tangent));
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_cuda_outputs_match_cpu(&label, &[cuda], &[cpu], 2e-5);
    }
}

/// Builds `sum(fori(initial) * fori(initial))` for a Fori loop with carry
/// `[2, 3]` whose captures broadcast into the carry from `[1, 3]` (`scale`)
/// and `[2, 1]` (`bias`), so the reverse kernel reduces both capture gradients
/// into multi-element outputs with atomics.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_group_fori_loss(
) -> Result<(TensorIr, TensorNodeId, BTreeMap<String, DynamicTensor>), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![2, 3])?;
    let index = body.input("index", vec![])?;
    let scale = body.input("scale", vec![1, 3])?;
    let bias = body.input("bias", vec![2, 1])?;
    let scaled = body.mul(carry, scale)?;
    let biased = body.add(scaled, bias)?;
    let shifted = body.add(biased, index)?;
    let next = body.tanh(shifted)?;
    let loop_plan = TensorForiExecutionPlan::new(0, 3, body.compile_cpu(next)?, "carry", "index")?;
    let mut graph = TensorIr::new();
    let initial = graph.input("initial", vec![2, 3])?;
    let scale = graph.input("scale", vec![1, 3])?;
    let bias = graph.input("bias", vec![2, 1])?;
    let output = graph.fori(
        initial,
        loop_plan,
        vec![("scale".to_string(), scale), ("bias".to_string(), bias)],
    )?;
    let squared = graph.mul(output, output)?;
    let loss = graph.sum(squared)?;
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            DynamicTensor::new(vec![2, 3], vec![0.2, -0.1, 0.3, -0.4, 0.5, 0.1])?,
        ),
        (
            "scale".to_string(),
            DynamicTensor::new(vec![1, 3], vec![0.8, 1.1, 0.6])?,
        ),
        (
            "bias".to_string(),
            DynamicTensor::new(vec![2, 1], vec![0.1, -0.2])?,
        ),
        (
            "cotangent".to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        ),
    ]);
    Ok((graph, loss, inputs))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_grouped_fori_vjp_matches_cpu_for_broadcast_captures_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let (graph, loss, inputs) = must!(cuda_group_fori_loss());
    let transformed = must!(graph.symbolic_vjp(loss, "cotangent"));
    assert_eq!(
        transformed
            .graph
            .lower_text()
            .matches("fori_vjp(group=")
            .count(),
        3
    );
    let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&[
        transformed.gradients["initial"],
        transformed.gradients["scale"],
        transformed.gradients["bias"],
    ]));
    let cpu = must!(plan.evaluate_many(&inputs));
    let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
    assert_cuda_outputs_match_cpu("grouped Fori VJP", &cuda, &cpu, 2e-5);
}

/// Returns a second input set that changes every lane except the scalar
/// `cotangent` seed, so a run that reads state left by the previous run cannot
/// match the CPU reference by accident.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_alternate_inputs(
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    inputs
        .iter()
        .map(|(name, tensor)| {
            let data = if name == "cotangent" {
                tensor.data().to_vec()
            } else {
                tensor
                    .data()
                    .iter()
                    .map(|value| 0.75 * value - 0.1)
                    .collect()
            };
            Ok((
                name.clone(),
                DynamicTensor::new(tensor.shape().to_vec(), data)?,
            ))
        })
        .collect()
}

/// Compiles `plan` once and executes it `runs` times, cycling through
/// `input_sets`. Every run must match the CPU reference, and every repeat of an
/// input set must reproduce that set's first CUDA result up to the summation
/// order of device atomics.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn assert_repeated_cuda_executions_match_cpu(
    label: &str,
    plan: &TensorExecutionPlan,
    input_sets: &[BTreeMap<String, DynamicTensor>],
    runs: usize,
) -> Result<(), String> {
    let compiled = CudaBackend::new(0).compile(plan.clone())?;
    let mut first_results = vec![None::<Vec<DynamicTensor>>; input_sets.len()];
    for run in 0..runs {
        let set = run % input_sets.len();
        let inputs = &input_sets[set];
        let run_label = format!("{label}: run {run} (input set {set})");
        let cuda = compiled
            .execute_many(inputs)
            .map_err(|error| format!("{run_label}: {error}"))?;
        let cpu = plan.evaluate_many(inputs)?;
        assert_cuda_outputs_match_cpu(&run_label, &cuda, &cpu, 2e-5);
        match &first_results[set] {
            Some(first) => assert_cuda_outputs_match_cpu(
                &format!("{run_label} against its first run"),
                &cuda,
                first,
                1e-6,
            ),
            None => first_results[set] = Some(cuda),
        }
    }
    Ok(())
}

/// Adds the full sums of `gradients` into one scalar so a single-output plan
/// consumes every member of a structural loop gradient group.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_sum_of_gradients(
    mut graph: TensorIr,
    gradients: &[TensorNodeId],
) -> Result<(TensorIr, TensorNodeId), String> {
    let mut total = None;
    for gradient in gradients {
        let sum = graph.sum(*gradient)?;
        total = Some(match total {
            Some(previous) => graph.add(previous, sum)?,
            None => sum,
        });
    }
    let total = total.ok_or_else(|| "no gradients to sum".to_string())?;
    Ok((graph, total))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_grouped_loop_capture_reductions_repeat_exactly_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    // Each plan returns one scalar that consumes every grouped gradient, so the gradient buffers
    // return to the pool after each run and the next run receives them, still holding the previous
    // gradients, for its atomically reduced capture outputs (Fori bias [2, 1] and scale [1, 3],
    // Scan bias [1, 3], each into a [2, 3] carry).
    let (graph, loss, inputs) = must!(cuda_group_fori_loss());
    let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
    let (graph, total) = must!(cuda_sum_of_gradients(
        vjp.graph,
        &[
            vjp.gradients["initial"],
            vjp.gradients["scale"],
            vjp.gradients["bias"],
        ],
    ));
    let plan = must!(graph.compile_cpu(total));
    let alternate = must!(cuda_alternate_inputs(&inputs));
    must!(assert_repeated_cuda_executions_match_cpu(
        "summed grouped Fori VJP",
        &plan,
        &[inputs, alternate],
        4,
    ));

    let (graph, loss, mut inputs) = must!(cuda_group_scan_loss(false, true));
    let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
    let gradients = [
        vjp.gradients["initial"],
        vjp.gradients["scale"],
        vjp.gradients["bias"],
    ];
    let (summed_graph, total) = must!(cuda_sum_of_gradients(vjp.graph.clone(), &gradients));
    let plan = must!(summed_graph.compile_cpu(total));
    let alternate = must!(cuda_alternate_inputs(&inputs));
    must!(assert_repeated_cuda_executions_match_cpu(
        "summed grouped Scan VJP",
        &plan,
        &[inputs.clone(), alternate],
        4,
    ));

    let (directional_graph, total) = must!(cuda_sum_of_gradients(vjp.graph, &gradients));
    let directional = must!(directional_graph.symbolic_jvp_with_tangent_inputs(
        total,
        &BTreeMap::from([
            ("initial".to_string(), "initial_tangent".to_string()),
            ("scale".to_string(), "scale_tangent".to_string()),
        ]),
    ));
    inputs.insert(
        "initial_tangent".to_string(),
        must!(DynamicTensor::new(
            vec![2, 3],
            vec![0.3, -0.5, 0.2, 0.7, -0.1, 0.4]
        )),
    );
    inputs.insert(
        "scale_tangent".to_string(),
        must!(DynamicTensor::new(
            vec![2, 3],
            vec![-0.2, 0.6, 0.1, 0.3, -0.4, 0.5]
        )),
    );
    let plan = must!(directional.graph.compile_cpu(directional.tangent));
    let alternate = must!(cuda_alternate_inputs(&inputs));
    must!(assert_repeated_cuda_executions_match_cpu(
        "summed grouped Scan VJP JVP",
        &plan,
        &[inputs, alternate],
        4,
    ));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_compiled_grouped_scan_vjp_value_and_grad_plan_executes_repeatedly_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    // Retaining the value and every gradient as plan outputs keeps all grouped Scan VJP results
    // alive after a run, as a compiled value-and-grad function does between training steps.
    for broadcast_output in [true, false] {
        let label = format!("grouped Scan VJP value and grad, broadcast_output={broadcast_output}");
        let (graph, loss, inputs) = must!(cuda_group_scan_loss(broadcast_output, true));
        let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
        let (plan, _) = must!(vjp.graph.compile_cpu_many(&[
            vjp.value,
            vjp.gradients["initial"],
            vjp.gradients["scale"],
            vjp.gradients["bias"],
        ]));
        let alternate = must!(cuda_alternate_inputs(&inputs));
        must!(assert_repeated_cuda_executions_match_cpu(
            &label,
            &plan,
            &[inputs, alternate],
            4,
        ));
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_compiled_grouped_scan_vjp_jvp_plan_executes_repeatedly_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    for broadcast_output in [true, false] {
        let label = format!("grouped Scan VJP JVP, broadcast_output={broadcast_output}");
        let (graph, loss, mut inputs) = must!(cuda_group_scan_loss(broadcast_output, true));
        let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
        // Directional derivatives of every gradient as separate outputs keep each Scan VJP JVP
        // group member alive after the run.
        let directional = must!(vjp.graph.symbolic_jvp_many_with_tangent_inputs(
            &[
                vjp.gradients["initial"],
                vjp.gradients["scale"],
                vjp.gradients["bias"],
            ],
            &BTreeMap::from([
                ("initial".to_string(), "initial_tangent".to_string()),
                ("scale".to_string(), "scale_tangent".to_string()),
            ]),
        ));
        assert_eq!(
            directional
                .graph
                .lower_text()
                .matches("scan_vjp_jvp(group=")
                .count(),
            3,
            "{label}"
        );
        let shape = inputs["initial"].shape().to_vec();
        let lanes = shape.iter().product::<usize>();
        let initial_tangent = [0.3, -0.5, 0.2, 0.7, -0.1, 0.4][..lanes].to_vec();
        let scale_tangent = [-0.2, 0.6, 0.1, 0.3, -0.4, 0.5][..lanes].to_vec();
        inputs.insert(
            "initial_tangent".to_string(),
            must!(DynamicTensor::new(shape.clone(), initial_tangent)),
        );
        inputs.insert(
            "scale_tangent".to_string(),
            must!(DynamicTensor::new(shape, scale_tangent)),
        );
        let (plan, _) = must!(directional.graph.compile_cpu_many(&directional.tangents));
        let alternate = must!(cuda_alternate_inputs(&inputs));
        must!(assert_repeated_cuda_executions_match_cpu(
            &label,
            &plan,
            &[inputs, alternate],
            4,
        ));
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_compiled_grouped_fori_vjp_plan_executes_repeatedly_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let (graph, loss, inputs) = must!(cuda_group_fori_loss());
    let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
    let (plan, _) = must!(vjp.graph.compile_cpu_many(&[
        vjp.value,
        vjp.gradients["initial"],
        vjp.gradients["scale"],
        vjp.gradients["bias"],
    ]));
    let alternate = must!(cuda_alternate_inputs(&inputs));
    must!(assert_repeated_cuda_executions_match_cpu(
        "grouped Fori VJP value and grad",
        &plan,
        &[inputs, alternate],
        4,
    ));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_compiled_scan_plan_retaining_carry_and_outputs_executes_repeatedly_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![2, 3]));
    let index = must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![2, 3]));
    let scaled = must!(body.mul(carry, scale));
    let activated = must!(body.tanh(scaled));
    let next = must!(body.add(activated, index));
    let output = must!(body.mul(next, scale));
    let (body_plan, _) = must!(body.compile_cpu_many(&[next, output]));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0, 3, body_plan, "carry", "index"
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![2, 3]));
    let scale = must!(graph.input("scale", vec![2, 3]));
    let (final_carry, outputs) =
        must!(graph.scan(initial, scan_plan, vec![("scale".to_string(), scale)]));
    // Both results of one Scan group stay alive as plan outputs after every run.
    let (plan, _) = must!(graph.compile_cpu_many(&[final_carry, outputs]));
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3],
                vec![0.4, -0.6, 0.3, -0.2, 0.5, 0.1]
            )),
        ),
        (
            "scale".to_string(),
            must!(DynamicTensor::new(
                vec![2, 3],
                vec![0.8, 1.1, 0.6, -0.7, 0.9, 1.2]
            )),
        ),
    ]);
    let alternate = must!(cuda_alternate_inputs(&inputs));
    must!(assert_repeated_cuda_executions_match_cpu(
        "Scan carry and outputs",
        &plan,
        &[inputs, alternate],
        4,
    ));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_compiled_cond_plan_alternates_branches_across_executions_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let (graph, loss) = must!(log_guard_cond_graph());
    let vjp = must!(graph.symbolic_vjp(loss, "seed"));
    let (plan, _) =
        must!(vjp
            .graph
            .compile_cpu_many(&[vjp.value, vjp.gradients["x"], vjp.gradients["y"],]));
    let inputs_at = |x: f64| -> Result<BTreeMap<String, DynamicTensor>, String> {
        Ok(BTreeMap::from([
            ("x".to_string(), DynamicTensor::new(vec![], vec![x])?),
            (
                "y".to_string(),
                DynamicTensor::new(vec![3], vec![1.0, 2.0, 3.0])?,
            ),
            ("seed".to_string(), DynamicTensor::new(vec![], vec![1.0])?),
        ]))
    };
    // Alternating the predicate switches the executed branch region on every run.
    must!(assert_repeated_cuda_executions_match_cpu(
        "Cond value and grad",
        &plan,
        &[must!(inputs_at(2.0)), must!(inputs_at(-1.0))],
        4,
    ));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_retained_adam_trains_through_grouped_scan_gradients_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let (learning_rate, beta1, beta2, epsilon) = (0.05, 0.9, 0.999, 1e-8);
    let (graph, loss, inputs) = must!(cuda_group_scan_loss(true, true));
    let vjp = must!(graph.symbolic_vjp(loss, "cotangent"));
    let names = ["initial", "scale", "bias"];
    let mut outputs = vec![vjp.value];
    outputs.extend(names.iter().map(|name| vjp.gradients[*name]));
    let (plan, output_ids) = must!(vjp.graph.compile_cpu_many(&outputs));
    let cuda = must!(CudaBackend::new(0).compile(plan.clone()));
    let retained = names
        .iter()
        .map(|name| name.to_string())
        .collect::<BTreeSet<_>>();

    // Host Adam in f64 over CPU gradients is the reference trajectory.
    let mut host_inputs = inputs.clone();
    let mut moments = names
        .iter()
        .map(|name| {
            let count = inputs[*name].data().len();
            (vec![0.0f64; count], vec![0.0f64; count])
        })
        .collect::<Vec<_>>();
    for step in 1..=4 {
        let cpu = must!(plan.evaluate_many(&host_inputs));
        must!(cuda.execute_retaining_without_output(&inputs, &retained));
        let cuda_loss = must!(cuda.computed_node_to_host(output_ids[0]));
        assert_cuda_outputs_match_cpu(
            &format!("Adam step {step} loss"),
            &[cuda_loss],
            &cpu[..1],
            1e-4,
        );
        for (index, name) in names.iter().enumerate() {
            must!(cuda.adam_step_input_from_node(
                name,
                output_ids[index + 1],
                learning_rate,
                beta1,
                beta2,
                epsilon,
            ));
            let (first, second) = &mut moments[index];
            let parameter = &host_inputs[*name];
            let correction1 = 1.0 - f64::from(beta1).powi(step);
            let correction2 = 1.0 - f64::from(beta2).powi(step);
            let updated = parameter
                .data()
                .iter()
                .zip(cpu[index + 1].data())
                .enumerate()
                .map(|(lane, (value, gradient))| {
                    first[lane] =
                        f64::from(beta1) * first[lane] + (1.0 - f64::from(beta1)) * gradient;
                    second[lane] = f64::from(beta2) * second[lane]
                        + (1.0 - f64::from(beta2)) * gradient * gradient;
                    value
                        - f64::from(learning_rate) * (first[lane] / correction1)
                            / ((second[lane] / correction2).sqrt() + f64::from(epsilon))
                })
                .collect::<Vec<_>>();
            let updated = must!(DynamicTensor::new(parameter.shape().to_vec(), updated));
            host_inputs.insert(name.to_string(), updated);
        }
    }
    for name in names {
        let device = must!(cuda.retained_input_to_host(name));
        assert_cuda_outputs_match_cpu(
            &format!("Adam parameter {name}"),
            &[device],
            &[host_inputs[name].clone()],
            1e-4,
        );
    }
}

/// Builds `sum(cond(x > 0, y * log(x), y * y + x * 0))`.
///
/// The inactive log branch would inject NaN into the value and gradients for
/// `x <= 0` if a backend evaluated both regions and selected afterwards.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn log_guard_cond_graph() -> Result<(TensorIr, quabla_core::tensor_ir::TensorNodeId), String> {
    let mut on_true = TensorIr::new();
    let true_a = on_true.input("a", vec![])?;
    let true_b = on_true.input("b", vec![3])?;
    let log_a = on_true.log(true_a)?;
    let true_output = on_true.mul(true_b, log_a)?;

    let mut on_false = TensorIr::new();
    let false_a = on_false.input("a", vec![])?;
    let false_b = on_false.input("b", vec![3])?;
    let zero = on_false.scalar_constant(0.0);
    let masked = on_false.mul(false_a, zero)?;
    let square = on_false.mul(false_b, false_b)?;
    let false_output = on_false.add(square, masked)?;

    let branches = TensorCondExecutionPlan::new(
        on_true.compile_cpu(true_output)?,
        on_false.compile_cpu(false_output)?,
    )?;
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![])?;
    let y = graph.input("y", vec![3])?;
    let zero = graph.scalar_constant(0.0);
    let predicate = graph.greater(x, zero)?;
    let selected = graph.cond_with_captures(
        predicate,
        branches,
        vec![("a".to_string(), x), ("b".to_string(), y)],
    )?;
    let loss = graph.sum(selected)?;
    Ok((graph, loss))
}

/// Compares a device executor with CPU for the log-guard `Cond` primal, its
/// symbolic VJP/JVP, and the second derivative through a VJP-produced `Cond`,
/// for both predicate outcomes. Every device value must stay finite.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn assert_log_guard_cond_matches_cpu(
    backend: &str,
    execute: impl Fn(
        &quabla_core::tensor_ir::TensorExecutionPlan,
        &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String>,
) -> Result<(), String> {
    let (graph, loss) = log_guard_cond_graph()?;
    let vjp = graph.symbolic_vjp(loss, "seed")?;
    let jvp = graph.symbolic_jvp(loss, "x")?;
    let second = vjp.graph.symbolic_jvp(vjp.gradients["x"], "x")?;
    let cases = [
        ("value", &graph, loss),
        ("vjp_x", &vjp.graph, vjp.gradients["x"]),
        ("vjp_y", &vjp.graph, vjp.gradients["y"]),
        ("jvp_x", &jvp.graph, jvp.tangent),
        ("second_x", &second.graph, second.tangent),
    ];
    for x in [2.0, -1.0] {
        let inputs = BTreeMap::from([
            ("x".to_string(), DynamicTensor::new(vec![], vec![x])?),
            (
                "y".to_string(),
                DynamicTensor::new(vec![3], vec![1.0, 2.0, 3.0])?,
            ),
            ("seed".to_string(), DynamicTensor::new(vec![], vec![1.0])?),
        ]);
        for (label, case_graph, node) in cases {
            let cpu = case_graph.evaluate(node, &inputs)?;
            let device = execute(&case_graph.compile_cpu(node)?, &inputs)?;
            if device.shape() != cpu.shape() {
                return Err(format!("{backend} {label} at x={x} changed shape"));
            }
            for (actual, expected) in device.data().iter().zip(cpu.data()) {
                if !actual.is_finite() || (actual - expected).abs() > 1e-4 * expected.abs().max(1.0)
                {
                    return Err(format!(
                        "{backend} {label} at x={x}: {actual} versus CPU {expected}"
                    ));
                }
            }
        }
    }
    // The unselected branch must not affect the gradient: for x <= 0 the log branch does not
    // participate at all.
    let negative = BTreeMap::from([
        ("x".to_string(), DynamicTensor::new(vec![], vec![-1.0])?),
        (
            "y".to_string(),
            DynamicTensor::new(vec![3], vec![1.0, 2.0, 3.0])?,
        ),
        ("seed".to_string(), DynamicTensor::new(vec![], vec![1.0])?),
    ]);
    let gradient_x = execute(&vjp.graph.compile_cpu(vjp.gradients["x"])?, &negative)?;
    let gradient_y = execute(&vjp.graph.compile_cpu(vjp.gradients["y"])?, &negative)?;
    if gradient_x.data() != [0.0] || gradient_y.data() != [2.0, 4.0, 6.0] {
        return Err(format!(
            "{backend} inactive-branch gradients leaked: {:?} {:?}",
            gradient_x.data(),
            gradient_y.data()
        ));
    }
    Ok(())
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_only_the_selected_cond_region_with_ad_parity() {
    must!(assert_log_guard_cond_matches_cpu("MLX", |plan, inputs| {
        MlxBackend.execute(plan, inputs)
    }));
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_returns_float_masks_from_greater_like_cpu() {
    // greater is a 0/1 float mask in the IR; MLX readback, accumulation, and Cond predicates must
    // match the CPU.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let y = must!(graph.input("y", vec![3]));
    let mask = must!(graph.greater(x, y));
    let doubled = must!(graph.add(mask, mask));
    let count = must!(graph.sum(doubled));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![3], vec![1.0, 2.0, 3.0])),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![3], vec![2.0, 2.0, 2.0])),
        ),
    ]);
    let mask_plan = must!(graph.compile_cpu(mask));
    let cpu_mask = must!(CpuBackend.execute(&mask_plan, &inputs));
    let cpu_count = must!(CpuBackend.execute(&must!(graph.compile_cpu(count)), &inputs));
    assert_eq!(cpu_mask.data(), &[0.0, 0.0, 1.0]);
    assert_eq!(cpu_count.data(), &[2.0]);
    assert_eq!(
        must!(MlxBackend.execute(&mask_plan, &inputs)).data(),
        cpu_mask.data()
    );
    let (plan, outputs) = must!(graph.compile_cpu_many(&[mask, count]));
    let mlx = must!(MlxBackend.execute_many(&plan, &outputs, &inputs));
    assert_eq!(mlx[0].data(), cpu_mask.data());
    assert_eq!(mlx[1].data(), cpu_count.data());

    let mut true_branch = TensorIr::new();
    let true_x = must!(true_branch.input("x", vec![]));
    let true_output = must!(true_branch.mul(true_x, true_x));
    let mut false_branch = TensorIr::new();
    let false_x = must!(false_branch.input("x", vec![]));
    let false_output = must!(false_branch.add(false_x, false_x));
    let branches = must!(TensorCondExecutionPlan::new(
        must!(true_branch.compile_cpu(true_output)),
        must!(false_branch.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![]));
    let zero = graph.scalar_constant(0.0);
    let predicate = must!(graph.greater(x, zero));
    let output = must!(graph.cond(predicate, branches));
    let plan = must!(graph.compile_cpu(output));
    for (value, expected) in [(3.0, 9.0), (-3.0, -6.0)] {
        let inputs = BTreeMap::from([(
            "x".to_string(),
            must!(DynamicTensor::new(vec![], vec![value])),
        )]);
        assert_eq!(
            must!(CpuBackend.execute(&plan, &inputs)).data(),
            &[expected]
        );
        assert_eq!(
            must!(MlxBackend.execute(&plan, &inputs)).data(),
            &[expected]
        );
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_nested_cond_regions_and_rejects_non_finite_predicates() {
    let mut inner_true = TensorIr::new();
    let inner_true_x = must!(inner_true.input("x", vec![]));
    let inner_true_output = must!(inner_true.mul(inner_true_x, inner_true_x));
    let mut inner_false = TensorIr::new();
    let inner_false_x = must!(inner_false.input("x", vec![]));
    let three = inner_false.scalar_constant(3.0);
    let inner_false_output = must!(inner_false.mul(inner_false_x, three));
    let inner_branches = must!(TensorCondExecutionPlan::new(
        must!(inner_true.compile_cpu(inner_true_output)),
        must!(inner_false.compile_cpu(inner_false_output)),
    ));

    let mut outer_true = TensorIr::new();
    let inner_predicate = must!(outer_true.input("inner", vec![]));
    must!(outer_true.input("x", vec![]));
    let outer_true_output = must!(outer_true.cond(inner_predicate, inner_branches));
    let mut outer_false = TensorIr::new();
    let outer_false_inner = must!(outer_false.input("inner", vec![]));
    let outer_false_x = must!(outer_false.input("x", vec![]));
    let zero = outer_false.scalar_constant(0.0);
    let ignored = must!(outer_false.mul(outer_false_inner, zero));
    let outer_false_output = must!(outer_false.add(outer_false_x, ignored));
    let outer_branches = must!(TensorCondExecutionPlan::new(
        must!(outer_true.compile_cpu(outer_true_output)),
        must!(outer_false.compile_cpu(outer_false_output)),
    ));

    let mut graph = TensorIr::new();
    let outer = must!(graph.input("outer", vec![]));
    must!(graph.input("inner", vec![]));
    must!(graph.input("x", vec![]));
    let output = must!(graph.cond(outer, outer_branches));
    let plan = must!(graph.compile_cpu(output));
    for (outer, inner, expected) in [(1.0, 1.0, 4.0), (1.0, 0.0, 6.0), (0.0, 1.0, 2.0)] {
        let inputs = BTreeMap::from([
            (
                "outer".to_string(),
                must!(DynamicTensor::new(vec![], vec![outer])),
            ),
            (
                "inner".to_string(),
                must!(DynamicTensor::new(vec![], vec![inner])),
            ),
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![], vec![2.0])),
            ),
        ]);
        assert_eq!(
            must!(MlxBackend.execute(&plan, &inputs)).data(),
            &[expected]
        );
    }

    let inputs = BTreeMap::from([
        (
            "outer".to_string(),
            must!(DynamicTensor::new(vec![], vec![f64::NAN])),
        ),
        (
            "inner".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        ),
    ]);
    let error = MlxBackend
        .execute(&plan, &inputs)
        .expect_err("MLX must reject a non-finite Cond predicate like CPU");
    assert!(error.contains("conditional predicate must be finite"));
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
    assert!(source.contains("quabla_fused_elementwise"));
    assert!(source.contains("input_0[quabla_offset_0(index)]"));
    assert!(source.contains("input_1[quabla_offset_1(index)]"));
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
fn kernel_ir_placement_propagation_rejects_concat_along_sharded_axis_only() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let y = must!(graph.input("y", vec![4, 2]));
    let along_rows = must!(graph.concat(vec![x, y], 0));
    let along_columns = must!(graph.concat(vec![x, y], 1));
    let mean_rows = must!(graph.mean(along_rows));
    let sharded = TensorPlacement::Mesh {
        mesh: two_cuda_mesh(),
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: 0,
            mesh_axis: "data".to_string(),
        },
    };
    let placements = BTreeMap::from([(x, sharded.clone()), (y, sharded.clone())]);

    // Local shard concatenation interleaves operands instead of producing a
    // contiguous shard of the global result.
    let error = must!(graph.compile_cpu(along_rows))
        .kernel_ir_with_placement_propagation(&placements)
        .expect_err("concat along a sharded axis needs redistribution");
    assert!(
        error.contains("concatenates along a sharded axis; explicit redistribution is required"),
        "{error}"
    );
    let error = must!(graph.compile_cpu(mean_rows))
        .sharding_plan(&placements)
        .expect_err("sharding plans must reject the same redistribution");
    assert!(
        error.contains("concatenates along a sharded axis"),
        "{error}"
    );

    let program =
        must!(must!(graph.compile_cpu(along_columns))
            .kernel_ir_with_placement_propagation(&placements));
    assert_eq!(program.nodes[program.output_node_id].placement, sharded);
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

fn data_axis_sharded(mesh: &TensorDeviceMesh) -> TensorPlacement {
    TensorPlacement::Mesh {
        mesh: mesh.clone(),
        partition: TensorPartitionSpec::Sharded {
            tensor_axis: 0,
            mesh_axis: "data".to_string(),
        },
    }
}

/// `y = tanh(x * w)` with `x` sharded on axis zero and `w` replicated. The
/// retained outputs are `[sum_axis(y, 0), mean(y)]`, while the sharding
/// schedule lists the `Mean` collective first because it has the lower node id.
fn mixed_reduction_data_parallel_plan() -> Result<
    (
        TensorExecutionPlan,
        TensorShardingPlan,
        TensorNodeId,
        TensorNodeId,
    ),
    String,
> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![8, 2])?;
    let w = graph.input("w", vec![1, 2])?;
    let scaled = graph.mul(x, w)?;
    let activated = graph.tanh(scaled)?;
    let mean = graph.mean(activated)?;
    let column_sum = graph.sum_axis(activated, 0)?;
    let (plan, outputs) = graph.compile_cpu_many(&[column_sum, mean])?;
    let mesh = two_cuda_mesh();
    let sharding = plan.sharding_plan(&BTreeMap::from([
        (x, data_axis_sharded(&mesh)),
        (
            w,
            TensorPlacement::Mesh {
                mesh,
                partition: TensorPartitionSpec::Replicated,
            },
        ),
    ]))?;
    Ok((plan, sharding, outputs[0], outputs[1]))
}

fn mixed_reduction_data_parallel_inputs() -> Result<BTreeMap<String, DynamicTensor>, String> {
    Ok(BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(
                vec![8, 2],
                vec![
                    -2.0, 1.5, -1.0, 0.5, -0.5, 0.25, 0.0, -0.75, 0.5, 1.0, 1.0, -1.25, 1.5, 2.0,
                    2.0, -0.5,
                ],
            )?,
        ),
        (
            "w".to_string(),
            DynamicTensor::new(vec![1, 2], vec![0.75, -0.4])?,
        ),
    ]))
}

/// Error from lowering the sharding schedule of `graph` compiled at `outputs`,
/// with `placements` keyed by graph node id.
fn cuda_data_parallel_program_error(
    graph: &TensorIr,
    outputs: &[TensorNodeId],
    placements: BTreeMap<TensorNodeId, TensorPlacement>,
    device_ordinals: &[usize],
) -> String {
    let result = graph.compile_cpu_many(outputs).and_then(|(plan, _)| {
        let sharding = plan.sharding_plan(&placements)?;
        plan.cuda_data_parallel_program(&sharding, device_ordinals)
    });
    match result {
        Ok(_) => "unexpectedly accepted".to_string(),
        Err(error) => error,
    }
}

#[test]
fn cuda_data_parallel_program_preserves_schedule_order_and_matches_cpu_oracle() {
    let (plan, sharding, column_sum, mean) = must!(mixed_reduction_data_parallel_plan());
    let program = must!(plan.cuda_data_parallel_program(&sharding, &[0, 1]));

    assert_eq!(
        program.collectives,
        vec![
            (mean, TensorReplicaReduction::Mean),
            (column_sum, TensorReplicaReduction::Sum),
        ]
    );
    assert_eq!(program.replica_plan.output_node_ids(), &[column_sum, mean]);
    assert_eq!(program.replica_plan.input_shape("x"), Ok(vec![4, 2]));
    assert_eq!(program.replica_plan.input_shape("w"), Ok(vec![1, 2]));

    // Simulate the NCCL schedule on CPU: run each replica on its axis-zero
    // shard, then apply every scheduled collective to the retained outputs.
    let inputs = must!(mixed_reduction_data_parallel_inputs());
    let expected = must!(plan.evaluate_many(&inputs));
    let mut replica_outputs = Vec::new();
    for replica in 0..2 {
        let mut shard_inputs = inputs.clone();
        let shard = must!(inputs["x"].slice_axis(0, replica * 4, 4));
        shard_inputs.insert("x".to_string(), shard);
        replica_outputs.push(must!(program.replica_plan.evaluate_many(&shard_inputs)));
    }
    for (node_id, reduction) in &program.collectives {
        let position = must!(plan
            .output_node_ids()
            .iter()
            .position(|output| output == node_id)
            .ok_or("scheduled node is not a retained output"));
        let scale = match reduction {
            TensorReplicaReduction::Sum => 1.0,
            TensorReplicaReduction::Mean => 0.5,
        };
        let reduced = (0..expected[position].data().len())
            .map(|index| {
                scale
                    * replica_outputs
                        .iter()
                        .map(|outputs| outputs[position].data()[index])
                        .sum::<f64>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            replica_outputs[0][position].shape(),
            expected[position].shape()
        );
        for (actual, reference) in reduced.iter().zip(expected[position].data()) {
            assert!(
                (actual - reference).abs() <= 1e-12,
                "node {node_id}: {actual} != {reference}"
            );
        }
    }
}

#[test]
fn cuda_data_parallel_program_rejects_mesh_outside_first_subset() {
    let (plan, sharding, _, _) = must!(mixed_reduction_data_parallel_plan());
    for (ordinals, reason) in [
        (vec![1, 0], "rank order"),
        (vec![0, 1, 2], "mesh size differs from device count"),
    ] {
        let error = plan
            .cuda_data_parallel_program(&sharding, &ordinals)
            .expect_err(reason);
        assert!(error.contains("do not match"), "{reason}: {error}");
    }

    let cuda = |ordinal| TensorDeviceId {
        backend: TensorDeviceBackend::Cuda,
        ordinal,
    };
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let output = must!(graph.mean_axis(x, 0));
    let two_dimensional = TensorDeviceMesh {
        devices: vec![cuda(0), cuda(1)],
        axis_names: vec!["data".to_string(), "model".to_string()],
        shape: vec![2, 1],
    };
    let error = cuda_data_parallel_program_error(
        &graph,
        &[output],
        BTreeMap::from([(x, data_axis_sharded(&two_dimensional))]),
        &[0, 1],
    );
    assert!(error.contains("only a 1-D mesh"), "{error}");

    let cpu_mesh = TensorDeviceMesh {
        devices: (0..2)
            .map(|ordinal| TensorDeviceId {
                backend: TensorDeviceBackend::Cpu,
                ordinal,
            })
            .collect(),
        axis_names: vec!["data".to_string()],
        shape: vec![2],
    };
    let error = cuda_data_parallel_program_error(
        &graph,
        &[output],
        BTreeMap::from([(x, data_axis_sharded(&cpu_mesh))]),
        &[0, 1],
    );
    assert!(error.contains("only CUDA devices"), "{error}");

    let mut transposed = TensorIr::new();
    let x = must!(transposed.input("x", vec![2, 4]));
    let output = must!(transposed.sum_axis(x, 1));
    let error = cuda_data_parallel_program_error(
        &transposed,
        &[output],
        BTreeMap::from([(
            x,
            TensorPlacement::Mesh {
                mesh: two_cuda_mesh(),
                partition: TensorPartitionSpec::Sharded {
                    tensor_axis: 1,
                    mesh_axis: "data".to_string(),
                },
            },
        )]),
        &[0, 1],
    );
    assert!(error.contains("only axis-zero batch sharding"), "{error}");
}

#[test]
fn cuda_data_parallel_program_rejects_schedules_needing_mid_graph_collectives() {
    let mesh = two_cuda_mesh();
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let reduced = must!(graph.mean_axis(x, 0));
    let consumer = must!(graph.tanh(reduced));
    let sharded_x = BTreeMap::from([(x, data_axis_sharded(&mesh))]);

    for outputs in [vec![consumer], vec![reduced, consumer]] {
        let error = cuda_data_parallel_program_error(&graph, &outputs, sharded_x.clone(), &[0, 1]);
        assert!(
            error.contains("mid-graph collective"),
            "{outputs:?}: {error}"
        );
    }

    let local = must!(graph.tanh(x));
    let error = cuda_data_parallel_program_error(&graph, &[reduced, local], sharded_x, &[0, 1]);
    assert!(error.contains("has no all-reduce"), "{error}");

    let error = cuda_data_parallel_program_error(
        &graph,
        &[consumer],
        BTreeMap::from([(
            x,
            TensorPlacement::Mesh {
                mesh,
                partition: TensorPartitionSpec::Replicated,
            },
        )]),
        &[0, 1],
    );
    assert!(error.contains("no all-reduce"), "{error}");
}

#[test]
fn cuda_data_parallel_program_rejects_redistribution_and_foreign_schedules() {
    let mesh = two_cuda_mesh();
    let replicated = TensorPlacement::Mesh {
        mesh: mesh.clone(),
        partition: TensorPartitionSpec::Replicated,
    };

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let z = must!(graph.input("z", vec![4, 2]));
    let joined = must!(graph.concat(vec![x, z], 0));
    let output = must!(graph.mean(joined));
    let error = cuda_data_parallel_program_error(
        &graph,
        &[output],
        BTreeMap::from([(x, data_axis_sharded(&mesh)), (z, data_axis_sharded(&mesh))]),
        &[0, 1],
    );
    assert!(
        error.contains("concatenates along a sharded axis"),
        "{error}"
    );

    // A replicated operand keeps its full batch extent on every replica, so it
    // either fails replica shape inference or silently broadcasts a shard.
    let full = must!(graph.add(x, z));
    let output = must!(graph.mean(full));
    let error = cuda_data_parallel_program_error(
        &graph,
        &[output],
        BTreeMap::from([(x, data_axis_sharded(&mesh)), (z, replicated.clone())]),
        &[0, 1],
    );
    assert!(error.contains("specialization failed"), "{error}");

    let mut broadcast = TensorIr::new();
    let x = must!(broadcast.input("x", vec![2, 2]));
    let z = must!(broadcast.input("z", vec![2, 2]));
    let full = must!(broadcast.add(x, z));
    let output = must!(broadcast.mean(full));
    let error = cuda_data_parallel_program_error(
        &broadcast,
        &[output],
        BTreeMap::from([(x, data_axis_sharded(&mesh)), (z, replicated)]),
        &[0, 1],
    );
    assert!(error.contains("replica-local shape [2, 2]"), "{error}");

    let mut uneven = TensorIr::new();
    let x = must!(uneven.input("x", vec![8, 1]));
    let z = must!(uneven.input("z", vec![4, 1]));
    let x_sum = must!(uneven.sum(x));
    let z_sum = must!(uneven.sum(z));
    let error = cuda_data_parallel_program_error(
        &uneven,
        &[x_sum, z_sum],
        BTreeMap::from([(x, data_axis_sharded(&mesh)), (z, data_axis_sharded(&mesh))]),
        &[0, 1],
    );
    assert!(error.contains("share one axis-zero extent"), "{error}");

    let (plan, _, _, _) = must!(mixed_reduction_data_parallel_plan());
    let mut other = TensorIr::new();
    let x = must!(other.input("x", vec![8, 2]));
    let output = must!(other.sum_axis(x, 0));
    let foreign = must!(must!(other.compile_cpu(output))
        .sharding_plan(&BTreeMap::from([(x, data_axis_sharded(&mesh))])));
    let error = plan
        .cuda_data_parallel_program(&foreign, &[0, 1])
        .expect_err("a schedule derived from another plan must be rejected");
    assert!(error.contains("sharding plan"), "{error}");
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

    // `usize::MAX` exceeds any CUDA device count, so this holds on every host.
    let error = backend
        .compile_data_parallel(plan, vec![0, usize::MAX])
        .expect_err("an ordinal beyond the CUDA device count must be rejected");
    assert!(error.contains("CUDA reports"), "{error}");

    let _: fn(&CudaDataParallelExecutionPlan) -> usize =
        CudaDataParallelExecutionPlan::replica_count;
}

/// Two-GPU NCCL parity for a mixed `Sum`/`Mean` sharding schedule. Requires
/// `QUABLA_CUDA_NCCL_TEST=1`, a loadable NCCL library, and CUDA ordinals 0 and 1.
#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[test]
fn cuda_data_parallel_sharded_schedule_matches_cpu_oracle_on_two_gpus() {
    if std::env::var_os("QUABLA_CUDA_NCCL_TEST").is_none() {
        return;
    }
    let (plan, sharding, column_sum, mean) = must!(mixed_reduction_data_parallel_plan());
    let inputs = must!(mixed_reduction_data_parallel_inputs());
    let expected = must!(plan.evaluate_many(&inputs));
    let parallel =
        must!(CudaBackend::new(0).compile_data_parallel_sharded(plan, &sharding, vec![0, 1]));
    let mut replica_inputs = Vec::new();
    for replica in 0..2 {
        let mut shard_inputs = inputs.clone();
        let shard = must!(inputs["x"].slice_axis(0, replica * 4, 4));
        shard_inputs.insert("x".to_string(), shard);
        replica_inputs.push(shard_inputs);
    }

    let result = must!(parallel.execute_sharded(&replica_inputs));

    assert_eq!(parallel.output_node_ids(), &[column_sum, mean]);
    for (position, (actual, reference)) in result.outputs.iter().zip(&expected).enumerate() {
        assert_eq!(actual.shape(), reference.shape());
        let max_error = actual
            .data()
            .iter()
            .zip(reference.data())
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "sharded-schedule output={position} cuda={:?} cpu={:?} max_abs_error={max_error:e}",
            actual.data(),
            reference.data()
        );
        assert!(max_error <= 1e-5, "output {position}: {max_error}");
    }
    println!("sharded-schedule timing={:?}", result.timing);

    for reduction in [TensorReplicaReduction::Sum, TensorReplicaReduction::Mean] {
        let error = parallel
            .execute_replicas(&replica_inputs, reduction)
            .expect_err("a mixed schedule cannot run under one caller reduction");
        assert!(error.contains("use execute_sharded"), "{error}");
    }
}

/// Splits the mixed-reduction inputs into two replica maps with a four-row
/// `x` shard each, plus a copy whose replica 1 lacks its shard. That replica
/// fails input validation after replica 0 has enqueued its work, which is an
/// execution failure after the plan's communicators exist.
#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[allow(clippy::type_complexity)]
fn two_gpu_replica_inputs(
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<
    (
        Vec<BTreeMap<String, DynamicTensor>>,
        Vec<BTreeMap<String, DynamicTensor>>,
    ),
    String,
> {
    let mut replica_inputs = Vec::new();
    for replica in 0..2 {
        let mut shard_inputs = inputs.clone();
        shard_inputs.insert("x".to_string(), inputs["x"].slice_axis(0, replica * 4, 4)?);
        replica_inputs.push(shard_inputs);
    }
    let mut failing_inputs = replica_inputs.clone();
    failing_inputs[1].remove("x");
    Ok((replica_inputs, failing_inputs))
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
fn assert_data_parallel_outputs_match(
    actual: &[DynamicTensor],
    expected: &[DynamicTensor],
    label: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (position, (actual, reference)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.shape(),
            reference.shape(),
            "{label} output {position}"
        );
        let max_error = actual
            .data()
            .iter()
            .zip(reference.data())
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error <= 1e-5, "{label} output {position}: {max_error}");
    }
}

/// Failure contract: an error inside a data-parallel call is returned, the
/// call aborts the plan's NCCL communicators, and the next call on the plan
/// or any clone recreates them. Requires `QUABLA_CUDA_NCCL_TEST=1`, a loadable
/// NCCL library, and CUDA ordinals 0 and 1.
#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[test]
fn cuda_data_parallel_recreates_communicators_after_a_failed_call_on_two_gpus() {
    if std::env::var_os("QUABLA_CUDA_NCCL_TEST").is_none() {
        return;
    }
    let (plan, sharding, _, _) = must!(mixed_reduction_data_parallel_plan());
    let inputs = must!(mixed_reduction_data_parallel_inputs());
    let expected = must!(plan.evaluate_many(&inputs));
    let (replica_inputs, failing_inputs) = must!(two_gpu_replica_inputs(&inputs));
    let parallel =
        must!(CudaBackend::new(0).compile_data_parallel_sharded(plan, &sharding, vec![0, 1]));

    let before = must!(parallel.execute_sharded(&replica_inputs));
    assert_data_parallel_outputs_match(&before.outputs, &expected, "before failure");
    // The second failure first recreates the communicators the first aborted.
    for attempt in 0..2 {
        let error = parallel
            .execute_sharded(&failing_inputs)
            .expect_err("a replica without its shard must fail");
        println!("failure-contract attempt={attempt} error={error}");
        assert!(error.contains("missing input \"x\""), "{error}");
        assert!(error.contains("next call recreates them"), "{error}");
    }
    // Clones share the communicators, so the clone observes the reset.
    let clone = parallel.clone();
    for call in 0..3 {
        let start = std::time::Instant::now();
        let result = must!(clone.execute_sharded(&replica_inputs));
        println!(
            "failure-contract recovered call={call} wall={:?} timing={:?}",
            start.elapsed(),
            result.timing
        );
        assert_data_parallel_outputs_match(&result.outputs, &expected, "after failure");
    }
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
fn cuda_free_bytes(
    contexts: &[std::sync::Arc<cudarc::driver::CudaContext>],
) -> Result<Vec<usize>, String> {
    contexts
        .iter()
        .map(|context| {
            context
                .synchronize()
                .and_then(|()| context.mem_get_info())
                .map(|(free, _)| free)
                .map_err(|error| format!("failed to query CUDA memory: {error:?}"))
        })
        .collect()
}

/// Communicator lifecycle: 50 compile, execute, drop cycles on two GPUs.
/// Even cycles drop the plan right after a successful call; odd cycles right
/// after a failed call that left replica 0's work enqueued. Every result
/// must match the CPU plan and device memory must return to its level after
/// the first cycle of each kind. Requires `QUABLA_CUDA_NCCL_TEST=1`.
#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[test]
fn cuda_data_parallel_compile_execute_drop_cycles_keep_device_memory_stable_on_two_gpus() {
    if std::env::var_os("QUABLA_CUDA_NCCL_TEST").is_none() {
        return;
    }
    const CYCLES: usize = 50;
    const BASELINE_CYCLE: usize = 1;
    // A leaked communicator pair, NVRTC module, or replica buffer set per
    // cycle would exceed this over the remaining 48 cycles.
    const MAX_DRIFT_BYTES: usize = 64 << 20;
    let (plan, sharding, _, _) = must!(mixed_reduction_data_parallel_plan());
    let inputs = must!(mixed_reduction_data_parallel_inputs());
    let expected = must!(plan.evaluate_many(&inputs));
    let (replica_inputs, failing_inputs) = must!(two_gpu_replica_inputs(&inputs));
    // Holding the primary contexts keeps them alive between cycles, so the
    // measurement sees allocations inside them rather than context teardown.
    let contexts = must!([0, 1]
        .into_iter()
        .map(cudarc::driver::CudaContext::new)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to create CUDA contexts: {error:?}")));
    let before = must!(cuda_free_bytes(&contexts));
    println!("lifecycle before free_bytes={before:?}");

    let start = std::time::Instant::now();
    let mut baseline = Vec::new();
    let mut after = Vec::new();
    for cycle in 0..CYCLES {
        let parallel = must!(CudaBackend::new(0).compile_data_parallel_sharded(
            plan.clone(),
            &sharding,
            vec![0, 1]
        ));
        let result = must!(parallel.execute_sharded(&replica_inputs));
        assert_data_parallel_outputs_match(&result.outputs, &expected, &format!("cycle {cycle}"));
        if cycle % 2 == 1 {
            let error = parallel
                .execute_sharded(&failing_inputs)
                .expect_err("a replica without its shard must fail");
            assert!(error.contains("next call recreates them"), "{error}");
        }
        drop(parallel);
        after = must!(cuda_free_bytes(&contexts));
        println!("lifecycle cycle={cycle} free_bytes={after:?}");
        if cycle == BASELINE_CYCLE {
            baseline = after.clone();
        }
    }
    println!(
        "lifecycle cycles={CYCLES} elapsed={:?} baseline_free_bytes={baseline:?} final_free_bytes={after:?}",
        start.elapsed()
    );
    for (device, (baseline, after)) in baseline.iter().zip(&after).enumerate() {
        let drift = baseline.saturating_sub(*after);
        assert!(
            drift <= MAX_DRIFT_BYTES,
            "device {device} lost {drift} bytes after cycle {BASELINE_CYCLE}"
        );
    }
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

    assert!(source.contains("quabla_fused_region_6"));
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
fn cuda_epilogue_test_inputs(
    specs: &[(&str, Vec<usize>)],
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    specs
        .iter()
        .enumerate()
        .map(|(seed, (name, shape))| {
            let count = shape.iter().product::<usize>();
            let data = (0..count)
                .map(|index| ((index * 7 + seed) % 11) as f64 * 0.1 - 0.5)
                .collect();
            Ok((name.to_string(), DynamicTensor::new(shape.clone(), data)?))
        })
        .collect()
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_matmul_bias_tanh_epilogue_only_fuses_plan_inputs_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let inputs = must!(cuda_epilogue_test_inputs(&[
        ("x", vec![4, 3]),
        ("w", vec![3, 2]),
        ("b", vec![1, 2]),
    ]));
    // In order: raw input, computed lhs, rhs, bias; only raw inputs may take the fused epilogue.
    for computed in [None, Some("x"), Some("w"), Some("b")] {
        let mut graph = TensorIr::new();
        let operand = |graph: &mut TensorIr, name: &str, shape| {
            let input = graph.input(name, shape)?;
            if computed == Some(name) {
                graph.tanh(input)
            } else {
                Ok(input)
            }
        };
        let x = must!(operand(&mut graph, "x", vec![4, 3]));
        let w = must!(operand(&mut graph, "w", vec![3, 2]));
        let b = must!(operand(&mut graph, "b", vec![1, 2]));
        let product = must!(graph.matmul(x, w));
        let shifted = must!(graph.add(product, b));
        let output = must!(graph.tanh(shifted));
        let plan = must!(graph.compile_cpu(output));

        let cpu = must!(plan.evaluate(&inputs));
        let cuda_plan = must!(CudaBackend::new(0).compile(plan));
        assert_eq!(cuda_plan.uses_fused_matmul_bias_tanh(), computed.is_none());
        let cuda = must!(cuda_plan.execute(&inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "CUDA matmul-bias-tanh with computed {computed:?} mismatch: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_two_layer_mlp_matches_cpu_across_single_and_multi_output_plans_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 3]));
    let w1 = must!(graph.input("w1", vec![3, 5]));
    let b1 = must!(graph.input("b1", vec![1, 5]));
    let w2 = must!(graph.input("w2", vec![5, 2]));
    let b2 = must!(graph.input("b2", vec![1, 2]));
    let product = must!(graph.matmul(x, w1));
    let shifted = must!(graph.add(product, b1));
    let hidden = must!(graph.tanh(shifted));
    let product = must!(graph.matmul(hidden, w2));
    let shifted = must!(graph.add(product, b2));
    let output = must!(graph.tanh(shifted));
    let transformed = must!(graph.symbolic_vjp(output, "seed"));
    let inputs = must!(cuda_epilogue_test_inputs(&[
        ("x", vec![4, 3]),
        ("w1", vec![3, 5]),
        ("b1", vec![1, 5]),
        ("w2", vec![5, 2]),
        ("b2", vec![1, 2]),
        ("seed", vec![4, 2]),
    ]));

    // When the first output is a matmul-bias-tanh of raw inputs, the remaining outputs must still
    // be evaluated per node.
    let plans = [
        must!(graph.compile_cpu_many(&[output])),
        must!(graph.compile_cpu_many(&[hidden, output])),
        must!(transformed.graph.compile_cpu_many(&[
            transformed.value,
            transformed.gradients["w1"],
            transformed.gradients["b1"],
            transformed.gradients["w2"],
            transformed.gradients["b2"],
        ])),
    ];
    for (plan, output_ids) in plans {
        let cpu = must!(plan.evaluate_many(&inputs));
        let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
        assert_eq!(cuda.len(), cpu.len());
        for (actual, expected) in cuda.iter().zip(cpu.iter()) {
            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.data().iter().zip(expected.data()) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "CUDA two-layer MLP mismatch: {actual} vs {expected}"
                );
            }
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_broadcast_batched_matmul_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
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

/// Executes `outputs` on the CPU and on MLX and requires identical values.
///
/// The readback tests below only move exactly representable values through layout views
/// (transpose, broadcast, slice) or add exactly representable constants, so the f32 MLX result
/// must equal the CPU reference bit for bit; any strided misread shows up as a wrong element.
#[cfg(all(feature = "mlx", target_os = "macos"))]
fn assert_mlx_readback_matches_cpu(
    graph: &TensorIr,
    outputs: &[TensorNodeId],
    inputs: &BTreeMap<String, DynamicTensor>,
) {
    let (plan, output_ids) = must!(graph.compile_cpu_many(outputs));
    let cpu = must!(plan.evaluate_many(inputs));
    let mlx = must!(MlxBackend.execute_many(&plan, &output_ids, inputs));
    assert_eq!(mlx.len(), cpu.len());
    for (index, (actual, expected)) in mlx.iter().zip(&cpu).enumerate() {
        assert_eq!(actual.shape(), expected.shape(), "output {index} shape");
        assert_eq!(actual.data(), expected.data(), "output {index} data");
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
fn iota_input(name: &str, shape: Vec<usize>) -> (String, DynamicTensor) {
    let count = shape.iter().product::<usize>();
    let data = (1..=count).map(|value| value as f64).collect();
    (name.to_string(), must_tensor(shape, data))
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
fn must_tensor(shape: Vec<usize>, data: Vec<f64>) -> DynamicTensor {
    DynamicTensor::new(shape, data).expect("valid test tensor")
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_readback_of_transposed_output_is_row_major() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let output = must!(graph.transpose(x, Some(vec![1, 0])));
    let inputs = BTreeMap::from([iota_input("x", vec![2, 3])]);
    assert_mlx_readback_matches_cpu(&graph, &[output], &inputs);
    let plan = must!(graph.compile_cpu(output));
    assert_eq!(
        must!(MlxBackend.execute(&plan, &inputs)).data(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_readback_of_broadcast_output_is_row_major() {
    let mut graph = TensorIr::new();
    let row = must!(graph.input("row", vec![1, 3]));
    let column = must!(graph.input("column", vec![2, 1]));
    let rows = must!(graph.broadcast_to(row, vec![2, 3]));
    let columns = must!(graph.broadcast_to(column, vec![2, 3]));
    let inputs = BTreeMap::from([
        iota_input("row", vec![1, 3]),
        iota_input("column", vec![2, 1]),
    ]);
    assert_mlx_readback_matches_cpu(&graph, &[rows], &inputs);
    assert_mlx_readback_matches_cpu(&graph, &[columns], &inputs);
    let plan = must!(graph.compile_cpu(rows));
    assert_eq!(
        must!(MlxBackend.execute(&plan, &inputs)).data(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_readback_of_rank3_permutation_is_row_major() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3, 4]));
    let permuted = must!(graph.transpose(x, Some(vec![2, 0, 1])));
    let swapped = must!(graph.transpose(x, Some(vec![0, 2, 1])));
    let inputs = BTreeMap::from([iota_input("x", vec![2, 3, 4])]);
    assert_mlx_readback_matches_cpu(&graph, &[permuted], &inputs);
    assert_mlx_readback_matches_cpu(&graph, &[swapped], &inputs);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_readback_of_sliced_transpose_is_row_major() {
    // A slice of a transpose is strided and starts at a non-zero storage offset.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3, 4]));
    let transposed = must!(graph.transpose(x, None));
    let rows = must!(graph.slice_axis(transposed, 0, 1, 3));
    let block = must!(graph.slice_axis(rows, 1, 1, 3));
    let inputs = BTreeMap::from([iota_input("x", vec![3, 4])]);
    assert_mlx_readback_matches_cpu(&graph, &[rows], &inputs);
    assert_mlx_readback_matches_cpu(&graph, &[block], &inputs);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_multi_output_readback_mixes_contiguous_and_strided_outputs() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let bias = must!(graph.input("bias", vec![1, 3]));
    let half = graph.scalar_constant(0.5);
    let shifted = must!(graph.add(x, half));
    let transposed = must!(graph.transpose(x, None));
    let broadcast = must!(graph.broadcast_to(bias, vec![4, 3]));
    let total = must!(graph.sum(shifted));
    let inputs = BTreeMap::from([iota_input("x", vec![2, 3]), iota_input("bias", vec![1, 3])]);
    assert_mlx_readback_matches_cpu(&graph, &[shifted, transposed, broadcast, total, x], &inputs);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_retained_input_readback_of_transposed_weight_is_row_major() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let weight = must!(graph.input("weight", vec![2, 3]));
    let weight_t = must!(graph.transpose(weight, None));
    let product = must!(graph.matmul(x, weight_t));
    let inputs = BTreeMap::from([
        iota_input("x", vec![2, 3]),
        iota_input("weight", vec![2, 3]),
    ]);
    let (plan, output_ids) = must!(graph.compile_cpu_many(&[weight_t, product]));
    let cpu = must!(plan.evaluate_many(&inputs));
    let state = must!(MlxRetainedInputs::upload(&inputs, ["weight".to_string()]));
    let dynamic_inputs = BTreeMap::from([iota_input("x", vec![2, 3])]);
    let mlx =
        must!(MlxBackend.execute_many_with_state(&plan, &output_ids, &dynamic_inputs, &state));
    assert_eq!(mlx[0].data(), cpu[0].data());
    for (actual, expected) in mlx[1].data().iter().zip(cpu[1].data()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "actual={actual}, expected={expected}"
        );
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_readback_of_elementwise_result_over_transpose_is_row_major() {
    // MLX elementwise kernels keep a column-major input layout for their output, so ordinary
    // arithmetic on a transpose also yields a non-row-major array.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let transposed = must!(graph.transpose(x, None));
    let half = graph.scalar_constant(0.5);
    let shifted = must!(graph.add(transposed, half));
    let doubled = must!(graph.add(transposed, transposed));
    let inputs = BTreeMap::from([iota_input("x", vec![2, 3])]);
    assert_mlx_readback_matches_cpu(&graph, &[shifted], &inputs);
    assert_mlx_readback_matches_cpu(&graph, &[doubled], &inputs);
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_cond_reads_offset_predicates_and_returns_strided_branch_outputs() {
    // The predicate is a scalar view at a non-zero storage offset of a transpose, and both
    // branches return non-row-major arrays computed from their captured input.
    let mut true_branch = TensorIr::new();
    let true_value = must!(true_branch.input("value", vec![2, 3]));
    let true_output = must!(true_branch.transpose(true_value, None));
    let mut false_branch = TensorIr::new();
    let false_value = must!(false_branch.input("value", vec![2, 3]));
    let false_transposed = must!(false_branch.transpose(false_value, None));
    let false_output = must!(false_branch.add(false_transposed, false_transposed));
    let branches = must!(TensorCondExecutionPlan::new(
        must!(true_branch.compile_cpu(true_output)),
        must!(false_branch.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 3]));
    let transposed = must!(graph.transpose(x, None));
    let predicate_row = must!(graph.slice_axis(transposed, 0, 1, 2));
    let predicate_cell = must!(graph.slice_axis(predicate_row, 1, 1, 2));
    let predicate = must!(graph.reshape(predicate_cell, vec![]));
    let output =
        must!(graph.cond_with_captures(predicate, branches, vec![("value".to_string(), x)],));
    // x[1][1] is the predicate; every other entry is non-zero so a misread flips the branch.
    for x_values in [
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0],
    ] {
        let inputs = BTreeMap::from([("x".to_string(), must_tensor(vec![2, 3], x_values))]);
        assert_mlx_readback_matches_cpu(&graph, &[output], &inputs);
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_serializes_concurrent_execution_across_threads() {
    // MLX's default GPU stream is process-wide; if the backend were not serialized, concurrent
    // evals from multiple threads would trigger the Metal "uncommitted encoder" assertion or hang.
    // Fori evaluates on every step, which raises the chance of interleaving.
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

fn f32_inputs(values: &[(&str, Vec<usize>, Vec<f64>)]) -> BTreeMap<String, DynamicTensor> {
    values
        .iter()
        .map(|(name, shape, data)| {
            (
                name.to_string(),
                DynamicTensor::new(shape.clone(), data.clone()).expect("valid test tensor"),
            )
        })
        .collect()
}

#[test]
fn typed_f32_input_adopts_weak_scalars_and_lowers_as_f32() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![2], TensorDType::F32));
    let tenth = graph.scalar_constant(0.1);
    let output = must!(graph.add(x, tenth));
    assert_eq!(graph.node_dtype(output), Ok(TensorDType::F32));
    assert!(graph.lower_text().contains("tensor<2xf32>"));
    assert!(graph.lower_text().contains("cast("));

    // The weak scalar is first converted to f32(0.1), then added to the f32 input and rounded.
    let inputs = f32_inputs(&[("x", vec![2], vec![1.0, 2.5])]);
    let plan = must!(graph.compile_cpu(output));
    assert_eq!(plan.output_dtype(), Ok(TensorDType::F32));
    let value = must!(CpuBackend.execute(&plan, &inputs));
    assert_eq!(value.dtype(), TensorDType::F32);
    let expected = [1.0_f32 + 0.1_f32, 2.5_f32 + 0.1_f32];
    for (actual, expected) in value.data().iter().zip(expected) {
        assert_eq!(actual.to_bits(), f64::from(expected).to_bits());
    }
}

#[test]
fn mixing_strong_f32_and_f64_tensors_is_a_trace_time_error() {
    let mut graph = TensorIr::new();
    let single = must!(graph.input_typed("single", vec![2], TensorDType::F32));
    let double = must!(graph.input("double", vec![2]));
    let node_count = graph.lower_text().lines().count();
    let error = graph
        .add(single, double)
        .expect_err("strong tensors of different dtypes must not be promoted");
    assert!(error.contains("astype"), "{error}");
    assert!(error.contains("f32") && error.contains("f64"), "{error}");
    assert_eq!(graph.lower_text().lines().count(), node_count);
    let error = graph
        .concat(vec![single, double], 0)
        .expect_err("concat must follow the same promotion rule");
    assert!(error.contains("astype"), "{error}");

    let cast = must!(graph.cast(double, TensorDType::F32));
    let sum = must!(graph.add(single, cast));
    assert_eq!(graph.node_dtype(sum), Ok(TensorDType::F32));
}

#[test]
fn cpu_f32_arithmetic_matches_ieee_single_precision_bitwise() {
    let xs = [0.1, 1.0 / 3.0, -7.25e-3, 12345.678, 3.0e38];
    let ys = [0.3, 2.0 / 7.0, 1.5e4, -0.001, 10.0];
    let zs = [0.7, -1.0e-8, 3.1, 99.5, -1.0];
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![5], TensorDType::F32));
    let y = must!(graph.input_typed("y", vec![5], TensorDType::F32));
    let z = must!(graph.input_typed("z", vec![5], TensorDType::F32));
    let product = must!(graph.mul(x, y));
    let fused = must!(graph.add(product, z));
    let quotient = must!(graph.div(x, y));
    let root = must!(graph.sqrt(product));
    let inputs = f32_inputs(&[
        ("x", vec![5], xs.to_vec()),
        ("y", vec![5], ys.to_vec()),
        ("z", vec![5], zs.to_vec()),
    ]);
    let expected_fused = (0..5)
        .map(|i| (xs[i] as f32) * (ys[i] as f32) + zs[i] as f32)
        .collect::<Vec<_>>();
    // Both the fused elementwise path and the generic interpreter path must round per node.
    let fused_plan = must!(graph.compile_cpu(fused));
    assert!(fused_plan.uses_fused_elementwise_kernel());
    for value in [
        must!(CpuBackend.execute(&fused_plan, &inputs)),
        must!(graph.evaluate(fused, &inputs)),
    ] {
        assert_eq!(value.dtype(), TensorDType::F32);
        for (actual, expected) in value.data().iter().zip(&expected_fused) {
            assert_eq!(actual.to_bits(), f64::from(*expected).to_bits());
        }
    }
    let quotient = must!(graph.evaluate(quotient, &inputs));
    let root = must!(graph.evaluate(root, &inputs));
    for i in 0..5 {
        let (x, y) = (xs[i] as f32, ys[i] as f32);
        assert_eq!(quotient.data()[i].to_bits(), f64::from(x / y).to_bits());
        let product = x * y;
        let expected_root = if product < 0.0 {
            f32::NAN
        } else {
            product.sqrt()
        };
        if expected_root.is_nan() {
            assert!(root.data()[i].is_nan());
        } else {
            assert_eq!(root.data()[i].to_bits(), f64::from(expected_root).to_bits());
        }
    }
}

#[test]
fn cast_round_trips_are_kept_and_same_dtype_casts_are_removed() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![]));
    let single = must!(graph.cast(x, TensorDType::F32));
    let double = must!(graph.cast(single, TensorDType::F64));
    let plan = must!(graph.compile_cpu(double));
    let inputs = f32_inputs(&[("x", vec![], vec![0.1])]);
    let value = must!(CpuBackend.execute(&plan, &inputs));
    assert_eq!(value.dtype(), TensorDType::F64);
    assert_eq!(value.data(), &[0.10000000149011612]);
    assert_eq!(plan.node_count(), 3);

    // A lossy round trip of a constant must not be cancelled during folding either.
    let mut graph = TensorIr::new();
    let tenth = graph.scalar_constant(0.1);
    let single = must!(graph.cast(tenth, TensorDType::F32));
    let double = must!(graph.cast(single, TensorDType::F64));
    let plan = must!(graph.compile_cpu(double));
    assert_eq!(plan.node_count(), 1);
    assert_eq!(
        must!(CpuBackend.execute(&plan, &BTreeMap::new())).data(),
        &[0.10000000149011612]
    );

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let squared = must!(graph.mul(x, x));
    let without_cast = must!(graph.compile_cpu(squared)).node_count();
    let same = must!(graph.cast(x, TensorDType::F64));
    let squared_after_cast = must!(graph.mul(same, same));
    let plan = must!(graph.compile_cpu(squared_after_cast));
    assert_eq!(plan.node_count(), without_cast);
    assert!(!plan.lower_text().contains("cast("));
}

#[test]
fn cast_ad_rules_convert_tangents_and_cotangents_between_dtypes() {
    // loss(x) = sum(cast(x, f32)^2), with x in f64.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let single = must!(graph.cast(x, TensorDType::F32));
    let squared = must!(graph.powi(single, 2));
    let loss = must!(graph.sum(squared));
    assert_eq!(graph.node_dtype(loss), Ok(TensorDType::F32));
    let point = [0.3, -1.7, 2.2];
    let inputs = f32_inputs(&[("x", vec![3], point.to_vec())]);

    let symbolic = must!(graph.symbolic_vjp(loss, "seed"));
    let gradient = symbolic.gradients["x"];
    assert_eq!(
        symbolic.graph.node_dtype(symbolic.cotangent),
        Ok(TensorDType::F32)
    );
    assert_eq!(symbolic.graph.node_dtype(gradient), Ok(TensorDType::F64));
    let mut seeded = inputs.clone();
    seeded.insert(
        "seed".to_string(),
        must!(DynamicTensor::new(vec![], vec![1.0])),
    );
    let symbolic_gradient = must!(symbolic.graph.evaluate(gradient, &seeded));
    assert_eq!(symbolic_gradient.dtype(), TensorDType::F64);
    let (_, eager_gradients) =
        must!(graph.value_and_vjp(loss, &inputs, must!(DynamicTensor::new(vec![], vec![1.0]))));
    assert_eq!(eager_gradients["x"].dtype(), TensorDType::F64);

    let loss_at = |values: &[f64]| -> f64 {
        values
            .iter()
            .map(|value| {
                let single = *value as f32;
                single * single
            })
            .sum::<f32>()
            .into()
    };
    // Central differences have no truncation error on a quadratic; the remaining error comes from
    // f32 rounding (~|f| 6e-8 / h).
    let step = 1e-2;
    for index in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[index] += step;
        minus[index] -= step;
        let finite_difference = (loss_at(&plus) - loss_at(&minus)) / (2.0 * step);
        for actual in [
            symbolic_gradient.data()[index],
            eager_gradients["x"].data()[index],
        ] {
            assert!(
                (actual - finite_difference).abs() < 1e-4,
                "gradient {actual} differs from finite difference {finite_difference}"
            );
        }
    }

    let tangent = must!(graph.symbolic_jvp(loss, "x"));
    assert_eq!(
        tangent.graph.node_dtype(tangent.tangent),
        Ok(TensorDType::F32)
    );
    let tangent_value = must!(tangent.graph.evaluate(tangent.tangent, &inputs));
    assert_eq!(tangent_value.dtype(), TensorDType::F32);
    let plus = point.map(|value| value + step);
    let minus = point.map(|value| value - step);
    let directional = (loss_at(&plus) - loss_at(&minus)) / (2.0 * step);
    assert!((tangent_value.data()[0] - directional).abs() < 1e-4);
}

#[test]
fn cse_keeps_casts_distinct_from_their_sources_and_merges_identical_casts() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let first = must!(graph.cast(x, TensorDType::F32));
    let second = must!(graph.cast(x, TensorDType::F32));
    let (_, outputs) = must!(graph.compile_cpu_many(&[x, first, second]));
    assert_ne!(outputs[0], outputs[1]);
    assert_eq!(outputs[1], outputs[2]);

    // Constants with the same value but different dtypes must not be merged either.
    let mut graph = TensorIr::new();
    let tenth = graph.scalar_constant(0.1);
    let single = must!(graph.cast(tenth, TensorDType::F32));
    let (plan, outputs) = must!(graph.compile_cpu_many(&[tenth, single]));
    assert_ne!(outputs[0], outputs[1]);
    assert_eq!(plan.node_dtype(outputs[1]), Ok(TensorDType::F32));
}

#[test]
fn f32_kernel_ir_validates_and_f64_kernel_ir_is_unchanged() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![2], TensorDType::F32));
    let doubled = must!(graph.add(x, x));
    let output = must!(graph.tanh(doubled));
    let program = must!(graph.compile_cpu(output)).kernel_ir();
    must!(program.validate());
    assert!(program
        .nodes
        .iter()
        .all(|node| node.dtype == TensorDType::F32));

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let doubled = must!(graph.add(x, x));
    let program = must!(graph.compile_cpu(doubled)).kernel_ir();
    must!(program.validate());
    assert!(program
        .nodes
        .iter()
        .all(|node| node.dtype == TensorDType::F64));
    assert!(graph.lower_text().contains("tensor<2xf64>"));
    assert!(!graph.lower_text().contains("f32"));
}

#[test]
fn f64_host_data_fed_to_an_f32_input_is_rounded() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![2], TensorDType::F32));
    let one = graph.scalar_constant(1.0);
    let output = must!(graph.mul(x, one));
    let inputs = f32_inputs(&[("x", vec![2], vec![0.1, 1.0 + 1e-12])]);
    for value in [
        must!(CpuBackend.execute(&must!(graph.compile_cpu(output)), &inputs)),
        must!(graph.evaluate(x, &inputs)),
    ] {
        assert_eq!(value.dtype(), TensorDType::F32);
        assert_eq!(value.data(), &[f64::from(0.1_f32), 1.0]);
    }
    let rounded = must!(DynamicTensor::with_dtype(
        vec![1],
        vec![0.1],
        TensorDType::F32
    ));
    assert_eq!(rounded.data(), &[f64::from(0.1_f32)]);
    assert_eq!(rounded.astype(TensorDType::F64).dtype(), TensorDType::F64);
}

#[test]
fn f32_fori_carry_adopts_a_typed_index_and_rejects_mixed_captures() {
    let mut body = TensorIr::new();
    let carry = must!(body.input_typed("carry", vec![], TensorDType::F32));
    let index = must!(body.input_typed("index", vec![], TensorDType::F32));
    let tenth = body.scalar_constant(0.1);
    let step = must!(body.mul(index, tenth));
    let next = must!(body.add(carry, step));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        4,
        must!(body.compile_cpu(next)),
        "carry",
        "index"
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input_typed("initial", vec![], TensorDType::F32));
    let output = must!(graph.fori(initial, loop_plan.clone(), vec![]));
    assert_eq!(graph.node_dtype(output), Ok(TensorDType::F32));
    let inputs = f32_inputs(&[("initial", vec![], vec![1.0])]);
    let value = must!(graph.evaluate(output, &inputs));
    let expected = (0..4).fold(1.0_f32, |carry, index| carry + index as f32 * 0.1_f32);
    assert_eq!(value.dtype(), TensorDType::F32);
    assert_eq!(value.data(), &[f64::from(expected)]);

    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let error = graph
        .fori(initial, loop_plan, vec![])
        .expect_err("an f64 carry must not bind an f32 loop region");
    assert!(error.contains("astype"), "{error}");
}

#[test]
fn cond_region_transforms_do_not_leak_non_finite_captures() {
    // The false branch avoids NaN with a greater mask; when region AD retains a capture, it must
    // not read the capture's value.
    let mut on_true = TensorIr::new();
    let true_x = must!(on_true.input("x", vec![3]));
    let true_squared = must!(on_true.mul(true_x, true_x));
    let true_output = must!(on_true.sum(true_squared));
    let mut on_false = TensorIr::new();
    let false_x = must!(on_false.input("x", vec![3]));
    let zero = on_false.scalar_constant(0.0);
    let positive = must!(on_false.greater(false_x, zero));
    let selected = must!(on_false.where_select(positive, false_x, zero));
    let false_output = must!(on_false.sum(selected));
    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    must!(graph.input("x", vec![3]));
    let output = must!(graph.cond(predicate, branches));
    let inputs = f32_inputs(&[
        ("predicate", vec![], vec![0.0]),
        ("x", vec![3], vec![1.0, f64::NAN, f64::NEG_INFINITY]),
        ("seed", vec![], vec![1.0]),
    ]);
    assert_eq!(must!(graph.evaluate(output, &inputs)).data(), &[1.0]);
    let vjp = must!(graph.symbolic_vjp(output, "seed"));
    assert_eq!(
        must!(vjp.graph.evaluate(vjp.gradients["x"], &inputs)).data(),
        &[1.0, 0.0, 0.0]
    );
    let jvp = must!(graph.symbolic_jvp(output, "x"));
    assert_eq!(must!(jvp.graph.evaluate(jvp.value, &inputs)).data(), &[1.0]);
    assert_eq!(
        must!(jvp.graph.evaluate(jvp.tangent, &inputs)).data(),
        &[1.0]
    );
}

// ---- Dtype phase D2: Bool masks, comparisons, and logical reductions ----

use quabla_core::tensor_ir::TensorComparison;

const BOOL_X: [f64; 6] = [1.0, 2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0];
const BOOL_Y: [f64; 6] = [0.5, 2.0, 2.0, f64::INFINITY, 0.0, f64::NAN];
const BOOL_W: [f64; 6] = [0.5, -1.5, 2.0, 0.25, 1.0, -0.75];
const BOOL_R: [f64; 6] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];

fn bool_inputs(x: &[f64]) -> BTreeMap<String, DynamicTensor> {
    let mut inputs = f32_inputs(&[
        ("x", vec![2, 3], x.to_vec()),
        ("y", vec![2, 3], BOOL_Y.to_vec()),
        ("w", vec![2, 3], BOOL_W.to_vec()),
        ("seed", vec![], vec![1.0]),
    ]);
    inputs.insert(
        "r".to_string(),
        DynamicTensor::with_dtype(vec![2, 3], BOOL_R.to_vec(), TensorDType::F32)
            .expect("valid test tensor"),
    );
    inputs
}

/// Output nodes paired with their expected CPU values.
type ExpectedOutputs = Vec<(TensorNodeId, Vec<f64>)>;

/// Every D2 primitive over `x`/`y` (NaN, ±inf and ties) plus the promotion
/// rules against an `f32` residual `r`. Outputs are exact 0/1 masks or
/// finite values, paired with their expected CPU results.
fn bool_primitive_program() -> Result<(TensorIr, ExpectedOutputs), String> {
    use TensorComparison::*;
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2, 3])?;
    let y = graph.input("y", vec![2, 3])?;
    let r = graph.input_typed("r", vec![2, 3], TensorDType::F32)?;
    let mut outputs = Vec::new();
    let expected_compare = [
        (Greater, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (GreaterEqual, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        (Less, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        (LessEqual, [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        (Equal, [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        (NotEqual, [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
    ];
    for (kind, expected) in expected_compare {
        outputs.push((graph.compare(x, y, kind)?, expected.to_vec()));
    }
    let greater = outputs[0].0;
    let less = outputs[2].0;
    let equal = outputs[4].0;
    let x_finite = graph.isfinite(x)?;
    let y_finite = graph.isfinite(y)?;
    let x_nan = graph.isnan(x)?;
    let y_nan = graph.isnan(y)?;
    outputs.extend([
        (x_finite, vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        (x_nan, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        (
            graph.logical_and(greater, x_finite)?,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            graph.logical_or(less, y_nan)?,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ),
        (
            graph.logical_not(equal)?,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        ),
        (graph.any(x_nan)?, vec![1.0]),
        (graph.all(y_finite)?, vec![0.0]),
        (graph.any_axis(greater, 1)?, vec![1.0, 0.0]),
        (graph.all_axis(y_finite, 0)?, vec![0.0, 1.0, 0.0]),
    ]);
    // An f32 residual times a Bool mask gives f32; mask * 2.0 is a weak f64 and adopts f32 when
    // combined with the f32 residual.
    let masked = graph.mul(r, x_finite)?;
    let two = graph.scalar_constant(2.0);
    let weak = graph.mul(x_finite, two)?;
    let shifted = graph.add(weak, r)?;
    let r32 = BOOL_R.map(|value| value as f32);
    let finite32 = [1.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0];
    outputs.push((
        masked,
        (0..6).map(|i| f64::from(r32[i] * finite32[i])).collect(),
    ));
    outputs.push((
        shifted,
        (0..6)
            .map(|i| f64::from(finite32[i] * 2.0 + r32[i]))
            .collect(),
    ));
    Ok((graph, outputs))
}

/// `loss(x, w)` guards non-finite `x` with `where(isfinite(x), x, 0)` before
/// any arithmetic, then masks a weighted residual with `safe > 0.5`.
fn masked_loss_graph() -> Result<(TensorIr, TensorNodeId), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2, 3])?;
    let w = graph.input("w", vec![2, 3])?;
    let finite = graph.isfinite(x)?;
    let zero = graph.scalar_constant(0.0);
    let safe = graph.where_select(finite, x, zero)?;
    let half = graph.scalar_constant(0.5);
    let active = graph.compare(safe, half, TensorComparison::Greater)?;
    let weighted = graph.mul(safe, w)?;
    let one = graph.scalar_constant(1.0);
    let shifted = graph.sub(weighted, one)?;
    let residual = graph.where_select(active, shifted, zero)?;
    let squared = graph.powi(residual, 2)?;
    let data_loss = graph.mean(squared)?;
    let safe_squared = graph.powi(safe, 2)?;
    let penalty = graph.sum(safe_squared)?;
    let tenth = graph.scalar_constant(0.1);
    let penalty = graph.mul(penalty, tenth)?;
    let loss = graph.add(data_loss, penalty)?;
    Ok((graph, loss))
}

/// Analytic value and gradients of `masked_loss_graph` at `BOOL_X`, `BOOL_W`:
/// safe = [1, 2, 0, 0, 0, 2], active = [1, 1, 0, 0, 0, 1].
const MASKED_LOSS: f64 = 22.5 / 6.0 + 0.9;
const MASKED_GRAD_X: [f64; 6] = [-1.0 / 12.0 + 0.2, 2.0 + 0.4, 0.0, 0.0, 0.0, 0.625 + 0.4];
const MASKED_GRAD_W: [f64; 6] = [-1.0 / 6.0, -8.0 / 3.0, 0.0, 0.0, 0.0, -5.0 / 3.0];

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} (tolerance {tolerance})"
        );
    }
}

#[test]
fn bool_dtype_rounds_to_zero_one_and_lowers_to_f32_on_devices() {
    assert_eq!(TensorDType::Bool.to_string(), "bool");
    assert_eq!(TensorDType::Bool.round(f64::NAN), 1.0);
    assert_eq!(TensorDType::Bool.round(-0.0), 0.0);
    assert_eq!(TensorDType::Bool.round(-2.5), 1.0);
    let tensor = must!(DynamicTensor::with_dtype(
        vec![4],
        vec![0.0, 3.0, f64::NAN, f64::NEG_INFINITY],
        TensorDType::Bool,
    ));
    assert_eq!(tensor.data(), &[0.0, 1.0, 1.0, 1.0]);
    assert_eq!(tensor.dtype(), TensorDType::Bool);
    assert_eq!(
        TensorDeviceBackend::Cpu.execution_dtype(TensorDType::Bool),
        Ok(TensorDType::Bool)
    );
    for backend in [TensorDeviceBackend::Cuda, TensorDeviceBackend::Mlx] {
        assert_eq!(
            backend.execution_dtype(TensorDType::Bool),
            Ok(TensorDType::F32)
        );
    }
}

#[test]
fn comparisons_logical_ops_and_reductions_follow_ieee_semantics_on_cpu() {
    let (graph, outputs) = must!(bool_primitive_program());
    let inputs = bool_inputs(&BOOL_X);
    let text = graph.lower_text();
    assert!(
        text.contains("compare(%0, %1, kind=greater_equal) : tensor<2x3xi1>"),
        "{text}"
    );
    assert!(text.contains(": tensor<i1>"), "{text}");
    for (index, (output, expected)) in outputs.iter().enumerate() {
        let dtype = if index < 15 {
            TensorDType::Bool
        } else {
            TensorDType::F32
        };
        assert_eq!(graph.node_dtype(*output), Ok(dtype), "output {index}");
        // The per-node interpreter and (when fusible) the fused interpreter must agree.
        let evaluated = must!(graph.evaluate(*output, &inputs));
        let plan = must!(graph.compile_cpu(*output));
        let executed = must!(CpuBackend.execute(&plan, &inputs));
        for value in [&evaluated, &executed] {
            assert_eq!(value.dtype(), dtype, "output {index}");
            assert_eq!(value.data(), expected.as_slice(), "output {index}");
        }
        let program = plan.kernel_ir();
        must!(program.validate());
    }
    let compare_plan = must!(graph.compile_cpu(outputs[1].0));
    assert!(compare_plan.uses_fused_elementwise_kernel());
    let kernel = compare_plan.kernel_ir();
    let compare_node = kernel.nodes.last().expect("kernel node");
    assert_eq!(compare_node.op, "greater_equal");
    assert_eq!(compare_node.dtype, TensorDType::Bool);

    // The StableHLO probe represents Bool as i1; comparisons are outside the verified subset and
    // are rejected explicitly.
    let mut probe = TensorIr::new();
    let mask = must!(probe.input_typed("mask", vec![3], TensorDType::Bool));
    let numeric = must!(probe.cast(mask, TensorDType::F64));
    let stablehlo = must!(probe.stablehlo_text(numeric));
    assert!(
        stablehlo.contains("stablehlo.convert %arg0 : (tensor<3xi1>) -> tensor<3xf64>"),
        "{stablehlo}"
    );
    let zero = probe.scalar_constant(0.0);
    let compared = must!(probe.compare(numeric, zero, TensorComparison::Less));
    assert!(probe.stablehlo_text(compared).is_err());

    // Constant comparisons fold into Bool constants at plan compile time.
    let mut constant = TensorIr::new();
    let one = constant.scalar_constant(1.0);
    let two = constant.scalar_constant(2.0);
    let less = must!(constant.compare(one, two, TensorComparison::Less));
    let plan = must!(constant.compile_cpu(less));
    assert_eq!(plan.node_count(), 1);
    assert_eq!(
        must!(CpuBackend.execute(&plan, &BTreeMap::new())).data(),
        &[1.0]
    );
}

#[test]
fn bool_operands_promote_like_arithmetic_and_reject_bool_arithmetic() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let r = must!(graph.input_typed("r", vec![2], TensorDType::F32));
    let zero = graph.scalar_constant(0.0);
    let mask = must!(graph.compare(x, zero, TensorComparison::Greater));
    let other = must!(graph.compare(x, zero, TensorComparison::Less));
    assert_eq!(graph.node_dtype(mask), Ok(TensorDType::Bool));

    let masked = must!(graph.mul(r, mask));
    assert_eq!(graph.node_dtype(masked), Ok(TensorDType::F32));
    let two = graph.scalar_constant(2.0);
    let weak = must!(graph.mul(mask, two));
    assert_eq!(graph.node_dtype(weak), Ok(TensorDType::F64));
    let adopted = must!(graph.add(weak, r));
    assert_eq!(graph.node_dtype(adopted), Ok(TensorDType::F32));
    let strong = must!(graph.add(weak, x));
    assert_eq!(graph.node_dtype(strong), Ok(TensorDType::F64));

    let node_count = graph.lower_text().lines().count();
    for (name, result) in [
        ("add", graph.add(mask, other)),
        ("mul", graph.mul(mask, other)),
        ("tanh", graph.tanh(mask)),
        ("sum", graph.sum(mask)),
        ("greater", graph.greater(mask, other)),
    ] {
        let error = result.expect_err("bool arithmetic must be rejected");
        assert!(error.contains(name) && error.contains("bool"), "{error}");
        assert!(error.contains("astype"), "{error}");
    }
    assert_eq!(graph.lower_text().lines().count(), node_count);
    for error in [
        graph.logical_and(mask, x).expect_err("float operand"),
        graph.logical_not(x).expect_err("float operand"),
        graph.any(x).expect_err("float operand"),
    ] {
        assert!(error.contains("requires bool operands"), "{error}");
    }
    assert!(graph.isnan(mask).is_err());

    // Bool can take part in data movement, select, and comparisons; two Bools compare as 0/1.
    let same = must!(graph.compare(mask, other, TensorComparison::Equal));
    let reshaped = must!(graph.reshape(same, vec![2, 1]));
    let selected = must!(graph.where_select(mask, same, other));
    assert_eq!(graph.node_dtype(reshaped), Ok(TensorDType::Bool));
    assert_eq!(graph.node_dtype(selected), Ok(TensorDType::Bool));

    // float->bool treats non-zero as true (NaN included); bool->float gives 0/1; legacy greater is
    // still a float mask.
    let truthy = must!(graph.cast(x, TensorDType::Bool));
    let numeric = must!(graph.cast(mask, TensorDType::F32));
    let legacy = must!(graph.greater(r, zero));
    assert_eq!(graph.node_dtype(truthy), Ok(TensorDType::Bool));
    assert_eq!(graph.node_dtype(numeric), Ok(TensorDType::F32));
    assert_eq!(graph.node_dtype(legacy), Ok(TensorDType::F32));
    let inputs = f32_inputs(&[
        ("x", vec![2], vec![f64::NAN, 0.0]),
        ("r", vec![2], vec![1.0, 1.0]),
    ]);
    assert_eq!(must!(graph.evaluate(truthy, &inputs)).data(), &[1.0, 0.0]);
    let inputs = f32_inputs(&[
        ("x", vec![2], vec![3.0, -1.0]),
        ("r", vec![2], vec![1.0, 1.0]),
    ]);
    assert_eq!(must!(graph.evaluate(numeric, &inputs)).data(), &[1.0, 0.0]);
}

#[test]
fn where_and_cond_accept_bool_and_legacy_float_predicates() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let zero = graph.scalar_constant(0.0);
    let bool_mask = must!(graph.compare(x, zero, TensorComparison::GreaterEqual));
    let float_mask = must!(graph.greater(x, zero));
    let negated = must!(graph.sub(zero, x));
    let bool_abs = must!(graph.where_select(bool_mask, x, negated));
    let float_abs = must!(graph.where_select(float_mask, x, negated));
    let inputs = f32_inputs(&[("x", vec![3], vec![-2.0, 0.0, 3.0])]);
    assert_eq!(
        must!(graph.evaluate(bool_abs, &inputs)).data(),
        &[2.0, 0.0, 3.0]
    );
    assert_eq!(
        must!(graph.evaluate(float_abs, &inputs)).data(),
        &[2.0, -0.0, 3.0]
    );

    // all(isfinite(x)) as a Cond predicate; the false branch guards non-finite values itself.
    let (graph, output) = must!(finite_guard_cond_graph());
    let finite = f32_inputs(&[("x", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(must!(graph.evaluate(output, &finite)).data(), &[91.0]);
    let guarded = f32_inputs(&[("x", vec![2, 3], BOOL_X.to_vec())]);
    assert_eq!(must!(graph.evaluate(output, &guarded)).data(), &[4.5]);

    // Bool branch results and Bool loop carries are rejected explicitly in D2.
    let mut branch = TensorIr::new();
    let captured = must!(branch.input("x", vec![]));
    let branch_zero = branch.scalar_constant(0.0);
    let branch_mask = must!(branch.compare(captured, branch_zero, TensorComparison::Less));
    let plan = must!(branch.compile_cpu(branch_mask));
    let error = TensorCondExecutionPlan::new(plan.clone(), plan)
        .expect_err("bool branch results are not supported");
    assert!(error.contains("astype"), "{error}");
    // Loop-region AD pairs every input with a tangent, so both Bool carries and Bool captures are
    // rejected.
    let mut body = TensorIr::new();
    let carry = must!(body.input_typed("carry", vec![], TensorDType::Bool));
    must!(body.input("index", vec![]));
    let error =
        TensorForiExecutionPlan::new(0, 2, must!(body.compile_cpu(carry)), "carry", "index")
            .expect_err("bool carries are not supported");
    assert!(error.contains("floating"), "{error}");
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![3]));
    must!(body.input("index", vec![]));
    let mask = must!(body.input_typed("mask", vec![3], TensorDType::Bool));
    let next = must!(body.where_select(mask, carry, carry));
    let error = TensorForiExecutionPlan::new(0, 2, must!(body.compile_cpu(next)), "carry", "index")
        .expect_err("bool captures are not supported");
    assert!(
        error.contains("\"mask\"") && error.contains("astype"),
        "{error}"
    );
}

fn finite_guard_cond_graph() -> Result<(TensorIr, TensorNodeId), String> {
    let mut on_true = TensorIr::new();
    let true_x = on_true.input("x", vec![2, 3])?;
    let true_squared = on_true.mul(true_x, true_x)?;
    let true_output = on_true.sum(true_squared)?;
    let mut on_false = TensorIr::new();
    let false_x = on_false.input("x", vec![2, 3])?;
    let finite = on_false.isfinite(false_x)?;
    let zero = on_false.scalar_constant(0.0);
    let safe = on_false.where_select(finite, false_x, zero)?;
    let safe_squared = on_false.mul(safe, safe)?;
    let summed = on_false.sum(safe_squared)?;
    let half = on_false.scalar_constant(0.5);
    let false_output = on_false.mul(summed, half)?;
    let branches = TensorCondExecutionPlan::new(
        on_true.compile_cpu(true_output)?,
        on_false.compile_cpu(false_output)?,
    )?;
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2, 3])?;
    let finite = graph.isfinite(x)?;
    let predicate = graph.all(finite)?;
    let output = graph.cond(predicate, branches)?;
    Ok((graph, output))
}

#[test]
fn bool_masks_have_no_derivatives_and_nan_guards_keep_gradients_finite() {
    let (graph, loss) = must!(masked_loss_graph());
    let inputs = bool_inputs(&BOOL_X);
    let seed = must!(DynamicTensor::new(vec![], vec![1.0]));
    let (value, eager) = must!(graph.value_and_vjp(loss, &inputs, seed.clone()));
    assert_close(value.data(), &[MASKED_LOSS], 1e-12);
    let symbolic = must!(graph.symbolic_vjp(loss, "seed"));
    for gradients in [
        [eager["x"].data().to_vec(), eager["w"].data().to_vec()],
        [
            must!(symbolic.graph.evaluate(symbolic.gradients["x"], &inputs))
                .data()
                .to_vec(),
            must!(symbolic.graph.evaluate(symbolic.gradients["w"], &inputs))
                .data()
                .to_vec(),
        ],
    ] {
        assert_close(&gradients[0], &MASKED_GRAD_X, 1e-12);
        assert_close(&gradients[1], &MASKED_GRAD_W, 1e-12);
    }

    // Central differences: the mask boundary (0.5) is far from every point, so the loss is locally
    // smooth in each coordinate.
    let finite_x = [1.0, 2.0, -0.3, 0.7, 0.2, 2.0];
    let finite_inputs = bool_inputs(&finite_x);
    let symbolic_gradient = must!(symbolic
        .graph
        .evaluate(symbolic.gradients["x"], &finite_inputs));
    let (_, tangent_graph_value) = {
        let transformed = must!(graph.symbolic_jvp(loss, "x"));
        let value = must!(transformed
            .graph
            .evaluate(transformed.tangent, &finite_inputs));
        ((), value)
    };
    let step = 1e-6;
    let mut directional = 0.0;
    for index in 0..6 {
        let mut plus = finite_x;
        let mut minus = finite_x;
        plus[index] += step;
        minus[index] -= step;
        let loss_at =
            |point: &[f64]| must_value(graph.evaluate(loss, &bool_inputs(point))).data()[0];
        let difference = (loss_at(&plus) - loss_at(&minus)) / (2.0 * step);
        directional += difference;
        assert!(
            (symbolic_gradient.data()[index] - difference).abs() < 1e-6,
            "gradient {} differs from finite difference {difference}",
            symbolic_gradient.data()[index]
        );
    }
    // symbolic_jvp seeds an all-ones direction, so the tangent equals the sum of the partial
    // derivatives.
    assert!((tangent_graph_value.data()[0] - directional).abs() < 1e-5);

    // The legacy float-mask form sum(x.gt(0) * x) lets 0 * NaN leak into values and gradients.
    let mut legacy = TensorIr::new();
    let x = must!(legacy.input("x", vec![2, 3]));
    let zero = legacy.scalar_constant(0.0);
    let mask = must!(legacy.greater(x, zero));
    let masked = must!(legacy.mul(mask, x));
    let legacy_loss = must!(legacy.sum(masked));
    let (legacy_value, legacy_gradients) =
        must!(legacy.value_and_vjp(legacy_loss, &inputs, seed.clone()));
    assert!(legacy_value.data()[0].is_nan());
    assert!(legacy_gradients["x"]
        .data()
        .iter()
        .all(|value| value.is_finite()));

    // Cond predicate all(isfinite(x)): the selected guarded branch still gives finite gradients
    // with NaN/inf present.
    let (cond_graph, cond_output) = must!(finite_guard_cond_graph());
    let transformed = must!(cond_graph.symbolic_vjp(cond_output, "seed"));
    let gradient = must!(transformed
        .graph
        .evaluate(transformed.gradients["x"], &inputs));
    assert_eq!(gradient.data(), &[1.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    let tangent = must!(cond_graph.symbolic_jvp(cond_output, "x"));
    assert_eq!(
        must!(tangent.graph.evaluate(tangent.tangent, &inputs)).data(),
        &[5.0]
    );
    let plan = must!(cond_graph.compile_cpu(cond_output));
    let eager = must!(plan.vjp(&inputs, seed.clone()));
    assert_eq!(eager["x"].data(), &[1.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
}

fn must_value(result: Result<DynamicTensor, String>) -> DynamicTensor {
    result.expect("evaluation succeeds")
}

#[test]
fn differentiating_bool_values_is_an_error_and_bool_inputs_have_no_gradient() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let mask = must!(graph.input_typed("mask", vec![3], TensorDType::Bool));
    let zero = graph.scalar_constant(0.0);
    let positive = must!(graph.compare(x, zero, TensorComparison::Greater));
    let both = must!(graph.logical_and(mask, positive));
    let squared = must!(graph.mul(x, x));
    let selected = must!(graph.where_select(both, squared, zero));
    let loss = must!(graph.sum(selected));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![3], vec![-1.0, 2.0, 3.0])),
        ),
        (
            "mask".to_string(),
            must!(DynamicTensor::with_dtype(
                vec![3],
                vec![1.0, 1.0, 0.0],
                TensorDType::Bool
            )),
        ),
    ]);
    let seed = must!(DynamicTensor::new(vec![], vec![1.0]));
    let (value, gradients) = must!(graph.value_and_vjp(loss, &inputs, seed.clone()));
    assert_eq!(value.data(), &[4.0]);
    assert_eq!(gradients["x"].data(), &[0.0, 4.0, 0.0]);
    assert!(!gradients.contains_key("mask"));
    let symbolic = must!(graph.symbolic_vjp(loss, "seed"));
    assert!(!symbolic.gradients.contains_key("mask"));
    let mut seeded = inputs.clone();
    seeded.insert("seed".to_string(), seed.clone());
    assert_eq!(
        must!(symbolic.graph.evaluate(symbolic.gradients["x"], &seeded)).data(),
        &[0.0, 4.0, 0.0]
    );

    let tangents = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![3], vec![1.0, 1.0, 1.0])),
    )]);
    let (_, tangent) = must!(graph.jvp(loss, &inputs, &tangents));
    assert_eq!(tangent.data(), &[4.0]);
    let (_, mask_tangent) = must!(graph.jvp(both, &inputs, &tangents));
    assert_eq!(mask_tangent.data(), &[0.0, 0.0, 0.0]);
    assert_eq!(mask_tangent.dtype(), TensorDType::F64);

    // A zero tangent is equivalent to omitting it (callers often supply a tangent per input); a
    // non-zero direction is an error.
    let mut with_mask = tangents.clone();
    with_mask.insert(
        "mask".to_string(),
        must!(DynamicTensor::filled(vec![3], 0.0)),
    );
    assert_eq!(
        must!(graph.jvp(loss, &inputs, &with_mask)).1.data(),
        tangent.data()
    );
    with_mask.insert(
        "mask".to_string(),
        must!(DynamicTensor::filled(vec![3], 1.0)),
    );
    for error in [
        graph
            .jvp(loss, &inputs, &with_mask)
            .map(|_| ())
            .expect_err("bool tangent"),
        graph
            .symbolic_jvp(loss, "mask")
            .map(|_| ())
            .expect_err("bool input"),
        graph
            .hessian_scalar(loss, "mask", &inputs)
            .map(|_| ())
            .expect_err("bool input"),
        graph
            .symbolic_jvp_with_tangent_inputs(
                loss,
                &BTreeMap::from([("mask".to_string(), "mask_tangent".to_string())]),
            )
            .map(|_| ())
            .expect_err("bool input"),
    ] {
        assert!(error.contains("bool input \"mask\""), "{error}");
    }
    for error in [
        graph
            .value_and_vjp(both, &inputs, must!(DynamicTensor::filled(vec![3], 1.0)))
            .map(|_| ())
            .expect_err("bool output"),
        graph
            .symbolic_vjp(both, "seed")
            .map(|_| ())
            .expect_err("bool output"),
    ] {
        assert!(error.contains("bool output"), "{error}");
    }

    // The tangent through Bool data movement (the reshape/cast/sum chain of any_axis) is zero.
    let all_positive = must!(graph.all_axis(positive, 0));
    let weights = must!(graph.cast(all_positive, TensorDType::F64));
    let weighted = must!(graph.mul(weights, squared));
    let total = must!(graph.sum(weighted));
    let transformed = must!(graph.symbolic_jvp(total, "x"));
    let (_, eager_tangent) = must!(graph.jvp(total, &inputs, &tangents));
    assert_eq!(
        must!(transformed.graph.evaluate(transformed.tangent, &inputs)).data(),
        eager_tangent.data()
    );
    assert_eq!(eager_tangent.data(), &[0.0]);
    let hessian = must!(graph.hessian_scalar(loss, "x", &inputs));
    assert_eq!(hessian[1][1], 2.0);
    assert_eq!(hessian[0][0], 0.0);
}

#[test]
fn cond_regions_capture_bool_masks_without_differentiating_them() {
    let branch = |masked: bool| -> Result<TensorExecutionPlan, String> {
        let mut region = TensorIr::new();
        let x = region.input("x", vec![3])?;
        let mask = region.input_typed("mask", vec![3], TensorDType::Bool)?;
        let squared = region.mul(x, x)?;
        let zero = region.scalar_constant(0.0);
        let selected = if masked {
            region.where_select(mask, squared, zero)?
        } else {
            let numeric = region.cast(mask, TensorDType::F64)?;
            region.add(squared, numeric)?
        };
        let output = region.sum(selected)?;
        region.compile_cpu(output)
    };
    let branches = must!(TensorCondExecutionPlan::new(
        must!(branch(true)),
        must!(branch(false))
    ));
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let mask = must!(graph.input_typed("mask", vec![3], TensorDType::Bool));
    let zero = graph.scalar_constant(0.0);
    let positive = must!(graph.compare(x, zero, TensorComparison::Greater));
    let predicate = must!(graph.any(positive));
    let output = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("x".to_string(), x), ("mask".to_string(), mask)],
    ));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![3], vec![1.0, -2.0, 3.0])),
        ),
        (
            "mask".to_string(),
            must!(DynamicTensor::with_dtype(
                vec![3],
                vec![1.0, 0.0, 1.0],
                TensorDType::Bool
            )),
        ),
        (
            "seed".to_string(),
            must!(DynamicTensor::new(vec![], vec![1.0])),
        ),
    ]);
    assert_eq!(must!(graph.evaluate(output, &inputs)).data(), &[10.0]);
    let vjp = must!(graph.symbolic_vjp(output, "seed"));
    assert!(!vjp.gradients.contains_key("mask"));
    assert_eq!(
        must!(vjp.graph.evaluate(vjp.gradients["x"], &inputs)).data(),
        &[2.0, 0.0, 6.0]
    );
    let jvp = must!(graph.symbolic_jvp(output, "x"));
    assert_eq!(
        must!(jvp.graph.evaluate(jvp.tangent, &inputs)).data(),
        &[8.0]
    );
}

/// Runs the D2 primitive program, the masked-loss value and symbolic
/// gradients, and the `all(isfinite(x))` Cond on a device against the CPU.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn assert_bool_parity(target: QuablaTarget) {
    let compiler = QuablaCompiler;
    let inputs = bool_inputs(&BOOL_X);
    let (graph, outputs) = must!(bool_primitive_program());
    let program = must!(QuablaMultiOutputProgram::new(
        graph,
        outputs.iter().map(|(output, _)| *output).collect()
    ));
    let device = must!(must!(compiler.compile_many(&program, target)).execute(&inputs));
    for ((actual, (_, expected)), index) in device.iter().zip(&outputs).zip(0..) {
        // Bool readback is tagged bool with values identical to the CPU; f32 outputs are likewise
        // native f32 arithmetic.
        assert_eq!(
            actual.data(),
            expected.as_slice(),
            "{target:?} output {index}"
        );
        let dtype = if index < 15 {
            TensorDType::Bool
        } else {
            TensorDType::F32
        };
        assert_eq!(actual.dtype(), dtype, "{target:?} output {index}");
    }

    let (graph, loss) = must!(masked_loss_graph());
    let vjp = must!(graph.symbolic_vjp(loss, "seed"));
    let outputs = vec![vjp.value, vjp.gradients["x"], vjp.gradients["w"]];
    let program = must!(QuablaMultiOutputProgram::new(vjp.graph, outputs));
    let cpu = must!(must!(compiler.compile_many(&program, QuablaTarget::Cpu)).execute(&inputs));
    let device = must!(must!(compiler.compile_many(&program, target)).execute(&inputs));
    let error = max_scaled_error(&device, &cpu);
    println!("{target:?} masked loss and gradients: scaled error {error:e}");
    assert!(error <= 1e-5, "device masked loss error {error:e}");
    assert!(device
        .iter()
        .all(|value| value.data().iter().all(|entry| entry.is_finite())));

    let (graph, output) = must!(finite_guard_cond_graph());
    let vjp = must!(graph.symbolic_vjp(output, "seed"));
    let program = must!(QuablaMultiOutputProgram::new(
        vjp.graph,
        vec![vjp.value, vjp.gradients["x"]]
    ));
    for x in [BOOL_X.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] {
        let inputs = bool_inputs(&x);
        let cpu = must!(must!(compiler.compile_many(&program, QuablaTarget::Cpu)).execute(&inputs));
        let device = must!(must!(compiler.compile_many(&program, target)).execute(&inputs));
        for (actual, expected) in device.iter().zip(&cpu) {
            assert_eq!(actual.data(), expected.data(), "{target:?} cond");
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_bool_masks_comparisons_and_guarded_gradients_match_cpu() {
    assert_bool_parity(QuablaTarget::Mlx);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_bool_masks_comparisons_and_guarded_gradients_match_cpu_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    assert_bool_parity(QuablaTarget::Cuda { device_ordinal: 0 });
}

// --- Pow (design slice S1b) ---

/// Bases and exponents covering the `powf` edge cases: `0^0`, `0^-1`,
/// `-0^-1`, a negative base with integer and non-integer exponents,
/// infinities, and NaN in either operand.
const POW_EDGE_BASES: [f64; 10] = [
    0.0,
    0.0,
    -0.0,
    -2.0,
    -2.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    2.0,
    1.0,
];
const POW_EDGE_EXPONENTS: [f64; 10] = [
    0.0,
    -1.0,
    -1.0,
    3.0,
    0.5,
    -0.5,
    3.0,
    0.0,
    f64::NAN,
    f64::NAN,
];

/// Equal as IEEE values, with NaN equal to NaN and signed zeros distinct.
fn same_float(actual: f64, expected: f64) -> bool {
    (actual.is_nan() && expected.is_nan())
        || (actual == expected && actual.is_sign_negative() == expected.is_sign_negative())
}

fn pow_inputs(base: &[f64], exponent: &[f64]) -> Result<BTreeMap<String, DynamicTensor>, String> {
    Ok(BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![base.len()], base.to_vec())?,
        ),
        (
            "y".to_string(),
            DynamicTensor::new(vec![exponent.len()], exponent.to_vec())?,
        ),
    ]))
}

#[test]
fn pow_follows_powf_for_edge_cases_broadcasting_and_f32() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![POW_EDGE_BASES.len()]));
    let y = must!(graph.input("y", vec![POW_EDGE_EXPONENTS.len()]));
    let output = must!(graph.pow(x, y));
    let inputs = must!(pow_inputs(&POW_EDGE_BASES, &POW_EDGE_EXPONENTS));
    let expected = [
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -8.0,
        f64::NAN,
        0.0,
        f64::NEG_INFINITY,
        1.0,
        f64::NAN,
        1.0,
    ];
    let plan = must!(graph.compile_cpu(output));
    for value in [
        must!(graph.evaluate(output, &inputs)),
        must!(plan.evaluate(&inputs)),
    ] {
        for (index, (actual, expected)) in value.data().iter().zip(expected).enumerate() {
            assert!(
                same_float(*actual, expected),
                "pow({}, {}) = {actual}, expected {expected}",
                POW_EDGE_BASES[index],
                POW_EDGE_EXPONENTS[index]
            );
            let reference = POW_EDGE_BASES[index].powf(POW_EDGE_EXPONENTS[index]);
            assert!(same_float(*actual, reference));
        }
    }

    // [2, 1] ** [3] broadcasts like the other binary ops.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let y = must!(graph.input("y", vec![3]));
    let output = must!(graph.pow(x, y));
    assert_eq!(must!(graph.node_shape(output)), vec![2, 3]);
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![2.0, 9.0])),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![3], vec![0.5, 2.0, -1.0])),
        ),
    ]);
    let value = must!(graph.evaluate(output, &inputs));
    assert_eq!(
        value.data(),
        &[2.0_f64.sqrt(), 4.0, 0.5, 3.0, 81.0, 1.0 / 9.0]
    );

    // A weak scalar exponent adopts the f32 base dtype; each f32 node is the correctly rounded
    // f64 result of its f32-rounded operands.
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![3], TensorDType::F32));
    let third = graph.scalar_constant(1.0 / 3.0);
    let output = must!(graph.pow(x, third));
    assert_eq!(must!(graph.node_dtype(output)), TensorDType::F32);
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![3], vec![0.1, 2.7, 1234.5])),
    )]);
    let value = must!(graph.evaluate(output, &inputs));
    assert_eq!(value.dtype(), TensorDType::F32);
    for (actual, base) in value.data().iter().zip([0.1_f64, 2.7, 1234.5]) {
        let expected = ((base as f32 as f64).powf((1.0_f64 / 3.0) as f32 as f64)) as f32 as f64;
        assert_eq!(*actual, expected);
    }

    // Bool operands are rejected instead of promoted to 0/1, and strong dtypes must match.
    let mut graph = TensorIr::new();
    let mask = must!(graph.input_typed("mask", vec![2], TensorDType::Bool));
    let x = must!(graph.input("x", vec![2]));
    let x32 = must!(graph.input_typed("x32", vec![2], TensorDType::F32));
    let two = graph.scalar_constant(2.0);
    for (base, exponent) in [(mask, two), (two, mask), (x, mask)] {
        let error = graph.pow(base, exponent).err().unwrap_or_default();
        assert!(error.contains("pow is not defined for bool"), "{error}");
    }
    let error = graph.pow(x, x32).err().unwrap_or_default();
    assert!(error.contains("mismatched dtypes"), "{error}");
}

/// Gradients of one route with respect to the base and the exponent.
type PowGradients = (Vec<f64>, Vec<f64>);

/// `sum(pow(x, y))` with every derivative route, for comparison with central
/// finite differences: runtime VJP and JVP, symbolic VJP, and symbolic JVP
/// with runtime tangent inputs.
fn pow_loss_derivatives(
    base: &[f64],
    exponent: &[f64],
    exponent_shape: Vec<usize>,
) -> Result<Vec<PowGradients>, String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![base.len()])?;
    let y = graph.input("y", exponent_shape.clone())?;
    let power = graph.pow(x, y)?;
    let loss = graph.sum(power)?;
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![base.len()], base.to_vec())?,
        ),
        (
            "y".to_string(),
            DynamicTensor::new(exponent_shape.clone(), exponent.to_vec())?,
        ),
    ]);
    let mut routes = Vec::new();

    let gradients = graph.vjp(loss, &inputs, DynamicTensor::filled(vec![], 1.0)?)?;
    routes.push((
        gradients["x"].data().to_vec(),
        gradients["y"].data().to_vec(),
    ));

    let symbolic = graph.symbolic_vjp(loss, "cotangent")?;
    let mut symbolic_inputs = inputs.clone();
    symbolic_inputs.insert("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?);
    routes.push((
        symbolic
            .graph
            .evaluate(symbolic.gradients["x"], &symbolic_inputs)?
            .data()
            .to_vec(),
        symbolic
            .graph
            .evaluate(symbolic.gradients["y"], &symbolic_inputs)?
            .data()
            .to_vec(),
    ));

    // Directional derivatives along one-hot directions recover each gradient entry.
    let tangent_names = BTreeMap::from([
        ("x".to_string(), "dx".to_string()),
        ("y".to_string(), "dy".to_string()),
    ]);
    let forward = graph.symbolic_jvp_with_tangent_inputs(loss, &tangent_names)?;
    let mut runtime_forward = (Vec::new(), Vec::new());
    let mut symbolic_forward = (Vec::new(), Vec::new());
    for (name, count) in [("x", base.len()), ("y", exponent.len())] {
        for index in 0..count {
            let mut directions = BTreeMap::from([
                (
                    "x".to_string(),
                    DynamicTensor::filled(vec![base.len()], 0.0)?,
                ),
                (
                    "y".to_string(),
                    DynamicTensor::filled(exponent_shape.clone(), 0.0)?,
                ),
            ]);
            let shape = directions[name].shape().to_vec();
            let mut data = vec![0.0; count];
            data[index] = 1.0;
            directions.insert(name.to_string(), DynamicTensor::new(shape, data)?);
            let (_, runtime) = graph.jvp(loss, &inputs, &directions)?;
            let mut forward_inputs = inputs.clone();
            forward_inputs.insert("dx".to_string(), directions["x"].clone());
            forward_inputs.insert("dy".to_string(), directions["y"].clone());
            let symbolic = forward.graph.evaluate(forward.tangent, &forward_inputs)?;
            let (runtime_slot, symbolic_slot) = if name == "x" {
                (&mut runtime_forward.0, &mut symbolic_forward.0)
            } else {
                (&mut runtime_forward.1, &mut symbolic_forward.1)
            };
            runtime_slot.push(runtime.data()[0]);
            symbolic_slot.push(symbolic.data()[0]);
        }
    }
    routes.push(runtime_forward);
    routes.push(symbolic_forward);
    Ok(routes)
}

fn assert_pow_gradients_match_finite_differences(
    base: &[f64],
    exponent: &[f64],
    exponent_shape: Vec<usize>,
) {
    let routes = must!(pow_loss_derivatives(base, exponent, exponent_shape.clone()));
    let loss = |base: &[f64], exponent: &[f64]| -> f64 {
        base.iter()
            .enumerate()
            .map(|(index, value)| value.powf(exponent[index % exponent.len()]))
            .sum()
    };
    let step = 1e-6;
    let difference = |perturb_base: bool, index: usize| {
        let (mut base_plus, mut exponent_plus) = (base.to_vec(), exponent.to_vec());
        let (mut base_minus, mut exponent_minus) = (base.to_vec(), exponent.to_vec());
        if perturb_base {
            base_plus[index] += step;
            base_minus[index] -= step;
        } else {
            exponent_plus[index] += step;
            exponent_minus[index] -= step;
        }
        (loss(&base_plus, &exponent_plus) - loss(&base_minus, &exponent_minus)) / (2.0 * step)
    };
    for (route, (base_gradient, exponent_gradient)) in routes.iter().enumerate() {
        for (index, actual) in base_gradient.iter().enumerate() {
            let expected = difference(true, index);
            assert!(
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "route {route} d/dx[{index}]: {actual} vs finite difference {expected}"
            );
        }
        for (index, actual) in exponent_gradient.iter().enumerate() {
            let expected = difference(false, index);
            assert!(
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "route {route} d/dy[{index}]: {actual} vs finite difference {expected}"
            );
        }
    }
}

#[test]
fn pow_gradients_in_both_operands_match_finite_differences() {
    assert_pow_gradients_match_finite_differences(
        &[0.7, 1.9, 3.2, 1.3, 0.25],
        &[-1.3, 0.4, 2.5, 1.0, 3.0],
        vec![5],
    );
    // A broadcast scalar exponent accumulates its gradient over every element.
    assert_pow_gradients_match_finite_differences(&[0.7, 1.9, 3.2], &[1.7], vec![]);
}

#[test]
fn pow_derivative_conventions_at_zero_and_negative_bases() {
    // Columns: 0^0.5 (sqrt-like, singular), 0^2, 0^1, 0^-1 (inf value), 0^0, (-2)^3, (-2)^2,
    // and (-2)^0.5 (NaN value).
    let base = [0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0];
    let exponent = [0.5, 2.0, 1.0, -1.0, 0.0, 3.0, 2.0, 0.5];
    let expected_base = [0.0, 0.0, 1.0, 0.0, 0.0, 12.0, -4.0, f64::NAN];
    // The exponent gradient is defined as 0 for every base <= 0 (JAX returns NaN for x < 0).
    let expected_exponent = [0.0; 8];
    // A forward pass with a zero base direction still multiplies it by the NaN base partial of
    // (-2)^0.5, whose value is NaN anyway.
    let mut expected_forward_exponent = expected_exponent;
    expected_forward_exponent[7] = f64::NAN;

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![8]));
    let y = must!(graph.input("y", vec![8]));
    let output = must!(graph.pow(x, y));
    let inputs = must!(pow_inputs(&base, &exponent));
    let ones = must!(DynamicTensor::filled(vec![8], 1.0));
    let zeros = must!(DynamicTensor::filled(vec![8], 0.0));

    let runtime = must!(graph.vjp(output, &inputs, ones.clone()));
    let symbolic = must!(graph.symbolic_vjp(output, "cotangent"));
    let mut symbolic_inputs = inputs.clone();
    symbolic_inputs.insert("cotangent".to_string(), ones.clone());
    let symbolic_base = must!(symbolic
        .graph
        .evaluate(symbolic.gradients["x"], &symbolic_inputs));
    let symbolic_exponent = must!(symbolic
        .graph
        .evaluate(symbolic.gradients["y"], &symbolic_inputs));
    let forward_base = must!(graph.jvp(
        output,
        &inputs,
        &BTreeMap::from([
            ("x".to_string(), ones.clone()),
            ("y".to_string(), zeros.clone())
        ])
    ))
    .1;
    let forward_exponent = must!(graph.jvp(
        output,
        &inputs,
        &BTreeMap::from([("x".to_string(), zeros), ("y".to_string(), ones)])
    ))
    .1;
    for (label, actual, expected) in [
        ("runtime VJP d/dx", runtime["x"].data(), &expected_base),
        ("runtime VJP d/dy", runtime["y"].data(), &expected_exponent),
        ("symbolic VJP d/dx", symbolic_base.data(), &expected_base),
        (
            "symbolic VJP d/dy",
            symbolic_exponent.data(),
            &expected_exponent,
        ),
        ("runtime JVP d/dx", forward_base.data(), &expected_base),
        (
            "runtime JVP d/dy",
            forward_exponent.data(),
            &expected_forward_exponent,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                same_float(*actual, *expected) || (*actual == 0.0 && *expected == 0.0),
                "{label}[{index}] at pow({}, {}): {actual}, expected {expected}",
                base[index],
                exponent[index]
            );
        }
    }

    // At the origin, pow(x, 0.5) and sqrt(x) share the zero subgradient.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![1]));
    let half = graph.scalar_constant(0.5);
    let power = must!(graph.pow(x, half));
    let root = must!(graph.sqrt(x));
    let origin = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![1], vec![0.0])),
    )]);
    let seed = must!(DynamicTensor::filled(vec![1], 1.0));
    assert_eq!(
        must!(graph.vjp(power, &origin, seed.clone()))["x"].data(),
        must!(graph.vjp(root, &origin, seed))["x"].data()
    );
}

/// Hessian of `pow(v[0], v[1])` with respect to the packed operands, by
/// symbolic forward-over-reverse (the compiled HVP route).
fn pow_symbolic_hessian(point: [f64; 2]) -> Result<Vec<Vec<f64>>, String> {
    let mut graph = TensorIr::new();
    let v = graph.input("v", vec![2])?;
    let x = graph.slice_axis(v, 0, 0, 1)?;
    let y = graph.slice_axis(v, 0, 1, 2)?;
    let power = graph.pow(x, y)?;
    let loss = graph.sum(power)?;
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let forward = reverse.graph.symbolic_jvp_with_tangent_inputs(
        reverse.gradients["v"],
        &BTreeMap::from([("v".to_string(), "dv".to_string())]),
    )?;
    let mut hessian = Vec::new();
    for column in 0..2 {
        let mut direction = vec![0.0; 2];
        direction[column] = 1.0;
        let inputs = BTreeMap::from([
            (
                "v".to_string(),
                DynamicTensor::new(vec![2], point.to_vec())?,
            ),
            ("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?),
            ("dv".to_string(), DynamicTensor::new(vec![2], direction)?),
        ]);
        hessian.push(
            forward
                .graph
                .evaluate(forward.tangent, &inputs)?
                .data()
                .to_vec(),
        );
    }
    Ok(hessian)
}

/// The same Hessian by the runtime second-order forward evaluator.
fn pow_runtime_hessian(point: [f64; 2]) -> Result<Vec<Vec<f64>>, String> {
    let mut graph = TensorIr::new();
    let v = graph.input("v", vec![2])?;
    let x = graph.slice_axis(v, 0, 0, 1)?;
    let y = graph.slice_axis(v, 0, 1, 2)?;
    let power = graph.pow(x, y)?;
    let loss = graph.sum(power)?;
    graph.hessian_scalar(
        loss,
        "v",
        &BTreeMap::from([(
            "v".to_string(),
            DynamicTensor::new(vec![2], point.to_vec())?,
        )]),
    )
}

#[test]
fn pow_hessians_agree_across_routes_and_with_the_closed_form() {
    let (x, y) = (1.7_f64, 0.6_f64);
    let log = x.ln();
    let mixed = x.powf(y - 1.0) * (1.0 + y * log);
    let expected = [
        [y * (y - 1.0) * x.powf(y - 2.0), mixed],
        [mixed, x.powf(y) * log * log],
    ];
    for hessian in [
        must!(pow_symbolic_hessian([x, y])),
        must!(pow_runtime_hessian([x, y])),
    ] {
        for (row, expected_row) in hessian.iter().zip(expected) {
            for (actual, expected) in row.iter().zip(expected_row) {
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "{actual} vs {expected}"
                );
            }
        }
    }

    // x ** 2.0 keeps the second derivative 2 at the origin (the finite limit for y >= 1), on the
    // symbolic and the runtime routes.
    for (point, expected_base_base) in [
        ([0.0, 2.0], 2.0),
        ([0.0, 1.0], 0.0),
        ([0.0, 0.5], 0.0),
        ([-2.0, 3.0], -12.0),
    ] {
        let symbolic = must!(pow_symbolic_hessian(point));
        let runtime = must!(pow_runtime_hessian(point));
        // Every entry is finite: the singular points are masked out of the reverse pass too.
        for (symbolic_row, runtime_row) in symbolic.iter().zip(&runtime) {
            for (symbolic, runtime) in symbolic_row.iter().zip(runtime_row) {
                assert!(symbolic.is_finite(), "{point:?}: {symbolic_row:?}");
                assert!(runtime.is_finite(), "{point:?}: {runtime_row:?}");
            }
        }
        assert_eq!(symbolic[0][0], expected_base_base, "{point:?}");
        assert_eq!(runtime[0][0], expected_base_base, "{point:?}");
        // The exponent-exponent entry is 0 for every base <= 0.
        assert_eq!(symbolic[1][1], 0.0, "{point:?}");
        assert_eq!(runtime[1][1], 0.0, "{point:?}");
        // symbolic[c][r] differentiates the r-th gradient entry along e_c, as runtime[r][c]
        // differentiates along e_r and then e_c, so the mixed partials pair up transposed.
        assert_eq!(symbolic[0][1], runtime[1][0], "{point:?}");
        assert_eq!(symbolic[1][0], runtime[0][1], "{point:?}");
    }
    // Where the value itself is infinite (0^-1) the runtime Hessian is the zero convention. The
    // symbolic route is NaN there because the reverse pass of `sum` broadcasts its cotangent with
    // `powi(v - v, 0)`, whose tangent `v - v` is NaN for an infinite `v`; that predates `pow`.
    assert_eq!(
        must!(pow_runtime_hessian([0.0, -1.0])),
        vec![vec![0.0, 0.0], vec![0.0, 0.0]]
    );
}

#[test]
fn pow_folds_constants_shares_cse_and_skips_constant_operand_derivatives() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let two = graph.scalar_constant(2.0);
    let three = graph.scalar_constant(3.0);
    let eight = must!(graph.pow(two, three));
    let scaled = must!(graph.mul(x, eight));
    let cube = must!(graph.pow(x, three));
    let same_cube = must!(graph.pow(x, three));
    let sum = must!(graph.add(scaled, cube));
    let output = must!(graph.add(sum, same_cube));
    let plan = must!(graph.compile_cpu(output));
    let text = plan.lower_text();
    assert!(text.contains("constant[value=8]"), "{text}");
    assert_eq!(text.matches("pow(").count(), 1, "{text}");
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2], vec![1.5, -2.0])),
    )]);
    assert_eq!(
        must!(plan.evaluate(&inputs)).data(),
        &[8.0 * 1.5 + 2.0 * 1.5_f64.powf(3.0), -16.0 - 16.0]
    );

    // A constant exponent emits no exponent derivative (no log), and a positive constant base
    // folds its logarithm.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let half = graph.scalar_constant(0.5);
    let root = must!(graph.pow(x, half));
    let two = graph.scalar_constant(2.0);
    let exponential = must!(graph.pow(two, x));
    let sum = must!(graph.add(root, exponential));
    let loss = must!(graph.sum(sum));
    let reverse = must!(graph.symbolic_vjp(loss, "cotangent"));
    let reverse_text = must!(reverse.graph.compile_cpu(reverse.gradients["x"])).lower_text();
    let forward = must!(graph.symbolic_jvp(loss, "x"));
    let forward_text = must!(forward.graph.compile_cpu(forward.tangent)).lower_text();
    for text in [&reverse_text, &forward_text] {
        assert!(!text.contains("log("), "{text}");
    }
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2], vec![0.0, 2.25])),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let gradient = must!(reverse.graph.evaluate(reverse.gradients["x"], &inputs));
    let ln2 = 2.0_f64.ln();
    // d/dx sqrt-like is 0 at the origin; d/dx 2^x = 2^x ln 2.
    let expected = [ln2, 1.0 / 3.0 + 2.0_f64.powf(2.25) * ln2];
    for (actual, expected) in gradient.data().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    }
    let tangent = must!(forward.graph.evaluate(forward.tangent, &inputs));
    assert!((tangent.data()[0] - expected.iter().sum::<f64>()).abs() < 1e-12);
}

#[test]
fn pow_fuses_into_elementwise_kernels_and_exports_stablehlo() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let bias = must!(graph.input("bias", vec![1, 3]));
    let shifted = must!(graph.add(x, bias));
    let activated = must!(graph.tanh(shifted));
    let exponent = graph.scalar_constant(1.5);
    let one = graph.scalar_constant(1.0);
    let positive = must!(graph.add(activated, one));
    let output = must!(graph.pow(positive, exponent));
    let plan = must!(graph.compile_cpu(output));
    assert!(plan.uses_fused_elementwise_kernel());
    assert_eq!(plan.fusion_regions().len(), 1);
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
    assert_eq!(
        must!(plan.evaluate(&inputs)),
        must!(graph.evaluate(output, &inputs))
    );
    let source = must!(plan.cuda_source());
    assert!(source.contains("powf("), "{source}");

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let y = must!(graph.input("y", vec![3]));
    let output = must!(graph.pow(x, y));
    let text = must!(graph.stablehlo_text(output));
    assert!(
        text.contains("%v2 = stablehlo.power %arg0, %arg1 : tensor<3xf64>"),
        "{text}"
    );
}

#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
type DeviceParityCase = (QuablaMultiOutputProgram, BTreeMap<String, DynamicTensor>);

/// f32 device programs: pow values at the edge cases; gradients in both
/// operands with a forward-over-reverse directional derivative of each; and a
/// fixed `Fori` whose loop body applies `pow` with a captured exponent, with
/// gradients through the loop.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn pow_device_programs() -> Result<Vec<DeviceParityCase>, String> {
    let base = [
        0.0,
        0.0,
        0.0,
        -2.0,
        -2.0,
        -1.5,
        1.5,
        0.3,
        f64::INFINITY,
        f64::NAN,
        2.0,
    ];
    let exponent = [
        0.0,
        -1.0,
        0.5,
        3.0,
        0.5,
        2.0,
        2.5,
        -0.7,
        -0.5,
        0.0,
        f64::NAN,
    ];
    let count = base.len();
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![count], TensorDType::F32)?;
    let y = graph.input_typed("y", vec![count], TensorDType::F32)?;
    let power = graph.pow(x, y)?;
    let values = QuablaMultiOutputProgram::new(graph.clone(), vec![power])?;
    let loss = graph.sum(power)?;
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let forward = reverse.graph.symbolic_jvp_many_with_tangent_inputs(
        &[reverse.gradients["x"], reverse.gradients["y"]],
        &BTreeMap::from([
            ("x".to_string(), "dx".to_string()),
            ("y".to_string(), "dy".to_string()),
        ]),
    )?;
    let mut outputs = forward.values.clone();
    outputs.extend(&forward.tangents);
    let derivatives = QuablaMultiOutputProgram::new(forward.graph, outputs)?;
    let mut inputs = pow_inputs(&base, &exponent)?;
    inputs.insert("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?);
    inputs.insert("dx".to_string(), DynamicTensor::filled(vec![count], 0.5)?);
    inputs.insert("dy".to_string(), DynamicTensor::filled(vec![count], -0.25)?);

    let mut body = TensorIr::new();
    let carry = body.input_typed("carry", vec![3], TensorDType::F32)?;
    let p = body.input_typed("p", vec![], TensorDType::F32)?;
    let step = body.pow(carry, p)?;
    let quarter = body.scalar_constant(0.25);
    let step = body.mul(step, quarter)?;
    let next = body.add(carry, step)?;
    let loop_plan = TensorForiExecutionPlan::new(0, 3, body.compile_cpu(next)?, "carry", "index")?;
    let mut graph = TensorIr::new();
    let initial = graph.input_typed("initial", vec![3], TensorDType::F32)?;
    let p = graph.input_typed("p", vec![], TensorDType::F32)?;
    let result = graph.fori(initial, loop_plan, vec![("p".to_string(), p)])?;
    let loss = graph.sum(result)?;
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let looped = QuablaMultiOutputProgram::new(
        reverse.graph,
        vec![
            reverse.value,
            reverse.gradients["initial"],
            reverse.gradients["p"],
        ],
    )?;
    let loop_inputs = BTreeMap::from([
        (
            "initial".to_string(),
            DynamicTensor::new(vec![3], vec![0.4, 1.1, 2.3])?,
        ),
        ("p".to_string(), DynamicTensor::new(vec![], vec![0.75])?),
        ("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?),
    ]);
    Ok(vec![
        (values, inputs.clone()),
        (derivatives, inputs),
        (looped, loop_inputs),
    ])
}

/// Device results match the CPU f32 reference within a few f32 ulps per op,
/// with NaN and infinity at the same positions.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn assert_device_parity(target: QuablaTarget, programs: Result<Vec<DeviceParityCase>, String>) {
    let compiler = QuablaCompiler;
    for (index, (program, inputs)) in must!(programs).into_iter().enumerate() {
        let cpu = must!(must!(compiler.compile_many(&program, QuablaTarget::Cpu)).execute(&inputs));
        let device = must!(must!(compiler.compile_many(&program, target)).execute(&inputs));
        assert_eq!(device.len(), cpu.len());
        for (output, (actual, expected)) in device.iter().zip(&cpu).enumerate() {
            assert_eq!(actual.shape(), expected.shape());
            for (element, (actual, expected)) in
                actual.data().iter().zip(expected.data()).enumerate()
            {
                let close = if expected.is_finite() {
                    (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)
                } else {
                    same_float(*actual, *expected)
                };
                assert!(
                    close,
                    "{target:?} program {index} output {output}[{element}]: {actual} vs CPU {expected}"
                );
            }
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_pow_values_derivatives_and_loop_bodies_match_cpu() {
    assert_device_parity(QuablaTarget::Mlx, pow_device_programs());
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_pow_values_derivatives_and_loop_bodies_match_cpu_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    assert_device_parity(
        QuablaTarget::Cuda { device_ordinal: 0 },
        pow_device_programs(),
    );
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_fuses_pow_chains_into_one_region_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2, 1]));
    let y = must!(graph.input("y", vec![1, 3]));
    let shifted = must!(graph.tanh(x));
    let one = graph.scalar_constant(1.0);
    let positive = must!(graph.add(shifted, one));
    let power = must!(graph.pow(positive, y));
    let output = must!(graph.sin(power));
    let plan = must!(graph.compile_cpu(output));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![2, 1], vec![-1.0, 2.0])),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![1, 3], vec![0.5, -1.5, 2.25])),
        ),
    ]);
    let cpu = must!(plan.evaluate(&inputs));
    let compiled = must!(CudaBackend::new(0).compile(plan));
    assert!(compiled.fused_region_count() >= 1);
    let cuda = must!(compiled.execute(&inputs));
    for (actual, expected) in cuda.data().iter().zip(cpu.data()) {
        assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
    }
}

/// `sqrt` inputs across its domain edges: negative (finite and infinite),
/// signed zeros, positive (finite and infinite), and NaN.
const SQRT_EDGE_INPUTS: [f64; 9] = [
    -4.0,
    -1.0e-3,
    f64::NEG_INFINITY,
    -0.0,
    0.0,
    0.25,
    4.0,
    f64::INFINITY,
    f64::NAN,
];

/// Value, gradient, and forward-over-reverse directional derivative of an
/// elementwise `sqrt`, as outputs `[sqrt, d sqrt, jvp(sqrt), jvp(d sqrt)]`.
/// The cotangent is a vector, so no reduction takes part and the outputs are
/// `sqrt` and its `SqrtDerivative` orders 1 and 2 scaled by the unit seeds.
fn sqrt_derivative_program(
    dtype: TensorDType,
) -> Result<(QuablaMultiOutputProgram, BTreeMap<String, DynamicTensor>), String> {
    let count = SQRT_EDGE_INPUTS.len();
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![count], dtype)?;
    let root = graph.sqrt(x)?;
    let reverse = graph.symbolic_vjp(root, "cotangent")?;
    let forward = reverse.graph.symbolic_jvp_many_with_tangent_inputs(
        &[reverse.value, reverse.gradients["x"]],
        &BTreeMap::from([("x".to_string(), "dx".to_string())]),
    )?;
    let mut outputs = forward.values.clone();
    outputs.extend(&forward.tangents);
    let program = QuablaMultiOutputProgram::new(forward.graph, outputs)?;
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![count], SQRT_EDGE_INPUTS.to_vec())?,
        ),
        (
            "cotangent".to_string(),
            DynamicTensor::filled(vec![count], 1.0)?,
        ),
        ("dx".to_string(), DynamicTensor::filled(vec![count], 1.0)?),
    ]);
    Ok((program, inputs))
}

#[test]
fn sqrt_derivatives_are_nan_below_zero_and_zero_at_the_origin() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    // Columns follow SQRT_EDGE_INPUTS. Every negative input, -inf included, is outside the
    // domain (NaN, as IEEE sqrt); every derivative order at +-0 is the zero subgradient.
    let expected_value = [nan, nan, nan, 0.0, 0.0, 0.5, 2.0, inf, nan];
    let expected_first = [nan, nan, nan, 0.0, 0.0, 1.0, 0.25, 0.0, nan];
    let expected_second = [nan, nan, nan, 0.0, 0.0, -2.0, -0.03125, -0.0, nan];
    let expected = [
        expected_value,
        expected_first,
        expected_first,
        expected_second,
    ];
    let (program, inputs) = must!(sqrt_derivative_program(TensorDType::F64));
    let compiled =
        must!(must!(QuablaCompiler.compile_many(&program, QuablaTarget::Cpu)).execute(&inputs));
    // Single-output CPU plans take the fused elementwise kernel where the chain allows it.
    let single = program
        .output_node_ids()
        .iter()
        .map(|output| CpuBackend.execute(&program.ir().compile_cpu(*output)?, &inputs))
        .collect::<Result<Vec<_>, _>>();
    let single = must!(single);
    let interpreted = program
        .output_node_ids()
        .iter()
        .map(|output| program.ir().evaluate(*output, &inputs))
        .collect::<Result<Vec<_>, _>>();
    let interpreted = must!(interpreted);
    for (route, results) in [
        ("compiled", &compiled),
        ("single-output", &single),
        ("interpreted", &interpreted),
    ] {
        for (output, (actual, expected)) in results.iter().zip(&expected).enumerate() {
            for (element, (actual, expected)) in actual.data().iter().zip(expected).enumerate() {
                // Zeros compare by value: IEEE sqrt(-0) is -0, the zero convention returns +0.
                assert!(
                    (*actual == 0.0 && *expected == 0.0) || same_float(*actual, *expected),
                    "{route} output {output}[{element}] at x = {}: {actual} vs {expected}",
                    SQRT_EDGE_INPUTS[element]
                );
            }
        }
    }
}

/// f32 device programs for `sqrt`: the edge-case values with derivative
/// orders 1 and 2, and a fixed `Fori` whose loop body applies `sqrt`, with
/// gradients through the loop. The loop starts at a negative, a zero, a
/// positive, an infinite, and a NaN carry.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn sqrt_device_programs() -> Result<Vec<DeviceParityCase>, String> {
    let derivatives = sqrt_derivative_program(TensorDType::F32)?;

    let mut body = TensorIr::new();
    let carry = body.input_typed("carry", vec![5], TensorDType::F32)?;
    let root = body.sqrt(carry)?;
    let quarter = body.scalar_constant(0.25);
    let step = body.mul(root, quarter)?;
    let next = body.add(carry, step)?;
    let loop_plan = TensorForiExecutionPlan::new(0, 3, body.compile_cpu(next)?, "carry", "index")?;
    let mut graph = TensorIr::new();
    let initial = graph.input_typed("initial", vec![5], TensorDType::F32)?;
    let result = graph.fori(initial, loop_plan, vec![])?;
    let loss = graph.sum(result)?;
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let looped = QuablaMultiOutputProgram::new(
        reverse.graph,
        vec![reverse.value, reverse.gradients["initial"]],
    )?;
    let loop_inputs = BTreeMap::from([
        (
            "initial".to_string(),
            DynamicTensor::new(vec![5], vec![-1.0, 0.0, 2.25, f64::INFINITY, f64::NAN])?,
        ),
        ("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?),
    ]);
    Ok(vec![derivatives, (looped, loop_inputs)])
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_sqrt_values_derivatives_and_loop_bodies_match_cpu() {
    assert_device_parity(QuablaTarget::Mlx, sqrt_device_programs());
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_sqrt_values_derivatives_and_loop_bodies_match_cpu_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    assert_device_parity(
        QuablaTarget::Cuda { device_ordinal: 0 },
        sqrt_device_programs(),
    );
}
