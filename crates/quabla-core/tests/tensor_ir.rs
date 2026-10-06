use std::collections::{BTreeMap, BTreeSet};

use quabla_core::compiler::QuablaCompileError;
use quabla_core::tensor_ir::{
    CpuBackend, DynamicTensor, SymbolicCotangent, TensorBackend, TensorBufferSlot,
    TensorCondExecutionPlan, TensorCustomRule, TensorDType, TensorDeviceBackend, TensorDeviceId,
    TensorDeviceMesh, TensorExecutionPlan, TensorForiExecutionPlan, TensorForiMultiExecutionPlan,
    TensorForiVjpJvpExecutionPlan, TensorFusionRegion, TensorIr, TensorNodeId, TensorPartitionSpec,
    TensorPlacement, TensorReplicaReduction, TensorScanExecutionPlan, TensorShardingPlan,
    UnaryMathKind,
};
use quabla_core::{QuablaCompiler, QuablaMultiOutputProgram, QuablaPrecision, QuablaTarget};

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
fn mlx_lowering_validation_rejects_non_finite_constants_in_inactive_regions_without_execution() {
    let mut branch = TensorIr::new();
    let a = must!(branch.input("a", vec![1, 1]));
    let b = must!(branch.input("b", vec![1, 1]));
    let infinity = branch.scalar_constant(f64::INFINITY);
    let shifted = must!(branch.add(a, infinity));
    let scaled = must!(branch.mul(shifted, b));
    let bad = must!(branch.compile_cpu(scaled));
    assert_eq!(
        bad.validate_mlx().map_err(|(op, _)| op),
        Err("constant".into())
    );
    let mut good = TensorIr::new();
    let a = must!(good.input("a", vec![1, 1]));
    let b = must!(good.input("b", vec![1, 1]));
    let sum = must!(good.add(a, b));
    let good = must!(good.compile_cpu(sum));
    let branches = must!(TensorCondExecutionPlan::new(good, bad));
    let mut graph = TensorIr::new();
    let predicate = graph.scalar_constant(1.0);
    let a = must!(graph.input("a", vec![1, 1]));
    let b = must!(graph.input("b", vec![1, 1]));
    let output = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("a".into(), a), ("b".into(), b)]
    ));
    let plan = must!(graph.compile_cpu(output));
    assert_eq!(
        plan.validate_mlx().map_err(|(op, _)| op),
        Err("constant".into())
    );
    // The unsupported region remains valid on CPU and the chosen branch is lazy.
    let inputs = BTreeMap::from([
        ("a".into(), must!(DynamicTensor::new(vec![1, 1], vec![2.0]))),
        ("b".into(), must!(DynamicTensor::new(vec![1, 1], vec![3.0]))),
    ]);
    assert_eq!(must!(plan.evaluate(&inputs)).data().as_ref(), &[5.0]);
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
fn compiler_float64_precision_is_a_cpu_no_op_and_an_mlx_error() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let tenth = graph.scalar_constant(0.1);
    let scaled = must!(graph.mul(x, tenth));
    let program = must!(QuablaMultiOutputProgram::new(graph, vec![scaled]));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2], vec![1.0, 3.0])),
    )]);
    let compiler = QuablaCompiler;

    let cpu = must!(compiler
        .compile_many_checked_with_precision(&program, QuablaTarget::Cpu, QuablaPrecision::Float64)
        .map_err(|error| format!("{error:?}")));
    assert_eq!(
        must!(cpu.execute(&inputs))[0].data().to_vec(),
        vec![0.1, 0.30000000000000004]
    );
    // MLX has no f64 arithmetic, whether or not this build includes it.
    let error = compiler
        .compile_many_checked_with_precision(&program, QuablaTarget::Mlx, QuablaPrecision::Float64)
        .expect_err("MLX must reject float64 precision");
    assert!(
        matches!(&error, QuablaCompileError::Unsupported { op, .. } if op == "float64"),
        "{error:?}"
    );
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
                .zip(expected.data().iter())
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
        assert_eq!(actual.data().as_ref(), expected.data().as_ref());
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
        value.data().as_ref(),
        &[
            11.0, 22.0, 33.0, 21.0, 42.0, 63.0, 31.0, 62.0, 93.0, 41.0, 82.0, 123.0, 44.0, 55.0,
            66.0, 84.0, 105.0, 126.0, 124.0, 155.0, 186.0, 164.0, 205.0, 246.0
        ]
    );

    let cotangent = must!(DynamicTensor::filled(vec![2, 4, 3], 1.0));
    let gradients = must!(graph.vjp(output, &inputs, cotangent));
    assert_eq!(gradients["x"].data().as_ref(), &[104.0; 6]);
    assert_eq!(gradients["y"].data().as_ref(), &[21.0; 4]);
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
        jvp_tangent.data().as_ref(),
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
    assert_eq!(value.data().as_ref(), &[30.0]);

    let gradients = must!(graph.vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert_eq!(gradients["x"].data().as_ref(), &[2.0, 4.0, 6.0, 8.0]);

    let tangents = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::filled(vec![2, 2], 1.0)),
    )]);
    let (_, tangent) = must!(graph.jvp(loss, &inputs, &tangents));
    assert_eq!(tangent.shape(), &[] as &[usize]);
    assert_eq!(tangent.data().as_ref(), &[20.0]);
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
        must!(graph.evaluate(selected, &inputs)).data().as_ref(),
        &[-6.0, -3.0, 0.0, 4.0]
    );
    let gradients = must!(graph.vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 1.0)),));
    assert_eq!(gradients["x"].data().as_ref(), &[3.0, 3.0, 3.0, 4.0]);
    let (_, tangent) = must!(graph.jvp(
        loss,
        &inputs,
        &BTreeMap::from([("x".to_string(), must!(DynamicTensor::filled(vec![4], 1.0)))]),
    ));
    assert_eq!(tangent.data().as_ref(), &[13.0]);

    let transformed = must!(graph.symbolic_jvp(loss, "x"));
    assert_eq!(
        must!(transformed.graph.evaluate(transformed.tangent, &inputs))
            .data()
            .as_ref(),
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
    assert_eq!(must!(plan.evaluate(&true_inputs)).data().as_ref(), &[16.0]);
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
    assert_eq!(tangent.data().as_ref(), &[16.0]);
    let gradients = must!(plan.vjp(&true_inputs, must!(DynamicTensor::new(vec![], vec![1.0]))));
    assert_eq!(gradients["x"].data().as_ref(), &[16.0]);
    assert_eq!(gradients["predicate"].data().as_ref(), &[0.0]);

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
    assert_eq!(must!(plan.evaluate(&false_inputs)).data().as_ref(), &[12.0]);
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
    assert_eq!(
        must!(graph.evaluate(output, &inputs)).data().as_ref(),
        &[2.0]
    );
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
            must!(transformed.graph.evaluate(transformed.value, &inputs))
                .data()
                .as_ref(),
            &[expected_value]
        );
        assert_eq!(
            must!(transformed.graph.evaluate(transformed.tangent, &inputs))
                .data()
                .as_ref(),
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
            must!(transformed.graph.evaluate(transformed.value, &inputs))
                .data()
                .as_ref(),
            &[expected_value]
        );
        assert_eq!(
            must!(transformed
                .graph
                .evaluate(transformed.gradients["x"], &inputs))
            .data()
            .as_ref(),
            &[expected_gradient]
        );
        assert_eq!(
            must!(transformed
                .graph
                .evaluate(transformed.gradients["predicate"], &inputs))
            .data()
            .as_ref(),
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
            .data()
            .as_ref(),
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
        must!(loop_plan.evaluate(must!(DynamicTensor::new(vec![], vec![1.0])), &external))
            .data()
            .as_ref(),
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
    assert_eq!(tangent.data().as_ref(), &[8.0]);
    let (value, initial_gradient, external_gradients) = must!(loop_plan.value_and_vjp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &external,
        must!(DynamicTensor::new(vec![], vec![1.0])),
    ));
    assert_eq!(value.data().as_ref(), &[12.0]);
    assert_eq!(initial_gradient.data().as_ref(), &[8.0]);
    assert_eq!(external_gradients["scale"].data().as_ref(), &[13.0]);

    let (_, tape) = must!(
        loop_plan.evaluate_with_tape(must!(DynamicTensor::new(vec![], vec![1.0])), &external,)
    );
    assert_eq!(tape.len(), 4);
    assert_eq!(must!(tape.carry_at(0)).data().as_ref(), &[1.0]);
    assert_eq!(must!(tape.carry_at(1)).data().as_ref(), &[2.0]);
    assert_eq!(must!(tape.carry_at(2)).data().as_ref(), &[5.0]);
    assert_eq!(must!(tape.carry_at(3)).data().as_ref(), &[12.0]);
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
        .data()
        .as_ref(),
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
    assert_eq!(carries[0].data().as_ref(), &[2.0]);
    assert_eq!(carries[1].data().as_ref(), &[18.0]);

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
    assert_eq!(initial_gradients[0].data().as_ref(), &[1.0]);
    assert_eq!(initial_gradients[1].data().as_ref(), &[9.0]);
    assert_eq!(external_gradients["scale"].data().as_ref(), &[12.0]);
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
    assert_eq!(carry.data().as_ref(), &[12.0]);
    assert_eq!(outputs.shape(), &[3]);
    assert_eq!(outputs.data().as_ref(), &[2.0, 5.0, 12.0]);
    let (_, _, tape) = must!(scan.evaluate_with_tape(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
    ));
    assert_eq!(tape.len(), 4);
    assert_eq!(must!(tape.carry_at(2)).data().as_ref(), &[5.0]);
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
    assert_eq!(carry.data().as_ref(), &[12.0]);
    assert_eq!(outputs.data().as_ref(), &[2.0, 5.0, 12.0]);
    assert_eq!(carry_tangent.data().as_ref(), &[13.0]);
    assert_eq!(output_tangents.data().as_ref(), &[1.0, 4.0, 13.0]);
    let (_, _, initial_gradient, gradients) = must!(scan.value_and_vjp(
        must!(DynamicTensor::new(vec![], vec![1.0])),
        &BTreeMap::from([(
            "scale".to_string(),
            must!(DynamicTensor::new(vec![], vec![2.0])),
        )]),
        must!(DynamicTensor::new(vec![], vec![0.0])),
        must!(DynamicTensor::new(vec![3], vec![1.0, 1.0, 1.0])),
    ));
    assert_eq!(initial_gradient.data().as_ref(), &[14.0]);
    assert_eq!(gradients["scale"].data().as_ref(), &[18.0]);
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

    assert_eq!(
        must!(graph.evaluate(final_carry, &inputs)).data().as_ref(),
        &[12.0]
    );
    assert_eq!(
        must!(graph.evaluate(outputs, &inputs)).data().as_ref(),
        &[2.0, 5.0, 12.0]
    );
    assert_eq!(
        must!(graph.evaluate(total, &inputs)).data().as_ref(),
        &[31.0]
    );
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
    assert_eq!(tangent.data().as_ref(), &[31.0]);
    let (_, gradients) =
        must!(graph.value_and_vjp(total, &inputs, must!(DynamicTensor::new(vec![], vec![1.0])),));
    assert_eq!(gradients["initial"].data().as_ref(), &[22.0]);
    assert_eq!(gradients["scale"].data().as_ref(), &[31.0]);

    let (_, final_carry_gradients) = must!(must!(graph.compile_cpu(final_carry))
        .value_and_vjp(&inputs, must!(DynamicTensor::new(vec![], vec![1.0])),));
    assert_eq!(final_carry_gradients["initial"].data().as_ref(), &[8.0]);
    assert_eq!(final_carry_gradients["scale"].data().as_ref(), &[13.0]);
    let (_, output_gradients) = must!(must!(graph.compile_cpu(outputs)).value_and_vjp(
        &inputs,
        must!(DynamicTensor::new(vec![3], vec![1.0, 1.0, 1.0])),
    ));
    assert_eq!(output_gradients["initial"].data().as_ref(), &[14.0]);
    assert_eq!(output_gradients["scale"].data().as_ref(), &[18.0]);
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
        .data()
        .as_ref(),
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
        .data()
        .as_ref(),
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
        .data()
        .as_ref(),
        &[22.0]
    );
    assert_eq!(
        must!(symbolic
            .graph
            .compile_cpu(symbolic.gradients["scale"])
            .and_then(|plan| plan.evaluate(&inputs)))
        .data()
        .as_ref(),
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
    assert_eq!(outputs[0].data().as_ref(), &[29.0]);
    assert_eq!(outputs[1].data().as_ref(), &[26.0]);

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if std::env::var_os("QUABLA_CUDA_TEST").is_some() {
        let cuda = must!(CudaBackend::default().execute_many(&plan, &output_ids, &inputs));
        for (actual, expected) in cuda.iter().zip(&outputs) {
            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
    assert_eq!(value.data().as_ref(), &[4.0]);
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
    assert_eq!(carry.data().as_ref(), &[7.0]);
    assert_eq!(outputs.data().as_ref(), &[3.0, 5.0, 7.0]);
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
    assert_eq!(tangents["carry"].data().as_ref(), &[12.0]);
    assert_eq!(tangents["scale"].data().as_ref(), &[12.0]);
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

    assert_eq!(
        must!(graph.evaluate(output, &inputs)).data().as_ref(),
        &[12.0]
    );
    assert!(graph.lower_text().contains("fori(carry="));
    assert_eq!(
        must!(must!(graph.compile_cpu(output)).evaluate(&inputs))
            .data()
            .as_ref(),
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
    assert_eq!(tangent.data().as_ref(), &[8.0]);
    let (_, gradients) = must!(graph.value_and_vjp(
        output,
        &inputs,
        must!(DynamicTensor::new(vec![], vec![1.0])),
    ));
    assert_eq!(gradients["initial"].data().as_ref(), &[8.0]);
    assert_eq!(gradients["scale"].data().as_ref(), &[13.0]);
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
    assert_eq!(tangent.data().as_ref(), &[8.0]);

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
    assert_eq!(tangent.data().as_ref(), &[13.0]);
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
    assert_eq!(initial_gradient.data().as_ref(), &[8.0]);
    assert_eq!(scale_gradient.data().as_ref(), &[13.0]);
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
        .data()
        .as_ref(),
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
    assert_eq!(value.data().as_ref(), &[12.0]);
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
        assert_eq!(
            must!(compiled.execute(&inputs)).data().as_ref(),
            &[expected]
        );
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
fn cuda_backend_runs_cond_inside_fori_and_scan_bodies_as_host_driven_loops_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
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
    let index = must!(body.input("index", vec![]));
    let predicate = must!(body.sub(carry, index));
    let selected = must!(body.cond(predicate, branches));
    let body_output = must!(body.add(selected, index));

    // The fused per-lane kernel cannot lower `Cond`, so both loops run their
    // body region (which owns the nested Cond regions) once per iteration.
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        4,
        must!(body.compile_cpu(body_output)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![]));
    let output = must!(graph.fori(initial, loop_plan, Vec::new()));
    let scan_plan = must!(TensorScanExecutionPlan::new(
        0,
        4,
        must!(body.compile_cpu_many(&[body_output, body_output])).0,
        "carry",
        "index",
    ));
    let (scan_carry, scan_outputs) = must!(graph.scan(initial, scan_plan, Vec::new()));
    let sources = vec![output, scan_carry, scan_outputs];
    let (plan, output_ids) = must!(graph.compile_cpu_many(&sources));
    for initial in [1.5, 0.0] {
        let inputs = BTreeMap::from([(
            "initial".to_string(),
            must!(DynamicTensor::new(vec![], vec![initial])),
        )]);
        let mut cpu = Vec::new();
        for source in &sources {
            cpu.push(must!(must!(graph.compile_cpu(*source)).evaluate(&inputs)));
        }
        let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
        assert_cuda_outputs_match_cpu("Cond inside loop bodies", &cuda, &cpu, 1e-6);
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_while_loops_as_host_driven_region_loops_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    let (graph, output, inputs) = must!(doubling_while_graph());
    let plan = must!(graph.compile_cpu(output));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_eq!(cuda.data().as_ref(), &[16.0]);
    // Forward mode runs the same While node over a packed (primal, tangent) carry.
    let transformed = must!(graph.symbolic_jvp_with_tangent_inputs(
        output,
        &BTreeMap::from([("scale".to_string(), "scale_tangent".to_string())]),
    ));
    let plan = must!(transformed.graph.compile_cpu(transformed.tangent));
    let mut inputs = inputs;
    inputs.insert(
        "scale_tangent".to_string(),
        must!(DynamicTensor::new(vec![], vec![1.0])),
    );
    let cpu = must!(plan.evaluate(&inputs));
    let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
    assert_cuda_outputs_match_cpu("While JVP", &[cuda], &[cpu], 1e-6);
}

/// `x <- x + (dt * k) * x` plus a slice/concat rotation: the scalar captures'
/// gradients are reduced inside the body VJP (`dt * sum(...)`), which the fused
/// kernel cannot express, and the rotation is not elementwise.
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(clippy::type_complexity)]
fn host_driven_fori_loss(
    rotate: bool,
) -> Result<(TensorIr, TensorNodeId, BTreeMap<String, DynamicTensor>), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![3])?;
    body.input("index", vec![])?;
    let rate = body.input("rate", vec![])?;
    let step = body.input("step", vec![])?;
    let factor = body.mul(step, rate)?;
    let increment = body.mul(factor, carry)?;
    let mut next = body.add(carry, increment)?;
    if rotate {
        let head = body.slice_axis(next, 0, 0, 1)?;
        let tail = body.slice_axis(next, 0, 1, 3)?;
        next = body.concat(vec![tail, head], 0)?;
    }
    let loop_plan = TensorForiExecutionPlan::new(0, 40, body.compile_cpu(next)?, "carry", "index")?;
    let mut graph = TensorIr::new();
    let initial = graph.input("initial", vec![3])?;
    let rate = graph.input("rate", vec![])?;
    let step = graph.input("step", vec![])?;
    let output = graph.fori(
        initial,
        loop_plan,
        vec![("rate".to_string(), rate), ("step".to_string(), step)],
    )?;
    let squared = graph.mul(output, output)?;
    let loss = graph.sum(squared)?;
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            DynamicTensor::new(vec![3], vec![0.5, -1.0, 2.0])?,
        ),
        ("rate".to_string(), DynamicTensor::new(vec![], vec![-0.7])?),
        ("step".to_string(), DynamicTensor::new(vec![], vec![0.05])?),
        ("seed".to_string(), DynamicTensor::new(vec![], vec![1.0])?),
        (
            "initial_tangent".to_string(),
            DynamicTensor::new(vec![3], vec![0.0; 3])?,
        ),
        (
            "rate_tangent".to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        ),
        (
            "step_tangent".to_string(),
            DynamicTensor::new(vec![], vec![0.0])?,
        ),
    ]);
    Ok((graph, loss, inputs))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_matches_cpu_for_host_driven_fori_vjp_and_hvp_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    for rotate in [false, true] {
        let (graph, loss, inputs) = must!(host_driven_fori_loss(rotate));
        let transformed = must!(graph.symbolic_vjp(loss, "seed"));
        let gradients = ["initial", "rate", "step"]
            .iter()
            .map(|name| transformed.gradients[*name])
            .collect::<Vec<_>>();
        let (plan, output_ids) = must!(transformed.graph.compile_cpu_many(&gradients));
        let mut cpu = Vec::new();
        for gradient in &gradients {
            cpu.push(must!(
                must!(transformed.graph.compile_cpu(*gradient)).evaluate(&inputs)
            ));
        }
        let cuda = must!(CudaBackend::new(0).execute_many(&plan, &output_ids, &inputs));
        assert_cuda_outputs_match_cpu("host-driven Fori VJP", &cuda, &cpu, 1e-4);

        // Forward-over-reverse: the directional derivative of the rate gradient.
        let tangents = BTreeMap::from([
            ("initial".to_string(), "initial_tangent".to_string()),
            ("rate".to_string(), "rate_tangent".to_string()),
            ("step".to_string(), "step_tangent".to_string()),
        ]);
        let hvp = must!(transformed
            .graph
            .symbolic_jvp_with_tangent_inputs(transformed.gradients["rate"], &tangents));
        let plan = must!(hvp.graph.compile_cpu(hvp.tangent));
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_cuda_outputs_match_cpu("host-driven Fori HVP", &[cuda], &[cpu], 1e-4);
    }
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
        for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
        for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
                .zip(cpu[index + 1].data().iter())
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
            for (actual, expected) in device.data().iter().zip(cpu.data().iter()) {
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
    if gradient_x.data().as_ref() != [0.0] || gradient_y.data().as_ref() != [2.0, 4.0, 6.0] {
        return Err(format!(
            "{backend} inactive-branch gradients leaked: {:?} {:?}",
            gradient_x.data().as_ref(),
            gradient_y.data().as_ref()
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
    assert_eq!(cpu_mask.data().as_ref(), &[0.0, 0.0, 1.0]);
    assert_eq!(cpu_count.data().as_ref(), &[2.0]);
    assert_eq!(
        must!(MlxBackend.execute(&mask_plan, &inputs))
            .data()
            .as_ref(),
        cpu_mask.data().as_ref()
    );
    let (plan, outputs) = must!(graph.compile_cpu_many(&[mask, count]));
    let mlx = must!(MlxBackend.execute_many(&plan, &outputs, &inputs));
    assert_eq!(mlx[0].data().as_ref(), cpu_mask.data().as_ref());
    assert_eq!(mlx[1].data().as_ref(), cpu_count.data().as_ref());

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
            must!(CpuBackend.execute(&plan, &inputs)).data().as_ref(),
            &[expected]
        );
        assert_eq!(
            must!(MlxBackend.execute(&plan, &inputs)).data().as_ref(),
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
            must!(MlxBackend.execute(&plan, &inputs)).data().as_ref(),
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
        for (actual, expected) in symbolic.data().iter().zip(direct[name].data().iter()) {
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
    assert_eq!(value.data().as_ref(), &[2.0, 4.0, 6.0, 8.0]);
    let (value, gradients) =
        must!(plan.value_and_vjp(&inputs, must!(DynamicTensor::filled(vec![2, 2], 1.0)),));
    assert_eq!(value.data().as_ref(), &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(gradients["x"].data().as_ref(), &[2.0, 2.0, 2.0, 2.0]);
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
        for (actual, reference) in reduced.iter().zip(expected[position].data().iter()) {
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
            .zip(reference.data().iter())
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "sharded-schedule output={position} cuda={:?} cpu={:?} max_abs_error={max_error:e}",
            actual.data().as_ref(),
            reference.data().as_ref()
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
            .zip(reference.data().iter())
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
    assert_eq!(
        must!(CpuBackend.execute(&plan, &inputs)).data().as_ref(),
        &[20.0]
    );
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
        must!(CpuBackend.execute(&plan, &inputs)).data().as_ref(),
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
            for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
        for (actual, expected) in cuda.data().iter().zip(cpu[name].data().iter()) {
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
        for (actual, expected) in cuda.data().iter().zip(direct[name].data().iter()) {
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
    assert_eq!(value.data().as_ref(), &[2.5, 3.5, 4.5]);

    let gradients = must!(graph.vjp(
        output,
        &inputs,
        must!(DynamicTensor::new(vec![3], vec![2.0, 3.0, 4.0])),
    ));
    assert_eq!(
        gradients["x"].data().as_ref(),
        &[1.0, 1.5, 2.0, 1.0, 1.5, 2.0]
    );
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
    assert_eq!(value.data().as_ref(), &[8.0, -15.0]);
    assert_eq!(tangent.data().as_ref(), &[5.0, 10.75]);
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
    assert_eq!(tangent.data().as_ref(), &[1.0]);
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
    assert_eq!(tangent.data().as_ref(), &[0.5]);
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
    assert_eq!(tangent.data().as_ref(), &[0.2]);
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
    assert_eq!(value.data().as_ref(), &[2.0, 3.0]);

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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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
    for (actual, expected) in mlx.data().iter().zip(cpu.data().iter()) {
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
        assert_eq!(
            actual.data().as_ref(),
            expected.data().as_ref(),
            "output {index} data"
        );
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
        must!(MlxBackend.execute(&plan, &inputs)).data().as_ref(),
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
        must!(MlxBackend.execute(&plan, &inputs)).data().as_ref(),
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
    assert_eq!(mlx[0].data().as_ref(), cpu[0].data().as_ref());
    for (actual, expected) in mlx[1].data().iter().zip(cpu[1].data().iter()) {
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
fn mlx_backend_solves_through_the_cpu_stream_lu() {
    if std::env::var_os("QUABLA_MLX_TEST").is_none() {
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
    let solution = must!(MlxBackend.execute(&plan, &inputs));
    assert_eq!(solution.data().as_ref(), &[2.0, 3.0]);
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
    assert_eq!(value.data().as_ref(), &[0.10000000149011612]);
    assert_eq!(plan.node_count(), 3);

    // A lossy round trip of a constant must not be cancelled during folding either.
    let mut graph = TensorIr::new();
    let tenth = graph.scalar_constant(0.1);
    let single = must!(graph.cast(tenth, TensorDType::F32));
    let double = must!(graph.cast(single, TensorDType::F64));
    let plan = must!(graph.compile_cpu(double));
    assert_eq!(plan.node_count(), 1);
    assert_eq!(
        must!(CpuBackend.execute(&plan, &BTreeMap::new()))
            .data()
            .as_ref(),
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
        assert_eq!(value.data().as_ref(), &[f64::from(0.1_f32), 1.0]);
    }
    let rounded = must!(DynamicTensor::with_dtype(
        vec![1],
        vec![0.1],
        TensorDType::F32
    ));
    assert_eq!(rounded.data().as_ref(), &[f64::from(0.1_f32)]);
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
    assert_eq!(value.data().as_ref(), &[f64::from(expected)]);

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
    assert_eq!(
        must!(graph.evaluate(output, &inputs)).data().as_ref(),
        &[1.0]
    );
    let vjp = must!(graph.symbolic_vjp(output, "seed"));
    assert_eq!(
        must!(vjp.graph.evaluate(vjp.gradients["x"], &inputs))
            .data()
            .as_ref(),
        &[1.0, 0.0, 0.0]
    );
    let jvp = must!(graph.symbolic_jvp(output, "x"));
    assert_eq!(
        must!(jvp.graph.evaluate(jvp.value, &inputs))
            .data()
            .as_ref(),
        &[1.0]
    );
    assert_eq!(
        must!(jvp.graph.evaluate(jvp.tangent, &inputs))
            .data()
            .as_ref(),
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
    assert_eq!(tensor.data().as_ref(), &[0.0, 1.0, 1.0, 1.0]);
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
            assert_eq!(value.data().as_ref(), expected.as_slice(), "output {index}");
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
        must!(CpuBackend.execute(&plan, &BTreeMap::new()))
            .data()
            .as_ref(),
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
    assert_eq!(
        must!(graph.evaluate(truthy, &inputs)).data().as_ref(),
        &[1.0, 0.0]
    );
    let inputs = f32_inputs(&[
        ("x", vec![2], vec![3.0, -1.0]),
        ("r", vec![2], vec![1.0, 1.0]),
    ]);
    assert_eq!(
        must!(graph.evaluate(numeric, &inputs)).data().as_ref(),
        &[1.0, 0.0]
    );
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
        must!(graph.evaluate(bool_abs, &inputs)).data().as_ref(),
        &[2.0, 0.0, 3.0]
    );
    assert_eq!(
        must!(graph.evaluate(float_abs, &inputs)).data().as_ref(),
        &[2.0, -0.0, 3.0]
    );

    // all(isfinite(x)) as a Cond predicate; the false branch guards non-finite values itself.
    let (graph, output) = must!(finite_guard_cond_graph());
    let finite = f32_inputs(&[("x", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(
        must!(graph.evaluate(output, &finite)).data().as_ref(),
        &[91.0]
    );
    let guarded = f32_inputs(&[("x", vec![2, 3], BOOL_X.to_vec())]);
    assert_eq!(
        must!(graph.evaluate(output, &guarded)).data().as_ref(),
        &[4.5]
    );

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
    assert_close(value.data().as_ref(), &[MASKED_LOSS], 1e-12);
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
    assert_eq!(gradient.data().as_ref(), &[1.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    let tangent = must!(cond_graph.symbolic_jvp(cond_output, "x"));
    assert_eq!(
        must!(tangent.graph.evaluate(tangent.tangent, &inputs))
            .data()
            .as_ref(),
        &[5.0]
    );
    let plan = must!(cond_graph.compile_cpu(cond_output));
    let eager = must!(plan.vjp(&inputs, seed.clone()));
    assert_eq!(eager["x"].data().as_ref(), &[1.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
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
    assert_eq!(value.data().as_ref(), &[4.0]);
    assert_eq!(gradients["x"].data().as_ref(), &[0.0, 4.0, 0.0]);
    assert!(!gradients.contains_key("mask"));
    let symbolic = must!(graph.symbolic_vjp(loss, "seed"));
    assert!(!symbolic.gradients.contains_key("mask"));
    let mut seeded = inputs.clone();
    seeded.insert("seed".to_string(), seed.clone());
    assert_eq!(
        must!(symbolic.graph.evaluate(symbolic.gradients["x"], &seeded))
            .data()
            .as_ref(),
        &[0.0, 4.0, 0.0]
    );

    let tangents = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![3], vec![1.0, 1.0, 1.0])),
    )]);
    let (_, tangent) = must!(graph.jvp(loss, &inputs, &tangents));
    assert_eq!(tangent.data().as_ref(), &[4.0]);
    let (_, mask_tangent) = must!(graph.jvp(both, &inputs, &tangents));
    assert_eq!(mask_tangent.data().as_ref(), &[0.0, 0.0, 0.0]);
    assert_eq!(mask_tangent.dtype(), TensorDType::F64);

    // A zero tangent is equivalent to omitting it (callers often supply a tangent per input); a
    // non-zero direction is an error.
    let mut with_mask = tangents.clone();
    with_mask.insert(
        "mask".to_string(),
        must!(DynamicTensor::filled(vec![3], 0.0)),
    );
    assert_eq!(
        must!(graph.jvp(loss, &inputs, &with_mask))
            .1
            .data()
            .as_ref(),
        tangent.data().as_ref()
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
        must!(transformed.graph.evaluate(transformed.tangent, &inputs))
            .data()
            .as_ref(),
        eager_tangent.data().as_ref()
    );
    assert_eq!(eager_tangent.data().as_ref(), &[0.0]);
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
    assert_eq!(
        must!(graph.evaluate(output, &inputs)).data().as_ref(),
        &[10.0]
    );
    let vjp = must!(graph.symbolic_vjp(output, "seed"));
    assert!(!vjp.gradients.contains_key("mask"));
    assert_eq!(
        must!(vjp.graph.evaluate(vjp.gradients["x"], &inputs))
            .data()
            .as_ref(),
        &[2.0, 0.0, 6.0]
    );
    let jvp = must!(graph.symbolic_jvp(output, "x"));
    assert_eq!(
        must!(jvp.graph.evaluate(jvp.tangent, &inputs))
            .data()
            .as_ref(),
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
            actual.data().as_ref(),
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
            assert_eq!(
                actual.data().as_ref(),
                expected.data().as_ref(),
                "{target:?} cond"
            );
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
        value.data().as_ref(),
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
        (
            "runtime VJP d/dx",
            runtime["x"].data().as_ref(),
            &expected_base,
        ),
        (
            "runtime VJP d/dy",
            runtime["y"].data().as_ref(),
            &expected_exponent,
        ),
        (
            "symbolic VJP d/dx",
            symbolic_base.data().as_ref(),
            &expected_base,
        ),
        (
            "symbolic VJP d/dy",
            symbolic_exponent.data().as_ref(),
            &expected_exponent,
        ),
        (
            "runtime JVP d/dx",
            forward_base.data().as_ref(),
            &expected_base,
        ),
        (
            "runtime JVP d/dy",
            forward_exponent.data().as_ref(),
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
        must!(graph.vjp(power, &origin, seed.clone()))["x"]
            .data()
            .as_ref(),
        must!(graph.vjp(root, &origin, seed))["x"].data().as_ref()
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
    // Where the value itself is infinite (0^-1) both routes give the zero convention: the reverse
    // pass of `sum` broadcasts its cotangent without reading the infinite value.
    assert_eq!(
        must!(pow_runtime_hessian([0.0, -1.0])),
        vec![vec![0.0, 0.0], vec![0.0, 0.0]]
    );
    assert_eq!(
        must!(pow_symbolic_hessian([0.0, -1.0])),
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
        must!(plan.evaluate(&inputs)).data().as_ref(),
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
                actual.data().iter().zip(expected.data().iter()).enumerate()
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
    for (actual, expected) in cuda.data().iter().zip(cpu.data().iter()) {
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

/// Symbolic forward-over-reverse HVP of `reduce(pow(v, -1))` along `direction`,
/// with `reduce` a global `sum` or `mean`.
fn reciprocal_symbolic_hvp(
    mean: bool,
    point: &[f64],
    direction: &[f64],
) -> Result<Vec<f64>, String> {
    let mut graph = TensorIr::new();
    let v = graph.input("v", vec![point.len()])?;
    let minus_one = graph.scalar_constant(-1.0);
    let reciprocal = graph.pow(v, minus_one)?;
    let loss = if mean {
        graph.mean(reciprocal)?
    } else {
        graph.sum(reciprocal)?
    };
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let forward = reverse.graph.symbolic_jvp_with_tangent_inputs(
        reverse.gradients["v"],
        &BTreeMap::from([("v".to_string(), "dv".to_string())]),
    )?;
    let inputs = BTreeMap::from([
        (
            "v".to_string(),
            DynamicTensor::new(vec![point.len()], point.to_vec())?,
        ),
        ("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?),
        (
            "dv".to_string(),
            DynamicTensor::new(vec![direction.len()], direction.to_vec())?,
        ),
    ]);
    Ok(forward
        .graph
        .evaluate(forward.tangent, &inputs)?
        .data()
        .to_vec())
}

#[test]
fn symbolic_hessians_stay_finite_where_the_loss_is_infinite() {
    // sum(1 / v) is infinite at v = [0, 2, -4]. The gradient -v^-2 and the Hessian diagonal
    // 2 v^-3 follow the pow zero convention at the origin and are finite elsewhere.
    let point = [0.0, 2.0, -4.0];
    let direction = [1.0, 1.0, 1.0];
    let expected = [0.0, 0.25, -0.03125];
    for mean in [false, true] {
        let scale = if mean { 1.0 / 3.0 } else { 1.0 };
        let mut graph = TensorIr::new();
        let v = must!(graph.input("v", vec![3]));
        let minus_one = graph.scalar_constant(-1.0);
        let reciprocal = must!(graph.pow(v, minus_one));
        let loss = if mean {
            must!(graph.mean(reciprocal))
        } else {
            must!(graph.sum(reciprocal))
        };
        let inputs = BTreeMap::from([(
            "v".to_string(),
            must!(DynamicTensor::new(vec![3], point.to_vec())),
        )]);
        assert_eq!(
            must!(graph.evaluate(loss, &inputs)).data()[0],
            f64::INFINITY
        );

        let symbolic = must!(reciprocal_symbolic_hvp(mean, &point, &direction));
        let runtime = must!(graph.hvp_scalar(
            loss,
            "v",
            &inputs,
            must!(DynamicTensor::new(vec![3], direction.to_vec())),
        ));
        let hessian = must!(graph.hessian_scalar(loss, "v", &inputs));
        for index in 0..3 {
            let expected = scale * expected[index];
            let row_sum = hessian[index].iter().sum::<f64>();
            for (route, actual) in [
                ("symbolic", symbolic[index]),
                ("runtime", runtime.data()[index]),
                ("hessian_scalar", row_sum),
            ] {
                assert!(
                    (actual - expected).abs() <= 1e-15,
                    "mean={mean} {route}[{index}]: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn reduction_cotangent_broadcast_ignores_nan_and_infinite_values() {
    // The reverse pass of sum and mean broadcasts the cotangent to the input shape. Its value and
    // its tangent must not read the input, so NaN and infinities leave the gradient of a linear
    // loss and its (zero) second derivative untouched, and the f32 dtype is kept.
    let point = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5];
    for (mean, gradient) in [(false, 1.0), (true, 0.25)] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input_typed("x", vec![4], TensorDType::F32));
        let loss = if mean {
            must!(graph.mean(x))
        } else {
            must!(graph.sum(x))
        };
        let reverse = must!(graph.symbolic_vjp(loss, "cotangent"));
        let forward = must!(reverse.graph.symbolic_jvp_with_tangent_inputs(
            reverse.gradients["x"],
            &BTreeMap::from([("x".to_string(), "dx".to_string())]),
        ));
        let inputs = BTreeMap::from([
            (
                "x".to_string(),
                must!(DynamicTensor::new(vec![4], point.to_vec())),
            ),
            (
                "cotangent".to_string(),
                must!(DynamicTensor::filled(vec![], 1.0)),
            ),
            ("dx".to_string(), must!(DynamicTensor::filled(vec![4], 1.0))),
        ]);
        let value = must!(forward.graph.evaluate(forward.value, &inputs));
        let tangent = must!(forward.graph.evaluate(forward.tangent, &inputs));
        assert_eq!(value.dtype(), TensorDType::F32, "mean={mean}");
        assert_eq!(tangent.dtype(), TensorDType::F32, "mean={mean}");
        assert_eq!(value.data().as_ref(), [gradient; 4], "mean={mean}");
        assert_eq!(tangent.data().as_ref(), [0.0; 4], "mean={mean}");
    }
}

/// NaN and both infinities, followed by a finite value.
const NON_FINITE_POINT: [f64; 4] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5];

#[test]
fn symbolic_vjp_gives_exact_zero_gradients_to_unused_non_finite_inputs() {
    // `y` receives no cotangent, so its gradient is the zero tensor whatever `y` holds, with the
    // f32 dtype of `y`; the gradient of `x` is not polluted either.
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![4], TensorDType::F32));
    must!(graph.input_typed("y", vec![4], TensorDType::F32));
    let loss = must!(graph.sum(x));
    let reverse = must!(graph.symbolic_vjp(loss, "cotangent"));
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            must!(DynamicTensor::new(vec![4], NON_FINITE_POINT.to_vec())),
        ),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![4], NON_FINITE_POINT.to_vec())),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    let x_gradient = must!(reverse.graph.evaluate(reverse.gradients["x"], &inputs));
    let y_gradient = must!(reverse.graph.evaluate(reverse.gradients["y"], &inputs));
    assert_eq!(x_gradient.data().as_ref(), [1.0; 4]);
    assert_eq!(y_gradient.data().as_ref(), [0.0; 4]);
    assert_eq!(y_gradient.shape(), [4]);
    assert_eq!(y_gradient.dtype(), TensorDType::F32);
}

#[test]
fn symbolic_jvp_gives_exact_zero_tangents_to_non_finite_inputs_without_a_direction() {
    // Only `x` has a tangent input, so the tangent of `x + y` is `dx` even where `y` is NaN or
    // infinite.
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![4], TensorDType::F32));
    let y = must!(graph.input_typed("y", vec![4], TensorDType::F32));
    let output = must!(graph.add(x, y));
    let forward = must!(graph.symbolic_jvp_with_tangent_inputs(
        output,
        &BTreeMap::from([("x".to_string(), "dx".to_string())]),
    ));
    let inputs = BTreeMap::from([
        ("x".to_string(), must!(DynamicTensor::filled(vec![4], 1.0))),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![4], NON_FINITE_POINT.to_vec())),
        ),
        (
            "dx".to_string(),
            must!(DynamicTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0])),
        ),
    ]);
    let tangent = must!(forward.graph.evaluate(forward.tangent, &inputs));
    assert_eq!(tangent.data().as_ref(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tangent.dtype(), TensorDType::F32);
}

#[test]
fn powi_zero_tangents_are_exact_zeros_at_non_finite_inputs() {
    // powi(x, 0) is the constant 1, so its tangent is 0 at every x, and so is the second
    // directional derivative of x + x, whose symbolic JVP seeds x with ones.
    let inputs = |names: &[&str]| {
        names
            .iter()
            .map(|name| {
                let data = if name.starts_with('d') {
                    vec![1.0; 4]
                } else {
                    NON_FINITE_POINT.to_vec()
                };
                DynamicTensor::new(vec![4], data).map(|value| (name.to_string(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
    };
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![4], TensorDType::F32));
    let ones = must!(graph.powi(x, 0));
    let forward = must!(graph.symbolic_jvp_with_tangent_inputs(
        ones,
        &BTreeMap::from([("x".to_string(), "dx".to_string())]),
    ));
    let bound = must!(inputs(&["x", "dx"]));
    assert_eq!(
        must!(forward.graph.evaluate(forward.value, &bound))
            .data()
            .as_ref(),
        [1.0; 4]
    );
    let tangent = must!(forward.graph.evaluate(forward.tangent, &bound));
    assert_eq!(tangent.data().as_ref(), [0.0; 4]);
    assert_eq!(tangent.dtype(), TensorDType::F32);

    let doubled = must!(graph.add(x, x));
    let first = must!(graph.symbolic_jvp(doubled, "x"));
    let second = must!(first.graph.symbolic_jvp_with_tangent_inputs(
        first.tangent,
        &BTreeMap::from([("x".to_string(), "dx".to_string())]),
    ));
    let bound = must!(inputs(&["x", "dx"]));
    assert_eq!(
        must!(second.graph.evaluate(second.value, &bound))
            .data()
            .as_ref(),
        [2.0; 4]
    );
    assert_eq!(
        must!(second.graph.evaluate(second.tangent, &bound))
            .data()
            .as_ref(),
        [0.0; 4]
    );
    // The ones seed of `symbolic_jvp` does not read `x`: the compiled tangent needs no inputs.
    assert_eq!(
        must!(first
            .graph
            .compile_cpu(first.tangent)
            .and_then(|plan| plan.evaluate(&BTreeMap::new())))
        .data()
        .as_ref(),
        [2.0; 4]
    );
}

#[test]
fn cond_vjp_gives_exact_zero_gradients_to_captures_unused_by_the_taken_branch() {
    // The false branch reads `y` only through a comparison, so the gradient of `y` there is 0
    // even for a non-finite `y`.
    let mut on_true = TensorIr::new();
    let true_x = must!(on_true.input("x", vec![4]));
    let true_y = must!(on_true.input("y", vec![4]));
    let true_output = must!(on_true.mul(true_x, true_y));
    let mut on_false = TensorIr::new();
    let false_x = must!(on_false.input("x", vec![4]));
    let false_y = must!(on_false.input("y", vec![4]));
    let two = on_false.scalar_constant(2.0);
    let doubled = must!(on_false.mul(false_x, two));
    let zero = on_false.scalar_constant(0.0);
    let positive = must!(on_false.greater(false_y, zero));
    let false_output = must!(on_false.where_select(positive, doubled, doubled));
    let branches = must!(TensorCondExecutionPlan::new(
        must!(on_true.compile_cpu(true_output)),
        must!(on_false.compile_cpu(false_output)),
    ));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let x = must!(graph.input("x", vec![4]));
    let y = must!(graph.input("y", vec![4]));
    let output = must!(graph.cond_with_captures(
        predicate,
        branches,
        vec![("x".to_string(), x), ("y".to_string(), y)],
    ));
    let loss = must!(graph.sum(output));
    let reverse = must!(graph.symbolic_vjp(loss, "cotangent"));
    let inputs = BTreeMap::from([
        (
            "predicate".to_string(),
            must!(DynamicTensor::new(vec![], vec![0.0])),
        ),
        ("x".to_string(), must!(DynamicTensor::filled(vec![4], 3.0))),
        (
            "y".to_string(),
            must!(DynamicTensor::new(vec![4], NON_FINITE_POINT.to_vec())),
        ),
        (
            "cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        ),
    ]);
    assert_eq!(
        must!(reverse.graph.evaluate(reverse.gradients["x"], &inputs))
            .data()
            .as_ref(),
        [2.0; 4]
    );
    assert_eq!(
        must!(reverse.graph.evaluate(reverse.gradients["y"], &inputs))
            .data()
            .as_ref(),
        [0.0; 4]
    );
}

#[test]
fn scan_vjp_gives_an_exact_zero_cotangent_to_an_unused_non_finite_output() {
    // Only the final carry reaches the loss, so the stacked outputs, which are infinite, receive
    // a zero cotangent: the gradients are those of carry * scale^3 and 0 for the offset.
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![]));
    must!(body.input("index", vec![]));
    let scale = must!(body.input("scale", vec![]));
    let offset = must!(body.input("offset", vec![]));
    let next = must!(body.mul(carry, scale));
    let output = must!(body.add(carry, offset));
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
    let offset = must!(graph.input("offset", vec![]));
    let (final_carry, _outputs) = must!(graph.scan(
        initial,
        scan_plan,
        vec![("scale".to_string(), scale), ("offset".to_string(), offset)],
    ));
    let reverse = must!(graph.symbolic_vjp(final_carry, "cotangent"));
    for offset in [f64::INFINITY, f64::NAN] {
        let inputs = BTreeMap::from([
            (
                "initial".to_string(),
                must!(DynamicTensor::filled(vec![], 1.0)),
            ),
            (
                "scale".to_string(),
                must!(DynamicTensor::filled(vec![], 2.0)),
            ),
            (
                "offset".to_string(),
                must!(DynamicTensor::filled(vec![], offset)),
            ),
            (
                "cotangent".to_string(),
                must!(DynamicTensor::filled(vec![], 1.0)),
            ),
        ]);
        for (name, expected) in [("initial", 8.0), ("scale", 12.0), ("offset", 0.0)] {
            assert_eq!(
                must!(reverse.graph.evaluate(reverse.gradients[name], &inputs))
                    .data()
                    .as_ref(),
                [expected],
                "offset={offset} {name}"
            );
        }
    }
}

/// `f(x, w) = sum(tanh(x * w))` with an auxiliary `mean(x)` that the loss
/// does not use, for the multi-output symbolic VJP tests.
fn symbolic_vjp_many_fixture(
    graph: &mut TensorIr,
) -> Result<(TensorNodeId, TensorNodeId, BTreeMap<String, DynamicTensor>), String> {
    let x = graph.input("x", vec![3])?;
    let w = graph.input("w", vec![3])?;
    let product = graph.mul(x, w)?;
    let activated = graph.tanh(product)?;
    let loss = graph.sum(activated)?;
    let aux = graph.mean(x)?;
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![3], vec![0.5, -1.0, 2.0])?,
        ),
        (
            "w".to_string(),
            DynamicTensor::new(vec![3], vec![0.3, -0.7, 1.1])?,
        ),
    ]);
    Ok((loss, aux, inputs))
}

#[test]
fn symbolic_vjp_many_ones_seed_matches_an_input_seed_and_retains_primals() {
    let mut graph = TensorIr::new();
    let (loss, aux, inputs) = must!(symbolic_vjp_many_fixture(&mut graph));
    let seeded = must!(graph.symbolic_vjp(loss, "seed"));
    let ones = must!(graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)]));
    assert_eq!(ones.cotangents.len(), 1);
    // The ones seed adds no graph input.
    assert!(ones.graph.input_node_id("seed").is_err());

    let mut seeded_inputs = inputs.clone();
    seeded_inputs.insert(
        "seed".to_string(),
        must!(DynamicTensor::filled(vec![], 1.0)),
    );
    for name in ["x", "w"] {
        assert_eq!(
            must!(ones.graph.evaluate(ones.gradients[name], &inputs))
                .data()
                .as_ref(),
            must!(seeded
                .graph
                .evaluate(seeded.gradients[name], &seeded_inputs))
            .data()
            .as_ref(),
            "{name}"
        );
    }
    assert_eq!(
        must!(ones.graph.evaluate(ones.primals[loss], &inputs))
            .data()
            .as_ref(),
        must!(graph.evaluate(loss, &inputs)).data().as_ref()
    );
    // The auxiliary node is rebuilt although the loss does not depend on it.
    assert_close(
        must!(ones.graph.evaluate(ones.primals[aux], &inputs))
            .data()
            .as_ref(),
        &[0.5],
        1e-15,
    );

    // One frozen program returns the value, the auxiliary output, and both gradients.
    let outputs = vec![
        ones.primals[loss],
        ones.primals[aux],
        ones.gradients["x"],
        ones.gradients["w"],
    ];
    let program = must!(QuablaMultiOutputProgram::new(ones.graph.clone(), outputs));
    let executable = must!(QuablaCompiler.compile_many(&program, QuablaTarget::Cpu));
    let values = must!(executable.execute(&inputs));
    assert_eq!(values.len(), 4);
    assert_eq!(
        values[0].data().as_ref(),
        must!(graph.evaluate(loss, &inputs)).data().as_ref()
    );
    assert_eq!(
        values[3].data().as_ref(),
        must!(seeded.graph.evaluate(seeded.gradients["w"], &seeded_inputs))
            .data()
            .as_ref()
    );
}

#[test]
fn symbolic_vjp_many_sums_the_vjps_of_several_outputs() {
    let mut graph = TensorIr::new();
    let (loss, aux, inputs) = must!(symbolic_vjp_many_fixture(&mut graph));
    let both = must!(graph.symbolic_vjp_many(&[
        (loss, SymbolicCotangent::Input("loss_bar".to_string())),
        (aux, SymbolicCotangent::Input("aux_bar".to_string())),
    ]));
    let mut seeded = inputs.clone();
    seeded.insert(
        "loss_bar".to_string(),
        must!(DynamicTensor::filled(vec![], 2.0)),
    );
    seeded.insert(
        "aux_bar".to_string(),
        must!(DynamicTensor::filled(vec![], 3.0)),
    );
    let (_, loss_gradients) =
        must!(graph.value_and_vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 2.0))));
    let (_, aux_gradients) =
        must!(graph.value_and_vjp(aux, &inputs, must!(DynamicTensor::filled(vec![], 3.0))));
    for name in ["x", "w"] {
        let expected = loss_gradients[name]
            .data()
            .iter()
            .zip(aux_gradients[name].data().iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>();
        assert_close(
            must!(both.graph.evaluate(both.gradients[name], &seeded))
                .data()
                .as_ref(),
            &expected,
            1e-14,
        );
    }

    // An output listed twice accumulates both seeds.
    let twice = must!(graph.symbolic_vjp_many(&[
        (loss, SymbolicCotangent::Ones),
        (loss, SymbolicCotangent::Ones),
    ]));
    let (_, doubled) =
        must!(graph.value_and_vjp(loss, &inputs, must!(DynamicTensor::filled(vec![], 2.0))));
    assert_close(
        must!(twice.graph.evaluate(twice.gradients["w"], &inputs))
            .data()
            .as_ref(),
        doubled["w"].data().as_ref(),
        1e-15,
    );
}

#[test]
fn symbolic_vjp_many_rejects_invalid_seeds() {
    let mut graph = TensorIr::new();
    let (loss, aux, _) = must!(symbolic_vjp_many_fixture(&mut graph));
    let error = graph
        .symbolic_vjp_many(&[])
        .expect_err("no outputs must be rejected");
    assert!(error.contains("at least one output"), "{error}");
    let error = graph
        .symbolic_vjp_many(&[
            (loss, SymbolicCotangent::Input("bar".to_string())),
            (aux, SymbolicCotangent::Input("bar".to_string())),
        ])
        .expect_err("duplicate seed names must be rejected");
    assert!(error.contains("duplicate cotangent input name"), "{error}");
    let error = graph
        .symbolic_vjp_many(&[(loss, SymbolicCotangent::Input("x".to_string()))])
        .expect_err("a seed name that shadows an input must be rejected");
    assert!(
        error.contains("conflicts with an existing input"),
        "{error}"
    );
    let mask = must!(graph.input_typed("mask", vec![], TensorDType::Bool));
    let error = graph
        .symbolic_vjp_many(&[(mask, SymbolicCotangent::Ones)])
        .expect_err("a bool output must be rejected");
    assert!(
        error.contains("cannot differentiate a bool output"),
        "{error}"
    );
}

#[test]
fn symbolic_jvp_over_a_ones_seeded_vjp_is_the_hessian_vector_product() {
    let mut graph = TensorIr::new();
    let (loss, _, inputs) = must!(symbolic_vjp_many_fixture(&mut graph));
    let reverse = must!(graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)]));
    let forward = must!(reverse.graph.symbolic_jvp_many_with_tangent_inputs(
        &[reverse.gradients["w"]],
        &BTreeMap::from([("w".to_string(), "w_dot".to_string())]),
    ));
    let direction = must!(DynamicTensor::new(vec![3], vec![1.0, -2.0, 0.5]));
    let mut tangent_inputs = inputs.clone();
    tangent_inputs.insert("w_dot".to_string(), direction.clone());
    let expected = must!(graph.hvp_scalar(loss, "w", &inputs, direction));
    assert_close(
        must!(forward.graph.evaluate(forward.tangents[0], &tangent_inputs))
            .data()
            .as_ref(),
        expected.data().as_ref(),
        1e-14,
    );
}

fn scalar(value: f64) -> Result<DynamicTensor, String> {
    DynamicTensor::new(vec![], vec![value])
}

#[test]
fn inline_splices_the_reachable_callee_with_bound_inputs() {
    let mut callee = TensorIr::new();
    let a = must!(callee.input("a", vec![3]));
    let b = must!(callee.input("b", vec![3]));
    let product = must!(callee.mul(a, b));
    let activated = must!(callee.tanh(product));
    let total = must!(callee.sum(activated));
    let one = callee.scalar_constant(1.0);
    let shifted = must!(callee.add(a, one));
    let _unused = must!(callee.exp(b));

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let y = must!(graph.input("y", vec![3]));
    let two = graph.scalar_constant(2.0);
    let doubled = must!(graph.mul(x, two));
    let before = graph.node_count();
    let outputs = must!(graph.inline(
        &callee,
        &BTreeMap::from([("a".to_string(), doubled), ("b".to_string(), y)]),
        &[total, shifted, a],
    ));
    // product, tanh, sum, the constant, and add; the unused exp is not copied
    // and the input `a` resolves to its binding.
    assert_eq!(graph.node_count(), before + 5);
    assert_eq!(outputs[2], doubled);

    let x_value = must!(DynamicTensor::new(vec![3], vec![0.5, -1.0, 2.0]));
    let y_value = must!(DynamicTensor::new(vec![3], vec![0.3, -0.7, 1.1]));
    let graph_inputs = BTreeMap::from([
        ("x".to_string(), x_value.clone()),
        ("y".to_string(), y_value.clone()),
    ]);
    let callee_inputs = BTreeMap::from([
        (
            "a".to_string(),
            must!(DynamicTensor::new(vec![3], vec![1.0, -2.0, 4.0])),
        ),
        ("b".to_string(), y_value),
    ]);
    for (inlined, source) in outputs.iter().zip([total, shifted, a]) {
        assert_eq!(
            must!(graph.evaluate(*inlined, &graph_inputs))
                .data()
                .as_ref(),
            must!(callee.evaluate(source, &callee_inputs))
                .data()
                .as_ref()
        );
    }
    // The spliced nodes differentiate like nodes traced in place.
    let gradients = must!(graph.symbolic_vjp_many(&[(outputs[0], SymbolicCotangent::Ones)]));
    let (_, expected) = must!(callee.value_and_vjp(total, &callee_inputs, must!(scalar(1.0))));
    let expected_x = expected["a"]
        .data()
        .iter()
        .map(|value| 2.0 * value)
        .collect::<Vec<_>>();
    assert_close(
        must!(gradients
            .graph
            .evaluate(gradients.gradients["x"], &graph_inputs))
        .data()
        .as_ref(),
        &expected_x,
        1e-15,
    );
    assert_close(
        must!(gradients
            .graph
            .evaluate(gradients.gradients["y"], &graph_inputs))
        .data()
        .as_ref(),
        expected["b"].data().as_ref(),
        1e-15,
    );
}

#[test]
fn inline_rejects_mismatched_unknown_and_missing_bindings() {
    let mut callee = TensorIr::new();
    let a = must!(callee.input("a", vec![2]));
    let b = must!(callee.input_typed("b", vec![2], TensorDType::F32));
    let a32 = must!(callee.cast(a, TensorDType::F32));
    let output = must!(callee.mul(a32, b));

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let x32 = must!(graph.input_typed("x32", vec![2], TensorDType::F32));
    let wide = must!(graph.input("wide", vec![3]));
    let cases = [
        (vec![("a", x), ("b", x)], "the shape and dtype must match"),
        (
            vec![("a", wide), ("b", x32)],
            "the shape and dtype must match",
        ),
        (vec![("a", x), ("c", x32)], "not a callee input"),
        (vec![("a", x)], "leaves callee input \"b\" unbound"),
    ];
    for (bindings, message) in cases {
        let bindings = bindings
            .into_iter()
            .map(|(name, node)| (name.to_string(), node))
            .collect::<BTreeMap<_, _>>();
        let before = graph.node_count();
        let error = graph
            .inline(&callee, &bindings, &[output])
            .expect_err("invalid bindings must be rejected");
        assert!(error.contains(message), "{error}");
        // Validation precedes every append except the unbound-input check,
        // which is reached only while splicing.
        if message != "leaves callee input \"b\" unbound" {
            assert_eq!(graph.node_count(), before);
        }
    }
    let error = graph
        .inline(
            &callee,
            &BTreeMap::from([("a".to_string(), x), ("b".to_string(), x32)]),
            &[output + 100],
        )
        .expect_err("an unknown output must be rejected");
    assert!(error.contains("does not exist"), "{error}");
}

#[test]
fn inline_preserves_dtypes_and_weak_constants() {
    let mut callee = TensorIr::new();
    let a = must!(callee.input_typed("a", vec![2], TensorDType::F32));
    let two = callee.scalar_constant(2.0);
    let four = must!(callee.mul(two, two));
    let scaled = must!(callee.mul(a, two));
    let mask = must!(callee.greater(a, two));

    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![2], TensorDType::F32));
    let outputs = must!(graph.inline(
        &callee,
        &BTreeMap::from([("a".to_string(), x)]),
        &[four, scaled, mask],
    ));
    assert_eq!(must!(graph.node_dtype(outputs[0])), TensorDType::F64);
    assert_eq!(must!(graph.node_dtype(outputs[1])), TensorDType::F32);
    assert_eq!(
        must!(graph.node_dtype(outputs[2])),
        must!(callee.node_dtype(mask))
    );
    // The weak constant still adopts the dtype of a strong f32 operand; a
    // strong f64 node would be rejected as a mixed-dtype operand.
    let sum = must!(graph.add(x, outputs[0]));
    assert_eq!(must!(graph.node_dtype(sum)), TensorDType::F32);
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::with_dtype(
            vec![2],
            vec![0.1, 3.0],
            TensorDType::F32
        )),
    )]);
    assert_eq!(
        must!(graph.evaluate(sum, &inputs)).dtype(),
        TensorDType::F32
    );
    assert_eq!(
        must!(graph.evaluate(outputs[1], &inputs)).data().as_ref(),
        must!(callee.evaluate(
            scaled,
            &BTreeMap::from([("a".to_string(), inputs["x"].clone())])
        ))
        .data()
        .as_ref()
    );
    let strong = must!(graph.input("strong", vec![]));
    assert!(graph.add(x, strong).is_err());
}

#[test]
fn compiled_plans_common_subexpressions_after_inline() {
    let mut callee = TensorIr::new();
    let a = must!(callee.input("a", vec![4]));
    let sine = must!(callee.sin(a));
    let two = callee.scalar_constant(2.0);
    let scaled = must!(callee.mul(sine, two));

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4]));
    let own_sine = must!(graph.sin(x));
    let outputs = must!(graph.inline(&callee, &BTreeMap::from([("a".to_string(), x)]), &[scaled]));
    // The graph holds sin(x) twice; the compiled plan holds it once.
    let (plan, _) = must!(graph.compile_cpu_many(&[own_sine, outputs[0]]));

    let mut direct = TensorIr::new();
    let x_direct = must!(direct.input("x", vec![4]));
    let sine_direct = must!(direct.sin(x_direct));
    let two_direct = direct.scalar_constant(2.0);
    let scaled_direct = must!(direct.mul(sine_direct, two_direct));
    let (direct_plan, _) = must!(direct.compile_cpu_many(&[sine_direct, scaled_direct]));
    assert_eq!(plan.node_count(), direct_plan.node_count());

    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![4], vec![0.1, 0.2, 0.3, 0.4])),
    )]);
    assert_eq!(
        must!(plan.evaluate_many(&inputs)),
        must!(direct_plan.evaluate_many(&inputs))
    );
}

/// A scalar loss over `x` and `s` through a Scan, a Fori, and a Cond region,
/// each with an explicit capture of `s`.
fn inline_region_fixture() -> Result<(TensorIr, TensorNodeId), String> {
    let mut scan_body = TensorIr::new();
    let carry = scan_body.input("carry", vec![])?;
    let index = scan_body.input("index", vec![])?;
    let scale = scan_body.input("scale", vec![])?;
    let scaled = scan_body.mul(carry, scale)?;
    let next = scan_body.add(scaled, index)?;
    let output = scan_body.sin(next)?;
    let scan_plan = TensorScanExecutionPlan::new(
        0,
        3,
        scan_body.compile_cpu_many(&[next, output])?.0,
        "carry",
        "index",
    )?;

    let mut fori_body = TensorIr::new();
    let carry = fori_body.input("carry", vec![])?;
    let shift = fori_body.input("shift", vec![])?;
    let product = fori_body.mul(carry, shift)?;
    let next = fori_body.tanh(product)?;
    let fori_plan =
        TensorForiExecutionPlan::new(0, 2, fori_body.compile_cpu(next)?, "carry", "index")?;

    let mut on_true = TensorIr::new();
    let captured = on_true.input("captured", vec![])?;
    let true_output = on_true.mul(captured, captured)?;
    let mut on_false = TensorIr::new();
    let captured = on_false.input("captured", vec![])?;
    let three = on_false.scalar_constant(3.0);
    let false_output = on_false.mul(captured, three)?;
    let branches = TensorCondExecutionPlan::new(
        on_true.compile_cpu(true_output)?,
        on_false.compile_cpu(false_output)?,
    )?;

    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![])?;
    let s = graph.input("s", vec![])?;
    let (final_carry, outputs) = graph.scan(x, scan_plan, vec![("scale".to_string(), s)])?;
    let looped = graph.fori(x, fori_plan, vec![("shift".to_string(), s)])?;
    let zero = graph.scalar_constant(0.0);
    let predicate = graph.greater(x, zero)?;
    let branched =
        graph.cond_with_captures(predicate, branches, vec![("captured".to_string(), s)])?;
    let output_sum = graph.sum(outputs)?;
    let partial = graph.add(final_carry, output_sum)?;
    let partial = graph.add(partial, looped)?;
    let loss = graph.add(partial, branched)?;
    Ok((graph, loss))
}

#[test]
fn inline_splices_region_graphs_and_keeps_two_splices_separate() {
    let (source, loss) = must!(inline_region_fixture());
    // The reverse-mode graph adds ScanVjp and ForiVjp groups to the forward
    // Scan group; the callee returns the loss and both gradients.
    let reverse = must!(source.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)]));
    let callee_outputs = [
        reverse.primals[loss],
        reverse.gradients["x"],
        reverse.gradients["s"],
    ];

    let mut graph = TensorIr::new();
    let p = must!(graph.input("p", vec![]));
    let q = must!(graph.input("q", vec![]));
    let minus_one = graph.scalar_constant(-1.0);
    let negated = must!(graph.mul(p, minus_one));
    let first = must!(graph.inline(
        &reverse.graph,
        &BTreeMap::from([("x".to_string(), p), ("s".to_string(), q)]),
        &callee_outputs,
    ));
    let second = must!(graph.inline(
        &reverse.graph,
        &BTreeMap::from([("x".to_string(), negated), ("s".to_string(), p)]),
        &callee_outputs,
    ));
    assert!(graph.lower_text().contains("scan_vjp(group="));

    let (p_value, q_value) = (0.7, -0.4);
    let graph_inputs = BTreeMap::from([
        ("p".to_string(), must!(scalar(p_value))),
        ("q".to_string(), must!(scalar(q_value))),
    ]);
    let (plan, plan_outputs) =
        must!(graph.compile_cpu_many(&[first.clone(), second.clone()].concat()));
    let plan_values = must!(plan.evaluate_many(&graph_inputs));
    assert_eq!(plan_outputs.len(), 6);
    for (splice, (x_value, s_value)) in [(0, (p_value, q_value)), (1, (-p_value, p_value))] {
        let callee_inputs = BTreeMap::from([
            ("x".to_string(), must!(scalar(x_value))),
            ("s".to_string(), must!(scalar(s_value))),
        ]);
        let outputs = if splice == 0 { &first } else { &second };
        for (index, (inlined, source_output)) in outputs.iter().zip(callee_outputs).enumerate() {
            let expected = must!(reverse.graph.evaluate(source_output, &callee_inputs));
            assert_eq!(
                must!(graph.evaluate(*inlined, &graph_inputs))
                    .data()
                    .as_ref(),
                expected.data().as_ref(),
                "splice {splice} output {index}"
            );
            assert_eq!(
                plan_values[3 * splice + index].data().as_ref(),
                expected.data().as_ref(),
                "compiled splice {splice} output {index}"
            );
        }
    }

    // Forward mode through the spliced reverse-mode regions is the HVP of
    // the source loss, as in hessian = jacobian(grad(f)).
    let forward = must!(graph.symbolic_jvp_many_with_tangent_inputs(
        &[first[2]],
        &BTreeMap::from([("q".to_string(), "q_dot".to_string())]),
    ));
    let mut tangent_inputs = graph_inputs.clone();
    tangent_inputs.insert("q_dot".to_string(), must!(scalar(1.0)));
    let callee_inputs = BTreeMap::from([
        ("x".to_string(), must!(scalar(p_value))),
        ("s".to_string(), must!(scalar(q_value))),
    ]);
    let expected = must!(source.hessian_scalar(loss, "s", &callee_inputs));
    assert_close(
        must!(forward.graph.evaluate(forward.tangents[0], &tangent_inputs))
            .data()
            .as_ref(),
        &[expected[0][0]],
        1e-12,
    );
}

fn vector(data: &[f64]) -> Result<DynamicTensor, String> {
    DynamicTensor::new(vec![data.len()], data.to_vec())
}

#[test]
fn array_constants_evaluate_with_a_strong_dtype() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let c = graph.constant(must!(vector(&[1.0, -2.0, 0.5])), false);
    assert_eq!(must!(graph.node_shape(c)), vec![3]);
    assert_eq!(must!(graph.node_dtype(c)), TensorDType::F64);
    let product = must!(graph.mul(x, c));
    let output = must!(graph.add(product, c));
    let inputs = BTreeMap::from([("x".to_string(), must!(vector(&[3.0, 4.0, -1.0])))]);
    assert_eq!(
        must!(graph.evaluate(output, &inputs)).data().as_ref(),
        &[4.0, -10.0, 0.0]
    );
    // The elements are not printed.
    assert!(graph
        .lower_text()
        .contains(&format!("%{c} = constant[dense] : tensor<3xf64>")));

    // A strong f32 constant meets an f32 operand; a strong f64 constant is a
    // dtype error that asks for astype, like an f64 input.
    let x32 = must!(graph.input_typed("x32", vec![3], TensorDType::F32));
    let c32 = graph.constant(
        must!(DynamicTensor::with_dtype(
            vec![3],
            vec![0.1, 0.2, 0.3],
            TensorDType::F32
        )),
        false,
    );
    let sum32 = must!(graph.add(x32, c32));
    assert_eq!(must!(graph.node_dtype(sum32)), TensorDType::F32);
    let before = graph.node_count();
    let error = graph
        .add(x32, c)
        .expect_err("strong f64 + f32 must be rejected");
    assert!(error.contains("mismatched dtypes f32 and f64"), "{error}");
    assert!(error.contains("astype"), "{error}");
    assert_eq!(graph.node_count(), before);
    // A weak constant adopts the strong operand's dtype.
    let weak = graph.constant(must!(vector(&[1.0, 2.0, 3.0])), true);
    let adopted = must!(graph.add(x32, weak));
    assert_eq!(must!(graph.node_dtype(adopted)), TensorDType::F32);

    // A bool constant selects like a traced mask.
    let mask = graph.constant(
        must!(DynamicTensor::with_dtype(
            vec![3],
            vec![1.0, 0.0, 1.0],
            TensorDType::Bool
        )),
        false,
    );
    let zero = graph.scalar_constant(0.0);
    let selected = must!(graph.where_select(mask, x, zero));
    let mut inputs = inputs;
    inputs.insert(
        "x32".to_string(),
        must!(DynamicTensor::with_dtype(
            vec![3],
            vec![0.0; 3],
            TensorDType::F32
        )),
    );
    assert_eq!(
        must!(graph.evaluate(selected, &inputs)).data().as_ref(),
        &[3.0, 0.0, -1.0]
    );
}

#[test]
fn array_constants_fold_like_per_node_execution() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![3], TensorDType::F32));
    let c = graph.constant(
        must!(DynamicTensor::with_dtype(
            vec![3],
            vec![0.1, 1.7, -2.3],
            TensorDType::F32
        )),
        false,
    );
    let two = graph.scalar_constant(2.0);
    let sine = must!(graph.sin(c));
    let scaled = must!(graph.mul(sine, two));
    let output = must!(graph.add(x, scaled));
    let plan = must!(graph.compile_cpu(output));
    // sin(c) * 2 folds into one f32 constant: x, the constant, and the add.
    assert_eq!(plan.node_count(), 3);
    assert!(!plan.lower_text().contains("sin("), "{}", plan.lower_text());
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::with_dtype(
            vec![3],
            vec![0.25, -1.5, 3.0],
            TensorDType::F32
        )),
    )]);
    let folded = must!(plan.evaluate(&inputs));
    let unfolded = must!(graph.evaluate(output, &inputs));
    assert_eq!(folded.dtype(), TensorDType::F32);
    assert_eq!(folded.data().as_ref(), unfolded.data().as_ref());

    // A zero divisor folds to the IEEE quotient that execution produces.
    let z = graph.constant(must!(vector(&[1.0, 0.0, 2.0])), false);
    let ones = graph.constant(must!(vector(&[1.0, 1.0, 1.0])), false);
    let quotient = must!(graph.div(ones, z));
    let plan = must!(graph.compile_cpu(quotient));
    let folded = must!(plan.evaluate(&BTreeMap::new()));
    assert_eq!(folded.data().as_ref(), &[1.0, f64::INFINITY, 0.5]);
    assert_eq!(
        folded.data().as_ref(),
        must!(graph.evaluate(quotient, &inputs)).data().as_ref()
    );
}

#[test]
fn array_constants_deduplicate_by_bits_in_compiled_plans() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let mut outputs = Vec::new();
    for data in [
        [1.5, nan],
        [1.5, nan],
        [0.0, 2.0],
        [-0.0, 2.0],
        [1.5, f64::from_bits(0x7ff8_0000_0000_0002)],
    ] {
        let c = graph.constant(must!(vector(&data)), false);
        outputs.push(must!(graph.mul(x, c)));
    }
    // The first two constants are separate allocations with equal bits and
    // merge, and so do their products; -0.0 and 0.0, and NaNs with different
    // payloads, stay apart.
    let (plan, plan_outputs) = must!(graph.compile_cpu_many(&outputs));
    assert_eq!(plan_outputs[0], plan_outputs[1]);
    assert_eq!(plan.node_count(), 1 + 4 + 4);
    let inputs = BTreeMap::from([("x".to_string(), must!(vector(&[2.0, 1.0])))]);
    let values = must!(plan.evaluate_many(&inputs));
    assert_eq!(values[2].data()[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(values[3].data()[0].to_bits(), (-0.0f64).to_bits());

    // dtype and weakness are part of the identity.
    let mut graph = TensorIr::new();
    let strong = graph.constant(must!(vector(&[1.0, 2.0])), false);
    let weak = graph.constant(must!(vector(&[1.0, 2.0])), true);
    let narrow = graph.constant(
        must!(DynamicTensor::with_dtype(
            vec![2],
            vec![1.0, 2.0],
            TensorDType::F32
        )),
        false,
    );
    let (plan, _) = must!(graph.compile_cpu_many(&[strong, weak, narrow]));
    assert_eq!(plan.node_count(), 3);
}

#[test]
fn array_constants_have_zero_tangents_and_receive_no_cotangent() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let c_data = [0.5, -1.0, 2.0];
    let c = graph.constant(must!(vector(&c_data)), false);
    let product = must!(graph.mul(x, c));
    let sine = must!(graph.sin(product));
    let shifted = must!(graph.add(sine, c));
    let loss = must!(graph.sum(shifted));
    let x_data = [0.3, 0.1, -0.2];
    let inputs = BTreeMap::from([("x".to_string(), must!(vector(&x_data)))]);
    let expected_gradient = x_data
        .iter()
        .zip(c_data)
        .map(|(x, c)| c * (x * c).cos())
        .collect::<Vec<_>>();

    let (_, gradients) = must!(graph.value_and_vjp(loss, &inputs, must!(scalar(1.0))));
    assert_eq!(gradients.keys().collect::<Vec<_>>(), vec!["x"]);
    assert_close(gradients["x"].data().as_ref(), &expected_gradient, 1e-15);

    let symbolic = must!(graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)]));
    assert_eq!(symbolic.gradients.keys().collect::<Vec<_>>(), vec!["x"]);
    assert_close(
        must!(symbolic.graph.evaluate(symbolic.gradients["x"], &inputs))
            .data()
            .as_ref(),
        &expected_gradient,
        1e-15,
    );

    let direction = [1.0, -2.0, 0.5];
    let expected_tangent = expected_gradient
        .iter()
        .zip(direction)
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    let tangents = BTreeMap::from([("x".to_string(), must!(vector(&direction)))]);
    let (_, tangent) = must!(graph.jvp(loss, &inputs, &tangents));
    assert_close(tangent.data().as_ref(), &[expected_tangent], 1e-15);
    let forward = must!(graph.symbolic_jvp_with_tangent_inputs(
        loss,
        &BTreeMap::from([("x".to_string(), "x_dot".to_string())]),
    ));
    let mut forward_inputs = inputs.clone();
    forward_inputs.insert("x_dot".to_string(), must!(vector(&direction)));
    assert_close(
        must!(forward.graph.evaluate(forward.tangent, &forward_inputs))
            .data()
            .as_ref(),
        &[expected_tangent],
        1e-15,
    );

    // Second order: d^2/dx^2 sum(sin(x * c)) = -c^2 sin(x * c) on the diagonal.
    let hvp = must!(graph.hvp_scalar(loss, "x", &inputs, must!(vector(&direction))));
    let expected_hvp = x_data
        .iter()
        .zip(c_data)
        .zip(direction)
        .map(|((x, c), direction)| -c * c * (x * c).sin() * direction)
        .collect::<Vec<_>>();
    assert_close(hvp.data().as_ref(), &expected_hvp, 1e-14);

    // The forward tangent of a constant output is a zero of its dtype.
    let (_, constant_tangent) = must!(graph.jvp(c, &inputs, &tangents));
    assert_eq!(constant_tangent.data().as_ref(), &[0.0, 0.0, 0.0]);
}

#[test]
fn inline_shares_array_constants_between_splices() {
    let mut callee = TensorIr::new();
    let a = must!(callee.input("a", vec![2]));
    let c = callee.constant(must!(vector(&[3.0, -4.0])), false);
    let product = must!(callee.mul(a, c));
    let output = must!(callee.sum(product));

    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let y = must!(graph.input("y", vec![2]));
    let first = must!(graph.inline(&callee, &BTreeMap::from([("a".to_string(), x)]), &[output]));
    let second = must!(graph.inline(&callee, &BTreeMap::from([("a".to_string(), y)]), &[output]));
    let inputs = BTreeMap::from([
        ("x".to_string(), must!(vector(&[1.0, 2.0]))),
        ("y".to_string(), must!(vector(&[-1.0, 0.5]))),
    ]);
    assert_eq!(
        must!(graph.evaluate(first[0], &inputs)).data().as_ref(),
        &[-5.0]
    );
    assert_eq!(
        must!(graph.evaluate(second[0], &inputs)).data().as_ref(),
        &[-5.0]
    );
    // Each splice copies the constant node; the plan holds it once.
    let (plan, _) = must!(graph.compile_cpu_many(&[first[0], second[0]]));
    assert_eq!(plan.lower_text().matches("constant[dense]").count(), 1);
    // A resident constant takes no temporary buffer slot.
    let buffers = must!(must!(graph.compile_cpu(first[0])).buffer_plan());
    let plan = must!(graph.compile_cpu(first[0]));
    let constant_id = plan
        .lower_text()
        .lines()
        .position(|line| line.contains("constant[dense]"))
        .expect("the plan keeps the constant");
    assert_eq!(buffers.node_slots[constant_id], None);
}

/// A graph whose elementwise chain reads an array constant, plus a gradient
/// that reads it too, so device plans run a fusion region with a constant
/// leaf and a multi-output program.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn constant_chain_fixture() -> Result<(TensorIr, Vec<TensorNodeId>), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![4])?;
    let c = graph.constant(
        DynamicTensor::new(vec![4], vec![0.5, -1.25, 2.0, 0.75])?,
        false,
    );
    let product = graph.mul(x, c)?;
    let sine = graph.sin(product)?;
    let shifted = graph.add(sine, c)?;
    let loss = graph.sum(shifted)?;
    let reverse = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    let outputs = vec![
        reverse.primals[shifted],
        reverse.primals[loss],
        reverse.gradients["x"],
    ];
    Ok((reverse.graph, outputs))
}

#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn assert_constant_chain_parity(target: QuablaTarget) {
    let (graph, outputs) = must!(constant_chain_fixture());
    let program = must!(QuablaMultiOutputProgram::new(graph, outputs));
    let compiler = QuablaCompiler;
    let cpu = must!(compiler.compile_many(&program, QuablaTarget::Cpu));
    let device = must!(compiler.compile_many(&program, target));
    for x in [[0.1, 0.2, -0.3, 0.4], [1.0, -2.0, 0.5, 0.0]] {
        let inputs = BTreeMap::from([("x".to_string(), must!(vector(&x)))]);
        let expected = must!(cpu.execute(&inputs));
        // Repeated executions read the same resident constant.
        for _ in 0..2 {
            let actual = must!(device.execute(&inputs));
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_close(actual.data().as_ref(), expected.data().as_ref(), 1e-5);
            }
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_backend_executes_array_constants_like_cpu() {
    assert_constant_chain_parity(QuablaTarget::Mlx);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_executes_array_constants_like_cpu_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    assert_constant_chain_parity(QuablaTarget::Cuda { device_ordinal: 0 });
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_backend_uploads_array_constants_once_per_plan_when_enabled() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4]));
    let c = graph.constant(must!(vector(&[0.5, -1.25, 2.0, 0.75])), false);
    let product = must!(graph.mul(x, c));
    let sine = must!(graph.sin(product));
    let output = must!(graph.add(sine, c));
    let cpu_plan = must!(graph.compile_cpu(output));
    let plan = must!(CudaBackend::new(0).compile(cpu_plan.clone()));
    // The chain runs as one fusion region whose leaves are `x` and the
    // constant's device buffer.
    assert!(plan.fused_region_count() >= 1);
    assert_eq!(plan.constant_upload_count(), 0);
    for x in [[0.1, 0.2, -0.3, 0.4], [1.0, -2.0, 0.5, 0.0], [0.0; 4]] {
        let inputs = BTreeMap::from([("x".to_string(), must!(vector(&x)))]);
        assert_close(
            must!(plan.execute(&inputs)).data().as_ref(),
            must!(cpu_plan.evaluate(&inputs)).data().as_ref(),
            1e-5,
        );
        assert_eq!(plan.constant_upload_count(), 1);
    }

    // A Cond region that returns a constant copies it into the parent's
    // buffer and keeps its own resident copy.
    let on_true = {
        let mut region = TensorIr::new();
        let captured = must!(region.input("captured", vec![4]));
        let two = region.scalar_constant(2.0);
        let output = must!(region.mul(captured, two));
        must!(region.compile_cpu(output))
    };
    let on_false = {
        let mut region = TensorIr::new();
        let captured = must!(region.input("captured", vec![4]));
        let output = region.constant(must!(vector(&[9.0, 8.0, 7.0, 6.0])), false);
        // Both branches must capture the same inputs, so the plan also keeps
        // the unused capture as a second output; the constant is the result.
        let (plan, _) = must!(region.compile_cpu_many(&[output, captured]));
        plan
    };
    let branches = must!(TensorCondExecutionPlan::new(on_true, on_false));
    let mut graph = TensorIr::new();
    let predicate = must!(graph.input("predicate", vec![]));
    let x = must!(graph.input("x", vec![4]));
    let conditional =
        must!(graph.cond_with_captures(predicate, branches, vec![("captured".to_string(), x)],));
    let cpu_plan = must!(graph.compile_cpu(conditional));
    let plan = must!(CudaBackend::new(0).compile(cpu_plan.clone()));
    for predicate in [0.0, 1.0, 0.0, 0.0] {
        let inputs = BTreeMap::from([
            ("predicate".to_string(), must!(scalar(predicate))),
            ("x".to_string(), must!(vector(&[1.0, 2.0, 3.0, 4.0]))),
        ]);
        assert_eq!(
            must!(plan.execute(&inputs)).data().as_ref(),
            must!(cpu_plan.evaluate(&inputs)).data().as_ref()
        );
    }

    // The fused loop kernel reads arrays only through captures, so a body with
    // an array constant runs as a host-driven region loop whose body program
    // keeps the constant resident across iterations and executions.
    let mut body = TensorIr::new();
    let carry = must!(body.input("carry", vec![2]));
    let _index = must!(body.input("index", vec![]));
    let step = body.constant(must!(vector(&[1.0, 2.0])), false);
    let next = must!(body.add(carry, step));
    let loop_plan = must!(TensorForiExecutionPlan::new(
        0,
        3,
        must!(body.compile_cpu(next)),
        "carry",
        "index",
    ));
    let mut graph = TensorIr::new();
    let initial = must!(graph.input("initial", vec![2]));
    let output = must!(graph.fori(initial, loop_plan, vec![]));
    let cpu_plan = must!(graph.compile_cpu(output));
    let plan = must!(CudaBackend::new(0).compile(cpu_plan.clone()));
    for start in [0.5, -1.0] {
        let inputs =
            BTreeMap::from([("initial".to_string(), must!(vector(&[start, 2.0 * start])))]);
        assert_eq!(
            must!(plan.execute(&inputs)).data().as_ref(),
            must!(cpu_plan.evaluate(&inputs)).data().as_ref()
        );
    }
}

/// The leading-axis slice `index` of a mapped value, as one example.
fn example_of(value: &DynamicTensor, index: usize) -> Result<DynamicTensor, String> {
    let shape = value.shape()[1..].to_vec();
    let count = shape.iter().product::<usize>();
    DynamicTensor::with_dtype(
        shape,
        value.data()[index * count..(index + 1) * count].to_vec(),
        value.dtype(),
    )
}

/// One example per step: a callee over scalar `x`, vector `v`, and matrix
/// `a` (mapped) and vector `w` and matrix `m` (unmapped) that reaches every
/// batching rule, returning its outputs and the scalar total.
fn batching_fixture() -> Result<(TensorIr, Vec<TensorNodeId>, TensorNodeId), String> {
    use quabla_core::tensor_ir::TensorComparison;
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![])?;
    let v = callee.input("v", vec![3])?;
    let w = callee.input("w", vec![3])?;
    let m = callee.input("m", vec![3, 2])?;
    let a = callee.input("a", vec![2, 2])?;
    // A mapped scalar against an unmapped vector: the padding case.
    let scaled = callee.mul(x, w)?;
    let powered = callee.pow(v, x)?;
    let less = callee.compare(v, w, TensorComparison::Less)?;
    let selected = callee.where_select(less, scaled, powered)?;
    let activated = callee.tanh(selected)?;
    let sin_v = callee.sin(v)?;
    let cos_w = callee.cos(w)?;
    let wave = callee.mul(sin_v, cos_w)?;
    let exp_x = callee.exp(x)?;
    let square = callee.powi(v, 2)?;
    let one = callee.scalar_constant(1.0);
    let shifted = callee.add(square, one)?;
    let root = callee.sqrt(shifted)?;
    let ratio = callee.div(exp_x, root)?;
    let w_square = callee.powi(w, 2)?;
    let w_shifted = callee.add(w_square, one)?;
    let log_w = callee.log(w_shifted)?;
    let mixed = callee.add(activated, wave)?;
    let mixed = callee.sub(mixed, ratio)?;
    let mixed = callee.add(mixed, log_w)?;
    // Matmul with the mapped operand on either side and on both.
    let row = callee.reshape(mixed, vec![1, 3])?;
    let left = callee.matmul(row, m)?;
    let m_t = callee.transpose(m, None)?;
    let column = callee.reshape(mixed, vec![3, 1])?;
    let right = callee.matmul(m_t, column)?;
    let a_squared = callee.matmul(a, a)?;
    let lower = callee.triangular(a_squared, true)?;
    let a_t = callee.transpose(a, Some(vec![1, 0]))?;
    let lower = callee.add(lower, a_t)?;
    let left_flat = callee.reshape(left, vec![2])?;
    let right_flat = callee.reshape(right, vec![2])?;
    let w_head = callee.slice_axis(w, 0, 0, 2)?;
    let joined = callee.concat(vec![left_flat, w_head, right_flat], 0)?;
    let tail = callee.slice_axis(joined, 0, 2, 5)?;
    let column_sums = callee.sum_axis(lower, 0)?;
    let row_means = callee.mean_axis(lower, 1)?;
    let reduced = callee.add(column_sums, row_means)?;
    let spread = callee.broadcast_to(x, vec![2, 2])?;
    let spread = callee.mul(spread, lower)?;
    let half = callee.scalar_constant(0.5);
    let mask = callee.greater(v, half)?;
    let narrowed = callee.cast(mixed, TensorDType::F32)?;
    let total = callee.sum(joined)?;
    let average = callee.mean(spread)?;
    let tail_total = callee.sum(tail)?;
    let total = callee.add(total, average)?;
    let total = callee.add(total, tail_total)?;
    let reduced_total = callee.sum(reduced)?;
    let total = callee.add(total, reduced_total)?;
    // Gather a mapped operand and scatter it into an unmapped base, which
    // the batching rule broadcasts.
    let picked = callee.gather(v, vec![2, 0, 2].into(), 0)?;
    let scattered = callee.scatter_add(w, picked, vec![1, 1, 0].into(), 0)?;
    let scattered_total = callee.sum(scattered)?;
    let total = callee.add(total, scattered_total)?;
    // An output that depends on no mapped input stays unmapped.
    let two = callee.scalar_constant(2.0);
    let sin_w = callee.sin(w)?;
    let unmapped = callee.mul(sin_w, two)?;
    let outputs = vec![
        selected, joined, reduced, spread, total, mask, narrowed, unmapped,
    ];
    Ok((callee, outputs, total))
}

fn batching_inputs(batch: usize) -> Result<BTreeMap<String, DynamicTensor>, String> {
    let values = |count: usize, offset: f64| {
        (0..count)
            .map(|index| 0.35 + 0.17 * index as f64 + offset)
            .collect::<Vec<_>>()
    };
    Ok(BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![batch], values(batch, 0.1))?,
        ),
        (
            "v".to_string(),
            DynamicTensor::new(vec![batch, 3], values(batch * 3, 0.05))?,
        ),
        (
            "w".to_string(),
            DynamicTensor::new(vec![3], vec![0.9, 0.2, 1.4])?,
        ),
        (
            "m".to_string(),
            DynamicTensor::new(vec![3, 2], vec![0.5, -1.0, 0.25, 2.0, -0.75, 1.5])?,
        ),
        (
            "a".to_string(),
            DynamicTensor::new(vec![batch, 2, 2], values(batch * 4, -0.6))?,
        ),
    ]))
}

const BATCHING_MAPPED: [(&str, bool); 5] = [
    ("x", true),
    ("v", true),
    ("w", false),
    ("m", false),
    ("a", true),
];

/// Batches `callee` over the inputs of [`batching_inputs`] and checks every
/// output against a per-example evaluation of the callee.
fn assert_batched_matches_examples(
    callee: &TensorIr,
    outputs: &[TensorNodeId],
    expected_mapped: &[bool],
) {
    let batch = 4;
    let inputs = must!(batching_inputs(batch));
    let mut graph = TensorIr::new();
    let mut bindings = BTreeMap::new();
    for (name, is_mapped) in BATCHING_MAPPED {
        let value = &inputs[name];
        let node = must!(graph.input(name, value.shape().to_vec()));
        bindings.insert(name.to_string(), (node, is_mapped));
    }
    let batched = must!(graph.inline_batched(callee, &bindings, batch, outputs));
    assert_eq!(
        batched
            .iter()
            .map(|(_, mapped)| *mapped)
            .collect::<Vec<_>>(),
        expected_mapped
    );
    for ((node, is_mapped), output) in batched.iter().zip(outputs) {
        let value = must!(graph.evaluate(*node, &inputs));
        for index in 0..batch {
            let mut example_inputs = BTreeMap::new();
            for (name, input_mapped) in BATCHING_MAPPED {
                let input = &inputs[name];
                example_inputs.insert(
                    name.to_string(),
                    if input_mapped {
                        must!(example_of(input, index))
                    } else {
                        input.clone()
                    },
                );
            }
            let expected = must!(callee.evaluate(*output, &example_inputs));
            let actual = if *is_mapped {
                must!(example_of(&value, index))
            } else {
                value.clone()
            };
            assert_eq!(actual.shape(), expected.shape(), "output {output}");
            assert_eq!(actual.dtype(), expected.dtype(), "output {output}");
            assert_close(actual.data().as_ref(), expected.data().as_ref(), 1e-13);
        }
    }
}

#[test]
fn inline_batched_matches_a_per_example_loop_for_every_batching_rule() {
    let (callee, outputs, _) = must!(batching_fixture());
    assert_batched_matches_examples(
        &callee,
        &outputs,
        &[true, true, true, true, true, true, true, false],
    );
}

#[test]
fn inline_batched_reverse_mode_graphs_give_per_example_gradients() {
    // Batching the VJP graph gives every example its own gradient, also for
    // the unmapped inputs `w` and `m` (JAX semantics); the reverse pass adds
    // broadcast, pad, and reduce-to-shape nodes to batch.
    let (callee, _, total) = must!(batching_fixture());
    let vjp = must!(callee.symbolic_vjp_many(&[(total, SymbolicCotangent::Ones)]));
    let names = ["x", "v", "w", "m", "a"];
    let outputs = names
        .iter()
        .map(|name| vjp.gradients[*name])
        .collect::<Vec<_>>();
    assert_batched_matches_examples(&vjp.graph, &outputs, &[true; 5]);
}

#[test]
fn inline_batched_graphs_batch_again_for_nested_vmap() {
    let mut callee = TensorIr::new();
    let x = must!(callee.input("x", vec![]));
    let w = must!(callee.input("w", vec![2]));
    let product = must!(callee.mul(x, w));
    let output = must!(callee.sin(product));

    // Inner vmap over x ([3]), then an outer vmap over w ([2, 2]) and x.
    let mut inner = TensorIr::new();
    let xs = must!(inner.input("xs", vec![3]));
    let w_inner = must!(inner.input("w", vec![2]));
    let batched = must!(inner.inline_batched(
        &callee,
        &BTreeMap::from([
            ("x".to_string(), (xs, true)),
            ("w".to_string(), (w_inner, false)),
        ]),
        3,
        &[output],
    ));
    let mut outer = TensorIr::new();
    let xss = must!(outer.input("xss", vec![2, 3]));
    let ws = must!(outer.input("ws", vec![2, 2]));
    let nested = must!(outer.inline_batched(
        &inner,
        &BTreeMap::from([
            ("xs".to_string(), (xss, true)),
            ("w".to_string(), (ws, true)),
        ]),
        2,
        &[batched[0].0],
    ));
    assert!(nested[0].1);
    let x_values = [0.1, 0.2, 0.3, -0.4, 0.5, 0.6];
    let w_values = [1.5, -2.0, 0.25, 3.0];
    let value = must!(outer.evaluate(
        nested[0].0,
        &BTreeMap::from([
            (
                "xss".to_string(),
                must!(DynamicTensor::new(vec![2, 3], x_values.to_vec()))
            ),
            (
                "ws".to_string(),
                must!(DynamicTensor::new(vec![2, 2], w_values.to_vec()))
            ),
        ])
    ));
    assert_eq!(value.shape(), &[2, 3, 2]);
    let mut expected = Vec::new();
    for outer_index in 0..2 {
        for inner_index in 0..3 {
            for element in 0..2 {
                expected.push(
                    (x_values[outer_index * 3 + inner_index] * w_values[outer_index * 2 + element])
                        .sin(),
                );
            }
        }
    }
    assert_close(value.data().as_ref(), &expected, 1e-15);
}

#[test]
fn inline_batched_batches_solve_and_rejects_regions_and_invalid_bindings() {
    use quabla_core::tensor_ir::BatchingError;
    let mut callee = TensorIr::new();
    let matrix = must!(callee.input("matrix", vec![2, 2]));
    let rhs = must!(callee.input("rhs", vec![2, 1]));
    let solved = must!(callee.solve(matrix, rhs));

    let mut graph = TensorIr::new();
    let matrix_node = must!(graph.input("matrix", vec![2, 2]));
    let rhs_node = must!(graph.input("rhs", vec![3, 2, 1]));
    // A mapped solve broadcasts its unmapped matrix over the batch.
    let batched = must!(graph.inline_batched(
        &callee,
        &BTreeMap::from([
            ("matrix".to_string(), (matrix_node, false)),
            ("rhs".to_string(), (rhs_node, true)),
        ]),
        3,
        &[solved],
    ));
    assert!(batched[0].1);
    assert_eq!(must!(graph.node_shape(batched[0].0)), vec![3, 2, 1]);
    let value = must!(graph.evaluate(
        batched[0].0,
        &BTreeMap::from([
            (
                "matrix".to_string(),
                must!(DynamicTensor::new(vec![2, 2], vec![2.0, 0.0, 1.0, 4.0])),
            ),
            (
                "rhs".to_string(),
                must!(DynamicTensor::new(
                    vec![3, 2, 1],
                    vec![2.0, 5.0, 4.0, 2.0, 0.0, 8.0]
                )),
            ),
        ]),
    ));
    assert_eq!(value.data().as_ref(), &[1.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    // An unmapped solve is copied as is.
    let rhs_single = must!(graph.input("rhs_single", vec![2, 1]));
    let copied = must!(graph.inline_batched(
        &callee,
        &BTreeMap::from([
            ("matrix".to_string(), (matrix_node, false)),
            ("rhs".to_string(), (rhs_single, false)),
        ]),
        3,
        &[solved],
    ));
    assert!(!copied[0].1);

    // Region nodes with a mapped operand are rejected, unmapped ones copied.
    let (regions, loss) = must!(inline_region_fixture());
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4]));
    let s = must!(graph.input("s", vec![]));
    let error = graph
        .inline_batched(
            &regions,
            &BTreeMap::from([("x".to_string(), (x, true)), ("s".to_string(), (s, false))]),
            4,
            &[loss],
        )
        .expect_err("a mapped region has no batching rule");
    assert!(
        matches!(error, BatchingError::Unsupported { op } if ["scan", "fori", "cond"].contains(&op)),
        "{error:?}"
    );
    assert!(error.to_string().contains("vmap cannot batch"), "{error}");
    let x_single = must!(graph.input("x_single", vec![]));
    let copied = must!(graph.inline_batched(
        &regions,
        &BTreeMap::from([
            ("x".to_string(), (x_single, false)),
            ("s".to_string(), (s, false)),
        ]),
        4,
        &[loss],
    ));
    assert!(!copied[0].1);

    // Bindings: a mapped binding needs the leading batch axis.
    let wrong = [
        ((x, false), "the input shape"),
        (
            (s, true),
            "a mapped binding must have shape [4, *input shape]",
        ),
    ];
    for (binding, message) in wrong {
        let error = graph
            .inline_batched(
                &regions,
                &BTreeMap::from([("x".to_string(), binding), ("s".to_string(), (s, false))]),
                4,
                &[loss],
            )
            .expect_err("a mismatched binding must be rejected");
        assert!(
            matches!(&error, BatchingError::Invalid(text) if text.contains(message)),
            "{error:?}"
        );
    }
    let error = graph
        .inline_batched(
            &regions,
            &BTreeMap::from([("x".to_string(), (x, true)), ("s".to_string(), (s, false))]),
            0,
            &[loss],
        )
        .expect_err("a zero batch must be rejected");
    assert!(error.to_string().contains("above zero"), "{error}");
}

fn log1p_input(values: &[f64]) -> Result<BTreeMap<String, DynamicTensor>, String> {
    Ok(BTreeMap::from([(
        "x".to_string(),
        DynamicTensor::new(vec![values.len()], values.to_vec())?,
    )]))
}

#[test]
fn log1p_keeps_ieee_values_without_raising() {
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![8]));
    let output = must!(graph.log1p(x));
    let points = [-2.0, -1.0, -0.5, -1e-12, 0.0, 1e-12, 3.0, f64::INFINITY];
    let inputs = must!(log1p_input(&points));
    let plan = must!(graph.compile_cpu(output));
    for value in [
        must!(graph.evaluate(output, &inputs)),
        must!(plan.evaluate(&inputs)),
    ] {
        let data = value.data();
        assert!(data[0].is_nan(), "log1p(-2) is NaN: {data:?}");
        assert_eq!(data[1], f64::NEG_INFINITY);
        for (actual, point) in data.iter().zip(points).skip(2) {
            assert_eq!(actual.to_bits(), point.ln_1p().to_bits(), "log1p({point})");
        }
    }
    assert!(graph.lower_text().contains("log1p(%"));

    // A constant operand folds through the same IEEE kernel instead of failing.
    let mut folded = TensorIr::new();
    let minus_one = folded.scalar_constant(-1.0);
    let output = must!(folded.log1p(minus_one));
    let value = must!(must!(folded.compile_cpu(output)).evaluate(&BTreeMap::new()));
    assert_eq!(value.data()[0], f64::NEG_INFINITY);
}

#[test]
fn log1p_derivatives_agree_across_runtime_and_symbolic_routes() {
    let points = [-0.75, -0.5, 0.0, 1e-9, 3.0];
    let expected = points.map(|point: f64| 1.0 / (1.0 + point));
    let inputs = must!(log1p_input(&points));
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![points.len()]));
    let output = must!(graph.log1p(x));
    let ones = must!(DynamicTensor::filled(vec![points.len()], 1.0));
    let close = |actual: &[f64]| {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1e-15 * expected.abs(),
                "{actual} vs {expected}"
            );
        }
    };

    close(&must!(graph.vjp(output, &inputs, ones.clone()))["x"].data());
    let tangents = BTreeMap::from([("x".to_string(), ones.clone())]);
    let (value, tangent) = must!(graph.jvp(output, &inputs, &tangents));
    close(&tangent.data());
    assert_eq!(value.data()[0], (-0.75_f64).ln_1p());

    let reverse = must!(graph.symbolic_vjp(output, "cotangent"));
    let mut symbolic_inputs = inputs.clone();
    symbolic_inputs.insert("cotangent".to_string(), ones.clone());
    close(
        &must!(reverse
            .graph
            .evaluate(reverse.gradients["x"], &symbolic_inputs))
        .data(),
    );
    let forward = must!(graph.symbolic_jvp_with_tangent_inputs(
        output,
        &BTreeMap::from([("x".to_string(), "dx".to_string())]),
    ));
    let mut forward_inputs = inputs.clone();
    forward_inputs.insert("dx".to_string(), ones);
    close(&must!(forward.graph.evaluate(forward.tangent, &forward_inputs)).data());

    // Second order: d2/dx2 log1p(x) = -1 / (1 + x)^2 on the runtime mixed
    // route and by symbolic forward-over-reverse.
    let point = 0.6_f64;
    let second = -1.0 / ((1.0 + point) * (1.0 + point));
    let mut scalar = TensorIr::new();
    let x = must!(scalar.input("x", vec![]));
    let log = must!(scalar.log1p(x));
    let point_inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![], vec![point])),
    )]);
    let runtime = must!(scalar.hessian_scalar(log, "x", &point_inputs));
    assert!((runtime[0][0] - second).abs() <= 1e-15, "{runtime:?}");
    let reverse = must!(scalar.symbolic_vjp(log, "cotangent"));
    let forward = must!(reverse.graph.symbolic_jvp_with_tangent_inputs(
        reverse.gradients["x"],
        &BTreeMap::from([("x".to_string(), "dx".to_string())]),
    ));
    let mut hessian_inputs = point_inputs.clone();
    hessian_inputs.insert(
        "cotangent".to_string(),
        must!(DynamicTensor::filled(vec![], 1.0)),
    );
    hessian_inputs.insert("dx".to_string(), must!(DynamicTensor::filled(vec![], 1.0)));
    let symbolic = must!(forward.graph.evaluate(forward.tangent, &hessian_inputs));
    assert!((symbolic.data()[0] - second).abs() <= 1e-15, "{symbolic:?}");
}

#[cfg(any(
    all(feature = "cuda", target_os = "linux"),
    all(feature = "mlx", target_os = "macos")
))]
type PlanWithInputs = (TensorExecutionPlan, BTreeMap<String, DynamicTensor>);

/// The gradient of `sum(log1p(x) * x)` and a Fori whose body applies
/// `log1p`, so device backends lower log1p as a per-node kernel and inside a
/// loop body.
#[cfg(any(
    all(feature = "cuda", target_os = "linux"),
    all(feature = "mlx", target_os = "macos")
))]
fn log1p_device_plans() -> Result<Vec<PlanWithInputs>, String> {
    let points = vec![-0.9, -0.5, 0.0, 1e-6, 0.25, 3.0];
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![points.len()])?;
    let log = graph.log1p(x)?;
    let product = graph.mul(log, x)?;
    let loss = graph.sum(product)?;
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let gradient_plan = reverse.graph.compile_cpu(reverse.gradients["x"])?;
    let gradient_inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![points.len()], points.clone())?,
        ),
        ("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?),
    ]);

    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![points.len()])?;
    let index = body.input("index", vec![])?;
    let log = body.log1p(carry)?;
    let next = body.add(log, index)?;
    let loop_plan = TensorForiExecutionPlan::new(0, 3, body.compile_cpu(next)?, "carry", "index")?;
    let mut looped = TensorIr::new();
    let initial = looped.input("x", vec![points.len()])?;
    let output = looped.fori(initial, loop_plan, vec![])?;
    let loop_inputs = BTreeMap::from([(
        "x".to_string(),
        DynamicTensor::new(
            vec![points.len()],
            points.iter().map(|point| point + 1.0).collect(),
        )?,
    )]);
    Ok(vec![
        (gradient_plan, gradient_inputs),
        (looped.compile_cpu(output)?, loop_inputs),
    ])
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_log1p_matches_cpu_values_and_gradients() {
    if std::env::var_os("QUABLA_MLX_TEST").is_none() {
        return;
    }
    for (plan, inputs) in must!(log1p_device_plans()) {
        let cpu = must!(plan.evaluate(&inputs));
        let mlx = must!(MlxBackend.execute(&plan, &inputs));
        assert_eq!(mlx.shape(), cpu.shape());
        for (device, host) in mlx.data().iter().zip(cpu.data().iter()) {
            assert!(
                (device - host).abs() <= 2e-6 * host.abs().max(1.0),
                "{device} vs {host}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_log1p_matches_cpu_values_and_gradients() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    for (plan, inputs) in must!(log1p_device_plans()) {
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        for (device, host) in cuda.data().iter().zip(cpu.data().iter()) {
            assert!(
                (device - host).abs() <= 2e-6 * host.abs().max(1.0),
                "{device} vs {host}"
            );
        }
    }
}

fn gather_scatter_inputs(dtype: TensorDType) -> BTreeMap<String, DynamicTensor> {
    let tensor = |shape: Vec<usize>, data: Vec<f64>| {
        DynamicTensor::with_dtype(shape, data, dtype).expect("valid test tensor")
    };
    BTreeMap::from([
        (
            "x".to_string(),
            tensor(vec![4, 2], (0..8).map(|v| 1.0 + v as f64).collect()),
        ),
        (
            "u".to_string(),
            tensor(
                vec![5, 2],
                (0..10).map(|v| 10.0 * (v as f64 - 4.0)).collect(),
            ),
        ),
        (
            "ct_gather".to_string(),
            tensor(vec![5, 2], (0..10).map(|v| 0.5 * v as f64 - 1.0).collect()),
        ),
        (
            "ct_scatter".to_string(),
            tensor(vec![4, 2], (0..8).map(|v| 3.0 - v as f64).collect()),
        ),
        (
            "tx".to_string(),
            tensor(vec![4, 2], (0..8).map(|v| 0.25 * v as f64).collect()),
        ),
        (
            "tu".to_string(),
            tensor(vec![5, 2], (0..10).map(|v| 2.0 - v as f64).collect()),
        ),
    ])
}

/// Rows `indices` of the row-major `[rows, 2]` matrix `data`.
fn take_rows(data: &[f64], indices: &[usize]) -> Vec<f64> {
    indices
        .iter()
        .flat_map(|row| data[2 * row..2 * row + 2].to_vec())
        .collect()
}

/// `base` with row `j` of `updates` added to row `indices[j]`, in order.
fn add_rows(base: &[f64], updates: &[f64], indices: &[usize]) -> Vec<f64> {
    let mut result = base.to_vec();
    for (source, row) in indices.iter().enumerate() {
        for column in 0..2 {
            result[2 * row + column] += updates[2 * source + column];
        }
    }
    result
}

#[test]
fn gather_and_scatter_add_follow_their_forward_and_derivative_rules() {
    // Row 2 repeats three times, so the gather cotangent takes the reordered
    // accumulation path; the values are small integers, exact in any order.
    let indices = [2, 0, 2, 2, 1];
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![4, 2]));
    let u = must!(graph.input("u", vec![5, 2]));
    let gathered = must!(graph.gather(x, indices.to_vec().into(), 0));
    let scattered = must!(graph.scatter_add(x, u, indices.to_vec().into(), 0));
    let inputs = gather_scatter_inputs(TensorDType::F64);
    let data = |name: &str| inputs[name].data().to_vec();
    let (xs, us) = (data("x"), data("u"));
    let zeros = vec![0.0; 8];
    let gather_value = take_rows(&xs, &indices);
    let scatter_value = add_rows(&xs, &us, &indices);
    assert_eq!(
        must!(graph.evaluate(gathered, &inputs)).data().as_ref(),
        gather_value
    );
    assert_eq!(
        must!(graph.evaluate(scattered, &inputs)).data().as_ref(),
        scatter_value
    );

    // Reverse mode: gather scatters its cotangent back; scatter_add passes the
    // cotangent to its base and gathers it for its updates.
    let gather_x = add_rows(&zeros, &data("ct_gather"), &indices);
    let scatter_u = take_rows(&data("ct_scatter"), &indices);
    let runtime = must!(graph.vjp(gathered, &inputs, inputs["ct_gather"].clone()));
    assert_eq!(runtime["x"].data().as_ref(), gather_x);
    let runtime = must!(graph.vjp(scattered, &inputs, inputs["ct_scatter"].clone()));
    assert_eq!(runtime["x"].data().as_ref(), data("ct_scatter"));
    assert_eq!(runtime["u"].data().as_ref(), scatter_u);
    for (output, seed, expected_x, expected_u) in [
        (gathered, "ct_gather", gather_x.clone(), vec![0.0; 10]),
        (
            scattered,
            "ct_scatter",
            data("ct_scatter"),
            scatter_u.clone(),
        ),
    ] {
        let vjp = must!(graph.symbolic_vjp(output, seed));
        let text = vjp.graph.lower_text();
        assert!(!text.contains("slice"), "{text}");
        assert_eq!(
            must!(vjp.graph.evaluate(vjp.gradients["x"], &inputs))
                .data()
                .as_ref(),
            expected_x
        );
        assert_eq!(
            must!(vjp.graph.evaluate(vjp.gradients["u"], &inputs))
                .data()
                .as_ref(),
            expected_u
        );
    }

    // Forward mode: gather of the tangent; scatter_add of both tangents.
    let tangents = BTreeMap::from([
        ("x".to_string(), inputs["tx"].clone()),
        ("u".to_string(), inputs["tu"].clone()),
    ]);
    let names = BTreeMap::from([
        ("x".to_string(), "tx".to_string()),
        ("u".to_string(), "tu".to_string()),
    ]);
    for (output, expected) in [
        (gathered, take_rows(&data("tx"), &indices)),
        (scattered, add_rows(&data("tx"), &data("tu"), &indices)),
    ] {
        let (_, runtime) = must!(graph.jvp(output, &inputs, &tangents));
        assert_eq!(runtime.data().as_ref(), expected);
        let jvp = must!(graph.symbolic_jvp_with_tangent_inputs(output, &names));
        assert_eq!(
            must!(jvp.graph.evaluate(jvp.tangent, &inputs))
                .data()
                .as_ref(),
            expected
        );
    }
}

#[test]
fn scatter_add_rounds_duplicates_in_index_order_and_its_reverse_mode_mirrors_it() {
    // In f32, 1 + 1e8 rounds back to 1e8, so the order of the three
    // contributions to row 0 decides the result: index order gives 0, and
    // the gather cotangent accumulates in the reverse order of the per-index
    // slices it replaces, giving 1.
    let mut graph = TensorIr::new();
    let base = must!(graph.input_typed("base", vec![1], TensorDType::F32));
    let updates = must!(graph.input_typed("updates", vec![3], TensorDType::F32));
    let scattered = must!(graph.scatter_add(base, updates, vec![0, 0, 0].into(), 0));
    let gathered = must!(graph.gather(base, vec![0, 0, 0].into(), 0));
    let zero = graph.scalar_constant(0.0);
    let shifted = must!(graph.add(base, zero));
    let f32_tensor = |shape: Vec<usize>, data: Vec<f64>| {
        DynamicTensor::with_dtype(shape, data, TensorDType::F32).expect("valid test tensor")
    };
    let inputs = BTreeMap::from([
        ("base".to_string(), f32_tensor(vec![1], vec![0.0])),
        (
            "updates".to_string(),
            f32_tensor(vec![3], vec![1.0, 1e8, -1e8]),
        ),
        ("ct".to_string(), f32_tensor(vec![3], vec![1.0, 1e8, -1e8])),
        (
            "ct_gather".to_string(),
            f32_tensor(vec![3], vec![1.0, 1.0, -1e8]),
        ),
        ("ct_shifted".to_string(), f32_tensor(vec![1], vec![1e8])),
    ]);
    assert_eq!(
        must!(graph.evaluate(scattered, &inputs)).data().as_ref(),
        [0.0]
    );
    let vjp = must!(graph.symbolic_vjp(gathered, "ct"));
    assert_eq!(
        must!(vjp.graph.evaluate(vjp.gradients["base"], &inputs))
            .data()
            .as_ref(),
        [1.0]
    );
    // The later `shifted` seeds 1e8 first; the gather then adds -1e8, 1, and
    // 1 into that running sum (giving 2), not their own sum -1e8 (giving 0).
    let vjp = must!(graph.symbolic_vjp_many(&[
        (gathered, SymbolicCotangent::Input("ct_gather".to_string())),
        (shifted, SymbolicCotangent::Input("ct_shifted".to_string())),
    ]));
    assert_eq!(
        must!(vjp.graph.evaluate(vjp.gradients["base"], &inputs))
            .data()
            .as_ref(),
        [2.0]
    );
    let runtime = must!(graph.value_and_vjp_many(
        &[
            (gathered, inputs["ct_gather"].clone()),
            (shifted, inputs["ct_shifted"].clone()),
        ],
        &inputs,
    ));
    assert_eq!(runtime.1["base"].data().as_ref(), [2.0]);
}

/// Gather and scatter_add along an inner axis with an index repeated three
/// times and f32 contributions whose sum depends on their order, so device
/// parity also checks the accumulation order.
#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
type GatherScatterCase = (TensorIr, Vec<TensorNodeId>, BTreeMap<String, DynamicTensor>);

#[cfg(any(
    all(feature = "mlx", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
fn gather_scatter_device_case() -> Result<GatherScatterCase, String> {
    let indices = vec![1, 3, 1, 1, 0];
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![3, 4], TensorDType::F32)?;
    let u = graph.input_typed("u", vec![3, 5], TensorDType::F32)?;
    let gathered = graph.gather(x, indices.clone().into(), 1)?;
    let scattered = graph.scatter_add(x, u, indices.into(), 1)?;
    // Positions 0, 2, and 3 of each row land on column 1; every other order
    // of their three sums rounds to a different f32 result.
    let updates = vec![
        1e8, 7.0, -1e8, 1.0, 0.25, //
        1.0, 5.0, 1e8, -1e8, 2.0, //
        -1e8, 3.0, 1e8, 0.5, 4.0,
    ];
    let inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::with_dtype(
                vec![3, 4],
                (0..12).map(|v| 0.5 * v as f64 - 2.0).collect(),
                TensorDType::F32,
            )?,
        ),
        (
            "u".to_string(),
            DynamicTensor::with_dtype(vec![3, 5], updates, TensorDType::F32)?,
        ),
    ]);
    Ok((graph, vec![gathered, scattered], inputs))
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_gather_and_scatter_add_match_cpu_bits_with_repeated_indices() {
    let (graph, outputs, inputs) = must!(gather_scatter_device_case());
    assert_mlx_readback_matches_cpu(&graph, &outputs, &inputs);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_gather_and_scatter_add_match_cpu_bits_with_repeated_indices() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    let (graph, outputs, inputs) = must!(gather_scatter_device_case());
    for output in outputs {
        let plan = must!(graph.compile_cpu(output));
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        assert_eq!(cuda.data().as_ref(), cpu.data().as_ref(), "output {output}");
    }
}

/// Inputs from `(name, shape, values)` triples.
fn shaped_inputs(
    entries: &[(&str, Vec<usize>, Vec<f64>)],
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    entries
        .iter()
        .map(|(name, shape, values)| {
            Ok((
                name.to_string(),
                DynamicTensor::new(shape.clone(), values.clone())?,
            ))
        })
        .collect()
}

/// `actual` equals `expected` within `tolerance` relative to each expected
/// entry; an expected zero must be matched exactly and an expected NaN by a NaN.
fn assert_relative(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (actual, expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{actual} differs from NaN");
        } else {
            assert!(
                actual == expected || (actual - expected).abs() <= tolerance * expected.abs(),
                "{actual} differs from {expected} (relative tolerance {tolerance})"
            );
        }
    }
}

fn assert_same_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "{actual} is not bitwise {expected}"
        );
    }
}

/// Checks `d output / d input` of the scalar `output` against `expected` (one
/// flat gradient per input name) through the runtime VJP and JVP and the
/// symbolic VJP and JVP; forward mode runs one unit direction per entry.
fn assert_gradient_routes(
    graph: &TensorIr,
    output: TensorNodeId,
    inputs: &BTreeMap<String, DynamicTensor>,
    expected: &[(&str, Vec<f64>)],
    tolerance: f64,
) {
    let one = must!(DynamicTensor::filled(vec![], 1.0));
    let runtime = must!(graph.vjp(output, inputs, one.clone()));
    let reverse = must!(graph.symbolic_vjp(output, "cotangent"));
    let mut reverse_inputs = inputs.clone();
    reverse_inputs.insert("cotangent".to_string(), one);
    for (name, gradient) in expected {
        assert_relative(&runtime[*name].data(), gradient, tolerance);
        let symbolic = must!(reverse
            .graph
            .evaluate(reverse.gradients[*name], &reverse_inputs));
        assert_relative(&symbolic.data(), gradient, tolerance);
        let forward = must!(graph.symbolic_jvp_with_tangent_inputs(
            output,
            &BTreeMap::from([(name.to_string(), "direction".to_string())]),
        ));
        let shape = inputs[*name].shape().to_vec();
        for (index, entry) in gradient.iter().enumerate() {
            let mut unit = vec![0.0; gradient.len()];
            unit[index] = 1.0;
            let direction = must!(DynamicTensor::new(shape.clone(), unit));
            let mut tangents = BTreeMap::new();
            for (other, value) in inputs {
                let zeros = must!(DynamicTensor::filled(value.shape().to_vec(), 0.0));
                tangents.insert(other.clone(), zeros);
            }
            tangents.insert(name.to_string(), direction.clone());
            let (_, tangent) = must!(graph.jvp(output, inputs, &tangents));
            assert_relative(&tangent.data(), &[*entry], tolerance);
            let mut forward_inputs = inputs.clone();
            forward_inputs.insert("direction".to_string(), direction);
            let symbolic = must!(forward.graph.evaluate(forward.tangent, &forward_inputs));
            assert_relative(&symbolic.data(), &[*entry], tolerance);
        }
    }
}

/// Checks the Hessian of the scalar `output` in `name` against `expected`
/// (row-major) on the runtime mixed route and by symbolic forward-over-reverse.
fn assert_hessian_routes(
    graph: &TensorIr,
    output: TensorNodeId,
    name: &str,
    inputs: &BTreeMap<String, DynamicTensor>,
    expected: &[f64],
    tolerance: f64,
) {
    let runtime = must!(graph.hessian_scalar(output, name, inputs));
    let flat = runtime.concat();
    assert_relative(&flat, expected, tolerance);
    let reverse = must!(graph.symbolic_vjp(output, "cotangent"));
    let forward = must!(reverse.graph.symbolic_jvp_with_tangent_inputs(
        reverse.gradients[name],
        &BTreeMap::from([(name.to_string(), "direction".to_string())]),
    ));
    let shape = inputs[name].shape().to_vec();
    let count = runtime.len();
    for column in 0..count {
        let mut unit = vec![0.0; count];
        unit[column] = 1.0;
        let mut forward_inputs = inputs.clone();
        forward_inputs.insert(
            "cotangent".to_string(),
            must!(DynamicTensor::filled(vec![], 1.0)),
        );
        forward_inputs.insert(
            "direction".to_string(),
            must!(DynamicTensor::new(shape.clone(), unit)),
        );
        let values = must!(forward.graph.evaluate(forward.tangent, &forward_inputs));
        let expected_column = (0..count)
            .map(|row| expected[row * count + column])
            .collect::<Vec<_>>();
        assert_relative(&values.data(), &expected_column, tolerance);
    }
}

const SPECIAL_POINTS: [f64; 18] = [
    f64::NEG_INFINITY,
    -750.0,
    -20.0,
    -1.0,
    -1e-10,
    -0.0,
    0.0,
    5e-324,
    1e-300,
    1e-8,
    0.5,
    3.0,
    6.0,
    30.0,
    709.0,
    710.0,
    f64::INFINITY,
    f64::NAN,
];

#[test]
fn expm1_erf_and_atan2_values_follow_the_f64_reference() {
    let count = SPECIAL_POINTS.len();
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![count]));
    let expm1 = must!(graph.expm1(x));
    let erf = must!(graph.erf(x));
    // Every ordered pair of special points, so signed zeros, infinities and
    // NaN meet in both operand positions.
    let pairs = count * count;
    let y_points = must!(graph.input("ys", vec![pairs]));
    let x_points = must!(graph.input("xs", vec![pairs]));
    let atan2 = must!(graph.atan2(y_points, x_points));
    let inputs = must!(shaped_inputs(&[
        ("x", vec![count], SPECIAL_POINTS.to_vec()),
        (
            "ys",
            vec![pairs],
            (0..pairs)
                .map(|index| SPECIAL_POINTS[index / count])
                .collect(),
        ),
        (
            "xs",
            vec![pairs],
            (0..pairs)
                .map(|index| SPECIAL_POINTS[index % count])
                .collect(),
        ),
    ]));
    let expected_atan2 = (0..pairs)
        .map(|index| SPECIAL_POINTS[index / count].atan2(SPECIAL_POINTS[index % count]))
        .collect::<Vec<_>>();
    for (output, expected) in [
        (expm1, SPECIAL_POINTS.map(f64::exp_m1).to_vec()),
        (erf, SPECIAL_POINTS.map(libm::erf).to_vec()),
        (atan2, expected_atan2),
    ] {
        assert_same_bits(&must!(graph.evaluate(output, &inputs)).data(), &expected);
        let plan = must!(graph.compile_cpu(output));
        assert_same_bits(&must!(plan.evaluate(&inputs)).data(), &expected);
    }
    let text = graph.lower_text();
    for name in ["expm1(%", "erf(%", "atan2(%"] {
        assert!(text.contains(name), "{name} is missing from {text}");
    }

    // atan2 broadcasts like the other binary ops, and float32 rounds the f64
    // result once.
    let mut broadcast = TensorIr::new();
    let y = must!(broadcast.input_typed("y", vec![3, 1], TensorDType::F32));
    let x = must!(broadcast.input_typed("x", vec![4], TensorDType::F32));
    let angle = must!(broadcast.atan2(y, x));
    let ys = [1.0, -0.0, -3.5];
    let xs = [2.0, -1.0, 0.0, 1e-30];
    let inputs = BTreeMap::from([
        (
            "y".to_string(),
            must!(DynamicTensor::with_dtype(
                vec![3, 1],
                ys.to_vec(),
                TensorDType::F32
            )),
        ),
        (
            "x".to_string(),
            must!(DynamicTensor::with_dtype(
                vec![4],
                xs.to_vec(),
                TensorDType::F32
            )),
        ),
    ]);
    let value = must!(broadcast.evaluate(angle, &inputs));
    assert_eq!(value.shape(), &[3, 4]);
    assert_eq!(value.dtype(), TensorDType::F32);
    let round = |value: f64| f64::from(value as f32);
    let expected = ys
        .iter()
        .flat_map(|y| xs.iter().map(move |x| round(round(*y).atan2(round(*x)))))
        .collect::<Vec<_>>();
    assert_same_bits(&value.data(), &expected);

    // Scalar constants fold through the same kernels.
    let mut folded = TensorIr::new();
    let tiny = folded.scalar_constant(1e-10);
    let output = must!(folded.expm1(tiny));
    let value = must!(must!(folded.compile_cpu(output)).evaluate(&BTreeMap::new()));
    assert_eq!(value.data()[0], 1e-10_f64.exp_m1());

    // Bool operands are rejected like the other math ops.
    let mut rejected = TensorIr::new();
    let mask = must!(rejected.input_typed("mask", vec![2], TensorDType::Bool));
    assert!(rejected.erf(mask).is_err());
    assert!(rejected.atan2(mask, mask).is_err());
    assert!(rejected.cumsum(mask, 0, false).is_err());
}

#[test]
fn expm1_and_erf_derivatives_agree_across_routes() {
    let points = vec![-3.0, -1e-6, 0.0, 0.5, 2.0];
    let inputs = must!(shaped_inputs(&[("x", vec![points.len()], points.clone())]));
    let coefficient = std::f64::consts::FRAC_2_SQRT_PI;
    for erf in [false, true] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input("x", vec![points.len()]));
        let output = if erf {
            must!(graph.erf(x))
        } else {
            must!(graph.expm1(x))
        };
        let total = must!(graph.sum(output));
        let expected = points
            .iter()
            .map(|point| {
                if erf {
                    coefficient * (-point * point).exp()
                } else {
                    point.exp()
                }
            })
            .collect::<Vec<_>>();
        assert_gradient_routes(&graph, total, &inputs, &[("x", expected)], 1e-15);
    }

    // Second derivatives: exp(x) and -2x * erf'(x).
    for (point, erf) in [(0.3_f64, false), (0.7, true), (-1.3, true)] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input("x", vec![]));
        let output = if erf {
            must!(graph.erf(x))
        } else {
            must!(graph.expm1(x))
        };
        let inputs = must!(shaped_inputs(&[("x", vec![], vec![point])]));
        let expected = if erf {
            -2.0 * point * coefficient * (-point * point).exp()
        } else {
            point.exp()
        };
        assert_hessian_routes(&graph, output, "x", &inputs, &[expected], 1e-14);
    }
}

/// `atan2(p[0], p[1])` of a two-entry input `p`, so a Hessian in `p` holds
/// every second partial.
fn atan2_of_pair() -> Result<(TensorIr, TensorNodeId), String> {
    let mut graph = TensorIr::new();
    let p = graph.input("p", vec![2])?;
    let y = graph.slice_axis(p, 0, 0, 1)?;
    let x = graph.slice_axis(p, 0, 1, 2)?;
    let y = graph.reshape(y, vec![])?;
    let x = graph.reshape(x, vec![])?;
    let angle = graph.atan2(y, x)?;
    Ok((graph, angle))
}

#[test]
fn atan2_derivatives_avoid_overflow_and_vanish_at_the_origin() {
    // (y, x, d/dy, d/dx): the derivatives are x / r^2 and -y / r^2, written
    // out exactly where r^2 overflows or underflows in f64.
    let cases = [
        (1.0, 1.0, 0.5, -0.5),
        (-2.0, 0.5, 0.5 / 4.25, 2.0 / 4.25),
        (0.0, -3.0, -1.0 / 3.0, 0.0),
        (3.0, 0.0, 0.0, -1.0 / 3.0),
        (1e200, 1e200, 5e-201, -5e-201),
        (1e-200, -1e-200, -5e199, -5e199),
        (0.0, 0.0, 0.0, 0.0),
        (-0.0, -0.0, 0.0, 0.0),
        (f64::INFINITY, 1.0, 0.0, 0.0),
    ];
    let mut graph = TensorIr::new();
    let y = must!(graph.input("y", vec![cases.len()]));
    let x = must!(graph.input("x", vec![cases.len()]));
    let angle = must!(graph.atan2(y, x));
    let total = must!(graph.sum(angle));
    let inputs = must!(shaped_inputs(&[
        (
            "y",
            vec![cases.len()],
            cases.iter().map(|case| case.0).collect()
        ),
        (
            "x",
            vec![cases.len()],
            cases.iter().map(|case| case.1).collect()
        ),
    ]));
    assert_gradient_routes(
        &graph,
        total,
        &inputs,
        &[
            ("y", cases.iter().map(|case| case.2).collect()),
            ("x", cases.iter().map(|case| case.3).collect()),
        ],
        1e-14,
    );

    // A NaN operand gives a NaN derivative.
    let nan_inputs = must!(shaped_inputs(&[
        ("y", vec![cases.len()], vec![f64::NAN; cases.len()]),
        ("x", vec![cases.len()], vec![1.0; cases.len()]),
    ]));
    let one = must!(DynamicTensor::filled(vec![], 1.0));
    let gradient = must!(graph.vjp(total, &nan_inputs, one));
    assert!(gradient["x"].data().iter().all(|value| value.is_nan()));

    // Hessian [[-2xy, y^2 - x^2], [y^2 - x^2, 2xy]] / r^4, also where r^4
    // overflows, and zero at the origin.
    let (pair, angle) = must!(atan2_of_pair());
    for (y, x, expected) in [
        (0.8_f64, -1.5_f64, None),
        (
            1e150,
            2e150,
            Some([-1.6e-301, -1.2e-301, -1.2e-301, 1.6e-301]),
        ),
        (0.0, 0.0, Some([0.0; 4])),
    ] {
        let expected = expected.unwrap_or_else(|| {
            let r4 = (x * x + y * y).powi(2);
            let mixed = (y * y - x * x) / r4;
            [-2.0 * x * y / r4, mixed, mixed, 2.0 * x * y / r4]
        });
        let inputs = must!(shaped_inputs(&[("p", vec![2], vec![y, x])]));
        assert_hessian_routes(&pair, angle, "p", &inputs, &expected, 1e-13);
    }
}

/// The f64 function each `UnaryMathKind` must evaluate: Rust std, and the
/// musl ports of the `libm` crate for the inverse hyperbolic functions.
fn unary_math_reference(kind: UnaryMathKind) -> fn(f64) -> f64 {
    match kind {
        UnaryMathKind::Tan => f64::tan,
        UnaryMathKind::Arcsin => f64::asin,
        UnaryMathKind::Arccos => f64::acos,
        UnaryMathKind::Arctan => f64::atan,
        UnaryMathKind::Sinh => f64::sinh,
        UnaryMathKind::Cosh => f64::cosh,
        UnaryMathKind::Arcsinh => libm::asinh,
        UnaryMathKind::Arccosh => libm::acosh,
        UnaryMathKind::Arctanh => libm::atanh,
        UnaryMathKind::Log2 => f64::log2,
        UnaryMathKind::Log10 => f64::log10,
        UnaryMathKind::Cbrt => f64::cbrt,
        UnaryMathKind::Floor => f64::floor,
        UnaryMathKind::Ceil => f64::ceil,
        UnaryMathKind::Round => f64::round_ties_even,
    }
}

#[test]
fn unary_math_and_fmod_values_follow_the_f64_reference() {
    let points = SPECIAL_POINTS
        .iter()
        .copied()
        .chain([
            -27.0,
            -2.5,
            -1.5,
            -0.5,
            1.0 - f64::EPSILON / 2.0,
            1.0,
            1.0 + f64::EPSILON,
            1.5,
            2.5,
            1e22,
            1e300,
        ])
        .collect::<Vec<_>>();
    let count = points.len();
    for dtype in [TensorDType::F64, TensorDType::F32] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input_typed("x", vec![count], dtype));
        let inputs = BTreeMap::from([(
            "x".to_string(),
            must!(DynamicTensor::with_dtype(
                vec![count],
                points.clone(),
                dtype
            )),
        )]);
        // float32 rounds the input and then the f64 result, once each.
        let round = |value: f64| match dtype {
            TensorDType::F32 => f64::from(value as f32),
            _ => value,
        };
        for kind in UnaryMathKind::ALL {
            let output = must!(graph.unary_math(x, kind));
            let reference = unary_math_reference(kind);
            let expected = points
                .iter()
                .map(|point| round(reference(round(*point))))
                .collect::<Vec<_>>();
            let value = must!(graph.evaluate(output, &inputs));
            assert_eq!(value.dtype(), dtype);
            assert_same_bits(&value.data(), &expected);
            let plan = must!(graph.compile_cpu(output));
            assert_same_bits(&must!(plan.evaluate(&inputs)).data(), &expected);
            assert!(graph.lower_text().contains(&format!("{}(%", kind.name())));
        }
    }

    // round is half-to-even; cbrt is the real cube root; poles give
    // infinities and the outside of a domain NaN.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![6]));
    let inputs = must!(shaped_inputs(&[(
        "x",
        vec![6],
        vec![-2.5, -0.5, 0.5, 1.5, 2.5, -27.0]
    )]));
    let rounded = must!(graph.unary_math(x, UnaryMathKind::Round));
    assert_same_bits(
        &must!(graph.evaluate(rounded, &inputs)).data(),
        &[-2.0, -0.0, 0.0, 2.0, 2.0, -27.0],
    );
    let root = must!(graph.unary_math(x, UnaryMathKind::Cbrt));
    assert_eq!(must!(graph.evaluate(root, &inputs)).data()[5], -3.0);
    let poles = must!(shaped_inputs(&[(
        "x",
        vec![6],
        vec![1.0, -1.0, 0.0, 2.0, -2.0, 0.5]
    )]));
    let atanh = must!(graph.unary_math(x, UnaryMathKind::Arctanh));
    let value = must!(graph.evaluate(atanh, &poles));
    let value = value.data();
    assert_eq!(&value[..3], &[f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    assert!(value[3].is_nan() && value[4].is_nan());
    let acosh = must!(graph.unary_math(x, UnaryMathKind::Arccosh));
    assert!(must!(graph.evaluate(acosh, &poles)).data()[5].is_nan());

    // Two kinds of the same input stay distinct under CSE: the kind is part
    // of the key.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![]));
    let tan = must!(graph.unary_math(x, UnaryMathKind::Tan));
    let sinh = must!(graph.unary_math(x, UnaryMathKind::Sinh));
    let total = must!(graph.add(tan, sinh));
    let inputs = must!(shaped_inputs(&[("x", vec![], vec![0.5])]));
    let plan = must!(graph.compile_cpu(total));
    assert_eq!(
        must!(plan.evaluate(&inputs)).data()[0],
        0.5_f64.tan() + 0.5_f64.sinh()
    );

    // The ops fuse into one CUDA elementwise kernel through their CUDA math
    // functions; `rintf` is the halfway-to-even round.
    let mut fused = TensorIr::new();
    let x = must!(fused.input("x", vec![3]));
    let one = fused.scalar_constant(1.0);
    // arccosh takes x + 1, inside its domain [1, inf) like x in [-1, 1] is
    // inside the others'.
    let shifted = must!(fused.add(x, one));
    let mut total = x;
    for kind in UnaryMathKind::ALL {
        let operand = if kind == UnaryMathKind::Arccosh {
            shifted
        } else {
            x
        };
        let value = must!(fused.unary_math(operand, kind));
        total = must!(fused.add(total, value));
    }
    let divisor = fused.scalar_constant(0.75);
    let output = must!(fused.fmod(total, divisor));
    let plan = must!(fused.compile_cpu(output));
    assert!(plan.uses_fused_elementwise_kernel());
    let inputs = must!(shaped_inputs(&[("x", vec![3], vec![0.25, 0.5, 0.75])]));
    let value = must!(plan.evaluate(&inputs));
    assert!(value.data().iter().all(|entry| entry.is_finite()));
    assert_eq!(value, must!(fused.evaluate(output, &inputs)));
    let source = must!(plan.cuda_source());
    for function in [
        "tanf(", "asinf(", "acosf(", "atanf(", "sinhf(", "coshf(", "asinhf(", "acoshf(", "atanhf(",
        "log2f(", "log10f(", "cbrtf(", "floorf(", "ceilf(", "rintf(", "fmodf(",
    ] {
        assert!(
            source.contains(function),
            "{function} is missing from {source}"
        );
    }

    // fmod is C fmod (`%` on f64) for every pair of special points, with
    // broadcasting.
    let pairs = SPECIAL_POINTS.len() * SPECIAL_POINTS.len();
    let mut graph = TensorIr::new();
    let xs = must!(graph.input("xs", vec![pairs]));
    let ys = must!(graph.input("ys", vec![pairs]));
    let remainder = must!(graph.fmod(xs, ys));
    let numerators = (0..pairs)
        .map(|index| SPECIAL_POINTS[index / SPECIAL_POINTS.len()])
        .collect::<Vec<_>>();
    let divisors = (0..pairs)
        .map(|index| SPECIAL_POINTS[index % SPECIAL_POINTS.len()])
        .collect::<Vec<_>>();
    let expected = numerators
        .iter()
        .zip(&divisors)
        .map(|(x, y)| x % y)
        .collect::<Vec<_>>();
    let inputs = must!(shaped_inputs(&[
        ("xs", vec![pairs], numerators),
        ("ys", vec![pairs], divisors),
    ]));
    assert_same_bits(&must!(graph.evaluate(remainder, &inputs)).data(), &expected);
    let plan = must!(graph.compile_cpu(remainder));
    assert_same_bits(&must!(plan.evaluate(&inputs)).data(), &expected);
    assert!(graph.lower_text().contains("fmod(%"));
    let mut broadcast = TensorIr::new();
    let x = must!(broadcast.input("x", vec![2, 1]));
    let y = must!(broadcast.input("y", vec![3]));
    let remainder = must!(broadcast.fmod(x, y));
    let inputs = must!(shaped_inputs(&[
        ("x", vec![2, 1], vec![7.5, -7.5]),
        ("y", vec![3], vec![2.0, -2.0, 4.0]),
    ]));
    let value = must!(broadcast.evaluate(remainder, &inputs));
    assert_eq!(value.shape(), &[2, 3]);
    assert_same_bits(&value.data(), &[1.5, 1.5, 3.5, -1.5, -1.5, -3.5]);

    // Scalar constants fold through the same kernels.
    let mut folded = TensorIr::new();
    let half = folded.scalar_constant(0.5);
    let three = folded.scalar_constant(3.0);
    let asin = must!(folded.unary_math(half, UnaryMathKind::Arcsin));
    let remainder = must!(folded.fmod(asin, three));
    let value = must!(must!(folded.compile_cpu(remainder)).evaluate(&BTreeMap::new()));
    assert_eq!(value.data()[0], 0.5_f64.asin() % 3.0);

    // Bool operands are rejected like the other math ops.
    let mut rejected = TensorIr::new();
    let mask = must!(rejected.input_typed("mask", vec![2], TensorDType::Bool));
    assert!(rejected.unary_math(mask, UnaryMathKind::Floor).is_err());
    assert!(rejected.fmod(mask, mask).is_err());
}

/// `(kind, points, first derivatives, second derivative at points[0])`.
type UnaryMathDerivativeCase = (UnaryMathKind, Vec<f64>, fn(f64) -> f64, fn(f64) -> f64);

#[test]
fn unary_math_derivatives_agree_across_routes() {
    let cases: Vec<UnaryMathDerivativeCase> = vec![
        (
            UnaryMathKind::Tan,
            vec![0.4, -1.3, 0.0, 1.5],
            |x| 1.0 + x.tan() * x.tan(),
            |x| 2.0 * x.tan() * (1.0 + x.tan() * x.tan()),
        ),
        (
            UnaryMathKind::Arcsin,
            vec![0.6, -0.999, 0.0, 0.3],
            |x| 1.0 / ((1.0 - x) * (1.0 + x)).sqrt(),
            |x| x / ((1.0 - x) * (1.0 + x)).powf(1.5),
        ),
        (
            UnaryMathKind::Arccos,
            vec![-0.6, 0.999, 0.0, 0.3],
            |x| -1.0 / ((1.0 - x) * (1.0 + x)).sqrt(),
            |x| -x / ((1.0 - x) * (1.0 + x)).powf(1.5),
        ),
        (
            UnaryMathKind::Arctan,
            vec![2.0, -30.0, 0.0, 0.5],
            |x| 1.0 / (1.0 + x * x),
            |x| -2.0 * x / (1.0 + x * x).powi(2),
        ),
        (
            UnaryMathKind::Sinh,
            vec![1.5, -5.0, 0.0, 20.0],
            f64::cosh,
            f64::sinh,
        ),
        (
            UnaryMathKind::Cosh,
            vec![1.5, -5.0, 0.0, 20.0],
            f64::sinh,
            f64::cosh,
        ),
        (
            UnaryMathKind::Arcsinh,
            vec![-3.0, 0.5, 0.0, 40.0],
            |x| 1.0 / (x * x + 1.0).sqrt(),
            |x| -x / (x * x + 1.0).powf(1.5),
        ),
        (
            UnaryMathKind::Arccosh,
            vec![1.5, 1.001, 10.0, 1e5],
            |x| 1.0 / ((x - 1.0) * (x + 1.0)).sqrt(),
            |x| -x / ((x - 1.0) * (x + 1.0)).powf(1.5),
        ),
        (
            UnaryMathKind::Arctanh,
            vec![0.6, -0.999, 0.0, 0.3],
            |x| 1.0 / ((1.0 - x) * (1.0 + x)),
            |x| 2.0 * x / ((1.0 - x) * (1.0 + x)).powi(2),
        ),
        (
            UnaryMathKind::Log2,
            vec![3.0, 1e-3, 1.0, 1e4],
            |x| 1.0 / (x * std::f64::consts::LN_2),
            |x| -1.0 / (x * x * std::f64::consts::LN_2),
        ),
        (
            UnaryMathKind::Log10,
            vec![3.0, 1e-3, 1.0, 1e4],
            |x| 1.0 / (x * std::f64::consts::LN_10),
            |x| -1.0 / (x * x * std::f64::consts::LN_10),
        ),
        (
            UnaryMathKind::Cbrt,
            vec![-0.2, -27.0, 1e-6, 8.0],
            |x| 1.0 / (3.0 * x.cbrt() * x.cbrt()),
            |x| -2.0 / (9.0 * x.cbrt().powi(5)),
        ),
    ];
    for (kind, points, first, second) in cases {
        let mut graph = TensorIr::new();
        let x = must!(graph.input("x", vec![points.len()]));
        let output = must!(graph.unary_math(x, kind));
        let total = must!(graph.sum(output));
        let inputs = must!(shaped_inputs(&[("x", vec![points.len()], points.clone())]));
        let expected = points.iter().map(|point| first(*point)).collect::<Vec<_>>();
        assert_gradient_routes(&graph, total, &inputs, &[("x", expected)], 2e-15);

        let mut scalar = TensorIr::new();
        let x = must!(scalar.input("x", vec![]));
        let output = must!(scalar.unary_math(x, kind));
        let inputs = must!(shaped_inputs(&[("x", vec![], vec![points[0]])]));
        assert_hessian_routes(&scalar, output, "x", &inputs, &[second(points[0])], 1e-14);
    }

    // floor, ceil and round have a zero derivative of every order, also at
    // their jumps.
    let points = vec![-1.5, -0.5, 0.0, 0.5, 2.0];
    let inputs = must!(shaped_inputs(&[("x", vec![5], points)]));
    for kind in [
        UnaryMathKind::Floor,
        UnaryMathKind::Ceil,
        UnaryMathKind::Round,
    ] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input("x", vec![5]));
        let output = must!(graph.unary_math(x, kind));
        let total = must!(graph.sum(output));
        assert_gradient_routes(&graph, total, &inputs, &[("x", vec![0.0; 5])], 0.0);
        assert_hessian_routes(&graph, total, "x", &inputs, &[0.0; 25], 0.0);
    }

    // arcsinh' does not overflow for large |x| and is 0 at +-inf.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![5]));
    let output = must!(graph.unary_math(x, UnaryMathKind::Arcsinh));
    let total = must!(graph.sum(output));
    let inputs = must!(shaped_inputs(&[(
        "x",
        vec![5],
        vec![1e200, -1e300, f64::INFINITY, f64::NEG_INFINITY, 0.0]
    )]));
    assert_gradient_routes(
        &graph,
        total,
        &inputs,
        &[("x", vec![1e-200, 1e-300, 0.0, 0.0, 1.0])],
        1e-15,
    );
}

#[test]
fn fmod_derivatives_are_one_and_minus_the_truncated_quotient() {
    // (x, y, -trunc(x / y)) with the quotient of the exact remainder:
    // fmod(0.3, 0.1) = 0.1 - 2.8e-17, so its quotient is 2.
    let cases = [
        (5.5, 2.0, -2.0),
        (-5.5, 2.0, 2.0),
        (7.0, -3.0, 2.0),
        (-7.25, -3.0, -2.0),
        (0.3, 0.1, -2.0),
        (1e-30, 1.0, -0.0),
        (4.0, f64::INFINITY, -0.0),
    ];
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![cases.len()]));
    let y = must!(graph.input("y", vec![cases.len()]));
    let remainder = must!(graph.fmod(x, y));
    let total = must!(graph.sum(remainder));
    let inputs = must!(shaped_inputs(&[
        (
            "x",
            vec![cases.len()],
            cases.iter().map(|case| case.0).collect()
        ),
        (
            "y",
            vec![cases.len()],
            cases.iter().map(|case| case.1).collect()
        ),
    ]));
    assert_gradient_routes(
        &graph,
        total,
        &inputs,
        &[
            ("x", vec![1.0; cases.len()]),
            ("y", cases.iter().map(|case| case.2).collect()),
        ],
        0.0,
    );

    // Piecewise linear: every second partial is zero.
    let mut pair = TensorIr::new();
    let p = must!(pair.input("p", vec![2]));
    let x = must!(pair.slice_axis(p, 0, 0, 1));
    let y = must!(pair.slice_axis(p, 0, 1, 2));
    let remainder = must!(pair.fmod(x, y));
    let output = must!(pair.sum(remainder));
    let inputs = must!(shaped_inputs(&[("p", vec![2], vec![7.0, -3.0])]));
    assert_hessian_routes(&pair, output, "p", &inputs, &[0.0; 4], 0.0);

    // A scalar constant divisor receives no cotangent and the other operand
    // still differentiates.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let divisor = graph.scalar_constant(2.0);
    let remainder = must!(graph.fmod(x, divisor));
    let total = must!(graph.sum(remainder));
    let inputs = must!(shaped_inputs(&[("x", vec![2], vec![5.5, -3.0])]));
    assert_gradient_routes(&graph, total, &inputs, &[("x", vec![1.0, 1.0])], 0.0);
}

#[test]
fn stop_gradient_passes_values_and_blocks_every_derivative() {
    let points = vec![0.5, -2.0, 3.0];
    let inputs = must!(shaped_inputs(&[("x", vec![3], points.clone())]));
    // sum(x * stop_gradient(x)) has gradient x and a zero Hessian.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let stopped = must!(graph.stop_gradient(x));
    let product = must!(graph.mul(x, stopped));
    let total = must!(graph.sum(product));
    let value = must!(graph.evaluate(total, &inputs));
    assert_eq!(value.data()[0], 0.25 + 4.0 + 9.0);
    assert_gradient_routes(&graph, total, &inputs, &[("x", points.clone())], 0.0);
    assert_hessian_routes(&graph, total, "x", &inputs, &[0.0; 9], 0.0);
    assert!(graph.lower_text().contains("stop_gradient(%"));

    // x - stop_gradient(x) is zero with gradient one.
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![3]));
    let stopped = must!(graph.stop_gradient(x));
    let difference = must!(graph.sub(x, stopped));
    let total = must!(graph.sum(difference));
    assert_eq!(must!(graph.evaluate(total, &inputs)).data()[0], 0.0);
    assert_gradient_routes(&graph, total, &inputs, &[("x", vec![1.0; 3])], 0.0);

    // Bool values pass through as data movement.
    let mut masks = TensorIr::new();
    let mask = must!(masks.input_typed("mask", vec![2], TensorDType::Bool));
    let stopped = must!(masks.stop_gradient(mask));
    let value = must!(masks.evaluate(
        stopped,
        &BTreeMap::from([(
            "mask".to_string(),
            must!(DynamicTensor::with_dtype(
                vec![2],
                vec![1.0, 0.0],
                TensorDType::Bool
            )),
        )]),
    ));
    assert_eq!(value.dtype(), TensorDType::Bool);
    assert_eq!(value.data().as_ref(), &[1.0, 0.0]);
}

#[test]
fn cumsum_rounds_every_running_sum_and_differentiates_by_reversal() {
    // float32: each running sum rounds, so 3e-8 increments never move 1.0,
    // while a single rounding of the exact sum would.
    let mut graph = TensorIr::new();
    let x = must!(graph.input_typed("x", vec![2, 4], TensorDType::F32));
    let forward = must!(graph.cumsum(x, 1, false));
    let backward = must!(graph.cumsum(x, -1, true));
    let down = must!(graph.cumsum(x, 0, false));
    let values = vec![1.0, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 1.0];
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::with_dtype(
            vec![2, 4],
            values.clone(),
            TensorDType::F32
        )),
    )]);
    let f32_values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let scan = |order: &[usize]| {
        let mut data = vec![0.0_f64; 8];
        for row in 0..2 {
            let mut running = 0.0_f32;
            for (step, column) in order.iter().enumerate() {
                let entry = f32_values[row * 4 + column];
                running = if step == 0 { entry } else { running + entry };
                data[row * 4 + column] = f64::from(running);
            }
        }
        data
    };
    let expected_forward = scan(&[0, 1, 2, 3]);
    let expected_backward = scan(&[3, 2, 1, 0]);
    assert_eq!(expected_forward[3], 1.0);
    let expected_down = (0..8)
        .map(|index| {
            if index < 4 {
                f64::from(f32_values[index])
            } else {
                f64::from(f32_values[index - 4] + f32_values[index])
            }
        })
        .collect::<Vec<_>>();
    for (output, expected) in [
        (forward, expected_forward),
        (backward, expected_backward),
        (down, expected_down),
    ] {
        assert_same_bits(&must!(graph.evaluate(output, &inputs)).data(), &expected);
        let plan = must!(graph.compile_cpu(output));
        assert_same_bits(&must!(plan.evaluate(&inputs)).data(), &expected);
    }
    assert!(graph
        .lower_text()
        .contains("cumsum(%0, axis=1, reverse=true)"));

    // d sum(cumsum(x) * w) / dx is the opposite-direction cumsum of w.
    let weights = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, 2.0];
    for (axis, reverse) in [(1, false), (1, true), (0, false), (0, true)] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input("x", vec![2, 4]));
        let w = must!(graph.input("w", vec![2, 4]));
        let scanned = must!(graph.cumsum(x, axis, reverse));
        let weighted = must!(graph.mul(scanned, w));
        let total = must!(graph.sum(weighted));
        let inputs = must!(shaped_inputs(&[
            (
                "x",
                vec![2, 4],
                (0..8).map(|index| 0.3 * index as f64 - 1.0).collect()
            ),
            ("w", vec![2, 4], weights.clone()),
        ]));
        let expected = (0..8)
            .map(|index| {
                let (row, column) = (index / 4, index % 4);
                let (position, extent) = if axis == 1 { (column, 4) } else { (row, 2) };
                // Entry `index` reaches every output at or after it in scan order.
                (0..extent)
                    .filter(|other| {
                        if reverse {
                            *other <= position
                        } else {
                            *other >= position
                        }
                    })
                    .map(|other| {
                        if axis == 1 {
                            weights[row * 4 + other]
                        } else {
                            weights[other * 4 + column]
                        }
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        assert_gradient_routes(&graph, total, &inputs, &[("x", expected)], 1e-15);
    }

    // sum(cumsum(x)^2) has Hessian 2 L^T L, with L the lower (forward) or
    // upper (reverse) triangle of ones.
    for (reverse, expected) in [
        (false, [6.0, 4.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0]),
        (true, [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 2.0, 4.0, 6.0]),
    ] {
        let mut graph = TensorIr::new();
        let x = must!(graph.input("x", vec![3]));
        let scanned = must!(graph.cumsum(x, 0, reverse));
        let squared = must!(graph.powi(scanned, 2));
        let total = must!(graph.sum(squared));
        let inputs = must!(shaped_inputs(&[("x", vec![3], vec![0.5, -1.0, 2.0])]));
        assert_hessian_routes(&graph, total, "x", &inputs, &expected, 0.0);
    }
}

/// A callee over the [`batching_inputs`] example shapes that uses every new
/// op with mapped operands, including an `atan2` of a mapped and an
/// unmapped operand of different ranks.
fn new_op_batching_fixture() -> Result<(TensorIr, Vec<TensorNodeId>, TensorNodeId), String> {
    let mut callee = TensorIr::new();
    let x = callee.input("x", vec![])?;
    let v = callee.input("v", vec![3])?;
    let w = callee.input("w", vec![3])?;
    callee.input("m", vec![3, 2])?;
    let a = callee.input("a", vec![2, 2])?;
    let grown = callee.expm1(v)?;
    let smoothed = callee.erf(v)?;
    let angle = callee.atan2(v, w)?;
    let scalar_angle = callee.atan2(x, w)?;
    let stopped = callee.stop_gradient(a)?;
    let rows = callee.cumsum(a, 1, true)?;
    let columns = callee.cumsum(a, 0, false)?;
    let product = callee.mul(stopped, a)?;
    let sums = [grown, smoothed, angle, scalar_angle, rows, columns, product]
        .into_iter()
        .map(|output| callee.sum(output))
        .collect::<Result<Vec<_>, _>>()?;
    let mut total = sums[0];
    for sum in &sums[1..] {
        total = callee.add(total, *sum)?;
    }
    let outputs = vec![grown, smoothed, angle, scalar_angle, stopped, rows, columns];
    Ok((callee, outputs, total))
}

#[test]
fn inline_batched_maps_the_new_ops_and_their_gradients() {
    let (callee, outputs, total) = must!(new_op_batching_fixture());
    assert_batched_matches_examples(&callee, &outputs, &[true; 7]);
    let vjp = must!(callee.symbolic_vjp_many(&[(total, SymbolicCotangent::Ones)]));
    let names = ["x", "v", "w", "a"];
    let outputs = names
        .iter()
        .map(|name| vjp.gradients[*name])
        .collect::<Vec<_>>();
    assert_batched_matches_examples(&vjp.graph, &outputs, &[true; 4]);
}

/// Device plans for the new ops: the gradient of a loss that uses all of
/// them (per-node kernels), their forward values, and a Fori whose body
/// applies the elementwise ones (loop-body lowering).
#[cfg(any(
    all(feature = "cuda", target_os = "linux"),
    all(feature = "mlx", target_os = "macos")
))]
fn new_op_device_plans() -> Result<Vec<PlanWithInputs>, String> {
    let points = vec![-2.5, -0.5, -1e-4, 0.0, 0.3, 1.7, 4.0];
    let weights = vec![1.0, -0.5, 2.0, 0.0, -3.0, 0.25, 1e-3];
    let count = points.len();
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![count])?;
    let w = graph.input("w", vec![count])?;
    let grown = graph.expm1(x)?;
    let smoothed = graph.erf(x)?;
    let angle = graph.atan2(x, w)?;
    let stopped = graph.stop_gradient(x)?;
    let scaled = graph.mul(x, stopped)?;
    let scanned = graph.cumsum(scaled, 0, true)?;
    let mut total = graph.mul(grown, x)?;
    for term in [smoothed, angle, scanned] {
        total = graph.add(total, term)?;
    }
    let loss = graph.sum(total)?;
    let reverse = graph.symbolic_vjp(loss, "cotangent")?;
    let gradient_inputs = BTreeMap::from([
        (
            "x".to_string(),
            DynamicTensor::new(vec![count], points.clone())?,
        ),
        (
            "w".to_string(),
            DynamicTensor::new(vec![count], weights.clone())?,
        ),
        ("cotangent".to_string(), DynamicTensor::filled(vec![], 1.0)?),
    ]);
    let mut plans = vec![
        (
            reverse.graph.compile_cpu(reverse.gradients["x"])?,
            gradient_inputs.clone(),
        ),
        (
            reverse.graph.compile_cpu(reverse.gradients["w"])?,
            gradient_inputs.clone(),
        ),
    ];
    for output in [grown, smoothed, angle, scanned] {
        plans.push((graph.compile_cpu(output)?, gradient_inputs.clone()));
    }

    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![count])?;
    let index = body.input("index", vec![])?;
    let grown = body.expm1(carry)?;
    let smoothed = body.erf(grown)?;
    let stopped = body.stop_gradient(smoothed)?;
    let angle = body.atan2(stopped, index)?;
    let loop_plan = TensorForiExecutionPlan::new(1, 4, body.compile_cpu(angle)?, "carry", "index")?;
    let mut looped = TensorIr::new();
    let initial = looped.input("x", vec![count])?;
    let output = looped.fori(initial, loop_plan, vec![])?;
    let loop_inputs = BTreeMap::from([("x".to_string(), DynamicTensor::new(vec![count], points)?)]);
    plans.push((looped.compile_cpu(output)?, loop_inputs));
    Ok(plans)
}

/// A float32 `cumsum` long enough that its rounding order matters, along
/// both axes and directions.
#[cfg(any(
    all(feature = "cuda", target_os = "linux"),
    all(feature = "mlx", target_os = "macos")
))]
fn cumsum_device_plans() -> Result<Vec<PlanWithInputs>, String> {
    let shape = vec![3, 300];
    let values = (0..900)
        .map(|index| ((index * 7919 % 1000) as f64 - 500.0) * 1.37e-3 + 1.0 / (1 + index) as f64)
        .collect::<Vec<_>>();
    let inputs = BTreeMap::from([(
        "x".to_string(),
        DynamicTensor::with_dtype(shape.clone(), values, TensorDType::F32)?,
    )]);
    let mut plans = Vec::new();
    for (axis, reverse) in [(1, false), (1, true), (0, false), (0, true)] {
        let mut graph = TensorIr::new();
        let x = graph.input_typed("x", shape.clone(), TensorDType::F32)?;
        let scanned = graph.cumsum(x, axis, reverse)?;
        plans.push((graph.compile_cpu(scanned)?, inputs.clone()));
    }
    Ok(plans)
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn mlx_new_ops_match_cpu_values_and_gradients() {
    if std::env::var_os("QUABLA_MLX_TEST").is_none() {
        return;
    }
    for (plan, inputs) in must!(new_op_device_plans()) {
        let cpu = must!(plan.evaluate(&inputs));
        let mlx = must!(MlxBackend.execute(&plan, &inputs));
        assert_eq!(mlx.shape(), cpu.shape());
        for (device, host) in mlx.data().iter().zip(cpu.data().iter()) {
            assert!(
                (device - host).abs() <= 4e-6 * host.abs().max(1.0),
                "{device} vs {host}"
            );
        }
    }
    // MLX scans in parallel, so float32 prefix sums agree only to rounding.
    for (plan, inputs) in must!(cumsum_device_plans()) {
        let cpu = must!(plan.evaluate(&inputs));
        let mlx = must!(MlxBackend.execute(&plan, &inputs));
        assert_eq!(mlx.shape(), cpu.shape());
        for (device, host) in mlx.data().iter().zip(cpu.data().iter()) {
            assert!(
                (device - host).abs() <= 1e-5 * host.abs().max(1.0),
                "{device} vs {host}"
            );
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_new_ops_match_cpu_values_and_gradients() {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return;
    }
    for (plan, inputs) in must!(new_op_device_plans()) {
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        for (device, host) in cuda.data().iter().zip(cpu.data().iter()) {
            assert!(
                (device - host).abs() <= 4e-6 * host.abs().max(1.0),
                "{device} vs {host}"
            );
        }
    }
    // One thread scans each line with __fadd_rn: bitwise the CPU reference.
    for (plan, inputs) in must!(cumsum_device_plans()) {
        let cpu = must!(plan.evaluate(&inputs));
        let cuda = must!(CudaBackend::new(0).execute(&plan, &inputs));
        assert_eq!(cuda.shape(), cpu.shape());
        assert_eq!(cuda.data().as_ref(), cpu.data().as_ref());
    }
}

// ---- While: traced-predicate loop regions ----

type WhileGraph = (TensorIr, TensorNodeId, BTreeMap<String, DynamicTensor>);

/// `carry * scale` while `carry < limit`; from 1 with scale 2 and limit 10
/// the body runs four times and returns 16.
fn doubling_while_graph() -> Result<WhileGraph, String> {
    let mut predicate = TensorIr::new();
    let carry = predicate.input("carry", vec![])?;
    let limit = predicate.input("limit", vec![])?;
    let keep_going = predicate.compare(carry, limit, TensorComparison::Less)?;
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![])?;
    let scale = body.input("scale", vec![])?;
    let next = body.mul(carry, scale)?;
    let loop_plan = quabla_core::tensor_ir::TensorWhileExecutionPlan::new(
        predicate.compile_cpu(keep_going)?,
        body.compile_cpu(next)?,
        "carry",
    )?;
    // The body ignores `limit` and the predicate ignores `scale`; both bind.
    assert_eq!(loop_plan.external_captures().len(), 2);

    let mut graph = TensorIr::new();
    let initial = graph.input("initial", vec![])?;
    let scale = graph.input("scale", vec![])?;
    let limit = graph.input("limit", vec![])?;
    let output = graph.while_loop(
        initial,
        loop_plan,
        vec![("scale".to_string(), scale), ("limit".to_string(), limit)],
    )?;
    let inputs = BTreeMap::from([
        (
            "initial".to_string(),
            DynamicTensor::new(vec![], vec![1.0])?,
        ),
        ("scale".to_string(), DynamicTensor::new(vec![], vec![2.0])?),
        ("limit".to_string(), DynamicTensor::new(vec![], vec![10.0])?),
    ]);
    Ok((graph, output, inputs))
}

#[test]
fn while_region_evaluates_forward_mode_and_rejects_reverse_mode() {
    let (graph, output, inputs) = must!(doubling_while_graph());
    assert!(graph.lower_text().contains("while(carry="));
    assert_eq!(
        must!(graph.evaluate(output, &inputs)).data().as_ref(),
        &[16.0]
    );
    assert_eq!(
        must!(must!(graph.compile_cpu(output)).evaluate(&inputs))
            .data()
            .as_ref(),
        &[16.0]
    );
    // d(c0 * s^4) = s^4 dc0 + 4 c0 s^3 ds.
    let scalar = |value: f64| DynamicTensor::new(vec![], vec![value]);
    let (_, tangent) = must!(graph.jvp(
        output,
        &inputs,
        &BTreeMap::from([
            ("initial".to_string(), must!(scalar(1.0))),
            ("scale".to_string(), must!(scalar(1.0))),
            ("limit".to_string(), must!(scalar(0.0))),
        ]),
    ));
    assert_eq!(tangent.data().as_ref(), &[48.0]);

    let symbolic = must!(graph.symbolic_jvp(output, "scale"));
    assert!(symbolic.graph.lower_text().contains("while(carry="));
    let tangent = must!(symbolic
        .graph
        .compile_cpu(symbolic.tangent)
        .and_then(|plan| plan.evaluate(&inputs)));
    assert_eq!(tangent.data().as_ref(), &[32.0]);

    let reverse = graph
        .value_and_vjp(output, &inputs, must!(scalar(1.0)))
        .map(|_| ())
        .unwrap_err();
    assert!(reverse.contains("fori_loop"), "{reverse}");
    let symbolic = graph
        .symbolic_vjp(output, "cotangent")
        .map(|_| ())
        .unwrap_err();
    assert!(symbolic.contains("while_loop"), "{symbolic}");
}

#[test]
fn while_region_requires_a_scalar_bool_predicate() {
    let mut predicate = TensorIr::new();
    let carry = must!(predicate.input("carry", vec![]));
    let mut body = TensorIr::new();
    let body_carry = must!(body.input("carry", vec![]));
    let error = quabla_core::tensor_ir::TensorWhileExecutionPlan::new(
        must!(predicate.compile_cpu(carry)),
        must!(body.compile_cpu(body_carry)),
        "carry",
    )
    .map(|_| ())
    .unwrap_err();
    assert!(error.contains("scalar bool"), "{error}");
}

/// A `custom_vjp`-style rule for `x * x` whose backward graph returns
/// `3 * g` instead of the true derivative, so a test can tell the rule
/// from the primal derivative. The residual is the operand.
fn custom_square_rule() -> Result<std::sync::Arc<TensorCustomRule>, String> {
    let mut forward = TensorIr::new();
    let x = forward.input("x", vec![2])?;
    let square = forward.mul(x, x)?;
    let mut backward = TensorIr::new();
    backward.input("residual", vec![2])?;
    let g = backward.input("g", vec![2])?;
    let three = backward.scalar_constant(3.0);
    let scaled = backward.mul(g, three)?;
    Ok(std::sync::Arc::new(TensorCustomRule::new(
        "custom_square".to_string(),
        forward,
        vec!["x".to_string()],
        1,
        vec![square, x],
        backward,
        vec!["residual".to_string()],
        vec!["g".to_string()],
        vec![Some(scaled)],
        None,
        false,
    )?))
}

#[test]
fn custom_rule_node_is_its_value_and_reverse_mode_applies_the_rule() {
    let rule = must!(custom_square_rule());
    let mut graph = TensorIr::new();
    let x = must!(graph.input("x", vec![2]));
    let value = must!(graph.mul(x, x));
    let wrapped = must!(graph.custom(rule.clone(), &[value], &[x]));
    let loss = must!(graph.sum(wrapped[0]));
    let inputs = BTreeMap::from([(
        "x".to_string(),
        must!(DynamicTensor::new(vec![2], vec![2.0, -1.0])),
    )]);

    // Plan compilation aliases the node to its value.
    let plan = must!(graph.compile_cpu(wrapped[0]));
    assert!(
        !plan.lower_text().contains("custom"),
        "{}",
        plan.lower_text()
    );
    assert_eq!(
        must!(plan.evaluate(&inputs)).data().to_vec(),
        vec![4.0, 1.0]
    );

    let vjp = must!(graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)]));
    let gradient = vjp.gradients["x"];
    let plan = must!(vjp.graph.compile_cpu(gradient));
    assert_eq!(
        must!(plan.evaluate(&inputs)).data().to_vec(),
        vec![3.0, 3.0]
    );

    let tangents = BTreeMap::from([("x".to_string(), "dx".to_string())]);
    let error = graph
        .symbolic_jvp_many_with_tangent_inputs(&[loss], &tangents)
        .err()
        .unwrap_or_default();
    assert!(error.contains("forward-mode differentiation"), "{error}");
    assert!(graph.has_reverse_only_custom_rule());
    assert_eq!(
        must!(graph.custom_rule_name(&[loss])).as_deref(),
        Some("custom_square")
    );
}

#[test]
fn custom_rule_node_batches_with_its_rule() {
    let rule = must!(custom_square_rule());
    let mut callee = TensorIr::new();
    let x = must!(callee.input("x", vec![2]));
    let value = must!(callee.mul(x, x));
    let wrapped = must!(callee.custom(rule, &[value], &[x]));

    let mut graph = TensorIr::new();
    let xs = must!(graph.input("xs", vec![3, 2]));
    let bindings = BTreeMap::from([("x".to_string(), (xs, true))]);
    let batched = must!(graph.inline_batched(&callee, &bindings, 3, &wrapped));
    assert!(batched[0].1);
    let loss = must!(graph.sum(batched[0].0));
    let vjp = must!(graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)]));
    let plan = must!(vjp.graph.compile_cpu(vjp.gradients["xs"]));
    let inputs = BTreeMap::from([(
        "xs".to_string(),
        must!(DynamicTensor::new(
            vec![3, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )),
    )]);
    assert_eq!(must!(plan.evaluate(&inputs)).data().to_vec(), vec![3.0; 6]);
}

#[test]
fn custom_rule_rejects_mismatched_graphs() {
    let mut forward = TensorIr::new();
    let x = must!(forward.input("x", vec![2]));
    let mut backward = TensorIr::new();
    must!(backward.input("g", vec![2]));
    let wrong = must!(backward.input("h", vec![3]));
    let error = TensorCustomRule::new(
        "bad".to_string(),
        forward,
        vec!["x".to_string()],
        1,
        vec![x],
        backward,
        vec![],
        vec!["g".to_string()],
        vec![Some(wrong)],
        None,
        false,
    )
    .err()
    .unwrap_or_default();
    assert!(error.contains("cotangent for operand 0"), "{error}");
}
