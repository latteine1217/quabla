#![cfg(all(feature = "cuda", target_os = "linux"))]
//! The opt-in `QuablaPrecision::Float64` CUDA lowering against the CPU.

use std::collections::BTreeMap;

use quabla_core::compiler::QuablaCompileError;
use quabla_core::tensor_ir::{DynamicTensor, TensorDType, TensorIr};
use quabla_core::{QuablaCompiler, QuablaMultiOutputProgram, QuablaPrecision, QuablaTarget};

const CUDA: QuablaTarget = QuablaTarget::Cuda { device_ordinal: 0 };

fn run(
    graph: TensorIr,
    outputs: Vec<usize>,
    target: QuablaTarget,
    precision: QuablaPrecision,
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<Vec<Vec<f64>>, String> {
    let program = QuablaMultiOutputProgram::new(graph, outputs)?;
    let executable = QuablaCompiler
        .compile_many_checked_with_precision(&program, target, precision)
        .map_err(|error| format!("{error:?}"))?;
    Ok(executable
        .execute(inputs)?
        .into_iter()
        .map(|tensor| tensor.data().to_vec())
        .collect())
}

#[test]
fn float64_lowering_keeps_double_constants_and_wide_reductions() -> Result<(), String> {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return Ok(());
    }
    let build = || -> Result<(TensorIr, Vec<usize>), String> {
        let mut graph = TensorIr::new();
        let x = graph.input("x", vec![1024])?;
        let tenth = graph.scalar_constant(0.1);
        let scaled = graph.mul(x, tenth)?;
        let exp = graph.exp(scaled)?;
        // 1024 lanes take the one-block-per-output reduction, whose `f32`
        // overflow guard must become a `DBL_MAX` guard in double.
        let total = graph.sum_axis(scaled, 0)?;
        Ok((graph, vec![scaled, exp, total]))
    };
    let values = (0..1024)
        .map(|i| 1e300 * (1.0 + (i as f64 * 0.37).sin()))
        .collect::<Vec<_>>();
    let small = (0..1024)
        .map(|i| (i as f64 * 0.37).sin())
        .collect::<Vec<_>>();
    for data in [values, small] {
        let inputs = BTreeMap::from([("x".to_string(), DynamicTensor::new(vec![1024], data)?)]);
        let (graph, outputs) = build()?;
        let device = run(graph, outputs, CUDA, QuablaPrecision::Float64, &inputs)?;
        let (graph, outputs) = build()?;
        let reference = run(
            graph,
            outputs,
            QuablaTarget::Cpu,
            QuablaPrecision::Default,
            &inputs,
        )?;
        // `x * 0.1` is one correctly rounded double multiply on both sides.
        assert_eq!(device[0], reference[0]);
        // CUDA and the CPU libm `exp` are each within one ulp; `exp` of the
        // huge lanes overflows to `inf` on both.
        for (actual, expected) in device[1].iter().zip(&reference[1]) {
            assert!(
                actual == expected || (actual - expected).abs() <= 1e-15 * expected.abs(),
                "{actual} {expected}"
            );
        }
        let (actual, expected) = (device[2][0], reference[2][0]);
        assert!(actual.is_finite(), "double reduction overflowed: {actual}");
        assert!(
            (actual - expected).abs() <= 1e-13 * expected.abs(),
            "{actual} {expected}"
        );
    }
    Ok(())
}

#[test]
fn float64_precision_only_changes_programs_with_float64_nodes() -> Result<(), String> {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return Ok(());
    }
    let tenth = |dtype: TensorDType| -> Result<(TensorIr, Vec<usize>), String> {
        let mut graph = TensorIr::new();
        let x = graph.input_typed("x", vec![1], dtype)?;
        let constant = graph.scalar_constant(0.1);
        let product = graph.mul(x, constant)?;
        Ok((graph, vec![product]))
    };
    let inputs = BTreeMap::from([("x".to_string(), DynamicTensor::new(vec![1], vec![1.0])?)]);
    let (graph, outputs) = tenth(TensorDType::F64)?;
    let native = run(graph, outputs, CUDA, QuablaPrecision::Float64, &inputs)?;
    assert_eq!(native, vec![vec![0.1]]);
    // 2^-12 is exact in f32, but its shortest f32 digits (0.00024414062) are
    // not exact in f64: the double kernel must see the full constant.
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![1])?;
    let constant = graph.scalar_constant(2f64.powi(-12));
    let product = graph.mul(x, constant)?;
    let third = BTreeMap::from([(
        "x".to_string(),
        DynamicTensor::new(vec![1], vec![1.0 / 3.0])?,
    )]);
    let native = run(graph, vec![product], CUDA, QuablaPrecision::Float64, &third)?;
    assert_eq!(native, vec![vec![2f64.powi(-12) / 3.0]]);
    // The default lowering keeps executing float64 nodes as float32.
    let (graph, outputs) = tenth(TensorDType::F64)?;
    let default = run(graph, outputs, CUDA, QuablaPrecision::Default, &inputs)?;
    assert_eq!(default, vec![vec![f64::from(0.1_f32)]]);
    // A float32 program has no float64 node and compiles as by default.
    let (graph, outputs) = tenth(TensorDType::F32)?;
    let single = run(graph, outputs, CUDA, QuablaPrecision::Float64, &inputs)?;
    assert_eq!(single, vec![vec![f64::from(0.1_f32)]]);
    Ok(())
}

#[test]
fn float64_precision_rejects_float32_nodes() -> Result<(), String> {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return Ok(());
    }
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    let y = graph.input_typed("y", vec![2], TensorDType::F32)?;
    let widened = graph.cast(y, TensorDType::F64)?;
    let sum = graph.add(x, widened)?;
    let program = QuablaMultiOutputProgram::new(graph, vec![sum])?;
    let error = QuablaCompiler
        .compile_many_checked_with_precision(&program, CUDA, QuablaPrecision::Float64)
        .expect_err("a float32 node must not run inside a double plan");
    let QuablaCompileError::Unsupported { op, message } = error else {
        return Err(format!(
            "expected an unsupported-operation error, got {error:?}"
        ));
    };
    assert_eq!(op, "float32");
    assert!(message.contains("float32 node"), "{message}");
    Ok(())
}
