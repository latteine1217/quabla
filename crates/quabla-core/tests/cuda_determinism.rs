#![cfg(all(feature = "cuda", target_os = "linux"))]
//! Repeated CUDA runs of one program return the same bits.
//!
//! A full `Sum`/`Mean` over several blocks and the gradient of a loop capture
//! that broadcasts into the carry add many partial results. They are added in
//! a fixed order, with no floating-point atomics, so the result depends only
//! on the inputs, the device and the program, never on the order in which the
//! hardware schedules blocks. Each case runs `RUNS` times and must return one
//! bit pattern, and it must match the CPU float64 reference within the device
//! tolerance.

mod support;

use std::collections::BTreeMap;

use quabla_core::tensor_ir::{
    DynamicTensor, TensorDType, TensorForiExecutionPlan, TensorIr, TensorNodeId,
    TensorScanExecutionPlan,
};
use quabla_core::{QuablaCompiler, QuablaMultiOutputProgram, QuablaPrecision, QuablaTarget};

const CUDA: QuablaTarget = QuablaTarget::Cuda { device_ordinal: 0 };
const RUNS: usize = 100;
const PRECISIONS: [QuablaPrecision; 2] = [QuablaPrecision::Default, QuablaPrecision::Float64];

type Inputs = BTreeMap<String, DynamicTensor>;
/// The outputs of one execution, each as its flat data.
type Outputs = Vec<Vec<f64>>;

/// The relative tolerance of a device result against the CPU float64 one.
fn tolerance(precision: QuablaPrecision) -> f64 {
    match precision {
        QuablaPrecision::Float64 => 1e-12,
        _ => 1e-5,
    }
}

/// Runs the program `RUNS` times on CUDA, requires one bit pattern, and
/// returns that result with the CPU float64 result.
fn repeated(
    build: impl Fn() -> Result<(TensorIr, Vec<TensorNodeId>), String>,
    inputs: &Inputs,
    precision: QuablaPrecision,
    case: &str,
) -> Result<(Outputs, Outputs), String> {
    let compile = |target, precision| -> Result<_, String> {
        let (graph, outputs) = build()?;
        let program = QuablaMultiOutputProgram::new(graph, outputs)?;
        QuablaCompiler
            .compile_many_checked_with_precision(&program, target, precision)
            .map_err(|error| format!("{error:?}"))
    };
    let device = compile(CUDA, precision)?;
    let data = |outputs: Vec<DynamicTensor>| {
        outputs
            .into_iter()
            .map(|tensor| tensor.data().to_vec())
            .collect::<Vec<_>>()
    };
    let bits = |values: &[Vec<f64>]| {
        values
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let first = data(device.execute(inputs)?);
    let expected = bits(&first);
    let mut distinct = vec![expected.clone()];
    for _ in 1..RUNS {
        let actual = bits(&data(device.execute(inputs)?));
        if !distinct.contains(&actual) {
            distinct.push(actual);
        }
    }
    assert_eq!(
        distinct.len(),
        1,
        "{case} ({precision:?}): {} distinct results over {RUNS} runs",
        distinct.len()
    );
    let reference = data(compile(QuablaTarget::Cpu, QuablaPrecision::Float64)?.execute(inputs)?);
    Ok((first, reference))
}

/// Checks `actual` against `reference` with an error bound relative to
/// `scale` (the magnitude the rounding errors are proportional to).
fn assert_close(actual: f64, reference: f64, scale: f64, precision: QuablaPrecision, case: &str) {
    let bound = tolerance(precision) * scale.max(1.0);
    assert!(
        (actual - reference).abs() <= bound,
        "{case} ({precision:?}): device {actual:e}, CPU {reference:e}, bound {bound:e}"
    );
}

fn values(count: usize, phase: f64) -> Vec<f64> {
    (0..count)
        .map(|i| (phase * i as f64).sin() * (1.0 + (i % 7) as f64))
        .collect()
}

fn reduction(count: usize, mean: bool) -> Result<(TensorIr, Vec<TensorNodeId>), String> {
    let mut graph = TensorIr::new();
    let x = graph.input_typed("x", vec![count], TensorDType::F64)?;
    let output = if mean { graph.mean(x)? } else { graph.sum(x)? };
    Ok((graph, vec![output]))
}

/// The fixed tree of the one-block `Sum`/`Mean` kernel in `T`: 256 lanes,
/// padded with zeros, halved pairwise, then added to `+0`.
fn one_block_tree<T>(values: &[T], zero: T, mean_divisor: Option<T>) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T>,
{
    let mut partial = [zero; 256];
    partial[..values.len()].copy_from_slice(values);
    let mut stride = 128;
    while stride > 0 {
        for thread in 0..stride {
            partial[thread] = partial[thread] + partial[thread + stride];
        }
        stride /= 2;
    }
    match mean_divisor {
        Some(divisor) => zero + partial[0] / divisor,
        None => zero + partial[0],
    }
}

#[test]
fn full_reductions_are_deterministic_and_match_the_cpu() -> Result<(), String> {
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    for count in [1, 7, 256, 257, 512, 1000, 65_539, 1 << 20] {
        let data = values(count, 0.37);
        let scale = data.iter().map(|value| value.abs()).sum::<f64>();
        let inputs = BTreeMap::from([(
            "x".to_string(),
            DynamicTensor::with_dtype(vec![count], data.clone(), TensorDType::F64)?,
        )]);
        for precision in PRECISIONS {
            for mean in [false, true] {
                let case = format!("{} of {count}", if mean { "mean" } else { "sum" });
                let (device, reference) =
                    repeated(|| reduction(count, mean), &inputs, precision, &case)?;
                let divisor = if mean { count as f64 } else { 1.0 };
                assert_close(
                    device[0][0],
                    reference[0][0],
                    scale / divisor,
                    precision,
                    &case,
                );
                // One block keeps the bits of its fixed tree (the kernel's value before the
                // multi-block pass was made deterministic).
                if count <= 256 {
                    let expected = if precision == QuablaPrecision::Float64 {
                        one_block_tree(&data, 0.0f64, mean.then_some(count as f64))
                    } else {
                        let single = data.iter().map(|value| *value as f32).collect::<Vec<_>>();
                        one_block_tree(&single, 0.0f32, mean.then_some(count as f32)) as f64
                    };
                    assert_eq!(
                        device[0][0].to_bits(),
                        expected.to_bits(),
                        "{case} ({precision:?})"
                    );
                }
            }
        }
    }
    // An all -0 input still sums to +0 over several blocks, the value of the atomic
    // accumulation into a cleared output used before.
    let inputs = BTreeMap::from([(
        "x".to_string(),
        DynamicTensor::with_dtype(vec![1000], vec![-0.0; 1000], TensorDType::F64)?,
    )]);
    for precision in PRECISIONS {
        let (device, _) = repeated(|| reduction(1000, false), &inputs, precision, "-0")?;
        assert_eq!(device[0][0].to_bits(), 0.0f64.to_bits());
    }
    Ok(())
}

/// The loop kinds whose VJP reduces a broadcast capture's gradient over the
/// carry lanes.
#[derive(Clone, Copy, Debug)]
enum Loop {
    /// `fori_loop`, gradient of the capture only (one-target kernel).
    Fori,
    /// `fori_loop`, gradients of the initial carry and the capture (group kernel).
    ForiGroup,
    /// `scan`, gradient of the capture only.
    Scan,
    /// `scan`, gradients of the initial carry and the capture.
    ScanGroup,
    /// `scan`, the directional derivative of both gradients along a capture
    /// tangent (forward-over-reverse group kernel).
    ScanDirectional,
}

/// `loss = sum(loop(c0, w))` for the body `tanh(c * w) + 0.5 c + 1e-4 i`,
/// whose capture `w` of shape `capture` broadcasts into the carry of shape
/// `carry`, and the requested gradients of `loss`.
fn loop_gradients(
    kind: Loop,
    carry: &[usize],
    capture: &[usize],
) -> Result<(TensorIr, Vec<TensorNodeId>), String> {
    let dtype = TensorDType::F64;
    let mut body = TensorIr::new();
    let state = body.input_typed("carry", carry.to_vec(), dtype)?;
    let index = body.input_typed("index", vec![], dtype)?;
    let weight = body.input_typed("w", capture.to_vec(), dtype)?;
    let product = body.mul(state, weight)?;
    let next = body.tanh(product)?;
    let half = body.scalar_constant(0.5);
    let damped = body.mul(state, half)?;
    let next = body.add(next, damped)?;
    let step = body.scalar_constant(1e-4);
    let shift = body.mul(index, step)?;
    let next = body.add(next, shift)?;
    let mut graph = TensorIr::new();
    let initial = graph.input_typed("c0", carry.to_vec(), dtype)?;
    let w = graph.input_typed("w", capture.to_vec(), dtype)?;
    let captures = vec![("w".to_string(), w)];
    let loss = match kind {
        Loop::Fori | Loop::ForiGroup => {
            let plan =
                TensorForiExecutionPlan::new(2, 10, body.compile_cpu(next)?, "carry", "index")?;
            let output = graph.fori(initial, plan, captures)?;
            graph.sum(output)?
        }
        Loop::Scan | Loop::ScanGroup | Loop::ScanDirectional => {
            let quarter = body.scalar_constant(0.25);
            let emitted = body.mul(next, quarter)?;
            let plan = TensorScanExecutionPlan::new(
                2,
                10,
                body.compile_cpu_many(&[next, emitted])?.0,
                "carry",
                "index",
            )?;
            let (last, outputs) = graph.scan(initial, plan, captures)?;
            let last = graph.sum(last)?;
            let outputs = graph.sum(outputs)?;
            graph.add(last, outputs)?
        }
    };
    let vjp = graph.symbolic_vjp(loss, "seed")?;
    let targets = match kind {
        Loop::Fori | Loop::Scan => vec![vjp.gradients["w"]],
        _ => vec![vjp.gradients["c0"], vjp.gradients["w"]],
    };
    if matches!(kind, Loop::ScanDirectional) {
        let jvp = vjp.graph.symbolic_jvp_many_with_tangent_inputs(
            &targets,
            &BTreeMap::from([("w".to_string(), "w_tangent".to_string())]),
        )?;
        return Ok((jvp.graph, jvp.tangents));
    }
    Ok((vjp.graph, targets))
}

fn loop_inputs(carry: &[usize], capture: &[usize]) -> Result<Inputs, String> {
    let lanes = carry.iter().product::<usize>();
    let weights = capture.iter().product::<usize>();
    let tensor = |shape: &[usize], data: Vec<f64>| {
        DynamicTensor::with_dtype(shape.to_vec(), data, TensorDType::F64)
    };
    Ok(BTreeMap::from([
        (
            "c0".to_string(),
            tensor(
                carry,
                (0..lanes).map(|i| 0.1 + 0.3 * (i as f64).sin()).collect(),
            )?,
        ),
        (
            "w".to_string(),
            tensor(
                capture,
                (0..weights).map(|i| 0.7 + 0.001 * i as f64).collect(),
            )?,
        ),
        (
            "w_tangent".to_string(),
            tensor(
                capture,
                (0..weights).map(|i| 1.0 - 0.002 * i as f64).collect(),
            )?,
        ),
        ("seed".to_string(), tensor(&[], vec![1.0])?),
    ]))
}

#[test]
fn loop_capture_gradients_are_deterministic_and_match_the_cpu() -> Result<(), String> {
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    let shapes: [(&[usize], &[usize]); 5] = [
        // Fewer lanes than one reduction block: one thread adds the lanes.
        (&[200], &[1]),
        // The same, one thread per capture element.
        (&[64, 100], &[64, 1]),
        // Many lanes per capture element: one block per element.
        (&[65_536], &[1]),
        // A capture over the leading axis: each element reduces a strided row.
        (&[256, 300], &[256, 1]),
        // A trailing capture under a leading broadcast axis, with a rank change.
        (&[300, 64], &[64]),
    ];
    for (carry, capture) in shapes {
        let inputs = loop_inputs(carry, capture)?;
        for kind in [
            Loop::Fori,
            Loop::ForiGroup,
            Loop::Scan,
            Loop::ScanGroup,
            Loop::ScanDirectional,
        ] {
            // The one-target scan kernel scans every output per lane and step; keep it small.
            if matches!(kind, Loop::Scan) && carry.iter().product::<usize>() > 20_000 {
                continue;
            }
            for precision in PRECISIONS {
                let case = format!("{kind:?} carry {carry:?} capture {capture:?}");
                let (device, reference) = repeated(
                    || loop_gradients(kind, carry, capture),
                    &inputs,
                    precision,
                    &case,
                )?;
                // Relative to the largest entry of each gradient: the float32 loop
                // itself rounds every lane, and a capture element that adds lanes of
                // both signs keeps their absolute, not relative, error.
                for (device, reference) in device.iter().zip(&reference) {
                    assert_eq!(device.len(), reference.len(), "{case}");
                    let peak = reference
                        .iter()
                        .fold(0.0f64, |peak, value| peak.max(value.abs()));
                    for (actual, expected) in device.iter().zip(reference) {
                        assert_close(*actual, *expected, peak, precision, &case);
                    }
                }
            }
        }
    }
    Ok(())
}

/// With fewer lanes than one reduction block, a capture gradient is the
/// serial sum, in lane order, of the per-lane gradients. Up to one warp of
/// lanes this is also the value of the atomic accumulation used before,
/// whose single-warp order was the lane order.
#[test]
fn short_capture_reductions_add_the_lanes_in_order() -> Result<(), String> {
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    for lanes in [1, 2, 8, 32, 33, 200] {
        let inputs = loop_inputs(&[lanes], &[1])?;
        let mut per_lane_inputs = inputs.clone();
        per_lane_inputs.insert(
            "w".to_string(),
            DynamicTensor::with_dtype(vec![lanes], vec![0.7; lanes], TensorDType::F64)?,
        );
        for precision in PRECISIONS {
            let case = format!("{lanes} lanes");
            let (reduced, _) = repeated(
                || loop_gradients(Loop::Fori, &[lanes], &[1]),
                &inputs,
                precision,
                &case,
            )?;
            let (per_lane, _) = repeated(
                || loop_gradients(Loop::Fori, &[lanes], &[lanes]),
                &per_lane_inputs,
                precision,
                &case,
            )?;
            let expected = if precision == QuablaPrecision::Float64 {
                per_lane[0].iter().fold(0.0f64, |sum, value| sum + value)
            } else {
                per_lane[0]
                    .iter()
                    .fold(0.0f32, |sum, value| sum + *value as f32) as f64
            };
            assert_eq!(
                reduced[0][0].to_bits(),
                expected.to_bits(),
                "{case} ({precision:?})"
            );
        }
    }
    Ok(())
}
