#![cfg(all(feature = "cuda", target_os = "linux"))]

use quabla_core::tensor_ir::{CudaBackend, DynamicTensor, TensorIr};
use std::collections::BTreeMap;
use std::time::Instant;

fn execute(
    shape: Vec<usize>,
    axis: usize,
    mean: bool,
    values: Vec<f64>,
) -> Result<Vec<f64>, String> {
    let mut graph = TensorIr::new();
    let input = graph.input("x", shape.clone())?;
    let output = if mean {
        graph.mean_axis(input, axis as isize)?
    } else {
        graph.sum_axis(input, axis as isize)?
    };
    let plan = CudaBackend::new(0).compile(graph.compile_cpu(output)?)?;
    let inputs = BTreeMap::from([("x".to_string(), DynamicTensor::new(shape, values)?)]);
    Ok(plan.execute(&inputs)?.data().to_vec())
}

#[test]
fn long_axis_reductions_match_wide_reference() -> Result<(), String> {
    if std::env::var_os("QUABLA_CUDA_TEST").is_none() {
        return Ok(());
    }
    for shape in [
        vec![255],
        vec![256],
        vec![257],
        vec![4097],
        vec![3, 1025],
        vec![1025, 3],
        vec![2, 513, 3],
    ] {
        for axis in 0..shape.len() {
            let count = shape.iter().product();
            let values = (0..count)
                .map(|i| (((i * 17 % 127) as f32 - 63.0) * 0.03125) as f64)
                .collect::<Vec<_>>();
            let stride = shape[axis + 1..].iter().product::<usize>();
            let extent = shape[axis];
            for mean in [false, true] {
                let actual = execute(shape.clone(), axis, mean, values.clone())?;
                for (i, value) in actual.iter().enumerate() {
                    let base = (i / stride) * stride * extent + i % stride;
                    let mut reference = (0..extent).map(|k| values[base + k * stride]).sum::<f64>();
                    if mean {
                        reference /= extent as f64;
                    }
                    assert!((value - reference).abs() <= 1e-5 + 1e-5 * reference.abs(), "shape={shape:?} axis={axis} mean={mean} actual={value} reference={reference}");
                }
            }
        }
    }
    // Large cancellation should retain small contributions without extra buffers.
    let mut cancellation = vec![0.0; 1024];
    cancellation[0] = 1e8;
    cancellation[1] = 1.0;
    cancellation[2] = -1e8;
    assert_eq!(execute(vec![1024], 0, false, cancellation)?, vec![1.0]);
    let mut mixed_infinity = vec![0.0; 1024];
    mixed_infinity[0] = f64::INFINITY;
    mixed_infinity[511] = f64::NEG_INFINITY;
    let mut finite_overflow = vec![0.0; 1024];
    finite_overflow[0] = f32::MAX as f64;
    finite_overflow[1] = f32::MAX as f64;
    finite_overflow[2] = -(f32::MAX as f64);
    for values in [
        mixed_infinity,
        finite_overflow,
        vec![-0.0; 1024],
        vec![f32::from_bits(1) as f64; 1024],
        vec![f64::INFINITY; 1024],
        vec![f64::NEG_INFINITY; 1024],
        vec![f64::NAN; 1024],
        vec![f32::MAX as f64; 1024],
    ] {
        let actual = execute(vec![1024], 0, false, values.clone())?[0];
        let reference = values.iter().fold(0.0f32, |sum, value| sum + *value as f32);
        assert!(actual.is_nan() && reference.is_nan() || actual == reference as f64);
    }
    Ok(())
}

#[test]
#[ignore = "isolated complete-operation timing probe"]
fn profile_long_axis_reduction() -> Result<(), String> {
    if std::env::var_os("QUABLA_CUDA_PROFILE").is_none() {
        return Ok(());
    }
    for (shape, axis) in [
        (vec![64, 16384], 1),
        (vec![16384, 64], 0),
        (vec![1048576], 0),
        (vec![64, 32], 1),
    ] {
        for mean in [false, true] {
            let mut graph = TensorIr::new();
            let input = graph.input("x", shape.clone())?;
            let output = if mean {
                graph.mean_axis(input, axis as isize)?
            } else {
                graph.sum_axis(input, axis as isize)?
            };
            let plan = CudaBackend::new(0).compile(graph.compile_cpu(output)?)?;
            let count = shape.iter().product::<usize>();
            let data = (0..count)
                .map(|i| ((i % 31) as f64 - 15.0) * 0.03125)
                .collect();
            let inputs =
                BTreeMap::from([("x".to_string(), DynamicTensor::new(shape.clone(), data)?)]);
            let mut samples = Vec::new();
            let mut checksum = 0.0;
            for _ in 0..11 {
                let start = Instant::now();
                checksum = plan.execute(&inputs)?.data().iter().sum::<f64>();
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            samples.remove(0);
            samples.sort_by(f64::total_cmp);
            println!("{{\"shape\":{shape:?},\"axis\":{axis},\"mean\":{mean},\"median_ms\":{},\"input_bytes\":{},\"output_bytes\":{},\"global_scratch_bytes\":0,\"checksum\":{checksum}}}", samples[5], count * 4, count / shape[axis] * 4);
        }
    }
    Ok(())
}
