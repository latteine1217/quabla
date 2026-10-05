use super::*;
use std::hint::black_box;
use std::time::Instant;

// These oracles retain the scalar coordinate indexing used before the fast paths.
fn previous_elementwise(
    lhs: &DynamicTensor,
    rhs: &DynamicTensor,
    f: impl Fn(f64, f64) -> f64,
) -> Result<DynamicTensor, String> {
    let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
    let lhs_strides = contiguous_strides(&lhs.shape);
    let rhs_strides = contiguous_strides(&rhs.shape);
    let data = (0..element_count(&shape)?)
        .map(|index| {
            f(
                lhs.data
                    .get(broadcast_offset(index, &shape, &lhs.shape, &lhs_strides)),
                rhs.data
                    .get(broadcast_offset(index, &shape, &rhs.shape, &rhs_strides)),
            )
        })
        .collect();
    DynamicTensor::new(shape, data)
}

fn previous_reduce(
    input: &DynamicTensor,
    axis: usize,
    scale: f64,
) -> Result<DynamicTensor, String> {
    let shape = reduced_shape(&input.shape, axis)?;
    let strides = contiguous_strides(&shape);
    let mut data = vec![0.0; element_count(&shape)?];
    for (index, value) in input.data.iter().enumerate() {
        let mut remaining = index;
        let mut output_index = 0;
        for source_axis in (0..input.shape.len()).rev() {
            let coordinate = remaining % input.shape[source_axis];
            remaining /= input.shape[source_axis];
            if source_axis != axis {
                let output_axis = if source_axis < axis {
                    source_axis
                } else {
                    source_axis - 1
                };
                output_index += coordinate * strides[output_axis];
            }
        }
        data[output_index] += value * scale;
    }
    DynamicTensor::new(shape, data)
}

fn assert_bits(actual: &DynamicTensor, expected: &DynamicTensor) {
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.dtype, expected.dtype);
    for (index, (actual, expected)) in actual.data.iter().zip(expected.data.iter()).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
    }
}

fn fixture(shape: &[usize], pattern: &[f64], dtype: TensorDType) -> DynamicTensor {
    DynamicTensor::with_dtype(
        shape.to_vec(),
        (0..element_count(shape).unwrap())
            .map(|index| pattern[index % pattern.len()])
            .collect(),
        dtype,
    )
    .unwrap()
}

#[test]
fn elementwise_fast_paths_and_broadcast_fallback_match_scalar_bits() {
    let patterns = [
        vec![0.0, -0.0, 1.0, -3.25, 1.0e20, -1.0e20, 1.0e-20],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 2.0],
    ];
    let shapes = [
        (vec![], vec![]),
        (vec![7], vec![7]),
        (vec![2, 1, 3, 2, 2], vec![2, 1, 3, 2, 2]),
        (vec![], vec![2, 3]),
        (vec![1, 1, 1, 1], vec![2, 3]),
        (vec![2, 3], vec![]),
        (vec![2, 3], vec![1, 1, 1, 1]),
        (vec![2, 1, 3], vec![1, 4, 1]),
        (vec![3], vec![2, 1]),
    ];
    let operations: [fn(f64, f64) -> f64; 5] = [
        |a, b| a + b,
        |a, b| a - b,
        |a, b| a * b,
        |a, b| a / b,
        |a, b| f64::from(a > b),
    ];
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for (lhs_shape, rhs_shape) in &shapes {
            for pattern in &patterns {
                let lhs = fixture(lhs_shape, pattern, dtype);
                let rhs = fixture(
                    rhs_shape,
                    &pattern.iter().rev().copied().collect::<Vec<_>>(),
                    dtype,
                );
                for operation in operations {
                    assert_bits(
                        &lhs.elementwise(&rhs, operation).unwrap().into_dtype(dtype),
                        &previous_elementwise(&lhs, &rhs, operation)
                            .unwrap()
                            .into_dtype(dtype),
                    );
                }
            }
        }
    }
}

#[test]
fn axis_reductions_preserve_linear_order_identity_and_rounding() {
    let patterns = [
        vec![1.0e20, 1.0, -1.0e20, 3.0, -0.0, 0.0, -7.0],
        vec![-0.0],
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 2.0],
        vec![1.25, -3.5, 2.0e-20, 8.0],
    ];
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for shape in [vec![1], vec![7], vec![2, 3, 4], vec![2, 1, 3, 2, 2, 1]] {
            for pattern in &patterns {
                let input = fixture(&shape, pattern, dtype);
                for (axis, &extent) in shape.iter().enumerate() {
                    for scale in [1.0, 1.0 / extent as f64, -0.0, -0.25] {
                        assert_bits(
                            &input.reduce_axis(axis, scale).unwrap().into_dtype(dtype),
                            &previous_reduce(&input, axis, scale)
                                .unwrap()
                                .into_dtype(dtype),
                        );
                    }
                }
            }
        }
    }
    for shape in [vec![], vec![3], vec![2, 3]] {
        let input = fixture(&shape, &[1.0], TensorDType::F64);
        assert_eq!(
            input.reduce_axis(shape.len(), 1.0).unwrap_err(),
            previous_reduce(&input, shape.len(), 1.0).unwrap_err()
        );
    }
}

#[test]
fn broadcasting_errors_precede_fast_paths_and_division_follows_ieee() {
    let lhs = fixture(&[2, 3], &[1.0], TensorDType::F64);
    let invalid = fixture(&[2], &[2.0], TensorDType::F64);
    assert_eq!(
        lhs.add(&invalid).unwrap_err(),
        previous_elementwise(&lhs, &invalid, |a, b| a + b).unwrap_err()
    );
    for shape in [vec![], vec![1, 1, 1], vec![2, 3]] {
        for zero in [0.0, -0.0] {
            let rhs = fixture(&shape, &[zero], TensorDType::F64);
            assert_bits(
                &lhs.div(&rhs).unwrap(),
                &previous_elementwise(&lhs, &rhs, |a, b| a / b).unwrap(),
            );
            assert_bits(
                &rhs.div(&lhs).unwrap(),
                &previous_elementwise(&rhs, &lhs, |a, b| a / b).unwrap(),
            );
        }
        let rhs = fixture(&shape, &[2.0], TensorDType::F64);
        assert!(lhs
            .sub(&rhs)
            .unwrap()
            .log()
            .unwrap()
            .data()
            .iter()
            .all(|value| value.is_nan()));
    }
}

#[test]
fn axis_reduction_ad_fixtures_match_scalar_oracles() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for axis in 0..3 {
            for mean in [false, true] {
                let mut graph = TensorIr::new();
                let x = graph.input_typed("x", vec![2, 3, 4], dtype)?;
                let squared = graph.mul(x, x)?;
                let output = if mean {
                    graph.mean_axis(squared, axis as isize)?
                } else {
                    graph.sum_axis(squared, axis as isize)?
                };
                let value = fixture(&[2, 3, 4], &[0.25, -0.5, 1.5, 2.0], dtype);
                let tangent = fixture(&[2, 3, 4], &[0.125, -0.25, 0.5], dtype);
                let inputs = BTreeMap::from([("x".into(), value.clone())]);
                let tangents = BTreeMap::from([("x".into(), tangent.clone())]);
                let scale = if mean {
                    1.0 / value.shape[axis] as f64
                } else {
                    1.0
                };
                let square = previous_elementwise(&value, &value, |a, b| a * b)?.into_dtype(dtype);
                let expected_value = previous_reduce(&square, axis, scale)?.into_dtype(dtype);
                let product = previous_elementwise(&value, &tangent, |a, b| a * b)?;
                let expected_tangent = previous_reduce(
                    &previous_elementwise(&product, &product, |a, b| a + b)?.into_dtype(dtype),
                    axis,
                    scale,
                )?
                .into_dtype(dtype);
                let (actual_value, actual_tangent) = graph.jvp(output, &inputs, &tangents)?;
                assert_bits(&actual_value, &expected_value);
                assert_bits(&actual_tangent, &expected_tangent);
                let cotangent =
                    DynamicTensor::filled(expected_value.shape.clone(), 0.75)?.into_dtype(dtype);
                let gradients = graph.vjp(output, &inputs, cotangent)?;
                let expanded = DynamicTensor::filled(value.shape.clone(), 0.75 * scale)?;
                let contribution = previous_elementwise(&expanded, &value, |a, b| a * b)?;
                let expected_gradient =
                    previous_elementwise(&contribution, &contribution, |a, b| a + b)?
                        .into_dtype(dtype);
                assert_bits(&gradients["x"], &expected_gradient);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "CPU release timing probe; run explicitly with --release --ignored --nocapture"]
fn cpu_kernel_paired_release_probe() {
    let shape = vec![16, 64, 1024];
    let input = fixture(&shape, &[0.25, -0.5, 1.5, 2.0], TensorDType::F64);
    let rhs = fixture(&shape, &[0.125, 0.25, -0.5], TensorDType::F64);
    let scalar = fixture(&[1, 1, 1, 1], &[0.125], TensorDType::F64);
    type ProbeOperation<'a> = Box<dyn Fn() -> DynamicTensor + 'a>;
    let operations: Vec<(&str, ProbeOperation<'_>, ProbeOperation<'_>)> = vec![
        (
            "equal_shape",
            Box::new(|| {
                previous_elementwise(black_box(&input), black_box(&rhs), |a, b| a * b).unwrap()
            }),
            Box::new(|| black_box(&input).mul(black_box(&rhs)).unwrap()),
        ),
        (
            "scalar_rhs",
            Box::new(|| {
                previous_elementwise(black_box(&input), black_box(&scalar), |a, b| a - b).unwrap()
            }),
            Box::new(|| black_box(&input).sub(black_box(&scalar)).unwrap()),
        ),
        (
            "scalar_lhs",
            Box::new(|| {
                previous_elementwise(black_box(&scalar), black_box(&input), |a, b| a - b).unwrap()
            }),
            Box::new(|| black_box(&scalar).sub(black_box(&input)).unwrap()),
        ),
        (
            "reduce_axis_0",
            Box::new(|| previous_reduce(black_box(&input), 0, 1.0 / 16.0).unwrap()),
            Box::new(|| black_box(&input).reduce_axis(0, 1.0 / 16.0).unwrap()),
        ),
        (
            "reduce_axis_1",
            Box::new(|| previous_reduce(black_box(&input), 1, 1.0 / 64.0).unwrap()),
            Box::new(|| black_box(&input).reduce_axis(1, 1.0 / 64.0).unwrap()),
        ),
        (
            "reduce_axis_2",
            Box::new(|| previous_reduce(black_box(&input), 2, 1.0 / 1024.0).unwrap()),
            Box::new(|| black_box(&input).reduce_axis(2, 1.0 / 1024.0).unwrap()),
        ),
    ];
    println!("case,round,first,old_ns,new_ns,elements");
    for (name, old, new) in operations {
        assert_bits(&new(), &old());
        black_box(old());
        black_box(new());
        for round in 0..12 {
            let measure = |operation: &dyn Fn() -> DynamicTensor| {
                let started = Instant::now();
                let output = operation();
                let elapsed = started.elapsed().as_nanos();
                black_box(output);
                elapsed
            };
            let (old_ns, new_ns) = if round % 2 == 0 {
                (measure(&old), measure(&new))
            } else {
                let new_ns = measure(&new);
                (measure(&old), new_ns)
            };
            println!(
                "{name},{round},{},{old_ns},{new_ns},{}",
                if round % 2 == 0 { "old" } else { "new" },
                input.data.len()
            );
        }
    }
}
