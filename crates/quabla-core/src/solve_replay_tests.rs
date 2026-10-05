use super::*;

fn tensor(n: usize, columns: usize, data: Vec<f64>) -> DynamicTensor {
    DynamicTensor::new(vec![n, columns], data).unwrap()
}

fn assert_bits(actual: &DynamicTensor, expected: &DynamicTensor) {
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.dtype, expected.dtype);
    assert_eq!(
        actual.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected
            .data
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

fn previous_mixed(matrix: &MixedTangent, rhs: &MixedTangent) -> Result<MixedTangent, String> {
    let value = matrix.value.solve(&rhs.value)?;
    let first = matrix
        .value
        .solve(&rhs.first.sub(&matrix.first.matmul(&value)?)?)?;
    let second = matrix
        .value
        .solve(&rhs.second.sub(&matrix.second.matmul(&value)?)?)?;
    let mixed_rhs = rhs
        .mixed
        .sub(&matrix.mixed.matmul(&value)?)?
        .sub(&matrix.first.matmul(&second)?)?
        .sub(&matrix.second.matmul(&first)?)?;
    Ok(MixedTangent {
        value,
        first,
        second,
        mixed: matrix.value.solve(&mixed_rhs)?,
    })
}

fn assert_mixed_bits(actual: &MixedTangent, expected: &MixedTangent, dtype: TensorDType) {
    for (actual, expected) in [
        (&actual.value, &expected.value),
        (&actual.first, &expected.first),
        (&actual.second, &expected.second),
        (&actual.mixed, &expected.mixed),
    ] {
        assert_bits(
            &actual.clone().into_dtype(dtype),
            &expected.clone().into_dtype(dtype),
        );
    }
}

fn mixed_fixture(n: usize, columns: usize, dtype: TensorDType) -> (MixedTangent, MixedTangent) {
    let matrix = MixedTangent {
        value: tensor(
            n,
            n,
            (0..n * n)
                .map(|i| {
                    let row = i / n;
                    let column = i % n;
                    if column == (row + 1) % n {
                        n as f64 + 1.0
                    } else {
                        ((row * 7 + column * 11) % 17) as f64 / 13.0 - 0.5
                    }
                })
                .collect(),
        )
        .into_dtype(dtype),
        first: tensor(n, n, (0..n * n).map(|i| (i % 5) as f64 / 19.0).collect()).into_dtype(dtype),
        second: tensor(n, n, (0..n * n).map(|i| (i % 7) as f64 / 23.0).collect()).into_dtype(dtype),
        mixed: tensor(n, n, (0..n * n).map(|i| (i % 3) as f64 / 29.0).collect()).into_dtype(dtype),
    };
    let component = |scale| {
        tensor(
            n,
            columns,
            (0..n * columns).map(|i| (i as f64 + 1.0) * scale).collect(),
        )
        .into_dtype(dtype)
    };
    let rhs = MixedTangent {
        value: component(0.17),
        first: component(-0.21),
        second: component(0.31),
        mixed: component(-0.41),
    };
    (matrix, rhs)
}

#[test]
fn replay_matches_lu_bits_after_multiple_swaps_and_ties() {
    for data in [
        vec![0., 2., 3., 4., 5., 6., 7., 8., 10.],
        vec![1., 2., 3., -1., -7., 7., 1., 8., 11.],
    ] {
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let matrix = tensor(3, 3, data.clone()).into_dtype(dtype);
            let plan = SolveReplayPlan::for_finite_dense(&matrix).unwrap();
            assert!(
                plan.pivots
                    .iter()
                    .enumerate()
                    .filter(|(i, row)| *i != **row)
                    .count()
                    >= 2
            );
            assert_eq!(plan.pivots[0], 2);
            for columns in [1, 4] {
                let rhs = tensor(
                    3,
                    columns,
                    (0..3 * columns).map(|i| i as f64 / 7.0 - 0.5).collect(),
                )
                .into_dtype(dtype);
                assert_bits(
                    &plan.solve_finite(&rhs).unwrap(),
                    &matrix.solve_lu(&rhs).unwrap(),
                );
            }
        }
    }
}

#[test]
fn mixed_reuse_matches_original_expression_bits_and_equations() {
    for dtype in [TensorDType::F32, TensorDType::F64] {
        for n in [3, 7, 17] {
            for columns in [1, 3] {
                let (matrix, rhs) = mixed_fixture(n, columns, dtype);
                let actual = matrix.solve_reusing_finite_factor(&rhs).unwrap().unwrap();
                let expected = previous_mixed(&matrix, &rhs).unwrap();
                assert_mixed_bits(&actual, &expected, dtype);
                let first_rhs = rhs
                    .first
                    .sub(&matrix.first.matmul(&actual.value).unwrap())
                    .unwrap();
                let second_rhs = rhs
                    .second
                    .sub(&matrix.second.matmul(&actual.value).unwrap())
                    .unwrap();
                let mixed_rhs = rhs
                    .mixed
                    .sub(&matrix.mixed.matmul(&actual.value).unwrap())
                    .unwrap()
                    .sub(&matrix.first.matmul(&actual.second).unwrap())
                    .unwrap()
                    .sub(&matrix.second.matmul(&actual.first).unwrap())
                    .unwrap();
                for (solution, rhs) in [
                    (&actual.value, &rhs.value),
                    (&actual.first, &first_rhs),
                    (&actual.second, &second_rhs),
                    (&actual.mixed, &mixed_rhs),
                ] {
                    for (actual, expected) in matrix
                        .value
                        .matmul(solution)
                        .unwrap()
                        .data
                        .iter()
                        .zip(rhs.data.iter())
                    {
                        assert!((actual - expected).abs() <= 1e-11 * expected.abs().max(1.0));
                    }
                }
            }
        }
    }
}

#[test]
fn reuse_declines_triangular_singular_and_exceptional_arithmetic() {
    let (matrix, rhs) = mixed_fixture(3, 2, TensorDType::F64);
    for lower in [true, false] {
        let mut triangular = matrix.clone();
        triangular.value = tensor(3, 3, vec![3., 1., 2., 1., 4., 2., 1., 2., 5.])
            .triangular(lower)
            .unwrap();
        assert!(triangular
            .solve_reusing_finite_factor(&rhs)
            .unwrap()
            .is_none());
        assert!(triangular
            .value
            .finite_triangular_solution(&rhs.value)
            .is_some());
    }
    let mut singular = matrix.clone();
    singular.value = tensor(3, 3, vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    assert!(singular
        .solve_reusing_finite_factor(&rhs)
        .unwrap()
        .is_none());
    assert_eq!(
        previous_mixed(&singular, &rhs).err().unwrap(),
        "solve requires a non-singular coefficient matrix"
    );
    for exceptional in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut exceptional_matrix = matrix.clone();
        let HostTensorStorage::F64(data) = &mut exceptional_matrix.value.data else {
            unreachable!()
        };
        Arc::make_mut(data)[1] = exceptional;
        assert!(exceptional_matrix
            .solve_reusing_finite_factor(&rhs)
            .unwrap()
            .is_none());
        for component in 0..4 {
            let mut exceptional_rhs = rhs.clone();
            let target = match component {
                0 => &mut exceptional_rhs.value,
                1 => &mut exceptional_rhs.first,
                2 => &mut exceptional_rhs.second,
                _ => &mut exceptional_rhs.mixed,
            };
            let HostTensorStorage::F64(data) = &mut target.data else {
                unreachable!()
            };
            Arc::make_mut(data)[0] = exceptional;
            assert!(matrix
                .solve_reusing_finite_factor(&exceptional_rhs)
                .unwrap()
                .is_none());
        }
    }
    let overflow = tensor(2, 2, vec![f64::MAX, f64::MAX, -f64::MAX, f64::MAX]);
    assert!(SolveReplayPlan::for_finite_dense(&overflow).is_none());
    let mut derived_overflow = matrix.clone();
    let HostTensorStorage::F64(data) = &mut derived_overflow.first.data else {
        unreachable!()
    };
    Arc::make_mut(data).fill(f64::MAX);
    let mut large_rhs = rhs.clone();
    let HostTensorStorage::F64(data) = &mut large_rhs.value.data else {
        unreachable!()
    };
    Arc::make_mut(data).fill(100.0);
    assert!(derived_overflow
        .solve_reusing_finite_factor(&large_rhs)
        .unwrap()
        .is_none());
    let tiny = tensor(2, 2, vec![1e-308, 1e-308, -1e-308, 1e-308]);
    let plan = SolveReplayPlan::for_finite_dense(&tiny).unwrap();
    assert!(plan
        .solve_finite(&tensor(2, 1, vec![f64::MAX, f64::MAX]))
        .is_none());
    let mut wrong_shape = rhs.clone();
    wrong_shape.value = tensor(2, 1, vec![1., 2.]);
    assert_eq!(
        matrix.solve_reusing_finite_factor(&wrong_shape).err(),
        previous_mixed(&matrix, &wrong_shape).err()
    );
}

#[test]
#[ignore = "CPU release timing probe; run explicitly after correctness tests"]
fn mixed_solve_reuse_timing_probe() {
    use std::{hint::black_box, time::Instant};
    for (n, columns, iterations) in [(8, 2, 2000), (32, 4, 300), (96, 4, 40), (192, 4, 8)] {
        let (matrix, rhs) = mixed_fixture(n, columns, TensorDType::F64);
        let old = previous_mixed(&matrix, &rhs).unwrap();
        let new = matrix.solve_reusing_finite_factor(&rhs).unwrap().unwrap();
        assert_mixed_bits(&new, &old, TensorDType::F64);
        for _ in 0..3 {
            black_box(previous_mixed(black_box(&matrix), black_box(&rhs)).unwrap());
            black_box(
                matrix
                    .solve_reusing_finite_factor(black_box(&rhs))
                    .unwrap()
                    .unwrap(),
            );
        }
        let measure = |reuse| {
            let start = Instant::now();
            for _ in 0..iterations {
                if reuse {
                    black_box(
                        matrix
                            .solve_reusing_finite_factor(black_box(&rhs))
                            .unwrap()
                            .unwrap(),
                    );
                } else {
                    black_box(previous_mixed(black_box(&matrix), black_box(&rhs)).unwrap());
                }
            }
            start.elapsed().as_secs_f64()
        };
        for round in 0..12 {
            let (old_seconds, new_seconds) = if round % 2 == 0 {
                (measure(false), measure(true))
            } else {
                let new_seconds = measure(true);
                (measure(false), new_seconds)
            };
            println!("{{\"n\":{n},\"rhs_columns\":{columns},\"iterations\":{iterations},\"round\":{round},\"old_seconds\":{old_seconds:.9},\"reuse_seconds\":{new_seconds:.9}}}");
        }
    }
}

#[test]
fn scalar_hessian_and_hvp_integrate_reused_solve() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![])?;
    let base = tensor(3, 3, vec![0., 2., 3., 4., 5., 6., 7., 8., 10.]);
    let derivative = tensor(3, 3, vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1, -0.1, 0.2, 0.4]);
    let rhs_value = tensor(3, 2, vec![0.2, 0.4, -0.3, 0.8, 0.9, -0.2]);
    let base_node = graph.constant(base.clone(), false);
    let derivative_node = graph.constant(derivative.clone(), false);
    let scaled = graph.mul(x, derivative_node)?;
    let matrix_node = graph.add(base_node, scaled)?;
    let rhs_node = graph.constant(rhs_value.clone(), false);
    let solution = graph.solve(matrix_node, rhs_node)?;
    let output = graph.sum(solution)?;
    let inputs = BTreeMap::from([("x".into(), DynamicTensor::filled(vec![], 0.25)?)]);
    let matrix = MixedTangent {
        value: base.add(&derivative.mul(&DynamicTensor::filled(vec![], 0.25)?)?)?,
        first: derivative.clone(),
        second: derivative.clone(),
        mixed: DynamicTensor::filled(vec![3, 3], 0.0)?,
    };
    let rhs = MixedTangent {
        value: rhs_value,
        first: DynamicTensor::filled(vec![3, 2], 0.0)?,
        second: DynamicTensor::filled(vec![3, 2], 0.0)?,
        mixed: DynamicTensor::filled(vec![3, 2], 0.0)?,
    };
    let expected = previous_mixed(&matrix, &rhs)?.mixed.sum_all()?.data.get(0);
    let hessian = graph.hessian_scalar(output, "x", &inputs)?;
    assert_eq!(hessian[0][0].to_bits(), expected.to_bits());
    let hvp = graph.hvp_scalar(output, "x", &inputs, DynamicTensor::filled(vec![], 0.3)?)?;
    assert!((hvp.data.get(0) - expected * 0.3).abs() < 1e-12);
    Ok(())
}
