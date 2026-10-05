use super::*;
use crate::tensor_ir::{CholeskyAdKind, TensorIr};

fn expanded(inputs: &[Array], n: usize, kind: Option<CholeskyAdKind>) -> Array {
    let mut graph = TensorIr::new();
    let mut bindings = BTreeMap::new();
    let ids = inputs
        .iter()
        .enumerate()
        .map(|(index, array)| {
            let name = format!("argument_{index}");
            bindings.insert(name.clone(), array.clone());
            graph
                .input_typed(name, vec![n, n], TensorDType::F64)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let output = if let Some(kind) = kind {
        graph.expanded_cholesky_ad(&ids, kind).unwrap()
    } else {
        graph.expanded_cholesky(ids[0]).unwrap()
    };
    let plan = graph.compile_cpu(output).unwrap();
    MlxBackend
        .execute_arrays_with_retained(&plan, &[plan.output_node_id], &BTreeMap::new(), &bindings)
        .unwrap()
        .remove(0)
}

fn equivalent(actual: &Array, expected: &Array, context: &str) {
    transforms::eval([actual, expected]).unwrap();
    let mut differing = 0usize;
    let mut max_absolute = 0.0f32;
    for (index, (&actual, &expected)) in actual
        .as_slice::<f32>()
        .iter()
        .zip(expected.as_slice::<f32>())
        .enumerate()
    {
        differing += usize::from(
            actual.to_bits() != expected.to_bits() && !(actual.is_nan() && expected.is_nan()),
        );
        if actual.is_finite() && expected.is_finite() {
            max_absolute = max_absolute.max((actual - expected).abs());
        }
        assert!(
            actual.to_bits() == expected.to_bits()
                || (actual.is_nan() && expected.is_nan())
                || (actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs()),
            "{context} lane {index}: {actual} != {expected}"
        );
    }
    println!("comparison {context}: differing_lanes={differing}, max_absolute={max_absolute}");
}

#[test]
fn native_metal_cholesky_matches_expanded_primal_and_derivatives() {
    let _guard = mlx_execution_guard();
    let stream = StreamOrDevice::gpu();
    for n in [1usize, 2, 8, 16] {
        let matrix = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    3.0 + (i / n) as f32 * 0.1
                } else {
                    0.03 * ((i * 7 % 11) as f32 - 5.0)
                }
            })
            .collect::<Vec<_>>();
        let direction = (0..n * n)
            .map(|i| 0.01 * ((i * 3 % 7) as f32 - 3.0))
            .collect::<Vec<_>>();
        let cotangent = (0..n * n)
            .map(|i| 0.04 * ((i * 5 % 13) as f32 - 6.0))
            .collect::<Vec<_>>();
        let mixed = (0..n * n)
            .map(|i| 0.02 * ((i * 11 % 17) as f32 - 8.0))
            .collect::<Vec<_>>();
        let array = |data: &[f32]| Array::from_slice(data, &[n as i32, n as i32]);
        for kind in [
            None,
            Some(CholeskyAdKind::Jvp),
            Some(CholeskyAdKind::Vjp),
            Some(CholeskyAdKind::Mixed),
            Some(CholeskyAdKind::VjpJvp),
        ] {
            let inputs = match kind {
                None => vec![array(&matrix)],
                Some(CholeskyAdKind::Jvp) => vec![array(&matrix), array(&direction)],
                Some(CholeskyAdKind::Vjp) => vec![array(&matrix), array(&cotangent)],
                _ => vec![
                    array(&matrix),
                    array(&direction),
                    array(&cotangent),
                    array(&mixed),
                ],
            };
            let actual = cholesky_backend::evaluate(
                &inputs.iter().collect::<Vec<_>>(),
                &[n, n],
                kind,
                &stream,
            )
            .unwrap();
            let expected = expanded(&inputs, n, kind);
            equivalent(&actual, &expected, &format!("n={n} kind={kind:?}"));
        }
    }
}

#[test]
fn native_metal_cholesky_keeps_exceptional_values_and_batches() {
    let _guard = mlx_execution_guard();
    let stream = StreamOrDevice::gpu();
    for matrix in [
        vec![0.0],
        vec![-1.0],
        vec![f32::INFINITY],
        vec![f32::NAN],
        vec![1e-30],
        vec![1e30],
        vec![0.0, 5.0, 0.0, 1.0],
        vec![f32::INFINITY, 99.0, 0.0, 1.0],
        vec![1.0, 99.0, f32::INFINITY, 1.0],
        vec![1.0, 99.0, f32::NAN, 1.0],
        vec![1e-40, 99.0, 1e-30, 1.0],
        vec![1e-20, 9.0, 1e-25, 1e20],
    ] {
        let n = if matrix.len() == 1 { 1 } else { 2 };
        for kind in [
            None,
            Some(CholeskyAdKind::Jvp),
            Some(CholeskyAdKind::Vjp),
            Some(CholeskyAdKind::Mixed),
            Some(CholeskyAdKind::VjpJvp),
        ] {
            let mut inputs = vec![Array::from_slice(&matrix, &[n as i32, n as i32])];
            let count = match kind {
                None => 1,
                Some(CholeskyAdKind::Jvp | CholeskyAdKind::Vjp) => 2,
                _ => 4,
            };
            inputs
                .extend((1..count).map(|_| {
                    Array::from_slice(&vec![0.1f32; matrix.len()], &[n as i32, n as i32])
                }));
            let actual = cholesky_backend::evaluate(
                &inputs.iter().collect::<Vec<_>>(),
                &[n, n],
                kind,
                &stream,
            )
            .unwrap();
            equivalent(
                &actual,
                &expanded(&inputs, n, kind),
                &format!("exceptional {matrix:?} {kind:?}"),
            );
        }
    }
    let matrices = [[4.0f32, 0.0, 1.0, 3.0], [9.0, 99.0, -0.5, 2.0]];
    for kind in [
        None,
        Some(CholeskyAdKind::Jvp),
        Some(CholeskyAdKind::Vjp),
        Some(CholeskyAdKind::Mixed),
        Some(CholeskyAdKind::VjpJvp),
    ] {
        let count = match kind {
            None => 1,
            Some(CholeskyAdKind::Jvp | CholeskyAdKind::Vjp) => 2,
            _ => 4,
        };
        let mut batched = vec![Array::from_slice(&matrices.concat(), &[1, 2, 2, 2])];
        let mut reference = Vec::new();
        batched.extend(
            (1..count).map(|index| Array::from_slice(&[index as f32 * 0.07; 8], &[1, 2, 2, 2])),
        );
        for matrix in &matrices {
            let mut inputs = vec![Array::from_slice(matrix, &[2, 2])];
            inputs.extend(
                (1..count).map(|index| Array::from_slice(&[index as f32 * 0.07; 4], &[2, 2])),
            );
            reference.push(expanded(&inputs, 2, kind));
        }
        let actual = cholesky_backend::evaluate(
            &batched.iter().collect::<Vec<_>>(),
            &[1, 2, 2, 2],
            kind,
            &stream,
        )
        .unwrap();
        let expected = ops::stack_device(&reference, &stream)
            .unwrap()
            .reshape_device(&[1, 2, 2, 2], &stream)
            .unwrap();
        equivalent(&actual, &expected, &format!("batch {kind:?}"));
    }
}

// Includes graph construction, execution, and host result export. The allocator
// metric is the absolute active Metal-array peak, excluding its reusable cache.
#[test]
#[ignore = "opt-in complete-call benchmark"]
fn native_metal_cholesky_complete_call_profile() {
    let _guard = mlx_execution_guard();
    let n = std::env::var("QUABLA_CHOLESKY_PROFILE_N")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let reverse = std::env::var("QUABLA_CHOLESKY_PROFILE_VJP").unwrap() == "1";
    let native = std::env::var("QUABLA_CHOLESKY_PROFILE_NATIVE").unwrap() == "1";
    let kind = reverse.then_some(CholeskyAdKind::Vjp);
    let matrix = (0..n * n)
        .map(|i| {
            if i / n == i % n {
                3.0 + (i / n) as f32 * 0.1
            } else {
                0.03 * ((i * 7 % 11) as f32 - 5.0)
            }
        })
        .collect::<Vec<_>>();
    let mut inputs = vec![Array::from_slice(&matrix, &[n as i32, n as i32])];
    if reverse {
        inputs.push(Array::from_slice(
            &vec![0.3f32; n * n],
            &[n as i32, n as i32],
        ));
    }
    transforms::eval(&inputs).unwrap();
    let stream = StreamOrDevice::gpu();
    for sample in 0..7 {
        mlx_rs::memory::reset_peak_memory().unwrap();
        let start = std::time::Instant::now();
        let result = if native {
            cholesky_backend::evaluate(&inputs.iter().collect::<Vec<_>>(), &[n, n], kind, &stream)
                .unwrap()
        } else {
            expanded(&inputs, n, kind)
        };
        result.eval().unwrap();
        let checksum = result
            .as_slice::<f32>()
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        std::hint::black_box(checksum);
        let elapsed_ns = start.elapsed().as_nanos();
        let peak = mlx_rs::memory::peak_memory().unwrap();
        println!("{{\"n\":{n},\"vjp\":{reverse},\"native\":{native},\"sample\":{sample},\"elapsed_ns\":{elapsed_ns},\"peak_active_device_bytes\":{peak},\"checksum\":{checksum}}}");
    }
}
