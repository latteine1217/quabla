use quabla_core::tensor_ir::{DynamicTensor, TensorDType, TensorIr};
use std::collections::BTreeMap;

#[test]
fn finite_primal_with_hidden_derivative_overflow_keeps_mixed_dual_semantics() -> Result<(), String>
{
    for (case, point, tangent) in [(0, 1e150, 1e160), (1, 1e-150, 1e200), (2, 1e-200, 1.0)] {
        let build = |force_mixed: bool| -> Result<_, String> {
            let mut graph = TensorIr::new();
            let x = graph.input("x", vec![2])?;
            if force_mixed {
                let unused = graph.scalar_constant(0.0);
                graph.cast(unused, TensorDType::F64)?;
            }
            let intermediate = if case == 0 {
                graph.mul(x, x)?
            } else {
                let coefficient = graph.scalar_constant(1e200);
                let scaled = graph.mul(coefficient, x)?;
                if case == 1 {
                    graph.mul(scaled, x)?
                } else {
                    graph.sin(scaled)?
                }
            };
            let zero = graph.scalar_constant(0.0);
            let masked = graph.mul(intermediate, zero)?;
            let loss = graph.sum(masked)?;
            Ok((graph, loss))
        };
        let (graph, loss) = build(false)?;
        let (reference, ref_loss) = build(true)?;
        let inputs = BTreeMap::from([("x".into(), DynamicTensor::filled(vec![2], point)?)]);
        assert_eq!(graph.evaluate(loss, &inputs)?.data().as_ref(), &[0.0]);
        let direction = DynamicTensor::filled(vec![2], tangent)?;
        let actual = graph.hvp_scalar(loss, "x", &inputs, direction.clone())?;
        let expected = reference.hvp_scalar(ref_loss, "x", &inputs, direction)?;
        assert!(expected.data().iter().any(|value| !value.is_finite()));
        let actual_h = graph.hessian_scalar(loss, "x", &inputs)?;
        let expected_h = reference.hessian_scalar(ref_loss, "x", &inputs)?;
        if case == 2 {
            assert!(expected_h.iter().flatten().any(|value| !value.is_finite()));
        }
        for (actual, expected) in actual
            .data()
            .iter()
            .zip(expected.data().iter())
            .chain(actual_h.iter().flatten().zip(expected_h.iter().flatten()))
        {
            assert!(
                actual.is_nan() && expected.is_nan() || actual.to_bits() == expected.to_bits(),
                "case {case}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn finite_hvp_and_hessian_match_mixed_dual_reference() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for exponent in [0, 1, 2, 3, 7] {
            let build = |force_mixed: bool| -> Result<_, String> {
                let mut graph = TensorIr::new();
                let x = graph.input_typed("x", vec![4], dtype)?;
                if force_mixed {
                    let unused = graph.scalar_constant(0.0);
                    graph.cast(unused, TensorDType::F64)?;
                }
                let p = graph.powi(x, exponent)?;
                let s = graph.sin(x)?;
                let term = graph.mul(p, s)?;
                let sum = graph.sum(term)?;
                let loss = graph.mul(sum, sum)?;
                Ok((graph, loss))
            };
            let (graph, loss) = build(false)?;
            let (reference, ref_loss) = build(true)?;
            let inputs = BTreeMap::from([(
                "x".into(),
                DynamicTensor::new(vec![4], vec![-0.7, 0.0, 0.2, 1.1])?,
            )]);
            let direction = DynamicTensor::new(vec![4], vec![0.3, -0.2, 0.8, -0.4])?;
            let actual = graph.hvp_scalar(loss, "x", &inputs, direction.clone())?;
            let expected = reference.hvp_scalar(ref_loss, "x", &inputs, direction)?;
            let actual_h = graph.hessian_scalar(loss, "x", &inputs)?;
            let expected_h = reference.hessian_scalar(ref_loss, "x", &inputs)?;
            for (a, b) in actual
                .data()
                .iter()
                .zip(expected.data().iter())
                .chain(actual_h.iter().flatten().zip(expected_h.iter().flatten()))
            {
                if dtype == TensorDType::F32 {
                    assert_eq!(a.to_bits(), b.to_bits());
                } else {
                    assert!(
                        (a - b).abs() <= 1e-11 * (1.0 + b.abs()),
                        "{a} != {b}, exponent {exponent}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn non_finite_hvp_and_hessian_preserve_the_mixed_dual_route() -> Result<(), String> {
    for point in [f64::NAN, f64::INFINITY, 1e150, 0.5] {
        for tangent in [1.0, f64::INFINITY] {
            let build = |force_mixed: bool| -> Result<_, String> {
                let mut graph = TensorIr::new();
                let x = graph.input("x", vec![1])?;
                if force_mixed {
                    let unused = graph.scalar_constant(0.0);
                    graph.cast(unused, TensorDType::F64)?;
                }
                let p = graph.powi(x, 3)?;
                let loss = graph.sum(p)?;
                Ok((graph, loss))
            };
            let (graph, loss) = build(false)?;
            let (reference, ref_loss) = build(true)?;
            let inputs = BTreeMap::from([("x".into(), DynamicTensor::new(vec![1], vec![point])?)]);
            let direction = DynamicTensor::new(vec![1], vec![tangent])?;
            let actual = graph.hvp_scalar(loss, "x", &inputs, direction.clone())?;
            let expected = reference.hvp_scalar(ref_loss, "x", &inputs, direction)?;
            let actual_h = graph.hessian_scalar(loss, "x", &inputs)?;
            let expected_h = reference.hessian_scalar(ref_loss, "x", &inputs)?;
            for (a, b) in actual
                .data()
                .iter()
                .zip(expected.data().iter())
                .chain(actual_h.iter().flatten().zip(expected_h.iter().flatten()))
            {
                assert!(
                    a.is_nan() && b.is_nan() || a.to_bits() == b.to_bits(),
                    "{a} != {b}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn smooth_matrix_and_reduction_higher_derivatives_match_mixed_duals() -> Result<(), String> {
    let build = |force_mixed: bool| -> Result<_, String> {
        let mut graph = TensorIr::new();
        let x = graph.input("x", vec![2, 3])?;
        if force_mixed {
            let unused = graph.scalar_constant(0.0);
            graph.cast(unused, TensorDType::F64)?;
        }
        let weights = graph.constant(
            DynamicTensor::new(vec![3, 2], vec![0.2, -0.1, 0.7, 0.5, -0.3, 0.4])?,
            false,
        );
        let matrix = graph.matmul(x, weights)?;
        let transposed = graph.transpose(matrix, None)?;
        let flattened = graph.reshape(transposed, vec![4])?;
        let exponential = graph.exp(flattened)?;
        let bounded = graph.tanh(exponential)?;
        let wave = graph.cos(flattened)?;
        let difference = graph.sub(bounded, wave)?;
        let mean = graph.mean(difference)?;
        let axis_mean = graph.mean_axis(transposed, 0)?;
        let axis_sum = graph.sum_axis(axis_mean, 0)?;
        let sum = graph.add(mean, axis_sum)?;
        let loss = graph.mul(sum, sum)?;
        Ok((graph, loss))
    };
    let (graph, loss) = build(false)?;
    let (reference, ref_loss) = build(true)?;
    let inputs = BTreeMap::from([(
        "x".into(),
        DynamicTensor::new(vec![2, 3], vec![-0.4, 0.2, 0.8, 0.1, -0.3, 0.5])?,
    )]);
    let direction = DynamicTensor::new(vec![2, 3], vec![0.3, 0.6, -0.7, 0.2, -0.4, 0.9])?;
    let actual = graph.hvp_scalar(loss, "x", &inputs, direction.clone())?;
    let expected = reference.hvp_scalar(ref_loss, "x", &inputs, direction)?;
    let actual_h = graph.hessian_scalar(loss, "x", &inputs)?;
    let expected_h = reference.hessian_scalar(ref_loss, "x", &inputs)?;
    for (a, b) in actual
        .data()
        .iter()
        .zip(expected.data().iter())
        .chain(actual_h.iter().flatten().zip(expected_h.iter().flatten()))
    {
        assert!((a - b).abs() < 1e-11 * (1.0 + b.abs()), "{a} != {b}");
    }
    Ok(())
}
