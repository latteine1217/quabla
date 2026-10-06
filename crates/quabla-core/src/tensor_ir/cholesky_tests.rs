use super::*;

fn matrix(n: usize, dtype: TensorDType) -> Result<DynamicTensor, String> {
    let values = (0..n * n)
        .map(|i| {
            let row = i / n;
            let column = i % n;
            if row == column {
                4.0 + row as f64
            } else {
                0.07 * (row + column + 1) as f64
            }
        })
        .collect();
    DynamicTensor::with_dtype(vec![n, n], values, dtype)
}

fn assert_close(actual: &DynamicTensor, expected: &DynamicTensor) {
    assert_eq!(actual.shape(), expected.shape());
    // Runtime jets reuse one reciprocal where the expansion divides, so they
    // agree to roundoff of the operand dtype rather than bit for bit.
    let tolerance = if actual.dtype() == TensorDType::F32 || expected.dtype() == TensorDType::F32 {
        1e-6
    } else {
        1e-12
    };
    for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
        assert!(
            actual.to_bits() == expected.to_bits()
                || actual.is_nan() && expected.is_nan()
                || (actual - expected).abs() <= tolerance + tolerance * expected.abs(),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn compact_cholesky_primal_preserves_staged_recurrence() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for n in [1, 2, 4, 8, 16] {
            let mut ir = TensorIr::new();
            let input = ir.input_typed("matrix", vec![n, n], dtype)?;
            let native = ir.cholesky(input)?;
            assert_eq!(ir.node_count(), 2);
            let expanded = ir.expanded_cholesky(input)?;
            let inputs = BTreeMap::from([("matrix".into(), matrix(n, dtype)?)]);
            let actual = ir.evaluate(native, &inputs)?;
            let expected = ir.evaluate(expanded, &inputs)?;
            assert_eq!(actual.dtype(), expected.dtype());
            assert_eq!(
                actual
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        for values in [
            vec![0.0],
            vec![-1.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![2.0, 99.0, 0.5, 2.0],
            vec![0.0; 4],
        ] {
            let n = if values.len() == 1 { 1 } else { 2 };
            let mut ir = TensorIr::new();
            let input = ir.input_typed("matrix", vec![n, n], dtype)?;
            let native = ir.cholesky(input)?;
            let expanded = ir.expanded_cholesky(input)?;
            let inputs = BTreeMap::from([(
                "matrix".into(),
                DynamicTensor::with_dtype(vec![n, n], values, dtype)?,
            )]);
            match (ir.evaluate(native, &inputs), ir.evaluate(expanded, &inputs)) {
                (Ok(actual), Ok(expected)) => assert_close(&actual, &expected),
                (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                other => panic!("native/reference mismatch: {other:?}"),
            }
        }
    }
    Ok(())
}

#[test]
fn compact_cholesky_derivatives_match_reference_and_stay_bounded() -> Result<(), String> {
    for n in [1, 2, 4, 8] {
        let mut native = TensorIr::new();
        let input = native.input("matrix", vec![n, n])?;
        let output = native.cholesky(input)?;
        let mut reference = TensorIr::new();
        let reference_input = reference.input("matrix", vec![n, n])?;
        let reference_output = reference.expanded_cholesky(reference_input)?;
        let inputs = BTreeMap::from([("matrix".into(), matrix(n, TensorDType::F64)?)]);
        let direction = DynamicTensor::new(
            vec![n, n],
            (0..n * n)
                .map(|i| (i as f64 + 1.0) * 0.013 - 0.04)
                .collect(),
        )?;
        let directions = BTreeMap::from([("matrix".into(), direction)]);
        let cotangent = DynamicTensor::new(
            vec![n, n],
            (0..n * n).map(|i| (i as f64 + 2.0) * 0.031).collect(),
        )?;
        let (_, actual_jvp) = native.jvp(output, &inputs, &directions)?;
        let (_, expected_jvp) = reference.jvp(reference_output, &inputs, &directions)?;
        assert_close(&actual_jvp, &expected_jvp);
        let (_, actual_vjp) = native.value_and_vjp(output, &inputs, cotangent.clone())?;
        let (_, expected_vjp) =
            reference.value_and_vjp(reference_output, &inputs, cotangent.clone())?;
        assert_close(&actual_vjp["matrix"], &expected_vjp["matrix"]);

        let forward = native.symbolic_jvp_with_tangent_inputs(
            output,
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        assert_eq!(forward.graph.node_count(), 4);
        let mut seeded = inputs.clone();
        seeded.insert("direction".into(), directions["matrix"].clone());
        assert_close(
            &forward.graph.evaluate(forward.tangent, &seeded)?,
            &expected_jvp,
        );
        let reverse = native.symbolic_vjp(output, "cotangent")?;
        assert!(reverse.graph.node_count() <= 6);
        seeded.insert("cotangent".into(), cotangent.clone());
        assert_close(
            &reverse
                .graph
                .evaluate(reverse.gradients["matrix"], &seeded)?,
            &expected_vjp["matrix"],
        );

        let actual_mixed = native.evaluate_mixed(output, &inputs, &directions, &directions)?;
        let expected_mixed =
            reference.evaluate_mixed(reference_output, &inputs, &directions, &directions)?;
        assert_close(&actual_mixed.mixed, &expected_mixed.mixed);
        let hvp = reverse.graph.symbolic_jvp_with_tangent_inputs(
            reverse.gradients["matrix"],
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        assert!(hvp.graph.node_count() <= 12);
        let old_reverse = reference.symbolic_vjp(reference_output, "cotangent")?;
        let old_hvp = old_reverse.graph.symbolic_jvp_with_tangent_inputs(
            old_reverse.gradients["matrix"],
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        assert_close(
            &hvp.graph.evaluate(hvp.tangent, &seeded)?,
            &old_hvp.graph.evaluate(old_hvp.tangent, &seeded)?,
        );
        let second_reverse = reverse
            .graph
            .symbolic_vjp(reverse.gradients["matrix"], "second_cotangent")?;
        let old_second_reverse = old_reverse
            .graph
            .symbolic_vjp(old_reverse.gradients["matrix"], "second_cotangent")?;
        seeded.insert("second_cotangent".into(), directions["matrix"].clone());
        assert_close(
            &second_reverse
                .graph
                .evaluate(second_reverse.gradients["matrix"], &seeded)?,
            &old_second_reverse
                .graph
                .evaluate(old_second_reverse.gradients["matrix"], &seeded)?,
        );
        if n <= 2 {
            let third = hvp.graph.symbolic_jvp_with_tangent_inputs(
                hvp.tangent,
                &BTreeMap::from([("matrix".into(), "third_direction".into())]),
            )?;
            let old_third = old_hvp.graph.symbolic_jvp_with_tangent_inputs(
                old_hvp.tangent,
                &BTreeMap::from([("matrix".into(), "third_direction".into())]),
            )?;
            seeded.insert("third_direction".into(), directions["matrix"].clone());
            assert_close(
                &third.graph.evaluate(third.tangent, &seeded)?,
                &old_third.graph.evaluate(old_third.tangent, &seeded)?,
            );
        }
    }
    Ok(())
}

#[test]
fn compact_cholesky_shape_checks_and_native_f32_derivatives() -> Result<(), String> {
    let mut ir = TensorIr::new();
    assert!(ir.input("empty_matrix", vec![0, 0]).is_err());
    assert!(ir.input("empty_batch", vec![0, 2, 2]).is_err());
    assert!(DynamicTensor::new(vec![0, 0], vec![]).is_err());
    for shape in [vec![], vec![4], vec![2, 3], vec![2, 2, 3]] {
        let input = ir.input(format!("bad_{shape:?}"), shape)?;
        assert!(ir.cholesky(input).is_err());
    }
    // Float32 jets round every operation to f32 and keep the derivative graphs
    // bounded. They agree with the scalar expansion to f32 roundoff: the jets
    // keep the quotient rule over the squared pivot, while the expansion's
    // division rules avoid squaring the denominator.
    let mut native = TensorIr::new();
    let input = native.input_typed("matrix", vec![4, 4], TensorDType::F32)?;
    let output = native.cholesky(input)?;
    let mut reference = TensorIr::new();
    let reference_input = reference.input_typed("matrix", vec![4, 4], TensorDType::F32)?;
    let reference_output = reference.expanded_cholesky(reference_input)?;
    let inputs = BTreeMap::from([("matrix".into(), matrix(4, TensorDType::F32)?)]);
    let direction = DynamicTensor::with_dtype(
        vec![4, 4],
        (0..16).map(|i| (i as f64 + 1.0) * 0.013 - 0.04).collect(),
        TensorDType::F32,
    )?;
    let directions = BTreeMap::from([("matrix".into(), direction.clone())]);
    let cotangent = DynamicTensor::with_dtype(
        vec![4, 4],
        (0..16).map(|i| (i as f64 + 2.0) * 0.031).collect(),
        TensorDType::F32,
    )?;
    let plan = native.compile_cpu(output)?;
    let expected_plan = reference.compile_cpu(reference_output)?;
    assert_close(
        &plan.value_and_vjp(&inputs, cotangent.clone())?.1["matrix"],
        &expected_plan.value_and_vjp(&inputs, cotangent.clone())?.1["matrix"],
    );
    assert_close(
        &native.jvp(output, &inputs, &directions)?.1,
        &reference.jvp(reference_output, &inputs, &directions)?.1,
    );
    let reverse = native.symbolic_vjp(output, "cotangent")?;
    assert!(reverse.graph.node_count() <= 6);
    let hvp = reverse.graph.symbolic_jvp_with_tangent_inputs(
        reverse.gradients["matrix"],
        &BTreeMap::from([("matrix".into(), "direction".into())]),
    )?;
    assert!(hvp.graph.node_count() <= 12);
    let old_reverse = reference.symbolic_vjp(reference_output, "cotangent")?;
    let old_hvp = old_reverse.graph.symbolic_jvp_with_tangent_inputs(
        old_reverse.gradients["matrix"],
        &BTreeMap::from([("matrix".into(), "direction".into())]),
    )?;
    let mut seeded = inputs.clone();
    seeded.insert("direction".into(), direction);
    seeded.insert("cotangent".into(), cotangent);
    assert_close(
        &reverse
            .graph
            .evaluate(reverse.gradients["matrix"], &seeded)?,
        &old_reverse
            .graph
            .evaluate(old_reverse.gradients["matrix"], &seeded)?,
    );
    let actual = hvp.graph.evaluate(hvp.tangent, &seeded)?;
    assert_eq!(actual.dtype(), TensorDType::F32);
    assert_close(&actual, &old_hvp.graph.evaluate(old_hvp.tangent, &seeded)?);
    let actual = native.evaluate_mixed(output, &inputs, &directions, &directions)?;
    let expected = reference.evaluate_mixed(reference_output, &inputs, &directions, &directions)?;
    assert_close(&actual.mixed, &expected.mixed);
    Ok(())
}

#[test]
fn compact_cholesky_exceptional_derivatives_preserve_reference() -> Result<(), String> {
    for values in [
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1e-300],
        vec![2.0, 99.0, 0.5, 2.0],
        vec![0.0; 4],
    ] {
        let n = if values.len() == 1 { 1 } else { 2 };
        let mut ir = TensorIr::new();
        let input = ir.input("matrix", vec![n, n])?;
        let native = ir.cholesky(input)?;
        let expanded = ir.expanded_cholesky(input)?;
        let inputs = BTreeMap::from([("matrix".into(), DynamicTensor::new(vec![n, n], values)?)]);
        let directions =
            BTreeMap::from([("matrix".into(), DynamicTensor::filled(vec![n, n], 0.37)?)]);
        match (
            ir.jvp(native, &inputs, &directions),
            ir.jvp(expanded, &inputs, &directions),
        ) {
            (Ok(actual), Ok(expected)) => assert_close(&actual.1, &expected.1),
            (Err(actual), Err(expected)) => assert_eq!(actual, expected),
            other => panic!("exceptional JVP mismatch: {other:?}"),
        }
        let cotangent = DynamicTensor::filled(vec![n, n], 0.23)?;
        match (
            ir.value_and_vjp(native, &inputs, cotangent.clone()),
            ir.value_and_vjp(expanded, &inputs, cotangent),
        ) {
            (Ok(actual), Ok(expected)) => assert_close(&actual.1["matrix"], &expected.1["matrix"]),
            (Err(actual), Err(expected)) => assert_eq!(actual, expected),
            other => panic!("exceptional VJP mismatch: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn compact_cholesky_symbolic_exceptional_ad_matches_quotient_rules() -> Result<(), String> {
    for values in [
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1e-300],
        vec![f64::INFINITY, 0.0, 1.0, 2.0],
        vec![f64::NAN, 0.0, 1.0, 2.0],
        vec![1e-300, 0.0, 1e-150, 2.0],
        vec![0.0; 4],
    ] {
        let n = if values.len() == 1 { 1 } else { 2 };
        let mut native = TensorIr::new();
        let input = native.input("matrix", vec![n, n])?;
        let output = native.cholesky(input)?;
        let mut reference = TensorIr::new();
        let input = reference.input("matrix", vec![n, n])?;
        let old = reference.expanded_cholesky(input)?;
        let inputs = BTreeMap::from([
            ("matrix".into(), DynamicTensor::new(vec![n, n], values)?),
            ("direction".into(), DynamicTensor::filled(vec![n, n], 0.37)?),
            ("cotangent".into(), DynamicTensor::filled(vec![n, n], 0.23)?),
        ]);
        let forward = native.symbolic_jvp_with_tangent_inputs(
            output,
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        let old_forward = reference.symbolic_jvp_with_tangent_inputs(
            old,
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        let reverse = native.symbolic_vjp(output, "cotangent")?;
        let old_reverse = reference.symbolic_vjp(old, "cotangent")?;
        let hvp = reverse.graph.symbolic_jvp_with_tangent_inputs(
            reverse.gradients["matrix"],
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        let old_hvp = old_reverse.graph.symbolic_jvp_with_tangent_inputs(
            old_reverse.gradients["matrix"],
            &BTreeMap::from([("matrix".into(), "direction".into())]),
        )?;
        for (actual, expected) in [
            (
                forward.graph.evaluate(forward.tangent, &inputs),
                old_forward.graph.evaluate(old_forward.tangent, &inputs),
            ),
            (
                reverse.graph.evaluate(reverse.gradients["matrix"], &inputs),
                old_reverse
                    .graph
                    .evaluate(old_reverse.gradients["matrix"], &inputs),
            ),
            (
                hvp.graph.evaluate(hvp.tangent, &inputs),
                old_hvp.graph.evaluate(old_hvp.tangent, &inputs),
            ),
        ] {
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => assert_close(&actual, &expected),
                (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                other => panic!("symbolic exceptional AD mismatch: {other:?}"),
            }
        }
    }
    Ok(())
}

#[test]
fn compact_cholesky_internal_batches_match_independent_matrices() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        let mut native = TensorIr::new();
        let input = native.input_typed("matrix", vec![2, 2, 2], dtype)?;
        let output = native.cholesky(input)?;
        let reference = native.expanded_cholesky(input)?;
        let point = DynamicTensor::with_dtype(
            vec![2, 2, 2],
            vec![4.0, 2.0, 2.0, 5.0, 3.0, -19.0, 0.5, 2.0],
            dtype,
        )?;
        let inputs = BTreeMap::from([("matrix".into(), point)]);
        assert_close(
            &native.evaluate(output, &inputs)?,
            &native.evaluate(reference, &inputs)?,
        );
        let cotangent = DynamicTensor::filled(vec![2, 2, 2], 0.23)?;
        assert_close(
            &native.value_and_vjp(output, &inputs, cotangent.clone())?.1["matrix"],
            &native.value_and_vjp(reference, &inputs, cotangent)?.1["matrix"],
        );
        let tangent = DynamicTensor::filled(vec![2, 2, 2], 0.37)?;
        let directions = BTreeMap::from([("matrix".into(), tangent)]);
        assert_close(
            &native.jvp(output, &inputs, &directions)?.1,
            &native.jvp(reference, &inputs, &directions)?.1,
        );
    }
    Ok(())
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
#[test]
fn compact_cholesky_mlx_keeps_resident_reference_results() -> Result<(), String> {
    if !crate::test_support::Gate::Mlx.enabled() {
        return Ok(());
    }
    let backend = MlxBackend;
    for n in [1, 2, 4] {
        let mut native = TensorIr::new();
        let input = native.input("matrix", vec![n, n])?;
        let output = native.cholesky(input)?;
        let mut old = TensorIr::new();
        let input = old.input("matrix", vec![n, n])?;
        let reference = old.expanded_cholesky(input)?;
        let inputs = BTreeMap::from([("matrix".into(), matrix(n, TensorDType::F64)?)]);
        let actual = backend.execute(&native.compile_cpu(output)?, &inputs)?;
        let expected = backend.execute(&old.compile_cpu(reference)?, &inputs)?;
        assert_eq!(
            actual
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let reverse = native.symbolic_vjp(output, "cotangent")?;
        let old_reverse = old.symbolic_vjp(reference, "cotangent")?;
        let mut inputs = inputs;
        inputs.insert("cotangent".into(), DynamicTensor::filled(vec![n, n], 0.37)?);
        let actual = backend.execute(
            &reverse.graph.compile_cpu(reverse.gradients["matrix"])?,
            &inputs,
        )?;
        let expected = backend.execute(
            &old_reverse
                .graph
                .compile_cpu(old_reverse.gradients["matrix"])?,
            &inputs,
        )?;
        assert_eq!(
            actual
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}
