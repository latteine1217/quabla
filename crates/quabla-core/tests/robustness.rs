use quabla_core::tensor_ir::{DynamicTensor, TensorIr, UnaryMathKind};

#[test]
fn finite_mean_survives_overflow_in_full_and_axis_reductions() -> Result<(), String> {
    for (shape, values, expected) in [
        (vec![2, 3], vec![1e308; 6], 1e308),
        (vec![3], vec![f64::MAX; 3], f64::MAX),
        (vec![4], vec![1e308, 1e308, -1e308, -1e308], 0.0),
        (vec![2], vec![f64::from_bits(1); 2], f64::from_bits(1)),
    ] {
        let input = DynamicTensor::new(shape.clone(), values)?;
        let mut graph = TensorIr::new();
        let x = graph.input("x", shape)?;
        let output = graph.mean(x)?;
        let result = graph
            .compile_cpu(output)?
            .evaluate(&std::collections::BTreeMap::from([(
                "x".to_string(),
                input.clone(),
            )]))?;
        assert_eq!(result.data().as_ref(), &[expected]);
        let axis = graph.mean_axis(x, -1)?;
        let result = graph
            .compile_cpu(axis)?
            .evaluate(&std::collections::BTreeMap::from([(
                "x".to_string(),
                input,
            )]))?;
        assert!(
            result.data().iter().all(|value| {
                *value == expected || (*value / expected - 1.0).abs() <= 2.0 * f64::EPSILON
            }),
            "axis result {:?}, expected {expected}",
            result.data()
        );
    }
    Ok(())
}

#[test]
fn invalid_slice_error_formatting_cannot_overflow() -> Result<(), String> {
    let value = DynamicTensor::new(vec![2], vec![1.0, 2.0])?;
    for axis in [0, 1, usize::MAX] {
        for start in [0, 2, usize::MAX] {
            assert!(value.slice_axis(axis, start, usize::MAX).is_err());
        }
    }
    Ok(())
}

#[test]
fn inferred_shapes_and_factory_capacity_are_checked() -> Result<(), String> {
    assert!(DynamicTensor::filled(vec![isize::MAX as usize], 0.0).is_err());
    let extent = 1usize << (usize::BITS / 2);
    let mut graph = TensorIr::new();
    let a = graph.input("a", vec![extent, 1])?;
    let b = graph.input("b", vec![1, extent])?;
    assert!(graph.add(a, b).is_err());
    assert!(graph.matmul(a, b).is_err());
    // A rejected builder operation must leave the existing graph usable.
    let output = graph.sum(a)?;
    graph.compile_cpu(output)?;
    Ok(())
}

#[test]
fn arccosh_rejects_every_argument_below_one() {
    // musl's `acosh` picks its branch from the exponent alone; the kernel must
    // keep large negative arguments out of it.
    for x in [
        -77698.54311101382,
        -3888930.75,
        -(2.0f64.powi(26)),
        -1e300,
        0.5,
    ] {
        assert!(UnaryMathKind::Arccosh.evaluate(x).is_nan(), "acosh({x})");
    }
    assert!(UnaryMathKind::Arccosh.evaluate(f64::NAN).is_nan());
    assert_eq!(UnaryMathKind::Arccosh.evaluate(1.0), 0.0);
    assert_eq!(
        UnaryMathKind::Arccosh.evaluate(f64::INFINITY),
        f64::INFINITY
    );
}
