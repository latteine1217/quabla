//! Batched `solve` and the `Linalg` decompositions (determinant sign and
//! log-magnitude, symmetric eigendecomposition): CPU values, the numeric and
//! symbolic derivative paths, vmap batching, and backend validation.

use std::collections::BTreeMap;

use quabla_core::tensor_ir::{DynamicTensor, LinalgKind, TensorDType, TensorIr, TensorNodeId};

fn tensor(shape: &[usize], data: &[f64]) -> DynamicTensor {
    DynamicTensor::new(shape.to_vec(), data.to_vec()).expect("valid test tensor")
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} (tolerance {tolerance})"
        );
    }
}

/// Two well-conditioned 3x3 systems with two right-hand-side columns each.
fn batched_system() -> (DynamicTensor, DynamicTensor) {
    let matrices = tensor(
        &[2, 3, 3],
        &[
            4.0, 1.0, -0.5, 0.3, 5.0, 1.0, -1.0, 0.2, 3.0, //
            2.0, -1.0, 0.0, 0.5, 6.0, -2.0, 1.0, 1.0, 4.0,
        ],
    );
    let rhs = tensor(
        &[2, 3, 2],
        &[
            1.0, -2.0, 0.5, 3.0, 2.0, 1.0, -1.0, 0.0, 2.0, 1.5, 0.25, -3.0,
        ],
    );
    (matrices, rhs)
}

fn slice_matrix(tensor: &DynamicTensor, index: usize) -> DynamicTensor {
    let shape = &tensor.shape()[1..];
    let size = shape.iter().product::<usize>();
    DynamicTensor::new(
        shape.to_vec(),
        tensor.data()[index * size..(index + 1) * size].to_vec(),
    )
    .expect("valid slice")
}

fn evaluate(
    graph: &TensorIr,
    output: TensorNodeId,
    inputs: &[(&str, &DynamicTensor)],
) -> Result<DynamicTensor, String> {
    let inputs = inputs
        .iter()
        .map(|(name, value)| (name.to_string(), (*value).clone()))
        .collect::<BTreeMap<_, _>>();
    graph.evaluate(output, &inputs)
}

#[test]
fn batched_solve_matches_per_matrix_solves_and_differentiates() -> Result<(), String> {
    let (matrices, rhs) = batched_system();
    let mut graph = TensorIr::new();
    let a = graph.input("a", vec![2, 3, 3])?;
    let b = graph.input("b", vec![2, 3, 2])?;
    let x = graph.solve(a, b)?;
    let loss = graph.mul(x, x)?;
    let loss = graph.sum(loss)?;
    let batched = evaluate(&graph, x, &[("a", &matrices), ("b", &rhs)])?;
    assert_eq!(batched.shape(), &[2, 3, 2]);

    let mut single = TensorIr::new();
    let a1 = single.input("a", vec![3, 3])?;
    let b1 = single.input("b", vec![3, 2])?;
    let x1 = single.solve(a1, b1)?;
    let mut expected = Vec::new();
    for index in 0..2 {
        let value = evaluate(
            &single,
            x1,
            &[
                ("a", &slice_matrix(&matrices, index)),
                ("b", &slice_matrix(&rhs, index)),
            ],
        )?;
        expected.extend(value.data().iter());
    }
    assert_eq!(batched.data().as_ref(), expected.as_slice());

    // The numeric and symbolic reverse modes agree on batched operands.
    let inputs = BTreeMap::from([
        ("a".to_string(), matrices.clone()),
        ("b".to_string(), rhs.clone()),
    ]);
    let numeric = graph.vjp(loss, &inputs, tensor(&[], &[1.0]))?;
    let symbolic = graph.symbolic_vjp(loss, "seed")?;
    let mut symbolic_inputs = inputs.clone();
    symbolic_inputs.insert("seed".into(), tensor(&[], &[1.0]));
    for name in ["a", "b"] {
        let value = symbolic
            .graph
            .evaluate(symbolic.gradients[name], &symbolic_inputs)?;
        assert_close(value.data().as_ref(), numeric[name].data().as_ref(), 1e-13);
    }
    // Central differences of the loss along a matrix direction.
    let direction = tensor(
        &[2, 3, 3],
        &[
            0.3, -0.1, 0.2, 0.0, 0.5, -0.4, 0.1, 0.2, -0.3, 0.2, 0.1, 0.0, -0.2, 0.3, 0.1, 0.4,
            0.0, -0.1,
        ],
    );
    let (_, tangent) = graph.jvp(
        loss,
        &inputs,
        &BTreeMap::from([
            ("a".to_string(), direction.clone()),
            ("b".to_string(), DynamicTensor::filled(vec![2, 3, 2], 0.0)?),
        ]),
    )?;
    let step = 1e-6;
    let shifted = |sign: f64| -> Result<f64, String> {
        let data = matrices
            .data()
            .iter()
            .zip(direction.data().iter())
            .map(|(value, delta)| value + sign * step * delta)
            .collect::<Vec<_>>();
        let value = evaluate(
            &graph,
            loss,
            &[("a", &tensor(&[2, 3, 3], &data)), ("b", &rhs)],
        )?;
        Ok(value.data()[0])
    };
    let difference = (shifted(1.0)? - shifted(-1.0)?) / (2.0 * step);
    assert_close(tangent.data().as_ref(), &[difference], 1e-7);
    Ok(())
}

#[test]
fn solve_shape_requires_equal_batch_axes() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let a = graph.input("a", vec![2, 3, 3])?;
    let b = graph.input("b", vec![3, 3, 1])?;
    let error = graph.solve(a, b).expect_err("batch axes differ");
    assert!(error.contains("same batch axes"), "{error}");
    let c = graph.input("c", vec![3, 1])?;
    let error = graph.solve(a, c).expect_err("ranks differ");
    assert!(error.contains("same rank"), "{error}");
    Ok(())
}

#[test]
fn slogdet_kinds_follow_numpy_conventions() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let a = graph.input("a", vec![4, 2, 2])?;
    let sign = graph.linalg(a, LinalgKind::DetSign)?;
    let log_abs = graph.linalg(a, LinalgKind::LogAbsDet)?;
    // det = -2 (pivoting swaps rows), det = 6, singular, and NaN.
    let matrices = tensor(
        &[4, 2, 2],
        &[
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            0.0,
            0.0,
            3.0,
            1.0,
            2.0,
            2.0,
            4.0,
            f64::NAN,
            0.0,
            0.0,
            1.0,
        ],
    );
    let signs = evaluate(&graph, sign, &[("a", &matrices)])?;
    let logs = evaluate(&graph, log_abs, &[("a", &matrices)])?;
    assert_eq!(&signs.data()[..3], &[-1.0, 1.0, 0.0]);
    assert!(signs.data()[3].is_nan());
    assert_close(&logs.data()[..2], &[2.0_f64.ln(), 6.0_f64.ln()], 1e-15);
    assert_eq!(logs.data()[2], f64::NEG_INFINITY);
    assert!(logs.data()[3].is_nan());
    assert_eq!(graph.node_shape(sign)?, vec![4]);
    Ok(())
}

#[test]
fn eigh_kinds_sort_and_normalize_signs() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let a = graph.input_typed("a", vec![2, 2], TensorDType::F32)?;
    let values = graph.linalg(a, LinalgKind::EighValues)?;
    let vectors = graph.linalg(a, LinalgKind::EighVectors)?;
    // [[2, 1], [1, 2]] (read through its symmetric part) has eigenvalues 1
    // and 3 with eigenvectors (1, -1) / sqrt(2) and (1, 1) / sqrt(2); ties
    // in magnitude make the first component positive.
    let matrix = tensor(&[2, 2], &[2.0, 0.0, 2.0, 2.0]).astype(TensorDType::F32);
    let w = evaluate(&graph, values, &[("a", &matrix)])?;
    let v = evaluate(&graph, vectors, &[("a", &matrix)])?;
    assert_eq!(w.dtype(), TensorDType::F32);
    assert_close(w.data().as_ref(), &[1.0, 3.0], 1e-7);
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(v.data().as_ref(), &[h, h, -h, h], 1e-7);
    let mut bad = TensorIr::new();
    let rectangular = bad.input("r", vec![2, 3])?;
    let error = bad
        .linalg(rectangular, LinalgKind::LogAbsDet)
        .expect_err("not square");
    assert!(error.contains("square"), "{error}");
    Ok(())
}

#[test]
fn numeric_and_symbolic_linalg_derivatives_agree() -> Result<(), String> {
    let matrix = tensor(&[3, 3], &[4.0, 1.0, -0.5, 1.0, 3.0, 0.2, -0.5, 0.2, 2.0]);
    let direction = tensor(&[3, 3], &[0.3, -0.1, 0.2, -0.1, 0.5, -0.4, 0.2, -0.4, -0.3]);
    for kind in [
        LinalgKind::LogAbsDet,
        LinalgKind::EighValues,
        LinalgKind::EighVectors,
        LinalgKind::DetSign,
    ] {
        let mut graph = TensorIr::new();
        let a = graph.input("a", vec![3, 3])?;
        let decomposed = graph.linalg(a, kind)?;
        let cubed = graph.powi(decomposed, 3)?;
        let loss = graph.sum(cubed)?;
        let inputs = BTreeMap::from([("a".to_string(), matrix.clone())]);
        let numeric = graph.vjp(loss, &inputs, tensor(&[], &[1.0]))?;
        let symbolic = graph.symbolic_vjp(loss, "seed")?;
        let mut symbolic_inputs = inputs.clone();
        symbolic_inputs.insert("seed".into(), tensor(&[], &[1.0]));
        let gradient = symbolic
            .graph
            .evaluate(symbolic.gradients["a"], &symbolic_inputs)?;
        assert_close(
            gradient.data().as_ref(),
            numeric["a"].data().as_ref(),
            1e-12,
        );
        let (_, tangent) = graph.jvp(
            loss,
            &inputs,
            &BTreeMap::from([("a".to_string(), direction.clone())]),
        )?;
        let projected = gradient
            .data()
            .iter()
            .zip(direction.data().iter())
            .map(|(g, d)| g * d)
            .sum::<f64>();
        assert_close(tangent.data().as_ref(), &[projected], 1e-12);
        // The numeric second-order (mixed jet) path against the symbolic one.
        let hessian = graph.hessian_scalar(loss, "a", &inputs)?;
        let step = 1e-5;
        for column in [0, 4, 7] {
            let shifted = |sign: f64| -> Result<Vec<f64>, String> {
                let mut data = matrix.data().to_vec();
                data[column] += sign * step;
                let inputs = BTreeMap::from([("a".to_string(), tensor(&[3, 3], &data))]);
                Ok(graph.vjp(loss, &inputs, tensor(&[], &[1.0]))?["a"]
                    .data()
                    .to_vec())
            };
            let (plus, minus) = (shifted(1.0)?, shifted(-1.0)?);
            let expected = plus
                .iter()
                .zip(&minus)
                .map(|(p, m)| (p - m) / (2.0 * step))
                .collect::<Vec<_>>();
            let actual = hessian.iter().map(|row| row[column]).collect::<Vec<_>>();
            assert_close(&actual, &expected, 1e-6);
        }
    }
    Ok(())
}

#[test]
fn vmap_batches_linalg_and_solve_operands() -> Result<(), String> {
    let mut callee = TensorIr::new();
    let a = callee.input("a", vec![2, 2])?;
    let b = callee.input("b", vec![2, 1])?;
    let x = callee.solve(a, b)?;
    let log_abs = callee.linalg(a, LinalgKind::LogAbsDet)?;
    let mut graph = TensorIr::new();
    let matrices = graph.input("a", vec![3, 2, 2])?;
    let rhs = graph.input("b", vec![2, 1])?;
    let outputs = graph
        .inline_batched(
            &callee,
            &BTreeMap::from([
                ("a".to_string(), (matrices, true)),
                ("b".to_string(), (rhs, false)),
            ]),
            3,
            &[x, log_abs],
        )
        .map_err(|error| error.to_string())?;
    assert!(outputs.iter().all(|(_, mapped)| *mapped));
    let data = tensor(
        &[3, 2, 2],
        &[2.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0],
    );
    let inputs = [("a", &data), ("b", &tensor(&[2, 1], &[2.0, 4.0]))];
    let solved = evaluate(&graph, outputs[0].0, &inputs)?;
    assert_eq!(
        solved.data().as_ref(),
        &[1.0, 1.0, 2.0, 4.0, 4.0 / 3.0, 2.0]
    );
    let logs = evaluate(&graph, outputs[1].0, &inputs)?;
    assert_close(
        logs.data().as_ref(),
        &[8.0_f64.ln(), 0.0, 3.0_f64.ln()],
        1e-15,
    );
    Ok(())
}

#[test]
fn mlx_validation_rejects_linalg() -> Result<(), String> {
    for kind in [LinalgKind::LogAbsDet, LinalgKind::EighVectors] {
        let mut graph = TensorIr::new();
        let a = graph.input("a", vec![2, 2])?;
        let output = graph.linalg(a, kind)?;
        let plan = graph.compile_cpu(output)?;
        assert_eq!(
            plan.validate_mlx().map_err(|(op, _)| op),
            Err(kind.name().to_string())
        );
    }
    Ok(())
}
