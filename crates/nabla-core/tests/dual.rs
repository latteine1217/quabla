use nabla_core::Dual;

#[test]
fn dual_derivative_matches_quadratic_product_rule() {
    let x = Dual::variable(3.0);
    let y = x * x + Dual::constant(2.0) * x + Dual::constant(1.0);

    assert_close(y.value(), 16.0, 1e-12);
    assert_close(y.derivative(), 8.0, 1e-12);
}

#[test]
fn dual_supports_common_elementary_functions() {
    let x = Dual::variable(0.7);
    let y = x.sin() * x.exp();
    let expected_value = 0.7_f64.sin() * 0.7_f64.exp();
    let expected_derivative = 0.7_f64.exp() * (0.7_f64.sin() + 0.7_f64.cos());

    assert_close(y.value(), expected_value, 1e-12);
    assert_close(y.derivative(), expected_derivative, 1e-12);
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}
