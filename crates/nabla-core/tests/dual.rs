use nabla_core::{forward_diff, forward_gradient, Dual};

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

#[test]
fn forward_diff_macro_rewrites_a_pure_scalar_closure_to_dual_arithmetic() {
    let bias = 0.25;
    let function = forward_diff!(|x| x.sin() * x.exp() + 2.0 * x + bias);
    let result = function(0.7);
    let expected_value = 0.7_f64.sin() * 0.7_f64.exp() + 2.0 * 0.7 + bias;
    let expected_derivative = 0.7_f64.exp() * (0.7_f64.sin() + 0.7_f64.cos()) + 2.0;

    assert_close(result.value(), expected_value, 1e-12);
    assert_close(result.derivative(), expected_derivative, 1e-12);
}

#[test]
fn forward_diff_macro_preserves_block_expressions_and_powi() {
    let function = forward_diff!(|x| {
        let shifted = x - 1.5;
        shifted.powi(3) / 3.0
    });
    let result = function(2.0);

    assert_close(result.value(), 1.0 / 24.0, 1e-12);
    assert_close(result.derivative(), 0.25, 1e-12);
}

#[test]
fn forward_gradient_macro_returns_ordered_multivariate_gradient() {
    let function = forward_gradient!(|x, y, scale| scale * (x * y + x.sin()) + y.powi(2));
    let result = function(0.7, -1.2, 0.5);
    let expected_value = 0.5 * (0.7 * -1.2 + 0.7_f64.sin()) + (-1.2_f64).powi(2);
    let expected_gradient = [
        0.5 * (-1.2 + 0.7_f64.cos()),
        0.5 * 0.7 + 2.0 * -1.2,
        0.7 * -1.2 + 0.7_f64.sin(),
    ];

    assert_close(result.value(), expected_value, 1e-12);
    assert_eq!(result.gradient().len(), expected_gradient.len());
    for (actual, expected) in result.gradient().iter().zip(expected_gradient) {
        assert_close(*actual, expected, 1e-12);
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}
