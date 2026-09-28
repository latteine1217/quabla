use quabla_core::optim::{gradient_descent, GradientDescent};

#[test]
fn gradient_descent_minimizes_quadratic_objective() {
    let optimizer = GradientDescent {
        learning_rate: 0.1,
        max_steps: 200,
        gradient_tolerance: 1e-10,
    };

    let result = gradient_descent([0.0, 0.0], optimizer, |params| {
        let dx = params[0] - 3.0;
        let dy = params[1] + 2.0;
        let loss = dx * dx + dy * dy;
        let gradient = [2.0 * dx, 2.0 * dy];

        (loss, gradient)
    });

    assert!(
        result.converged,
        "fit should converge on a convex quadratic"
    );
    assert!(result.loss < 1e-16, "loss={}", result.loss);
    assert_close(result.params[0], 3.0, 1e-8);
    assert_close(result.params[1], -2.0, 1e-8);
}

#[test]
fn gradient_descent_stops_when_initial_gradient_is_small() {
    let optimizer = GradientDescent {
        learning_rate: 0.1,
        max_steps: 200,
        gradient_tolerance: 1e-10,
    };

    let result = gradient_descent([3.0, -2.0], optimizer, |params| {
        let dx = params[0] - 3.0;
        let dy = params[1] + 2.0;
        let loss = dx * dx + dy * dy;
        let gradient = [2.0 * dx, 2.0 * dy];

        (loss, gradient)
    });

    assert!(result.converged);
    assert_eq!(result.steps, 0);
    assert_close(result.params[0], 3.0, 1e-12);
    assert_close(result.params[1], -2.0, 1e-12);
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}
