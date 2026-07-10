use nabla_core::models::{
    finite_difference_parameter_gradient, lotka_volterra_loss, lotka_volterra_loss_gradient,
    LotkaVolterraProblem,
};
use nabla_core::ode::{rk4_final_state, rk4_integrate};
use nabla_core::optim::{gradient_descent, GradientDescent};
use nabla_core::Dual;

#[test]
fn rk4_integrates_exponential_growth_with_small_error() {
    let y0 = [1.0_f64];
    let trajectory = rk4_integrate(&y0, 0.0, 0.01, 100, |_, y, dy| {
        dy[0] = y[0];
    });

    let final_value = trajectory[100][0];
    assert_close(final_value, std::f64::consts::E, 1e-7);
}

#[test]
fn rk4_final_state_matches_integrated_trajectory_endpoint() {
    let y0 = [1.0_f64, 0.5];
    let final_state = rk4_final_state(&y0, 0.0, 0.01, 100, |_, y, dy| {
        dy[0] = y[0];
        dy[1] = -2.0 * y[1];
    });
    let trajectory = rk4_integrate(&y0, 0.0, 0.01, 100, |_, y, dy| {
        dy[0] = y[0];
        dy[1] = -2.0 * y[1];
    });
    let trajectory_endpoint = &trajectory[100];

    assert_close(final_state[0], trajectory_endpoint[0], 1e-12);
    assert_close(final_state[1], trajectory_endpoint[1], 1e-12);
}

#[test]
fn rk4_final_state_returns_initial_state_for_zero_steps() {
    let y0 = [1.0_f64, 0.5];
    let mut rhs_called = false;
    let final_state = rk4_final_state(&y0, 0.0, 0.01, 0, |_, _y, _dy| {
        rhs_called = true;
    });

    assert!(!rhs_called, "rhs should not be called when steps is zero");
    assert_close(final_state[0], y0[0], 1e-12);
    assert_close(final_state[1], y0[1], 1e-12);
}

#[test]
fn lotka_volterra_gradient_matches_finite_difference() {
    let params = [1.4, 0.9, 0.8, 1.1];
    let problem = LotkaVolterraProblem {
        initial_state: [1.2, 0.9],
        target_final_state: [1.8, 0.5],
        t0: 0.0,
        dt: 0.02,
        steps: 50,
    };

    let autodiff_grad = lotka_volterra_loss(
        &[
            Dual::variable(params[0]),
            Dual::constant(params[1]),
            Dual::constant(params[2]),
            Dual::constant(params[3]),
        ],
        problem,
    )
    .derivative();

    let finite_difference_grad = finite_difference_parameter_gradient(params, 0, problem, 1e-5);

    assert_close(autodiff_grad, finite_difference_grad, 1e-6);
}

#[test]
fn lotka_volterra_full_gradient_matches_finite_difference() {
    let params = [1.4, 0.9, 0.8, 1.1];
    let problem = LotkaVolterraProblem {
        initial_state: [1.2, 0.9],
        target_final_state: [1.8, 0.5],
        t0: 0.0,
        dt: 0.02,
        steps: 50,
    };

    let autodiff_gradient = lotka_volterra_loss_gradient(params, problem);

    for (parameter_index, actual) in autodiff_gradient.iter().enumerate() {
        let expected = finite_difference_parameter_gradient(params, parameter_index, problem, 1e-5);

        assert_close(*actual, expected, 1e-6);
    }
}

#[test]
fn lotka_volterra_parameter_fit_reduces_loss() {
    let initial_params = [1.0, 0.7, 1.2, 0.8];
    let problem = LotkaVolterraProblem {
        initial_state: [1.2, 0.9],
        target_final_state: [1.8, 0.5],
        t0: 0.0,
        dt: 0.02,
        steps: 50,
    };
    let initial_loss = lotka_volterra_loss(&initial_params, problem);
    let optimizer = GradientDescent {
        learning_rate: 0.01,
        max_steps: 40,
        gradient_tolerance: 1e-10,
    };

    let result = gradient_descent(initial_params, optimizer, |params| {
        (
            lotka_volterra_loss(&params, problem),
            lotka_volterra_loss_gradient(params, problem),
        )
    });

    assert!(
        result.loss < initial_loss,
        "initial_loss={}, final_loss={}",
        initial_loss,
        result.loss
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}
