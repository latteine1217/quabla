use quabla_core::models::{
    finite_difference_parameter_gradient, lotka_volterra_loss, lotka_volterra_loss_gradient,
    lotka_volterra_loss_gradient_alpha, LotkaVolterraProblem,
};
use quabla_core::ode::{rk4_final_state, rk4_integrate};
use quabla_core::optim::{gradient_descent, GradientDescent};
use quabla_core::Dual;

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

// Preserve the allocation-based implementation as an independent numerical oracle.
fn reference_trajectory<T, F>(
    initial: &[T],
    t0: f64,
    dt: f64,
    steps: usize,
    mut rhs: F,
) -> Vec<Vec<T>>
where
    T: quabla_core::ode::StateScalar,
    F: FnMut(f64, &[T], &mut [T]),
{
    let mut state = initial.to_vec();
    let mut result = vec![state.clone()];
    for step in 0..steps {
        let t = t0 + step as f64 * dt;
        let mut k1 = vec![T::from(0.0); state.len()];
        rhs(t, &state, &mut k1);
        let y2 = state
            .iter()
            .zip(&k1)
            .map(|(v, k)| *v + *k * (0.5 * dt))
            .collect::<Vec<_>>();
        let mut k2 = vec![T::from(0.0); state.len()];
        rhs(t + 0.5 * dt, &y2, &mut k2);
        let y3 = state
            .iter()
            .zip(&k2)
            .map(|(v, k)| *v + *k * (0.5 * dt))
            .collect::<Vec<_>>();
        let mut k3 = vec![T::from(0.0); state.len()];
        rhs(t + 0.5 * dt, &y3, &mut k3);
        let y4 = state
            .iter()
            .zip(&k3)
            .map(|(v, k)| *v + *k * dt)
            .collect::<Vec<_>>();
        let mut k4 = vec![T::from(0.0); state.len()];
        rhs(t + dt, &y4, &mut k4);
        state = (0..state.len())
            .map(|i| state[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0))
            .collect();
        result.push(state.clone());
    }
    result
}

#[test]
fn rk4_workspace_matches_reference_bits_and_mutable_partial_rhs() {
    type RhsCalls = Vec<(u64, Vec<u64>)>;
    fn run(reference: bool, final_only: bool) -> (Vec<Vec<f64>>, RhsCalls) {
        let mut calls = Vec::new();
        let rhs = |t: f64, state: &[f64], output: &mut [f64]| {
            assert!(output.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
            // Alternate partial writes to expose stale stage data after reuse.
            let index = calls.len() % state.len();
            output[index] = state[index] * 0.37 + t.sin();
            calls.push((t.to_bits(), state.iter().map(|v| v.to_bits()).collect()));
        };
        let initial = [1.25, -0.5, 0.0];
        let values = if reference {
            reference_trajectory(&initial, -0.25, 0.03, 17, rhs)
        } else if final_only {
            vec![rk4_final_state(&initial, -0.25, 0.03, 17, rhs)]
        } else {
            rk4_integrate(&initial, -0.25, 0.03, 17, rhs)
        };
        (values, calls)
    }
    let (expected, expected_calls) = run(true, false);
    let (actual, calls) = run(false, false);
    assert_eq!(calls.len(), 17 * 4);
    assert_eq!(calls, expected_calls);
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let (final_state, calls) = run(false, true);
    assert_eq!(calls, expected_calls);
    for (actual, expected) in final_state[0].iter().zip(expected.last().unwrap()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn rk4_workspace_matches_dual_reference_bits() {
    let initial = [Dual::variable(1.25), Dual::constant(-0.5)];
    let rhs = |t: f64, state: &[Dual], output: &mut [Dual]| {
        output[0] = state[0] * 0.7 + state[1] * t;
        output[1] = state[0] * -0.2 + state[1] * 0.3;
    };
    let expected = reference_trajectory(&initial, 0.1, 0.02, 31, rhs);
    let actual = rk4_integrate(&initial, 0.1, 0.02, 31, rhs);
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert_eq!(actual.value().to_bits(), expected.value().to_bits());
        assert_eq!(
            actual.derivative().to_bits(),
            expected.derivative().to_bits()
        );
    }
    assert_eq!(
        rk4_final_state(&initial, 0.1, 0.02, 31, rhs),
        *expected.last().unwrap()
    );
}

#[test]
fn rk4_workspace_preserves_zero_steps_and_empty_state() {
    let initial = [-0.0, 1.0];
    let trajectory = rk4_integrate(&initial, 0.0, 0.1, 0, |_, _, _| panic!("zero steps"));
    assert_eq!(trajectory.len(), 1);
    assert_eq!(trajectory[0][0].to_bits(), initial[0].to_bits());
    let mut calls = 0;
    assert!(
        rk4_final_state::<f64, _>(&[], 0.0, 0.1, 3, |_, state, output| {
            assert!(state.is_empty() && output.is_empty());
            calls += 1;
        })
        .is_empty()
    );
    assert_eq!(calls, 12);
}

#[test]
fn lotka_volterra_alpha_helper_matches_full_gradient_bits() {
    let params = [1.4, 0.9, 0.8, 1.1];
    for steps in [0, 1, 50] {
        let problem = LotkaVolterraProblem {
            initial_state: [1.2, 0.9],
            target_final_state: [1.8, 0.5],
            t0: 0.0,
            dt: 0.02,
            steps,
        };
        assert_eq!(
            lotka_volterra_loss_gradient_alpha(params, problem).to_bits(),
            lotka_volterra_loss_gradient(params, problem)[0].to_bits()
        );
    }
}
