use crate::autodiff::Dual;
use crate::ode::rk4_final_state;

#[derive(Clone, Copy, Debug)]
pub struct LotkaVolterraProblem {
    pub initial_state: [f64; 2],
    pub target_final_state: [f64; 2],
    pub t0: f64,
    pub dt: f64,
    pub steps: usize,
}

pub fn lotka_volterra_loss<T>(params: &[T; 4], problem: LotkaVolterraProblem) -> T
where
    T: Copy
        + From<f64>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Mul<f64, Output = T>,
{
    let initial_state = [
        T::from(problem.initial_state[0]),
        T::from(problem.initial_state[1]),
    ];
    let final_state = rk4_final_state(
        &initial_state,
        problem.t0,
        problem.dt,
        problem.steps,
        |_, state, derivative| {
            let prey = state[0];
            let predator = state[1];
            let alpha = params[0];
            let beta = params[1];
            let gamma = params[2];
            let delta = params[3];

            derivative[0] = alpha * prey - beta * prey * predator;
            derivative[1] = delta * prey * predator - gamma * predator;
        },
    );

    let prey_error = final_state[0] - T::from(problem.target_final_state[0]);
    let predator_error = final_state[1] - T::from(problem.target_final_state[1]);

    (prey_error * prey_error + predator_error * predator_error) * 0.5
}

pub fn finite_difference_parameter_gradient(
    params: [f64; 4],
    parameter_index: usize,
    problem: LotkaVolterraProblem,
    epsilon: f64,
) -> f64 {
    assert!(
        parameter_index < params.len(),
        "parameter_index out of range"
    );
    assert!(epsilon > 0.0, "epsilon must be positive");

    let mut plus = params;
    plus[parameter_index] += epsilon;
    let mut minus = params;
    minus[parameter_index] -= epsilon;

    let plus_loss = lotka_volterra_loss(&plus, problem);
    let minus_loss = lotka_volterra_loss(&minus, problem);

    (plus_loss - minus_loss) / (2.0 * epsilon)
}

pub fn lotka_volterra_loss_gradient_alpha(params: [f64; 4], problem: LotkaVolterraProblem) -> f64 {
    lotka_volterra_loss_parameter_gradient(params, 0, problem)
}

pub fn lotka_volterra_loss_gradient(params: [f64; 4], problem: LotkaVolterraProblem) -> [f64; 4] {
    [
        lotka_volterra_loss_parameter_gradient(params, 0, problem),
        lotka_volterra_loss_parameter_gradient(params, 1, problem),
        lotka_volterra_loss_parameter_gradient(params, 2, problem),
        lotka_volterra_loss_parameter_gradient(params, 3, problem),
    ]
}

fn lotka_volterra_loss_parameter_gradient(
    params: [f64; 4],
    parameter_index: usize,
    problem: LotkaVolterraProblem,
) -> f64 {
    let dual_params = std::array::from_fn(|index| {
        if index == parameter_index {
            Dual::variable(params[index])
        } else {
            Dual::constant(params[index])
        }
    });

    lotka_volterra_loss(&dual_params, problem).derivative()
}
