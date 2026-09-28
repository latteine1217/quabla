use quabla_core::models::{
    lotka_volterra_loss, lotka_volterra_loss_gradient, LotkaVolterraProblem,
};
use quabla_core::optim::{gradient_descent, GradientDescent};

fn main() {
    let initial_params = [1.0, 0.7, 1.2, 0.8];
    let problem = LotkaVolterraProblem {
        initial_state: [1.2, 0.9],
        target_final_state: [1.8, 0.5],
        t0: 0.0,
        dt: 0.02,
        steps: 50,
    };
    let optimizer = GradientDescent {
        learning_rate: 0.01,
        max_steps: 40,
        gradient_tolerance: 1e-10,
    };
    let initial_loss = lotka_volterra_loss(&initial_params, problem);
    let fit = gradient_descent(initial_params, optimizer, |params| {
        (
            lotka_volterra_loss(&params, problem),
            lotka_volterra_loss_gradient(params, problem),
        )
    });

    println!("initial loss: {initial_loss:.12}");
    println!("final loss: {:.12}", fit.loss);
    println!("steps: {}", fit.steps);
    println!("converged: {}", fit.converged);
    println!(
        "params: [{:.12}, {:.12}, {:.12}, {:.12}]",
        fit.params[0], fit.params[1], fit.params[2], fit.params[3]
    );
}
