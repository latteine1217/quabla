use nabla_core::models::{lotka_volterra_loss, lotka_volterra_loss_gradient, LotkaVolterraProblem};

fn main() {
    let params = [1.4, 0.9, 0.8, 1.1];
    let problem = LotkaVolterraProblem {
        initial_state: [1.2, 0.9],
        target_final_state: [1.8, 0.5],
        t0: 0.0,
        dt: 0.02,
        steps: 50,
    };

    let loss = lotka_volterra_loss(&params, problem);
    let gradient = lotka_volterra_loss_gradient(params, problem);

    println!("loss: {loss:.12}");
    println!(
        "gradient: [{:.12}, {:.12}, {:.12}, {:.12}]",
        gradient[0], gradient[1], gradient[2], gradient[3]
    );
}
