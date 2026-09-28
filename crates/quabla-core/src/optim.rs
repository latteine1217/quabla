#[derive(Clone, Copy, Debug)]
pub struct GradientDescent {
    pub learning_rate: f64,
    pub max_steps: usize,
    pub gradient_tolerance: f64,
}

#[derive(Clone, Debug)]
pub struct FitResult<const N: usize> {
    pub params: [f64; N],
    pub loss: f64,
    pub steps: usize,
    pub converged: bool,
}

pub fn gradient_descent<const N: usize, F>(
    initial_params: [f64; N],
    optimizer: GradientDescent,
    mut objective: F,
) -> FitResult<N>
where
    F: FnMut([f64; N]) -> (f64, [f64; N]),
{
    assert!(
        optimizer.learning_rate > 0.0,
        "learning_rate must be positive"
    );
    assert!(
        optimizer.gradient_tolerance >= 0.0,
        "gradient_tolerance must be non-negative"
    );

    let mut params = initial_params;

    for step in 0..optimizer.max_steps {
        let (loss, gradient) = objective(params);
        let gradient_norm = l2_norm(&gradient);

        if gradient_norm <= optimizer.gradient_tolerance {
            return FitResult {
                params,
                loss,
                steps: step,
                converged: true,
            };
        }

        for i in 0..N {
            params[i] -= optimizer.learning_rate * gradient[i];
        }
    }

    let (loss, gradient) = objective(params);

    FitResult {
        params,
        loss,
        steps: optimizer.max_steps,
        converged: l2_norm(&gradient) <= optimizer.gradient_tolerance,
    }
}

fn l2_norm<const N: usize>(values: &[f64; N]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
