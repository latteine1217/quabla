pub trait StateScalar:
    Copy + From<f64> + std::ops::Add<Output = Self> + std::ops::Mul<f64, Output = Self>
{
}

impl<T> StateScalar for T where
    T: Copy + From<f64> + std::ops::Add<Output = Self> + std::ops::Mul<f64, Output = Self>
{
}

pub fn rk4_integrate<T, F>(
    initial_state: &[T],
    t0: f64,
    dt: f64,
    steps: usize,
    mut rhs: F,
) -> Vec<Vec<T>>
where
    T: StateScalar,
    F: FnMut(f64, &[T], &mut [T]),
{
    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut state = initial_state.to_vec();

    trajectory.push(state.clone());

    for step in 0..steps {
        let t = t0 + step as f64 * dt;
        state = rk4_step(&state, t, dt, &mut rhs);
        trajectory.push(state.clone());
    }

    trajectory
}

pub fn rk4_final_state<T, F>(
    initial_state: &[T],
    t0: f64,
    dt: f64,
    steps: usize,
    mut rhs: F,
) -> Vec<T>
where
    T: StateScalar,
    F: FnMut(f64, &[T], &mut [T]),
{
    let mut state = initial_state.to_vec();

    for step in 0..steps {
        let t = t0 + step as f64 * dt;
        state = rk4_step(&state, t, dt, &mut rhs);
    }

    state
}

fn rk4_step<T, F>(state: &[T], t: f64, dt: f64, rhs: &mut F) -> Vec<T>
where
    T: StateScalar,
    F: FnMut(f64, &[T], &mut [T]),
{
    let dimension = state.len();
    let k1 = derivative(dimension, t, state, rhs);
    let y2 = add_scaled(state, &k1, 0.5 * dt);
    let k2 = derivative(dimension, t + 0.5 * dt, &y2, rhs);
    let y3 = add_scaled(state, &k2, 0.5 * dt);
    let k3 = derivative(dimension, t + 0.5 * dt, &y3, rhs);
    let y4 = add_scaled(state, &k3, dt);
    let k4 = derivative(dimension, t + dt, &y4, rhs);

    (0..dimension)
        .map(|i| state[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0))
        .collect()
}

fn derivative<T, F>(dimension: usize, t: f64, state: &[T], rhs: &mut F) -> Vec<T>
where
    T: StateScalar,
    F: FnMut(f64, &[T], &mut [T]),
{
    let mut output = vec![T::from(0.0); dimension];
    rhs(t, state, &mut output);
    output
}

fn add_scaled<T>(state: &[T], derivative: &[T], scale: f64) -> Vec<T>
where
    T: StateScalar,
{
    state
        .iter()
        .zip(derivative)
        .map(|(value, derivative)| *value + *derivative * scale)
        .collect()
}
