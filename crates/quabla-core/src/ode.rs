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

    if steps == 0 {
        return trajectory;
    }
    let mut workspace = Rk4Workspace::new(&state);

    for step in 0..steps {
        let t = t0 + step as f64 * dt;
        workspace.step(&mut state, t, dt, &mut rhs);
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
    if steps == 0 {
        return state;
    }
    let mut workspace = Rk4Workspace::new(&state);

    for step in 0..steps {
        let t = t0 + step as f64 * dt;
        workspace.step(&mut state, t, dt, &mut rhs);
    }

    state
}

// Reuse the four derivative buffers and one intermediate state for every step.
// Each RHS call still receives a freshly zeroed derivative buffer, including
// callbacks that only write some components.
struct Rk4Workspace<T> {
    k1: Vec<T>,
    k2: Vec<T>,
    k3: Vec<T>,
    k4: Vec<T>,
    intermediate: Vec<T>,
}

impl<T: StateScalar> Rk4Workspace<T> {
    fn new(state: &[T]) -> Self {
        Self {
            k1: state.to_vec(),
            k2: state.to_vec(),
            k3: state.to_vec(),
            k4: state.to_vec(),
            intermediate: state.to_vec(),
        }
    }

    fn step<F>(&mut self, state: &mut [T], t: f64, dt: f64, rhs: &mut F)
    where
        F: FnMut(f64, &[T], &mut [T]),
    {
        self.k1.fill(T::from(0.0));
        rhs(t, state, &mut self.k1);
        for (i, value) in state.iter().enumerate() {
            self.intermediate[i] = *value + self.k1[i] * (0.5 * dt);
        }
        self.k2.fill(T::from(0.0));
        rhs(t + 0.5 * dt, &self.intermediate, &mut self.k2);
        for (i, value) in state.iter().enumerate() {
            self.intermediate[i] = *value + self.k2[i] * (0.5 * dt);
        }
        self.k3.fill(T::from(0.0));
        rhs(t + 0.5 * dt, &self.intermediate, &mut self.k3);
        for (i, value) in state.iter().enumerate() {
            self.intermediate[i] = *value + self.k3[i] * dt;
        }
        self.k4.fill(T::from(0.0));
        rhs(t + dt, &self.intermediate, &mut self.k4);
        for (i, value) in state.iter_mut().enumerate() {
            *value = *value
                + (self.k1[i] + self.k2[i] * 2.0 + self.k3[i] * 2.0 + self.k4[i]) * (dt / 6.0);
        }
    }
}
