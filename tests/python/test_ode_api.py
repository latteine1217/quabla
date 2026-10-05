"""Fixed-step ODE integrators: convergence order, derivatives, and devices."""

import math
import os

import quabla as qb


def decay(y, t, k):
    return -k * y


def raises(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def test_methods_converge_at_their_order():
    y0, k, end = qb.array([1.0, 2.0]), qb.array(0.7), 2.0
    exact = math.exp(-0.7 * end)
    for method, order in (("euler", 1), ("heun", 2), ("rk4", 4)):
        errors = []
        for steps in (20, 40):
            y = qb.ode.odeint(decay, y0, (0.0, end), steps=steps, method=method, args=(k,))
            errors.append(abs(y.tolist()[0] - exact))
        observed = math.log2(errors[0] / errors[1])
        assert abs(observed - order) < 0.1, (method, observed)


def test_time_dependent_right_hand_side_uses_exact_time_points():
    # dy/dt = cos(t) integrates to sin(t); RK4 with 50 steps is accurate to ~1e-10.
    y = qb.ode.odeint(lambda y, t: qb.cos(t) + 0.0 * y, qb.array([0.0]), (0.0, 1.0), steps=50)
    assert abs(y.item() - math.sin(1.0)) < 1e-9
    # Reversed spans integrate backwards.
    back = qb.ode.odeint(lambda y, t: qb.cos(t) + 0.0 * y, qb.array([math.sin(1.0)]), (1.0, 0.0), steps=50)
    assert abs(back.item()) < 1e-9


def test_rk4_conserves_oscillator_energy_closely():
    def oscillator(state, t):
        position, velocity = state[0:1], state[1:2]
        return qb.concat([velocity, -position], 0)

    final = qb.ode.odeint(oscillator, qb.array([1.0, 0.0]), (0.0, 20.0), steps=2000)
    x, v = final.tolist()
    assert abs(x * x + v * v - 1.0) < 1e-9
    assert abs(x - math.cos(20.0)) < 1e-8


def test_solve_is_differentiable_and_jittable():
    y0, k, end = qb.array([1.0, 2.0]), qb.array(0.7), 2.0

    def loss(rate):
        return qb.ode.odeint(decay, y0, (0.0, end), steps=40, args=(rate,)).sum()

    analytic = -end * math.exp(-0.7 * end) * 3.0
    eager = qb.grad(loss)(k).item()
    staged = qb.jit(qb.grad(loss))(k).item()
    assert eager == staged
    assert abs(eager - analytic) < 1e-6 * abs(analytic)
    # A traced end time is a loop operand: d y(T) / dT = -k y(T).
    def at(end_time):
        return qb.ode.odeint(decay, y0, (0.0, end_time), steps=40, args=(k,)).sum()

    slope = qb.grad(at)(qb.array(end)).item()
    assert abs(slope - (-0.7 * 3.0 * math.exp(-0.7 * end))) < 1e-6


def test_saved_trajectory_starts_at_the_initial_state():
    y0, k = qb.array([1.0, 2.0]), qb.array(0.7)
    path = qb.ode.odeint(decay, y0, (0.0, 2.0), steps=4, args=(k,), save=True)
    assert path.shape == [5, 2]
    assert path.tolist()[0] == [1.0, 2.0]
    final = qb.ode.odeint(decay, y0, (0.0, 2.0), steps=4, args=(k,))
    assert path.tolist()[-1] == final.tolist()
    staged = qb.jit(lambda rate: qb.ode.odeint(decay, y0, (0.0, 2.0), steps=4, args=(rate,), save=True))
    assert staged(k).tolist() == path.tolist()


def test_invalid_arguments():
    y0 = qb.array([1.0])
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=0)
    raises(TypeError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=True)
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, method="rk45")
    raises(TypeError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, args=[1.0])
    raises(TypeError, qb.ode.odeint, decay, y0, 1.0, steps=4)
    raises(TypeError, qb.ode.odeint, None, y0, (0.0, 1.0), steps=4)


def test_optional_device_parity():
    y0, k = qb.array([1.0, 2.0], dtype=qb.float32), qb.array(0.7, dtype=qb.float32)
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue

        def solve(rate):
            return qb.ode.odeint(decay, y0, (0.0, 2.0), steps=40, args=(rate,))

        functions = [solve]
        # CUDA loop VJPs require operands of the carry's shape, and the
        # solver passes the scalar t0 and dt; reverse mode runs on MLX only.
        if device == "mlx":
            functions.append(qb.grad(lambda rate: solve(rate).sum()))
        for function in functions:
            expected = qb.jit(function)(k).tolist()
            actual = qb.jit(function, device=device)(k).tolist()
            expected = expected if isinstance(expected, list) else [expected]
            actual = actual if isinstance(actual, list) else [actual]
            for got, want in zip(actual, expected):
                assert abs(got - want) <= 1e-5 * max(1.0, abs(want)), (device, got, want)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
