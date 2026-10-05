"""ODE integrators: convergence order, adaptive accuracy, derivatives, and devices."""

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

        def saved(rate):
            return qb.ode.odeint(decay, y0, (0.0, 2.0), steps=40, args=(rate,), save=True)

        # The scalar operands (rate, t0, dt) broadcast against the vector
        # state, so their loop VJP reduces over the state axis every step.
        functions = [
            solve,
            qb.grad(lambda rate: solve(rate).sum()),
            qb.grad(lambda rate: saved(rate).sum()),
            # Forward-over-reverse (HVP) of the scalar-operand loop.
            lambda rate: qb.jvp(
                qb.grad(lambda r: solve(r).sum()), (rate,), (qb.ones([], qb.float32),)
            )[1],
            lambda rate: qb.jvp(solve, (rate,), (qb.ones([], qb.float32),))[1],
        ]
        for function in functions:
            expected = qb.jit(function)(k).tolist()
            actual = qb.jit(function, device=device)(k).tolist()
            expected = expected if isinstance(expected, list) else [expected]
            actual = actual if isinstance(actual, list) else [actual]
            for got, want in zip(actual, expected):
                assert abs(got - want) <= 1e-5 * max(1.0, abs(want)), (device, got, want)


# ---- Adaptive Dormand-Prince 5(4) ----


def van_der_pol(state, t, mu):
    x, v = state[0:1], state[1:2]
    return qb.concat([v, mu * (1.0 - x * x) * v - x], 0)


def lotka_volterra(state, t, a, b, c, d):
    x, y = state[0:1], state[1:2]
    return qb.concat([a * x - b * x * y, d * x * y - c * y], 0)


def lotka_volterra_invariant(state, a, b, c, d):
    x, y = state
    return d * x - c * math.log(x) + b * y - a * math.log(y)


def test_dopri5_error_shrinks_with_the_tolerance():
    y0, k, end = qb.array([1.0, 2.0]), qb.array(0.7), 3.0
    exact = [math.exp(-0.7 * end), 2.0 * math.exp(-0.7 * end)]
    errors, steps = [], []
    for rtol in (1e-4, 1e-6, 1e-8):
        y, info = qb.ode.odeint(
            decay, y0, (0.0, end), method="dopri5", args=(k,), rtol=rtol, atol=rtol * 1e-3, info=True
        )
        errors.append(max(abs(a - b) for a, b in zip(y.tolist(), exact)))
        steps.append(int(info["accepted_steps"].item()))
        assert info["success"].item() and info["t"].item() == end
    # The global error tracks the tolerance: within two orders of magnitude
    # per 100x tighter rtol, and it always decreases.
    assert errors[0] > errors[1] > errors[2], errors
    for loose, tight in zip(errors, errors[1:]):
        assert 3.0 < loose / tight < 1e4, errors
    assert errors[1] < 1e-5 and errors[2] < 1e-7, errors
    # Fifth order: 100x tighter tolerance costs about 100^(1/5) = 2.5x the steps.
    assert steps[0] < steps[1] < steps[2] and steps[2] < 12 * steps[0], steps


def test_dopri5_time_dependent_and_backward_integration():
    rhs = lambda y, t: qb.cos(t) + 0.0 * y  # noqa: E731
    y = qb.ode.odeint(rhs, qb.array([0.0]), (0.0, 4.0), method="dopri5", rtol=1e-9, atol=1e-12)
    assert abs(y.item() - math.sin(4.0)) < 1e-9
    back = qb.ode.odeint(rhs, qb.array([math.sin(4.0)]), (4.0, 0.0), method="dopri5", rtol=1e-9, atol=1e-12)
    assert abs(back.item()) < 1e-9
    # An empty span returns the initial state.
    same = qb.ode.odeint(rhs, qb.array([0.5]), (1.0, 1.0), method="dopri5")
    assert same.item() == 0.5


def test_dopri5_van_der_pol_matches_a_fine_fixed_step_reference():
    y0, mu = qb.array([2.0, 0.0]), qb.array(1.0)
    reference = qb.ode.odeint(van_der_pol, y0, (0.0, 10.0), steps=20000, args=(mu,))
    y, info = qb.ode.odeint(
        van_der_pol, y0, (0.0, 10.0), method="dopri5", args=(mu,), rtol=1e-8, atol=1e-10, info=True
    )
    assert max(abs(a - b) for a, b in zip(y.tolist(), reference.tolist())) < 1e-6
    accepted = int(info["accepted_steps"].item())
    assert 20 < accepted < 1000, accepted
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        return
    # SciPy's RK45 is the same pair with an I-controller; the step counts
    # should agree to within a small factor.
    scipy = solve_ivp(
        lambda t, s: [s[1], (1.0 - s[0] ** 2) * s[1] - s[0]],
        (0.0, 10.0),
        [2.0, 0.0],
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    scipy_steps = len(scipy.t) - 1
    assert 0.5 < accepted / scipy_steps < 2.0, (accepted, scipy_steps)
    assert max(abs(a - b) for a, b in zip(y.tolist(), scipy.y[:, -1])) < 1e-5


def test_dopri5_conserves_the_lotka_volterra_invariant():
    params = tuple(qb.array(value) for value in (1.5, 1.0, 3.0, 1.0))
    floats = (1.5, 1.0, 3.0, 1.0)
    y0 = qb.array([10.0, 5.0])
    drift = []
    for rtol in (1e-6, 1e-9):
        y = qb.ode.odeint(
            lotka_volterra, y0, (0.0, 15.0), method="dopri5", args=params, rtol=rtol, atol=rtol * 1e-3, max_steps=4096
        )
        drift.append(
            abs(lotka_volterra_invariant(y.tolist(), *floats) - lotka_volterra_invariant(y0.tolist(), *floats))
        )
    assert drift[1] < drift[0] and drift[1] < 1e-6, drift


def test_dopri5_gradients_match_finite_differences_and_fixed_step_rk4():
    y0, end = qb.array([1.0, 0.5]), 3.0
    mu = qb.array(1.3)

    def loss(mu, y0, end, method="dopri5"):
        kwargs = {"steps": 4000} if method == "rk4" else {"rtol": 1e-9, "atol": 1e-12, "max_steps": 400}
        y = qb.ode.odeint(van_der_pol, y0, (0.0, end), method=method, args=(mu,), **kwargs)
        return (y * y).sum()

    gradient = qb.grad(loss, argnums=(0, 1, 2))(mu, y0, qb.array(end))
    staged = qb.jit(qb.grad(loss, argnums=(0, 1, 2)))(mu, y0, qb.array(end))
    reference = qb.grad(lambda m, y, e: loss(m, y, e, "rk4"), argnums=(0, 1, 2))(mu, y0, qb.array(end))
    flat = [gradient[0].item(), *gradient[1].tolist(), gradient[2].item()]
    flat_staged = [staged[0].item(), *staged[1].tolist(), staged[2].item()]
    flat_reference = [reference[0].item(), *reference[1].tolist(), reference[2].item()]
    for value, other in zip(flat, flat_staged):
        assert abs(value - other) <= 1e-12 * max(1.0, abs(value)), (flat, flat_staged)
    for value, other in zip(flat, flat_reference):
        assert abs(value - other) < 1e-6 * max(1.0, abs(other)), (flat, flat_reference)
    # Central differences on the eager solve.
    h = 1e-5
    primal = lambda m, y, e: loss(qb.array(m), qb.array(y), qb.array(e)).item()  # noqa: E731
    base = (1.3, [1.0, 0.5], end)
    fd = [
        (primal(1.3 + h, base[1], end) - primal(1.3 - h, base[1], end)) / (2 * h),
        (primal(1.3, [1.0 + h, 0.5], end) - primal(1.3, [1.0 - h, 0.5], end)) / (2 * h),
        (primal(1.3, [1.0, 0.5 + h], end) - primal(1.3, [1.0, 0.5 - h], end)) / (2 * h),
        (primal(1.3, base[1], end + h) - primal(1.3, base[1], end - h)) / (2 * h),
    ]
    for value, other in zip(flat, fd):
        assert abs(value - other) < 1e-5 * max(1.0, abs(other)), (flat, fd)


def test_dopri5_jit_matches_eager_and_reports_exhaustion():
    y0, k = qb.array([1.0, 2.0]), qb.array(0.7)

    def solve(k, end):
        return qb.ode.odeint(decay, y0, (0.0, end), method="dopri5", args=(k,), max_steps=64, info=True)

    eager_y, eager_info = solve(k, 2.0)
    staged_y, staged_info = qb.jit(solve)(k, qb.array(2.0))
    assert max(abs(a - b) for a, b in zip(eager_y.tolist(), staged_y.tolist())) < 1e-14
    assert staged_info["accepted_steps"].item() == eager_info["accepted_steps"].item()
    assert staged_info["success"].item()
    # Too few attempts: eager raises; staged reports the time reached.
    raises(
        RuntimeError,
        qb.ode.odeint,
        decay,
        y0,
        (0.0, 50.0),
        method="dopri5",
        args=(k,),
        rtol=1e-12,
        atol=1e-14,
        max_steps=5,
    )
    _, info = qb.jit(
        lambda k: qb.ode.odeint(
            decay, y0, (0.0, 50.0), method="dopri5", args=(k,), rtol=1e-12, atol=1e-14, max_steps=5, info=True
        )
    )(k)
    assert not info["success"].item() and 0.0 < info["t"].item() < 50.0


def test_dopri5_argument_validation():
    y0 = qb.array([1.0])
    raises(TypeError, qb.ode.odeint, decay, y0, (0.0, 1.0), method="dopri5", steps=10, args=(1.0,))
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), method="dopri5", save=True, args=(1.0,))
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), method="dopri5", rtol=0.0, atol=0.0, args=(1.0,))
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), method="dopri5", max_steps=0, args=(1.0,))
    raises(TypeError, qb.ode.odeint, decay, y0, (0.0, 1.0), args=(1.0,))
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, info=True, args=(1.0,))


def test_optional_device_dopri5_parity():
    y0 = qb.array([2.0, 0.0], dtype=qb.float32)

    def loss(mu):
        y = qb.ode.odeint(van_der_pol, y0, (0.0, 3.0), method="dopri5", args=(mu,), rtol=1e-4, atol=1e-6, max_steps=96)
        return (y * y).sum()

    mu = qb.array(1.0, dtype=qb.float32)
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue
        # The body slices and concatenates its packed carry and takes norms,
        # so CUDA runs it as a host-driven loop over the body's device program.
        for operation in (loss, qb.value_and_grad(loss)):
            cpu = qb.tree.flatten(qb.jit(operation)(mu))[0]
            actual = qb.tree.flatten(qb.jit(operation, device=device)(mu))[0]
            for value, expected in zip(actual, cpu):
                assert abs(value.item() - expected.item()) < 1e-3 * max(1.0, abs(expected.item())), (
                    device,
                    value.item(),
                    expected.item(),
                )


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
