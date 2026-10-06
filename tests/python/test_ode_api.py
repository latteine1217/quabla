"""ODE integrators: convergence order, adaptive accuracy, derivatives, and devices."""

import math

import quabla as qb

from _support import devices, raises, run


def decay(y, t, k):
    return -k * y


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
    for device in devices():

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
    for device in devices():
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


# ---- Rosenbrock 2(3) for stiff problems ----


def robertson(y, t):
    a, b, c = y[0:1], y[1:2], y[2:3]
    return qb.concat([-0.04 * a + 1e4 * b * c, 0.04 * a - 1e4 * b * c - 3e7 * b * b, 3e7 * b * b], 0)


def prothero_robinson(y, t, lam):
    # y' = lam (y - cos t) - sin t has the solution cos t + (y0 - 1) e^(lam t);
    # it depends on t explicitly, so the method's df/dt term matters.
    return lam * (y - qb.cos(t)) - qb.sin(t)


# References from SciPy 1.18.1 solve_ivp(method="Radau", rtol=1e-12,
# atol=1e-14 for Van der Pol, 1e-18 for Robertson) with the analytic
# Jacobian. Radau at rtol=1e-10 agrees to 1e-13 and BDF at rtol=1e-12 to
# 1e-11 (Robertson; BDF at rtol=1e-10 to 1e-8 for Van der Pol).
VAN_DER_POL_MU_1000_AT_1000 = [-1.8636462548081643, 0.0007535430865435212]
ROBERTSON_REFERENCE = {
    40.0: [0.7158270687194048, 9.185534764557822e-06, 0.2841637457458297],
    1e5: [0.01786592114210887, 7.274751468440222e-08, 0.9821340061103758],
}


def test_rosenbrock23_stiff_van_der_pol_matches_the_reference():
    # mu = 1000 over [0, 1000] crosses one relaxation jump near t = 807.
    errors = []
    for rtol in (1e-5, 1e-7):
        y, info = qb.ode.odeint(
            van_der_pol,
            qb.array([2.0, 0.0]),
            (0.0, 1000.0),
            method="rosenbrock23",
            args=(qb.array(1000.0),),
            rtol=rtol,
            atol=rtol * 1e-3,
            max_steps=20000,
            info=True,
        )
        assert info["success"].item() and info["t"].item() == 1000.0
        errors.append(max(abs(a - b) for a, b in zip(y.tolist(), VAN_DER_POL_MU_1000_AT_1000)))
    # Second order with a third-order error estimate: the error falls by
    # about 100^(2/3) = 22 per 100x tighter tolerance.
    assert errors[0] < 1e-4 and errors[1] < 5e-6, errors
    assert 5.0 < errors[0] / errors[1] < 100.0, errors


def test_rosenbrock23_robertson_kinetics_and_conservation():
    for end, rtol, bound in ((40.0, 1e-6, 1e-8), (1e5, 1e-6, 1e-6)):
        y = qb.ode.odeint(
            robertson,
            qb.array([1.0, 0.0, 0.0]),
            (0.0, end),
            method="rosenbrock23",
            rtol=rtol,
            atol=rtol * 1e-6,
            max_steps=10000,
        )
        values, reference = y.tolist(), ROBERTSON_REFERENCE[end]
        assert abs(values[0] - reference[0]) < bound and abs(values[2] - reference[2]) < bound, (end, values)
        # The intermediate species is O(1e-5) to O(1e-7): compare relatively.
        assert abs(values[1] / reference[1] - 1.0) < 1e-3, (end, values)
        # A Rosenbrock step preserves linear invariants (e^T W = e^T when
        # e^T f = 0), so the total mass stays one to rounding.
        assert abs(sum(values) - 1.0) < 1e-13, values


def test_rosenbrock23_needs_far_fewer_steps_than_dopri5_on_stiff_problems():
    y0, mu = qb.array([2.0, 0.0]), qb.array(1000.0)
    counts = {}
    for method in ("rosenbrock23", "dopri5"):
        y, info = qb.ode.odeint(
            van_der_pol, y0, (0.0, 3.0), method=method, args=(mu,), rtol=1e-4, atol=1e-7, max_steps=20000, info=True
        )
        counts[method] = int(info["accepted_steps"].item() + info["rejected_steps"].item())
        counts[method + " y"] = y.tolist()
    # dopri5 is held to |h lambda| < 3.3 by stability (lambda ~ -mu x^2 = -3000),
    # about a thousand steps per time unit; the L-stable method follows the
    # slow solution in a few dozen steps.
    assert counts["dopri5"] > 2000 and counts["rosenbrock23"] < 100, counts
    for a, b in zip(counts["rosenbrock23 y"], counts["dopri5 y"]):
        assert abs(a - b) < 1e-4, counts


def test_rosenbrock23_linear_and_time_dependent_closed_forms():
    # Eigenvalues -1 and -1000 with eigenvectors [1, 1] and [1, -1]: from
    # [2, 0] the solution is e^-t [1, 1] + e^-1000t [1, -1].
    a = qb.array([[-500.5, 499.5], [499.5, -500.5]])
    y = qb.ode.odeint(
        lambda y, t, a: qb.matmul(a, y),
        qb.array([2.0, 0.0]),
        (0.0, 1.0),
        method="rosenbrock23",
        args=(a,),
        rtol=1e-8,
        atol=1e-10,
        max_steps=5000,
    )
    for value in y.tolist():
        assert abs(value - math.exp(-1.0)) < 1e-6, y.tolist()
    errors = []
    for rtol in (1e-4, 1e-6):
        y = qb.ode.odeint(
            prothero_robinson,
            qb.array([2.0]),
            (0.0, 2.0),
            method="rosenbrock23",
            args=(qb.array(-1e4),),
            rtol=rtol,
            atol=rtol,
            max_steps=5000,
        )
        errors.append(abs(y.item() - math.cos(2.0)))
    assert errors[0] < 1e-4 and errors[1] < 1e-5 and errors[1] < errors[0], errors
    # Backward integration of a non-stiff decay grows the state back.
    back = qb.ode.odeint(
        decay, qb.array([math.exp(-1.4)]), (2.0, 0.0), method="rosenbrock23", args=(0.7,), rtol=1e-8, atol=1e-12,
        max_steps=4000,
    )
    assert abs(back.item() - 1.0) < 1e-5, back.item()


def test_rosenbrock23_float32_state():
    y, info = qb.ode.odeint(
        prothero_robinson,
        qb.array([2.0], dtype=qb.float32),
        (0.0, 2.0),
        method="rosenbrock23",
        args=(qb.array(-1000.0, dtype=qb.float32),),
        rtol=1e-4,
        atol=1e-6,
        info=True,
    )
    assert y.dtype == qb.float32 and info["t"].dtype == qb.float32
    assert abs(y.item() - math.cos(2.0)) < 2e-5, y.item()
    assert int(info["accepted_steps"].item()) < 400


def test_rosenbrock23_gradients_match_closed_forms_and_finite_differences():
    # y(T) = cos T + (y0 - 1) e^(lam T): d/dy0 = e^(lam T), d/dlam = (y0 - 1) T e^(lam T).
    end, lam = 0.5, -5.0

    def loss(y0, lam):
        y = qb.ode.odeint(
            prothero_robinson, y0, (0.0, end), method="rosenbrock23", args=(lam,), rtol=1e-9, atol=1e-12, max_steps=3000
        )
        return y.sum()

    gy, gl = qb.grad(loss, argnums=(0, 1))(qb.array([2.0]), qb.array(lam))
    decay_factor = math.exp(lam * end)
    assert abs(gy.item() - decay_factor) < 1e-5 * decay_factor, gy.item()
    assert abs(gl.item() - end * decay_factor) < 1e-5 * end * decay_factor, gl.item()

    # Van der Pol with mu = 1000: J changes along the solution, so the
    # reverse pass differentiates through the Jacobian and the solves.
    def solve(mu, y0):
        return qb.ode.odeint(
            van_der_pol, y0, (0.0, 1.0), method="rosenbrock23", args=(mu,), rtol=1e-6, atol=1e-9, max_steps=300
        )

    def vdp_loss(mu, y0):
        return (solve(mu, y0) ** 2).sum()

    mu, y0 = qb.array(1000.0), qb.array([2.0, 0.0])
    eager = qb.grad(vdp_loss, argnums=(0, 1))(mu, y0)
    staged = qb.jit(qb.grad(vdp_loss, argnums=(0, 1)))(mu, y0)
    flat = [eager[0].item(), *eager[1].tolist()]
    assert flat == [staged[0].item(), *staged[1].tolist()]
    h = 1e-4
    primal = lambda m, y: vdp_loss(qb.array(m), qb.array(y)).item()  # noqa: E731
    fd = [
        (primal(1000.0 + h, [2.0, 0.0]) - primal(1000.0 - h, [2.0, 0.0])) / (2 * h),
        (primal(1000.0, [2.0 + h, 0.0]) - primal(1000.0, [2.0 - h, 0.0])) / (2 * h),
        (primal(1000.0, [2.0, h]) - primal(1000.0, [2.0, -h])) / (2 * h),
    ]
    for value, other in zip(flat, fd):
        assert abs(value - other) < 1e-5 * max(abs(other), 1e-3), (flat, fd)


def test_rosenbrock23_jit_matches_eager_and_validates_arguments():
    y0, mu = qb.array([2.0, 0.0]), qb.array(1000.0)

    def solve(mu):
        return qb.ode.odeint(
            van_der_pol, y0, (0.0, 1.0), method="rosenbrock23", args=(mu,), max_steps=200, info=True
        )

    eager_y, eager_info = solve(mu)
    staged_y, staged_info = qb.jit(solve)(mu)
    assert eager_y.tolist() == staged_y.tolist()
    assert eager_info["accepted_steps"].item() == staged_info["accepted_steps"].item()
    raises(RuntimeError, qb.ode.odeint, van_der_pol, y0, (0.0, 10.0), method="rosenbrock23", args=(mu,), max_steps=5)
    raises(TypeError, qb.ode.odeint, decay, qb.array([1.0]), (0.0, 1.0), method="rosenbrock23", steps=10, args=(1.0,))
    raises(ValueError, qb.ode.odeint, decay, qb.array([1.0]), (0.0, 1.0), method="rosenbrock23", save=True, args=(1.0,))
    raises(
        ValueError, qb.ode.odeint, decay, qb.array([1.0]), (0.0, 1.0), method="rosenbrock23", rtol=-1.0, args=(1.0,)
    )


# ---- saveat ----


def oscillator(y, t, w):
    # From [1, 0] the solution is [cos(w t), -w sin(w t)].
    return qb.concat([y[1:2], -(w * w) * y[0:1]], 0)


SAVE_TIMES = [0.0, 0.1234, 0.5, 0.77, 1.0, 1.61, 2.0]


def saveat_error(states, w=1.3):
    exact = [[math.cos(w * s), -w * math.sin(w * s)] for s in SAVE_TIMES]
    return max(abs(a - b) for row, ref in zip(states.tolist(), exact) for a, b in zip(row, ref))


def test_saveat_fixed_step_hermite_keeps_the_method_order():
    y0, w = qb.array([1.0, 0.0]), qb.array(1.3)
    for method, order in (("euler", 1), ("heun", 2), ("rk4", 4)):
        errors = []
        for steps in (40, 80):
            states = qb.ode.odeint(oscillator, y0, (0.0, 2.0), steps=steps, method=method, args=(w,), saveat=SAVE_TIMES)
            assert states.shape == [len(SAVE_TIMES), 2]
            assert states.tolist()[0] == [1.0, 0.0]
            final = qb.ode.odeint(oscillator, y0, (0.0, 2.0), steps=steps, method=method, args=(w,))
            for a, b in zip(states.tolist()[-1], final.tolist()):
                assert abs(a - b) < 1e-15, (method, a, b)
            errors.append(saveat_error(states))
        observed = math.log2(errors[0] / errors[1])
        assert abs(observed - order) < 0.15, (method, observed)


def test_saveat_adaptive_dense_output_tracks_the_tolerance():
    y0, w = qb.array([1.0, 0.0]), qb.array(1.3)
    for method, rtols, lowest in (("dopri5", (1e-6, 1e-8, 1e-10), 1e-9), ("rosenbrock23", (1e-6, 1e-8, 1e-10), 1e-6)):
        errors = []
        for rtol in rtols:
            states, info = qb.ode.odeint(
                oscillator, y0, (0.0, 2.0), method=method, args=(w,), saveat=SAVE_TIMES, rtol=rtol, atol=rtol,
                max_steps=5000, info=True,
            )
            final = qb.ode.odeint(oscillator, y0, (0.0, 2.0), method=method, args=(w,), rtol=rtol, atol=rtol, max_steps=5000)
            # The steps do not depend on the save times.
            plain = qb.ode.odeint(
                oscillator, y0, (0.0, 2.0), method=method, args=(w,), rtol=rtol, atol=rtol, max_steps=5000, info=True
            )[1]
            assert info["accepted_steps"].item() == plain["accepted_steps"].item()
            for a, b in zip(states.tolist()[-1], final.tolist()):
                assert abs(a - b) < 1e-15, (method, a, b)
            errors.append(saveat_error(states))
        # dopri5's quartic interpolant and rosenbrock23's quadratic one are
        # as accurate as the steps: 100x tighter tolerances cut the error by
        # about 100^(4/5) = 40 and 100^(2/3) = 22.
        expected = 40.0 if method == "dopri5" else 22.0
        for loose, tight in zip(errors, errors[1:]):
            assert expected / 3.0 < loose / tight < expected * 3.0, (method, errors)
        assert errors[-1] < lowest, (method, errors)


def test_saveat_jit_backward_and_shapes():
    y0, w = qb.array([1.0, 0.0]), qb.array(1.3)
    for method, kwargs in (("rk4", {"steps": 50}), ("dopri5", {"rtol": 1e-8}), ("rosenbrock23", {"rtol": 1e-6})):

        def solve(w, times, method=method, kwargs=kwargs):
            return qb.ode.odeint(oscillator, y0, (0.0, 2.0), method=method, args=(w,), saveat=times, **kwargs)

        eager = solve(w, qb.array(SAVE_TIMES)).tolist()
        assert eager == qb.jit(solve)(w, qb.array(SAVE_TIMES)).tolist(), method
    # Backward integration takes the save times from t0 down to t1.
    end = [math.cos(2.6), -1.3 * math.sin(2.6)]
    back = qb.ode.odeint(
        oscillator, qb.array(end), (2.0, 0.0), method="dopri5", args=(w,), saveat=SAVE_TIMES[::-1], rtol=1e-10, atol=1e-12
    )
    exact = [[math.cos(1.3 * s), -1.3 * math.sin(1.3 * s)] for s in SAVE_TIMES[::-1]]
    assert max(abs(a - b) for row, ref in zip(back.tolist(), exact) for a, b in zip(row, ref)) < 1e-9
    # A matrix state stacks to [len(saveat), *shape].
    y0 = qb.array([[1.0, 2.0], [3.0, 4.0]])
    for method, kwargs in (("heun", {"steps": 100}), ("dopri5", {}), ("rosenbrock23", {})):
        states = qb.ode.odeint(decay, y0, (0.0, 1.0), method=method, args=(0.5,), saveat=[0.25, 1.0], **kwargs)
        assert states.shape == [2, 2, 2]
        assert abs(states.tolist()[0][1][1] - 4.0 * math.exp(-0.125)) < 1e-4, method
    # float32 states keep their dtype.
    rate = qb.array(0.7, dtype=qb.float32)
    for method, kwargs in (("rk4", {"steps": 20}), ("dopri5", {"rtol": 1e-5, "atol": 1e-7}), ("rosenbrock23", {"rtol": 1e-4})):
        states = qb.ode.odeint(
            decay, qb.array([1.0], dtype=qb.float32), (0.0, 1.0), method=method, args=(rate,), saveat=[0.25, 0.5, 1.0],
            **kwargs,
        )
        assert states.dtype == qb.float32
        for row, time in zip(states.tolist(), (0.25, 0.5, 1.0)):
            assert abs(row[0] - math.exp(-0.7 * time)) < 1e-4, (method, row, time)


def test_saveat_gradients_match_finite_differences():
    y0, w = qb.array([1.0, 0.0]), qb.array(1.3)
    for method, kwargs in (
        ("heun", {"steps": 2000}),
        ("rk4", {"steps": 200}),
        ("dopri5", {"rtol": 1e-10, "atol": 1e-12}),
        ("rosenbrock23", {"rtol": 1e-8, "atol": 1e-10, "max_steps": 2000}),
    ):

        def loss(w, y0, method=method, kwargs=kwargs):
            states = qb.ode.odeint(oscillator, y0, (0.0, 2.0), method=method, args=(w,), saveat=SAVE_TIMES, **kwargs)
            return (states * states).sum()

        eager = qb.grad(loss, argnums=(0, 1))(w, y0)
        staged = qb.jit(qb.grad(loss, argnums=(0, 1)))(w, y0)
        flat = [eager[0].item(), *eager[1].tolist()]
        assert flat == [staged[0].item(), *staged[1].tolist()], method
        h = 1e-5
        fd = [
            (loss(qb.array(1.3 + h), y0).item() - loss(qb.array(1.3 - h), y0).item()) / (2 * h),
            (loss(w, qb.array([1.0 + h, 0.0])).item() - loss(w, qb.array([1.0 - h, 0.0])).item()) / (2 * h),
            (loss(w, qb.array([1.0, h])).item() - loss(w, qb.array([1.0, -h])).item()) / (2 * h),
        ]
        # The adaptive meshes move with the parameters, which the discrete
        # gradient ignores; the gap is of the size of the solution error.
        bound = 1e-5 if method == "rosenbrock23" else 1e-7
        for value, other in zip(flat, fd):
            assert abs(value - other) < bound * max(1.0, abs(other)), (method, flat, fd)
    # d y(s) / ds = f(y(s)): the derivative with respect to a save time.
    grad = qb.grad(
        lambda times: qb.ode.odeint(
            decay, qb.array([1.0]), (0.0, 1.0), method="dopri5", args=(0.7,), saveat=times, rtol=1e-10, atol=1e-12
        ).sum()
    )(qb.array([0.3, 0.6]))
    for value, time in zip(grad.tolist(), (0.3, 0.6)):
        assert abs(value + 0.7 * math.exp(-0.7 * time)) < 1e-8, grad.tolist()


def test_saveat_invalid_arguments_and_unreached_times():
    y0 = qb.array([1.0])
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, args=(1.0,), save=True, saveat=[0.5])
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, args=(1.0,), saveat=[[0.5]])
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, args=(1.0,), saveat=0.5)
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), steps=4, args=(1.0,), saveat=[0.5, 0.25])
    raises(ValueError, qb.ode.odeint, decay, y0, (0.0, 1.0), method="dopri5", args=(1.0,), saveat=[0.5, 1.5])
    raises(ValueError, qb.ode.odeint, decay, y0, (1.0, 0.0), method="rosenbrock23", args=(1.0,), saveat=[0.25, 0.5])
    # Under jit the times cannot be checked: a time outside the span, or one
    # the bounded loop did not reach, reads as NaN.
    staged = qb.jit(
        lambda times: qb.ode.odeint(decay, y0, (0.0, 1.0), method="dopri5", args=(0.7,), saveat=times)
    )(qb.array([-1.0, 0.0, 0.5, 2.0])).tolist()
    assert math.isnan(staged[0][0]) and staged[1] == [1.0] and math.isnan(staged[3][0]), staged
    assert abs(staged[2][0] - math.exp(-0.35)) < 1e-6, staged
    states, info = qb.jit(
        lambda k: qb.ode.odeint(
            decay, y0, (0.0, 50.0), method="rosenbrock23", args=(k,), saveat=[0.0, 49.0, 50.0], rtol=1e-12,
            atol=1e-14, max_steps=5, info=True,
        )
    )(qb.array(0.7))
    assert not info["success"].item()
    assert states.tolist()[0] == [1.0] and all(math.isnan(row[0]) for row in states.tolist()[1:])


def test_optional_device_rosenbrock23_and_saveat_parity():
    y0 = qb.array([2.0, 0.0], dtype=qb.float32)
    mu = qb.array(5.0, dtype=qb.float32)

    def stiff(mu):
        y = qb.ode.odeint(van_der_pol, y0, (0.0, 2.0), method="rosenbrock23", args=(mu,), rtol=1e-4, atol=1e-6, max_steps=64)
        return (y * y).sum()

    def saved(method, kwargs):
        def loss(mu):
            states = qb.ode.odeint(van_der_pol, y0, (0.0, 2.0), method=method, args=(mu,), saveat=[0.5, 1.25, 2.0], **kwargs)
            return (states * states).sum()

        return loss

    explicit = [saved("rk4", {"steps": 40}), saved("dopri5", {"rtol": 1e-4, "atol": 1e-6, "max_steps": 96})]
    for device in devices():
        losses = explicit + [stiff, saved("rosenbrock23", {"rtol": 1e-4, "atol": 1e-6, "max_steps": 64})]
        for loss in losses:
            for operation in (loss, qb.value_and_grad(loss)):
                cpu = qb.tree.flatten(qb.jit(operation)(mu))[0]
                actual = qb.tree.flatten(qb.jit(operation, device=device)(mu))[0]
                for value, expected in zip(actual, cpu):
                    assert abs(value.item() - expected.item()) < 1e-3 * max(1.0, abs(expected.item())), (
                        device,
                        value.item(),
                        expected.item(),
                    )


# ---- vmap of odeint ----


def forced_oscillator(y, t, k):
    return qb.stack([y[1], -k[0] * y[0] - k[1] * y[1] + 0.1 * qb.sin(t)], 0)


def batched_problem(dtype):
    y0 = qb.array([[1.0, 0.0], [0.5, -0.3], [-0.2, 0.8]], dtype=dtype)
    k = qb.array([[2.0, 0.1], [3.0, 0.3], [1.5, 0.05]], dtype=dtype)
    return y0, k


VMAP_SOLVES = [
    {"method": "euler", "steps": 30},
    {"method": "heun", "steps": 30},
    {"method": "rk4", "steps": 30},
    {"method": "rk4", "steps": 30, "save": True},
    {"method": "rk4", "steps": 30, "saveat": [0.0, 0.4, 1.1, 1.5]},
    {"method": "dopri5", "rtol": 1e-7, "atol": 1e-9},
    {"method": "dopri5", "saveat": [0.0, 0.4, 1.1, 1.5]},
    {"method": "rosenbrock23", "rtol": 1e-5, "atol": 1e-8},
    {"method": "rosenbrock23", "saveat": [0.4, 1.1, 1.5]},
]


def solver(options):
    def solve(y0, k):
        return qb.ode.odeint(forced_oscillator, y0, (0.0, 1.5), args=(k,), **options)

    return solve


def stacked_examples(function, y0, k, in_axes):
    """A Python loop over the batch of unbatched solves."""
    results = [
        function(y0 if in_axes[0] is None else y0[index], k if in_axes[1] is None else k[index])
        for index in range(3)
    ]
    return qb.tree.map(lambda *leaves: qb.stack(list(leaves), 0), *results)


def assert_trees_close(actual, expected, tolerance):
    for got, want in zip(qb.tree.leaves(actual), qb.tree.leaves(expected)):
        assert got.shape == want.shape and got.dtype == want.dtype, (got, want)
        for a, b in zip(got.to_flat_list(), want.to_flat_list()):
            assert abs(a - b) <= tolerance * max(1.0, abs(b)), (a, b)


def test_vmap_of_odeint_matches_a_loop_over_examples():
    # Every method is a bounded fori_loop (or scan), so vmap batches the
    # loop region: over the initial states, over the parameters, or both.
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        y0, k = batched_problem(dtype)
        for options in VMAP_SOLVES:
            solve = solver(options)
            for in_axes in ((0, None), (None, 0), (0, 0)):
                args = (y0 if in_axes[0] == 0 else y0[0], k if in_axes[1] == 0 else k[0])
                expected = stacked_examples(solve, *args, in_axes)
                batched = qb.vmap(solve, in_axes=in_axes)(*args)
                assert_trees_close(batched, expected, tolerance)
                assert_trees_close(qb.jit(qb.vmap(solve, in_axes=in_axes))(*args), expected, tolerance)


def test_vmap_of_odeint_gradients_matches_per_example_gradients():
    y0, k = batched_problem(qb.float64)
    for options in (VMAP_SOLVES[2], VMAP_SOLVES[5], VMAP_SOLVES[6], VMAP_SOLVES[7]):
        solve = solver(options)

        def loss(y0, k):
            return (solve(y0, k) ** 2).sum()

        gradient = qb.grad(loss, argnums=(0, 1))
        for in_axes in ((0, 0), (None, 0)):
            args = (y0 if in_axes[0] == 0 else y0[0], k)
            expected = stacked_examples(gradient, *args, in_axes)
            assert_trees_close(qb.vmap(gradient, in_axes=in_axes)(*args), expected, 1e-10)


def test_optional_device_vmap_of_odeint_parity():
    precisions = {"mlx": (None,), "cuda": (None, "float64")}
    runs = [(device, precision) for device in devices(*precisions) for precision in precisions[device]]
    for device, precision in runs:
        dtype = qb.float64 if precision == "float64" else qb.float32
        tolerance = 1e-10 if precision == "float64" else 1e-4
        y0, k = batched_problem(dtype)
        for options in (VMAP_SOLVES[2], VMAP_SOLVES[4], VMAP_SOLVES[6], VMAP_SOLVES[7], VMAP_SOLVES[8]):
            solve = solver(options)
            operations = [
                qb.vmap(solve),
                qb.vmap(qb.grad(lambda y0, k: (solve(y0, k) ** 2).sum(), argnums=1)),
            ]
            for operation in operations:
                expected = qb.jit(operation)(y0, k)
                device_options = {"device": device}
                if precision is not None:
                    device_options["precision"] = precision
                actual = qb.jit(operation, **device_options)(y0, k)
                assert_trees_close(actual, expected, tolerance)


if __name__ == "__main__":
    run(globals())
