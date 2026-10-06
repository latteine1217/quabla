"""Explicit and Rosenbrock ODE integrators over the control-flow regions.

`odeint(f, y0, t_span, steps=..., method="rk4")` integrates `dy/dt =
f(y, t, *args)` from `t_span[0]` to `t_span[1]` in `steps` equal steps. A
traced state forms one `fori_loop` region (or a `scan` region when
`save=True`), so the solve is differentiable and runs under `jit` on every
device the loop regions support; an eager state runs the same steps eagerly.
Time points are computed as `t0 + i * dt` rather than accumulated, so long
integrations do not drift.

`method="dopri5"` is the adaptive Dormand-Prince 5(4) pair with error
control, and `method="rosenbrock23"` the adaptive, L-stable Rosenbrock 2(3)
method of MATLAB's ode23s for stiff problems. Both run as a bounded loop of
`max_steps` iterations whose body stops advancing once `t1` is reached (the
bounded-while formulation of diffrax), so they stay reverse-mode
differentiable, unlike `while_loop`.

`saveat=ts` returns the states at the times `ts` from each method's
continuous extension, written inside the loop with masked updates, so the
adaptive methods do not shorten their steps to hit the save times.

Like the loop bodies it builds on, `f` must receive every traced value it
reads through `args`: closing over an outer tracer raises `TracerError`.
Eager arrays and Python numbers may be closed over; they become constants.
"""

import math
import operator

from . import _quabla
from ._array import asarray, eye, zeros
from ._control import _bindings, fori_loop, scan
from ._ops import abs as _abs
from ._ops import exp, log, maximum, mean, minimum, sqrt, square, stop_gradient
from ._transforms import jvp, vmap
from .linalg import solve

__all__ = ["odeint"]

# Butcher tableaux of the explicit methods: (stage nodes c, stage weights
# a, solution weights b). Classical RK4 is fourth order, Heun's method (the
# explicit trapezoid) second order, and Euler first order.
_METHODS = {
    "euler": ((0.0,), ((),), (1.0,)),
    "heun": ((0.0, 1.0), ((), (1.0,)), (0.5, 0.5)),
    "rk4": (
        (0.0, 0.5, 0.5, 1.0),
        ((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
        (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    ),
}
_ADAPTIVE = ("dopri5", "rosenbrock23")


# Dormand-Prince 5(4) (Hairer, Norsett & Wanner, Solving ODEs I, Table
# II.5.2). The fifth-order weights equal the last stage row, so the final
# slope is f at the new state and is reused as the next first slope (FSAL).
# `_DOPRI5_ERROR` is the fifth- minus the embedded fourth-order weights.
_DOPRI5_NODES = (1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0)
_DOPRI5_ROWS = (
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
)
_DOPRI5_WEIGHTS = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0)
_DOPRI5_ERROR = (
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
)
# Shampine's fourth-order continuous extension of the pair, in the form of
# Hairer's DOPRI5 code (subroutine CONTD5; Solving ODEs I, Section II.6):
# the weights of its last coefficient over the seven slopes.
_DOPRI5_DENSE = (
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
)

# Rosenbrock 2(3) of Shampine & Reichelt, "The MATLAB ODE Suite", SIAM J.
# Sci. Comput. 18 (1997), Section 3 (MATLAB's ode23s, Rosenbrock23 of
# DifferentialEquations.jl). With W = I - h d J, J = df/dy and T = df/dt at
# (t, y), d = 1 / (2 + sqrt(2)) and e32 = 6 + sqrt(2):
#   F0 = f(t, y)                k1 = W^-1 (F0 + h d T)
#   F1 = f(t + h/2, y + h/2 k1) k2 = W^-1 (F1 - k1) + k1
#   y_new = y + h k2            (second order, L-stable)
#   F2 = f(t + h, y_new)        k3 = W^-1 (F2 - e32 (k2 - F1) - 2 (k1 - F0) + h d T)
#   error = h/6 (k1 - 2 k2 + k3), the difference to the third-order solution.
# F2 equals the next step's F0 (first same as last), but the next step's
# time derivative T comes from a JVP that evaluates f at that point anyway,
# so F0 is taken from that JVP instead of being carried.
_ROS_D = 1.0 / (2.0 + math.sqrt(2.0))
_ROS_E32 = 6.0 + math.sqrt(2.0)

# PI step-size controller (Hairer & Wanner II.4; Soederlind's 0.7/0.4 gains
# over the order of the error estimate: 5 for dopri5, 3 for rosenbrock23).
# Rejected steps fall back to the plain I-controller and may only shrink,
# as in Hairer's DOPRI5 code.
_SAFETY, _FACTOR_MIN, _FACTOR_MAX = 0.9, 0.2, 10.0
_ERROR_ORDER = {"dopri5": 5.0, "rosenbrock23": 3.0}
# Carry header: time, next step magnitude, last accepted error norm, and the
# accepted/rejected step counts; the state, dopri5's FSAL slope, and the
# saved states follow.
_HEADER = 5


def _combine(weights, slopes):
    total = None
    for weight, slope in zip(weights, slopes):
        if weight:
            term = weight * slope
            total = term if total is None else total + term
    return total


def _rms(x):
    return sqrt(mean(square(x)))


def _initial_step(f, y0, f0, t0, direction, rtol, atol, order, args):
    """Starting step of Hairer & Wanner, Solving ODEs I, Section II.4."""
    y0, f0 = stop_gradient(y0), stop_gradient(f0)
    scale = atol + rtol * _abs(y0)
    d0, d1 = _rms(y0 / scale), _rms(f0 / scale)
    tiny = minimum(d0, d1) < 1e-5
    # The guarded denominators keep the unselected `where` branch finite.
    h0 = _quabla.where(tiny, 1e-6, 0.01 * d0 / _quabla.where(tiny, 1.0, d1))
    f1 = stop_gradient(asarray(f(y0 + (direction * h0) * f0, t0 + direction * h0, *args)))
    d2 = _rms((f1 - f0) / scale) / h0
    largest = maximum(d1, d2)
    flat = largest <= 1e-15
    h1 = _quabla.where(
        flat,
        maximum(h0 * 1e-3, 1e-6),
        exp(log(0.01 / _quabla.where(flat, 1.0, largest)) / order),
    )
    return stop_gradient(minimum(100.0 * h0, h1))


def _dopri5_attempt(f, y, k1, t, dt, args):
    """A Dormand-Prince step from the FSAL slope `k1`: the new state, its
    error estimate, the next FSAL slope, and the dense output
    `theta -> y(t + theta dt)` over the flattened state."""
    slopes = [k1]
    for node, row in zip(_DOPRI5_NODES, _DOPRI5_ROWS):
        stage = y + dt * _combine(row, slopes)
        slopes.append(asarray(f(stage, t + node * dt, *args)))
    y_new = y + dt * _combine(_DOPRI5_WEIGHTS, slopes)
    k7 = asarray(f(y_new, t + dt, *args))
    error = dt * _combine(_DOPRI5_ERROR, [*slopes, k7])

    def dense(theta):
        # y + theta (r2 + (1 - theta) (r3 + theta (r4 + (1 - theta) r5))):
        # the quartic through y and y_new with the end slopes k1 and k7,
        # fourth-order accurate inside the step.
        size = math.prod(y.shape)
        flat = [slope.reshape([size]) for slope in (*slopes, k7)]
        r2 = (y_new - y).reshape([size])
        r3 = dt * flat[0] - r2
        r4 = r2 - dt * flat[6] - r3
        r5 = dt * _combine(_DOPRI5_DENSE, flat)
        rest = 1.0 - theta
        return y.reshape([size]) + theta * (r2 + rest * (r3 + theta * (r4 + rest * r5)))

    return y_new, error, k7, dense


def _linearization(f, shape, size, dtype, args):
    """`(y, t) -> (f(y, t), J, T)` over the flattened state, with J = df/dy
    `[size, size]` and T = df/dt `[size]`.

    One forward-mode pass of `size + 1` batched tangents (the columns of J
    and the time direction) shares a single primal evaluation of f; a
    forward-mode `jacobian` plus a separate time JVP would evaluate f twice.
    The transform is built once per solve and takes the state and time as
    arguments, so an eager solve reuses one staged program for every step
    instead of baking each step's state into a new trace.
    """
    basis = _quabla.concat([eye(size, dtype=dtype), zeros([1, size], dtype)], 0)
    time_basis = _quabla.concat([zeros([size], dtype), asarray([1.0], dtype=dtype)], 0)

    def flat(y, t, *args):
        return asarray(f(y.reshape(shape), t, *args)).reshape([size])

    def tangent(vy, vt, y, t, *args):
        return jvp(lambda y, t: flat(y, t, *args), (y, t), (vy, vt))

    batched = vmap(tangent, in_axes=(0, 0, None, None, *([None] * len(args))), out_axes=(None, 0))

    def linearize(y, t, args):
        value, columns = batched(basis, time_basis, y, t, *args)
        # Row i of `columns` is J e_i, the i-th column of J; the last row is T.
        return value, columns[:size].transpose([1, 0]), columns[size]

    return linearize


def _rosenbrock23_attempt(f, linearize, y, t, dt, args):
    """A Rosenbrock 2(3) step (see `_ROS_D`): the new state, its error
    estimate, and the dense output `theta -> y(t + theta dt)` over the
    flattened state.

    The three stages share W but are sequential (each right-hand side needs
    the previous stage), so they cannot be one multi-column solve. Each
    calls `solve`, LU with partial pivoting: backward stable, and three
    factorizations (about n^3 flops) cost less than the library's ways of
    reusing one factorization, `inv(W)` and three products (about 7n^3/3
    flops, and not backward stable for an ill-conditioned W) or `qr(W)` and
    triangular solves (about 8n^3/3 flops with Q formed). The reverse pass
    differentiates through J as well, so gradients are those of the
    discrete scheme.
    """
    shape = y.shape
    size = math.prod(shape)
    y = y.reshape([size])
    f0, jac, slope = linearize(y, t, args)
    gamma = dt * _ROS_D
    w = eye(size, dtype=y.dtype) - gamma * jac
    k1 = solve(w, f0 + gamma * slope)
    f1 = asarray(f((y + (0.5 * dt) * k1).reshape(shape), t + 0.5 * dt, *args)).reshape([size])
    k2 = solve(w, f1 - k1) + k1
    y_new = y + dt * k2
    f2 = asarray(f(y_new.reshape(shape), t + dt, *args)).reshape([size])
    k3 = solve(w, f2 - _ROS_E32 * (k2 - f1) - 2.0 * (k1 - f0) + gamma * slope)
    error = (dt / 6.0) * (k1 - 2.0 * k2 + k3)

    def dense(theta):
        # The method's own second-order interpolant (Shampine & Reichelt,
        # Section 3): y + h (s (1 - s) k1 + s (s - 2d) k2) / (1 - 2d).
        scale = 1.0 / (1.0 - 2.0 * _ROS_D)
        first = theta * (1.0 - theta) * scale
        second = theta * (theta - 2.0 * _ROS_D) * scale
        return y + dt * (first * k1 + second * k2)

    return y_new.reshape(shape), error.reshape(shape), dense


def _hermite(y, y_new, k1, k2, dt, size):
    """Cubic Hermite interpolant `theta -> y(t + theta dt)` from the step's
    end values and slopes. Its interpolation error is O(dt^4), so the saved
    states keep the global order of each fixed-step method (up to 4)."""
    y, delta = y.reshape([size]), (y_new - y).reshape([size])
    start, end = dt * k1.reshape([size]), dt * k2.reshape([size])

    def dense(theta):
        quadratic = 3.0 * delta - 2.0 * start - end
        cubic = start + end - 2.0 * delta
        return y + theta * (start + theta * (quadratic + theta * cubic))

    return dense


def _save(dense, saveat, saved, t, t_new, dt, direction, take):
    """Write the save times inside the step (t, t_new] from its dense
    output and keep the others. Accepted steps tile the span, so each save
    time after t0 is written by exactly one step."""
    times = saveat.reshape([saveat.shape[0], 1])
    inside = ((times - t) * direction > 0.0) & ((times - t_new) * direction <= 0.0)
    if take is not None:
        inside = inside & take
    # A finished iteration has dt = 0; the guard keeps the unselected theta
    # finite, so its zero cotangent does not turn into NaN.
    safe = _quabla.where(dt.equal(0.0), 1.0, dt)
    theta = minimum(maximum((times - t) / safe, 0.0), 1.0)
    return _quabla.where(inside, dense(theta), saved)


def _initial_saved(saveat, y0, t0, size):
    """y0 at the save times equal to t0 and NaN elsewhere until a step
    reaches them, so a time outside the span, or one a `jit` solve did not
    reach, reads as NaN instead of a plausible state."""
    times = saveat.reshape([saveat.shape[0], 1])
    unset = zeros([1, size], y0.dtype) + float("nan")
    return _quabla.where(times.equal(t0), y0.reshape([1, size]), unset)


def _adaptive_step(f, method, linearize, carry, t1, direction, saveat, shape, size, rtol, atol, args):
    """One masked attempt; finished or rejected attempts leave the solution."""
    t, h, error_prev = carry[0], carry[1], carry[2]
    accepted, rejected = carry[3], carry[4]
    y = carry[_HEADER : _HEADER + size].reshape(shape)
    offset = _HEADER + size
    remaining = (t1 - t) * direction
    active = remaining > 0.0
    # As in Hairer's DOPRI5, a step within 1% of t1 is stretched to land on
    # it rather than leaving a sliver for one more step.
    last = 1.01 * h >= remaining
    # Only the step that lands on t1 depends on t1; other step sizes are
    # controller outputs and carry no gradient (discretize-then-optimize).
    dt = _quabla.where(last, remaining, h) * direction
    k1 = None
    if method == "dopri5":
        k1 = carry[offset : offset + size].reshape(shape)
        offset += size
        y_new, error, k7, dense = _dopri5_attempt(f, y, k1, t, dt, args)
    else:
        y_new, error, dense = _rosenbrock23_attempt(f, linearize, y, t, dt, args)
    scale = atol + rtol * maximum(_abs(stop_gradient(y)), _abs(stop_gradient(y_new)))
    norm = _rms(stop_gradient(error) / scale)
    accept = norm <= 1.0
    # A NaN or overflowing error norm compares false, so it is rejected and
    # takes the largest shrink.
    finite = norm < 1e10
    bounded = _quabla.where(finite, maximum(norm, 1e-10), 1e10)
    order = _ERROR_ORDER[method]
    grow = _SAFETY * exp((0.4 / order) * log(error_prev) - (0.7 / order) * log(bounded))
    shrink = _SAFETY * exp((-1.0 / order) * log(bounded))
    factor = _quabla.where(
        accept,
        minimum(maximum(grow, _FACTOR_MIN), _FACTOR_MAX),
        minimum(maximum(shrink, _FACTOR_MIN), 1.0),
    )
    moving = active.astype(carry.dtype)
    passed = moving * accept.astype(carry.dtype)
    take = passed > 0.5
    t_new = _quabla.where(last, t1, t + dt)
    header = [
        _quabla.where(take, t_new, t),
        _quabla.where(active, stop_gradient(_abs(dt) * factor), h),
        _quabla.where(take, maximum(bounded, 1e-4), error_prev),
        accepted + passed,
        rejected + (moving - passed),
    ]
    pieces = [
        *(value.reshape([1]) for value in header),
        _quabla.where(take, y_new, y).reshape([size]),
    ]
    if k1 is not None:
        pieces.append(_quabla.where(take, k7, k1).reshape([size]))
    if saveat is not None:
        count = saveat.shape[0]
        saved = carry[offset:].reshape([count, size])
        saved = _save(dense, saveat, saved, t, t_new, dt, direction, take)
        pieces.append(saved.reshape([count * size]))
    return _quabla.concat(pieces, 0)


def _adaptive(f, method, y0, t0, t1, rtol, atol, max_steps, saveat, args):
    shape, size = y0.shape, math.prod(y0.shape)
    direction = _quabla.where(t1 >= t0, 1.0, -1.0).astype(y0.dtype)
    f0 = asarray(f(y0, t0, *args))
    h0 = _initial_step(f, y0, f0, t0, direction, rtol, atol, _ERROR_ORDER[method], args)
    initial = [
        t0.reshape([1]),
        h0.reshape([1]),
        asarray([1.0, 0.0, 0.0], dtype=y0.dtype),
        y0.reshape([size]),
    ]
    if method == "dopri5":
        initial.append(f0.reshape([size]))
    if saveat is not None:
        initial.append(_initial_saved(saveat, y0, t0, size).reshape([saveat.shape[0] * size]))
    carry = _quabla.concat(_bindings(initial), 0)
    linearize = None
    if method == "rosenbrock23":
        linearize = _linearization(f, shape, size, y0.dtype, args)
    extra = () if saveat is None else (saveat,)

    def body(_, carry, t1, direction, *operands):
        times = operands[0] if extra else None
        args = operands[len(extra) :]
        return _adaptive_step(
            f, method, linearize, carry, t1, direction, times, shape, size, rtol, atol, args
        )

    operands = (t1, direction, *extra, *args)
    if isinstance(carry, _quabla.TraceTensor):
        carry = fori_loop(0, max_steps, body, carry, operands=operands)
    else:
        # Eager solves stop at t1 instead of running masked iterations.
        for _ in range(max_steps):
            if not bool((t1 - carry[0]) * direction > 0.0):
                break
            carry = body(None, carry, *operands)
    if saveat is None:
        y = carry[_HEADER : _HEADER + size].reshape(shape)
    else:
        count = saveat.shape[0]
        y = carry[carry.shape[0] - count * size :].reshape([count, *shape])
    info = {
        "t": carry[0],
        "accepted_steps": carry[3],
        "rejected_steps": carry[4],
        "success": (t1 - carry[0]) * direction <= 0.0,
    }
    return y, info


def _step(f, method, y, t, dt, args, first=None):
    """One explicit Runge-Kutta step; `first` is the first slope f(y, t)
    when the caller already has it."""
    nodes, coefficients, weights = _METHODS[method]
    slopes = [] if first is None else [first]
    for node, row in list(zip(nodes, coefficients))[len(slopes) :]:
        stage = y
        for coefficient, slope in zip(row, slopes):
            if coefficient:
                stage = stage + (dt * coefficient) * slope
        slopes.append(asarray(f(stage, t + node * dt, *args)))
    increment = weights[0] * slopes[0]
    for weight, slope in zip(weights[1:], slopes[1:]):
        increment = increment + weight * slope
    return y + dt * increment


def _fixed_saveat(f, method, y0, t0, t1, steps, saveat, args):
    """Fixed steps with the states at `saveat` from cubic Hermite
    interpolation of each step. The end slope f(y_new, t_new) is carried
    into the next step as its first stage f(y, t), which every method
    evaluates anyway, so the interpolation costs one extra evaluation of f
    per solve, not per step, and the steps are those of the solve without
    `saveat`."""
    shape, size, count = y0.shape, math.prod(y0.shape), saveat.shape[0]
    dt = (t1 - t0) / steps
    direction = _quabla.where(t1 >= t0, 1.0, -1.0).astype(y0.dtype)
    f0 = asarray(f(y0, t0, *args))
    initial = [
        y0.reshape([size]),
        f0.reshape([size]),
        _initial_saved(saveat, y0, t0, size).reshape([count * size]),
    ]
    carry = _quabla.concat(_bindings(initial), 0)

    def advance(i, carry, t0, t1, dt, direction, saveat, *args):
        y = carry[:size].reshape(shape)
        k1 = carry[size : 2 * size].reshape(shape)
        saved = carry[2 * size :].reshape([count, size])
        t, t_next = t0 + i * dt, t0 + (i + 1) * dt
        y_new = _step(f, method, y, t, dt, args, first=k1)
        k2 = asarray(f(y_new, t_next, *args))
        # The last step closes the window at t1 itself, which t0 + steps * dt
        # can miss by rounding.
        final = asarray(i, dtype=y0.dtype) >= steps - 1
        upper = _quabla.where(final, t1, t_next)
        dense = _hermite(y, y_new, k1, k2, dt, size)
        saved = _save(dense, saveat, saved, t, upper, dt, direction, None)
        return _quabla.concat(
            [y_new.reshape([size]), k2.reshape([size]), saved.reshape([count * size])], 0
        )

    operands = (t0, t1, dt, direction, saveat, *args)
    carry = fori_loop(0, steps, advance, carry, operands=operands)
    return carry[2 * size :].reshape([count, *shape])


def _positive_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _save_times(saveat, t0, t1, dtype):
    """`saveat` as a 1-D array of the state dtype, checked against the span
    when its values and the span are concrete."""
    saveat = asarray(saveat, dtype=dtype)
    if len(saveat.shape) != 1 or saveat.shape[0] < 1:
        raise ValueError(f"saveat must be a non-empty 1-D array of times, got shape {saveat.shape}")
    if any(isinstance(value, _quabla.TraceTensor) for value in (saveat, t0, t1)):
        return saveat
    times, start, end = saveat.tolist(), t0.item(), t1.item()
    sign = 1.0 if end >= start else -1.0
    if any(sign * (later - earlier) < 0.0 for earlier, later in zip(times, times[1:])):
        raise ValueError("saveat must be ordered in the direction of integration from t0 to t1")
    if not all(sign * (time - start) >= 0.0 and sign * (end - time) >= 0.0 for time in times):
        raise ValueError(f"saveat times must lie inside the span [{start}, {end}]")
    return saveat


def odeint(
    f,
    y0,
    t_span,
    *,
    steps=None,
    method="rk4",
    args=(),
    save=False,
    saveat=None,
    rtol=1e-6,
    atol=1e-9,
    max_steps=512,
    info=False,
):
    """Integrate `dy/dt = f(y, t, *args)` over `t_span = (t0, t1)`.

    Fixed-step methods (`"rk4"`, the default, `"heun"`, `"euler"`) take
    `steps` equal steps and return the state at `t1`, or with `save=True`
    the `steps + 1` states at `t0, ..., t1` stacked on axis zero.

    The adaptive methods choose their own steps for the error tolerance
    `atol + rtol * |y|` and return the state at `t1`: `"dopri5"`, the
    explicit Dormand-Prince 5(4) pair, and `"rosenbrock23"`, the L-stable
    Rosenbrock 2(3) method of MATLAB's ode23s for stiff problems, which
    forms the Jacobian of `f` by forward mode (one batched pass of
    `y0.size + 1` tangents, the last giving df/dt) and solves three linear
    systems per step. They run a bounded loop: a traced solve always costs
    `max_steps` step attempts (finished iterations are masked), while an
    eager solve stops at `t1`. An eager solve that does not reach `t1`
    raises `RuntimeError`; under `jit` the result is the state at the last
    time reached, so pass `info=True` to also get `{"t", "accepted_steps",
    "rejected_steps", "success"}`.

    `saveat`, a 1-D array of times ordered from `t0` to `t1` and inside the
    span, returns the states at those times stacked on axis zero instead
    of the final state, for every method. They come from each method's
    continuous extension, without shortening steps to land on them:
    Shampine's fourth-order interpolant for `"dopri5"`, the method's own
    second-order interpolant for `"rosenbrock23"`, and cubic Hermite
    interpolation of the step's end values and slopes for the fixed-step
    methods. Under `jit` a save time the solve did not reach, or one
    outside the span, is NaN. `saveat` and `save=True` cannot be combined.

    `y0` is a single array; `t0`, `t1`, `saveat`, and `args` may be numbers
    or arrays (traced values among them are passed to the loop region as
    operands).
    """
    if not callable(f):
        raise TypeError("odeint requires a callable f(y, t, *args)")
    if method not in _ADAPTIVE and method not in _METHODS:
        raise ValueError(
            f"method must be one of {sorted([*_METHODS, *_ADAPTIVE])}, got {method!r}"
        )
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise TypeError("t_span must be a pair (t0, t1)")
    y0 = asarray(y0)
    t0, t1 = (asarray(t, dtype=y0.dtype) for t in t_span)
    if saveat is not None:
        if save:
            raise ValueError("save=True and saveat cannot be combined")
        saveat = _save_times(saveat, t0, t1, y0.dtype)
    if method in _ADAPTIVE:
        if steps is not None:
            raise TypeError(f"{method} chooses its own steps; use max_steps instead of steps")
        if save:
            raise ValueError("save=True requires a fixed-step method; use saveat")
        if not (float(rtol) >= 0.0 and float(atol) >= 0.0 and float(rtol) + float(atol) > 0.0):
            raise ValueError("rtol and atol must be non-negative and not both zero")
        if y0.dtype not in (_quabla.float32, _quabla.float64):
            raise TypeError(f"{method} requires a floating state, got {y0.dtype}")
        max_steps = _positive_int(max_steps, "max_steps")
        y, details = _adaptive(
            f, method, y0, t0, t1, float(rtol), float(atol), max_steps, saveat, args
        )
        success = details["success"]
        if not isinstance(success, _quabla.TraceTensor) and not bool(success):
            raise RuntimeError(
                f"{method} stopped at t={details['t'].item()} before t1 after "
                f"max_steps={max_steps} step attempts; increase max_steps or "
                "loosen rtol/atol"
            )
        return (y, details) if info else y
    if steps is None:
        raise TypeError(f"method {method!r} requires steps")
    steps = _positive_int(steps, "steps")
    if info:
        raise ValueError("info=True is only available for the adaptive methods")
    if saveat is not None:
        return _fixed_saveat(f, method, y0, t0, t1, steps, saveat, args)
    dt = (t1 - t0) / steps
    # The loop index is a scalar array of the state dtype inside a region and
    # a Python int otherwise; both give `t0 + i * dt` in the state dtype.
    operands = (t0, dt, *args)

    def advance(i, y, t0, dt, *args):
        return _step(f, method, y, t0 + i * dt, dt, args)

    if not save:
        return fori_loop(0, steps, advance, y0, operands=operands)

    def record(y, i, t0, dt, *args):
        next_y = advance(i, y, t0, dt, *args)
        return next_y, next_y

    _, states = scan(record, y0, length=steps, operands=operands)
    return _quabla.concat([y0.reshape([1, *y0.shape]), states], 0)
