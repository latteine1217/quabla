"""Explicit ODE integrators over the control-flow regions.

`odeint(f, y0, t_span, steps=..., method="rk4")` integrates `dy/dt =
f(y, t, *args)` from `t_span[0]` to `t_span[1]` in `steps` equal steps. A
traced state forms one `fori_loop` region (or a `scan` region when
`save=True`), so the solve is differentiable and runs under `jit` on every
device the loop regions support; an eager state runs the same steps eagerly.
Time points are computed as `t0 + i * dt` rather than accumulated, so long
integrations do not drift.

`method="dopri5"` is the adaptive Dormand-Prince 5(4) pair with error
control. It runs as a bounded loop of `max_steps` iterations whose body
stops advancing once `t1` is reached (the bounded-while formulation of
diffrax), so it stays reverse-mode differentiable, unlike `while_loop`.

Like the loop bodies it builds on, `f` must receive every traced value it
reads through `args`: closing over an outer tracer raises `TracerError`.
Eager arrays and Python numbers may be closed over; they become constants.
"""

import math
import operator

from . import _quabla
from ._array import asarray
from ._control import _bindings, fori_loop, scan
from ._ops import abs as _abs
from ._ops import exp, log, maximum, mean, minimum, sqrt, square, stop_gradient

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
# PI step-size controller (Hairer & Wanner II.4; Soederlind's 0.7/0.4 gains
# over the error order 5). Rejected steps fall back to the plain I-controller
# and may only shrink, as in Hairer's DOPRI5 code.
_SAFETY, _FACTOR_MIN, _FACTOR_MAX = 0.9, 0.2, 10.0
_ALPHA, _BETA = 0.7 / 5.0, 0.4 / 5.0
# Carry header: time, next step magnitude, last accepted error norm, and the
# accepted/rejected step counts; the state and the FSAL slope follow.
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


def _initial_step(f, y0, f0, t0, direction, rtol, atol, args):
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
        exp(log(0.01 / _quabla.where(flat, 1.0, largest)) / 5.0),
    )
    return stop_gradient(minimum(100.0 * h0, h1))


def _dopri5_step(f, carry, t1, direction, shape, size, rtol, atol, args):
    """One masked attempt; finished or rejected attempts leave the solution."""
    t, h, error_prev = carry[0], carry[1], carry[2]
    accepted, rejected = carry[3], carry[4]
    y = carry[_HEADER : _HEADER + size].reshape(shape)
    k1 = carry[_HEADER + size :].reshape(shape)
    remaining = (t1 - t) * direction
    active = remaining > 0.0
    # As in Hairer's DOPRI5, a step within 1% of t1 is stretched to land on
    # it rather than leaving a sliver for one more step.
    last = 1.01 * h >= remaining
    # Only the step that lands on t1 depends on t1; other step sizes are
    # controller outputs and carry no gradient (discretize-then-optimize).
    dt = _quabla.where(last, remaining, h) * direction
    slopes = [k1]
    for node, row in zip(_DOPRI5_NODES, _DOPRI5_ROWS):
        stage = y + dt * _combine(row, slopes)
        slopes.append(asarray(f(stage, t + node * dt, *args)))
    y_new = y + dt * _combine(_DOPRI5_WEIGHTS, slopes)
    k7 = asarray(f(y_new, t + dt, *args))
    error = dt * _combine(_DOPRI5_ERROR, [*slopes, k7])
    scale = atol + rtol * maximum(_abs(stop_gradient(y)), _abs(stop_gradient(y_new)))
    norm = _rms(stop_gradient(error) / scale)
    accept = norm <= 1.0
    # A NaN or overflowing error norm compares false, so it is rejected and
    # takes the largest shrink.
    finite = norm < 1e10
    bounded = _quabla.where(finite, maximum(norm, 1e-10), 1e10)
    grow = _SAFETY * exp(_BETA * log(error_prev) - _ALPHA * log(bounded))
    shrink = _SAFETY * exp(-0.2 * log(bounded))
    factor = _quabla.where(
        accept,
        minimum(maximum(grow, _FACTOR_MIN), _FACTOR_MAX),
        minimum(maximum(shrink, _FACTOR_MIN), 1.0),
    )
    moving = active.astype(carry.dtype)
    passed = moving * accept.astype(carry.dtype)
    take = passed > 0.5
    header = [
        _quabla.where(take, _quabla.where(last, t1, t + dt), t),
        _quabla.where(active, stop_gradient(_abs(dt) * factor), h),
        _quabla.where(take, maximum(bounded, 1e-4), error_prev),
        accepted + passed,
        rejected + (moving - passed),
    ]
    return _quabla.concat(
        [
            *(value.reshape([1]) for value in header),
            _quabla.where(take, y_new, y).reshape([size]),
            _quabla.where(take, k7, k1).reshape([size]),
        ],
        0,
    )


def _dopri5(f, y0, t0, t1, rtol, atol, max_steps, args):
    shape, size = y0.shape, math.prod(y0.shape)
    direction = _quabla.where(t1 >= t0, 1.0, -1.0).astype(y0.dtype)
    f0 = asarray(f(y0, t0, *args))
    h0 = _initial_step(f, y0, f0, t0, direction, rtol, atol, args)
    pieces = _bindings(
        (
            t0.reshape([1]),
            h0.reshape([1]),
            asarray([1.0, 0.0, 0.0], dtype=y0.dtype),
            y0.reshape([size]),
            f0.reshape([size]),
        )
    )
    carry = _quabla.concat(pieces, 0)

    def body(_, carry, t1, direction, *args):
        return _dopri5_step(f, carry, t1, direction, shape, size, rtol, atol, args)

    operands = (t1, direction, *args)
    if isinstance(carry, _quabla.TraceTensor):
        carry = fori_loop(0, max_steps, body, carry, operands=operands)
    else:
        # Eager solves stop at t1 instead of running masked iterations.
        for _ in range(max_steps):
            if not bool((t1 - carry[0]) * direction > 0.0):
                break
            carry = body(None, carry, *operands)
    y = carry[_HEADER : _HEADER + size].reshape(shape)
    info = {
        "t": carry[0],
        "accepted_steps": carry[3],
        "rejected_steps": carry[4],
        "success": (t1 - carry[0]) * direction <= 0.0,
    }
    return y, info


def _step(f, method, y, t, dt, args):
    nodes, coefficients, weights = _METHODS[method]
    slopes = []
    for node, row in zip(nodes, coefficients):
        stage = y
        for coefficient, slope in zip(row, slopes):
            if coefficient:
                stage = stage + (dt * coefficient) * slope
        slopes.append(asarray(f(stage, t + node * dt, *args)))
    increment = weights[0] * slopes[0]
    for weight, slope in zip(weights[1:], slopes[1:]):
        increment = increment + weight * slope
    return y + dt * increment


def _positive_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def odeint(
    f,
    y0,
    t_span,
    *,
    steps=None,
    method="rk4",
    args=(),
    save=False,
    rtol=1e-6,
    atol=1e-9,
    max_steps=512,
    info=False,
):
    """Integrate `dy/dt = f(y, t, *args)` over `t_span = (t0, t1)`.

    Fixed-step methods (`"rk4"`, the default, `"heun"`, `"euler"`) take
    `steps` equal steps and return the state at `t1`, or with `save=True`
    the `steps + 1` states at `t0, ..., t1` stacked on axis zero.

    `method="dopri5"` chooses its own steps for the error tolerance
    `atol + rtol * |y|` and returns the state at `t1`. It runs a bounded
    loop: a traced solve always costs `max_steps` step attempts (finished
    iterations are masked), while an eager solve stops at `t1`. An eager
    solve that does not reach `t1` raises `RuntimeError`; under `jit` the
    result is the state at the last time reached, so pass `info=True` to
    also get `{"t", "accepted_steps", "rejected_steps", "success"}`.

    `y0` is a single array; `t0`, `t1`, and `args` may be numbers or arrays
    (traced values among them are passed to the loop region as operands).
    """
    if not callable(f):
        raise TypeError("odeint requires a callable f(y, t, *args)")
    if method != "dopri5" and method not in _METHODS:
        raise ValueError(
            f"method must be one of {sorted([*_METHODS, 'dopri5'])}, got {method!r}"
        )
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise TypeError("t_span must be a pair (t0, t1)")
    y0 = asarray(y0)
    t0, t1 = (asarray(t, dtype=y0.dtype) for t in t_span)
    if method == "dopri5":
        if steps is not None:
            raise TypeError("dopri5 chooses its own steps; use max_steps instead of steps")
        if save:
            raise ValueError("save=True requires a fixed-step method")
        if not (float(rtol) >= 0.0 and float(atol) >= 0.0 and float(rtol) + float(atol) > 0.0):
            raise ValueError("rtol and atol must be non-negative and not both zero")
        if y0.dtype not in (_quabla.float32, _quabla.float64):
            raise TypeError(f"dopri5 requires a floating state, got {y0.dtype}")
        max_steps = _positive_int(max_steps, "max_steps")
        y, details = _dopri5(f, y0, t0, t1, float(rtol), float(atol), max_steps, args)
        success = details["success"]
        if not isinstance(success, _quabla.TraceTensor) and not bool(success):
            raise RuntimeError(
                f"dopri5 stopped at t={details['t'].item()} before t1 after "
                f"max_steps={max_steps} step attempts; increase max_steps or "
                "loosen rtol/atol"
            )
        return (y, details) if info else y
    if steps is None:
        raise TypeError(f"method {method!r} requires steps")
    steps = _positive_int(steps, "steps")
    if info:
        raise ValueError("info=True is only available for method='dopri5'")
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
