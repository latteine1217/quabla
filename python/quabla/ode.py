"""Fixed-step explicit ODE integrators over the control-flow regions.

`odeint(f, y0, t_span, steps=..., method="rk4")` integrates `dy/dt =
f(y, t, *args)` from `t_span[0]` to `t_span[1]` in `steps` equal steps. A
traced state forms one `fori_loop` region (or a `scan` region when
`save=True`), so the solve is differentiable and runs under `jit` on every
device the loop regions support; an eager state runs the same steps eagerly.

Like the loop bodies it builds on, `f` must receive every traced value it
reads through `args`: closing over an outer tracer raises `TracerError`.
Eager arrays and Python numbers may be closed over; they become constants.

Time points are computed as `t0 + i * dt` rather than accumulated, so long
integrations do not drift. Step-size control needs a loop with a traced
termination condition and is not provided.
"""

import operator

from . import _quabla
from ._array import asarray
from ._control import fori_loop, scan

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


def odeint(f, y0, t_span, *, steps, method="rk4", args=(), save=False):
    """Integrate `dy/dt = f(y, t, *args)` over `t_span = (t0, t1)`.

    Returns the state at `t1`, or with `save=True` the states at the
    `steps + 1` equally spaced times `t0, ..., t1` stacked on axis zero.
    `y0` is a single array; `t0`, `t1`, and `args` may be numbers or arrays
    (traced values among them are passed to the loop region as operands).
    `method` is `"rk4"` (default), `"heun"`, or `"euler"`.
    """
    if not callable(f):
        raise TypeError("odeint requires a callable f(y, t, *args)")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    if isinstance(steps, bool):
        raise TypeError("steps must be a positive integer")
    steps = operator.index(steps)
    if steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise TypeError("t_span must be a pair (t0, t1)")
    y0 = asarray(y0)
    t0, t1 = (asarray(t, dtype=y0.dtype) for t in t_span)
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
