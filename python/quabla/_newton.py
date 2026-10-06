"""Root finding: `newton(f, x0, args=...)` solves `f(x, *args) = 0` by
Newton's method with a backtracking line search.

The iteration is a `while_loop`, so an eager solve runs in Python and a
traced one is a single loop region that stops as soon as it converges;
Newton converges in a handful of iterations, so a bounded `fori_loop`
would mostly run masked iterations up to `maxiter`. Derivatives with
respect to `args` come from the implicit function theorem at the root (see
`_implicit`), not from the iterations, which also makes the `while_loop`'s
lack of a reverse mode irrelevant.
"""

import math

from . import _quabla
from ._array import asarray, eye, zeros
from ._control import _bindings, while_loop
from ._implicit import (
    ORDER,
    args_vjp,
    check_args,
    finish,
    implicit,
    positive_int,
    split_args,
    tolerance,
)
from ._quabla import Tensor, TraceTensor, float32, float64
from ._transforms import jacobian
from .linalg import qr as _qr
from .linalg import solve as _solve

__all__ = ["newton"]

# Armijo sufficient-decrease constant for the merit function ||f||^2 / 2
# (Nocedal & Wright, Numerical Optimization, Section 3.1), and the smallest
# step fraction the backtracking tries before declaring a stall: 30
# halvings shrink the Newton step by about 1e-9.
_ARMIJO = 1e-4
_MIN_STEP = 2.0**-30

# Loop status in the carry.
_RUNNING, _CONVERGED, _STALLED = 0.0, 1.0, 2.0


def _flat_function(f, shape, dtype):
    size = math.prod(shape)

    def apply(v, *args):
        out = f(v.reshape(list(shape)), *args)
        if not isinstance(out, (Tensor, TraceTensor)):
            out = asarray(out, dtype=dtype)
        if list(out.shape) != list(shape):
            raise ValueError(
                f"newton requires f(x, *args) to return an array of x's shape {list(shape)}, "
                f"got {list(out.shape)}"
            )
        if out.dtype != dtype:
            raise TypeError(
                f"newton requires f(x, *args) to return x's dtype {dtype}, got {out.dtype}; "
                "pass args of that dtype or cast the result"
            )
        return out.reshape([size])

    return apply


def _norm(v):
    # sqrt(v . v): `Tensor.norm` lowers its rescaling maximum to `n` scalar
    # comparisons, which every iteration would pay for.
    return (v * v).sum().sqrt()


def _jacobian(apply, x, args):
    """The dense `[n, n]` Jacobian of `apply` with respect to `x`; the array
    leaves of `args` are explicit arguments, as a loop body requires."""
    arrays, rebuild = split_args(tuple(args))
    return jacobian(lambda v, *values: apply(v, *rebuild(*values)))(x, *arrays)


def _newton_step(jacobian, fx):
    """The Newton step `-J^-1 f` from a Householder QR of `J`, and whether
    `R` has an exactly zero pivot. LU (`linalg.solve`) would raise on an
    exactly singular `J` (a function with no root, such as `exp(x) + 1`,
    drives the Jacobian to underflow); here the zero pivots are replaced by
    ones so the step stays finite, and the caller stops with failure. QR
    costs about twice the flops of LU and is backward stable without
    pivoting."""
    n = fx.shape[0]
    q, r = _qr(jacobian)
    identity = eye(n, dtype=fx.dtype)
    pivots = (r * identity).sum(axis=1)
    singular = _quabla.equal(pivots, 0.0).astype(fx.dtype)
    safe = r + identity * singular.reshape([1, n])
    step = safe.solve_triangular((q.T @ fx.reshape([n, 1])), False, False).reshape([n])
    return -step, singular.sum() > 0.5


def _iteration(apply, tol, maxiter):
    """Newton's method with a backtracking (Armijo) line search on the merit
    function `phi(x) = ||f(x)||^2 / 2`, for which the Newton direction `dx`
    has slope `-2 phi`.

    An iteration converges when the full Newton step is small,
    `||dx|| <= tol * (1 + ||x||)`; that step is still applied, and with
    quadratic convergence it leaves an error of order `tol^2`, so the
    default `tol = sqrt(eps)` gives a root accurate to rounding. Testing the
    step instead of `||f||` keeps the test independent of the scale of `f`.
    Otherwise the step `x + t dx` is taken with the largest `t = 2^-i` that
    decreases `phi` sufficiently; when no `t >= 2^-30` does (a local minimum
    of `phi` that is not a root, or a NaN) or the Jacobian is exactly
    singular, the iteration stops and reports failure. NaN comparisons are false, so every
    test is written to fail, not pass, on NaN.
    """

    def solve(x0, *args):
        n = x0.shape[0]
        dtype = x0.dtype
        arrays, rebuild = split_args(args)
        carry = _quabla.concat(_bindings((x0, zeros([2], dtype))), 0)

        def running(carry, *arrays):
            return _quabla.equal(carry[n + 1], _RUNNING) & (carry[n] < maxiter)

        def searching(state, x, dx, phi0, skip, *arrays):
            t, phi = state[0], state[1]
            sufficient = phi <= (1.0 - 2.0 * _ARMIJO * t) * phi0
            return (skip < 0.5) & sufficient.logical_not() & (t > _MIN_STEP)

        def halve(state, x, dx, phi0, skip, *arrays):
            t = 0.5 * state[0]
            trial = apply(x + t * dx, *rebuild(*arrays))
            return _quabla.concat([t.reshape([1]), (0.5 * (trial * trial).sum()).reshape([1])], 0)

        def step(carry, *arrays):
            args = rebuild(*arrays)
            x, k = carry[:n], carry[n]
            fx = apply(x, *args)
            dx, singular = _newton_step(_jacobian(apply, x, args), fx)
            small = (_norm(dx) <= tol * (1.0 + _norm(x))) & singular.logical_not()
            phi0 = 0.5 * (fx * fx).sum()
            trial = apply(x + dx, *args)
            state = _quabla.concat(
                _bindings((asarray([1.0], dtype=dtype), (0.5 * (trial * trial).sum()).reshape([1]))),
                0,
            )
            # No search after a converged or singular step; loop operands must
            # be floating, so the flag travels as 0 or 1.
            flag = (small | singular).astype(dtype)
            state = while_loop(searching, halve, state, operands=(x, dx, phi0, flag, *arrays))
            t, phi = state[0], state[1]
            accepted = (small | (phi <= (1.0 - 2.0 * _ARMIJO * t) * phi0)) & singular.logical_not()
            x = _quabla.where(accepted, x + t * dx, x)
            status = _quabla.where(
                small, _CONVERGED, _quabla.where(accepted, _RUNNING, _STALLED)
            ).astype(dtype)
            return _quabla.concat([x, (k + 1.0).reshape([1]), status.reshape([1])], 0)

        carry = while_loop(running, step, carry, operands=tuple(arrays))
        converged = _quabla.equal(carry[n + 1], _CONVERGED).astype(dtype)
        return carry[:n], _quabla.concat([carry[n].reshape([1]), converged.reshape([1])], 0)

    return solve


def _solver(apply, tol, maxiter, order):
    def adjoint(order, x, operands, x_bar):
        # J^T lambda = x_bar with the dense Jacobian at the root, then
        # args_bar = -(df/dargs)^T lambda; the initial guess gets none.
        _, *args = operands
        lam = _solve(_jacobian(apply, x, args).T, x_bar)
        return (None, *args_vjp(lambda *values: apply(x, *values), tuple(args), -lam))

    return implicit("newton", _iteration(apply, tol, maxiter), adjoint, order)


def newton(f, x0, *, args=(), tol=None, maxiter=50, info=False):
    """A root `x` of `f(x, *args) = 0`, by Newton's method from `x0`.

    `f` maps an array shaped like `x0` (any shape; the solve treats it as a
    flat vector of `n` unknowns) to an array of the same shape and dtype.
    Each iteration forms the dense `n x n` Jacobian with `quabla.jacobian`
    and solves with its Householder QR, so this suits small and moderate
    systems; an exactly singular Jacobian stops the iteration with failure
    instead of raising. A backtracking line search on `||f||^2` makes the iteration
    robust far from the root. An iteration converges when the Newton step
    satisfies `||dx|| <= tol * (1 + ||x||)` (the step is still applied);
    `tol` defaults to the square root of the machine epsilon of `x0`'s
    dtype, which quadratic convergence turns into a root accurate to
    rounding. At most `maxiter` iterations run.

    Derivatives with respect to the array leaves of `args` follow the
    implicit function theorem, `dx/dargs = -J^-1 df/dargs` at the root,
    computed from one dense Jacobian and solve with the transposed Jacobian;
    `x0` receives no gradient. Pass every array `f` depends on through
    `args`: values a Python `f` closes over are not differentiated, and
    under a transform closing over a traced value raises. Reverse mode
    composes twice (`grad(grad(...))`, `hessian` through `jacobian`'s
    reverse mode); forward mode (`jvp`) is not supported. `vmap` batches
    the solve over `x0` and `args` and composes with these derivatives in
    both orders: each example stops at its own convergence test, because
    the batched loop freezes an example's iterate once it has converged or
    stalled, so a batch costs the iterations of its slowest example. The
    solve cannot run inside a `cond`, `fori_loop`, or `scan` body (use
    `fori_loop(..., unroll=True)`).

    An eager solve that does not converge raises `RuntimeError`; with
    `info=True` it returns `(x, info)` instead, where `info` holds
    `"iterations"`, `"residual_norm"` (`||f(x)||`) and `"success"`, per
    example under `vmap`, which is the only report under `jit` and `vmap`.
    """
    if not callable(f):
        raise TypeError("newton requires a callable f(x, *args)")
    check_args(args)
    x0 = x0 if isinstance(x0, (Tensor, TraceTensor)) else asarray(x0)
    if x0.dtype not in (float32, float64):
        raise TypeError(f"newton requires a float32 or float64 x0, got {x0.dtype}")
    shape = list(x0.shape)
    if tol is None:
        tol = math.sqrt(2.0 ** (-23 if x0.dtype == float32 else -52))
    tol = tolerance(tol, "tol")
    maxiter = positive_int(maxiter, "maxiter")
    apply = _flat_function(f, shape, x0.dtype)
    x, stats = _solver(apply, tol, maxiter, ORDER)(x0.reshape([math.prod(shape)]), *args)
    x = x.reshape(shape)
    stats = stats.stop_gradient()
    details = {
        "iterations": stats[0],
        "residual_norm": _norm(asarray(f(x.stop_gradient(), *args))).stop_gradient(),
        "success": stats[1] > 0.5,
    }
    return finish(
        "newton",
        x,
        details,
        info,
        "start closer to a root, increase maxiter, or check that f has a root "
        "with a nonsingular Jacobian",
    )
