"""Matrix-free Krylov solvers behind `linalg.cg` and `linalg.gmres`.

Both iterate in a `while_loop`, so an eager solve runs in Python and a
traced one is a single loop region whose trip count is the actual number of
iterations (a bounded `fori_loop` would pay for `maxiter` iterations on
every call, and `maxiter` is a safety cap far above the typical count).
`while_loop` has no reverse mode, which costs nothing here: derivatives
come from the implicit rule of `_implicit`, never from the iterations.

Internally every vector is flat (`[n]`); the public functions reshape to
the shape of `b` around the user's `matvec` and preconditioner. The loop
carry is a single array, so the solver state is packed into one vector and
sliced apart in each iteration, as in `ode.py`.
"""

import math

from . import _quabla
from ._array import arange, asarray, eye, zeros
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
    transpose,
)
from ._quabla import Tensor, TraceTensor, float32, float64

__all__ = ["cg", "gmres"]


def _flat_operator(function, shape, dtype, name):
    """`function` on flat vectors: `v [n] -> function(v.reshape(shape),
    *args).reshape([n])`, checking that the result keeps the shape and dtype
    of the right-hand side."""
    size = math.prod(shape)

    def apply(v, *args):
        out = function(v.reshape(list(shape)), *args)
        if not isinstance(out, (Tensor, TraceTensor)):
            out = asarray(out, dtype=dtype)
        if list(out.shape) != list(shape):
            raise ValueError(
                f"{name} must return an array of the right-hand side's shape "
                f"{list(shape)}, got {list(out.shape)}"
            )
        if out.dtype != dtype:
            raise TypeError(
                f"{name} must return the right-hand side's dtype {dtype}, got {out.dtype}; "
                "pass args of that dtype or cast the result"
            )
        return out.reshape([size])

    return apply


def _identity(v, *args):
    return v


def _dot(u, v):
    return (u * v).sum()


def _norm(v):
    """The 2-norm as `sqrt(v . v)`. `Tensor.norm` rescales by `max|v|`,
    which avoids overflow but lowers its maximum to a chain of `n` scalar
    comparisons, a cost every loop iteration would pay; the squared norm
    only overflows beyond about 1e154 in float64 and 1e19 in float32, where
    the CG and GMRES inner products overflow anyway."""
    return _dot(v, v).sqrt()


def _linear_adjoint(build, apply, precondition, symmetric):
    """The implicit rule of `x = A(args)^-1 b`: `lambda = A^-T x_bar` solved
    by `build(operator, preconditioner, order)` (with `A` itself when it is
    symmetric), `b_bar = lambda`, no cotangent for the initial guess, and
    `args_bar = -(d(A x)/d args)^T lambda`."""

    def adjoint(order, x, operands, x_bar):
        _, _, *args = operands
        if symmetric:
            transposed = build(apply, precondition, order)
        else:
            transposed = build(
                transpose(apply),
                precondition if precondition is _identity else transpose(precondition),
                order,
            )
        lam, _ = transposed(x_bar, zeros(list(x_bar.shape), x_bar.dtype), *args)
        args_bar = args_vjp(lambda *values: apply(x, *values), tuple(args), -lam)
        return (lam, None, *args_bar)

    return adjoint


def _threshold(b, tol, atol):
    """The residual norm at which a solve stops: `max(tol * ||b||, atol)`."""
    return (_norm(b) * tol).maximum(atol)


# -- conjugate gradients ----------------------------------------------------------


def _cg_solve(apply, precondition, tol, atol, maxiter):
    """Preconditioned conjugate gradients (Hestenes-Stiefel; Saad, Iterative
    Methods for Sparse Linear Systems, Algorithm 9.1). The carry is
    `[x, r, p, r.z, k]`. The loop runs while `||r|| > threshold`, a
    comparison that is false for NaN, so a breakdown (an operator that is
    not positive definite) stops the loop and reports failure instead of
    running to `maxiter`."""

    def solve(b, x0, *args):
        n = b.shape[0]
        arrays, rebuild = split_args(args)
        r = b - apply(x0, *args)
        z = precondition(r, *args)
        threshold = _threshold(b, tol, atol)
        carry = _quabla.concat(
            _bindings((x0, r, z, _dot(r, z).reshape([1]), zeros([1], b.dtype))), 0
        )

        def unpack(carry):
            return carry[:n], carry[n : 2 * n], carry[2 * n : 3 * n], carry[3 * n], carry[3 * n + 1]

        def running(carry, threshold, *arrays):
            _, r, _, _, k = unpack(carry)
            return (_norm(r) > threshold) & (k < maxiter)

        def step(carry, threshold, *arrays):
            args = rebuild(*arrays)
            x, r, p, rz, k = unpack(carry)
            ap = apply(p, *args)
            alpha = rz / _dot(p, ap)
            x = x + alpha * p
            r = r - alpha * ap
            z = precondition(r, *args)
            rz_next = _dot(r, z)
            p = z + (rz_next / rz) * p
            return _quabla.concat(
                [x, r, p, rz_next.reshape([1]), (k + 1.0).reshape([1])], 0
            )

        carry = while_loop(running, step, carry, operands=(threshold, *arrays))
        x, r, _, _, k = unpack(carry)
        converged = (_norm(r) <= threshold).astype(b.dtype)
        return x, _quabla.concat(_bindings((k.reshape([1]), converged.reshape([1]))), 0)

    return solve


def _cg(apply, precondition, tol, atol, maxiter, order):
    def build(apply, precondition, order):
        return _cg(apply, precondition, tol, atol, maxiter, order)

    solve = _cg_solve(apply, precondition, tol, atol, maxiter)
    return implicit("cg", solve, _linear_adjoint(build, apply, precondition, True), order)


# -- GMRES ------------------------------------------------------------------------


def _gmres_solve(apply, precondition, tol, atol, restart, maxiter):
    """Restarted GMRES(m) with right preconditioning (Saad, Algorithm 9.5):
    each cycle builds an Arnoldi basis `V` of `K_m(A M, r)`, reduces the
    Hessenberg matrix to an upper-triangular `R` with Givens rotations, and
    updates `x += M V y` with `R y = g`. Right preconditioning makes the
    minimized residual the true residual `b - A x`, so the stopping test does
    not depend on `M`.

    Orthogonalization is classical Gram-Schmidt applied twice (CGS2), which
    is as stable as modified Gram-Schmidt (Giraud, Langou & Rozloznik, 2005)
    and is two dense products per step instead of a loop of `j` dependent
    dot products, so it suits a traced step whose `j` is an array. The rows
    of `V` beyond step `j` are zero, so `V w` needs no mask. The Givens
    rotations accumulate into an orthogonal `Q` (`R = Q H`), so applying
    them to a new column is one product and the residual norm of the
    least-squares problem is `beta |Q[j + 1, 0]|`.
    """

    def solve(b, x0, *args):
        n = b.shape[0]
        m = min(restart, n)
        dtype = b.dtype
        arrays, rebuild = split_args(args)
        threshold = _threshold(b, tol, atol)
        rows = arange(m + 1, dtype=dtype)
        columns = arange(m, dtype=dtype)
        basis_size, triangle_size, rotation_size = (m + 1) * n, m * m, (m + 1) * (m + 1)

        def unpack_cycle(carry):
            q_end = basis_size + triangle_size + rotation_size
            return (
                carry[:basis_size].reshape([m + 1, n]),
                carry[basis_size : basis_size + triangle_size].reshape([m, m]),
                carry[basis_size + triangle_size : q_end].reshape([m + 1, m + 1]),
                carry[q_end],
                carry[q_end + 1],
            )

        def arnoldi_running(carry, beta, threshold, *arrays):
            *_, j, estimate = unpack_cycle(carry)
            return (j < m) & (estimate > threshold)

        def arnoldi_step(carry, beta, threshold, *arrays):
            args = rebuild(*arrays)
            basis, triangle, rotation, j, _ = unpack_cycle(carry)
            current = _quabla.equal(rows, j).astype(dtype)
            following = _quabla.equal(rows, j + 1.0).astype(dtype)
            v = (current.reshape([1, m + 1]) @ basis).reshape([n])
            w = apply(precondition(v, *args), *args)
            h = zeros([m + 1], dtype)
            for _ in range(2):
                projection = (basis @ w.reshape([n, 1])).reshape([m + 1])
                w = w - (projection.reshape([1, m + 1]) @ basis).reshape([n])
                h = h + projection
            norm = _norm(w)
            # A zero `w` is a lucky breakdown: the solution lies in the
            # current space, the residual estimate below becomes zero, and
            # the next basis vector is never used.
            positive = norm > 0.0
            following_vector = _quabla.where(positive, w / _quabla.where(positive, norm, 1.0), 0.0)
            column = rotation @ (h + norm * following).reshape([m + 1, 1])
            column = column.reshape([m + 1])
            # The new rotation zeroes entry j + 1 (still `norm`: the earlier
            # rotations act on rows up to j only) against entry j, with the
            # hypotenuse scaled to avoid overflow.
            a = _dot(current, column)
            scale = a.abs().maximum(norm)
            nonzero = scale > 0.0
            safe = _quabla.where(nonzero, scale, 1.0)
            rho = scale * ((a / safe) * (a / safe) + (norm / safe) * (norm / safe)).sqrt()
            rho_safe = _quabla.where(nonzero, rho, 1.0)
            cosine = _quabla.where(nonzero, a / rho_safe, 1.0)
            sine = norm / rho_safe
            row_j = (current.reshape([1, m + 1]) @ rotation).reshape([m + 1])
            row_next = (following.reshape([1, m + 1]) @ rotation).reshape([m + 1])
            new_j = cosine * row_j + sine * row_next
            new_next = cosine * row_next - sine * row_j
            rotation = (
                rotation
                + current.reshape([m + 1, 1]) * (new_j - row_j).reshape([1, m + 1])
                + following.reshape([m + 1, 1]) * (new_next - row_next).reshape([1, m + 1])
            )
            r_column = column * (1.0 - current - following) + rho * current
            triangle = triangle + r_column[:m].reshape([m, 1]) * _quabla.equal(
                columns, j
            ).astype(dtype).reshape([1, m])
            basis = basis + following.reshape([m + 1, 1]) * following_vector.reshape([1, n])
            estimate = beta * new_next[0].abs()
            return _quabla.concat(
                [
                    basis.reshape([basis_size]),
                    triangle.reshape([triangle_size]),
                    rotation.reshape([rotation_size]),
                    (j + 1.0).reshape([1]),
                    estimate.reshape([1]),
                ],
                0,
            )

        def unpack(carry):
            return carry[:n], carry[n : 2 * n], carry[2 * n], carry[2 * n + 1]

        def running(carry, b, threshold, *arrays):
            _, r, cycles, _ = unpack(carry)
            return (_norm(r) > threshold) & (cycles < maxiter)

        def cycle(carry, b, threshold, *arrays):
            args = rebuild(*arrays)
            x, r, cycles, iterations = unpack(carry)
            beta = _norm(r)
            start = _quabla.concat(
                _bindings(
                    (
                        (r / beta).reshape([n]),
                        zeros([m * n + triangle_size], dtype),
                        eye(m + 1, dtype=dtype).reshape([rotation_size]),
                        zeros([1], dtype),
                        beta.reshape([1]),
                    )
                ),
                0,
            )
            state = while_loop(
                arnoldi_running, arnoldi_step, start, operands=(beta, threshold, *arrays)
            )
            basis, triangle, rotation, steps, _ = unpack_cycle(state)
            # Columns beyond the steps taken are zero; a unit diagonal and a
            # zero right-hand side there make their coefficients zero.
            used = _quabla.less(columns, steps).astype(dtype)
            triangle = triangle + eye(m, dtype=dtype) * (1.0 - used).reshape([1, m])
            g = beta * rotation[:m, 0] * used
            y = triangle.solve_triangular(g.reshape([m, 1]), False, False)
            update = (y.reshape([1, m]) @ basis[:m]).reshape([n])
            x = x + precondition(update, *args)
            r = b - apply(x, *args)
            return _quabla.concat(
                [x, r, (cycles + 1.0).reshape([1]), (iterations + steps).reshape([1])], 0
            )

        r = b - apply(x0, *args)
        carry = _quabla.concat(_bindings((x0, r, zeros([2], dtype))), 0)
        carry = while_loop(running, cycle, carry, operands=(b, threshold, *arrays))
        x, r, _, iterations = unpack(carry)
        converged = (_norm(r) <= threshold).astype(dtype)
        return x, _quabla.concat(_bindings((iterations.reshape([1]), converged.reshape([1]))), 0)

    return solve


def _gmres(apply, precondition, tol, atol, restart, maxiter, order):
    def build(apply, precondition, order):
        return _gmres(apply, precondition, tol, atol, restart, maxiter, order)

    solve = _gmres_solve(apply, precondition, tol, atol, restart, maxiter)
    return implicit("gmres", solve, _linear_adjoint(build, apply, precondition, False), order)


# -- public functions -------------------------------------------------------------


def _setup(name, matvec, b, x0, args, tol, atol, M):
    if not callable(matvec):
        raise TypeError(f"{name} requires a callable matvec(x, *args)")
    if M is not None and not callable(M):
        raise TypeError(f"{name} requires M to be a callable M(x, *args) or None")
    check_args(args)
    b = b if isinstance(b, (Tensor, TraceTensor)) else asarray(b)
    if b.dtype not in (float32, float64):
        raise TypeError(f"{name} requires a float32 or float64 right-hand side, got {b.dtype}")
    shape = list(b.shape)
    size = math.prod(shape)
    if x0 is None:
        x0 = zeros(shape, b.dtype)
    else:
        x0 = asarray(x0, dtype=b.dtype)
        if list(x0.shape) != shape:
            raise ValueError(
                f"{name} requires x0 of the right-hand side's shape {shape}, got {list(x0.shape)}"
            )
    apply = _flat_operator(matvec, shape, b.dtype, "matvec")
    precondition = _identity if M is None else _flat_operator(M, shape, b.dtype, "M")
    return (
        apply,
        precondition,
        b.reshape([size]),
        x0.reshape([size]),
        shape,
        tolerance(tol, "tol"),
        tolerance(atol, "atol"),
    )


def _result(name, matvec, b, x, stats, args, info, advice):
    x = x.reshape(list(b.shape))
    stats = stats.stop_gradient()
    residual = b - asarray(matvec(x.stop_gradient(), *args))
    details = {
        "iterations": stats[0],
        "residual_norm": _norm(residual).stop_gradient(),
        "success": stats[1] > 0.5,
    }
    return finish(name, x, details, info, advice)


def cg(matvec, b, *, args=(), x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None, info=False):
    """The solution `x` of `A x = b` for a symmetric positive-definite
    operator `A`, by (preconditioned) conjugate gradients.

    `A` is matrix-free: `matvec(x, *args)` returns `A x` for `x` shaped like
    `b` (any shape; the solve treats it as a flat vector). `M`, if given, is
    a symmetric positive-definite preconditioner `M(x, *args)` approximating
    `A^-1 x`, such as the Jacobi preconditioner `x / diag(A)`. Iteration
    stops when `||b - A x|| <= max(tol * ||b||, atol)` (the recursively
    updated residual, as in SciPy) or after `maxiter` iterations (default
    `10 * b.size`); a breakdown such as an indefinite `A` also stops it.

    Derivatives with respect to `b` and the array leaves of `args` come from
    the implicit function theorem at the solution, not from the iterations:
    the gradient solves `A lambda = x_bar` with the same method, so it costs
    about one more solve, and `x0` receives no gradient. Pass every array
    `A` depends on through `args` (a tuple of arrays or pytrees): values a
    Python `matvec` closes over are not differentiated, and under a
    transform closing over a traced value raises. Reverse mode composes
    twice (`grad(grad(...))`); forward mode (`jvp`, `jacfwd`, `hessian`),
    `vmap`, and reverse `jacobian` of the solution are not supported,
    because the loop has no batching rule. The solve cannot run inside a
    `cond`, `fori_loop`, or `scan` body (use `fori_loop(..., unroll=True)`).

    An eager solve that does not converge raises `RuntimeError`; with
    `info=True` it returns `(x, info)` instead, where `info` holds
    `"iterations"`, `"residual_norm"` (the true `||b - A x||`) and
    `"success"`, which is the only report under `jit`.
    """
    apply, precondition, flat_b, flat_x0, shape, tol, atol = _setup(
        "cg", matvec, b, x0, args, tol, atol, M
    )
    maxiter = 10 * flat_b.shape[0] if maxiter is None else positive_int(maxiter, "maxiter")
    solve = _cg(apply, precondition, tol, atol, maxiter, ORDER)
    x, stats = solve(flat_b, flat_x0, *args)
    return _result(
        "cg", matvec, flat_b.reshape(shape), x, stats, args, info,
        "increase maxiter, add a preconditioner M, or check that A is positive definite",
    )


def gmres(
    matvec, b, *, args=(), x0=None, tol=1e-5, atol=0.0, restart=20, maxiter=None, M=None, info=False
):
    """The solution `x` of `A x = b` for a general nonsingular operator `A`,
    by restarted GMRES with right preconditioning.

    `matvec(x, *args)` returns `A x` for `x` shaped like `b`; `M(x, *args)`,
    if given, approximates `A^-1 x`. Each cycle builds a Krylov basis of up
    to `restart` vectors (at most `b.size`), orthogonalized by classical
    Gram-Schmidt applied twice, and minimizes the residual over it with
    Givens rotations; the next cycle restarts from the new residual. A
    smaller `restart` uses less memory (`restart * b.size` for the basis)
    but may converge more slowly or stagnate. Iteration stops when the true
    residual `||b - A x|| <= max(tol * ||b||, atol)` or after `maxiter`
    cycles (default `10 * b.size`).

    Derivatives follow the implicit function theorem as in `cg`: the
    gradient solves `A^T lambda = x_bar` by GMRES, applying `A^T` (and
    `M^T`) as the VJP of `matvec` (and `M`) with respect to `x`. The same
    rules apply to `args`, `info`, and the supported transforms as in `cg`;
    `info["iterations"]` counts Arnoldi steps over all cycles.
    """
    apply, precondition, flat_b, flat_x0, shape, tol, atol = _setup(
        "gmres", matvec, b, x0, args, tol, atol, M
    )
    restart = positive_int(restart, "restart")
    maxiter = 10 * flat_b.shape[0] if maxiter is None else positive_int(maxiter, "maxiter")
    solve = _gmres(apply, precondition, tol, atol, restart, maxiter, ORDER)
    x, stats = solve(flat_b, flat_x0, *args)
    return _result(
        "gmres", matvec, flat_b.reshape(shape), x, stats, args, info,
        "increase maxiter or restart, or add a preconditioner M",
    )
