"""Implicit-function-theorem derivatives for the iterative solvers
(`linalg.cg`, `linalg.gmres`, and the Newton root finder `_newton.newton`).

A solver iterates in a `while_loop`, which has no reverse mode, and the
derivative of the iterations is not the derivative of the solution anyway:
it depends on where the loop stopped. Each solve is instead wrapped in a
`custom_vjp` whose backward pass applies the implicit function theorem at
the computed solution: for `F(x, theta) = 0`, `theta_bar = -(dF/dtheta)^T
lambda` with `(dF/dx)^T lambda = x_bar`. The backward pass solves that
adjoint system itself, so it costs about one more solve.

A `custom_vjp` backward pass that reads the solution as a residual can be
differentiated again only if the solution's own dependence on the operands
also goes through a rule. `implicit` therefore nests the rule: at order `k`
the forward pass computes the solution with the order `k - 1` function and
the backward pass solves the adjoint system with order `k - 1` solves,
down to the plain iteration at order 0. Rules are built when a call is
staged (there are no lazy rules), so the depth is bounded by `ORDER`:
reverse mode composes `ORDER` times (`grad(grad(...))`), and one more
reverse pass reaches a `while_loop` and raises.
"""

import operator

from . import tree
from ._custom import custom_vjp
from ._quabla import Tensor, TraceTensor, bool_
from ._transforms import vjp

# Reverse-mode passes the solvers support: first derivatives and
# `grad(grad(...))`. Each level traces the solve about twice more when a
# call is staged, and nothing extra runs unless that level is differentiated.
ORDER = 2


def implicit(name, solve, adjoint, order=ORDER):
    """`solve(*operands) -> (x, stats)` with reverse-mode derivatives of
    `order` passes. `adjoint(order, x, operands, x_bar)` returns one
    cotangent per operand (a pytree, `None` for zero), using solves of the
    given order. `stats` is a floating vector of solver statistics, whose
    cotangent is ignored."""
    if order == 0:
        return solve
    inner = implicit(name, solve, adjoint, order - 1)
    solve.__qualname__ = name
    rule = custom_vjp(solve)

    def forward(*operands):
        x, stats = inner(*operands)
        return (x, stats), (x, operands)

    def backward(residuals, cotangents):
        x, operands = residuals
        return tuple(adjoint(order - 1, x, operands, cotangents[0]))

    rule.defvjp(forward, backward)
    return rule


def args_vjp(function, args, cotangent):
    """The cotangent of each array leaf of `args` (a tuple) through
    `function(*args)` for the output cotangent `cotangent`, as a tuple of
    pytrees shaped like `args` with `None` at non-array or bool leaves."""
    leaves, structure = tree.flatten(args)
    positions = [
        position
        for position, leaf in enumerate(leaves)
        if isinstance(leaf, (Tensor, TraceTensor)) and leaf.dtype != bool_
    ]
    if not positions:
        return tuple(None for _ in args)

    def of_leaves(*selected):
        full = list(leaves)
        for position, value in zip(positions, selected):
            full[position] = value
        return function(*structure.unflatten(full))

    _, pullback = vjp(of_leaves, *[leaves[position] for position in positions])
    gradients = pullback(cotangent)
    full = [None] * len(leaves)
    for position, gradient in zip(positions, gradients):
        full[position] = gradient
    return tuple(structure.unflatten(full))


def split_args(args):
    """`(arrays, rebuild)`: the array leaves of the tuple `args`, which a
    loop takes as operands, and `rebuild(*arrays)`, which restores `args`
    with other leaves (Python numbers, static values) unchanged, so a loop
    body closes over those as constants only."""
    leaves, structure = tree.flatten(args)
    positions = [
        position
        for position, leaf in enumerate(leaves)
        if isinstance(leaf, (Tensor, TraceTensor))
    ]

    def rebuild(*arrays):
        full = list(leaves)
        for position, value in zip(positions, arrays):
            full[position] = value
        return tuple(structure.unflatten(full))

    return [leaves[position] for position in positions], rebuild


def transpose(function):
    """`v -> A^T v` for the linear map `A: v -> function(v, *args)`, as
    the VJP of `function` with respect to `v`; linearity makes the point of
    linearization irrelevant, so it is `v` itself. The array leaves of `args`
    are explicit VJP primals rather than closed over, which a loop body
    requires of its operands."""

    def transposed(v, *args):
        arrays, rebuild = split_args(args)
        _, pullback = vjp(lambda u, *values: function(u, *rebuild(*values)), v, *arrays)
        return pullback(v)[0]

    return transposed


def positive_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def tolerance(value, name):
    value = float(value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be a non-negative number, got {value}")
    return value


def check_args(args):
    if not isinstance(args, tuple):
        raise TypeError(f"args must be a tuple, got {type(args).__name__}")


def finish(name, x, info, want_info, advice):
    """`x`, or `(x, info)` when `want_info`. An eager solve that did not
    converge raises `RuntimeError` unless `want_info`, which leaves the check
    to the caller; a traced one cannot raise, so `info["success"]` is the
    only report."""
    if want_info:
        return x, info
    success = info["success"]
    if not isinstance(success, TraceTensor) and not bool(success):
        raise RuntimeError(
            f"{name} did not converge after {int(info['iterations'].item())} "
            f"iterations (residual norm {info['residual_norm'].item():.3e}); {advice}, "
            "or pass info=True to inspect the result"
        )
    return x
