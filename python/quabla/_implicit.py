"""Implicit-function-theorem derivatives for the iterative solvers
(`linalg.cg`, `linalg.gmres`, and the Newton root finder `newton`).

A solver iterates in a `while_loop`, which has no reverse mode, and the
derivative of the iterations is not the derivative of the solution anyway:
it depends on where the loop stopped. Each solve is instead wrapped in a
`custom_vjp` whose backward pass applies the implicit function theorem at
the computed solution: for `F(x, theta) = 0`, `theta_bar = -(dF/dtheta)^T
lambda` with `(dF/dx)^T lambda = x_bar`. The backward pass solves that
adjoint system itself, so it costs about one more solve.

Forward mode applies the same theorem: the rule also carries a tangent
graph (`custom_vjp._defjvp`), `x_dot = -(dF/dx)^-1 (dF/dtheta) theta_dot`,
which reads the computed solution and solves one more system with
`dF/dx`. Reverse mode keeps using the backward pass, so adding the tangent
changes no reverse-mode result.

A rule that reads the solution (the backward pass keeps it as a residual,
the tangent graph as an input) can be differentiated again only if the
solution's own dependence on the operands also goes through a rule, and
the systems the rule solves must be differentiable the same way.
`implicit` therefore nests the rule: at order `k >= 1` the forward pass
computes the solution with the order `k - 1` function, and the backward
pass and the tangent graph solve with order `k - 1` solves. Order 0 is a
reverse-only rule over the plain iteration: its backward pass solves with
plain iterations and it has no tangent graph, so a derivative of an order
beyond the supported depth raises (forward mode names the rule, reverse
mode reaches a `while_loop`) instead of differentiating the iterations.
Rules are built when a call is staged (there are no lazy rules), so the
depth is bounded by `ORDER`: every combination of `ORDER` forward and
reverse passes (`grad(grad(...))`, `jvp(grad(...))`, `jvp(jvp(...))`, ...)
uses the rules, and so does one more reverse pass.
"""

import operator

from . import tree
from ._custom import custom_vjp
from ._quabla import Tensor, TraceTensor, bool_
from ._transforms import jvp, vjp

# Derivative passes the solvers support: first and second derivatives in
# any combination of forward and reverse mode. Each level traces the solve
# about three times more when a call is staged, and nothing extra runs
# unless that level is differentiated.
ORDER = 2


def implicit(name, solve, adjoint, tangent, order=ORDER):
    """`solve(*operands) -> (x, stats)` with derivatives of `order` passes
    (see the module notes; a negative order is the plain `solve`).
    `adjoint(order, x, operands, x_bar)` returns one cotangent per operand
    (a pytree, `None` for zero) and `tangent(order, x, operands, tangents)`
    the tangent of `x` (`None` for zero) for the tangents of the leaves of
    `operands` (`tree.leaves` order, `None` for a leaf without one), both
    using solves of the given order. `stats` is a floating vector of solver
    statistics, whose cotangent is ignored and whose tangent is zero."""
    if order < 0:
        return solve
    inner = implicit(name, solve, adjoint, tangent, order - 1)
    solve.__qualname__ = name
    if order == 0:

        def beyond(*operands):
            return solve(*operands)

        beyond.__qualname__ = f"{name} beyond its supported derivative order"
        rule = custom_vjp(beyond)
    else:
        rule = custom_vjp(solve)

    def forward(*operands):
        x, stats = inner(*operands)
        return (x, stats), (x, operands)

    def backward(residuals, cotangents):
        x, operands = residuals
        return tuple(adjoint(order - 1, x, operands, cotangents[0]))

    rule.defvjp(forward, backward)
    if order > 0:

        def forward_mode(primals, tangents, outputs):
            return tangent(order - 1, outputs[0], primals, tangents), None

        rule._defjvp(forward_mode)
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


def args_jvp(function, args, tangents):
    """The tangent of `function(*args)` (a tuple `args`) for the tangents of
    the leaves of `args` (`tree.leaves` order, `None` for a leaf without
    one), or `None` when no array leaf has a tangent."""
    leaves, structure = tree.flatten(args)
    positions = [
        position
        for position, leaf in enumerate(leaves)
        if isinstance(leaf, (Tensor, TraceTensor))
        and leaf.dtype != bool_
        and tangents[position] is not None
    ]
    if not positions:
        return None

    def of_leaves(*selected):
        full = list(leaves)
        for position, value in zip(positions, selected):
            full[position] = value
        return function(*structure.unflatten(full))

    _, output_tangent = jvp(
        of_leaves,
        tuple(leaves[position] for position in positions),
        tuple(tangents[position] for position in positions),
    )
    return output_tangent


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
