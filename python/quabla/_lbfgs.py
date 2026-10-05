"""Host-side full-batch L-BFGS over one jitted value-and-gradient function.

The iterate is one float64 host vector holding every parameter leaf; the loss
sees the leaves in their own dtypes. The algorithm is Nocedal and Wright,
Numerical Optimization (2nd ed.): the two-loop recursion (Algorithm 7.4) with
the scaling ``H0 = (s'y / y'y) I`` (eq. 7.20), and a strong Wolfe line search
(Algorithms 3.5 and 3.6) whose trial steps come from safeguarded cubic
interpolation (eq. 3.59).
"""

import math as _math
import operator as _operator
from collections import deque as _deque

from . import tree as _tree
from ._quabla import Tensor as _Tensor
from ._quabla import float64 as _float64
from ._transforms import jit as _jit
from ._transforms import value_and_grad as _value_and_grad

# Wolfe constants recommended for quasi-Newton methods (N&W section 3.1).
_C1 = 1e-4
_C2 = 0.9
# Function evaluations one line search may spend (bracketing plus zoom).
_LINE_SEARCH_EVALUATIONS = 25
# A curvature pair is stored only when cos(s, y) exceeds this bound, which
# keeps rho = 1 / s'y bounded and the inverse Hessian approximation positive
# definite; pairs from nearly flat or nonconvex steps are skipped.
_CURVATURE_EPS = 1e-10
# Zoom interpolation keeps trials this fraction of the bracket away from its
# ends, so every evaluation shrinks the bracket.
_SAFEGUARD = 0.1


def _dot(a, b):
    return float((a * b).sum())


def _max_abs(vector):
    # Python abs() normalizes a -0.0 maximum to 0.0 for reporting.
    return abs(float(vector.abs().max()))


def _finite(*values):
    return all(_math.isfinite(value) for value in values)


class _Layout:
    """Map a floating Tensor pytree to and from one float64 host vector."""

    def __init__(self, params):
        leaves, self.definition = _tree.flatten(params)
        if not leaves:
            raise ValueError("L-BFGS requires at least one parameter")
        for leaf in leaves:
            if not isinstance(leaf, _Tensor) or leaf.dtype.name not in (
                "float32",
                "float64",
            ):
                raise TypeError("L-BFGS parameters must be floating-point Tensors")
        self.shapes = [list(leaf.shape) for leaf in leaves]
        self.dtypes = [leaf.dtype for leaf in leaves]
        self.sizes = [_math.prod(shape) for shape in self.shapes]
        self.size = sum(self.sizes)
        if self.size == 0:
            raise ValueError("L-BFGS requires at least one parameter element")
        # A float32 leaf rounds the trial point; the stored iterate must be
        # the rounded point so that s pairs with the gradient difference y.
        self.exact = all(dtype == _float64 for dtype in self.dtypes)

    def vector(self, tree, name):
        leaves, definition = _tree.flatten(tree)
        if definition != self.definition:
            raise ValueError(f"{name} pytree structure must match parameters")
        data = []
        for leaf, shape in zip(leaves, self.shapes):
            if list(leaf.shape) != shape:
                raise ValueError(f"{name} shape must match parameter shape")
            data.extend(leaf.to_flat_list())
        return _Tensor([self.size], data, _float64)

    def tree(self, vector):
        leaves, offset = [], 0
        for shape, dtype, size in zip(self.shapes, self.dtypes, self.sizes):
            piece = vector[offset : offset + size]
            leaves.append(piece.reshape(shape).astype(dtype))
            offset += size
        return _tree.unflatten(self.definition, leaves)


def _cubic_minimizer(first, second, lower, upper):
    """Minimizer of the cubic interpolating two (step, value, slope) points.

    N&W eq. 3.59, clamped to ``[lower, upper]``. Falls back to the midpoint
    when an endpoint is not finite or the cubic has no interior minimizer.
    """
    midpoint = 0.5 * (lower + upper)
    a1, f1, g1 = first
    a2, f2, g2 = second
    if not _finite(a1, f1, g1, a2, f2, g2) or a1 == a2:
        return midpoint
    d1 = g1 + g2 - 3 * (f1 - f2) / (a1 - a2)
    discriminant = d1 * d1 - g1 * g2
    if not (_math.isfinite(discriminant) and discriminant >= 0):
        return midpoint
    d2 = _math.copysign(_math.sqrt(discriminant), a2 - a1)
    denominator = g2 - g1 + 2 * d2
    if denominator == 0:
        return midpoint
    step = a2 - (a2 - a1) * ((g2 + d2 - d1) / denominator)
    if not _math.isfinite(step):
        return midpoint
    return min(max(step, lower), upper)


def _zoom(phi, f0, slope0, lo, hi, budget):
    """N&W Algorithm 3.6. ``lo`` satisfies sufficient decrease with the lowest
    value seen; the bracket [lo, hi] contains strong Wolfe points."""
    used = 0
    while used < budget:
        width = abs(hi[0] - lo[0])
        if width <= 1e-12 * max(abs(hi[0]), abs(lo[0])):
            break
        left = min(lo[0], hi[0]) + _SAFEGUARD * width
        right = max(lo[0], hi[0]) - _SAFEGUARD * width
        trial = phi(_cubic_minimizer(lo[:3], hi[:3], left, right))
        used += 1
        step, value, slope = trial[:3]
        if (
            not _finite(value, slope)
            or value > f0 + _C1 * step * slope0
            or value >= lo[1]
        ):
            hi = trial
            continue
        if abs(slope) <= -_C2 * slope0:
            return trial, used
        if slope * (hi[0] - lo[0]) >= 0:
            hi = lo
        lo = trial
    # Budget or bracket exhausted: lo still satisfies sufficient decrease
    # (or is the start point, which the caller treats as a failure).
    return lo, used


def _strong_wolfe(phi, f0, slope0, step, budget):
    """N&W Algorithm 3.5. Returns the accepted point (step, value, slope,
    payload) and the evaluation count; step 0 means no acceptable step."""
    start = (0.0, f0, slope0, None)
    previous = start
    used = 0
    while used < budget:
        current = phi(step)
        used += 1
        value, slope = current[1], current[2]
        # A non-finite loss or slope is treated as a failed sufficient
        # decrease, so the search backtracks toward the last finite point.
        if (
            not _finite(value, slope)
            or value > f0 + _C1 * step * slope0
            or (previous is not start and value >= previous[1])
        ):
            point, more = _zoom(phi, f0, slope0, previous, current, budget - used)
            return point, used + more
        if abs(slope) <= -_C2 * slope0:
            return current, used
        if slope >= 0:
            point, more = _zoom(phi, f0, slope0, current, previous, budget - used)
            return point, used + more
        # Still descending: extrapolate between 1.1x and 10x the step.
        next_step = _cubic_minimizer(previous[:3], current[:3], 1.1 * step, 10 * step)
        previous, step = current, next_step
    return previous, used


def _direction(gradient, pairs):
    """N&W Algorithm 7.4: ``-H g`` from the stored ``(s, y, rho)`` pairs."""
    q = gradient
    alphas = []
    for s, y, rho in reversed(pairs):
        alpha = rho * _dot(s, q)
        alphas.append(alpha)
        q = q - y * alpha
    if pairs:
        s, y, rho = pairs[-1]
        q = q * (1 / (rho * _dot(y, y)))
    for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
        beta = rho * _dot(y, q)
        q = q + s * (alpha - beta)
    return q * -1.0


class LBFGS:
    """Full-batch L-BFGS minimizer for a scalar loss over a parameter pytree.

    ``minimize(fun, params, *args)`` jits ``value_and_grad(fun)`` once on the
    CPU and returns ``(params, info)``. ``info`` holds ``iterations``,
    ``evaluations`` (loss-and-gradient calls), ``loss``, ``grad_norm`` (the
    infinity norm), ``reason``, and ``converged``. ``reason`` is one of
    ``"gradient_tolerance"`` (``grad_norm <= tolerance_grad``),
    ``"change_tolerance"`` (``|f_k - f_k+1| <= tolerance_change *
    max(|f_k|, |f_k+1|)``; unlike scipy's ``ftol`` there is no floor of 1, so
    small PINN losses are not stopped early), ``"max_iterations"``,
    ``"max_evaluations"``, or ``"line_search_failed"`` (no decrease along the
    quasi-Newton direction nor along steepest descent, typical at the
    float32 resolution limit); only the first two set ``converged``.
    """

    def __init__(
        self,
        history=10,
        max_iterations=500,
        max_evaluations=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search="strong_wolfe",
    ):
        self.history = _operator.index(history)
        self.max_iterations = _operator.index(max_iterations)
        if self.history <= 0 or self.max_iterations <= 0:
            raise ValueError("history and max_iterations must be positive")
        if max_evaluations is not None:
            max_evaluations = _operator.index(max_evaluations)
            if max_evaluations <= 0:
                raise ValueError("max_evaluations must be positive")
        self.max_evaluations = max_evaluations
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        if not (
            _finite(self.tolerance_grad, self.tolerance_change)
            and self.tolerance_grad >= 0
            and self.tolerance_change >= 0
        ):
            raise ValueError("tolerances must be finite and nonnegative")
        if line_search != "strong_wolfe":
            raise ValueError('line_search must be "strong_wolfe"')
        self.line_search = line_search

    def minimize(self, fun, params, *args):
        """Minimize ``fun(params, *args)`` over ``params`` from its value."""
        if not callable(fun):
            raise TypeError("fun must be callable")
        layout = _Layout(params)
        value_and_grad = _jit(_value_and_grad(fun), device="cpu")
        evaluations = 0

        def evaluate(x):
            nonlocal evaluations
            evaluations += 1
            point = layout.tree(x)
            loss, gradients = value_and_grad(point, *args)
            if _math.prod(loss.shape) != 1:
                raise ValueError("L-BFGS loss must be a scalar")
            if not layout.exact:
                x = layout.vector(point, "parameter")
            return x, float(loss), layout.vector(gradients, "gradient")

        x, f, g = evaluate(layout.vector(params, "parameter"))
        if not (_math.isfinite(f) and _math.isfinite(_max_abs(g))):
            raise ValueError("L-BFGS initial loss and gradient must be finite")
        pairs = _deque(maxlen=self.history)
        iterations = 0
        while True:
            grad_norm = _max_abs(g)
            if grad_norm <= self.tolerance_grad:
                reason = "gradient_tolerance"
                break
            if iterations >= self.max_iterations:
                reason = "max_iterations"
                break
            budget = _LINE_SEARCH_EVALUATIONS
            if self.max_evaluations is not None:
                budget = min(budget, self.max_evaluations - evaluations)
            if budget <= 0:
                reason = "max_evaluations"
                break
            direction = _direction(g, pairs)
            slope = _dot(g, direction)
            if not (slope < 0 and _math.isfinite(slope)):
                # Roundoff can make -Hg a non-descent direction; restart
                # from steepest descent.
                pairs.clear()
                direction = g * -1.0
                slope = -_dot(g, g)
            # Unit steps suit a scaled quasi-Newton direction; the first
            # steepest-descent step is limited to unit length.
            step = 1.0 if pairs else min(1.0, 1 / _math.sqrt(-slope))

            def phi(alpha, x=x, direction=direction):
                trial, value, gradient = evaluate(x + direction * alpha)
                return alpha, value, _dot(gradient, direction), (trial, gradient)

            point, _ = _strong_wolfe(phi, f, slope, step, budget)
            if point[3] is None:
                if pairs:
                    # Discard curvature information and retry along steepest
                    # descent before giving up (as L-BFGS-B does).
                    pairs.clear()
                    continue
                reason = (
                    "max_evaluations"
                    if self.max_evaluations is not None
                    and evaluations >= self.max_evaluations
                    else "line_search_failed"
                )
                break
            x_new, g_new = point[3]
            f_new = point[1]
            s, y = x_new - x, g_new - g
            sy = _dot(s, y)
            if sy > _CURVATURE_EPS * _math.sqrt(_dot(s, s) * _dot(y, y)):
                pairs.append((s, y, 1 / sy))
            f_old, x, f, g = f, x_new, f_new, g_new
            iterations += 1
            if abs(f_old - f) <= self.tolerance_change * max(abs(f_old), abs(f)):
                grad_norm = _max_abs(g)
                reason = "change_tolerance"
                break
        info = {
            "iterations": iterations,
            "evaluations": evaluations,
            "loss": f,
            "grad_norm": grad_norm,
            "reason": reason,
            "converged": reason in ("gradient_tolerance", "change_tolerance"),
        }
        return layout.tree(x), info
