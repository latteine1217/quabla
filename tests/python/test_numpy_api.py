"""jax.numpy-style compositions: shape functions, products, calculus and
elementwise numerics, and `linalg.norm`/`matrix_power`/`pinv`.

Every function is checked against closed forms (no NumPy needed), against
NumPy in float64 and float32, eagerly and under `jit` and `vmap`, and its
gradient against central differences of the eager float64 function itself.
Random inputs avoid kinks and ties, so a central difference with step 1e-6
is accurate to about 1e-8 relative.
"""

import math
import warnings

try:
    import numpy as np
except ImportError:  # NumPy is optional; the comparisons against it skip.
    np = None

import quabla as qb

from _support import devices, raises, run, skip

INF, NAN = math.inf, math.nan


def flat(x):
    return x.to_flat_list() if isinstance(x, qb.Tensor) else [float(v) for v in x]


def same(actual, expected, tolerance=0.0):
    """`actual` (a Tensor) equals the flat list `expected` within
    `tolerance` relative to max(1, |expected|); NaN and infinities must
    match exactly."""
    values = flat(actual)
    assert len(values) == len(expected), (values, expected)
    for got, want in zip(values, expected):
        if math.isnan(want) or math.isinf(want):
            assert (math.isnan(got) and math.isnan(want)) or got == want, (values, expected)
        else:
            assert abs(got - want) <= tolerance * max(1.0, abs(want)), (values, expected)


def assert_close(actual, expected, tolerance):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    special = ~np.isfinite(expected)
    assert np.array_equal(actual[special], expected[special], equal_nan=True), (actual, expected)
    finite = ~special
    error = np.max(np.abs(actual[finite] - expected[finite]), initial=0.0)
    scale = max(1.0, np.max(np.abs(expected[finite]), initial=0.0))
    assert error <= tolerance * scale, (error, tolerance * scale)


# Closed forms: no NumPy needed.


def test_shape_functions_match_closed_forms():
    x = qb.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    same(qb.flip(x), [6, 5, 4, 3, 2, 1])
    same(qb.flip(x, axis=1), [3, 2, 1, 6, 5, 4])
    same(qb.roll(x, 1), [6, 1, 2, 3, 4, 5])
    same(qb.roll(x, -1, axis=1), [2, 3, 1, 5, 6, 4])
    # Shifts of a repeated axis add up.
    same(qb.roll(x, (1, 1), axis=(1, 1)), [2, 3, 1, 5, 6, 4])
    v = qb.asarray([1.0, 2.0, 3.0])
    same(qb.pad(v, 2), [0, 0, 1, 2, 3, 0, 0])
    same(qb.pad(v, (1, 2), constant_values=(-1, 9)), [-1, 1, 2, 3, 9, 9])
    same(qb.pad(v, 2, mode="edge"), [1, 1, 1, 2, 3, 3, 3])
    same(qb.pad(v, 4, mode="reflect"), [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
    same(qb.pad(v, 2, mode="symmetric"), [2, 1, 1, 2, 3, 3, 2])
    same(qb.pad(v, 4, mode="wrap"), [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    assert qb.pad(x, ((0, 1), (2, 0))).shape == [3, 5]
    same(qb.tile(v, 2), [1, 2, 3, 1, 2, 3])
    assert qb.tile(v, (2, 1)).shape == [2, 3]
    same(qb.repeat(v, 2), [1, 1, 2, 2, 3, 3])
    same(qb.repeat(x, [2, 0], axis=0), [1, 2, 3, 1, 2, 3])
    same(qb.moveaxis(x, 0, -1), [1, 4, 2, 5, 3, 6])
    same(qb.swapaxes(x, 0, 1), [1, 4, 2, 5, 3, 6])
    assert qb.ravel(x).shape == [6] and qb.ravel(qb.asarray(2.0)).shape == [1]
    same(qb.diagonal(x), [1, 5])
    same(qb.diagonal(x, 1), [2, 6])
    same(qb.diagonal(x, -1), [4])
    assert qb.trace(x).item() == 6.0 and qb.trace(x, offset=2).item() == 3.0
    same(qb.diag(qb.asarray([1.0, 2.0])), [1, 0, 0, 2])
    same(qb.diag(qb.asarray([1.0, 2.0]), 1), [0, 1, 0, 0, 0, 2, 0, 0, 0])
    same(qb.diag(qb.asarray([1.0, 2.0]), -1), [0, 0, 0, 1, 0, 0, 0, 2, 0])
    same(qb.diag(x, 1), [2, 6])
    # An infinite entry stays on the diagonal (a product with the identity
    # would put NaN off it).
    same(qb.diag(qb.asarray([INF, 1.0])), [INF, 0, 0, 1])


def test_products_match_closed_forms():
    a = qb.asarray([1.0, 2.0])
    b = qb.asarray([3.0, 4.0, 5.0])
    same(qb.outer(a, b), [3, 4, 5, 6, 8, 10])
    same(qb.kron(a, b), [3, 4, 5, 6, 8, 10])
    same(qb.kron(qb.eye(2), qb.asarray([[1.0, 2.0]])), [1, 2, 0, 0, 0, 0, 1, 2])
    assert qb.dot(a, a).item() == 5.0
    same(qb.dot(qb.asarray(2.0), a), [2, 4])
    m = qb.asarray([[1.0, 2.0], [3.0, 4.0]])
    same(qb.dot(m, a), [5, 11])
    same(qb.tensordot(m, m, 2), [30])
    same(qb.tensordot(m, m, ([0], [0])), [10, 14, 14, 20])
    ex, ey, ez = qb.eye(3)[0], qb.eye(3)[1], qb.eye(3)[2]
    same(qb.cross(ex, ey), flat(ez))
    same(qb.cross(ey, ex), [0, 0, -1])
    same(qb.cross(qb.asarray([1.0, 0.0]), qb.asarray([0.0, 1.0])), [1])
    same(qb.cross(qb.asarray([1.0, 0.0]), ez), [0, -1, 0])
    same(qb.cross(qb.stack([ex, ey], 1), qb.stack([ey, ez], 1), axis=0), [0, 1, 0, 0, 1, 0])


def test_calculus_matches_closed_forms():
    t = qb.linspace(0.0, 2.0, 5)
    cubic = t * t * t
    # Third differences of a cubic sampled at spacing h are 6 h^3.
    same(qb.diff(cubic, 3), [6 * 0.5**3] * 2, 1e-14)
    same(qb.diff(qb.asarray([1.0, 4.0]), prepend=0.0, append=2.0), [1, 3, -2])
    same(qb.diff(qb.asarray([True, True, False])), [0, 1])
    # The trapezoidal rule is exact for a linear integrand.
    same(qb.trapezoid(3.0 * t + 1.0, t), [8.0], 1e-15)
    same(qb.trapezoid(qb.ones([4]), dx=0.5), [1.5])
    same(qb.trapezoid(qb.asarray([[2.0, 3.0]]), axis=0), [0, 0])
    xp = qb.asarray([0.0, 1.0, 3.0])
    fp = qb.asarray([1.0, 3.0, -1.0])
    query = qb.asarray([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, INF, -INF, NAN])
    same(qb.interp(query, xp, fp), [1, 1, 2, 3, 1, -1, -1, -1, 1, NAN])
    same(qb.interp(query, xp, fp, left=-5.0, right=7.0), [-5, 1, 2, 3, 1, -1, 7, 7, -5, NAN])
    same(qb.interp(query, qb.asarray([1.0]), qb.asarray([2.0])), [2, 2, 2, 2, 2, 2, 2, 2, 2, NAN])
    # 2 x^2 - 3 x + 1, from a list of numbers and from an array.
    same(qb.polyval([2.0, -3.0, 1.0], qb.asarray([0.0, 1.0, 2.0])), [1, 0, 3])
    same(qb.polyval(qb.asarray([2.0, -3.0, 1.0]), qb.asarray([-1.0, INF])), [6, INF])
    same(qb.polyval([4.0], qb.asarray([1.0, 2.0])), [4, 4])


def test_elementwise_numerics_match_closed_forms():
    log2 = math.log(2.0)
    same(qb.logaddexp(0.0, 0.0), [log2], 1e-16)
    same(qb.logaddexp(qb.asarray([1000.0, -1000.0]), 1000.0), [1000 + log2, 1000.0], 1e-16)
    # Far apart: the smaller term vanishes without cancellation.
    same(qb.logaddexp(qb.asarray([1.0]), qb.asarray([-1e20])), [1.0])
    same(
        qb.logaddexp(
            qb.asarray([-INF, INF, -INF, NAN, INF]), qb.asarray([-INF, INF, 2.0, 0.0, -INF])
        ),
        [-INF, INF, 2.0, NAN, INF],
    )
    same(qb.hypot(3.0, qb.asarray([4.0, -4.0, 0.0])), [5, 5, 3], 1e-16)
    same(
        qb.hypot(qb.asarray([0.0, INF, NAN, NAN]), qb.asarray([0.0, NAN, -INF, 1.0])),
        [0, INF, INF, NAN],
    )
    # Exact powers of two, including the subnormal and the largest one.
    same(
        qb.exp2(qb.asarray([-1074.0, -1.0, 0.0, 10.0, 1023.0])),
        [2.0**-1074, 0.5, 1, 1024, 2.0**1023],
    )
    f32 = qb.exp2(qb.asarray([-149.0, 127.0], dtype=qb.float32))
    assert f32.dtype == qb.float32
    same(f32, [2.0**-149, 2.0**127])
    special = qb.asarray([1.0, INF, -INF, NAN, -0.0])
    assert qb.isinf(special).tolist() == [False, True, True, False, False]
    same(qb.nan_to_num(special), [1, 1.7976931348623157e308, -1.7976931348623157e308, 0, 0])
    same(qb.nan_to_num(special, nan=-1.0, posinf=2.0, neginf=-2.0), [1, 2, -2, -1, 0])
    narrow = qb.nan_to_num(special.astype(qb.float32))
    assert narrow.dtype == qb.float32
    same(narrow, [1, 3.4028234663852886e38, -3.4028234663852886e38, 0, 0])


def test_float32_hypot_and_norm_do_not_overflow_or_underflow():
    for scale in (1e30, 1e-30, 1e-42):
        a = qb.asarray([3.0 * scale], dtype=qb.float32)
        b = qb.asarray([4.0 * scale], dtype=qb.float32)
        for result in (qb.hypot(a, b), qb.jit(qb.hypot)(a, b)):
            assert result.dtype == qb.float32
            same(result, [5.0 * scale], 2e-7 if scale > 1e-40 else 0.3)
        gradient = qb.grad(lambda u, v: qb.hypot(u, v).sum(), argnums=(0, 1))(a, b)
        same(gradient[0], [0.6], 1e-6 if scale > 1e-40 else 0.3)
        same(gradient[1], [0.8], 1e-6 if scale > 1e-40 else 0.3)
    big = qb.asarray([[3e30, 4e30], [0.0, 1e30]], dtype=qb.float32)
    same(qb.linalg.norm(big[0]), [5e30], 2e-7)
    same(qb.linalg.norm(big, axis=1), [5e30, 1e30], 2e-7)
    same(qb.linalg.norm(big, ord=3, axis=1), [(27 + 64) ** (1 / 3) * 1e30, 1e30], 1e-6)
    same(qb.linalg.norm(big, ord="fro"), [math.sqrt(26) * 1e30], 2e-7)
    same(qb.linalg.norm(big * 1e-60, ord=3, axis=1), [(27 + 64) ** (1 / 3) * 1e-30, 1e-30], 1e-6)
    same(
        qb.linalg.norm(big, ord=-1.5, axis=1),
        [((3**-1.5 + 4**-1.5) ** (1 / -1.5)) * 1e30, 0.0],
        1e-6,
    )


def test_linalg_matches_closed_forms():
    v = qb.asarray([3.0, -4.0, 0.0])
    for order, expected in (
        (None, 5),
        (2, 5),
        (1, 7),
        (INF, 4),
        (-INF, 0),
        (0, 2),
        (3, 91 ** (1 / 3)),
    ):
        same(qb.linalg.norm(v, order), [expected], 1e-15)
    same(qb.linalg.norm(qb.asarray([INF, 1.0]), 3), [INF])
    same(qb.linalg.norm(qb.asarray([NAN, 1.0]), 3), [NAN])
    m = qb.asarray([[1.0, -2.0], [3.0, 4.0]])
    # The singular values of m are sqrt(15 +- sqrt(125)), from m^T m.
    largest, smallest = math.sqrt(15 + math.sqrt(125)), math.sqrt(15 - math.sqrt(125))
    for order, expected in (
        ("fro", math.sqrt(30)),
        (1, 6),
        (-1, 4),
        (INF, 7),
        (-INF, 3),
        (2, largest),
        (-2, smallest),
        ("nuc", largest + smallest),
    ):
        same(qb.linalg.norm(m, order), [expected], 1e-14)
    assert qb.linalg.norm(m, keepdims=True).shape == [1, 1]
    assert qb.linalg.norm(m, 1, axis=1, keepdims=True).shape == [2, 1]
    # A rotation by pi/3 to the sixth power is the identity.
    c, s = math.cos(math.pi / 3), math.sin(math.pi / 3)
    rotation = qb.asarray([[c, -s], [s, c]])
    same(qb.linalg.matrix_power(rotation, 6), [1, 0, 0, 1], 1e-14)
    same(qb.linalg.matrix_power(rotation, -1), [c, s, -s, c], 1e-14)
    same(qb.linalg.matrix_power(rotation, 0), [1, 0, 0, 1])
    same(qb.linalg.matrix_power(m, 3), flat(m @ m @ m), 1e-14)
    same(
        qb.linalg.pinv(qb.asarray([[2.0, 0.0], [0.0, 0.0], [0.0, 4.0]])),
        [0.5, 0, 0, 0, 0, 0.25],
        1e-15,
    )
    # A singular value at the cutoff is dropped: the pseudo-inverse of a
    # rank-one matrix is A^T / ||A||_F^2.
    rank_one = qb.asarray([[1.0, 2.0], [2.0, 4.0]])
    same(qb.linalg.pinv(rank_one), flat(rank_one / 25.0), 1e-14)


def test_dtypes_are_kept():
    for dtype in (qb.float32, qb.float64):
        x = qb.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        results = (
            qb.flip(x),
            qb.roll(x, 1),
            qb.pad(x, 1),
            qb.pad(x, 1, "reflect"),
            qb.tile(x, 2),
            qb.repeat(x, 2),
            qb.moveaxis(x, 0, 1),
            qb.ravel(x),
            qb.diag(x[0]),
            qb.trace(x),
            qb.outer(x, x),
            qb.dot(x, x),
            qb.tensordot(x, x, 1),
            qb.kron(x, x),
            qb.cross(x, x),
            qb.diff(x),
            qb.trapezoid(x),
            qb.interp(x, x[0], x[1]),
            qb.polyval([1.0, 2.0], x),
            qb.logaddexp(x, 1.0),
            qb.hypot(1.0, x),
            qb.exp2(x),
            qb.nan_to_num(x),
            qb.linalg.norm(x, 3, axis=0),
            qb.linalg.matrix_power(x, 2),
            qb.linalg.pinv(x),
        )
        assert all(result.dtype == dtype for result in results)


def test_unsupported_arguments_raise_clear_errors():
    x = qb.ones([2, 3])
    raises(ValueError, qb.pad, x, 1, mode="mean", match="mode must be one of")
    raises(ValueError, qb.pad, x, -1, match="non-negative")
    raises(TypeError, qb.pad, x, 1.5, match="pad_width must be an int")
    raises(TypeError, qb.repeat, x, qb.asarray(2.0), match="no integer arrays")
    raises(ValueError, qb.repeat, x, 0, match="positive")
    raises(ValueError, qb.repeat, x, [1, 2], axis=1, match="2 entries")
    raises(ValueError, qb.tile, x, (0, 1), match="positive")
    raises(ValueError, qb.diagonal, x, 3, match="empty diagonal")
    raises(ValueError, qb.diagonal, x, 0, 1, 1, match="cannot be the same")
    raises(ValueError, qb.diag, qb.ones([2, 2, 2]), match="1-D or 2-D")
    raises(ValueError, qb.tensordot, x, x, 1, match="shape mismatch")
    raises(ValueError, qb.tensordot, x, x, 3, match="out of range")
    raises(
        ValueError,
        qb.cross,
        qb.ones([4, 3]),
        qb.ones([4, 3]),
        axis=0,
        match="dimension must be 2 or 3",
    )
    raises(ValueError, qb.diff, x, 3, match="would be empty")
    raises(
        ValueError, qb.interp, x, qb.asarray([0.0, 0.0]), qb.asarray([1.0, 2.0]), match="increasing"
    )
    raises(ValueError, qb.interp, x, qb.asarray([0.0, 1.0]), qb.asarray([1.0]), match="same length")
    raises(ValueError, qb.linalg.norm, x, "fro", axis=1, match="for vectors")
    raises(ValueError, qb.linalg.norm, x, 3, match="for matrices")
    raises(ValueError, qb.linalg.norm, qb.ones([2, 2, 2]), 1, match="dimensions")
    raises(TypeError, qb.linalg.matrix_power, qb.eye(2), 1.5, match="integer")
    raises(ValueError, qb.linalg.matrix_power, x, 2, match="square")


def test_trace_keeps_the_v0_1_call_form():
    from quabla import _compat

    _compat._WARNED.discard("trace")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = qb.trace(lambda a: a, [("x", (2, 2))])
        assert qb.trace(qb.eye(3)).item() == 3.0
    assert isinstance(legacy, qb.legacy.TraceResult)
    assert [w.category for w in seen] == [DeprecationWarning]
    assert "quabla.legacy.trace" in str(seen[0].message)
    assert "trace" in qb.__all__ and qb.trace is qb._ops.trace


# Gradients against central differences of the eager float64 function.


def central_difference(function, arguments, step=1e-6):
    """d sum(function(*arguments)) / d argument, for each argument, from
    flat lists, with quabla's eager float64 evaluation."""
    gradients = []
    for position, argument in enumerate(arguments):
        values, shape = argument.to_flat_list(), argument.shape
        gradient = []
        for index in range(len(values)):
            total = []
            for sign in (1.0, -1.0):
                moved = list(values)
                moved[index] += sign * step
                shifted = list(arguments)
                shifted[position] = qb.Tensor(shape, moved)
                total.append(function(*shifted).sum().item())
            gradient.append((total[0] - total[1]) / (2 * step))
        gradients.append(gradient)
    return gradients


def check_gradient(function, *arguments, tolerance=1e-7):
    analytic = qb.grad(lambda *a: function(*a).sum(), argnums=tuple(range(len(arguments))))(
        *arguments
    )
    numeric = central_difference(function, arguments)
    for got, want in zip(analytic, numeric):
        same(got, want, tolerance)


def weights(shape, seed):
    """Deterministic weights without NumPy, so the losses are not symmetric."""
    count = math.prod(shape)
    return qb.Tensor(list(shape), [math.sin(1.7 * (i + seed)) + 0.3 for i in range(count)])


def test_gradients_match_central_differences():
    x = weights((2, 3), 1)
    y = weights((3, 2), 2)
    v = weights((3,), 3)
    w = weights((3,), 4)
    square = weights((3, 3), 5) + 3.0 * qb.eye(3)
    tall = weights((4, 2), 6)
    cases = [
        (lambda a: qb.flip(a, 1) * weights((2, 3), 7), (x,)),
        (lambda a: qb.roll(a, 2) * weights((2, 3), 7), (x,)),
        (lambda a: qb.pad(a, ((1, 2), (3, 1)), "reflect") * weights((5, 7), 7), (x,)),
        (lambda a: qb.pad(a, 2, "wrap") * weights((6, 7), 7), (x,)),
        (lambda a: qb.pad(a, 1, constant_values=2.0) * weights((4, 5), 7), (x,)),
        (lambda a: qb.tile(a, (2, 1, 2)) * weights((2, 2, 6), 7), (x,)),
        (lambda a: qb.repeat(a, [1, 0, 3], axis=1) * weights((2, 4), 7), (x,)),
        (lambda a: qb.repeat(a, 2) * weights((12,), 7), (x,)),
        (lambda a: qb.diagonal(a, 1) * weights((2,), 7), (x,)),
        (lambda a: qb.trace(a, -1), (y,)),
        (lambda a: qb.diag(a, -1) * weights((4, 4), 7), (v,)),
        (lambda a, b: qb.outer(a, b) * weights((6, 3), 7), (x, v)),
        (lambda a, b: qb.dot(a, b) * weights((2, 2), 7), (x, y)),
        (lambda a, b: qb.tensordot(a, b, ([0, 1], [1, 0])), (x, y)),
        (lambda a, b: qb.kron(a, b) * weights((6, 6), 7), (x, y)),
        (lambda a, b: qb.cross(a, b) * weights((3,), 7), (v, w)),
        (lambda a: qb.diff(a, 2) * weights((2, 1), 7), (x,)),
        (lambda a, b: qb.trapezoid(a, b), (x, qb.asarray([0.0, 0.3, 1.1]))),
        (lambda p, a: qb.polyval(p, a) * weights((2, 3), 7), (v, x)),
        (lambda a, b: qb.logaddexp(a, b) * weights((2, 3), 7), (x, y.T)),
        (lambda a, b: qb.hypot(a, b) * weights((2, 3), 7), (x, y.T)),
        (lambda a: qb.exp2(a) * weights((2, 3), 7), (x,)),
        (lambda a: qb.linalg.norm(a, axis=1), (x,)),
        (lambda a: qb.linalg.norm(a, 3, axis=0), (x,)),
        (lambda a: qb.linalg.norm(a, -1.5, axis=0), (x,)),
        (lambda a: qb.linalg.norm(a, 1, axis=0), (x,)),
        (lambda a: qb.linalg.norm(a, INF, axis=0), (x,)),
        (lambda a: qb.linalg.norm(a, "fro"), (x,)),
        (lambda a: qb.linalg.norm(a, INF), (square,)),
        (lambda a: qb.linalg.norm(a, 2), (square,)),
        (lambda a: qb.linalg.norm(a, "nuc"), (tall,)),
        (lambda a: qb.linalg.matrix_power(a, 5) * weights((3, 3), 7), (square / 3.0,)),
        (lambda a: qb.linalg.matrix_power(a, -2) * weights((3, 3), 7), (square,)),
        (lambda a: qb.linalg.pinv(a) * weights((2, 4), 7), (tall,)),
    ]
    for function, arguments in cases:
        check_gradient(function, *arguments)


def test_interp_gradients_in_x_xp_and_fp():
    xp = qb.asarray([0.0, 1.0, 3.0, 4.0])
    fp = qb.asarray([1.0, 3.0, -1.0, 0.5])
    query = qb.asarray([-0.5, 0.25, 1.5, 2.9, 3.5, 4.5])
    check_gradient(lambda a, b, c: qb.interp(a, b, c) * weights((6,), 3), query, xp, fp)
    check_gradient(lambda a, b, c: qb.interp(a, b, c, -2.0, 5.0), query, xp, fp)
    # At a knot the derivative in x is the slope to the right, at the last
    # knot the last slope; outside the range it is zero.
    knots = qb.asarray([-1.0, 0.0, 1.0, 3.0, 4.0, 5.0, INF])
    slope = qb.grad(lambda a: qb.interp(a, xp, fp).sum())(knots)
    same(slope, [0, 2, -2, 1.5, 1.5, 0, 0])
    # An infinite query puts no NaN into the gradients of xp and fp.
    for gradient in qb.grad(lambda b, c: qb.interp(knots, b, c).sum(), argnums=(0, 1))(xp, fp):
        assert all(math.isfinite(value) for value in flat(gradient))


def test_gradients_at_ties_kinks_and_the_origin():
    tie = qb.grad(lambda a, b: qb.logaddexp(a, b).sum(), argnums=(0, 1))
    same(tie(qb.asarray([2.0, -INF]), qb.asarray([2.0, 0.0]))[0], [0.5, 0])
    same(tie(qb.asarray([2.0]), qb.asarray([2.0]))[1], [0.5])
    same(tie(qb.asarray([1e30]), qb.asarray([0.0]))[1], [0.0])
    origin = qb.grad(lambda a, b: qb.hypot(a, b).sum(), argnums=(0, 1))(
        qb.asarray([0.0, 3.0]), qb.asarray([0.0, -4.0])
    )
    same(origin[0], [0, 0.6], 1e-15)
    same(origin[1], [0, -0.8], 1e-15)
    replaced = qb.grad(lambda a: qb.nan_to_num(a).sum())(qb.asarray([1.0, INF, NAN]))
    same(replaced, [1, 0, 0])
    same(qb.grad(lambda a: qb.exp2(a).sum())(qb.asarray([3.0])), [8 * math.log(2.0)], 1e-15)


# Transforms: jit and vmap agree with the eager results.


def transform_cases(dtype=qb.float64):
    x = weights((3, 4), 1)
    v = weights((4,), 2)
    m = weights((4, 4), 3) + 3.0 * qb.eye(4)
    xp = qb.asarray([0.0, 1.0, 2.5, 3.0], dtype=dtype)
    fp = qb.asarray([1.0, -1.0, 2.0, 0.0], dtype=dtype)
    return [
        (lambda a: qb.flip(a), (x,)),
        (lambda a: qb.roll(a, (1, -1), (0, 1)), (x,)),
        (lambda a: qb.pad(a, ((2, 1), (0, 3)), "symmetric"), (x,)),
        (lambda a: qb.pad(a, 1, constant_values=-3.0), (x,)),
        (lambda a: qb.tile(a, (2, 2)), (x,)),
        (lambda a: qb.repeat(a, [2, 1, 0, 1], axis=1), (x,)),
        (lambda a: qb.swapaxes(qb.moveaxis(a, 1, 0), 0, 1), (x,)),
        (lambda a: qb.ravel(a), (x,)),
        (lambda a: qb.diagonal(a, 1), (x,)),
        (lambda a: qb.trace(a), (m,)),
        (lambda a: qb.diag(a, 2), (v,)),
        (lambda a, b: qb.outer(a, b), (x, v)),
        (lambda a, b: qb.dot(a, b), (x, m)),
        (lambda a, b: qb.tensordot(a, b, ([1], [0])), (x, m)),
        (lambda a, b: qb.kron(a, b), (v, m)),
        (lambda a, b: qb.cross(a[:, :3], b[:3]), (x, v)),
        (lambda a: qb.diff(a, 2, axis=-1), (x,)),
        (lambda a: qb.trapezoid(a, dx=0.25), (x,)),
        (lambda a: qb.interp(a, xp, fp, -2.0), (x,)),
        (lambda a, b: qb.interp(a, xp, b), (x, fp)),
        (lambda p, a: qb.polyval(p, a), (v, x)),
        (lambda a, b: qb.logaddexp(a, b), (x, v)),
        (lambda a, b: qb.hypot(a, b), (x, v)),
        (lambda a: qb.exp2(a), (x,)),
        (lambda a: qb.isinf(a / (a - a.max())), (x,)),
        (lambda a: qb.nan_to_num(a / (a - a.max())), (x,)),
        (lambda a: qb.linalg.norm(a, 3, axis=1), (x,)),
        (lambda a: qb.linalg.norm(a, INF), (m,)),
        (lambda a: qb.linalg.matrix_power(a, 3), (m,)),
    ]


SVD_CASES = [
    (lambda a: qb.linalg.norm(a, 2), "square"),
    (lambda a: qb.linalg.norm(a, "nuc"), "square"),
    (lambda a: qb.linalg.pinv(a), "tall"),
    (lambda a: qb.linalg.matrix_power(a, -2), "square"),
]


def svd_cases():
    operands = {"square": weights((3, 3), 5) + 3.0 * qb.eye(3), "tall": weights((4, 2), 6)}
    return [(function, (operands[kind],)) for function, kind in SVD_CASES]


def batched(arguments):
    """Two examples per argument, stacked along a new leading axis."""
    return tuple(qb.stack([a, a * 0.5 + 0.125], 0) for a in arguments)


def test_functions_under_jit_and_vmap_match_eager():
    for function, arguments in transform_cases() + svd_cases():
        expected = function(*arguments)
        same(qb.jit(function)(*arguments), flat(expected), 1e-15)
        stacked = batched(arguments)
        mapped = qb.vmap(function)(*stacked)
        examples = [function(*(a[i] for a in stacked)) for i in range(2)]
        same(mapped, [value for example in examples for value in flat(example)], 1e-14)


def test_device_results_match_cpu():
    for device in devices():
        cases = transform_cases(qb.float32) + svd_cases()
        for function, arguments in cases:
            arguments = tuple(a.astype(qb.float32) for a in arguments)
            expected = qb.jit(function)(*arguments)
            same(qb.jit(function, device=device)(*arguments), flat(expected), 2e-5)
            mapped = qb.jit(qb.vmap(function), device=device)(*batched(arguments))
            same(mapped, flat(qb.vmap(function)(*batched(arguments))), 2e-5)
            if expected.dtype == qb.bool_:
                continue
            argnums = tuple(range(len(arguments)))
            loss = qb.grad(lambda *a, f=function: f(*a).sum(), argnums=argnums)
            gradients = qb.jit(loss, device=device)(*arguments)
            for got, want in zip(gradients, qb.jit(loss)(*arguments)):
                same(got, flat(want), 2e-5)


# NumPy references on random and edge inputs, float64 and float32.


def numpy_cases(rng):
    a = rng.normal(size=(3, 4, 5))
    b = rng.normal(size=(2, 5, 3))
    v = rng.normal(size=5)
    xp = np.sort(rng.uniform(-1.0, 1.0, size=7))
    fp = rng.normal(size=7)
    query = np.concatenate([rng.uniform(-1.5, 1.5, size=30), xp, [INF, -INF, NAN]])
    edges = np.array([0.0, -0.0, 1.0, -2.5, 1e300, -1e300, INF, -INF, NAN])
    p1, p2 = np.meshgrid(edges, edges)
    square = rng.normal(size=(2, 4, 4)) + 4.0 * np.eye(4)
    tall = rng.normal(size=(2, 5, 3))
    deficient = tall.copy()
    deficient[1, :, 2] = deficient[1, :, 0] - deficient[1, :, 1]
    cases = [
        (qb.flip, np.flip, (a,), {"axis": (0, 2)}),
        (qb.roll, np.roll, (a, 7), {}),
        (qb.roll, np.roll, (a, (2, -9), (1, 2)), {}),
        (qb.tile, np.tile, (a, (2, 1, 1, 2)), {}),
        (qb.repeat, np.repeat, (a, 3), {"axis": -1}),
        (qb.repeat, np.repeat, (a, [0, 2, 1, 4]), {"axis": 1}),
        (qb.moveaxis, np.moveaxis, (a, (0, 2), (2, 1)), {}),
        (qb.swapaxes, np.swapaxes, (a, 0, -1), {}),
        (qb.ravel, np.ravel, (a,), {}),
        (qb.diagonal, np.diagonal, (a, -2, 2, 1), {}),
        (qb.trace, np.trace, (a, 1), {"axis1": 0, "axis2": 2}),
        (qb.diag, np.diag, (v, -3), {}),
        (qb.diag, np.diag, (a[0], 2), {}),
        (qb.outer, np.outer, (a[0], v), {}),
        (qb.dot, np.dot, (a, b), {}),
        (qb.dot, np.dot, (v, b), {}),
        (qb.dot, np.dot, (a, v), {}),
        (qb.tensordot, np.tensordot, (a, b[0], 1), {}),
        (qb.tensordot, np.tensordot, (a, b, ([2, 0], [1, 2])), {}),
        (qb.kron, np.kron, (a[0], b[1]), {}),
        (qb.cross, np.cross, (a[..., :3], v[:3]), {}),
        (qb.cross, np.cross, (b, b[:, ::-1] + 1.0), {"axisa": 2, "axisb": 2, "axisc": 0}),
        (qb.diff, np.diff, (a, 3), {"axis": 2, "prepend": 0.5}),
        (qb.trapezoid, np.trapezoid, (a, v), {}),
        (qb.trapezoid, np.trapezoid, (a, a), {"axis": 0}),
        (qb.trapezoid, np.trapezoid, (a,), {"dx": 0.2, "axis": 1}),
        (qb.interp, np.interp, (query, xp, fp), {}),
        (qb.interp, np.interp, (query, xp, fp, 2.0, -3.0), {}),
        (qb.polyval, np.polyval, (v, a), {}),
        (qb.logaddexp, np.logaddexp, (p1, p2), {}),
        (qb.logaddexp, np.logaddexp, (a, b[0, :, :1].T[:, None, :1]), {}),
        (qb.hypot, np.hypot, (p1, p2), {}),
        (qb.hypot, np.hypot, (a, v), {}),
        (qb.exp2, np.exp2, (a * 10.0,), {}),
        (qb.isinf, np.isinf, (edges,), {}),
        # The default replacements are the limits of the operand's dtype.
        (qb.nan_to_num, np.nan_to_num, (edges,), {}, True),
        (qb.nan_to_num, np.nan_to_num, (edges,), {"nan": 1.0, "posinf": 2.0, "neginf": -3.0}),
        (qb.linalg.norm, np.linalg.norm, (a,), {}),
        (qb.linalg.matrix_power, np.linalg.matrix_power, (square, 7), {}),
        (qb.linalg.matrix_power, np.linalg.matrix_power, (square, -3), {}),
        (qb.linalg.pinv, lambda m: np.linalg.pinv(m, rtol=None), (tall,), {}),
        # An explicit cutoff, so that float32 rounding of the zero singular
        # value cannot cross it.
        (qb.linalg.pinv, np.linalg.pinv, (deficient.transpose(0, 2, 1),), {"rtol": 1e-5}),
    ]
    for mode in ("constant", "edge", "reflect", "symmetric", "wrap"):
        cases.append((qb.pad, np.pad, (a, ((1, 2), (0, 9), (6, 1))), {"mode": mode}))
    cases.append((qb.pad, np.pad, (a, 2), {"constant_values": ((1.5, -2.0),)}))
    for order in (None, 2, 1, INF, -INF, 0, 3, 0.5, -1.5):
        cases.append((qb.linalg.norm, np.linalg.norm, (a, order), {"axis": 1}))
    for order in (None, "fro", 1, -1, INF, -INF, 2, -2, "nuc"):
        cases.append(
            (qb.linalg.norm, np.linalg.norm, (a, order), {"axis": (2, 0), "keepdims": True})
        )
    return cases


def test_functions_match_numpy_in_float64_and_float32():
    if np is None:
        skip("numpy is not installed")
    rng = np.random.default_rng(0)
    # The edge inputs make NumPy warn about the infinities and NaN it returns.
    with np.errstate(all="ignore"):
        for case in numpy_cases(rng):
            for dtype, tolerance in ((np.float64, 1e-12), (np.float32, 2e-5)):
                compare_with_numpy(*case, dtype=dtype, tolerance=tolerance)


def compare_with_numpy(function, reference, arguments, keywords, native=False, *, dtype, tolerance):
    """`function` eagerly and under `jit` against `reference` on the array
    arguments rounded to `dtype`; the reference runs in float64 on the
    rounded values, or in `dtype` when `native`."""
    narrowed = [v.astype(dtype) if isinstance(v, np.ndarray) else v for v in arguments]
    expected = np.asarray(
        reference(
            *[
                v.astype(np.float64) if isinstance(v, np.ndarray) and not native else v
                for v in narrowed
            ],
            **keywords,
        )
    )
    if expected.dtype == bool:
        expected_dtype = qb.bool_
    else:
        # Values beyond the float32 range saturate as the float32 result does.
        expected = expected.astype(dtype)
        expected_dtype = qb.float32 if dtype == np.float32 else qb.float64
    actual = function(*narrowed, **keywords)
    assert actual.dtype == expected_dtype
    assert_close(actual.numpy(), expected, tolerance)
    positions = [i for i, value in enumerate(narrowed) if isinstance(value, np.ndarray)]

    def traced(*values):
        merged = list(narrowed)
        for i, value in zip(positions, values):
            merged[i] = value
        return function(*merged, **keywords)

    assert_close(qb.jit(traced)(*[narrowed[i] for i in positions]).numpy(), expected, tolerance)


if __name__ == "__main__":
    if np is None:
        print("note: numpy is not installed; NumPy comparisons are skipped")
    run(globals())
