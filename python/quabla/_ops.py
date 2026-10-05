"""Module-level math functions (docs/api_v0_2_design.md, section 3.2, D4).

Each function calls the method of the same name on `Tensor` or
`TraceTensor`, so eager and traced code share one spelling: `qb.sin(x)` is
`x.sin()`, with the method's dtype rules and errors. Other operands (Python
numbers, nested lists, NumPy arrays) go through `asarray` first, except that
the second operand of `maximum`/`minimum` stays a Python number when it is
one, keeping the weak scalar typing of the method form:
`qb.maximum(x, 0.0)` keeps the dtype of `x`. `power` keeps a Python number
in either position the same way.

`abs`, `sum`, `max`, `min`, `any`, and `all` shadow Python builtins, so they
are attributes of `quabla` but not listed in `__all__`: `from quabla import *`
leaves the builtins alone.

The remaining functions (activations, `softmax`/`logsumexp`, `var`/`std`,
and the shape helpers) are compositions of those methods, so they work
eagerly, under every transform, and on every device that runs the methods
they use, with derivatives from the composed ops.
"""

import builtins
import math
import numbers
import operator

from ._array import _shape, asarray, full
from ._quabla import Tensor, TraceTensor
from ._quabla import where as _where

__all__ = [
    "astype",
    "broadcast_to",
    "cholesky",
    "clip",
    "cos",
    "exp",
    "expand_dims",
    "full_like",
    "gelu",
    "log",
    "log1p",
    "log_softmax",
    "logsumexp",
    "matmul",
    "maximum",
    "mean",
    "meshgrid",
    "minimum",
    "norm",
    "ones_like",
    "power",
    "reciprocal",
    "relu",
    "reshape",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "softmax",
    "softplus",
    "solve",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "tanh",
    "transpose",
    "tril",
    "triu",
    "var",
    "zeros_like",
]

_ARRAYS = (Tensor, TraceTensor)


def _array(x):
    return x if isinstance(x, _ARRAYS) else asarray(x)


def _unary(name):
    def function(x):
        return getattr(_array(x), name)()

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"Elementwise `{name}`; `quabla.{name}(x)` is `x.{name}()`."
    return function


def _reduction(name):
    def function(x, axis=None, keepdims=False):
        return getattr(_array(x), name)(axis=axis, keepdims=keepdims)

    function.__name__ = function.__qualname__ = name
    function.__doc__ = (
        f"`{name}` over `axis` (an int, a sequence of ints, or `None` for all axes); "
        f"`quabla.{name}(x, axis, keepdims)` is `x.{name}(axis, keepdims)`."
    )
    return function


def _symmetric_binary(name):
    def function(x1, x2):
        if isinstance(x1, numbers.Number) and isinstance(x2, _ARRAYS):
            x1, x2 = x2, x1
        return getattr(_array(x1), name)(x2 if isinstance(x2, numbers.Number) else _array(x2))

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"Elementwise `{name}` of two arrays, or of an array and a Python number."
    return function


abs = _unary("abs")
cos = _unary("cos")
exp = _unary("exp")
log = _unary("log")
log1p = _unary("log1p")
relu = _unary("relu")
sigmoid = _unary("sigmoid")
sin = _unary("sin")
softplus = _unary("softplus")
sqrt = _unary("sqrt")
tanh = _unary("tanh")
cholesky = _unary("cholesky")
tril = _unary("tril")
triu = _unary("triu")

sum = _reduction("sum")
mean = _reduction("mean")
max = _reduction("max")
min = _reduction("min")
norm = _reduction("norm")
any = _reduction("any")
all = _reduction("all")

maximum = _symmetric_binary("maximum")
minimum = _symmetric_binary("minimum")


def power(x1, x2):
    """Elementwise `x1 ** x2`; `quabla.power(x, y)` is `x ** y`.

    Either operand may be a Python number, which stays a weak scalar adopting
    the dtype of the array operand. Values follow `f64::powf`: NaN for a
    negative base with a non-integer exponent, `0 ** 0 == 1`, IEEE results
    for infinities and NaN. Traced code lowers a non-negative Python int
    exponent to the exact `powi` and every other exponent, including a traced
    array, to the differentiable `pow` op, whose derivative conventions at
    `x1 <= 0` are listed in `docs/api.md`.
    """
    if isinstance(x1, numbers.Number) and not isinstance(x2, numbers.Number):
        # Called directly: `x1 ** x2` would let a NumPy scalar base take over.
        return _array(x2).__rpow__(x1)
    return _array(x1) ** (x2 if isinstance(x2, numbers.Number) else _array(x2))


def matmul(x1, x2):
    """Matrix product with NumPy's rules: for operands of rank 2 or more it
    is `x1 @ x2`, batched over the leading axes. A rank-1 `x1` is a row
    vector and a rank-1 `x2` a column vector, whose unit axis is removed
    from the result: vector @ vector is a scalar, matrix @ vector a vector.
    The vector cases reshape around the same `matmul` op."""
    a, b = _array(x1), _array(x2)
    a_vector, b_vector = len(a.shape) == 1, len(b.shape) == 1
    if not (a_vector or b_vector):
        return a.matmul(b)
    if a_vector:
        a = a.reshape([1, a.shape[0]])
    if b_vector:
        b = b.reshape([b.shape[0], 1])
    product = a.matmul(b)
    shape = list(product.shape)
    if b_vector:
        del shape[-1]
    if a_vector:
        del shape[-1 if b_vector else -2]
    return product.reshape(shape)


def solve(a, b):
    """Solves `a @ x == b` for `x`; `quabla.solve(a, b)` is `a.solve(b)`."""
    return _array(a).solve(_array(b))


def reshape(x, shape):
    """`x` with a new shape of the same size (an int or a sequence of ints);
    one extent may be `-1`, inferred from the size."""
    return _array(x).reshape(_shape(shape))


def transpose(x, axes=None):
    """`x` with its axes permuted by `axes` (reversed when `None`)."""
    return _array(x).transpose(None if axes is None else list(axes))


def broadcast_to(x, shape):
    """`x` broadcast to `shape` (an int or a sequence of ints)."""
    return _array(x).broadcast_to(_shape(shape))


def astype(x, dtype):
    """`x` converted to `dtype`; `quabla.astype(x, dtype)` is `x.astype(dtype)`."""
    return _array(x).astype(dtype)


def _axes(axis, ndim):
    """`axis` (an int or a sequence of ints) as a tuple of distinct
    non-negative axes of a rank-`ndim` array; `None` means every axis."""
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, numbers.Integral) else tuple(axis)
    normalized = []
    for value in axes:
        value = operator.index(value)
        if not -ndim <= value < ndim:
            raise ValueError(f"axis {value} is out of bounds for rank {ndim}")
        normalized.append(value % ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"repeated axis in {axis!r}")
    return tuple(normalized)


# Activations and elementwise helpers.


def square(x):
    """Elementwise `x * x`."""
    x = _array(x)
    return x * x


def reciprocal(x):
    """Elementwise `1 / x`."""
    return 1.0 / _array(x)


def sign(x):
    """Elementwise sign: -1 for negative values, 0 for zeros of either sign,
    +1 for positive values, and NaN for NaN. Built from comparisons, so its
    derivative is zero everywhere, as in JAX. The NaN comes from `x - x`
    rather than a NaN constant, which the CUDA and MLX backends reject, and
    its derivative cancels to zero too."""
    x = _array(x)
    signs = (x > 0.0).astype(x.dtype) - (x < 0.0).astype(x.dtype)
    return _where(x.isnan(), x - x, signs)


def clip(x, lo=None, hi=None):
    """`x` limited to `[lo, hi]` elementwise as `minimum(maximum(x, lo), hi)`;
    either bound may be `None`. A NaN in `x` or in a bound propagates, and at
    a bound the derivative goes to the bound (`maximum`/`minimum` route ties
    to their right operand), so `x` gets zero there."""
    x = _array(x)
    if lo is not None:
        x = maximum(x, lo)
    if hi is not None:
        x = minimum(x, hi)
    return x


def silu(x):
    """SiLU (swish), `x * sigmoid(x)`, on the overflow-free `sigmoid`."""
    x = _array(x)
    return x * x.sigmoid()


# 2 * sqrt(2 / pi): the tanh-approximation argument, doubled for `sigmoid`.
_GELU_SCALE = 2.0 * math.sqrt(2.0 / math.pi)


def gelu(x, approximate=True):
    """GELU with the tanh approximation
    `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, evaluated through the
    identity `0.5 (1 + tanh(u)) == sigmoid(2u)`: the stable `sigmoid` keeps
    full relative accuracy in the negative tail, where `1 + tanh(u)` would
    cancel to zero. `approximate=False` (the exact `erf` form) is not
    available yet; it arrives with the native `erf` op."""
    if not approximate:
        raise NotImplementedError(
            "gelu(approximate=False) needs erf, which arrives with the native erf op; "
            "use the default tanh approximation"
        )
    x = _array(x)
    return x * (x * (x * x * 0.044715 + 1.0) * _GELU_SCALE).sigmoid()


# Normalizations and reductions.
#
# TODO: shift by `stop_gradient(max)` once the native `stop_gradient` op
# lands. The shift cancels from every result below, so its derivative
# contributes only rounding today, but it still costs a backward pass
# through `max`.


def softmax(x, axis=-1):
    """`exp(x) / sum(exp(x))` along `axis` (an int or a sequence of ints),
    computed as `exp(x - m) / sum(exp(x - m))` with `m` the maximum along
    `axis`, so no `exp` overflows: `softmax([1000, 0])` is `[1, 0]` with a
    finite gradient. As in JAX, a slice that holds `+inf` or only `-inf`
    gives NaN."""
    x = _array(x)
    unnormalized = (x - x.max(axis=axis, keepdims=True)).exp()
    return unnormalized / unnormalized.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """`log(softmax(x))` along `axis`, computed as
    `(x - m) - log(sum(exp(x - m)))` with `m` the maximum along `axis`: it
    neither overflows nor takes the log of an underflowed probability, so
    `log_softmax([1000, 0])` is `[0, -1000]`."""
    x = _array(x)
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def logsumexp(x, axis=None, keepdims=False):
    """`log(sum(exp(x)))` over `axis` (an int, a sequence of ints, or `None`
    for all axes), computed as `m + log(sum(exp(x - m)))` with `m` the
    maximum over `axis`. Where `m` is not finite the shift is zero instead,
    as in JAX: a slice of only `-inf` gives `-inf`, one holding `+inf` gives
    `+inf`, and NaN propagates. The gradient is `softmax(x)`."""
    x = _array(x)
    axes = None if axis is None else _axes(axis, len(x.shape))
    peak = x.max(axis=axes, keepdims=True)
    shift = _where(peak.isfinite(), peak, 0.0)
    result = (x - shift).exp().sum(axis=axes, keepdims=True).log() + shift
    return result if keepdims else squeeze(result, axes)


def var(x, axis=None, keepdims=False, ddof=0):
    """Variance over `axis` (an int, a sequence of ints, or `None` for all
    axes), `sum((x - mean(x))**2) / (n - ddof)` for `n` reduced elements.
    Two passes, the mean first and then the squared deviations, so there is
    no cancellation of `E[x^2] - E[x]^2` for data far from zero. `n <= ddof`
    divides by zero (inf or NaN), as NumPy does with a warning."""
    x = _array(x)
    axes = _axes(axis, len(x.shape))
    count = math.prod(x.shape[a] for a in axes)
    deviation = x - x.mean(axis=None if axis is None else axes, keepdims=True)
    squares = (deviation * deviation).sum(axis=None if axis is None else axes, keepdims=keepdims)
    return squares / float(builtins.max(count - ddof, 0))


def std(x, axis=None, keepdims=False, ddof=0):
    """Standard deviation, `sqrt(var(x, axis, keepdims, ddof))`. Its
    derivative at zero variance is zero, following the `sqrt` convention
    (JAX gives NaN there)."""
    return var(x, axis=axis, keepdims=keepdims, ddof=ddof).sqrt()


# Shape helpers: reshapes, slices, and broadcasts of their operand.


def squeeze(x, axis=None):
    """`x` without the unit axes `axis` (an int or a sequence of ints), or
    without every unit axis when `axis` is `None`. Selecting an axis whose
    extent is not 1 raises `ValueError`."""
    x = _array(x)
    shape = x.shape
    if axis is None:
        axes = {a for a, extent in enumerate(shape) if extent == 1}
    else:
        axes = set(_axes(axis, len(shape)))
        if builtins.any(shape[a] != 1 for a in axes):
            raise ValueError(
                f"cannot squeeze axis {axis!r} of shape {list(shape)}: its extent is not 1"
            )
    if not axes:
        return x
    return x.reshape([extent for a, extent in enumerate(shape) if a not in axes])


def expand_dims(x, axis):
    """`x` with unit axes inserted so that they sit at positions `axis` (an
    int or a sequence of ints) of the result."""
    x = _array(x)
    count = 1 if isinstance(axis, numbers.Integral) else len(tuple(axis))
    ndim = len(x.shape) + count
    axes = _axes(axis, ndim)
    extents = iter(x.shape)
    return x.reshape([1 if a in axes else next(extents) for a in range(ndim)])


def split(x, indices_or_sections, axis=0):
    """`x` split along `axis` into a list of slices: `n` equal pieces for an
    int `n` (which must divide the extent), or the pieces between the given
    increasing split indices. A piece may not be empty, since quabla arrays
    have no zero-extent axes."""
    x = _array(x)
    (axis,) = _axes(axis, len(x.shape))
    extent = x.shape[axis]
    if isinstance(indices_or_sections, numbers.Integral):
        sections = operator.index(indices_or_sections)
        if sections <= 0 or extent % sections:
            raise ValueError(
                f"cannot split axis {axis} of extent {extent} into {sections} equal pieces"
            )
        width = extent // sections
        bounds = [width * k for k in range(sections + 1)]
    else:
        bounds = [0, *(operator.index(i) for i in indices_or_sections), extent]
    prefix = (slice(None),) * axis
    pieces = []
    for start, stop in zip(bounds, bounds[1:]):
        if len(range(*slice(start, stop).indices(extent))) == 0:
            raise ValueError(
                f"split indices {indices_or_sections!r} give an empty piece "
                f"[{start}:{stop}] of axis {axis} with extent {extent}"
            )
        pieces.append(x[prefix + (slice(start, stop),)])
    return pieces


def meshgrid(*xs, indexing="xy"):
    """Coordinate arrays from the 1-D arrays `xs` (others are flattened), as
    a list. Every result has one axis per input; with `indexing="ij"` the
    axes follow the inputs, and with the default `"xy"` (Cartesian) the
    first two are swapped, as in NumPy."""
    if indexing not in ("xy", "ij"):
        raise ValueError(f'indexing must be "xy" or "ij", got {indexing!r}')
    arrays = [_array(x).reshape(-1) for x in xs]
    positions = list(range(len(arrays)))
    if indexing == "xy" and len(arrays) > 1:
        positions[0], positions[1] = 1, 0
    shape = [0] * len(arrays)
    for array, position in zip(arrays, positions):
        shape[position] = array.shape[0]
    grids = []
    for array, position in zip(arrays, positions):
        view = [1] * len(arrays)
        view[position] = shape[position]
        grids.append(array.reshape(view).broadcast_to(shape))
    return grids


def full_like(x, fill_value, dtype=None):
    """A `Tensor` with the shape of `x` filled with `fill_value`, of the
    dtype of `x` unless `dtype` is given. For a traced `x` it is a constant
    of the trace: it does not depend on the values of `x`."""
    x = _array(x)
    return full(x.shape, fill_value, x.dtype if dtype is None else dtype)


def zeros_like(x, dtype=None):
    """`full_like(x, 0, dtype)`."""
    return full_like(x, 0.0, dtype)


def ones_like(x, dtype=None):
    """`full_like(x, 1, dtype)`."""
    return full_like(x, 1.0, dtype)
