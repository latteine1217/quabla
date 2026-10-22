"""Module-level math functions (docs/api_v0_2_design.md, section 3.2, D4).

Each function calls the method of the same name on `Tensor` or
`TraceTensor`, so eager and traced code share one spelling: `qb.sin(x)` is
`x.sin()`, with the method's dtype rules and errors. Other operands (Python
numbers, nested lists, NumPy arrays) go through `asarray` first, except that
the second operand of `maximum`/`minimum` stays a Python number when it is
one, keeping the weak scalar typing of the method form:
`qb.maximum(x, 0.0)` keeps the dtype of `x`. `power` keeps a Python number
in either position the same way.

`abs`, `round`, `sum`, `max`, `min`, `any`, and `all` shadow Python builtins,
so they are attributes of `quabla` but not listed in `__all__`:
`from quabla import *` leaves the builtins alone.

The remaining functions (activations, `softmax`/`logsumexp`, `var`/`std`,
and the shape helpers) are compositions of those methods, so they work
eagerly, under every transform, and on every device that runs the methods
they use, with derivatives from the composed ops.

The jax.numpy-style functions added in v0.4 (`flip` through `nan_to_num`,
after the shape helpers) are compositions of the same kind: reshapes,
transposes, broadcasts, slices, static-index gathers, `concat`, `where`, and
`matmul`. Their integer arguments (shifts, pad widths, repeats, axes) are
static Python ints, since quabla has no integer arrays.
"""

from __future__ import annotations

import builtins
import math
import numbers
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from . import linalg as _linalg
from ._array import _shape, arange, asarray, full
from ._quabla import Tensor, TraceTensor, bool_, concat, float32, stack
from ._quabla import where as _where

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from ._typing import Array, ArrayLike, Axis, DType, Reduction, Shape

__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "astype",
    "atan2",
    "broadcast_to",
    "cbrt",
    "ceil",
    "cholesky",
    "clip",
    "cos",
    "cosh",
    "cross",
    "cumsum",
    "diag",
    "diagonal",
    "diff",
    "dot",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expand_dims",
    "expm1",
    "flip",
    "floor",
    "fmod",
    "full_like",
    "gelu",
    "hypot",
    "interp",
    "isinf",
    "kron",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_softmax",
    "logaddexp",
    "logsumexp",
    "matmul",
    "maximum",
    "mean",
    "meshgrid",
    "minimum",
    "mod",
    "moveaxis",
    "nan_to_num",
    "norm",
    "ones_like",
    "outer",
    "pad",
    "polyval",
    "power",
    "prod",
    "ravel",
    "reciprocal",
    "relu",
    "remainder",
    "repeat",
    "reshape",
    "roll",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "sinh",
    "softmax",
    "softplus",
    "solve",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "stop_gradient",
    "swapaxes",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "trace",
    "transpose",
    "trapezoid",
    "tril",
    "triu",
    "var",
    "zeros_like",
]

_ARRAYS = (Tensor, TraceTensor)


def _is_number(value: object) -> TypeIs[float]:
    """Whether `value` is a Python or NumPy number, which stays a weak
    scalar."""
    return isinstance(value, numbers.Number)


def _is_integral(value: object) -> TypeIs[int]:
    """Whether `value` is a Python or NumPy integer."""
    return isinstance(value, numbers.Integral)


def _array(x: ArrayLike) -> Array:
    return x if isinstance(x, _ARRAYS) else asarray(x)


def _unary(name: str) -> Callable[[ArrayLike], Array]:
    def function(x: ArrayLike) -> Array:
        return getattr(_array(x), name)()

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"Elementwise `{name}`; `quabla.{name}(x)` is `x.{name}()`."
    return function


def _reduction(name: str) -> Reduction:
    def function(x: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Array:
        return getattr(_array(x), name)(axis=axis, keepdims=keepdims)

    function.__name__ = function.__qualname__ = name
    function.__doc__ = (
        f"`{name}` over `axis` (an int, a sequence of ints, or `None` for all axes); "
        f"`quabla.{name}(x, axis, keepdims)` is `x.{name}(axis, keepdims)`."
    )
    return function


def _symmetric_binary(name: str) -> Callable[[ArrayLike, ArrayLike], Array]:
    def function(x1: ArrayLike, x2: ArrayLike) -> Array:
        if _is_number(x1) and isinstance(x2, _ARRAYS):
            x1, x2 = x2, x1
        return getattr(_array(x1), name)(x2 if _is_number(x2) else _array(x2))

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"Elementwise `{name}` of two arrays, or of an array and a Python number."
    return function


abs = _unary("abs")
cos = _unary("cos")
exp = _unary("exp")
expm1 = _unary("expm1")
erf = _unary("erf")
erfc = _unary("erfc")
log = _unary("log")
log1p = _unary("log1p")
relu = _unary("relu")
sigmoid = _unary("sigmoid")
sin = _unary("sin")
softplus = _unary("softplus")
sqrt = _unary("sqrt")
tanh = _unary("tanh")
tan = _unary("tan")
arcsin = _unary("arcsin")
arccos = _unary("arccos")
arctan = _unary("arctan")
sinh = _unary("sinh")
cosh = _unary("cosh")
arcsinh = _unary("arcsinh")
arccosh = _unary("arccosh")
arctanh = _unary("arctanh")
log2 = _unary("log2")
log10 = _unary("log10")
cbrt = _unary("cbrt")
floor = _unary("floor")
ceil = _unary("ceil")
round = _unary("round")
stop_gradient = _unary("stop_gradient")
cholesky = _unary("cholesky")
tril = _unary("tril")
triu = _unary("triu")

sum = _reduction("sum")
mean = _reduction("mean")
max = _reduction("max")
min = _reduction("min")
prod = _reduction("prod")
norm = _reduction("norm")
any = _reduction("any")
all = _reduction("all")

maximum = _symmetric_binary("maximum")
minimum = _symmetric_binary("minimum")


def power(x1: ArrayLike, x2: ArrayLike) -> Array:
    """Elementwise `x1 ** x2`; `quabla.power(x, y)` is `x ** y`.

    Either operand may be a Python number, which stays a weak scalar adopting
    the dtype of the array operand. Values follow `f64::powf`: NaN for a
    negative base with a non-integer exponent, `0 ** 0 == 1`, IEEE results
    for infinities and NaN. Traced code lowers a non-negative Python int
    exponent to the exact `powi` and every other exponent, including a traced
    array, to the differentiable `pow` op, whose derivative conventions at
    `x1 <= 0` are listed in `docs/api.md`.
    """
    if _is_number(x1) and not _is_number(x2):
        # Called directly: `x1 ** x2` would let a NumPy scalar base take over.
        return _array(x2).__rpow__(x1)
    return _array(x1) ** (x2 if _is_number(x2) else _array(x2))


def atan2(x1: ArrayLike, x2: ArrayLike) -> Array:
    """Elementwise four-quadrant `atan2(x1, x2)`, the angle of the point
    `(x2, x1)`; `quabla.atan2(y, x)` is `y.atan2(x)`.

    Either operand may be a Python number, which adopts the dtype of the
    array operand. Values follow `f64::atan2`. The derivative is
    `(x2, -x1) / (x1^2 + x2^2)`, evaluated without overflow, and is defined as
    zero at the origin (see `docs/api.md`).
    """
    if _is_number(x1) and not _is_number(x2):
        x2 = _array(x2)
        return asarray(x1, dtype=x2.dtype).atan2(x2)
    return _array(x1).atan2(x2 if _is_number(x2) else _array(x2))


def fmod(x1: ArrayLike, x2: ArrayLike) -> Array:
    """Elementwise C `fmod(x1, x2)`: the remainder `x1 - n x2` of the quotient
    `n` truncated toward zero, which has the sign of `x1` (NumPy's `fmod`);
    `quabla.fmod(x, y)` is `x.fmod(y)`.

    Either operand may be a Python number, which adopts the dtype of the
    array operand. The result is exact; it is NaN for `x2 == 0` or an
    infinite `x1`, and `x1` for an infinite `x2`. The derivative is `1` in
    `x1` and `-trunc(x1 / x2)` in `x2` (zero second derivatives).
    """
    if _is_number(x1) and not _is_number(x2):
        x2 = _array(x2)
        return asarray(x1, dtype=x2.dtype).fmod(x2)
    return _array(x1).fmod(x2 if _is_number(x2) else _array(x2))


def mod(x1: ArrayLike, x2: ArrayLike) -> Array:
    """Elementwise floor-mod remainder with NumPy's `mod`/`remainder`
    semantics: `x1 - floor(x1 / x2) x2`, which has the sign of `x2`, as
    Python's `%` on floats.

    It is computed from the exact `fmod` the way NumPy does it: where the
    truncated remainder `r = fmod(x1, x2)` is nonzero and its sign differs
    from that of `x2`, the result is `r + x2`; a zero result is `0` with the
    sign of `x2`. Forming `x1 - floor(x1 / x2) * x2` instead would round the
    quotient and the product and lose the small remainders. The derivative
    is `1` in `x1` and `-floor(x1 / x2)` in `x2`. `quabla.remainder` is the
    same function.
    """
    r = fmod(x1, x2)
    y = asarray(x2, dtype=r.dtype) if _is_number(x2) else _array(x2)
    y_negative = y < 0.0
    adjust = ((r < 0.0) & (y > 0.0)) | ((r > 0.0) & y_negative)
    floored = _where(adjust, r + y, r)
    # `r * r` is +0 for either zero, so the zero result takes the sign of `x2`.
    zero = r * r
    return _where(r.equal(0.0), _where(y_negative, -zero, zero), floored)


remainder = mod


def cumsum(x: ArrayLike, axis: int | None = None, reverse: bool = False) -> Array:
    """Inclusive prefix sums along `axis`, over the flattened array when
    `axis` is `None` (as in NumPy), from the last entry when `reverse`;
    `quabla.cumsum(x, axis)` is `x.cumsum(axis)`."""
    return _array(x).cumsum(axis=axis, reverse=reverse)


def matmul(x1: ArrayLike, x2: ArrayLike) -> Array:
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


def solve(a: ArrayLike, b: ArrayLike) -> Array:
    """Solves `a @ x == b` for `x`; `quabla.solve` is `quabla.linalg.solve`
    (batched over leading axes, `b` a vector `[n]` or a stack `[..., n, k]`)."""
    return _linalg.solve(a, b)


def reshape(x: ArrayLike, shape: Shape) -> Array:
    """`x` with a new shape of the same size (an int or a sequence of ints);
    one extent may be `-1`, inferred from the size."""
    return _array(x).reshape(_shape(shape))


def transpose(x: ArrayLike, axes: Sequence[int] | None = None) -> Array:
    """`x` with its axes permuted by `axes` (reversed when `None`)."""
    return _array(x).transpose(None if axes is None else list(axes))


def broadcast_to(x: ArrayLike, shape: Shape) -> Array:
    """`x` broadcast to `shape` (an int or a sequence of ints)."""
    return _array(x).broadcast_to(_shape(shape))


def astype(x: ArrayLike, dtype: DType) -> Array:
    """`x` converted to `dtype`; `quabla.astype(x, dtype)` is `x.astype(dtype)`."""
    return _array(x).astype(dtype)


def _axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    """`axis` (an int or a sequence of ints) as a tuple of distinct
    non-negative axes of a rank-`ndim` array; `None` means every axis."""
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if _is_integral(axis) else tuple(axis)
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


def square(x: ArrayLike) -> Array:
    """Elementwise `x * x`."""
    x = _array(x)
    return x * x


def reciprocal(x: ArrayLike) -> Array:
    """Elementwise `1 / x`."""
    return 1.0 / _array(x)


def sign(x: ArrayLike) -> Array:
    """Elementwise sign: -1 for negative values, 0 for zeros of either sign,
    +1 for positive values, and NaN for NaN. Built from comparisons, so its
    derivative is zero everywhere, as in JAX. The NaN comes from `x - x`
    rather than a NaN constant, which the CUDA and MLX backends reject, and
    its derivative cancels to zero too."""
    x = _array(x)
    signs = (x > 0.0).astype(x.dtype) - (x < 0.0).astype(x.dtype)
    return _where(x.isnan(), x - x, signs)


def clip(
    x: ArrayLike, lo: ArrayLike | None = None, hi: ArrayLike | None = None
) -> Array:
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


def silu(x: ArrayLike) -> Array:
    """SiLU (swish), `x * sigmoid(x)`, on the overflow-free `sigmoid`."""
    x = _array(x)
    return x * x.sigmoid()


# 2 * sqrt(2 / pi): the tanh-approximation argument, doubled for `sigmoid`.
_GELU_SCALE = 2.0 * math.sqrt(2.0 / math.pi)
_SQRT_HALF = math.sqrt(0.5)


def gelu(x: ArrayLike, approximate: bool = True) -> Array:
    """GELU with the tanh approximation
    `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, evaluated through the
    identity `0.5 (1 + tanh(u)) == sigmoid(2u)`: the stable `sigmoid` keeps
    full relative accuracy in the negative tail, where `1 + tanh(u)` would
    cancel to zero. `approximate=False` gives the exact
    `0.5 x (1 + erf(x / sqrt(2)))`, evaluated as `0.5 x erfc(-x / sqrt(2))`:
    `erfc` keeps full relative accuracy in the negative tail, where
    `1 + erf` would cancel to zero."""
    x = _array(x)
    if not approximate:
        return x * 0.5 * erfc(x * -_SQRT_HALF)
    return x * (x * (x * x * 0.044715 + 1.0) * _GELU_SCALE).sigmoid()


# Normalizations and reductions. The max shift cancels from every result
# below, so it is taken through `stop_gradient`: derivatives skip `max`
# and carry no rounding from the cancelled terms.


def softmax(x: ArrayLike, axis: int | Sequence[int] = -1) -> Array:
    """`exp(x) / sum(exp(x))` along `axis` (an int or a sequence of ints),
    computed as `exp(x - m) / sum(exp(x - m))` with `m` the maximum along
    `axis`, so no `exp` overflows: `softmax([1000, 0])` is `[1, 0]` with a
    finite gradient. As in JAX, a slice that holds `+inf` or only `-inf`
    gives NaN."""
    x = _array(x)
    unnormalized = (x - stop_gradient(x.max(axis=axis, keepdims=True))).exp()
    return unnormalized / unnormalized.sum(axis=axis, keepdims=True)


def log_softmax(x: ArrayLike, axis: int | Sequence[int] = -1) -> Array:
    """`log(softmax(x))` along `axis`, computed as
    `(x - m) - log(sum(exp(x - m)))` with `m` the maximum along `axis`: it
    neither overflows nor takes the log of an underflowed probability, so
    `log_softmax([1000, 0])` is `[0, -1000]`."""
    x = _array(x)
    shifted = x - stop_gradient(x.max(axis=axis, keepdims=True))
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def logsumexp(x: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Array:
    """`log(sum(exp(x)))` over `axis` (an int, a sequence of ints, or `None`
    for all axes), computed as `m + log(sum(exp(x - m)))` with `m` the
    maximum over `axis`. Where `m` is not finite the shift is zero instead,
    as in JAX: a slice of only `-inf` gives `-inf`, one holding `+inf` gives
    `+inf`, and NaN propagates. The gradient is `softmax(x)`."""
    x = _array(x)
    axes = None if axis is None else _axes(axis, len(x.shape))
    peak = stop_gradient(x.max(axis=axes, keepdims=True))
    shift = _where(peak.isfinite(), peak, 0.0)
    result = (x - shift).exp().sum(axis=axes, keepdims=True).log() + shift
    return result if keepdims else squeeze(result, axes)


def var(
    x: ArrayLike, axis: Axis = None, keepdims: bool = False, ddof: int = 0
) -> Array:
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


def std(
    x: ArrayLike, axis: Axis = None, keepdims: bool = False, ddof: int = 0
) -> Array:
    """Standard deviation, `sqrt(var(x, axis, keepdims, ddof))`. Its
    derivative at zero variance is zero, following the `sqrt` convention
    (JAX gives NaN there)."""
    return var(x, axis=axis, keepdims=keepdims, ddof=ddof).sqrt()


# Shape helpers: reshapes, slices, and broadcasts of their operand.


def squeeze(x: ArrayLike, axis: Axis = None) -> Array:
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


def expand_dims(x: ArrayLike, axis: int | Sequence[int]) -> Array:
    """`x` with unit axes inserted so that they sit at positions `axis` (an
    int or a sequence of ints) of the result."""
    x = _array(x)
    count = 1 if _is_integral(axis) else len(tuple(axis))
    ndim = len(x.shape) + count
    axes = _axes(axis, ndim)
    extents = iter(x.shape)
    return x.reshape([1 if a in axes else next(extents) for a in range(ndim)])


def split(
    x: ArrayLike, indices_or_sections: int | Sequence[int], axis: int = 0
) -> list[Array]:
    """`x` split along `axis` into a list of slices: `n` equal pieces for an
    int `n` (which must divide the extent), or the pieces between the given
    increasing split indices. A piece may not be empty, since quabla arrays
    have no zero-extent axes."""
    x = _array(x)
    (axis,) = _axes(axis, len(x.shape))
    extent = x.shape[axis]
    if _is_integral(indices_or_sections):
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


def meshgrid(*xs: ArrayLike, indexing: str = "xy") -> list[Array]:
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


def full_like(x: ArrayLike, fill_value: float, dtype: DType | None = None) -> Tensor:
    """A `Tensor` with the shape of `x` filled with `fill_value`, of the
    dtype of `x` unless `dtype` is given. For a traced `x` it is a constant
    of the trace: it does not depend on the values of `x`."""
    x = _array(x)
    return full(x.shape, fill_value, x.dtype if dtype is None else dtype)


def zeros_like(x: ArrayLike, dtype: DType | None = None) -> Tensor:
    """`full_like(x, 0, dtype)`."""
    return full_like(x, 0.0, dtype)


def ones_like(x: ArrayLike, dtype: DType | None = None) -> Tensor:
    """`full_like(x, 1, dtype)`."""
    return full_like(x, 1.0, dtype)


# jax.numpy-style compositions (v0.4). Shape arguments are static Python
# ints; every function takes eager and traced arrays alike.


def _axis(axis: Any, ndim: int) -> int:
    """A single axis as a non-negative int."""
    (value,) = _axes(_static_int(axis, "axis"), ndim)
    return value


def _along(axis: int, ndim: int, index: int | slice) -> tuple[int | slice, ...]:
    """An index tuple that applies `index` (an int or a slice) to `axis`."""
    return (slice(None),) * axis + (index,) + (slice(None),) * (ndim - axis - 1)


def _static_int(value: Any, name: str) -> int:
    if isinstance(value, _ARRAYS):
        raise TypeError(
            f"{name} must be a Python int: quabla has no integer arrays, and it "
            "determines the shape of the result"
        )
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None


def _static_ints(value: Any, name: str) -> tuple[int, ...]:
    """`value` (an int or a sequence of ints) as a tuple of ints."""
    if isinstance(value, _ARRAYS):
        # Raises: an array is never a static int.
        return (_static_int(value, name),)
    try:
        return (operator.index(value),)
    except TypeError:
        pass
    try:
        items = tuple(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an int or a sequence of ints, got {type(value).__name__}"
        ) from None
    return tuple(_static_int(item, name) for item in items)


def _operands(x1: ArrayLike, x2: ArrayLike) -> tuple[Array, Array]:
    """The two operands of an elementwise binary function as arrays; a
    Python number adopts the dtype of the array beside it."""
    if _is_number(x1) and not _is_number(x2):
        x2 = _array(x2)
        return asarray(x1, dtype=x2.dtype), x2
    x1 = _array(x1)
    if _is_number(x2):
        return x1, asarray(x2, dtype=x1.dtype)
    return x1, _array(x2)


def _finfo_max(dtype: DType) -> float:
    """The largest finite value of the float `dtype`."""
    return 3.4028234663852886e38 if dtype == float32 else 1.7976931348623157e308


def flip(m: ArrayLike, axis: Axis = None) -> Array:
    """`m` with the order of its entries reversed along `axis` (an int, a
    sequence of ints, or `None` for every axis), as a reversed slice."""
    x = _array(m)
    ndim = len(x.shape)
    axes = _axes(None if axis is None else _static_ints(axis, "axis"), ndim)
    index = tuple(
        slice(None, None, -1) if a in axes and x.shape[a] > 1 else slice(None)
        for a in range(ndim)
    )
    if builtins.all(item == slice(None) for item in index):
        return x
    return x[index]


def roll(a: ArrayLike, shift: int | Sequence[int], axis: Axis = None) -> Array:
    """`a` with its entries shifted cyclically by `shift` positions along
    `axis`; entries leaving one end re-enter at the other. With `axis=None`
    the flattened array is rolled and the shape restored. `shift` and `axis`
    may be equal-length sequences (or one of them an int, which is
    broadcast); shifts of a repeated axis add up, as in NumPy. Each rolled
    axis is two slices and a `concat`."""
    x = _array(a)
    if axis is None:
        shape = x.shape
        total = builtins.sum(_static_ints(shift, "shift"))
        return roll(x.reshape([-1]), total, 0).reshape(shape)
    shifts, axes = _static_ints(shift, "shift"), _static_ints(axis, "axis")
    if len(shifts) == 1:
        shifts = shifts * len(axes)
    elif len(axes) == 1:
        axes = axes * len(shifts)
    if len(shifts) != len(axes):
        raise ValueError(f"shift {shift!r} and axis {axis!r} have different lengths")
    ndim = len(x.shape)
    totals = {}
    for value, ax in zip(shifts, axes):
        ax = _axis(ax, ndim)
        totals[ax] = totals.get(ax, 0) + value
    for ax, value in totals.items():
        extent = x.shape[ax]
        split_at = extent - value % extent
        if split_at != extent:
            head = x[_along(ax, ndim, slice(split_at, None))]
            tail = x[_along(ax, ndim, slice(None, split_at))]
            x = concat([head, tail], ax)
    return x


def _pairs(value: Any, ndim: int, name: str) -> list[tuple[Any, Any]]:
    """`value` broadcast to one `(before, after)` pair per axis, following
    NumPy's `np.broadcast_to(value, (ndim, 2))`."""

    def pair(item: Any) -> tuple[Any, Any]:
        if _is_number(item):
            return (item, item)
        item = tuple(item)
        if len(item) in (1, 2):
            return (item[0], item[-1])
        raise ValueError(f"{name} {value!r} cannot be broadcast to shape ({ndim}, 2)")

    if isinstance(value, _ARRAYS):
        raise TypeError(f"{name} must be Python numbers or sequences of them, not an array")
    if _is_number(value):
        rows = [pair(value)]
    else:
        items = tuple(value)
        if builtins.all(_is_number(item) for item in items):
            rows = [pair(items)]
        else:
            rows = [pair(item) for item in items]
    if len(rows) == 1:
        rows = rows * ndim
    if len(rows) != ndim:
        raise ValueError(f"{name} {value!r} cannot be broadcast to shape ({ndim}, 2)")
    return rows


def _pad_index(i: int, n: int, mode: str) -> int:
    """The source index of position `i` (counted from the start of the
    unpadded axis of extent `n`, negative before it) under `mode`."""
    if mode == "edge":
        return builtins.min(builtins.max(i, 0), n - 1)
    if mode == "wrap":
        return i % n
    if mode == "reflect":
        # Mirrored about the end entries, which are not repeated: period 2(n-1).
        if n == 1:
            return 0
        j = i % (2 * n - 2)
        return j if j < n else 2 * n - 2 - j
    # "symmetric": mirrored about the array edges, end entries repeated.
    j = i % (2 * n)
    return j if j < n else 2 * n - 1 - j


_PAD_MODES = ("constant", "edge", "reflect", "symmetric", "wrap")


def pad(
    array: ArrayLike,
    pad_width: int | Sequence[Any],
    mode: str = "constant",
    constant_values: float | Sequence[Any] = 0,
) -> Array:
    """`array` padded along each axis by `pad_width` entries, with NumPy's
    broadcasting of `pad_width` (an int, `(before, after)`, or one pair per
    axis). Modes: `"constant"` (`constant_values`, numbers broadcast the
    same way), `"edge"`, `"reflect"`, `"symmetric"`, and `"wrap"`; the last
    three repeat their pattern periodically when the width exceeds the
    extent, as NumPy does. Constant padding concatenates constant blocks;
    the other modes are one static-index gather per padded axis, whose
    gradient sums the cotangents of every copy of an entry."""
    if mode not in _PAD_MODES:
        raise ValueError(f"mode must be one of {_PAD_MODES}, got {mode!r}")
    x = _array(array)
    ndim = len(x.shape)
    widths = [
        (_static_int(before, "pad_width"), _static_int(after, "pad_width"))
        for before, after in _pairs(pad_width, ndim, "pad_width")
    ]
    if builtins.any(width < 0 for pair in widths for width in pair):
        raise ValueError(f"pad_width must be non-negative, got {pad_width!r}")
    values = _pairs(constant_values, ndim, "constant_values") if mode == "constant" else []
    for axis, (before, after) in enumerate(widths):
        if not (before or after):
            continue
        extent = x.shape[axis]
        if mode != "constant":
            positions = range(-before, extent + after)
            x = x.gather([_pad_index(i, extent, mode) for i in positions], axis=axis)
            continue
        pieces = [x]
        for width, value, at_end in (
            (before, values[axis][0], False),
            (after, values[axis][1], True),
        ):
            if width:
                block = list(x.shape)
                block[axis] = width
                constant = full(block, value, x.dtype)
                pieces = pieces + [constant] if at_end else [constant] + pieces
        x = concat(pieces, axis)
    return x


def tile(A: ArrayLike, reps: int | Sequence[int]) -> Array:
    """`A` repeated `reps[i]` times along axis `i`, NumPy's `tile`: the
    shorter of `A.shape` and `reps` is padded with leading ones. One
    broadcast between two reshapes; every rep must be at least 1, since
    quabla arrays have no zero-extent axes."""
    x = _array(A)
    reps = _static_ints(reps, "reps")
    if builtins.any(r < 1 for r in reps):
        raise ValueError(f"reps must be positive (empty arrays are not supported), got {reps!r}")
    rank = builtins.max(len(reps), len(x.shape))
    shape = [1] * (rank - len(x.shape)) + list(x.shape)
    reps = (1,) * (rank - len(reps)) + reps
    if builtins.all(r == 1 for r in reps):
        return x if list(x.shape) == shape else x.reshape(shape)
    interleaved = [e for extent in shape for e in (1, extent)]
    target = [e for r, extent in zip(reps, shape) for e in (r, extent)]
    tiled = x.reshape(interleaved).broadcast_to(target)
    return tiled.reshape([r * extent for r, extent in zip(reps, shape)])


def repeat(
    a: ArrayLike, repeats: int | Sequence[int], axis: int | None = None
) -> Array:
    """Each entry of `a` repeated `repeats` times along `axis` (over the
    flattened array when `axis` is `None`). `repeats` is an int, broadcast
    to every entry (a broadcast between reshapes), or one non-negative int
    per entry along `axis` (a static-index gather). The counts are static
    Python ints and the result may not be empty."""
    x = _array(a)
    if axis is None:
        x = x.reshape([-1])
        axis = 0
    axis = _axis(axis, len(x.shape))
    counts = _static_ints(repeats, "repeats")
    extent = x.shape[axis]
    if len(counts) == 1:
        (count,) = counts
        if count < 1:
            raise ValueError(
                f"repeats must be positive (empty arrays are not supported), got {count}"
            )
        if count == 1:
            return x
        shape = list(x.shape)
        expanded = shape[: axis + 1] + [1] + shape[axis + 1 :]
        target = shape[: axis + 1] + [count] + shape[axis + 1 :]
        shape[axis] *= count
        return x.reshape(expanded).broadcast_to(target).reshape(shape)
    if len(counts) != extent:
        raise ValueError(f"repeats has {len(counts)} entries for axis {axis} with extent {extent}")
    if builtins.any(count < 0 for count in counts) or not builtins.any(counts):
        raise ValueError(f"repeats must be non-negative with a positive sum, got {counts!r}")
    return x.gather([i for i, count in enumerate(counts) for _ in range(count)], axis=axis)


def moveaxis(
    a: ArrayLike, source: int | Sequence[int], destination: int | Sequence[int]
) -> Array:
    """`a` with the axes `source` moved to the positions `destination`
    (ints or equal-length sequences of ints); the other axes keep their
    order."""
    x = _array(a)
    ndim = len(x.shape)
    sources = _axes(_static_ints(source, "source"), ndim)
    destinations = _axes(_static_ints(destination, "destination"), ndim)
    if len(sources) != len(destinations):
        raise ValueError("source and destination must have the same number of axes")
    order = [axis for axis in range(ndim) if axis not in sources]
    for destination_axis, source_axis in sorted(zip(destinations, sources)):
        order.insert(destination_axis, source_axis)
    return x if order == list(range(ndim)) else x.transpose(order)


def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> Array:
    """`a` with axes `axis1` and `axis2` interchanged."""
    x = _array(a)
    ndim = len(x.shape)
    first, second = _axis(axis1, ndim), _axis(axis2, ndim)
    if first == second:
        return x
    order = list(range(ndim))
    order[first], order[second] = second, first
    return x.transpose(order)


def ravel(a: ArrayLike) -> Array:
    """`a` flattened to one axis in row-major order (a 0-D array gives
    shape `[1]`)."""
    return _array(a).reshape([-1])


def diagonal(a: ArrayLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
    """The diagonal `offset` (above the main one when positive, below when
    negative) of the matrices spanned by `axis1` and `axis2`, as NumPy: both
    axes are removed and the diagonal becomes the last axis. The square
    block holding the diagonal is flattened and read with a stride of
    `length + 1`, one static-index gather. An empty diagonal raises."""
    x = _array(a)
    ndim = len(x.shape)
    if ndim < 2:
        raise ValueError(f"diagonal requires an array of rank 2 or more, got shape {list(x.shape)}")
    first, second = _axis(axis1, ndim), _axis(axis2, ndim)
    if first == second:
        raise ValueError("axis1 and axis2 cannot be the same")
    offset = _static_int(offset, "offset")
    order = [axis for axis in range(ndim) if axis not in (first, second)] + [first, second]
    if order != list(range(ndim)):
        x = x.transpose(order)
    rows, columns = x.shape[-2], x.shape[-1]
    row, column = builtins.max(-offset, 0), builtins.max(offset, 0)
    length = builtins.min(rows - row, columns - column)
    if length <= 0:
        raise ValueError(f"offset {offset} gives an empty diagonal of a {rows} x {columns} matrix")
    block = x[..., row : row + length, column : column + length]
    flat = block.reshape(list(block.shape[:-2]) + [length * length])
    return flat if length == 1 else flat[..., :: length + 1]


def trace(a: ArrayLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
    """`trace(a, offset=0, axis1=0, axis2=1)`: the sum along a diagonal,
    `diagonal(a, offset, axis1, axis2).sum(-1)`, as NumPy."""
    return diagonal(a, offset=offset, axis1=axis1, axis2=axis2).sum(axis=-1)


def diag(v: ArrayLike, k: int = 0) -> Array:
    """For a 1-D `v`, the square matrix with `v` on diagonal `k` and zeros
    elsewhere, a `where` over a constant diagonal mask (so an infinite or
    NaN entry does not leak off the diagonal, as it would through a product
    with the identity); for a 2-D `v`, its diagonal `k` (see `diagonal`)."""
    x = _array(v)
    k = _static_int(k, "k")
    if len(x.shape) == 2:
        return diagonal(x, k)
    if len(x.shape) != 1:
        raise ValueError(f"diag requires a 1-D or 2-D input, got shape {list(x.shape)}")
    size = x.shape[0] + builtins.abs(k)
    if k:
        x = concat([x, full([builtins.abs(k)], 0.0, x.dtype)], 0)
    # Entry (i, i + k) holds v[i]: for k >= 0 the value is indexed by the
    # row, for k < 0 by the column, so `v` padded to `size` is broadcast
    # along the other axis and masked to the diagonal.
    values = x.reshape([size, 1] if k >= 0 else [1, size]).broadcast_to([size, size])
    index = arange(float(size))
    mask = (index.reshape([1, size]) - index.reshape([size, 1])).equal(float(k))
    return _where(mask, values, 0.0)


# Products.


def outer(a: ArrayLike, b: ArrayLike) -> Array:
    """The outer product of `a` and `b`, both flattened:
    `result[i, j] = a[i] * b[j]`."""
    a, b = ravel(a), ravel(b)
    return a.reshape([a.shape[0], 1]) * b.reshape([1, b.shape[0]])


def tensordot(
    a: ArrayLike, b: ArrayLike, axes: int | Sequence[int | Sequence[int]] = 2
) -> Array:
    """The sum of products of `a` and `b` over the axes `axes`: an int `n`
    pairs the last `n` axes of `a` with the first `n` of `b`; a pair
    `(axes_a, axes_b)` of ints or equal-length sequences names them. The
    result has the free axes of `a`, then those of `b`. One `matmul`
    between transposes and reshapes."""
    a, b = _array(a), _array(b)
    rank_a, rank_b = len(a.shape), len(b.shape)
    if _is_integral(axes):
        count = operator.index(axes)
        if not 0 <= count <= builtins.min(rank_a, rank_b):
            raise ValueError(
                f"axes={count} is out of range for operands of rank {rank_a} and {rank_b}"
            )
        axes_a, axes_b = list(range(rank_a - count, rank_a)), list(range(count))
    else:
        try:
            axes_a, axes_b = axes
        except (TypeError, ValueError):
            raise ValueError(
                f"axes must be an int or a pair of axis sequences, got {axes!r}"
            ) from None
        axes_a = list(_axes(_static_ints(axes_a, "axes"), rank_a))
        axes_b = list(_axes(_static_ints(axes_b, "axes"), rank_b))
    if len(axes_a) != len(axes_b):
        raise ValueError(f"axes {axes!r} pair {len(axes_a)} axes of a with {len(axes_b)} of b")
    for axis_a, axis_b in zip(axes_a, axes_b):
        if a.shape[axis_a] != b.shape[axis_b]:
            raise ValueError(
                f"shape mismatch for sum: axis {axis_a} of a has extent {a.shape[axis_a]}, "
                f"axis {axis_b} of b has extent {b.shape[axis_b]}"
            )
    free_a = [axis for axis in range(rank_a) if axis not in axes_a]
    free_b = [axis for axis in range(rank_b) if axis not in axes_b]
    shape_a = [a.shape[axis] for axis in free_a]
    shape_b = [b.shape[axis] for axis in free_b]
    inner = math.prod(a.shape[axis] for axis in axes_a)
    order_a, order_b = free_a + axes_a, axes_b + free_b
    if order_a != list(range(rank_a)):
        a = a.transpose(order_a)
    if order_b != list(range(rank_b)):
        b = b.transpose(order_b)
    product = a.reshape([math.prod(shape_a), inner]) @ b.reshape([inner, math.prod(shape_b)])
    return product.reshape(shape_a + shape_b)


def dot(a: ArrayLike, b: ArrayLike) -> Array:
    """NumPy's `dot`: with a 0-D operand it is the elementwise product; with
    a 1-D operand, or two operands of rank at most 2, it is `matmul`;
    otherwise it sums over the last axis of `a` and the second-to-last of
    `b`, giving the free axes of `a` followed by those of `b` (a
    `tensordot`, where `matmul` would batch over the leading axes)."""
    a, b = _array(a), _array(b)
    rank_a, rank_b = len(a.shape), len(b.shape)
    if rank_a == 0 or rank_b == 0:
        return a * b
    if rank_a == 1 or rank_b == 1 or (rank_a == 2 and rank_b == 2):
        return matmul(a, b)
    return tensordot(a, b, ([rank_a - 1], [rank_b - 2]))


def kron(a: ArrayLike, b: ArrayLike) -> Array:
    """The Kronecker product: the operand of smaller rank gets leading unit
    axes, and the result has extent `a.shape[i] * b.shape[i]` per axis.
    One broadcast product of interleaved reshapes."""
    a, b = _array(a), _array(b)
    rank = builtins.max(len(a.shape), len(b.shape))
    if rank == 0:
        return a * b
    shape_a = [1] * (rank - len(a.shape)) + list(a.shape)
    shape_b = [1] * (rank - len(b.shape)) + list(b.shape)
    spread_a = a.reshape([e for extent in shape_a for e in (extent, 1)])
    spread_b = b.reshape([e for extent in shape_b for e in (1, extent)])
    return (spread_a * spread_b).reshape([x * y for x, y in zip(shape_a, shape_b)])


def cross(
    a: ArrayLike,
    b: ArrayLike,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> Array:
    """The cross product of the vectors along `axisa` of `a` and `axisb` of
    `b` (`axis` sets all three axes), with the other axes broadcast, as
    NumPy; the result's vector axis is `axisc`. A 2-vector has a zero third
    component; two 2-vectors give the scalar z component without a vector
    axis (deprecated in NumPy 2, kept by JAX)."""
    if axis is not None:
        axisa = axisb = axisc = axis
    a, b = moveaxis(a, axisa, -1), moveaxis(b, axisb, -1)
    size_a, size_b = a.shape[-1], b.shape[-1]
    if size_a not in (2, 3) or size_b not in (2, 3):
        raise ValueError(
            "incompatible dimensions for cross product (dimension must be 2 or 3), "
            f"got {size_a} and {size_b}"
        )
    a0, a1, b0, b1 = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    z = a0 * b1 - a1 * b0
    if size_a == size_b == 2:
        return z
    if size_a == 2:
        b2 = b[..., 2]
        x, y = a1 * b2, -(a0 * b2)
    elif size_b == 2:
        a2 = a[..., 2]
        x, y = -(a2 * b1), a2 * b0
    else:
        a2, b2 = a[..., 2], b[..., 2]
        x, y = a1 * b2 - a2 * b1, a2 * b0 - a0 * b2
    shape = list(z.shape)
    # Broadcast every component to the common batch shape before stacking.
    components = [c if list(c.shape) == shape else c.broadcast_to(shape) for c in (x, y, z)]
    return moveaxis(stack(components, -1), -1, axisc)


# Calculus and elementwise numerics.


def _edge_values(value: ArrayLike, x: Array, axis: int, name: str) -> Array:
    """A `prepend`/`append` operand of `diff`: a number or a 0-D array is
    broadcast to `x`'s shape with extent 1 along `axis`."""
    shape = list(x.shape)
    shape[axis] = 1
    if _is_number(value):
        return full(shape, value, x.dtype)
    value = _array(value)
    if len(value.shape) == 0:
        return value.broadcast_to(shape)
    if len(value.shape) != len(shape):
        raise ValueError(f"{name} must be a scalar or have the rank of the input")
    return value


def diff(
    a: ArrayLike,
    n: int = 1,
    axis: int = -1,
    prepend: ArrayLike | None = None,
    append: ArrayLike | None = None,
) -> Array:
    """The `n`-th forward difference along `axis`, `a[1:] - a[:-1]` applied
    `n` times, after `prepend` and `append` (numbers or arrays) are joined
    to `a` along `axis`, as NumPy; boolean input gives `a[1:] != a[:-1]`.
    The result may not be empty, so `n` must be below the extent."""
    x = _array(a)
    n = _static_int(n, "n")
    if n < 0:
        raise ValueError(f"order must be non-negative, got {n}")
    ndim = len(x.shape)
    if ndim == 0:
        raise ValueError("diff requires an input of rank 1 or more")
    axis = _axis(axis, ndim)
    if prepend is not None or append is not None:
        pieces = [x]
        if prepend is not None:
            pieces.insert(0, _edge_values(prepend, x, axis, "prepend"))
        if append is not None:
            pieces.append(_edge_values(append, x, axis, "append"))
        x = concat(pieces, axis)
    if n >= x.shape[axis] and n:
        raise ValueError(
            f"diff of order {n} along an axis of extent {x.shape[axis]} would be empty"
        )
    for _ in range(n):
        upper = x[_along(axis, ndim, slice(1, None))]
        lower = x[_along(axis, ndim, slice(None, -1))]
        x = upper.not_equal(lower) if x.dtype == bool_ else upper - lower
    return x


def trapezoid(
    y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0, axis: int = -1
) -> Array:
    """The integral of `y` along `axis` by the trapezoidal rule,
    `sum(d * (y[1:] + y[:-1]) / 2)`, with `d = diff(x)` for sample points
    `x` (1-D along `axis`, or of `y`'s shape) or the spacing `dx` otherwise,
    as NumPy. An axis of extent 1 integrates to zero."""
    y = _array(y)
    ndim = len(y.shape)
    if ndim == 0:
        raise ValueError("trapezoid requires an input of rank 1 or more")
    axis = _axis(axis, ndim)
    if y.shape[axis] < 2:
        total = y.sum(axis=axis)
        # Zero, but computed from `y` so that traced code keeps a traced result.
        return _where(total.isnan().logical_or(total.isnan().logical_not()), 0.0, total)
    if x is None:
        d = dx
    else:
        x = _array(x)
        if len(x.shape) == 1:
            d = diff(x)
            shape = [1] * ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    upper = y[_along(axis, ndim, slice(1, None))]
    lower = y[_along(axis, ndim, slice(None, -1))]
    return (d * (upper + lower) / 2.0).sum(axis=axis)


def interp(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: ArrayLike | None = None,
    right: ArrayLike | None = None,
) -> Array:
    """The piecewise-linear interpolant through `(xp, fp)` at `x`, as NumPy:
    `fp[0]` (or `left`) below `xp[0]`, `fp[-1]` (or `right`) above
    `xp[-1]`, and NaN at a NaN `x`. `xp` must be strictly increasing, which
    is checked for an eager `xp` only.

    Without integer arrays there is no `searchsorted`, so every interval is
    evaluated: interval `i` contributes `fp[i] (1 - t) + fp[i+1] t`, with
    `t = (x - xp[i]) / (xp[i+1] - xp[i])`, where `xp[i] <= x < xp[i+1]`
    (the last interval closed) and zero elsewhere. That costs
    O(x.size * len(xp)) work and memory; it is exact at the knots and
    differentiable in `x`, `xp`, and `fp`. At an interior knot the
    derivative in `x` is the slope of the interval to its right, at `xp[-1]`
    that of the last interval, as in JAX. Outside interval `i`, `x` is
    replaced by `xp[i]` before `t` is formed, so an infinite `x` puts no NaN
    into the gradients."""
    x, xp, fp = _array(x), _array(xp), _array(fp)
    if len(xp.shape) != 1 or len(fp.shape) != 1 or xp.shape != fp.shape:
        raise ValueError(
            f"xp and fp must be 1-D of the same length, got {list(xp.shape)} and {list(fp.shape)}"
        )
    count = xp.shape[0]
    if isinstance(xp, Tensor) and count > 1 and not (xp[1:] > xp[:-1]).all().item():
        raise ValueError("interp requires a strictly increasing xp")
    if count == 1:
        value = fp[0]
    else:
        column = x.reshape(list(x.shape) + [1])
        lo, hi = xp[:-1], xp[1:]
        last = asarray([False] * (count - 2) + [True])
        below_hi = (column < hi).logical_or(column.equal(hi).logical_and(last))
        inside = (column >= lo).logical_and(below_hi)
        t = (_where(inside, column, lo) - lo) / (hi - lo)
        pieces = fp[:-1] * (1.0 - t) + fp[1:] * t
        value = _where(inside, pieces, 0.0).sum(axis=-1)
    value = _where(x > xp[-1], fp[-1] if right is None else right, value)
    value = _where(x < xp[0], fp[0] if left is None else left, value)
    return _where(x.isnan(), x, value)


def polyval(p: ArrayLike, x: ArrayLike) -> Array:
    """The polynomial with coefficients `p` (highest degree first) at `x`,
    by Horner's rule; `p` is a 1-D array or a sequence of numbers (Python
    numbers keep the dtype of `x`). The loop starts from `p[0]`, so an
    infinite `x` gives the polynomial's infinite limit where NumPy and JAX,
    which start from `0 * x`, give NaN; a constant `p` gives NaN there in
    all three."""
    x = _array(x)
    if isinstance(p, (list, tuple)) and builtins.all(_is_number(c) for c in p):
        coefficients = list(p)
    else:
        p = _array(p)
        if len(p.shape) != 1:
            raise ValueError(f"p must be 1-D, got shape {list(p.shape)}")
        coefficients = [p[i] for i in range(p.shape[0])]
    if not coefficients:
        raise ValueError("p must hold at least one coefficient")
    if len(coefficients) == 1:
        return x * 0.0 + coefficients[0]
    result = x * coefficients[0] + coefficients[1]
    for coefficient in coefficients[2:]:
        result = result * x + coefficient
    return result


def logaddexp(x1: ArrayLike, x2: ArrayLike) -> Array:
    """`log(exp(x1) + exp(x2))` elementwise, as `hi + log1p(exp(lo - hi))`
    with `hi` and `lo` the larger and smaller operand: `exp` cannot
    overflow and the sum keeps full accuracy. The expression is smooth in
    `(hi, lo)`, so the gradient is `(sigmoid(x1 - x2), sigmoid(x2 - x1))`,
    split evenly at ties. Where `lo - hi` is NaN (infinities of the same
    sign, or a NaN operand) the result is `x1 + x2`, as in JAX: two `-inf`
    give `-inf`."""
    a, b = _operands(x1, x2)
    first = a >= b
    hi, lo = _where(first, a, b), _where(first, b, a)
    delta = lo - hi
    return _where(delta.isnan(), a + b, hi + delta.exp().log1p())


def hypot(x1: ArrayLike, x2: ArrayLike) -> Array:
    """`sqrt(x1**2 + x2**2)` elementwise, scaled by `m = max(|x1|, |x2|)`
    as `m * sqrt((x1/m)**2 + (x2/m)**2)`, so it neither overflows nor
    underflows where the squares would (float32 beyond about 1e19 or below
    about 1e-19). `m` enters through `stop_gradient`, which leaves the
    gradient exactly `(x1, x2) / hypot`; at the origin it is zero. An
    infinite operand gives `inf` even beside a NaN, as NumPy."""
    a, b = _operands(x1, x2)
    absolute_a, absolute_b = a.abs(), b.abs()
    scale = stop_gradient(maximum(absolute_a, absolute_b))
    scale = _where(scale > 0.0, scale, 1.0)
    result = scale * (square(a / scale) + square(b / scale)).sqrt()
    infinite_a = isinf(a)
    infinite = infinite_a.logical_or(isinf(b))
    return _where(infinite, _where(infinite_a, absolute_a, absolute_b), result)


def exp2(x: ArrayLike) -> Array:
    """`2 ** x` elementwise, through `power` rather than `exp(x * ln 2)`, so
    an integer `x` gives the exact power of two."""
    return power(2.0, x)


def isinf(x: ArrayLike) -> Array:
    """Elementwise test for positive or negative infinity (neither finite
    nor NaN), as `bool_`."""
    x = _array(x)
    return x.isfinite().logical_not().logical_and(x.isnan().logical_not())


def nan_to_num(
    x: ArrayLike,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> Array:
    """`x` with NaN replaced by `nan`, `+inf` by `posinf`, and `-inf` by
    `neginf`, which default to the largest and the most negative finite
    value of the dtype, as NumPy. `copy` is accepted for NumPy
    compatibility: arrays are immutable, so the result is always new. The
    gradient passes through finite entries and is zero at replaced ones."""
    del copy
    x = _array(x)
    if x.dtype == bool_:
        return x
    largest = _finfo_max(x.dtype)
    value = _where(x > largest, largest if posinf is None else posinf, x)
    value = _where(x < -largest, -largest if neginf is None else neginf, value)
    return _where(x.isnan(), nan, value)
