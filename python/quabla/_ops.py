"""Module-level math functions (docs/api_v0_2_design.md, section 3.2, D4).

Each function calls the method of the same name on `Tensor` or
`TraceTensor`, so eager and traced code share one spelling: `qb.sin(x)` is
`x.sin()`, with the method's dtype rules and errors. Other operands (Python
numbers, nested lists, NumPy arrays) go through `asarray` first, except that
the second operand of `maximum`/`minimum` stays a Python number when it is
one, keeping the weak scalar typing of the method form:
`qb.maximum(x, 0.0)` keeps the dtype of `x`.

`abs`, `sum`, `max`, `min`, `any`, and `all` shadow Python builtins, so they
are attributes of `quabla` but not listed in `__all__`: `from quabla import *`
leaves the builtins alone.
"""

import numbers

from ._array import _shape, asarray
from ._quabla import Tensor, TraceTensor

__all__ = [
    "astype",
    "broadcast_to",
    "cholesky",
    "cos",
    "exp",
    "log",
    "matmul",
    "maximum",
    "mean",
    "minimum",
    "norm",
    "relu",
    "reshape",
    "sigmoid",
    "sin",
    "softplus",
    "solve",
    "sqrt",
    "tanh",
    "transpose",
    "tril",
    "triu",
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


def matmul(x1, x2):
    """Matrix product; `quabla.matmul(a, b)` is `a @ b`."""
    return _array(x1).matmul(_array(x2))


def solve(a, b):
    """Solves `a @ x == b` for `x`; `quabla.solve(a, b)` is `a.solve(b)`."""
    return _array(a).solve(_array(b))


def reshape(x, shape):
    """`x` with a new shape of the same size (an int or a sequence of ints)."""
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
