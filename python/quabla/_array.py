"""Array construction for the v0.2 API (docs/api_v0_2_design.md, section 3.2).

`array`/`asarray` accept Python scalars, rectangular nested lists and tuples,
NumPy arrays and scalars, objects with the buffer protocol or `__array__`,
and quabla arrays. dtype inference (design D3): NumPy `float64`, `float32`,
and `bool_` keep their dtype, integers become `float64` (there is no integer
dtype), `float16` raises `TypeError`; Python numbers and lists become
`float64`, or `bool_` when every element is a Python `bool`. Imported data is
always copied into owned storage, so a tensor never aliases a NumPy array.

NumPy stays optional (design D2): nothing here imports it.
"""

from __future__ import annotations

import abc
import operator
from typing import TYPE_CHECKING, Any, overload

from ._quabla import Tensor, TensorView, TraceTensor

if TYPE_CHECKING:
    from ._typing import Array as _Array
    from ._typing import ArrayLike, ConcreteLike, DType, Shape

__all__ = [
    "Array",
    "arange",
    "array",
    "asarray",
    "eye",
    "full",
    "linspace",
    "ones",
    "zeros",
]


if TYPE_CHECKING:
    # Type checkers do not follow `ABC.register`, so they see the union the
    # runtime class stands for.
    Array = Tensor | TensorView | TraceTensor
else:

    class Array(abc.ABC):
        """Abstract base of the quabla array types, for `isinstance` checks and
        annotations: `Tensor`, `TensorView`, and `TraceTensor` are registered
        (design D1 keeps `Tensor` as the concrete class name)."""

        __slots__ = ()

    Array.register(Tensor)
    Array.register(TensorView)
    Array.register(TraceTensor)

# Handled by the nested-sequence parser in the extension. NumPy scalars are
# not listed: they take the buffer path, which keeps float32 and bool.
_NESTED_TYPES = (list, tuple, bool, int, float)


def _convert(obj: Any, dtype: DType | None) -> _Array:
    if isinstance(obj, (Tensor, TraceTensor)):
        return obj if dtype is None or obj.dtype == dtype else obj.astype(dtype)
    if isinstance(obj, TensorView):
        return _convert(obj.to_tensor(), dtype)
    if isinstance(obj, _NESTED_TYPES):
        return Tensor._from_nested(obj, dtype)
    tensor = Tensor._from_buffer(obj, dtype)
    if tensor is None and hasattr(type(obj), "__array__"):
        tensor = Tensor._from_buffer(obj.__array__(), dtype)
    if tensor is None:
        raise TypeError(
            f"cannot convert {type(obj).__name__} to a quabla Tensor; expected a number, "
            "nested lists or tuples of numbers, a NumPy array, or an object with the buffer "
            "protocol or __array__"
        )
    return tensor


@overload
def array(obj: TraceTensor, dtype: DType | None = None) -> TraceTensor: ...
@overload
def array(obj: ConcreteLike, dtype: DType | None = None) -> Tensor: ...
@overload
def array(obj: ArrayLike, dtype: DType | None = None) -> _Array: ...
def array(obj: ArrayLike, dtype: DType | None = None) -> _Array:
    """A new `Tensor` holding the values of `obj`, converted to `dtype` if
    given. Always returns a new object; storage is immutable, so a `Tensor`
    input shares its values instead of copying them. A `TraceTensor` is
    returned as is (or cast), since traced values are not data."""
    result = _convert(obj, dtype)
    if result is obj and isinstance(obj, Tensor):
        result = obj.astype(obj.dtype)
    return result


@overload
def asarray(obj: TraceTensor, dtype: DType | None = None) -> TraceTensor: ...
@overload
def asarray(obj: ConcreteLike, dtype: DType | None = None) -> Tensor: ...
@overload
def asarray(obj: ArrayLike, dtype: DType | None = None) -> _Array: ...
def asarray(obj: ArrayLike, dtype: DType | None = None) -> _Array:
    """Like `array`, but returns a `Tensor` (or `TraceTensor`) input
    unchanged when it already has `dtype`."""
    return _convert(obj, dtype)


def _shape(shape: Any) -> list[int]:
    """A shape as a list of extents; a bare integer is a rank-1 shape."""
    if isinstance(shape, (list, tuple)):
        return [operator.index(extent) for extent in shape]
    return [operator.index(shape)]


def _typed(tensor: Tensor, dtype: DType | None) -> Tensor:
    return tensor if dtype is None or tensor.dtype == dtype else tensor.astype(dtype)


def zeros(shape: Shape, dtype: DType | None = None) -> Tensor:
    """A tensor of zeros; `dtype` defaults to `float64`."""
    return _typed(Tensor.zeros(_shape(shape)), dtype)


def ones(shape: Shape, dtype: DType | None = None) -> Tensor:
    """A tensor of ones; `dtype` defaults to `float64`."""
    return _typed(Tensor.ones(_shape(shape)), dtype)


@overload
def full(
    shape: Shape, fill_value: TraceTensor, dtype: DType | None = None
) -> TraceTensor: ...
@overload
def full(
    shape: Shape, fill_value: ConcreteLike, dtype: DType | None = None
) -> Tensor: ...
@overload
def full(shape: Shape, fill_value: ArrayLike, dtype: DType | None = None) -> _Array: ...
def full(shape: Shape, fill_value: ArrayLike, dtype: DType | None = None) -> _Array:
    """A tensor filled with `fill_value`, whose dtype is inferred as in
    `asarray` unless `dtype` is given."""
    return asarray(fill_value, dtype).broadcast_to(_shape(shape))


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: DType | None = None,
) -> Tensor:
    """Values from `start` (inclusive) to `stop` (exclusive) by `step`;
    `arange(n)` counts from zero. The range must be non-empty."""
    if stop is None:
        start, stop = 0.0, start
    return _typed(Tensor.arange(start, stop, step), dtype)


def linspace(start: float, stop: float, num: int, dtype: DType | None = None) -> Tensor:
    """`num` evenly spaced values from `start` to `stop`, both inclusive."""
    return _typed(Tensor.linspace(start, stop, num), dtype)


def eye(n: int, m: int | None = None, dtype: DType | None = None) -> Tensor:
    """An `n` x `m` identity matrix (`m` defaults to `n`)."""
    return _typed(Tensor.eye(n, m), dtype)
