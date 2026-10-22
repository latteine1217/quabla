"""Type aliases shared by the annotations of the public API.

Annotations only: each module that uses them imports this under
`typing.TYPE_CHECKING` with `from __future__ import annotations`, so the
annotations stay unevaluated and cost nothing on import.
"""

from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

from ._quabla import Tensor, TensorView, TraceTensor, dtype


class SupportsArray(Protocol):
    """A NumPy array, or any object that converts to one through
    `__array__` (NumPy is optional, so it is not imported)."""

    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...


# A value that converts to a `Tensor`: an eager quabla array, a Python scalar
# (a weak scalar that adopts the other operand's dtype), nested sequences of
# numbers, or an object with `__array__`.
ConcreteLike: TypeAlias = (
    Tensor | TensorView | bool | int | float | Sequence[Any] | SupportsArray
)
# A value an array operation accepts: a concrete value or a traced array.
ArrayLike: TypeAlias = ConcreteLike | TraceTensor
# The result of an array operation: eager under no transform, traced inside
# one.
Array: TypeAlias = Tensor | TraceTensor
DType: TypeAlias = dtype
Shape: TypeAlias = int | Sequence[int]
Axis: TypeAlias = int | Sequence[int] | None
# Nested lists, tuples, dicts, registered nodes, and `None` around array
# leaves; not expressible precisely in the type system.
PyTree: TypeAlias = Any


class Reduction(Protocol):
    """A reduction function such as `quabla.sum`."""

    def __call__(
        self, x: ArrayLike, axis: Axis = None, keepdims: bool = False
    ) -> Array: ...
