# Public `quabla` package over the compiled extension `quabla._quabla`.
#
# The package re-exports the extension's public names and the
# pure-Python layers built on them. The extension's other names are the
# private helpers those layers call.

import importlib.metadata as _metadata
import sys as _sys
import warnings as _warnings
from typing import TYPE_CHECKING as _TYPE_CHECKING

from . import _array, _control, _custom, _deprecated, _errors, _io, _newton, _ops, _quabla, _transforms
from . import tree as tree
# Kept out of `__all__`, so a star import does not shadow the standard
# library module `random`.
from . import random as random
from . import optim as optim
from . import distributed as distributed
from . import ode as ode
from . import linalg as linalg
from ._control import *  # noqa: F403
from ._devices import ShapeDtype as ShapeDtype
from ._devices import devices as devices
from ._array import *  # noqa: F403
from ._errors import *  # noqa: F403
from ._io import *  # noqa: F403
from ._newton import *  # noqa: F403
from ._ops import *  # noqa: F403

# Re-exported as attributes but kept out of `__all__`, so a star import
# does not shadow the builtins of the same names.
from ._ops import abs as abs
from ._ops import all as all
from ._ops import any as any
from ._ops import max as max
from ._ops import min as min
from ._ops import round as round
from ._ops import sum as sum
from ._transforms import *  # noqa: F403
from ._custom import *  # noqa: F403

from ._quabla import (
    Compiler,
    Executable,
    Program,
    Tensor,
    TensorView,
    TraceTensor,
    bool_,
    concat,
    dtype,
    einsum,
    equal,
    float32,
    float64,
    greater,
    greater_equal,
    isfinite,
    isnan,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    not_equal,
    stack,
    where,
)

__doc__ = _quabla.__doc__

# Type checkers see the deprecated names; at run time `__getattr__`
# resolves them so that each explicit use warns.
if _TYPE_CHECKING:
    from ._quabla import TensorCudaAdamOptimizer as TensorCudaAdamOptimizer
    from ._quabla import TensorMlxAdamOptimizer as TensorMlxAdamOptimizer

# The version of the installed distribution. The Cargo manifest of
# `quabla-python` is the single source of truth: maturin derives the package
# metadata version from it (`dynamic = ["version"]` in pyproject.toml), and
# the extension carries the same string, which serves a source tree that is
# importable without installed metadata.
try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    __version__ = _quabla.__version__

# Static forms that type checkers follow, so the re-exports are public.
__all__ = [
    "Compiler",
    "Executable",
    "Program",
    "Tensor",
    "TensorCudaAdamOptimizer",
    "TensorMlxAdamOptimizer",
    "TensorView",
    "TraceTensor",
    "bool_",
    "concat",
    "dtype",
    "einsum",
    "equal",
    "float32",
    "float64",
    "greater",
    "greater_equal",
    "isfinite",
    "isnan",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "not_equal",
    "stack",
    "where",
]
__all__ += _array.__all__
__all__ += _errors.__all__
__all__ += _io.__all__
__all__ += _newton.__all__
__all__ += _ops.__all__
__all__ += _transforms.__all__
__all__ += _custom.__all__
__all__ += [
    "ShapeDtype",
    "devices",
    "optim",
    "distributed",
    "ode",
    "linalg",
]
__all__ += _control.__all__


def __getattr__(name):
    if name in _deprecated.REPLACEMENTS:
        # `from quabla import *` binds the deprecated names silently, so that
        # only explicit use warns.
        if not _deprecated.is_silent_lookup(_sys._getframe(1)):
            _warnings.warn(_deprecated.message(name), DeprecationWarning, stacklevel=2)
        return getattr(_quabla, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_deprecated.REPLACEMENTS))
