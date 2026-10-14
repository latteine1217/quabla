# Public `quabla` package over the compiled extension `quabla._quabla`.
#
# The package re-exports the extension's public names (`_NATIVE`) and the
# pure-Python layers built on them. The extension's other names are the
# private helpers those layers call.

import importlib.metadata as _metadata

from . import _array, _custom, _errors, _io, _newton, _ops, _quabla, _transforms
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

_NATIVE = [
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
globals().update({name: getattr(_quabla, name) for name in _NATIVE})

__doc__ = _quabla.__doc__

# The version of the installed distribution. The Cargo manifest of
# `quabla-python` is the single source of truth: maturin derives the package
# metadata version from it (`dynamic = ["version"]` in pyproject.toml), and
# the extension carries the same string, which serves a source tree that is
# importable without installed metadata.
try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    __version__ = _quabla.__version__

__all__ = list(
    dict.fromkeys(
        _NATIVE
        + _array.__all__
        + _errors.__all__
        + _io.__all__
        + _newton.__all__
        + _ops.__all__
        + _transforms.__all__
        + _custom.__all__
        + [
            "ShapeDtype",
            "devices",
            "optim",
            "distributed",
            "ode",
            "linalg",
            "cond",
            "fori_loop",
            "scan",
            "while_loop",
        ]
    )
)
