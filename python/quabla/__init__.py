# Public `quabla` package over the compiled extension `quabla._quabla`.
#
# The v0.1 wheel shipped a maturin-generated shim that star-imported the
# native submodule `quabla.quabla`. This package keeps that public surface
# unchanged: every extension name is re-exported, `__all__` and `__doc__`
# mirror the extension as before, and `quabla.quabla` stays available both
# as an attribute and as an importable module path. Later v0.2 layers are
# pure Python on top of these names.

import sys as _sys

from . import _array, _errors, _ops, _quabla, _transforms
from . import tree as tree
# Kept out of `__all__`, so a star import does not shadow the standard
# library module `random`.
from . import random as random
from . import optim as optim
from . import distributed as distributed
from . import legacy as legacy
from . import ode as ode
from . import _compat
from ._control import *  # noqa: F403
from ._devices import ShapeDtype as ShapeDtype
from ._devices import devices as devices
from ._array import *  # noqa: F403
from ._errors import *  # noqa: F403
from ._ops import *  # noqa: F403

# Re-exported as attributes but kept out of `__all__`, so a star import
# does not shadow the builtins of the same names.
from ._ops import abs as abs
from ._ops import all as all
from ._ops import any as any
from ._ops import max as max
from ._ops import min as min
from ._ops import sum as sum
from ._quabla import *  # noqa: F403

# After the extension's names: the v0.2 `grad` and `jit` replace the v0.1
# functions of the same names and dispatch the v0.1 call forms to them (D17).
from ._transforms import *  # noqa: F403

__doc__ = _quabla.__doc__
# The extension's names followed by the pure-Python v0.2 layers; a new list,
# so these additions do not mutate the extension's own `__all__`.
__all__ = list(
    dict.fromkeys(
        list(_quabla.__all__)
        + _array.__all__
        + _errors.__all__
        + _ops.__all__
        + _transforms.__all__
        + [
            "ShapeDtype",
            "devices",
            "optim",
            "distributed",
            "legacy",
            "ode",
            "cond",
            "fori_loop",
            "scan",
        ]
    )
)

# v0.1 compatibility alias for the native module. Registering it in
# sys.modules keeps `import quabla.quabla` and `from quabla.quabla import X`
# working, not only attribute access.
_sys.modules[__name__ + ".quabla"] = _quabla

# Resolve migrated names lazily so a package import does not warn on behalf
# of an application. Internal code imports the extension directly.
for _name in _compat.REPLACEMENTS:
    globals().pop(_name, None)


def __getattr__(name):
    if name in _compat.REPLACEMENTS:
        # `from quabla import *` resolves every name in `__all__`; it binds the
        # v0.1 names silently so that only explicit access warns (design 5.1)
        # and a star import does not spend the once-per-name warnings.
        if not _compat.is_star_import(_sys._getframe(1)):
            _compat.warn(name)
        return (
            optim.Adam
            if name == "Adam"
            else _quabla
            if name == "quabla"
            else getattr(_quabla, name)
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_compat.REPLACEMENTS))
