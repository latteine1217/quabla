# Public `quabla` package over the compiled extension `quabla._quabla`.
#
# The v0.1 wheel shipped a maturin-generated shim that star-imported the
# native submodule `quabla.quabla`. This package keeps that public surface
# unchanged: every extension name is re-exported, `__all__` and `__doc__`
# mirror the extension as before, and `quabla.quabla` stays available both
# as an attribute and as an importable module path. Later v0.2 layers are
# pure Python on top of these names.

import sys as _sys

from . import _quabla
from ._quabla import *  # noqa: F403

__doc__ = _quabla.__doc__
# A copy, so later additions to this package do not mutate the extension's list.
__all__ = list(_quabla.__all__)

# v0.1 compatibility alias for the native module. Registering it in
# sys.modules keeps `import quabla.quabla` and `from quabla.quabla import X`
# working, not only attribute access.
quabla = _quabla
_sys.modules[__name__ + ".quabla"] = _quabla
