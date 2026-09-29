import importlib
import importlib.machinery
import pathlib
import pickle

import quabla

# The 132 non-underscore names of `dir(quabla)` on the v0.1 build, generated
# before the move to the mixed Rust/Python layout. Every later build must keep
# exporting them (docs/api_v0_2_design.md, slice S0 and Appendix A).
V0_1_NAMES_FILE = pathlib.Path(__file__).with_name("v0_1_public_names.txt")


def v0_1_names():
    names = V0_1_NAMES_FILE.read_text().split()
    assert len(names) == len(set(names)) == 132
    return set(names)


def test_v0_1_names_are_exported():
    missing = v0_1_names() - set(dir(quabla))
    assert not missing, f"v0.1 names missing from quabla: {sorted(missing)}"
    for name in v0_1_names():
        getattr(quabla, name)


def test_all_covers_v0_1_star_import_and_resolves():
    # v0.1 took __all__ from the extension, which does not list the `quabla`
    # alias, so `from quabla import *` never bound it.
    assert v0_1_names() - {"quabla"} <= set(quabla.__all__)
    for name in quabla.__all__:
        getattr(quabla, name)


def test_private_extension_module():
    native = quabla._quabla
    assert native.__name__ == "quabla._quabla"
    assert isinstance(native.__spec__.loader, importlib.machinery.ExtensionFileLoader)
    assert native.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
    # The public package itself is pure Python.
    assert quabla.__file__.endswith("__init__.py")


def test_quabla_alias_resolves():
    assert quabla.quabla is quabla._quabla
    assert importlib.import_module("quabla.quabla") is quabla._quabla

    import quabla.quabla as alias
    from quabla.quabla import Tensor

    assert alias is quabla._quabla
    assert Tensor is quabla.Tensor


def test_v0_1_pickled_function_path_loads():
    # A v0.1 pickle names module-level functions by `quabla.quabla.<name>`.
    assert pickle.loads(b"cquabla.quabla\ngrad\n.") is quabla.grad


def test_dtype_reprs_are_unchanged():
    assert quabla.dtype.__module__ == "quabla"
    assert repr(quabla.float32) == "quabla.float32"
    assert repr(quabla.float64) == "quabla.float64"
    assert repr(quabla.bool_) == "quabla.bool_"
    tensor = quabla.Tensor([2], [1.0, 2.0], dtype=quabla.float32)
    assert repr(tensor) == "Tensor(shape=[2], dtype=quabla.float32)"


if __name__ == "__main__":
    # Run every test_* function in definition order so new tests cannot be left out of a manual list
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
