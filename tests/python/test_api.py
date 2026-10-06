import collections
import ctypes
import dataclasses
import gc
import importlib
import importlib.machinery
import inspect
import math
import pathlib
import pickle
import struct
import subprocess
import sys
import warnings

import quabla
import quabla as qb

# NumPy is optional for quabla, so NumPy-dependent tests skip without it.
try:
    import numpy as np
except ImportError:
    np = None

from _support import assert_close, enabled, require, run, skip
from _support import raises as assert_raises

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


def test_star_import_binds_deprecated_names_without_warning():
    # A fresh interpreter keeps the once-per-name warning state clean. The star
    # import must neither warn (so it survives -W error) nor spend the warning
    # that explicit access still emits.
    script = """
import warnings
from quabla import *
assert Adam is not None and tensor_jit_fn is not None and Matrix is not None
import quabla
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    quabla.Adam
    from quabla import Matrix
assert [str(w.message).split(" is")[0] for w in caught] == ["quabla.Adam", "quabla.Matrix"]
"""
    subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", script], check=True
    )


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
    # A v0.1 pickle names module-level functions by `quabla.quabla.<name>`;
    # `grad` there is the native v0.1 function, which `quabla.grad` wraps (D17).
    assert pickle.loads(b"cquabla.quabla\ngrad\n.") is quabla._quabla.grad
    assert pickle.loads(b"cquabla.quabla\ntensor_jit_fn\n.") is quabla.tensor_jit_fn


def test_dtype_reprs_are_unchanged():
    assert quabla.dtype.__module__ == "quabla"
    assert repr(quabla.float32) == "quabla.float32"
    assert repr(quabla.float64) == "quabla.float64"
    assert repr(quabla.bool_) == "quabla.bool_"
    tensor = quabla.Tensor([2], [1.0, 2.0], dtype=quabla.float32)
    # Tensors show their values since v0.3; the dtype prints by name.
    assert repr(tensor) == "Tensor([1., 2.], dtype=float32)"


def requires_numpy(test):
    def run():
        if np is None:
            skip("numpy is not installed")
        test()

    run.__name__ = test.__name__
    return run


def assert_tensor(tensor, values, dtype):
    assert isinstance(tensor, qb.Tensor), type(tensor)
    assert tensor.dtype == dtype, (tensor.dtype, dtype)
    assert tensor.tolist() == values, (tensor.tolist(), values)


def test_importing_quabla_does_not_import_numpy():
    code = "import sys, quabla; assert 'numpy' not in sys.modules, 'numpy was imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_array_dtype_mapping_from_python_data():
    # (input, expected dtype, expected tolist())
    cases = [
        (1.5, qb.float64, 1.5),
        (2, qb.float64, 2.0),
        (True, qb.bool_, True),
        ([1, 2.5], qb.float64, [1.0, 2.5]),
        ((1, 2), qb.float64, [1.0, 2.0]),
        ([True, False], qb.bool_, [True, False]),
        ([True, 2.0], qb.float64, [1.0, 2.0]),
        ([[1.0, 2.0], (3.0, 4.0)], qb.float64, [[1.0, 2.0], [3.0, 4.0]]),
        (
            [[[1.0], [2.0]], [[3.0], [4.0]]],
            qb.float64,
            [[[1.0], [2.0]], [[3.0], [4.0]]],
        ),
    ]
    for value, dtype, expected in cases:
        assert_tensor(qb.array(value), expected, dtype)
        assert_tensor(qb.asarray(value), expected, dtype)
    assert qb.array([1.0, 2.0]).shape == [2]
    assert qb.array(1.5).shape == []
    # An explicit dtype rounds the values.
    assert_tensor(
        qb.array([0.1, 2.0], dtype=qb.float32), [0.10000000149011612, 2.0], qb.float32
    )
    assert_tensor(qb.array([0.0, 3.0], dtype=qb.bool_), [False, True], qb.bool_)
    assert_tensor(qb.array([True, False], dtype=qb.float64), [1.0, 0.0], qb.float64)


def test_array_rejects_ragged_empty_and_non_numeric_data():
    assert_raises(ValueError, qb.array, [[1.0, 2.0], [3.0]], match="ragged")
    assert_raises(ValueError, qb.array, [[1.0, 2.0], 3.0], match="ragged")
    assert_raises(ValueError, qb.array, [1.0, [2.0]], match="ragged")
    assert_raises(ValueError, qb.array, [], match="empty sequence")
    assert_raises(ValueError, qb.array, [[], []], match="empty sequence")
    assert_raises(TypeError, qb.array, [1.0, "a"], match="type str")
    assert_raises(TypeError, qb.array, "abc", match="cannot convert str")
    assert_raises(TypeError, qb.array, {"a": 1.0}, match="cannot convert dict")
    assert_raises(TypeError, qb.array, [1.0], dtype="float32")


def test_array_and_asarray_on_quabla_arrays():
    tensor = qb.array([1.0, 2.0], dtype=qb.float32)
    assert qb.asarray(tensor) is tensor
    assert qb.asarray(tensor, dtype=qb.float32) is tensor
    copied = qb.array(tensor)
    assert copied is not tensor
    assert_tensor(copied, [1.0, 2.0], qb.float32)
    assert_tensor(qb.asarray(tensor, dtype=qb.float64), [1.0, 2.0], qb.float64)
    view = qb.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]).slice(1, 0, 1)
    assert_tensor(qb.asarray(view), [[1.0], [3.0]], qb.float64)


def test_array_abc_registers_tensor_types():
    tensor = qb.array([1.0, 2.0])
    view = tensor.slice(0, 0, 1)
    assert isinstance(tensor, qb.Array)
    assert isinstance(view, qb.Array)
    assert issubclass(qb.TraceTensor, qb.Array)
    assert not isinstance([1.0], qb.Array)
    seen = []

    def loss(x):
        seen.append(isinstance(x, qb.Array))
        return x.sum()

    qb.tensor_jit_fn(loss, [("x", [2])])
    assert seen == [True]


def test_tolist_item_and_float():
    tensor = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert tensor.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert qb.array(2.5).tolist() == 2.5
    assert qb.array([True]).tolist() == [True]
    assert qb.array(2.5).item() == 2.5
    assert qb.array([[2.5]]).item() == 2.5
    assert qb.array(True).item() is True
    assert float(qb.array([[3.0]])) == 3.0
    assert float(qb.array(0.1, dtype=qb.float32)) == 0.10000000149011612
    assert float(qb.array(True)) == 1.0
    assert_raises(
        ValueError, tensor.item, match="exactly one element, got shape [2, 3]"
    )
    assert_raises(TypeError, float, tensor, match="single-element tensors")


def test_single_element_tensors_stay_tensor_operands():
    # `float(t)` must not let a Tensor slip into scalar-operand parsing,
    # which would turn it into a weak scalar and drop its dtype.
    scalar = qb.array(2.0, dtype=qb.float32)
    # A tensor exponent keeps its float32 dtype, so the strict promotion rejects the float64 base.
    assert_raises(
        ValueError, lambda: qb.array([1.0, 2.0]) ** scalar, match="mismatched dtypes"
    )

    def mixed(x):
        return (x * scalar).sum()

    # Traced, the tensor is a strong float32 constant (slice S3b): a float64
    # tracer rejects it, a float32 tracer keeps float32.
    for function in [
        mixed,
        lambda x: (scalar - x).sum(),
        lambda x: qb.where(x > 0.0, x, scalar).sum(),
    ]:
        assert_raises(
            ValueError,
            qb.tensor_jit_fn,
            function,
            [("x", [2])],
            match="mismatched dtypes",
        )
        compiled = qb.tensor_jit_fn(function, [("x", [2], qb.float32)])
        assert (
            compiled({"x": qb.array([1.0, 2.0], dtype=qb.float32)}).dtype == qb.float32
        )


def test_factories():
    assert_tensor(qb.zeros(3), [0.0, 0.0, 0.0], qb.float64)
    assert_tensor(qb.zeros((2, 1), dtype=qb.float32), [[0.0], [0.0]], qb.float32)
    assert_tensor(qb.ones([2]), [1.0, 1.0], qb.float64)
    assert_tensor(qb.ones((), dtype=qb.bool_), True, qb.bool_)
    assert_tensor(qb.full((2, 2), 2.5), [[2.5, 2.5], [2.5, 2.5]], qb.float64)
    assert_tensor(qb.full(2, True), [True, True], qb.bool_)
    assert_tensor(
        qb.full(2, 0.1, dtype=qb.float32), [0.10000000149011612] * 2, qb.float32
    )
    assert_tensor(qb.arange(3), [0.0, 1.0, 2.0], qb.float64)
    assert_tensor(qb.arange(1, 2, 0.5, dtype=qb.float32), [1.0, 1.5], qb.float32)
    assert_tensor(qb.linspace(0.0, 1.0, 3), [0.0, 0.5, 1.0], qb.float64)
    assert_tensor(qb.eye(2), [[1.0, 0.0], [0.0, 1.0]], qb.float64)
    assert_tensor(qb.eye(1, 2, dtype=qb.bool_), [[True, False]], qb.bool_)
    assert_raises(ValueError, qb.zeros, (2, 0))
    assert_raises(ValueError, qb.arange, 0)


def test_buffer_export_is_read_only_and_keeps_the_tensor_alive():
    tensor = qb.array([[1.0, 2.0], [3.0, 4.0]])
    before = sys.getrefcount(tensor)
    view = memoryview(tensor)
    assert sys.getrefcount(tensor) == before + 1
    assert (view.format, view.shape, view.strides) == ("d", (2, 2), (16, 8))
    assert view.readonly
    assert view.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    view.release()
    assert sys.getrefcount(tensor) == before

    # The view owns a reference to the storage: it outlives the last name.
    view = memoryview(tensor)
    del tensor
    gc.collect()
    assert view.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    view.release()

    for dtype, fmt, itemsize, values in [
        (qb.float32, "f", 4, [0.5, 2.0]),
        (qb.bool_, "?", 1, [True, False]),
    ]:
        view = memoryview(qb.array(values, dtype=dtype))
        assert (view.format, view.itemsize, view.tolist()) == (fmt, itemsize, values)
    assert memoryview(qb.array(1.5)).shape == ()
    assert bytes(qb.array([1.0])) == memoryview(qb.array([1.0])).tobytes()
    # A writable request (ctypes.from_buffer asks for one) is refused.
    assert_raises(
        (BufferError, TypeError), ctypes.c_double.from_buffer, qb.array([1.0])
    )


@requires_numpy
def test_numpy_dtype_mapping():
    # (NumPy dtype, expected quabla dtype); integers become float64.
    cases = [
        (np.float64, qb.float64),
        (np.float32, qb.float32),
        (np.bool_, qb.bool_),
        (np.int8, qb.float64),
        (np.int16, qb.float64),
        (np.int32, qb.float64),
        (np.int64, qb.float64),
        (np.uint8, qb.float64),
        (np.uint16, qb.float64),
        (np.uint32, qb.float64),
        (np.uint64, qb.float64),
    ]
    for np_dtype, dtype in cases:
        source = np.array([0, 1, 1, 0]).astype(np_dtype)
        tensor = qb.array(source)
        assert tensor.dtype == dtype, (np_dtype, tensor.dtype)
        expected = (
            [False, True, True, False] if dtype == qb.bool_ else [0.0, 1.0, 1.0, 0.0]
        )
        assert tensor.tolist() == expected, (np_dtype, tensor.tolist())
        # NumPy scalars keep the same mapping.
        assert qb.array(source[1]).dtype == dtype
        assert qb.array(source[1]).shape == []
    assert qb.array(np.array(2.5)).tolist() == 2.5
    assert (
        qb.array(np.array([1, 2], dtype=np.int64), dtype=qb.float32).dtype == qb.float32
    )
    assert_raises(TypeError, qb.array, np.zeros(2, dtype=np.float16), match="float16")
    assert_raises(TypeError, qb.array, np.zeros(2, dtype=np.complex128))
    assert_raises(TypeError, qb.array, np.array([1, "a"], dtype=object))
    assert_raises(TypeError, qb.array, np.ones(2, dtype=">f8"), match="byte order")
    assert_raises(ValueError, qb.array, np.zeros((2, 0)))


@requires_numpy
def test_numpy_round_trips():
    base = np.arange(24.0).reshape(2, 3, 4) / 7.0
    sources = [
        base.reshape(-1),
        base.reshape(6, 4),
        base,
        base[:, ::2, 1:],
        base.transpose(2, 0, 1),
        base[::-1, :, ::-3],
        np.asfortranarray(base),
    ]
    for source in sources:
        for np_dtype, dtype in [(np.float64, qb.float64), (np.float32, qb.float32)]:
            typed = source.astype(np_dtype)
            tensor = qb.array(typed)
            assert tensor.shape == list(typed.shape)
            assert tensor.dtype == dtype
            result = tensor.numpy()
            assert result.dtype == np_dtype
            assert result.shape == typed.shape
            assert np.array_equal(result, typed)
            assert tensor.tolist() == typed.tolist()
        mask = source > 1.0
        tensor = qb.array(mask)
        assert tensor.dtype == qb.bool_
        assert np.array_equal(tensor.numpy(), mask)
        assert tensor.numpy().dtype == np.bool_


@requires_numpy
def test_typed_buffer_import_owns_values_for_contiguous_and_strided_sources():
    for dtype in [
        np.float32,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]:
        for layout in ["contiguous", "transposed", "reversed"]:
            base = np.arange(24).reshape(2, 3, 4).astype(dtype)
            if dtype == np.float32:
                base.reshape(-1)[:4] = [-0.0, float("nan"), float("inf"), 0.1]
            elif dtype in [np.int64, np.uint64]:
                base.reshape(-1)[0] = 2**53 + 1
            source = (
                base
                if layout == "contiguous"
                else (
                    base.transpose(2, 0, 1)
                    if layout == "transposed"
                    else base[:, :, ::-1]
                )
            )
            expected_dtype = np.float32 if dtype == np.float32 else np.float64
            expected = source.astype(expected_dtype, order="C")
            tensor = qb.array(source)
            assert memoryview(tensor).tobytes() == expected.tobytes(order="C")
            assert tensor.shape == list(source.shape)
            source.flat[0] = 42
            assert memoryview(tensor).tobytes() == expected.tobytes(order="C")


@requires_numpy
def test_numpy_export_is_zero_copy_for_float64_and_read_only():
    tensor = qb.array([1.0, 2.0, 3.0])
    first, second = tensor.numpy(), np.asarray(tensor)
    assert np.shares_memory(first, second)
    assert not first.flags.writeable
    try:
        first[0] = 5.0
    except ValueError:
        pass
    else:
        raise AssertionError("a Tensor export must be read-only")
    writable = np.array(tensor)
    assert writable.flags.writeable and not np.shares_memory(writable, first)
    writable[0] = 5.0
    assert tensor.tolist() == [1.0, 2.0, 3.0]
    # The NumPy array keeps the storage alive after the tensor is gone.
    del tensor, second
    gc.collect()
    assert first.tolist() == [1.0, 2.0, 3.0]


@requires_numpy
def test_numpy_array_protocol_dtype_and_copy():
    tensor = qb.array([[1.0, 2.0], [3.0, 4.0]])
    default = tensor.__array__()
    assert default.dtype == np.float64 and not default.flags.writeable
    copied = tensor.__array__(copy=True)
    assert copied.flags.writeable and not np.shares_memory(copied, default)
    no_copy = tensor.__array__(copy=False)
    assert np.shares_memory(no_copy, default)
    converted = tensor.__array__(np.float32)
    assert converted.dtype == np.float32 and converted.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert_raises(
        ValueError, tensor.__array__, np.float32, copy=False, match="without a copy"
    )
    assert tensor.__array__(np.float32, copy=True).flags.writeable
    assert np.shares_memory(tensor.__array__(np.float64, copy=False), default)
    assert np.asarray(tensor, dtype=np.float32).dtype == np.float32
    assert np.array(qb.array([True, False])).dtype == np.bool_
    assert np.asarray(qb.array(1.5, dtype=qb.float32)).dtype == np.float32
    if int(np.__version__.split(".")[0]) >= 2:
        assert np.shares_memory(np.asarray(tensor, copy=False), default)
        assert np.asarray(tensor, copy=True).flags.writeable


@requires_numpy
def test_numpy_round_trip_of_a_million_values_is_fast():
    # Design target (section 6, S1): under 5 ms per round trip; v0.1 took
    # about 58 ms through to_flat_list(). Best of several runs, with a
    # generous bound so a loaded machine does not flake.
    import time

    source = np.random.default_rng(0).random(1_000_000)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        result = qb.array(source).numpy()
        timings.append(time.perf_counter() - start)
    assert np.array_equal(result, source)
    assert min(timings) < 0.005, timings


def traced(function, *arrays):
    """Evaluates `function` on `arrays` through a CPU trace."""
    specs = [
        (f"a{index}", array.shape, array.dtype) for index, array in enumerate(arrays)
    ]
    compiled = qb.tensor_jit_fn(function, specs)
    return compiled({f"a{index}": array for index, array in enumerate(arrays)})


def test_module_level_unary_and_reduction_ops_match_methods():
    x = qb.array([[0.25, 0.5], [1.5, 2.0]])
    unary = [
        "abs",
        "cos",
        "exp",
        "log",
        "log1p",
        "relu",
        "sigmoid",
        "sin",
        "softplus",
        "sqrt",
        "tanh",
    ]
    for name in unary:
        function = getattr(qb, name)
        assert function.__name__ == name
        expected = getattr(x, name)()
        assert_close(function(x), expected)
        assert_close(traced(function, x), expected)
    for name in ["sum", "mean", "max", "min", "norm"]:
        function = getattr(qb, name)
        assert_close(function(x), getattr(x, name)())
        assert_close(function(x, axis=0), getattr(x, name)(axis=0))
        assert_close(
            function(x, 1, keepdims=True), getattr(x, name)(axis=1, keepdims=True)
        )
        assert function(x, axis=(0, 1), keepdims=True).shape == [1, 1]
        assert_close(
            traced(lambda a, f=function: f(a, axis=1), x), getattr(x, name)(axis=1)
        )
    mask = qb.array([[True, False], [True, True]])
    assert qb.any(mask).item() is True
    assert qb.all(mask).item() is False
    assert qb.all(mask, axis=1).tolist() == [False, True]


def test_norm_scales_to_avoid_overflow_and_underflow():
    def bits(values):
        return [struct.pack("<d", v) for v in values]

    for scale in (1e20, 1e-25, 1.0):
        x = qb.array([scale, -scale, 0.0], dtype=qb.float32)
        expected = math.sqrt(2.0) * scale
        eager = qb.norm(x).item()
        assert math.isclose(eager, expected, rel_tol=1e-6), (scale, eager)
        assert bits([qb.jit(qb.norm)(x).item()]) == bits([eager])
        gradient = qb.grad(qb.norm)(x).tolist()
        assert all(
            math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-30)
            for g, w in zip(gradient, [0.5**0.5, -(0.5**0.5), 0.0])
        ), (scale, gradient)
    rows = qb.array([[3.0, 4.0], [1e20, 0.0]], dtype=qb.float32)
    assert_close(qb.norm(rows, axis=1), [5.0, 1e20], 1e-6 * 1e20)
    assert qb.jit(lambda a: qb.norm(a, axis=0, keepdims=True))(rows).shape == [1, 2]
    # Zero, infinite and NaN scales keep the unscaled results.
    assert qb.norm(qb.zeros([3])).item() == 0.0
    assert qb.grad(qb.norm)(qb.zeros([3])).tolist() == [0.0, 0.0, 0.0]
    assert qb.norm(qb.array([math.inf, 1.0])).item() == math.inf
    assert math.isnan(qb.jit(qb.norm)(qb.array([math.nan, 1.0])).item())


def test_module_level_ops_accept_python_scalars_and_lists():
    assert_tensor(qb.sin(0.0), 0.0, qb.float64)
    assert_close(qb.exp([0.0, 1.0]), [1.0, math.e])
    assert_close(qb.sum([[1.0, 2.0], [3.0, 4.0]], axis=0), [4.0, 6.0])
    assert_close(qb.matmul([[1.0, 2.0]], [[3.0], [4.0]]), [[11.0]])
    assert_close(qb.solve([[2.0, 0.0], [0.0, 4.0]], [[2.0], [8.0]]), [[1.0], [2.0]])
    # A Python number stays weak in maximum/minimum on either side.
    x = qb.array([-1.0, 2.0], dtype=qb.float32)
    for result in [qb.maximum(x, 0.0), qb.maximum(0.0, x)]:
        assert_tensor(result, [0.0, 2.0], qb.float32)
    assert_tensor(qb.minimum(1.0, x), [-1.0, 1.0], qb.float32)
    assert_tensor(
        qb.minimum(x, qb.array([0.0, 3.0], dtype=qb.float32)), [-1.0, 2.0], qb.float32
    )
    assert_tensor(qb.maximum(1.0, 2.0), 2.0, qb.float64)
    assert_raises(
        ValueError, qb.maximum, x, qb.array([0.0, 3.0]), match="mismatched dtypes"
    )


def float_bits(values):
    """IEEE bit patterns of `values` (floats read back from a tensor)."""
    return [struct.pack("<d", value) for value in values]


EXTREME_POINTS = [
    -1000.0,
    -100.0,
    -80.0,
    -20.0,
    -15.0,
    -1.0,
    0.0,
    1.0,
    20.0,
    100.0,
    1000.0,
]


def exact_sigmoid(value):
    """`1 / (1 + e^-x)` in float64 without overflow, as the reference."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    decay = math.exp(value)
    return decay / (1.0 + decay)


def test_log1p_follows_ieee_semantics_eagerly_and_traced():
    points = [-2.0, -1.0, -0.5, -1e-10, 0.0, 1e-10, 3.0]
    for dtype in [qb.float32, qb.float64]:
        x = qb.array(points, dtype=dtype)
        for result in [qb.log1p(x), qb.jit(qb.log1p)(x)]:
            values = result.tolist()
            assert result.dtype == dtype
            assert math.isnan(values[0]) and values[1] == -math.inf, values
            rounded = qb.array([math.log1p(p) for p in points[2:]], dtype=dtype)
            assert values[2:] == rounded.tolist(), (values, rounded.tolist())
    # log1p keeps the small-argument precision that log(1 + x) loses.
    tiny = qb.array([1e-10], dtype=qb.float32)
    assert qb.jit(qb.log1p)(tiny).item() == qb.array(1e-10, dtype=qb.float32).item()
    gradient = qb.grad(lambda t: qb.sum(qb.log1p(t)))(qb.array([-0.5, 0.0, 3.0]))
    assert_close(gradient, [2.0, 1.0, 0.25], 1e-15)


def test_sigmoid_and_softplus_are_stable_and_eager_matches_jit_bitwise():
    # Relative tolerances: about one ulp for values, a few ulp for gradients,
    # which compose several rounded ops.
    tolerances = {qb.float32: (1.2e-7, 6e-7), qb.float64: (3e-16, 1e-15)}
    for dtype, (value_tolerance, gradient_tolerance) in tolerances.items():
        x = qb.array(EXTREME_POINTS, dtype=dtype)
        for name in ["sigmoid", "softplus", "log1p"]:
            function = getattr(qb, name)
            argument = (
                x if name != "log1p" else qb.array([-0.5, 1e-10, 3.0], dtype=dtype)
            )
            eager, jitted = function(argument), qb.jit(function)(argument)
            assert eager.dtype == jitted.dtype == dtype
            assert float_bits(eager.tolist()) == float_bits(jitted.tolist()), (
                name,
                eager.tolist(),
                jitted.tolist(),
            )
        sigmoid = qb.jit(qb.sigmoid)(x).tolist()
        softplus = qb.jit(qb.softplus)(x).tolist()
        gradient = qb.jit(qb.grad(lambda t: qb.sum(qb.sigmoid(t))))(x).tolist()
        softplus_gradient = qb.jit(qb.grad(lambda t: qb.sum(qb.softplus(t))))(
            x
        ).tolist()
        for index, point in enumerate(EXTREME_POINTS):
            expected = exact_sigmoid(point)
            # s(1 - s) as s(x) s(-x), without the cancellation in 1 - s.
            slope = expected * exact_sigmoid(-point)
            log_term = max(point, 0.0) + math.log1p(math.exp(-abs(point)))
            for actual, reference, tolerance in [
                (sigmoid[index], expected, value_tolerance),
                (softplus[index], log_term, value_tolerance),
                (softplus_gradient[index], expected, gradient_tolerance),
                (gradient[index], slope, gradient_tolerance),
            ]:
                assert math.isfinite(actual), (point, actual)
                # float32 subnormals (|x| = 100) keep only a few significant bits.
                if dtype == qb.float32 and 0.0 < reference < 1.2e-38:
                    assert abs(actual - reference) <= 2.0**-149, (
                        point,
                        actual,
                        reference,
                    )
                else:
                    assert abs(actual - reference) <= tolerance * reference, (
                        dtype,
                        point,
                        actual,
                        reference,
                    )
    # The textbook 1 / (1 + exp(-x)) underflowed or formed 0 * inf here.
    x32 = qb.array([-100.0, -80.0, -20.0], dtype=qb.float32)
    gradient = qb.grad(lambda t: qb.sum(qb.sigmoid(t)))(x32).tolist()
    assert all(math.isfinite(value) and value > 0.0 for value in gradient), gradient
    assert qb.sigmoid(qb.array(-80.0, dtype=qb.float32)).item() > 0.0
    softplus = qb.jit(qb.softplus)(qb.array(-20.0, dtype=qb.float32)).item()
    assert abs(softplus - 2.0611536e-9) <= 1e-15, softplus


# Signed zeros, subnormals, infinities, NaNs of both signs and a payload,
# rounding and domain edges.
EAGER_JIT_POINTS = [
    0.0,
    -0.0,
    1e-310,
    -1e-30,
    0.5,
    -1.5,
    2.5,
    0.1,
    1.0,
    -1.0,
    20.0,
    -88.5,
    1e30,
    -1e300,
    math.inf,
    -math.inf,
    math.nan,
    struct.unpack("<d", struct.pack("<Q", 0xFFF8000000001234))[0],
]


# A well-conditioned matrix; its Gram matrix is symmetric positive definite.
SPD_ROOT = [[2.0, 0.5, 0.1], [0.3, 1.5, 0.2], [0.1, 0.4, 1.2]]


def eager_jit_summary(result):
    """dtype, shape, and element bit patterns of a result (or a raised type)."""
    if isinstance(result, BaseException):
        return type(result).__name__
    if result.dtype == qb.bool_:
        bits = [bool(value) for value in flat_values(result.tolist())]
    else:
        bits = float_bits(flat_values(result.tolist()))
    return str(result.dtype), list(result.shape), bits


def flat_values(values):
    if not isinstance(values, list):
        return [values]
    return [item for value in values for item in flat_values(value)]


def eager_and_jit(function, *operands):
    results = []
    for evaluate in (function, qb.jit(function)):
        try:
            results.append(eager_jit_summary(evaluate(*operands)))
        except (ValueError, TypeError) as error:
            results.append(eager_jit_summary(error))
    return results


def test_eager_ops_match_jit_bitwise():
    # An eager op evaluates the graph its trace records, so its value,
    # dtype, and errors equal those of the CPU jit, signed zeros and NaN
    # signs included. Eager abs(+0.0) used to return -0.0 and
    # sqrt(-0.0) returned -0.0, among others.
    unary = ["abs", "relu", "sigmoid", "softplus", "sqrt", "exp", "log", "log1p",
             "expm1", "erf", "erfc", "tanh", "sin", "cos", "tan", "arcsin", "arctanh",
             "cbrt", "floor", "round", "isnan", "isfinite", "__neg__"]
    binary = [
        lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b,
        lambda a, b: a**b, lambda a, b: a.atan2(b), lambda a, b: a.fmod(b),
        lambda a, b: a.maximum(b), lambda a, b: a.minimum(b), lambda a, b: a.gt(b),
        lambda a, b: a > b, lambda a, b: a.equal(b),
    ]
    count = len(EAGER_JIT_POINTS)
    for dtype in (qb.float32, qb.float64):
        x = qb.array(EAGER_JIT_POINTS, dtype=dtype)
        lhs, rhs = x.reshape(count, 1), x.reshape(1, count)
        pairs = (x.reshape(count, 1) + qb.zeros([1, count]).astype(dtype)).reshape(-1)
        mask = x > 0.5
        cases = [(lambda a, m=m: getattr(a, m)(), [x]) for m in unary]
        cases += [(function, [lhs, rhs]) for function in binary]
        cases += [(function, [pairs, pairs[::-1]]) for function in binary]
        for scalar in (0.0, -0.0, 0.1, math.inf, math.nan):
            cases += [
                (lambda a, s=scalar: a + s, [x]),
                (lambda a, s=scalar: s - a, [x]),
                (lambda a, s=scalar: a * s, [x]),
                (lambda a, s=scalar: s / a, [x]),
                (lambda a, s=scalar: a**s, [x]),
                (lambda a, s=scalar: a.maximum(s), [x]),
                (lambda a, s=scalar: a <= s, [x]),
            ]
        cube = qb.array([EAGER_JIT_POINTS[i % count] for i in range(60)], dtype=dtype)
        cube = cube.reshape(4, 3, 5)
        for method in ("sum", "mean", "max", "min", "prod", "norm"):
            for axis in (None, 1, (0, 2)):
                cases.append((lambda a, m=method, ax=axis: getattr(a, m)(axis=ax), [cube]))
        cases += [
            (lambda a: a.cumsum(axis=1, reverse=True), [cube]),
            (lambda a, u: a.scatter_add([1, 1, 0], u, axis=1), [cube, cube]),
            (lambda a: a.reshape(12, 5) @ a.reshape(5, 12), [cube]),
            (lambda m, a: qb.where(m, a, 2.5), [mask, x]),
            (lambda m, a: (m * 2.0) + a, [mask, x]),
            (lambda m, a: (m * 0.1).sum() + a, [mask, x]),
            (lambda b: (b @ b.T).cholesky(), [qb.array(SPD_ROOT, dtype=dtype)]),
            (lambda a: qb.eye(3).astype(a.dtype).solve(a.reshape(3, 6)), [x]),
        ]
        for function, operands in cases:
            eager, jitted = eager_and_jit(function, *operands)
            assert eager == jitted, (dtype, eager, jitted)
    # Eager Cholesky keeps its documented validation, which the traced op
    # (lower triangle, NaN for an indefinite matrix) omits.
    assert_raises(ValueError, (-qb.eye(3)).cholesky, match="positive-definite")
    asymmetric = qb.array([[2.0, 1.0], [0.0, 2.0]])
    assert_raises(ValueError, asymmetric.cholesky, match="symmetric")
    zeros = qb.array([0.0, -0.0])
    assert float_bits(zeros.abs().tolist()) == float_bits([0.0, 0.0])
    assert float_bits(zeros.sqrt().tolist()) == float_bits([0.0, 0.0])


def test_fused_elementwise_regions_match_their_ops_one_by_one():
    # The CPU runs a jit plan's elementwise subgraphs, and an eager
    # composite, as one fused program over tiles of elements. Its bits equal
    # those of the same ops run one by one as eager single-op calls, NaN
    # payloads, signed zeros, and float32 rounding after every op included.
    def relu(a):
        return qb.where(qb.logical_or(qb.isnan(a), a > 0.0), a, 0.0)

    def sigmoid(a):
        decay = qb.where(a.gt(0.0), a, 0.0 - a) * -1.0
        decay = decay.exp()
        denominator = decay + 1.0
        return qb.where(a > 0.0, 1.0 / denominator, decay / denominator)

    chains = [
        relu,
        sigmoid,
        lambda a: a * a.sin() + a.exp() / (1.0 + a * a),
        lambda a: qb.where(a > 0.5, a.log1p(), a.tanh() * 2.0) - a.sqrt(),
        lambda a: (a.astype(qb.float64) * 3.0).astype(qb.float32).astype(a.dtype) + a,
    ]
    count = len(EAGER_JIT_POINTS)
    for dtype in (qb.float32, qb.float64):
        x = qb.array(EAGER_JIT_POINTS, dtype=dtype)
        pairs = (x.reshape(count, 1) + qb.zeros([1, count]).astype(dtype)).reshape(-1)
        swapped = (x.reshape(1, count) + qb.zeros([count, 1]).astype(dtype)).reshape(-1)
        for chain in chains:
            one_by_one = eager_jit_summary(chain(x))
            assert eager_jit_summary(qb.jit(chain)(x)) == one_by_one, (dtype, chain)
        expected = eager_jit_summary(relu(x))
        assert eager_jit_summary(x.relu()) == expected
        expected = eager_jit_summary(sigmoid(x))
        assert eager_jit_summary(x.sigmoid()) == expected
        # Two NaN operands of a commutative op give the first one's NaN.
        both = [
            lambda a, b: (a * b + b * a) * (a + b) - (b + a),
            lambda a, b: qb.where(a > b, a * b, b + a) * 0.5 + b,
        ]
        for function in both:
            one_by_one = eager_jit_summary(function(pairs, swapped))
            assert eager_jit_summary(qb.jit(function)(pairs, swapped)) == one_by_one
            scalar = qb.array(math.nan, dtype=dtype)
            one_by_one = eager_jit_summary(function(pairs, scalar))
            assert eager_jit_summary(qb.jit(function)(pairs, scalar)) == one_by_one


def test_maximum_minimum_relu_and_extrema_propagate_nan_at_every_position():
    nan = math.nan
    for dtype in [qb.float32, qb.float64]:
        for position in range(3):
            values = [1.0, 2.0, 3.0]
            values[position] = nan
            x = qb.array(values, dtype=dtype)
            other = qb.array([2.0, 2.0, 2.0], dtype=dtype)
            cases = [
                (lambda t: qb.max(t), 0),
                (lambda t: qb.min(t), 0),
                (lambda t: qb.max(qb.reshape(t, (1, 3)), axis=1), 0),
                (lambda t: qb.relu(t), position),
                (lambda t: qb.maximum(t, other), position),
                (lambda t: qb.maximum(other, t), position),
                (lambda t: qb.minimum(t, other), position),
                (lambda t: qb.minimum(other, t), position),
                (lambda t: qb.maximum(t, 0.5), position),
                (lambda t: qb.minimum(0.5, t), position),
            ]
            for function, nan_index in cases:
                eager = function(x).to_flat_list()
                jitted = qb.jit(function)(x).to_flat_list()
                for result in [eager, jitted]:
                    assert math.isnan(result[nan_index]), (values, result)
                    others = [v for i, v in enumerate(result) if i != nan_index]
                    assert not any(math.isnan(v) for v in others), (values, result)
                assert [math.isnan(v) or v for v in eager] == [
                    math.isnan(v) or v for v in jitted
                ], (values, eager, jitted)


def test_maximum_minimum_keep_nan_free_values_and_tie_gradients():
    x = qb.array([-1.0, 0.0, 2.0, -0.0, 3.0])
    y = qb.array([0.0, 0.0, 1.0, 0.0, 3.0])
    assert float_bits(qb.maximum(x, y).tolist()) == float_bits(
        [0.0, 0.0, 2.0, 0.0, 3.0]
    )
    assert float_bits(qb.jit(qb.maximum)(x, y).tolist()) == float_bits(
        [0.0, 0.0, 2.0, 0.0, 3.0]
    )
    assert float_bits(qb.minimum(x, y).tolist()) == float_bits(
        [-1.0, 0.0, 1.0, 0.0, 3.0]
    )
    signed_zeros = qb.array([-0.0, 0.0])
    assert float_bits([qb.max(signed_zeros).item()]) == float_bits([0.0])
    assert float_bits([qb.jit(qb.max)(signed_zeros).item()]) == float_bits([0.0])
    # Ties route the whole subgradient to the right operand, as before.
    for function in [qb.maximum, qb.minimum]:
        dx, dy = qb.grad(lambda a, b: qb.sum(function(a, b)), argnums=(0, 1))(x, y)
        mask = (
            [0.0, 0.0, 1.0, 0.0, 0.0]
            if function is qb.maximum
            else [1.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert dx.tolist() == mask, (function, dx.tolist())
        assert dy.tolist() == [1.0 - m for m in mask], (function, dy.tolist())
    relu_gradient = qb.grad(lambda t: qb.sum(qb.relu(t)))(qb.array([-1.0, 0.0, 2.0]))
    assert relu_gradient.tolist() == [0.0, 0.0, 1.0]
    # A max reduction splits a tie equally (JAX's chooser rule).
    gradient = qb.grad(lambda t: qb.max(t))(qb.array([3.0, 1.0, 3.0]))
    assert gradient.tolist() == [0.5, 0.0, 0.5]



def extremum_cases():
    """A [4, 3, 4] array for max/min checks: NaN, infinities, signed zeros, ties."""
    nan, inf = math.nan, math.inf
    return [
        [[1.0, 3.0, 2.0, 3.0], [-1.0, -1.0, -4.0, 0.5], [0.25, 8.0, 8.0, -8.0]],
        [[nan, 1.0, 2.0, 0.5], [1.0, 2.0, 3.0, nan], [-inf, -inf, -inf, -inf]],
        [[inf, -inf, 1.0, -2.0], [-0.0, 0.0, -0.0, -1.0], [0.0, -0.0, 1e-30, -1e-30]],
        [[-0.0, -0.0, -0.0, -0.0], [0.0, 0.0, 0.0, 0.0], [1e30, -1e30, 5.0, -5.0]],
    ]


def numpy_extremum(data, name, axis, keepdims):
    """NumPy's max or min with the sign of a zero result set by -0 < +0
    (IEEE 754-2019 maximum/minimum, NumPy's own result on arm64; its x86
    builds keep whichever zero comes last)."""
    result = getattr(data, name)(axis=axis, keepdims=keepdims)
    zeros = data == 0
    preferred = zeros & (np.signbit(data) == (name == "min"))
    found = preferred.any(axis=axis, keepdims=keepdims)
    zero = np.where(found == (name == "max"), 0.0, -0.0).astype(data.dtype)
    return np.where(result == 0, zero, result)


def test_traced_max_min_are_one_node_and_match_numpy():
    # One ExtremumAxis node replaces a slice and a maximum per element, so a
    # staged max, its gradient and the scaled norm keep their size for any
    # input size.
    def sized(count):
        x = qb.arange(float(count))
        return (
            lowered_node_count(lambda v: v.max(), x),
            lowered_node_count(qb.grad(lambda v: v.min()), x),
            lowered_node_count(lambda v: qb.linalg.norm(v), x),
        )

    assert sized(6) == sized(600), (sized(6), sized(600))
    assert sized(600)[0] <= 2, sized(600)

    for function in [
        lambda t: t.max(axis=(0, 0)),
        lambda t: t.max(axis=2),
        lambda t: (t > 0).max(),
    ]:
        assert_raises(ValueError, function, qb.ones([2, 2]))
        assert_raises(ValueError, qb.jit(function), qb.ones([2, 2]))

    if np is None:
        print("SKIP traced max/min vs NumPy: numpy missing")
        return
    axes = [None, 0, 1, -1, 2, (0, 2), (1, 2), (0, 1, 2), (-1, 0), ()]
    for dtype, np_dtype in [(qb.float64, np.float64), (qb.float32, np.float32)]:
        data = np.array(extremum_cases(), dtype=np_dtype)
        x = qb.array(data)
        for name in ["max", "min"]:
            for axis in axes:
                for keepdims in [False, True]:
                    expected = numpy_extremum(data, name, axis, keepdims)

                    def reduce(t, name=name, axis=axis, keepdims=keepdims):
                        return getattr(t, name)(axis=axis, keepdims=keepdims)

                    for result in [reduce(x), qb.jit(reduce)(x)]:
                        actual = np.asarray(result)
                        assert result.dtype == dtype
                        assert actual.shape == expected.shape, (name, axis, keepdims)
                        # Equal values and NaNs, and the NumPy sign of each zero.
                        assert np.array_equal(actual, expected, equal_nan=True), (name, axis)
                        assert np.array_equal(np.signbit(actual), np.signbit(expected)), (
                            name,
                            axis,
                        )
        # vmap reduces each example's own axes.
        batched = qb.jit(qb.vmap(lambda t: t.max(axis=(0, -1))))(x)
        assert np.array_equal(np.asarray(batched), data.max(axis=(1, 2)), equal_nan=True)
        batched = qb.vmap(lambda t: t.min(keepdims=True))(x)
        expected = data.min(axis=(1, 2), keepdims=True)
        assert np.array_equal(np.asarray(batched), expected, equal_nan=True)


def f32_rows(rows):
    """`rows` rounded to float32, without NumPy."""
    return [[struct.unpack("<f", struct.pack("<f", value))[0] for value in row] for row in rows]


def test_traced_max_min_gradients_split_ties_and_compose():
    for dtype in [qb.float64, qb.float32]:
        # A tie across both reduced axes shares the gradient three ways, as
        # in JAX; reducing one axis after the other would split it unevenly.
        x = qb.array([[1.0, 1.0], [1.0, 0.0]], dtype=dtype)
        expected = [[1 / 3, 1 / 3], [1 / 3, 0.0]]
        if dtype == qb.float32:
            expected = f32_rows(expected)
        for gradient in [qb.grad(lambda t: t.max())(x), qb.jit(qb.grad(lambda t: t.max()))(x)]:
            assert gradient.dtype == dtype and gradient.tolist() == expected, gradient.tolist()
        direction = qb.array([[3.0, 6.0], [9.0, 1.0]], dtype=dtype)
        assert qb.jvp(lambda t: t.max(), (x,), (direction,))[1].item() == 6.0
        # Per-axis reductions split within each line.
        weights = qb.array([2.0, -4.0], dtype=dtype)
        gradient = qb.grad(lambda t: (t.min(axis=0) * weights).sum())(x)
        assert gradient.tolist() == [[1.0, 0.0], [1.0, -4.0]], gradient.tolist()

        # max(x)^2 with a two-way tie: gradient 2 m e and Hessian 2 e e^T,
        # with e = 1/2 on the tied entries, in every differentiation order.
        v = qb.array([2.0, -1.0, 2.0], dtype=dtype)

        def squared(t):
            return t.max() ** 2

        assert qb.grad(squared)(v).tolist() == [2.0, 0.0, 2.0]
        hessian = [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
        assert qb.hessian(squared)(v).tolist() == hessian
        assert qb.jit(qb.hessian(squared))(v).tolist() == hessian
        assert qb.jacobian(qb.grad(squared))(v).tolist() == hessian

        # A NaN extremum gives NaN derivatives instead of an error.
        nan = qb.array([1.0, math.nan, -2.0], dtype=dtype)
        for function in [lambda t: t.max(), lambda t: t.min()]:
            for result in [qb.grad(function)(nan), qb.jit(qb.grad(function))(nan)]:
                assert all(math.isnan(g) for g in result.tolist()), result.tolist()
            ones = qb.ones([3], dtype=dtype)
            assert math.isnan(qb.jvp(function, (nan,), (ones,))[1].item())

    # Gradient and Hessian against central differences away from ties.
    def loss(t):
        return (qb.sin(t).max(axis=1) ** 3).sum() + (t * t).min(axis=0).sum()

    x0 = qb.array([[0.3, -1.2, 0.8], [-0.4, 0.9, 0.1]])
    gradient = qb.jit(qb.grad(loss))(x0).tolist()
    hessian = qb.jit(qb.hessian(loss))(x0).tolist()
    step = 1e-6
    for row in range(2):
        for column in range(3):
            delta = qb.zeros([2, 3]).tolist()
            delta[row][column] = step
            delta = qb.array(delta)
            estimate = (loss(x0 + delta).item() - loss(x0 - delta).item()) / (2 * step)
            assert abs(gradient[row][column] - estimate) <= 1e-8, (row, column, estimate)
            up = qb.grad(loss)(x0 + delta).tolist()
            down = qb.grad(loss)(x0 - delta).tolist()
            for r in range(2):
                for c in range(3):
                    estimate = (up[r][c] - down[r][c]) / (2 * step)
                    assert abs(hessian[r][c][row][column] - estimate) <= 1e-6, (
                        (r, c, row, column),
                        hessian[r][c][row][column],
                        estimate,
                    )


def test_module_level_shape_and_linear_algebra_ops():
    x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_tensor(
        qb.reshape(x, (3, 2)), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], qb.float64
    )
    assert qb.reshape(x, 6).shape == [6]
    assert qb.transpose(x).shape == [3, 2]
    assert qb.transpose(x, (1, 0)).tolist() == x.transpose([1, 0]).tolist()
    assert (
        qb.broadcast_to(qb.array([1.0, 2.0, 3.0]), (2, 3)).tolist()
        == [[1.0, 2.0, 3.0]] * 2
    )
    assert qb.astype(x, qb.float32).dtype == qb.float32
    square = qb.array([[4.0, 2.0], [2.0, 3.0]])
    assert_close(qb.cholesky(square), square.cholesky())
    assert qb.tril(square).tolist() == [[4.0, 0.0], [2.0, 3.0]]
    assert qb.triu(square).tolist() == [[4.0, 2.0], [0.0, 3.0]]
    assert_close(qb.matmul(square, x[:, 0:2]), square @ x[:, 0:2])
    assert_close(traced(lambda a, b: qb.matmul(a, b), square, x), square @ x)
    assert_close(
        traced(lambda a: qb.reshape(qb.transpose(a), (6,)), x),
        x.transpose().reshape([6]),
    )
    assert_close(traced(lambda a: qb.tril(qb.cholesky(a)), square), square.cholesky())


@requires_numpy
def test_module_level_ops_accept_numpy_arrays():
    source = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    result = qb.sin(source)
    assert result.dtype == qb.float32
    assert np.allclose(result.numpy(), np.sin(source))
    assert_close(qb.sum(source.astype(np.float64), axis=1), [1.5, 3.5])


def test_traced_negation_and_integer_powers_with_gradients():
    x = qb.array([0.5, -1.25, 2.0])
    values = x.tolist()
    assert_close(traced(lambda a: -a, x), [-value for value in values])
    for exponent in [0, 1, 2, 3, np.int64(2) if np is not None else 2]:
        assert_close(traced(lambda a, n=exponent: a**n, x), x ** int(exponent))
        value, gradients = qb.tensor_value_and_grad_fn(
            lambda a, n=exponent: (a**n).sum(), [("x", [3])]
        )({"x": x})
        n = int(exponent)
        assert_close(value, sum(v**n for v in values))
        assert_close(gradients["x"], [n * v ** (n - 1) if n else 0.0 for v in values])
    value, gradients = qb.tensor_value_and_grad_fn(
        lambda a: qb.sum(-(a**2) + qb.sin(a)), [("x", [3])]
    )({"x": x})
    assert_close(value, sum(-(v**2) + math.sin(v) for v in values))
    assert_close(gradients["x"], [-2.0 * v + math.cos(v) for v in values])
    # -0.0 keeps its sign, as for eager tensors.
    assert math.copysign(1.0, traced(lambda a: -a, qb.array([0.0])).item()) == -1.0
    # float32 stays float32.
    x32 = qb.array([1.5, -2.0], dtype=qb.float32)
    assert_tensor(traced(lambda a: -(a**2), x32), [-2.25, -4.0], qb.float32)


def test_power_values_follow_powf_eagerly_and_traced():
    base = qb.array([0.0, 0.0, -2.0, -2.0, 4.0, 2.0, math.inf])
    exponent = qb.array([0.0, -1.0, 3.0, 0.5, 0.5, -1.0, -0.5])
    expected = [1.0, math.inf, -8.0, math.nan, 2.0, 0.5, 0.0]

    def same(actual, expected):
        assert len(actual) == len(expected)
        for lhs, rhs in zip(actual, expected):
            assert (math.isnan(lhs) and math.isnan(rhs)) or lhs == rhs, (
                actual,
                expected,
            )

    for result in [
        base**exponent,
        qb.power(base, exponent),
        traced(lambda a, b: a**b, base, exponent),
        traced(lambda a, b: qb.power(a, b), base, exponent),
    ]:
        same(result.tolist(), expected)
    x = qb.array([0.25, 1.0, 2.25])
    values = x.tolist()
    for function, reference in [
        (lambda a: a**0.5, [v**0.5 for v in values]),
        (lambda a: a**-1, [1.0 / v for v in values]),
        (lambda a: a**2.0, [v * v for v in values]),
        (lambda a: 2.0**a, [2.0**v for v in values]),
        (lambda a: qb.power(3, a), [3.0**v for v in values]),
        (lambda a: qb.power(a, 1.5), [v**1.5 for v in values]),
    ]:
        assert_close(function(x), reference)
        assert_close(traced(function, x), reference)
    # A single-element tensor exponent is an elementwise operand that broadcasts, never a scalar.
    assert_close(x ** qb.array(2.0), [v * v for v in values])
    assert_close(x ** qb.array([2.0]), [v * v for v in values])
    assert_close(qb.power([1.0, 2.0], [[2.0], [3.0]]), [[1.0, 4.0], [1.0, 8.0]])
    assert_close(qb.power(2.0, 3.0), 8.0)
    # Python numbers are weak scalars: float32 stays float32 on both sides.
    x32 = qb.array([0.25, 4.0], dtype=qb.float32)
    for function in [lambda a: a**0.5, lambda a: 2.0**a, lambda a: qb.power(a, -1)]:
        assert function(x32).dtype == qb.float32
        assert traced(function, x32).dtype == qb.float32
    assert_tensor(traced(lambda a: a**0.5, x32), [0.5, 2.0], qb.float32)
    # Strong dtypes stay strict, and bool operands are rejected, eagerly and traced.
    assert_raises(ValueError, lambda: x ** qb.array([2.0, 2.0, 2.0], dtype=qb.float32))
    mask = qb.array([True, False])
    for function in [lambda m: m**0.5, lambda m: 2.0**m, lambda m: qb.power(m, 1.5)]:
        assert_raises(ValueError, function, mask, match="pow is not defined for bool")
        assert_raises(
            ValueError,
            qb.tensor_jit_fn,
            function,
            [("m", [2], qb.bool_)],
            match="pow is not defined for bool",
        )
    assert_raises(
        TypeError,
        qb.tensor_jit_fn,
        lambda a: pow(a, 2, 3),
        [("x", [2])],
        match="modulo",
    )
    assert_raises(TypeError, lambda: pow(x, 2, 3), match="modulo")
    assert_raises(
        ValueError,
        qb.tensor_jit_fn,
        lambda a: -a,
        [("x", [2], qb.bool_)],
        match="negative is not defined for bool",
    )


def test_traced_power_gradients_in_both_operands():
    x = qb.array([0.5, 1.5, 2.25])
    y = qb.array([0.5, -1.0, 2.5])
    xs, ys = x.tolist(), y.tolist()
    for function, base_gradient in [
        (lambda a: (a**0.5).sum(), [0.5 * v**-0.5 for v in xs]),
        (lambda a: (a**-1).sum(), [-(v**-2) for v in xs]),
        (lambda a: (2.0**a).sum(), [2.0**v * math.log(2.0) for v in xs]),
    ]:
        value, gradients = qb.tensor_value_and_grad_fn(function, [("x", [3])])({"x": x})
        assert_close(value, function(x))
        assert_close(gradients["x"], base_gradient)
    value, gradients = qb.tensor_value_and_grad_fn(
        lambda a, b: qb.power(a, b).sum(), [("x", [3]), ("y", [3])]
    )({"x": x, "y": y})
    assert_close(value, sum(a**b for a, b in zip(xs, ys)))
    assert_close(gradients["x"], [b * a ** (b - 1.0) for a, b in zip(xs, ys)])
    assert_close(gradients["y"], [a**b * math.log(a) for a, b in zip(xs, ys)])
    # Conventions at x <= 0: d/dx is the zero subgradient where x == 0 and y < 1 (like sqrt at
    # the origin) and the finite limit for y >= 1; d/dy is 0 for every x <= 0.
    base = qb.array([0.0, 0.0, 0.0, -2.0])
    exponent = qb.array([0.5, 1.0, 2.0, 3.0])
    _, gradients = qb.tensor_value_and_grad_fn(
        lambda a, b: (a**b).sum(), [("x", [4]), ("y", [4])]
    )({"x": base, "y": exponent})
    assert gradients["x"].tolist() == [0.0, 1.0, 0.0, 12.0]
    assert gradients["y"].tolist() == [0.0, 0.0, 0.0, 0.0]
    # Second derivatives compose through the traced graph: d^2/dx^2 x^2.0 is 2, also at 0.
    inputs = {"x": qb.array([0.0, 1.5])}
    traced_power = qb.trace_tensor(lambda a: (a**2.0).sum(), [("x", [2])])
    hessian = traced_power.graph.hessian_scalar(
        traced_power.output.node_id, "x", inputs
    )
    assert hessian == [[2.0, 0.0], [0.0, 2.0]]
    hvp = qb.tensor_hvp_scalar_fn(lambda a: (a**2.0).sum(), [("x", [2])], "x")
    assert hvp(inputs, qb.array([1.0, -1.0])).tolist() == [2.0, -2.0]


def assert_power_device_parity(value_and_grad_fn):
    x = qb.array([0.0, 0.5, 1.5, 2.25, -2.0])
    y = qb.array([0.5, 0.5, -1.0, 2.5, 3.0])
    specs = [("x", [5]), ("y", [5])]

    def loss(a, b):
        return (a**b + 2.0**b * a**0.5 + a**2).sum()

    cpu_value, cpu_gradients = qb.tensor_value_and_grad_fn(loss, specs)(
        {"x": x, "y": y}
    )
    value, gradients = value_and_grad_fn(loss, specs, ["x", "y"])({"x": x, "y": y})
    assert math.isnan(cpu_value.item()) and math.isnan(value.item())
    for name in ["x", "y"]:
        for actual, expected in zip(
            gradients[name].tolist(), cpu_gradients[name].tolist()
        ):
            if math.isnan(expected):
                assert math.isnan(actual), (name, actual)
            else:
                assert abs(actual - expected) <= 1e-5 * max(1.0, abs(expected)), (
                    name,
                    actual,
                    expected,
                )


def test_mlx_power_matches_cpu():
    require("mlx")
    assert_power_device_parity(qb.tensor_value_and_grad_mlx_fn)


def test_cuda_power_matches_cpu():
    require("cuda")
    assert_power_device_parity(
        lambda loss, specs, names: qb.tensor_value_and_grad_cuda_fn(
            loss, specs, names, 0
        )
    )


# -- S2: pytrees and errors ---------------------------------------


def test_tree_flatten_unflatten_and_map():
    tree = {"b": [1.0, (2.0, None)], "a": qb.array(3.0), "c": None}
    leaves, treedef = qb.tree.flatten(tree)
    # dict keys are traversed in sorted order; None is an empty container
    assert leaves[1:] == [1.0, 2.0] and isinstance(leaves[0], qb.Tensor)
    assert treedef.num_leaves == 3
    assert repr(treedef) == "TreeDef({'a': *, 'b': [*, (*, None)], 'c': None})"
    rebuilt = qb.tree.unflatten(treedef, leaves)
    assert rebuilt["b"] == [1.0, (2.0, None)] and rebuilt["c"] is None
    assert qb.tree.flatten({"c": None, "b": [0, (0, None)], "a": 0})[1] == treedef
    assert hash(qb.tree.flatten((1.0,))[1]) == hash(qb.tree.flatten((2.0,))[1])
    assert qb.tree.flatten((1.0,))[1] != qb.tree.flatten([1.0])[1]
    doubled = qb.tree.map(
        lambda x, y: x + y, {"a": 1.0, "b": [2.0]}, {"a": 10.0, "b": [20.0]}
    )
    assert doubled == {"a": 11.0, "b": [22.0]}
    assert_raises(
        ValueError,
        qb.tree.map,
        lambda x, y: x,
        [1.0],
        (1.0,),
        match="structures differ",
    )
    assert_raises(
        ValueError, qb.tree.unflatten, treedef, [1.0], match="has 3 leaves, got 1"
    )
    # Keys of any mutually sortable type are traversed in sorted order.
    leaves, int_keys = qb.tree.flatten({10: "b", 2: "a"})
    assert leaves == ["a", "b"] and repr(int_keys) == "TreeDef({2: *, 10: *})"
    assert qb.tree.unflatten(int_keys, [1, 2]) == {2: 1, 10: 2}
    assert int_keys != qb.tree.structure({"2": 0, "10": 0})
    assert_raises(TypeError, qb.tree.flatten, {1: 2.0, "a": 3.0}, match="sortable")
    # Other container subclasses stay leaves of the tree utilities.
    ordered = collections.OrderedDict(a=1.0)
    assert qb.tree.leaves([ordered]) == [ordered]


@qb.tree.register_dataclass
@dataclasses.dataclass(frozen=True)
class FrozenLayer:
    w: object
    b: object


@dataclasses.dataclass
class ScaledLayer:
    w: object
    activation: str = "tanh"
    scale: float = 1.0


qb.tree.register_dataclass(ScaledLayer, meta_fields=("activation",))

Point = collections.namedtuple("Point", "x y")


class Interval:
    """A node registered with explicit flatten functions; `closed` is aux data."""

    def __init__(self, low, high, closed=True):
        self.low, self.high, self.closed = low, high, closed


qb.tree.register(
    Interval,
    lambda value: ((value.low, value.high), value.closed),
    lambda closed, children: Interval(*children, closed),
)


def test_tree_namedtuple_dataclass_and_registered_nodes():
    tree = {"p": Point(1.0, (2.0, None)), "layer": FrozenLayer(3.0, [4.0])}
    leaves, treedef = qb.tree.flatten(tree)
    assert leaves == [3.0, 4.0, 1.0, 2.0]
    assert repr(treedef) == (
        "TreeDef({'layer': FrozenLayer(w=*, b=[*]), 'p': Point(x=*, y=(*, None))})"
    )
    rebuilt = qb.tree.unflatten(treedef, [v * 10 for v in leaves])
    assert type(rebuilt["p"]) is Point and rebuilt["p"] == Point(10.0, (20.0, None))
    assert rebuilt["layer"] == FrozenLayer(30.0, [40.0])
    assert qb.tree.leaves(tree) == leaves and qb.tree.structure(tree) == treedef
    assert treedef.unflatten(leaves) == tree
    paths, path_def = qb.tree.flatten_with_path(tree)
    assert path_def == treedef
    assert [path for path, _ in paths] == [
        ("layer", "w"),
        ("layer", "b", 0),
        ("p", "x"),
        ("p", "y", 0),
    ]
    # Class and aux data are part of the structure: a NamedTuple differs from
    # a tuple and from another NamedTuple class with the same fields.
    assert qb.tree.structure(Point(1, 2)) != qb.tree.structure((1, 2))
    other_point = collections.namedtuple("Point", "x y")
    assert qb.tree.structure(Point(1, 2)) != qb.tree.structure(other_point(1, 2))
    tanh = ScaledLayer(1.0, "tanh", 2.0)
    assert qb.tree.leaves(tanh) == [1.0, 2.0]
    assert qb.tree.structure(tanh) != qb.tree.structure(ScaledLayer(1.0, "relu", 2.0))
    assert repr(qb.tree.structure(tanh)) == "TreeDef(ScaledLayer(w=*, scale=*, aux=('tanh',)))"
    assert qb.tree.map(lambda a, b: a + b, tanh, tanh) == ScaledLayer(2.0, "tanh", 4.0)
    interval = qb.tree.map(lambda v: v * 2, Interval(1.0, 2.0, closed=False))
    assert (interval.low, interval.high, interval.closed) == (2.0, 4.0, False)
    assert qb.tree.flatten_with_path(Interval(1.0, 2.0))[0] == [((0,), 1.0), ((1,), 2.0)]
    assert_raises(
        ValueError,
        qb.tree.map,
        lambda a, b: a,
        Point(1.0, 2.0),
        (1.0, 2.0),
        match="structures differ",
    )
    # Registration errors.
    assert_raises(ValueError, qb.tree.register, dict, len, len, match="built-in")
    assert_raises(ValueError, qb.tree.register_dataclass, FrozenLayer, match="already")
    assert_raises(TypeError, qb.tree.register_dataclass, Interval, match="dataclass type")

    @dataclasses.dataclass
    class Partial:
        a: object
        b: object

    assert_raises(
        ValueError,
        qb.tree.register_dataclass,
        Partial,
        data_fields=("a",),
        meta_fields=(),
        match="exactly once",
    )

    class Unhashable:
        def __init__(self, value):
            self.value = value

    qb.tree.register(Unhashable, lambda v: ((v.value,), [1]), lambda aux, c: Unhashable(*c))
    assert_raises(TypeError, qb.tree.flatten, Unhashable(1.0), match="must be hashable")


def test_transforms_accept_namedtuple_dataclass_and_registered_nodes():
    x = qb.array([1.0, 2.0])

    def loss(params, x):
        return qb.sum((params.w * x + params.b) ** 2)

    params = FrozenLayer(qb.array([0.5, -1.0]), qb.array(0.25))
    grads = qb.grad(loss)(params, x)
    assert type(grads) is FrozenLayer
    residual = [0.5 * 1.0 + 0.25, -1.0 * 2.0 + 0.25]
    assert_close(grads.w, [2 * residual[0] * 1.0, 2 * residual[1] * 2.0])
    assert_close(grads.b, 2 * sum(residual))
    value, point_grads = qb.value_and_grad(lambda p: qb.sum(p.x * p.y))(Point(x, x * 3.0))
    assert type(point_grads) is Point
    assert_close(point_grads.x, [3.0, 6.0])
    assert_close(point_grads.y, [1.0, 2.0])
    assert_close(value, 15.0)
    # A meta field is static: each value gets its own trace; leaves of the
    # same class, values, and shapes reuse one.
    traces = []

    def apply(layer, x):
        traces.append(layer.activation)
        out = layer.w * x * layer.scale
        return qb.tanh(out) if layer.activation == "tanh" else qb.relu(out)

    jitted = qb.jit(apply)
    jitted(ScaledLayer(x, "tanh", 2.0), x)
    jitted(ScaledLayer(x * 2.0, "tanh", 2.0), x)
    assert_close(jitted(ScaledLayer(x, "relu", 2.0), x), [2.0, 8.0])
    assert traces == ["tanh", "relu"]
    # Two classes with the same fields never share a program.
    other_point = collections.namedtuple("Point", "x y")
    swap = qb.jit(lambda p: p)
    assert type(swap(Point(x, x))) is Point
    assert type(swap(other_point(x, x))) is other_point
    # Registered nodes with aux data round-trip through jit and grad.
    width = qb.jit(lambda i: i.high - i.low)(Interval(x, x * 3.0, closed=False))
    assert_close(width, [2.0, 4.0])
    interval_grad = qb.grad(lambda i: qb.sum(i.high * i.low))(Interval(x, x * 3.0, False))
    assert type(interval_grad) is Interval and interval_grad.closed is False
    assert_close(interval_grad.low, [3.0, 6.0])
    # vmap: NamedTuple and dataclass in_axes/out_axes prefixes.
    xs = qb.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batched = qb.vmap(
        lambda p: Point(p.x * p.y, qb.sum(p.x)), in_axes=(Point(0, None),), out_axes=Point(0, 0)
    )(Point(xs, qb.array([10.0, 100.0])))
    assert type(batched) is Point
    assert_close(batched.x, [[10.0, 200.0], [30.0, 400.0], [50.0, 600.0]])
    assert_close(batched.y, [3.0, 7.0, 11.0])
    layer_out = qb.vmap(apply, in_axes=(ScaledLayer(None, "tanh", None), 0))(
        ScaledLayer(qb.array([1.0, 1.0]), "tanh", 0.5), xs
    )
    assert_close(layer_out, [[math.tanh(0.5 * v) for v in row] for row in xs.tolist()])
    assert_raises(
        ValueError,
        qb.vmap(apply, in_axes=(ScaledLayer(None, "relu", None), 0)),
        ScaledLayer(qb.array([1.0, 1.0]), "tanh", 0.5),
        xs,
        match="does not match the structure",
    )
    # jacobian blocks keep the argument structure.
    jac = qb.jacobian(lambda p: p.x * p.y)(Point(x, x * 2.0))
    assert type(jac) is Point
    assert_close(jac.x, [[2.0, 0.0], [0.0, 4.0]])
    # Unregistered dataclasses and other container subclasses are rejected.

    @dataclasses.dataclass
    class Plain:
        w: object

    assert_raises(TypeError, qb.jit(lambda p: p.w), Plain(x), match="register_dataclass")
    assert_raises(
        TypeError,
        qb.jit(lambda p: p["a"]),
        collections.OrderedDict(a=x),
        match="not a pytree container",
    )
    # dict arguments may have integer keys.
    assert_close(qb.grad(lambda d: qb.sum(d[0] * d[1]))({1: x, 0: x * 2.0})[1], [2.0, 4.0])


def test_jit_static_argnames_and_keyword_arguments():
    x = qb.array([1.0, 2.0])
    traces = []

    def power(x, n, *, offset=0.0, scale):
        traces.append((n, offset, scale))
        return qb.sum(x**n) * scale + offset

    jitted = qb.jit(power, static_argnames=("n", "scale"))
    assert_close(jitted(x, 2, scale=1.0), 5.0)
    # n by keyword binds to its position and reuses the positional trace.
    assert_close(jitted(x, n=2, scale=1.0), 5.0)
    assert_close(jitted(x=x, n=2, scale=1.0), 5.0)
    assert traces == [(2, 0.0, 1.0)]
    assert_close(jitted(x, 3, scale=2.0), 18.0)
    assert len(traces) == 2
    # A keyword-only argument that is not static is rejected with a hint.
    assert_raises(TypeError, jitted, x, 2, scale=1.0, offset=1.0, match="static_argnames")
    assert_raises(TypeError, jitted, x, 2, x=x, scale=1.0, match="multiple values")
    assert_raises(TypeError, jitted, x, 2, scale=[1.0], match="hashable")
    assert_raises(ValueError, qb.jit, power, static_argnames="missing", match="not parameters")
    # static_argnums implies the keyword spelling, and vice versa.
    by_number = qb.jit(power, static_argnums=1, static_argnames="scale")
    assert_close(by_number(x, n=2, scale=1.0), 5.0)
    assert_close(by_number(x, 2, scale=1.0), 5.0)
    # The lowered program checks keyword statics too.
    compiled = jitted.lower(x, 2, scale=1.0).compile()
    assert_close(compiled(x, n=2, scale=1.0), 5.0)
    assert_raises(ValueError, compiled, x, 2, scale=3.0, match="statics")
    # Static keywords reach a plain function traced inside another transform.
    assert_close(qb.grad(lambda x: jitted(x, 2, scale=0.5))(x), [1.0, 2.0])
    # Eviction messages describe keyword statics.
    small = qb.jit(power, static_argnames=("n", "scale"), max_traces=1)
    small(x, 2, scale=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        small(x, 2, scale=2.0)
    assert "with scale=static 1.0" in str(caught[0].message)


def test_transform_decorators_take_keywords_without_the_function():
    @qb.jit(device="cpu", static_argnums=1)
    def scaled(x, factor):
        return x * factor

    assert isinstance(scaled, qb._transforms._Jit)
    assert_close(scaled(qb.array([1.0, 2.0]), 3), [3.0, 6.0])

    @qb.jit(static_argnames="n")
    def repeat(x, n):
        return x * float(n)

    assert_close(repeat(qb.array(2.0), n=4), 8.0)

    @qb.grad(argnums=1)
    def second(a, b):
        return qb.sum(a * b * b)

    assert_close(second(qb.array(2.0), qb.array(3.0)), 12.0)

    @qb.value_and_grad(has_aux=True)
    def with_aux(a):
        return qb.sum(a * a), qb.sum(a)

    (value, aux), gradient = with_aux(qb.array([1.0, 2.0]))
    assert_close(value, 5.0)
    assert_close(aux, 3.0)
    assert_close(gradient, [2.0, 4.0])


def test_error_classes_subclass_the_builtins_raised_before():
    assert issubclass(qb.TracerError, qb.QuablaError) and issubclass(
        qb.TracerError, TypeError
    )
    assert issubclass(qb.RetraceLimitError, ValueError)
    error = qb.UnsupportedOperationError("no", op="jit", device="mlx")
    assert isinstance(error, ValueError) and isinstance(error, NotImplementedError)
    assert (error.op, error.device, str(error)) == ("jit", "mlx", "no")
    assert qb.UnsupportedOperationError("no").op is None


# -- S2: CPU transforms ---------------------------------------


def masked_loss(x, w):
    y = qb.tanh(x * w)
    return qb.sum(qb.where(x > 0.0, y, 0.0) ** 2)


QX = [-1.0, 0.5, 1.0, 2.0]
QW = [0.3, -0.2, 0.1, 0.4]


def test_value_and_grad_matches_tensor_value_and_grad_fn():
    for dtype in [qb.float64, qb.float32]:
        x, w = qb.array(QX, dtype=dtype), qb.array(QW, dtype=dtype)
        specs = [("x", [4], dtype), ("w", [4], dtype)]
        old_value, old_grads = qb.tensor_value_and_grad_fn(masked_loss, specs)(
            {"x": x, "w": w}
        )
        value, (gx, gw) = qb.value_and_grad(masked_loss, argnums=(0, 1))(x, w)
        tolerance = 1e-6 if dtype == qb.float32 else 1e-14
        for tensor in [value, gx, gw]:
            assert tensor.dtype == dtype, tensor.dtype
        assert_close(value, old_value, tolerance)
        assert_close(gx, old_grads["x"], tolerance)
        assert_close(gw, old_grads["w"], tolerance)
        assert_close(qb.grad(masked_loss, 1)(x, w), old_grads["w"], tolerance)
        assert_close(qb.jit(qb.grad(masked_loss))(x, w), old_grads["x"], tolerance)
        grad_scalar = qb.tensor_grad_scalar_fn(masked_loss, specs)({"x": x, "w": w})
        assert_close(qb.grad(masked_loss, -1)(x, w), grad_scalar["w"], tolerance)


def test_argnums_forms_select_and_structure_gradients():
    x, w = qb.array(QX), qb.array(QW)
    gx, gw = qb.grad(masked_loss, argnums=(0, 1))(x, w)
    (only_w,) = qb.grad(masked_loss, argnums=(1,))(x, w)
    assert_close(only_w, gw)
    (w_first, x_second) = qb.grad(masked_loss, argnums=(1, 0))(x, w)
    assert_close(w_first, gw)
    assert_close(x_second, gx)
    assert isinstance(qb.grad(masked_loss, argnums=1)(x, w), qb.Tensor)
    assert_raises(
        ValueError, qb.grad(masked_loss, argnums=2), x, w, match="out of range"
    )
    assert_raises(
        ValueError, qb.grad(masked_loss, argnums=(0, -2)), x, w, match="duplicate"
    )
    assert_raises(
        TypeError, qb.grad(masked_loss, argnums=(0, "w")), x, w, match="tuple of ints"
    )


def init_mlp(dtype):
    def values(count, seed):
        return [math.sin((index + 1) * seed) * 0.9 for index in range(count)]

    return {
        "layers": [
            {
                "w": qb.array(values(6, 0.37), dtype=dtype).reshape([2, 3]),
                "b": qb.zeros([3], dtype),
            },
            {
                "w": qb.array(values(3, 0.71), dtype=dtype).reshape([3, 1]),
                "b": qb.ones([1], dtype),
            },
        ]
    }


def mlp_loss(params, x, y):
    hidden = x
    for index, layer in enumerate(params["layers"]):
        hidden = hidden @ layer["w"] + layer["b"]
        if index + 1 < len(params["layers"]):
            hidden = qb.tanh(hidden)
    return qb.mean((hidden - y) ** 2)


def test_dict_params_of_an_mlp_match_the_name_keyed_helper():
    for dtype in [qb.float64, qb.float32]:
        params = init_mlp(dtype)
        x = qb.array([[0.5, -1.0], [0.25, 0.75], [-0.5, 1.5], [1.0, 0.0]], dtype=dtype)
        y = qb.array([[0.1], [-0.2], [0.3], [0.0]], dtype=dtype)
        value, grads = qb.jit(qb.value_and_grad(mlp_loss))(params, x, y)
        assert qb.tree.flatten(grads)[1] == qb.tree.flatten(params)[1]

        def named(w0, b0, w1, b1, x, y):
            layers = [{"w": w0, "b": b0}, {"w": w1, "b": b1}]
            return mlp_loss({"layers": layers}, x, y)

        layers = params["layers"]
        inputs = {
            "w0": layers[0]["w"],
            "b0": layers[0]["b"],
            "w1": layers[1]["w"],
            "b1": layers[1]["b"],
            "x": x,
            "y": y,
        }
        specs = [(name, tensor.shape, dtype) for name, tensor in inputs.items()]
        old_value, old_grads = qb.tensor_value_and_grad_fn(named, specs)(inputs)
        tolerance = 1e-6 if dtype == qb.float32 else 1e-13
        assert value.dtype == dtype
        assert_close(value, old_value, tolerance)
        for index in range(2):
            for leaf in ["w", "b"]:
                gradient = grads["layers"][index][leaf]
                assert gradient.dtype == dtype
                assert_close(gradient, old_grads[f"{leaf}{index}"], tolerance)


def test_has_aux_returns_auxiliary_pytrees_from_the_same_program():
    calls = []

    def loss(w, x):
        calls.append(1)
        prediction = x * w
        return qb.sum(prediction**2), {
            "prediction": prediction,
            "scale": 2.0,
            "none": None,
        }

    w, x = qb.array([1.0, -2.0]), qb.array([3.0, 4.0])
    (value, aux), grads = qb.value_and_grad(loss, has_aux=True)(w, x)
    assert_close(value, 9.0 + 64.0)
    assert_close(grads, [18.0, -64.0])
    assert_close(aux["prediction"], [3.0, -8.0])
    assert aux["scale"] == 2.0 and aux["none"] is None
    grads, aux = qb.grad(loss, has_aux=True)(w, x)
    assert_close(grads, [18.0, -64.0])
    assert_close(aux["prediction"], [3.0, -8.0])
    assert len(calls) == 2  # one trace per transform, reused by later calls
    qb.grad(loss, has_aux=True)(w, x)
    assert len(calls) == 2
    assert_raises(
        TypeError,
        qb.grad(lambda w: qb.sum(w), has_aux=True),
        w,
        match="(value, aux) pair",
    )


def test_grad_errors_and_python_scalar_arguments():
    x = qb.array([1.0, 2.0])
    assert_raises(
        ValueError, qb.grad(lambda x: x * 2.0), x, match="requires a scalar output"
    )
    assert_raises(
        ValueError, qb.grad(lambda x: qb.sum(x) > 1.0), x, match="bool output"
    )
    assert_raises(TypeError, qb.grad(lambda x: (x, x)), x, match="single scalar array")
    assert_raises(ValueError, qb.grad(lambda x: 3.0), x, match="constant")
    # A differentiated Python scalar becomes a float64 array ...
    derivative = qb.grad(lambda x: x**3)(2.0)
    assert_tensor(derivative, 12.0, qb.float64)
    # ... while other Python scalars are static weak constants: float32 stays float32.
    x32 = qb.array([1.0, 2.0], dtype=qb.float32)
    gradient = qb.grad(lambda x, scale: qb.sum(x * x) * scale)(x32, 3.0)
    assert_tensor(gradient, [6.0, 12.0], qb.float32)
    # bool leaves get no gradient.
    mask = qb.array([True, False])
    gx, gmask = qb.grad(lambda x, m: qb.sum(qb.where(m, x, 0.0)), argnums=(0, 1))(
        x, mask
    )
    assert_close(gx, [1.0, 0.0])
    assert gmask is None
    assert_raises(
        TypeError,
        qb.grad(lambda x: qb.sum(x)),
        "abc",
        match="unsupported argument leaf",
    )


def test_second_order_by_direct_composition():
    # Reverse over reverse (design Exp 3).
    f = lambda x: qb.sin(x) * x**2  # noqa: E731
    exact = -math.sin(1.3) * 1.3**2 + 4.0 * 1.3 * math.cos(1.3) + 2.0 * math.sin(1.3)
    assert_close(qb.grad(qb.grad(f))(1.3), exact, 1e-14)
    assert_close(qb.jit(qb.grad(qb.grad(f)))(qb.array(1.3)), exact, 1e-14)
    # jit takes over the differentiated positions of the transforms inside it.
    assert_close(qb.jit(qb.grad(qb.grad(f)))(1.3), exact, 1e-14)

    # Forward over reverse is a Hessian-vector product.
    def cubic(x):
        return qb.sum(x**3) + qb.sum(x) * x[0]

    x, v = qb.array([1.0, 2.0, -1.0]), qb.array([0.5, -1.0, 2.0])
    _, hvp = qb.jvp(qb.grad(cubic), (x,), (v,))
    old = qb.tensor_hvp_scalar_fn(cubic, [("x", [3])], "x")({"x": x}, v)
    assert_close(hvp, old, 1e-14)


def test_jvp_matches_tensor_jvp_fn_and_keeps_float32():
    def model(x, w):
        return qb.tanh(x @ w)

    for dtype in [qb.float64, qb.float32]:
        x = qb.array([[0.5, -1.0], [2.0, 0.25]], dtype=dtype)
        w = qb.array([[0.3], [-0.7]], dtype=dtype)
        dx = qb.array([[1.0, 0.0], [0.5, -0.5]], dtype=dtype)
        dw = qb.array([[0.2], [0.1]], dtype=dtype)
        specs = [("x", [2, 2], dtype), ("w", [2, 1], dtype)]
        old_value, old_tangent = qb.tensor_jvp_fn(model, specs)(
            {"x": x, "w": w}, {"x": dx, "w": dw}
        )
        value, tangent = qb.jvp(model, (x, w), (dx, dw))
        tolerance = 1e-6 if dtype == qb.float32 else 1e-14
        assert value.dtype == tangent.dtype == dtype
        assert_close(value, old_value, tolerance)
        assert_close(tangent, old_tangent, tolerance)
    # Python number tangents adopt the primal dtype; pytrees of outputs work.
    x32 = qb.array(2.0, dtype=qb.float32)
    (square, pair), (dsquare, dpair) = qb.jvp(
        lambda x: (x * x, [x, 1.0]), (x32,), (1.0,)
    )
    assert_tensor(dsquare, 4.0, qb.float32)
    assert_tensor(dpair[0], 1.0, qb.float32)
    assert pair[1] == 1.0 and dpair[1] == 0.0
    assert_raises(
        TypeError,
        qb.jvp,
        model,
        (x, w),
        (dx, qb.array([[0.2], [0.1]])),
        match="dtype quabla.float32",
    )
    assert_raises(ValueError, qb.jvp, model, (x, w), (dx,), match="must match")
    assert_raises(
        ValueError, qb.jvp, model, (x, w), (dx, [dw]), match="pytree structure"
    )


def test_vjp_matches_tensor_vjp_fn_with_pytree_outputs():
    def model(x, w):
        return qb.tanh(x @ w)

    x = qb.array([[0.5, -1.0], [2.0, 0.25]])
    w = qb.array([[0.3], [-0.7]])
    ct = qb.array([[1.0], [-2.0]])
    old_value, old_grads = qb.tensor_vjp_fn(model, [("x", [2, 2]), ("w", [2, 1])])(
        {"x": x, "w": w}, ct
    )
    value, pullback = qb.vjp(model, x, w)
    assert_close(value, old_value)
    gx, gw = pullback(ct)
    assert_close(gx, old_grads["x"])
    assert_close(gw, old_grads["w"])
    # Several outputs: the pullback sums their VJPs.
    out, pullback = qb.vjp(
        lambda x: {"a": x * 2.0, "b": qb.sum(x)}, qb.array([1.0, 2.0])
    )
    assert_close(out["a"], [2.0, 4.0])
    (gx,) = pullback({"a": qb.array([1.0, 3.0]), "b": 1.0})
    assert_close(gx, [3.0, 7.0])
    assert_raises(ValueError, pullback, [1.0, 1.0], match="pytree structure")
    out, pullback, aux = qb.vjp(
        lambda x: (x * x, x + 1.0), qb.array([3.0]), has_aux=True
    )
    assert_close(pullback(qb.array([1.0]))[0], [6.0])
    assert_close(aux, [4.0])


def test_jacobian_and_hessian_match_the_dense_helpers():
    def model(x, w):
        return qb.tanh(x @ w)

    x = qb.array([[0.5, -1.0], [2.0, 0.25]])
    w = qb.array([[0.3], [-0.7]])
    specs = [("x", [2, 2]), ("w", [2, 1])]
    old = qb.tensor_jacobian_fn(model, specs, "w")({"x": x, "w": w})
    jac = qb.jacobian(model, argnums=1)(x, w)
    assert jac.shape == [2, 1, 2, 1]
    assert_close(jac.reshape([2, 2]), old)
    jx, jw = qb.jacobian(model, argnums=(0, 1))(x, w)
    assert jx.shape == [2, 1, 2, 2] and jw.shape == [2, 1, 2, 1]

    def loss(x):
        return qb.sum(qb.sin(x) * x**2) + qb.sum(x) ** 2

    point = qb.array([0.5, -1.0, 2.0])
    old_hessian = qb.tensor_hessian_scalar_fn(loss, [("x", [3])], "x")({"x": point})
    hessian = qb.hessian(loss)(point)
    assert hessian.shape == [3, 3]
    assert_close(hessian, old_hessian, 1e-13)
    assert_close(qb.jit(qb.hessian(loss))(point), old_hessian, 1e-13)
    point32 = qb.array([0.5, -1.0, 2.0], dtype=qb.float32)
    assert qb.hessian(loss)(point32).dtype == qb.float32
    # A pytree argument gives pytree blocks.
    blocks = qb.jacobian(lambda p: p["a"] * p["b"])(
        {"a": qb.array([2.0]), "b": qb.array([3.0])}
    )
    assert_close(blocks["a"], [[3.0]])
    assert_close(blocks["b"], [[2.0]])
    # Staged through vmap (slice S4), the Hessian differentiates again:
    # d/dx sum(H(x)) of this loss, by central differences of the Hessian.
    third = qb.grad(lambda t: qb.sum(qb.hessian(loss)(t)))(point)
    step = 1e-5
    for index in range(3):
        shift = [step if k == index else 0.0 for k in range(3)]
        plus = qb.hessian(loss)(
            qb.array([p + d for p, d in zip(point.tolist(), shift)])
        )
        minus = qb.hessian(loss)(
            qb.array([p - d for p, d in zip(point.tolist(), shift)])
        )
        expected = (qb.sum(plus).item() - qb.sum(minus).item()) / (2.0 * step)
        assert_close(third[index], expected, 1e-7)


def test_jit_matches_tensor_jit_fn_with_pytrees_and_static_argnums():
    x, w = qb.array(QX), qb.array(QW)
    old = qb.tensor_jit_fn(masked_loss, [("x", [4]), ("w", [4])])({"x": x, "w": w})
    assert_close(qb.jit(masked_loss)(x, w), old)
    assert_close(qb.jit(masked_loss, device="cpu")(x, w), old)
    out = qb.jit(lambda p: {"sum": p[0] + p[1], "list": [p[0] * 2.0, None, 3]})((x, w))
    assert_close(out["sum"], [a + b for a, b in zip(QX, QW)])
    assert out["list"][1:] == [None, 3]

    traces = []

    def scaled(x, power):
        traces.append(power)
        return qb.sum(x**power)

    jitted = qb.jit(scaled, static_argnums=1)
    assert_close(jitted(x, 2), sum(v * v for v in QX))
    assert_close(jitted(x, 3), sum(v**3 for v in QX))
    jitted(x, 2)
    assert traces == [2, 3]
    assert_raises(TypeError, qb.jit(scaled, static_argnums=1), x, [2], match="hashable")
    # Static values that compare equal but trace differently get their own
    # cache entries: 1, True, and 1.0; 0.0 and -0.0; and the same inside tuples.
    by_type = qb.jit(
        lambda x, n: x * (100.0 if n is True else 10.0 if type(n) is float else 1.0),
        static_argnums=1,
    )
    for value, factor in [(1, 1.0), (True, 100.0), (1.0, 10.0), ((1,), 1.0)]:
        assert_close(by_type(x, value), [v * factor for v in QX])
    signed = qb.jit(lambda x, z: x * math.copysign(1.0, z[0]), static_argnums=1)
    assert_close(signed(x, (0.0,)), QX)
    assert_close(signed(x, (-0.0,)), [-v for v in QX])
    assert_raises(ValueError, by_type.lower(x, 3).compile(), x, 3.0, match="statics")
    # A NaN static value matches itself instead of retracing on every call.
    nan_traces = []
    nan_static = qb.jit(lambda x, n: nan_traces.append(n) or x, static_argnums=1)
    for _ in range(3):
        nan_static(x, float("nan"))
    assert len(nan_traces) == 1
    # NumPy data is accepted as an array leaf.
    if np is not None:
        assert_close(qb.jit(masked_loss)(np.array(QX), np.array(QW)), old)
    # jit of a plain function inside another trace traces through it.
    inner = qb.jit(lambda x: x * x)
    assert_close(qb.grad(lambda x: qb.sum(inner(x)))(x), [2.0 * v for v in QX])


def test_jit_device_selection_checks_the_build():
    for device in ("mlx", "cuda:1"):
        target = device.split(":")[0]
        if qb.Compiler().capability(target):
            assert callable(qb.jit(masked_loss, device=device))
        else:
            error = assert_raises(
                qb.UnsupportedOperationError, qb.jit, masked_loss, device=device
            )
            assert (error.op, error.device) == ("jit", device)
            assert isinstance(error, ValueError)
    assert_raises(ValueError, qb.jit, masked_loss, device="tpu", match="device must be")
    assert_raises(ValueError, qb.jit, masked_loss, max_traces=0, match="positive int")


def test_trace_cache_keys_and_lru_eviction():
    import warnings

    traces = []

    def loss(x, scale):
        traces.append((x.shape, x.dtype, scale))
        return qb.sum(x * scale)

    step = qb.jit(loss, max_traces=3)
    step(qb.array([1.0, 2.0]), 2.0)
    step(qb.array([3.0, 4.0]), 2.0)  # same shapes, dtypes, and static value: cached
    assert len(traces) == 1
    step(qb.array([1.0, 2.0, 3.0]), 2.0)  # new shape
    step(qb.array([1.0, 2.0], dtype=qb.float32), 2.0)  # new dtype
    assert len(traces) == 3
    assert traces[2][1] == qb.float32
    # A hit makes the float64[2] trace the most recently used, so a fourth
    # signature evicts the float64[3] trace with a warning instead of failing.
    step(qb.array([5.0, 6.0]), 2.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert step(qb.array([1.0, 2.0]), 5.0).item() == 15.0
    assert len(traces) == 4
    (warning,) = caught
    assert warning.category is qb.RetraceWarning
    assert issubclass(qb.RetraceWarning, UserWarning)
    assert warning.filename == __file__
    message = str(warning.message)
    assert "max_traces=3" in message and "jit(" in message
    assert "evicted its least recently used trace:\n  (float64[3], 2.0)" in message
    assert "new signature:\n  (float64[2], 5.0)" in message
    step(qb.array([7.0, 8.0]), 2.0)  # still cached
    assert len(traces) == 4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step(qb.array([1.0, 2.0, 3.0]), 2.0)  # evicted, so traced again
    assert len(traces) == 5 and len(caught) == 1
    assert "(float32[2], 2.0)" in str(caught[0].message)
    # 1, 1.0, and True are different static values.
    counter = []
    typed = qb.jit(lambda x, s: counter.append(type(s)) or x * s)
    for scale in [1, 1.0, True, 1.0, math.nan, math.nan]:
        typed(qb.array(2.0), scale)
    assert counter == [
        int,
        float,
        bool,
        float,
    ]  # a NaN static value is cached like any other


def test_grad_recreated_per_call_reuses_the_trace():
    traces = []

    def loss(x):
        traces.append(1)
        return qb.sum(x * x)

    x = qb.array([1.0, 2.0])
    for _ in range(3):
        assert_close(qb.grad(loss)(x), [2.0, 4.0])
        assert_close(qb.jit(qb.grad(loss))(x), [2.0, 4.0])
        qb.jvp(loss, (x,), (x,))
    assert len(traces) == 3  # grad, jit(grad), and jvp each traced once


def test_transforms_called_on_tracers_inline_their_graphs():
    f = lambda x: qb.sin(x) * x**2  # noqa: E731
    first = 2.0 * 1.3 * math.sin(1.3) + 1.3**2 * math.cos(1.3)
    second = -math.sin(1.3) * 1.3**2 + 4.0 * 1.3 * math.cos(1.3) + 2.0 * math.sin(1.3)
    df = qb.grad(f)
    # grad inside value_and_grad: the inner VJP graph is inlined and
    # differentiated again.
    value, derivative = qb.value_and_grad(lambda x: df(x))(qb.array(1.3))
    assert_close(value, first, 1e-14)
    assert_close(derivative, second, 1e-14)
    value32, derivative32 = qb.value_and_grad(lambda x: df(x))(
        qb.array(1.3, dtype=qb.float32)
    )
    assert value32.dtype == derivative32.dtype == qb.float32
    assert_close(derivative32, second, 1e-5)
    # jvp, vjp, and jit(grad) inside grad; Python number tangents and
    # cotangents become constants of the primal dtype.
    assert_close(qb.grad(lambda x: qb.jvp(f, (x,), (1.0,))[1])(1.3), second, 1e-14)
    assert_close(qb.grad(lambda x: qb.vjp(f, x)[1](1.0)[0])(1.3), second, 1e-14)
    assert_close(qb.grad(lambda x: qb.jit(qb.grad(f))(x))(1.3), second, 1e-14)
    assert_close(qb.jit(lambda x: df(x) * 2.0)(qb.array(1.3)), 2.0 * first, 1e-14)

    # A traced tangent: d/dv of J(x) v is the gradient of the scalar output.
    def cubic(x):
        return qb.sum(x**3) + qb.sum(x) * x[0]

    x, v = qb.array([1.0, 2.0, -1.0]), qb.array([0.5, -1.0, 2.0])
    by_tangent = qb.grad(lambda x, v: qb.jvp(cubic, (x,), (v,))[1], argnums=1)(x, v)
    assert_close(by_tangent, qb.grad(cubic)(x), 1e-14)
    # HVP through a Python function equals the direct composition.
    _, hvp = qb.jvp(lambda x: qb.grad(cubic)(x) * 3.0, (x,), (v,))
    _, direct = qb.jvp(qb.grad(cubic), (x,), (v,))
    assert_close(hvp, [3.0 * value for value in direct.tolist()], 1e-14)
    # A traced cotangent of a nested vjp.
    pullback_grad = qb.grad(lambda c, x: qb.sum(qb.vjp(qb.sin, x)[1](c)[0]))(v, x)
    assert_close(pullback_grad, [math.cos(value) for value in x.tolist()], 1e-14)
    # With eager primals, a traced cotangent inlines the pullback with the
    # primals as constants (slice S3b).
    eager_pullback_grad = qb.grad(lambda c: qb.sum(qb.vjp(qb.sin, x)[1](c)[0]))(v)
    assert_close(eager_pullback_grad, [math.cos(value) for value in x.tolist()], 1e-14)

    # Coordinate derivatives inside a loss, then gradients in the weights,
    # with pytree parameters, has_aux, and a static Python scalar.
    def model(x, p, scale):
        y = qb.tanh(x * p["a"]) * p["b"] * scale
        return qb.sum(y), qb.mean(y)

    def loss(p, x):
        du, aux = qb.grad(model, has_aux=True)(x, p, 2.0)
        return qb.sum(du**2) + aux

    params = {"a": qb.array(0.7), "b": qb.array(-1.3)}
    points = qb.array([0.1, -0.4, 0.9])
    value, grads = qb.value_and_grad(loss)(params, points)

    def reference(a, b):
        xs = points.tolist()
        du = [2.0 * a * b / math.cosh(a * x) ** 2 for x in xs]
        return sum(d * d for d in du) + sum(
            2.0 * b * math.tanh(a * x) for x in xs
        ) / len(xs)

    assert_close(value, reference(0.7, -1.3), 1e-13)
    h = 1e-6
    fd_a = (reference(0.7 + h, -1.3) - reference(0.7 - h, -1.3)) / (2.0 * h)
    fd_b = (reference(0.7, -1.3 + h) - reference(0.7, -1.3 - h)) / (2.0 * h)
    assert_close(grads["a"], fd_a, 1e-8)
    assert_close(grads["b"], fd_b, 1e-8)


def test_inlined_transforms_cache_their_staged_graph_and_reject_unsupported_calls():
    traces = []

    def u(x, w):
        traces.append(1)
        return qb.sin(x * w)

    u_xx = qb.grad(qb.grad(u))
    x, w = qb.array([0.1, 0.2, 0.3]), qb.array(1.5)
    total = qb.jit(lambda x, w: u_xx(x[0], w) + u_xx(x[1], w) + u_xx(x[2], w))(x, w)
    assert_close(
        total, sum(-(1.5**2) * math.sin(1.5 * p) for p in [0.1, 0.2, 0.3]), 1e-14
    )
    assert len(traces) == 1  # three calls with one signature stage u once

    # Eager arguments and tangents of an inner call bind as constants (S3b).
    assert_close(qb.jit(lambda x: qb.grad(u)(x[0], w))(x), 1.5 * math.cos(0.15), 1e-14)
    v_eager = qb.array([1.0, 1.0, 1.0])
    assert_close(
        qb.jit(lambda t: qb.jvp(qb.sin, (t,), (v_eager,))[1])(x),
        [math.cos(value) for value in x.tolist()],
        1e-14,
    )
    error = assert_raises(
        qb.UnsupportedOperationError,
        qb.tensor_vmap_fn,
        lambda row: qb.grad(lambda y: qb.sum(y * y))(row),
        [("row", [2])],
        3,
        match="batched tracer",
    )
    assert error.op == "nested transform"
    # Second-order reverse mode through a region stays an explicit error.

    def scan_loss(initial, scale):
        carry, outputs = qb.tensor_scan_region(
            0,
            3,
            lambda index, current, s: (current * current * s + index, current * s),
            initial,
            [scale],
        )
        return carry + outputs.sum()

    g = qb.grad(scan_loss, argnums=1)
    assert_raises(
        ValueError,
        qb.grad(lambda i, s: g(i, s), argnums=1),
        qb.array(0.4),
        qb.array(0.8),
        match="not implemented",
    )
    # Forward over reverse through the inlined region matches the direct
    # composition, and two splices in one trace stay separate executions.
    i0, s0 = qb.array(0.4), qb.array(0.8)
    _, direct = qb.jvp(g, (i0, s0), (0.0, 1.0))
    _, nested = qb.jvp(lambda i, s: g(i, s) * 2.0, (i0, s0), (0.0, 1.0))
    assert_close(nested, 2.0 * direct.item(), 1e-13)
    both = qb.jit(lambda i, s: g(i, s) + g(s, i))(i0, s0)
    assert_close(both, g(i0, s0).item() + g(s0, i0).item(), 1e-13)


PINN_POINTS = 8


def pinn_u(x, w):  # one collocation point, scalar x
    return qb.sin(x * w)


pinn_u_xx = qb.grad(qb.grad(pinn_u))  # exact d2u/dx2 at one point


def pinn_loss(w, x):
    # The per-point form of design 2.4 until vmap (slice S4) batches it.
    u_xx = qb.stack([pinn_u_xx(x[i], w) for i in range(x.shape[0])], 0)
    return qb.mean((u_xx + math.pi**2 * qb.sin(math.pi * x)) ** 2)


def readme_symbolic_jvp_loss():
    """The v0.1 README "At a Glance" loss: two symbolic coordinate JVPs."""
    u = qb.trace_tensor(
        lambda x, w: (x * w).sin(), [("x", [PINN_POINTS, 1]), ("w", [1, 1])]
    )
    u_xx = u.symbolic_jvp("x").symbolic_jvp("x")
    x = u_xx.graph.input("x")
    return (u_xx.output + math.pi**2 * (math.pi * x).sin()).powi(2).mean()


def test_pinn_loss_with_a_nested_second_derivative_reproduces_the_readme():
    x, w = qb.linspace(0.05, 0.95, PINN_POINTS), qb.array(2.5)
    step = qb.jit(qb.value_and_grad(pinn_loss))
    value, grad_w = step(w, x)
    # Same loss and gradient as the README's symbolic_jvp().symbolic_jvp() path.
    plan = readme_symbolic_jvp_loss().compile_cpu()
    readme_inputs = {"x": x.reshape([PINN_POINTS, 1]), "w": w.reshape([1, 1])}
    old_value, old_grads = plan.evaluate_value_and_vjp(
        readme_inputs, qb.Tensor([], [1.0])
    )
    assert_close(value, old_value.item(), 1e-12)
    assert_close(grad_w, old_grads["w"].item(), 1e-12)
    params, adam = {"w": w}, qb.Adam(learning_rate=0.05)
    for _ in range(300):
        value, grad_w = step(params["w"], x)
        params = adam.step(params, {"w": grad_w})
    assert abs(params["w"].item() - math.pi) < 1e-5, params["w"].item()
    assert f"{params['w'].item():.6f}" == "3.141593"
    assert value.item() < 1e-10


def test_pinn_loss_with_eager_forcing_terms_reproduces_the_readme():
    # The first S3b repro: the points are Python floats, so `qb.sin(pi * x)`
    # is an eager Tensor that meets a tracer and is captured as a constant.
    def loss(w, xs):
        residuals = [
            (pinn_u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2 for x in xs
        ]
        return qb.mean(qb.stack(residuals, 0))

    xs = qb.linspace(0.05, 0.95, PINN_POINTS).tolist()
    step = qb.jit(qb.value_and_grad(loss))
    value, grad_w = step(qb.array(2.5), xs)
    traced_value, traced_grad_w = qb.jit(qb.value_and_grad(pinn_loss))(
        qb.array(2.5), qb.array(xs)
    )
    assert_close(value, traced_value, 1e-12)
    assert_close(grad_w, traced_grad_w, 1e-12)
    w, adam = {"w": qb.array(2.5)}, qb.Adam(learning_rate=0.05)
    for _ in range(300):
        value, grad_w = step(w["w"], xs)
        w = adam.step(w, {"w": grad_w})
    assert f"{w['w'].item():.6f}" == "3.141593", w["w"].item()
    assert value.item() < 1e-10
    # The single-point form of the issue report.
    value, grad_w = qb.jit(
        qb.value_and_grad(
            lambda w, xs: (
                (
                    qb.grad(qb.grad(lambda x, w: qb.sin(x * w)))(xs[0], w)
                    + math.pi**2 * qb.sin(math.pi * xs[0])
                )
                ** 2
            )
        )
    )(qb.array(2.5), [0.3])
    residual = -(2.5**2) * math.sin(0.75) + math.pi**2 * math.sin(0.3 * math.pi)
    assert_close(value, residual**2, 1e-12)


def test_closures_over_arrays_are_captured_in_transforms():
    points = qb.linspace(0.0, 1.0, 8)
    xs = points.tolist()

    def f(w):
        return (qb.sin(points * w)).sum()

    w = qb.array(1.3)
    expected_value = sum(math.sin(x * 1.3) for x in xs)
    expected_grad = sum(x * math.cos(x * 1.3) for x in xs)
    assert_close(qb.jit(f)(w), expected_value, 1e-14)
    assert_close(qb.grad(f)(w), expected_grad, 1e-14)
    value, gradient = qb.value_and_grad(f)(w)
    assert_close(value, expected_value, 1e-14)
    assert_close(gradient, expected_grad, 1e-14)
    assert_close(qb.jit(qb.grad(f))(w), expected_grad, 1e-14)
    # Nested: grad(grad), a jvp of the gradient, and a closure inside an
    # inner transform that an outer one differentiates again.
    expected_second = -sum(x * x * math.sin(x * 1.3) for x in xs)
    assert_close(qb.grad(qb.grad(f))(w), expected_second, 1e-13)
    assert_close(qb.jvp(qb.grad(f), (w,), (1.0,))[1], expected_second, 1e-13)
    assert_close(
        qb.grad(lambda w: qb.grad(f)(w) * 2.0)(w), 2.0 * expected_second, 1e-13
    )
    assert_close(qb.hessian(f)(w), expected_second, 1e-13)

    # Like jax.jit, the array is baked in when the function is traced:
    # rebinding the closed-over name does not retrace.
    scale = qb.array([1.0, 2.0])
    scaled = qb.jit(lambda t: t * scale)
    x = qb.array([3.0, 4.0])
    assert scaled(x).tolist() == [3.0, 8.0]
    scale = qb.array([10.0, 20.0])  # noqa: F841 (read by the lambda on a retrace)
    assert scaled(x).tolist() == [3.0, 8.0]
    assert scaled(qb.array([1.0, 1.0, 1.0]).slice(0, 0, 2).to_tensor()).tolist() == [
        1.0,
        2.0,
    ]


def test_captured_constants_keep_their_dtype():
    c32 = qb.array([0.5, -2.0], dtype=qb.float32)
    x32 = qb.array([1.5, 0.25], dtype=qb.float32)
    result = qb.jit(lambda t: t * c32 + 1.0)(x32)
    assert_tensor(result, (x32 * c32 + 1.0).tolist(), qb.float32)
    gradient = qb.grad(lambda t: qb.sum(qb.sin(t * c32)))(x32)
    assert gradient.dtype == qb.float32
    assert_close(
        gradient,
        [c * math.cos(x * c) for x, c in zip(x32.tolist(), c32.tolist())],
        1e-6,
    )
    # A float64 constant is strong: with a float32 tracer it is a dtype error
    # that asks for astype, as the same eager operation is.
    c64 = qb.array([0.5, -2.0])
    for function in [
        lambda t: t * c64,
        lambda t: c64 * t,
        lambda t: qb.maximum(t, c64),
    ]:
        assert_raises(ValueError, qb.jit(function), x32, match="astype")
    assert_raises(ValueError, lambda: x32 * c64, match="astype")
    assert qb.jit(lambda t: t * c64.astype(qb.float32))(x32).dtype == qb.float32
    # Python scalars stay weak; bool masks select in either role.
    assert qb.jit(lambda t: t * 3.0)(x32).dtype == qb.float32
    mask = qb.array([True, False])
    assert_tensor(qb.jit(lambda t: qb.where(mask, t, 0.0))(x32), [1.5, 0.0], qb.float32)
    assert_tensor(qb.jit(lambda t: (t > 1.0) & mask)(x32), [True, False], qb.bool_)
    assert_tensor(qb.jit(lambda t: mask | (t > 1.0))(x32), [True, False], qb.bool_)


def assert_constant_device_parity(value_and_grad_fn):
    c = qb.array([0.5, -1.25, 2.0, 0.75])
    x = qb.array([0.1, 0.2, -0.3, 0.4])
    specs = [("x", [4])]

    def loss(t):
        return (qb.sin(t * c) + c * t).sum()

    cpu_value, cpu_gradients = qb.tensor_value_and_grad_fn(loss, specs)({"x": x})
    device = value_and_grad_fn(loss, specs, ["x"])
    for _ in range(2):  # the second call reuses the uploaded constant
        value, gradients = device({"x": x})
        assert_close(value, cpu_value, 1e-5)
        assert_close(gradients["x"], cpu_gradients["x"], 1e-5)


def test_mlx_captured_constants_match_cpu():
    require("mlx")
    assert_constant_device_parity(qb.tensor_value_and_grad_mlx_fn)


def test_cuda_captured_constants_match_cpu():
    require("cuda")
    assert_constant_device_parity(
        lambda loss, specs, names: qb.tensor_value_and_grad_cuda_fn(
            loss, specs, names, 0
        )
    )


def test_nested_second_derivatives_grow_the_graph_linearly_in_the_points():
    from quabla import _transforms

    def sizes(points):
        x, w = qb.linspace(0.05, 0.95, points), qb.array(2.5)
        staged = qb.value_and_grad(pinn_loss)._stage(
            (tuple, (_transforms._LEAF, _transforms._LEAF)), [w, x], ["w", "x"]
        )
        executable = staged.graph._compile_cpu(
            [output for output in staged.outputs], staged.input_names
        )
        return staged.graph._node_count, executable.node_count

    counts = {points: sizes(points) for points in [1, 2, 4, 8]}
    for index in range(2):
        per_point = counts[2][index] - counts[1][index]
        assert counts[4][index] - counts[2][index] == 2 * per_point, counts
        assert counts[8][index] - counts[4][index] == 4 * per_point, counts


def test_legacy_grad_and_jit_call_forms_keep_v0_1_results():
    def model(a, b):
        return a @ b

    specs = [("a", (2, 3)), ("b", (3, 2))]
    values = {
        "a": qb.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "b": qb.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    }
    cotangent = qb.Matrix([[1.0, 0.5], [-1.0, 2.0]])
    for gradients in [
        qb.grad(model, specs, values, cotangent),
        qb.grad(model, tuple(specs), values, output_cotangent=cotangent),
        qb.grad(
            function=model, input_specs=specs, values=values, output_cotangent=cotangent
        ),
    ]:
        assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
        assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]

    @qb.jit([("a", (2, 2))])
    def double(a):
        return a + a

    assert type(double).__name__ == "JitFunction"
    assert double({"a": qb.Matrix([[1.0, 2.0], [3.0, 4.0]])}).to_list() == [
        [2.0, 4.0],
        [6.0, 8.0],
    ]
    assert type(qb.jit(input_specs=[("a", (2, 2))])).__name__ == "JitTransform"
    # A list of ints is v0.2 argnums, not a v0.1 input spec list.
    x, y = qb.array([1.0, 2.0]), qb.array([3.0, 4.0])
    grad_x, grad_y = qb.grad(lambda a, b: qb.sum(a * b), [0, 1])(x, y)
    assert (grad_x.tolist(), grad_y.tolist()) == ([3.0, 4.0], [1.0, 2.0])
    assert qb.grad is not qb._quabla.grad and qb.jit is not qb._quabla.jit
    assert list(inspect.signature(qb.grad).parameters) == ["fun", "argnums", "has_aux"]


def test_tracer_escapes_raise_tracer_error():
    x = qb.array([1.0, 2.0])
    escapes = [
        (lambda t: t if t else t, "cannot drive Python control flow"),
        (lambda t: float(t), "float() needs a concrete value"),
        (lambda t: int(t), "int() needs a concrete value"),
        (lambda t: [0.0, 1.0][t], "an integer index needs a concrete value"),
        (lambda t: t.item(), "item() needs a concrete value"),
        (lambda t: t.tolist(), "tolist() needs a concrete value"),
        (lambda t: t.numpy(), "numpy() needs a concrete value"),
    ]
    if np is not None:
        escapes += [
            (
                lambda t: np.asarray(t),
                "conversion to a NumPy array needs a concrete value",
            ),
            (lambda t: np.sin(t), "conversion to a NumPy array needs a concrete value"),
        ]
    for function, message in escapes:
        error = assert_raises(qb.TracerError, qb.jit(function), x, match=message)
        assert isinstance(error, TypeError)
    # The v0.1 helpers raise the same class, which is still a TypeError.
    assert_raises(
        TypeError,
        qb.tensor_jit_fn,
        lambda t: t if t else t,
        [("x", [])],
        match="control flow",
    )


def test_eager_arrays_meeting_tracers_are_captured_as_constants():
    x = qb.array([1.0, 2.0])
    data = qb.array([3.0, 4.0])
    matrix = qb.array([[2.0, 1.0], [1.0, 3.0]])
    cases = [
        (lambda t: t * data, [3.0, 8.0]),
        (lambda t: data * t, [3.0, 8.0]),
        (lambda t: data - t, [2.0, 2.0]),
        (lambda t: data / t, [3.0, 2.0]),
        (lambda t: data**t, [3.0, 16.0]),
        (lambda t: t**data, [1.0, 16.0]),
        (lambda t: qb.power(data, t), [3.0, 16.0]),
        (lambda t: (t > data).astype(qb.float64), [0.0, 0.0]),
        (lambda t: data.less(t).astype(qb.float64), [0.0, 0.0]),
        (lambda t: qb.maximum(t, data), [3.0, 4.0]),
        (lambda t: qb.minimum(data, t), [1.0, 2.0]),
        (lambda t: qb.where(t > 1.0, t, data), [3.0, 2.0]),
        (lambda t: qb.where(data > 3.5, t, 0.0), [0.0, 2.0]),
        (lambda t: (t.reshape([1, 2]) @ data.reshape([2, 1])).reshape([1]), [11.0]),
        (lambda t: (data.reshape([1, 2]) @ t.reshape([2, 1])).reshape([1]), [11.0]),
        (lambda t: qb.matmul(matrix, t.reshape([2, 1])).reshape([2]), [4.0, 7.0]),
        (lambda t: qb.solve(matrix, t.reshape([2, 1])).reshape([2]), [0.2, 0.6]),
        (lambda t: matrix.solve(t.reshape([2, 1])).reshape([2]), [0.2, 0.6]),
        (lambda t: qb.concat([t, data], 0), [1.0, 2.0, 3.0, 4.0]),
        (lambda t: qb.stack([data, t], 0).reshape([4]), [3.0, 4.0, 1.0, 2.0]),
        (
            lambda t: qb.einsum("ij,jk->ik", [matrix, t.reshape([2, 1])]).reshape([2]),
            [4.0, 7.0],
        ),
        (lambda t: t * data.slice(0, 0, 2), [3.0, 8.0]),
        (lambda t: t + qb.sin(0.5), [1.0 + math.sin(0.5), 2.0 + math.sin(0.5)]),
    ]
    for function, expected in cases:
        assert_close(qb.jit(function)(x), expected, 1e-14)
    # The same operations differentiate through the constants.
    assert_close(qb.grad(lambda t: qb.sum(data * t * t))(x), [6.0, 16.0], 1e-14)
    assert_close(
        qb.grad(lambda t: qb.sum(qb.solve(matrix, t.reshape([2, 1]))))(x),
        [0.4, 0.2],
        1e-14,
    )
    # Other operand types keep their TypeError.
    error = assert_raises(
        TypeError, qb.jit(lambda t: t + "a"), x, match="numeric scalar operand"
    )
    assert not isinstance(error, qb.TracerError)
    error = assert_raises(TypeError, lambda: data + "a", match="numeric scalar operand")
    assert not isinstance(error, qb.TracerError)
    assert_raises(TypeError, lambda: data @ "a")
    assert_raises(TypeError, lambda: data.matmul("a"), match="matmul expects a Tensor")


# -- vmap (slice S4) -------------------------------------------------------------------


def per_example(function, mapped, *args):
    """`function` applied example by example along axis 0 of the arguments
    whose `mapped` flag is set, with the results stacked along axis 0."""
    size = next(arg.shape[0] for arg, flag in zip(args, mapped) if flag)
    rows = [
        function(
            *[qb.asarray(arg[i]) if flag else arg for arg, flag in zip(args, mapped)]
        )
        for i in range(size)
    ]
    return qb.stack([qb.asarray(row) for row in rows], 0)


def test_vmap_in_axes_and_out_axes_select_the_batch_axis():
    x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = qb.array([0.5, -1.0, 2.0])
    assert_close(qb.vmap(lambda a, b: a * b, in_axes=(0, None))(x, w), x * w)
    assert_close(qb.vmap(qb.sum)(x), [6.0, 15.0])
    assert_close(qb.vmap(qb.sum, in_axes=1)(x), [5.0, 7.0, 9.0])
    assert_close(qb.vmap(qb.sum, in_axes=-1)(x), [5.0, 7.0, 9.0])
    assert_close(qb.vmap(lambda a: a * 2.0, in_axes=1, out_axes=1)(x), x * 2.0)
    assert_close(qb.vmap(lambda a: a * 2.0, out_axes=-1)(x), (x * 2.0).transpose())
    # A mapped scalar meets an unmapped vector: each example is a vector.
    s = qb.array([1.0, 2.0])
    assert_close(
        qb.vmap(lambda a, b: a * b, in_axes=(0, None))(s, w),
        [w.tolist(), (w * 2.0).tolist()],
    )
    # A list works like a tuple, and in_axes is a pytree prefix per argument.
    params = {"scale": qb.array([2.0, 3.0]), "shift": qb.array(1.0)}

    def affine(p, v):
        return p["scale"] * v + p["shift"]

    points = qb.array([[1.0, 2.0], [3.0, 4.0]])
    batched = qb.vmap(affine, in_axes=[{"scale": 0, "shift": None}, 0])(params, points)
    assert_close(batched, [[3.0, 5.0], [10.0, 13.0]])
    # Pytree results with per-leaf out_axes; out_axes=None for an output that
    # does not depend on the mapped arguments, which is not broadcast.
    both = qb.vmap(
        lambda a, b: {"prod": a * b, "w": b + 0.0},
        in_axes=(1, None),
        out_axes={"prod": 1, "w": None},
    )
    result = both(x, qb.array(3.0))
    assert_close(result["prod"], x * 3.0)
    assert_close(result["w"], 3.0)
    # Unmapped and constant results are broadcast over the batch.
    const = qb.vmap(
        lambda a, b: (b * 2.0, 7.0, qb.array([1.0, 2.0])), in_axes=(0, None)
    )(s, qb.array(0.5))
    assert_close(const[0], [1.0, 1.0])
    assert_close(const[1], [7.0, 7.0])
    assert_close(const[2], [[1.0, 2.0], [1.0, 2.0]])
    # A static Python scalar with in_axes None stays static.
    assert_close(qb.vmap(lambda a, k: a * k, in_axes=(0, None))(s, 3.0), [3.0, 6.0])
    # The batch size is part of the signature: a new size retraces.
    assert_close(qb.vmap(qb.sum)(qb.array([[1.0], [2.0], [3.0]])), [1.0, 2.0, 3.0])


def test_vmap_rejects_invalid_axes_and_unbatchable_ops():
    x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    add = qb.vmap(lambda a, b: a + b)
    assert_raises(
        ValueError, add, x, qb.array([1.0, 2.0, 3.0]), match="inconsistent sizes"
    )
    assert_raises(
        ValueError, qb.vmap(qb.sum, in_axes=None), x, match="at least one mapped"
    )
    assert_raises(
        ValueError, qb.vmap(lambda a, k: a * k), x, 2.0, match="static argument"
    )
    assert_raises(ValueError, qb.vmap(qb.sum, in_axes=2), x, match="out of range")
    assert_raises(
        ValueError, qb.vmap(add, in_axes=(0, 0, 0)), x, x, match="does not match"
    )
    assert_raises(
        ValueError, qb.vmap(lambda a: a, out_axes=None), x, match="out_axes=None"
    )
    assert_raises(ValueError, qb.vmap(lambda a: a, out_axes=3), x, match="out of range")
    assert_raises(TypeError, qb.vmap, qb.sum, in_axes="0")
    # A mapped solve batches over the leading axis (an unmapped operand is
    # broadcast); ops without a batching rule raise UnsupportedOperationError (D16).
    matrix = qb.array([[2.0, 0.0], [1.0, 3.0]])
    rhs = qb.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    assert_close(
        qb.vmap(lambda m, b: qb.solve(m, b), in_axes=(None, 0))(matrix, rhs),
        [[[0.5], [0.5]], [[1.5], [5.0 / 6.0]]],
    )
    # An unmapped solve is fine.
    unmapped_solve = qb.vmap(
        lambda m, b, k: qb.solve(m, b) * k, in_axes=(None, None, 0)
    )
    assert_close(
        unmapped_solve(matrix, qb.asarray(rhs[0]), qb.array([1.0, 2.0])),
        [[[0.5], [0.5]], [[1.0], [1.0]]],
    )

    def looped(initial, scale):
        return qb.tensor_fori_loop_region(
            0, 3, lambda i, c, s: c + i * s, initial, [scale]
        )

    # A mapped loop region batches its body, as the deprecated helper does.
    assert_close(qb.vmap(looped)(qb.array([1.0, 2.0]), qb.array([2.0, 3.0])), [7.0, 11.0])
    old = qb.tensor_vmap_fn(looped, [("initial", []), ("scale", [])], 2)
    assert_close(
        old({"initial": qb.array([1.0, 2.0]), "scale": qb.array([2.0, 3.0])}),
        [7.0, 11.0],
    )
    # A mapped cond predicate selects each example's branch.
    assert_close(
        qb.vmap(lambda x: qb.cond(x > 0.0, lambda t: t * 2.0, lambda t: -t, x))(
            qb.array([1.0, -2.0])
        ),
        [2.0, 2.0],
    )


def test_vmap_matches_the_tensor_vmap_helpers():
    def model(x, weight):
        return x.matmul(weight).tanh()

    x = qb.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [1.0, 2.0]], [[3.0, 0.0], [0.0, 3.0]]]
    )
    weight = qb.array([[[1.0], [-1.0]], [[1.0], [0.5]], [[2.0], [1.0]]])
    old = qb.tensor_vmap_fn(model, [("x", [2, 2]), ("weight", [2, 1])], 3)
    assert_close(qb.vmap(model)(x, weight), old({"x": x, "weight": weight}), 1e-15)
    # The helpers align a mapped [B] operand with a mapped [B, 3] one by
    # trailing axes, so a per-example scalar times a vector fails there;
    # quabla.vmap batches the per-example graph and handles it.
    rows = qb.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])
    scales = qb.array([0.8, 1.1])
    assert_raises(
        ValueError,
        qb.tensor_vmap_fn,
        lambda r, c: r * c,
        [("r", [3]), ("c", [])],
        2,
        match="cannot broadcast",
    )
    assert_close(
        qb.vmap(lambda r, c: r * c)(rows, scales),
        [[0.08, 0.16, 0.24], [-0.44, 0.55, 0.66]],
    )

    def dot(a, b):
        return (a * b).sum()

    xt = qb.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    b = qb.array([2.0, -1.0])
    old = qb.tensor_vmap_fn(
        dot, [("x", [2]), ("weight", [2])], 3, in_axes=[-1, None], out_axis=-1
    )
    assert_close(
        qb.vmap(dot, in_axes=(-1, None), out_axes=-1)(xt, b),
        old({"x": xt, "weight": b}),
    )
    cube = qb.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    old = qb.tensor_vmap_fn(
        lambda t: t.transpose().mean(axis=1),
        [("x", [2, 3])],
        2,
        in_axes=[2],
        out_axis=1,
    )
    new = qb.vmap(lambda t: t.transpose().mean(axis=1), in_axes=2, out_axes=1)(cube)
    assert_close(new, old({"x": cube}))

    # jvp of vmap: forward mode over the batched function.
    def f(x, s):
        return qb.sin(x * s) * s + x**2

    # `s` has one element per example: the helpers cannot batch a scalar
    # against a vector (above).
    specs = [("x", [3]), ("s", [1])]
    values = {
        "x": qb.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]]),
        "s": qb.array([[0.8], [1.1]]),
    }
    tangents = {
        "x": qb.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]),
        "s": qb.array([[0.3], [-0.2]]),
    }
    old_out, old_tangent = qb.tensor_vmap_jvp_fn(f, specs, 2)(values, tangents)
    out, tangent = qb.jvp(
        qb.vmap(f), (values["x"], values["s"]), (tangents["x"], tangents["s"])
    )
    assert_close(out, old_out, 1e-15)
    assert_close(tangent, old_tangent, 1e-14)
    # vjp of vmap: mapped cotangents give per-example gradients, and the
    # gradient of an unmapped argument sums over the batch, as the helper's.
    cotangent = qb.array([[1.0, -0.5, 0.25], [-0.75, 0.5, 1.0]])
    old_out, old_grads = qb.tensor_vmap_vjp_fn(f, specs, 2)(values, cotangent)
    out, pullback = qb.vjp(qb.vmap(f), values["x"], values["s"])
    gx, gs = pullback(cotangent)
    assert_close(out, old_out, 1e-15)
    assert_close(gx, old_grads["x"], 1e-14)
    assert_close(gs, old_grads["s"], 1e-14)
    s_shared = qb.array([0.9])
    old_out, old_grads = qb.tensor_vmap_vjp_fn(f, specs, 2, in_axes=[0, None])(
        {"x": values["x"], "s": s_shared}, cotangent
    )
    out, pullback = qb.vjp(qb.vmap(f, in_axes=(0, None)), values["x"], s_shared)
    gx, gs = pullback(cotangent)
    assert_close(gx, old_grads["x"], 1e-14)
    assert_close(gs, old_grads["s"], 1e-14)
    # Where the semantics differ: vmap(grad) with respect to an unmapped
    # argument gives one gradient per example (JAX); the helper's VJP sums
    # them over the batch.
    per_point = qb.vmap(
        qb.grad(lambda x, s: qb.sum(f(x, s)), argnums=1), in_axes=(0, None)
    )
    per_example = per_point(values["x"], s_shared)
    assert per_example.shape == [2, 1]
    ones = qb.ones([2, 3])
    _, summed = qb.tensor_vmap_vjp_fn(f, specs, 2, in_axes=[0, None])(
        {"x": values["x"], "s": s_shared}, ones
    )
    assert_close(qb.sum(per_example, 0), summed["s"], 1e-14)

    # HVP of the summed per-example scalar loss with respect to a mapped input.
    def g(x, s):
        return qb.sum(qb.tanh(x * s) * x)

    direction = qb.array([[0.7, -0.4, 0.2], [0.1, 0.3, -0.5]])
    old_hvp = qb.tensor_vmap_hvp_scalar_fn(g, specs, 2, "x")(values, direction)
    batched_loss = qb.grad(lambda x, s: qb.sum(qb.vmap(g)(x, s)))
    _, hvp = qb.jvp(
        batched_loss, (values["x"], values["s"]), (direction, qb.zeros([2, 1]))
    )
    assert_close(hvp, old_hvp, 1e-14)
    # The same HVP per example, as vmap of a jvp of grad.
    per_example_hvp = qb.vmap(
        lambda x, s, v: qb.jvp(qb.grad(g), (x, s), (v, qb.zeros([1])))[1]
    )
    assert_close(per_example_hvp(values["x"], values["s"], direction), old_hvp, 1e-14)


def test_vmap_composes_with_grad_jit_and_itself():
    def f(x, w):
        return qb.sum(qb.tanh(x * w) * x)

    xs = qb.array(
        [[0.1, -0.2, 0.3], [0.5, 0.25, -1.0], [2.0, -0.7, 0.4], [0.0, 1.0, -1.5]]
    )
    w = qb.array([0.9, -0.3, 1.2])
    # vmap(grad) with respect to the mapped argument, and jit(vmap(...)).
    per_x = qb.vmap(qb.grad(f), in_axes=(0, None))
    for batched in [per_x, qb.jit(per_x)]:
        assert_close(
            batched(xs, w), per_example(qb.grad(f), (True, False), xs, w), 1e-14
        )
    # Per-example gradients with respect to the unmapped argument (JAX).
    per_w = qb.vmap(qb.grad(f, argnums=1), in_axes=(0, None))(xs, w)
    assert per_w.shape == [4, 3]
    assert_close(per_w, per_example(qb.grad(f, argnums=1), (True, False), xs, w), 1e-14)
    both = qb.vmap(qb.grad(f, argnums=(0, 1)), in_axes=(0, None))(xs, w)
    assert_close(both[1], per_w, 1e-15)
    # grad of a loss over vmap with respect to an unmapped parameter: an
    # ordinary gradient of a scalar, equal to the per-point loop form.

    def loss(w, xs):
        return qb.mean(qb.vmap(f, in_axes=(0, None))(xs, w) ** 2)

    def loop_loss(w, xs):
        return qb.mean(qb.stack([f(xs[i], w) for i in range(xs.shape[0])], 0) ** 2)

    value, gradient = qb.value_and_grad(loss)(w, xs)
    loop_value, loop_gradient = qb.value_and_grad(loop_loss)(w, xs)
    assert_close(value, loop_value, 1e-14)
    assert_close(gradient, loop_gradient, 1e-14)
    assert_close(qb.jit(qb.grad(loss))(w, xs), loop_gradient, 1e-14)
    # vmap inside a plain jit function, on tracers and eager arrays.
    inside = qb.jit(lambda xs: qb.vmap(f, in_axes=(0, None))(xs, w))
    assert_close(inside(xs), per_example(f, (True, False), xs, w), 1e-14)
    # Nested vmap: an outer batch of parameter vectors.
    ws = qb.array([[0.9, -0.3, 1.2], [0.1, 0.2, 0.3]])
    nested = qb.vmap(qb.vmap(f, in_axes=(0, None)), in_axes=(None, 0))(xs, ws)
    assert nested.shape == [2, 4]
    for j in range(2):
        expected = per_example(f, (True, False), xs, qb.asarray(ws[j]))
        assert_close(qb.asarray(nested[j]), expected, 1e-14)
    # A jvp of grad inside vmap: per-example Hessian-vector products.
    hvp_rows = qb.vmap(
        lambda x: qb.jvp(qb.grad(lambda t: qb.sum(t**3)), (x,), (qb.ones([3]),))[1]
    )(xs)
    assert_close(hvp_rows, xs * 6.0, 1e-14)
    # Captured eager arrays inside vmap are unmapped constants.
    offset = qb.array([1.0, 2.0, 3.0])
    assert_close(qb.vmap(lambda x: x + offset)(xs), xs + offset)
    weighted = qb.vmap(lambda x: qb.sum(x * offset))(xs)
    assert_close(weighted, per_example(lambda x: qb.sum(x * offset), (True,), xs))


def test_vmap_keeps_float32_and_the_dtype_rules():
    x32 = qb.array([[0.5, -1.0], [2.0, 0.25]], dtype=qb.float32)
    w32 = qb.array([0.3, -0.7], dtype=qb.float32)
    result = qb.vmap(lambda x, w: qb.sin(x * w) * 2.0, in_axes=(0, None))(x32, w32)
    assert result.dtype == qb.float32
    second = qb.vmap(qb.grad(qb.grad(lambda x, w: qb.sin(x * w[0]))), in_axes=(0, None))
    derivative = second(qb.array([0.1, 0.2], dtype=qb.float32), w32)
    assert derivative.dtype == qb.float32
    assert_close(
        derivative, [-(0.3**2) * math.sin(0.03), -(0.3**2) * math.sin(0.06)], 1e-6
    )
    c32 = qb.array([1.0, 2.0], dtype=qb.float32)
    assert qb.vmap(lambda x: x * c32)(x32).dtype == qb.float32
    # A strong float64 constant with a float32 tracer asks for astype.
    assert_raises(
        ValueError, qb.vmap(lambda x: x * qb.array([1.0, 2.0])), x32, match="astype"
    )
    mask = qb.vmap(lambda x: x > 0.0)(x32)
    assert mask.dtype == qb.bool_ and mask.tolist() == [[True, False], [True, True]]


def test_pinn_in_the_canonical_vmap_form_reproduces_the_readme():
    u_xx = qb.vmap(qb.grad(qb.grad(pinn_u)), in_axes=(0, None))  # design 2.4 and Q1

    def loss(w, x):
        return qb.mean((u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2)

    x, w = qb.linspace(0.05, 0.95, PINN_POINTS), qb.array(2.5)
    assert_close(u_xx(x, w), [-(2.5**2) * math.sin(2.5 * p) for p in x.tolist()], 1e-14)
    step = qb.jit(qb.value_and_grad(loss))
    value, grad_w = step(w, x)
    # The per-point S3 form and the README's symbolic_jvp path agree.
    loop_value, loop_grad = qb.jit(qb.value_and_grad(pinn_loss))(w, x)
    assert_close(value, loop_value, 1e-13)
    assert_close(grad_w, loop_grad, 1e-13)
    plan = readme_symbolic_jvp_loss().compile_cpu()
    readme_inputs = {"x": x.reshape([PINN_POINTS, 1]), "w": w.reshape([1, 1])}
    old_value, old_grads = plan.evaluate_value_and_vjp(
        readme_inputs, qb.Tensor([], [1.0])
    )
    assert_close(value, old_value.item(), 1e-12)
    assert_close(grad_w, old_grads["w"].item(), 1e-12)
    # Finite differences in w.
    h = 1e-6
    fd = (step(w + h, x)[0].item() - step(w - h, x)[0].item()) / (2.0 * h)
    assert_close(grad_w, fd, 1e-7)
    params, adam = {"w": w}, qb.Adam(learning_rate=0.05)
    for _ in range(300):
        value, grad_w = step(params["w"], x)
        params = adam.step(params, {"w": grad_w})
    assert f"{params['w'].item():.6f}" == "3.141593", params["w"].item()
    assert value.item() < 1e-10


def test_vmapped_pinn_plan_size_does_not_grow_with_the_points():
    from quabla import _transforms

    u_xx = qb.vmap(qb.grad(qb.grad(pinn_u)), in_axes=(0, None))

    def loss(w, x):
        return qb.mean((u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2)

    def sizes(points):
        x, w = qb.linspace(0.05, 0.95, points), qb.array(2.5)
        staged = qb.value_and_grad(loss)._stage(
            (tuple, (_transforms._LEAF, _transforms._LEAF)), [w, x], ["w", "x"]
        )
        executable = staged.graph._compile_cpu(staged.outputs, staged.input_names)
        return staged.graph._node_count, executable.node_count

    # Graph / plan nodes on this build: 92 / 42 for any number of points,
    # against 516 / 206 for the per-point form with 8 points (S3).
    assert sizes(8) == sizes(64) == sizes(1)


def test_pinn_gradients_in_vector_parameters_through_vmap_match_the_loop_form():
    def u(x, p):
        return p["a"] * qb.tanh(p["b"] * x) + p["c"] * x**3

    u_xx = qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None))
    point_u_xx = qb.grad(qb.grad(u))

    def loss(p, x):
        return qb.mean((u_xx(x, p) - qb.sin(x)) ** 2)

    def loop_loss(p, x):
        residual = qb.stack(
            [point_u_xx(x[i], p) for i in range(x.shape[0])], 0
        ) - qb.sin(x)
        return qb.mean(residual**2)

    params = {"a": qb.array(0.7), "b": qb.array(-1.3), "c": qb.array(0.2)}
    x = qb.linspace(-1.0, 1.0, 6)
    value, grads = qb.jit(qb.value_and_grad(loss))(params, x)
    loop_value, loop_grads = qb.value_and_grad(loop_loss)(params, x)
    assert_close(value, loop_value, 1e-13)
    for name in "abc":
        assert_close(grads[name], loop_grads[name], 1e-13)
        h = 1e-6
        plus = dict(params, **{name: params[name] + h})
        minus = dict(params, **{name: params[name] - h})
        fd = (qb.jit(loss)(plus, x).item() - qb.jit(loss)(minus, x).item()) / (2.0 * h)
        assert_close(grads[name], fd, 1e-6)


def central_jacobian(function, value, step=1e-6):
    """The `[*out, *in]` Jacobian of an array function of one array by
    central differences."""
    base = value.to_flat_list()
    out_shape = qb.asarray(function(value)).shape
    columns = []
    for element in range(len(base)):
        plus, minus = list(base), list(base)
        plus[element] += step
        minus[element] -= step
        upper = qb.asarray(function(qb.array(plus).reshape(value.shape))).to_flat_list()
        lower = qb.asarray(
            function(qb.array(minus).reshape(value.shape))
        ).to_flat_list()
        columns.append([(a - b) / (2.0 * step) for a, b in zip(upper, lower)])
    rows = [[column[row] for column in columns] for row in range(len(columns[0]))]
    return qb.array(rows).reshape(list(out_shape) + list(value.shape))


def test_jacobian_and_hessian_are_staged_through_vmap():
    def model(x, p):
        return {"y": qb.tanh(x @ p["w"]) * p["s"], "z": qb.sum(x**2) * p["s"]}

    x = qb.array([[0.5, -1.0], [2.0, 0.25]])
    p = {"w": qb.array([[0.3], [-0.7]]), "s": qb.array(1.5)}
    jac = qb.jacobian(model, argnums=(0, 1))(x, p)
    assert jac["y"][0].shape == [2, 1, 2, 2] and jac["y"][1]["w"].shape == [2, 1, 2, 1]
    assert jac["z"][1]["s"].shape == []

    # Every block against central differences in the elements of its leaf.
    blocks = [
        (lambda t: model(t, p)["y"], x, jac["y"][0]),
        (lambda t: model(t, p)["z"], x, jac["z"][0]),
        (lambda t: model(x, dict(p, w=t))["y"], p["w"], jac["y"][1]["w"]),
        (lambda t: model(x, dict(p, s=t))["y"], p["s"], jac["y"][1]["s"]),
        (lambda t: model(x, dict(p, w=t))["z"], p["w"], jac["z"][1]["w"]),
    ]
    for function, value, block in blocks:
        assert_close(block, central_jacobian(function, value), 1e-8)

    # Staged: jit compiles it, a trace can call it, and grad differentiates it.
    def cubic(v):
        return qb.sum(v**3) + v[0] * v[1]

    v = qb.array([0.5, -1.0, 2.0])
    expected = [[3.0, 1.0, 0.0], [1.0, -6.0, 0.0], [0.0, 0.0, 12.0]]
    assert_close(qb.hessian(cubic)(v), expected, 1e-14)
    assert_close(qb.jit(qb.hessian(cubic))(v), expected, 1e-14)
    doubled = [[2.0 * entry for entry in row] for row in expected]
    assert_close(qb.jit(lambda t: qb.hessian(cubic)(t) * 2.0)(v), doubled, 1e-14)
    assert_close(
        qb.grad(lambda t: qb.sum(qb.hessian(cubic)(t)))(v), [6.0, 6.0, 6.0], 1e-14
    )
    # Per-example Hessians under vmap, and a Jacobian of vmap.
    vs = qb.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    per_example_hessians = qb.vmap(qb.hessian(cubic))(vs)
    assert per_example_hessians.shape == [2, 3, 3]
    assert_close(
        qb.asarray(per_example_hessians[1]), qb.hessian(cubic)(qb.asarray(vs[1])), 1e-14
    )
    batch_jacobian = qb.jacobian(qb.vmap(qb.sin))(vs)
    assert batch_jacobian.shape == [2, 3, 2, 3]
    assert_close(batch_jacobian[1, 2, 1, 2], math.cos(-0.5), 1e-14)
    assert_close(batch_jacobian[0, 2, 1, 2], 0.0)
    # float32 stays float32; a scalar argument gives a scalar block.
    assert qb.hessian(cubic)(v.astype(qb.float32)).dtype == qb.float32
    assert_close(
        qb.jacobian(lambda t: qb.sin(t) * 2.0)(qb.array(0.3)),
        2.0 * math.cos(0.3),
        1e-15,
    )


# -- closures over tracers (slice S4b) --------------------------------------------------


def test_closures_over_tracers_of_an_enclosing_trace_are_lifted():
    # The issue repro: d/dw of (d/dx sin(x w))^2 at x = 0.3.
    def u(x, w):
        return qb.sin(x * w)

    def loss(w):
        u_x = qb.grad(lambda x: u(x, w))  # w is a tracer of the enclosing trace
        return u_x(0.3) ** 2

    def exact_loss(w):
        return (w * math.cos(0.3 * w)) ** 2

    def exact_grad(w):
        c, s = math.cos(0.3 * w), math.sin(0.3 * w)
        return 2.0 * w * c * (c - 0.3 * w * s)

    def central(function, w, h=1e-6):
        return (function(w + h) - function(w - h)) / (2.0 * h)

    value, gradient = qb.value_and_grad(loss)(qb.array(1.2))
    assert_close(value, exact_loss(1.2), 1e-15)
    assert_close(gradient, exact_grad(1.2), 1e-15)
    assert_close(gradient, central(exact_loss, 1.2), 1e-8)
    assert_close(qb.jit(qb.grad(loss))(qb.array(0.7)), exact_grad(0.7), 1e-15)

    # The inner transform differentiates only its explicit argument: jvp,
    # vjp, jit(grad), jit of a plain function, vmap (the capture is
    # unmapped), vmap(grad), and hessian closures, each differentiated in w.
    x0, w0 = 0.3, qb.array(1.2)
    d_x = math.cos(0.36) - 0.36 * math.sin(0.36)  # d/dw of w cos(x w)
    assert_close(
        qb.grad(lambda w: qb.jvp(lambda t: u(t, w), (x0,), (1.0,))[1])(w0), d_x, 1e-15
    )
    assert_close(
        qb.grad(lambda w: qb.vjp(lambda t: u(t, w), x0)[1](1.0)[0])(w0), d_x, 1e-15
    )
    assert_close(
        qb.grad(lambda w: qb.jit(qb.grad(lambda t: u(t, w)))(x0))(w0), d_x, 1e-15
    )
    assert_close(
        qb.grad(lambda w: qb.jit(lambda t: u(t, w))(x0))(w0),
        0.3 * math.cos(0.36),
        1e-15,
    )
    points = [0.1, 0.2, 0.3]
    xs = qb.array(points)
    assert_close(
        qb.grad(lambda w: qb.sum(qb.vmap(lambda t: u(t, w))(xs)))(w0),
        sum(p * math.cos(1.2 * p) for p in points),
        1e-14,
    )
    assert_close(
        qb.grad(lambda w: qb.sum(qb.vmap(qb.grad(lambda t: u(t, w)))(xs)))(w0),
        sum(math.cos(1.2 * p) - 1.2 * p * math.sin(1.2 * p) for p in points),
        1e-14,
    )
    vs = [0.4, -0.2]
    assert_close(
        qb.grad(
            lambda w: qb.sum(qb.hessian(lambda v: qb.sum(qb.sin(v * w)))(qb.array(vs)))
        )(w0),
        sum(-2.4 * math.sin(1.2 * v) - 1.44 * v * math.cos(1.2 * v) for v in vs),
        1e-14,
    )

    # has_aux returning the captured tracer itself, and a captured tracer
    # next to a captured eager constant (S3b).
    def aux_loss(w):
        derivative, aux = qb.grad(lambda t: (u(t, w), w * 2.0), has_aux=True)(x0)
        return derivative + aux

    assert_close(qb.grad(aux_loss)(w0), d_x + 2.0, 1e-15)
    c = qb.array(0.7)
    assert_close(
        qb.grad(lambda w: qb.grad(lambda t: qb.sin(t * w * c))(x0))(w0),
        0.7 * math.cos(0.252) - 0.3 * 1.2 * 0.49 * math.sin(0.252),
        1e-15,
    )


def test_closures_capture_several_tracers_from_several_levels():
    # The innermost function captures y from the enclosing trace and w from
    # two levels up; both are constants for d/dx and differentiated outside.
    def outer(w):
        def middle(y):
            return qb.grad(lambda x: qb.sin(x * y * w))(0.3) * y

        return qb.grad(middle)(0.5)

    def exact(w, y=0.5):
        a = 0.3 * y * w
        return 2.0 * y * w * math.cos(a) - 0.3 * y**2 * w**2 * math.sin(a)

    h = 1e-6
    assert_close(outer(qb.array(1.2)), exact(1.2), 1e-15)
    value, gradient = qb.value_and_grad(outer)(qb.array(1.2))
    assert_close(value, exact(1.2), 1e-15)
    assert_close(gradient, (exact(1.2 + h) - exact(1.2 - h)) / (2.0 * h), 1e-8)


def test_captured_tracers_rebind_on_every_call_of_a_cached_closure():
    from quabla import _transforms

    # A function whose closure refers to a different tracer on each call:
    # the staged graph is reused, but its capture input binds to the tracer
    # of this call, never to the one it was staged with.
    box = {}

    def inner(x):
        return qb.sum(qb.sin(x * box["w"]))

    g = qb.grad(inner)

    def loss(w):
        box["w"] = w
        first = g(0.3)
        box["w"] = w * 2.0
        return first + 3.0 * g(0.3)

    def exact(w):
        return w * math.cos(0.3 * w) + 6.0 * w * math.cos(0.6 * w)

    h = 1e-6
    for w in [1.2, 0.7]:  # two outer traces through two outer transforms
        value, gradient = qb.value_and_grad(loss)(qb.array(w))
        assert_close(value, exact(w), 1e-14)
        assert_close(gradient, (exact(w + h) - exact(w - h)) / (2.0 * h), 1e-8)
        assert_close(qb.jit(loss)(qb.array(w)), exact(w), 1e-14)
    # One staged graph serves all six calls in the four outer traces.
    inline_cache = _transforms._CACHES[inner][
        (("grad", 0, False), _transforms._INLINED)
    ]
    assert len(inline_cache.entries) == 1
    # A capture of another shape is another entry; its signature says so.
    vector = qb.jit(lambda w: (box.__setitem__("w", w), g(0.3))[1])(
        qb.array([0.5, 1.0])
    )
    assert_close(vector, 0.5 * math.cos(0.15) + math.cos(0.3), 1e-15)
    assert len(inline_cache.entries) == 2
    rendered = [_transforms._describe_signature(key) for key in inline_cache.entries]
    assert rendered == [
        "(float64[],) capturing [float64[]]",
        "(float64[],) capturing [float64[2]]",
    ]

    # A call that captures nothing this time (the closure now holds an eager
    # array, a constant) is inlined with constants only, and stays exact.
    def mixed_loss(w):
        box["w"] = w
        first = g(0.3)
        box["w"] = qb.array(2.0)
        return first + g(0.3)

    value, gradient = qb.value_and_grad(mixed_loss)(qb.array(1.2))
    assert_close(value, 1.2 * math.cos(0.36) + 2.0 * math.cos(0.6), 1e-14)
    assert_close(gradient, math.cos(0.36) - 0.36 * math.sin(0.36), 1e-14)
    assert len(inline_cache.entries) == 3

    # A closure applied in a loop inside one trace is staged once.
    traces = []

    def u(x, w):
        traces.append(1)
        return qb.sin(x * w)

    def loop_loss(w, x):
        u_x = qb.grad(lambda t: u(t, w))
        return u_x(x[0]) + u_x(x[1]) + u_x(x[2])

    xs = qb.array([0.1, 0.2, 0.3])
    assert_close(
        qb.jit(loop_loss)(qb.array(1.5), xs),
        sum(1.5 * math.cos(1.5 * p) for p in [0.1, 0.2, 0.3]),
        1e-14,
    )
    # The first call stages the closure; each later call only traces it
    # again to find the tracers it captures, and reuses the staged graph.
    assert len(traces) == 3


def jax_style_net(params, x):  # scalar x
    hidden = qb.tanh(params["w1"] * x + params["b1"])
    return qb.sum(params["w2"] * hidden) + params["b2"]


def test_jax_style_pinn_closing_over_params_matches_explicit_arguments():
    xs = qb.linspace(0.05, 0.95, PINN_POINTS)
    forcing = math.pi**2 * qb.sin(math.pi * xs)  # an eager constant, captured

    def closure_loss(params):
        # The common JAX form: params closed over, per-point vmap(grad(grad)).
        u_xx = qb.vmap(qb.grad(qb.grad(lambda x: jax_style_net(params, x))))
        boundary = jax_style_net(params, 0.0) ** 2 + jax_style_net(params, 1.0) ** 2
        return qb.mean((u_xx(xs) + forcing) ** 2) + boundary

    explicit_u_xx = qb.vmap(
        qb.grad(qb.grad(lambda x, p: jax_style_net(p, x))), in_axes=(0, None)
    )

    def explicit_loss(params):  # the S4 form: params as an unmapped argument
        boundary = jax_style_net(params, 0.0) ** 2 + jax_style_net(params, 1.0) ** 2
        return qb.mean((explicit_u_xx(xs, params) + forcing) ** 2) + boundary

    width = 6
    params = {
        "w1": qb.array([0.9 * math.sin(i + 1.0) for i in range(width)]),
        "b1": qb.array([0.3 * math.cos(i + 2.0) for i in range(width)]),
        "w2": qb.array([0.7 * math.sin(2.0 * i + 0.5) for i in range(width)]),
        "b2": qb.array(0.1),
    }
    closure_step = qb.jit(qb.value_and_grad(closure_loss))
    explicit_step = qb.jit(qb.value_and_grad(explicit_loss))
    # Finite differences of one parameter element at the initial point.
    h = 1e-6
    w1 = params["w1"].tolist()
    plus = dict(params, w1=qb.array([w1[0] + h] + w1[1:]))
    minus = dict(params, w1=qb.array([w1[0] - h] + w1[1:]))
    fd = (qb.jit(closure_loss)(plus).item() - qb.jit(closure_loss)(minus).item()) / (
        2.0 * h
    )
    assert_close(closure_step(params)[1]["w1"][0], fd, 1e-6)

    closure_params, explicit_params = params, params
    closure_adam, explicit_adam = (
        qb.Adam(learning_rate=0.01),
        qb.Adam(learning_rate=0.01),
    )
    losses = []
    for _ in range(5):
        value, grads = closure_step(closure_params)
        explicit_value, explicit_grads = explicit_step(explicit_params)
        assert_close(value, explicit_value, 1e-12)
        for name in params:
            assert_close(grads[name], explicit_grads[name], 1e-12)
        losses.append(value.item())
        closure_params = closure_adam.step(closure_params, grads)
        explicit_params = explicit_adam.step(explicit_params, explicit_grads)
    assert losses[-1] < losses[0], losses


def test_escaped_tracers_raise_tracer_error_instead_of_mixing_graphs():
    leaked = []
    x = qb.array([1.0, 2.0])
    qb.jit(lambda t: leaked.append(t) or t)(x)
    # A tracer used after its trace ended, in either operand position, in a
    # module function, or returned from a later trace.
    for function in [
        lambda t: t * leaked[0],
        lambda t: leaked[0] + t,
        lambda t: qb.where(t > 0.0, leaked[0], t),
        lambda t: leaked[0] * 2.0,
    ]:
        error = assert_raises(qb.TracerError, qb.jit(function), x, match="escaped")
        assert isinstance(error, TypeError)
    assert_raises(qb.TracerError, qb.grad(lambda t: qb.sum(qb.sin(t) + leaked[0])), x)
    # The trace of a v0.1 helper is not an enclosing transform: its tracers
    # cannot be captured by a v0.2 transform inside it.
    assert_raises(
        qb.TracerError,
        qb.tensor_jit_fn,
        lambda a: qb.grad(lambda y: y * a)(1.0),
        [("a", [])],
        match="tensor_* helper",
    )


def lowered_node_count(function, *args):
    text = qb.jit(function).lower(*args).as_text()
    return len([line for line in text.splitlines() if line.lstrip().startswith("%")])


def test_jit_gather_and_scatter_add_are_constant_size_and_match_eager():
    # Each traces to one IR node, so neither the staged program nor its
    # gradient grows with the index count; they used to emit one slice or one
    # padded add per index.
    def sized(count):
        rows = 8
        indices = [(5 * k + 3) % rows for k in range(count)]
        x = qb.array([[0.25 * i - 0.5 * j for j in range(3)] for i in range(rows)])
        u = qb.array([[0.125 * k + j for j in range(3)] for k in range(count)])
        scatter = lambda b, v: b.scatter_add(indices, v, axis=0)  # noqa: E731
        loss = lambda y: qb.sum(qb.sin(y.gather(indices, axis=0)))  # noqa: E731
        return (
            lowered_node_count(scatter, x, u),
            lowered_node_count(qb.grad(loss), x),
        )

    assert sized(64) == sized(1024), (sized(64), sized(1024))

    # Values and derivatives match eager for repeated indices on both axes.
    # Dyadic data keeps every sum exact, so equality holds in any order.
    x = qb.array([[0.25 * i - 0.5 * j for j in range(5)] for i in range(4)])
    devices = ["cpu"] + [device for device in ("mlx", "cuda") if enabled(device)]
    for axis, indices in ((0, [3, 1, 3, 0, 3]), (1, [4, 0, 4, 4, 2, 1])):
        shape = [4, 5]
        shape[axis] = len(indices)
        u = qb.array([[0.5 * i + 0.25 * j for j in range(shape[1])] for i in range(shape[0])])
        w = qb.array([[1.0 - 0.75 * i + j for j in range(shape[1])] for i in range(shape[0])])
        c = qb.array([[0.5 * i - j for j in range(5)] for i in range(4)])
        zeros = qb.zeros([4, 5])
        for device in devices:
            jit = lambda f: qb.jit(f, device=device)  # noqa: E731
            assert_close(jit(lambda y: y.gather(indices, axis=axis))(x), x.gather(indices, axis=axis), 0)
            assert_close(
                jit(lambda b, v: b.scatter_add(indices, v, axis=axis))(x, u),
                x.scatter_add(indices, u, axis=axis),
                0,
            )
            gather_grad = jit(qb.grad(lambda y: qb.sum(y.gather(indices, axis=axis) * w)))(x)
            assert_close(gather_grad, zeros.scatter_add(indices, w, axis=axis), 0)
            base_grad, update_grad = jit(
                qb.grad(lambda b, v: qb.sum(b.scatter_add(indices, v, axis=axis) * c), argnums=(0, 1))
            )(x, u)
            assert_close(base_grad, c, 0)
            assert_close(update_grad, c.gather(indices, axis=axis), 0)
            stacked = qb.stack([x, x * 2.0, x - 1.0])
            stacked_u = qb.stack([u, u - 0.5, u * 4.0])
            assert_close(
                jit(qb.vmap(lambda y: y.gather(indices, axis=axis)))(stacked),
                stacked.gather(indices, axis=axis + 1),
                0,
            )
            assert_close(
                jit(qb.vmap(lambda b, v: b.scatter_add(indices, v, axis=axis)))(
                    stacked, stacked_u
                ),
                stacked.scatter_add(indices, stacked_u, axis=axis + 1),
                0,
            )


SPECIAL_POINTS = [
    -math.inf,
    -750.0,
    -20.0,
    -1.0,
    -1e-10,
    -0.0,
    0.0,
    5e-324,
    1e-300,
    1e-8,
    0.5,
    3.0,
    6.0,
    30.0,
    709.0,
    710.0,
    math.inf,
    math.nan,
]


def assert_relative(actual, expected, tolerance):
    """Each float of `actual` is within `tolerance` of `expected` relative to
    the expected magnitude; NaN must match NaN and infinities must be equal."""
    assert len(actual) == len(expected), (actual, expected)
    for lhs, rhs in zip(actual, expected):
        if math.isnan(rhs):
            assert math.isnan(lhs), (lhs, rhs)
        elif math.isinf(rhs) or rhs == 0.0:
            assert lhs == rhs, (lhs, rhs)
        else:
            assert abs(lhs - rhs) <= tolerance * abs(rhs), (lhs, rhs, actual, expected)


def float32_round(value):
    # Values from 2**128 - 2**103 up round to infinity, which struct rejects.
    if abs(value) >= 2.0**128 - 2.0**103:
        return math.copysign(math.inf, value)
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_expm1_erf_and_atan2_values_match_math_eagerly_and_traced():
    pairs = [(y, x) for y in SPECIAL_POINTS for x in SPECIAL_POINTS]
    for dtype in [qb.float64, qb.float32]:
        x = qb.array(SPECIAL_POINTS, dtype=dtype)
        ys = qb.array([y for y, _ in pairs], dtype=dtype)
        xs = qb.array([x for _, x in pairs], dtype=dtype)
        inputs = x.tolist()
        y_inputs, x_inputs = ys.tolist(), xs.tolist()
        cases = [
            (qb.expm1, (x,), [math.inf if p >= 710.0 else math.expm1(p) for p in inputs]),
            (qb.erf, (x,), [math.erf(p) for p in inputs]),
            (qb.erfc, (x,), [math.erfc(p) for p in inputs]),
            (qb.atan2, (ys, xs), [math.atan2(*p) for p in zip(y_inputs, x_inputs)]),
        ]
        for function, arguments, reference in cases:
            eager = function(*arguments)
            jitted = qb.jit(function)(*arguments)
            assert eager.dtype == jitted.dtype == dtype
            assert float_bits(eager.tolist()) == float_bits(jitted.tolist())
            if dtype == qb.float32:
                reference = [float32_round(value) for value in reference]
                # One float32 rounding of the f64 result, which may differ from
                # the platform libm's f64 by an ulp only below float32 resolution.
                assert_relative(eager.tolist(), reference, 1.2e-7)
            elif function is qb.erfc:
                # Both are within about an ulp of the true erfc, which loses
                # relative resolution only where the result is subnormal.
                normal = [(v, r) for v, r in zip(eager.tolist(), reference) if abs(r) > 1e-300]
                assert_relative([v for v, _ in normal], [r for _, r in normal], 1e-15)
            else:
                # math.erf is the platform libm; quabla uses the musl port.
                assert_relative(eager.tolist(), reference, 4.5e-16)
    # expm1 keeps the small-argument precision that exp(x) - 1 loses.
    tiny = qb.array([1e-10], dtype=qb.float32)
    assert qb.jit(qb.expm1)(tiny).item() == tiny.item()
    # atan2 broadcasts and keeps a Python number weak in either position.
    y = qb.array([[1.0], [-2.0]], dtype=qb.float32)
    x = qb.array([0.5, -3.0, 0.0], dtype=qb.float32)
    for result in [qb.atan2(y, x), qb.jit(qb.atan2)(y, x)]:
        assert result.shape == [2, 3] and result.dtype == qb.float32
    assert qb.atan2(x, 1.0).dtype == qb.float32
    assert qb.atan2(1.0, x).dtype == qb.float32
    assert_close(qb.atan2(1.0, x), [math.atan2(1.0, p) for p in x.tolist()], 1e-7)
    assert_close(qb.jit(lambda t: qb.atan2(t, 2.0))(x), qb.atan2(x, 2.0), 0)
    assert_raises(ValueError, qb.erf, qb.array([True]), match="bool")
    # erfc keeps full relative accuracy where 1 - erf(x) cancels.
    tail = qb.array([5.0, 10.0, 26.0])
    assert_relative(qb.erfc(tail).tolist(), [math.erfc(p) for p in tail.tolist()], 2e-16)
    assert qb.erfc(tail).tolist()[-1] > 0.0
    assert_raises(ValueError, qb.erfc, qb.array([True]), match="bool")
    for name in ["expm1", "erf", "erfc", "atan2", "cumsum", "stop_gradient", "prod"]:
        assert name in qb.__all__ and getattr(qb, name).__name__ == name


def test_expm1_erf_and_atan2_gradients_and_hessians_match_closed_forms():
    coefficient = 2.0 / math.sqrt(math.pi)
    points = [-3.0, -1e-6, 0.0, 0.5, 2.0]
    x = qb.array(points)
    for function, derivative in [
        (qb.expm1, math.exp),
        (qb.erf, lambda p: coefficient * math.exp(-p * p)),
        (qb.erfc, lambda p: -coefficient * math.exp(-p * p)),
    ]:
        reference = [derivative(p) for p in points]
        for transform in [qb.grad, lambda f: qb.jit(qb.grad(f))]:
            gradient = transform(lambda t, f=function: qb.sum(f(t)))(x)
            assert_relative(gradient.tolist(), reference, 1e-15)
        _, tangent = qb.jvp(function, (x,), (qb.ones([5]),))
        assert_relative(tangent.tolist(), reference, 1e-15)
    hessian = qb.hessian(lambda t: qb.sum(qb.erf(t)))(x).tolist()
    complementary = qb.hessian(lambda t: qb.sum(qb.erfc(t)))(x).tolist()
    for row, point in enumerate(points):
        expected = -2.0 * point * coefficient * math.exp(-point * point)
        assert_relative([hessian[row][row]], [expected], 1e-15)
        assert_relative([complementary[row][row]], [-expected], 1e-15)

    # atan2: (x, -y) / (x^2 + y^2) without overflow or underflow, zero at the
    # origin, and a Hessian that matches the closed form and finite differences.
    ys = [1.0, -2.0, 1e200, 1e-200, 0.0]
    xs = [1.0, 0.5, 1e200, -1e-200, 0.0]
    expected_y = [0.5, 0.5 / 4.25, 5e-201, -5e199, 0.0]
    expected_x = [-0.5, 2.0 / 4.25, -5e-201, -5e199, 0.0]
    gradient = qb.jit(qb.grad(lambda a, b: qb.sum(qb.atan2(a, b)), argnums=(0, 1)))
    for actual, expected in zip(gradient(qb.array(ys), qb.array(xs)), [expected_y, expected_x]):
        assert_relative(actual.tolist(), expected, 4e-15)

    def angle(p):
        return qb.atan2(p[0], p[1])

    for y, x in [(0.8, -1.5), (-0.3, 0.2), (1e150, 2e150)]:
        r4 = (x * x + y * y) ** 2 if abs(x) < 1e100 else None
        hessian = qb.hessian(angle)(qb.array([y, x])).tolist()
        if r4 is None:
            expected = [[-1.6e-301, -1.2e-301], [-1.2e-301, 1.6e-301]]
        else:
            mixed = (y * y - x * x) / r4
            expected = [[-2 * x * y / r4, mixed], [mixed, 2 * x * y / r4]]
        for row in range(2):
            assert_relative(hessian[row], expected[row], 1e-13)
        if r4 is not None:
            # Central differences of the gradient, O(h^2) accurate.
            step = 1e-5
            for column in range(2):
                shift = [0.0, 0.0]
                shift[column] = step
                plus = qb.grad(angle)(qb.array([y + shift[0], x + shift[1]])).tolist()
                minus = qb.grad(angle)(qb.array([y - shift[0], x - shift[1]])).tolist()
                for row in range(2):
                    difference = (plus[row] - minus[row]) / (2 * step)
                    assert abs(difference - hessian[row][column]) <= 1e-8, (
                        difference,
                        hessian,
                    )
    assert qb.hessian(angle)(qb.array([0.0, 0.0])).tolist() == [[0.0, 0.0], [0.0, 0.0]]


def _math_reference(function, pole=None):
    """`function` with NumPy's IEEE results where `math` raises: NaN outside
    the domain, a signed infinity on overflow, and `pole(p)` at a pole."""

    def reference(p):
        if pole is not None and pole(p) is not None:
            return pole(p)
        try:
            return function(p)
        except OverflowError:
            return math.copysign(math.inf, p) if function is math.sinh else math.inf
        except ValueError:
            return math.nan

    return reference


def _integral(function):
    """`math.floor`/`math.ceil`/`round` as a float; infinities and NaN pass."""
    return lambda p: p if not math.isfinite(p) else float(function(p))


# Python's `round` of a float rounds halfway cases to even, like NumPy's.
_UNARY_MATH_REFERENCES = {
    "tan": _math_reference(math.tan),
    "arcsin": _math_reference(math.asin),
    "arccos": _math_reference(math.acos),
    "arctan": _math_reference(math.atan),
    "sinh": _math_reference(math.sinh),
    "cosh": _math_reference(math.cosh),
    "arcsinh": _math_reference(math.asinh),
    "arccosh": _math_reference(math.acosh),
    "arctanh": _math_reference(
        math.atanh, lambda p: math.copysign(math.inf, p) if abs(p) == 1.0 else None
    ),
    "log2": _math_reference(math.log2, lambda p: -math.inf if p == 0.0 else None),
    "log10": _math_reference(math.log10, lambda p: -math.inf if p == 0.0 else None),
    "floor": _integral(math.floor),
    "ceil": _integral(math.ceil),
    "round": _integral(round),
}
# math.cbrt is new in Python 3.11; the exact cubes of the edge test cover
# older versions.
if hasattr(math, "cbrt"):
    _UNARY_MATH_REFERENCES["cbrt"] = math.cbrt
UNARY_MATH_NAMES = [
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "log2",
    "log10",
    "cbrt",
    "floor",
    "ceil",
    "round",
]
UNARY_MATH_POINTS = SPECIAL_POINTS + [
    -1e300,
    -27.0,
    -2.5,
    -1.5,
    -1.0 - 2.0**-52,
    -1.0,
    -(1.0 - 2.0**-53),
    -0.5,
    0.1,
    0.25,
    1.0 - 2.0**-53,
    1.0,
    1.0 + 2.0**-52,
    1.5,
    2.5,
    27.0,
    1e22,
    1e300,
]


def test_unary_math_values_match_math_eagerly_and_traced():
    for dtype in [qb.float64, qb.float32]:
        x = qb.array(UNARY_MATH_POINTS, dtype=dtype)
        inputs = x.tolist()
        for name in UNARY_MATH_NAMES:
            function = getattr(qb, name)
            eager = function(x)
            jitted = qb.jit(function)(x)
            method = getattr(x, name)()
            assert eager.dtype == jitted.dtype == method.dtype == dtype, name
            assert float_bits(eager.tolist()) == float_bits(jitted.tolist()), name
            assert float_bits(eager.tolist()) == float_bits(method.tolist()), name
            if name not in _UNARY_MATH_REFERENCES:
                continue
            reference = [_UNARY_MATH_REFERENCES[name](p) for p in inputs]
            if dtype == qb.float32:
                # One float32 rounding of the f64 result.
                reference = [float32_round(value) for value in reference]
                assert_relative(eager.tolist(), reference, 1.2e-7)
            elif name in ("floor", "ceil", "round"):
                assert_relative(eager.tolist(), reference, 0)
            else:
                # The CPU uses Rust std (the platform libm) and, for the
                # inverse hyperbolic functions, the musl port of `libm`.
                assert_relative(eager.tolist(), reference, 4.5e-16)
    assert_raises(ValueError, qb.tan, qb.array([True]), match="bool")
    for name in UNARY_MATH_NAMES:
        assert getattr(qb, name).__name__ == name
        # `round` shadows a builtin, so the star import leaves it out.
        assert (name in qb.__all__) == (name != "round"), name
    for name in ["fmod", "mod", "remainder"]:
        assert name in qb.__all__


def test_unary_math_edge_values():
    def values(name, points, dtype=qb.float64):
        return getattr(qb, name)(qb.array(points, dtype=dtype)).tolist()

    nan, inf, half_pi = math.nan, math.inf, math.pi / 2
    # Out-of-domain inputs give NaN and poles give infinities, never errors.
    assert_relative(values("arcsin", [-1.0, 1.0, 1.5, -inf]), [-half_pi, half_pi, nan, nan], 0)
    assert_relative(values("arccos", [-1.0, 1.0, -1.5]), [math.pi, 0.0, nan], 0)
    assert_relative(values("arctanh", [-1.0, 1.0, 2.0, 0.0]), [-inf, inf, nan, 0.0], 0)
    assert_relative(values("arccosh", [1.0, 0.5, -inf, inf]), [0.0, nan, nan, inf], 0)
    assert_relative(values("log2", [0.0, -0.0, -1.0, inf]), [-inf, -inf, nan, inf], 0)
    assert_relative(values("log10", [0.0, -2.0]), [-inf, nan], 0)
    assert_relative(values("arctan", [inf, -inf, 1e300]), [half_pi, -half_pi, half_pi], 0)
    assert_relative(values("sinh", [711.0, -711.0, -inf]), [inf, -inf, -inf], 0)
    assert_relative(values("cosh", [-711.0, inf]), [inf, inf], 0)
    assert_relative(values("tan", [inf, 0.0]), [nan, 0.0], 0)
    for dtype in [qb.float64, qb.float32]:
        # Exact powers of ten and two give exact integers.
        assert values("log10", [10.0**k for k in range(11)], dtype) == list(range(11))
        powers = range(-126, 128, 7)
        assert values("log2", [2.0**k for k in powers], dtype) == list(powers)
        # Real cube roots with the sign of x, exact for exact cubes.
        cubes = [-1e9, -27.0, -8.0, -0.125, 0.0, 0.125, 8.0, 27.0, 1e9]
        roots = [-1e3, -3.0, -2.0, -0.5, 0.0, 0.5, 2.0, 3.0, 1e3]
        assert values("cbrt", cubes, dtype) == roots
        assert values("cbrt", [-inf, inf], dtype) == [-inf, inf]
        # Halfway cases round to even, as in NumPy and JAX.
        halves = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        assert values("round", halves, dtype) == [-4.0, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0]
        assert values("floor", [-1.5, -0.5, 0.5, 1.5], dtype) == [-2.0, -1.0, 0.0, 1.0]
        assert values("ceil", [-1.5, -0.5, 0.5, 1.5], dtype) == [-1.0, -0.0, 1.0, 2.0]
    assert math.copysign(1.0, values("round", [-0.5])[0]) == -1.0
    assert math.copysign(1.0, values("ceil", [-0.5])[0]) == -1.0
    big = [2.0**52 + 1.0, 0.49999999999999994, -(2.0**53)]
    assert values("round", big) == [2.0**52 + 1.0, 0.0, -(2.0**53)]
    # The inverse hyperbolic functions keep their relative accuracy near
    # their zeros, at the edge of the domain, and at large magnitudes.
    edge = 1.0 - 2.0**-53
    assert_relative(values("arccosh", [1.0 + 2.0**-52]), [math.acosh(1.0 + 2.0**-52)], 4.5e-16)
    assert_relative(values("arcsinh", [1e-300, -1e300]), [1e-300, -math.asinh(1e300)], 4.5e-16)
    assert_relative(values("arctanh", [1e-300, edge]), [1e-300, math.atanh(edge)], 4.5e-16)
    if np is not None:
        points = np.array(UNARY_MATH_POINTS)
        with np.errstate(all="ignore"):
            for name in UNARY_MATH_NAMES:
                expected = getattr(np, name)(points)
                actual = np.array(getattr(qb, name)(qb.array(points)).tolist())
                same = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
                close = np.abs(actual - expected) <= 4.5e-16 * np.abs(expected)
                assert np.all(same | close), (name, points[~(same | close)])


_LN2, _LN10 = math.log(2.0), math.log(10.0)


def _real_cbrt(p):
    return math.copysign(abs(p) ** (1.0 / 3.0), p)


# (first derivative, second derivative, interior points) per function. The
# arcsin/arccos references use the factored (1 - p)(1 + p): 1 - p * p loses
# relative accuracy near |p| = 1 (5e-14 at p = 0.999).
_UNARY_MATH_DERIVATIVES = {
    "tan": (
        lambda p: 1.0 + math.tan(p) ** 2,
        lambda p: 2.0 * math.tan(p) * (1.0 + math.tan(p) ** 2),
        [-1.3, -0.4, 0.0, 0.7, 1.5],
    ),
    "arcsin": (
        lambda p: 1.0 / math.sqrt((1.0 - p) * (1.0 + p)),
        lambda p: p / ((1.0 - p) * (1.0 + p)) ** 1.5,
        [-0.99, -0.3, 0.0, 0.6, 0.999],
    ),
    "arccos": (
        lambda p: -1.0 / math.sqrt((1.0 - p) * (1.0 + p)),
        lambda p: -p / ((1.0 - p) * (1.0 + p)) ** 1.5,
        [-0.99, -0.3, 0.0, 0.6, 0.999],
    ),
    "arctan": (
        lambda p: 1.0 / (1.0 + p * p),
        lambda p: -2.0 * p / (1.0 + p * p) ** 2,
        [-30.0, -0.5, 0.0, 1.0, 4.0],
    ),
    "sinh": (math.cosh, math.sinh, [-5.0, -0.5, 0.0, 1.0, 20.0]),
    "cosh": (math.sinh, math.cosh, [-5.0, -0.5, 0.0, 1.0, 20.0]),
    "arcsinh": (
        lambda p: 1.0 / math.sqrt(p * p + 1.0),
        lambda p: -p / (p * p + 1.0) ** 1.5,
        [-40.0, -1.0, 0.0, 0.5, 1.5],
    ),
    "arccosh": (
        lambda p: 1.0 / math.sqrt((p - 1.0) * (p + 1.0)),
        lambda p: -p / ((p - 1.0) * (p + 1.0)) ** 1.5,
        [1.001, 1.5, 2.0, 10.0, 1e5],
    ),
    "arctanh": (
        lambda p: 1.0 / ((1.0 - p) * (1.0 + p)),
        lambda p: 2.0 * p / ((1.0 - p) * (1.0 + p)) ** 2,
        [-0.99, -0.3, 0.0, 0.6, 0.999],
    ),
    "log2": (
        lambda p: 1.0 / (p * _LN2),
        lambda p: -1.0 / (p * p * _LN2),
        [1e-3, 0.5, 1.0, 3.0, 1e4],
    ),
    "log10": (
        lambda p: 1.0 / (p * _LN10),
        lambda p: -1.0 / (p * p * _LN10),
        [1e-3, 0.5, 1.0, 3.0, 1e4],
    ),
    "cbrt": (
        lambda p: 1.0 / (3.0 * _real_cbrt(p) ** 2),
        lambda p: -2.0 / (9.0 * _real_cbrt(p) ** 5),
        [-27.0, -0.2, 1e-6, 1.0, 8.0],
    ),
}


def test_unary_math_derivatives_match_closed_forms_and_finite_differences():
    for name, (first, second, points) in _UNARY_MATH_DERIVATIVES.items():
        function = getattr(qb, name)
        x = qb.array(points)
        expected_first = [first(p) for p in points]
        expected_second = [second(p) for p in points]

        def total(t, f=function):
            return qb.sum(f(t))

        routes = [
            qb.grad(total)(x),
            qb.jit(qb.grad(total))(x),
            qb.vmap(qb.grad(function))(x),
            qb.jvp(function, (x,), (qb.ones([len(points)]),))[1],
        ]
        for gradient in routes:
            # The IR and the closed form round differently, within a few ulp.
            assert_relative(gradient.tolist(), expected_first, 2e-15)
        hessian = qb.jit(qb.hessian(total))(x).tolist()
        size = len(points)
        assert all(hessian[i][j] == 0.0 for i in range(size) for j in range(size) if i != j)
        diagonal = [hessian[i][i] for i in range(size)]
        for actual in [diagonal, qb.vmap(qb.grad(qb.grad(function)))(x).tolist()]:
            assert_relative(actual, expected_second, 1e-13)
        # Central differences of the first derivative, O(h^2) accurate.
        gradient = qb.grad(function)
        for point, expected in zip(points, expected_second):
            step = 1e-6 * max(1.0, abs(point))
            if name in ("arcsin", "arccos", "arctanh") and abs(point) > 0.9:
                step = 1e-8
            if (name == "arccosh" and point < 1.01) or (name == "cbrt" and abs(point) < 1e-3):
                step = 1e-10
            difference = (
                gradient(qb.array(point + step)).item() - gradient(qb.array(point - step)).item()
            ) / (2.0 * step)
            assert abs(difference - expected) <= 1e-5 * max(1.0, abs(expected)), (
                name,
                point,
                difference,
                expected,
            )
        # float32: the derivative graph runs in float32, within a few ulp.
        x32 = qb.array(points, dtype=qb.float32)
        gradient32 = qb.jit(qb.grad(total))(x32)
        assert gradient32.dtype == qb.float32
        assert_relative(gradient32.tolist(), [first(p) for p in x32.tolist()], 2e-6)


def test_unary_math_derivative_edges():
    def gradient(name, points):
        function = getattr(qb, name)
        return qb.jit(qb.grad(lambda t: qb.sum(function(t))))(qb.array(points)).tolist()

    def second(name, points):
        function = getattr(qb, name)
        return qb.jit(qb.vmap(qb.grad(qb.grad(function))))(qb.array(points)).tolist()

    inf = math.inf
    # arcsinh' = 1 / sqrt(x^2 + 1) is formed as r / sqrt(1 + u^2) with
    # r = 1 / max(|x|, 1), so it neither overflows nor turns NaN at +-inf.
    assert_relative(
        gradient("arcsinh", [1e200, -1e300, inf, -inf]), [1e-200, 1e-300, 0.0, 0.0], 1e-15
    )
    assert second("arcsinh", [1e200, inf, -inf]) == [0.0, 0.0, 0.0]
    assert_relative(second("arcsinh", [-3.0]), [3.0 / 10.0**1.5], 1e-15)
    # arccosh' = 1 / (sqrt(x - 1) sqrt(x + 1)): no overflow, accurate near 1.
    above_one = 1.0 + 2.0**-52
    assert_relative(
        gradient("arccosh", [1e200, above_one]),
        [1e-200, 1.0 / math.sqrt(2.0**-52 * (2.0 + 2.0**-52))],
        1e-15,
    )
    # The factored (1 - x)(1 + x) keeps the poles accurate at 1 - 2^-53.
    edge = 1.0 - 2.0**-53
    assert gradient("arcsin", [edge, -edge]) == [2.0**26, 2.0**26]
    assert_relative(gradient("arctanh", [edge]), [1.0 / (2.0**-53 * (2.0 - 2.0**-53))], 1e-15)
    # Poles and out-of-domain points follow IEEE arithmetic, never errors.
    assert gradient("arcsin", [1.0, -1.0]) == [inf, inf]
    assert all(math.isnan(value) for value in gradient("arcsin", [1.5, -2.0]))
    assert all(math.isnan(value) for value in gradient("arccosh", [0.5]))
    assert gradient("arctanh", [1.0]) == [inf]
    assert gradient("log2", [0.0]) == [inf]
    assert gradient("cbrt", [0.0, inf]) == [inf, 0.0]
    assert gradient("arctan", [1e200, inf]) == [0.0, 0.0]
    # The piecewise-constant functions have a zero derivative of every
    # order, also at their jumps, and composite gradients route around them.
    for name in ("floor", "ceil", "round"):
        function = getattr(qb, name)
        points = [-1.5, -0.5, 0.0, 0.5, 1.0, 2.5, inf]
        assert gradient(name, points) == [0.0] * len(points)
        assert second(name, points) == [0.0] * len(points)
        _, tangent = qb.jvp(function, (qb.array(points),), (qb.ones([len(points)]),))
        assert tangent.tolist() == [0.0] * len(points)
        product = qb.grad(lambda t, f=function: qb.sum(f(t) * t))(qb.array([1.25]))
        assert product.tolist() == [function(qb.array(1.25)).item()]
    # tan' reuses the primal: 1 + tan(x)^2 near pi / 2.
    near_pole = 1.5707963
    assert_relative(gradient("tan", [near_pole]), [1.0 + math.tan(near_pole) ** 2], 1e-15)
    # Batched shapes under vmap and jit keep the per-entry derivative.
    batch = qb.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
    batched = qb.jit(qb.vmap(qb.grad(lambda row: qb.sum(qb.arctanh(row) * qb.cosh(row)))))(batch)
    for row, values in zip(batch.tolist(), batched.tolist()):
        expected = [math.cosh(p) / (1.0 - p * p) + math.atanh(p) * math.sinh(p) for p in row]
        assert_relative(values, expected, 1e-14)


# Every sign combination, zeros of both signs, infinities, NaN, and a zero
# divisor.
_REMAINDER_NUMERATORS = [
    5.5, -5.5, 6.0, -6.0, 0.0, -0.0, 1e-30, -1e-30, 1e20, -1e20, math.inf, math.nan,
]
_REMAINDER_DIVISORS = [2.0, -2.0, 3.0, -3.0, 0.1, -0.1, math.inf, -math.inf, 0.0, math.nan]


def _python_fmod(x, y):
    try:
        return math.fmod(x, y)
    except ValueError:
        return math.nan


def _python_mod(x, y):
    # Python's float % is NumPy's floor-mod, also for signed zeros and
    # infinite divisors.
    try:
        return x % y
    except ZeroDivisionError:
        return math.nan


def _same_float(lhs, rhs):
    if math.isnan(rhs):
        return math.isnan(lhs)
    return lhs == rhs and math.copysign(1.0, lhs) == math.copysign(1.0, rhs)


def test_fmod_and_mod_sign_conventions_match_numpy():
    xs = [x for x in _REMAINDER_NUMERATORS for _ in _REMAINDER_DIVISORS]
    ys = [y for _ in _REMAINDER_NUMERATORS for y in _REMAINDER_DIVISORS]
    for dtype in [qb.float64, qb.float32]:
        x, y = qb.array(xs, dtype=dtype), qb.array(ys, dtype=dtype)
        pairs = list(zip(x.tolist(), y.tolist()))
        for function, reference in [
            (qb.fmod, _python_fmod),
            (qb.mod, _python_mod),
            (qb.remainder, _python_mod),
        ]:
            expected = [reference(a, b) for a, b in pairs]
            if dtype == qb.float32:
                expected = [float32_round(value) for value in expected]
            for result in [function(x, y), qb.jit(function)(x, y)]:
                assert result.dtype == dtype
                for actual, wanted, pair in zip(result.tolist(), expected, pairs):
                    assert _same_float(actual, wanted), (function, dtype, pair, actual, wanted)
        if np is not None:
            numpy_dtype = np.float64 if dtype == qb.float64 else np.float32
            a = np.array(x.tolist(), dtype=numpy_dtype)
            b = np.array(y.tolist(), dtype=numpy_dtype)
            with np.errstate(all="ignore"):
                for function, numpy_function in [(qb.fmod, np.fmod), (qb.mod, np.mod)]:
                    expected = numpy_function(a, b).astype(np.float64).tolist()
                    for actual, wanted in zip(function(x, y).tolist(), expected):
                        assert _same_float(actual, wanted), (function, actual, wanted)
    assert qb.fmod.__name__ == "fmod" and qb.remainder is qb.mod
    # fmod is exact, also for a large quotient, where x - trunc(x / y) * y
    # would lose every digit; mod adds y to a small negative remainder.
    big = qb.array([1e20, -1e20, 2.0**60 + 1.0])
    assert qb.fmod(big, 0.1).tolist() == [math.fmod(v, 0.1) for v in big.tolist()]
    assert qb.mod(big, -3.0).tolist() == [v % -3.0 for v in big.tolist()]
    assert qb.mod(qb.array([-1e-30]), 1.0).tolist() == [1.0]
    # Broadcasting and weak Python scalars in either position.
    x32 = qb.array([[7.5], [-7.5]], dtype=qb.float32)
    divisors = [2.0, -2.0, 4.0]
    y32 = qb.array(divisors, dtype=qb.float32)
    for function, reference in [(qb.fmod, _python_fmod), (qb.mod, _python_mod)]:
        expected = [[reference(a, b) for b in divisors] for a in [7.5, -7.5]]
        for result in [function(x32, y32), qb.jit(function)(x32, y32)]:
            assert result.shape == [2, 3] and result.dtype == qb.float32
            assert result.tolist() == expected
        assert function(x32, 2.0).dtype == qb.float32
        assert function(-7.0, y32).dtype == qb.float32
        assert function(-7.0, y32).tolist() == [reference(-7.0, b) for b in divisors]
        traced = qb.jit(lambda t, f=function: f(t, -2.0))(y32)
        assert traced.tolist() == [reference(b, -2.0) for b in divisors]
    assert qb.array([7.5, -7.5]).fmod(2.0).tolist() == [1.5, -1.5]
    assert_raises(ValueError, qb.fmod, qb.array([True]), qb.array([True]), match="bool")


def test_fmod_and_mod_derivatives_are_piecewise_linear():
    x = qb.array([5.5, -5.5, 7.0, -7.25, 0.3, 1e-30])
    y = qb.array([2.0, 2.0, -3.0, -3.0, 0.1, 1.0])
    pairs = list(zip(x.tolist(), y.tolist()))
    # d fmod / dy is -trunc(x / y) of the exact quotient: 0.3 < 3 * 0.1 in
    # binary, so fmod(0.3, 0.1) is 0.1 - 2.8e-17 and the quotient is 2.
    fmod_y = [-float(round((a - math.fmod(a, b)) / b)) for a, b in pairs]
    mod_y = [-float(round((a - a % b) / b)) for a, b in pairs]
    assert fmod_y == [-2.0, 2.0, 2.0, -2.0, -2.0, -0.0]
    assert mod_y == [-2.0, 3.0, 3.0, -2.0, -2.0, -0.0]
    for function, expected_y in [(qb.fmod, fmod_y), (qb.mod, mod_y)]:

        def total(a, b, f=function):
            return qb.sum(f(a, b))

        for transform in [qb.grad, lambda f, **options: qb.jit(qb.grad(f, **options))]:
            gx, gy = transform(total, argnums=(0, 1))(x, y)
            assert gx.tolist() == [1.0] * 6 and gy.tolist() == expected_y, (function, gy)
        _, tangent = qb.jvp(function, (x, y), (qb.ones([6]), qb.full([6], 2.0)))
        assert tangent.tolist() == [1.0 + 2.0 * d for d in expected_y]
        per_entry = qb.vmap(qb.grad(function, argnums=(0, 1)))(x, y)
        assert per_entry[0].tolist() == [1.0] * 6 and per_entry[1].tolist() == expected_y

        def on_pair(p, f=function):
            return f(p[0], p[1])

        assert qb.jit(qb.hessian(on_pair))(qb.array([7.0, -3.0])).tolist() == [[0.0] * 2] * 2
    # A Python number operand gets no derivative; the array operand does.
    assert qb.grad(lambda a: qb.sum(qb.fmod(a, 2.0)))(x).tolist() == [1.0] * 6
    divisors = qb.array([2.0, -2.0])
    assert qb.grad(lambda b: qb.sum(qb.mod(5.5, b)))(divisors).tolist() == [-2.0, 3.0]
    # Central differences agree away from the jumps.
    step = 1e-7
    for a, b in [(5.5, 2.0), (-7.25, -3.0), (4.0, 0.7), (-4.0, 0.7)]:
        for function, expected in [
            (qb.fmod, -float(math.trunc(a / b))),
            (qb.mod, -float(math.floor(a / b))),
        ]:
            difference = (
                function(qb.array(a), qb.array(b + step)).item()
                - function(qb.array(a), qb.array(b - step)).item()
            ) / (2.0 * step)
            assert abs(difference - expected) <= 1e-6, (function, a, b, difference, expected)


def test_stop_gradient_keeps_values_and_zeroes_every_derivative():
    x = qb.array([0.5, -2.0, 3.0])
    assert_tensor(qb.stop_gradient(x), x.tolist(), qb.float64)
    assert_tensor(qb.jit(qb.stop_gradient)(x), x.tolist(), qb.float64)
    assert qb.grad(lambda t: qb.sum(qb.sin(qb.stop_gradient(t))))(x).tolist() == [0.0] * 3
    # The straight-through pattern: the value of f, the gradient of x.
    def pattern(t):
        return qb.sum(t - qb.stop_gradient(t) + qb.stop_gradient(t**2))

    assert pattern(x).item() == 0.25 + 4.0 + 9.0
    assert qb.jit(qb.grad(pattern))(x).tolist() == [1.0] * 3
    value, tangent = qb.jvp(lambda t: qb.stop_gradient(t) * t, (x,), (qb.ones([3]),))
    assert value.tolist() == [0.25, 4.0, 9.0] and tangent.tolist() == x.tolist()
    assert qb.hessian(lambda t: qb.sum(t * qb.stop_gradient(t)))(x).tolist() == [
        [0.0] * 3
    ] * 3
    assert qb.vmap(qb.grad(lambda t: qb.stop_gradient(t) * t))(x).tolist() == x.tolist()
    mask = qb.array([True, False])
    assert qb.jit(qb.stop_gradient)(mask).tolist() == [True, False]


def test_cumsum_matches_numpy_and_differentiates_by_reversal():
    data = [[1.0, -2.0, 3.5, 0.25], [4.0, 0.5, -1.0, 2.0], [0.0, 7.0, -3.0, 1.0]]
    x = qb.array(data)

    def reference(rows, axis, reverse):
        if axis is None:
            flat = [value for row in rows for value in row]
            order = flat[::-1] if reverse else flat
            sums, running = [], 0.0
            for value in order:
                running += value
                sums.append(running)
            return sums[::-1] if reverse else sums
        if axis in (1, -1):
            return [reference([row], None, reverse) for row in rows]
        columns = [list(column) for column in zip(*rows)]
        return [list(row) for row in zip(*[reference([c], None, reverse) for c in columns])]

    for axis in [None, 0, 1, -1]:
        for reverse in [False, True]:
            expected = reference(data, axis, reverse)
            results = [
                qb.cumsum(x, axis=axis, reverse=reverse),
                x.cumsum(axis, reverse),
                qb.jit(lambda t, a=axis, r=reverse: qb.cumsum(t, a, r))(x),
            ]
            for result in results:
                assert result.tolist() == expected, (axis, reverse, result.tolist())
            if np is not None and not reverse:
                assert result.tolist() == np.cumsum(np.array(data), axis=axis).tolist()
    # float32 rounds every running sum: 3e-8 increments never move 1.0.
    small = qb.array([1.0, 3e-8, 3e-8, 3e-8], dtype=qb.float32)
    for result in [small.cumsum(), qb.jit(qb.cumsum)(small)]:
        assert result.dtype == qb.float32 and result.tolist()[-1] == 1.0
    if np is not None:
        sequential = np.array([1.0, 3e-8, 3e-8, 3e-8], dtype=np.float32)
        running = [sequential[0]]
        for value in sequential[1:]:
            running.append(np.float32(running[-1] + value))
        assert small.cumsum().tolist() == [float(value) for value in running]
    # The gradient of sum(cumsum(x) * w) is the opposite-direction cumsum of w.
    w = qb.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.25, 2.0], [3.0, 1.0, -2.0, 1.0]])
    for axis in [0, 1]:
        for reverse in [False, True]:
            def loss(t, a=axis, r=reverse):
                return qb.sum(qb.cumsum(t, a, r) * w)

            gradient = qb.jit(qb.grad(loss))(x)
            assert gradient.tolist() == w.cumsum(axis, not reverse).tolist()
            _, tangent = qb.jvp(lambda t, a=axis, r=reverse: qb.cumsum(t, a, r), (x,), (w,))
            assert tangent.tolist() == w.cumsum(axis, reverse).tolist()
    flat_gradient = qb.grad(lambda t: qb.sum(qb.cumsum(t) * qb.cumsum(t)))(x)
    assert flat_gradient.shape == [3, 4]
    # vmap shifts the axis past the batch axis; axis=None flattens each example.
    for axis in [None, 0, -1]:
        mapped = qb.jit(qb.vmap(lambda t, a=axis: qb.cumsum(t, a, True)))(x)
        expected = [qb.array(row).cumsum(axis if axis is None else 0, True).tolist() for row in data]
        assert mapped.tolist() == expected
    batched = qb.stack([x, x * 2.0])
    assert qb.vmap(lambda t: t.cumsum(1))(batched).tolist() == batched.cumsum(2).tolist()
    assert_raises(ValueError, qb.cumsum, qb.array([True, False]), match="bool")
    assert_raises(ValueError, x.cumsum, 2)


def test_narrow_buffer_import_regressions():
    if np is None:
        skip("numpy is not installed")
    script = pathlib.Path(__file__).with_name("test_buffer_narrow_import.py")
    subprocess.run([sys.executable, str(script)], check=True)


def same_values(actual, expected):
    """Bitwise equality of two value lists, NaN matching NaN."""
    return len(actual) == len(expected) and all(
        struct.pack("<d", a) == struct.pack("<d", b) or (math.isnan(a) and math.isnan(b))
        for a, b in zip(actual, expected)
    )


def central_gradient(function, value, step=1e-6):
    """The gradient of a scalar array function by central differences."""
    base = value.to_flat_list()
    gradient = []
    for element in range(len(base)):
        plus, minus = list(base), list(base)
        plus[element] += step
        minus[element] -= step
        lhs = function(qb.array(plus).reshape(value.shape)).item()
        rhs = function(qb.array(minus).reshape(value.shape)).item()
        gradient.append((lhs - rhs) / (2 * step))
    return qb.array(gradient).reshape(value.shape)


def assert_matches_finite_differences(function, value, tolerance=1e-6):
    analytic = qb.grad(function)(value)
    assert_close(analytic, central_gradient(function, value), tolerance)


def assert_eager_matches_cpu_jit(function, *args):
    eager, traced = function(*args), qb.jit(function)(*args)
    assert eager.dtype == traced.dtype and eager.shape == traced.shape
    assert same_values(eager.to_flat_list(), traced.to_flat_list()), (
        eager.tolist(),
        traced.tolist(),
    )


def reference_logsumexp(values):
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def test_softmax_log_softmax_and_logsumexp_stay_finite_at_extreme_logits():
    for dtype in (qb.float32, qb.float64):
        x = qb.array([1000.0, 0.0], dtype=dtype)
        assert_tensor(qb.softmax(x), [1.0, 0.0], dtype)
        assert_tensor(qb.log_softmax(x), [0.0, -1000.0], dtype)
        assert_tensor(qb.logsumexp(x), 1000.0, dtype)
        assert_tensor(qb.grad(lambda t: qb.logsumexp(t))(x), [1.0, 0.0], dtype)
        for function in (
            lambda t: qb.softmax(t)[0],
            lambda t: qb.log_softmax(t)[1],
            lambda t: qb.sum(qb.log_softmax(t) * qb.array([0.25, 0.75], dtype=dtype)),
        ):
            gradient = qb.grad(function)(x).tolist()
            assert all(math.isfinite(v) and abs(v) <= 1.0 for v in gradient), gradient
        # Both entries at -1000 underflow every unshifted exp.
        low = qb.array([-1000.0, -1000.0], dtype=dtype)
        assert_close(qb.logsumexp(low), -1000.0 + math.log(2.0), 1e-7)
        assert_close(qb.softmax(low), [0.5, 0.5], 0.0)
    nan, inf = math.nan, math.inf
    cases = {
        (-inf, -inf): -inf,
        (inf, 0.0): inf,
        (inf, -inf): inf,
        (-inf, 1.5): 1.5,
    }
    for values, expected in cases.items():
        for dtype in (qb.float32, qb.float64):
            assert qb.logsumexp(qb.array(values, dtype=dtype)).item() == expected
            assert qb.jit(qb.logsumexp)(qb.array(values, dtype=dtype)).item() == expected
    assert math.isnan(qb.logsumexp(qb.array([nan, 0.0])).item())


def test_softmax_and_logsumexp_match_math_over_axes():
    rows = [[0.5, -1.25, 3.0], [-0.75, 2.0, 0.125]]
    x = qb.array(rows)
    for row, actual in zip(rows, qb.logsumexp(x, axis=1).tolist()):
        assert abs(actual - reference_logsumexp(row)) <= 1e-15 * abs(actual)
    columns = list(zip(*rows))
    assert_close(qb.logsumexp(x, axis=0), [reference_logsumexp(c) for c in columns], 1e-15)
    assert_close(
        qb.logsumexp(x, axis=(0, 1), keepdims=True),
        [[reference_logsumexp(rows[0] + rows[1])]],
        1e-15,
    )
    assert_close(qb.logsumexp(x), reference_logsumexp(rows[0] + rows[1]), 1e-15)
    expected = [[math.exp(v - reference_logsumexp(row)) for v in row] for row in rows]
    assert_close(qb.softmax(x), expected, 1e-15)
    assert_close(qb.softmax(x, axis=-1).sum(axis=-1), [1.0, 1.0], 1e-15)
    assert_close(
        qb.log_softmax(x, axis=0),
        [[v - reference_logsumexp(c) for v, c in zip(row, columns)] for row in rows],
        1e-15,
    )
    weights = qb.array([[0.3, -1.0, 2.0], [1.5, 0.25, -0.5]])
    for function in (
        lambda t: qb.logsumexp(t, axis=1).sum(),
        lambda t: qb.sum(qb.softmax(t, axis=0) * weights),
        lambda t: qb.sum(qb.log_softmax(t) * weights),
    ):
        assert_matches_finite_differences(function, x)
        assert_eager_matches_cpu_jit(function, x)
    assert_close(qb.grad(lambda t: qb.logsumexp(t, axis=1).sum())(x), qb.softmax(x), 1e-15)
    batched = qb.vmap(qb.softmax)(x)
    assert_close(batched, qb.softmax(x, axis=-1), 0.0)
    assert_close(qb.vmap(qb.logsumexp)(x), qb.logsumexp(x, axis=1), 0.0)


def test_var_and_std_are_two_pass_and_differentiate():
    offset = qb.array([1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0])
    assert qb.var(offset).item() == 2.0 / 3.0
    assert qb.var(offset, ddof=1).item() == 1.0
    assert qb.std(offset, ddof=1).item() == 1.0
    # The one-pass formula loses every digit at this offset.
    naive = qb.mean(offset * offset) - qb.mean(offset) ** 2
    assert abs(naive.item() - 2.0 / 3.0) > 0.1
    offset32 = qb.array([1e4 + 1.0, 1e4 + 2.0, 1e4 + 3.0], dtype=qb.float32)
    two_thirds32 = struct.unpack("<f", struct.pack("<f", 2.0 / 3.0))[0]
    assert_tensor(qb.var(offset32), two_thirds32, qb.float32)
    rows = [[1.0, 4.0, -2.0, 0.5], [3.0, 3.5, -1.0, 8.0], [0.25, -6.0, 2.0, 1.0]]
    x = qb.array(rows)

    def reference(values, ddof):
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - ddof)

    assert_close(qb.var(x, axis=1), [reference(r, 0) for r in rows], 1e-15)
    assert_close(qb.var(x, axis=0, ddof=1), [reference(c, 1) for c in zip(*rows)], 1e-15)
    assert_close(qb.var(x, axis=(0, 1)), reference(rows[0] + rows[1] + rows[2], 0), 1e-15)
    assert qb.var(x, axis=-1, keepdims=True).shape == [3, 1]
    assert_close(qb.std(x, axis=1), [math.sqrt(reference(r, 0)) for r in rows], 1e-15)
    for function in (
        lambda t: qb.var(t, axis=1).sum(),
        lambda t: qb.std(t, axis=0, ddof=1).sum(),
        lambda t: qb.std(t),
    ):
        assert_matches_finite_differences(function, x)
        assert_eager_matches_cpu_jit(function, x)
    # d var / dx = 2 (x - mean) / (n - ddof); zero variance gives std a zero
    # derivative under the sqrt convention.
    assert_close(qb.grad(lambda t: qb.var(t, ddof=1))(offset), [-1.0, 0.0, 1.0], 0.0)
    flat = qb.array([2.0, 2.0, 2.0])
    assert_tensor(qb.grad(qb.std)(flat), [0.0, 0.0, 0.0], qb.float64)
    assert math.isnan(qb.var(qb.array([5.0]), ddof=1).item())
    assert_close(qb.vmap(qb.var)(x), qb.var(x, axis=1), 0.0)
    assert_raises(ValueError, qb.var, x, axis=2)
    assert_raises(ValueError, qb.var, x, axis=(1, -1))


def test_silu_gelu_clip_sign_square_and_reciprocal():
    points = [-50.0, -8.0, -5.0, -3.0, -0.5, 0.0, 0.5, 3.0, 50.0]

    def gelu_reference(v):
        u = math.sqrt(2.0 / math.pi) * (v + 0.044715 * v**3)
        return v / (1.0 + math.exp(-2.0 * u)) if u > -700 else 0.0

    for dtype, tolerance in ((qb.float64, 1e-15), (qb.float32, 1e-6)):
        x = qb.array(points, dtype=dtype)
        for actual, v in zip(qb.silu(x).tolist(), points):
            expected = v / (1.0 + math.exp(-v))
            assert abs(actual - expected) <= tolerance * max(abs(expected), 1e-30), (v, actual)
        for actual, v in zip(qb.gelu(x).tolist(), points):
            expected = gelu_reference(v)
            # Relative accuracy holds in the negative tail too (-5 gives -2.3e-7).
            assert abs(actual - expected) <= 4 * tolerance * abs(expected), (dtype, v, actual)
        for function in (qb.silu, qb.gelu, qb.sign, qb.square, qb.reciprocal):
            assert function(x).dtype == dtype
            assert_eager_matches_cpu_jit(function, x)
    tanh_form = [
        0.5 * v * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3)))
        for v in (-1.0, 0.25, 2.0)
    ]
    assert_close(qb.gelu(qb.array([-1.0, 0.25, 2.0])), tanh_form, 1e-15)
    assert qb.silu(qb.array([-1e4], dtype=qb.float32)).item() == 0.0
    assert qb.gelu(qb.array([-1e4], dtype=qb.float32)).item() == 0.0
    exact = [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in (-1.0, 0.25, 2.0)]
    assert_close(qb.gelu(qb.array([-1.0, 0.25, 2.0]), approximate=False), exact, 1e-15)
    # Through erfc, the exact form keeps its relative accuracy in the far
    # negative tail, where 1 + erf(x / sqrt(2)) cancels to zero.
    tail = [-38.0, -20.0, -8.0]
    reference = [0.5 * v * math.erfc(-v / math.sqrt(2.0)) for v in tail]
    assert_relative(qb.gelu(qb.array(tail), approximate=False).tolist(), reference, 1e-13)
    tail32 = qb.gelu(qb.array([-12.0, -8.0], dtype=qb.float32), approximate=False).tolist()
    assert_relative(tail32, [0.5 * v * math.erfc(-v / math.sqrt(2.0)) for v in (-12.0, -8.0)], 1e-5)
    assert_matches_finite_differences(
        lambda t: qb.gelu(t, approximate=False).sum(), qb.array([-3.0, -0.75, 0.0, 0.5, 2.5])
    )
    smooth = qb.array([-3.0, -0.75, 0.0, 0.5, 2.5])
    for function in (lambda t: qb.silu(t).sum(), lambda t: qb.gelu(t).sum()):
        assert_matches_finite_differences(function, smooth)
    assert_close(qb.grad(lambda t: qb.square(t).sum())(smooth), smooth * 2.0, 0.0)
    assert_close(
        qb.grad(lambda t: qb.reciprocal(t).sum())(qb.array([2.0, -4.0])), [-0.25, -0.0625], 0.0
    )

    nan, inf = math.nan, math.inf
    values = qb.array([nan, -2.0, -1.0, 0.5, 3.0, -inf, inf])
    clipped = qb.clip(values, -1.0, 1.0).tolist()
    assert math.isnan(clipped[0]) and clipped[1:] == [-1.0, -1.0, 0.5, 1.0, -1.0, 1.0]
    assert qb.clip(values, lo=0.0).tolist()[1:] == [0.0, 0.0, 0.5, 3.0, 0.0, inf]
    assert qb.clip(values, hi=0.0).tolist()[1:] == [-2.0, -1.0, 0.0, 0.0, -inf, 0.0]
    bounds = qb.array([0.0, 0.0, -0.5, 1.0, 2.0, 0.0, 0.0])
    assert qb.clip(values, bounds, 2.5).tolist()[1:] == [0.0, -0.5, 1.0, 2.5, 0.0, 2.5]
    assert math.isnan(qb.clip(qb.array([1.0]), nan, 2.0).item())
    gradient = qb.grad(lambda t: qb.clip(t, -1.0, 1.0).sum())(qb.array([-2.0, -0.5, 0.75, 4.0]))
    assert_tensor(gradient, [0.0, 1.0, 1.0, 0.0], qb.float64)
    x32 = qb.array([0.5, 4.0], dtype=qb.float32)
    assert_tensor(qb.clip(x32, 0.0, 1.0), [0.5, 1.0], qb.float32)

    signs = qb.sign(qb.array([nan, -2.0, -0.0, 0.0, 3.0, -inf, inf], dtype=qb.float32))
    assert signs.dtype == qb.float32
    assert math.isnan(signs.tolist()[0]) and signs.tolist()[1:] == [-1.0, 0.0, 0.0, 1.0, -1.0, 1.0]
    gradient = qb.grad(lambda t: qb.sign(t).sum())(qb.array([-2.0, 0.0, 3.0]))
    assert_tensor(gradient, [0.0, 0.0, 0.0], qb.float64)
    assert_close(qb.vmap(qb.gelu)(smooth), qb.gelu(smooth), 0.0)


def test_reshape_accepts_varargs_and_one_inferred_extent():
    x = qb.arange(6.0)
    for shape in ([2, 3], (2, 3), [2, -1], (-1, 3)):
        assert x.reshape(shape).shape == [2, 3]
    assert x.reshape(3, 2).shape == [3, 2]
    assert x.reshape(-1, 1).shape == [6, 1]
    assert x.reshape(2, 3).reshape(-1).tolist() == x.tolist()
    assert x.reshape(6).shape == [6]
    assert qb.reshape(x, (3, -1)).shape == [3, 2]
    assert qb.reshape(x, -1).shape == [6]
    assert qb.array([5.0]).reshape([]).shape == []
    assert_raises(ValueError, x.reshape, -1, -1, match="one dimension")
    assert_raises(ValueError, x.reshape, 4, -1, match="size 6")
    assert_raises(ValueError, x.reshape, 2, -2, match="negative")
    assert_raises(TypeError, x.reshape)
    assert_raises(ValueError, x.reshape, 4, 2)
    traced = qb.jit(lambda t: t.reshape(3, -1) * 2.0)(x)
    assert traced.shape == [3, 2] and traced.tolist() == [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]]
    # Under vmap the extents are per example.
    batched = qb.vmap(lambda t: t.reshape(-1, 2))(x.reshape(2, 3, 1).broadcast_to([2, 3, 2]))
    assert batched.shape == [2, 3, 2]
    gradient = qb.grad(lambda t: (t.reshape(-1) * qb.arange(6.0)).sum())(x.reshape(2, 3))
    assert gradient.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_transpose_property_reverses_axes():
    x = qb.arange(24.0).reshape(2, 3, 4)
    assert x.T.shape == [4, 3, 2]
    assert x.T.tolist() == x.transpose().tolist()
    assert qb.arange(3.0).T.tolist() == [0.0, 1.0, 2.0]
    m = qb.array([[1.0, 2.0], [3.0, 4.0]])
    assert qb.jit(lambda t: t.T @ t)(m).tolist() == (m.T @ m).tolist()
    gradient = qb.grad(lambda t: (t.T * qb.array([[1.0, 2.0], [3.0, 4.0]])).sum())(m)
    assert gradient.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_indexing_supports_none_ellipsis_and_strides():
    x = qb.arange(24.0).reshape(2, 3, 4)
    keys = [
        (None,),
        (Ellipsis, 0),
        (slice(None), None),
        (None, Ellipsis, None),
        (Ellipsis, None, 1),
        (0, Ellipsis),
        (Ellipsis,),
        (1, None, slice(None, None, 2), -1),
        (slice(None, None, -1),),
        (slice(None), slice(None, None, 2)),
        (Ellipsis, slice(None, None, -2)),
        (slice(1, None, -1), 2, slice(3, 0, -2)),
        (slice(None), slice(-1, None)),
        (Ellipsis, slice(1, 3), None),
    ]
    if np is not None:
        reference = np.arange(24.0).reshape(2, 3, 4)
        for key in keys:
            expected = reference[key]
            for value in (x[key], qb.jit(lambda t, key=key: t[key])(x)):
                assert value.shape == list(expected.shape), (key, value.shape)
                assert value.tolist() == expected.tolist(), key
    assert x[None].shape == [1, 2, 3, 4]
    assert x[..., 0].tolist() == x[:, :, 0].tolist()
    assert x[0, None, ::-1, 0].tolist() == [[8.0, 4.0, 0.0]]
    # Every key form differentiates: the gradient scatters back to the
    # selected coordinates, twice for a coordinate selected twice.
    weights = qb.arange(1.0, 7.0).reshape(1, 2, 3)
    gradient = qb.grad(lambda t: (t[None, ::-1, 2::-2, 1] * weights[..., :2]).sum())(x)
    # t[::-1, 2::-2, 1][b, k] is x[1 - b, 2 - 2k, 1], weighted by weights[0, b, k].
    expected = [[[0.0] * 4 for _ in range(3)] for _ in range(2)]
    for b in range(2):
        for k in range(2):
            expected[1 - b][2 - 2 * k][1] += weights.tolist()[0][b][k]
    assert gradient.tolist() == expected
    batched = qb.vmap(lambda row: row[::-1, None])(qb.arange(6.0).reshape(2, 3))
    assert batched.tolist() == [[[2.0], [1.0], [0.0]], [[5.0], [4.0], [3.0]]]
    assert_raises(IndexError, x.__getitem__, (Ellipsis, 0, Ellipsis))
    assert_raises(IndexError, x.__getitem__, (0, 0, 0, 0))
    assert_raises(IndexError, x.__getitem__, (None, 0, 0, 0, 0))
    assert_raises(ValueError, x.__getitem__, slice(2, 1))
    assert_raises(TypeError, x.__getitem__, "a")
    assert_raises(IndexError, qb.jit(lambda t: t[..., 0, ...]), x)


def test_squeeze_expand_dims_split_and_meshgrid():
    x = qb.arange(6.0).reshape(1, 2, 1, 3)
    assert qb.squeeze(x).shape == [2, 3]
    assert qb.squeeze(x, axis=0).shape == [2, 1, 3]
    assert qb.squeeze(x, axis=(0, -2)).shape == [2, 3]
    assert qb.squeeze(qb.array([[4.0]])).shape == []
    assert_raises(ValueError, qb.squeeze, x, axis=1, match="extent is not 1")
    y = qb.arange(6.0).reshape(2, 3)
    assert qb.expand_dims(y, 0).shape == [1, 2, 3]
    assert qb.expand_dims(y, -1).shape == [2, 3, 1]
    assert qb.expand_dims(y, (0, 3)).shape == [1, 2, 3, 1]
    assert qb.expand_dims(y, (1, -1)).shape == [2, 1, 3, 1]
    assert_raises(ValueError, qb.expand_dims, y, 4)

    z = qb.arange(12.0).reshape(3, 4)
    pieces = qb.split(z, 2, axis=1)
    assert [p.tolist() for p in pieces] == [
        [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]],
        [[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]],
    ]
    assert [p.shape for p in qb.split(z, [1, -1])] == [[1, 4], [1, 4], [1, 4]]
    assert [p.shape for p in qb.split(z, [1, 3], axis=-1)] == [[3, 1], [3, 2], [3, 1]]
    assert_raises(ValueError, qb.split, z, 3, axis=1, match="equal")
    assert_raises(ValueError, qb.split, z, [2, 2], match="empty")
    gradient = qb.grad(lambda t: (qb.split(t, [1], axis=0)[1] * 2.0).sum())(z)
    assert gradient.tolist() == [[0.0] * 4, [2.0] * 4, [2.0] * 4]
    traced = qb.jit(lambda t: qb.split(t, 4, axis=1)[3])(z)
    assert traced.tolist() == [[3.0], [7.0], [11.0]]

    a, b, c = qb.array([1.0, 2.0, 3.0]), qb.array([10.0, 20.0]), qb.array([5.0])
    for indexing in ("xy", "ij"):
        grids = qb.meshgrid(a, b, c, indexing=indexing)
        if np is not None:
            expected = np.meshgrid([1.0, 2.0, 3.0], [10.0, 20.0], [5.0], indexing=indexing)
            assert [g.tolist() for g in grids] == [e.tolist() for e in expected]
    xs, ys = qb.meshgrid(a, b)
    assert xs.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert ys.tolist() == [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
    assert [g.tolist() for g in qb.meshgrid(a)] == [[1.0, 2.0, 3.0]]
    assert_raises(ValueError, qb.meshgrid, a, indexing="xyz")
    gradient = qb.grad(lambda t: (qb.meshgrid(t, b)[0] * qb.meshgrid(t, b)[1]).sum())(a)
    assert gradient.tolist() == [30.0, 30.0, 30.0]


def test_like_constructors_follow_shape_and_dtype():
    x = qb.arange(6.0).reshape(2, 3).astype(qb.float32)
    assert_tensor(qb.zeros_like(x), [[0.0] * 3] * 2, qb.float32)
    assert_tensor(qb.ones_like(x), [[1.0] * 3] * 2, qb.float32)
    assert_tensor(qb.full_like(x, 2.5), [[2.5] * 3] * 2, qb.float32)
    assert_tensor(qb.full_like(x, 7.0, dtype=qb.float64), [[7.0] * 3] * 2, qb.float64)
    assert_tensor(qb.zeros_like(qb.array([True, False])), [False, False], qb.bool_)
    assert_tensor(qb.ones_like([1.0, 2.0]), [1.0, 1.0], qb.float64)
    shifted = qb.jit(lambda t: t + qb.ones_like(t))(x)
    assert_tensor(shifted, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], qb.float32)
    assert_tensor(qb.jit(qb.zeros_like)(x), [[0.0] * 3] * 2, qb.float32)
    gradient = qb.grad(lambda t: (t * qb.full_like(t, 3.0)).sum())(qb.array([1.0, 2.0]))
    assert_tensor(gradient, [3.0, 3.0], qb.float64)
    doubled = qb.vmap(lambda t: t * qb.full_like(t, 2.0))(x)
    assert_tensor(doubled, [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]], qb.float32)


def test_matmul_follows_numpy_rules_for_vectors():
    v, w = qb.array([1.0, 2.0, 3.0]), qb.array([4.0, -1.0, 0.5])
    m = qb.arange(6.0).reshape(2, 3)
    stack = qb.arange(24.0).reshape(2, 3, 4)
    assert_tensor(qb.matmul(v, w), 3.5, qb.float64)
    assert_tensor(qb.matmul(m, v), [8.0, 26.0], qb.float64)
    assert_tensor(qb.matmul(qb.array([1.0, -1.0]), m), [-3.0, -3.0, -3.0], qb.float64)
    assert qb.matmul(v, stack).shape == [2, 4]
    assert qb.matmul(stack, qb.ones([4])).shape == [2, 3]
    assert qb.matmul(m, m.T).shape == [2, 2]
    if np is not None:
        s = np.arange(24.0).reshape(2, 3, 4)
        assert qb.matmul(v, stack).tolist() == np.matmul([1.0, 2.0, 3.0], s).tolist()
        assert qb.matmul(stack, qb.ones([4])).tolist() == np.matmul(s, np.ones(4)).tolist()
    assert_tensor(qb.jit(qb.matmul)(v, w), 3.5, qb.float64)
    assert_tensor(qb.jit(qb.matmul)(m, v), [8.0, 26.0], qb.float64)
    gx, gy = qb.grad(lambda a, b: qb.matmul(a, b), argnums=(0, 1))(v, w)
    assert gx.tolist() == w.tolist() and gy.tolist() == v.tolist()
    gm = qb.grad(lambda a: qb.matmul(a, v).sum())(m)
    assert gm.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert_tensor(qb.vmap(qb.matmul, in_axes=(0, None))(m, v), [8.0, 26.0], qb.float64)
    assert_raises(ValueError, qb.matmul, v, qb.ones([2]))


def test_prod_is_exact_with_zeros_and_differentiates_without_division():
    # NumPy supplies the reference products; it is an optional dependency.
    if np is not None:
        rng = np.random.default_rng(3)
        data = rng.uniform(-2.0, 2.0, size=(3, 5, 4))
        for axis in [None, 0, 1, -1, (0, 2), (2, 0, 1)]:
            for keepdims in [False, True]:
                expected = np.prod(data, axis=axis, keepdims=keepdims)
                value = qb.prod(data, axis=axis, keepdims=keepdims)
                assert value.shape == list(np.shape(expected))
                assert_close(value, expected.tolist(), 1e-15)
                assert_close(qb.asarray(data).prod(axis, keepdims), expected.tolist(), 1e-15)
                jitted = qb.jit(lambda t, a=axis, k=keepdims: qb.prod(t, axis=a, keepdims=k))(data)
                assert jitted.tolist() == value.tolist()
        # Eager float32 values round exactly like the traced graph.
        data32 = qb.asarray(rng.uniform(0.5, 1.5, size=(4, 37)), dtype=qb.float32)
        for axis in [None, 1]:
            eager = qb.prod(data32, axis=axis)
            assert eager.dtype == qb.float32
            jitted = qb.jit(lambda t, a=axis: qb.prod(t, a))(data32)
            assert float_bits(np.ravel(eager.tolist()).tolist()) == float_bits(np.ravel(jitted.tolist()).tolist())
            reference = np.prod(np.asarray(data32, dtype=np.float64), axis=axis)
            assert_relative(np.ravel(eager.tolist()).tolist(), np.ravel(reference).tolist(), 1e-6)
    # Gradients are the products of the other entries: with one zero only
    # that entry has a nonzero gradient; with two zeros every entry is zero.
    one_zero = qb.array([2.0, 0.0, -3.0, 7.0, 5.0])
    assert qb.grad(qb.prod)(one_zero).tolist() == [0.0, -210.0, 0.0, 0.0, 0.0]
    two_zeros = qb.array([2.0, 0.0, -3.0, 0.0, 5.0])
    assert qb.grad(qb.prod)(two_zeros).tolist() == [0.0] * 5
    no_zero = qb.array([2.0, -0.5, 3.0, 4.0])
    assert qb.grad(qb.prod)(no_zero).tolist() == [-6.0, 24.0, -4.0, -3.0]
    _, tangent = qb.jvp(qb.prod, (one_zero,), (qb.ones([5]),))
    assert tangent.item() == -210.0
    # The Hessian holds the products of all entries but two, zero on the
    # diagonal, and stays exact with a zero entry.
    hessian = qb.hessian(qb.prod)(one_zero).tolist()
    values = one_zero.tolist()
    for i in range(5):
        for j in range(5):
            expected = 0.0 if i == j else math.prod(v for k, v in enumerate(values) if k not in (i, j))
            assert hessian[i][j] == expected, (i, j, hessian[i][j], expected)
    hessian = qb.hessian(qb.prod)(two_zeros).tolist()
    assert hessian[1][3] == hessian[3][1] == -30.0
    assert sum(abs(v) for row in hessian for v in row) == 60.0
    # Batched rows and gradients under vmap and jit.
    rows = qb.array([[1.0, 2.0, 0.0], [3.0, -1.0, 2.0]])
    assert qb.vmap(qb.prod)(rows).tolist() == [0.0, -6.0]
    assert qb.jit(qb.grad(lambda t: qb.prod(t, axis=1).sum()))(rows).tolist() == [
        [0.0, 0.0, 2.0],
        [-2.0, 6.0, -3.0],
    ]
    # Large values: the pairwise tree keeps 1e30 * 1e30 * 1e-30 * 1e-30 finite
    # in float32, and a product past the range is inf.
    big = qb.array([1e30, 1e30, 1e-30, 1e-30], dtype=qb.float32)
    assert qb.prod(big).item() == 1.0
    assert qb.prod(qb.array([1e200, 1e200])).item() == math.inf
    assert qb.prod(qb.array([-1e200, 1e200])).item() == -math.inf
    assert_raises(ValueError, qb.prod, qb.array([True, False]), match="bool")


if __name__ == "__main__":
    run(globals())
