import collections
import ctypes
import gc
import importlib
import importlib.machinery
import inspect
import math
import os
import pathlib
import pickle
import subprocess
import sys

import quabla
import quabla as qb

# NumPy is optional for quabla, so NumPy-dependent tests skip without it.
try:
    import numpy as np
except ImportError:
    np = None

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
    assert repr(tensor) == "Tensor(shape=[2], dtype=quabla.float32)"


def requires_numpy(test):
    def run():
        if np is None:
            print(f"skipped {test.__name__}: numpy is not installed")
            return
        test()

    run.__name__ = test.__name__
    return run


def assert_raises(error, function, *args, match=None, **kwargs):
    try:
        function(*args, **kwargs)
    except error as exc:
        if match is not None:
            assert match in str(exc), f"{match!r} not in {str(exc)!r}"
        return exc
    raise AssertionError(f"{function} did not raise {error.__name__}")


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
        ([[[1.0], [2.0]], [[3.0], [4.0]]], qb.float64, [[[1.0], [2.0]], [[3.0], [4.0]]]),
    ]
    for value, dtype, expected in cases:
        assert_tensor(qb.array(value), expected, dtype)
        assert_tensor(qb.asarray(value), expected, dtype)
    assert qb.array([1.0, 2.0]).shape == [2]
    assert qb.array(1.5).shape == []
    # An explicit dtype rounds the values.
    assert_tensor(qb.array([0.1, 2.0], dtype=qb.float32), [0.10000000149011612, 2.0], qb.float32)
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
    assert_raises(ValueError, tensor.item, match="exactly one element, got shape [2, 3]")
    assert_raises(TypeError, float, tensor, match="single-element tensors")


def test_single_element_tensors_stay_tensor_operands():
    # `float(t)` must not let a Tensor slip into scalar-operand parsing,
    # which would turn it into a weak scalar and drop its dtype.
    scalar = qb.array(2.0, dtype=qb.float32)
    # A tensor exponent keeps its float32 dtype, so the strict promotion rejects the float64 base.
    assert_raises(ValueError, lambda: qb.array([1.0, 2.0]) ** scalar, match="mismatched dtypes")

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
            ValueError, qb.tensor_jit_fn, function, [("x", [2])], match="mismatched dtypes"
        )
        compiled = qb.tensor_jit_fn(function, [("x", [2], qb.float32)])
        assert compiled({"x": qb.array([1.0, 2.0], dtype=qb.float32)}).dtype == qb.float32


def test_factories():
    assert_tensor(qb.zeros(3), [0.0, 0.0, 0.0], qb.float64)
    assert_tensor(qb.zeros((2, 1), dtype=qb.float32), [[0.0], [0.0]], qb.float32)
    assert_tensor(qb.ones([2]), [1.0, 1.0], qb.float64)
    assert_tensor(qb.ones((), dtype=qb.bool_), True, qb.bool_)
    assert_tensor(qb.full((2, 2), 2.5), [[2.5, 2.5], [2.5, 2.5]], qb.float64)
    assert_tensor(qb.full(2, True), [True, True], qb.bool_)
    assert_tensor(qb.full(2, 0.1, dtype=qb.float32), [0.10000000149011612] * 2, qb.float32)
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
    assert_raises((BufferError, TypeError), ctypes.c_double.from_buffer, qb.array([1.0]))


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
        expected = [False, True, True, False] if dtype == qb.bool_ else [0.0, 1.0, 1.0, 0.0]
        assert tensor.tolist() == expected, (np_dtype, tensor.tolist())
        # NumPy scalars keep the same mapping.
        assert qb.array(source[1]).dtype == dtype
        assert qb.array(source[1]).shape == []
    assert qb.array(np.array(2.5)).tolist() == 2.5
    assert qb.array(np.array([1, 2], dtype=np.int64), dtype=qb.float32).dtype == qb.float32
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
    assert converted.dtype == np.float32 and converted.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert_raises(ValueError, tensor.__array__, np.float32, copy=False, match="without a copy")
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


def assert_close(actual, expected, tolerance=1e-12):
    actual, expected = qb.asarray(actual), qb.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    for lhs, rhs in zip(actual.to_flat_list(), expected.to_flat_list()):
        assert abs(lhs - rhs) <= tolerance * max(1.0, abs(rhs)), (actual.tolist(), expected.tolist())


def traced(function, *arrays):
    """Evaluates `function` on `arrays` through a CPU trace."""
    specs = [(f"a{index}", array.shape, array.dtype) for index, array in enumerate(arrays)]
    compiled = qb.tensor_jit_fn(function, specs)
    return compiled({f"a{index}": array for index, array in enumerate(arrays)})


def test_module_level_unary_and_reduction_ops_match_methods():
    x = qb.array([[0.25, 0.5], [1.5, 2.0]])
    unary = ["abs", "cos", "exp", "log", "relu", "sigmoid", "sin", "softplus", "sqrt", "tanh"]
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
        assert_close(function(x, 1, keepdims=True), getattr(x, name)(axis=1, keepdims=True))
        assert function(x, axis=(0, 1), keepdims=True).shape == [1, 1]
        assert_close(traced(lambda a, f=function: f(a, axis=1), x), getattr(x, name)(axis=1))
    mask = qb.array([[True, False], [True, True]])
    assert qb.any(mask).item() is True
    assert qb.all(mask).item() is False
    assert qb.all(mask, axis=1).tolist() == [False, True]


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
    assert_tensor(qb.minimum(x, qb.array([0.0, 3.0], dtype=qb.float32)), [-1.0, 2.0], qb.float32)
    assert_tensor(qb.maximum(1.0, 2.0), 2.0, qb.float64)
    assert_raises(ValueError, qb.maximum, x, qb.array([0.0, 3.0]), match="mismatched dtypes")


def test_module_level_shape_and_linear_algebra_ops():
    x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_tensor(qb.reshape(x, (3, 2)), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], qb.float64)
    assert qb.reshape(x, 6).shape == [6]
    assert qb.transpose(x).shape == [3, 2]
    assert qb.transpose(x, (1, 0)).tolist() == x.transpose([1, 0]).tolist()
    assert qb.broadcast_to(qb.array([1.0, 2.0, 3.0]), (2, 3)).tolist() == [[1.0, 2.0, 3.0]] * 2
    assert qb.astype(x, qb.float32).dtype == qb.float32
    square = qb.array([[4.0, 2.0], [2.0, 3.0]])
    assert_close(qb.cholesky(square), square.cholesky())
    assert qb.tril(square).tolist() == [[4.0, 0.0], [2.0, 3.0]]
    assert qb.triu(square).tolist() == [[4.0, 2.0], [0.0, 3.0]]
    assert_close(qb.matmul(square, x[:, 0:2]), square @ x[:, 0:2])
    assert_close(traced(lambda a, b: qb.matmul(a, b), square, x), square @ x)
    assert_close(traced(lambda a: qb.reshape(qb.transpose(a), (6,)), x), x.transpose().reshape([6]))
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
            assert (math.isnan(lhs) and math.isnan(rhs)) or lhs == rhs, (actual, expected)

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
    assert_raises(TypeError, qb.tensor_jit_fn, lambda a: pow(a, 2, 3), [("x", [2])], match="modulo")
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
    hessian = traced_power.graph.hessian_scalar(traced_power.output.node_id, "x", inputs)
    assert hessian == [[2.0, 0.0], [0.0, 2.0]]
    hvp = qb.tensor_hvp_scalar_fn(lambda a: (a**2.0).sum(), [("x", [2])], "x")
    assert hvp(inputs, qb.array([1.0, -1.0])).tolist() == [2.0, -2.0]


def assert_power_device_parity(value_and_grad_fn):
    x = qb.array([0.0, 0.5, 1.5, 2.25, -2.0])
    y = qb.array([0.5, 0.5, -1.0, 2.5, 3.0])
    specs = [("x", [5]), ("y", [5])]

    def loss(a, b):
        return (a**b + 2.0**b * a**0.5 + a**2).sum()

    cpu_value, cpu_gradients = qb.tensor_value_and_grad_fn(loss, specs)({"x": x, "y": y})
    value, gradients = value_and_grad_fn(loss, specs, ["x", "y"])({"x": x, "y": y})
    assert math.isnan(cpu_value.item()) and math.isnan(value.item())
    for name in ["x", "y"]:
        for actual, expected in zip(gradients[name].tolist(), cpu_gradients[name].tolist()):
            if math.isnan(expected):
                assert math.isnan(actual), (name, actual)
            else:
                assert abs(actual - expected) <= 1e-5 * max(1.0, abs(expected)), (name, actual, expected)


def test_mlx_power_matches_cpu():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return
    assert_power_device_parity(qb.tensor_value_and_grad_mlx_fn)


def test_cuda_power_matches_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return
    assert_power_device_parity(
        lambda loss, specs, names: qb.tensor_value_and_grad_cuda_fn(loss, specs, names, 0)
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
    doubled = qb.tree.map(lambda x, y: x + y, {"a": 1.0, "b": [2.0]}, {"a": 10.0, "b": [20.0]})
    assert doubled == {"a": 11.0, "b": [22.0]}
    assert_raises(ValueError, qb.tree.map, lambda x, y: x, [1.0], (1.0,), match="structures differ")
    assert_raises(ValueError, qb.tree.unflatten, treedef, [1.0], match="has 3 leaves, got 1")
    assert_raises(TypeError, qb.tree.flatten, {1: 2.0}, match="keys must be strings")
    # Container subclasses are leaves (D6: no NamedTuple support in v0.2).
    point = collections.namedtuple("Point", "x y")(1.0, 2.0)
    assert qb.tree.flatten([point])[0] == [point]


def test_error_classes_subclass_the_builtins_raised_before():
    assert issubclass(qb.TracerError, qb.QuablaError) and issubclass(qb.TracerError, TypeError)
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
        old_value, old_grads = qb.tensor_value_and_grad_fn(masked_loss, specs)({"x": x, "w": w})
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
    assert_raises(ValueError, qb.grad(masked_loss, argnums=2), x, w, match="out of range")
    assert_raises(ValueError, qb.grad(masked_loss, argnums=(0, -2)), x, w, match="duplicate")
    assert_raises(TypeError, qb.grad(masked_loss, argnums=(0, "w")), x, w, match="tuple of ints")


def init_mlp(dtype):
    def values(count, seed):
        return [math.sin((index + 1) * seed) * 0.9 for index in range(count)]

    return {
        "layers": [
            {"w": qb.array(values(6, 0.37), dtype=dtype).reshape([2, 3]), "b": qb.zeros([3], dtype)},
            {"w": qb.array(values(3, 0.71), dtype=dtype).reshape([3, 1]), "b": qb.ones([1], dtype)},
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
            "w0": layers[0]["w"], "b0": layers[0]["b"],
            "w1": layers[1]["w"], "b1": layers[1]["b"], "x": x, "y": y,
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
        return qb.sum(prediction**2), {"prediction": prediction, "scale": 2.0, "none": None}

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
        TypeError, qb.grad(lambda w: qb.sum(w), has_aux=True), w, match="(value, aux) pair"
    )


def test_grad_errors_and_python_scalar_arguments():
    x = qb.array([1.0, 2.0])
    assert_raises(ValueError, qb.grad(lambda x: x * 2.0), x, match="requires a scalar output")
    assert_raises(ValueError, qb.grad(lambda x: qb.sum(x) > 1.0), x, match="bool output")
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
    gx, gmask = qb.grad(lambda x, m: qb.sum(qb.where(m, x, 0.0)), argnums=(0, 1))(x, mask)
    assert_close(gx, [1.0, 0.0])
    assert gmask is None
    assert_raises(
        TypeError, qb.grad(lambda x: qb.sum(x)), "abc", match="unsupported argument leaf"
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
        old_value, old_tangent = qb.tensor_jvp_fn(model, specs)({"x": x, "w": w}, {"x": dx, "w": dw})
        value, tangent = qb.jvp(model, (x, w), (dx, dw))
        tolerance = 1e-6 if dtype == qb.float32 else 1e-14
        assert value.dtype == tangent.dtype == dtype
        assert_close(value, old_value, tolerance)
        assert_close(tangent, old_tangent, tolerance)
    # Python number tangents adopt the primal dtype; pytrees of outputs work.
    x32 = qb.array(2.0, dtype=qb.float32)
    (square, pair), (dsquare, dpair) = qb.jvp(lambda x: (x * x, [x, 1.0]), (x32,), (1.0,))
    assert_tensor(dsquare, 4.0, qb.float32)
    assert_tensor(dpair[0], 1.0, qb.float32)
    assert pair[1] == 1.0 and dpair[1] == 0.0
    assert_raises(TypeError, qb.jvp, model, (x, w), (dx, qb.array([[0.2], [0.1]])),
                  match="dtype quabla.float32")
    assert_raises(ValueError, qb.jvp, model, (x, w), (dx,), match="must match")
    assert_raises(ValueError, qb.jvp, model, (x, w), (dx, [dw]), match="pytree structure")


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
    out, pullback = qb.vjp(lambda x: {"a": x * 2.0, "b": qb.sum(x)}, qb.array([1.0, 2.0]))
    assert_close(out["a"], [2.0, 4.0])
    (gx,) = pullback({"a": qb.array([1.0, 3.0]), "b": 1.0})
    assert_close(gx, [3.0, 7.0])
    assert_raises(ValueError, pullback, [1.0, 1.0], match="pytree structure")
    out, pullback, aux = qb.vjp(lambda x: (x * x, x + 1.0), qb.array([3.0]), has_aux=True)
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
    blocks = qb.jacobian(lambda p: p["a"] * p["b"])({"a": qb.array([2.0]), "b": qb.array([3.0])})
    assert_close(blocks["a"], [[3.0]])
    assert_close(blocks["b"], [[2.0]])
    assert_raises(
        qb.UnsupportedOperationError, qb.grad(qb.hessian(loss)), point, match="cannot be transformed"
    )


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
    # NumPy data is accepted as an array leaf.
    if np is not None:
        assert_close(qb.jit(masked_loss)(np.array(QX), np.array(QW)), old)
    # jit of a plain function inside another trace traces through it.
    inner = qb.jit(lambda x: x * x)
    assert_close(qb.grad(lambda x: qb.sum(inner(x)))(x), [2.0 * v for v in QX])


def test_jit_devices_beyond_cpu_are_not_implemented_yet():
    error = assert_raises(qb.UnsupportedOperationError, qb.jit, masked_loss, device="mlx")
    assert (error.op, error.device) == ("jit", "mlx")
    assert isinstance(error, ValueError)
    assert_raises(qb.UnsupportedOperationError, qb.jit, masked_loss, device="cuda:1")
    assert_raises(ValueError, qb.jit, masked_loss, device="tpu", match="device must be")
    assert_raises(ValueError, qb.jit, masked_loss, max_traces=0, match="positive int")


def test_trace_cache_keys_and_retrace_limit():
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
    error = assert_raises(qb.RetraceLimitError, step, qb.array([1.0, 2.0]), 5.0)
    message = str(error)
    assert "max_traces=3" in message and "jit(" in message
    assert "(float64[2], 2.0)" in message and "(float32[2], 2.0)" in message
    assert "new signature:\n  (float64[2], 5.0)" in message
    assert isinstance(error, ValueError)
    # 1, 1.0, and True are different static values.
    counter = []
    typed = qb.jit(lambda x, s: counter.append(type(s)) or x * s)
    for scale in [1, 1.0, True, 1.0, math.nan, math.nan]:
        typed(qb.array(2.0), scale)
    assert counter == [int, float, bool, float]  # a NaN static value is cached like any other


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
    value32, derivative32 = qb.value_and_grad(lambda x: df(x))(qb.array(1.3, dtype=qb.float32))
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
        return sum(d * d for d in du) + sum(2.0 * b * math.tanh(a * x) for x in xs) / len(xs)

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
    assert_close(total, sum(-(1.5**2) * math.sin(1.5 * p) for p in [0.1, 0.2, 0.3]), 1e-14)
    assert len(traces) == 1  # three calls with one signature stage u once

    # jacobian and hessian cannot be staged until vmap exists.
    def loss(x):
        return qb.sum(x**3)

    error = assert_raises(
        qb.UnsupportedOperationError, qb.grad(qb.hessian(loss)), x, match="cannot be transformed"
    )
    assert error.op == "hessian" and isinstance(error, NotImplementedError)
    error = assert_raises(
        qb.UnsupportedOperationError,
        qb.jit(lambda x: qb.sum(qb.hessian(loss)(x))),
        x,
        match="called on traced values",
    )
    assert error.op == "hessian"
    assert_raises(
        qb.UnsupportedOperationError, qb.grad(lambda x: qb.sum(qb.jacobian(qb.sin)(x))), x
    )
    # Eager arguments and tangents of an inner call bind as constants (S3b);
    # vmap composition is slice S4.
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
        ValueError, qb.grad(lambda i, s: g(i, s), argnums=1), qb.array(0.4), qb.array(0.8),
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
    u = qb.trace_tensor(lambda x, w: (x * w).sin(), [("x", [PINN_POINTS, 1]), ("w", [1, 1])])
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
    old_value, old_grads = plan.evaluate_value_and_vjp(readme_inputs, qb.Tensor([], [1.0]))
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
        residuals = [(pinn_u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2 for x in xs]
        return qb.mean(qb.stack(residuals, 0))

    xs = qb.linspace(0.05, 0.95, PINN_POINTS).tolist()
    step = qb.jit(qb.value_and_grad(loss))
    value, grad_w = step(qb.array(2.5), xs)
    traced_value, traced_grad_w = qb.jit(qb.value_and_grad(pinn_loss))(qb.array(2.5), qb.array(xs))
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
                qb.grad(qb.grad(lambda x, w: qb.sin(x * w)))(xs[0], w)
                + math.pi**2 * qb.sin(math.pi * xs[0])
            )
            ** 2
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
    assert_close(qb.grad(lambda w: qb.grad(f)(w) * 2.0)(w), 2.0 * expected_second, 1e-13)
    assert_close(qb.hessian(f)(w), expected_second, 1e-13)

    # Like jax.jit, the array is baked in when the function is traced:
    # rebinding the closed-over name does not retrace.
    scale = qb.array([1.0, 2.0])
    scaled = qb.jit(lambda t: t * scale)
    x = qb.array([3.0, 4.0])
    assert scaled(x).tolist() == [3.0, 8.0]
    scale = qb.array([10.0, 20.0])  # noqa: F841 (read by the lambda on a retrace)
    assert scaled(x).tolist() == [3.0, 8.0]
    assert scaled(qb.array([1.0, 1.0, 1.0]).slice(0, 0, 2).to_tensor()).tolist() == [1.0, 2.0]


def test_captured_constants_keep_their_dtype():
    c32 = qb.array([0.5, -2.0], dtype=qb.float32)
    x32 = qb.array([1.5, 0.25], dtype=qb.float32)
    result = qb.jit(lambda t: t * c32 + 1.0)(x32)
    assert_tensor(result, (x32 * c32 + 1.0).tolist(), qb.float32)
    gradient = qb.grad(lambda t: qb.sum(qb.sin(t * c32)))(x32)
    assert gradient.dtype == qb.float32
    assert_close(gradient, [c * math.cos(x * c) for x, c in zip(x32.tolist(), c32.tolist())], 1e-6)
    # A float64 constant is strong: with a float32 tracer it is a dtype error
    # that asks for astype, as the same eager operation is.
    c64 = qb.array([0.5, -2.0])
    for function in [lambda t: t * c64, lambda t: c64 * t, lambda t: qb.maximum(t, c64)]:
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
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return
    assert_constant_device_parity(qb.tensor_value_and_grad_mlx_fn)


def test_cuda_captured_constants_match_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return
    assert_constant_device_parity(
        lambda loss, specs, names: qb.tensor_value_and_grad_cuda_fn(loss, specs, names, 0)
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
        qb.grad(function=model, input_specs=specs, values=values, output_cotangent=cotangent),
    ]:
        assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
        assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]

    @qb.jit([("a", (2, 2))])
    def double(a):
        return a + a

    assert type(double).__name__ == "JitFunction"
    assert double({"a": qb.Matrix([[1.0, 2.0], [3.0, 4.0]])}).to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert type(qb.jit(input_specs=[("a", (2, 2))])).__name__ == "JitTransform"
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
            (lambda t: np.asarray(t), "conversion to a NumPy array needs a concrete value"),
            (lambda t: np.sin(t), "conversion to a NumPy array needs a concrete value"),
        ]
    for function, message in escapes:
        error = assert_raises(qb.TracerError, qb.jit(function), x, match=message)
        assert isinstance(error, TypeError)
    # The v0.1 helpers raise the same class, which is still a TypeError.
    assert_raises(
        TypeError, qb.tensor_jit_fn, lambda t: t if t else t, [("x", [])], match="control flow"
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
        (lambda t: qb.einsum("ij,jk->ik", [matrix, t.reshape([2, 1])]).reshape([2]), [4.0, 7.0]),
        (lambda t: t * data.slice(0, 0, 2), [3.0, 8.0]),
        (lambda t: t + qb.sin(0.5), [1.0 + math.sin(0.5), 2.0 + math.sin(0.5)]),
    ]
    for function, expected in cases:
        assert_close(qb.jit(function)(x), expected, 1e-14)
    # The same operations differentiate through the constants.
    assert_close(qb.grad(lambda t: qb.sum(data * t * t))(x), [6.0, 16.0], 1e-14)
    assert_close(
        qb.grad(lambda t: qb.sum(qb.solve(matrix, t.reshape([2, 1]))))(x), [0.4, 0.2], 1e-14
    )
    # Other operand types keep their TypeError.
    error = assert_raises(TypeError, qb.jit(lambda t: t + "a"), x, match="numeric scalar operand")
    assert not isinstance(error, qb.TracerError)
    error = assert_raises(TypeError, lambda: data + "a", match="numeric scalar operand")
    assert not isinstance(error, qb.TracerError)
    assert_raises(TypeError, lambda: data @ "a")
    assert_raises(TypeError, lambda: data.matmul("a"), match="matmul expects a Tensor")


if __name__ == "__main__":
    # Run every test_* function in definition order so new tests cannot be left out of a manual list
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
