"""Packaging-facing helpers: `quabla.__version__`, value-showing `repr` of
eager tensors, the extension type stub, and `quabla.save` / `quabla.load`."""

import ast
import io
import json
import math
import pathlib
import re
import struct
import tempfile
import typing

import quabla as qb

from _support import raises, run, skip


def bits(tensor):
    """The exact bytes of a tensor, so NaN, -0.0, and float32 rounding
    compare bit for bit."""
    return memoryview(tensor).tobytes()


def same_tensor(a, b):
    return (
        type(a) is type(b) is qb.Tensor
        and a.dtype == b.dtype
        and a.shape == b.shape
        and bits(a) == bits(b)
    )


def test_version_matches_the_extension_and_the_cargo_manifest():
    assert isinstance(qb.__version__, str)
    assert qb.__version__ == qb._quabla.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", qb.__version__), qb.__version__
    manifest = pathlib.Path(__file__).parents[2] / "crates/quabla-python/Cargo.toml"
    if manifest.exists():
        declared = re.search(r'^version = "([^"]+)"', manifest.read_text(), re.M).group(1)
        assert qb.__version__ == declared
    # A star import must not rebind the importer's `__version__`.
    assert "__version__" not in qb.__all__
    assert "__version__" not in qb._quabla.__all__


def test_type_stub_declares_the_extension_names():
    package = pathlib.Path(qb.__file__).parent
    stub = package / "_quabla.pyi"
    assert (package / "py.typed").is_file()
    tree = ast.parse(stub.read_text())
    declared = {}
    exported = None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            declared[node.name] = node
        elif isinstance(node, ast.AnnAssign):
            declared[node.target.id] = node
        elif isinstance(node, ast.Assign) and node.targets[0].id == "__all__":
            exported = ast.literal_eval(node.value)
    assert exported == qb._quabla.__all__
    assert set(qb._quabla.__all__) <= set(declared), set(qb._quabla.__all__) - set(declared)
    # Every member the stub types on the array classes exists at runtime.
    for name in ("dtype", "Tensor", "TensorView", "TraceTensor"):
        runtime = getattr(qb._quabla, name)
        members = {
            member.name
            for member in declared[name].body
            if isinstance(member, ast.FunctionDef) and member.name != "__init__"
        }
        missing = {member for member in members if not hasattr(runtime, member)}
        assert not missing, (name, missing)


def test_eager_repr_shows_values_like_numpy():
    assert repr(qb.array([1.0, 2.0], dtype=qb.float32)) == "Tensor([1., 2.], dtype=float32)"
    assert repr(qb.array([1.5, 100.25])) == "Tensor([  1.5 , 100.25], dtype=float64)"
    assert repr(qb.array([0.1, 0.3], dtype=qb.float32)) == "Tensor([0.1, 0.3], dtype=float32)"
    assert repr(qb.array(2.5)) == "Tensor(2.5, dtype=float64)"
    assert repr(qb.array([True, False])) == "Tensor([ True, False], dtype=bool)"
    assert repr(qb.array([1e-5, 2.5e-7])) == "Tensor([1.0e-05, 2.5e-07], dtype=float64)"
    assert (
        repr(qb.array([-0.0, math.nan, -math.inf])) == "Tensor([ -0.,  nan, -inf], dtype=float64)"
    )
    assert repr(qb.arange(6.0).reshape([2, 3])) == (
        "Tensor([[0., 1., 2.],\n        [3., 4., 5.]], dtype=float64)"
    )


def test_large_tensor_repr_is_summarized():
    text = repr(qb.arange(1001.0))
    assert text == "Tensor([   0.,    1.,    2., ...,  998.,  999., 1000.], dtype=float64)"
    # Exactly the threshold is printed in full.
    assert "..." not in repr(qb.zeros((1000,)))
    lines = repr(qb.zeros((50, 50), dtype=qb.float32)).splitlines()
    assert len(lines) == 7 and lines[3] == "        ...,", lines
    assert all(len(line) <= 75 for line in lines)


def test_repr_formats_elements_like_numpy_when_available():
    try:
        import numpy as np
    except ImportError:
        skip("numpy is not installed")

    def elements(text):
        body = text[text.index("(") + 1 :]
        body = re.split(r",\s*(dtype|shape)=|\)$", body)[0]
        return [token for token in re.split(r"[\s,\[\]]+", body) if token]

    cases = [
        [0.1, 0.2, 0.3],
        [1.0 / 3.0, -2.0 / 3.0],
        [1.5, -100.25, 1e3],
        [0.0, 1999.0],
        [1e-5, 2.5e-7, -3.0],
        [1e-5, 1e30],
        [123456789.0, 1.0],
        [math.nan, math.inf, -0.0, 2.0],
        [[1.0, 2.5], [3.0, 4.0]],
        [[[1.0], [2.0]], [[3.0], [4.0]]],
        [True, False],
    ]
    for values in cases:
        for dtype in (qb.float64, qb.float32):
            expected = np.array(values, dtype=None if dtype == qb.float64 else np.float32)
            if expected.dtype == np.bool_:
                actual = qb.array(values)
            else:
                actual = qb.array(values, dtype=dtype)
            assert elements(repr(actual)) == elements(repr(expected)), (actual, expected)
    big = np.arange(3000.0).reshape(3, 1000) * 0.37
    assert elements(repr(qb.array(big))) == elements(repr(big))


def test_traced_repr_shows_shape_and_dtype_only():
    seen = []

    def f(x):
        seen.append(repr(x))
        return x * 2.0

    qb.jit(f)(qb.array([1.0, 2.0], dtype=qb.float32))
    assert re.fullmatch(r"TraceTensor\(node_id=\d+, shape=\[2\], dtype=float32\)", seen[0]), seen


def test_save_load_round_trips_arrays_scalars_and_containers():
    tree = {
        "w": qb.array([[0.1, -2.5], [math.nan, math.inf]], dtype=qb.float32),
        "b": qb.array([-0.0, 1.0 / 3.0, -math.inf, 5e-324]),
        "mask": qb.array([True, False, True]),
        "scalar_tensor": qb.array(7.0),
        "nested": [qb.array([1.0]), (None, 3, 2.5, True, "tanh"), {}],
        "float": 0.1,
        "inf": -math.inf,
        "big": 2**80,
        3: "int key",
    }
    buffer = io.BytesIO()
    qb.save(buffer, tree)
    loaded = qb.load(io.BytesIO(buffer.getvalue()))

    assert list(loaded) == list(tree)
    for key in ("w", "b", "mask", "scalar_tensor"):
        assert same_tensor(loaded[key], tree[key]), key
    assert loaded["scalar_tensor"].shape == []
    nested = loaded["nested"]
    assert type(nested) is list and type(nested[1]) is tuple and nested[2] == {}
    assert same_tensor(nested[0], tree["nested"][0])
    assert nested[1] == (None, 3, 2.5, True, "tanh")
    assert [type(value) for value in nested[1]] == [type(None), int, float, bool, str]
    assert loaded["float"] == 0.1 and loaded["inf"] == -math.inf and loaded["big"] == 2**80
    assert loaded[3] == "int key"
    assert qb.load(io.BytesIO(_saved(None))) is None
    assert math.isnan(qb.load(io.BytesIO(_saved(math.nan))))


def test_save_load_accepts_paths_and_views():
    view = qb.arange(6.0).reshape([2, 3]).slice(1, 0, 2)
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "params.qb"
        qb.save(path, {"view": view})
        loaded = qb.load(str(path))
    assert same_tensor(loaded["view"], view.to_tensor())


def test_optimizer_state_survives_a_round_trip():
    params = {"w": qb.array([1.0, -2.0]), "b": qb.array(0.5)}
    opt = qb.optim.Adam(1e-2)
    state = opt.init(params)
    grads = {"w": qb.array([0.3, -0.1]), "b": qb.array(1.0)}
    params, state = opt.update(params, grads, state)

    buffer = io.BytesIO()
    qb.save(buffer, {"params": params, "opt": state})
    restored = qb.load(io.BytesIO(buffer.getvalue()))
    assert type(restored["opt"]["step"]) is int and restored["opt"]["step"] == state["step"]

    expected = opt.update(params, grads, state)
    resumed = opt.update(restored["params"], grads, restored["opt"])
    for name in params:
        assert same_tensor(resumed[0][name], expected[0][name]), name


class Pair(typing.NamedTuple):
    a: float
    b: float


def test_save_rejects_values_it_cannot_round_trip():
    raises(TypeError, qb.save, io.BytesIO(), {"pair": Pair(1.0, 2.0)}, match="NamedTuples")
    raises(TypeError, qb.save, io.BytesIO(), {(1, 2): 1.0}, match="dict keys")
    raises(TypeError, qb.save, io.BytesIO(), [b"bytes"], match="bytes")
    raises(TypeError, qb.save, io.BytesIO(), {"key": qb.random.key(0)}, match="Key")

    def traced(x):
        qb.save(io.BytesIO(), x)
        return x

    raises(TypeError, qb.jit(traced), qb.array([1.0]), match="traced value")


def _saved(tree):
    buffer = io.BytesIO()
    qb.save(buffer, tree)
    return buffer.getvalue()


def _with_header(header, data=b""):
    encoded = json.dumps(header).encode()
    return b"\x93QUABLA\n" + struct.pack("<Q", len(encoded)) + encoded + data


def test_file_layout_is_magic_size_json_header_then_little_endian_data():
    content = _saved({"x": qb.array([1.0, 2.0], dtype=qb.float32)})
    assert content[:8] == b"\x93QUABLA\n"
    (size,) = struct.unpack("<Q", content[8:16])
    header = json.loads(content[16 : 16 + size])
    assert header == {
        "format": "quabla",
        "version": 1,
        "tree": {
            "type": "dict",
            "keys": ["x"],
            "values": [
                {"type": "array", "dtype": "float32", "shape": [2], "offset": 0, "nbytes": 8}
            ],
        },
    }
    assert content[16 + size :] == struct.pack("<2f", 1.0, 2.0)


def test_load_rejects_malformed_files():
    def bad(content, match):
        raises(ValueError, qb.load, io.BytesIO(content), match=match)

    good = _saved({"x": qb.array([1.0, 2.0])})
    bad(b"\x80\x04pickle", "bad magic")
    bad(good[:12], "incomplete header size")
    bad(good[:20], "incomplete header")
    bad(good[:-1], "past the end")
    bad(_with_header({"format": "quabla", "version": 2, "tree": {"type": "none"}}), "version 2")
    bad(_with_header({"format": "other", "version": 1}), "not a quabla header")
    bad(b"\x93QUABLA\n" + struct.pack("<Q", 3) + b"{x}", "malformed")
    bad(_with_header({"format": "quabla", "version": 1, "tree": {"type": "pickle"}}), "unknown")
    bad(_with_header({"format": "quabla", "version": 1, "tree": {"type": "int", "value": True}}), "bad int")
    array = {"type": "array", "dtype": "float64", "shape": [3], "offset": 0, "nbytes": 16}
    bad(_with_header({"format": "quabla", "version": 1, "tree": array}, bytes(16)), "do not hold")
    array = {"type": "array", "dtype": "int8", "shape": [1], "offset": 0, "nbytes": 1}
    bad(_with_header({"format": "quabla", "version": 1, "tree": array}, bytes(1)), "bad array")
    nan = b'{"format":"quabla","version":1,"tree":{"type":"float","value":NaN}}'
    bad(b"\x93QUABLA\n" + struct.pack("<Q", len(nan)) + nan, "NaN is not valid JSON")


if __name__ == "__main__":
    run(globals())
