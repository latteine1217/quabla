"""Saving and loading pytrees of arrays (`quabla.save` / `quabla.load`).

The file format is self-contained and needs neither NumPy nor pickle, so
loading a file never executes code from it. Layout (version 1):

    offset 0   magic        8 bytes, b"\\x93QUABLA\\n"
    offset 8   header size  unsigned 64-bit little-endian integer N
    offset 16  header       N bytes of UTF-8 JSON (strict: no NaN tokens)
    offset 16+N data        the raw bytes of every array, concatenated

The header is `{"format": "quabla", "version": 1, "tree": NODE}`, where a
NODE is one of

    {"type": "none"}
    {"type": "list",  "items": [NODE, ...]}
    {"type": "tuple", "items": [NODE, ...]}
    {"type": "dict",  "keys": [KEY, ...], "values": [NODE, ...]}
    {"type": "bool",  "value": true}
    {"type": "int",   "value": 3}
    {"type": "float", "value": "0x1.8000000000000p+0"}   (float.hex, exact)
    {"type": "str",   "value": "tanh"}
    {"type": "array", "dtype": "float32", "shape": [2, 3],
                      "offset": 0, "nbytes": 24}

dict keys are JSON strings or integers and keep their insertion order.
Array data is C-ordered and little-endian: float64 and float32 as IEEE
binary64/binary32, bool as one byte (0 or 1) per element; `offset` is
relative to the start of the data section.

Python scalars round-trip exactly and keep their type, so an optimizer state
such as `{"step": 3, "m": ..., "v": ...}` loads back unchanged. NamedTuples
and classes registered with `quabla.tree.register`/`register_dataclass` are
not serialized: rebuilding them would mean importing classes named by the
file. Save their fields (for example as a dict) instead.
"""

import array as _array
import json
import os
import struct
import sys

from ._quabla import Tensor, TensorView, TraceTensor

__all__ = ["load", "save"]

MAGIC = b"\x93QUABLA\n"
VERSION = 1
_HEADER_SIZE = struct.Struct("<Q")
# dtype name -> (bytes per element, memoryview/array format).
_DTYPES = {"float64": (8, "d"), "float32": (4, "f"), "bool": (1, "?")}
_BIG_ENDIAN = sys.byteorder == "big"


def save(file, tree):
    """Writes the pytree `tree` to `file`, a path or a binary file object.

    Leaves may be quabla arrays (`Tensor` or `TensorView`; float64, float32,
    and bool keep their dtype and bits), `None`, and Python `bool`, `int`,
    `float`, and `str` values. Containers may be `dict` (string or integer
    keys), `list`, and `tuple`. Anything else, including traced values,
    NamedTuples, and registered pytree classes, raises `TypeError`. See the
    module docstring for the format; `load` reads it back.
    """
    chunks = []
    header = {"format": "quabla", "version": VERSION, "tree": _encode(tree, chunks, [0])}
    header = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    payload = [MAGIC, _HEADER_SIZE.pack(len(header)), header, *chunks]
    if hasattr(file, "write"):
        for chunk in payload:
            file.write(chunk)
    else:
        with open(os.fspath(file), "wb") as handle:
            for chunk in payload:
                handle.write(chunk)


def load(file):
    """Reads a pytree written by `save` from `file`, a path or a binary file
    object. Arrays come back as `Tensor`s with their saved dtype and shape,
    tuples as tuples, and dicts in their saved key order. A malformed or
    truncated file raises `ValueError`."""
    if hasattr(file, "read"):
        content = file.read()
    else:
        with open(os.fspath(file), "rb") as handle:
            content = handle.read()
    content = memoryview(content)
    prefix = len(MAGIC) + _HEADER_SIZE.size
    if bytes(content[: len(MAGIC)]) != MAGIC:
        raise ValueError("not a quabla save file (bad magic bytes)")
    if len(content) < prefix:
        raise ValueError("truncated quabla save file (incomplete header size)")
    (size,) = _HEADER_SIZE.unpack(content[len(MAGIC) : prefix])
    if size > len(content) - prefix:
        raise ValueError("truncated quabla save file (incomplete header)")
    try:
        header = json.loads(bytes(content[prefix : prefix + size]), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed quabla save file header: {error}") from None
    if not isinstance(header, dict) or header.get("format") != "quabla":
        raise ValueError("malformed quabla save file header: not a quabla header")
    if header.get("version") != VERSION:
        raise ValueError(
            f"unsupported quabla save file version {header.get('version')!r}; this quabla "
            f"reads version {VERSION}"
        )
    return _decode(header.get("tree"), content[prefix + size :])


def _reject_constant(name):
    raise ValueError(f"malformed quabla save file header: {name} is not valid JSON")


def _encode(value, chunks, offset):
    kind = type(value)
    if value is None:
        return {"type": "none"}
    if kind is bool or kind is int or kind is str:
        return {"type": kind.__name__, "value": value}
    if kind is float:
        return {"type": "float", "value": value.hex()}
    if kind is list or kind is tuple:
        return {"type": kind.__name__, "items": [_encode(item, chunks, offset) for item in value]}
    if kind is dict:
        for key in value:
            if type(key) is not str and type(key) is not int:
                raise TypeError(
                    f"quabla.save supports dict keys of type str or int, got {key!r}"
                )
        return {
            "type": "dict",
            "keys": list(value),
            "values": [_encode(item, chunks, offset) for item in value.values()],
        }
    if kind is TensorView:
        value = value.to_tensor()
        kind = Tensor
    if kind is Tensor:
        data = _array_bytes(value)
        node = {
            "type": "array",
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "offset": offset[0],
            "nbytes": len(data),
        }
        chunks.append(data)
        offset[0] += len(data)
        return node
    if kind is TraceTensor:
        raise TypeError("quabla.save cannot save a traced value; call it outside a transform")
    raise TypeError(
        f"quabla.save cannot save a {kind.__qualname__}: leaves must be quabla arrays, None, "
        "bool, int, float, or str, and containers dict, list, or tuple (NamedTuples and "
        "registered pytree classes are not serialized; save their fields instead)"
    )


def _array_bytes(tensor):
    """The C-ordered little-endian bytes of `tensor`."""
    data = memoryview(tensor).tobytes()
    width, code = _DTYPES[tensor.dtype.name]
    if _BIG_ENDIAN and width > 1:
        values = _array.array(code, data)
        values.byteswap()
        data = values.tobytes()
    return data


def _decode(node, data):
    if not isinstance(node, dict):
        raise ValueError(f"malformed quabla save file header: bad tree node {node!r}")
    kind = node.get("type")
    if kind == "none":
        return None
    if kind in ("bool", "int", "str"):
        value = node.get("value")
        if type(value).__name__ != kind:
            raise ValueError(f"malformed quabla save file header: bad {kind} node {node!r}")
        return value
    if kind == "float":
        try:
            return float.fromhex(node["value"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"malformed quabla save file header: bad float node {node!r}") from None
    if kind in ("list", "tuple"):
        items = node.get("items")
        if type(items) is not list:
            raise ValueError(f"malformed quabla save file header: bad {kind} node")
        items = [_decode(item, data) for item in items]
        return items if kind == "list" else tuple(items)
    if kind == "dict":
        keys, values = node.get("keys"), node.get("values")
        if (
            type(keys) is not list
            or type(values) is not list
            or len(keys) != len(values)
            or any(type(key) is not str and type(key) is not int for key in keys)
        ):
            raise ValueError("malformed quabla save file header: bad dict node")
        return {key: _decode(value, data) for key, value in zip(keys, values)}
    if kind == "array":
        return _decode_array(node, data)
    raise ValueError(f"malformed quabla save file header: unknown node type {kind!r}")


def _decode_array(node, data):
    dtype, shape = node.get("dtype"), node.get("shape")
    offset, nbytes = node.get("offset"), node.get("nbytes")
    if (
        dtype not in _DTYPES
        or type(shape) is not list
        or any(type(extent) is not int or extent < 0 for extent in shape)
        or type(offset) is not int
        or type(nbytes) is not int
        or offset < 0
    ):
        raise ValueError(f"malformed quabla save file header: bad array node {node!r}")
    width, code = _DTYPES[dtype]
    count = 1
    for extent in shape:
        count *= extent
    if nbytes != count * width:
        raise ValueError(
            f"malformed quabla save file header: {nbytes} bytes do not hold a {dtype} array "
            f"of shape {shape}"
        )
    if offset + nbytes > len(data):
        raise ValueError("truncated quabla save file (array data past the end of the file)")
    raw = data[offset : offset + nbytes]
    if width == 1:
        flat = raw.cast(code)
    else:
        # Copied into an `array`, whose storage is aligned for its element
        # type; the slice of the file content need not be.
        flat = _array.array(code)
        flat.frombytes(raw)
        if _BIG_ENDIAN:
            flat.byteswap()
    # The buffer import copies the 1-D data into a new tensor, keeping the
    # dtype; the reshape restores the saved shape.
    return Tensor._from_buffer(flat, None).reshape(shape)
