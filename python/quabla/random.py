"""Keyed pseudo-random sampling (v0.3 plan item 6), in the style of
`jax.random`.

A `Key` is an explicit, immutable value: there is no global state, and the
same key always gives the same numbers, on every platform. Derive fresh keys
with `split` or `fold_in` and use each key once; sampling with a key after
splitting it reuses the stream its children were drawn from.

The generator is SplitMix64 (Steele, Lea, and Flood, 2014), shared with
`Tensor.split_key`, `Tensor.random_normal`, and `Tensor.glorot_normal`: a
key seeds a 64-bit Weyl sequence whose outputs are mixed by the SplitMix64
finalizer. `split(key, n)` returns the next `n` outputs of the stream seeded
by `key`. Uniform samples take the top 53 bits of one output each, giving
`[minval, maxval)` with endpoints clamped after rounding to the dtype.
Normal samples use the Box-Muller cosine branch on two outputs each, so
their magnitude is bounded by about 8.6 standard deviations. SplitMix64 is
a fast statistical generator, not a cryptographic one.

Sampling runs eagerly on the host and returns arrays. Inside a transformed
function, a sample drawn from a closure or static key is a constant of the
trace: every call of the compiled program sees the same numbers. Keys are
not traced, so draw samples outside and pass them in as arguments when they
must change between calls.
"""

import math
import operator

from ._quabla import Tensor, TraceTensor, bool_, float32, float64

__all__ = [
    "Key",
    "bernoulli",
    "fold_in",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "key",
    "normal",
    "split",
    "uniform",
]

_MASK = (1 << 64) - 1


class Key:
    """A random key: an immutable 64-bit generator seed. Create keys with
    `key(seed)`, `split`, or `fold_in`. Keys compare and hash by value, so
    they may be static arguments of `jit`."""

    __slots__ = ("_value",)

    def __init__(self, value):
        object.__setattr__(self, "_value", operator.index(value) & _MASK)

    def __setattr__(self, name, value):
        raise AttributeError("Key is immutable")

    @property
    def value(self):
        """The 64-bit seed as an int, as accepted by `Tensor.random_normal`."""
        return self._value

    def __eq__(self, other):
        return type(other) is Key and other._value == self._value

    def __hash__(self):
        return hash((Key, self._value))

    def __repr__(self):
        return f"Key({self._value:#018x})"

    def __reduce__(self):
        return (Key, (self._value,))


def _not_traced(value, what):
    if type(value) is TraceTensor:
        raise TypeError(
            f"{what} is a tracer: quabla.random samples on the host and keys are not "
            "traced, so draw samples outside the transformed function and pass them in "
            "as arguments"
        )


def _key_value(key, what="key"):
    _not_traced(key, what)
    if type(key) is not Key:
        raise TypeError(
            f"{what} must be a quabla.random.Key (from quabla.random.key(seed)), got "
            f"{type(key).__name__}"
        )
    return key._value


def _shape(shape):
    _not_traced(shape, "shape")
    if isinstance(shape, int):
        shape = (shape,)
    return [operator.index(extent) for extent in shape]


def _float_dtype(dtype, what):
    if dtype != float64 and dtype != float32:
        raise TypeError(f"{what} requires dtype float32 or float64, got {dtype!r}")
    return dtype


def key(seed):
    """A key from the integer `seed`, taken modulo 2**64. `key(seed)` and
    `Tensor.random_normal(shape, seed)` draw from the same stream."""
    _not_traced(seed, "seed")
    return Key(seed)


def split(key, num=2):
    """A tuple of `num` new keys derived from `key`."""
    num = operator.index(num)
    if num < 1:
        raise ValueError(f"split requires num >= 1, got {num}")
    return tuple(Key(value) for value in Tensor.split_key(_key_value(key), num))


def fold_in(key, data):
    """A new key derived from `key` and the integer `data` (taken modulo
    2**64), for example a step or device index."""
    _not_traced(data, "data")
    return Key(Tensor.fold_in_key(_key_value(key), operator.index(data) & _MASK))


def uniform(key, shape=(), dtype=float64, minval=0.0, maxval=1.0):
    """Samples uniform on `[minval, maxval)` with shape `shape` and dtype
    `dtype`; `minval` and `maxval` are finite Python numbers."""
    _not_traced(minval, "minval")
    _not_traced(maxval, "maxval")
    return Tensor.random_uniform(
        _shape(shape),
        _key_value(key),
        float(minval),
        float(maxval),
        _float_dtype(dtype, "uniform"),
    )


def normal(key, shape=(), dtype=float64):
    """Standard normal samples with shape `shape` and dtype `dtype`."""
    return _scaled_normal(key, shape, dtype, 1.0, "normal")


def bernoulli(key, p=0.5, shape=()):
    """`bool_` samples that are `True` with probability `p`, a Python number
    in `[0, 1]`: `uniform(key, shape) < p`."""
    _not_traced(p, "p")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"bernoulli requires 0 <= p <= 1, got {p}")
    samples = uniform(key, shape)
    return Tensor(samples.shape, [float(value < p) for value in samples.to_flat_list()], bool_)


def _scaled_normal(key, shape, dtype, stddev, what):
    samples = Tensor.random_normal(_shape(shape), _key_value(key), 0.0, stddev)
    if _float_dtype(dtype, what) == float32:
        samples = samples.astype(float32)
    return samples


def _fans(shape, what):
    """`(fan_in, fan_out)` of a weight of `shape`, as in JAX and Flax:
    `shape[-2]` and `shape[-1]` times the receptive field, the product of
    the leading dimensions."""
    shape = _shape(shape)
    if len(shape) < 2:
        raise ValueError(f"{what} requires a weight shape of rank >= 2, got {shape}")
    receptive = math.prod(shape[:-2])
    return shape[-2] * receptive, shape[-1] * receptive


def glorot_normal(key, shape, dtype=float64):
    """Glorot (Xavier) normal weights: standard deviation
    `sqrt(2 / (fan_in + fan_out))`, without truncation (JAX's
    `glorot_normal` truncates at two standard deviations). For a 2-D shape
    this equals `Tensor.glorot_normal(shape, key.value)`."""
    fan_in, fan_out = _fans(shape, "glorot_normal")
    stddev = math.sqrt(2.0 / (fan_in + fan_out))
    return _scaled_normal(key, shape, dtype, stddev, "glorot_normal")


def glorot_uniform(key, shape, dtype=float64):
    """Glorot (Xavier) uniform weights on `[-limit, limit)` with
    `limit = sqrt(6 / (fan_in + fan_out))`, as in JAX."""
    fan_in, fan_out = _fans(shape, "glorot_uniform")
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(key, shape, dtype, -limit, limit)


def he_normal(key, shape, dtype=float64):
    """He (Kaiming) normal weights: standard deviation `sqrt(2 / fan_in)`,
    without truncation (JAX's `he_normal` truncates at two standard
    deviations)."""
    fan_in, _ = _fans(shape, "he_normal")
    return _scaled_normal(key, shape, dtype, math.sqrt(2.0 / fan_in), "he_normal")
