"""Keyed sampling in `quabla.random`: determinism, bounds, dtypes, key
derivation, distribution statistics, initializers, and use with `jit`.

The statistical checks draw 100,000 samples from fixed keys, so they are
deterministic; their tolerances are five standard errors of each statistic
under the target distribution."""

import math
import pickle

import quabla as qb
from quabla import random

from _support import raises, run

N = 100_000


def moments(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def correlation(a, b):
    mean_a, var_a = moments(a)
    mean_b, var_b = moments(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / (len(a) - 1)
    return covariance / math.sqrt(var_a * var_b)


def test_keys_are_explicit_immutable_values():
    key = random.key(42)
    assert key == random.key(42) and hash(key) == hash(random.key(42))
    assert key != random.key(43) and key.value == 42
    assert random.key(-1).value == 2**64 - 1
    assert repr(key) == "Key(0x000000000000002a)"
    assert pickle.loads(pickle.dumps(key)) == key
    raises(AttributeError, setattr, key, "_value", 1)
    first, second = random.split(key)
    assert first != second and (first, second) == random.split(key)
    assert len(random.split(key, 5)) == 5 and len(set(random.split(key, 5))) == 5
    # split draws the stream of Tensor.split_key, so existing seeds carry over.
    assert [k.value for k in random.split(key, 3)] == qb.Tensor.split_key(42, 3)
    assert random.fold_in(key, 7) == random.fold_in(key, 7)
    assert random.fold_in(key, 7) != random.fold_in(key, 8)
    assert random.fold_in(key, 0) not in random.split(key, 4)
    raises(ValueError, random.split, key, 0)
    raises(TypeError, random.uniform, 42, (3,), match="quabla.random.Key")


def test_same_key_gives_the_same_numbers_on_every_platform():
    key = random.key(0)
    assert random.uniform(key, (4,)).tolist() == random.uniform(key, (4,)).tolist()
    # Golden values pin the generator: the top 53 bits of the first SplitMix64
    # outputs for seed 0, scaled by 2**-53 (checked against a pure-Python
    # SplitMix64).
    assert random.uniform(key, (3,)).tolist() == [
        0.8833108082136426,
        0.43152799704850997,
        0.026433771592597743,
    ]
    assert random.normal(random.key(42), (2, 3)).tolist() == (
        qb.Tensor.random_normal([2, 3], 42).tolist()
    )
    assert random.glorot_normal(random.key(7), (4, 5)).tolist() == (
        qb.Tensor.glorot_normal([4, 5], 7).tolist()
    )
    a, b = random.split(key)
    assert random.normal(a, (8,)).tolist() != random.normal(b, (8,)).tolist()


def test_shapes_and_dtypes():
    key = random.key(1)
    scalar = random.uniform(key)
    assert scalar.shape == [] and scalar.dtype == qb.float64
    assert random.uniform(key, 5).shape == [5]
    sample = random.normal(key, (2, 3), dtype=qb.float32)
    assert sample.shape == [2, 3] and sample.dtype == qb.float32
    assert random.uniform(key, (4,), qb.float32).dtype == qb.float32
    coins = random.bernoulli(key, 0.5, (6,))
    assert coins.dtype == qb.bool_ and coins.shape == [6]
    raises(TypeError, random.normal, key, (2,), qb.bool_, match="float32 or float64")
    raises(ValueError, random.bernoulli, key, 1.5)


def test_uniform_bounds_and_statistics():
    key = random.key(2026)
    values = random.uniform(key, (N,)).to_flat_list()
    assert min(values) >= 0.0 and max(values) < 1.0
    mean, variance = moments(values)
    # Standard errors: sqrt(1/12/N) for the mean, sqrt(1/180/N) for the variance.
    assert abs(mean - 0.5) < 5 * math.sqrt(1 / 12 / N), mean
    assert abs(variance - 1 / 12) < 5 * math.sqrt(1 / 180 / N), variance
    shifted = random.uniform(key, (N,), qb.float64, -3.0, 5.0).to_flat_list()
    assert min(shifted) >= -3.0 and max(shifted) < 5.0
    narrow = random.uniform(key, (N,), qb.float32, 0.0, 1.0).to_flat_list()
    assert min(narrow) >= 0.0 and max(narrow) < 1.0
    # Rounding to float32 would reach maxval; samples stay inside the interval.
    tight = random.uniform(key, (1000,), qb.float32, 1.0, 1.0 + 1e-12).to_flat_list()
    assert set(tight) == {1.0}
    tight64 = random.uniform(key, (1000,), qb.float64, 1.0, 1.0 + 2**-52).to_flat_list()
    assert set(tight64) == {1.0}
    raises(ValueError, random.uniform, key, (2,), qb.float32, 0.1, 0.1 + 1e-9, match="float32")
    raises(ValueError, random.uniform, key, (2,), qb.float64, 1.0, 1.0)
    raises(ValueError, random.uniform, key, (2,), qb.float64, -1e308, 1e308)


def test_normal_statistics():
    values = random.normal(random.key(11), (N,)).to_flat_list()
    mean, variance = moments(values)
    assert abs(mean) < 5 / math.sqrt(N), mean
    assert abs(variance - 1.0) < 5 * math.sqrt(2 / N), variance
    within = sum(abs(value) < 1.0 for value in values) / N
    expected = math.erf(1 / math.sqrt(2))
    assert abs(within - expected) < 5 * math.sqrt(expected * (1 - expected) / N), within
    narrow = random.normal(random.key(11), (N,), qb.float32).to_flat_list()
    assert abs(moments(narrow)[1] - 1.0) < 5 * math.sqrt(2 / N)


def test_bernoulli_statistics():
    values = random.bernoulli(random.key(5), 0.3, (N,)).to_flat_list()
    rate = sum(values) / N
    assert abs(rate - 0.3) < 5 * math.sqrt(0.3 * 0.7 / N), rate
    assert set(random.bernoulli(random.key(5), 0.0, (100,)).to_flat_list()) == {0.0}
    assert set(random.bernoulli(random.key(5), 1.0, (100,)).to_flat_list()) == {1.0}


def test_split_and_folded_keys_are_independent():
    key = random.key(99)
    first, second = random.split(key)
    a = random.normal(first, (N,)).to_flat_list()
    b = random.normal(second, (N,)).to_flat_list()
    c = random.normal(random.fold_in(key, 1), (N,)).to_flat_list()
    bound = 5 / math.sqrt(N)
    assert abs(correlation(a, b)) < bound
    assert abs(correlation(a, c)) < bound
    assert abs(correlation(b, c)) < bound
    # Adjacent samples of one stream are uncorrelated too.
    assert abs(correlation(a[:-1], a[1:])) < bound


def test_initializers_use_jax_fans():
    key = random.key(3)
    weights = random.glorot_normal(key, (200, 300)).to_flat_list()
    expected = math.sqrt(2 / 500)
    assert abs(math.sqrt(moments(weights)[1]) / expected - 1) < 5 / math.sqrt(2 * len(weights))
    limit = math.sqrt(6 / 500)
    uniform = random.glorot_uniform(key, (200, 300), qb.float32)
    assert uniform.dtype == qb.float32
    values = uniform.to_flat_list()
    assert -limit <= min(values) and max(values) < limit
    he = random.he_normal(key, (200, 300)).to_flat_list()
    expected = math.sqrt(2 / 200)
    assert abs(math.sqrt(moments(he)[1]) / expected - 1) < 5 / math.sqrt(2 * len(he))
    # Convolution kernels: fan_in = 3 * 3 * 16 = 144, fan_out = 3 * 3 * 32 = 288.
    kernel = random.glorot_uniform(key, (3, 3, 16, 32)).to_flat_list()
    assert max(abs(value) for value in kernel) < math.sqrt(6 / (144 + 288))
    raises(ValueError, random.glorot_normal, key, (5,), match="rank >= 2")


def test_keys_are_host_values_under_transforms():
    key = random.key(8)
    noise = random.normal(key, (2,))
    x = qb.array([1.0, 2.0])
    # A sample drawn from a closure key is a constant of the trace.
    jitted = qb.jit(lambda x: x + random.normal(key, (2,)))
    assert jitted(x).tolist() == jitted(x).tolist() == (x + noise).tolist()
    # A key may be static; it is part of the cache key.
    static = qb.jit(lambda x, k: x + random.normal(k, (2,)), static_argnums=1)
    assert static(x, key).tolist() == (x + noise).tolist()
    assert static(x, random.key(9)).tolist() != static(x, key).tolist()
    raises(TypeError, qb.jit(lambda x, k: x), x, key, match="cannot be a traced argument")
    raises(TypeError, qb.jit(lambda s: random.key(s)), qb.array(1.0), match="tracer")
    raises(
        TypeError,
        qb.grad(lambda x: qb.sum(random.uniform(key, (2,), minval=x))),
        qb.array(0.0),
        match="tracer",
    )


if __name__ == "__main__":
    run(globals())
