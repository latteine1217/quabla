"""Bounded, deterministic property tests of shapes, indexing and tracing.

The scalar/list oracles use only the standard library, so every built wheel
runs these tests without optional NumPy or a fuzzing dependency. Failures
include the seed and generated case for direct replay.
"""

import itertools
import math
import random

import quabla as qb

from _support import raises, run

SEEDS = (103, 1701, 65537, 8675309)


def assert_numeric(actual, expected, seed, case):
    assert len(actual) == len(expected), (seed, case, actual, expected)
    assert all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(actual, expected)
    ), (seed, case, actual, expected)


def coordinates(shape):
    return itertools.product(*(range(extent) for extent in shape))


def offset(coordinate, shape):
    return sum(
        index * math.prod(shape[axis + 1 :]) for axis, index in enumerate(coordinate)
    )


def broadcast_value(values, shape, coordinate):
    local = coordinate[len(coordinate) - len(shape) :] if shape else ()
    local = tuple(0 if size == 1 else index for size, index in zip(shape, local))
    return values[offset(local, shape)]


def test_broadcast_shape_inference_and_values_match_scalar_coordinates():
    for seed in SEEDS:
        rng = random.Random(seed)
        for case in range(32):
            target = [rng.randint(1, 4) for _ in range(rng.randint(0, 4))]
            shapes = []
            values = []
            for _ in range(2):
                rank = rng.randint(0, len(target))
                shape = (
                    [rng.choice((1, extent)) for extent in target[-rank:]]
                    if rank
                    else []
                )
                shapes.append(shape)
                values.append(
                    [float(rng.randint(-8, 8)) for _ in range(math.prod(shape))]
                )
            rank = max(map(len, shapes))
            padded = [[1] * (rank - len(shape)) + shape for shape in shapes]
            expected_shape = [max(pair) for pair in zip(*padded)]
            expected = [
                broadcast_value(values[0], shapes[0], coordinate)
                + broadcast_value(values[1], shapes[1], coordinate)
                for coordinate in coordinates(expected_shape)
            ]
            left, right = [
                qb.Tensor(shape, data) for shape, data in zip(shapes, values)
            ]
            for output in (left + right, qb.jit(lambda a, b: a + b)(left, right)):
                assert output.shape == expected_shape, (seed, case, shapes)
                assert output.to_flat_list() == expected, (seed, case, shapes)


def test_indexing_and_transpose_match_cartesian_coordinates():
    for seed in SEEDS:
        rng = random.Random(seed)
        for case in range(32):
            shape = [rng.randint(1, 4) for _ in range(rng.randint(1, 4))]
            values = [float(i) for i in range(math.prod(shape))]
            tensor = qb.Tensor(shape, values)
            axis = rng.randrange(len(shape))
            extent = shape[axis]
            kind = rng.randrange(4)
            selections = [list(range(size)) for size in shape]
            key = [slice(None)] * len(shape)
            expected_shape = list(shape)
            if kind == 0:
                index = rng.randrange(-extent, extent)
                key[axis] = index
                selections[axis] = [index % extent]
                expected_shape.pop(axis)
            elif kind == 1:
                selected = [
                    rng.randrange(-extent, extent) for _ in range(rng.randint(1, 5))
                ]
                key[axis] = selected
                selections[axis] = [index % extent for index in selected]
                expected_shape[axis] = len(selected)
            else:
                selected = (
                    slice(None, None, -1)
                    if kind == 2
                    else slice(rng.randrange(extent), None, rng.randint(1, 3))
                )
                key[axis] = selected
                selections[axis] = list(range(extent))[selected]
                expected_shape[axis] = len(selections[axis])
            key = tuple(key)
            expected = [
                values[offset(c, shape)] for c in itertools.product(*selections)
            ]

            def select(x):
                if kind == 1:
                    return x.gather(key[axis], axis=axis)
                return x[key]

            for output in (select(tensor), qb.jit(select)(tensor)):
                assert output.shape == expected_shape, (seed, case, shape, key)
                assert output.to_flat_list() == expected, (seed, case, shape, key)
            permutation = rng.sample(range(len(shape)), len(shape))
            new_shape = [shape[axis] for axis in permutation]
            expected = []
            for coordinate in coordinates(new_shape):
                original = [0] * len(shape)
                for axis, index in zip(permutation, coordinate):
                    original[axis] = index
                expected.append(values[offset(original, shape)])
            for output in (
                tensor.transpose(permutation),
                qb.jit(lambda x: x.transpose(permutation))(tensor),
            ):
                assert output.shape == new_shape, (seed, case, permutation)
                assert output.to_flat_list() == expected, (seed, case, permutation)


def test_generated_traces_and_gradients_match_scalar_oracles():
    for seed in SEEDS:
        rng = random.Random(seed)
        for case in range(24):
            values = [rng.randint(-8, 8) / 16 for _ in range(rng.randint(1, 16))]
            scale, shift = rng.randint(1, 8) / 8, rng.randint(-4, 4) / 8
            operations = [rng.choice(("sin", "tanh", "shift")) for _ in range(12)]

            def model(x):
                for operation in operations:
                    if operation == "sin":
                        x = x.sin()
                    elif operation == "tanh":
                        x = x.tanh()
                    else:
                        x = x * scale + shift
                return x

            expected = []
            for value in values:
                for operation in operations:
                    value = (
                        math.sin(value)
                        if operation == "sin"
                        else math.tanh(value)
                        if operation == "tanh"
                        else value * scale + shift
                    )
                expected.append(value)
            point = qb.array(values)
            assert_numeric(qb.jit(model)(point).to_flat_list(), expected, seed, case)

            def loss(x):
                return ((x * scale + shift) ** 2).sum()

            expected_gradient = [
                2 * scale * (value * scale + shift) for value in values
            ]
            for gradient in (qb.grad(loss)(point), qb.jit(qb.grad(loss))(point)):
                assert_numeric(gradient.to_flat_list(), expected_gradient, seed, case)
            # Invalid tracing must release its trace frame before the next transform.
            raises(ValueError, qb.jit(lambda x: x.reshape([len(values) + 1])), point)
            assert_numeric(qb.jit(model)(point).to_flat_list(), expected, seed, case)


def test_nested_roundtrips_and_invalid_indices_preserve_tensor_state():
    for seed in SEEDS:
        rng = random.Random(seed)
        for case in range(32):
            shape = [rng.randint(1, 4) for _ in range(rng.randint(1, 4))]
            dtype = rng.choice((qb.float64, qb.float32, qb.bool_))
            values = [rng.randint(-8, 8) / 8 for _ in range(math.prod(shape))]
            tensor = qb.Tensor(shape, values, dtype=dtype)
            original = tensor.to_flat_list()
            roundtrip = qb.asarray(tensor.tolist(), dtype=dtype)
            assert roundtrip.shape == shape, (seed, case, shape)
            assert roundtrip.dtype == dtype, (seed, case, dtype)
            assert roundtrip.to_flat_list() == original, (seed, case, values)
            axis = rng.randrange(len(shape))
            invalid = rng.choice((shape[axis], -shape[axis] - 1))
            key = [slice(None)] * len(shape)
            key[axis] = invalid
            key = tuple(key)
            raises(IndexError, lambda: tensor[key])
            raises(IndexError, qb.jit(lambda x: x[key]), tensor)
            assert tensor.to_flat_list() == original, (seed, case, key)
            assert qb.jit(lambda x: x.reshape([-1]))(tensor).to_flat_list() == original


if __name__ == "__main__":
    run(globals())
