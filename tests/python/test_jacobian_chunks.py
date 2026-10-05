"""Analytic dense derivatives across basis batches and pytree boundaries."""

import math
from unittest.mock import patch

import quabla as qb


def assert_close(actual, expected, tolerance=1e-12):
    expected = qb.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    for value, reference in zip(actual.to_flat_list(), expected.to_flat_list()):
        assert math.isclose(value, reference, rel_tol=tolerance, abs_tol=tolerance), (
            value,
            reference,
        )


def test_jacobian_pytree_leaf_and_batch_boundaries():
    x = [float(index) / 16 for index in range(63)]
    y = [0.5, -0.25, 2.0, -1.0]
    params = {
        "mask": qb.array([True, False]),
        "x": qb.array(x, dtype=qb.float32),
        "y": qb.array(y),
    }

    def model(p):
        x64 = p["x"].astype(qb.float64)
        return {
            "loss": qb.sum(x64**3) + qb.sum(p["y"] ** 2),
            "vector": x64[:2] * p["y"][:2],
            "fixed": qb.array([2.0, 3.0], dtype=qb.float32),
            "mask": p["mask"],
        }

    for transform in (qb.jacobian(model), qb.jit(qb.jacobian(model))):
        result = transform(params)
        assert_close(result["loss"]["x"], [3 * value**2 for value in x])
        assert_close(result["loss"]["y"], [2 * value for value in y])
        assert_close(
            result["vector"]["x"],
            [
                [y[row] if row == column else 0.0 for column in range(63)]
                for row in range(2)
            ],
        )
        assert_close(
            result["vector"]["y"],
            [
                [x[row] if row == column else 0.0 for column in range(4)]
                for row in range(2)
            ],
        )
        assert result["fixed"]["x"].dtype == qb.float32
        assert_close(result["fixed"]["x"], qb.zeros([2, 63], qb.float32))
        assert_close(result["mask"]["y"], qb.zeros([2, 4]))
        for blocks in result.values():
            assert blocks["mask"] is None


def test_jacobian_multiple_argnums():
    x = qb.array([float(index) / 16 for index in range(65)])
    y = qb.array(2.0)
    dx, dy = qb.jacobian(lambda x, y: qb.sum(x * y), argnums=(0, 1))(x, y)
    assert_close(dx, qb.full([65], 2.0))
    assert_close(dy, sum(x.to_flat_list()))


def test_float32_jacobian_before_at_and_after_batch_boundaries():
    for size in (63, 64, 65, 129):
        values = [float(index % 17 - 8) / 16 for index in range(size)]
        point = qb.array(values, dtype=qb.float32)
        result = qb.jacobian(lambda x: qb.sum(qb.sin(x) * x))(point)
        assert result.dtype == qb.float32
        assert_close(
            result,
            [math.sin(value) + value * math.cos(value) for value in values],
            1e-6,
        )


def test_hessian_and_nested_products_across_basis_batches():
    values = [float(index - 32) / 16 for index in range(65)]
    point = qb.array(values)
    direction_values = [float(index % 5 - 2) for index in range(65)]
    direction = qb.array(direction_values)

    def cubic(x):
        return qb.sum(x**3) + x[63] * x[64]

    expected = [
        [
            6 * values[row]
            if row == column
            else float((row, column) in ((63, 64), (64, 63)))
            for column in range(65)
        ]
        for row in range(65)
    ]
    assert_close(qb.jit(qb.hessian(cubic))(point), expected)
    assert_close(
        qb.grad(lambda x: qb.sum(qb.hessian(cubic)(x)))(point), qb.full([65], 6.0)
    )
    jacobian = qb.jacobian(lambda x: qb.sum(x**3))
    primal, tangent = qb.jvp(jacobian, (point,), (direction,))
    assert_close(primal, [3 * value**2 for value in values])
    expected_product = [6 * value * d for value, d in zip(values, direction_values)]
    assert_close(tangent, expected_product)
    _, pullback = qb.vjp(jacobian, point)
    assert_close(pullback(direction)[0], expected_product)
    assert_close(
        qb.vmap(jacobian)(qb.stack([point, point * 2.0])),
        [[3 * value**2 for value in values], [12 * value**2 for value in values]],
    )


def test_chunked_jacobian_captures_different_outer_tracers():
    point = qb.array([float(index) / 32 for index in range(65)])

    def outer(weight):
        derivative = qb.jacobian(lambda x: qb.sum(x * x * weight))(point)
        return derivative.sum()

    derivative = qb.jit(qb.grad(outer))
    for weight in (qb.ones([65]), qb.full([65], 3.0)):
        assert_close(derivative(weight), point * 2.0)


def test_radix_coordinates_preserve_identity_across_multiple_digits():
    # A small radix exercises the large-index backend path without allocating
    # millions of elements. Multiple digits must identify every basis row.
    values = [float(index - 32) / 16 for index in range(65)]
    point = qb.array(values, dtype=qb.float32)
    with patch("quabla._transforms._JACOBIAN_INDEX_RADIX", 8):
        result = qb.jit(qb.jacobian(lambda x: x * x))(point)
        derivative = qb.grad(lambda x: qb.sum(qb.jacobian(lambda y: y * y)(x)))(point)
    assert result.dtype == qb.float32
    assert_close(
        result,
        [
            [2 * values[row] if row == column else 0.0 for column in range(65)]
            for row in range(65)
        ],
    )
    assert_close(derivative, qb.full([65], 2.0, qb.float32))


def test_adaptive_jacobian_output_batches_casts_and_unused_leaves():
    params = {
        "x": qb.array([index / 16 for index in range(129)], dtype=qb.float32),
        "y": qb.array([0.25, -0.5, 1.0, 2.0, -2.0]),
        "unused": qb.ones([3]),
    }

    def model(p):
        x = p["x"].astype(qb.float64)
        return {"vector": x[:65] ** 2, "loss": (p["y"] ** 3).sum()}

    # Outputs cross a seed batch boundary while input/output casts and
    # disconnected leaves retain the public output-first block contract.
    result = qb.jit(qb.jacobian(model))(params)
    assert result["vector"]["x"].dtype == qb.float64
    assert_close(
        result["vector"]["x"],
        [
            [row / 8 if row == column else 0.0 for column in range(129)]
            for row in range(65)
        ],
    )
    assert_close(result["loss"]["y"], [0.1875, 0.75, 3.0, 12.0, 12.0])
    assert_close(result["vector"]["y"], qb.zeros([65, 5]))
    assert_close(result["loss"]["unused"], qb.zeros([3]))
    params["x"] = params["x"].astype(qb.float64)
    reverse_result = qb.jacobian(model)(params)
    for output in result:
        for leaf in params:
            assert_close(reverse_result[output][leaf], result[output][leaf])


def test_adaptive_jacobian_nonlinear_rounding_contract():
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        values = [index / 16 for index in range(8)]
        point = qb.array(values, dtype=dtype)
        actual = qb.jacobian(lambda x: (x * x).sin().tanh().sum())(point)
        expected = [
            2
            * value
            * math.cos(value * value)
            * (1 - math.tanh(math.sin(value * value)) ** 2)
            for value in values
        ]
        assert actual.dtype == dtype
        for value, reference in zip(actual.to_flat_list(), expected):
            assert abs(value - reference) <= tolerance + tolerance * abs(reference)


def test_adaptive_jacobian_accepts_arbitrary_pytree_key_paths():
    def model(__quabla_cotangent):
        return (__quabla_cotangent["jacobian/0"] ** 2).sum()

    point = qb.array([index / 16 for index in range(8)])
    result = qb.jit(qb.jacobian(model))({"jacobian/0": point})
    assert_close(result["jacobian/0"], point * 2.0)


def test_jacobian_widening_preserves_output_precision():
    point = qb.array([index / 10 for index in range(8)], dtype=qb.float32)
    result = qb.jacobian(lambda x: x.astype(qb.float64).sin().sum())(point)
    assert result.dtype == qb.float64
    assert_close(result, [math.cos(value) for value in point.to_flat_list()])

    def vector(x):
        narrow = x.astype(qb.float32)
        return (narrow * narrow).sin().tanh().astype(qb.float64)

    wide_point = point.astype(qb.float64)
    forward_reference = qb.jacobian(vector)(wide_point).sum(axis=0)
    scalar_result = qb.jacobian(lambda x: vector(x).sum())(wide_point)
    assert_close(scalar_result, forward_reference)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
