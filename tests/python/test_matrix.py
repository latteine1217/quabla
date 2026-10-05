import array
import math
import os
import runpy

import quabla


def assert_close_rows(actual, expected, tol=1e-12):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert abs(actual_value - expected_value) <= tol


def test_matrix_matmul():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = quabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    c = a @ b

    assert c.shape == (2, 2)
    assert c.to_list() == [[58.0, 64.0], [139.0, 154.0]]


def test_tensor_solve_and_jit_vjp():
    matrix = quabla.Tensor([2, 2], [3.0, 1.0, 1.0, 2.0])
    rhs = quabla.Tensor([2, 1], [9.0, 8.0])
    assert matrix.solve(rhs).to_flat_list() == [2.0, 3.0]

    value_and_grad = quabla.tensor_value_and_grad_fn(
        lambda a, b: a.solve(b).sum(), [("a", [2, 2]), ("b", [2, 1])]
    )
    value, gradients = value_and_grad({"a": matrix, "b": rhs})
    assert value.to_flat_list() == [5.0]
    assert_close_rows([gradients["a"].to_flat_list()], [[-0.4, -0.6, -0.8, -1.2]])
    assert_close_rows([gradients["b"].to_flat_list()], [[0.2, 0.4]])


def test_tensor_neural_primitives_and_trace_gradients():
    values = quabla.Tensor([3], [-1.0, 0.0, 1.0])
    assert values.relu().to_flat_list() == [0.0, 0.0, 1.0]
    assert values.abs().to_flat_list() == [1.0, -0.0, 1.0]

    sigmoid = values.sigmoid().to_flat_list()
    softplus = values.softplus().to_flat_list()
    expected_sigmoid = [1.0 / (1.0 + math.exp(-value)) for value in [-1.0, 0.0, 1.0]]
    assert_close_rows([sigmoid], [expected_sigmoid])
    assert_close_rows(
        [softplus], [[math.log1p(math.exp(value)) for value in [-1.0, 0.0, 1.0]]]
    )
    assert quabla.Tensor([2], [-1000.0, 1000.0]).softplus().to_flat_list() == [0.0, 1000.0]

    for name, expected_gradient in [
        ("relu", [0.0, 0.0, 1.0]),
        ("abs", [-1.0, -1.0, 1.0]),
        ("sigmoid", [value * (1.0 - value) for value in expected_sigmoid]),
        ("softplus", expected_sigmoid),
    ]:
        transform = quabla.tensor_value_and_grad_fn(
            lambda x, name=name: getattr(x, name)().sum(), [("x", [3])]
        )
        _, gradients = transform({"x": values})
        assert_close_rows([gradients["x"].to_flat_list()], [expected_gradient])


def test_tensor_triangular_projections_preserve_trace_gradients():
    values = quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert values.tril().to_flat_list() == [1.0, 0.0, 0.0, 4.0, 5.0, 0.0]
    assert values.triu().to_flat_list() == [1.0, 2.0, 3.0, 0.0, 5.0, 6.0]

    for name, expected_gradient in [
        ("tril", [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        ("triu", [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
    ]:
        transform = quabla.tensor_value_and_grad_fn(
            lambda x, name=name: getattr(x, name)().sum(), [("x", [2, 3])]
        )
        _, gradients = transform({"x": values})
        assert gradients["x"].to_flat_list() == expected_gradient


def test_tensor_triangular_solve_preserves_vjp_and_transpose_contract():
    matrix = quabla.Tensor([2, 2], [2.0, 9.0, 3.0, 4.0])
    rhs = quabla.Tensor([2, 1], [2.0, 11.0])
    assert matrix.solve_triangular(rhs).to_flat_list() == [1.0, 2.0]
    assert matrix.solve_triangular(
        quabla.Tensor([2, 1], [8.0, 8.0]), transpose=True
    ).to_flat_list() == [1.0, 2.0]

    transform = quabla.tensor_value_and_grad_fn(
        lambda a, b: a.solve_triangular(b).sum(), [("a", [2, 2]), ("b", [2, 1])]
    )
    _, gradients = transform({"a": matrix, "b": rhs})
    assert_close_rows([gradients["a"].to_flat_list()], [[-0.125, 0.0, -0.25, -0.5]])
    assert_close_rows([gradients["b"].to_flat_list()], [[0.125, 0.25]])


def test_tensor_cholesky_reference_trace_preserves_vjp():
    matrix = quabla.Tensor([2, 2], [4.0, 2.0, 2.0, 5.0])
    expected = [2.0, 0.0, 1.0, 2.0]
    assert matrix.cholesky().to_flat_list() == expected

    transform = quabla.tensor_value_and_grad_fn(
        lambda a: a.cholesky().sum(), [("a", [2, 2])]
    )
    value, gradients = transform({"a": matrix})
    assert value.to_flat_list() == [5.0]
    assert_close_rows([gradients["a"].to_flat_list()], [[0.1875, 0.0, 0.25, 0.25]])


def test_tensor_stateless_random_keys_and_glorot_initializer_are_reproducible():
    first_keys = quabla.Tensor.split_key(1234, 2)
    second_keys = quabla.Tensor.split_key(1234, 2)
    assert first_keys == second_keys
    assert first_keys[0] != first_keys[1]

    first = quabla.Tensor.random_normal([2, 3], first_keys[0])
    second = quabla.Tensor.random_normal([2, 3], first_keys[0])
    different = quabla.Tensor.random_normal([2, 3], first_keys[1])
    assert first.shape == [2, 3]
    assert first.to_flat_list() == second.to_flat_list()
    assert first.to_flat_list() != different.to_flat_list()

    glorot = quabla.Tensor.glorot_normal([2, 3], first_keys[0])
    assert glorot.shape == [2, 3]
    assert all(math.isfinite(value) for value in glorot.to_flat_list())


def test_matrix_add():
    a = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = quabla.Matrix([[0.5, 1.5], [2.5, 3.5]])

    c = a + b

    assert c.shape == (2, 2)
    assert c.to_list() == [[1.5, 3.5], [5.5, 7.5]]


def test_matrix_sub():
    a = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = quabla.Matrix([[0.5, 1.5], [2.5, 3.5]])

    c = a - b

    assert c.shape == (2, 2)
    assert c.to_list() == [[0.5, 0.5], [0.5, 0.5]]


def test_matrix_mul():
    a = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = quabla.Matrix([[0.5, 1.5], [2.5, 3.5]])

    c = a * b

    assert c.shape == (2, 2)
    assert c.to_list() == [[0.5, 3.0], [7.5, 14.0]]


def test_matrix_transpose_method_and_property():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    transposed = a.transpose()

    assert transposed.shape == (3, 2)
    assert transposed.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert a.T.to_list() == transposed.to_list()


def test_matrix_reshape_preserves_row_major_order():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    reshaped = a.reshape(3, 2)

    assert reshaped.shape == (3, 2)
    assert reshaped.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_matrix_reshape_rejects_size_change():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    try:
        a.reshape(4, 2)
    except ValueError as exc:
        assert "cannot reshape" in str(exc)
    else:
        raise AssertionError("expected reshape size change to fail")


def test_matrix_concat_axis():
    top = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    bottom = quabla.Matrix([[5.0, 6.0]])
    left = quabla.Matrix([[1.0], [2.0]])
    right = quabla.Matrix([[3.0, 4.0], [5.0, 6.0]])

    vertical = quabla.concat([top, bottom], axis=0)
    horizontal = quabla.concat([left, right], axis=1)

    assert vertical.shape == (3, 2)
    assert vertical.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert horizontal.shape == (2, 3)
    assert horizontal.to_list() == [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]


def test_matrix_supports_scalar_literals():
    a = quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])

    assert (a + 1.5).to_list() == [[2.5, -0.5], [4.5, -2.5]]
    assert (1.5 + a).to_list() == [[2.5, -0.5], [4.5, -2.5]]
    assert (a - 1.5).to_list() == [[-0.5, -3.5], [1.5, -5.5]]
    assert (5.0 - a).to_list() == [[4.0, 7.0], [2.0, 9.0]]
    assert (a * 2.0).to_list() == [[2.0, -4.0], [6.0, -8.0]]
    assert (2.0 * a).to_list() == [[2.0, -4.0], [6.0, -8.0]]


def test_matrix_broadcasts_scalar_matrix_for_elementwise_ops():
    a = quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])
    scalar = quabla.Matrix([[2.0]])

    assert (a + scalar).to_list() == [[3.0, 0.0], [5.0, -2.0]]
    assert (scalar + a).to_list() == [[3.0, 0.0], [5.0, -2.0]]
    assert (a - scalar).to_list() == [[-1.0, -4.0], [1.0, -6.0]]
    assert (scalar - a).to_list() == [[1.0, 4.0], [-1.0, 6.0]]
    assert (a * scalar).to_list() == [[2.0, -4.0], [6.0, -8.0]]
    assert (scalar * a).to_list() == [[2.0, -4.0], [6.0, -8.0]]


def test_matrix_broadcasts_row_and_column_matrices_for_elementwise_ops():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = quabla.Matrix([[10.0, 20.0, 30.0]])
    column = quabla.Matrix([[2.0], [3.0]])

    assert (a + row).to_list() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
    assert (row - a).to_list() == [[9.0, 18.0, 27.0], [6.0, 15.0, 24.0]]
    assert (a * column).to_list() == [[2.0, 4.0, 6.0], [12.0, 15.0, 18.0]]
    assert (a / column).to_list() == [[0.5, 1.0, 1.5], [4.0 / 3.0, 5.0 / 3.0, 2.0]]


def test_tensor_broadcasts_trailing_axes_and_reshapes():
    a = quabla.Tensor([2, 1, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = quabla.Tensor([1, 4, 1], [10.0, 20.0, 30.0, 40.0])

    output = a + b

    assert output.shape == [2, 4, 3]
    assert output.to_flat_list() == [
        11.0,
        12.0,
        13.0,
        21.0,
        22.0,
        23.0,
        31.0,
        32.0,
        33.0,
        41.0,
        42.0,
        43.0,
        14.0,
        15.0,
        16.0,
        24.0,
        25.0,
        26.0,
        34.0,
        35.0,
        36.0,
        44.0,
        45.0,
        46.0,
    ]
    assert output.reshape([3, 2, 4]).shape == [3, 2, 4]


def test_tensor_rejects_data_with_the_wrong_size():
    try:
        quabla.Tensor([2, 2], [1.0, 2.0, 3.0])
    except ValueError as exc:
        assert "data length" in str(exc)
    else:
        raise AssertionError("expected tensor data size mismatch to fail")


def test_tensor_ones_and_full_validate_rank_n_shapes():
    assert quabla.Tensor.ones([2, 3]).to_flat_list() == [1.0] * 6
    assert quabla.Tensor.full([2, 1, 2], -0.25).to_flat_list() == [-0.25] * 4


def test_tensor_arange_creates_coordinate_vectors_and_rejects_invalid_ranges():
    assert quabla.Tensor.arange(0.0, 1.0, 0.25).to_flat_list() == [0.0, 0.25, 0.5, 0.75]
    assert quabla.Tensor.arange(1.0, -0.5, -0.5).to_flat_list() == [1.0, 0.5, 0.0]
    for arguments in [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)]:
        try:
            quabla.Tensor.arange(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"arange accepted invalid arguments {arguments}")


def test_tensor_linspace_includes_endpoints_for_collocation_grids():
    assert quabla.Tensor.linspace(-1.0, 1.0, 5).to_flat_list() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert quabla.Tensor.linspace(2.0, 5.0, 1).to_flat_list() == [2.0]
    try:
        quabla.Tensor.linspace(0.0, 1.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("linspace accepted num=0")


def test_tensor_eye_creates_square_and_rectangular_identity_arrays():
    assert quabla.Tensor.eye(3).to_flat_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert quabla.Tensor.eye(2, 3).to_flat_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    try:
        quabla.Tensor.eye(0)
    except ValueError:
        pass
    else:
        raise AssertionError("eye accepted zero rows")


def test_tensor_transpose_reorders_rank_n_axes_and_validates_permutations():
    tensor = quabla.Tensor([2, 3, 2], [float(value) for value in range(12)])

    reversed_axes = tensor.transpose()
    assert reversed_axes.shape == [2, 3, 2]
    assert reversed_axes.to_flat_list() == [0.0, 6.0, 2.0, 8.0, 4.0, 10.0, 1.0, 7.0, 3.0, 9.0, 5.0, 11.0]

    permuted = tensor.transpose([1, -1, 0])
    assert permuted.shape == [3, 2, 2]
    assert permuted.to_flat_list() == [0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0]

    for axes in ([0, 0, 1], [0, 1], [0, 1, 3]):
        try:
            tensor.transpose(axes)
        except ValueError:
            pass
        else:
            raise AssertionError(f"transpose accepted invalid axes {axes}")


def test_tensor_reductions_match_trace_tensor_axis_semantics():
    tensor = quabla.Tensor([2, 3, 2], [float(value) for value in range(1, 13)])

    assert tensor.sum().shape == []
    assert tensor.sum().to_flat_list() == [78.0]
    assert tensor.mean().to_flat_list() == [6.5]

    axis_one = tensor.sum(axis=1)
    assert axis_one.shape == [2, 2]
    assert axis_one.to_flat_list() == [9.0, 12.0, 27.0, 30.0]

    last_axis = tensor.mean(axis=-1)
    assert last_axis.shape == [2, 3]
    assert last_axis.to_flat_list() == [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]

    multiple_axes = tensor.sum(axis=[0, -1])
    assert multiple_axes.shape == [3]
    assert multiple_axes.to_flat_list() == [18.0, 26.0, 34.0]

    kept_axes = tensor.mean(axis=(0, 2), keepdims=True)
    assert kept_axes.shape == [1, 3, 1]
    assert kept_axes.to_flat_list() == [4.5, 6.5, 8.5]

    all_kept = tensor.sum(keepdims=True)
    assert all_kept.shape == [1, 1, 1]
    assert all_kept.to_flat_list() == [78.0]

    assert tensor.sum(axis=[]).shape == [2, 3, 2]
    assert tensor.sum(axis=[]).to_flat_list() == tensor.to_flat_list()

    for axis in (3, -4, [0, 0]):
        try:
            tensor.sum(axis=axis)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sum accepted invalid axis {axis}")

    assert tensor.norm().to_flat_list() == [math.sqrt(650.0)]
    assert tensor.norm(axis=[0, -1], keepdims=True).shape == [1, 3, 1]
    assert_close_rows(
        [tensor.norm(axis=[0, -1], keepdims=True).to_flat_list()],
        [[math.sqrt(118.0), math.sqrt(206.0), math.sqrt(326.0)]],
    )
    assert tensor.max().to_flat_list() == [12.0]
    assert tensor.min().to_flat_list() == [1.0]
    assert tensor.max(axis=[0, -1]).to_flat_list() == [8.0, 10.0, 12.0]
    assert tensor.min(axis=[0, -1], keepdims=True).shape == [1, 3, 1]
    assert tensor.min(axis=[0, -1], keepdims=True).to_flat_list() == [1.0, 3.0, 5.0]


def test_tensor_elementwise_math_matches_python_math():
    tensor = quabla.Tensor([2, 2], [0.0, 1.0, -1.0, 4.0])

    assert_close_rows([tensor.tanh().to_flat_list()], [[math.tanh(value) for value in tensor.to_flat_list()]])
    assert_close_rows([tensor.exp().to_flat_list()], [[math.exp(value) for value in tensor.to_flat_list()]])
    assert_close_rows([tensor.sin().to_flat_list()], [[math.sin(value) for value in tensor.to_flat_list()]])
    assert_close_rows([tensor.cos().to_flat_list()], [[math.cos(value) for value in tensor.to_flat_list()]])
    assert tensor.powi(2).to_flat_list() == [0.0, 1.0, 1.0, 16.0]

    positive = quabla.Tensor([2], [1.0, math.e**2])
    assert_close_rows([positive.log().to_flat_list()], [[0.0, 2.0]])
    assert positive.sqrt().to_flat_list() == [1.0, math.e]


def test_tensor_supports_numeric_scalars_on_both_sides():
    tensor = quabla.Tensor([2, 2], [1.0, -2.0, 3.0, 4.0])

    assert (tensor + 2.0).to_flat_list() == [3.0, 0.0, 5.0, 6.0]
    assert (2.0 + tensor).to_flat_list() == [3.0, 0.0, 5.0, 6.0]
    assert (tensor - 2.0).to_flat_list() == [-1.0, -4.0, 1.0, 2.0]
    assert (2.0 - tensor).to_flat_list() == [1.0, 4.0, -1.0, -2.0]
    assert (tensor * -0.5).to_flat_list() == [-0.5, 1.0, -1.5, -2.0]
    assert (-0.5 * tensor).to_flat_list() == [-0.5, 1.0, -1.5, -2.0]
    assert (tensor / 2.0).to_flat_list() == [0.5, -1.0, 1.5, 2.0]
    assert (12.0 / tensor).to_flat_list() == [12.0, -6.0, 4.0, 3.0]
    assert (-tensor).to_flat_list() == [-1.0, 2.0, -3.0, -4.0]

    try:
        tensor / 0.0
    except ValueError:
        pass
    else:
        raise AssertionError("tensor division accepted a zero scalar")


def test_tensor_power_supports_integer_and_float_exponents():
    tensor = quabla.Tensor([2], [4.0, 9.0])

    assert (tensor ** 2).to_flat_list() == [16.0, 81.0]
    assert_close_rows([(tensor ** 0.5).to_flat_list()], [[2.0, 3.0]])

    try:
        pow(tensor, 2, 3)
    except TypeError:
        pass
    else:
        raise AssertionError("tensor power accepted a modulo argument")


def test_tensor_comparisons_and_where_support_rank_n_broadcasting():
    values = quabla.Tensor([2, 1, 3], [-1.0, 0.0, 1.0, 2.0, -2.0, 3.0])
    mask = values.gt(0.0)
    on_true = quabla.Tensor([1, 4, 1], [10.0, 20.0, 30.0, 40.0])
    on_false = quabla.Tensor.full([2, 1, 3], -1.0)

    assert mask.shape == [2, 1, 3]
    assert mask.to_flat_list() == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    selected = quabla.where(mask, on_true, on_false)
    assert selected.shape == [2, 4, 3]
    assert selected.to_flat_list() == [
        -1.0, -1.0, 10.0, -1.0, -1.0, 20.0, -1.0, -1.0, 30.0, -1.0, -1.0, 40.0,
        10.0, -1.0, 10.0, 20.0, -1.0, 20.0, 30.0, -1.0, 30.0, 40.0, -1.0, 40.0,
    ]


def test_tensor_maximum_and_minimum_broadcast_and_choose_rhs_on_ties():
    left = quabla.Tensor([2, 1], [2.0, 0.0])
    right = quabla.Tensor([1, 3], [2.0, 1.0, -1.0])

    assert left.maximum(right).shape == [2, 3]
    assert left.maximum(right).to_flat_list() == [2.0, 2.0, 2.0, 2.0, 1.0, 0.0]
    assert left.minimum(right).to_flat_list() == [2.0, 1.0, -1.0, 0.0, 0.0, -1.0]
    assert left.maximum(1.0).to_flat_list() == [2.0, 1.0]
    assert left.minimum(1.0).to_flat_list() == [1.0, 0.0]


def test_tensor_gather_and_scatter_add_support_negative_and_repeated_indices():
    source = quabla.Tensor([2, 3], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert source.gather([2, -3], axis=1).to_flat_list() == [2.0, 0.0, 5.0, 3.0]

    base = quabla.Tensor.zeros([2, 3])
    updates = quabla.Tensor([2, 3], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert base.scatter_add([1, 1, 0], updates, axis=1).to_flat_list() == [
        30.0,
        30.0,
        0.0,
        60.0,
        90.0,
        0.0,
    ]

    for indices in ([], [3]):
        try:
            source.gather(indices, axis=1)
        except (IndexError, ValueError):
            pass
        else:
            raise AssertionError(f"gather accepted invalid indices {indices}")


def test_einsum_scoped_matmul_subset_matches_tensor_matmul():
    left = quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    right = quabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.0, 3.0, 1.0])
    expected = (left @ right).to_flat_list()

    assert quabla.einsum("ij,jk->ik", [left, right]).to_flat_list() == expected
    assert quabla.einsum("...ij,...jk->...ik", [left, right]).to_flat_list() == expected

    for equation in ("ij,ij->ij", "ij,jk->ij"):
        try:
            quabla.einsum(equation, [left, right])
        except ValueError:
            pass
        else:
            raise AssertionError(f"einsum accepted unsupported equation {equation}")


def test_tensor_broadcast_to_materializes_rank_n_contiguous_storage():
    tensor = quabla.Tensor([1, 2, 1], [2.0, -3.0])
    output = tensor.broadcast_to([3, 2, 4])

    assert output.shape == [3, 2, 4]
    assert output.to_flat_list() == [2.0] * 4 + [-3.0] * 4 + [2.0] * 4 + [-3.0] * 4 + [2.0] * 4 + [-3.0] * 4

    try:
        tensor.broadcast_to([3, 3, 4])
    except ValueError:
        pass
    else:
        raise AssertionError("broadcast_to accepted incompatible shape")


def test_tensor_matmul_broadcasts_batch_axes():
    lhs = quabla.Tensor(
        [2, 2, 3],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            2.0,
            0.0,
            1.0,
            1.0,
            3.0,
            2.0,
        ],
    )
    rhs = quabla.Tensor([1, 3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    output = lhs.matmul(rhs)

    assert output.shape == [2, 2, 2]
    assert output.to_flat_list() == [58.0, 64.0, 139.0, 154.0, 25.0, 28.0, 56.0, 62.0]


def test_tensor_concat_supports_rank_n_axes_and_validates_shapes():
    lhs = quabla.Tensor([2, 1, 2], [1.0, 2.0, 5.0, 6.0])
    rhs = quabla.Tensor([2, 2, 2], [3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    result = quabla.concat([lhs, rhs], axis=1)

    assert result.shape == [2, 3, 2]
    assert result.to_flat_list() == [1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0]
    try:
        quabla.concat([lhs, quabla.Tensor([2, 2, 3], [0.0] * 12)], axis=1)
    except ValueError as error:
        assert "cannot concatenate" in str(error)
    else:
        raise AssertionError("concat accepted incompatible Tensor shapes")


def test_tensor_slice_creates_a_strided_read_only_view():
    tensor = quabla.Tensor([3, 4], [float(value) for value in range(12)])

    view = tensor.slice(axis=1, start=1, length=2, step=2)

    assert view.shape == [3, 2]
    assert view.strides == [4, 2]
    assert view.offset == 1
    assert view.to_flat_list() == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    assert view.to_tensor().to_flat_list() == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]


def test_tensor_getitem_supports_contiguous_slices_and_negative_indices():
    tensor = quabla.Tensor([3, 4], [float(value) for value in range(12)])

    assert tensor[1, 1:3].shape == [2]
    assert tensor[1, 1:3].to_flat_list() == [5.0, 6.0]
    assert tensor[:, -3:-1].shape == [3, 2]
    assert tensor[:, -3:-1].to_flat_list() == [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
    assert tensor[-1, -1].shape == []
    assert tensor[-1, -1].to_flat_list() == [11.0]

    for index in (slice(None, None, 2), slice(1, 1), (0, 1, 2)):
        try:
            tensor[index]
        except (IndexError, ValueError):
            pass
        else:
            raise AssertionError(f"Tensor accepted unsupported index {index!r}")


def test_tensor_stack_supports_negative_axes():
    first = quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    second = quabla.Tensor([2, 2], [5.0, 6.0, 7.0, 8.0])
    stacked = quabla.stack([first, second], axis=-1)
    assert stacked.shape == [2, 2, 2]
    assert stacked.to_flat_list() == [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]


def test_trace_tensor_evaluates_rank_n_scalar_loss_vjp():
    def loss(x, y):
        return (x * y + x).sum()

    traced = quabla.trace_tensor(loss, [("x", [2, 1, 3]), ("y", [1, 4, 1])])
    inputs = {
        "x": quabla.Tensor([2, 1, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "y": quabla.Tensor([1, 4, 1], [10.0, 20.0, 30.0, 40.0]),
    }

    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )

    assert traced.output.shape == []
    assert value.to_flat_list() == [2184.0]
    assert gradients["x"].shape == [2, 1, 3]
    assert gradients["x"].to_flat_list() == [104.0] * 6
    assert gradients["y"].shape == [1, 4, 1]
    assert gradients["y"].to_flat_list() == [21.0] * 4

    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        inputs,
        {
            "x": quabla.Tensor([2, 1, 3], [1.0] * 6),
            "y": quabla.Tensor([1, 4, 1], [1.0] * 4),
        },
    )
    assert tangent.shape == []
    assert tangent.to_flat_list() == [708.0]


def test_compiler_facade_unifies_trace_transform_compile_and_execute():
    compiler = quabla.Compiler()
    assert compiler.capabilities()["cpu"] is True

    program = compiler.trace(
        lambda x, weight: (x * weight).sum(),
        [("x", [2]), ("weight", [2])],
    )
    assert program.output_shape == []
    assert "mul" in program.lower_text()

    executable = program.compile("cpu")
    assert executable.target == "cpu"
    assert executable.node_count > 0
    value = executable({
        "x": quabla.Tensor([2], [2.0, 3.0]),
        "weight": quabla.Tensor([2], [4.0, 5.0]),
    })
    assert value.to_flat_list() == [23.0]

    jvp = program.jvp("x")
    assert jvp.output_shape == []
    gradients = program.vjp("loss_cotangent")
    assert set(gradients) == {"x", "weight"}
    gradient = gradients["x"].compile()({
        "x": quabla.Tensor([2], [2.0, 3.0]),
        "weight": quabla.Tensor([2], [4.0, 5.0]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    })
    assert gradient.to_flat_list() == [4.0, 5.0]


def test_python_entrypoints_keep_backend_errors_for_unbuilt_targets():
    # In builds without a backend, the Python entry points keep the pre-facade error timing and
    # messages: MLX construction succeeds and the first execution fails; CUDA reports the backend
    # build hint at compile time.
    compiler = quabla.Compiler()
    program = compiler.trace(lambda x: x * 2.0, [("x", [2])])
    inputs = {"x": quabla.Tensor([2], [1.0, 2.0])}
    mlx_message = "MLX backend is unavailable: build Quabla on macOS with --features mlx"
    cuda_message = "CUDA backend is unavailable for device 0: build Quabla on Linux with --features cuda"

    if not compiler.capability("mlx"):
        batched_inputs = {"x": quabla.Tensor([1, 2], [1.0, 2.0])}
        executables = [
            (program.compile("mlx"), (inputs,)),
            (quabla.tensor_vmap_mlx_fn(lambda x: x * 2.0, [("x", [2])], 1), (batched_inputs,)),
            (
                quabla.tensor_value_and_grad_mlx_fn(
                    lambda x: (x * 2.0).sum(), [("x", [2])], ["x"]
                ),
                (inputs,),
            ),
            (
                quabla.tensor_vmap_jvp_mlx_fn(lambda x: x * 2.0, [("x", [2])], 1),
                (batched_inputs, batched_inputs),
            ),
        ]
        for executable, arguments in executables:
            try:
                executable(*arguments)
                assert False, "expected an unbuilt MLX backend to reject execution"
            except ValueError as error:
                assert mlx_message in str(error)

    if not compiler.capability("cuda"):
        compile_calls = [
            lambda: program.compile("cuda"),
            lambda: quabla.tensor_jit_cuda_fn(lambda x: x * 2.0, [("x", [2])]),
            lambda: quabla.tensor_value_and_grad_cuda_fn(
                lambda x: (x * 2.0).sum(), [("x", [2])], ["x"]
            ),
            lambda: quabla.tensor_vmap_jvp_cuda_fn(lambda x: x * 2.0, [("x", [2])], 1),
        ]
        for compile_call in compile_calls:
            try:
                compile_call()
                assert False, "expected an unbuilt CUDA backend to reject compilation"
            except ValueError as error:
                assert cuda_message in str(error)


def test_compiler_facade_cuda_matches_cpu_for_primal_jvp_and_vjp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    compiler = quabla.Compiler()
    assert compiler.capability("cuda") is True
    program = compiler.trace(
        lambda x, weight, bias: (x * weight + bias).tanh().sum(),
        [("x", [2]), ("weight", [2]), ("bias", [1])],
    )
    inputs = {
        "x": quabla.Tensor([2], [-0.5, 1.25]),
        "weight": quabla.Tensor([2], [1.5, -0.75]),
        "bias": quabla.Tensor([1], [0.25]),
    }

    cpu_value = program.compile("cpu")(inputs)
    cuda_value = program.compile("cuda")(inputs)
    assert_close_rows(
        [cuda_value.to_flat_list()], [cpu_value.to_flat_list()], tol=1e-5
    )

    cpu_jvp = program.jvp("x").compile("cpu")({**inputs})
    cuda_jvp = program.jvp("x").compile("cuda")({**inputs})
    assert_close_rows([cuda_jvp.to_flat_list()], [cpu_jvp.to_flat_list()], tol=1e-5)

    cotangent = quabla.Tensor([], [1.0])
    cpu_gradients = program.vjp("loss_cotangent")
    cuda_gradients = program.vjp("loss_cotangent")
    transformed_inputs = {**inputs, "loss_cotangent": cotangent}
    for name in ("x", "weight", "bias"):
        cpu_gradient = cpu_gradients[name].compile("cpu")(transformed_inputs)
        cuda_gradient = cuda_gradients[name].compile("cuda")(transformed_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )


def test_compiler_facade_cuda_executes_nonlinear_scan_hvp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * current * captured_scale + index,
                (current * current * captured_scale + index) * captured_scale,
            ),
            initial,
            [scale],
        )
        return carry + outputs.sum()

    compiler = quabla.Compiler()
    loss = compiler.trace(scan_loss, [("initial", []), ("scale", [])])
    # d/d(scale) [d(loss)/d(scale)] with the scalar VJP seed fixed at one.
    hvp = loss.vjp("loss_cotangent")["scale"].jvp("scale")
    inputs = {
        "initial": quabla.Tensor([], [0.4]),
        "scale": quabla.Tensor([], [0.8]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    cpu = hvp.compile("cpu")(inputs)
    cuda = hvp.compile("cuda")(inputs)
    assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=2e-5)


def test_compiler_facade_cuda_executes_equal_count_reshaped_scan_hvp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                (current * captured_scale + index).reshape([1, 2]),
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    compiler = quabla.Compiler()
    loss = compiler.trace(scan_loss, [("initial", [2]), ("scale", [2])])
    hvp = loss.vjp("loss_cotangent")["scale"].jvp("scale")
    inputs = {
        "initial": quabla.Tensor([2], [0.4, -0.6]),
        "scale": quabla.Tensor([2], [0.8, 1.1]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    cpu = hvp.compile("cpu")(inputs)
    cuda = hvp.compile("cuda")(inputs)
    assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=2e-5)


def test_compiler_facade_cuda_executes_broadcast_scan_primal():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_values(initial, scale):
        _, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                (current * captured_scale + index).broadcast_to([2, 3]),
            ),
            initial,
            [scale],
        )
        return outputs

    compiler = quabla.Compiler()
    program = compiler.trace(scan_values, [("initial", [2, 1]), ("scale", [2, 1])])
    inputs = {
        "initial": quabla.Tensor([2, 1], [0.4, -0.6]),
        "scale": quabla.Tensor([2, 1], [0.8, 1.1]),
    }
    cpu = program.compile("cpu")(inputs)
    cuda = program.compile("cuda")(inputs)
    assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_compiler_facade_cuda_aggregates_broadcast_scan_vjp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                (current * captured_scale + index).broadcast_to([2, 3]),
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    compiler = quabla.Compiler()
    program = compiler.trace(scan_loss, [("initial", [2, 1]), ("scale", [2, 1])])
    inputs = {
        "initial": quabla.Tensor([2, 1], [0.4, -0.6]),
        "scale": quabla.Tensor([2, 1], [0.8, 1.1]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    gradients = program.vjp("loss_cotangent")
    for name in ("initial", "scale"):
        cpu = gradients[name].compile("cpu")(inputs)
        cuda = gradients[name].compile("cuda")(inputs)
        assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=2e-5)


def test_compiler_facade_cuda_executes_broadcast_scan_hvp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                (current * captured_scale + index).broadcast_to([2, 3]),
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    program = quabla.Compiler().trace(
        scan_loss, [("initial", [2, 1]), ("scale", [2, 1])]
    )
    hvp = program.vjp("loss_cotangent")["scale"].jvp("scale")
    inputs = {
        "initial": quabla.Tensor([2, 1], [0.4, -0.6]),
        "scale": quabla.Tensor([2, 1], [0.8, 1.1]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    cpu = hvp.compile("cpu")(inputs)
    cuda = hvp.compile("cuda")(inputs)
    assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=3e-5)


def test_compiler_facade_cuda_aggregates_broadcast_capture_scan_hvp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                (current * captured_scale + index).broadcast_to([2, 3]),
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    program = quabla.Compiler().trace(
        scan_loss, [("initial", [2, 1]), ("scale", [1, 1])]
    )
    hvp = program.vjp("loss_cotangent")["scale"].jvp("scale")
    inputs = {
        "initial": quabla.Tensor([2, 1], [0.4, -0.6]),
        "scale": quabla.Tensor([1, 1], [0.8]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    cpu = hvp.compile("cpu")(inputs)
    cuda = hvp.compile("cuda")(inputs)
    assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=3e-5)



def test_cuda_value_and_grad_groups_broadcast_scan_vjp_targets():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    # Requesting both gradients fuses them into one grouped Scan VJP kernel, which must reduce the
    # per-step output cotangents onto the carry lanes exactly like the single-target kernel.
    def make_loss(nonlinear):
        def body(index, current, scale):
            scaled = current * scale
            nxt = (scaled.tanh() if nonlinear else scaled) + index
            return nxt, nxt.broadcast_to([2, 3])

        def loss(initial, scale):
            carry, outputs = quabla.tensor_scan_region(0, 3, body, initial, [scale])
            return carry.sum() + (outputs * outputs).sum()

        return loss

    specs = [("initial", [2, 1]), ("scale", [2, 1])]
    inputs = {
        "initial": quabla.Tensor([2, 1], [0.4, -0.6]),
        "scale": quabla.Tensor([2, 1], [0.8, 1.1]),
    }
    for nonlinear in (False, True):
        loss = make_loss(nonlinear)
        cpu_value, cpu_gradients = quabla.tensor_value_and_grad_fn(loss, specs)(inputs)
        value, gradients = quabla.tensor_value_and_grad_cuda_fn(
            loss, specs, ["initial", "scale"]
        )(inputs)
        actual = [value.to_flat_list()] + [
            gradients[name].to_flat_list() for name in ("initial", "scale")
        ]
        expected = [cpu_value.to_flat_list()] + [
            cpu_gradients[name].to_flat_list() for name in ("initial", "scale")
        ]
        for actual_row, expected_row in zip(actual, expected):
            assert len(actual_row) == len(expected_row)
            for actual_value, expected_value in zip(actual_row, expected_row):
                tolerance = 2e-5 * max(1.0, abs(expected_value))
                assert abs(actual_value - expected_value) <= tolerance, (
                    nonlinear,
                    actual,
                    expected,
                )


def test_cuda_value_and_grad_reexecutes_grouped_scan_vjp_plan():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    # Training loops call one compiled value-and-grad function per step, so its grouped Scan VJP
    # plan must run any number of times; alternating inputs exposes state left by the prior call.
    def body(index, current, scale):
        nxt = (current * scale).tanh() + index
        return nxt, nxt.broadcast_to([2, 3])

    def loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(0, 3, body, initial, [scale])
        return carry.sum() + (outputs * outputs).sum()

    specs = [("initial", [2, 1]), ("scale", [2, 1])]
    input_sets = [
        {
            "initial": quabla.Tensor([2, 1], [0.4, -0.6]),
            "scale": quabla.Tensor([2, 1], [0.8, 1.1]),
        },
        {
            "initial": quabla.Tensor([2, 1], [-0.3, 0.5]),
            "scale": quabla.Tensor([2, 1], [1.2, -0.7]),
        },
    ]
    cpu = quabla.tensor_value_and_grad_fn(loss, specs)
    cuda = quabla.tensor_value_and_grad_cuda_fn(loss, specs, ["initial", "scale"])
    for call in range(3):
        inputs = input_sets[call % len(input_sets)]
        cpu_value, cpu_gradients = cpu(inputs)
        value, gradients = cuda(inputs)
        actual = [value.to_flat_list()] + [
            gradients[name].to_flat_list() for name in ("initial", "scale")
        ]
        expected = [cpu_value.to_flat_list()] + [
            cpu_gradients[name].to_flat_list() for name in ("initial", "scale")
        ]
        for actual_row, expected_row in zip(actual, expected):
            assert len(actual_row) == len(expected_row)
            for actual_value, expected_value in zip(actual_row, expected_row):
                tolerance = 2e-5 * max(1.0, abs(expected_value))
                assert abs(actual_value - expected_value) <= tolerance, (
                    call,
                    actual,
                    expected,
                )


def test_compiler_facade_cuda_rejects_indexed_unequal_lane_scan_hvp():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                (current * captured_scale + index).broadcast_to([2, 3]) + index,
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    program = quabla.Compiler().trace(
        scan_loss, [("initial", [2, 1]), ("scale", [2, 1])]
    )
    hvp = program.vjp("loss_cotangent")["scale"].jvp("scale")
    try:
        hvp.compile("cuda")
    except ValueError as error:
        assert "requires a direct broadcast from the carry shape" in str(error)
    else:
        raise AssertionError("CUDA Scan HVP accepted an indexed unequal-lane output")


def test_compiler_facade_cuda_executes_packed_pair_scan_bodies():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    # The nonlinear JVP tangent half reads the primal half; the row swap reads the other half.
    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                (current * captured_scale).tanh(),
                (current * captured_scale).tanh(),
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    program = quabla.Compiler().trace(scan_loss, [("initial", [3]), ("scale", [3])])
    inputs = {
        "initial": quabla.Tensor([3], [0.2, -0.3, 0.5]),
        "scale": quabla.Tensor([3], [0.8, 1.1, 0.6]),
    }
    for name in ("initial", "scale"):
        jvp = program.jvp(name)
        cpu = jvp.compile("cpu")(inputs)
        cuda = jvp.compile("cuda")(inputs)
        assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=2e-5)

    def swap_rows(initial, scale):
        _, outputs = quabla.tensor_scan_region(
            0,
            2,
            lambda index, current, captured_scale: (
                quabla.concat([current.slice(0, 1, 2), current.slice(0, 0, 1)], 0)
                * captured_scale,
                quabla.concat([current.slice(0, 1, 2), current.slice(0, 0, 1)], 0)
                * captured_scale,
            ),
            initial,
            [scale],
        )
        return outputs

    program = quabla.Compiler().trace(swap_rows, [("initial", [2, 3]), ("scale", [3])])
    inputs = {
        "initial": quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "scale": quabla.Tensor([3], [0.5, 1.0, 2.0]),
    }
    cpu = program.compile("cpu")(inputs)
    cuda = program.compile("cuda")(inputs)
    assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def _scan_rejection_program(body, carry_shape, capture_shape, result="outputs"):
    def function(initial, capture):
        carry, outputs = quabla.tensor_scan_region(0, 2, body, initial, [capture])
        if result == "loss":
            return carry.sum() + outputs.sum()
        return outputs

    return quabla.Compiler().trace(
        function, [("initial", carry_shape), ("capture", capture_shape)]
    )


def test_compiler_facade_cuda_rejects_every_unsupported_scan_lane_form():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def primal(body, carry_shape, capture_shape):
        return _scan_rejection_program(body, carry_shape, capture_shape)

    def gradient(body, carry_shape, capture_shape, name):
        program = _scan_rejection_program(body, carry_shape, capture_shape, "loss")
        return program.vjp("loss_cotangent")[name]

    def hvp(body, carry_shape, capture_shape):
        return gradient(body, carry_shape, capture_shape, "capture").jvp("capture")

    def reshaped_capture(index, current, capture):
        return (
            (current * capture.reshape([1, 2])).tanh(),
            (current * capture.reshape([1, 2])).tanh(),
        )
    def scalar_capture(index, current, capture):
        return (
            current * capture.tanh(),
            current * capture.tanh(),
        )
    def linear(index, current, capture):
        return (
            current * capture + index,
            current * capture + index,
        )
    cases = [
        # Primal Scan: lane shapes and capture shapes.
        (
            "reduced per-step output",
            primal(lambda i, c, s: (c * s, (c * s).sum()), [3], [3]),
            "must broadcast the carry shape",
        ),
        (
            "output-shaped capture on unequal lanes",
            primal(lambda i, c, s: (c + i, c.broadcast_to([2, 3]) * s), [2, 1], [2, 3]),
            "cannot broadcast to carry shape",
        ),
        (
            "capture that cannot follow an equal-count output reshape",
            primal(lambda i, c, s: (c * s, (c * s).reshape([6])), [2, 3], [3]),
            "cannot broadcast to output shape",
        ),
        # Primal Scan: non-elementwise and indexed bodies.
        (
            "matmul body",
            primal(lambda i, c, w: (c.matmul(w), c.matmul(w)), [2, 2], [2, 2]),
            "unsupported matmul operation",
        ),
        (
            "reduction inside the body",
            primal(lambda i, c, s: (c - c.sum() * s, c * s), [3], []),
            "unsupported sum operation",
        ),
        (
            "transpose body",
            primal(lambda i, c, s: (c.transpose() * s, c * s), [2, 2], []),
            "unsupported transpose operation",
        ),
        (
            "indexed gather on an unpacked carry",
            primal(lambda i, c, s: (c.gather([2, 0, 1], 0) * s, c * s), [3], [3]),
            "unsupported gather operation",
        ),
        (
            "reshape that moves a capture axis",
            primal(reshaped_capture, [2, 2], [2, 1]),
            "changes the per-lane broadcast layout",
        ),
        (
            "reshape before an unequal-lane broadcast",
            primal(
                lambda i, c, s: (c * s, (c * s).reshape([1, 2]).broadcast_to([2, 2])),
                [2, 1],
                [1],
            ),
            "changes the per-lane broadcast layout",
        ),
        # First-order ScanVjp.
        (
            "VJP of an indexed unequal-lane output",
            gradient(
                lambda i, c, s: (c * s + i, (c * s + i).broadcast_to([2, 3]) + i),
                [2, 1],
                [2, 1],
                "initial",
            ),
            "requires a direct broadcast from the carry shape",
        ),
        (
            "VJP through a capture reshape",
            gradient(reshaped_capture, [2, 2], [2, 1], "capture"),
            "changes the per-lane broadcast layout",
        ),
        (
            "VJP needing an in-body capture reduction",
            gradient(scalar_capture, [3], [], "capture"),
            "did not expose an elementwise contribution",
        ),
        (
            "VJP of a packed-pair Scan JVP",
            _scan_rejection_program(linear, [3], [3], "loss")
            .jvp("initial")
            .vjp("jvp_cotangent")["initial"],
            "unsupported pad_slice operation",
        ),
        # Forward-over-reverse ScanVjpJvp.
        (
            "HVP through a capture reshape",
            hvp(reshaped_capture, [2, 2], [2, 1]),
            "changes the per-lane broadcast layout",
        ),
        (
            "HVP needing an in-body capture reduction",
            hvp(scalar_capture, [3], []),
            "did not expose an elementwise contribution",
        ),
    ]
    for label, program, expected in cases:
        try:
            program.compile("cuda")
        except ValueError as error:
            assert expected in str(error), f"{label}: {error}"
        else:
            raise AssertionError(f"CUDA accepted unsupported Scan form: {label}")


def test_compiler_facade_rejects_scan_derivatives_beyond_forward_over_reverse():
    gradient = _scan_rejection_program(
        lambda index, current, capture: (
            (current * capture).tanh(),
            (current * capture).tanh(),
        ),
        [2],
        [2],
        "loss",
    ).vjp("loss_cotangent")["capture"]
    cases = [
        (
            "VJP of a Scan VJP",
            lambda: gradient.vjp("second_cotangent"),
            "symbolic VJP through a Scan VJP result is not implemented",
        ),
        (
            "JVP of a Scan VJP JVP",
            lambda: gradient.jvp("capture").jvp("initial"),
            "symbolic JVP through a Scan VJP JVP result is not implemented",
        ),
    ]
    for label, transform, expected in cases:
        try:
            transform()
        except ValueError as error:
            assert expected in str(error), f"{label}: {error}"
        else:
            raise AssertionError(f"accepted unsupported Scan derivative: {label}")



def test_trace_tensor_batched_matmul_scalar_loss_vjp():
    def loss(x, y):
        return x.matmul(y).sum()

    traced = quabla.trace_tensor(loss, [("x", [2, 2, 3]), ("y", [1, 3, 2])])
    inputs = {
        "x": quabla.Tensor(
            [2, 2, 3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0],
        ),
        "y": quabla.Tensor([1, 3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    }

    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )

    assert value.to_flat_list() == [586.0]
    assert gradients["x"].to_flat_list() == [15.0, 19.0, 23.0] * 4
    assert gradients["y"].to_flat_list() == [8.0, 8.0, 10.0, 10.0, 12.0, 12.0]


def test_trace_tensor_symbolic_jvp_keeps_parameter_gradients():
    traced = quabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [1]), ("weight", [1]), ("forcing", [1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    assert second_derivative.output.shape == [1]
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).sum()
    inputs = {
        "x": quabla.Tensor([1], [0.3]),
        "weight": quabla.Tensor([1], [1.2]),
        "forcing": quabla.Tensor([1], [0.5]),
    }
    _, gradients = second_derivative.graph.evaluate_value_and_vjp(
        residual_loss.node_id,
        inputs,
        quabla.Tensor([], [1.0]),
    )
    assert abs(gradients["weight"].to_flat_list()[0]) > 1e-8


def test_trace_tensor_symbolic_vjp_matches_cpu_vjp():
    traced = quabla.trace_tensor(
        lambda x, bias: (x + bias).tanh().sum(),
        [("x", [2, 1]), ("bias", [1, 3])],
    )
    inputs = {
        "x": quabla.Tensor([2, 1], [-1.0, 2.0]),
        "bias": quabla.Tensor([1, 3], [0.0, 1.0, -1.0]),
    }
    cotangent = quabla.Tensor([], [1.0])
    _, direct = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, cotangent
    )
    gradients = traced.symbolic_vjp("loss_cotangent")
    transformed_inputs = {**inputs, "loss_cotangent": cotangent}

    assert set(gradients) == {"x", "bias"}
    for name, gradient_trace in gradients.items():
        value = gradient_trace.graph.evaluate(
            gradient_trace.output.node_id, transformed_inputs
        )
        assert value.shape == direct[name].shape
        assert value.to_flat_list() == direct[name].to_flat_list()


def test_trace_tensor_symbolic_vjp_supports_composed_loss_nodes():
    traced = quabla.trace_tensor(
        lambda x, weight: (x * weight).tanh(),
        [("x", [1]), ("weight", [1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    loss = second_derivative.output.powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    inputs = {
        "x": quabla.Tensor([1], [0.5]),
        "weight": quabla.Tensor([1], [0.3]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    gradient = gradients["weight"].graph.evaluate(
        gradients["weight"].output.node_id, inputs
    )

    assert abs(gradient.to_flat_list()[0]) > 1e-8


def test_trace_tensor_concat_supports_rank_n_ad_and_symbolic_transforms():
    traced = quabla.trace_tensor(
        lambda left, right: quabla.concat([left, right], axis=1).powi(2).sum(),
        [("left", [2, 1, 2]), ("right", [2, 2, 2])],
    )
    inputs = {
        "left": quabla.Tensor([2, 1, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": quabla.Tensor([2, 2, 2], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    }
    cotangent = quabla.Tensor([], [1.0])
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, cotangent
    )
    assert value.to_flat_list() == [650.0]
    assert gradients["left"].to_flat_list() == [2.0, 4.0, 6.0, 8.0]
    assert gradients["right"].to_flat_list() == [
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
    ]

    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        inputs,
        {
            "left": quabla.Tensor.ones([2, 1, 2]),
            "right": quabla.Tensor.zeros([2, 2, 2]),
        },
    )
    assert tangent.to_flat_list() == [20.0]

    symbolic_gradients = traced.symbolic_vjp("loss_cotangent")
    transformed_inputs = {**inputs, "loss_cotangent": cotangent}
    for name, gradient in symbolic_gradients.items():
        assert gradient.graph.evaluate(gradient.output.node_id, transformed_inputs).to_flat_list() == gradients[name].to_flat_list()

    first_derivative = traced.symbolic_jvp("left")
    assert first_derivative.graph.evaluate(first_derivative.output.node_id, inputs).to_flat_list() == [20.0]


def test_trace_tensor_slice_supports_rank_n_ad_and_symbolic_transforms():
    traced = quabla.trace_tensor(
        lambda x: x.slice(-2, 1, 3).powi(2).sum(),
        [("x", [2, 4, 2])],
    )
    inputs = {"x": quabla.Tensor([2, 4, 2], [float(value) for value in range(1, 17)])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [716.0]
    assert gradients["x"].to_flat_list() == [
        0.0,
        0.0,
        6.0,
        8.0,
        10.0,
        12.0,
        0.0,
        0.0,
        0.0,
        0.0,
        22.0,
        24.0,
        26.0,
        28.0,
        0.0,
        0.0,
    ]
    symbolic_gradient = traced.symbolic_vjp("loss_cotangent")["x"]
    assert symbolic_gradient.graph.evaluate(
        symbolic_gradient.output.node_id,
        {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])},
    ).to_flat_list() == gradients["x"].to_flat_list()


def test_trace_tensor_getitem_preserves_slice_ad_and_backend_parity():
    traced = quabla.trace_tensor(lambda x: x[1, 1:3].powi(2).sum(), [("x", [3, 4])])
    inputs = {"x": quabla.Tensor([3, 4], [float(value) for value in range(12)])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [61.0]
    assert gradients["x"].to_flat_list() == [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    symbolic = traced.symbolic_vjp("loss_cotangent")["x"]
    symbolic_value = symbolic.graph.evaluate(
        symbolic.output.node_id,
        {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])},
    )
    assert symbolic_value.to_flat_list() == gradients["x"].to_flat_list()

    cpu = traced.output.compile_cpu().evaluate(inputs)
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx = traced.output.compile_mlx().evaluate(inputs)
        assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)
        mlx_gradient = symbolic.output.compile_mlx().evaluate(
            {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])}
        )
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [gradients["x"].to_flat_list()], tol=1e-5
        )
    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda = traced.output.compile_cuda().evaluate(inputs)
        assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_trace_tensor_slice_rejects_invalid_ranges():
    traced = quabla.trace_tensor(lambda x: x, [("x", [2, 3])])
    for args in ((0, 3, 2), (1, 0, 4), (2, 0, 1)):
        try:
            traced.output.slice(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"slice{args} should reject an invalid range")


def test_trace_tensor_broadcast_to_supports_rank_n_ad_and_symbolic_transforms():
    traced = quabla.trace_tensor(
        lambda x: x.broadcast_to([2, 3, 2]).powi(2).sum(), [("x", [1, 3, 1])]
    )
    inputs = {"x": quabla.Tensor([1, 3, 1], [1.0, 2.0, 3.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [56.0]
    assert gradients["x"].to_flat_list() == [8.0, 16.0, 24.0]
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        inputs,
        {"x": quabla.Tensor.ones([1, 3, 1])},
    )
    assert tangent.to_flat_list() == [48.0]
    symbolic_gradient = traced.symbolic_vjp("loss_cotangent")["x"]
    assert symbolic_gradient.graph.evaluate(
        symbolic_gradient.output.node_id,
        {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])},
    ).to_flat_list() == gradients["x"].to_flat_list()


def test_trace_tensor_stack_supports_symbolic_vjp():
    traced = quabla.trace_tensor(
        lambda left, right: quabla.stack([left, right], axis=-1).powi(2).sum(),
        [("left", [2, 2]), ("right", [2, 2])],
    )
    inputs = {
        "left": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": quabla.Tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    }
    gradient = traced.symbolic_vjp("loss_cotangent")["right"]
    result = gradient.graph.evaluate(
        gradient.output.node_id,
        {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])},
    )
    assert result.to_flat_list() == [10.0, 12.0, 14.0, 16.0]


def test_cuda_trace_tensor_concat_keeps_primal_and_symbolic_vjp_on_device():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda left, right: quabla.concat([left, right], axis=1).powi(2).sum(),
        [("left", [2, 1, 2]), ("right", [2, 2, 2])],
    )
    inputs = {
        "left": quabla.Tensor([2, 1, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": quabla.Tensor([2, 2, 2], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    }
    assert_close_rows([traced.output.compile_cuda().evaluate(inputs).to_flat_list()], [[650.0]], tol=1e-5)

    gradient = traced.symbolic_vjp("loss_cotangent")["right"].output.compile_cuda()
    result = gradient.evaluate({**inputs, "loss_cotangent": quabla.Tensor([], [1.0])})
    assert_close_rows(
        [result.to_flat_list()],
        [[10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]],
        tol=1e-5,
    )


def test_mlx_trace_tensor_compiles_and_matches_cpu():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, y, bias: quabla.concat([x, y], axis=0)
        .add(bias.broadcast_to([2, 2]))
        .tanh()
        .sum(),
        [("x", [1, 2]), ("y", [1, 2]), ("bias", [1, 2])],
    )
    inputs = {
        "x": quabla.Tensor([1, 2], [-1.0, 2.0]),
        "y": quabla.Tensor([1, 2], [0.5, -0.25]),
        "bias": quabla.Tensor([1, 2], [0.25, -0.5]),
    }
    cpu = traced.output.compile_cpu().evaluate(inputs)
    plan = traced.output.compile_mlx()
    assert plan.backend == "mlx"
    assert_close_rows([plan.evaluate(inputs).to_flat_list()], [cpu.to_flat_list()], tol=1e-5)
    assert plan.evaluate_device(inputs) is None


def test_mlx_strided_outputs_read_back_in_row_major_order():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    cases = [
        (lambda x: x.transpose([1, 0]), [2, 3], [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        (lambda x: x.broadcast_to([2, 3]), [1, 3], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
    ]
    for function, shape, expected in cases:
        traced = quabla.trace_tensor(function, [("x", shape)])
        count = shape[0] * shape[1]
        inputs = {"x": quabla.Tensor(shape, [float(value) for value in range(1, count + 1)])}
        assert traced.output.compile_cpu().evaluate(inputs).to_flat_list() == expected
        assert traced.output.compile_mlx().evaluate(inputs).to_flat_list() == expected


def test_mlx_execution_plan_retains_static_inputs_across_calls():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x.matmul(weight) + bias).tanh(),
        [("x", [2, 2]), ("weight", [2, 2]), ("bias", [1, 2])],
    )
    inputs = {
        "x": quabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": quabla.Tensor([2, 2], [1.0, -1.0, 0.5, 1.5]),
        "bias": quabla.Tensor([1, 2], [0.25, -0.5]),
    }
    cpu = traced.output.compile_cpu().evaluate(inputs)
    plan = traced.output.compile_mlx()
    plan.retain_inputs(inputs, ["weight", "bias"])
    assert plan.retained_input_names == ["bias", "weight"]
    dynamic_inputs = {"x": inputs["x"]}
    assert_close_rows(
        [plan.evaluate(dynamic_inputs).to_flat_list()], [cpu.to_flat_list()], tol=1e-5
    )
    assert plan.evaluate_device(dynamic_inputs) is None

    plan.clear_retained_inputs()
    assert plan.retained_input_names == []
    try:
        plan.evaluate(dynamic_inputs)
    except ValueError as error:
        assert 'missing input "weight"' in str(error)
    else:
        raise AssertionError("cleared retained MLX input unexpectedly remained available")


def test_mlx_symbolic_vjp_executes_mlp_bias_gradient():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x.matmul(weight) + bias).tanh().sum(),
        [("x", [2, 2]), ("weight", [2, 3]), ("bias", [1, 3])],
    )
    inputs = {
        "x": quabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": quabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.5, 1.5, -0.5]),
        "bias": quabla.Tensor([1, 3], [0.25, -0.5, 1.0]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    gradient = traced.symbolic_vjp("loss_cotangent")["bias"].output
    cpu = gradient.compile_cpu().evaluate(inputs)
    mlx = gradient.compile_mlx().evaluate(inputs)
    assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_mlx_trace_tensor_executes_masked_loss():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x: quabla.where(x.gt(0.0), x.powi(2), x).sum(), [("x", [2, 2])]
    )
    inputs = {"x": quabla.Tensor([2, 2], [-2.0, -1.0, 1.0, 3.0])}
    cpu = traced.output.compile_cpu().evaluate(inputs)
    mlx = traced.output.compile_mlx().evaluate(inputs)
    assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_mlx_symbolic_vjp_executes_masked_loss_gradient():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x: quabla.where(x.gt(0.0), x.powi(2), x).sum(), [("x", [2, 2])]
    )
    inputs = {
        "x": quabla.Tensor([2, 2], [-2.0, -1.0, 1.0, 3.0]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }
    gradient = traced.symbolic_vjp("loss_cotangent")["x"].output
    cpu = gradient.compile_cpu().evaluate(inputs)
    mlx = gradient.compile_mlx().evaluate(inputs)
    assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_mlx_greater_returns_float_mask_like_cpu():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    inputs = {
        "x": quabla.Tensor([3], [1.0, 2.0, 3.0]),
        "y": quabla.Tensor([3], [2.0, 2.0, 2.0]),
    }
    for function, expected in [
        (lambda x, y: x.gt(y), [0.0, 0.0, 1.0]),
        (lambda x, y: (x.gt(y) + x.gt(y)).sum(), [2.0]),
    ]:
        traced = quabla.trace_tensor(function, [("x", [3]), ("y", [3])])
        assert traced.output.compile_cpu().evaluate(inputs).to_flat_list() == expected
        assert traced.output.compile_mlx().evaluate(inputs).to_flat_list() == expected


def test_cuda_trace_tensor_slice_keeps_primal_and_symbolic_vjp_on_device():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x: x.slice(1, 1, 3).powi(2).sum(), [("x", [2, 4, 2])]
    )
    inputs = {"x": quabla.Tensor([2, 4, 2], [float(value) for value in range(1, 17)])}
    assert_close_rows([traced.output.compile_cuda().evaluate(inputs).to_flat_list()], [[716.0]], tol=1e-5)
    gradient = traced.symbolic_vjp("loss_cotangent")["x"].output.compile_cuda()
    result = gradient.evaluate({**inputs, "loss_cotangent": quabla.Tensor([], [1.0])})
    assert_close_rows(
        [result.to_flat_list()],
        [[0.0, 0.0, 6.0, 8.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 22.0, 24.0, 26.0, 28.0, 0.0, 0.0]],
        tol=1e-5,
    )


def test_cuda_trace_tensor_broadcast_to_keeps_primal_and_symbolic_vjp_on_device():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x: x.broadcast_to([2, 3, 2]).powi(2).sum(), [("x", [1, 3, 1])]
    )
    inputs = {"x": quabla.Tensor([1, 3, 1], [1.0, 2.0, 3.0])}
    assert_close_rows([traced.output.compile_cuda().evaluate(inputs).to_flat_list()], [[56.0]], tol=1e-5)
    gradient = traced.symbolic_vjp("loss_cotangent")["x"].output.compile_cuda()
    result = gradient.evaluate({**inputs, "loss_cotangent": quabla.Tensor([], [1.0])})
    assert_close_rows([result.to_flat_list()], [[8.0, 16.0, 24.0]], tol=1e-5)


def test_cuda_multi_parameter_sgd_keeps_gradient_plans_synchronized():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    weight_gradient = gradients["weight"].output.compile_cuda()
    bias_gradient = gradients["bias"].output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([2], [-1.0, 1.0]),
        "target": quabla.Tensor([2], [-1.0, 3.0]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
        "weight": quabla.Tensor([1], [0.0]),
        "bias": quabla.Tensor([1], [0.0]),
    }
    retained = ["x", "target", "loss_cotangent", "weight", "bias"]

    for _ in range(100):
        weight_gradient.sgd_step(inputs, "weight", 0.05, retained)
        bias_gradient.sgd_step(inputs, "bias", 0.05, retained)
        weight_gradient.sync_retained_input_to("weight", bias_gradient)
        bias_gradient.sync_retained_input_to("bias", weight_gradient)

    assert abs(weight_gradient.retained_input("weight").to_flat_list()[0] - 2.0) < 1e-4
    assert abs(bias_gradient.retained_input("bias").to_flat_list()[0] - 1.0) < 1e-4


def test_cuda_adam_keeps_optimizer_state_on_device():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight: x * weight,
        [("x", [1]), ("weight", [1])],
    )
    target = traced.graph.input("target", [1])
    loss = (traced.output - target).powi(2).sum()
    gradient = loss.symbolic_vjp("loss_cotangent")["weight"].output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([1], [2.0]),
        "weight": quabla.Tensor([1], [0.0]),
        "target": quabla.Tensor([1], [6.0]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }

    for _ in range(2):
        gradient.adam_step(inputs, "weight", 0.1, ["x", "target", "loss_cotangent"])

    weight = gradient.retained_input("weight").to_flat_list()[0]
    assert 0.19 < weight < 0.21


def test_cuda_plan_evaluates_with_static_inputs_retained_on_device():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, bias: (x + bias).tanh(),
        [("x", [2, 1]), ("bias", [1, 2])],
    )
    plan = traced.output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([2, 1], [-1.0, 2.0]),
        "bias": quabla.Tensor([1, 2], [0.0, 1.0]),
    }
    plan.evaluate_device(inputs, ["x", "bias"])
    plan.evaluate_device(inputs, ["x", "bias"])
    plan.synchronize()
    assert plan.backend in {"cublas", "nvrtc"}
    assert plan.benchmark_device(inputs, 2, ["x", "bias"]) > 0.0
    actual = plan.evaluate(inputs).to_flat_list()
    expected = [math.tanh(-1.0), 0.0, math.tanh(2.0), math.tanh(3.0)]
    for value, target in zip(actual, expected):
        assert abs(value - target) < 1e-5


def test_cuda_plan_reuses_dead_temporary_buffers():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: ((x.matmul(weight) + bias).tanh() + bias),
        [("x", [2, 2]), ("weight", [2, 2]), ("bias", [1, 2])],
    )
    plan = traced.output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": quabla.Tensor([2, 2], [1.0, 2.0, -1.0, 1.0]),
        "bias": quabla.Tensor([1, 2], [0.25, -0.5]),
    }

    plan.evaluate(inputs)
    first = plan.evaluate(inputs).to_flat_list()
    first_buffer_count = plan.device_buffer_count
    second = plan.evaluate(inputs).to_flat_list()

    assert first == second
    assert first_buffer_count == plan.device_buffer_count
    assert first_buffer_count < plan.node_count


def test_cuda_fuses_rank_two_matmul_bias_tanh_epilogue():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x.matmul(weight) + bias).tanh(),
        [("x", [2, 2]), ("weight", [2, 3]), ("bias", [1, 3])],
    )
    plan = traced.output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": quabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.5, 1.5, -0.5]),
        "bias": quabla.Tensor([1, 3], [0.25, -0.5, 1.0]),
    }

    actual = plan.evaluate(inputs).to_flat_list()
    expected = [
        math.tanh(-0.75),
        math.tanh(0.5),
        math.tanh(-1.0),
        math.tanh(2.25),
        math.tanh(1.5),
        math.tanh(2.0),
    ]
    assert plan.fused_matmul_bias_tanh
    for value, target in zip(actual, expected):
        assert abs(value - target) < 1e-5


def _cuda_mlp_tensor(shape, seed):
    count = math.prod(shape)
    return quabla.Tensor(shape, [((index * 7 + seed) % 11) * 0.1 - 0.5 for index in range(count)])


def test_cuda_matmul_bias_tanh_with_computed_operands_matches_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    specs = [("x", [4, 3]), ("w", [3, 2]), ("b", [1, 2])]
    inputs = {name: _cuda_mlp_tensor(shape, seed) for seed, (name, shape) in enumerate(specs)}
    # The fused epilogue binds only plan inputs; computed operands must use the per-node program.
    functions = {
        "lhs": lambda x, w, b: (x.tanh().matmul(w) + b).tanh(),
        "rhs": lambda x, w, b: (x.matmul(w.tanh()) + b).tanh(),
        "bias": lambda x, w, b: (x.matmul(w) + b.tanh()).tanh(),
    }
    for name, function in functions.items():
        traced = quabla.trace_tensor(function, specs)
        plan = traced.output.compile_cuda()
        cpu = traced.output.compile_cpu().evaluate(inputs).to_flat_list()
        assert not plan.fused_matmul_bias_tanh, name
        assert_close_rows([plan.evaluate(inputs).to_flat_list()], [cpu], tol=1e-5)


def test_cuda_two_layer_mlp_forward_and_value_and_grad_match_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    specs = [
        ("x", [4, 3]),
        ("target", [4, 2]),
        ("w1", [3, 5]),
        ("b1", [1, 5]),
        ("w2", [5, 2]),
        ("b2", [1, 2]),
    ]
    inputs = {name: _cuda_mlp_tensor(shape, seed) for seed, (name, shape) in enumerate(specs)}

    def forward(x, target, w1, b1, w2, b2):
        return ((x.matmul(w1) + b1).tanh().matmul(w2) + b2).tanh()

    traced = quabla.trace_tensor(forward, specs)
    cpu = traced.output.compile_cpu().evaluate(inputs).to_flat_list()
    cuda = traced.output.compile_cuda().evaluate(inputs).to_flat_list()
    assert_close_rows([cuda], [cpu], tol=1e-5)

    def loss(x, target, w1, b1, w2, b2):
        return (forward(x, target, w1, b1, w2, b2) - target).powi(2).mean()

    parameters = ["w1", "b1", "w2", "b2"]
    value, gradients = quabla.tensor_value_and_grad_cuda_fn(loss, specs, parameters)(inputs)
    cpu_value, cpu_gradients = quabla.tensor_value_and_grad_fn(loss, specs)(inputs)
    assert_close_rows([value.to_flat_list()], [cpu_value.to_flat_list()], tol=1e-5)
    for name in parameters:
        assert_close_rows(
            [gradients[name].to_flat_list()],
            [cpu_gradients[name].to_flat_list()],
            tol=1e-5,
        )


def test_cuda_fuses_elementwise_tail_after_matmul():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: ((x.matmul(weight) + bias).tanh()).sin(),
        [("x", [2, 2]), ("weight", [2, 3]), ("bias", [1, 3])],
    )
    inputs = {
        "x": quabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": quabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.5, 1.5, -0.5]),
        "bias": quabla.Tensor([1, 3], [0.25, -0.5, 1.0]),
    }
    cpu = traced.output.compile_cpu().evaluate(inputs).to_flat_list()
    plan = traced.output.compile_cuda()
    cuda = plan.evaluate(inputs).to_flat_list()

    assert plan.fused_region_count == 1
    assert_close_rows([cuda], [cpu], tol=1e-5)


def test_cuda_multi_parameter_adam_keeps_gradient_plans_synchronized():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    weight_gradient = gradients["weight"].output.compile_cuda()
    bias_gradient = gradients["bias"].output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([2], [-1.0, 1.0]),
        "target": quabla.Tensor([2], [-1.0, 3.0]),
        "loss_cotangent": quabla.Tensor([], [1.0]),
        "weight": quabla.Tensor([1], [0.0]),
        "bias": quabla.Tensor([1], [0.0]),
    }
    retained = ["x", "target", "loss_cotangent", "weight", "bias"]
    optimizer = quabla.cuda_adam_optimizer(
        {"weight": weight_gradient, "bias": bias_gradient},
        inputs,
        0.05,
        retained,
    )

    for _ in range(200):
        optimizer.step()

    trained = optimizer.parameters()
    assert abs(trained["weight"].to_flat_list()[0] - 2.0) < 1e-3
    assert abs(trained["bias"].to_flat_list()[0] - 1.0) < 1e-3


def test_cuda_adam_optimizer_updates_minibatch_inputs_without_resetting_state():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight: x * weight,
        [("x", [1]), ("weight", [1])],
    )
    target = traced.graph.input("target", [1])
    loss = (traced.output - target).powi(2).sum()
    gradient = loss.symbolic_vjp("loss_cotangent")["weight"].output

    def make_optimizer() -> object:
        return quabla.cuda_adam_optimizer(
            {"weight": gradient.compile_cuda()},
            {
                "x": quabla.Tensor([1], [1.0]),
                "target": quabla.Tensor([1], [1.0]),
                "loss_cotangent": quabla.Tensor([], [1.0]),
                "weight": quabla.Tensor([1], [0.0]),
            },
            0.1,
            ["x", "target", "loss_cotangent", "weight"],
        )

    static = make_optimizer()
    dynamic = make_optimizer()
    static.step()
    dynamic.step()
    static.step()
    dynamic.step({"target": quabla.Tensor([1], [-10.0])})

    static_weight = static.parameters()["weight"].to_flat_list()[0]
    dynamic_weight = dynamic.parameters()["weight"].to_flat_list()[0]
    assert abs(static_weight - dynamic_weight) > 1e-4


def test_cuda_adam_vjp_optimizer_updates_parameters_from_one_shared_graph():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    optimizer = quabla.cuda_adam_vjp_optimizer(
        {"weight": gradients["weight"], "bias": gradients["bias"]},
        {
            "x": quabla.Tensor([2], [-1.0, 1.0]),
            "target": quabla.Tensor([2], [-1.0, 3.0]),
            "loss_cotangent": quabla.Tensor([], [1.0]),
            "weight": quabla.Tensor([1], [0.0]),
            "bias": quabla.Tensor([1], [0.0]),
        },
        0.05,
        ["x", "target", "loss_cotangent", "weight", "bias"],
    )

    for _ in range(200):
        optimizer.step()

    trained = optimizer.parameters()
    assert abs(trained["weight"].to_flat_list()[0] - 2.0) < 1e-3
    assert abs(trained["bias"].to_flat_list()[0] - 1.0) < 1e-3


def test_cuda_batched_matmul_vjp_matches_cpu_trace_evaluation():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda lhs, rhs: lhs @ rhs,
        [("lhs", [2, 3, 2]), ("rhs", [1, 2, 4])],
    )
    loss = traced.output.sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    inputs = {
        "lhs": quabla.Tensor(
            [2, 3, 2],
            [1.0, -2.0, 0.5, 3.0, -1.0, 4.0, 2.0, 1.0, -3.0, 0.25, 0.75, -2.0],
        ),
        "rhs": quabla.Tensor(
            [1, 2, 4], [0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.5, -0.7]
        ),
        "loss_cotangent": quabla.Tensor([], [1.0]),
    }

    for name, gradient_trace in gradients.items():
        cpu = gradient_trace.graph.evaluate(gradient_trace.output.node_id, inputs)
        cuda = gradient_trace.output.compile_cuda().evaluate(inputs)
        assert cuda.shape == cpu.shape
        for actual, expected in zip(cuda.to_flat_list(), cpu.to_flat_list()):
            assert abs(actual - expected) < 1e-5


def test_cuda_global_reductions_match_cpu_and_reset_output_buffers():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    values = [0.125 * (index % 13) for index in range(8192)]
    inputs = {"x": quabla.Tensor([128, 64], values)}

    for fn in (lambda x: x.sum(), lambda x: x.mean()):
        traced = quabla.trace_tensor(fn, [("x", [128, 64])])
        cpu = traced.graph.evaluate(traced.output.node_id, inputs).to_flat_list()[0]
        cuda = traced.output.compile_cuda()
        first = cuda.evaluate(inputs).to_flat_list()[0]
        second = cuda.evaluate(inputs).to_flat_list()[0]

        assert abs(first - cpu) <= 1e-3
        assert abs(second - cpu) <= 1e-3


def test_multi_axis_keepdims_reductions_match_cpu_on_mlx_and_cuda():
    if os.environ.get("QUABLA_MLX_TEST") is None and os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x: x.mean(axis=[0, 2], keepdims=True), [("x", [2, 3, 2])]
    )
    inputs = {"x": quabla.Tensor([2, 3, 2], [float(value) for value in range(1, 13)])}
    cpu = traced.output.compile_cpu().evaluate(inputs)

    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx = traced.output.compile_mlx().evaluate(inputs)
        assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)

    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda = traced.output.compile_cuda().evaluate(inputs)
        assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_cuda_sqrt_composite_lowering_matches_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(lambda x: x.sqrt(), [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], [0.25, 1.0, 4.0, 9.0])}
    cpu = traced.graph.evaluate(traced.output.node_id, inputs)
    cuda = traced.output.compile_cuda().evaluate(inputs)

    for actual, expected in zip(cuda.to_flat_list(), cpu.to_flat_list()):
        assert abs(actual - expected) <= 1e-5


def test_cuda_checked_div_and_log_lowering_matches_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, scale: (x / scale).log(),
        [("x", [2, 2]), ("scale", [])],
    )
    inputs = {
        "x": quabla.Tensor([2, 2], [0.5, 1.0, 2.0, 8.0]),
        "scale": quabla.Tensor([], [0.5]),
    }
    cpu = traced.graph.evaluate(traced.output.node_id, inputs)
    cuda = traced.output.compile_cuda().evaluate(inputs)

    for actual, expected in zip(cuda.to_flat_list(), cpu.to_flat_list()):
        assert abs(actual - expected) <= 1e-5


def test_trace_tensor_tanh_scalar_loss_supports_jvp_and_vjp():
    def loss(x):
        return x.tanh().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected_derivative = [1.0 - math.tanh(item) ** 2 for item in values]
    assert abs(value.to_flat_list()[0] - sum(math.tanh(item) for item in values)) <= 1e-12
    assert_close_rows([gradients["x"].to_flat_list()], [expected_derivative])
    assert abs(tangent.to_flat_list()[0] - sum(expected_derivative)) <= 1e-12

    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)
    expected_second = [-2.0 * math.tanh(item) * derivative for item, derivative in zip(values, expected_derivative)]
    assert len(hessian) == 4
    for row in range(4):
        for col in range(4):
            expected = expected_second[row] if row == col else 0.0
            assert abs(hessian[row][col] - expected) <= 1e-12

    direction = [1.0, 2.0, 3.0, 4.0]
    hvp = traced.graph.hvp_scalar(
        traced.output.node_id, "x", inputs, quabla.Tensor([2, 2], direction)
    )
    assert_close_rows([hvp.to_flat_list()], [[second * direction_item for second, direction_item in zip(expected_second, direction)]])


def test_trace_tensor_subtraction_supports_jvp_and_vjp():
    def loss(x):
        return (x.tanh() - x).sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [-math.tanh(item) ** 2 for item in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_supports_numeric_scalar_literals():
    def loss(x):
        return (2.0 * x + 1.0).sum()

    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    assert value.to_flat_list() == [24.0]
    assert gradients["x"].to_flat_list() == [2.0] * 4
    assert tangent.to_flat_list() == [8.0]


def test_trace_tensor_division_supports_jvp_and_vjp():
    def loss(x):
        return (x / (x + 2.0)).sum()

    values = [1.0, 2.0, 3.0, 4.0]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [2.0 / (value + 2.0) ** 2 for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_exp_supports_jvp_and_vjp():
    def loss(x):
        return x.exp().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.exp(value) for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_mean_supports_jvp_and_vjp():
    def loss(x):
        return x.exp().mean()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.exp(value) / 4.0 for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_sin_supports_jvp_and_vjp():
    def loss(x):
        return x.sin().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.cos(value) for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_sqrt_supports_jvp_and_vjp():
    def loss(x):
        return x.sqrt().sum()

    values = [1.0, 4.0, 9.0, 16.0]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [0.5, 0.25, 1.0 / 6.0, 0.125]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_sqrt_and_norm_define_zero_subgradient_and_preserve_symbolic_ad():
    traced = quabla.trace_tensor(lambda x: x.sqrt().sum(), [("x", [3])])
    inputs = {"x": quabla.Tensor([3], [0.0, 1.0, 4.0])}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([3], [1.0, 1.0, 1.0])}
    )
    symbolic = traced.symbolic_vjp("loss_cotangent")["x"]

    assert gradients["x"].to_flat_list() == [0.0, 0.5, 0.25]
    assert tangent.to_flat_list() == [0.75]
    assert symbolic.graph.evaluate(
        symbolic.output.node_id,
        {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])},
    ).to_flat_list() == [0.0, 0.5, 0.25]
    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)
    assert_close_rows(
        hessian,
        [[0.0, 0.0, 0.0], [0.0, -0.25, 0.0], [0.0, 0.0, -1.0 / 32.0]],
    )

    symbolic_inputs = {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])}
    cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )
    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )

    norm = quabla.trace_tensor(lambda x: x.norm(), [("x", [2])])
    norm_inputs = {"x": quabla.Tensor([2], [3.0, 4.0])}
    value, norm_gradients = norm.graph.evaluate_value_and_vjp(
        norm.output.node_id, norm_inputs, quabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [5.0]
    assert_close_rows([norm_gradients["x"].to_flat_list()], [[0.6, 0.8]])


def test_trace_tensor_reduction_extrema_choose_last_tied_coordinate_for_gradients():
    inputs = {"x": quabla.Tensor([2, 3], [1.0, 5.0, 5.0, 2.0, 2.0, 0.0])}
    cotangent = quabla.Tensor([], [1.0])

    for operation, expected_value, expected_gradient in (
        ("max", [7.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("min", [1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ):
        traced = quabla.trace_tensor(
            lambda x: getattr(x, operation)(axis=1).sum(), [("x", [2, 3])]
        )
        value, gradients = traced.graph.evaluate_value_and_vjp(
            traced.output.node_id, inputs, cotangent
        )
        symbolic = traced.symbolic_vjp("loss_cotangent")["x"]
        symbolic_inputs = {**inputs, "loss_cotangent": cotangent}

        assert value.to_flat_list() == expected_value
        assert gradients["x"].to_flat_list() == expected_gradient
        assert symbolic.graph.evaluate(
            symbolic.output.node_id, symbolic_inputs
        ).to_flat_list() == expected_gradient

        cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
        if os.environ.get("QUABLA_MLX_TEST") is not None:
            mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
            assert_close_rows(
                [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
            )
        if os.environ.get("QUABLA_CUDA_TEST") is not None:
            cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
            assert_close_rows(
                [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
            )


def test_trace_tensor_gather_and_scatter_add_preserve_repeated_index_gradients():
    gather = quabla.trace_tensor(
        lambda x: x.gather([2, 0, 2], axis=1).sum(), [("x", [2, 3])]
    )
    gather_inputs = {"x": quabla.Tensor([2, 3], [0.0] * 6)}
    _, gather_gradients = gather.graph.evaluate_value_and_vjp(
        gather.output.node_id, gather_inputs, quabla.Tensor([], [1.0])
    )
    assert gather_gradients["x"].to_flat_list() == [1.0, 0.0, 2.0, 1.0, 0.0, 2.0]

    scatter = quabla.trace_tensor(
        lambda base, updates, weight: (base.scatter_add([1, 1, 0], updates, axis=1) * weight).sum(),
        [("base", [1, 3]), ("updates", [1, 3]), ("weight", [1, 3])],
    )
    scatter_inputs = {
        "base": quabla.Tensor.zeros([1, 3]),
        "updates": quabla.Tensor([1, 3], [2.0, 3.0, 5.0]),
        "weight": quabla.Tensor([1, 3], [1.0, 2.0, 4.0]),
    }
    _, scatter_gradients = scatter.graph.evaluate_value_and_vjp(
        scatter.output.node_id, scatter_inputs, quabla.Tensor([], [1.0])
    )
    assert scatter_gradients["base"].to_flat_list() == [1.0, 2.0, 4.0]
    assert scatter_gradients["updates"].to_flat_list() == [2.0, 2.0, 1.0]

    symbolic = scatter.symbolic_vjp("loss_cotangent")["updates"]
    symbolic_inputs = {**scatter_inputs, "loss_cotangent": quabla.Tensor([], [1.0])}
    assert symbolic.graph.evaluate(
        symbolic.output.node_id, symbolic_inputs
    ).to_flat_list() == [2.0, 2.0, 1.0]

    cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )
    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )


def test_trace_tensor_einsum_scoped_matmul_subset_preserves_backend_ad():
    traced = quabla.trace_tensor(
        lambda left, right: quabla.einsum("ij,jk->ik", [left, right]).sum(),
        [("left", [2, 2]), ("right", [2, 2])],
    )
    inputs = {
        "left": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": quabla.Tensor([2, 2], [1.0, 0.0, 2.0, 1.0]),
    }
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    assert gradients["left"].to_flat_list() == [1.0, 3.0, 1.0, 3.0]
    assert gradients["right"].to_flat_list() == [4.0, 4.0, 6.0, 6.0]

    symbolic = traced.symbolic_vjp("loss_cotangent")["left"]
    symbolic_inputs = {**inputs, "loss_cotangent": quabla.Tensor([], [1.0])}
    cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
    assert cpu_gradient.to_flat_list() == gradients["left"].to_flat_list()
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )
    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )


def test_trace_tensor_where_routes_gradients_and_masks_condition_derivatives():
    def loss(x):
        condition = x.gt(0.0)
        return quabla.where(condition, x.powi(2), x * 3.0).sum()

    traced = quabla.trace_tensor(loss, [("x", [4])])
    inputs = {"x": quabla.Tensor([4], [-2.0, -1.0, 0.0, 2.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([4], [1.0] * 4)}
    )
    transformed = traced.symbolic_jvp("x")
    transformed_value = transformed.graph.evaluate(transformed.output.node_id, inputs)
    plan = traced.output.compile_cpu()
    plan_value, plan_gradients = plan.evaluate_value_and_vjp(
        inputs, quabla.Tensor([], [1.0])
    )

    assert value.to_flat_list() == [-5.0]
    assert gradients["x"].to_flat_list() == [3.0, 3.0, 3.0, 4.0]
    assert tangent.to_flat_list() == [13.0]
    assert transformed_value.to_flat_list() == [13.0]
    assert plan_value.to_flat_list() == value.to_flat_list()
    assert plan_gradients["x"].to_flat_list() == gradients["x"].to_flat_list()


def test_tensor_cpu_plan_data_parallel_value_and_grad_matches_single_plan():
    traced = quabla.trace_tensor(
        lambda x, target, weight: ((x * weight - target).powi(2)).mean(),
        [("x", [4, 1]), ("target", [4, 1]), ("weight", [1, 1])],
    )
    plan = traced.output.compile_cpu()
    inputs = {
        "x": quabla.Tensor([4, 1], [-2.0, -1.0, 1.0, 2.0]),
        "target": quabla.Tensor([4, 1], [-3.0, -1.0, 3.0, 5.0]),
        "weight": quabla.Tensor([1, 1], [0.5]),
    }
    single_value, single_gradients = plan.evaluate_value_and_vjp(
        inputs, quabla.Tensor([], [1.0])
    )
    parallel_value, parallel_gradients = plan.value_and_grad_data_parallel(
        inputs, ["x", "target"], 2, "mean"
    )

    assert parallel_value.to_flat_list() == single_value.to_flat_list()
    assert parallel_gradients["x"].to_flat_list() == single_gradients["x"].to_flat_list()
    assert parallel_gradients["target"].to_flat_list() == single_gradients["target"].to_flat_list()
    assert parallel_gradients["weight"].to_flat_list() == single_gradients["weight"].to_flat_list()


def test_tensor_cuda_data_parallel_constructor_rejects_unavailable_collective_runtime():
    # The rejection only happens where the collective runtime is missing (no
    # cuda-nccl build, fewer than two GPUs, or no loadable NCCL). Hosts that
    # opt into the two-GPU suite have that runtime, so construction succeeds.
    if os.environ.get("QUABLA_CUDA_NCCL_TEST") is not None:
        return
    try:
        quabla.tensor_value_and_grad_data_parallel_cuda_fn(
            lambda x, weight: ((x * weight).powi(2)).mean(),
            [("x", [4, 1]), ("weight", [1, 1])],
            ["weight"],
            ["x"],
            [0, 1],
        )
    except ValueError as error:
        message = str(error).lower()
        assert (
            "cuda-nccl" in message
            or "data-parallel backend is unavailable" in message
            or "cuda reports" in message
        )
    else:
        raise AssertionError("expected unavailable CUDA collective runtime to be rejected")


def test_trace_tensor_maximum_and_minimum_route_tie_gradients_to_rhs():
    left = quabla.Tensor([3], [2.0, 3.0, 0.0])
    right = quabla.Tensor([3], [2.0, 1.0, 4.0])
    cotangent = quabla.Tensor([], [1.0])

    for operation, expected_value, expected_left, expected_right in (
        ("maximum", [2.0, 3.0, 4.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]),
        ("minimum", [2.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]),
    ):
        traced = quabla.trace_tensor(
            lambda x, y: getattr(x, operation)(y).sum(),
            [("x", [3]), ("y", [3])],
        )
        inputs = {"x": left, "y": right}
        value, gradients = traced.graph.evaluate_value_and_vjp(
            traced.output.node_id, inputs, cotangent
        )
        symbolic = traced.symbolic_vjp("loss_cotangent")
        symbolic_inputs = {**inputs, "loss_cotangent": cotangent}

        assert value.to_flat_list() == [sum(expected_value)]
        assert gradients["x"].to_flat_list() == expected_left
        assert gradients["y"].to_flat_list() == expected_right
        assert symbolic["x"].graph.evaluate(
            symbolic["x"].output.node_id, symbolic_inputs
        ).to_flat_list() == expected_left
        assert symbolic["y"].graph.evaluate(
            symbolic["y"].output.node_id, symbolic_inputs
        ).to_flat_list() == expected_right

        cpu = traced.output.compile_cpu().evaluate(inputs)
        if os.environ.get("QUABLA_MLX_TEST") is not None:
            mlx = traced.output.compile_mlx().evaluate(inputs)
            assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)
        if os.environ.get("QUABLA_CUDA_TEST") is not None:
            cuda = traced.output.compile_cuda().evaluate(inputs)
            assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_trace_tensor_log_supports_jvp_and_vjp():
    def loss(x):
        return x.log().sum()

    values = [1.0, 2.0, 3.0, 4.0]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [1.0 / value for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_cos_supports_jvp_and_vjp():
    def loss(x):
        return x.cos().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )
    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)
    hvp = traced.graph.hvp_scalar(
        traced.output.node_id,
        "x",
        inputs,
        quabla.Tensor([2, 2], [1.0, 2.0, -1.0, 0.5]),
    )

    expected = [-math.sin(value) for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12
    expected_diagonal = [-math.cos(value) for value in values]
    assert_close_rows(
        hessian,
        [
            [expected_diagonal[row] if row == column else 0.0 for column in range(4)]
            for row in range(4)
        ],
    )
    assert_close_rows(
        [hvp.to_flat_list()],
        [[expected_diagonal[index] * value for index, value in enumerate([1.0, 2.0, -1.0, 0.5])]],
    )


def test_trace_tensor_powi_supports_second_order_ad():
    def loss(x):
        return x.powi(3).sum()

    values = [0.0, 1.0, -2.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )
    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)

    expected_gradient = [3.0 * value * value for value in values]
    expected_diagonal = [6.0 * value for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected_gradient])
    assert abs(tangent.to_flat_list()[0] - sum(expected_gradient)) <= 1e-12
    assert_close_rows(
        hessian,
        [
            [expected_diagonal[row] if row == column else 0.0 for column in range(4)]
            for row in range(4)
        ],
    )


def test_trace_tensor_axis_reductions_support_ad():
    def loss(x):
        return x.mean(axis=0).powi(2).sum()

    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    traced = quabla.trace_tensor(loss, [("x", [2, 3])])
    inputs = {"x": quabla.Tensor([2, 3], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 3], [1.0] * 6)}
    )
    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)

    expected_gradient = [2.5, 3.5, 4.5, 2.5, 3.5, 4.5]
    assert_close_rows([gradients["x"].to_flat_list()], [expected_gradient])
    assert abs(tangent.to_flat_list()[0] - 21.0) <= 1e-12
    assert_close_rows(
        hessian,
        [
            [0.5 if row % 3 == column % 3 else 0.0 for column in range(6)]
            for row in range(6)
        ],
    )

    output = quabla.trace_tensor(lambda x: x.sum(axis=1), [("x", [2, 3])])
    value, output_tangent = output.graph.evaluate_jvp(
        output.output.node_id,
        inputs,
        {"x": quabla.Tensor([2, 3], [1.0] * 6)},
    )
    _, output_gradients = output.graph.evaluate_value_and_vjp(
        output.output.node_id, inputs, quabla.Tensor([2], [2.0, 3.0])
    )
    assert value.shape == [2]
    assert value.to_flat_list() == [6.0, 15.0]
    assert output_tangent.to_flat_list() == [3.0, 3.0]
    assert output_gradients["x"].to_flat_list() == [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]

    multi_axis = quabla.trace_tensor(
        lambda x: x.mean(axis=[0, 2], keepdims=True), [("x", [2, 3, 2])]
    )
    multi_inputs = {"x": quabla.Tensor([2, 3, 2], [float(value) for value in range(1, 13)])}
    multi_value, multi_gradients = multi_axis.graph.evaluate_value_and_vjp(
        multi_axis.output.node_id,
        multi_inputs,
        quabla.Tensor([1, 3, 1], [2.0, 3.0, 4.0]),
    )
    assert multi_value.shape == [1, 3, 1]
    assert multi_value.to_flat_list() == [4.5, 6.5, 8.5]
    assert multi_gradients["x"].to_flat_list() == [
        0.5,
        0.5,
        0.75,
        0.75,
        1.0,
        1.0,
        0.5,
        0.5,
        0.75,
        0.75,
        1.0,
        1.0,
    ]

    last_axis = quabla.trace_tensor(lambda x: x.mean(axis=-1), [("x", [2, 3])])
    last_axis_value = last_axis.graph.evaluate(last_axis.output.node_id, inputs)
    assert last_axis_value.shape == [2]
    assert last_axis_value.to_flat_list() == [2.0, 5.0]


def test_trace_tensor_transpose_supports_rank_n_ad():
    def loss(x):
        return x.transpose([2, 0, 1]).powi(2).sum()

    values = [float(value) for value in range(1, 13)]
    traced = quabla.trace_tensor(loss, [("x", [2, 2, 3])])
    inputs = {"x": quabla.Tensor([2, 2, 3], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2, 3], [1.0] * 12)}
    )

    expected = [2.0 * value for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12

    output = quabla.trace_tensor(lambda x: x.transpose(), [("x", [2, 2, 3])])
    value = output.graph.evaluate(output.output.node_id, inputs)
    assert value.shape == [3, 2, 2]
    assert value.to_flat_list() == [1.0, 7.0, 4.0, 10.0, 2.0, 8.0, 5.0, 11.0, 3.0, 9.0, 6.0, 12.0]


def test_trace_tensor_reshape_supports_jvp_and_vjp():
    def loss(x):
        return x.reshape([4]).exp().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = quabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": quabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.exp(value) for value in values]
    assert gradients["x"].shape == [2, 2]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_tensor_grad_scalar_fn_reuses_a_compiled_plan():
    gradient = quabla.tensor_grad_scalar_fn(
        lambda x: (2.0 * x).sum(), [("x", [2, 2])]
    )

    first = gradient({"x": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])})
    second = gradient({"x": quabla.Tensor([2, 2], [5.0, 6.0, 7.0, 8.0])})

    assert first["x"].to_flat_list() == [2.0] * 4
    assert second["x"].to_flat_list() == [2.0] * 4


def test_tensor_value_and_grad_fn_returns_scalar_value_and_gradients():
    value_and_grad = quabla.tensor_value_and_grad_fn(
        lambda x, scale: ((x * scale).sin()).sum(),
        [("x", [2, 2]), ("scale", [])],
    )
    first_value, first_gradients = value_and_grad(
        {
            "x": quabla.Tensor([2, 2], [0.0, 0.5, 1.0, -0.25]),
            "scale": quabla.Tensor([], [2.0]),
        }
    )
    second_value, second_gradients = value_and_grad(
        {
            "x": quabla.Tensor([2, 2], [0.0, 0.5, 1.0, -0.25]),
            "scale": quabla.Tensor([], [1.0]),
        }
    )

    values = [0.0, 0.5, 1.0, -0.25]
    assert abs(first_value.to_flat_list()[0] - sum(math.sin(2.0 * value) for value in values)) < 1e-12
    assert_close_rows(
        [first_gradients["x"].to_flat_list()],
        [[2.0 * math.cos(2.0 * value) for value in values]],
    )
    assert abs(
        first_gradients["scale"].to_flat_list()[0]
        - sum(value * math.cos(2.0 * value) for value in values)
    ) < 1e-12
    assert abs(second_value.to_flat_list()[0] - sum(math.sin(value) for value in values)) < 1e-12
    assert_close_rows(
        [second_gradients["x"].to_flat_list()],
        [[math.cos(value) for value in values]],
    )


def test_tensor_hessian_and_hvp_scalar_fn_reuse_a_compiled_plan():
    hessian = quabla.tensor_hessian_scalar_fn(
        lambda x: x.powi(3).sum(), [("x", [2, 2])], "x"
    )
    hvp = quabla.tensor_hvp_scalar_fn(
        lambda x: x.powi(3).sum(), [("x", [2, 2])], "x"
    )
    values = [0.5, -1.0, 2.0, 0.25]
    direction = [1.0, 2.0, -0.5, 3.0]
    inputs = {"x": quabla.Tensor([2, 2], values)}

    actual_hessian = hessian(inputs)
    actual_hvp = hvp(inputs, quabla.Tensor([2, 2], direction))

    assert len(actual_hessian) == len(values)
    for row, value in enumerate(values):
        for column in range(len(values)):
            expected = 6.0 * value if row == column else 0.0
            assert abs(actual_hessian[row][column] - expected) <= 1e-12
    assert_close_rows(
        [actual_hvp.to_flat_list()],
        [[6.0 * value * tangent for value, tangent in zip(values, direction)]],
    )


def test_tensor_hessian_and_hvp_scalar_fn_support_fixed_fori_regions():
    def loop_loss(initial, scale):
        return quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        )

    hessian = quabla.tensor_hessian_scalar_fn(
        loop_loss, [("initial", []), ("scale", [])], "scale"
    )
    hvp = quabla.tensor_hvp_scalar_fn(
        loop_loss, [("initial", []), ("scale", [])], "scale"
    )
    inputs = {
        "initial": quabla.Tensor([], [1.0]),
        "scale": quabla.Tensor([], [2.0]),
    }

    assert hessian(inputs) == [[12.0]]
    assert hvp(inputs, quabla.Tensor([], [3.0])).to_flat_list() == [36.0]


def test_tensor_hvp_scalar_fn_supports_nonlinear_scan_regions():
    def scan_loss(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * current * captured_scale + index,
                (current * current * captured_scale + index) * captured_scale,
            ),
            initial,
            [scale],
        )
        return carry + outputs.sum()

    traced = quabla.trace_tensor(scan_loss, [("initial", []), ("scale", [])])
    hvp = quabla.tensor_hvp_scalar_fn(
        scan_loss, [("initial", []), ("scale", [])], "scale"
    )
    inputs = {
        "initial": quabla.Tensor([], [0.4]),
        "scale": quabla.Tensor([], [0.8]),
    }
    actual = hvp(inputs, quabla.Tensor([], [1.0])).to_flat_list()[0]
    step = 1e-4

    def evaluate(scale):
        return traced.graph.evaluate(
            traced.output.node_id,
            {"initial": quabla.Tensor([], [0.4]), "scale": quabla.Tensor([], [scale])},
        ).to_flat_list()[0]

    expected = (evaluate(0.8 + step) - 2.0 * evaluate(0.8) + evaluate(0.8 - step)) / (step * step)
    assert abs(actual - expected) <= 2e-5


def test_tensor_jit_fn_reuses_a_compiled_plan():
    compiled = quabla.tensor_jit_fn(
        lambda x, y: (x.matmul(y)).tanh(),
        [("x", [2, 2]), ("y", [2, 2])],
    )

    first = compiled(
        {
            "x": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
            "y": quabla.Tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
        }
    )
    second = compiled(
        {
            "x": quabla.Tensor([2, 2], [0.0, 1.0, -1.0, 0.5]),
            "y": quabla.Tensor([2, 2], [2.0, 0.0, 0.0, 2.0]),
        }
    )

    assert_close_rows(
        [first.to_flat_list()],
        [[math.tanh(1.0), math.tanh(2.0), math.tanh(3.0), math.tanh(4.0)]],
    )
    assert_close_rows(
        [second.to_flat_list()],
        [[0.0, math.tanh(2.0), math.tanh(-2.0), math.tanh(1.0)]],
    )


def test_trace_tensor_rejects_data_dependent_python_branches():
    def model(x):
        if x.gt(0.0):
            return x
        return x * -1.0

    try:
        quabla.trace_tensor(model, [("x", [1])])
        assert False, "expected traced Python branch rejection"
    except TypeError as error:
        assert "cannot drive Python control flow" in str(error)


def test_tensor_fori_loop_unrolls_differentiable_carry():
    traced = quabla.trace_tensor(
        lambda x: quabla.tensor_fori_loop(0, 3, lambda _, carry: carry * x, x),
        [("x", [1])],
    )
    inputs = {"x": quabla.Tensor([1], [2.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([1], [1.0])
    )

    assert value.to_flat_list() == [16.0]
    assert gradients["x"].to_flat_list() == [32.0]
    try:
        quabla.tensor_fori_loop(2, 1, lambda _, carry: carry, traced.output)
        assert False, "expected invalid loop bounds"
    except ValueError as error:
        assert "upper >= lower" in str(error)


def test_tensor_fori_loop_region_traces_once_and_differentiates_parent_inputs():
    traced = quabla.trace_tensor(
        lambda initial, scale: quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        ).sum(),
        [("initial", [1]), ("scale", [1])],
    )
    inputs = {
        "initial": quabla.Tensor([1], [1.0]),
        "scale": quabla.Tensor([1], [2.0]),
    }
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        inputs,
        {
            "initial": quabla.Tensor([1], [1.0]),
            "scale": quabla.Tensor([1], [0.0]),
        },
    )

    assert value.to_flat_list() == [12.0]
    assert tangent.to_flat_list() == [8.0]
    assert gradients["initial"].to_flat_list() == [8.0]
    assert gradients["scale"].to_flat_list() == [13.0]


def test_tensor_fori_loop_region_supports_symbolic_jvp():
    traced = quabla.trace_tensor(
        lambda initial, scale: quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        ),
        [("initial", []), ("scale", [])],
    )
    tangent = traced.symbolic_jvp("initial")
    value = tangent.graph.evaluate(
        tangent.output.node_id,
        {
            "initial": quabla.Tensor([], [1.0]),
            "scale": quabla.Tensor([], [2.0]),
        },
    )
    assert value.to_flat_list() == [8.0]


def test_tensor_fori_loop_region_supports_symbolic_vjp():
    traced = quabla.trace_tensor(
        lambda initial, scale: quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        ),
        [("initial", []), ("scale", [])],
    )
    gradients = traced.symbolic_vjp("seed")
    inputs = {
        "initial": quabla.Tensor([], [1.0]),
        "scale": quabla.Tensor([], [2.0]),
        "seed": quabla.Tensor([], [1.0]),
    }
    assert (
        gradients["initial"].graph.evaluate(gradients["initial"].output.node_id, inputs).to_flat_list()
        == [8.0]
    )
    assert (
        gradients["scale"].graph.evaluate(gradients["scale"].output.node_id, inputs).to_flat_list()
        == [13.0]
    )


def test_tensor_scan_unrolls_differentiable_outputs():
    traced = quabla.trace_tensor(
        lambda x: quabla.tensor_scan(3, lambda _, carry: (carry * x, carry * x), x)[1].sum(),
        [("x", [1])],
    )
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id,
        {"x": quabla.Tensor([1], [2.0])},
        quabla.Tensor([], [1.0]),
    )
    assert gradients["x"].to_flat_list() == [48.0]


def test_tensor_scan_region_traces_once_and_differentiates_carry_and_outputs():
    def scan_total(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                current * captured_scale + index,
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    traced = quabla.trace_tensor(scan_total, [("initial", [1]), ("scale", [1])])
    inputs = {
        "initial": quabla.Tensor([1], [1.0]),
        "scale": quabla.Tensor([1], [2.0]),
    }
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, quabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        inputs,
        {
            "initial": quabla.Tensor([1], [0.0]),
            "scale": quabla.Tensor([1], [1.0]),
        },
    )

    assert value.to_flat_list() == [31.0]
    assert tangent.to_flat_list() == [31.0]
    assert gradients["initial"].to_flat_list() == [22.0]
    assert gradients["scale"].to_flat_list() == [31.0]


def test_tensor_scan_region_supports_symbolic_jvp():
    def scan_total(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                current * captured_scale + index,
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    traced = quabla.trace_tensor(scan_total, [("initial", []), ("scale", [])])
    tangent = traced.symbolic_jvp("initial")
    value = tangent.graph.evaluate(
        tangent.output.node_id,
        {
            "initial": quabla.Tensor([], [1.0]),
            "scale": quabla.Tensor([], [2.0]),
        },
    )
    assert value.to_flat_list() == [22.0]


def test_tensor_scan_region_supports_symbolic_vjp():
    def scan_total(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                current * captured_scale + index,
            ),
            initial,
            [scale],
        )
        return carry.sum() + outputs.sum()

    traced = quabla.trace_tensor(scan_total, [("initial", []), ("scale", [])])
    gradients = traced.symbolic_vjp("seed")
    inputs = {
        "initial": quabla.Tensor([], [1.0]),
        "scale": quabla.Tensor([], [2.0]),
        "seed": quabla.Tensor([], [1.0]),
    }
    assert (
        gradients["initial"].graph.evaluate(gradients["initial"].output.node_id, inputs).to_flat_list()
        == [22.0]
    )
    assert (
        gradients["scale"].graph.evaluate(gradients["scale"].output.node_id, inputs).to_flat_list()
        == [31.0]
    )


def test_tensor_cond_fn_only_executes_selected_branch():
    condition = quabla.tensor_cond_fn(
        lambda x: x.log(),
        lambda x: x * 2.0,
        [("x", [1])],
    )
    negative = {"x": quabla.Tensor([1], [-2.0])}
    assert condition(False, negative).to_flat_list() == [-4.0]
    try:
        condition(True, negative)
        assert False, "expected selected log branch to reject a negative input"
    except ValueError as error:
        assert "log" in str(error)

    try:
        quabla.tensor_cond_fn(lambda x: x, lambda x: x.sum(), [("x", [1])])
        assert False, "expected mismatched branch output shapes"
    except ValueError as error:
        assert "branch output shapes differ" in str(error)


def test_tensor_cond_value_and_grad_selects_matching_branch_vjp():
    condition = quabla.tensor_cond_value_and_grad_fn(
        lambda x: (x * x).sum(),
        lambda x: x.powi(3).sum(),
        [("x", [1])],
    )
    values = {"x": quabla.Tensor([1], [3.0])}
    true_value, true_gradients = condition(True, values)
    false_value, false_gradients = condition(False, values)
    assert true_value.to_flat_list() == [9.0]
    assert true_gradients["x"].to_flat_list() == [6.0]
    assert false_value.to_flat_list() == [27.0]
    assert false_gradients["x"].to_flat_list() == [27.0]


def test_tensor_cond_jvp_selects_matching_branch_direction():
    condition = quabla.tensor_cond_jvp_fn(
        lambda x: x * x,
        lambda x: x.powi(3),
        [("x", [1])],
    )
    values = {"x": quabla.Tensor([1], [3.0])}
    tangents = {"x": quabla.Tensor([1], [2.0])}
    true_value, true_tangent = condition(True, values, tangents)
    false_value, false_tangent = condition(False, values, tangents)
    assert true_value.to_flat_list() == [9.0]
    assert true_tangent.to_flat_list() == [12.0]
    assert false_value.to_flat_list() == [27.0]
    assert false_tangent.to_flat_list() == [54.0]


def test_tensor_cond_traces_a_lazy_tensor_predicate_with_explicit_captures():
    traced = quabla.trace_tensor(
        lambda predicate, x: quabla.tensor_cond(
            predicate,
            lambda captured: captured * captured,
            lambda captured: captured * 3.0,
            [x + x],
        ),
        [("predicate", []), ("x", [])],
    )
    true_inputs = {
        "predicate": quabla.Tensor([], [1.0]),
        "x": quabla.Tensor([], [2.0]),
    }
    true_value, true_gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, true_inputs, quabla.Tensor([], [1.0])
    )
    assert true_value.to_flat_list() == [16.0]
    assert true_gradients["x"].to_flat_list() == [16.0]
    assert true_gradients["predicate"].to_flat_list() == [0.0]

    false_inputs = {
        "predicate": quabla.Tensor([], [0.0]),
        "x": quabla.Tensor([], [2.0]),
    }
    false_value, false_gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, false_inputs, quabla.Tensor([], [1.0])
    )
    assert false_value.to_flat_list() == [12.0]
    assert false_gradients["x"].to_flat_list() == [6.0]


def test_tensor_cond_supports_nested_tensor_predicates():
    traced = quabla.trace_tensor(
        lambda outer, inner, x: quabla.tensor_cond(
            outer,
            lambda inner_predicate, captured: quabla.tensor_cond(
                inner_predicate,
                lambda nested: nested * nested,
                lambda nested: nested * 3.0,
                [captured],
            ),
            lambda inner_predicate, captured: captured + inner_predicate * 0.0,
            [inner, x],
        ),
        [("outer", []), ("inner", []), ("x", [])],
    )
    for outer, inner, expected_value, expected_gradient in [
        (1.0, 1.0, 4.0, 4.0),
        (1.0, 0.0, 6.0, 3.0),
        (0.0, 1.0, 2.0, 1.0),
    ]:
        value, gradients = traced.graph.evaluate_value_and_vjp(
            traced.output.node_id,
            {
                "outer": quabla.Tensor([], [outer]),
                "inner": quabla.Tensor([], [inner]),
                "x": quabla.Tensor([], [2.0]),
            },
            quabla.Tensor([], [1.0]),
        )
        assert value.to_flat_list() == [expected_value]
        assert gradients["x"].to_flat_list() == [expected_gradient]


def test_tensor_cond_symbolic_ad_supports_nested_regions():
    traced = quabla.trace_tensor(
        lambda outer, inner, x: quabla.tensor_cond(
            outer,
            lambda inner_predicate, captured: quabla.tensor_cond(
                inner_predicate,
                lambda nested: nested * nested,
                lambda nested: nested * 3.0,
                [captured],
            ),
            lambda inner_predicate, captured: captured + inner_predicate * 0.0,
            [inner, x],
        ),
        [("outer", []), ("inner", []), ("x", [])],
    )
    symbolic_jvp = traced.symbolic_jvp("x")
    symbolic_vjp = traced.symbolic_vjp("seed")["x"]
    for outer, inner, expected_gradient in [
        (1.0, 1.0, 4.0),
        (1.0, 0.0, 3.0),
        (0.0, 1.0, 1.0),
    ]:
        inputs = {
            "outer": quabla.Tensor([], [outer]),
            "inner": quabla.Tensor([], [inner]),
            "x": quabla.Tensor([], [2.0]),
        }
        jvp_value = symbolic_jvp.graph.evaluate(symbolic_jvp.output.node_id, inputs)
        assert jvp_value.to_flat_list() == [expected_gradient]
        vjp_value = symbolic_vjp.graph.evaluate(
            symbolic_vjp.output.node_id,
            {**inputs, "seed": quabla.Tensor([], [1.0])},
        )
        assert vjp_value.to_flat_list() == [expected_gradient]


def test_tensor_cond_supports_second_order_ad_through_symbolic_regions():
    traced = quabla.trace_tensor(
        lambda predicate, x: quabla.tensor_cond(
            predicate,
            lambda captured: captured * captured,
            lambda captured: captured * 3.0,
            [x + x],
        ),
        [("predicate", []), ("x", [])],
    )
    for predicate, expected in [(1.0, 8.0), (0.0, 0.0)]:
        inputs = {
            "predicate": quabla.Tensor([], [predicate]),
            "x": quabla.Tensor([], [2.0]),
        }
        assert traced.graph.hessian_scalar(traced.output.node_id, "x", inputs) == [[expected]]
        hvp = traced.graph.hvp_scalar(
            traced.output.node_id, "x", inputs, quabla.Tensor([], [1.0])
        )
        assert hvp.to_flat_list() == [expected]


def device_cond_compilers():
    compilers = []
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        compilers.append(("mlx", lambda output: output.compile_mlx()))
    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        compilers.append(("cuda", lambda output: output.compile_cuda()))
    return compilers


def test_device_tensor_cond_matches_cpu_without_inactive_branch_nan():
    traced = quabla.trace_tensor(
        lambda x, y: quabla.tensor_cond(
            x.gt(0.0),
            lambda a, b: b * a.log(),
            lambda a, b: b * b + a * 0.0,
            [x, y],
        ).sum(),
        [("x", []), ("y", [3])],
    )
    gradients = traced.symbolic_vjp("seed")
    tangent = traced.symbolic_jvp("x")
    outputs = [
        ("value", traced.output),
        ("vjp_x", gradients["x"].output),
        ("vjp_y", gradients["y"].output),
        ("jvp_x", tangent.output),
    ]
    for x in (2.0, -1.0):
        inputs = {
            "x": quabla.Tensor([], [x]),
            "y": quabla.Tensor([3], [1.0, 2.0, 3.0]),
            "seed": quabla.Tensor([], [1.0]),
        }
        for label, output in outputs:
            cpu = output.compile_cpu().evaluate(inputs).to_flat_list()
            assert all(math.isfinite(value) for value in cpu), label
            for backend, compile_device in device_cond_compilers():
                device = compile_device(output).evaluate(inputs).to_flat_list()
                assert all(math.isfinite(value) for value in device), (backend, label, x)
                assert_close_rows([device], [cpu], tol=1e-4)
    negative = {
        "x": quabla.Tensor([], [-1.0]),
        "y": quabla.Tensor([3], [1.0, 2.0, 3.0]),
        "seed": quabla.Tensor([], [1.0]),
    }
    assert gradients["x"].output.compile_cpu().evaluate(negative).to_flat_list() == [0.0]
    assert gradients["y"].output.compile_cpu().evaluate(negative).to_flat_list() == [
        2.0,
        4.0,
        6.0,
    ]


def test_device_tensor_cond_matches_cpu_for_nested_regions():
    traced = quabla.trace_tensor(
        lambda outer, inner, x: quabla.tensor_cond(
            outer,
            lambda inner_predicate, captured: quabla.tensor_cond(
                inner_predicate,
                lambda nested: nested * nested,
                lambda nested: nested * 3.0,
                [captured],
            ),
            lambda inner_predicate, captured: captured + inner_predicate * 0.0,
            [inner, x],
        ),
        [("outer", []), ("inner", []), ("x", [])],
    )
    gradient = traced.symbolic_vjp("seed")["x"].output
    for outer, inner, expected_value, expected_gradient in [
        (1.0, 1.0, 4.0, 4.0),
        (1.0, 0.0, 6.0, 3.0),
        (0.0, 1.0, 2.0, 1.0),
    ]:
        inputs = {
            "outer": quabla.Tensor([], [outer]),
            "inner": quabla.Tensor([], [inner]),
            "x": quabla.Tensor([], [2.0]),
            "seed": quabla.Tensor([], [1.0]),
        }
        for _, compile_device in device_cond_compilers():
            assert_close_rows(
                [compile_device(traced.output).evaluate(inputs).to_flat_list()],
                [[expected_value]],
                tol=1e-5,
            )
            assert_close_rows(
                [compile_device(gradient).evaluate(inputs).to_flat_list()],
                [[expected_gradient]],
                tol=1e-5,
            )


def test_tensor_cond_rejects_vmapped_predicates_before_device_lowering():
    vmap_functions = [quabla.tensor_vmap_fn]
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        vmap_functions.append(quabla.tensor_vmap_mlx_fn)
    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        vmap_functions.append(quabla.tensor_vmap_cuda_fn)
    for vmap_function in vmap_functions:
        try:
            vmap_function(
                lambda x, flag: quabla.tensor_cond(
                    flag, lambda a: a, lambda a: a * 2.0, [x]
                ),
                [("x", [2]), ("flag", [])],
                3,
                in_axes=[0, None],
            )
            assert False, "expected vmapped tensor_cond to reject"
        except ValueError as error:
            assert "vmapped predicates" in str(error)


def test_tensor_jit_batch_fn_bounds_batch_shape_specialization():
    compiled = quabla.tensor_jit_batch_fn(
        lambda x, weight: (x.matmul(weight)).tanh(),
        ["x", "weight"],
        in_axes=[0, None],
        max_specializations=2,
    )
    weight = quabla.Tensor([1, 1], [2.0])
    first = compiled({"x": quabla.Tensor([2, 1], [0.5, -1.0]), "weight": weight})
    second = compiled(
        {"x": quabla.Tensor([3, 1], [0.0, 1.0, -0.5]), "weight": weight}
    )

    assert_close_rows(
        [first.to_flat_list()], [[math.tanh(1.0), math.tanh(-2.0)]]
    )
    assert_close_rows(
        [second.to_flat_list()], [[0.0, math.tanh(2.0), math.tanh(-1.0)]]
    )
    assert compiled.specialization_count == 2
    try:
        compiled({"x": quabla.Tensor([4, 1], [0.0] * 4), "weight": weight})
        assert False, "expected bounded specialization error"
    except ValueError as error:
        assert "max_specializations" in str(error)
    try:
        compiled({"x": quabla.Tensor([2, 2], [0.0] * 4), "weight": weight})
        assert False, "expected non-batch shape error"
    except ValueError as error:
        assert "non-batch shapes" in str(error)


def test_tensor_value_and_grad_batch_fn_specializes_collocation_batches():
    compiled = quabla.tensor_value_and_grad_batch_fn(
        lambda x, weight: (x * weight).powi(2).mean(),
        ["x", "weight"],
        in_axes=[0, None],
        max_specializations=2,
    )
    weight = quabla.Tensor([1], [2.0])

    first_value, first_gradients = compiled(
        {"x": quabla.Tensor([2, 1], [1.0, 2.0]), "weight": weight}
    )
    second_value, second_gradients = compiled(
        {"x": quabla.Tensor([3, 1], [1.0, 2.0, 3.0]), "weight": weight}
    )

    assert_close_rows([first_value.to_flat_list()], [[10.0]])
    assert_close_rows([first_gradients["x"].to_flat_list()], [[4.0, 8.0]])
    assert_close_rows([first_gradients["weight"].to_flat_list()], [[10.0]])
    assert_close_rows([second_value.to_flat_list()], [[56.0 / 3.0]])
    assert_close_rows(
        [second_gradients["x"].to_flat_list()], [[8.0 / 3.0, 16.0 / 3.0, 8.0]]
    )
    assert_close_rows([second_gradients["weight"].to_flat_list()], [[56.0 / 3.0]])
    assert compiled.specialization_count == 2
    try:
        compiled({"x": quabla.Tensor([4, 1], [1.0] * 4), "weight": weight})
        assert False, "expected bounded specialization error"
    except ValueError as error:
        assert "max_specializations" in str(error)


def test_tensor_jit_batch_cuda_fn_specializes_bounded_batches():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    compiled = quabla.tensor_jit_batch_cuda_fn(
        lambda x, weight: (x.matmul(weight)).tanh(),
        ["x", "weight"],
        in_axes=[0, None],
        max_specializations=2,
    )
    weight = quabla.Tensor([1, 1], [2.0])
    first = compiled({"x": quabla.Tensor([2, 1], [0.5, -1.0]), "weight": weight})
    second = compiled(
        {"x": quabla.Tensor([3, 1], [0.0, 1.0, -0.5]), "weight": weight}
    )

    assert_close_rows(
        [first.to_flat_list()], [[math.tanh(1.0), math.tanh(-2.0)]], tol=1e-5
    )
    assert_close_rows(
        [second.to_flat_list()], [[0.0, math.tanh(2.0), math.tanh(-1.0)]], tol=1e-5
    )
    assert compiled.specialization_count == 2
    try:
        compiled({"x": quabla.Tensor([4, 1], [0.0] * 4), "weight": weight})
        assert False, "expected bounded CUDA specialization error"
    except ValueError as error:
        assert "max_specializations" in str(error)


def test_tensor_value_and_grad_batch_cuda_fn_specializes_collocation_batches():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    compiled = quabla.tensor_value_and_grad_batch_cuda_fn(
        lambda x, weight: (x * weight).powi(2).mean(),
        ["x", "weight"],
        ["x", "weight"],
        in_axes=[0, None],
        max_specializations=2,
    )
    weight = quabla.Tensor([1], [2.0])
    first_value, first_gradients = compiled(
        {"x": quabla.Tensor([2, 1], [1.0, 2.0]), "weight": weight}
    )
    second_value, second_gradients = compiled(
        {"x": quabla.Tensor([3, 1], [1.0, 2.0, 3.0]), "weight": weight}
    )

    assert_close_rows([first_value.to_flat_list()], [[10.0]], tol=1e-5)
    assert_close_rows([first_gradients["x"].to_flat_list()], [[4.0, 8.0]], tol=1e-5)
    assert_close_rows([first_gradients["weight"].to_flat_list()], [[10.0]], tol=1e-5)
    assert_close_rows([second_value.to_flat_list()], [[56.0 / 3.0]], tol=1e-5)
    assert_close_rows(
        [second_gradients["x"].to_flat_list()], [[8.0 / 3.0, 16.0 / 3.0, 8.0]], tol=1e-5
    )
    assert_close_rows([second_gradients["weight"].to_flat_list()], [[56.0 / 3.0]], tol=1e-5)
    assert compiled.specialization_count == 2


def test_tensor_vmap_fn_traces_one_batched_plan():
    mapped = quabla.tensor_vmap_fn(
        lambda x, weight: x.matmul(weight).tanh(),
        [("x", [2, 2]), ("weight", [2, 1])],
        3,
    )

    result = mapped(
        {
            "x": quabla.Tensor(
                [3, 2, 2],
                [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 3.0],
            ),
            "weight": quabla.Tensor([3, 2, 1], [1.0, -1.0, 1.0, 0.5, 2.0, 1.0]),
        }
    )

    assert result.shape == [3, 2, 1]
    assert_close_rows(
        [result.to_flat_list()],
        [[math.tanh(value) for value in [1.0, -1.0, 2.5, 2.0, 6.0, 3.0]]],
    )

    try:
        quabla.tensor_vmap_fn(lambda x: x, [("x", [2])], 0)
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("tensor_vmap_fn accepted a zero batch size")


def test_tensor_vmap_fori_loop_region_preserves_batched_carry_and_capture():
    mapped = quabla.tensor_vmap_fn(
        lambda initial, scale: quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry + index * captured_scale,
            initial,
            [scale],
        ),
        [("initial", []), ("scale", [])],
        2,
    )

    result = mapped(
        {
            "initial": quabla.Tensor([2], [1.0, 2.0]),
            "scale": quabla.Tensor([2], [2.0, 3.0]),
        }
    )

    assert result.shape == [2]
    assert_close_rows([result.to_flat_list()], [[7.0, 11.0]])


def test_tensor_vmap_scan_region_places_batch_axis_after_time_axis():
    mapped = quabla.tensor_vmap_fn(
        lambda initial, scale: quabla.tensor_scan_region(
            0,
            3,
            lambda _, carry, captured_scale: (
                carry + captured_scale,
                carry + captured_scale,
            ),
            initial,
            [scale],
        )[1],
        [("initial", []), ("scale", [])],
        2,
    )

    result = mapped(
        {
            "initial": quabla.Tensor([2], [1.0, 2.0]),
            "scale": quabla.Tensor([2], [2.0, 3.0]),
        }
    )

    assert result.shape == [2, 3]
    assert_close_rows([result.to_flat_list()], [[3.0, 5.0, 7.0, 5.0, 8.0, 11.0]])


def test_tensor_vmap_fori_region_vjp_preserves_per_example_gradients():
    def function(initial, scale):
        return quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry + index * captured_scale,
            initial,
            [scale],
        )
    values = {
        "initial": quabla.Tensor([2], [1.0, 2.0]),
        "scale": quabla.Tensor([2], [2.0, 3.0]),
    }
    cotangent = quabla.Tensor([2], [1.0, 1.0])
    vjp = quabla.tensor_vmap_vjp_fn(function, [("initial", []), ("scale", [])], 2)
    output, gradients = vjp(values, cotangent)

    assert_close_rows([output.to_flat_list()], [[7.0, 11.0]])
    assert_close_rows([gradients["initial"].to_flat_list()], [[1.0, 1.0]])
    assert_close_rows([gradients["scale"].to_flat_list()], [[3.0, 3.0]])

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_vjp_cuda_fn(function, [("initial", []), ("scale", [])], 2)
    cuda_output, cuda_gradients = cuda(values, cotangent)
    assert_close_rows([cuda_output.to_flat_list()], [output.to_flat_list()], tol=3e-5)
    for name in ["initial", "scale"]:
        assert_close_rows(
            [cuda_gradients[name].to_flat_list()],
            [gradients[name].to_flat_list()],
            tol=3e-5,
        )


def test_tensor_vmap_fori_region_vjp_executes_on_mlx():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    vjp = quabla.tensor_vmap_vjp_mlx_fn(
        lambda initial, scale: quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry + index * captured_scale,
            initial,
            [scale],
        ),
        [("initial", []), ("scale", [])],
        2,
    )
    output, gradients = vjp(
        {
            "initial": quabla.Tensor([2], [1.0, 2.0]),
            "scale": quabla.Tensor([2], [2.0, 3.0]),
        },
        quabla.Tensor([2], [1.0, 1.0]),
    )

    assert vjp.backend == "mlx"
    assert_close_rows([output.to_flat_list()], [[7.0, 11.0]], tol=1e-5)
    assert_close_rows([gradients["initial"].to_flat_list()], [[1.0, 1.0]], tol=1e-5)
    assert_close_rows([gradients["scale"].to_flat_list()], [[3.0, 3.0]], tol=1e-5)


def test_tensor_vmap_scan_region_jvp_preserves_batch_major_output_layout():
    jvp = quabla.tensor_vmap_jvp_fn(
        lambda initial, scale: quabla.tensor_scan_region(
            0,
            3,
            lambda _, carry, captured_scale: (
                carry + captured_scale,
                carry + captured_scale,
            ),
            initial,
            [scale],
        )[1],
        [("initial", []), ("scale", [])],
        2,
    )

    output, tangent = jvp(
        {
            "initial": quabla.Tensor([2], [1.0, 2.0]),
            "scale": quabla.Tensor([2], [2.0, 3.0]),
        },
        {
            "initial": quabla.Tensor([2], [1.0, 1.0]),
            "scale": quabla.Tensor([2], [1.0, 1.0]),
        },
    )

    assert output.shape == [2, 3]
    assert_close_rows([output.to_flat_list()], [[3.0, 5.0, 7.0, 5.0, 8.0, 11.0]])
    assert_close_rows([tangent.to_flat_list()], [[2.0, 3.0, 4.0, 2.0, 3.0, 4.0]])


def test_tensor_vmap_scan_region_vjp_matches_cpu_on_cuda():
    def function(initial, scale):
        return quabla.tensor_scan_region(
            0,
            3,
            lambda index, carry, captured_scale: (
                (carry * captured_scale + index).tanh(),
                (carry * captured_scale + index).tanh(),
            ),
            initial,
            [scale],
        )[1]
    values = {
        "initial": quabla.Tensor([2], [0.2, -0.3]),
        "scale": quabla.Tensor([2], [0.8, 1.1]),
    }
    cotangent = quabla.Tensor([2, 3], [1.0, -0.5, 0.25, -0.75, 0.5, 1.0])
    cpu = quabla.tensor_vmap_vjp_fn(
        function, [("initial", []), ("scale", [])], 2
    )
    expected_value, expected_gradients = cpu(values, cotangent)

    assert expected_value.shape == [2, 3]
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_vjp_cuda_fn(
        function, [("initial", []), ("scale", [])], 2
    )
    value, gradients = cuda(values, cotangent)
    assert value.shape == [2, 3]
    assert_close_rows([value.to_flat_list()], [expected_value.to_flat_list()], tol=3e-5)
    for name in ["initial", "scale"]:
        assert_close_rows(
            [gradients[name].to_flat_list()],
            [expected_gradients[name].to_flat_list()],
            tol=3e-5,
        )


def test_tensor_vmap_hvp_scalar_scan_region_matches_finite_difference_and_cuda():
    def function(initial, scale):
        return quabla.tensor_scan_region(
            0,
            3,
            lambda index, carry, captured_scale: (
                (carry * captured_scale + index).tanh(),
                (carry * captured_scale + index).tanh(),
            ),
            initial,
            [scale],
        )[0]
    specs = [("initial", []), ("scale", [])]
    values = {
        "initial": quabla.Tensor([2], [0.2, -0.3]),
        "scale": quabla.Tensor([2], [0.8, 1.1]),
    }
    direction = quabla.Tensor([2], [0.7, -0.4])
    hvp = quabla.tensor_vmap_hvp_scalar_fn(function, specs, 2, "initial")
    actual = hvp(values, direction)

    vjp = quabla.tensor_vmap_vjp_fn(function, specs, 2)
    epsilon = 1e-4
    plus_values = {
        "initial": quabla.Tensor([2], [0.2 + epsilon * 0.7, -0.3 - epsilon * 0.4]),
        "scale": values["scale"],
    }
    minus_values = {
        "initial": quabla.Tensor([2], [0.2 - epsilon * 0.7, -0.3 + epsilon * 0.4]),
        "scale": values["scale"],
    }
    _, plus_gradients = vjp(plus_values, quabla.Tensor([2], [1.0, 1.0]))
    _, minus_gradients = vjp(minus_values, quabla.Tensor([2], [1.0, 1.0]))
    expected = [
        (plus - minus) / (2.0 * epsilon)
        for plus, minus in zip(
            plus_gradients["initial"].to_flat_list(),
            minus_gradients["initial"].to_flat_list(),
        )
    ]
    assert_close_rows([actual.to_flat_list()], [expected], tol=2e-3)

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_hvp_scalar_cuda_fn(function, specs, 2, "initial")
    cuda_actual = cuda(values, direction)
    assert_close_rows([cuda_actual.to_flat_list()], [actual.to_flat_list()], tol=3e-5)


def test_tensor_vmap_hvp_scalar_restores_nonleading_input_axis_on_cuda():
    def function(x, scale):
        return (x * x * scale).sum()
    values = {
        "x": quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "scale": quabla.Tensor([2], [0.5, -1.0]),
    }
    direction = quabla.Tensor([2, 3], [0.1, 0.2, 0.3, -0.4, 0.5, -0.6])
    kwargs = {"in_axes": [-1, None], "out_axis": -1}
    cpu = quabla.tensor_vmap_hvp_scalar_fn(
        function, [("x", [2]), ("scale", [2])], 3, "x", **kwargs
    )
    actual = cpu(values, direction)
    assert actual.shape == [2, 3]
    assert_close_rows(
        [actual.to_flat_list()], [[0.1, 0.2, 0.3, 0.8, -1.0, 1.2]], tol=1e-12
    )

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_hvp_scalar_cuda_fn(
        function, [("x", [2]), ("scale", [2])], 3, "x", **kwargs
    )
    cuda_actual = cuda(values, direction)
    assert cuda_actual.shape == [2, 3]
    assert_close_rows([cuda_actual.to_flat_list()], [actual.to_flat_list()], tol=3e-5)


def test_tensor_vmap_hvp_scalar_scan_capture_matches_finite_difference_and_cuda():
    def function(initial, scale):
        return quabla.tensor_scan_region(
            0,
            3,
            lambda index, carry, captured_scale: (
                (carry * captured_scale + index).tanh(),
                (carry * captured_scale + index).tanh(),
            ),
            initial,
            [scale],
        )[0]
    specs = [("initial", []), ("scale", [])]
    values = {
        "initial": quabla.Tensor([2], [0.2, -0.3]),
        "scale": quabla.Tensor([2], [0.8, 1.1]),
    }
    direction = quabla.Tensor([2], [0.5, -0.6])
    actual = quabla.tensor_vmap_hvp_scalar_fn(function, specs, 2, "scale")(
        values, direction
    )
    vjp = quabla.tensor_vmap_vjp_fn(function, specs, 2)
    epsilon = 1e-4
    plus_values = {
        "initial": values["initial"],
        "scale": quabla.Tensor([2], [0.8 + epsilon * 0.5, 1.1 - epsilon * 0.6]),
    }
    minus_values = {
        "initial": values["initial"],
        "scale": quabla.Tensor([2], [0.8 - epsilon * 0.5, 1.1 + epsilon * 0.6]),
    }
    _, plus_gradients = vjp(plus_values, quabla.Tensor([2], [1.0, 1.0]))
    _, minus_gradients = vjp(minus_values, quabla.Tensor([2], [1.0, 1.0]))
    expected = [
        (plus - minus) / (2.0 * epsilon)
        for plus, minus in zip(
            plus_gradients["scale"].to_flat_list(),
            minus_gradients["scale"].to_flat_list(),
        )
    ]
    assert_close_rows([actual.to_flat_list()], [expected], tol=2e-3)

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_hvp_scalar_cuda_fn(function, specs, 2, "scale")
    cuda_actual = cuda(values, direction)
    assert_close_rows([cuda_actual.to_flat_list()], [actual.to_flat_list()], tol=3e-5)


def test_tensor_vmap_hvp_scalar_fori_capture_matches_exact_hessian_and_cuda():
    def function(initial, scale):
        return quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        )
    values = {
        "initial": quabla.Tensor([2], [1.0, 2.0]),
        "scale": quabla.Tensor([2], [0.5, 1.5]),
    }
    direction = quabla.Tensor([2], [0.7, -0.4])
    actual = quabla.tensor_vmap_hvp_scalar_fn(
        function, [("initial", []), ("scale", [])], 2, "scale"
    )(values, direction)
    assert_close_rows([actual.to_flat_list()], [[2.1, -7.2]], tol=1e-12)

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_hvp_scalar_cuda_fn(
        function, [("initial", []), ("scale", [])], 2, "scale"
    )
    cuda_actual = cuda(values, direction)
    assert_close_rows(
        [cuda_actual.to_flat_list()], [actual.to_flat_list()], tol=3e-5
    )


def test_tensor_vmap_hvp_scalar_fori_capture_restores_nonleading_axis_on_cuda():
    def function(initial, scale):
        return quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        ).sum()
    specs = [("initial", [2]), ("scale", [2])]
    values = {
        "initial": quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "scale": quabla.Tensor([2, 3], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    }
    direction = quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    kwargs = {"in_axes": [1, 1], "out_axis": -1}
    cpu = quabla.tensor_vmap_hvp_scalar_fn(
        function, specs, 3, "scale", **kwargs
    )
    actual = cpu(values, direction)
    assert actual.shape == [2, 3]
    assert_close_rows(
        [actual.to_flat_list()], [[3.0, 24.0, 81.0, 192.0, 375.0, 648.0]], tol=1e-12
    )

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_hvp_scalar_cuda_fn(
        function, specs, 3, "scale", **kwargs
    )
    cuda_actual = cuda(values, direction)
    assert cuda_actual.shape == [2, 3]
    assert_close_rows([cuda_actual.to_flat_list()], [actual.to_flat_list()], tol=3e-5)


def test_tensor_vmap_fori_jvp_preserves_mapped_capture_and_nonleading_axes():
    def function(initial, scale):
        return quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry + index * captured_scale,
            initial,
            [scale],
        )
    values = {
        "initial": quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "scale": quabla.Tensor([2, 3], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    }
    tangents = {
        "initial": quabla.Tensor([2, 3], [0.0] * 6),
        "scale": quabla.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    jvp = quabla.tensor_vmap_jvp_fn(
        function,
        [("initial", [2]), ("scale", [2])],
        3,
        in_axes=[1, 1],
        out_axis=1,
    )
    output, tangent = jvp(values, tangents)
    assert_close_rows(
        [output.to_flat_list()], [[2.5, 5.0, 7.5, 10.0, 12.5, 15.0]]
    )
    assert_close_rows([tangent.to_flat_list()], [[3.0, 6.0, 9.0, 12.0, 15.0, 18.0]])

    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    cuda = quabla.tensor_vmap_jvp_cuda_fn(
        function,
        [("initial", [2]), ("scale", [2])],
        3,
        in_axes=[1, 1],
        out_axis=1,
    )
    cuda_output, cuda_tangent = cuda(values, tangents)
    assert_close_rows(
        [cuda_output.to_flat_list()], [output.to_flat_list()], tol=3e-5
    )
    assert_close_rows(
        [cuda_tangent.to_flat_list()], [tangent.to_flat_list()], tol=3e-5
    )



def test_tensor_vmap_fn_supports_in_axes_out_axis_and_unmapped_inputs():
    mapped = quabla.tensor_vmap_fn(
        lambda x, weight: (x * weight).sum(),
        [("x", [2]), ("weight", [2])],
        3,
        in_axes=[-1, None],
        out_axis=-1,
    )

    result = mapped(
        {
            "x": quabla.Tensor([2, 3], [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
            "weight": quabla.Tensor([2], [2.0, -1.0]),
        }
    )

    assert result.shape == [3]
    assert_close_rows([result.to_flat_list()], [[0.0, 2.0, 4.0]])


def test_tensor_vmap_fn_reductions_and_transpose_preserve_batch_axis():
    mapped = quabla.tensor_vmap_fn(
        lambda x: x.transpose().mean(axis=1),
        [("x", [2, 3])],
        2,
        in_axes=[2],
        out_axis=1,
    )

    result = mapped(
        {
            "x": quabla.Tensor(
                [2, 3, 2],
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0],
            )
        }
    )

    assert result.shape == [3, 2]
    assert_close_rows([result.to_flat_list()], [[2.5, 25.0, 3.5, 35.0, 4.5, 45.0]])


def test_tensor_vmap_cuda_and_mlx_fn_use_the_same_batched_trace():
    input_specs = [("x", [2, 2]), ("weight", [2, 1])]
    values = {
        "x": quabla.Tensor(
            [3, 2, 2],
            [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 3.0],
        ),
        "weight": quabla.Tensor([3, 2, 1], [1.0, -1.0, 1.0, 0.5, 2.0, 1.0]),
    }
    expected = [[math.tanh(value) for value in [1.0, -1.0, 2.5, 2.0, 6.0, 3.0]]]
    nonleading_values = {"x": quabla.Tensor([2, 3], [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])}
    nonleading_expected = [[3.0, 7.0, 11.0]]

    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx_compiled = quabla.tensor_vmap_mlx_fn(
            lambda x, weight: x.matmul(weight).tanh(), input_specs, 3
        )
        assert mlx_compiled.backend == "mlx"
        assert_close_rows([mlx_compiled(values).to_flat_list()], expected, tol=1e-5)
        mlx_nonleading = quabla.tensor_vmap_mlx_fn(
            lambda x: x.sum(), [("x", [2])], 3, in_axes=[1]
        )
        assert_close_rows(
            [mlx_nonleading(nonleading_values).to_flat_list()], nonleading_expected, tol=1e-5
        )

    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda_compiled = quabla.tensor_vmap_cuda_fn(
            lambda x, weight: x.matmul(weight).tanh(), input_specs, 3
        )
        assert cuda_compiled.backend in {"cublas", "nvrtc"}
        assert_close_rows([cuda_compiled(values).to_flat_list()], expected, tol=1e-5)
        cuda_nonleading = quabla.tensor_vmap_cuda_fn(
            lambda x: x.sum(), [("x", [2])], 3, in_axes=[1]
        )
        assert_close_rows(
            [cuda_nonleading(nonleading_values).to_flat_list()], nonleading_expected, tol=1e-5
        )


def test_tensor_vmap_vjp_matches_per_example_loop_and_aggregates_unmapped_gradient():
    vjp = quabla.tensor_vmap_vjp_fn(
        lambda x, weight: (x * weight).tanh().sum(),
        [("x", [2]), ("weight", [2])],
        3,
        in_axes=[1, None],
    )
    x_values = [0.0, 0.5, -1.0, 1.0, -0.5, 2.0]
    weight_values = [2.0, -1.0]
    cotangent_values = [1.0, 2.0, -0.5]
    output, gradients = vjp(
        {
            "x": quabla.Tensor([2, 3], x_values),
            "weight": quabla.Tensor([2], weight_values),
        },
        quabla.Tensor([3], cotangent_values),
    )

    per_example = [
        [x_values[row * 3 + batch] for row in range(2)] for batch in range(3)
    ]
    expected_output = [
        sum(math.tanh(value * weight) for value, weight in zip(example, weight_values))
        for example in per_example
    ]
    expected_x = [
        cotangent_values[batch]
        * weight_values[row]
        * (1.0 - math.tanh(x_values[row * 3 + batch] * weight_values[row]) ** 2)
        for row in range(2)
        for batch in range(3)
    ]
    expected_weight = [
        sum(
            cotangent_values[batch]
            * x_values[row * 3 + batch]
            * (1.0 - math.tanh(x_values[row * 3 + batch] * weight_values[row]) ** 2)
            for batch in range(3)
        )
        for row in range(2)
    ]

    assert vjp.node_count > 0
    assert_close_rows([output.to_flat_list()], [expected_output])
    assert_close_rows([gradients["x"].to_flat_list()], [expected_x])
    assert_close_rows([gradients["weight"].to_flat_list()], [expected_weight])

    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx_vjp = quabla.tensor_vmap_vjp_mlx_fn(
            lambda x, weight: (x * weight).tanh().sum(),
            [("x", [2]), ("weight", [2])],
            3,
            in_axes=[1, None],
        )
        assert mlx_vjp.backend == "mlx"
        mlx_output, mlx_gradients = mlx_vjp(
            {
                "x": quabla.Tensor([2, 3], x_values),
                "weight": quabla.Tensor([2], weight_values),
            },
            quabla.Tensor([3], cotangent_values),
        )
        assert_close_rows([mlx_output.to_flat_list()], [expected_output], tol=1e-5)
        assert_close_rows([mlx_gradients["x"].to_flat_list()], [expected_x], tol=1e-5)
        assert_close_rows(
            [mlx_gradients["weight"].to_flat_list()], [expected_weight], tol=1e-5
        )


def test_tensor_vmap_jvp_matches_per_example_loop_with_nonleading_axes():
    jvp = quabla.tensor_vmap_jvp_fn(
        lambda x: x.tanh(),
        [("x", [2])],
        3,
        in_axes=[1],
        out_axis=1,
    )
    values = [0.0, 0.5, -1.0, 1.0, -0.5, 2.0]
    tangents = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0]
    output, output_tangent = jvp(
        {"x": quabla.Tensor([2, 3], values)},
        {"x": quabla.Tensor([2, 3], tangents)},
    )

    assert jvp.node_count > 0
    assert_close_rows([output.to_flat_list()], [[math.tanh(value) for value in values]])
    assert_close_rows(
        [output_tangent.to_flat_list()],
        [[direction * (1.0 - math.tanh(value) ** 2) for value, direction in zip(values, tangents)]],
    )

    if os.environ.get("QUABLA_MLX_TEST") is not None:
        mlx_jvp = quabla.tensor_vmap_jvp_mlx_fn(
            lambda x: x.tanh(),
            [("x", [2])],
            3,
            in_axes=[1],
            out_axis=1,
        )
        assert mlx_jvp.backend == "mlx"
        mlx_output, mlx_tangent = mlx_jvp(
            {"x": quabla.Tensor([2, 3], values)},
            {"x": quabla.Tensor([2, 3], tangents)},
        )
        assert_close_rows(
            [mlx_output.to_flat_list()],
            [[math.tanh(value) for value in values]],
            tol=1e-5,
        )
        assert_close_rows(
            [mlx_tangent.to_flat_list()],
            [[direction * (1.0 - math.tanh(value) ** 2) for value, direction in zip(values, tangents)]],
            tol=1e-5,
        )

    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda_jvp = quabla.tensor_vmap_jvp_cuda_fn(
            lambda x: x.tanh(),
            [("x", [2])],
            3,
            in_axes=[1],
            out_axis=1,
        )
        assert cuda_jvp.backend in {"cublas", "nvrtc"}
        cuda_output, cuda_tangent = cuda_jvp(
            {"x": quabla.Tensor([2, 3], values)},
            {"x": quabla.Tensor([2, 3], tangents)},
        )
        assert_close_rows(
            [cuda_output.to_flat_list()],
            [[math.tanh(value) for value in values]],
            tol=2e-5,
        )
        assert_close_rows(
            [cuda_tangent.to_flat_list()],
            [[direction * (1.0 - math.tanh(value) ** 2) for value, direction in zip(values, tangents)]],
            tol=2e-5,
        )


def test_tensor_vmap_cuda_vjp_matches_cpu_single_batched_plan():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    def function(x, weight):
        return (x * weight).tanh().sum()
    input_specs = [("x", [2]), ("weight", [2])]
    values = {
        "x": quabla.Tensor([2, 3], [0.0, 0.5, -1.0, 1.0, -0.5, 2.0]),
        "weight": quabla.Tensor([2], [2.0, -1.0]),
    }
    cotangent = quabla.Tensor([3], [1.0, 2.0, -0.5])
    cpu = quabla.tensor_vmap_vjp_fn(function, input_specs, 3, in_axes=[1, None])
    cuda = quabla.tensor_vmap_vjp_cuda_fn(function, input_specs, 3, in_axes=[1, None])

    expected_value, expected_gradients = cpu(values, cotangent)
    value, gradients = cuda(values, cotangent)

    assert cuda.node_count > 0
    assert cuda.backend in {"cublas", "nvrtc"}
    assert_close_rows([value.to_flat_list()], [expected_value.to_flat_list()], tol=2e-5)
    assert_close_rows(
        [gradients["x"].to_flat_list()], [expected_gradients["x"].to_flat_list()], tol=2e-5
    )
    assert_close_rows(
        [gradients["weight"].to_flat_list()],
        [expected_gradients["weight"].to_flat_list()],
        tol=2e-5,
    )


def test_tensor_vmap_batched_mlp_gradients_match_loop_on_cpu_and_cuda():
    def function(x, weight):
        return (x.matmul(weight).tanh()).sum()
    input_specs = [("x", [2]), ("weight", [2, 1])]
    values = {
        "x": quabla.Tensor([3, 2], [1.0, 2.0, -1.0, 0.5, 0.25, -2.0]),
        "weight": quabla.Tensor([2, 1], [0.75, -0.5]),
    }
    cotangent = quabla.Tensor([3], [1.0, -0.25, 2.0])
    cpu = quabla.tensor_vmap_vjp_fn(function, input_specs, 3, in_axes=[0, None])
    value, gradients = cpu(values, cotangent)

    examples = [[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]]
    weights = [0.75, -0.5]
    cotangents = [1.0, -0.25, 2.0]
    activations = [sum(x * weight for x, weight in zip(example, weights)) for example in examples]
    expected_value = [math.tanh(activation) for activation in activations]
    expected_x = [
        cotangents[batch] * weights[column] * (1.0 - math.tanh(activations[batch]) ** 2)
        for batch in range(3)
        for column in range(2)
    ]
    expected_weight = [
        sum(
            cotangents[batch]
            * examples[batch][column]
            * (1.0 - math.tanh(activations[batch]) ** 2)
            for batch in range(3)
        )
        for column in range(2)
    ]

    assert_close_rows([value.to_flat_list()], [expected_value])
    assert_close_rows([gradients["x"].to_flat_list()], [expected_x])
    assert_close_rows([gradients["weight"].to_flat_list()], [expected_weight])

    if os.environ.get("QUABLA_CUDA_TEST") is not None:
        cuda = quabla.tensor_vmap_vjp_cuda_fn(function, input_specs, 3, in_axes=[0, None])
        cuda_value, cuda_gradients = cuda(values, cotangent)
        assert_close_rows([cuda_value.to_flat_list()], [expected_value], tol=2e-5)
        assert_close_rows([cuda_gradients["x"].to_flat_list()], [expected_x], tol=2e-5)
        assert_close_rows(
            [cuda_gradients["weight"].to_flat_list()], [expected_weight], tol=2e-5
        )


def test_tensor_jit_cuda_fn_reuses_a_callable_cuda_plan():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    compiled = quabla.tensor_jit_cuda_fn(
        lambda x, weight, bias: (x @ weight + bias).tanh(),
        [("x", [2, 2]), ("weight", [2, 2]), ("bias", [1, 2])],
    )
    first = compiled(
        {
            "x": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
            "weight": quabla.Tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
            "bias": quabla.Tensor([1, 2], [0.0, 1.0]),
        }
    )
    second = compiled(
        {
            "x": quabla.Tensor([2, 2], [0.0, 1.0, -1.0, 0.5]),
            "weight": quabla.Tensor([2, 2], [2.0, 0.0, 0.0, 2.0]),
            "bias": quabla.Tensor([1, 2], [1.0, -1.0]),
        }
    )

    assert compiled.backend in {"cublas", "nvrtc"}
    expected_first = [math.tanh(1.0), math.tanh(3.0), math.tanh(3.0), math.tanh(5.0)]
    expected_second = [math.tanh(1.0), math.tanh(1.0), math.tanh(-1.0), math.tanh(0.0)]
    for actual, expected in zip(first.to_flat_list(), expected_first):
        assert abs(actual - expected) <= 1e-5
    for actual, expected in zip(second.to_flat_list(), expected_second):
        assert abs(actual - expected) <= 1e-5


def test_tensor_value_and_grad_cuda_fn_uses_one_callable_plan():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    value_and_grad = quabla.tensor_value_and_grad_cuda_fn(
        lambda x, target, weight, bias: ((x * weight + bias - target).powi(2)).mean(),
        [("x", [2]), ("target", [2]), ("weight", [1]), ("bias", [1])],
        ["weight", "bias"],
    )
    inputs = {
        "x": quabla.Tensor([2], [-1.0, 1.0]),
        "target": quabla.Tensor([2], [-1.0, 3.0]),
        "weight": quabla.Tensor([1], [0.0]),
        "bias": quabla.Tensor([1], [0.0]),
    }

    value, gradients = value_and_grad(inputs)
    cpu_value_and_grad = quabla.tensor_value_and_grad_fn(
        lambda x, target, weight, bias: ((x * weight + bias - target).powi(2)).mean(),
        [("x", [2]), ("target", [2]), ("weight", [1]), ("bias", [1])],
    )
    cpu_value, cpu_gradients = cpu_value_and_grad(inputs)
    assert abs(value.to_flat_list()[0] - 5.0) < 1e-5
    assert abs(gradients["weight"].to_flat_list()[0] + 4.0) < 1e-5
    assert abs(gradients["bias"].to_flat_list()[0] + 2.0) < 1e-5
    assert abs(value.to_flat_list()[0] - cpu_value.to_flat_list()[0]) < 1e-5
    assert abs(
        gradients["weight"].to_flat_list()[0]
        - cpu_gradients["weight"].to_flat_list()[0]
    ) < 1e-5
    assert abs(
        gradients["bias"].to_flat_list()[0]
        - cpu_gradients["bias"].to_flat_list()[0]
    ) < 1e-5


def test_tensor_value_and_grad_mlx_fn_uses_one_symbolic_plan():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    value_and_grad = quabla.tensor_value_and_grad_mlx_fn(
        lambda x, target, weight, bias: ((x * weight + bias - target).powi(2)).mean(),
        [("x", [2]), ("target", [2]), ("weight", [1]), ("bias", [1])],
        ["weight", "bias"],
    )
    inputs = {
        "x": quabla.Tensor([2], [-1.0, 1.0]),
        "target": quabla.Tensor([2], [-1.0, 3.0]),
        "weight": quabla.Tensor([1], [0.0]),
        "bias": quabla.Tensor([1], [0.0]),
    }
    value, gradients = value_and_grad(inputs)
    cpu_value, cpu_gradients = quabla.tensor_value_and_grad_fn(
        lambda x, target, weight, bias: ((x * weight + bias - target).powi(2)).mean(),
        [("x", [2]), ("target", [2]), ("weight", [1]), ("bias", [1])],
    )(inputs)
    assert abs(value.to_flat_list()[0] - cpu_value.to_flat_list()[0]) < 1e-5
    assert abs(
        gradients["weight"].to_flat_list()[0]
        - cpu_gradients["weight"].to_flat_list()[0]
    ) < 1e-5
    assert abs(
        gradients["bias"].to_flat_list()[0]
        - cpu_gradients["bias"].to_flat_list()[0]
    ) < 1e-5


def test_tensor_value_and_grad_mlx_fn_supports_fixed_fori_regions():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    def function(initial, scale):
        return quabla.tensor_fori_loop_region(
            0,
            3,
            lambda index, carry, captured_scale: carry * captured_scale + index,
            initial,
            [scale],
        ).powi(2).mean()
    inputs = {
        "initial": quabla.Tensor([], [1.0]),
        "scale": quabla.Tensor([], [2.0]),
    }
    mlx = quabla.tensor_value_and_grad_mlx_fn(
        function,
        [("initial", []), ("scale", [])],
        ["initial", "scale"],
    )
    cpu = quabla.tensor_value_and_grad_fn(
        function,
        [("initial", []), ("scale", [])],
    )

    mlx_value, mlx_gradients = mlx(inputs)
    cpu_value, cpu_gradients = cpu(inputs)
    assert abs(mlx_value.to_flat_list()[0] - cpu_value.to_flat_list()[0]) < 2e-5
    assert abs(
        mlx_gradients["initial"].to_flat_list()[0]
        - cpu_gradients["initial"].to_flat_list()[0]
    ) < 1e-5
    assert abs(
        mlx_gradients["scale"].to_flat_list()[0]
        - cpu_gradients["scale"].to_flat_list()[0]
    ) < 1e-5


def test_tensor_value_and_grad_mlx_fn_supports_fixed_scan_regions():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    def function(initial, scale):
        carry, outputs = quabla.tensor_scan_region(
            0,
            3,
            lambda index, current, captured_scale: (
                current * captured_scale + index,
                current * captured_scale + index,
            ),
            initial,
            [scale],
        )
        return carry + outputs.sum()

    inputs = {
        "initial": quabla.Tensor([], [1.0]),
        "scale": quabla.Tensor([], [2.0]),
    }
    mlx = quabla.tensor_value_and_grad_mlx_fn(
        function,
        [("initial", []), ("scale", [])],
        ["initial", "scale"],
    )
    cpu = quabla.tensor_value_and_grad_fn(
        function,
        [("initial", []), ("scale", [])],
    )

    mlx_value, mlx_gradients = mlx(inputs)
    cpu_value, cpu_gradients = cpu(inputs)
    assert abs(mlx_value.to_flat_list()[0] - cpu_value.to_flat_list()[0]) < 2e-5
    assert abs(
        mlx_gradients["initial"].to_flat_list()[0]
        - cpu_gradients["initial"].to_flat_list()[0]
    ) < 1e-5
    assert abs(
        mlx_gradients["scale"].to_flat_list()[0]
        - cpu_gradients["scale"].to_flat_list()[0]
    ) < 1e-5


def test_tensor_value_and_grad_batch_mlx_fn_specializes_collocation_batches():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    def function(x, target, weight):
        return ((x * weight - target).powi(2)).mean()
    mlx = quabla.tensor_value_and_grad_batch_mlx_fn(
        function,
        ["x", "target", "weight"],
        ["weight"],
        in_axes=[0, 0, None],
        max_specializations=2,
    )
    cpu = quabla.tensor_value_and_grad_batch_fn(
        function,
        ["x", "target", "weight"],
        in_axes=[0, 0, None],
        max_specializations=2,
    )
    weight = quabla.Tensor([1], [0.5])
    first_inputs = {
        "x": quabla.Tensor([2, 1], [-1.0, 2.0]),
        "target": quabla.Tensor([2, 1], [-2.0, 3.0]),
        "weight": weight,
    }
    second_inputs = {
        "x": quabla.Tensor([3, 1], [-2.0, 1.0, 3.0]),
        "target": quabla.Tensor([3, 1], [-1.0, 2.0, 4.0]),
        "weight": weight,
    }
    for inputs in (first_inputs, second_inputs):
        mlx_value, mlx_gradients = mlx(inputs)
        cpu_value, cpu_gradients = cpu(inputs)
        assert_close_rows(
            [mlx_value.to_flat_list()], [cpu_value.to_flat_list()], tol=1e-5
        )
        assert_close_rows(
            [mlx_gradients["weight"].to_flat_list()],
            [cpu_gradients["weight"].to_flat_list()],
            tol=1e-5,
        )
    assert mlx.specialization_count == 2
    try:
        mlx(
            {
                "x": quabla.Tensor([4, 1], [0.0] * 4),
                "target": quabla.Tensor([4, 1], [0.0] * 4),
                "weight": weight,
            }
        )
        assert False, "expected bounded MLX specialization error"
    except ValueError as error:
        assert "max_specializations" in str(error)


def test_cuda_adam_loss_optimizer_owns_scalar_loss_and_parameters():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).mean()
    optimizer = quabla.cuda_adam_loss_optimizer(
        loss,
        ["weight", "bias"],
        {
            "x": quabla.Tensor([2], [-1.0, 1.0]),
            "target": quabla.Tensor([2], [-1.0, 3.0]),
            "weight": quabla.Tensor([1], [0.0]),
            "bias": quabla.Tensor([1], [0.0]),
        },
        0.05,
        ["x", "target"],
    )

    initial_loss = optimizer.loss().to_flat_list()[0]
    # loss() runs a forward-only plan; the backward plan allocates its buffers
    # on the first step(), so count from there to check that training does not
    # grow device memory.
    optimizer.step()
    initial_buffers = optimizer.device_buffer_count
    for _ in range(249):
        optimizer.step()
    final_loss = optimizer.loss().to_flat_list()[0]
    trained = optimizer.parameters()

    assert final_loss < initial_loss * 1e-4
    assert abs(trained["weight"].to_flat_list()[0] - 2.0) < 2e-3
    assert abs(trained["bias"].to_flat_list()[0] - 1.0) < 2e-3
    assert optimizer.device_buffer_count == initial_buffers


def test_mlx_adam_loss_optimizer_keeps_parameters_and_moments_on_device():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).mean()
    optimizer = quabla.mlx_adam_loss_optimizer(
        loss,
        ["weight", "bias"],
        {
            "x": quabla.Tensor([2], [-1.0, 1.0]),
            "target": quabla.Tensor([2], [-1.0, 3.0]),
            "weight": quabla.Tensor([1], [0.0]),
            "bias": quabla.Tensor([1], [0.0]),
        },
        0.05,
        ["x", "target"],
    )

    initial_loss = optimizer.loss().to_flat_list()[0]
    for _ in range(250):
        optimizer.step()
    final_loss = optimizer.loss().to_flat_list()[0]
    trained = optimizer.parameters()

    assert final_loss < initial_loss * 1e-4
    assert abs(trained["weight"].to_flat_list()[0] - 2.0) < 2e-3
    assert abs(trained["bias"].to_flat_list()[0] - 1.0) < 2e-3


def test_mlx_adam_loss_optimizer_refreshes_dynamic_minibatches():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    traced = quabla.trace_tensor(
        lambda x, target, weight, bias: (x * weight) + bias,
        [("x", [2]), ("target", [2]), ("weight", [1]), ("bias", [1])],
    )
    loss = ((traced.output - traced.graph.input("target")).powi(2)).mean()
    initial_batch = {
        "x": quabla.Tensor([2], [-1.0, 1.0]),
        "target": quabla.Tensor([2], [-1.0, 3.0]),
        "weight": quabla.Tensor([1], [0.0]),
        "bias": quabla.Tensor([1], [0.0]),
    }
    refreshed_batch = {
        "x": quabla.Tensor([2], [-2.0, 2.0]),
        "target": quabla.Tensor([2], [-3.0, 5.0]),
    }
    optimizer = quabla.mlx_adam_loss_optimizer(
        loss, ["weight", "bias"], initial_batch, 0.05, []
    )

    optimizer.step(initial_batch)
    initial_refreshed_loss = optimizer.loss(refreshed_batch).to_flat_list()[0]
    for _ in range(250):
        optimizer.step(refreshed_batch)
    final_refreshed_loss = optimizer.loss().to_flat_list()[0]
    trained = optimizer.parameters()

    assert final_refreshed_loss < initial_refreshed_loss * 1e-4
    assert abs(trained["weight"].to_flat_list()[0] - 2.0) < 2e-3
    assert abs(trained["bias"].to_flat_list()[0] - 1.0) < 2e-3


def test_mlx_poisson_pinn_matches_cpu_reference():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    collocation = [0.15, 0.35, 0.55, 0.75, 0.9]
    coordinates = collocation + [0.0, 1.0]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in collocation] + [0.0, 0.0]
    traced = quabla.trace_tensor(
        lambda x, weight, forcing, boundary_target, boundary_mask: (x * weight).sin(),
        [
            ("x", [7, 1]),
            ("weight", [1, 1]),
            ("forcing", [7, 1]),
            ("boundary_target", [7, 1]),
            ("boundary_mask", [7, 1]),
        ],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    graph = second_derivative.graph
    x = graph.input("x")
    weight = graph.input("weight")
    residual = second_derivative.output + graph.input("forcing")
    boundary_error = (x * weight).sin() - graph.input("boundary_target")
    error = quabla.where(graph.input("boundary_mask").gt(0.0), boundary_error, residual)
    loss = (error * error).mean()
    inputs = {
        "x": quabla.Tensor([7, 1], coordinates),
        "forcing": quabla.Tensor([7, 1], forcing),
        "boundary_target": quabla.Tensor([7, 1], [0.0] * 7),
        "boundary_mask": quabla.Tensor([7, 1], [0.0] * 5 + [1.0] * 2),
        "weight": quabla.Tensor([1, 1], [2.5]),
    }

    cpu_optimizer = quabla.Adam(learning_rate=0.01)
    cpu_parameters = {"weight": inputs["weight"]}
    for _ in range(2000):
        _, gradients = graph.evaluate_value_and_vjp(
            loss.node_id,
            {**inputs, **cpu_parameters},
            quabla.Tensor([], [1.0]),
        )
        cpu_parameters = cpu_optimizer.step(cpu_parameters, {"weight": gradients["weight"]})
    cpu_loss, _ = graph.evaluate_value_and_vjp(
        loss.node_id,
        {**inputs, **cpu_parameters},
        quabla.Tensor([], [1.0]),
    )

    mlx_optimizer = quabla.mlx_adam_loss_optimizer(
        loss,
        ["weight"],
        inputs,
        0.01,
        ["x", "forcing", "boundary_target", "boundary_mask"],
    )
    mlx_initial = mlx_optimizer.loss().to_flat_list()[0]
    for _ in range(2000):
        mlx_optimizer.step()
    mlx_loss = mlx_optimizer.loss().to_flat_list()[0]
    mlx_weight = mlx_optimizer.parameters()["weight"].to_flat_list()[0]
    cpu_weight = cpu_parameters["weight"].to_flat_list()[0]

    assert mlx_loss < mlx_initial * 1e-8
    assert cpu_loss.to_flat_list()[0] < 1e-20
    assert abs(mlx_weight - cpu_weight) < 2e-3


def test_mlx_two_layer_poisson_pinn_matches_cpu_reference():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return

    example = runpy.run_path(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples", "pinn_mlp_mlx.py")
    )
    metrics = example["run"]()
    assert metrics["mlx_final"] < metrics["mlx_initial"] * 1e-4
    assert metrics["mlx_residual"] < 1e-5
    assert metrics["mlx_boundary"] < 1e-5
    assert metrics["max_parameter_difference"] < 2e-3


def test_tensor_vjp_fn_reuses_a_compiled_plan_with_runtime_cotangent():
    vjp = quabla.tensor_vjp_fn(lambda x: x.tanh(), [("x", [2, 2])])
    values = [0.0, 1.0, -1.0, 0.5]
    cotangent = [1.0, 2.0, 3.0, 4.0]

    output, gradients = vjp(
        {"x": quabla.Tensor([2, 2], values)},
        quabla.Tensor([2, 2], cotangent),
    )

    assert_close_rows([output.to_flat_list()], [[math.tanh(value) for value in values]])
    assert_close_rows(
        [gradients["x"].to_flat_list()],
        [[cot * (1.0 - math.tanh(value) ** 2) for value, cot in zip(values, cotangent)]],
    )


def test_tensor_jvp_fn_reuses_a_compiled_plan_with_runtime_tangent():
    jvp = quabla.tensor_jvp_fn(lambda x: x.tanh(), [("x", [2, 2])])
    values = [0.0, 1.0, -1.0, 0.5]
    tangent = [1.0, 2.0, 3.0, 4.0]

    output, output_tangent = jvp(
        {"x": quabla.Tensor([2, 2], values)},
        {"x": quabla.Tensor([2, 2], tangent)},
    )

    assert_close_rows([output.to_flat_list()], [[math.tanh(value) for value in values]])
    assert_close_rows(
        [output_tangent.to_flat_list()],
        [[direction * (1.0 - math.tanh(value) ** 2) for value, direction in zip(values, tangent)]],
    )


def test_tensor_jacobian_fn_reuses_a_compiled_plan():
    jacobian = quabla.tensor_jacobian_fn(
        lambda x: x.powi(2), [("x", [2, 2])], "x"
    )
    result = jacobian({"x": quabla.Tensor([2, 2], [1.0, 2.0, -3.0, 0.5])})

    assert_close_rows(
        result,
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, -6.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def test_adam_updates_tensor_parameters_with_persistent_moments():
    optimizer = quabla.Adam(learning_rate=0.1)
    parameters = {"weight": quabla.Tensor([2], [1.0, 2.0])}
    gradients = {"weight": quabla.Tensor([2], [0.5, -0.5])}

    first = optimizer.step(parameters, gradients)
    second = optimizer.step(first, gradients)

    assert_close_rows([first["weight"].to_flat_list()], [[0.9, 2.1]], tol=1e-7)
    assert_close_rows([second["weight"].to_flat_list()], [[0.8, 2.2]], tol=1e-7)


def test_adam_invalid_step_does_not_advance_optimizer_state():
    parameters = {"weight": quabla.Tensor([1], [1.0])}
    gradients = {"weight": quabla.Tensor([1], [0.5])}
    reference = quabla.Adam(learning_rate=0.1)
    expected = reference.step(reference.step(parameters, gradients), gradients)

    optimizer = quabla.Adam(learning_rate=0.1)
    first = optimizer.step(parameters, gradients)
    try:
        optimizer.step(first, {})
        assert False, "expected Adam to reject a missing gradient"
    except ValueError:
        pass
    actual = optimizer.step(first, gradients)

    assert_close_rows(
        [actual["weight"].to_flat_list()],
        [expected["weight"].to_flat_list()],
        tol=1e-12,
    )


def test_sum_gradients_combines_named_tensors():
    combined = quabla.sum_gradients(
        [
            {"weight": quabla.Tensor([2], [1.0, -2.0])},
            {"weight": quabla.Tensor([2], [0.5, 3.0])},
        ]
    )

    assert combined["weight"].to_flat_list() == [1.5, 1.0]
    filtered = quabla.sum_gradients(
        [
            {"weight": quabla.Tensor([1], [1.0]), "forcing": quabla.Tensor([1], [5.0])},
            {"weight": quabla.Tensor([1], [2.0]), "target": quabla.Tensor([1], [7.0])},
        ],
        ["weight"],
    )
    assert filtered["weight"].to_flat_list() == [3.0]


def test_poisson_residual_training_converges_with_symbolic_jvp_and_adam():
    coordinate = 0.5
    target_weight = 1.0
    target_value = math.tanh(target_weight * coordinate)
    forcing = 2.0 * target_weight**2 * target_value * (1.0 - target_value**2)
    traced = quabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [1]), ("weight", [1]), ("forcing", [1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    loss = (residual * residual).sum()
    parameters = {"weight": quabla.Tensor([1], [0.3])}
    optimizer = quabla.Adam(learning_rate=0.03)

    def value_and_grad(current):
        return second_derivative.graph.evaluate_value_and_vjp(
            loss.node_id,
            {
                "x": quabla.Tensor([1], [coordinate]),
                "forcing": quabla.Tensor([1], [forcing]),
                **current,
            },
            quabla.Tensor([], [1.0]),
        )

    initial_loss, _ = value_and_grad(parameters)
    for _ in range(500):
        _, gradients = value_and_grad(parameters)
        parameters = optimizer.step(parameters, {"weight": gradients["weight"]})
    final_loss, _ = value_and_grad(parameters)

    assert final_loss.to_flat_list()[0] < initial_loss.to_flat_list()[0] * 1e-12
    assert abs(parameters["weight"].to_flat_list()[0] - target_weight) < 1e-5


def test_batched_poisson_collocation_training_converges():
    coordinates = [0.2, 0.4, 0.6, 0.8]
    target_weight = 1.0
    forcing_values = [
        2.0
        * target_weight**2
        * math.tanh(target_weight * coordinate)
        * (1.0 - math.tanh(target_weight * coordinate) ** 2)
        for coordinate in coordinates
    ]
    traced = quabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [4, 1]), ("weight", [1, 1]), ("forcing", [4, 1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    loss = (residual * residual).mean()
    parameters = {"weight": quabla.Tensor([1, 1], [0.3])}
    optimizer = quabla.Adam(learning_rate=0.03)

    def value_and_grad(current):
        return second_derivative.graph.evaluate_value_and_vjp(
            loss.node_id,
            {
                "x": quabla.Tensor([4, 1], coordinates),
                "forcing": quabla.Tensor([4, 1], forcing_values),
                **current,
            },
            quabla.Tensor([], [1.0]),
        )

    initial_loss, _ = value_and_grad(parameters)
    for _ in range(500):
        _, gradients = value_and_grad(parameters)
        parameters = optimizer.step(parameters, {"weight": gradients["weight"]})
    final_loss, _ = value_and_grad(parameters)

    assert final_loss.to_flat_list()[0] < initial_loss.to_flat_list()[0] * 1e-12
    assert abs(parameters["weight"].to_flat_list()[0] - target_weight) < 1e-5


def test_poisson_training_aggregates_boundary_and_residual_gradients():
    coordinates = [0.2, 0.4, 0.6, 0.8]
    target_weight = 1.0
    forcing_values = [
        2.0
        * math.tanh(coordinate)
        * (1.0 - math.tanh(coordinate) ** 2)
        for coordinate in coordinates
    ]
    residual_trace = quabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [4, 1]), ("weight", [1, 1]), ("forcing", [4, 1])],
    )
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    residual_plan = second_derivative.graph.compile_cpu(residual_loss.node_id)
    boundary_trace = quabla.trace_tensor(
        lambda x, weight, target: (x * weight).tanh(),
        [("x", [4, 1]), ("weight", [1, 1]), ("target", [4, 1])],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    boundary_plan = boundary_trace.graph.compile_cpu(boundary_loss.node_id)
    parameters = {"weight": quabla.Tensor([1, 1], [0.3])}
    optimizer = quabla.Adam(learning_rate=0.03)

    def loss_and_gradient(current):
        residual_value, residual_gradients = residual_plan.evaluate_vjp(
            {"x": quabla.Tensor([4, 1], coordinates), "forcing": quabla.Tensor([4, 1], forcing_values), **current},
            quabla.Tensor([], [1.0]),
        )
        boundary_value, boundary_gradients = boundary_plan.evaluate_vjp(
            {"x": quabla.Tensor([4, 1], [1.0] * 4), "target": quabla.Tensor([4, 1], [math.tanh(1.0)] * 4), **current},
            quabla.Tensor([], [1.0]),
        )
        return residual_value.to_flat_list()[0] + boundary_value.to_flat_list()[0], {
            "weight": residual_gradients["weight"] + boundary_gradients["weight"]
        }

    initial_loss, _ = loss_and_gradient(parameters)
    for _ in range(500):
        _, gradients = loss_and_gradient(parameters)
        parameters = optimizer.step(parameters, gradients)
    final_loss, _ = loss_and_gradient(parameters)

    assert final_loss < initial_loss * 1e-12
    assert abs(parameters["weight"].to_flat_list()[0] - target_weight) < 1e-5


def test_symbolic_jvp_residual_backpropagates_through_two_layer_mlp():
    traced = quabla.trace_tensor(
        lambda x, w1, b1, w2, b2, forcing: (x.matmul(w1) + b1).tanh().matmul(w2)
        + b2,
        [
            ("x", [4, 1]),
            ("w1", [1, 2]),
            ("b1", [1, 2]),
            ("w2", [2, 1]),
            ("b2", [1, 1]),
            ("forcing", [4, 1]),
        ],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    loss = (residual * residual).mean()
    _, gradients = second_derivative.graph.evaluate_value_and_vjp(
        loss.node_id,
        {
            "x": quabla.Tensor([4, 1], [0.2, 0.4, 0.6, 0.8]),
            "w1": quabla.Tensor([1, 2], [0.7, -0.4]),
            "b1": quabla.Tensor([1, 2], [0.1, -0.2]),
            "w2": quabla.Tensor([2, 1], [0.5, -0.3]),
            "b2": quabla.Tensor([1, 1], [0.05]),
            "forcing": quabla.Tensor([4, 1], [0.2, -0.1, 0.3, -0.2]),
        },
        quabla.Tensor([], [1.0]),
    )

    for name in ["w1", "b1", "w2"]:
        assert any(abs(value) > 1e-8 for value in gradients[name].to_flat_list())


def test_two_layer_mlp_poisson_training_converges():
    coordinates = [0.2, 0.4, 0.6, 0.8]
    parameter_specs = [
        ("x", [4, 1]), ("w1", [1, 2]), ("b1", [1, 2]),
        ("w2", [2, 1]), ("b2", [1, 1]), ("forcing", [4, 1]),
    ]
    def model(x, w1, b1, w2, b2, forcing):
        return (x.matmul(w1) + b1).tanh().matmul(w2) + b2
    residual_trace = quabla.trace_tensor(model, parameter_specs)
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    boundary_trace = quabla.trace_tensor(
        lambda x, w1, b1, w2, b2, target: (x.matmul(w1) + b1).tanh().matmul(w2) + b2,
        [
            ("x", [4, 1]), ("w1", [1, 2]), ("b1", [1, 2]),
            ("w2", [2, 1]), ("b2", [1, 1]), ("target", [4, 1]),
        ],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    teacher = {
        "w1": quabla.Tensor([1, 2], [1.2, -0.7]), "b1": quabla.Tensor([1, 2], [0.1, -0.2]),
        "w2": quabla.Tensor([2, 1], [0.8, 0.5]), "b2": quabla.Tensor([1, 1], [0.05]),
    }
    forcing = second_derivative.graph.evaluate(
        second_derivative.output.node_id,
        {"x": quabla.Tensor([4, 1], coordinates), "forcing": quabla.Tensor([4, 1], [0.0] * 4), **teacher},
    )
    boundary_coordinates = [0.0, 0.0, 1.0, 1.0]
    boundary_target = boundary_trace.graph.evaluate(
        boundary_trace.output.node_id,
        {"x": quabla.Tensor([4, 1], boundary_coordinates), "target": quabla.Tensor([4, 1], [0.0] * 4), **teacher},
    )
    parameters = {
        "w1": quabla.Tensor([1, 2], [0.3, -0.1]), "b1": quabla.Tensor([1, 2], [0.0, 0.0]),
        "w2": quabla.Tensor([2, 1], [0.2, 0.1]), "b2": quabla.Tensor([1, 1], [0.0]),
    }
    optimizer = quabla.Adam(learning_rate=0.02)

    def loss_and_grad(current):
        residual_value, residual_gradients = second_derivative.graph.evaluate_value_and_vjp(
            residual_loss.node_id,
            {"x": quabla.Tensor([4, 1], coordinates), "forcing": quabla.Tensor([4, 1], [-value for value in forcing.to_flat_list()]), **current},
            quabla.Tensor([], [1.0]),
        )
        boundary_value, boundary_gradients = boundary_trace.graph.evaluate_value_and_vjp(
            boundary_loss.node_id,
            {"x": quabla.Tensor([4, 1], boundary_coordinates), "target": boundary_target, **current},
            quabla.Tensor([], [1.0]),
        )
        return residual_value.to_flat_list()[0] + boundary_value.to_flat_list()[0], {
            name: residual_gradients[name] + boundary_gradients[name] for name in current
        }

    initial_loss, _ = loss_and_grad(parameters)
    for _ in range(2000):
        _, gradients = loss_and_grad(parameters)
        parameters = optimizer.step(parameters, gradients)
    final_loss, _ = loss_and_grad(parameters)

    assert final_loss < initial_loss * 1e-5


def test_standard_sine_poisson_training_recovers_pi():
    coordinates = [0.15, 0.35, 0.55, 0.75, 0.9]
    forcing = [math.pi**2 * math.sin(math.pi * coordinate) for coordinate in coordinates]
    residual_trace = quabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).sin(),
        [("x", [5, 1]), ("weight", [1, 1]), ("forcing", [5, 1])],
    )
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    boundary_trace = quabla.trace_tensor(
        lambda x, weight, target: (x * weight).sin(),
        [("x", [2, 1]), ("weight", [1, 1]), ("target", [2, 1])],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    parameters = {"weight": quabla.Tensor([1, 1], [2.5])}
    optimizer = quabla.Adam(learning_rate=0.01)

    def loss_and_grad(current):
        residual_value, residual_gradients = second_derivative.graph.evaluate_value_and_vjp(
            residual_loss.node_id,
            {"x": quabla.Tensor([5, 1], coordinates), "forcing": quabla.Tensor([5, 1], forcing), **current},
            quabla.Tensor([], [1.0]),
        )
        boundary_value, boundary_gradients = boundary_trace.graph.evaluate_value_and_vjp(
            boundary_loss.node_id,
            {"x": quabla.Tensor([2, 1], [0.0, 1.0]), "target": quabla.Tensor([2, 1], [0.0, 0.0]), **current},
            quabla.Tensor([], [1.0]),
        )
        return residual_value.to_flat_list()[0] + boundary_value.to_flat_list()[0], {
            "weight": residual_gradients["weight"] + boundary_gradients["weight"]
        }

    initial_loss, _ = loss_and_grad(parameters)
    for _ in range(2000):
        _, gradients = loss_and_grad(parameters)
        parameters = optimizer.step(parameters, gradients)
    final_loss, _ = loss_and_grad(parameters)

    assert final_loss < initial_loss * 1e-12
    assert abs(parameters["weight"].to_flat_list()[0] - math.pi) < 1e-8


def test_trace_tensor_compile_cpu_eliminates_unreachable_nodes():
    def model(x):
        x.tanh()
        return x + x

    traced = quabla.trace_tensor(model, [("x", [2, 2])])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    direct_plan = traced.output.compile_cpu()

    assert plan.node_count == 2
    assert direct_plan.node_count == plan.node_count
    assert plan.lower_text() == "\n".join(
        [
            "%0 = input[name=x] : tensor<2x2xf64>",
            "%1 = add(%0, %0) : tensor<2x2xf64>",
        ]
    )
    assert plan.kernel_ir() == [
        {
            "id": 0,
            "op": "input",
            "shape": [2, 2],
            "dtype": "f64",
            "layout": "row_major_contiguous",
            "placement": "unplaced",
            "effect": "input",
            "alias_of": None,
            "inputs": [],
            "name": "x",
        },
        {
            "id": 1,
            "op": "add",
            "shape": [2, 2],
            "dtype": "f64",
            "layout": "row_major_contiguous",
            "placement": "unplaced",
            "effect": "pure",
            "alias_of": None,
            "inputs": [0, 0],
        },
    ]
    plan.validate_kernel_ir()
    inputs = {"x": quabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])}
    assert plan.evaluate(inputs).to_flat_list() == [2.0, 4.0, 6.0, 8.0]
    _, gradients = plan.evaluate_vjp(inputs, quabla.Tensor([2, 2], [1.0] * 4))
    assert gradients["x"].to_flat_list() == [2.0] * 4
    _, alias_gradients = plan.evaluate_value_and_vjp(
        inputs, quabla.Tensor([2, 2], [1.0] * 4)
    )
    assert alias_gradients["x"].to_flat_list() == [2.0] * 4
    _, tangent = plan.evaluate_jvp(inputs, {"x": quabla.Tensor([2, 2], [1.0] * 4)})
    assert tangent.to_flat_list() == [2.0] * 4


def test_kernel_ir_marks_reshape_as_logical_alias_candidate():
    traced = quabla.trace_tensor(lambda x: x.reshape([4]), [("x", [2, 2])])
    plan = traced.output.compile_cpu()
    assert plan.kernel_ir()[1]["alias_of"] == 0
    plan.validate_kernel_ir()


def test_cpu_execution_plan_exposes_liveness_buffer_schedule():
    def model(x, y, z, q, r):
        first = x + y
        second = first + z
        third = second + q
        return third + r

    traced = quabla.trace_tensor(
        model,
        [("x", [2]), ("y", [2]), ("z", [2]), ("q", [2]), ("r", [2])],
    )
    buffers = traced.output.compile_cpu().buffer_plan()

    assert buffers == {
        "slots": [{"id": 0, "element_count": 2}, {"id": 1, "element_count": 2}],
        "node_slots": [None, None, None, None, None, 0, 1, 0, 1],
        "node_aliases": [None] * 9,
        "output_backing_node_id": 8,
    }


def test_cpu_execution_plan_exposes_elementwise_fusion_regions():
    traced = quabla.trace_tensor(
        lambda x, weight, bias: ((x @ weight + bias).tanh()).sin(),
        [("x", [2, 3]), ("weight", [3, 4]), ("bias", [1, 4])],
    )

    assert traced.output.compile_cpu().fusion_regions() == [
        {"output_node_id": 6, "node_ids": [4, 5, 6], "input_node_ids": [2, 3]}
    ]


def test_trace_tensor_compile_cpu_commons_identical_pure_nodes():
    def model(x):
        return x.tanh() + x.tanh()

    traced = quabla.trace_tensor(model, [("x", [2, 2])])
    plan = traced.graph.compile_cpu(traced.output.node_id)

    assert plan.node_count == 3
    assert plan.kernel_ir()[2]["inputs"] == [1, 1]
    assert_close_rows(
        [plan.evaluate({"x": quabla.Tensor([2, 2], [0.0, 1.0, -1.0, 0.5])}).to_flat_list()],
        [[0.0, 2.0 * math.tanh(1.0), -2.0 * math.tanh(1.0), 2.0 * math.tanh(0.5)]],
    )


def test_matrix_neg_and_scalar_div():
    a = quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])

    negated = -a
    scaled = a / 2.0

    assert negated.shape == (2, 2)
    assert negated.to_list() == [[-1.0, 2.0], [-3.0, 4.0]]
    assert scaled.shape == (2, 2)
    assert scaled.to_list() == [[0.5, -1.0], [1.5, -2.0]]


def test_matrix_elementwise_division():
    a = quabla.Matrix([[2.0, 6.0], [12.0, 20.0]])
    b = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    scalar = quabla.Matrix([[2.0]])

    divided = a / b
    scaled = a / scalar

    assert divided.shape == (2, 2)
    assert divided.to_list() == [[2.0, 3.0], [4.0, 5.0]]
    assert scaled.shape == (2, 2)
    assert scaled.to_list() == [[1.0, 3.0], [6.0, 10.0]]


def test_matrix_integer_power():
    a = quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])

    squared = a**2

    assert squared.shape == (2, 2)
    assert squared.to_list() == [[1.0, 4.0], [9.0, 16.0]]


def test_matrix_float_power():
    a = quabla.Matrix([[1.0, 4.0], [9.0, 16.0]])

    rooted = a**0.5

    assert rooted.shape == (2, 2)
    assert rooted.to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_sum():
    a = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])

    total = a.sum()

    assert total.shape == (1, 1)
    assert total.to_list() == [[10.0]]


def test_matrix_sum_axis():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    column_totals = a.sum(axis=0)
    row_totals = a.sum(axis=1)

    assert column_totals.shape == (1, 3)
    assert column_totals.to_list() == [[5.0, 7.0, 9.0]]
    assert row_totals.shape == (2, 1)
    assert row_totals.to_list() == [[6.0], [15.0]]


def test_matrix_mean():
    a = quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])

    average = a.mean()

    assert average.shape == (1, 1)
    assert average.to_list() == [[2.5]]


def test_matrix_mean_axis():
    a = quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    column_means = a.mean(axis=0)
    row_means = a.mean(axis=1)

    assert column_means.shape == (1, 3)
    assert column_means.to_list() == [[2.5, 3.5, 4.5]]
    assert row_means.shape == (2, 1)
    assert row_means.to_list() == [[2.0], [5.0]]


def test_matrix_tanh():
    a = quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.tanh()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.tanh(0.0), math.tanh(1.0)], [math.tanh(-1.0), math.tanh(2.0)]],
    )


def test_matrix_exp():
    a = quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.exp()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.exp(0.0), math.exp(1.0)], [math.exp(-1.0), math.exp(2.0)]],
    )


def test_matrix_log():
    a = quabla.Matrix([[1.0, 2.0], [4.0, 8.0]])

    output = a.log()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.log(1.0), math.log(2.0)], [math.log(4.0), math.log(8.0)]],
    )


def test_matrix_sqrt():
    a = quabla.Matrix([[1.0, 4.0], [9.0, 16.0]])

    output = a.sqrt()

    assert output.shape == (2, 2)
    assert output.to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_sin():
    a = quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.sin()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.sin(0.0), math.sin(1.0)], [math.sin(-1.0), math.sin(2.0)]],
    )


def test_matrix_cos():
    a = quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.cos()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.cos(0.0), math.cos(1.0)], [math.cos(-1.0), math.cos(2.0)]],
    )


def test_matrix_gt_and_where_support_masks():
    a = quabla.Matrix([[-1.0, 0.5, 2.0], [3.0, -4.0, 5.0]])
    positive = a.gt(0.0)
    selected = quabla.where(
        positive,
        a,
        quabla.Matrix([[0.0]]),
    )

    assert positive.shape == (2, 3)
    assert positive.to_list() == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    assert selected.to_list() == [[0.0, 0.5, 2.0], [3.0, 0.0, 5.0]]


def test_matrix_rejects_incompatible_shapes():
    a = quabla.Matrix([[1.0, 2.0]])
    b = quabla.Matrix([[3.0, 4.0]])

    try:
        a @ b
    except ValueError as err:
        assert "incompatible" in str(err)
    else:
        raise AssertionError("expected incompatible shapes to raise ValueError")


def test_trace_graph_records_matmul():
    graph = quabla.TraceGraph()
    a = graph.input("a", (2, 3))
    b = graph.input("b", (3, 2))

    c = a @ b

    assert c.shape == (2, 2)
    assert c.node_id == 2
    assert graph.describe() == [
        "0 input a shape=(2, 3)",
        "1 input b shape=(3, 2)",
        "2 matmul inputs=[0, 1] shape=(2, 2)",
    ]


def test_trace_function_records_python_matmul():
    def model(a, b):
        return a @ b

    traced = quabla.trace(model, [("a", (2, 3)), ("b", (3, 2))])

    assert traced.output.shape == (2, 2)
    assert traced.output.node_id == 2
    assert traced.graph.describe() == [
        "0 input a shape=(2, 3)",
        "1 input b shape=(3, 2)",
        "2 matmul inputs=[0, 1] shape=(2, 2)",
    ]
    assert traced.graph.ir() == [
        {"id": 0, "op": "input", "shape": (2, 3), "inputs": [], "name": "a"},
        {"id": 1, "op": "input", "shape": (3, 2), "inputs": [], "name": "b"},
        {"id": 2, "op": "matmul", "shape": (2, 2), "inputs": [0, 1]},
    ]
    assert traced.graph.vjp_ir(traced.output.node_id) == [
        {"id": 0, "op": "cotangent_seed", "shape": (2, 2), "inputs": [], "target": 2},
        {"id": 1, "op": "transpose", "shape": (2, 3), "inputs": [1]},
        {"id": 2, "op": "matmul", "shape": (2, 3), "inputs": [0, 1], "target": 0},
        {"id": 3, "op": "transpose", "shape": (3, 2), "inputs": [0]},
        {"id": 4, "op": "matmul", "shape": (3, 2), "inputs": [3, 0], "target": 1},
    ]
    gradients = traced.graph.evaluate_vjp(
        traced.output.node_id,
        {
            "a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": quabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        },
        quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )
    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]


def test_trace_graph_ir_includes_backend_lowering_attrs():
    def model(a, b):
        powered = a**2.5
        reduced = powered.mean(axis=0)
        return quabla.concat([reduced, b], axis=1)

    traced = quabla.trace(model, [("a", (2, 2)), ("b", (1, 1))])
    ir = traced.graph.ir()

    assert ir[2] == {
        "id": 2,
        "op": "powf",
        "shape": (2, 2),
        "inputs": [0],
        "attrs": {"exponent": 2.5},
    }
    assert ir[3] == {
        "id": 3,
        "op": "mean",
        "shape": (1, 2),
        "inputs": [2],
        "attrs": {"axis": 0},
    }
    assert ir[4] == {
        "id": 4,
        "op": "concat",
        "shape": (1, 3),
        "inputs": [3, 1],
        "attrs": {"axis": 1},
    }


def test_trace_graph_lowers_to_deterministic_backend_text():
    def model(a, b):
        powered = a**2.5
        reduced = powered.mean(axis=0)
        return quabla.concat([reduced, b], axis=1)

    traced = quabla.trace(model, [("a", (2, 2)), ("b", (1, 1))])

    assert traced.graph.lower_text() == "\n".join(
        [
            "%0 = input[name=a] : tensor<2x2xf64>",
            "%1 = input[name=b] : tensor<1x1xf64>",
            "%2 = powf(%0) {exponent=2.5} : tensor<2x2xf64>",
            "%3 = mean(%2) {axis=0} : tensor<1x2xf64>",
            "%4 = concat(%3, %1) {axis=1} : tensor<1x3xf64>",
        ]
    )


def test_trace_graph_exports_verified_stablehlo_subset():
    def model(a, b):
        return (a + b).tanh()

    traced = quabla.trace_tensor(model, [("a", [2, 1]), ("b", [2, 1])])

    assert traced.graph.stablehlo_text(traced.output.node_id) == "\n".join(
        [
            "module {",
            "  func.func @main(%arg0: tensor<2x1xf64> // a, %arg1: tensor<2x1xf64> // b) -> tensor<2x1xf64> {",
            "    %v2 = stablehlo.add %arg0, %arg1 : tensor<2x1xf64>",
            "    %v3 = stablehlo.tanh %v2 : tensor<2x1xf64>",
            "    return %v3 : tensor<2x1xf64>",
            "  }",
            "}",
        ]
    ) + "\n"


def test_trace_graph_evaluates_direct_jvp():
    def model(a, b):
        return (a @ b) + (a * b.T)

    traced = quabla.trace(model, [("a", (2, 2)), ("b", (2, 2))])
    primal, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        },
        {
            "a": quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
            "b": quabla.Matrix([[0.0, 1.0], [2.0, -1.0]]),
        },
    )

    assert primal.to_list() == [[24.0, 36.0], [61.0, 82.0]]
    assert tangent.to_list() == [[17.5, 16.5], [14.0, 21.0]]


def test_trace_graph_compiles_an_immutable_cpu_execution_plan():
    def model(a, b):
        return (a @ b).tanh()

    traced = quabla.trace(model, [("a", (2, 2)), ("b", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    values = {
        "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        "b": quabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
    }

    assert plan.output_node_id == traced.output.node_id
    assert plan.output_shape == (2, 2)
    assert plan.evaluate(values).to_list() == traced.graph.evaluate(
        traced.output.node_id, values
    ).to_list()


def test_cpu_execution_plan_evaluates_vjp():
    def model(a, b):
        return a @ b

    traced = quabla.trace(model, [("a", (2, 2)), ("b", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    values = {
        "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        "b": quabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
    }
    cotangent = quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]])

    plan_gradients = plan.evaluate_vjp(values, cotangent)
    graph_gradients = traced.graph.evaluate_vjp(traced.output.node_id, values, cotangent)

    assert set(plan_gradients) == {"a", "b"}
    assert plan_gradients["a"].to_list() == graph_gradients["a"].to_list()
    assert plan_gradients["b"].to_list() == graph_gradients["b"].to_list()


def test_cpu_execution_plan_eliminates_unreachable_trace_nodes():
    def model(a):
        _unreachable = a.exp()
        return a + 1.0

    traced = quabla.trace(model, [("a", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)

    assert len(traced.graph.ir()) == 4
    assert plan.output_node_id == traced.output.node_id
    assert plan.node_count == 3
    assert plan.evaluate(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    ).to_list() == [[2.0, 3.0], [4.0, 5.0]]


def test_cpu_execution_plan_eliminates_repeated_unary_subexpressions():
    def model(a):
        return a.exp() + a.exp()

    traced = quabla.trace(model, [("a", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    values = {"a": quabla.Matrix([[0.0, 1.0], [2.0, 3.0]])}

    assert len(traced.graph.ir()) == 4
    assert plan.node_count == 3
    assert plan.evaluate(values).to_list() == traced.graph.evaluate(
        traced.output.node_id, values
    ).to_list()


def test_grad_traces_and_evaluates_vjp():
    def model(a, b):
        return a @ b

    gradients = quabla.grad(
        model,
        [("a", (2, 3)), ("b", (3, 2))],
        {
            "a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": quabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        },
        quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )

    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]


def test_grad_fn_reuses_transform_callable():
    def model(a, b):
        return a @ b

    grad_model = quabla.grad_fn(
        model,
        [("a", (2, 3)), ("b", (3, 2))],
        quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )
    gradients = grad_model(
        {
            "a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": quabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        }
    )

    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]


def test_grad_fn_evaluates_add_vjp():
    def model(a, b):
        return a + b

    cotangent = quabla.Matrix([[0.5, 1.5], [2.5, 3.5]])
    grad_model = quabla.grad_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
        cotangent,
    )
    gradients = grad_model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        }
    )

    assert gradients["a"].to_list() == cotangent.to_list()
    assert gradients["b"].to_list() == cotangent.to_list()


def test_grad_fn_evaluates_mul_vjp():
    def model(a, b):
        return a * b

    grad_model = quabla.grad_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
        quabla.Matrix([[2.0, 3.0], [4.0, 5.0]]),
    )
    gradients = grad_model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[0.5, 1.5], [2.5, 3.5]]),
        }
    )

    assert gradients["a"].to_list() == [[1.0, 4.5], [10.0, 17.5]]
    assert gradients["b"].to_list() == [[2.0, 6.0], [12.0, 20.0]]


def test_grad_fn_evaluates_square_sum_vjp():
    def model(a):
        return (a * a).sum()

    grad_model = quabla.grad_fn(
        model,
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert gradients["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]


def test_grad_fn_evaluates_square_mean_vjp():
    grad_model = quabla.grad_fn(
        lambda a: (a * a).mean(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert gradients["a"].to_list() == [[0.5, 1.0], [1.5, 2.0]]


def test_grad_fn_evaluates_squared_error_vjp():
    def model(pred, target):
        error = pred - target
        return (error * error).sum()

    grad_model = quabla.grad_fn(
        model,
        [("pred", (2, 2)), ("target", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_model(
        {
            "pred": quabla.Matrix([[1.0, 3.0], [2.0, 5.0]]),
            "target": quabla.Matrix([[0.5, 1.0], [3.0, 1.5]]),
        }
    )

    assert gradients["pred"].to_list() == [[1.0, 4.0], [-2.0, 7.0]]
    assert gradients["target"].to_list() == [[-1.0, -4.0], [2.0, -7.0]]


def test_grad_scalar_fn_seeds_scalar_cotangent():
    def model(pred, target):
        error = pred - target
        return (error * error).sum()

    grad_model = quabla.grad_scalar_fn(
        model,
        [("pred", (2, 2)), ("target", (2, 2))],
    )
    gradients = grad_model(
        {
            "pred": quabla.Matrix([[1.0, 3.0], [2.0, 5.0]]),
            "target": quabla.Matrix([[0.5, 1.0], [3.0, 1.5]]),
        }
    )

    assert gradients["pred"].to_list() == [[1.0, 4.0], [-2.0, 7.0]]
    assert gradients["target"].to_list() == [[-1.0, -4.0], [2.0, -7.0]]


def test_grad_scalar_decorator_factory():
    @quabla.grad_scalar([("pred", (2, 2)), ("target", (2, 2))])
    def model(pred, target):
        error = pred - target
        return (error * error).sum()

    gradients = model(
        {
            "pred": quabla.Matrix([[1.0, 3.0], [2.0, 5.0]]),
            "target": quabla.Matrix([[0.5, 1.0], [3.0, 1.5]]),
        }
    )

    assert gradients["pred"].to_list() == [[1.0, 4.0], [-2.0, 7.0]]
    assert gradients["target"].to_list() == [[-1.0, -4.0], [2.0, -7.0]]


def test_grad_fn_traces_once_at_transform_creation():
    calls = {"count": 0}

    def model(a, b):
        calls["count"] += 1
        return a + b

    grad_model = quabla.grad_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
        quabla.Matrix([[1.0, 1.0], [1.0, 1.0]]),
    )

    assert calls["count"] == 1

    first = grad_model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        }
    )
    second = grad_model(
        {
            "a": quabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
            "b": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert first["a"].to_list() == [[1.0, 1.0], [1.0, 1.0]]
    assert first["b"].to_list() == [[1.0, 1.0], [1.0, 1.0]]
    assert second["a"].to_list() == [[1.0, 1.0], [1.0, 1.0]]
    assert second["b"].to_list() == [[1.0, 1.0], [1.0, 1.0]]
    assert calls["count"] == 1


def test_grad_scalar_fn_traces_once_at_transform_creation():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a * a).sum()

    grad_model = quabla.grad_scalar_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    first = grad_model({"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})
    second = grad_model({"a": quabla.Matrix([[2.0, 4.0], [6.0, 8.0]])})

    assert first["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert second["a"].to_list() == [[4.0, 8.0], [12.0, 16.0]]
    assert calls["count"] == 1


def test_grad_scalar_decorator_traces_once_at_decoration_time():
    calls = {"count": 0}

    @quabla.grad_scalar([("a", (2, 2))])
    def model(a):
        calls["count"] += 1
        return (a * a).sum()

    assert calls["count"] == 1

    first = model({"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})
    second = model({"a": quabla.Matrix([[2.0, 4.0], [6.0, 8.0]])})

    assert first["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert second["a"].to_list() == [[4.0, 8.0], [12.0, 16.0]]
    assert calls["count"] == 1


def test_value_and_grad_fn_returns_value_and_traces_once():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a * a).sum()

    value_and_grad = quabla.value_and_grad_fn(
        model,
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    assert calls["count"] == 1

    first_value, first_gradients = value_and_grad(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )
    second_value, second_gradients = value_and_grad(
        {"a": quabla.Matrix([[2.0, 4.0], [6.0, 8.0]])}
    )

    assert first_value.to_list() == [[30.0]]
    assert first_gradients["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert second_value.to_list() == [[120.0]]
    assert second_gradients["a"].to_list() == [[4.0, 8.0], [12.0, 16.0]]
    assert calls["count"] == 1


def test_vjp_fn_traces_once_and_uses_runtime_cotangent():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a**2) + (a * 3.0)

    vjp_model = quabla.vjp_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    primal, gradients = vjp_model(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])},
        quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )

    assert primal.to_list() == [[4.0, 10.0], [18.0, 28.0]]
    assert gradients["a"].to_list() == [[5.0, 3.5], [-9.0, 22.0]]
    assert calls["count"] == 1


def test_jacobian_fn_traces_once_and_returns_dense_jacobian():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a**2) + (a * 3.0)

    jacobian_model = quabla.jacobian_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    jacobian = jacobian_model(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert jacobian.shape == (4, 4)
    assert jacobian.to_list() == [
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 7.0, 0.0, 0.0],
        [0.0, 0.0, 9.0, 0.0],
        [0.0, 0.0, 0.0, 11.0],
    ]
    assert calls["count"] == 1


def test_jacobians_fn_traces_once_and_returns_dense_jacobians_for_each_input():
    calls = {"count": 0}

    def model(a, b):
        calls["count"] += 1
        return a + b

    jacobians_model = quabla.jacobians_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
    )

    assert calls["count"] == 1

    jacobians = jacobians_model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
        }
    )

    expected_identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert set(jacobians) == {"a", "b"}
    assert jacobians["a"].shape == (4, 4)
    assert jacobians["a"].to_list() == expected_identity
    assert jacobians["b"].shape == (4, 4)
    assert jacobians["b"].to_list() == expected_identity
    assert calls["count"] == 1


def test_jvp_fn_traces_once_and_returns_primal_and_tangent():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a**2) + (a * 3.0)

    jvp_model = quabla.jvp_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    primal, tangent = jvp_model(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])},
        {"a": quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]])},
    )

    assert primal.to_list() == [[4.0, 10.0], [18.0, 28.0]]
    assert tangent.to_list() == [[5.0, 3.5], [-9.0, 22.0]]
    assert calls["count"] == 1


def test_jvp_fn_supports_multiple_input_tangents():
    calls = {"count": 0}

    def model(a, b):
        calls["count"] += 1
        return a * b

    jvp_model = quabla.jvp_fn(model, [("a", (2, 2)), ("b", (2, 2))])

    assert calls["count"] == 1

    primal, tangent = jvp_model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
        },
        {
            "a": quabla.Matrix([[0.5, 1.0], [1.5, 2.0]]),
            "b": quabla.Matrix([[2.0, 3.0], [4.0, 5.0]]),
        },
    )

    assert primal.to_list() == [[10.0, 40.0], [90.0, 160.0]]
    assert tangent.to_list() == [[7.0, 26.0], [57.0, 100.0]]
    assert calls["count"] == 1


def test_jit_and_vjp_support_transpose():
    @quabla.jit([("a", (2, 3))])
    def transpose_primal(a):
        return a.T

    output = transpose_primal(
        {"a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    )

    assert output.shape == (3, 2)
    assert output.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    transpose_vjp = quabla.vjp_fn(lambda a: a.transpose(), [("a", (2, 3))])
    primal, gradients = transpose_vjp(
        {"a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        quabla.Matrix([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    )

    assert primal.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert gradients["a"].shape == (2, 3)
    assert gradients["a"].to_list() == [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]


def test_jit_and_vjp_support_scalar_matrix_broadcast():
    @quabla.jit([("a", (2, 2)), ("s", (1, 1))])
    def shift(a, s):
        return a + s

    output = shift(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "s": quabla.Matrix([[10.0]]),
        }
    )

    assert output.to_list() == [[11.0, 12.0], [13.0, 14.0]]

    scaled_sum_grad = quabla.grad_fn(
        lambda a, s: (a * s).sum(),
        [("a", (2, 2)), ("s", (1, 1))],
        quabla.Matrix([[1.0]]),
    )
    gradients = scaled_sum_grad(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "s": quabla.Matrix([[3.0]]),
        }
    )

    assert gradients["a"].to_list() == [[3.0, 3.0], [3.0, 3.0]]
    assert gradients["s"].shape == (1, 1)
    assert gradients["s"].to_list() == [[10.0]]


def test_jit_and_vjp_support_reshape():
    @quabla.jit([("a", (2, 3))])
    def reshape_primal(a):
        return a.reshape(3, 2)

    output = reshape_primal(
        {"a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    )

    assert output.shape == (3, 2)
    assert output.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    reshape_vjp = quabla.vjp_fn(lambda a: a.reshape(3, 2), [("a", (2, 3))])
    primal, gradients = reshape_vjp(
        {"a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        quabla.Matrix([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    )

    assert primal.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert gradients["a"].shape == (2, 3)
    assert gradients["a"].to_list() == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]


def test_jit_and_grad_support_scalar_literals():
    @quabla.jit([("a", (2, 2))])
    def affine(a):
        return (a * 2.0) + 1.0

    output = affine({"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})

    assert output.to_list() == [[3.0, 5.0], [7.0, 9.0]]

    grad_affine_sum = quabla.grad_fn(
        lambda a: ((a * 2.0) + 1.0).sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_affine_sum(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert gradients["a"].to_list() == [[2.0, 2.0], [2.0, 2.0]]


def test_jit_and_grad_support_neg_and_scalar_division():
    @quabla.jit([("a", (2, 2))])
    def normalized(a):
        return -a / 2.0

    output = normalized({"a": quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])})

    assert output.to_list() == [[-0.5, 1.0], [-1.5, 2.0]]

    grad_normalized_sum = quabla.grad_fn(
        lambda a: (-a / 2.0).sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_normalized_sum(
        {"a": quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])}
    )

    assert gradients["a"].to_list() == [[-0.5, -0.5], [-0.5, -0.5]]


def test_jit_and_grad_support_elementwise_division():
    @quabla.jit([("a", (2, 2)), ("b", (2, 2))])
    def ratio(a, b):
        return a / b

    output = ratio(
        {
            "a": quabla.Matrix([[2.0, 6.0], [12.0, 20.0]]),
            "b": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert output.to_list() == [[2.0, 3.0], [4.0, 5.0]]

    grad_ratio_sum = quabla.grad_fn(
        lambda a, b: (a / b).sum(),
        [("a", (2, 2)), ("b", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_ratio_sum(
        {
            "a": quabla.Matrix([[2.0, 6.0], [12.0, 20.0]]),
            "b": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[1.0, 0.5], [1.0 / 3.0, 0.25]],
    )
    assert_close_rows(
        gradients["b"].to_list(),
        [[-2.0, -1.5], [-4.0 / 3.0, -1.25]],
    )


def test_jit_and_vjp_support_row_and_column_matrix_broadcast():
    @quabla.jit([("a", (2, 3)), ("row", (1, 3)), ("column", (2, 1))])
    def broadcasted(a, row, column):
        return (a + row) * column

    values = {
        "a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "row": quabla.Matrix([[10.0, 20.0, 30.0]]),
        "column": quabla.Matrix([[2.0], [3.0]]),
    }

    output = broadcasted(values)

    assert output.to_list() == [[22.0, 44.0, 66.0], [42.0, 75.0, 108.0]]

    grad_broadcasted_sum = quabla.grad_fn(
        lambda a, row, column: (((a + row) * column).sum()),
        [("a", (2, 3)), ("row", (1, 3)), ("column", (2, 1))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_broadcasted_sum(values)

    assert gradients["a"].to_list() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert gradients["row"].to_list() == [[5.0, 5.0, 5.0]]
    assert gradients["column"].to_list() == [[66.0], [75.0]]


def test_jit_and_vjp_support_where_masks():
    @quabla.jit([("a", (2, 3))])
    def relu_like(a):
        return quabla.where(a.gt(0.0), a, a * 0.1)

    values = {
        "a": quabla.Matrix([[-1.0, 0.5, 2.0], [3.0, -4.0, 5.0]]),
    }

    output = relu_like(values)

    assert output.to_list() == [[-0.1, 0.5, 2.0], [3.0, -0.4, 5.0]]

    grad_relu_like_sum = quabla.grad_fn(
        lambda a: quabla.where(a.gt(0.0), a, a * 0.1).sum(),
        [("a", (2, 3))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_relu_like_sum(values)

    assert gradients["a"].to_list() == [[0.1, 1.0, 1.0], [1.0, 0.1, 1.0]]


def test_jit_and_grad_support_integer_power():
    @quabla.jit([("a", (2, 2))])
    def square(a):
        return a**2

    output = square({"a": quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])})

    assert output.to_list() == [[1.0, 4.0], [9.0, 16.0]]

    grad_square_sum = quabla.grad_fn(
        lambda a: (a**2).sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_square_sum(
        {"a": quabla.Matrix([[1.0, -2.0], [3.0, -4.0]])}
    )

    assert gradients["a"].to_list() == [[2.0, -4.0], [6.0, -8.0]]


def test_jit_and_grad_support_float_power():
    @quabla.jit([("a", (2, 2))])
    def cube(a):
        return a**3.0

    output = cube({"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})

    assert output.to_list() == [[1.0, 8.0], [27.0, 64.0]]

    grad_cube_sum = quabla.grad_fn(
        lambda a: (a**3.0).sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    gradients = grad_cube_sum(
        {"a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert gradients["a"].to_list() == [[3.0, 12.0], [27.0, 48.0]]


def test_jit_and_vjp_support_axis_reductions():
    @quabla.jit([("a", (2, 3))])
    def column_means(a):
        return a.mean(axis=0)

    @quabla.jit([("a", (2, 3))])
    def row_totals(a):
        return a.sum(axis=1)

    values = {"a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

    assert column_means(values).to_list() == [[2.5, 3.5, 4.5]]
    assert row_totals(values).to_list() == [[6.0], [15.0]]

    grad_column_means = quabla.grad_fn(
        lambda a: a.mean(axis=0).sum(),
        [("a", (2, 3))],
        quabla.Matrix([[1.0]]),
    )
    grad_row_totals = quabla.grad_fn(
        lambda a: a.sum(axis=1).sum(),
        [("a", (2, 3))],
        quabla.Matrix([[1.0]]),
    )

    assert grad_column_means(values)["a"].to_list() == [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
    assert grad_row_totals(values)["a"].to_list() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]


def test_jit_and_vjp_support_concat():
    @quabla.jit([("a", (1, 2)), ("b", (2, 2))])
    def vertical_concat(a, b):
        return quabla.concat([a, b], axis=0)

    @quabla.jit([("left", (2, 1)), ("right", (2, 2))])
    def horizontal_concat(left, right):
        return quabla.concat([left, right], axis=1)

    vertical_values = {
        "a": quabla.Matrix([[1.0, 2.0]]),
        "b": quabla.Matrix([[3.0, 4.0], [5.0, 6.0]]),
    }
    horizontal_values = {
        "left": quabla.Matrix([[1.0], [2.0]]),
        "right": quabla.Matrix([[3.0, 4.0], [5.0, 6.0]]),
    }

    assert vertical_concat(vertical_values).to_list() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    assert horizontal_concat(horizontal_values).to_list() == [
        [1.0, 3.0, 4.0],
        [2.0, 5.0, 6.0],
    ]

    grad_vertical = quabla.grad_fn(
        lambda a, b: quabla.concat([a, b], axis=0).sum(),
        [("a", (1, 2)), ("b", (2, 2))],
        quabla.Matrix([[1.0]]),
    )
    grad_horizontal = quabla.grad_fn(
        lambda left, right: quabla.concat([left, right], axis=1).sum(),
        [("left", (2, 1)), ("right", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    vertical_gradients = grad_vertical(vertical_values)
    horizontal_gradients = grad_horizontal(horizontal_values)

    assert vertical_gradients["a"].to_list() == [[1.0, 1.0]]
    assert vertical_gradients["b"].to_list() == [[1.0, 1.0], [1.0, 1.0]]
    assert horizontal_gradients["left"].to_list() == [[1.0], [1.0]]
    assert horizontal_gradients["right"].to_list() == [[1.0, 1.0], [1.0, 1.0]]


def test_grad_fn_evaluates_tanh_vjp():
    grad_model = quabla.grad_fn(
        lambda a: a.tanh().sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [
            [1.0 - math.tanh(0.0) ** 2, 1.0 - math.tanh(1.0) ** 2],
            [1.0 - math.tanh(-1.0) ** 2, 1.0 - math.tanh(2.0) ** 2],
        ],
    )


def test_grad_fn_evaluates_exp_vjp():
    grad_model = quabla.grad_fn(
        lambda a: a.exp().sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[math.exp(0.0), math.exp(1.0)], [math.exp(-1.0), math.exp(2.0)]],
    )


def test_grad_fn_evaluates_log_vjp():
    grad_model = quabla.grad_fn(
        lambda a: a.log().sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[1.0, 2.0], [4.0, 8.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[1.0, 0.5], [0.25, 0.125]],
    )


def test_grad_fn_evaluates_sqrt_vjp():
    grad_model = quabla.grad_fn(
        lambda a: a.sqrt().sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[1.0, 4.0], [9.0, 16.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[0.5, 0.25], [1.0 / 6.0, 0.125]],
    )


def test_grad_fn_evaluates_sin_vjp():
    grad_model = quabla.grad_fn(
        lambda a: a.sin().sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[math.cos(0.0), math.cos(1.0)], [math.cos(-1.0), math.cos(2.0)]],
    )


def test_grad_fn_evaluates_cos_vjp():
    grad_model = quabla.grad_fn(
        lambda a: a.cos().sum(),
        [("a", (2, 2))],
        quabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": quabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[-math.sin(0.0), -math.sin(1.0)], [-math.sin(-1.0), -math.sin(2.0)]],
    )


def test_jit_decorator_factory_evaluates_primal():
    @quabla.jit([("a", (2, 3)), ("b", (3, 2)), ("bias", (2, 2))])
    def model(a, b, bias):
        return (a @ b) + bias

    output = model(
        {
            "a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": quabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "bias": quabla.Matrix([[0.1, 0.2], [0.3, 0.4]]),
        }
    )

    assert output.shape == (2, 2)
    assert output.to_list() == [[58.1, 64.2], [139.3, 154.4]]


def test_jit_traces_once_at_decoration_time():
    calls = {"count": 0}

    @quabla.jit([("a", (2, 2)), ("b", (2, 2))])
    def model(a, b):
        calls["count"] += 1
        return a + b

    assert calls["count"] == 1

    first = model(
        {
            "a": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": quabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        }
    )
    second = model(
        {
            "a": quabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
            "b": quabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert first.to_list() == [[6.0, 8.0], [10.0, 12.0]]
    assert second.to_list() == [[11.0, 22.0], [33.0, 44.0]]
    assert calls["count"] == 1


def test_grad_fn_evaluates_composed_add_matmul_vjp():
    def model(a, b, bias):
        return (a @ b) + bias

    grad_model = quabla.grad_fn(
        model,
        [("a", (2, 3)), ("b", (3, 2)), ("bias", (2, 2))],
        quabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )
    gradients = grad_model(
        {
            "a": quabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": quabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "bias": quabla.Matrix([[0.1, 0.2], [0.3, 0.4]]),
        }
    )

    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]
    assert gradients["bias"].to_list() == [[1.0, 0.5], [-1.0, 2.0]]


def f32(values):
    # Python-side IEEE f32 rounding reference: array("f") stores C floats.
    return array.array("f", values).tolist()


def test_dtype_objects_compare_hash_and_print():
    assert quabla.float32 == quabla.float32
    assert quabla.float32 != quabla.float64
    assert {quabla.float32: "single", quabla.float64: "double"}[quabla.float32] == "single"
    assert repr(quabla.float32) == "quabla.float32"
    assert str(quabla.float64) == "f64"
    assert quabla.float32.name == "float32"


def test_float32_tensor_rounds_values_and_converts_with_astype():
    single = quabla.Tensor([2], [0.1, 0.2], dtype=quabla.float32)
    assert single.to_flat_list() == f32([0.1, 0.2])
    assert single.dtype == quabla.float32
    assert quabla.Tensor([2], [0.1, 0.2]).dtype == quabla.float64
    assert "float32" in repr(single)

    widened = single.astype(quabla.float64)
    assert widened.dtype == quabla.float64
    assert widened.to_flat_list() == f32([0.1, 0.2])
    narrowed = quabla.Tensor([1], [0.1]).astype(quabla.float32)
    assert narrowed.to_flat_list() == f32([0.1])

    # Eager ops: f32 tensors with weak scalars stay f32 and round per element; tensors of different
    # dtypes need an explicit cast.
    other = quabla.Tensor([2], [0.3, 0.7], dtype=quabla.float32)
    expected = [f32([f32([a * b])[0] + f32([0.1])[0]])[0] for a, b in zip(f32([0.1, 0.2]), f32([0.3, 0.7]))]
    result = single * other + 0.1
    assert result.dtype == quabla.float32
    assert result.to_flat_list() == expected
    assert (single.reshape([1, 2]) @ other.reshape([2, 1])).dtype == quabla.float32
    assert single.sum().dtype == quabla.float32
    try:
        single + quabla.Tensor([2], [0.3, 0.7])
    except ValueError as error:
        assert "astype" in str(error)
    else:
        raise AssertionError("mixing float32 and float64 tensors must fail")


def test_trace_tensor_accepts_typed_specs_and_keeps_two_tuple_specs_float64():
    traced = quabla.trace_tensor(
        lambda x, y: x * y + 0.1,
        [("x", [2], quabla.float32), ("y", [2], quabla.float32)],
    )
    assert traced.output.dtype == quabla.float32
    assert "tensor<2xf32>" in traced.graph.lower_text()
    plan = traced.output.compile_cpu()
    plan.validate_kernel_ir()
    assert {node["dtype"] for node in plan.kernel_ir()} == {"f32"}
    inputs = {
        "x": quabla.Tensor([2], [0.1, 1.0 / 3.0]),
        "y": quabla.Tensor([2], [0.3, 3.0]),
    }
    value = plan.evaluate(inputs)
    assert value.dtype == quabla.float32
    x, y = f32([0.1, 1.0 / 3.0]), f32([0.3, 3.0])
    assert value.to_flat_list() == [
        f32([f32([a * b])[0] + f32([0.1])[0]])[0] for a, b in zip(x, y)
    ]

    legacy = quabla.trace_tensor(lambda x: x + 1.0, [("x", [2])])
    assert legacy.output.dtype == quabla.float64
    assert "f32" not in legacy.graph.lower_text()
    assert {node["dtype"] for node in legacy.output.compile_cpu().kernel_ir()} == {"f64"}

    try:
        quabla.trace_tensor(lambda x, y: x + y, [("x", [2], quabla.float32), ("y", [2])])
    except ValueError as error:
        assert "astype" in str(error)
    else:
        raise AssertionError("mixing float32 and float64 traced tensors must fail")
    try:
        quabla.trace_tensor(lambda x: x, [("x", [2], "f32")])
    except TypeError as error:
        assert "(name, shape, dtype)" in str(error)
    else:
        raise AssertionError("a string dtype must be rejected")


def test_trace_tensor_astype_rounds_and_differentiates_through_casts():
    traced = quabla.trace_tensor(
        lambda x: x.astype(quabla.float32).powi(2).sum(), [("x", [3])]
    )
    assert traced.output.dtype == quabla.float32
    point = [0.3, -1.7, 2.2]
    inputs = {"x": quabla.Tensor([3], point), "seed": quabla.Tensor([], [1.0])}
    gradient = traced.symbolic_vjp("seed")["x"]
    assert gradient.output.dtype == quabla.float64
    values = gradient.output.compile_cpu().evaluate(inputs).to_flat_list()
    assert_close_rows([values], [[2.0 * value for value in f32(point)]], tol=1e-6)

    round_trip = quabla.trace_tensor(
        lambda x: x.astype(quabla.float32).astype(quabla.float64), [("x", [1])]
    )
    result = round_trip.output.compile_cpu().evaluate({"x": quabla.Tensor([1], [0.1])})
    assert result.dtype == quabla.float64
    assert result.to_flat_list() == [0.10000000149011612]

    value_and_grad = quabla.tensor_value_and_grad_fn(
        lambda w: (w * w).sum(), [("w", [2], quabla.float32)]
    )
    value, gradients = value_and_grad({"w": quabla.Tensor([2], [0.1, 0.2])})
    assert value.dtype == quabla.float32
    assert gradients["w"].dtype == quabla.float32
    assert_close_rows([gradients["w"].to_flat_list()], [[0.2, 0.4]], tol=1e-7)


def test_float32_fori_region_uses_a_float32_loop_index():
    traced = quabla.trace_tensor(
        lambda initial: quabla.tensor_fori_loop_region(
            0, 4, lambda index, carry: carry + index * 0.1, initial, []
        ),
        [("initial", [], quabla.float32)],
    )
    assert traced.output.dtype == quabla.float32
    result = traced.output.compile_cpu().evaluate({"initial": quabla.Tensor([], [1.0])})
    expected = 1.0
    for index in range(4):
        expected = f32([expected + f32([f32([float(index)])[0] * f32([0.1])[0]])[0]])[0]
    assert result.to_flat_list() == [expected]


def f32_mlp_value_and_gradients(dtype):
    specs = [
        ("x", [8, 3], dtype),
        ("w1", [3, 16], dtype),
        ("b1", [1, 16], dtype),
        ("w2", [16, 1], dtype),
        ("b2", [1, 1], dtype),
    ]
    traced = quabla.trace_tensor(
        lambda x, w1, b1, w2, b2: (((x @ w1 + b1).tanh() @ w2 + b2) * 0.5).powi(2).mean(),
        specs,
    )
    gradients = traced.symbolic_vjp("seed")
    outputs = [traced.output] + [gradients[name].output for name, _, _ in specs[1:]]

    def values(count, seed):
        return [math.sin((index + 1.0) * seed) * 0.9 for index in range(count)]

    inputs = {
        "x": quabla.Tensor([8, 3], values(24, 0.37)),
        "w1": quabla.Tensor([3, 16], values(48, 1.13)),
        "b1": quabla.Tensor([1, 16], values(16, 0.71)),
        "w2": quabla.Tensor([16, 1], values(16, 2.03)),
        "b2": quabla.Tensor([1, 1], [0.1]),
        "seed": quabla.Tensor([], [1.0]),
    }
    return outputs, inputs


def max_scaled_error(actual, expected):
    return max(
        abs(a - e) / max(1.0, abs(e))
        for actual_row, expected_row in zip(actual, expected)
        for a, e in zip(actual_row, expected_row)
    )


def assert_float32_mlp_device_parity(compile_device):
    # Device and CPU f32 references share the same rounded inputs and differ only in per-op rounding
    # and reduction order; a few f32 ulps per op (about 6e-8 at unit scale) stay below 1e-6 on a
    # graph about 10 layers deep, ten times tighter than the existing 1e-5 tolerance against the f64
    # reference.
    outputs, inputs = f32_mlp_value_and_gradients(quabla.float32)
    reference_outputs, _ = f32_mlp_value_and_gradients(quabla.float64)
    cpu = [output.compile_cpu().evaluate(inputs) for output in outputs]
    device = [compile_device(output).evaluate(inputs) for output in outputs]
    reference = [output.compile_cpu().evaluate(inputs) for output in reference_outputs]
    assert all(value.dtype == quabla.float32 for value in cpu + device)
    device_rows = [value.to_flat_list() for value in device]
    assert max_scaled_error(device_rows, [value.to_flat_list() for value in cpu]) <= 1e-6
    assert max_scaled_error(device_rows, [value.to_flat_list() for value in reference]) <= 1e-5


def test_mlx_float32_mlp_matches_the_cpu_float32_reference():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return
    assert_float32_mlp_device_parity(lambda output: output.compile_mlx())


def test_cuda_float32_mlp_matches_the_cpu_float32_reference():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return
    assert_float32_mlp_device_parity(lambda output: output.compile_cuda())


# ---- Dtype phase D2: bool masks, comparisons, and logical reductions ----

NAN = float("nan")
INF = float("inf")
BOOL_X = [1.0, 2.0, NAN, INF, -INF, 2.0]
BOOL_Y = [0.5, 2.0, 2.0, INF, 0.0, NAN]
# (method, operator or None, expected mask of x ? y)
COMPARISONS = [
    ("greater", lambda a, b: a > b, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("greater_equal", lambda a, b: a >= b, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
    ("less", lambda a, b: a < b, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ("less_equal", lambda a, b: a <= b, [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
    ("equal", None, [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
    ("not_equal", None, [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
]


def bool_xy():
    return quabla.Tensor([2, 3], BOOL_X), quabla.Tensor([2, 3], BOOL_Y)


def expect_error(action, *fragments, error=ValueError):
    try:
        action()
    except error as caught:
        for fragment in fragments:
            assert fragment in str(caught), (fragment, str(caught))
    else:
        raise AssertionError(f"expected {error.__name__} containing {fragments}")


def test_bool_dtype_object_and_eager_comparisons_follow_ieee_semantics():
    assert repr(quabla.bool_) == "quabla.bool_"
    assert str(quabla.bool_) == "bool"
    assert quabla.bool_.name == "bool"
    assert {quabla.bool_: "mask"}[quabla.bool_] == "mask"
    assert quabla.bool_ != quabla.float64
    mask = quabla.Tensor([4], [0.0, 3.0, NAN, -INF], dtype=quabla.bool_)
    assert mask.dtype == quabla.bool_
    assert mask.to_flat_list() == [0.0, 1.0, 1.0, 1.0]
    assert "quabla.bool_" in repr(mask)

    x, y = bool_xy()
    for name, operator, expected in COMPARISONS:
        results = [getattr(x, name)(y), getattr(quabla, name)(x, y)]
        if operator is not None:
            results.append(operator(x, y))
        for result in results:
            assert result.dtype == quabla.bool_, name
            assert result.to_flat_list() == expected, name
    # Python scalars are weakly typed; a scalar on the left goes through the reflected operators and
    # functions.
    assert (x > 1.5).to_flat_list() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert (1.5 < x).to_flat_list() == (x > 1.5).to_flat_list()
    assert quabla.greater(1.5, x).to_flat_list() == (x < 1.5).to_flat_list()
    single = quabla.Tensor([2], [0.1, 0.2], dtype=quabla.float32)
    assert single.equal(0.1).to_flat_list() == [1.0, 0.0]
    assert x.astype(quabla.bool_).to_flat_list() == [1.0] * 6
    assert (x > 1.5).astype(quabla.float32).dtype == quabla.float32


def test_bool_logical_ops_and_reductions_match_between_eager_and_traced():
    x, y = bool_xy()

    def program(x, y):
        greater = x > y
        return [
            x.isfinite(),
            quabla.isnan(x),
            greater & x.isfinite(),
            quabla.logical_or(x < y, y.isnan()),
            ~x.equal(y),
            quabla.logical_not(greater),
            x.isnan().any(),
            y.isfinite().all(),
            greater.any(axis=1),
            y.isfinite().all(axis=0, keepdims=True),
            x.isfinite().all(axis=[0, 1]),
        ]

    expected = [
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0],
        [0.0],
        [1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0],
    ]
    eager = program(x, y)
    assert [value.to_flat_list() for value in eager] == expected
    assert all(value.dtype == quabla.bool_ for value in eager)
    assert eager[9].shape == [1, 3]

    inputs = {"x": x, "y": y}
    for index, row in enumerate(expected):
        traced = quabla.trace_tensor(
            lambda x, y, index=index: program(x, y)[index],
            [("x", [2, 3]), ("y", [2, 3])],
        )
        assert traced.output.dtype == quabla.bool_
        plan = traced.output.compile_cpu()
        plan.validate_kernel_ir()
        assert plan.kernel_ir()[-1]["dtype"] == "bool"
        value = plan.evaluate(inputs)
        assert value.dtype == quabla.bool_
        assert value.to_flat_list() == row, index
    assert (
        "xi1>" in quabla.trace_tensor(lambda x: x > 0.0, [("x", [2])]).graph.lower_text()
    )

    expect_error(lambda: x & y, "logical_and requires bool operands")
    expect_error(lambda: x.any(), "any requires bool operands")
    expect_error(lambda: x.isfinite().isnan(), "isnan is not defined for bool tensors")
    expect_error(
        lambda: quabla.trace_tensor(lambda x: x.all(), [("x", [2])]),
        "all requires bool operands",
    )


def test_tensor_ordering_operators_keep_identity_equality_and_hashing():
    x, y = bool_xy()
    assert x == x and not (x == y) and x != y
    assert {x: "x", y: "y"}[x] == "x"
    seen = []

    def record(a, b):
        seen.extend([a == a, a == b, {a: 1}[a]])
        return a > b

    quabla.trace_tensor(record, [("a", [2]), ("b", [2])])
    assert seen == [True, False, 1]


def test_legacy_gt_masks_and_extrema_tie_rules_are_unchanged():
    x = quabla.Tensor([4], [-1.0, 0.0, 2.0, NAN])
    mask = x.gt(0.0)
    assert mask.dtype == quabla.float64
    assert mask.to_flat_list() == [0.0, 0.0, 1.0, 0.0]
    single = quabla.Tensor([2], [0.5, -0.5], dtype=quabla.float32)
    assert single.gt(0.0).dtype == quabla.float32
    traced = quabla.trace_tensor(lambda x: x.gt(0.0), [("x", [4])])
    assert traced.output.dtype == quabla.float64
    assert traced.output.compile_cpu().evaluate({"x": x}).to_flat_list() == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    ties = quabla.Tensor([3], [-1.0, 0.0, 1.0])
    other = quabla.Tensor([3], [1.0, 0.0, -1.0])
    assert ties.abs().to_flat_list() == [1.0, -0.0, 1.0]
    for name, expected_gradient in [
        ("relu", [0.0, 0.0, 1.0]),
        ("abs", [-1.0, -1.0, 1.0]),
    ]:
        _, gradients = quabla.tensor_value_and_grad_fn(
            lambda x, name=name: getattr(x, name)().sum(), [("x", [3])]
        )({"x": ties})
        assert gradients["x"].to_flat_list() == expected_gradient
    # The subgradient of maximum/minimum at ties is still routed to the right operand.
    _, gradients = quabla.tensor_value_and_grad_fn(
        lambda a, b: a.maximum(b).sum() + a.minimum(b).sum(), [("a", [3]), ("b", [3])]
    )({"a": ties, "b": other})
    assert gradients["a"].to_flat_list() == [1.0, 0.0, 1.0]
    assert gradients["b"].to_flat_list() == [1.0, 2.0, 1.0]


def test_bool_operands_promote_to_float_and_bool_arithmetic_is_rejected():
    residual = quabla.Tensor([3], [0.1, -0.2, 0.3], dtype=quabla.float32)
    x = quabla.Tensor([3], [1.0, -1.0, 2.0])
    mask = x > 0.0
    masked = residual * mask
    assert masked.dtype == quabla.float32
    assert masked.to_flat_list() == f32([0.1, 0.0, 0.3])
    weak = mask * 2.0
    assert weak.dtype == quabla.float64
    adopted = weak + residual
    assert adopted.dtype == quabla.float32
    assert adopted.to_flat_list() == f32(
        [f32([2.0])[0] + f32([0.1])[0], -0.2, f32([2.0])[0] + f32([0.3])[0]]
    )
    expect_error(lambda: weak + weak.astype(quabla.float32) + x, "astype")

    specs = [("r", [3], quabla.float32), ("x", [3])]
    traced = quabla.trace_tensor(lambda r, x: r * (x > 0.0), specs)
    assert traced.output.dtype == quabla.float32
    traced = quabla.trace_tensor(lambda r, x: (x > 0.0) * 2.0 + r, specs)
    assert traced.output.dtype == quabla.float32
    values = traced.output.compile_cpu().evaluate({"r": residual, "x": x})
    assert values.to_flat_list() == adopted.to_flat_list()

    for action in [lambda m: m + m, lambda m: m * m]:
        expect_error(lambda: action(mask), "not defined for bool tensors", "astype")
        expect_error(
            lambda: quabla.trace_tensor(lambda x: action(x > 0.0), [("x", [3])]),
            "not defined for bool tensors",
        )
    for name in [
        "tanh",
        "exp",
        "sqrt",
        "sum",
        "relu",
        "abs",
        "sigmoid",
        "softplus",
        "norm",
        "max",
    ]:
        expect_error(
            lambda: getattr(mask, name)(), f"{name} is not defined for bool tensors"
        )
        expect_error(
            lambda: quabla.trace_tensor(
                lambda x: getattr(x > 0.0, name)(), [("x", [3])]
            ),
            f"{name} is not defined for bool tensors",
        )
    expect_error(lambda: -mask, "not defined for bool tensors")


def test_eager_bool_truthiness_follows_numpy_and_traced_tensors_refuse():
    x = quabla.Tensor([2], [1.0, -1.0])
    taken = "yes" if (x.sum() > 0.5) else "no"
    assert taken == "no"
    assert bool((x > 0.0).any())
    assert not bool((x > 0.0).all())
    expect_error(lambda: bool(x > 0.0), "ambiguous", ".any()")
    # Float tensors keep their original object truthiness (always true).
    assert bool(quabla.Tensor([1], [0.0]))
    expect_error(
        lambda: quabla.trace_tensor(lambda x: x if bool(x > 0.0) else x, [("x", [])]),
        "control flow",
        error=TypeError,
    )


def test_nan_guards_keep_values_and_gradients_finite_while_float_masks_leak():
    x = quabla.Tensor([2, 3], BOOL_X)
    guarded = quabla.tensor_value_and_grad_fn(
        lambda x: quabla.where(x.isfinite(), x, 0.0).powi(2).sum(), [("x", [2, 3])]
    )
    value, gradients = guarded({"x": x})
    assert value.to_flat_list() == [9.0]
    assert gradients["x"].to_flat_list() == [2.0, 4.0, 0.0, 0.0, 0.0, 4.0]

    # all(isfinite(x)) as a Cond predicate: non-finite inputs take the guarded branch, with finite
    # values and gradients.
    traced = quabla.trace_tensor(
        lambda x: quabla.tensor_cond(
            x.isfinite().all(),
            lambda a: (a * a).sum(),
            lambda a: quabla.where(a.isfinite(), a, 0.0).powi(2).sum() * 0.5,
            [x],
        ),
        [("x", [2, 3])],
    )
    gradient = traced.symbolic_vjp("seed")["x"].output
    for values, expected_value, expected_gradient in [
        (BOOL_X, 4.5, [1.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 91.0, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
    ]:
        inputs = {"x": quabla.Tensor([2, 3], values), "seed": quabla.Tensor([], [1.0])}
        assert traced.output.compile_cpu().evaluate(inputs).to_flat_list() == [
            expected_value
        ]
        assert (
            gradient.compile_cpu().evaluate(inputs).to_flat_list() == expected_gradient
        )

    # Legacy form: multiplying by a float mask gives 0 * NaN = NaN, and gt cannot detect NaN either.
    leaked = (x.gt(0.0) * x).sum().to_flat_list()[0]
    assert math.isnan(leaked)


def test_where_and_cond_accept_bool_and_legacy_float_predicates():
    x = quabla.Tensor([3], [-2.0, 0.0, 3.0])
    for mask in [x >= 0.0, x.gt(0.0)]:
        selected = quabla.where(mask, x, 0.0 - x)
        assert selected.to_flat_list() == [2.0, 0.0, 3.0]
        traced = quabla.trace_tensor(
            lambda x, float_mask=(mask.dtype == quabla.float64): quabla.where(
                x.gt(0.0) if float_mask else x >= 0.0, x, 0.0 - x
            ),
            [("x", [3])],
        )
        assert traced.output.compile_cpu().evaluate({"x": x}).to_flat_list() == [
            2.0,
            0.0,
            3.0,
        ]

    for predicate in [lambda x: (x > 0.0).any(), lambda x: x.sum().gt(0.0)]:
        traced = quabla.trace_tensor(
            lambda x, predicate=predicate: quabla.tensor_cond(
                predicate(x), lambda a: a.sum(), lambda a: (a * 2.0).sum(), [x]
            ),
            [("x", [3])],
        )
        for values, expected in [([-2.0, 0.0, 3.0], 1.0), ([-2.0, -1.0, 0.0], -6.0)]:
            result = traced.output.compile_cpu().evaluate(
                {"x": quabla.Tensor([3], values)}
            )
            assert result.to_flat_list() == [expected]


def masked_residual_loss(x, w, mask):
    # PINN style: the mask selects points with large residuals, and a Bool mask input further
    # restricts the active points.
    active = (x > 0.5) & mask
    residual = quabla.where(active, x * w - 1.0, 0.0)
    return residual.powi(2).mean() + (x * w).powi(2).sum() * 0.1


MASKED_SPECS = [("x", [2, 3]), ("w", [2, 3]), ("mask", [2, 3], quabla.bool_)]


def masked_inputs():
    return {
        "x": quabla.Tensor([2, 3], [1.0, 2.0, -0.3, 0.7, 0.2, 2.0]),
        "w": quabla.Tensor([2, 3], [0.5, -1.5, 2.0, 0.25, 1.0, -0.75]),
        "mask": quabla.Tensor([2, 3], [1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=quabla.bool_),
    }


def test_masked_loss_gradients_match_finite_differences_and_skip_bool_inputs():
    inputs = masked_inputs()
    value, gradients = quabla.tensor_value_and_grad_fn(
        masked_residual_loss, MASKED_SPECS
    )(inputs)
    assert sorted(gradients) == ["w", "x"]
    step = 1e-6
    for name in ["x", "w"]:
        base = inputs[name].to_flat_list()
        for index in range(6):
            shifted = []
            for delta in (step, -step):
                point = list(base)
                point[index] += delta
                moved = dict(inputs, **{name: quabla.Tensor([2, 3], point)})
                shifted.append(
                    masked_residual_loss(moved["x"], moved["w"], moved["mask"])
                )
            difference = (shifted[0] - shifted[1]).to_flat_list()[0] / (2.0 * step)
            assert abs(gradients[name].to_flat_list()[index] - difference) < 1e-6
    assert value.to_flat_list() == masked_residual_loss(**inputs).to_flat_list()

    expect_error(
        lambda: quabla.tensor_hessian_scalar_fn(
            masked_residual_loss, MASKED_SPECS, "mask"
        )(inputs),
        'bool input "mask"',
    )
    traced = quabla.trace_tensor(masked_residual_loss, MASKED_SPECS)
    expect_error(lambda: traced.symbolic_jvp("mask"), 'bool input "mask"')
    expect_error(
        lambda: quabla.trace_tensor(lambda x: x > 0.0, [("x", [2])]).symbolic_vjp(
            "seed"
        ),
        "bool output",
    )
    if os.environ.get("QUABLA_MLX_TEST") is not None:
        expect_error(
            lambda: quabla.tensor_value_and_grad_mlx_fn(
                masked_residual_loss, MASKED_SPECS, ["mask"]
            ),
            'bool input "mask"',
        )


def assert_bool_device_parity(compile_device):
    x, y = bool_xy()
    inputs = {"x": x, "y": y}
    outputs = [
        lambda x, y: x > y,
        lambda x, y: x >= y,
        lambda x, y: x < y,
        lambda x, y: x <= y,
        lambda x, y: x.equal(y),
        lambda x, y: x.not_equal(y),
        lambda x, y: (x > y) & x.isfinite(),
        lambda x, y: (x < y) | y.isnan(),
        lambda x, y: ~x.isfinite(),
        lambda x, y: (x > y).any(axis=1),
        lambda x, y: y.isfinite().all(axis=0, keepdims=True),
        lambda x, y: quabla.where(x.isfinite(), x, 0.0) * (x > 0.5),
    ]
    for index, function in enumerate(outputs):
        traced = quabla.trace_tensor(function, [("x", [2, 3]), ("y", [2, 3])])
        cpu = traced.output.compile_cpu().evaluate(inputs)
        device = compile_device(traced.output).evaluate(inputs)
        assert device.dtype == cpu.dtype, index
        assert device.to_flat_list() == cpu.to_flat_list(), index

    # any/all as Cond predicates, checking values and gradients of the NaN-guarded branch.
    traced = quabla.trace_tensor(
        lambda x: quabla.tensor_cond(
            x.isfinite().all(),
            lambda a: (a * a).sum(),
            lambda a: quabla.where(a.isfinite(), a, 0.0).powi(2).sum() * 0.5,
            [x],
        ),
        [("x", [2, 3])],
    )
    gradient = traced.symbolic_vjp("seed")["x"].output
    for values in [BOOL_X, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]:
        cond_inputs = {
            "x": quabla.Tensor([2, 3], values),
            "seed": quabla.Tensor([], [1.0]),
        }
        for output in [traced.output, gradient]:
            assert (
                compile_device(output).evaluate(cond_inputs).to_flat_list()
                == output.compile_cpu().evaluate(cond_inputs).to_flat_list()
            )

    traced = quabla.trace_tensor(masked_residual_loss, MASKED_SPECS)
    gradients = traced.symbolic_vjp("seed")
    inputs = dict(masked_inputs(), seed=quabla.Tensor([], [1.0]))
    for output in [traced.output, gradients["x"].output, gradients["w"].output]:
        cpu = output.compile_cpu().evaluate(inputs).to_flat_list()
        device = compile_device(output).evaluate(inputs).to_flat_list()
        assert max_scaled_error([device], [cpu]) <= 1e-5


def test_mlx_bool_masks_and_guarded_gradients_match_cpu():
    if os.environ.get("QUABLA_MLX_TEST") is None:
        return
    assert_bool_device_parity(lambda output: output.compile_mlx())


def test_cuda_bool_masks_and_guarded_gradients_match_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") is None:
        return
    assert_bool_device_parity(lambda output: output.compile_cuda())


if __name__ == "__main__":
    # Run every test_* function in definition order so new tests cannot be left out of a manual list
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
