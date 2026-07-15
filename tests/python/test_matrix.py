import math
import os

import nabla


def assert_close_rows(actual, expected, tol=1e-12):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert abs(actual_value - expected_value) <= tol


def test_matrix_matmul():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = nabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    c = a @ b

    assert c.shape == (2, 2)
    assert c.to_list() == [[58.0, 64.0], [139.0, 154.0]]


def test_tensor_solve_and_jit_vjp():
    matrix = nabla.Tensor([2, 2], [3.0, 1.0, 1.0, 2.0])
    rhs = nabla.Tensor([2, 1], [9.0, 8.0])
    assert matrix.solve(rhs).to_flat_list() == [2.0, 3.0]

    value_and_grad = nabla.tensor_value_and_grad_fn(
        lambda a, b: a.solve(b).sum(), [("a", [2, 2]), ("b", [2, 1])]
    )
    value, gradients = value_and_grad({"a": matrix, "b": rhs})
    assert value.to_flat_list() == [5.0]
    assert_close_rows([gradients["a"].to_flat_list()], [[-0.4, -0.6, -0.8, -1.2]])
    assert_close_rows([gradients["b"].to_flat_list()], [[0.2, 0.4]])


def test_tensor_neural_primitives_and_trace_gradients():
    values = nabla.Tensor([3], [-1.0, 0.0, 1.0])
    assert values.relu().to_flat_list() == [0.0, 0.0, 1.0]
    assert values.abs().to_flat_list() == [1.0, -0.0, 1.0]

    sigmoid = values.sigmoid().to_flat_list()
    softplus = values.softplus().to_flat_list()
    expected_sigmoid = [1.0 / (1.0 + math.exp(-value)) for value in [-1.0, 0.0, 1.0]]
    assert_close_rows([sigmoid], [expected_sigmoid])
    assert_close_rows(
        [softplus], [[math.log1p(math.exp(value)) for value in [-1.0, 0.0, 1.0]]]
    )
    assert nabla.Tensor([2], [-1000.0, 1000.0]).softplus().to_flat_list() == [0.0, 1000.0]

    for name, expected_gradient in [
        ("relu", [0.0, 0.0, 1.0]),
        ("abs", [-1.0, -1.0, 1.0]),
        ("sigmoid", [value * (1.0 - value) for value in expected_sigmoid]),
        ("softplus", expected_sigmoid),
    ]:
        transform = nabla.tensor_value_and_grad_fn(
            lambda x, name=name: getattr(x, name)().sum(), [("x", [3])]
        )
        _, gradients = transform({"x": values})
        assert_close_rows([gradients["x"].to_flat_list()], [expected_gradient])


def test_tensor_stateless_random_keys_and_glorot_initializer_are_reproducible():
    first_keys = nabla.Tensor.split_key(1234, 2)
    second_keys = nabla.Tensor.split_key(1234, 2)
    assert first_keys == second_keys
    assert first_keys[0] != first_keys[1]

    first = nabla.Tensor.random_normal([2, 3], first_keys[0])
    second = nabla.Tensor.random_normal([2, 3], first_keys[0])
    different = nabla.Tensor.random_normal([2, 3], first_keys[1])
    assert first.shape == [2, 3]
    assert first.to_flat_list() == second.to_flat_list()
    assert first.to_flat_list() != different.to_flat_list()

    glorot = nabla.Tensor.glorot_normal([2, 3], first_keys[0])
    assert glorot.shape == [2, 3]
    assert all(math.isfinite(value) for value in glorot.to_flat_list())


def test_matrix_add():
    a = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = nabla.Matrix([[0.5, 1.5], [2.5, 3.5]])

    c = a + b

    assert c.shape == (2, 2)
    assert c.to_list() == [[1.5, 3.5], [5.5, 7.5]]


def test_matrix_sub():
    a = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = nabla.Matrix([[0.5, 1.5], [2.5, 3.5]])

    c = a - b

    assert c.shape == (2, 2)
    assert c.to_list() == [[0.5, 0.5], [0.5, 0.5]]


def test_matrix_mul():
    a = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = nabla.Matrix([[0.5, 1.5], [2.5, 3.5]])

    c = a * b

    assert c.shape == (2, 2)
    assert c.to_list() == [[0.5, 3.0], [7.5, 14.0]]


def test_matrix_transpose_method_and_property():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    transposed = a.transpose()

    assert transposed.shape == (3, 2)
    assert transposed.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert a.T.to_list() == transposed.to_list()


def test_matrix_reshape_preserves_row_major_order():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    reshaped = a.reshape(3, 2)

    assert reshaped.shape == (3, 2)
    assert reshaped.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_matrix_reshape_rejects_size_change():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    try:
        a.reshape(4, 2)
    except ValueError as exc:
        assert "cannot reshape" in str(exc)
    else:
        raise AssertionError("expected reshape size change to fail")


def test_matrix_concat_axis():
    top = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    bottom = nabla.Matrix([[5.0, 6.0]])
    left = nabla.Matrix([[1.0], [2.0]])
    right = nabla.Matrix([[3.0, 4.0], [5.0, 6.0]])

    vertical = nabla.concat([top, bottom], axis=0)
    horizontal = nabla.concat([left, right], axis=1)

    assert vertical.shape == (3, 2)
    assert vertical.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert horizontal.shape == (2, 3)
    assert horizontal.to_list() == [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]


def test_matrix_supports_scalar_literals():
    a = nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])

    assert (a + 1.5).to_list() == [[2.5, -0.5], [4.5, -2.5]]
    assert (1.5 + a).to_list() == [[2.5, -0.5], [4.5, -2.5]]
    assert (a - 1.5).to_list() == [[-0.5, -3.5], [1.5, -5.5]]
    assert (5.0 - a).to_list() == [[4.0, 7.0], [2.0, 9.0]]
    assert (a * 2.0).to_list() == [[2.0, -4.0], [6.0, -8.0]]
    assert (2.0 * a).to_list() == [[2.0, -4.0], [6.0, -8.0]]


def test_matrix_broadcasts_scalar_matrix_for_elementwise_ops():
    a = nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])
    scalar = nabla.Matrix([[2.0]])

    assert (a + scalar).to_list() == [[3.0, 0.0], [5.0, -2.0]]
    assert (scalar + a).to_list() == [[3.0, 0.0], [5.0, -2.0]]
    assert (a - scalar).to_list() == [[-1.0, -4.0], [1.0, -6.0]]
    assert (scalar - a).to_list() == [[1.0, 4.0], [-1.0, 6.0]]
    assert (a * scalar).to_list() == [[2.0, -4.0], [6.0, -8.0]]
    assert (scalar * a).to_list() == [[2.0, -4.0], [6.0, -8.0]]


def test_matrix_broadcasts_row_and_column_matrices_for_elementwise_ops():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = nabla.Matrix([[10.0, 20.0, 30.0]])
    column = nabla.Matrix([[2.0], [3.0]])

    assert (a + row).to_list() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]
    assert (row - a).to_list() == [[9.0, 18.0, 27.0], [6.0, 15.0, 24.0]]
    assert (a * column).to_list() == [[2.0, 4.0, 6.0], [12.0, 15.0, 18.0]]
    assert (a / column).to_list() == [[0.5, 1.0, 1.5], [4.0 / 3.0, 5.0 / 3.0, 2.0]]


def test_tensor_broadcasts_trailing_axes_and_reshapes():
    a = nabla.Tensor([2, 1, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = nabla.Tensor([1, 4, 1], [10.0, 20.0, 30.0, 40.0])

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
        nabla.Tensor([2, 2], [1.0, 2.0, 3.0])
    except ValueError as exc:
        assert "data length" in str(exc)
    else:
        raise AssertionError("expected tensor data size mismatch to fail")


def test_tensor_ones_and_full_validate_rank_n_shapes():
    assert nabla.Tensor.ones([2, 3]).to_flat_list() == [1.0] * 6
    assert nabla.Tensor.full([2, 1, 2], -0.25).to_flat_list() == [-0.25] * 4


def test_tensor_arange_creates_coordinate_vectors_and_rejects_invalid_ranges():
    assert nabla.Tensor.arange(0.0, 1.0, 0.25).to_flat_list() == [0.0, 0.25, 0.5, 0.75]
    assert nabla.Tensor.arange(1.0, -0.5, -0.5).to_flat_list() == [1.0, 0.5, 0.0]
    for arguments in [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)]:
        try:
            nabla.Tensor.arange(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"arange accepted invalid arguments {arguments}")


def test_tensor_linspace_includes_endpoints_for_collocation_grids():
    assert nabla.Tensor.linspace(-1.0, 1.0, 5).to_flat_list() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert nabla.Tensor.linspace(2.0, 5.0, 1).to_flat_list() == [2.0]
    try:
        nabla.Tensor.linspace(0.0, 1.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("linspace accepted num=0")


def test_tensor_eye_creates_square_and_rectangular_identity_arrays():
    assert nabla.Tensor.eye(3).to_flat_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert nabla.Tensor.eye(2, 3).to_flat_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    try:
        nabla.Tensor.eye(0)
    except ValueError:
        pass
    else:
        raise AssertionError("eye accepted zero rows")


def test_tensor_transpose_reorders_rank_n_axes_and_validates_permutations():
    tensor = nabla.Tensor([2, 3, 2], [float(value) for value in range(12)])

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
    tensor = nabla.Tensor([2, 3, 2], [float(value) for value in range(1, 13)])

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
    tensor = nabla.Tensor([2, 2], [0.0, 1.0, -1.0, 4.0])

    assert_close_rows([tensor.tanh().to_flat_list()], [[math.tanh(value) for value in tensor.to_flat_list()]])
    assert_close_rows([tensor.exp().to_flat_list()], [[math.exp(value) for value in tensor.to_flat_list()]])
    assert_close_rows([tensor.sin().to_flat_list()], [[math.sin(value) for value in tensor.to_flat_list()]])
    assert_close_rows([tensor.cos().to_flat_list()], [[math.cos(value) for value in tensor.to_flat_list()]])
    assert tensor.powi(2).to_flat_list() == [0.0, 1.0, 1.0, 16.0]

    positive = nabla.Tensor([2], [1.0, math.e**2])
    assert_close_rows([positive.log().to_flat_list()], [[0.0, 2.0]])
    assert positive.sqrt().to_flat_list() == [1.0, math.e]


def test_tensor_supports_numeric_scalars_on_both_sides():
    tensor = nabla.Tensor([2, 2], [1.0, -2.0, 3.0, 4.0])

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
    tensor = nabla.Tensor([2], [4.0, 9.0])

    assert (tensor ** 2).to_flat_list() == [16.0, 81.0]
    assert_close_rows([(tensor ** 0.5).to_flat_list()], [[2.0, 3.0]])

    try:
        pow(tensor, 2, 3)
    except TypeError:
        pass
    else:
        raise AssertionError("tensor power accepted a modulo argument")


def test_tensor_comparisons_and_where_support_rank_n_broadcasting():
    values = nabla.Tensor([2, 1, 3], [-1.0, 0.0, 1.0, 2.0, -2.0, 3.0])
    mask = values.gt(0.0)
    on_true = nabla.Tensor([1, 4, 1], [10.0, 20.0, 30.0, 40.0])
    on_false = nabla.Tensor.full([2, 1, 3], -1.0)

    assert mask.shape == [2, 1, 3]
    assert mask.to_flat_list() == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    selected = nabla.where(mask, on_true, on_false)
    assert selected.shape == [2, 4, 3]
    assert selected.to_flat_list() == [
        -1.0, -1.0, 10.0, -1.0, -1.0, 20.0, -1.0, -1.0, 30.0, -1.0, -1.0, 40.0,
        10.0, -1.0, 10.0, 20.0, -1.0, 20.0, 30.0, -1.0, 30.0, 40.0, -1.0, 40.0,
    ]


def test_tensor_maximum_and_minimum_broadcast_and_choose_rhs_on_ties():
    left = nabla.Tensor([2, 1], [2.0, 0.0])
    right = nabla.Tensor([1, 3], [2.0, 1.0, -1.0])

    assert left.maximum(right).shape == [2, 3]
    assert left.maximum(right).to_flat_list() == [2.0, 2.0, 2.0, 2.0, 1.0, 0.0]
    assert left.minimum(right).to_flat_list() == [2.0, 1.0, -1.0, 0.0, 0.0, -1.0]
    assert left.maximum(1.0).to_flat_list() == [2.0, 1.0]
    assert left.minimum(1.0).to_flat_list() == [1.0, 0.0]


def test_tensor_gather_and_scatter_add_support_negative_and_repeated_indices():
    source = nabla.Tensor([2, 3], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert source.gather([2, -3], axis=1).to_flat_list() == [2.0, 0.0, 5.0, 3.0]

    base = nabla.Tensor.zeros([2, 3])
    updates = nabla.Tensor([2, 3], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
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
    left = nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    right = nabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.0, 3.0, 1.0])
    expected = (left @ right).to_flat_list()

    assert nabla.einsum("ij,jk->ik", [left, right]).to_flat_list() == expected
    assert nabla.einsum("...ij,...jk->...ik", [left, right]).to_flat_list() == expected

    for equation in ("ij,ij->ij", "ij,jk->ij"):
        try:
            nabla.einsum(equation, [left, right])
        except ValueError:
            pass
        else:
            raise AssertionError(f"einsum accepted unsupported equation {equation}")


def test_tensor_broadcast_to_materializes_rank_n_contiguous_storage():
    tensor = nabla.Tensor([1, 2, 1], [2.0, -3.0])
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
    lhs = nabla.Tensor(
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
    rhs = nabla.Tensor([1, 3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    output = lhs.matmul(rhs)

    assert output.shape == [2, 2, 2]
    assert output.to_flat_list() == [58.0, 64.0, 139.0, 154.0, 25.0, 28.0, 56.0, 62.0]


def test_tensor_concat_supports_rank_n_axes_and_validates_shapes():
    lhs = nabla.Tensor([2, 1, 2], [1.0, 2.0, 5.0, 6.0])
    rhs = nabla.Tensor([2, 2, 2], [3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    result = nabla.concat([lhs, rhs], axis=1)

    assert result.shape == [2, 3, 2]
    assert result.to_flat_list() == [1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0]
    try:
        nabla.concat([lhs, nabla.Tensor([2, 2, 3], [0.0] * 12)], axis=1)
    except ValueError as error:
        assert "cannot concatenate" in str(error)
    else:
        raise AssertionError("concat accepted incompatible Tensor shapes")


def test_tensor_slice_creates_a_strided_read_only_view():
    tensor = nabla.Tensor([3, 4], [float(value) for value in range(12)])

    view = tensor.slice(axis=1, start=1, length=2, step=2)

    assert view.shape == [3, 2]
    assert view.strides == [4, 2]
    assert view.offset == 1
    assert view.to_flat_list() == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    assert view.to_tensor().to_flat_list() == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]


def test_tensor_getitem_supports_contiguous_slices_and_negative_indices():
    tensor = nabla.Tensor([3, 4], [float(value) for value in range(12)])

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
    first = nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    second = nabla.Tensor([2, 2], [5.0, 6.0, 7.0, 8.0])
    stacked = nabla.stack([first, second], axis=-1)
    assert stacked.shape == [2, 2, 2]
    assert stacked.to_flat_list() == [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]


def test_trace_tensor_evaluates_rank_n_scalar_loss_vjp():
    def loss(x, y):
        return (x * y + x).sum()

    traced = nabla.trace_tensor(loss, [("x", [2, 1, 3]), ("y", [1, 4, 1])])
    inputs = {
        "x": nabla.Tensor([2, 1, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "y": nabla.Tensor([1, 4, 1], [10.0, 20.0, 30.0, 40.0]),
    }

    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
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
            "x": nabla.Tensor([2, 1, 3], [1.0] * 6),
            "y": nabla.Tensor([1, 4, 1], [1.0] * 4),
        },
    )
    assert tangent.shape == []
    assert tangent.to_flat_list() == [708.0]


def test_trace_tensor_batched_matmul_scalar_loss_vjp():
    def loss(x, y):
        return x.matmul(y).sum()

    traced = nabla.trace_tensor(loss, [("x", [2, 2, 3]), ("y", [1, 3, 2])])
    inputs = {
        "x": nabla.Tensor(
            [2, 2, 3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0],
        ),
        "y": nabla.Tensor([1, 3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    }

    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )

    assert value.to_flat_list() == [586.0]
    assert gradients["x"].to_flat_list() == [15.0, 19.0, 23.0] * 4
    assert gradients["y"].to_flat_list() == [8.0, 8.0, 10.0, 10.0, 12.0, 12.0]


def test_trace_tensor_symbolic_jvp_keeps_parameter_gradients():
    traced = nabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [1]), ("weight", [1]), ("forcing", [1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    assert second_derivative.output.shape == [1]
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).sum()
    inputs = {
        "x": nabla.Tensor([1], [0.3]),
        "weight": nabla.Tensor([1], [1.2]),
        "forcing": nabla.Tensor([1], [0.5]),
    }
    _, gradients = second_derivative.graph.evaluate_value_and_vjp(
        residual_loss.node_id,
        inputs,
        nabla.Tensor([], [1.0]),
    )
    assert abs(gradients["weight"].to_flat_list()[0]) > 1e-8


def test_trace_tensor_symbolic_vjp_matches_cpu_vjp():
    traced = nabla.trace_tensor(
        lambda x, bias: (x + bias).tanh().sum(),
        [("x", [2, 1]), ("bias", [1, 3])],
    )
    inputs = {
        "x": nabla.Tensor([2, 1], [-1.0, 2.0]),
        "bias": nabla.Tensor([1, 3], [0.0, 1.0, -1.0]),
    }
    cotangent = nabla.Tensor([], [1.0])
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
    traced = nabla.trace_tensor(
        lambda x, weight: (x * weight).tanh(),
        [("x", [1]), ("weight", [1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    loss = second_derivative.output.powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    inputs = {
        "x": nabla.Tensor([1], [0.5]),
        "weight": nabla.Tensor([1], [0.3]),
        "loss_cotangent": nabla.Tensor([], [1.0]),
    }
    gradient = gradients["weight"].graph.evaluate(
        gradients["weight"].output.node_id, inputs
    )

    assert abs(gradient.to_flat_list()[0]) > 1e-8


def test_trace_tensor_concat_supports_rank_n_ad_and_symbolic_transforms():
    traced = nabla.trace_tensor(
        lambda left, right: nabla.concat([left, right], axis=1).powi(2).sum(),
        [("left", [2, 1, 2]), ("right", [2, 2, 2])],
    )
    inputs = {
        "left": nabla.Tensor([2, 1, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": nabla.Tensor([2, 2, 2], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    }
    cotangent = nabla.Tensor([], [1.0])
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
            "left": nabla.Tensor.ones([2, 1, 2]),
            "right": nabla.Tensor.zeros([2, 2, 2]),
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
    traced = nabla.trace_tensor(
        lambda x: x.slice(-2, 1, 3).powi(2).sum(),
        [("x", [2, 4, 2])],
    )
    inputs = {"x": nabla.Tensor([2, 4, 2], [float(value) for value in range(1, 17)])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
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
        {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])},
    ).to_flat_list() == gradients["x"].to_flat_list()


def test_trace_tensor_getitem_preserves_slice_ad_and_backend_parity():
    traced = nabla.trace_tensor(lambda x: x[1, 1:3].powi(2).sum(), [("x", [3, 4])])
    inputs = {"x": nabla.Tensor([3, 4], [float(value) for value in range(12)])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [61.0]
    assert gradients["x"].to_flat_list() == [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    symbolic = traced.symbolic_vjp("loss_cotangent")["x"]
    symbolic_value = symbolic.graph.evaluate(
        symbolic.output.node_id,
        {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])},
    )
    assert symbolic_value.to_flat_list() == gradients["x"].to_flat_list()

    cpu = traced.output.compile_cpu().evaluate(inputs)
    if os.environ.get("NABLA_MLX_TEST") is not None:
        mlx = traced.output.compile_mlx().evaluate(inputs)
        assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)
    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda = traced.output.compile_cuda().evaluate(inputs)
        assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_trace_tensor_slice_rejects_invalid_ranges():
    traced = nabla.trace_tensor(lambda x: x, [("x", [2, 3])])
    for args in ((0, 3, 2), (1, 0, 4), (2, 0, 1)):
        try:
            traced.output.slice(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"slice{args} should reject an invalid range")


def test_trace_tensor_broadcast_to_supports_rank_n_ad_and_symbolic_transforms():
    traced = nabla.trace_tensor(
        lambda x: x.broadcast_to([2, 3, 2]).powi(2).sum(), [("x", [1, 3, 1])]
    )
    inputs = {"x": nabla.Tensor([1, 3, 1], [1.0, 2.0, 3.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [56.0]
    assert gradients["x"].to_flat_list() == [8.0, 16.0, 24.0]
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        inputs,
        {"x": nabla.Tensor.ones([1, 3, 1])},
    )
    assert tangent.to_flat_list() == [48.0]
    symbolic_gradient = traced.symbolic_vjp("loss_cotangent")["x"]
    assert symbolic_gradient.graph.evaluate(
        symbolic_gradient.output.node_id,
        {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])},
    ).to_flat_list() == gradients["x"].to_flat_list()


def test_trace_tensor_stack_supports_symbolic_vjp():
    traced = nabla.trace_tensor(
        lambda left, right: nabla.stack([left, right], axis=-1).powi(2).sum(),
        [("left", [2, 2]), ("right", [2, 2])],
    )
    inputs = {
        "left": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": nabla.Tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    }
    gradient = traced.symbolic_vjp("loss_cotangent")["right"]
    result = gradient.graph.evaluate(
        gradient.output.node_id,
        {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])},
    )
    assert result.to_flat_list() == [10.0, 12.0, 14.0, 16.0]


def test_cuda_trace_tensor_concat_keeps_primal_and_symbolic_vjp_on_device():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda left, right: nabla.concat([left, right], axis=1).powi(2).sum(),
        [("left", [2, 1, 2]), ("right", [2, 2, 2])],
    )
    inputs = {
        "left": nabla.Tensor([2, 1, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": nabla.Tensor([2, 2, 2], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    }
    assert_close_rows([traced.output.compile_cuda().evaluate(inputs).to_flat_list()], [[650.0]], tol=1e-5)

    gradient = traced.symbolic_vjp("loss_cotangent")["right"].output.compile_cuda()
    result = gradient.evaluate({**inputs, "loss_cotangent": nabla.Tensor([], [1.0])})
    assert_close_rows(
        [result.to_flat_list()],
        [[10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]],
        tol=1e-5,
    )


def test_mlx_trace_tensor_compiles_and_matches_cpu():
    if os.environ.get("NABLA_MLX_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, y, bias: nabla.concat([x, y], axis=0)
        .add(bias.broadcast_to([2, 2]))
        .tanh()
        .sum(),
        [("x", [1, 2]), ("y", [1, 2]), ("bias", [1, 2])],
    )
    inputs = {
        "x": nabla.Tensor([1, 2], [-1.0, 2.0]),
        "y": nabla.Tensor([1, 2], [0.5, -0.25]),
        "bias": nabla.Tensor([1, 2], [0.25, -0.5]),
    }
    cpu = traced.output.compile_cpu().evaluate(inputs)
    plan = traced.output.compile_mlx()
    assert plan.backend == "mlx"
    assert_close_rows([plan.evaluate(inputs).to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_mlx_symbolic_vjp_executes_mlp_bias_gradient():
    if os.environ.get("NABLA_MLX_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: (x.matmul(weight) + bias).tanh().sum(),
        [("x", [2, 2]), ("weight", [2, 3]), ("bias", [1, 3])],
    )
    inputs = {
        "x": nabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": nabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.5, 1.5, -0.5]),
        "bias": nabla.Tensor([1, 3], [0.25, -0.5, 1.0]),
        "loss_cotangent": nabla.Tensor([], [1.0]),
    }
    gradient = traced.symbolic_vjp("loss_cotangent")["bias"].output
    cpu = gradient.compile_cpu().evaluate(inputs)
    mlx = gradient.compile_mlx().evaluate(inputs)
    assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_mlx_trace_tensor_executes_masked_loss():
    if os.environ.get("NABLA_MLX_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x: nabla.where(x.gt(0.0), x.powi(2), x).sum(), [("x", [2, 2])]
    )
    inputs = {"x": nabla.Tensor([2, 2], [-2.0, -1.0, 1.0, 3.0])}
    cpu = traced.output.compile_cpu().evaluate(inputs)
    mlx = traced.output.compile_mlx().evaluate(inputs)
    assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_mlx_symbolic_vjp_executes_masked_loss_gradient():
    if os.environ.get("NABLA_MLX_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x: nabla.where(x.gt(0.0), x.powi(2), x).sum(), [("x", [2, 2])]
    )
    inputs = {
        "x": nabla.Tensor([2, 2], [-2.0, -1.0, 1.0, 3.0]),
        "loss_cotangent": nabla.Tensor([], [1.0]),
    }
    gradient = traced.symbolic_vjp("loss_cotangent")["x"].output
    cpu = gradient.compile_cpu().evaluate(inputs)
    mlx = gradient.compile_mlx().evaluate(inputs)
    assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_cuda_trace_tensor_slice_keeps_primal_and_symbolic_vjp_on_device():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x: x.slice(1, 1, 3).powi(2).sum(), [("x", [2, 4, 2])]
    )
    inputs = {"x": nabla.Tensor([2, 4, 2], [float(value) for value in range(1, 17)])}
    assert_close_rows([traced.output.compile_cuda().evaluate(inputs).to_flat_list()], [[716.0]], tol=1e-5)
    gradient = traced.symbolic_vjp("loss_cotangent")["x"].output.compile_cuda()
    result = gradient.evaluate({**inputs, "loss_cotangent": nabla.Tensor([], [1.0])})
    assert_close_rows(
        [result.to_flat_list()],
        [[0.0, 0.0, 6.0, 8.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 22.0, 24.0, 26.0, 28.0, 0.0, 0.0]],
        tol=1e-5,
    )


def test_cuda_trace_tensor_broadcast_to_keeps_primal_and_symbolic_vjp_on_device():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x: x.broadcast_to([2, 3, 2]).powi(2).sum(), [("x", [1, 3, 1])]
    )
    inputs = {"x": nabla.Tensor([1, 3, 1], [1.0, 2.0, 3.0])}
    assert_close_rows([traced.output.compile_cuda().evaluate(inputs).to_flat_list()], [[56.0]], tol=1e-5)
    gradient = traced.symbolic_vjp("loss_cotangent")["x"].output.compile_cuda()
    result = gradient.evaluate({**inputs, "loss_cotangent": nabla.Tensor([], [1.0])})
    assert_close_rows([result.to_flat_list()], [[8.0, 16.0, 24.0]], tol=1e-5)


def test_cuda_multi_parameter_sgd_keeps_gradient_plans_synchronized():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    weight_gradient = gradients["weight"].output.compile_cuda()
    bias_gradient = gradients["bias"].output.compile_cuda()
    inputs = {
        "x": nabla.Tensor([2], [-1.0, 1.0]),
        "target": nabla.Tensor([2], [-1.0, 3.0]),
        "loss_cotangent": nabla.Tensor([], [1.0]),
        "weight": nabla.Tensor([1], [0.0]),
        "bias": nabla.Tensor([1], [0.0]),
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
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight: x * weight,
        [("x", [1]), ("weight", [1])],
    )
    target = traced.graph.input("target", [1])
    loss = (traced.output - target).powi(2).sum()
    gradient = loss.symbolic_vjp("loss_cotangent")["weight"].output.compile_cuda()
    inputs = {
        "x": nabla.Tensor([1], [2.0]),
        "weight": nabla.Tensor([1], [0.0]),
        "target": nabla.Tensor([1], [6.0]),
        "loss_cotangent": nabla.Tensor([], [1.0]),
    }

    for _ in range(2):
        gradient.adam_step(inputs, "weight", 0.1, ["x", "target", "loss_cotangent"])

    weight = gradient.retained_input("weight").to_flat_list()[0]
    assert 0.19 < weight < 0.21


def test_cuda_plan_evaluates_with_static_inputs_retained_on_device():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, bias: (x + bias).tanh(),
        [("x", [2, 1]), ("bias", [1, 2])],
    )
    plan = traced.output.compile_cuda()
    inputs = {
        "x": nabla.Tensor([2, 1], [-1.0, 2.0]),
        "bias": nabla.Tensor([1, 2], [0.0, 1.0]),
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
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: ((x.matmul(weight) + bias).tanh() + bias),
        [("x", [2, 2]), ("weight", [2, 2]), ("bias", [1, 2])],
    )
    plan = traced.output.compile_cuda()
    inputs = {
        "x": nabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": nabla.Tensor([2, 2], [1.0, 2.0, -1.0, 1.0]),
        "bias": nabla.Tensor([1, 2], [0.25, -0.5]),
    }

    plan.evaluate(inputs)
    first = plan.evaluate(inputs).to_flat_list()
    first_buffer_count = plan.device_buffer_count
    second = plan.evaluate(inputs).to_flat_list()

    assert first == second
    assert first_buffer_count == plan.device_buffer_count
    assert first_buffer_count < plan.node_count


def test_cuda_fuses_rank_two_matmul_bias_tanh_epilogue():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: (x.matmul(weight) + bias).tanh(),
        [("x", [2, 2]), ("weight", [2, 3]), ("bias", [1, 3])],
    )
    plan = traced.output.compile_cuda()
    inputs = {
        "x": nabla.Tensor([2, 2], [-1.0, 0.0, 1.0, 2.0]),
        "weight": nabla.Tensor([2, 3], [1.0, -1.0, 2.0, 0.5, 1.5, -0.5]),
        "bias": nabla.Tensor([1, 3], [0.25, -0.5, 1.0]),
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


def test_cuda_multi_parameter_adam_keeps_gradient_plans_synchronized():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    weight_gradient = gradients["weight"].output.compile_cuda()
    bias_gradient = gradients["bias"].output.compile_cuda()
    inputs = {
        "x": nabla.Tensor([2], [-1.0, 1.0]),
        "target": nabla.Tensor([2], [-1.0, 3.0]),
        "loss_cotangent": nabla.Tensor([], [1.0]),
        "weight": nabla.Tensor([1], [0.0]),
        "bias": nabla.Tensor([1], [0.0]),
    }
    retained = ["x", "target", "loss_cotangent", "weight", "bias"]
    optimizer = nabla.cuda_adam_optimizer(
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
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight: x * weight,
        [("x", [1]), ("weight", [1])],
    )
    target = traced.graph.input("target", [1])
    loss = (traced.output - target).powi(2).sum()
    gradient = loss.symbolic_vjp("loss_cotangent")["weight"].output

    def make_optimizer() -> object:
        return nabla.cuda_adam_optimizer(
            {"weight": gradient.compile_cuda()},
            {
                "x": nabla.Tensor([1], [1.0]),
                "target": nabla.Tensor([1], [1.0]),
                "loss_cotangent": nabla.Tensor([], [1.0]),
                "weight": nabla.Tensor([1], [0.0]),
            },
            0.1,
            ["x", "target", "loss_cotangent", "weight"],
        )

    static = make_optimizer()
    dynamic = make_optimizer()
    static.step()
    dynamic.step()
    static.step()
    dynamic.step({"target": nabla.Tensor([1], [-10.0])})

    static_weight = static.parameters()["weight"].to_flat_list()[0]
    dynamic_weight = dynamic.parameters()["weight"].to_flat_list()[0]
    assert abs(static_weight - dynamic_weight) > 1e-4


def test_cuda_adam_vjp_optimizer_updates_parameters_from_one_shared_graph():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    optimizer = nabla.cuda_adam_vjp_optimizer(
        {"weight": gradients["weight"], "bias": gradients["bias"]},
        {
            "x": nabla.Tensor([2], [-1.0, 1.0]),
            "target": nabla.Tensor([2], [-1.0, 3.0]),
            "loss_cotangent": nabla.Tensor([], [1.0]),
            "weight": nabla.Tensor([1], [0.0]),
            "bias": nabla.Tensor([1], [0.0]),
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
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda lhs, rhs: lhs @ rhs,
        [("lhs", [2, 3, 2]), ("rhs", [1, 2, 4])],
    )
    loss = traced.output.sum()
    gradients = loss.symbolic_vjp("loss_cotangent")
    inputs = {
        "lhs": nabla.Tensor(
            [2, 3, 2],
            [1.0, -2.0, 0.5, 3.0, -1.0, 4.0, 2.0, 1.0, -3.0, 0.25, 0.75, -2.0],
        ),
        "rhs": nabla.Tensor(
            [1, 2, 4], [0.2, -0.4, 0.6, 0.8, -0.1, 0.3, 0.5, -0.7]
        ),
        "loss_cotangent": nabla.Tensor([], [1.0]),
    }

    for name, gradient_trace in gradients.items():
        cpu = gradient_trace.graph.evaluate(gradient_trace.output.node_id, inputs)
        cuda = gradient_trace.output.compile_cuda().evaluate(inputs)
        assert cuda.shape == cpu.shape
        for actual, expected in zip(cuda.to_flat_list(), cpu.to_flat_list()):
            assert abs(actual - expected) < 1e-5


def test_cuda_global_reductions_match_cpu_and_reset_output_buffers():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    values = [0.125 * (index % 13) for index in range(8192)]
    inputs = {"x": nabla.Tensor([128, 64], values)}

    for fn in (lambda x: x.sum(), lambda x: x.mean()):
        traced = nabla.trace_tensor(fn, [("x", [128, 64])])
        cpu = traced.graph.evaluate(traced.output.node_id, inputs).to_flat_list()[0]
        cuda = traced.output.compile_cuda()
        first = cuda.evaluate(inputs).to_flat_list()[0]
        second = cuda.evaluate(inputs).to_flat_list()[0]

        assert abs(first - cpu) <= 1e-3
        assert abs(second - cpu) <= 1e-3


def test_multi_axis_keepdims_reductions_match_cpu_on_mlx_and_cuda():
    if os.environ.get("NABLA_MLX_TEST") is None and os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x: x.mean(axis=[0, 2], keepdims=True), [("x", [2, 3, 2])]
    )
    inputs = {"x": nabla.Tensor([2, 3, 2], [float(value) for value in range(1, 13)])}
    cpu = traced.output.compile_cpu().evaluate(inputs)

    if os.environ.get("NABLA_MLX_TEST") is not None:
        mlx = traced.output.compile_mlx().evaluate(inputs)
        assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)

    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda = traced.output.compile_cuda().evaluate(inputs)
        assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_cuda_sqrt_composite_lowering_matches_cpu():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(lambda x: x.sqrt(), [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], [0.25, 1.0, 4.0, 9.0])}
    cpu = traced.graph.evaluate(traced.output.node_id, inputs)
    cuda = traced.output.compile_cuda().evaluate(inputs)

    for actual, expected in zip(cuda.to_flat_list(), cpu.to_flat_list()):
        assert abs(actual - expected) <= 1e-5


def test_cuda_checked_div_and_log_lowering_matches_cpu():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, scale: (x / scale).log(),
        [("x", [2, 2]), ("scale", [])],
    )
    inputs = {
        "x": nabla.Tensor([2, 2], [0.5, 1.0, 2.0, 8.0]),
        "scale": nabla.Tensor([], [0.5]),
    }
    cpu = traced.graph.evaluate(traced.output.node_id, inputs)
    cuda = traced.output.compile_cuda().evaluate(inputs)

    for actual, expected in zip(cuda.to_flat_list(), cpu.to_flat_list()):
        assert abs(actual - expected) <= 1e-5


def test_trace_tensor_tanh_scalar_loss_supports_jvp_and_vjp():
    def loss(x):
        return x.tanh().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
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
        traced.output.node_id, "x", inputs, nabla.Tensor([2, 2], direction)
    )
    assert_close_rows([hvp.to_flat_list()], [[second * direction_item for second, direction_item in zip(expected_second, direction)]])


def test_trace_tensor_subtraction_supports_jvp_and_vjp():
    def loss(x):
        return (x.tanh() - x).sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [-math.tanh(item) ** 2 for item in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_supports_numeric_scalar_literals():
    def loss(x):
        return (2.0 * x + 1.0).sum()

    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    assert value.to_flat_list() == [24.0]
    assert gradients["x"].to_flat_list() == [2.0] * 4
    assert tangent.to_flat_list() == [8.0]


def test_trace_tensor_division_supports_jvp_and_vjp():
    def loss(x):
        return (x / (x + 2.0)).sum()

    values = [1.0, 2.0, 3.0, 4.0]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [2.0 / (value + 2.0) ** 2 for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_exp_supports_jvp_and_vjp():
    def loss(x):
        return x.exp().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.exp(value) for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_mean_supports_jvp_and_vjp():
    def loss(x):
        return x.exp().mean()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.exp(value) / 4.0 for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_sin_supports_jvp_and_vjp():
    def loss(x):
        return x.sin().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.cos(value) for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_sqrt_supports_jvp_and_vjp():
    def loss(x):
        return x.sqrt().sum()

    values = [1.0, 4.0, 9.0, 16.0]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [0.5, 0.25, 1.0 / 6.0, 0.125]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_sqrt_and_norm_define_zero_subgradient_and_preserve_symbolic_ad():
    traced = nabla.trace_tensor(lambda x: x.sqrt().sum(), [("x", [3])])
    inputs = {"x": nabla.Tensor([3], [0.0, 1.0, 4.0])}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([3], [1.0, 1.0, 1.0])}
    )
    symbolic = traced.symbolic_vjp("loss_cotangent")["x"]

    assert gradients["x"].to_flat_list() == [0.0, 0.5, 0.25]
    assert tangent.to_flat_list() == [0.75]
    assert symbolic.graph.evaluate(
        symbolic.output.node_id,
        {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])},
    ).to_flat_list() == [0.0, 0.5, 0.25]
    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)
    assert hessian.shape == [3, 3]
    assert_close_rows(
        [hessian.to_flat_list()],
        [[0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, 0.0, -1.0 / 32.0]],
    )

    symbolic_inputs = {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])}
    cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
    if os.environ.get("NABLA_MLX_TEST") is not None:
        mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )
    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )

    norm = nabla.trace_tensor(lambda x: x.norm(), [("x", [2])])
    norm_inputs = {"x": nabla.Tensor([2], [3.0, 4.0])}
    value, norm_gradients = norm.graph.evaluate_value_and_vjp(
        norm.output.node_id, norm_inputs, nabla.Tensor([], [1.0])
    )
    assert value.to_flat_list() == [5.0]
    assert norm_gradients["x"].to_flat_list() == [0.6, 0.8]


def test_trace_tensor_reduction_extrema_choose_last_tied_coordinate_for_gradients():
    inputs = {"x": nabla.Tensor([2, 3], [1.0, 5.0, 5.0, 2.0, 2.0, 0.0])}
    cotangent = nabla.Tensor([], [1.0])

    for operation, expected_value, expected_gradient in (
        ("max", [7.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("min", [1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ):
        traced = nabla.trace_tensor(
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
        if os.environ.get("NABLA_MLX_TEST") is not None:
            mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
            assert_close_rows(
                [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
            )
        if os.environ.get("NABLA_CUDA_TEST") is not None:
            cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
            assert_close_rows(
                [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
            )


def test_trace_tensor_gather_and_scatter_add_preserve_repeated_index_gradients():
    gather = nabla.trace_tensor(
        lambda x: x.gather([2, 0, 2], axis=1).sum(), [("x", [2, 3])]
    )
    gather_inputs = {"x": nabla.Tensor([2, 3], [0.0] * 6)}
    _, gather_gradients = gather.graph.evaluate_value_and_vjp(
        gather.output.node_id, gather_inputs, nabla.Tensor([], [1.0])
    )
    assert gather_gradients["x"].to_flat_list() == [1.0, 0.0, 2.0, 1.0, 0.0, 2.0]

    scatter = nabla.trace_tensor(
        lambda base, updates, weight: (base.scatter_add([1, 1, 0], updates, axis=1) * weight).sum(),
        [("base", [1, 3]), ("updates", [1, 3]), ("weight", [1, 3])],
    )
    scatter_inputs = {
        "base": nabla.Tensor.zeros([1, 3]),
        "updates": nabla.Tensor([1, 3], [2.0, 3.0, 5.0]),
        "weight": nabla.Tensor([1, 3], [1.0, 2.0, 4.0]),
    }
    _, scatter_gradients = scatter.graph.evaluate_value_and_vjp(
        scatter.output.node_id, scatter_inputs, nabla.Tensor([], [1.0])
    )
    assert scatter_gradients["base"].to_flat_list() == [1.0, 2.0, 4.0]
    assert scatter_gradients["updates"].to_flat_list() == [2.0, 2.0, 1.0]

    symbolic = scatter.symbolic_vjp("loss_cotangent")["updates"]
    symbolic_inputs = {**scatter_inputs, "loss_cotangent": nabla.Tensor([], [1.0])}
    assert symbolic.graph.evaluate(
        symbolic.output.node_id, symbolic_inputs
    ).to_flat_list() == [2.0, 2.0, 1.0]

    cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
    if os.environ.get("NABLA_MLX_TEST") is not None:
        mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )
    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )


def test_trace_tensor_einsum_scoped_matmul_subset_preserves_backend_ad():
    traced = nabla.trace_tensor(
        lambda left, right: nabla.einsum("ij,jk->ik", [left, right]).sum(),
        [("left", [2, 2]), ("right", [2, 2])],
    )
    inputs = {
        "left": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
        "right": nabla.Tensor([2, 2], [1.0, 0.0, 2.0, 1.0]),
    }
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    assert gradients["left"].to_flat_list() == [1.0, 3.0, 1.0, 3.0]
    assert gradients["right"].to_flat_list() == [4.0, 4.0, 6.0, 6.0]

    symbolic = traced.symbolic_vjp("loss_cotangent")["left"]
    symbolic_inputs = {**inputs, "loss_cotangent": nabla.Tensor([], [1.0])}
    cpu_gradient = symbolic.output.compile_cpu().evaluate(symbolic_inputs)
    assert cpu_gradient.to_flat_list() == gradients["left"].to_flat_list()
    if os.environ.get("NABLA_MLX_TEST") is not None:
        mlx_gradient = symbolic.output.compile_mlx().evaluate(symbolic_inputs)
        assert_close_rows(
            [mlx_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )
    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda_gradient = symbolic.output.compile_cuda().evaluate(symbolic_inputs)
        assert_close_rows(
            [cuda_gradient.to_flat_list()], [cpu_gradient.to_flat_list()], tol=1e-5
        )


def test_trace_tensor_where_routes_gradients_and_masks_condition_derivatives():
    def loss(x):
        condition = x.gt(0.0)
        return nabla.where(condition, x.powi(2), x * 3.0).sum()

    traced = nabla.trace_tensor(loss, [("x", [4])])
    inputs = {"x": nabla.Tensor([4], [-2.0, -1.0, 0.0, 2.0])}
    value, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([4], [1.0] * 4)}
    )
    transformed = traced.symbolic_jvp("x")
    transformed_value = transformed.graph.evaluate(transformed.output.node_id, inputs)
    plan = traced.output.compile_cpu()
    plan_value, plan_gradients = plan.evaluate_value_and_vjp(
        inputs, nabla.Tensor([], [1.0])
    )

    assert value.to_flat_list() == [-5.0]
    assert gradients["x"].to_flat_list() == [3.0, 3.0, 3.0, 4.0]
    assert tangent.to_flat_list() == [13.0]
    assert transformed_value.to_flat_list() == [13.0]
    assert plan_value.to_flat_list() == value.to_flat_list()
    assert plan_gradients["x"].to_flat_list() == gradients["x"].to_flat_list()


def test_trace_tensor_maximum_and_minimum_route_tie_gradients_to_rhs():
    left = nabla.Tensor([3], [2.0, 3.0, 0.0])
    right = nabla.Tensor([3], [2.0, 1.0, 4.0])
    cotangent = nabla.Tensor([], [1.0])

    for operation, expected_value, expected_left, expected_right in (
        ("maximum", [2.0, 3.0, 4.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]),
        ("minimum", [2.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]),
    ):
        traced = nabla.trace_tensor(
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
        if os.environ.get("NABLA_MLX_TEST") is not None:
            mlx = traced.output.compile_mlx().evaluate(inputs)
            assert_close_rows([mlx.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)
        if os.environ.get("NABLA_CUDA_TEST") is not None:
            cuda = traced.output.compile_cuda().evaluate(inputs)
            assert_close_rows([cuda.to_flat_list()], [cpu.to_flat_list()], tol=1e-5)


def test_trace_tensor_log_supports_jvp_and_vjp():
    def loss(x):
        return x.log().sum()

    values = [1.0, 2.0, 3.0, 4.0]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [1.0 / value for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_trace_tensor_cos_supports_jvp_and_vjp():
    def loss(x):
        return x.cos().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )
    hessian = traced.graph.hessian_scalar(traced.output.node_id, "x", inputs)
    hvp = traced.graph.hvp_scalar(
        traced.output.node_id,
        "x",
        inputs,
        nabla.Tensor([2, 2], [1.0, 2.0, -1.0, 0.5]),
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
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
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
    traced = nabla.trace_tensor(loss, [("x", [2, 3])])
    inputs = {"x": nabla.Tensor([2, 3], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 3], [1.0] * 6)}
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

    output = nabla.trace_tensor(lambda x: x.sum(axis=1), [("x", [2, 3])])
    value, output_tangent = output.graph.evaluate_jvp(
        output.output.node_id,
        inputs,
        {"x": nabla.Tensor([2, 3], [1.0] * 6)},
    )
    _, output_gradients = output.graph.evaluate_value_and_vjp(
        output.output.node_id, inputs, nabla.Tensor([2], [2.0, 3.0])
    )
    assert value.shape == [2]
    assert value.to_flat_list() == [6.0, 15.0]
    assert output_tangent.to_flat_list() == [3.0, 3.0]
    assert output_gradients["x"].to_flat_list() == [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]

    multi_axis = nabla.trace_tensor(
        lambda x: x.mean(axis=[0, 2], keepdims=True), [("x", [2, 3, 2])]
    )
    multi_inputs = {"x": nabla.Tensor([2, 3, 2], [float(value) for value in range(1, 13)])}
    multi_value, multi_gradients = multi_axis.graph.evaluate_value_and_vjp(
        multi_axis.output.node_id,
        multi_inputs,
        nabla.Tensor([1, 3, 1], [2.0, 3.0, 4.0]),
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

    last_axis = nabla.trace_tensor(lambda x: x.mean(axis=-1), [("x", [2, 3])])
    last_axis_value = last_axis.graph.evaluate(last_axis.output.node_id, inputs)
    assert last_axis_value.shape == [2]
    assert last_axis_value.to_flat_list() == [2.0, 5.0]


def test_trace_tensor_transpose_supports_rank_n_ad():
    def loss(x):
        return x.transpose([2, 0, 1]).powi(2).sum()

    values = [float(value) for value in range(1, 13)]
    traced = nabla.trace_tensor(loss, [("x", [2, 2, 3])])
    inputs = {"x": nabla.Tensor([2, 2, 3], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2, 3], [1.0] * 12)}
    )

    expected = [2.0 * value for value in values]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12

    output = nabla.trace_tensor(lambda x: x.transpose(), [("x", [2, 2, 3])])
    value = output.graph.evaluate(output.output.node_id, inputs)
    assert value.shape == [3, 2, 2]
    assert value.to_flat_list() == [1.0, 7.0, 4.0, 10.0, 2.0, 8.0, 5.0, 11.0, 3.0, 9.0, 6.0, 12.0]


def test_trace_tensor_reshape_supports_jvp_and_vjp():
    def loss(x):
        return x.reshape([4]).exp().sum()

    values = [0.0, 1.0, -1.0, 0.5]
    traced = nabla.trace_tensor(loss, [("x", [2, 2])])
    inputs = {"x": nabla.Tensor([2, 2], values)}
    _, gradients = traced.graph.evaluate_value_and_vjp(
        traced.output.node_id, inputs, nabla.Tensor([], [1.0])
    )
    _, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id, inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)}
    )

    expected = [math.exp(value) for value in values]
    assert gradients["x"].shape == [2, 2]
    assert_close_rows([gradients["x"].to_flat_list()], [expected])
    assert abs(tangent.to_flat_list()[0] - sum(expected)) <= 1e-12


def test_tensor_grad_scalar_fn_reuses_a_compiled_plan():
    gradient = nabla.tensor_grad_scalar_fn(
        lambda x: (2.0 * x).sum(), [("x", [2, 2])]
    )

    first = gradient({"x": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])})
    second = gradient({"x": nabla.Tensor([2, 2], [5.0, 6.0, 7.0, 8.0])})

    assert first["x"].to_flat_list() == [2.0] * 4
    assert second["x"].to_flat_list() == [2.0] * 4


def test_tensor_value_and_grad_fn_returns_scalar_value_and_gradients():
    value_and_grad = nabla.tensor_value_and_grad_fn(
        lambda x, scale: ((x * scale).sin()).sum(),
        [("x", [2, 2]), ("scale", [])],
    )
    first_value, first_gradients = value_and_grad(
        {
            "x": nabla.Tensor([2, 2], [0.0, 0.5, 1.0, -0.25]),
            "scale": nabla.Tensor([], [2.0]),
        }
    )
    second_value, second_gradients = value_and_grad(
        {
            "x": nabla.Tensor([2, 2], [0.0, 0.5, 1.0, -0.25]),
            "scale": nabla.Tensor([], [1.0]),
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
    hessian = nabla.tensor_hessian_scalar_fn(
        lambda x: x.powi(3).sum(), [("x", [2, 2])], "x"
    )
    hvp = nabla.tensor_hvp_scalar_fn(
        lambda x: x.powi(3).sum(), [("x", [2, 2])], "x"
    )
    values = [0.5, -1.0, 2.0, 0.25]
    direction = [1.0, 2.0, -0.5, 3.0]
    inputs = {"x": nabla.Tensor([2, 2], values)}

    actual_hessian = hessian(inputs)
    actual_hvp = hvp(inputs, nabla.Tensor([2, 2], direction))

    assert len(actual_hessian) == len(values)
    for row, value in enumerate(values):
        for column in range(len(values)):
            expected = 6.0 * value if row == column else 0.0
            assert abs(actual_hessian[row][column] - expected) <= 1e-12
    assert_close_rows(
        [actual_hvp.to_flat_list()],
        [[6.0 * value * tangent for value, tangent in zip(values, direction)]],
    )


def test_tensor_jit_fn_reuses_a_compiled_plan():
    compiled = nabla.tensor_jit_fn(
        lambda x, y: (x.matmul(y)).tanh(),
        [("x", [2, 2]), ("y", [2, 2])],
    )

    first = compiled(
        {
            "x": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
            "y": nabla.Tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
        }
    )
    second = compiled(
        {
            "x": nabla.Tensor([2, 2], [0.0, 1.0, -1.0, 0.5]),
            "y": nabla.Tensor([2, 2], [2.0, 0.0, 0.0, 2.0]),
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


def test_tensor_vmap_fn_traces_one_batched_plan():
    mapped = nabla.tensor_vmap_fn(
        lambda x, weight: x.matmul(weight).tanh(),
        [("x", [2, 2]), ("weight", [2, 1])],
        3,
    )

    result = mapped(
        {
            "x": nabla.Tensor(
                [3, 2, 2],
                [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 3.0],
            ),
            "weight": nabla.Tensor([3, 2, 1], [1.0, -1.0, 1.0, 0.5, 2.0, 1.0]),
        }
    )

    assert result.shape == [3, 2, 1]
    assert_close_rows(
        [result.to_flat_list()],
        [[math.tanh(value) for value in [1.0, -1.0, 2.5, 2.0, 6.0, 3.0]]],
    )

    try:
        nabla.tensor_vmap_fn(lambda x: x, [("x", [2])], 0)
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("tensor_vmap_fn accepted a zero batch size")


def test_tensor_vmap_fn_supports_in_axes_out_axis_and_unmapped_inputs():
    mapped = nabla.tensor_vmap_fn(
        lambda x, weight: (x * weight).sum(),
        [("x", [2]), ("weight", [2])],
        3,
        in_axes=[-1, None],
        out_axis=-1,
    )

    result = mapped(
        {
            "x": nabla.Tensor([2, 3], [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
            "weight": nabla.Tensor([2], [2.0, -1.0]),
        }
    )

    assert result.shape == [3]
    assert_close_rows([result.to_flat_list()], [[0.0, 2.0, 4.0]])


def test_tensor_vmap_fn_reductions_and_transpose_preserve_batch_axis():
    mapped = nabla.tensor_vmap_fn(
        lambda x: x.transpose().mean(axis=1),
        [("x", [2, 3])],
        2,
        in_axes=[2],
        out_axis=1,
    )

    result = mapped(
        {
            "x": nabla.Tensor(
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
        "x": nabla.Tensor(
            [3, 2, 2],
            [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 3.0],
        ),
        "weight": nabla.Tensor([3, 2, 1], [1.0, -1.0, 1.0, 0.5, 2.0, 1.0]),
    }
    expected = [[math.tanh(value) for value in [1.0, -1.0, 2.5, 2.0, 6.0, 3.0]]]
    nonleading_values = {"x": nabla.Tensor([2, 3], [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])}
    nonleading_expected = [[3.0, 7.0, 11.0]]

    if os.environ.get("NABLA_MLX_TEST") is not None:
        mlx_compiled = nabla.tensor_vmap_mlx_fn(
            lambda x, weight: x.matmul(weight).tanh(), input_specs, 3
        )
        assert mlx_compiled.backend == "mlx"
        assert_close_rows([mlx_compiled(values).to_flat_list()], expected, tol=1e-5)
        mlx_nonleading = nabla.tensor_vmap_mlx_fn(
            lambda x: x.sum(), [("x", [2])], 3, in_axes=[1]
        )
        assert_close_rows(
            [mlx_nonleading(nonleading_values).to_flat_list()], nonleading_expected, tol=1e-5
        )

    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda_compiled = nabla.tensor_vmap_cuda_fn(
            lambda x, weight: x.matmul(weight).tanh(), input_specs, 3
        )
        assert cuda_compiled.backend in {"cublas", "nvrtc"}
        assert_close_rows([cuda_compiled(values).to_flat_list()], expected, tol=1e-5)
        cuda_nonleading = nabla.tensor_vmap_cuda_fn(
            lambda x: x.sum(), [("x", [2])], 3, in_axes=[1]
        )
        assert_close_rows(
            [cuda_nonleading(nonleading_values).to_flat_list()], nonleading_expected, tol=1e-5
        )


def test_tensor_vmap_vjp_matches_per_example_loop_and_aggregates_unmapped_gradient():
    vjp = nabla.tensor_vmap_vjp_fn(
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
            "x": nabla.Tensor([2, 3], x_values),
            "weight": nabla.Tensor([2], weight_values),
        },
        nabla.Tensor([3], cotangent_values),
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


def test_tensor_vmap_jvp_matches_per_example_loop_with_nonleading_axes():
    jvp = nabla.tensor_vmap_jvp_fn(
        lambda x: x.tanh(),
        [("x", [2])],
        3,
        in_axes=[1],
        out_axis=1,
    )
    values = [0.0, 0.5, -1.0, 1.0, -0.5, 2.0]
    tangents = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0]
    output, output_tangent = jvp(
        {"x": nabla.Tensor([2, 3], values)},
        {"x": nabla.Tensor([2, 3], tangents)},
    )

    assert jvp.node_count > 0
    assert_close_rows([output.to_flat_list()], [[math.tanh(value) for value in values]])
    assert_close_rows(
        [output_tangent.to_flat_list()],
        [[direction * (1.0 - math.tanh(value) ** 2) for value, direction in zip(values, tangents)]],
    )


def test_tensor_vmap_cuda_vjp_matches_cpu_single_batched_plan():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    function = lambda x, weight: (x * weight).tanh().sum()
    input_specs = [("x", [2]), ("weight", [2])]
    values = {
        "x": nabla.Tensor([2, 3], [0.0, 0.5, -1.0, 1.0, -0.5, 2.0]),
        "weight": nabla.Tensor([2], [2.0, -1.0]),
    }
    cotangent = nabla.Tensor([3], [1.0, 2.0, -0.5])
    cpu = nabla.tensor_vmap_vjp_fn(function, input_specs, 3, in_axes=[1, None])
    cuda = nabla.tensor_vmap_vjp_cuda_fn(function, input_specs, 3, in_axes=[1, None])

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
    function = lambda x, weight: (x.matmul(weight).tanh()).sum()
    input_specs = [("x", [2]), ("weight", [2, 1])]
    values = {
        "x": nabla.Tensor([3, 2], [1.0, 2.0, -1.0, 0.5, 0.25, -2.0]),
        "weight": nabla.Tensor([2, 1], [0.75, -0.5]),
    }
    cotangent = nabla.Tensor([3], [1.0, -0.25, 2.0])
    cpu = nabla.tensor_vmap_vjp_fn(function, input_specs, 3, in_axes=[0, None])
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

    if os.environ.get("NABLA_CUDA_TEST") is not None:
        cuda = nabla.tensor_vmap_vjp_cuda_fn(function, input_specs, 3, in_axes=[0, None])
        cuda_value, cuda_gradients = cuda(values, cotangent)
        assert_close_rows([cuda_value.to_flat_list()], [expected_value], tol=2e-5)
        assert_close_rows([cuda_gradients["x"].to_flat_list()], [expected_x], tol=2e-5)
        assert_close_rows(
            [cuda_gradients["weight"].to_flat_list()], [expected_weight], tol=2e-5
        )


def test_tensor_jit_cuda_fn_reuses_a_callable_cuda_plan():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    compiled = nabla.tensor_jit_cuda_fn(
        lambda x, weight, bias: (x @ weight + bias).tanh(),
        [("x", [2, 2]), ("weight", [2, 2]), ("bias", [1, 2])],
    )
    first = compiled(
        {
            "x": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
            "weight": nabla.Tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
            "bias": nabla.Tensor([1, 2], [0.0, 1.0]),
        }
    )
    second = compiled(
        {
            "x": nabla.Tensor([2, 2], [0.0, 1.0, -1.0, 0.5]),
            "weight": nabla.Tensor([2, 2], [2.0, 0.0, 0.0, 2.0]),
            "bias": nabla.Tensor([1, 2], [1.0, -1.0]),
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
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    value_and_grad = nabla.tensor_value_and_grad_cuda_fn(
        lambda x, target, weight, bias: ((x * weight + bias - target).powi(2)).mean(),
        [("x", [2]), ("target", [2]), ("weight", [1]), ("bias", [1])],
        ["weight", "bias"],
    )
    inputs = {
        "x": nabla.Tensor([2], [-1.0, 1.0]),
        "target": nabla.Tensor([2], [-1.0, 3.0]),
        "weight": nabla.Tensor([1], [0.0]),
        "bias": nabla.Tensor([1], [0.0]),
    }

    value, gradients = value_and_grad(inputs)
    cpu_value_and_grad = nabla.tensor_value_and_grad_fn(
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


def test_cuda_adam_loss_optimizer_owns_scalar_loss_and_parameters():
    if os.environ.get("NABLA_CUDA_TEST") is None:
        return

    traced = nabla.trace_tensor(
        lambda x, weight, bias: (x * weight) + bias,
        [("x", [2]), ("weight", [1]), ("bias", [1])],
    )
    target = traced.graph.input("target", [2])
    loss = (traced.output - target).powi(2).mean()
    optimizer = nabla.cuda_adam_loss_optimizer(
        loss,
        ["weight", "bias"],
        {
            "x": nabla.Tensor([2], [-1.0, 1.0]),
            "target": nabla.Tensor([2], [-1.0, 3.0]),
            "weight": nabla.Tensor([1], [0.0]),
            "bias": nabla.Tensor([1], [0.0]),
        },
        0.05,
        ["x", "target"],
    )

    initial_loss = optimizer.loss().to_flat_list()[0]
    initial_buffers = optimizer.device_buffer_count
    for _ in range(250):
        optimizer.step()
    final_loss = optimizer.loss().to_flat_list()[0]
    trained = optimizer.parameters()

    assert final_loss < initial_loss * 1e-4
    assert abs(trained["weight"].to_flat_list()[0] - 2.0) < 2e-3
    assert abs(trained["bias"].to_flat_list()[0] - 1.0) < 2e-3
    assert optimizer.device_buffer_count == initial_buffers


def test_tensor_vjp_fn_reuses_a_compiled_plan_with_runtime_cotangent():
    vjp = nabla.tensor_vjp_fn(lambda x: x.tanh(), [("x", [2, 2])])
    values = [0.0, 1.0, -1.0, 0.5]
    cotangent = [1.0, 2.0, 3.0, 4.0]

    output, gradients = vjp(
        {"x": nabla.Tensor([2, 2], values)},
        nabla.Tensor([2, 2], cotangent),
    )

    assert_close_rows([output.to_flat_list()], [[math.tanh(value) for value in values]])
    assert_close_rows(
        [gradients["x"].to_flat_list()],
        [[cot * (1.0 - math.tanh(value) ** 2) for value, cot in zip(values, cotangent)]],
    )


def test_tensor_jvp_fn_reuses_a_compiled_plan_with_runtime_tangent():
    jvp = nabla.tensor_jvp_fn(lambda x: x.tanh(), [("x", [2, 2])])
    values = [0.0, 1.0, -1.0, 0.5]
    tangent = [1.0, 2.0, 3.0, 4.0]

    output, output_tangent = jvp(
        {"x": nabla.Tensor([2, 2], values)},
        {"x": nabla.Tensor([2, 2], tangent)},
    )

    assert_close_rows([output.to_flat_list()], [[math.tanh(value) for value in values]])
    assert_close_rows(
        [output_tangent.to_flat_list()],
        [[direction * (1.0 - math.tanh(value) ** 2) for value, direction in zip(values, tangent)]],
    )


def test_tensor_jacobian_fn_reuses_a_compiled_plan():
    jacobian = nabla.tensor_jacobian_fn(
        lambda x: x.powi(2), [("x", [2, 2])], "x"
    )
    result = jacobian({"x": nabla.Tensor([2, 2], [1.0, 2.0, -3.0, 0.5])})

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
    optimizer = nabla.Adam(learning_rate=0.1)
    parameters = {"weight": nabla.Tensor([2], [1.0, 2.0])}
    gradients = {"weight": nabla.Tensor([2], [0.5, -0.5])}

    first = optimizer.step(parameters, gradients)
    second = optimizer.step(first, gradients)

    assert_close_rows([first["weight"].to_flat_list()], [[0.9, 2.1]], tol=1e-7)
    assert_close_rows([second["weight"].to_flat_list()], [[0.8, 2.2]], tol=1e-7)


def test_adam_invalid_step_does_not_advance_optimizer_state():
    parameters = {"weight": nabla.Tensor([1], [1.0])}
    gradients = {"weight": nabla.Tensor([1], [0.5])}
    reference = nabla.Adam(learning_rate=0.1)
    expected = reference.step(reference.step(parameters, gradients), gradients)

    optimizer = nabla.Adam(learning_rate=0.1)
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
    combined = nabla.sum_gradients(
        [
            {"weight": nabla.Tensor([2], [1.0, -2.0])},
            {"weight": nabla.Tensor([2], [0.5, 3.0])},
        ]
    )

    assert combined["weight"].to_flat_list() == [1.5, 1.0]
    filtered = nabla.sum_gradients(
        [
            {"weight": nabla.Tensor([1], [1.0]), "forcing": nabla.Tensor([1], [5.0])},
            {"weight": nabla.Tensor([1], [2.0]), "target": nabla.Tensor([1], [7.0])},
        ],
        ["weight"],
    )
    assert filtered["weight"].to_flat_list() == [3.0]


def test_poisson_residual_training_converges_with_symbolic_jvp_and_adam():
    coordinate = 0.5
    target_weight = 1.0
    target_value = math.tanh(target_weight * coordinate)
    forcing = 2.0 * target_weight**2 * target_value * (1.0 - target_value**2)
    traced = nabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [1]), ("weight", [1]), ("forcing", [1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    loss = (residual * residual).sum()
    parameters = {"weight": nabla.Tensor([1], [0.3])}
    optimizer = nabla.Adam(learning_rate=0.03)

    def value_and_grad(current):
        return second_derivative.graph.evaluate_value_and_vjp(
            loss.node_id,
            {
                "x": nabla.Tensor([1], [coordinate]),
                "forcing": nabla.Tensor([1], [forcing]),
                **current,
            },
            nabla.Tensor([], [1.0]),
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
    traced = nabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [4, 1]), ("weight", [1, 1]), ("forcing", [4, 1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    loss = (residual * residual).mean()
    parameters = {"weight": nabla.Tensor([1, 1], [0.3])}
    optimizer = nabla.Adam(learning_rate=0.03)

    def value_and_grad(current):
        return second_derivative.graph.evaluate_value_and_vjp(
            loss.node_id,
            {
                "x": nabla.Tensor([4, 1], coordinates),
                "forcing": nabla.Tensor([4, 1], forcing_values),
                **current,
            },
            nabla.Tensor([], [1.0]),
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
    residual_trace = nabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).tanh(),
        [("x", [4, 1]), ("weight", [1, 1]), ("forcing", [4, 1])],
    )
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    residual_plan = second_derivative.graph.compile_cpu(residual_loss.node_id)
    boundary_trace = nabla.trace_tensor(
        lambda x, weight, target: (x * weight).tanh(),
        [("x", [4, 1]), ("weight", [1, 1]), ("target", [4, 1])],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    boundary_plan = boundary_trace.graph.compile_cpu(boundary_loss.node_id)
    parameters = {"weight": nabla.Tensor([1, 1], [0.3])}
    optimizer = nabla.Adam(learning_rate=0.03)

    def loss_and_gradient(current):
        residual_value, residual_gradients = residual_plan.evaluate_vjp(
            {"x": nabla.Tensor([4, 1], coordinates), "forcing": nabla.Tensor([4, 1], forcing_values), **current},
            nabla.Tensor([], [1.0]),
        )
        boundary_value, boundary_gradients = boundary_plan.evaluate_vjp(
            {"x": nabla.Tensor([4, 1], [1.0] * 4), "target": nabla.Tensor([4, 1], [math.tanh(1.0)] * 4), **current},
            nabla.Tensor([], [1.0]),
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
    traced = nabla.trace_tensor(
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
            "x": nabla.Tensor([4, 1], [0.2, 0.4, 0.6, 0.8]),
            "w1": nabla.Tensor([1, 2], [0.7, -0.4]),
            "b1": nabla.Tensor([1, 2], [0.1, -0.2]),
            "w2": nabla.Tensor([2, 1], [0.5, -0.3]),
            "b2": nabla.Tensor([1, 1], [0.05]),
            "forcing": nabla.Tensor([4, 1], [0.2, -0.1, 0.3, -0.2]),
        },
        nabla.Tensor([], [1.0]),
    )

    for name in ["w1", "b1", "w2"]:
        assert any(abs(value) > 1e-8 for value in gradients[name].to_flat_list())


def test_two_layer_mlp_poisson_training_converges():
    coordinates = [0.2, 0.4, 0.6, 0.8]
    parameter_specs = [
        ("x", [4, 1]), ("w1", [1, 2]), ("b1", [1, 2]),
        ("w2", [2, 1]), ("b2", [1, 1]), ("forcing", [4, 1]),
    ]
    model = lambda x, w1, b1, w2, b2, forcing: (x.matmul(w1) + b1).tanh().matmul(w2) + b2
    residual_trace = nabla.trace_tensor(model, parameter_specs)
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    boundary_trace = nabla.trace_tensor(
        lambda x, w1, b1, w2, b2, target: (x.matmul(w1) + b1).tanh().matmul(w2) + b2,
        [
            ("x", [4, 1]), ("w1", [1, 2]), ("b1", [1, 2]),
            ("w2", [2, 1]), ("b2", [1, 1]), ("target", [4, 1]),
        ],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    teacher = {
        "w1": nabla.Tensor([1, 2], [1.2, -0.7]), "b1": nabla.Tensor([1, 2], [0.1, -0.2]),
        "w2": nabla.Tensor([2, 1], [0.8, 0.5]), "b2": nabla.Tensor([1, 1], [0.05]),
    }
    forcing = second_derivative.graph.evaluate(
        second_derivative.output.node_id,
        {"x": nabla.Tensor([4, 1], coordinates), "forcing": nabla.Tensor([4, 1], [0.0] * 4), **teacher},
    )
    boundary_coordinates = [0.0, 0.0, 1.0, 1.0]
    boundary_target = boundary_trace.graph.evaluate(
        boundary_trace.output.node_id,
        {"x": nabla.Tensor([4, 1], boundary_coordinates), "target": nabla.Tensor([4, 1], [0.0] * 4), **teacher},
    )
    parameters = {
        "w1": nabla.Tensor([1, 2], [0.3, -0.1]), "b1": nabla.Tensor([1, 2], [0.0, 0.0]),
        "w2": nabla.Tensor([2, 1], [0.2, 0.1]), "b2": nabla.Tensor([1, 1], [0.0]),
    }
    optimizer = nabla.Adam(learning_rate=0.02)

    def loss_and_grad(current):
        residual_value, residual_gradients = second_derivative.graph.evaluate_value_and_vjp(
            residual_loss.node_id,
            {"x": nabla.Tensor([4, 1], coordinates), "forcing": nabla.Tensor([4, 1], [-value for value in forcing.to_flat_list()]), **current},
            nabla.Tensor([], [1.0]),
        )
        boundary_value, boundary_gradients = boundary_trace.graph.evaluate_value_and_vjp(
            boundary_loss.node_id,
            {"x": nabla.Tensor([4, 1], boundary_coordinates), "target": boundary_target, **current},
            nabla.Tensor([], [1.0]),
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
    residual_trace = nabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).sin(),
        [("x", [5, 1]), ("weight", [1, 1]), ("forcing", [5, 1])],
    )
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    boundary_trace = nabla.trace_tensor(
        lambda x, weight, target: (x * weight).sin(),
        [("x", [2, 1]), ("weight", [1, 1]), ("target", [2, 1])],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    parameters = {"weight": nabla.Tensor([1, 1], [2.5])}
    optimizer = nabla.Adam(learning_rate=0.01)

    def loss_and_grad(current):
        residual_value, residual_gradients = second_derivative.graph.evaluate_value_and_vjp(
            residual_loss.node_id,
            {"x": nabla.Tensor([5, 1], coordinates), "forcing": nabla.Tensor([5, 1], forcing), **current},
            nabla.Tensor([], [1.0]),
        )
        boundary_value, boundary_gradients = boundary_trace.graph.evaluate_value_and_vjp(
            boundary_loss.node_id,
            {"x": nabla.Tensor([2, 1], [0.0, 1.0]), "target": nabla.Tensor([2, 1], [0.0, 0.0]), **current},
            nabla.Tensor([], [1.0]),
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

    traced = nabla.trace_tensor(model, [("x", [2, 2])])
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
        {"id": 0, "op": "input", "shape": [2, 2], "inputs": [], "name": "x"},
        {"id": 1, "op": "add", "shape": [2, 2], "inputs": [0, 0]},
    ]
    plan.validate_kernel_ir()
    inputs = {"x": nabla.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])}
    assert plan.evaluate(inputs).to_flat_list() == [2.0, 4.0, 6.0, 8.0]
    _, gradients = plan.evaluate_vjp(inputs, nabla.Tensor([2, 2], [1.0] * 4))
    assert gradients["x"].to_flat_list() == [2.0] * 4
    _, alias_gradients = plan.evaluate_value_and_vjp(
        inputs, nabla.Tensor([2, 2], [1.0] * 4)
    )
    assert alias_gradients["x"].to_flat_list() == [2.0] * 4
    _, tangent = plan.evaluate_jvp(inputs, {"x": nabla.Tensor([2, 2], [1.0] * 4)})
    assert tangent.to_flat_list() == [2.0] * 4


def test_trace_tensor_compile_cpu_commons_identical_pure_nodes():
    def model(x):
        return x.tanh() + x.tanh()

    traced = nabla.trace_tensor(model, [("x", [2, 2])])
    plan = traced.graph.compile_cpu(traced.output.node_id)

    assert plan.node_count == 3
    assert plan.kernel_ir()[2]["inputs"] == [1, 1]
    assert_close_rows(
        [plan.evaluate({"x": nabla.Tensor([2, 2], [0.0, 1.0, -1.0, 0.5])}).to_flat_list()],
        [[0.0, 2.0 * math.tanh(1.0), -2.0 * math.tanh(1.0), 2.0 * math.tanh(0.5)]],
    )


def test_matrix_neg_and_scalar_div():
    a = nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])

    negated = -a
    scaled = a / 2.0

    assert negated.shape == (2, 2)
    assert negated.to_list() == [[-1.0, 2.0], [-3.0, 4.0]]
    assert scaled.shape == (2, 2)
    assert scaled.to_list() == [[0.5, -1.0], [1.5, -2.0]]


def test_matrix_elementwise_division():
    a = nabla.Matrix([[2.0, 6.0], [12.0, 20.0]])
    b = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])
    scalar = nabla.Matrix([[2.0]])

    divided = a / b
    scaled = a / scalar

    assert divided.shape == (2, 2)
    assert divided.to_list() == [[2.0, 3.0], [4.0, 5.0]]
    assert scaled.shape == (2, 2)
    assert scaled.to_list() == [[1.0, 3.0], [6.0, 10.0]]


def test_matrix_integer_power():
    a = nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])

    squared = a**2

    assert squared.shape == (2, 2)
    assert squared.to_list() == [[1.0, 4.0], [9.0, 16.0]]


def test_matrix_float_power():
    a = nabla.Matrix([[1.0, 4.0], [9.0, 16.0]])

    rooted = a**0.5

    assert rooted.shape == (2, 2)
    assert rooted.to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_sum():
    a = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])

    total = a.sum()

    assert total.shape == (1, 1)
    assert total.to_list() == [[10.0]]


def test_matrix_sum_axis():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    column_totals = a.sum(axis=0)
    row_totals = a.sum(axis=1)

    assert column_totals.shape == (1, 3)
    assert column_totals.to_list() == [[5.0, 7.0, 9.0]]
    assert row_totals.shape == (2, 1)
    assert row_totals.to_list() == [[6.0], [15.0]]


def test_matrix_mean():
    a = nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])

    average = a.mean()

    assert average.shape == (1, 1)
    assert average.to_list() == [[2.5]]


def test_matrix_mean_axis():
    a = nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    column_means = a.mean(axis=0)
    row_means = a.mean(axis=1)

    assert column_means.shape == (1, 3)
    assert column_means.to_list() == [[2.5, 3.5, 4.5]]
    assert row_means.shape == (2, 1)
    assert row_means.to_list() == [[2.0], [5.0]]


def test_matrix_tanh():
    a = nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.tanh()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.tanh(0.0), math.tanh(1.0)], [math.tanh(-1.0), math.tanh(2.0)]],
    )


def test_matrix_exp():
    a = nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.exp()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.exp(0.0), math.exp(1.0)], [math.exp(-1.0), math.exp(2.0)]],
    )


def test_matrix_log():
    a = nabla.Matrix([[1.0, 2.0], [4.0, 8.0]])

    output = a.log()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.log(1.0), math.log(2.0)], [math.log(4.0), math.log(8.0)]],
    )


def test_matrix_sqrt():
    a = nabla.Matrix([[1.0, 4.0], [9.0, 16.0]])

    output = a.sqrt()

    assert output.shape == (2, 2)
    assert output.to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_sin():
    a = nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.sin()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.sin(0.0), math.sin(1.0)], [math.sin(-1.0), math.sin(2.0)]],
    )


def test_matrix_cos():
    a = nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])

    output = a.cos()

    assert output.shape == (2, 2)
    assert_close_rows(
        output.to_list(),
        [[math.cos(0.0), math.cos(1.0)], [math.cos(-1.0), math.cos(2.0)]],
    )


def test_matrix_gt_and_where_support_masks():
    a = nabla.Matrix([[-1.0, 0.5, 2.0], [3.0, -4.0, 5.0]])
    positive = a.gt(0.0)
    selected = nabla.where(
        positive,
        a,
        nabla.Matrix([[0.0]]),
    )

    assert positive.shape == (2, 3)
    assert positive.to_list() == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    assert selected.to_list() == [[0.0, 0.5, 2.0], [3.0, 0.0, 5.0]]


def test_matrix_rejects_incompatible_shapes():
    a = nabla.Matrix([[1.0, 2.0]])
    b = nabla.Matrix([[3.0, 4.0]])

    try:
        a @ b
    except ValueError as err:
        assert "incompatible" in str(err)
    else:
        raise AssertionError("expected incompatible shapes to raise ValueError")


def test_trace_graph_records_matmul():
    graph = nabla.TraceGraph()
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

    traced = nabla.trace(model, [("a", (2, 3)), ("b", (3, 2))])

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
            "a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": nabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        },
        nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )
    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]


def test_trace_graph_ir_includes_backend_lowering_attrs():
    def model(a, b):
        powered = a**2.5
        reduced = powered.mean(axis=0)
        return nabla.concat([reduced, b], axis=1)

    traced = nabla.trace(model, [("a", (2, 2)), ("b", (1, 1))])
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
        return nabla.concat([reduced, b], axis=1)

    traced = nabla.trace(model, [("a", (2, 2)), ("b", (1, 1))])

    assert traced.graph.lower_text() == "\n".join(
        [
            "%0 = input[name=a] : tensor<2x2xf64>",
            "%1 = input[name=b] : tensor<1x1xf64>",
            "%2 = powf(%0) {exponent=2.5} : tensor<2x2xf64>",
            "%3 = mean(%2) {axis=0} : tensor<1x2xf64>",
            "%4 = concat(%3, %1) {axis=1} : tensor<1x3xf64>",
        ]
    )


def test_trace_graph_evaluates_direct_jvp():
    def model(a, b):
        return (a @ b) + (a * b.T)

    traced = nabla.trace(model, [("a", (2, 2)), ("b", (2, 2))])
    primal, tangent = traced.graph.evaluate_jvp(
        traced.output.node_id,
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        },
        {
            "a": nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
            "b": nabla.Matrix([[0.0, 1.0], [2.0, -1.0]]),
        },
    )

    assert primal.to_list() == [[24.0, 36.0], [61.0, 82.0]]
    assert tangent.to_list() == [[17.5, 16.5], [14.0, 21.0]]


def test_trace_graph_compiles_an_immutable_cpu_execution_plan():
    def model(a, b):
        return (a @ b).tanh()

    traced = nabla.trace(model, [("a", (2, 2)), ("b", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    values = {
        "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        "b": nabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
    }

    assert plan.output_node_id == traced.output.node_id
    assert plan.output_shape == (2, 2)
    assert plan.evaluate(values).to_list() == traced.graph.evaluate(
        traced.output.node_id, values
    ).to_list()


def test_cpu_execution_plan_evaluates_vjp():
    def model(a, b):
        return a @ b

    traced = nabla.trace(model, [("a", (2, 2)), ("b", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    values = {
        "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        "b": nabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
    }
    cotangent = nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]])

    plan_gradients = plan.evaluate_vjp(values, cotangent)
    graph_gradients = traced.graph.evaluate_vjp(traced.output.node_id, values, cotangent)

    assert set(plan_gradients) == {"a", "b"}
    assert plan_gradients["a"].to_list() == graph_gradients["a"].to_list()
    assert plan_gradients["b"].to_list() == graph_gradients["b"].to_list()


def test_cpu_execution_plan_eliminates_unreachable_trace_nodes():
    def model(a):
        unused = a.exp()
        return a + 1.0

    traced = nabla.trace(model, [("a", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)

    assert len(traced.graph.ir()) == 4
    assert plan.output_node_id == traced.output.node_id
    assert plan.node_count == 3
    assert plan.evaluate(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    ).to_list() == [[2.0, 3.0], [4.0, 5.0]]


def test_cpu_execution_plan_eliminates_repeated_unary_subexpressions():
    def model(a):
        return a.exp() + a.exp()

    traced = nabla.trace(model, [("a", (2, 2))])
    plan = traced.graph.compile_cpu(traced.output.node_id)
    values = {"a": nabla.Matrix([[0.0, 1.0], [2.0, 3.0]])}

    assert len(traced.graph.ir()) == 4
    assert plan.node_count == 3
    assert plan.evaluate(values).to_list() == traced.graph.evaluate(
        traced.output.node_id, values
    ).to_list()


def test_grad_traces_and_evaluates_vjp():
    def model(a, b):
        return a @ b

    gradients = nabla.grad(
        model,
        [("a", (2, 3)), ("b", (3, 2))],
        {
            "a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": nabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        },
        nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )

    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]


def test_grad_fn_reuses_transform_callable():
    def model(a, b):
        return a @ b

    grad_model = nabla.grad_fn(
        model,
        [("a", (2, 3)), ("b", (3, 2))],
        nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )
    gradients = grad_model(
        {
            "a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": nabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        }
    )

    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]


def test_grad_fn_evaluates_add_vjp():
    def model(a, b):
        return a + b

    cotangent = nabla.Matrix([[0.5, 1.5], [2.5, 3.5]])
    grad_model = nabla.grad_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
        cotangent,
    )
    gradients = grad_model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        }
    )

    assert gradients["a"].to_list() == cotangent.to_list()
    assert gradients["b"].to_list() == cotangent.to_list()


def test_grad_fn_evaluates_mul_vjp():
    def model(a, b):
        return a * b

    grad_model = nabla.grad_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
        nabla.Matrix([[2.0, 3.0], [4.0, 5.0]]),
    )
    gradients = grad_model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[0.5, 1.5], [2.5, 3.5]]),
        }
    )

    assert gradients["a"].to_list() == [[1.0, 4.5], [10.0, 17.5]]
    assert gradients["b"].to_list() == [[2.0, 6.0], [12.0, 20.0]]


def test_grad_fn_evaluates_square_sum_vjp():
    def model(a):
        return (a * a).sum()

    grad_model = nabla.grad_fn(
        model,
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert gradients["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]


def test_grad_fn_evaluates_square_mean_vjp():
    grad_model = nabla.grad_fn(
        lambda a: (a * a).mean(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert gradients["a"].to_list() == [[0.5, 1.0], [1.5, 2.0]]


def test_grad_fn_evaluates_squared_error_vjp():
    def model(pred, target):
        error = pred - target
        return (error * error).sum()

    grad_model = nabla.grad_fn(
        model,
        [("pred", (2, 2)), ("target", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_model(
        {
            "pred": nabla.Matrix([[1.0, 3.0], [2.0, 5.0]]),
            "target": nabla.Matrix([[0.5, 1.0], [3.0, 1.5]]),
        }
    )

    assert gradients["pred"].to_list() == [[1.0, 4.0], [-2.0, 7.0]]
    assert gradients["target"].to_list() == [[-1.0, -4.0], [2.0, -7.0]]


def test_grad_scalar_fn_seeds_scalar_cotangent():
    def model(pred, target):
        error = pred - target
        return (error * error).sum()

    grad_model = nabla.grad_scalar_fn(
        model,
        [("pred", (2, 2)), ("target", (2, 2))],
    )
    gradients = grad_model(
        {
            "pred": nabla.Matrix([[1.0, 3.0], [2.0, 5.0]]),
            "target": nabla.Matrix([[0.5, 1.0], [3.0, 1.5]]),
        }
    )

    assert gradients["pred"].to_list() == [[1.0, 4.0], [-2.0, 7.0]]
    assert gradients["target"].to_list() == [[-1.0, -4.0], [2.0, -7.0]]


def test_grad_scalar_decorator_factory():
    @nabla.grad_scalar([("pred", (2, 2)), ("target", (2, 2))])
    def model(pred, target):
        error = pred - target
        return (error * error).sum()

    gradients = model(
        {
            "pred": nabla.Matrix([[1.0, 3.0], [2.0, 5.0]]),
            "target": nabla.Matrix([[0.5, 1.0], [3.0, 1.5]]),
        }
    )

    assert gradients["pred"].to_list() == [[1.0, 4.0], [-2.0, 7.0]]
    assert gradients["target"].to_list() == [[-1.0, -4.0], [2.0, -7.0]]


def test_grad_fn_traces_once_at_transform_creation():
    calls = {"count": 0}

    def model(a, b):
        calls["count"] += 1
        return a + b

    grad_model = nabla.grad_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
        nabla.Matrix([[1.0, 1.0], [1.0, 1.0]]),
    )

    assert calls["count"] == 1

    first = grad_model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        }
    )
    second = grad_model(
        {
            "a": nabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
            "b": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
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

    grad_model = nabla.grad_scalar_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    first = grad_model({"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})
    second = grad_model({"a": nabla.Matrix([[2.0, 4.0], [6.0, 8.0]])})

    assert first["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert second["a"].to_list() == [[4.0, 8.0], [12.0, 16.0]]
    assert calls["count"] == 1


def test_grad_scalar_decorator_traces_once_at_decoration_time():
    calls = {"count": 0}

    @nabla.grad_scalar([("a", (2, 2))])
    def model(a):
        calls["count"] += 1
        return (a * a).sum()

    assert calls["count"] == 1

    first = model({"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})
    second = model({"a": nabla.Matrix([[2.0, 4.0], [6.0, 8.0]])})

    assert first["a"].to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert second["a"].to_list() == [[4.0, 8.0], [12.0, 16.0]]
    assert calls["count"] == 1


def test_value_and_grad_fn_returns_value_and_traces_once():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a * a).sum()

    value_and_grad = nabla.value_and_grad_fn(
        model,
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    assert calls["count"] == 1

    first_value, first_gradients = value_and_grad(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )
    second_value, second_gradients = value_and_grad(
        {"a": nabla.Matrix([[2.0, 4.0], [6.0, 8.0]])}
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

    vjp_model = nabla.vjp_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    primal, gradients = vjp_model(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])},
        nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )

    assert primal.to_list() == [[4.0, 10.0], [18.0, 28.0]]
    assert gradients["a"].to_list() == [[5.0, 3.5], [-9.0, 22.0]]
    assert calls["count"] == 1


def test_jacobian_fn_traces_once_and_returns_dense_jacobian():
    calls = {"count": 0}

    def model(a):
        calls["count"] += 1
        return (a**2) + (a * 3.0)

    jacobian_model = nabla.jacobian_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    jacobian = jacobian_model(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
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

    jacobians_model = nabla.jacobians_fn(
        model,
        [("a", (2, 2)), ("b", (2, 2))],
    )

    assert calls["count"] == 1

    jacobians = jacobians_model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
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

    jvp_model = nabla.jvp_fn(model, [("a", (2, 2))])

    assert calls["count"] == 1

    primal, tangent = jvp_model(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])},
        {"a": nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]])},
    )

    assert primal.to_list() == [[4.0, 10.0], [18.0, 28.0]]
    assert tangent.to_list() == [[5.0, 3.5], [-9.0, 22.0]]
    assert calls["count"] == 1


def test_jvp_fn_supports_multiple_input_tangents():
    calls = {"count": 0}

    def model(a, b):
        calls["count"] += 1
        return a * b

    jvp_model = nabla.jvp_fn(model, [("a", (2, 2)), ("b", (2, 2))])

    assert calls["count"] == 1

    primal, tangent = jvp_model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
        },
        {
            "a": nabla.Matrix([[0.5, 1.0], [1.5, 2.0]]),
            "b": nabla.Matrix([[2.0, 3.0], [4.0, 5.0]]),
        },
    )

    assert primal.to_list() == [[10.0, 40.0], [90.0, 160.0]]
    assert tangent.to_list() == [[7.0, 26.0], [57.0, 100.0]]
    assert calls["count"] == 1


def test_jit_and_vjp_support_transpose():
    @nabla.jit([("a", (2, 3))])
    def transpose_primal(a):
        return a.T

    output = transpose_primal(
        {"a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    )

    assert output.shape == (3, 2)
    assert output.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    transpose_vjp = nabla.vjp_fn(lambda a: a.transpose(), [("a", (2, 3))])
    primal, gradients = transpose_vjp(
        {"a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        nabla.Matrix([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    )

    assert primal.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert gradients["a"].shape == (2, 3)
    assert gradients["a"].to_list() == [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]


def test_jit_and_vjp_support_scalar_matrix_broadcast():
    @nabla.jit([("a", (2, 2)), ("s", (1, 1))])
    def shift(a, s):
        return a + s

    output = shift(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "s": nabla.Matrix([[10.0]]),
        }
    )

    assert output.to_list() == [[11.0, 12.0], [13.0, 14.0]]

    scaled_sum_grad = nabla.grad_fn(
        lambda a, s: (a * s).sum(),
        [("a", (2, 2)), ("s", (1, 1))],
        nabla.Matrix([[1.0]]),
    )
    gradients = scaled_sum_grad(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "s": nabla.Matrix([[3.0]]),
        }
    )

    assert gradients["a"].to_list() == [[3.0, 3.0], [3.0, 3.0]]
    assert gradients["s"].shape == (1, 1)
    assert gradients["s"].to_list() == [[10.0]]


def test_jit_and_vjp_support_reshape():
    @nabla.jit([("a", (2, 3))])
    def reshape_primal(a):
        return a.reshape(3, 2)

    output = reshape_primal(
        {"a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    )

    assert output.shape == (3, 2)
    assert output.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    reshape_vjp = nabla.vjp_fn(lambda a: a.reshape(3, 2), [("a", (2, 3))])
    primal, gradients = reshape_vjp(
        {"a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        nabla.Matrix([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    )

    assert primal.to_list() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert gradients["a"].shape == (2, 3)
    assert gradients["a"].to_list() == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]


def test_jit_and_grad_support_scalar_literals():
    @nabla.jit([("a", (2, 2))])
    def affine(a):
        return (a * 2.0) + 1.0

    output = affine({"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})

    assert output.to_list() == [[3.0, 5.0], [7.0, 9.0]]

    grad_affine_sum = nabla.grad_fn(
        lambda a: ((a * 2.0) + 1.0).sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_affine_sum(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert gradients["a"].to_list() == [[2.0, 2.0], [2.0, 2.0]]


def test_jit_and_grad_support_neg_and_scalar_division():
    @nabla.jit([("a", (2, 2))])
    def normalized(a):
        return -a / 2.0

    output = normalized({"a": nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])})

    assert output.to_list() == [[-0.5, 1.0], [-1.5, 2.0]]

    grad_normalized_sum = nabla.grad_fn(
        lambda a: (-a / 2.0).sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_normalized_sum(
        {"a": nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])}
    )

    assert gradients["a"].to_list() == [[-0.5, -0.5], [-0.5, -0.5]]


def test_jit_and_grad_support_elementwise_division():
    @nabla.jit([("a", (2, 2)), ("b", (2, 2))])
    def ratio(a, b):
        return a / b

    output = ratio(
        {
            "a": nabla.Matrix([[2.0, 6.0], [12.0, 20.0]]),
            "b": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert output.to_list() == [[2.0, 3.0], [4.0, 5.0]]

    grad_ratio_sum = nabla.grad_fn(
        lambda a, b: (a / b).sum(),
        [("a", (2, 2)), ("b", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_ratio_sum(
        {
            "a": nabla.Matrix([[2.0, 6.0], [12.0, 20.0]]),
            "b": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
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
    @nabla.jit([("a", (2, 3)), ("row", (1, 3)), ("column", (2, 1))])
    def broadcasted(a, row, column):
        return (a + row) * column

    values = {
        "a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "row": nabla.Matrix([[10.0, 20.0, 30.0]]),
        "column": nabla.Matrix([[2.0], [3.0]]),
    }

    output = broadcasted(values)

    assert output.to_list() == [[22.0, 44.0, 66.0], [42.0, 75.0, 108.0]]

    grad_broadcasted_sum = nabla.grad_fn(
        lambda a, row, column: (((a + row) * column).sum()),
        [("a", (2, 3)), ("row", (1, 3)), ("column", (2, 1))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_broadcasted_sum(values)

    assert gradients["a"].to_list() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert gradients["row"].to_list() == [[5.0, 5.0, 5.0]]
    assert gradients["column"].to_list() == [[66.0], [75.0]]


def test_jit_and_vjp_support_where_masks():
    @nabla.jit([("a", (2, 3))])
    def relu_like(a):
        return nabla.where(a.gt(0.0), a, a * 0.1)

    values = {
        "a": nabla.Matrix([[-1.0, 0.5, 2.0], [3.0, -4.0, 5.0]]),
    }

    output = relu_like(values)

    assert output.to_list() == [[-0.1, 0.5, 2.0], [3.0, -0.4, 5.0]]

    grad_relu_like_sum = nabla.grad_fn(
        lambda a: nabla.where(a.gt(0.0), a, a * 0.1).sum(),
        [("a", (2, 3))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_relu_like_sum(values)

    assert gradients["a"].to_list() == [[0.1, 1.0, 1.0], [1.0, 0.1, 1.0]]


def test_jit_and_grad_support_integer_power():
    @nabla.jit([("a", (2, 2))])
    def square(a):
        return a**2

    output = square({"a": nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])})

    assert output.to_list() == [[1.0, 4.0], [9.0, 16.0]]

    grad_square_sum = nabla.grad_fn(
        lambda a: (a**2).sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_square_sum(
        {"a": nabla.Matrix([[1.0, -2.0], [3.0, -4.0]])}
    )

    assert gradients["a"].to_list() == [[2.0, -4.0], [6.0, -8.0]]


def test_jit_and_grad_support_float_power():
    @nabla.jit([("a", (2, 2))])
    def cube(a):
        return a**3.0

    output = cube({"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])})

    assert output.to_list() == [[1.0, 8.0], [27.0, 64.0]]

    grad_cube_sum = nabla.grad_fn(
        lambda a: (a**3.0).sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    gradients = grad_cube_sum(
        {"a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]])}
    )

    assert gradients["a"].to_list() == [[3.0, 12.0], [27.0, 48.0]]


def test_jit_and_vjp_support_axis_reductions():
    @nabla.jit([("a", (2, 3))])
    def column_means(a):
        return a.mean(axis=0)

    @nabla.jit([("a", (2, 3))])
    def row_totals(a):
        return a.sum(axis=1)

    values = {"a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

    assert column_means(values).to_list() == [[2.5, 3.5, 4.5]]
    assert row_totals(values).to_list() == [[6.0], [15.0]]

    grad_column_means = nabla.grad_fn(
        lambda a: a.mean(axis=0).sum(),
        [("a", (2, 3))],
        nabla.Matrix([[1.0]]),
    )
    grad_row_totals = nabla.grad_fn(
        lambda a: a.sum(axis=1).sum(),
        [("a", (2, 3))],
        nabla.Matrix([[1.0]]),
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
    @nabla.jit([("a", (1, 2)), ("b", (2, 2))])
    def vertical_concat(a, b):
        return nabla.concat([a, b], axis=0)

    @nabla.jit([("left", (2, 1)), ("right", (2, 2))])
    def horizontal_concat(left, right):
        return nabla.concat([left, right], axis=1)

    vertical_values = {
        "a": nabla.Matrix([[1.0, 2.0]]),
        "b": nabla.Matrix([[3.0, 4.0], [5.0, 6.0]]),
    }
    horizontal_values = {
        "left": nabla.Matrix([[1.0], [2.0]]),
        "right": nabla.Matrix([[3.0, 4.0], [5.0, 6.0]]),
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

    grad_vertical = nabla.grad_fn(
        lambda a, b: nabla.concat([a, b], axis=0).sum(),
        [("a", (1, 2)), ("b", (2, 2))],
        nabla.Matrix([[1.0]]),
    )
    grad_horizontal = nabla.grad_fn(
        lambda left, right: nabla.concat([left, right], axis=1).sum(),
        [("left", (2, 1)), ("right", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    vertical_gradients = grad_vertical(vertical_values)
    horizontal_gradients = grad_horizontal(horizontal_values)

    assert vertical_gradients["a"].to_list() == [[1.0, 1.0]]
    assert vertical_gradients["b"].to_list() == [[1.0, 1.0], [1.0, 1.0]]
    assert horizontal_gradients["left"].to_list() == [[1.0], [1.0]]
    assert horizontal_gradients["right"].to_list() == [[1.0, 1.0], [1.0, 1.0]]


def test_grad_fn_evaluates_tanh_vjp():
    grad_model = nabla.grad_fn(
        lambda a: a.tanh().sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [
            [1.0 - math.tanh(0.0) ** 2, 1.0 - math.tanh(1.0) ** 2],
            [1.0 - math.tanh(-1.0) ** 2, 1.0 - math.tanh(2.0) ** 2],
        ],
    )


def test_grad_fn_evaluates_exp_vjp():
    grad_model = nabla.grad_fn(
        lambda a: a.exp().sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[math.exp(0.0), math.exp(1.0)], [math.exp(-1.0), math.exp(2.0)]],
    )


def test_grad_fn_evaluates_log_vjp():
    grad_model = nabla.grad_fn(
        lambda a: a.log().sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[1.0, 2.0], [4.0, 8.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[1.0, 0.5], [0.25, 0.125]],
    )


def test_grad_fn_evaluates_sqrt_vjp():
    grad_model = nabla.grad_fn(
        lambda a: a.sqrt().sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[1.0, 4.0], [9.0, 16.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[0.5, 0.25], [1.0 / 6.0, 0.125]],
    )


def test_grad_fn_evaluates_sin_vjp():
    grad_model = nabla.grad_fn(
        lambda a: a.sin().sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[math.cos(0.0), math.cos(1.0)], [math.cos(-1.0), math.cos(2.0)]],
    )


def test_grad_fn_evaluates_cos_vjp():
    grad_model = nabla.grad_fn(
        lambda a: a.cos().sum(),
        [("a", (2, 2))],
        nabla.Matrix([[1.0]]),
    )

    gradients = grad_model(
        {"a": nabla.Matrix([[0.0, 1.0], [-1.0, 2.0]])}
    )

    assert_close_rows(
        gradients["a"].to_list(),
        [[-math.sin(0.0), -math.sin(1.0)], [-math.sin(-1.0), -math.sin(2.0)]],
    )


def test_jit_decorator_factory_evaluates_primal():
    @nabla.jit([("a", (2, 3)), ("b", (3, 2)), ("bias", (2, 2))])
    def model(a, b, bias):
        return (a @ b) + bias

    output = model(
        {
            "a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": nabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "bias": nabla.Matrix([[0.1, 0.2], [0.3, 0.4]]),
        }
    )

    assert output.shape == (2, 2)
    assert output.to_list() == [[58.1, 64.2], [139.3, 154.4]]


def test_jit_traces_once_at_decoration_time():
    calls = {"count": 0}

    @nabla.jit([("a", (2, 2)), ("b", (2, 2))])
    def model(a, b):
        calls["count"] += 1
        return a + b

    assert calls["count"] == 1

    first = model(
        {
            "a": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
            "b": nabla.Matrix([[5.0, 6.0], [7.0, 8.0]]),
        }
    )
    second = model(
        {
            "a": nabla.Matrix([[10.0, 20.0], [30.0, 40.0]]),
            "b": nabla.Matrix([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert first.to_list() == [[6.0, 8.0], [10.0, 12.0]]
    assert second.to_list() == [[11.0, 22.0], [33.0, 44.0]]
    assert calls["count"] == 1


def test_grad_fn_evaluates_composed_add_matmul_vjp():
    def model(a, b, bias):
        return (a @ b) + bias

    grad_model = nabla.grad_fn(
        model,
        [("a", (2, 3)), ("b", (3, 2)), ("bias", (2, 2))],
        nabla.Matrix([[1.0, 0.5], [-1.0, 2.0]]),
    )
    gradients = grad_model(
        {
            "a": nabla.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": nabla.Matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "bias": nabla.Matrix([[0.1, 0.2], [0.3, 0.4]]),
        }
    )

    assert gradients["a"].to_list() == [[11.0, 14.0, 17.0], [9.0, 11.0, 13.0]]
    assert gradients["b"].to_list() == [[-3.0, 8.5], [-3.0, 11.0], [-3.0, 13.5]]
    assert gradients["bias"].to_list() == [[1.0, 0.5], [-1.0, 2.0]]


if __name__ == "__main__":
    test_tensor_stateless_random_keys_and_glorot_initializer_are_reproducible()
    test_matrix_matmul()
    test_matrix_add()
    test_matrix_sub()
    test_matrix_mul()
    test_matrix_transpose_method_and_property()
    test_matrix_reshape_preserves_row_major_order()
    test_matrix_reshape_rejects_size_change()
    test_matrix_concat_axis()
    test_matrix_supports_scalar_literals()
    test_matrix_broadcasts_scalar_matrix_for_elementwise_ops()
    test_matrix_broadcasts_row_and_column_matrices_for_elementwise_ops()
    test_tensor_broadcasts_trailing_axes_and_reshapes()
    test_tensor_rejects_data_with_the_wrong_size()
    test_tensor_ones_and_full_validate_rank_n_shapes()
    test_tensor_arange_creates_coordinate_vectors_and_rejects_invalid_ranges()
    test_tensor_linspace_includes_endpoints_for_collocation_grids()
    test_tensor_eye_creates_square_and_rectangular_identity_arrays()
    test_tensor_transpose_reorders_rank_n_axes_and_validates_permutations()
    test_tensor_reductions_match_trace_tensor_axis_semantics()
    test_tensor_elementwise_math_matches_python_math()
    test_tensor_supports_numeric_scalars_on_both_sides()
    test_tensor_power_supports_integer_and_float_exponents()
    test_tensor_comparisons_and_where_support_rank_n_broadcasting()
    test_tensor_broadcast_to_materializes_rank_n_contiguous_storage()
    test_tensor_matmul_broadcasts_batch_axes()
    test_tensor_concat_supports_rank_n_axes_and_validates_shapes()
    test_tensor_slice_creates_a_strided_read_only_view()
    test_tensor_getitem_supports_contiguous_slices_and_negative_indices()
    test_tensor_stack_supports_negative_axes()
    test_trace_tensor_evaluates_rank_n_scalar_loss_vjp()
    test_trace_tensor_batched_matmul_scalar_loss_vjp()
    test_trace_tensor_symbolic_jvp_keeps_parameter_gradients()
    test_trace_tensor_symbolic_vjp_matches_cpu_vjp()
    test_trace_tensor_symbolic_vjp_supports_composed_loss_nodes()
    test_trace_tensor_concat_supports_rank_n_ad_and_symbolic_transforms()
    test_trace_tensor_slice_supports_rank_n_ad_and_symbolic_transforms()
    test_trace_tensor_getitem_preserves_slice_ad_and_backend_parity()
    test_trace_tensor_slice_rejects_invalid_ranges()
    test_trace_tensor_broadcast_to_supports_rank_n_ad_and_symbolic_transforms()
    test_trace_tensor_stack_supports_symbolic_vjp()
    test_cuda_trace_tensor_concat_keeps_primal_and_symbolic_vjp_on_device()
    test_mlx_trace_tensor_compiles_and_matches_cpu()
    test_mlx_symbolic_vjp_executes_mlp_bias_gradient()
    test_mlx_trace_tensor_executes_masked_loss()
    test_mlx_symbolic_vjp_executes_masked_loss_gradient()
    test_cuda_trace_tensor_slice_keeps_primal_and_symbolic_vjp_on_device()
    test_cuda_trace_tensor_broadcast_to_keeps_primal_and_symbolic_vjp_on_device()
    test_cuda_multi_parameter_sgd_keeps_gradient_plans_synchronized()
    test_cuda_adam_keeps_optimizer_state_on_device()
    test_cuda_plan_evaluates_with_static_inputs_retained_on_device()
    test_cuda_plan_reuses_dead_temporary_buffers()
    test_cuda_fuses_rank_two_matmul_bias_tanh_epilogue()
    test_cuda_multi_parameter_adam_keeps_gradient_plans_synchronized()
    test_cuda_adam_optimizer_updates_minibatch_inputs_without_resetting_state()
    test_cuda_adam_vjp_optimizer_updates_parameters_from_one_shared_graph()
    test_cuda_batched_matmul_vjp_matches_cpu_trace_evaluation()
    test_cuda_global_reductions_match_cpu_and_reset_output_buffers()
    test_multi_axis_keepdims_reductions_match_cpu_on_mlx_and_cuda()
    test_cuda_sqrt_composite_lowering_matches_cpu()
    test_cuda_checked_div_and_log_lowering_matches_cpu()
    test_trace_tensor_tanh_scalar_loss_supports_jvp_and_vjp()
    test_trace_tensor_subtraction_supports_jvp_and_vjp()
    test_trace_tensor_supports_numeric_scalar_literals()
    test_trace_tensor_division_supports_jvp_and_vjp()
    test_trace_tensor_exp_supports_jvp_and_vjp()
    test_trace_tensor_mean_supports_jvp_and_vjp()
    test_trace_tensor_sin_supports_jvp_and_vjp()
    test_trace_tensor_sqrt_supports_jvp_and_vjp()
    test_trace_tensor_where_routes_gradients_and_masks_condition_derivatives()
    test_trace_tensor_log_supports_jvp_and_vjp()
    test_trace_tensor_cos_supports_jvp_and_vjp()
    test_trace_tensor_powi_supports_second_order_ad()
    test_trace_tensor_axis_reductions_support_ad()
    test_trace_tensor_transpose_supports_rank_n_ad()
    test_trace_tensor_reshape_supports_jvp_and_vjp()
    test_tensor_grad_scalar_fn_reuses_a_compiled_plan()
    test_tensor_value_and_grad_fn_returns_scalar_value_and_gradients()
    test_tensor_hessian_and_hvp_scalar_fn_reuse_a_compiled_plan()
    test_tensor_jit_fn_reuses_a_compiled_plan()
    test_tensor_vmap_fn_traces_one_batched_plan()
    test_tensor_vmap_fn_supports_in_axes_out_axis_and_unmapped_inputs()
    test_tensor_vmap_fn_reductions_and_transpose_preserve_batch_axis()
    test_tensor_vmap_cuda_and_mlx_fn_use_the_same_batched_trace()
    test_tensor_vmap_vjp_matches_per_example_loop_and_aggregates_unmapped_gradient()
    test_tensor_vmap_jvp_matches_per_example_loop_with_nonleading_axes()
    test_tensor_vmap_cuda_vjp_matches_cpu_single_batched_plan()
    test_tensor_vmap_batched_mlp_gradients_match_loop_on_cpu_and_cuda()
    test_tensor_jit_cuda_fn_reuses_a_callable_cuda_plan()
    test_tensor_value_and_grad_cuda_fn_uses_one_callable_plan()
    test_cuda_adam_loss_optimizer_owns_scalar_loss_and_parameters()
    test_tensor_vjp_fn_reuses_a_compiled_plan_with_runtime_cotangent()
    test_tensor_jvp_fn_reuses_a_compiled_plan_with_runtime_tangent()
    test_tensor_jacobian_fn_reuses_a_compiled_plan()
    test_adam_updates_tensor_parameters_with_persistent_moments()
    test_adam_invalid_step_does_not_advance_optimizer_state()
    test_sum_gradients_combines_named_tensors()
    test_poisson_residual_training_converges_with_symbolic_jvp_and_adam()
    test_batched_poisson_collocation_training_converges()
    test_poisson_training_aggregates_boundary_and_residual_gradients()
    test_symbolic_jvp_residual_backpropagates_through_two_layer_mlp()
    test_two_layer_mlp_poisson_training_converges()
    test_standard_sine_poisson_training_recovers_pi()
    test_trace_tensor_compile_cpu_eliminates_unreachable_nodes()
    test_trace_tensor_compile_cpu_commons_identical_pure_nodes()
    test_matrix_neg_and_scalar_div()
    test_matrix_elementwise_division()
    test_matrix_integer_power()
    test_matrix_float_power()
    test_matrix_sum()
    test_matrix_sum_axis()
    test_matrix_mean()
    test_matrix_mean_axis()
    test_matrix_tanh()
    test_matrix_exp()
    test_matrix_log()
    test_matrix_sqrt()
    test_matrix_sin()
    test_matrix_cos()
    test_matrix_gt_and_where_support_masks()
    test_matrix_rejects_incompatible_shapes()
    test_trace_graph_records_matmul()
    test_trace_function_records_python_matmul()
    test_trace_graph_ir_includes_backend_lowering_attrs()
    test_trace_graph_lowers_to_deterministic_backend_text()
    test_trace_graph_evaluates_direct_jvp()
    test_trace_graph_compiles_an_immutable_cpu_execution_plan()
    test_cpu_execution_plan_evaluates_vjp()
    test_cpu_execution_plan_eliminates_unreachable_trace_nodes()
    test_cpu_execution_plan_eliminates_repeated_unary_subexpressions()
    test_grad_traces_and_evaluates_vjp()
    test_grad_fn_reuses_transform_callable()
    test_grad_fn_evaluates_add_vjp()
    test_grad_fn_evaluates_mul_vjp()
    test_grad_fn_evaluates_square_sum_vjp()
    test_grad_fn_evaluates_square_mean_vjp()
    test_grad_fn_evaluates_squared_error_vjp()
    test_grad_scalar_fn_seeds_scalar_cotangent()
    test_grad_scalar_decorator_factory()
    test_grad_fn_traces_once_at_transform_creation()
    test_grad_scalar_fn_traces_once_at_transform_creation()
    test_grad_scalar_decorator_traces_once_at_decoration_time()
    test_value_and_grad_fn_returns_value_and_traces_once()
    test_vjp_fn_traces_once_and_uses_runtime_cotangent()
    test_jacobian_fn_traces_once_and_returns_dense_jacobian()
    test_jacobians_fn_traces_once_and_returns_dense_jacobians_for_each_input()
    test_jvp_fn_traces_once_and_returns_primal_and_tangent()
    test_jvp_fn_supports_multiple_input_tangents()
    test_jit_and_vjp_support_transpose()
    test_jit_and_vjp_support_scalar_matrix_broadcast()
    test_jit_and_vjp_support_reshape()
    test_jit_and_grad_support_scalar_literals()
    test_jit_and_grad_support_neg_and_scalar_division()
    test_jit_and_grad_support_elementwise_division()
    test_jit_and_vjp_support_row_and_column_matrix_broadcast()
    test_jit_and_vjp_support_where_masks()
    test_jit_and_grad_support_integer_power()
    test_jit_and_grad_support_float_power()
    test_jit_and_vjp_support_axis_reductions()
    test_jit_and_vjp_support_concat()
    test_grad_fn_evaluates_tanh_vjp()
    test_grad_fn_evaluates_exp_vjp()
    test_grad_fn_evaluates_log_vjp()
    test_grad_fn_evaluates_sqrt_vjp()
    test_grad_fn_evaluates_sin_vjp()
    test_grad_fn_evaluates_cos_vjp()
    test_jit_decorator_factory_evaluates_primal()
    test_jit_traces_once_at_decoration_time()
    test_grad_fn_evaluates_composed_add_matmul_vjp()
