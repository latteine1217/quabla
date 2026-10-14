"""Eager `Tensor`, dtype, and `Compiler` behaviour."""

import array
import math

import quabla

from _support import run


def assert_close_rows(actual, expected, tol=1e-12):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert abs(actual_value - expected_value) <= tol


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

    # Division by zero follows IEEE 754, like traced and device execution.
    assert (tensor / 0.0).to_flat_list() == [math.inf, -math.inf, math.inf, math.inf]
    zeros = quabla.Tensor([2, 2], [0.0, -0.0, 1.0, 1.0])
    assert (tensor / zeros).to_flat_list()[:2] == [math.inf, math.inf]
    assert math.isnan((tensor * 0.0 / 0.0).to_flat_list()[0])


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

    # Strided slices lower to a contiguous slice plus a gather.
    assert tensor[::2].to_flat_list() == [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]

    for index in (slice(1, 1), (0, 1, 2)):
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


NAN = float("nan")


INF = float("inf")


BOOL_X = [1.0, 2.0, NAN, INF, -INF, 2.0]


BOOL_Y = [0.5, 2.0, 2.0, INF, 0.0, NAN]


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


def test_bool_dtype_object_and_eager_comparisons_follow_ieee_semantics():
    assert repr(quabla.bool_) == "quabla.bool_"
    assert str(quabla.bool_) == "bool"
    assert quabla.bool_.name == "bool"
    assert {quabla.bool_: "mask"}[quabla.bool_] == "mask"
    assert quabla.bool_ != quabla.float64
    mask = quabla.Tensor([4], [0.0, 3.0, NAN, -INF], dtype=quabla.bool_)
    assert mask.dtype == quabla.bool_
    assert mask.to_flat_list() == [0.0, 1.0, 1.0, 1.0]
    assert repr(mask) == "Tensor([False,  True,  True,  True], dtype=bool)"

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



def test_program_exports_the_verified_stablehlo_subset():
    program = quabla.Compiler().trace(
        lambda a, b: (a + b).tanh(), [("a", [2, 1]), ("b", [2, 1])]
    )
    assert program.stablehlo_text() == "\n".join(
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


if __name__ == "__main__":
    run(globals())
