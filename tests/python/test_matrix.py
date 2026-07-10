import math

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
