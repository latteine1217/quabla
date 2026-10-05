"""Ordered AOT Program inspection and compatibility with the legacy facade."""

import quabla as qb


def assert_close(actual, expected):
    expected = qb.asarray(expected)
    assert actual.shape == expected.shape
    assert all(
        abs(a - b) < 1e-10
        for a, b in zip(actual.to_flat_list(), expected.to_flat_list())
    ), (actual.to_flat_list(), expected.to_flat_list())


def assert_raises(kind, function, *args, match=None):
    try:
        function(*args)
    except kind as error:
        if match is not None:
            assert match in str(error), str(error)
        return error
    raise AssertionError(f"expected {kind.__name__}")


def test_native_program_retains_order_shapes_and_duplicate_outputs():
    graph = qb._quabla.TensorTraceGraph()
    x = graph.input("x", [2], qb.float64)
    squared = x * x
    program = graph._as_program([squared.sum(), x + 1, squared, x + 1])
    assert isinstance(program, qb.Program)
    assert program.output_shapes == [[], [2], [2], [2]]
    assert "outputs: [[], [2], [2], [2]]" in program.lower_text()
    executable = program.compile()
    assert isinstance(executable, qb.Executable)
    assert executable.target == "cpu"
    values = executable.evaluate({"x": qb.array([2.0, 3.0])})
    assert isinstance(values, list)
    assert len(values) == 4
    for actual, expected in zip(values, (13.0, [3.0, 4.0], [4.0, 9.0], [3.0, 4.0])):
        assert_close(actual, expected)
    assert_close(executable({"x": qb.array([1.0, 4.0])})[0], 17.0)
    assert executable.node_count > 0


def test_many_transforms_reject_without_selecting_the_first_result():
    graph = qb._quabla.TensorTraceGraph()
    x = graph.input("x", [], qb.float64)
    program = graph._as_program([x, x * x])
    assert_raises(qb.UnsupportedOperationError, program.jvp, "x", match="multi-output")
    assert_raises(
        qb.UnsupportedOperationError, program.vjp, "seed", match="multi-output"
    )
    assert_raises(qb.UnsupportedOperationError, lambda: program.output_shape)
    # A one-element ordered result list has the same explicit list contract.
    one = graph._as_program([x])
    assert isinstance(one.compile()({"x": qb.array(2.0)}), list)


def test_program_rejects_outputs_from_another_graph():
    left, right = qb._quabla.TensorTraceGraph(), qb._quabla.TensorTraceGraph()
    output = right.input("other", [], qb.float64)
    assert_raises(ValueError, left._as_program, [output], match="different graph")


def test_empty_program_supports_inspection_and_explicit_native_rejection():
    graph = qb._quabla.TensorTraceGraph()
    program = graph._as_program([])
    assert isinstance(program, qb.Program)
    assert program.output_shapes == []
    assert "outputs: []" in program.lower_text()
    assert_raises(
        qb.UnsupportedOperationError, program.compile, match="no array outputs"
    )


def test_lowered_facade_preserves_pytree_and_constant_outputs():
    x = qb.array([2.0, 3.0])
    lowered = qb.jit(lambda x: {"a": x.sum(), "z": (x * x, None, True, 7.0)}).lower(x)
    assert isinstance(lowered.program, qb.Program)
    assert lowered.program.output_shapes == [[], [2]]
    assert lowered.as_text() == lowered.program.lower_text()
    result = lowered.compile()(x)
    assert_close(result["a"], 5.0)
    assert_close(result["z"][0], [4.0, 9.0])
    assert result["z"][1:] == (None, True, 7.0)
    assert_raises(ValueError, lowered.compile(), qb.array([1.0]), match="match")
    constant = qb.jit(lambda x: (3.0, True, None)).lower(x)
    assert isinstance(constant.program, qb.Program)
    assert constant.program.output_shapes == []
    assert constant.compile()(x) == (3.0, True, None)


def test_single_output_compiler_facade_keeps_tensor_and_ad_contract():
    program = qb.Compiler().trace(lambda x: (x * x).sum(), [("x", [2])])
    assert program.output_shape == []
    assert program.output_shapes == [[]]
    assert "outputs:" not in program.lower_text()
    inputs = {"x": qb.array([2.0, 3.0])}
    assert isinstance(program.compile().evaluate(inputs), qb.Tensor)
    assert_close(program.compile()(inputs), 13.0)
    tangent = program.jvp("x").compile()
    assert_close(tangent(inputs), 10.0)
    gradient = program.vjp("seed")["x"].compile()
    assert_close(gradient({**inputs, "seed": qb.array(1.0)}), [4.0, 6.0])
    assert_raises(ValueError, program.compile, "unknown", match="target")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
