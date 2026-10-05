"""S7 eager, staged, and differentiated control-flow wrapper checks."""

import os

import quabla as qb
from quabla._control import cond, fori_loop, scan


def assert_close(actual, expected, tolerance=1e-10):
    expected = qb.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert all(
        abs(a - b) <= tolerance
        for a, b in zip(actual.to_flat_list(), expected.to_flat_list())
    ), (actual.to_flat_list(), expected.to_flat_list())


def assert_raises(kind, function, *args, match=None, **kwargs):
    try:
        function(*args, **kwargs)
    except kind as error:
        if match is not None:
            assert match in str(error), str(error)
        return error
    raise AssertionError(f"expected {kind.__name__}")


def loop_loss(initial, scale, *, unroll=False):
    return fori_loop(
        0,
        3,
        lambda i, carry, s: carry * s + i,
        initial,
        operands=(scale,),
        unroll=unroll,
    ).sum()


def scan_loss(initial, scale, *, unroll=False):
    carry, outputs = scan(
        lambda carry, i, s: (carry * s + i, carry * s + i),
        initial,
        length=3,
        operands=(scale,),
        unroll=unroll,
    )
    return carry.sum() + outputs.sum()


def test_cond_concrete_is_lazy_and_keeps_operands():
    calls = []
    x = qb.array([2.0])

    def on_true(value):
        calls.append("true")
        return value * 3.0

    def on_false(value):
        calls.append("false")
        return value - 4.0

    assert_close(cond(True, on_true, on_false, x), [6.0])
    assert_close(cond(qb.array(False), on_true, on_false, x), [-2.0])
    assert calls == ["true", "false"]
    # A concrete predicate remains lazy inside a trace.
    assert_close(qb.jit(lambda t: cond(True, lambda y: y, lambda y: 1 / 0, t))(x), x)


def test_cond_dynamic_predicate_is_lazy_and_differentiable():
    def function(pred, x):
        return cond(pred, lambda t: t * t, lambda t: t.log(), x)

    compiled = qb.jit(function)
    assert_close(compiled(qb.array(True), qb.array(-2.0)), 4.0)
    assert_close(compiled(qb.array(False), qb.array(2.0)), qb.array(2.0).log())
    assert_close(qb.grad(function, argnums=1)(qb.array(True), qb.array(-2.0)), -4.0)
    second = qb.grad(qb.grad(function, argnums=1), argnums=1)
    assert_close(second(qb.array(True), qb.array(-2.0)), 2.0)


def test_eager_loop_and_scan_argument_order_and_bounds():
    initial, scale = qb.array(1.0), qb.array(2.0)
    assert_close(loop_loss(initial, scale), 12.0)
    assert_close(scan_loss(initial, scale), 31.0)
    carry, outputs = scan(
        lambda carry, i: (carry + i, carry + i),
        initial,
        length=3,
    )
    assert_close(carry, 4.0)
    assert_close(outputs, [1.0, 2.0, 4.0])
    assert_close(fori_loop(2, 4, lambda i, carry: carry + i, initial), 6.0)
    assert fori_loop(2, 2, lambda i, carry: 1 / 0, initial) is initial


def test_region_and_unrolled_loop_have_analytic_derivatives():
    initial, scale = qb.array(1.0), qb.array(2.0)
    for unroll in (False, True):

        def function(x, s):
            return loop_loss(x, s, unroll=unroll)

        value, gradients = qb.value_and_grad(function, argnums=(0, 1))(initial, scale)
        assert_close(value, 12.0)
        assert_close(gradients[0], 8.0)
        assert_close(gradients[1], 13.0)
        _, tangent = qb.jvp(function, (initial, scale), (qb.array(1.0), qb.array(0.0)))
        assert_close(tangent, 8.0)
        _, pullback = qb.vjp(function, initial, scale)
        dx, ds = pullback(qb.array(2.0))
        assert_close(dx, 16.0)
        assert_close(ds, 26.0)
        _, hvp = qb.jvp(
            qb.grad(function, argnums=1),
            (initial, scale),
            (qb.array(0.0), qb.array(1.0)),
        )
        assert_close(hvp, 12.0)


def test_region_and_unrolled_scan_differentiate_carry_and_outputs():
    initial, scale = qb.array(1.0), qb.array(2.0)
    for unroll in (False, True):

        def function(x, s):
            return scan_loss(x, s, unroll=unroll)

        value, gradients = qb.value_and_grad(function, argnums=(0, 1))(initial, scale)
        assert_close(value, 31.0)
        assert_close(gradients[0], 22.0)
        assert_close(gradients[1], 31.0)
        _, tangent = qb.jvp(function, (initial, scale), (qb.array(0.0), qb.array(1.0)))
        assert_close(tangent, 31.0)
        _, hvp = qb.jvp(
            qb.grad(function, argnums=1),
            (initial, scale),
            (qb.array(0.0), qb.array(1.0)),
        )
        assert_close(hvp, 26.0)


def test_wrappers_match_existing_regions_and_trace_once():
    counts = []

    def body(i, carry, scale):
        counts.append(1)
        return carry * scale + i

    wrapped = qb.jit(lambda x, s: fori_loop(0, 3, body, x, operands=(s,)))
    native = qb.jit(lambda x, s: qb._quabla.tensor_fori_loop_region(0, 3, body, x, [s]))
    x, s = qb.array([1.0, 2.0]), qb.array([2.0, 3.0])
    assert_close(wrapped(x, s), [12.0, 59.0])
    assert len(counts) == 1
    assert_close(wrapped(x, s), native(x, s))
    assert len(counts) == 2
    wrapped_scan = qb.jit(
        lambda x, s: scan(lambda c, i, s: (c * s + i, c), x, length=3, operands=(s,))
    )
    native_scan = qb.jit(
        lambda x, s: qb._quabla.tensor_scan_region(
            0, 3, lambda i, c, s: (c * s + i, c), x, [s]
        )
    )
    for actual, expected in zip(wrapped_scan(x, s), native_scan(x, s)):
        assert_close(actual, expected)


def test_mixed_eager_and_traced_explicit_operands_keep_dtype():
    x = qb.array([1.0, 2.0], dtype=qb.float32)
    scale = qb.array(2.0, dtype=qb.float32)
    loop = qb.jit(
        lambda x: fori_loop(0, 3, lambda i, c, s: c + s, x, operands=(scale,))
    )
    result = loop(x)
    assert result.dtype == qb.float32
    assert_close(result, [7.0, 8.0])
    # The carry itself may be eager when an explicit operand is traced.
    assert_close(qb.grad(lambda s: loop_loss(qb.array(1.0), s))(qb.array(2.0)), 13.0)
    assert_close(
        qb.jit(lambda pred: cond(pred, lambda s: s + 1, lambda s: s - 1, scale))(
            qb.array(True)
        ),
        3.0,
    )


def test_nested_regions_and_float32_region_indices():
    def nested(initial, scale):
        return fori_loop(
            0,
            2,
            lambda i, carry, s: cond(
                s > 0, lambda c, s: c + s, lambda c, s: c - s, carry, s
            ),
            initial,
            operands=(scale,),
        )

    initial, scale = qb.array(1.0), qb.array(2.0)
    assert_close(qb.jit(nested)(initial, scale), 5.0)
    assert_close(qb.grad(nested, argnums=1)(initial, scale), 2.0)
    indices = []

    def add_index(i, carry):
        indices.append(i.dtype)
        return carry + i

    result = qb.jit(lambda x: fori_loop(0, 3, add_index, x))(
        qb.array(1.0, dtype=qb.float32)
    )
    assert indices == [qb.float32]
    assert result.dtype == qb.float32
    assert_close(result, 4.0)


def test_implicit_tracer_capture_errors_name_explicit_operands():
    functions = [
        lambda x: fori_loop(0, 2, lambda i, carry: carry + x, x),
        lambda x: scan(lambda carry, i: (carry + x, carry), x, length=2),
        lambda x: cond(x > 0, lambda: x * 2, lambda: x),
        lambda x: fori_loop(0, 2, lambda i, carry: x, x),
        lambda x: scan(lambda carry, i: (carry, x), x, length=2),
    ]
    for function in functions:
        assert_raises(
            qb.TracerError, qb.jit(function), qb.array(1.0), match="operands="
        )
    assert_close(
        qb.jit(lambda x: fori_loop(0, 2, lambda i, c, x: c + x, x, operands=(x,)))(
            qb.array(1.0)
        ),
        3.0,
    )


def test_rejections_preserve_user_exceptions_and_single_array_scope():
    x = qb.array(1.0)
    assert_raises(ValueError, scan, lambda c, i: (c, c), x, length=0, match="positive")
    assert_raises(ValueError, fori_loop, 3, 2, lambda i, c: c, x)
    assert_raises(ValueError, fori_loop, -1, 2, lambda i, c: c, x)
    assert_raises(
        TypeError, scan, lambda c, i: (c, c), (x,), length=2, match="single array"
    )
    for stage in (lambda f: f, qb.jit):
        assert_raises(
            ValueError,
            stage(lambda t: fori_loop(0, 2, lambda i, c: c.reshape([1]), t)),
            x,
            match="shape",
        )
        assert_raises(
            TypeError,
            stage(lambda t: scan(lambda c, i: c, t, length=2)),
            x,
            match="tuple",
        )

        def broken(t):
            def body(i, carry):
                raise ValueError("user body failure")

            return fori_loop(0, 1, body, t)

        error = assert_raises(ValueError, stage(broken), x, match="user body failure")
        assert type(error) is ValueError

        def misleading(t):
            def body(i, carry):
                raise ValueError("user body returned data from a different graph")

            return fori_loop(0, 1, body, t)

        error = assert_raises(
            ValueError, stage(misleading), x, match="user body returned"
        )
        assert type(error) is ValueError
    assert_raises(
        ValueError,
        qb.jit(lambda p: cond(p, lambda: p, lambda: p)),
        qb.array([True]),
        match="scalar",
    )


def test_optional_device_loop_vjp_and_hvp_parity():
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue
        initial = qb.array([1.0, 2.0], dtype=qb.float32)
        scale = qb.array([2.0, 3.0], dtype=qb.float32)
        derivative = qb.grad(loop_loss, argnums=1)

        def hvp(x, s):
            return qb.jvp(
                derivative,
                (x, s),
                (qb.zeros([2], qb.float32), qb.ones([2], qb.float32)),
            )[1]

        for operation in (qb.value_and_grad(loop_loss, argnums=(0, 1)), hvp):
            cpu = qb.jit(operation)(initial, scale)
            actual = qb.jit(operation, device=device)(initial, scale)
            cpu_leaves, _ = qb.tree.flatten(cpu)
            actual_leaves, _ = qb.tree.flatten(actual)
            for value, expected in zip(actual_leaves, cpu_leaves):
                assert_close(value, expected, 1e-4)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
