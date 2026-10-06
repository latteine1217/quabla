"""S7 eager, staged, and differentiated control-flow wrapper checks."""

import os

import quabla as qb
from quabla._control import cond, fori_loop, scan, while_loop


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


def ignoring_cond(x, a):
    # As in JAX, a branch may ignore an operand the other branch reads (the
    # first cond) or return a constant (the second cond).
    return cond(
        x.sum() > 0.0,
        lambda t, s: (t * s).sum(),
        lambda t, s: (t * t).sum(),
        x,
        a,
    ) + cond(x.sum() > 0.0, lambda t: qb.ones([], t.dtype), lambda t: t.sum(), x)


def test_traced_cond_branches_may_ignore_operands_or_return_constants():
    x, a = qb.array([1.0, 2.0]), qb.array([0.5, -1.0])
    for point, value, grad_x, grad_a in [
        (x, -0.5, [0.5, -1.0], [1.0, 2.0]),
        (-x, 2.0, [-1.0, -3.0], [0.0, 0.0]),
    ]:
        assert_close(qb.jit(ignoring_cond)(point, a), value)
        gradients = qb.grad(ignoring_cond, argnums=(0, 1))(point, a)
        assert_close(gradients[0], grad_x)
        assert_close(gradients[1], grad_a)
    # An ignored operand is referenced without being read, so NaN and inf
    # in it reach neither the value nor the gradients.
    poisoned = qb.array([float("nan"), float("inf")])
    assert_close(qb.jit(ignoring_cond)(-x, poisoned), 2.0)
    assert_close(qb.grad(ignoring_cond, argnums=1)(-x, poisoned), [0.0, 0.0])
    # Without operands both branches may return constants.
    constant = qb.jit(
        lambda t: cond(
            t.sum() > 0.0, lambda: qb.array([1.0, 2.0]), lambda: 3.0 * qb.ones([2])
        )
    )
    assert_close(constant(x), [1.0, 2.0])
    assert_close(constant(-x), [3.0, 3.0])


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


def test_loop_and_scan_bodies_may_ignore_operands():
    initial, unused, scale = qb.array([1.0, 2.0]), qb.array([0.5, -1.0]), qb.array(3.0)
    for unroll in (False, True):

        def loop(x, u, s):
            return fori_loop(
                0, 2, lambda i, c, u_, s_: c * s_, x, operands=(u, s), unroll=unroll
            ).sum()

        def scanned(x, u, s):
            return scan(
                lambda c, i, u_, s_: (c * s_, c),
                x,
                length=2,
                operands=(u, s),
                unroll=unroll,
            )[1].sum()

        for function, value, grad_x, grad_s in [
            (loop, 27.0, [9.0, 9.0], 18.0),
            (scanned, 12.0, [4.0, 4.0], 3.0),
        ]:
            assert_close(qb.jit(function)(initial, unused, scale), value)
            gradients = qb.grad(function, argnums=(0, 1, 2))(initial, unused, scale)
            assert_close(gradients[0], grad_x)
            assert_close(gradients[1], [0.0, 0.0])
            assert_close(gradients[2], grad_s)


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

        operations = (
            qb.value_and_grad(loop_loss, argnums=(0, 1)),
            hvp,
            qb.value_and_grad(ignoring_cond, argnums=(0, 1)),
        )
        for operation in operations:
            # The negated carry takes the other branch of `ignoring_cond`.
            for x in (initial, -initial):
                cpu = qb.jit(operation)(x, scale)
                actual = qb.jit(operation, device=device)(x, scale)
                cpu_leaves, _ = qb.tree.flatten(cpu)
                actual_leaves, _ = qb.tree.flatten(actual)
                for value, expected in zip(actual_leaves, cpu_leaves):
                    assert_close(value, expected, 1e-4)


def rotate(carry, scale):
    # A slice/concat rotation plus a full reduction: neither is lane-local, so
    # CUDA runs loops over this body host-driven instead of as fused kernels.
    return qb.concat([carry[1:], carry[:1]], 0) * scale + carry.sum() * 0.05


def host_driven_scan_loss(initial, scale):
    def body(carry, i, scale):
        nxt = (rotate(carry, scale) + i * 0.01).tanh()
        # The step output has fewer lanes than the carry.
        return nxt, nxt[:2] * nxt[1:3]

    carry, outputs = scan(body, initial, length=9, operands=(scale,))
    return (carry**2).sum() + (outputs**3).sum()


def host_driven_fori_loss(initial, scale):
    final = fori_loop(
        0, 9, lambda i, carry, scale: rotate(carry, scale).sin(), initial, operands=(scale,)
    )
    return (final**2).sum()


def host_driven_while(initial, scale):
    # Slot 0 counts iterations; the body rotates the other slots.
    counted = qb.concat([qb.zeros([1], initial.dtype), initial], 0)
    return while_loop(
        lambda c, scale: c[0] < 6.0,
        lambda c, scale: qb.concat([c[:1] + 1.0, rotate(c[1:], scale).sin()], 0),
        counted,
        operands=(scale,),
    )


def test_scan_hvp_with_a_rotating_body_matches_finite_differences():
    # The CPU reference of the CUDA test below, checked against central
    # differences of the gradient (truncation error O(h^2) ~ 1e-10).
    initial = qb.array([0.3, -0.2, 0.5, 0.1, -0.4])
    scale = qb.array([0.9, 1.1, 0.8, 1.05, 0.95])
    direction = qb.array([0.5, -1.0, 0.25, 2.0, -0.75])
    gradient = qb.jit(qb.grad(host_driven_scan_loss, argnums=(0, 1)))
    h = 1e-5
    for argnums in (0, 1):
        tangents = [qb.zeros([5]), qb.zeros([5])]
        tangents[argnums] = direction
        hvp = qb.jit(
            lambda x, s, t=tuple(tangents): qb.jvp(
                qb.grad(host_driven_scan_loss, argnums=(0, 1)), (x, s), t
            )[1]
        )(initial, scale)
        shifted = [[initial, scale], [initial, scale]]
        shifted[0][argnums] = shifted[0][argnums] + direction * h
        shifted[1][argnums] = shifted[1][argnums] - direction * h
        plus, minus = gradient(*shifted[0]), gradient(*shifted[1])
        for exact, upper, lower in zip(hvp, plus, minus):
            assert_close(exact, (upper - lower) / (2 * h), 1e-7)


def test_optional_cuda_host_driven_loop_derivatives_match_cpu():
    # Bodies that are not elementwise run as host-driven region loops on CUDA,
    # including Hessian-vector products (forward over reverse) through `scan`,
    # which used to be rejected there.
    if os.environ.get("QUABLA_CUDA_TEST") != "1":
        return
    for dtype, precision, tolerance in (
        (qb.float32, None, 1e-4),
        (qb.float64, "float64", 1e-11),
    ):
        initial = qb.array([0.3, -0.2, 0.5, 0.1, -0.4], dtype=dtype)
        scale = qb.array([0.9, 1.1, 0.8, 1.05, 0.95], dtype=dtype)
        direction = qb.array([0.5, -1.0, 0.25, 2.0, -0.75], dtype=dtype)
        zero = qb.zeros([5], dtype)

        def hvp(loss, argnums):
            gradient = qb.grad(loss, argnums=argnums)

            def run(x, s):
                tangents = (direction, zero) if argnums == 0 else (zero, direction)
                return qb.jvp(gradient, (x, s), tangents)[1]

            return run

        operations = {
            "scan value_and_grad": qb.value_and_grad(host_driven_scan_loss, argnums=(0, 1)),
            "scan hvp initial": hvp(host_driven_scan_loss, 0),
            "scan hvp scale": hvp(host_driven_scan_loss, 1),
            "scan jvp": lambda x, s: qb.jvp(host_driven_scan_loss, (x, s), (zero, direction)),
            "fori hvp initial": hvp(host_driven_fori_loss, 0),
            "fori jvp": lambda x, s: qb.jvp(host_driven_fori_loss, (x, s), (direction, zero)),
            "while jvp": lambda x, s: qb.jvp(host_driven_while, (x, s), (zero, direction)),
        }
        for name, operation in operations.items():
            expected = qb.tree.leaves(qb.jit(operation)(initial, scale))
            options = {"device": "cuda"}
            if precision is not None:
                options["precision"] = precision
            actual = qb.tree.leaves(qb.jit(operation, **options)(initial, scale))
            assert len(actual) == len(expected), name
            for got, want in zip(actual, expected):
                assert got.dtype == dtype, (name, got.dtype)
                scale_of = max([1.0] + [abs(value) for value in want.to_flat_list()])
                assert_close(got, want, tolerance * scale_of)


def collatz_like(carry, limit, scale):
    # Grows the carry until its first entry reaches `limit`; the trip count
    # depends on traced values.
    return while_loop(
        lambda c, limit, scale: c[0] < limit,
        lambda c, limit, scale: c * scale + 1.0,
        carry,
        operands=(limit, scale),
    )


def test_while_loop_eager_and_traced_agree():
    carry, limit, scale = qb.array([1.0, -2.0]), qb.array(100.0), qb.array(1.5)
    eager = collatz_like(carry, limit, scale)
    expected = [1.0, -2.0]
    while expected[0] < 100.0:
        expected = [value * 1.5 + 1.0 for value in expected]
    assert_close(eager, expected, 1e-12)
    assert_close(qb.jit(collatz_like)(carry, limit, scale), expected, 1e-12)
    # The trip count follows the traced bound, not the traced example.
    staged = qb.jit(collatz_like)
    assert_close(staged(carry, qb.array(1.0), scale), [1.0, -2.0], 0.0)
    assert_close(staged(carry, qb.array(1.5), scale), [2.5, -2.0], 0.0)
    # A predicate that is false at entry returns the initial carry eagerly.
    assert while_loop(lambda c: c < 0.0, lambda c: c + 1.0, 3.0).item() == 3.0


def test_while_loop_operands_may_be_ignored_and_closures_rejected():
    x = qb.array(1.0)

    def ignores_operand(x, unused):
        return while_loop(lambda c, u: c < 5.0, lambda c, u: c + 2.0, x, operands=(unused,))

    assert qb.jit(ignores_operand)(x, qb.array(7.0)).item() == 5.0

    def only_predicate_reads(x, bound):
        return while_loop(lambda c, b: c < b, lambda c, b: c * 2.0, x, operands=(bound,))

    assert qb.jit(only_predicate_reads)(x, qb.array(9.0)).item() == 16.0

    def closes_over(x, bound):
        return while_loop(lambda c: c < bound, lambda c: c * 2.0, x)

    assert_raises(qb.TracerError, qb.jit(closes_over), x, qb.array(9.0), match="operands=")
    assert_raises(
        TypeError,
        qb.jit(lambda x: while_loop(lambda c: c, lambda c: c + 1.0, x)),
        x,
        match="scalar bool",
    )
    assert_raises(
        ValueError,
        qb.jit(lambda x: while_loop(lambda c: c[0] < 1.0, lambda c: c[0:1], x)),
        qb.array([0.0, 1.0]),
        match="carry shape",
    )


def test_while_loop_forward_mode_matches_the_analytic_derivative():
    # c <- c * s while c < 50: from 1 with s = 2 the body runs six times, so
    # the result is s^6 and d/ds = 6 s^5 at a fixed trip count.
    def power(scale):
        return while_loop(
            lambda c, s: c < 50.0, lambda c, s: c * s, qb.array(1.0), operands=(scale,)
        )

    scale = qb.array(2.0)
    value, tangent = qb.jvp(power, (scale,), (qb.array(1.0),))
    assert value.item() == 64.0
    assert tangent.item() == 6.0 * 2.0**5
    staged = qb.jit(lambda s: qb.jvp(power, (s,), (qb.array(1.0),)))(scale)
    assert staged[0].item() == 64.0 and staged[1].item() == 192.0
    # The tangent of the carry follows the primal iterations.
    def scaled(initial):
        return while_loop(lambda c: c < 50.0, lambda c: c * 3.0, initial)

    assert qb.jvp(scaled, (qb.array(1.0),), (qb.array(1.0),))[1].item() == 81.0


def test_while_loop_reverse_mode_and_vmap_are_rejected_clearly():
    def power(scale):
        return while_loop(
            lambda c, s: c < 50.0, lambda c, s: c * s, qb.array(1.0), operands=(scale,)
        )

    for transform in (qb.grad, lambda f: qb.jit(qb.grad(f))):
        error = assert_raises(Exception, transform(power), qb.array(2.0))
        assert "while_loop" in str(error) and "fori_loop" in str(error), str(error)
    for transform in (qb.vmap, lambda f: qb.jit(qb.vmap(f))):
        assert_raises(
            qb.UnsupportedOperationError,
            transform(power),
            qb.array([2.0, 3.0]),
            match="vmap cannot batch a while_loop",
        )


def test_optional_device_while_loop_parity():
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue
        carry = qb.array([1.0, -2.0], dtype=qb.float32)
        limit, scale = qb.array(100.0, dtype=qb.float32), qb.array(1.5, dtype=qb.float32)
        expected = qb.jit(collatz_like)(carry, limit, scale)

        def forward(scale):
            return qb.jvp(
                lambda s: collatz_like(carry, limit, s), (scale,), (qb.ones([], qb.float32),)
            )

        assert_close(qb.jit(collatz_like, device=device)(carry, limit, scale), expected, 1e-4)
        cpu_value, cpu_tangent = qb.jit(forward)(scale)
        value, tangent = qb.jit(forward, device=device)(scale)
        assert_close(value, cpu_value, 1e-4)
        assert_close(tangent, cpu_tangent, 1e-3 * abs(cpu_tangent.to_flat_list()[0]))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
