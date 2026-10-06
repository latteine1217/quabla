"""S7 eager, staged, and differentiated control-flow wrapper checks."""

import math

import quabla as qb
from quabla._control import cond, fori_loop, scan, while_loop

from _support import devices, require, run
from _support import raises as assert_raises


def assert_close(actual, expected, tolerance=1e-10):
    expected = qb.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert all(
        abs(a - b) <= tolerance
        for a, b in zip(actual.to_flat_list(), expected.to_flat_list())
    ), (actual.to_flat_list(), expected.to_flat_list())


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
    for device in devices():
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
    require("cuda")
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


def test_while_loop_reverse_mode_is_rejected_clearly():
    def power(scale):
        return while_loop(
            lambda c, s: c < 50.0, lambda c, s: c * s, qb.array(1.0), operands=(scale,)
        )

    for transform in (qb.grad, lambda f: qb.jit(qb.grad(f))):
        error = assert_raises(Exception, transform(power), qb.array(2.0))
        assert "while_loop" in str(error) and "fori_loop" in str(error), str(error)
    # Batched, each example stops at its own trip count: 2^6 and 3^4.
    for transform in (qb.vmap, lambda f: qb.jit(qb.vmap(f))):
        assert_close(transform(power)(qb.array([2.0, 3.0])), [64.0, 81.0])


def test_optional_device_while_loop_parity():
    for device in devices():
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


# ---- vmap of while_loop ----


def halving(carry, limit):
    # Halves the carry until its sum is at most `limit`; a carry already
    # below the limit runs no iteration.
    return while_loop(lambda c, n: c.sum() > n, lambda c, n: c * 0.5, carry, operands=(limit,))


def heron(target, start):
    # Heron's square root. For target 0 from start 0 the loop stops at once,
    # but its body would compute 0 / 0.
    return while_loop(
        lambda c, a: (c * c - a).abs() > 1e-6 * a,
        lambda c, a: 0.5 * (c + a / c),
        start,
        operands=(target,),
    )


def squaring(carry):
    # Squares the carry until it reaches 1e3; for an example that starts
    # above it, the body would overflow to inf.
    return while_loop(lambda c: c < 1e3, lambda c: c * c, carry)


def nested_whiles(carry, limit):
    # Slot 0 counts the outer iterations down; each runs `halving` on the
    # rest, so the inner trip count differs per outer iteration and example.
    def outer(c, n):
        inner = halving(c[1:], n)
        return qb.concat([c[:1] - 1.0, inner + 1.0], 0)

    return while_loop(lambda c, n: c[0] > 0.0, outer, carry, operands=(limit,))


def while_in_fori(carry, limit):
    return fori_loop(0, 3, lambda i, c, n: halving(c + i, n), carry, operands=(limit,))


def fori_in_while(carry, limit):
    return while_loop(
        lambda c, n: c.sum() > n,
        lambda c, n: fori_loop(0, 2, lambda i, d: d * 0.75, c),
        carry,
        operands=(limit,),
    )


def never_runs(carry, flag):
    # The predicate reads only `flag`: unmapped, every example shares one
    # (zero) trip count even though the carry is mapped.
    return while_loop(lambda c, f: f > 0.0, lambda c, f: c * 2.0, carry, operands=(flag,))


def while_arguments(dtype):
    carries = qb.array(
        [[1.0, 2.0, 0.5], [100.0, 3.0, 7.0], [0.1, 0.1, 0.1], [40.0, -1.0, 2.0]], dtype=dtype
    )
    limits = qb.array([0.5, 1.0, 2.0, 1e-3], dtype=dtype)
    return carries, limits


def test_vmap_batches_while_loop_like_a_loop_over_examples():
    for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-6)):
        carries, limits = while_arguments(dtype)
        counted = qb.concat([qb.array([[3.0], [1.0], [0.0], [2.0]], dtype=dtype), carries], 1)
        cases = [
            # A mapped predicate through the carry, the operand, or both.
            (halving, (carries, limits), (0, 0)),
            (halving, (carries, limits[1]), (0, None)),
            (halving, (carries[1], limits), (None, 0)),
            (heron, (limits * 10.0, limits * 10.0), (0, 0)),
            (nested_whiles, (counted, limits), (0, 0)),
            (nested_whiles, (counted, limits[0]), (0, None)),
            (while_in_fori, (carries, limits), (0, 0)),
            (fori_in_while, (carries, limits), (0, 0)),
            (never_runs, (carries, qb.array(0.0, dtype=dtype)), (0, None)),
            # A batch axis that is not leading.
            (halving, (carries.transpose(), limits), (1, 0)),
        ]
        for function, args, in_axes in cases:
            assert_vmap_matches_examples(function, args, in_axes, tolerance)
        # The trip counts do differ per example: slot 0 counts them.
        def counted_halving(c, n):
            return while_loop(
                lambda s, n: s[1:].sum() > n,
                lambda s, n: qb.concat([s[:1] + 1.0, s[1:] * 0.5], 0),
                qb.concat([qb.zeros([1], dtype), c], 0),
                operands=(n,),
            )[0]

        trips = qb.vmap(counted_halving)(carries, limits)
        assert trips.tolist() == [3.0, 7.0, 0.0, 16.0], trips.tolist()
        # Nested vmap: the outer map batches the inner map's while again.
        pairs = qb.stack([carries, carries * 3.0], 0)
        assert_tree_close(
            qb.vmap(qb.vmap(halving, in_axes=(0, None)), in_axes=(0, 0))(pairs, limits[:2]),
            per_example(
                lambda c, n: per_example(halving, (c, n), (0, None)), (pairs, limits[:2]), (0, 0)
            ),
            tolerance,
        )


def test_vmap_of_while_loop_freezes_finished_examples():
    for dtype in (qb.float64, qb.float32):
        targets = qb.array([0.0, 2.0, 1e6, 0.25], dtype=dtype)
        values = qb.vmap(heron)(targets, targets)
        assert values.tolist()[0] == 0.0
        assert_tree_close(values, per_example(heron, (targets, targets), (0, 0)), 0.0)

        # The tangent of the finished example is its initial tangent, not
        # the NaN its masked body computes.
        def root_and_tangent(a):
            return qb.jvp(lambda a: heron(a, a), (a,), (qb.ones([], dtype),))

        pairs = qb.vmap(root_and_tangent)(targets)
        assert_tree_close(pairs, per_example(root_and_tangent, (targets,), (0,)), 0.0)
        assert pairs[1].tolist()[0] == 1.0
        assert all(value == value for value in pairs[1].tolist()), pairs[1].tolist()
        big = 1e200 if dtype == qb.float64 else 1e30
        starts = qb.array([1.5, big, 2.0], dtype=dtype)
        squares = qb.vmap(squaring)(starts)
        assert squares.tolist()[1] == starts.tolist()[1]
        assert_tree_close(squares, per_example(squaring, (starts,), (0,)), 0.0)


def test_vmap_of_while_loop_derivatives_matches_per_example_derivatives():
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        carries = qb.array([[1.0, -2.0], [3.0, 0.5], [0.2, 0.1]], dtype=dtype)
        limits = qb.array([100.0, 30.0, 5e3], dtype=dtype)
        scales = qb.array([1.5, 2.0, 1.1], dtype=dtype)
        cases = [
            # Forward mode with mapped primals and tangents.
            (
                lambda c, n, s, t: qb.jvp(
                    collatz_like, (c, n, s), (qb.zeros_like(c), qb.zeros_like(n), t)
                ),
                (carries, limits, scales, scales * 0.5),
                (0, 0, 0, 0),
            ),
            # Only the tangent mapped, as a forward-mode Jacobian maps it.
            (
                lambda t, c, n, s: qb.jvp(
                    collatz_like, (c, n, s), (t, qb.zeros_like(n), qb.zeros_like(s))
                )[1],
                (carries, carries[0], limits[0], scales[0]),
                (0, None, None, None),
            ),
            (qb.jacobian(collatz_like), (carries, limits, scales), (0, 0, 0)),
            (qb.jacobian(collatz_like, argnums=2), (carries, limits, scales), (0, 0, 0)),
        ]
        for function, args, in_axes in cases:
            assert_vmap_matches_examples(function, args, in_axes, tolerance)
        # The JVP of a vmapped while.
        _, tangent = qb.jvp(
            qb.vmap(collatz_like),
            (carries, limits, scales),
            (qb.zeros_like(carries), qb.zeros_like(limits), qb.ones_like(scales)),
        )
        expected = per_example(
            lambda c, n, s: qb.jvp(
                collatz_like, (c, n, s), (qb.zeros_like(c), qb.zeros_like(n), qb.ones_like(s))
            )[1],
            (carries, limits, scales),
            (0, 0, 0),
        )
        assert_tree_close(tangent, expected, tolerance)

        # A forward-mode Jacobian through a while and a fori loop together.
        def chained(c, s):
            start = collatz_like(c, limits[0], s)
            return fori_loop(0, 3, lambda i, d, s: (d * s).sin(), start, operands=(s,))

        directions = [qb.eye(2, dtype=dtype)[k] for k in range(2)]
        forward = qb.stack(
            [
                qb.jvp(chained, (carries[0], scales[0]), (d, qb.zeros([], dtype)))[1]
                for d in directions
            ],
            1,
        )
        assert_tree_close(qb.jacobian(chained)(carries[0], scales[0]), forward, tolerance)
        assert_tree_close(qb.jit(qb.jacobian(chained))(carries[0], scales[0]), forward, tolerance)


def test_loop_bodies_may_ignore_their_carry():
    # A body whose result does not read the carry is valid, as in JAX: it
    # returns the same value every iteration.
    x, y = qb.array([1.0, 2.0]), qb.array([3.0, -0.0])
    cases = (
        (lambda x, y: fori_loop(0, 3, lambda i, c, y: y * 2.0, x, operands=(y,)), [6.0, -0.0]),
        (lambda x, y: fori_loop(0, 3, lambda i, c, y: y * i, x, operands=(y,)), [6.0, -0.0]),
        (lambda x, y: fori_loop(0, 3, lambda i, c, y: qb.ones([2]), x, operands=(y,)), [1.0, 1.0]),
        (lambda x, y: fori_loop(0, 0, lambda i, c, y: y, x, operands=(y,)), [1.0, 2.0]),
        (
            lambda x, y: scan(lambda c, i, y: (y * 2.0, y), x, length=3, operands=(y,))[0],
            [6.0, -0.0],
        ),
        (
            lambda x, y: while_loop(
                lambda c, y: c[0] < 3.0, lambda c, y: y * 2.0, x, operands=(y,)
            ),
            [6.0, -0.0],
        ),
    )
    for function, expected in cases:
        for transform in (lambda f: f, qb.jit):
            result = transform(function)(x, y).tolist()
            # The signed zero survives: the retained carry never reaches the
            # value.
            assert [str(value) for value in result] == [str(value) for value in expected], result

    # The carry gets a zero derivative and the operand its own.
    def loss(x, y):
        return fori_loop(0, 3, lambda i, c, y: y * y, x, operands=(y,)).sum()

    gx, gy = qb.grad(loss, argnums=(0, 1))(x, qb.array([3.0, 1.0]))
    assert gx.tolist() == [0.0, 0.0] and gy.tolist() == [6.0, 2.0]
    assert_close(
        qb.vmap(lambda x, y: fori_loop(0, 3, lambda i, c, y: y * y, x, operands=(y,)))(
            qb.stack([x, x]), qb.stack([y, 2.0 * y])
        ),
        [[9.0, 0.0], [36.0, 0.0]],
    )


def test_optional_device_vmap_of_while_loop_matches_cpu():
    precisions = {"mlx": (None,), "cuda": (None, "float64")}
    runs = [(device, precision) for device in devices(*precisions) for precision in precisions[device]]
    for device, precision in runs:
        dtype = qb.float64 if precision == "float64" else qb.float32
        tolerance = 1e-11 if precision == "float64" else 1e-5
        carries, limits = while_arguments(dtype)
        counted = qb.concat([qb.array([[3.0], [1.0], [0.0], [2.0]], dtype=dtype), carries], 1)
        targets = qb.array([0.0, 2.0, 1e6, 0.25], dtype=dtype)
        rotating = qb.array(
            [[0.3, -0.2, 0.5, 0.1, -0.4], [0.2, 0.1, -0.3, 0.6, 0.0], [-0.5, 0.4, 0.2, -0.1, 0.3]],
            dtype=dtype,
        )
        rotation_scale = qb.array([0.9, 1.1, 0.8, 1.05, 0.95], dtype=dtype)
        scales = qb.array([1.5, 2.0, 1.1, 1.3], dtype=dtype)
        operations = {
            "halving": (qb.vmap(halving), (carries, limits)),
            "halving capture": (qb.vmap(halving, in_axes=(None, 0)), (carries[1], limits)),
            "heron": (qb.vmap(heron), (targets, targets)),
            "heron jvp": (
                qb.vmap(lambda a: qb.jvp(lambda a: heron(a, a), (a,), (qb.ones([], dtype),))),
                (targets,),
            ),
            "nested": (qb.vmap(nested_whiles), (counted, limits)),
            "while in fori": (qb.vmap(while_in_fori), (carries, limits)),
            "rotating": (qb.vmap(host_driven_while, in_axes=(0, None)), (rotating, rotation_scale)),
            "jacobian": (
                qb.vmap(qb.jacobian(collatz_like, argnums=2)),
                (carries[:, :2], limits * 1e4 + 50.0, scales),
            ),
        }
        for name, (operation, args) in operations.items():
            expected = qb.jit(operation)(*args)
            options = {"device": device}
            if precision is not None:
                options["precision"] = precision
            actual = qb.jit(operation, **options)(*args)
            assert_tree_close(actual, expected, tolerance)


# ---- vmap of fori_loop and scan regions ----


def per_example(function, args, in_axes):
    """The meaning of `vmap`: a Python loop over the batch of unbatched
    calls, stacked on a leading axis."""
    batch = next(arg.shape[axis] for arg, axis in zip(args, in_axes) if axis is not None)
    examples = [arg if axis is None else qb.moveaxis(arg, axis, 0) for arg, axis in zip(args, in_axes)]
    results = [
        function(*[arg if axis is None else arg[index] for arg, axis in zip(examples, in_axes)])
        for index in range(batch)
    ]
    return qb.tree.map(lambda *leaves: qb.stack(list(leaves), 0), *results)


def assert_tree_close(actual, expected, tolerance):
    actual, expected = qb.tree.leaves(actual), qb.tree.leaves(expected)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype, (got.dtype, want.dtype)
        scale = max([1.0] + [abs(value) for value in want.to_flat_list()])
        assert_close(got, want, tolerance * scale)


def assert_vmap_matches_examples(function, args, in_axes, tolerance):
    expected = per_example(function, args, in_axes)
    assert_tree_close(qb.vmap(function, in_axes=in_axes)(*args), expected, tolerance)
    # Loops inside and around jit batch the same way.
    assert_tree_close(qb.jit(qb.vmap(function, in_axes=in_axes))(*args), expected, tolerance)
    assert_tree_close(qb.vmap(qb.jit(function), in_axes=in_axes)(*args), expected, tolerance)


def batched_loop_arguments(dtype):
    carries = qb.array(
        [[0.3, -0.2, 0.5], [0.1, 0.4, -0.6], [1.0, 0.2, 0.0], [-0.5, 0.9, 0.3]], dtype=dtype
    )
    scales = qb.array(
        [[0.9, 1.1, 0.8], [1.05, 0.95, 0.7], [0.5, 0.6, 1.2], [1.3, 0.8, 0.9]], dtype=dtype
    )
    rows = qb.array(
        [[[0.1 * b + 0.03 * t - 0.02 * k for k in range(3)] for t in range(5)] for b in range(4)],
        dtype=dtype,
    )
    return carries, scales, rows


def mapped_fori(carry, scale):
    return fori_loop(
        0, 5, lambda i, c, s: (c * s).sin() + 0.1 * i, carry, operands=(scale,)
    )


def mapped_scan(carry, scale, rows):
    # The per-step input is row i of `rows`, selected with the traced index
    # as jax.lax.scan over xs would.
    steps = rows.shape[0]

    def body(c, i, s, rows):
        mask = qb.arange(steps).astype(rows.dtype).equal(i).astype(rows.dtype)
        row = (rows * mask.reshape([steps, 1])).sum(0)
        nxt = (c * s + row).tanh()
        return nxt, (nxt * row)[:2]

    return scan(body, carry, length=steps, operands=(scale, rows))


def nested_loops(carry, scale):
    # A scan whose body runs a fori_loop over the outer carry.
    def body(c, i, s):
        inner = fori_loop(0, 2, lambda j, d, s: (d * s).cos() + 0.1 * d, c, operands=(s,))
        return inner, inner.sum()

    return scan(body, carry, length=3, operands=(scale,))


def test_vmap_batches_fori_loop_and_scan_like_a_loop_over_examples():
    for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-6)):
        carries, scales, rows = batched_loop_arguments(dtype)
        cases = [
            (mapped_fori, (carries, scales), (0, 0)),
            # A mapped carry with an unmapped capture, and the reverse: the
            # carry depends on the mapped capture, so it is mapped too.
            (mapped_fori, (carries, scales[0]), (0, None)),
            (mapped_fori, (carries[0], scales), (None, 0)),
            (mapped_scan, (carries, scales, rows), (0, 0, 0)),
            (mapped_scan, (carries, scales[0], rows[0]), (0, None, None)),
            (mapped_scan, (carries[0], scales, rows[0]), (None, 0, None)),
            # Mapped per-step inputs (scan over xs).
            (mapped_scan, (carries[0], scales[0], rows), (None, None, 0)),
            (nested_loops, (carries, scales), (0, 0)),
            (nested_loops, (carries[0], scales), (None, 0)),
            # A batch axis that is not leading.
            (mapped_fori, (carries.transpose(), scales[0]), (1, None)),
        ]
        for function, args, in_axes in cases:
            assert_vmap_matches_examples(function, args, in_axes, tolerance)
        # Nested vmap: the outer map batches the inner map's loop regions again.
        pairs = qb.stack([carries, carries * 0.5], 0)
        assert_tree_close(
            qb.vmap(qb.vmap(mapped_fori, in_axes=(0, None)), in_axes=(0, 0))(pairs, scales[:2]),
            per_example(
                lambda c, s: per_example(mapped_fori, (c, s), (0, None)), (pairs, scales[:2]), (0, 0)
            ),
            tolerance,
        )


def loop_losses():
    def fori_loss(carry, scale):
        return (mapped_fori(carry, scale) ** 2).sum()

    def scan_loss(carry, scale, rows):
        final, outputs = mapped_scan(carry, scale, rows)
        return (final**2).sum() + (outputs**3).sum()

    def nested_loss(carry, scale):
        final, outputs = nested_loops(carry, scale)
        return (final**2).sum() + outputs.sum()

    return fori_loss, scan_loss, nested_loss


def test_vmap_of_loop_gradients_and_jvps_matches_per_example_derivatives():
    fori_loss, scan_loss, nested_loss = loop_losses()
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        carries, scales, rows = batched_loop_arguments(dtype)
        cases = [
            (qb.grad(fori_loss, argnums=(0, 1)), (carries, scales), (0, 0)),
            (qb.grad(fori_loss, argnums=(0, 1)), (carries, scales[0]), (0, None)),
            (qb.grad(fori_loss, argnums=(0, 1)), (carries[0], scales), (None, 0)),
            (qb.grad(scan_loss, argnums=(0, 1, 2)), (carries, scales, rows), (0, 0, 0)),
            (qb.grad(scan_loss, argnums=(0, 1, 2)), (carries[0], scales[0], rows), (None, None, 0)),
            (qb.grad(nested_loss, argnums=(0, 1)), (carries, scales[0]), (0, None)),
            (
                lambda c, s, t: qb.jvp(fori_loss, (c, s), (t, qb.zeros([3], dtype))),
                (carries, scales, scales * 0.5),
                (0, 0, 0),
            ),
            # Only the tangent is mapped, as a forward-mode Jacobian maps it.
            (
                lambda t, c, s, r: qb.jvp(scan_loss, (c, s, r), (t, qb.zeros([3], dtype), qb.zeros([5, 3], dtype))),
                (carries, carries[0], scales[0], rows[0]),
                (0, None, None, None),
            ),
            (
                lambda t, c, s: qb.jvp(qb.grad(nested_loss), (c, s), (t, qb.zeros([3], dtype)))[1],
                (carries, carries[0], scales[0]),
                (0, None, None),
            ),
        ]
        for function, args, in_axes in cases:
            assert_vmap_matches_examples(function, args, in_axes, tolerance)


def basis(size, dtype):
    return [qb.eye(size, dtype=dtype)[index] for index in range(size)]


def test_hessian_and_jacobian_through_loops_match_per_direction_calls():
    fori_loss, scan_loss, nested_loss = loop_losses()
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        carries, scales, rows = batched_loop_arguments(dtype)
        carry, scale, row = carries[0], scales[0], rows[0]
        directions = basis(3, dtype)
        for loss, extra in ((fori_loss, ()), (scan_loss, (row,)), (nested_loss, ())):
            for argnums in (0, 1):
                arguments = (carry, scale, *extra)
                # Row j of the Hessian is the HVP along basis vector j, one
                # unbatched forward-over-reverse call per direction.
                expected = qb.stack(
                    [
                        qb.jvp(
                            qb.grad(loss, argnums=argnums),
                            arguments,
                            tuple(
                                direction if index == argnums else qb.zeros_like(value)
                                for index, value in enumerate(arguments)
                            ),
                        )[1]
                        for direction in directions
                    ],
                    0,
                )
                hessian = qb.hessian(loss, argnums=argnums)
                assert_tree_close(hessian(*arguments), expected, tolerance)
                assert_tree_close(qb.jit(hessian)(*arguments), expected, tolerance)
        # Forward-mode Jacobian (more outputs than inputs): columns are JVPs.
        def outputs_of(scale):
            return mapped_scan(carry, scale, row)[1].reshape([10])

        forward = qb.stack(
            [qb.jvp(outputs_of, (scale,), (direction,))[1] for direction in directions], 1
        )
        assert_tree_close(qb.jacobian(outputs_of)(scale), forward, tolerance)
        # Reverse-mode Jacobian (fewer outputs than inputs): rows are VJPs.
        def summary(rows):
            final, outputs = mapped_scan(carry, scale, rows)
            return qb.stack([final.sum(), (outputs**2).sum()], 0)

        _, pullback = qb.vjp(summary, row)
        reverse = qb.stack([pullback(direction)[0] for direction in basis(2, dtype)], 0)
        assert_tree_close(qb.jacobian(summary)(row), reverse, tolerance)
        assert_tree_close(qb.jacobian(mapped_fori)(carry, scale), qb.stack(
            [qb.jvp(mapped_fori, (carry, scale), (direction, qb.zeros([3], dtype)))[1] for direction in directions], 1
        ), tolerance)


def test_hessian_through_a_linear_loop_matches_the_closed_form():
    # c <- c * s five times gives x * s**5, so the loss sum((x * s**5)**2)
    # has the Hessian diag(2 * s**10) in x and diag(90 * x**2 * s**8) in s.
    def loss(x, s):
        return (fori_loop(0, 5, lambda i, c, s: c * s, x, operands=(s,)) ** 2).sum()

    x, s = qb.array([0.5, -1.5, 2.0]), qb.array([1.1, 0.9, -0.7])
    xs, ss = x.to_flat_list(), s.to_flat_list()
    expected_x = [[2 * ss[i] ** 10 if i == j else 0.0 for j in range(3)] for i in range(3)]
    expected_s = [
        [90 * xs[i] ** 2 * ss[i] ** 8 if i == j else 0.0 for j in range(3)] for i in range(3)
    ]
    assert_close(qb.hessian(loss)(x, s), expected_x, 1e-12)
    assert_close(qb.hessian(loss, argnums=1)(x, s), expected_s, 1e-11)


# ---- vmap of cond regions ----


def batched_cond_arguments(dtype):
    # Row sums 0.6, -0.9, 1.2, 0.7 and flags 1, -1, 0.5, -0.2: data and flag
    # predicates take both branches within the batch.
    flags = qb.array([1.0, -1.0, 0.5, -0.2], dtype=dtype)
    xs = qb.array(
        [[0.3, -0.2, 0.5], [-0.4, 0.1, -0.6], [1.0, 0.2, 0.0], [-0.5, 0.9, 0.3]], dtype=dtype
    )
    ws = qb.array(
        [[0.9, 1.1, 0.8], [1.05, 0.95, 0.7], [0.5, 0.6, 1.2], [1.3, 0.8, 0.9]], dtype=dtype
    )
    return flags, xs, ws


def piecewise(x, w):
    # The predicate depends on the mapped data.
    return cond(x.sum() > 0.0, lambda v, w: (v * w).sin() * 2.0, lambda v, w: v * v - w, x, w)


def gated(flag, x, w):
    # The predicate is its own argument, mapped or not independently of the
    # operands.
    return cond(flag > 0.0, lambda v, w: (v * w).tanh(), lambda v, w: v.cos() + w, x, w)


def cond_with_loops(flag, x, w):
    def scanned(v, w):
        final, outputs = scan(lambda c, i, w: (c * w + 0.1, (c * w).sum()), v, length=2, operands=(w,))
        return final + outputs.sum()

    return cond(
        flag > 0.0,
        lambda v, w: fori_loop(0, 3, lambda i, c, w: (c * w).sin() + 0.1 * i, v, operands=(w,)),
        scanned,
        x,
        w,
    )


def scan_carry_cond(flag, x, w):
    # w reaches only the discarded scan outputs, so under an unmapped flag
    # and a mapped w the true branch alone would be unmapped and the false
    # branch mapped: the true branch is broadcast.
    return cond(
        flag > 0.0,
        lambda v, w: scan(lambda c, i, w: (c * 0.9 + 0.1, c * w), v, length=2, operands=(w,))[0],
        lambda v, w: v * w,
        x,
        w,
    )


def loops_with_conds(x, w):
    # A cond on the loop index (never mapped) and one on the carry (mapped
    # with the carry) inside a fori_loop body, and a cond inside a scan body.
    def body(i, c, w):
        stepped = cond(i < 1, lambda c, w: c * w, lambda c, w: c.sin() + w, c, w)
        return cond(stepped.sum() > 0.0, lambda s: s * 0.9, lambda s: s.cos(), stepped)

    def step(c, i, w):
        nxt = cond(c.sum() > 0.5, lambda c, w: c * w, lambda c, w: c + 0.1 * w, c, w)
        return nxt, nxt.sum()

    looped = fori_loop(0, 3, body, x, operands=(w,))
    final, outputs = scan(step, looped, length=3, operands=(w,))
    return final, outputs


def test_vmap_batches_cond_like_a_loop_over_examples():
    for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-6)):
        flags, xs, ws = batched_cond_arguments(dtype)
        flag, x, w = flags[0], xs[0], ws[0]
        cases = [
            # Mapped data predicate.
            (piecewise, (xs, ws), (0, 0)),
            (piecewise, (xs, w), (0, None)),
            # Unmapped data predicate with a mapped operand.
            (piecewise, (x, ws), (None, 0)),
            # A batch axis that is not leading.
            (piecewise, (xs.transpose(), w), (1, None)),
        ]
        # Every mapping of the flag and the two operands.
        for flag_axis in (0, None):
            for x_axis in (0, None):
                for w_axis in (0, None):
                    if flag_axis is x_axis is w_axis is None:
                        continue
                    args = tuple(
                        batched if axis == 0 else batched[0]
                        for batched, axis in ((flags, flag_axis), (xs, x_axis), (ws, w_axis))
                    )
                    cases.append((gated, args, (flag_axis, x_axis, w_axis)))
        cases += [
            (cond_with_loops, (flags, xs, w), (0, 0, None)),
            (cond_with_loops, (flag, x, ws), (None, None, 0)),
            (cond_with_loops, (flags, x, w), (0, None, None)),
            (scan_carry_cond, (flag, x, ws), (None, None, 0)),
            (scan_carry_cond, (flags, x, ws), (0, None, 0)),
            (loops_with_conds, (xs, w), (0, None)),
            (loops_with_conds, (x, ws), (None, 0)),
        ]
        for function, args, in_axes in cases:
            assert_vmap_matches_examples(function, args, in_axes, tolerance)
        # Nested vmap: the inner map batches the flag, the outer one the data.
        pairs = qb.stack([xs, xs * -0.5], 0)
        inner = qb.vmap(gated, in_axes=(0, 0, None))
        assert_tree_close(
            qb.vmap(inner, in_axes=(None, 0, None))(flags, pairs, w),
            per_example(lambda p: per_example(gated, (flags, p, w), (0, 0, None)), (pairs,), (0,)),
            tolerance,
        )


def cond_losses():
    def piecewise_loss(x, w):
        return (piecewise(x, w) ** 2).sum()

    def gated_loss(flag, x, w):
        return (gated(flag, x, w) ** 2).sum() + gated(flag, x, w).sum()

    def loops_loss(x, w):
        final, outputs = loops_with_conds(x, w)
        return (final**2).sum() + outputs.sum()

    def cond_loops_loss(flag, x, w):
        return (cond_with_loops(flag, x, w) ** 2).sum()

    return piecewise_loss, gated_loss, loops_loss, cond_loops_loss


def test_vmap_of_cond_gradients_and_jvps_matches_per_example_derivatives():
    piecewise_loss, gated_loss, loops_loss, cond_loops_loss = cond_losses()
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        flags, xs, ws = batched_cond_arguments(dtype)
        flag, x, w = flags[0], xs[0], ws[0]
        zeros = qb.zeros([3], dtype)
        cases = [
            (qb.grad(piecewise_loss, argnums=(0, 1)), (xs, ws), (0, 0)),
            (qb.grad(piecewise_loss, argnums=(0, 1)), (x, ws), (None, 0)),
            (qb.grad(gated_loss, argnums=(1, 2)), (flags, xs, w), (0, 0, None)),
            (qb.grad(gated_loss, argnums=(1, 2)), (flags, x, w), (0, None, None)),
            (qb.grad(gated_loss, argnums=(1, 2)), (flag, xs, w), (None, 0, None)),
            (qb.grad(loops_loss, argnums=(0, 1)), (xs, w), (0, None)),
            (qb.grad(cond_loops_loss, argnums=(1, 2)), (flags, xs, ws), (0, 0, 0)),
            (
                lambda f, a, b, t: qb.jvp(gated_loss, (f, a, b), (qb.zeros([], dtype), t, zeros)),
                (flags, xs, w, ws),
                (0, 0, None, 0),
            ),
            # Only the tangent is mapped, as a forward-mode Jacobian maps it.
            (
                lambda t, a, b: qb.jvp(piecewise_loss, (a, b), (t, zeros)),
                (xs, x, w),
                (0, None, None),
            ),
            (
                lambda t, a, b: qb.jvp(qb.grad(loops_loss), (a, b), (t, zeros))[1],
                (xs, x, w),
                (0, None, None),
            ),
        ]
        for function, args, in_axes in cases:
            assert_vmap_matches_examples(function, args, in_axes, tolerance)
        # Reverse mode over vmap: the mapped x gets one gradient per example,
        # and the shared w the sum of the examples' gradients.
        def batched_loss(x, w):
            return qb.vmap(gated_loss, in_axes=(0, 0, None))(flags, x, w).sum()

        gradients = [qb.grad(gated_loss, argnums=(1, 2))(flags[b], xs[b], w) for b in range(4)]
        expected_x = qb.stack([gradient[0] for gradient in gradients], 0)
        expected_w = sum((gradient[1] for gradient in gradients[1:]), gradients[0][1])
        assert_tree_close(
            qb.grad(batched_loss, argnums=(0, 1))(xs, w), (expected_x, expected_w), tolerance
        )
        assert_tree_close(
            qb.jit(qb.grad(batched_loss, argnums=(0, 1)))(xs, w),
            (expected_x, expected_w),
            tolerance,
        )


def test_hessian_and_jacobian_through_cond_match_per_direction_calls():
    piecewise_loss, gated_loss, loops_loss, cond_loops_loss = cond_losses()
    for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
        flags, xs, ws = batched_cond_arguments(dtype)
        directions = basis(3, dtype)
        losses = [
            (piecewise_loss, (xs[0], ws[0])),
            (piecewise_loss, (xs[1], ws[1])),
            (lambda x, w: gated_loss(flags[1], x, w), (xs[1], ws[1])),
            (loops_loss, (xs[0], ws[0])),
            (lambda x, w: cond_loops_loss(flags[0], x, w), (xs[0], ws[0])),
            (lambda x, w: cond_loops_loss(flags[1], x, w), (xs[0], ws[0])),
        ]
        for loss, arguments in losses:
            for argnums in (0, 1):
                expected = qb.stack(
                    [
                        qb.jvp(
                            qb.grad(loss, argnums=argnums),
                            arguments,
                            tuple(
                                direction if index == argnums else qb.zeros_like(value)
                                for index, value in enumerate(arguments)
                            ),
                        )[1]
                        for direction in directions
                    ],
                    0,
                )
                hessian = qb.hessian(loss, argnums=argnums)
                assert_tree_close(hessian(*arguments), expected, tolerance)
                assert_tree_close(qb.jit(hessian)(*arguments), expected, tolerance)

        # Hessian of a loss over a vmap whose predicate is mapped: the sum of
        # the examples' Hessians.
        def batched_loss(w):
            return qb.vmap(gated_loss, in_axes=(0, 0, None))(flags, xs, w).sum()

        expected = qb.hessian(lambda w: gated_loss(flags[0], xs[0], w))(ws[0])
        for b in range(1, 4):
            expected = expected + qb.hessian(lambda w, b=b: gated_loss(flags[b], xs[b], w))(ws[0])
        assert_tree_close(qb.hessian(batched_loss)(ws[0]), expected, tolerance)

        # Forward-mode Jacobian (as many outputs as inputs): columns are JVPs.
        for flag in (flags[0], flags[1]):
            def outputs_of(w, flag=flag):
                return gated(flag, xs[0], w)

            forward = qb.stack(
                [qb.jvp(outputs_of, (ws[0],), (direction,))[1] for direction in directions], 1
            )
            assert_tree_close(qb.jacobian(outputs_of)(ws[0]), forward, tolerance)

        # Reverse-mode Jacobian (fewer outputs than inputs): rows are VJPs.
        def summary(x):
            final, outputs = loops_with_conds(x, ws[0])
            return qb.stack([final.sum(), (outputs**2).sum()], 0)

        _, pullback = qb.vjp(summary, xs[0])
        reverse = qb.stack([pullback(direction)[0] for direction in basis(2, dtype)], 0)
        assert_tree_close(qb.jacobian(summary)(xs[0]), reverse, tolerance)


def scan_final_carry(v, w):
    # Keeps the final carry v w^2 and discards the stacked outputs [v, v w].
    return scan(lambda c, i, w: (c * w[0], c), v, length=2, operands=(w,))[0]


def scan_output_sum(v, w):
    # Keeps the outputs [v w, v w^2], summed to v (w + w^2), and discards the
    # final carry.
    return scan(lambda c, i, w: (c * w[0], c * w[0]), v, length=2, operands=(w,))[1].sum(0)


def discarding_scan_losses(branch):
    """Squared losses around a scan region that drops one of its results."""

    def in_cond(v, w):
        return (cond(v.sum() > 0, branch, lambda v, w: v * w[0], v, w) ** 2).sum()

    def at_top(v, w):
        return (branch(v, w) ** 2).sum()

    def in_fori(v, w):
        looped = fori_loop(0, 2, lambda i, c, w: branch(c, w), v, operands=(w,))
        return (looped**2).sum()

    return {"cond": in_cond, "top": at_top, "fori": in_fori}


def discarding_scan_closed_form(keep_outputs, kind, v, w):
    """(loss, dloss/dv, dloss/dw, d2loss/dv2, d2loss/dw2) for scalar v and w."""
    # The kept scan result is g = v s(w); the cond's false branch is v w.
    s, ds, d2s = (w + w * w, 1 + 2 * w, 2.0) if keep_outputs else (w * w, 2 * w, 2.0)
    if kind == "cond" and v <= 0:
        s, ds, d2s = w, 1.0, 0.0
    if kind == "fori":
        # Two applications: v s^2.
        s, ds, d2s = s * s, 2 * s * ds, 2 * (ds * ds + s * d2s)
    g, g_v, g_w, g_ww = v * s, s, v * ds, v * d2s
    return g * g, 2 * g * g_v, 2 * g * g_w, 2 * g_v * g_v, 2 * (g_w * g_w + g * g_ww)


def test_reverse_mode_through_a_scan_that_discards_a_result():
    # Regression: a scan region whose final carry or stacked outputs were
    # unused (and so removed from the traced region) failed reverse mode
    # with "symbolic Scan group ... has no output result".
    def f(v):
        return cond(v.sum() > 0, scan_final_carry, lambda v, w: v * 2.0, v, qb.array([3.0])).sum()

    for call in (qb.grad(f), qb.jit(qb.grad(f))):
        assert_close(call(qb.array([1.0])), [9.0], 0.0)
        assert_close(call(qb.array([-1.0])), [2.0], 0.0)

    for keep_outputs, branch in ((False, scan_final_carry), (True, scan_output_sum)):
        for kind, loss in discarding_scan_losses(branch).items():
            for dtype, tolerance in ((qb.float64, 1e-12), (qb.float32, 1e-5)):
                for v0 in (0.5, -1.0):
                    w0 = 1.5
                    value, d_v, d_w, h_v, h_w = discarding_scan_closed_form(
                        keep_outputs, kind, v0, w0
                    )
                    v = qb.array([v0], dtype=dtype)
                    w = qb.array([w0], dtype=dtype)
                    scale = max(abs(value), abs(d_v), abs(d_w), abs(h_v), abs(h_w), 1.0)
                    tol = tolerance * scale
                    gradient = qb.value_and_grad(loss, argnums=(0, 1))
                    for call in (gradient, qb.jit(gradient)):
                        actual, (g_v, g_w) = call(v, w)
                        assert_close(actual, value, tol)
                        assert_close(g_v, [d_v], tol)
                        assert_close(g_w, [d_w], tol)
                    for argnums, expected in ((0, h_v), (1, h_w)):
                        hessian = qb.hessian(loss, argnums=argnums)
                        assert_close(hessian(v, w), [[expected]], tol)
                        assert_close(qb.jit(hessian)(v, w), [[expected]], tol)
                    # Forward-over-reverse directly: the JVP of the gradient.
                    _, h_vw = qb.jvp(
                        qb.grad(loss, argnums=1), (v, w), (qb.zeros_like(v), qb.ones_like(w))
                    )
                    assert_close(h_vw, [h_w], tol)
                    if dtype == qb.float32:
                        continue
                    # Central differences of the eager CPU primal; the step
                    # never crosses the cond predicate at v = 0.
                    step = 1e-6
                    for index in (0, 1):
                        shift = [qb.array([0.0]), qb.array([0.0])]
                        shift[index] = qb.array([step])
                        upper = loss(v + shift[0], w + shift[1])
                        lower = loss(v - shift[0], w - shift[1])
                        fd = (upper - lower) / (2 * step)
                        assert_close(fd, (d_v, d_w)[index], 1e-6 * scale)


def test_optional_device_discarded_scan_result_parity():
    for device in devices():
        w = qb.array([1.5], dtype=qb.float32)
        vs = (qb.array([0.5], dtype=qb.float32), qb.array([-1.0], dtype=qb.float32))
        for branch in (scan_final_carry, scan_output_sum):
            for kind, loss in discarding_scan_losses(branch).items():
                operations = (
                    qb.value_and_grad(loss, argnums=(0, 1)),
                    qb.hessian(loss, argnums=0),
                    qb.hessian(loss, argnums=1),
                )
                for operation in operations:
                    for v in vs:
                        cpu = qb.jit(operation)(v, w)
                        actual = qb.jit(operation, device=device)(v, w)
                        cpu_leaves, _ = qb.tree.flatten(cpu)
                        actual_leaves, _ = qb.tree.flatten(actual)
                        for value, expected in zip(actual_leaves, cpu_leaves):
                            scale = max(1.0, abs(expected.item()))
                            assert_close(value, expected, 1e-4 * scale)


def safe_log(x):
    # log on the positive side, sqrt(1 - x) elsewhere. Each branch is NaN or
    # infinite on the other side: log(0) = -inf with derivative inf,
    # log(-1) = NaN, and sqrt(1 - 2) = NaN with a NaN derivative.
    return cond(x > 0.0, lambda v: qb.log(v), lambda v: qb.sqrt(1.0 - v), x)


def test_vmap_of_cond_keeps_the_unselected_branch_out_of_values_and_derivatives():
    for dtype, tolerance in ((qb.float64, 1e-14), (qb.float32, 1e-6)):
        points = qb.array([2.0, 0.0, -1.0, 0.5], dtype=dtype)
        value = [math.log(2.0), 1.0, math.sqrt(2.0), math.log(0.5)]
        first = [0.5, -0.5, -0.5 / math.sqrt(2.0), 2.0]
        second = [-0.25, -0.25, -0.25 / 2.0**1.5, -4.0]
        diagonal = [[first[i] if i == j else 0.0 for j in range(4)] for i in range(4)]
        hessian = [[second[i] if i == j else 0.0 for j in range(4)] for i in range(4)]
        batched = qb.vmap(safe_log)

        def total(x):
            return batched(x).sum()

        results = [
            (batched(points), value),
            (qb.jit(batched)(points), value),
            (qb.vmap(qb.grad(safe_log))(points), first),
            # Reverse mode through the select: the where VJP gives the
            # unselected branch a zero cotangent, and its 0 * inf or 0 * NaN
            # derivative is masked instead of added.
            (qb.grad(total)(points), first),
            (qb.jit(qb.grad(total))(points), first),
            (qb.jvp(batched, (points,), (qb.ones_like(points),))[1], first),
            (qb.jacobian(batched)(points), diagonal),
            (qb.jacobian(lambda x: batched(x)[:2])(points), [row for row in diagonal[:2]]),
            (qb.hessian(total)(points), hessian),
            (qb.vmap(qb.grad(qb.grad(safe_log)))(points), second),
        ]
        for actual, expected in results:
            assert actual.dtype == dtype, actual.dtype
            assert all(math.isfinite(item) for item in actual.to_flat_list()), actual
            assert_close(actual, qb.asarray(expected, dtype=dtype), tolerance * 4)


def test_vmap_of_cond_with_a_float_predicate_selects_nonzero_examples():
    def scaled(flag, x):
        return cond(flag, lambda v: v * 2.0, lambda v: -v, x)

    flags = qb.array([2.0, 0.0, -0.0, -3.0])
    x = qb.array([1.0, 2.0, 3.0, 4.0])
    assert_close(qb.vmap(scaled)(flags, x), [2.0, -2.0, -3.0, 8.0], 0.0)
    # The reference is the traced cond (an eager float array is always
    # truthy, so an eager cond would take the true branch for zero).
    assert_close(qb.vmap(scaled)(flags, x), per_example(qb.jit(scaled), (flags, x), (0, 0)), 0.0)
    # An unbatched cond rejects a non-finite predicate; a batched one cannot
    # raise per example and returns NaN for it.
    assert_raises(ValueError, qb.jit(scaled), qb.asarray(math.nan), x[0], match="finite")
    result = qb.vmap(scaled)(qb.array([1.0, math.nan, math.inf]), x[:3]).to_flat_list()
    assert result[0] == 2.0 and math.isnan(result[1]) and math.isnan(result[2]), result


def test_optional_device_vmap_of_cond_matches_cpu():
    piecewise_loss, gated_loss, loops_loss, _ = cond_losses()
    precisions = {"mlx": (None,), "cuda": (None, "float64")}
    runs = [(device, precision) for device in devices(*precisions) for precision in precisions[device]]
    for device, precision in runs:
        dtype = qb.float64 if precision == "float64" else qb.float32
        tolerance = 1e-11 if precision == "float64" else 1e-5
        flags, xs, ws = batched_cond_arguments(dtype)
        points = qb.array([2.0, 0.0, -1.0, 0.5], dtype=dtype)
        operations = {
            # A mapped predicate: a select, no predicate readback.
            "vmap mapped flag": (qb.vmap(gated), (flags, xs, ws)),
            "vmap mapped data": (qb.vmap(piecewise, in_axes=(0, None)), (xs, ws[0])),
            # An unmapped predicate: one cond with both regions batched.
            "vmap unmapped flag": (qb.vmap(gated, in_axes=(None, 0, None)), (flags[1], xs, ws[0])),
            "vmap grad": (qb.vmap(qb.grad(piecewise_loss, argnums=(0, 1))), (xs, ws)),
            "grad over vmap": (
                qb.grad(lambda x, w: qb.vmap(gated_loss, in_axes=(0, 0, None))(flags, x, w).sum(), argnums=(0, 1)),
                (xs, ws[0]),
            ),
            "hessian": (qb.hessian(piecewise_loss), (xs[0], ws[0])),
            "hessian unselected nan": (qb.hessian(lambda x: qb.vmap(safe_log)(x).sum()), (points,)),
            "grad unselected nan": (qb.grad(lambda x: qb.vmap(safe_log)(x).sum()), (points,)),
            "vmap loops with conds": (qb.vmap(loops_with_conds, in_axes=(0, None)), (xs, ws[0])),
            "jacobian": (qb.jacobian(lambda w: gated(flags[0], xs[0], w)), (ws[0],)),
        }
        for name, (operation, args) in operations.items():
            expected = qb.jit(operation)(*args)
            options = {"device": device}
            if precision is not None:
                options["precision"] = precision
            actual = qb.jit(operation, **options)(*args)
            try:
                assert_tree_close(actual, expected, tolerance)
            except AssertionError as error:
                raise AssertionError(f"{device} {precision} {name}") from error


def test_optional_device_vmap_of_loops_matches_cpu():
    fori_loss, scan_loss, _ = loop_losses()
    precisions = {"mlx": (None,), "cuda": (None, "float64")}
    runs = [(device, precision) for device in devices(*precisions) for precision in precisions[device]]
    for device, precision in runs:
        dtype = qb.float64 if precision == "float64" else qb.float32
        tolerance = 1e-11 if precision == "float64" else 1e-5
        carries, scales, rows = batched_loop_arguments(dtype)
        rotating = qb.array(
            [[0.3, -0.2, 0.5, 0.1, -0.4], [0.2, 0.1, -0.3, 0.6, 0.0], [-0.5, 0.4, 0.2, -0.1, 0.3]],
            dtype=dtype,
        )
        rotation_scale = qb.array([0.9, 1.1, 0.8, 1.05, 0.95], dtype=dtype)
        operations = {
            "vmap fori": (qb.vmap(mapped_fori, in_axes=(0, None)), (carries, scales[0])),
            "vmap fori capture": (qb.vmap(mapped_fori, in_axes=(None, 0)), (carries[0], scales)),
            # Not elementwise: CUDA runs these loops host-driven.
            "vmap rotating fori": (
                qb.vmap(host_driven_fori_loss, in_axes=(0, None)),
                (rotating, rotation_scale),
            ),
            "vmap scan xs": (qb.vmap(mapped_scan, in_axes=(None, None, 0)), (carries[0], scales[0], rows)),
            "vmap nested": (qb.vmap(nested_loops), (carries, scales)),
            "vmap grad": (qb.vmap(qb.grad(scan_loss, argnums=(0, 1, 2))), (carries, scales, rows)),
            "hessian fori": (qb.hessian(fori_loss), (carries[0], scales[0])),
            "hessian scan": (qb.hessian(scan_loss, argnums=1), (carries[0], scales[0], rows[0])),
            "hessian rotating scan": (
                qb.hessian(host_driven_scan_loss),
                (rotating[0], rotation_scale),
            ),
            "jacobian scan": (qb.jacobian(lambda s: mapped_scan(carries[0], s, rows[0])[1].reshape([10])), (scales[0],)),
        }
        for name, (operation, args) in operations.items():
            expected = qb.jit(operation)(*args)
            options = {"device": device}
            if precision is not None:
                options["precision"] = precision
            actual = qb.jit(operation, **options)(*args)
            assert_tree_close(actual, expected, tolerance)


def test_eager_cond_predicate_matches_the_traced_rule():
    # A floating predicate selects the true branch when nonzero, eagerly as
    # under jit; it used to be decided by `bool(Tensor)`, which is always true
    # for a floating Tensor, so an eager 0.0 took the true branch.
    def f(p, x):
        return qb.cond(p, lambda v: v + 1.0, lambda v: v - 1.0, x)

    x = qb.array(0.0)
    for p in (0.0, -0.0, 2.0, -3.0):
        for dtype in (qb.float32, qb.float64):
            pred = qb.array(p, dtype=dtype)
            assert f(pred, x).item() == qb.jit(f)(pred, x).item() == (1.0 if p != 0 else -1.0)
    for bad in (float("nan"), float("inf")):
        for call in (f, qb.jit(f)):
            try:
                call(qb.array(bad), x)
            except ValueError as error:
                assert "finite" in str(error)
            else:
                raise AssertionError(f"a {bad} predicate was accepted")
    try:
        f(qb.array([1.0, 0.0]), x)
    except ValueError as error:
        assert "scalar" in str(error)
    else:
        raise AssertionError("a vector predicate was accepted")


if __name__ == "__main__":
    run(globals())
