"""Custom differentiation rules: custom_vjp, custom_jvp, and checkpoint."""

import math
import os

import quabla as qb


def raises(kind, function, *args, match=None):
    try:
        function(*args)
    except kind as error:
        if match is not None:
            assert match in str(error), str(error)
        return
    raise AssertionError(f"expected {kind.__name__}")


def assert_close(actual, expected, tolerance=1e-12):
    actual = actual.tolist() if hasattr(actual, "tolist") else actual
    expected = expected.tolist() if hasattr(expected, "tolist") else expected
    if isinstance(actual, list):
        assert len(actual) == len(expected), (actual, expected)
        for got, want in zip(actual, expected):
            assert_close(got, want, tolerance)
        return
    assert abs(actual - expected) <= tolerance * max(1.0, abs(expected)), (actual, expected)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# -- custom_vjp ----------------------------------------------------------------


@qb.custom_vjp
def log1pexp(x):
    return qb.log(1.0 + qb.exp(x))


def log1pexp_fwd(x):
    return log1pexp(x), x


def log1pexp_bwd(x, g):
    return (g * (1.0 - 1.0 / (1.0 + qb.exp(x))),)


log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)


def test_custom_vjp_value_is_the_function_and_gradient_is_the_rule():
    x = qb.array([0.0, 10.0, 100.0, 1000.0])
    assert_close(log1pexp(x)[0:3], qb.log(1.0 + qb.exp(x))[0:3])

    def total(v):
        return qb.sum(log1pexp(v))

    expected = [sigmoid(value) for value in (0.0, 10.0, 100.0, 1000.0)]
    assert_close(qb.grad(total)(x), expected)
    # The naive gradient overflows to NaN at 1000; the rule does not.
    naive = qb.grad(lambda v: qb.sum(qb.log(1.0 + qb.exp(v))))(x).tolist()
    assert math.isnan(naive[3])
    value, gradient = qb.value_and_grad(total)(x)
    assert_close(gradient, expected)
    out, pullback = qb.vjp(log1pexp, x)
    (cotangent,) = pullback(qb.array([1.0, 2.0, 3.0, 4.0]))
    assert_close(cotangent, [w * s for w, s in zip([1.0, 2.0, 3.0, 4.0], expected)])
    assert_close(qb.jit(qb.grad(total))(x), expected)
    assert_close(qb.jacobian(log1pexp)(qb.array([0.0, 2.0])), [[0.5, 0.0], [0.0, sigmoid(2.0)]])


def test_custom_vjp_backward_is_differentiable():
    second = qb.grad(qb.grad(lambda v: log1pexp(v)))(qb.array(1.0)).item()
    assert abs(second - sigmoid(1.0) * (1.0 - sigmoid(1.0))) < 1e-12
    hessian = qb.hessian(lambda v: qb.sum(log1pexp(v)))(qb.array([0.0, 1.0]))
    assert_close(hessian, [[0.25, 0.0], [0.0, sigmoid(1.0) * (1.0 - sigmoid(1.0))]])


def test_custom_vjp_forward_mode_raises():
    raises(
        ValueError,
        qb.jvp,
        log1pexp,
        (qb.array(1.0),),
        (qb.array(1.0),),
        match="forward-mode differentiation",
    )
    # A call that does not depend on the primals does not need a forward rule.
    primal, tangent = qb.jvp(lambda x: x * log1pexp(qb.array(0.0)), (qb.array(2.0),), (1.0,))
    assert_close(tangent, math.log(2.0))


def identity(lo, hi, x):
    return x


def clip_fwd(lo, hi, x):
    return x, None


def clip_bwd(lo, hi, residuals, g):
    return (qb.minimum(qb.maximum(g, lo), hi),)


clip_gradient = qb.custom_vjp(identity, nondiff_argnums=(0, 1))
clip_gradient.defvjp(clip_fwd, clip_bwd)


def test_custom_vjp_clips_gradients_with_nondiff_arguments():
    def loss(x):
        return qb.sum(clip_gradient(-1.0, 1.0, x) * qb.array([5.0, -0.5, -3.0]))

    x = qb.array([1.0, 2.0, 3.0])
    assert_close(qb.grad(loss)(x), [1.0, -0.5, -1.0])
    assert_close(clip_gradient(-1.0, 1.0, x), x)


@qb.custom_vjp
def scaled_pair(params, scale):
    return {"sum": params["a"] * scale + params["b"], "product": params["a"] * params["b"]}


def scaled_pair_fwd(params, scale):
    return scaled_pair(params, scale), (params["a"], params["b"], scale)


def scaled_pair_bwd(residuals, cotangent):
    a, b, scale = residuals
    g_sum, g_product = cotangent["sum"], cotangent["product"]
    # Deliberately doubled, so the test sees the rule and not the derivative.
    return (
        {"a": 2.0 * (g_sum * scale + g_product * b), "b": 2.0 * (g_sum + g_product * a)},
        None,
    )


scaled_pair.defvjp(scaled_pair_fwd, scaled_pair_bwd)


def test_custom_vjp_takes_and_returns_pytrees():
    params = {"a": qb.array([1.0, 2.0]), "b": qb.array([3.0, -1.0])}
    scale = qb.array(0.5)

    def loss(params, scale):
        out = scaled_pair(params, scale)
        return qb.sum(out["sum"]) + qb.sum(out["product"])

    grads, scale_grad = qb.grad(loss, argnums=(0, 1))(params, scale)
    assert_close(grads["a"], [2.0 * (0.5 + 3.0), 2.0 * (0.5 - 1.0)])
    assert_close(grads["b"], [2.0 * (1.0 + 1.0), 2.0 * (1.0 + 2.0)])
    assert_close(scale_grad, 0.0)


def test_custom_vjp_under_vmap_and_jit():
    x = qb.array([0.0, 1.0, 1000.0])
    expected = [sigmoid(value) for value in (0.0, 1.0, 1000.0)]
    assert_close(qb.vmap(qb.grad(lambda v: log1pexp(v)))(x), expected)
    assert_close(qb.grad(lambda v: qb.sum(qb.vmap(log1pexp)(v)))(x), expected)
    # A shared (unmapped) operand receives the sum of the per-example cotangents.
    w = qb.array(2.0)
    shared = qb.grad(lambda w: qb.sum(qb.vmap(lambda v: log1pexp(v * w))(x)), argnums=0)(w)
    assert_close(shared, sum(v * sigmoid(2.0 * v) for v in (0.0, 1.0, 1000.0)))
    assert_close(qb.jit(qb.vmap(qb.grad(lambda v: log1pexp(v))))(x), expected)


def test_custom_vjp_rejects_closures_and_bad_rules():
    def outer(y):
        @qb.custom_vjp
        def closes(x):
            return x * y

        closes.defvjp(lambda x: (x * y, None), lambda r, g: (g,))
        return qb.sum(closes(qb.array([1.0])))

    raises(TypeError, qb.grad(outer), qb.array(2.0), match="closes over a traced value")

    @qb.custom_vjp
    def bad(x):
        return x * 2.0

    bad.defvjp(lambda x: (x * 2.0, None), lambda r, g: (g[0:1],))
    raises(TypeError, qb.grad(lambda x: qb.sum(bad(x))), qb.array([1.0, 2.0]), match="shape")
    undefined = qb.custom_vjp(lambda x: x)
    raises(AttributeError, undefined, qb.array(1.0))


# -- custom_jvp ----------------------------------------------------------------


@qb.custom_jvp
def sinc(x):
    # sin(0) / 0 is NaN but not selected; its derivative still poisons the
    # naive gradient at 0 through the unselected branch.
    return qb.where(qb.equal(x, 0.0), 1.0, qb.sin(x) / x)


@sinc.defjvp
def sinc_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    safe = qb.where(qb.equal(x, 0.0), 1.0, x)
    slope = qb.where(qb.equal(x, 0.0), 0.0, (safe * qb.cos(safe) - qb.sin(safe)) / (safe * safe))
    return sinc(x), t * slope


def test_custom_jvp_forward_and_reverse_use_the_rule():
    x = qb.array([0.0, 1.0])
    slope = math.cos(1.0) - math.sin(1.0)
    primal, tangent = qb.jvp(sinc, (x,), (qb.array([1.0, 2.0]),))
    assert_close(primal, [1.0, math.sin(1.0)])
    assert_close(tangent, [0.0, 2.0 * slope])
    assert_close(qb.grad(lambda v: qb.sum(sinc(v)))(x), [0.0, slope])
    assert_close(qb.jit(qb.grad(lambda v: qb.sum(sinc(v))))(x), [0.0, slope])
    assert_close(qb.jacobian(sinc)(x), [[0.0, 0.0], [0.0, slope]])
    # The rule replaces a gradient that is NaN at the removable singularity.
    naive = qb.grad(lambda v: qb.sum(sinc.fun(v)))(x).tolist()
    assert math.isnan(naive[0])


def log_domain_norm(x):
    # |x| through exp(log(.) / 2): the value at 0 is exp(-inf) = 0, but the
    # derivative of log there makes the naive gradient NaN.
    return qb.exp(0.5 * qb.log(qb.sum(x * x)))


safe_norm = qb.custom_jvp(log_domain_norm)


@safe_norm.defjvp
def safe_norm_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    norm = safe_norm(x)
    return norm, qb.sum(x * t) / qb.where(qb.equal(norm, 0.0), 1.0, norm)


def test_custom_jvp_fixes_a_nan_gradient_at_zero():
    zero = qb.array([0.0, 0.0])
    naive = qb.grad(log_domain_norm)(zero).tolist()
    assert all(math.isnan(value) for value in naive)
    assert_close(qb.grad(safe_norm)(zero), [0.0, 0.0])
    assert_close(qb.grad(safe_norm)(qb.array([3.0, 4.0])), [0.6, 0.8])
    # Second order differentiates the rule: the Hessian of |x| at (3, 4).
    hessian = qb.hessian(safe_norm)(qb.array([3.0, 4.0]))
    assert_close(hessian, [[16.0 / 125.0, -12.0 / 125.0], [-12.0 / 125.0, 9.0 / 125.0]])


def test_custom_jvp_with_nondiff_arguments_and_vmap():
    def power(n, x):
        return x**n

    power = qb.custom_jvp(power, nondiff_argnums=(0,))

    @power.defjvp
    def power_jvp(n, primals, tangents):
        (x,), (t,) = primals, tangents
        return power(n, x), 10.0 * n * x ** (n - 1) * t  # scaled to observe the rule

    x = qb.array([1.0, 2.0])
    assert_close(qb.vmap(qb.grad(lambda v: power(3, v)))(x), [30.0, 120.0])
    assert_close(qb.grad(lambda v: qb.sum(qb.vmap(lambda u: power(3, u))(v)))(x), [30.0, 120.0])


# -- checkpoint ----------------------------------------------------------------


def mlp(params, x, checkpointed):
    for w, b in params:
        layer = lambda h, w=w, b=b: qb.tanh(h @ w + b)  # noqa: E731
        x = qb.checkpoint(layer)(x) if checkpointed else layer(x)
    return qb.sum(x * x)


def mlp_params():
    params = []
    for layer in range(3):
        w = qb.array([[0.1 * (i + 2 * j + layer) - 0.4 for j in range(4)] for i in range(4)])
        b = qb.array([0.05 * (j - layer) for j in range(4)])
        params.append((w, b))
    return params


def test_checkpoint_gradients_match_the_unchecked_function():
    params = mlp_params()
    x = qb.array([[0.3, -0.2, 0.5, 0.1], [-0.4, 0.6, 0.2, -0.3]])
    plain = qb.grad(lambda p: mlp(p, x, False))(params)
    remat = qb.grad(lambda p: mlp(p, x, True))(params)
    for (w0, b0), (w1, b1) in zip(plain, remat):
        assert_close(w1, w0, 1e-14)
        assert_close(b1, b0, 1e-14)
    assert_close(mlp(params, x, True), mlp(params, x, False), 0.0)
    jitted = qb.jit(qb.grad(lambda p: mlp(p, x, True)))(params)
    for (w0, _), (w1, _) in zip(plain, jitted):
        assert_close(w1, w0, 1e-14)


def test_checkpoint_recomputes_in_the_backward_pass():
    params = mlp_params()
    x = qb.array([[0.3, -0.2, 0.5, 0.1]])

    def compiled_nodes(checkpointed):
        lowered = qb.jit(qb.grad(lambda p: mlp(p, x, checkpointed))).lower(params)
        return lowered.program.compile("cpu").node_count

    # Without the rule the backward pass reuses the forward tanh values; with
    # it, each layer is recomputed from its input, which plan CSE keeps.
    assert compiled_nodes(True) > compiled_nodes(False), (
        compiled_nodes(True),
        compiled_nodes(False),
    )


def test_checkpoint_higher_order_forward_mode_and_static_arguments():
    def f(x):
        return qb.sum(qb.sin(x) * qb.exp(x))

    g = qb.checkpoint(f)
    x = qb.array([0.3, -0.7])
    assert_close(qb.hessian(g)(x), qb.hessian(f)(x), 1e-13)
    assert_close(qb.jvp(g, (x,), (qb.array([1.0, 2.0]),))[1], qb.jvp(f, (x,), (qb.array([1.0, 2.0]),))[1])
    assert_close(qb.grad(qb.grad(lambda s: g(s * x)))(qb.array(1.0)), qb.grad(qb.grad(lambda s: f(s * x)))(qb.array(1.0)), 1e-13)

    def scaled(x, power):
        return qb.sum(x**power)

    h = qb.remat(scaled, static_argnums=1)
    assert_close(qb.grad(lambda v: h(v, 3))(x), [3.0 * 0.09, 3.0 * 0.49])


def test_custom_rules_inside_control_flow_bodies_raise():
    def body(i, carry):
        return log1pexp(carry)

    raises(
        ValueError,
        qb.grad(lambda x: qb.fori_loop(0, 2, body, x)),
        qb.array(0.5),
        match="custom differentiation rule",
    )


def test_custom_rules_on_devices_match_cpu():
    params = mlp_params()
    x = qb.array([[0.3, -0.2, 0.5, 0.1]])
    functions = (
        qb.grad(lambda v: qb.sum(log1pexp(v))),
        qb.grad(lambda v: qb.sum(sinc(v))),
        qb.grad(lambda p: mlp(p, x, True)),
    )
    arguments = (qb.array([0.0, 1.0, 50.0]), qb.array([0.0, 1.0]), params)
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue
        for function, argument in zip(functions, arguments):
            expected = qb.tree.leaves(qb.jit(function)(argument))
            actual = qb.tree.leaves(qb.jit(function, device=device)(argument))
            for got, want in zip(actual, expected):
                assert_close(got, want, 1e-5)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
