"""Device jit, ahead-of-time signatures and migration compatibility."""

import math
import os
import warnings

import quabla as qb


def raises(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except kind as error:
        return error
    raise AssertionError(f"expected {kind.__name__}")


def close(a, b, tolerance=1e-5):
    assert a.shape == b.shape
    assert (
        max((abs(x - y) for x, y in zip(a.to_flat_list(), b.to_flat_list())), default=0)
        < tolerance
    )


def test_devices_are_build_capabilities():
    assert set(qb.devices()) == {
        target for target, built in qb.Compiler().capabilities().items() if built
    }
    raises(ValueError, qb.jit, lambda x: x, device="cuda:-1")
    raises(ValueError, qb.ShapeDtype, [True])
    raises(ValueError, qb.ShapeDtype, [-1])
    raises(TypeError, qb.ShapeDtype, [2], "float32")


def test_lower_has_no_evaluation_and_keeps_pytrees_and_statics():
    traces = []

    def f(params, scale):
        traces.append(1)
        return {
            "value": qb.sum(params["x"] ** 2) * scale,
            "aux": (params["x"], None, 7),
        }

    lowered = qb.jit(f, static_argnums=1).lower(
        {"x": qb.ShapeDtype([2], qb.float32)}, 2
    )
    assert traces == [1]
    assert "f32" in lowered.as_text()
    compiled = lowered.compile()
    x = qb.array([1.0, 3.0], dtype=qb.float32)
    result = compiled({"x": x}, 2)
    assert result["value"].item() == 20.0
    close(result["aux"][0], x)
    assert result["aux"][1:] == (None, 7)
    assert traces == [1]
    raises(ValueError, compiled, {"x": x}, 3)
    raises(ValueError, compiled, {"x": qb.array([1.0, 3.0])}, 2)
    assert qb.jit(lambda: (None, 3)).lower().compile()() == (None, 3)


def test_deprecated_names_warn_once_and_preserve_objects():
    from quabla import _compat

    for name in ("Matrix", "trace_tensor", "Adam"):
        _compat._WARNED.discard(name)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", DeprecationWarning)
            first = getattr(qb, name)
            assert first is getattr(qb, name)
        assert len(seen) == 1 and seen[0].category is DeprecationWarning
        assert "0.x" in str(seen[0].message)
    assert qb.Adam is qb.optim.Adam
    assert qb.legacy.Matrix is qb._quabla.Matrix
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", DeprecationWarning)
        assert qb.legacy.grad is qb._quabla.grad
        assert qb.legacy.jit is qb._quabla.jit
        qb.grad(lambda x: x * x)(qb.array(2.0))
    assert not seen


def test_device_jit_matches_cpu_for_nested_transforms_and_pytrees():
    ran = False
    for device, gate in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(gate) != "1":
            continue
        assert device in qb.devices(), f"{gate} requested but target not built"
        ran = True

        def loss(params, x):
            y = qb.tanh(x * params["w"] + params["b"])
            return qb.mean(y * y), {"prediction": y, "static": 4}

        params = {
            "w": qb.array(0.3, dtype=qb.float32),
            "b": qb.array(0.1, dtype=qb.float32),
        }
        x = qb.array([0.2, -0.4, 0.8], dtype=qb.float32)
        fun = qb.value_and_grad(loss, has_aux=True)
        expected = qb.jit(fun)(params, x)
        actual = qb.jit(fun, device=device)(params, x)
        close(actual[0][0], expected[0][0])
        close(actual[0][1]["prediction"], expected[0][1]["prediction"])
        for name in params:
            close(actual[1][name], expected[1][name])
        close(
            qb.jit(qb.hessian(lambda y: qb.sum(qb.sin(y) ** 2)), device=device)(x),
            qb.hessian(lambda y: qb.sum(qb.sin(y) ** 2))(x),
        )
        compiled = qb.jit(fun, device=device).lower(params, x).compile()
        close(compiled(params, x)[0][0], expected[0][0])
        f = qb.jit(lambda y: y * y, device=device)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", UserWarning)
            f(qb.array([2.0, 3.0]))
            f(qb.array([3.0, 4.0]))
            f(qb.array([1.0]))
        assert len(seen) == 1
        # The warning names the caller's line, not quabla's cache layers.
        assert seen[0].filename == __file__, seen[0].filename
        if device == "mlx":
            unsupported = qb.jit(lambda a: a + float("inf"), device="mlx")
            error = raises(
                qb.UnsupportedOperationError,
                unsupported,
                qb.array([[2.0]], dtype=qb.float32),
            )
            assert (error.op, error.device) == ("constant", "mlx")
    if not ran:
        print("SKIP device jit runtime: GPU gates unset")


def test_device_stable_activations_and_nan_propagation_match_cpu():
    ran = False
    for device, gate in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(gate) != "1":
            continue
        assert device in qb.devices(), f"{gate} requested but target not built"
        ran = True
        # Normal float32 range only: GPUs may flush subnormal results to zero.
        x = qb.array([-30.0, -5.0, -1.0, 0.0, 0.5, 3.0, 25.0], dtype=qb.float32)
        y = qb.array([-2.0, -5.0, 0.0, 0.0, 1.0, 2.0, 30.0], dtype=qb.float32)
        functions = {
            "log1p": lambda t, u: qb.log1p(qb.abs(t)),
            "softplus": lambda t, u: qb.softplus(t),
            "sigmoid": lambda t, u: qb.sigmoid(t),
            "maximum": lambda t, u: qb.maximum(t, u),
            "minimum": lambda t, u: qb.minimum(t, u),
            "relu": lambda t, u: qb.relu(t),
        }
        for name, function in functions.items():
            value = qb.jit(function, device=device)(x, y)
            expected = qb.jit(function)(x, y)
            for actual, reference in zip(value.to_flat_list(), expected.to_flat_list()):
                assert abs(actual - reference) <= 2e-6 * max(1.0, abs(reference)), (
                    device,
                    name,
                    value.tolist(),
                    expected.tolist(),
                )
            gradient = qb.grad(lambda t, u: qb.sum(function(t, u)), argnums=(0, 1))
            for actual, reference in zip(
                qb.jit(gradient, device=device)(x, y), qb.jit(gradient)(x, y)
            ):
                close(actual, reference, 2e-6)
        nan = qb.array([1.0, math.nan, 3.0], dtype=qb.float32)
        other = qb.array([2.0, 2.0, 2.0], dtype=qb.float32)
        for function in [
            lambda t: qb.max(t),
            lambda t: qb.min(t),
            lambda t: qb.relu(t),
            lambda t: qb.maximum(other, t),
            lambda t: qb.minimum(t, other),
        ]:
            result = qb.jit(function, device=device)(nan).to_flat_list()
            assert any(math.isnan(value) for value in result), (device, result)
    if not ran:
        print("SKIP device activation parity: GPU gates unset")


def test_device_compositions_and_shape_helpers_match_cpu():
    ran = False
    for device, gate in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(gate) != "1":
            continue
        assert device in qb.devices(), f"{gate} requested but target not built"
        ran = True
        x = qb.array(
            [[1000.0, 0.0, -3.0, 2.5], [-0.5, 1.25, 30.0, -30.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=qb.float32,
        )
        v = qb.array([0.5, -1.0, 2.0, 0.25], dtype=qb.float32)
        functions = {
            "softmax": lambda t: qb.softmax(t),
            "log_softmax": lambda t: qb.log_softmax(t, axis=0),
            "logsumexp": lambda t: qb.logsumexp(t, axis=-1),
            "var": lambda t: qb.var(t, axis=1, ddof=1),
            "std": lambda t: qb.std(t * 1e-3, axis=0),
            "silu": lambda t: qb.silu(t),
            "gelu": lambda t: qb.gelu(t),
            "clip": lambda t: qb.clip(t, -1.0, 2.0),
            "sign": lambda t: qb.sign(t),
            "index": lambda t: t[None, ::-1, ..., 1::2].T,
            "split": lambda t: qb.split(t, [1], axis=1)[1].reshape(-1),
            "squeeze": lambda t: qb.squeeze(qb.expand_dims(t, (0, 2)), 0),
            "matmul": lambda t: qb.matmul(t * 1e-3, v),
        }
        for name, function in functions.items():
            value = qb.jit(function, device=device)(x)
            expected = qb.jit(function)(x)
            assert value.shape == expected.shape, (device, name)
            for actual, reference in zip(value.to_flat_list(), expected.to_flat_list()):
                assert abs(actual - reference) <= 4e-6 * max(1.0, abs(reference)), (
                    device,
                    name,
                    value.tolist(),
                    expected.tolist(),
                )
            weights = qb.arange(1.0, 1.0 + len(expected.to_flat_list()))
            weights = weights.reshape(expected.shape).astype(qb.float32) * 0.1

            def loss(t, function=function, weights=weights):
                return qb.sum(function(t) * weights)

            gradient = qb.jit(qb.grad(loss), device=device)(x)
            reference = qb.jit(qb.grad(loss))(x)
            assert all(math.isfinite(g) for g in reference.to_flat_list()), (name, reference)
            for actual, expected_gradient in zip(
                gradient.to_flat_list(), reference.to_flat_list()
            ):
                assert abs(actual - expected_gradient) <= 4e-5 * max(
                    1.0, abs(expected_gradient)
                ), (device, name, gradient.tolist(), reference.tolist())
        special = qb.array([math.nan, -2.0, 0.0, 3.0, -math.inf], dtype=qb.float32)
        signs = qb.jit(qb.sign, device=device)(special).tolist()
        assert math.isnan(signs[0]) and signs[1:] == [-1.0, 0.0, 1.0, -1.0], (device, signs)
        bounds = qb.array([[-math.inf, -math.inf], [math.inf, 0.0]], dtype=qb.float32)
        totals = qb.jit(lambda t: qb.logsumexp(t, axis=1), device=device)(bounds)
        assert totals.tolist() == [-math.inf, math.inf], (device, totals.tolist())
    if not ran:
        print("SKIP device compositions: GPU gates unset")


def close_relative(a, b, tolerance):
    assert a.shape == b.shape, (a.shape, b.shape)
    for x, y in zip(a.to_flat_list(), b.to_flat_list()):
        assert abs(x - y) <= tolerance * max(1.0, abs(y)), (x, y, a.tolist(), b.tolist())


def test_device_expm1_erf_erfc_atan2_stop_gradient_cumsum_and_prod_match_cpu():
    ran = False
    for device, gate in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(gate) != "1":
            continue
        assert device in qb.devices(), f"{gate} requested but target not built"
        ran = True
        x = qb.array([-6.0, -1.0, -1e-4, 0.0, 0.3, 1.7, 4.0], dtype=qb.float32)
        y = qb.array([2.0, -0.5, 0.0, 0.0, -3.0, 0.25, 1e-3], dtype=qb.float32)
        functions = {
            "expm1": lambda t, u: qb.expm1(t),
            "erf": lambda t, u: qb.erf(t),
            "erfc": lambda t, u: qb.erfc(t),
            "gelu_exact": lambda t, u: qb.gelu(t, approximate=False),
            "prod": lambda t, u: qb.prod(qb.reshape(t * u + 1.0, [7, 1]), axis=0),
            "atan2": lambda t, u: qb.atan2(t, u),
            "stop_gradient": lambda t, u: t * qb.stop_gradient(t * u) - qb.stop_gradient(t),
            "cumsum": lambda t, u: qb.cumsum(t * u),
            "cumsum_reverse": lambda t, u: qb.cumsum(qb.sin(t) * u, reverse=True),
        }
        for name, function in functions.items():
            value = qb.jit(function, device=device)(x, y)
            expected = qb.jit(function)(x, y)
            # erff, erfcf, expm1f and atan2f (and the MLX erfc fit) are
            # within a few float32 ulp of the correctly rounded CPU values;
            # prod is the same tree of rounded multiplications everywhere.
            close_relative(value, expected, 4e-6)
            gradient = qb.grad(lambda t, u: qb.sum(function(t, u)), argnums=(0, 1))
            for actual, reference in zip(
                qb.jit(gradient, device=device)(x, y), qb.jit(gradient)(x, y)
            ):
                close_relative(actual, reference, 4e-6)
        point = qb.array([0.8, -1.5], dtype=qb.float32)
        for function in [
            lambda p: qb.atan2(p[0], p[1]),
            lambda p: qb.sum(qb.erf(p) * qb.expm1(p)),
            lambda p: qb.sum(qb.erfc(p)) * qb.prod(p),
        ]:
            close_relative(
                qb.jit(qb.hessian(function), device=device)(point),
                qb.jit(qb.hessian(function))(point),
                4e-6,
            )
        # A long float32 scan: CUDA scans each line sequentially with
        # __fadd_rn and matches the CPU bitwise; MLX scans in parallel.
        long = qb.array(
            [[((i * 7919 % 1000) - 500) * 1.37e-3 + 1.0 / (1 + i) for i in range(300)]] * 2,
            dtype=qb.float32,
        )
        for axis, reverse in [(1, False), (1, True), (0, True)]:
            def scan(t, a=axis, r=reverse):
                return qb.cumsum(t, a, r)

            value = qb.jit(scan, device=device)(long)
            expected = qb.jit(scan)(long)
            if device == "cuda":
                assert value.tolist() == expected.tolist(), (axis, reverse)
            else:
                close_relative(value, expected, 1e-5)
        batched = qb.jit(qb.vmap(lambda t: qb.cumsum(t, reverse=True)), device=device)(long)
        close_relative(batched, qb.jit(qb.vmap(lambda t: qb.cumsum(t, reverse=True)))(long), 1e-5)
    if not ran:
        print("SKIP device new-op parity: GPU gates unset")


def test_jit_precision_argument():
    raises(ValueError, qb.jit, lambda x: x, precision="float16")
    raises(ValueError, qb.jit, lambda x: x, device="cpu", precision="float32")
    # The CPU already runs float64 natively: the argument changes nothing.
    x = qb.array([0.1, 1.0 / 3.0, 2.5])
    value = qb.jit(lambda t: qb.exp(t) * qb.erf(t), precision="float64")(x)
    assert value.tolist() == qb.jit(lambda t: qb.exp(t) * qb.erf(t))(x).tolist()
    if "mlx" in qb.devices():
        error = raises(
            qb.UnsupportedOperationError, qb.jit, lambda t: t, device="mlx", precision="float64"
        )
        assert (error.op, error.device) == ("float64", "mlx")


def close_normwise(actual, expected, tolerance):
    """`max|actual - expected| <= tolerance * max|expected|` per output."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    got, want = actual.to_flat_list(), expected.to_flat_list()
    scale = max((abs(y) for y in want), default=0.0)
    error = max((abs(x - y) for x, y in zip(got, want)), default=0.0)
    assert error <= tolerance * max(scale, 1e-300), (error, scale)
    return error / max(scale, 1e-300)


def test_cuda_float64_precision_matches_cpu():
    if os.environ.get("QUABLA_CUDA_TEST") != "1":
        print("SKIP CUDA float64 precision: QUABLA_CUDA_TEST unset")
        return
    assert "cuda" in qb.devices(), "QUABLA_CUDA_TEST requested but target not built"

    def series(count, scale=1.0, shift=0.0):
        return [scale * math.sin(1.37 * i + 0.4) + shift for i in range(count)]

    def full_rank(rows, columns, seed):
        return qb.array(
            [[math.sin((i + 1) * (j + 2) * 0.731 + seed) for j in range(columns)]
             for i in range(rows)]
        )

    def compare(name, function, *args):
        expected = qb.jit(function)(*args)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = qb.jit(function, device="cuda", precision="float64")(*args)
        assert not caught, (name, [str(warning.message) for warning in caught])
        for got, want in zip(qb.tree.leaves(actual), qb.tree.leaves(expected), strict=True):
            assert got.dtype == qb.float64, (name, got.dtype)
            # Device and CPU differ only in libm rounding and summation order.
            close_normwise(got, want, 1e-12)

    x = qb.array(series(1000, 0.9, 1.0))
    y = qb.array(series(1000, 2.0))
    compare(
        "elementwise",
        lambda a, b: qb.exp(a) + qb.log(a) * qb.erf(b) + qb.erfc(b) + qb.log1p(a)
        + qb.expm1(b) + qb.tanh(b) * qb.sin(a) + a**b + qb.atan2(b, a) + qb.sqrt(a),
        x,
        y,
    )
    # A float32 sum of 1e6 values loses about six digits; float64 keeps them.
    many = qb.array([1.0 + 1e-9 * i for i in range(1_000_000)])
    compare("sum", lambda a: (qb.sum(a), qb.mean(a)), many)
    # The mean's VJP scale 1/4096 is exact in float32 but its shortest float32
    # digits are not exact in float64; the double kernels must keep 2^-12.
    compare("mean gradient", qb.grad(lambda a: qb.mean(a * a)), qb.array(series(4096)))
    matrix = qb.array([series(400, 1.0, 0.1 * row) for row in range(300)])
    compare("axis sums", lambda a: (qb.sum(a, axis=0), qb.mean(a, axis=1)), matrix)
    lhs = qb.array([series(96, 1.0, 0.01 * row) for row in range(64)])
    rhs = qb.array([series(48, 1.0, -0.02 * row) for row in range(96)])
    compare("matmul", lambda a, b: (a @ b, qb.reshape(a, [4, 16, 96]) @ b), lhs, rhs)
    spd = qb.jit(lambda a: a @ a.T + 5.0 * qb.eye(5))(full_rank(5, 5, 0.2))
    compare(
        "linalg",
        lambda a, b: (
            qb.linalg.solve(a, b),
            qb.linalg.cholesky(a),
            *qb.linalg.eigh(a),
            *qb.linalg.slogdet(a),
            *qb.linalg.qr(b),
            qb.linalg.svd(b, compute_uv=False),
        ),
        spd,
        full_rank(5, 3, 1.1),
    )
    compare(
        "cholesky gradient",
        qb.grad(lambda a: qb.sum(qb.linalg.cholesky(a) * qb.linalg.cholesky(a))),
        spd,
    )

    # A PINN residual u'' + sin(pi x) on a tanh network, differentiated
    # with respect to the parameters.
    def network(params, point):
        hidden = qb.tanh(point * params["w1"] + params["b1"])
        return qb.sum(hidden * params["w2"])

    def pinn_loss(params, points):
        second = qb.vmap(qb.grad(qb.grad(network, argnums=1), argnums=1), in_axes=(None, 0))
        residual = second(params, points) + qb.sin(math.pi * points)
        return qb.mean(residual * residual)

    params = {
        "w1": qb.array(series(16, 1.5)),
        "b1": qb.array(series(16, 0.5, 0.1)),
        "w2": qb.array(series(16, 0.8, -0.05)),
    }
    points = qb.array([i / 63.0 for i in range(64)])
    compare("pinn value_and_grad", qb.value_and_grad(pinn_loss), params, points)

    def oscillator(state, t):
        return qb.stack([state[1], -state[0] - 0.1 * state[1]])

    compare(
        "dopri5",
        lambda y0: qb.ode.odeint(
            oscillator, y0, (0.0, 5.0), method="dopri5", rtol=1e-10, atol=1e-12,
            max_steps=4000,
        ),
        qb.array([1.0, 0.0]),
    )
    compare(
        "control flow",
        lambda v: (
            qb.fori_loop(0, 20, lambda i, c: 0.9 * c + 0.1 * qb.sin(c), v),
            qb.scan(lambda c, i: (c * 1.01 + 0.001, qb.sum(c)), v, length=8)[1],
            qb.while_loop(lambda c: qb.sum(c) < 1e4, lambda c: c * 1.5 + 1.0, v),
            qb.cond(qb.sum(v) > 0.0, lambda u: qb.exp(u), lambda u: -u, v),
        ),
        qb.array(series(8, 0.5, 0.6)),
    )
    compare(
        "loop gradients",
        qb.grad(
            lambda v: qb.sum(
                qb.fori_loop(0, 6, lambda i, c, w: qb.tanh(c) * c + w, v, operands=(v,))
            )
            + qb.sum(qb.scan(lambda c, i: (qb.sin(c), c * c), v, length=4)[1])
        ),
        qb.array(series(8, 0.5)),
    )

    # The default lowering still warns and computes in float32.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default = qb.jit(lambda a: qb.sum(a), device="cuda")(many)
    assert any("float32" in str(warning.message) for warning in caught)
    assert abs(default.item() - qb.jit(lambda a: qb.sum(a))(many).item()) > 1e-9

    # One plan has one floating element type: float32 values are rejected.
    mixed = qb.jit(lambda a, b: a + b.astype(qb.float64), device="cuda", precision="float64")
    error = raises(
        qb.UnsupportedOperationError, mixed, qb.array([1.0]), qb.array([2.0], dtype=qb.float32)
    )
    assert error.op == "float32", error.op
    # A program without float64 values compiles as with the default.
    single = qb.array([0.5, 1.5], dtype=qb.float32)
    value = qb.jit(lambda a: qb.exp(a), device="cuda", precision="float64")(single)
    assert value.dtype == qb.float32
    assert value.tolist() == qb.jit(lambda a: qb.exp(a), device="cuda")(single).tolist()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
