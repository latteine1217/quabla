"""Device jit, ahead-of-time signatures and migration compatibility."""

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
            unsupported = qb.jit(lambda a, b: qb.solve(a, b), device="mlx")
            error = raises(
                qb.UnsupportedOperationError,
                unsupported,
                qb.array([[2.0]], dtype=qb.float32),
                qb.array([[1.0]], dtype=qb.float32),
            )
            assert (error.op, error.device) == ("solve", "mlx")
    if not ran:
        print("SKIP device jit runtime: GPU gates unset")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
