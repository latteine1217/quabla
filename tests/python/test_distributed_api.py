"""S8 wrapper contracts; NCCL parity is opt-in on a two-GPU host."""

import os

import quabla as qb
from quabla import distributed


def expect(error_type, fragment, operation):
    try:
        operation()
    except error_type as error:
        assert fragment in str(error), str(error)
        return error
    raise AssertionError(f"expected {error_type.__name__}: {fragment}")


def loss(params, batch, scale=1.0):
    return (
        (batch["x"] * params["w"] + params["b"] - batch["y"]).powi(2)
    ).mean() * scale


def arguments(dtype=qb.float64, size=8):
    params = {"w": qb.array([[0.5]], dtype=dtype), "b": qb.array([[0.25]], dtype=dtype)}
    batch = {
        "x": qb.array([[float(i - 4)] for i in range(size)], dtype=dtype),
        "y": qb.array([[float(2 * i - 7)] for i in range(size)], dtype=dtype),
    }
    return params, batch


def transform(fun=loss, **kwargs):
    return distributed.value_and_grad(
        fun, devices=["cuda:0", "cuda:1"], shard_argnums=1, **kwargs
    )


def close(left, right, tolerance=1e-5):
    assert left.shape == right.shape
    for a, b in zip(left.to_flat_list(), right.to_flat_list()):
        assert abs(a - b) <= tolerance, (a, b)


def test_invalid_device_and_reduction_configuration():
    expect(
        ValueError,
        "at least two",
        lambda: distributed.value_and_grad(loss, devices=["cuda"], shard_argnums=1),
    )
    expect(
        ValueError,
        "duplicate",
        lambda: distributed.value_and_grad(
            loss, devices=["cuda", "cuda:0"], shard_argnums=1
        ),
    )
    expect(
        qb.UnsupportedOperationError,
        "CUDA",
        lambda: distributed.value_and_grad(
            loss, devices=["cpu", "cuda:1"], shard_argnums=1
        ),
    )
    expect(
        ValueError,
        "device must",
        lambda: distributed.value_and_grad(
            loss, devices=["cuda:-1", "cuda:1"], shard_argnums=1
        ),
    )
    expect(
        TypeError,
        "sequence",
        lambda: distributed.value_and_grad(loss, devices="cuda:0", shard_argnums=1),
    )
    expect(TypeError, "callable", lambda: transform(1))
    expect(ValueError, "reduction", lambda: transform(reduction="median"))


def test_invalid_sharding_and_gradient_requests():
    params, batch = arguments()
    expect(
        ValueError,
        "mapped-input gradients",
        lambda: transform(argnums=1)(params, batch),
    )
    expect(ValueError, "out of range", lambda: transform(argnums=2)(params, batch))
    expect(ValueError, "duplicate", lambda: transform(argnums=(0, 0))(params, batch))
    expect(TypeError, "tuple of ints", lambda: transform(argnums=True)(params, batch))
    expect(
        ValueError, "replicated parameter", lambda: transform(argnums=())(params, batch)
    )
    expect(
        ValueError,
        "batch argument",
        lambda: distributed.value_and_grad(
            loss, devices=["cuda:0", "cuda:1"], shard_argnums=()
        )(params, batch),
    )
    expect(ValueError, "equal device shards", lambda: transform()(*arguments(size=7)))
    expect(
        ValueError,
        "same axis-zero",
        lambda: transform()(params, {"x": batch["x"], "y": qb.zeros((4, 1))}),
    )
    expect(
        ValueError,
        "rank",
        lambda: transform()(params, {"x": qb.array(1.0), "y": batch["y"]}),
    )
    expect(TypeError, "array", lambda: transform()(params, {"x": 1.0, "y": batch["y"]}))
    expect(TypeError, "positional", lambda: transform()(params, batch=batch))
    expect(
        qb.UnsupportedOperationError,
        "another transform",
        lambda: qb.trace_tensor(
            lambda w, x: transform()({"w": w, "b": w}, {"x": x, "y": x})[0],
            [("w", [1, 1]), ("x", [8, 1])],
        ),
    )


def cpu_shards(params, batch, reduction, scale=1.0):
    half = batch["x"].shape[0] // 2
    evaluate = qb.tensor_value_and_grad_fn(
        lambda w, b, x, y: loss({"w": w, "b": b}, {"x": x, "y": y}, scale),
        [
            ("w", params["w"].shape, params["w"].dtype),
            ("b", params["b"].shape, params["b"].dtype),
        ]
        + [
            (name, [half, *value.shape[1:]], value.dtype)
            for name, value in batch.items()
        ],
    )
    results = []
    for replica in range(2):
        value, all_gradients = evaluate(
            {
                **params,
                **{
                    name: value.slice(0, replica * half, half).to_tensor()
                    for name, value in batch.items()
                },
            }
        )
        results.append((value, {name: all_gradients[name] for name in params}))
    factor = qb.array(0.5 if reduction == "mean" else 1.0, dtype=results[0][0].dtype)
    value = (results[0][0] + results[1][0]) * factor
    gradients = qb.tree.map(lambda a, b: (a + b) * factor, results[0][1], results[1][1])
    return value, gradients


def test_adapter_structure_signatures_and_cache_bound():
    original = qb._quabla.tensor_value_and_grad_data_parallel_cuda_fn
    builds = []

    def cpu_constructor(fun, specs, parameters, mapped, devices, reduction):
        builds.append((specs, parameters, mapped, devices, reduction))
        shards = [
            (name, [shape[0] // 2, *shape[1:]] if name in mapped else shape, dtype)
            for name, shape, dtype in specs
        ]
        traced = qb.trace_tensor(fun, shards)
        plan = traced.output.compile_cpu()

        def execute(inputs):
            value, gradients = plan.value_and_grad_data_parallel(
                inputs, mapped, 2, reduction
            )
            return value, {name: gradients[name] for name in parameters}

        return execute

    qb._quabla.tensor_value_and_grad_data_parallel_cuda_fn = cpu_constructor
    try:
        params, batch = arguments()
        evaluate = transform()
        value, grads = evaluate(params, batch)
        expected_value, expected_grads = cpu_shards(params, batch, "mean")
        close(value, expected_value)
        qb.tree.map(close, grads, expected_grads)
        assert builds[0][1] == ["params/b", "params/w"]
        assert builds[0][2] == ["batch/x", "batch/y"]
        assert builds[0][3] == [0, 1]
        # Data changes reuse the compiled signature; shapes/dtypes/static
        # values each select a distinct program.
        evaluate({**params, "w": qb.array([[1.0]])}, batch)
        assert len(builds) == 1
        evaluate(*arguments(size=4))
        float32_params, float32_batch = arguments(dtype=qb.float32)
        float32_value, float32_grads = evaluate(float32_params, float32_batch)
        expected32, expected_grads32 = cpu_shards(float32_params, float32_batch, "mean")
        close(float32_value, expected32)
        qb.tree.map(close, float32_grads, expected_grads32)
        evaluate(params, batch, 2.0)
        assert len(builds) == 4
        for scale in (3.0, 4.0, 5.0, 6.0):
            evaluate(params, batch, scale)
        expect(
            qb.RetraceLimitError, "max_traces=8", lambda: evaluate(params, batch, 7.0)
        )
        sum_value, sum_grads = transform(reduction="sum")(params, batch)
        expected_sum, expected_sum_grads = cpu_shards(params, batch, "sum")
        close(sum_value, expected_sum)
        qb.tree.map(close, sum_grads, expected_sum_grads)
        value, (tuple_grads,) = transform(argnums=(0,))(params, batch)
        qb.tree.map(close, tuple_grads, expected_grads)
        expect(
            ValueError,
            "scalar loss",
            lambda: transform(lambda p, b: b["x"] * p["w"])(params, batch),
        )
        expect(
            TypeError,
            "single scalar",
            lambda: transform(lambda p, b: (loss(p, b), {}))(params, batch),
        )
        # Multiple selected parameter arguments preserve their pytree order.
        multi = distributed.value_and_grad(
            lambda w, b, data: loss({"w": w[0], "b": b}, data),
            devices=["cuda:0", "cuda:1"],
            shard_argnums=-1,
            argnums=(1, 0),
        )
        _, (bias_grad, weight_grads) = multi([params["w"]], params["b"], batch)
        close(bias_grad, expected_grads["b"])
        close(weight_grads[0], expected_grads["w"])
    finally:
        qb._quabla.tensor_value_and_grad_data_parallel_cuda_fn = original


def test_feature_unavailable_contract():
    if os.environ.get("QUABLA_CUDA_NCCL_TEST") == "1":
        return
    params, batch = arguments()
    error = expect(
        qb.UnsupportedOperationError, "CUDA", lambda: transform()(params, batch)
    )
    assert error.op == "distributed.value_and_grad"
    assert error.device == "cuda"


def test_nccl_numerical_parity():
    if os.environ.get("QUABLA_CUDA_NCCL_TEST") != "1":
        return
    for dtype in (qb.float64, qb.float32):
        params, batch = arguments(dtype=dtype)
        for reduction in ("sum", "mean"):
            actual, grads = transform(reduction=reduction)(params, batch)
            expected, expected_grads = cpu_shards(params, batch, reduction)
            close(actual, expected)
            qb.tree.map(close, grads, expected_grads)
            legacy = qb.tensor_value_and_grad_data_parallel_cuda_fn(
                lambda w, b, x, y: loss({"w": w, "b": b}, {"x": x, "y": y}),
                [
                    (name, value.shape, dtype)
                    for name, value in [
                        ("w", params["w"]),
                        ("b", params["b"]),
                        ("x", batch["x"]),
                        ("y", batch["y"]),
                    ]
                ],
                ["w", "b"],
                ["x", "y"],
                [0, 1],
                reduction,
            )
            legacy_value, legacy_grads = legacy({**params, **batch})
            close(actual, legacy_value)
            qb.tree.map(close, grads, legacy_grads)
            assert actual.dtype == legacy_value.dtype
            for name in grads:
                assert grads[name].dtype == legacy_grads[name].dtype
            print(
                f"NCCL parity passed: dtype={dtype.name} reduction={reduction} loss={actual.to_flat_list()[0]}"
            )


def test_public_namespace_is_all():
    public = sorted(name for name in dir(distributed) if not name.startswith("_"))
    assert public == ["value_and_grad"], public
    assert distributed.__all__ == public


if __name__ == "__main__":
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            nccl_enabled = os.environ.get("QUABLA_CUDA_NCCL_TEST") == "1"
            if name == "test_nccl_numerical_parity" and not nccl_enabled:
                print(f"SKIP {name}: QUABLA_CUDA_NCCL_TEST=1 required")
                continue
            if name == "test_feature_unavailable_contract" and nccl_enabled:
                print(f"SKIP {name}: NCCL runtime validation enabled")
                continue
            test()
            print(f"PASS {name}")
