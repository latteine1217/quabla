"""Analytic optimizer updates and CPU/device Trainer behavior."""

import collections
import dataclasses
import math
import os
from unittest.mock import patch

import quabla as qb
from quabla.optim import Adam, SGD, Trainer


def raises(kind, fun, *args, **kwargs):
    try:
        fun(*args, **kwargs)
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def scalar(value):
    return value.to_flat_list()[0]


def test_adam_pure_analytic_updates_and_state_independence():
    params = {"layers": [(qb.array([3.0]), None), qb.array([-1.0], dtype=qb.float32)]}
    grads = {"layers": [(qb.array([2.0]), None), qb.array([-4.0], dtype=qb.float32)]}
    optimizer = Adam(0.1, b1=0.5, b2=0.75, eps=0.01)
    original = optimizer.init(params)
    updated, state = optimizer.update(params, grads, original)
    assert scalar(params["layers"][0][0]) == 3.0
    assert original["step"] == 0 and scalar(original["m"]["layers"][0][0]) == 0
    assert math.isclose(scalar(updated["layers"][0][0]), 3 - 0.1 * 2 / 2.01)
    assert updated["layers"][1].dtype == qb.float32
    assert scalar(state["m"]["layers"][0][0]) == 1
    updated2, state2 = optimizer.update(updated, grads, state)
    assert state["step"] == 1 and state2["step"] == 2
    assert math.isclose(scalar(updated2["layers"][0][0]), 3 - 0.2 * 2 / 2.01)
    repeated, _ = optimizer.update(params, grads, original)
    assert (
        repeated["layers"][0][0].to_flat_list()
        == updated["layers"][0][0].to_flat_list()
    )


def test_legacy_adam_step_matches_native_and_accepts_alias_keywords():
    native = qb._quabla.Adam(0.05, beta1=0.8, beta2=0.95, epsilon=0.01)
    new = Adam(0.05, beta1=0.8, beta2=0.95, epsilon=0.01)
    a = b = {"w": qb.array([1.0, -2.0], dtype=qb.float32)}
    for gradients in ([0.5, 2.0], [-1.0, 1.5], [3.0, -0.25]):
        grads = {"w": qb.array(gradients)}
        a = native.step(a, grads)
        b = new.step(b, grads)
        assert a["w"].to_flat_list() == b["w"].to_flat_list()
    # Native compatibility permits extra gradient keys and keeps moment state.
    assert new.step(b, {"w": qb.array([1.0, 1.0]), "extra": qb.array(0.0)})


def test_sgd_fused_update_matches_f64_intermediates_and_keeps_inputs():
    for dtype in (qb.float32, qb.float64):
        for gradient_dtype in (qb.float32, qb.float64):
            parameters = qb.array(
                [0.0, -0.0, 0.1, 1e20, math.inf, math.nan], dtype=dtype
            )
            gradient = qb.array(
                [-0.0, 0.0, 0.3, 1.0, math.inf, 1.0], dtype=gradient_dtype
            )
            parameter_bits = memoryview(parameters).tobytes()
            gradient_bits = memoryview(gradient).tobytes()
            expected = (
                parameters.astype(qb.float64) - gradient.astype(qb.float64) * 0.017
            ).astype(dtype)
            actual, state = SGD(0.017).update(parameters, gradient, None)
            assert state is None and actual.dtype == dtype
            assert memoryview(actual).tobytes() == memoryview(expected).tobytes()
            assert memoryview(parameters).tobytes() == parameter_bits
            assert memoryview(gradient).tobytes() == gradient_bits


def test_sgd_pytree_analytic_update():
    optimizer = SGD(0.25)
    params = (qb.array([2.0, -1.0]), {"empty": None, "bias": qb.array(4.0)})
    grads = (qb.array([4.0, -2.0]), {"empty": None, "bias": qb.array(8.0)})
    state = optimizer.init(params)
    updated, next_state = optimizer.update(params, grads, state)
    assert state is next_state is None
    assert updated[0].to_flat_list() == [1.0, -0.5]
    assert scalar(updated[1]["bias"]) == 2
    assert params[0].to_flat_list() == [2.0, -1.0]


def test_pure_adam_matches_native_trajectory_with_varying_gradients():
    for dtype in (qb.float32, qb.float64):
        optimizer = Adam(0.03, b1=0.7, b2=0.95, eps=0.001)
        native = qb._quabla.Adam(0.03, 0.7, 0.95, 0.001)
        functional = legacy = {"w": qb.array([1.0, -2.0], dtype=dtype)}
        state = optimizer.init(functional)
        for values in ([2.0, -4.0], [-1.0, 0.0], [0.5, 1.0], [0.0, -3.0]):
            gradients = {"w": qb.array(values, dtype=dtype)}
            functional, state = optimizer.update(functional, gradients, state)
            legacy = native.step(legacy, gradients)
            assert functional["w"].dtype == dtype
            for actual, expected in zip(
                functional["w"].to_flat_list(), legacy["w"].to_flat_list()
            ):
                assert math.isclose(actual, expected, abs_tol=1e-14)


def test_optimizer_invalid_hyperparameters_structures_and_state():
    for value in (0, -1, math.nan, math.inf):
        raises(ValueError, Adam, learning_rate=value)
        raises(ValueError, SGD, learning_rate=value)
        raises(ValueError, Adam, eps=value)
    for value in (-0.1, 1, math.nan, math.inf):
        raises(ValueError, Adam, b1=value)
        raises(ValueError, Adam, b2=value)
    for optimizer in (Adam(), SGD()):
        raises(ValueError, optimizer.init, {})
        raises(TypeError, optimizer.init, qb.array(True))
        params = {"w": qb.array([1.0])}
        state = optimizer.init(params)
        raises(ValueError, optimizer.update, params, {"other": qb.array([1.0])}, state)
        raises(ValueError, optimizer.update, params, {"w": qb.array([1.0, 2.0])}, state)
        raises(TypeError, optimizer.update, params, {"w": qb.array([True])}, state)
    optimizer = Adam()
    params = qb.array([1.0])
    state = optimizer.init(params)
    raises(ValueError, optimizer.update, params, params, {**state, "step": -1})
    raises(
        ValueError,
        optimizer.update,
        params,
        params,
        {**state, "m": qb.array([1.0, 2.0])},
    )


def loss(params, x, target):
    return ((x * params["layers"][0] + params["bias"] - target).powi(2)).mean()


def train(device, optimizer):
    params = {"layers": [qb.array([0.0])], "bias": qb.array([0.0])}
    x, target = qb.array([-1.0, 1.0]), qb.array([-1.0, 3.0])
    trainer = Trainer(
        loss, params, optimizer, x, target, device=device, batch_argnums=(0, 1)
    )
    initial = scalar(trainer.loss())
    trainer.step(qb.array([-2.0, 2.0]), qb.array([-3.0, 5.0]))
    for _ in range(299):
        trainer.step()
    assert scalar(trainer.loss()) < initial * 1e-5
    assert abs(scalar(trainer.params["layers"][0]) - 2) < 0.01
    assert abs(scalar(trainer.params["bias"]) - 1) < 0.01
    return trainer


def test_cpu_trainer_adam_and_sgd_convergence():
    train("cpu", Adam(0.05))
    train("cpu", SGD(0.05))


@dataclasses.dataclass(frozen=True)
class Affine:
    weight: object
    bias: object
    name: str = "affine"


qb.tree.register_dataclass(Affine, meta_fields=("name",))
Pair = collections.namedtuple("Pair", "first second")


def test_optimizers_and_trainer_keep_custom_node_parameters():
    def affine_loss(params, x, target):
        layer = params.first
        prediction = layer.weight * x + layer.bias + params.second
        return ((prediction - target) ** 2).mean()

    params = Pair(Affine(qb.array([0.0]), qb.array([0.0])), qb.array(0.0))
    grads = Pair(Affine(qb.array([1.0]), qb.array([2.0])), qb.array(4.0))
    updated, state = SGD(0.5).update(params, grads, None)
    assert type(updated) is Pair and type(updated.first) is Affine
    assert updated.first.name == "affine"
    assert scalar(updated.first.bias) == -1.0 and scalar(updated.second) == -2.0
    adam = Adam(0.1)
    updated, state = adam.update(params, grads, adam.init(params))
    assert type(state["m"]) is Pair and type(state["v"].first) is Affine
    raises(ValueError, adam.update, params, (grads.first, grads.second), state)
    x, target = qb.array([-1.0, 1.0]), qb.array([-1.0, 3.0])
    trainer = Trainer(affine_loss, params, Adam(0.05), x, target)
    initial = scalar(trainer.loss())
    for _ in range(300):
        trainer.step()
    assert scalar(trainer.loss()) < initial * 1e-5
    final = trainer.params
    assert type(final) is Pair and type(final.first) is Affine
    assert abs(scalar(final.first.weight) - 2) < 0.01
    assert abs(scalar(final.first.bias) + scalar(final.second) - 1) < 0.01
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda:0", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue
        on_device = Trainer(affine_loss, params, Adam(0.05), x, target, device=device)
        for _ in range(300):
            on_device.step()
        assert type(on_device.params.first) is Affine, device
        assert abs(scalar(on_device.params.first.weight) - 2) < 0.01, device


def test_trainer_retained_data_batch_order_and_latest_loss():
    def objective(p, retained, x, y):
        return ((p * x - y + retained).powi(2)).mean()

    trainer = Trainer(
        objective,
        qb.array([0.0]),
        SGD(0.1),
        qb.array([1.0]),
        qb.array([2.0]),
        qb.array([0.0]),
        batch_argnums=(2, 1),
    )
    trainer.step(qb.array([3.0]), qb.array([4.0]))
    assert math.isclose(scalar(trainer.params), 1.6)
    assert math.isclose(scalar(trainer.loss()), (1.6 * 4 - 3 + 1) ** 2)
    before = scalar(trainer.params)
    raises(ValueError, trainer.step, qb.array([1.0]))
    raises(ValueError, trainer.step, qb.array([1.0, 2.0]), qb.array([4.0]))
    assert scalar(trainer.params) == before
    raises(
        ValueError,
        Trainer,
        objective,
        qb.array([0.0]),
        SGD(),
        qb.array([1.0]),
        batch_argnums=(1,),
    )
    raises(
        ValueError,
        Trainer,
        objective,
        qb.array([0.0]),
        SGD(),
        qb.array([1.0]),
        batch_argnums=(0, 0),
    )


def test_cpu_failed_step_preserves_latest_data_parameters_and_state():
    trainer = Trainer(
        lambda params, x: params["w"] * x,
        {"w": qb.array([1.0, 2.0])},
        Adam(),
        qb.array([3.0, 4.0]),
        batch_argnums=(0,),
    )
    assert trainer.loss().to_flat_list() == [3.0, 8.0]
    raises(ValueError, trainer.step, qb.array([5.0, 6.0]))
    assert trainer.loss().to_flat_list() == [3.0, 8.0]
    assert trainer.params["w"].to_flat_list() == [1.0, 2.0]
    assert trainer._state["step"] == 0


def test_device_trainer_rejects_sgd():
    for device in ("mlx", "cuda:0"):
        raises(
            qb.UnsupportedOperationError,
            Trainer,
            lambda p: p.sum(),
            qb.array([1.0]),
            SGD(),
            device=device,
        )


def test_device_factory_retains_only_fixed_data_and_reuses_native_step():
    calls = []

    class Executor:
        def step(self, *args):
            calls.append(args)

        def parameters(self):
            return {name: captured["inputs"][name] for name in captured["parameters"]}

    captured = {}

    def factory(traced, names, inputs, lr, retained, b1, b2, eps):
        captured.update(
            parameters=names,
            inputs=inputs,
            retained=retained,
            hyperparameters=(lr, b1, b2, eps),
        )
        return Executor()

    def objective(params, fixed, batch):
        return ((params["layers"][0] * batch["x"] + fixed).powi(2)).mean()

    with (
        patch("quabla._devices.require_device", return_value=("mlx", 0)),
        patch.object(qb._quabla, "mlx_adam_loss_optimizer", factory),
    ):
        trainer = Trainer(
            objective,
            {"layers": [qb.array([1.0])]},
            Adam(0.05),
            qb.array([2.0]),
            {"x": qb.array([3.0])},
            device="mlx",
            batch_argnums=(1,),
        )
        assert captured["parameters"] == ["params/layers/0"]
        assert captured["retained"] == ["fixed"]
        assert captured["hyperparameters"] == (0.05, 0.9, 0.999, 1e-8)
        trainer.step({"x": qb.array([4.0])})
        assert set(calls[-1][0]) == {"batch/x"}
        assert scalar(calls[-1][0]["batch/x"]) == 4
        with patch(
            "quabla.tree.flatten", side_effect=AssertionError("unexpected flatten")
        ):
            trainer.step()
        assert calls[-1] == ()
        assert scalar(trainer.params["layers"][0]) == 1


def test_mlx_trainer_convergence():
    if os.environ.get("QUABLA_MLX_TEST") == "1":
        train("mlx", Adam(0.05))


def test_cuda_trainer_convergence():
    if os.environ.get("QUABLA_CUDA_TEST") == "1":
        train("cuda:0", Adam(0.05))


def test_device_trainer_keeps_parameters_the_loss_ignores():
    # A parameter the loss does not depend on (here, an output bias that the
    # second derivative removes) is pruned from the compiled gradient plan;
    # device trainers must still accept it and leave it unchanged, like CPU.
    def u(params, x):
        return (qb.tanh(x * params["w1"]) * params["w2"]).sum() + params["b2"]

    def loss(params, xs):
        second = qb.vmap(lambda x: qb.grad(qb.grad(lambda z: u(params, z)))(x))(xs)
        return (second**2).sum()

    params = {
        "w1": qb.array([0.5, -0.3], dtype=qb.float32),
        "w2": qb.array([1.0, 0.7], dtype=qb.float32),
        "b2": qb.array(0.2, dtype=qb.float32),
    }
    xs = qb.array([0.1, 0.4, 0.9], dtype=qb.float32)
    for device, flag in (("mlx", "QUABLA_MLX_TEST"), ("cuda:0", "QUABLA_CUDA_TEST")):
        if os.environ.get(flag) != "1":
            continue
        expected = Trainer(loss, params, Adam(0.01), xs)
        actual = Trainer(loss, params, Adam(0.01), xs, device=device)
        for _ in range(2):
            expected.step()
            actual.step()
        assert actual.params["b2"].tolist() == params["b2"].tolist()
        for name in ("w1", "w2"):
            for got, want in zip(
                actual.params[name].tolist(), expected.params[name].tolist()
            ):
                assert math.isclose(got, want, rel_tol=1e-5), (device, name, got, want)


def test_trainer_releases_replaced_batch_payload():
    import sys

    original = qb.ones([16])
    references = sys.getrefcount(original)
    trainer = Trainer(
        lambda w, batch: ((w * batch) ** 2).mean(),
        qb.array(0.5),
        SGD(0.01),
        original,
        batch_argnums=(0,),
    )
    trainer.step(qb.full([16], 2.0))
    assert sys.getrefcount(original) == references
    assert math.isclose(float(trainer.loss()), (0.5 - 0.01 * 4.0) ** 2 * 4.0)


def test_public_namespace_is_all():
    import quabla.optim as optim

    public = sorted(name for name in dir(optim) if not name.startswith("_"))
    assert public == ["Adam", "SGD", "Trainer"], public
    assert sorted(optim.__all__) == public


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            if (
                name == "test_mlx_trainer_convergence"
                and os.environ.get("QUABLA_MLX_TEST") != "1"
            ):
                print(f"SKIP {name} (set QUABLA_MLX_TEST=1)")
                continue
            if (
                name == "test_cuda_trainer_convergence"
                and os.environ.get("QUABLA_CUDA_TEST") != "1"
            ):
                print(f"SKIP {name} (set QUABLA_CUDA_TEST=1)")
                continue
            test()
            print(f"PASS {name}")
