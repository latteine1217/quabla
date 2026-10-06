"""Analytic optimizer updates and CPU/device Trainer behavior."""

import collections
import dataclasses
import math
from unittest.mock import patch

import quabla as qb
from quabla.optim import Adam, SGD, Trainer

from _support import devices, enabled, raises, require, run


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
    for device in ("mlx", "cuda:0"):
        if not enabled(device):
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


def test_trainer_precision_is_validated_like_jit():
    from quabla.optim import AdamW

    for device in ("cpu", "mlx", "cuda:0"):
        for bad in ("float32", "fp64", 64):
            raises(
                ValueError,
                Trainer,
                lambda p: p.sum(),
                qb.array([1.0]),
                Adam(),
                device=device,
                precision=bad,
            )
    # MLX has no float64 arithmetic; rejected before any device is touched.
    raises(
        qb.UnsupportedOperationError,
        Trainer,
        lambda p: p.sum(),
        qb.array([1.0]),
        Adam(),
        device="mlx",
        precision="float64",
    )
    # The CPU already trains in float64, so the opt-in changes nothing.
    params = {"layers": [qb.array([0.5])], "bias": qb.array([-0.25])}
    x, target = qb.array([-1.0, 1.0]), qb.array([-1.0, 3.0])
    default = Trainer(loss, params, AdamW(0.05, clip_norm=0.5), x, target)
    explicit = Trainer(
        loss, params, AdamW(0.05, clip_norm=0.5), x, target, precision="float64"
    )
    for _ in range(5):
        default.step()
        explicit.step()
    assert default.params["bias"].tolist() == explicit.params["bias"].tolist()


def test_device_factory_retains_only_fixed_data_and_reuses_native_step():
    from quabla.optim import AdamW

    calls = []

    class Executor:
        def step(self, *args):
            calls.append(args)

        def parameters(self):
            return {name: captured["inputs"][name] for name in captured["parameters"]}

    captured = {}

    def factory(traced, names, inputs, retained, **options):
        captured.update(
            parameters=names,
            inputs=inputs,
            retained=retained,
            options=options,
        )
        return Executor()

    def objective(params, fixed, batch):
        return ((params["layers"][0] * batch["x"] + fixed).powi(2)).mean()

    adam_options = {
        "optimizer": "adam",
        "learning_rate": 0.05,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "clip_norm": None,
    }
    with (
        patch("quabla._devices.require_device", return_value=("mlx", 0)),
        patch.object(qb._quabla, "_mlx_trainer_optimizer", factory),
    ):
        # AdamW's decay, SGD, and clip_norm reach the native executor as is.
        for optimizer, options in (
            (
                AdamW(0.01, weight_decay=0.2, clip_norm=3.0),
                dict(
                    adam_options,
                    learning_rate=0.01,
                    weight_decay=0.2,
                    clip_norm=3.0,
                ),
            ),
            (
                SGD(0.5, clip_norm=1.5),
                dict(
                    adam_options,
                    optimizer="sgd",
                    learning_rate=0.5,
                    beta1=0.0,
                    beta2=0.0,
                    epsilon=1.0,
                    clip_norm=1.5,
                ),
            ),
        ):
            Trainer(
                objective,
                {"layers": [qb.array([1.0])]},
                optimizer,
                qb.array([2.0]),
                {"x": qb.array([3.0])},
                device="mlx",
            )
            assert captured["options"] == options, captured["options"]
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
        assert captured["options"] == adam_options, captured["options"]
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
    from quabla.optim import AdamW

    require("mlx")
    train("mlx", Adam(0.05))
    train("mlx", AdamW(0.05, weight_decay=1e-5, clip_norm=1.0))
    train("mlx", SGD(0.05, clip_norm=5.0))


def test_cuda_trainer_convergence():
    from quabla.optim import AdamW

    require("cuda")
    train("cuda:0", Adam(0.05))
    train("cuda:0", AdamW(0.05, weight_decay=1e-5, clip_norm=1.0))
    train("cuda:0", SGD(0.05, clip_norm=5.0))


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
    for device in devices("mlx", "cuda:0"):
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


def test_schedules_match_hand_computed_optax_values():
    from quabla.optim import (
        constant,
        cosine_decay,
        exponential_decay,
        piecewise_constant,
        warmup_cosine_decay,
    )

    def close(schedule, expected):
        for step, value in expected.items():
            got = schedule(step)
            assert math.isclose(got, value, rel_tol=1e-15, abs_tol=1e-15), (
                step,
                got,
                value,
            )

    close(constant(0.5), {0: 0.5, 7: 0.5})
    close(
        exponential_decay(1.0, 10, 0.5),
        {0: 1.0, 5: math.sqrt(0.5), 10: 0.5, 20: 0.25},
    )
    close(exponential_decay(1.0, 10, 0.5, staircase=True), {9: 1.0, 15: 0.5})
    close(
        exponential_decay(1.0, 10, 0.5, transition_begin=5),
        {0: 1.0, 5: 1.0, 15: 0.5},
    )
    close(exponential_decay(1.0, 10, 0.5, end_value=0.3), {10: 0.5, 20: 0.3})
    close(exponential_decay(1.0, 10, 2.0, end_value=3.0), {10: 2.0, 20: 3.0})
    # 2 * (0.9 * 0.5 * (1 + cos(pi t / 100)) + 0.1)
    close(
        cosine_decay(2.0, 100, alpha=0.1),
        {
            0: 2.0,
            25: 2 * (0.9 * 0.5 * (1 + math.sqrt(0.5)) + 0.1),
            50: 1.1,
            100: 0.2,
            150: 0.2,
        },
    )
    # decay_steps includes the warmup: peak at 10, end value from 110 on.
    close(
        warmup_cosine_decay(0.0, 1.0, 10, 110, end_value=0.1),
        {0: 0.0, 5: 0.5, 10: 1.0, 60: 0.55, 110: 0.1, 200: 0.1},
    )
    close(warmup_cosine_decay(0.5, 1.0, 0, 100), {0: 1.0, 50: 0.5, 100: 0.0})
    close(
        piecewise_constant([10, 20], [1.0, 0.1, 0.01]),
        {0: 1.0, 9: 1.0, 10: 0.1, 19: 0.1, 20: 0.01, 99: 0.01},
    )
    raises(ValueError, exponential_decay, 1.0, 0, 0.5)
    raises(ValueError, exponential_decay, 1.0, 10, 0.0)
    raises(ValueError, cosine_decay, 1.0, 0)
    raises(ValueError, warmup_cosine_decay, 0.0, 1.0, 10, 10)
    raises(ValueError, piecewise_constant, [10, 10], [1.0, 2.0, 3.0])
    raises(ValueError, piecewise_constant, [10], [1.0])


def test_adamw_decoupled_decay_against_reference():
    from quabla.optim import AdamW

    # One step by hand: m_hat = 2, v_hat = 4, so the Adam direction is 2/2.01;
    # the decay adds weight_decay * p with the pre-update p.
    params = {"w": qb.array([3.0]), "v": qb.array([-1.0], dtype=qb.float32)}
    grads = {"w": qb.array([2.0]), "v": qb.array([-4.0], dtype=qb.float32)}
    optimizer = AdamW(0.1, b1=0.5, b2=0.75, eps=0.01, weight_decay=0.5)
    updated, state = optimizer.update(params, grads, optimizer.init(params))
    assert math.isclose(scalar(updated["w"]), 3 - 0.1 * (2 / 2.01 + 0.5 * 3))
    assert updated["v"].dtype == qb.float32
    assert state["m"]["v"].dtype == qb.float64 and state["step"] == 1

    # Multi-step f64 trajectory against a plain-Python reference.
    lr, b1, b2, eps, wd = 0.03, 0.8, 0.95, 1e-3, 0.2
    p, m, v = [1.0, -2.0], [0.0, 0.0], [0.0, 0.0]
    params = qb.array(p)
    optimizer = AdamW(lr, b1, b2, eps, wd)
    state = optimizer.init(params)
    for k, g in enumerate(([2.0, -4.0], [-1.0, 0.0], [0.5, 1.0]), start=1):
        params, state = optimizer.update(params, qb.array(g), state)
        for i in range(2):
            m[i] = b1 * m[i] + (1 - b1) * g[i]
            v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
            direction = (m[i] / (1 - b1**k)) / (math.sqrt(v[i] / (1 - b2**k)) + eps)
            p[i] -= lr * (direction + wd * p[i])
        for got, want in zip(params.to_flat_list(), p):
            assert math.isclose(got, want, rel_tol=1e-14), (got, want)

    # Zero decay is bitwise Adam, including infinite parameters.
    params = qb.array([0.5, math.inf, -3.0], dtype=qb.float32)
    grads = qb.array([1.0, 1.0, -2.0], dtype=qb.float32)
    adam, adamw = Adam(0.1), AdamW(0.1, weight_decay=0.0)
    a, _ = adam.update(params, grads, adam.init(params))
    w, _ = adamw.update(params, grads, adamw.init(params))
    assert memoryview(a).tobytes() == memoryview(w).tobytes()
    for value in (-1e-3, math.nan, math.inf):
        raises(ValueError, AdamW, weight_decay=value)
    raises(AttributeError, getattr, AdamW(), "step")


def test_schedules_drive_adam_and_sgd_at_the_completed_step():
    from quabla.optim import piecewise_constant

    calls = []

    def schedule(step):
        calls.append(step)
        return [0.1, 0.01][min(step, 1)]

    # b1 = b2 = 0 makes the Adam direction g / (|g| + eps).
    optimizer = Adam(schedule, b1=0.0, b2=0.0, eps=1e-8)
    params = qb.array([1.0])
    state = optimizer.init(params)
    expected = 1.0
    for rate in (0.1, 0.01, 0.01):
        params, state = optimizer.update(params, qb.array([2.0]), state)
        expected -= rate * 2 / (2 + 1e-8)
        assert math.isclose(scalar(params), expected, rel_tol=1e-15)
    assert calls == [0, 1, 2]
    raises(TypeError, optimizer.step, {"w": params}, {"w": params})

    sgd = SGD(piecewise_constant([1], [0.5, 0.25]))
    params, state = qb.array([1.0]), sgd.init(qb.array([1.0]))
    assert state == {"step": 0}
    params, state = sgd.update(params, qb.array([2.0]), state)
    assert scalar(params) == 0.0 and state == {"step": 1}
    params, state = sgd.update(params, qb.array([2.0]), state)
    assert scalar(params) == -0.5 and state == {"step": 2}
    raises(ValueError, sgd.update, params, params, None)
    raises(ValueError, sgd.update, params, params, {"step": -1})
    # A schedule must yield finite nonnegative rates.
    for bad in (-1.0, math.nan, math.inf):
        broken = SGD(lambda step, bad=bad: bad)
        raises(ValueError, broken.update, params, params, broken.init(params))


def test_clip_by_global_norm_scaling_overflow_and_nonfinite():
    from quabla.optim import clip_by_global_norm

    grads = {"a": qb.array([3.0, 0.0]), "b": (qb.array(4.0), None)}
    same, norm = clip_by_global_norm(grads, 10.0)
    assert norm == 5.0 and same is grads
    clipped, norm = clip_by_global_norm(grads, 1.0)
    assert norm == 5.0
    assert math.isclose(clipped["a"].to_flat_list()[0], 0.6)
    assert clipped["a"].to_flat_list()[1] == 0.0
    assert math.isclose(scalar(clipped["b"][0]), 0.8) and clipped["b"][1] is None

    # Scaling by max |g| avoids the overflow and underflow of sum(g * g).
    for magnitude in (1e200, 1e-200):
        _, norm = clip_by_global_norm([qb.array([magnitude, magnitude])], 1e300)
        assert math.isclose(norm, math.sqrt(2) * magnitude, rel_tol=1e-15)
    clipped, norm = clip_by_global_norm([qb.array([1e200, -1e200])], 1.0)
    for value, sign in zip(clipped[0].to_flat_list(), (1, -1)):
        assert math.isclose(value, sign * math.sqrt(0.5), rel_tol=1e-15)
    # float32 leaves are measured in float64 and keep their dtype.
    clipped, norm = clip_by_global_norm([qb.array([3e30, 4e30], dtype=qb.float32)], 1.0)
    assert clipped[0].dtype == qb.float32
    assert math.isclose(norm, 5e30, rel_tol=1e-7)
    assert math.isclose(clipped[0].to_flat_list()[1], 0.8, rel_tol=1e-7)
    zero, norm = clip_by_global_norm([qb.zeros([3])], 1.0)
    assert norm == 0.0 and zero[0].to_flat_list() == [0.0] * 3

    # Non-finite norms never become finite clipped values.
    for bad in (math.nan, math.inf, -math.inf):
        clipped, norm = clip_by_global_norm(
            {"a": qb.array([bad, 1.0]), "b": qb.array([2.0])}, 1.0
        )
        assert not math.isfinite(norm)
        assert math.isnan(norm) == math.isnan(bad)
        assert all(math.isnan(x) for x in clipped["a"].to_flat_list())
        assert all(math.isnan(x) for x in clipped["b"].to_flat_list())
    for value in (0.0, -1.0, math.nan, math.inf):
        raises(ValueError, clip_by_global_norm, grads, value)
    raises(TypeError, clip_by_global_norm, [qb.array([True])], 1.0)


def test_clip_norm_option_clips_before_the_update():
    from quabla.optim import AdamW, clip_by_global_norm

    params = {"w": qb.array([1.0, -1.0]), "b": qb.array(0.5)}
    grads = {"w": qb.array([30.0, -40.0]), "b": qb.array(0.0)}
    clipped, _ = clip_by_global_norm(grads, 2.0)
    for make in (
        lambda clip: Adam(0.1, clip_norm=clip),
        lambda clip: AdamW(0.1, weight_decay=0.1, clip_norm=clip),
        lambda clip: SGD(0.1, clip_norm=clip),
    ):
        reference = make(None)
        expected, _ = reference.update(params, clipped, reference.init(params))
        optimizer = make(2.0)
        actual, _ = optimizer.update(params, grads, optimizer.init(params))
        for name in ("w", "b"):
            assert actual[name].to_flat_list() == expected[name].to_flat_list()
    for value in (0.0, -1.0, math.nan):
        raises(ValueError, Adam, clip_norm=value)
        raises(ValueError, SGD, clip_norm=value)
    raises(TypeError, Adam(clip_norm=1.0).step, {"w": qb.array([1.0])}, {})


def test_cpu_trainer_adamw_and_schedules():
    from quabla.optim import AdamW, warmup_cosine_decay

    train("cpu", AdamW(0.05, weight_decay=1e-5))
    train("cpu", Adam(warmup_cosine_decay(0.0, 0.08, 20, 400, end_value=0.01)))
    rates = []

    def schedule(step):
        rates.append(step)
        return 0.05

    trainer = train("cpu", SGD(schedule))
    assert rates == list(range(300)) and trainer._state == {"step": 300}


def test_device_trainer_sets_scheduled_rates():
    from quabla.optim import piecewise_constant

    events = []

    class Executor:
        learning_rate = None

        def __setattr__(self, name, value):
            events.append(("set", value))
            object.__setattr__(self, name, value)

        def step(self, *args):
            events.append(("step", self.learning_rate))

    def factory(traced, names, inputs, retained, **options):
        events.append(("create", options["learning_rate"]))
        return Executor()

    with (
        patch("quabla._devices.require_device", return_value=("mlx", 0)),
        patch.object(qb._quabla, "_mlx_trainer_optimizer", factory),
    ):
        trainer = Trainer(
            lambda p, x: (p * x).sum(),
            qb.array([1.0]),
            Adam(piecewise_constant([1, 2], [0.0, 0.5, 0.25])),
            qb.array([2.0]),
            device="mlx",
            batch_argnums=(0,),
        )
        trainer.step()
        trainer.step(qb.array([3.0]))
        trainer.step()
    assert events == [
        ("create", 0.0),
        ("set", 0.0),
        ("step", 0.0),
        ("set", 0.5),
        ("step", 0.5),
        ("set", 0.25),
        ("step", 0.25),
    ]


def test_mlx_trainer_schedule_matches_cpu():
    require("mlx")
    from quabla.optim import warmup_cosine_decay

    schedule = warmup_cosine_decay(0.0, 0.05, 3, 12, end_value=0.005)
    params = {"layers": [qb.array([0.0])], "bias": qb.array([0.0])}
    x, target = qb.array([-1.0, 1.0]), qb.array([-1.0, 3.0])
    cpu = Trainer(loss, params, Adam(schedule), x, target)
    mlx = Trainer(loss, params, Adam(schedule), x, target, device="mlx")
    assert mlx._executor.learning_rate == 0.0
    for _ in range(12):
        cpu.step()
        mlx.step()
    assert math.isclose(mlx._executor.learning_rate, schedule(11), rel_tol=1e-6)
    for got, want in (
        (mlx.params["bias"], cpu.params["bias"]),
        (mlx.params["layers"][0], cpu.params["layers"][0]),
    ):
        assert math.isclose(scalar(got), scalar(want), rel_tol=1e-5), (got, want)
    raises(ValueError, setattr, mlx._executor, "learning_rate", -1.0)
    train("mlx", Adam(warmup_cosine_decay(0.0, 0.08, 20, 400, end_value=0.01)))


def test_cuda_trainer_schedule():
    require("cuda")
    from quabla.optim import warmup_cosine_decay

    schedule = warmup_cosine_decay(0.0, 0.05, 3, 12, end_value=0.005)
    params = {"layers": [qb.array([0.0])], "bias": qb.array([0.0])}
    x, target = qb.array([-1.0, 1.0]), qb.array([-1.0, 3.0])
    cpu = Trainer(loss, params, Adam(schedule), x, target)
    cuda = Trainer(loss, params, Adam(schedule), x, target, device="cuda:0")
    for _ in range(12):
        cpu.step()
        cuda.step()
    for got, want in (
        (cuda.params["bias"], cpu.params["bias"]),
        (cuda.params["layers"][0], cpu.params["layers"][0]),
    ):
        assert math.isclose(scalar(got), scalar(want), rel_tol=1e-5), (got, want)
    raises(ValueError, setattr, cuda._executor, "learning_rate", math.nan)
    train("cuda:0", Adam(warmup_cosine_decay(0.0, 0.08, 20, 400, end_value=0.01)))


def pinn_model(params, x):
    hidden = qb.tanh(params["w1"] * x + params["b1"])
    return (hidden * params["w2"]).sum() + params["b2"]


def pinn_loss(params, xs):
    """1-D Poisson residual u'' = -pi^2 sin(pi x) with u(0) = u(1) = 0.

    ``params["ignored"]`` never enters the loss: its gradient is zero and the
    device plans prune it (the v0.2.3 regression).
    """
    second = qb.vmap(lambda x: qb.grad(qb.grad(lambda z: pinn_model(params, z)))(x))(xs)
    residual = second + math.pi**2 * qb.sin(math.pi * xs)
    zero, one = qb.array(0.0, dtype=xs.dtype), qb.array(1.0, dtype=xs.dtype)
    boundary = pinn_model(params, zero) ** 2 + pinn_model(params, one) ** 2
    return (residual**2).mean() + boundary


def pinn_problem(dtype):
    width = 6
    params = {
        "w1": qb.array([0.9 - 0.3 * i for i in range(width)], dtype=dtype),
        "b1": qb.array([0.1 * i - 0.2 for i in range(width)], dtype=dtype),
        "w2": qb.array([0.5 * (-1) ** i + 0.05 * i for i in range(width)], dtype=dtype),
        "b2": qb.array(0.05, dtype=dtype),
        "ignored": qb.array([1.5, -2.0], dtype=dtype),
    }
    batches = [
        qb.array([0.1 * i + 0.05 for i in range(10)], dtype=dtype),
        qb.array([0.1 * i + 0.02 for i in range(10)], dtype=dtype),
    ]
    return params, batches


def assert_trainers_agree(expected, actual, rel, absolute, label):
    for name, want in expected.params.items():
        got = actual.params[name]
        assert got.dtype == want.dtype, (label, name, got.dtype, want.dtype)
        for a, b in zip(got.reshape(-1).tolist(), want.reshape(-1).tolist()):
            assert math.isclose(a, b, rel_tol=rel, abs_tol=absolute), (label, name, a, b)


def pinn_parity_configs(norm):
    """Optimizer factories with clipping above and below the first global norm."""
    from quabla.optim import AdamW, warmup_cosine_decay

    def schedule():
        return warmup_cosine_decay(0.0, 0.02, 2, 10, end_value=0.002)

    return {
        "adam": lambda: Adam(0.01),
        "adam, clip not triggered": lambda: Adam(0.01, clip_norm=100 * norm),
        "adam, clip triggered": lambda: Adam(0.01, clip_norm=norm / 4),
        "adamw": lambda: AdamW(0.01, weight_decay=0.1),
        "adamw, clip not triggered": lambda: AdamW(
            0.01, weight_decay=0.1, clip_norm=100 * norm
        ),
        "adamw, clip triggered": lambda: AdamW(
            0.01, weight_decay=0.1, clip_norm=norm / 4
        ),
        "adam, schedule, clip": lambda: Adam(schedule(), clip_norm=norm / 3),
        "adamw, schedule, clip": lambda: AdamW(
            schedule(), weight_decay=0.05, clip_norm=norm / 3
        ),
        "sgd, schedule": lambda: SGD(schedule()),
        "sgd, clip triggered": lambda: SGD(0.001, clip_norm=norm / 4),
    }


def run_pinn_parity(device, dtype, rel, absolute, **options):
    from quabla.optim import clip_by_global_norm

    params, batches = pinn_problem(dtype)
    _, norm = clip_by_global_norm(qb.grad(pinn_loss)(params, batches[0]), 1.0)
    assert norm > 1.0, norm
    for label, make in pinn_parity_configs(norm).items():
        cpu = Trainer(pinn_loss, params, make(), batches[0], batch_argnums=(0,))
        device_trainer = Trainer(
            pinn_loss,
            params,
            make(),
            batches[0],
            device=device,
            batch_argnums=(0,),
            **options,
        )
        for step in range(8):
            # Replace the collocation batch on some steps, reuse it on others.
            batch = () if step % 3 else (batches[step % 2],)
            cpu.step(*batch)
            device_trainer.step(*batch)
            want, got = scalar(cpu.loss()), scalar(device_trainer.loss())
            assert math.isclose(got, want, rel_tol=rel), (label, step, got, want)
            assert_trainers_agree(cpu, device_trainer, rel, absolute, (label, step))
        ignored = device_trainer.params["ignored"].tolist()
        if "adamw" in label:
            # AdamW decays even parameters the loss ignores, like the CPU.
            assert ignored != params["ignored"].tolist(), label
        else:
            assert ignored == params["ignored"].tolist(), label


def test_device_trainer_matches_cpu_for_every_optimizer():
    # float32 device plans agree with the float64-accumulating CPU trainer to
    # float32 rounding; the trajectories differ by a few ulps per step.
    for device in devices("mlx", "cuda:0"):
        run_pinn_parity(device, qb.float32, rel=1e-6, absolute=1e-6)


def test_cuda_trainer_float64_precision():
    require("cuda")
    # Native double agrees with the CPU to double rounding, far below float32.
    run_pinn_parity("cuda:0", qb.float64, rel=1e-12, absolute=1e-13, precision="float64")
    params, batches = pinn_problem(qb.float64)
    trainer = Trainer(
        pinn_loss, params, Adam(0.01), batches[0], device="cuda:0", precision="float64"
    )
    trainer.step()
    # Parameters stay float64 on the device: no value is a float32 rounding.
    values = trainer.params["w1"].tolist()
    assert any(value != float(qb.array(value).astype(qb.float32)) for value in values)
    # One plan has one floating element type: float32 data cannot join a
    # float64 lowering, exactly as for jit.
    raises(
        qb.UnsupportedOperationError,
        Trainer,
        lambda p, x: ((p * x.astype(qb.float64)) ** 2).sum(),
        qb.array([1.0, 2.0]),
        Adam(0.01),
        qb.array([3.0, 4.0], dtype=qb.float32),
        device="cuda:0",
        precision="float64",
    )


def test_device_clip_norm_edge_cases_match_cpu():
    from quabla.optim import AdamW

    f32 = qb.float32

    def nan_gradient(p):
        return qb.sqrt(p["a"]).sum() + (p["b"] ** 2).sum()

    def infinite_gradient(p):
        return (1.0 / p["a"]).sum() + (p["b"] ** 2).sum()

    def scaled_quadratic(scale):
        return lambda p: scale * ((p["a"] ** 2).sum() + 1.5 * (p["b"] ** 2).sum())

    def two_parameters(a):
        # "ignored" never enters a loss: a non-finite clipped norm still makes
        # its zero gradient NaN on the CPU, which devices must reproduce.
        return {
            "a": qb.array(a, dtype=f32),
            "b": qb.array([2.0, 3.0], dtype=f32),
            "ignored": qb.array([0.5], dtype=f32),
        }

    cases = [
        # A NaN or infinite global norm turns every gradient into NaN; without
        # clipping only the non-finite parameter becomes NaN.
        ("nan, clipped", nan_gradient, two_parameters([-1.0]), Adam(0.1, clip_norm=1.0)),
        ("nan, unclipped", nan_gradient, two_parameters([-1.0]), Adam(0.1)),
        ("inf, clipped", infinite_gradient, two_parameters([0.0]), SGD(0.1, clip_norm=1.0)),
        # A zero norm leaves the gradients unchanged (no division by zero).
        (
            "zero norm",
            lambda p: ((p["a"] - 1.0) ** 2).sum(),
            {"a": qb.array([1.0, 1.0], dtype=f32)},
            AdamW(0.1, weight_decay=0.1, clip_norm=1.0),
        ),
        # |g| ~ 1e30: a float32 sum of squares would overflow to inf.
        (
            "huge gradients",
            scaled_quadratic(1e30),
            two_parameters([1.0, -0.5]),
            SGD(0.1, clip_norm=1.0),
        ),
        # |g| ~ 1e-25: a float32 sum of squares would underflow to zero and
        # skip the clip.
        (
            "tiny gradients",
            scaled_quadratic(1e-25),
            two_parameters([1.0, -0.5]),
            SGD(1e24, clip_norm=1e-25),
        ),
    ]
    for device in devices("mlx", "cuda:0"):
        for label, objective, params, optimizer in cases:
            cpu = Trainer(objective, params, optimizer)
            device_trainer = Trainer(objective, params, optimizer, device=device)
            for _ in range(2):
                cpu.step()
                device_trainer.step()
            for name, want in cpu.params.items():
                got = device_trainer.params[name].tolist()
                for a, b in zip(got, want.tolist()):
                    assert (math.isnan(a) and math.isnan(b)) or math.isclose(
                        a, b, rel_tol=1e-6, abs_tol=1e-6
                    ), (device, label, name, got, want.tolist())
        # Clipping changed the tiny-gradient step (factor 1e-25 / 1.1e-24):
        # unclipped SGD moves b by 0.6, clipped by 0.054, so the reduction did
        # not underflow to a zero norm.
        tiny = Trainer(scaled_quadratic(1e-25), two_parameters([1.0, -0.5]), SGD(1e24))
        tiny.step()
        clipped = Trainer(
            scaled_quadratic(1e-25),
            two_parameters([1.0, -0.5]),
            SGD(1e24, clip_norm=1e-25),
            device=device,
        )
        clipped.step()
        assert scalar(clipped.params["b"]) > scalar(tiny.params["b"]) + 0.1, device

        # A parameter larger than one reduction grid (256 blocks of 256
        # threads) exercises the grid-stride partial sums.
        size = 70_000
        weights = qb.array([math.sin(i) for i in range(size)], dtype=f32)

        def wide(p):
            return ((p["w"] - 0.5) ** 2).sum() + (p["v"] ** 2).sum()

        params = {"w": weights, "v": qb.array([3.0], dtype=f32)}
        cpu = Trainer(wide, params, SGD(0.01, clip_norm=10.0))
        device_trainer = Trainer(wide, params, SGD(0.01, clip_norm=10.0), device=device)
        for _ in range(3):
            cpu.step()
            device_trainer.step()
        assert_trainers_agree(cpu, device_trainer, 1e-5, 1e-6, (device, "wide"))


def rosenbrock(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def test_lbfgs_rosenbrock_converges_from_the_classic_start():
    from quabla.optim import LBFGS

    solution, info = LBFGS(tolerance_grad=1e-10).minimize(
        rosenbrock, qb.array([-1.2, 1.0])
    )
    x, y = solution.to_flat_list()
    assert abs(x - 1) < 1e-6 and abs(y - 1) < 1e-6, (x, y, info)
    assert info["converged"] and info["loss"] < 1e-12, info
    assert info["iterations"] < 60 and info["evaluations"] < 80, info
    try:
        from scipy.optimize import minimize
    except ImportError:
        return

    def fun(p):
        a, b = p
        value = (1 - a) ** 2 + 100 * (b - a * a) ** 2
        return value, [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)]

    reference = minimize(
        fun, [-1.2, 1.0], jac=True, method="L-BFGS-B", options={"gtol": 1e-10}
    )
    print(
        f"  rosenbrock: quabla {info['iterations']} it / {info['evaluations']} ev; "
        f"scipy L-BFGS-B {reference.nit} it / {reference.nfev} ev"
    )
    assert info["iterations"] <= 2 * reference.nit + 10, (info, reference.nit)


def test_lbfgs_ill_conditioned_quadratic_reaches_exact_minimizer():
    from quabla.optim import LBFGS

    n = 12
    # A = D + u u' with eigenvalues spanning about 1e4; b fixed.
    diagonal = [10.0 ** (4 * i / (n - 1)) for i in range(n)]
    u = [math.sin(i + 1.0) for i in range(n)]
    a = qb.array(
        [[diagonal[i] * (i == j) + u[i] * u[j] for j in range(n)] for i in range(n)]
    )
    b = qb.array([math.cos(3.0 * i) for i in range(n)])
    exact = qb.solve(a, b.reshape([n, 1])).reshape([n]).to_flat_list()

    def quadratic(x, a, b):
        return (
            0.5 * (x * qb.matmul(a, x.reshape([n, 1])).reshape([n])).sum()
            - (b * x).sum()
        )

    # With tolerances at zero the run ends where float64 loss differences
    # (about 1e-16 * |f|) can no longer certify a decrease, so the line search
    # stops; the iterate is then the minimizer to that resolution.
    scale = max(abs(e) for e in exact)
    results = {}
    for history in (10, 20):
        solution, info = LBFGS(
            history=history,
            max_iterations=1000,
            tolerance_grad=0.0,
            tolerance_change=0.0,
        ).minimize(quadratic, qb.zeros([n]), a, b)
        error = max(abs(s - e) for s, e in zip(solution.to_flat_list(), exact))
        results[history] = (error, info)
        assert info["reason"] == "line_search_failed", info
    print(f"  quadratic (cond ~7e3): {results}")
    assert results[20][0] < 1e-8 * scale and results[10][0] < 1e-7 * scale, results
    assert results[20][1]["iterations"] < 120, results
    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError:
        return
    matrix, vector = np.array(a.tolist()), np.array(b.tolist())
    reference = minimize(
        lambda x: (0.5 * x @ matrix @ x - vector @ x, matrix @ x - vector),
        np.zeros(n),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 0.0, "ftol": 0.0, "maxiter": 1000},
    )
    reference_error = float(np.abs(reference.x - np.array(exact)).max())
    print(f"  scipy L-BFGS-B m=10: {reference.nit} it, error {reference_error:.2e}")
    # Same memory: no more iterations than scipy needs (plus slack), and at
    # least comparable accuracy.
    assert results[10][1]["iterations"] <= 1.25 * reference.nit + 10, reference.nit
    assert results[10][0] <= 4 * reference_error + 1e-12, reference_error


def test_lbfgs_pytree_float32_and_nonfinite_backtracking():
    from quabla.optim import LBFGS

    # Linear least squares over a pytree with exact solution w = [1, -2], b = 3.
    xs = qb.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    ys = qb.array([1.0, 4.0, 2.0, 7.0])

    def residual(params, xs, ys):
        prediction = qb.matmul(xs, params["w"].reshape([2, 1])).reshape([4])
        return ((prediction + params["affine"][0] - ys) ** 2).sum()

    params = {"w": qb.array([0.0, 0.0]), "affine": (qb.array(0.0), None)}
    # The change test is purely relative, so a zero-residual fit is driven to
    # the gradient tolerance rather than stopped once |f| is below 1e-9.
    solution, info = LBFGS(tolerance_grad=1e-10).minimize(residual, params, xs, ys)
    assert info["reason"] == "gradient_tolerance", info
    assert solution["affine"][1] is None and info["converged"], info
    for got, want in zip(
        solution["w"].to_flat_list() + [scalar(solution["affine"][0])],
        [1.0, -2.0, 3.0],
    ):
        assert abs(got - want) < 1e-9, (solution, info)
    assert scalar(params["w"]) == 0.0

    # float32 parameters keep their dtype; the line search stops at the
    # float32 resolution instead of looping.
    solution, info = LBFGS(max_iterations=200).minimize(
        rosenbrock, qb.array([-1.2, 1.0], dtype=qb.float32)
    )
    assert solution.dtype == qb.float32, solution
    assert all(abs(v - 1) < 1e-2 for v in solution.to_flat_list()), (solution, info)
    assert info["reason"] != "max_iterations", info

    # f(x) = -10 x - log(1 - x) has its minimum at 0.9. From 0 the first
    # trial step (unit length along -g) lands on x = 1, where the loss is
    # infinite; the line search must backtrack into the domain.
    solution, info = LBFGS(tolerance_grad=1e-10, tolerance_change=0.0).minimize(
        lambda x: (-10 * x - qb.log(1 - x)).sum(), qb.array([0.0])
    )
    assert abs(scalar(solution) - 0.9) < 1e-9 and info["converged"], (solution, info)

    raises(ValueError, LBFGS, history=0)
    raises(ValueError, LBFGS, tolerance_grad=-1.0)
    raises(ValueError, LBFGS, line_search="backtracking")
    raises(TypeError, LBFGS().minimize, rosenbrock, qb.array([True]))
    raises(ValueError, LBFGS().minimize, lambda x: qb.log(x).sum(), qb.array([-1.0]))
    capped = LBFGS(max_evaluations=5).minimize(rosenbrock, qb.array([-1.2, 1.0]))[1]
    assert capped["reason"] == "max_evaluations" and capped["evaluations"] <= 5


def test_lbfgs_after_adam_on_a_pinn_like_problem():
    from quabla.optim import LBFGS

    # u'' = -pi^2 sin(pi x) on [0, 1] with u(0) = u(1) = 0, solved by a small
    # tanh network; u'' comes from nested grad inside vmap, as in PINNs.
    def u(params, x):
        return (qb.tanh(x * params["w1"] + params["b1"]) * params["w2"]).sum()

    def objective(params, xs):
        second = qb.vmap(lambda x: qb.grad(qb.grad(lambda z: u(params, z)))(x))(xs)
        source = qb.sin(xs * math.pi) * math.pi**2
        boundary = u(params, qb.array(0.0)) ** 2 + u(params, qb.array(1.0)) ** 2
        return ((second + source) ** 2).mean() + boundary

    width = 6
    params = {
        "w1": qb.array([math.sin(3.1 * k) for k in range(width)]),
        "b1": qb.array([math.cos(1.7 * k) for k in range(width)]),
        "w2": qb.array([0.3 * math.sin(2.3 * k + 1) for k in range(width)]),
    }
    xs = qb.linspace(0.0, 1.0, 17)
    trainer = Trainer(objective, params, Adam(0.02), xs)
    for _ in range(300):
        trainer.step()
    adam_loss = scalar(trainer.loss())
    solution, info = LBFGS(max_iterations=300).minimize(objective, trainer.params, xs)
    print(f"  pinn: adam {adam_loss:.3e} -> lbfgs {info['loss']:.3e} ({info})")
    assert info["loss"] < 1e-3 * adam_loss, (adam_loss, info)
    probe = [0.25, 0.5, 0.75]
    error = max(
        abs(u(solution, qb.array(x)).item() - math.sin(math.pi * x)) for x in probe
    )
    assert error < 1e-2, error


def test_public_namespace_is_all():
    import quabla.optim as optim

    public = sorted(name for name in dir(optim) if not name.startswith("_"))
    assert public == [
        "Adam",
        "AdamW",
        "LBFGS",
        "SGD",
        "Trainer",
        "clip_by_global_norm",
        "constant",
        "cosine_decay",
        "exponential_decay",
        "piecewise_constant",
        "warmup_cosine_decay",
    ], public
    assert sorted(optim.__all__) == public


if __name__ == "__main__":
    run(globals())
