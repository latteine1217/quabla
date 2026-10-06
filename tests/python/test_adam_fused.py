"""Pure fused host Adam equivalence to the former eager expression."""

import math
import os
import struct

import quabla as qb
from quabla.optim import Adam, AdamW, Trainer


def reference(parameter, gradient, state, optimizer):
    step = state["step"] + 1
    gradient = gradient.astype(qb.float64)
    m = state["m"].astype(qb.float64) * optimizer.b1 + gradient * (1 - optimizer.b1)
    v = state["v"].astype(qb.float64) * optimizer.b2 + gradient * gradient * (
        1 - optimizer.b2
    )
    delta = (m / (1 - optimizer.b1**step)) / (
        (v / (1 - optimizer.b2**step)).sqrt() + optimizer.eps
    )
    if optimizer.weight_decay:
        delta = delta + parameter.astype(qb.float64) * optimizer.weight_decay
    parameter = (parameter.astype(qb.float64) - delta * optimizer.learning_rate).astype(
        parameter.dtype
    )
    return parameter, {"step": step, "m": m, "v": v}


def assert_same(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    for a, b in zip(actual.to_flat_list(), expected.to_flat_list()):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("d", a) == struct.pack(
            "d", b
        )


def test_long_varying_gradient_trajectory_and_input_state_immutability():
    for dtype in (qb.float32, qb.float64):
        optimizer = Adam(0.031, b1=0.71, b2=0.953, eps=0.0013)
        parameters = qb.array([1.0, -2.0, 0.0, 1e-20], dtype=dtype).reshape(
            (1, 2, 1, 2, 1, 1)
        )
        expected = parameters
        state = optimizer.init(parameters)
        expected_state = state
        for step in range(250):
            gradients = qb.array(
                [math.sin(step), -math.cos(step), (step % 7 - 3) * 0.01, 1e-20],
                dtype=qb.float32 if step % 2 else qb.float64,
            ).reshape(parameters.shape)
            before = [x.to_flat_list() for x in (parameters, state["m"], state["v"])]
            previous_parameters, previous_state = parameters, state
            parameters, state = optimizer.update(parameters, gradients, state)
            expected, expected_state = reference(
                expected, gradients, expected_state, optimizer
            )
            assert_same(parameters, expected)
            assert_same(state["m"], expected_state["m"])
            assert_same(state["v"], expected_state["v"])
            assert state["m"].dtype == state["v"].dtype == qb.float64
            assert before == [
                x.to_flat_list()
                for x in (
                    previous_parameters,
                    previous_state["m"],
                    previous_state["v"],
                )
            ]


def test_nonfinite_values_and_float32_input_moments():
    for dtype in (qb.float32, qb.float64):
        optimizer = Adam(0.1, b1=0.5, b2=0.75, eps=0.01)
        parameters = qb.array([1.0, -2.0, 3.0, 4.0], dtype=dtype)
        gradients = qb.array([math.nan, math.inf, -math.inf, -0.0], dtype=dtype)
        state = {
            "step": 12,
            "m": qb.array([0.25, -0.5, 0.0, 0.1], dtype=qb.float32),
            "v": qb.array([0.1, 0.2, -1.0, 0.3], dtype=qb.float32),
        }
        actual, actual_state = optimizer.update(parameters, gradients, state)
        expected, expected_state = reference(parameters, gradients, state, optimizer)
        assert_same(actual, expected)
        assert_same(actual_state["m"], expected_state["m"])
        assert_same(actual_state["v"], expected_state["v"])



def test_device_adam_and_adamw_kernels_match_the_reference():
    """Device update kernels against the reference expression.

    The loss ``sum(g * p)`` has gradient exactly ``g``, so the device and the
    reference see the same gradients; a new ``g`` each step goes through
    batch replacement. Non-finite gradients propagate as on the CPU. Float32
    plans agree to a few float32 ulps, float64 plans to a few float64 ulps.
    """
    devices = [
        (device, dtype, options, tolerance)
        for device, flag, dtype, options, tolerance in (
            ("mlx", "QUABLA_MLX_TEST", qb.float32, {}, 1e-6),
            ("cuda:0", "QUABLA_CUDA_TEST", qb.float32, {}, 1e-6),
            ("cuda:0", "QUABLA_CUDA_TEST", qb.float64, {"precision": "float64"}, 1e-14),
        )
        if os.environ.get(flag) == "1"
    ]
    for device, dtype, options, tolerance in devices:
        for optimizer in (
            Adam(0.031, b1=0.71, b2=0.953, eps=0.0013),
            AdamW(0.031, b1=0.71, b2=0.953, eps=0.0013, weight_decay=0.37),
            # The default beta2 = 0.999, where 1 - beta2 cancels in float32.
            AdamW(0.031, weight_decay=0.37),
        ):
            parameters = qb.array([1.0, -2.0, 0.0, 1e-20, 4.0, 0.5], dtype=dtype)

            def gradients(step):
                values = [math.sin(step), -math.cos(step), (step % 7 - 3) * 0.01, 1e-20]
                # The last two lanes turn NaN and infinite on their steps.
                values += [math.nan if step == 5 else 0.25, math.inf if step == 9 else -1.0]
                return qb.array(values, dtype=dtype)

            trainer = Trainer(
                lambda p, g: (g * p).sum(),
                parameters,
                optimizer,
                gradients(0),
                device=device,
                batch_argnums=(0,),
                **options,
            )
            expected, state = parameters, optimizer.init(parameters)
            for step in range(12):
                trainer.step(gradients(step))
                expected, state = reference(expected, gradients(step), state, optimizer)
                got = trainer.params.tolist()
                for a, b in zip(got, expected.tolist()):
                    assert (math.isnan(a) and math.isnan(b)) or math.isclose(
                        a, b, rel_tol=tolerance, abs_tol=tolerance * 1e-2
                    ), (device, dtype, type(optimizer).__name__, step, got, expected.tolist())


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
