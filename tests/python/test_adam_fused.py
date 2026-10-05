"""Pure fused host Adam equivalence to the former eager expression."""

import math
import struct

import quabla as qb
from quabla.optim import Adam


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


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
