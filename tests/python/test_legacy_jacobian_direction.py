"""Legacy adaptive directions preserve callbacks, blocks and checked errors."""

import argparse
import importlib
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--site")
args = parser.parse_args()
if args.site:
    sys.path.insert(0, args.site)
quabla = importlib.import_module("quabla")
qb = importlib.import_module("quabla.legacy")


def test_numeric_expansion_and_callback_count():
    calls = []

    def model(x, unused):
        calls.append(1)
        value = (x + 0.25).sin().tanh()
        return quabla.concat([value] * 4, axis=1)

    x = qb.Matrix([[-0.0, 0.5]])
    unused = qb.Matrix([[3.0]])
    function = qb.jacobians_fn(model, [("x", (1, 2)), ("unused", (1, 1))])
    assert len(calls) == 1
    for _ in range(3):
        blocks = function({"x": x, "unused": unused})
        assert set(blocks) == {"x", "unused"}
        for row, values in enumerate(blocks["x"].to_list()):
            lane = row % 2
            shifted = [0.25, 0.75][lane]
            sine = math.sin(shifted)
            expected = math.cos(shifted) * (1 - math.tanh(sine) ** 2)
            for col, value in enumerate(values):
                reference = expected if col == lane else 0.0
                assert abs(value - reference) <= 1e-12 + 1e-12 * abs(reference)
        assert blocks["unused"].to_list() == [[0.0]] * 8
    assert len(calls) == 1
    assert x.to_list() == [[-0.0, 0.5]]
    assert unused.to_list() == [[3.0]]


def test_tiny_division_under_comparison_keeps_reverse_success():
    def model(x):
        value = (x / x).gt(0.0)
        return quabla.concat([value, value], axis=1)

    point = {"x": qb.Matrix([[1e-300]])}
    result = qb.jacobians_fn(model, [("x", (1, 1))])(point)
    assert result["x"].to_list() == [[0.0], [0.0]]
    try:
        qb.jvp_fn(model, [("x", (1, 1))])(point, {"x": qb.Matrix([[1.0]])})
    except ValueError as error:
        assert "division by zero" in str(error)
    else:
        raise AssertionError("general legacy JVP checked error changed")


if __name__ == "__main__":
    test_numeric_expansion_and_callback_count()
    test_tiny_division_under_comparison_keeps_reverse_success()
    print("PASS legacy numeric blocks/callback/alias and tiny-Gt reverse contract")
