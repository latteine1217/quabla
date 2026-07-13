"""Measure frozen rank-N CPU-plan throughput with deterministic inputs.

Build the extension in release mode before running this benchmark:
    .venv/bin/maturin develop --release
    .venv/bin/python examples/benchmark_tensor_cpu.py
"""

import argparse
import time

import nabla


def tensor(shape: list[int], offset: int) -> nabla.Tensor:
    count = 1
    for extent in shape:
        count *= extent
    data = [((index + offset) % 23 - 11) / 23.0 for index in range(count)]
    return nabla.Tensor(shape, data)


def benchmark(iterations: int, batch: int, width: int) -> None:
    traced = nabla.trace_tensor(
        lambda x, w1, b1, w2, b2: (((x @ w1 + b1).tanh() @ w2 + b2).sin()).mean(),
        [
            ("x", [batch, width]),
            ("w1", [width, width]),
            ("b1", [1, width]),
            ("w2", [width, width]),
            ("b2", [1, width]),
        ],
    )
    plan = traced.output.compile_cpu()
    inputs = {
        "x": tensor([batch, width], 0),
        "w1": tensor([width, width], 1),
        "b1": tensor([1, width], 2),
        "w2": tensor([width, width], 3),
        "b2": tensor([1, width], 4),
    }
    cotangent = nabla.Tensor([], [1.0])

    for _ in range(10):
        plan.evaluate(inputs)
        plan.evaluate_value_and_vjp(inputs, cotangent)

    start = time.perf_counter()
    primal_checksum = 0.0
    for _ in range(iterations):
        primal_checksum += plan.evaluate(inputs).to_flat_list()[0]
    primal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    gradient_checksum = 0.0
    for _ in range(iterations):
        value, gradients = plan.evaluate_value_and_vjp(inputs, cotangent)
        gradient_checksum += value.to_flat_list()[0] + gradients["w1"].to_flat_list()[0]
    vjp_seconds = time.perf_counter() - start

    print(f"shape=[{batch}, {width}], nodes={plan.node_count}, iterations={iterations}")
    print(
        f"primal: {iterations / primal_seconds:.2f} eval/s "
        f"({primal_seconds * 1e3 / iterations:.3f} ms/eval), checksum={primal_checksum:.12f}"
    )
    print(
        f"value_and_vjp: {iterations / vjp_seconds:.2f} eval/s "
        f"({vjp_seconds * 1e3 / iterations:.3f} ms/eval), checksum={gradient_checksum:.12f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch <= 0 or args.width <= 0:
        parser.error("iterations, batch, and width must be positive")
    benchmark(args.iterations, args.batch, args.width)


if __name__ == "__main__":
    main()
