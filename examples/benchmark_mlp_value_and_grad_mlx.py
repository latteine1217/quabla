"""Measure MLX batched MLP value-and-gradient specializations.

Run with the release extension on Apple silicon:

    .venv/bin/maturin develop --release --features mlx
    .venv/bin/python examples/benchmark_mlp_value_and_grad_mlx.py

The timed region calls ``tensor_value_and_grad_batch_mlx_fn``. It includes
dynamic input uploads and the loss/gradient host readbacks required by this
diagnostic API. It does not represent device-resident optimizer throughput.
"""

import argparse
import json
import math
import platform
import time

import quabla


def tensor(shape, values):
    return quabla.Tensor(shape, values)


def build_inputs(batch: int, width: int):
    features = 2
    x = [((index * 7) % 19 - 9) / 10.0 for index in range(batch * features)]
    target = [((index * 5) % 13 - 6) / 10.0 for index in range(batch)]
    w1 = [0.03 * (index + 1) for index in range(features * width)]
    b1 = [0.01 * (index - width // 2) for index in range(width)]
    w2 = [0.02 * (index + 1) for index in range(width)]
    return {
        "x": tensor([batch, features], x),
        "target": tensor([batch, 1], target),
        "w1": tensor([features, width], w1),
        "b1": tensor([1, width], b1),
        "w2": tensor([width, 1], w2),
        "b2": tensor([1, 1], [0.0]),
    }


def build_function(max_specializations: int):
    return quabla.tensor_value_and_grad_batch_mlx_fn(
        lambda x, target, w1, b1, w2, b2: (
            ((x.matmul(w1) + b1).tanh().matmul(w2) + b2 - target).powi(2)
        ).mean(),
        ["x", "target", "w1", "b1", "w2", "b2"],
        ["w1", "b1", "w2", "b2"],
        in_axes=[0, 0, None, None, None, None],
        max_specializations=max_specializations,
    )


def evaluate_or_raise(function, inputs):
    value, gradients = function(inputs)
    outputs = [value.to_flat_list()[0]]
    outputs.extend(component for gradient in gradients.values() for component in gradient.to_flat_list())
    if not all(math.isfinite(component) for component in outputs):
        raise RuntimeError("MLX MLP value-and-grad benchmark produced a non-finite output")
    return value.to_flat_list()[0]


def benchmark(iterations: int, warmup_iterations: int, batch: int, width: int) -> None:
    inputs = build_inputs(batch, width)
    function = build_function(max_specializations=1)
    compile_start = time.perf_counter()
    initial_loss = evaluate_or_raise(function, inputs)
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(warmup_iterations):
        evaluate_or_raise(function, inputs)
    warmup_loss = evaluate_or_raise(function, inputs)

    start = time.perf_counter()
    for _ in range(iterations):
        final_loss = evaluate_or_raise(function, inputs)
    seconds = time.perf_counter() - start

    print(
        "metadata="
        + json.dumps(
            {
                "backend": "mlx.gpu",
                "benchmark": "batched_two_layer_mlp_value_and_grad",
                "compile_seconds": compile_seconds,
                "dtype": "f32",
                "host": platform.platform(),
                "iterations": iterations,
                "shape": {"batch": batch, "features": 2, "hidden_width": width},
                "synchronization_policy": "each invocation materializes loss and requested gradients on host",
                "transfer_policy": "dynamic input tensors upload per invocation; diagnostic outputs read back per invocation",
                "warmup_iterations": warmup_iterations,
            },
            sort_keys=True,
        )
    )
    print(
        f"loss={initial_loss:.8e}->{warmup_loss:.8e}->{final_loss:.8e}, "
        f"compile={compile_seconds * 1e3:.2f} ms, specializations={function.specialization_count}"
    )
    print(
        f"value_and_grad: {iterations / seconds:.2f} calls/s "
        f"({seconds * 1e3 / iterations:.3f} ms/call), calls={iterations}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if min(args.iterations, args.warmup_iterations, args.batch, args.width) <= 0:
        parser.error("iterations, warmup-iterations, batch, and width must be positive")
    benchmark(args.iterations, args.warmup_iterations, args.batch, args.width)


if __name__ == "__main__":
    main()
