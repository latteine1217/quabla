"""Measure MLX batched matmul and two-layer MLP forward plans.

Run with the release extension on Apple silicon:

    .venv/bin/maturin develop --release --features mlx
    .venv/bin/python examples/benchmark_forward_mlx.py

Without --retain-static-inputs, the public primal plan uploads every host input
and materializes its output on each call. With retained static inputs, only the
dynamic inputs upload per call. Timings are therefore end-to-end diagnostics,
not device-only kernel throughput.
"""

import argparse
import json
import math
import platform
import time

import quabla


def tensor(shape, values):
    return quabla.Tensor(shape, values)


def values(count, multiplier):
    return [((index * multiplier) % 31 - 15) / 16.0 for index in range(count)]


def benchmark_plan(
    name,
    plan,
    inputs,
    iterations,
    warmup_iterations,
    shape,
    device_only,
    retained_input_names,
):
    retained_input_names = list(retained_input_names)
    if retained_input_names:
        plan.retain_inputs(inputs, retained_input_names)
    dynamic_inputs = {
        name: value for name, value in inputs.items() if name not in retained_input_names
    }
    compile_start = time.perf_counter()
    initial = plan(dynamic_inputs).to_flat_list()
    compile_seconds = time.perf_counter() - compile_start
    if not all(math.isfinite(value) for value in initial):
        raise RuntimeError(f"{name} produced a non-finite initial output")

    evaluate = plan.evaluate_device if device_only else lambda values: plan(values).to_flat_list()
    for _ in range(warmup_iterations):
        evaluate(dynamic_inputs)

    start = time.perf_counter()
    for _ in range(iterations):
        output = evaluate(dynamic_inputs)
    seconds = time.perf_counter() - start
    if not device_only and not all(math.isfinite(value) for value in output):
        raise RuntimeError(f"{name} produced a non-finite output")

    if retained_input_names:
        transfer_policy = (
            "retained inputs upload once; dynamic input tensors upload per invocation; "
            "output readback excluded"
            if device_only
            else "retained inputs upload once; dynamic input tensors upload per invocation; "
            "output reads back per invocation"
        )
    else:
        transfer_policy = (
            "all input tensors upload per invocation; output readback excluded"
            if device_only
            else "all input tensors upload per invocation; output reads back per invocation"
        )

    print(
        "metadata="
        + json.dumps(
            {
                "backend": "mlx.gpu",
                "benchmark": name,
                "compile_seconds": compile_seconds,
                "dtype": "f32",
                "host": platform.platform(),
                "iterations": iterations,
                "shape": shape,
                "synchronization_policy": "each invocation evaluates the MLX GPU output",
                "transfer_policy": transfer_policy,
                "retained_input_names": retained_input_names,
                "warmup_iterations": warmup_iterations,
            },
            sort_keys=True,
        )
    )
    print(
        f"{name}{'_device_only' if device_only else ''}: {iterations / seconds:.2f} calls/s "
        f"({seconds * 1e3 / iterations:.3f} ms/call), "
        f"compile={compile_seconds * 1e3:.2f} ms, "
        f"output_elements={0 if device_only else len(output)}"
    )


def benchmark(
    iterations,
    warmup_iterations,
    batch,
    features,
    width,
    outputs,
    device_only,
    retain_static_inputs,
):
    matmul_inputs = {
        "x": tensor([batch, features], values(batch * features, 7)),
        "weight": tensor([features, outputs], values(features * outputs, 11)),
    }
    matmul = quabla.trace_tensor(
        lambda x, weight: x.matmul(weight),
        [("x", [batch, features]), ("weight", [features, outputs])],
    ).output.compile_mlx()
    benchmark_plan(
        "batched_matmul_forward",
        matmul,
        matmul_inputs,
        iterations,
        warmup_iterations,
        {"batch": batch, "features": features, "outputs": outputs},
        device_only,
        ["weight"] if retain_static_inputs else [],
    )

    mlp_inputs = {
        "x": tensor([batch, features], values(batch * features, 7)),
        "w1": tensor([features, width], values(features * width, 11)),
        "b1": tensor([1, width], values(width, 13)),
        "w2": tensor([width, outputs], values(width * outputs, 17)),
        "b2": tensor([1, outputs], values(outputs, 19)),
    }
    mlp = quabla.trace_tensor(
        lambda x, w1, b1, w2, b2: (x.matmul(w1) + b1).tanh().matmul(w2) + b2,
        [
            ("x", [batch, features]),
            ("w1", [features, width]),
            ("b1", [1, width]),
            ("w2", [width, outputs]),
            ("b2", [1, outputs]),
        ],
    ).output.compile_mlx()
    benchmark_plan(
        "two_layer_mlp_forward",
        mlp,
        mlp_inputs,
        iterations,
        warmup_iterations,
        {"batch": batch, "features": features, "hidden_width": width, "outputs": outputs},
        device_only,
        ["w1", "b1", "w2", "b2"] if retain_static_inputs else [],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--device-only", action="store_true")
    parser.add_argument("--retain-static-inputs", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.warmup_iterations, args.batch, args.features, args.width, args.outputs) <= 0:
        parser.error("all sizes and iteration counts must be positive")
    benchmark(
        args.iterations,
        args.warmup_iterations,
        args.batch,
        args.features,
        args.width,
        args.outputs,
        args.device_only,
        args.retain_static_inputs,
    )


if __name__ == "__main__":
    main()
