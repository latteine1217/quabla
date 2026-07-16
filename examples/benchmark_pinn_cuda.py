"""Measure a device-resident two-layer Poisson PINN Adam training step.

Run on Linux with a visible NVIDIA device, CUDA driver libraries, and the
release extension:
    maturin develop --release --features cuda
    python examples/benchmark_pinn_cuda.py

The timed region contains only `optimizer.step()` calls. A following `loss()`
readback synchronizes the stream, so reported step time includes all queued GPU
work but excludes one explicit diagnostic host readback.
"""

import argparse
import json
import math
import platform
import time

import nabla


def build_optimizer(width: int, device_ordinal: int):
    coordinates = [0.2, 0.4, 0.6, 0.8]
    boundary_coordinates = [0.0, 0.0, 1.0, 1.0]
    specs = [
        ("x", [4, 1]),
        ("x_boundary", [4, 1]),
        ("w1", [1, width]),
        ("b1", [1, width]),
        ("w2", [width, 1]),
        ("b2", [1, 1]),
        ("forcing", [4, 1]),
        ("target", [4, 1]),
    ]
    traced = nabla.trace_tensor(
        lambda x, x_boundary, w1, b1, w2, b2, forcing, target: (
            (x @ w1 + b1).tanh() @ w2
        )
        + b2,
        specs,
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    graph = second_derivative.graph
    x_boundary = graph.input("x_boundary")
    w1 = graph.input("w1")
    b1 = graph.input("b1")
    w2 = graph.input("w2")
    b2 = graph.input("b2")
    forcing = graph.input("forcing")
    target = graph.input("target")
    boundary = ((x_boundary @ w1 + b1).tanh() @ w2) + b2
    residual_loss = (second_derivative.output + forcing).powi(2).mean()
    boundary_loss = (boundary - target).powi(2).mean()
    loss = residual_loss + boundary_loss

    teacher = {
        "w1": nabla.Tensor([1, width], [0.2 * (index + 1) for index in range(width)]),
        "b1": nabla.Tensor([1, width], [0.05 * (index - width // 2) for index in range(width)]),
        "w2": nabla.Tensor([width, 1], [0.1 * (index + 1) for index in range(width)]),
        "b2": nabla.Tensor([1, 1], [0.05]),
    }
    teacher_inputs = {
        "x": nabla.Tensor([4, 1], coordinates),
        "x_boundary": nabla.Tensor([4, 1], boundary_coordinates),
        "forcing": nabla.Tensor([4, 1], [0.0] * 4),
        "target": nabla.Tensor([4, 1], [0.0] * 4),
        **teacher,
    }
    second_value = graph.evaluate(second_derivative.output.node_id, teacher_inputs)
    boundary_target = graph.evaluate(boundary.node_id, teacher_inputs)
    split = nabla.Tensor.split_key(2026, 2)
    inputs = {
        "x": nabla.Tensor([4, 1], coordinates),
        "x_boundary": nabla.Tensor([4, 1], boundary_coordinates),
        "forcing": nabla.Tensor([4, 1], [-value for value in second_value.to_flat_list()]),
        "target": boundary_target,
        "w1": nabla.Tensor.glorot_normal([1, width], split[0]),
        "b1": nabla.Tensor([1, width], [0.0] * width),
        "w2": nabla.Tensor.glorot_normal([width, 1], split[1]),
        "b2": nabla.Tensor([1, 1], [0.0]),
    }
    optimizer = nabla.cuda_adam_loss_optimizer(
        loss,
        ["w1", "b1", "w2", "b2"],
        inputs,
        0.02,
        ["x", "x_boundary", "forcing", "target"],
        device_ordinal=device_ordinal,
    )
    return optimizer


def benchmark(steps: int, warmup_steps: int, width: int, device_ordinal: int) -> None:
    compile_start = time.perf_counter()
    optimizer = build_optimizer(width, device_ordinal)
    compile_seconds = time.perf_counter() - compile_start
    initial_loss = optimizer.loss().to_flat_list()[0]
    for _ in range(warmup_steps):
        optimizer.step()
    optimizer.loss()  # Explicit synchronization after warm-up.
    buffers_after_warmup = optimizer.device_buffer_count

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.step()
    final_loss = optimizer.loss().to_flat_list()[0]
    seconds = time.perf_counter() - start
    buffers_after_timing = optimizer.device_buffer_count
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise RuntimeError("PINN benchmark produced a non-finite loss")
    if buffers_after_timing != buffers_after_warmup:
        raise RuntimeError(
            "CUDA buffer count changed across timed static training: "
            f"{buffers_after_warmup} -> {buffers_after_timing}"
        )

    print(
        "metadata="
        + json.dumps(
            {
                "benchmark": "two_layer_poisson_pinn_adam_step",
                "compile_seconds": compile_seconds,
                "device_ordinal": device_ordinal,
                "dtype": "f32",
                "host": platform.platform(),
                "iterations": steps,
                "shape": {"collocation": [4, 1], "hidden_width": width},
                "synchronization_policy": "loss readback after warm-up and timed region",
                "transfer_policy": "static collocation, parameters, gradients, and Adam state retained on device",
                "warmup_iterations": warmup_steps,
            },
            sort_keys=True,
        )
    )
    print(
        f"loss={initial_loss:.8e}->{final_loss:.8e}, buffers={buffers_after_timing}, "
        f"compile={compile_seconds * 1e3:.2f} ms"
    )
    print(
        f"training: {steps / seconds:.2f} steps/s "
        f"({seconds * 1e3 / steps:.3f} ms/step), steps={steps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--device-ordinal", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.warmup_steps, args.width) <= 0 or args.device_ordinal < 0:
        parser.error("steps, warmup-steps, and width must be positive; device-ordinal must be non-negative")
    benchmark(args.steps, args.warmup_steps, args.width, args.device_ordinal)


if __name__ == "__main__":
    main()
