"""Measure device-resident MLX Poisson PINN Adam steps on Apple silicon.

Run with the release extension:

    .venv/bin/maturin develop --release --features mlx
    .venv/bin/python examples/benchmark_pinn_mlx.py

The timed region contains only `optimizer.step()` calls. Each step evaluates
its updated MLX arrays, so it synchronizes the GPU stream without adding the
host readback performed by the diagnostic `loss()` calls.
"""

import argparse
import json
import math
import platform
import time

import nabla


def build_optimizer():
    collocation = [0.15, 0.35, 0.55, 0.75, 0.9]
    coordinates = collocation + [0.0, 1.0]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in collocation] + [0.0, 0.0]
    traced = nabla.trace_tensor(
        lambda x, weight, forcing, boundary_target, boundary_mask: (x * weight).sin(),
        [
            ("x", [7, 1]),
            ("weight", [1, 1]),
            ("forcing", [7, 1]),
            ("boundary_target", [7, 1]),
            ("boundary_mask", [7, 1]),
        ],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    graph = second_derivative.graph
    x = graph.input("x")
    weight = graph.input("weight")
    residual = second_derivative.output + graph.input("forcing")
    boundary_error = (x * weight).sin() - graph.input("boundary_target")
    error = nabla.where(graph.input("boundary_mask").gt(0.0), boundary_error, residual)
    loss = (error * error).mean()
    return nabla.mlx_adam_loss_optimizer(
        loss,
        ["weight"],
        {
            "x": nabla.Tensor([7, 1], coordinates),
            "forcing": nabla.Tensor([7, 1], forcing),
            "boundary_target": nabla.Tensor([7, 1], [0.0] * 7),
            "boundary_mask": nabla.Tensor([7, 1], [0.0] * 5 + [1.0] * 2),
            "weight": nabla.Tensor([1, 1], [2.5]),
        },
        0.01,
        ["x", "forcing", "boundary_target", "boundary_mask"],
    )


def benchmark(steps: int, warmup_steps: int) -> None:
    compile_start = time.perf_counter()
    optimizer = build_optimizer()
    compile_seconds = time.perf_counter() - compile_start
    initial_loss = optimizer.loss().to_flat_list()[0]
    for _ in range(warmup_steps):
        optimizer.step()
    warmup_loss = optimizer.loss().to_flat_list()[0]

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.step()
    seconds = time.perf_counter() - start
    final_loss = optimizer.loss().to_flat_list()[0]
    if not all(math.isfinite(value) for value in (initial_loss, warmup_loss, final_loss)):
        raise RuntimeError("MLX PINN benchmark produced a non-finite loss")

    print(
        "metadata="
        + json.dumps(
            {
                "benchmark": "one_parameter_poisson_pinn_adam_step",
                "compile_seconds": compile_seconds,
                "device": "mlx.gpu",
                "dtype": "f32",
                "host": platform.platform(),
                "iterations": steps,
                "shape": {"collocation": [5, 1], "boundary": [2, 1]},
                "synchronization_policy": "each step evaluates updated MLX arrays; diagnostic loss readbacks excluded from timing",
                "transfer_policy": "static inputs, parameter, gradients, and Adam state retained on device",
                "warmup_iterations": warmup_steps,
            },
            sort_keys=True,
        )
    )
    print(
        f"loss={initial_loss:.8e}->{warmup_loss:.8e}->{final_loss:.8e}, "
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
    args = parser.parse_args()
    if min(args.steps, args.warmup_steps) <= 0:
        parser.error("steps and warmup-steps must be positive")
    benchmark(args.steps, args.warmup_steps)


if __name__ == "__main__":
    main()
