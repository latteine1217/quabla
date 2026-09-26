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


def build_optimizer(collocation_count: int):
    collocation = [(index + 0.5) / collocation_count for index in range(collocation_count)]
    coordinates = collocation + [0.0, 1.0]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in collocation] + [0.0, 0.0]
    sample_count = collocation_count + 2
    traced = nabla.trace_tensor(
        lambda x, weight, forcing, boundary_target, boundary_mask: (x * weight).sin(),
        [
            ("x", [sample_count, 1]),
            ("weight", [1, 1]),
            ("forcing", [sample_count, 1]),
            ("boundary_target", [sample_count, 1]),
            ("boundary_mask", [sample_count, 1]),
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
            "x": nabla.Tensor([sample_count, 1], coordinates),
            "forcing": nabla.Tensor([sample_count, 1], forcing),
            "boundary_target": nabla.Tensor([sample_count, 1], [0.0] * sample_count),
            "boundary_mask": nabla.Tensor([sample_count, 1], [0.0] * collocation_count + [1.0] * 2),
            "weight": nabla.Tensor([1, 1], [2.5]),
        },
        0.01,
        ["x", "forcing", "boundary_target", "boundary_mask"],
    )


def benchmark(steps: int, warmup_steps: int, collocation_count: int) -> None:
    compile_start = time.perf_counter()
    optimizer = build_optimizer(collocation_count)
    compile_seconds = time.perf_counter() - compile_start
    initial_loss = optimizer.loss().to_flat_list()[0]
    for _ in range(warmup_steps):
        optimizer.step()
    warmup_loss = optimizer.loss().to_flat_list()[0]

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.step()
    seconds = time.perf_counter() - start
    diagnostic_reads = 20
    readback_started = time.perf_counter()
    final_loss = [optimizer.loss().to_flat_list()[0] for _ in range(diagnostic_reads)][-1]
    diagnostic_seconds = time.perf_counter() - readback_started
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
                "shape": {"collocation": [collocation_count, 1], "boundary": [2, 1]},
                "synchronization_policy": "each step evaluates updated MLX arrays; diagnostic loss readbacks excluded from timing",
                "transfer_policy": "static inputs, parameter, gradients, and Adam state retained on device",
                "warmup_iterations": warmup_steps,
                "diagnostic_readback_calls": diagnostic_reads,
                "diagnostic_readback_seconds": diagnostic_seconds,
                "diagnostic_readback_milliseconds_per_call": diagnostic_seconds * 1e3 / diagnostic_reads,
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

    print(f"diagnostic_readback: {diagnostic_seconds * 1e3 / diagnostic_reads:.3f} ms/call, "
          f"calls={diagnostic_reads}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--collocation-count", type=int, default=1_024)
    args = parser.parse_args()
    if min(args.steps, args.warmup_steps, args.collocation_count) <= 0:
        parser.error("steps, warmup-steps, and collocation-count must be positive")
    benchmark(args.steps, args.warmup_steps, args.collocation_count)


if __name__ == "__main__":
    main()
