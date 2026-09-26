"""Measure device-resident MLX Adam steps through a fixed-bound loop region.

Run after building the Python extension with the ``mlx`` feature:

    maturin develop --features mlx
    .venv/bin/python examples/benchmark_loop_value_and_grad_mlx.py

The timed interval contains only ``optimizer.step()``. It excludes diagnostic
loss and parameter readbacks, but each step evaluates the updated MLX arrays so
the GPU stream is synchronized before the next measured iteration. Optimizer
construction (trace, symbolic VJP, and MLX plan build) and repeated diagnostic
loss readbacks are timed separately.
"""

import argparse
import json
import platform
import time

import nabla


def build_optimizer(width: int, length: int):
    loss = nabla.trace_tensor(
        lambda initial, scale: nabla.tensor_fori_loop_region(
            0,
            length,
            lambda index, carry, captured_scale: carry.tanh()
            + captured_scale * (index + 1.0),
            initial,
            [scale],
        ).powi(2).mean(),
        [("initial", [width]), ("scale", [width])],
    )
    inputs = {
        "initial": nabla.Tensor([width], [0.05] * width),
        "scale": nabla.Tensor([width], [0.01] * width),
    }
    return nabla.mlx_adam_loss_optimizer(loss, ["scale"], inputs, 1e-3, ["initial"])


def benchmark(steps: int, warmup_steps: int, width: int, length: int) -> None:
    compile_started = time.perf_counter()
    optimizer = build_optimizer(width, length)
    compile_seconds = time.perf_counter() - compile_started
    for _ in range(warmup_steps):
        optimizer.step()

    started = time.perf_counter()
    for _ in range(steps):
        optimizer.step()
    elapsed = time.perf_counter() - started
    diagnostic_reads = 20
    readback_started = time.perf_counter()
    final_loss = [optimizer.loss().to_flat_list()[0] for _ in range(diagnostic_reads)][-1]
    diagnostic_seconds = time.perf_counter() - readback_started
    if not (final_loss >= 0.0 and final_loss < float("inf")):
        raise RuntimeError("MLX loop benchmark produced a non-finite loss")

    print(
        json.dumps(
            {
                "benchmark": "fixed_bound_fori_value_and_grad_adam_step",
                "backend": "mlx",
                "compile_seconds": compile_seconds,
                "diagnostic_readback_calls": diagnostic_reads,
                "diagnostic_readback_milliseconds_per_call": diagnostic_seconds * 1e3 / diagnostic_reads,
                "platform": platform.platform(),
                "width": width,
                "loop_length": length,
                "warmup_steps": warmup_steps,
                "timed_steps": steps,
                "elapsed_seconds": elapsed,
                "milliseconds_per_step": elapsed * 1e3 / steps,
                "final_loss": final_loss,
                "synchronization_policy": "each step evaluates updated device arrays; diagnostic loss readback excluded from timing",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--length", type=int, default=16)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0 or args.width <= 0 or args.length <= 0:
        parser.error("steps, width, and length must be positive; warmup-steps must be non-negative")
    benchmark(args.steps, args.warmup_steps, args.width, args.length)
