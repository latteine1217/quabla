"""Measure device-resident MLX Poisson PINN Adam steps on Apple silicon.

Run with the release extension:

    .venv/bin/maturin develop --release --features mlx
    .venv/bin/python examples/benchmark_pinn_mlx.py

The timed region contains only `optimizer.step()` calls. Each step evaluates
its updated MLX arrays, so it synchronizes the GPU stream without adding the
host readback performed by the diagnostic `loss()` calls.

`--hidden-layers N` (N > 0) replaces the one-parameter `sin(x * weight)` model
with an N-layer tanh MLP of `--hidden-width` units trained on the same Poisson
residual and boundary loss. The default `--hidden-layers 0` keeps the
one-parameter benchmark unchanged.
"""

import argparse
import json
import math
import platform
import time

import quabla


def mlp_parameter_specs(hidden_layers: int, hidden_width: int):
    widths = [1] + [hidden_width] * hidden_layers + [1]
    specs = []
    for layer, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        specs.append((f"w{layer}", [fan_in, fan_out]))
        specs.append((f"b{layer}", [1, fan_out]))
    return specs


def mlp(x, parameters):
    # parameters alternates weight and bias; tanh on every layer except the output.
    layer_count = len(parameters) // 2
    value = x
    for layer in range(layer_count):
        value = value @ parameters[2 * layer] + parameters[2 * layer + 1]
        if layer + 1 < layer_count:
            value = value.tanh()
    return value


def mlp_initial_parameters(specs):
    keys = quabla.Tensor.split_key(2026, len(specs) // 2)
    return {
        name: quabla.Tensor.glorot_normal(shape, keys[index // 2])
        if name.startswith("w")
        else quabla.Tensor(shape, [0.0] * shape[1])
        for index, (name, shape) in enumerate(specs)
    }


def build_optimizer(collocation_count: int, hidden_layers: int = 0, hidden_width: int = 0):
    collocation = [(index + 0.5) / collocation_count for index in range(collocation_count)]
    coordinates = collocation + [0.0, 1.0]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in collocation] + [0.0, 0.0]
    sample_count = collocation_count + 2
    if hidden_layers > 0:
        return build_mlp_optimizer(
            coordinates, forcing, collocation_count, hidden_layers, hidden_width
        )
    traced = quabla.trace_tensor(
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
    error = quabla.where(graph.input("boundary_mask").gt(0.0), boundary_error, residual)
    loss = (error * error).mean()
    return quabla.mlx_adam_loss_optimizer(
        loss,
        ["weight"],
        {
            "x": quabla.Tensor([sample_count, 1], coordinates),
            "forcing": quabla.Tensor([sample_count, 1], forcing),
            "boundary_target": quabla.Tensor([sample_count, 1], [0.0] * sample_count),
            "boundary_mask": quabla.Tensor([sample_count, 1], [0.0] * collocation_count + [1.0] * 2),
            "weight": quabla.Tensor([1, 1], [2.5]),
        },
        0.01,
        ["x", "forcing", "boundary_target", "boundary_mask"],
    )


def build_mlp_optimizer(coordinates, forcing, collocation_count, hidden_layers, hidden_width):
    sample_count = collocation_count + 2
    parameter_specs = mlp_parameter_specs(hidden_layers, hidden_width)
    parameter_names = [name for name, _ in parameter_specs]
    traced = quabla.trace_tensor(
        lambda x, forcing, boundary_target, boundary_mask, *parameters: mlp(x, parameters),
        [
            ("x", [sample_count, 1]),
            ("forcing", [sample_count, 1]),
            ("boundary_target", [sample_count, 1]),
            ("boundary_mask", [sample_count, 1]),
            *parameter_specs,
        ],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    graph = second_derivative.graph
    parameters = [graph.input(name) for name in parameter_names]
    residual = second_derivative.output + graph.input("forcing")
    boundary_error = mlp(graph.input("x"), parameters) - graph.input("boundary_target")
    error = quabla.where(graph.input("boundary_mask").gt(0.0), boundary_error, residual)
    loss = (error * error).mean()
    return quabla.mlx_adam_loss_optimizer(
        loss,
        parameter_names,
        {
            "x": quabla.Tensor([sample_count, 1], coordinates),
            "forcing": quabla.Tensor([sample_count, 1], forcing),
            "boundary_target": quabla.Tensor([sample_count, 1], [0.0] * sample_count),
            "boundary_mask": quabla.Tensor([sample_count, 1], [0.0] * collocation_count + [1.0] * 2),
            **mlp_initial_parameters(parameter_specs),
        },
        0.01,
        ["x", "forcing", "boundary_target", "boundary_mask"],
    )


def benchmark(
    steps: int, warmup_steps: int, collocation_count: int, hidden_layers: int = 0, hidden_width: int = 0
) -> None:
    compile_start = time.perf_counter()
    optimizer = build_optimizer(collocation_count, hidden_layers, hidden_width)
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
    shape = {"collocation": [collocation_count, 1], "boundary": [2, 1]}
    if hidden_layers > 0:
        shape["hidden_layers"] = hidden_layers
        shape["hidden_width"] = hidden_width
        shape["parameter_count"] = sum(
            size[0] * size[1] for _, size in mlp_parameter_specs(hidden_layers, hidden_width)
        )

    print(
        "metadata="
        + json.dumps(
            {
                "benchmark": "mlp_poisson_pinn_adam_step"
                if hidden_layers > 0
                else "one_parameter_poisson_pinn_adam_step",
                "compile_seconds": compile_seconds,
                "device": "mlx.gpu",
                "dtype": "f32",
                "host": platform.platform(),
                "iterations": steps,
                "shape": shape,
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
    parser.add_argument("--hidden-layers", type=int, default=0)
    parser.add_argument("--hidden-width", type=int, default=64)
    args = parser.parse_args()
    if min(args.steps, args.warmup_steps, args.collocation_count, args.hidden_width) <= 0:
        parser.error("steps, warmup-steps, collocation-count, and hidden-width must be positive")
    if args.hidden_layers < 0:
        parser.error("hidden-layers must be non-negative")
    benchmark(
        args.steps, args.warmup_steps, args.collocation_count, args.hidden_layers, args.hidden_width
    )


if __name__ == "__main__":
    main()
