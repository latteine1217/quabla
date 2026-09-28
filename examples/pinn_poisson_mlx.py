"""Train and cross-check a one-parameter Poisson PINN on MLX and CPU.

Build the extension on Apple silicon with:

    .venv/bin/maturin develop --release --features mlx
    .venv/bin/python examples/pinn_poisson_mlx.py

The loss is one fixed-shape Tensor IR graph. It combines collocation residuals
with boundary errors through `quabla.where`, so MLX evaluates one symbolic VJP
and updates its retained parameter and Adam state without per-step host reads.
"""

import math

import quabla


def build_loss():
    collocation = [0.15, 0.35, 0.55, 0.75, 0.9]
    coordinates = collocation + [0.0, 1.0]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in collocation] + [0.0, 0.0]
    traced = quabla.trace_tensor(
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
    selected_error = quabla.where(
        graph.input("boundary_mask").gt(0.0), boundary_error, residual
    )
    loss = (selected_error * selected_error).mean()
    inputs = {
        "x": quabla.Tensor([7, 1], coordinates),
        "forcing": quabla.Tensor([7, 1], forcing),
        "boundary_target": quabla.Tensor([7, 1], [0.0] * 7),
        "boundary_mask": quabla.Tensor([7, 1], [0.0] * 5 + [1.0] * 2),
        "weight": quabla.Tensor([1, 1], [2.5]),
    }
    return graph, loss, inputs


def train_cpu(graph, loss, inputs, steps):
    optimizer = quabla.Adam(learning_rate=0.01)
    parameters = {"weight": inputs["weight"]}
    initial_loss = None
    for _ in range(steps):
        value, gradients = graph.evaluate_value_and_vjp(
            loss.node_id,
            {**inputs, **parameters},
            quabla.Tensor([], [1.0]),
        )
        if initial_loss is None:
            initial_loss = value.to_flat_list()[0]
        parameters = optimizer.step(parameters, {"weight": gradients["weight"]})
    final_loss, _ = graph.evaluate_value_and_vjp(
        loss.node_id,
        {**inputs, **parameters},
        quabla.Tensor([], [1.0]),
    )
    return initial_loss, final_loss.to_flat_list()[0], parameters["weight"]


def main() -> None:
    steps = 2000
    graph, loss, inputs = build_loss()
    cpu_initial, cpu_final, cpu_weight = train_cpu(graph, loss, inputs, steps)
    optimizer = quabla.mlx_adam_loss_optimizer(
        loss, ["weight"], inputs, 0.01, ["x", "forcing", "boundary_target", "boundary_mask"]
    )
    mlx_initial = optimizer.loss().to_flat_list()[0]
    for _ in range(steps):
        optimizer.step()
    mlx_final = optimizer.loss().to_flat_list()[0]
    mlx_weight = optimizer.parameters()["weight"].to_flat_list()[0]
    cpu_weight_value = cpu_weight.to_flat_list()[0]

    if not (math.isfinite(mlx_final) and mlx_final < mlx_initial * 1e-8):
        raise RuntimeError(f"MLX PINN loss did not converge: {mlx_initial} -> {mlx_final}")
    if abs(mlx_weight - cpu_weight_value) > 2e-3:
        raise RuntimeError(
            f"MLX/CPU PINN weights diverged: mlx={mlx_weight}, cpu={cpu_weight_value}"
        )
    print(
        f"cpu_loss={cpu_initial:.3e}->{cpu_final:.3e} "
        f"mlx_loss={mlx_initial:.3e}->{mlx_final:.3e} "
        f"weight={mlx_weight:.8f}"
    )


if __name__ == "__main__":
    main()
