"""Train a two-layer Poisson PINN with MLX-resident multi-parameter Adam.

Build the release extension on Apple silicon, then run:

    .venv/bin/maturin develop --release --features mlx
    .venv/bin/python examples/pinn_mlp_mlx.py

The training loop keeps four MLP parameters, their gradients, and Adam moments
on the MLX GPU stream. CPU is used only as a deterministic reference after the
same fixed number of optimization steps.
"""

import math

import quabla


def build_problem(width: int):
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
    traced = quabla.trace_tensor(
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
    boundary = ((x_boundary @ w1 + b1).tanh() @ w2) + b2
    residual_loss = (second_derivative.output + graph.input("forcing")).powi(2).mean()
    boundary_loss = (boundary - graph.input("target")).powi(2).mean()
    loss = residual_loss + boundary_loss

    teacher = {
        "w1": quabla.Tensor([1, width], [1.2, -0.7]),
        "b1": quabla.Tensor([1, width], [0.1, -0.2]),
        "w2": quabla.Tensor([width, 1], [0.8, 0.5]),
        "b2": quabla.Tensor([1, 1], [0.05]),
    }
    teacher_inputs = {
        "x": quabla.Tensor([4, 1], coordinates),
        "x_boundary": quabla.Tensor([4, 1], boundary_coordinates),
        "forcing": quabla.Tensor([4, 1], [0.0] * 4),
        "target": quabla.Tensor([4, 1], [0.0] * 4),
        **teacher,
    }
    second_value = graph.evaluate(second_derivative.output.node_id, teacher_inputs)
    boundary_target = graph.evaluate(boundary.node_id, teacher_inputs)
    keys = quabla.Tensor.split_key(2026, 2)
    inputs = {
        "x": quabla.Tensor([4, 1], coordinates),
        "x_boundary": quabla.Tensor([4, 1], boundary_coordinates),
        "forcing": quabla.Tensor([4, 1], [-value for value in second_value.to_flat_list()]),
        "target": boundary_target,
        "w1": quabla.Tensor.glorot_normal([1, width], keys[0]),
        "b1": quabla.Tensor([1, width], [0.0] * width),
        "w2": quabla.Tensor.glorot_normal([width, 1], keys[1]),
        "b2": quabla.Tensor([1, 1], [0.0]),
    }
    return graph, loss, residual_loss, boundary_loss, inputs


def train_cpu(graph, loss, inputs, steps):
    optimizer = quabla.Adam(learning_rate=0.02)
    parameter_names = ["w1", "b1", "w2", "b2"]
    parameters = {name: inputs[name] for name in parameter_names}
    initial_loss = graph.evaluate(loss.node_id, {**inputs, **parameters}).to_flat_list()[0]
    for _ in range(steps):
        _, gradients = graph.evaluate_value_and_vjp(
            loss.node_id,
            {**inputs, **parameters},
            quabla.Tensor([], [1.0]),
        )
        parameters = optimizer.step(parameters, {name: gradients[name] for name in parameter_names})
    final_loss = graph.evaluate(loss.node_id, {**inputs, **parameters}).to_flat_list()[0]
    return initial_loss, final_loss, parameters


def run(width: int = 2, steps: int = 1500):
    graph, loss, residual_loss, boundary_loss, inputs = build_problem(width)
    cpu_initial, cpu_final, cpu_parameters = train_cpu(graph, loss, inputs, steps)
    optimizer = quabla.mlx_adam_loss_optimizer(
        loss,
        ["w1", "b1", "w2", "b2"],
        inputs,
        0.02,
        ["x", "x_boundary", "forcing", "target"],
    )
    mlx_initial = optimizer.loss().to_flat_list()[0]
    for _ in range(steps):
        optimizer.step()
    mlx_final = optimizer.loss().to_flat_list()[0]
    mlx_parameters = optimizer.parameters()
    mlx_inputs = {**inputs, **mlx_parameters}
    mlx_residual = graph.evaluate(residual_loss.node_id, mlx_inputs).to_flat_list()[0]
    mlx_boundary = graph.evaluate(boundary_loss.node_id, mlx_inputs).to_flat_list()[0]
    max_parameter_difference = max(
        abs(mlx_parameters[name].to_flat_list()[index] - cpu_parameters[name].to_flat_list()[index])
        for name in mlx_parameters
        for index in range(len(mlx_parameters[name].to_flat_list()))
    )

    if (
        not math.isfinite(mlx_final)
        or mlx_final >= mlx_initial * 1e-4
        or mlx_residual >= 1e-5
        or mlx_boundary >= 1e-5
        or max_parameter_difference >= 2e-3
    ):
        raise RuntimeError(
            "MLX MLP PINN did not match the CPU reference: "
            f"loss={mlx_initial}->{mlx_final}, residual={mlx_residual}, "
            f"boundary={mlx_boundary}, parameter_difference={max_parameter_difference}"
        )
    return {
        "cpu_initial": cpu_initial,
        "cpu_final": cpu_final,
        "mlx_initial": mlx_initial,
        "mlx_final": mlx_final,
        "mlx_residual": mlx_residual,
        "mlx_boundary": mlx_boundary,
        "max_parameter_difference": max_parameter_difference,
    }


def main() -> None:
    metrics = run()
    print(
        f"cpu_loss={metrics['cpu_initial']:.8e}->{metrics['cpu_final']:.8e} "
        f"mlx_loss={metrics['mlx_initial']:.8e}->{metrics['mlx_final']:.8e}, "
        f"residual={metrics['mlx_residual']:.3e}, boundary={metrics['mlx_boundary']:.3e}, "
        f"max_parameter_difference={metrics['max_parameter_difference']:.3e}"
    )


if __name__ == "__main__":
    main()
