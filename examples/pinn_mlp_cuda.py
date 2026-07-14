"""Train a two-layer Poisson PINN with CUDA-resident multi-parameter Adam.

Build on Linux with `maturin develop --features cuda`, then run this file.
The loss-aware optimizer owns one shared CUDA value-and-gradient plan. Static
collocation tensors, parameters, and Adam state stay device-resident; the loop
does not construct VJP dictionaries or materialize gradients on the host.
"""

import math

import nabla


def main() -> None:
    coordinates = [0.2, 0.4, 0.6, 0.8]
    boundary_coordinates = [0.0, 0.0, 1.0, 1.0]
    specs = [
        ("x", [4, 1]),
        ("x_boundary", [4, 1]),
        ("w1", [1, 2]),
        ("b1", [1, 2]),
        ("w2", [2, 1]),
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
    parameter_names = ["w1", "b1", "w2", "b2"]

    teacher = {
        "w1": nabla.Tensor([1, 2], [1.2, -0.7]),
        "b1": nabla.Tensor([1, 2], [0.1, -0.2]),
        "w2": nabla.Tensor([2, 1], [0.8, 0.5]),
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
    inputs = {
        "x": nabla.Tensor([4, 1], coordinates),
        "x_boundary": nabla.Tensor([4, 1], boundary_coordinates),
        "forcing": nabla.Tensor([4, 1], [-value for value in second_value.to_flat_list()]),
        "target": boundary_target,
        "w1": nabla.Tensor.glorot_normal([1, 2], nabla.Tensor.split_key(2026, 2)[0]),
        "b1": nabla.Tensor([1, 2], [0.0, 0.0]),
        "w2": nabla.Tensor.glorot_normal([2, 1], nabla.Tensor.split_key(2026, 2)[1]),
        "b2": nabla.Tensor([1, 1], [0.0]),
    }
    optimizer = nabla.cuda_adam_loss_optimizer(
        loss,
        parameter_names,
        inputs,
        0.02,
        ["x", "x_boundary", "forcing", "target"],
    )

    initial_loss = optimizer.loss().to_flat_list()[0]
    for _ in range(1500):
        optimizer.step()
    final_loss = optimizer.loss().to_flat_list()[0]
    trained = optimizer.parameters()
    final_inputs = {**inputs, **trained}
    final_residual_loss = graph.evaluate(residual_loss.node_id, final_inputs).to_flat_list()[0]
    final_boundary_loss = graph.evaluate(boundary_loss.node_id, final_inputs).to_flat_list()[0]
    if (
        not math.isfinite(final_loss)
        or final_loss >= initial_loss * 1e-4
        or final_residual_loss >= 1e-5
        or final_boundary_loss >= 1e-5
    ):
        raise RuntimeError(f"CUDA MLP PINN did not converge: {initial_loss} -> {final_loss}")
    print(
        f"loss={initial_loss:.8e}->{final_loss:.8e}, "
        f"residual={final_residual_loss:.8e}, boundary={final_boundary_loss:.8e}"
    )


if __name__ == "__main__":
    main()
