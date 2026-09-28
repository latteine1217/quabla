"""Run a batched one-parameter Poisson PINN with CUDA-resident Adam updates.

Build the Python extension on Linux with `maturin develop --features cuda`.
The model uses broadcast-batched matmul for two collocation points. The gradient
plan retains its parameter, collocation inputs, and output gradient on the
device after the first iteration. Adam does not return the parameter, gradient,
or optimizer moments to Python until training is complete.
"""

import math

import quabla


def main() -> None:
    coordinates = [0.25, 0.5]
    target_weight = 1.0
    forcing = [
        2.0
        * target_weight**2
        * math.tanh(target_weight * coordinate)
        * (1.0 - math.tanh(target_weight * coordinate) ** 2)
        for coordinate in coordinates
    ]

    traced = quabla.trace_tensor(
        lambda x, weight, forcing: (x @ weight).tanh(),
        [("x", [2, 1, 1]), ("weight", [1, 1, 1]), ("forcing", [2, 1, 1])],
    )
    second_derivative = traced.symbolic_jvp("x").symbolic_jvp("x")
    loss = (second_derivative.output + second_derivative.graph.input("forcing")).powi(2).sum()

    loss_plan = loss.compile_cuda()
    weight_gradient_plan = loss.symbolic_vjp("loss_cotangent")["weight"].output.compile_cuda()
    inputs = {
        "x": quabla.Tensor([2, 1, 1], coordinates),
        "forcing": quabla.Tensor([2, 1, 1], forcing),
        "loss_cotangent": quabla.Tensor([], [1.0]),
        "weight": quabla.Tensor([1, 1, 1], [0.3]),
    }

    initial_loss = loss_plan.evaluate(inputs).to_flat_list()[0]
    for _ in range(1000):
        weight_gradient_plan.adam_step(
            inputs,
            "weight",
            0.01,
            ["x", "forcing", "loss_cotangent"],
        )
    weight = weight_gradient_plan.retained_input("weight")
    final_loss = loss_plan.evaluate({**inputs, "weight": weight}).to_flat_list()[0]

    if not math.isfinite(final_loss) or final_loss >= initial_loss:
        raise RuntimeError(f"CUDA PINN loss did not decrease: {initial_loss} -> {final_loss}")
    print(
        f"weight={weight.to_flat_list()[0]:.8f} "
        f"loss={initial_loss:.8e}->{final_loss:.8e}"
    )


if __name__ == "__main__":
    main()
