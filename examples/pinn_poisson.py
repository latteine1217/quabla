"""Train a one-parameter PINN for u_xx + pi^2 sin(pi x) = 0 on [0, 1]."""

import math

import quabla


def main() -> None:
    coordinates = [0.15, 0.35, 0.55, 0.75, 0.9]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in coordinates]

    residual_trace = quabla.trace_tensor(
        lambda x, weight, forcing: (x * weight).sin(),
        [("x", [5, 1]), ("weight", [1, 1]), ("forcing", [5, 1])],
    )
    second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
    residual = second_derivative.output + second_derivative.graph.input("forcing")
    residual_loss = (residual * residual).mean()
    residual_plan = residual_loss.compile_cpu()

    boundary_trace = quabla.trace_tensor(
        lambda x, weight, target: (x * weight).sin(),
        [("x", [2, 1]), ("weight", [1, 1]), ("target", [2, 1])],
    )
    boundary_error = boundary_trace.output - boundary_trace.graph.input("target")
    boundary_loss = (boundary_error * boundary_error).mean()
    boundary_plan = boundary_loss.compile_cpu()

    parameters = {"weight": quabla.Tensor([1, 1], [2.5])}
    optimizer = quabla.Adam(learning_rate=0.01)

    for _ in range(2000):
        _, residual_gradients = residual_plan.evaluate_value_and_vjp(
            {
                "x": quabla.Tensor([5, 1], coordinates),
                "forcing": quabla.Tensor([5, 1], forcing),
                **parameters,
            },
            quabla.Tensor([], [1.0]),
        )
        _, boundary_gradients = boundary_plan.evaluate_value_and_vjp(
            {
                "x": quabla.Tensor([2, 1], [0.0, 1.0]),
                "target": quabla.Tensor([2, 1], [0.0, 0.0]),
                **parameters,
            },
            quabla.Tensor([], [1.0]),
        )
        parameters = optimizer.step(
            parameters,
            quabla.sum_gradients([residual_gradients, boundary_gradients], ["weight"]),
        )

    print(parameters["weight"].to_flat_list()[0])


if __name__ == "__main__":
    main()
