"""Train a two-layer tanh-MLP Poisson PINN on a device and check it on the CPU.

A fixed teacher network manufactures the problem: the forcing is minus its
second derivative at four collocation points and the boundary targets are
its values at x = 0 and x = 1, so a zero loss is reachable. `optim.Trainer`
keeps the parameters and Adam moments on the device; the same training on
the CPU is the reference for the trained parameters.

    python examples/pinn_mlp.py --device mlx    # extension built with mlx
    python examples/pinn_mlp.py --device cuda   # extension built with cuda
"""

import argparse
import math

import quabla as qb


def u(params, x):
    hidden = qb.tanh(x * params["w1"] + params["b1"])
    return qb.sum(hidden * params["w2"]) + params["b2"]


u_xx = qb.vmap(qb.grad(qb.grad(u, argnums=1), argnums=1), in_axes=(None, 0))
u_batch = qb.vmap(u, in_axes=(None, 0))


def losses(params, x, forcing, x_boundary, target):
    residual = qb.mean((u_xx(params, x) + forcing) ** 2)
    boundary = qb.mean((u_batch(params, x_boundary) - target) ** 2)
    return residual, boundary


def loss(params, *data):
    residual, boundary = losses(params, *data)
    return residual + boundary


def problem(width):
    teacher = {
        "w1": qb.array([1.2, -0.7]),
        "b1": qb.array([0.1, -0.2]),
        "w2": qb.array([0.8, 0.5]),
        "b2": qb.array(0.05),
    }
    x = qb.array([0.2, 0.4, 0.6, 0.8])
    x_boundary = qb.array([0.0, 0.0, 1.0, 1.0])
    data = (x, -u_xx(teacher, x), x_boundary, u_batch(teacher, x_boundary))
    params = {
        "w1": qb.array([math.sin(1.3 * (i + 1)) for i in range(width)]),
        "b1": qb.zeros([width]),
        "w2": qb.array([math.cos(0.7 * (i + 1)) for i in range(width)]),
        "b2": qb.array(0.0),
    }
    return params, data


def train(params, data, device, steps):
    trainer = qb.optim.Trainer(
        loss, params, qb.optim.Adam(learning_rate=0.02), *data, device=device
    )
    initial = trainer.loss().item()
    for _ in range(steps):
        trainer.step()
    return initial, trainer.loss().item(), trainer.params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", help="cpu, mlx, or cuda[:N]")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()

    params, data = problem(width=2)
    _, cpu_final, cpu_params = train(params, data, "cpu", args.steps)
    initial, final, trained = train(params, data, args.device, args.steps)
    residual, boundary = (value.item() for value in losses(trained, *data))
    difference = max(
        abs(a - b)
        for name in trained
        for a, b in zip(
            trained[name].reshape(-1).tolist(), cpu_params[name].reshape(-1).tolist()
        )
    )
    print(
        f"{args.device}: loss={initial:.3e}->{final:.3e} (cpu {cpu_final:.3e}), "
        f"residual={residual:.3e}, boundary={boundary:.3e}, "
        f"max_parameter_difference={difference:.3e}"
    )
    # The device trains in float32, the CPU in float64.
    if not (math.isfinite(final) and final < initial * 1e-4 and difference < 2e-3):
        raise SystemExit(f"{args.device} training does not match the CPU reference")


if __name__ == "__main__":
    main()
