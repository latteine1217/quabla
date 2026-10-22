"""Train a one-parameter PINN for u_xx + pi^2 sin(pi x) = 0 on [0, 1].

The model is u(x) = sin(w x), so the exact solution is w = pi. The PDE
residual differentiates the model twice per collocation point with
`vmap(grad(grad(u)))`, and `optim.Trainer` runs Adam on the chosen device:

    python examples/pinn_poisson.py
    python examples/pinn_poisson.py --device mlx    # extension built with mlx
    python examples/pinn_poisson.py --device cuda   # extension built with cuda
"""

import argparse
import math

import quabla as qb


def u(w, x):
    return qb.sin(w * x)


u_xx = qb.vmap(qb.grad(qb.grad(u, argnums=1), argnums=1), in_axes=(None, 0))


def loss(w, x, forcing, x_boundary):
    residual = u_xx(w, x) + forcing
    boundary = qb.vmap(u, in_axes=(None, 0))(w, x_boundary)
    return qb.mean(residual**2) + qb.mean(boundary**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", help="cpu, mlx, or cuda[:N]")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    # MLX has no float64 arithmetic, so it trains in float32.
    dtype = qb.float32 if args.device == "mlx" else qb.float64
    x = qb.array([0.15, 0.35, 0.55, 0.75, 0.9], dtype=dtype)
    forcing = math.pi**2 * qb.sin(math.pi * x)
    x_boundary = qb.array([0.0, 1.0], dtype=dtype)
    trainer = qb.optim.Trainer(
        loss,
        qb.array(2.5, dtype=dtype),
        qb.optim.Adam(learning_rate=0.01),
        x,
        forcing,
        x_boundary,
        device=args.device,
    )
    initial = trainer.loss().item()
    for _ in range(args.steps):
        trainer.step()
    w = trainer.params.item()
    print(f"w={w:.6f} (pi={math.pi:.6f}) loss={initial:.3e}->{trainer.loss().item():.3e}")


if __name__ == "__main__":
    main()
