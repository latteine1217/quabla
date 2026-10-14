"""Train a small MLP PINN for the 1D Poisson problem and plot it for the README.

Problem: u''(x) = -pi^2 sin(pi x) on [0, 1] with u(0) = u(1) = 0, whose exact
solution is u(x) = sin(pi x). The network is a one-hidden-layer tanh MLP; the
PDE residual differentiates it twice per collocation point with
`vmap(grad(grad(u)))` for the exact u'' (no finite differences), and training
runs Adam on the CPU, so every build of the extension (CPU, MLX, or CUDA) can
regenerate the figure.

Requires matplotlib, which is not a Quabla dependency. From the repository
root, with the extension built (`maturin develop --release`):

    python -m pip install matplotlib
    python examples/plot_readme_figure.py

Writes docs/assets/pinn_poisson.png and prints the metrics shown in it.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import quabla as qb  # noqa: E402

WIDTH = 16
COLLOCATION = 32
STEPS = 3000
LEARNING_RATE = 0.01
PLOT_POINTS = 201
SEED = 2026
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "pinn_poisson.png"


def mlp(params, x):
    """The network at one point x."""
    hidden = qb.tanh(x * params["w1"] + params["b1"])
    return qb.sum(hidden * params["w2"]) + params["b2"]


u = qb.vmap(mlp, in_axes=(None, 0))
u_xx = qb.vmap(qb.grad(qb.grad(mlp, argnums=1), argnums=1), in_axes=(None, 0))


def loss(params, x, x_boundary):
    residual = u_xx(params, x) + math.pi**2 * qb.sin(math.pi * x)
    return qb.mean(residual**2) + qb.mean(u(params, x_boundary) ** 2)


def train():
    keys = qb.Tensor.split_key(SEED, 2)
    params = {
        "w1": qb.Tensor.glorot_normal([1, WIDTH], keys[0]).reshape([WIDTH]),
        "b1": qb.zeros([WIDTH]),
        "w2": qb.Tensor.glorot_normal([WIDTH, 1], keys[1]).reshape([WIDTH]),
        "b2": qb.array(0.0),
    }
    # Interior collocation points: a uniform grid that excludes the boundary.
    step = 1.0 / (COLLOCATION + 1)
    collocation = [step * (i + 1) for i in range(COLLOCATION)]
    x, x_boundary = qb.array(collocation), qb.array([0.0, 1.0])
    value_and_grad = qb.jit(qb.value_and_grad(loss))
    optimizer = qb.optim.Adam(learning_rate=LEARNING_RATE)
    losses = []
    for _ in range(STEPS):
        value, gradients = value_and_grad(params, x, x_boundary)
        losses.append(value.item())
        params = optimizer.step(params, gradients)
    return collocation, params, losses


def main() -> None:
    collocation, parameters, losses = train()
    grid = [i / (PLOT_POINTS - 1) for i in range(PLOT_POINTS)]
    predicted = u(parameters, qb.array(grid)).tolist()
    exact = [math.sin(math.pi * x) for x in grid]
    max_error = max(abs(p - e) for p, e in zip(predicted, exact))
    print(
        f"steps={STEPS} loss={losses[0]:.3e}->{losses[-1]:.3e} "
        f"max_abs_error={max_error:.3e} (on {PLOT_POINTS} points)"
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=150)
    fig.patch.set_facecolor("white")

    left.plot(grid, exact, color="#8a8984", linewidth=2.5, label="Exact: sin(πx)")
    left.plot(
        grid,
        predicted,
        color="#2a78d6",
        linewidth=1.5,
        linestyle="--",
        label="Quabla PINN",
    )
    left.plot(
        collocation,
        [0.0] * len(collocation),
        linestyle="none",
        marker="|",
        markersize=8,
        color="#52514e",
        label="Collocation points",
    )
    left.set_title(f"Solution (max abs error {max_error:.1e})", fontsize=10)
    left.set_xlabel("x")
    left.set_ylabel("u(x)")
    left.legend(
        frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.08)
    )

    right.semilogy(range(1, STEPS + 1), losses, color="#2a78d6", linewidth=1.5)
    right.set_title("Training loss (PDE residual + boundary)", fontsize=10)
    right.set_xlabel("Adam step")
    right.set_ylabel("Loss")

    for axis in (left, right):
        axis.set_facecolor("white")
        axis.grid(True, color="#e5e4e0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("1D Poisson PINN: u'' = -π² sin(πx), u(0) = u(1) = 0", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
