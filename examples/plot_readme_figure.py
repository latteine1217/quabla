"""Train a small MLP PINN for the 1D Poisson problem and plot it for the README.

Problem: u''(x) = -pi^2 sin(pi x) on [0, 1] with u(0) = u(1) = 0, whose exact
solution is u(x) = sin(pi x). The network is a one-hidden-layer tanh MLP; the
PDE residual uses two symbolic coordinate JVPs for the exact u'' (no finite
differences), and training runs Adam on the CPU backend, so every build of
the extension (CPU, MLX, or CUDA) can regenerate the figure.

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

import quabla  # noqa: E402

WIDTH = 16
COLLOCATION = 32
STEPS = 3000
LEARNING_RATE = 0.01
PLOT_POINTS = 201
SEED = 2026
PARAMETERS = ["w1", "b1", "w2", "b2"]
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "pinn_poisson.png"


def mlp(x, w1, b1, w2, b2):
    return (x @ w1 + b1).tanh() @ w2 + b2


def build_loss_plan():
    """Freeze residual + boundary loss into one CPU plan."""
    specs = [
        ("x", [COLLOCATION, 1]),
        ("x_boundary", [2, 1]),
        ("w1", [1, WIDTH]),
        ("b1", [1, WIDTH]),
        ("w2", [WIDTH, 1]),
        ("b2", [1, 1]),
    ]
    traced = quabla.trace_tensor(
        lambda x, x_boundary, w1, b1, w2, b2: mlp(x, w1, b1, w2, b2), specs
    )
    u_xx = traced.symbolic_jvp("x").symbolic_jvp("x")
    graph = u_xx.graph
    x = graph.input("x")
    params = [graph.input(name) for name in PARAMETERS]
    residual = u_xx.output + math.pi**2 * (math.pi * x).sin()
    boundary = mlp(graph.input("x_boundary"), *params)
    loss = residual.powi(2).mean() + boundary.powi(2).mean()
    return loss.compile_cpu()


def build_predict_plan():
    specs = [
        ("x", [PLOT_POINTS, 1]),
        ("w1", [1, WIDTH]),
        ("b1", [1, WIDTH]),
        ("w2", [WIDTH, 1]),
        ("b2", [1, 1]),
    ]
    return quabla.trace_tensor(mlp, specs).output.compile_cpu()


def train():
    keys = quabla.Tensor.split_key(SEED, 2)
    parameters = {
        "w1": quabla.Tensor.glorot_normal([1, WIDTH], keys[0]),
        "b1": quabla.Tensor([1, WIDTH], [0.0] * WIDTH),
        "w2": quabla.Tensor.glorot_normal([WIDTH, 1], keys[1]),
        "b2": quabla.Tensor([1, 1], [0.0]),
    }
    # Interior collocation points: a uniform grid that excludes the boundary.
    step = 1.0 / (COLLOCATION + 1)
    collocation = [step * (i + 1) for i in range(COLLOCATION)]
    data = {
        "x": quabla.Tensor([COLLOCATION, 1], collocation),
        "x_boundary": quabla.Tensor([2, 1], [0.0, 1.0]),
    }
    plan = build_loss_plan()
    optimizer = quabla.Adam(learning_rate=LEARNING_RATE)
    seed = quabla.Tensor([], [1.0])
    losses = []
    for _ in range(STEPS):
        value, gradients = plan.evaluate_value_and_vjp({**data, **parameters}, seed)
        losses.append(value.to_flat_list()[0])
        parameters = optimizer.step(
            parameters, {name: gradients[name] for name in PARAMETERS}
        )
    return collocation, parameters, losses


def main() -> None:
    collocation, parameters, losses = train()
    grid = [i / (PLOT_POINTS - 1) for i in range(PLOT_POINTS)]
    predicted = (
        build_predict_plan()
        .evaluate({"x": quabla.Tensor([PLOT_POINTS, 1], grid), **parameters})
        .to_flat_list()
    )
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
