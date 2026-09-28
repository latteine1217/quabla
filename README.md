# Quabla

**Differentiable scientific computing for physics-informed neural networks
(PINNs): a Rust compiler and runtime with a JAX-style Python API that runs one
traced program on CPU, CUDA, and Apple silicon.**

[![CI](https://github.com/latteine1217/quabla/actions/workflows/ci.yml/badge.svg)](https://github.com/latteine1217/quabla/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#installation-from-source)
[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-orange)](https://github.com/latteine1217/quabla/releases/tag/v0.1.0)

![A tanh MLP trained with Quabla matches the exact solution of a 1D Poisson problem; its training loss falls from about 50 to 2.5e-4.](docs/assets/pinn_poisson.png)

*A 16-unit tanh MLP trained as a PINN on the CPU backend (3,000 Adam steps)
matches the exact solution `sin(πx)` to a max abs error of 2.4e-4.
Reproduce with `python examples/plot_readme_figure.py`.*

## Why Quabla

- **One program, three backends.** Trace once, then compile for the `f64` CPU
  reference, CUDA (Linux), or MLX (Apple silicon); device backends are
  validated operation by operation against the CPU reference.
- **Exact higher-order derivatives.** JVP, VJP, Jacobians, Hessians, and
  Hessian-vector products compose with each other and with `vmap`, so a PINN
  residual uses an exact `u_xx` that is itself differentiable in the weights.
- **Differentiable control flow.** `cond`, `fori_loop`, and `scan` are IR
  regions, not traced Python branches, with VJP and forward-over-reverse HVP
  on CPU and MLX and, for elementwise loop bodies, on CUDA.
- **Explicit rather than silent.** `float32`, `float64`, and `bool` convert
  only through `astype`, and an operation a backend does not support raises an
  error instead of falling back to the host.

## At a Glance

Two symbolic JVPs give an exact `u_xx`; the PDE-residual loss built from it is
compiled once and trained with reverse-mode gradients and Adam:

```python
import math

import quabla

# Fit u(x) = sin(w x) to u'' = -pi^2 sin(pi x); the exact solution has w = pi.
u = quabla.trace_tensor(lambda x, w: (x * w).sin(), [("x", [8, 1]), ("w", [1, 1])])
u_xx = u.symbolic_jvp("x").symbolic_jvp("x")  # exact d2u/dx2, no finite differences
x = u_xx.graph.input("x")
loss = (u_xx.output + math.pi**2 * (math.pi * x).sin()).powi(2).mean()
plan = loss.compile_cpu()  # frozen execution plan, reused every step

points = {"x": quabla.Tensor.linspace(0.05, 0.95, 8).reshape([8, 1])}
params, adam = {"w": quabla.Tensor([1, 1], [2.5])}, quabla.Adam(learning_rate=0.05)
seed = quabla.Tensor([], [1.0])  # d loss / d loss
for _ in range(300):
    value, grads = plan.evaluate_value_and_vjp({**points, **params}, seed)
    params = adam.step(params, {"w": grads["w"]})
print(f"w = {params['w'].to_flat_list()[0]:.6f}, loss = {value.to_flat_list()[0]:.1e}")
```

```text
w = 3.141593, loss = 1.5e-13
```

The [Quickstart](#quickstart) covers dtypes, masks, and the multi-backend
compiler facade.

## Status

Quabla v0.1 is a research-grade, source-only pre-release, published as a git
tag and GitHub Release without prebuilt wheels; the 0.x API may change between
releases. Python functions are traced into a rank-N tensor IR, transformed
with composable automatic differentiation, and compiled to frozen execution
plans by a Rust core exposed to Python through PyO3.

The primary Python API is the rank-N path: the `quabla.Compiler` facade
(`Compiler.trace(fn, input_specs) -> Program`, `Program.jvp(...)` /
`Program.vjp(...)`, `Program.compile("cpu" | "cuda" | "mlx")`),
`trace_tensor(...)`, the eager `Tensor` class, and the `tensor_*_fn` helpers
(`tensor_value_and_grad_fn`, `tensor_jit_fn`, `tensor_vmap_fn`,
`tensor_hvp_scalar_fn`, their `_cuda`/`_mlx` variants, and the control-flow
builders `tensor_cond`, `tensor_fori_loop_region`, `tensor_scan_region`). The
2D `Matrix`, `trace(...)`, `TraceGraph`, `grad*`, and `jit(...)` API is legacy:
kept for compatibility, outside the compiler facade, without new backend
features.

"JAX-style" describes the API only: Quabla does not depend on JAX and is not
affiliated with JAX or with the `nabla-ml` project. It was formerly named
Nabla and was renamed to Quabla (after □, the d'Alembert operator) before
v0.1.

## Backend Support

`float64` programs execute as `f32` on CUDA and MLX; `bool` values are held as
`f32` `0`/`1` on both devices. The authoritative per-feature status and its
validation records are in [docs/jax_like_roadmap.md](docs/jax_like_roadmap.md).

| Backend | Platform | Build feature | Execution dtype | Autodiff | Control flow | Notable limitations |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | macOS, Linux | default | `f64` reference; each `float32` op is the `f64` result rounded to `f32` | Runtime and symbolic JVP/VJP, dense Jacobian and Hessian, HVP, `vmap` JVP/VJP/HVP | `cond`, `fori`, `scan` regions with JVP, VJP, and forward-over-reverse HVP | Interprets frozen plans (no machine-code JIT) |
| CUDA | Linux, NVIDIA driver | `cuda` | `f32` | Symbolic JVP/VJP plans, multi-output value-and-gradient, `vmap` JVP/VJP/HVP, device SGD/Adam | `cond` via one host predicate readback; `fori`/`scan` as fused kernels for pure-elementwise bodies, with first-order VJP and restricted HVP | Loop bodies with matmul, reductions, `solve`, or nested regions are rejected; `cond` inside a loop body is rejected; requires `libnvrtc` at runtime (cuBLAS optional, cuSOLVER for `solve`) |
| CUDA + NCCL | Linux, two or more GPUs on one node | `cuda-nccl` | `f32` | Scalar value-and-gradient with all-reduced replicated parameter gradients | As CUDA | Single node; equal axis-zero batch shards; mapped-input gradients rejected; optimizer update on the host; requires a loadable `libnccl.so` |
| MLX | macOS, Apple silicon | `mlx` | `f32` | Symbolic JVP/VJP, multi-output value-and-gradient, `vmap` JVP/VJP, device Adam | `cond` via one host predicate readback; `fori`/`scan` dispatched from the host on device-resident arrays, with first-order VJP and forward-over-reverse HVP | `solve` rejected (MLX 0.32.2 `linalg::solve` is CPU-stream only); `vmap` HVP not lowered; no fused Metal loop kernels |

## Installation From Source

Prerequisites:

- Rust stable toolchain (`rustup`).
- Python 3.10 or newer and maturin 1.x (1.9.3 or later).
- MLX builds: macOS on Apple silicon, CMake, and Xcode's Metal Toolchain
  (`xcodebuild -downloadComponent MetalToolchain`). The Python `mlx` wheel is
  not used. The build writes the compiled Metal library to
  `~/.mlx/lib/<key>/mlx.metallib`, or to `$MLX_RS_METAL_PATH` when that
  variable is set at build time; the extension loads it from that path at run
  time, so keep it in place (or rebuild after removing it).
- CUDA builds: Linux with an NVIDIA driver. CUDA libraries are loaded at run
  time: `libnvrtc.so` is required, `libcublas` is used for rank-two `f32`
  GEMM when present (otherwise an NVRTC tiled kernel), and `libcusolver` is
  required for `solve`. Put their directory on `LD_LIBRARY_PATH`. The
  two-GPU validation used CUDA 12.6.
- CUDA + NCCL builds: additionally NCCL loadable as `libnccl.so` (for example
  through `LD_LIBRARY_PATH`) and at least two CUDA devices. Ordinary `cuda`
  builds neither link nor load NCCL.

Create a virtual environment and build the extension in place:

```sh
git clone https://github.com/latteine1217/quabla.git
cd quabla
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "maturin>=1.9.3,<2"

# Pick one:
maturin develop --release                        # CPU only
maturin develop --release --features mlx         # macOS, Apple silicon
maturin develop --release --features cuda        # Linux, CUDA
maturin develop --release --features cuda-nccl   # Linux, CUDA + NCCL

python -c "import quabla; print(quabla.Compiler().capabilities())"
```

`Compiler.capabilities()` reports which targets this build contains, for
example `{'cpu': True, 'cuda': False, 'mlx': True}` for an MLX build. It is
not a hardware probe.

## Quickstart

Trace a scalar loss in `float32`, mask it with a comparison, and evaluate its
value and gradients on the CPU:

```python
import quabla


def loss(x, w):
    y = (x * w).tanh()
    mask = x > 0.0  # comparison operators return quabla.bool_ masks
    return quabla.where(mask, y, 0.0).powi(2).sum()


specs = [("x", [4], quabla.float32), ("w", [4], quabla.float32)]
inputs = {
    "x": quabla.Tensor([4], [-1.0, 0.5, 1.0, 2.0], dtype=quabla.float32),
    "w": quabla.Tensor([4], [0.3, -0.2, 0.1, 0.4], dtype=quabla.float32),
}

mask = inputs["x"] > 0.0
print(mask.dtype, mask.to_flat_list())

value_and_grad = quabla.tensor_value_and_grad_fn(loss, specs)
value, grads = value_and_grad(inputs)
print(value.dtype, value.to_flat_list())
print(grads["w"].to_flat_list())
```

```text
bool [0.0, 1.0, 1.0, 1.0]
f32 [0.4608122408390045]
[0.0, -0.09867792576551437, 0.19735585153102875, 1.484932780265808]
```

Continuing in the same session, the same function goes through the compiler
facade, which compiles one traced program for every target in the build:

```python
compiler = quabla.Compiler()
print(compiler.capabilities())

program = compiler.trace(loss, specs)
for target in ("cpu", "mlx", "cuda"):
    if compiler.capability(target):
        print(target, program.compile(target)(inputs).to_flat_list())

# Symbolic VJP: one gradient Program per input, compiled like any other.
grad_w = program.vjp("loss_bar")["w"].compile("cpu")
seed = quabla.Tensor([], [1.0], dtype=quabla.float32)
print(grad_w({**inputs, "loss_bar": seed}).to_flat_list())
```

Output of an MLX build on Apple silicon (a CUDA build prints a `cuda` line
instead of the `mlx` line):

```text
{'cpu': True, 'cuda': False, 'mlx': True}
cpu [0.4608122408390045]
mlx [0.4608122408390045]
[-0.0, -0.09867792576551437, 0.19735585153102875, 1.484932780265808]
```

The masked coordinate has a zero gradient: `where` routes no derivative
through the unselected branch or its predicate. See
[examples/README.md](examples/README.md) for PINN training on each
backend.

## Known Limitations

- The CPU backend interprets frozen plans; there is no machine-code JIT.
- Shapes are static (bounded per-batch-size specialization, no symbolic
  dimensions), and dtypes are limited to `float32`, `float64`, and `bool`.
- Indexing takes static Python integers; dynamic index tensors and
  boolean-mask indexing are unsupported.
- CUDA loop bodies must be pure elementwise. MLX rejects `solve` and has no
  `vmap` HVP lowering.
- Data parallelism is single-node CUDA + NCCL only.

The full list is in [docs/api.md](docs/api.md#known-limitations).

## Documentation

- [docs/api.md](docs/api.md): compiler facade, API reference, and the full
  list of known limitations.
- [examples/README.md](examples/README.md): PINN examples and benchmarks.
- [docs/jax_like_roadmap.md](docs/jax_like_roadmap.md): roadmap, per-phase
  status (P0-P7, dtype phases D1-D6), validation records, and open items.
- [docs/design.md](docs/design.md): original core design notes.
- [CHANGELOG.md](CHANGELOG.md): release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to propose changes and which
  [development gates](CONTRIBUTING.md#development-gates) to run.
- [SECURITY.md](SECURITY.md): how to report a vulnerability.
- [CITATION.cff](CITATION.cff): citation metadata for academic use.

## License

Quabla is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT), at your option.
