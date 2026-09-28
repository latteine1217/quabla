# Quabla

Quabla is a Rust-native scientific machine learning (SciML) compiler and
runtime with a JAX-like programming model. Python functions are traced into a
rank-N tensor IR, transformed with composable automatic differentiation
(JVP, VJP, Jacobians, Hessians, Hessian-vector products, and `vmap`), and
compiled to frozen execution plans. Structured control flow (`cond`,
`fori_loop`, `scan`) is part of the IR rather than traced Python branches.
Plans run on a CPU reference backend (`f64`), on CUDA (Linux; NVRTC kernels
plus cuBLAS/cuSOLVER, with optional NCCL data parallelism), or on MLX (Apple
silicon). The Python API is a PyO3 extension. "JAX-like" describes the API
style only: Quabla does not depend on JAX and is not affiliated with JAX or
with the `nabla-ml` project.

The project was formerly named Nabla and was renamed to Quabla (after □, the
d'Alembert operator) before the v0.1 release.

## Status

Quabla v0.1 is a research-grade, source-only pre-release: it is released as a
git tag and GitHub Release, with no prebuilt wheels. The 0.x API may change
between releases. Backend support is validated operation by operation;
unsupported operations fail explicitly instead of falling back to the host.

The primary Python API is the rank-N path:

- `quabla.Compiler` facade: `Compiler.trace(fn, input_specs) -> Program`,
  `Program.jvp(...)` / `Program.vjp(...)`, `Program.compile(target) ->
  Executable` for `"cpu"`, `"cuda"`, or `"mlx"`.
- `trace_tensor(...)` with `TraceTensor` values, the eager `Tensor` class, and
  the `tensor_*_fn` helpers (`tensor_value_and_grad_fn`, `tensor_jit_fn`,
  `tensor_vmap_fn`, `tensor_hvp_scalar_fn`, their `_cuda`/`_mlx` variants, and
  the structured control-flow builders `tensor_cond`,
  `tensor_fori_loop_region`, and `tensor_scan_region`).

The 2D `Matrix`, `trace(...)`, `TraceGraph`, `grad*`, and `jit(...)` API is
legacy: it remains available for compatibility but is outside the compiler
facade and does not receive new backend features.

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
