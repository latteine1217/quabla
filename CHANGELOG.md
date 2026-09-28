# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, minor releases may change the public API.

## [0.1.0] - Unreleased

First public release: a research-grade, source-only pre-release published as a
git tag and GitHub Release, without prebuilt wheels. The project was developed
under the name Nabla and renamed to Quabla before this release; the Python
package, import name, crates, types (`Quabla*`), and environment variables
(`QUABLA_*`) all use the new name.

### Added

- **Tensor IR and dtypes.** A rank-N tensor IR with per-node dtypes
  `float64`, `float32`, and `bool`. Tensors of different float dtypes must be
  converted explicitly with `astype`; Python scalars are weakly typed and adopt
  the other operand's dtype. On the CPU, each `float32` operation is the `f64`
  result rounded to `f32`, which gives an `f32` reference for backend parity.
- **Automatic differentiation.** Runtime and symbolic JVP and VJP, dense
  Jacobians and Hessians, Hessian-vector products, and `vmap` composed with JVP,
  VJP, and HVP.
- **Operations.** Elementwise arithmetic and math (`tanh`, `exp`, `log`,
  `sqrt`, `sin`, `cos`, integer `powi`), `relu`, `abs`, `sigmoid`, `softplus`,
  matmul and a scoped `einsum` subset, multi-axis `sum`/`mean`/`max`/`min`
  reductions, L2 `norm`, contiguous-slice indexing, static-index `gather` and
  `scatter_add`, `solve`, `solve_triangular`, `cholesky`, `tril`/`triu`, and
  `where`.
- **Bool masks.** Comparisons (`greater`, `greater_equal`, `less`,
  `less_equal`, `equal`, `not_equal`, and the `<`, `<=`, `>`, `>=` operators),
  logical `&`, `|`, `~`, `isfinite`, `isnan`, and `any`/`all` reductions.
  `where` and `cond` accept bool predicates. `==` and `!=` keep Python identity
  semantics, and `.gt()` keeps returning a float mask.
- **Structured control flow.** `cond`, `fori_loop`, and `scan` regions with
  explicit captures, supporting JVP, VJP, and forward-over-reverse HVP. Traced
  Python branches are rejected instead of being specialized silently, and
  bounded batch JIT specialization caches plans per batch size.
- **Compiler.** A `Compiler` → `Program` → `compile(target)` → `Executable`
  facade in Rust and Python, with ordered multi-output programs in Rust. It
  includes scalar constant folding, dead-node pruning, common-subexpression
  elimination, layout canonicalization, buffer planning, elementwise fusion
  regions, a structured kernel IR, `lower_text()`, and a `stablehlo_text()`
  interoperability probe for a static `f64` subset.
- **CPU backend.** The default `f64` reference backend, which interprets frozen
  execution plans.
- **CUDA backend (Linux, `cuda` feature).** NVRTC-generated fused elementwise
  kernels, cuBLAS matmul, cuSOLVER `solve`, device-resident `fori`/`scan` kernels
  for a restricted body subset, device-predicate `cond`, retained device
  buffers, and device SGD/Adam optimizers. CUDA libraries are loaded at runtime,
  so building does not require a CUDA toolkit.
- **NCCL data parallelism (Linux, `cuda-nccl` feature).** Single-node data
  parallelism that shards axis-zero batches across explicit GPUs, all-reduces
  gradients according to a typed sharding plan, reuses NCCL communicators across
  calls, and recreates them after a failed call.
- **MLX backend (Apple silicon, `mlx` feature).** `f32` execution with
  multi-output value-and-gradient plans, device-resident Adam training, and
  `cond`/`fori`/`scan` regions. MLX evaluation is serialized across threads.
- **Python API.** A PyO3 extension exposing `Tensor`, `trace_tensor`, the
  `tensor_*_fn` transform helpers, `Compiler`, dtype objects
  (`quabla.float32`, `quabla.float64`, `quabla.bool_`), and stateless random
  keys with Glorot initialization. The earlier 2D `Matrix`/`trace` API remains
  available as a legacy interface.
- **Rust macros.** The `forward_diff!` and `forward_gradient!` procedural
  macros in `quabla-macros`, a restricted source-level forward-mode prototype.
- **Examples and validation.** Poisson and MLP PINN examples for the CPU, CUDA,
  and MLX backends, benchmarks, and two-GPU data-parallel parity and training
  validation scripts.
- **Project infrastructure.** Dual MIT/Apache-2.0 license files, a GitHub
  Actions workflow for the Linux development gates (fmt, ruff, clippy
  including a compile-only CUDA/NCCL check, `cargo test`, and the Python test
  matrix), and an on-demand macOS MLX check.

### Validated

- CPU, MLX, and CUDA backends agree within documented `f32` tolerances across
  the Rust and Python test suites. The MLX suites run on Apple silicon (M3), the
  CUDA suites on a GTX 1660 SUPER, and the data-parallel suites on two RTX 3090s.
- Two-GPU data-parallel training matches single-GPU and CPU references over
  200 steps (maximum relative loss difference 2.7e-7), and scaled runs up to
  65,536 rows and hidden width 512 also match. The steady-state collective
  median is 0.106 ms in the 256-row, width-32 configuration.

### Known Limitations

- There is no general machine-code JIT; CPU plans are interpreted.
- Shapes are static; there are no symbolic dimensions.
- CUDA loop bodies cannot contain matmul, reductions, or nested regions.
- `float16`, `bfloat16`, and integer dtypes are not available, and host storage
  is `f64`.
- MLX rejects `solve` because MLX 0.25.3 provides it only on a CPU stream.
- Data parallelism is limited to a single node, and parameter updates happen on
  the host.

See the README's Known Limitations section and `docs/jax_like_roadmap.md` for
details and planned work.

[0.1.0]: https://github.com/latteine1217/quabla/releases/tag/v0.1.0
