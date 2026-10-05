# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, minor releases may add to the public Python API and deprecate names;
deprecated names keep working until 1.0 (see
[CONTRIBUTING.md](CONTRIBUTING.md)).

## [Unreleased]

### Changed

- Traced `gather` and `scatter_add` lower to native IR nodes instead of one
  slice, or one padded add, per index, so a staged program and its gradient
  keep a constant node count. With 16000 indices into a 4000 x 4 array, jit
  `scatter_add` takes 0.22 ms instead of 513 ms on the CPU and 0.42 ms
  instead of 201 ms on Apple silicon, and jit `grad` of a gather loss 1.5 ms
  instead of 514 ms and 0.46 ms instead of 256 ms. On CUDA (GTX 1660 SUPER)
  with 4000 indices, the first call compiles in 22 ms instead of 320 s and a
  call takes 0.26 ms instead of 184 ms. Results, including gradients, JVPs,
  and HVPs, are bitwise unchanged on the CPU, CUDA, and Metal, except that a
  `-0.0` entry is no longer turned into `+0.0` by an added zero.

## [0.2.2] - 2026-10-06

### Changed

- `float32` Cholesky JVPs, VJPs, and second derivatives use the native
  kernels instead of a scalar expansion whose graph grew as O(n^3); the CPU
  rounds every jet operation to `f32`. A `float32` Cholesky gradient at
  n = 512 takes 0.15 s on the CPU, 0.09 s on CUDA (GTX 1660 SUPER), and
  0.09 s on Apple silicon. Staged gradients, JVPs, and HVPs on the CPU and
  on Metal, and gradients on CUDA, are bitwise unchanged; forward-over-forward
  results, and CUDA JVPs and HVPs, can differ in the last bit.

## [0.2.1] - 2026-10-06

### Changed

- Native Cholesky derivative kernels on CUDA, and the Metal primal and
  derivative kernels, run one block or threadgroup of 256 threads per matrix
  instead of one thread. Results are bitwise identical to v0.2.0. A
  `float64` Cholesky gradient at n = 512 takes 0.08 s instead of 2.6 s on a
  GTX 1660 SUPER, and 0.09 s instead of 5.4 s on Apple silicon.

### Fixed

- `from quabla import *` binds deprecated and legacy names without emitting
  a `DeprecationWarning` for each, so it no longer fails under
  `-W error::DeprecationWarning` or uses up the once-per-name warning that
  explicit access emits.

## [0.2.0] - 2026-10-06

The JAX-style Python API described in `docs/api_v0_2_design.md` (slices S0
through S9), released as a source-only pre-release like v0.1.0. All v0.1 call
forms remain available; migrated names warn once as described below.

### Added

- **Devices and AOT.** Explicit CPU/CUDA/MLX `jit`, `cuda:N`, built-target
  `devices()`, `ShapeDtype` and `.lower().compile()` with ordered output
  programs; eager typed MLX/CUDA lowering checks and a once-per-function
  warning for float64 device execution.
- **Training and control flow.** Pure pytree `optim.Adam`/`SGD`, CPU and
  device Adam `optim.Trainer`, and `cond`/`fori_loop`/`scan` wrappers with
  explicit region operands and the existing backend limits.
- **Distributed.** Experimental single-node
  `distributed.value_and_grad` with equal axis-zero shards, replicated
  parameter gradients, bounded signature caching and explicit Sum/Mean.
- **Migration.** `quabla.legacy` preserves the 2D API without warnings;
  migrated top-level names warn once, legacy `grad`/`jit` forms still
  dispatch, and `quabla.Adam` aliases the new Adam with its old `step`.
- **Memory and kernels.** Host tensors store `float32` and `bool` in 4 and 1
  bytes per element instead of widening to `f64`; native `Cholesky` on CPU,
  CUDA, and Metal with first- and second-order AD for logical-`float64`
  graphs; `fori`/`scan` gradients on CPU and MLX (and eligible CUDA loops)
  keep about 2*sqrt(T) carries instead of a full tape; dense Jacobians are
  staged in bounded basis batches.
- **Packaging.** The Python package is a mixed Rust/Python project: a
  pure-Python `quabla` package over the compiled `quabla._quabla` extension.
- **Arrays and NumPy.** `quabla.array`/`asarray` from nested lists, scalars,
  and NumPy arrays; `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`;
  `Tensor.numpy()`, `tolist()`, `item()`, `__array__`, and the buffer
  protocol; module-level math functions such as `quabla.sin`; `-x` and
  `x ** n` on traced values.
- **Pow.** A differentiable elementwise `Pow` op on CPU, CUDA, and MLX,
  exposed as `quabla.power` and `x ** y` for float and tensor exponents.
- **Function transforms.** `grad`, `value_and_grad`, `jvp`, `vjp`,
  `jacobian`, `hessian`, `jit`, and `vmap` with `argnums`, `has_aux`,
  pytrees (`quabla.tree`), and trace-on-first-call caching. Transforms nest
  inside traced functions, eager arrays and closed-over tracers are captured
  automatically, and the canonical PINN form
  `vmap(grad(grad(u)), in_axes=(0, None))` works. New error classes
  `QuablaError`, `TracerError`, `RetraceLimitError`, and
  `UnsupportedOperationError`.

### Changed

- `solve` with a triangular coefficient matrix uses substitution instead of
  LU, and the legacy `tensor_hessian_scalar_fn`/`tensor_hvp_scalar_fn` use
  symbolic forward-over-reverse for finite `float64` graphs. Results can
  differ from v0.1.0 in the last bits, and such Hessians are not bitwise
  symmetric.
- Rust: `DynamicTensor::data()` returns `Cow<'_, [f64]>` instead of
  `&[f64]`. The Rust crate APIs are not covered by the compatibility policy.

### Fixed

- `jit(static_argnums=...)` distinguishes static values that compare equal,
  such as `1`, `True`, and `1.0`, or `0.0` and `-0.0`, instead of reusing a
  program traced for another value; a NaN static value no longer retraces on
  every call.
- Traced `cond`, `fori_loop`, and `scan` bodies may ignore operands, and
  `cond` branches may return constants, as in eager execution.
- `grad(f, [0, 1])` treats a list of ints as `argnums` instead of the v0.1
  input-spec form.
- The float64-on-device warning points at the caller's line.
- Square-root derivatives are NaN for negative inputs, including `-inf`, on
  every backend; MLX previously returned 0 and CPU/CUDA returned 0 or inf at
  `-inf`.
- Symbolic Hessians, HVPs, and zero gradients no longer turn into NaN when an
  input or the loss value is infinite or NaN.

## [0.1.0] - 2026-09-28

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
- MLX rejects `solve` because MLX 0.32.2 provides it only on a CPU stream.
- Data parallelism is limited to a single node, and parameter updates happen on
  the host.

See the README's Known Limitations section and `docs/jax_like_roadmap.md` for
details and planned work.

[Unreleased]: https://github.com/latteine1217/quabla/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/latteine1217/quabla/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/latteine1217/quabla/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/latteine1217/quabla/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/latteine1217/quabla/releases/tag/v0.1.0
