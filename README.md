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
| MLX | macOS, Apple silicon | `mlx` | `f32` | Symbolic JVP/VJP, multi-output value-and-gradient, `vmap` JVP/VJP, device Adam | `cond` via one host predicate readback; `fori`/`scan` dispatched from the host on device-resident arrays, with first-order VJP and forward-over-reverse HVP | `solve` rejected (MLX 0.25.3 `linalg::solve` is CPU-stream only); `vmap` HVP not lowered; no fused Metal loop kernels |

## Installation From Source

Prerequisites:

- Rust stable toolchain (`rustup`).
- Python 3.10 or newer and maturin 1.x (1.9.3 or later).
- MLX builds: macOS on Apple silicon, CMake, and Xcode's Metal Toolchain
  (`xcodebuild -downloadComponent MetalToolchain`). The Python `mlx` wheel is
  not used.
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
[Examples And Benchmarks](#examples-and-benchmarks) for PINN training on each
backend.

## Known Limitations

- No machine-code JIT for the CPU backend, which interprets frozen plans.
  CUDA generates NVRTC kernels; MLX dispatches MLX operations.
- Shapes are static. Variable batch sizes use bounded per-size
  specialization (`tensor_jit_batch_fn` and its variants); there are no
  symbolic dimensions.
- Dtypes are `float32`, `float64`, and `bool`. `float16`/`bfloat16`, integer
  tensors, native device `f64`, and mixed-precision training are not
  implemented. Host storage is `f64` for every dtype.
- `gather`/`scatter_add` take static Python integer indices; dynamic index
  tensors, boolean-mask indexing, strided or empty slices, and ellipsis are
  unsupported.
- CUDA loop bodies must be pure elementwise: matmul, reductions, `solve`,
  nested regions, and `cond` inside `fori`/`scan` are rejected. `cond`
  results and loop-region inputs must be floating, not `bool`.
- Derivatives beyond forward-over-reverse of `fori`/`scan` regions are
  explicit errors. Vmapped `cond` predicates are rejected at trace time.
- MLX rejects `solve` (and `solve_triangular`, which composes it), and
  `vmap` HVP has no MLX lowering. Native CUDA/MLX Cholesky and
  triangular-solve kernels are not implemented.
- Data parallelism is single-node CUDA + NCCL only: equal axis-zero batch
  shards, replicated parameter gradients, and an optimizer on the host. There
  is no multi-node transport, tensor parallelism, or sharded matmul.
- StableHLO export (`stablehlo_text`) covers only a small inspection subset
  and is not an execution path.
- The legacy 2D `Matrix`/`TraceGraph` API is not migrated to the facade.

## Development Gates

Run these before treating a change as verified. CI (`.github/workflows/ci.yml`)
runs the Linux gates on every push and pull request; the macOS `mlx` check
(`macos-mlx.yml`) runs on demand, and the GPU runtime suites run manually on
GPU hosts.

```sh
cargo fmt --all --check
ruff check tests examples
cargo clippy --workspace --all-targets -- -D warnings
# macOS:
cargo clippy --workspace --all-targets --features quabla-core/mlx -- -D warnings
# Linux:
cargo clippy --workspace --all-targets --features quabla-core/cuda-nccl \
  -- -D warnings
cargo test --workspace

maturin develop --release              # add --features mlx or --features cuda
python tests/python/test_matrix.py
```

Device suites are opt-in through environment variables:

```sh
# MLX build on Apple silicon:
QUABLA_MLX_TEST=1 python tests/python/test_matrix.py
# CUDA build with an NVIDIA GPU:
QUABLA_CUDA_TEST=1 python tests/python/test_matrix.py
QUABLA_CUDA_TEST=1 cargo test -p quabla-core --features cuda
# Two CUDA GPUs and a loadable libnccl.so:
QUABLA_CUDA_NCCL_TEST=1 cargo test -p quabla-core --features cuda-nccl
```

The Rust MLX tests run whenever `quabla-core/mlx` is enabled on macOS
(`cargo test --workspace --features quabla-core/mlx`).

## Documentation And License

- [docs/jax_like_roadmap.md](docs/jax_like_roadmap.md): execution roadmap,
  per-phase status (P0-P7, dtype phases D1-D6), and validation records.
- [docs/design.md](docs/design.md): original core design notes.
- [CHANGELOG.md](CHANGELOG.md): release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to propose changes and which gates
  to run.
- [SECURITY.md](SECURITY.md): how to report a vulnerability.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): community standards and how to
  report a conduct issue.
- [CITATION.cff](CITATION.cff): citation metadata for academic use.

Quabla is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT), at your option.

## Examples And Benchmarks

The commands below assume the virtual environment from
[Installation From Source](#installation-from-source) is active and the
extension was built with the named feature.

CPU build: a one-dimensional Poisson PINN and the frozen CPU plan benchmark.

```sh
python examples/pinn_poisson.py
python examples/benchmark_tensor_cpu.py
```

MLX build (`--features mlx`): one-graph Poisson PINNs keep parameters and Adam
moments on the MLX GPU stream; the two-layer MLP example checks PDE and
boundary residuals plus trained parameters against CPU:

```sh
python examples/pinn_poisson_mlx.py
python examples/pinn_mlp_mlx.py
python examples/benchmark_pinn_mlx.py
python examples/benchmark_mlp_value_and_grad_mlx.py
python examples/benchmark_forward_mlx.py
python examples/benchmark_loop_value_and_grad_mlx.py
```

Use `examples/benchmark_forward_mlx.py --device-only` to exclude only output
readback; dynamic host input upload remains part of that timing. Add
`--retain-static-inputs` to upload model weights once and time only dynamic
input upload on later calls. `examples/benchmark_pinn_mlx.py --hidden-layers N
--hidden-width W` replaces the one-parameter model with a tanh MLP.

The MLX benchmarks report reproducibility metadata for both the one-parameter
Poisson optimizer path and an end-to-end batched MLP value-and-gradient path.
The former excludes diagnostic readbacks; the latter includes input uploads and
loss/gradient readbacks, so neither is a CUDA/JAX comparison.

CUDA build (`--features cuda`): a two-point batched Poisson PINN whose Adam
optimizer state and collocation tensors stay on the GPU, and a two-layer,
four-parameter Poisson PINN with combined residual and boundary losses:

```sh
python examples/pinn_poisson_cuda.py
python examples/pinn_mlp_cuda.py
```

Measure end-to-end broadcast-batched matmul or fused elementwise-chain
throughput (including the current host/device transfers), and device-resident
two-layer Poisson PINN Adam steps with separate compile, warm-up, and
synchronized training timings:

```sh
python examples/benchmark_tensor_cuda.py
python examples/benchmark_tensor_cuda.py --elementwise --rows 1024 --inner 1024
python examples/benchmark_tensor_cuda.py --rank-two --device-resident
python examples/benchmark_pinn_cuda.py
```

CUDA + NCCL build (`--features cuda-nccl`, two GPUs):
`examples/validate_data_parallel_cuda.py` checks single-call two-GPU parity
against the CPU reference, and
`examples/validate_data_parallel_training_cuda.py` checks 200-step training
parity against one GPU and CPU. Both use CUDA ordinals 0 and 1 (select the
devices with `CUDA_VISIBLE_DEVICES`) and need `libnccl.so` to be loadable.
On any Linux machine with two GPUs and NCCL:

```sh
maturin develop --release --features cuda-nccl
python examples/validate_data_parallel_cuda.py
python examples/validate_data_parallel_training_cuda.py
# Rust two-GPU tests: schedule parity, communicator recovery, lifecycle.
QUABLA_CUDA_NCCL_TEST=1 cargo test -p quabla-core --features cuda-nccl \
  --test tensor_ir on_two_gpus -- --nocapture
```

`--grid`, `--hidden`, `--steps`, and `--modes` scale or restrict the training
check, and `--output` also writes its JSON report to a file (`--help` lists
all flags); `--modes single_gpu,cpu` is a dry run for a host with one GPU and
no NCCL.

Rust-only examples of the core crate (no Python extension needed):

```sh
cargo run -p quabla-core --example lotka_volterra
cargo run -p quabla-core --example fit_lotka_volterra
```

The first prints the Lotka-Volterra loss and its four-parameter gradient; the
second reports initial loss, final loss, optimization steps, and the fitted
parameter vector.

## Compiler Facade

The rank-N compiler path has one explicit lifecycle. New Python integrations
should use this facade instead of coupling to a backend-specific execution
plan class. It is the canonical entrypoint for new rank-N compiler features;
the legacy 2D `Matrix` tracer is intentionally outside this migration boundary.
The [Quickstart](#quickstart) shows the lifecycle end to end.

`Program.jvp(input_name)` and `Program.vjp(cotangent_name)` produce new
programs that can be compiled through the same interface. `compile("cuda")`
and `compile("mlx")` use the existing verified backend lowerings; build
availability is reported by `compiler.capabilities()`, while unsupported IR
operations still fail explicitly during backend compilation or execution.
The `trace_tensor(...)` and `tensor_*_fn` helpers remain supported; most of
them compile through the facade internally, as listed below.

| Layer | Stable responsibility | Public boundary |
| --- | --- | --- |
| Frontend | Trace a Python function with fixed input shapes | `Compiler.trace(...) -> Program` |
| Transform | Build symbolic first-order coordinate JVP or named VJP programs | `Program.jvp(...)`, `Program.vjp(...)` |
| Compilation | Freeze reachability-pruned IR and select one target | `Program.compile(target, device_ordinal=0)` |
| Runtime | Bind eager `Tensor` inputs and execute the frozen plan | `Executable(inputs)` / `Executable.evaluate(inputs)` |
| Inspection | Report build-time target availability and frozen graph text | `Compiler.capabilities()`, `Program.lower_text()` |

`Compiler.capabilities()` answers only whether this extension was built with a
target backend. It is not a hardware probe and it does not guarantee that a
specific graph lowers: CUDA and MLX retain their operation-specific validation
rules. CUDA structured Scan HVP is a restricted facade capability: the
verified subset requires a fixed-bound, pure-elementwise Scan with same-shaped
explicit captures and either equal carry/output shapes, an equal-count output
reshape, or a per-step output that directly broadcasts a carry-shaped value.
Its symbolic JVP paired carry uses a leading extent of two. In the direct
broadcast case, `ScanVjp` and `ScanVjpJvp` aggregate output cotangents, and
their tangents for HVP, back to each carry lane before reverse replay;
broadcast capture gradients use device-side atomic reduction. General
unequal-count output graphs and indexed bodies remain explicit CUDA rejections.

Current migration status:

- The facade owns the new rank-N `TensorTraceGraph` lifecycle.
- Single-output CPU helpers (`tensor_jit_fn`, `tensor_*grad*_fn`, `tensor_vjp_fn`,
  `tensor_jvp_fn`, `tensor_jacobian_fn`, CPU vmap, batch, and cond helpers),
  `tensor_vmap_mlx_fn`, and the single-output CUDA helpers
  (`tensor_jit_cuda_fn`, `tensor_vmap_cuda_fn`,
  `tensor_vmap_hvp_scalar_cuda_fn`, `tensor_jit_batch_cuda_fn`) compile
  through `QuablaCompiler`; their public signatures and results are
  unchanged. The multi-output helpers (`tensor_value_and_grad_{mlx,cuda}_fn`,
  their batch variants, `tensor_vmap_{vjp,jvp}_{mlx,cuda}_fn`,
  `mlx_adam_loss_optimizer`, `cuda_adam_vjp_optimizer`, and
  `cuda_adam_loss_optimizer`) compile one ordered multi-output program through
  `QuablaCompiler::compile_many` and read their outputs by position. Retained
  MLX inputs, device-resident CUDA buffers, and Adam state stay in those
  executors rather than in `QuablaExecutable`; `docs/jax_like_roadmap.md`
  records that remaining gap and the helpers that stay outside the facade.
- The Rust `QuablaCompiler::compile` rejects a target missing from the build
  before lowering. These Python entrypoints, including `Program.compile`, keep
  their earlier errors in such a build: MLX constructs and fails on first
  execution, CUDA fails at compile time, each with the backend's build
  instructions. Check `Compiler.capability(target)` first when that matters.
- `Program.jvp(...)` is the one-named-input symbolic coordinate derivative;
  `Program.vjp(...)` returns one `Program` per original input. Runtime tangent
  maps and multi-output compiler programs (`QuablaMultiOutputProgram`) remain
  lower-level Rust APIs.
- CPU facade execution is covered by Rust and installed-extension Python tests.
  CUDA facade parity is verified on the Linux GTX 1660 SUPER host for a
  nonlinear scalar loss, its symbolic coordinate JVP, and VJP programs for
  every input. The same host also verifies a nonlinear fixed-bound Scan HVP
  through the paired-carry CUDA path against CPU, including a carry/output
  reshape with equal element count, plus primal, first-order VJP, and HVP
  parity for a direct `[2, 1] -> [2, 3]` broadcast Scan. `Program.compile`
  shares its MLX and CUDA lowering with `compile_mlx()`/`compile_cuda()`;
  MLX CPU parity for those plans is covered by the Rust MLX tests and the
  Python matrix with `QUABLA_MLX_TEST=1` on Apple silicon.

The corresponding Rust lifecycle is
`QuablaCompiler -> QuablaProgram -> QuablaExecutable` in
`quabla_core::compiler`. The ownership boundaries are:

```text
Python frontend / PyO3 -> TensorTraceGraph -> TensorIr / AD transforms
                       -> frozen TensorExecutionPlan -> CPU | CUDA | MLX
```

## Current Capabilities

### Rank-N Tensor API

- Pure-Rust `TensorIr` compiler core (`quabla_core::tensor_ir`) for dynamic
  rank-N tensors with typed dtypes, CPU evaluation, direct JVP and
  reverse-mode VJP with cotangent reduction back through broadcast axes,
  symbolic JVP/VJP graph transforms, structured `Cond`/`Fori`/`Scan` regions,
  and deterministic lowering text. The Python rank-N API below traces into it;
  the legacy 2D `TraceGraph` is a separate tracer.
- Python-facing eager `Tensor` class backed by contiguous row-major Rust storage,
  with `zeros`/`ones`/`full`/`arange`/`linspace`/`eye` creation, positive runtime
  shape validation, rank-N trailing-axis broadcasting for add/subtract/multiply/divide,
  numeric scalars on either side of those arithmetic operations,
  scalar `**` exponents,
  NumPy-style batched `matmul`, rank-N `quabla.concat([...], axis=...)`,
  permutation-validated `transpose(axes=None)`, global or single-axis `sum`/`mean`/L2 `norm`,
  common elementwise math (`tanh`, `exp`, `log`, `sqrt`, `sin`, `cos`, `powi`),
  `gt(...)` masks, `maximum(...)`/`minimum(...)`, broadcasted `quabla.where(...)`, materialized `broadcast_to(shape)`,
  and element-count-preserving reshape.
  Eager `Tensor` operations run immediately on the host; differentiation and
  compilation apply to traced functions, whose arguments are `TraceTensor`
  values (see `trace_tensor` below). `Tensor.slice(...)` returns a
  zero-copy, read-only `TensorView` with explicit shape, strides, and offset;
  source tensors and views share immutable storage, while every arithmetic
  operation returns a new contiguous allocation.
- Neural and linear-algebra primitives on eager and traced tensors: `relu`
  (zero subgradient at zero), `abs`, `sigmoid`, numerically stable
  `softplus`, rank-N `tril`/`triu` over the last two axes, and rank-2
  `solve(rhs)` (partial-pivot LU), `solve_triangular(rhs, lower=True,
  transpose=False)`, and `cholesky()`, each with JVP/VJP rules. `solve`
  rejects non-square, rank-mismatched, and singular inputs; on CUDA it lowers
  to cuSOLVER `Sgetrf`/`Sgetrs`, while MLX rejects it.
- Element dtypes `quabla.float32` and `quabla.float64` (dtype phase D1).
  `Tensor(shape, data, dtype=quabla.float32)` rounds `data` to `f32`, and
  `Tensor.dtype`/`Tensor.astype(dtype)` and `TraceTensor.dtype`/
  `TraceTensor.astype(dtype)` inspect and convert; `to_flat_list()` of a
  `float32` tensor returns the rounded values. Input specs accept `(name,
  shape)` (still `float64`) or `(name, shape, dtype)`. Mixing `float32` and
  `float64` tensors raises an error that asks for an explicit `astype`, while
  Python scalars adopt the tensor's dtype. Host data bound to a `float32`
  input is rounded like a JAX jit argument. On the CPU each `float32` op is the
  `f64` result rounded to `f32` (bit-exact IEEE for `+ - * / sqrt`), which
  serves as the reference for CUDA and MLX; both execute `float32` natively and
  keep lowering `float64` programs to `f32` kernels as before. `kernel_ir()`
  reports `"f32"`/`"f64"`. Host storage stays `f64`, factories such as
  `zeros`/`arange` create `float64`, and batch-specialized functions
  (`tensor_jit_batch_fn` and its value-and-grad variants) trace `float64`
  inputs.
- Boolean masks with dtype `quabla.bool_` (dtype phase D2; `str` is `"bool"`,
  and `lower_text`/`stablehlo_text` print it as `i1`). `Tensor` and
  `TraceTensor` provide `greater`, `greater_equal`, `less`, `less_equal`,
  `equal`, `not_equal` (also as `quabla.greater(a, b)` etc.) and the
  operators `<`, `<=`, `>`, `>=`; comparisons follow IEEE semantics, so any
  comparison with `NaN` is false except `not_equal`. `==`/`!=` are not
  overloaded: tensors keep identity equality and stay hashable. Bool masks
  combine with `logical_and`/`logical_or`/`logical_not` or `&`/`|`/`~`
  (bool operands only), `isfinite()`/`isnan()` classify float tensors, and
  `any`/`all` reduce with the `axis`/`keepdims` contract of `sum`. `where`
  and `tensor_cond` accept bool predicates as well as the existing 0/1
  float masks, and `quabla.where` accepts Python scalar branches, so
  `quabla.where(x.isfinite(), x, 0.0)` guards `NaN`/`inf` before any
  arithmetic (a float mask product such as `x.gt(0.0) * x` still yields
  `NaN`, because `0 * NaN` is `NaN`). `gt()` keeps returning a 0/1 float mask
  of the operand dtype. In arithmetic a bool operand becomes 0/1 of the
  other operand's float dtype (`residual_f32 * mask` is `float32`); with a
  Python scalar it becomes a weak `float64` that still adopts a later
  `float32` operand (`mask * 2.0 + residual_f32` is `float32`). Arithmetic
  between two bool tensors, unary math, and reductions other than
  `any`/`all` on bool raise an error that names `astype`, which converts in
  both directions (nonzero and `NaN` become true). A single-element eager
  bool tensor works in `if`; larger ones raise like NumPy, float tensors
  keep their previous always-true truthiness, and `TraceTensor` still
  refuses Python control flow. Comparisons, logical ops, and `any`/`all`
  have zero derivatives, `where` gives no gradient to its predicate, bool
  inputs are omitted from gradient dictionaries, and requesting a
  derivative with respect to a bool input or of a bool output is an error.
  CUDA and MLX hold bool values as `f32` `0`/`1`; `cond` results and loop
  (`fori`/`scan`) region inputs must stay floating for now.
- Python `trace_tensor(fn, input_specs)` bridge for the rank-N `TensorIr` core.
  `TensorTraceGraph.stablehlo_text(output_node_id)` exports the verified static
  `f64` input/add/multiply/tanh subset as deterministic textual StableHLO for
  compiler-tool inspection. It rejects unsupported operations and is not an
  execution backend or a complete StableHLO lowering.
- `tensor_jacobian_fn(fn, input_specs, input_name)` freezes one rank-N trace
  and returns an output-flat by input-flat dense Jacobian for the selected input.
  Its `TraceTensor` values currently support broadcasted add/subtract/multiply/divide,
  batched `matmul`, rank-N `concat`, `stack([...], axis=...)`, `slice(axis, start, stop)`, `broadcast_to(shape)`, rank-N `transpose`, `tanh`, `exp`, `sin`, `cos`, `sqrt`, non-negative integer `powi`, `log`, reshape, global or single-axis `sum`/`mean`/L2 `norm`, `maximum`/`minimum`, `gt`/`where` masks, and the bool comparison, logical, `isfinite`/`isnan`, and `any`/`all` operations. `stack` is composed from reshape plus concat, so it inherits the same direct and symbolic CPU/CUDA AD rules. `concat` is linear: direct and symbolic VJP split the upstream cotangent with internal slice nodes, while its JVP and mixed second-direction transform concatenate the corresponding tangents. `slice` supports normalized negative axes and uses a zero-padded internal reverse node, keeping direct and symbolic gradients on the selected original coordinates. `broadcast_to` is a dedicated shape node whose VJP reduces repeated axes back to the input shape. `sqrt` is a native IR primitive: negative values follow IEEE floating-point `NaN` semantics, while every derivative order at zero is defined as zero, avoiding `log(0)` during higher-order AD. Comparisons are explicitly non-differentiable; `where` routes VJP/JVP contributions only through the selected data branch. `maximum` and `minimum` are composed from those primitives and route equality subgradients to their right operand. `TensorTraceGraph.evaluate_vjp(...)` and
  `TensorTraceGraph.evaluate_jvp(...)` execute the corresponding rank-N CPU
  reverse and forward transforms. `TensorTraceGraph.hessian_scalar(...)`
  computes an exact dense Hessian for one named input and a scalar output using
  mixed second-direction AD, not finite differences.
  `TensorTraceGraph.hvp_scalar(...)` returns the corresponding exact
  Hessian-vector product without materializing that dense matrix.
  `TensorTraceResult.symbolic_jvp(input_name)` instead emits a new transformable
  rank-N trace whose output is the coordinate JVP; it can be applied again for
  second derivatives and then differentiated with VJP with respect to model
  parameters. Its rules cover every current rank-N `TensorIr` primitive.
  A `TraceTensor` cannot be used as a Python boolean, preventing accidental
  data-dependent host branches during tracing; use `quabla.where` for
  elementwise selection, or the structured `tensor_cond` and loop-region APIs
  described below.
  `tensor_fori_loop(lower, upper, body, init)` statically unrolls a
  fixed-bounds, shape-preserving TraceTensor carry into that same IR.
  `tensor_fori_loop_region(lower, upper, body, init, operands)` instead
  traces `body(index, carry, *operands)` once into a runtime CPU loop region;
  captures must be explicit and the scalar `index` is a TraceTensor.
  On Linux CUDA, its first-order VJP also lowers to a device kernel when the
  body is pure elementwise and every explicit capture broadcasts to the carry
  shape. The kernel keeps a per-element carry tape on device; grouped `ForiVjp`
  results share that tape and reverse traversal. `tensor_scan_region` also
  lowers primal and first-order VJP Scan when its output matches the carry
  shape. Scalar and trailing-axis broadcast captures use device-side atomic
  gradient reduction. CUDA also lowers `ForiVjpJvp` for pure-elementwise,
  fixed-bound regions whose explicit captures match the carry shape, keeping
  primal and tangent carry tapes on device. CUDA `ScanVjpJvp` supports the
  same pure-elementwise fixed-bound subset with broadcast-compatible explicit
  captures and matching carry/output shapes,
  plus a per-step output that directly broadcasts a carry-shaped value; the
  latter aggregates primal and tangent output cotangents per carry lane.
  `tensor_scan_region(lower, upper, body, init, operands)` applies the same
  one-time region tracing contract to a `(next_carry, output)` body and returns
  the final carry plus a leading-axis stack of fixed-shape outputs.
  `tensor_scan(length, body, init)` similarly returns a final carry and a
  leading-axis stack of fixed-shape outputs, but remains the compatibility API
  that statically unrolls the body.
  `TensorTraceResult.symbolic_vjp(cotangent_name)` and
  `TraceTensor.symbolic_vjp(cotangent_name)` emit one transformable
  gradient trace per original input, all sharing a graph with the explicit
  output-cotangent input. This is the reverse graph needed for backend lowering;
  the established eager VJP evaluators remain available.
  `tensor_grad_scalar_fn(fn, input_specs)` traces and compiles a rank-N
  scalar-loss function once, then returns a reusable Python gradient callable
  backed by its frozen CPU plan.
- `tensor_value_and_grad_fn(fn, input_specs)` traces and compiles a rank-N
  scalar-loss function once, then returns its scalar value and named VJP
  gradients from one frozen-plan execution.
- `tensor_jit_cuda_fn(fn, input_specs, device_ordinal=0)` traces and compiles
  a fixed-shape rank-N function to a reusable callable CUDA plan on Linux.
- `tensor_value_and_grad_cuda_fn(fn, input_specs, parameter_names,
  device_ordinal=0)` traces a scalar loss and compiles its value plus named
  VJP gradients into one CUDA union plan. It inserts the scalar loss cotangent
  internally and returns host values only when its callable is invoked.
  `cuda_adam_loss_optimizer(loss, parameter_names, inputs, learning_rate, ...)`
  accepts a scalar `TraceTensor` or `TensorTraceResult`, retains static inputs,
  parameters, and Adam state on the GPU, and performs device-only `step()`
  calls. `loss()` is an explicit diagnostic host readback.
- `tensor_value_and_grad_mlx_fn(fn, input_specs, parameter_names)` builds one
  symbolic VJP graph and evaluates its scalar loss plus requested gradients
  from a shared MLX GPU value table. Results are materialized on the host when
  called. `mlx_adam_loss_optimizer(loss, parameter_names, inputs,
  learning_rate, ...)` retains parameters, selected static inputs, gradients,
  and bias-corrected Adam moments on the MLX GPU stream; `step()` performs no
  parameter or gradient host readback, while `loss()` and `parameters()` are
  explicit diagnostics. Inputs omitted from `retained_input_names` can be
  replaced through `step({"batch_input": tensor, ...})` without resetting the
  retained parameter or moment state.
- `tensor_vmap_fn(fn, input_specs, batch_size, in_axes=None, out_axis=0)`
  traces one fixed-size batched plan. `input_specs` describe one example;
  `in_axes` supports mapped axes (including normalized negative axes) and
  unmapped `None` inputs, while `out_axis` selects the returned batch layout.
  The trace tracks batch semantics through elementwise operations, matmul,
  reductions, transpose, reshape, broadcast, concat, slice, and `where`.
  `tensor_vmap_cuda_fn(...)` and `tensor_vmap_mlx_fn(...)` lower the same
  canonical trace for CUDA and MLX. Batch extent remains static.
- `tensor_jit_batch_fn(fn, input_names, in_axes=None, batch_axis=0,
  max_specializations=4)` lazily traces CPU plans for observed batch sizes.
  Non-batch dimensions and unmapped inputs must remain fixed; the bounded cache
  raises instead of silently retracing after its specialization limit.
- `tensor_value_and_grad_batch_fn(...)` applies the same bounded CPU batch
  specialization contract to scalar losses and returns frozen-plan VJP
  gradients, including aggregated gradients for unmapped parameters.
- `tensor_value_and_grad_batch_mlx_fn(...)` caches one MLX multi-output
  value-and-gradient plan per bounded batch specialization. Its returned loss
  and requested gradients are diagnostic host readbacks; use the MLX Adam API
  for device-resident parameter updates.
- `tensor_jit_batch_cuda_fn(...)` and `tensor_value_and_grad_batch_cuda_fn(...)`
  apply the same bounded-specialization contract to CUDA. Each observed batch
  size owns one CUDA plan; runtime execution requires a Linux CUDA driver and
  `libnvrtc.so`.
- `tensor_cond(predicate, on_true, on_false, operands)` traces a scalar
  Tensor predicate into CPU Tensor IR. Branches receive only explicit operands,
  which become parent-region capture bindings; the CPU evaluator materializes
  only the selected branch and supports direct first-order JVP/VJP. MLX and
  CUDA lower the same regions, including symbolic JVP/VJP and nested `Cond`:
  each reads the scalar predicate back to the host once per `Cond` evaluation
  (one device stream synchronization) and runs only the selected branch on the
  device, so an inactive branch such as `log(x)` for `x <= 0` cannot inject
  NaN into values or gradients. CUDA compiles both branch regions with the
  parent plan and binds captures as device buffers; it rejects `Cond` inside
  fused `Fori`/`Scan` device-loop bodies. Vmapped predicates or operands are
  rejected at trace time.
  `tensor_cond_fn(on_true, on_false, input_specs)` remains the host-boolean
  function-level boundary; `tensor_cond_value_and_grad_fn(...)` and
  `tensor_cond_jvp_fn(...)` apply the matching branch VJP or JVP.
- `tensor_vmap_jvp_fn(...)` and `tensor_vmap_vjp_fn(...)` apply the same
  batch-layout contract to runtime tangents and cotangents. Mapped input
  gradients are restored to their declared `in_axes`; unmapped input gradients
  aggregate over the mapped batch as required by reverse-mode AD.
  `tensor_vmap_vjp_cuda_fn(...)`, `tensor_vmap_vjp_mlx_fn(...)`,
  `tensor_vmap_jvp_cuda_fn(...)`, and `tensor_vmap_jvp_mlx_fn(...)` lower
  their primal plus AD outputs into one CUDA or MLX multi-output plan.
- `tensor_vmap_hvp_scalar_fn(fn, input_specs, batch_size, input_name, ...)`
  differentiates the sum of one scalar loss per mapped example through symbolic
  VJP then runtime-tangent JVP. `tensor_vmap_hvp_scalar_cuda_fn(...)` lowers
  that structural HVP plan to CUDA for the supported loop subset. The selected
  HVP input must be mapped; unmapped parameter HVPs and MLX lowering remain
  explicit future work.
- `Tensor.sum(axis=None, keepdims=False)` and `Tensor.mean(...)` accept a
  single integer axis or a sequence of normalized axes. Trace tensors expose
  the same contract; multi-axis reductions lower to existing axis-reduction
  and reshape nodes, preserving CPU, CUDA, MLX, JVP, and VJP behavior.
- `Tensor.norm(axis=None, keepdims=False)` and `TraceTensor.norm(...)` provide
  an L2 norm over the selected axes. It is composed as `sqrt(sum(x.powi(2)))`
  and therefore preserves the same normalized-axis, backend, and AD contract.
- `Tensor.max(axis=None, keepdims=False)` / `Tensor.min(...)` and traced
  equivalents reduce one or more axes. Their static-shape trace lowering uses
  existing slice and selection nodes; when extrema tie, the final coordinate
  in row-major reduction order receives the derivative. A native fused extrema
  reduction remains a compiler-performance follow-up.
- `Tensor.gather(indices, axis=0)` and `Tensor.scatter_add(indices, updates,
  axis=0)` support static integer index sequences, including negative and
  repeated indices. `scatter_add` is functional and accumulates repeated
  destinations; its traced lowering uses slice/concat/pad-slice so VJPs route
  repeated gather/scatter coordinates correctly. Dynamic tensor indices
  (pending `I32` index tensors), boolean-mask indexing, and assignment-style
  scatter are not yet supported.
- `quabla.einsum("ij,jk->ik", [lhs, rhs])` and
  `quabla.einsum("...ij,...jk->...ik", [lhs, rhs])` are scoped matrix-product
  spellings that lower directly to the existing rank-N `matmul` plan. Other
  einsum equations are rejected rather than silently interpreted.
- `Tensor` and `TraceTensor` support `__getitem__` with integer and
  contiguous unit-step slice tuples, including negative indices. Integer
  indices lower to a length-one slice plus reshape, so reverse-mode AD and
  CUDA/MLX lowering preserve the same semantics. Empty/strided slices,
  ellipsis, and advanced indexing are not supported.
- `tensor_hessian_scalar_fn(fn, input_specs, input_name)` and
  `tensor_hvp_scalar_fn(fn, input_specs, input_name)` freeze a scalar rank-N
  trace for dense Hessian or Hessian-vector-product evaluation. They are
  correctness-first CPU transforms; dense Hessian materialization is O(n^2).
  `tensor_jit_fn(fn, input_specs)` similarly returns a reusable rank-N primal
  callable backed by a frozen CPU plan.
  `tensor_vjp_fn(fn, input_specs)` returns a reusable rank-N VJP callable that
  accepts an output cotangent at invocation time.
  `tensor_jvp_fn(fn, input_specs)` returns a reusable rank-N JVP callable that
  accepts input tangents at invocation time.
  `TraceTensor` add/subtract/multiply/divide accept numeric scalar literals on either
  side, lowered as broadcastable rank-0 constant IR nodes.
- `TensorTraceGraph.compile_cpu(output_node_id)` freezes an immutable
  `TensorCpuExecutionPlan` containing only output-reachable rank-N IR nodes.
  The plan remaps operands after dead-code elimination, commons structurally
  identical pure nodes, and evaluates without accessing the mutable trace graph;
  it is a CPU execution boundary, not yet a machine-code JIT.
  `plan.lower_text()` emits its optimized deterministic rank-N lowering artifact.
  `plan.evaluate_vjp(...)` and
  `plan.evaluate_jvp(...)` execute cached-plan AD transforms without returning
  to the mutable tracer. `plan.kernel_ir()` exposes the same plan as structured
  op, operand, shape, and input-name records for later backend lowering.
  For a scalar replica-local loss,
  `plan.value_and_grad_data_parallel(inputs, mapped_input_names, shard_count,
  reduction="mean")` is the deterministic CPU reference for equal axis-zero
  sharding; it validates the shard contract and returns the aggregated value
  and gradients without claiming CPU parallel execution.
- The optional Linux `cuda-nccl` feature adds an experimental single-node
  data-parallel path. `tensor_value_and_grad_data_parallel_cuda_fn(...)`
  shards named axis-zero batch inputs across explicit CUDA ordinals and
  all-reduces only requested replicated parameter gradients. It reports
  enqueue, collective, and output-readback timing after a call. On two RTX
  3090s, single calls match the deterministic CPU reference exactly for `Sum`
  and `Mean`, and 200 SGD steps of a small MLP
  (`examples/validate_data_parallel_training_cuda.py`) stay within `2.1e-8`
  normwise parameter error of the single-GPU run. NCCL communicators are
  created once per compiled callable; the steady-state collective takes about
  0.1 ms per step there. Mapped-input gradients are rejected rather than
  all-reduced, and parameter updates run on the host.
- `TensorBackend` defines the rank-N plan execution contract. `CpuBackend` is
  the default implementation. On Linux, the optional `cuda` feature adds an
  NVRTC-compiled FP32 `CudaBackend` and immutable `TensorCudaExecutionPlan`.
  It lowers rank-two and broadcast-batched matmul, rank-N concat plus the internal slice/pad-slice reverse nodes, global and single-axis reductions, transpose,
  broadcasting, `where`, `div`, `log`, and the current differentiable unary
  primitives. Global sum/mean use a block-parallel reduction with atomic scalar
  accumulation. `compile_cuda()` retains its CUDA context and loaded module, so
  repeated `evaluate()` calls do not recompile NVRTC code. Pure elementwise
  graphs use one fused CUDA kernel; general plans additionally fuse maximal
  elementwise tails after materialized leaves such as matmul outputs and bias.
  `plan.fused_region_count` exposes the selected regions. `div` and `log`
  retain their checked-domain semantics through general per-operation lowering.
  It requires a CUDA driver plus `libnvrtc.so` at runtime. Host inputs and
  outputs still cross the device boundary on every `evaluate()`. Intermediate
  device buffers are reclaimed after their final consumer and reused by later
  nodes; retained inputs, the output buffer, and optimizer state keep stable
  allocations. `plan.device_buffer_count` reports the plan's current retained
  plus reusable device-buffer count for benchmark diagnostics.
  `TensorCudaExecutionPlan.evaluate_device(inputs, retained_input_names=...)`
  executes without materializing an output; supplied retained inputs upload on
  the first call and remain device-resident for subsequent static evaluations.
  Call `synchronize()` before timing asynchronous device-only evaluations, or
  use `benchmark_device(...)` to repeat and synchronize static evaluations in
  Rust without per-iteration Python dispatch. Rank-two FP32 GEMM automatically
  uses cuBLAS when `libcublas` is available; otherwise it uses the NVRTC tiled
  fallback. Inspect `plan.backend` to report the selected backend in a run log.
  A CUDA plan whose output is a gradient for one retained input can call
  `sgd_step(inputs, parameter_name, learning_rate, retained_input_names)` or
  `adam_step(inputs, parameter_name, learning_rate, retained_input_names,
  beta1=0.9, beta2=0.999, epsilon=1e-8)` to update that parameter on the
  device, then `retained_input(name)` to read it back after training. Adam
  keeps its first and second moments in device memory. Separate gradient plans
  can share retained parameters with
  `sync_retained_input_to(source_name, target_plan, target_name=None)`, which
  performs a same-device GPU-to-GPU copy after each participating update.
  `cuda_adam_step({parameter_name: gradient_plan, ...}, inputs, learning_rate,
  retained_input_names=...)` performs those Adam updates and cross-plan
  parameter synchronizations as one Python call while retaining every mapped
  parameter automatically. It still requires the caller to construct the
  symbolic gradient plans explicitly. For static collocation tensors,
  `cuda_adam_optimizer(...)` stores the Rust input tensors once and exposes
  `step()` plus `parameters()`. Calling `step({"batch_input": tensor, ...})`
  refreshes only the supplied mini-batch tensors without rebuilding plans or
  resetting GPU-resident parameter and Adam state.
  `cuda_adam_vjp_optimizer({parameter_name: symbolic_vjp_result, ...}, ...)`
  is the preferred multi-parameter path: it compiles all outputs from one
  symbolic VJP graph into one union plan, computes every gradient from the
  same parameter snapshot, and updates them without any gradient D2H copy.
  Fused GEMM+bias+activation kernels remain future work; multi-GPU execution
  is the separate `cuda-nccl` data-parallel path described above.
- `Tensor.split_key(key, count)`, `Tensor.random_normal(shape, key, ...)`, and
  `Tensor.glorot_normal(shape, key)` provide stateless, deterministic host-side
  initialization. They do not retain global RNG state; generated parameters are
  uploaded once when a CUDA optimizer is created.
- On Apple silicon, the `mlx` feature executes supported frozen
  Tensor IR plans as MLX arrays on `StreamOrDevice::gpu()`. Python exposes this
  through `TensorTraceGraph.compile_mlx()`, `TraceTensor.compile_mlx()`, and
  `TensorTraceResult.compile_mlx()`. The backend has CPU-parity coverage for
  elementwise operations, matmul, global reductions, reshape, transpose,
  concat, and broadcast, plus `Cond`, `Fori`, and `Scan` regions with their
  first-order VJP and forward-over-reverse (HVP) forms; loop carries and
  captures stay on the GPU stream between host-dispatched iterations.
  `tensor_value_and_grad_mlx_fn(...)` uses the same
  backend for a scalar loss and named symbolic VJP gradients. Contiguous
  `slice` and internal reverse `pad_slice` nodes execute on MLX through
  device-side take/pad lowering. `TensorMlxExecutionPlan.retain_inputs(...)`
  retains selected fixed bindings on the MLX GPU stream; later `evaluate(...)`
  and `evaluate_device(...)` calls may omit those bindings. Call
  `clear_retained_inputs()` to release that plan-owned state.
  `mlx_adam_loss_optimizer(...)` keeps fixed parameters and Adam moments on
  that GPU stream across `step()` calls; `loss()` and `parameters()` are the
  only host diagnostics. It is not a JIT,
  and unsupported reverse graphs return an explicit error rather than falling
  back to the host. Build requirements are listed under
  [Installation From Source](#installation-from-source).
- Python `Adam` updates immutable dictionaries of named rank-N `Tensor`
  parameters from VJP gradients. The test suite includes a manufactured 1D
  Poisson residual in which two symbolic coordinate JVP transforms form
  `u_xx`, then VJP and Adam recover one scalar MLP weight. This is a
  vertical-slice correctness proof. It includes batched collocation points and
  explicitly aggregated boundary/residual VJPs, but is not yet a general PINN
  framework.

### Legacy 2D API

These `Matrix`/`TraceGraph` entrypoints predate the rank-N path and are kept
for compatibility; they are deliberately 2D and outside the compiler facade.

- Python-facing `Matrix` class backed by Rust storage, elementwise add/subtract,
  elementwise multiply/divide, numeric scalar add/subtract/multiply/divide,
  elementwise tanh/exp/log/sqrt/sin/cos, greater-than masks, `where` select,
  concat, transpose, reshape, matmul, and global or axis-0/axis-1 sum/mean
  reductions.
- Python-facing `TraceGraph` / `TraceMatrix` seed that records add, subtract,
  multiply, divide, greater-than, `where`, concat, powf, tanh, exp, log, sqrt,
  sin, cos, transpose, reshape, matmul, and sum/mean nodes.
- Eager and traced add/subtract/multiply support numeric scalar literals. Traced
  scalar literals are recorded as shape-local scalar constant nodes. Unary
  negation and division by numeric scalars lower to scalar multiplication.
  Non-negative integer powers lower to repeated multiplication, while
  floating-point powers lower to a traced `powf` primitive with reverse-mode
  support for scalar exponents. Eager and traced elementwise add/subtract/multiply/divide
  support 2D NumPy-style broadcasting where each axis is equal or one side is
  `1`, including reverse-mode cotangent reduction back to the original input
  shapes. Sum/mean reductions accept `axis=0` or `axis=1` and keep reduced rank
  as `(1, cols)` or `(rows, 1)`, with reverse-mode broadcast back to the input
  shape. Concat accepts `axis=0` or `axis=1`, validates non-concatenated
  dimensions, and splits reverse-mode cotangents back to each input. Reshape
  preserves row-major storage order and requires element count to stay unchanged.
  General rank-N broadcasting is available on eager `Tensor` and the rank-N
  `TraceTensor` path; `Matrix` and the legacy `TraceGraph` transforms remain
  deliberately 2D.
- Python `trace(fn, input_specs)` function that runs user code with traced
  inputs.
- Structured `graph.ir()` output for AD and backend lowering experiments,
  including typed `attrs` for lowering-sensitive parameters such as scalar
  constants, `powf` exponents, reduction axes, and concat axes.
- Deterministic `graph.lower_text()` backend-text seed that renders the traced
  graph with SSA-like node ids, operand references, tensor shapes, and typed
  attributes. This is an inspection/lowering-contract prototype, not an MLIR
  or StableHLO-compatible serialization yet.
- Minimal `graph.vjp_ir(output_node_id)` reverse-mode seed for add, subtract,
  multiply, divide, `where`, concat, powf, tanh, exp, log, sqrt, sin, cos,
  transpose, reshape, matmul, and sum/mean outputs.
- CPU `graph.evaluate_vjp(output_node_id, inputs, output_cotangent)` evaluator
  with reverse cotangent accumulation over
  add/sub/mul/div/where/concat/powf/tanh/exp/log/sqrt/sin/cos/transpose/reshape/matmul/sum/mean
  trace graphs. Greater-than mask nodes are treated as non-differentiable
  control values.
- Python `grad(fn, input_specs, values, output_cotangent)` convenience API.
- Python `grad_fn(fn, input_specs, output_cotangent)` callable transform for
  reusing a cached trace across value dictionaries.
- Python `grad_scalar_fn(fn, input_specs)` callable transform that seeds a
  `(1, 1)` scalar-loss cotangent automatically over a cached trace.
- Python `grad_scalar(input_specs)` decorator factory for scalar-loss gradient
  functions that trace once when the decorator is applied.
- Python `value_and_grad_fn(fn, input_specs, output_cotangent)` callable
  transform that returns both the primal output and VJP gradients from one
  cached trace evaluation.
- Python `vjp_fn(fn, input_specs)` callable transform that accepts the output
  cotangent at call time and returns `(primal_output, gradients)`.
- Python `jacobian_fn(fn, input_specs)` callable transform for a single matrix
  input, returning a dense row-major Jacobian via repeated VJP over one cached
  trace.
- Python `jacobians_fn(fn, input_specs)` callable transform for one or more
  matrix inputs, returning a dictionary of dense row-major Jacobian matrices by
  input name via repeated VJP over one cached trace.
- CPU `graph.evaluate_jvp(output_node_id, inputs, input_tangents)` evaluator
  that propagates primal values and tangents in one forward pass.
- Python `jvp_fn(fn, input_specs)` callable transform for one or more matrix
  inputs, returning `(primal_output, tangent_output)` through direct
  forward-mode propagation instead of materializing dense Jacobians.
- `graph.compile_cpu(output_node_id)` freezes only the output-reachable traced
  subgraph into an immutable `CpuExecutionPlan`, eliminating unused nodes while
  remapping internal operands and commoning identical pure unary subexpressions.
  It supports CPU primal, JVP, VJP, and value-and-VJP evaluation; reusable AD
  transforms and `jit(...)` evaluate this frozen plan. It is an execution-plan
  boundary, not machine-code JIT compilation.
- Python `jit(input_specs)` decorator factory that traces a function once and
  evaluates a frozen CPU execution plan for later value dictionaries.

### Rust Core Crate

- Scalar forward-mode automatic differentiation with `Dual`.
- Experimental `forward_diff!(|x| expression)` procedural macro for a
  one-input pure scalar expression subset. It parses the closure AST at compile
  time and rewrites its input as a `Dual` seed, yielding a closure that returns
  primal and derivative values without a runtime tape. Mutation, arbitrary
  control flow, multi-input Jacobians, and captured methods without `Dual`
  arithmetic remain intentionally unsupported.
- `forward_gradient!(|x, y, ...| scalar_expression)` extends the same source
  subset to multiple active scalar inputs. It emits one forward-mode seeded
  expression per parameter and returns `ForwardGradient { value, gradient }`;
  its O(input dimension) cost is deliberate and suitable only as a small-model
  compiler reference, not as a reverse-mode replacement.
- Compile-time shaped `Tensor2<ROWS, COLS>` with shape-checked matmul.
- Fixed-step RK4 ODE integration.
- Final-state RK4 integration for loss functions that do not need full
  trajectories.
- Lotka-Volterra parameter-inference loss.
- Full four-parameter Lotka-Volterra gradient.
- Generic fixed-size gradient descent fitting loop.
- Central finite-difference gradient checks.

## Roadmap

[docs/jax_like_roadmap.md](docs/jax_like_roadmap.md) records the execution
order, each phase's status, and the evidence behind it. Open items listed
there include `I32` index tensors for dynamic gather/scatter (dtype phase D2),
the remaining dtype phases (`float16`/`bfloat16`, native device `f64`, typed
host storage, mixed precision), symbolic dimensions and vmappable `cond`
predicates, an attributable end-to-end CUDA fusion speedup and CUDA use of the
shared buffer plan, device-placement APIs and native CUDA/MLX factorization
kernels, moving retained device state into the facade `Executable`, and, for
distributed execution, mapped-input gradient concatenation, multi-node
transport, and tensor-parallel matmul.
