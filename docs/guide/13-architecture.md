# 13. Architecture

This chapter describes how Quabla is built: the crates, the tensor IR, the
autodiff and batching transforms, plan compilation, the three backends, and
the Python layer that ties them together. It is written for contributors and
for users who want to understand performance or error behavior at the
implementation level.

Code references name functions and types rather than line numbers. Most of
the compiler core lives in one large file,
`crates/quabla-core/src/tensor_ir.rs`; search it for the names given here.

## 13.1 Overview

```text
 Python                      python/quabla/*.py
   qb.grad / vmap / jit      _transforms.py: trace cache, pytrees, staging
   qb.cond / fori_loop ...   _control.py: region builders
   qb.optim, linalg, ode     pure-Python libraries over the primitives
        |
        v  PyO3 extension quabla._quabla (crates/quabla-python)
   Tensor (eager)            tensor.rs
   TraceTensor, TensorTraceGraph   tensor_trace.rs
        |
        v  crates/quabla-core
   TensorIr  --symbolic JVP/VJP, batching (vmap)-->  TensorIr
        |
        v  compile_cpu_many: DCE, canonicalize, fold, CSE
   TensorExecutionPlan (frozen, immutable)
        |
        +--> CPU interpreter (f64 reference)
        +--> CUDA: NVRTC-generated kernels, cuBLAS, cuSOLVER, NCCL
        +--> MLX:  mlx-rs operations on the GPU stream
```

The central design choice is that **every transform is a graph-to-graph
rewrite on one IR**, and every backend executes the same frozen plan. A
derivative is not a runtime tape: it is a new graph built from the same
primitive operations, which is why derivatives compose to any order, can be
batched, and lower to every backend.

## 13.2 Crates

| Crate | Role |
| --- | --- |
| `crates/quabla-core` | The tensor IR, CPU interpreter, runtime and symbolic autodiff, batching, plan compilation, and the CUDA and MLX backends. Pure Rust; no Python dependency. |
| `crates/quabla-python` | The PyO3 extension module `quabla._quabla`: eager `Tensor`, the tracer, the compiler facade, the legacy APIs. |
| `crates/quabla-macros` | Experimental `forward_diff!`/`forward_gradient!` source-to-source forward-mode macros over a small Rust expression subset. |
| `python/quabla` | The public Python package: the v0.2+ API written in Python on top of the extension. |

Cargo features: `quabla-core` has `cuda` (Linux only; `cudarc` with dynamic
loading), `cuda-nccl` (adds NCCL), and `mlx` (macOS only; `mlx-rs`). The
Python crate forwards `cuda`, `cuda-nccl`, and `mlx`. Without a feature,
the backend types exist as stubs that return "backend unavailable" errors,
so the rest of the code compiles unchanged.

### Modules of `quabla-core`

| Module | Responsibility |
| --- | --- |
| `tensor_ir.rs` | The IR, the CPU interpreter, runtime and symbolic AD, batching, plan compilation, CUDA source generation helpers |
| `tensor_ir/elementwise.rs` | `UnaryMathKind` (exp, log, sin, tanh, erf, tan, ..., round) and `BinaryMathKind` (pow, atan2, fmod): every rule of an elementwise math function, from its value and derivatives to its CUDA spelling and constant folding |
| `tensor_ir/extremum.rs` | `max`/`min` along axes (IEEE 754-2019 semantics) and their derivative rules |
| `tensor_ir/linalg.rs` | Host kernels for the decompositions (`LinalgKind`): LU sign/log-determinant, Jacobi `eigh`, Householder QR, Jacobi SVD |
| `tensor_ir/cholesky.rs` | Cholesky derivative kinds and the jet-based higher-order expansion |
| `tensor_ir/custom.rs` | `TensorCustomRule`: the payload behind `custom_vjp`, `custom_jvp`, and `checkpoint` |
| `tensor_ir/region_batching.rs` | `vmap` rules for region nodes (`Cond`, `Fori`, `Scan`, `While`, and their derivative variants) |
| `tensor_ir/host_storage.rs` | `HostTensorStorage { F64, F32, Bool }`: immutable `Arc` storage sized to the dtype |
| `tensor_ir/device_optimizer.rs` | Optimizer hyperparameters shared by the CUDA and MLX trainers |
| `tensor_ir/cuda*.rs` | CUDA backend: codegen and execution, host-driven loops and graph replay, decompositions, optimizers, the `CudaReal` element type |
| `tensor_ir/mlx*.rs` | MLX backend: lowering, CPU-stream linear algebra, Metal Cholesky and scatter kernels |
| `compiler.rs` | The stable facade: `QuablaCompiler`, `QuablaProgram`, `QuablaExecutable`, multi-output variants |
| `autodiff.rs`, `tensor.rs`, `ode.rs`, `models.rs`, `optim.rs` | The original scalar `Dual`, `Tensor2<R, C>`, RK4, and Lotka-Volterra demonstrations |

### Modules of `quabla-python`

| Module | Responsibility |
| --- | --- |
| `tensor.rs` | Eager `Tensor` and `TensorView`; methods dispatch to the tracer when an operand is traced |
| `tensor_trace.rs` | `TensorTraceGraph`, `TraceTensor`, the bridge methods used by `_transforms.py`, region builders, staged executables, device trainer executors, and the v0.1 `tensor_*_fn` helpers |
| `compiler.rs` | The Python `Compiler`/`Program`/`Executable` facade |
| `interop.rs` | Construction from sequences and the buffer protocol, `tolist`, read-only buffer export (never links NumPy) |
| `repr.rs` | NumPy-style printing |
| `trace.rs`, `matrix.rs`, `optim.rs` | The legacy 2D `TraceGraph`/`Matrix` API and the v0.1 host `Adam` |
| `dtype.rs`, `errors.rs` | dtype objects; raising the Python exception classes from Rust |

## 13.3 The Tensor IR

A program is a `TensorIr`: an immutable, `Arc`-shared vector of
`TensorNode`s plus a map from input names to node ids. Node ids
(`TensorNodeId`) are indices into that vector, and nodes only refer to
earlier nodes, so the graph is in topological order by construction.

Each `TensorNode` has an op, a static shape, a dtype (`TensorDType::{F32,
F64, Bool}`), and a `weak` flag. The flag implements weak typing: a weak
value (from a Python scalar) that meets a strong one receives an explicit
`Cast`, which is how `0.5 * x32` stays `float32`.

The private `enum TensorOp` holds every operation. Its variants fall into
groups:

| Group | Variants |
| --- | --- |
| Leaves | `Input`, `ScalarConstant`, `Constant` |
| Elementwise | `Cast`, `Add`, `Sub`, `Mul`, `Div`, `Greater`, `Compare`, `Where`, `Sqrt`, `SqrtDerivative`, `Powi`, `UnaryMath { kind }`, `BinaryMath { kind }`, `StopGradient` |
| Reductions | `Sum`, `SumAxis`, `Mean`, `MeanAxis`, `ExtremumAxis`, `CumSum` |
| Shape and indexing | `Reshape`, `Transpose`, `Concat`, `Slice`, `PadSlice`, `Gather`, `ScatterAdd`, `Broadcast` |
| Linear algebra | `Matmul`, `Solve`, `Cholesky`, `CholeskyAd`, `Triangular`, `Linalg { kind }` |
| Regions | `Region(RegionNode)`, whose `RegionKind` is `Cond`, `Fori`, `ForiJvp`, `ForiVjp`, `ForiVjpJvp`, `While`, `Scan`, `ScanVjp`, or `ScanVjpJvp` |
| Custom rules | `Custom` |

Many user-facing functions are *compositions*, not ops: `maximum` is a
`Where` over a comparison, `softmax` and `norm` are built from reductions and
elementwise ops, the v0.4 jax.numpy functions are reshapes, slices, gathers,
and `concat`. Keeping the op set small keeps every transform and backend
small.

### Regions

A region node (`TensorOp::Region`, defined in `tensor_ir/region.rs`) owns
frozen sub-plans and a list of *captures* that bind parent nodes to the
region's input names. Its `RegionKind` holds the kind's operand slots (the
predicate, or the carry and its tangents and cotangents), its plan, and,
for multi-result kinds, the selected result and group; `RegionNode` answers
the kind-independent questions (operands, group, name, frozen plans) once,
while evaluation, differentiation, and batching match on the kind. For
example `TensorForiExecutionPlan`
holds the bounds, the body plan, and the names of the carry and index
inputs; `TensorCondExecutionPlan` holds two branch plans;
`TensorWhileExecutionPlan` holds a predicate and a body region;
`TensorScanExecutionPlan`'s body returns `(carry, output)` and outputs are
stacked on a leading axis.

Because a region's inputs are exactly its captures, the Python layer insists
that traced values enter a region through `operands=` (the "operand rule" of
[Chapter 4](04-control-flow.md#41-the-operand-rule)).

Several nodes can share one execution: the reverse pass of a loop produces
one `ForiVjp`/`ScanVjp` node per gradient output, tagged with a common
`group` id and a target that selects the output, so the backward loop runs
once.

### Custom rules

`Custom { value, operands, rule, output, group }` is the identity on `value`
for evaluation. Its `TensorCustomRule` carries the forward graph (outputs and
residuals), the backward graph, an optional tangent graph, and a
`rematerialize` flag for `checkpoint`. The symbolic transforms splice these
graphs in place of differentiating `value`; plan compilation aliases the
node to `value`. Region bodies are the exception: they are compiled with
`compile_region`, which keeps their `Custom` nodes, because a region's
derivatives are built from its frozen body later, so the loop JVP, VJP,
and HVP of a body that calls a function with a custom rule are those of
the rule. Every backend evaluates a `Custom` node in a region body as the
identity on its value, as it does `stop_gradient`.

## 13.4 Autodiff

There are two families of differentiation in the core.

**Symbolic transforms** produce new IR and are what the core API uses:

- `TensorIr::symbolic_jvp` (and the `_with_tangent_inputs` and `_many`
  variants; the per-op rules are in `symbolic_jvp_many_with_seed`) emit, for
  each node, a primal and a tangent node.
- `TensorIr::symbolic_vjp` / `symbolic_vjp_many` (core in
  `symbolic_vjp_many_impl`) emit the reverse graph, accumulating cotangents
  and reducing them back through broadcast axes.
- Region nodes have their own rules (`symbolic_jvp_fori`,
  `symbolic_vjp_scan`, ...) that build new region nodes: the JVP of a `Fori`
  is a `ForiJvp` over a packed `(primal, tangent)` carry; the VJP is a group
  of `ForiVjp` nodes; differentiating a `ForiVjp` forward gives
  `ForiVjpJvp`, which is how forward-over-reverse Hessians work through loops.
- Hessian-vector products are forward-over-reverse: `scalar_hvp_graph` runs
  `symbolic_vjp` with a fresh cotangent input, then a symbolic JVP of the
  resulting gradient.

Since derivative graphs are ordinary IR, applying a transform again needs no
special support; `grad(grad(f))` is two symbolic VJPs, `hessian` is a
symbolic JVP of a VJP vectorized with `vmap`.

**Runtime (direct) evaluation** computes derivatives numerically without
building a graph: `TensorIr::vjp`, `value_and_vjp_many`, `jvp`, `jvp_many`,
and a mixed second-order dual evaluator (`evaluate_mixed`). The v0.1 helpers
and the reference paths of `hessian_scalar`/`hvp_scalar` use them, and tests
use them as an independent check of the symbolic rules.

### Loop derivatives and checkpointing

The reverse pass of a loop needs every carry. `TensorCarryCheckpoints` stores
one carry per block of about `sqrt(T)` iterations during the forward pass,
then re-runs each block forward and consumes it in reverse. Short loops keep
the full tape. Reverse mode through `While` is rejected because its trip
count is not fixed. Reverse-over-reverse through loop regions is rejected;
forward-over-reverse (`ForiVjpJvp`, `ScanVjpJvp`) is supported.

## 13.5 Batching (`vmap`)

`TensorIr::inline_batched` copies a graph into another while adding a batch
axis to every node that depends on a mapped input (`push_batched` holds the
per-op rules). Nodes that do not depend on mapped inputs are copied
unchanged, which is why unmapped parameters are not replicated.

Regions are batched by `region_batching.rs`:

1. decide which region inputs are mapped;
2. batch the body plan;
3. for loops, iterate to a fixed point on which carries become mapped (a
   carry that depends on a mapped capture is mapped from the first
   iteration);
4. rebuild the node with the ordinary constructors.

A `Cond` with a mapped predicate is rewritten to evaluate both branches and
select with `Where`. A `While` with a mapped predicate becomes "run while any
example continues", with the body masked per example. Reverse-mode loop
nodes map every input and receive one new group id. Custom rules batch
through `TensorCustomRule::batched`.

## 13.6 Plan Compilation

`TensorIr::compile_cpu_many` turns a traced graph into an immutable
`TensorExecutionPlan`, in one pass:

1. **Dead-code elimination** by reachability from the requested outputs.
2. **Canonicalization** (`canonicalize_tensor_op`): drop same-dtype strong
   casts, collapse reshape chains, remove identity broadcasts, and alias
   `Custom` nodes to their value.
3. **Constant folding** of scalar and array constants, rounded to the node
   dtype.
4. **Constant deduplication** by bit pattern.
5. **Common-subexpression elimination** keyed on the pure op, operands,
   dtype, and weak flag.
6. **Final pruning**, and detection of fully elementwise plans (used by CUDA
   fusion).

The output of `jit(...).lower(...).as_text()` is this frozen plan.

`compiler.rs` wraps the lifecycle in stable types: `QuablaCompiler::compile`
and `compile_many_checked_with_precision` freeze the program, check that the
target is built, run the target's validation (`validate_cuda`,
`validate_mlx`, `validate_cuda_float64`), and lower to a
`QuablaExecutable::{Cpu, Cuda, CudaFloat64, Mlx}`. Validation is where
unsupported operations are rejected with an explicit error, before any
execution.

The `TensorBackend` trait (`name`, `execute(plan, inputs)`) is implemented
by `CpuBackend`, `CudaBackend`, and `MlxBackend`.

## 13.7 Backends

### CPU

`evaluate_tensor_nodes_with_outputs` interprets the plan node by node.
Values are computed in `f64` and stored through `TensorDType::round`, so a
`float32` program is the `float64` result rounded after each operation:
bit-exact IEEE `float32` for `+ - * / sqrt`, and the reference that the GPU
backends are tested against. The interpreter reuses a buffer in place at
its last use and caches grouped loop derivatives.

### CUDA

`crates/quabla-core/src/tensor_ir/cuda.rs` (via `cudarc`, with libraries
loaded dynamically):

- Generates plan-specialized CUDA C, compiles it with NVRTC once per plan,
  and keeps the module loaded.
- Fuses elementwise regions into single kernels.
- Uses cuBLAS for GEMM and cuSOLVER for solves and decompositions.
- Lowers purely elementwise loop bodies to fused loop kernels; every other
  loop body runs in `cuda_host_loop.rs` as a host-driven loop on
  device-resident carries. After `CUDA_GRAPH_EAGER_RUNS` (two) eager
  iterations, a capturable body is recorded as a CUDA graph and replayed.
- Device optimizers (fused SGD/Adam/AdamW steps and the global-norm clip)
  live in `cuda_optimizer.rs`.
- `CudaReal` selects `f32` (default) or `f64` (`precision="float64"`).
- The `cuda-nccl` feature adds `CudaDataParallelExecutionPlan`.

### MLX

`crates/quabla-core/src/tensor_ir/mlx.rs` (via `mlx-rs`):

- Lowers plans to MLX operations on the GPU stream; intermediates stay on
  the device.
- `mlx_linalg.rs` builds LU, `eigh`, QR, and SVD on MLX's CPU stream, since
  MLX has GPU kernels for none of them; custom Metal kernels cover Cholesky
  and a deterministic `scatter_add`.
- A process-wide lock (`MLX_EXECUTION_LOCK`) serializes all MLX graph
  construction, evaluation, and readback. It is taken once at each public
  entry point and never by internal helpers (the mutex is not reentrant).
  Retained parameters and optimizer state are shared across calls that may
  run on different threads, which is why evaluation stays serialized.

## 13.8 The Python Layer

`python/quabla/_transforms.py` implements the core API's transforms. Each
transform (`_ValueAndGrad`, `_Jvp`, `_Vjp`, `_Vmap`, `_Jacobian`, `_Jit`) is a
`_Transform` subclass.

1. **Signature.** `_signature` computes the cache key: pytree structure
   (including node classes and aux data), shape and dtype per array leaf,
   and static values (Python scalars, static arguments).
2. **Cache.** `_TraceCache` is an LRU of at most `max_traces` entries
   (`_DEFAULT_MAX_TRACES = 8`) that warns with `RetraceWarning` on eviction.
   Caches are keyed by the root function and the chain of transforms
   applied to it.
3. **Trace.** `_trace` creates a `TensorTraceGraph`, adds one named input per
   array leaf (the name is the pytree path, which is what `as_text()`
   shows), calls the Python function inside `_begin_trace`/`_end_trace`,
   and lifts the outputs.
4. **Transform.** The bridge methods on `TensorTraceGraph`
   (`_symbolic_vjp`, `_symbolic_jvp`, `_inline_batched`) call the core
   transforms.
5. **Compile and run.** `_compile` calls
   `compile_many_checked_with_precision` and returns a `StagedExecutable`
   for the selected target.

**Nesting.** A transformed function called on tracers does not compile;
it *stages* its graph and inlines it into the enclosing trace (`_inline`).
That is how `jit(value_and_grad(loss))` with an inner `vmap(grad(grad(u)))`
becomes one graph.

**Closures over tracers.** Tracers of an enclosing trace that a function
closes over are lifted as extra hidden inputs. A probe re-trace discovers
which tracers this call captures, and the cache keys on their shapes and
dtypes.

`_control.py` maps `cond`, `fori_loop`, `scan`, and `while_loop` onto the
extension's region builders (`tensor_cond`, `tensor_fori_loop_region`,
`tensor_scan_region`, `tensor_while_loop_region`, or the unrolled
`tensor_fori_loop`/`tensor_scan`), and runs a Python loop for eager inputs.

`_custom.py` traces `fun` and the rule functions and records them with
`TensorTraceGraph._custom_rule` and `_custom_call`.

The libraries (`optim.py`, `linalg.py`, `ode.py`, `_krylov.py`,
`_newton.py`, `_lbfgs.py`) are ordinary Python over these primitives. For
example, `odeint` is a `fori_loop`; `cg` is a `custom_vjp` whose forward pass
is a `while_loop` and whose backward pass is another solve. `optim.Trainer`
on a device traces the loss once and hands it to the native executor
`_cuda_trainer_optimizer` or `_mlx_trainer_optimizer`, which compile the
loss, its gradient, and the optimizer update into retained device state.

## 13.9 Invariants Worth Knowing

- **Immutability.** Host storage is immutable and shared through `Arc`;
  views cannot alias mutable state. IR graphs and plans are immutable once
  built.
- **No silent fallback.** A backend either lowers an op or rejects it during
  validation. There is no host fallback path.
- **CPU is the reference.** Every device lowering is tested against the CPU
  result of the same plan.
- **Explicit precision.** One compiled program has one floating type per
  device execution; `float32`/`float64` mixing is rejected at the API.
- **Safety comments.** Every `unsafe` block carries a `// SAFETY:` comment
  naming its invariant (`clippy::undocumented_unsafe_blocks` is enabled).
