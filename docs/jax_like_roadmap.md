# Rust SciML Runtime Roadmap

JAX can be summarized as:

```text
JAX = NumPy-like arrays + composable autodiff + XLA compilation
```

This project should not compete by rebuilding that stack one feature at a time.
The Rust-specific bet is to move more correctness and transformation work into
types, macros, compiler passes, and explicit backend contracts.

## Layer 1: Autodiff Transformation

Current state:

- Scalar forward-mode AD with `Dual`.
- Model gradients are computed by seeding one parameter at a time.
- `nabla-core::tensor_ir::TensorIr` now provides a pure-Rust dynamic rank-N IR
  for input, broadcasted add/multiply, and scalar sum; it evaluates on CPU and
  generates both direct JVPs and VJPs that reduce cotangents over broadcast
  axes.

Target direction:

- Represent differentiable programs in a transformable IR instead of only
  runtime operator overloading.
- Support Jacobians, Hessians, and mixed higher-order derivatives needed by
  PINNs and inverse problems.
- Explore source-to-source AD through procedural macros before attempting a
  compiler plugin.
- Keep an LLVM/Enzyme-style path as a possible backend for low-level kernels.

Key risk:

- Loops, branches, mutation, and borrowing make source transformation hard.
  Rust's borrow checker is an advantage only if the AD transform preserves
  aliasing and lifetime semantics instead of bypassing them.

Near-term milestone:

- Extend `TensorIr` with reductions, reshape, matmul, and unary primitives.
- Generate forward-mode transforms from the same IR.
- Compare generated derivatives against finite differences.

## Layer 2: Typed Arrays And Layout

Current state:

- `Tensor2<ROWS, COLS>` stores row-major `f64` data.
- Matrix multiplication enforces the inner dimension at compile time:
  `(M, K) x (K, N) -> (M, N)`.
- Python-facing `Tensor` provides eager rank-N contiguous row-major storage,
  `zeros`/`ones`/`full`/`arange`/`linspace`/`eye` creation, rank-N concat,
  positive runtime shape validation, trailing-axis broadcasting for
  add/subtract/multiply/divide and scalar powers, NumPy-style batched matmul,
  permutation-validated transpose, global or single-axis sum/mean, and
  common elementwise math, `gt(...)` masks, broadcasted `where`, and
  materialized `broadcast_to`, and element-count-preserving reshape.
  `Tensor.slice(...)` creates a zero-copy,
  read-only `TensorView` with explicit stride and offset metadata; the shared
  backing allocation is immutable, so no mutable aliasing is exposed.

Target direction:

- Use const generics for static shapes where possible.
- Provide dynamic-shape tensors for data-dependent workloads.
- Make layout explicit: contiguous, strided, transposed, sliced, and device
  views should be different states with clear aliasing rules.
- Avoid promising zero-cost slicing until the stride/view representation is
  proven in tests and benchmarks.

Key risk:

- Static shapes alone are not enough. Real SciML workloads need batching,
  adaptive grids, irregular observation sets, and dynamic shapes.

Near-term milestone:

- Carry rank-N tensor operations and stride-aware views into a generic trace
  representation without weakening their shape and aliasing invariants.
- Add compile-time checked `matmul` for static tensors.
- Extend dynamic tensor operations with reductions and explicit stride-aware
  views before carrying them into the trace IR.

## Layer 3: Backend, Compilation, And Sharding

Current state:

- CPU loops remain the default execution path, with frozen execution plans but
  no general machine-code JIT.
- Linux has an NVRTC CUDA path for rank-N plans, including device-resident
  symbolic VJP/JVP evaluation and parameter updates.
- Apple silicon has an experimental MLX primal-plan backend. It executes
  supported frozen Tensor IR nodes on `StreamOrDevice::gpu()` and deliberately
  reports unsupported reverse-only nodes rather than falling back to CPU.
- There is no distributed runtime.

Target direction:

- Separate frontend tensor expressions from backend execution.
- Start with a CPU backend contract.
- Add a lowering path to a kernel IR before choosing LLVM, MLIR, CUDA, WGSL, or
  another backend.
- Model sharding as a typed placement/layout property, not as an afterthought.
- Use `Send` and `Sync` boundaries to make cross-device execution explicit.

Key risk:

- Distributed sharding is a systems project. It should not be started before the
  tensor IR and backend contract are stable.

Near-term milestone:

- Define backend traits for allocation, copy, elementwise map, and matmul.
- Add a CPU backend implementation.
- Add an execution trace that can later be lowered or partitioned.

## Product Direction

The scientific learning workflow should be:

```text
Python API -> Rust Matrix/Tensor objects -> trace/IR -> AD transform
           -> optimizer or backend lowering -> CPU/GPU/sharded execution
```

ODE/PDE solvers are one kind of differentiable model, not the whole product.

### Compiler Facade

The public compiler lifecycle is now explicit instead of being distributed
across tracing and backend-specific execution-plan classes:

```text
frontend trace -> Program -> JVP/VJP transform -> compile(target) -> Executable
```

`nabla_core::compiler` owns the Rust-facing `NablaCompiler`, `NablaProgram`,
and `NablaExecutable` contracts. The Python bridge exposes the same lifecycle
as `nabla.Compiler`, `Program`, and `Executable`. Existing function-specific
helpers remain compatibility APIs while they migrate internally.

This facade is a boundary, not a claim that every IR operation lowers on every
target. It distinguishes build availability from per-program support: CUDA and
MLX can reject unsupported graphs explicitly, and incomplete structured GPU
HVP lowering must not be represented as a general supported capability.

Implementation status (2026-07-29): the core facade now owns immutable
single-output programs and their `freeze`, symbolic tangent-input JVP, symbolic
VJP, target selection, and execution contracts. `NablaTarget::is_built()` is a
compile-time feature/platform check; it is deliberately separate from
per-program lowering validation. The Python facade traces rank-N functions,
exposes `Program.jvp(input_name)` and `Program.vjp(cotangent_name)`, then
returns a uniform `Executable` for CPU, CUDA, or MLX. It does not yet replace
the legacy 2D `TraceGraph` API or make the existing function-specific helpers
delegate internally.

Verification boundary (2026-07-29): the core lifecycle test checks CPU
execution and generated JVP/VJP program structure. The installed PyO3
extension test checks Python `Compiler.trace`, `Program.compile`, execution,
and re-compilation of a VJP result. On `cuda-host` (GTX 1660 SUPER), the same
facade test compiles a nonlinear scalar loss, its symbolic coordinate JVP, and
one symbolic VJP program per input through CUDA, then compares all results to
CPU at `1e-5` tolerance. The same GTX 1660 SUPER run validates a nonlinear
fixed-bound Scan HVP through symbolic VJP then JVP: CUDA's paired carry result
matches CPU at `2e-5` tolerance. A second parity case verifies an output reshape
whose rank differs from the carry while preserving element count. The full
CUDA Python matrix executes both acceptance paths and an indexed unequal-lane
rejection path on that GPU; no Scan test is left outside the matrix runner.
Workspace
tests and all-feature Clippy pass on the local Apple-silicon host. MLX facade
execution still requires a build with the `mlx` feature and target-host
validation for each supported operation set.

Implementation status (2026-07-29, continued): CUDA primal Scan separates
carry-lane and per-step-output-lane counts. For broadcast-compatible bodies,
each output lane replays the pure elementwise carry recurrence from its mapped
initial carry value, so a `[2, 1]` carry can emit `[2, 3]` outputs without CPU
fallback. First-order `ScanVjp` now supports the verifiable unequal-count
subset where the per-step output directly broadcasts a carry-shaped value: the
kernel aggregates all mapped output cotangents for a carry lane before running
the body VJP. GTX 1660 SUPER parity covers initial-carry and explicit-capture
gradients for that path. `ScanVjpJvp` now implements the same direct-broadcast
subset, aggregating both primal and tangent output cotangents before its
forward-over-reverse recurrence. It also accepts broadcast-compatible explicit
captures and atomically reduces their directional gradients. GTX 1660 SUPER
parity covers nonlinear HVPs for both same-shaped and broadcast captures.
General unequal-count output graphs remain explicit rejections, rather
than silently treating one output lane as one carry lane.

## Python Bridge

Current state:

- `nabla-python` exposes a minimal `Matrix` class through PyO3.
- Python can use Rust-owned matrix storage and call Rust add/sub/mul/div/powf/tanh/exp/log/sqrt/sin/cos/gt/where/concat/transpose/reshape/matmul/sum/mean.
- `TraceGraph` and `TraceMatrix` can record input, scalar constant, add,
  subtract, multiply, divide, greater-than, `where`, concat, powf, tanh, exp,
  log, sqrt, sin, cos, transpose, reshape, and matmul nodes, plus sum/mean reductions.
  Numeric scalar literals are supported for eager and traced add/subtract/multiply. Unary negation and
  division by numeric scalars lower to scalar multiplication; non-negative
  integer powers lower to repeated multiplication, while floating-point powers
  lower to a traced `powf` primitive with reverse-mode support for scalar
  exponents. Eager and traced elementwise add/subtract/multiply/divide support
  2D NumPy-style broadcasting where each axis is equal or one side is `1`,
  including reverse-mode cotangent reduction back to the original input shapes.
  Sum/mean reductions accept `axis=0` or `axis=1` and keep reduced rank as
  `(1, cols)` or `(rows, 1)`, with reverse-mode broadcast back to the input
  shape. Concat accepts `axis=0` or `axis=1`, validates non-concatenated
  dimensions, and splits reverse-mode cotangents back to each input. Reshape
  preserves row-major storage order and requires element count to stay unchanged.
  Eager `Tensor` supports general rank-N trailing-axis broadcasting and
  NumPy-style batched matmul, but it is intentionally not part of the legacy
  `TraceGraph`, AD transforms, or CPU plan lowering; those paths remain 2D
  `Matrix`-based. The separate `TensorTraceGraph` provides a rank-N
  add/multiply/sum AD subset.
- `nabla-core::TensorIr` is the rank-N compiler-core path, but it is not yet
  exposed through the legacy Python tracer. `trace_tensor(...)` now exposes its
  add/subtract/multiply/divide/matmul/rank-N-concat/rank-N-transpose/tanh/exp/sin/cos/sqrt/non-negative-integer-powi/log/reshape/global-or-single-axis-sum/mean subset through separate `TensorTraceGraph` and
  `TraceTensor` classes, with Python-facing CPU VJP and JVP evaluation. This
  preserves the established 2D Python transform API while migration proceeds
  operation by operation.
- `TraceTensor` add/subtract/multiply/divide accept numeric scalar literals on either
  side; these lower to rank-0 constant nodes with zero JVP tangent and no VJP
  input gradient.
- `TraceTensor.reshape(shape)` preserves row-major element order, validates the
  element count at trace time, and reshapes primal values and every AD tangent
  or cotangent consistently.
- `TraceTensor.slice(axis, start, stop)` selects one contiguous rank-N range.
  Negative axes are normalized at trace time; VJP reconstructs the original
  shape with a zero-padded internal node, so CPU and CUDA reverse graphs remain
  differentiable without host fallback.
- `TraceTensor.broadcast_to(shape)` records shape expansion instead of
  materializing it in Python. Its reverse transform reduces broadcast axes to
  the original shape; the CUDA backend lowers it to a rank-N indexed copy.
- `TraceTensor.mean()` performs a global mean reduction to a rank-0 scalar;
  its VJP broadcasts the scalar cotangent back over the input with `1/N`
  scaling.
- `TraceTensor.sum(axis=...)` and `TraceTensor.mean(axis=...)` reduce one
  normalized axis (including negative Python-style indices), remove that axis
  from the result shape, and expand cotangents back over the reduced axis.
- `TraceTensor.transpose(axes=None)` permutes every rank-N axis; omitting
  `axes` reverses the axis order, while a provided list must be a complete
  permutation. VJP uses the inverse permutation to restore input layout.
- `TraceTensor.log()` requires strictly positive runtime tensor values and
  returns a domain error instead of producing NaN for non-positive inputs.
- `TensorTraceGraph.hessian_scalar(output, input_name, inputs)` computes an
  exact dense Hessian for a scalar output and one named input using a mixed
  second-direction forward transform. Its O(n^2) basis evaluation is a
  correctness-oriented implementation, not yet an optimized HVP API.
- `TensorTraceGraph.hvp_scalar(...)` reuses the same exact mixed transform to
  compute a Hessian-vector product in O(n) basis evaluations without allocating
  the dense Hessian.
- `TensorTraceResult.symbolic_jvp(input_name)` is the composable coordinate
  derivative path: it constructs a new IR with primal/tangent pairs rather than
  returning a runtime tensor. Reapplying it constructs higher coordinate
  derivatives while leaving parameter inputs available to downstream VJP.
- `tensor_grad_scalar_fn(fn, input_specs)` traces and compiles a rank-N scalar
  loss once, returning a reusable Python gradient callable backed by the frozen
  plan rather than retracing on every invocation.
- `tensor_value_and_grad_fn(fn, input_specs)` traces and compiles a rank-N
  scalar loss once, returning its scalar value and named VJP gradients from one
  frozen-plan execution.
- `tensor_jit_fn(fn, input_specs)` traces and compiles a rank-N function once,
  returning a reusable primal callable backed by that same frozen-plan boundary.
- `tensor_jit_cuda_fn(fn, input_specs, device_ordinal=0)` traces and compiles
  a fixed-shape rank-N function to a reusable callable CUDA plan on Linux.
- `tensor_value_and_grad_cuda_fn(fn, input_specs, parameter_names,
  device_ordinal=0)` traces a scalar function once and compiles its primal
  value plus requested named gradients into one CUDA union plan. It owns its
  scalar cotangent input rather than requiring it from Python.
- `cuda_adam_loss_optimizer(loss, parameter_names, inputs, learning_rate, ...)`
  accepts a scalar `TraceTensor` or `TensorTraceResult`, creates the same union
  plan, and retains static inputs, parameters, and Adam state on device.
  `step()` has no parameter or gradient host readback; `loss()` is an explicit
  diagnostic readback.
- `tensor_vmap_fn(fn, input_specs, batch_size, in_axes=None, out_axis=0)`
  traces one fixed-size batched plan from per-example input shapes. It supports
  mapped normalized positive/negative axes, unmapped (`None`) inputs, and a
  requested output batch axis; `tensor_vmap_cuda_fn(...)` and
  `tensor_vmap_mlx_fn(...)` lower that same canonical trace. Batch extent
  remains static.
- `tensor_hessian_scalar_fn(fn, input_specs, input_name)` and
  `tensor_hvp_scalar_fn(fn, input_specs, input_name)` freeze a scalar rank-N
  trace for dense Hessian or Hessian-vector-product evaluation. Dense Hessian
  materialization remains a correctness-first O(n^2) CPU path.
- `tensor_vmap_hvp_scalar_fn(fn, input_specs, batch_size, input_name, ...)`
  and `tensor_vmap_hvp_scalar_cuda_fn(...)` trace one scalar loss per mapped
  example, differentiate their sum through symbolic VJP then runtime-tangent
  JVP, and return the selected mapped input's HVP in its declared public axis
  layout. The selected input must be mapped; unmapped parameter HVPs and MLX
  lowering are not part of this interface yet.
- `tensor_vjp_fn(fn, input_specs)` traces and compiles once, returning a
  reusable rank-N VJP callable whose output cotangent is supplied at invocation
  time.
- `tensor_jvp_fn(fn, input_specs)` traces and compiles once, returning a
  reusable rank-N JVP callable whose input tangents are supplied at invocation
  time.
- `tensor_jacobian_fn(fn, input_specs, input_name)` traces and compiles once,
  then constructs a dense Jacobian for one named input using direct JVP basis
  directions. It is correctness-oriented rather than a large-scale Jacobian
  materialization strategy.
- `TensorTraceGraph.compile_cpu(output_node_id)` produces an immutable
  `TensorCpuExecutionPlan` after output-reachability DCE, operand-id remap, and
  exact structural CSE for pure nodes. It provides the rank-N CPU execution
  boundary for later native code lowering; it does not yet generate machine
  code. Its `lower_text()` is the deterministic optimized artifact to map onto
  a later kernel IR. The same immutable plan
  now exposes CPU primal, VJP, and JVP evaluation. `kernel_ir()` exposes
  structured op, operand, shape, and input-name records without parsing text.
- `TensorBackend` is the rank-N execution contract and `CpuBackend` is its
  default implementation. The Linux-only `cuda` feature adds an
  NVRTC-compiled FP32 `CudaBackend` plus an immutable `CudaExecutionPlan` used
  by Python `compile_cuda()`. The retained context/module remove repeated NVRTC
  compilation across evaluations. Current lowering covers rank-two and
  broadcast-batched matmul,
  rank-N concat with GPU-resident slice/pad-slice reverse nodes, global and axis reductions, transpose, broadcasting, `where`, `div`, `log`,
  and the supported unary primitives. Pure elementwise graphs use one fused
  CUDA kernel; general graphs also lower maximal elementwise tails after
  materialized leaves through `fusion_regions()`. Intermediate CUDA
  buffers are recycled by last-use liveness, while retained inputs, the output,
  and Adam state stay allocated. `TensorCudaExecutionPlan.device_buffer_count`
  exposes the current allocation count for diagnostics. It is still preview
  infrastructure: host inputs and outputs transfer per general evaluation. A
  gradient-output plan can retain one input and apply device-side SGD or Adam
  updates without materializing that parameter, gradient, or Adam moments on
  the host. Separate gradient plans can synchronize retained inputs through
  same-device GPU-to-GPU copies after each update. Python
  `cuda_adam_step(...)` batches the per-parameter update and synchronization
  sequence, while leaving symbolic gradient-plan construction explicit.
  `cuda_adam_optimizer(...)` stores inputs across steps; passing a dictionary
  to `step(...)` refreshes only mini-batch tensors without rebuilding plans or
  resetting GPU-resident parameter and Adam state. For symbolic VJP results
  from one graph, `cuda_adam_vjp_optimizer(...)` constructs one union plan,
  executes shared forward/reverse nodes once, and updates every parameter from
  that one device-resident gradient snapshot. It still lacks fused GEMM and
  distributed AD.
- `TensorIr.symbolic_vjp(output, cotangent_name)` now emits a transformable
  reverse graph with explicit cotangent input and one output node per original
  input gradient. `TensorTraceResult.symbolic_vjp(cotangent_name)` and
  `TraceTensor.symbolic_vjp(cotangent_name)` expose that graph to Python as
  named gradient trace results. This gives backend lowering
  an AD graph instead of a CPU-only reverse evaluator.
- `trace(fn, input_specs)` can execute a Python function with traced inputs and
  return a `TraceResult`.
- `TraceGraph.ir()` exposes structured node dictionaries with ids, ops, shapes,
inputs, input names, and typed `attrs` for lowering-sensitive parameters such
as scalar constants, `powf` exponents, reduction axes, and concat axes.
- `TraceGraph.lower_text()` renders that same trace as deterministic backend
  text with SSA-like ids, operand references, tensor shapes, and typed
  attributes. It is an inspection and lowering-contract prototype, not an
  MLIR or StableHLO-compatible serialization.
- `TraceGraph.vjp_ir(output_node_id)` can generate a minimal reverse-mode VJP IR
  for add, subtract, multiply, divide, `where`, concat, powf, tanh, exp, log,
  sqrt, sin, cos, transpose, reshape, matmul, and sum/mean outputs.
- `TraceGraph.evaluate(output_node_id, inputs)` can execute the traced primal
  graph on CPU.
- `TraceGraph.evaluate_vjp(...)` can execute reverse cotangent accumulation over
  the current add/sub/mul/div/where/concat/powf/tanh/exp/log/sqrt/sin/cos/transpose/reshape/matmul/sum/mean
  trace graphs on CPU. Greater-than mask nodes are treated as non-differentiable
  control values.
- `grad(fn, specs, values, cotangent)` wraps trace and VJP evaluation for Python
  users.
- `grad_fn(fn, specs, cotangent)` returns a reusable Python callable over value
  dictionaries by tracing once at transform creation.
- `grad_scalar_fn(fn, specs)` returns a reusable Python callable for scalar-loss
  gradients with an implicit `(1, 1)` cotangent seed and a cached trace.
- `grad_scalar(specs)` can be used as a decorator factory for scalar-loss
  gradient functions that trace once when the decorator is applied.
- `value_and_grad_fn(fn, specs, cotangent)` returns both the primal output and
  VJP gradients from one cached trace evaluation.
- `vjp_fn(fn, specs)` returns a reusable VJP callable that accepts an output
  cotangent at call time and returns primal output plus input cotangents.
- `jacobian_fn(fn, specs)` returns a reusable single-input dense Jacobian
  callable by applying repeated VJPs to one cached trace.
- `jacobians_fn(fn, specs)` returns reusable per-input dense Jacobian matrices
  as a dictionary keyed by input name, using repeated VJPs over one cached trace.
- `TraceGraph.evaluate_jvp(output_node_id, inputs, input_tangents)` evaluates
  primal values and tangents together through direct CPU forward-mode AD.
- `jvp_fn(fn, specs)` returns a reusable one-or-more-input JVP callable using
  that direct forward-mode evaluator rather than materializing dense Jacobians.
- `TraceGraph.compile_cpu(output_node_id)` freezes only the output-reachable
  traced subgraph into an immutable `CpuExecutionPlan`, eliminating unused
  nodes, remapping internal operands, and commoning identical pure unary
  subexpressions. It supports reusable CPU primal, JVP, VJP, and value-and-VJP
  evaluation; AD transforms and `jit` retain this plan. It establishes the
  execution boundary for later code generation, but is not a machine-code
  compiler yet.
- `jit(specs)` can be used as a decorator factory that traces a function once
  when the decorator is applied, then evaluates a frozen CPU execution plan for
  value dictionaries.
- `TensorTraceGraph.compile_mlx()`, `TraceTensor.compile_mlx()`, and
  `TensorTraceResult.compile_mlx()` expose the experimental MLX primal backend
  to Python. It supports elementwise arithmetic, unary math, matmul, global
  sum/mean, axis reductions, reshape, transpose, concat, and broadcast.
  Slice and internal zero-padding are rejected explicitly
  when a plan uses them.
- The Python bridge does not yet provide general compiled JIT lowering.
- `Tensor.split_key(...)`, `Tensor.random_normal(...)`, and
  `Tensor.glorot_normal(...)` provide deterministic stateless initialization
  without a global RNG.

Target direction:

- Keep Python as the user-facing authoring layer.
- Use Rust-owned tensor objects as tracer values under decorators such as
  `jit` and `grad`.
- Prefer tracing first. AST parsing can be explored later for a smaller,
  explicitly supported Python subset.
- Make Python object lifetime and Rust/device memory ownership explicit.

Near-term milestone:

- Add compiled lowering behind the current Python `jit` decorator factory.
- Expand IR-to-AD beyond the current primitive set.
- Add IR-to-backend lowering passes.

## Execution Roadmap

This section is the implementation order. A phase is not complete until its
acceptance checks pass on every backend it claims to support. The sequence
prioritizes a usable GPU-resident PINN training path before broader API surface
area, distributed execution, or source-to-source AD research.

### P0. Numerical Contract And Reproducible Baseline

Goal: make backend correctness and performance regressions observable before
adding new compiler behavior.

- Define the public precision contract: CPU may retain its `f64` reference
  behavior while CUDA and MLX execute `f32`; backend comparison tests must use
  documented, operation-appropriate tolerances.
- Maintain differential tests for CPU, CUDA, and MLX primal execution, JVP,
  VJP, and the supported higher-order transforms.
- Keep reproducible benchmarks for batched MLP forward, `value_and_grad`, a
  Poisson residual, batched matmul, and an optimizer step. Record device,
  dtype, shapes, warm-up policy, synchronization policy, and transfer policy.
- Preserve explicit failure for unsupported backend operations. No backend may
  silently fall back to host execution.

Acceptance checks:

- `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and the Python test suite pass on the default build.
- The MLX feature passes its Rust and Python suites on Apple silicon.
- The CUDA Rust and Python suites pass on the configured GTX 1660 SUPER,
  including retained device execution, cuBLAS matmul, CUSOLVER solve,
  symbolic VJP/JVP, multi-output materialization, and device optimizer checks.
- `git diff --check` passes and benchmark artifacts report enough metadata to
  reproduce a result.

### P1. GPU-Resident Value-And-Grad Training Path

Goal: make a small PINN train through one high-level CUDA API without copying
parameters or gradients to the host on every step.

Status: implementation and Rust runtime validation complete on the configured
GTX 1660 SUPER. The 2026-07-28 driver-backed suite passed all 47 CUDA core
tests and the release Python suite, including retained device buffers,
symbolic VJP/JVP, multi-output materialization, cuBLAS, CUSOLVER, SGD, and
Adam. Benchmark execution remains separate from correctness validation.

- Add `tensor_value_and_grad_cuda_fn` for a scalar loss and named parameters.
  It must compile a shared forward/reverse union plan rather than one plan per
  parameter.
- Expose a reusable CUDA training callable that owns parameter, optimizer, and
  static collocation buffers. Dynamic mini-batch tensors may be refreshed
  explicitly without retracing or resetting optimizer state.
- Add a narrow, documented initializer and stateless random-key API needed to
  construct MLP parameters reproducibly.
- Provide a CUDA PINN integration example and test for Poisson/Helmholtz:
  loss decreases, boundary residual is bounded, PDE residual is bounded, and
  GPU allocation count stabilizes across training steps.
- Measure and eliminate avoidable H2D/D2H copies in the training loop.

Acceptance checks:

- A Python example trains a two-layer PINN using only Nabla tensor APIs after
  initialization.
- Profiling confirms no per-step parameter or gradient device-to-host copy.
- CPU reference and CUDA loss/gradient checks agree within the documented
  `f32` tolerance.

### P2. Composable `vmap`

Goal: replace fixed axis-0 shape lifting with a transform that composes with
AD and backend lowering.

- Introduce per-primitive batching rules for elementwise operators, matmul,
  reductions, transpose, broadcast, concat, slice, and `where`.
- Support `in_axes`, `out_axes`, unmapped (`None`) arguments, and normalized
  negative axes. Keep batch extent static until shape polymorphism exists.
- Validate `vmap(grad)`, `grad(vmap)`, `vmap(jvp)`, and `vmap(vjp)` against a
  loop reference while asserting that execution does not use a Python
  per-example loop.
- Include the batching signature in compilation cache identity.

Acceptance checks:

- Per-example gradients for a batched MLP agree with the loop reference on CPU
  and CUDA.
- CUDA and MLX execute one batched plan for their supported primitive subsets.

Implementation status (2026-07-15): batch-axis propagation and public
`in_axes`/`out_axis`/`None` layout normalization are implemented for the
existing primitive surface. CPU coverage validates non-leading axes, unmapped
arguments, reductions, transpose, batched JVP, and batched VJP. CUDA now
lowers batched VJP primal and gradient outputs into one union plan; CPU and
CUDA tests compare per-example gradients of a batched MLP with a loop reference.
MLX now lowers the same batched VJP primal and gradient outputs into one
multi-output GPU plan; its CPU-parity coverage includes a non-leading mapped
axis and an unmapped parameter. MLX also lowers batched primal/JVP pairs from
explicit runtime tangent IR inputs in one multi-output GPU plan, with
non-leading input and output axes covered against the CPU reference.
CUDA now lowers the same explicit-tangent batched JVP union plan; its runtime
parity test is gated on `NABLA_CUDA_TEST` until the configured CUDA host exposes
its NVIDIA driver and runtime libraries.

### P3. SciML Array And Linear-Algebra Surface

Goal: cover the array semantics required by PINNs, operator learning, and
inverse problems without attempting full NumPy compatibility.

- Add multi-axis reductions, `keepdims`, extrema, norms, gather/scatter, and
  an explicit indexing/view model with differentiability rules.
- Add a scoped `einsum` subset, `solve`, Cholesky, triangular solve, and their
  AD rules where numerically well-defined.
- Add common neural primitives: `relu`, `sigmoid`, `softplus`, `abs`,
  `maximum`, and `minimum` with explicit subgradient policy.
- Add dtype and device-placement APIs. Implicit cross-device copies are an
  error.

Implementation status (2026-07-15): multi-axis `sum`/`mean` and `keepdims`
are available for eager and traced tensors. They preserve symbolic AD and
lower through the existing CPU, CUDA, and MLX reduction/reshape paths.
Elementwise `maximum`/`minimum` are also available on eager and traced tensors;
they compose `greater` and `where`, with equality subgradients routed to the
right operand. `max`/`min` reductions now accept the same axis/keepdims
contract and lower static traces through slice plus selection nodes; repeated
extrema route the derivative to the final row-major coordinate. Native `sqrt`
and L2 `norm` are available across CPU, CUDA, and MLX; negative values follow
IEEE floating-point `NaN` semantics and all
derivative orders at zero use an explicit zero subgradient. Native fused extrema
reductions and other norm variants remain pending. Static integer-index
`gather` and functional `scatter_add` are available with repeated-index VJP
accumulation; dynamic tensor indices, masks, and assignment-style scatter
remain pending the dtype/index IR design. A scoped matrix-product `einsum`
subset lowers directly to rank-N `matmul`. Rank-2 `solve(A, B)` now has a
partial-pivot LU CPU reference implementation, eager and traced Python APIs,
and direct/symbolic JVP/VJP plus mixed-tangent rules; it rejects non-square,
rank-mismatched, and singular inputs explicitly. CUDA now has a CUSOLVER
`Sgetrf`/`Sgetrs` rank-2 lowering that keeps factorization and right-hand-side
buffers on device; it passed driver-backed CUSOLVER validation on the configured
GTX 1660 SUPER on 2026-07-28.
`cholesky()` is available as a rank-2 eager factorization with symmetry/SPD
validation and as a static trace-time recurrence, so existing JVP/VJP machinery
provides verified derivatives. Native CUDA/MLX factorization lowering and
broader contractions remain pending. MLX 0.25.3
only exposes `linalg::solve` on a CPU stream, so Nabla rejects it on the MLX GPU
backend rather than silently falling back. `relu`, `abs`, `sigmoid`, and a
numerically stable `softplus` are available on eager and traced tensors; `relu`
uses a zero subgradient at zero, while `abs` follows the existing `where`
tie-rule and has derivative -1 at zero. Dtype/device APIs remain pending.
`tril` and `triu` now project the last two axes of rank-N tensors across CPU,
CUDA, and MLX, with direct/symbolic JVP/VJP and mixed-tangent propagation; they
provide the gradient masking primitive required before triangular solve and
Cholesky can be added.
`solve_triangular(rhs, lower=True, transpose=False)` is now available for
rank-2 eager and traced tensors. Its reference lowering composes triangular
projection with `solve`, preserving existing AD semantics; unit-diagonal and a
dedicated CUDA TRSM lowering remain pending performance work.
Eager and traced tensors now share a
differentiable integer/contiguous-slice indexing subset; it lowers through
Slice/PadSlice on CPU, CUDA, and MLX. Strided/empty slices, ellipsis, and
advanced indexing remain pending the explicit gather/scatter design.

Acceptance checks:

- Reference SciML examples use no NumPy operation inside the differentiated
  model body.
- Each new primitive has CPU finite-difference or analytic AD checks and
  backend parity coverage.

### P4. Compiler Passes And Kernel Performance

Goal: turn the current frozen plan into a proper backend-neutral compilation
pipeline.

- Formalize `kernel_ir()` with typed shapes, dtypes, layouts, placements,
  alias information, and effects.
- Add canonicalization, constant folding, CSE, DCE, broadcast/layout
  propagation, fusion partitioning, and buffer planning passes.
- Prioritize CUDA fusion for GEMM+bias+activation, residual/loss reductions,
  activation backward chains, and batched MLP blocks.
- Evaluate CUDA Graphs for static training steps. Treat MLIR/StableHLO export
  as an optional interoperability path, not a prerequisite for useful JIT.

Implementation status (2026-07-29): `kernel_ir()` now carries a typed
`TensorDType` contract rather than a stringly-typed dtype field. Eager tensors
currently emit logical `f64`, while CUDA/MLX `f32` execution remains an
explicit backend lowering decision rather than an implicit metadata rewrite.
`TensorDeviceBackend::precision()` records this boundary as logical/execution
dtype pairs: CPU is `f64`/`f64`, CUDA and MLX are `f64`/`f32`.
`TensorIr::stablehlo_text(output)` is an explicit interoperability probe for
static `f64` input/add/multiply/tanh graphs. It emits deterministic textual
StableHLO and rejects every operation outside that verified subset; it is not
an executable MLIR pipeline, a fallback backend, or a claim of broad StableHLO
coverage.
The IR also carries scalar/row-major-contiguous layout, backend-neutral `unplaced`
placement, and input/pure effect metadata. The validator enforces these current
invariants; physical placement, aliases, effects beyond inputs, and buffer
planning remain pending. `reshape` additionally exposes a logical alias
candidate in the IR; physical buffer reuse is still deferred to buffer planning.
Scalar-only constant subgraphs are folded during CPU-plan compilation when the
fold preserves checked runtime semantics; division by zero and invalid log/powi
domains intentionally remain runtime operations. A final DCE pass then removes
the unreachable source constants and intermediates introduced by folding or CSE
before any backend sees the plan. `buffer_plan()` now provides a backend-neutral
exact-size temporary-slot schedule from final-use liveness; inputs remain
external bindings and reshape aliases preserve their backing storage. CUDA has
not yet been refactored to consume this common schedule. Canonicalization also
removes identity reshape/broadcast nodes and collapses reshape chains before
backend lowering, avoiding unnecessary layout copies.
The base plan remains `Unplaced`, while `kernel_ir_with_placements(...)` can
apply validated typed compiler metadata without changing execution: a
`TensorDeviceId`, shaped `TensorDeviceMesh`, and
`TensorPartitionSpec::{Replicated, Sharded}`. This is an IR contract only; no
backend lowers it to multi-device work yet.

`examples/benchmark_pinn_cuda.py` records reproducible compilation, warm-up,
and synchronized device-resident Poisson PINN Adam-step timings, including
shape, transfer, and buffer-stability metadata. `cuda-host` has a visible GTX
1660 SUPER and the CUDA runtime needed for functional parity tests. On
2026-07-30, the release extension ran its width-16, four-collocation-point
two-layer Poisson PINN for 1,000 timed steps after 100 warm-up steps at
711.02 steps/s (1.406 ms/step). Compilation took 374.02 ms, loss decreased
from `1.07943933e3` to `6.26469124e-3`, and the retained CUDA buffer count
remained 58. Timed steps exclude diagnostic readback but are synchronized by
the following `loss()` call. This is a small fixed-shape GTX baseline, not a
cross-backend comparison or a large-model throughput claim. The same run with
width 64 compiled in 1,248.56 ms and reached 693.11 steps/s (1.443 ms/step),
with loss `4.54379500e5 -> 8.85865356e2` and the same 58 buffers. The 2.6%
step-time difference at four collocation points shows that this harness is
dominated by fixed launch/dispatch work; a larger-collocation benchmark is
required before attributing throughput changes to GEMM or fusion scaling. The
benchmark now accepts `--collocation` while retaining its fixed-shape/device
resident contract. At width 64, 512 collocation points ran at 657.96 steps/s
(1.520 ms/step; 1,338.54 ms compile) and 8,192 points ran at 368.81 steps/s
(2.711 ms/step; 1,605.91 ms compile), both with 58 retained buffers. These
measurements establish that the larger workload is no longer dominated solely
by fixed dispatch cost; they remain one-device baselines, not evidence of a
fusion speedup until a before/after comparison exists.

`examples/benchmark_pinn_mlx.py` records the corresponding MLX compile,
warm-up, fixed-shape training-step, dtype, device, synchronization, and
transfer metadata. On the local Apple silicon host (2026-07-28), its
1,000-step baseline was 1,906.15 steps/s (0.525 ms/step) for the one-parameter,
seven-point Poisson PINN; loss decreased from `5.51` to `1.28e-12`. This is a
local small-graph baseline, not a cross-backend or large-model performance
claim.
`examples/benchmark_loop_value_and_grad_mlx.py` records fixed-bound `Fori`
Adam steps through the same device-resident reverse path. It times only
`optimizer.step()` after warm-up and excludes diagnostic loss readback.
`examples/benchmark_mlp_value_and_grad_mlx.py` additionally records the
compile, warm-up, and steady-state timing of a batched two-layer MLP scalar
value-and-gradient specialization. It explicitly includes dynamic input upload
and loss/gradient diagnostic readback in each timed call, separating that
end-to-end measurement from the device-resident optimizer benchmark. On the
same local Apple silicon host (2026-07-28), its batch-256, width-64 baseline
was 2,699.41 calls/s (0.370 ms/call); this is likewise not a cross-backend
comparison.
`examples/benchmark_forward_mlx.py` records corresponding end-to-end batched
matmul and two-layer MLP forward baselines. Its explicit host transfer and
materialization metadata prevents those diagnostics from being interpreted as
device-only kernel throughput. `TensorMlxExecutionPlan.evaluate_device(...)`
also evaluates and synchronizes a primal output without materializing it on
the host, so the same script can exclude output readback while still reporting
the dynamic host-input upload that remains in each invocation.
`TensorMlxExecutionPlan.retain_inputs(...)` now retains selected static
bindings across primal calls, and `benchmark_forward_mlx.py
--retain-static-inputs` records that mode separately from the all-dynamic
baseline. On the same host (2026-07-28), retaining static weights improved a
batch-256, feature-64, output-64 matmul from 0.182 to 0.172 ms/call and a
two-layer width-128 MLP from 0.231 to 0.221 ms/call. These device-only
measurements exclude output readback but still include dynamic-input upload.

`fusion_regions()` now exposes maximal elementwise partitions with explicit
materialized inputs. It prevents unsafe fusion across values consumed by a
non-elementwise operation, such as a matmul, and is the lowering contract for
partial CUDA fusion. The configured GTX 1660 SUPER passed driver-backed fusion,
buffer reuse, symbolic VJP, and device optimizer coverage on 2026-07-28.

Acceptance checks:

- Each pass has IR-level golden tests and preserves CPU reference results.
- Benchmarks demonstrate an attributable speedup for at least one end-to-end
  MLP/PINN training step, not only an isolated microkernel.

### P5. Shape Polymorphism And Structured Control Flow

Goal: support practical variable batch sizes and iterative differentiable
programs without tracing arbitrary Python control flow.

- Add symbolic dimensions and shape constraints with bounded specialization.
- Add explicit IR operations analogous to `cond`, `scan`, and `fori_loop`.
- Define JVP/VJP semantics for each control-flow operation and reject
  data-dependent Python branches during tracing.

Implementation status (2026-07-27): `tensor_jit_batch_fn(...)` provides the
first bounded-specialization path for CPU plans. It traces lazily on the first
observed mapped batch size, caches a fixed number of batch-size plans, and
rejects changes to non-batch dimensions or unmapped inputs.
`tensor_value_and_grad_batch_fn(...)` uses the same cache contract for scalar
losses and performs their VJP from the frozen plan, so variable collocation
batch sizes now have a CPU reference for both values and gradients.
`tensor_value_and_grad_batch_mlx_fn(...)` mirrors that bounded specialization
contract on MLX for requested scalar-loss gradients, with one multi-output GPU
plan per batch size and explicit result readback.
`tensor_jit_batch_cuda_fn(...)` and
`tensor_value_and_grad_batch_cuda_fn(...)` now provide the same bounded cache
contract for CUDA primal and scalar value-and-gradient plans. Their Rust
runtime coverage now passes on the configured Linux host with `libnvrtc.so`.
Symbolic
dimensions and explicit control-flow IR remain pending.
`TraceTensor.__bool__` now rejects data-dependent Python branches explicitly,
so tracing cannot silently specialize to one host-side branch.
`tensor_cond_fn(...)` now freezes true/false CPU branch regions and evaluates
only the host-boolean-selected branch; its scalar value-and-gradient variant
selects the matching branch VJP, while `tensor_cond_jvp_fn(...)` selects its
matching JVP. This establishes lazy branch semantics without misrepresenting
`where` as control flow. `TensorRegion` and `TensorCondExecutionPlan` now own
the identical-capture validation and selected-branch primal/JVP/VJP execution
in `nabla-core`, rather than leaving this contract in the Python bridge.
`TensorOp::Cond` now owns two frozen `TensorRegion` plans plus ordered,
explicit named capture bindings to parent node IDs. CPU execution evaluates the
scalar predicate first and materializes only the selected branch; DCE, remap,
buffer planning, direct JVP, and direct VJP preserve those bindings. The
current regression covers a capture from a parent intermediate, lazy inactive
branch failure avoidance, and branch-selected first-order derivatives.
`tensor_cond(predicate, on_true, on_false, operands)` exposes the same
contract to Python tracing, including nested CPU regions. Its operands are the
only legal branch captures. Symbolic JVP and VJP now rewrite `Cond` regions:
they preserve the predicate and use branch-selected primal/tangent or gradient
regions. Region transform outputs retain a stable capture ABI even if DCE
would otherwise remove an unused capture in one branch. Public scalar
Hessian/HVP calls through `Cond` compose symbolic VJP with symbolic JVP, so
second coordinate derivatives needed by PINNs work on CPU. General mixed and
higher-order region transforms remain pending. CUDA and MLX explicitly reject
`Cond` before device lowering; neither backend eagerly materializes both
branches. Vmappable conditional predicates remain pending.
`tensor_fori_loop(...)` remains the compatibility API that statically unrolls
fixed host-integer bounds with a same-graph, shape-preserving TraceTensor
carry. `TensorForiExecutionPlan` now owns a frozen fixed-bounds body region:
CPU primal execution iterates its carry at runtime, JVP propagates carry and
explicit-capture tangents, and VJP records a carry tape then replays body VJPs
in reverse. `TensorOp::Fori` embeds that plan in parent Tensor IR with explicit
capture bindings; DCE/remap, CPU execution, direct JVP/VJP, text lowering, and
CPU compilation preserve the region boundary. `tensor_fori_loop_region(lower,
upper, body, init, operands)` exposes one-time Python tracing of
`body(index, carry, *operands)`, where the scalar index is a TraceTensor and
all external captures are explicit. Batch specialization rejects `Fori` rather
than silently unrolling or changing layout. `tensor_vmap_fn` supports
structured `Fori` with a canonical leading mapped carry and mapped captures
of the same batch extent; unmapped captures remain shared across examples.
The loop body must preserve the carry batch axis. CPU `vmap` JVP and VJP reuse
the structural loop transforms and preserve per-example carry/capture
gradients. MLX executes the same vmapped `Fori` VJP plan for its supported
first-order region path. MLX now executes the primal
fixed-bound `Fori` body by
host-dispatching device-resident MLX arrays: carry and captures remain on the
GPU between iterations, but this is not yet a fused Metal control-flow kernel.
CUDA lowers a restricted primal `Fori` body to one device kernel: each thread
owns one carry element and iterates the fixed bounds in device code. It also
lowers first-order `ForiVjp` to a single device kernel when the body is pure
elementwise and every explicit capture broadcasts to the carry shape. That
kernel writes a global-memory carry tape during its forward recurrence, then
replays the body VJP in reverse in the same thread; it does not materialize
carries or gradients on the host. Body reductions, matmul/solve, nested
regions, and broadcasted captures for `ForiVjpJvp` remain explicit rejections.
CUDA also lowers `Scan`
primal and first-order `ScanVjp` when its per-step output matches the carry
shape, plus the verified direct-broadcast output subset for unequal lane
counts, and explicit captures broadcast to it. A shared CUDA Scan
execution writes final carry and time-major outputs in one kernel, retaining
the sibling result in a device-buffer cache; its VJP reuses a carry tape and
combines final-carry/output cotangents during reverse replay. CUDA `ForiVjp`
and `ScanVjp` support scalar and legal trailing-axis broadcast captures by
emitting per-element contributions, mapping each carry index to its broadcast
target offset, and atomically reducing on device. `ScanVjpJvp` supports the
same direct-broadcast unequal-lane subset and broadcast-compatible captures;
arbitrary unequal carry/output graphs remain explicit rejections. First-order VJP source groups now compile
to one multi-output kernel, so all requested gradient targets reuse one reverse
traversal and device carry tape. Linux CUDA
compilation and GTX 1660 SUPER CPU-parity coverage exercise nonlinear loops
for both initial-carry and same-shape capture gradients.
Symbolic JVP through `Fori` emits a paired primal `Fori` and structural
`ForiJvp` tangent node, preserving separate primal/tangent carries through the
runtime loop rather than statically expanding it. Symbolic VJP through `Fori` uses a
fixed-bound carry tape and grouped `ForiVjp` result nodes, so gradients of the
initial carry and explicit captures reuse one reverse loop. JVP through a
`ForiVjp` result now emits a grouped `ForiVjpJvp` node. Its execution plan
records primal and directional carry tapes, then replays compiled JVPs of each
body-gradient in reverse. Public `hessian_scalar` and `hvp_scalar` therefore
select this forward-over-reverse transform for fixed `Fori` regions rather than
the CPU mixed-direction fallback. `ForiVjpJvp` has no further derivative rule.
CUDA lowers its restricted pure-elementwise, same-shape-capture subset with
primal and tangent carry tapes in one device kernel; batch specialization
still rejects it. MLX executes first-order `ForiVjp` results by retaining the
forward carry tape and replaying a precompiled body VJP plan on the GPU; carry
and capture gradients remain device-resident until the requested plan output
is materialized. MLX also executes `ForiVjpJvp` by retaining primal/tangent
carry tapes, then replaying precompiled gradient-JVP plans on the GPU. `tensor_scan(...)` remains the compatibility API that statically unrolls a fixed-length
carry/output loop. `TensorScanExecutionPlan` executes one frozen two-result
body region and returns final carry plus a leading-time-axis output stack.
`TensorOp::Scan` embeds the paired results in parent Tensor IR with a shared
execution group, so CPU primal evaluation executes the body once even when
both results are consumed. Direct JVP and VJP likewise process both results
together: JVP caches paired tangents, while VJP jointly replays carry/output
cotangents through the forward carry tape. `tensor_scan_region(lower, upper,
body, init, operands)` exposes one-time Python tracing of that region with
explicit captures. Symbolic JVP packs primal/tangent carries and per-step
outputs into a paired Scan region, so the transformed graph retains runtime
scan control flow. Symbolic VJP emits grouped `ScanVjp` result nodes: final
carry and output cotangents are replayed jointly through one reverse scan tape,
then initial-carry and explicit-capture gradients reuse that cached result.
MLX executes primal `Scan` with the same device-resident host-dispatch model
as `Fori`, including one shared execution cache when final carry and stacked
outputs are both consumed. `tensor_vmap_fn` applies the same leading-axis
contract to `Scan`: final carry retains batch axis zero and stacked outputs
are normalized from the native time-major stack to `[batch, time, ...]`
before the requested `out_axis` layout is applied. Direct CUDA vmap tracing
now has GTX 1660 SUPER CPU-parity coverage for Scan primal, JVP, nonlinear
VJP with initial-carry and capture gradients, and nonlinear per-example HVP
with respect to mapped initial carry and mapped capture. The HVP wrapper also
restores a selected input's non-leading public `in_axes` layout for an
elementwise loss. It requires one scalar loss per example and lowers the sum
through the existing structural `ScanVjpJvp` path; its CPU result is checked
against a central finite-difference VJP gradient. `TensorExecutionPlan::specialize_batch` still rejects structured Scan
for the bounded JIT-batch API rather than unrolling or falling back to CPU.
MLX executes first-order `ScanVjp` results
with a device-resident forward carry tape and separate precompiled body VJP
plans for final-carry and per-step-output cotangents; their carry/capture
contributions are accumulated on the GPU. `ScanVjpJvp` now binds primal/tangent
initial carry, final-carry cotangent and tangent, stacked-output cotangent and
tangent, plus primal/tangent explicit captures. Its CPU and MLX plans retain
primal/tangent carry tapes, slice each output-cotangent pair during reverse
replay, and add directional contributions from both body outputs. The
acceptance gate is a nonlinear Scan loss that consumes both final carry and
stacked outputs: CPU HVP matches a central finite-difference reference, and
MLX matches CPU without materializing intermediate carries or gradients on the
host. Further derivatives of `ScanVjpJvp` remain explicit errors.

Control-flow completion status:

`TensorRegion` now provides explicit capture bindings for `Cond`, `Fori`, and
`Scan`. CPU evaluates only the selected conditional region; structural JVP,
VJP, Hessian, and HVP paths preserve that laziness. Kernel IR, canonicalization,
and buffer planning retain the region boundary, while CUDA and MLX reject
device-predicate `Cond` explicitly. Python exposes `tensor_cond(...)` after
lazy inactive-branch, nested-condition, and AD coverage. This closes the
original CPU control-flow increment; device-predicate lowering remains a
separate backend project.

Next implementation order:

Completed: symbolic JVP now has a multi-output result contract while preserving
the single-output API. Retained sibling `ScanVjpJvp` targets lower as one CUDA
source group; the first node launches the shared kernel, caches sibling device
buffers, and GTX 1660 SUPER parity covers carry plus explicit-capture targets.

1. Run P6's representative larger MLP/PINN MLX performance validation and
   report compile, device-step, and diagnostic-readback timing separately.
2. Keep arbitrary indexed or non-elementwise unequal-lane Scan bodies as
   explicit CUDA rejections until their reduction structure is represented in
   device reverse IR; add a rejection test for every unsupported form.
3. P7 needs a host exposing at least two CUDA devices and NCCL. When that
   prerequisite is available, connect the typed sharding schedule to the
   Python training interface, add two-GPU loss/gradient parity, then measure
   collective and readback boundaries separately.

Implementation note (2026-07-29): `TensorForiTape` retains the fixed-bound
forward carry sequence and is reused by direct and symbolic reverse paths.
`TensorMultiRegion` and `TensorForiMultiExecutionPlan` retain shared
multi-output body plans, execute independently shaped carries in one loop, and
reverse all body outputs jointly in each iteration. Parent `ForiVjp` nodes
select the initial-carry or explicit-capture gradient and share one cached
reverse pass per source loop. `ForiVjpJvp` reuses a shared directional reverse
pass for all selected gradients in that source group. CUDA now fuses each
restricted first-order `ForiVjp` and `ScanVjp` source group into one
multi-output device kernel. CUDA `ForiVjpJvp` currently lowers one selected
directional-gradient result per kernel. `ScanVjpJvp` also lowers the verified
direct-broadcast unequal-lane subset, one selected directional-gradient result
per kernel.
MLX supports first-order reverse and forward-over-reverse `Fori`
paths, but not higher derivatives of `ForiVjpJvp`.

Acceptance checks:

- A time-stepping differentiable model and multiple collocation batch sizes run
  with bounded recompilation and verified gradients.

### P6. MLX Reverse-Mode Parity

Goal: make the Apple backend a usable training backend rather than primal-only
execution.

Implementation status (2026-07-24): `tensor_value_and_grad_mlx_fn(...)`
lowers a scalar loss and its requested symbolic VJP gradients into one
multi-output MLX plan. `mlx_adam_loss_optimizer(...)` retains selected static
inputs, parameters, gradient outputs, and bias-corrected Adam moments across
steps; the MLX executor evaluates updates on `StreamOrDevice::gpu()` and does
not materialize parameters or gradients on the host during `step()`. `loss()`
and `parameters()` are explicit diagnostic readbacks. The training API has
scalar-loss parity for its supported Tensor IR operations. The fixed-shape
one-parameter Poisson PINN in `examples/pinn_poisson_mlx.py` combines PDE and
boundary terms into one graph, converges on MLX, and checks its final parameter
against the CPU reference within `2e-3`. The two-layer Poisson MLP integration
in `examples/pinn_mlp_mlx.py` covers second coordinate derivatives, four
device-resident Adam parameters, residual and boundary losses, and CPU parity.
Its MLX-gated integration test reaches residual below `1e-5`, boundary loss
below `1e-5`, and maximum CPU/MLX parameter difference below `2e-3` after
1,500 fixed-shape steps. Release MLX performance validation records three
separate boundaries: a 1,024-collocation Poisson PINN compiles in `37.86 ms`,
runs device-resident Adam at `0.935 ms/step`, and performs diagnostic loss
readback at `0.486 ms/call` on Apple silicon. The batched two-layer MLP
value-and-gradient diagnostic path at batch `4,096` and width `256` compiles
in `402.97 ms` and takes `1.753 ms/call`; its host loss/gradient materialization
is deliberately not compared with device-resident optimizer throughput.
Unretained mini-batch inputs can be supplied to `step(inputs)` and replace only
their host binding; retained parameters and Adam moments stay on the device.
Contiguous rank-N `Slice` and reverse-mode `PadSlice` now lower to MLX
device-side take/pad operations; primal and symbolic-VJP parity are covered
against the CPU reference.

Remaining work is performance validation on representative larger MLP/PINN
workloads; no MLX result is presented as a CUDA/JAX comparison.

Acceptance checks:

- The P1-style one-parameter Poisson PINN runs on MLX and agrees with CPU
  within its `f32` tolerance. Completed by
  `test_mlx_poisson_pinn_matches_cpu_reference`.

### P7. Distributed Sharding

Goal: add multi-device execution only after single-device plans, layout, and
memory contracts are stable.

Implementation status (2026-07-30): typed `TensorPlacement` metadata now
models unplaced, single-device, and mesh placement. Mesh validation rejects
duplicate devices, invalid topology, unknown mesh axes, out-of-range tensor
axes, and static extents that cannot be evenly sharded. Existing Python
`kernel_ir()` output continues to project the default placement as
`"unplaced"`; there is no distributed execution, implicit copy, or collective.
`kernel_ir_with_placement_propagation(...)` is an opt-in compiler check that
propagates supported local placements through elementwise operations,
transpose, and broadcast. It rejects mixing a placed tensor with an unplaced
non-scalar, reductions over a sharded axis, sharded matmul/solve/triangular
operations, and reshape or slice cases that require redistribution. It does
not insert copies or collectives.
`TensorExecutionPlan::sharding_plan(...)` extends that contract for legal
sharded reductions: it emits an ordered backend-neutral `TensorAllReduce`
schedule after each replica-local reduction and changes that result's
placement to replicated on the same mesh. The schedule distinguishes `Sum`
from `Mean`; a CUDA/NCCL backend must still lower and execute it. Sharded
matmul, solve, triangular, reshape, and slice redistribution remain rejected.
`TensorCpuExecutionPlan.value_and_grad_data_parallel(...)` exposes the same
reference from Python. `TensorExecutionPlan::value_and_vjp_data_parallel(...)`
is the deterministic CPU scalar-loss reference: it rebuilds a shape-specialized
IR for equal axis-zero shards, aggregates replica-local `Sum` or `Mean` losses
explicitly, concatenates mapped-input gradients, and adds replicated gradients
in increasing shard order. It has single-device VJP parity coverage but does
not yet spawn CPU workers.

The optional Linux `cuda-nccl` feature now adds
`CudaBackend::compile_data_parallel(plan, device_ordinals)`. It creates one
identical retained-output CUDA plan per explicitly selected GPU;
`CudaDataParallelExecutionPlan::execute_replicas(...)` accepts already-sharded
input maps, enqueues all replicas, NCCL-all-reduces every retained output in
place with `Sum` or `Mean`, and reports separate host-observed enqueue,
collective, and rank-zero readback durations. Ordinary `cuda` builds neither
link nor load NCCL, and the API returns a feature error instead of silently
falling back to one GPU. The execution primitive does not yet consume a
`TensorShardingPlan`, infer general input sharding, or implement cross-node
communicator setup. Python now exposes the same restricted contract through
`tensor_value_and_grad_data_parallel_cuda_fn(...)`: full-batch mapped inputs
are split on axis zero, requested parameters must be replicated, and the
callable returns only all-reduced parameter gradients plus timing diagnostics.
Mapped-input gradients are deliberately rejected rather than incorrectly
all-reduced. On `cuda-host`, Linux feature compilation, the core device contract,
and the Python constructor contract pass; the host exposes one GTX 1660 SUPER
and no NCCL library, so no collective or two-GPU numerical result is claimed.

Remaining implementation order:

1. Validate the Python scalar value-and-gradient callable on two GPUs against
   the deterministic CPU oracle for both `Sum` and `Mean`; only then consider
   mapped-input gradient concatenation as a separate output contract.
2. Bind `TensorShardingPlan::all_reduces` to CUDA lowering, preserving its
   operation order and rejecting schedules not represented by the first
   data-parallel subset.
3. Add failure-handling and communicator-lifecycle coverage before considering
   multi-node transport or tensor-parallel matmul.

Acceptance checks:

- Two-GPU data-parallel training matches the single-GPU reference within
  documented tolerance and records collective timing separately.

### Research Track: Rust Source-To-Source AD

This track is intentionally not on the critical path. Start with a procedural
macro over a pure, restricted Rust expression subset and compare it against
Tensor IR AD and Enzyme. Do not claim support for arbitrary borrowing,
mutation, dynamic dispatch, async code, or Python callbacks until their
semantics are explicitly modeled and tested.

Implementation status (2026-07-27): the `nabla-macros` procedural-macro crate
exports `forward_diff!(|x| expression)`. It parses a one-argument closure and
rewrites the parameter to a `Dual::variable` seed at expansion time, so Rust's
operator and method resolution produces forward-mode value/derivative pairs
without a runtime tape. Analytic tests cover captured scalar constants, block
expressions, elementary methods, and `powi`. This is a source-level prototype,
not general source-to-source AD: mutation, arbitrary control flow, multiple
outputs, and type-generic captures remain unsupported.
`forward_gradient!(|x, y, ...| scalar_expression)` now adds an ordered
multi-input scalar gradient by expanding one Dual seed per input and returning
`ForwardGradient`. It is intentionally O(input dimension); dense Jacobian,
HVP, and reverse-mode source transforms remain future work.
