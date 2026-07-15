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
- `tensor_vmap_fn(fn, input_specs, batch_size)` performs a compiler-level,
  fixed-size axis-0 batch transform: it prepends `batch_size` to each
  per-example input shape, traces once, and returns one reusable CPU plan.
  `tensor_vmap_cuda_fn(...)` and `tensor_vmap_mlx_fn(...)` lower the same trace
  to CUDA and MLX. This initial contract intentionally excludes `in_axes=None`,
  nonzero/negative axes, `out_axes`, and dynamic batch sizes.
- `tensor_hessian_scalar_fn(fn, input_specs, input_name)` and
  `tensor_hvp_scalar_fn(fn, input_specs, input_name)` freeze a scalar rank-N
  trace for dense Hessian or Hessian-vector-product evaluation. Dense Hessian
  materialization remains a correctness-first O(n^2) CPU path.
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
  CUDA kernel; general graphs still lower per operation. Intermediate CUDA
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
- The CUDA feature passes its Rust and Python suites on the configured Linux
  GPU, including a retained-device execution check.
- `git diff --check` passes and benchmark artifacts report enough metadata to
  reproduce a result.

### P1. GPU-Resident Value-And-Grad Training Path

Goal: make a small PINN train through one high-level CUDA API without copying
parameters or gradients to the host on every step.

Status: completed. The CUDA implementation is verified on the configured
Linux GPU with a two-layer Poisson PINN integration run.

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
buffers on device; it has passed Linux feature compilation but remains pending
runtime validation on a host with `libcusolver` and a visible NVIDIA device.
Cholesky, triangular solve, and broader contractions remain pending. MLX 0.25.3
only exposes `linalg::solve` on a CPU stream, so Nabla rejects it on the MLX GPU
backend rather than silently falling back. `relu`, `abs`, `sigmoid`, and a
numerically stable `softplus` are available on eager and traced tensors; `relu`
uses a zero subgradient at zero, while `abs` follows the existing `where`
tie-rule and has derivative -1 at zero. Dtype/device APIs remain pending.
`tril` and `triu` now project the last two axes of rank-N tensors across CPU,
CUDA, and MLX, with direct/symbolic JVP/VJP and mixed-tangent propagation; they
provide the gradient masking primitive required before triangular solve and
Cholesky can be added.
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

Acceptance checks:

- A time-stepping differentiable model and multiple collocation batch sizes run
  with bounded recompilation and verified gradients.

### P6. MLX Reverse-Mode Parity

Goal: make the Apple backend a usable training backend rather than primal-only
execution.

- Lower rank-N `slice` and internal `pad_slice` correctly on MLX.
- Add MLX device-resident parameters, gradients, and Adam state.
- Expose MLX `value_and_grad` and training APIs with the same semantic contract
  as CUDA for supported operations.

Acceptance checks:

- The P1 PINN example runs on MLX and agrees with CPU within its `f32`
  tolerance.

### P7. Distributed Sharding

Goal: add multi-device execution only after single-device plans, layout, and
memory contracts are stable.

- Model mesh, placement, and partition specifications in IR metadata.
- Implement data parallelism first, including deterministic gradient
  all-reduce; add tensor parallel matmul only after that path is stable.
- Lower collectives through NCCL on CUDA. Evaluate multi-node transport only
  after single-node semantics and failure handling are tested.

Acceptance checks:

- Two-GPU data-parallel training matches the single-GPU reference within
  documented tolerance and records collective timing separately.

### Research Track: Rust Source-To-Source AD

This track is intentionally not on the critical path. Start with a procedural
macro over a pure, restricted Rust expression subset and compare it against
Tensor IR AD and Enzyme. Do not claim support for arbitrary borrowing,
mutation, dynamic dispatch, async code, or Python callbacks until their
semantics are explicitly modeled and tested.
