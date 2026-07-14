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
  CUDA kernel; general graphs still lower per operation. It is still preview
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
  sum/mean, reshape, transpose, concat, and broadcast. Axis reductions,
  comparisons/`where`, slice, and internal zero-padding are rejected explicitly
  when a plan uses them.
- The Python bridge does not yet provide general compiled JIT lowering.

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
