# Nabla

Nabla is an early Rust-native SciML compiler-runtime experiment. The target is
a JAX-like scientific computing stack shaped around Rust's strengths:
compile-time shape checks, explicit memory layout, transformable autodiff, and
eventual backend/sharding contracts.

The current implementation is still small. It has a typed tensor seed, scalar
forward-mode AD, a differentiable Lotka-Volterra inverse problem, and a basic
optimizer. It also has a minimal PyO3 Python extension seed with a Rust-owned
Matrix, a TraceGraph recorder, and a Python `trace(fn, input_specs)` entrypoint.
It also exposes a structured graph IR seed through `graph.ir()` and a minimal
reverse-mode graph VJP path for the current primitive set through `grad(...)` /
`grad_fn(...)` / `grad_scalar_fn(...)` / `grad_scalar(...)` /
`value_and_grad_fn(...)` / `vjp_fn(...)` / `jacobian_fn(...)` /
`jacobians_fn(...)` / `jvp_fn(...)`.
Reusable `grad_*`, `value_and_grad_fn(...)`, `vjp_fn(...)`,
`jacobian_fn(...)`, `jacobians_fn(...)`, `jvp_fn(...)`, and `jit(...)`
transforms trace once when the callable/decorator is created, then evaluate the
cached graph on later calls. It does not yet have
source-to-source AD, compiled JIT lowering, GPU execution, or distributed
sharding.

Run the verified one-dimensional Poisson PINN example after `maturin develop`:

```sh
.venv/bin/python examples/pinn_poisson.py
```

Measure the frozen CPU plan in an optimized extension build:

```sh
.venv/bin/maturin develop --release
.venv/bin/python examples/benchmark_tensor_cpu.py
```

## Current Capabilities

- Scalar forward-mode automatic differentiation with `Dual`.
- Compile-time shaped `Tensor2<ROWS, COLS>` with shape-checked matmul.
- Pure-Rust `TensorIr` compiler core for dynamic rank-N input tensors with
  broadcasted add/multiply, scalar `sum`, CPU evaluation, direct JVP and
  reverse-mode VJP with cotangent reduction back through broadcast axes, and
  deterministic lowering text. It is deliberately independent of the current
  PyO3 2D `TraceGraph` while that bridge is migrated incrementally.
- Python-facing eager `Tensor` class backed by contiguous row-major Rust storage,
  with positive runtime shape validation, rank-N trailing-axis broadcasting for
  add/subtract/multiply/divide, NumPy-style batched `matmul`, and
  element-count-preserving reshape. `Tensor` is not yet traceable,
  differentiable, or part of the CPU plan. `Tensor.slice(...)` returns a
  zero-copy, read-only `TensorView` with explicit shape, strides, and offset;
  source tensors and views share immutable storage, while every arithmetic
  operation returns a new contiguous allocation.
- Python `trace_tensor(fn, input_specs)` bridge for the rank-N `TensorIr` core.
- `tensor_jacobian_fn(fn, input_specs, input_name)` freezes one rank-N trace
  and returns an output-flat by input-flat dense Jacobian for the selected input.
  Its `TraceTensor` values currently support broadcasted add/subtract/multiply/divide,
  batched `matmul`, rank-N `transpose`, `tanh`, `exp`, `sin`, `cos`, `sqrt`, non-negative integer `powi`, `log`, reshape, and global or single-axis `sum`/`mean`; `TensorTraceGraph.evaluate_vjp(...)` and
  `TensorTraceGraph.evaluate_jvp(...)` execute the corresponding rank-N CPU
  reverse and forward transforms. `TensorTraceGraph.hessian_scalar(...)`
  computes an exact dense Hessian for one named input and a scalar output using
  mixed second-direction AD, not finite differences. The established 2D `trace`
  API remains separate while its wider primitive set is migrated.
  `TensorTraceGraph.hvp_scalar(...)` returns the corresponding exact
  Hessian-vector product without materializing that dense matrix.
  `TensorTraceResult.symbolic_jvp(input_name)` instead emits a new transformable
  rank-N trace whose output is the coordinate JVP; it can be applied again for
  second derivatives and then differentiated with VJP with respect to model
  parameters. Its rules cover every current rank-N `TensorIr` primitive.
  `tensor_grad_scalar_fn(fn, input_specs)` traces and compiles a rank-N
  scalar-loss function once, then returns a reusable Python gradient callable
  backed by its frozen CPU plan.
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
- `TensorBackend` defines the rank-N plan execution contract; `CpuBackend` is
  the current implementation used by `plan.evaluate(...)`. LLVM, MLIR, GPU, and
  distributed backends are future implementations of this boundary.
- Python `Adam` updates immutable dictionaries of named rank-N `Tensor`
  parameters from VJP gradients. The test suite includes a manufactured 1D
  Poisson residual in which two symbolic coordinate JVP transforms form
  `u_xx`, then VJP and Adam recover one scalar MLP weight. This is a
  vertical-slice correctness proof. It includes batched collocation points and
  explicitly aggregated boundary/residual VJPs, but is not yet a general PINN
  framework.
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
  General rank-N broadcasting is available on eager `Tensor` and the separate
  `TraceTensor` add/multiply/sum path; `Matrix` and the legacy `TraceGraph`
  transforms remain deliberately 2D.
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
- Fixed-step RK4 ODE integration.
- Final-state RK4 integration for loss functions that do not need full
  trajectories.
- Lotka-Volterra parameter-inference loss.
- Full four-parameter Lotka-Volterra gradient.
- Generic fixed-size gradient descent fitting loop.
- Central finite-difference gradient checks.

## Run

```sh
cargo test
cargo run -p nabla-core --example lotka_volterra
cargo run -p nabla-core --example fit_lotka_volterra
```

Build the Python extension with maturin:

```sh
python -m pip install maturin
maturin develop
python tests/python/test_matrix.py
```

Expected example output shape:

```text
loss: ...
gradient: [..., ..., ..., ...]
```

The fitting example reports initial loss, final loss, optimization steps, and
the fitted parameter vector.

## Development Gates

Run these before treating a change as verified:

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## Roadmap

1. Expand typed tensors: static shapes, dynamic shapes, explicit strides/views.
2. Add compiled lowering behind the current Python-facing `jit` wrapper.
3. Expand reverse-mode AD transforms beyond the current primitive set.
4. Define backend traits, then add CPU lowering and later GPU/sharding paths.

See [docs/jax_like_roadmap.md](docs/jax_like_roadmap.md) for the architecture
direction.
