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
cached graph on later calls. It does not yet have source-to-source AD, a
general compiled JIT, or distributed sharding.

Run the verified one-dimensional Poisson PINN example after `maturin develop`:

```sh
.venv/bin/python examples/pinn_poisson.py
```

On Linux with the `cuda` feature, run the verified two-point batched Poisson
PINN, whose Adam optimizer state and collocation tensors stay on the GPU:

```sh
maturin develop --features cuda
python examples/pinn_poisson_cuda.py
```

Run the verified two-layer, four-parameter CUDA Poisson PINN with combined
residual and boundary losses:

```sh
python examples/pinn_mlp_cuda.py
```

Measure the frozen CPU plan in an optimized extension build:

```sh
.venv/bin/maturin develop --release
.venv/bin/python examples/benchmark_tensor_cpu.py
```

On Linux with the `cuda` feature, measure end-to-end broadcast-batched matmul
or fused elementwise-chain throughput (including the current host/device
transfers):

```sh
maturin develop --release --features cuda
python examples/benchmark_tensor_cuda.py
python examples/benchmark_tensor_cuda.py --elementwise --rows 1024 --inner 1024
python examples/benchmark_tensor_cuda.py --rank-two --device-resident
```

Measure device-resident two-layer Poisson PINN Adam steps with separate compile,
warm-up, and synchronized training timings:

```sh
python examples/benchmark_pinn_cuda.py
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
  with `zeros`/`ones`/`full`/`arange`/`linspace`/`eye` creation, positive runtime
  shape validation, rank-N trailing-axis broadcasting for add/subtract/multiply/divide,
  numeric scalars on either side of those arithmetic operations,
  scalar `**` exponents,
  NumPy-style batched `matmul`, rank-N `nabla.concat([...], axis=...)`,
  permutation-validated `transpose(axes=None)`, global or single-axis `sum`/`mean`/L2 `norm`,
  common elementwise math (`tanh`, `exp`, `log`, `sqrt`, `sin`, `cos`, `powi`),
  `gt(...)` masks, `maximum(...)`/`minimum(...)`, broadcasted `nabla.where(...)`, materialized `broadcast_to(shape)`,
  and element-count-preserving reshape.
  `Tensor` is not yet traceable,
  differentiable, or part of the CPU plan. `Tensor.slice(...)` returns a
  zero-copy, read-only `TensorView` with explicit shape, strides, and offset;
  source tensors and views share immutable storage, while every arithmetic
  operation returns a new contiguous allocation.
- Python `trace_tensor(fn, input_specs)` bridge for the rank-N `TensorIr` core.
- `tensor_jacobian_fn(fn, input_specs, input_name)` freezes one rank-N trace
  and returns an output-flat by input-flat dense Jacobian for the selected input.
  Its `TraceTensor` values currently support broadcasted add/subtract/multiply/divide,
  batched `matmul`, rank-N `concat`, `stack([...], axis=...)`, `slice(axis, start, stop)`, `broadcast_to(shape)`, rank-N `transpose`, `tanh`, `exp`, `sin`, `cos`, `sqrt`, non-negative integer `powi`, `log`, reshape, global or single-axis `sum`/`mean`/L2 `norm`, `maximum`/`minimum`, and `gt`/`where` masks. `stack` is composed from reshape plus concat, so it inherits the same direct and symbolic CPU/CUDA AD rules. `concat` is linear: direct and symbolic VJP split the upstream cotangent with internal slice nodes, while its JVP and mixed second-direction transform concatenate the corresponding tangents. `slice` supports normalized negative axes and uses a zero-padded internal reverse node, keeping direct and symbolic gradients on the selected original coordinates. `broadcast_to` is a dedicated shape node whose VJP reduces repeated axes back to the input shape. `sqrt` is a native IR primitive: negative values follow IEEE floating-point `NaN` semantics, while every derivative order at zero is defined as zero, avoiding `log(0)` during higher-order AD. Comparisons are explicitly non-differentiable; `where` routes VJP/JVP contributions only through the selected data branch. `maximum` and `minimum` are composed from those primitives and route equality subgradients to their right operand. `TensorTraceGraph.evaluate_vjp(...)` and
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
  A `TraceTensor` cannot be used as a Python boolean, preventing accidental
  data-dependent host branches during tracing; use `nabla.where` for
  elementwise selection while structured `cond`/loop IR is pending.
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
- `tensor_vmap_jvp_fn(...)` and `tensor_vmap_vjp_fn(...)` apply the same
  batch-layout contract to runtime tangents and cotangents. Mapped input
  gradients are restored to their declared `in_axes`; unmapped input gradients
  aggregate over the mapped batch as required by reverse-mode AD.
  `tensor_vmap_vjp_cuda_fn(...)` lowers the primal plus all requested VJP
  outputs into one CUDA union plan.
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
  repeated gather/scatter coordinates correctly. Dynamic tensor indices,
  boolean masks, and assignment-style scatter are not yet supported.
- `nabla.einsum("ij,jk->ik", [lhs, rhs])` and
  `nabla.einsum("...ij,...jk->...ik", [lhs, rhs])` are scoped matrix-product
  spellings that lower directly to the existing rank-N `matmul` plan. Other
  einsum equations are rejected rather than silently interpreted.
- `Tensor` and `TraceTensor` support `__getitem__` with integer and
  contiguous unit-step slice tuples, including negative indices. Integer
  indices lower to a length-one slice plus reshape, so reverse-mode AD and
  CUDA/MLX lowering preserve the same semantics. Empty/strided slices,
  ellipsis, and fancy indexing remain intentionally unsupported pending the
  gather/scatter API.
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
  Fused GEMM and distributed backends remain future work.
- `Tensor.split_key(key, count)`, `Tensor.random_normal(shape, key, ...)`, and
  `Tensor.glorot_normal(shape, key)` provide stateless, deterministic host-side
  initialization. They do not retain global RNG state; generated parameters are
  uploaded once when a CUDA optimizer is created.
- On Apple silicon, the experimental `mlx` feature executes supported frozen
  Tensor IR plans as MLX arrays on `StreamOrDevice::gpu()`. Python exposes this
  through `TensorTraceGraph.compile_mlx()`, `TraceTensor.compile_mlx()`, and
  `TensorTraceResult.compile_mlx()`. The backend has CPU-parity coverage for
  elementwise operations, matmul, global reductions, reshape, transpose,
  concat, and broadcast. It is a primal execution backend, not a JIT: reverse
  graphs that contain unsupported operations such as `slice` or internal
  zero-padding return an explicit error rather
  than falling back to the host. Building the native MLX dependency requires
  Xcode's Metal Toolchain in addition to CMake:
  `xcodebuild -downloadComponent MetalToolchain`. The Python MLX wheel is not
  used by this Rust backend.
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
4. Add buffer planning, fused GEMM, and GPU-resident optimizer execution to
   eliminate per-step host transfers and allocations.

See [docs/jax_like_roadmap.md](docs/jax_like_roadmap.md) for the architecture
direction.
