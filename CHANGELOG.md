# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, minor releases may add to the public Python API and deprecate names;
deprecated names keep working until 1.0 (see
[CONTRIBUTING.md](CONTRIBUTING.md)).

## [Unreleased]

### Added

- `vmap` batches `fori_loop` and `scan` regions, and the JVP, VJP, and
  forward-over-reverse regions derived from them, on the CPU, CUDA, and
  MLX. A mapped capture, carry, or scanned operand batches the loop body
  with exactly those inputs mapped; a carry that depends on a mapped
  capture is mapped from the first iteration (the carry fixed point JAX
  computes), unmapped operands stay unbatched, and stacked scan outputs are
  `[B, length, ...]`. `vmap`, `jacobian` in both modes, and `hessian` now
  work through loops, nested loops, and loops inside `jit`, and `vmap` of
  `ode.odeint` works for every method, with `save` and `saveat`.
- `vmap` batches `while_loop` regions (and the forward-mode `while_loop`
  regions of `jvp` and forward `jacobian`) on the CPU, CUDA, and MLX, with
  JAX's semantics: a predicate that depends on a mapped value runs the loop
  while any example continues, and the body selects
  `where(pred, body(carry), carry)` per example, so each example stops at
  its own trip count and a finished example's carry and tangent stay bit
  for bit what an unbatched loop returns (`NaN` or `inf` that the body
  computes for it is discarded); a predicate that reads only unmapped
  values keeps one trip count for the batch. The batched predicate stays
  one scalar, so CUDA still reads back one flag per iteration.
- `vmap` of `linalg.cg`, `linalg.gmres`, and `newton` over the right-hand
  side, the initial guess, and `args`, composed with their implicit
  derivatives in both orders (`vmap(grad(...))` and `grad` of a loss over
  `vmap`): each example stops at its own tolerance, and `info=True` reports
  per-example `iterations`, `residual_norm`, and `success`. The
  reverse-mode `jacobian` and `hessian` of a `cg` or `gmres` solution, which
  batch the adjoint solve, work too.

### Changed

- The `UnsupportedOperationError` of `vmap` names only `cond` regions, the
  ones that still have no batching rule.

### Fixed

- A `fori_loop`, `scan`, or `while_loop` body whose result does not read
  the carry (it depends only on the operands or the index, or is a
  constant) no longer fails at trace time; as in JAX, it returns that value
  every iteration, and the carry gets a zero derivative.

## [0.4.0] - 2026-10-06

The v0.4 plan in `docs/jax_like_roadmap.md`: the jax.numpy functions a PINN
code base reaches for, stiff and implicit solvers, MLX factorizations,
device `Trainer` parity, and faster host-driven CUDA loops, plus native
`max`/`min` reductions. All v0.3 call forms keep working; gradients of
`max`/`min` at ties and a few float32 device optimizer results change as
listed under Changed.

### Added

- PyPI distributions built by the `Wheels` workflow: `quabla` (Linux x86_64
  wheels with the CPU and CUDA + NCCL backends, macOS arm64 CPU wheels, and
  the source distribution) and `quabla-mlx` (macOS arm64 wheels with the CPU
  and MLX backends). Both provide the `quabla` package; install one per
  environment. Wheels cover CPython 3.10 to 3.14. Version tags publish to
  TestPyPI and then PyPI through trusted publishing.
- `linalg.cg` (symmetric positive-definite operators, optional
  preconditioner) and `linalg.gmres` (restarted GMRES with right
  preconditioning, CGS2 Arnoldi, and Givens rotations): matrix-free solvers
  for `matvec(x, *args)`, run eagerly or as one `while_loop` region under
  `jit`. Gradients with respect to `b` and the arrays in `args` follow the
  implicit function theorem (one adjoint solve) instead of differentiating
  the iterations; reverse mode composes twice. `info=True` returns the
  iteration count, true residual norm, and success flag; an eager solve that
  does not converge raises `RuntimeError` otherwise.
- `quabla.newton(f, x0, *, args=(), tol=None, maxiter=50, info=False)`:
  Newton's method for `f(x, *args) = 0` with a dense Jacobian, Householder
  QR steps, and an Armijo backtracking line search on `||f||^2`, run as one
  `while_loop` region under `jit`. Gradients with respect to the arrays in
  `args` follow the implicit function theorem at the root; reverse mode
  composes twice. An exactly singular Jacobian stops with failure instead of
  raising; `info=True` returns the iteration count, residual norm, and
  success flag.
- `quabla.ode.odeint(method="rosenbrock23")`: the adaptive, L-stable
  Rosenbrock 2(3) method of Shampine and Reichelt (MATLAB's ode23s) for
  stiff problems. It forms `df/dy` and `df/dt` by one batched forward-mode
  pass per step and solves with `W = I - h d J` by LU; step control,
  `rtol`/`atol`, `max_steps`, `info`, and the eager and `jit` behavior match
  `dopri5`. It runs on every backend and is
  reverse-mode differentiable, through the Jacobian, with respect to the
  initial state, time span, and parameters.
- `odeint(saveat=ts)` for every method: the states at the times `ts`,
  stacked on axis zero, from each method's continuous extension (Shampine's
  fourth-order interpolant for `dopri5`, the method's own for
  `rosenbrock23`, cubic Hermite for the fixed-step methods), written by
  masked updates inside the loop so adaptive steps are not shortened, and
  differentiable.
- jax.numpy-style functions composed from existing ops, so they
  differentiate and run under `jit` and `vmap` on every device: `flip`,
  `roll`, `pad` (`constant`, `edge`, `reflect`, `symmetric`, `wrap`),
  `tile`, `repeat`, `moveaxis`, `swapaxes`, `ravel`, `diag`, `diagonal`,
  `trace`, `outer`, `dot`, `tensordot`, `kron`, `cross`, `diff`,
  `trapezoid`, `interp`, `polyval`, `logaddexp`, `hypot`, `exp2`, `isinf`,
  and `nan_to_num`. `logaddexp` and `hypot` neither overflow nor underflow
  and have finite gradients at ties and at the origin; `interp` evaluates
  every interval (no `searchsorted` without integer arrays) and is
  differentiable in `x`, `xp`, and `fp`.
- `quabla.linalg.norm` with NumPy's vector and matrix `ord` values and
  `axis`/`keepdims` (the vector 2-norm and `"fro"` reuse the scaled `norm`
  method, and general `p` norms are scaled likewise),
  `quabla.linalg.matrix_power`, and `quabla.linalg.pinv` (NumPy's
  `rtol=None` cutoff).
- MLX runs `solve` and every `quabla.linalg` function (`solve_triangular`,
  `cho_solve`, `slogdet`, `det`, `inv`, `eigh`, `qr`, `svd`, `lstsq`) and
  their derivatives instead of raising `UnsupportedOperationError`. MLX
  0.32.2 has LU, `eigh`, QR, and SVD only on its CPU stream, so LAPACK
  factors there in `float32` and the CPU backend's conventions (eigenvalue
  order, vector signs, the sign of `R`, the Householder completion of
  complete and full bases, NaN for non-finite matrices) are applied on the
  GPU stream; `solve` substitutes with the LU factors in a Metal kernel and
  raises on an exactly zero pivot like the CPU, after reading back one flag.
- Native elementwise `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`,
  `arcsinh`, `arccosh`, `arctanh`, `log2`, `log10`, `cbrt`, `floor`, `ceil`,
  and `round` (half to even) as module functions, `Tensor` and
  `TraceTensor` methods, and one IR op on the CPU, CUDA (float32 and
  `precision="float64"`), and MLX, with derivatives of every order under
  `grad`, `jvp`, `vmap`, and `hessian`. Out-of-domain inputs give NaN, not
  an error. The derivative forms avoid cancellation near the domain edges
  and overflow at large magnitudes (`arcsinh'(1e200) == 1e-200`).
  `quabla.round` is not in `__all__`, like `abs`.
- `fmod(x1, x2)` (C `fmod`, sign of `x1`, exact) as a native broadcasting
  op on every backend with partials `1` and `-trunc(x1 / x2)`, and
  `mod`/`remainder` with NumPy's floor-mod semantics (sign of `x2`),
  composed from `fmod` as NumPy does.
- `examples/benchmark_host_loop_cuda.py` measures the per-iteration cost of
  host-driven CUDA loops and compares two builds bit for bit.
- Device `optim.Trainer` parity: CUDA and MLX accept `AdamW`, `SGD`, and
  `clip_norm` (previously `UnsupportedOperationError`), with the CPU's update
  definitions and operation order, including AdamW's decay of parameters the
  loss ignores. Global-norm clipping is computed on the device without
  reading gradients back: `m * sqrt(sum((g / m)^2))` with `m = max |g|` over
  all parameters (accumulated in float64 on CUDA, in float32 on MLX; every
  term is at most one, so neither overflows), then the factor
  `min(1, clip_norm / norm)` scales every gradient before the moment updates;
  a zero norm leaves gradients unchanged and a non-finite norm makes every
  parameter NaN, as on the CPU. Over eight steps of a small PINN, float32
  device parameters agree with the CPU trainer to about `6e-8`.
- `Trainer(..., precision="float64")`, validated as in `jit`: on CUDA a
  float64 loss trains natively in double with float64 parameters and moments
  on the device (agreeing with the CPU trainer to about `1e-16` over eight
  PINN steps); a no-op on the CPU; `UnsupportedOperationError` on MLX.

### Changed

- The device Adam updates behind `Trainer` and the deprecated
  `cuda_adam_loss_optimizer`, `cuda_adam_vjp_optimizer`, and
  `mlx_adam_loss_optimizer` follow the CPU's operation order and form
  `1 - beta` and the bias corrections in float64, rounding them once.
  `Trainer` passes its hyperparameters in float64 (the deprecated factories
  still take float32), so device runs use the CPU's exact betas instead of
  their float32 roundings. Device Adam trajectories change at the float32
  rounding level. The `learning_rate` attribute of a CUDA executor holds the
  float64 value that was set rather than its float32 rounding.
- `quabla.trace` is the array `trace(a, offset=0, axis1=0, axis2=1)`. The
  v0.1 call form `trace(function, input_specs)` keeps working and warns on
  the call, as `grad` and `jit` do, instead of on attribute access.
- Host-driven CUDA loops (bodies that are not elementwise) run each
  iteration as one CUDA graph launch: after two eager iterations a region
  whose program only launches NVRTC kernels, device copies and cuBLAS
  products is recorded once, and later iterations retarget its input and
  output copies. Loop-invariant captures are bound once per loop instead of
  every iteration, and the loop index is bound without an intermediate copy.
  On a GTX 1660 SUPER a small `fori_loop` or `scan` iteration drops from
  about 120-180 µs to about 30-50 µs, a `while_loop` iteration from about
  220 µs to about 100 µs, and reverse-pass iterations about fourfold; results
  are bit-identical to the previous implementation.
- The gradient of a `max` or `min` reduction at a tie is split equally among
  the tied entries (JAX's rule), including ties across several reduced axes,
  instead of going entirely to the last one. This reaches every function
  built on the reductions, such as `linalg.norm` with `ord=inf` and the
  matrix 1- and inf-norms. A NaN maximum has a NaN gradient.
- `max` and `min` order the sign of zero as IEEE 754-2019 `maximum` and
  `minimum` do (and NumPy on arm64): the `max` of `-0.0` and `+0.0` is
  `+0.0` and the `min` is `-0.0`, eagerly and under `jit` on every backend,
  instead of whichever zero came last.
- Traced `max` and `min` reductions are one native IR op instead of a slice,
  a reshape and an elementwise `maximum` per reduced element, so a staged
  program has a constant node count in the reduced size: a 600-element `jit`
  `max` has 2 nodes instead of 3,597 (its gradient 11 instead of 9,590) and
  takes 2.3 µs per call on the CPU instead of 470 µs, 0.18 ms instead of
  14 ms on MLX, and 0.14 ms instead of 40 ms on a GTX 1660 SUPER. The scaled
  `norm` gains as much. CUDA reduces each line with one thread, or one
  block of 256 threads for lines of 256 or more entries, with no atomics, so
  results are run-to-run identical and equal to the CPU bit for bit in
  float32 and `precision="float64"`; MLX restores the sign of a zero result,
  which its own reduction does not keep.

### Fixed

- Hessian-vector products (forward over reverse) through a `scan` whose body
  is not elementwise, or whose step output has a different lane count than
  its carry, run on CUDA as host-driven loops instead of being rejected.
- A CUDA request on a host without the NVIDIA driver library, or a
  data-parallel request without NCCL, raises an error naming the missing
  library instead of a Rust panic.

## [0.3.0] - 2026-10-06

The JAX-ecosystem gaps of the v0.3 plan in `docs/jax_like_roadmap.md`:
optimizers, linear algebra, ODE integration, custom derivatives, pytrees,
random numbers, numerically stable compositions, packaging, and CUDA loop and
`float64` support. All v0.2 call forms keep working.

### Changed

- A transformed function that needs more than `max_traces` traces (8 by
  default) evicts its least recently used trace and emits the new
  `quabla.RetraceWarning` instead of raising `RetraceLimitError`, so a
  training loop whose batch shape changes keeps running. Trace caches,
  including `distributed.value_and_grad`, no longer raise
  `RetraceLimitError`; the class remains for existing handlers.
- `norm` computes `m * sqrt(sum((x / m)**2))` with `m = max|x|`, so it no
  longer overflows or underflows: the `float32` norm of `[1e20, 1e20]` is
  1.414e20 with gradient 0.707 instead of `inf` with gradient 0. Results and
  gradients can change in the last bits.
- `quabla.tree.flatten` treats NamedTuples as nodes (they were leaves) and
  accepts any mutually orderable dict keys, such as ints (they were
  rejected); dict entries are visited in sorted key order.
- Indexing accepts `None`, `...`, and strided slices such as `x[::2]` and
  `x[::-1]`, which used to be rejected.
- `solve` accepts stacks of matrices with leading batch axes on the CPU and
  CUDA, and `vmap` batches it (it used to reject a mapped `solve`).
- The `repr` of an eager `Tensor` shows its values NumPy-style and always
  names the dtype, for example `Tensor([1., 2.], dtype=float32)`; a
  `TraceTensor` repr adds its dtype.
- The package version has one source, `crates/quabla-python/Cargo.toml`;
  `pyproject.toml` reads it dynamically.
- CUDA runs `fori_loop` and `scan` bodies that are not purely elementwise
  (slices, concatenations, `cond`, array constants, broadcast operands in the
  VJP) as host-driven loops over per-node device programs instead of
  rejecting them, so adaptive `odeint` and reverse-mode `odeint` run on CUDA.
  Such loops cost a kernel launch sequence per iteration.

### Added

- Optimizers: `optim.AdamW` with decoupled weight decay; learning-rate
  schedules (`constant`, `exponential_decay`, `cosine_decay`,
  `warmup_cosine_decay`, `piecewise_constant`, or any callable of the step)
  for Adam, AdamW, SGD, and the CPU and device `Trainer`;
  `optim.clip_by_global_norm` and a `clip_norm=` option; and `optim.LBFGS`,
  a full-batch L-BFGS with a strong Wolfe line search for the usual
  "Adam, then L-BFGS" PINN schedule (Rosenbrock from (-1.2, 1) converges in
  38 iterations).
- Native elementwise and scan operations on CPU, CUDA, and MLX with forward,
  reverse, and second-order derivatives: `expm1`, `erf`, `atan2`,
  `stop_gradient`, and `cumsum(axis, reverse)`.
- Numerically stable compositions: `softmax`, `log_softmax`, `logsumexp`
  (shifted by `stop_gradient(max)`), two-pass `var` and `std`, `silu`,
  `gelu` (tanh approximation or exact `erf` form), `clip`, `sign`,
  `square`, and `reciprocal`.
- Shape helpers: `reshape(-1)` and varargs, `.T`, `squeeze`,
  `expand_dims`, `split`, `meshgrid`, `zeros_like`, `ones_like`,
  `full_like`, and NumPy-style 1-D operands for `quabla.matmul`.
- Pytrees: NamedTuples, dataclasses registered with
  `tree.register_dataclass`, and custom nodes registered with
  `tree.register`, in every transform, optimizer, and `Trainer`; plus
  `tree.leaves`, `tree.structure`, and `tree.flatten_with_path`.
- `quabla.random`: explicit keys (`key`, `split`, `fold_in`) with
  `uniform`, `normal`, `bernoulli`, and the `glorot_normal`,
  `glorot_uniform`, and `he_normal` initializers.
- `jit(static_argnames=...)`, keyword arguments to jitted functions, and
  keyword-only decorator forms of `jit`, `grad`, and `value_and_grad`.
- `quabla.ode.odeint`: fixed-step RK4, Heun, and Euler integration over a
  single loop region, and adaptive Dormand-Prince 5(4) (`method="dopri5"`,
  `rtol`, `atol`, `max_steps`) with PI step-size control run as a bounded
  loop, all differentiable with respect to the initial state, time span,
  and parameters.
- `quabla.while_loop` with a traced predicate on the CPU, CUDA, and MLX,
  with forward-mode derivatives; reverse mode and `vmap` raise and point to a
  bounded `fori_loop`, as in JAX.
- `quabla.linalg`: `solve`, `solve_triangular`, `cholesky`, `cho_solve`,
  `slogdet`, `det`, `inv`, `eigh` (cyclic Jacobi on the CPU, cuSOLVER
  `syevd` on CUDA), `qr` (Householder; cuSOLVER `geqrf`/`orgqr`), `svd`
  (one-sided Jacobi; cuSOLVER `gesvdj`), and `lstsq` (through `qr`), with
  derivatives of every order written without explicit inverses.
- `prod`, exact with zeros and differentiated without division, and a
  native `erfc`; exact `gelu` uses `erfc` so its negative tail keeps
  relative accuracy.
- `jit(..., device="cuda", precision="float64")` runs `float64` programs as
  native double on CUDA (cuBLAS `dgemm`, double cuSOLVER and NVRTC kernels),
  matching the CPU `float64` reference to about `1e-14`. The default keeps
  `float32` execution because consumer GPUs run `float64` much slower (an
  MLP step is about 9x slower on a GTX 1660 SUPER).
- `quabla.custom_vjp`, `quabla.custom_jvp`, and `quabla.checkpoint`
  (`remat`) with JAX's interfaces, under `jit`, `vmap`, and on every device.
- `quabla.__version__`, `quabla.save`/`quabla.load` for pytrees of arrays in
  a versioned format that is never unpickled, a `_quabla.pyi` type stub with
  `py.typed`, and a wheel workflow that builds CPU (Linux) and MLX (macOS)
  wheels as artifacts without publishing them.

## [0.2.3] - 2026-10-06

### Changed

- The CPU executor and eager tensors return IEEE 754 values for `log` of a
  non-positive value and for division by zero (`±inf`, `NaN`), as CUDA and
  MLX already did, instead of raising. A guard such as
  `where(x > 0, log(x), 0)` now works under CPU `jit`, which evaluates both
  branches. The legacy 2D `Matrix` API keeps its checked errors.
- Division derivatives no longer square the denominator: `d(l/r)` is
  `(dl - (l/r) dr) / r`, and the native Cholesky jets use the same rule.
  `float32` gradients with |denominator| below about 1e-19 or above 1e19 no
  longer underflow, overflow, or fail (`d(x/y)/dy` at `x = y = 1e-25` is
  `-1e25`, not an error or `-inf`). Division derivatives can change in the
  last bits.
- The CPU runs purely elementwise plans node by node instead of through a
  per-element interpreter: `jit(lambda x, y: x + y)` over 2^20 `float32`
  elements takes 19 ms instead of 76 ms, the same as eager. Results are
  unchanged.
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

### Added

- `quabla.log1p`, `Tensor.log1p`, and `TraceTensor.log1p`: a native
  elementwise `ln(1 + x)` with derivative `1 / (1 + x)` that follows IEEE
  semantics instead of raising (`log1p(-1)` is `-inf`, `x < -1` gives
  `NaN`). It lowers to `log1pf` on CUDA and `log1p` on MLX.

### Fixed

- Device `Trainer`s (MLX and CUDA) accept parameters the loss does not
  depend on, such as an output bias that a second derivative removes, and
  leave them unchanged like the CPU `Trainer`; they used to fail with
  "... is not a plan input" or "CUDA plan has no input".
- Traced `softplus` adds `log1p(exp(-|x|))` instead of `log(1 + exp(-|x|))`,
  which rounded the correction away in `float32`: `jit(softplus)(-20)` now
  returns 2.06e-9 instead of 0. Eager `softplus` evaluates the same per-op
  rounded expression, so eager and CPU `jit` agree bitwise.
- `sigmoid` is `where(x > 0, 1 / (1 + z), z / (1 + z))` with
  `z = exp(-|x|)`. The textbook form overflowed `exp(-x)` for large negative
  `x`, so its `float32` gradient was `NaN` there on CPU and MLX. Results
  where the old form was finite change by at most one ulp; eager and CPU
  `jit` agree bitwise.
- `maximum`, `minimum`, `relu`, and the `max`/`min` reductions propagate
  `NaN` from any operand or position, like NumPy and JAX, eagerly and under
  `jit`; they used to drop it depending on the operand order. Results and
  gradient routing for `NaN`-free inputs, ties included, are unchanged.

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

[Unreleased]: https://github.com/latteine1217/quabla/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/latteine1217/quabla/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/latteine1217/quabla/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/latteine1217/quabla/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/latteine1217/quabla/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/latteine1217/quabla/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/latteine1217/quabla/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/latteine1217/quabla/releases/tag/v0.1.0
