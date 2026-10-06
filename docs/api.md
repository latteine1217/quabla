# Quabla API Reference

This document is the reference companion to the [README](../README.md). It
describes the Python API on `main` in two layers:

- The **core API** released in v0.2.0 (slices S0-S9 of
  [api_v0_2_design.md](api_v0_2_design.md)): arrays with NumPy interop,
  module-level math, and the JAX-style function transforms `grad`,
  `value_and_grad`, `jvp`, `vjp`, `jacobian`, `hessian`, `vmap`, and `jit`.
  New code should start here. Device execution uses explicit
  `jit(device="cpu" | "cuda:N" | "mlx")`.
- The **v0.1 layer**, released in v0.1.0 and kept unchanged: the rank-N
  compiler facade (`Compiler`, `Program`, `Executable`), which is the
  advanced layer for explicit compilation; the `trace_tensor` and `tensor_*_fn`
  helpers with their device variants and device optimizers; the control-flow
  builders; and the legacy 2D `Matrix` API. The design deprecates most
  `tensor_*_fn` helpers in favour of the core API from v0.2. Migrated top-level
  names warn once, and all old call forms keep working through 0.x.

Paths in code spans (for example `examples/pinn_poisson.py`) are relative to
the repository root. Per-feature status and validation records are in
[jax_like_roadmap.md](jax_like_roadmap.md). The examples below use
`import quabla as qb`.

## Contents

- [Core API](#core-api)
  - [Arrays and NumPy](#arrays-and-numpy)
  - [Function Transforms](#function-transforms)
  - [Custom Differentiation Rules](#custom-differentiation-rules)
  - [Pytrees](#pytrees)
  - [Random Numbers](#random-numbers)
  - [Saving and Loading](#saving-and-loading)
  - [Version and Type Stubs](#version-and-type-stubs)
- [Device Execution](#device-execution)
- [Compiler Facade](#compiler-facade)
- [v0.1 API Reference](#v01-api-reference)
  - [Rank-N Tensor API](#rank-n-tensor-api)
  - [Legacy 2D API](#legacy-2d-api)
  - [Rust Core Crate](#rust-core-crate)
- [Known Limitations](#known-limitations)

## Core API

### Arrays and NumPy

```python
qb.array(obj, dtype=None)       # always a new object
qb.asarray(obj, dtype=None)     # returns obj unchanged if already a Tensor of dtype
qb.zeros(shape, dtype=None), qb.ones(...), qb.full(shape, fill_value, ...)
qb.arange(start, stop=None, step=1.0, dtype=None), qb.linspace(start, stop, num, ...)
qb.eye(n, m=None, dtype=None)
```

- `array`/`asarray` accept Python scalars, rectangular nested lists and
  tuples, NumPy arrays and scalars, objects with the buffer protocol or
  `__array__`, and Quabla arrays. The result is an eager `Tensor`; imported
  data is always copied, so a tensor never aliases a NumPy array.
- dtype inference: NumPy `float64`, `float32`, and `bool_` keep their dtype;
  NumPy integers become `float64` (there is no integer dtype) and `float16`
  raises `TypeError`. Python numbers and lists become `float64`, or `bool_`
  when every element is a Python `bool`. Pass `dtype=qb.float32` to convert.
- Export: `Tensor.numpy()`, `tolist()`, `item()`, `float(t)`, `np.asarray(t)`
  (through `__array__`), and the buffer protocol. `to_flat_list()` keeps
  working. `qb.Array` is an abstract base class that `Tensor`, `TensorView`,
  and `TraceTensor` are registered with.
- NumPy is optional: importing `quabla` never imports it.
- `repr` of an eager `Tensor` shows its values, following NumPy's default
  print options, and always the dtype:
  `Tensor([1., 2.], dtype=float32)`, `Tensor([ True, False], dtype=bool)`.
  Floats print with at most 8 fractional digits and aligned decimal points,
  and switch to scientific notation when the largest finite magnitude is at
  least `1e8`, the smallest nonzero one is below `1e-4`, or their ratio
  exceeds `1e3`; rows wrap at 75 columns. A tensor with more than 1000
  elements is summarized to the first and last 3 entries of every longer
  axis around `...`, and only those entries are read. A `TraceTensor` has no
  values, so its `repr` shows its node, shape, and dtype only:
  `TraceTensor(node_id=3, shape=[2], dtype=float32)`.
- Module-level functions call the method of the same name on a `Tensor` or
  `TraceTensor`, so eager and traced code share one spelling (`qb.sin(x)` is
  `x.sin()`): `sin`, `cos`, `tan`, `tanh`, `sinh`, `cosh`, `arcsin`,
  `arccos`, `arctan`, `arcsinh`, `arccosh`, `arctanh`, `exp`, `expm1`,
  `log`, `log1p`, `log2`, `log10`, `erf`, `erfc`, `sqrt`, `cbrt`, `floor`,
  `ceil`, `round`, `relu`, `sigmoid`, `softplus`, `stop_gradient`, `sum`,
  `mean`, `prod`, `max`, `min`, `any`, `all`, `norm`, `cumsum`, `matmul`,
  `transpose`,
  `reshape`, `broadcast_to`, `astype`, `maximum`, `minimum`, `atan2`,
  `fmod`, `power`, `solve` (see `qb.linalg` below), `cholesky`, `tril`, and
  `triu`, next to the v0.1
  functions `where`, `concat`, `stack`, `einsum`, and the comparison and
  logical functions. Other operands (numbers, lists, NumPy arrays) go
  through `asarray`; a Python number stays a weak scalar. `abs`, `round`,
  `sum`, `max`, `min`, `any`, and `all` are attributes of `quabla` but not
  in `__all__`, so `from quabla import *` leaves the builtins alone.
- `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `arcsinh`, `arccosh`,
  `arctanh`, `log2`, `log10`, `cbrt`, `floor`, `ceil`, and `round` are one
  native IR op (`UnaryMath`) evaluated with each platform's own function:
  Rust std on the CPU (the musl `asinh`, `acosh`, and `atanh` of the `libm`
  crate, since std's `acosh` loses accuracy near 1), the CUDA math library
  (`tanf`, `asinf`, ..., `rintf`, and their double forms under
  `precision="float64"`), and the MLX op of the same name; MLX has no
  `cbrt`, so it takes `|x|^(1/3)` from `power` and one Newton step, within
  about one float32 ulp. `round` rounds halfway cases to even, as NumPy and
  JAX do. Values follow IEEE semantics and never raise: outside a domain
  the result is NaN (`arcsin(2)`, `log2(-1)`, `arccosh(0.5)`), at a pole an
  infinity (`arctanh(1)`, `log10(0)`). Derivatives are built from IR ops,
  so every order is again a graph on every backend: `tan' = 1 + tan(x)^2`
  and `cbrt' = 1 / (3 cbrt(x)^2)` reuse the computed value;
  `arcsin' = 1 / sqrt((1 - x)(1 + x))` (negated for `arccos`) and
  `arctanh' = 1 / ((1 - x)(1 + x))` use the factored form, which stays
  accurate next to `|x| = 1`; `arccosh' = 1 / (sqrt(x - 1) sqrt(x + 1))`,
  which neither loses accuracy near 1 nor overflows for large `x`;
  `arcsinh' = 1 / sqrt(x^2 + 1)` is formed as `r / sqrt(1 + u^2)` with
  `r = 1 / max(|x|, 1)` and `u = r` for `|x| > 1` (else `x`), so it does
  not overflow (`1e-200` at `x = 1e200`) and is zero at `+-inf`;
  `arctan' = 1 / (1 + x^2)`, `sinh' = cosh`, `cosh' = sinh`,
  `log2' = 1 / (x ln 2)`, `log10' = 1 / (x ln 10)`; `floor`, `ceil`, and
  `round` have a zero derivative of every order. Outside a domain a
  derivative is NaN or, where the formula stays finite (`arctanh`, `log2`
  of a negative `x`), the value of the formula, as for `log`. On MLX,
  float32 subnormal inputs count as zeros (Metal flushes them), as for
  MLX's other ops.
- `fmod(x1, x2)` is C `fmod`, NumPy's `fmod`: the remainder of the quotient
  truncated toward zero, exact, with the sign of `x1`; NaN for `x2 == 0`
  or an infinite `x1`, and `x1` for an infinite `x2`. It is a native
  broadcasting op (`fmodf` on CUDA; on MLX, whose `remainder` is the
  floor-mod, the exact `remainder(|x1|, |x2|)` with the sign of `x1`). Its
  partials are `1` and `-trunc(x1 / x2)`, where the quotient is that of the
  computed remainder, `round((x1 - fmod(x1, x2)) / x2)`, so it is exact and
  consistent with the value at quotients that round to an integer.
  `mod(x1, x2)` (also `remainder`) is NumPy's floor-mod, with the sign of
  `x2`, composed from `fmod` the way NumPy computes it: `r + x2` where
  `r = fmod(x1, x2)` is nonzero with a sign other than that of `x2`, and a
  zero result signed like `x2`. It never forms `x1 - floor(x1 / x2) x2`,
  which loses small remainders to rounding; its partials are `1` and
  `-floor(x1 / x2)`.
- `prod(x, axis=None, keepdims=False)` multiplies the entries over `axis`
  (an int, a sequence of ints, or every axis) as a pairwise tree of
  multiplications, each rounded to the dtype: an extent `n` rounds each
  result at most `ceil(log2 n)` times, and the eager value and the traced
  value on every backend agree bitwise. Nothing is divided, so the
  derivative is exact with zeros: `d prod / d x_i` is the product of the
  other entries, which is nonzero only at the zero entry when there is
  exactly one zero and zero everywhere with two or more, and the Hessian
  and every higher derivative are again products of the remaining entries
  (JAX's `reduce_prod` derivative uses the same tree). Intermediate
  products can overflow or underflow in an order that differs from NumPy's
  sequential product. `bool` arrays are rejected.
- Numerically stable compositions, built from the methods above, so they
  work eagerly, under `jit`/`grad`/`vmap`, and on every device:
  - `softmax(x, axis=-1)` and `log_softmax(x, axis=-1)` subtract the maximum
    along `axis` before `exp`; `log_softmax` is
    `(x - m) - log(sum(exp(x - m)))`. `softmax([1000, 0])` is `[1, 0]` and
    `log_softmax` gives `[0, -1000]`, with finite gradients, in `float32` too.
    A slice holding `+inf` or only `-inf` gives NaN, as in JAX.
  - `logsumexp(x, axis=None, keepdims=False)` is `m + log(sum(exp(x - m)))`
    with the shift `m` set to zero where the maximum is not finite, as in
    JAX: an all-`-inf` slice gives `-inf`, a slice holding `+inf` gives
    `+inf`, and NaN propagates. Its gradient is `softmax`.
  - `var(x, axis=None, keepdims=False, ddof=0)` and `std(...)` take two
    passes (the mean, then the mean squared deviation), never
    `E[x^2] - E[x]^2`, so `var(1e8 + [1, 2, 3])` is exactly `2/3` in
    `float64`. `n <= ddof` divides by zero (inf or NaN). `std` has
    derivative zero at zero variance (the `sqrt` convention; JAX gives NaN).
  - `silu(x)` is `x * sigmoid(x)`; `gelu(x, approximate=True)` is the tanh
    approximation `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`,
    evaluated as `x * sigmoid(2u)` so the negative tail keeps its relative
    accuracy. `approximate=False` is the exact `0.5 x (1 + erf(x / sqrt(2)))`,
    evaluated as `0.5 x erfc(-x / sqrt(2))`, so the negative tail keeps its
    relative accuracy where `1 + erf` would cancel to zero; what remains is
    the conditioning of the tail itself (in `float32`, a relative error of
    about `x^2 eps` from the rounding of `x`).
  - `clip(x, lo=None, hi=None)` is `minimum(maximum(x, lo), hi)`: NaN
    propagates, and at a bound the derivative goes to the bound, so `x`
    gets zero there. `sign(x)` is -1, 0, or +1 (NaN for NaN) with derivative
    zero. `square(x)` is `x * x` and `reciprocal(x)` is `1 / x`.
  - The max shift of `softmax`, `log_softmax`, and `logsumexp` is
    differentiated through; its contribution cancels to rounding.
- Shape helpers, all reshapes, slices, and broadcasts that differentiate and
  work under `vmap`:
  - `x.reshape(3, 2)`, `x.reshape((3, 2))`, and `quabla.reshape(x, shape)`
    accept one `-1` extent, inferred from the size. `x.T` reverses the axes
    (`x.transpose()`). Indexing takes `None`, `...`, and strided slices
    (`x[None]`, `x[..., 0]`, `x[:, None]`, `x[::-1]`).
  - `squeeze(x, axis=None)`, `expand_dims(x, axis)`,
    `split(x, indices_or_sections, axis=0)` (a list; pieces may not be
    empty), and `meshgrid(*xs, indexing="xy")` (a list; `"ij"` keeps the
    input order) follow NumPy.
  - `zeros_like(x, dtype=None)`, `ones_like(...)`, and
    `full_like(x, fill_value, dtype=None)` return a `Tensor` of the shape and
    dtype of `x`; for a traced `x` it is a constant of the trace.
  - `quabla.matmul` follows NumPy for rank-1 operands: vector @ vector is a
    scalar, matrix @ vector and vector @ matrix drop the vector's unit axis.
    The `@` operator keeps requiring rank-2 or higher operands.
- jax.numpy-style functions with NumPy signatures and semantics, composed
  from reshapes, transposes, broadcasts, slices, static-index gathers,
  `concat`, `where`, and `matmul`, so they differentiate and run under
  `jit` and `vmap` on every device. Shifts, pad widths, repeats, and axes
  are static Python ints (there are no integer arrays), and no result may
  be empty:
  - Shapes: `flip`, `roll`, `pad(x, pad_width, mode="constant",
    constant_values=0)` with modes `constant`, `edge`, `reflect`,
    `symmetric`, and `wrap`, `tile`, `repeat` (an int, or one count per
    entry), `moveaxis`, `swapaxes`, `ravel`, `diag(v, k=0)` (1-D to 2-D and
    back), `diagonal(x, offset=0, axis1=0, axis2=1)`, and `trace` with the
    same arguments. `trace(function, input_specs)`, the v0.1 call form, is
    still dispatched to `quabla.legacy.trace` with its deprecation warning.
  - Products: `outer`, `dot` (NumPy's rules: a `tensordot` over the last
    axis of `a` and the second-to-last of `b` beyond rank 2), `tensordot(a,
    b, axes=2)`, `kron`, and `cross(a, b, axisa=-1, axisb=-1, axisc=-1,
    axis=None)` for 3-vectors and 2-vectors.
  - Calculus: `diff(x, n=1, axis=-1, prepend=None, append=None)`,
    `trapezoid(y, x=None, dx=1.0, axis=-1)`, `polyval(p, x)` by Horner's
    rule (from `p[0]`, so an infinite `x` gives the polynomial's limit where
    NumPy gives NaN), and `interp(x, xp, fp, left=None, right=None)` for a
    strictly increasing `xp` (checked when `xp` is eager). Without
    `searchsorted`, `interp` evaluates every interval, O(`x.size *
    len(xp)`) work and memory, and is exact at the knots and differentiable
    in `x`, `xp`, and `fp`; its derivative in `x` at a knot is the slope to
    the right (at `xp[-1]`, the last slope), as in JAX.
  - Elementwise: `logaddexp(a, b)` as `hi + log1p(exp(lo - hi))` (no
    overflow, gradient split evenly at ties, `x1 + x2` where both are
    infinite of one sign); `hypot(a, b)` scaled by `max(|a|, |b|)`, so it
    neither overflows nor underflows in `float32`, with gradient
    `(a, b) / hypot` and zero at the origin; `exp2(x)` as `2 ** x`, exact
    for integer `x`; `isinf(x)`; and `nan_to_num(x, copy=True, nan=0.0,
    posinf=None, neginf=None)`, whose defaults are the dtype's finite
    limits.
- Operators: `-x` and `x ** y` work on traced values as on eager ones;
  `x ** y` with a non-integer or tensor exponent is the differentiable
  elementwise `pow` op (`qb.power`), described under
  [Rank-N Tensor API](#rank-n-tensor-api).
- The dtype rules, bool masks, and the operation set are the v0.1 rules of
  the [Rank-N Tensor API](#rank-n-tensor-api).

### Function Transforms

The v0.2 transforms ([api_v0_2_design.md](api_v0_2_design.md), sections
3.3-3.5; see [Device Execution](#device-execution)) take
pytree arguments (see [Pytrees](#pytrees)) of arrays and Python scalars,
and need no input specs:

```python
qb.grad(fun, argnums=0, has_aux=False)            # -> grads, or (grads, aux)
qb.value_and_grad(fun, argnums=0, has_aux=False)  # -> (value, grads)
qb.jvp(fun, primals, tangents)                    # -> (out, tangent_out)
qb.vjp(fun, *primals, has_aux=False)              # -> (out, vjp_fun[, aux])
qb.jacobian(fun, argnums=0)                       # blocks [*out.shape, *in.shape]
qb.hessian(fun, argnums=0)                        # blocks [*in.shape, *in.shape]
qb.vmap(fun, in_axes=0, out_axes=0)               # -> batched fun
qb.jit(fun, device=None, static_argnums=(), max_traces=8, static_argnames=(),
       precision=None)
```

`jit`, `grad`, and `value_and_grad` called with keywords only return a
decorator: `@qb.jit(device="mlx", static_argnums=1)`, `@qb.grad(argnums=1)`.

- Every transformed function traces and compiles one CPU program on its
  first call per signature (pytree structure, array shapes and dtypes, and
  static values) and reuses it afterwards. It keeps at most `max_traces`
  traces (8 by default); a new signature beyond that evicts the least
  recently used trace with a `quabla.RetraceWarning`, so a training loop
  whose batch shape varies keeps running but retraces. Python scalars are static
  weak constants, so they keep `float32` programs in `float32`, except in
  differentiated positions, where they become `float64` arrays. Arrays and
  Python values read from closures are fixed at trace time; tracers of an
  enclosing trace read from closures are not (see "Closures over tracers").
- Gradients mirror the pytree of the selected arguments and keep their
  dtypes; `bool_` leaves get `None`. Transforms compose when passed to each
  other directly (`grad(grad(f))`, `jvp(grad(f), (x,), (v,))`,
  `jit(value_and_grad(f))`) and when a transformed function is called
  inside a function that another transform traces: `grad`,
  `value_and_grad`, `jvp`, `vjp`, `vmap`, and `jit` stage their graph once
  per signature and inline it into the enclosing trace, so a derivative can
  be used inside a loss that is differentiated again:

  ```python
  def u(x, w):                                          # one point, scalar x
      return qb.sin(x * w)

  u_xx = qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None))  # d2u/dx2 per point

  def loss(w, x):                                       # x: [n] points
      return qb.mean((u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2)

  x = qb.linspace(0.05, 0.95, 8)
  value, grad_w = qb.jit(qb.value_and_grad(loss))(qb.array(2.5), x)
  ```

  Eager array arguments of such an inner call bind as constants. Each
  inner call adds its graph once, so per-point Python loops grow the graph
  linearly; `vmap` keeps it independent of the number of points.
  `jacobian` and `hessian` are forward mode vectorized with `vmap` over the
  input elements (as `jax.jacfwd`), so they compose too, for example
  `grad(lambda x: qb.sum(qb.hessian(f)(x)))` or `vmap(hessian(f))`.
  Second-order reverse mode through `fori`/`scan` regions is rejected with
  a `ValueError`.
- `vmap(fun, in_axes=0, out_axes=0)` vectorizes `fun`, which sees one
  example, with JAX semantics. `in_axes` is an int (the mapped axis,
  negative counts from the end), `None` (an argument shared by every
  example), or a tuple, list, or dict of them matching the arguments as a
  pytree prefix; all mapped axes must have one size, and Python scalars can
  only be unmapped. `out_axes` places the batch axis of each result leaf
  the same way: results that do not depend on a mapped argument are
  broadcast over the batch, and `None` returns them unbatched (a mapped
  result with `out_axes=None` is an error). `fun` is staged for one example
  and the graph is batched node by node, so `vmap` composes with every
  transform in both directions and with itself: `vmap(grad(f))`,
  `jit(vmap(f))`, `grad` of a loss over `vmap`, `jvp`/`vjp` of `vmap`, and
  `vmap(vmap(f))`. `vmap(grad(f, argnums=1), in_axes=(0, None))` returns
  one gradient per example, `[B, *w.shape]`, as in JAX, while
  `vjp(vmap(f, in_axes=(0, None)), x, w)` sums the unmapped gradient over
  the batch, as reverse mode must. A mapped `solve` broadcasts an unmapped
  operand over the batch. A `fori_loop` or `scan` region with a mapped
  operand batches its body region with exactly the mapped operands mapped,
  and so do the loop's JVP, VJP, and forward-over-reverse regions, so
  `vmap`, `jacobian`, and `hessian` work through loops, nested loops
  included. As in JAX, a carry that depends on a mapped capture is mapped
  from the first iteration, unmapped captures stay unbatched, and a scan's
  stacked outputs are `[B, length, ...]`; a reverse-mode loop region maps
  every input once any operand is mapped, since its gradients are per
  example. A `while_loop` batches as in JAX: when its predicate depends on a
  mapped value, the batched loop runs while any example's predicate holds
  and keeps every finished example's carry unchanged, so each example stops
  at its own trip count; otherwise the batch shares one trip count. A mapped
  `cond` raises `quabla.UnsupportedOperationError`.
- Closures and constants: an eager array (`Tensor` or `TensorView`,
  including a result such as `qb.sin(math.pi * 0.3)`) that meets a traced
  value becomes a constant of the graph. This covers arithmetic in either
  order, comparisons, `**`, `@`, `maximum`/`minimum`, `where`, `concat`,
  `stack`, `solve`, the logical ops, and eager arguments, tangents, and
  cotangents of an inner transformed call, so closures over module-level
  arrays work:

  ```python
  points = qb.linspace(0.0, 1.0, 8)
  f = lambda w: qb.sum(qb.sin(points * w))
  qb.grad(f)(qb.array(1.0))
  ```

  As with `jax.jit`, the value is fixed at trace time: rebinding `points`
  after the first call does not retrace (tensors are immutable, so
  rebinding is the only change possible); pass data that changes as an
  argument. A constant keeps its dtype (strong): a `float64` array with a
  `float32` traced value is a dtype error asking for `astype`, while Python
  scalars stay weak. Every captured array is stored in the graph, so large
  ones grow memory and compile time; compiled plans merge equal constants
  and fold elementwise ops on them. CUDA uploads each constant once per
  compiled plan (a loop body with a captured array runs as a host-driven
  region loop whose body program keeps it resident), and MLX once per
  constant.
- Closures over tracers: a function passed to a transform inside a traced
  function may close over tracers of that trace, or of any trace enclosing
  it, as in JAX:

  ```python
  def loss(params, x):                                  # x: [n] points
      u_xx = qb.vmap(qb.grad(qb.grad(lambda t: net(params, t))))
      return qb.mean((u_xx(x) - forcing(x)) ** 2)

  qb.jit(qb.value_and_grad(loss))(params, x)
  ```

  The inner transform treats a closed-over tracer as a constant: it
  differentiates only its explicit arguments, and `vmap` leaves the value
  unmapped. The enclosing transform differentiates through it, so the
  gradient above includes the dependence of `u_xx` on `params`; the result
  equals passing `params` as an unmapped argument
  (`vmap(grad(grad(net_x)), in_axes=(0, None))(x, params)`). This works for
  `grad`, `value_and_grad`, `jvp`, `vjp`, `vmap`, `jacobian`, `hessian`,
  and `jit`, with `has_aux`, with eager arguments of the inner call, next to
  captured arrays, and across several levels of nesting. A transformed
  function that captured tracers when it was staged is traced again on
  each call to find the tracers it captures this time, and reuses its
  transformed graph while their shapes and dtypes match, so a closure that
  refers to a different tracer on each call binds the right one. A tracer
  used after its trace ended (stored in a global or a container and used in
  a later trace), or inside a `tensor_*` helper or region trace, raises
  `quabla.TracerError`.
- Arguments are positional, except for `jit`: a keyword argument that
  names the next positional parameter of `fun` is bound to that position
  (`f(x, n=2)` is `f(x, 2)` and shares its trace), and any other keyword
  argument, including a keyword-only parameter, must be static.
  `static_argnames` and `static_argnums` imply each other through `fun`'s
  signature, as in JAX, so `jit(f, static_argnames="n")` makes `n` static
  whether it is passed by position or by keyword. Static keyword values
  must be hashable and are part of the trace cache key. A static keyword
  argument cannot be passed to a `jit` of another transform
  (`jit(grad(f))`); pass it positionally there.
- `quabla.grad` and `quabla.jit` still accept the legacy 2D forms
  `grad(fn, input_specs, values, output_cotangent)` and `jit(input_specs)`
  and dispatch them to the v0.1 functions below.
- A `TraceTensor` has no value while it is traced: Python control flow,
  `float()`, `int()`, `item()`, `tolist()`, `numpy()`, and `np.asarray` on
  it raise `quabla.TracerError`. A NumPy array is not captured: as the
  left operand it converts the tracer and raises `TracerError`, as the
  right one it is a `TypeError`; convert it with `qb.asarray` first.
- Errors: `quabla.QuablaError` is the base of `TracerError` (a `TypeError`),
  `RetraceLimitError` (a `ValueError`), and `UnsupportedOperationError` (a
  `ValueError` and `NotImplementedError` with `.op` and `.device`). Trace
  caches no longer raise `RetraceLimitError`; it remains for existing
  handlers. `quabla.RetraceWarning` (a `UserWarning`) reports an evicted
  trace.

### Custom Differentiation Rules

`custom_vjp`, `custom_jvp`, and `checkpoint` (alias `remat`) follow JAX:

```python
@qb.custom_vjp
def log1pexp(x):
    return qb.log(1.0 + qb.exp(x))

def log1pexp_fwd(x):
    return log1pexp(x), x                      # (output, residuals)

def log1pexp_bwd(x, g):                        # (residuals, cotangent)
    return (g * (1.0 - 1.0 / (1.0 + qb.exp(x))),)

log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)

@qb.custom_jvp
def safe_norm(x):
    return qb.sqrt(qb.sum(x * x))

@safe_norm.defjvp
def safe_norm_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    norm = safe_norm(x)
    return norm, qb.sum(x * t) / qb.where(qb.equal(norm, 0.0), 1.0, norm)

layer = qb.checkpoint(lambda h: qb.tanh(h @ w + b))
```

- `custom_vjp(fun, nondiff_argnums=())`, then `f.defvjp(fwd, bwd)`:
  `fwd(*args)` returns `(out, residuals)` with `out` structured like
  `fun(*args)`; `bwd(*nondiff_args, residuals, cotangent)` returns a tuple
  with one cotangent per differentiable argument, each with that
  argument's pytree structure, `None` for zeros. Reverse mode (`grad`,
  `value_and_grad`, `vjp`, `jacobian`, which uses reverse mode for such
  functions) evaluates `fwd` and `bwd`; forward mode (`jvp`) raises
  `ValueError`, as JAX does. `bwd` is ordinary traced code, so `grad` of
  `grad` and `hessian` (forward over reverse) differentiate it.
- `custom_jvp(fun, nondiff_argnums=())`, then `f.defjvp(rule)` (usable as
  a decorator): `rule(*nondiff_args, primals, tangents)` returns
  `(primal_out, tangent_out)`, with `tangent_out` linear in `tangents`.
  Forward mode uses the rule. Reverse mode uses its transpose, obtained by
  differentiating `tangent_out` with respect to `tangents` at zero tangents,
  which is exact for a linear rule; higher orders differentiate the rule.
  The primal value is always `fun(*args)`; the rule's `primal_out` is not
  used.
- `checkpoint(fun, static_argnums=())`: values and derivatives of every
  order equal those of `fun`. In reverse mode the backward pass recomputes
  the intermediates of `fun` from its arguments and the values it closes
  over instead of reading them from the forward pass: the compiled plan
  contains the recomputation (plan CSE does not merge it back), so the
  forward intermediates are not live until the backward pass. The memory
  saving depends on the backend releasing buffers after their last use.
- Outside any transform the wrapped function simply calls `fun`. Inside
  one, the call is staged: the rule graphs are traced from `fwd`, `bwd`, or
  the JVP rule (or derived from `fun` for `checkpoint`) once per call, and
  the IR records them in `custom` nodes that the symbolic transforms apply
  and plan compilation removes, so the result runs under `jit` on every
  device. `vmap` batches the rules with the function, including for a
  later reverse pass (`grad(vmap(f))`). A call of the function itself
  inside its own `fwd`, `bwd`, or JVP rule evaluates `fun` without the
  rule.
- Arguments at `nondiff_argnums` (`static_argnums` for `checkpoint`) are
  passed through unchanged and not differentiated; they must not hold
  traced arrays. Python scalars in other arguments are constants. `fun`,
  `fwd`, `bwd`, and the JVP rule must not close over tracers of an
  enclosing transform (a `TypeError`); pass such values as arguments.
  `checkpoint` may close over them and differentiates through them. Every
  traced output must be floating-point.
- A body of `cond`, `fori_loop`, or `scan` that calls such a
  function raises `ValueError`, since compiled regions do not keep rules.

### Pytrees

Transform arguments and results, optimizer parameters, and `Trainer` data
are pytrees: nested containers whose leaves are arrays or Python values.
The containers are:

- `dict`, traversed in sorted key order; keys must be mutually sortable
  (all strings, or all integers, for example);
- `list` and `tuple`; `None` is an empty container;
- every `NamedTuple` class, automatically, as in JAX: the result of a
  transform rebuilds the same class;
- classes registered with `quabla.tree.register_dataclass` or
  `quabla.tree.register`.

Dataclasses are not containers until registered, as in JAX:

```python
@qb.tree.register_dataclass
@dataclasses.dataclass(frozen=True)
class Layer:
    w: qb.Tensor
    b: qb.Tensor

qb.tree.register_dataclass(Mlp, data_fields=("layers",), meta_fields=("activation",))
qb.tree.register(Interval, flatten_fn, unflatten_fn)
```

`register_dataclass(cls, data_fields=None, meta_fields=())` makes the
`data_fields` children (default: every `__init__` field not in
`meta_fields`, in declaration order) and the `meta_fields` static; together
they must name every `__init__` field once. Values are rebuilt with
`cls(**fields)`, so frozen dataclasses work and `__post_init__` runs again.
`register(cls, flatten_fn, unflatten_fn)` is the general form:
`flatten_fn(value)` returns `(children, aux_data)` and
`unflatten_fn(aux_data, children)` rebuilds the value. Only exact
instances of a registered class are containers, and a class can be
registered once.

The structure includes each node's class and aux data (the meta fields of a
dataclass), and is part of every trace cache key: two classes with the
same fields, or two values of a meta field, never share a trace, so aux
data must be hashable and a new value retraces. Gradients, `vmap` results,
and Jacobian blocks keep the custom classes of their arguments; `vmap`
`in_axes`/`out_axes` prefixes may use the same classes (with equal aux
data), for example `in_axes=(Point(0, None),)`. Other container subclasses
(`OrderedDict`, a `list` subclass) and unregistered dataclasses are
rejected as transform arguments.

`quabla.tree` provides `flatten(tree) -> (leaves, treedef)`,
`unflatten(treedef, leaves)`, `leaves(tree)`, `structure(tree)`,
`map(f, tree, *rest)` over trees of one structure, and
`flatten_with_path(tree) -> ([(path, leaf), ...], treedef)`, where a path
is a tuple of dict keys, field names, and integer positions. A `TreeDef`
compares and hashes by structure and has `num_leaves` and
`unflatten(leaves)`.

### Random Numbers

`quabla.random` samples with explicit keys, in the style of `jax.random`:

```python
key = qb.random.key(0)
k1, k2 = qb.random.split(key)              # split(key, num=2) -> tuple of keys
w = qb.random.glorot_normal(k1, (64, 32))  # float64 by default
noise = qb.random.normal(k2, (128,), dtype=qb.float32)
step_key = qb.random.fold_in(key, step)
```

| Function | Result |
| --- | --- |
| `key(seed)` | a `Key` from an integer seed (modulo 2**64) |
| `split(key, num=2)` | a tuple of `num` new keys |
| `fold_in(key, data)` | a new key from a key and an integer |
| `uniform(key, shape=(), dtype=float64, minval=0.0, maxval=1.0)` | samples on `[minval, maxval)` |
| `normal(key, shape=(), dtype=float64)` | standard normal samples |
| `bernoulli(key, p=0.5, shape=())` | `bool_` samples, `True` with probability `p` |
| `glorot_normal(key, shape, dtype=float64)` | std `sqrt(2 / (fan_in + fan_out))` |
| `glorot_uniform(key, shape, dtype=float64)` | `[-l, l)` with `l = sqrt(6 / (fan_in + fan_out))` |
| `he_normal(key, shape, dtype=float64)` | std `sqrt(2 / fan_in)` |

- A `Key` is an immutable, hashable value; there is no global state, and the
  same key gives the same numbers on every platform. Use each key once:
  sampling with a key after splitting it reuses the stream of its children.
- The generator is SplitMix64, shared with `Tensor.split_key`,
  `Tensor.random_normal`, and `Tensor.glorot_normal`: `split(key, n)` is
  `Tensor.split_key(key.value, n)`, `normal(key(s), shape)` equals
  `Tensor.random_normal(shape, s)`, and `glorot_normal` equals
  `Tensor.glorot_normal` for 2-D shapes. Uniform samples use the top 53
  bits of one generator output each and are clamped after rounding, so
  `float32` samples also stay below `maxval`. Normal samples use the
  Box-Muller transform, bounded at about 8.6 standard deviations.
  SplitMix64 is a statistical generator, not a cryptographic one.
- Initializers compute fans as JAX does: `fan_in = shape[-2] * r` and
  `fan_out = shape[-1] * r`, with `r` the product of the leading
  dimensions; shapes must have rank 2 or more. `glorot_normal` and
  `he_normal` are untruncated normals (JAX truncates at two standard
  deviations and rescales).
- Sampling is eager on the host. Keys are not traced: a key passed as a
  traced argument of a transform, or a `seed`, `minval`, `maxval`, or `p`
  that is a tracer, raises `TypeError`. Inside a traced function, a sample
  drawn from a closure key, or from a key passed as a static argument (one
  trace per key), is a constant of the trace, so every call of the compiled
  program sees the same numbers. Draw samples outside and pass them as
  arguments when they must change between calls.

### Saving and Loading

```python
qb.save("checkpoint.qb", {"params": params, "opt": opt_state})
restored = qb.load("checkpoint.qb")
```

`save(file, tree)` writes a pytree to a path or a binary file object, and
`load(file)` reads it back. Leaves may be quabla arrays (`Tensor` or
`TensorView`), `None`, and Python `bool`, `int`, `float`, and `str` values;
containers may be `dict` with string or integer keys, `list`, and `tuple`.
Arrays load as `Tensor`s with the saved shape and dtype, bit for bit;
scalars keep their type and value (`float` exactly), dicts their key order,
and tuples stay tuples, so an optimizer state such as the
`{"step": 3, "m": ..., "v": ...}` of `optim.Adam` resumes unchanged.

NamedTuples and classes registered with `quabla.tree.register` or
`register_dataclass` are not serialized, since rebuilding them would mean
importing classes named by the file; save their fields, for example as a
dict, and rebuild the objects after loading. Other leaves (NumPy arrays,
`random.Key` values) and traced values raise `TypeError`.

The format needs neither NumPy nor pickle, so `load` never executes code
from the file; a malformed or truncated file raises `ValueError`. Version 1
is laid out as:

| Offset | Size | Content |
| --- | --- | --- |
| 0 | 8 | magic bytes `b"\x93QUABLA\n"` |
| 8 | 8 | header size `N`, unsigned little-endian |
| 16 | `N` | header: strict UTF-8 JSON |
| `16 + N` | rest | raw array data, concatenated |

The header is `{"format": "quabla", "version": 1, "tree": NODE}`, where a
node is `{"type": "none"}`, `{"type": "list" | "tuple", "items": [...]}`,
`{"type": "dict", "keys": [...], "values": [...]}`,
`{"type": "bool" | "int" | "str", "value": ...}`,
`{"type": "float", "value": float.hex(x)}`, or
`{"type": "array", "dtype": "float64" | "float32" | "bool", "shape": [...],
"offset": ..., "nbytes": ...}`. Array data is C-ordered and little-endian
(IEEE binary64 or binary32; one byte, 0 or 1, per bool), at `offset` bytes
from the start of the data section. A reader rejects any other version.

### Version and Type Stubs

`quabla.__version__` is the installed version string (for example
`"0.2.3"`). The Cargo manifest of `crates/quabla-python` is its single
source: maturin derives the package metadata from it, and the extension
carries the same string for a source tree without installed metadata.

The package ships `py.typed` and a stub, `quabla/_quabla.pyi`, for the
compiled extension. It types `Tensor`, `TensorView`, `TraceTensor`,
`dtype` and the dtype objects in detail, including that an operator with a
traced operand gives a `TraceTensor`; the other extension functions and
classes (the comparison functions, the v0.1 low-level transform entry
points, execution plans, and legacy classes) are declared with their
parameter names but loose (`Any`) types. The
pure-Python modules (`quabla.optim`, `quabla.tree`, the transforms) are not
annotated yet, so checkers infer their types.

## Device Execution

`qb.jit(fun, device=None, static_argnums=(), max_traces=8)` selects CPU by
default, or explicitly `"cuda"`, `"cuda:N"`, or `"mlx"`. A missing build
target raises `UnsupportedOperationError`; device lowering rejections carry
`.op` and `.device`. Invalid device spellings
are `ValueError`. Eager array operations remain on the host. Device calls
upload inputs and return host output pytrees; logical float64 device programs
emit one `UserWarning` per compiled function because execution is float32.
`qb.devices()` lists built targets, not physical GPU ordinals.

`qb.jit(fun, device="cuda", precision="float64")` executes float64 programs
natively in double precision on CUDA, without the warning. Consumer GPUs run
f64 arithmetic at a small fraction of their f32 rate (a matmul-heavy
value-and-gradient step takes about 9x longer on a GTX 1660 SUPER), so
float32 execution stays the default. Under `precision="float64"`
every operation the CUDA backend lowers runs in double: the NVRTC kernels
(elementwise, reductions, `cumsum`, Cholesky and its derivatives, fused and
host-driven `fori`/`scan`/`while`/`cond` regions) and cuBLAS `dgemm` and
cuSOLVER `getrf`/`getrs`/`syevd`/`geqrf`/`orgqr`/`gesvdj`; results match
the CPU float64 reference to about `1e-13` relative. One compiled program
has one floating element type, so a program that also contains float32
values raises `UnsupportedOperationError` with `.op == "float32"`; cast them
to float64 or use the default precision. A program without float64 values
compiles as with the default. The argument is a no-op on the CPU, which
already runs float64 natively, raises `UnsupportedOperationError` on MLX,
which has no float64 arithmetic, and is part of the trace-cache
configuration, so the two precisions never share a compiled program. Other
values than `None` and `"float64"` are `ValueError`. `qb.optim.Trainer`
takes the same `precision` keyword; the NCCL data-parallel paths keep
float32 device execution.

`qb.jit(fun).lower(*args)` accepts arrays or `qb.ShapeDtype(shape, dtype)`
inside pytrees and traces without execution. The lowered object exposes
`.as_text()`, an ordered-output facade `.program`, and `.compile()` returning
a positional callable that rejects changed structures, shapes, dtypes or
static values. The facade's `.output_shapes` lists traced outputs (including
duplicates); `.program.compile(target)(inputs_dict)` returns their ordered
list. `.compile()` on the lowered object also restores constants and pytree
structure. Transform the Python function before lowering; multi-output facade
`Program.jvp`/`vjp` are explicitly rejected. Existing `Compiler.trace` keeps
its single-output Tensor contract.

`qb.optim.Adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8)` and
`qb.optim.SGD(learning_rate=1e-2)` provide `init(params)` and pure
`update(params, grads, state) -> (params, state)` over floating Tensor pytrees.
Adam also preserves the old dictionary `step` and the keyword aliases
`beta1`, `beta2`, `epsilon`. Pure host moments use float64 as the old host
Adam does, and parameters keep their dtype.
`qb.optim.AdamW(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4)`
applies decoupled weight decay as optax's `adamw` does:
`p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)` with the
pre-update `p`; its state and precision rules are Adam's, and it has no
legacy `step`.

Every `learning_rate` also accepts a schedule, a callable `f(step) -> float`
evaluated on the host at the number of completed updates (0 for the first
update, as in optax). Adam and AdamW keep that count in `state["step"]`; a
scheduled SGD's state is `{"step": n}` instead of `None`. The schedules follow
optax's formulas and argument order: `constant(value)`,
`exponential_decay(init_value, transition_steps, decay_rate,
transition_begin=0, staircase=False, end_value=None)`,
`cosine_decay(init_value, decay_steps, alpha=0.0)`,
`warmup_cosine_decay(init_value, peak_value, warmup_steps, decay_steps,
end_value=0.0)` (where `decay_steps` includes the warmup), and
`piecewise_constant(boundaries, values)`, which returns `values[i]` from step
`boundaries[i - 1]` on. Invalid configurations raise `ValueError` where optax
warns, and a schedule value that is negative or not finite raises when used.

`qb.optim.clip_by_global_norm(grads, max_norm)` returns
`(clipped_grads, global_norm)`. The norm over all leaves is computed in
float64 as `m * sqrt(sum((g / m) ** 2))` with `m = max |g|`, so it neither
overflows nor underflows; leaves are scaled by `min(1, max_norm / norm)` and
keep their dtype. A NaN or infinite norm returns all-NaN gradients rather
than finite values; test `math.isfinite(global_norm)` to skip such steps.
The keyword `clip_norm=` on `Adam`, `AdamW`, and `SGD` applies the same
clipping to every `update`.

`qb.optim.LBFGS(history=10, max_iterations=500, max_evaluations=None,
tolerance_grad=1e-7, tolerance_change=1e-9, line_search="strong_wolfe")`
is a host-side full-batch minimizer for the "Adam, then L-BFGS" stage of PINN
training. `minimize(fun, params, *args)` jits `value_and_grad(fun)` on the CPU
once, iterates over one float64 vector of all parameter leaves (float32 leaves
are evaluated, and their iterate stored, at float32 precision), and returns
`(params, info)`. It uses the two-loop recursion with `H0 = (s'y / y'y) I`,
skips curvature pairs with `s'y <= 1e-10 ||s|| ||y||`, and a strong Wolfe line
search (`c1=1e-4`, `c2=0.9`, safeguarded cubic interpolation, at most 25
evaluations) that backtracks from non-finite losses. A failed line search
restarts once from steepest descent. `info` holds `iterations`,
`evaluations`, `loss`, `grad_norm` (infinity norm), `converged`, and `reason`:
`gradient_tolerance`, `change_tolerance`
(`|f_k - f_k+1| <= tol * max(|f_k|, |f_k+1|)`, scipy's `ftol` test without
its floor of 1, so small PINN losses are not stopped early), `max_iterations`,
`max_evaluations`, or `line_search_failed` (usual at the float32 resolution
limit).

`qb.optim.Trainer(loss, params, optimizer, *data, device="cpu", batch_argnums=(), precision=None)`
traces `loss(params, *data)`. Every device supports Adam, AdamW, and SGD
with a constant rate or a schedule, and with `clip_norm`. A schedule is
evaluated on the host and sets the native optimizer's `learning_rate` before
each step. Device updates follow the CPU definitions in the CPU's operation
order: AdamW's decay `weight_decay * p` uses the pre-update `p` and is scaled
by the scheduled rate, and the bias corrections and `1 - beta` are formed in
float64. `clip_norm` reduces the global norm on the device, without reading
gradients back: `m * sqrt(sum((g / m) ** 2))` with `m = max |g|` over all
parameters, accumulated in float64 on CUDA and in float32 on MLX (every term
is at most one, so neither overflows), then every gradient is scaled by
`min(1, clip_norm / norm)` before the moment updates. As on the CPU, a zero
norm leaves the gradients unchanged and a NaN or infinite norm makes every
parameter NaN. A parameter the loss ignores has a zero gradient: it changes
only by AdamW's decay, or to NaN after a non-finite clipped norm, as on the
CPU. Device plans compute in float32 by default and agree with the CPU
trainer to float32 rounding. `precision="float64"` (validated as in `jit`)
runs a float64 loss natively on CUDA with float64 parameters and moments,
agreeing with the CPU to float64 rounding; it is a no-op on the CPU and
raises `UnsupportedOperationError` on MLX.
`batch_argnums` indexes data positions, excluding params. `step(*batch)`
replaces them in the declared order, with unchanged pytree/shapes/dtypes;
`step()` reuses data without Python flattening on devices. Device parameters
and moments stay retained; `.loss()` and `.params` explicitly read back.
Compare the native and new training path with
`python examples/benchmark_v02_training.py --device cuda:0 --max-step-ratio 1.02`
(or `--device mlx`). This reports training parity and interleaved synchronized
step samples separately from trace/compile and diagnostic readback.

`qb.cond(pred, true_fun, false_fun, *operands)` runs one branch lazily.
`qb.fori_loop(lower, upper, body, init, operands=(), unroll=False)` calls
`body(i, carry, *operands)`. `qb.scan(f, init, length=n, operands=(), unroll=False)`
calls `f(carry, i, *operands) -> (carry, output)` and stacks outputs on axis zero.
Bounds are static non-negative integers, scan length is positive, and carries
are single arrays. Region indices are scalar arrays; eager/unrolled indices
are Python integers. Pass every outer tracer used by a region as an operand;
branches and bodies may ignore operands, and `cond` branches may return
constants of the other branch's dtype. Unsupported implicit capture raises
`TracerError` naming `operands=`.

`qb.while_loop(cond_fun, body_fun, init, operands=())` repeats
`carry = body_fun(carry, *operands)` while `cond_fun(carry, *operands)`, a
scalar bool, is true. A traced carry or operand forms one region; CPU and
MLX evaluate the predicate, read it back to the host once per iteration, and
then run the body, so the trip count may depend on traced values. CUDA
does the same with device-resident carry buffers: the predicate and body
regions are compiled once and only the scalar predicate is read back per
iteration. Forward mode (`jvp`) runs the
same loop over a packed primal/tangent carry. Reverse mode (`grad`, `vjp`,
and the reverse-mode `jacobian`/`hessian`) raises an error naming
`fori_loop`, as in JAX: a data-dependent trip count leaves no fixed tape, so
write a bounded `fori_loop` whose body masks finished iterations with
`where` instead. A body or predicate may ignore its carry. `vmap` batches a
while loop as `jax.vmap` does: when the predicate depends on a mapped value,
the predicate region becomes "any example continues" and the body selects
`where(pred, body(carry), carry)` per example, so a finished example keeps
its carry (and forward-mode tangent) bit for bit while the others iterate,
and whatever the body computes for it meanwhile, `NaN` or `inf` included,
is discarded. The body still runs on finished examples, so a batch costs
the trip count of its slowest example; backends still read back one scalar
flag per iteration.

`qb.linalg` holds dense linear algebra over the last two axes, batched over
leading axes that broadcast like NumPy's. Every function takes eager or
traced arrays and differentiates to every order, because each derivative rule
is written with `solve`, `matmul`, and the decompositions themselves:

- `solve(a, b)` solves `a @ x == b` per matrix by LU with partial pivoting
  (`quabla.solve` is the same function). `b` is a vector `[n]` only when it
  has rank one (NumPy 2 semantics), otherwise a stack `[..., n, k]`. A
  singular matrix raises. `vmap` batches it, broadcasting an unmapped
  operand over the batch.
- `solve_triangular(a, b, trans=0, lower=False)` reads one triangle of `a`;
  `trans=1` (or `"T"`) solves `a^T x = b`. `cholesky(a)` returns the lower
  factor, and `cho_solve(c, b, lower=True)` solves `a x = b` from a Cholesky
  factor with two triangular solves.
- `slogdet(a)` returns `SlogdetResult(sign, logabsdet)` from the LU factor:
  `log|det|` is a sum of `log|u_ii|`, so it does not overflow where `det`
  does. An exactly singular matrix gives `(0, -inf)` and a non-finite one
  `(nan, nan)`, as in NumPy. `det(a)` is `sign * exp(logabsdet)` (exactly 0
  when singular). The sign has a zero derivative; the gradient of
  `logabsdet` is `solve(a^T, g I)` and its tangent `trace(solve(a, da))`, so
  no explicit inverse is formed.
- `inv(a)` is `solve(a, I)`; `solve(a, b)` is faster and more accurate than
  `inv(a) @ b`.
- `eigh(a)` returns `EighResult(eigenvalues, eigenvectors)` of the symmetric
  part `(a + a^T) / 2`: eigenvalues ascending, eigenvectors as columns, each
  with its largest-magnitude component (the first on ties) positive. The CPU
  uses cyclic Jacobi rotations in float64 that stop once the off-diagonal
  norm is at most `eps * ||a||_F`, so eigenvalues are accurate to about
  `eps * ||a||_F`; CUDA uses cuSOLVER `syevd` in float32. Derivatives use
  `dw = diag(V^T dS V)` and `dV = V (F o (V^T dS V))` with `dS` the symmetric
  part of `da` and `F_ij = 1 / (w_j - w_i)` off the diagonal (VJPs are the
  transposes).
- `qr(a, mode="reduced")` returns `QRResult(Q, R)` with `a = Q R` for any
  `[..., m, n]`: with `k = min(m, n)`, `"reduced"` gives `Q` `[..., m, k]`
  and `R` `[..., k, n]`, `"complete"` an orthogonal `Q` `[..., m, m]` and `R`
  `[..., m, n]`, and `"r"` returns `R` alone. The CPU uses Householder
  reflections in float64 and forms `Q` from the reflectors, so `Q` is
  orthonormal to working precision even for a rank-deficient `a`; CUDA uses
  cuSOLVER `geqrf` and `orgqr` in float32. The diagonal of `R` is made
  non-negative (with the matching columns of `Q`), so the factorization is
  unique for full column rank and matches NumPy's up to those signs.
  Derivatives follow JAX's rule: with `B = da R^-1`, `C = Q^T B`, and
  `W = L - L^T` for the strictly lower triangle `L` of `C`,
  `dQ = Q (W - C) + B` and `dR = (C - W) R`, with `R^-1` applied by `solve`.
  They need full column rank; for a wide `a = [X | Y]` (`m < n`), `X` takes
  this rule and `dR_Y = dQ^T Y + Q^T dY`, so `X` must have full rank. The
  extra columns of a complete `Q` are not unique, and differentiating them
  raises.
- `svd(a, full_matrices=False, compute_uv=True)` returns
  `SVDResult(U, S, Vh)` with `a = U diag(S) Vh`, `S` descending, `U`
  `[..., m, k]` and `Vh` `[..., k, n]` (`[..., m, m]` and `[..., n, n]` with
  `full_matrices=True`), or `S` alone with `compute_uv=False`. Unlike NumPy
  and JAX, `full_matrices` defaults to false. Each column of `U` has its
  largest-magnitude component (the first on ties) positive, with the
  matching row of `Vh` signed alike. The CPU uses one-sided (Hestenes)
  Jacobi rotations in float64 on `a` (or `a^T` when wide) until every pair
  of columns is orthogonal to `m * eps`, which gives each singular value to
  high relative accuracy when `a` is well conditioned up to a column
  scaling; columns for zero singular values and the complements of the full
  form come from a Householder completion. CUDA uses cuSOLVER `gesvdj`
  (Jacobi, sorted) in float32. The derivatives follow JAX's `svd_jvp_rule`:
  `dS = diag(U^T da V)`, defined wherever the singular values are distinct;
  `dU = U (F o (dP S + S dP^T)) + (I - U U^T) da V S^-1` (last term for
  `m > n`) and `dV = V (F o (S dP + dP^T S)) + (I - V V^T) da^T U S^-1`
  (last term for `m < n`), with `dP = U^T da V` and
  `F_ij = 1 / (s_j^2 - s_i^2)`. As in JAX, `F` is set to zero where
  `s_i == s_j`, so for repeated singular values the vector derivatives are
  finite but not derivatives (the vectors are not unique), and a zero
  singular value of a non-square `a` makes them infinite or NaN. The
  complements of the full form have no derivative unless `a` is square.
- `lstsq(a, b, return_residuals=False)` solves `a x ~= b` from `qr`: for a
  tall or square `a` of full column rank, the least-squares solution
  `solve_triangular(R, Q^T b)`; for a wide `a` of full row rank, the
  minimum-norm solution `Q solve_triangular(R, b, trans=1)` from `qr(a^T)`.
  Shapes and broadcasting follow `solve` (`b` `[..., m, k]` or a vector
  `[m]`). With `return_residuals`, it returns `LstsqResult(solution,
  residuals)`, the residuals being `sum((b - a x)**2)` per column, formed
  from the residual itself. Unlike NumPy's SVD-based `lstsq`, a
  rank-deficient `a` is not supported: `R` is singular and the result is
  non-finite or meaningless, without an error. Derivatives come from those
  of `qr` and the triangular solves.
- `norm(x, ord=None, axis=None, keepdims=False)` follows NumPy: an int
  `axis` gives a vector norm, a pair of axes a matrix norm, and `axis=None`
  with `ord=None` the 2-norm of the flattened `x` (another `ord` then needs
  a 1-D or 2-D `x`). Vector orders are `None`/2 (the scaled method
  `x.norm`, which does not overflow in `float32`), 1, `inf`, `-inf`, 0 (the
  count of nonzero entries, derivative zero), and any real `p`, computed as
  `m * sum((|x| / m)**p)**(1/p)` with `m` the largest magnitude (the
  smallest for `p < 0`) through `stop_gradient`, so the powers neither
  overflow nor underflow and the gradient is exact. Matrix orders are
  `None`/`"fro"` (scaled), 1/-1 and `inf`/`-inf` (extreme absolute column
  and row sums), and 2/-2/`"nuc"` from the singular values.
- `matrix_power(a, n)` takes an integer `n` and multiplies by repeated
  squaring; `n = 0` gives the identity and a negative `n` powers `inv(a)`.
- `pinv(a, rtol=None)` is `V diag(1/s) U^T` from `svd`, dropping singular
  values at or below `rtol * max(s)`; `rtol=None` is NumPy's
  `max(m, n) * eps` (JAX uses ten times that). Its derivatives are those of
  `svd`: exact for distinct nonzero singular values, infinite or NaN for a
  rank-deficient non-square `a`.

Derivatives at exactly singular matrices are undefined: the derivatives of
`slogdet`, `det`, and `inv` call `solve`, which raises on the CPU and MLX and
reports the singular factor on CUDA (JAX returns non-finite values). A repeated
eigenvalue makes `F` infinite, so the eigenvector derivative is inf or NaN,
as in JAX, while the eigenvalue derivative stays defined.

On MLX, whose LU, `eigh`, QR, and SVD factorizations exist only on its CPU
stream, each factorization runs there with LAPACK (`getrf`, `syevd`,
`geqrf`/`orgqr`, `gesdd`) in float32, and the CPU backend's conventions
(ascending eigenvalues, signed vectors, a non-negative diagonal of `R`, the
Householder completion of a complete `Q` or full SVD basis, NaN for a
non-finite matrix) are applied on the GPU stream; the two streams share
unified memory, so nothing is copied between them. `solve` (and so `inv`,
`solve_triangular`, `cho_solve`, `lstsq`, and every derivative that solves)
substitutes with the LU factors in a Metal kernel and reads one flag back
to raise on an exactly zero pivot like the CPU. Exact zero pivots depend on
rounding: a matrix that is singular only in exact arithmetic, such as one
with linearly dependent rows, can leave a pivot of rounding size, and is
then solved with huge values instead of raising; LAPACK's float32 rounding
can do so where the CPU's float64 elimination finds an exact zero.

Iterative and implicit solvers take functions instead of matrices:

- `qb.linalg.cg(matvec, b, *, args=(), x0=None, tol=1e-5, atol=0.0,
  maxiter=None, M=None, info=False)` solves `matvec(x, *args) = b` for a
  symmetric positive-definite operator by preconditioned conjugate
  gradients (`M(r, *args)` applies the preconditioner).
- `qb.linalg.gmres(matvec, b, *, args=(), x0=None, tol=1e-5, atol=0.0,
  restart=20, maxiter=None, M=None, info=False)` solves a general operator
  by restarted, right-preconditioned GMRES with a twice-applied classical
  Gram-Schmidt Arnoldi process and Givens rotations.
- `qb.newton(f, x0, *, args=(), tol=None, maxiter=50, info=False)` finds a
  root of `f(x, *args) = 0` by Newton's method with a dense Jacobian,
  Householder QR steps, and an Armijo backtracking line search on
  `||f||^2`; `tol` defaults to the square root of the dtype's machine
  epsilon. An exactly singular Jacobian stops the iteration with failure.

Each runs eagerly in Python or, under `jit`, as one `while_loop` region that
stops at convergence (`||b - A x|| <= max(tol * ||b||, atol)` for the linear
solvers, `||dx|| <= tol * (1 + ||x||)` for `newton`). Derivatives with
respect to `b` and the array leaves of `args` follow the implicit function
theorem at the solution, one adjoint solve, instead of differentiating the
iterations; `x0` gets no gradient. Pass every array the operator depends on
through `args`: values a Python function closes over are not differentiated.
Reverse mode composes twice (`grad(grad(...))`, and the reverse-mode
`jacobian` and `hessian` of a solution); forward mode (`jvp`) is not
supported, and a solve cannot run inside a `cond`, `fori_loop`, or `scan`
body. `vmap` batches a solve over `b`, `x0`, and the leaves of `args`, and
composes with the derivatives in both orders (`vmap(grad(...))`,
`grad` of a loss over `vmap`): each example stops at its own tolerance,
because the batched loop freezes an example's state once it has converged,
so the batch costs the iterations of its slowest example. An eager solve
that does not converge raises `RuntimeError`; `info=True` returns
`(x, info)` with `"iterations"`, `"residual_norm"`, and `"success"`, per
example under `vmap`, and the only report under `jit` and `vmap`.

`qb.ode.odeint(f, y0, (t0, t1), steps=n, method="rk4", args=(), save=False, saveat=None)`
integrates `dy/dt = f(y, t, *args)` with `n` equal steps of classical RK4,
Heun's method (`"heun"`), or forward Euler (`"euler"`), at times
`t0 + i * dt` so long integrations do not drift. A traced state forms one
`fori_loop` region (a `scan` region with `save=True`, which returns the
`n + 1` states stacked on axis zero), so solves differentiate with respect
to `y0`, `t0`, `t1`, and `args` and run under `jit` on every device. `y0` is
a single array, and `f` must receive traced values through `args`, as loop
bodies do.

`qb.ode.odeint(f, y0, (t0, t1), method="dopri5", rtol=1e-6, atol=1e-9,
max_steps=512, info=False)` is the adaptive Dormand-Prince 5(4) pair with
first-same-as-last stages. Steps are accepted when the RMS norm of the
embedded error estimate scaled by `atol + rtol * max(|y|, |y_new|)` is at
most one; a PI controller (gains 0.7/5 and 0.4/5, safety 0.9, factor
clipped to [0.2, 10]) picks the next step, rejected steps fall back to the
I-controller, and the first step follows Hairer and Wanner's starting-step
algorithm. The last step lands exactly on `t1`, and integration may run
backward. The solve is a bounded `fori_loop` whose body stops advancing
once `t1` is reached, so it is reverse-mode differentiable; a traced solve
therefore always pays `max_steps` step attempts. It runs on every device;
its body slices and concatenates the packed carry and takes norms, so CUDA
runs it as a host-driven region loop (see the CUDA loop lowering notes).
Step sizes are controller
outputs without gradients (only the final step depends on `t1`), so
derivatives are those of the discrete scheme on the chosen mesh and are
piecewise smooth across accept/reject changes. An eager solve stops at
`t1` and raises `RuntimeError` when `max_steps` attempts do not reach it;
under `jit` it cannot raise, and `info=True` returns `(y, info)` with
`t`, `accepted_steps`, `rejected_steps`, and a bool `success`.

`qb.ode.odeint(f, y0, (t0, t1), method="rosenbrock23", rtol=1e-6,
atol=1e-9, max_steps=512, info=False)` is the adaptive, L-stable Rosenbrock
2(3) method of Shampine and Reichelt (MATLAB's ode23s) for stiff problems:
`W = I - h d J` with `d = 1 / (2 + sqrt(2))`, a second-order solution, and
a third-order error estimate. Each step forms `J = df/dy` and `df/dt` at
the current point in one batched forward-mode pass of `y0.size + 1`
tangents, so `f` needs no hand-written Jacobian, and solves three linear
systems with `W` through `solve` (LU with partial pivoting). Step control,
tolerances, `max_steps`, `info`, and the eager and `jit` behavior are those
of `dopri5`, with the controller gains over the error order 3. Gradients
differentiate through the Jacobian and the solves, so they are those of the
discrete scheme. It runs on every backend. An exactly singular `W` raises
like `solve`.

`saveat=ts` (any method) returns the states at the times `ts` stacked on
axis zero instead of the final state. `ts` is a 1-D array ordered from `t0`
to `t1` and inside the span (checked when it is concrete); under `jit`, a
time outside the span or one the bounded loop did not reach is NaN. The
states come from each method's continuous extension, written with masked
updates inside the loop, so adaptive steps are not shortened to land on
the save times: Shampine's fourth-order interpolant for `dopri5` (Hairer's
CONTD5 form), the method's own second-order interpolant for
`rosenbrock23`, and cubic Hermite interpolation of each step's end values
and slopes for the fixed-step methods, which keeps their order. Saved
states differentiate with respect to `y0`, `args`, the span, and the save
times. `saveat` cannot be combined with `save=True`.

`qb.distributed.value_and_grad(fun, devices=["cuda:0", "cuda:1"], shard_argnums=(1,),
argnums=0, reduction="mean")` is experimental single-node CUDA/NCCL execution.
Arguments and replicated parameters may be pytrees. Full batches split equally
on axis zero; `sum` sums shard losses/parameter gradients and `mean` averages
them. Mapped-input gradients and nested execution inside another transform
are rejected. Optimizer updates stay on the host. The cache retains at most
eight signatures; changed static values, shapes or dtypes specialize it.

The v0.1 entrypoints below keep their original spec/dictionary call forms:

| Task | CUDA (Linux, `cuda` feature) | MLX (Apple silicon, `mlx` feature) |
| --- | --- | --- |
| Primal function, any target | `Compiler().trace(fn, specs).compile("cuda")` | `Compiler().trace(fn, specs).compile("mlx")` |
| Symbolic JVP/VJP programs | `Program.jvp(name)`, `Program.vjp(cotangent_name)`, then `compile(target)` | same |
| Scalar loss with named gradients | `tensor_value_and_grad_cuda_fn` | `tensor_value_and_grad_mlx_fn` |
| Bounded batch-size specialization | `tensor_jit_batch_cuda_fn`, `tensor_value_and_grad_batch_cuda_fn` | `tensor_value_and_grad_batch_mlx_fn` |
| `vmap` and its JVP/VJP/HVP | `tensor_vmap_cuda_fn`, `tensor_vmap_{jvp,vjp}_cuda_fn`, `tensor_vmap_hvp_scalar_cuda_fn` | `tensor_vmap_mlx_fn`, `tensor_vmap_{jvp,vjp}_mlx_fn` |
| Device-resident Adam training | `cuda_adam_vjp_optimizer`, `cuda_adam_loss_optimizer` | `mlx_adam_loss_optimizer` |
| Data parallelism (`cuda-nccl`) | `tensor_value_and_grad_data_parallel_cuda_fn` | not available |

The core control-flow wrappers reuse the region builders `tensor_cond`,
`tensor_fori_loop_region`, `tensor_scan_region`, and
`tensor_while_loop_region`, with their existing backend limits.
`Compiler.capabilities()` reports which targets the build contains; it is
not a hardware probe. The PINN examples in
`examples/pinn_poisson_{mlx,cuda}.py` and `examples/pinn_mlp_{mlx,cuda}.py`
train on each device with these helpers. Their behaviour and limits are
described under [Compiler Facade](#compiler-facade) and
[Rank-N Tensor API](#rank-n-tensor-api).

## Compiler Facade

The rank-N compiler path has one explicit lifecycle. It is the advanced
layer of the v0.2 design, for explicit compilation and inspection from
name-keyed input specs on CPU, CUDA and MLX. Integrations that need a
compiled program should use this facade instead of coupling to a
backend-specific execution plan class; the legacy 2D `Matrix` tracer is
intentionally outside it. The [Quickstart](../README.md#quickstart) shows the
lifecycle end to end.

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
unequal-count output graphs and indexed bodies have no fused `ScanVjpJvp`
kernel; like their primal Scan and first-order `ScanVjp`, they run as
host-driven region loops.

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

## v0.1 API Reference

These entrypoints shipped in v0.1.0 and work unchanged on `main`. The v0.2
design ([Appendix A](api_v0_2_design.md#appendix-a-name-mapping)) maps each
of them to its core-API equivalent and deprecates most `tensor_*_fn` helpers
from v0.2 on; migrated names now warn once on top-level access, and every
name keeps working through 0.x.

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
  `**` with a scalar or broadcast tensor exponent and `c ** x` with a
  Python-number base (also `quabla.power(x1, x2)`),
  NumPy-style batched `matmul`, rank-N `quabla.concat([...], axis=...)`,
  permutation-validated `transpose(axes=None)`, global or single-axis `sum`/`mean`/L2 `norm`,
  common elementwise math (`tanh`, `exp`, `expm1`, `log`, `log1p`, `erf`, `sqrt`, `sin`, `cos`, `powi`),
  broadcast `atan2(y, x)`, `cumsum(axis=None, reverse=False)`, `stop_gradient()`,
  `gt(...)` masks, `maximum(...)`/`minimum(...)`, broadcasted `quabla.where(...)`, materialized `broadcast_to(shape)`,
  and element-count-preserving reshape.
  Eager `Tensor` operations run immediately on the host; differentiation and
  compilation apply to traced functions, whose arguments are `TraceTensor`
  values (see `trace_tensor` below). `Tensor.slice(...)` returns a
  zero-copy, read-only `TensorView` with explicit shape, strides, and offset;
  source tensors and views share immutable storage, while every arithmetic
  operation returns a new contiguous allocation.
- Neural and linear-algebra primitives on eager and traced tensors: `relu`
  (zero subgradient at zero), `abs`, `sigmoid`, `softplus`, rank-N
  `tril`/`triu` over the last two axes, and `solve(rhs)` (partial-pivot LU
  per matrix of the leading batch axes, which must match),
  `solve_triangular(rhs, lower=True, transpose=False)`, and `cholesky()`,
  each with JVP/VJP rules and batched over leading axes. `solve` rejects
  non-square, rank-mismatched, batch-mismatched, and singular inputs; on
  CUDA it lowers to cuSOLVER `Sgetrf`/`Sgetrs` per batch element, and on MLX
  to LAPACK `getrf` on MLX's CPU stream with a Metal substitution kernel. `sigmoid` is
  `where(x > 0, 1 / (1 + z), z / (1 + z))` with `z = exp(-|x|)`, so neither
  branch overflows and its gradient stays finite for every finite `x`;
  `softplus` is `maximum(x, 0) + log1p(exp(-|x|))`. Eager `Tensor` evaluates
  both with the same per-op rounding as CPU `jit`, so the two agree bitwise.
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
  reports `"f32"`/`"f64"`. Immutable host storage uses 8/4/1 bytes per
  element for F64/F32/Bool, with no persistent widened cache. Factories such as
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
  `f64` input/add/multiply/power/tanh subset as deterministic textual StableHLO for
  compiler-tool inspection. It rejects unsupported operations and is not an
  execution backend or a complete StableHLO lowering.
- `tensor_jacobian_fn(fn, input_specs, input_name)` freezes one rank-N trace
  and returns an output-flat by input-flat dense Jacobian for the selected input.
  Its `TraceTensor` values currently support broadcasted add/subtract/multiply/divide,
  batched `matmul`, rank-N `concat`, `stack([...], axis=...)`, `slice(axis, start, stop)`, `broadcast_to(shape)`, rank-N `transpose`, `tanh`, `exp`, `sin`, `cos`, `sqrt`, non-negative integer `powi`, `pow` (`x ** y` and `quabla.power`), `log`, `log1p`, reshape, global or single-axis `sum`/`mean`/L2 `norm`, `maximum`/`minimum`, `gt`/`where` masks, and the bool comparison, logical, `isfinite`/`isnan`, and `any`/`all` operations. `stack` is composed from reshape plus concat, so it inherits the same direct and symbolic CPU/CUDA AD rules. `concat` is linear: direct and symbolic VJP split the upstream cotangent with internal slice nodes, while its JVP and mixed second-direction transform concatenate the corresponding tangents. `slice` supports normalized negative axes and uses a zero-padded internal reverse node, keeping direct and symbolic gradients on the selected original coordinates. `broadcast_to` is a dedicated shape node whose VJP reduces repeated axes back to the input shape. `sqrt` is a native IR primitive: negative values, `-inf` included, follow IEEE floating-point `NaN` semantics for the value and every derivative order, while every derivative order at zero is defined as zero, avoiding `log(0)` during higher-order AD; CPU, CUDA, and MLX agree on these points. A `TraceTensor` `x ** y` lowers a non-negative Python int `y` to the exact `powi` and every other exponent (floats, negative ints, traced tensors, and `c ** x`) to the elementwise `pow` op, which follows `f64::powf` (NaN for a negative base with a non-integer exponent, `0 ** 0 == 1`) and differentiates in both operands: `d/dx = y x^(y-1)`, defined as zero where `x == 0` and `y < 1` (the `sqrt` convention; `y >= 1` keeps the finite limit, so `x ** 2.0` has second derivative 2 at zero), and `d/dy = x^y ln x`, defined as zero for every `x <= 0` (JAX returns `NaN` for `x < 0`). The rules mask these points out of the inner power too, so second derivatives stay finite there wherever the value is finite; at `x <= 0` the two mixed second partials follow the conventions and need not be equal. `pow` lowers to NVRTC `powf` on CUDA, including fused elementwise kernels and elementwise loop bodies, and to `power` on MLX. Comparisons are explicitly non-differentiable; `where` routes VJP/JVP contributions only through the selected data branch. `maximum` and `minimum` are composed from those primitives as `where(isnan(x) | (x > y), x, y)` and `where(isnan(x) | (x < y), x, y)`: they route equality subgradients to their right operand and propagate `NaN` from either operand like NumPy, and so does `relu`; the `max`/`min` reductions also propagate `NaN` from any position, but split tie derivatives equally (see `Tensor.max`). `log1p` is a native IR primitive, `ln(1 + x)` with derivative `1 / (1 + x)`, that follows IEEE semantics on every backend instead of raising: `log1p(-1)` is `-inf` and `log1p(x)` is `NaN` for `x < -1`; it lowers to `log1pf` on CUDA and to `log1p` on MLX. `expm1` (`exp(x) - 1`, accurate for small `|x|`, derivative `exp(x)`), `erf` (the f64 musl `erf` of the `libm` crate on the CPU, derivative `2 / sqrt(pi) * exp(-x^2)`), `erfc` (`1 - erf(x)` without cancellation: the musl `erfc` on the CPU and `erfcf` on CUDA, derivative `-2 / sqrt(pi) * exp(-x^2)`; MLX has no `erfc`, so it evaluates the Numerical Recipes Chebyshev fit `t exp(-z^2 + P(t))`, `t = 1 / (1 + z / 2)`, relative error below `1.2e-7`, with `exp(-z^2)` split as `exp(-h^2) exp(-(z - h)(z + h))` for `h = floor(16 z) / 16` and the reflection `2 - erfc(-x)` for negative `x`, within about 8 float32 ulp of the correctly rounded value; CUDA loop bodies do not lower `erfc` yet), and the broadcasting binary `atan2(y, x)` (`f64::atan2`) are native IR primitives that lower to `expm1f`, `erff` and `atan2f` on CUDA (per-node kernels and elementwise loop bodies) and to `erf` and `arctan2` on MLX, where `expm1` is Kahan's `(u - 1) * x / log(u)` with `u = exp(x)` for `|x| < 0.5` and `exp(x) - 1` elsewhere, because MLX's own `expm1` is off by hundreds of float32 ulp; device float32 results differ from the correctly rounded CPU values by a few ulp (MLX `erf` by up to about 13). The `atan2` partials `x / (x^2 + y^2)` and `-y / (x^2 + y^2)` are evaluated after dividing both operands by `s = |x| + |y|`, so they neither overflow nor underflow before the true value does, and every derivative order is defined as zero where `s` is zero or infinite (JAX's rule gives NaN at the origin and for tiny inputs whose squares underflow); a NaN operand gives a NaN derivative. `stop_gradient(x)` is the identity on values with a zero derivative in every mode (zero JVP tangent, no VJP contribution, zero Hessian blocks), so `x - stop_gradient(x) + stop_gradient(f(x))` has the value of `f(x)` and the gradient of `x`; backends execute it as a copy, and an eager `Tensor.stop_gradient()` returns the value. `cumsum(x, axis=None, reverse=False)` is a native inclusive prefix sum (`axis=None` flattens like NumPy, `reverse` scans from the last entry) whose CPU evaluation rounds every running sum to the dtype; its JVP is the cumsum of the tangent, its VJP the opposite-direction cumsum of the cotangent, and `vmap` shifts its axis. CUDA scans each line in one thread with `__fadd_rn`, matching the CPU float32 result bitwise; MLX uses its parallel `cumsum`, whose float32 rounding can differ from the sequential CPU sums. `TensorTraceGraph.evaluate_vjp(...)` and
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
  Every loop node outside these fused subsets (bodies that slice,
  concatenate, reshape across lanes, reduce, contain `Cond` or array
  constants; capture gradients that reduce over broadcast axes inside the
  body VJP; `While`; Hessian-vector products through such a `Scan`) runs as
  a host-driven region loop instead: each region (body, predicate, body JVP
  or body VJP) is compiled once to the per-node CUDA program, and the host
  launches it once per iteration on device-resident carry buffers, with no
  host copy per iteration except the `While` predicate. Loop-invariant
  captures are bound once per loop execution. After two eager iterations, a
  region whose program only launches NVRTC kernels, device copies and cuBLAS
  products (no `Cond`, nested loop or cuSOLVER call) is recorded once as a
  CUDA graph, and every later iteration retargets the graph's input and
  output copies and launches it as one graph; the replay runs the same
  kernels in the same order, so results are bit-identical to eager
  iterations. Reverse passes keep the carry tape on the device, in full for
  short loops and as square-root checkpoint blocks replayed in reverse
  otherwise (the CPU scheme), and accumulate each capture gradient once per
  iteration in reverse order, its broadcast axes reduced inside the body
  VJP; float32 results differ from the CPU only by rounding order. Launch
  cost still dominates small bodies: on a GTX 1660 SUPER (WSL) a small
  rotating body costs about 30-50 µs per `fori_loop` or `scan` iteration
  (120-180 µs before graph replay), about 100 µs per `while_loop` iteration
  (which waits for its predicate), and about 120-140 µs per iteration of
  their reverse passes; `examples/benchmark_host_loop_cuda.py` measures
  them.
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
  parent plan and binds captures as device buffers; a `Fori`/`Scan` body
  containing `Cond` runs as a host-driven region loop. Vmapped predicates or
  operands are
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
  an L2 norm over the selected axes, composed as
  `m * sqrt(sum((x / m).powi(2)))` with `m = max|x|` over those axes (1 when
  `m` is zero, infinite, or `NaN`), so the sum of squares neither overflows
  nor underflows for `float32` magnitudes above about 1e19 or below about
  1e-19. It preserves the normalized-axis, backend, and AD contract of its
  parts; gradients equal `x / norm` up to rounding.
- `Tensor.max(axis=None, keepdims=False)` / `Tensor.min(...)` and traced
  equivalents reduce one or more axes. A traced reduction is one native IR
  node (the reduced axes are moved last and flattened), so staged programs
  and their derivatives have a constant size in the reduced extent. NaN
  propagates and the sign of zero is ordered as in IEEE 754-2019 `maximum`
  (`max` of `-0.0` and `+0.0` is `+0.0`), on every backend. Derivatives
  follow JAX's rule: tied extrema, across all reduced axes, share the
  derivative equally, and a NaN extremum has a NaN derivative.
- `Tensor.gather(indices, axis=0)` and `Tensor.scatter_add(indices, updates,
  axis=0)` support static integer index sequences, including negative and
  repeated indices. `scatter_add` is functional and accumulates repeated
  destinations in index order. Each traces to one native `gather` or
  `scatter_add` IR node on the CPU, CUDA, and MLX, so staged programs and
  their derivatives stay the same size for any index count; the VJP of one is
  the other. Dynamic tensor indices
  (pending `I32` index tensors), boolean-mask indexing, and assignment-style
  scatter are not yet supported.
- `quabla.einsum("ij,jk->ik", [lhs, rhs])` and
  `quabla.einsum("...ij,...jk->...ik", [lhs, rhs])` are scoped matrix-product
  spellings that lower directly to the existing rank-N `matmul` plan. Other
  einsum equations are rejected rather than silently interpreted.
- `Tensor` and `TraceTensor` support `__getitem__` with NumPy basic
  indexing: integers (negative ones count from the end), slices of any
  non-zero step, `None` (a new unit axis), and one `...`. Integer indices
  lower to a length-one slice plus reshape, a strided slice to the
  contiguous slice spanning its elements plus a `gather`, and `None` to a
  final reshape, so reverse-mode AD and CUDA/MLX lowering preserve the same
  semantics. Empty slices and advanced (array or boolean-mask) indexing are
  not supported.
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
- `Tensor.split_key(key, count)`, `Tensor.random_normal(shape, key, ...)`,
  `Tensor.random_uniform(shape, key, minval, maxval, dtype)`,
  `Tensor.fold_in_key(key, data)`, and `Tensor.glorot_normal(shape, key)`
  provide stateless, deterministic host-side initialization; `quabla.random`
  is the keyed API over them. They do not retain global RNG state; generated parameters are
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
  [Installation From Source](../README.md#installation-from-source).
- Python `Adam` updates immutable dictionaries of named rank-N `Tensor`
  parameters from VJP gradients, including the gradient dictionaries that
  `qb.grad` and `qb.value_and_grad` return for a flat dict of parameters
  (README "At a Glance"). It is the deprecated top-level alias of
  `quabla.optim.Adam` (see [Device Execution](#device-execution)), which keeps
  this stateful `step`. The test suite includes a manufactured 1D
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
- Python `grad(fn, input_specs, values, output_cotangent)` convenience API
  (`quabla.grad` dispatches this form here; other calls are the v0.2
  transform above).
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
  evaluates a frozen CPU execution plan for later value dictionaries
  (`quabla.jit` dispatches a non-callable first argument here).

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

## Known Limitations

- No machine-code JIT for the CPU backend, which interprets frozen plans.
  CUDA generates NVRTC kernels; MLX dispatches MLX operations.
- Shapes are static. Variable batch sizes use bounded per-size
  specialization (`tensor_jit_batch_fn` and its variants); there are no
  symbolic dimensions.
- Dtypes are `float32`, `float64`, and `bool`. `float16`/`bfloat16`, integer
  tensors, and mixed-precision training are not implemented. Native device
  `f64` is opt-in on CUDA through `jit(precision="float64")` and
  `Trainer(precision="float64")`, and unavailable on MLX. Host storage is physically typed F64/F32/Bool; narrow
  storage widens transiently only when an F64 view is requested.
- `gather`/`scatter_add` take static Python integer indices; dynamic index
  tensors, boolean-mask indexing, and empty slices are unsupported.
- CUDA loop bodies must be pure elementwise: matmul, reductions, `solve`,
  nested regions, and `cond` inside `fori`/`scan` are rejected. `cond`
  results and loop-region inputs must be floating, not `bool`.
- Derivatives beyond forward-over-reverse of `fori`/`scan` regions are
  explicit errors. Vmapped `cond` predicates are rejected at trace time.
- MLX runs its factorizations on the CPU stream (see `quabla.linalg`), and
  `vmap` HVP has no MLX lowering. Triangular solves use the general LU
  `solve` on devices.
- Third or higher Cholesky derivatives use a scalar expansion whose graph
  grows as O(n^3) and is impractical beyond small matrices; first and second
  orders use the native kernels in `float32` and `float64`.
- Data parallelism is single-node CUDA + NCCL only: equal axis-zero batch
  shards, replicated parameter gradients, and an optimizer on the host. There
  is no multi-node transport, tensor parallelism, or sharded matmul.
- StableHLO export (`stablehlo_text`) covers only a small inspection subset
  and is not an execution path.
- The legacy 2D `Matrix`/`TraceGraph` API is not migrated to the facade.
- `quabla.vmap` cannot batch `cond` regions over a mapped argument
  (`fori_loop`, `scan`, and `while_loop` regions batch). A batched
  `while_loop` whose predicate is per example runs its body on every
  example until the slowest one finishes. On CUDA, the
  forward-mode `scan` region inside a batched directional derivative (a
  forward-mode `jacobian` or `hessian` through `scan`) runs host-driven:
  its packed `[primal, tangent]` carry gains a leading batch axis, which the
  fused packed-pair kernel does not lower.
- `qb.linalg` has no `eig` (non-symmetric). Batched CUDA solves and decompositions issue one cuSOLVER call per
  batch element. Derivatives at exactly singular matrices raise instead of
  returning non-finite values.
- `custom_vjp`, `custom_jvp`, and `checkpoint` functions cannot be called
  inside control-flow bodies, and their rules cannot close over tracers of
  an enclosing transform (`checkpoint` can).
  `jacobian` and `hessian` are dense: their basis constant and result grow
  quadratically with the number of input elements.
