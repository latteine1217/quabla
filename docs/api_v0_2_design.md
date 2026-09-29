# Quabla v0.2 Python API Design

Status: accepted by the owner on 2026-09-29; implementation pending. Scope:
slices S0 through S9 (plus S1b) ship as v0.2, and the repository becomes
public after they land; S10 is deferred. Decisions D1-D17 are accepted as
recommended, with one change: traced float exponents get a real `Pow` IR op
(section 3.2, slice S1b) instead of being rejected. The resolved open
questions are recorded in section 7. Base: `main` at `7c07a22`. Citations: `py/` is `crates/quabla-python/src/`, `core/` is
`crates/quabla-core/src/`; "Exp N" is an experiment in
[Appendix B](#appendix-b-experiments).

## 1. Summary

Quabla v0.1 exposes 132 public Python names, and every combination of
transform, backend, and batching has its own entrypoint. Users write input
shape specs up front, seed cotangents by hand, and reach into compiler objects
(`graph.input`, `.output`, `compile_cpu()`) to express an ordinary PINN loss.
v0.2 adds a JAX-style layer of composable function transforms (`grad`,
`value_and_grad`, `jvp`, `vjp`, `vmap`, `jit`, `jacobian`, `hessian`) that
trace on first call, take the backend as a `device=` parameter, accept NumPy
data and nested lists, and return NumPy-convertible arrays. Device-resident
training keeps its current performance through `quabla.optim.Trainer`, which
maps onto the existing retained CUDA and MLX Adam executors. The explicit
`Compiler`/`Program`/`Executable` facade stays as the advanced layer, the 2D
`Matrix` API moves to `quabla.legacy`, and every existing name keeps working
through 0.x as a deprecated alias. Most of the work is a thin pure-Python
package over the extension. The main Rust additions are a graph-inlining
primitive in the core, which makes nested transforms composable, and NumPy
buffer interop in the bridge.

### Goals

- G1. A core namespace of about 13 functions that covers the README examples,
  with no shape specs, no manual cotangent seeds, and no graph handles.
- G2. Transforms compose: `grad(grad(f))`, `jvp(grad(f))`, `vmap(grad(f))`,
  and "coordinate derivatives inside a loss, then gradients in the weights".
- G3. One program on CPU, CUDA, or MLX, selected by `jit(..., device=...)`.
- G4. NumPy interop in both directions, with dtype inference.
- G5. No regression on device-resident MLX/CUDA Adam training.
- G6. No existing user code breaks in 0.x.

### Non-goals

- New numerics, new ops, new backends, or new AD rules (except the gaps
  listed under Rust work: `inline`, `__neg__`/`__pow__` on tracers, buffer
  interop, typed lowering errors).
- Removing or changing the behaviour of any existing name in 0.x.
- Full NumPy API compatibility (`docs/design.md:42` already excludes it).
- Symbolic shapes, dynamic indexing, integer dtypes, multi-node
  distribution, and user-registered pytree nodes.

## 2. Current State

### 2.1 Public surface by concept

Counts come from `dir(quabla)` on the installed build (132 non-underscore
names); the full per-name list is in [Appendix A](#appendix-a-name-mapping).

| Group | Count | Examples | Registered at |
| --- | ---: | --- | --- |
| `tensor_*` transform helpers | 36 | `tensor_value_and_grad_mlx_fn`, `tensor_vmap_hvp_scalar_cuda_fn`, `tensor_cond`, `tensor_scan_region` | `py/lib.rs:440-519` |
| `Tensor*` result, plan, and graph classes | 36 | `TensorBatchCudaValueAndGradFunction`, `TensorMlxExecutionPlan`, `TensorTraceResult` | `py/lib.rs:387-414, 471-481` |
| Legacy 2D API | 26 | `Matrix`, `trace`, `grad`, `jit`, `grad_scalar_fn`, `TraceGraph` | `py/lib.rs:365, 415-439` |
| dtypes and module-level ops | 19 | `float32`, `bool_`, `where`, `concat`, `greater`, `logical_and` | `py/lib.rs:368-386, 520-523` |
| Device optimizer factories | 5 | `mlx_adam_loss_optimizer`, `cuda_adam_vjp_optimizer` | `py/lib.rs:496, 516-519` |
| Other | 10 | `Tensor`, `TraceTensor`, `Compiler`, `Program`, `Adam`, `trace_tensor`, `quabla` | `py/lib.rs:362-367, 387-391` |

`quabla` itself appears in the list because maturin generates
`quabla/__init__.py` as `from .quabla import *` over the native submodule
`quabla/quabla.*.so` (`pyproject.toml:52-55`, no `python-source`).

### 2.2 How it works today

1. **Specs and tracing.** Helpers take `input_specs`, a list of `(name,
   shape[, dtype])` tuples (`py/tensor_trace.rs:24-56`). Tracing creates one
   `TensorTraceGraph` (`Arc<Mutex<TensorIr>>`, `py/tensor_trace.rs:91-95`),
   calls the function positionally with `TraceTensor`s, and requires exactly
   one `TraceTensor` back (`py/tensor_trace.rs:4375-4399`; Exp 2).
2. **Transforms produce new graphs.** `symbolic_jvp(name)` seeds a ones
   tangent (`powi(x, 0)`, `core/tensor_ir.rs:2109-2127`), so it is the
   coordinate JVP `J·1`, not a gradient. `symbolic_vjp(ct_name)` adds a named
   cotangent input and returns one gradient node per input
   (`core/tensor_ir.rs:2672-2690`). Both replay into a fresh `TensorIr` that
   the bridge wraps as a new `TensorTraceResult`
   (`py/tensor_trace.rs:1679-1741`). Runtime-tangent JVP is
   `symbolic_jvp_with_tangent_inputs` (`core/tensor_ir.rs:2136`).
3. **vmap is a trace-time batch tracker** with one level of batching:
   `batch_axis: Option<usize>` on each tracer (`py/tensor_trace.rs:97-105,
   4531-4594`).
4. **Compile.** Single outputs snapshot the IR and go through
   `QuablaCompiler` (`py/tensor_trace.rs:562-614`); value-and-gradient helpers
   use `compile_many` (`py/tensor_trace.rs:622-650`). MLX validates lazily, on
   first execution (`core/compiler.rs:374-382`).
5. **Execute.** Callables take `dict[str, Tensor]` only
   (`py/tensor_trace.rs:7125-7138`) and return host `Tensor`s. CPU
   value-and-gradient seeds the cotangent internally
   (`py/tensor_trace.rs:3250-3262`); `plan.evaluate_value_and_vjp` does not.
6. **Caching.** Non-batch helpers trace once at construction. Batch helpers
   cache one plan per batch size, up to `max_specializations=4`, and raise on
   overflow (`py/tensor_trace.rs:207-280, 5236-5279`); they trace every input
   as `float64` (`py/tensor_trace.rs:3906-3910`), so a `float32` input yields
   a `float64` result (Exp 5).
7. **Device training.** `mlx_adam_loss_optimizer`/`cuda_adam_loss_optimizer`
   take a traced loss, parameter names, and host inputs, keep parameters,
   Adam moments, and retained inputs on the device, and `step(inputs=None)`
   replaces only non-retained inputs (`py/tensor_trace.rs:6069-6127,
   7041-7096, 4219-4227, 6823-6858`). Host `Adam.step(params, grads)` works
   on dicts (`py/optim.rs:75-166`).

### 2.3 Pain points

| # | Pain point | Evidence |
| --- | --- | --- |
| P1 | Name explosion: 9 spellings of value-and-grad (CPU/CUDA/MLX × plain/batch, data-parallel, cond, legacy) | Appendix A |
| P2 | `quabla.grad`, `quabla.jit`, `quabla.trace` are the legacy 2D API | `py/trace.rs:3311-3446` |
| P3 | Shape specs up front; names instead of positions | `py/tensor_trace.rs:24-56` |
| P4 | Compiler internals in user code: `symbolic_jvp`, `graph.input`, `.output`, `compile_cpu()`, manual seed `Tensor([], [1.0])` | README "At a Glance" |
| P5 | Construction only as `Tensor(shape, flat_data)`; nested lists and 2D NumPy raise `TypeError`; a float32 NumPy array becomes `float64`; no `numpy()`/`tolist()`/`item()`/`float()` | Exp 1, Exp 6; `py/tensor.rs:1516-1521` |
| P6 | NumPy round trip is element-by-element: 29 ms each way for 1e6 values vs about 1 ms for a copy | Exp 6 |
| P7 | Tracers lack `-x` and `x ** n` although eager tensors support them | Exp 2; `py/tensor.rs:1916-1940` vs `py/tensor_trace.rs:1945-2000` |
| P8 | Single-output traces; no aux outputs | `py/tensor_trace.rs:4389-4392` |
| P9 | Almost every failure is a `ValueError` with a string; MLX op rejections surface on first execution | `py/tensor_trace.rs` passim; `core/compiler.rs:380` |

### 2.4 Before and after

**README "At a Glance", today** (loop excerpt; imports and print omitted):

```python
u = quabla.trace_tensor(lambda x, w: (x * w).sin(), [("x", [8, 1]), ("w", [1, 1])])
u_xx = u.symbolic_jvp("x").symbolic_jvp("x")  # exact d2u/dx2, no finite differences
x = u_xx.graph.input("x")
loss = (u_xx.output + math.pi**2 * (math.pi * x).sin()).powi(2).mean()
plan = loss.compile_cpu()  # frozen execution plan, reused every step
points = {"x": quabla.Tensor.linspace(0.05, 0.95, 8).reshape([8, 1])}
params, adam = {"w": quabla.Tensor([1, 1], [2.5])}, quabla.Adam(learning_rate=0.05)
seed = quabla.Tensor([], [1.0])  # d loss / d loss
for _ in range(300):
    value, grads = plan.evaluate_value_and_vjp({**points, **params}, seed)
    params = adam.step(params, {"w": grads["w"]})
```

**v0.2:**

```python
import math
import quabla as qb

def u(x, w):                                   # one collocation point, scalar x
    return qb.sin(x * w)

u_xx = qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None))   # exact d2u/dx2 per point

def loss(w, x):
    return qb.mean((u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2)

x, w = qb.linspace(0.05, 0.95, 8), qb.array(2.5)
opt = qb.optim.Adam(learning_rate=0.05)
state = opt.init(w)
step = qb.jit(qb.value_and_grad(loss))         # traced and compiled on first call
for _ in range(300):
    value, g = step(w, x)
    w, state = opt.update(w, g, state)
print(f"w = {w.item():.6f}, loss = {value.item():.1e}")
```

For device-resident training, `qb.optim.Trainer(loss, w, qb.optim.Adam(0.05),
x, device="mlx")` replaces the loop body with `trainer.step()`. The `u_xx`
subgraph is today's (two coordinate JVPs, §3.4), differentiated by one VJP.

**Quickstart, today:** specs list, `Tensor([4], [...], dtype=...)` per input,
`tensor_value_and_grad_fn(loss, specs)`, then `Compiler().trace(loss, specs)`,
`program.compile(target)`, `program.vjp("loss_bar")["w"]` and a manual seed.

**v0.2:**

```python
def loss(x, w):
    y = qb.tanh(x * w)
    return qb.sum(qb.where(x > 0.0, y, 0.0) ** 2)

x = qb.array([-1.0, 0.5, 1.0, 2.0], dtype=qb.float32)
w = qb.array([0.3, -0.2, 0.1, 0.4], dtype=qb.float32)
value, (gx, gw) = qb.value_and_grad(loss, argnums=(0, 1))(x, w)
print(value.dtype, value.item(), gw.tolist())
for device in qb.devices():                    # ["cpu", "mlx"] on an MLX build
    print(device, qb.jit(loss, device=device)(x, w).item())
```

## 3. Proposed API

### 3.1 Layering

| Layer | Names | Stability in 0.x |
| --- | --- | --- |
| Core transforms | `grad`, `value_and_grad`, `jvp`, `vjp`, `vmap`, `jit`, `jacobian`, `hessian` | new, primary |
| Core arrays | `array`, `asarray`, `Tensor` (+ `Array` ABC), `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, dtypes `float32`/`float64`/`bool_` | new + kept |
| Core control flow | `cond`, `fori_loop`, `scan` | new |
| Ops | `sin`, `exp`, `tanh`, `sum`, `mean`, `matmul`, `where`, `concat`, ... (§3.2) | new + kept |
| `quabla.optim` | `Adam`, `SGD`, `Trainer` | new |
| `quabla.tree` | `flatten`, `unflatten`, `map` | new |
| `quabla.distributed` | `value_and_grad` (CUDA + NCCL) | new, experimental |
| Advanced | `Compiler`, `Program`, `Executable`, `ShapeDtype`, `devices`, plan classes, device optimizer types | kept |
| `quabla.legacy` | the 26 legacy 2D names | moved |
| Compatibility aliases | every other v0.1 name | deprecated, removed at 1.0 |

### 3.2 Arrays

```python
qb.array(obj, dtype=None) -> Tensor      # always copies
qb.asarray(obj, dtype=None) -> Tensor    # returns obj unchanged if already a Tensor of dtype
qb.zeros(shape, dtype=None); qb.ones(...); qb.full(shape, value, dtype=None)
qb.arange(start, stop=None, step=1.0, dtype=None); qb.linspace(start, stop, num, dtype=None)
qb.eye(n, m=None, dtype=None)
Tensor.numpy() -> np.ndarray; Tensor.tolist(); Tensor.item(); float(t); np.asarray(t)
```

- **Inputs:** Python scalars, rectangular nested lists/tuples, NumPy arrays
  and scalars, objects with `__array__` or the buffer protocol, `Tensor`.
- **dtype inference:** NumPy `float64`/`float32`/`bool_` map to the same
  Quabla dtype; integers → `float64` (no integer dtype, `docs/api.md:568-570`);
  `float16` → `TypeError`; Python floats and lists → `float64` (D3), except
  that a Python `bool` or a list whose elements are all `bool` gives `bool_`,
  as in NumPy. Before S1 a `float32` ndarray became `float64` (Exp 6).
  Zero-extent shapes stay unsupported, so empty inputs raise `ValueError`.
- **Fast path:** input through PyO3 `PyBuffer<f64>`/`PyBuffer<f32>` (safe
  C-order copy of any layout; `bool` data goes through
  `memoryview.tobytes()`, since PyO3 has no `bool` buffer element);
  non-native byte order is rejected explicitly, because PyO3 0.29 accepts
  `>` as native on little-endian hosts. Import always copies, so a tensor
  never aliases foreign memory. Output through `__array__(dtype=None,
  copy=None)` backed by a read-only buffer export of the immutable
  `Arc<Vec<f64>>` storage (`py/tensor.rs:14-24`). A `float64` export can be
  zero-copy while the exporter holds an `Arc` clone (an `unsafe`
  `__getbuffer__` with a `// SAFETY:` comment); `float32` exports copy,
  because host storage is `f64` (`docs/api.md:570`). NumPy stays optional
  (D2). `qb.array` of a `Tensor` returns a new object that shares the
  immutable storage.
- **`Tensor` stays the class name;** `qb.Array` is an ABC with `Tensor`,
  `TensorView`, and `TraceTensor` registered (D1).
- **Module-level ops** dispatch to the existing methods of `Tensor` and
  `TraceTensor` (as `py/lib.rs:129-140` does) and accept Python scalars:
  `abs cos exp log relu sigmoid sin softplus sqrt tanh maximum minimum sum
  mean max min norm any all matmul reshape transpose broadcast_to astype
  solve cholesky tril triu`, plus the kept `where`, `concat`, `stack`,
  `einsum`, comparisons, and logical ops, plus `qb.power`. The only new IR op
  is `Pow` (below); no `tan`. `abs`, `sum`, `max`, `min`, `any`, and `all`
  are attributes of `quabla` but not in `__all__`, so `from quabla import *`
  does not shadow the builtins.
- **Operators:** add `__neg__`, `__pow__`, and `__rpow__` to `TraceTensor`
  (Exp 2). Traced `x ** p` keeps using `powi` (`core/tensor_ir.rs:4406`) for
  a non-negative Python int `p` (exact and cheaper, like JAX's
  `integer_pow`) and lowers every other exponent, including an
  integer-valued float, a negative int, and a traced tensor exponent, to the
  elementwise `Pow` IR op (slice S1b); `c ** x` lowers through `__rpow__`.
  A Python number is a weak scalar that adopts the other operand's dtype, and
  `bool` operands are rejected as by the unary math ops. `Pow` follows the
  eager `f64::powf` semantics (`py/tensor.rs:1314`) with NaN for a negative
  base and a non-integer exponent, `0 ** 0 == 1`, and IEEE results for
  infinities and NaN, and differentiates in both operands: `d/dx = y
  x^(y-1)`, `d/dy = x^y ln x`. The base gradient is zero where `x == 0` and
  `y < 1` (the zero subgradient of `sqrt` at the origin, so `pow(x, 0.5)`
  and `sqrt(x)` agree there) and keeps the finite limit for `y >= 1`, so
  `x ** 2.0` still has second derivative 2 at zero. The exponent gradient is
  zero for every `x <= 0`: at `x == 0` the power is piecewise constant in
  `y`, and for `x < 0` no real derivative in `y` exists; unlike JAX, which
  returns NaN for `x < 0`, the zero keeps the backward pass finite for a
  learnable exponent at an integer value. The symbolic rules mask the
  singular points out of the inner power as well, so Hessians and HVPs stay
  finite there wherever the value is finite; at `x <= 0` the two mixed second
  partials follow these conventions and need not be equal. It lowers on CPU, CUDA (NVRTC `powf`, fused elementwise kernels and
  elementwise loop bodies), and MLX (`power`), and to `stablehlo.power` in the
  StableHLO export. Eager `Tensor ** Tensor`, including a single-element
  tensor exponent, is a broadcast elementwise power with the strict dtype
  promotion of the other binary ops; a tensor is never unwrapped into a
  scalar. `==` keeps identity semantics (`docs/api.md:156-157`).

### 3.3 Transforms

```python
qb.grad(fun, argnums=0, has_aux=False)            -> fun'(*args) -> grads
qb.value_and_grad(fun, argnums=0, has_aux=False)  -> fun'(*args) -> (value, grads)
qb.jvp(fun, primals, tangents)                    -> (out, tangent_out)
qb.vjp(fun, *primals, has_aux=False)              -> (out, vjp_fun)   # vjp_fun(ct) -> tuple
qb.jacobian(fun, argnums=0)                       -> fun'(*args) -> array[*out.shape, *in.shape]
qb.hessian(fun, argnums=0)                        -> fun'(*args) -> array[*in.shape, *in.shape]
qb.vmap(fun, in_axes=0, out_axes=0)               -> batched fun
qb.jit(fun, device=None, static_argnums=(), max_traces=8) -> compiled fun (+ .lower(...))
```

- **Arguments** are positional pytrees (§3.11). `argnums` is an int or tuple
  of ints; `grads` mirrors the pytree structure of the selected arguments.
  `bool_` leaves get no gradient, as today (`docs/api.md:176-178`).
- **`has_aux`:** `fun` returns `(scalar, aux_pytree)`. Needs multi-output
  tracing in the bridge (P8), compiled through the existing `compile_many`
  (`py/tensor_trace.rs:622-650`).
- **HVP** has no separate name: `qb.jvp(qb.grad(f), (x,), (v,))` lowers to
  symbolic VJP followed by a tangent-input JVP, which is the graph
  `build_vmap_hvp_scalar_graph` already builds (`py/tensor_trace.rs:6384-6457`).
- **`jacobian`/`hessian`** are dense and CPU-only in v0.2: per-column JVP as
  in `TensorJacobianFunction` (`py/tensor_trace.rs:4324-4370`) and the core
  `hessian_scalar` (`core/tensor_ir.rs:5536`). Inside `jit(device="cuda"|"mlx")`
  they raise `UnsupportedOperationError`.
- Every transformed function is staged: calling `qb.grad(f)(x)` without `jit`
  traces and compiles on the CPU, cached like `jit`. Eager op-by-op
  differentiation does not exist today and is not added.

### 3.4 Composition semantics

**Mechanism.** Called with concrete arrays, a transformed function traces a
fresh graph, transforms, compiles, and caches. Called with `TraceTensor`s
(inside an outer transform), it traces the inner function into a fresh graph
with the tracers' shapes, dtypes, and batch axes, transforms it, and
**inlines** the result into the caller's graph: callee inputs bind to the
caller's nodes and synthetic seeds (cotangent `1`) to constants. This is how
a derivative is used inside a loss without the graph handles of P4.

No inline or splice helper exists in `core/tensor_ir.rs` today. It is a
replay loop like the symbolic transforms: map each callee `Input` to its
binding and push `remap_tensor_op(op)` with `push_node`
(`core/tensor_ir.rs:12053, 6063`); region ops clone their compiled region
plans. Estimate: 200-300 lines of Rust plus tests (D8).

**Transform lowering.**

| Transform | Lowering |
| --- | --- |
| `grad` / `value_and_grad` | `symbolic_vjp` with a reserved cotangent input bound to `1.0` |
| `grad` w.r.t. a scalar leaf (shape `[]`, or a mapped scalar under `vmap`) | `symbolic_jvp` (ones tangent): forward mode, same result, cheaper; this is the README's path |
| `jvp` | `symbolic_jvp_with_tangent_inputs` |
| `vjp` | `symbolic_vjp`; `vjp_fun(ct)` executes the plan with `ct` bound |
| `vmap` | batched tracing (`py/tensor_trace.rs:4531`), then inlined with `batch_axis` cleared and `out_axes` applied |

**Supported and rejected compositions.**

| Expression | Maps to | v0.2 |
| --- | --- | --- |
| `grad(grad(f))`, scalar `f` | VJP over VJP; verified on CPU, `f'' = 1.6896969463271838` vs exact `...836` (Exp 3) | yes |
| `jvp(grad(f))` (HVP) | forward-over-reverse (Exp 3) | yes; MLX `vmap` HVP not lowered (`docs/api.md:579-580`) |
| `grad` of a loss that calls `vmap(grad(grad(u)), in_axes=(0, None))` (PINN) | JVP, JVP, then VJP: the README graph | yes |
| `vmap(grad(f))` w.r.t. a mapped argument | VJP with a ones cotangent over the batch; exact because mapped examples are independent (Exp 5: per-example `x` gradients) | yes |
| `vmap(grad(f), in_axes=(0, None))` w.r.t. the unmapped argument | JAX returns per-example gradients `[B, *w.shape]`; the vmap VJP sums them (Exp 5) | rejected with a pointer to `grad(lambda w: vmap(f)(x, w).sum())`; see Q2 |
| `grad(vmap(f))`-style aggregated losses | existing vmap VJP | yes |
| `vmap(vmap(f))` | `batch_axis` is one `Option<usize>` (`py/tensor_trace.rs:104`) | rejected; nested batching is Rust work |
| `vmap` over `cond` | rejected at trace time (`py/tensor_trace.rs:4421-4426`) | rejected |
| `grad`/HVP through `fori_loop`/`scan` | region VJP and forward-over-reverse | yes (CUDA: elementwise bodies only) |
| second-order reverse through regions | explicit error (`docs/api.md:577-578`) | rejected |
| `jacobian`/`hessian` under device `jit` | no device lowering | rejected |

### 3.5 Tracing and caching

- **Cache key:** `(treedef, per-leaf (shape, dtype), static argument values,
  device, transform config)`. Dtype is part of the key, which fixes the
  float32-to-float64 widening of the batch helpers (§2.2 item 6) for the new
  API only; the deprecated helpers keep their behaviour.
- **Python scalars** in argument positions are static: they are part of the
  key and become weak constants. Graph inputs are always strong
  (`core/tensor_ir.rs:3424`), so a traced Python float would be a strong
  `float64` input and would break `float32` programs with the mixed-dtype
  error (`docs/api.md:140-141`). To vary a scalar without retracing, pass
  `qb.array(v, dtype=...)` (Decision D7).
- **Bound:** `max_traces=8` per function (batch helpers use 4,
  `py/tensor_trace.rs:5236`); overflow raises `RetraceLimitError` listing the
  keys, never evicts silently (`docs/api.md:268-270`). A new batch size is
  just a new key.
- **AOT:** `qb.jit(f, device=...).lower(*args)` accepts arrays or
  `qb.ShapeDtype(shape, dtype)` leaves and returns a `Lowered` with
  `.program` (a facade `Program`), `.as_text()` (`Program.lower_text`,
  `py/compiler.rs:114-121`), and `.compile()`. `Compiler`/`Program`/`Executable`
  remain for name-keyed, spec-first use (`py/compiler.rs:79-165`).
- **Tracer errors:** control flow on a tracer (`py/tensor_trace.rs:1916-1920`)
  becomes a `TracerError` (a `TypeError`) pointing to `qb.where`/`qb.cond`.

### 3.6 Devices

```python
qb.devices() -> list[str]          # built targets, from Compiler.capabilities (py/compiler.rs:92-98)
qb.jit(f, device="cpu" | "cuda" | "cuda:1" | "mlx")
```

- `device=None` means `"cpu"`; no automatic GPU selection. A target missing
  from the build raises at compile time (`core/compiler.rs:363`).
- Eager ops run on the host; only `jit` targets a device, uploading inputs
  and downloading outputs per call as today (`docs/api.md:389-391`).
- `float64` programs run as `f32` on CUDA and MLX (README Backend Support).
  `jit` on a device with `float64` inputs emits one `UserWarning` per
  function (Q6).
- MLX entrypoints must take the process-wide MLX lock once
  (`core/tensor_ir/mlx.rs:43`); new bridge entrypoints follow that rule.
- Optional later slice (S10): skip re-upload of inputs that are the same
  immutable `Tensor` object as on the previous call, using the existing
  retained-input machinery (`py/tensor_trace.rs:2649` for MLX,
  `py/tensor_trace.rs:2893` for CUDA).

### 3.7 Device-resident training

`jit(value_and_grad(loss), device="mlx")` returns host values every call,
which is slower than the existing fused executors that keep parameters and
Adam moments on the device. v0.2 exposes those executors as one object:

```python
trainer = qb.optim.Trainer(loss, params, optimizer, *data, device="mlx",
                           batch_argnums=())   # data positions replaced per step
trainer.step(*batch)        # no host readback
trainer.loss() -> Tensor    # explicit readback
trainer.params -> pytree    # explicit readback
```

| `device` | Backing executor | Notes |
| --- | --- | --- |
| `cpu` | `jit(value_and_grad(loss))` + host optimizer update | same interface, host math |
| `mlx` | `mlx_adam_loss_optimizer` (`py/tensor_trace.rs:6069`) | Adam only |
| `cuda` | `cuda_adam_loss_optimizer` (`py/tensor_trace.rs:7041`) | Adam only |

`Trainer` traces `loss(params, *data)` once with the new tracer, names
parameter leaves by pytree path, and passes the resulting trace plus those
names to the factory. Data leaves not listed in `batch_argnums` become
retained inputs; batch leaves go through `step(inputs)`, which replaces them
without resetting state (`py/tensor_trace.rs:4221-4227, 6825-6835`). A
no-argument `step()` does no Python-side flattening, so per-step cost is the
current Rust call. Baseline on this machine: 660 µs per MLX step and 17 µs
per CPU step for the README problem (Exp 4). Device SGD is rejected in v0.2:
CUDA has only a per-plan `sgd_step` (`py/tensor_trace.rs:2943`), outside the
shared union plan (`py/tensor_trace.rs:6797`), and MLX has none
(`core/tensor_ir/mlx.rs:94-104`). See D10 and D11.

### 3.8 Optimizers

```python
opt = qb.optim.Adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8)
opt = qb.optim.SGD(learning_rate=1e-2)
state = opt.init(params)                        # pytree of moments (host)
params, state = opt.update(params, grads, state)  # pure, pytree-aware
```

Optimizer objects are hyperparameter specs: pure host updates for custom
loops, and the configuration `Trainer` lowers to device plans. The existing
`quabla.Adam` becomes a deprecated alias of `quabla.optim.Adam` that keeps
its stateful `step(params_dict, grads_dict)` (`py/optim.rs:99-166`).

### 3.9 Control flow

```python
qb.cond(pred, true_fun, false_fun, *operands)
qb.fori_loop(lower, upper, body_fun, init_val, *, operands=(), unroll=False)
qb.scan(f, init, *, length, operands=(), unroll=False)   # f(carry, i, *operands) -> (carry, y)
```

| New | Maps to | Differences from JAX |
| --- | --- | --- |
| `cond` | `tensor_cond(pred, t, f, operands)` (`py/tensor_trace.rs:4407`) | same signature; a concrete `pred` just calls one branch |
| `fori_loop` | `tensor_fori_loop_region` (`py/tensor_trace.rs:4652`); `unroll=True` → `tensor_fori_loop` | body gets `(i, carry, *operands)`; captures must be explicit |
| `scan` | `tensor_scan_region` (`py/tensor_trace.rs:4788`); `unroll=True` → `tensor_scan` | no `xs`: the per-step input is the index (dynamic indexing is unsupported, `docs/api.md:571-573`); single-array carry |

Region bodies trace in a fresh graph, so closing over an outer tracer fails
with a "different graph" error; v0.2 makes it a `TracerError` naming
`operands=`. Implicit capture (medium) and tuple carries (significant) are
later Rust work (D13).

### 3.10 Data parallel

```python
qb.distributed.value_and_grad(fun, *, devices, shard_argnums, argnums=0, reduction="mean")
```

This maps to `tensor_value_and_grad_data_parallel_cuda_fn(function,
input_specs, parameter_names, mapped_input_names, device_ordinals,
reduction)` (`py/tensor_trace.rs:5763-5764`). It keeps the current contract:
single node, equal axis-zero shards, replicated parameter gradients only, and
a host optimizer. It is documented as experimental (Decision D12).

### 3.11 Pytrees

- Containers: `dict` (string keys, traversed in sorted order), `list`,
  `tuple`, `None` (empty). Leaves: `Tensor`, `TensorView`, NumPy arrays,
  Python numbers (static, §3.5).
- IR input names are key paths (`params/layers/0/w`), which keeps
  `lower_text()` readable and plugs into the name-keyed device executors.
  Names starting with `__quabla_` are reserved (e.g.
  `py/tensor_trace.rs:6094-6098`).
- `quabla.tree.{flatten, unflatten, map}` are public; NamedTuples,
  dataclasses, and registration are out of scope (D6). Flatten plus key costs
  about 3 µs for 8 leaves (Exp 4).

### 3.12 Errors

```text
QuablaError(Exception)
├── UnsupportedOperationError(QuablaError, ValueError, NotImplementedError)  .op, .device
├── TracerError(QuablaError, TypeError)
└── RetraceLimitError(QuablaError, ValueError)
```

Each class also subclasses the builtin raised today, so existing
`except ValueError`/`except TypeError` handlers keep working. The core
reports errors as `String` (e.g. `core/tensor_ir/mlx.rs:953-955`), so
`UnsupportedOperationError` needs a typed rejection at the lowering sites
(about 18 "unsupported" sites across `core/tensor_ir.rs`, `cuda.rs`, and
`mlx.rs` by grep) and an eager MLX validation pass at compile time, since MLX
lowering is lazy today (`core/compiler.rs:380`). Estimate: small to medium
Rust work (Decision D16).

## 4. Decisions

Format: options, then **recommendation** and reason.

**D1. Rename `Tensor` to `Array`?** (a) Keep `Tensor` and add a `qb.Array`
ABC registered for `Tensor`/`TensorView`/`TraceTensor`; (b) rename with a
`Tensor` alias; (c) keep `Tensor` only. **(a):** no churn in 272 tests and all
examples, "tensor" is familiar to PINN users, and the ABC gives `isinstance`
and annotations without PyO3 `extends` changing struct layouts.

**D2. NumPy bridge.** (a) Buffer protocol plus `__array__`; (b) the
`rust-numpy` crate; (c) pure Python via `to_flat_list`. **(a):** no NumPy
C-API build dependency and NumPy stays optional; (c) costs 29 ms per 1e6
values (Exp 6); (b) ties builds to NumPy's ABI.

**D3. Default dtype of new arrays.** (a) Infer from NumPy, Python data →
`float64`; (b) `float32` everywhere (JAX); (c) a global switch. **(a):** it
matches today's constructor default (`py/tensor.rs:1518-1520`) and the `f64`
CPU reference, so no existing result changes; (b) silently changes CPU
numerics; (c) can be added later.

**D4. Where math functions live.** (a) Top level (`qb.sin`, MLX/Torch style);
(b) `quabla.numpy`; (c) methods only. **(a):** one array type and no
NumPy-compatibility promise (`docs/design.md:42`); methods stay.

**D5. Argument selection.** (a) `argnums` over positional pytrees; (b)
argument names; (c) both. **(a):** JAX-compatible and spec-free; names live on
internally as pytree paths, and `Compiler` keeps name-keyed specs.

**D6. Pytree scope.** (a) dict/list/tuple/None; (b) plus NamedTuple and
dataclasses; (c) user registration. **(a):** covers parameter dicts and layer
lists; (b) and (c) can follow without breaking anything.

**D7. Cache key and Python scalars.** (a) Key on treedef, shapes, dtypes,
static values, and device, with Python scalars static; (b) the same with
scalars traced as rank-0 inputs; (c) shapes only. **(a), with `max_traces=8`
and `RetraceLimitError`:** (b) needs weak input nodes
(`core/tensor_ir.rs:3424`) or breaks `float32` programs; (c) repeats the batch
helpers' dtype widening.

**D8. Composition mechanism.** (a) Inline the transformed callee graph (new
core primitive); (b) transform the outer graph in place and remap node ids;
(c) keep graph handles. **(a):** tracers hold raw node ids
(`py/tensor_trace.rs:97-105`) that (b) would invalidate, and (c) is P4. (a)
reuses `remap_tensor_op`/`push_node`, about 200-300 lines of Rust.

**D9. Device API.** (a) `jit(f, device=...)`, default `"cpu"`; (b) a global
or context-manager default; (c) per-array placement. **(a) for 0.2:** it
follows the README's "explicit rather than silent"; (b) can be layered on
later (Q4); (c) needs device arrays, which do not exist.

**D10. Optimizer API style.** (a) Hyperparameter objects with pure
`init`/`update`, consumed by `Trainer`; (b) stateful Torch-style `step`; (c)
optax-style gradient transformations. **(a):** pure updates compose with CPU
`jit`, and the same object configures device plans; (c) is more machinery
than two optimizers need.

**D11. Device-resident training.** (a) A stateful `optim.Trainer` over the
existing Adam executors; (b) a functional `state = step(state)` handle over
the same mutable device state; (c) device arrays plus `jit(train_step)` with
buffer donation. **(a) now, (c) long term:** (a) keeps today's per-step cost;
(b) looks pure but is not; (c) needs a device array type and plan binding by
handle, several weeks of Rust.

**D12. Data-parallel placement.** (a) `quabla.distributed.value_and_grad`;
(b) `jit(..., devices=[0, 1])`; (c) only the deprecated helper. **(a):** (b)
implies general sharding that does not exist (`docs/api.md:582-584`).

**D13. Control-flow namespace and captures.** (a) Top-level `cond`,
`fori_loop`, `scan` with explicit `operands=`; (b) `quabla.lax`; (c) (a) plus
implicit closure capture. **(a) in 0.2, (c) later:** "lax" means nothing
outside JAX, and implicit capture is Rust work.

**D14. Deprecation mechanism and timeline.** (a) PEP 562 module
`__getattr__`: `DeprecationWarning` once per name from 0.2, legacy names moved
to `quabla.legacy`, deprecated names removed at 1.0; (b) Rust
`PyErr::warn` per function; (c) documentation only. **(a):** covers classes
and functions, zero hot-path cost, no Rust edits. `CONTRIBUTING.md:8-9` ("change
without deprecation periods") and `CHANGELOG.md:6` must state the new policy.

**D15. Implementation layer.** (a) A pure-Python package `python/quabla/`
over the extension renamed `quabla._quabla`; (b) everything in PyO3. **(a):**
pytrees, caching, dispatch, and warnings are Python work, and the hot path
stays in Rust. It needs `python-source = "python"` and
`module-name = "quabla._quabla"` (`pyproject.toml:52-55`) and
`#[pymodule(name = "_quabla")]` (`py/lib.rs:360`).

**D16. Error model.** (a) Classes that subclass today's builtins, with typed
lowering rejections in Rust; (b) message-prefix matching in Python; (c) no
change. **(a):** (b) is fragile and (c) leaves P9.

**D17. `grad`/`jit`/`trace` collide with legacy.** (a) Dispatch on the call
shape: legacy `grad(fn, specs, values, ct)` takes four positional arguments
(`py/trace.rs:3320-3326`) and legacy `jit(specs)` takes a list, not a
callable (`py/trace.rs:3444`); the legacy form warns, and `trace` stays
legacy-only. (b) Rename the new functions. (c) Break the legacy forms.
**(a):** the signatures cannot be confused, and nothing breaks.

## 5. Compatibility and Migration

### 5.1 Status vocabulary (Appendix A)

| Status | 0.2 behaviour | 1.0 |
| --- | --- | --- |
| core | part of the new API, no warning | kept |
| advanced | kept for explicit compiler or device control, no warning | kept |
| deprecated | unchanged behaviour; `DeprecationWarning` once per name on first attribute access | removed |
| legacy | moved to `quabla.legacy.<name>` (no warning there); top-level access warns | top-level removed; `quabla.legacy` kept pending Q9 |

Old names reach the same Rust functions, so results, error types, and
messages are unchanged. Result types warn only on explicit top-level access.

### 5.2 Migration rules

| Old pattern | New pattern |
| --- | --- |
| `specs = [("x", [4], quabla.float32), ...]` | pass arrays; dtype comes from the data |
| `fn(inputs_dict)` with name keys | `fn(*args)` with pytrees; names only in `Compiler` |
| `tensor_X_fn(f, specs)` / `_cuda_fn` / `_mlx_fn` | `qb.jit(qb.X(f), device=...)` |
| `tensor_X_batch_*_fn(f, names, max_specializations=4)` | `qb.jit(qb.X(f), device=..., max_traces=4)` |
| `traced.symbolic_jvp("x").symbolic_jvp("x")` + `graph.input` | `qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None))` inside the loss |
| `plan.evaluate_value_and_vjp(inputs, Tensor([], [1.0]))` | `qb.value_and_grad(loss)(...)` |
| `mlx_adam_loss_optimizer(loss, names, inputs, lr, retained)` | `qb.optim.Trainer(loss, params, qb.optim.Adam(lr), *data, device="mlx")` |
| `t.to_flat_list()` | `t.numpy()`, `t.tolist()`, `t.item()` |
| list-of-lists results from `tensor_jacobian_fn`/`tensor_hessian_scalar_fn` | a `Tensor` shaped `[*out, *in]` / `[*in, *in]` |

### 5.3 Documentation changes

README "At a Glance" and "Quickstart" become the §2.4 versions. `docs/api.md`
gains a leading "Core API" section; the rank-N helper text moves under
"Compatibility API (deprecated)" and the facade under "Advanced". The PINN
examples and `plot_readme_figure.py` move to the new API; benchmarks switch
after S6 shows parity. CHANGELOG `[0.2.0]` links Appendix A under Deprecated.

## 6. Implementation Plan

Each slice is releasable on its own and keeps `tests/python/test_matrix.py`
(272 tests) passing unchanged. New tests go in `tests/python/test_api.py`.
The gates are those under "Development Gates" (`CONTRIBUTING.md:18-35`), with
`ruff check` extended to `python/`.

| Slice | Scope | Rust? | Files | Verification | Risks |
| --- | --- | --- | --- | --- | --- |
| **S0 Packaging** | `python/quabla/__init__.py` re-exports all 132 names from `quabla._quabla`; `quabla.quabla` aliased (attribute and `sys.modules` entry); class `__module__` values unchanged without `#[pyclass(module = ...)]` edits, because PyO3 does not derive them from the extension's module name (`dtype` already reports `quabla`, the other classes `builtins`); module-level functions report `quabla._quabla` instead of `quabla.quabla` | bridge (module name only) | `pyproject.toml`, `py/lib.rs`, `python/quabla/__init__.py`, CI ruff path | a name-set test asserts the v0.1 names are a subset of `dir(quabla)`; old tests; `maturin build --sdist` then install; macOS MLX and Linux CI | maturin `python-source` with a `cdylib` named `quabla`; stale `.so` in dev venvs |
| **S1 Arrays** | `array`, `asarray`, factories, `numpy()`, `tolist()`, `item()`, `__float__`, `__array__`, buffer import/export, `Array` ABC, module ops, tracer `__neg__`/`__pow__` | bridge | `py/tensor.rs`, `py/interop.rs`, `py/tensor_trace.rs`, `python/quabla/_array.py`, `python/quabla/_ops.py` | dtype-mapping table tests; NumPy round trip of 1e6 values under 5 ms; `unsafe` buffer export with `// SAFETY:` and a refcount test | buffer lifetime soundness; NumPy 1.x vs 2.x `__array__(copy=)` |
| **S1b Pow op** | elementwise `Pow` IR op with JVP/VJP/HVP rules, CPU/CUDA/MLX lowering, fusion and CSE handling, `qb.power`, tracer `x ** y` for non-integer and tensor exponents | **core** | `core/tensor_ir.rs`, `core/tensor_ir/{cuda,mlx}.rs`, `py/tensor_trace.rs` | finite-difference checks for both operands incl. negative base, zero base, and integer-valued float exponents; CPU/MLX/CUDA parity; `powi` path unchanged for integer exponents. Landed: the conventions of §3.2, eager `Tensor ** Tensor` and `c ** Tensor`, and `qb.power` | NaN/inf conventions at `x <= 0`; derivative at `x == 0`; the symbolic Hessian of a loss whose value is infinite is NaN, because the reverse pass of `sum` broadcasts with `powi(v - v, 0)` (predates `Pow`) |
| **S2 CPU transforms** | `grad`, `value_and_grad`, `jvp`, `vjp`, `jacobian`, `hessian`, `jit(device="cpu")`, pytrees, `argnums`, `has_aux`, cache and `RetraceLimitError`, `quabla.tree` | bridge (tuple outputs from traces) | `py/tensor_trace.rs:4375-4399`, `python/quabla/_transforms.py` | parity against `tensor_*_fn` on shared fixtures; retrace-bound tests; float32 keeps float32 | Python overhead on tiny graphs (3 µs vs 17 µs, Exp 4) |
| **S3 Composition** | `TensorIr::inline` + bridge binding; nested transforms; README PINN example | **core** | `core/tensor_ir.rs`, `py/tensor_trace.rs`, `python/quabla/_transforms.py` | Rust tests for inline, including `Cond`/`Fori`/`Scan` regions; `grad(grad)` vs exact; README example reproduces `w = 3.141593` | shared-subexpression duplication after inlining (freeze already commons pure nodes, `docs/api.md:352-353`) |
| **S4 vmap** | `vmap(in_axes, out_axes)` over pytrees; `vmap(grad)`; rejection of unmapped per-example gradients and nested vmap | bridge | `py/tensor_trace.rs:4490-4594` | parity with `tensor_vmap_{,jvp_,vjp_,hvp_scalar_}fn` | `out_axes` pytrees vs the single `out_axis` today |
| **S5 Devices and errors** | `jit(device=...)`, `"cuda:N"`, `devices()`, `lower()`/`ShapeDtype`, error hierarchy, typed lowering rejections, eager MLX validation, `float64`-on-device warning | **core** (errors, MLX validation) | `core/compiler.rs`, `core/tensor_ir/{cuda,mlx}.rs`, `py/compiler.rs` | `QUABLA_MLX_TEST=1` suite on macOS; `QUABLA_CUDA_TEST=1` on the CUDA host; unbuilt-target error test in CI | MLX lock discipline for new entrypoints (`core/tensor_ir/mlx.rs:43`) |
| **S6 Optim and Trainer** | `optim.Adam`/`SGD` (pure), `Trainer` on CPU/MLX/CUDA; `quabla.Adam` alias | none | `python/quabla/optim.py` | `benchmark_pinn_mlx.py` and `benchmark_pinn_cuda.py` with old and new APIs on the same host: step time within 2%; convergence parity with `examples/pinn_poisson_mlx.py` | parameter naming mismatches between pytree paths and the factories' name lists |
| **S7 Control flow** | `cond`, `fori_loop`, `scan` wrappers and the capture error | bridge (error text) | `python/quabla/_control.py`, `py/tensor_trace.rs:4407-4960` | parity with the region tests; CUDA elementwise loop VJP/HVP on the CUDA host | JAX argument order (`scan` body gets `(carry, i)`) |
| **S8 Distributed** | `quabla.distributed.value_and_grad` | none | `python/quabla/distributed.py` | two-GPU NCCL run of the `validate_data_parallel_cuda.py` equivalent on the GPU cluster | NCCL host availability |
| **S9 Deprecation and docs** | `__getattr__` warnings, `quabla.legacy`, `grad`/`jit` dual dispatch, README/api.md/examples/CHANGELOG/CONTRIBUTING | none | `python/quabla/{__init__,legacy,_compat}.py`, docs | each deprecated name warns exactly once and returns the v0.1 object; the old suite passes with warnings ignored; README snippets run as tests | warnings in users' `-W error` CI (intended) |
| S10 (optional) | identity-based input retention in device `jit` | bridge | `py/tensor_trace.rs:2634-2660, 2893-2941` | benchmark shows no re-upload for static inputs | cache invalidation if storage ever becomes mutable |

Rust summary: S1b (`Pow`), S3 (inline), and S5 (typed errors, MLX
validation) touch the core; S0, S1, S2, S4, and S7 touch only the bridge; S6, S8, and S9 are pure
Python. The critical path is S0 → S2 → S3. S1 can run in parallel with S2,
and S5 through S8 are independent after S3.

## 7. Resolved Questions

Resolved by the owner on 2026-09-29. Items marked "as recommended" follow
this document's proposal without a separate owner discussion.

1. **PINN idiom:** JAX semantics only. The canonical form is
   `vmap(grad(grad(u)), in_axes=(0, None))` over a per-point `u`; no
   `elementwise_grad` shortcut ships, because it is silently wrong when points
   are coupled.
2. **Per-example gradients w.r.t. unmapped arguments** under `vmap`:
   rejected with an explicit error in v0.2 (as recommended).
3. **Python scalars:** static (as recommended).
4. **Device selection:** explicit `device=` only in v0.2; no global default
   device (as recommended by D9).
5. **Traced float exponents:** a real `Pow` IR op (owner decision; slice S1b).
6. **`float64` on devices:** warn once per `jit` (as recommended).
7. **Device-resident training:** a stateful `optim.Trainer` wrapping the
   existing MLX/CUDA Adam executors (owner decision); device arrays with
   donation remain a later design.
8. **Release vehicle:** no 0.1.1; S0 ships inside v0.2, since the repository
   is not public yet.
9. **`quabla.legacy` at 1.0:** decided at 1.0; kept through 0.x.
10. **Tracer display name:** keep `TraceTensor` (as recommended).

## Appendix A: Name Mapping

All 132 names from `dir(quabla)` on the v0.1 build, in `dir()` order.

| v0.1 name | Status | v0.2 equivalent |
| --- | --- | --- |
| `Adam` | deprecated | alias of `qb.optim.Adam` (keeps `.step(params, grads)`) |
| `Compiler` | advanced | unchanged |
| `CpuExecutionPlan` | legacy | `quabla.legacy.CpuExecutionPlan` |
| `Executable` | advanced | unchanged |
| `GradFunction` | legacy | `quabla.legacy.GradFunction` |
| `GradScalarFunction` | legacy | `quabla.legacy.GradScalarFunction` |
| `GradScalarTransform` | legacy | `quabla.legacy.GradScalarTransform` |
| `JacobianFunction` | legacy | `quabla.legacy.JacobianFunction` |
| `JacobiansFunction` | legacy | `quabla.legacy.JacobiansFunction` |
| `JitFunction` | legacy | `quabla.legacy.JitFunction` |
| `JitTransform` | legacy | `quabla.legacy.JitTransform` |
| `JvpFunction` | legacy | `quabla.legacy.JvpFunction` |
| `Matrix` | legacy | `quabla.legacy.Matrix` |
| `Program` | advanced | unchanged (`jit(f).lower(...).program`) |
| `Tensor` | core | unchanged; `qb.array`/`qb.asarray` construct it |
| `TensorBatchCudaJitFunction` | deprecated | result type of `tensor_jit_batch_cuda_fn` |
| `TensorBatchCudaValueAndGradFunction` | deprecated | result type of `tensor_value_and_grad_batch_cuda_fn` |
| `TensorBatchJitFunction` | deprecated | result type of `tensor_jit_batch_fn` |
| `TensorBatchMlxValueAndGradFunction` | deprecated | result type of `tensor_value_and_grad_batch_mlx_fn` |
| `TensorBatchValueAndGradFunction` | deprecated | result type of `tensor_value_and_grad_batch_fn` |
| `TensorCondFunction` | deprecated | result type of `tensor_cond_fn` |
| `TensorCondJvpFunction` | deprecated | result type of `tensor_cond_jvp_fn` |
| `TensorCondValueAndGradFunction` | deprecated | result type of `tensor_cond_value_and_grad_fn` |
| `TensorCpuExecutionPlan` | advanced | plan class (`kernel_ir`, `buffer_plan`, `value_and_grad_data_parallel`) |
| `TensorCudaAdamOptimizer` | advanced | executor behind `Trainer(device="cuda")` |
| `TensorCudaExecutionPlan` | advanced | plan class (`evaluate_device`, `benchmark_device`, `sgd_step`, `adam_step`) |
| `TensorCudaValueAndGradFunction` | deprecated | result type of `tensor_value_and_grad_cuda_fn` |
| `TensorGradScalarFunction` | deprecated | result type of `tensor_grad_scalar_fn` |
| `TensorHessianScalarFunction` | deprecated | result type of `tensor_hessian_scalar_fn` |
| `TensorHvpScalarFunction` | deprecated | result type of `tensor_hvp_scalar_fn` |
| `TensorJacobianFunction` | deprecated | result type of `tensor_jacobian_fn` |
| `TensorJitFunction` | deprecated | result type of `tensor_jit_fn` |
| `TensorJvpFunction` | deprecated | result type of `tensor_jvp_fn` |
| `TensorMlxAdamOptimizer` | advanced | executor behind `Trainer(device="mlx")` |
| `TensorMlxExecutionPlan` | advanced | plan class (`retain_inputs`, `evaluate_device`) |
| `TensorMlxValueAndGradFunction` | deprecated | result type of `tensor_value_and_grad_mlx_fn` |
| `TensorTraceGraph` | deprecated | composition (§3.4); `Program` for inspection |
| `TensorTraceResult` | deprecated | `Program` |
| `TensorValueAndGradFunction` | deprecated | result type of `tensor_value_and_grad_fn` |
| `TensorView` | core | unchanged |
| `TensorVjpFunction` | deprecated | result type of `tensor_vjp_fn` |
| `TensorVmapCudaFunction` | deprecated | result type of `tensor_vmap_cuda_fn` |
| `TensorVmapCudaHvpScalarFunction` | deprecated | result type of `tensor_vmap_hvp_scalar_cuda_fn` |
| `TensorVmapCudaJvpFunction` | deprecated | result type of `tensor_vmap_jvp_cuda_fn` |
| `TensorVmapCudaVjpFunction` | deprecated | result type of `tensor_vmap_vjp_cuda_fn` |
| `TensorVmapFunction` | deprecated | result type of `tensor_vmap_fn` |
| `TensorVmapHvpScalarFunction` | deprecated | result type of `tensor_vmap_hvp_scalar_fn` |
| `TensorVmapJvpFunction` | deprecated | result type of `tensor_vmap_jvp_fn` |
| `TensorVmapMlxFunction` | deprecated | result type of `tensor_vmap_mlx_fn` |
| `TensorVmapMlxJvpFunction` | deprecated | result type of `tensor_vmap_jvp_mlx_fn` |
| `TensorVmapMlxVjpFunction` | deprecated | result type of `tensor_vmap_vjp_mlx_fn` |
| `TensorVmapVjpFunction` | deprecated | result type of `tensor_vmap_vjp_fn` |
| `TraceGraph` | legacy | `quabla.legacy.TraceGraph` |
| `TraceMatrix` | legacy | `quabla.legacy.TraceMatrix` |
| `TraceResult` | legacy | `quabla.legacy.TraceResult` |
| `TraceTensor` | core | tracer type inside transforms; registered as `qb.Array` |
| `ValueAndGradFunction` | legacy | `quabla.legacy.ValueAndGradFunction` |
| `VjpFunction` | legacy | `quabla.legacy.VjpFunction` |
| `bool_` | core | unchanged |
| `concat` | core | unchanged |
| `cuda_adam_loss_optimizer` | deprecated | `qb.optim.Trainer(..., device="cuda")` |
| `cuda_adam_optimizer` | deprecated | `qb.optim.Trainer(..., device="cuda")` |
| `cuda_adam_step` | deprecated | `qb.optim.Trainer(..., device="cuda")` |
| `cuda_adam_vjp_optimizer` | deprecated | `qb.optim.Trainer(..., device="cuda")` |
| `dtype` | core | unchanged |
| `einsum` | core | unchanged |
| `equal` | core | unchanged |
| `float32` | core | unchanged |
| `float64` | core | unchanged |
| `grad` | core + legacy | `qb.grad(fun, argnums)`; the 4-argument legacy form dispatches to `quabla.legacy.grad` with a warning (D17) |
| `grad_fn` | legacy | `quabla.legacy.grad_fn` |
| `grad_scalar` | legacy | `quabla.legacy.grad_scalar` |
| `grad_scalar_fn` | legacy | `quabla.legacy.grad_scalar_fn` |
| `greater` | core | unchanged |
| `greater_equal` | core | unchanged |
| `isfinite` | core | unchanged |
| `isnan` | core | unchanged |
| `jacobian_fn` | legacy | `quabla.legacy.jacobian_fn` |
| `jacobians_fn` | legacy | `quabla.legacy.jacobians_fn` |
| `jit` | core + legacy | `qb.jit(fun, device=...)`; the `jit(input_specs)` form dispatches to `quabla.legacy.jit` with a warning (D17) |
| `jvp_fn` | legacy | `quabla.legacy.jvp_fn` |
| `less` | core | unchanged |
| `less_equal` | core | unchanged |
| `logical_and` | core | unchanged |
| `logical_not` | core | unchanged |
| `logical_or` | core | unchanged |
| `mlx_adam_loss_optimizer` | deprecated | `qb.optim.Trainer(..., device="mlx")` |
| `not_equal` | core | unchanged |
| `quabla` | deprecated | alias of the native module `quabla._quabla` |
| `stack` | core | unchanged |
| `sum_gradients` | deprecated | `qb.tree.map(operator.add, a, b)` |
| `tensor_cond` | deprecated | `qb.cond(pred, true_fun, false_fun, *operands)` |
| `tensor_cond_fn` | deprecated | Python `if` on a concrete flag + `qb.jit(branch)` |
| `tensor_cond_jvp_fn` | deprecated | Python `if` + `qb.jvp(branch, ...)` |
| `tensor_cond_value_and_grad_fn` | deprecated | Python `if` + `qb.value_and_grad(branch)` |
| `tensor_fori_loop` | deprecated | `qb.fori_loop(..., unroll=True)` |
| `tensor_fori_loop_region` | deprecated | `qb.fori_loop(lower, upper, body, init, operands=...)` |
| `tensor_grad_scalar_fn` | deprecated | `qb.grad(f, argnums=(0, ..., n-1))` |
| `tensor_hessian_scalar_fn` | deprecated | `qb.hessian(f, argnums=i)` (returns a `Tensor`) |
| `tensor_hvp_scalar_fn` | deprecated | `qb.jvp(qb.grad(f), (x,), (v,))` |
| `tensor_jacobian_fn` | deprecated | `qb.jacobian(f, argnums=i)` (returns a `Tensor`) |
| `tensor_jit_batch_cuda_fn` | deprecated | `qb.jit(f, device="cuda", max_traces=4)` |
| `tensor_jit_batch_fn` | deprecated | `qb.jit(f, max_traces=4)` |
| `tensor_jit_cuda_fn` | deprecated | `qb.jit(f, device="cuda")` |
| `tensor_jit_fn` | deprecated | `qb.jit(f)` |
| `tensor_jvp_fn` | deprecated | `qb.jvp(f, primals, tangents)` |
| `tensor_scan` | deprecated | `qb.scan(f, init, length=n, unroll=True)` |
| `tensor_scan_region` | deprecated | `qb.scan(f, init, length=upper-lower, operands=...)` |
| `tensor_value_and_grad_batch_cuda_fn` | deprecated | `qb.jit(qb.value_and_grad(f, argnums), device="cuda", max_traces=4)` |
| `tensor_value_and_grad_batch_fn` | deprecated | `qb.jit(qb.value_and_grad(f, argnums), max_traces=4)` |
| `tensor_value_and_grad_batch_mlx_fn` | deprecated | `qb.jit(qb.value_and_grad(f, argnums), device="mlx", max_traces=4)` |
| `tensor_value_and_grad_cuda_fn` | deprecated | `qb.jit(qb.value_and_grad(f, argnums), device="cuda")` |
| `tensor_value_and_grad_data_parallel_cuda_fn` | deprecated | `qb.distributed.value_and_grad(f, devices=..., shard_argnums=...)` |
| `tensor_value_and_grad_fn` | deprecated | `qb.value_and_grad(f, argnums)` |
| `tensor_value_and_grad_mlx_fn` | deprecated | `qb.jit(qb.value_and_grad(f, argnums), device="mlx")` |
| `tensor_vjp_fn` | deprecated | `qb.vjp(f, *primals)` |
| `tensor_vmap_cuda_fn` | deprecated | `qb.jit(qb.vmap(f, in_axes, out_axes), device="cuda")` |
| `tensor_vmap_fn` | deprecated | `qb.vmap(f, in_axes, out_axes)` |
| `tensor_vmap_hvp_scalar_cuda_fn` | deprecated | `qb.jit(<vmap HVP composition>, device="cuda")` |
| `tensor_vmap_hvp_scalar_fn` | deprecated | `qb.jvp(qb.grad(lambda x: qb.sum(qb.vmap(f)(x))), (x,), (v,))` |
| `tensor_vmap_jvp_cuda_fn` | deprecated | `qb.jit(<jvp of vmap>, device="cuda")` |
| `tensor_vmap_jvp_fn` | deprecated | `qb.jvp(qb.vmap(f, ...), primals, tangents)` |
| `tensor_vmap_jvp_mlx_fn` | deprecated | `qb.jit(<jvp of vmap>, device="mlx")` |
| `tensor_vmap_mlx_fn` | deprecated | `qb.jit(qb.vmap(f, in_axes, out_axes), device="mlx")` |
| `tensor_vmap_vjp_cuda_fn` | deprecated | `qb.jit(<vjp of vmap>, device="cuda")` |
| `tensor_vmap_vjp_fn` | deprecated | `qb.vjp(qb.vmap(f, ...), *primals)` |
| `tensor_vmap_vjp_mlx_fn` | deprecated | `qb.jit(<vjp of vmap>, device="mlx")` |
| `trace` | legacy | `quabla.legacy.trace` |
| `trace_tensor` | deprecated | `qb.jit(f).lower(...)` or `Compiler().trace(f, specs)` |
| `value_and_grad_fn` | legacy | `quabla.legacy.value_and_grad_fn` |
| `vjp_fn` | legacy | `quabla.legacy.vjp_fn` |
| `where` | core | unchanged |

Totals: core 22, core + legacy dispatch 2 (`grad`, `jit`), advanced 8,
deprecated 76, legacy 24.

## Appendix B: Experiments

All experiments ran with `.venv/bin/python` (Python 3.14, NumPy 2.5.3, MLX
build, `Compiler().capabilities() == {'cpu': True, 'cuda': False, 'mlx': True}`)
from the session scratchpad; no repository file was changed.

| Exp | What | Result |
| --- | --- | --- |
| 1 | `Tensor` construction and conversion | `Tensor([[1,2],[3,4]])` and `Tensor(np.ones((2,2)))` raise `TypeError`; `Tensor([2,2], np.ones(4))` works; `np.asarray(t)` gives an object array; `float(t)`, `t.tolist()`, `t.numpy()` fail; `t ** 0.5` works eagerly |
| 2 | Tracer operators and outputs | `x ** 2`, `x ** 0.5`, `-x` raise `TypeError` in a trace; `1.0 - x` works; a tuple return raises "must return a TraceTensor"; `if tracer:` raises the control-flow `TypeError` |
| 3 | Composition on today's IR | reverse-over-reverse `f''(1.3) = 1.6896969463271838` (exact `1.6896969463271836`); forward-over-reverse and JVP∘JVP agree; VJP over JVP∘JVP works |
| 4 | Step cost | README loop: 17.2 µs per CPU step, `w = 3.141593`; `mlx_adam_loss_optimizer`: 659.6 µs per step, same `w`; Python pytree flatten plus key: 3.0 µs for 8 leaves |
| 5 | vmap VJP and batch dtype | per-example `x` gradients `[2,4,6,8,10,12]`; unmapped `w` gradient summed over the batch `[17,29,45]`; `tensor_jit_batch_fn` returns `f64` for a `float32` input |
| 6 | NumPy interop cost | `Tensor([1e6], ndarray)`: 29.1 ms; `to_flat_list()` → ndarray: 28.8 ms; `ndarray.copy()`: 0.97 ms; a `float32` ndarray becomes a `float64` tensor; a 2D ndarray raises |
