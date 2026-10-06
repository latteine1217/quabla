# 12. Performance, Debugging, and Limitations

## 12.1 Where Time Goes

A transformed function has two costs:

| Cost | When | Scales with |
| --- | --- | --- |
| Trace and compile | First call per signature | Graph size (number of IR nodes), and on CUDA the NVRTC compile |
| Execute | Every call | The work in the plan, plus host/device transfers and launches |

Keep the first small and paid once, and the second free of avoidable
transfers.

### Keep the graph small

Inspect any function's compiled graph with `lower`:

```python
text = qb.jit(f).lower(qb.ShapeDtype((n,))).as_text()
print(len(text.splitlines()), "lines")
```

The biggest graph-size mistake is a Python loop over data points around a
transform. Each inner transformed call inlines its graph once:

```python
d = qb.grad(qb.grad(lambda x: qb.sin(x) ** 2))

def per_point_loop(xs):
    return qb.stack([d(xs[i]) for i in range(xs.shape[0])])

def per_point_vmap(xs):
    return qb.vmap(d)(xs)
```

| Points | `per_point_loop` lines | `per_point_vmap` lines |
| --- | --- | --- |
| 4 | 147 | 35 |
| 16 | 579 | 35 |

Other sources of graph growth:

- **Large captured constants.** Every eager array a traced function closes
  over is stored in the graph. Pass large data as arguments instead.
- **`unroll=True`** on long loops, and Python `for` loops over many
  iterations in traced code. Use `fori_loop`/`scan` regions.
- **Dense `jacobian`/`hessian`** of large inputs: their basis and result grow
  quadratically. Use `jvp(grad(f))` for Hessian-vector products.
- **High-order Cholesky derivatives** (third and above) use a scalar
  expansion that is only practical for small matrices.

### Avoid retracing

A new signature retraces and recompiles. Watch for `quabla.RetraceWarning`,
which names the evicted and the new signature. Typical causes:

- varying batch sizes: pad or bucket batches to a few fixed sizes, or raise
  `max_traces` if there are only a few;
- Python scalars that change every call (a step counter, a learning rate):
  pass them as arrays;
- static arguments or dataclass meta fields that change every call;
- a new closure created every iteration (`qb.jit(lambda ...)` inside the
  loop): build transformed functions once, outside the loop.

To make retracing an error during development:

```python
import warnings
warnings.simplefilter("error", qb.RetraceWarning)
```

### Avoid readbacks on devices

Every `.item()`, `float()`, `tolist()`, `numpy()`, and `trainer.loss()` on a
device result synchronizes and copies. In a training loop, log every N steps
instead of every step, and prefer `Trainer.step()` (no arguments), which
reuses retained device buffers without touching Python pytrees.

Every `cond` and every `while_loop` iteration reads a scalar predicate back
on CUDA and MLX. Prefer `where` for cheap selections, and bounded
`fori_loop`s with masking where the trip count is predictable.

### CPU specifics

The CPU backend interprets frozen plans; there is no machine-code JIT. It is
the reference implementation, optimized for correctness: `float32`
operations are computed in `float64` and rounded. Use it for development,
validation, and small models; use a GPU backend for throughput.

## 12.2 Debugging

### See what you compiled

```python
lowered = qb.jit(qb.value_and_grad(loss)).lower(params, x)
print(lowered.as_text())
```

The text shows each IR node with its operands, shape, and dtype, after dead
code elimination, constant folding, and CSE. Look for unexpected dtypes
(`f64` in a `float32` model), unexpected broadcasts, and node counts that
grow with data size.

### Print values

Inside a traced function, `print(x)` prints the tracer at trace time, not the
values. To inspect intermediate values, run the function eagerly (call it
without any transform; every operation then executes immediately), or
return the intermediate as an extra output (`has_aux=True` for `grad`).

### Compare against the CPU

When a device result looks wrong, run the same jitted function with
`device="cpu"` on the same `float32` inputs. The CPU is the reference, and
the backends are validated against it operation by operation. Expect
agreement to about `1e-5` relative in `float32`; a larger difference is a
bug worth reporting.

### Check gradients numerically

```python
def fd_check(f, x, v, eps=1e-6):
    analytic = qb.sum(qb.grad(f)(x) * v).item()
    numeric = (f(x + eps * v).item() - f(x - eps * v).item()) / (2 * eps)
    return analytic, numeric
```

Run checks in `float64`; `float32` finite differences are too noisy for
tight comparisons.

### Hunting NaN

- `mask * x` propagates NaN from `x` even where the mask is zero (`0 * NaN`
  is NaN). Use `qb.where(mask, x, 0.0)`.
- `qb.where` evaluates both branches. If the unselected branch is NaN, its
  value is dropped, but if you take `log` of a negative value *before*
  `where`, guard the input too: `qb.log(qb.where(x > 0, x, 1.0))`. Or use
  `qb.cond` for scalars.
- Prefer the stable built-ins: `softplus`, `logsumexp`, `log_softmax`,
  `logaddexp`, `hypot`, `norm`, `var`, `log1p`, `expm1`.
- Quabla picks finite conventions where textbook derivatives are undefined
  (for example every derivative of `sqrt` at zero is zero); see the
  [API Reference](../api.md#arrays-and-numpy) for the exact rule of each
  function.
- A non-finite clipped norm makes every clipped gradient NaN by design;
  check `math.isfinite(norm)`.

## 12.3 Common Errors

| Message (abbreviated) | Cause | Fix |
| --- | --- | --- |
| `TracerError: TraceTensor cannot drive Python control flow` | `if`/`while` on a traced value | `qb.where`, `qb.cond`, loop primitives |
| `TracerError: control-flow regions cannot close over an outer tracer; pass it explicitly using operands=` | A loop body or branch uses a traced value from outside | Pass it via `operands=` (`cond`: positional) |
| `ValueError: tensor + operands have mismatched dtypes f64 and f32` | Mixing `float32` and `float64` arrays | `astype` one side; create constants with the right dtype |
| `ValueError: grad requires a scalar output` | The loss is not rank 0 | Reduce with `qb.sum`/`qb.mean`; use `jacobian`/`vjp` for vectors |
| `ValueError: reverse-mode differentiation through while_loop is not supported` | `grad` of a `while_loop` | Bounded `fori_loop` with masking, or `jvp` |
| `ValueError: forward-mode differentiation ... of custom_vjp(...) is not supported` | `jvp` of a `custom_vjp` | Use reverse mode, or define `custom_jvp` |
| `UnsupportedOperationError: ... target is unavailable in this build` | `device=` names a backend that was not compiled in | Install `quabla-mlx` or build with `--features` |
| `UnsupportedOperationError` with `.op == "float32"` | A `precision="float64"` program contains `float32` values | Cast everything to `float64` |
| `RuntimeError: ... stopped at t=... before t1 after max_steps=...` | An adaptive ODE solve did not finish | Raise `max_steps`, loosen `rtol`/`atol`, or use `rosenbrock23` for stiff problems |
| `RetraceWarning: ... exceeded max_traces` | Many signatures | See [Avoid retracing](#avoid-retracing) |

## 12.4 Known Limitations

Quabla is a research-grade 0.x library. These limits are deliberate or not
yet addressed:

- **Shapes are static.** There are no symbolic dimensions; each shape is a
  separate compiled program.
- **Dtypes** are `float32`, `float64`, and `bool` only: no integers,
  `float16`, `bfloat16`, or complex numbers, and no mixed-precision
  training.
- **Indexing** takes static Python integers and slices. Dynamic index
  arrays, boolean-mask indexing, in-place assignment, and empty slices are
  not supported.
- **Loop carries** are single arrays; `cond` results and carries must be
  floating-point.
- **`while_loop`** has no reverse-mode derivative (as in JAX).
  Reverse-over-reverse through `fori_loop`/`scan` regions is rejected;
  forward-over-reverse (Hessians, HVPs) works.
- **`jacobian`/`hessian`** return dense arrays.
- **`qb.linalg`** has no non-symmetric `eig`; `lstsq` requires full rank;
  batched CUDA decompositions issue one cuSOLVER call per batch element.
- **CPU** interprets plans (no machine-code JIT).
- **CUDA** runs non-elementwise loop bodies host-driven, one launch per
  iteration.
- **MLX** has no `float64`, runs factorizations on its CPU stream, and has no
  `vmap` HVP lowering.
- **Data parallelism** is single-node CUDA + NCCL only, with the optimizer
  on the host.
- **Random sampling** is eager and on the host; keys cannot be traced.

The authoritative and complete list is in the
[API Reference](../api.md#known-limitations), and per-feature status is in
the [roadmap](../jax_like_roadmap.md).
