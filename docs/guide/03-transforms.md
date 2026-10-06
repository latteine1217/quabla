# 3. Function Transforms

Transforms turn a Python function into a new function: its derivative, its
vectorized version, or its compiled version. They are the core of Quabla.

```python
qb.grad(fun, argnums=0, has_aux=False)            # -> grads, or (grads, aux)
qb.value_and_grad(fun, argnums=0, has_aux=False)  # -> (value, grads)
qb.jvp(fun, primals, tangents)                    # -> (out, tangent_out)
qb.vjp(fun, *primals, has_aux=False)              # -> (out, vjp_fun[, aux])
qb.jacobian(fun, argnums=0)                       # dense Jacobian blocks
qb.hessian(fun, argnums=0)                        # dense Hessian blocks
qb.vmap(fun, in_axes=0, out_axes=0)               # batched fun
qb.jit(fun, device=None, static_argnums=(), max_traces=8,
       static_argnames=(), precision=None)        # compiled fun
```

All of them accept *pytrees* (nested dicts, lists, tuples, NamedTuples, and
registered classes; see [Chapter 6](06-pytrees-random-io.md#61-pytrees)) of
arrays and Python scalars as arguments, and return results with matching
structure.

## 3.1 How a Transform Executes

On the first call with a given *signature* (pytree structure, array shapes
and dtypes, static argument values), a transformed function:

1. calls your function once with `TraceTensor` placeholders to record a
   graph;
2. applies the transform to the graph symbolically (for example, builds the
   reverse-mode derivative graph);
3. compiles the graph into a frozen CPU execution plan (or a device plan for
   `jit(device=...)`);
4. executes the plan.

Later calls with the same signature skip steps 1 to 3. `grad`, `vmap`, and
the other transforms cache their traces exactly like `jit` does, so
`qb.grad(f)` is already compiled; wrapping it in `jit` mainly matters for
selecting a device and for fusing several transforms into one plan.

## 3.2 `grad` and `value_and_grad`

`grad` differentiates a scalar-valued function with respect to its first
argument, or the arguments selected by `argnums`. Gradients mirror the
pytree of the selected argument and keep its dtypes:

```python
def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return qb.mean((pred - y) ** 2)

params = {"w": qb.array([[0.5], [-0.25]]), "b": qb.array([0.1])}
x = qb.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = qb.array([[1.0], [2.0], [3.0]])

grads = qb.grad(loss)(params, x, y)
print(qb.tree.map(lambda t: t.shape, grads))

value, grads = qb.value_and_grad(loss)(params, x, y)
print(value)

g_params, g_x = qb.grad(loss, argnums=(0, 1))(params, x, y)   # a tuple of gradients
print(g_x.shape)
```

```text
{'b': [1], 'w': [2, 1]}
Tensor(2.12666667, dtype=float64)
[3, 2]
```

- The output must be a rank-0 floating array; a shape `[1]` result raises
  `ValueError` (reduce it, or use `jacobian` or `vjp` for non-scalar
  outputs).
- `bool_` leaves of a differentiated argument get `None` as their gradient.
- Python scalars in a differentiated position become `float64` arrays, so
  they can be differentiated too.

### Auxiliary outputs

With `has_aux=True` the function returns `(loss, aux)`. `aux` can be any
pytree and is returned unchanged next to the gradient:

```python
def loss_aux(params, x, y):
    pred = x @ params["w"] + params["b"]
    return qb.mean((pred - y) ** 2), {"pred": pred}

(value, aux), grads = qb.value_and_grad(loss_aux, has_aux=True)(params, x, y)
grads, aux = qb.grad(loss_aux, has_aux=True)(params, x, y)
```

### Decorator form

Called with keywords only, `grad`, `value_and_grad`, and `jit` return a
decorator:

```python
@qb.grad(argnums=1)
def d_dx(params, x):
    ...
```

## 3.3 Forward and Reverse Mode: `jvp` and `vjp`

`jvp` pushes a tangent forward through the function. `primals` and
`tangents` are tuples with one entry per argument:

```python
f = lambda x: qb.sin(x) * x
out, tangent = qb.jvp(f, (qb.array([0.0, 1.0]),), (qb.array([1.0, 1.0]),))
print(out, tangent)        # tangent = (cos(x) x + sin(x)) * v
```

```text
Tensor([0.        , 0.84147098], dtype=float64) Tensor([0.        , 1.38177329], dtype=float64)
```

`vjp` evaluates the function and returns a closure that pulls a cotangent
back to the inputs. The closure returns a tuple with one entry per primal:

```python
out, vjp_fun = qb.vjp(f, qb.array([0.0, 1.0]))
print(vjp_fun(qb.array([1.0, 1.0])))
```

```text
(Tensor([0.        , 1.38177329], dtype=float64),)
```

Use `jvp` when you have few inputs and many outputs (or need a directional
derivative), and `vjp`/`grad` when you have many inputs and a scalar or
small output.

## 3.4 `jacobian` and `hessian`

```python
def h(x):
    return qb.stack([x[0] * x[1], qb.sin(x[0])])

print(qb.jacobian(h)(qb.array([1.0, 2.0])))                    # [out, in]
print(qb.hessian(lambda x: qb.sum(x ** 3))(qb.array([1.0, 2.0])))
```

```text
Tensor([[2.        , 1.        ],
        [0.54030231, 0.        ]], dtype=float64)
Tensor([[ 6.,  0.],
        [ 0., 12.]], dtype=float64)
```

- A Jacobian block has shape `[*out.shape, *in.shape]`; a Hessian block
  `[*in.shape, *in.shape]`. With pytree inputs or outputs, the blocks are
  nested accordingly.
- `jacobian` uses reverse mode when the output has fewer floating elements
  than the input, and forward mode otherwise; either direction is vectorized
  with `vmap` over basis vectors, as `jax.jacfwd`/`jax.jacrev` do.
- `hessian` is forward-over-reverse.
- Results are dense, so memory grows quadratically with the input size. For
  large problems prefer Hessian-vector products, `jvp(grad(f), (x,), (v,))`.

## 3.5 Higher-Order Derivatives

Transforms compose. Each transformed call stages its graph and inlines it
into any enclosing trace, so a derivative can be used inside a function that
is differentiated again. This is what PINNs need: a PDE residual built from
`u_xx` that is itself differentiated with respect to the network weights.

```python
import math

def u(x, w):                                          # the model at one point
    return qb.sin(x * w)

u_xx = qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None))  # exact d2u/dx2 per point

def loss(w, x):
    return qb.mean((u_xx(x, w) + math.pi**2 * qb.sin(math.pi * x)) ** 2)

x = qb.linspace(0.05, 0.95, 8)
value, grad_w = qb.jit(qb.value_and_grad(loss))(qb.array(2.5), x)
```

Every derivative is exact: no finite differences are involved, and the
second derivative is a graph built from the same IR operations as the
model, so it runs on every backend.

Prefer `vmap` over a Python loop for per-point derivatives. Each inner
transformed call adds its graph once to the enclosing trace, so a loop over
`n` points grows the graph linearly with `n`, while `vmap` keeps it
independent of `n`.

## 3.6 `vmap`

`vmap` vectorizes a function written for one example:

```python
def per_example(w, x):                    # x: one example, shape [2]
    return qb.sum(qb.tanh(x * w))

w = qb.array([0.5, 1.0])
xs = qb.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(qb.vmap(per_example, in_axes=(None, 0))(w, xs))                  # one value per row
print(qb.vmap(qb.grad(per_example), in_axes=(None, 0))(w, xs).shape)   # per-example grads
```

```text
Tensor([1.42614474, 1.90447755, 1.98660201], dtype=float64)
[3, 2]
```

`in_axes` says which axis of each argument is mapped:

- an `int` maps that axis (negative counts from the end);
- `None` shares the argument across the batch (Python scalars must be
  `None`);
- a tuple, list, or dict gives one entry per argument as a pytree prefix, for
  example `in_axes=({"a": 0, "b": None},)`.

All mapped axes must have the same size. `out_axes` places the batch axis of
each result the same way; a result that does not depend on any mapped input
is broadcast over the batch (or returned unbatched with `out_axes=None`).

```python
qb.vmap(lambda x: x * 2.0, in_axes=1, out_axes=1)(xs)      # map over columns
```

`vmap` composes with every transform and with itself, including control flow
and solvers:

- `vmap(grad(f, argnums=1), in_axes=(0, None))` returns one gradient per
  example, shape `[B, *w.shape]`.
- `vjp(vmap(f, in_axes=(0, None)), x, w)` sums the unmapped gradient over the
  batch, as reverse mode must.
- `fori_loop`, `scan`, `while_loop`, and `cond` are batched as regions
  ([Chapter 4](04-control-flow.md#46-control-flow-under-vmap)), so `jacobian`
  and `hessian` work through loops, and `vmap` composes with `odeint`,
  `linalg.cg`, `linalg.gmres`, and `newton`.

## 3.7 Closures and Constants

A function may read arrays from its enclosing scope. An eager array that
meets a traced value becomes a constant of the graph:

```python
points = qb.linspace(0.0, 1.0, 8)
f = lambda w: qb.sum(qb.sin(points * w))
qb.grad(f)(qb.array(1.0))
```

As in JAX, the constant is fixed at trace time: rebinding `points` later does
not retrace. Pass data that changes as an argument. Every captured array is
stored in the graph, so capturing large arrays increases memory and compile
time.

A function passed to a transform *inside* a traced function may also close
over the enclosing tracers. The inner transform treats them as constants (it
differentiates only its own arguments), and the outer transform
differentiates through them:

```python
def loss(params, x):
    u_xx = qb.vmap(qb.grad(qb.grad(lambda t: net(params, t))))   # closes over params
    return qb.mean((u_xx(x) - forcing(x)) ** 2)

qb.jit(qb.value_and_grad(loss))(params, x)   # includes d u_xx / d params
```

A tracer stored in a global and used after its trace ended raises
`TracerError`.

## 3.8 `jit`

`jit` compiles a function and caches one plan per signature. It is also the
way to run on a device:

```python
step = qb.jit(qb.value_and_grad(loss))                  # CPU
step = qb.jit(qb.value_and_grad(loss), device="cuda:0") # CUDA device 0
step = qb.jit(qb.value_and_grad(loss), device="mlx")    # Apple GPU
```

Device execution, `precision="float64"`, and ahead-of-time lowering are
covered in [Chapter 10](10-devices.md).

### Static arguments

Arguments listed in `static_argnums` or `static_argnames` are Python values
baked into the trace; each distinct value gets its own trace. They must be
hashable:

```python
@qb.jit(static_argnums=1)
def power_sum(x, n):
    return qb.sum(x ** n)

power_sum(qb.array([1.0, 2.0]), 3)     # Tensor(9.)
power_sum(qb.array([1.0, 2.0]), n=2)   # Tensor(5.); n is static by name too
```

`static_argnums` and `static_argnames` imply each other through the
function's signature. A keyword argument that names the next positional
parameter binds to that position; any other keyword argument must be static.
Arguments of the other transforms are positional only.

### The trace cache

Each transformed function keeps at most `max_traces` traces (8 by default).
A new signature beyond that evicts the least recently used trace and emits a
`quabla.RetraceWarning` that names both signatures:

```python
j = qb.jit(lambda x: x * 2.0, max_traces=2)
for n in (1, 2, 3):
    j(qb.zeros((n,)))       # the third call warns
```

```text
RetraceWarning: jit(<lambda>) exceeded max_traces=2 and evicted its least recently used trace:
  (float64[1],)
new signature:
  (float64[3],)
...
```

Frequent eviction means you are compiling on every call. The usual causes are
a varying batch size (pad batches to a few fixed sizes) and a static argument
that changes every call.

### Python scalars

A Python scalar argument in a non-differentiated position is a *static weak
constant*: its value is baked into the graph and is part of the signature,
and it adopts the dtype of the arrays it meets, so it keeps a `float32`
program in `float32`. Each new value therefore traces again:

```python
def f(x, lr):
    print("trace", lr)
    return x * lr

j = qb.jit(f)
j(qb.array(1.0), 0.5)             # trace 0.5
j(qb.array(1.0), 0.25)            # trace 0.25: a new value, a new trace
j(qb.array(1.0), qb.array(0.1))   # trace TraceTensor(...): an array argument
j(qb.array(1.0), qb.array(0.2))   # cached
```

To vary a scalar between calls without retracing, pass it as
`qb.array(value)` (with the dtype of the program). In a differentiated
position a Python scalar becomes a `float64` array argument, so it does not
retrace.

## 3.9 Rules for Traced Code

- No Python control flow on traced values: `if x > 0`, `while err > tol`,
  `float(x)`, `x.item()`, and `np.asarray(x)` raise `TracerError`. Use
  `qb.where`, `qb.cond`, or the loops of [Chapter 4](04-control-flow.md).
- Python control flow on *static* things is fine: loops over a fixed range,
  branches on shapes (`x.shape[0]`), branches on static arguments.
- Side effects run once per trace, not once per call.
- Keep shapes static. Each new shape compiles a new plan.

```python
qb.grad(lambda x: x if x > 0 else -x)(qb.array(1.0))
```

```text
quabla.TracerError: TraceTensor cannot drive Python control flow; use quabla.where for elementwise selection or an explicit control-flow primitive
```

## 3.10 Errors

| Exception | Base classes | Raised when |
| --- | --- | --- |
| `qb.QuablaError` | `Exception` | Base of the Quabla-specific errors |
| `qb.TracerError` | `QuablaError`, `TypeError` | A traced value is used where a concrete one is needed |
| `qb.UnsupportedOperationError` | `QuablaError`, `ValueError`, `NotImplementedError` | A backend or transform does not support an operation; `.op` and `.device` say which |
| `qb.RetraceLimitError` | `QuablaError`, `ValueError` | Kept for existing handlers; caches now evict instead |
| `qb.RetraceWarning` | `UserWarning` | A trace was evicted from a cache |

The full contract of every transform, including the composition rules for
`cond` under `vmap` and keyword arguments, is in the
[API Reference](../api.md#function-transforms).
