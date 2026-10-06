# 4. Control Flow

Python `if` and `while` cannot inspect a traced value, because a traced value
has no data until the compiled plan runs. Quabla instead provides four
structured control-flow primitives. Each one is recorded as an IR *region*
(a sub-graph with explicit inputs), so it is compiled once, differentiated,
vectorized with `vmap`, and lowered to every backend.

| Primitive | Semantics | Reverse mode |
| --- | --- | --- |
| `qb.cond(pred, true_fun, false_fun, *operands)` | Run one branch | Yes |
| `qb.fori_loop(lower, upper, body, init, *, operands=(), unroll=False)` | `carry = body(i, carry, *operands)` for `i` in `range(lower, upper)` | Yes |
| `qb.scan(f, init, *, length, operands=(), unroll=False)` | `carry, y_i = f(carry, i, *operands)`; stack the `y_i` | Yes |
| `qb.while_loop(cond_fun, body, init, *, operands=())` | `carry = body(carry, *operands)` while `cond_fun(carry, *operands)` | No (forward mode only) |

All four also work eagerly, outside any transform: they then simply run in
Python.

## 4.1 The Operand Rule

A region cannot close over a traced value from the enclosing function. Every
traced array the region uses must be passed explicitly: as `operands=` for
the loops, or as trailing positional arguments for `cond`. The region
function receives them as extra arguments:

```python
def decay(x0, a):
    h = 0.01                                   # Python scalars may be closed over
    def body(i, x, a):                         # a arrives as an operand
        return x - h * a * x
    return qb.fori_loop(0, 100, body, x0, operands=(a,))
```

Forgetting this raises a clear error:

```python
def bad(x0, a):
    return qb.fori_loop(0, 10, lambda i, x: x * a, x0)   # closes over traced a

qb.jit(bad)(x0, a)
```

```text
quabla.TracerError: control-flow regions cannot close over an outer tracer; pass it explicitly using operands= (cond takes positional operands)
```

Closing over Python scalars and eager (non-traced) arrays is allowed; they
become constants of the region.

Other rules shared by the loops:

- The carry is a **single array**, not a pytree. Pack several state variables
  into one array (for example with `qb.concat` and slicing) if needed.
- The body must return a carry of the same shape and dtype.
- Bounds (`lower`, `upper`, `length`) are static non-negative Python
  integers.
- Carries and `cond` results must be floating-point, not `bool_`.

## 4.2 `cond`

```python
def safe_log(x):
    return qb.cond(x > 0.0, lambda x: qb.log(x), lambda x: qb.zeros_like(x), x)

safe_log(qb.array(2.0))                # Tensor(0.69314718)
qb.jit(safe_log)(qb.array(-1.0))       # Tensor(0.)
qb.grad(safe_log)(qb.array(2.0))       # Tensor(0.5)
```

- `pred` is a scalar, either `bool_` or a floating value tested with `!= 0`.
- Only the selected branch runs. Unlike `qb.where`, which evaluates both
  sides, the inactive branch cannot inject NaN into values or gradients (here
  `log` of a negative number is never computed).
- On CUDA and MLX the predicate is read back to the host once per `cond`
  evaluation (one stream synchronization), and the selected branch then runs
  on the device.
- The two branches must return the same shape; a branch may return a
  constant of the other branch's dtype.

Use `qb.where` for cheap elementwise selection and `qb.cond` when a branch is
expensive or would produce non-finite values.

## 4.3 `fori_loop`

```python
x0, a = qb.array([1.0, 2.0]), qb.array(0.5)
decay(x0, a)                                        # eager: Python loop
qb.jit(decay)(x0, a)                                # one compiled loop region
qb.grad(lambda a: qb.sum(decay(x0, a)))(a)          # reverse mode through the loop
qb.hessian(lambda x: qb.sum(decay(x, a) ** 2))(x0)  # second order through the loop
```

```text
Tensor([0.60577044, 1.21154087], dtype=float64)
Tensor([0.60577044, 1.21154087], dtype=float64)
Tensor(-1.82644353, dtype=float64)
Tensor([[0.73391564, 0.        ],
        [0.        , 0.73391564]], dtype=float64)
```

The loop index has a different type depending on how the loop runs:

| Situation | `i` is |
| --- | --- |
| Eager call (no transform) | a Python `int` |
| Traced region (the default under a transform) | a scalar array of the carry's dtype |
| `unroll=True` under a transform | a Python `int`; the body is traced once per iteration |

`unroll=True` inlines every iteration into the graph. It gives the backend a
straight-line program (useful for short loops with small bodies on CUDA,
where every region iteration costs a kernel launch) at the price of a graph
that grows with the trip count.

### How loop gradients are stored

Reverse mode through a loop needs the carry of every iteration. Instead of
storing all `T` carries, the CPU and MLX backends store square-root
checkpoint blocks and replay each block during the backward pass, which
reduces carry storage from O(T·C) to O(√T·C) at the cost of a measured
replay. CUDA uses the same scheme for host-driven loops and keeps a full
on-device tape for its fused elementwise loop kernels.

Second-order derivatives through `fori_loop` and `scan` work in every
combination of modes but one: forward-over-reverse (`hessian`,
`jvp(grad(f))`), reverse-over-forward (`grad` of a `jvp`), and
reverse-over-reverse (`grad(grad(f))` with the loop inside `f`, or the
`hessian` of a function that calls a `custom_vjp` solver), but not forward
over forward through a `fori_loop`. Reverse mode over a loop's reverse pass
is computed from the loop's forward-mode and forward-over-reverse passes
(the Hessian is symmetric), with the same checkpointing, so it needs the
loop body's forward mode. A third derivative pass through a loop raises.

## 4.4 `scan`

`scan` is a `fori_loop` that also emits one output per step and stacks them
on a new leading axis:

```python
def rollout(x0, w):
    def step(c, i, w):
        c = qb.tanh(c * w + 0.1)
        return c, c * 2.0                       # (next carry, per-step output)
    return qb.scan(step, x0, length=4, operands=(w,))

carry, ys = qb.jit(rollout)(qb.array([0.5, 1.0]), qb.array(0.8))
print(carry, ys.shape)                          # ys: [length, *output.shape]
```

```text
Tensor([0.41178338, 0.47153359], dtype=float64) [4, 2]
```

Note the argument order of the step function, `f(carry, i, *operands)`. It
differs from `jax.lax.scan`, which passes a slice of `xs` instead of an
index. To scan over a sequence, pass the sequence as an operand and index it
inside the body; the index is a traced scalar there, so use it in arithmetic
(for example a one-hot weighting) or switch to `unroll=True` to get a Python
integer for slicing.

`length` must be positive. The per-step outputs must have a fixed shape.

## 4.5 `while_loop`

`while_loop` is the one loop whose trip count depends on traced values:

```python
def newton_sqrt(a):
    def cond_fun(x, a):
        return qb.abs(x * x - a) > 1e-12
    def body_fun(x, a):
        return 0.5 * (x + a / x)
    return qb.while_loop(cond_fun, body_fun, qb.ones_like(a), operands=(a,))

newton_sqrt(qb.array(2.0))                               # Tensor(1.41421356)
qb.jvp(newton_sqrt, (qb.array(2.0),), (qb.array(1.0),))  # forward mode works
```

`cond_fun` must return a scalar `bool_`. The backends evaluate the predicate,
read it back to the host once per iteration, and then run the body.

Reverse mode through `while_loop` is rejected, as in JAX: a data-dependent
trip count leaves no fixed tape to reverse.

```text
ValueError: reverse-mode differentiation through while_loop is not supported because its trip count is data dependent; use forward mode (jvp) or rewrite it as a bounded fori_loop whose body masks finished iterations ...
```

The fix is to bound the iteration count and freeze converged iterations with
`where`:

```python
def sqrt_bounded(a):
    def body(i, x, a):
        done = qb.abs(x * x - a) <= 1e-12
        return qb.where(done, x, 0.5 * (x + a / x))
    return qb.fori_loop(0, 30, body, qb.ones_like(a), operands=(a,))

qb.grad(sqrt_bounded)(qb.array(2.0))     # Tensor(0.35355339) = 1 / (2 sqrt 2)
```

For iterative solvers of linear and nonlinear systems, prefer the built-in
`qb.linalg.cg`, `qb.linalg.gmres`, and `qb.newton`
([Chapter 8](08-linear-algebra.md#84-iterative-solvers-and-implicit-differentiation)):
they run as a `while_loop` and differentiate the *solution* through the
implicit function theorem, which is both cheaper and more accurate than
differentiating the iterations.

## 4.6 Control Flow Under `vmap`

`vmap` batches regions with JAX semantics:

```python
qb.vmap(newton_sqrt)(qb.array([2.0, 9.0, 1e6]))     # per-example trip counts
qb.vmap(safe_log)(qb.array([2.0, -1.0]))            # per-example predicate
qb.vmap(qb.grad(safe_log))(qb.array([2.0, -1.0]))
```

```text
Tensor([   1.41421356,    3.        , 1000.        ], dtype=float64)
Tensor([0.69314718, 0.        ], dtype=float64)
Tensor([0.5, 0. ], dtype=float64)
```

- **`fori_loop` and `scan`**: the body region is batched with exactly the
  mapped operands mapped. A carry that depends on a mapped capture is mapped
  from the first iteration; a batched scan's stacked outputs are
  `[B, length, ...]`.
- **`cond` with an unmapped predicate** stays one lazy `cond` with batched
  branches.
- **`cond` with a mapped predicate** runs *both* branches for every example
  and selects with `where`. It costs both branches, and a branch sees inputs
  it would not see unbatched (here `log(-1.0)`), but those values and their
  derivatives never reach the result for that example.
- **`while_loop` with a mapped predicate** runs while any example continues;
  finished examples keep their carry unchanged bit for bit. The batch costs
  the trip count of its slowest example.

## 4.7 Backend Notes

| Backend | `cond` | `fori_loop` / `scan` | `while_loop` |
| --- | --- | --- | --- |
| CPU | Interpreted region | Interpreted region with checkpointed reverse pass | One predicate evaluation per iteration |
| MLX | One predicate readback | Dispatched from the host on device-resident arrays | One predicate readback per iteration |
| CUDA | One predicate readback | Fused single kernel when the body is purely elementwise; otherwise a host-driven loop (one CUDA graph launch per iteration) | Host-driven, one predicate readback per iteration |

On CUDA, a host-driven iteration costs tens of microseconds, so long loops
with tiny bodies are launch-bound. See
[Devices](10-devices.md#104-cuda) for the exact rules on which bodies fuse.

A `cond`, `fori_loop`, `scan`, or `while_loop` body may call a
`custom_vjp`, `custom_jvp`, or `checkpoint` function, or a solver from
Chapter 8: the compiled region keeps the custom rule, so the derivatives of
the region apply it, as they do outside control flow. Forward mode through a
body that calls a `custom_vjp` function needs the function's forward-mode
rule (the solvers of Chapter 8 have one), and a `while_loop` body is still
differentiated in forward mode only.
