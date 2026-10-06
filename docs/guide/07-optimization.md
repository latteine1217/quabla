# 7. Optimization

`qb.optim` provides first-order optimizers with a pure functional interface,
learning-rate schedules, global-norm clipping, a full-batch L-BFGS
minimizer, and `Trainer`, which keeps parameters and optimizer state on a
device between steps.

| Name | Kind |
| --- | --- |
| `Adam`, `AdamW`, `SGD` | Optimizers: `init(params)` and `update(params, grads, state)` |
| `constant`, `exponential_decay`, `cosine_decay`, `warmup_cosine_decay`, `piecewise_constant` | Schedules: `f(step) -> float` |
| `clip_by_global_norm` | Gradient clipping |
| `LBFGS` | Full-batch quasi-Newton minimizer |
| `Trainer` | A retained-buffer training loop on CPU, CUDA, or MLX |

## 7.1 The Functional Optimizer Interface

```python
def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return qb.mean((pred - y) ** 2)

x = qb.linspace(-1.0, 1.0, 32).reshape(32, 1)
y = 3.0 * x - 0.5
params = {"w": qb.zeros((1, 1)), "b": qb.zeros((1,))}

opt = qb.optim.Adam(learning_rate=0.1)
state = opt.init(params)                       # {'m': ..., 'v': ..., 'step': 0}
step = qb.jit(qb.value_and_grad(loss))

for i in range(300):
    value, grads = step(params, x, y)
    params, state = opt.update(params, grads, state)

print(value.item(), params["w"].item(), params["b"].item())
```

```text
9.047797907759665e-14 3.0000004277208028 -0.49999999706101544
```

- `update` is pure: it returns new parameters and a new state and modifies
  nothing. Parameters can be any pytree of floating arrays; gradients must
  have the same structure (which `grad` guarantees).
- Parameters keep their dtype. Host Adam moments are kept in `float64`.
- The update itself runs eagerly on the host. For device-resident training
  loops use [`Trainer`](#75-trainer).

| Optimizer | Signature | Rule |
| --- | --- | --- |
| `Adam` | `Adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, *, clip_norm=None)` | Bias-corrected Adam |
| `AdamW` | `AdamW(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4, *, clip_norm=None)` | Decoupled weight decay as optax's `adamw`: `p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)` |
| `SGD` | `SGD(learning_rate=1e-2, *, clip_norm=None)` | `p - lr * g`; its state is `None` (or `{"step": n}` with a schedule) |

For compatibility with v0.1, `Adam` also accepts `beta1`, `beta2`, and
`epsilon`, and keeps a stateful `step(params, grads)` method that returns
new parameters while storing the moments inside the optimizer object. New
code should use `init`/`update`.

## 7.2 Learning-Rate Schedules

Every `learning_rate` accepts either a float or a schedule, a callable
`f(step) -> float` evaluated on the host at the number of *completed*
updates (0 for the first update, as in optax). The built-in schedules follow
optax's formulas and argument order:

```python
sched = qb.optim.warmup_cosine_decay(
    init_value=0.0, peak_value=0.1, warmup_steps=10, decay_steps=100
)
print([round(sched(s), 4) for s in (0, 5, 10, 55, 100)])
print([qb.optim.piecewise_constant([10, 20], [1.0, 0.5, 0.1])(s) for s in (0, 10, 25)])
```

```text
[0.0, 0.05, 0.1, 0.05, 0.0]
[1.0, 0.5, 0.1]
```

| Schedule | Signature |
| --- | --- |
| `constant` | `constant(value)` |
| `exponential_decay` | `exponential_decay(init_value, transition_steps, decay_rate, transition_begin=0, staircase=False, end_value=None)` |
| `cosine_decay` | `cosine_decay(init_value, decay_steps, alpha=0.0)` |
| `warmup_cosine_decay` | `warmup_cosine_decay(init_value, peak_value, warmup_steps, decay_steps, end_value=0.0)`; `decay_steps` includes the warmup |
| `piecewise_constant` | `piecewise_constant(boundaries, values)`; `values[i]` from step `boundaries[i - 1]` on |

Invalid configurations raise `ValueError` (where optax only warns), and a
schedule that returns a negative or non-finite rate raises when used. Any
Python callable works as a custom schedule.

## 7.3 Gradient Clipping

```python
grads = {"a": qb.array([3.0]), "b": qb.array([4.0])}
clipped, norm = qb.optim.clip_by_global_norm(grads, 1.0)
print(clipped, norm)
```

```text
{'a': Tensor([0.6], dtype=float64), 'b': Tensor([0.8], dtype=float64)} 5.0
```

The global norm over all leaves is computed in `float64` as
`m * sqrt(sum((g / m) ** 2))` with `m = max |g|`, so it neither overflows nor
underflows. Leaves are scaled by `min(1, max_norm / norm)` and keep their
dtype. A NaN or infinite norm returns all-NaN gradients rather than silently
finite ones; test `math.isfinite(norm)` to skip such a step.

`clip_norm=` on `Adam`, `AdamW`, and `SGD` applies the same clipping inside
every `update`:

```python
opt = qb.optim.AdamW(learning_rate=sched, weight_decay=1e-2, clip_norm=1.0)
```

## 7.4 L-BFGS

PINN training commonly runs Adam first and finishes with L-BFGS, which
converges much faster near a minimum. `LBFGS.minimize` runs the whole
full-batch optimization on the host:

```python
def rosenbrock(p):
    x, y = p["x"], p["y"]
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

solver = qb.optim.LBFGS()
solution, info = solver.minimize(rosenbrock, {"x": qb.array(-1.2), "y": qb.array(1.0)})
print(solution)
print(info)
```

```text
{'x': Tensor(1., dtype=float64), 'y': Tensor(1., dtype=float64)}
{'iterations': 37, 'evaluations': 46, 'loss': 1.5246593064776902e-22, 'grad_norm': 4.1183323418449303e-10, 'reason': 'gradient_tolerance', 'converged': True}
```

```python
qb.optim.LBFGS(
    history=10,               # curvature pairs kept
    max_iterations=500,
    max_evaluations=None,
    tolerance_grad=1e-7,      # stop when max |g| <= this
    tolerance_change=1e-9,    # stop when the relative loss change <= this
    line_search="strong_wolfe",
).minimize(fun, params, *args)   # fun(params, *args) -> scalar
```

- `minimize` jits `value_and_grad(fun)` on the CPU once and iterates over
  one `float64` vector of all parameter leaves. `float32` leaves are
  evaluated, and stored, at `float32` precision.
- The line search is a strong Wolfe search (`c1=1e-4`, `c2=0.9`) that
  backtracks from non-finite losses; a failed search restarts once from
  steepest descent.
- `info["reason"]` is one of `gradient_tolerance`, `change_tolerance`,
  `max_iterations`, `max_evaluations`, or `line_search_failed`. The last is
  common for `float32` problems that have reached the resolution limit of
  the dtype; it is not necessarily a failure.
- The change test omits scipy's floor of 1, so tiny PINN losses are not
  stopped early.

## 7.5 `Trainer`

`Trainer` packages "trace once, then step many times" and, on a device,
keeps parameters and optimizer moments in device memory so that a step
transfers nothing back to the host:

```python
qb.optim.Trainer(loss, params, optimizer, *data,
                 device="cpu", batch_argnums=(), precision=None)
```

`loss(params, *data)` must return a scalar. `optimizer` is an `Adam`,
`AdamW`, or `SGD` instance, with a constant rate or a schedule, and with or
without `clip_norm`.

```python
params = {"w": qb.zeros((1, 1)), "b": qb.zeros((1,))}
trainer = qb.optim.Trainer(loss, params, qb.optim.Adam(learning_rate=0.1), x, y)
for _ in range(300):
    trainer.step()
print(trainer.loss(), trainer.params["w"].item())
```

```text
Tensor(6.49246356e-14, dtype=float64) 3.0000004277208028
```

| Member | Meaning |
| --- | --- |
| `step()` | One optimizer step with the current data |
| `step(*batch)` | Replace the data arguments named by `batch_argnums`, in that order, then step. Shapes, dtypes, and pytree structure must match the initial data. |
| `loss()` | Evaluate the loss at the current parameters and latest data (a device readback) |
| `params` | Read the current parameters back into their original pytree |

### Mini-batches

`batch_argnums` indexes the `data` arguments (excluding `params`). Data not
listed there is uploaded once and retained on the device; listed data is
replaced on each `step(...)` call that passes it:

```python
xf, yf = x.astype(qb.float32), y.astype(qb.float32)
pf = {"w": qb.zeros((1, 1), dtype=qb.float32), "b": qb.zeros((1,), dtype=qb.float32)}
trainer = qb.optim.Trainer(
    loss, pf, qb.optim.Adam(learning_rate=0.1), xf, yf,
    device="mlx", batch_argnums=(0, 1),
)
for i in range(300):
    trainer.step(xf, yf)            # upload a new batch (here: the same one)
print(trainer.loss(), trainer.params["w"].item())   # on Apple silicon
```

```text
Tensor(1.8750973e-13, dtype=float32) 3.0000007152557373
```

### Device semantics

- On `"cpu"`, `Trainer` is exactly `jit(value_and_grad(loss))` followed by
  `optimizer.update`.
- On `"cuda"`, `"cuda:N"`, and `"mlx"`, the loss and its gradient compile
  into one device plan, and the optimizer update runs on the device with the
  CPU's rules in the CPU's operation order. Global-norm clipping is reduced
  on the device too, so gradients are never read back.
- A schedule is evaluated on the host before every step and sets the native
  optimizer's rate.
- Device plans compute in `float32` and agree with the CPU trainer up to
  `float32` rounding. `precision="float64"` trains a `float64` loss natively
  in double precision on CUDA (parameters and moments included); it is a
  no-op on the CPU and rejected on MLX.
- `step()` without arguments on a device reuses the retained data without
  flattening any Python pytree, which keeps per-step overhead minimal.

`examples/benchmark_v02_training.py` compares `Trainer` with the native
device Adam executor and reports compile, step, and readback timings
separately.
