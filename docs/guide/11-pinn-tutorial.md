# 11. Tutorial: Training a PINN

This chapter builds a complete physics-informed neural network (PINN) with
the core API, trains it with Adam followed by L-BFGS, checks it against the
exact solution, and then trains it on a GPU. It uses only features
introduced in earlier chapters.

## 11.1 The Problem

Solve the 1D Poisson problem

```text
u''(x) = -pi^2 sin(pi x)   on (0, 1),     u(0) = u(1) = 0,
```

whose exact solution is `u(x) = sin(pi x)`. A PINN represents `u` by a
neural network `u_theta(x)` and minimizes

```text
L(theta) = mean_i (u_theta''(x_i) - f(x_i))^2  +  mean_b u_theta(x_b)^2
```

over interior collocation points `x_i` and the boundary points `x_b`. The
first term needs the exact second derivative of the network with respect to
its *input*, and the optimizer needs the gradient of `L` with respect to the
*weights*, which differentiates through that second derivative. This nesting
is what Quabla's composable transforms are built for.

## 11.2 The Model

Parameters are a list of layer dicts, which is a pytree, so the transforms
and optimizers handle it without any registration:

```python
import math
import quabla as qb

def init_mlp(key, sizes, dtype=qb.float64):
    keys = qb.random.split(key, len(sizes) - 1)
    return [
        {"w": qb.random.glorot_normal(k, (m, n), dtype=dtype),
         "b": qb.zeros((n,), dtype=dtype)}
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]

def mlp(params, x):                       # x: [batch, 1] -> [batch, 1]
    h = x
    for layer in params[:-1]:
        h = qb.tanh(h @ layer["w"] + layer["b"])
    last = params[-1]
    return h @ last["w"] + last["b"]

def u(params, x):
    """The network as a scalar function of one scalar coordinate."""
    return mlp(params, x.reshape(1, 1)).reshape(())
```

The Python `for` loop over layers is fine: it runs over a static list, so it
is unrolled once at trace time.

`u` is the per-point view of the network. Writing the model for a single
point and vectorizing it with `vmap` is the idiomatic pattern: derivatives
with respect to `x` are then ordinary `grad`s of a scalar function.

## 11.3 The Loss

```python
def forcing(x):
    return -math.pi**2 * qb.sin(math.pi * x)

def loss(params, x_interior, x_boundary):
    u_xx = qb.vmap(qb.grad(qb.grad(lambda t: u(params, t))))
    residual = u_xx(x_interior) - forcing(x_interior)
    boundary = qb.vmap(lambda t: u(params, t))(x_boundary)
    return qb.mean(residual ** 2) + qb.mean(boundary ** 2)
```

Reading `u_xx` from the inside out:

1. `lambda t: u(params, t)` is the network as a function of `t` alone. It
   closes over `params`, which is a tracer of the enclosing transform.
2. `qb.grad(qb.grad(...))` is the exact second derivative in `t`. The inner
   transforms treat the closed-over `params` as constants; the outer
   `value_and_grad` (below) differentiates through them, so the weight
   gradient includes the dependence of `u_xx` on the weights.
3. `qb.vmap(...)` evaluates it at every collocation point. The graph size is
   independent of the number of points.

An equivalent spelling passes `params` explicitly:
`qb.vmap(qb.grad(qb.grad(u, argnums=1), argnums=1), in_axes=(None, 0))`.

## 11.4 Training With Adam

```python
params = init_mlp(qb.random.key(0), [1, 16, 16, 1])
x_interior = qb.linspace(0.0, 1.0, 34)[1:-1]     # 32 interior points
x_boundary = qb.array([0.0, 1.0])

step = qb.jit(qb.value_and_grad(loss))
opt = qb.optim.Adam(learning_rate=qb.optim.exponential_decay(1e-2, 1000, 0.5))
state = opt.init(params)

for i in range(2000):
    value, grads = step(params, x_interior, x_boundary)
    params, state = opt.update(params, grads, state)
    if i % 500 == 0:
        print(f"adam step {i:4d}  loss {value.item():.3e}")
```

```text
adam step    0  loss 5.283e+01
adam step  500  loss 1.191e-03
adam step 1000  loss 4.313e-04
adam step 1500  loss 2.557e-04
```

The first call of `step` traces `loss`, builds the derivative graph (third
derivatives of the network with respect to mixed input/weight directions),
and compiles it. Every later call executes the frozen plan. The full 2000
steps took about 0.3 s on the CPU backend of the Apple silicon machine used
for this guide.

## 11.5 Refining With L-BFGS

Adam makes fast early progress; a quasi-Newton method converges much faster
near a minimum. `LBFGS.minimize` takes the same loss and the extra
arguments:

```python
params, info = qb.optim.LBFGS(max_iterations=500).minimize(
    loss, params, x_interior, x_boundary
)
print(f"lbfgs {info['reason']} after {info['iterations']} iterations, loss {info['loss']:.3e}")
```

```text
lbfgs max_iterations after 500 iterations, loss 4.614e-07
```

The loss drops by almost three orders of magnitude.

## 11.6 Checking the Solution

```python
x_test = qb.linspace(0.0, 1.0, 201)
pred = qb.vmap(lambda t: u(params, t))(x_test)
err = qb.max(qb.abs(pred - qb.sin(math.pi * x_test)))
print(f"max abs error {err.item():.2e}")
```

```text
max abs error 1.02e-04
```

Always validate a PINN against a reference solution or an independent
residual evaluation on points that were not used in training; a small
training loss alone does not guarantee an accurate solution.

## 11.7 Training on a GPU

To train on a device, convert the parameters and data to `float32` and hand
the same `loss` to `optim.Trainer`. Parameters and Adam moments then stay on
the device across steps:

```python
p32 = qb.tree.map(lambda t: t.astype(qb.float32),
                  init_mlp(qb.random.key(0), [1, 16, 16, 1]))
trainer = qb.optim.Trainer(
    loss, p32, qb.optim.Adam(learning_rate=1e-2),
    x_interior.astype(qb.float32), x_boundary.astype(qb.float32),
    device="mlx",                       # or "cuda", "cuda:1"
)
for i in range(2000):
    trainer.step()
print(f"loss {trainer.loss().item():.3e}")
params = trainer.params                 # read back once at the end
```

```text
loss 1.235e-04
```

For this tiny model the GPU is not faster than the CPU: per-step dispatch
dominates. GPUs pay off for wider networks and many collocation points; the
benchmarks in `examples/` (`benchmark_pinn_mlx.py`, `benchmark_pinn_cuda.py`)
let you measure the crossover on your hardware.

## 11.8 Variations

**Resampled collocation points.** Draw fresh points each step on the host
and pass them through `batch_argnums`, which keeps one compiled plan because
the shape does not change:

```python
trainer = qb.optim.Trainer(loss, p32, qb.optim.Adam(1e-2), x0, xb,
                           device="mlx", batch_argnums=(0,))
base = qb.random.key(1)
for i in range(steps):
    x = qb.random.uniform(qb.random.fold_in(base, i), (32,), dtype=qb.float32)
    trainer.step(x)
```

**Higher dimensions.** For `u(x, y)`, write `u` for one point `p` of shape
`[2]` and take the Laplacian as the trace of the Hessian:

```python
def laplacian(params, p):
    return qb.trace(qb.hessian(lambda q: u2(params, q))(p))

lap = qb.vmap(lambda p: laplacian(params, p))(points)     # points: [n, 2]
```

**Time-dependent problems.** Treat `t` as another input coordinate and use
`qb.grad(..., argnums=...)` or `jacobian` for `u_t`; or, for method-of-lines
formulations, integrate the semi-discrete system with `qb.ode.odeint`
([Chapter 9](09-ode.md)).

**Inverse problems.** Add the unknown physical coefficient to the parameter
pytree (for example `{"net": mlp_params, "k": qb.array(1.0)}`) and use it in
the residual; `grad` returns its gradient alongside the network's.

The repository's `examples/` directory contains further PINN scripts,
including device-resident training with the lower-level v0.1 helpers.
