# 9. Differential Equations

`qb.ode.odeint` integrates an ordinary differential equation
`dy/dt = f(y, t, *args)` and is differentiable with respect to the initial
state, the time span, the save times, and every array in `args`. It runs
eagerly, under `jit` on every backend, and under `vmap`.

```python
qb.ode.odeint(f, y0, (t0, t1), *, steps=None, method="rk4", args=(),
              save=False, saveat=None, rtol=1e-6, atol=1e-9,
              max_steps=512, info=False)
```

| `method` | Kind | Step control |
| --- | --- | --- |
| `"rk4"` (default) | Classical 4th-order Runge-Kutta | `steps` equal steps |
| `"heun"` | 2nd-order Heun | `steps` equal steps |
| `"euler"` | Forward Euler | `steps` equal steps |
| `"dopri5"` | Dormand-Prince 5(4), explicit | Adaptive, `rtol`/`atol` |
| `"rosenbrock23"` | Rosenbrock 2(3), L-stable (MATLAB's `ode23s`) | Adaptive, for stiff problems |

## 9.1 The Right-Hand Side

`f(y, t, *args)` receives the state `y` (one array), the time `t` (a scalar
array), and the extra arguments. Two rules follow from the fact that the
integrator is a loop region ([Chapter 4](04-control-flow.md#41-the-operand-rule)):

- `y0` is a **single array**. Pack a multi-component state into one array.
- `f` must receive traced values (parameters you differentiate with respect
  to) through `args`, not through a closure.

```python
def decay(y, t, k):
    return -k * y

y0, k = qb.array([1.0, 2.0]), qb.array(0.5)
y1 = qb.ode.odeint(decay, y0, (0.0, 2.0), steps=50, args=(k,))
print(y1)                                    # exact: y0 * exp(-1)
```

```text
Tensor([0.36787944, 0.73575888], dtype=float64)
```

## 9.2 Fixed-Step Methods

`steps` is required and fixes the step size `dt = (t1 - t0) / steps`. Step
times are computed as `t0 + i * dt`, so long integrations do not accumulate
drift in `t`.

`save=True` returns the `steps + 1` states at `t0, ..., t1` stacked on axis
zero:

```python
traj = qb.ode.odeint(decay, y0, (0.0, 2.0), steps=4, args=(k,), save=True)
traj.shape          # [5, 2]
```

A traced fixed-step solve is one `fori_loop` region (a `scan` region with
`save=True`), so it differentiates in reverse mode with checkpointing:

```python
loss = lambda k: qb.sum(qb.ode.odeint(decay, y0, (0.0, 2.0), steps=50, args=(k,)))
qb.grad(loss)(k)    # Tensor(-2.20727664); exact: -2 * 3 * exp(-1) = -2.2072766
```

The gradient is that of the discrete RK4 scheme, which here agrees with the
continuous derivative to the scheme's accuracy.

## 9.3 Adaptive Integration: `dopri5`

The adaptive methods choose their own steps. A step is accepted when the RMS
norm of the error estimate, scaled by `atol + rtol * max(|y|, |y_new|)`, is
at most one; a PI controller picks the next step size.

```python
def lotka_volterra(y, t, theta):
    prey, pred = y[0], y[1]
    a, b, c, d = theta[0], theta[1], theta[2], theta[3]
    return qb.stack([a * prey - b * prey * pred, c * prey * pred - d * pred])

theta = qb.array([1.5, 1.0, 1.0, 3.0])
y, info = qb.ode.odeint(lotka_volterra, qb.array([1.0, 1.0]), (0.0, 5.0),
                        method="dopri5", args=(theta,), info=True)
print(y)
print(info)
```

```text
Tensor([6.09850076, 0.62814261], dtype=float64)
{'t': Tensor(5., dtype=float64), 'accepted_steps': Tensor(53., dtype=float64), 'rejected_steps': Tensor(0., dtype=float64), 'success': Tensor( True, dtype=bool)}
```

How an adaptive solve behaves depends on whether it is traced:

| | Eager | Traced (`jit`, `grad`, `vmap`) |
| --- | --- | --- |
| Iterations | Stops at `t1` | Always runs `max_steps` attempts; iterations after `t1` are masked |
| Not reaching `t1` | Raises `RuntimeError` | Cannot raise: returns the state at the last time reached; check `info["success"]` |

The traced form is a bounded `fori_loop` so that it is reverse-mode
differentiable. Consequently **`max_steps` is the cost of a traced solve**:
set it to a comfortable bound for your problem, not to a huge safety margin.

Step sizes are controller outputs without gradients, so derivatives are
those of the discrete scheme on the chosen step mesh. They are piecewise
smooth: a parameter change that flips an accept/reject decision changes the
mesh.

## 9.4 Stiff Problems: `rosenbrock23`

A stiff problem forces an explicit method to take tiny steps for stability
long after accuracy would allow large ones. `rosenbrock23` is L-stable: each
step forms the Jacobian `df/dy` (and `df/dt`) by forward mode, in one batched
pass, and solves three linear systems with `W = I - h d J`. `f` needs no
hand-written Jacobian.

```python
def relax(y, t, lam):                 # y relaxes to cos(t) with rate lam
    return -lam * (y - qb.cos(t))

for method in ("rosenbrock23", "dopri5"):
    y, info = qb.ode.odeint(relax, qb.array([0.0]), (0.0, 1.0), method=method,
                            args=(qb.array(1e4),), rtol=1e-4, atol=1e-7,
                            max_steps=100000, info=True)
    print(method, y, info["accepted_steps"])
```

```text
rosenbrock23 Tensor([0.54038875], dtype=float64) Tensor(151., dtype=float64)
dopri5 Tensor([0.54038302], dtype=float64) Tensor(3036., dtype=float64)
```

`rosenbrock23` is second order, so tight tolerances need many steps; with
the defaults (`rtol=1e-6`, `atol=1e-9`) this problem exceeds the default
`max_steps=512`, and the eager solve says so:

```text
RuntimeError: rosenbrock23 stopped at t=0.38979589336576453 before t1 after max_steps=512 step attempts; increase max_steps or loosen rtol/atol
```

Use `rosenbrock23` with moderate tolerances for stiff problems and `dopri5`
for non-stiff problems or high accuracy. The `W` solves use `solve`, so a
singular `W` raises like `solve`.

## 9.5 Dense Output: `saveat`

`saveat=ts` returns the states at the times `ts`, stacked on axis zero,
instead of only the final state. It works with every method:

```python
ts = qb.linspace(0.0, 5.0, 6)
ys = qb.ode.odeint(lotka_volterra, qb.array([1.0, 1.0]), (0.0, 5.0),
                   method="dopri5", args=(theta,), saveat=ts)
ys.shape            # [6, 2]
```

- `ts` is 1-D, ordered from `t0` to `t1`, and inside the span.
- Steps are not shortened to land on the save times. Values come from each
  method's continuous extension: Shampine's 4th-order interpolant for
  `dopri5`, the method's own interpolant for `rosenbrock23`, and cubic
  Hermite interpolation for the fixed-step methods.
- Under `jit`, a save time outside the span or one the bounded loop did not
  reach is NaN.
- `saveat` and `save=True` cannot be combined.

## 9.6 Parameter Estimation

Because the saved states are differentiable in `args`, fitting ODE
parameters to observations is a short program:

```python
observed = ys                                    # data at times ts

def fit_loss(theta):
    pred = qb.ode.odeint(lotka_volterra, qb.array([1.0, 1.0]), (0.0, 5.0),
                         method="dopri5", args=(theta,), saveat=ts, max_steps=200)
    return qb.mean((pred - observed) ** 2)

step = qb.jit(qb.value_and_grad(fit_loss))
value, grad = step(theta + 0.1)
```

```text
(Tensor(0.50908184, dtype=float64), Tensor([ 8.83731214, -2.42140546,  5.48501747,  0.26664252], dtype=float64))
```

Feed `step` into Adam or L-BFGS ([Chapter 7](07-optimization.md)) to
recover `theta`. Use `qb.vmap` over initial conditions or parameter sets to
fit several trajectories at once.

## 9.7 Devices

`odeint` runs under `jit(device="cuda")` and `jit(device="mlx")`. The
adaptive integrators slice and concatenate a packed carry and take norms in
their loop body, so on CUDA they run as host-driven region loops (one launch
per step attempt); see [Devices](10-devices.md#104-cuda).
