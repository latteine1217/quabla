# 15. Migrating From the v0.1 API

Quabla v0.1 exposed transforms as spec-based helpers (`trace_tensor`,
`tensor_value_and_grad_fn`, `tensor_vmap_mlx_fn`, ...), with one spelling per
combination of transform and backend. v0.2 replaced them with the
composable core API. This chapter shows how to move existing code.

## 15.1 Compatibility Guarantees

- Every Python name released in v0.1 keeps working, with unchanged behavior,
  through all 0.x releases.
- Names that have a replacement emit a `DeprecationWarning` once per name on
  explicit access. The message names the replacement:

  ```text
  DeprecationWarning: quabla.tensor_jit_fn is deprecated; use qb.jit(f). The old name remains available through 0.x.
  ```

- `from quabla import *` binds the old names silently; only explicit access
  warns.
- The legacy 2D API (`Matrix`, `TraceGraph`, `grad_fn`, ...) is available
  without warnings from `quabla.legacy`.
- `quabla.Adam` aliases `quabla.optim.Adam`, which keeps the v0.1 stateful
  `step(params, grads)` method.
- `qb.grad(fn, input_specs, values, output_cotangent)`, `qb.jit(input_specs)`,
  and `qb.trace(fn, input_specs)` (the v0.1 call forms) are still dispatched
  to the v0.1 functions.
- Deprecated names are removed at 1.0.

To find every use of a deprecated name in your code, turn the warnings into
errors while running your tests:

```sh
python -W error::DeprecationWarning your_script.py
```

## 15.2 The Conceptual Change

| v0.1 | Core API |
| --- | --- |
| Inputs are declared up front as `(name, shape, dtype)` specs | Inputs are ordinary arguments; shapes and dtypes come from the values on the first call |
| Results are dicts keyed by input name | Results mirror the argument pytrees |
| One helper per transform and backend (`tensor_vmap_vjp_cuda_fn`) | Transforms compose (`jit(vjp(vmap(f)), device="cuda")`) |
| Derivatives of a trace via `symbolic_jvp(name)` on a `TensorTraceResult` | `grad`, `jvp`, `jacobian` of a Python function |
| Bounded batch specialization via `tensor_jit_batch_fn(max_specializations=4)` | Every transform caches up to `max_traces` signatures |
| Device optimizers (`mlx_adam_loss_optimizer`, `cuda_adam_vjp_optimizer`) | `optim.Trainer(..., device=...)` |

## 15.3 A Worked Example

The v0.1 example `examples/pinn_poisson.py` fits one weight `w` so that
`sin(w x)` solves `u'' = -pi^2 sin(pi x)` with zero boundary values. In
v0.1 it builds two separately traced plans, takes symbolic JVPs by input
name, and sums the gradients by hand:

```python
residual_trace = quabla.trace_tensor(
    lambda x, weight, forcing: (x * weight).sin(),
    [("x", [5, 1]), ("weight", [1, 1]), ("forcing", [5, 1])],
)
second_derivative = residual_trace.symbolic_jvp("x").symbolic_jvp("x")
residual = second_derivative.output + second_derivative.graph.input("forcing")
residual_plan = (residual * residual).mean().compile_cpu()
# ... a second plan for the boundary loss ...
for _ in range(2000):
    _, residual_gradients = residual_plan.evaluate_value_and_vjp({...}, quabla.Tensor([], [1.0]))
    _, boundary_gradients = boundary_plan.evaluate_value_and_vjp({...}, quabla.Tensor([], [1.0]))
    parameters = optimizer.step(
        parameters,
        quabla.sum_gradients([residual_gradients, boundary_gradients], ["weight"]),
    )
```

With the core API, the model is written for one point, the loss is one
function, and `Trainer` runs the loop:

```python
import math
import quabla as qb

def u(x, w):
    return qb.sin(x * w)

u_xx = qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None))

def loss(params, x, x_boundary):
    residual = u_xx(x, params["weight"]) + math.pi**2 * qb.sin(math.pi * x)
    boundary = qb.vmap(u, in_axes=(0, None))(x_boundary, params["weight"])
    return qb.mean(residual ** 2) + qb.mean(boundary ** 2)

trainer = qb.optim.Trainer(
    loss, {"weight": qb.array(2.5)}, qb.optim.Adam(learning_rate=0.01),
    qb.array([0.15, 0.35, 0.55, 0.75, 0.9]), qb.array([0.0, 1.0]),
)
for _ in range(2000):
    trainer.step()
print(trainer.params["weight"].item())
```

```text
3.1415926535897936
```

Both versions converge to the same value. Changing the backend of the new
version is one keyword (`device="mlx"` or `device="cuda"`, with `float32`
inputs).

## 15.4 Name Mapping

The most common replacements:

| v0.1 name | Replacement |
| --- | --- |
| `trace_tensor(fn, specs)` | `qb.jit(fn).lower(...)` or `qb.Compiler().trace(fn, specs)` |
| `tensor_jit_fn` | `qb.jit(f)` |
| `tensor_jit_cuda_fn` | `qb.jit(f, device="cuda")` |
| `tensor_jit_batch_fn` | `qb.jit(f, max_traces=4)` |
| `tensor_value_and_grad_fn` | `qb.value_and_grad(f, argnums)` |
| `tensor_value_and_grad_{mlx,cuda}_fn` | `qb.jit(qb.value_and_grad(f, argnums), device="mlx" \| "cuda")` |
| `tensor_value_and_grad_batch_*_fn` | The same with `max_traces=4` |
| `tensor_grad_scalar_fn` | `qb.grad(f, argnums=(0, ..., n-1))` |
| `tensor_vjp_fn`, `tensor_jvp_fn` | `qb.vjp(f, *primals)`, `qb.jvp(f, primals, tangents)` |
| `tensor_jacobian_fn` | `qb.jacobian(f, argnums=i)` |
| `tensor_hessian_scalar_fn` | `qb.hessian(f, argnums=i)` |
| `tensor_hvp_scalar_fn` | `qb.jvp(qb.grad(f), (x,), (v,))` |
| `tensor_vmap_fn` | `qb.vmap(f, in_axes, out_axes)` |
| `tensor_vmap_{mlx,cuda}_fn` | `qb.jit(qb.vmap(f, in_axes, out_axes), device=...)` |
| `tensor_vmap_{jvp,vjp}_*_fn` | `qb.jvp`/`qb.vjp` of `qb.vmap(f)`, under `jit(device=...)` |
| `tensor_vmap_hvp_scalar_fn` | `qb.jvp(qb.grad(lambda x: qb.sum(qb.vmap(f)(x))), (x,), (v,))` |
| `tensor_cond` | `qb.cond(pred, true_fun, false_fun, *operands)` |
| `tensor_fori_loop_region` / `tensor_fori_loop` | `qb.fori_loop(..., operands=...)` / with `unroll=True` |
| `tensor_scan_region` / `tensor_scan` | `qb.scan(f, init, length=n, operands=...)` / with `unroll=True` |
| `tensor_value_and_grad_data_parallel_cuda_fn` | `qb.distributed.value_and_grad(f, devices=..., shard_argnums=...)` |
| `mlx_adam_loss_optimizer`, `cuda_adam_*_optimizer`, `cuda_adam_step` | `qb.optim.Trainer(..., device="mlx" \| "cuda")` |
| `sum_gradients` | `qb.tree.map(operator.add, a, b)` |
| `Adam` | `qb.optim.Adam` (`init`/`update`; `step` still works) |
| `Matrix`, `TraceGraph`, `grad_fn`, `jvp_fn`, ... | `quabla.legacy.<name>` |

The complete mapping, with the status of each name, is Appendix A of the
[v0.2 API design](../api_v0_2_design.md#appendix-a-name-mapping); the
deprecation messages come from `python/quabla/_compat.py`.

## 15.5 Things That Behave Differently

- **Specs versus values.** The new API infers shapes and dtypes from the
  first call. To trace without data, pass `qb.ShapeDtype` placeholders to
  `jit(...).lower`.
- **Gradient containers.** v0.1 returned dicts keyed by input name; the new
  transforms return gradients shaped like the differentiated argument.
- **Retracing instead of errors.** The v0.1 batch helpers raised once their
  specialization limit was reached; transform caches now evict the least
  recently used trace and warn with `RetraceWarning`.
- **Jacobian layout.** `tensor_jacobian_fn` returned an output-flat by
  input-flat matrix; `qb.jacobian` returns blocks of shape
  `[*out.shape, *in.shape]`.
- **Device helpers versus `Trainer`.** The v0.1 device optimizers expose
  low-level controls (retained input names, per-plan synchronization).
  `Trainer` covers the common training loop; the low-level helpers remain
  available when you need those controls.
