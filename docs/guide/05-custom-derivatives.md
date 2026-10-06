# 5. Custom Derivatives and Checkpointing

Automatic differentiation of a correct formula is not always a good
derivative. A formula can be accurate for values and still overflow in its
derivative, or a derivative can be undefined at a point where you want a
specific convention. Quabla follows JAX here with three wrappers:

| Wrapper | Replaces | Typical use |
| --- | --- | --- |
| `qb.custom_vjp(fun)` + `defvjp(fwd, bwd)` | The reverse-mode rule | Stable gradients, implicit layers, gradients through external computations |
| `qb.custom_jvp(fun)` + `defjvp(rule)` | The forward-mode rule (reverse mode is derived by transposition) | Conventions at singular points, cheaper tangents |
| `qb.checkpoint(fun)` (alias `qb.remat`) | Nothing: it changes what reverse mode stores | Trading recomputation for memory |

And one primitive, `qb.stop_gradient`, for blocking a derivative.

## 5.1 `custom_vjp`

```python
@qb.custom_vjp
def log1pexp(x):
    return qb.log(1.0 + qb.exp(x))

def log1pexp_fwd(x):
    return log1pexp(x), x                     # (output, residuals for bwd)

def log1pexp_bwd(x, g):                       # (residuals, output cotangent)
    return (g * qb.sigmoid(x),)               # one cotangent per argument

log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)

x = qb.array([0.0, 100.0, 1000.0])
qb.grad(lambda x: qb.sum(log1pexp(x)))(x)                  # custom rule
qb.grad(lambda x: qb.sum(qb.log(1.0 + qb.exp(x))))(x)      # plain autodiff
```

```text
Tensor([0.5, 1. , 1. ], dtype=float64)
Tensor([0.5, 1. , nan], dtype=float64)
```

The automatic derivative of `log(1 + exp(x))` divides two overflowing
quantities at `x = 1000`; the custom rule uses the stable `sigmoid`. (For
this particular function you would simply use `qb.softplus`, which is already
stable.)

The contract:

- `fwd(*args)` returns `(out, residuals)`. `out` must have the structure of
  `fun(*args)`; `residuals` is any pytree passed to `bwd`.
- `bwd(*nondiff_args, residuals, cotangent)` returns a **tuple** with one
  cotangent per differentiable argument, each with that argument's pytree
  structure. `None` means a zero cotangent.
- Reverse mode (`grad`, `value_and_grad`, `vjp`, and `jacobian`, which uses
  reverse mode for such functions) evaluates `fwd` and `bwd`.
- Forward mode (`jvp`) raises `ValueError`, as in JAX: only a reverse rule
  was defined.
- `bwd` is ordinary traced code, so `grad(grad(...))` and `hessian`
  differentiate through it.
- `vmap` batches the rules together with the function, and `jit` compiles
  them on every device: `qb.jit(qb.vmap(qb.grad(log1pexp)))(x)` works.

### Non-differentiable arguments

Arguments listed in `nondiff_argnums` are passed through unchanged and get
no cotangent. They are passed first to `bwd`, and they must not be traced
arrays (use them for Python values such as a scale factor or a tolerance):

```python
from functools import partial

@partial(qb.custom_vjp, nondiff_argnums=(0,))
def scaled(k, x):
    return k * x

scaled.defvjp(lambda k, x: (scaled(k, x), None),     # fwd(k, x)
              lambda k, res, g: (k * g,))             # bwd(k, residuals, g)

qb.grad(lambda x: qb.sum(scaled(3.0, x)))(qb.array([1.0, 2.0]))   # [3., 3.]
```

## 5.2 `custom_jvp`

```python
@qb.custom_jvp
def safe_norm(x):
    return qb.sqrt(qb.sum(x * x))

@safe_norm.defjvp
def safe_norm_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    norm = safe_norm(x)
    return norm, qb.sum(x * t) / qb.where(qb.equal(norm, 0.0), 1.0, norm)

qb.grad(safe_norm)(qb.zeros((3,)))            # [0., 0., 0.] instead of NaN
qb.grad(safe_norm)(qb.array([3.0, 4.0]))      # [0.6, 0.8]
qb.jvp(safe_norm, (qb.array([3.0, 4.0]),), (qb.array([1.0, 0.0]),))   # (5., 0.6)
```

- `rule(*nondiff_args, primals, tangents)` returns
  `(primal_out, tangent_out)`; `tangent_out` must be **linear** in
  `tangents`.
- Forward mode uses the rule directly. Reverse mode transposes it, by
  differentiating `tangent_out` with respect to `tangents` at zero tangents,
  which is exact for a linear rule. Higher orders differentiate the rule.
- The primal value is always `fun(*args)`; the rule's `primal_out` is not
  used for the value.
- Inside its own rule, a call of the wrapped function evaluates `fun` without
  the rule, which is what makes `norm = safe_norm(x)` above legal.

## 5.3 `checkpoint`

Reverse mode normally stores the intermediates of the forward pass until the
backward pass consumes them. `checkpoint(fun)` instead recomputes `fun`'s
intermediates during the backward pass from its arguments. Values and
derivatives of every order are unchanged; only memory and compute change.

```python
w = qb.array([[0.5, -0.2], [0.1, 0.3]])
b = qb.array([0.1, 0.0])

layer = qb.checkpoint(lambda h, w, b: qb.tanh(h @ w + b))

def net(h, w, b):
    for _ in range(4):
        h = layer(h, w, b)
    return qb.sum(h)

qb.grad(net, argnums=1)(qb.ones((3, 2)), w, b)
```

- The recomputation is part of the compiled plan (plan CSE does not merge it
  back with the forward pass), so the forward intermediates are not live
  until the backward pass. The actual memory saving depends on the backend
  freeing buffers after their last use.
- `static_argnums` marks arguments that are passed through unchanged.
- Unlike the custom-rule wrappers, a checkpointed function may close over
  tracers of an enclosing transform, and it differentiates through them.

## 5.4 Common Rules for All Three Wrappers

- Outside any transform, the wrapped function simply calls `fun`.
- Inside a transform, the call is staged: the rule graphs are traced once per
  call and recorded in a `custom` IR node. The symbolic transforms apply the
  rule, and plan compilation removes the node, so nothing custom remains at
  run time and the result runs on every backend.
- `fun`, `fwd`, `bwd`, and the JVP rule must not close over tracers of an
  enclosing transform (`TypeError`); pass such values as arguments.
- Every traced output must be floating-point.
- Python scalars in differentiable arguments are constants.
- These functions may be called inside `cond`, `fori_loop`, `scan`, and
  `while_loop` bodies. The compiled region keeps the rule, so the region's
  derivatives (`grad`, `jvp`, Hessians, and `vmap` of them) apply it, as at
  the top level; forward mode through a `custom_vjp` function still raises,
  and a `while_loop` has forward mode only.

## 5.5 `stop_gradient`

`qb.stop_gradient(x)` is the identity on values with a zero derivative in
every mode. A common use is the straight-through estimator, which takes the
value of one expression and the gradient of another:

```python
f = lambda x: x - qb.stop_gradient(x) + qb.stop_gradient(qb.round(x))
f(qb.array(1.3))             # Tensor(1.)  -- the value of round(x)
qb.grad(f)(qb.array(1.3))    # Tensor(1.)  -- the gradient of x
```

Other uses: freezing part of a model, target networks, and fixed-point
iterations where only the final step should be differentiated.
