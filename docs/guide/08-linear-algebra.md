# 8. Linear Algebra and Solvers

`qb.linalg` provides dense linear algebra over the last two axes of an
array, batched over leading axes, together with matrix-free Krylov solvers.
`qb.newton` solves nonlinear systems. Everything here differentiates to every
order, and the iterative solvers differentiate through the implicit function
theorem rather than through their iterations.

## 8.1 Dense Linear Algebra

```python
a = qb.array([[4.0, 1.0], [1.0, 3.0]])
b = qb.array([1.0, 2.0])

qb.linalg.solve(a, b)                       # [0.09090909, 0.63636364]
c = qb.linalg.cholesky(a)                   # lower factor
qb.linalg.cho_solve(c, b)                   # same solution from the factor
qb.linalg.slogdet(a)                        # SlogdetResult(sign=1., logabsdet=2.3979)
qb.linalg.det(a)                            # 11.
w, v = qb.linalg.eigh(a)                    # eigenvalues ascending: [2.382, 4.618]
q, r = qb.linalg.qr(qb.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))   # [3, 2], [2, 2]
u, s, vh = qb.linalg.svd(qb.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
qb.linalg.lstsq(qb.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]), qb.array([1.0, 2.0, 2.0]))
```

| Function | Result | Method (CPU) |
| --- | --- | --- |
| `solve(a, b)` | `x` with `a @ x == b` | LU with partial pivoting |
| `solve_triangular(a, b, trans=0, lower=False)` | Triangular solve; `trans=1` solves `a^T x = b` | Substitution |
| `cholesky(a)` / `cho_solve(c, b, lower=True)` | Lower factor / solve from it | Cholesky |
| `inv(a)` | `solve(a, I)` | LU |
| `det(a)`, `slogdet(a)` | Determinant; `SlogdetResult(sign, logabsdet)` | LU, sum of `log|u_ii|` |
| `eigh(a)` | `EighResult(eigenvalues, eigenvectors)` of the symmetric part | Cyclic Jacobi, `float64` |
| `qr(a, mode="reduced")` | `QRResult(Q, R)`; modes `reduced`, `complete`, `r` | Householder, `float64` |
| `svd(a, full_matrices=False, compute_uv=True)` | `SVDResult(U, S, Vh)`, `S` descending | One-sided Jacobi, `float64` |
| `lstsq(a, b, return_residuals=False)` | Least squares (tall) or minimum norm (wide) | From `qr` |
| `norm(x, ord=None, axis=None, keepdims=False)` | Vector or matrix norm, NumPy semantics | Scaled to avoid overflow |
| `matrix_power(a, n)` | `a^n` for an integer `n` | Repeated squaring |
| `pinv(a, rtol=None)` | Pseudo-inverse | From `svd` |

`qb.solve` and `qb.cholesky` are the same functions at the top level.

### Conventions

- `b` is treated as a vector `[n]` only when it has rank one (NumPy 2
  semantics); otherwise it is a stack of right-hand sides `[..., n, k]`.
- Leading (batch) axes broadcast like NumPy's:

  ```python
  A = qb.stack([a, 2.0 * a])                         # [2, 2, 2]
  qb.linalg.solve(A, b).shape                         # [2, 2]
  qb.linalg.solve(A, qb.ones((2, 2, 3))).shape        # [2, 2, 3]
  ```

- Factorizations are made unique: eigenvectors and singular vectors have
  their largest-magnitude component positive, and `R` has a non-negative
  diagonal. Results match NumPy up to those sign choices.
- `svd` defaults to `full_matrices=False`, unlike NumPy and JAX.
- A singular matrix raises in `solve` (CPU and MLX). `slogdet` of an exactly
  singular matrix gives `(0, -inf)` as in NumPy.
- `lstsq` does not support rank-deficient matrices (unlike NumPy's SVD-based
  `lstsq`); the result is then non-finite or meaningless without an error.
- Prefer `solve(a, b)` to `inv(a) @ b`: it is faster and more accurate.

### Derivatives

Every derivative rule is written with `solve`, `matmul`, and the
decompositions themselves, so derivatives of every order exist and run on
every backend. No explicit inverse is formed:

```python
qb.grad(lambda a: qb.linalg.slogdet(a).logabsdet)(a)       # == inv(a).T
qb.grad(lambda a: qb.sum(qb.linalg.eigh(a).eigenvalues ** 2))(a)
```

```text
Tensor([[ 0.27272727, -0.09090909],
        [-0.09090909,  0.36363636]], dtype=float64)
Tensor([[8., 2.],
        [2., 6.]], dtype=float64)
```

Derivatives are undefined at some points, and Quabla behaves as JAX does
there: a repeated eigenvalue or singular value makes the *vector* derivatives
infinite or NaN (the vectors are not unique) while the value derivatives
stay defined; derivatives at exactly singular matrices raise on the CPU and
MLX. The exact rules are listed in the
[API Reference](../api.md#device-execution) under `qb.linalg`.

### Backend notes

- **CPU**: decompositions run in `float64` and are rounded for `float32`
  inputs.
- **CUDA**: cuSOLVER (`getrf`/`getrs`, `syevd`, `geqrf`/`orgqr`, `gesvdj`),
  one call per batch element, in `float32` or, with
  `precision="float64"`, in `float64`.
- **MLX**: MLX 0.32 has no GPU kernels for LU, `eigh`, QR, or SVD, so they
  run with LAPACK on MLX's CPU stream (unified memory, no copies), and the
  CPU's sign conventions are applied on the GPU. `solve` substitutes in a
  Metal kernel and reads one pivot flag back to raise on a zero pivot.

## 8.2 Norms

`qb.linalg.norm` follows NumPy: an int `axis` gives a vector norm, a pair of
axes a matrix norm.

```python
qb.linalg.norm(a, "fro")     # 5.19615242
qb.linalg.norm(b, 1)         # 3.
```

Vector orders: `None`/2, 1, `inf`, `-inf`, 0 (count of nonzeros), and any
real `p`. Matrix orders: `None`/`"fro"`, 1, -1, `inf`, `-inf`, 2, -2, and
`"nuc"`. All are computed with scaling, so `float32` norms neither overflow
nor underflow, and the gradient is exact.

## 8.3 Choosing a Solver

| Problem | Use |
| --- | --- |
| Small or medium dense `A x = b` | `linalg.solve` (or `cho_solve` for SPD matrices) |
| Large, symmetric positive-definite operator available only as `x -> A x` | `linalg.cg` |
| Large, general (non-symmetric) operator | `linalg.gmres` |
| Nonlinear `f(x) = 0` | `qb.newton` |
| Minimize a scalar function | `optim.LBFGS` ([Chapter 7](07-optimization.md#74-l-bfgs)) |

## 8.4 Iterative Solvers and Implicit Differentiation

The iterative solvers take a *function* instead of a matrix:

```python
qb.linalg.cg(matvec, b, *, args=(), x0=None, tol=1e-5, atol=0.0,
             maxiter=None, M=None, info=False)
qb.linalg.gmres(matvec, b, *, args=(), x0=None, tol=1e-5, atol=0.0,
                restart=20, maxiter=None, M=None, info=False)
qb.newton(f, x0, *, args=(), tol=None, maxiter=50, info=False)
```

`matvec(x, *args)` applies the operator; `M(r, *args)` optionally applies a
preconditioner. Here is a 1D finite-difference Laplacian, scaled by a
parameter `k`, without ever forming the matrix:

```python
n = 8

def laplacian(x, k):
    left = qb.concat([qb.zeros((1,)), x[:-1]], axis=0)
    right = qb.concat([x[1:], qb.zeros((1,))], axis=0)
    return k * (2.0 * x - left - right)

rhs = qb.ones((n,))
x, info = qb.linalg.cg(laplacian, rhs, args=(qb.array(1.0),), info=True)
print(x)
print(info)
```

```text
Tensor([ 4.,  7.,  9., 10., 10.,  9.,  7.,  4.], dtype=float64)
{'iterations': Tensor(4., dtype=float64), 'residual_norm': Tensor(0., dtype=float64), 'success': Tensor( True, dtype=bool)}
```

### Differentiating a solution

Derivatives with respect to `b` and to the array leaves of `args` use the
implicit function theorem at the solution: one adjoint solve, independent of
how many iterations the forward solve took.

```python
def total(k):
    return qb.sum(qb.linalg.cg(laplacian, rhs, args=(k,), tol=1e-10))

qb.grad(total)(qb.array(2.0))                      # -15. (= -60 / k^2)
qb.jit(qb.grad(total))(qb.array(2.0))              # same, compiled
qb.vmap(total)(qb.array([1.0, 2.0]))               # [60., 30.]
```

**Pass every array the operator depends on through `args`.** Values that
`matvec` closes over are treated as constants and are not differentiated.

Other rules:

- Reverse mode composes twice (`grad(grad(...))`, reverse-mode `jacobian`
  and `hessian`). Forward mode (`jvp`) is not supported. `x0` gets no
  gradient.
- Eagerly the solver runs in Python; under `jit` it is one `while_loop`
  region that stops at convergence: `||b - A x|| <= max(tol * ||b||, atol)`
  for the linear solvers.
- Under `vmap`, each example stops at its own tolerance (converged examples
  are frozen), so a batch costs the iterations of its slowest example.
- An eager solve that does not converge raises `RuntimeError`. Under `jit`
  it cannot raise; pass `info=True` to get `iterations`, `residual_norm`,
  and `success` (per example under `vmap`).
- A solver cannot run inside a `cond`, `fori_loop`, or `scan` body.

`gmres` is restarted, right-preconditioned GMRES with a twice-applied
classical Gram-Schmidt Arnoldi process:

```python
def upwind(x, c):
    left = qb.concat([qb.zeros((1,)), x[:-1]], axis=0)
    return x + c * (x - left)

qb.linalg.gmres(upwind, rhs, args=(qb.array(0.5),))
```

## 8.5 Newton's Method

`qb.newton(f, x0, args=...)` finds a root of `f(x, *args) = 0` with a dense
Jacobian (formed with `qb.jacobian`), Householder QR steps, and an Armijo
backtracking line search on `||f||^2`. `tol` defaults to the square root of
the dtype's machine epsilon, and convergence is `||dx|| <= tol * (1 + ||x||)`.

```python
def f(x, p):
    return x ** 3 - p                         # root: the cube root of p

p = qb.array([8.0, 27.0])
qb.newton(f, qb.array([1.0, 1.0]), args=(p,))                            # [2., 3.]
qb.grad(lambda p: qb.sum(qb.newton(f, qb.array([1.0, 1.0]), args=(p,))))(p)
```

```text
Tensor([2., 3.], dtype=float64)
Tensor([0.08333333, 0.03703704], dtype=float64)
```

The gradient is `1 / (3 p^(2/3))`, obtained from the implicit function
theorem. The differentiation, `jit`, `vmap`, `info`, and error rules are the
same as for the Krylov solvers. An exactly singular Jacobian stops the
iteration with failure.
