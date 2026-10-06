"""Dense linear algebra over the last two axes, batched over leading axes.

Every dense function takes eager or traced arrays, so it runs eagerly, under
`jit`, `grad`, `jacfwd`/`jacrev`, and `vmap`, and has derivatives of every
order (each derivative rule is written with `solve`, `matmul`, and the
decompositions themselves). Leading batch axes of two operands broadcast
against each other as in NumPy.

- `solve(a, b)`: `a @ x == b` by LU with partial pivoting, per matrix.
  `b` is a vector `[n]` only when it has rank one (NumPy 2 semantics);
  otherwise it is a stack `[..., n, k]`.
- `solve_triangular(a, b, trans=0, lower=False)`: reads only the `lower`
  (or upper) triangle of `a`; `trans=1` solves `a^T x = b`.
- `cholesky(a)`: the lower factor `L` with `a = L L^T`.
- `cho_solve(c, b, lower=True)`: solves `a x = b` from a Cholesky factor of
  `a` with two triangular solves.
- `slogdet(a)`: `(sign, logabsdet)` from the LU factorization; `det(a)` is
  `sign * exp(logabsdet)`.
- `inv(a)`: `solve(a, I)`; prefer `solve(a, b)` to `inv(a) @ b`, which is
  slower and less accurate.
- `eigh(a)`: eigenvalues (ascending) and eigenvectors of the symmetric part
  `(a + a^T) / 2`.
- `qr(a, mode="reduced")`: `a = Q R` by Householder reflections, with the
  diagonal of `R` non-negative; `a` may be any `[..., m, n]`.
- `svd(a, full_matrices=False, compute_uv=True)`: `a = U diag(S) Vh` by
  one-sided Jacobi rotations, singular values descending.
- `lstsq(a, b)`: the least-squares solution of `a @ x ~= b` for a tall `a`
  of full column rank (the minimum-norm solution for a wide `a` of full row
  rank), from `qr`.
- `norm(x, ord=None, axis=None, keepdims=False)`: vector and matrix norms
  as NumPy, scaled so that float32 does not overflow.
- `matrix_power(a, n)`: by repeated squaring; a negative `n` uses `inv`.
- `pinv(a, rtol=None)`: the pseudo-inverse from `svd`.

No derivative forms an explicit inverse: the gradient of `logabsdet` is
`solve(a^T, g I)` and its tangent `trace(solve(a, da))`. The derivatives of
`slogdet`, `det`, and `inv` at an exactly singular matrix are undefined:
`slogdet` itself returns `(0, -inf)` there, but the derivative's `solve`
raises on the CPU and MLX (and reports the singular factor on CUDA), where
JAX returns non-finite values. The eigenvector derivative divides by eigenvalue
gaps, so it is infinite or NaN for a repeated eigenvalue, as in JAX; the
eigenvalue derivative stays defined. The `qr` derivative needs a matrix of
full column rank (for a wide `a`, its leading square block of full rank)
and applies `R^-1` with `solve`; the `svd` derivatives follow JAX's rule and
are exact for distinct nonzero singular values (see `svd`). The extra
columns of `mode="complete"` and `full_matrices=True` are not unique, so
differentiating them raises.

Devices: CPU, CUDA (cuSOLVER `getrf`/`getrs`, `syevd`, `geqrf`/`orgqr`,
and `gesvdj`, `float32`), and MLX, whose factorizations exist only on its CPU
stream: LAPACK `getrf`, `syevd`, `geqrf`/`orgqr`, and `gesdd` run there in
`float32`, the CPU's sign and completion conventions are applied on the GPU
stream, and `solve` substitutes with the LU factors in a Metal kernel after
reading back one flag that raises for an exactly zero pivot. Whether a pivot
of a singular matrix is exactly zero depends on rounding, so a matrix that
is singular only in exact arithmetic may be solved with huge values instead
of raising, on any backend.

Two matrix-free iterative solvers take a function `matvec(x, *args)`
instead of a matrix (see `_krylov.py`): `cg` for symmetric positive-definite
operators and `gmres` for general ones. They differ from the functions
above: their derivatives come from the implicit function theorem at the
solution, in forward and reverse mode (any two passes), not from the
iterations.
"""

import collections
import math
import numbers
import operator

from ._array import asarray, eye, zeros
from ._krylov import cg, gmres
from ._quabla import Tensor, TraceTensor, concat, float32, where

__all__ = [
    "EighResult",
    "LstsqResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "cg",
    "cho_solve",
    "cholesky",
    "det",
    "eigh",
    "gmres",
    "inv",
    "lstsq",
    "matrix_power",
    "norm",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
]

SlogdetResult = collections.namedtuple("SlogdetResult", ["sign", "logabsdet"])
SlogdetResult.__doc__ = "`(sign, logabsdet)` of `slogdet`; a pytree like any namedtuple."
EighResult = collections.namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
EighResult.__doc__ = "`(eigenvalues, eigenvectors)` of `eigh`; a pytree like any namedtuple."
QRResult = collections.namedtuple("QRResult", ["Q", "R"])
QRResult.__doc__ = "`(Q, R)` of `qr`; a pytree like any namedtuple."
SVDResult = collections.namedtuple("SVDResult", ["U", "S", "Vh"])
SVDResult.__doc__ = "`(U, S, Vh)` of `svd`; a pytree like any namedtuple."
LstsqResult = collections.namedtuple("LstsqResult", ["solution", "residuals"])
LstsqResult.__doc__ = "`(solution, residuals)` of `lstsq`; a pytree like any namedtuple."


def _array(x):
    return x if isinstance(x, (Tensor, TraceTensor)) else asarray(x)


def _square(a, name):
    a = _array(a)
    shape = tuple(a.shape)
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(f"{name} requires a stack of square matrices [..., n, n], got {list(shape)}")
    return a


def _matrix(a, name):
    a = _array(a)
    if len(a.shape) < 2:
        raise ValueError(f"{name} requires a stack of matrices [..., m, n], got {list(a.shape)}")
    return a


def _matrix_transpose(x):
    axes = list(range(len(x.shape)))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return x.transpose(axes)


def _broadcast_batch(*shapes):
    """The NumPy broadcast of the batch shapes `shapes`."""
    rank = max(len(shape) for shape in shapes)
    result = []
    for axis in range(rank):
        extents = {shape[axis - rank + len(shape)] for shape in shapes if axis >= rank - len(shape)}
        extents.discard(1)
        if len(extents) > 1:
            raise ValueError(f"cannot broadcast batch shapes {[list(shape) for shape in shapes]}")
        result.append(extents.pop() if extents else 1)
    return tuple(result)


def _to_batch(x, batch):
    shape = tuple(batch) + tuple(x.shape[-2:])
    return x if tuple(x.shape) == shape else x.broadcast_to(list(shape))


def _solve_operands(a, b, name):
    """`a` and `b` broadcast to a common batch, `b` as a stack of column
    blocks, and whether `b` was a vector."""
    a = _square(a, name)
    b = _array(b)
    vector = len(b.shape) == 1
    if vector:
        b = b.reshape([b.shape[0], 1])
    elif len(b.shape) < 2:
        raise ValueError(f"{name} requires a right-hand side of rank one or more, got {list(b.shape)}")
    if b.shape[-2] != a.shape[-1]:
        raise ValueError(
            f"{name} requires matrix shape {list(a.shape)} and right-hand side shape "
            f"{list(b.shape)} to agree on rows"
        )
    batch = _broadcast_batch(tuple(a.shape[:-2]), tuple(b.shape[:-2]))
    return _to_batch(a, batch), _to_batch(b, batch), vector


def _restore_vector(x, vector):
    return x.reshape(list(x.shape[:-1])) if vector else x


def solve(a, b):
    """The solution `x` of `a @ x == b` for every matrix of the leading axes,
    by LU with partial pivoting; `b` is `[..., n, k]`, or a vector `[n]`
    shared by every matrix. A singular matrix raises."""
    a, b, vector = _solve_operands(a, b, "solve")
    return _restore_vector(a.solve(b), vector)


def _trans_flag(trans):
    if trans in (0, "N"):
        return False
    if trans in (1, "T", "C"):
        return True
    raise ValueError(f"trans must be 0, 1, 'N', 'T', or 'C', got {trans!r}")


def solve_triangular(a, b, trans=0, lower=False):
    """The solution `x` of `t @ x == b` (`trans=0`) or `t^T @ x == b`
    (`trans=1`), where `t` is the lower (`lower=True`) or upper triangle of
    `a`; the other triangle of `a` is never read. Shapes follow `solve`."""
    a, b, vector = _solve_operands(a, b, "solve_triangular")
    return _restore_vector(a.solve_triangular(b, bool(lower), _trans_flag(trans)), vector)


def cholesky(a):
    """The lower Cholesky factor `L` (`a = L @ L^T`) of each symmetric
    positive-definite matrix of the leading axes."""
    return _square(a, "cholesky").cholesky()


def cho_solve(c, b, lower=True):
    """The solution `x` of `a @ x == b` given the Cholesky factor `c` of
    `a`: lower (`a = c @ c^T`) by default, upper (`a = c^T @ c`) with
    `lower=False`. Two triangular solves; `a` is never formed."""
    if lower:
        y = solve_triangular(c, b, trans=0, lower=True)
        return solve_triangular(c, y, trans=1, lower=True)
    y = solve_triangular(c, b, trans=1, lower=False)
    return solve_triangular(c, y, trans=0, lower=False)


def slogdet(a):
    """`SlogdetResult(sign, logabsdet)` of each matrix of the leading axes,
    from an LU factorization with partial pivoting: `det(a) = sign *
    exp(logabsdet)` without overflow. An exactly singular matrix gives
    `(0, -inf)`, a non-finite one `(nan, nan)`. `sign` has a zero derivative;
    the gradient of `logabsdet` is `a^-T`, computed with `solve`."""
    a = _square(a, "slogdet")
    return SlogdetResult(a._linalg("det_sign"), a._linalg("log_abs_det"))


def det(a):
    """The determinant `sign * exp(logabsdet)` of each matrix of the leading
    axes (see `slogdet`); exactly `0` for an exactly singular matrix, whose
    derivative is undefined here."""
    sign, logabsdet = slogdet(a)
    return sign * logabsdet.exp()


def inv(a):
    """The inverse of each matrix of the leading axes, as `solve(a, I)`.
    To apply an inverse, `solve(a, b)` is faster and more accurate than
    `inv(a) @ b`."""
    a = _square(a, "inv")
    n = a.shape[-1]
    identity = eye(n, dtype=a.dtype)
    return a.solve(_to_batch(identity, tuple(a.shape[:-2])))


def eigh(a):
    """`EighResult(eigenvalues, eigenvectors)` of the symmetric part
    `(a + a^T) / 2` of each matrix of the leading axes: eigenvalues
    ascending (`[..., n]`), eigenvectors as the columns of `[..., n, n]`, so
    `a @ v == v * w[..., None, :]`. Each eigenvector has its largest-magnitude
    component (the first on ties) positive. The CPU uses cyclic Jacobi
    rotations in float64, accurate to about `eps * ||a||_F`; CUDA uses
    cuSOLVER `syevd` in float32."""
    a = _square(a, "eigh")
    return EighResult(a._linalg("eigh_values"), a._linalg("eigh_vectors"))


def qr(a, mode="reduced"):
    """The QR factorization `a = Q @ R` of each matrix of the leading axes
    (`[..., m, n]`, any `m` and `n`), by Householder reflections in float64
    on the CPU, as `QRResult(Q, R)`.

    With `k = min(m, n)`, `mode="reduced"` gives `Q` `[..., m, k]` with
    orthonormal columns and an upper-triangular `R` `[..., k, n]`;
    `mode="complete"` gives an orthogonal `Q` `[..., m, m]` and `R`
    `[..., m, n]` (zero rows below the first `k`); `mode="r"` returns `R`
    alone. The diagonal of `R` is non-negative, with the matching columns of
    `Q` signed alike, which makes the factorization unique for full column
    rank (LAPACK, and so NumPy and JAX, leave those signs to the
    reflectors). Derivatives exist for `Q` and `R` of the reduced
    factorization when `a` has full column rank, or for a wide `a` when its
    leading `m x m` block has full rank; the extra columns of a complete `Q`
    are not unique and have none."""
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(f'mode must be "reduced", "complete", or "r", got {mode!r}')
    a = _matrix(a, "qr")
    m, n = a.shape[-2], a.shape[-1]
    complete = mode == "complete" and m > n
    r = a._linalg("qr_r")
    if complete:
        padding = zeros(list(a.shape[:-2]) + [m - n, n], dtype=r.dtype)
        r = concat([r, padding], len(a.shape) - 2)
    if mode == "r":
        return r
    return QRResult(a._linalg("qr_q_complete" if complete else "qr_q"), r)


def svd(a, full_matrices=False, compute_uv=True):
    """The singular value decomposition `a = U @ diag(S) @ Vh` of each
    matrix of the leading axes (`[..., m, n]`), as `SVDResult(U, S, Vh)`,
    or the singular values `S` alone when `compute_uv` is false.

    With `k = min(m, n)`, `S` is `[..., k]` in descending order, `U`
    `[..., m, k]`, and `Vh` `[..., k, n]`; `full_matrices=True` extends `U`
    to `[..., m, m]` and `Vh` to `[..., n, n]` with orthonormal complements.
    Unlike NumPy and JAX, `full_matrices` defaults to false, the economical
    form that has derivatives. Each column of `U` has its largest-magnitude
    component (the first on ties) positive, with the matching row of `Vh`
    signed alike. The CPU uses one-sided (Hestenes) Jacobi rotations in
    float64, which determine even the small singular values to high relative
    accuracy; CUDA uses cuSOLVER `gesvdj` in float32.

    The derivative of `S` is `U^T dA V` on the diagonal, defined wherever
    the singular values are distinct. The derivatives of `U` and `Vh`
    follow JAX: they are exact for distinct nonzero singular values; for
    repeated ones the vectors are not unique and the rule masks the infinite
    `1 / (s_j^2 - s_i^2)` terms to zero, giving finite values that are not
    derivatives, and a zero singular value of a non-square `a` makes them
    infinite or NaN. The extra columns of `full_matrices=True` have no
    derivative unless `a` is square."""
    a = _matrix(a, "svd")
    s = a._linalg("svd_s")
    if not compute_uv:
        return s
    m, n = a.shape[-2], a.shape[-1]
    u = a._linalg("svd_u_full" if full_matrices and m > n else "svd_u")
    vh = a._linalg("svd_vh_full" if full_matrices and m < n else "svd_vh")
    return SVDResult(u, s, vh)


def lstsq(a, b, return_residuals=False):
    """The least-squares solution `x` of `a @ x ~= b` for every matrix of the
    leading axes, from `qr` (leading batch axes broadcast as in `solve`).

    For a tall or square `a` (`[..., m, n]`, `m >= n`) of full column rank,
    `x = solve_triangular(R, Q^T b)` minimizes `||a x - b||`; for a wide `a`
    of full row rank, `x = Q solve_triangular(R, b, trans=1)` from `qr(a^T)`
    is the solution of minimum norm. `b` is `[..., m, k]`, or a vector `[m]`
    (rank one), giving `x` of shape `[..., n, k]` or `[n]`. Unlike NumPy's
    SVD-based `lstsq`, a rank-deficient `a` is not supported: `R` is then
    singular and the solution is non-finite or meaningless, without an
    error; use `svd` to handle one.

    With `return_residuals`, returns `LstsqResult(solution, residuals)`, the
    residuals being the squared norms `sum((b - a @ x)**2)` of each column
    (`[..., k]`, or a scalar for a vector `b`), formed from the residual
    itself rather than by cancellation. Derivatives come from those of `qr`
    and the triangular solves."""
    a = _matrix(a, "lstsq")
    b = _array(b)
    vector = len(b.shape) == 1
    if vector:
        b = b.reshape([b.shape[0], 1])
    elif len(b.shape) < 2:
        raise ValueError(f"lstsq requires a right-hand side of rank one or more, got {list(b.shape)}")
    if b.shape[-2] != a.shape[-2]:
        raise ValueError(
            f"lstsq requires matrix shape {list(a.shape)} and right-hand side shape "
            f"{list(b.shape)} to agree on rows"
        )
    batch = _broadcast_batch(tuple(a.shape[:-2]), tuple(b.shape[:-2]))
    a, b = _to_batch(a, batch), _to_batch(b, batch)
    m, n = a.shape[-2], a.shape[-1]
    if m >= n:
        q, r = qr(a)
        x = solve_triangular(r, _matrix_transpose(q) @ b)
    else:
        q, r = qr(_matrix_transpose(a))
        x = q @ solve_triangular(r, b, trans=1)
    solution = _restore_vector(x, vector)
    if not return_residuals:
        return solution
    residual = b - a @ x
    residuals = (residual * residual).sum(axis=-2)
    if vector:
        residuals = residuals.reshape(list(residuals.shape[:-1]))
    return LstsqResult(solution, residuals)


def _norm_axes(axis, ndim):
    """`axis` (an int or a sequence of ints) as distinct non-negative axes."""
    try:
        axes = (operator.index(axis),)
    except TypeError:
        axes = tuple(operator.index(value) for value in axis)
    normalized = []
    for value in axes:
        if not -ndim <= value < ndim:
            raise ValueError(f"axis {value} is out of bounds for rank {ndim}")
        normalized.append(value % ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"repeated axis in {axis!r}")
    return tuple(normalized)


def _vector_norm(x, ord, axis, keepdims):
    if ord is None or ord == 2:
        # The scaled native norm: no overflow or underflow of the squares.
        return x.norm(axis=axis, keepdims=keepdims)
    magnitude = x.abs()
    if ord == math.inf:
        return magnitude.max(axis=axis, keepdims=keepdims)
    if ord == -math.inf:
        return magnitude.min(axis=axis, keepdims=keepdims)
    if ord == 0:
        return x.not_equal(0.0).astype(x.dtype).sum(axis=axis, keepdims=keepdims)
    if ord == 1:
        return magnitude.sum(axis=axis, keepdims=keepdims)
    # (sum |x|^p)^(1/p) scaled by the dominant magnitude `m` (the largest
    # for p > 0, the smallest for p < 0), so that no |x/m|^p overflows. The
    # result is independent of `m`, which therefore enters through
    # `stop_gradient` and leaves the exact gradient.
    p = float(ord)
    extreme = magnitude.max if p > 0 else magnitude.min
    peak = extreme(axis=axis, keepdims=True).stop_gradient()
    # A zero (or NaN) peak divides by one instead.
    scale = where(peak > 0.0, peak, 1.0)
    result = ((magnitude / scale) ** p).sum(axis=axis, keepdims=True) ** (1.0 / p) * scale
    # An infinite scale makes the ratios NaN; the norm is then infinite.
    result = where(peak.isfinite().logical_or(peak.isnan()), result, peak)
    return result if keepdims else result.reshape(
        [extent for index, extent in enumerate(result.shape) if index != axis]
    )


def _matrix_norm(x, ord, axes, keepdims):
    row, column = axes
    if ord is None or ord == "fro":
        return x.norm(axis=axes, keepdims=keepdims)
    kept = [1 if index in axes else extent for index, extent in enumerate(x.shape)]
    if ord in (1, -1, math.inf, -math.inf):
        # Induced 1-norm: largest absolute column sum; inf-norm: row sum.
        summed, reduced = (row, column) if ord in (1, -1) else (column, row)
        sums = x.abs().sum(axis=summed, keepdims=True)
        result = sums.max(axis=reduced, keepdims=True) if ord > 0 else sums.min(
            axis=reduced, keepdims=True
        )
    elif ord in (2, -2, "nuc"):
        rest = [index for index in range(len(x.shape)) if index not in axes]
        values = svd(x.transpose(rest + [row, column]), compute_uv=False)
        if ord == "nuc":
            result = values.sum(axis=-1)
        else:
            result = values[..., 0] if ord == 2 else values[..., -1]
        result = result.reshape(kept)
    else:
        raise ValueError(f"invalid norm order {ord!r} for matrices")
    if keepdims:
        return result
    return result.reshape([extent for index, extent in enumerate(x.shape) if index not in axes])


def norm(x, ord=None, axis=None, keepdims=False):
    """The vector or matrix norm of `x`, as NumPy's `linalg.norm`.

    `axis` selects a vector norm (an int) or a matrix norm (a pair of
    axes); with `axis=None`, `ord=None` is the 2-norm of the flattened `x`
    of any rank, and any other `ord` needs a 1-D (vector) or 2-D (matrix)
    `x`. Vector orders: `None` or 2 (the scaled native `norm`, which does
    not overflow in float32), 1, `inf`, `-inf`, 0 (the count of nonzero
    entries, with a zero derivative), and any other real `p`,
    `sum(|x|**p)**(1/p)`, scaled by the largest magnitude (smallest for
    `p < 0`) so that the powers neither overflow nor underflow. Matrix
    orders: `None` or `"fro"` (scaled), 1 and -1 (max and min absolute
    column sum), `inf` and `-inf` (row sums), and 2, -2, and `"nuc"` (the
    largest, the smallest, and the sum of the singular values, from `svd`,
    so CPU and CUDA only). `keepdims` keeps the reduced axes with extent
    1."""
    x = _array(x)
    ndim = len(x.shape)
    if axis is None:
        if ord is None:
            result = x.norm()
            return result.reshape([1] * ndim) if keepdims else result
        if ndim not in (1, 2):
            raise ValueError(f"improper number of dimensions to norm: ord={ord!r} with rank {ndim}")
        axes = tuple(range(ndim))
    else:
        axes = _norm_axes(axis, ndim)
    if isinstance(ord, str) and len(axes) == 1:
        raise ValueError(f"invalid norm order {ord!r} for vectors")
    if not (ord is None or isinstance(ord, str) or isinstance(ord, numbers.Real)) or (
        isinstance(ord, numbers.Real) and math.isnan(ord)
    ):
        raise ValueError(f"invalid norm order {ord!r}")
    if len(axes) == 1:
        return _vector_norm(x, ord, axes[0], keepdims)
    if len(axes) == 2:
        return _matrix_norm(x, ord, axes, keepdims)
    raise ValueError(f"norm takes one axis (vector) or two (matrix), got {axis!r}")


def matrix_power(a, n):
    """`a` raised to the integer power `n` for each square matrix of the
    leading axes, by repeated squaring (about `2 log2 |n|` products); `n = 0`
    gives the identity and a negative `n` powers `inv(a)`."""
    a = _square(a, "matrix_power")
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError(f"exponent must be an integer, got {type(n).__name__}") from None
    if n == 0:
        identity = eye(a.shape[-1], dtype=a.dtype)
        return _to_batch(identity, tuple(a.shape[:-2]))
    if n < 0:
        a, n = inv(a), -n
    result, power = None, a
    while True:
        if n & 1:
            result = power if result is None else result @ power
        n >>= 1
        if not n:
            return result
        power = power @ power


def pinv(a, rtol=None):
    """The Moore-Penrose pseudo-inverse of each matrix of the leading axes,
    `V diag(1/s) U^T` from `svd`, with every singular value at or below
    `rtol * max(s)` treated as zero. `rtol=None` means
    `max(m, n) * eps(dtype)`, NumPy's `rtol=None` cutoff (NumPy's legacy
    `rcond` default is a fixed 1e-15, JAX uses ten times this). Derivatives
    come from those of `svd`: exact for distinct nonzero singular values,
    infinite or NaN for a rank-deficient non-square `a` (see `svd`). CPU
    and CUDA only."""
    a = _matrix(a, "pinv")
    m, n = a.shape[-2], a.shape[-1]
    if rtol is None:
        rtol = max(m, n) * (2.0**-23 if a.dtype == float32 else 2.0**-52)
    u, s, vh = svd(a)
    kept = s > s[..., :1] * rtol
    inverse = where(kept, 1.0 / where(kept, s, 1.0), 0.0)
    return _matrix_transpose(vh) @ (inverse.reshape(list(inverse.shape) + [1]) * _matrix_transpose(u))
