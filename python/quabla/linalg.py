"""Dense linear algebra over the last two axes, batched over leading axes.

Every function takes eager or traced arrays, so it runs eagerly, under
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

No derivative forms an explicit inverse: the gradient of `logabsdet` is
`solve(a^T, g I)` and its tangent `trace(solve(a, da))`. The derivatives of
`slogdet`, `det`, and `inv` at an exactly singular matrix are undefined:
`slogdet` itself returns `(0, -inf)` there, but the derivative's `solve`
raises on the CPU (and reports the singular factor on CUDA), where JAX
returns non-finite values. The eigenvector derivative divides by eigenvalue
gaps, so it is infinite or NaN for a repeated eigenvalue, as in JAX; the
eigenvalue derivative stays defined.

Devices: CPU and CUDA (cuSOLVER `getrf`/`getrs` and `syevd`, `float32`).
MLX rejects `solve`, `slogdet`, and `eigh`, because MLX's LU and eigh
factorizations only run on its CPU stream.
"""

import collections

from ._array import asarray, eye
from ._quabla import Tensor, TraceTensor

__all__ = [
    "EighResult",
    "SlogdetResult",
    "cho_solve",
    "cholesky",
    "det",
    "eigh",
    "inv",
    "slogdet",
    "solve",
    "solve_triangular",
]

SlogdetResult = collections.namedtuple("SlogdetResult", ["sign", "logabsdet"])
SlogdetResult.__doc__ = "`(sign, logabsdet)` of `slogdet`; a pytree like any namedtuple."
EighResult = collections.namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
EighResult.__doc__ = "`(eigenvalues, eigenvectors)` of `eigh`; a pytree like any namedtuple."


def _array(x):
    return x if isinstance(x, (Tensor, TraceTensor)) else asarray(x)


def _square(a, name):
    a = _array(a)
    shape = tuple(a.shape)
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(f"{name} requires a stack of square matrices [..., n, n], got {list(shape)}")
    return a


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
