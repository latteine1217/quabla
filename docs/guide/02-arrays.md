# 2. Arrays and Operations

This chapter covers the array type and the operation library. Everything
here works the same way on eager `Tensor`s and, inside a transform, on traced
`TraceTensor`s, so the functions you write for eager experimentation are the
functions you later differentiate and compile.

## 2.1 Creating Arrays

```python
import quabla as qb

x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])      # nested lists
y = qb.array([1, 2, 3], dtype=qb.float32)              # explicit dtype
print(x)
print(x.shape, x.ndim, x.dtype)
print(y)
```

```text
Tensor([[1., 2., 3.],
        [4., 5., 6.]], dtype=float64)
[2, 3] 2 f64
Tensor([1., 2., 3.], dtype=float32)
```

| Function | Result |
| --- | --- |
| `qb.array(obj, dtype=None)` | A new array from a scalar, nested list/tuple, NumPy array, buffer, or Quabla array. Always copies. |
| `qb.asarray(obj, dtype=None)` | Like `array`, but returns `obj` itself if it already is a Quabla array of that dtype. |
| `qb.zeros(shape)`, `qb.ones(shape)`, `qb.full(shape, value)` | Constant arrays. |
| `qb.zeros_like(x)`, `qb.ones_like(x)`, `qb.full_like(x, value)` | Same shape and dtype as `x`. |
| `qb.arange(start, stop=None, step=1.0)` | Evenly spaced values in `[start, stop)`. |
| `qb.linspace(start, stop, num)` | `num` values from `start` to `stop` inclusive. |
| `qb.eye(n, m=None)` | Identity matrix. |

Every creation function accepts `dtype=`. `shape` is a tuple or list of
non-negative integers; `()` is a scalar (rank 0). Random arrays come from
`qb.random` ([Chapter 6](06-pytrees-random-io.md#62-random-numbers)).

Arrays are **immutable**. There is no in-place assignment (`x[0] = 1` is not
supported); every operation returns a new array. Use `qb.where`,
`scatter_add`, or `concat` to build modified copies.

## 2.2 Dtypes and Promotion

There are exactly three element types:

| dtype | `str(dtype)` | Storage | Notes |
| --- | --- | --- | --- |
| `qb.float64` | `f64` | 8 bytes | The default for Python numbers and lists |
| `qb.float32` | `f32` | 4 bytes | The native GPU type |
| `qb.bool_` | `bool` | 1 byte | Results of comparisons; masks |

There are no integer, `float16`, or `bfloat16` arrays. Inference rules:

- Python floats, ints, and lists of them become `float64`; a list whose
  elements are all Python `bool`s becomes `bool_`.
- NumPy `float64`, `float32`, and `bool_` keep their dtype; NumPy integers
  become `float64`; NumPy `float16` raises `TypeError`.

Promotion is deliberately strict:

```python
x = qb.array([1.0, 2.0, 3.0])                     # float64
r = qb.array([1.0, 2.0, 3.0], dtype=qb.float32)   # float32

x + r            # ValueError: ... mismatched dtypes f64 and f32;
                 #   convert one of them explicitly with astype
x.astype(qb.float32) + r                          # fine, float32
(r * 2.0).dtype                                   # f32: Python scalars are weak
```

- Two arrays of different float dtypes never promote silently. Convert one
  with `astype`. This rule exists because a stray `float64` constant in a
  `float32` model would otherwise double memory traffic and change GPU
  numerics without warning.
- Python scalars are *weak*: they adopt the dtype of the array they meet, so
  `0.5 * r` stays `float32`.
- Under a transform, a constant array captured from a closure keeps its
  dtype ("strong"), so the same rule applies to closures.

## 2.3 Booleans and Masks

Comparisons return `bool_` arrays. Combine masks with `&`, `|`, `~` (or
`logical_and`, `logical_or`, `logical_not`) and reduce them with `any` and
`all`:

```python
x = qb.array([1.0, 2.0, 3.0])
m = (x > 1.0) & (x < 3.0)
print(m, ~m)
print(qb.any(m), qb.all(m))
print(qb.where(m, x, 0.0))     # select per element; scalars broadcast
print(x * m)                   # a bool operand becomes 0/1 of the float dtype
```

```text
Tensor([False,  True, False], dtype=bool) Tensor([ True, False,  True], dtype=bool)
Tensor( True, dtype=bool) Tensor(False, dtype=bool)
Tensor([0., 2., 0.], dtype=float64)
Tensor([0., 2., 0.], dtype=float64)
```

Rules worth remembering:

- `==` and `!=` are **not** overloaded: tensors compare by identity and stay
  hashable. Use `qb.equal(a, b)` and `qb.not_equal(a, b)`.
- Comparisons follow IEEE semantics: anything compared with NaN is false,
  except `not_equal`.
- Arithmetic on two bool arrays, and unary math on a bool array, raise an
  error that names `astype`. A bool array mixed with a float array acts as
  0/1 of the float dtype.
- `where` does not differentiate through its predicate, and the unselected
  branch contributes no gradient. Prefer `qb.where(x.isfinite(), x, 0.0)` to
  `mask * x` for guarding NaN and infinity: `0 * NaN` is NaN, while `where`
  drops the NaN entirely.
- A single-element eager bool array works in `if`; larger ones raise like
  NumPy. A traced array in `if` always raises `TracerError`.

## 2.4 Indexing, Slicing, and Views

Indexing follows NumPy *basic* indexing: integers (negative counts from the
end), slices with any non-zero step, `None` for a new axis, and one `...`.

```python
x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x[1, -1]          # Tensor(6.)
x[:, 1]           # second column
x[::-1, 0]        # Tensor([4., 1.])
x[..., ::-1]      # reverse the last axis
x[None].shape     # [1, 2, 3]
x.T.shape         # [3, 2]
```

Indices must be static Python integers. There are no integer index arrays,
no boolean-mask indexing (`x[x > 0]`), and no empty slices. For
static-but-irregular index sets use `gather` and `scatter_add`:

```python
x.gather([2, 0], axis=1)                                   # columns 2 and 0
qb.zeros((3,)).scatter_add([0, 2, 0], qb.array([1.0, 2.0, 3.0]))
```

```text
Tensor([[3., 1.],
        [6., 4.]], dtype=float64)
Tensor([4., 0., 2.], dtype=float64)
```

`scatter_add` is functional (it returns a new array) and accumulates repeated
destinations in index order. Each of the two traces to a single IR node, and
the derivative of one is the other, so their cost does not grow with the
number of indices.

`x.slice(axis, start, length, step=1)` returns a zero-copy, read-only
`TensorView` that shares storage with `x`. Views behave like arrays in every
operation; arithmetic on them allocates a new contiguous `Tensor`.

## 2.5 Shapes and Broadcasting

Binary operations broadcast with NumPy's trailing-axis rules:

```python
x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # [2, 3]
x + qb.array([10.0, 20.0, 30.0])                    # [3] broadcasts over rows
x + qb.array([[100.0], [200.0]])                    # [2, 1] broadcasts over columns
```

Shape manipulation functions, all differentiable and usable under `vmap`:

| Function | Notes |
| --- | --- |
| `x.reshape(3, -1)`, `qb.reshape(x, shape)` | One `-1` extent is inferred. |
| `x.T`, `qb.transpose(x, axes=None)` | Reverses axes by default. |
| `qb.moveaxis`, `qb.swapaxes`, `qb.ravel` | NumPy semantics. |
| `qb.expand_dims(x, axis)`, `qb.squeeze(x, axis=None)` | Add or remove unit axes. |
| `qb.broadcast_to(x, shape)` | Materializes the broadcast. |
| `qb.concat(xs, axis)`, `qb.stack(xs, axis=0)`, `qb.split(x, n, axis=0)` | `concat` requires `axis`; `split` takes a section count or a list of indices. |
| `qb.tile`, `qb.repeat`, `qb.flip`, `qb.roll`, `qb.pad` | `pad` modes: `constant`, `edge`, `reflect`, `symmetric`, `wrap`. |
| `qb.diag`, `qb.diagonal`, `qb.trace`, `qb.tril`, `qb.triu` | Matrix structure helpers. |
| `qb.meshgrid(*xs, indexing="xy")` | Returns a list; `indexing="ij"` keeps input order. |

Every shift, pad width, repeat count, and axis is a static Python `int`.

## 2.6 Reductions

`sum`, `mean`, `prod`, `max`, `min`, `any`, `all`, and `norm` take
`axis=None` (all axes), an int, or a sequence of ints, plus `keepdims`:

```python
x = qb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
qb.sum(x, axis=0)                   # Tensor([5., 7., 9.])
x.mean()                            # Tensor(3.5)
qb.max(x, axis=1, keepdims=True)    # [[3.], [6.]]
x.cumsum(axis=1)                    # inclusive prefix sum; reverse=True scans backward
```

Numerically careful variants are built in:

- `x.norm()` and `qb.linalg.norm` scale by the largest magnitude before
  squaring, so `float32` norms neither overflow nor underflow.
- `qb.var` and `qb.std` take two passes (mean, then mean squared deviation),
  never `E[x^2] - E[x]^2`; `ddof` is supported.
- `qb.logsumexp`, `qb.softmax`, and `qb.log_softmax` subtract the maximum
  first: `qb.softmax(qb.array([1000.0, 0.0]))` is `[1., 0.]`.
- `qb.max`/`qb.min` propagate NaN and split the derivative equally between
  tied extrema, as JAX does.

`qb.sum`, `qb.max`, `qb.min`, `qb.any`, `qb.all`, `qb.abs`, and `qb.round` are
attributes of the module but not in `__all__`, so `from quabla import *` does
not shadow the Python builtins.

## 2.7 The Math Library

Module-level functions dispatch to the method of the same name, so
`qb.sin(x)` is `x.sin()`. Non-array operands go through `asarray`.

| Group | Functions |
| --- | --- |
| Arithmetic | `+ - * / **`, `-x`, `@`, `qb.power`, `qb.maximum`, `qb.minimum`, `qb.fmod`, `qb.mod` (`remainder`), `qb.square`, `qb.reciprocal`, `qb.sign`, `qb.clip`, `qb.abs` |
| Exponential and logarithmic | `exp`, `expm1`, `exp2`, `log`, `log1p`, `log2`, `log10`, `logaddexp`, `sqrt`, `cbrt`, `hypot` |
| Trigonometric and hyperbolic | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `atan2`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` |
| Special | `erf`, `erfc` |
| Rounding | `floor`, `ceil`, `round` (half to even); derivative zero |
| Activations | `relu`, `sigmoid`, `softplus`, `silu`, `gelu(approximate=True)`, `softmax`, `log_softmax` |
| Classification | `isfinite`, `isnan`, `isinf`, `nan_to_num` |
| Products | `matmul`, `dot`, `outer`, `tensordot`, `kron`, `cross`, `einsum` (matrix-product equations only) |
| Calculus | `diff`, `trapezoid`, `polyval`, `interp` |
| Statistics | `var`, `std`, `logsumexp` |
| Linear algebra | `solve`, `cholesky`, and the `qb.linalg` module ([Chapter 8](08-linear-algebra.md)) |

Numerical conventions are designed for differentiable scientific code:

- Functions never raise on domain errors; they return NaN or infinity under
  IEEE rules (`arcsin(2)` is NaN, `log10(0)` is `-inf`).
- Derivative rules are themselves IR graphs, so every function has
  derivatives of every order on every backend.
- At points where a textbook derivative is undefined but the function is
  well-behaved in practice, Quabla picks a finite convention and documents
  it. For example, every derivative order of `sqrt` at zero is zero, and
  `x ** y` has `d/dx = 0` where `x == 0` and `y < 1`.
- `sigmoid`, `softplus`, `logaddexp`, `hypot`, `gelu`, and the norms are
  formulated so that neither branch overflows.

`qb.matmul` follows NumPy for rank-1 operands (vector @ vector is a scalar);
the `@` operator requires rank 2 or more. Batched `matmul` broadcasts
leading axes.

The exact formula and accuracy statement of each function are listed in the
[API Reference](../api.md#arrays-and-numpy).

## 2.8 NumPy Interoperability

```python
import numpy as np

t = qb.asarray(np.arange(6.0).reshape(2, 3))   # copies; dtype float64
a = np.asarray(t)                              # via __array__
print(a.dtype, a.flags.writeable)
print(t.numpy().dtype, t.tolist(), qb.array(3.0).item())
m = memoryview(t.astype(qb.float32))           # buffer protocol
print(m.format, m.shape, m.readonly)
```

```text
float64 False
float64 [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] 3.0
f (2, 3) True
```

- Import always copies, so a Quabla array never aliases a NumPy array that
  could later be mutated.
- Export through `__array__` and the buffer protocol is zero-copy and
  read-only when the dtype matches.
- `item()`, `float(t)`, `tolist()`, and `numpy()` read values; they raise
  `TracerError` on a traced value.
- A NumPy array that meets a traced value is *not* captured automatically:
  wrap it with `qb.asarray` first.

## 2.9 Printing

`repr` follows NumPy's default print options and always shows the dtype.
Arrays with more than 1000 elements are summarized to the first and last
three entries of each long axis, and only those entries are read. A
`TraceTensor` prints its node id, shape, and dtype, since it has no values:

```text
TraceTensor(node_id=3, shape=[2], dtype=float32)
```
