# 1. Getting Started

Quabla is a differentiable array library aimed at scientific machine
learning, physics-informed neural networks (PINNs) in particular. You write
ordinary Python functions over arrays; Quabla traces them into a typed tensor
IR, transforms that IR (differentiation, vectorization), compiles it into a
frozen execution plan, and runs the plan on the CPU, on an NVIDIA GPU through
CUDA, or on Apple silicon through MLX.

The Python API follows JAX closely. If you know `jax.grad`, `jax.vmap`, and
`jax.jit`, most of this guide will look familiar; the differences are noted
where they matter. Quabla does not depend on JAX.

## 1.1 Installation

### From PyPI

Two distributions provide the same `quabla` package. Install exactly one of
them per environment:

```sh
pip install quabla       # Linux x86_64: CPU + CUDA; macOS arm64: CPU
pip install quabla-mlx   # macOS 14+ on Apple silicon: CPU + MLX
```

Wheels exist for CPython 3.10 to 3.14. On other platforms pip builds the
source distribution, which needs a Rust toolchain and enables only the CPU
backend.

The Linux wheel does not bundle CUDA. The CPU backend works everywhere;
`device="cuda"` additionally needs an NVIDIA driver and `libnvrtc` on the
library path (see [Devices](10-devices.md#103-cuda-runtime-requirements)).

### From source

A source build is needed for development and for some feature combinations
(for example CUDA with NCCL on your own machine). You need a stable Rust
toolchain, Python 3.10+, and maturin 1.9.3 or later:

```sh
git clone https://github.com/latteine1217/quabla.git
cd quabla
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "maturin>=1.9.3,<2"

maturin develop --release                        # CPU only
maturin develop --release --features mlx         # macOS, Apple silicon
maturin develop --release --features cuda        # Linux, CUDA
maturin develop --release --features cuda-nccl   # Linux, CUDA + NCCL
```

The MLX build additionally needs CMake and Xcode's Metal toolchain. The
complete prerequisites per backend are in
[Installation From Source](https://github.com/latteine1217/quabla/blob/main/README.md#installation-from-source).

### Verifying the installation

```python
import quabla as qb

print(qb.__version__)
print(qb.devices())                    # targets compiled into this build
print(qb.Compiler().capabilities())    # the same, as a dict
```

```text
0.5.0
['cpu', 'mlx']
{'cpu': True, 'cuda': False, 'mlx': True}
```

`devices()` reports which backends the extension was *built* with. It does
not probe hardware: a CUDA build on a machine without a GPU still lists
`"cuda"`, and running a program there raises an error that names the missing
driver or library.

NumPy is optional. Importing `quabla` never imports NumPy, but if NumPy is
installed, arrays convert to and from it (see
[Arrays](02-arrays.md#28-numpy-interoperability)).

## 1.2 A First Gradient

```python
import quabla as qb

def f(x):
    return qb.sum(qb.sin(x) ** 2)

x = qb.array([0.0, 0.5, 1.0])
print(f(x))            # an ordinary eager evaluation
df = qb.grad(f)        # a new function: x -> df/dx
print(df(x))           # 2 sin(x) cos(x) = sin(2x)
print(qb.jit(df)(x))   # the same, compiled once and cached
```

```text
Tensor(0.93792227, dtype=float64)
Tensor([0.        , 0.84147098, 0.90929743], dtype=float64)
Tensor([0.        , 0.84147098, 0.90929743], dtype=float64)
```

Three things happened:

1. `f(x)` ran eagerly. Every operation on a `Tensor` executes immediately on
   the host and returns a new `Tensor`.
2. `qb.grad(f)` returned a function. Calling it *traced* `f`: Quabla called
   `f` once with a placeholder value, recorded the operations into a graph,
   built the reverse-mode derivative of that graph symbolically, compiled the
   result into a CPU plan, and executed it.
3. `qb.jit(df)` did the same for the composed function and cached the
   compiled plan. Later calls with the same argument shapes and dtypes reuse
   it without calling Python again.

## 1.3 The Mental Model

### Eager arrays and traced values

Quabla has two kinds of array value:

| Type | Has data? | Where it appears |
| --- | --- | --- |
| `Tensor` (and its zero-copy view `TensorView`) | Yes | Results of `qb.array`, eager operations, and every transform's output |
| `TraceTensor` | No, only shape and dtype | Arguments that a transform passes to your function while tracing it |

The same Python code runs on both. `qb.sin(x)` calls `x.sin()`, which
computes values for a `Tensor` and records a graph node for a `TraceTensor`:

```python
def show(x):
    print("tracing with", x)
    return qb.sin(x)

g = qb.jit(show)
g(qb.array([1.0, 2.0]))         # first call: traces
g(qb.array([3.0, 4.0]))         # same shape and dtype: cached, no print
g(qb.array([1.0, 2.0, 3.0]))    # new shape: traces again
```

```text
tracing with TraceTensor(node_id=0, shape=[2], dtype=float64)
tracing with TraceTensor(node_id=0, shape=[3], dtype=float64)
```

This is the key rule of tracing: **your Python function runs once per
argument signature, not once per call**. A Python `print`, a counter, or a
random draw inside a jitted function happens at trace time only. Because a
`TraceTensor` has no value, Python control flow on it (`if x > 0:`),
`float(x)`, and `x.item()` raise `quabla.TracerError`; use `qb.where` or the
structured control flow of [Chapter 4](04-control-flow.md) instead.

### Transforms are function-to-function

Every transform takes a function and returns a function: `grad(f)`,
`vmap(f)`, `jit(f)`. They compose freely, in any order:

```python
qb.jit(qb.vmap(qb.grad(qb.grad(u)), in_axes=(0, None)))
```

reads inside out: the second derivative of `u`, vectorized over a batch of
points, compiled.

### Shapes and dtypes are static

A compiled program is specialized to its argument shapes, dtypes, pytree
structure, and static arguments. There are no symbolic dimensions. A new
shape is a new trace (Quabla keeps up to 8 per function by default; see
[Transforms](03-transforms.md#38-jit)).

There are three dtypes: `float64` (the default), `float32`, and `bool_`.
There is no integer dtype. Mixing `float32` and `float64` arrays is an error
that asks for an explicit `astype`; Python scalars adapt to the array they
meet. The [Arrays](02-arrays.md#22-dtypes-and-promotion) chapter has the
details.

### One program, several backends

The CPU backend is the `float64` reference implementation: it interprets the
frozen plan, rounding every `float32` operation to `f32`. CUDA and MLX lower
the same plan to GPU kernels and run in `f32` (CUDA can opt into native
`f64`). You choose a backend per compiled function:

```python
step = qb.jit(qb.value_and_grad(loss), device="mlx")    # or "cuda", "cuda:1", "cpu"
```

An operation that a backend cannot lower raises
`UnsupportedOperationError`; Quabla never silently falls back to the host.

## 1.4 The Three API Layers

| Layer | Entry points | Use it for |
| --- | --- | --- |
| Core API (v0.2+) | `qb.array`, `qb.grad`, `qb.vmap`, `qb.jit`, `qb.optim`, `qb.linalg`, `qb.ode`, ... | All new code. The rest of this guide uses it. |
| Compiler facade | `qb.Compiler`, `Program`, `Executable`, `qb.jit(f).lower(...)` | Explicit compilation from named input specs, inspection of the IR, integration with other tools. See [Devices](10-devices.md#107-the-compiler-facade). |
| v0.1 API | `trace_tensor`, `tensor_*_fn`, device optimizers, `Matrix` | Existing code. Kept working through 0.x; migrated names warn once. See [Migration](15-migration.md). |

## 1.5 Where to Go Next

- Continue with [Arrays and Operations](02-arrays.md) for the array model.
- Jump to [Function Transforms](03-transforms.md) if you already know JAX.
- Jump to [the PINN tutorial](11-pinn-tutorial.md) to see a complete
  training script.
