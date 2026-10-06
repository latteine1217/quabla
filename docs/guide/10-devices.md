# 10. Devices and Compilation

The same traced program runs on three backends. This chapter explains how to
select one, what each backend does with your program, how precision works on
GPUs, and the lower-level compilation APIs.

| Backend | Platform | Build feature | Execution dtype |
| --- | --- | --- | --- |
| CPU | macOS, Linux | default | `float64` reference; each `float32` op is the `float64` result rounded to `float32` |
| CUDA | Linux, NVIDIA GPU | `cuda` | `float32`; native `float64` with `precision="float64"` |
| CUDA + NCCL | Linux, 2+ GPUs on one node | `cuda-nccl` | `float32` |
| MLX | macOS 14+, Apple silicon | `mlx` | `float32` |

## 10.1 Choosing a Device

The device is a property of a compiled function, selected with `jit`:

```python
def loss(params, x):
    return qb.mean(qb.tanh(x @ params["w"] + params["b"]) ** 2)

params = {"w": qb.array([[0.5], [-0.3]], dtype=qb.float32),
          "b": qb.array([0.1], dtype=qb.float32)}
x = qb.array([[1.0, 2.0], [3.0, 4.0]], dtype=qb.float32)

value, grads = qb.jit(qb.value_and_grad(loss), device="mlx")(params, x)
```

| `device` | Meaning |
| --- | --- |
| `None` or `"cpu"` | The CPU interpreter (default) |
| `"cuda"`, `"cuda:N"` | CUDA device ordinal 0 or N |
| `"mlx"` | MLX's GPU stream |

Behavior:

- Inputs are host arrays. A device call uploads them, runs the plan, and
  returns host `Tensor`s with the original pytree structure. (For training
  loops that keep state on the device, use
  [`optim.Trainer`](07-optimization.md#75-trainer).)
- Eager operations outside `jit` always run on the host.
- A target missing from the build raises `UnsupportedOperationError` with
  `.op` and `.device`. An invalid spelling (`"gpu"`) raises `ValueError`.
- An operation that a backend cannot lower raises
  `UnsupportedOperationError` when the function is compiled; Quabla never
  falls back to the host silently.
- `qb.devices()` lists built targets. It does not enumerate GPUs.

Device results agree with the CPU `float32` reference up to rounding. The
test suites check GPU results with absolute and relative tolerances of
`1e-5`.

## 10.2 Precision

Programs whose arrays are `float64` (the default dtype) still run in `float32`
on a GPU unless you opt in, and `jit` warns once per compiled function:

```text
UserWarning: mlx executes float64 programs as float32
```

For GPU work, create parameters and data as `float32` explicitly (or convert
them once with `qb.tree.map(lambda t: t.astype(qb.float32), params)`). This
keeps the CPU reference and the device computing the same thing and avoids
the warning.

On CUDA, `precision="float64"` runs a `float64` program natively in double
precision, including NVRTC kernels, cuBLAS `dgemm`, and cuSOLVER:

```python
step = qb.jit(qb.value_and_grad(loss), device="cuda", precision="float64")
```

- Results match the CPU `float64` reference to about `1e-13` relative.
- Consumer GPUs run `f64` at a small fraction of their `f32` rate (a
  matmul-heavy step took about 9x longer on a GTX 1660 SUPER), so `float32`
  stays the default.
- One compiled program has one floating type: a program that also contains
  `float32` values raises `UnsupportedOperationError` with
  `.op == "float32"`.
- The option is a no-op on the CPU, raises `UnsupportedOperationError` on MLX
  (which has no `float64` arithmetic), and is part of the trace-cache key.
- `Trainer(..., precision="float64")` accepts the same option. The NCCL
  data-parallel path stays `float32`.

`bool_` values are held as `float32` 0/1 on both GPUs.

## 10.3 CUDA Runtime Requirements

The CUDA backend loads its libraries at run time:

| Library | Required for |
| --- | --- |
| NVIDIA driver | Everything |
| `libnvrtc` | Everything: kernels are generated and compiled with NVRTC |
| `libcublas` | Optional: rank-2 GEMM (otherwise an NVRTC tiled kernel) |
| `libcusolver` | `solve` and the decompositions |
| `libnccl` | Only the `cuda-nccl` data-parallel path |

Put their directory on `LD_LIBRARY_PATH`. The loaded `libnvrtc` must not be
newer than the CUDA version the driver supports; otherwise loading
generated kernels fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. A missing
library produces an error that names it.

## 10.4 CUDA

The CUDA backend generates CUDA C source for the frozen plan, compiles it
with NVRTC once per compiled function, and keeps the module loaded, so later
calls do not recompile.

- **Fusion.** A purely elementwise graph becomes one fused kernel. General
  plans fuse maximal elementwise chains that follow materialized values such
  as `matmul` outputs.
- **Memory.** Intermediate device buffers are released after their last
  consumer and reused by later nodes.
- **Reductions.** Axis sums and means of length 256 or more use a block
  reduction.
- **`cond`** reads its predicate back to the host once, then runs the
  selected branch on the device.

Loops are where CUDA performance needs the most attention:

| Loop body | Lowering | Cost per iteration |
| --- | --- | --- |
| Purely elementwise (no `matmul`, reduction, slicing across lanes, `cond`, nested loop, or array constant) | One fused kernel for the whole loop, including first-order VJP and, for a subset, the HVP | Negligible |
| Anything else | Host-driven loop: the body is compiled once as a device program, and the host launches it per iteration on device-resident carries | About 30-50 µs on a GTX 1660 SUPER, after the body is recorded as a CUDA graph |
| `while_loop` | Host-driven, plus one predicate readback | About 100 µs |

After two eager iterations, a host-driven body that launches only NVRTC
kernels, device copies, and cuBLAS products is recorded as a CUDA graph and
replayed; replay runs the same kernels in the same order, so results are
bit-identical. Bodies containing `cond`, nested loops, or cuSOLVER calls
launch eagerly. Reverse passes keep the carry tape on the device (in full
for short loops, as square-root checkpoint blocks otherwise).

Practical consequences:

- Long loops with tiny non-elementwise bodies are launch-bound. Prefer
  vectorizing across the loop (with `vmap` or array operations) or
  `unroll=True` for short loops.
- Adaptive `odeint` and the Krylov solvers run host-driven.
- `examples/benchmark_host_loop_cuda.py` measures these costs on your GPU.

## 10.5 MLX

The MLX backend lowers the plan to MLX operations on the GPU stream; arrays
stay on the device between operations, and loop carries stay on the device
between host-dispatched iterations.

- No `float64` arithmetic. `float64` programs run as `float32` with a
  warning; `precision="float64"` is rejected.
- LU, `eigh`, QR, and SVD run with LAPACK on MLX's CPU stream (MLX 0.32 has
  no GPU kernels for them); unified memory means no copies.
- `vmap` of Hessian-vector products has no MLX lowering.
- Float32 subnormal inputs count as zero (Metal flushes them).
- All MLX graph construction, evaluation, and readback in a process are
  serialized by one lock, so calling MLX-compiled functions from several
  Python threads is safe but does not run them concurrently.

MLX builds need the Metal library `mlx.metallib` at run time; see
[Installation From Source](https://github.com/latteine1217/quabla/blob/main/README.md#installation-from-source).

## 10.6 Ahead-of-Time Lowering

`jit(fun).lower(*args)` traces without executing. Arguments may be real
arrays or `qb.ShapeDtype(shape, dtype)` placeholders anywhere in the pytree:

```python
lowered = qb.jit(loss).lower(
    {"w": qb.ShapeDtype((2, 1), qb.float32), "b": qb.ShapeDtype((1,), qb.float32)},
    qb.ShapeDtype((2, 2), qb.float32),
)
print(lowered.as_text())
compiled = lowered.compile()
compiled(params, x)            # Tensor(0.07218059, dtype=float32)
```

```text
%0 = input[name=params/b] : tensor<1xf32>
%1 = input[name=params/w] : tensor<2x1xf32>
%2 = input[name=x] : tensor<2x2xf32>
%3 = matmul(%2, %1) : tensor<2x1xf32>
%4 = add(%3, %0) : tensor<2x1xf32>
%5 = tanh(%4) : tensor<2x1xf32>
%6 = powi(%5, 2) : tensor<2x1xf32>
%7 = mean(%6) : tensor<f32>
outputs: [[]]
```

- `as_text()` prints the frozen IR after dead-code elimination, constant
  folding, and common-subexpression elimination. Inputs are named by their
  pytree path. This is the quickest way to see what your function compiles
  to, or to check that a derivative graph has the size you expect.
- `compile()` returns a positional callable that rejects a changed
  structure, shape, dtype, or static value (`ValueError`).
- `.program` is an ordered-output `Program` (see below);
  `.program.compile(target)(inputs_dict)` returns the list of outputs.
- Transform the Python function before lowering
  (`qb.jit(qb.grad(f)).lower(...)`).

## 10.7 The Compiler Facade

The facade is the explicit, name-keyed compilation API. It is useful when
integrating Quabla into another tool, or when you want one compiled
artifact per target without the pytree machinery:

```python
compiler = qb.Compiler()
print(compiler.capabilities())           # {'cpu': True, 'cuda': False, 'mlx': True}

specs = [("x", [2], qb.float64), ("w", [2], qb.float64)]
program = compiler.trace(lambda x, w: qb.sum(qb.sin(x * w)), specs)
print(program.lower_text())

inputs = {"x": qb.array([1.0, 2.0]), "w": qb.array([0.5, 0.25])}
print(program.compile("cpu")(inputs))

grads = program.vjp("y_bar")             # {input name: Program}; y_bar is the cotangent input
print(grads["w"].compile("mlx")({**inputs, "y_bar": qb.array(1.0)}))
```

```text
%0 = input[name=x] : tensor<2xf64>
%1 = input[name=w] : tensor<2xf64>
%2 = mul(%0, %1) : tensor<2xf64>
%3 = sin(%2) : tensor<2xf64>
%4 = sum(%3) : tensor<f64>
Tensor(0.95885108, dtype=float64)
Tensor([0.87758255, 1.7551651 ], dtype=float64)
```

| Stage | API |
| --- | --- |
| Trace | `Compiler.trace(fn, specs) -> Program`; a spec is `(name, shape)` (float64) or `(name, shape, dtype)` |
| Transform | `Program.vjp(cotangent_name) -> {input: Program}`; `Program.jvp(input_name) -> Program`, the derivative along an all-ones tangent of that input |
| Compile | `Program.compile(target, device_ordinal=0) -> Executable` |
| Run | `Executable(inputs)` or `Executable.evaluate(inputs)` |
| Inspect | `Compiler.capabilities()`, `Compiler.capability(target)`, `Program.lower_text()`, `Program.output_shape` |

`capabilities()` reports what the build contains; it is not a hardware probe
and does not promise that every graph lowers on every target.

## 10.8 Data Parallelism (Experimental)

With the `cuda-nccl` build feature, `qb.distributed.value_and_grad` shards a
batch across GPUs of one node and all-reduces parameter gradients with NCCL:

```python
step = qb.distributed.value_and_grad(
    loss,
    devices=["cuda:0", "cuda:1"],
    shard_argnums=(1,),          # positional arguments split on axis 0
    argnums=0,                   # replicated parameters to differentiate
    reduction="mean",            # combine shard losses: "mean" or "sum"
)
value, grads = step(params, batch)
params, state = opt.update(params, grads, state)   # optimizer on the host
```

- Sharded arguments are pytrees of full batches whose leading axis divides
  evenly by the number of devices.
- `reduction` combines the per-shard losses (and gradients) independently of
  the loss's own reduction.
- Gradients with respect to sharded inputs, auxiliary outputs, and calls
  inside other transforms are rejected. The optimizer update runs on the
  host.
- Each callable caches up to eight signatures and creates its NCCL
  communicators once.

On two RTX 3090s, 200 SGD steps of a small MLP stay within `2.1e-8` normwise
parameter error of the single-GPU run
(`examples/validate_data_parallel_training_cuda.py`). There is no multi-node
transport, tensor parallelism, or sharded matmul.

## 10.9 Checklist for Device Training

1. Make parameters and data `float32` (or use `precision="float64"` on
   CUDA when you need double precision).
2. Validate on the CPU first: run the same function with `device="cpu"` and
   compare, since the CPU is the reference.
3. Use `optim.Trainer` (or `jit(value_and_grad)` plus a host optimizer for
   small models) and keep shapes fixed across steps.
4. On CUDA, check loop bodies: keep them elementwise where possible.
5. Read values back (`trainer.loss()`, `.item()`) only when you log, not on
   every step.
