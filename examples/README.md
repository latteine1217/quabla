# Examples And Benchmarks

Runnable examples and benchmarks for each backend. Every command below is run
from the repository root, not from this directory.

The commands below assume the virtual environment from
[Installation From Source](../README.md#installation-from-source) is active and the
extension was built with the named feature.

CPU build: a one-dimensional Poisson PINN and the frozen CPU plan benchmark.

```sh
python examples/pinn_poisson.py
python examples/benchmark_tensor_cpu.py
```

`examples/plot_readme_figure.py` trains a small tanh-MLP Poisson PINN on the
CPU backend and regenerates the README figure `docs/assets/pinn_poisson.png`.
It needs matplotlib (`python -m pip install matplotlib`), which is not a
Quabla dependency.

MLX build (`--features mlx`): one-graph Poisson PINNs keep parameters and Adam
moments on the MLX GPU stream; the two-layer MLP example checks PDE and
boundary residuals plus trained parameters against CPU:

```sh
python examples/pinn_poisson_mlx.py
python examples/pinn_mlp_mlx.py
python examples/benchmark_pinn_mlx.py
python examples/benchmark_mlp_value_and_grad_mlx.py
python examples/benchmark_forward_mlx.py
python examples/benchmark_loop_value_and_grad_mlx.py
```

Use `examples/benchmark_forward_mlx.py --device-only` to exclude only output
readback; dynamic host input upload remains part of that timing. Add
`--retain-static-inputs` to upload model weights once and time only dynamic
input upload on later calls. `examples/benchmark_pinn_mlx.py --hidden-layers N
--hidden-width W` replaces the one-parameter model with a tanh MLP.

The MLX benchmarks report reproducibility metadata for both the one-parameter
Poisson optimizer path and an end-to-end batched MLP value-and-gradient path.
The former excludes diagnostic readbacks; the latter includes input uploads and
loss/gradient readbacks, so neither is a CUDA/JAX comparison.

CUDA build (`--features cuda`): a two-point batched Poisson PINN whose Adam
optimizer state and collocation tensors stay on the GPU, and a two-layer,
four-parameter Poisson PINN with combined residual and boundary losses:

```sh
python examples/pinn_poisson_cuda.py
python examples/pinn_mlp_cuda.py
```

Measure end-to-end broadcast-batched matmul or fused elementwise-chain
throughput (including the current host/device transfers), and device-resident
two-layer Poisson PINN Adam steps with separate compile, warm-up, and
synchronized training timings:

```sh
python examples/benchmark_tensor_cuda.py
python examples/benchmark_tensor_cuda.py --elementwise --rows 1024 --inner 1024
python examples/benchmark_tensor_cuda.py --rank-two --device-resident
python examples/benchmark_pinn_cuda.py
```

CUDA + NCCL build (`--features cuda-nccl`, two GPUs):
`examples/validate_data_parallel_cuda.py` checks single-call two-GPU parity
against the CPU reference, and
`examples/validate_data_parallel_training_cuda.py` checks 200-step training
parity against one GPU and CPU. Both use CUDA ordinals 0 and 1 (select the
devices with `CUDA_VISIBLE_DEVICES`) and need `libnccl.so` to be loadable.
On any Linux machine with two GPUs and NCCL:

```sh
maturin develop --release --features cuda-nccl
python examples/validate_data_parallel_cuda.py
python examples/validate_data_parallel_training_cuda.py
# Rust two-GPU tests: schedule parity, communicator recovery, lifecycle.
QUABLA_CUDA_NCCL_TEST=1 cargo test -p quabla-core --features cuda-nccl \
  --test tensor_ir on_two_gpus -- --nocapture
```

`--grid`, `--hidden`, `--steps`, and `--modes` scale or restrict the training
check, and `--output` also writes its JSON report to a file (`--help` lists
all flags); `--modes single_gpu,cpu` is a dry run for a host with one GPU and
no NCCL.

Rust-only examples of the core crate (no Python extension needed):

```sh
cargo run -p quabla-core --example lotka_volterra
cargo run -p quabla-core --example fit_lotka_volterra
```

The first prints the Lotka-Volterra loss and its four-parameter gradient; the
second reports initial loss, final loss, optimization steps, and the fitted
parameter vector.
