# Examples And Benchmarks

Runnable examples and benchmarks for each backend. Every command below is run
from the repository root, not from this directory.

The commands below assume the virtual environment from
[Installation From Source](../README.md#installation-from-source) is active and the
extension was built with the named feature.

PINN examples: a one-parameter Poisson PINN, and a two-layer tanh-MLP
Poisson PINN that trains on a device with `optim.Trainer` and checks the
trained parameters against the same training on the CPU. Both take
`--device cpu | mlx | cuda[:N]` (the default is `cpu`):

```sh
python examples/pinn_poisson.py
python examples/pinn_mlp.py --device mlx    # --features mlx
python examples/pinn_mlp.py --device cuda   # --features cuda
```

`examples/plot_readme_figure.py` trains a small tanh-MLP Poisson PINN on the
CPU and regenerates the README figure `docs/assets/pinn_poisson.png`.
It needs matplotlib (`python -m pip install matplotlib`), which is not a
Quabla dependency.

Measure the per-iteration cost of host-driven CUDA loops (bodies that are not
elementwise: `fori_loop`, `scan`, `while_loop`, their reverse passes, and a
`scan` Hessian-vector product). `--dump` writes the exact results and
`--compare` checks a second build against them bit for bit:

```sh
python examples/benchmark_host_loop_cuda.py --dump results.json
python examples/benchmark_host_loop_cuda.py --compare results.json
```

CUDA + NCCL build (`--features cuda-nccl`, two GPUs):
`examples/validate_data_parallel_cuda.py` checks single-call
`distributed.value_and_grad` parity against CPU replicas, and
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
