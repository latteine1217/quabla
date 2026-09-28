# Contributing to Quabla

Thank you for your interest in Quabla. This guide covers what a change needs
before it can be reviewed.

## Scope and Status

Quabla is a research-grade project at version 0.x. APIs, backends, and
internal representations change without deprecation periods. For anything
beyond a small fix, open an issue first to discuss the approach.

## Building From Source

Follow [Installation From Source](README.md#installation-from-source) in the
README. It lists the prerequisites for the CPU, MLX, CUDA, and CUDA + NCCL
builds and the `maturin develop` command for each.

## Development Gates

Run the gates listed under
[Development Gates](README.md#development-gates) in the README before opening
a pull request. In brief:

```sh
cargo fmt --all --check
ruff check tests examples
cargo clippy --workspace --all-targets -- -D warnings
# macOS:
cargo clippy --workspace --all-targets --features quabla-core/mlx -- -D warnings
# Linux:
cargo clippy --workspace --all-targets --features quabla-core/cuda-nccl \
  -- -D warnings
cargo test --workspace

maturin develop --release
python tests/python/test_matrix.py
```

CI (`.github/workflows/ci.yml`) runs the Linux gates on every pull request.
The macOS MLX check (`.github/workflows/macos-mlx.yml`) runs on pull requests
that touch the MLX backend or the Cargo manifests.

GPU runtime suites are opt-in and need the matching hardware, so CI does not
run them:

- `QUABLA_MLX_TEST=1`: MLX build on Apple silicon.
- `QUABLA_CUDA_TEST=1`: CUDA build with an NVIDIA GPU.
- `QUABLA_CUDA_NCCL_TEST=1`: CUDA + NCCL build with at least two GPUs and a
  loadable `libnccl.so`.

The README shows the exact command for each suite. If a change touches a GPU
backend and you do not have the hardware, say so in the pull request.

## Commits and Pull Requests

- Write commit messages in the
  [Conventional Commits](https://www.conventionalcommits.org/) format, for
  example `fix: accept zero tangents for bool inputs in jvp`.
- Write code comments and documentation in English.
- In the pull request, state which gates you ran and on which hardware (OS,
  CPU or Apple silicon model, GPU model and driver/CUDA version). Name any
  gate you could not run.
- Keep a pull request to one logical change.

## Code of Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Quabla is licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE)
or the [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in Quabla by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
