# Contributing to Quabla

Thank you for your interest in Quabla. This guide covers what a change needs
before it can be reviewed.

## Scope and Status

Quabla is a research-grade project at version 0.x. For anything beyond a
small fix, open an issue first to discuss the approach.

Public Python names follow the compatibility policy accepted in the v0.2 API
design ([docs/api_v0_2_design.md](docs/api_v0_2_design.md), decision D14 and
section 5):

- Every Python name released in v0.1 keeps working, with unchanged
  behaviour, through all 0.x releases.
- From v0.2, names that have a replacement in the new API emit a
  `DeprecationWarning` once per name on first access, and the legacy 2D
  `Matrix` API moves to `quabla.legacy`, with a warning on top-level access.
- Deprecated names and the top-level legacy names are removed at 1.0;
  whether `quabla.legacy` stays is decided then. The new API and the
  `Compiler` facade are kept.

Backends, the tensor IR, execution plans, and the Rust crate APIs are not
covered by this policy and may change between releases.

## Building From Source

Follow [Installation From Source](README.md#installation-from-source) in the
README. It lists the prerequisites for the CPU, MLX, CUDA, and CUDA + NCCL
builds and the `maturin develop` command for each.

## Development Gates

Run these before opening a pull request:

```sh
cargo fmt --all --check
ruff check tests examples python
cargo clippy --workspace --all-targets -- -D warnings
# macOS:
cargo clippy --workspace --all-targets --features quabla-core/mlx -- -D warnings
# Linux:
cargo clippy --workspace --all-targets --features quabla-core/cuda-nccl \
  -- -D warnings
cargo test --workspace

maturin develop --release              # add --features mlx or --features cuda
python tests/python/test_matrix.py
python tests/python/test_api.py
python tests/python/test_devices_api.py
python tests/python/test_lower_program.py
python tests/python/test_optim_api.py
python tests/python/test_control_api.py
python tests/python/test_random_api.py
python tests/python/test_distributed_api.py
python tests/python/test_adam_fused.py
python tests/python/test_jacobian_chunks.py
python tests/python/test_cholesky_native.py
python tests/python/test_buffer_narrow_import.py
python tests/python/test_legacy_jacobian_direction.py
python tests/python/test_ode_api.py
python tests/python/test_linalg_api.py
python tests/python/test_packaging_api.py
python tests/python/test_custom_api.py
python tests/python/test_numpy_api.py
actionlint .github/workflows/*.yml     # when a workflow changes
```

The `quabla` package is a mixed Rust/Python project: the pure-Python
package lives in `python/quabla/`, and the compiled extension is its
private submodule `quabla._quabla`. `maturin develop` installs the package
in editable mode and writes the extension next to it as
`python/quabla/_quabla.*.so` (ignored by git), so edits to the Python files
take effect without a rebuild while Rust changes still need one. The first
`maturin develop` in an environment that holds a pre-0.2 build replaces
that build; if `quabla.__file__` does not point into `python/quabla/`
afterwards, run `python -m pip uninstall quabla` and build again.

NumPy is an optional runtime dependency: importing `quabla` never imports
it, and the NumPy interop tests in `tests/python/test_api.py` print
`skipped` without it. Install it (`python -m pip install numpy`) to run
them; CI does.

CI (`.github/workflows/ci.yml`) runs the Linux gates on every push to `main`
and every pull request. The macOS MLX check
(`.github/workflows/macos-mlx.yml`) runs on pull requests that touch the MLX
backend or the Cargo manifests, and on demand. The wheel build
(`.github/workflows/wheels.yml`) runs on version tags and on demand. It
builds the `quabla` and `quabla-mlx` distributions; a version tag then
publishes them to TestPyPI and PyPI, each upload waiting for approval in the
`testpypi` or `pypi` GitHub environment, and a manual run can publish to
TestPyPI only. To build and check a wheel locally, see
[Building Wheels](README.md#building-wheels) in the README.

The extension's type stub `python/quabla/_quabla.pyi` is maintained by hand.
When an extension class or function is added or renamed, update the stub;
`tests/python/test_packaging_api.py` fails while an exported name is
missing from it or a typed array method does not exist at run time. The
package version lives only in `crates/quabla-python/Cargo.toml`
(`pyproject.toml` declares it dynamic).

GPU runtime suites are opt-in through environment variables and need the
matching hardware, so CI does not run them:

```sh
# MLX build on Apple silicon:
QUABLA_MLX_TEST=1 python tests/python/test_matrix.py
# CUDA build with an NVIDIA GPU:
QUABLA_CUDA_TEST=1 python tests/python/test_matrix.py
QUABLA_CUDA_TEST=1 cargo test -p quabla-core --features cuda
# Two CUDA GPUs and a loadable libnccl.so:
QUABLA_CUDA_NCCL_TEST=1 cargo test -p quabla-core --features cuda-nccl
```

The Rust MLX tests run whenever `quabla-core/mlx` is enabled on macOS
(`cargo test --workspace --features quabla-core/mlx`). If a change touches a
GPU backend and you do not have the hardware, say so in the pull request.

## Commits and Pull Requests

- Write commit messages in the
  [Conventional Commits](https://www.conventionalcommits.org/) format, for
  example `fix: accept zero tangents for bool inputs in jvp`.
- Write code comments and documentation in English.
- In the pull request, state which gates you ran and on which hardware (OS,
  CPU or Apple silicon model, GPU model and driver/CUDA version). Name any
  gate you could not run.
- Keep a pull request to one logical change.

## License

Quabla is licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE)
or the [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in Quabla by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
