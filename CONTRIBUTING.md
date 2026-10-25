# Contributing to Quabla

Thank you for your interest in Quabla. This guide covers what a change needs
before it can be reviewed.

## Scope and Status

For anything beyond a small fix, open an issue first to discuss the
approach.

From 1.0 the public Python API follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- A name exported by `quabla` (its `__all__` and public submodules,
  `quabla.distributed` included) keeps its behaviour through 1.x. Minor
  releases may add names and arguments.
- A name to be removed first emits a `DeprecationWarning` for at least one
  minor release and is removed only in the next major release.
- The minimum supported Rust version (`rust-version` in `Cargo.toml`) is
  raised only in minor releases, and the CHANGELOG names the new version.
- Numerical results may change in the last bits between releases when an
  operation's evaluation order changes; such changes are listed in the
  CHANGELOG.

v1.0 removed the v0.1 names that the v0.2 API design
([docs/api_v0_2_design.md](docs/api_v0_2_design.md), decision D14 and
section 5) had deprecated, together with `quabla.legacy`.

Backends, the tensor IR, execution plans, the private module
`quabla._quabla`, and the Rust crate APIs are not covered by this policy
and may change between releases.

## Building From Source

Follow [Installation From Source](README.md#installation-from-source) in the
README. It lists the prerequisites for the CPU, MLX, CUDA, and CUDA + NCCL
builds and the `maturin develop` command for each.

## Development Gates

Run these before opening a pull request:

```sh
cargo fmt --all --check
ruff check tests examples python
pyright                    # type check of python/quabla (pyrightconfig.json)
cargo clippy --workspace --all-targets -- -D warnings
# macOS:
cargo clippy --workspace --all-targets --features quabla-core/mlx -- -D warnings
# Linux:
cargo clippy --workspace --all-targets --features quabla-core/cuda-nccl \
  -- -D warnings
cargo test --workspace
cargo deny check           # dependency audit (deny.toml); cargo install cargo-deny

maturin develop --release              # add --features mlx or --features cuda
tests/python/run_suites.sh python        # every tests/python/test_*.py
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
it, and the tests that need it report `SKIP` without it. Install it
(`python -m pip install numpy`) to run them; CI does.

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

Each Python suite is a plain script built on the shared harness
`tests/python/_support.py`, and runs on its own with `python
tests/python/test_x.py`. Its runner calls the `test_*` functions in
definition order and prints one line per test: `PASS name`, `PASS name
(gate off: ...)` when part of the test was gated off, `SKIP name: reason`
when the test could not run, or `FAIL name` with the traceback; then a
summary line such as `test_api.py: 102 passed, 4 skipped, 0 failed`. The
script exits non-zero when any test failed. A new suite ends with
`run(globals())` and takes `raises`, `skip`, and the gate helpers from the
harness.

`tests/python/run_suites.sh PYTHON` runs every `tests/python/test_*.py`
with the given interpreter, so a new suite needs no list entry, and exits
non-zero when any suite failed. CI runs it on Python 3.10 to 3.14, the
versions the classifiers in `pyproject.toml` list, and the wheel build runs
it against every wheel it builds (`tests/python/check_wheels.sh`).

GPU runtime suites are opt-in through environment variables and need the
matching hardware, so CI does not run them. A gate is on exactly when its
variable is `1`; any other value, `0` included, leaves it off. The Python
suites (`enabled`, `require`, and `devices` in `tests/python/_support.py`)
and the Rust tests (`Gate` in `crates/quabla-core/tests/support/mod.rs`)
apply the same rule. A gated-off Python test reports `SKIP` with the
variable to set, and one whose gate is on for a build without that target
fails; a gated-off Rust test returns early and passes:

```sh
# MLX build on Apple silicon:
QUABLA_MLX_TEST=1 python tests/python/test_api.py
# CUDA build with an NVIDIA GPU:
QUABLA_CUDA_TEST=1 python tests/python/test_api.py
QUABLA_CUDA_TEST=1 cargo test -p quabla-core --features cuda
# Two CUDA GPUs and a loadable libnccl.so:
QUABLA_CUDA_NCCL_TEST=1 cargo test -p quabla-core --features cuda-nccl
```

The Rust MLX tests run whenever `quabla-core/mlx` is enabled on macOS
(`cargo test --workspace --features quabla-core/mlx`); a few of them also
need `QUABLA_MLX_TEST=1`. If a change touches a GPU backend and you do not
have the hardware, say so in the pull request.

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
