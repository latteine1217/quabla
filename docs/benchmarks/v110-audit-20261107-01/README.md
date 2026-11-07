# 1.1.0 robustness audit — 2026-11-07

Status: retained first snapshot; the [next snapshot](../v110-audit-20261107-02/README.md)
adds the `saveat` mesh-knot derivative fix and is the current Python
acceptance target. This is a correctness audit, not a
performance comparison or a published 1.1.0 release. The tested source
still carries package version 1.0.3; the owner designated the fixes for
1.1.0. Runtime files are frozen by `source.json` and `source.sha256`
(134 files). Documentation may be updated without replacing that snapshot.

## Current verification

| Target | Result | Evidence |
| --- | --- | --- |
| macOS arm64 CPU wheels, Python 3.10, 3.11, 3.12, 3.13, 3.14 | Each: 20 suites, 371 passed, 47 skipped, 0 failed | `python-py310.log` through `python-py314.log`; wheels under `target/v110-python-matrix/` |
| Local Python 3.13.11, CPU and MLX gates enabled | 20 suites, 408 passed, 10 skipped, 0 failed | `python-py313-mlx.log` |
| Local Rust workspace, MLX enabled | Passed; gated/ignored tests are not GPU evidence | `rust-local-mlx.log` |
| Local static/dependency gates | Ruff, Pyright, Rust formatting, MLX Clippy and cargo-deny passed | `pyright.log`, `clippy-local-mlx.log`, `deny.log` |
| Independent CPU and MLX probes | Each: 131 checks passed with four fixed seeds | `device_probes.py`, `probes-cpu-strict.log`, `probes-mlx-strict.log` |
| Linux x86_64 CUDA/NCCL, Python 3.13.11 | Job 6939 COMPLETED, exit 0: Rust 381 passed/14 ignored, three isolated NCCL tests passed, Python 411 passed/7 skipped, eight parity cases passed | `accounting.txt`, `rust-6939.log`, `python-6939.log`, `parity-6939.json`, `completion-6939.json` |
| Independent CUDA probes | Job 6940 COMPLETED, exit 0: 131 checks passed | `accounting.txt`, `probes-cuda-6940.json`, `probes.sha256` |

The independent probes use scalar analytical derivatives of a PINN-style
loss with nested input derivatives and parameter gradients, diagonal
linear-system solutions, and extreme-scale interpolation/integration.
Their seeds are 103, 1701, 65537 and 8675309. The strict probe uses relative
accuracy and an absolute tolerance proportional to the expected vector's
scale. Initial probe results and `device_probes-v1.py` are retained as
historical evidence; the strict results are the current acceptance checks.

The first matrix runner was edited while Bash was executing it, causing a
trailing parser error after four successful per-version suites. Those raw
per-version logs each end with `all suites passed`. Python 3.13 was then
built and tested using a fixed script copy. `python-matrix.sh` is the
reusable runner; do not modify a script while it is executing.

## Runtime fixes under verification

- Krylov RHS/initial-guess scaling, exact homogeneous solutions, stable
  residual norms, and true-residual convergence verification.
- L-BFGS directional-slope rescaling and stable learning-rate schedules.
- Consistent eager/traced control-flow predicate validation.
- Complete partial stream writes and strict saved-file metadata validation.
- Extreme finite-interval interpolation, trapezoidal integration and
  fixed-step ODE grids, including their relevant derivative checks.

Durable regressions are in the corresponding `tests/python/test_*.py`
suites; release notes are in `CHANGELOG.md` under Unreleased. Reproductions
were run before their fixes, then the affected suites and the complete
local suite were rerun.

## Scope and remaining acceptance

The remote checkout is `/home/junyi/quabla-v110-audit-20261107-01`; work is
scheduled in partition `r740`, compute node `acmt20` (not the head node).
Source hashes are checked before and after each remote run. A scheduling
dependency or successful environment probe is not a numerical result:
acceptance requires terminal Slurm accounting, exit code zero, every
required log, the parity JSON, and the completion marker.

Jobs 6939/6940 and their raw evidence have been independently checked after
completion. This completes first-snapshot verification; newer Python fixes
need the later snapshots' checks, not this older Python result.
Finite tests cannot prove absence of all bugs. Linux aarch64, Linux wheels
on all five CPythons, physical memory exhaustion and performance regression
measurement are outside the currently executed validation scope.
