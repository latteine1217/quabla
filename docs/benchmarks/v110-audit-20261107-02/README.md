# 1.1.0 robustness audit, second Python snapshot — 2026-11-07

Status: local checks pass; CUDA/NCCL job 6941 COMPLETED with exit 0. The
[third snapshot](../v110-audit-20261107-03/README.md) adds the GELU fix and
is now the current Python acceptance target. This source
snapshot superseded the first snapshot for Python acceptance after finding
an additional `saveat` time-gradient defect. Neither snapshot is a published
1.1.0 release; package metadata remains 1.0.3 during the audit.

## Identity and native verification

`source.json` and `source.sha256` freeze 135 files. All native sources and
Cargo manifests match the [first snapshot](../v110-audit-20261107-01/README.md).
The remote Python snapshot reuses that run's CUDA/NCCL extension only after
asserting identical native source hashes and identical extension bytes.
`PYTHONPATH` selects this checkout, and preflight checks both the Python
package and native module paths. No in-flight source snapshot is overwritten.

## Current results

| Check | Result | Evidence |
| --- | --- | --- |
| macOS arm64 CPU wheel, each CPython 3.10–3.14 | Each: 20 suites, 372 passed, 47 skipped, 0 failed | `python-py310.log` through `python-py314.log`; `local-python-verification.json` |
| Local Python 3.13.11 with MLX enabled | 20 suites, 409 passed, 10 skipped, 0 failed | `python-py313-mlx.log` |
| Independent CPU and MLX analytical probes | Each: 137 checks passed, four fixed seeds | `device_probes.py`, `probes-cpu.log`, `probes-mlx.log` |
| Ruff and Pyright | Passed | `pyright.log`; Ruff checked the source/test/example trees |
| Rust formatting, MLX Clippy/workspace tests and dependency audit | Native sources unchanged; first snapshot's results retained | First snapshot's `rust-local-mlx.log`, `clippy-local-mlx.log`, `deny.log` |
| Linux x86_64 CUDA/NCCL Python, parity and independent probes | Job 6941 COMPLETED, exit 0: Python 412 passed/7 skipped, 137 probes, eight parity cases passed; maximum loss/gradient error 1.49e-8 | `accounting.txt`, `python-6941.log`, `probes-cuda-6941.json`, `parity-6941.json`, `completion-6941.json` |

The newest regression integrates `y'=1` and samples exactly at an internal
fixed-grid knot. The sampled value must have zero derivative in the terminal
time and unit derivative in the query time. The previous clipping of the
interpolation fraction discarded these derivatives at the knot. The fix
uses the already-validated interval mask to bound the query before forming
the fraction, without a second clipping operation. Euler, Heun and RK4
regressions run eagerly and under `jit`; the independent device probes
also check them in float32. An additional manual scan of all 15 internal
knots, three methods and both integration directions passed 180 derivative
checks.

## Remaining acceptance and scope

Terminal Slurm accounting, all required logs, parity errors, the completion
JSON and the archived source hashes have been independently checked. This
completes second-snapshot verification; newer Python fixes require the
third snapshot's checks. Preserve unsuccessful runs with their actual status.
The first snapshot's Rust/CUDA suite is reusable only because its native
sources and binary are identical; its older Python result cannot replace
this run's latest-source checks.

These checks do not prove the absence of every possible bug. Linux aarch64,
the full Linux wheel/Python installation matrix, physical RAM exhaustion,
and performance regression measurement remain outside this executed scope.
No release, tag, commit or publication is implied by passing these checks.
