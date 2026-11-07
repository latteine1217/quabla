# 1.1.0 robustness audit, current snapshot — 2026-11-07

Status: all executed acceptance checks pass. Jobs 6942 and 6943 are
COMPLETED with exit code 0; terminal accounting, full logs, source identities,
parity errors and the completion marker have been independently checked.
This audit targets a future 1.1.0 bug-fix release. The
tested package metadata remains 1.0.3; no release or publication has occurred.

## Source and evidence

`source.json`, `source.sha256`, and `source-snapshot.tar.gz` preserve the
135-file snapshot. Native sources, Cargo manifests and the reused CUDA/NCCL
extension are identical to the [first snapshot](../v110-audit-20261107-01/README.md).
Its job 6939 completed with 381 Rust tests passing, 14 ignored, and three
additional isolated NCCL regressions passing. Jobs 6939/6940 also completed
their older Python/parity/probe checks; later Python fixes are accepted
only through this snapshot's checks.

| Current-source check | Result | Evidence |
| --- | --- | --- |
| macOS arm64 CPU wheel, CPython 3.10–3.14 | Each: 20 suites, 373 passed, 47 skipped, 0 failed | `python-py310.log` through `python-py314.log`, `local-python-verification.json` |
| Local Python 3.13.11 with MLX enabled | 20 suites, 410 passed, 10 skipped, 0 failed | `python-py313-mlx.log` |
| Independent CPU and MLX analytical probes | Each: 140 checks passed | `device_probes.py`, `probes-cpu.log`, `probes-mlx.log` |
| Ruff and Pyright | Passed | Source/test/example Ruff gate; `pyright.log` |
| Current-source CUDA analytical probes and two-GPU parity | 140 checks and eight parity cases passed; maximum loss/gradient error 1.49e-8 | `probes-cuda-6942.json`, `parity-6942.json` |
| Latest Linux x86_64 CUDA/NCCL full Python suite | Job 6942 COMPLETED, exit 0: 20 suites, 413 passed, 7 skipped, 0 failed | `accounting.txt`, `python-6942.log`, `completion-6942.json` |
| Direct CUDA float64 boundary checks | Job 6943 COMPLETED, exit 0: 11 checks passed | `float64-final/`, `cuda_float64_probes.py`, `float64-probes.sha256` |

The most recent additional defect was approximate GELU: finite large inputs
kept finite values but produced NaN first/second derivatives when the cubic
overflowed. For finite `|x| > 40`, the sigmoid tail is already beyond the
float64 underflow range (argument magnitude above 4600). The fix binds the
unselected cubic to a finite input and uses the corresponding linear/zero
limit, preserving the ordinary interval and prior NaN/infinity behaviour.
CPU float32/float64 regressions cover values and first/second derivatives;
the device probes repeat the float32 cases with exact component checks.

The [second snapshot](../v110-audit-20261107-02/README.md) preserves the
preceding `saveat` mesh-knot derivative fix and its local results. All raw
snapshots are retained; no active checkout is overwritten. An attempted
cancellation of obsolete queued job 6941 was deliberately not executed
because it had already started, so that run is allowed to complete.

## Acceptance and remaining scope

`acceptance.json` records the checked source/archive hashes, Python matrix,
Slurm accounting, CUDA probes and parity maxima. The final CUDA float64
probe additionally checks extreme GELU values/gradients/curvature, interval
interpolation and its gradient, trapezoid values/endpoint gradients, and
fixed-step ODE grids. The `float64-final/` directory contains the definitive
post-completion download; any earlier root-level copies of those log files
are not the acceptance source. Runtime files still match the 135-file
snapshot. A subsequent runtime change requires new verification.

Finite tests cannot establish that every possible input is bug-free.
Linux aarch64, the complete Linux wheel/Python installation matrix,
physical RAM exhaustion and performance regression measurements remain
outside the executed scope. The supported public API signatures and dtype
policy remain unchanged. No git commit, tag, push or publication is implied.
