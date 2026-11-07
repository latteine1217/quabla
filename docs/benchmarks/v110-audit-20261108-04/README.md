# 1.1.0 robustness audit, current snapshot — 2026-11-08

Status: functional acceptance passed within the recorded scope. Jobs
6947–6950 are COMPLETED with exit code 0. `verify_acceptance.py` checks
terminal accounting, final source/archive/binary identities, uploaded driver
identity, all full-suite summaries, probe hashes and parity errors.
`acceptance.json` and `verification.log` preserve the checked results.
Package metadata remains 1.0.3. This record validates a source snapshot; it
does not publish or tag a release.

## Repairs and source identity

Mean retries finite-input overflow at a safe scale on CPU, MLX and CUDA,
while preserving ordinary zero cancellation and existing non-finite-input
handling. CPU axis means recover subnormal values lost by per-lane scaling.
Variance uses binary scaling and sample centering before squared deviations;
standard deviation uses a scaled norm. Regressions cover overflowing sums,
constant data, adjacent large floats, finite standard deviations whose
variance overflows, gradients, vmap, JVPs and Hessian-vector products.
Public signatures, dtype policy and `sum` behaviour remain unchanged.

`source.json`, `source.sha256` and `source-snapshot.tar.gz` preserve 135 files.
The CUDA-only constant-table array length omitted in snapshot 02 is fixed.
This snapshot rebuilds native CUDA/NCCL code rather than reusing the prior
2026-11-07 extension. Sources are checked before and after execution.

The archive includes 135 files from the tested workspace. Three context
files (`README.md`, `AGENTS.md`, `examples/benchmark_frameworks.py`) include
pre-existing documentation/performance work outside the bug-fix commit.
The verifier retains all 135 archived hashes and original compute-node
checks, and compares the other 132 files with the current tree. README's
bug-fix audit index was added after numerical validation; no runtime source
changed. Existing unrelated workspace edits are not part of this commit.

## Verification

| Check | Observed result / next requirement | Evidence |
| --- | --- | --- |
| macOS arm64 CPU wheels, Python 3.10–3.14 | Each: 20 suites, 377 passed, 48 skipped, 0 failed | `python-py310.log` through `python-py314.log` |
| Local Python 3.13.11, MLX enabled | 20 suites, 415 passed, 10 skipped, 0 failed, repeated after native rebuild | `python-py313-mlx-rebuilt.log` |
| Local Rust, MLX enabled | 337 passed, 16 ignored, 0 failed | `rust-mlx.log` |
| Fmt, Clippy, Ruff, Pyright, dependency audit | Passed | `fmt.log`, `clippy.log`, `ruff.log`, `pyright.log`, `deny.log` |
| Independent analytical probes | CPU 140 and MLX 140 passed | `probes-{cpu,mlx}.json` |
| Adjacent-value spread, gradient, vmap, mean JVP probes | CPU 57 and MLX 28 passed; one Metal subnormal variance excluded | `spread-{cpu,mlx}.json` |
| Variance/std Hessian-vector products | CPU 14 and MLX 7 passed | `curvature-{cpu,mlx}.json` |
| Decimal statistics oracle, precision 128, seed 1701 | CPU 480 passed; MLX 216 passed, 24 Metal subnormal outputs excluded | `statistics-{cpu,mlx}.json` |
| CUDA/NCCL full build and native/Python suites | Job 6947 COMPLETED, exit 0: Rust 382 passed/14 ignored; Python 418 passed/7 skipped; three additional isolated NCCL cases passed | `remote-final/`, `accounting.txt` |
| Independent CUDA probes | 140 analytical, 41 direct float64, 57 spread/composition, 14 curvature and 480 Decimal comparisons passed | `remote-final/`, `acceptance.json` |
| Two-GPU sum/mean parity | Eight cases, five changed-parameter calls each; maximum loss and gradient errors both 1.4901161193847656e-8, tolerance 1e-5 | `remote-final/parity-6947.json` |
| Native two-GPU parity and timing observability | Six sum/mean calls pass; replica enqueue, collective and output readback timings finite and nonnegative; jobs 6948–6950 COMPLETED, exit 0 | `remote-final/nccl-timing-6950.json`, `accounting.txt` |

The Python summary verifier counts each suite's own summary, excluding the
nested buffer-import subprocess output printed inside `test_api.py`.

## Measured cost and limits

Three sequential baseline/candidate process pairs measure float32 `[64,64]`
reductions with precreated host inputs and scalar host return/synchronization.
Each process records first-call costs separately, warms up 20 calls and saves
seven batches of 100 calls. Baseline source is snapshot 20261107-03, retained
under `target/v110-mean-baseline/python`; its Python package path and old
native extension are recorded in each raw JSON. Candidate sources match this
snapshot. All compared outputs agree within float32 tolerance.

`timing-summary.json` and `timing-{baseline,candidate}-{1,2,3}.json` retain the
observed overhead, including CPU var/std and MLX mean/var/std regressions.
| Device | Operation | Baseline median (us) | Candidate median (us) | Paired ratio range |
| --- | --- | --- | --- | --- |
| cpu | mean | 4.89 | 5.07 | 0.99–1.04x |
| cpu | var | 11.42 | 29.40 | 2.56–2.76x |
| cpu | std | 11.42 | 46.21 | 4.02–4.24x |
| mlx | mean | 178.34 | 366.44 | 1.98–2.37x |
| mlx | var | 220.67 | 843.54 | 3.03–5.28x |
| mlx | std | 234.87 | 614.05 | 2.41–3.81x |

The measurements are host-returning microbenchmarks with timing variability,
not resident training or compiled-framework comparisons. No speedup is
claimed, and PINN training performance of this candidate remains unmeasured.

Metal flushes float32 subnormal results to zero; excluded probe cases are
explicit, not passing checks. Finite tests cannot prove the absence of all
bugs. Linux aarch64, the full Linux wheel/Python installation matrix and
physical RAM exhaustion remain outside this audit's executed scope.

## Rejected and superseded attempts

[Snapshot 01](../v110-audit-20261108-01/README.md) preserves the precision and
zero-cancellation failures; job 6944 was cancelled after a Rust failure.
[Snapshot 02](../v110-audit-20261108-02/README.md) preserves the CUDA build
failure (6945) and a reproducible std-gradient overflow. [Snapshot 03](../v110-audit-20261108-03/README.md)
repairs that gradient but retained the CUDA declaration error; job 6946 was
cancelled before starting. No failed or cancelled job is passing evidence.
