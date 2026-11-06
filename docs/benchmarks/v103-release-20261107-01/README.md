# Version 1.0.3 release acceptance

Run ID: `v103-release-20261107-01` (2026-11-07, Asia/Taipei).
[quabla 1.0.3](https://pypi.org/project/quabla/1.0.3/) and
[quabla-mlx 1.0.3](https://pypi.org/project/quabla-mlx/1.0.3/) are published.
All 21 distribution files match the verified GitHub artifacts by SHA-256 on
both TestPyPI and PyPI: 16 `quabla` files and five `quabla-mlx` files.

The final tag points to `0fe097509a15d8048522818d49e017c13246e7d4`.
It differs from the runtime-tested commit `c888f11` only in the citation
version and date; CFF 1.2 schema validation passes. The initial package
workflow was cancelled before upload and the owner explicitly approved the
citation-only tag update, protected by a lease on the original tag object.
The final workflow is [42269943238](https://github.com/latteine1217/quabla/actions/runs/42269943238).
All seven build/publication jobs pass.

## Acceptance scope

- The matrix builds 20 wheels: CPython 3.10–3.14 for Linux x86_64 and aarch64
  (CPU/CUDA/NCCL), macOS arm64 CPU, and macOS arm64 MLX. Every wheel and the
  sdist pass all 20 top-level Python suites on their hosted CPU runners.
- [Corrected-source CI](https://github.com/latteine1217/quabla/actions/runs/42268863301)
  passes formatting, Ruff, Pyright, default/CUDA-NCCL Clippy, Rust tests,
  the five-interpreter Python matrix and Rust 1.88 checks. The dependency
  audit and the [macOS/MLX and MSRV checks](https://github.com/latteine1217/quabla/actions/runs/42267070723)
  pass. The latter checks the same runtime source before the citation-only change.
- The downloaded macOS CPU wheel passes all 20 suites on the local host:
  352 passes, 46 explicit hardware skips. The downloaded MLX wheel passes
  all 20 suites with real Metal enabled: 388 passes, 10 explicit skips.
  Both use fresh CPython 3.13.11 environments and assert imports resolve
  inside them. The loaded Metal native-library hash matches the wheel.
- The [shipping Linux wheel](../v103-release-wheel-lab-20261107-01/README.md)
  passes six selected suites (159 passes, four skips) and 40 real two-GPU
  public-API parity calls in Slurm job 6938. Maximum loss and gradient
  absolute errors are both `1.4901161193847656e-8`, below `1e-5`. The executed
  native library and transferred wheel match the GitHub artifact hashes.
- [Candidate validation](../v103-lab-2gpu-20261107-01/README.md) separately
  covers the complete CUDA workspace and Python suites, mixed NCCL
  schedules, communicator failure/clone recovery and 50 lifecycle cycles.
  The final device-memory drift against its warmed baseline is zero bytes.

Counts exclude the narrow-buffer child suite's four repeated passes.
Hosted wheels do not exercise GPUs; physical release-wheel checks cover
Python 3.13 on Apple M3/macOS 27.0.1 and two RTX 3090 GPUs/CUDA 12.6.
Arm CUDA hardware remains unvalidated. No home-gpu work, benchmark baseline
change or training-speedup claim is part of this release acceptance.

## Evidence and reproduction

`verification.json` indexes the run, scope, source identities and evidence
hashes. `artifact-manifest.json` contains all 21 filenames and SHA-256 hashes;
the four registry payloads are the complete accepted API responses. Earlier
partial API listings observed during index updates were retained in the
working logs and are not used as the final inventories. `wheels.json` and
the CI JSON files retain job/step outcomes; the provider links retain full
logs. The Python logs, native identity, Twine output and citation validation
are preserved alongside them.

Download this workflow's `dist-*` artifacts using `gh run download`, then
run `verify_artifacts.py ARTIFACT_DIRECTORY` with Python to verify the
20-wheel platform/interpreter matrix, metadata, bundled Metal kernels and
MLX license. `twine==6.2.0 check` checks all wheels and the source archive.
Compare each registry response's filename/SHA-256 mapping with the manifest.
The linked GPU record captures Slurm resources, loader paths, exact commands
and raw numerical results; environment preparation belongs on the head node
and execution on the compute node.
