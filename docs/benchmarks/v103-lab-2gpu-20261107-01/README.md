# Unreleased 1.0.3: lab-server CUDA/NCCL validation

Run ID: `v103-lab-2gpu-20261107-01`. This is a correctness and lifecycle
validation of the existing candidate, not a training performance comparison.
Package versions intentionally remain 1.0.2. No home-gpu work is part of this run.

## Execution and source identity

The isolated checkout is `/home/junyi/quabla-v103-validation-20261107`.
Both attempts use Slurm partition `r740`, node `acmt20`, two RTX 3090 GPUs,
four CPUs and 16 GiB host memory. The driver is 560.35.05, CUDA is 12.6,
Rust/Cargo are 1.98.1, and CPython is 3.13.11. Exact dependency versions,
NCCL library hash and the initial scheduler snapshot are in `environment.txt`.
The job stdout records the actual GPUs, runtime loader probes and tool versions.

`source.json` pins the first attempt; `source-retry.json` also pins the retry
script. Both include the same 124 candidate source/test/configuration files
as the prior robustness verification, plus the suite runner and submission
scripts. The job checks these hashes before execution and again at completion.

## Attempts

- Job 6936, `validate.sbatch`: **FAILED**, exit 127, after core unit and
  integration tests completed. The Python Rust test executable could not load
  `libpython3.13.so.1.0`. The original stdout, stderr and Rust log are preserved.
- Job 6937, `validate-retry.sbatch`: **COMPLETED**, exit `0:0`, elapsed
  `00:12:37`. The retry adds the installed
  CPython library directory to `LD_LIBRARY_PATH`, repeats build/Clippy, runs
  the remaining workspace/doc tests, then executes the three gated NCCL tests
  serially, public API parity and all Python suites. Core tests already
  completed in 6936 are retained rather than repeated.

## Verified results

- Both all-target CUDA/NCCL Clippy runs pass.
- Core tests in 6936: 355 passed, 13 ignored. Remaining workspace/doc tests
  in 6937: 26 passed, one ignored. The three NCCL tests initially return with
  their gate off; all three are subsequently run with the gate on in isolation.
  The aggregate is 381 distinct Rust tests/doc tests passed, 14 ignored;
  the three NCCL reruns are not counted as additional distinct tests.
- Mixed sharding schedule: maximum absolute error `1.7069616564668877e-7`.
- Two intentional failures return errors, then three calls through a clone
  recover and match the CPU oracle.
- Fifty compile/execute/drop cycles pass, including 25 intentional failures.
  Both GPUs have 24,929,959,936 free bytes at cycle 1 and cycle 49: observed
  final drift is zero bytes per GPU against a 64 MiB acceptance threshold.
- Public F32 API: eight shape/reduction cases, five parameter updates each
  (40 calls); maximum loss and gradient absolute errors are both
  `1.4901161193847656e-8`, below `1e-5`. Parameters follow the CPU oracle
  updates, so this checks changed-input calls rather than optimizer divergence.
- All 20 top-level Python suites pass: 391 passed, seven explicitly skipped,
  zero failed. Six skips require Metal; one is the feature-unavailable test
  that intentionally skips when NCCL runtime validation is enabled. The
  distributed suite passes its F32 sum/mean runtime checks and F64 rejection
  contract. Property tests (four tests, 480 generated cases) and all six
  robustness tests pass.
- The API suite also invokes `test_buffer_narrow_import.py` in an isolated
  child interpreter: four additional passes, subsequently repeated by the
  top-level runner. Counting that child summary produces 21 suite executions
  and 395 passes; it does not mean 21 distinct suites or 395 distinct tests.
  Earlier 1.0.3 summaries used that inclusive count. Their original raw logs
  and summary JSON are preserved; the roadmap now uses top-level counts.
- Final Slurm accounting is saved in `accounting.txt`. The completion source
  check passes for all 127 manifest entries, including the unchanged 124-file
  candidate. The independently read post-run extension SHA-256 matches both
  the parity report and completion marker:
  `81037711bcb754b3958f48c137f48959819ba993a11a534834f9d652779546ae`.

No source API, numerical policy, release version or published artifact changes
are part of this validation. These memory measurements cover 50 cycles against
the warmed cycle-1 baseline; they do not establish unlimited-run memory behavior.
`verification.json` contains the checked counts, scope, identities and evidence
hashes. The first attempt remains recorded as failed, despite its retained
completed core tests.
