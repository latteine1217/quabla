# Release wheel CUDA/NCCL acceptance

Run ID: `v103-release-wheel-lab-20261107-01`. Slurm job 6938 completed with
exit `0:0` on lab-server's `r740` partition, node `acmt20`, using two RTX 3090
GPUs, four CPUs and 4 GiB host memory. The job's GPU/driver details are retained
in its stdout. CUDA/NVRTC 12.6 and the existing shared NCCL library were used.
No home-gpu work was performed.

This checks the actual CPython 3.13 Linux x86_64 wheel from
[release workflow 42269943238](https://github.com/latteine1217/quabla/actions/runs/42269943238),
commit `0fe097509a15d8048522818d49e017c13246e7d4`, rather than a source build.
Its SHA-256 is
`5455a872b91035c09fba8e10e708bcbc49357d928e8693f95f0b4d92d6cff0e5`.
The wheel was installed into a fresh CPython 3.13.11 environment on the head
node; all numerical evaluation ran through Slurm on the compute node. The
job verifies that imports resolve inside the environment, reports version
1.0.3, loads CUDA/NVRTC/NCCL, and checks the transferred wheel's hash.

Six selected suites pass: API, distributed, robustness, properties, tensor,
and linear algebra. Their top-level results are 159 passes, four explicit
skips, zero failures. Three skips require Metal; the feature-unavailable
contract skips because the NCCL runtime is enabled. The narrow-buffer child
suite's four repeated passes are not added to this total.

The public F32 data-parallel example passes all four shapes, both sum/mean
reductions and five updated-parameter steps per case (40 calls). Maximum
loss and gradient absolute errors are both `1.4901161193847656e-8`, below
`1e-5`. Native extension identity is embedded in the parity JSON.

This supplements the full candidate and communicator lifecycle validation
in [the earlier two-GPU record](../v103-lab-2gpu-20261107-01/README.md).
It does not rerun that record's Rust lifecycle tests or claim a training
speedup. The source/test files used here match the release commit; the
selected-suite scope is intentional.

`validate.sbatch` captures resources, loader paths, gates and exact commands.
To repeat it, use the release checkout, install the pinned wheel and NumPy
into `.venv`, put the wheel in `wheels/`, copy `wheel.sha256` to the checkout
root, prepare the environment on the head node, and submit the script with
`sbatch`. The node needs the stated CUDA module and NCCL loader path. Raw
stdout/stderr, parity JSON and final accounting are retained here;
`verification.json` records the checked outcomes and evidence hashes.
