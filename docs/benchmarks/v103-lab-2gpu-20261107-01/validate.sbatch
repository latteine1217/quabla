#!/bin/bash
#SBATCH --job-name=quabla-v103-2gpu
#SBATCH --output=docs/benchmarks/v103-lab-2gpu-20261107-01/job-%j.out
#SBATCH --error=docs/benchmarks/v103-lab-2gpu-20261107-01/job-%j.err
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module load rust/stable cuda/12.6
export RUSTUP_TOOLCHAIN=stable
export PATH="$HOME/.local/bin:/opt/rust/cargo/bin:$PATH"
export UV_OFFLINE=1 CARGO_NET_OFFLINE=true PYTHONUNBUFFERED=1
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:?}"
export PYO3_PYTHON="$PWD/.venv/bin/python" VIRTUAL_ENV="$PWD/.venv"
export QUABLA_CUDA_TEST=1 QUABLA_CUDA_NCCL_TEST=0
export QUABLA_NCCL_LIB_DIR=/home/junyi/nabla-p7/.deps/nccl/lib
export LD_LIBRARY_PATH="$QUABLA_NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"
run=docs/benchmarks/v103-lab-2gpu-20261107-01
stage=preflight
trap 'rc=$?; printf "stage=%s exit_code=%s\n" "$stage" "$rc"' EXIT
printf 'job=%s node=%s started=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$(date -Is)"
nvidia-smi
module list
cargo --version
rustc --version
uv --version
uv run --no-sync python - <<'PY'
import ctypes
import hashlib
import json
import sys
from pathlib import Path

for name in ("libcuda.so.1", "libnvrtc.so.12", "libnccl.so"):
    ctypes.CDLL(name)
    print("loader passed:", name)
manifest = json.loads(Path("docs/benchmarks/v103-lab-2gpu-20261107-01/source.json").read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in manifest.items())
print("source manifest passed:", len(manifest), "files")
print(sys.version)
PY
stage=build
uv run --no-sync maturin develop --uv --release --features cuda-nccl --offline --locked
stage=clippy
cargo clippy --locked --offline --workspace --all-targets --features quabla-core/cuda-nccl -- -D warnings > "$run/clippy-$SLURM_JOB_ID.log" 2>&1
stage=rust-workspace
cargo test --locked --offline --release --workspace --features quabla-core/cuda-nccl > "$run/rust-$SLURM_JOB_ID.log" 2>&1
# Keep communicator recovery and allocator checks isolated from other GPU tests.
export QUABLA_CUDA_NCCL_TEST=1
for test in \
    cuda_data_parallel_sharded_schedule_matches_cpu_oracle_on_two_gpus \
    cuda_data_parallel_recreates_communicators_after_a_failed_call_on_two_gpus \
    cuda_data_parallel_compile_execute_drop_cycles_keep_device_memory_stable_on_two_gpus
do
    stage="$test"
    timeout 240 cargo test --locked --offline --release -p quabla-core --features cuda-nccl --test tensor_ir "$test" -- --exact --nocapture --test-threads=1 > "$run/$test-$SLURM_JOB_ID.log" 2>&1
done
stage=python-parity
timeout 240 uv run --no-sync python examples/validate_data_parallel_cuda.py --output "$run/parity-$SLURM_JOB_ID.json" > "$run/parity-$SLURM_JOB_ID.log" 2>&1
stage=python-suites
uv run --no-sync bash tests/python/run_suites.sh "$PWD/.venv/bin/python" > "$run/python-$SLURM_JOB_ID.log" 2>&1
stage=final-identity
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import quabla._quabla as native

run = Path("docs/benchmarks/v103-lab-2gpu-20261107-01")
manifest = json.loads((run / "source.json").read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in manifest.items())
extension = Path(native.__file__)
result = {"status": "passed", "job_id": os.environ["SLURM_JOB_ID"],
          "source_files": len(manifest), "extension_path": str(extension),
          "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}
(run / f"completion-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
stage=passed
