#!/bin/bash
#SBATCH --job-name=quabla-v110-audit
#SBATCH --output=docs/benchmarks/v110-audit-20261108-04/job-%j.out
#SBATCH --error=docs/benchmarks/v110-audit-20261108-04/job-%j.err
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
export LD_LIBRARY_PATH="/home/junyi/.local/share/uv/python/cpython-3.13.11-linux-x86_64-gnu/lib:$QUABLA_NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"
run=docs/benchmarks/v110-audit-20261108-04
stage=preflight
trap 'rc=$?; printf "stage=%s exit_code=%s\n" "$stage" "$rc"' EXIT
printf 'job=%s node=%s started=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$(date -Is)"
nvidia-smi
module list
cargo --version
rustc --version
uv --version
sha256sum -c "$run/source.sha256" > "$run/source-before-$SLURM_JOB_ID.log"
uv run --no-sync python - <<'PY'
import ctypes
import sys
for name in ("libpython3.13.so.1.0", "libcuda.so.1", "libnvrtc.so.12", "libnccl.so"):
    ctypes.CDLL(name)
    print("loader passed:", name)
print(sys.version)
PY
stage=build
uv run --no-sync maturin develop --uv --release --features cuda-nccl --offline --locked
stage=clippy
cargo clippy --locked --offline --workspace --all-targets --features quabla-core/cuda-nccl -- -D warnings > "$run/clippy-$SLURM_JOB_ID.log" 2>&1
stage=rust-workspace
cargo test --locked --offline --release --workspace --features quabla-core/cuda-nccl -- --test-threads=1 > "$run/rust-$SLURM_JOB_ID.log" 2>&1
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
stage=independent-probes
uv run --no-sync python "$run/device_probes.py" --device cuda > "$run/probes-cuda-$SLURM_JOB_ID.json" 2>&1
uv run --no-sync python "$run/cuda_float64_probes.py" > "$run/float64-$SLURM_JOB_ID.json" 2> "$run/float64-$SLURM_JOB_ID.err"
uv run --no-sync python "$run/spread_probes.py" --device cuda:0 > "$run/spread-cuda-$SLURM_JOB_ID.json"
stage=final-identity
sha256sum -c "$run/source.sha256" > "$run/source-after-$SLURM_JOB_ID.log"
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import quabla
import quabla._quabla as native
run = Path("docs/benchmarks/v110-audit-20261108-04")
extension = Path(native.__file__)
result = {
    "status": "passed", "job_id": os.environ["SLURM_JOB_ID"],
    "package_version": quabla.__version__,
    "extension_path": str(extension),
    "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
}
(run / f"completion-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
stage=passed
