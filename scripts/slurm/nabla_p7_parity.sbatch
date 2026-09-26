#!/bin/bash
#SBATCH --job-name=nabla-p7-parity
#SBATCH --output=logs/nabla_p7_parity_%j.out
#SBATCH --error=logs/nabla_p7_parity_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/nabla-p7}"
mkdir -p logs
source /etc/profile.d/modules.sh
module load rust/stable
module load cuda/12.6
export RUSTUP_TOOLCHAIN=stable
export PYTHONUNBUFFERED=1
NABLA_NCCL_LIB_DIR="${NABLA_NCCL_LIB_DIR:-$PWD/.deps/nccl/lib}"
export LD_LIBRARY_PATH="$NABLA_NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"

command -v cargo >/dev/null 2>&1 || { echo "[ERR] cargo is unavailable" >&2; exit 2; }
command -v nvcc >/dev/null 2>&1 || { echo "[ERR] nvcc is unavailable" >&2; exit 3; }
python3 -c 'import ctypes; ctypes.CDLL("libnccl.so")' || {
  echo "[ERR] libnccl.so cannot be loaded from $NABLA_NCCL_LIB_DIR" >&2
  exit 4
}
[ -x .venv/bin/python ] || {
  echo "[ERR] .venv is unavailable; build the cuda-nccl extension on the head node first" >&2
  exit 5
}

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
cargo --version
rustc --version

# Run both checks so one failure does not hide the other result.
status=0
echo "=== caller-reduction parity (Python value-and-grad)"
.venv/bin/python -u examples/validate_data_parallel_cuda.py || status=$?
# Prebuild on the head node with
#   cargo test --offline -p nabla-core --features cuda-nccl --test tensor_ir --no-run
# so this step only runs the already compiled test.
echo "=== sharding-plan schedule parity (Rust, mixed Sum/Mean)"
NABLA_CUDA_NCCL_TEST=1 cargo test --offline -p nabla-core --features cuda-nccl \
  --test tensor_ir cuda_data_parallel_sharded_schedule_matches_cpu_oracle_on_two_gpus \
  -- --exact --nocapture || status=$?
exit "$status"
