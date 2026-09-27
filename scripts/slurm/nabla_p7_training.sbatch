#!/bin/bash
#SBATCH --job-name=nabla-p7-training
#SBATCH --output=logs/nabla_p7_training_%j.out
#SBATCH --error=logs/nabla_p7_training_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Multi-step two-GPU training parity and steady-state collective timing.
# Build the cuda-nccl extension into .venv on the head node first. Optional:
# NABLA_BASELINE_PYTHONPATH names a directory holding another build of the
# extension; its two-GPU timing runs first for a before/after comparison.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/nabla-p7}"
mkdir -p logs
source /etc/profile.d/modules.sh
module load cuda/12.6
export PYTHONUNBUFFERED=1
NABLA_NCCL_LIB_DIR="${NABLA_NCCL_LIB_DIR:-$PWD/.deps/nccl/lib}"
export LD_LIBRARY_PATH="$NABLA_NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"

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

status=0
if [ -n "${NABLA_BASELINE_PYTHONPATH:-}" ]; then
  echo "=== baseline extension two-GPU timing ($NABLA_BASELINE_PYTHONPATH)"
  PYTHONPATH="$NABLA_BASELINE_PYTHONPATH" .venv/bin/python -u \
    examples/validate_data_parallel_training_cuda.py --modes two_gpu_mean \
    --output "logs/nabla_p7_training_${SLURM_JOB_ID}_baseline.json" || status=$?
fi
echo "=== training parity: two GPUs vs single GPU vs CPU"
.venv/bin/python -u examples/validate_data_parallel_training_cuda.py \
  --output "logs/nabla_p7_training_${SLURM_JOB_ID}.json" || status=$?
exit "$status"
