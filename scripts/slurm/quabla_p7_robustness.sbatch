#!/bin/bash
#SBATCH --job-name=quabla-p7-robustness
#SBATCH --output=logs/quabla_p7_robustness_%j.out
#SBATCH --error=logs/quabla_p7_robustness_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Two-GPU failure recovery, communicator lifecycle, and larger-scale training
# parity with per-step timing. Build on the head node first:
#   .venv/bin/maturin develop --release --features cuda-nccl
#   cargo test --offline -p quabla-core --features cuda-nccl --test tensor_ir --no-run
# Every step has its own timeout, so a hang is reported as exit 124 instead
# of consuming the job's time limit.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/nabla-p7}"
mkdir -p logs
source /etc/profile.d/modules.sh
module load rust/stable
module load cuda/12.6
export RUSTUP_TOOLCHAIN=stable
export PYTHONUNBUFFERED=1
QUABLA_NCCL_LIB_DIR="${QUABLA_NCCL_LIB_DIR:-$PWD/.deps/nccl/lib}"
export LD_LIBRARY_PATH="$QUABLA_NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"

command -v cargo >/dev/null 2>&1 || { echo "[ERR] cargo is unavailable" >&2; exit 2; }
command -v nvcc >/dev/null 2>&1 || { echo "[ERR] nvcc is unavailable" >&2; exit 3; }
python3 -c 'import ctypes; ctypes.CDLL("libnccl.so")' || {
  echo "[ERR] libnccl.so cannot be loaded from $QUABLA_NCCL_LIB_DIR" >&2
  exit 4
}
[ -x .venv/bin/python ] || {
  echo "[ERR] .venv is unavailable; build the cuda-nccl extension on the head node first" >&2
  exit 5
}

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

gpu_memory() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
}

# Run every step so one failure does not hide the other results.
status=0
run() {
  local label="$1" limit="$2"
  shift 2
  echo "=== $label (timeout ${limit}s)"
  local start=$SECONDS code=0
  timeout "$limit" "$@" || code=$?
  echo "=== $label exit=$code elapsed=$((SECONDS - start))s"
  [ "$code" -eq 0 ] || status=$code
}

echo "=== GPU memory before Rust tests"; gpu_memory
for test in \
  cuda_data_parallel_sharded_schedule_matches_cpu_oracle_on_two_gpus \
  cuda_data_parallel_recreates_communicators_after_a_failed_call_on_two_gpus \
  cuda_data_parallel_compile_execute_drop_cycles_keep_device_memory_stable_on_two_gpus; do
  run "$test" 300 env QUABLA_CUDA_NCCL_TEST=1 cargo test --offline -p quabla-core \
    --features cuda-nccl --test tensor_ir "$test" -- --exact --nocapture
done
echo "=== GPU memory after Rust tests"; gpu_memory

run "training parity, default 256 rows, width 32" 240 \
  .venv/bin/python -u examples/validate_data_parallel_training_cuda.py \
  --output "logs/quabla_p7_robustness_${SLURM_JOB_ID}_g16_h32.json"

# Larger scale: 50 steps without the f64 CPU mode, which is too slow here.
# The loss only needs to decrease; the first-step gradient comparison keeps
# the parity check sensitive (see the script's tolerance comment).
for config in "128 256" "128 512" "256 256" "256 512"; do
  read -r grid hidden <<< "$config"
  run "training parity, $((grid * grid)) rows, width $hidden" 240 \
    .venv/bin/python -u examples/validate_data_parallel_training_cuda.py \
    --modes two_gpu_mean,two_gpu_sum,single_gpu --steps 50 --warmup 10 \
    --grid "$grid" --hidden "$hidden" --max-loss-ratio 1.0 \
    --output "logs/quabla_p7_robustness_${SLURM_JOB_ID}_g${grid}_h${hidden}.json"
done
echo "=== GPU memory at end"; gpu_memory
exit "$status"
