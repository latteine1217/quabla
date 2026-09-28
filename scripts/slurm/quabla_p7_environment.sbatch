#!/bin/bash
#SBATCH --job-name=quabla-p7-env
#SBATCH --output=logs/quabla_p7_env_%j.out
#SBATCH --error=logs/quabla_p7_env_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=8G
#SBATCH --time=00:03:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/nabla-p7}"
mkdir -p logs
source /etc/profile.d/modules.sh
module load rust/stable
module load cuda/12.6
export RUSTUP_TOOLCHAIN=stable
QUABLA_NCCL_LIB_DIR="${QUABLA_NCCL_LIB_DIR:-$PWD/.deps/nccl/lib}"
export LD_LIBRARY_PATH="$QUABLA_NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"

command -v cargo >/dev/null 2>&1 || {
  echo "[ERR] cargo is unavailable; expected $HOME/.cargo/bin/cargo or a PATH entry" >&2
  exit 2
}
command -v rustc >/dev/null 2>&1 || {
  echo "[ERR] rustc is unavailable; expected $HOME/.cargo/bin/rustc or a PATH entry" >&2
  exit 3
}
command -v nvcc >/dev/null 2>&1 || {
  echo "[ERR] nvcc is unavailable" >&2
  exit 4
}
python3 -c 'import ctypes; ctypes.CDLL("libnccl.so")' || {
  echo "[ERR] libnccl.so cannot be loaded from $QUABLA_NCCL_LIB_DIR" >&2
  exit 5
}

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader

echo "nccl-libraries"
ls -l "$QUABLA_NCCL_LIB_DIR"/libnccl.so* 2>/dev/null || true
for path in /usr/lib/x86_64-linux-gnu/libnccl.so* /usr/local/cuda/lib64/libnccl.so*; do
  [ -e "$path" ] && printf '%s\n' "$path"
done
find "$HOME/.local" -maxdepth 5 -iname 'libnccl.so*' -type f 2>/dev/null | head -20 || true
ldconfig -p 2>/dev/null | grep -i nccl || true

echo "toolchain"
command -v cargo
command -v rustc
cargo --version
rustc --version
