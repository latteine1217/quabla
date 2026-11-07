#!/bin/bash
#SBATCH --job-name=quabla-v110-nccl-timing
#SBATCH --output=docs/benchmarks/v110-audit-20261108-04/nccl-timing-%j.out
#SBATCH --error=docs/benchmarks/v110-audit-20261108-04/nccl-timing-%j.err
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module load cuda/12.6
export PATH="$HOME/.local/bin:$PATH" UV_OFFLINE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/junyi/.local/share/uv/python/cpython-3.13.11-linux-x86_64-gnu/lib:/home/junyi/nabla-p7/.deps/nccl/lib:${LD_LIBRARY_PATH:-}"
run=docs/benchmarks/v110-audit-20261108-04
sha256sum -c "$run/source.sha256" > "$run/nccl-timing-source-before-$SLURM_JOB_ID.log"
sha256sum "$run/nccl_timing_probes.py"
.venv/bin/python "$run/nccl_timing_probes.py" --device cuda:0 > "$run/nccl-timing-$SLURM_JOB_ID.json"
sha256sum -c "$run/source.sha256" > "$run/nccl-timing-source-after-$SLURM_JOB_ID.log"
cat "$run/nccl-timing-$SLURM_JOB_ID.json"
