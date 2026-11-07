#!/bin/bash
#SBATCH --job-name=quabla-v110-f64
#SBATCH --output=docs/benchmarks/v110-audit-20261107-03/float64-job-%j.out
#SBATCH --error=docs/benchmarks/v110-audit-20261107-03/float64-job-%j.err
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module load cuda/12.6
export PYTHONPATH="$PWD/python" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/junyi/.local/share/uv/python/cpython-3.13.11-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
run=docs/benchmarks/v110-audit-20261107-03
sha256sum -c "$run/source.sha256" > "$run/float64-source-before-$SLURM_JOB_ID.log"
sha256sum -c "$run/float64-probes.sha256"
.venv/bin/python "$run/cuda_float64_probes.py" > "$run/probes-cuda-float64-$SLURM_JOB_ID.json"
sha256sum -c "$run/source.sha256" > "$run/float64-source-after-$SLURM_JOB_ID.log"
cat "$run/probes-cuda-float64-$SLURM_JOB_ID.json"
