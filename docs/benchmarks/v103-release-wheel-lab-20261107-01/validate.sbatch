#!/bin/bash
#SBATCH --job-name=quabla-v103-wheel
#SBATCH --output=docs/benchmarks/v103-release-wheel-lab-20261107-01/job-%j.out
#SBATCH --error=docs/benchmarks/v103-release-wheel-lab-20261107-01/job-%j.err
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module load cuda/12.6
export PATH="$HOME/.local/bin:$PATH"
export UV_OFFLINE=1 PYTHONUNBUFFERED=1
export QUABLA_CUDA_TEST=1 QUABLA_CUDA_NCCL_TEST=1
export LD_LIBRARY_PATH="/home/junyi/nabla-p7/.deps/nccl/lib:/home/junyi/.local/share/uv/python/cpython-3.13.11-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
run=docs/benchmarks/v103-release-wheel-lab-20261107-01
nvidia-smi
sha256sum -c wheel.sha256
uv run --no-project --python "$PWD/.venv/bin/python" python - <<'PY'
import ctypes
import hashlib
import json
import sys
from pathlib import Path
import quabla
for name in ("libcuda.so.1", "libnvrtc.so.12", "libnccl.so"):
    ctypes.CDLL(name)
assert quabla.__version__ == "1.0.3"
assert quabla.__file__.startswith(sys.prefix), quabla.__file__
assert quabla.devices() == ["cpu", "cuda"]
native = Path(quabla._quabla.__file__)
print(json.dumps({"version": quabla.__version__, "native": str(native),
                  "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}))
PY
timeout 180 .venv/bin/python examples/validate_data_parallel_cuda.py --output "$run/parity-$SLURM_JOB_ID.json"
for suite in test_api.py test_distributed_api.py test_robustness_api.py test_properties_api.py test_tensor_api.py test_linalg_api.py; do
    printf '=== %s ===\n' "$suite"
    timeout 120 .venv/bin/python "tests/python/$suite"
done
printf '{"status":"passed","job_id":"%s"}\n' "$SLURM_JOB_ID"
