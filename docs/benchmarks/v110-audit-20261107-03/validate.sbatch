#!/bin/bash
#SBATCH --job-name=quabla-v110-cycle6
#SBATCH --output=docs/benchmarks/v110-audit-20261107-03/job-%j.out
#SBATCH --error=docs/benchmarks/v110-audit-20261107-03/job-%j.err
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module load cuda/12.6
export PATH="$HOME/.local/bin:$PATH"
export UV_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/python"
export QUABLA_CUDA_TEST=1 QUABLA_CUDA_NCCL_TEST=1
export LD_LIBRARY_PATH="/home/junyi/.local/share/uv/python/cpython-3.13.11-linux-x86_64-gnu/lib:/home/junyi/nabla-p7/.deps/nccl/lib:${LD_LIBRARY_PATH:-}"
run=docs/benchmarks/v110-audit-20261107-03
sha256sum -c "$run/source.sha256" > "$run/source-before-$SLURM_JOB_ID.log"
nvidia-smi
.venv/bin/python - <<'PY'
import ctypes
import hashlib
import json
from pathlib import Path
import quabla
import quabla._quabla as native
root = Path.cwd()
assert Path(quabla.__file__).is_relative_to(root / "python"), quabla.__file__
assert Path(native.__file__).is_relative_to(root / "python"), native.__file__
old_root = Path("/home/junyi/quabla-v110-audit-20261107-01")
old = json.loads((old_root / "docs/benchmarks/v110-audit-20261107-01/source.json").read_text())
new = json.loads(Path("docs/benchmarks/v110-audit-20261107-03/source.json").read_text())
for path in new:
    if path.startswith("crates/") or path in ("Cargo.toml", "Cargo.lock"):
        assert new[path] == old[path], path
extension = Path(native.__file__)
assert hashlib.sha256(extension.read_bytes()).digest() == hashlib.sha256(
    (old_root / "python/quabla" / extension.name).read_bytes()).digest()
for name in ("libcuda.so.1", "libnvrtc.so.12", "libnccl.so"):
    ctypes.CDLL(name)
print("native source identity and binary reuse verified; Python source:", quabla.__file__)
PY
timeout 300 .venv/bin/python "$run/device_probes.py" --device cuda:0 > "$run/probes-cuda-$SLURM_JOB_ID.json"
timeout 240 .venv/bin/python examples/validate_data_parallel_cuda.py --output "$run/parity-$SLURM_JOB_ID.json" > "$run/parity-$SLURM_JOB_ID.log" 2>&1
bash tests/python/run_suites.sh "$PWD/.venv/bin/python" > "$run/python-$SLURM_JOB_ID.log" 2>&1
sha256sum -c "$run/source.sha256" > "$run/source-after-$SLURM_JOB_ID.log"
printf '{"status":"passed","job_id":"%s"}\n' "$SLURM_JOB_ID" > "$run/completion-$SLURM_JOB_ID.json"
cat "$run/probes-cuda-$SLURM_JOB_ID.json" "$run/completion-$SLURM_JOB_ID.json"
