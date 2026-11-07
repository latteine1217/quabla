#!/bin/bash
# Build and test each supported CPython wheel in an isolated environment.
set -euo pipefail
run=docs/benchmarks/v110-audit-20261108-01
matrix=target/v110-python-matrix-mean
mkdir -p "$matrix"
if [ "$#" -eq 0 ]; then
    set -- 3.10 3.11 3.12 3.13 3.14
fi
for version in "$@"; do
    tag=${version/./}
    interpreter=$(uv python find "$version")
    stage="$matrix/py$tag"
    mkdir -p "$stage/wheels"
    printf 'python=%s interpreter=%s\n' "$version" "$interpreter"
    uvx --from 'maturin>=1.9.3,<2' maturin build --release --locked \
        --interpreter "$interpreter" --out "$stage/wheels" > "$run/build-py$tag.log" 2>&1
    uv venv --python "$interpreter" "$stage/venv"
    uv pip install --python "$stage/venv/bin/python" numpy "$stage"/wheels/*.whl
    env QUABLA_MLX_TEST=0 QUABLA_CUDA_TEST=0 QUABLA_CUDA_NCCL_TEST=0 \
        bash tests/python/run_suites.sh "$PWD/$stage/venv/bin/python" > "$run/python-py$tag.log" 2>&1
    printf 'python=%s passed\n' "$version"
done
