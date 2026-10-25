#!/usr/bin/env bash
# Runs every Python test suite (tests/python/test_*.py) with one interpreter:
#
#     tests/python/run_suites.sh .venv/bin/python
#
# CI, the wheel builds, and contributors run the same command, and a new
# suite is picked up without editing a list. Every suite runs even after a
# failure; the script exits non-zero when any suite failed and names them.
# The hardware gates (QUABLA_*_TEST) pass through from the environment.
set -u

if [ "$#" -ne 1 ]; then
    echo "usage: $0 PYTHON" >&2
    exit 2
fi
python=$1
here=$(cd "$(dirname "$0")" && pwd)

"$python" -c 'import sys, quabla; print("python", sys.version.split()[0], "quabla", quabla.__version__, "from", quabla.__file__)' || exit 1

failed=()
for suite in "$here"/test_*.py; do
    echo "=== $(basename "$suite") ==="
    "$python" "$suite" || failed+=("$(basename "$suite")")
done

if [ "${#failed[@]}" -ne 0 ]; then
    echo "failed suites: ${failed[*]}" >&2
    exit 1
fi
echo "all suites passed"
