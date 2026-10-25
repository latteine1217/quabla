#!/usr/bin/env bash
# Installs each CPython wheel in DIST into a fresh virtual environment and
# runs every Python test suite against it (run_suites.sh):
#
#     tests/python/check_wheels.sh dist
#
# The wheel builds call it after building one wheel per version in
# `versions`, each on PATH as `pythonX.Y`. The environments are left as
# `.venv-wheel-X.Y` for later checks. Every version runs even after a
# failure; the script exits non-zero when any version failed.
set -u

if [ "$#" -ne 1 ]; then
    echo "usage: $0 DIST" >&2
    exit 2
fi
dist=$1
here=$(cd "$(dirname "$0")" && pwd)
versions="3.10 3.11 3.12 3.13 3.14"

failed=()
for version in $versions; do
    echo "::group::Python $version"
    venv=".venv-wheel-$version"
    # The suites run from the source tree, so check that `import quabla`
    # resolves to the installed wheel and not to python/quabla.
    if ! { "python$version" -m venv "$venv" &&
        "$venv/bin/python" -m pip install "$dist"/*-cp"${version/./}"-*.whl 'numpy>=2,<3' &&
        "$venv/bin/python" -c 'import quabla, sys; assert quabla.__file__.startswith(sys.prefix), quabla.__file__' &&
        "$here/run_suites.sh" "$venv/bin/python"; }; then
        echo "::error::the cp${version/./} wheel failed its tests"
        failed+=("$version")
    fi
    echo "::endgroup::"
done

if [ "${#failed[@]}" -ne 0 ]; then
    echo "failed Python versions: ${failed[*]}" >&2
    exit 1
fi
