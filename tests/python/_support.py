"""Shared harness for the script-style Python test suites.

Every `tests/python/test_*.py` runs as a plain script
(`python tests/python/test_x.py`). Python puts the script's directory first
on `sys.path`, so `import _support` finds this file from any working
directory.

Hardware gates. The GPU suites are opt in, one environment variable per
target, and a gate is on exactly when its variable is the string "1"; any
other value, "0" and the empty string included, leaves it off:

    QUABLA_MLX_TEST        MLX (Apple silicon, built with --features mlx)
    QUABLA_CUDA_TEST       CUDA (an NVIDIA GPU, built with --features cuda)
    QUABLA_CUDA_NCCL_TEST  NCCL data parallelism (two CUDA GPUs and libnccl)

A test names its targets as "mlx", "cuda" (or "cuda:N"), and "nccl".
`enabled(target)` answers whether a gate is on, `require(target)` skips the
test when it is off, and `devices()` lists the enabled GPU targets for a
parity loop and skips the test when there are none. A gate that is on for a
build without that target fails the test instead of passing it vacuously.

Reporting. `run(globals())` is every script's `__main__` block: it runs the
`test_*` functions in definition order and prints one line per test,

    PASS name
    PASS name (gate off: QUABLA_CUDA_TEST)   part of the test was gated off
    SKIP name: reason                        the test called skip or require
    FAIL name                                followed by the traceback

then a summary line, and exits with status 1 when any test failed.
"""

import os
import sys
import traceback

GATES = {
    "mlx": "QUABLA_MLX_TEST",
    "cuda": "QUABLA_CUDA_TEST",
    "nccl": "QUABLA_CUDA_NCCL_TEST",
}

# Gates found off while the current test runs; `run` reports them.
_gated_off = []


class Skip(Exception):
    """Raised by `skip`; `run` reports the test as SKIP with the reason."""


def skip(reason):
    raise Skip(reason)


def _gate(target):
    name = target.split(":", 1)[0]
    if name not in GATES:
        raise ValueError(f"unknown test target {target!r}; expected one of {sorted(GATES)}")
    return name, GATES[name]


def enabled(target):
    """Whether the gate of `target` ("mlx", "cuda", "cuda:N", "nccl") is on."""
    name, variable = _gate(target)
    if os.environ.get(variable) != "1":
        if variable not in _gated_off:
            _gated_off.append(variable)
        return False
    built = "cuda" if name == "nccl" else name
    import quabla

    assert built in quabla.devices(), f"{variable}=1 but this build has no {built} target"
    return True


def require(target):
    """Skip the calling test unless the gate of `target` is on."""
    if not enabled(target):
        skip(f"set {_gate(target)[1]}=1")


def devices(*targets):
    """The enabled targets among `targets` (default "mlx" and "cuda") in
    order; skips the calling test when none is enabled."""
    targets = targets or ("mlx", "cuda")
    chosen = [target for target in targets if enabled(target)]
    if not chosen:
        variables = dict.fromkeys(_gate(target)[1] for target in targets)
        skip("set " + " or ".join(f"{variable}=1" for variable in variables))
    return chosen


def raises(kind, function, *args, match=None, **kwargs):
    """Call `function` and return the `kind` exception it must raise; `match`
    must then be a substring of its message."""
    try:
        function(*args, **kwargs)
    except kind as error:
        assert match is None or match in str(error), str(error)
        return error
    raise AssertionError(f"expected {kind.__name__}")


def assert_close(actual, expected, tolerance=1e-12):
    """Elementwise `|actual - expected| <= tolerance * max(1, |expected|)`:
    absolute near zero and relative for large values. Arrays, nested lists,
    and scalars compare through `tolist`; nested lengths must agree."""
    actual = actual.tolist() if hasattr(actual, "tolist") else actual
    expected = expected.tolist() if hasattr(expected, "tolist") else expected
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), (actual, expected)
        for got, want in zip(actual, expected):
            assert_close(got, want, tolerance)
        return
    assert abs(actual - expected) <= tolerance * max(1.0, abs(expected)), (actual, expected)


def run(namespace, skip_reason=None):
    """Run the `test_*` functions of a script's `namespace` in definition
    order; with `skip_reason`, report each as skipped without running it."""
    script = os.path.basename(namespace.get("__file__", "?"))
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    for name, test in list(namespace.items()):
        if not (name.startswith("test_") and callable(test)):
            continue
        _gated_off.clear()
        try:
            if skip_reason is not None:
                skip(skip_reason)
            test()
        except Skip as reason:
            status, line = "SKIP", f"SKIP {name}: {reason}"
        except Exception:
            status, line = "FAIL", f"FAIL {name}\n{traceback.format_exc()}"
        else:
            note = f" (gate off: {', '.join(_gated_off)})" if _gated_off else ""
            status, line = "PASS", f"PASS {name}{note}"
        counts[status] += 1
        print(line, flush=True)
    print(
        f"{script}: {counts['PASS']} passed, {counts['SKIP']} skipped, "
        f"{counts['FAIL']} failed",
        flush=True,
    )
    if counts["FAIL"]:
        sys.exit(1)
