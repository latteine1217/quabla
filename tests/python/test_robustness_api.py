"""Reproducible boundary regressions; a native panic must fail the suite."""

import subprocess
import sys

import quabla as qb

from _support import raises, run


def test_high_rank_repr_does_not_crash_the_interpreter():
    # A separate interpreter detects native stack exhaustion as well as Python errors.
    code = """
import quabla as qb
rank = 50000
value = qb.Tensor([1] * rank, [1.5])
text = repr(value)
assert text.startswith('Tensor(' + '[' * rank)
assert text.count(']') == rank
assert '1.5' in text and 'dtype=float64' in text
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


def test_deep_nested_conversion_does_not_crash_the_interpreter():
    code = """
import quabla as qb
rank = 20000
value = 1.5
for _ in range(rank):
    value = [value]
array = qb.asarray(value)
assert array.shape == [1] * rank
result = array.tolist()
for _ in range(rank):
    assert type(result) is list and len(result) == 1
    result = result[0]
assert result == 1.5
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


def test_cyclic_input_and_nonadvancing_arange_terminate_with_value_errors():
    # Timeouts also detect regressions that hang rather than raising a native panic.
    code = """
import quabla as qb
cycle = []
cycle.append(cycle)
class Endless(list):
    def __iter__(self):
        import itertools
        return itertools.repeat(1.0)
for call in [lambda: qb.asarray(cycle), lambda: qb.arange(1e16, 1e16 + 2, 0.25),
             lambda: qb.arange(-1e16, -1e16 - 2, -0.25),
             lambda: qb.asarray(Endless([1.0])),
             lambda: qb.Tensor([1], Endless([1.0]))]:
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError('invalid input was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


def test_unaddressable_factories_raise_python_errors_without_allocation():
    extent = sys.maxsize
    for factory in (
        lambda: qb.zeros([extent]),
        lambda: qb.ones([extent]),
        lambda: qb.full([extent], 1.0),
        lambda: qb.linspace(0.0, 1.0, extent),
        lambda: qb.eye(extent, 1),
        lambda: qb.Tensor.split_key(1, extent),
        lambda: qb.Tensor.random_normal([extent], 1),
        lambda: qb.array([1.0]).broadcast_to([extent]),
        lambda: qb.arange(0.0, 1e308, 1.0),
    ):
        raises(ValueError, factory)
    # Overflow of stop-start alone must not reject a small representable range.
    maximum = sys.float_info.max
    assert qb.arange(-maximum, maximum, maximum).to_flat_list() == [-maximum, 0.0]


def test_derived_shapes_are_validated_before_compilation():
    extent = 1 << (sys.maxsize.bit_length() // 2 + 1)
    # The operands fit individually; their broadcast or matrix result does not.
    for function in (lambda a, b: a + b, lambda a, b: a @ b):
        raises(
            ValueError,
            qb.jit(function).lower,
            qb.ShapeDtype([extent, 1]),
            qb.ShapeDtype([1, extent]),
        )
    raises(
        ValueError,
        qb.jit(lambda x: qb.scan(lambda c, i: (c, c), x, length=sys.maxsize)).lower,
        qb.ShapeDtype([]),
    )


def test_sequence_conversion_rejects_overflow_before_entering_native_methods():
    class UnsizedSequence:
        def __getitem__(self, index):
            if index >= 2:
                raise IndexError(index)
            return index

    valid = qb.Tensor(range(1, 3), range(2))
    assert valid.shape == [1, 2] and valid.to_flat_list() == [0.0, 1.0]
    assert valid.reshape(range(1, 3)).shape == [1, 2]
    assert valid.transpose(None).shape == [2, 1]
    assert valid.gather(range(2), axis=1).to_flat_list() == [0.0, 1.0]
    assert qb.Tensor([2], UnsizedSequence()).to_flat_list() == [0.0, 1.0]
    point = qb.array([1.0])
    for call in (
        lambda: qb.Tensor(range(sys.maxsize), []),
        lambda: qb.Tensor([], range(sys.maxsize)),
        lambda: point.reshape(range(sys.maxsize)),
        lambda: point.transpose(range(sys.maxsize)),
        lambda: point.sum(axis=range(sys.maxsize)),
        lambda: point.gather(range(sys.maxsize)),
        lambda: qb.concat(range(sys.maxsize), axis=0),
        lambda: qb.stack(range(sys.maxsize)),
        lambda: qb.Compiler().trace(lambda x: x, [("x", range(sys.maxsize))]),
        lambda: qb.jit(lambda x: x.transpose(range(sys.maxsize)))(point),
    ):
        # A PyO3 PanicException inherits BaseException and must escape this check.
        raises((TypeError, ValueError, MemoryError), call)


if __name__ == "__main__":
    run(globals())
