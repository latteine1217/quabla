"""Narrow buffer imports preserve F64 conversion before final dtype rounding."""

import array
import sys

import quabla as qb

try:
    import numpy as np
except ImportError:  # NumPy is optional for quabla; every test here needs it.
    np = None


def assert_f32_import(source):
    expected = np.asarray(source).astype(np.float64).astype(np.float32)
    result = qb.asarray(source, dtype=qb.float32)
    actual = result.numpy()
    assert result.dtype == qb.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    return actual


def test_integer_widths_and_layouts():
    for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        bounds = np.iinfo(dtype)
        source = np.array([bounds.min, 0, 1, bounds.max] * 6, dtype=dtype).reshape(4, 6)
        for view in (source, source.T, source[::-1, ::2], source[:, ::-1]):
            assert_f32_import(view)
        assert_f32_import(source[0, 0])
        assert qb.asarray(source).dtype == qb.float64


def test_wide_integer_double_rounding():
    # The midpoint's +/- 1 neighbors distinguish conversion through F64 from
    # direct integer-to-F32 conversion; both neighbors first round to the midpoint.
    midpoint = 2**62 + 2**38
    for dtype, values in (
        (
            np.int64,
            [
                -(2**63),
                -midpoint - 1,
                -midpoint + 1,
                2**53 + 1,
                midpoint - 1,
                midpoint,
                midpoint + 1,
                2**63 - 1,
            ],
        ),
        (
            np.uint64,
            [
                0,
                2**53 + 1,
                midpoint - 1,
                midpoint,
                midpoint + 1,
                2**63 + 1,
                2**64 - 1,
                1,
            ],
        ),
    ):
        source = np.array(values, dtype=dtype).reshape(2, 4)
        for view in (source, source.T, source[::-1, ::-1]):
            assert_f32_import(view)
            np.testing.assert_array_equal(
                qb.asarray(view, dtype=qb.bool_).numpy(), view.astype(np.float64) != 0
            )
        assert_f32_import(source[0, 0])
    assert_f32_import(array.array("q", [midpoint - 1, midpoint + 1]))


def test_float64_narrowing_and_foreign_ownership():
    source = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-300, 1e30, 1.0])
    for view in (source, source.reshape(2, 4).T, source[::-2]):
        assert_f32_import(view)
    original = np.arange(8, dtype=np.int64)
    actual = assert_f32_import(original)
    original[:] = -7
    np.testing.assert_array_equal(actual, np.arange(8, dtype=np.float32))
    assert not actual.flags.writeable


def test_endian_rejection_and_bool_import():
    opposite = ">" if sys.byteorder == "little" else "<"
    for dtype in ("i2", "i4", "i8", "u8", "f8"):
        source = np.array([0, 1, 2, 3], dtype=opposite + dtype)
        try:
            qb.asarray(source, dtype=qb.float32)
        except TypeError as error:
            assert "native byte order" in str(error)
        else:
            raise AssertionError("non-native buffer accepted")
    source = np.array([[True, False], [False, True]], dtype=np.bool_)
    assert_f32_import(source.T)


if __name__ == "__main__":
    if np is None:
        print("skipped test_buffer_narrow_import: numpy is not installed")
    else:
        for name, test in list(globals().items()):
            if name.startswith("test_") and callable(test):
                test()
