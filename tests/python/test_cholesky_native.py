"""Staged native Cholesky contracts and derivative composition regressions."""

import math

import quabla as qb

from _support import run


def reference_cholesky(matrix):
    """The previous staged lower-triangle recurrence, including scalar rounding."""
    n = matrix.shape[0]
    rows = []
    for row in range(n):
        current = []
        for column in range(n):
            if column > row:
                current.append(qb.asarray(0.0, dtype=matrix.dtype))
                continue
            reduced = matrix[row, column]
            for inner in range(column):
                other = current[inner] if row == column else rows[column][inner]
                reduced = reduced - current[inner] * other
            value = reduced.sqrt() if row == column else reduced / rows[column][column]
            current.append(value)
        rows.append(current)
    return qb.concat(
        [
            qb.concat([value.reshape([1]) for value in row], 0).reshape([1, n])
            for row in rows
        ],
        0,
    )


def assert_close(actual, expected, tolerance=1e-12):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    for value, reference in zip(actual.to_flat_list(), expected.to_flat_list()):
        assert (
            value == reference
            or math.isnan(value)
            and math.isnan(reference)
            or math.isclose(value, reference, rel_tol=tolerance, abs_tol=tolerance)
        ), (value, reference)


def test_native_cholesky_primal_shapes_and_staged_validation():
    for dtype in (qb.float64, qb.float32):
        for values in (
            [[4.0, 2.0], [2.0, 5.0]],
            [[4.0, 99.0], [2.0, 5.0]],
            [[0.0]],
            [[-1.0]],
            [[math.nan]],
        ):
            matrix = qb.asarray(values, dtype=dtype)
            assert_close(
                qb.jit(qb.cholesky)(matrix), qb.jit(reference_cholesky)(matrix)
            )
        lowered = qb.jit(qb.cholesky).lower(
            qb.asarray([[4.0, 2.0], [2.0, 5.0]], dtype=dtype)
        )
        assert "cholesky(" in lowered.as_text()
        assert lowered.program.compile().node_count == 2


def test_native_cholesky_complete_derivative_composition():
    for dtype in (qb.float64, qb.float32):
        first = qb.asarray([[4.0, 2.0], [2.0, 5.0]], dtype=dtype)
        second = qb.asarray([[3.0, -19.0], [0.5, 2.0]], dtype=dtype)
        batch = qb.stack([first, second])

        def loss(matrix):
            return qb.cholesky(matrix).sum()

        def reference_loss(matrix):
            return reference_cholesky(matrix).sum()

        tolerance = 1e-12 if dtype == qb.float64 else 1e-5
        for transform, reference in (
            (qb.vmap(qb.cholesky), qb.vmap(reference_cholesky)),
            (qb.vmap(qb.grad(loss)), qb.vmap(qb.grad(reference_loss))),
        ):
            assert_close(transform(batch), reference(batch), tolerance)
            assert_close(qb.jit(transform)(batch), qb.jit(reference)(batch), tolerance)
        for transform, reference in (
            (qb.jacobian(qb.cholesky), qb.jacobian(reference_cholesky)),
            (qb.hessian(loss), qb.hessian(reference_loss)),
        ):
            assert_close(transform(first), reference(first), tolerance)
            assert_close(qb.jit(transform)(first), qb.jit(reference)(first), tolerance)
        direction = qb.asarray([[0.17, 0.29], [-0.13, 0.31]], dtype=dtype)
        _, actual = qb.jvp(qb.cholesky, (first,), (direction,))
        _, expected = qb.jvp(reference_cholesky, (first,), (direction,))
        assert_close(actual, expected, tolerance)


if __name__ == "__main__":
    run(globals())
