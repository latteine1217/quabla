"""quabla.linalg: batched solves, determinants, the symmetric eigensolver,
QR, SVD, and least squares.

Values are compared with NumPy, and derivatives with central differences of
NumPy references. Tolerances scale with the condition of each test matrix:
every matrix here has a condition number below ~50 and, for `eigh`,
eigenvalue gaps above ~0.3, so a float64 solve is accurate to ~1e-13 and a
central difference with step 1e-6 to ~1e-8.
"""

import os
import warnings

try:
    import numpy as np
except ImportError:  # NumPy is an optional dependency; the suite needs it.
    np = None

import quabla as qb


def assert_close(actual, expected, tolerance):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    error = np.max(np.abs(actual - expected), initial=0.0)
    scale = max(1.0, np.max(np.abs(expected), initial=0.0))
    assert error <= tolerance * scale, (error, tolerance * scale)


def raises(kind, function, *args, match=None, **kwargs):
    try:
        function(*args, **kwargs)
    except kind as error:
        assert match is None or match in str(error), str(error)
        return error
    raise AssertionError(f"expected {kind.__name__}")


def well_conditioned(rng, shape):
    """Random matrices with a dominant diagonal: condition number below ~10."""
    n = shape[-1]
    return rng.normal(size=shape) + 3.0 * np.sqrt(n) * np.eye(n)


def symmetric_with_gaps(rng, batch, n):
    """Symmetric matrices with eigenvalues 1, 2, ..., n (gaps of one)."""
    q, _ = np.linalg.qr(rng.normal(size=batch + (n, n)))
    values = np.arange(1.0, n + 1.0)
    return (q * values[..., None, :]) @ np.swapaxes(q, -1, -2)


def central_difference(function, x, step=1e-6):
    """d function / dx for a scalar-valued NumPy `function` of `x`."""
    gradient = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def test_solve_is_batched_and_broadcasts_like_numpy():
    rng = np.random.default_rng(0)
    a = well_conditioned(rng, (2, 3, 4, 4))
    b = rng.normal(size=(2, 3, 4, 2))
    for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-5)):
        qa, qbb = qb.asarray(a, dtype=dtype), qb.asarray(b, dtype=dtype)
        expected = np.linalg.solve(a, b)
        assert_close(qb.linalg.solve(qa, qbb), expected, tolerance)
        assert_close(qb.jit(qb.linalg.solve)(qa, qbb), expected, tolerance)
        assert qb.linalg.solve(qa, qbb).dtype == dtype
    # `quabla.solve` is `quabla.linalg.solve`; an unbatched matrix broadcasts.
    assert_close(qb.solve(a[0, 0], b), np.linalg.solve(a[0, 0], b), 1e-13)
    assert_close(qb.solve(a, b[0, 0]), np.linalg.solve(a, b[0, 0]), 1e-13)
    # A rank-one right-hand side is a vector (NumPy 2), also under batching.
    vector = rng.normal(size=4)
    assert_close(qb.linalg.solve(a[0, 0], vector), np.linalg.solve(a[0, 0], vector), 1e-13)
    assert_close(
        qb.linalg.solve(a, vector), np.linalg.solve(a, vector[:, None])[..., 0], 1e-13
    )
    raises(ValueError, qb.linalg.solve, a[..., :3], b, match="square")
    raises(ValueError, qb.linalg.solve, a, b[..., :3, :], match="agree on rows")
    raises(ValueError, qb.linalg.solve, a[:, :2], b[:, :3], match="broadcast")
    singular = np.stack([np.eye(2), [[1.0, 2.0], [2.0, 4.0]]])
    raises(ValueError, qb.linalg.solve, singular, np.ones((2, 2, 1)), match="non-singular")


def test_solve_derivatives_match_finite_differences():
    rng = np.random.default_rng(1)
    a = well_conditioned(rng, (3, 3, 3))
    b = rng.normal(size=(3, 3, 2))
    weights = rng.normal(size=b.shape)
    qweights = qb.asarray(weights)

    def loss(a, b):
        return (qb.linalg.solve(a, b) ** 2 * qweights).sum()

    def reference(a, b):
        return (np.linalg.solve(a, b) ** 2 * weights).sum()

    grad_a, grad_b = qb.grad(loss, argnums=(0, 1))(a, b)
    assert_close(grad_a, central_difference(lambda x: reference(x, b), a), 1e-8)
    assert_close(grad_b, central_difference(lambda x: reference(a, x), b), 1e-8)
    jitted = qb.jit(qb.grad(loss, argnums=(0, 1)))(a, b)
    assert_close(jitted[0], grad_a, 1e-13)
    # Forward mode: d solve(a, b)[direction] = solve(a, db - da x).
    direction = rng.normal(size=a.shape)
    _, tangent = qb.jvp(lambda m: qb.linalg.solve(m, b), (a,), (direction,))
    x = np.linalg.solve(a, b)
    assert_close(tangent, np.linalg.solve(a, -direction @ x), 1e-12)
    # Second order: the Hessian of a batched solve against a reference
    # built from per-matrix solves.
    hessian = qb.hessian(lambda m: (qb.linalg.solve(m, b[:1]) ** 2).sum())(a[:1])
    per_matrix = qb.hessian(lambda m: (qb.linalg.solve(m, b[0]) ** 2).sum())(a[0])
    assert_close(np.asarray(hessian)[0, :, :, 0], per_matrix, 1e-12)


def test_vmap_batches_solve():
    rng = np.random.default_rng(2)
    a = well_conditioned(rng, (5, 3, 3))
    b = rng.normal(size=(5, 3, 2))
    expected = np.linalg.solve(a, b)
    assert_close(qb.vmap(qb.linalg.solve)(a, b), expected, 1e-13)
    assert_close(qb.vmap(qb.linalg.solve, in_axes=(None, 0))(a[0], b), np.linalg.solve(a[0], b), 1e-13)
    assert_close(qb.vmap(qb.linalg.solve, in_axes=(0, None))(a, b[0]), np.linalg.solve(a, b[0]), 1e-13)
    assert_close(qb.jit(qb.vmap(qb.linalg.solve))(a, b), expected, 1e-13)
    # Nested vmap over a [2, 5] grid of systems and vmap of a gradient.
    grid_a, grid_b = np.stack([a, a + 0.5 * np.eye(3)]), np.stack([b, -b])
    assert_close(qb.vmap(qb.vmap(qb.linalg.solve))(grid_a, grid_b), np.linalg.solve(grid_a, grid_b), 1e-13)

    def loss(m, rhs):
        return (qb.linalg.solve(m, rhs) ** 2).sum()

    per_example = qb.vmap(qb.grad(loss))(a, b)
    for index in range(5):
        assert_close(per_example[index], qb.grad(loss)(a[index], b[index]), 1e-13)


def test_triangular_and_cholesky_solves():
    rng = np.random.default_rng(3)
    lower = np.tril(well_conditioned(rng, (2, 4, 4)))
    b = rng.normal(size=(2, 4, 3))
    noise = rng.normal(size=lower.shape)
    # The other triangle is never read.
    a = lower + np.triu(noise, 1)
    assert_close(qb.linalg.solve_triangular(a, b, lower=True), np.linalg.solve(lower, b), 1e-13)
    upper = np.swapaxes(lower, -1, -2)
    assert_close(qb.linalg.solve_triangular(upper + np.tril(noise, -1), b), np.linalg.solve(upper, b), 1e-13)
    assert_close(
        qb.linalg.solve_triangular(a, b, trans=1, lower=True), np.linalg.solve(upper, b), 1e-13
    )
    assert_close(
        qb.linalg.solve_triangular(a, b, trans="T", lower=True), np.linalg.solve(upper, b), 1e-13
    )
    raises(ValueError, qb.linalg.solve_triangular, a, b, trans=2)

    spd = lower @ upper
    factor = qb.linalg.cholesky(spd)
    assert_close(factor, np.linalg.cholesky(spd), 1e-12)
    assert_close(qb.jit(qb.linalg.cholesky)(spd), np.linalg.cholesky(spd), 1e-12)
    assert_close(qb.linalg.cho_solve(factor, b), np.linalg.solve(spd, b), 1e-12)
    upper_factor = np.swapaxes(np.asarray(factor), -1, -2)
    assert_close(qb.linalg.cho_solve(upper_factor, b, lower=False), np.linalg.solve(spd, b), 1e-12)

    def loss(m):
        return (qb.linalg.cho_solve(qb.linalg.cholesky(m), b) ** 2).sum()

    def reference(m):
        return (np.linalg.solve((m + np.swapaxes(m, -1, -2)) / 2, b) ** 2).sum()

    # Cholesky reads one triangle, so compare the symmetrized gradient.
    gradient = np.asarray(qb.grad(loss)(spd))
    gradient = (gradient + np.swapaxes(gradient, -1, -2)) / 2
    assert_close(gradient, central_difference(reference, spd), 1e-7)


def test_slogdet_det_and_inv():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(3, 4, 4)) + 2.0 * np.eye(4)
    for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-5)):
        sign, logabsdet = qb.linalg.slogdet(qb.asarray(a, dtype=dtype))
        expected_sign, expected_log = np.linalg.slogdet(a)
        assert_close(sign, expected_sign, 0.0)
        assert_close(logabsdet, expected_log, tolerance)
        assert logabsdet.dtype == dtype
        assert_close(qb.linalg.det(qb.asarray(a, dtype=dtype)), np.linalg.det(a), tolerance)
        assert_close(qb.linalg.inv(qb.asarray(a, dtype=dtype)), np.linalg.inv(a), 10 * tolerance)
    result = qb.jit(qb.linalg.slogdet)(a)
    assert isinstance(result, qb.linalg.SlogdetResult)
    assert_close(result.logabsdet, np.linalg.slogdet(a)[1], 1e-13)
    # The log-magnitude does not overflow where the determinant does.
    huge = np.diag([1e200, 1e200, 1e-10])
    assert_close(qb.linalg.slogdet(huge).logabsdet, np.linalg.slogdet(huge)[1], 1e-15)
    # An odd permutation flips the sign.
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert qb.linalg.slogdet(swap).sign.item() == -1.0
    # Exactly singular: sign 0, log|det| -inf, det 0; non-finite: NaN.
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    sign, logabsdet = qb.linalg.slogdet(singular)
    assert sign.item() == 0.0 and logabsdet.item() == -np.inf
    assert qb.linalg.det(singular).item() == 0.0
    assert np.isnan(qb.linalg.slogdet(np.array([[np.nan, 0.0], [0.0, 1.0]])).logabsdet.item())
    raises(ValueError, qb.linalg.slogdet, np.ones((2, 3)), match="square")


def test_slogdet_derivatives():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 3, 3)) + 2.0 * np.eye(3)
    inverse = np.linalg.inv(a)
    # d log|det A| / dA = A^-T, and the sign has a zero derivative.
    assert_close(qb.grad(lambda m: qb.linalg.slogdet(m).logabsdet.sum())(a), np.swapaxes(inverse, -1, -2), 1e-13)
    assert_close(qb.grad(lambda m: qb.linalg.slogdet(m).sign.sum())(a), np.zeros_like(a), 0.0)
    direction = rng.normal(size=a.shape)
    _, tangent = qb.jvp(lambda m: qb.linalg.slogdet(m).logabsdet, (a,), (direction,))
    assert_close(tangent, np.trace(inverse @ direction, axis1=-2, axis2=-1), 1e-13)
    # d det / dA = det A^-T.
    assert_close(
        qb.grad(lambda m: qb.linalg.det(m).sum())(a),
        np.linalg.det(a)[:, None, None] * np.swapaxes(inverse, -1, -2),
        1e-12,
    )
    # Hessian of log|det A|: d2 / dA_ij dA_kl = -(A^-1)_li (A^-1)_jk.
    single, single_inverse = a[0], inverse[0]
    hessian = qb.hessian(lambda m: qb.linalg.slogdet(m).logabsdet)(single)
    expected = -np.einsum("li,jk->ijkl", single_inverse, single_inverse)
    assert_close(hessian, expected, 1e-12)
    # Reverse over forward: grad of the directional derivative is H . direction.
    single_direction = qb.asarray(direction[0])
    reverse_over_forward = qb.grad(
        lambda m: qb.jvp(lambda x: qb.linalg.slogdet(x).logabsdet, (m,), (single_direction,))[1]
    )(single)
    assert_close(reverse_over_forward, np.einsum("ijkl,kl->ij", expected, direction[0]), 1e-12)
    # The inverse differentiates through solve: d inv(A) = -A^-1 dA A^-1.
    _, tangent = qb.jvp(qb.linalg.inv, (a,), (direction,))
    assert_close(tangent, -inverse @ direction @ inverse, 1e-12)


def test_eigh_values_vectors_and_conventions():
    rng = np.random.default_rng(6)
    for n in (1, 2, 5, 8):
        a = rng.normal(size=(3, n, n))
        a = a + np.swapaxes(a, -1, -2)
        expected_values, expected_vectors = np.linalg.eigh(a)
        for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-6)):
            values, vectors = qb.linalg.eigh(qb.asarray(a, dtype=dtype))
            assert values.dtype == dtype and vectors.dtype == dtype
            values, vectors = np.asarray(values, np.float64), np.asarray(vectors, np.float64)
            assert_close(values, expected_values, tolerance)
            assert np.all(np.diff(values, axis=-1) >= 0.0)
            assert_close(np.swapaxes(vectors, -1, -2) @ vectors, np.broadcast_to(np.eye(n), a.shape), 10 * tolerance)
            assert_close((vectors * values[..., None, :]) @ np.swapaxes(vectors, -1, -2), a, 10 * tolerance)
            # Distinct eigenvalues: the vectors agree with NumPy up to sign,
            # and the sign makes the largest-magnitude component positive.
            aligned = np.sign(np.sum(vectors * expected_vectors, axis=-2, keepdims=True))
            assert_close(vectors, expected_vectors * aligned, 1e3 * tolerance)
            largest = np.take_along_axis(vectors, np.argmax(np.abs(vectors), axis=-2)[..., None, :], axis=-2)
            assert np.all(largest > 0.0)
    a = rng.normal(size=(4, 4))
    symmetric = (a + a.T) / 2
    # Only the symmetric part is decomposed.
    assert_close(qb.linalg.eigh(a).eigenvalues, np.linalg.eigvalsh(symmetric), 1e-13)
    result = qb.jit(qb.linalg.eigh)(symmetric)
    assert isinstance(result, qb.linalg.EighResult)
    assert_close(result.eigenvalues, np.linalg.eigvalsh(symmetric), 1e-13)
    # Clustered and repeated eigenvalues keep a small residual.
    clustered = symmetric_with_gaps(rng, (), 6)
    clustered = clustered + 1e-9 * np.diag(np.arange(6.0)) + np.eye(6) * 0.0
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    repeated = (q * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])) @ q.T
    for matrix in (clustered, repeated, np.zeros((3, 3)), np.diag([1e150, -1e-150, 1.0])):
        values, vectors = (np.asarray(part) for part in qb.linalg.eigh(matrix))
        norm = max(np.linalg.norm(matrix), np.finfo(float).tiny)
        assert np.linalg.norm(matrix @ vectors - vectors * values) <= 1e-14 * norm
        assert_close(vectors.T @ vectors, np.eye(len(matrix)), 1e-14)
    assert np.all(np.isnan(np.asarray(qb.linalg.eigh(np.array([[np.inf, 0.0], [0.0, 1.0]])).eigenvalues)))


def test_eigh_derivatives_match_finite_differences():
    rng = np.random.default_rng(7)
    a = symmetric_with_gaps(rng, (2,), 4)
    weights = rng.normal(size=(2, 4))
    vector_weights = rng.normal(size=(2, 4, 4))

    def reference_eigh(m):
        values, vectors = np.linalg.eigh((m + np.swapaxes(m, -1, -2)) / 2)
        # NumPy's sign is arbitrary; apply the quabla convention.
        largest = np.take_along_axis(vectors, np.argmax(np.abs(vectors), axis=-2)[..., None, :], axis=-2)
        return values, vectors * np.sign(largest)

    qweights, qvector_weights = qb.asarray(weights), qb.asarray(vector_weights)

    def value_loss(m):
        return (qb.linalg.eigh(m).eigenvalues * qweights).sum()

    def vector_loss(m):
        return (qb.linalg.eigh(m).eigenvectors ** 3 * qvector_weights).sum()

    expected_values = central_difference(lambda m: (reference_eigh(m)[0] * weights).sum(), a)
    expected_vectors = central_difference(lambda m: (reference_eigh(m)[1] ** 3 * vector_weights).sum(), a)
    assert_close(qb.grad(value_loss)(a), expected_values, 1e-8)
    assert_close(qb.grad(vector_loss)(a), expected_vectors, 1e-7)
    assert_close(qb.jit(qb.grad(vector_loss))(a), expected_vectors, 1e-7)
    # The eigenvalue gradient is the symmetric V diag(w) V^T.
    vectors = np.asarray(qb.linalg.eigh(a).eigenvectors)
    assert_close(qb.grad(value_loss)(a), (vectors * weights[..., None, :]) @ np.swapaxes(vectors, -1, -2), 1e-13)
    # Forward mode against differences of the outputs themselves.
    direction = rng.normal(size=a.shape)
    direction = direction + np.swapaxes(direction, -1, -2)
    _, (value_tangent, vector_tangent) = qb.jvp(lambda m: tuple(qb.linalg.eigh(m)), (a,), (direction,))
    step = 1e-6
    plus, minus = reference_eigh(a + step * direction), reference_eigh(a - step * direction)
    assert_close(value_tangent, (plus[0] - minus[0]) / (2 * step), 1e-8)
    assert_close(vector_tangent, (plus[1] - minus[1]) / (2 * step), 1e-8)
    # Second order: the Hessian of the vector loss against differences of
    # the reverse-mode gradient.
    single = a[0]
    single_weights = qb.asarray(vector_weights[0])
    hessian = np.asarray(qb.hessian(lambda m: (qb.linalg.eigh(m).eigenvectors ** 3 * single_weights).sum())(single))
    gradient = qb.jit(qb.grad(lambda m: (qb.linalg.eigh(m).eigenvectors ** 3 * single_weights).sum()))
    for index in [(0, 1), (2, 3), (1, 1)]:
        plus, minus = single.copy(), single.copy()
        plus[index] += step
        minus[index] -= step
        column = (np.asarray(gradient(plus)) - np.asarray(gradient(minus))) / (2 * step)
        assert_close(hessian[(slice(None), slice(None)) + index], column, 1e-6)
    # vmap of eigh and of its gradient.
    assert_close(qb.vmap(lambda m: qb.linalg.eigh(m).eigenvalues)(a), np.linalg.eigvalsh(a), 1e-13)
    per_example = qb.vmap(qb.grad(lambda m, w: (qb.linalg.eigh(m).eigenvalues * w).sum()))(a, weights)
    assert_close(per_example, qb.grad(value_loss)(a), 1e-13)
    # A repeated eigenvalue makes the eigenvector derivative non-finite (as in
    # JAX), while the eigenvalue derivative stays defined.
    repeated = np.diag([1.0, 1.0, 2.0])
    assert not np.all(np.isfinite(np.asarray(qb.grad(lambda m: qb.linalg.eigh(m).eigenvectors.sum())(repeated))))
    assert_close(qb.grad(lambda m: qb.linalg.eigh(m).eigenvalues.sum())(repeated), np.eye(3), 1e-15)


def signed_qr(a):
    """NumPy's reduced QR with the diagonal of R made non-negative."""
    q, r = np.linalg.qr(a)
    signs = np.where(np.diagonal(r, axis1=-2, axis2=-1) < 0, -1.0, 1.0)
    return q * signs[..., None, :], r * signs[..., :, None]


def signed_svd(a):
    """NumPy's reduced SVD with each column of U signed so its
    largest-magnitude component is positive (quabla's convention)."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    largest = np.take_along_axis(u, np.argmax(np.abs(u), axis=-2)[..., None, :], axis=-2)
    signs = np.where(largest < 0, -1.0, 1.0)
    return u * signs, s, vh * np.swapaxes(signs, -1, -2)


def transposed(x):
    return np.swapaxes(x, -1, -2)


def directional_difference(function, x, direction, step=1e-6):
    """The central difference of a NumPy `function` of `x` along `direction`."""
    return (function(x + step * direction) - function(x - step * direction)) / (2 * step)


def test_qr_matches_numpy_and_is_orthonormal():
    rng = np.random.default_rng(10)
    for shape in ((5, 3), (4, 4), (3, 5), (2, 6, 4), (7, 1), (1, 7)):
        a = rng.normal(size=shape)
        m, n = shape[-2:]
        k = min(m, n)
        expected_q, expected_r = signed_qr(a)
        for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 5e-6)):
            q, r = qb.linalg.qr(qb.asarray(a, dtype=dtype))
            assert q.dtype == dtype and r.dtype == dtype
            q, r = np.asarray(q, dtype=np.float64), np.asarray(r, dtype=np.float64)
            assert_close(q @ r, a, tolerance)
            assert_close(transposed(q) @ q, np.broadcast_to(np.eye(k), shape[:-2] + (k, k)), tolerance)
            assert np.all(np.tril(r, -1) == 0.0)
            assert np.all(np.diagonal(r, axis1=-2, axis2=-1) >= 0.0)
            # Full column rank makes the factorization unique.
            assert_close(r, expected_r, 10 * tolerance)
            assert_close(q, expected_q, 10 * tolerance)
        q, r = qb.linalg.qr(a, mode="complete")
        assert tuple(q.shape) == shape[:-2] + (m, m) and tuple(r.shape) == shape
        q, r = np.asarray(q), np.asarray(r)
        assert_close(q @ r, a, 1e-13)
        assert_close(transposed(q) @ q, np.broadcast_to(np.eye(m), shape[:-2] + (m, m)), 1e-13)
        assert_close(qb.linalg.qr(a, mode="r"), expected_r, 1e-12)
        jitted = qb.jit(lambda x: tuple(qb.linalg.qr(x)))(a)
        assert_close(jitted[0], expected_q, 1e-12)
        assert_close(jitted[1], expected_r, 1e-12)
    # A rank-deficient matrix still gets an orthonormal Q and A = Q R.
    deficient = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 4))
    q, r = (np.asarray(t) for t in qb.linalg.qr(deficient))
    assert_close(q @ r, deficient, 1e-13)
    assert_close(q.T @ q, np.eye(4), 1e-13)
    assert all(np.isnan(np.asarray(t)).all() for t in qb.linalg.qr(np.array([[np.nan, 1.0], [0.0, 1.0]])))
    raises(ValueError, qb.linalg.qr, np.ones(3), match="stack of matrices")
    raises(ValueError, qb.linalg.qr, np.ones((2, 2)), mode="full", match="mode")


def test_qr_derivatives_match_central_differences():
    rng = np.random.default_rng(11)
    for shape in ((5, 3), (3, 3), (3, 5), (2, 4, 3)):
        a = rng.normal(size=shape)
        direction = rng.normal(size=shape)
        for index, name in ((0, "Q"), (1, "R")):
            weights = rng.normal(size=signed_qr(a)[index].shape)

            def reference(x, index=index, weights=weights):
                return np.sum(signed_qr(x)[index] * weights)

            gradient = qb.grad(lambda x, i=index, w=weights: (qb.linalg.qr(x)[i] * qb.asarray(w)).sum())(a)
            assert_close(gradient, central_difference(reference, a), 1e-7)
            _, tangent = qb.jvp(lambda x, i=index: qb.linalg.qr(x)[i], (a,), (direction,))
            expected = directional_difference(lambda x, i=index: signed_qr(x)[i], a, direction)
            assert_close(tangent, expected, 1e-7)
    # Second derivatives: the Hessian is the derivative of the gradient.
    a = rng.normal(size=(4, 3))

    def gradient_of(x):
        return np.asarray(qb.grad(lambda t: (qb.linalg.qr(t).R ** 3).sum())(x))

    hessian = np.asarray(qb.hessian(lambda t: (qb.linalg.qr(t).R ** 3).sum())(a))
    direction = rng.normal(size=a.shape)
    assert_close(
        np.tensordot(hessian, direction, axes=2),
        directional_difference(gradient_of, a, direction),
        1e-7,
    )
    # The complement columns of a complete tall Q are not unique.
    raises(Exception, qb.grad(lambda x: qb.linalg.qr(x, mode="complete").Q.sum()), a, match="not differentiable")
    assert_close(
        qb.grad(lambda x: qb.linalg.qr(x, mode="complete").R.sum())(a),
        qb.grad(lambda x: qb.linalg.qr(x).R.sum())(a),
        1e-15,
    )


def test_svd_matches_numpy_and_is_orthonormal():
    rng = np.random.default_rng(12)
    for shape in ((5, 3), (4, 4), (3, 5), (2, 6, 4), (7, 1), (1, 7)):
        a = rng.normal(size=shape)
        m, n = shape[-2:]
        k = min(m, n)
        expected_u, expected_s, expected_vh = signed_svd(a)
        for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 5e-6)):
            u, s, vh = qb.linalg.svd(qb.asarray(a, dtype=dtype))
            assert u.dtype == s.dtype == vh.dtype == dtype
            u, s, vh = (np.asarray(t, dtype=np.float64) for t in (u, s, vh))
            # Relative accuracy of every singular value, the smallest included.
            assert np.max(np.abs(s - expected_s) / expected_s) <= 10 * tolerance
            assert np.all(np.diff(s, axis=-1) <= 0)
            assert_close((u * s[..., None, :]) @ vh, a, tolerance)
            eye = np.broadcast_to(np.eye(k), shape[:-2] + (k, k))
            assert_close(transposed(u) @ u, eye, tolerance)
            assert_close(vh @ transposed(vh), eye, tolerance)
            # Distinct singular values make the signed vectors unique.
            assert_close(u, expected_u, 100 * tolerance)
            assert_close(vh, expected_vh, 100 * tolerance)
        assert_close(qb.linalg.svd(a, compute_uv=False), expected_s, 1e-13)
        u, s, vh = qb.linalg.svd(a, full_matrices=True)
        assert tuple(u.shape) == shape[:-2] + (m, m) and tuple(vh.shape) == shape[:-2] + (n, n)
        u, vh = np.asarray(u), np.asarray(vh)
        assert_close(transposed(u) @ u, np.broadcast_to(np.eye(m), shape[:-2] + (m, m)), 1e-13)
        assert_close(vh @ transposed(vh), np.broadcast_to(np.eye(n), shape[:-2] + (n, n)), 1e-13)
        assert_close((u[..., :k] * np.asarray(s)[..., None, :]) @ vh[..., :k, :], a, 1e-13)
        assert_close(qb.jit(lambda x: qb.linalg.svd(x, compute_uv=False))(a), expected_s, 1e-13)
    # A matrix with widely spread singular values: one-sided Jacobi keeps
    # each one to high relative accuracy when the columns are scaled.
    scales = np.logspace(0, -9, 5)
    graded = rng.normal(size=(7, 5)) * scales
    exact = np.linalg.svd(graded / scales, compute_uv=False)
    values = np.asarray(qb.linalg.svd(graded, compute_uv=False))
    assert values[-1] < 1e-8 * values[0] and np.all(values > 0)
    np.testing.assert_allclose(np.prod(values), np.prod(exact) * np.prod(scales), rtol=1e-12)
    # Rank deficiency: zero singular values and a completed orthonormal U.
    deficient = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 4))
    u, s, vh = (np.asarray(t) for t in qb.linalg.svd(deficient))
    assert np.all(s[2:] < 1e-14 * s[0])
    assert_close((u * s) @ vh, deficient, 1e-13)
    assert_close(u.T @ u, np.eye(4), 1e-13)
    u, s, vh = (np.asarray(t) for t in qb.linalg.svd(np.zeros((3, 2))))
    assert np.all(s == 0.0)
    assert_close(u.T @ u, np.eye(2), 0.0)
    assert all(np.isnan(np.asarray(t)).all() for t in qb.linalg.svd(np.array([[np.inf, 1.0]])))


def test_svd_derivatives_match_central_differences():
    rng = np.random.default_rng(13)
    for shape in ((5, 3), (3, 3), (3, 5), (2, 4, 3)):
        a = rng.normal(size=shape)
        direction = rng.normal(size=shape)
        for index in range(3):
            weights = rng.normal(size=signed_svd(a)[index].shape)

            def reference(x, index=index, weights=weights):
                return np.sum(signed_svd(x)[index] * weights)

            gradient = qb.grad(lambda x, i=index, w=weights: (qb.linalg.svd(x)[i] * qb.asarray(w)).sum())(a)
            assert_close(gradient, central_difference(reference, a), 1e-7)
            _, tangent = qb.jvp(lambda x, i=index: qb.linalg.svd(x)[i], (a,), (direction,))
            expected = directional_difference(lambda x, i=index: signed_svd(x)[i], a, direction)
            assert_close(tangent, expected, 1e-7)
    a = rng.normal(size=(4, 3))
    # The singular values alone: the gradient of their sum is U Vh.
    u, _, vh = signed_svd(a)
    assert_close(qb.grad(lambda x: qb.linalg.svd(x, compute_uv=False).sum())(a), u @ vh, 1e-13)

    def gradient_of(x):
        return np.asarray(qb.grad(lambda t: (qb.linalg.svd(t, compute_uv=False) ** 3).sum())(x))

    hessian = np.asarray(qb.hessian(lambda t: (qb.linalg.svd(t, compute_uv=False) ** 3).sum())(a))
    direction = rng.normal(size=a.shape)
    assert_close(
        np.tensordot(hessian, direction, axes=2),
        directional_difference(gradient_of, a, direction),
        1e-7,
    )
    raises(Exception, qb.grad(lambda x: qb.linalg.svd(x, full_matrices=True).U.sum()), a, match="not differentiable")
    square = rng.normal(size=(3, 3))
    assert_close(
        qb.grad(lambda x: qb.linalg.svd(x, full_matrices=True).U.sum())(square),
        qb.grad(lambda x: qb.linalg.svd(x).U.sum())(square),
        1e-15,
    )


def test_lstsq_matches_numpy():
    rng = np.random.default_rng(14)
    a = rng.normal(size=(8, 3))
    for b in (rng.normal(size=8), rng.normal(size=(8, 2))):
        x, residuals = qb.linalg.lstsq(a, b, return_residuals=True)
        expected, expected_residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
        assert_close(x, expected, 1e-13)
        assert_close(residuals, expected_residuals.reshape(np.shape(residuals)), 1e-12)
        assert_close(qb.linalg.lstsq(qb.asarray(a, dtype=qb.float32), qb.asarray(b, dtype=qb.float32)), expected, 1e-5)
    # Leading batch axes broadcast; a wide matrix gets the minimum-norm solution.
    batched = rng.normal(size=(2, 7, 4))
    b = rng.normal(size=(7, 3))
    expected = np.stack([np.linalg.lstsq(matrix, b, rcond=None)[0] for matrix in batched])
    assert_close(qb.linalg.lstsq(batched, b), expected, 1e-13)
    assert_close(qb.jit(qb.linalg.lstsq)(batched, b), expected, 1e-13)
    wide = rng.normal(size=(3, 6))
    rhs = rng.normal(size=3)
    assert_close(qb.linalg.lstsq(wide, rhs), np.linalg.lstsq(wide, rhs, rcond=None)[0], 1e-13)
    for matrix, vector in ((a, rng.normal(size=8)), (wide, rhs)):
        gradient = qb.grad(lambda m, v=vector: (qb.linalg.lstsq(m, v) ** 2).sum())(matrix)
        expected = central_difference(
            lambda m, v=vector: np.sum(np.linalg.lstsq(m, v, rcond=None)[0] ** 2), matrix
        )
        assert_close(gradient, expected, 1e-7)
    raises(ValueError, qb.linalg.lstsq, a, rng.normal(size=7), match="agree on rows")

def _gram(x):
    """`x^T x` over the last two axes of a rank-3 traced array."""
    return x.transpose([0, 2, 1]) @ x


def test_optional_device_parity():
    rng = np.random.default_rng(8)
    a = well_conditioned(rng, (2, 3, 3)).astype(np.float32)
    b = rng.normal(size=(2, 3, 2)).astype(np.float32)
    symmetric = symmetric_with_gaps(rng, (2,), 3).astype(np.float32)
    # Singular values 3, 2, 1 (and 3, 2 for the wide matrix): well separated.
    left, _ = np.linalg.qr(rng.normal(size=(2, 5, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(2, 3, 3)))
    tall = ((left * np.array([3.0, 2.0, 1.0])) @ transposed(right)).astype(np.float32)
    wide = transposed(tall[..., :2, :]).copy()
    rhs = rng.normal(size=(2, 5, 2)).astype(np.float32)
    functions = (
        (lambda m: tuple(qb.linalg.qr(m)), (tall,)),
        # The complement columns are not unique: compare their orthogonality.
        (lambda m: (lambda q, r: (_gram(q), r))(*qb.linalg.qr(m, mode="complete")), (tall,)),
        (lambda m: tuple(qb.linalg.qr(m)), (wide,)),
        (qb.grad(lambda m: (qb.linalg.qr(m).Q ** 3).sum() + (qb.linalg.qr(m).R ** 3).sum()), (tall,)),
        (lambda m: tuple(qb.linalg.svd(m)), (tall,)),
        (lambda m: tuple(qb.linalg.svd(m)), (wide,)),
        (lambda m: (lambda u, s, vh: (_gram(u), s, vh))(*qb.linalg.svd(m, full_matrices=True)), (tall,)),
        (lambda m: (lambda u, s, vh: (u, s, _gram(vh.transpose([0, 2, 1]))))(*qb.linalg.svd(m, full_matrices=True)), (wide,)),
        (qb.grad(lambda m: (qb.linalg.svd(m).U ** 3).sum() + (qb.linalg.svd(m).Vh ** 3).sum()), (tall,)),
        (qb.linalg.lstsq, (tall, rhs)),
        (qb.linalg.solve, (a, b)),
        (qb.grad(lambda m, r: (qb.linalg.solve(m, r) ** 2).sum()), (a, b)),
        (lambda m: tuple(qb.linalg.slogdet(m)), (a,)),
        (qb.grad(lambda m: qb.linalg.slogdet(m).logabsdet.sum()), (a,)),
        (lambda m: tuple(qb.linalg.eigh(m)), (symmetric,)),
        (qb.grad(lambda m: (qb.linalg.eigh(m).eigenvectors ** 3).sum()), (symmetric,)),
    )
    for device, gate in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")):
        if os.environ.get(gate) != "1":
            continue
        for function, arguments in functions:
            assert_device_matches_cpu(device, function, arguments, 5e-4)
        singular = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        sign, logabsdet = qb.jit(lambda m: tuple(qb.linalg.slogdet(m)), device=device)(singular)
        assert sign.item() == 0.0 and logabsdet.item() == -np.inf


def assert_device_matches_cpu(device, function, arguments, tolerance):
    """`function` under `jit` on `device` against the CPU, output by output."""
    expected = qb.jit(function)(*arguments)
    actual = qb.jit(function, device=device)(*arguments)
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype, (got.dtype, want.dtype)
        assert_close(got, want, tolerance)


def ill_conditioned(rng, batch, n, condition):
    """Matrices with singular values spaced log-uniformly from 1 to `1 / condition`."""
    left, _ = np.linalg.qr(rng.normal(size=batch + (n, n)))
    right, _ = np.linalg.qr(rng.normal(size=batch + (n, n)))
    values = np.logspace(0.0, -np.log10(condition), n)
    return (left * values) @ transposed(right)


def test_mlx_linalg_matches_the_cpu():
    """Every `qb.linalg` function and `solve` on MLX against the CPU: values
    and derivatives, batched, ill-conditioned, singular, and non-finite.

    MLX factors with LAPACK on its CPU stream in float32 and applies the
    CPU's conventions on the GPU stream, so outputs, signs and completed
    bases included, agree with the CPU's float64 kernels to float32
    accuracy. Unless stated otherwise every matrix has a condition number
    below about 100 and eigenvalue or singular value gaps above about 0.1.
    """
    if os.environ.get("QUABLA_MLX_TEST") != "1":
        return
    rng = np.random.default_rng(15)

    def f32(x):
        return np.asarray(x, dtype=np.float32)

    a = f32(well_conditioned(rng, (2, 3, 4, 4)))
    b = f32(rng.normal(size=(2, 3, 4, 2)))
    vector = f32(rng.normal(size=4))
    direction = f32(rng.normal(size=a.shape))
    lower = np.tril(well_conditioned(rng, (3, 4, 4)))
    noisy = f32(lower + np.triu(rng.normal(size=lower.shape), 1))
    spd = f32(lower @ transposed(lower))
    rhs = f32(rng.normal(size=(3, 4, 3)))
    symmetric = f32(symmetric_with_gaps(rng, (2, 3), 5))
    symmetric_direction = rng.normal(size=symmetric.shape)
    symmetric_direction = f32(symmetric_direction + transposed(symmetric_direction))
    left, _ = np.linalg.qr(rng.normal(size=(2, 6, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(2, 4, 4)))
    tall = f32((left * np.array([4.0, 3.0, 2.0, 1.0])) @ transposed(right))
    small, _ = np.linalg.qr(rng.normal(size=(2, 3, 3)))
    # A [2, 3, 6] matrix with singular values 3, 2, 1.
    wide = f32(transposed((left[..., :3] * np.array([3.0, 2.0, 1.0])) @ transposed(small)))
    tall_rhs = f32(rng.normal(size=(2, 6, 2)))
    weights = qb.asarray(f32(rng.normal(size=(2, 6, 4))))

    def cubed_gradient(function):
        return qb.grad(lambda m: (function(m) ** 3).sum())

    cases = (
        (qb.linalg.solve, (a, b)),
        (qb.solve, (a, vector)),
        (qb.grad(lambda m, r: (qb.linalg.solve(m, r) ** 2).sum(), argnums=(0, 1)), (a, b)),
        (lambda m, d: qb.jvp(lambda x: qb.linalg.solve(x, b), (m,), (d,))[1], (a, direction)),
        (qb.hessian(lambda m: (qb.linalg.solve(m, b[0, 0]) ** 2).sum()), (a[0, 0],)),
        (qb.vmap(qb.grad(lambda m, r: (qb.linalg.solve(m, r) ** 2).sum())), (a[0], b[0])),
        (
            lambda m, r: tuple(
                qb.linalg.solve_triangular(m, r, trans=trans, lower=low)
                for trans in (0, 1)
                for low in (True, False)
            ),
            (noisy, rhs),
        ),
        (qb.grad(lambda m: (qb.linalg.solve_triangular(m, rhs, lower=True) ** 2).sum()), (noisy,)),
        (lambda m, r: qb.linalg.cho_solve(qb.linalg.cholesky(m), r), (spd, rhs)),
        (qb.grad(lambda m: (qb.linalg.cho_solve(qb.linalg.cholesky(m), rhs) ** 2).sum()), (spd,)),
        (lambda m: tuple(qb.linalg.slogdet(m)), (a,)),
        (qb.linalg.det, (a,)),
        (qb.linalg.inv, (a,)),
        (qb.grad(lambda m: qb.linalg.slogdet(m).logabsdet.sum()), (a,)),
        (qb.grad(lambda m: qb.linalg.det(m).sum()), (a,)),
        (lambda m, d: qb.jvp(qb.linalg.inv, (m,), (d,))[1], (a, direction)),
        (qb.hessian(lambda m: qb.linalg.slogdet(m).logabsdet), (a[0, 0],)),
        (lambda m: tuple(qb.linalg.eigh(m)), (symmetric,)),
        # Components of equal magnitude: the first one is made positive.
        (lambda m: tuple(qb.linalg.eigh(m)), (f32([[2.0, 0.0], [2.0, 2.0]]),)),
        (cubed_gradient(lambda m: qb.linalg.eigh(m).eigenvalues), (symmetric,)),
        (cubed_gradient(lambda m: qb.linalg.eigh(m).eigenvectors), (symmetric,)),
        (
            lambda m, d: qb.jvp(lambda x: tuple(qb.linalg.eigh(x)), (m,), (d,))[1],
            (symmetric, symmetric_direction),
        ),
        (lambda m: tuple(qb.linalg.qr(m)), (tall,)),
        (lambda m: tuple(qb.linalg.qr(m)), (wide,)),
        (lambda m: tuple(qb.linalg.qr(m)), (a,)),
        # The complete Q follows the CPU's Householder completion exactly.
        (lambda m: tuple(qb.linalg.qr(m, mode="complete")), (tall,)),
        (lambda m: qb.linalg.qr(m, mode="r"), (wide,)),
        (cubed_gradient(lambda m: qb.linalg.qr(m).Q), (tall,)),
        (cubed_gradient(lambda m: qb.linalg.qr(m).R), (wide,)),
        (lambda m: tuple(qb.linalg.svd(m)), (tall,)),
        (lambda m: tuple(qb.linalg.svd(m)), (wide,)),
        # So do the extra columns of a full U and the extra rows of a full Vh.
        (lambda m: tuple(qb.linalg.svd(m, full_matrices=True)), (tall,)),
        (lambda m: tuple(qb.linalg.svd(m, full_matrices=True)), (wide,)),
        (lambda m: qb.linalg.svd(m, compute_uv=False), (a,)),
        (cubed_gradient(lambda m: qb.linalg.svd(m, compute_uv=False)), (tall,)),
        (cubed_gradient(lambda m: qb.linalg.svd(m).U * weights), (tall,)),
        (cubed_gradient(lambda m: qb.linalg.svd(m).Vh), (wide,)),
        (lambda m, r: tuple(qb.linalg.lstsq(m, r, return_residuals=True)), (tall, tall_rhs)),
        (qb.linalg.lstsq, (wide, tall_rhs[..., :3, :])),
        (qb.grad(lambda m, r: (qb.linalg.lstsq(m, r) ** 2).sum()), (tall, tall_rhs)),
    )
    for function, arguments in cases:
        assert_device_matches_cpu("mlx", function, arguments, 1e-5)

    # Condition 1e3: the float32 error bound grows to about condition * eps.
    ill = f32(ill_conditioned(rng, (2,), 5, 1e3))
    ill_rhs = f32(rng.normal(size=(2, 5, 2)))
    for function, arguments in (
        (qb.linalg.solve, (ill, ill_rhs)),
        (lambda m: tuple(qb.linalg.slogdet(m)), (ill,)),
        (lambda m: qb.linalg.svd(m, compute_uv=False), (ill,)),
        (lambda m: tuple(qb.linalg.qr(m)), (ill,)),
        (lambda m: qb.linalg.eigh(m @ m.transpose([0, 2, 1])).eigenvalues, (ill,)),
    ):
        assert_device_matches_cpu("mlx", function, arguments, 2e-4)

    def on_mlx(function):
        return qb.jit(function, device="mlx")

    # Exactly singular: slogdet and det follow NumPy; solve, inv, and the
    # logabsdet gradient raise like the CPU. A zero pivot is detected
    # exactly, so the matrix has a zero row: a matrix that is singular only
    # in exact arithmetic (dependent rows) can leave a pivot of rounding
    # size under LAPACK's float32 rounding, and is then solved.
    singular = f32([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], np.eye(3)])
    sign, logabsdet = (np.asarray(t) for t in on_mlx(lambda m: tuple(qb.linalg.slogdet(m)))(singular))
    assert sign.tolist() == [0.0, 1.0] and logabsdet.tolist() == [-np.inf, 0.0]
    assert np.asarray(on_mlx(qb.linalg.det)(singular)).tolist() == [0.0, 1.0]
    for function, arguments in (
        (qb.linalg.solve, (singular, f32(np.ones((2, 3, 1))))),
        (qb.linalg.inv, (singular,)),
        (qb.grad(lambda m: qb.linalg.slogdet(m).logabsdet.sum()), (singular,)),
    ):
        raises(ValueError, qb.jit(function), *arguments, match="non-singular")
        raises(ValueError, on_mlx(function), *arguments, match="non-singular")
    # Rank deficiency: the factorizations stay orthonormal and exact.
    deficient = f32(rng.normal(size=(2, 6, 2)) @ rng.normal(size=(2, 2, 4)))
    u, s, vh = (np.asarray(t, dtype=np.float64) for t in on_mlx(lambda m: tuple(qb.linalg.svd(m)))(deficient))
    assert_close(s, np.linalg.svd(deficient.astype(np.float64), compute_uv=False), 1e-5)
    assert_close((u * s[..., None, :]) @ vh, deficient, 1e-5)
    assert_close(transposed(u) @ u, np.broadcast_to(np.eye(4), (2, 4, 4)), 1e-5)
    q, r = (np.asarray(t, dtype=np.float64) for t in on_mlx(lambda m: tuple(qb.linalg.qr(m)))(deficient))
    assert_close(q @ r, deficient, 1e-5)
    assert_close(transposed(q) @ q, np.broadcast_to(np.eye(4), (2, 4, 4)), 1e-5)
    assert np.all(np.diagonal(r, axis1=-2, axis2=-1) >= 0.0)
    semidefinite = deficient[..., :4, :] @ transposed(deficient[..., :4, :])
    assert_device_matches_cpu("mlx", lambda m: qb.linalg.eigh(m).eigenvalues, (semidefinite,), 1e-4)

    # A non-finite matrix gives NaN outputs without disturbing its batch.
    mixed = tall.copy()
    mixed[1, 2, 1] = np.nan
    square = symmetric[0, :2].copy()
    square[0, 0, 0] = np.inf
    for function, argument in (
        (lambda m: tuple(qb.linalg.svd(m)), mixed),
        (lambda m: tuple(qb.linalg.qr(m)), mixed),
        (lambda m: tuple(qb.linalg.eigh(m)), square),
        (lambda m: tuple(qb.linalg.slogdet(m)), square),
    ):
        for got, want in zip(on_mlx(function)(argument), qb.jit(function)(argument)):
            got, want = np.asarray(got), np.asarray(want)
            nonfinite = 1 if argument is mixed else 0
            assert np.all(np.isnan(got[nonfinite])) and np.all(np.isnan(want[nonfinite]))
            assert_close(got[1 - nonfinite], want[1 - nonfinite], 1e-4)

    # float64 programs run in float32 on MLX and keep their logical dtype.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_device_matches_cpu("mlx", qb.linalg.solve, (a.astype(np.float64), b.astype(np.float64)), 1e-4)
        assert_device_matches_cpu("mlx", lambda m: tuple(qb.linalg.svd(m)), (tall.astype(np.float64),), 1e-4)


if __name__ == "__main__":
    if np is None:
        print("skipped test_linalg_api: numpy is not installed")
        raise SystemExit(0)
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
