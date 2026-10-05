"""quabla.linalg: batched solves, determinants, and the symmetric eigensolver.

Values are compared with NumPy, and derivatives with central differences of
NumPy references. Tolerances scale with the condition of each test matrix:
every matrix here has a condition number below ~50 and, for `eigh`,
eigenvalue gaps above ~0.3, so a float64 solve is accurate to ~1e-13 and a
central difference with step 1e-6 to ~1e-8.
"""

import os

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


def test_optional_device_parity():
    rng = np.random.default_rng(8)
    a = well_conditioned(rng, (2, 3, 3)).astype(np.float32)
    b = rng.normal(size=(2, 3, 2)).astype(np.float32)
    symmetric = symmetric_with_gaps(rng, (2,), 3).astype(np.float32)
    functions = (
        (qb.linalg.solve, (a, b)),
        (qb.grad(lambda m, r: (qb.linalg.solve(m, r) ** 2).sum()), (a, b)),
        (lambda m: tuple(qb.linalg.slogdet(m)), (a,)),
        (qb.grad(lambda m: qb.linalg.slogdet(m).logabsdet.sum()), (a,)),
        (lambda m: tuple(qb.linalg.eigh(m)), (symmetric,)),
        (qb.grad(lambda m: (qb.linalg.eigh(m).eigenvectors ** 3).sum()), (symmetric,)),
    )
    if os.environ.get("QUABLA_MLX_TEST") == "1":
        for function, arguments in functions:
            raises(qb.UnsupportedOperationError, qb.jit(function, device="mlx"), *arguments)
    if os.environ.get("QUABLA_CUDA_TEST") == "1":
        for function, arguments in functions:
            expected = qb.jit(function)(*arguments)
            actual = qb.jit(function, device="cuda")(*arguments)
            expected = expected if isinstance(expected, tuple) else (expected,)
            actual = actual if isinstance(actual, tuple) else (actual,)
            for got, want in zip(actual, expected):
                assert_close(got, want, 5e-4)
        singular = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        sign, logabsdet = qb.jit(lambda m: tuple(qb.linalg.slogdet(m)), device="cuda")(singular)
        assert sign.item() == 0.0 and logabsdet.item() == -np.inf


if __name__ == "__main__":
    if np is None:
        print("skipped test_linalg_api: numpy is not installed")
        raise SystemExit(0)
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
