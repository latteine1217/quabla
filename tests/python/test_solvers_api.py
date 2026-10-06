"""Iterative and implicit solvers: linalg.cg, linalg.gmres, and the Newton
root finder `quabla.newton`.

Solutions are checked against the dense `linalg.solve` and closed forms;
derivatives against the derivatives of the dense solve, analytic
implicit-function-theorem results, and central finite differences of the
iterative solve itself.
"""

import math
import os

import quabla as qb
from quabla import newton


def raises(kind, function, *args, match=None, **kwargs):
    try:
        function(*args, **kwargs)
    except kind as error:
        if match is not None:
            assert match in str(error), str(error)
        return
    raise AssertionError(f"expected {kind.__name__}")


def assert_close(actual, expected, tolerance):
    actual = actual.tolist() if hasattr(actual, "tolist") else actual
    expected = expected.tolist() if hasattr(expected, "tolist") else expected
    if isinstance(actual, list):
        assert len(actual) == len(expected), (actual, expected)
        for got, want in zip(actual, expected):
            assert_close(got, want, tolerance)
        return
    assert abs(actual - expected) <= tolerance * max(1.0, abs(expected)), (actual, expected)


def matrix(rows, columns, seed):
    """A deterministic dense matrix with entries in [-1, 1]."""
    return qb.array(
        [[math.sin(seed + 1.7 * i + 3.1 * j + 0.37 * i * j) for j in range(columns)] for i in range(rows)]
    )


def spd(n, seed=0.0):
    b = matrix(n, n, seed)
    return b @ b.T + n * qb.eye(n)


def nonsymmetric(n, seed=0.5):
    return matrix(n, n, seed) + 3.0 * qb.eye(n)


def vector(n, seed=2.0):
    return qb.array([math.cos(seed + 0.9 * i) for i in range(n)])


def matvec(x, a, *_):
    return (a @ x.reshape([-1, 1])).reshape([-1])


def central_difference(function, value, step=1e-6):
    return (function(value + step) - function(value - step)) / (2.0 * step)


# -- conjugate gradients ----------------------------------------------------------


def test_cg_matches_dense_solve():
    n = 8
    a, b = spd(n), vector(n)
    expected = qb.linalg.solve(a, b)
    x, info = qb.linalg.cg(matvec, b, args=(a,), tol=1e-12, info=True)
    assert_close(x, expected, 1e-11)
    assert bool(info["success"])
    assert 1 <= info["iterations"].item() <= n + 2, info
    assert info["residual_norm"].item() <= 1e-11 * b.norm().item()
    # A traced solve is one loop region with the same arithmetic.
    staged = qb.jit(lambda b, a: qb.linalg.cg(matvec, b, args=(a,), tol=1e-12))(b, a)
    assert_close(staged, x, 1e-14)
    # A warm start at the solution needs no iteration; a zero right-hand side
    # returns zero.
    _, warm = qb.linalg.cg(matvec, b, args=(a,), x0=expected, tol=1e-8, info=True)
    assert warm["iterations"].item() == 0.0
    assert qb.linalg.cg(matvec, qb.zeros([n]), args=(a,)).tolist() == [0.0] * n


def test_cg_jacobi_preconditioner_needs_fewer_iterations():
    # D A0 D with D spanning four decades: condition number about 1e8, which
    # the Jacobi preconditioner x / diag(A) removes.
    n = 24
    scale = qb.array([10.0 ** (4.0 * i / (n - 1)) for i in range(n)])
    a = spd(n, 1.0) * scale.reshape([n, 1]) * scale.reshape([1, n])
    diagonal = (a * qb.eye(n)).sum(axis=1)
    b = vector(n)
    expected = qb.linalg.solve(a, b)
    plain, plain_info = qb.linalg.cg(matvec, b, args=(a,), tol=1e-10, info=True)
    jacobi, jacobi_info = qb.linalg.cg(
        matvec, b, args=(a, diagonal), tol=1e-10, M=lambda r, a, d: r / d, info=True
    )
    for x in (plain, jacobi):
        relative = (x - expected).norm().item() / expected.norm().item()
        assert relative < 1e-6, relative
    assert bool(plain_info["success"]) and bool(jacobi_info["success"])
    plain_count, jacobi_count = plain_info["iterations"].item(), jacobi_info["iterations"].item()
    assert jacobi_count <= n and jacobi_count * 3 < plain_count, (jacobi_count, plain_count)


def poisson(u, kappa):
    """-kappa u'' on a uniform grid of the unit interval with u = 0 at both
    ends, scaled by h^2: kappa * (2 u_i - u_{i-1} - u_{i+1}). Matrix-free."""
    zero = qb.zeros([1], u.dtype)
    left = qb.concat([zero, u[:-1]], 0)
    right = qb.concat([u[1:], zero], 0)
    return kappa * (2.0 * u - left - right)


def test_cg_solves_matrix_free_poisson():
    n = 63
    h = 1.0 / (n + 1)
    grid = qb.array([(i + 1) * h for i in range(n)])
    # -u'' = pi^2 sin(pi x) has u = sin(pi x); the second-order stencil is
    # accurate to about (pi h)^2 / 12 relative.
    rhs = (math.pi**2 * h * h) * qb.sin(math.pi * grid)
    kappa = qb.array(1.0)
    u, info = qb.linalg.cg(poisson, rhs, args=(kappa,), tol=1e-12, info=True)
    assert bool(info["success"])
    # The stencil has n distinct eigenvalues, so CG needs about n steps; the
    # symmetry of sin(pi x) halves the Krylov space it reaches.
    assert info["iterations"].item() <= n
    error = (u - qb.sin(math.pi * grid)).abs().max().item()
    assert error < (math.pi * h) ** 2 / 12 * 1.05, error
    dense = qb.linalg.solve(qb.jit(qb.jacobian(lambda v: poisson(v, kappa)))(u), rhs)
    assert_close(u, dense, 1e-10)
    # x = L^-1 b / kappa, so d sum(x) / d kappa = -sum(x) / kappa.
    gradient = qb.grad(lambda k: qb.linalg.cg(poisson, rhs, args=(k,), tol=1e-12).sum())(qb.array(2.0))
    assert_close(gradient, -u.sum().item() / 4.0, 1e-9)
    # A Python number in args is a constant.
    assert_close(qb.linalg.cg(poisson, rhs, args=(1.0,), tol=1e-12), u, 1e-14)


def test_cg_float32_matches_float64():
    n = 10
    a, b = spd(n, 3.0), vector(n)
    x64 = qb.linalg.cg(matvec, b, args=(a,), tol=1e-12)
    x32, info = qb.linalg.cg(
        matvec, b.astype(qb.float32), args=(a.astype(qb.float32),), tol=1e-6, info=True
    )
    assert x32.dtype == qb.float32 and info["residual_norm"].dtype == qb.float32
    assert bool(info["success"])
    assert_close(x32, x64, 1e-5)
    gradient = qb.jit(
        qb.grad(lambda a: qb.linalg.cg(matvec, b.astype(qb.float32), args=(a,), tol=1e-6).sum())
    )(a.astype(qb.float32))
    expected = qb.grad(lambda a: qb.linalg.solve(a, b).sum())(a)
    assert gradient.dtype == qb.float32
    assert_close(gradient, expected, 1e-5)


def test_solvers_keep_the_shape_of_b():
    # A diagonal operator on [3, 4] arrays: x = b / d, and the gradient of
    # sum(x) is 1 / d for b and -b / d^2 for d.
    d = qb.reshape(qb.arange(1.0, 13.0), [3, 4])
    b = qb.reshape(qb.cos(qb.arange(12.0)), [3, 4])
    for solver in (qb.linalg.cg, qb.linalg.gmres):
        x = solver(lambda x, d: d * x, b, args=(d,), tol=1e-13)
        assert x.shape == [3, 4]
        assert_close(x, b / d, 1e-12)
        b_bar, d_bar = qb.grad(
            lambda b, d: solver(lambda x, d: d * x, b, args=(d,), tol=1e-13).sum(), argnums=(0, 1)
        )(b, d)
        assert_close(b_bar, 1.0 / d, 1e-12)
        assert_close(d_bar, -b / (d * d), 1e-12)


def test_cg_breakdown_on_an_indefinite_operator_is_reported():
    # p^T A p = 0 at the first step: the infinite step, then NaN, stops the
    # loop within two iterations instead of running to maxiter.
    indefinite = qb.array([[1.0, 0.0], [0.0, -1.0]])
    x, info = qb.linalg.cg(matvec, qb.array([1.0, 1.0]), args=(indefinite,), maxiter=100, info=True)
    assert not bool(info["success"]) and info["iterations"].item() <= 2.0
    raises(RuntimeError, qb.linalg.cg, matvec, qb.array([1.0, 1.0]), args=(indefinite,), match="positive definite")


# -- GMRES ------------------------------------------------------------------------


def test_gmres_matches_dense_solve_on_nonsymmetric_matrices():
    n = 9
    a, b = nonsymmetric(n), vector(n)
    expected = qb.linalg.solve(a, b)
    x, info = qb.linalg.gmres(matvec, b, args=(a,), tol=1e-12, info=True)
    assert_close(x, expected, 1e-11)
    assert bool(info["success"]) and info["iterations"].item() <= n
    assert info["residual_norm"].item() <= 1e-11 * b.norm().item()
    staged = qb.jit(lambda b, a: qb.linalg.gmres(matvec, b, args=(a,), tol=1e-12))(b, a)
    assert_close(staged, x, 1e-14)
    # cg is not for nonsymmetric matrices; gmres also solves SPD ones.
    s = spd(n)
    assert_close(qb.linalg.gmres(matvec, b, args=(s,), tol=1e-12), qb.linalg.solve(s, b), 1e-11)
    # A right preconditioner changes the iteration, not the solution.
    diagonal = (a * qb.eye(n)).sum(axis=1)
    preconditioned = qb.linalg.gmres(
        matvec, b, args=(a, diagonal), tol=1e-12, M=lambda r, a, d: r / d
    )
    assert_close(preconditioned, expected, 1e-11)


def test_gmres_restart_behaviour():
    # Eigenvalues with real parts in [2.9, 9.8]: restarted GMRES converges
    # (with an eigenvalue near zero it may stagnate for small restarts).
    n = 16
    a, b = nonsymmetric(n, 1.5) + 3.0 * qb.eye(n), vector(n, 0.3)
    expected = qb.linalg.solve(a, b)
    _, full = qb.linalg.gmres(matvec, b, args=(a,), tol=1e-10, restart=n, info=True)
    x, short = qb.linalg.gmres(matvec, b, args=(a,), tol=1e-10, restart=3, info=True)
    assert bool(full["success"]) and bool(short["success"])
    assert_close(x, expected, 1e-8)
    # A full basis converges within n steps; restarting discards the basis,
    # so GMRES(3) needs more steps for the same tolerance.
    assert full["iterations"].item() <= n
    assert short["iterations"].item() > full["iterations"].item(), (short, full)
    # GMRES(1) stagnates on a rotation: r^T A r = 0, so a one-dimensional
    # Krylov space never reduces the residual, while GMRES(2) is exact.
    rotation = qb.array([[0.0, 1.0], [-1.0, 0.0]])
    rhs = qb.array([1.0, 0.0])
    x, stuck = qb.linalg.gmres(matvec, rhs, args=(rotation,), restart=1, maxiter=50, info=True)
    assert not bool(stuck["success"])
    assert_close(stuck["residual_norm"], 1.0, 1e-12)
    raises(RuntimeError, qb.linalg.gmres, matvec, rhs, args=(rotation,), restart=1, maxiter=5, match="did not converge")
    assert_close(qb.linalg.gmres(matvec, rhs, args=(rotation,), restart=2, tol=1e-14), [0.0, 1.0], 1e-14)


def test_gmres_float32_matches_float64():
    n = 10
    a, b = nonsymmetric(n, 2.5), vector(n)
    x64 = qb.linalg.gmres(matvec, b, args=(a,), tol=1e-12)
    x32 = qb.jit(lambda b, a: qb.linalg.gmres(matvec, b, args=(a,), tol=1e-6))(
        b.astype(qb.float32), a.astype(qb.float32)
    )
    assert x32.dtype == qb.float32
    assert_close(x32, x64, 1e-5)


# -- derivatives of the linear solvers --------------------------------------------


def linear_losses(solver, tol=1e-12):
    target = vector(9, 4.0)

    def iterative(b, a):
        x = solver(matvec, b, args=(a,), tol=tol)
        return ((x - target) * (x - target)).sum()

    def direct(b, a):
        x = qb.linalg.solve(a, b)
        return ((x - target) * (x - target)).sum()

    return iterative, direct


def test_linear_solver_gradients_match_the_dense_solve():
    b = vector(9, 1.0)
    for solver, a in ((qb.linalg.cg, spd(9, 2.0)), (qb.linalg.gmres, nonsymmetric(9, 3.0))):
        iterative, direct = linear_losses(solver)
        expected = qb.grad(direct, argnums=(0, 1))(b, a)
        for transform in (lambda f: f, qb.jit):
            gradients = transform(qb.grad(iterative, argnums=(0, 1)))(b, a)
            for got, want in zip(gradients, expected):
                assert_close(got, want, 1e-9)
        value, gradients = qb.value_and_grad(iterative, argnums=(0, 1))(b, a)
        assert_close(value, direct(b, a), 1e-12)
        # vjp with a cotangent, through a pytree of operator parameters.
        _, pullback = qb.vjp(
            lambda b, p: solver(lambda x, p: matvec(x, p["a"]) * p["s"], b, args=(p,), tol=1e-12),
            b,
            {"a": a, "s": qb.array(2.0)},
        )
        cotangent = vector(9, 5.0)
        b_bar, p_bar = pullback(cotangent)
        _, dense_pullback = qb.vjp(lambda b, p: qb.linalg.solve(p["a"] * p["s"], b), b, {"a": a, "s": qb.array(2.0)})
        dense_b_bar, dense_p_bar = dense_pullback(cotangent)
        assert_close(b_bar, dense_b_bar, 1e-9)
        assert_close(p_bar["a"], dense_p_bar["a"], 1e-9)
        assert_close(p_bar["s"], dense_p_bar["s"], 1e-9)


def test_preconditioned_gradients_match_the_dense_solve():
    # The adjoint solve applies M^T (the VJP of M) for gmres and M for cg; a
    # non-diagonal M checks the transpose. M depends on `a` through args, but
    # the solution does not depend on M, so it adds nothing to the gradient.
    b = vector(9, 1.0)
    for solver, a in ((qb.linalg.cg, spd(9, 2.0)), (qb.linalg.gmres, nonsymmetric(9, 3.0))):
        if solver is qb.linalg.cg:
            def precondition(r, a, d):
                return r / d
        else:
            def precondition(r, a, d):
                return r / d + 0.05 * qb.concat([r[1:], r[:1]], 0)

        def loss(b, a):
            d = (a * qb.eye(9)).sum(axis=1)
            return solver(matvec, b, args=(a, d), tol=1e-13, M=precondition).sum()

        expected = qb.grad(lambda b, a: qb.linalg.solve(a, b).sum(), argnums=(0, 1))(b, a)
        for got, want in zip(qb.grad(loss, argnums=(0, 1))(b, a), expected):
            assert_close(got, want, 1e-9)


def test_linear_solver_gradients_match_finite_differences():
    b = vector(9, 1.0)
    direction = matrix(9, 9, 7.0)
    for solver, a in ((qb.linalg.cg, spd(9, 2.0)), (qb.linalg.gmres, nonsymmetric(9, 3.0))):
        iterative, _ = linear_losses(solver, tol=1e-13)
        if solver is qb.linalg.cg:
            direction = direction + direction.T

        def along(t):
            return iterative(b, a + t * direction).item()

        gradient = qb.grad(iterative, argnums=1)(b, a)
        directional = (gradient * direction).sum().item()
        assert abs(directional - central_difference(along, 0.0)) < 1e-6 * max(1.0, abs(directional))


def test_linear_solver_second_derivatives():
    b = vector(9, 1.0)
    for solver, a in ((qb.linalg.cg, spd(9, 2.0)), (qb.linalg.gmres, nonsymmetric(9, 3.0))):
        iterative, direct = linear_losses(solver)

        def curvature(loss):
            return qb.grad(qb.grad(lambda t: loss(b * t, a * (2.0 - t))))(qb.array(1.0))

        assert_close(curvature(iterative), curvature(direct), 1e-8)


def test_linear_solver_unsupported_transforms_raise():
    a, b = spd(4), vector(4)
    raises(
        ValueError,
        qb.jvp,
        lambda b: qb.linalg.cg(matvec, b, args=(a,)),
        (b,),
        (b,),
        match="forward-mode differentiation",
    )
    raises(Exception, qb.vmap(lambda b: qb.linalg.cg(matvec, b, args=(a,))), qb.stack([b, b]), match="while_loop")
    # A third reverse pass reaches the plain iteration.
    raises(
        ValueError,
        qb.grad(qb.grad(qb.grad(lambda t: qb.linalg.cg(matvec, b, args=(a * t,)).sum()))),
        qb.array(1.0),
        match="while_loop",
    )


def test_linear_solver_argument_validation():
    a, b = spd(4), vector(4)
    raises(TypeError, qb.linalg.cg, matvec, b, args=a)
    raises(TypeError, qb.linalg.cg, 3.0, b)
    raises(ValueError, qb.linalg.cg, matvec, b, args=(a,), tol=-1.0)
    raises(ValueError, qb.linalg.cg, matvec, b, args=(a,), maxiter=0)
    raises(ValueError, qb.linalg.gmres, matvec, b, args=(a,), restart=0)
    raises(ValueError, qb.linalg.cg, matvec, b, args=(a,), x0=qb.zeros([3]))
    raises(ValueError, qb.linalg.cg, lambda x: x[:2], b, match="matvec must return")
    raises(TypeError, qb.linalg.cg, lambda x: x.astype(qb.float64), b.astype(qb.float32), match="dtype")
    raises(TypeError, qb.linalg.gmres, matvec, b, args=(a,), M=1.0)
    raises(RuntimeError, qb.linalg.cg, matvec, b, args=(a,), maxiter=1, tol=1e-12, match="did not converge")
    # Under jit the report is info["success"].
    _, info = qb.jit(lambda b: qb.linalg.cg(matvec, b, args=(a,), maxiter=1, tol=1e-12, info=True))(b)
    assert not bool(info["success"]) and info["iterations"].item() <= 2.0


# -- Newton root finding ----------------------------------------------------------


def circle_and_line(x, p, q):
    """x^2 + y^2 = p and x = q y; the root with y > 0 is y = sqrt(p / (1 + q^2))."""
    return qb.concat([x[0:1] * x[0:1] + x[1:2] * x[1:2] - p, x[0:1] - q * x[1:2]], 0)


def test_root_of_a_nonlinear_system_and_its_ift_gradient():
    p, q, x0 = qb.array(2.0), qb.array(0.5), qb.array([1.0, 1.0])
    y = math.sqrt(2.0 / 1.25)
    x, info = newton(circle_and_line, x0, args=(p, q), info=True)
    assert_close(x, [0.5 * y, y], 1e-15)
    assert bool(info["success"]) and info["iterations"].item() <= 8
    assert info["residual_norm"].item() < 1e-15
    staged = qb.jit(lambda p, q: newton(circle_and_line, x0, args=(p, q)))(p, q)
    assert_close(staged, x, 1e-15)

    # sum(x) = (1 + q) sqrt(p / (1 + q^2)).
    def total(p, q):
        return newton(circle_and_line, x0, args=(p, q)).sum()

    dp = (1.5) / (2.0 * math.sqrt(2.0 * 1.25))
    dq = y + 1.5 * (-0.5 * math.sqrt(2.0) * 1.25**-1.5)
    for transform in (lambda f: f, qb.jit):
        gradient = transform(qb.grad(total, argnums=(0, 1)))(p, q)
        assert_close(list(gradient), [dp, dq], 1e-12)
    # The Jacobian dx/dp = -J^-1 df/dp, by reverse mode.
    assert_close(qb.jacobian(lambda p: newton(circle_and_line, x0, args=(p, q)))(p), [0.5 * dp / 1.5, dp / 1.5], 1e-12)
    # The initial guess receives no gradient.
    assert_close(qb.grad(lambda x0: newton(circle_and_line, x0, args=(p, q)).sum())(x0), [0.0, 0.0], 0.0)
    fd = central_difference(lambda t: total(qb.array(t), q).item(), 2.0)
    assert abs(fd - dp) < 1e-8


def test_root_second_derivatives_match_closed_forms():
    def cube_root(a):
        return newton(lambda x, a: x * x * x - a, qb.array(1.0), args=(a,))

    a = qb.array(8.0)
    assert_close(cube_root(a), 2.0, 1e-15)
    # d a^(1/3) = a^(-2/3) / 3 and d^2 = -2 a^(-5/3) / 9.
    assert_close(qb.grad(cube_root)(a), 1.0 / 12.0, 1e-14)
    assert_close(qb.grad(qb.grad(cube_root))(a), -2.0 / (9.0 * 32.0), 1e-13)
    assert_close(qb.jit(qb.grad(qb.grad(cube_root)))(a), -2.0 / (9.0 * 32.0), 1e-13)

    # Hessian of sum(x) = (1 + q) sqrt(p / (1 + q^2)) at p = 2, q = 1.
    def total(pq):
        return newton(circle_and_line, qb.array([1.0, 0.5]), args=(pq[0], pq[1])).sum()

    def closed_form(pq):
        p, q = pq[0], pq[1]
        return (1.0 + q) * (p / (1.0 + q * q)).sqrt()

    point = qb.array([2.0, 1.0])
    assert_close(qb.hessian(total)(point), qb.hessian(closed_form)(point), 1e-11)


def test_root_line_search_and_float32():
    # Undamped Newton on arctan diverges from x0 = 10 (|x| grows every step);
    # the line search keeps it on the way to the root at 0.
    x, info = newton(lambda x: qb.atan2(x, qb.ones_like(x)), qb.array(10.0), info=True)
    assert bool(info["success"]) and abs(x.item()) < 1e-15
    x32, info32 = newton(
        circle_and_line,
        qb.array([1.0, 1.0], dtype=qb.float32),
        args=(qb.array(2.0, dtype=qb.float32), qb.array(0.5, dtype=qb.float32)),
        info=True,
    )
    y = math.sqrt(2.0 / 1.25)
    assert x32.dtype == qb.float32 and bool(info32["success"])
    assert_close(x32, [0.5 * y, y], 2e-7)
    gradient = qb.jit(
        qb.grad(
            lambda p: newton(
                circle_and_line, qb.array([1.0, 1.0], dtype=qb.float32), args=(p, qb.array(0.5, dtype=qb.float32))
            ).sum()
        )
    )(qb.array(2.0, dtype=qb.float32))
    assert gradient.dtype == qb.float32
    assert_close(gradient, 1.5 / (2.0 * math.sqrt(2.5)), 1e-6)


def test_root_reports_non_convergence():
    # exp(x) + 1 has no real root: the iterates run off to -inf.
    def no_root(x):
        return qb.exp(x) + 1.0

    raises(RuntimeError, newton, no_root, qb.array(1.0), match="did not converge")
    x, info = newton(no_root, qb.array(1.0), info=True)
    assert not bool(info["success"])
    _, info = qb.jit(lambda x0: newton(no_root, x0, info=True))(qb.array(1.0))
    assert not bool(info["success"])
    # An exactly singular Jacobian at the start stops the solve rather than
    # raising, and maxiter bounds the iterations.
    x, info = newton(lambda x: x * x - 1.0, qb.array(0.0), info=True)
    assert not bool(info["success"]) and x.item() == 0.0
    _, info = newton(lambda x: x * x - 2.0, qb.array(100.0), maxiter=2, info=True)
    assert not bool(info["success"]) and info["iterations"].item() == 2.0
    raises(TypeError, newton, no_root, qb.array(1.0), args=1.0)
    raises(ValueError, newton, lambda x: x[:1], qb.array([1.0, 2.0]), match="shape")
    raises(ValueError, newton, no_root, qb.array(1.0), maxiter=0)
    raises(
        ValueError,
        qb.jvp,
        lambda a: newton(lambda x, a: x * x - a, qb.array(1.0), args=(a,)),
        (qb.array(2.0),),
        (qb.array(1.0),),
        match="forward-mode differentiation",
    )


# -- devices ----------------------------------------------------------------------


def test_optional_device_solvers_match_cpu():
    devices = [device for device, gate in (("mlx", "QUABLA_MLX_TEST"), ("cuda", "QUABLA_CUDA_TEST")) if os.environ.get(gate) == "1"]
    if not devices:
        return
    a32, n32 = spd(8).astype(qb.float32), nonsymmetric(8).astype(qb.float32)
    b32 = vector(8).astype(qb.float32)
    p32 = qb.array(2.0, dtype=qb.float32)

    def cg_loss(b, a):
        return qb.linalg.cg(matvec, b, args=(a,), tol=1e-6).sum()

    def gmres_loss(b, a):
        return qb.linalg.gmres(matvec, b, args=(a,), tol=1e-6).sum()

    def root_loss(p):
        x0 = qb.array([1.0, 1.0], dtype=qb.float32)
        return newton(circle_and_line, x0, args=(p, qb.array(0.5, dtype=qb.float32))).sum()

    cases = (
        (qb.value_and_grad(cg_loss, argnums=(0, 1)), (b32, a32)),
        (qb.value_and_grad(gmres_loss, argnums=(0, 1)), (b32, n32)),
        (qb.value_and_grad(root_loss), (p32,)),
    )
    for device in devices:
        for function, arguments in cases:
            expected = qb.tree.leaves(qb.jit(function)(*arguments))
            actual = qb.tree.leaves(qb.jit(function, device=device)(*arguments))
            for got, want in zip(actual, expected):
                assert_close(got, want, 1e-4)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
