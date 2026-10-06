"""Iterative and implicit solvers: linalg.cg, linalg.gmres, and the Newton
root finder `quabla.newton`.

Solutions are checked against the dense `linalg.solve` and closed forms;
derivatives against the derivatives of the dense solve, analytic
implicit-function-theorem results, and central finite differences of the
iterative solve itself.
"""

import math

import quabla as qb
from quabla import newton

from _support import assert_close, devices, raises, run


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


def test_linear_solver_forward_mode_matches_the_dense_solve():
    # x_dot = A^-1 (b_dot - A_dot x) from the implicit rule, against forward
    # mode through the dense solve, for perturbations of b and of the
    # operator's parameters (a pytree), in both dtypes and under jit.
    for dtype, tolerance in ((qb.float64, 1e-10), (qb.float32, 2e-4)):
        b = vector(9, 1.0).astype(dtype)
        b_dot = vector(9, 6.0).astype(dtype)
        for solver, a in ((qb.linalg.cg, spd(9, 2.0)), (qb.linalg.gmres, nonsymmetric(9, 3.0))):
            a = a.astype(dtype)
            a_dot = matrix(9, 9, 8.0).astype(dtype)
            if solver is qb.linalg.cg:
                a_dot = a_dot + a_dot.T
            params = {"a": a, "s": qb.array(2.0, dtype=dtype)}
            params_dot = {"a": a_dot, "s": qb.array(-0.5, dtype=dtype)}
            tol = 1e-12 if dtype == qb.float64 else 1e-6

            def iterative(b, p, solver=solver, tol=tol):
                return solver(lambda x, p: matvec(x, p["a"]) * p["s"], b, args=(p,), tol=tol)

            def direct(b, p):
                return qb.linalg.solve(p["a"] * p["s"], b)

            expected = qb.jvp(direct, (b, params), (b_dot, params_dot))
            for transform in (lambda f: f, qb.jit):
                actual = transform(lambda b, p: qb.jvp(iterative, (b, p), (b_dot, params_dot)))(
                    b, params
                )
                for got, want in zip(actual, expected):
                    assert got.dtype == dtype
                    assert_close(got, want, tolerance)
            # The initial guess does not move the solution.
            x0 = qb.ones([9], dtype)
            _, tangent = qb.jvp(
                lambda x0: solver(matvec, b, args=(a,), x0=x0, tol=tol), (x0,), (b_dot,)
            )
            assert_close(tangent, qb.zeros([9], dtype), 0.0)


def test_linear_solver_forward_jacobian_and_second_derivatives():
    b = vector(6, 1.0)
    for solver, a in ((qb.linalg.cg, spd(6, 2.0)), (qb.linalg.gmres, nonsymmetric(6, 3.0))):

        def solve(b, a, solver=solver):
            return solver(matvec, b, args=(a,), tol=1e-13)

        # The forward-mode Jacobian (vmap of jvp) is A^-1.
        jacobian = qb.vmap(lambda t: qb.jvp(lambda b: solve(b, a), (b,), (t,))[1])(qb.eye(6))
        assert_close(jacobian.T, qb.linalg.inv(a), 1e-10)
        # Forward over reverse, reverse over forward, and forward over
        # forward match the dense solve.
        direction = matrix(6, 6, 9.0)
        if solver is qb.linalg.cg:
            direction = direction + direction.T

        def scaled(t, solve):
            return solve(b * (1.0 + t), a + t * direction)

        def dense(b, a):
            return qb.linalg.solve(a, b)

        def loss(t, solve):
            x = scaled(t, solve)
            return (x * x).sum()

        t = qb.array(0.3)
        one = qb.array(1.0)
        for second in (
            lambda f: qb.jvp(qb.grad(f), (t,), (one,))[1],
            lambda f: qb.grad(lambda t: qb.jvp(f, (t,), (one,))[1])(t),
            lambda f: qb.jvp(lambda t: qb.jvp(f, (t,), (one,))[1], (t,), (one,))[1],
        ):
            assert_close(
                second(lambda t: loss(t, solve)), second(lambda t: loss(t, dense)), 1e-9
            )
            assert_close(
                qb.jit(lambda t: second(lambda t: loss(t, solve)))(t),
                second(lambda t: loss(t, dense)),
                1e-9,
            )
        # The second derivative of the solution itself, x'' along the path.
        curvature = qb.jvp(
            lambda t: qb.jvp(lambda t: scaled(t, solve), (t,), (one,))[1], (t,), (one,)
        )[1]
        dense_curvature = qb.jvp(
            lambda t: qb.jvp(lambda t: scaled(t, dense), (t,), (one,))[1], (t,), (one,)
        )[1]
        assert_close(curvature, dense_curvature, 1e-9)
        # Central differences of the iterative tangent.
        def tangent(t, solve=solve):
            return qb.jvp(lambda t: loss(t, solve), (t,), (one,))[1]

        second_derivative = qb.jvp(tangent, (t,), (one,))[1].item()
        difference = central_difference(lambda t: tangent(qb.array(t)).item(), 0.3, 1e-5)
        assert abs(second_derivative - difference) < 1e-6 * max(1.0, abs(second_derivative))


def test_linear_solver_derivative_depth():
    # Every combination of two forward and reverse passes uses the implicit
    # rules (above); a third reverse pass still does, and a third pass that
    # involves forward mode raises instead of differentiating the iterations.
    a, b = spd(4), vector(4)

    def solution(t):
        return qb.linalg.cg(matvec, b, args=(a * t,), tol=1e-13).sum()

    def dense(t):
        return qb.linalg.solve(a * t, b).sum()

    t, one = qb.array(1.0), qb.array(1.0)
    third = qb.grad(qb.grad(qb.grad(solution)))(t)
    assert_close(third, qb.grad(qb.grad(qb.grad(dense)))(t), 1e-9)
    # A fourth reverse pass reaches the plain iteration.
    raises(
        ValueError,
        qb.grad(qb.grad(qb.grad(qb.grad(solution)))),
        t,
        match="while_loop",
    )
    for third_order in (
        lambda: qb.jvp(qb.grad(qb.grad(solution)), (t,), (one,)),
        lambda: qb.jvp(lambda t: qb.jvp(lambda t: qb.jvp(solution, (t,), (one,))[1], (t,), (one,))[1], (t,), (one,)),
    ):
        raises(ValueError, third_order, match="beyond its supported derivative order")


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


def test_root_forward_mode_matches_closed_forms():
    # dx = -J^-1 (df/dargs) dargs at the root, by the rule's tangent graph.
    def cube_root(a):
        return newton(lambda x, a: x * x * x - a, qb.array(1.0), args=(a,))

    a, one = qb.array(8.0), qb.array(1.0)
    value, tangent = qb.jvp(cube_root, (a,), (one,))
    assert_close(value, 2.0, 1e-15)
    assert_close(tangent, 1.0 / 12.0, 1e-14)
    # d^2 a^(1/3) = -2 a^(-5/3) / 9, in every combination of two passes.
    second = -2.0 / (9.0 * 32.0)
    assert_close(qb.jvp(lambda a: qb.jvp(cube_root, (a,), (one,))[1], (a,), (one,))[1], second, 1e-13)
    assert_close(qb.jvp(qb.grad(cube_root), (a,), (one,))[1], second, 1e-13)
    assert_close(qb.grad(lambda a: qb.jvp(cube_root, (a,), (one,))[1])(a), second, 1e-13)
    assert_close(qb.jit(lambda a: qb.jvp(cube_root, (a,), (one,)))(a)[1], 1.0 / 12.0, 1e-14)

    # A system: sum(x) = (1 + q) sqrt(p / (1 + q^2)); its forward Jacobian
    # (vmap of jvp) and forward-over-reverse Hessian match the closed form.
    x0 = qb.array([1.0, 0.5])

    def total(pq):
        return newton(circle_and_line, x0, args=(pq[0], pq[1])).sum()

    def closed_form(pq):
        p, q = pq[0], pq[1]
        return (1.0 + q) * (p / (1.0 + q * q)).sqrt()

    point = qb.array([2.0, 1.0])
    basis = qb.eye(2)
    forward = qb.vmap(lambda v: qb.jvp(total, (point,), (v,))[1])(basis)
    assert_close(forward, qb.grad(closed_form)(point), 1e-12)
    hvps = qb.vmap(lambda v: qb.jvp(qb.grad(total), (point,), (v,))[1])(basis)
    assert_close(hvps, qb.hessian(closed_form)(point), 1e-11)
    # Batched roots: vmap of the forward mode against per-example calls.
    points = qb.array([[2.0, 1.0], [0.5, -0.3], [3.0, 2.0]])
    batched = qb.vmap(lambda pq: qb.jvp(total, (pq,), (qb.array([1.0, -1.0]),))[1])(points)
    for index in range(3):
        expected = qb.jvp(total, (points[index],), (qb.array([1.0, -1.0]),))[1]
        assert_close(batched[index], expected, 1e-13)
    # The initial guess does not move the root.
    _, tangent = qb.jvp(lambda x0: newton(circle_and_line, x0, args=(point[0], point[1])), (x0,), (qb.ones([2]),))
    assert_close(tangent, [0.0, 0.0], 0.0)


# -- vmap -------------------------------------------------------------------------


def per_example(function, args, in_axes):
    """The meaning of `vmap`: unbatched calls on each example, stacked."""
    batch = next(arg.shape[0] for arg, axis in zip(args, in_axes) if axis is not None)
    results = [
        function(*[arg if axis is None else arg[index] for arg, axis in zip(args, in_axes)])
        for index in range(batch)
    ]
    return qb.tree.map(lambda *leaves: qb.stack(list(leaves), 0), *results)


def assert_tree_close(actual, expected, tolerance):
    actual, expected = qb.tree.leaves(actual), qb.tree.leaves(expected)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype and list(got.shape) == list(want.shape)
        assert_close(got, want, tolerance)


def linear_batch(dtype):
    """Four systems whose solves take very different iteration counts: a
    scaled identity (one iteration), two well-conditioned SPD matrices, and
    a zero right-hand side, which needs no iteration and whose masked body
    computes 0 / 0 while the others iterate; the last matrix is graded over
    three decades, the slowest."""
    n = 10
    graded = qb.diag(qb.array([10.0 ** (k / 3.0) for k in range(n)])) + 0.1 * spd(n, 2.0)
    matrices = qb.stack([3.0 * qb.eye(n), spd(n, 1.0), spd(n, 4.0), graded])
    rhs = qb.stack([vector(n, 0.3), vector(n, 1.3), qb.zeros([n]), vector(n, 2.3)])
    return matrices.astype(dtype), rhs.astype(dtype)


# GMRES with a short restart, so that its iteration counts spread as widely
# as CG's (one to over a hundred Arnoldi steps here).
LINEAR_SOLVERS = ((qb.linalg.cg, {}), (qb.linalg.gmres, {"restart": 5}))


def test_vmap_of_linear_solvers_matches_per_example_solves():
    for dtype, tol, tolerance in ((qb.float64, 1e-10, 1e-12), (qb.float32, 1e-5, 1e-5)):
        matrices, rhs = linear_batch(dtype)
        for solver, options in LINEAR_SOLVERS:

            def solve(b, a, solver=solver, options=options):
                return solver(matvec, b, args=(a,), tol=tol, info=True, **options)

            for in_axes, args in (
                ((0, 0), (rhs, matrices)),
                ((0, None), (rhs, matrices[3])),
                ((None, 0), (rhs[3], matrices)),
            ):
                expected = per_example(solve, args, in_axes)
                for transform in (lambda f: f, qb.jit):
                    x, info = transform(qb.vmap(solve, in_axes=in_axes))(*args)
                    assert_tree_close(x, expected[0], tolerance)
                    # Each example stops at its own tolerance and reports its
                    # own iterations, residual, and success.
                    assert info["iterations"].tolist() == expected[1]["iterations"].tolist()
                    assert info["success"].tolist() == expected[1]["success"].tolist()
                    assert_close(info["residual_norm"], expected[1]["residual_norm"], tolerance)
            x, info = qb.vmap(solve)(rhs, matrices)
            iterations = info["iterations"].tolist()
            assert iterations[0] == 1.0 and iterations[2] == 0.0, iterations
            assert len(set(iterations)) == 4 and all(info["success"].tolist()), iterations
            assert x[2].tolist() == [0.0] * 10
            # The solutions solve their own systems.
            dense = qb.stack([qb.linalg.solve(matrices[i], rhs[i]) for i in range(4)], 0)
            assert_close(x, dense, 100 * tol)


def test_vmap_and_grad_of_linear_solvers_compose():
    target = vector(10, 4.0)
    for dtype, tolerance in ((qb.float64, 1e-9), (qb.float32, 1e-4)):
        matrices, rhs = linear_batch(dtype)
        goal = target.astype(dtype)
        for solver, options in LINEAR_SOLVERS:

            def loss(b, a, solver=solver, options=options):
                tol = 1e-12 if dtype == qb.float64 else 1e-6
                x = solver(matvec, b, args=(a,), tol=tol, **options)
                return ((x - goal) * (x - goal)).sum()

            gradient = qb.grad(loss, argnums=(0, 1))
            expected = per_example(gradient, (rhs, matrices), (0, 0))
            # vmap of grad, and grad of a vmapped solve: per-example gradients.
            assert_tree_close(qb.vmap(gradient)(rhs, matrices), expected, tolerance)
            assert_tree_close(qb.jit(qb.vmap(gradient))(rhs, matrices), expected, tolerance)
            batched_total = qb.grad(lambda b, a: qb.vmap(loss)(b, a).sum(), argnums=(0, 1))
            assert_tree_close(batched_total(rhs, matrices), expected, tolerance)
            # A matrix shared by the batch receives the sum of the examples'
            # gradients.
            shared = qb.grad(
                lambda b, a: qb.vmap(loss, in_axes=(0, None))(b, a).sum(), argnums=1
            )(rhs, matrices[1])
            summed = per_example(lambda b: qb.grad(loss, argnums=1)(b, matrices[1]), (rhs,), (0,))
            assert_close(shared, summed.sum(axis=0), tolerance)
            # Against the dense solve's gradients.
            dense = per_example(
                qb.grad(
                    lambda b, a: ((qb.linalg.solve(a, b) - goal) ** 2).sum(), argnums=(0, 1)
                ),
                (rhs, matrices),
                (0, 0),
            )
            assert_tree_close(qb.vmap(gradient)(rhs, matrices), dense, 10 * tolerance)


def test_reverse_jacobian_and_hessian_of_linear_solvers_match_the_dense_solve():
    # Both batch the adjoint solve over the output cotangents, a vmapped
    # while_loop inside the solver's backward pass.
    b = vector(6, 1.0)
    for solver, a in ((qb.linalg.cg, spd(6, 2.0)), (qb.linalg.gmres, nonsymmetric(6, 3.0))):

        def solution(b, a, solver=solver):
            return solver(matvec, b, args=(a,), tol=1e-13)

        def loss(b, a, solver=solver):
            return (solution(b, a) ** 2).sum()

        for argnums in (0, 1):
            dense = qb.jacobian(lambda b, a: qb.linalg.solve(a, b), argnums=argnums)(b, a)
            assert_close(qb.jacobian(solution, argnums=argnums)(b, a), dense, 1e-10)
        dense = qb.hessian(lambda b, a: (qb.linalg.solve(a, b) ** 2).sum())(b, a)
        assert_close(qb.hessian(loss)(b, a), dense, 1e-10)
        assert_close(qb.jit(qb.hessian(loss))(b, a), dense, 1e-10)


def test_vmap_of_newton_matches_per_example_roots():
    for dtype, tolerance in ((qb.float64, 1e-13), (qb.float32, 1e-6)):
        # p = -1 has no real root: that example stalls and reports failure
        # while the others converge in their own iteration counts.
        p = qb.array([2.0, 0.5, 400.0, -1.0], dtype=dtype)
        q = qb.array([0.5, 1.0, -3.0, 0.2], dtype=dtype)
        starts = qb.array([[1.0, 1.0], [-1.0, -2.0], [5.0, 5.0], [0.3, 0.7]], dtype=dtype)

        def root(x0, p, q):
            return newton(circle_and_line, x0, args=(p, q), info=True)

        for in_axes, args in (
            ((None, 0, 0), (starts[0], p, q)),
            ((0, None, None), (starts, p[0], q[0])),
            ((0, 0, 0), (starts, p, q)),
        ):
            expected = per_example(root, args, in_axes)
            for transform in (lambda f: f, qb.jit):
                x, info = transform(qb.vmap(root, in_axes=in_axes))(*args)
                assert_tree_close(x, expected[0], tolerance)
                assert info["iterations"].tolist() == expected[1]["iterations"].tolist()
                assert info["success"].tolist() == expected[1]["success"].tolist()
                assert_close(info["residual_norm"], expected[1]["residual_norm"], tolerance)
        _, info = qb.vmap(root)(starts, p, q)
        assert info["success"].tolist() == [True, True, True, False]
        assert len(set(info["iterations"].tolist()[:3])) > 1, info["iterations"].tolist()


def test_vmap_and_grad_of_newton_compose():
    # sum(x) = (1 + q) sqrt(p / (1 + q^2)) at the root with y > 0.
    x0 = qb.array([1.0, 1.0])

    def total(p, q):
        return newton(circle_and_line, x0, args=(p, q)).sum()

    def closed_form(p, q):
        return (1.0 + q) * (p / (1.0 + q * q)).sqrt()

    p, q = qb.array([2.0, 0.5, 400.0]), qb.array([0.5, 1.0, 3.0])
    gradient = qb.grad(total, argnums=(0, 1))
    expected = per_example(qb.grad(closed_form, argnums=(0, 1)), (p, q), (0, 0))
    assert_tree_close(qb.vmap(gradient)(p, q), expected, 1e-11)
    assert_tree_close(qb.jit(qb.vmap(gradient))(p, q), expected, 1e-11)
    assert_tree_close(
        qb.grad(lambda p, q: qb.vmap(total)(p, q).sum(), argnums=(0, 1))(p, q), expected, 1e-11
    )
    # Second order composes too: the vmapped Hessian in (p, q).
    def stacked(pq, function):
        return function(pq[0], pq[1])

    points = qb.stack([p, q], 1)
    assert_tree_close(
        qb.vmap(qb.hessian(lambda pq: stacked(pq, total)))(points),
        per_example(qb.hessian(lambda pq: stacked(pq, closed_form)), (points,), (0,)),
        1e-9,
    )


def test_solvers_compose_with_odeint_and_loops_under_vmap():
    # A root whose parameter comes from an ODE solve (a fori_loop region):
    # p(k) = y(1) for y' = -k y, y(0) = 2. The gradient in k goes through the
    # solver's implicit rule and the loop's reverse rule, and matches the
    # closed-form root at the same integrated p. (A second reverse pass would
    # need the reverse mode of the loop's own reverse pass, which no loop
    # has; the Hessian of a solve alone composes, see above.)
    x0 = qb.array([1.0, 1.0])

    def decayed(k):
        return qb.ode.odeint(
            lambda y, t, k: -k * y, qb.array([2.0]), (0.0, 1.0), steps=16, args=(k,)
        )[0]

    def total(k):
        return newton(circle_and_line, x0, args=(decayed(k), qb.array(0.5))).sum()

    def closed_form(k):
        return 1.5 * (decayed(k) / 1.25).sqrt()

    rates = qb.array([0.2, 0.7, 1.5])
    expected = per_example(qb.grad(closed_form), (rates,), (0,))
    assert_tree_close(qb.vmap(qb.grad(total))(rates), expected, 1e-12)
    assert_tree_close(qb.jit(qb.vmap(qb.grad(total)))(rates), expected, 1e-12)
    # The Hessian of the integrated parameter itself is forward over reverse
    # through the loop, also under vmap.
    assert_tree_close(
        qb.vmap(qb.hessian(decayed))(rates), per_example(qb.hessian(decayed), (rates,), (0,)), 1e-12
    )

    # A forward-mode Jacobian through an ODE solve followed by a while loop,
    # under vmap: halve the state until its sum is below one.
    def halved(y0):
        y = qb.ode.odeint(lambda y, t: -0.5 * y, y0, (0.0, 1.0), steps=8)
        return qb.while_loop(lambda c: c.sum() > 1.0, lambda c: c * 0.5, y)

    starts = qb.array([[3.0, 4.0], [0.5, 0.25], [40.0, 1.0]])
    assert_tree_close(
        qb.vmap(qb.jacobian(halved))(starts), per_example(qb.jacobian(halved), (starts,), (0,)), 1e-12
    )


# -- devices ----------------------------------------------------------------------


def test_optional_device_solvers_match_cpu():
    targets = devices()
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
        # Forward mode (the implicit rules' tangent graphs) and forward over
        # reverse.
        (lambda b, a: qb.jvp(cg_loss, (b, a), (b * 0.5, a * 0.1)), (b32, a32)),
        (lambda b, a: qb.jvp(gmres_loss, (b, a), (b * 0.5, a * 0.1)), (b32, n32)),
        (lambda p: qb.jvp(root_loss, (p,), (qb.ones_like(p),)), (p32,)),
        (lambda p: qb.jvp(qb.grad(root_loss), (p,), (qb.ones_like(p),)), (p32,)),
    )
    for device in targets:
        for function, arguments in cases:
            expected = qb.tree.leaves(qb.jit(function)(*arguments))
            actual = qb.tree.leaves(qb.jit(function, device=device)(*arguments))
            for got, want in zip(actual, expected):
                assert_close(got, want, 1e-4)


def test_optional_device_vmap_of_solvers_matches_cpu():
    precisions = {"mlx": (None,), "cuda": (None, "float64")}
    runs = [(device, precision) for device in devices(*precisions) for precision in precisions[device]]
    for device, precision in runs:
        dtype = qb.float64 if precision == "float64" else qb.float32
        tolerance = 1e-10 if precision == "float64" else 1e-4
        tol = 1e-10 if precision == "float64" else 1e-5
        # Without the graded system: its condition number (about 1e3) turns
        # float32 rounding differences into solution differences near the
        # tolerance, which says nothing about the device.
        matrices, rhs = (value[:3] for value in linear_batch(dtype))
        p = qb.array([2.0, 0.5, 400.0, -1.0], dtype=dtype)
        q = qb.array([0.5, 1.0, -3.0, 0.2], dtype=dtype)
        starts = qb.array([[1.0, 1.0], [-1.0, -2.0], [5.0, 5.0], [0.3, 0.7]], dtype=dtype)
        x0 = qb.array([1.0, 1.0], dtype=dtype)

        def reported(result):
            # The iteration count may differ by one where a device's rounding
            # moves a residual across the threshold, so it is checked apart.
            x, info = result
            return (x, info["residual_norm"], info["success"]), info["iterations"]

        operations = {}
        for solver, options in LINEAR_SOLVERS:

            def solve(b, a, solver=solver, options=options):
                return reported(solver(matvec, b, args=(a,), tol=tol, info=True, **options))

            def loss(b, a, solver=solver, options=options):
                return (solver(matvec, b, args=(a,), tol=tol, **options) ** 2).sum()

            name = solver.__name__
            operations[name] = (qb.vmap(solve), (rhs, matrices))
            operations[f"{name} shared matrix"] = (
                qb.vmap(solve, in_axes=(0, None)),
                (rhs, matrices[1]),
            )
            operations[f"{name} vmap grad"] = (
                qb.vmap(qb.grad(loss, argnums=(0, 1))),
                (rhs, matrices),
            )
        operations["newton"] = (
            qb.vmap(
                lambda x0, p, q: reported(newton(circle_and_line, x0, args=(p, q), info=True))
            ),
            # Without p = -1: where a solve without a root stalls depends on
            # rounding.
            (starts[:3], p[:3], q[:3]),
        )
        operations["newton grad"] = (
            qb.grad(
                lambda p, q: qb.vmap(
                    lambda p, q: newton(circle_and_line, x0, args=(p, q)).sum()
                )(p, q).sum(),
                argnums=(0, 1),
            ),
            (p[:3], q[:3]),
        )
        for name, (operation, arguments) in operations.items():
            expected = qb.jit(operation)(*arguments)
            options = {"device": device}
            if precision is not None:
                options["precision"] = precision
            actual = qb.jit(operation, **options)(*arguments)
            if "grad" not in name:
                assert_close(actual[1], expected[1], 1.0)
                actual, expected = actual[0], expected[0]
            for got, want in zip(qb.tree.leaves(actual), qb.tree.leaves(expected)):
                assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
                assert_close(got, want, tolerance)


if __name__ == "__main__":
    run(globals())
