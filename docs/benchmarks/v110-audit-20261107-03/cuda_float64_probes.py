"""Direct CUDA float64 checks for the newly repaired numerical boundaries."""

import hashlib
import json
import math
from pathlib import Path

import quabla as qb

device = "cuda:0"
checks = 0
x = qb.array([1e200, -1e200, 1e308, -1e308])


def loss(x):
    return qb.gelu(x).sum()


gradient = qb.grad(loss)
for function, expected in (
    (qb.gelu, [1e200, 0.0, 1e308, 0.0]),
    (gradient, [1.0, 0.0, 1.0, 0.0]),
    (qb.grad(lambda x: gradient(x).sum()), [0.0] * 4),
):
    actual = qb.jit(function, device=device)(x).to_flat_list()
    assert actual == expected, (actual, expected)
    checks += 1

xp, fp = qb.array([-1e308, 1e308]), qb.array([0.0, 1.0])


def interpolate(x):
    return qb.interp(x, xp, fp)


for function, expected in ((interpolate, 0.5), (qb.grad(interpolate), 5e-309)):
    actual = qb.jit(function, device=device)(qb.array(0.0)).item()
    assert math.isclose(actual, expected, rel_tol=1e-12), (actual, expected)
    checks += 1

for y, t, expected in (
    (qb.array([1e308, 1e308]), qb.array([0.0, 1e-308]), 1.0),
    (qb.array([1e-308, 1e-308]), xp, 2.0),
):
    actual = qb.jit(qb.trapezoid, device=device)(y, t).item()
    assert math.isclose(actual, expected, rel_tol=1e-12), (actual, expected)
    checks += 1

integral_gradient = qb.jit(qb.grad(lambda y: qb.trapezoid(y, xp)), device=device)
assert integral_gradient(qb.array([1e-308, 1e-308])).to_flat_list() == [1e308, 1e308]
checks += 1

for method in ("euler", "heun", "rk4"):
    def solve(y):
        return qb.ode.odeint(lambda y, t: y * 0.0 + 1e-308, y,
                            (-1e308, 1e308), steps=16, method=method)

    actual = qb.jit(solve, device=device)(qb.array([0.0])).item()
    assert math.isclose(actual, 2.0, rel_tol=1e-12), (method, actual)
    checks += 1

print(json.dumps({
    "status": "passed", "device": device, "dtype": "float64", "checks": checks,
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}))
