"""Independent bounded correctness probes for CPU, MLX, and CUDA."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import quabla as qb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    options = parser.parse_args()
    device = options.device
    checks = 0

    def check(actual, expected, tolerance=3e-5):
        nonlocal checks
        values = actual.to_flat_list()
        assert len(values) == len(expected), (device, values, expected)
        absolute = tolerance * max(abs(value) for value in expected)
        assert all(math.isclose(a, b, rel_tol=tolerance, abs_tol=absolute)
                   for a, b in zip(values, expected)), (device, values, expected)
        checks += 1

    dtype = qb.float32
    xp = qb.array([-3e38, 3e38], dtype=dtype)
    fp = qb.array([0.0, 1.0], dtype=dtype)
    check(qb.jit(qb.interp, device=device)(qb.array([-3e38, 0.0, 3e38], dtype=dtype), xp, fp),
          [0.0, 0.5, 1.0])
    integral = qb.jit(qb.trapezoid, device=device)
    check(integral(qb.array([3e-38, 3e-38], dtype=dtype), xp), [18.0])
    check(integral(qb.array([2e38, 2e38], dtype=dtype), qb.array([0.0, 1e-30], dtype=dtype)), [2e8])

    def nested_loss(p, xs):
        def model(x):
            return p[0] * qb.sin(p[1] * x)
        curvature = qb.vmap(qb.grad(qb.grad(model)))(xs)
        return (curvature * curvature).sum()

    differentiate = qb.jit(qb.value_and_grad(nested_loss), device=device)
    for seed in (103, 1701, 65537, 8675309):
        rng = random.Random(seed)
        for count in (1, 2, 4, 8):
            points = [rng.uniform(-1.0, 1.0) for _ in range(count)]
            p0, p1 = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)
            s = sum(math.sin(p1 * x) ** 2 for x in points)
            ds = sum(2.0 * x * math.sin(p1 * x) * math.cos(p1 * x) for x in points)
            value, gradient = differentiate(qb.array([p0, p1], dtype=dtype),
                                            qb.array(points, dtype=dtype))
            check(value, [p0**2 * p1**4 * s])
            check(gradient, [2.0 * p0 * p1**4 * s,
                             p0**2 * (4.0 * p1**3 * s + p1**4 * ds)])

            diagonal = [float(i + 1) for i in range(count)]
            for scale in (1e-3, 1.0, 1e3):
                rhs = qb.array([scale * x for x in points], dtype=dtype)
                diagonal_array = qb.array(diagonal, dtype=dtype)
                for solver in (qb.linalg.cg, qb.linalg.gmres):
                    def solve(b, d):
                        return solver(lambda x, d: x * d, b, args=(d,), info=True)
                    x, info = qb.jit(solve, device=device)(rhs, diagonal_array)
                    assert bool(info["success"]), (device, seed, count, solver, info)
                    check(x, [scale * value / d for value, d in zip(points, diagonal)], 2e-4)

    print(json.dumps({
        "status": "passed", "device": device, "checks": checks,
        "seeds": [103, 1701, 65537, 8675309], "package_version": qb.__version__,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
