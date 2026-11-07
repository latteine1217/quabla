"""Seeded local NumPy comparisons on well-conditioned positive-definite systems."""

import hashlib
import json
from pathlib import Path

import numpy as np
import quabla as qb

rng = np.random.default_rng(1100)
checks = 0
for dtype, device, tolerance in ((qb.float64, "cpu", 1e-10), (qb.float32, "mlx", 8e-5)):
    for n in (1, 2, 4, 8):
        for case in range(8):
            matrix = rng.normal(size=(n, n))
            a, b = matrix.T @ matrix + np.eye(n), rng.normal(size=n)
            qa, qb_rhs = qb.array(a, dtype=dtype), qb.array(b, dtype=dtype)

            def reconstruct(a, b):
                result = qb.linalg.qr(a)
                return result.Q @ result.R

            for function, expected in (
                (lambda a, b: qb.linalg.solve(a, b), np.linalg.solve(a, b)),
                (lambda a, b: qb.linalg.cholesky(a), np.linalg.cholesky(a)),
                (lambda a, b: qb.linalg.inv(a), np.linalg.inv(a)),
                (reconstruct, a),
            ):
                result = qb.jit(function, device=device)(qa, qb_rhs)
                np.testing.assert_allclose(np.array(result.tolist()), expected,
                                           rtol=tolerance, atol=tolerance)
                checks += 1
print(json.dumps({
    "status": "passed", "seed": 1100, "checks": checks,
    "numpy_version": np.__version__,
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}))
