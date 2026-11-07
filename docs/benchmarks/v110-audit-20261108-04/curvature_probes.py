"""Analytical variance and standard-deviation Hessian-vector products."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import quabla as qb

parser = argparse.ArgumentParser()
parser.add_argument('--device', required=True)
args = parser.parse_args()
checks = 0
cases = [(qb.float32, [1e-15, 1.0, 1e15, 1e30])]
if args.device != 'mlx':
    cases.append((qb.float64, [1e-150, 1.0, 1e150, 1e308]))
for dtype, scales in cases:
    for scale in scales:
        x = qb.array([-scale, 0.0, scale], dtype=dtype)
        direction = qb.array([1.0, -2.0, 1.0], dtype=dtype)
        def curvature(x, direction):
            return qb.jvp(qb.grad(qb.std), (x,), (direction,))[1]
        actual = qb.jit(curvature, device=args.device)(x, direction).to_flat_list()
        reference = (1.0 / scale) / (3.0 * math.sqrt(2.0 / 3.0))
        assert all(math.isclose(v / reference, w, rel_tol=2e-5) for v, w in zip(actual, [1, -2, 1])), (dtype, scale, actual)
        checks += 1
        maximum_scale = 1e15 if dtype == qb.float32 else 1e150
        if scale <= maximum_scale:
            def variance_curvature(x, direction):
                return qb.jvp(qb.grad(qb.var), (x,), (direction,))[1]
            actual = qb.jit(variance_curvature, device=args.device)(x, direction).to_flat_list()
            assert all(math.isclose(v, w, rel_tol=2e-5) for v, w in zip(actual, [2 / 3, -4 / 3, 2 / 3])), (dtype, scale, actual)
            checks += 1
print(json.dumps(dict(status='passed', device=args.device, checks=checks,
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
