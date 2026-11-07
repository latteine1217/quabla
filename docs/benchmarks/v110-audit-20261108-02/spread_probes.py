"""Analytical reduction composition checks at binary exponent boundaries."""
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
for dtype, bits, exponents in (
    (qb.float32, 23, [-50, -20, 0, 20, 50, 100]),
    (qb.float64, 52, [-500, -250, 0, 250, 500, 1000]),
):
    if args.device == 'mlx' and dtype == qb.float64:
        continue
    for exponent in exponents:
        base = math.ldexp(1.0, exponent)
        delta = math.ldexp(1.0, exponent - bits)
        spread = delta * 0.5
        x = qb.array([base, base + delta], dtype=dtype)
        actual = qb.jit(qb.std, device=args.device)(x).item()
        assert math.isclose(actual, spread, rel_tol=1e-6), (dtype, exponent, actual, spread)
        checks += 1
        gradient = qb.jit(qb.grad(qb.std), device=args.device)(x).to_flat_list()
        assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(gradient, [-0.5, 0.5])), (dtype, exponent, gradient)
        checks += 1
        variance = spread * spread
        maximum = 3.4028234663852886e38 if dtype == qb.float32 else 1.7976931348623157e308
        if 0.0 < variance <= maximum:
            actual = qb.jit(qb.var, device=args.device)(x).item()
            assert math.isclose(actual, variance, rel_tol=1e-6), (dtype, exponent, actual, variance)
            checks += 1
        rows = qb.stack([x, x])
        actual = qb.jit(qb.vmap(qb.std), device=args.device)(rows).to_flat_list()
        assert all(math.isclose(v, spread, rel_tol=1e-6) for v in actual), actual
        checks += 1
        def directional(x):
            return qb.jvp(qb.mean, (x,), (qb.ones_like(x),))[1]
        assert qb.jit(directional, device=args.device)(x).item() == 1.0
        checks += 1
print(json.dumps(dict(status='passed', device=args.device, checks=checks,
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
