"""Fixed-seed mean/spread comparisons against a high-precision Decimal oracle."""
import argparse
import decimal
import hashlib
import json
import math
import random
from pathlib import Path
import quabla as qb

parser = argparse.ArgumentParser()
parser.add_argument('--device', required=True)
args = parser.parse_args()
checks = 0
skipped = 0
rng = random.Random(1701)
functions = {f.__name__: qb.jit(f, device=args.device, static_argnames=(() if f is qb.mean else ('ddof',))) for f in (qb.mean, qb.var, qb.std)}
cases = [(qb.float32, 23, [-100, -50, 0, 50, 100])]
if args.device != 'mlx':
    cases.append((qb.float64, 52, [-1000, -500, 0, 500, 1000]))
with decimal.localcontext() as context:
    context.prec = 128
    for dtype, bits, exponents in cases:
        tolerance = 2e-5 if dtype == qb.float32 else 2e-12
        minimum = 1.1754943508222875e-38 if dtype == qb.float32 else 2.2250738585072014e-308
        subnormal = math.ldexp(1.0, -149 if dtype == qb.float32 else -1074)
        maximum = 3.4028234663852886e38 if dtype == qb.float32 else 1.7976931348623157e308
        for exponent in exponents:
            base = math.ldexp(1.0, exponent)
            for count in (2, 3, 7, 32):
                for narrow in (False, True):
                    values = ([base * (1.0 + rng.randrange(16) * math.ldexp(1.0, -bits)) for _ in range(count)]
                              if narrow else [base * rng.uniform(-1.0, 1.0) for _ in range(count)])
                    x = qb.array(values, dtype=dtype)
                    exact = [decimal.Decimal.from_float(v) for v in x.to_flat_list()]
                    mean = sum(exact) / count
                    for ddof in (0, 1):
                        variance = sum((v - mean) ** 2 for v in exact) / (count - ddof)
                        for name, expected in (('mean', float(mean)), ('var', float(variance)), ('std', float(variance.sqrt()))):
                            if args.device == 'mlx' and 0 < abs(expected) < minimum:
                                skipped += 1
                                continue
                            if abs(expected) > maximum:
                                expected = math.copysign(math.inf, expected)
                            actual = functions[name](x, ddof=ddof).item() if name != 'mean' else functions[name](x).item()
                            if math.isinf(expected):
                                assert actual == expected, (dtype, exponent, count, narrow, ddof, name, actual, expected)
                            else:
                                assert abs(actual - expected) <= max(tolerance * abs(expected), 8 * subnormal), (dtype, exponent, count, narrow, ddof, name, actual, expected)
                            checks += 1
print(json.dumps(dict(status='passed', device=args.device, checks=checks, skipped=skipped,
                      seed=1701, decimal_precision=128,
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
