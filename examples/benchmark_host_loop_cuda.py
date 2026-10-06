"""Measure the per-iteration cost of host-driven CUDA loops.

A loop body that slices, concatenates or reduces across lanes cannot run as
one fused per-lane kernel, so CUDA runs it as a host-driven region loop: the
host launches the body's device program once per iteration. This script
times `fori_loop`, `scan` and `while_loop` with such a body (a rotation by
slice/concat plus a full reduction), a `fori_loop` body that adds a cuBLAS
matrix product, the reverse passes of `fori_loop` and `scan`, and a
Hessian-vector product through `scan`, at two trip counts. The difference
divided by the difference in iterations is the per-iteration cost, without
the fixed transfer and launch cost of one call.

Run on Linux with a CUDA driver and NVRTC available:
    maturin develop --release --features cuda
    python examples/benchmark_host_loop_cuda.py [--dump results.json]

`--dump` writes every result as exact float hex strings, so two builds can be
compared bit for bit (`--compare old.json`).
"""

import argparse
import json
import statistics
import time

import quabla as qb


def rotate(c, s):
    # Slice/concat rotation plus a full reduction: not lane-local.
    rolled = qb.concat([c[1:], c[:1]], 0)
    return rolled * s + c.sum() * 1e-3


def fori(steps):
    def run(x, s):
        return qb.fori_loop(
            0, steps, lambda i, c, s: rotate(c, s) + i * 1e-4, x, operands=(s,)
        )

    return run


def scan(steps):
    def run(x, s):
        def body(c, i, s):
            nxt = rotate(c, s) + i * 1e-4
            return nxt, nxt[:4]

        return qb.scan(body, x, length=steps, operands=(s,))

    return run


def fori_matmul(steps):
    def run(x, s):
        # `--lanes` must be a square: the carry and the operand are viewed as matrices.
        side = int(round(x.shape[0] ** 0.5))

        def body(i, c, s):
            mixed = (c.reshape([side, side]) @ s.reshape([side, side])).reshape(
                [side * side]
            )
            return (rotate(c, s) + mixed * 0.1).tanh()

        return qb.fori_loop(0, steps, body, x, operands=(s,))

    return run


def while_(steps):
    def run(x, s):
        # Slot 0 counts iterations; the predicate stops after `steps` of them.
        limit = float(steps)
        counted = qb.concat([qb.zeros([1], qb.float32), x], 0)

        def body(c, s):
            return qb.concat([c[:1] + 1.0, rotate(c[1:], s)], 0)

        return qb.while_loop(lambda c, s: c[0] < limit, body, counted, operands=(s,))

    return run


def fori_grad(steps):
    loop = fori(steps)
    return qb.value_and_grad(lambda x, s: (loop(x, s) ** 2).sum(), argnums=(0, 1))


def scan_grad(steps):
    loop = scan(steps)

    def loss(x, s):
        carry, outputs = loop(x, s)
        return (carry**2).sum() + (outputs**2).sum()

    return qb.value_and_grad(loss, argnums=(0, 1))


def fori_matmul_grad(steps):
    loop = fori_matmul(steps)
    return qb.value_and_grad(lambda x, s: (loop(x, s) ** 2).sum(), argnums=(0, 1))


def scan_hvp(steps):
    loop = scan(steps)

    def loss(x, s):
        carry, outputs = loop(x, s)
        return (carry**2).sum() + (outputs**2).sum()

    gradient = qb.grad(loss, argnums=1)
    return lambda x, s: qb.jvp(gradient, (x, s), (qb.zeros_like(x), qb.ones_like(s)))[1]


CASES = {
    "fori": fori,
    "scan": scan,
    "while": while_,
    "fori_matmul": fori_matmul,
    "fori_grad": fori_grad,
    "scan_grad": scan_grad,
    "fori_matmul_grad": fori_matmul_grad,
    "scan_hvp": scan_hvp,
}


def leaves(result):
    flat, _ = qb.tree.flatten(result)
    return [[float(value).hex() for value in leaf.to_flat_list()] for leaf in flat]


def median_seconds(function, args, repeats):
    function(*args)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--short", type=int, default=100)
    parser.add_argument("--long", type=int, default=1100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--dump", help="write exact results to this JSON file")
    parser.add_argument(
        "--compare",
        help="assert the results this JSON file holds are equal bit for bit",
    )
    options = parser.parse_args()

    n = options.lanes
    x = qb.array([((i % 13) - 6) / 13.0 for i in range(n)], dtype=qb.float32)
    s = qb.array([0.995 + (i % 7) * 1e-3 for i in range(n)], dtype=qb.float32)
    report = {}
    exact = {}
    for name in options.cases.split(","):
        timings = {}
        for steps in (options.short, options.long):
            function = qb.jit(CASES[name](steps), device=options.device)
            timings[steps] = median_seconds(function, (x, s), options.repeats)
            exact[f"{name}/{steps}"] = leaves(function(x, s))
        per_iteration = (timings[options.long] - timings[options.short]) / (
            options.long - options.short
        )
        report[name] = {
            "short_ms": timings[options.short] * 1e3,
            "long_ms": timings[options.long] * 1e3,
            "us_per_iteration": per_iteration * 1e6,
        }
        print(json.dumps({"case": name, "lanes": n, **report[name]}))
    if options.dump:
        with open(options.dump, "w") as handle:
            json.dump(exact, handle)
    if options.compare:
        with open(options.compare) as handle:
            expected = json.load(handle)
        compared = [key for key in exact if key in expected]
        for key in compared:
            assert expected[key] == exact[key], f"{key} differs from {options.compare}"
        print(f"bit-identical to {options.compare}: {', '.join(compared)}")


if __name__ == "__main__":
    main()
