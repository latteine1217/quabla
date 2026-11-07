"""Bounded host-returning reduction timing, not resident training performance."""
import json
import platform
import statistics
import time

import quabla as qb
import quabla._quabla as native

results = []
x = qb.array([[1.0 + (i % 13) / 10 for i in range(64)] for _ in range(64)], dtype=qb.float32)
for device in ('cpu', 'mlx'):
    for function in (qb.mean, qb.var, qb.std):
        execute = qb.jit(function, device=device)
        start = time.perf_counter_ns()
        value = execute(x).item()
        first = time.perf_counter_ns() - start
        for _ in range(20):
            execute(x).item()
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            for _ in range(100):
                execute(x).item()
            samples.append((time.perf_counter_ns() - start) / 100)
        results.append(dict(device=device, operation=function.__name__, value=value,
                            first_call_ns=first, samples_ns=samples, median_ns=statistics.median(samples)))
print(json.dumps(dict(python=platform.python_version(), platform=platform.platform(),
                     package_path=qb.__file__, native_path=native.__file__,
                     contract='float32 [64,64], precreated host input, jit, scalar host return and item synchronization; 7 batches of 100 after 20 warmups',
                     results=results), indent=2))
