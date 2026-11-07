"""Check native two-GPU sum/mean parity and timing observability."""
import hashlib
import json
import math
from pathlib import Path
import quabla as qb
from quabla import _quabla as native

results = []
for reduction, loss_factor, gradient in (('sum', 5.0, 5.0), ('mean', 2.5, 2.5)):
    program = native.tensor_value_and_grad_data_parallel_cuda_fn(
        lambda weight, x: (x * weight).mean(),
        [('weight', [1], qb.float32), ('x', [4], qb.float32)],
        ['weight'], ['x'], [0, 1], reduction)
    assert program.replica_count == 2 and program.device_ordinals == [0, 1]
    assert program.last_timing is None
    for weight in (2.0, 1.5, -0.25):
        loss, gradients = program({'weight': qb.array([weight], dtype=qb.float32),
                                   'x': qb.array([1., 2., 3., 4.], dtype=qb.float32)})
        assert loss.item() == weight * loss_factor, (reduction, loss)
        assert gradients['weight'].item() == gradient, (reduction, gradients)
        timing = program.last_timing
        assert set(timing) == {'replica_enqueue', 'collective', 'output_readback'}, timing
        assert all(math.isfinite(v) and v >= 0 for v in timing.values()), timing
        results.append(dict(reduction=reduction, weight=weight, loss=loss.item(), gradient=gradient, timing=timing))
print(json.dumps(dict(status='passed', checks=len(results), results=results,
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
