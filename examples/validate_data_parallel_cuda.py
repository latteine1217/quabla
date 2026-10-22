"""Check complete two-GPU loss/gradient calls against CPU replica oracles.

`distributed.value_and_grad` splits the batch across CUDA ordinals 0 and 1
and reduces the shard losses and gradients with NCCL. The oracle evaluates
the same two shards on the CPU with `value_and_grad` and combines them with
the same reduction.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import quabla
import quabla as qb


DEVICE_ORDINALS = [0, 1]
TOLERANCE = 1e-5
STEPS = 5


def loss_fn(params, x, target):
    return qb.mean((x * params["weight"] + params["bias"] - target) ** 2)


def maximum_error(actual, reference):
    assert actual.shape == reference.shape
    pairs = list(
        zip(actual.reshape(-1).tolist(), reference.reshape(-1).tolist(), strict=True)
    )
    assert all(math.isfinite(a) and math.isfinite(b) for a, b in pairs)
    return max(abs(a - b) for a, b in pairs)


def cpu_replicas(params, x, target, reduction):
    """The CPU value and gradients of `len(DEVICE_ORDINALS)` equal shards."""
    replicas = len(DEVICE_ORDINALS)
    size = x.shape[0] // replicas
    value_and_grad = qb.value_and_grad(loss_fn)
    results = [
        value_and_grad(
            params,
            x.slice(0, index * size, size).to_tensor(),
            target.slice(0, index * size, size).to_tensor(),
        )
        for index in range(replicas)
    ]
    factor = 1.0 / replicas if reduction == "mean" else 1.0
    value = sum((result[0] for result in results[1:]), results[0][0]) * factor
    gradients = {
        name: sum((result[1][name] for result in results[1:]), results[0][1][name])
        * factor
        for name in params
    }
    return value, gradients


def validate_case(shape, parameter_shape, dtype, reduction):
    count = math.prod(shape)
    x = qb.array([(i % 17 - 8) / 16 for i in range(count)], dtype=dtype).reshape(shape)
    target = qb.array([(i % 13 - 6) / 16 for i in range(count)], dtype=dtype).reshape(
        shape
    )
    params = {
        "weight": qb.full(parameter_shape, 0.25, dtype),
        "bias": qb.full(parameter_shape, -0.125, dtype),
    }
    cuda_fn = qb.distributed.value_and_grad(
        loss_fn,
        devices=[f"cuda:{ordinal}" for ordinal in DEVICE_ORDINALS],
        shard_argnums=(1, 2),
        reduction=reduction,
    )
    results = []
    for step in range(STEPS):
        cpu_value, cpu_gradients = cpu_replicas(params, x, target, reduction)
        cuda_value, cuda_gradients = cuda_fn(params, x, target)
        assert set(cuda_gradients) == {"weight", "bias"}
        loss_error = maximum_error(cuda_value, cpu_value)
        gradient_errors = {
            name: maximum_error(cuda_gradients[name], cpu_gradients[name])
            for name in ("weight", "bias")
        }
        assert loss_error <= TOLERANCE, (shape, dtype.name, reduction, step, loss_error)
        assert all(error <= TOLERANCE for error in gradient_errors.values()), (
            gradient_errors
        )
        results.append(
            {
                "step": step,
                "cpu_loss": cpu_value.item(),
                "cuda_loss": cuda_value.item(),
                "loss_abs_error": loss_error,
                "gradient_abs_errors": gradient_errors,
            }
        )
        # Use the CPU oracle update for common inputs: each subsequent call
        # tests changed parameters without conflating accumulated optimizer drift.
        params = {
            name: params[name] - 0.01 * cpu_gradients[name] for name in ("weight", "bias")
        }
    return {
        "shape": shape,
        "parameter_shape": parameter_shape,
        "dtype": dtype.name,
        "reduction": reduction,
        "steps": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [
        validate_case(shape, parameter_shape, dtype, reduction)
        for shape, parameter_shape in (
            ([8], [1]),
            ([10, 3], [1, 3]),
            ([8, 2, 3], [1, 2, 3]),
            ([1024, 2], [1, 2]),
        )
        # Data parallelism runs float32 programs only (float64 raises).
        for dtype in (quabla.float32,)
        for reduction in ("sum", "mean")
    ]
    extension = Path(quabla._quabla.__file__)
    report = {
        "status": "passed",
        "tolerance": TOLERANCE,
        "device_ordinals": DEVICE_ORDINALS,
        "extension_path": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "results": results,
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
