"""Check complete two-GPU loss/gradient calls against CPU replica oracles."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import quabla


DEVICE_ORDINALS = [0, 1]
TOLERANCE = 1e-5
STEPS = 5


def loss_fn(x, target, weight, bias):
    return ((x * weight + bias - target).powi(2)).mean()


def flat(tensor):
    return tensor.to_flat_list()


def maximum_error(actual, reference):
    assert actual.shape == reference.shape
    pairs = list(zip(flat(actual), flat(reference), strict=True))
    assert all(math.isfinite(a) and math.isfinite(b) for a, b in pairs)
    return max(abs(a - b) for a, b in pairs)


def validate_case(shape, parameter_shape, dtype, reduction):
    count = math.prod(shape)
    parameter_count = math.prod(parameter_shape)
    inputs = {
        "x": quabla.Tensor(
            shape, [(i % 17 - 8) / 16 for i in range(count)], dtype=dtype
        ),
        "target": quabla.Tensor(
            shape, [(i % 13 - 6) / 16 for i in range(count)], dtype=dtype
        ),
        "weight": quabla.Tensor(parameter_shape, [0.25] * parameter_count, dtype=dtype),
        "bias": quabla.Tensor(parameter_shape, [-0.125] * parameter_count, dtype=dtype),
    }
    specs = [(name, tensor.shape, dtype) for name, tensor in inputs.items()]
    cpu_plan = quabla.trace_tensor(loss_fn, specs).output.compile_cpu()
    cuda_fn = quabla.tensor_value_and_grad_data_parallel_cuda_fn(
        loss_fn, specs, ["weight", "bias"], ["x", "target"], DEVICE_ORDINALS, reduction
    )
    assert cuda_fn.replica_count == len(DEVICE_ORDINALS)
    assert cuda_fn.device_ordinals == DEVICE_ORDINALS
    results = []
    for step in range(STEPS):
        cpu_value, cpu_gradients = cpu_plan.value_and_grad_data_parallel(
            inputs, ["x", "target"], len(DEVICE_ORDINALS), reduction
        )
        cuda_value, cuda_gradients = cuda_fn(inputs)
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
        timing = cuda_fn.last_timing
        assert timing is not None
        assert set(timing) == {"replica_enqueue", "collective", "output_readback"}
        assert all(
            math.isfinite(duration) and duration >= 0.0 for duration in timing.values()
        )
        results.append(
            {
                "step": step,
                "cpu_loss": flat(cpu_value)[0],
                "cuda_loss": flat(cuda_value)[0],
                "loss_abs_error": loss_error,
                "gradient_abs_errors": gradient_errors,
                "timing_seconds": timing,
            }
        )
        # Use the CPU oracle update for common inputs: each subsequent call
        # tests changed parameters without conflating accumulated optimizer drift.
        for name in ("weight", "bias"):
            inputs[name] = quabla.Tensor(
                parameter_shape,
                [
                    value - 0.01 * gradient
                    for value, gradient in zip(
                        flat(inputs[name]), flat(cpu_gradients[name]), strict=True
                    )
                ],
                dtype=dtype,
            )
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
        for dtype in (quabla.float64, quabla.float32)
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
