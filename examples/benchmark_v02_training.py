"""Compare native retained Adam and v0.2 Trainer on a fixed Poisson PINN.

Run on a release MLX or CUDA build:
    python examples/benchmark_v02_training.py --device mlx --max-step-ratio 1.02
    python examples/benchmark_v02_training.py --device cuda:0 --max-step-ratio 1.02

Both plans use u(x) = sin(x*w), the model from benchmark_pinn_mlx.py,
with its exact second derivative -w*w*sin(x*w). The identical expression
in both traces isolates Trainer overhead from differentiation graph choices.
All data, parameters, and Adam moments stay retained on the device. Timed
batches alternate execution order to reduce thermal and scheduling bias.
The throughput gate uses complete batches including one loss readback that
synchronizes CUDA, amortized over all steps in the batch. Submission and
readback times are also reported separately; submission alone is not GPU
execution time. This script does not write or replace any baseline artifact.
"""

import argparse
import json
import math
import platform
import statistics
import sys
import time

import quabla as qb
from quabla import _quabla as native
from quabla._devices import require_device


def make_problem(collocation_count):
    coordinates = [
        (index + 0.5) / collocation_count for index in range(collocation_count)
    ]
    forcing = [math.pi**2 * math.sin(math.pi * x) for x in coordinates]
    count = collocation_count + 2

    def tensor(values):
        return qb.array(values, dtype=qb.float32).reshape([count, 1])

    data = (
        tensor(coordinates + [0.0, 1.0]),
        tensor(forcing + [0.0, 0.0]),
        tensor([0.0] * count),
        tensor([0.0] * collocation_count + [1.0, 1.0]),
    )
    return {"weight": qb.array([[2.5]], dtype=qb.float32)}, data


def poisson_loss(params, x, forcing, boundary_target, boundary_mask):
    weight = params["weight"]
    prediction = (x * weight).sin()
    residual = -(weight * weight) * prediction + forcing
    boundary_error = prediction - boundary_target
    error = native.where(boundary_mask.gt(0.0), boundary_error, residual)
    return (error * error).mean()


def build_native(params, data, device, learning_rate):
    target, ordinal = require_device(device, "benchmark_v02_training")
    names = ["weight", "x", "forcing", "boundary_target", "boundary_mask"]
    values = [params["weight"], *data]
    traced = native.trace_tensor(
        lambda weight, x, forcing, boundary_target, boundary_mask: poisson_loss(
            {"weight": weight}, x, forcing, boundary_target, boundary_mask
        ),
        [(name, value.shape, value.dtype) for name, value in zip(names, values)],
    )
    kwargs = {"device_ordinal": ordinal} if target == "cuda" else {}
    return getattr(native, f"{target}_adam_loss_optimizer")(
        traced.output,
        ["weight"],
        dict(zip(names, values)),
        learning_rate,
        names[1:],
        **kwargs,
    )


def snapshot(executor, is_trainer):
    started = time.perf_counter()
    loss = executor.loss().to_flat_list()[0]
    loss_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parameters = executor.params if is_trainer else executor.parameters()
    weight = parameters["weight"].to_flat_list()[0]
    parameter_seconds = time.perf_counter() - started
    return {"loss": loss, "weight": weight}, {
        "loss_seconds": loss_seconds,
        "parameter_seconds": parameter_seconds,
    }


def timed_batch(executor, steps):
    started = time.perf_counter()
    for _ in range(steps):
        executor.step()
    submitted = time.perf_counter()
    loss = executor.loss().to_flat_list()[0]
    synchronized = time.perf_counter()
    return {
        "synchronized_seconds_per_step": (synchronized - started) / steps,
        "submission_seconds_per_step": (submitted - started) / steps,
        "synchronizing_loss_seconds": synchronized - submitted,
        "loss": loss,
    }


def compare_checkpoints(checkpoints, rtol, atol):
    failures = []
    largest_loss_error = largest_weight_error = 0.0
    for checkpoint in checkpoints:
        for key in ("loss", "weight"):
            baseline = checkpoint["native"][key]
            candidate = checkpoint["trainer"][key]
            finite = math.isfinite(baseline) and math.isfinite(candidate)
            error = abs(candidate - baseline) if finite else math.inf
            if key == "loss":
                largest_loss_error = max(largest_loss_error, error)
            else:
                largest_weight_error = max(largest_weight_error, error)
            if not finite or not math.isclose(
                baseline, candidate, rel_tol=rtol, abs_tol=atol
            ):
                failures.append(
                    {
                        "step": checkpoint["step"],
                        "quantity": key,
                        "native": baseline,
                        "trainer": candidate,
                    }
                )
    return {
        "passed": not failures,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_loss_error": largest_loss_error,
        "max_absolute_weight_error": largest_weight_error,
        "failures": failures,
    }


def benchmark(
    device,
    samples,
    steps_per_sample,
    warmup_steps,
    collocation_count,
    learning_rate,
    max_step_ratio,
    rtol,
    atol,
):
    params, data = make_problem(collocation_count)
    started = time.perf_counter()
    baseline = build_native(params, data, device, learning_rate)
    native_compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    trainer = qb.optim.Trainer(
        poisson_loss, params, qb.optim.Adam(learning_rate), *data, device=device
    )
    trainer_compile_seconds = time.perf_counter() - started
    native_initial, native_read = snapshot(baseline, False)
    trainer_initial, trainer_read = snapshot(trainer, True)
    checkpoints = [{"step": 0, "native": native_initial, "trainer": trainer_initial}]
    for _ in range(warmup_steps):
        baseline.step()
        trainer.step()
    native_warm, _ = snapshot(baseline, False)
    trainer_warm, _ = snapshot(trainer, True)
    checkpoints.append(
        {"step": warmup_steps, "native": native_warm, "trainer": trainer_warm}
    )
    measurements = {"native": [], "trainer": []}
    for sample in range(samples):
        order = (("native", baseline), ("trainer", trainer))
        if sample % 2:
            order = tuple(reversed(order))
        batch_values = {}
        for label, executor in order:
            timing = timed_batch(executor, steps_per_sample)
            measurements[label].append(timing)
            parameters = (
                executor.params if label == "trainer" else executor.parameters()
            )
            batch_values[label] = {
                "loss": timing["loss"],
                "weight": parameters["weight"].to_flat_list()[0],
            }
        checkpoints.append(
            {"step": warmup_steps + (sample + 1) * steps_per_sample, **batch_values}
        )
    medians = {
        name: {
            key: statistics.median(item[key] for item in values)
            for key in (
                "synchronized_seconds_per_step",
                "submission_seconds_per_step",
                "synchronizing_loss_seconds",
            )
        }
        for name, values in measurements.items()
    }
    ratio = (
        medians["trainer"]["synchronized_seconds_per_step"]
        / medians["native"]["synchronized_seconds_per_step"]
    )
    parity = compare_checkpoints(checkpoints, rtol, atol)
    performance_passed = max_step_ratio is None or ratio <= max_step_ratio
    convergence = {
        label: checkpoints[-1][label]["loss"] < checkpoints[0][label]["loss"]
        for label in ("native", "trainer")
    }
    result = {
        "benchmark": "v02_retained_adam_poisson_pinn",
        "device": device,
        "host": platform.platform(),
        "dtype": "float32",
        "collocation_count": collocation_count,
        "parameter_count": 1,
        "initial_weight": 2.5,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "samples": samples,
        "steps_per_sample": steps_per_sample,
        "total_steps_per_executor": warmup_steps + samples * steps_per_sample,
        "compile_seconds": {
            "native": native_compile_seconds,
            "trainer": trainer_compile_seconds,
        },
        "initial_readback": {"native": native_read, "trainer": trainer_read},
        "timing_medians": medians,
        "timing_samples": measurements,
        "trainer_to_native_step_ratio": ratio,
        "max_step_ratio": max_step_ratio,
        "performance_gate_passed": performance_passed,
        "trajectory_parity": parity,
        "loss_decreased": convergence,
        "checkpoints": checkpoints,
        "synchronization_policy": "each timed batch ends with loss readback; synchronized step time includes that amortized readback",
        "transfer_policy": "all fixed data, parameters, gradients, and Adam moments retained; explicit checkpoint readbacks outside step submission timing",
        "compile_policy": "native built first; construction timing includes trace and device initialization and is descriptive only",
        "passed": parity["passed"] and performance_passed and all(convergence.values()),
    }

    # A numerical failure may produce NaN, which is not valid JSON.
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        return value

    print(json.dumps(finite_json(result), sort_keys=True, allow_nan=False))
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--steps-per-sample", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--collocation-count", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-step-ratio", type=float)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if (
        args.device != "mlx"
        and args.device != "cuda"
        and not (args.device.startswith("cuda:") and args.device[5:].isdigit())
    ):
        parser.error("device must be mlx, cuda, or cuda:N")
    if min(args.samples, args.steps_per_sample, args.collocation_count) <= 0:
        parser.error(
            "samples, steps-per-sample, and collocation-count must be positive"
        )
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    if args.max_step_ratio is not None and (
        not math.isfinite(args.max_step_ratio) or args.max_step_ratio <= 0
    ):
        parser.error("max-step-ratio must be positive and finite")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("rtol and atol must be nonnegative and finite")
    passed = benchmark(**vars(args))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
