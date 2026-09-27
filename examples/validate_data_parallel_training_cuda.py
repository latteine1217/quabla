"""Validate two-GPU data-parallel MLP training against single-GPU and CPU runs.

Build on Linux with `maturin develop --release --features cuda-nccl` and run on
a host with CUDA ordinals 0 and 1 and a loadable NCCL library. `--modes
single_gpu,cpu` is a dry run for hosts with one GPU and no NCCL.

Every mode trains the same deterministic tanh MLP regression with the same
host-side SGD update, so the only differences are the value-and-gradient
backend and its floating-point evaluation order:

- `two_gpu_mean`: `tensor_value_and_grad_data_parallel_cuda_fn`, `mean`.
- `two_gpu_sum`: the same callable with `sum`; the loss is divided by the
  replica count and the learning rate is scaled by `1 / replica_count`
  (a power of two, so the host update is exact).
- `single_gpu`: `tensor_value_and_grad_cuda_fn` on the full batch.
- `cpu`: the full-batch CPU `tensor_value_and_grad_fn` (f64 evaluation).

The script fails when any compared pair exceeds the tolerances below or when
training does not reduce the loss. Timing is reported for the first step and
for the steady state after `--warmup` steps. `--grid` and `--hidden` scale the
batch (`grid * grid` rows) and the MLP width; the defaults are the original
256-row, width-32 check.
"""

import argparse
import array
import ctypes
import json
import math
import platform
import statistics
import subprocess
import sys
import time

import nabla


DEFAULT_GRID = 16  # grid * grid == batch input points on [-1, 1]^2.
DEFAULT_HIDDEN = 32
PARAMETER_NAMES = ["w1", "b1", "w2", "b2", "w3", "b3"]
MAPPED_INPUTS = ["x", "target"]
DEVICE_ORDINALS = [0, 1]
PARAMETER_KEY = 2026
ALL_MODES = ["two_gpu_mean", "two_gpu_sum", "single_gpu", "cpu"]
COMPARISONS = [
    ("two_gpu_mean", "single_gpu"),
    ("two_gpu_mean", "cpu"),
    ("single_gpu", "cpu"),
    ("two_gpu_sum", "two_gpu_mean"),
]

# Tolerances are relative. CUDA evaluates in f32 (unit roundoff u = 2^-24,
# about 6.0e-8); the CPU reference evaluates in f64. The longest reduction is
# the batch mean (256 rows by default), whose worst-case relative rounding
# bound is gamma_256 = 256 u, about 1.5e-5. Two GPUs change only the
# summation order (two 128-row partial means plus one NCCL reduction), so
# each step's gradient may differ at that level, and SGD carries such
# differences into later steps. 1e-4 (about 6.5 gamma_256) allows that
# accumulation over the run while still rejecting reduction bugs such as a
# Sum/Mean mix-up or a dropped shard, which produce O(1) relative
# differences. Larger batches keep the same ratio to gamma_batch, so the
# tolerance scales linearly with the batch above 256 rows; hidden widths up
# to the batch size add shorter reductions than the batch mean. The same
# tolerance applies to the first step's gradients, which every mode
# evaluates at identical inputs: a reduction bug shifts them by O(1)
# normwise, which keeps the check sensitive when a short, wide run barely
# moves the loss and its trajectories alone would not expose the bug.
BASE_REL_TOLERANCE = 1e-4
BASE_TOLERANCE_BATCH = 256
# The run must actually train, otherwise matching trajectories prove little.
# `--max-loss-ratio` relaxes this for short scale runs that rely on the
# first-step gradient comparison instead.
MAX_FINAL_TO_INITIAL_LOSS = 0.5
TIMING_KEYS = ["replica_enqueue", "collective", "output_readback"]


def loss_fn(x, target, w1, b1, w2, b2, w3, b3):
    hidden = (x @ w1 + b1).tanh()
    hidden = (hidden @ w2 + b2).tanh()
    return (hidden @ w3 + b3 - target).powi(2).mean()


def parameter_shapes(hidden):
    return {
        "w1": [2, hidden],
        "b1": [1, hidden],
        "w2": [hidden, hidden],
        "b2": [1, hidden],
        "w3": [hidden, 1],
        "b3": [1, 1],
    }


def input_specs(batch, hidden):
    return [("x", [batch, 2]), ("target", [batch, 1])] + list(
        parameter_shapes(hidden).items()
    )


def relative_tolerance(batch):
    return BASE_REL_TOLERANCE * max(1.0, batch / BASE_TOLERANCE_BATCH)


def to_f32(values):
    """Round to f32 so every backend starts from identical representable data."""
    return array.array("f", values).tolist()


def build_problem(grid, hidden):
    coordinates = [-1.0 + 2.0 * index / (grid - 1) for index in range(grid)]
    points = [(u, v) for u in coordinates for v in coordinates]
    x = to_f32([value for point in points for value in point])
    target = to_f32(
        [math.sin(math.pi * u) * math.cos(math.pi * v) for u, v in points]
    )
    weight_keys = nabla.Tensor.split_key(PARAMETER_KEY, 3)
    parameters = {}
    for name, shape in parameter_shapes(hidden).items():
        if name.startswith("w"):
            key = weight_keys[int(name[1]) - 1]
            values = nabla.Tensor.glorot_normal(shape, key).to_flat_list()
        else:
            values = [0.0] * shape[0] * shape[1]
        parameters[name] = to_f32(values)
    return x, target, parameters


def build_callable(mode, specs):
    if mode == "cpu":
        return nabla.tensor_value_and_grad_fn(loss_fn, specs)
    if mode == "single_gpu":
        return nabla.tensor_value_and_grad_cuda_fn(
            loss_fn, specs, PARAMETER_NAMES, DEVICE_ORDINALS[0]
        )
    reduction = mode.removeprefix("two_gpu_")
    return nabla.tensor_value_and_grad_data_parallel_cuda_fn(
        loss_fn,
        specs,
        PARAMETER_NAMES,
        MAPPED_INPUTS,
        DEVICE_ORDINALS,
        reduction,
    )


def train(mode, steps, lr, grid, hidden):
    batch = grid * grid
    shapes = parameter_shapes(hidden)
    x, target, parameters = build_problem(grid, hidden)
    # Sum returns replica_count times the mean loss and gradient.
    scale = len(DEVICE_ORDINALS) if mode == "two_gpu_sum" else 1
    step_lr = lr / scale

    construct_start = time.perf_counter()
    value_and_grad = build_callable(mode, input_specs(batch, hidden))
    construct_seconds = time.perf_counter() - construct_start

    fixed = {
        "x": nabla.Tensor([batch, 2], x),
        "target": nabla.Tensor([batch, 1], target),
    }
    losses = []
    timings = []
    first_gradient = None
    for _ in range(steps):
        inputs = dict(fixed)
        for name, shape in shapes.items():
            inputs[name] = nabla.Tensor(shape, parameters[name])
        call_start = time.perf_counter()
        value, gradients = value_and_grad(inputs)
        call_wall = time.perf_counter() - call_start
        timing = {"call_wall": call_wall}
        if mode.startswith("two_gpu_"):
            timing.update(value_and_grad.last_timing)
        timings.append(timing)
        (loss,) = value.to_flat_list()
        losses.append(loss / scale)
        gradients = {name: gradients[name].to_flat_list() for name in PARAMETER_NAMES}
        if first_gradient is None:
            # Dividing by the power-of-two scale is exact.
            first_gradient = [
                grad / scale for name in PARAMETER_NAMES for grad in gradients[name]
            ]
        for name in PARAMETER_NAMES:
            parameters[name] = [
                parameter - step_lr * grad
                for parameter, grad in zip(parameters[name], gradients[name])
            ]
    return {
        "construct_seconds": construct_seconds,
        "losses": losses,
        "first_gradient": first_gradient,
        "parameters": parameters,
        "timings": timings,
    }


def compare(candidate, reference):
    loss_abs = [abs(a - b) for a, b in zip(candidate["losses"], reference["losses"])]
    loss_rel = [
        error / abs(b) for error, b in zip(loss_abs, reference["losses"])
    ]
    actual = [value for name in PARAMETER_NAMES for value in candidate["parameters"][name]]
    expected = [value for name in PARAMETER_NAMES for value in reference["parameters"][name]]
    parameter_abs = max(abs(a - b) for a, b in zip(actual, expected))
    # Normwise over the whole parameter vector: the symmetric target keeps
    # the output bias near zero, so elementwise or per-tensor relative errors
    # would divide by values at the f32 noise floor.
    parameter_scale = max(abs(b) for b in expected)
    gradient_abs = max(
        abs(a - b)
        for a, b in zip(candidate["first_gradient"], reference["first_gradient"])
    )
    gradient_scale = max(abs(b) for b in reference["first_gradient"])
    return {
        "loss_max_abs": max(loss_abs),
        "loss_max_rel": max(loss_rel),
        "parameter_max_abs": parameter_abs,
        "parameter_inf_norm": parameter_scale,
        "parameter_max_normwise_rel": parameter_abs / parameter_scale,
        "first_gradient_max_abs": gradient_abs,
        "first_gradient_max_normwise_rel": gradient_abs / gradient_scale,
    }


def summarize_timing(timings, warmup):
    summary = {}
    steady = timings[warmup:]
    for key in timings[0]:
        values = [timing[key] for timing in steady]
        summary[key] = {
            "first_step": timings[0][key],
            "steady_median": statistics.median(values) if values else None,
            "steady_min": min(values) if values else None,
            "steady_max": max(values) if values else None,
            "steady_steps": len(values),
        }
    return summary


def command_output(command):
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


def nccl_version():
    try:
        library = ctypes.CDLL("libnccl.so")
    except OSError as error:
        return f"unavailable: {error}"
    version = ctypes.c_int()
    if library.ncclGetVersion(ctypes.byref(version)) != 0:
        return "unavailable: ncclGetVersion failed"
    code = version.value
    # NCCL >= 2.9 encodes major * 10000 + minor * 100 + patch.
    return f"{code // 10000}.{code % 10000 // 100}.{code % 100}"


def metadata(args, modes, tolerance):
    batch = args.grid * args.grid
    return {
        "modes": modes,
        "steps": args.steps,
        "warmup_steps": args.warmup,
        "learning_rate": args.lr,
        "optimizer": "host-side SGD, identical in every mode",
        "batch": batch,
        "hidden": args.hidden,
        "mapped_inputs": MAPPED_INPUTS,
        "parameter_shapes": parameter_shapes(args.hidden),
        "device_ordinals": DEVICE_ORDINALS,
        "parameter_key": PARAMETER_KEY,
        "data": f"{args.grid}x{args.grid} grid on [-1, 1]^2, target sin(pi u) cos(pi v)",
        "tolerances": {
            "loss_max_rel": tolerance,
            "parameter_max_normwise_rel": tolerance,
            "first_gradient_max_normwise_rel": tolerance,
            "max_final_to_initial_loss": args.max_loss_ratio,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": command_output(["nvcc", "--version"]).splitlines()[-1:],
        "nccl": nccl_version() if any(m.startswith("two_gpu_") for m in modes) else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", default=",".join(ALL_MODES))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_GRID, help="batch is grid * grid rows"
    )
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument(
        "--max-loss-ratio",
        type=float,
        default=MAX_FINAL_TO_INITIAL_LOSS,
        help="fail unless final loss <= ratio * initial loss",
    )
    parser.add_argument("--output", help="also write the JSON report to this path")
    args = parser.parse_args()
    modes = [mode for mode in args.modes.split(",") if mode]
    unknown = sorted(set(modes) - set(ALL_MODES))
    if unknown or not modes:
        parser.error(f"unknown or empty --modes {unknown}; choose from {ALL_MODES}")
    if args.steps <= args.warmup:
        parser.error("--steps must exceed --warmup")
    if args.grid < 2 or args.hidden < 1:
        parser.error("--grid must be at least 2 and --hidden at least 1")
    return args, modes


def main():
    args, modes = parse_args()
    runs = {
        mode: train(mode, args.steps, args.lr, args.grid, args.hidden)
        for mode in modes
    }
    tolerance = relative_tolerance(args.grid * args.grid)

    failures = []
    comparisons = {}
    for candidate, reference in COMPARISONS:
        if candidate not in runs or reference not in runs:
            continue
        result = compare(runs[candidate], runs[reference])
        comparisons[f"{candidate} vs {reference}"] = result
        if result["loss_max_rel"] > tolerance:
            failures.append(f"{candidate} vs {reference}: loss {result['loss_max_rel']:e}")
        if result["parameter_max_normwise_rel"] > tolerance:
            failures.append(
                f"{candidate} vs {reference}: parameters "
                f"{result['parameter_max_normwise_rel']:e}"
            )
        if result["first_gradient_max_normwise_rel"] > tolerance:
            failures.append(
                f"{candidate} vs {reference}: first-step gradients "
                f"{result['first_gradient_max_normwise_rel']:e}"
            )

    report_runs = {}
    for mode, run in runs.items():
        initial, final = run["losses"][0], run["losses"][-1]
        if not (math.isfinite(final) and final <= initial * args.max_loss_ratio):
            failures.append(f"{mode}: loss did not train ({initial} -> {final})")
        report_runs[mode] = {
            "construct_seconds": run["construct_seconds"],
            "initial_loss": initial,
            "final_loss": final,
            "timing_seconds": summarize_timing(run["timings"], args.warmup),
        }

    report = {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "metadata": metadata(args, modes, tolerance),
        "runs": report_runs,
        "comparisons": comparisons,
        "loss_trajectories": {mode: run["losses"] for mode, run in runs.items()},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
