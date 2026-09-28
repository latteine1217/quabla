"""Measure CUDA frozen-plan throughput with deterministic inputs.

Run on Linux with a CUDA driver and NVRTC available:
    maturin develop --release --features cuda
    python examples/benchmark_tensor_cuda.py

This is an end-to-end benchmark. Each evaluation includes the current Python
Tensor host-to-device transfer and output device-to-host transfer, so it does
not claim kernel-only throughput.
"""

import argparse
import json
import platform
import time

import quabla


def tensor(shape: list[int], offset: int) -> quabla.Tensor:
    count = 1
    for extent in shape:
        count *= extent
    data = [((index + offset) % 23 - 11) / 23.0 for index in range(count)]
    return quabla.Tensor(shape, data)


def benchmark(
    iterations: int,
    batch: int,
    rows: int,
    inner: int,
    cols: int,
    rank_two: bool,
    activation: bool,
    elementwise: bool,
    device_resident: bool,
) -> None:
    lhs_shape = [rows, inner] if rank_two else [batch, rows, inner]
    rhs_shape = [inner, cols] if rank_two else [1, inner, cols]
    if elementwise:
        def model(x):
            return ((x * 0.75 + 0.125).tanh().sin().exp())
        traced = quabla.trace_tensor(model, [("x", lhs_shape)])
        inputs = {"x": tensor(lhs_shape, 0)}
    else:
        model = (lambda lhs, rhs: (lhs @ rhs).tanh()) if activation else (lambda lhs, rhs: lhs @ rhs)
        traced = quabla.trace_tensor(model, [("lhs", lhs_shape), ("rhs", rhs_shape)])
        inputs = {
            "lhs": tensor(lhs_shape, 0),
            "rhs": tensor(rhs_shape, 1),
        }
    compile_start = time.perf_counter()
    plan = traced.output.compile_cuda()
    compile_seconds = time.perf_counter() - compile_start
    warmup_iterations = 0 if device_resident else 5

    if device_resident:
        retained = list(inputs)
        seconds = plan.benchmark_device(inputs, iterations, retained)
        checksum = None
    else:
        for _ in range(warmup_iterations):
            plan.evaluate(inputs)
        start = time.perf_counter()
        checksum = 0.0
        for _ in range(iterations):
            checksum += plan.evaluate(inputs).to_flat_list()[0]
        seconds = time.perf_counter() - start
    print(
        "metadata="
        + json.dumps(
            {
                "backend": plan.backend,
                "compile_seconds": compile_seconds,
                "device_ordinal": plan.device_ordinal,
                "dtype": "f32",
                "host": platform.platform(),
                "iterations": iterations,
                "synchronization_policy": "benchmark_device synchronizes each run"
                if device_resident
                else "evaluate synchronizes before host result materialization",
                "transfer_policy": "retained-device-inputs"
                if device_resident
                else "host-to-device inputs and device-to-host output per evaluation",
                "warmup_iterations": warmup_iterations,
            },
            sort_keys=True,
        )
    )
    if elementwise:
        elements = iterations
        for extent in lhs_shape:
            elements *= extent
        print(
            f"elementwise_shape={lhs_shape}, iterations={iterations}, "
            f"compile={compile_seconds * 1e3:.2f} ms, backend={plan.backend}"
        )
        print(
            f"end_to_end: {iterations / seconds:.2f} eval/s "
            f"({seconds * 1e3 / iterations:.3f} ms/eval), "
            f"{elements / seconds / 1e9:.4f} Gelem/s, "
            f"device_resident={device_resident}, checksum={checksum}"
        )
    else:
        matrix_count = 1 if rank_two else batch
        operations = 2 * matrix_count * rows * inner * cols * iterations
        print(
            f"lhs={lhs_shape}, rhs={rhs_shape}, "
            f"activation={activation}, iterations={iterations}, compile={compile_seconds * 1e3:.2f} ms, "
            f"backend={plan.backend}"
        )
        print(
            f"end_to_end: {iterations / seconds:.2f} eval/s "
            f"({seconds * 1e3 / iterations:.3f} ms/eval), "
            f"{operations / seconds / 1e9:.4f} GFLOP/s, "
            f"device_resident={device_resident}, checksum={checksum}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--inner", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--rank-two", action="store_true")
    parser.add_argument("--activation", action="store_true")
    parser.add_argument("--elementwise", action="store_true")
    parser.add_argument("--device-resident", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.batch, args.rows, args.inner, args.cols) <= 0:
        parser.error("all dimensions and iterations must be positive")
    benchmark(
        args.iterations,
        args.batch,
        args.rows,
        args.inner,
        args.cols,
        args.rank_two,
        args.activation,
        args.elementwise,
        args.device_resident,
    )


if __name__ == "__main__":
    main()
