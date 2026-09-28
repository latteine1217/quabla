import json

import quabla


INPUT_SPECS = [
    ("x", [8, 1]),
    ("target", [8, 1]),
    ("weight", [1, 1]),
]
MAPPED_INPUTS = ["x", "target"]
DEVICE_ORDINALS = [0, 1]
TOLERANCE = 1e-5


def loss_fn(x, target, weight):
    return ((x * weight - target).powi(2)).mean()


def scalar(tensor):
    values = tensor.to_flat_list()
    assert len(values) == 1
    return values[0]


def validate_reduction(cpu_plan, inputs, reduction):
    cpu_value, cpu_gradients = cpu_plan.value_and_grad_data_parallel(
        inputs, MAPPED_INPUTS, len(DEVICE_ORDINALS), reduction
    )
    cuda_fn = quabla.tensor_value_and_grad_data_parallel_cuda_fn(
        loss_fn,
        INPUT_SPECS,
        ["weight"],
        MAPPED_INPUTS,
        DEVICE_ORDINALS,
        reduction,
    )
    cuda_value, cuda_gradients = cuda_fn(inputs)

    assert cuda_fn.replica_count == len(DEVICE_ORDINALS)
    assert cuda_fn.device_ordinals == DEVICE_ORDINALS
    assert set(cuda_gradients) == {"weight"}

    cpu_loss = scalar(cpu_value)
    cuda_loss = scalar(cuda_value)
    cpu_weight_gradient = scalar(cpu_gradients["weight"])
    cuda_weight_gradient = scalar(cuda_gradients["weight"])
    loss_error = abs(cuda_loss - cpu_loss)
    gradient_error = abs(cuda_weight_gradient - cpu_weight_gradient)
    assert loss_error <= TOLERANCE, (reduction, cpu_loss, cuda_loss, loss_error)
    assert gradient_error <= TOLERANCE, (
        reduction,
        cpu_weight_gradient,
        cuda_weight_gradient,
        gradient_error,
    )

    timing = cuda_fn.last_timing
    assert timing is not None
    assert set(timing) == {"replica_enqueue", "collective", "output_readback"}
    assert all(duration >= 0.0 for duration in timing.values())
    return {
        "reduction": reduction,
        "cpu_loss": cpu_loss,
        "cuda_loss": cuda_loss,
        "loss_abs_error": loss_error,
        "cpu_weight_gradient": cpu_weight_gradient,
        "cuda_weight_gradient": cuda_weight_gradient,
        "gradient_abs_error": gradient_error,
        "timing_seconds": timing,
    }


def main():
    inputs = {
        "x": quabla.Tensor([8, 1], [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]),
        "target": quabla.Tensor(
            [8, 1], [-7.0, -5.0, -3.0, -1.0, 3.0, 5.0, 7.0, 9.0]
        ),
        "weight": quabla.Tensor([1, 1], [0.5]),
    }
    traced = quabla.trace_tensor(loss_fn, INPUT_SPECS)
    cpu_plan = traced.output.compile_cpu()
    results = [
        validate_reduction(cpu_plan, inputs, reduction)
        for reduction in ("sum", "mean")
    ]
    print(json.dumps({"status": "passed", "results": results}, indent=2))


if __name__ == "__main__":
    main()
