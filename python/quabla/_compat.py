"""Once-per-name migration warnings from the accepted v0.2 name mapping."""

import dis
import warnings

REPLACEMENTS = {
    "Adam": "qb.optim.Adam, which keeps the .step(params, grads) method",
    "CpuExecutionPlan": "quabla.legacy.CpuExecutionPlan",
    "GradFunction": "quabla.legacy.GradFunction",
    "GradScalarFunction": "quabla.legacy.GradScalarFunction",
    "GradScalarTransform": "quabla.legacy.GradScalarTransform",
    "JacobianFunction": "quabla.legacy.JacobianFunction",
    "JacobiansFunction": "quabla.legacy.JacobiansFunction",
    "JitFunction": "quabla.legacy.JitFunction",
    "JitTransform": "quabla.legacy.JitTransform",
    "JvpFunction": "quabla.legacy.JvpFunction",
    "Matrix": "quabla.legacy.Matrix",
    "TensorBatchCudaJitFunction": "result type of tensor_jit_batch_cuda_fn",
    "TensorBatchCudaValueAndGradFunction": "result type of tensor_value_and_grad_batch_cuda_fn",
    "TensorBatchJitFunction": "result type of tensor_jit_batch_fn",
    "TensorBatchMlxValueAndGradFunction": "result type of tensor_value_and_grad_batch_mlx_fn",
    "TensorBatchValueAndGradFunction": "result type of tensor_value_and_grad_batch_fn",
    "TensorCondFunction": "result type of tensor_cond_fn",
    "TensorCondJvpFunction": "result type of tensor_cond_jvp_fn",
    "TensorCondValueAndGradFunction": "result type of tensor_cond_value_and_grad_fn",
    "TensorCudaValueAndGradFunction": "result type of tensor_value_and_grad_cuda_fn",
    "TensorGradScalarFunction": "result type of tensor_grad_scalar_fn",
    "TensorHessianScalarFunction": "result type of tensor_hessian_scalar_fn",
    "TensorHvpScalarFunction": "result type of tensor_hvp_scalar_fn",
    "TensorJacobianFunction": "result type of tensor_jacobian_fn",
    "TensorJitFunction": "result type of tensor_jit_fn",
    "TensorJvpFunction": "result type of tensor_jvp_fn",
    "TensorMlxValueAndGradFunction": "result type of tensor_value_and_grad_mlx_fn",
    "TensorTraceGraph": "composition (§3.4); Program for inspection",
    "TensorTraceResult": "Program",
    "TensorValueAndGradFunction": "result type of tensor_value_and_grad_fn",
    "TensorVjpFunction": "result type of tensor_vjp_fn",
    "TensorVmapCudaFunction": "result type of tensor_vmap_cuda_fn",
    "TensorVmapCudaHvpScalarFunction": "result type of tensor_vmap_hvp_scalar_cuda_fn",
    "TensorVmapCudaJvpFunction": "result type of tensor_vmap_jvp_cuda_fn",
    "TensorVmapCudaVjpFunction": "result type of tensor_vmap_vjp_cuda_fn",
    "TensorVmapFunction": "result type of tensor_vmap_fn",
    "TensorVmapHvpScalarFunction": "result type of tensor_vmap_hvp_scalar_fn",
    "TensorVmapJvpFunction": "result type of tensor_vmap_jvp_fn",
    "TensorVmapMlxFunction": "result type of tensor_vmap_mlx_fn",
    "TensorVmapMlxJvpFunction": "result type of tensor_vmap_jvp_mlx_fn",
    "TensorVmapMlxVjpFunction": "result type of tensor_vmap_vjp_mlx_fn",
    "TensorVmapVjpFunction": "result type of tensor_vmap_vjp_fn",
    "TraceGraph": "quabla.legacy.TraceGraph",
    "TraceMatrix": "quabla.legacy.TraceMatrix",
    "TraceResult": "quabla.legacy.TraceResult",
    "ValueAndGradFunction": "quabla.legacy.ValueAndGradFunction",
    "VjpFunction": "quabla.legacy.VjpFunction",
    "cuda_adam_loss_optimizer": 'qb.optim.Trainer(..., device="cuda")',
    "cuda_adam_optimizer": 'qb.optim.Trainer(..., device="cuda")',
    "cuda_adam_step": 'qb.optim.Trainer(..., device="cuda")',
    "cuda_adam_vjp_optimizer": 'qb.optim.Trainer(..., device="cuda")',
    "grad_fn": "quabla.legacy.grad_fn",
    "grad_scalar": "quabla.legacy.grad_scalar",
    "grad_scalar_fn": "quabla.legacy.grad_scalar_fn",
    "jacobian_fn": "quabla.legacy.jacobian_fn",
    "jacobians_fn": "quabla.legacy.jacobians_fn",
    "jvp_fn": "quabla.legacy.jvp_fn",
    "mlx_adam_loss_optimizer": 'qb.optim.Trainer(..., device="mlx")',
    "quabla": "the native module quabla._quabla",
    "sum_gradients": "qb.tree.map(operator.add, a, b)",
    "tensor_cond": "qb.cond(pred, true_fun, false_fun, *operands)",
    "tensor_cond_fn": "Python if on a concrete flag + qb.jit(branch)",
    "tensor_cond_jvp_fn": "Python if + qb.jvp(branch, ...)",
    "tensor_cond_value_and_grad_fn": "Python if + qb.value_and_grad(branch)",
    "tensor_fori_loop": "qb.fori_loop(..., unroll=True)",
    "tensor_fori_loop_region": "qb.fori_loop(lower, upper, body, init, operands=...)",
    "tensor_grad_scalar_fn": "qb.grad(f, argnums=(0, ..., n-1))",
    "tensor_hessian_scalar_fn": "qb.hessian(f, argnums=i) (returns a Tensor)",
    "tensor_hvp_scalar_fn": "qb.jvp(qb.grad(f), (x,), (v,))",
    "tensor_jacobian_fn": "qb.jacobian(f, argnums=i) (returns a Tensor)",
    "tensor_jit_batch_cuda_fn": 'qb.jit(f, device="cuda", max_traces=4)',
    "tensor_jit_batch_fn": "qb.jit(f, max_traces=4)",
    "tensor_jit_cuda_fn": 'qb.jit(f, device="cuda")',
    "tensor_jit_fn": "qb.jit(f)",
    "tensor_jvp_fn": "qb.jvp(f, primals, tangents)",
    "tensor_scan": "qb.scan(f, init, length=n, unroll=True)",
    "tensor_scan_region": "qb.scan(f, init, length=upper-lower, operands=...)",
    "tensor_value_and_grad_batch_cuda_fn": 'qb.jit(qb.value_and_grad(f, argnums), device="cuda", max_traces=4)',
    "tensor_value_and_grad_batch_fn": "qb.jit(qb.value_and_grad(f, argnums), max_traces=4)",
    "tensor_value_and_grad_batch_mlx_fn": 'qb.jit(qb.value_and_grad(f, argnums), device="mlx", max_traces=4)',
    "tensor_value_and_grad_cuda_fn": 'qb.jit(qb.value_and_grad(f, argnums), device="cuda")',
    "tensor_value_and_grad_data_parallel_cuda_fn": "qb.distributed.value_and_grad(f, devices=..., shard_argnums=...)",
    "tensor_value_and_grad_fn": "qb.value_and_grad(f, argnums)",
    "tensor_value_and_grad_mlx_fn": 'qb.jit(qb.value_and_grad(f, argnums), device="mlx")',
    "tensor_vjp_fn": "qb.vjp(f, *primals)",
    "tensor_vmap_cuda_fn": 'qb.jit(qb.vmap(f, in_axes, out_axes), device="cuda")',
    "tensor_vmap_fn": "qb.vmap(f, in_axes, out_axes)",
    "tensor_vmap_hvp_scalar_cuda_fn": 'qb.jit(<vmap HVP composition>, device="cuda")',
    "tensor_vmap_hvp_scalar_fn": "qb.jvp(qb.grad(lambda x: qb.sum(qb.vmap(f)(x))), (x,), (v,))",
    "tensor_vmap_jvp_cuda_fn": 'qb.jit(<jvp of vmap>, device="cuda")',
    "tensor_vmap_jvp_fn": "qb.jvp(qb.vmap(f, ...), primals, tangents)",
    "tensor_vmap_jvp_mlx_fn": 'qb.jit(<jvp of vmap>, device="mlx")',
    "tensor_vmap_mlx_fn": 'qb.jit(qb.vmap(f, in_axes, out_axes), device="mlx")',
    "tensor_vmap_vjp_cuda_fn": 'qb.jit(<vjp of vmap>, device="cuda")',
    "tensor_vmap_vjp_fn": "qb.vjp(qb.vmap(f, ...), *primals)",
    "tensor_vmap_vjp_mlx_fn": 'qb.jit(<vjp of vmap>, device="mlx")',
    "trace_tensor": "qb.jit(f).lower(...) or Compiler().trace(f, specs)",
    "value_and_grad_fn": "quabla.legacy.value_and_grad_fn",
    "vjp_fn": "quabla.legacy.vjp_fn",
}
_WARNED = set()


def is_star_import(frame):
    """Whether `frame` is resolving a name for `from quabla import *`.

    A package star import reaches the module `__getattr__` twice per name:
    first from importlib's `_handle_fromlist`, which probes `__all__` with
    `recursive=True`, then from the importing frame while it sits on
    IMPORT_STAR (3.10, 3.11) or on CALL_INTRINSIC_1 with
    INTRINSIC_IMPORT_STAR (3.12+).
    """
    if frame.f_code.co_name == "_handle_fromlist" and frame.f_globals.get(
        "__name__"
    ) in ("importlib._bootstrap", "_frozen_importlib"):
        return bool(frame.f_locals.get("recursive"))
    for instruction in dis.get_instructions(frame.f_code):
        if instruction.offset == frame.f_lasti:
            return (
                instruction.opname == "IMPORT_STAR"
                or instruction.argrepr == "INTRINSIC_IMPORT_STAR"
            )
    return False


def warn(name):
    if name not in _WARNED:
        warnings.warn(
            f"quabla.{name} is deprecated; use {REPLACEMENTS.get(name, 'quabla.legacy.' + name)}. "
            "The old name remains available through 0.x.",
            DeprecationWarning,
            stacklevel=3,
        )
        _WARNED.add(name)
