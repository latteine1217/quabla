"""Experimental single-node CUDA/NCCL data parallel differentiation.

Batch arguments contain full, equally sized axis-zero batches. Parameters
are replicated and only their gradients are reduced; optimizer updates stay
on the host. The reduction combines per-shard losses, so ``sum`` sums shard
losses and ``mean`` averages them, independently of the loss's own reduction.
"""

# Imported under private aliases so the public module namespace is `__all__`.
from . import _quabla
from ._devices import parse_device as _parse_device
from ._errors import UnsupportedOperationError as _UnsupportedOperationError
from ._quabla import Tensor as _Tensor
from ._quabla import TraceTensor as _TraceTensor
from ._quabla import bool_ as _bool_
from ._transforms import (
    _DEFAULT_MAX_TRACES,
    _TraceCache,
    _argnums_config,
    _argument_ranges,
    _leaf_names,
    _normalize_argnums,
    _signature,
)
from .tree import _unflatten

__all__ = ["value_and_grad"]


class _DistributedValueAndGrad:
    def __init__(self, fun, devices, shard_argnums, argnums, reduction):
        if not callable(fun):
            raise TypeError("distributed.value_and_grad requires a callable")
        if isinstance(devices, str):
            raise TypeError("devices must be a sequence of CUDA devices")
        ordinals = []
        for device in devices:
            target, ordinal = _parse_device(device)
            if target != "cuda":
                raise _UnsupportedOperationError(
                    "distributed.value_and_grad requires CUDA devices and NCCL",
                    op="distributed.value_and_grad",
                    device=device,
                )
            ordinals.append(ordinal)
        if len(ordinals) < 2:
            raise ValueError(
                "distributed.value_and_grad requires at least two CUDA devices"
            )
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("devices must not contain duplicate CUDA ordinals")
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self._fun = fun
        self._devices = ordinals
        self._shard_argnums = _argnums_config(shard_argnums)
        self._argnums = _argnums_config(argnums)
        self._reduction = reduction
        self._cache = _TraceCache(_DEFAULT_MAX_TRACES)

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError(
                "distributed.value_and_grad takes positional arguments only"
            )
        count = len(args)
        argnums = _normalize_argnums(self._argnums, count, "argnums")
        shard_argnums = _normalize_argnums(self._shard_argnums, count, "shard_argnums")
        if not argnums:
            raise ValueError("argnums must select at least one replicated parameter")
        if not shard_argnums:
            raise ValueError("shard_argnums must select at least one batch argument")
        if set(argnums) & set(shard_argnums):
            raise ValueError(
                "mapped-input gradients are unsupported; argnums must be replicated"
            )
        in_node, leaves, arrays, key, traced = _signature(args, (), argnums)
        if traced:
            raise _UnsupportedOperationError(
                "distributed.value_and_grad cannot run inside another transform",
                op="distributed.value_and_grad",
                device="cuda",
            )
        names = _leaf_names(self._fun, in_node)
        ranges = _argument_ranges(in_node)
        mapped = [position for argnum in shard_argnums for position in ranges[argnum]]
        parameters = [position for argnum in argnums for position in ranges[argnum]]
        if not mapped or not parameters:
            raise ValueError("batch and parameter arguments must have array leaves")
        batch_size = None
        for position in mapped:
            leaf = leaves[position]
            if type(leaf) is not _Tensor:
                raise TypeError("every mapped batch leaf must be an array")
            if not leaf.shape:
                raise ValueError("every mapped batch leaf must have rank at least one")
            extent = leaf.shape[0]
            if extent == 0 or extent % len(self._devices):
                raise ValueError(
                    "batch size must be positive and divisible into equal device shards"
                )
            if batch_size is not None and extent != batch_size:
                raise ValueError(
                    "all mapped batch leaves must have the same axis-zero batch size"
                )
            batch_size = extent
        for position in parameters:
            if type(leaves[position]) is not _Tensor:
                raise TypeError("every differentiated parameter leaf must be an array")
        active_parameters = [
            position for position in parameters if leaves[position].dtype != _bool_
        ]
        if not active_parameters:
            raise ValueError(
                "argnums must contain at least one non-bool parameter array"
            )

        def build():
            # Keep only static leaves in the adapter: array values vary without
            # retracing, and retaining their data would pin an old batch.
            template = [None if type(leaf) is _Tensor else leaf for leaf in leaves]
            array_positions = [
                position
                for position, leaf in enumerate(leaves)
                if type(leaf) is _Tensor
            ]

            def flat_loss(*inputs):
                values = template.copy()
                for position, value in zip(array_positions, inputs):
                    values[position] = value
                result = self._fun(*_unflatten(in_node, iter(values)))
                if type(result) is not _TraceTensor:
                    raise TypeError(
                        "distributed.value_and_grad requires a single scalar array loss"
                    )
                if result.shape != []:
                    raise ValueError(
                        f"distributed.value_and_grad requires a scalar loss, got {result.shape}"
                    )
                return result

            specs = [
                (names[p], leaves[p].shape, leaves[p].dtype) for p in array_positions
            ]
            try:
                return _quabla.tensor_value_and_grad_data_parallel_cuda_fn(
                    flat_loss,
                    specs,
                    [names[p] for p in active_parameters],
                    [names[p] for p in mapped],
                    self._devices,
                    self._reduction,
                )
            except ValueError as error:
                message = str(error).lower()
                if (
                    "cuda-nccl" in message
                    or "data-parallel backend is unavailable" in message
                ):
                    raise _UnsupportedOperationError(
                        str(error), op="distributed.value_and_grad", device="cuda"
                    ) from error
                raise

        executable = self._cache.lookup(key, build, self)
        inputs = dict(
            zip(
                (name for name, leaf in zip(names, leaves) if type(leaf) is _Tensor),
                arrays,
            )
        )
        value, gradients = executable(inputs)
        grad_values = [
            None if leaves[p].dtype == _bool_ else gradients[names[p]]
            for p in parameters
        ]
        grad_node = (
            in_node[1][argnums[0]]
            if isinstance(self._argnums, int)
            else (tuple, tuple(in_node[1][argnum] for argnum in argnums))
        )
        return value, _unflatten(grad_node, iter(grad_values))


def value_and_grad(fun, *, devices, shard_argnums, argnums=0, reduction="mean"):
    """Return a scalar loss and replicated parameter gradients using NCCL.

    ``shard_argnums`` selects positional batch pytrees, whose array leaves
    share a full axis-zero batch size divisible by the number of devices.
    ``argnums`` selects replicated parameter pytrees and follows the usual
    ``value_and_grad`` result structure (an integer selects one pytree; a
    tuple selects a tuple of pytrees). Mapped-input gradients, auxiliary
    outputs, and calls inside other transforms are unsupported. Programs
    cache up to eight shape/dtype/static-value signatures per callable.
    """
    return _DistributedValueAndGrad(fun, devices, shard_argnums, argnums, reduction)
