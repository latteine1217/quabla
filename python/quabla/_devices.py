"""Explicit build targets and abstract array inputs for ahead-of-time tracing."""

from dataclasses import dataclass

from ._errors import UnsupportedOperationError
from ._quabla import Compiler, bool_, float32, float64

__all__ = ["ShapeDtype", "devices"]


def parse_device(device):
    if device is None or device == "cpu":
        return "cpu", 0
    if device in ("cuda", "mlx"):
        return device, 0
    if isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit():
        return "cuda", int(device[5:])
    raise ValueError(
        f"device must be None, 'cpu', 'cuda', 'cuda:N', or 'mlx', got {device!r}"
    )


def require_device(device, op):
    target, ordinal = parse_device(device)
    if not Compiler().capability(target):
        raise UnsupportedOperationError(
            f"{target} target is unavailable in this build", op=op, device=device
        )
    return target, ordinal


def devices():
    """Built targets; this does not probe hardware or enumerate physical GPUs."""
    return [target for target, built in Compiler().capabilities().items() if built]


@dataclass(frozen=True)
class ShapeDtype:
    """An array signature without storage, accepted by `jit(...).lower`."""

    shape: tuple
    dtype: object = float64

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(type(extent) is not int or extent < 0 for extent in shape):
            raise ValueError("shape extents must be non-negative integers")
        if self.dtype not in (float32, float64, bool_):
            raise TypeError("dtype must be quabla.float32, float64, or bool_")
        object.__setattr__(self, "shape", shape)
