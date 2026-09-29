"""Exception classes of the v0.2 API (docs/api_v0_2_design.md, section 3.12, D16).

Each class also subclasses the builtin exception that the same failure raised
before v0.2, so existing `except ValueError` and `except TypeError` handlers
keep catching it. The extension raises `TracerError` by importing it from
this module, so the class is the same object in Python and Rust.
"""

__all__ = [
    "QuablaError",
    "RetraceLimitError",
    "TracerError",
    "UnsupportedOperationError",
]


class QuablaError(Exception):
    """Base class of the errors specific to quabla."""


class UnsupportedOperationError(QuablaError, ValueError, NotImplementedError):
    """An operation, transform, or composition that this build or device does
    not support. `op` names it and `device` is the target device, when one
    applies; both are `None` otherwise."""

    def __init__(self, message, *, op=None, device=None):
        super().__init__(message)
        self.op = op
        self.device = device


class TracerError(QuablaError, TypeError):
    """A traced value was used where a concrete value is required: Python
    control flow, or conversion to a Python number or a NumPy array. (An
    eager `Tensor` that meets a traced value is captured as a constant.)"""


class RetraceLimitError(QuablaError, ValueError):
    """A transformed function needed more traces than its `max_traces`
    bound; the message lists the cached signatures and the new one."""
