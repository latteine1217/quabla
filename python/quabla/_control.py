"""Fixed-bounds control flow with explicit region operands (design 3.9).

Loops carry one array. Region bodies see a scalar array index; unrolled
and eager bodies see a Python integer. Scan outputs stack on axis zero.
"""

import operator

from . import _quabla
from ._array import asarray
from ._errors import TracerError
from ._quabla import TensorTraceGraph, TraceTensor

__all__ = ["cond", "fori_loop", "scan"]

_FOREIGN_OUTPUT_ERRORS = {
    f"{name} body returned a TraceTensor from a different graph"
    for name in (
        "tensor_fori_loop",
        "tensor_fori_loop_region",
        "tensor_scan",
        "tensor_scan_region",
    )
} | {"trace_tensor function returned a tensor from a different graph"}


def _region_call(function, *args):
    try:
        return function(*args)
    except (TypeError, ValueError) as error:
        message = str(error)
        if (
            message in _FOREIGN_OUTPUT_ERRORS
            or message == "cannot combine TraceTensor values from different graphs"
            or (
                isinstance(error, TracerError)
                and message.startswith(
                    "a TraceTensor of another trace was combined with this trace"
                )
                and "tensor_* helper or region trace" in message
            )
        ):
            raise TracerError(
                "control-flow regions cannot close over an outer tracer; "
                "pass it explicitly using operands= (cond takes positional operands)"
            ) from error
        raise


def _bindings(values):
    """Bind eager constants to the same graph as the explicit tracers."""
    arrays = [asarray(value) for value in values]
    anchor = next((value for value in arrays if isinstance(value, TraceTensor)), None)
    if anchor is None or all(isinstance(value, TraceTensor) for value in arrays):
        return arrays
    # Inlining constants avoids adding arithmetic dependencies on the anchor,
    # which could change shapes or propagate a NaN from an unused value.
    graph = TensorTraceGraph()
    graph.input("anchor", anchor.shape, anchor.dtype)
    constants = [
        graph._constant(value) for value in arrays if not isinstance(value, TraceTensor)
    ]
    bound = iter(graph._inline(["anchor"], [anchor], constants))
    return [
        value if isinstance(value, TraceTensor) else next(bound) for value in arrays
    ]


def cond(pred, true_fun, false_fun, *operands):
    """Run the selected branch; traced scalar predicates form lazy regions.

    Branches receive the positional operands. Pass every outer tracer used
    by a traced branch explicitly as an operand.
    """
    if not isinstance(pred, TraceTensor):
        return (true_fun if bool(pred) else false_fun)(*operands)
    predicate, *captures = _bindings((pred, *operands))
    # Constant branch results bind to the region through one of its inputs; a
    # cond without operands passes the predicate as that hidden input.
    hidden = not captures
    if hidden:
        captures = [predicate]
    return _region_call(
        _quabla.tensor_cond,
        predicate,
        _branch(true_fun, hidden),
        _branch(false_fun, hidden),
        captures,
    )


def _branch(function, hidden):
    def traced(*values):
        result = function(*values[1:]) if hidden else function(*values)
        if isinstance(result, TraceTensor):
            return result
        return _bindings((values[0], result))[1]

    return traced


def _bound(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a static integer")
    result = operator.index(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _carry(value):
    if isinstance(value, (tuple, list, dict)):
        raise TypeError("control-flow carry must be a single array, not a pytree")
    return asarray(value)


def _same_carry(initial, result):
    result = _carry(result)
    if result.shape != initial.shape or result.dtype != initial.dtype:
        raise ValueError("control-flow body must preserve the carry shape and dtype")
    return result


def fori_loop(lower, upper, body_fun, init_val, *, operands=(), unroll=False):
    """Apply `body_fun(i, carry, *operands)` over static [lower, upper).

    By default a traced body forms one runtime region; `unroll=True` expands
    each iteration in the enclosing trace. The carry is a single array.
    """
    lower, upper = _bound(lower, "lower"), _bound(upper, "upper")
    if upper < lower:
        raise ValueError("fori_loop requires upper >= lower")
    initial, *captures = _bindings((_carry(init_val), *operands))

    def body(index, carry, *values):
        return _same_carry(initial, body_fun(index, carry, *values))

    if isinstance(initial, TraceTensor):
        if unroll:
            return _region_call(
                _quabla.tensor_fori_loop,
                lower,
                upper,
                lambda index, carry: body(index, carry, *captures),
                initial,
            )
        return _region_call(
            _quabla.tensor_fori_loop_region, lower, upper, body, initial, captures
        )
    carry = initial
    for index in range(lower, upper):
        carry = body(index, carry, *captures)
    return carry


def scan(f, init, *, length, operands=(), unroll=False):
    """Apply `f(carry, i, *operands) -> (carry, y)` and stack the outputs.

    `length` is a positive static integer. Traced scans form a runtime region
    unless `unroll=True`. Carries and per-step outputs are single arrays.
    """
    length = _bound(length, "length")
    if length == 0:
        raise ValueError("scan length must be positive")
    initial, *captures = _bindings((_carry(init), *operands))

    def body(index, carry, *values):
        result = f(carry, index, *values)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("scan body must return a (carry, output) tuple")
        return _same_carry(initial, result[0]), _carry(result[1])

    if isinstance(initial, TraceTensor):
        if unroll:
            return _region_call(
                _quabla.tensor_scan,
                length,
                lambda index, carry: body(index, carry, *captures),
                initial,
            )
        return _region_call(
            _quabla.tensor_scan_region, 0, length, body, initial, captures
        )
    carry, outputs = initial, []
    for index in range(length):
        carry, output = body(index, carry, *captures)
        if outputs and (
            output.shape != outputs[0].shape or output.dtype != outputs[0].dtype
        ):
            raise ValueError("scan body must preserve the output shape and dtype")
        outputs.append(output)
    return carry, _quabla.stack(outputs, 0)
