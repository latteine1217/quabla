"""Eager and traced tensors accept the same operands and report the same errors.

`Tensor` and `TraceTensor` methods share one operand parser. Every binary
method and operator is called with each kind of operand, once on an eager
tensor and once on a tracer inside `jit`, and the two outcomes are compared:
the result dtype, shape and value bits, or the exception type and message.
"""

import struct

import quabla as qb

from _support import run


def matrix():
    return qb.Tensor([2, 2], [2.0, 0.5, 0.5, 1.5])


def flag():
    return qb.Tensor([2, 2], [1.0, 0.0, 1.0, 1.0], dtype=qb.bool_)


# Operands that are not tracers; a tracer operand is covered by the traced
# side itself. A view is a `TensorView` of a row-major slice.
OPERANDS = {
    "tensor": matrix,
    "view": lambda: qb.Tensor([3, 2], [2.0, 0.5, 0.5, 1.5, 9.0, 9.0]).slice(0, 0, 2),
    "float": lambda: 0.5,
    "int": lambda: 3,
    "bool": lambda: True,
    "str": lambda: "a",
    "none": lambda: None,
}

# Named methods that take one operand, with the receiver they need.
METHODS = {
    "add": matrix, "sub": matrix, "mul": matrix, "div": matrix,
    "maximum": matrix, "minimum": matrix, "atan2": matrix, "fmod": matrix,
    "gt": matrix, "greater": matrix, "greater_equal": matrix, "less": matrix,
    "less_equal": matrix, "equal": matrix, "not_equal": matrix,
    "matmul": matrix, "solve": matrix, "solve_triangular": matrix,
    "logical_and": flag, "logical_or": flag,
}

# Operators as `(receiver, operand) -> result`, with the receiver they need;
# the `r` forms put the receiver on the right.
OPERATORS = {
    "+": (matrix, lambda x, y: x + y), "r+": (matrix, lambda x, y: y + x),
    "-": (matrix, lambda x, y: x - y), "r-": (matrix, lambda x, y: y - x),
    "*": (matrix, lambda x, y: x * y), "r*": (matrix, lambda x, y: y * x),
    "/": (matrix, lambda x, y: x / y), "r/": (matrix, lambda x, y: y / x),
    "**": (matrix, lambda x, y: x ** y), "r**": (matrix, lambda x, y: y ** x),
    "@": (matrix, lambda x, y: x @ y), "r@": (matrix, lambda x, y: y @ x),
    "&": (flag, lambda x, y: x & y), "r&": (flag, lambda x, y: y & x),
    "|": (flag, lambda x, y: x | y), "r|": (flag, lambda x, y: y | x),
    ">": (matrix, lambda x, y: x > y), "<": (matrix, lambda x, y: x < y),
}


def summary(outcome):
    if isinstance(outcome, BaseException):
        # When both operands return NotImplemented, Python names the
        # receiver's class in its own message; the class is the only part
        # that may differ.
        message = str(outcome).replace("builtins.TraceTensor", "builtins.Tensor")
        return type(outcome).__name__, message
    values = outcome.tolist()
    flat = []
    stack = [values]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(reversed(value))
        else:
            flat.append(value)
    if outcome.dtype == qb.bool_:
        bits = [bool(value) for value in flat]
    else:
        bits = [struct.pack("<d", value).hex() for value in flat]
    return str(outcome.dtype), list(outcome.shape), bits


def eager_and_traced(receiver, call, operand):
    outcomes = []
    for evaluate in (lambda x: call(x, operand()), qb.jit(lambda x: call(x, operand()))):
        try:
            outcomes.append(summary(evaluate(receiver())))
        except (ValueError, TypeError) as error:
            outcomes.append(summary(error))
    return outcomes


def method_call(name):
    def call(x, y):
        return getattr(x, name)(y)

    return call


def cases():
    for name, receiver in METHODS.items():
        call = method_call(name)
        for kind, operand in OPERANDS.items():
            yield f"{name}({kind})", receiver, call, operand
    for name, (receiver, call) in OPERATORS.items():
        for kind, operand in OPERANDS.items():
            yield f"{name} {kind}", receiver, call, operand


def test_eager_and_traced_operands_agree():
    mismatches = []
    for case, receiver, call, operand in cases():
        eager, traced = eager_and_traced(receiver, call, operand)
        if eager != traced:
            mismatches.append(f"{case}: eager {eager} != traced {traced}")
    assert not mismatches, "\n".join(mismatches)


def test_operand_errors_name_both_array_types():
    # One text per operand role, the same for eager and traced code.
    for evaluate in (lambda f: f(matrix()), lambda f: qb.jit(f)(matrix())):
        for call, message in (
            (lambda x: x + "a", "expected a Tensor, TraceTensor, or numeric scalar operand"),
            (lambda x: x.maximum(None), "expected a Tensor, TraceTensor, or numeric scalar operand"),
            (lambda x: x.matmul("a"), "matmul expects a Tensor or TraceTensor operand, got str"),
            (lambda x: x.solve(1.0), "solve expects a Tensor or TraceTensor operand, got float"),
        ):
            try:
                evaluate(call)
            except TypeError as error:
                assert str(error) == message, (str(error), message)
            else:
                raise AssertionError(f"accepted an invalid operand: {message}")


if __name__ == "__main__":
    run(globals())
