"""Custom differentiation rules: `custom_vjp`, `custom_jvp`, and
`checkpoint` (v0.3 plan item 9).

Called on eager arrays outside any transform, a wrapped function simply
runs `fun`. Called while a transform traces (on tracers, or inside `jit`,
`grad`, `vmap`, ...), the call is staged: `fun` is traced into a graph of
its own, the rule graphs are traced from the user's rule functions (or
derived symbolically), and `TensorTraceGraph._custom_call` splices the
primal graph into the enclosing trace with every floating-point output
wrapped in a `Custom` node that carries the rule (see
`crates/quabla-core/src/tensor_ir/custom.rs`). The symbolic transforms
apply the rule when they meet the node, and plan compilation removes it,
so the result runs on every backend.

- `custom_vjp`: the forward graph is `fwd` (outputs and residuals), the
  backward graph is `bwd`; forward mode raises, as in JAX.
- `custom_jvp`: the tangent graph is the user's JVP rule; reverse mode
  transposes it, by differentiating the rule's tangent output, which is
  linear in the tangents, with respect to the tangents.
- `checkpoint`: both rules are derived from `fun`; the residuals are the
  operands, so reverse mode recomputes the intermediates of `fun` instead
  of keeping them from the forward pass.
"""

import functools
import threading

from ._array import zeros
from ._quabla import Tensor, TensorTraceGraph, TraceTensor, bool_
from ._transforms import (
    _Aval,
    _Static,
    _is_traced,
    _normalize_argnums,
    _trace,
    _traced_signature,
)
from .tree import _LEAF, _describe, _flatten, _leaf_count, _leaf_paths, _unflatten

__all__ = ["checkpoint", "custom_jvp", "custom_vjp", "remat"]

_RESIDUAL_PREFIX = "__quabla_residual/"
_COTANGENT_PREFIX = "__quabla_cotangent/"
_TANGENT_PREFIX = "__quabla_tangent/"


def _name(fun):
    return getattr(fun, "__qualname__", None) or repr(fun)


def _zero_like(value):
    """The zero tangent or cotangent a rule receives for a leaf the custom
    node does not wrap: zeros for an eager array, `0` for a number."""
    if isinstance(value, Tensor):
        return zeros(value.shape, value.dtype)
    if type(value) in (bool, int, float):
        return type(value)(0)
    return None


class _Call:
    """One staged call: the differentiable arguments flattened into leaves
    (array leaves become operands of the rule), and the `nondiff_argnums`
    arguments, which every traced function closes over unchanged."""

    def __init__(self, nondiff_argnums, args):
        positions = (
            _normalize_argnums(nondiff_argnums, len(args), "nondiff_argnums")
            if nondiff_argnums
            else ()
        )
        self.nondiff = [(position, args[position]) for position in sorted(positions)]
        diff_args = tuple(arg for index, arg in enumerate(args) if index not in positions)
        self.in_node, self.leaves, _, self.bindings = _traced_signature(
            diff_args, (), frozenset()
        )
        self.names = []
        for index, node in enumerate(self.in_node[1]):
            self.names.extend(_leaf_paths(node, f"arg{index}"))
        self.nondiff_values = [value for _, value in self.nondiff]

    def full_args(self, diff_args):
        args = list(diff_args)
        for position, value in self.nondiff:
            args.insert(position, value)
        return args

    def trace(self, function):
        """Traces `function` called with every argument, the differentiable
        ones as inputs named `self.names`."""
        return _trace(
            lambda *diff_args: function(*self.full_args(diff_args)),
            self.in_node,
            self.leaves,
            self.names,
        )


def _reject_captures(staged, what, role):
    if staged.captures:
        raise TypeError(
            f"the {role} of {what} closes over a traced value of an enclosing "
            "transform; a custom rule only differentiates with respect to explicit "
            "arguments, so pass the value as an argument (or through nondiff_argnums "
            "when it is not differentiated)"
        )


def _wrapped_positions(staged, what):
    """Output leaves the custom node wraps: the traced ones, which must be
    floating-point. Constant outputs pass through unchanged."""
    positions = []
    for position, output in enumerate(staged.outputs):
        if not _is_traced(output):
            continue
        if output.dtype == bool_:
            raise TypeError(
                f"{what} returns a bool array; functions with a custom differentiation "
                "rule must return floating-point arrays (or constants)"
            )
        positions.append(position)
    return positions


def _as_graph_leaf(graph, value, reference, what):
    """`value` returned by a rule function as a tracer of its `graph` with the
    shape and dtype of `reference`: a tracer is checked, an eager array
    becomes a constant, and `None` or a number means zero."""
    if value is None or type(value) in (bool, int, float):
        if value:
            raise TypeError(f"{what} must be an array or None, got {value!r}")
        value = zeros(reference.shape, reference.dtype)
    if isinstance(value, Tensor):
        value = graph._constant(value)
    if not _is_traced(value):
        raise TypeError(f"{what} must be an array, got {type(value).__name__}")
    if list(value.shape) != list(reference.shape) or value.dtype != reference.dtype:
        raise TypeError(
            f"{what} must have shape {list(reference.shape)} and dtype "
            f"{reference.dtype!r}, got shape {list(value.shape)} and dtype {value.dtype!r}"
        )
    return value


def _match_leaves(node, reference, leaves, what):
    """The leaves of a rule result with structure `node`, one per leaf of
    `reference` in order; a `None` subtree stands for zeros (`None`)."""
    if node is None:
        return [None] * _leaf_count(reference)
    if reference is _LEAF:
        if node is not _LEAF:
            raise TypeError(f"{what} has structure {_describe(node)}, expected an array")
        return [next(leaves)]
    if (
        node is _LEAF
        or node[:-1] != reference[:-1]
        or len(node[-1]) != len(reference[-1])
    ):
        raise TypeError(
            f"{what} has structure {_describe(node)}, expected {_describe(reference)}"
        )
    matched = []
    for child, expected in zip(node[-1], reference[-1]):
        matched.extend(_match_leaves(child, expected, leaves, what))
    return matched


def _operand_positions(call):
    """Positions of the array leaves (the rule operands) among the leaves."""
    return [
        position
        for position, leaf in enumerate(call.leaves)
        if type(leaf) in (Tensor, TraceTensor)
    ]


def _input_tracers(staged):
    return [staged.graph.input(name) for name in staged.input_names]


# The wrapped functions whose rule is being built on this thread. A rule
# function commonly calls its own function (`fwd` returning `f(x)`, a JVP
# rule using `f(x)`); such a call evaluates the plain function instead of
# staging the rule again, which would recurse without end (JAX builds its
# rules lazily instead).
_BUILDING = threading.local()


def _building():
    owners = getattr(_BUILDING, "owners", None)
    if owners is None:
        owners = _BUILDING.owners = set()
    return owners


def _call(owner, args, what, build_rule):
    """A call of the wrapped function `owner`: plain `owner.fun(*args)`
    outside any transform (and inside its own rule functions), staged with
    the rule `build_rule(call, staged, wrapped)` returns otherwise."""
    if id(owner) in _building() or not (
        TensorTraceGraph._tracing() or _has_tracer(args)
    ):
        return owner.fun(*args)
    _building().add(id(owner))
    try:
        return _staged_call(owner.fun, owner.nondiff_argnums, args, what, build_rule)
    finally:
        _building().discard(id(owner))


def _staged_call(fun, nondiff_argnums, args, what, build_rule):
    """Stages one call of `fun` with the rule `build_rule(call, staged,
    wrapped)` returns, into the trace in progress, and returns its result."""
    call = _Call(nondiff_argnums, args)
    staged = call.trace(fun)
    wrapped = _wrapped_positions(staged, what)
    outputs = list(staged.outputs)
    if wrapped:
        rule = build_rule(call, staged, wrapped)
        spliced = staged.graph._custom_call(
            rule,
            staged.input_names,
            call.bindings + list(staged.captures),
            [outputs[position] for position in wrapped],
        )
        for position, output in zip(wrapped, spliced):
            outputs[position] = output
    return _unflatten(staged.out_node, iter(outputs))


def _has_tracer(args):
    leaves = []
    _flatten(tuple(args), leaves)
    return any(type(leaf) is TraceTensor for leaf in leaves)


class custom_vjp:
    """A function with a custom reverse-mode rule, as `jax.custom_vjp`.

    `f = custom_vjp(fun)` then `f.defvjp(fwd, bwd)`: `fwd(*args)` returns
    `(out, residuals)` with `out` like `fun(*args)`, and
    `bwd(*nondiff_args, residuals, cotangent)` returns one cotangent per
    differentiable argument (a tuple; `None` for zero), each with the
    pytree structure of its argument. `f(*args)` is `fun(*args)`; reverse
    mode (`grad`, `vjp`, `value_and_grad`, reverse `jacobian`) uses `fwd`
    and `bwd`, and forward mode (`jvp`, forward `jacobian`, `hessian`)
    raises `ValueError`. The arguments at `nondiff_argnums` are passed to
    `bwd` first and are not differentiated; they must not be traced arrays.
    `fun`, `fwd`, and `bwd` must not close over traced values of an
    enclosing transform.
    """

    def __init__(self, fun, nondiff_argnums=()):
        if not callable(fun):
            raise TypeError(f"custom_vjp expects a callable, got {type(fun).__name__}")
        functools.update_wrapper(self, fun)
        self.fun = fun
        self.nondiff_argnums = tuple(nondiff_argnums)
        self.fwd = None
        self.bwd = None

    def defvjp(self, fwd, bwd):
        if not callable(fwd) or not callable(bwd):
            raise TypeError("defvjp expects callables fwd and bwd")
        self.fwd = fwd
        self.bwd = bwd

    def __call__(self, *args):
        what = f"custom_vjp function {_name(self.fun)}"
        if self.fwd is None:
            raise AttributeError(f"no VJP is defined for {what}; call defvjp first")
        return _call(self, args, what, self._rule)

    def _rule(self, call, staged, wrapped):
        what = f"custom_vjp({_name(self.fun)})"
        _reject_captures(staged, what, "function")
        forward = call.trace(self.fwd)
        _reject_captures(forward, what, "fwd")
        node = forward.out_node
        if not (type(node) is tuple and node[0] is tuple and len(node[1]) == 2):
            raise TypeError(f"the fwd of {what} must return a pair (output, residuals)")
        out_node, residual_node = node[1]
        if out_node != staged.out_node:
            raise TypeError(
                f"the fwd of {what} returns an output with structure "
                f"{_describe(out_node)}, but the function returns {_describe(staged.out_node)}"
            )
        count = _leaf_count(out_node)
        forward_out, residuals = forward.outputs[:count], forward.outputs[count:]
        for position in wrapped:
            reference, value = staged.outputs[position], forward_out[position]
            if not _is_traced(value) or (
                list(value.shape) != list(reference.shape) or value.dtype != reference.dtype
            ):
                raise TypeError(
                    f"the fwd of {what} returns an output leaf that does not match the "
                    f"function's: expected shape {list(reference.shape)} and dtype "
                    f"{reference.dtype!r}"
                )

        # bwd is traced on the residuals and the output cotangents; constant
        # residuals and the cotangents of constant outputs are passed as values.
        leaves, names, residual_outputs, residual_names = [], [], [], []
        for index, residual in enumerate(residuals):
            if _is_traced(residual):
                name = f"{_RESIDUAL_PREFIX}{index}"
                leaves.append(_Aval(residual.shape, residual.dtype))
                names.append(name)
                residual_outputs.append(residual)
                residual_names.append(name)
            else:
                leaves.append(_Static(residual))
                names.append(None)
        cotangent_names = []
        wrapped_set = set(wrapped)
        for position, output in enumerate(staged.outputs):
            if position in wrapped_set:
                name = f"{_COTANGENT_PREFIX}{len(cotangent_names)}"
                leaves.append(_Aval(output.shape, output.dtype))
                names.append(name)
                cotangent_names.append(name)
            else:
                leaves.append(_Static(_zero_like(output)))
                names.append(None)
        nondiff = call.nondiff_values
        backward = _trace(
            lambda res, cotangent: self.bwd(*nondiff, res, cotangent),
            (tuple, (residual_node, staged.out_node)),
            leaves,
            names,
        )
        _reject_captures(backward, what, "bwd")
        argument_nodes = call.in_node[1]
        node = backward.out_node
        if not (
            type(node) is tuple and node[0] in (tuple, list) and len(node[1]) == len(argument_nodes)
        ):
            raise TypeError(
                f"the bwd of {what} must return a tuple with one cotangent per "
                f"differentiable argument ({len(argument_nodes)}), got structure "
                f"{_describe(node)}"
            )
        results = iter(backward.outputs)
        cotangents = []
        for index, (child, reference) in enumerate(zip(node[1], argument_nodes)):
            cotangents.extend(
                _match_leaves(
                    child,
                    reference,
                    results,
                    f"the cotangent bwd of {what} returns for argument {index}",
                )
            )
        backward_outputs = []
        for position in _operand_positions(call):
            leaf = call.leaves[position]
            if leaf.dtype == bool_:
                backward_outputs.append(None)
                continue
            backward_outputs.append(
                _as_graph_leaf(
                    backward.graph,
                    cotangents[position],
                    leaf,
                    f"the cotangent bwd of {what} returns for {call.names[position]}",
                )
            )
        return TensorTraceGraph._custom_rule(
            what,
            forward.graph,
            forward.input_names,
            len(wrapped),
            [forward_out[position] for position in wrapped] + residual_outputs,
            backward.graph,
            residual_names,
            cotangent_names,
            backward_outputs,
            None,
            [],
            [],
            False,
        )


class custom_jvp:
    """A function with a custom forward-mode rule, as `jax.custom_jvp`.

    `f = custom_jvp(fun)` then `f.defjvp(rule)`:
    `rule(*nondiff_args, primals, tangents)` returns
    `(primal_out, tangent_out)`, where `primals` and `tangents` are tuples
    of the differentiable arguments and their tangents and `tangent_out` is
    linear in `tangents`. `f(*args)` is `fun(*args)`; forward mode uses the
    rule, and reverse mode uses its transpose, derived by differentiating
    `tangent_out` with respect to `tangents`, so higher-order derivatives
    differentiate the rule. The arguments at `nondiff_argnums` are passed to
    the rule first; `fun` and the rule must not close over traced values of
    an enclosing transform.
    """

    def __init__(self, fun, nondiff_argnums=()):
        if not callable(fun):
            raise TypeError(f"custom_jvp expects a callable, got {type(fun).__name__}")
        functools.update_wrapper(self, fun)
        self.fun = fun
        self.nondiff_argnums = tuple(nondiff_argnums)
        self.jvp = None

    def defjvp(self, jvp):
        if not callable(jvp):
            raise TypeError("defjvp expects a callable")
        self.jvp = jvp
        return jvp

    def __call__(self, *args):
        what = f"custom_jvp function {_name(self.fun)}"
        if self.jvp is None:
            raise AttributeError(f"no JVP is defined for {what}; call defjvp first")
        return _call(self, args, what, self._rule)

    def _rule(self, call, staged, wrapped):
        what = f"custom_jvp({_name(self.fun)})"
        _reject_captures(staged, what, "function")
        operands = _operand_positions(call)
        # The rule is traced on the primals, as inputs named like the primal
        # graph's, and one tangent input per floating-point primal.
        leaves = list(call.leaves)
        names = list(call.names)
        tangent_names = []
        for position, leaf in enumerate(call.leaves):
            if position in operands and leaf.dtype != bool_:
                name = f"{_TANGENT_PREFIX}{call.names[position]}"
                leaves.append(_Aval(leaf.shape, leaf.dtype))
                names.append(name)
                tangent_names.append(name)
            else:
                if position in operands:
                    tangent_names.append(None)
                leaves.append(_Static(None if position in operands else _zero_like(leaf)))
                names.append(None)
        nondiff = call.nondiff_values
        tangent = _trace(
            lambda primals, tangents: self.jvp(*nondiff, primals, tangents),
            (tuple, (call.in_node, call.in_node)),
            leaves,
            names,
        )
        _reject_captures(tangent, what, "JVP rule")
        node = tangent.out_node
        if not (type(node) is tuple and node[0] is tuple and len(node[1]) == 2):
            raise TypeError(
                f"the JVP rule of {what} must return a pair (primal_out, tangent_out)"
            )
        primal_node, tangent_node = node[1]
        if primal_node != staged.out_node:
            raise TypeError(
                f"the JVP rule of {what} returns a primal output with structure "
                f"{_describe(primal_node)}, but the function returns "
                f"{_describe(staged.out_node)}"
            )
        results = iter(tangent.outputs[_leaf_count(primal_node) :])
        tangent_leaves = _match_leaves(
            tangent_node, staged.out_node, results, f"the tangent output of {what}"
        )
        tangent_outputs = [
            _as_graph_leaf(
                tangent.graph,
                tangent_leaves[position],
                staged.outputs[position],
                f"the tangent output of {what}",
            )
            for position in wrapped
        ]

        # The transpose: reverse mode over the linear map tangents ->
        # tangent_out, at zero tangents, with the primals as residuals.
        cotangent_names = [f"{_COTANGENT_PREFIX}{index}" for index in range(len(wrapped))]
        transpose, _, gradients = tangent.graph._symbolic_vjp(
            tangent_outputs, cotangent_names, []
        )
        operand_names = staged.input_names
        operand_leaves = [call.leaves[position] for position in operands]
        tangent_inputs = [name for name in tangent_names if name is not None]
        tangent_zeros = [
            zeros(leaf.shape, leaf.dtype)
            for leaf, name in zip(operand_leaves, tangent_names)
            if name is not None
        ]
        transpose_outputs = [gradients[name] for name in tangent_inputs]
        inline_names = list(operand_names) + tangent_inputs + cotangent_names

        def backward(*flat):
            primals = list(flat[: len(operand_names)])
            cotangents = list(flat[len(operand_names) :])
            return tuple(
                transpose._inline(
                    inline_names, primals + tangent_zeros + cotangents, transpose_outputs
                )
            )

        backward_leaves = [_Aval(leaf.shape, leaf.dtype) for leaf in operand_leaves] + [
            _Aval(staged.outputs[position].shape, staged.outputs[position].dtype)
            for position in wrapped
        ]
        backward_staged = _trace(
            backward,
            (tuple, tuple([_LEAF] * len(backward_leaves))),
            backward_leaves,
            list(operand_names) + cotangent_names,
        )
        transposed = iter(backward_staged.outputs)
        backward_outputs = [
            next(transposed) if name is not None else None for name in tangent_names
        ]
        return TensorTraceGraph._custom_rule(
            what,
            staged.graph,
            operand_names,
            len(wrapped),
            [staged.outputs[position] for position in wrapped] + _input_tracers(staged),
            backward_staged.graph,
            list(operand_names),
            cotangent_names,
            backward_outputs,
            tangent.graph,
            tangent_names,
            tangent_outputs,
            False,
        )


class _Checkpoint:
    def __init__(self, fun, static_argnums):
        functools.update_wrapper(self, fun)
        self.fun = fun
        self.nondiff_argnums = static_argnums

    def __call__(self, *args):
        return _call(self, args, f"checkpoint function {_name(self.fun)}", self._rule)

    def _rule(self, call, staged, wrapped):
        what = f"checkpoint({_name(self.fun)})"
        outputs = [staged.outputs[position] for position in wrapped]
        operand_names = staged.input_names
        inputs = _input_tracers(staged)
        cotangent_names = [f"{_COTANGENT_PREFIX}{index}" for index in range(len(wrapped))]
        # Reverse mode: the VJP graph of `fun` replays its forward pass from
        # the inputs, which are the residuals, so the intermediates are
        # recomputed during the backward pass.
        backward, _, gradients = staged.graph._symbolic_vjp(outputs, cotangent_names, [])
        backward_outputs = [gradients.get(name) for name in operand_names]
        tangent_names = [
            None if tracer.dtype == bool_ else f"{_TANGENT_PREFIX}{name}"
            for name, tracer in zip(operand_names, inputs)
        ]
        tangent_inputs = {
            name: tangent for name, tangent in zip(operand_names, tangent_names) if tangent
        }
        if tangent_inputs:
            tangent, _, tangent_outputs = staged.graph._symbolic_jvp(outputs, tangent_inputs)
        else:
            tangent, tangent_outputs = None, []
        return TensorTraceGraph._custom_rule(
            what,
            staged.graph,
            operand_names,
            len(wrapped),
            outputs + inputs,
            backward,
            list(operand_names),
            cotangent_names,
            backward_outputs,
            tangent,
            tangent_names,
            tangent_outputs,
            True,
        )


def checkpoint(fun, *, static_argnums=()):
    """`fun` with rematerialized reverse mode, as `jax.checkpoint`.

    Values are unchanged. Under reverse mode the intermediates of `fun` are
    not kept from the forward pass: the backward pass recomputes them from
    the arguments (and the values `fun` closes over), trading compute for
    memory. Derivatives of every order equal those of `fun`. The arguments
    at `static_argnums` are not differentiated and must not be traced
    arrays.
    """
    if not callable(fun):
        raise TypeError(f"checkpoint expects a callable, got {type(fun).__name__}")
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    return _Checkpoint(fun, tuple(static_argnums))


remat = checkpoint
