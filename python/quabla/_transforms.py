"""Composable staged function transforms (docs/api_v0_2_design.md,
sections 3.3-3.6 and 3.11, decisions D5-D8 and D16-D17; slices S2-S5).

Every transform is staged, as `jit` is: the first call with a new argument
signature traces the function into a Tensor IR graph, applies the symbolic
transform (reverse mode for `grad`/`value_and_grad`/`vjp`, forward mode for
`jvp`, eligible smaller input/output basis for `jacobian`), compiles one
multi-output program, and caches it. CPU is
the default; an outer `jit(device=...)` selects CUDA or MLX explicitly.
Later calls with the same signature only flatten the arguments and run the
program. The signature is the pytree structure of the positional arguments
plus the shape and dtype of every array leaf and the value of every static
leaf (D7): Python `bool`/`int`/`float` leaves are static weak constants, so
they keep a `float32` program in `float32`, and a new value retraces. In a
differentiated position (`argnums`, the primals of `jvp`/`vjp`) a Python
scalar becomes a `float64` array instead, since a constant has no
derivative. A function may be traced at most `max_traces` times (8 unless
`jit` sets it) before `RetraceLimitError`; the cache never evicts.

Values that the function reads from its closure or from globals are baked
into the trace, as with `jax.jit`: an eager array that meets a traced value
becomes a constant of the graph (slice S3b), so rebinding the name after the
first call does not retrace. Pass values that change as arguments.

Composition works by staging (slice S3, design 3.4 and D8). A transform
whose function is itself a transform (`grad(grad(f))`, `jvp(grad(f), ...)`,
`jit(value_and_grad(f))`, `hessian`) transforms the inner transform's graph
directly. A transformed function called on traced values, inside a function
that another transform is tracing, stages itself for the tracers' shapes and
dtypes in a graph of its own, cached per signature like a compiled program,
and inlines that graph into the enclosing trace (`TensorIr::inline`): its
inputs bind to the tracers, so a derivative can be used inside a loss that
is differentiated again. `jit` of a plain Python function traces through it
instead, which is exact. `vmap` stages its function for one example and
splices that graph batched into its own (`TensorIr::inline_batched`, slice
S4), so it composes the same way; dense `jacobian` and `hessian` splice
bounded derivative basis batches, so they compose as well.

A function staged inside another trace may close over that trace's tracers
(`grad(lambda x: net(params, x))` with traced `params`, slice S4b). Each
trace in progress is registered with the bridge, which lifts such a tracer
into the inner graph as an implicit input (`__quabla_capture/N`) when an op
first meets it. The inner transform treats the input as a constant, as JAX
treats closed-over values, and inlining binds it back to the captured
tracer, so outer transforms differentiate through it. Since a closure can
refer to a different tracer on every call, a staged graph that captured
tracers is reused only after tracing its function again to find the
tracers of this call, which must have the shapes and dtypes it was staged
with.
"""

import inspect
import weakref
import warnings

from . import _quabla
from ._array import arange, asarray, zeros
from ._devices import ShapeDtype, require_device
from ._errors import RetraceLimitError, UnsupportedOperationError
from ._quabla import Tensor, TensorTraceGraph, TraceTensor, bool_, float32, float64
from .tree import _LEAF, _describe, _flatten, _leaf_count, _leaf_paths, _unflatten

__all__ = ["grad", "hessian", "jacobian", "jit", "jvp", "value_and_grad", "vjp", "vmap"]

_STATIC_TYPES = (bool, int, float)


class _Aval:
    """The shape and dtype of an array leaf without data: `vmap` stages its
    function for one example with these leaves."""

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


# Leaves that become graph inputs when a function is staged: arrays, the
# tracers of an enclosing trace when a transformed function is inlined, and
# the per-example leaves of `vmap`.
_ARRAY_LEAVES = (Tensor, TraceTensor, _Aval)
_DEFAULT_MAX_TRACES = 8
# Graph input names starting with `__quabla_` are reserved (design 3.11).
_TANGENT_PREFIX = "__quabla_tangent/"
_COTANGENT_PREFIX = "__quabla_cotangent/"
# Bounds each dense Jacobian JVP batch without changing the public transform.
_JACOBIAN_CHUNK_SIZE = 64
# Coordinate digits stay exact on backends that execute float64 as float32.
_JACOBIAN_INDEX_RADIX = 1 << 24

# -- argument signatures -------------------------------------------------------


class _Static:
    """An argument made static as a whole by `jit(static_argnums=...)`."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _Text(str):
    """A string that reprs without quotes, for rendering signatures."""

    def __repr__(self):
        return str(self)


def _static_key(value):
    # The type keeps 1, 1.0, and True apart; `hex` keeps -0.0 apart from 0.0
    # and makes a NaN equal to itself.
    return (type(value), value.hex() if type(value) is float else value)


def _as_array_leaf(leaf):
    if isinstance(leaf, (tuple, list, dict)):
        raise TypeError(
            f"{type(leaf).__name__} is not a pytree container: only dict, list, tuple, and "
            "None are (NamedTuple and container subclasses are not supported in v0.2)"
        )
    try:
        return asarray(leaf)
    except TypeError as error:
        raise TypeError(
            f"unsupported argument leaf of type {type(leaf).__name__}: transformed functions "
            "take pytrees of arrays and Python scalars"
        ) from error


def _normalize_argnums(argnums, count, what):
    """`argnums` as a tuple of non-negative positions below `count`."""
    single = isinstance(argnums, int)
    positions = (argnums,) if single else tuple(argnums)
    normalized = []
    for position in positions:
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError(
                f"{what} must be an int or a tuple of ints, got {argnums!r}"
            )
        if not -count <= position < count:
            raise ValueError(
                f"{what}={argnums!r} is out of range for a call with {count} positional arguments"
            )
        normalized.append(position % count)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{what}={argnums!r} contains a duplicate position")
    return tuple(normalized)


def _signature(args, static_argnums, converted_argnums):
    """Flattens positional arguments for a call.

    Returns `(in_node, leaves, arrays, key, traced)`: the structure node of
    the argument tuple, every leaf (array leaves converted to `Tensor`), the
    array leaves in order, the cache key, and whether a leaf is a tracer of
    an enclosing trace.
    """
    leaves = []
    nodes = []
    for index, arg in enumerate(args):
        if index in static_argnums:
            try:
                hash(arg)
            except TypeError:
                raise TypeError(
                    f"static argument {index} must be hashable, got {type(arg).__name__}"
                ) from None
            nodes.append(_LEAF)
            leaves.append(_Static(arg))
            continue
        start = len(leaves)
        nodes.append(_flatten(arg, leaves))
        if index in converted_argnums:
            for position in range(start, len(leaves)):
                if type(leaves[position]) in _STATIC_TYPES:
                    leaves[position] = asarray(leaves[position])
    in_node = (tuple, tuple(nodes))
    key = []
    arrays = []
    traced = False
    for position, leaf in enumerate(leaves):
        kind = type(leaf)
        if kind is Tensor or kind is _Aval:
            pass
        elif kind in _STATIC_TYPES:
            key.append(_static_key(leaf))
            continue
        elif kind is _Static:
            key.append((_Static, leaf.value))
            continue
        elif kind is TraceTensor:
            traced = True
            continue
        else:
            leaf = leaves[position] = _as_array_leaf(leaf)
        arrays.append(leaf)
        key.append((tuple(leaf.shape), leaf.dtype))
    return in_node, leaves, arrays, (in_node, tuple(key)), traced


def _traced_signature(args, static_argnums, converted_argnums):
    """Flattens a call whose arguments hold tracers of an enclosing trace.

    Returns `(in_node, leaves, key, bindings)`. Each tracer leaf stands for
    an input of the staged callee with the tracer's shape and dtype, and a
    Python scalar in a differentiated position becomes a `float64` array
    leaf as in `_signature`; `bindings` holds, per array leaf in order, the
    value its callee input is bound to when the callee is inlined: the
    tracer, or the Python scalar or eager array as a constant of the
    enclosing trace.
    """
    leaves = []
    nodes = []
    key = []
    bindings = []
    for index, arg in enumerate(args):
        if index in static_argnums:
            nodes.append(_LEAF)
            leaves.append(_Static(arg))
            key.append((_Static, arg))
            continue
        start = len(leaves)
        nodes.append(_flatten(arg, leaves))
        for position in range(start, len(leaves)):
            leaf = leaves[position]
            kind = type(leaf)
            if kind is TraceTensor:
                if leaf._batched:
                    raise UnsupportedOperationError(
                        "a transformed function was called on a batched tracer of a "
                        "tensor_vmap_* helper; batch transformed functions with quabla.vmap "
                        "instead",
                        op="nested transform",
                    )
                bindings.append(leaf)
            elif kind in _STATIC_TYPES:
                if index not in converted_argnums:
                    key.append(_static_key(leaf))
                    continue
                bindings.append(leaf)
                leaf = leaves[position] = asarray(leaf)
            else:
                leaf = leaves[position] = _as_array_leaf(leaf)
                bindings.append(leaf)
            key.append((tuple(leaf.shape), leaf.dtype))
    in_node = (tuple, tuple(nodes))
    return in_node, leaves, (in_node, tuple(key)), bindings


def _describe_signature(key):
    if type(key) is _CaptureKey:
        key, avals = key
        captured = ", ".join(_describe_aval(shape, dtype) for shape, dtype in avals)
        return f"{_describe_signature(key)} capturing [{captured}]"
    in_node, parts = key
    rendered = []
    for part in parts:
        if part[0] is _Static:
            rendered.append(_Text(f"static {part[1]!r}"))
        elif isinstance(part[0], tuple):
            rendered.append(_Text(_describe_aval(part[0], part[1])))
        else:
            value = float.fromhex(part[1]) if part[0] is float else part[1]
            rendered.append(_Text(repr(value)))
    return repr(_unflatten(in_node, iter(rendered)))


def _describe_aval(shape, dtype):
    return f"{dtype.name}[{','.join(str(extent) for extent in shape)}]"


def _argument_ranges(in_node):
    """The `range` of leaf positions of each positional argument."""
    ranges = []
    start = 0
    for child in in_node[1]:
        count = _leaf_count(child)
        ranges.append(range(start, start + count))
        start += count
    return ranges


def _parameter_names(fun, count):
    """Names for the first `count` positional parameters of `fun`: its own
    parameter names where they are positional, `argN` otherwise."""
    names = [f"arg{index}" for index in range(count)]
    try:
        parameters = list(inspect.signature(fun).parameters.values())
    except (TypeError, ValueError):
        return names
    for index, parameter in enumerate(parameters[:count]):
        if parameter.kind not in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            break
        names[index] = parameter.name
    return names


def _leaf_names(fun, in_node):
    """Graph input names for the leaves of an argument tuple: key paths
    rooted at the parameter names (`params/layers/0/w`), unique."""
    arguments = in_node[1]
    names = []
    for name, node in zip(_parameter_names(fun, len(arguments)), arguments):
        names.extend(_leaf_paths(node, name))
    seen = {}
    for index, name in enumerate(names):
        if name in seen:
            seen[name] += 1
            names[index] = f"{name}#{seen[name]}"
        else:
            seen[name] = 0
    return names


# -- staging -------------------------------------------------------------------


class _Staged:
    """A traced and possibly transformed program: `graph` takes the array
    leaves of the arguments as inputs named `input_names`, and `outputs` are
    the flat result leaves, each a tracer of `graph` or a constant.

    `captures` holds the tracers of enclosing traces that the function
    closed over, one per capture input; those inputs follow the argument
    inputs in `input_names` (the tangent and cotangent inputs of `jvp` and
    `vjp` graphs come after them). `probe` finds the captures of a later
    call (see `_Transform._inline_staged`), or is `None` without captures.
    A staged graph kept in an inline cache has `captures` set to `None`.
    """

    __slots__ = ("graph", "input_names", "outputs", "out_node", "captures", "probe")

    def __init__(self, graph, input_names, outputs, out_node, captures=(), probe=None):
        self.graph = graph
        self.input_names = input_names
        self.outputs = outputs
        self.out_node = out_node
        self.captures = captures
        self.probe = probe


def _capture_names(staged):
    """The capture input names of a staged function (not of a `jvp`/`vjp`
    graph, whose tangent or cotangent inputs follow them)."""
    return staged.input_names[len(staged.input_names) - len(staged.captures) :]


class _Probe:
    """Traces the root function of a staged graph that captured tracers
    again, with the same argument avals, to find the tracers that a new call
    captures. The function is held weakly where possible, since the caches
    holding the probe are keyed weakly on it."""

    __slots__ = ("fun", "in_node", "leaves", "names")

    def __init__(self, fun, in_node, leaves, names):
        try:
            self.fun = weakref.ref(fun)
        except TypeError:
            self.fun = lambda: fun
        self.in_node = in_node
        self.leaves = [
            _Aval(leaf.shape, leaf.dtype) if type(leaf) in _ARRAY_LEAVES else leaf
            for leaf in leaves
        ]
        self.names = names

    def captures(self):
        return _trace(self.fun(), self.in_node, self.leaves, self.names).captures


def _trace(fun, in_node, leaves, names):
    graph = TensorTraceGraph()
    values = []
    input_names = []
    for leaf, name in zip(leaves, names):
        kind = type(leaf)
        if kind is Tensor or kind is TraceTensor or kind is _Aval:
            values.append(graph.input(name, leaf.shape, leaf.dtype))
            input_names.append(name)
        elif kind is _Static:
            values.append(leaf.value)
        else:
            values.append(leaf)
    # While `fun` runs, tracers of enclosing traces that meet this graph are
    # lifted into it as capture inputs (see the module notes).
    graph._begin_trace()
    try:
        result = fun(*_unflatten(in_node, iter(values)))
        outputs = []
        out_node = _flatten(result, outputs)
        outputs = [
            graph._lift(output) if _is_traced(output) else output for output in outputs
        ]
    finally:
        captured = graph._end_trace()
    if not captured:
        return _Staged(graph, input_names, outputs, out_node)
    return _Staged(
        graph,
        input_names + [name for name, _ in captured],
        outputs,
        out_node,
        [source for _, source in captured],
        _Probe(fun, in_node, leaves, names),
    )


def _stage(fun, in_node, leaves, names):
    if isinstance(fun, _Transform):
        return fun._stage(in_node, leaves, names)
    return _trace(fun, in_node, leaves, names)


def _is_traced(value):
    return type(value) is TraceTensor


def _splice(staged, bindings):
    """Inlines `staged` into the enclosing trace, binding its inputs in order
    to `bindings`, and returns its result pytree; constant results (Python
    scalars, `None`) are returned unchanged."""
    traced = [output for output in staged.outputs if _is_traced(output)]
    spliced = iter(
        staged.graph._inline(staged.input_names, bindings, traced) if traced else ()
    )
    flat = [
        next(spliced) if _is_traced(output) else output for output in staged.outputs
    ]
    return _unflatten(staged.out_node, iter(flat))


def _traced_like(value, shape, dtype, what):
    """`value` as an inline binding for a callee input of `shape` and `dtype`:
    a tracer of the enclosing trace, a Python number for a scalar input,
    which the bridge binds as a constant of `dtype` (as `_as_like` adopts the
    primal dtype for a number), or an eager array, bound as a constant."""
    if type(value) is TraceTensor:
        actual_shape, actual_dtype = value.shape, value.dtype
    elif type(value) in _STATIC_TYPES:
        actual_shape, actual_dtype = [], dtype
    else:
        value = _as_array_leaf(value)
        actual_shape, actual_dtype = value.shape, value.dtype
    if actual_shape != shape or actual_dtype != dtype:
        raise TypeError(
            f"{what} must have shape {shape} and dtype {dtype!r} to match its primal, got "
            f"shape {actual_shape} and dtype {actual_dtype!r}"
        )
    return value


class _Program:
    """A compiled program with its output assembly: `template` holds, per
    flat result leaf, the index of an executable output or a constant."""

    __slots__ = ("executable", "template", "out_node")

    def __init__(self, graph, input_names, outputs, out_node, target="cpu", ordinal=0):
        traced = [output for output in outputs if _is_traced(output)]
        self.executable = (
            graph._compile(traced, input_names, target, ordinal) if traced else None
        )
        self.template = []
        index = 0
        for output in outputs:
            if _is_traced(output):
                self.template.append(index)
                index += 1
            else:
                self.template.append(_Const(output))
        self.out_node = out_node

    def run(self, inputs):
        results = self.executable(inputs) if self.executable is not None else ()
        flat = [
            results[slot] if type(slot) is int else slot.value for slot in self.template
        ]
        if self.out_node is _LEAF:
            return flat[0]
        return _unflatten(self.out_node, iter(flat))


class _Const:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _CapturedTracers(Exception):
    """Raised instead of compiling a staged function that captured tracers
    of an enclosing trace: a call with eager arguments inside a trace, which
    is inlined into that trace instead. `value` holds the staged graphs."""

    def __init__(self, value):
        super().__init__("the staged function captured tracers of an enclosing trace")
        self.value = value


def _compilable(value):
    """Staged graphs for compilation (a tuple led by the `_Staged` that
    holds the captures), which needs a graph without captures."""
    if value[0].captures:
        raise _CapturedTracers(value)
    return value


# -- trace caches ----------------------------------------------------------------


class _CaptureKey(tuple):
    """`(signature key, capture avals)`: the key of a staged graph that
    captured tracers of shapes and dtypes `capture avals`."""


class _TraceCache:
    """Compiled programs (or staged graphs) of one transformed function
    keyed by signature. `probes` maps the signature of a staged function
    that captured tracers to its `_Probe`; such graphs are keyed by
    `_CaptureKey`."""

    __slots__ = ("entries", "max_traces", "probes")

    def __init__(self, max_traces):
        self.entries = {}
        self.max_traces = max_traces
        self.probes = {}

    def lookup(self, key, build, transform):
        entry = self.entries.get(key)
        if entry is None:
            self.ensure_room(key, transform)
            entry = build()
            self.entries[key] = entry
        return entry

    def store_staged(self, key, value, transform):
        """Caches the staged graphs `value` of a call with signature `key`
        for inlining (see `_Transform._inline_staged`) and returns the
        tracers they captured."""
        captures = value[0].captures
        if captures:
            self.probes[key] = value[0].probe
        if key in self.probes:
            key = _CaptureKey((key, _avals(captures)))
        self.ensure_room(key, transform)
        for staged in value:
            if type(staged) is _Staged:
                staged.captures = None  # the cache must not keep old traces alive
        self.entries[key] = value
        return list(captures)

    def ensure_room(self, key, transform):
        if len(self.entries) >= self.max_traces:
            cached = "\n".join(f"  {_describe_signature(k)}" for k in self.entries)
            raise RetraceLimitError(
                f"{transform!r} exceeded max_traces={self.max_traces}; cached signatures:\n"
                f"{cached}\nnew signature:\n  {_describe_signature(key)}\n"
                "Python scalar arguments are static and part of the signature: pass "
                "quabla.array(value, dtype=...) to vary one without retracing, or raise "
                "max_traces with quabla.jit"
            )


# Trace caches by root Python function, then by transform chain, so that
# `quabla.grad(f)(x)` re-created in a loop reuses the trace of `f`, as a jit
# cache keyed on the function does. Weak keys drop the programs with `f`.
_CACHES = weakref.WeakKeyDictionary()
# Chain suffix of the caches of staged graphs for calls on tracers; no
# transform has this config, so these never share a cache with a program.
_INLINED = ("inlined",)


class _Transform:
    """Base of the transformed callables. `_config` is a hashable description
    of this transform; `_chain` lists the configs from the innermost
    transform out, which with the root function identifies the program."""

    _max_traces = _DEFAULT_MAX_TRACES

    def __init__(self, fun, config):
        if not callable(fun):
            raise TypeError(f"expected a callable, got {type(fun).__name__}")
        self._fun = fun
        self._config = config
        if isinstance(fun, _Transform):
            self._root = fun._root
            self._chain = fun._chain + (config,)
        else:
            self._root = fun
            self._chain = (config,)
        self._cache = None
        self._inlined_cache = None
        self._converted_by_count = {}

    def _own_differentiated(self, count):
        """Positions this transform differentiates for `count` arguments."""
        return ()

    def _converted_positions(self, count):
        """Positions whose Python scalars become arrays: the ones this
        transform or an inner transform differentiates."""
        converted = self._converted_by_count.get(count)
        if converted is None:
            converted = frozenset(self._own_differentiated(count))
            if isinstance(self._fun, _Transform):
                converted |= self._fun._converted_positions(count)
            self._converted_by_count[count] = converted
        return converted

    def _trace_cache(self):
        """Compiled programs, keyed by argument signature."""
        cache = self._cache
        if cache is None:
            cache = self._cache = self._cache_for(self._chain)
        return cache

    def _inline_cache(self):
        """Staged graphs for calls on tracers, keyed like `_trace_cache`."""
        cache = self._inlined_cache
        if cache is None:
            cache = self._inlined_cache = self._cache_for(self._chain + (_INLINED,))
        return cache

    def _cache_for(self, chain):
        try:
            by_chain = _CACHES.get(self._root)
            if by_chain is None:
                by_chain = _CACHES[self._root] = {}
        except TypeError:
            # The root cannot be weakly referenced or hashed: cache on this
            # object only.
            by_chain = {}
        cache = by_chain.get(chain)
        if cache is None:
            cache = by_chain[chain] = _TraceCache(self._max_traces)
        return cache

    def _names(self, in_node):
        return _leaf_names(self._root, in_node)

    def _inlines(self, key):
        """Whether a call with eager arguments and signature `key` must be
        inlined into the enclosing trace instead of compiled: it runs inside
        a trace and its function captured tracers when it was staged."""
        return key in self._inline_cache().probes and TensorTraceGraph._tracing()

    def _compiled(self, key, compile):
        """The compiled program for `key`, or `None` when the function
        captured tracers of an enclosing trace, which requires inlining."""
        try:
            return self._trace_cache().lookup(key, compile, self)
        except _CapturedTracers as captured:
            self._inline_cache().store_staged(key, captured.value, self)
            return None

    def __call__(self, *args):
        converted = self._converted_positions(len(args))
        in_node, leaves, arrays, key, traced = _signature(args, (), converted)
        if traced or self._inlines(key):
            return self._call_traced(args, (), converted)
        program = self._compiled(key, lambda: self._compile(in_node, leaves))
        if program is None:
            return self._call_traced(args, (), converted)
        return program.run(arrays)

    def _compile(self, in_node, leaves):
        (staged,) = _compilable((self._stage(in_node, leaves, self._names(in_node)),))
        return _Program(
            staged.graph, staged.input_names, staged.outputs, staged.out_node
        )

    def _call_traced(self, args, static_argnums, converted_argnums):
        """A call on tracers of an enclosing trace: stages this transform for
        their shapes and dtypes and inlines it into that trace."""
        in_node, leaves, key, bindings = _traced_signature(
            args, static_argnums, converted_argnums
        )
        (staged,), captures = self._inline_staged(
            key, lambda: (self._stage(in_node, leaves, self._names(in_node)),)
        )
        return _splice(staged, bindings + captures)

    def _inline_staged(self, key, stage):
        """The staged graphs of a call on tracers with signature `key`, from
        the inline cache or `stage()`, which returns a tuple of them led by
        the `_Staged` that holds the captures; and the tracers of enclosing
        traces that this call captures, to bind after its argument inputs.

        A function that captured nothing is staged once per signature. One
        that captured tracers is traced again on every call to find the
        tracers it captures now (a closure may refer to a different tracer
        each time); the transformed graph is reused when their shapes and
        dtypes match those it was staged with, and inlining binds it to them.
        """
        cache = self._inline_cache()
        probe = cache.probes.get(key)
        if probe is None:
            value = cache.entries.get(key)
            if value is not None:
                return value, []
        else:
            captures = probe.captures()
            value = cache.entries.get(_CaptureKey((key, _avals(captures))))
            if value is not None:
                return value, captures
        value = stage()
        return value, cache.store_staged(key, value, self)

    def __repr__(self):
        name = getattr(self._root, "__qualname__", None) or repr(self._root)
        for config in self._chain:
            name = f"{config[0]}({name})"
        return name


def _stage_value_and_grad(staged, in_node, leaves, names, argnums, has_aux, what):
    """Reverse mode over a staged scalar function: returns the value part
    (`value` or `(value, aux)`) and the gradients of the `argnums` leaves."""
    out_node = staged.out_node
    outputs = staged.outputs
    if has_aux:
        if not (
            type(out_node) is tuple and out_node[0] is tuple and len(out_node[1]) == 2
        ):
            raise TypeError(
                f"{what} with has_aux=True requires fun to return a (value, aux) pair"
            )
        value_node, aux_node = out_node[1]
    else:
        value_node, aux_node = out_node, None
    if value_node is not _LEAF:
        raise TypeError(
            f"{what} requires fun to return a single scalar array as its value"
        )
    value = outputs[0]
    if not _is_traced(value):
        raise ValueError(
            f"{what} requires the value to be computed from the arguments, got a constant "
            f"{type(value).__name__}"
        )
    if value.shape != []:
        raise ValueError(
            f"{what} requires a scalar output, got shape {value.shape}; use jacobian or vjp "
            "for non-scalar outputs"
        )
    aux = outputs[1:]
    retained = [value] + [leaf for leaf in aux if _is_traced(leaf)]
    graph, rebuilt, gradients = staged.graph._symbolic_vjp([value], [None], retained)
    rebuilt = iter(rebuilt)
    value = next(rebuilt)
    aux = [next(rebuilt) if _is_traced(leaf) else leaf for leaf in aux]

    ranges = _argument_ranges(in_node)
    grad_leaves = []
    for argnum in argnums:
        for position in ranges[argnum]:
            leaf = leaves[position]
            if type(leaf) not in _ARRAY_LEAVES:
                raise TypeError(
                    f"{what} cannot differentiate with respect to static argument {argnum}"
                )
            grad_leaves.append(
                None if leaf.dtype == bool_ else gradients[names[position]]
            )
    if len(argnums) == 1 and not isinstance(argnums, _Tuple):
        grads_node = in_node[1][argnums[0]]
    else:
        grads_node = (tuple, tuple(in_node[1][argnum] for argnum in argnums))
    return graph, value, value_node, aux, aux_node, grad_leaves, grads_node


def _avals(tracers):
    return tuple((tuple(tracer.shape), tracer.dtype) for tracer in tracers)


class _Tuple(tuple):
    """`argnums` given as a tuple, so a one-element tuple keeps its tuple
    structure in the result (`argnums=(0,)` gives `(grad,)`)."""


def _argnums_config(argnums):
    if isinstance(argnums, int):
        return argnums
    return _Tuple(argnums)


class _ValueAndGrad(_Transform):
    def __init__(self, fun, argnums, has_aux, kind):
        argnums = _argnums_config(argnums)
        super().__init__(fun, (kind, argnums, bool(has_aux)))
        self._argnums = argnums
        self._has_aux = bool(has_aux)
        self._kind = kind

    def _own_differentiated(self, count):
        return self._positions(count)

    def _positions(self, count):
        positions = _normalize_argnums(self._argnums, count, "argnums")
        return _Tuple(positions) if isinstance(self._argnums, _Tuple) else positions

    def _stage(self, in_node, leaves, names):
        staged = _stage(self._fun, in_node, leaves, names)
        argnums = self._positions(len(in_node[1]))
        graph, value, value_node, aux, aux_node, grad_leaves, grads_node = (
            _stage_value_and_grad(
                staged, in_node, leaves, names, argnums, self._has_aux, self._kind
            )
        )
        value_part = [value] + aux
        value_part_node = (
            (tuple, (value_node, aux_node)) if self._has_aux else value_node
        )
        if self._kind == "value_and_grad":
            outputs = value_part + grad_leaves
            out_node = (tuple, (value_part_node, grads_node))
        elif self._has_aux:
            outputs = grad_leaves + aux
            out_node = (tuple, (grads_node, aux_node))
        else:
            outputs = grad_leaves
            out_node = grads_node
        return _Staged(
            graph, staged.input_names, outputs, out_node, staged.captures, staged.probe
        )


class _Jit(_Transform):
    def __init__(self, fun, device, static_argnums, max_traces):
        self._target, self._ordinal = require_device(device, "jit")
        self._device = device or "cpu"
        self._warned_precision = False
        if isinstance(static_argnums, int):
            static_argnums = (static_argnums,)
        static_argnums = tuple(static_argnums)
        if (
            not isinstance(max_traces, int)
            or isinstance(max_traces, bool)
            or max_traces < 1
        ):
            raise ValueError(f"max_traces must be a positive int, got {max_traces!r}")
        self._max_traces = max_traces
        super().__init__(
            fun, ("jit", self._target, self._ordinal, static_argnums, max_traces)
        )
        self._static_argnums = static_argnums
        self._static_by_count = {}

    def _stage(self, in_node, leaves, names):
        return _stage(self._fun, in_node, leaves, names)

    def _compile(self, in_node, leaves):
        (staged,) = _compilable((self._stage(in_node, leaves, self._names(in_node)),))
        return self._compile_staged(staged)

    def _compile_staged(self, staged):
        program = _Program(
            staged.graph,
            staged.input_names,
            staged.outputs,
            staged.out_node,
            self._target,
            self._ordinal,
        )
        if self._target != "cpu" and not self._warned_precision:
            if any(
                _is_traced(output) and output.dtype == float64
                for output in staged.outputs
            ) or any(
                staged.graph.input(name).dtype == float64 for name in staged.input_names
            ):
                warnings.warn(
                    f"{self._target} executes float64 programs as float32",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_precision = True
        return program

    def lower(self, *args):
        """Trace without execution; ShapeDtype leaves need no host storage."""
        abstract = []
        node = _flatten(args, abstract)
        args = _unflatten(
            node,
            iter(
                [
                    _Aval(list(leaf.shape), leaf.dtype)
                    if isinstance(leaf, ShapeDtype)
                    else leaf
                    for leaf in abstract
                ]
            ),
        )
        statics = self._statics(len(args)) if self._static_argnums else ()
        converted = self._converted_positions(len(args))
        in_node, leaves, _, key, traced = _signature(args, statics, converted)
        if traced:
            raise TypeError(
                "lower() requires arrays or ShapeDtype, not enclosing tracers"
            )
        (staged,) = _compilable((self._stage(in_node, leaves, self._names(in_node)),))
        return _Lowered(self, staged, key, statics, converted)

    def _statics(self, count):
        statics = self._static_by_count.get(count)
        if statics is None:
            statics = frozenset(
                _normalize_argnums(self._static_argnums, count, "static_argnums")
            )
            self._static_by_count[count] = statics
        return statics

    def __call__(self, *args):
        count = len(args)
        statics = self._statics(count) if self._static_argnums else ()
        converted = self._converted_positions(count)
        in_node, leaves, arrays, key, traced = _signature(args, statics, converted)
        if not (traced or self._inlines(key)):
            program = self._compiled(key, lambda: self._compile(in_node, leaves))
            if program is not None:
                return program.run(arrays)
        if isinstance(self._fun, _Transform):
            return self._call_traced(args, statics, converted)
        # Inside another trace, tracing through a plain function inlines it.
        return self._fun(*args)


class _Lowered:
    """A staged ordered-output program, compiled only on explicit request."""

    def __init__(self, jit, staged, key, statics, converted):
        self.program = staged.graph._as_program(
            [output for output in staged.outputs if _is_traced(output)]
        )
        self._jit = jit
        self._staged = staged
        self._key = key
        self._statics = statics
        self._converted = converted
        self._compiled_program = None

    def as_text(self):
        return self.program.lower_text()

    def compile(self):
        if self._compiled_program is None:
            self._compiled_program = self._jit._compile_staged(self._staged)

        def execute(*args):
            _, _, arrays, key, traced = _signature(args, self._statics, self._converted)
            if traced or key != self._key:
                raise ValueError(
                    "compiled lower() call must match its input structure, shapes, dtypes and statics"
                )
            return self._compiled_program.run(arrays)

        return execute


# -- forward mode ------------------------------------------------------------------


def _as_like(value, shape, dtype, what):
    """`value` as a Tensor of `shape` and `dtype`; a Python number is a weak
    scalar that adopts `dtype`, every other value must match it exactly."""
    if type(value) in _STATIC_TYPES:
        value = asarray(value, dtype)
    else:
        value = _as_array_leaf(value)
    if value.shape != shape or value.dtype != dtype:
        raise TypeError(
            f"{what} must have shape {shape} and dtype {dtype!r} to match its primal, got "
            f"shape {value.shape} and dtype {value.dtype!r}"
        )
    return value


def _zeros_like_constant(value):
    if isinstance(value, Tensor):
        return zeros(value.shape, value.dtype)
    if type(value) in _STATIC_TYPES:
        return 0.0
    return None


class _JvpProgram:
    __slots__ = ("program", "tangent_positions")

    def __init__(self, program, tangent_positions):
        self.program = program
        self.tangent_positions = tangent_positions


class _Jvp(_Transform):
    def __init__(self, fun):
        super().__init__(fun, ("jvp",))

    def __call__(self, primals, tangents):
        if type(primals) not in (tuple, list) or type(tangents) not in (tuple, list):
            raise TypeError("jvp takes primals and tangents as tuples or lists")
        if len(primals) != len(tangents):
            raise ValueError(
                f"jvp got {len(primals)} primals and {len(tangents)} tangents; they must match"
            )
        primals = tuple(primals)
        converted = frozenset(range(len(primals)))
        in_node, leaves, arrays, key, traced = _signature(primals, (), converted)
        tangent_leaves = []
        tangent_node = _flatten(tuple(tangents), tangent_leaves)
        if (
            traced
            or self._inlines(key)
            or any(_is_traced(tangent) for tangent in tangent_leaves)
        ):
            return self._jvp_traced(primals, converted, tangent_node, tangent_leaves)
        entry = self._compiled(key, lambda: self._compile(in_node, leaves))
        if entry is None:
            return self._jvp_traced(primals, converted, tangent_node, tangent_leaves)
        if tangent_node != in_node:
            raise ValueError(
                "jvp tangents must have the pytree structure of the primals"
            )
        inputs = list(arrays)
        for position in entry.tangent_positions:
            primal = leaves[position]
            inputs.append(
                _as_like(
                    tangent_leaves[position],
                    primal.shape,
                    primal.dtype,
                    "a jvp tangent",
                )
            )
        return entry.program.run(inputs)

    def _jvp_traced(self, primals, converted, tangent_node, tangent_leaves):
        """`jvp` on tracers of an enclosing trace: the staged JVP graph is
        inlined with its tangent inputs bound to the traced tangents."""
        in_node, leaves, key, bindings = _traced_signature(primals, (), converted)
        (staged, tangent_positions), captures = self._inline_staged(
            key, lambda: self._stage_jvp(in_node, leaves)
        )
        if tangent_node != in_node:
            raise ValueError(
                "jvp tangents must have the pytree structure of the primals"
            )
        bindings += captures
        for position in tangent_positions:
            primal = leaves[position]
            bindings.append(
                _traced_like(
                    tangent_leaves[position],
                    primal.shape,
                    primal.dtype,
                    "a jvp tangent",
                )
            )
        return _splice(staged, bindings)

    def _compile(self, in_node, leaves):
        staged, tangent_positions = _compilable(self._stage_jvp(in_node, leaves))
        program = _Program(
            staged.graph, staged.input_names, staged.outputs, staged.out_node
        )
        return _JvpProgram(program, tangent_positions)

    def _stage_jvp(self, in_node, leaves):
        """The forward-mode graph, taking the primal inputs and the capture
        inputs followed by one tangent input per leaf in the returned
        `tangent_positions`; captured tracers get no tangent."""
        names = self._names(in_node)
        staged = _stage(self._fun, in_node, leaves, names)
        tangent_positions = [
            position
            for position, leaf in enumerate(leaves)
            if type(leaf) in _ARRAY_LEAVES and leaf.dtype != bool_
        ]
        if not tangent_positions:
            raise ValueError("jvp requires at least one floating-point primal")
        tangent_names = {
            names[position]: _TANGENT_PREFIX + names[position]
            for position in tangent_positions
        }
        traced = [output for output in staged.outputs if _is_traced(output)]
        if traced:
            graph, values, tangents = staged.graph._symbolic_jvp(traced, tangent_names)
        else:
            graph, values, tangents = staged.graph, [], []
        values, tangents = iter(values), iter(tangents)
        primal_out = []
        tangent_out = []
        for output in staged.outputs:
            if _is_traced(output):
                primal_out.append(next(values))
                tangent_out.append(next(tangents))
            else:
                primal_out.append(output)
                tangent_out.append(_zeros_like_constant(output))
        staged = _Staged(
            graph,
            staged.input_names + [tangent_names[names[p]] for p in tangent_positions],
            primal_out + tangent_out,
            (tuple, (staged.out_node, staged.out_node)),
            staged.captures,
            staged.probe,
        )
        return staged, tangent_positions


# -- reverse mode with a runtime cotangent ------------------------------------------


class _VjpProgram:
    """The forward program of `vjp` and the reverse program of its pullback,
    which takes the primals and one cotangent per seeded output leaf."""

    __slots__ = ("forward", "reverse", "cotangent_specs", "out_node", "primal_count")

    def __init__(self, forward, reverse, cotangent_specs, out_node, primal_count):
        self.forward = forward
        self.reverse = reverse
        self.cotangent_specs = cotangent_specs
        self.out_node = out_node
        self.primal_count = primal_count


class VjpFunction:
    """The pullback returned by `quabla.vjp`: `vjp_fun(cotangent)` maps a
    cotangent shaped like the primal output to one cotangent pytree per
    primal. Each call evaluates the forward pass again (no residuals are
    stored). Called on a traced cotangent inside another transform, it
    inlines the reverse-mode graph with the eager primals as constants."""

    __slots__ = ("_program", "_arrays", "_transform", "_primals")

    def __init__(self, program, arrays, transform, primals):
        self._program = program
        self._arrays = arrays
        self._transform = transform
        self._primals = primals

    def __call__(self, cotangent):
        program = self._program
        cotangent_leaves = []
        if _flatten(cotangent, cotangent_leaves) != program.out_node:
            raise ValueError(
                "the cotangent must have the pytree structure of the primal output"
            )
        if any(_is_traced(leaf) for leaf in cotangent_leaves):
            return self._transform._traced_pullback(self._primals)(cotangent)
        inputs = list(self._arrays)
        for position, shape, dtype in program.cotangent_specs:
            inputs.append(
                _as_like(cotangent_leaves[position], shape, dtype, "a vjp cotangent")
            )
        return program.reverse.run(inputs)

    def __repr__(self):
        return f"VjpFunction(primals={self._program.primal_count})"


class _TracedVjpFunction:
    """The pullback returned by `quabla.vjp` called on tracers inside
    another transform: each call inlines the staged reverse-mode graph into
    the enclosing trace, with the cotangents bound to tracers or Python
    numbers."""

    __slots__ = ("_reverse", "_cotangent_specs", "_out_node", "_bindings")

    def __init__(self, reverse, cotangent_specs, out_node, bindings):
        self._reverse = reverse
        self._cotangent_specs = cotangent_specs
        self._out_node = out_node
        self._bindings = bindings

    def __call__(self, cotangent):
        cotangent_leaves = []
        if _flatten(cotangent, cotangent_leaves) != self._out_node:
            raise ValueError(
                "the cotangent must have the pytree structure of the primal output"
            )
        bindings = list(self._bindings)
        for position, shape, dtype in self._cotangent_specs:
            bindings.append(
                _traced_like(
                    cotangent_leaves[position], shape, dtype, "a vjp cotangent"
                )
            )
        return _splice(self._reverse, bindings)

    def __repr__(self):
        return f"VjpFunction(primals={len(self._reverse.out_node[1])}, traced)"


class _Vjp(_Transform):
    def __init__(self, fun, has_aux):
        super().__init__(fun, ("vjp", bool(has_aux)))
        self._has_aux = bool(has_aux)

    def __call__(self, *primals):
        converted = frozenset(range(len(primals)))
        in_node, leaves, arrays, key, traced = _signature(primals, (), converted)
        entry = None
        if not (traced or self._inlines(key)):
            entry = self._compiled(key, lambda: self._compile(in_node, leaves))
        if entry is None:
            in_node, leaves, key, bindings = _traced_signature(primals, (), converted)
            (forward, reverse, cotangent_specs, out_node), captures = (
                self._inline_staged(key, lambda: self._stage_vjp(in_node, leaves))
            )
            bindings += captures
            result = _splice(forward, bindings)
            pullback = _TracedVjpFunction(reverse, cotangent_specs, out_node, bindings)
        else:
            result = entry.forward.run(arrays)
            pullback = VjpFunction(entry, arrays, self, primals)
        if self._has_aux:
            out, aux = result
            return out, pullback, aux
        return result, pullback

    def _traced_pullback(self, primals):
        """The pullback at eager `primals` for a traced cotangent: the staged
        reverse graph, inlined with the primals bound as constants."""
        converted = frozenset(range(len(primals)))
        in_node, leaves, key, bindings = _traced_signature(primals, (), converted)
        (_, reverse, cotangent_specs, out_node), captures = self._inline_staged(
            key, lambda: self._stage_vjp(in_node, leaves)
        )
        return _TracedVjpFunction(
            reverse, cotangent_specs, out_node, bindings + captures
        )

    def _compile(self, in_node, leaves):
        forward, reverse, cotangent_specs, out_node = _compilable(
            self._stage_vjp(in_node, leaves)
        )
        return _VjpProgram(
            _Program(
                forward.graph, forward.input_names, forward.outputs, forward.out_node
            ),
            _Program(
                reverse.graph, reverse.input_names, reverse.outputs, reverse.out_node
            ),
            cotangent_specs,
            out_node,
            len(in_node[1]),
        )

    def _stage_vjp(self, in_node, leaves):
        """The forward graph, the reverse graph (the primal and capture inputs
        followed by one cotangent input per entry of `cotangent_specs`), the cotangent
        specs `(output leaf position, shape, dtype)`, and the structure of
        the differentiated output."""
        names = self._names(in_node)
        staged = _stage(self._fun, in_node, leaves, names)
        out_node, outputs = staged.out_node, staged.outputs
        if self._has_aux:
            if not (
                type(out_node) is tuple
                and out_node[0] is tuple
                and len(out_node[1]) == 2
            ):
                raise TypeError(
                    "vjp with has_aux=True requires fun to return an (out, aux) pair"
                )
            out_node = out_node[1][0]
            outputs = outputs[: _leaf_count(out_node)]
        seeded = []
        cotangent_specs = []
        for position, output in enumerate(outputs):
            if _is_traced(output) and output.dtype != bool_:
                seeded.append(output)
                cotangent_specs.append((position, output.shape, output.dtype))
        if not seeded:
            raise ValueError(
                "vjp requires a floating-point output computed from the primals"
            )
        cotangent_names = [
            f"{_COTANGENT_PREFIX}{index}" for index in range(len(seeded))
        ]
        graph, _, gradients = staged.graph._symbolic_vjp(seeded, cotangent_names, [])
        grad_outputs = [
            gradients[name]
            if type(leaf) in _ARRAY_LEAVES and leaf.dtype != bool_
            else None
            for leaf, name in zip(leaves, names)
        ]
        reverse = _Staged(
            graph,
            staged.input_names + cotangent_names,
            grad_outputs,
            in_node,
            staged.captures,
            staged.probe,
        )
        return staged, reverse, cotangent_specs, out_node


# -- vectorization -------------------------------------------------------------------


def _axes_config(axes, what):
    """`in_axes`/`out_axes` in a hashable form: an int or `None` applies to
    a whole subtree, and tuples, lists, and dicts of them are pytree
    prefixes. `None` is a leaf here, unlike in argument pytrees."""
    if axes is None or (isinstance(axes, int) and not isinstance(axes, bool)):
        return axes
    kind = type(axes)
    if kind is tuple or kind is list:
        return (tuple, tuple(_axes_config(child, what) for child in axes))
    if kind is dict:
        if not all(type(key) is str for key in axes):
            raise TypeError(f"vmap {what} dict keys must be strings, got {axes!r}")
        keys = tuple(sorted(axes))
        return (dict, keys, tuple(_axes_config(axes[key], what) for key in keys))
    raise TypeError(
        f"vmap {what} must be an int, None, or a tuple, list, or dict of them, got {axes!r}"
    )


def _leaf_axes(spec, node, axes, what):
    """Appends the axis of every leaf of the structure `node` to `axes`,
    where `spec` (from `_axes_config`) is a prefix of that structure. Lists
    and tuples match each other, as their leaves are positional in both."""
    if spec is None or isinstance(spec, int):
        axes.extend([spec] * _leaf_count(node))
        return
    if spec[0] is tuple and type(node) is tuple and node[0] in (tuple, list):
        if len(spec[1]) == len(node[1]):
            for child_spec, child in zip(spec[1], node[1]):
                _leaf_axes(child_spec, child, axes, what)
            return
    elif (
        spec[0] is dict
        and type(node) is tuple
        and node[0] is dict
        and spec[1] == node[1]
    ):
        for child_spec, child in zip(spec[2], node[2]):
            _leaf_axes(child_spec, child, axes, what)
        return
    raise ValueError(
        f"vmap {what} {_render_axes(spec)!r} does not match the structure {_describe(node)} of "
        f"the {'arguments' if what == 'in_axes' else 'result'}: give an int or None, or a "
        "tuple/dict with one entry per element"
    )


def _render_axes(spec):
    """The user-facing form of an `_axes_config` spec, for messages."""
    if spec is None or isinstance(spec, int):
        return spec
    if spec[0] is dict:
        return {key: _render_axes(child) for key, child in zip(spec[1], spec[2])}
    return tuple(_render_axes(child) for child in spec[1])


def _move_batch_axis(value, axis):
    """`value` (batch axis leading) with the batch axis moved to `axis`."""
    if axis == 0:
        return value
    rank = len(value.shape)
    return value.transpose(list(range(1, axis + 1)) + [0] + list(range(axis + 1, rank)))


class _Vmap(_Transform):
    """Vectorization by staging: the function is staged for one example
    (the mapped axes removed), then that graph is spliced into the caller's
    graph with every node that depends on a mapped argument batched
    (`TensorIr::inline_batched`). The result is an ordinary graph, so vmap
    composes with every transform, including another vmap."""

    def __init__(self, fun, in_axes, out_axes):
        in_config = _axes_config(in_axes, "in_axes")
        out_config = _axes_config(out_axes, "out_axes")
        super().__init__(fun, ("vmap", in_config, out_config))
        self._in_axes = in_config
        self._out_axes = out_config

    def _stage(self, in_node, leaves, names):
        in_axes = []
        _leaf_axes(self._in_axes, in_node, in_axes, "in_axes")
        graph = TensorTraceGraph()
        example_leaves = []
        bound = {}
        batch = None
        for leaf, name, axis in zip(leaves, names, in_axes):
            if type(leaf) not in _ARRAY_LEAVES:
                if axis is not None:
                    value = leaf.value if type(leaf) is _Static else leaf
                    raise ValueError(
                        f"vmap cannot map the static argument leaf {name!r} ({value!r}) along "
                        f"axis {axis}: Python scalars are constants; pass an array to map it, "
                        "or give it in_axes None"
                    )
                example_leaves.append(leaf)
                continue
            tracer = graph.input(name, leaf.shape, leaf.dtype)
            if axis is None:
                example_leaves.append(_Aval(leaf.shape, leaf.dtype))
                bound[name] = (tracer, False)
                continue
            shape = list(leaf.shape)
            if not -len(shape) <= axis < len(shape):
                raise ValueError(
                    f"vmap in_axes {axis} is out of range for the argument leaf {name!r} of "
                    f"shape {shape}"
                )
            axis %= len(shape)
            if batch is None:
                batch = shape[axis]
            elif shape[axis] != batch:
                raise ValueError(
                    f"vmap got inconsistent sizes for the mapped axes: {name!r} has "
                    f"{shape[axis]} along axis {axis}, the earlier mapped leaves {batch}"
                )
            if axis:
                tracer = tracer.transpose(
                    [axis] + [a for a in range(len(shape)) if a != axis]
                )
            example_leaves.append(_Aval(shape[:axis] + shape[axis + 1 :], leaf.dtype))
            bound[name] = (tracer, True)
        if batch is None:
            raise ValueError(
                "vmap needs at least one mapped array argument, but in_axes maps none"
            )
        inner = _stage(self._fun, in_node, example_leaves, names)
        # Captured tracers of enclosing traces are unmapped inputs here too.
        capture_names = _capture_names(inner)
        for name, source in zip(capture_names, inner.captures):
            bound[name] = (graph.input(name, source.shape, source.dtype), False)
        traced = [output for output in inner.outputs if _is_traced(output)]
        if traced:
            bindings = [bound[name] for name in inner.input_names]
            spliced = iter(
                inner.graph._inline_batched(
                    inner.input_names,
                    [tracer for tracer, _ in bindings],
                    [mapped for _, mapped in bindings],
                    batch,
                    traced,
                )
            )
        out_axes = []
        _leaf_axes(self._out_axes, inner.out_node, out_axes, "out_axes")
        outputs = []
        for output, out_axis in zip(inner.outputs, out_axes):
            mapped = False
            if _is_traced(output):
                output, mapped = next(spliced)
            outputs.append(_place_output(graph, output, mapped, out_axis, batch))
        input_names = [name for name in names if name in bound] + capture_names
        return _Staged(
            graph, input_names, outputs, inner.out_node, inner.captures, inner.probe
        )


def _place_output(graph, value, mapped, out_axis, batch):
    """A vmap result leaf in the caller's layout: its batch axis at
    `out_axis`, where an unmapped value is broadcast over the batch, or the
    unmapped value itself for `out_axis=None`."""
    if out_axis is None:
        if mapped:
            raise ValueError(
                "vmap out_axes=None requires an output that does not depend on the mapped "
                "arguments, but this output does; give it an integer out_axes"
            )
        return value
    if not _is_traced(value):
        value = graph._constant(value if isinstance(value, Tensor) else asarray(value))
    if not mapped:
        value = value.broadcast_to([batch] + list(value.shape))
    rank = len(value.shape)
    if not -rank <= out_axis < rank:
        raise ValueError(
            f"vmap out_axes {out_axis} is out of range for an output of rank {rank} "
            "(batch axis included)"
        )
    return _move_batch_axis(value, out_axis % rank)


# -- dense Jacobians ------------------------------------------------------------------


def _size(shape):
    size = 1
    for extent in shape:
        size *= extent
    return size


def _graft(node, block_node):
    """The structure `node` with each leaf replaced by `block_node`."""
    if node is _LEAF:
        return block_node
    if node is None:
        return None
    if node[0] is dict:
        return (dict, node[1], tuple(_graft(child, block_node) for child in node[2]))
    return (node[0], tuple(_graft(child, block_node) for child in node[1]))


def _jacobian_basis_indices(graph, total):
    """Exact radix digits of flattened indices, using only linear storage.

    Every digit is below 2**24, so backend float32 execution preserves it.
    Broadcasting digits avoids Python lists and quadratic identity constants.
    Each padded coordinate vector has fewer than twice `total` elements.
    """
    radix = _JACOBIAN_INDEX_RADIX
    if total <= radix:
        return [graph._constant(arange(total, dtype=float64))]
    groups = (total + radix - 1) // radix
    padded = groups * radix
    low = (
        graph._constant(arange(radix, dtype=float64))
        .reshape([1, radix])
        .broadcast_to([groups, radix])
        .reshape([padded])
        .slice(0, 0, total)
    )
    high = [
        digit.reshape([groups, 1])
        .broadcast_to([groups, radix])
        .reshape([padded])
        .slice(0, 0, total)
        for digit in _jacobian_basis_indices(graph, groups)
    ]
    return [low] + high


class _Jacobian(_Transform):
    """Dense Jacobian using the smaller eligible input/output basis.

    Floating outputs use reverse mode when Q < P and the graph does not
    narrow gradients before F64 output blocks; otherwise the transform
    keeps forward mode. Bounded basis batches avoid a quadratic identity
    constant. The result stays one graph that inlines and differentiates
    normally, preserving pytree blocks and output dtypes."""

    def __init__(self, fun, argnums, kind="jacobian"):
        argnums = _argnums_config(argnums)
        super().__init__(fun, (kind, argnums))
        self._argnums = argnums

    def _own_differentiated(self, count):
        return self._positions(count)

    def _positions(self, count):
        return _normalize_argnums(self._argnums, count, "argnums")

    def _stage(self, in_node, leaves, names):
        kind = self._config[0]
        argnums = self._positions(len(in_node[1]))
        ranges = _argument_ranges(in_node)
        selected = []
        for argnum in argnums:
            for position in ranges[argnum]:
                leaf = leaves[position]
                if type(leaf) not in _ARRAY_LEAVES:
                    raise TypeError(
                        f"{kind} cannot differentiate with respect to static argument {argnum}"
                    )
                if leaf.dtype != bool_:
                    selected.append(position)
        if not selected:
            raise ValueError(f"{kind} requires a floating-point argument in argnums")
        staged = _stage(self._fun, in_node, leaves, names)
        graph = TensorTraceGraph()
        primals = {}
        for leaf, name in zip(leaves, names):
            if type(leaf) in _ARRAY_LEAVES:
                primals[name] = graph.input(name, leaf.shape, leaf.dtype)
        for name, source in zip(_capture_names(staged), staged.captures):
            primals[name] = graph.input(name, source.shape, source.dtype)
        sizes = [_size(leaves[position].shape) for position in selected]
        total = sum(sizes)
        offsets = []
        offset = 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        traced = [output for output in staged.outputs if _is_traced(output)]
        pieces = [[[] for _ in selected] for _ in traced]
        output_sizes = [_size(output.shape) for output in traced]
        output_total = sum(output_sizes)
        reverse = (
            0 < output_total < total
            and all(output.dtype != bool_ for output in traced)
            and (
                all(output.dtype == float32 for output in traced)
                or not staged.graph._has_f32_nodes
            )
        )
        if reverse:
            cotangent_names = []
            occupied = set(staged.input_names)
            for index in range(len(traced)):
                name = f"{_COTANGENT_PREFIX}jacobian/{index}"
                while name in occupied:
                    name += "_"
                occupied.add(name)
                cotangent_names.append(name)
            reverse_graph, _, gradients = staged.graph._symbolic_vjp(
                traced, cotangent_names, []
            )
            indices = _jacobian_basis_indices(graph, output_total)
            output_offsets = []
            offset = 0
            for size in output_sizes:
                output_offsets.append(offset)
                offset += size
            output_indices = [
                [
                    digit.slice(0, offset, offset + size).reshape([1, size])
                    for digit in indices
                ]
                for offset, size in zip(output_offsets, output_sizes)
            ]
            for start in range(0, output_total, _JACOBIAN_CHUNK_SIZE):
                stop = min(start + _JACOBIAN_CHUNK_SIZE, output_total)
                batch = stop - start
                rows = [
                    digit.slice(0, start, stop).reshape([batch, 1]) for digit in indices
                ]
                cotangents = []
                for output, columns in zip(traced, output_indices):
                    basis = rows[0].equal(columns[0])
                    for row, column in zip(rows[1:], columns[1:]):
                        basis = basis.logical_and(row.equal(column))
                    cotangents.append(
                        basis.astype(output.dtype).reshape([batch] + list(output.shape))
                    )
                spliced = reverse_graph._inline_batched(
                    staged.input_names + cotangent_names,
                    [primals[name] for name in staged.input_names] + cotangents,
                    [False] * len(staged.input_names) + [True] * len(cotangents),
                    batch,
                    [gradients[names[position]] for position in selected],
                )
                for index, (position, (columns, mapped)) in enumerate(
                    zip(selected, spliced)
                ):
                    if not mapped:
                        columns = columns.broadcast_to(
                            [batch] + list(leaves[position].shape)
                        )
                    for output, output_pieces, offset, size in zip(
                        traced, pieces, output_offsets, output_sizes
                    ):
                        lower, upper = max(start, offset), min(stop, offset + size)
                        if lower < upper:
                            output_pieces[index].append(
                                columns.slice(0, lower - start, upper - start).astype(
                                    output.dtype
                                )
                            )
        elif traced and total:
            tangent_names = {names[p]: _TANGENT_PREFIX + names[p] for p in selected}
            jvp_graph, _, jvp_tangents = staged.graph._symbolic_jvp(
                traced, tangent_names
            )
            indices = _jacobian_basis_indices(graph, total)
            leaf_indices = [
                [
                    digit.slice(0, offset, offset + size).reshape([1, size])
                    for digit in indices
                ]
                for offset, size in zip(offsets, sizes)
            ]
            for start in range(0, total, _JACOBIAN_CHUNK_SIZE):
                stop = min(start + _JACOBIAN_CHUNK_SIZE, total)
                batch = stop - start
                rows = [
                    digit.slice(0, start, stop).reshape([batch, 1]) for digit in indices
                ]
                tangents = []
                for position, columns in zip(selected, leaf_indices):
                    basis = rows[0].equal(columns[0])
                    for row, column in zip(rows[1:], columns[1:]):
                        basis = basis.logical_and(row.equal(column))
                    tangents.append(
                        basis.astype(leaves[position].dtype).reshape(
                            [batch] + list(leaves[position].shape)
                        )
                    )
                spliced = jvp_graph._inline_batched(
                    staged.input_names + [tangent_names[names[p]] for p in selected],
                    [primals[name] for name in staged.input_names] + tangents,
                    [False] * len(staged.input_names) + [True] * len(selected),
                    batch,
                    jvp_tangents,
                )
                for output, output_pieces, (columns, mapped) in zip(
                    traced, pieces, spliced
                ):
                    if not mapped:
                        columns = columns.broadcast_to([batch] + list(output.shape))
                    for leaf_pieces, offset, size in zip(output_pieces, offsets, sizes):
                        lower, upper = max(start, offset), min(stop, offset + size)
                        if lower < upper:
                            leaf_pieces.append(
                                columns.slice(0, lower - start, upper - start)
                            )
        pieces = iter(pieces)
        blocks = []
        for output in staged.outputs:
            if _is_traced(output):
                output_pieces = next(pieces)
            by_position = {}
            for index, (position, size) in enumerate(zip(selected, sizes)):
                leaf = leaves[position]
                if not _is_traced(output) or not size:
                    shape = (
                        list(output.shape)
                        if isinstance(output, (Tensor, TraceTensor))
                        else []
                    )
                    dtype = (
                        output.dtype
                        if isinstance(output, (Tensor, TraceTensor))
                        else leaf.dtype
                    )
                    by_position[position] = zeros(shape + list(leaf.shape), dtype)
                    continue
                chunks = output_pieces[index]
                block = chunks[0] if len(chunks) == 1 else _quabla.concat(chunks, 0)
                if not reverse:
                    block = _move_batch_axis(block, len(output.shape))
                by_position[position] = block.reshape(
                    list(output.shape) + list(leaf.shape)
                )
            for argnum in argnums:
                blocks.extend(by_position.get(position) for position in ranges[argnum])
        if len(argnums) == 1 and not isinstance(self._argnums, _Tuple):
            block_node = in_node[1][argnums[0]]
        else:
            block_node = (tuple, tuple(in_node[1][argnum] for argnum in argnums))
        return _Staged(
            graph,
            list(primals),
            blocks,
            _graft(staged.out_node, block_node),
            staged.captures,
            staged.probe,
        )


# -- public API ----------------------------------------------------------------------


def _new_grad(fun, argnums=0, has_aux=False):
    return _ValueAndGrad(fun, argnums, has_aux, "grad")


def _new_jit(fun, device=None, static_argnums=(), max_traces=_DEFAULT_MAX_TRACES):
    return _Jit(fun, device, static_argnums, max_traces)


def value_and_grad(fun, argnums=0, has_aux=False):
    """`fun` transformed to return `(value, grads)`, where `grads` has the
    pytree structure of the arguments selected by `argnums` (an int, or a
    tuple giving a tuple of gradients). `fun` must return a scalar array, or
    `(scalar, aux)` with `has_aux=True`, giving `((value, aux), grads)`.
    `bool_` leaves get `None` instead of a gradient. Staged and cached per
    argument signature (see the module notes); wrap it in `jit(device=...)`
    for device execution."""
    return _ValueAndGrad(fun, argnums, has_aux, "value_and_grad")


def grad(*args, **kwargs):
    """`grad(fun, argnums=0, has_aux=False)`: `fun` transformed to return the
    gradient of its scalar output with respect to the arguments selected by
    `argnums`, or `(grads, aux)` with `has_aux=True`; see `value_and_grad`.

    The v0.1 form `grad(function, input_specs, values, output_cotangent)` of
    the 2D `Matrix` API still works unchanged: it is recognized by its input
    spec list (design D17).
    """
    if _is_legacy_grad_call(args, kwargs):
        return _quabla.grad(*args, **kwargs)
    return _new_grad(*args, **kwargs)


def jit(*args, **kwargs):
    """`jit(fun, device=None, static_argnums=(), max_traces=8)`: `fun` traced
    and compiled on its first call per argument signature, then run from the
    cache. `device=None` means `"cpu"`; device targets are explicit.
    Arguments at `static_argnums` are static as a whole and must be hashable;
    a signature beyond `max_traces` raises `RetraceLimitError`.

    The v0.1 decorator form `jit(input_specs)` of the 2D `Matrix` API still
    works unchanged: it is recognized by its non-callable spec list (D17).
    """
    if _is_legacy_jit_call(args, kwargs):
        return _quabla.jit(*args, **kwargs)
    return _new_jit(*args, **kwargs)


grad.__signature__ = inspect.signature(_new_grad)
jit.__signature__ = inspect.signature(_new_jit)

_LEGACY_GRAD_KEYWORDS = frozenset(
    {"function", "input_specs", "values", "output_cotangent"}
)


def _is_input_specs(value):
    if isinstance(value, list):
        return True
    return isinstance(value, tuple) and not all(isinstance(item, int) for item in value)


def _is_legacy_grad_call(args, kwargs):
    # v0.1 takes four arguments with an input spec list second; the new form
    # takes at most three, with an int or a tuple of ints second.
    if len(args) > 3 or not _LEGACY_GRAD_KEYWORDS.isdisjoint(kwargs):
        return True
    return len(args) >= 2 and _is_input_specs(args[1])


def _is_legacy_jit_call(args, kwargs):
    # v0.1 takes a single input spec list, which is not callable.
    return "input_specs" in kwargs or (len(args) >= 1 and not callable(args[0]))


def jvp(fun, primals, tangents):
    """`(fun(*primals), J @ tangents)`: forward mode with one tangent per
    primal leaf, with the pytree structure, shape, and dtype of the primals
    (a Python number tangent adopts its primal's dtype). Python scalar
    primals become `float64` arrays; `bool_` primals take no tangent.
    `jvp(grad(f), (x,), (v,))` is a Hessian-vector product."""
    return _Jvp(fun)(primals, tangents)


def vjp(fun, *primals, has_aux=False):
    """`(fun(*primals), vjp_fun)`, or `(out, vjp_fun, aux)` with
    `has_aux=True`; `vjp_fun(cotangent)` returns a tuple with one cotangent
    pytree per primal (`None` for `bool_` leaves). Python scalar primals
    become `float64` arrays."""
    return _Vjp(fun, has_aux)(*primals)


def jacobian(fun, argnums=0):
    """`fun` transformed to return its dense Jacobian: for each output leaf,
    blocks of shape `[*out.shape, *in.shape]` with the pytree structure of
    the arguments selected by `argnums`. Uses reverse mode when floating
    output elements are fewer than selected input elements, forward mode
    otherwise. Graphs containing F32 nodes retain forward mode for F64
    output blocks to preserve their precision. Both directions use bounded basis batches and compose with
    other staged transforms. Direction selection can change last-bit
    rounding; it preserves shape, dtype, and pytree structure."""
    return _Jacobian(fun, argnums)


def hessian(fun, argnums=0):
    """`fun` transformed to return the dense Hessian of its scalar output,
    `jacobian(grad(fun, argnums), argnums)`: blocks of shape
    `[*in.shape, *in.shape]`, forward over reverse."""
    return _Jacobian(_ValueAndGrad(fun, argnums, False, "grad"), argnums, "hessian")


def vmap(fun, in_axes=0, out_axes=0):
    """`fun` vectorized over a batch axis, with JAX semantics. `in_axes`
    gives the mapped axis of each argument (an int), or `None` for an
    argument shared by every example; an int or `None` applies to every
    leaf of an argument, and a tuple, list, or dict gives axes per element
    as a pytree prefix of the arguments. The mapped axes must agree in size.
    `out_axes` places the batch axis of each result leaf the same way;
    `None` requires a result that does not depend on the mapped arguments.
    `fun` sees one example, without the mapped axes. Staged and cached like
    the other transforms; composes with them and with itself: for example
    `vmap(grad(grad(u)), in_axes=(0, None))` is the second derivative of a
    scalar `u(x, w)` at every point of `x`."""
    return _Vmap(fun, in_axes, out_axes)
