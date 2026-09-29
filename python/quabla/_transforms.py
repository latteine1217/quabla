"""Composable function transforms on the CPU (docs/api_v0_2_design.md,
sections 3.3-3.5 and 3.11, decisions D5-D8 and D16-D17; slice S2).

Every transform is staged, as `jit` is: the first call with a new argument
signature traces the function into a Tensor IR graph, applies the symbolic
transform (reverse mode for `grad`/`value_and_grad`/`vjp`, forward mode for
`jvp`/`jacobian`), compiles one multi-output CPU program, and caches it.
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
into the trace, as with `jax.jit`: pass values that change as arguments.

Composition works by staging (slice S3, design 3.4 and D8). A transform
whose function is itself a transform (`grad(grad(f))`, `jvp(grad(f), ...)`,
`jit(value_and_grad(f))`, `hessian`) transforms the inner transform's graph
directly. A transformed function called on traced values, inside a function
that another transform is tracing, stages itself for the tracers' shapes and
dtypes in a graph of its own, cached per signature like a compiled program,
and inlines that graph into the enclosing trace (`TensorIr::inline`): its
inputs bind to the tracers, so a derivative can be used inside a loss that
is differentiated again. `jit` of a plain Python function traces through it
instead, which is exact. `jacobian` and `hessian` evaluate column by column
and cannot be staged or inlined until `vmap` exists.
"""

import inspect
import weakref

from . import _quabla
from ._array import asarray, eye, zeros
from ._errors import RetraceLimitError, TracerError, UnsupportedOperationError
from ._quabla import Tensor, TensorTraceGraph, TraceTensor, bool_, stack
from .tree import _LEAF, _flatten, _leaf_count, _leaf_paths, _unflatten

__all__ = ["grad", "hessian", "jacobian", "jit", "jvp", "value_and_grad", "vjp"]

_STATIC_TYPES = (bool, int, float)
# Leaves that become graph inputs when a function is staged: arrays, and the
# tracers of an enclosing trace when a transformed function is inlined.
_ARRAY_LEAVES = (Tensor, TraceTensor)
_DEFAULT_MAX_TRACES = 8
# Graph input names starting with `__quabla_` are reserved (design 3.11).
_TANGENT_PREFIX = "__quabla_tangent/"
_COTANGENT_PREFIX = "__quabla_cotangent/"

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
            raise TypeError(f"{what} must be an int or a tuple of ints, got {argnums!r}")
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
        if kind is Tensor:
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


_EAGER_IN_TRACE = (
    "cannot pass an eager array to a transformed function called on traced values: traced "
    "functions do not capture arrays as constants yet; pass the array as an argument of the "
    "outer transformed function (it becomes a graph input), or use a Python scalar"
)


def _traced_signature(args, static_argnums, converted_argnums):
    """Flattens a call whose arguments hold tracers of an enclosing trace.

    Returns `(in_node, leaves, key, bindings)`. Each tracer leaf stands for
    an input of the staged callee with the tracer's shape and dtype, and a
    Python scalar in a differentiated position becomes a `float64` array
    leaf as in `_signature`; `bindings` holds, per array leaf in order, the
    value its callee input is bound to when the callee is inlined: the
    tracer, or the Python scalar as a constant. Eager arrays cannot be bound,
    since constants are not captured.
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
                        "a transformed function was called on a batched tracer inside vmap; "
                        "composing transforms with vmap is not implemented yet",
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
                _as_array_leaf(leaf)
                raise TracerError(_EAGER_IN_TRACE)
            key.append((tuple(leaf.shape), leaf.dtype))
    in_node = (tuple, tuple(nodes))
    return in_node, leaves, (in_node, tuple(key)), bindings


def _describe_signature(key):
    in_node, parts = key
    rendered = []
    for part in parts:
        if part[0] is _Static:
            rendered.append(_Text(f"static {part[1]!r}"))
        elif isinstance(part[0], tuple):
            shape = ",".join(str(extent) for extent in part[0])
            rendered.append(_Text(f"{part[1].name}[{shape}]"))
        else:
            value = float.fromhex(part[1]) if part[0] is float else part[1]
            rendered.append(_Text(repr(value)))
    return repr(_unflatten(in_node, iter(rendered)))


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
        if parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
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
    the flat result leaves, each a tracer of `graph` or a constant."""

    __slots__ = ("graph", "input_names", "outputs", "out_node")

    def __init__(self, graph, input_names, outputs, out_node):
        self.graph = graph
        self.input_names = input_names
        self.outputs = outputs
        self.out_node = out_node


def _trace(fun, in_node, leaves, names):
    graph = TensorTraceGraph()
    values = []
    input_names = []
    for leaf, name in zip(leaves, names):
        kind = type(leaf)
        if kind is Tensor or kind is TraceTensor:
            values.append(graph.input(name, leaf.shape, leaf.dtype))
            input_names.append(name)
        elif kind is _Static:
            values.append(leaf.value)
        else:
            values.append(leaf)
    result = fun(*_unflatten(in_node, iter(values)))
    outputs = []
    out_node = _flatten(result, outputs)
    return _Staged(graph, input_names, outputs, out_node)


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
    spliced = iter(staged.graph._inline(staged.input_names, bindings, traced) if traced else ())
    flat = [next(spliced) if _is_traced(output) else output for output in staged.outputs]
    return _unflatten(staged.out_node, iter(flat))


def _traced_like(value, shape, dtype, what):
    """`value` as an inline binding for a callee input of `shape` and `dtype`:
    a tracer of the enclosing trace, or a Python number for a scalar input,
    which the bridge binds as a constant of `dtype` (as `_as_like` adopts the
    primal dtype for a number)."""
    if type(value) is TraceTensor:
        actual_shape, actual_dtype = value.shape, value.dtype
    elif type(value) in _STATIC_TYPES:
        actual_shape, actual_dtype = [], dtype
    else:
        _as_array_leaf(value)
        raise TracerError(_EAGER_IN_TRACE)
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

    def __init__(self, graph, input_names, outputs, out_node):
        traced = [output for output in outputs if _is_traced(output)]
        self.executable = graph._compile_cpu(traced, input_names) if traced else None
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
        flat = [results[slot] if type(slot) is int else slot.value for slot in self.template]
        if self.out_node is _LEAF:
            return flat[0]
        return _unflatten(self.out_node, iter(flat))


class _Const:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


# -- trace caches ----------------------------------------------------------------


class _TraceCache:
    """Compiled programs of one transformed function keyed by signature."""

    __slots__ = ("entries", "max_traces")

    def __init__(self, max_traces):
        self.entries = {}
        self.max_traces = max_traces

    def lookup(self, key, build, transform):
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_traces:
                cached = "\n".join(f"  {_describe_signature(k)}" for k in self.entries)
                raise RetraceLimitError(
                    f"{transform!r} exceeded max_traces={self.max_traces}; cached signatures:\n"
                    f"{cached}\nnew signature:\n  {_describe_signature(key)}\n"
                    "Python scalar arguments are static and part of the signature: pass "
                    "quabla.array(value, dtype=...) to vary one without retracing, or raise "
                    "max_traces with quabla.jit"
                )
            entry = build()
            self.entries[key] = entry
        return entry


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

    _stageable = True
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

    def _call_traced(self, args, static_argnums, converted_argnums):
        """A call on tracers of an enclosing trace: stages this transform for
        their shapes and dtypes and inlines it into that trace."""
        in_node, leaves, key, bindings = _traced_signature(
            args, static_argnums, converted_argnums
        )
        staged = self._inline_cache().lookup(
            key, lambda: self._stage(in_node, leaves, self._names(in_node)), self
        )
        return _splice(staged, bindings)

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
        if not (type(out_node) is tuple and out_node[0] is tuple and len(out_node[1]) == 2):
            raise TypeError(f"{what} with has_aux=True requires fun to return a (value, aux) pair")
        value_node, aux_node = out_node[1]
    else:
        value_node, aux_node = out_node, None
    if value_node is not _LEAF:
        raise TypeError(f"{what} requires fun to return a single scalar array as its value")
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
            grad_leaves.append(None if leaf.dtype == bool_ else gradients[names[position]])
    if len(argnums) == 1 and not isinstance(argnums, _Tuple):
        grads_node = in_node[1][argnums[0]]
    else:
        grads_node = (tuple, tuple(in_node[1][argnum] for argnum in argnums))
    return graph, value, value_node, aux, aux_node, grad_leaves, grads_node


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
        graph, value, value_node, aux, aux_node, grad_leaves, grads_node = _stage_value_and_grad(
            staged, in_node, leaves, names, argnums, self._has_aux, self._kind
        )
        value_part = [value] + aux
        value_part_node = (tuple, (value_node, aux_node)) if self._has_aux else value_node
        if self._kind == "value_and_grad":
            outputs = value_part + grad_leaves
            out_node = (tuple, (value_part_node, grads_node))
        elif self._has_aux:
            outputs = grad_leaves + aux
            out_node = (tuple, (grads_node, aux_node))
        else:
            outputs = grad_leaves
            out_node = grads_node
        return _Staged(graph, staged.input_names, outputs, out_node)

    def __call__(self, *args):
        converted = self._converted_positions(len(args))
        in_node, leaves, arrays, key, traced = _signature(args, (), converted)
        if traced:
            return self._call_traced(args, (), converted)
        program = self._trace_cache().lookup(key, lambda: self._compile(in_node, leaves), self)
        return program.run(arrays)

    def _compile(self, in_node, leaves):
        staged = self._stage(in_node, leaves, self._names(in_node))
        return _Program(staged.graph, staged.input_names, staged.outputs, staged.out_node)


class _Jit(_Transform):
    def __init__(self, fun, device, static_argnums, max_traces):
        if device not in (None, "cpu"):
            _check_device(device)
        if isinstance(static_argnums, int):
            static_argnums = (static_argnums,)
        static_argnums = tuple(static_argnums)
        if not isinstance(max_traces, int) or isinstance(max_traces, bool) or max_traces < 1:
            raise ValueError(f"max_traces must be a positive int, got {max_traces!r}")
        self._max_traces = max_traces
        super().__init__(fun, ("jit", "cpu", static_argnums, max_traces))
        self._static_argnums = static_argnums
        self._static_by_count = {}

    def _stage(self, in_node, leaves, names):
        return _stage(self._fun, in_node, leaves, names)

    def _statics(self, count):
        statics = self._static_by_count.get(count)
        if statics is None:
            statics = frozenset(_normalize_argnums(self._static_argnums, count, "static_argnums"))
            self._static_by_count[count] = statics
        return statics

    def __call__(self, *args):
        if not getattr(self._fun, "_stageable", True):
            # jacobian and hessian evaluate column by column on the CPU; they
            # are already compiled, so jit only forwards the call.
            return self._fun(*args)
        count = len(args)
        statics = self._statics(count) if self._static_argnums else ()
        converted = self._converted_positions(count)
        in_node, leaves, arrays, key, traced = _signature(args, statics, converted)
        if traced:
            if isinstance(self._fun, _Transform):
                return self._call_traced(args, statics, converted)
            # Inside another trace, tracing through a plain function inlines it.
            return self._fun(*args)
        program = self._trace_cache().lookup(key, lambda: self._compile(in_node, leaves), self)
        return program.run(arrays)

    def _compile(self, in_node, leaves):
        staged = self._stage(in_node, leaves, self._names(in_node))
        return _Program(staged.graph, staged.input_names, staged.outputs, staged.out_node)


def _check_device(device):
    if isinstance(device, str) and (
        device in ("cuda", "mlx") or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise UnsupportedOperationError(
            f"quabla.jit(device={device!r}) is not implemented yet: the v0.2 transforms "
            "compile for 'cpu' only so far; use the tensor_*_cuda_fn / tensor_*_mlx_fn "
            "helpers for device execution meanwhile",
            op="jit",
            device=device,
        )
    raise ValueError(f"device must be None, 'cpu', 'cuda', 'cuda:N', or 'mlx', got {device!r}")


# -- forward mode ------------------------------------------------------------------


def _as_like(value, shape, dtype, what):
    """`value` as a Tensor of `shape` and `dtype`; a Python number is a weak
    scalar that adopts `dtype`, every other value must match it exactly."""
    if type(value) in _STATIC_TYPES:
        value = asarray(value, dtype)
    elif _is_traced(value):
        raise TracerError(
            f"{what} is traced but its primals are eager arrays: traced functions do not "
            "capture arrays as constants yet; pass the primals as arguments of the outer "
            "transformed function"
        )
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
        if traced or any(_is_traced(tangent) for tangent in tangent_leaves):
            return self._jvp_traced(primals, converted, tangent_node, tangent_leaves)
        entry = self._trace_cache().lookup(key, lambda: self._compile(in_node, leaves), self)
        if tangent_node != in_node:
            raise ValueError("jvp tangents must have the pytree structure of the primals")
        inputs = list(arrays)
        for position in entry.tangent_positions:
            primal = leaves[position]
            inputs.append(
                _as_like(tangent_leaves[position], primal.shape, primal.dtype, "a jvp tangent")
            )
        return entry.program.run(inputs)

    def _jvp_traced(self, primals, converted, tangent_node, tangent_leaves):
        """`jvp` on tracers of an enclosing trace: the staged JVP graph is
        inlined with its tangent inputs bound to the traced tangents."""
        in_node, leaves, key, bindings = _traced_signature(primals, (), converted)
        staged, tangent_positions = self._inline_cache().lookup(
            key, lambda: self._stage_jvp(in_node, leaves), self
        )
        if tangent_node != in_node:
            raise ValueError("jvp tangents must have the pytree structure of the primals")
        for position in tangent_positions:
            primal = leaves[position]
            bindings.append(
                _traced_like(tangent_leaves[position], primal.shape, primal.dtype, "a jvp tangent")
            )
        return _splice(staged, bindings)

    def _compile(self, in_node, leaves):
        staged, tangent_positions = self._stage_jvp(in_node, leaves)
        program = _Program(staged.graph, staged.input_names, staged.outputs, staged.out_node)
        return _JvpProgram(program, tangent_positions)

    def _stage_jvp(self, in_node, leaves):
        """The forward-mode graph, taking the primal inputs followed by one
        tangent input per leaf in the returned `tangent_positions`."""
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
            names[position]: _TANGENT_PREFIX + names[position] for position in tangent_positions
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
    stored)."""

    __slots__ = ("_program", "_arrays")

    def __init__(self, program, arrays):
        self._program = program
        self._arrays = arrays

    def __call__(self, cotangent):
        program = self._program
        cotangent_leaves = []
        if _flatten(cotangent, cotangent_leaves) != program.out_node:
            raise ValueError("the cotangent must have the pytree structure of the primal output")
        inputs = list(self._arrays)
        for position, shape, dtype in program.cotangent_specs:
            inputs.append(_as_like(cotangent_leaves[position], shape, dtype, "a vjp cotangent"))
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
            raise ValueError("the cotangent must have the pytree structure of the primal output")
        bindings = list(self._bindings)
        for position, shape, dtype in self._cotangent_specs:
            bindings.append(
                _traced_like(cotangent_leaves[position], shape, dtype, "a vjp cotangent")
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
        if traced:
            in_node, leaves, key, bindings = _traced_signature(primals, (), converted)
            forward, reverse, cotangent_specs, out_node = self._inline_cache().lookup(
                key, lambda: self._stage_vjp(in_node, leaves), self
            )
            result = _splice(forward, bindings)
            pullback = _TracedVjpFunction(reverse, cotangent_specs, out_node, bindings)
        else:
            entry = self._trace_cache().lookup(key, lambda: self._compile(in_node, leaves), self)
            result = entry.forward.run(arrays)
            pullback = VjpFunction(entry, arrays)
        if self._has_aux:
            out, aux = result
            return out, pullback, aux
        return result, pullback

    def _compile(self, in_node, leaves):
        forward, reverse, cotangent_specs, out_node = self._stage_vjp(in_node, leaves)
        return _VjpProgram(
            _Program(forward.graph, forward.input_names, forward.outputs, forward.out_node),
            _Program(reverse.graph, reverse.input_names, reverse.outputs, reverse.out_node),
            cotangent_specs,
            out_node,
            len(in_node[1]),
        )

    def _stage_vjp(self, in_node, leaves):
        """The forward graph, the reverse graph (the primal inputs followed by
        one cotangent input per entry of `cotangent_specs`), the cotangent
        specs `(output leaf position, shape, dtype)`, and the structure of
        the differentiated output."""
        names = self._names(in_node)
        staged = _stage(self._fun, in_node, leaves, names)
        out_node, outputs = staged.out_node, staged.outputs
        if self._has_aux:
            if not (type(out_node) is tuple and out_node[0] is tuple and len(out_node[1]) == 2):
                raise TypeError("vjp with has_aux=True requires fun to return an (out, aux) pair")
            out_node = out_node[1][0]
            outputs = outputs[: _leaf_count(out_node)]
        seeded = []
        cotangent_specs = []
        for position, output in enumerate(outputs):
            if _is_traced(output) and output.dtype != bool_:
                seeded.append(output)
                cotangent_specs.append((position, output.shape, output.dtype))
        if not seeded:
            raise ValueError("vjp requires a floating-point output computed from the primals")
        cotangent_names = [f"{_COTANGENT_PREFIX}{index}" for index in range(len(seeded))]
        graph, _, gradients = staged.graph._symbolic_vjp(seeded, cotangent_names, [])
        grad_outputs = [
            gradients[name] if type(leaf) in _ARRAY_LEAVES and leaf.dtype != bool_ else None
            for leaf, name in zip(leaves, names)
        ]
        reverse = _Staged(graph, staged.input_names + cotangent_names, grad_outputs, in_node)
        return staged, reverse, cotangent_specs, out_node


# -- dense Jacobians ------------------------------------------------------------------


class _JacobianProgram:
    __slots__ = ("executable", "selected", "out_node", "outputs", "tangent_index")

    def __init__(self, executable, selected, out_node, outputs, tangent_index):
        self.executable = executable
        self.selected = selected
        self.out_node = out_node
        self.outputs = outputs
        self.tangent_index = tangent_index


class _Jacobian(_Transform):
    """Dense Jacobian by one forward-mode evaluation per input element, as
    `TensorJacobianFunction` does (design 3.3: dense and CPU-only in v0.2)."""

    _stageable = False

    def __init__(self, fun, argnums, kind="jacobian"):
        argnums = _argnums_config(argnums)
        super().__init__(fun, (kind, argnums))
        self._argnums = argnums

    def _own_differentiated(self, count):
        return self._positions(count)

    def _positions(self, count):
        return _normalize_argnums(self._argnums, count, "argnums")

    def _stage(self, in_node, leaves, names):
        raise self._unstaged_error()

    def _unstaged_error(self):
        kind = self._config[0]
        return UnsupportedOperationError(
            f"{self!r} cannot be transformed further or called on traced values: {kind} "
            "evaluates one forward-mode column per input element instead of staging a single "
            "program, and staging it needs vmap, which is not implemented yet. Inside a traced "
            "function, use grad(grad(f)) for the second derivative of a scalar function, or "
            "jvp(grad(f), (x,), (v,)) for a Hessian-vector product",
            op=kind,
        )

    def __call__(self, *args):
        count = len(args)
        in_node, leaves, arrays, key, traced = _signature(
            args, (), self._converted_positions(count)
        )
        if traced:
            raise self._unstaged_error()
        entry = self._trace_cache().lookup(key, lambda: self._compile(in_node, leaves), self)
        argnums = self._positions(count)
        ranges = _argument_ranges(in_node)
        # blocks[output][position]: the Jacobian block of one output leaf with
        # respect to one selected input leaf.
        blocks = [{} for _ in entry.outputs]
        zero_tangents = [
            zeros(leaves[position].shape, leaves[position].dtype) for position in entry.selected
        ]
        for slot, position in enumerate(entry.selected):
            leaf = leaves[position]
            size = 1
            for extent in leaf.shape:
                size *= extent
            basis = eye(size, dtype=leaf.dtype)
            columns = [[] for _ in entry.outputs]
            for element in range(size):
                tangents = list(zero_tangents)
                tangents[slot] = basis[element].reshape(leaf.shape)
                results = entry.executable(arrays + tangents) if entry.executable else ()
                for index, output in enumerate(entry.outputs):
                    if _is_traced_spec(output):
                        columns[index].append(results[entry.tangent_index[index]])
            for index, output in enumerate(entry.outputs):
                blocks[index][position] = _jacobian_block(output, columns[index], leaf)
        rendered = []
        for index in range(len(entry.outputs)):
            parts = []
            for argnum in argnums:
                parts.append(
                    _unflatten(
                        in_node[1][argnum],
                        iter([blocks[index].get(position) for position in ranges[argnum]]),
                    )
                )
            if len(argnums) == 1 and not isinstance(self._argnums, _Tuple):
                rendered.append(parts[0])
            else:
                rendered.append(tuple(parts))
        if entry.out_node is _LEAF:
            return rendered[0]
        return _unflatten(entry.out_node, iter(rendered))

    def _compile(self, in_node, leaves):
        names = self._names(in_node)
        staged = _stage(self._fun, in_node, leaves, names)
        argnums = self._positions(len(in_node[1]))
        ranges = _argument_ranges(in_node)
        selected = []
        for argnum in argnums:
            for position in ranges[argnum]:
                leaf = leaves[position]
                if type(leaf) is not Tensor:
                    raise TypeError(
                        f"{self._config[0]} cannot differentiate with respect to static "
                        f"argument {argnum}"
                    )
                if leaf.dtype != bool_:
                    selected.append(position)
        if not selected:
            raise ValueError(f"{self._config[0]} requires a floating-point argument in argnums")
        tangent_names = {names[p]: _TANGENT_PREFIX + names[p] for p in selected}
        traced = [output for output in staged.outputs if _is_traced(output)]
        outputs = []
        tangent_index = {}
        executable = None
        if traced:
            graph, _, tangents = staged.graph._symbolic_jvp(traced, tangent_names)
            executable = graph._compile_cpu(
                tangents, staged.input_names + [tangent_names[names[p]] for p in selected]
            )
        traced_index = 0
        for index, output in enumerate(staged.outputs):
            if _is_traced(output):
                outputs.append(_TracedSpec(output.shape, output.dtype))
                tangent_index[index] = traced_index
                traced_index += 1
            else:
                outputs.append(output)
        return _JacobianProgram(executable, selected, staged.out_node, outputs, tangent_index)


class _TracedSpec:
    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def _is_traced_spec(output):
    return type(output) is _TracedSpec


def _jacobian_block(output, columns, leaf):
    """The block `[*output.shape, *leaf.shape]` from the per-element columns."""
    if not _is_traced_spec(output):
        shape = list(output.shape) if isinstance(output, Tensor) else []
        dtype = output.dtype if isinstance(output, Tensor) else leaf.dtype
        return zeros(shape + leaf.shape, dtype)
    stacked = stack(columns, len(output.shape))
    return stacked.reshape(output.shape + leaf.shape)


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
    argument signature (see the module notes); CPU only in this release."""
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
    cache. `device=None` means `"cpu"`, the only device of this release.
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

_LEGACY_GRAD_KEYWORDS = frozenset({"function", "input_specs", "values", "output_cotangent"})


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
    the arguments selected by `argnums`. Evaluated on the CPU with one
    forward-mode pass per input element."""
    return _Jacobian(fun, argnums)


def hessian(fun, argnums=0):
    """`fun` transformed to return the dense Hessian of its scalar output,
    `jacobian(grad(fun, argnums), argnums)`: blocks of shape
    `[*in.shape, *in.shape]`, evaluated forward-over-reverse on the CPU."""
    return _Jacobian(_ValueAndGrad(fun, argnums, False, "grad"), argnums, "hessian")
