"""Pytree utilities (docs/api_v0_2_design.md, section 3.11, decision D6;
extended with node registration in v0.3).

A pytree is a nested structure of containers:

- `dict`, traversed in sorted key order; keys must be hashable and mutually
  sortable (all strings, or all integers, for example);
- `list` and `tuple`;
- `None`, an empty container;
- every `NamedTuple` class, whose children are its fields and whose
  rebuilt value is an instance of the same class;
- every class registered with `register` or `register_dataclass`.

Every other object is a leaf. Other container subclasses (`OrderedDict`, a
`list` subclass) and unregistered dataclasses are rejected where a
transform expects array leaves, rather than silently becoming leaves.

The transforms flatten their arguments and results with the private helpers
below, whose structure descriptions are plain nested tuples so that they hash
and compare cheaply as part of a trace cache key; `TreeDef` wraps one for the
public API. A custom node records its class and its auxiliary data, so two
classes, or two values of a static (`meta`) field, never share a trace.
"""

import dataclasses

__all__ = [
    "TreeDef",
    "flatten",
    "flatten_with_path",
    "leaves",
    "map",
    "register",
    "register_dataclass",
    "structure",
    "unflatten",
]

# Structure node of a leaf. Container nodes are `None`, `(list, children)`,
# `(tuple, children)`, `(dict, keys, children)`, and `(cls, aux, children)`
# for a NamedTuple or registered class, with `children` a tuple of nodes, so
# no container node is equal to this string and `node[-1]` is always the
# children of a container node.
_LEAF = "*"


class _NodeType:
    """How a registered class flattens: `flatten(value)` returns
    `(children, aux)`, `unflatten(aux, children)` rebuilds the value, and
    `names` (or `None` for positions) labels the children in key paths."""

    __slots__ = ("flatten", "unflatten", "names")

    def __init__(self, flatten, unflatten, names=None):
        self.flatten = flatten
        self.unflatten = unflatten
        self.names = names


# Registered node types by exact class.
_REGISTRY = {}
_BUILTIN_NODES = (dict, list, tuple, type(None))


def _is_namedtuple_class(kind):
    return issubclass(kind, tuple) and hasattr(kind, "_fields")


def register(cls, flatten_fn, unflatten_fn):
    """Registers `cls` as a pytree node type.

    `flatten_fn(value)` returns `(children, aux_data)`: an iterable of child
    pytrees and auxiliary data that is hashable and comparable, since it
    becomes part of the structure and so of every trace cache key.
    `unflatten_fn(aux_data, children)` rebuilds a value from the same aux
    data and a tuple of children; the transforms call it with arrays or
    tracers as leaves, so it must not inspect them. Only exact instances of
    `cls` are nodes, as with `jax.tree_util.register_pytree_node`. Returns
    `cls`.
    """
    _check_registrable(cls)
    if not (callable(flatten_fn) and callable(unflatten_fn)):
        raise TypeError("register requires callable flatten_fn and unflatten_fn")
    _REGISTRY[cls] = _NodeType(flatten_fn, unflatten_fn)
    return cls


def register_dataclass(cls=None, data_fields=None, meta_fields=()):
    """Registers the dataclass `cls` as a pytree node type and returns it,
    so it also works as a bare decorator (`@register_dataclass`).

    `data_fields` are the children, in the given order; `None` means every
    field passed to `__init__` that is not in `meta_fields`, in declaration
    order. `meta_fields` are static: their values must be hashable, are part
    of the trace cache key, and are passed through unchanged. Together they
    must name every `__init__` field exactly once. Values are rebuilt by
    calling `cls(**fields)`, which supports frozen dataclasses and runs
    `__post_init__` again, as in JAX.
    """
    if cls is None:
        return lambda cls: register_dataclass(cls, data_fields, meta_fields)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"register_dataclass requires a dataclass type, got {cls!r}")
    _check_registrable(cls)
    init_fields = [field.name for field in dataclasses.fields(cls) if field.init]
    meta_fields = tuple(meta_fields)
    if data_fields is None:
        data_fields = tuple(name for name in init_fields if name not in meta_fields)
    data_fields = tuple(data_fields)
    named = data_fields + meta_fields
    if len(set(named)) != len(named) or set(named) != set(init_fields):
        raise ValueError(
            f"data_fields {data_fields!r} and meta_fields {meta_fields!r} of "
            f"{cls.__name__} must name each __init__ field {tuple(init_fields)!r} exactly once"
        )

    def flatten(value):
        children = tuple(getattr(value, name) for name in data_fields)
        return children, tuple(getattr(value, name) for name in meta_fields)

    def unflatten(aux, children):
        return cls(**dict(zip(data_fields, children)), **dict(zip(meta_fields, aux)))

    _REGISTRY[cls] = _NodeType(flatten, unflatten, data_fields)
    return cls


def _check_registrable(cls):
    if not isinstance(cls, type):
        raise TypeError(f"only classes can be registered as pytree nodes, got {cls!r}")
    if cls in _BUILTIN_NODES:
        raise ValueError(f"{cls.__name__} is a built-in pytree node and cannot be registered")
    if cls in _REGISTRY:
        raise ValueError(f"{cls.__qualname__} is already registered as a pytree node")


def _sorted_keys(tree):
    try:
        return tuple(sorted(tree))
    except TypeError:
        raise TypeError(
            "pytree dict keys must be mutually sortable (all strings or all numbers, "
            f"for example), got {list(tree)!r}"
        ) from None


def _flatten(tree, leaves):
    """Appends the leaves of `tree` to `leaves` and returns its structure node."""
    kind = type(tree)
    if kind is tuple or kind is list:
        return (kind, tuple([_flatten(child, leaves) for child in tree]))
    if kind is dict:
        keys = _sorted_keys(tree)
        return (dict, keys, tuple([_flatten(tree[key], leaves) for key in keys]))
    if tree is None:
        return None
    node_type = _REGISTRY.get(kind)
    if node_type is not None:
        children, aux = node_type.flatten(tree)
        try:
            hash(aux)
        except TypeError:
            raise TypeError(
                f"the pytree aux data of {kind.__qualname__} must be hashable, got "
                f"{aux!r}; it is part of the trace cache key"
            ) from None
        return (kind, aux, tuple([_flatten(child, leaves) for child in children]))
    if _is_namedtuple_class(kind):
        return (kind, None, tuple([_flatten(child, leaves) for child in tree]))
    leaves.append(tree)
    return _LEAF


def _unflatten(node, leaves):
    """Rebuilds the structure `node` from the iterator `leaves`."""
    if node is _LEAF:
        return next(leaves)
    if node is None:
        return None
    kind = node[0]
    if kind is dict:
        return {key: _unflatten(child, leaves) for key, child in zip(node[1], node[2])}
    children = [_unflatten(child, leaves) for child in node[-1]]
    if kind is list:
        return children
    if kind is tuple:
        return tuple(children)
    node_type = _REGISTRY.get(kind)
    if node_type is not None:
        return node_type.unflatten(node[1], tuple(children))
    return kind(*children)


def _leaf_count(node):
    if node is _LEAF:
        return 1
    if node is None:
        return 0
    return sum(_leaf_count(child) for child in node[-1])


def _child_keys(node):
    """The key of each child of the container `node`: dict keys, field
    names of NamedTuples and registered dataclasses, positions otherwise."""
    kind = node[0]
    if kind is dict:
        return node[1]
    if kind is not list and kind is not tuple:
        node_type = _REGISTRY.get(kind)
        names = node_type.names if node_type is not None else getattr(kind, "_fields", None)
        if names is not None:
            return names
    return range(len(node[-1]))


def _key_paths(node, prefix, paths):
    """Appends the key path tuple of every leaf of `node`, in leaf order."""
    if node is _LEAF:
        paths.append(prefix)
    elif node is not None:
        for key, child in zip(_child_keys(node), node[-1]):
            _key_paths(child, prefix + (key,), paths)
    return paths


def _leaf_paths(node, prefix):
    """The `/`-joined key path of every leaf of `node`, in leaf order."""
    return [
        "/".join([prefix] + [str(key) for key in path])
        for path in _key_paths(node, (), [])
    ]


def _describe(node, leaves=None):
    """A readable rendering of a structure node, with `*` for leaves, or the
    next string of the iterator `leaves` for each leaf when given."""
    if node is _LEAF:
        return _LEAF if leaves is None else next(leaves)
    if node is None:
        return "None"
    kind = node[0]
    if kind is dict:
        items = ", ".join(
            f"{key!r}: {_describe(child, leaves)}" for key, child in zip(node[1], node[2])
        )
        return "{" + items + "}"
    children = [_describe(child, leaves) for child in node[-1]]
    if kind is list:
        return "[" + ", ".join(children) + "]"
    if kind is tuple:
        return "(" + ", ".join(children) + ("," if len(children) == 1 else "") + ")"
    keys = _child_keys(node)
    if not isinstance(keys, range):
        children = [f"{key}={child}" for key, child in zip(keys, children)]
    if node[1] is not None and node[1] != ():
        children.append(f"aux={node[1]!r}")
    return f"{kind.__qualname__}(" + ", ".join(children) + ")"


class TreeDef:
    """The structure of a pytree with its leaves removed. Tree definitions
    compare and hash by structure, including dict keys, node classes, and
    the aux data of registered nodes."""

    __slots__ = ("_node", "num_leaves")

    def __init__(self, node):
        self._node = node
        self.num_leaves = _leaf_count(node)

    def __eq__(self, other):
        return isinstance(other, TreeDef) and self._node == other._node

    def __hash__(self):
        return hash(self._node)

    def __repr__(self):
        return f"TreeDef({_describe(self._node)})"

    def unflatten(self, leaves):
        """`unflatten(self, leaves)`."""
        return unflatten(self, leaves)


def flatten(tree):
    """The leaves of `tree` in traversal order and its `TreeDef`."""
    leaves = []
    node = _flatten(tree, leaves)
    return leaves, TreeDef(node)


def leaves(tree):
    """The leaves of `tree` in traversal order."""
    result = []
    _flatten(tree, result)
    return result


def structure(tree):
    """The `TreeDef` of `tree`."""
    return TreeDef(_flatten(tree, []))


def flatten_with_path(tree):
    """`(path, leaf)` pairs in traversal order and the `TreeDef` of `tree`.
    A path is a tuple of keys: dict keys, field names of NamedTuples and
    registered dataclasses, and integer positions of lists, tuples, and
    other registered nodes."""
    leaves = []
    node = _flatten(tree, leaves)
    return list(zip(_key_paths(node, (), []), leaves)), TreeDef(node)


def unflatten(treedef, leaves):
    """The pytree with structure `treedef` and the given leaves, in order."""
    leaves = list(leaves)
    if len(leaves) != treedef.num_leaves:
        raise ValueError(f"{treedef!r} has {treedef.num_leaves} leaves, got {len(leaves)} values")
    return _unflatten(treedef._node, iter(leaves))


def map(f, tree, *rest):
    """Applies `f` leafwise: `f(leaf, *corresponding leaves of rest)`. Every
    tree in `rest` must have the structure of `tree`."""
    leaves, treedef = flatten(tree)
    columns = [leaves]
    for other in rest:
        other_leaves, other_def = flatten(other)
        if other_def != treedef:
            raise ValueError(f"pytree structures differ: {treedef!r} and {other_def!r}")
        columns.append(other_leaves)
    return _unflatten(treedef._node, iter([f(*values) for values in zip(*columns)]))
