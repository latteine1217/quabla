"""Pytree utilities (docs/api_v0_2_design.md, section 3.11, decision D6).

A pytree is a nested structure of containers: `dict` with string keys,
traversed in sorted key order, `list`, `tuple`, and `None`, which is an empty
container. Every other object is a leaf. Only these exact types are
containers, so a NamedTuple, an `OrderedDict`, or a dataclass is a leaf;
registering further node types is out of scope for v0.2 (D6).

The transforms flatten their arguments and results with the private helpers
below, whose structure descriptions are plain nested tuples so that they hash
and compare cheaply as part of a trace cache key; `TreeDef` wraps one for the
public API.
"""

__all__ = ["TreeDef", "flatten", "map", "unflatten"]

# Structure node of a leaf. Container nodes are `None`, `(list, children)`,
# `(tuple, children)`, and `(dict, keys, children)`, with `children` a tuple of
# nodes, so no container node is equal to this string.
_LEAF = "*"


def _flatten(tree, leaves):
    """Appends the leaves of `tree` to `leaves` and returns its structure node."""
    kind = type(tree)
    if kind is tuple or kind is list:
        return (kind, tuple([_flatten(child, leaves) for child in tree]))
    if kind is dict:
        for key in tree:
            if type(key) is not str:
                raise TypeError(
                    f"pytree dict keys must be strings, got a key of type {type(key).__name__}"
                )
        keys = tuple(sorted(tree))
        return (dict, keys, tuple([_flatten(tree[key], leaves) for key in keys]))
    if tree is None:
        return None
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
    children = [_unflatten(child, leaves) for child in node[1]]
    return children if kind is list else tuple(children)


def _leaf_count(node):
    if node is _LEAF:
        return 1
    if node is None:
        return 0
    return sum(_leaf_count(child) for child in node[-1])


def _leaf_paths(node, prefix):
    """The `/`-joined key path of every leaf of `node`, in leaf order."""
    if node is _LEAF:
        return [prefix]
    if node is None:
        return []
    keys = node[1] if node[0] is dict else range(len(node[1]))
    paths = []
    for key, child in zip(keys, node[-1]):
        paths.extend(_leaf_paths(child, f"{prefix}/{key}"))
    return paths


def _describe(node):
    """A readable rendering of a structure node, with `*` for leaves."""
    if node is _LEAF:
        return _LEAF
    if node is None:
        return "None"
    if node[0] is dict:
        items = ", ".join(f"{key!r}: {_describe(child)}" for key, child in zip(node[1], node[2]))
        return "{" + items + "}"
    children = [_describe(child) for child in node[1]]
    if node[0] is list:
        return "[" + ", ".join(children) + "]"
    return "(" + ", ".join(children) + ("," if len(children) == 1 else "") + ")"


class TreeDef:
    """The structure of a pytree with its leaves removed. Tree definitions
    compare and hash by structure, including dict keys."""

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


def flatten(tree):
    """The leaves of `tree` in traversal order and its `TreeDef`."""
    leaves = []
    node = _flatten(tree, leaves)
    return leaves, TreeDef(node)


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
