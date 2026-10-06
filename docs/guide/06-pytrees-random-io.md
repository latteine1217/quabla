# 6. Pytrees, Random Numbers, and Serialization

## 6.1 Pytrees

Models have many parameters, and passing each one as a separate argument
does not scale. Quabla's transforms, optimizers, `Trainer`, and `save`/`load`
therefore operate on *pytrees*: nested containers whose leaves are arrays or
Python values. A transform flattens its arguments to a list of leaves,
traces with one input per array leaf, and rebuilds results with the same
structure.

### Built-in containers

| Container | Notes |
| --- | --- |
| `dict` | Traversed in sorted key order; keys must be mutually sortable (all strings, or all ints). |
| `list`, `tuple` | Traversed in order. |
| `None` | An empty container (no leaves). |
| Every `NamedTuple` class | Automatic, as in JAX; results rebuild the same class. |
| Classes registered with `qb.tree.register_dataclass` or `qb.tree.register` | See below. |

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: qb.Tensor
    y: qb.Tensor

qb.grad(lambda p: p.x * p.y)(Point(qb.array(1.0), qb.array(2.0)))
```

```text
Point(x=Tensor(2., dtype=float64), y=Tensor(1., dtype=float64))
```

Other container subclasses (`OrderedDict`, a `list` subclass) and
unregistered dataclasses are rejected as transform arguments.

### Registering dataclasses

A dataclass becomes a container once registered. *Data fields* are children
(traced and differentiated); *meta fields* are static, hashable values that
are part of the trace signature:

```python
import dataclasses

@qb.tree.register_dataclass
@dataclasses.dataclass(frozen=True)
class Layer:
    w: qb.Tensor
    b: qb.Tensor

@dataclasses.dataclass(frozen=True)
class Mlp:
    layers: list
    activation: str          # static: selects code, not a parameter

qb.tree.register_dataclass(Mlp, data_fields=("layers",), meta_fields=("activation",))

def forward(model, x):
    act = {"tanh": qb.tanh, "relu": qb.relu}[model.activation]
    for layer in model.layers[:-1]:
        x = act(x @ layer.w + layer.b)
    last = model.layers[-1]
    return x @ last.w + last.b

k1, k2 = qb.random.split(qb.random.key(0))
model = Mlp(
    [Layer(qb.random.glorot_normal(k1, (2, 8)), qb.zeros((8,))),
     Layer(qb.random.glorot_normal(k2, (8, 1)), qb.zeros((1,)))],
    "tanh",
)
grads = qb.grad(lambda m, x: qb.sum(forward(m, x) ** 2))(model, qb.ones((4, 2)))
print(type(grads).__name__, type(grads.layers[0]).__name__, grads.activation)
```

```text
Mlp Layer tanh
```

- `register_dataclass(cls, data_fields=None, meta_fields=())`: by default
  every `__init__` field not in `meta_fields` is a data field, in
  declaration order. Together they must name every `__init__` field once.
- Values are rebuilt with `cls(**fields)`, so frozen dataclasses work and
  `__post_init__` runs again on reconstruction.
- A class can be registered once; only exact instances (not subclasses) are
  containers.
- Because meta fields are part of the signature, a new meta value retraces.

The general form takes two functions:

```python
qb.tree.register(Interval, flatten_fn, unflatten_fn)
# flatten_fn(value) -> (children, aux_data)
# unflatten_fn(aux_data, children) -> value
```

### The `qb.tree` module

| Function | Result |
| --- | --- |
| `flatten(tree)` | `(leaves, treedef)` |
| `unflatten(treedef, leaves)`, `treedef.unflatten(leaves)` | The rebuilt tree |
| `leaves(tree)` | The leaves only |
| `structure(tree)` | The `TreeDef`; compares and hashes by structure |
| `map(f, tree, *rest)` | Apply `f` leafwise over trees of one structure |
| `flatten_with_path(tree)` | `([(path, leaf), ...], treedef)`; a path is a tuple of dict keys, field names, and integer positions |

```python
leaves, treedef = qb.tree.flatten({"b": 1.0, "a": [qb.zeros((2,)), None]})
print(len(leaves), treedef)
print([path for path, _ in qb.tree.flatten_with_path(model)[0]])
print(qb.tree.map(lambda a, b: a + b, {"a": qb.array(1.0)}, {"a": qb.array(2.0)}))
```

```text
2 TreeDef({'a': [*, None], 'b': *})
[('layers', 0, 'w'), ('layers', 0, 'b'), ('layers', 1, 'w'), ('layers', 1, 'b')]
{'a': Tensor(3., dtype=float64)}
```

`vmap`'s `in_axes` and `out_axes` may use the same containers as pytree
prefixes, for example `in_axes=(Point(0, None),)`.

## 6.2 Random Numbers

`qb.random` samples with explicit keys, in the style of `jax.random`. There is
no global random state: the same key always gives the same numbers, on every
platform.

```python
key = qb.random.key(42)
k1, k2, k3 = qb.random.split(key, 3)

qb.random.uniform(k1, (3,))                       # [0, 1), float64
qb.random.normal(k2, (2,), dtype=qb.float32)
qb.random.bernoulli(k3, 0.5, (4,))                # bool_
qb.random.normal(qb.random.fold_in(key, 7), ())   # derive a key from an integer
```

| Function | Result |
| --- | --- |
| `key(seed)` | A `Key` from an integer seed (modulo 2**64) |
| `split(key, num=2)` | A tuple of `num` new keys |
| `fold_in(key, data)` | A new key from a key and an integer (for example the step number) |
| `uniform(key, shape=(), dtype=float64, minval=0.0, maxval=1.0)` | Samples on `[minval, maxval)` |
| `normal(key, shape=(), dtype=float64)` | Standard normal samples |
| `bernoulli(key, p=0.5, shape=())` | `bool_` samples |
| `glorot_normal(key, shape, dtype=float64)` | Normal, std `sqrt(2 / (fan_in + fan_out))` |
| `glorot_uniform(key, shape, dtype=float64)` | Uniform on `[-l, l)`, `l = sqrt(6 / (fan_in + fan_out))` |
| `he_normal(key, shape, dtype=float64)` | Normal, std `sqrt(2 / fan_in)` |

Usage rules:

- **Use each key once.** Split a key into children before sampling;
  sampling with a key after splitting it reuses its children's stream.
- **Sampling is eager and on the host.** A key cannot be a traced argument of
  a transform. Inside a traced function, a sample drawn from a closure key is
  a constant of the trace, so every call of the compiled program sees the
  same numbers. To resample collocation points or dropout masks every step,
  draw them outside and pass them in as arguments.
- Initializers compute fans as JAX does (`fan_in = shape[-2] * r`,
  `fan_out = shape[-1] * r`, `r` the product of leading dimensions) and need
  rank 2 or more. `glorot_normal` and `he_normal` are untruncated.
- The generator is SplitMix64, a statistical (not cryptographic) generator.
  Normal samples use Box-Muller.

A typical training loop derives one key per step:

```python
base = qb.random.key(0)
for step in range(num_steps):
    k = qb.random.fold_in(base, step)
    x = qb.random.uniform(k, (n, 1), minval=0.0, maxval=1.0)   # fresh points
    params, state = update(params, state, x)
```

## 6.3 Saving and Loading

```python
qb.save("checkpoint.qb", {"params": params, "opt": opt_state, "step": 1200})
restored = qb.load("checkpoint.qb")
```

`save(file, tree)` takes a path or a binary file object.

- Leaves may be Quabla arrays, `None`, and Python `bool`, `int`, `float`, and
  `str`. Containers may be `dict` (string or integer keys), `list`, and
  `tuple`.
- Arrays load as `Tensor`s with the same shape and dtype, bit for bit.
  Scalars keep their type and exact value, dicts their key order, tuples
  stay tuples. An optimizer state such as Adam's `{"step", "m", "v"}`
  therefore resumes unchanged.
- NamedTuples and registered classes are **not** serialized (rebuilding them
  would mean importing classes named by the file). Save their fields, for
  example with `dataclasses.asdict` or `qb.tree.flatten_with_path`, and
  rebuild the objects after loading.
- The format needs neither NumPy nor pickle, so `load` never executes code
  from the file. A malformed or truncated file raises `ValueError`.

The file is an 8-byte magic string, an 8-byte header length, a UTF-8 JSON
header that describes the tree, and the raw little-endian array data. The
byte layout is specified in the
[API Reference](../api.md#saving-and-loading), so other tools can read it.
