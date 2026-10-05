"""Pure pytree optimizers and retained device training plans."""

# Imported under private aliases so the public module namespace is `__all__`.
import math as _math
import operator as _operator

from . import _quabla
from . import tree as _tree
from ._array import asarray as _asarray
from ._array import zeros as _zeros
from ._errors import UnsupportedOperationError as _UnsupportedOperationError
from ._quabla import Tensor as _Tensor
from ._quabla import float64 as _float64
from ._transforms import _leaf_names, _trace
from ._transforms import jit as _jit
from ._transforms import value_and_grad as _value_and_grad

__all__ = ["Adam", "SGD", "Trainer"]


def _positive(value, name):
    value = float(value)
    if not _math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _parameters(params):
    leaves, definition = _tree.flatten(params)
    if not leaves:
        raise ValueError("optimizer requires at least one parameter")
    for leaf in leaves:
        if not isinstance(leaf, _Tensor) or leaf.dtype.name not in (
            "float32",
            "float64",
        ):
            raise TypeError("optimizer parameters must be floating-point Tensors")
    return leaves, definition


def _matching(params, other, name):
    leaves, definition = _parameters(params)
    return leaves, _matching_values(leaves, definition, other, name), definition


def _matching_values(leaves, definition, other, name):
    values, other_definition = _tree.flatten(other)
    if definition != other_definition:
        raise ValueError(f"{name} pytree structure must match parameters")
    for parameter, value in zip(leaves, values):
        if not isinstance(value, _Tensor):
            raise TypeError(f"{name} leaves must be Tensors")
        if parameter.shape != value.shape:
            raise ValueError(f"{name} shape must match parameter shape")
        if value.dtype.name not in ("float32", "float64"):
            raise TypeError(f"{name} leaves must be floating-point Tensors")
    return values


class Adam:
    """Adam configuration with pure updates and the v0.1 stateful ``step``.

    ``beta1``, ``beta2``, and ``epsilon`` remain accepted constructor aliases.
    Pure state stores float64 moments, independently of stateful legacy steps.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        *,
        beta1=None,
        beta2=None,
        epsilon=None,
    ):
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.b1 = float(b1 if beta1 is None else beta1)
        self.b2 = float(b2 if beta2 is None else beta2)
        self.eps = _positive(eps if epsilon is None else epsilon, "epsilon")
        if not (0 <= self.b1 < 1 and 0 <= self.b2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        self._legacy = _quabla.Adam(self.learning_rate, self.b1, self.b2, self.eps)

    def init(self, params):
        """Create independent zero moments for ``params``."""
        leaves, definition = _parameters(params)
        return {
            "step": 0,
            "m": _tree.unflatten(
                definition, [_zeros(p.shape, dtype=_float64) for p in leaves]
            ),
            "v": _tree.unflatten(
                definition, [_zeros(p.shape, dtype=_float64) for p in leaves]
            ),
        }

    def update(self, params, grads, state):
        """Return updated parameters and state without mutating either input."""
        parameters, gradients, definition = _matching(params, grads, "gradient")
        if type(state) is not dict or set(state) != {"step", "m", "v"}:
            raise ValueError("Adam state must contain step, m, and v")
        if type(state["step"]) is not int or state["step"] < 0:
            raise ValueError("Adam state step must be a nonnegative integer")
        first = _matching_values(parameters, definition, state["m"], "first moment")
        second = _matching_values(parameters, definition, state["v"], "second moment")
        step = state["step"] + 1
        moments, variances, updated = [], [], []
        hyperparameters = (
            self.learning_rate,
            self.b1,
            self.b2,
            self.eps,
            1 - self.b1**step,
            1 - self.b2**step,
        )
        for parameter, gradient, m, v in zip(parameters, gradients, first, second):
            parameter, m, v = parameter._adam_update(gradient, m, v, hyperparameters)
            updated.append(parameter)
            moments.append(m)
            variances.append(v)
        return _tree.unflatten(definition, updated), {
            "step": step,
            "m": _tree.unflatten(definition, moments),
            "v": _tree.unflatten(definition, variances),
        }

    def step(self, parameters, gradients):
        """Apply a stateful v0.1 dictionary update with native compatibility."""
        return self._legacy.step(parameters, gradients)


class SGD:
    """Plain stochastic gradient descent with pure pytree updates."""

    def __init__(self, learning_rate=1e-2):
        self.learning_rate = _positive(learning_rate, "learning_rate")

    def init(self, params):
        _parameters(params)
        return None

    def update(self, params, grads, state):
        parameters, gradients, definition = _matching(params, grads, "gradient")
        if state is not None:
            raise ValueError("SGD state must be None")
        updated = [
            p._sgd_update(g, self.learning_rate) for p, g in zip(parameters, gradients)
        ]
        return _tree.unflatten(definition, updated), None


class Trainer:
    """Trace ``loss(params, *data)`` and keep optimizer state between steps.

    ``batch_argnums`` indexes the data arguments (excluding parameters).
    ``step(*batch)`` replaces those arguments in the declared order. An empty
    call reuses the latest batch; ``loss()`` evaluates the current parameters.
    Device plans keep parameter and moment buffers on the device until readback.
    """

    def __init__(self, loss, params, optimizer, *data, device="cpu", batch_argnums=()):
        from ._devices import parse_device, require_device

        if not callable(loss):
            raise TypeError("loss must be callable")
        if not isinstance(optimizer, (Adam, SGD)):
            raise TypeError("optimizer must be Adam or SGD")
        target, ordinal = parse_device(device)
        parameters, self._param_def = _parameters(params)
        if isinstance(batch_argnums, int):
            batch_argnums = (batch_argnums,)
        self._batch_argnums = tuple(_operator.index(i) for i in batch_argnums)
        if len(set(self._batch_argnums)) != len(self._batch_argnums):
            raise ValueError("batch_argnums must be distinct")
        if any(i < 0 or i >= len(data) for i in self._batch_argnums):
            raise ValueError("batch_argnums indexes data arguments, excluding params")
        self._data = tuple(_tree.map(_asarray, item) for item in data)
        self._batch_specs = []
        for i in self._batch_argnums:
            leaves, definition = _tree.flatten(self._data[i])
            self._batch_specs.append(
                ([(tuple(leaf.shape), leaf.dtype) for leaf in leaves], definition)
            )
        self._params = params
        self._optimizer = optimizer
        self._executor = None
        if target == "cpu":
            self._state = optimizer.init(params)
            self._value_and_grad = _jit(_value_and_grad(loss), device="cpu")
            self._evaluate_loss = _jit(loss, device="cpu")
            return
        if not isinstance(optimizer, Adam):
            raise _UnsupportedOperationError(
                "device Trainer supports Adam only", op="Trainer", device=device
            )
        require_device(device, "Trainer")
        arguments = (params, *self._data)
        leaves, definition = _tree.flatten(arguments)
        names = _leaf_names(loss, definition._node)
        # Escape path collisions, including user keys already containing '#'.
        occupied = set()
        for i, name in enumerate(names):
            while name in occupied:
                name += "#"
            names[i] = name
            occupied.add(name)
        staged = _trace(loss, definition._node, leaves, names)
        if len(staged.outputs) != 1 or not isinstance(
            staged.outputs[0], _quabla.TraceTensor
        ):
            raise ValueError("Trainer loss must return one scalar Tensor")
        if _math.prod(staged.outputs[0].shape) != 1:
            raise ValueError("Trainer loss must be scalar")
        self._parameter_names = names[: len(parameters)]
        self._batch_names = []
        offset = len(parameters)
        retained = []
        for i, argument in enumerate(self._data):
            count = len(_tree.flatten(argument)[0])
            argument_names = names[offset : offset + count]
            if i not in self._batch_argnums:
                retained.extend(argument_names)
            self._batch_names.append(argument_names)
            offset += count
        factory = getattr(_quabla, f"{target}_adam_loss_optimizer")
        kwargs = {"device_ordinal": ordinal} if target == "cuda" else {}
        self._executor = factory(
            staged.outputs[0],
            self._parameter_names,
            dict(zip(names, leaves)),
            optimizer.learning_rate,
            retained,
            optimizer.b1,
            optimizer.b2,
            optimizer.eps,
            **kwargs,
        )

    @property
    def params(self):
        """Read current parameters into their original pytree structure."""
        if self._executor is None:
            return self._params
        values = self._executor.parameters()
        return _tree.unflatten(
            self._param_def, [values[name] for name in self._parameter_names]
        )

    def step(self, *batch):
        """Advance one optimizer step, optionally replacing dynamic data."""
        if self._executor is not None and not batch:
            return self._executor.step()
        inputs = {}
        pending_data = self._data
        if batch:
            if len(batch) != len(self._batch_argnums):
                raise ValueError("batch argument count must match batch_argnums")
            replacement = []
            for value, (old_specs, old_def) in zip(batch, self._batch_specs):
                value = _tree.map(_asarray, value)
                leaves, definition = _tree.flatten(value)
                if definition != old_def:
                    raise ValueError("batch pytree structure must match initial data")
                if any(
                    tuple(leaf.shape) != shape or leaf.dtype != dtype
                    for leaf, (shape, dtype) in zip(leaves, old_specs)
                ):
                    raise ValueError("batch shapes and dtypes must match initial data")
                replacement.append((value, leaves))
            data = list(self._data)
            for index, (value, leaves) in zip(self._batch_argnums, replacement):
                data[index] = value
                if self._executor is not None:
                    inputs.update(zip(self._batch_names[index], leaves))
            pending_data = tuple(data)
        if self._executor is not None:
            result = self._executor.step(inputs)
            self._data = pending_data
            return result
        _, gradients = self._value_and_grad(self._params, *pending_data)
        params, state = self._optimizer.update(self._params, gradients, self._state)
        self._params, self._state, self._data = params, state, pending_data

    def loss(self):
        """Evaluate loss with current parameters and the latest data."""
        if self._executor is not None:
            return self._executor.loss()
        return self._evaluate_loss(self._params, *self._data)
