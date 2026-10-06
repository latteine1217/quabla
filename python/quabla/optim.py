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
from ._lbfgs import LBFGS
from ._schedules import (
    constant,
    cosine_decay,
    exponential_decay,
    piecewise_constant,
    warmup_cosine_decay,
)

__all__ = [
    "Adam",
    "AdamW",
    "LBFGS",
    "SGD",
    "Trainer",
    "clip_by_global_norm",
    "constant",
    "cosine_decay",
    "exponential_decay",
    "piecewise_constant",
    "warmup_cosine_decay",
]


def _positive(value, name):
    value = float(value)
    if not _math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _learning_rate(value):
    """A positive constant rate, or a schedule called with the step count."""
    return value if callable(value) else _positive(value, "learning_rate")


def _rate_at(learning_rate, step):
    if not callable(learning_rate):
        return learning_rate
    value = float(learning_rate(step))
    if not _math.isfinite(value) or value < 0:
        raise ValueError(
            f"learning-rate schedule returned {value!r} at step {step}; "
            "rates must be finite and nonnegative"
        )
    return value


def _clip_norm(value):
    return None if value is None else _positive(value, "clip_norm")


def clip_by_global_norm(grads, max_norm):
    """Scale a gradient pytree so its global L2 norm is at most ``max_norm``.

    Returns ``(clipped_grads, global_norm)``. The norm over all leaves is
    computed in float64 as ``m * sqrt(sum((g / m) ** 2))`` with ``m`` the
    largest ``|g|``, so it neither overflows nor underflows for extreme
    magnitudes. Leaves are multiplied by ``min(1, max_norm / global_norm)``
    and rounded once to their dtype; below the bound the input leaves are
    returned unchanged. A NaN or infinite global norm returns all-NaN leaves
    instead of finite values, so the following update exposes the bad step;
    test ``math.isfinite(global_norm)`` to skip such a step instead.
    """
    max_norm = _positive(max_norm, "max_norm")
    leaves, definition = _tree.flatten(grads)
    for leaf in leaves:
        if not isinstance(leaf, _Tensor) or leaf.dtype.name not in (
            "float32",
            "float64",
        ):
            raise TypeError("gradient leaves must be floating-point Tensors")
    wide = [leaf.astype(_float64) for leaf in leaves if _math.prod(leaf.shape)]
    peaks = [float(value.abs().max()) for value in wide]
    # Python's max() is order-dependent with NaN, so propagate it explicitly.
    peak = _math.nan if any(_math.isnan(p) for p in peaks) else max(peaks, default=0)
    scale = peak if _math.isfinite(peak) and peak > 0 else 1.0
    total = sum(float(((value / scale) * (value / scale)).sum()) for value in wide)
    norm = scale * _math.sqrt(total)
    if norm <= max_norm:
        return grads, norm
    factor = max_norm / norm if _math.isfinite(norm) else _math.nan
    clipped = [(leaf.astype(_float64) * factor).astype(leaf.dtype) for leaf in leaves]
    return _tree.unflatten(definition, clipped), norm


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


class _AdamBase:
    """Pure Adam state and update shared by Adam and AdamW.

    State stores float64 moments and the completed-update count, which is
    both the bias-correction exponent and the learning-rate schedule step.
    """

    weight_decay = 0.0

    def _configure(self, learning_rate, b1, b2, eps, clip_norm):
        self.learning_rate = _learning_rate(learning_rate)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = _positive(eps, "epsilon")
        if not (0 <= self.b1 < 1 and 0 <= self.b2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        self.clip_norm = _clip_norm(clip_norm)

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
        if self.clip_norm is not None:
            gradients, _ = clip_by_global_norm(gradients, self.clip_norm)
        step = state["step"] + 1
        moments, variances, updated = [], [], []
        hyperparameters = (
            _rate_at(self.learning_rate, state["step"]),
            self.b1,
            self.b2,
            self.eps,
            1 - self.b1**step,
            1 - self.b2**step,
        )
        for parameter, gradient, m, v in zip(parameters, gradients, first, second):
            parameter, m, v = parameter._adam_update(
                gradient, m, v, hyperparameters, self.weight_decay
            )
            updated.append(parameter)
            moments.append(m)
            variances.append(v)
        return _tree.unflatten(definition, updated), {
            "step": step,
            "m": _tree.unflatten(definition, moments),
            "v": _tree.unflatten(definition, variances),
        }


class Adam(_AdamBase):
    """Adam configuration with pure updates and the v0.1 stateful ``step``.

    ``learning_rate`` is a positive float or a schedule ``f(step) -> float``
    called with the completed-update count. ``clip_norm`` clips each update's
    gradients with ``clip_by_global_norm`` first. ``beta1``, ``beta2``, and
    ``epsilon`` remain accepted constructor aliases. Pure state stores float64
    moments, independently of stateful legacy steps.
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
        clip_norm=None,
    ):
        self._configure(
            learning_rate,
            b1 if beta1 is None else beta1,
            b2 if beta2 is None else beta2,
            eps if epsilon is None else epsilon,
            clip_norm,
        )
        # The native v0.1 optimizer has a fixed rate and no clipping.
        self._legacy = None
        if not callable(self.learning_rate) and self.clip_norm is None:
            self._legacy = _quabla.Adam(self.learning_rate, self.b1, self.b2, self.eps)

    def step(self, parameters, gradients):
        """Apply a stateful v0.1 dictionary update with native compatibility."""
        if self._legacy is None:
            raise TypeError(
                "legacy Adam.step needs a constant learning_rate and no "
                "clip_norm; use init/update"
            )
        return self._legacy.step(parameters, gradients)


class AdamW(_AdamBase):
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019).

    Each update is ``p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)``
    with the pre-update ``p``, as optax's ``adamw``; the decay is not part of
    the gradient, so it does not enter the moments. State, precision, schedule,
    and ``clip_norm`` rules are those of ``Adam``.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
        *,
        clip_norm=None,
    ):
        self._configure(learning_rate, b1, b2, eps, clip_norm)
        self.weight_decay = float(weight_decay)
        if not _math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


class SGD:
    """Plain stochastic gradient descent with pure pytree updates.

    With a constant ``learning_rate`` the state is ``None``; a schedule keeps
    ``{"step": n}``, the completed-update count it is evaluated at.
    ``clip_norm`` clips each update's gradients with ``clip_by_global_norm``.
    """

    def __init__(self, learning_rate=1e-2, *, clip_norm=None):
        self.learning_rate = _learning_rate(learning_rate)
        self.clip_norm = _clip_norm(clip_norm)

    def init(self, params):
        _parameters(params)
        return {"step": 0} if callable(self.learning_rate) else None

    def update(self, params, grads, state):
        parameters, gradients, definition = _matching(params, grads, "gradient")
        if callable(self.learning_rate):
            if (
                type(state) is not dict
                or set(state) != {"step"}
                or type(state["step"]) is not int
                or state["step"] < 0
            ):
                raise ValueError("scheduled SGD state must be {'step': int >= 0}")
            rate = _rate_at(self.learning_rate, state["step"])
            state = {"step": state["step"] + 1}
        elif state is not None:
            raise ValueError("SGD state must be None")
        else:
            rate = self.learning_rate
        if self.clip_norm is not None:
            gradients, _ = clip_by_global_norm(gradients, self.clip_norm)
        updated = [p._sgd_update(g, rate) for p, g in zip(parameters, gradients)]
        return _tree.unflatten(definition, updated), state


class Trainer:
    """Trace ``loss(params, *data)`` and keep optimizer state between steps.

    ``batch_argnums`` indexes the data arguments (excluding parameters).
    ``step(*batch)`` replaces those arguments in the declared order. An empty
    call reuses the latest batch; ``loss()`` evaluates the current parameters.
    Device plans keep parameter and moment buffers on the device until readback.
    Every device accepts Adam, AdamW, and SGD with the CPU's update rules,
    including ``clip_norm``: the global norm and its clip factor are reduced on
    the device, so gradients are never read back. A schedule is evaluated on
    the host and sets the native rate before every step. Device plans compute
    in float32 by default and agree with the CPU trainer up to that rounding.

    ``precision`` is ``jit``'s opt-in: ``"float64"`` on CUDA runs a float64
    loss natively in double and keeps parameters and moments in float64 on
    the device. The CPU already trains in float64, and MLX, which has no
    float64 arithmetic, rejects it.
    """

    def __init__(
        self,
        loss,
        params,
        optimizer,
        *data,
        device="cpu",
        batch_argnums=(),
        precision=None,
    ):
        from ._devices import parse_device, require_device

        if not callable(loss):
            raise TypeError("loss must be callable")
        if not isinstance(optimizer, (Adam, AdamW, SGD)):
            raise TypeError("optimizer must be Adam, AdamW, or SGD")
        if precision not in (None, "float64"):
            raise ValueError(f'precision must be None or "float64", got {precision!r}')
        target, ordinal = parse_device(device)
        if precision == "float64" and target == "mlx":
            raise _UnsupportedOperationError(
                'MLX has no float64 arithmetic; precision="float64" is supported on '
                "the CPU and CUDA",
                op="float64",
                device=device,
            )
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
        factory = getattr(_quabla, f"_{target}_trainer_optimizer")
        if isinstance(optimizer, SGD):
            # The native SGD rule ignores the Adam hyperparameters.
            rule = {
                "optimizer": "sgd",
                "beta1": 0.0,
                "beta2": 0.0,
                "epsilon": 1.0,
                "weight_decay": 0.0,
            }
        else:
            rule = {
                "optimizer": "adam",
                "beta1": optimizer.b1,
                "beta2": optimizer.b2,
                "epsilon": optimizer.eps,
                "weight_decay": optimizer.weight_decay,
            }
        if target == "cuda":
            rule.update(precision=precision, device_ordinal=ordinal)
        self._executor = factory(
            staged.outputs[0],
            self._parameter_names,
            dict(zip(names, leaves)),
            retained,
            learning_rate=_rate_at(optimizer.learning_rate, 0),
            clip_norm=optimizer.clip_norm,
            **rule,
        )
        # Completed device steps, at which a host schedule is evaluated.
        self._device_steps = 0

    def _device_step(self, *inputs):
        schedule = self._optimizer.learning_rate
        if callable(schedule):
            self._executor.learning_rate = _rate_at(schedule, self._device_steps)
        result = self._executor.step(*inputs)
        self._device_steps += 1
        return result

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
            return self._device_step()
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
            result = self._device_step(inputs)
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
