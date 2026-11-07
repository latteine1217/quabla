"""Learning-rate schedules: host callables ``schedule(step) -> float``.

``step`` counts completed optimizer updates and starts at 0, so the first
update uses ``schedule(0)``, as optax's ``scale_by_schedule`` does. Formulas
and argument order follow the optax schedule of the same role; invalid
configurations raise ``ValueError`` where optax warns and falls back.
"""

from __future__ import annotations

import math as _math
import operator as _operator
import sys as _sys
from collections.abc import Callable, Sequence


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not _math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _steps(value: int, name: str) -> int:
    value = _operator.index(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def constant(value: float) -> Callable[[int], float]:
    """Return ``value`` at every step (optax ``constant_schedule``)."""
    value = _finite(value, "value")

    def schedule(step: int) -> float:
        return value

    return schedule


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: float | None = None,
) -> Callable[[int], float]:
    """``init_value * decay_rate ** ((step - transition_begin) / transition_steps)``.

    The rate holds ``init_value`` until ``transition_begin``; ``staircase``
    floors the exponent; ``end_value`` bounds the decay from below when
    ``decay_rate < 1`` and the growth from above otherwise. Matches optax
    ``exponential_decay`` with the same argument order.
    """
    init_value = _finite(init_value, "init_value")
    transition_steps = _steps(transition_steps, "transition_steps")
    decay_rate = _finite(decay_rate, "decay_rate")
    if decay_rate <= 0:
        raise ValueError("decay_rate must be positive")
    transition_begin = _operator.index(transition_begin)
    if transition_begin < 0:
        raise ValueError("transition_begin must be nonnegative")
    clip = max if decay_rate < 1 else min
    if end_value is not None:
        end_value = _finite(end_value, "end_value")
    staircase = bool(staircase)

    def schedule(step: int) -> float:
        count = step - transition_begin
        if count <= 0:
            value = init_value
        else:
            exponent = count / transition_steps
            if staircase:
                exponent = _math.floor(exponent)
            try:
                factor = decay_rate**exponent
            except OverflowError:
                factor = _math.inf
            if init_value == 0.0:
                value = init_value
            elif not _math.isfinite(factor) or factor < _sys.float_info.min:
                # Combine scales before exponentiating: an overflowing or
                # subnormal power can still give a representable product.
                log_value = _math.log(abs(init_value)) + exponent * _math.log(decay_rate)
                try:
                    value = _math.copysign(_math.exp(log_value), init_value)
                except OverflowError:
                    if end_value is None:
                        raise
                    value = _math.copysign(_math.inf, init_value)
            else:
                value = init_value * factor
        if end_value is not None:
            value = clip(value, end_value)
        return value

    return schedule


def cosine_decay(init_value: float, decay_steps: int, alpha: float = 0.0) -> Callable[[int], float]:
    """``init_value * ((1 - alpha) * 0.5 * (1 + cos(pi * t / decay_steps)) + alpha)``.

    ``t = min(step, decay_steps)``, so the rate stays at
    ``alpha * init_value`` after ``decay_steps`` (optax
    ``cosine_decay_schedule`` with ``exponent=1``).
    """
    init_value = _finite(init_value, "init_value")
    decay_steps = _steps(decay_steps, "decay_steps")
    alpha = _finite(alpha, "alpha")

    def schedule(step: int) -> float:
        count = min(step, decay_steps)
        cosine = 0.5 * (1 + _math.cos(_math.pi * count / decay_steps))
        return init_value * (cosine + alpha * (1.0 - cosine))

    return schedule


def warmup_cosine_decay(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
) -> Callable[[int], float]:
    """Linear warmup to ``peak_value``, then cosine decay to ``end_value``.

    ``decay_steps`` includes the warmup, as in optax
    ``warmup_cosine_decay_schedule``: the rate is ``peak_value`` at step
    ``warmup_steps`` and ``end_value`` from step ``decay_steps`` on.
    """
    init_value = _finite(init_value, "init_value")
    peak_value = _finite(peak_value, "peak_value")
    end_value = _finite(end_value, "end_value")
    warmup_steps = _operator.index(warmup_steps)
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    decay_steps = _operator.index(decay_steps)
    if decay_steps <= warmup_steps:
        raise ValueError("decay_steps must exceed warmup_steps")
    decay = cosine_decay(1.0, decay_steps - warmup_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            # Weighted endpoints avoid overflowing their difference when
            # finite endpoints have opposite signs.
            fraction = step / warmup_steps
            return init_value * (1.0 - fraction) + peak_value * fraction
        fraction = decay(step - warmup_steps)
        return peak_value * fraction + end_value * (1.0 - fraction)

    return schedule


def piecewise_constant(boundaries: Sequence[int], values: Sequence[float]) -> Callable[[int], float]:
    """``values[i]`` for ``boundaries[i - 1] <= step < boundaries[i]``.

    ``values`` has one more entry than the strictly increasing
    ``boundaries``. A value takes effect at its boundary step, as a scale in
    optax ``piecewise_constant_schedule`` does.
    """
    boundaries = [_operator.index(b) for b in boundaries]
    values = [_finite(v, "values") for v in values]
    if len(values) != len(boundaries) + 1:
        raise ValueError("values must have one more entry than boundaries")
    if any(b < 0 for b in boundaries) or any(
        a >= b for a, b in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("boundaries must be nonnegative and strictly increasing")

    def schedule(step: int) -> float:
        return values[sum(step >= b for b in boundaries)]

    return schedule
