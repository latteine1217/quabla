"""Learning-rate schedules: host callables ``schedule(step) -> float``.

``step`` counts completed optimizer updates and starts at 0, so the first
update uses ``schedule(0)``, as optax's ``scale_by_schedule`` does. Formulas
and argument order follow the optax schedule of the same role; invalid
configurations raise ``ValueError`` where optax warns and falls back.
"""

import math as _math
import operator as _operator


def _finite(value, name):
    value = float(value)
    if not _math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _steps(value, name):
    value = _operator.index(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def constant(value):
    """Return ``value`` at every step (optax ``constant_schedule``)."""
    value = _finite(value, "value")

    def schedule(step):
        return value

    return schedule


def exponential_decay(
    init_value,
    transition_steps,
    decay_rate,
    transition_begin=0,
    staircase=False,
    end_value=None,
):
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
    if end_value is not None:
        end_value = _finite(end_value, "end_value")
        clip = max if decay_rate < 1 else min
    staircase = bool(staircase)

    def schedule(step):
        count = step - transition_begin
        if count <= 0:
            value = init_value
        else:
            exponent = count / transition_steps
            if staircase:
                exponent = _math.floor(exponent)
            value = init_value * decay_rate**exponent
        if end_value is not None:
            value = clip(value, end_value)
        return value

    return schedule


def cosine_decay(init_value, decay_steps, alpha=0.0):
    """``init_value * ((1 - alpha) * 0.5 * (1 + cos(pi * t / decay_steps)) + alpha)``.

    ``t = min(step, decay_steps)``, so the rate stays at
    ``alpha * init_value`` after ``decay_steps`` (optax
    ``cosine_decay_schedule`` with ``exponent=1``).
    """
    init_value = _finite(init_value, "init_value")
    decay_steps = _steps(decay_steps, "decay_steps")
    alpha = _finite(alpha, "alpha")

    def schedule(step):
        count = min(step, decay_steps)
        cosine = 0.5 * (1 + _math.cos(_math.pi * count / decay_steps))
        return init_value * ((1 - alpha) * cosine + alpha)

    return schedule


def warmup_cosine_decay(
    init_value, peak_value, warmup_steps, decay_steps, end_value=0.0
):
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
    alpha = 0.0 if peak_value == 0 else end_value / peak_value
    decay = cosine_decay(peak_value, decay_steps - warmup_steps, alpha)

    def schedule(step):
        if step < warmup_steps:
            # optax linear_schedule: the fraction of warmup left, times the
            # rise, plus the end point.
            fraction = 1 - step / warmup_steps
            return (init_value - peak_value) * fraction + peak_value
        return decay(step - warmup_steps)

    return schedule


def piecewise_constant(boundaries, values):
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

    def schedule(step):
        return values[sum(step >= b for b in boundaries)]

    return schedule
