//! Array ops composed of other ops, each defined once over the [`Primitives`]
//! it is made of, which a `TraceTensor` implements by recording graph nodes.
//! An eager `Tensor` runs the same definition: its first call with operands
//! of a new shape, dtype, and weak type traces the composite and compiles
//! the graph to one fused CPU kernel (`TensorIr::fused_kernel`), which later
//! calls run without building a graph. The kernel computes the graph's value
//! bit for bit, so an eager composite equals its traced graph.

use quabla_core::tensor_ir::{TensorComparison, TensorDType};

use crate::tensor::bool_operation_error;

/// The primitive ops the composites are made of. Each method behaves as the
/// `TraceTensor` builder method of the same meaning: the same graph node,
/// dtype promotion, weak type, and errors.
pub(crate) trait Primitives: Sized {
    fn dtype(&self) -> Result<TensorDType, String>;

    /// `self op rhs` for an op name of `TraceTensor::binary`.
    fn binary(&self, rhs: &Self, op: &'static str) -> Result<Self, String>;

    /// `self op value` with a Python number, a weak scalar.
    fn scalar_binary(&self, value: f64, op: &'static str) -> Result<Self, String>;

    /// `value op self` with a Python number, a weak scalar.
    fn scalar_left_binary(&self, value: f64, op: &'static str) -> Result<Self, String>;

    fn compare(&self, rhs: &Self, kind: TensorComparison) -> Result<Self, String>;

    fn compare_scalar(&self, value: f64, kind: TensorComparison) -> Result<Self, String>;

    fn isnan(&self) -> Result<Self, String>;

    fn logical_or(&self, rhs: &Self) -> Result<Self, String>;

    /// `where(self, on_true, on_false)`.
    fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self, String>;

    /// A weak scalar operand that can meet `self`.
    fn scalar(&self, value: f64) -> Result<Self, String>;

    fn exp(&self) -> Result<Self, String>;

    fn log1p(&self) -> Result<Self, String>;

    fn ensure_not_bool(&self, op: &str) -> Result<(), String> {
        if self.dtype()? == TensorDType::Bool {
            return Err(bool_operation_error(op));
        }
        Ok(())
    }
}

/// `isnan(x) | ordered`, the `maximum`/`minimum` mask for a float `x`: a
/// NaN `x` selects itself, and a NaN right operand fails the ordered
/// comparison and is selected, so NaN propagates from either side like
/// NumPy and JAX.
fn nan_or<T: Primitives>(x: &T, ordered: T) -> Result<T, String> {
    x.isnan()?.logical_or(&ordered)
}

/// The value of floating `maximum`/`minimum` where `x == y`, with JAX's
/// derivative there: half to each operand. It equals the common value, and
/// between `+0` and `-0` it is IEEE 754-2019's `maximum` (`+0`) or `minimum`
/// (`-0`), as the `max`/`min` reductions are, so the result does not
/// depend on the operand order.
///
/// For `|x| <= 1`, `(x + y) / 2` is exact, and `+0 + -0 == +0` gives the
/// maximum's zero; the minimum negates both operands around the sum so its
/// zero is `-0`. Larger values use `x / 2 + y / 2`, which cannot overflow,
/// is exact for normal numbers, and keeps equal infinities.
fn tie<T: Primitives>(x: &T, y: &T, maximum: bool) -> Result<T, String> {
    let big = x
        .compare_scalar(1.0, TensorComparison::Greater)?
        .logical_or(&x.compare_scalar(-1.0, TensorComparison::Less)?)?;
    let halves = x
        .scalar_binary(0.5, "mul")?
        .binary(&y.scalar_binary(0.5, "mul")?, "add")?;
    let small = if maximum {
        x.binary(y, "add")?.scalar_binary(0.5, "mul")?
    } else {
        x.scalar_binary(-1.0, "mul")?
            .binary(&y.scalar_binary(-1.0, "mul")?, "add")?
            .scalar_binary(-0.5, "mul")?
    };
    big.select(&halves, &small)
}

/// [`tie`] against the Python number `y`, whose branch is known in advance.
fn tie_scalar<T: Primitives>(x: &T, y: f64, maximum: bool) -> Result<T, String> {
    if y.abs() > 1.0 {
        return x.scalar_binary(0.5, "mul")?.scalar_binary(0.5 * y, "add");
    }
    if maximum {
        x.scalar_binary(y, "add")?.scalar_binary(0.5, "mul")
    } else {
        x.scalar_binary(-1.0, "mul")?
            .scalar_binary(-y, "add")?
            .scalar_binary(-0.5, "mul")
    }
}

/// Floating `maximum` (`maximum == true`) or `minimum` with JAX's rules: a
/// NaN in either operand propagates, the larger (smaller) operand is
/// selected and takes the whole derivative, and ties are [`tie`]. Ties used
/// to select `y` with its whole derivative, so `maximum(+0, -0)` was `-0`.
fn extremum<T: Primitives>(x: &T, y: &T, maximum: bool) -> Result<T, String> {
    let (first, second) = if maximum {
        (TensorComparison::Greater, TensorComparison::Less)
    } else {
        (TensorComparison::Less, TensorComparison::Greater)
    };
    let take_x = nan_or(x, x.compare(y, first)?)?;
    let take_y = nan_or(y, x.compare(y, second)?)?;
    take_x.select(x, &take_y.select(y, &tie(x, y, maximum)?)?)
}

/// [`extremum`] against the Python number `y`.
fn extremum_scalar<T: Primitives>(x: &T, y: f64, maximum: bool) -> Result<T, String> {
    let (first, second) = if maximum {
        (TensorComparison::Greater, TensorComparison::Less)
    } else {
        (TensorComparison::Less, TensorComparison::Greater)
    };
    let constant = x.scalar(y)?;
    let take_x = nan_or(x, x.compare_scalar(y, first)?)?;
    // A NaN `y` fails both comparisons and reaches the tie, whose value is
    // then NaN as well.
    let take_y = x.compare_scalar(y, second)?;
    take_x.select(x, &take_y.select(&constant, &tie_scalar(x, y, maximum)?)?)
}

/// A `bool` left operand cannot be NaN or tie with a derivative; it keeps
/// the legacy `greater` mask so its promotion and errors stay unchanged.
pub(crate) fn maximum<T: Primitives>(x: &T, y: &T) -> Result<T, String> {
    if !x.dtype()?.is_floating() {
        return x.binary(y, "greater")?.select(x, y);
    }
    extremum(x, y, true)
}

pub(crate) fn maximum_scalar<T: Primitives>(x: &T, y: f64) -> Result<T, String> {
    if !x.dtype()?.is_floating() {
        return x.scalar_binary(y, "greater")?.select(x, &x.scalar(y)?);
    }
    extremum_scalar(x, y, true)
}

pub(crate) fn minimum<T: Primitives>(x: &T, y: &T) -> Result<T, String> {
    if !x.dtype()?.is_floating() {
        return y.binary(x, "greater")?.select(x, y);
    }
    extremum(x, y, false)
}

pub(crate) fn minimum_scalar<T: Primitives>(x: &T, y: f64) -> Result<T, String> {
    if !x.dtype()?.is_floating() {
        return x.scalar_left_binary(y, "greater")?.select(x, &x.scalar(y)?);
    }
    extremum_scalar(x, y, false)
}

/// `max(x, 0)`, with a zero subgradient at zero.
pub(crate) fn relu<T: Primitives>(x: &T) -> Result<T, String> {
    x.ensure_not_bool("relu")?;
    // Not `maximum(x, 0)`, whose tie at zero shares the derivative: JAX's
    // `relu` has derivative 0 at 0, and both zeros map to `+0`.
    nan_or(x, x.compare_scalar(0.0, TensorComparison::Greater)?)?.select(x, &x.scalar(0.0)?)
}

/// `where(x > 0, x, 0 - x)`: zero takes the `0 - x` branch, which gives
/// `+0` for both signed zeros.
pub(crate) fn abs<T: Primitives>(x: &T) -> Result<T, String> {
    x.ensure_not_bool("abs")?;
    x.scalar_binary(0.0, "greater")?
        .select(x, &x.scalar_left_binary(0.0, "sub")?)
}

/// `exp(-|x|)`, which never overflows. `abs` takes its `0 - x` branch at
/// zero, so the derivative there is that of `exp(x)`.
fn exp_negative_abs<T: Primitives>(x: &T) -> Result<T, String> {
    abs(x)?.scalar_binary(-1.0, "mul")?.exp()
}

/// `where(x > 0, 1 / (1 + z), z / (1 + z))` with `z = exp(-|x|)`. Both
/// branches stay finite for every finite `x`, so the unselected branch
/// cannot form `0 * inf` in the gradient (the textbook `1 / (1 + exp(-x))`
/// overflows `exp` for large negative `x`). Zero takes the `z / (1 + z)`
/// branch, which matches the branch `abs` takes.
pub(crate) fn sigmoid<T: Primitives>(x: &T) -> Result<T, String> {
    x.ensure_not_bool("sigmoid")?;
    let decay = exp_negative_abs(x)?;
    let denominator = decay.scalar_binary(1.0, "add")?;
    let positive = denominator.scalar_left_binary(1.0, "div")?;
    let negative = decay.binary(&denominator, "div")?;
    x.compare_scalar(0.0, TensorComparison::Greater)?
        .select(&positive, &negative)
}

/// `relu(x) + log1p(exp(-|x|))`; `log1p` keeps the correction accurate
/// where `exp(-|x|)` is below the dtype epsilon. `relu`'s zero derivative
/// at 0 and the `-1` that `abs` takes there sum to `sigmoid(0) = 1/2`.
pub(crate) fn softplus<T: Primitives>(x: &T) -> Result<T, String> {
    x.ensure_not_bool("softplus")?;
    let linear = relu(x)?;
    let correction = exp_negative_abs(x)?.log1p()?;
    linear.binary(&correction, "add")
}
