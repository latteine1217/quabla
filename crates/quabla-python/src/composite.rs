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

/// `where(isnan(x) | (x > y), x, y)`: ties select `y`, and NaN in either
/// operand propagates. A `bool` left operand cannot be NaN; it keeps the
/// legacy `greater` mask so its promotion and errors stay unchanged.
pub(crate) fn maximum<T: Primitives>(x: &T, y: &T) -> Result<T, String> {
    if !x.dtype()?.is_floating() {
        return x.binary(y, "greater")?.select(x, y);
    }
    nan_or(x, x.compare(y, TensorComparison::Greater)?)?.select(x, y)
}

pub(crate) fn maximum_scalar<T: Primitives>(x: &T, y: f64) -> Result<T, String> {
    let mask = if x.dtype()?.is_floating() {
        nan_or(x, x.compare_scalar(y, TensorComparison::Greater)?)?
    } else {
        x.scalar_binary(y, "greater")?
    };
    mask.select(x, &x.scalar(y)?)
}

/// `where(isnan(x) | (y > x), x, y)`: ties select `y`, and NaN in either
/// operand propagates.
pub(crate) fn minimum<T: Primitives>(x: &T, y: &T) -> Result<T, String> {
    if !x.dtype()?.is_floating() {
        return y.binary(x, "greater")?.select(x, y);
    }
    nan_or(x, y.compare(x, TensorComparison::Greater)?)?.select(x, y)
}

pub(crate) fn minimum_scalar<T: Primitives>(x: &T, y: f64) -> Result<T, String> {
    let mask = if x.dtype()?.is_floating() {
        nan_or(x, x.compare_scalar(y, TensorComparison::Less)?)?
    } else {
        x.scalar_left_binary(y, "greater")?
    };
    mask.select(x, &x.scalar(y)?)
}

/// `maximum(x, 0)`, with a zero subgradient at zero.
pub(crate) fn relu<T: Primitives>(x: &T) -> Result<T, String> {
    x.ensure_not_bool("relu")?;
    maximum_scalar(x, 0.0)
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

/// `maximum(x, 0) + log1p(exp(-|x|))`; `log1p` keeps the correction
/// accurate where `exp(-|x|)` is below the dtype epsilon.
pub(crate) fn softplus<T: Primitives>(x: &T) -> Result<T, String> {
    x.ensure_not_bool("softplus")?;
    let linear = maximum_scalar(x, 0.0)?;
    let correction = exp_negative_abs(x)?.log1p()?;
    linear.binary(&correction, "add")
}
