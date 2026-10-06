//! The [`UnaryMathKind`] functions of the `UnaryMath` op and the binary
//! `Fmod` op: their `f64` reference values and derivatives, the CUDA
//! function each one lowers to, and the symbolic derivative rules, which are
//! written with IR ops so that every derivative order is again an ordinary
//! graph on every backend.
//!
//! Values follow IEEE semantics and never raise: an input outside the domain
//! of a function (`arcsin(2)`, `log2(-1)`, `arccosh(0.5)`) gives NaN, a pole
//! gives an infinity (`arctanh(1) == inf`, `log10(0) == -inf`), as in NumPy.

use super::{TensorComparison, TensorIr, TensorNodeId, TensorOp};

/// An elementwise function of one operand, executed by the `UnaryMath` op.
/// The kinds share one op because they share every rule except their value
/// and derivative; each still lowers to the platform's own function (Rust
/// std or the `libm` crate on the CPU, the CUDA math library, and the MLX op
/// where MLX has one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryMathKind {
    /// `tan(x)`; derivative `1 + tan(x)^2`, formed from the result.
    Tan,
    /// `arcsin(x)` on `[-1, 1]`; derivative `1 / sqrt((1 - x)(1 + x))`.
    Arcsin,
    /// `arccos(x)` on `[-1, 1]`; derivative `-1 / sqrt((1 - x)(1 + x))`.
    Arccos,
    /// `arctan(x)`; derivative `1 / (1 + x^2)`.
    Arctan,
    /// `sinh(x)`; derivative `cosh(x)`.
    Sinh,
    /// `cosh(x)`; derivative `sinh(x)`.
    Cosh,
    /// `arcsinh(x)`; derivative `1 / sqrt(x^2 + 1)`, evaluated without
    /// forming `x^2` for `|x| > 1` (see `TensorIr::arcsinh_derivative`).
    Arcsinh,
    /// `arccosh(x)` on `[1, inf]`; derivative
    /// `1 / (sqrt(x - 1) sqrt(x + 1))`.
    Arccosh,
    /// `arctanh(x)` on `[-1, 1]`; derivative `1 / ((1 - x)(1 + x))`.
    Arctanh,
    /// `log2(x)`; derivative `1 / (x ln 2)`.
    Log2,
    /// `log10(x)`; derivative `1 / (x ln 10)`.
    Log10,
    /// The real cube root, negative for negative `x`; derivative
    /// `1 / (3 cbrt(x)^2)`, formed from the result.
    Cbrt,
    /// The largest integer not above `x`; derivative zero.
    Floor,
    /// The smallest integer not below `x`; derivative zero.
    Ceil,
    /// The nearest integer, halfway cases to the even one (NumPy's and
    /// JAX's `round`); derivative zero.
    Round,
}

impl UnaryMathKind {
    pub const ALL: [Self; 15] = [
        Self::Tan,
        Self::Arcsin,
        Self::Arccos,
        Self::Arctan,
        Self::Sinh,
        Self::Cosh,
        Self::Arcsinh,
        Self::Arccosh,
        Self::Arctanh,
        Self::Log2,
        Self::Log10,
        Self::Cbrt,
        Self::Floor,
        Self::Ceil,
        Self::Round,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Tan => "tan",
            Self::Arcsin => "arcsin",
            Self::Arccos => "arccos",
            Self::Arctan => "arctan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Arcsinh => "arcsinh",
            Self::Arccosh => "arccosh",
            Self::Arctanh => "arctanh",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Cbrt => "cbrt",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
        }
    }

    /// The `f64` function, the CPU reference that a `float32` node rounds
    /// once. The inverse hyperbolic functions use the musl implementations
    /// of the `libm` crate: Rust std's `acosh` is `ln(x + sqrt(x^2 - 1))`,
    /// which loses relative accuracy near `1`, where musl switches to
    /// `log1p`.
    pub(super) fn function(self) -> fn(f64) -> f64 {
        match self {
            Self::Tan => f64::tan,
            Self::Arcsin => f64::asin,
            Self::Arccos => f64::acos,
            Self::Arctan => f64::atan,
            Self::Sinh => f64::sinh,
            Self::Cosh => f64::cosh,
            Self::Arcsinh => libm::asinh,
            Self::Arccosh => libm::acosh,
            Self::Arctanh => libm::atanh,
            Self::Log2 => f64::log2,
            Self::Log10 => f64::log10,
            Self::Cbrt => f64::cbrt,
            Self::Floor => f64::floor,
            Self::Ceil => f64::ceil,
            Self::Round => f64::round_ties_even,
        }
    }

    /// The `f64` value of [`Self::function`] at `x`.
    pub fn evaluate(self, x: f64) -> f64 {
        self.function()(x)
    }

    /// Whether the function is piecewise constant, so that every derivative
    /// order is zero (and no derivative expression is built).
    pub(super) fn is_piecewise_constant(self) -> bool {
        matches!(self, Self::Floor | Self::Ceil | Self::Round)
    }

    /// The first derivative at `x`, the numeric counterpart of the symbolic
    /// rule `TensorIr::unary_math_derivative` (same formula and the same
    /// conventions at poles, domain edges, and infinities).
    pub(super) fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Tan => {
                let value = x.tan();
                1.0 + value * value
            }
            Self::Arcsin => 1.0 / ((1.0 - x) * (1.0 + x)).sqrt(),
            Self::Arccos => -1.0 / ((1.0 - x) * (1.0 + x)).sqrt(),
            Self::Arctan => 1.0 / (1.0 + x * x),
            Self::Sinh => x.cosh(),
            Self::Cosh => x.sinh(),
            Self::Arcsinh => arcsinh_derivative(x),
            Self::Arccosh => 1.0 / ((x - 1.0).sqrt() * (x + 1.0).sqrt()),
            Self::Arctanh => 1.0 / ((1.0 - x) * (1.0 + x)),
            Self::Log2 => std::f64::consts::LOG2_E / x,
            Self::Log10 => std::f64::consts::LOG10_E / x,
            Self::Cbrt => {
                let value = x.cbrt();
                1.0 / (3.0 * value * value)
            }
            Self::Floor | Self::Ceil | Self::Round => 0.0,
        }
    }

    /// The second derivative at `x`, the derivative of the symbolic first
    /// derivative rule.
    pub(super) fn second_derivative(self, x: f64) -> f64 {
        match self {
            // d/dx (1 + t^2) = 2 t (1 + t^2).
            Self::Tan => {
                let value = x.tan();
                2.0 * value * (1.0 + value * value)
            }
            // d/dx ((1 - x)(1 + x))^(-1/2) = x d^3.
            Self::Arcsin | Self::Arccos => {
                let derivative = self.derivative(x);
                x * derivative * derivative * derivative
            }
            // d/dx (1 + x^2)^(-1) = -2x d^2.
            Self::Arctan => {
                let derivative = self.derivative(x);
                -2.0 * x * derivative * derivative
            }
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            // -x d^3, with `x d` (at most 1 in magnitude) formed first so
            // that `d^3` cannot underflow before the result does. The limit
            // at +-inf is 0, which the symbolic rule also gives.
            Self::Arcsinh => {
                if x.is_infinite() {
                    return 0.0;
                }
                let derivative = arcsinh_derivative(x);
                -(x * derivative) * derivative * derivative
            }
            Self::Arccosh => {
                let derivative = self.derivative(x);
                -(x * derivative) * derivative * derivative
            }
            // d/dx ((1 - x)(1 + x))^(-1) = 2x d^2.
            Self::Arctanh => {
                let derivative = self.derivative(x);
                2.0 * x * derivative * derivative
            }
            Self::Log2 => -std::f64::consts::LOG2_E / (x * x),
            Self::Log10 => -std::f64::consts::LOG10_E / (x * x),
            // d/dx (3 v^2)^(-1) with v' = 1 / (3 v^2): -2 / (9 v^5).
            Self::Cbrt => {
                let value = x.cbrt();
                let derivative = 1.0 / (3.0 * value * value);
                -2.0 * derivative * derivative / value
            }
            Self::Floor | Self::Ceil | Self::Round => 0.0,
        }
    }

    /// The single-precision CUDA math function; `double_precision_source`
    /// rewrites each to its double overload (`tanf` -> `tan`). `rintf`
    /// rounds halfway cases to even in the default rounding mode, whereas
    /// `roundf` rounds them away from zero.
    pub(super) fn cuda_function(self) -> &'static str {
        match self {
            Self::Tan => "tanf",
            Self::Arcsin => "asinf",
            Self::Arccos => "acosf",
            Self::Arctan => "atanf",
            Self::Sinh => "sinhf",
            Self::Cosh => "coshf",
            Self::Arcsinh => "asinhf",
            Self::Arccosh => "acoshf",
            Self::Arctanh => "atanhf",
            Self::Log2 => "log2f",
            Self::Log10 => "log10f",
            Self::Cbrt => "cbrtf",
            Self::Floor => "floorf",
            Self::Ceil => "ceilf",
            Self::Round => "rintf",
        }
    }
}

/// `1 / sqrt(x^2 + 1)` as `1 / hypot(x, 1)`, which does not overflow for
/// large `|x|` (`x^2` overflows above about `1.3e154`, where the true value
/// is still about `1e-154`).
fn arcsinh_derivative(x: f64) -> f64 {
    1.0 / x.hypot(1.0)
}

/// C `fmod(x, y)`: `x - n y` with `n = trunc(x / y)` evaluated exactly, so
/// the result has the sign of `x` and is exact. Rust's `%` on `f64` is fmod.
pub(super) fn fmod(x: f64, y: f64) -> f64 {
    x % y
}

/// `d fmod(x, y) / dy = -n`, where `n` is the integer quotient of the
/// computed remainder: `x = n y + fmod(x, y)` holds exactly, so
/// `(x - fmod(x, y)) / y` is within rounding of the integer `n` and
/// rounding it to the nearest integer recovers `n` (exactly while
/// `|n| < 2^52`). This is `-trunc(x / y)` of the exact quotient; truncating
/// the rounded quotient `x / y` instead can be off by one where the
/// quotient rounds up to an integer. NaN where `fmod` is NaN; zero for an
/// infinite `y` (where `fmod(x, y) == x`).
pub(super) fn fmod_y_partial(x: f64, y: f64) -> f64 {
    -((x - fmod(x, y)) / y).round_ties_even()
}

impl TensorIr {
    /// Elementwise `kind(input)`; see [`UnaryMathKind`]. `Bool` inputs are
    /// rejected.
    pub fn unary_math(
        &mut self,
        input: TensorNodeId,
        kind: UnaryMathKind,
    ) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        self.push_derived(TensorOp::UnaryMath { input, kind }, shape)
    }

    /// Elementwise C `fmod(x, y)`, the remainder of the quotient truncated
    /// toward zero, with the sign of `x` (NaN for `y == 0` or an infinite
    /// `x`, `x` for an infinite `y`). Operands broadcast and promote like
    /// arithmetic operands, and `Bool` operands are rejected. The partials
    /// are `1` and `-trunc(x / y)` (see [`fmod_y_partial`]); both are
    /// piecewise constant, so every second partial is zero.
    pub fn fmod(&mut self, x: TensorNodeId, y: TensorNodeId) -> Result<TensorNodeId, String> {
        for operand in [x, y] {
            if self.node(operand)?.dtype == super::TensorDType::Bool {
                return Err(
                    "fmod is not defined for bool tensors; use logical_and/logical_or/logical_not \
                     (& | ~) or convert explicitly with astype"
                        .to_string(),
                );
            }
        }
        self.binary("fmod", x, y, |x, y| TensorOp::Fmod { x, y })
    }

    /// The symbolic first derivative of `kind` at `input`, whose value
    /// `kind(input)` is the node `value`; `None` for the piecewise-constant
    /// kinds, whose derivative is zero.
    pub(super) fn unary_math_derivative(
        &mut self,
        kind: UnaryMathKind,
        input: TensorNodeId,
        value: TensorNodeId,
    ) -> Result<Option<TensorNodeId>, String> {
        let one = self.scalar_constant(1.0);
        let derivative = match kind {
            UnaryMathKind::Floor | UnaryMathKind::Ceil | UnaryMathKind::Round => return Ok(None),
            UnaryMathKind::Tan => {
                let squared = self.mul(value, value)?;
                self.add(one, squared)?
            }
            // (1 - x)(1 + x) instead of 1 - x^2: near |x| = 1 the factor that
            // vanishes is formed exactly, so the pole keeps its relative
            // accuracy.
            UnaryMathKind::Arcsin | UnaryMathKind::Arccos => {
                let below = self.sub(one, input)?;
                let above = self.add(one, input)?;
                let product = self.mul(below, above)?;
                let root = self.sqrt(product)?;
                let numerator = if kind == UnaryMathKind::Arcsin {
                    one
                } else {
                    self.scalar_constant(-1.0)
                };
                self.div(numerator, root)?
            }
            UnaryMathKind::Arctan => {
                let squared = self.mul(input, input)?;
                let denominator = self.add(one, squared)?;
                self.div(one, denominator)?
            }
            UnaryMathKind::Sinh => self.unary_math(input, UnaryMathKind::Cosh)?,
            UnaryMathKind::Cosh => self.unary_math(input, UnaryMathKind::Sinh)?,
            UnaryMathKind::Arcsinh => self.arcsinh_derivative(input)?,
            // Two square roots instead of sqrt((x - 1)(x + 1)): the product
            // overflows above about 1.3e154, the product of the roots does
            // not, and each factor near x = 1 is still formed exactly.
            UnaryMathKind::Arccosh => {
                let below = self.sub(input, one)?;
                let above = self.add(input, one)?;
                let below_root = self.sqrt(below)?;
                let above_root = self.sqrt(above)?;
                let product = self.mul(below_root, above_root)?;
                self.div(one, product)?
            }
            UnaryMathKind::Arctanh => {
                let below = self.sub(one, input)?;
                let above = self.add(one, input)?;
                let product = self.mul(below, above)?;
                self.div(one, product)?
            }
            UnaryMathKind::Log2 | UnaryMathKind::Log10 => {
                let scale = self.scalar_constant(if kind == UnaryMathKind::Log2 {
                    std::f64::consts::LOG2_E
                } else {
                    std::f64::consts::LOG10_E
                });
                self.div(scale, input)?
            }
            UnaryMathKind::Cbrt => {
                let squared = self.mul(value, value)?;
                let third = self.scalar_constant(1.0 / 3.0);
                self.div(third, squared)?
            }
        };
        Ok(Some(derivative))
    }

    /// `1 / sqrt(x^2 + 1)` without forming `x^2` for `|x| > 1`: with
    /// `s = max(|x|, 1)` (selected by comparisons, so `s >= 1` and `r = 1 / s`
    /// lies in `(0, 1]`) and `u = r` where `|x| > 1`, else `u = x`, the
    /// derivative is `r / sqrt(1 + u^2)`: `1 / sqrt(1 + x^2)` for `|x| <= 1`
    /// and `(1 / |x|) / sqrt(1 + 1 / x^2)` above. Nothing overflows, every
    /// unselected branch is finite, and the value at `+-inf` is `0`; a NaN
    /// input gives NaN through `u`.
    fn arcsinh_derivative(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let one = self.scalar_constant(1.0);
        let minus_one = self.scalar_constant(-1.0);
        let zero = self.scalar_constant(0.0);
        let positive = self.compare(input, one, TensorComparison::Greater)?;
        let negative = self.compare(input, minus_one, TensorComparison::Less)?;
        let large = self.logical_or(positive, negative)?;
        let negated = self.sub(zero, input)?;
        let lower = self.where_select(negative, negated, one)?;
        let scale = self.where_select(positive, input, lower)?;
        let reciprocal = self.div(one, scale)?;
        let bounded = self.where_select(large, reciprocal, input)?;
        let squared = self.mul(bounded, bounded)?;
        let shifted = self.add(one, squared)?;
        let root = self.sqrt(shifted)?;
        self.div(reciprocal, root)
    }

    /// The symbolic `d fmod(x, y) / dy = -round((x - fmod(x, y)) / y)` of
    /// [`fmod_y_partial`], where `value` is the node `fmod(x, y)`.
    pub(super) fn fmod_y_partial(
        &mut self,
        x: TensorNodeId,
        y: TensorNodeId,
        value: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let multiple = self.sub(x, value)?;
        let quotient = self.div(multiple, y)?;
        let integer = self.unary_math(quotient, UnaryMathKind::Round)?;
        let zero = self.scalar_constant(0.0);
        self.sub(zero, integer)
    }
}
