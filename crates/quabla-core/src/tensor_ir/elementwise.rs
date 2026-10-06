//! The elementwise math functions: one [`UnaryMathKind`] per function of one
//! operand (the `UnaryMath` op) and the binary `Fmod` op. Everything that
//! defines one function lives here, so adding a function means adding a kind
//! and answering the questions its exhaustive matches ask:
//!
//! - its name in IR text and errors, and its `f64` CPU reference (a
//!   `float32` node rounds that result once);
//! - its derivatives: the symbolic rule (`TensorIr::unary_math_chain`),
//!   written with IR ops so that every derivative order is again an ordinary
//!   graph on every backend and shared by the symbolic JVP and VJP, and the
//!   numeric rules of the CPU evaluator's AD paths (`numeric_chain`,
//!   `numeric_mixed`);
//! - its CUDA spelling ([`UnaryMathKind::cuda_function`]) and which CUDA
//!   paths admit it ([`UnaryMathKind::cuda_fusable`],
//!   [`UnaryMathKind::cuda_loop_lowerable`]); the MLX lowering is the
//!   exhaustive `mlx_unary_math` of the MLX backend;
//! - its constant folding and its StableHLO spelling, if any.
//!
//! Values follow IEEE semantics and never raise: an input outside the domain
//! of a function (`arcsin(2)`, `log(-1)`, `arccosh(0.5)`) gives NaN, a pole
//! gives an infinity (`arctanh(1) == inf`, `log10(0) == -inf`), as in NumPy.

use super::{
    erf_derivative, erf_second_derivative, DynamicTensor, MixedTangent, TensorComparison, TensorIr,
    TensorNodeId, TensorOp,
};

/// An elementwise function of one operand, executed by the `UnaryMath` op.
/// The kinds share one op because they share every rule except their value
/// and derivative; each still lowers to the platform's own function (Rust
/// std or the `libm` crate on the CPU, the CUDA math library, and the MLX op
/// where MLX has one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryMathKind {
    /// `exp(x)`; derivative `exp(x)`, the result itself.
    Exp,
    /// The natural logarithm; derivative `1 / x`. `log(0) == -inf`, and a
    /// negative input gives NaN.
    Log,
    /// `ln(1 + x)`, accurate for small `|x|`; `-1` maps to `-inf` and
    /// `x < -1` to NaN. Derivative `1 / (1 + x)`.
    Log1p,
    /// `exp(x) - 1`, accurate for small `|x|`; derivative `exp(x)`.
    Expm1,
    /// The error function (the f64 musl `erf` of the `libm` crate on the
    /// CPU); derivative `2 / sqrt(pi) * exp(-x^2)`.
    Erf,
    /// The complementary error function `1 - erf(x)` (the f64 musl `erfc` of
    /// the `libm` crate on the CPU), which keeps full relative accuracy where
    /// `erf(x)` is close to one; derivative `-2 / sqrt(pi) * exp(-x^2)`.
    Erfc,
    /// `sin(x)`; derivative `cos(x)`.
    Sin,
    /// `cos(x)`; derivative `-sin(x)`.
    Cos,
    /// `tanh(x)`; derivative `1 - tanh(x)^2`, formed from the result.
    Tanh,
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

/// How the symbolic derivative of a [`UnaryMathKind`] scales an incoming
/// tangent or cotangent `s`; both the symbolic JVP and VJP apply it through
/// `TensorIr::apply_unary_chain`.
#[derive(Clone, Copy, Debug)]
pub(super) enum UnaryChain {
    /// The derivative is zero everywhere: no expression is built.
    Zero,
    /// `s * d`.
    Times(TensorNodeId),
    /// `0 - s * d`: `cos` negates the product with `sin(x)`, not the factor.
    NegatedTimes(TensorNodeId),
    /// `s / d`: `log` divides by `x` and `log1p` by `1 + x` instead of
    /// multiplying by a reciprocal.
    Over(TensorNodeId),
}

impl UnaryMathKind {
    pub const ALL: [Self; 24] = [
        Self::Exp,
        Self::Log,
        Self::Log1p,
        Self::Expm1,
        Self::Erf,
        Self::Erfc,
        Self::Sin,
        Self::Cos,
        Self::Tanh,
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

    /// The name of the op in IR text and error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Log1p => "log1p",
            Self::Expm1 => "expm1",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
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
    /// once. The error functions and the inverse hyperbolic functions use
    /// the musl implementations of the `libm` crate: Rust std has no `erf`,
    /// and its `acosh` is `ln(x + sqrt(x^2 - 1))`, which loses relative
    /// accuracy near `1`, where musl switches to `log1p`.
    pub(super) fn function(self) -> fn(f64) -> f64 {
        match self {
            Self::Exp => f64::exp,
            Self::Log => f64::ln,
            Self::Log1p => f64::ln_1p,
            Self::Expm1 => f64::exp_m1,
            Self::Erf => libm::erf,
            Self::Erfc => libm::erfc,
            Self::Sin => f64::sin,
            Self::Cos => f64::cos,
            Self::Tanh => f64::tanh,
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

    /// The value a plan folds a scalar constant operand `x` to, or `None` to
    /// keep the node. `log` folds only positive operands, so a non-positive
    /// constant logarithm stays in the plan as written; array constants fold
    /// with [`Self::function`] for every kind.
    pub(super) fn fold_scalar(self, x: f64) -> Option<f64> {
        match self {
            Self::Log => (x > 0.0).then(|| x.ln()),
            _ => Some(self.evaluate(x)),
        }
    }

    /// Whether the function is piecewise constant, so that every derivative
    /// order is zero (and no derivative expression is built).
    pub(super) fn is_piecewise_constant(self) -> bool {
        matches!(self, Self::Floor | Self::Ceil | Self::Round)
    }

    /// The first-derivative factor of the numeric (CPU evaluator) AD rules
    /// at input `x` with value `value`: the numeric counterpart of the
    /// symbolic rule [`TensorIr::unary_math_chain`] (same formula and the
    /// same conventions at poles, domain edges, and infinities). `cos` gives
    /// `sin(x)`, which [`Self::numeric_chain`] negates after the product.
    pub(super) fn derivative(self, x: f64, value: f64) -> f64 {
        match self {
            Self::Exp => value,
            Self::Log => 1.0 / x,
            Self::Log1p => 1.0 / (1.0 + x),
            Self::Expm1 => x.exp(),
            Self::Erf => erf_derivative(x),
            Self::Erfc => -erf_derivative(x),
            Self::Sin => x.cos(),
            Self::Cos => x.sin(),
            Self::Tanh => 1.0 - value * value,
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

    /// The second derivative at input `x` with value `value`, the derivative
    /// of the symbolic first derivative rule, for the kinds whose numeric
    /// second-order rule is `m d + (f s) d2` (see [`Self::numeric_mixed`];
    /// `exp`, `log`, `log1p`, `sin` and `cos` form theirs from the first
    /// derivative and the value instead and never call this).
    pub(super) fn second_derivative(self, x: f64, value: f64) -> f64 {
        match self {
            Self::Expm1 => x.exp(),
            Self::Erf => erf_second_derivative(x),
            Self::Erfc => -erf_second_derivative(x),
            Self::Tanh => -2.0 * value * (1.0 - value * value),
            // d/dx (1 + t^2) = 2 t (1 + t^2).
            Self::Tan => {
                let value = x.tan();
                2.0 * value * (1.0 + value * value)
            }
            // d/dx ((1 - x)(1 + x))^(-1/2) = x d^3.
            Self::Arcsin | Self::Arccos => {
                let derivative = self.derivative(x, value);
                x * derivative * derivative * derivative
            }
            // d/dx (1 + x^2)^(-1) = -2x d^2.
            Self::Arctan => {
                let derivative = self.derivative(x, value);
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
                let derivative = self.derivative(x, value);
                -(x * derivative) * derivative * derivative
            }
            // d/dx ((1 - x)(1 + x))^(-1) = 2x d^2.
            Self::Arctanh => {
                let derivative = self.derivative(x, value);
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
            Self::Exp
            | Self::Log
            | Self::Log1p
            | Self::Sin
            | Self::Cos
            | Self::Floor
            | Self::Ceil
            | Self::Round => 0.0,
        }
    }

    /// `seed` times the derivative at `input` (whose value is `value`), the
    /// first-order rule of the numeric AD paths (`TensorIr::value_and_vjp_many`
    /// and `TensorIr::jvp_many`); `None` where the derivative is zero
    /// everywhere. `cos` negates the product `seed * sin(x)`, which differs
    /// from multiplying by `-sin(x)` in the sign of a NaN.
    pub(super) fn numeric_chain(
        self,
        seed: &DynamicTensor,
        input: &DynamicTensor,
        value: &DynamicTensor,
    ) -> Result<Option<DynamicTensor>, String> {
        if self.is_piecewise_constant() {
            return Ok(None);
        }
        let derivative = input.elementwise(value, |x, value| self.derivative(x, value))?;
        let product = seed.mul(&derivative)?;
        Ok(Some(if self == Self::Cos {
            product.neg()?
        } else {
            product
        }))
    }

    /// The second-order forward rule of the mixed-dual evaluator: the value,
    /// both first-order tangents and the mixed tangent of `kind(x)` from
    /// those of `x` (`f`, `s`, `m` below), in `f64` without rounding. Most
    /// kinds use `m d + (f s) d2`; `exp`, `sin`, `cos`, `log` and `log1p`
    /// keep the forms they were introduced with, which round differently.
    pub(super) fn numeric_mixed(
        self,
        input: &MixedTangent,
        shape: &[usize],
    ) -> Result<MixedTangent, String> {
        let value = input.value.map_f64(self.function())?;
        if self.is_piecewise_constant() {
            let zero = DynamicTensor::filled(shape.to_vec(), 0.0)?;
            return Ok(MixedTangent {
                value,
                first: zero.clone(),
                second: zero.clone(),
                mixed: zero,
            });
        }
        let derivative = input
            .value
            .elementwise(&value, |x, value| self.derivative(x, value))?;
        let first = input.first.mul(&derivative)?;
        let second = input.second.mul(&derivative)?;
        let product = input.first.mul(&input.second)?;
        let (first, second, mixed) = match self {
            // (m + f s) e^x.
            Self::Exp => (first, second, input.mixed.add(&product)?.mul(&value)?),
            // m cos(x) - (f s) sin(x).
            Self::Sin => (
                first,
                second,
                input.mixed.mul(&derivative)?.sub(&product.mul(&value)?)?,
            ),
            // -(m sin(x)) - (f s) cos(x), every first-order term negated
            // after its product.
            Self::Cos => (
                first.neg()?,
                second.neg()?,
                input
                    .mixed
                    .mul(&derivative)?
                    .neg()?
                    .sub(&product.mul(&value)?)?,
            ),
            // m d - (f s) d^2 with d = 1 / x or 1 / (1 + x).
            Self::Log | Self::Log1p => (
                first,
                second,
                input
                    .mixed
                    .mul(&derivative)?
                    .sub(&product.mul(&derivative.mul(&derivative)?)?)?,
            ),
            _ => {
                let second_derivative = input
                    .value
                    .elementwise(&value, |x, value| self.second_derivative(x, value))?;
                (
                    first,
                    second,
                    input
                        .mixed
                        .mul(&derivative)?
                        .add(&product.mul(&second_derivative)?)?,
                )
            }
        };
        Ok(MixedTangent {
            value,
            first,
            second,
            mixed,
        })
    }

    /// Bounds on `|f'|` and `|f''|` over the inputs of a node whose largest
    /// value magnitude is `magnitude`, for the kinds the finite symbolic
    /// second-order route admits (`TensorIr::supports_finite_symbolic_second_order`);
    /// `None` keeps every other kind on the mixed-dual route.
    pub(super) fn second_order_bounds(self, magnitude: f64) -> Option<(f64, f64)> {
        match self {
            Self::Sin | Self::Cos => Some((1.0, 1.0)),
            Self::Tanh => Some((1.0, 2.0)),
            Self::Exp => Some((magnitude, magnitude)),
            _ => None,
        }
    }

    /// Whether the forward rule emits the derivative expression before the
    /// value node. Only the node order of a JVP graph depends on it; it is
    /// kept per kind so that lowered JVP programs keep their numbering.
    pub(super) fn jvp_emits_derivative_first(self) -> bool {
        matches!(self, Self::Log1p | Self::Expm1 | Self::Erf | Self::Erfc)
    }

    /// The single-precision CUDA math function; `double_precision_source`
    /// rewrites each to its double overload (`tanf` -> `tan`) for
    /// `precision="float64"`. `rintf` rounds halfway cases to even in the
    /// default rounding mode, whereas `roundf` rounds them away from zero.
    pub(super) fn cuda_function(self) -> &'static str {
        match self {
            Self::Exp => "expf",
            Self::Log => "logf",
            Self::Log1p => "log1pf",
            Self::Expm1 => "expm1f",
            Self::Erf => "erff",
            Self::Erfc => "erfcf",
            Self::Sin => "sinf",
            Self::Cos => "cosf",
            Self::Tanh => "tanhf",
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

    /// Whether whole-plan and region fusion inline the kind into one CUDA
    /// elementwise kernel (`is_fusable_elementwise_compute_op`). `log`,
    /// `log1p`, `expm1`, `erf` and `erfc` are not admitted yet: each runs as
    /// its own per-node kernel, and admitting one changes which neighbouring
    /// operations share a kernel (and so may contract into an FMA).
    pub(super) fn cuda_fusable(self) -> bool {
        !matches!(
            self,
            Self::Log | Self::Log1p | Self::Expm1 | Self::Erf | Self::Erfc
        )
    }

    /// Whether a fused CUDA `fori`/`scan` body may contain the kind
    /// (`cuda_fori_body_is_lowerable`, `cuda_scan_body_is_lowerable`).
    /// `erfc` is not admitted yet: a loop whose body uses it runs as a
    /// host-driven region loop of per-node kernels instead.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn cuda_loop_lowerable(self) -> bool {
        self != Self::Erfc
    }

    /// The StableHLO operation of the kind, for those the export verifies.
    pub(super) fn stablehlo_name(self) -> Option<&'static str> {
        match self {
            Self::Tanh => Some("stablehlo.tanh"),
            _ => None,
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

    /// The symbolic derivative of `kind` at `input` as the scaling it applies
    /// to an incoming tangent or cotangent (see [`UnaryChain`]), shared by
    /// the symbolic JVP and VJP. `value` is the node `kind(input)`; only the
    /// kinds whose derivative is formed from the result (`exp`, `tanh`,
    /// `tan`, `cbrt`) need it, and the forward rule of the kinds that emit
    /// their derivative first ([`UnaryMathKind::jvp_emits_derivative_first`])
    /// passes `None`.
    pub(super) fn unary_math_chain(
        &mut self,
        kind: UnaryMathKind,
        input: TensorNodeId,
        value: Option<TensorNodeId>,
    ) -> Result<UnaryChain, String> {
        let value = || {
            value.ok_or_else(|| format!("the {} derivative is formed from its value", kind.name()))
        };
        Ok(match kind {
            UnaryMathKind::Exp => UnaryChain::Times(value()?),
            UnaryMathKind::Tanh => {
                let value = value()?;
                let one = self.scalar_constant(1.0);
                let squared = self.mul(value, value)?;
                UnaryChain::Times(self.sub(one, squared)?)
            }
            UnaryMathKind::Sin => UnaryChain::Times(self.unary_math(input, UnaryMathKind::Cos)?),
            UnaryMathKind::Cos => {
                UnaryChain::NegatedTimes(self.unary_math(input, UnaryMathKind::Sin)?)
            }
            UnaryMathKind::Log => UnaryChain::Over(input),
            UnaryMathKind::Log1p => {
                let one = self.scalar_constant(1.0);
                UnaryChain::Over(self.add(one, input)?)
            }
            UnaryMathKind::Expm1 => UnaryChain::Times(self.unary_math(input, UnaryMathKind::Exp)?),
            UnaryMathKind::Erf => UnaryChain::Times(self.erf_derivative(input, 1.0)?),
            UnaryMathKind::Erfc => UnaryChain::Times(self.erf_derivative(input, -1.0)?),
            _ => match self.unary_math_derivative(kind, input, value()?)? {
                Some(derivative) => UnaryChain::Times(derivative),
                None => UnaryChain::Zero,
            },
        })
    }

    /// `seed` scaled by `chain`; `None` for a zero derivative.
    pub(super) fn apply_unary_chain(
        &mut self,
        chain: UnaryChain,
        seed: TensorNodeId,
    ) -> Result<Option<TensorNodeId>, String> {
        Ok(match chain {
            UnaryChain::Zero => None,
            UnaryChain::Times(derivative) => Some(self.mul(seed, derivative)?),
            UnaryChain::NegatedTimes(derivative) => {
                let product = self.mul(seed, derivative)?;
                let zero = self.scalar_constant(0.0);
                Some(self.sub(zero, product)?)
            }
            UnaryChain::Over(divisor) => Some(self.div(seed, divisor)?),
        })
    }

    /// The forward rule of `kind`: the value `kind(input)` and its tangent
    /// for the input tangent `tangent`, `None` where the derivative is zero.
    pub(super) fn unary_math_jvp(
        &mut self,
        kind: UnaryMathKind,
        input: TensorNodeId,
        tangent: TensorNodeId,
    ) -> Result<(TensorNodeId, Option<TensorNodeId>), String> {
        let (value, chain) = if kind.jvp_emits_derivative_first() {
            let chain = self.unary_math_chain(kind, input, None)?;
            (self.unary_math(input, kind)?, chain)
        } else {
            let value = self.unary_math(input, kind)?;
            (value, self.unary_math_chain(kind, input, Some(value))?)
        };
        Ok((value, self.apply_unary_chain(chain, tangent)?))
    }

    /// Symbolic `sign * 2 / sqrt(pi) * exp(-(x * x))`: `erf'` for `sign = 1`
    /// and `erfc'` for `sign = -1`.
    fn erf_derivative(&mut self, input: TensorNodeId, sign: f64) -> Result<TensorNodeId, String> {
        let squared = self.mul(input, input)?;
        let zero = self.scalar_constant(0.0);
        let negated = self.sub(zero, squared)?;
        let decay = self.unary_math(negated, UnaryMathKind::Exp)?;
        let scale = self.scalar_constant(sign * std::f64::consts::FRAC_2_SQRT_PI);
        self.mul(scale, decay)
    }

    /// The symbolic first derivative of the inverse trigonometric and
    /// hyperbolic kinds, `tan`, `log2`, `log10`, `cbrt` and the piecewise
    /// constant kinds at `input`, whose value `kind(input)` is the node
    /// `value`; `None` for the piecewise-constant kinds, whose derivative is
    /// zero.
    fn unary_math_derivative(
        &mut self,
        kind: UnaryMathKind,
        input: TensorNodeId,
        value: TensorNodeId,
    ) -> Result<Option<TensorNodeId>, String> {
        let one = self.scalar_constant(1.0);
        let derivative = match kind {
            UnaryMathKind::Floor | UnaryMathKind::Ceil | UnaryMathKind::Round => return Ok(None),
            UnaryMathKind::Exp
            | UnaryMathKind::Log
            | UnaryMathKind::Log1p
            | UnaryMathKind::Expm1
            | UnaryMathKind::Erf
            | UnaryMathKind::Erfc
            | UnaryMathKind::Sin
            | UnaryMathKind::Cos
            | UnaryMathKind::Tanh => {
                return Err(format!(
                    "the {} derivative is built by unary_math_chain",
                    kind.name()
                ))
            }
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
