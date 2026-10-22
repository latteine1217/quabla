//! The elementwise math functions: one [`UnaryMathKind`] per function of one
//! operand (the `UnaryMath` op) and one [`BinaryMathKind`] per function of two
//! (the `BinaryMath` op). Everything that defines one function lives here, so
//! adding a function means adding a kind and answering the questions its
//! exhaustive matches ask:
//!
//! - its name in IR text and errors, and its `f64` CPU reference (a
//!   `float32` node rounds that result once);
//! - its derivatives: the symbolic rules (`TensorIr::unary_math_chain`,
//!   `TensorIr::binary_math_jvp`, `TensorIr::binary_math_vjp`), written with
//!   IR ops so that every derivative order is again an ordinary graph on
//!   every backend, and the numeric rules of the CPU evaluator's AD paths
//!   (`numeric_chain`, `numeric_partial`, `numeric_mixed`);
//! - its CUDA spelling (`cuda_function`) and which CUDA paths admit it
//!   (`cuda_fusable`, `cuda_loop_lowerable`); the MLX lowering is the
//!   exhaustive `mlx_unary_math` / `mlx_binary_math` of the MLX backend;
//! - its constant folding and its StableHLO spelling, if any.
//!
//! Values follow IEEE semantics and never raise: an input outside the domain
//! of a function (`arcsin(2)`, `log(-1)`, `arccosh(0.5)`) gives NaN, a pole
//! gives an infinity (`arctanh(1) == inf`, `log10(0) == -inf`), as in NumPy.

use super::{TensorComparison, TensorDType, TensorIr, TensorNodeId, TensorOp};

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

    /// Whether the forward rule emits the derivative expression before the
    /// value node. Only the node order of a JVP graph depends on it; it is
    /// kept per kind so that lowered JVP programs keep their numbering.
    pub(super) fn jvp_emits_derivative_first(self) -> bool {
        matches!(self, Self::Log1p | Self::Expm1 | Self::Erf | Self::Erfc)
    }

    /// The single-precision CUDA math function; `double_precision_source`
    /// rewrites each to its double overload (`tanf` -> `tan`) for `float64`
    /// programs. `rintf` rounds halfway cases to even in the
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

/// C `fmod(x, y)`: `x - n y` with `n = trunc(x / y)` evaluated exactly, so
/// the result has the sign of `x` and is exact. Rust's `%` on `f64` is fmod.
pub(super) fn fmod(x: f64, y: f64) -> f64 {
    x % y
}

/// An elementwise function of two operands, executed by the `BinaryMath`
/// op. Operands broadcast and promote like arithmetic operands, and `Bool`
/// operands are rejected. As for [`UnaryMathKind`], every rule of a kind
/// lives here: its value, partials, CUDA spelling and admission, folding and
/// StableHLO spelling; the MLX lowering is the exhaustive `mlx_binary_math`.
///
/// `add`, `sub`, `mul` and `div` stay ops of their own: their derivatives
/// are linear and structural (no partial expression), CUDA spells them as
/// operators rather than functions, and the plan rewrites (matmul epilogues,
/// constant folding of zero divisors, shape reduction of broadcast
/// cotangents) match them directly. `greater` and `compare` have a `Bool`
/// result and no derivative, and `where` selects rather than computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryMathKind {
    /// `lhs ** rhs` with `f64::powf` semantics: NaN for a negative base with
    /// a non-integer exponent, `0 ** 0 == 1`, and IEEE results for
    /// infinities and NaN. The partials follow the conventions of
    /// [`pow_base_derivative`] and [`pow_exponent_derivative`] at `x <= 0`.
    Pow,
    /// The four-quadrant `atan2(lhs, rhs)` (`lhs` is `y`) with `f64::atan2`
    /// semantics. The partials `x / (x^2 + y^2)` and `-y / (x^2 + y^2)` are
    /// formed after scaling by `|x| + |y|`, so they neither overflow nor
    /// underflow, and are defined as `0` where `|x| + |y|` is `0` or
    /// infinite; see [`atan2_derivatives`].
    Atan2,
    /// C `fmod(lhs, rhs)`: the result has the sign of `lhs`. Its partials are
    /// `1` and `-trunc(x / y)` (see [`fmod_y_partial`]); both are piecewise
    /// constant, so every second partial is zero.
    Fmod,
}

impl BinaryMathKind {
    pub const ALL: [Self; 3] = [Self::Pow, Self::Atan2, Self::Fmod];

    /// The name of the op in IR text and error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pow => "pow",
            Self::Atan2 => "atan2",
            Self::Fmod => "fmod",
        }
    }

    /// The `f64` function, the CPU reference that a `float32` node rounds
    /// once.
    pub(super) fn function(self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Pow => f64::powf,
            Self::Atan2 => f64::atan2,
            Self::Fmod => fmod,
        }
    }

    /// The `f64` value of [`Self::function`] at `(lhs, rhs)`.
    pub fn evaluate(self, lhs: f64, rhs: f64) -> f64 {
        self.function()(lhs, rhs)
    }

    /// The single-precision CUDA math function; `double_precision_source`
    /// rewrites each to its double overload for `float64` programs.
    pub(super) fn cuda_function(self) -> &'static str {
        match self {
            Self::Pow => "powf",
            Self::Atan2 => "atan2f",
            Self::Fmod => "fmodf",
        }
    }

    /// Whether whole-plan and region fusion inline the kind into one CUDA
    /// elementwise kernel. `atan2` is not admitted yet and runs as its own
    /// per-node kernel (see [`UnaryMathKind::cuda_fusable`]).
    pub(super) fn cuda_fusable(self) -> bool {
        self != Self::Atan2
    }

    /// Whether a fused CUDA `fori`/`scan` body may contain the kind; every
    /// binary kind is admitted.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn cuda_loop_lowerable(self) -> bool {
        true
    }

    /// The StableHLO operation of the kind, for those the export verifies.
    pub(super) fn stablehlo_name(self) -> Option<&'static str> {
        match self {
            Self::Pow => Some("stablehlo.power"),
            _ => None,
        }
    }
}

fn pow_base_is_singular(base: f64, exponent: f64) -> bool {
    base == 0.0 && exponent < 1.0
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
}

impl TensorIr {
    /// Elementwise `kind(lhs, rhs)`; see [`BinaryMathKind`]. Operands
    /// broadcast and promote like arithmetic operands, and `Bool` operands
    /// are rejected.
    pub fn binary_math(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
        kind: BinaryMathKind,
    ) -> Result<TensorNodeId, String> {
        for operand in [lhs, rhs] {
            if self.node(operand)?.dtype == TensorDType::Bool {
                return Err(format!(
                    "{} is not defined for bool tensors; use logical_and/logical_or/logical_not \
                     (& | ~) or convert explicitly with astype",
                    kind.name()
                ));
            }
        }
        self.binary(kind.name(), lhs, rhs, |lhs, rhs| TensorOp::BinaryMath {
            lhs,
            rhs,
            kind,
        })
    }

    /// Elementwise `base ** exponent` with `f64::powf` semantics: NaN for a
    /// negative base with a non-integer exponent, `0 ** 0 == 1`, and IEEE
    /// results for infinities and NaN. Operands broadcast and promote like
    /// arithmetic operands, so a weak scalar exponent adopts the base dtype.
    /// `Bool` operands are rejected, as by the unary math ops, instead of
    /// being promoted to 0/1.
    pub fn pow(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        self.binary_math(base, exponent, BinaryMathKind::Pow)
    }

    /// Elementwise `atan2(y, x)`; operands broadcast and promote like
    /// arithmetic operands, and `Bool` operands are rejected. See
    /// [`BinaryMathKind::Atan2`].
    pub fn atan2(&mut self, y: TensorNodeId, x: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary_math(y, x, BinaryMathKind::Atan2)
    }

    /// Elementwise C `fmod(x, y)`, the remainder of the quotient truncated
    /// toward zero, with the sign of `x` (NaN for `y == 0` or an infinite
    /// `x`, `x` for an infinite `y`). Operands broadcast and promote like
    /// arithmetic operands, and `Bool` operands are rejected. The partials
    /// are `1` and `-trunc(x / y)` (see [`fmod_y_partial`]); both are
    /// piecewise constant, so every second partial is zero.
    pub fn fmod(&mut self, x: TensorNodeId, y: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary_math(x, y, BinaryMathKind::Fmod)
    }

    /// The forward rule of `kind`: the value `kind(lhs, rhs)` and its
    /// tangent for the operand tangents `lhs_tangent` and `rhs_tangent`.
    pub(super) fn binary_math_jvp(
        &mut self,
        kind: BinaryMathKind,
        (lhs, lhs_tangent): (TensorNodeId, TensorNodeId),
        (rhs, rhs_tangent): (TensorNodeId, TensorNodeId),
    ) -> Result<(TensorNodeId, TensorNodeId), String> {
        match kind {
            BinaryMathKind::Atan2 => {
                let (y_partial, x_partial) = self.atan2_partials(lhs, rhs)?;
                let y_term = self.mul(lhs_tangent, y_partial)?;
                let x_term = self.mul(rhs_tangent, x_partial)?;
                Ok((self.binary_math(lhs, rhs, kind)?, self.add(y_term, x_term)?))
            }
            // d fmod = dx - trunc(x / y) dy; the x tangent passes through
            // unscaled (broadcast by the add).
            BinaryMathKind::Fmod => {
                let value = self.binary_math(lhs, rhs, kind)?;
                let y_partial = self.fmod_y_partial(lhs, rhs, value)?;
                let y_term = self.mul(rhs_tangent, y_partial)?;
                Ok((value, self.add(lhs_tangent, y_term)?))
            }
            BinaryMathKind::Pow => {
                let value = self.binary_math(lhs, rhs, kind)?;
                Ok((value, self.pow_tangent(lhs, lhs_tangent, rhs, rhs_tangent)?))
            }
        }
    }

    /// The reverse rule of `kind` at `value = kind(lhs, rhs)`: calls
    /// `emit(graph, index, contribution)` with the cotangent contribution of
    /// operand `index` (before it is reduced to the operand shape), for each
    /// operand that is `differentiable`. A constant operand receives no
    /// cotangent, so its partial is not emitted (`atan2` still forms both of
    /// its partials, which share their scaling).
    pub(super) fn binary_math_vjp(
        &mut self,
        kind: BinaryMathKind,
        (lhs, rhs, value): (TensorNodeId, TensorNodeId, TensorNodeId),
        upstream: TensorNodeId,
        differentiable: [bool; 2],
        mut emit: impl FnMut(&mut Self, usize, TensorNodeId) -> Result<(), String>,
    ) -> Result<(), String> {
        match kind {
            BinaryMathKind::Atan2 => {
                let (y_partial, x_partial) = self.atan2_partials(lhs, rhs)?;
                for (index, partial) in [y_partial, x_partial].into_iter().enumerate() {
                    if differentiable[index] {
                        let contribution = self.mul(upstream, partial)?;
                        emit(self, index, contribution)?;
                    }
                }
            }
            // `x` receives the cotangent itself (partial 1) and `y` its
            // product with `-trunc(x / y)`.
            BinaryMathKind::Fmod => {
                if differentiable[0] {
                    emit(self, 0, upstream)?;
                }
                if differentiable[1] {
                    let partial = self.fmod_y_partial(lhs, rhs, value)?;
                    let contribution = self.mul(upstream, partial)?;
                    emit(self, 1, contribution)?;
                }
            }
            BinaryMathKind::Pow => {
                for (index, differentiable) in differentiable.into_iter().enumerate() {
                    if !differentiable {
                        continue;
                    }
                    let derivative = if index == 0 {
                        self.pow_base_derivative(lhs, rhs)?
                    } else {
                        self.pow_exponent_derivative(lhs, rhs)?
                    };
                    let contribution = self.mul(upstream, derivative)?;
                    emit(self, index, contribution)?;
                }
            }
        }
        Ok(())
    }

    /// `exponent - 1`, folded to a constant of the same dtype and weakness
    /// when the exponent is a constant, so nested derivatives of a constant
    /// exponent keep recognizing it.
    fn pow_lowered_exponent(&mut self, exponent: TensorNodeId) -> Result<TensorNodeId, String> {
        if let Some(value) = self.scalar_constant_value(exponent) {
            let node = self.node(exponent)?;
            let (dtype, weak) = (node.dtype, node.weak);
            return Ok(self.constant_like(dtype.round(value - 1.0), dtype, weak));
        }
        let one = self.scalar_constant(1.0);
        self.sub(exponent, one)
    }

    /// Symbolic `d pow(x, y) / dx` with the conventions of the free function
    /// `pow_base_derivative`: `where(m, 0, y * pow(where(m, 1, x), y - 1))`
    /// with `m = (x == 0) & (y < 1)`. Masking the inner base keeps `0 * inf`
    /// out of the reverse pass through this expression, so Hessians stay
    /// finite at the origin. A constant exponent decides `y < 1` statically.
    fn pow_base_derivative(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let zero = self.scalar_constant(0.0);
        let one = self.scalar_constant(1.0);
        let singular = match self.scalar_constant_value(exponent) {
            // No point is singular unless `y < 1` (a NaN exponent is never singular).
            Some(value) if !pow_base_is_singular(0.0, value) => None,
            Some(_) => Some(self.compare(base, zero, TensorComparison::Equal)?),
            None => {
                let at_origin = self.compare(base, zero, TensorComparison::Equal)?;
                let below_one = self.compare(exponent, one, TensorComparison::Less)?;
                Some(self.logical_and(at_origin, below_one)?)
            }
        };
        let safe_base = match singular {
            Some(mask) => self.where_select(mask, one, base)?,
            None => base,
        };
        let lowered = self.pow_lowered_exponent(exponent)?;
        let power = self.pow(safe_base, lowered)?;
        let derivative = self.mul(exponent, power)?;
        match singular {
            Some(mask) => self.where_select(mask, zero, derivative),
            None => Ok(derivative),
        }
    }

    /// Symbolic `d pow(x, y) / dy` with the conventions of the free function
    /// `pow_exponent_derivative`: `pow(s, y) * log(s)` with
    /// `s = where(x <= 0, 1, x)`, which is exactly `0` for `x <= 0` and keeps
    /// the checked-domain `log` away from non-positive values.
    fn pow_exponent_derivative(
        &mut self,
        base: TensorNodeId,
        exponent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        if let Some(value) = self.scalar_constant_value(base) {
            if value <= 0.0 {
                return Ok(self.scalar_constant(0.0));
            }
            // A positive constant base reuses the primal `pow(base, y)` (CSE)
            // and folds its logarithm.
            let node = self.node(base)?;
            let (dtype, weak) = (node.dtype, node.weak);
            let log = self.constant_like(dtype.round(value.ln()), dtype, weak);
            let power = self.pow(base, exponent)?;
            return self.mul(power, log);
        }
        let zero = self.scalar_constant(0.0);
        let one = self.scalar_constant(1.0);
        let non_positive = self.compare(base, zero, TensorComparison::LessEqual)?;
        let safe_base = self.where_select(non_positive, one, base)?;
        let power = self.pow(safe_base, exponent)?;
        let log = self.log(safe_base)?;
        self.mul(power, log)
    }

    /// Tangent of `pow(x, y)`; the term of an operand whose tangent is the
    /// constant zero (a constant operand) is left out rather than multiplied
    /// by a derivative that may be non-finite.
    fn pow_tangent(
        &mut self,
        base: TensorNodeId,
        base_tangent: TensorNodeId,
        exponent: TensorNodeId,
        exponent_tangent: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let mut tangent = None;
        if self.scalar_constant_value(base_tangent) != Some(0.0) {
            let derivative = self.pow_base_derivative(base, exponent)?;
            tangent = Some(self.mul(base_tangent, derivative)?);
        }
        if self.scalar_constant_value(exponent_tangent) != Some(0.0) {
            let derivative = self.pow_exponent_derivative(base, exponent)?;
            let term = self.mul(exponent_tangent, derivative)?;
            tangent = Some(match tangent {
                Some(tangent) => self.add(tangent, term)?,
                None => term,
            });
        }
        Ok(match tangent {
            Some(tangent) => tangent,
            None => self.scalar_constant(0.0),
        })
    }

    /// Symbolic `(d atan2(y, x) / dy, d atan2(y, x) / dx)` with the scaling
    /// and conventions of the free function `atan2_derivatives`. Where
    /// `s = |x| + |y|` is `0` or infinite, the operands, `s` and `q` are
    /// replaced by the constants `0`, `1` and `1`, so both partials are
    /// exactly `0` there, every unselected branch stays finite, and the
    /// derivatives of the partials are `0` as well.
    fn atan2_partials(
        &mut self,
        y: TensorNodeId,
        x: TensorNodeId,
    ) -> Result<(TensorNodeId, TensorNodeId), String> {
        let zero = self.scalar_constant(0.0);
        let one = self.scalar_constant(1.0);
        let magnitude = |graph: &mut Self, value: TensorNodeId| {
            let negative = graph.compare(value, zero, TensorComparison::Less)?;
            let negated = graph.sub(zero, value)?;
            graph.where_select(negative, negated, value)
        };
        let x_magnitude = magnitude(self, x)?;
        let y_magnitude = magnitude(self, y)?;
        let scale = self.add(x_magnitude, y_magnitude)?;
        // `s == inf` as "not finite and not NaN", without an infinite
        // constant, which device programs cannot bind.
        let at_origin = self.compare(scale, zero, TensorComparison::Equal)?;
        let finite = self.isfinite(scale)?;
        let infinite_or_nan = self.logical_not(finite)?;
        let number = self.compare(scale, scale, TensorComparison::Equal)?;
        let unbounded = self.logical_and(infinite_or_nan, number)?;
        let degenerate = self.logical_or(at_origin, unbounded)?;
        let safe_scale = self.where_select(degenerate, one, scale)?;
        let safe_x = self.where_select(degenerate, zero, x)?;
        let safe_y = self.where_select(degenerate, zero, y)?;
        let scaled_x = self.div(safe_x, safe_scale)?;
        let scaled_y = self.div(safe_y, safe_scale)?;
        let x_squared = self.mul(scaled_x, scaled_x)?;
        let y_squared = self.mul(scaled_y, scaled_y)?;
        let q = self.add(x_squared, y_squared)?;
        let safe_q = self.where_select(degenerate, one, q)?;
        let y_partial = self.div(scaled_x, safe_q)?;
        let y_partial = self.div(y_partial, safe_scale)?;
        let negated_y = self.sub(zero, scaled_y)?;
        let x_partial = self.div(negated_y, safe_q)?;
        let x_partial = self.div(x_partial, safe_scale)?;
        Ok((y_partial, x_partial))
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
