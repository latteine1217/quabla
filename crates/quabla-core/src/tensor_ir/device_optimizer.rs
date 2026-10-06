//! Hyperparameters of the fused device optimizer steps.
//!
//! The CUDA and MLX training plans apply one of these rules to every retained
//! parameter after each value-and-gradient execution. The definitions are
//! those of the host optimizers in `quabla.optim` (`SGD`, `Adam`, `AdamW`, and
//! `clip_by_global_norm`), so a device trainer and the CPU trainer agree up to
//! rounding.
//!
//! The element update rules themselves live here once, as [`adam_element`]
//! and [`sgd_element`] over an [`AdamArith`]: the eager host update, the
//! v0.1 host optimizer, the CUDA kernels (generated as C source), and the
//! MLX graph all evaluate them through their own arithmetic adapter.

use std::convert::Infallible;

/// The update rule of one fused device optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeviceUpdateRule {
    /// `p - g * learning_rate`.
    Sgd,
    /// Adam with bias correction:
    /// `p - ((m / c1) / (sqrt(v / c2) + epsilon)) * learning_rate`, where
    /// `c1 = 1 - beta1^t` and `c2 = 1 - beta2^t` at update count `t`. A
    /// nonzero `weight_decay` gives AdamW's decoupled decay: the direction
    /// gains `weight_decay * p` with the pre-update `p`, so the schedule's
    /// learning rate scales the decay too. Zero keeps the Adam expression
    /// exactly, as `0 * inf` would otherwise introduce a NaN.
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    },
}

/// One fused device optimizer configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceOptimizerConfig {
    pub rule: DeviceUpdateRule,
    /// Rate of the next step; host schedules replace it between steps.
    pub learning_rate: f64,
    /// Global-norm bound of `clip_by_global_norm`, applied to all gradients
    /// together before the update.
    pub clip_norm: Option<f64>,
}

impl DeviceOptimizerConfig {
    /// Plain Adam without decay or clipping, the v0.1 device optimizer.
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            rule: DeviceUpdateRule::Adam {
                beta1,
                beta2,
                epsilon,
                weight_decay: 0.0,
            },
            learning_rate,
            clip_norm: None,
        }
    }

    /// Checks the ranges the host optimizers enforce. A zero learning rate is
    /// valid: schedules such as warmup and cosine decay start or end at zero,
    /// and a zero Adam step still advances the moments.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.learning_rate.is_finite() && self.learning_rate >= 0.0) {
            return Err("device optimizer learning_rate must be finite and nonnegative".into());
        }
        if let Some(clip_norm) = self.clip_norm {
            if !(clip_norm.is_finite() && clip_norm > 0.0) {
                return Err("device optimizer clip_norm must be positive and finite".into());
            }
        }
        if let DeviceUpdateRule::Adam {
            beta1,
            beta2,
            epsilon,
            weight_decay,
        } = self.rule
        {
            if !((0.0..1.0).contains(&beta1) && (0.0..1.0).contains(&beta2)) {
                return Err("device Adam beta1 and beta2 must be in [0, 1)".into());
            }
            if !(epsilon.is_finite() && epsilon > 0.0) {
                return Err("device Adam epsilon must be positive and finite".into());
            }
            if !(weight_decay.is_finite() && weight_decay >= 0.0) {
                return Err("device AdamW weight_decay must be finite and nonnegative".into());
            }
        }
        Ok(())
    }

    /// Bias corrections `(1 - beta1^t, 1 - beta2^t)` of update `t`, computed in
    /// float64 with `powf` exactly as the host optimizer's `b ** t`; device
    /// float32 plans round them once.
    pub fn adam_corrections(beta1: f64, beta2: f64, step: u64) -> (f64, f64) {
        // `step` counts updates; it is far below 2^53, so the conversion is exact.
        let step = step as f64;
        (1.0 - beta1.powf(step), 1.0 - beta2.powf(step))
    }
}

/// Operation order of the Adam element rule.
///
/// Both orders compute the same mathematical update and differ only in how
/// the products associate, so their results differ at the ulp level. The
/// modern entry points (`optim.Adam.update`/`AdamW.update`, which the CPU
/// `Trainer` calls, and the fused CUDA and MLX steps behind every device
/// `Trainer` and `cuda_adam_vjp_optimizer`, `cuda_adam_loss_optimizer`, and
/// `mlx_adam_loss_optimizer`) evaluate [`AdamOrder::Canonical`]. The v0.1
/// entry points `_quabla.Adam.step` (and `optim.Adam.step`, which calls it)
/// and the legacy CUDA `quabla_adam` kernel (`adam_step`, `cuda_adam_step`,
/// `cuda_adam_optimizer`) evaluate [`AdamOrder::V01`]: the 0.x compatibility
/// promise keeps their v0.1 results bit for bit, so they switch to the
/// canonical order only at v1.0 (item A1 of `docs/jax_like_roadmap.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdamOrder {
    /// `m' = m * b1 + g * (1 - b1)`, `v' = v * b2 + (g * g) * (1 - b2)`,
    /// `d = (m' / c1) / (sqrt(v' / c2) + eps)`, plus `weight_decay * p` when
    /// the decay is nonzero, and `p' = p - d * lr`. `1 - beta` is the
    /// coefficient formed in float64 (see [`AdamCoefficients`]).
    Canonical,
    /// `m' = b1 * m + (1 - b1) * g`, `v' = b2 * v + ((1 - b2) * g) * g`, and
    /// `p' = p - (lr * (m' / c1)) / (sqrt(v' / c2) + eps)`, with `1 - beta`
    /// formed in the element arithmetic and no weight decay, as v0.1 wrote it.
    V01,
}

/// The scalars of one Adam update, in the representation of an
/// [`AdamArith`]: numbers for host arithmetic, kernel parameter names for
/// the CUDA source, device scalars for MLX.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamCoefficients<V> {
    pub learning_rate: V,
    pub beta1: V,
    pub beta2: V,
    /// `1 - beta1` and `1 - beta2` formed in float64 from the float64 betas
    /// and rounded once: in float32, `1 - beta2` of the rounded 0.999 is
    /// 1.3e-5 relative off. [`AdamOrder::V01`] does not read them.
    pub one_minus_beta1: V,
    pub one_minus_beta2: V,
    pub epsilon: V,
    /// Bias corrections `1 - beta^t` of update `t`.
    pub correction1: V,
    pub correction2: V,
    /// AdamW's decoupled decay; zero is plain Adam. [`AdamOrder::V01`] does
    /// not read it.
    pub weight_decay: V,
}

impl AdamCoefficients<f64> {
    /// The canonical coefficients of update `step` (counting from one), with
    /// the float64 `powf` bias corrections of
    /// [`DeviceOptimizerConfig::adam_corrections`].
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
        step: u64,
    ) -> Self {
        let (correction1, correction2) =
            DeviceOptimizerConfig::adam_corrections(beta1, beta2, step);
        Self::with_corrections(
            learning_rate,
            beta1,
            beta2,
            epsilon,
            correction1,
            correction2,
            weight_decay,
        )
    }

    /// Coefficients with bias corrections computed by the caller, as
    /// `optim.Adam.update` passes Python's `1 - b ** t`.
    pub fn with_corrections(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        correction1: f64,
        correction2: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            one_minus_beta1: 1.0 - beta1,
            one_minus_beta2: 1.0 - beta2,
            epsilon,
            correction1,
            correction2,
            weight_decay,
        }
    }

    /// The coefficients of `_quabla.Adam.step` at update `step`: float64 bias
    /// corrections by `powi`, as v0.1 computed them, and no weight decay.
    pub fn v01(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, step: u64) -> Self {
        // v0.1 converted the count with `as i32`; kept for its exact results.
        let exponent = step as i32;
        Self::with_corrections(
            learning_rate,
            beta1,
            beta2,
            epsilon,
            1.0 - beta1.powi(exponent),
            1.0 - beta2.powi(exponent),
            0.0,
        )
    }

    /// The bias corrections of the legacy CUDA `quabla_adam` kernel: v0.1
    /// formed them in float32 from the float32 betas with `powf`.
    pub fn v01_float32_corrections(beta1: f32, beta2: f32, step: u64) -> (f32, f32) {
        let step = step as f32;
        (1.0 - beta1.powf(step), 1.0 - beta2.powf(step))
    }

    /// Converts every coefficient, for example rounds them to float32.
    pub fn map<W>(self, mut convert: impl FnMut(f64) -> W) -> AdamCoefficients<W> {
        AdamCoefficients {
            learning_rate: convert(self.learning_rate),
            beta1: convert(self.beta1),
            beta2: convert(self.beta2),
            one_minus_beta1: convert(self.one_minus_beta1),
            one_minus_beta2: convert(self.one_minus_beta2),
            epsilon: convert(self.epsilon),
            correction1: convert(self.correction1),
            correction2: convert(self.correction2),
            weight_decay: convert(self.weight_decay),
        }
    }
}

/// The arithmetic one backend evaluates the optimizer rules in.
///
/// [`adam_element`] and [`sgd_element`] write each rule once as a sequence
/// of these operations; an adapter evaluates them in float64 on the host
/// ([`F64Arith`]), emits them as CUDA C (`CudaSourceArith`), or builds them
/// as lazy MLX array operations. Every operation rounds once in the
/// adapter's element type, so two adapters of the same type agree bit for
/// bit when the rule is the same.
pub trait AdamArith {
    type Value;
    type Error;

    fn constant(&mut self, value: f64) -> Result<Self::Value, Self::Error>;
    fn add(&mut self, lhs: &Self::Value, rhs: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn sub(&mut self, lhs: &Self::Value, rhs: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn mul(&mut self, lhs: &Self::Value, rhs: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn div(&mut self, lhs: &Self::Value, rhs: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn sqrt(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error>;

    /// Names an intermediate result; the CUDA emitter declares a local.
    fn bind(
        &mut self,
        _name: &'static str,
        value: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(value)
    }

    /// `update(value)` when `flag` is nonzero, else `value` unchanged. The
    /// flag is a coefficient: a host test for host and MLX arithmetic, a
    /// branch on the kernel argument in CUDA. `value` must be bound.
    fn if_nonzero(
        &mut self,
        flag: &Self::Value,
        value: Self::Value,
        update: impl FnOnce(&mut Self, &Self::Value) -> Result<Self::Value, Self::Error>,
    ) -> Result<Self::Value, Self::Error>;
}

/// The updated parameter and moments `(p', m', v')` of one element.
pub type AdamUpdate<V> = (V, V, V);

/// One Adam/AdamW element update: returns `(p', m', v')` from the parameter,
/// gradient, and both moments in the operation order `order`.
///
/// The gradient is the one the update sees: `clip_by_global_norm`'s factor
/// is applied before, by each backend in its own precision (the host and the
/// CUDA kernels scale in float64 and round once, MLX in float32).
pub fn adam_element<A: AdamArith>(
    arith: &mut A,
    order: AdamOrder,
    k: &AdamCoefficients<A::Value>,
    parameter: &A::Value,
    gradient: &A::Value,
    first: &A::Value,
    second: &A::Value,
) -> Result<AdamUpdate<A::Value>, A::Error> {
    match order {
        AdamOrder::Canonical => {
            let decayed = arith.mul(first, &k.beta1)?;
            let fresh = arith.mul(gradient, &k.one_minus_beta1)?;
            let first = arith.add(&decayed, &fresh)?;
            let first = arith.bind("first", first)?;
            let decayed = arith.mul(second, &k.beta2)?;
            let square = arith.mul(gradient, gradient)?;
            let fresh = arith.mul(&square, &k.one_minus_beta2)?;
            let second = arith.add(&decayed, &fresh)?;
            let second = arith.bind("second", second)?;
            let numerator = arith.div(&first, &k.correction1)?;
            let scaled = arith.div(&second, &k.correction2)?;
            let root = arith.sqrt(&scaled)?;
            let denominator = arith.add(&root, &k.epsilon)?;
            let delta = arith.div(&numerator, &denominator)?;
            let delta = arith.bind("delta", delta)?;
            // Zero decay keeps the Adam expression exactly: `0 * inf` would
            // otherwise turn an infinite parameter's step into NaN.
            let delta = arith.if_nonzero(&k.weight_decay, delta, |arith, delta| {
                let decay = arith.mul(&k.weight_decay, parameter)?;
                arith.add(delta, &decay)
            })?;
            let step = arith.mul(&delta, &k.learning_rate)?;
            let parameter = arith.sub(parameter, &step)?;
            Ok((parameter, first, second))
        }
        AdamOrder::V01 => {
            let one = arith.constant(1.0)?;
            let decayed = arith.mul(&k.beta1, first)?;
            let one_minus_beta1 = arith.sub(&one, &k.beta1)?;
            let fresh = arith.mul(&one_minus_beta1, gradient)?;
            let first = arith.add(&decayed, &fresh)?;
            let first = arith.bind("first", first)?;
            let decayed = arith.mul(&k.beta2, second)?;
            let one_minus_beta2 = arith.sub(&one, &k.beta2)?;
            let fresh = arith.mul(&one_minus_beta2, gradient)?;
            let fresh = arith.mul(&fresh, gradient)?;
            let second = arith.add(&decayed, &fresh)?;
            let second = arith.bind("second", second)?;
            let corrected = arith.div(&first, &k.correction1)?;
            let numerator = arith.mul(&k.learning_rate, &corrected)?;
            let scaled = arith.div(&second, &k.correction2)?;
            let root = arith.sqrt(&scaled)?;
            let denominator = arith.add(&root, &k.epsilon)?;
            let step = arith.div(&numerator, &denominator)?;
            let parameter = arith.sub(parameter, &step)?;
            Ok((parameter, first, second))
        }
    }
}

/// One SGD element update `p - g * lr`. v0.1's CUDA kernel wrote `lr * g`;
/// multiplication commutes exactly, so it needs no separate order.
pub fn sgd_element<A: AdamArith>(
    arith: &mut A,
    learning_rate: &A::Value,
    parameter: &A::Value,
    gradient: &A::Value,
) -> Result<A::Value, A::Error> {
    let step = arith.mul(gradient, learning_rate)?;
    arith.sub(parameter, &step)
}

/// Host float64 arithmetic. It records whether a division met a zero
/// divisor, which the eager `Tensor` update reports as an error.
#[derive(Debug, Default)]
pub struct F64Arith {
    pub zero_divisor: bool,
}

impl F64Arith {
    /// [`adam_element`] in float64.
    ///
    /// Inlined so each caller's loop is compiled for its one constant order,
    /// as the expression written in place was. Which input NaN an operation
    /// propagates when two meet (for example `inf * 0` beside a NaN
    /// gradient at `beta = 0`) is the compiler's choice; an out-of-line
    /// version vectorized the two moment sums and flipped that choice on
    /// x86_64, changing NaN sign bits of `optim.Adam.update`.
    #[inline]
    pub fn adam(
        &mut self,
        order: AdamOrder,
        k: &AdamCoefficients<f64>,
        parameter: f64,
        gradient: f64,
        first: f64,
        second: f64,
    ) -> (f64, f64, f64) {
        let Ok(updated) = adam_element(self, order, k, &parameter, &gradient, &first, &second);
        updated
    }

    /// [`sgd_element`] in float64.
    #[inline]
    pub fn sgd(&mut self, learning_rate: f64, parameter: f64, gradient: f64) -> f64 {
        let Ok(updated) = sgd_element(self, &learning_rate, &parameter, &gradient);
        updated
    }
}

impl AdamArith for F64Arith {
    type Value = f64;
    type Error = Infallible;

    fn constant(&mut self, value: f64) -> Result<f64, Infallible> {
        Ok(value)
    }
    fn add(&mut self, lhs: &f64, rhs: &f64) -> Result<f64, Infallible> {
        Ok(lhs + rhs)
    }
    fn sub(&mut self, lhs: &f64, rhs: &f64) -> Result<f64, Infallible> {
        Ok(lhs - rhs)
    }
    fn mul(&mut self, lhs: &f64, rhs: &f64) -> Result<f64, Infallible> {
        Ok(lhs * rhs)
    }
    fn div(&mut self, lhs: &f64, rhs: &f64) -> Result<f64, Infallible> {
        self.zero_divisor |= *rhs == 0.0;
        Ok(lhs / rhs)
    }
    fn sqrt(&mut self, value: &f64) -> Result<f64, Infallible> {
        Ok(value.sqrt())
    }
    fn if_nonzero(
        &mut self,
        flag: &f64,
        value: f64,
        update: impl FnOnce(&mut Self, &f64) -> Result<f64, Infallible>,
    ) -> Result<f64, Infallible> {
        if *flag != 0.0 {
            update(self, &value)
        } else {
            Ok(value)
        }
    }
}

/// CUDA C emitter: each value is a single-precision C expression and
/// [`AdamArith::bind`] appends a local declaration to `statements`. Kernel
/// sources are written in `float` and rewritten to `double` for float64
/// plans like every generated kernel.
#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
#[derive(Debug, Default)]
pub(crate) struct CudaSourceArith {
    pub(crate) statements: String,
}

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
impl CudaSourceArith {
    /// The coefficients as the update kernels' parameter names.
    pub(crate) fn coefficients() -> AdamCoefficients<String> {
        AdamCoefficients {
            learning_rate: "learning_rate".into(),
            beta1: "beta1".into(),
            beta2: "beta2".into(),
            one_minus_beta1: "one_minus_beta1".into(),
            one_minus_beta2: "one_minus_beta2".into(),
            epsilon: "epsilon".into(),
            correction1: "correction1".into(),
            correction2: "correction2".into(),
            weight_decay: "weight_decay".into(),
        }
    }
}

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
impl AdamArith for CudaSourceArith {
    type Value = String;
    type Error = Infallible;

    fn constant(&mut self, value: f64) -> Result<String, Infallible> {
        Ok(format!("{value:?}f"))
    }
    fn add(&mut self, lhs: &String, rhs: &String) -> Result<String, Infallible> {
        Ok(format!("({lhs} + {rhs})"))
    }
    fn sub(&mut self, lhs: &String, rhs: &String) -> Result<String, Infallible> {
        Ok(format!("({lhs} - {rhs})"))
    }
    fn mul(&mut self, lhs: &String, rhs: &String) -> Result<String, Infallible> {
        Ok(format!("({lhs} * {rhs})"))
    }
    fn div(&mut self, lhs: &String, rhs: &String) -> Result<String, Infallible> {
        Ok(format!("({lhs} / {rhs})"))
    }
    fn sqrt(&mut self, value: &String) -> Result<String, Infallible> {
        Ok(format!("sqrtf({value})"))
    }
    fn bind(&mut self, name: &'static str, value: String) -> Result<String, Infallible> {
        self.statements
            .push_str(&format!("    float {name} = {value};\n"));
        Ok(name.to_string())
    }
    fn if_nonzero(
        &mut self,
        flag: &String,
        value: String,
        update: impl FnOnce(&mut Self, &String) -> Result<String, Infallible>,
    ) -> Result<String, Infallible> {
        let Ok(updated) = update(self, &value);
        self.statements
            .push_str(&format!("    if ({flag} != 0.0f) {value} = {updated};\n"));
        Ok(value)
    }
}

/// CUDA statements of one Adam element update over the given operand
/// expressions, and the `(p', m', v')` expressions to store; the kernel
/// templates in `cuda_optimizer.rs` wrap them with loads and stores.
#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
pub(crate) fn cuda_adam_update(
    order: AdamOrder,
    parameter: &str,
    gradient: &str,
    first: &str,
    second: &str,
) -> (String, [String; 3]) {
    let mut arith = CudaSourceArith::default();
    let Ok((parameter, first, second)) = adam_element(
        &mut arith,
        order,
        &CudaSourceArith::coefficients(),
        &parameter.to_string(),
        &gradient.to_string(),
        &first.to_string(),
        &second.to_string(),
    );
    (arith.statements, [parameter, first, second])
}

/// The CUDA expression of one SGD element update.
#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
pub(crate) fn cuda_sgd_update(parameter: &str, gradient: &str) -> String {
    let Ok(updated) = sgd_element(
        &mut CudaSourceArith::default(),
        &CudaSourceArith::coefficients().learning_rate,
        &parameter.to_string(),
        &gradient.to_string(),
    );
    updated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_matches_the_host_optimizer_ranges() {
        let adam = DeviceOptimizerConfig::adam(1e-3, 0.9, 0.999, 1e-8);
        assert_eq!(adam.validate(), Ok(()));
        // Schedules reach a zero rate.
        let zero_rate = DeviceOptimizerConfig {
            learning_rate: 0.0,
            ..adam
        };
        assert_eq!(zero_rate.validate(), Ok(()));
        let sgd = DeviceOptimizerConfig {
            rule: DeviceUpdateRule::Sgd,
            learning_rate: 0.5,
            clip_norm: Some(1.0),
        };
        assert_eq!(sgd.validate(), Ok(()));
        let invalid = [
            DeviceOptimizerConfig {
                learning_rate: -1.0,
                ..adam
            },
            DeviceOptimizerConfig {
                learning_rate: f64::NAN,
                ..adam
            },
            DeviceOptimizerConfig {
                clip_norm: Some(0.0),
                ..adam
            },
            DeviceOptimizerConfig {
                clip_norm: Some(f64::INFINITY),
                ..sgd
            },
            DeviceOptimizerConfig::adam(1e-3, 1.0, 0.999, 1e-8),
            DeviceOptimizerConfig::adam(1e-3, 0.9, f64::NAN, 1e-8),
            DeviceOptimizerConfig::adam(1e-3, 0.9, 0.999, 0.0),
            DeviceOptimizerConfig {
                rule: DeviceUpdateRule::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: -1e-4,
                },
                ..adam
            },
        ];
        for config in invalid {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }

    #[test]
    fn corrections_are_the_float64_powers() {
        assert_eq!(
            DeviceOptimizerConfig::adam_corrections(0.9, 0.999, 1),
            (1.0 - 0.9, 1.0 - 0.999)
        );
        let (first, second) = DeviceOptimizerConfig::adam_corrections(0.9, 0.999, 3);
        assert_eq!(first, 1.0 - 0.9f64.powf(3.0));
        assert_eq!(second, 1.0 - 0.999f64.powf(3.0));
        // In float32 the rounded beta2 loses 1.3e-5 of `1 - beta2`, which is
        // why the device kernels receive it formed in float64.
        let single = f64::from(1.0 - 0.999f32);
        assert!((single - (1.0 - 0.999)).abs() / (1.0 - 0.999) > 1e-5);
    }

    /// Values that reach every branch of the rule: signed zeros, subnormal
    /// and huge magnitudes, infinities, and NaN.
    const EDGES: [f64; 11] = [
        0.0,
        -0.0,
        1e-300,
        1e-40,
        1.5,
        -2.5,
        1e30,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    fn same_bits(left: (f64, f64, f64), right: (f64, f64, f64)) -> bool {
        left.0.to_bits() == right.0.to_bits()
            && left.1.to_bits() == right.1.to_bits()
            && left.2.to_bits() == right.2.to_bits()
    }

    #[test]
    fn each_order_evaluates_its_documented_expression() {
        let coefficient_sets = [
            AdamCoefficients::new(1e-3, 0.9, 0.999, 1e-8, 0.0, 1),
            AdamCoefficients::new(0.05, 0.9, 0.999, 1e-8, 0.1, 7),
            AdamCoefficients::new(0.3, 0.5, 0.75, 0.01, 1e300, 3),
            AdamCoefficients::v01(0.05, 0.8, 0.95, 0.01, 4),
        ];
        let mut arith = F64Arith::default();
        for k in coefficient_sets {
            for p in EDGES {
                for g in EDGES {
                    for m in EDGES {
                        for v in EDGES {
                            // The canonical expression as `AdamOrder::Canonical` documents it.
                            let first = m * k.beta1 + g * k.one_minus_beta1;
                            let second = v * k.beta2 + (g * g) * k.one_minus_beta2;
                            let mut delta = (first / k.correction1)
                                / ((second / k.correction2).sqrt() + k.epsilon);
                            if k.weight_decay != 0.0 {
                                delta += k.weight_decay * p;
                            }
                            let canonical = (p - delta * k.learning_rate, first, second);
                            // The v0.1 expression of `_quabla.Adam.step`.
                            let first = k.beta1 * m + (1.0 - k.beta1) * g;
                            let second = k.beta2 * v + (1.0 - k.beta2) * g * g;
                            let v01 = (
                                p - k.learning_rate * (first / k.correction1)
                                    / ((second / k.correction2).sqrt() + k.epsilon),
                                first,
                                second,
                            );
                            let got = arith.adam(AdamOrder::Canonical, &k, p, g, m, v);
                            assert!(same_bits(got, canonical), "{k:?} {p} {g} {m} {v}");
                            let got = arith.adam(AdamOrder::V01, &k, p, g, m, v);
                            assert!(same_bits(got, v01), "{k:?} {p} {g} {m} {v}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_orders_differ_only_by_rounding() {
        // Without decay the two orders are the same update up to the
        // association of the products, so they differ at the ulp level and
        // nowhere more; this is the change v1.0 makes to the v0.1 entry points.
        let k = AdamCoefficients::new(0.05, 0.9, 0.999, 1e-8, 0.0, 3);
        let mut arith = F64Arith::default();
        let mut differing = 0;
        for index in 0..2000 {
            let x = f64::from(index);
            let (p, g, m, v) = (0.5 - x * 1e-3, (x * 0.37).sin(), (x * 0.11).cos(), x * 1e-4);
            // Rounding is relative to the operands; the moment sums may cancel.
            let scale = 1.0 + p.abs() + g.abs() + m.abs() + v.abs();
            let canonical = arith.adam(AdamOrder::Canonical, &k, p, g, m, v);
            let v01 = arith.adam(AdamOrder::V01, &k, p, g, m, v);
            for (a, b) in [
                (canonical.0, v01.0),
                (canonical.1, v01.1),
                (canonical.2, v01.2),
            ] {
                assert!(
                    (a - b).abs() <= 8.0 * f64::EPSILON * scale.max(a.abs()),
                    "{a} {b}"
                );
            }
            differing += usize::from(!same_bits(canonical, v01));
        }
        assert!(
            differing > 0,
            "the orders should differ at the ulp level somewhere"
        );
        // The v0.1 host corrections use `powi`, the canonical ones `powf`.
        let v01 = AdamCoefficients::v01(0.05, 0.9, 0.999, 1e-8, 3);
        assert_eq!(v01.correction1, 1.0 - 0.9f64.powi(3));
        assert_eq!(v01.weight_decay, 0.0);
        assert_eq!(k.correction1, 1.0 - 0.9f64.powf(3.0));
        // The legacy CUDA kernel's corrections are float32 powers.
        let (c1, c2) = AdamCoefficients::v01_float32_corrections(0.9, 0.999, 3);
        assert_eq!((c1, c2), (1.0 - 0.9f32.powf(3.0), 1.0 - 0.999f32.powf(3.0)));
        assert!((f64::from(c2) - k.correction2).abs() / k.correction2 > 1e-6);
    }

    #[test]
    fn zero_decay_keeps_infinite_parameters_and_flags_zero_divisors() {
        let mut arith = F64Arith::default();
        let adam = AdamCoefficients::new(0.1, 0.9, 0.999, 1e-8, 0.0, 1);
        let (parameter, _, _) =
            arith.adam(AdamOrder::Canonical, &adam, f64::INFINITY, 1.0, 0.0, 0.0);
        assert_eq!(parameter, f64::INFINITY);
        assert!(!arith.zero_divisor);
        let no_epsilon = AdamCoefficients::with_corrections(0.1, 0.9, 0.999, 0.0, 0.1, 0.001, 0.0);
        arith.adam(AdamOrder::Canonical, &no_epsilon, 1.0, 0.0, 0.0, 0.0);
        assert!(arith.zero_divisor);
        assert_eq!(F64Arith::default().sgd(0.5, 1.0, 3.0), 1.0 - 3.0 * 0.5);
    }

    #[test]
    fn cuda_statements_are_the_kernel_expressions() {
        // The legacy `quabla_adam` kernel: `beta1 * m + (1.0f - beta1) * g`,
        // `beta2 * v + (1.0f - beta2) * g * g`, and
        // `p -= learning_rate * (m / c1) / (sqrtf(v / c2) + epsilon)`.
        let (statements, [parameter, first, second]) = cuda_adam_update(
            AdamOrder::V01,
            "parameter[index]",
            "gradient_value",
            "first_moment[index]",
            "second_moment[index]",
        );
        assert_eq!(
            statements,
            "    float first = ((beta1 * first_moment[index]) + ((1.0f - beta1) * gradient_value));\n\
             \x20   float second = ((beta2 * second_moment[index]) + (((1.0f - beta2) * gradient_value) * gradient_value));\n"
        );
        assert_eq!((first.as_str(), second.as_str()), ("first", "second"));
        assert_eq!(
            parameter,
            "(parameter[index] - ((learning_rate * (first / correction1)) / (sqrtf((second / correction2)) + epsilon)))"
        );
        // The fused `quabla_device_adam` kernel, with its runtime decay branch.
        let (statements, [parameter, _, _]) = cuda_adam_update(
            AdamOrder::Canonical,
            "value",
            "gradient_value",
            "first_moment[index]",
            "second_moment[index]",
        );
        assert_eq!(
            statements,
            "    float first = ((first_moment[index] * beta1) + (gradient_value * one_minus_beta1));\n\
             \x20   float second = ((second_moment[index] * beta2) + ((gradient_value * gradient_value) * one_minus_beta2));\n\
             \x20   float delta = ((first / correction1) / (sqrtf((second / correction2)) + epsilon));\n\
             \x20   if (weight_decay != 0.0f) delta = (delta + (weight_decay * value));\n"
        );
        assert_eq!(parameter, "(value - (delta * learning_rate))");
        assert_eq!(
            cuda_sgd_update("parameter[index]", "gradient_value"),
            "(parameter[index] - (gradient_value * learning_rate))"
        );
    }
}
