//! Hyperparameters of the fused device optimizer steps.
//!
//! The CUDA and MLX training plans apply one of these rules to every retained
//! parameter after each value-and-gradient execution. The definitions are
//! those of the host optimizers in `quabla.optim` (`SGD`, `Adam`, `AdamW`, and
//! `clip_by_global_norm`), so a device trainer and the CPU trainer agree up to
//! rounding.

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
}
