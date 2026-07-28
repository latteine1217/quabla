use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    value: f64,
    derivative: f64,
}

/// Primal scalar value and an ordered forward-mode gradient.
///
/// `forward_gradient!` preserves the closure parameter order in `gradient`.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardGradient {
    value: f64,
    gradient: Vec<f64>,
}

impl ForwardGradient {
    pub fn new(value: f64, gradient: Vec<f64>) -> Self {
        Self { value, gradient }
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }
}

impl Dual {
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    pub fn value(self) -> f64 {
        self.value
    }

    pub fn derivative(self) -> f64 {
        self.derivative
    }

    pub fn sin(self) -> Self {
        Self::new(self.value.sin(), self.derivative * self.value.cos())
    }

    pub fn cos(self) -> Self {
        Self::new(self.value.cos(), -self.derivative * self.value.sin())
    }

    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self::new(value, self.derivative * value)
    }

    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(1.0);
        }

        Self::new(
            self.value.powi(n),
            self.derivative * f64::from(n) * self.value.powi(n - 1),
        )
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.value + rhs.value, self.derivative + rhs.derivative)
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + Self::constant(rhs)
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;

    fn add(self, rhs: Dual) -> Self::Output {
        Dual::constant(self) + rhs
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value - rhs.value, self.derivative - rhs.derivative)
    }
}

impl Sub<f64> for Dual {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Self::constant(rhs)
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;

    fn sub(self, rhs: Dual) -> Self::Output {
        Dual::constant(self) - rhs
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.value * rhs.value,
            self.derivative * rhs.value + self.value * rhs.derivative,
        )
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Self::constant(rhs)
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;

    fn mul(self, rhs: Dual) -> Self::Output {
        Dual::constant(self) * rhs
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(
            self.value / rhs.value,
            (self.derivative * rhs.value - self.value * rhs.derivative) / (rhs.value * rhs.value),
        )
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self / Self::constant(rhs)
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;

    fn div(self, rhs: Dual) -> Self::Output {
        Dual::constant(self) / rhs
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.value, -self.derivative)
    }
}
