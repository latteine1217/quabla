use super::{sqrt_derivative_value, DynamicTensor, TensorDType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CholeskyAdKind {
    Jvp,
    Vjp,
    Mixed,
    VjpJvp,
}

// Two independent directions suffice for a Hessian entry. Keeping one jet per
// matrix element avoids a scalar IR tape proportional to cubic arithmetic.
#[derive(Clone, Copy, Default)]
struct Jet {
    value: f64,
    first: f64,
    second: f64,
    mixed: f64,
}

impl Jet {
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            first: self.first + rhs.first,
            second: self.second + rhs.second,
            mixed: self.mixed + rhs.mixed,
        }
    }
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            first: -self.first,
            second: -self.second,
            mixed: -self.mixed,
        }
    }
    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            first: self.first * rhs.value + self.value * rhs.first,
            second: self.second * rhs.value + self.value * rhs.second,
            mixed: ((self.mixed * rhs.value + self.first * rhs.second) + self.second * rhs.first)
                + self.value * rhs.mixed,
        }
    }
    fn reciprocal(self) -> Result<Self, String> {
        if self.value == 0.0 {
            return Err("division by zero is not supported".into());
        }
        let value = 1.0 / self.value;
        let squared = value * value;
        Ok(Self {
            value,
            first: -(self.first * squared),
            second: -(self.second * squared),
            mixed: -(self.mixed * squared) + 2.0 * self.first * self.second * squared * value,
        })
    }
    fn div(self, rhs: Self, symbolic: bool) -> Result<Self, String> {
        let reciprocal = rhs.reciprocal()?;
        let squared = reciprocal.value * reciprocal.value;
        // Follow the existing JVP rule, including the reciprocal's arithmetic
        // order; the primal recurrence uses direct division.
        Ok(Self {
            value: self.value / rhs.value,
            first: if symbolic {
                (self.first * rhs.value - self.value * rhs.first) / (rhs.value * rhs.value)
            } else {
                self.first * reciprocal.value - (self.value * rhs.first) * squared
            },
            second: if symbolic {
                (self.second * rhs.value - self.value * rhs.second) / (rhs.value * rhs.value)
            } else {
                self.second * reciprocal.value - (self.value * rhs.second) * squared
            },
            mixed: (((self.mixed * reciprocal.value - (self.first * rhs.second) * squared)
                - (self.second * rhs.first) * squared)
                - (self.value * rhs.mixed) * squared)
                + ((((self.value * rhs.first) * rhs.second) * (squared * reciprocal.value)) * 2.0),
        })
    }
    fn sqrt_derivative(self, order: u32) -> Self {
        let derivative = sqrt_derivative_value(self.value, order + 1);
        Self {
            value: sqrt_derivative_value(self.value, order),
            first: self.first * derivative,
            second: self.second * derivative,
            mixed: self.mixed * derivative
                + (self.first * self.second) * sqrt_derivative_value(self.value, order + 2),
        }
    }
}

fn factor(input: &[Jet], n: usize, symbolic: bool) -> Result<(Vec<Jet>, Vec<Jet>), String> {
    let mut values = vec![Jet::default(); n * n];
    let mut residuals = vec![Jet::default(); n * n];
    for row in 0..n {
        for column in 0..=row {
            let offset = row * n + column;
            let mut reduced = input[offset];
            for inner in 0..column {
                reduced = reduced.sub(values[row * n + inner].mul(values[column * n + inner]));
            }
            residuals[offset] = reduced;
            values[offset] = if row == column {
                reduced.sqrt_derivative(0)
            } else {
                reduced.div(values[column * n + column], symbolic)?
            };
        }
    }
    Ok((values, residuals))
}

fn reverse(
    values: &[Jet],
    residuals: &[Jet],
    mut cotangents: Vec<Jet>,
    n: usize,
    symbolic: bool,
) -> Result<Vec<Jet>, String> {
    let mut gradient = vec![Jet::default(); n * n];
    for row in (0..n).rev() {
        for column in (0..=row).rev() {
            let offset = row * n + column;
            let upstream = cotangents[offset];
            let reduced = if row == column {
                upstream.mul(residuals[offset].sqrt_derivative(1))
            } else {
                let diagonal = column * n + column;
                let (left, right) = if symbolic {
                    (
                        upstream.div(values[diagonal], true)?,
                        upstream
                            .mul(residuals[offset])
                            .neg()
                            .div(values[diagonal].mul(values[diagonal]), true)?,
                    )
                } else {
                    let reciprocal = values[diagonal].reciprocal()?;
                    (
                        upstream.mul(reciprocal),
                        upstream
                            .mul(residuals[offset])
                            .mul(reciprocal.mul(reciprocal))
                            .neg(),
                    )
                };
                cotangents[diagonal] = cotangents[diagonal].add(right);
                left
            };
            gradient[offset] = reduced;
            for inner in (0..column).rev() {
                let product_cotangent = reduced.neg();
                let left = row * n + inner;
                let right = column * n + inner;
                cotangents[left] = cotangents[left].add(product_cotangent.mul(values[right]));
                cotangents[right] = cotangents[right].add(product_cotangent.mul(values[left]));
            }
        }
    }
    Ok(gradient)
}

pub(super) fn evaluate(
    kind: CholeskyAdKind,
    inputs: &[&DynamicTensor],
) -> Result<DynamicTensor, String> {
    evaluate_with_rules(kind, inputs, false)
}

pub(super) fn evaluate_symbolic(
    kind: CholeskyAdKind,
    inputs: &[&DynamicTensor],
) -> Result<DynamicTensor, String> {
    evaluate_with_rules(kind, inputs, true)
}

fn reference(kind: CholeskyAdKind, inputs: &[&DynamicTensor]) -> Result<DynamicTensor, String> {
    let mut graph = super::TensorIr::new();
    let mut arguments = Vec::with_capacity(inputs.len());
    let mut bindings = std::collections::BTreeMap::new();
    for (index, value) in inputs.iter().enumerate() {
        let name = format!("cholesky_argument_{index}");
        arguments.push(graph.input_typed(
            name.clone(),
            value.shape().to_vec(),
            TensorDType::F64,
        )?);
        bindings.insert(name, (*value).clone());
    }
    let output = graph.expanded_cholesky_ad(&arguments, kind)?;
    graph.evaluate(output, &bindings)
}

fn evaluate_with_rules(
    kind: CholeskyAdKind,
    inputs: &[&DynamicTensor],
    symbolic: bool,
) -> Result<DynamicTensor, String> {
    let input = inputs[0];
    if symbolic
        && inputs
            .iter()
            .any(|input| input.data().iter().any(|value| !value.is_finite()))
    {
        return reference(kind, inputs);
    }
    let n = input.shape()[input.shape().len() - 1];
    if input.shape().len() > 2 {
        let count = input.data().len() / (n * n);
        let mut result = Vec::with_capacity(input.data().len());
        for batch in 0..count {
            let operands = inputs
                .iter()
                .map(|tensor| {
                    DynamicTensor::with_dtype(
                        vec![n, n],
                        tensor.data()[batch * n * n..(batch + 1) * n * n].to_vec(),
                        tensor.dtype(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let references = operands.iter().collect::<Vec<_>>();
            result.extend(
                evaluate_with_rules(kind, &references, symbolic)?
                    .data()
                    .iter()
                    .copied(),
            );
        }
        return DynamicTensor::with_dtype(input.shape().to_vec(), result, TensorDType::F64);
    }
    let data = inputs
        .iter()
        .map(|tensor| tensor.data())
        .collect::<Vec<_>>();
    let jets = (0..n * n)
        .map(|i| Jet {
            value: data[0][i],
            first: if matches!(
                kind,
                CholeskyAdKind::Jvp | CholeskyAdKind::Mixed | CholeskyAdKind::VjpJvp
            ) {
                data[1][i]
            } else {
                0.0
            },
            second: if kind == CholeskyAdKind::Mixed {
                data[2][i]
            } else {
                0.0
            },
            mixed: if kind == CholeskyAdKind::Mixed {
                data[3][i]
            } else {
                0.0
            },
        })
        .collect::<Vec<_>>();
    let (values, residuals) = factor(&jets, n, symbolic)?;
    // Squared denominators outside the normal exponent range can change
    // quotient-rule exceptional behavior. Retain the scalar reference there.
    if symbolic
        && (values.iter().any(|jet| !jet.value.is_finite())
            || (0..n).any(|i| {
                let diagonal = values[i * n + i].value;
                let squared = diagonal * diagonal;
                let fourth = squared * squared;
                fourth == 0.0 || !fourth.is_finite()
            }))
    {
        return reference(kind, inputs);
    }
    let result = match kind {
        CholeskyAdKind::Jvp => values.iter().map(|jet| jet.first).collect(),
        CholeskyAdKind::Mixed => values.iter().map(|jet| jet.mixed).collect(),
        CholeskyAdKind::Vjp | CholeskyAdKind::VjpJvp => {
            let offset = if kind == CholeskyAdKind::Vjp { 1 } else { 2 };
            let cotangents = (0..n * n)
                .map(|i| Jet {
                    value: data[offset][i],
                    first: if kind == CholeskyAdKind::VjpJvp {
                        data[3][i]
                    } else {
                        0.0
                    },
                    ..Jet::default()
                })
                .collect();
            reverse(&values, &residuals, cotangents, n, symbolic)?
                .iter()
                .map(|jet| {
                    if kind == CholeskyAdKind::Vjp {
                        jet.value
                    } else {
                        jet.first
                    }
                })
                .collect()
        }
    };
    DynamicTensor::with_dtype(input.shape().to_vec(), result, TensorDType::F64)
}
