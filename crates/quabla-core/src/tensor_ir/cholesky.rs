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

// Every scalar operation rounds to the jet's dtype, so a `float32` jet follows
// the CPU `f32` reference (each operation is the `f64` result rounded to
// `f32`) in the same operation order as the scalar Cholesky expansion. For
// `float64` the rounding is the identity.
impl Jet {
    fn add(self, rhs: Self, t: TensorDType) -> Self {
        Self {
            value: t.round(self.value + rhs.value),
            first: t.round(self.first + rhs.first),
            second: t.round(self.second + rhs.second),
            mixed: t.round(self.mixed + rhs.mixed),
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
    fn sub(self, rhs: Self, t: TensorDType) -> Self {
        self.add(rhs.neg(), t)
    }
    fn mul(self, rhs: Self, t: TensorDType) -> Self {
        let r = |value: f64| t.round(value);
        Self {
            value: r(self.value * rhs.value),
            first: r(r(self.first * rhs.value) + r(self.value * rhs.first)),
            second: r(r(self.second * rhs.value) + r(self.value * rhs.second)),
            mixed: r(r(r(r(self.mixed * rhs.value) + r(self.first * rhs.second))
                + r(self.second * rhs.first))
                + r(self.value * rhs.mixed)),
        }
    }
    fn reciprocal(self, t: TensorDType) -> Result<Self, String> {
        let r = |value: f64| t.round(value);
        if self.value == 0.0 {
            return Err("division by zero is not supported".into());
        }
        let value = r(1.0 / self.value);
        let squared = r(value * value);
        Ok(Self {
            value,
            first: -r(self.first * squared),
            second: -r(self.second * squared),
            mixed: r(-r(self.mixed * squared)
                + r(r(r(r(2.0 * self.first) * self.second) * squared) * value)),
        })
    }
    fn div(self, rhs: Self, symbolic: bool, t: TensorDType) -> Result<Self, String> {
        let r = |value: f64| t.round(value);
        let reciprocal = rhs.reciprocal(t)?;
        let squared = r(reciprocal.value * reciprocal.value);
        // Follow the existing JVP rule, including the reciprocal's arithmetic
        // order; the primal recurrence uses direct division.
        Ok(Self {
            value: r(self.value / rhs.value),
            first: if symbolic {
                r(r(r(self.first * rhs.value) - r(self.value * rhs.first))
                    / r(rhs.value * rhs.value))
            } else {
                r(r(self.first * reciprocal.value) - r(r(self.value * rhs.first) * squared))
            },
            second: if symbolic {
                r(r(r(self.second * rhs.value) - r(self.value * rhs.second))
                    / r(rhs.value * rhs.value))
            } else {
                r(r(self.second * reciprocal.value) - r(r(self.value * rhs.second) * squared))
            },
            mixed: r(r(r(r(
                r(self.mixed * reciprocal.value) - r(r(self.first * rhs.second) * squared)
            ) - r(r(self.second * rhs.first) * squared))
                - r(r(self.value * rhs.mixed) * squared))
                + r(
                    r(r(r(self.value * rhs.first) * rhs.second) * r(squared * reciprocal.value))
                        * 2.0,
                )),
        })
    }
    fn sqrt_derivative(self, order: u32, t: TensorDType) -> Self {
        let r = |value: f64| t.round(value);
        let derivative = r(sqrt_derivative_value(self.value, order + 1));
        Self {
            value: r(sqrt_derivative_value(self.value, order)),
            first: r(self.first * derivative),
            second: r(self.second * derivative),
            mixed: r(r(self.mixed * derivative)
                + r(r(self.first * self.second) * r(sqrt_derivative_value(self.value, order + 2)))),
        }
    }
}

fn factor(
    input: &[Jet],
    n: usize,
    symbolic: bool,
    t: TensorDType,
) -> Result<(Vec<Jet>, Vec<Jet>), String> {
    let mut values = vec![Jet::default(); n * n];
    let mut residuals = vec![Jet::default(); n * n];
    for row in 0..n {
        for column in 0..=row {
            let offset = row * n + column;
            let mut reduced = input[offset];
            for inner in 0..column {
                reduced = reduced.sub(
                    values[row * n + inner].mul(values[column * n + inner], t),
                    t,
                );
            }
            residuals[offset] = reduced;
            values[offset] = if row == column {
                reduced.sqrt_derivative(0, t)
            } else {
                reduced.div(values[column * n + column], symbolic, t)?
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
    t: TensorDType,
) -> Result<Vec<Jet>, String> {
    let mut gradient = vec![Jet::default(); n * n];
    for row in (0..n).rev() {
        for column in (0..=row).rev() {
            let offset = row * n + column;
            let upstream = cotangents[offset];
            let reduced = if row == column {
                upstream.mul(residuals[offset].sqrt_derivative(1, t), t)
            } else {
                let diagonal = column * n + column;
                let (left, right) = if symbolic {
                    (
                        upstream.div(values[diagonal], true, t)?,
                        upstream.mul(residuals[offset], t).neg().div(
                            values[diagonal].mul(values[diagonal], t),
                            true,
                            t,
                        )?,
                    )
                } else {
                    let reciprocal = values[diagonal].reciprocal(t)?;
                    (
                        upstream.mul(reciprocal, t),
                        upstream
                            .mul(residuals[offset], t)
                            .mul(reciprocal.mul(reciprocal, t), t)
                            .neg(),
                    )
                };
                cotangents[diagonal] = cotangents[diagonal].add(right, t);
                left
            };
            gradient[offset] = reduced;
            for inner in (0..column).rev() {
                let product_cotangent = reduced.neg();
                let left = row * n + inner;
                let right = column * n + inner;
                cotangents[left] = cotangents[left].add(product_cotangent.mul(values[right], t), t);
                cotangents[right] =
                    cotangents[right].add(product_cotangent.mul(values[left], t), t);
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
        arguments.push(graph.input_typed(name.clone(), value.shape().to_vec(), value.dtype())?);
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
    let t = input.dtype();
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
        return DynamicTensor::with_dtype(input.shape().to_vec(), result, t);
    }
    // Operands enter the jets at the result dtype, as graph inputs of a
    // `float32` node are rounded before its first operation.
    let data = inputs
        .iter()
        .map(|tensor| tensor.data().iter().map(|value| t.round(*value)).collect())
        .collect::<Vec<Vec<f64>>>();
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
    let (values, residuals) = factor(&jets, n, symbolic, t)?;
    // Squared denominators outside the normal exponent range can change
    // quotient-rule exceptional behavior. Retain the scalar reference there.
    if symbolic
        && (values.iter().any(|jet| !jet.value.is_finite())
            || (0..n).any(|i| {
                let diagonal = values[i * n + i].value;
                let squared = t.round(diagonal * diagonal);
                let fourth = t.round(squared * squared);
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
            reverse(&values, &residuals, cotangents, n, symbolic, t)?
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
    DynamicTensor::with_dtype(input.shape().to_vec(), result, t)
}
