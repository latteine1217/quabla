//! Host kernels of the dense [`LinalgKind`] decompositions: the sign and
//! log-magnitude of a determinant from an LU factorization with partial
//! pivoting, and the symmetric eigendecomposition by cyclic Jacobi rotations.
//!
//! Every kernel works in `f64` on each matrix of the leading batch axes; the
//! evaluator rounds the result to the node dtype once, so a `float32` input
//! gets the `f64` answer correctly rounded rather than an `f32` recurrence.

use super::{element_count, DynamicTensor};

/// Which output of a dense decomposition a `Linalg` node computes. Each kind
/// is a separate node so the IR keeps one output per node; CSE merges the
/// shared factorization inputs, and a kind that is never used costs nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinalgKind {
    /// `sign(det(A))` in `{-1, 0, 1}` (NaN for a non-finite input), shape
    /// `[...]`. Piecewise constant: its derivative is zero.
    DetSign,
    /// `log|det(A)|`, shape `[...]`; `-inf` for an exactly singular `A`.
    LogAbsDet,
    /// The eigenvalues of `(A + A^T) / 2` in ascending order, shape `[..., n]`.
    EighValues,
    /// The orthonormal eigenvectors of `(A + A^T) / 2` as columns, ordered like
    /// [`Self::EighValues`], each with its largest-magnitude component (the
    /// first on ties) positive; shape `[..., n, n]`.
    EighVectors,
}

impl LinalgKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::DetSign => "det_sign",
            Self::LogAbsDet => "log_abs_det",
            Self::EighValues => "eigh_values",
            Self::EighVectors => "eigh_vectors",
        }
    }

    /// The kind named by [`Self::name`].
    pub fn from_name(name: &str) -> Result<Self, String> {
        [
            Self::DetSign,
            Self::LogAbsDet,
            Self::EighValues,
            Self::EighVectors,
        ]
        .into_iter()
        .find(|kind| kind.name() == name)
        .ok_or_else(|| format!("unknown linalg kind {name:?}"))
    }

    /// The output shape for a square-matrix stack `input` (`[..., n, n]`).
    pub(super) fn output_shape(self, input: &[usize]) -> Result<Vec<usize>, String> {
        let rank = input.len();
        if rank < 2 || input[rank - 2] != input[rank - 1] {
            return Err(format!(
                "{} requires a stack of square matrices [..., n, n], got {input:?}",
                self.name()
            ));
        }
        Ok(match self {
            Self::DetSign | Self::LogAbsDet => input[..rank - 2].to_vec(),
            Self::EighValues => input[..rank - 1].to_vec(),
            Self::EighVectors => input.to_vec(),
        })
    }
}

/// Evaluates `kind` on a floating-point `input` for eager arrays, rounded
/// once to the input dtype like the IR evaluator rounds a `Linalg` node.
pub fn evaluate_eager(kind: LinalgKind, input: &DynamicTensor) -> Result<DynamicTensor, String> {
    if !input.dtype.is_floating() {
        return Err(format!("{} requires a floating-point tensor", kind.name()));
    }
    Ok(evaluate(kind, input)?.into_dtype(input.dtype))
}

/// Evaluates `kind` on every matrix of `input` in `f64`.
pub(super) fn evaluate(kind: LinalgKind, input: &DynamicTensor) -> Result<DynamicTensor, String> {
    let shape = kind.output_shape(&input.shape)?;
    let n = *input.shape.last().expect("validated matrix rank");
    let batch = element_count(&input.shape[..input.shape.len() - 2])?;
    let data = input.data();
    let mut output = Vec::with_capacity(element_count(&shape)?);
    for index in 0..batch {
        let matrix = &data[index * n * n..(index + 1) * n * n];
        match kind {
            LinalgKind::DetSign | LinalgKind::LogAbsDet => {
                let (sign, log_abs) = slogdet(matrix, n);
                output.push(if kind == LinalgKind::DetSign {
                    sign
                } else {
                    log_abs
                });
            }
            LinalgKind::EighValues | LinalgKind::EighVectors => {
                let (values, vectors) = eigh(matrix, n);
                output.extend(if kind == LinalgKind::EighValues {
                    values
                } else {
                    vectors
                });
            }
        }
    }
    DynamicTensor::new(shape, output)
}

/// `(sign, log|det|)` of one row-major `n x n` matrix from Doolittle LU with
/// partial pivoting: `det = (-1)^swaps * prod(u_ii)`. Summing `ln|u_ii|`
/// never forms the product, so the result cannot overflow or underflow where
/// `det` itself would. An exactly zero pivot column means `det = 0`:
/// `(0, -inf)`, as NumPy returns. A non-finite entry gives `(NaN, NaN)`.
pub(super) fn slogdet(matrix: &[f64], n: usize) -> (f64, f64) {
    if matrix.iter().any(|value| !value.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let mut factor = matrix.to_vec();
    let mut sign = 1.0;
    let mut log_abs = 0.0;
    for pivot in 0..n {
        let pivot_row = (pivot..n)
            .max_by(|&left, &right| {
                factor[left * n + pivot]
                    .abs()
                    .total_cmp(&factor[right * n + pivot].abs())
            })
            .expect("pivot range is non-empty");
        let diagonal = factor[pivot_row * n + pivot];
        if diagonal == 0.0 {
            return (0.0, f64::NEG_INFINITY);
        }
        if pivot_row != pivot {
            for column in 0..n {
                factor.swap(pivot * n + column, pivot_row * n + column);
            }
            sign = -sign;
        }
        if diagonal < 0.0 {
            sign = -sign;
        }
        log_abs += diagonal.abs().ln();
        for row in pivot + 1..n {
            let multiplier = factor[row * n + pivot] / diagonal;
            for column in pivot + 1..n {
                factor[row * n + column] -= multiplier * factor[pivot * n + column];
            }
        }
    }
    (sign, log_abs)
}

/// Sweeps after which a Jacobi iteration that has not converged is reported
/// as NaN. Cyclic Jacobi converges quadratically once the off-diagonal mass
/// is small; well-conditioned and clustered spectra alike need fewer than 15
/// sweeps at `f64` precision, so reaching this bound means non-finite data.
const JACOBI_MAX_SWEEPS: usize = 64;

/// Eigenvalues (ascending) and row-major eigenvectors (columns) of the
/// symmetric part `(A + A^T) / 2` of one row-major `n x n` matrix, by cyclic
/// Jacobi rotations (Golub & Van Loan, Algorithm 8.5.3, with Rutishauser's
/// rotation formulas).
///
/// The sweep stops when the off-diagonal Frobenius norm is at most
/// `f64::EPSILON * ||A||_F` (the norm is invariant under the rotations), so
/// every eigenvalue carries an absolute error of order `eps * ||A||_F`, the
/// accuracy of a backward-stable method. Each eigenvector is normalized so
/// its largest-magnitude component (the lowest index on ties) is positive;
/// the eigenvectors of a repeated eigenvalue span the right space but are
/// not unique. A non-finite input yields NaN everywhere.
pub(super) fn eigh(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let nan = || (vec![f64::NAN; n], vec![f64::NAN; n * n]);
    if matrix.iter().any(|value| !value.is_finite()) {
        return nan();
    }
    let mut a = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            a[row * n + column] = 0.5 * (matrix[row * n + column] + matrix[column * n + row]);
        }
    }
    let mut v = vec![0.0; n * n];
    for index in 0..n {
        v[index * n + index] = 1.0;
    }
    let norm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let tolerance = f64::EPSILON * norm;
    let off_diagonal = |a: &[f64]| {
        let mut sum = 0.0;
        for row in 0..n {
            for column in 0..n {
                if row != column {
                    sum += a[row * n + column] * a[row * n + column];
                }
            }
        }
        sum.sqrt()
    };
    let mut converged = norm == 0.0 || off_diagonal(&a) <= tolerance;
    let mut sweeps = 0;
    while !converged {
        if sweeps == JACOBI_MAX_SWEEPS {
            return nan();
        }
        sweeps += 1;
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let (app, aqq) = (a[p * n + p], a[q * n + q]);
                // tan of the rotation angle as the smaller root of
                // t^2 + 2 theta t - 1 = 0; for huge theta, t ~ 1 / (2 theta)
                // avoids squaring theta.
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let (arp, arq) = (a[r * n + p], a[r * n + q]);
                        let new_rp = arp - s * (arq + tau * arp);
                        let new_rq = arq + s * (arp - tau * arq);
                        a[r * n + p] = new_rp;
                        a[p * n + r] = new_rp;
                        a[r * n + q] = new_rq;
                        a[q * n + r] = new_rq;
                    }
                    let (vrp, vrq) = (v[r * n + p], v[r * n + q]);
                    v[r * n + p] = vrp - s * (vrq + tau * vrp);
                    v[r * n + q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
        converged = off_diagonal(&a) <= tolerance;
    }
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&left, &right| a[left * n + left].total_cmp(&a[right * n + right]));
    let values = order.iter().map(|&index| a[index * n + index]).collect();
    let mut vectors = vec![0.0; n * n];
    for (column, &source) in order.iter().enumerate() {
        let mut largest = 0;
        for row in 1..n {
            if v[row * n + source].abs() > v[largest * n + source].abs() {
                largest = row;
            }
        }
        let flip = if v[largest * n + source] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for row in 0..n {
            vectors[row * n + column] = flip * v[row * n + source];
        }
    }
    (values, vectors)
}
