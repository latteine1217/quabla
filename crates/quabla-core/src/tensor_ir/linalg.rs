//! Host kernels of the dense [`LinalgKind`] decompositions: the sign and
//! log-magnitude of a determinant from an LU factorization with partial
//! pivoting, the symmetric eigendecomposition by cyclic Jacobi rotations,
//! the QR factorization by Householder reflections, and the singular value
//! decomposition by one-sided (Hestenes) Jacobi rotations.
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
    /// The orthonormal factor `Q` of the reduced QR factorization `A = Q R`
    /// of `A` (`[..., m, n]`), shape `[..., m, k]` with `k = min(m, n)`.
    /// The sign of each column follows [`Self::QrR`].
    QrQ,
    /// The upper-triangular factor `R` of the reduced QR factorization,
    /// shape `[..., k, n]`, with a non-negative diagonal.
    QrR,
    /// The `m x m` orthogonal `Q` of the complete QR factorization: the
    /// reduced `Q` followed by an orthonormal basis of its complement.
    QrQComplete,
    /// The left singular vectors of the reduced SVD `A = U diag(s) Vh`,
    /// shape `[..., m, k]`; each column has its largest-magnitude component
    /// (the first on ties) positive.
    SvdU,
    /// The singular values in descending order, shape `[..., k]`.
    SvdS,
    /// The right singular vectors as rows, shape `[..., k, n]`, signed to
    /// match [`Self::SvdU`].
    SvdVh,
    /// The `m x m` orthogonal `U` of the full SVD: [`Self::SvdU`] followed
    /// by an orthonormal basis of its complement.
    SvdUFull,
    /// The `n x n` orthogonal `Vh` of the full SVD: [`Self::SvdVh`]
    /// followed by an orthonormal basis of the complement of its rows.
    SvdVhFull,
}

impl LinalgKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::DetSign => "det_sign",
            Self::LogAbsDet => "log_abs_det",
            Self::EighValues => "eigh_values",
            Self::EighVectors => "eigh_vectors",
            Self::QrQ => "qr_q",
            Self::QrR => "qr_r",
            Self::QrQComplete => "qr_q_complete",
            Self::SvdU => "svd_u",
            Self::SvdS => "svd_s",
            Self::SvdVh => "svd_vh",
            Self::SvdUFull => "svd_u_full",
            Self::SvdVhFull => "svd_vh_full",
        }
    }

    /// The kind named by [`Self::name`].
    pub fn from_name(name: &str) -> Result<Self, String> {
        [
            Self::DetSign,
            Self::LogAbsDet,
            Self::EighValues,
            Self::EighVectors,
            Self::QrQ,
            Self::QrR,
            Self::QrQComplete,
            Self::SvdU,
            Self::SvdS,
            Self::SvdVh,
            Self::SvdUFull,
            Self::SvdVhFull,
        ]
        .into_iter()
        .find(|kind| kind.name() == name)
        .ok_or_else(|| format!("unknown linalg kind {name:?}"))
    }

    /// Whether the kind is defined only for square matrices.
    fn square_only(self) -> bool {
        matches!(
            self,
            Self::DetSign | Self::LogAbsDet | Self::EighValues | Self::EighVectors
        )
    }

    /// The output shape for a matrix stack `input` (`[..., m, n]`, square
    /// for the determinant and eigen kinds).
    pub(super) fn output_shape(self, input: &[usize]) -> Result<Vec<usize>, String> {
        let rank = input.len();
        if rank < 2 {
            return Err(format!(
                "{} requires a stack of matrices [..., m, n], got {input:?}",
                self.name()
            ));
        }
        let (m, n) = (input[rank - 2], input[rank - 1]);
        if self.square_only() && m != n {
            return Err(format!(
                "{} requires a stack of square matrices [..., n, n], got {input:?}",
                self.name()
            ));
        }
        let k = m.min(n);
        let batch = &input[..rank - 2];
        Ok(match self {
            Self::DetSign | Self::LogAbsDet => batch.to_vec(),
            Self::EighValues => input[..rank - 1].to_vec(),
            Self::EighVectors => input.to_vec(),
            Self::QrQ | Self::SvdU => [batch, &[m, k]].concat(),
            Self::QrR | Self::SvdVh => [batch, &[k, n]].concat(),
            Self::QrQComplete | Self::SvdUFull => [batch, &[m, m]].concat(),
            Self::SvdS => [batch, &[k]].concat(),
            Self::SvdVhFull => [batch, &[n, n]].concat(),
        })
    }
}

/// Evaluates `kind` on every matrix of `input` in `f64`.
pub(super) fn evaluate(kind: LinalgKind, input: &DynamicTensor) -> Result<DynamicTensor, String> {
    let shape = kind.output_shape(&input.shape)?;
    let rank = input.shape.len();
    let (m, n) = (input.shape[rank - 2], input.shape[rank - 1]);
    let batch = element_count(&input.shape[..rank - 2])?;
    let data = input.data();
    let mut output = Vec::with_capacity(element_count(&shape)?);
    for index in 0..batch {
        let matrix = &data[index * m * n..(index + 1) * m * n];
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
            LinalgKind::QrQ | LinalgKind::QrR | LinalgKind::QrQComplete => {
                let complete = kind == LinalgKind::QrQComplete;
                let (q, r) = qr(matrix, m, n, complete);
                output.extend(if kind == LinalgKind::QrR { r } else { q });
            }
            LinalgKind::SvdU
            | LinalgKind::SvdS
            | LinalgKind::SvdVh
            | LinalgKind::SvdUFull
            | LinalgKind::SvdVhFull => {
                let full = matches!(kind, LinalgKind::SvdUFull | LinalgKind::SvdVhFull);
                let (u, s, vh) = svd(matrix, m, n, full);
                output.extend(match kind {
                    LinalgKind::SvdS => s,
                    LinalgKind::SvdU | LinalgKind::SvdUFull => u,
                    _ => vh,
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

/// The Euclidean norm of `values`, scaled by the largest magnitude so that
/// no square overflows or underflows.
fn scaled_norm(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let scale = values
        .clone()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    scale
        * values
            .map(|value| (value / scale) * (value / scale))
            .sum::<f64>()
            .sqrt()
}

/// `(Q, R)` of one row-major `m x n` matrix by Householder reflections in
/// `f64` (Golub & Van Loan, Algorithm 5.2.1, with LAPACK `dlarfg`'s choice
/// of the reflector sign, which avoids cancellation).
///
/// With `k = min(m, n)`, `R` is `k x n` upper triangular and `Q` is `m x k`,
/// or `m x m` when `complete`, so that `A = Q[:, :k] R`. Each row of `R` and
/// the matching column of `Q` are then negated where needed to make the
/// diagonal of `R` non-negative, which makes the factorization unique for a
/// matrix of full column rank. `Q` is formed by applying the reflectors to
/// the identity, so it is orthonormal to working precision even when `A` is
/// rank deficient. A non-finite input yields NaN everywhere.
pub(super) fn qr(matrix: &[f64], m: usize, n: usize, complete: bool) -> (Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    let q_columns = if complete { m } else { k };
    if matrix.iter().any(|value| !value.is_finite()) {
        return (vec![f64::NAN; m * q_columns], vec![f64::NAN; k * n]);
    }
    let mut a = matrix.to_vec();
    // Reflector `j` is `I - tau_j v_j v_j^T` with `v_j[j] = 1`; the tail of
    // `v_j` overwrites column `j` below the diagonal.
    let mut taus = vec![0.0; k];
    for j in 0..k {
        let alpha = a[j * n + j];
        let tail = scaled_norm((j + 1..m).map(|row| a[row * n + j]));
        if tail == 0.0 {
            continue;
        }
        let beta = -alpha.signum() * alpha.hypot(tail);
        let scale = 1.0 / (alpha - beta);
        for row in j + 1..m {
            a[row * n + j] *= scale;
        }
        taus[j] = (beta - alpha) / beta;
        a[j * n + j] = beta;
        for column in j + 1..n {
            let mut dot = a[j * n + column];
            for row in j + 1..m {
                dot += a[row * n + j] * a[row * n + column];
            }
            let update = taus[j] * dot;
            a[j * n + column] -= update;
            for row in j + 1..m {
                a[row * n + column] -= update * a[row * n + j];
            }
        }
    }
    let mut q = vec![0.0; m * q_columns];
    for index in 0..q_columns {
        q[index * q_columns + index] = 1.0;
    }
    for j in (0..k).rev() {
        if taus[j] == 0.0 {
            continue;
        }
        for column in 0..q_columns {
            let mut dot = q[j * q_columns + column];
            for row in j + 1..m {
                dot += a[row * n + j] * q[row * q_columns + column];
            }
            let update = taus[j] * dot;
            q[j * q_columns + column] -= update;
            for row in j + 1..m {
                q[row * q_columns + column] -= update * a[row * n + j];
            }
        }
    }
    let mut r = vec![0.0; k * n];
    for row in 0..k {
        let flip = if a[row * n + row] < 0.0 { -1.0 } else { 1.0 };
        for column in row..n {
            r[row * n + column] = flip * a[row * n + column];
        }
        if flip < 0.0 {
            for q_row in 0..m {
                q[q_row * q_columns + row] = -q[q_row * q_columns + row];
            }
        }
    }
    (q, r)
}

/// Sweeps after which a one-sided Jacobi iteration that has not converged is
/// reported as NaN; converging inputs need about 5 to 15 sweeps.
const SVD_MAX_SWEEPS: usize = 80;

/// The factors of one SVD: `U` (row-major), the singular values, and `Vh`
/// (row-major).
type SvdFactors = (Vec<f64>, Vec<f64>, Vec<f64>);

/// `(U, s, Vh)` of one row-major `m x n` matrix: `U` is `m x k` (`m x m`
/// when `full`), `s` holds the `k = min(m, n)` singular values in
/// descending order, and `Vh` is `k x n` (`n x n` when `full`).
///
/// A wide matrix is decomposed through its transpose. The columns of `U`
/// for zero singular values, and the extra columns of a full `U` (or rows of
/// a full `Vh`), complete an orthonormal basis. Each of the first `k`
/// columns of `U` is then signed so its largest-magnitude component (the
/// first on ties) is positive, with the matching row of `Vh` signed alike.
/// A non-finite input yields NaN everywhere.
pub(super) fn svd(matrix: &[f64], m: usize, n: usize, full: bool) -> SvdFactors {
    let k = m.min(n);
    let u_columns = if full { m } else { k };
    let vh_rows = if full { n } else { k };
    let nan = || {
        (
            vec![f64::NAN; m * u_columns],
            vec![f64::NAN; k],
            vec![f64::NAN; vh_rows * n],
        )
    };
    if matrix.iter().any(|value| !value.is_finite()) {
        return nan();
    }
    // The columns of the tall operand: `A` itself, or `A^T` for a wide `A`.
    let (rows, columns) = if m >= n { (m, n) } else { (n, m) };
    let mut tall = vec![vec![0.0; rows]; columns];
    for (row, values) in matrix.chunks(n).enumerate() {
        for (column, &value) in values.iter().enumerate() {
            if m >= n {
                tall[column][row] = value;
            } else {
                tall[row][column] = value;
            }
        }
    }
    let Some((values, left, right)) = hestenes(tall, rows) else {
        return nan();
    };
    // `left` spans the column space of the tall operand and `right` its row
    // space; for a wide `A` they swap roles.
    let (mut u_basis, mut v_basis) = if m >= n { (left, right) } else { (right, left) };
    complete_basis(&mut u_basis, m, u_columns);
    complete_basis(&mut v_basis, n, vh_rows);
    for index in 0..k {
        let column = &u_basis[index];
        let mut largest = 0;
        for row in 1..m {
            if column[row].abs() > column[largest].abs() {
                largest = row;
            }
        }
        if column[largest] < 0.0 {
            u_basis[index].iter_mut().for_each(|value| *value = -*value);
            v_basis[index].iter_mut().for_each(|value| *value = -*value);
        }
    }
    let mut u = vec![0.0; m * u_columns];
    for (column, values) in u_basis.iter().enumerate() {
        for (row, &value) in values.iter().enumerate() {
            u[row * u_columns + column] = value;
        }
    }
    (u, values, v_basis.concat())
}

/// The singular values (descending), the left singular vectors of nonzero
/// singular values, and all right singular vectors of a tall matrix.
type HestenesFactors = (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// One-sided Jacobi SVD (Hestenes; Demmel & Veselic, 1992) of the tall
/// matrix `G` given by its `columns.len() <= rows` columns: plane rotations
/// of column pairs, accumulated in `V`, until every pair of columns of
/// `G V` is orthogonal to `rows * eps` in angle. The singular values are
/// then the column norms, accurate to a small multiple of `eps` relative to
/// each value when `G` is well conditioned up to a column scaling, and the
/// normalized columns are the left singular vectors.
///
/// Returns the singular values in descending order (ties keep their column
/// order), the normalized columns of `G V` with a nonzero norm in that
/// order (fewer than the column count for a rank-deficient `G`), and the
/// columns of `V` in that order. `None` when the sweeps do not converge.
fn hestenes(mut g: Vec<Vec<f64>>, rows: usize) -> Option<HestenesFactors> {
    let columns = g.len();
    let mut v = (0..columns)
        .map(|index| {
            let mut column = vec![0.0; columns];
            column[index] = 1.0;
            column
        })
        .collect::<Vec<_>>();
    let tolerance = rows.max(1) as f64 * f64::EPSILON;
    let rotate = |lhs: &mut [f64], rhs: &mut [f64], c: f64, s: f64| {
        for (l, r) in lhs.iter_mut().zip(rhs.iter_mut()) {
            let (left, right) = (*l, *r);
            *l = c * left - s * right;
            *r = s * left + c * right;
        }
    };
    let mut converged = columns < 2;
    let mut sweeps = 0;
    while !converged {
        if sweeps == SVD_MAX_SWEEPS {
            return None;
        }
        sweeps += 1;
        converged = true;
        for p in 0..columns {
            for q in p + 1..columns {
                let (head, tail) = g.split_at_mut(q);
                let (gp, gq) = (&mut head[p], &mut tail[0]);
                // The cosine of the angle between the columns, from scaled
                // norms so huge or tiny columns neither overflow nor flush
                // to zero.
                let norm_p = scaled_norm(gp.iter().copied());
                let norm_q = scaled_norm(gq.iter().copied());
                if norm_p == 0.0 || norm_q == 0.0 {
                    continue;
                }
                let gamma = gp
                    .iter()
                    .zip(gq.iter())
                    .map(|(l, r)| (l / norm_p) * (r / norm_q))
                    .sum::<f64>();
                if gamma.abs() <= tolerance {
                    continue;
                }
                converged = false;
                // Rutishauser's rotation that zeroes the off-diagonal entry of
                // the pair's Gram matrix: `t` is the smaller root of
                // `t^2 + 2 zeta t - 1 = 0`, with `t ~ 1 / (2 zeta)` for huge
                // `zeta` to avoid squaring it.
                let zeta = (norm_q / norm_p - norm_p / norm_q) / (2.0 * gamma);
                let t = if zeta.abs() > 1e150 {
                    0.5 / zeta
                } else {
                    zeta.signum() / (zeta.abs() + (zeta * zeta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = c * t;
                rotate(gp, gq, c, s);
                let (head, tail) = v.split_at_mut(q);
                rotate(&mut head[p], &mut tail[0], c, s);
            }
        }
    }
    let norms = g
        .iter()
        .map(|column| scaled_norm(column.iter().copied()))
        .collect::<Vec<_>>();
    let mut order = (0..columns).collect::<Vec<_>>();
    order.sort_by(|&lhs, &rhs| norms[rhs].total_cmp(&norms[lhs]));
    let values = order.iter().map(|&index| norms[index]).collect();
    let left = order
        .iter()
        .filter(|&&index| norms[index] > 0.0)
        .map(|&index| g[index].iter().map(|value| value / norms[index]).collect())
        .collect();
    let right = order.iter().map(|&index| v[index].clone()).collect();
    Some((values, left, right))
}

/// Extends the orthonormal vectors `basis` (each of length `dimension`) to
/// `count` vectors with the trailing columns of the complete Householder
/// `Q` of the matrix whose columns are `basis`: those columns are
/// orthonormal and orthogonal to the span of `basis` to working precision.
fn complete_basis(basis: &mut Vec<Vec<f64>>, dimension: usize, count: usize) {
    let known = basis.len();
    if known >= count {
        return;
    }
    let mut matrix = vec![0.0; dimension * known];
    for (column, values) in basis.iter().enumerate() {
        for (row, &value) in values.iter().enumerate() {
            matrix[row * known + column] = value;
        }
    }
    let (q, _) = qr(&matrix, dimension, known, true);
    for column in known..count {
        basis.push(
            (0..dimension)
                .map(|row| q[row * dimension + column])
                .collect(),
        );
    }
}
