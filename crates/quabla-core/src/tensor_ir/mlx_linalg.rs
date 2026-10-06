// The caller holds MLX_EXECUTION_LOCK throughout graph construction and evaluation.
//
// MLX lowering of `Solve` and the dense `LinalgKind` decompositions. MLX 0.32
// implements its LU, symmetric eigen, QR, and SVD factorizations with LAPACK
// on the CPU stream only, so each factorization is built on `Stream::cpu()`
// and everything around it (symmetrization, the CPU backend's sign
// conventions, determinants, and the triangular solves) stays on the GPU
// stream. MLX orders work across the two streams with events, and on Apple
// silicon both read the same unified-memory buffers, so no data is copied
// between them; LAPACK itself copies each input into the column-major layout
// it factors in place.
//
// Results follow the conventions of the CPU kernels in `linalg.rs` in
// float32, the MLX execution dtype: a matrix with a non-finite entry gives
// NaN outputs (the CPU's rule) and is replaced by zeros before it reaches
// LAPACK, whose eigen and SVD drivers report non-convergence on such input;
// an error in a CPU-stream task would only surface at a later evaluation.
use std::collections::HashMap;
use std::ffi::CString;

use mlx_rs::error::Exception;
use mlx_rs::{linalg, ops, Array, Dtype, Stream, StreamOrDevice};
use mlx_sys::*;

use super::super::LinalgKind;

/// Factorizations already built in one lowering, keyed by the IR node of
/// their input, so the outputs of one decomposition (QR's `Q` and `R`, the
/// SVD's `U`, `s`, and `Vh`, the eigenpairs) and the solves and determinant
/// of one matrix share one LAPACK call, as CSE shares their input node.
#[derive(Default)]
pub(super) struct Factorizations {
    lu: HashMap<usize, (Array, Array)>,
    eigh: HashMap<usize, (Array, Array)>,
    qr: HashMap<(usize, bool), (Array, Array)>,
    svd: HashMap<usize, (Array, Array, Array)>,
}

fn fail(error: Exception) -> String {
    error.to_string()
}

fn extent(value: usize) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| "MLX linalg extent exceeds i32".to_string())
}

fn mlx_shape(shape: &[usize]) -> Result<Vec<i32>, String> {
    shape.iter().map(|value| extent(*value)).collect()
}

/// Runs `build` with `stream` as the default stream of every op it creates.
fn on<T>(stream: &Stream, build: impl FnOnce() -> Result<T, Exception>) -> Result<T, String> {
    mlx_rs::with_stream(stream, build).map_err(fail)
}

/// `(m, n)` of the matrix stack `a` (`[..., m, n]`).
fn matrix_extents(a: &Array) -> (usize, usize) {
    let shape = a.shape();
    let rank = shape.len();
    (shape[rank - 2] as usize, shape[rank - 1] as usize)
}

/// Whether every entry of each matrix of `a` is finite, as `[..., 1, 1]`.
fn finite_matrices(a: &Array) -> Result<Array, Exception> {
    a.is_finite()?.all_axes(&[-2, -1], true)
}

/// `values` where `finite` (broadcast against it) holds and NaN elsewhere.
fn nan_unless(finite: &Array, values: &Array) -> Result<Array, Exception> {
    ops::r#where(finite, values, Array::from_f32(f32::NAN))
}

/// `-1` where `values < 0` and `1` elsewhere (zero and NaN included), the
/// sign flip the CPU kernels apply.
fn flips(values: &Array) -> Result<Array, Exception> {
    ops::r#where(
        values.lt(Array::from_f32(0.0))?,
        Array::from_f32(-1.0),
        Array::from_f32(1.0),
    )
}

/// Entries `start..stop` of axis `axis` (`-1`: columns, `-2`: rows) of the
/// matrix stack `a`.
fn select(a: &Array, start: usize, stop: usize, axis: i32) -> Result<Array, Exception> {
    let indices = (start as i32..stop as i32).collect::<Vec<_>>();
    let count = indices.len() as i32;
    a.take_axis(Array::from_slice(&indices, &[count]), axis)
}

impl Factorizations {
    /// The packed LU factor of `P A = L U` and LAPACK's 0-based pivot rows
    /// (`getrf` by partial pivoting) of matrix node `id`.
    fn lu(&mut self, id: usize, a: &Array) -> Result<(Array, Array), String> {
        if let Some(factor) = self.lu.get(&id) {
            return Ok(factor.clone());
        }
        let factor = on(&Stream::cpu(), || linalg::lu_factor(a))?;
        self.lu.insert(id, factor.clone());
        Ok(factor)
    }

    /// The eigenvalues (ascending) and signed eigenvectors of the symmetric
    /// part of matrix node `id`, with the CPU's conventions (see
    /// [`LinalgKind::EighVectors`]).
    fn eigh(&mut self, id: usize, a: &Array, gpu: &Stream) -> Result<(Array, Array), String> {
        if let Some(pair) = self.eigh.get(&id) {
            return Ok(pair.clone());
        }
        // `a / 2 + a^T / 2` cannot overflow where `(a + a^T) / 2` would, and
        // is exactly symmetric because addition commutes.
        let (finite, symmetric) = on(gpu, || {
            let half = Array::from_f32(0.5);
            let symmetric = a
                .multiply(&half)?
                .add(a.swap_axes(-1, -2)?.multiply(&half)?)?;
            let finite = finite_matrices(&symmetric)?;
            let safe = ops::r#where(&finite, &symmetric, Array::from_f32(0.0))?;
            Ok((finite, safe))
        })?;
        // LAPACK `syevd` on the lower triangle: ascending eigenvalues and the
        // eigenvectors as columns.
        let (values, vectors) = on(&Stream::cpu(), || linalg::eigh(&symmetric, Some("L")))?;
        let pair = on(gpu, || {
            let vectors = vectors.multiply(largest_component_flips(&vectors)?)?;
            Ok((
                nan_unless(&finite.squeeze_axes(&[-1])?, &values)?,
                nan_unless(&finite, &vectors)?,
            ))
        })?;
        self.eigh.insert(id, pair.clone());
        Ok(pair)
    }

    /// `(Q, R)` of matrix node `id` with a non-negative diagonal of `R`:
    /// reduced (`Q` is `m x k`, `R` is `k x n`) or, with `complete` and
    /// `m > n`, `Q` completed to `m x m` (see [`householder`]).
    fn qr(
        &mut self,
        id: usize,
        a: &Array,
        complete: bool,
        gpu: &Stream,
    ) -> Result<(Array, Array), String> {
        if let Some(pair) = self.qr.get(&(id, complete)) {
            return Ok(pair.clone());
        }
        let (finite, safe) = on(gpu, || {
            let finite = finite_matrices(a)?;
            Ok((
                finite.clone(),
                ops::r#where(&finite, a, Array::from_f32(0.0))?,
            ))
        })?;
        let (q, r) = householder(&safe, complete, gpu)?;
        let pair = on(gpu, || {
            Ok((nan_unless(&finite, &q)?, nan_unless(&finite, &r)?))
        })?;
        self.qr.insert((id, complete), pair.clone());
        Ok(pair)
    }

    /// The reduced SVD `(U, s, Vh)` (`U` is `m x k`, `Vh` is `k x n`) of
    /// matrix node `id`, singular values descending, with each column of `U`
    /// signed so its largest-magnitude component (the first on ties) is
    /// positive and the matching row of `Vh` signed alike. LAPACK's extra
    /// columns of a full `U` and rows of a full `Vh` are dropped; the full
    /// kinds complete the basis as the CPU kernel does (see [`complement`]).
    fn svd(&mut self, id: usize, a: &Array, gpu: &Stream) -> Result<(Array, Array, Array), String> {
        if let Some(factors) = self.svd.get(&id) {
            return Ok(factors.clone());
        }
        let (m, n) = matrix_extents(a);
        let k = m.min(n);
        let (finite, safe) = on(gpu, || {
            let finite = finite_matrices(a)?;
            Ok((
                finite.clone(),
                ops::r#where(&finite, a, Array::from_f32(0.0))?,
            ))
        })?;
        // LAPACK `gesdd`: singular values descending, full `U` and `Vh`.
        let (u, s, vh) = on(&Stream::cpu(), || linalg::svd(&safe))?;
        let factors = on(gpu, || {
            let u = select(&u, 0, k, -1)?;
            let vh = select(&vh, 0, k, -2)?;
            let signs = largest_component_flips(&u)?;
            Ok((
                nan_unless(&finite, &u.multiply(&signs)?)?,
                nan_unless(&finite.squeeze_axes(&[-1])?, &s)?,
                nan_unless(&finite, &vh.multiply(signs.swap_axes(-1, -2)?)?)?,
            ))
        })?;
        self.svd.insert(id, factors.clone());
        Ok(factors)
    }
}

/// `+-1` per column of `vectors` (`[..., d, c]`, as `[..., 1, c]`) that
/// makes the column's largest-magnitude component, the first on ties,
/// positive: the eigen- and singular-vector convention of the CPU kernels.
fn largest_component_flips(vectors: &Array) -> Result<Array, Exception> {
    let largest = ops::indexing::argmax_axis(vectors.abs()?, -2, true)?;
    flips(&vectors.take_along_axis(&largest, -2)?)
}

/// The Householder QR of `a` (finite) on the CPU stream with the CPU
/// kernel's signs: each row of `R` and the matching column of `Q` negated
/// where the diagonal of `R` is negative.
///
/// LAPACK's `geqrf` chooses each reflector's sign as the CPU kernel does
/// (`beta = -sign(alpha) ||x||`) and skips a column whose subdiagonal is
/// already zero, so the two agree to rounding even for rank-deficient `a`.
/// For a complete `Q` of a tall `a` (`m > n`), `a` is padded with `m - n`
/// zero columns: they stay zero under the first `n` reflectors, so their own
/// reflectors are the identity and `orgqr` returns `H_1 ... H_n` applied to
/// all `m` columns of the identity, the CPU kernel's completion.
fn householder(a: &Array, complete: bool, gpu: &Stream) -> Result<(Array, Array), String> {
    let (m, n) = matrix_extents(a);
    let padded = if complete && m > n {
        let mut shape = a.shape().to_vec();
        let last = shape.len() - 1;
        shape[last] = extent(m - n)?;
        on(gpu, || {
            ops::concatenate_axis(&[a.clone(), Array::zeros::<f32>(&shape)?], -1)
        })?
    } else {
        a.clone()
    };
    let (q, r) = on(&Stream::cpu(), || linalg::qr(&padded))?;
    on(gpu, || {
        let signs = flips(&r.diagonal(0, -2, -1)?)?;
        // `triu` keeps the subdiagonal +0.0 after a row is negated.
        let r = ops::triu(r.multiply(signs.expand_dims(-1)?)?, 0)?;
        Ok((q.multiply(signs.expand_dims(-2)?)?, r))
    })
}

/// The orthonormal complement of the orthonormal columns `basis`
/// (`[..., d, c]`, `c < d`), as `[..., d, d - c]`: the trailing columns of
/// the complete Householder `Q` of `basis`, unsigned, which is how the CPU
/// kernel completes a full SVD basis.
fn complement(basis: &Array, gpu: &Stream) -> Result<Array, String> {
    let (d, c) = matrix_extents(basis);
    let (q, _) = householder(basis, true, gpu)?;
    on(gpu, || select(&q, c, d, -1))
}

/// Evaluates one [`LinalgKind`] output of matrix node `id` with value `a`
/// (IR shapes have no zero extents, so every matrix has entries).
pub(super) fn evaluate(
    factorizations: &mut Factorizations,
    id: usize,
    a: &Array,
    kind: LinalgKind,
    gpu: &StreamOrDevice,
) -> Result<Array, String> {
    let gpu = gpu.as_ref();
    let (m, n) = matrix_extents(a);
    let k = m.min(n);
    match kind {
        LinalgKind::DetSign | LinalgKind::LogAbsDet => {
            let (lu, pivots) = factorizations.lu(id, a)?;
            on(gpu, || {
                let finite = finite_matrices(a)?.squeeze_axes(&[-2, -1])?;
                let diagonal = lu.diagonal(0, -2, -1)?;
                if kind == LinalgKind::LogAbsDet {
                    // `det = (-1)^swaps prod(u_ii)`: summing `ln|u_ii|` never
                    // forms the product, so it cannot overflow or underflow,
                    // and a zero pivot gives `-inf`.
                    let log = diagonal.abs()?.log()?.sum_axis(-1, false)?;
                    return nan_unless(&finite, &log);
                }
                // Each pivot row other than its own index is one row swap;
                // the product of the swap and diagonal signs is exact.
                let order =
                    Array::arange::<i32, u32>(None, extent(n).map_err(Exception::custom)?, None)?;
                let swaps = ops::r#where(
                    pivots.ne(&order)?,
                    Array::from_f32(-1.0),
                    Array::from_f32(1.0),
                )?;
                let sign = swaps.multiply(flips(&diagonal)?)?.prod_axis(-1, false)?;
                let singular = diagonal.eq(Array::from_f32(0.0))?.any_axis(-1, false)?;
                let sign = ops::r#where(&singular, Array::from_f32(0.0), &sign)?;
                nan_unless(&finite, &sign)
            })
        }
        LinalgKind::EighValues => Ok(factorizations.eigh(id, a, gpu)?.0),
        LinalgKind::EighVectors => Ok(factorizations.eigh(id, a, gpu)?.1),
        LinalgKind::QrQ => Ok(factorizations.qr(id, a, false, gpu)?.0),
        LinalgKind::QrR => Ok(factorizations.qr(id, a, false, gpu)?.1),
        LinalgKind::QrQComplete => Ok(factorizations.qr(id, a, m > n, gpu)?.0),
        LinalgKind::SvdU | LinalgKind::SvdUFull => {
            let u = factorizations.svd(id, a, gpu)?.0;
            if kind == LinalgKind::SvdU || m == k {
                return Ok(u);
            }
            let rest = complement(&u, gpu)?;
            on(gpu, || ops::concatenate_axis(&[u, rest], -1))
        }
        LinalgKind::SvdS => Ok(factorizations.svd(id, a, gpu)?.1),
        LinalgKind::SvdVh | LinalgKind::SvdVhFull => {
            let vh = factorizations.svd(id, a, gpu)?.2;
            if kind == LinalgKind::SvdVh || n == k {
                return Ok(vh);
            }
            let rest = complement(&on(gpu, || vh.swap_axes(-1, -2))?, gpu)?;
            on(gpu, || {
                ops::concatenate_axis(&[vh, rest.swap_axes(-1, -2)?], -2)
            })
        }
    }
}

/// Threads of the substitution kernel that share one matrix, each owning a
/// subset of the right-hand-side columns.
const SOLVE_THREADGROUP: usize = 256;

/// Solves `a @ x == rhs` for every matrix of the leading batch axes
/// (`a`: `[..., n, n]` node `id`, `rhs`: `[..., n, k]`) like the CPU's
/// `solve`: LU with partial pivoting, and an error for an exactly singular
/// matrix.
///
/// MLX's own `linalg::solve` multiplies by explicit triangular inverses,
/// which is not backward stable and raises on a CPU-stream worker for a
/// singular factor. Here only `getrf` runs on the CPU stream; a Metal
/// kernel then applies LAPACK's row interchanges and the two triangular
/// substitutions, with one thread per right-hand-side column. Detecting a
/// zero pivot reads one flag back, so a solve synchronizes with the host
/// once, like a `cond` predicate.
pub(super) fn solve(
    factorizations: &mut Factorizations,
    id: usize,
    a: &Array,
    rhs: &Array,
    shape: &[usize],
    gpu: &StreamOrDevice,
) -> Result<Array, String> {
    let dims = mlx_shape(shape)?;
    let (lu, pivots) = factorizations.lu(id, a)?;
    let singular = on(&Stream::cpu(), || {
        lu.diagonal(0, -2, -1)?.eq(Array::from_f32(0.0))?.any(false)
    })?;
    if singular
        .try_item_cast::<bool>()
        .map_err(|error| format!("MLX solve pivot readback failed: {error}"))?
    {
        return Err("solve requires a non-singular coefficient matrix".to_string());
    }
    let rank = shape.len();
    let (n, columns) = (shape[rank - 2], shape[rank - 1]);
    let batch = shape[..rank - 2].iter().product::<usize>();
    let threads = columns.min(SOLVE_THREADGROUP);
    // Forward substitution with the unit lower factor, then back substitution
    // with the upper one, both in the row order of the CPU's `solve_lu`.
    let body = format!(
        "const ulong n={n},k={columns};\n\
         const ulong b=threadgroup_position_in_grid.x;\n\
         const ulong tid=thread_position_in_threadgroup.x,T=threads_per_threadgroup.x;\n\
         lu+=b*n*n;pivots+=b*n;rhs+=b*n*k;out+=b*n*k;\n\
         for(ulong j=tid;j<k;j+=T){{\n\
         for(ulong i=0;i<n;++i)out[i*k+j]=rhs[i*k+j];\n\
         for(ulong i=0;i<n;++i){{ulong p=pivots[i];if(p!=i){{float t=out[i*k+j];out[i*k+j]=out[p*k+j];out[p*k+j]=t;}}}}\n\
         for(ulong i=1;i<n;++i){{float v=out[i*k+j];for(ulong l=0;l<i;++l)v-=lu[i*n+l]*out[l*k+j];out[i*k+j]=v;}}\n\
         for(ulong c=n;c>0;--c){{ulong i=c-1;float v=out[i*k+j];for(ulong l=i+1;l<n;++l)v-=lu[i*n+l]*out[l*k+j];out[i*k+j]=v/lu[i*n+i];}}\n\
         }}\n"
    );
    metal_kernel(
        c"quabla_lu_solve",
        &[(c"lu", &lu), (c"pivots", &pivots), (c"rhs", rhs)],
        &dims,
        &body,
        (batch * threads, threads),
        gpu.as_ref(),
    )
}

fn check(status: i32, operation: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "MLX Metal solve {operation} failed with status {status}"
        ))
    }
}

// Each wrapper owns exactly one C handle. C constructors copy borrowed names
// and arrays; output extraction creates a separate owned mlx_array handle.
macro_rules! owned_handle {
    ($name:ident, $raw:ty, $free:ident) => {
        struct $name($raw);
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: this handle is owned and is freed exactly once.
                unsafe {
                    $free(self.0);
                }
            }
        }
    };
}
owned_handle!(Names, mlx_vector_string, mlx_vector_string_free);
owned_handle!(Arrays, mlx_vector_array, mlx_vector_array_free);
owned_handle!(Kernel, mlx_fast_metal_kernel, mlx_fast_metal_kernel_free);
owned_handle!(
    Config,
    mlx_fast_metal_kernel_config,
    mlx_fast_metal_kernel_config_free
);

/// Applies the Metal kernel `body` to the named `inputs` (made row
/// contiguous by MLX) with one float32 output `out` of shape `shape`, on a
/// grid of `grid.0` threads in threadgroups of `grid.1`.
fn metal_kernel(
    name: &std::ffi::CStr,
    inputs: &[(&std::ffi::CStr, &Array)],
    shape: &[i32],
    body: &str,
    grid: (usize, usize),
    stream: &Stream,
) -> Result<Array, String> {
    let body = CString::new(body).map_err(|error| error.to_string())?;
    let input_names = inputs
        .iter()
        .map(|(name, _)| name.as_ptr())
        .collect::<Vec<_>>();
    let output_names = [c"out".as_ptr()];
    let handles = inputs
        .iter()
        .map(|(_, array)| array.as_ptr())
        .collect::<Vec<_>>();
    let threads = i32::try_from(grid.0).map_err(|_| "MLX solve grid exceeds i32")?;
    let threadgroup = i32::try_from(grid.1).map_err(|_| "MLX solve threadgroup exceeds i32")?;
    // SAFETY: every pointer references a live, correctly sized buffer for the
    // duration of the call (`input_names`, `output_names`, `handles`, `shape`,
    // and the NUL-terminated `body`, `name`, and empty header); MLX copies the
    // names and input handles. RAII frees every handle on every error path,
    // and the extracted output is separately owned before its vector drops.
    unsafe {
        let input_names = Names(mlx_vector_string_new_data(
            input_names.as_ptr().cast_mut(),
            input_names.len(),
        ));
        let output_names = Names(mlx_vector_string_new_data(
            output_names.as_ptr().cast_mut(),
            output_names.len(),
        ));
        if input_names.0.ctx.is_null() || output_names.0.ctx.is_null() {
            return Err("MLX Metal solve name allocation failed".into());
        }
        let kernel = Kernel(mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names.0,
            output_names.0,
            body.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        ));
        if kernel.0.ctx.is_null() {
            return Err("MLX Metal solve kernel creation failed".into());
        }
        let config = Config(mlx_fast_metal_kernel_config_new());
        if config.0.ctx.is_null() {
            return Err("MLX Metal solve configuration creation failed".into());
        }
        check(
            mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                shape.as_ptr(),
                shape.len(),
                Dtype::Float32.into(),
            ),
            "output shape",
        )?;
        check(
            mlx_fast_metal_kernel_config_set_grid(config.0, threads, 1, 1),
            "grid",
        )?;
        check(
            mlx_fast_metal_kernel_config_set_thread_group(config.0, threadgroup, 1, 1),
            "thread group",
        )?;
        let inputs = Arrays(mlx_vector_array_new_data(handles.as_ptr(), handles.len()));
        let mut outputs = Arrays(mlx_vector_array_new());
        if inputs.0.ctx.is_null() || outputs.0.ctx.is_null() {
            return Err("MLX Metal solve array-vector allocation failed".into());
        }
        check(
            mlx_fast_metal_kernel_apply(
                &mut outputs.0,
                kernel.0,
                inputs.0,
                config.0,
                stream.as_ptr(),
            ),
            "apply",
        )?;
        let mut output = mlx_array_new();
        let status = mlx_vector_array_get(&mut output, outputs.0, 0);
        if status != 0 {
            mlx_array_free(output);
            check(status, "output extraction")?;
        }
        Ok(Array::from_ptr(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MLX's `argmax` returns the first of tied maxima, which is what makes
    /// the sign convention pick the first of equal-magnitude components.
    #[test]
    fn sign_flips_follow_the_first_largest_component() -> Result<(), String> {
        if std::env::var_os("QUABLA_MLX_TEST").is_none() {
            return Ok(());
        }
        let _guard = super::super::mlx_execution_guard();
        // Columns (0.5, -0.5, 0.25), (-0.5, 0.5, -0.25), and (0.1, -0.75, 0.75).
        let vectors = Array::from_slice(
            &[0.5f32, -0.5, 0.1, -0.5, 0.5, -0.75, 0.25, -0.25, 0.75],
            &[3, 3],
        );
        let signs = on(&Stream::gpu(), || largest_component_flips(&vectors))?;
        signs.eval().map_err(fail)?;
        assert_eq!(signs.shape(), &[1, 3]);
        assert_eq!(signs.as_slice::<f32>(), &[1.0, -1.0, -1.0]);
        Ok(())
    }
}
