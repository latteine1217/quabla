//! cuSOLVER lowering of the QR and SVD [`LinalgKind`]s: `geqrf` + `orgqr`
//! for QR and the Jacobi SVD `gesvdj`, each called once per matrix of the
//! leading batch axes on the solver handle's stream, followed by small
//! kernels that convert the column-major results to the row-major layout
//! and the sign conventions of the CPU kernels in `linalg.rs`.

use std::sync::{Arc, Mutex};

use cudarc::cusolver::sys as cusolver_sys;
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, DevicePtrMut, PushKernelArg};

use super::{
    launch_cuda_transpose_copy, linalg_kernel, linalg_launch_config, CudaReal, CudaSolver,
};
use crate::tensor_ir::LinalgKind;

/// Device kernels of this lowering. `quabla_column_major_load` copies each
/// row-major `m x n` matrix into a column-major slot of `stride` floats.
/// `quabla_qr_signs` records the sign of each diagonal entry of the `geqrf`
/// factor (`+1` for zero), and `quabla_qr_r`/`quabla_qr_q` read `R` and `Q`
/// out row-major with those signs applied to the rows of `R` and the
/// leading columns of `Q`, so the diagonal of `R` is non-negative.
/// `quabla_svd_signs` flips each of the first `k` columns of `U` (and the
/// matching row of `Vh`) so its largest-magnitude component, the first on
/// ties, is positive.
pub(super) const CUDA_DECOMPOSITION_SOURCE: &str = r#"
extern "C" __global__ void quabla_column_major_load(const float* input, float* factor, unsigned long long batch, unsigned long long m, unsigned long long n, unsigned long long stride) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * m * n) return;
    unsigned long long b = index / (m * n);
    unsigned long long local = index % (m * n);
    factor[b * stride + local / n + (local % n) * m] = input[index];
}

extern "C" __global__ void quabla_qr_signs(const float* factor, float* signs, unsigned long long batch, unsigned long long m, unsigned long long k, unsigned long long stride) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * k) return;
    unsigned long long b = index / k;
    unsigned long long i = index % k;
    signs[index] = factor[b * stride + i + i * m] < 0.0f ? -1.0f : 1.0f;
}

extern "C" __global__ void quabla_qr_r(const float* factor, const float* signs, float* out, unsigned long long batch, unsigned long long m, unsigned long long n, unsigned long long k, unsigned long long stride) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * k * n) return;
    unsigned long long b = index / (k * n);
    unsigned long long local = index % (k * n);
    unsigned long long i = local / n;
    unsigned long long j = local % n;
    out[index] = j >= i ? signs[b * k + i] * factor[b * stride + i + j * m] : 0.0f;
}

extern "C" __global__ void quabla_qr_q(const float* factor, const float* signs, float* out, unsigned long long batch, unsigned long long m, unsigned long long columns, unsigned long long k, unsigned long long stride) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * m * columns) return;
    unsigned long long b = index / (m * columns);
    unsigned long long local = index % (m * columns);
    unsigned long long i = local / columns;
    unsigned long long j = local % columns;
    float value = factor[b * stride + i + j * m];
    out[index] = j < k ? signs[b * k + j] * value : value;
}

extern "C" __global__ void quabla_svd_signs(float* u, float* vh, unsigned long long batch, unsigned long long m, unsigned long long u_columns, unsigned long long n, unsigned long long vh_rows, unsigned long long k) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * k) return;
    unsigned long long b = index / k;
    unsigned long long j = index % k;
    float* left = u + b * m * u_columns;
    float* right = vh + b * vh_rows * n;
    unsigned long long largest = 0;
    for (unsigned long long row = 1; row < m; ++row) {
        if (fabsf(left[row * u_columns + j]) > fabsf(left[largest * u_columns + j])) largest = row;
    }
    if (left[largest * u_columns + j] < 0.0f) {
        for (unsigned long long row = 0; row < m; ++row) left[row * u_columns + j] = -left[row * u_columns + j];
        for (unsigned long long column = 0; column < n; ++column) right[j * n + column] = -right[j * n + column];
    }
}
"#;

/// Whether `kind` is lowered here rather than by `launch_cusolver_linalg`.
pub(super) fn is_decomposition(kind: LinalgKind) -> bool {
    matches!(
        kind,
        LinalgKind::QrQ
            | LinalgKind::QrR
            | LinalgKind::QrQComplete
            | LinalgKind::SvdU
            | LinalgKind::SvdS
            | LinalgKind::SvdVh
            | LinalgKind::SvdUFull
            | LinalgKind::SvdVhFull
    )
}

/// The shapes of one lowering: `batch` row-major `m x n` input matrices.
#[derive(Clone, Copy)]
pub(super) struct MatrixStack {
    pub(super) batch: usize,
    pub(super) m: usize,
    pub(super) n: usize,
}

fn as_i32(value: usize, what: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("CUSOLVER {what} exceeds i32"))
}

fn as_u64(value: usize) -> u64 {
    value as u64
}

/// Copies the row-major matrices of `matrix` into column-major slots of
/// `stride` floats of a new buffer.
fn load_column_major<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    matrix: &CudaSlice<T>,
    stack: MatrixStack,
    stride: usize,
) -> Result<CudaSlice<T>, String> {
    let mut factor = stream
        .alloc_zeros::<T>((stack.batch * stride).max(1))
        .map_err(|error| format!("failed to allocate CUSOLVER factor buffer: {error:?}"))?;
    let kernel = linalg_kernel(module, "quabla_column_major_load")?;
    let mut launch = stream.launch_builder(&kernel);
    let (batch, m, n, stride_u64) = (
        as_u64(stack.batch),
        as_u64(stack.m),
        as_u64(stack.n),
        as_u64(stride),
    );
    launch.arg(matrix);
    launch.arg(&mut factor);
    launch.arg(&batch);
    launch.arg(&m);
    launch.arg(&n);
    launch.arg(&stride_u64);
    // SAFETY: the arguments match `quabla_column_major_load(const float*, float*, unsigned long
    // long x4)`; `matrix` holds `batch * m * n` floats and `factor` `batch * stride` floats with
    // `stride >= m * n`, and the kernel guards its index against `batch * m * n`.
    unsafe { launch.launch(linalg_launch_config(stack.batch * stack.m * stack.n)?) }
        .map_err(|error| format!("failed to launch CUDA column-major load: {error:?}"))?;
    Ok(factor)
}

/// Checks the per-matrix `info` words of a cuSOLVER call: any nonzero value
/// is an invalid argument (negative) or a failed factorization (positive,
/// for `gesvdj` a Jacobi iteration that did not converge).
fn check_info(
    stream: &Arc<CudaStream>,
    info: &CudaSlice<i32>,
    batch: usize,
    name: &str,
) -> Result<(), String> {
    let mut host = vec![0_i32; batch];
    stream
        .memcpy_dtoh(info, &mut host)
        .map_err(|error| format!("failed to read CUSOLVER {name} status: {error:?}"))?;
    match host.iter().enumerate().find(|(_, status)| **status != 0) {
        Some((index, status)) => Err(format!(
            "CUSOLVER {name} failed with devInfo={status} for batch element {index}"
        )),
        None => Ok(()),
    }
}

/// Evaluates one QR or SVD [`LinalgKind`] output for the row-major matrix
/// stack `matrix` into `output`.
pub(super) fn launch<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    solver: &Arc<Mutex<CudaSolver<T>>>,
    kind: LinalgKind,
    matrix: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    stack: MatrixStack,
) -> Result<(), String> {
    match kind {
        LinalgKind::QrQ | LinalgKind::QrR | LinalgKind::QrQComplete => {
            launch_qr(stream, module, solver, kind, matrix, output, stack)
        }
        _ => launch_svd(stream, module, solver, kind, matrix, output, stack),
    }
}

/// QR by `geqrf` and, for `Q`, `orgqr` on a column-major copy whose slots
/// hold `max(n, q_columns)` columns, so `orgqr` can expand the `k`
/// reflectors to all `m` columns of a complete `Q` in place.
fn launch_qr<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    solver: &Arc<Mutex<CudaSolver<T>>>,
    kind: LinalgKind,
    matrix: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    stack: MatrixStack,
) -> Result<(), String> {
    let name = kind.name();
    let MatrixStack { batch, m, n } = stack;
    let k = m.min(n);
    let q_columns = if kind == LinalgKind::QrQComplete {
        m
    } else {
        k
    };
    let stride = m * n.max(q_columns);
    let mut factor = load_column_major(stream, module, matrix, stack, stride)?;
    let alloc_real = |count: usize| {
        stream
            .alloc_zeros::<T>(count.max(1))
            .map_err(|error| format!("failed to allocate CUSOLVER {name} buffer: {error:?}"))
    };
    let mut taus = alloc_real(batch * k)?;
    let mut signs = alloc_real(batch * k)?;
    let mut info = stream
        .alloc_zeros::<i32>(batch.max(1))
        .map_err(|error| format!("failed to allocate CUSOLVER {name} status: {error:?}"))?;
    let (m_i32, n_i32, k_i32, q_i32) = (
        as_i32(m, "QR rows")?,
        as_i32(n, "QR columns")?,
        as_i32(k, "QR rank")?,
        as_i32(q_columns, "QR columns")?,
    );
    let want_q = kind != LinalgKind::QrR;
    let solver = solver
        .lock()
        .map_err(|_| "CUSOLVER handle lock is poisoned".to_string())?;
    let handle = &solver.handle;
    let element = std::mem::size_of::<T>() as u64;
    let int = std::mem::size_of::<i32>() as u64;
    let mut workspace = [0_i32; 2];
    {
        let (factor_ptr, _factor_guard) = factor.device_ptr_mut(stream);
        let (tau_ptr, _tau_guard) = taus.device_ptr_mut(stream);
        // SAFETY: the live handle owns this stream; `factor` holds at least one column-major
        // `m x max(n, q_columns)` matrix with `lda = m`, `taus` at least `k` floats, and the
        // workspace sizes are written to valid host locations.
        unsafe {
            (T::GEQRF_BUFFER_SIZE)(
                handle.cu(),
                m_i32,
                n_i32,
                factor_ptr as *mut T,
                m_i32,
                &mut workspace[0],
            )
            .result()
            .map_err(|error| format!("CUSOLVER geqrf workspace query failed: {error:?}"))?;
            if want_q {
                (T::ORGQR_BUFFER_SIZE)(
                    handle.cu(),
                    m_i32,
                    q_i32,
                    k_i32,
                    factor_ptr as *const T,
                    m_i32,
                    tau_ptr as *const T,
                    &mut workspace[1],
                )
                .result()
                .map_err(|error| format!("CUSOLVER orgqr workspace query failed: {error:?}"))?;
            }
        }
    }
    let workspace = usize::try_from(workspace[0].max(workspace[1]))
        .map_err(|_| "CUSOLVER returned a negative workspace size".to_string())?;
    let workspace_i32 = as_i32(workspace, "workspace size")?;
    let mut scratch = alloc_real(workspace)?;
    {
        let (factor_ptr, _factor_guard) = factor.device_ptr_mut(stream);
        let (tau_ptr, _tau_guard) = taus.device_ptr_mut(stream);
        let (info_ptr, _info_guard) = info.device_ptr_mut(stream);
        let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(stream);
        for index in 0..batch as u64 {
            // SAFETY: each pointer is offset by whole batch elements inside its buffer (`stride`
            // floats of matrix, `k` reflector scalars, one info word); the workspace holds the
            // queried size and is reused sequentially on the handle's stream, which also ordered
            // the load kernel that wrote `factor`; the guards outlive the loop.
            unsafe {
                (T::GEQRF)(
                    handle.cu(),
                    m_i32,
                    n_i32,
                    (factor_ptr + index * stride as u64 * element) as *mut T,
                    m_i32,
                    (tau_ptr + index * k as u64 * element) as *mut T,
                    scratch_ptr as *mut T,
                    workspace_i32,
                    (info_ptr + index * int) as *mut i32,
                )
                .result()
                .map_err(|error| format!("CUSOLVER geqrf failed: {error:?}"))?;
            }
        }
    }
    check_info(stream, &info, batch, "geqrf")?;
    let (batch_u64, m_u64, n_u64, k_u64, stride_u64, q_u64) = (
        as_u64(batch),
        as_u64(m),
        as_u64(n),
        as_u64(k),
        as_u64(stride),
        as_u64(q_columns),
    );
    {
        let kernel = linalg_kernel(module, "quabla_qr_signs")?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&factor);
        launch.arg(&mut signs);
        launch.arg(&batch_u64);
        launch.arg(&m_u64);
        launch.arg(&k_u64);
        launch.arg(&stride_u64);
        // SAFETY: the arguments match `quabla_qr_signs(const float*, float*, unsigned long long
        // x4)`; `factor` holds `batch` slots of `stride >= m * k` floats, `signs` `batch * k`
        // floats, and the kernel guards its index against `batch * k`.
        unsafe { launch.launch(linalg_launch_config(batch * k)?) }
            .map_err(|error| format!("failed to launch CUDA QR sign kernel: {error:?}"))?;
    }
    if !want_q {
        let kernel = linalg_kernel(module, "quabla_qr_r")?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&factor);
        launch.arg(&signs);
        launch.arg(output);
        launch.arg(&batch_u64);
        launch.arg(&m_u64);
        launch.arg(&n_u64);
        launch.arg(&k_u64);
        launch.arg(&stride_u64);
        // SAFETY: the arguments match `quabla_qr_r(const float*, const float*, float*, unsigned
        // long long x5)`; `factor` holds `batch` column-major `m x n` slots of `stride` floats,
        // `signs` `batch * k` floats, `output` `batch * k * n` floats (the `qr_r` node), and
        // the kernel guards its index against `batch * k * n`.
        return unsafe { launch.launch(linalg_launch_config(batch * k * n)?) }
            .map(|_| ())
            .map_err(|error| format!("failed to launch CUDA QR R kernel: {error:?}"));
    }
    {
        let (factor_ptr, _factor_guard) = factor.device_ptr_mut(stream);
        let (tau_ptr, _tau_guard) = taus.device_ptr_mut(stream);
        let (info_ptr, _info_guard) = info.device_ptr_mut(stream);
        let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(stream);
        for index in 0..batch as u64 {
            // SAFETY: as for `geqrf`; each slot holds `m * q_columns <= stride` floats, so
            // `orgqr` may overwrite all `q_columns` columns in place, and the sign kernel that
            // read the diagonal of `R` was ordered before this call on the same stream.
            unsafe {
                (T::ORGQR)(
                    handle.cu(),
                    m_i32,
                    q_i32,
                    k_i32,
                    (factor_ptr + index * stride as u64 * element) as *mut T,
                    m_i32,
                    (tau_ptr + index * k as u64 * element) as *const T,
                    scratch_ptr as *mut T,
                    workspace_i32,
                    (info_ptr + index * int) as *mut i32,
                )
                .result()
                .map_err(|error| format!("CUSOLVER orgqr failed: {error:?}"))?;
            }
        }
    }
    check_info(stream, &info, batch, "orgqr")?;
    let kernel = linalg_kernel(module, "quabla_qr_q")?;
    let mut launch = stream.launch_builder(&kernel);
    launch.arg(&factor);
    launch.arg(&signs);
    launch.arg(output);
    launch.arg(&batch_u64);
    launch.arg(&m_u64);
    launch.arg(&q_u64);
    launch.arg(&k_u64);
    launch.arg(&stride_u64);
    // SAFETY: the arguments match `quabla_qr_q(const float*, const float*, float*, unsigned long
    // long x5)`; `factor` holds `batch` column-major `m x q_columns` slots of `stride` floats,
    // `signs` `batch * k` floats, `output` `batch * m * q_columns` floats (the `qr_q` or
    // `qr_q_complete` node), and the kernel guards its index against that count.
    unsafe { launch.launch(linalg_launch_config(batch * m * q_columns)?) }
        .map(|_| ())
        .map_err(|error| format!("failed to launch CUDA QR Q kernel: {error:?}"))
}

/// The SVD by `gesvdj` (two-sided Jacobi, singular values sorted
/// descending), economical unless a full basis is requested.
fn launch_svd<T: CudaReal>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    solver: &Arc<Mutex<CudaSolver<T>>>,
    kind: LinalgKind,
    matrix: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    stack: MatrixStack,
) -> Result<(), String> {
    let name = kind.name();
    let MatrixStack { batch, m, n } = stack;
    let k = m.min(n);
    let full = matches!(kind, LinalgKind::SvdUFull | LinalgKind::SvdVhFull);
    let (u_columns, vh_rows) = if full { (m, n) } else { (k, k) };
    let mut factor = load_column_major(stream, module, matrix, stack, m * n)?;
    let alloc_real = |count: usize| {
        stream
            .alloc_zeros::<T>(count.max(1))
            .map_err(|error| format!("failed to allocate CUSOLVER {name} buffer: {error:?}"))
    };
    let mut values = alloc_real(batch * k)?;
    let mut left = alloc_real(batch * m * u_columns)?;
    let mut right = alloc_real(batch * n * vh_rows)?;
    let mut info = stream
        .alloc_zeros::<i32>(batch.max(1))
        .map_err(|error| format!("failed to allocate CUSOLVER {name} status: {error:?}"))?;
    let (m_i32, n_i32) = (as_i32(m, "SVD rows")?, as_i32(n, "SVD columns")?);
    let economy = i32::from(!full);
    let jobz = cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let solver = solver
        .lock()
        .map_err(|_| "CUSOLVER handle lock is poisoned".to_string())?;
    let handle = &solver.handle;
    let mut params: cusolver_sys::gesvdjInfo_t = std::ptr::null_mut();
    // SAFETY: `params` is a valid location for the created handle, which is destroyed below
    // on every path after its last use.
    unsafe { cusolver_sys::cusolverDnCreateGesvdjInfo(&mut params) }
        .result()
        .map_err(|error| format!("CUSOLVER gesvdj parameter creation failed: {error:?}"))?;
    let result = (|| {
        // SAFETY: `params` is the live parameter handle created above.
        unsafe { cusolver_sys::cusolverDnXgesvdjSetSortEig(params, 1) }
            .result()
            .map_err(|error| format!("CUSOLVER gesvdj sort setting failed: {error:?}"))?;
        let element = std::mem::size_of::<T>() as u64;
        let int = std::mem::size_of::<i32>() as u64;
        let mut workspace = 0_i32;
        let (factor_ptr, _factor_guard) = factor.device_ptr_mut(stream);
        let (values_ptr, _values_guard) = values.device_ptr_mut(stream);
        let (left_ptr, _left_guard) = left.device_ptr_mut(stream);
        let (right_ptr, _right_guard) = right.device_ptr_mut(stream);
        let (info_ptr, _info_guard) = info.device_ptr_mut(stream);
        // SAFETY: the live handle owns this stream; the buffers hold at least one column-major
        // `m x n` input (`lda = m`), `k` values, an `m x u_columns` `U` (`ldu = m`), and an
        // `n x vh_rows` `V` (`ldv = n`); `workspace` is a valid host output location.
        unsafe {
            (T::GESVDJ_BUFFER_SIZE)(
                handle.cu(),
                jobz,
                economy,
                m_i32,
                n_i32,
                factor_ptr as *const T,
                m_i32,
                values_ptr as *const T,
                left_ptr as *const T,
                m_i32,
                right_ptr as *const T,
                n_i32,
                &mut workspace,
                params,
            )
        }
        .result()
        .map_err(|error| format!("CUSOLVER gesvdj workspace query failed: {error:?}"))?;
        let workspace_len = usize::try_from(workspace)
            .map_err(|_| "CUSOLVER returned a negative workspace size".to_string())?;
        let mut scratch = alloc_real(workspace_len)?;
        let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(stream);
        for index in 0..batch as u64 {
            // SAFETY: each pointer is offset by whole batch elements inside its buffer (sizes as
            // in the workspace query); the workspace holds the queried size and is reused
            // sequentially on the handle's stream, which also ordered the load kernel; the
            // guards outlive the loop.
            unsafe {
                (T::GESVDJ)(
                    handle.cu(),
                    jobz,
                    economy,
                    m_i32,
                    n_i32,
                    (factor_ptr + index * (m * n) as u64 * element) as *mut T,
                    m_i32,
                    (values_ptr + index * k as u64 * element) as *mut T,
                    (left_ptr + index * (m * u_columns) as u64 * element) as *mut T,
                    m_i32,
                    (right_ptr + index * (n * vh_rows) as u64 * element) as *mut T,
                    n_i32,
                    scratch_ptr as *mut T,
                    workspace,
                    (info_ptr + index * int) as *mut i32,
                    params,
                )
            }
            .result()
            .map_err(|error| format!("CUSOLVER gesvdj failed: {error:?}"))?;
        }
        Ok::<_, String>(())
    })();
    // SAFETY: `params` was created above and is not used after this call.
    let destroyed = unsafe { cusolver_sys::cusolverDnDestroyGesvdjInfo(params) }.result();
    result?;
    destroyed.map_err(|error| format!("CUSOLVER gesvdj parameter release failed: {error:?}"))?;
    drop(solver);
    check_info(stream, &info, batch, "gesvdj")?;
    if kind == LinalgKind::SvdS {
        return stream
            .memcpy_dtod(&values, output)
            .map_err(|error| format!("failed to copy CUDA singular values: {error:?}"));
    }
    // `gesvdj` returns `V` column-major, which is `Vh` row-major; `U` is
    // column-major and is transposed into place.
    let mut u = alloc_real(batch * m * u_columns)?;
    launch_cuda_transpose_copy(stream, module, &left, &mut u, batch, u_columns, m)?;
    let kernel = linalg_kernel(module, "quabla_svd_signs")?;
    let mut launch = stream.launch_builder(&kernel);
    let (batch_u64, m_u64, u_u64, n_u64, vh_u64, k_u64) = (
        as_u64(batch),
        as_u64(m),
        as_u64(u_columns),
        as_u64(n),
        as_u64(vh_rows),
        as_u64(k),
    );
    launch.arg(&mut u);
    launch.arg(&mut right);
    launch.arg(&batch_u64);
    launch.arg(&m_u64);
    launch.arg(&u_u64);
    launch.arg(&n_u64);
    launch.arg(&vh_u64);
    launch.arg(&k_u64);
    // SAFETY: the arguments match `quabla_svd_signs(float*, float*, unsigned long long x6)`; `u`
    // holds `batch` row-major `m x u_columns` matrices and `right` `batch` row-major
    // `vh_rows x n` matrices, and each thread (guarded against `batch * k`) touches only its own
    // column of `U` and row of `Vh`.
    unsafe { launch.launch(linalg_launch_config(batch * k)?) }
        .map_err(|error| format!("failed to launch CUDA SVD sign kernel: {error:?}"))?;
    let source = if matches!(kind, LinalgKind::SvdU | LinalgKind::SvdUFull) {
        &u
    } else {
        &right
    };
    stream
        .memcpy_dtod(source, output)
        .map_err(|error| format!("failed to copy CUDA {name} output: {error:?}"))
}
