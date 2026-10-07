//! Element type of the floating device buffers of one CUDA plan.
//!
//! The default lowering keeps every buffer `f32` and runs logical `float64`
//! nodes in single precision. The opt-in double lowering
//! ([`crate::compiler::QuablaPrecision::Float64`]) instantiates the same plan with
//! `f64` buffers: the generated NVRTC program is rewritten to `double` by
//! [`double_precision_source`], and cuBLAS and cuSOLVER calls use their `D`
//! variants. Keeping one element type per plan (rather than per node) lets
//! every kernel, pool, and library call stay a single generic code path.

use std::borrow::Cow;
use std::ffi::c_int;
use std::fmt::Debug;

use cudarc::cublas::result::CublasError;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cusolver::sys::{
    self as cusolver_sys, cublasFillMode_t, cublasOperation_t, cusolverDnHandle_t,
    cusolverEigMode_t, cusolverStatus_t, gesvdjInfo_t,
};
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use super::super::{HostTensorStorage, TensorDType};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// cuSOLVER entry-point signatures, generic over the element type.
type GetrfBufferSizeFn<T> =
    unsafe fn(cusolverDnHandle_t, c_int, c_int, *mut T, c_int, *mut c_int) -> cusolverStatus_t;
type GetrfFn<T> = unsafe fn(
    cusolverDnHandle_t,
    c_int,
    c_int,
    *mut T,
    c_int,
    *mut T,
    *mut c_int,
    *mut c_int,
) -> cusolverStatus_t;
type GetrsFn<T> = unsafe fn(
    cusolverDnHandle_t,
    cublasOperation_t,
    c_int,
    c_int,
    *const T,
    c_int,
    *const c_int,
    *mut T,
    c_int,
    *mut c_int,
) -> cusolverStatus_t;
type SyevdBufferSizeFn<T> = unsafe fn(
    cusolverDnHandle_t,
    cusolverEigMode_t,
    cublasFillMode_t,
    c_int,
    *const T,
    c_int,
    *const T,
    *mut c_int,
) -> cusolverStatus_t;
type SyevdFn<T> = unsafe fn(
    cusolverDnHandle_t,
    cusolverEigMode_t,
    cublasFillMode_t,
    c_int,
    *mut T,
    c_int,
    *mut T,
    *mut T,
    c_int,
    *mut c_int,
) -> cusolverStatus_t;
type GeqrfFn<T> = unsafe fn(
    cusolverDnHandle_t,
    c_int,
    c_int,
    *mut T,
    c_int,
    *mut T,
    *mut T,
    c_int,
    *mut c_int,
) -> cusolverStatus_t;
type OrgqrBufferSizeFn<T> = unsafe fn(
    cusolverDnHandle_t,
    c_int,
    c_int,
    c_int,
    *const T,
    c_int,
    *const T,
    *mut c_int,
) -> cusolverStatus_t;
type OrgqrFn<T> = unsafe fn(
    cusolverDnHandle_t,
    c_int,
    c_int,
    c_int,
    *mut T,
    c_int,
    *const T,
    *mut T,
    c_int,
    *mut c_int,
) -> cusolverStatus_t;
type GesvdjBufferSizeFn<T> = unsafe fn(
    cusolverDnHandle_t,
    cusolverEigMode_t,
    c_int,
    c_int,
    c_int,
    *const T,
    c_int,
    *const T,
    *const T,
    c_int,
    *const T,
    c_int,
    *mut c_int,
    gesvdjInfo_t,
) -> cusolverStatus_t;
type GesvdjFn<T> = unsafe fn(
    cusolverDnHandle_t,
    cusolverEigMode_t,
    c_int,
    c_int,
    c_int,
    *mut T,
    c_int,
    *mut T,
    *mut T,
    c_int,
    *mut T,
    c_int,
    *mut T,
    c_int,
    *mut c_int,
    gesvdjInfo_t,
) -> cusolverStatus_t;

/// Element type of every floating (and `0`/`1` bool) buffer of one CUDA plan.
///
/// Sealed: only `f32` (the default) and `f64` (`precision="float64"`) exist.
/// The associated `fn` constants are the matching cuSOLVER entry points, so a
/// call site reads `(T::GETRF)(...)` for both precisions.
pub trait CudaReal:
    sealed::Sealed
    + DeviceRepr
    + ValidAsZeroBits
    + Copy
    + Default
    + Debug
    + PartialEq
    + Send
    + Sync
    + Unpin
    + 'static
{
    const GETRF_BUFFER_SIZE: GetrfBufferSizeFn<Self>;
    const GETRF: GetrfFn<Self>;
    const GETRS: GetrsFn<Self>;
    const SYEVD_BUFFER_SIZE: SyevdBufferSizeFn<Self>;
    const SYEVD: SyevdFn<Self>;
    const GEQRF_BUFFER_SIZE: GetrfBufferSizeFn<Self>;
    const GEQRF: GeqrfFn<Self>;
    const ORGQR_BUFFER_SIZE: OrgqrBufferSizeFn<Self>;
    const ORGQR: OrgqrFn<Self>;
    const GESVDJ_BUFFER_SIZE: GesvdjBufferSizeFn<Self>;
    const GESVDJ: GesvdjFn<Self>;

    fn from_f64(value: f64) -> Self;

    fn to_f64(self) -> f64;

    /// The values of a host input in this element type (bool as `0`/`1`).
    fn host_values(storage: &HostTensorStorage) -> Cow<'_, [Self]>;

    /// Wraps a device readback as host storage of the node's logical dtype.
    fn host_storage(values: Vec<Self>, dtype: TensorDType) -> HostTensorStorage;

    /// The NVRTC program for this element type, from the single-precision
    /// source every code generator emits.
    fn program_source(source: String) -> Result<String, String>;

    /// # Safety
    /// As [`Gemm::gemm`]: the configuration must describe valid regions of
    /// `a`, `b`, and `c` on the handle's stream.
    unsafe fn gemm(
        blas: &CudaBlas,
        config: GemmConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError>;

    /// # Safety
    /// As [`Gemm::gemm_strided_batched`].
    unsafe fn gemm_strided_batched(
        blas: &CudaBlas,
        config: StridedBatchedConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError>;
}

impl CudaReal for f32 {
    const GETRF_BUFFER_SIZE: GetrfBufferSizeFn<f32> = cusolver_sys::cusolverDnSgetrf_bufferSize;
    const GETRF: GetrfFn<f32> = cusolver_sys::cusolverDnSgetrf;
    const GETRS: GetrsFn<f32> = cusolver_sys::cusolverDnSgetrs;
    const SYEVD_BUFFER_SIZE: SyevdBufferSizeFn<f32> = cusolver_sys::cusolverDnSsyevd_bufferSize;
    const SYEVD: SyevdFn<f32> = cusolver_sys::cusolverDnSsyevd;
    const GEQRF_BUFFER_SIZE: GetrfBufferSizeFn<f32> = cusolver_sys::cusolverDnSgeqrf_bufferSize;
    const GEQRF: GeqrfFn<f32> = cusolver_sys::cusolverDnSgeqrf;
    const ORGQR_BUFFER_SIZE: OrgqrBufferSizeFn<f32> = cusolver_sys::cusolverDnSorgqr_bufferSize;
    const ORGQR: OrgqrFn<f32> = cusolver_sys::cusolverDnSorgqr;
    const GESVDJ_BUFFER_SIZE: GesvdjBufferSizeFn<f32> = cusolver_sys::cusolverDnSgesvdj_bufferSize;
    const GESVDJ: GesvdjFn<f32> = cusolver_sys::cusolverDnSgesvdj;

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn host_values(storage: &HostTensorStorage) -> Cow<'_, [Self]> {
        storage.to_f32()
    }

    fn host_storage(values: Vec<Self>, dtype: TensorDType) -> HostTensorStorage {
        HostTensorStorage::from_f32(values, dtype)
    }

    fn program_source(source: String) -> Result<String, String> {
        Ok(source)
    }

    unsafe fn gemm(
        blas: &CudaBlas,
        config: GemmConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError> {
        // SAFETY: forwarded under the caller's contract.
        unsafe { <CudaBlas as Gemm<f32>>::gemm(blas, config, a, b, c) }
    }

    unsafe fn gemm_strided_batched(
        blas: &CudaBlas,
        config: StridedBatchedConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError> {
        // SAFETY: forwarded under the caller's contract.
        unsafe { <CudaBlas as Gemm<f32>>::gemm_strided_batched(blas, config, a, b, c) }
    }
}

impl CudaReal for f64 {
    const GETRF_BUFFER_SIZE: GetrfBufferSizeFn<f64> = cusolver_sys::cusolverDnDgetrf_bufferSize;
    const GETRF: GetrfFn<f64> = cusolver_sys::cusolverDnDgetrf;
    const GETRS: GetrsFn<f64> = cusolver_sys::cusolverDnDgetrs;
    const SYEVD_BUFFER_SIZE: SyevdBufferSizeFn<f64> = cusolver_sys::cusolverDnDsyevd_bufferSize;
    const SYEVD: SyevdFn<f64> = cusolver_sys::cusolverDnDsyevd;
    const GEQRF_BUFFER_SIZE: GetrfBufferSizeFn<f64> = cusolver_sys::cusolverDnDgeqrf_bufferSize;
    const GEQRF: GeqrfFn<f64> = cusolver_sys::cusolverDnDgeqrf;
    const ORGQR_BUFFER_SIZE: OrgqrBufferSizeFn<f64> = cusolver_sys::cusolverDnDorgqr_bufferSize;
    const ORGQR: OrgqrFn<f64> = cusolver_sys::cusolverDnDorgqr;
    const GESVDJ_BUFFER_SIZE: GesvdjBufferSizeFn<f64> = cusolver_sys::cusolverDnDgesvdj_bufferSize;
    const GESVDJ: GesvdjFn<f64> = cusolver_sys::cusolverDnDgesvdj;

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn host_values(storage: &HostTensorStorage) -> Cow<'_, [Self]> {
        storage.to_f64()
    }

    fn host_storage(values: Vec<Self>, dtype: TensorDType) -> HostTensorStorage {
        HostTensorStorage::from_f64(values, dtype)
    }

    fn program_source(source: String) -> Result<String, String> {
        double_precision_source(&source)
    }

    unsafe fn gemm(
        blas: &CudaBlas,
        config: GemmConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError> {
        // SAFETY: forwarded under the caller's contract.
        unsafe { <CudaBlas as Gemm<f64>>::gemm(blas, config, a, b, c) }
    }

    unsafe fn gemm_strided_batched(
        blas: &CudaBlas,
        config: StridedBatchedConfig<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError> {
        // SAFETY: forwarded under the caller's contract.
        unsafe { <CudaBlas as Gemm<f64>>::gemm_strided_batched(blas, config, a, b, c) }
    }
}

/// Bit-pattern constants of the single-precision sources and their double
/// counterparts: quiet NaN, +inf, and `FLT_MAX` (the overflow guard of the
/// wide axis reduction, which becomes `DBL_MAX`).
const DOUBLE_CONSTANTS: [(&str, &str); 3] = [
    (
        "__int_as_float(0x7fc00000)",
        "__longlong_as_double(0x7ff8000000000000LL)",
    ),
    (
        "__int_as_float(0x7f800000)",
        "__longlong_as_double(0x7ff0000000000000LL)",
    ),
    ("3.4028234663852886e38f", "1.7976931348623157e308"),
];

/// Single-precision CUDA math functions whose double overload is the same
/// name without the trailing `f`.
const SUFFIXED_MATH: &[&str] = &[
    "acosf",
    "acoshf",
    "asinf",
    "asinhf",
    "atan2f",
    "atanf",
    "atanhf",
    "cbrtf",
    "ceilf",
    "copysignf",
    "cosf",
    "coshf",
    "cospif",
    "erfcf",
    "erfcinvf",
    "erfcxf",
    "erff",
    "erfinvf",
    "exp10f",
    "exp2f",
    "expf",
    "expm1f",
    "fabsf",
    "fdimf",
    "floorf",
    "fmaf",
    "fmaxf",
    "fminf",
    "fmodf",
    "frexpf",
    "hypotf",
    "ilogbf",
    "ldexpf",
    "lgammaf",
    "llrintf",
    "llroundf",
    "log10f",
    "log1pf",
    "log2f",
    "logbf",
    "logf",
    "lrintf",
    "lroundf",
    "nearbyintf",
    "nextafterf",
    "normcdff",
    "normcdfinvf",
    "powf",
    "rcbrtf",
    "remainderf",
    "rintf",
    "roundf",
    "rsqrtf",
    "sincosf",
    "sinf",
    "sinhf",
    "sinpif",
    "sqrtf",
    "tanf",
    "tanhf",
    "tgammaf",
    "truncf",
];

/// The double-precision spelling of one identifier of a generated
/// single-precision program, or `None` to keep it.
fn double_identifier(identifier: &str) -> Option<String> {
    if identifier == "float" {
        return Some("double".to_string());
    }
    if SUFFIXED_MATH.contains(&identifier) {
        return Some(identifier[..identifier.len() - 1].to_string());
    }
    // Correctly rounded arithmetic: `__fadd_rn` -> `__dadd_rn`, `__fmaf_rn` -> `__fma_rn`.
    for (single, double) in [
        ("__fadd_", "__dadd_"),
        ("__fsub_", "__dsub_"),
        ("__fmul_", "__dmul_"),
        ("__fdiv_", "__ddiv_"),
        ("__fsqrt_", "__dsqrt_"),
        ("__frcp_", "__drcp_"),
        ("__fmaf_", "__fma_"),
    ] {
        if let Some(mode) = identifier.strip_prefix(single) {
            if matches!(mode, "rn" | "rz" | "ru" | "rd") {
                return Some(format!("{double}{mode}"));
            }
        }
    }
    // Fast single-precision intrinsics become the accurate double functions.
    match identifier {
        "__expf" | "__logf" | "__sinf" | "__cosf" | "__tanf" | "__powf" | "__exp10f"
        | "__log2f" | "__log10f" => Some(identifier[2..identifier.len() - 1].to_string()),
        _ => None,
    }
}

/// Identifiers with no double counterpart under a plain rename; a program
/// that still contains one after the rewrite would mix precisions.
fn is_single_precision_only(identifier: &str) -> bool {
    matches!(
        identifier,
        "__int_as_float"
            | "__float_as_int"
            | "__float_as_uint"
            | "__uint_as_float"
            | "__saturatef"
            | "__fdividef"
            | "FLT_MAX"
            | "FLT_MIN"
            | "FLT_EPSILON"
            | "CUDART_INF_F"
            | "CUDART_NAN_F"
    ) || [
        "__float2",
        "__int2float",
        "__uint2float",
        "__ll2float",
        "__ull2float",
    ]
    .iter()
    .any(|prefix| identifier.starts_with(prefix))
        || (identifier.starts_with("float") || identifier.starts_with("make_float"))
            && identifier.ends_with(|c: char| c.is_ascii_digit())
}

/// Rewrites a generated single-precision NVRTC program to double precision.
///
/// Every code generator emits `float` sources; rewriting the finished
/// program keeps one generator per kernel instead of a type parameter on
/// each `format!`. The rewrite is lexical and complete for what the
/// generators emit: the `float` type, `f`-suffixed floating literals, the
/// `f`-suffixed math functions, the correctly rounded `__f*_r?` intrinsics,
/// and the bit-pattern constants in [`DOUBLE_CONSTANTS`]. Comments and
/// string literals are copied unchanged. Any single-precision-only
/// identifier left afterwards is an error rather than a silently `float`
/// lane inside a double program.
pub(crate) fn double_precision_source(source: &str) -> Result<String, String> {
    let mut source = source.to_string();
    for (single, double) in DOUBLE_CONSTANTS {
        source = source.replace(single, double);
    }
    let bytes = source.as_bytes();
    let mut output = String::with_capacity(source.len() + source.len() / 8);
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        let start = index;
        if byte == b'/' && next == Some(b'/') {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            output.push_str(&source[start..index]);
        } else if byte == b'/' && next == Some(b'*') {
            index += 2;
            while index < bytes.len() && !source[index..].starts_with("*/") {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
            output.push_str(&source[start..index]);
        } else if byte == b'"' || byte == b'\'' {
            index += 1;
            while index < bytes.len() && bytes[index] != byte {
                index += if bytes[index] == b'\\' { 2 } else { 1 };
            }
            index = (index + 1).min(bytes.len());
            output.push_str(&source[start..index]);
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let identifier = &source[start..index];
            match double_identifier(identifier) {
                Some(double) => output.push_str(&double),
                None if is_single_precision_only(identifier) => {
                    return Err(format!(
                        "CUDA float64 lowering cannot rewrite the single-precision identifier \
                         {identifier:?} of a generated kernel"
                    ));
                }
                None => output.push_str(identifier),
            }
        } else if byte.is_ascii_digit()
            || (byte == b'.' && next.is_some_and(|c| c.is_ascii_digit()))
        {
            index = numeric_literal_end(bytes, index);
            let literal = &source[start..index];
            let hexadecimal = literal.starts_with("0x") || literal.starts_with("0X");
            if !hexadecimal && (literal.ends_with('f') || literal.ends_with('F')) {
                output.push_str(&literal[..literal.len() - 1]);
            } else {
                output.push_str(literal);
            }
        } else {
            let character = source[index..].chars().next().expect("index is in bounds");
            index += character.len_utf8();
            output.push(character);
        }
    }
    Ok(output)
}

/// The end of the C numeric literal starting at `start`: digits, a decimal
/// point, a signed exponent (not for hexadecimal integers), and a suffix.
fn numeric_literal_end(bytes: &[u8], start: usize) -> usize {
    let hexadecimal = bytes[start] == b'0' && matches!(bytes.get(start + 1), Some(b'x' | b'X'));
    let mut index = start;
    while index < bytes.len() {
        let byte = bytes[index];
        let exponent_sign = !hexadecimal
            && matches!(byte, b'+' | b'-')
            && index > start
            && matches!(bytes[index - 1], b'e' | b'E');
        if byte.is_ascii_alphanumeric() || byte == b'.' || exponent_sign {
            index += 1;
        } else {
            break;
        }
    }
    index
}

#[cfg(test)]
mod tests {
    use super::double_precision_source;

    #[test]
    fn rewrites_types_literals_math_and_intrinsics() -> Result<(), String> {
        let single = "extern \"C\" __global__ void k(const float* x, float* out) {\n\
            // float stays in comments: expf\n\
            float a = expf(x[0]) + 1.5f + 2.0e-3f + .5f + 1e10F + 256U + 0x7fULL;\n\
            out[0] = __fadd_rn(a, __fmul_rn(erff(a), log1pf(a))) + (float)3;\n\
            out[1] = fabsf(a) > 3.4028234663852886e38f ? __int_as_float(0x7fc00000) : -__int_as_float(0x7f800000);\n\
            out[2] = addf(a, mulf(a, 1.0f));\n}";
        let double = double_precision_source(single)?;
        assert_eq!(
            double,
            "extern \"C\" __global__ void k(const double* x, double* out) {\n\
            // float stays in comments: expf\n\
            double a = exp(x[0]) + 1.5 + 2.0e-3 + .5 + 1e10 + 256U + 0x7fULL;\n\
            out[0] = __dadd_rn(a, __dmul_rn(erf(a), log1p(a))) + (double)3;\n\
            out[1] = fabs(a) > 1.7976931348623157e308 ? __longlong_as_double(0x7ff8000000000000LL) : -__longlong_as_double(0x7ff0000000000000LL);\n\
            out[2] = addf(a, mulf(a, 1.0));\n}"
        );
        Ok(())
    }

    #[test]
    fn rejects_identifiers_without_a_double_rename() {
        for single in [
            "float y = __float_as_int(x);",
            "float y = __int_as_float(1);",
            "float2 y;",
            "float y = FLT_EPSILON;",
            "float y = __fdividef(a, b);",
        ] {
            assert!(
                double_precision_source(single).is_err(),
                "{single:?} must be rejected"
            );
        }
    }
}
