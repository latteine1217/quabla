//! Array interop for the eager `Tensor`: construction from nested Python
//! sequences and from any buffer-protocol object, conversion to nested lists
//! and Python scalars, and a read-only buffer export.
//!
//! The bridge never links against NumPy (design D2 in
//! `docs/api_v0_2_design.md`): NumPy arrays are read through the buffer
//! protocol, and `Tensor.numpy()`/`Tensor.__array__` import NumPy lazily and
//! hand it the buffer export. Importing copies into owned dtype-sized storage,
//! so a tensor never aliases foreign memory. Exports pin immutable storage
//! without copying for float64, float32, and bool.

use std::ffi::{c_int, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use pyo3::buffer::{Element, ElementType, PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyList, PyMemoryView, PyTuple};
use quabla_core::tensor_ir::{HostTensorStorage, TensorDType};

use crate::tensor::PyTensor;

fn is_nested_sequence(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>()
}

/// Builds a tensor from a Python scalar or rectangular nested lists/tuples of
/// scalars. The inferred dtype is `bool` when every element is a Python
/// `bool` and `float64` otherwise (integers included; there is no integer
/// dtype). An explicit `dtype` rounds the values to it.
pub(crate) fn tensor_from_nested(
    value: &Bound<'_, PyAny>,
    dtype: Option<TensorDType>,
) -> PyResult<PyTensor> {
    // The first element at each depth defines the shape; `fill_nested`
    // then checks that every other element agrees with it.
    let mut shape = Vec::new();
    let mut probe = value.clone();
    while is_nested_sequence(&probe) {
        let extent = probe.len()?;
        shape.push(extent);
        if extent == 0 {
            return Err(PyValueError::new_err(
                "cannot build a tensor from an empty sequence: tensor extents must be greater than zero",
            ));
        }
        probe = probe.get_item(0)?;
    }

    let mut data = Vec::with_capacity(shape.iter().product());
    let mut all_bool = true;
    fill_nested(value, &shape, 0, &mut data, &mut all_bool)?;
    let inferred = if all_bool {
        TensorDType::Bool
    } else {
        TensorDType::F64
    };
    PyTensor::from_shape_data_typed(shape, data, dtype.unwrap_or(inferred))
        .map_err(PyValueError::new_err)
}

fn fill_nested(
    value: &Bound<'_, PyAny>,
    shape: &[usize],
    depth: usize,
    data: &mut Vec<f64>,
    all_bool: &mut bool,
) -> PyResult<()> {
    let nested = is_nested_sequence(value);
    if depth == shape.len() {
        if nested {
            return Err(ragged_error(shape, depth));
        }
        if let Ok(flag) = value.cast::<PyBool>() {
            data.push(f64::from(flag.is_true()));
            return Ok(());
        }
        *all_bool = false;
        let number = value.extract::<f64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "cannot convert an element of type {} to a tensor value; nested sequences must \
                 hold Python numbers",
                value
                    .get_type()
                    .name()
                    .map_or_else(|_| "<unknown>".to_string(), |name| name.to_string())
            ))
        })?;
        data.push(number);
        return Ok(());
    }
    if !nested || value.len()? != shape[depth] {
        return Err(ragged_error(shape, depth));
    }
    for item in value.try_iter()? {
        fill_nested(&item?, shape, depth + 1, data, all_bool)?;
    }
    Ok(())
}

fn ragged_error(shape: &[usize], depth: usize) -> PyErr {
    PyValueError::new_err(format!(
        "nested sequence is ragged: the first elements imply shape {shape:?}, but an element at \
         depth {depth} does not match it"
    ))
}

/// Builds a tensor from a buffer-protocol object (NumPy arrays and scalars,
/// `memoryview`, `array.array`) of any layout, copying it in C order.
/// `float64`, `float32`, and `bool` keep their dtype; integers become
/// `float64`; `float16` and other formats are rejected. Returns `None` when
/// `value` does not implement the buffer protocol.
pub(crate) fn tensor_from_buffer(
    value: &Bound<'_, PyAny>,
    dtype: Option<TensorDType>,
) -> PyResult<Option<PyTensor>> {
    let py = value.py();
    let view = match PyMemoryView::from(value) {
        Ok(view) => view,
        Err(error) if error.is_instance_of::<PyTypeError>(py) => return Ok(None),
        Err(error) => return Err(error),
    };
    // Rank-0 exporters (NumPy scalars and 0-d arrays) leave `shape` null,
    // which `PyUntypedBuffer` rejects; read them as one-element rank-1
    // views and restore the empty shape.
    let rank = view.getattr("ndim")?.extract::<usize>()?;
    let view = if rank == 0 {
        let format = view.getattr("format")?;
        view.call_method1("cast", ("B",))?
            .call_method1("cast", (format,))?
            .cast_into::<PyMemoryView>()?
    } else {
        view
    };
    let buffer = PyUntypedBuffer::get(&view)?;
    let shape = if rank == 0 {
        Vec::new()
    } else {
        buffer.shape().to_vec()
    };
    let format = buffer.format().to_owned();
    if !is_native_byte_order(&format) {
        return Err(PyTypeError::new_err(format!(
            "buffer element format {format:?} is not in native byte order; convert it first, \
             for example with numpy.asarray(a, dtype=a.dtype.newbyteorder(\"=\"))"
        )));
    }
    if matches!(
        ElementType::from_format(&format),
        ElementType::Float { bytes: 4 }
    ) {
        let typed = buffer.into_typed::<f32>()?;
        // Preserve the old widening/rounding NaN policy without an F64 array temporary.
        let round = |value: f32| TensorDType::F32.round(f64::from(value)) as f32;
        let values = if let Some(values) = typed.as_slice(py) {
            values.iter().map(|value| round(value.get())).collect()
        } else {
            typed.to_vec(py)?.into_iter().map(round).collect()
        };
        let storage = quabla_core::tensor_ir::HostTensorStorage::F32(Arc::new(values))
            .into_dtype(dtype.unwrap_or(TensorDType::F32));
        let tensor = quabla_core::tensor_ir::DynamicTensor::from_storage(shape, storage)
            .map_err(PyValueError::new_err)?;
        return PyTensor::from_dynamic_tensor(tensor)
            .map(Some)
            .map_err(PyValueError::new_err);
    }
    if ElementType::from_format(&format) == ElementType::Bool {
        drop(buffer);
        let bytes = view.call_method0("tobytes")?;
        let values = bytes
            .cast::<PyBytes>()?
            .as_bytes()
            .iter()
            .map(|byte| u8::from(*byte != 0))
            .collect();
        let storage = quabla_core::tensor_ir::HostTensorStorage::Bool(Arc::new(values))
            .into_dtype(dtype.unwrap_or(TensorDType::Bool));
        let tensor = quabla_core::tensor_ir::DynamicTensor::from_storage(shape, storage)
            .map_err(PyValueError::new_err)?;
        return PyTensor::from_dynamic_tensor(tensor)
            .map(Some)
            .map_err(PyValueError::new_err);
    }
    let target = dtype.unwrap_or(TensorDType::F64);
    let storage = match ElementType::from_format(&format) {
        ElementType::Float { bytes: 8 } if target == TensorDType::F64 => {
            HostTensorStorage::F64(Arc::new(buffer.into_typed::<f64>()?.to_vec(py)?))
        }
        ElementType::Float { bytes: 8 } => buffer_storage::<f64>(py, buffer, |v| v, target)?,
        ElementType::SignedInteger { bytes: 1 } => {
            buffer_storage::<i8>(py, buffer, f64::from, target)?
        }
        ElementType::SignedInteger { bytes: 2 } => {
            buffer_storage::<i16>(py, buffer, f64::from, target)?
        }
        ElementType::SignedInteger { bytes: 4 } => {
            buffer_storage::<i32>(py, buffer, f64::from, target)?
        }
        // Retain the historical F64 rounding before the requested dtype conversion.
        ElementType::SignedInteger { bytes: 8 } => {
            buffer_storage::<i64>(py, buffer, |v| v as f64, target)?
        }
        ElementType::UnsignedInteger { bytes: 1 } => {
            buffer_storage::<u8>(py, buffer, f64::from, target)?
        }
        ElementType::UnsignedInteger { bytes: 2 } => {
            buffer_storage::<u16>(py, buffer, f64::from, target)?
        }
        ElementType::UnsignedInteger { bytes: 4 } => {
            buffer_storage::<u32>(py, buffer, f64::from, target)?
        }
        ElementType::UnsignedInteger { bytes: 8 } => {
            buffer_storage::<u64>(py, buffer, |v| v as f64, target)?
        }
        ElementType::Float { bytes: 2 } => {
            return Err(PyTypeError::new_err(
                "float16 data is not supported; convert it to float32 or float64 first",
            ))
        }
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unsupported buffer element format {format:?}; quabla accepts float64, float32, \
                 bool, and integer data in native byte order"
            )))
        }
    };
    let tensor = quabla_core::tensor_ir::DynamicTensor::from_storage(shape, storage)
        .map_err(PyValueError::new_err)?;
    PyTensor::from_dynamic_tensor(tensor)
        .map(Some)
        .map_err(PyValueError::new_err)
}

/// Whether a `struct` format's byte-order prefix, if any, is the host's.
/// Checked here because PyO3's element check accepts `>` (big-endian) on
/// little-endian hosts, which would reinterpret the bytes silently.
fn is_native_byte_order(format: &CStr) -> bool {
    match format.to_bytes() {
        [b'@' | b'=', _] => true,
        [b'<', _] => cfg!(target_endian = "little"),
        [b'>' | b'!', _] => cfg!(target_endian = "big"),
        _ => true,
    }
}

/// Copies into the requested physical storage, preserving conversion through F64
/// per element without allocating a full widened array for narrow output dtypes.
fn buffer_storage<T: Element>(
    py: Python<'_>,
    buffer: PyUntypedBuffer,
    convert: impl Fn(T) -> f64,
    dtype: TensorDType,
) -> PyResult<HostTensorStorage> {
    let typed: PyBuffer<T> = buffer.into_typed()?;
    // Contiguous exporters copy straight into final storage. Strided exporters
    // require a C-order source copy, but never an additional widened F64 array.
    if let Some(values) = typed.as_slice(py) {
        return Ok(converted_storage(
            values.iter().map(|value| value.get()),
            convert,
            dtype,
        ));
    }
    Ok(converted_storage(
        typed.to_vec(py)?.into_iter(),
        convert,
        dtype,
    ))
}

fn converted_storage<T>(
    values: impl Iterator<Item = T>,
    convert: impl Fn(T) -> f64,
    dtype: TensorDType,
) -> HostTensorStorage {
    match dtype {
        TensorDType::F64 => HostTensorStorage::F64(Arc::new(values.map(convert).collect())),
        TensorDType::F32 => {
            HostTensorStorage::F32(Arc::new(values.map(|v| convert(v) as f32).collect()))
        }
        TensorDType::Bool => HostTensorStorage::Bool(Arc::new(
            values.map(|v| u8::from(convert(v) != 0.0)).collect(),
        )),
    }
}

/// A Python scalar of the tensor dtype: `bool` for `bool` tensors, `float`
/// otherwise.
fn python_scalar(py: Python<'_>, value: f64, dtype: TensorDType) -> Bound<'_, PyAny> {
    match dtype {
        TensorDType::Bool => PyBool::new(py, value != 0.0).to_owned().into_any(),
        TensorDType::F32 | TensorDType::F64 => PyFloat::new(py, value).into_any(),
    }
}

/// Nested Python lists in C order; a rank-0 tensor gives a bare scalar, as
/// `numpy.ndarray.tolist` does.
pub(crate) fn nested_list<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: &[f64],
    dtype: TensorDType,
) -> PyResult<Bound<'py, PyAny>> {
    let Some((extent, inner)) = shape.split_first() else {
        return Ok(python_scalar(py, data[0], dtype));
    };
    let chunk = data.len() / extent;
    let items = data
        .chunks(chunk)
        .map(|chunk| nested_list(py, inner, chunk, dtype))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

/// The element of a single-element tensor as a Python scalar.
pub(crate) fn single_item<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: &[f64],
    dtype: TensorDType,
) -> PyResult<Bound<'py, PyAny>> {
    match data {
        [value] => Ok(python_scalar(py, *value, dtype)),
        _ => Err(PyValueError::new_err(format!(
            "item() requires a tensor with exactly one element, got shape {shape:?}"
        ))),
    }
}

/// An export pins the immutable physical storage for every dtype.
enum ExportStorage {
    F64(Arc<Vec<f64>>),
    F32(Arc<Vec<f32>>),
    Bool(Arc<Vec<u8>>),
}

/// Everything a filled `Py_buffer` points into. It lives in `view.internal`
/// from `__getbuffer__` until `__releasebuffer__`, so the data, shape, and
/// strides pointers stay valid for the whole export regardless of what
/// happens to the exporting `Tensor` object.
pub(crate) struct BufferExport {
    storage: ExportStorage,
    shape: Vec<isize>,
    strides: Vec<isize>,
}

impl BufferExport {
    pub(crate) fn new(
        shape: &[usize],
        storage: &quabla_core::tensor_ir::HostTensorStorage,
        dtype: TensorDType,
    ) -> Self {
        use quabla_core::tensor_ir::HostTensorStorage;
        assert_eq!(storage.dtype(), dtype);
        let storage = match storage {
            HostTensorStorage::F64(values) => ExportStorage::F64(Arc::clone(values)),
            HostTensorStorage::F32(values) => ExportStorage::F32(Arc::clone(values)),
            HostTensorStorage::Bool(values) => ExportStorage::Bool(Arc::clone(values)),
        };
        let itemsize = storage.itemsize();
        let shape = shape
            .iter()
            .map(|extent| *extent as isize)
            .collect::<Vec<_>>();
        // C-contiguous strides in bytes.
        let mut strides = vec![itemsize; shape.len()];
        for axis in (1..shape.len()).rev() {
            strides[axis - 1] = strides[axis] * shape[axis];
        }
        Self {
            storage,
            shape,
            strides,
        }
    }

    fn data_ptr(&self) -> *const c_void {
        match &self.storage {
            ExportStorage::F64(values) => values.as_ptr().cast(),
            ExportStorage::F32(values) => values.as_ptr().cast(),
            ExportStorage::Bool(values) => values.as_ptr().cast(),
        }
    }

    fn element_count(&self) -> usize {
        match &self.storage {
            ExportStorage::F64(values) => values.len(),
            ExportStorage::F32(values) => values.len(),
            ExportStorage::Bool(values) => values.len(),
        }
    }

    fn format(&self) -> &'static CStr {
        match self.storage {
            ExportStorage::F64(_) => c"d",
            ExportStorage::F32(_) => c"f",
            ExportStorage::Bool(_) => c"?",
        }
    }
}

impl ExportStorage {
    fn itemsize(&self) -> isize {
        match self {
            ExportStorage::F64(_) => size_of::<f64>() as isize,
            ExportStorage::F32(_) => size_of::<f32>() as isize,
            ExportStorage::Bool(_) => size_of::<u8>() as isize,
        }
    }
}

/// Fills `view` with a read-only, C-contiguous export and moves `export`
/// into `view.internal`; `release_buffer` frees it.
///
/// # Safety
///
/// `view` must be null or point to a `Py_buffer` that CPython passed to
/// `bf_getbuffer`, and `owner` must be the exporting object.
pub(crate) unsafe fn fill_buffer(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    export: BufferExport,
    owner: Bound<'_, PyAny>,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("buffer view pointer is null"));
    }
    if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err(
            "Tensor buffers are read-only; use numpy.array(tensor) for a writable copy",
        ));
    }
    let non_trivial_axes = export.shape.iter().filter(|extent| **extent > 1).count();
    if flags & ffi::PyBUF_F_CONTIGUOUS == ffi::PyBUF_F_CONTIGUOUS && non_trivial_axes > 1 {
        return Err(PyBufferError::new_err(
            "Tensor buffers are C-contiguous, not Fortran-contiguous",
        ));
    }

    let export = Box::new(export);
    let itemsize = export.storage.itemsize();
    let len = export.element_count() as isize * itemsize;
    let requested_shape = flags & ffi::PyBUF_ND == ffi::PyBUF_ND;
    let requested_strides = flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES;
    let requested_format = flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT;
    let ndim = if requested_shape {
        export.shape.len() as c_int
    } else {
        1
    };
    let format = if requested_format {
        export.format().as_ptr().cast_mut()
    } else {
        ptr::null_mut()
    };
    let buf = export.data_ptr().cast_mut();
    // The boxed vectors do not move when the box is turned into a raw
    // pointer, so these pointers stay valid until `release_buffer`.
    let shape = if requested_shape {
        export.shape.as_ptr().cast_mut()
    } else {
        ptr::null_mut()
    };
    let strides = if requested_strides {
        export.strides.as_ptr().cast_mut()
    } else {
        ptr::null_mut()
    };

    // SAFETY: `view` is non-null and, per this function's contract, points
    // to a `Py_buffer` owned by the caller of `bf_getbuffer` for the duration
    // of the export. `buf`, `shape`, and `strides` point into heap
    // allocations owned by `export`, which moves into `internal` and is
    // freed only by `release_buffer`. `format` points to a `'static` C
    // string that consumers never write through. `readonly = 1` and the
    // `PyBUF_WRITABLE` rejection above keep consumers from writing through
    // `buf`, so the shared typed storage stays immutable. `obj` takes a
    // strong reference that `PyBuffer_Release` drops.
    unsafe {
        (*view).buf = buf;
        (*view).obj = owner.into_ptr();
        (*view).len = len;
        (*view).itemsize = itemsize;
        (*view).readonly = 1;
        (*view).ndim = ndim;
        (*view).format = format;
        (*view).shape = shape;
        (*view).strides = strides;
        (*view).suboffsets = ptr::null_mut();
        (*view).internal = Box::into_raw(export).cast();
    }
    Ok(())
}

/// Frees the `BufferExport` that `fill_buffer` stored in `view.internal`.
///
/// # Safety
///
/// `view` must be a `Py_buffer` filled by `fill_buffer` whose export has not
/// been released yet; CPython calls `bf_releasebuffer` exactly once per
/// successful `bf_getbuffer`.
pub(crate) unsafe fn release_buffer(view: *mut ffi::Py_buffer) {
    // SAFETY: per the contract, `view` is valid and `internal` is either
    // null or the pointer produced by `Box::into_raw` in `fill_buffer` and
    // not freed since; it is reset to null so it cannot be freed twice.
    unsafe {
        let internal = (*view).internal.cast::<BufferExport>();
        if !internal.is_null() {
            (*view).internal = ptr::null_mut();
            drop(Box::from_raw(internal));
        }
    }
}

#[cfg(test)]
mod typed_export_tests {
    use super::*;
    use quabla_core::tensor_ir::HostTensorStorage;

    #[test]
    fn all_typed_exports_pin_original_storage_without_conversion() {
        for dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let storage = HostTensorStorage::from_f64(vec![0.0, 1.0, 2.0, 3.0], dtype);
            let pointer = match &storage {
                HostTensorStorage::F64(values) => values.as_ptr().cast::<c_void>(),
                HostTensorStorage::F32(values) => values.as_ptr().cast::<c_void>(),
                HostTensorStorage::Bool(values) => values.as_ptr().cast::<c_void>(),
            };
            let export = BufferExport::new(&[2, 2], &storage, dtype);
            assert_eq!(export.data_ptr(), pointer);
            assert_eq!(export.element_count(), 4);
            let (format, itemsize) = match dtype {
                TensorDType::F64 => (c"d", 8),
                TensorDType::F32 => (c"f", 4),
                TensorDType::Bool => (c"?", 1),
            };
            assert_eq!(export.format(), format);
            assert_eq!(export.strides, [2 * itemsize, itemsize]);
            drop(storage);
            assert_eq!(export.data_ptr(), pointer);
            assert_eq!(export.element_count(), 4);
        }
    }
}
