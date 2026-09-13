// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Importing a NumPy array's memory rather than a copy of it, and exporting
//! tensor memory through the buffer protocol.
//!
//! The export side already had a path: `__array_interface__` hands out the
//! address and NumPy builds a read-only view over it. This adds the other two
//! halves of the same idea.
//!
//! **Import.** `from_numpy_shared` copied, and said so in a comment that had
//! outlived its excuse. Sharing needs somewhere to keep the array alive for as
//! long as any tensor reads it, which is what `TensorData::from_foreign` and
//! its type-erased owner are for: the owner here is a `Py<PyAny>` holding the
//! array, and dropping the last tensor drops the reference.
//!
//! **Export.** The buffer protocol reaches everything that predates or
//! declines `__array_interface__` -- `memoryview`, `numpy.frombuffer`, an
//! audio or image library taking raw frames. Read-only, for exactly the reason
//! the array interface is read-only: several tensors can share one buffer, and
//! a writer would change all of them at once.

use crate::tensor::PyTensor;
use engine::tensor::{DataType, Shape, TensorData};
use engine::{Device, Tensor};
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

/// Our dtype for a NumPy type number, if we have one.
fn dtype_from_type_num(num: c_int) -> Option<DataType> {
    use numpy::npyffi::NPY_TYPES;
    // `NPY_LONG` is 32-bit on Windows and 64-bit elsewhere, so it is resolved
    // by width rather than by name.
    let long_is_64 = size_of::<std::os::raw::c_long>() == 8;
    match num {
        x if x == NPY_TYPES::NPY_FLOAT as c_int => Some(DataType::Float32),
        x if x == NPY_TYPES::NPY_DOUBLE as c_int => Some(DataType::Float64),
        x if x == NPY_TYPES::NPY_INT as c_int => Some(DataType::Int32),
        x if x == NPY_TYPES::NPY_LONGLONG as c_int => Some(DataType::Int64),
        x if x == NPY_TYPES::NPY_LONG as c_int => Some(if long_is_64 {
            DataType::Int64
        } else {
            DataType::Int32
        }),
        x if x == NPY_TYPES::NPY_BOOL as c_int => Some(DataType::Bool),
        _ => None,
    }
}

/// A tensor over a NumPy array's own memory.
///
/// Every requirement below is checked and refused rather than worked around,
/// because the one thing a caller who asked to share must not get is a silent
/// copy: they asked in order to see each other's writes, and a copy answers
/// the call and not the question. `from_numpy` is the copying one and takes
/// anything convertible.
pub(crate) fn tensor_from_array_shared(
    array: &Bound<PyAny>,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let untyped: &Bound<PyUntypedArray> = array
        .cast()
        .map_err(|_| PyValueError::new_err("from_numpy_shared expects a NumPy array"))?;

    if !untyped.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "from_numpy_shared needs a C-contiguous array; the engine reads tensor storage in \
             contiguous order. Use numpy.ascontiguousarray, or from_numpy to copy",
        ));
    }

    let descr = untyped.dtype();
    let dtype = dtype_from_type_num(descr.num()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "from_numpy_shared cannot share dtype {descr}; minitensor stores float32, float64, \
             int32, int64 and bool. Use from_numpy to convert"
        ))
    })?;
    // `None` means byte order does not apply -- a single-byte type -- which is
    // fine. `Some(false)` is a byte-swapped array: our element width, not our
    // element bytes.
    if descr.is_native_byteorder() == Some(false) {
        return Err(PyValueError::new_err(
            "from_numpy_shared needs native byte order; use numpy.ndarray.byteswap, or \
             from_numpy to copy",
        ));
    }

    let shape: Vec<usize> = untyped.shape().to_vec();
    let numel: usize = shape.iter().product();
    let size = numel * dtype.size_in_bytes();
    // No accessor for this on `PyUntypedArrayMethods`; the field is what
    // `PyArray_DATA` reads in C.
    let data = unsafe { (*untyped.as_array_ptr()).data } as *mut u8;

    if size != 0 && !(data as usize).is_multiple_of(dtype.size_in_bytes()) {
        return Err(PyValueError::new_err(
            "from_numpy_shared needs an aligned array; use from_numpy to copy",
        ));
    }

    // One strong reference to the array, released when the last tensor sharing
    // it is dropped.
    let owner: Py<PyAny> = array.clone().unbind();
    // SAFETY: contiguity, dtype, byte order, alignment and length are checked
    // above, and `owner` holds the array -- and so its buffer -- alive for as
    // long as this storage exists. NumPy does not reallocate an array's data
    // in place, so the pointer stays good.
    //
    // The owner outlives this call and is dropped wherever the last tensor
    // sharing it is, which for a tensor freed inside a rayon fold is a thread
    // holding no GIL. That is sound rather than lucky: pyo3's `Drop for Py<T>`
    // checks whether the thread is attached and hands the reference to
    // `register_decref` when it is not, so the decref happens later on a
    // thread that is.
    let storage =
        unsafe { TensorData::from_foreign(data, size, dtype, numel, Box::new(owner) as Box<_>) };

    Ok(Tensor::new(
        Arc::new(storage),
        Shape::new(shape),
        dtype,
        Device::cpu(),
        requires_grad,
    ))
}

/// The `struct`-module code for one dtype, NUL-terminated for the C field.
fn buffer_format(dtype: DataType) -> &'static std::ffi::CStr {
    match dtype {
        DataType::Float32 => c"f",
        DataType::Float64 => c"d",
        DataType::Int32 => c"i",
        // `q` is `long long`, which `i64` is on every platform this builds
        // for; `l` would be 32-bit on Windows.
        DataType::Int64 => c"q",
        DataType::Bool => c"?",
    }
}

/// Shape and stride arrays owned for the lifetime of one exported buffer.
struct Bookkeeping {
    shape: Box<[ffi::Py_ssize_t]>,
    strides: Box<[ffi::Py_ssize_t]>,
}

/// Fill in `view` for `tensor`, exporting its buffer read-only.
///
/// # Safety
///
/// `view` must be a `Py_buffer` for CPython to fill in, which is what
/// `PyObject_GetBuffer` passes. `owner` must be the object being exported from,
/// and is what the view will hold a reference to.
pub(crate) unsafe fn fill_buffer_view(
    tensor: &Tensor,
    owner: Bound<'_, PyAny>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("Py_buffer must not be null"));
    }
    if flags & ffi::PyBUF_WRITABLE == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err(
            "minitensor tensors export read-only buffers, because several tensors can share one \
             and a writer would change all of them; use numpy_copy() for an array to write into",
        ));
    }
    if tensor.device() != Device::cpu() {
        return Err(PyBufferError::new_err(
            "a non-CPU tensor has no host buffer to export; call .cpu() first",
        ));
    }

    let storage = tensor.data();
    let itemsize = tensor.dtype().size_in_bytes();
    let dims = tensor.shape().dims();

    // `buf` is the storage pointer and `len` below is computed from the
    // *shape*, so the two agree only while every tensor's shape matches the
    // storage it was built over. Nothing in `Tensor::new` enforces that, and
    // the cost of it being false here is Python reading past the allocation --
    // the worst failure this file could have. One comparison per export, not
    // per element, turns it into an exception. No path in the library is known
    // to break it (the whole Python suite runs with the alignment and length
    // assertions in `TensorData` enabled and none fire); this is a guard on an
    // assumption, not a fix for an observed bug. `__array_interface__` hands
    // out the same pointer under the same assumption.
    if tensor.numel() != storage.numel() {
        return Err(PyBufferError::new_err(format!(
            "tensor shape covers {} elements but its storage holds {}; refusing to export a              buffer that would read past the allocation",
            tensor.numel(),
            storage.numel(),
        )));
    }

    // Shape and strides must outlive this call and be reclaimed on release, so
    // they are boxed and parked in `internal` -- the field the protocol
    // reserves for exactly this.
    let shape: Box<[ffi::Py_ssize_t]> = dims.iter().map(|&d| d as ffi::Py_ssize_t).collect();
    let mut stride = itemsize as ffi::Py_ssize_t;
    let mut strides: Vec<ffi::Py_ssize_t> = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        strides[axis] = stride;
        stride *= dims[axis] as ffi::Py_ssize_t;
    }
    let book = Box::new(Bookkeeping {
        shape,
        strides: strides.into_boxed_slice(),
    });

    // SAFETY: `view` is valid per this function's contract, and every field is
    // either set or left at the value CPython documents as "not provided".
    unsafe {
        (*view).buf = storage.as_ptr() as *mut c_void;
        (*view).obj = owner.into_ptr();
        (*view).len = (tensor.numel() * itemsize) as ffi::Py_ssize_t;
        (*view).readonly = 1;
        (*view).itemsize = itemsize as ffi::Py_ssize_t;
        (*view).format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT {
            buffer_format(tensor.dtype()).as_ptr() as *mut c_char
        } else {
            std::ptr::null_mut()
        };
        (*view).ndim = book.shape.len() as c_int;
        (*view).shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND {
            book.shape.as_ptr() as *mut ffi::Py_ssize_t
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
            book.strides.as_ptr() as *mut ffi::Py_ssize_t
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = Box::into_raw(book) as *mut c_void;
    }
    Ok(())
}

/// Reclaim what [`fill_buffer_view`] parked in `internal`.
///
/// # Safety
///
/// `view` must be one this module filled in; CPython guarantees that by
/// pairing every release with a successful export.
pub(crate) unsafe fn release_buffer_view(view: *mut ffi::Py_buffer) {
    if view.is_null() {
        return;
    }
    // SAFETY: `internal` is the `Box` leaked above, and this runs exactly once
    // per successful export, so the reclaim is not a double free.
    unsafe {
        let internal = (*view).internal as *mut Bookkeeping;
        if !internal.is_null() {
            drop(Box::from_raw(internal));
            (*view).internal = std::ptr::null_mut();
        }
    }
}

/// Build the shared tensor and wrap it, for the `from_numpy_shared` binding.
pub(crate) fn shared_pytensor(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(tensor_from_array_shared(
        array,
        requires_grad,
    )?))
}
