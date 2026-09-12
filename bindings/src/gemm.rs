// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Sending one dense product to the BLAS that `numpy` already brought.
//!
//! Every install of this library has `numpy` in it, and every install of
//! `numpy` has a tuned BLAS in it -- OpenBLAS on the wheels, whatever the
//! distributor chose otherwise. On a single large `float32` or `float64`
//! product that BLAS beats the engine's own kernel by 1.6-2.3x, which is not a
//! gap a portable Rust kernel closes: it is per-architecture assembly, chosen
//! at load time from a table of them.
//!
//! Reaching it costs three array headers and one call. The engine's buffers are
//! already contiguous and row-major -- the layout invariant the whole engine is
//! built on -- which is exactly what an array header wants, so nothing is
//! copied, converted or allocated on either side, and `matmul` writes its
//! answer straight into the output buffer the engine allocated.
//!
//! `matmul` and not `PyArray_MatrixProduct2`, which is the C entry point behind
//! `dot` and looks like the cheaper way in. It is the wrong kernel: given an
//! `out` it takes a route that measured 1.7x slower than `matmul` on a thin-`k`
//! product (4096x128 @ 128x4096), which is the shape a wide linear layer
//! produces and the last one to hand to a slower path.
//!
//! Everything else stays where it is. Batched products already saturate the
//! thread pool with one whole matrix per worker and are within 5-45% of NumPy
//! or ahead of it; integers do not reach a BLAS at all, in either library.

use engine::ops::linalg::{Gemm, GemmProvider, Storage, set_gemm_provider};
use numpy::npyffi::{
    NPY_ARRAY_ALIGNED, NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_F_CONTIGUOUS, NPY_ARRAY_WRITEABLE,
    NpyTypes, PY_ARRAY_API, get_type_object, npy_intp,
};
use numpy::{Element, PyArrayDescrMethods};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyTuple;
use std::os::raw::c_void;

/// Multiply-accumulates below which the call costs more than it saves.
///
/// Three array headers, a ufunc dispatch and a return object measure at a
/// little over a microsecond. Under about 1.3e5 multiply-accumulates the
/// engine's kernel finishes inside that: measured on 48x48x48 squares the two
/// are within 20% of each other, and every size below that the engine wins
/// outright. So the provider declines and one branch is all it cost.
const MIN_FLOPS: usize = 1 << 17;

/// The shortest contraction worth handing over.
///
/// `k` is the multiply-accumulates each output element takes, so a small one
/// means a product bound by writing its answer rather than by computing it.
/// There NumPy's matmul has nothing to offer -- measured 1.4-2.9x *slower*
/// than the engine's kernel at `k` of 8 or 16 on a 512x512 output, and slower
/// still on `k = 1`, the outer product. The engine's kernel writes the output
/// once from its own thread pool, which is all such a product can be.
///
/// Above it the ordering reverses and stays reversed: 1.2-1.9x for the engine
/// on square products, and 8-58x on a matrix-vector product, where the
/// blocked kernel is at its worst and a BLAS drops to a `gemv` written for
/// exactly that shape.
const MIN_K: usize = 32;

/// `numpy.matmul`, looked up once.
///
/// A module attribute lookup per product would be two dictionary probes on the
/// path this exists to shorten, and the answer cannot change: rebinding
/// `numpy.matmul` mid-process is not something a caller does by accident and
/// not something the engine should follow if they do.
static MATMUL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A dense GEMM through NumPy's C-API, on memory the engine owns.
struct NumpyGemm;

/// One `[rows, cols]` array header over `data`, owning nothing.
///
/// `storage` says which way the two axes run in memory. A `Transposed` operand
/// is a C-contiguous `[cols, rows]` block, which is the same thing as an
/// F-contiguous `[rows, cols]` one -- so the header describes it exactly and
/// nothing is copied to make it readable.
///
/// The flags are stated rather than left to be derived: a wrong contiguity
/// flag is the kind of thing that shows up as a silent 2x on one shape, and
/// for a two-dimensional block both answers are known here.
///
/// # Safety
///
/// `data` must point at `rows * cols` initialised `T`, stay valid and stay put
/// for as long as the returned object lives, and be aligned for `T`. The header
/// is created without `NPY_ARRAY_OWNDATA` and with no base object, so NumPy
/// reads and writes the buffer and never frees it -- the caller's borrow is
/// what keeps it alive, and the returned `Bound` is dropped before that borrow
/// ends.
unsafe fn header<'py, T: Element>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    storage: Storage,
    data: *mut T,
    writeable: bool,
) -> Option<Bound<'py, PyAny>> {
    let item = size_of::<T>() as npy_intp;
    let mut dims = [rows as npy_intp, cols as npy_intp];
    let (mut strides, contiguity) = match storage {
        Storage::RowMajor => ([cols as npy_intp * item, item], NPY_ARRAY_C_CONTIGUOUS),
        Storage::Transposed => ([item, rows as npy_intp * item], NPY_ARRAY_F_CONTIGUOUS),
    };
    let mut flags = contiguity | NPY_ARRAY_ALIGNED;
    if writeable {
        flags |= NPY_ARRAY_WRITEABLE;
    }
    let ptr = unsafe {
        PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            get_type_object(py, NpyTypes::PyArray_Type),
            T::get_dtype(py).into_dtype_ptr(),
            2,
            dims.as_mut_ptr(),
            strides.as_mut_ptr(),
            data.cast::<c_void>(),
            flags,
            std::ptr::null_mut(),
        )
    };
    (!ptr.is_null()).then(|| unsafe { Bound::from_owned_ptr(py, ptr) })
}

/// `out = lhs @ rhs` through NumPy, or `false` with `out` untouched.
///
/// # Safety
///
/// Nothing unsafe escapes: the three headers borrow the request's slices and
/// are dropped inside this call. `PyArray_MatrixProduct2` writes only into
/// `out`, whose header is the only writeable one.
fn matrix_product<T: Element + Zero + Copy>(request: &mut Gemm<'_, T>) -> bool {
    if request.flops() < MIN_FLOPS || request.k < MIN_K {
        return false;
    }

    Python::attach(|py| {
        let Some(matmul) = MATMUL
            .get_or_try_init(py, || -> PyResult<Py<PyAny>> {
                Ok(py.import("numpy")?.getattr("matmul")?.unbind())
            })
            .ok()
        else {
            // No `numpy` to reach, which is not an error here -- the engine
            // has its own kernel and this was only ever an offer.
            unsafe { ffi::PyErr_Clear() };
            return false;
        };

        let (Some(lhs), Some(rhs), Some(out)) = (
            unsafe {
                header(
                    py,
                    request.m,
                    request.k,
                    request.lhs_storage,
                    request.lhs.as_ptr().cast_mut(),
                    false,
                )
            },
            unsafe {
                header(
                    py,
                    request.k,
                    request.n,
                    request.rhs_storage,
                    request.rhs.as_ptr().cast_mut(),
                    false,
                )
            },
            unsafe {
                header(
                    py,
                    request.m,
                    request.n,
                    Storage::RowMajor,
                    request.out.as_mut_ptr(),
                    true,
                )
            },
        ) else {
            // Only a header allocation can have failed here, which means the
            // interpreter is out of memory. Leave the error for whatever asks
            // Python next and let the engine's kernel produce the answer.
            unsafe { ffi::PyErr_Clear() };
            return false;
        };

        // The destination goes third and positionally: a ufunc takes its
        // outputs that way, so this is one tuple rather than a tuple and a
        // keyword dictionary.
        let arguments = PyTuple::new(py, [lhs, rhs, out]);
        let called = arguments.and_then(|arguments| matmul.call1(py, arguments));
        if called.is_err() {
            // The engine's kernel writes into a zeroed buffer, and a failure
            // part way through leaves this one no longer zeroed.
            request.out.fill(T::zero());
            return false;
        }
        true
    })
}

impl GemmProvider for NumpyGemm {
    fn gemm_f32(&self, mut request: Gemm<'_, f32>) -> bool {
        matrix_product(&mut request)
    }

    fn gemm_f64(&self, mut request: Gemm<'_, f64>) -> bool {
        matrix_product(&mut request)
    }
}

/// A value of `T` to re-zero a partly written output with.
///
/// `Element` does not require `num_traits::Zero` and the two float types are
/// the only ones that reach here, so this is the whole of it.
trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

/// Point the engine's dense GEMM at NumPy, once, while the module loads.
pub fn install_gemm_provider() {
    set_gemm_provider(Box::new(NumpyGemm));
}
