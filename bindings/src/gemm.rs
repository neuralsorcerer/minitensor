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
use pyo3::types::{PyModule, PyTuple};
use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Multiply-accumulates below which the call costs more than it saves.
///
/// Three array headers, a ufunc dispatch and a return object measure at a
/// little over a microsecond. Under about 1.3e5 multiply-accumulates the
/// engine's kernel finishes inside that: measured on 48x48x48 squares the two
/// are within 20% of each other, and every size below that the engine wins
/// outright. So the provider declines and one branch is all it cost.
const DEFAULT_MIN_FLOPS: usize = 1 << 17;

/// The live thresholds, as atomics rather than constants.
///
/// Not because a caller is expected to tune them -- the defaults are measured,
/// and a wrong value here costs throughput on every product -- but because the
/// two paths have to be comparable *on one build* to be tested at all. The
/// question a differential test asks is whether the delegated answer equals
/// the native one for a given shape, and with the threshold compiled in there
/// is no way to ask it: the shapes above the threshold have no native answer to
/// compare against and the shapes below have no delegated one.
///
/// A relaxed atomic load is a plain load on every architecture this builds
/// for, so this is the same instruction the constant was, and it keeps the
/// provider itself a `OnceLock` with no lock on the path.
static MIN_FLOPS: AtomicUsize = AtomicUsize::new(DEFAULT_MIN_FLOPS);

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
const DEFAULT_MIN_K: usize = 32;

/// See [`MIN_FLOPS`] for why this is an atomic and not a constant.
static MIN_K: AtomicUsize = AtomicUsize::new(DEFAULT_MIN_K);

/// `numpy.matmul`, looked up once.
///
/// A module attribute lookup per product would be two dictionary probes on the
/// path this exists to shorten, and the answer cannot change: rebinding
/// `numpy.matmul` mid-process is not something a caller does by accident and
/// not something the engine should follow if they do.
static MATMUL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A dense GEMM through NumPy's C-API, on memory the engine owns.
struct NumpyGemm;

/// One `[batch, rows, cols]` array header over `data`, owning nothing.
///
/// `storage` says which way the two matrix axes run in memory. A `Transposed`
/// operand is a C-contiguous `[cols, rows]` block, which is the same thing as
/// an F-contiguous `[rows, cols]` one -- so the header describes it exactly and
/// nothing is copied to make it readable.
///
/// A `batch` of 1 still produces a three-dimensional header. `matmul` treats a
/// leading axis as a stack and gives the same answer either way, and one shape
/// here means one code path to get right rather than two.
///
/// The contiguity flags are stated rather than left to be derived: a wrong one
/// shows up as a silent 2x on one shape. They describe the *matrix* axes, and
/// with a batch in front only the row-major case is contiguous as a whole --
/// a stack of F-contiguous matrices is neither, so it is described by its
/// strides alone and both flags are left off. NumPy reads the strides; the
/// flags are an optimisation hint it is entitled to trust, so an honest
/// omission costs nothing and a wrong claim would cost correctness.
///
/// # Safety
///
/// `data` must point at `batch * rows * cols` initialised `T`, stay valid and
/// stay put for as long as the returned object lives, and be aligned for `T`.
/// The header is created without `NPY_ARRAY_OWNDATA` and with no base object,
/// so NumPy reads and writes the buffer and never frees it -- the caller's
/// borrow is what keeps it alive, and the returned `Bound` is dropped before
/// that borrow ends.
unsafe fn header<'py, T: Element>(
    py: Python<'py>,
    batch: usize,
    rows: usize,
    cols: usize,
    storage: Storage,
    data: *mut T,
    writeable: bool,
) -> Option<Bound<'py, PyAny>> {
    let item = size_of::<T>() as npy_intp;
    let matrix = (rows * cols) as npy_intp * item;
    let mut dims = [batch as npy_intp, rows as npy_intp, cols as npy_intp];
    let (mut strides, contiguity) = match storage {
        // A stack of row-major matrices packed end to end is itself
        // row-major, so this holds for any batch.
        Storage::RowMajor => (
            [matrix, cols as npy_intp * item, item],
            NPY_ARRAY_C_CONTIGUOUS,
        ),
        Storage::Transposed => (
            [matrix, item, rows as npy_intp * item],
            if batch == 1 {
                NPY_ARRAY_F_CONTIGUOUS
            } else {
                0
            },
        ),
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
            3,
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
    if request.flops() < MIN_FLOPS.load(Ordering::Relaxed)
        || request.k < MIN_K.load(Ordering::Relaxed)
    {
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
                    request.batch,
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
                    request.batch,
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
                    request.batch,
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

/// Where the boundary between the two GEMM paths currently sits.
///
/// `(min_flops, min_k)`: a product is offered to NumPy only when it has at
/// least `min_flops` multiply-accumulates and contracts over at least `min_k`.
#[pyfunction]
fn gemm_thresholds() -> (usize, usize) {
    (
        MIN_FLOPS.load(Ordering::Relaxed),
        MIN_K.load(Ordering::Relaxed),
    )
}

/// Move the boundary, returning where it was.
///
/// For tests and for measuring, not for tuning: the defaults come from the
/// numbers in this module's header, and a benchmark that wants to compare the
/// two paths is the reason this is reachable at all. `set_gemm_thresholds(0,
/// 0)` delegates every product it can; a very large pair delegates none, which
/// is how a differential test gets both answers out of one build.
#[pyfunction]
#[pyo3(signature = (min_flops, min_k))]
fn set_gemm_thresholds(min_flops: usize, min_k: usize) -> (usize, usize) {
    (
        MIN_FLOPS.swap(min_flops, Ordering::Relaxed),
        MIN_K.swap(min_k, Ordering::Relaxed),
    )
}

/// Whether a NumPy-backed provider is installed and reachable.
#[pyfunction]
fn gemm_provider_installed() -> bool {
    engine::ops::linalg::gemm_provider_installed()
}

pub fn register_gemm_module(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "dispatch")?;
    module.setattr(
        "__doc__",
        "Where dense products go: the NumPy provider, and the size boundary \
         that decides when it is worth the crossing.",
    )?;
    module.add_function(wrap_pyfunction!(gemm_thresholds, &module)?)?;
    module.add_function(wrap_pyfunction!(set_gemm_thresholds, &module)?)?;
    module.add_function(wrap_pyfunction!(gemm_provider_installed, &module)?)?;
    module.add("DEFAULT_MIN_FLOPS", DEFAULT_MIN_FLOPS)?;
    module.add("DEFAULT_MIN_K", DEFAULT_MIN_K)?;
    parent.add_submodule(&module)?;
    Ok(())
}
