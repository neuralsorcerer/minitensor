// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Sending work to the `numpy` every install already has, where it measured
//! faster than the engine.
//!
//! Two kinds of work cross, both measured rather than assumed. A dense product
//! goes to the BLAS `numpy` brought: on a single large `float32` or `float64`
//! product it beats the engine's own kernel by 1.6-2.3x, which is not a gap a
//! portable Rust kernel closes -- it is per-architecture assembly, chosen at
//! load time from a table of them. And a handful of element-wise functions
//! go to `numpy`'s ufuncs, listed in [`delegated`] with the measurements that
//! put them there: in float64 the engine has nothing but scalar `libm` for
//! them, where `numpy` runs vectorized loops that are both faster and at least
//! as accurate.
//!
//! Reaching either costs array headers and one call. The engine's buffers are
//! already contiguous and row-major -- the layout invariant the whole engine is
//! built on -- which is exactly what an array header wants, so nothing is
//! copied, converted or allocated on either side, and the answer is written
//! straight into the output buffer the engine allocated.
//!
//! `matmul` and not `PyArray_MatrixProduct2`, which is the C entry point behind
//! `dot` and looks like the cheaper way in. It is the wrong kernel: given an
//! `out` it takes a route that measured 1.7x slower than `matmul` on a thin-`k`
//! product (4096x128 @ 128x4096), which is the shape a wide linear layer
//! produces and the last one to hand to a slower path.
//!
//! Everything else stays where it is: the engine's element-wise kernels and
//! reductions beat `numpy` by 1.2-5x, because they are parallel, fused, or
//! both, and `numpy` is neither.

// The provider and everything it needs. Not built under `--features blas`:
// that build linked a BLAS into the engine, and reaching through the
// interpreter for another one would be a cost with nothing on the other side
// of it. The knobs and their reporting below are built either way, so
// `_core.dispatch` still answers -- see `PROVIDER_EXPECTED`.
use engine::ops::provider::Ufunc;
#[cfg(not(feature = "blas"))]
use engine::ops::provider::{Gemm, Provider, Storage, set_provider};
#[cfg(not(feature = "blas"))]
use numpy::npyffi::{
    NPY_ARRAY_ALIGNED, NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_F_CONTIGUOUS, NPY_ARRAY_WRITEABLE,
    NpyTypes, PY_ARRAY_API, get_type_object, npy_intp,
};
#[cfg(not(feature = "blas"))]
use numpy::{Element, PyArrayDescrMethods};
#[cfg(not(feature = "blas"))]
use pyo3::ffi;
use pyo3::prelude::*;
#[cfg(not(feature = "blas"))]
use pyo3::sync::PyOnceLock;
use pyo3::types::PyModule;
#[cfg(not(feature = "blas"))]
use pyo3::types::{IntoPyDict, PyTuple};
#[cfg(not(feature = "blas"))]
use std::mem::MaybeUninit;
#[cfg(not(feature = "blas"))]
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

#[cfg(not(feature = "blas"))]
/// `numpy.matmul`, looked up once.
///
/// A module attribute lookup per product would be two dictionary probes on the
/// path this exists to shorten, and the answer cannot change: rebinding
/// `numpy.matmul` mid-process is not something a caller does by accident and
/// not something the engine should follow if they do.
static MATMUL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[cfg(not(feature = "blas"))]
/// Dense GEMMs and element-wise ufuncs through NumPy, on memory the engine
/// owns.
struct Numpy;

#[cfg(not(feature = "blas"))]
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

#[cfg(not(feature = "blas"))]
/// `out = lhs @ rhs` through NumPy, or `false` with `out` untouched.
///
/// # Safety
///
/// Nothing unsafe escapes: the three headers borrow the request's slices and
/// are dropped inside this call. `matmul` writes only into `out`, whose header
/// is the only writeable one.
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
            // has its own kernel and this was only ever an offer. The failure
            // was fetched into the `PyErr` that `ok()` dropped, so nothing is
            // left pending.
            return false;
        };

        // One at a time, stopping at the first failure: only a header
        // allocation can fail here, which means the interpreter is out of
        // memory, and the C-API must not be called again with that error
        // pending. It is cleared and the engine's kernel produces the answer.
        let lhs = unsafe {
            header(
                py,
                request.batch,
                request.m,
                request.k,
                request.lhs_storage,
                request.lhs.as_ptr().cast_mut(),
                false,
            )
        };
        let rhs = lhs.as_ref().and_then(|_| unsafe {
            header(
                py,
                request.batch,
                request.k,
                request.n,
                request.rhs_storage,
                request.rhs.as_ptr().cast_mut(),
                false,
            )
        });
        let out = rhs.as_ref().and_then(|_| unsafe {
            header(
                py,
                request.batch,
                request.m,
                request.n,
                Storage::RowMajor,
                request.out.as_mut_ptr(),
                true,
            )
        });
        let (Some(lhs), Some(rhs), Some(out)) = (lhs, rhs, out) else {
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

#[cfg(not(feature = "blas"))]
impl Provider for Numpy {
    fn gemm_f32(&self, mut request: Gemm<'_, f32>) -> bool {
        matrix_product(&mut request)
    }

    fn gemm_f64(&self, mut request: Gemm<'_, f64>) -> bool {
        matrix_product(&mut request)
    }

    fn unary_f32(&self, op: Ufunc, input: &[f32], out: &mut [MaybeUninit<f32>]) -> bool {
        delegated(op, false) && apply_ufunc(op, input, out)
    }

    fn unary_f64(&self, op: Ufunc, input: &[f64], out: &mut [MaybeUninit<f64>]) -> bool {
        delegated(op, true) && apply_ufunc(op, input, out)
    }
}

/// Whether `numpy` computes `op` faster than the engine, in float64 or not.
///
/// Measured as NumPy's time over the engine's own kernel (below 1 means NumPy
/// is faster), over 1e3, 1e5 and 1e6 elements on a 4-core x86-64 container:
///
/// ```text
///   float64     1e3   1e5   1e6
///   tanh       0.19  0.39  0.41
///   sinh       0.17  0.34  0.34
///   cosh       0.22  0.41  0.48
///   asinh      0.21  0.49  0.53
///   atanh      0.30  0.50  0.51
///   log1p      0.23  0.53  0.56
///   log10      0.21  0.52  0.53
///   tan        0.28  0.77  0.74
///   expm1      0.38  0.76  0.80
///   log2       0.36  0.88  0.94
///   exp        0.32  0.78  0.92
///   log        0.35  0.89  0.92
///   exp2       0.35  0.24  0.93
///   cbrt       0.12  0.08  0.29
/// ```
///
/// Float64 has no wider type to compute in and round from, which is the trick
/// every float32 kernel in `ops::simd::transcendental` is built on, so the
/// engine's float64 arms are scalar `libm` calls, one element at a time;
/// NumPy's are vectorized loops, and for `tanh` the more accurate of the two
/// as well (1 ulp against glibc's 2).
///
/// No float32 function crosses. The two NumPy was faster at are the two where
/// its answer is worse: the engine's float32 `cbrt` and `atanh` are correctly
/// rounded on every input, and NumPy's miss on about a third and on 4.7% of
/// them. Speed that costs the answer is not what delegation is for, so `atanh`
/// got a vectorized kernel instead and `cbrt` keeps its own.
///
/// Delegated, a million elements take 2.5-14x less time than the engine's
/// loop did, and 2.8-4.5x less than calling NumPy directly -- [`apply_ufunc`]
/// runs its loop on every thread the engine would have used.
fn delegated(_op: Ufunc, float64: bool) -> bool {
    float64
}

/// Elements below which a ufunc is not worth the crossing.
///
/// A crossing is an attach, two array headers and a ufunc dispatch, a little
/// over a microsecond. Measured as the engine's time over the delegated one,
/// at 256 elements float32 `cbrt` still read 0.75 and float64 `exp2` 1.00;
/// by 1024 every delegated function read 1.7 or more. Half-way it is.
const DEFAULT_MIN_UFUNC_LEN: usize = 512;

/// See [`MIN_FLOPS`] for why this is an atomic and not a constant.
static MIN_UFUNC_LEN: AtomicUsize = AtomicUsize::new(DEFAULT_MIN_UFUNC_LEN);

#[cfg(not(feature = "blas"))]
/// Each delegated ufunc, looked up once, in the order of [`UFUNC_ORDER`].
static UFUNCS: [PyOnceLock<Py<PyAny>>; UFUNC_ORDER.len()] =
    [const { PyOnceLock::new() }; UFUNC_ORDER.len()];

/// Every [`Ufunc`], for giving each its slot in the ufunc cache and for
/// listing what this build delegates.
const UFUNC_ORDER: [Ufunc; 14] = [
    Ufunc::Tanh,
    Ufunc::Sinh,
    Ufunc::Cosh,
    Ufunc::Tan,
    Ufunc::Asinh,
    Ufunc::Atanh,
    Ufunc::Expm1,
    Ufunc::Log1p,
    Ufunc::Log2,
    Ufunc::Log10,
    Ufunc::Exp,
    Ufunc::Exp2,
    Ufunc::Log,
    Ufunc::Cbrt,
];

#[cfg(not(feature = "blas"))]
/// `out = numpy.<op>(input)`, or `false` with nothing promised about `out`.
///
/// A large input is cut into one chunk per thread of the engine's pool, and
/// each chunk's call runs on its own thread. A ufunc releases the interpreter
/// lock inside its loop, so the chunks run at once: NumPy's vectorized loop
/// and the engine's parallelism together, where a single call is one core.
/// On a million float64 that took `exp`, `log` and `exp2` from 0.86-0.91x of
/// the engine's own four-core scalar loop to ahead of it.
///
/// The threads are scoped ones of their own, not the engine's pool. A pool
/// worker waiting for the interpreter lock stalls every task queued behind
/// it, and another Python thread can hold that lock while it waits on exactly
/// such a task -- a deadlock. A scoped thread waits on the lock and nothing
/// else, and the calling thread gives the lock up while it joins them.
fn apply_ufunc<T: Element>(op: Ufunc, input: &[T], out: &mut [MaybeUninit<T>]) -> bool {
    let len = input.len();
    if len < MIN_UFUNC_LEN.load(Ordering::Relaxed) || len != out.len() {
        return false;
    }
    let slot = UFUNC_ORDER
        .iter()
        .position(|&known| known == op)
        .expect("every Ufunc is in UFUNC_ORDER");
    let chunks = (len / UFUNC_CHUNK_MIN).clamp(1, engine::ops::provider::pool_threads());
    if chunks == 1 {
        return Python::attach(|py| call_ufunc(py, op, slot, input, out));
    }
    let chunk = len.div_ceil(chunks);
    Python::attach(|py| {
        py.detach(|| {
            std::thread::scope(|scope| {
                let mut pieces = input.chunks(chunk).zip(out.chunks_mut(chunk));
                let (first_in, first_out) = pieces.next().expect("at least one chunk");
                let handles: Vec<_> = pieces
                    .map(|(source, target)| {
                        scope.spawn(move || {
                            Python::attach(|py| call_ufunc(py, op, slot, source, target))
                        })
                    })
                    .collect();
                let first = Python::attach(|py| call_ufunc(py, op, slot, first_in, first_out));
                handles
                    .into_iter()
                    .fold(first, |all, handle| handle.join().unwrap_or(false) && all)
            })
        })
    })
}

#[cfg(not(feature = "blas"))]
/// Elements a thread of [`apply_ufunc`] is given at the least: below this a
/// thread costs more to start than its share of the loop.
const UFUNC_CHUNK_MIN: usize = 1 << 16;

#[cfg(not(feature = "blas"))]
/// One ufunc call over `input` into `out`, on the calling thread, which holds
/// the interpreter lock.
fn call_ufunc<T: Element>(
    py: Python<'_>,
    op: Ufunc,
    slot: usize,
    input: &[T],
    out: &mut [MaybeUninit<T>],
) -> bool {
    let Some(ufunc) = UFUNCS[slot]
        .get_or_try_init(py, || -> PyResult<Py<PyAny>> {
            Ok(py.import("numpy")?.getattr(op.numpy_name())?.unbind())
        })
        .ok()
    else {
        return false;
    };
    let Some((errors, ignore)) = QUIET
        .get_or_try_init(py, || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
            let config = py.import("numpy._core._ufunc_config")?;
            let ignore = config
                .getattr("_make_extobj")?
                .call((), Some(&[("all", "ignore")].into_py_dict(py)?))?;
            Ok((
                config.getattr("_extobj_contextvar")?.unbind(),
                ignore.unbind(),
            ))
        })
        .ok()
    else {
        // A NumPy without the error-state variable this was written against:
        // the engine computes the function instead, which costs speed and
        // never a warning.
        return false;
    };
    let len = input.len();
    let source = unsafe {
        header(
            py,
            1,
            1,
            len,
            Storage::RowMajor,
            input.as_ptr().cast_mut(),
            false,
        )
    };
    let target = source.as_ref().and_then(|_| unsafe {
        header(
            py,
            1,
            1,
            len,
            Storage::RowMajor,
            out.as_mut_ptr().cast::<T>(),
            true,
        )
    });
    let (Some(source), Some(target)) = (source, target) else {
        unsafe { ffi::PyErr_Clear() };
        return false;
    };
    // Floating-point errors are silenced for the call, as the engine's own
    // kernels never report them: `log(-1)` is NaN there, not NaN and a
    // `RuntimeWarning` -- and under `-W error` a warning is an exception, which
    // would send every call with such a value back to the slow path.
    // `np.errstate` does this by setting a context variable, and costs 2.3
    // microseconds a call to build and tear down; setting the variable
    // directly costs a small fraction of that. The variable is per thread, so
    // each chunk's thread sets its own.
    let (errors, ignore) = (errors.as_ptr(), ignore.as_ptr());
    // SAFETY: both are live objects held by `QUIET`, and the GIL is held.
    let token = unsafe { ffi::PyContextVar_Set(errors, ignore) };
    if token.is_null() {
        unsafe { ffi::PyErr_Clear() };
        return false;
    }
    // The output goes second and positionally, which is how a ufunc takes it
    // -- see `matrix_product`.
    let called = PyTuple::new(py, [source, target])
        .and_then(|arguments| ufunc.call1(py, arguments))
        .is_ok();
    // SAFETY: `token` came from setting `errors` just above, on this thread,
    // and is released here after restoring the caller's state.
    unsafe {
        if ffi::PyContextVar_Reset(errors, token) != 0 {
            ffi::PyErr_Clear();
        }
        ffi::Py_DECREF(token);
    }
    called
}

#[cfg(not(feature = "blas"))]
/// NumPy's error-state context variable, and the state that ignores every
/// floating-point error; see [`apply_ufunc`].
static QUIET: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

#[cfg(not(feature = "blas"))]
/// A value of `T` to re-zero a partly written output with.
///
/// `Element` does not require `num_traits::Zero` and the two float types are
/// the only ones that reach here, so this is the whole of it.
trait Zero {
    fn zero() -> Self;
}

#[cfg(not(feature = "blas"))]
impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

#[cfg(not(feature = "blas"))]
impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

#[cfg(not(feature = "blas"))]
/// Point the engine's dense GEMM at NumPy, once, while the module loads.
pub fn install_provider() {
    set_provider(Box::new(Numpy));
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
fn provider_installed() -> bool {
    engine::ops::provider::provider_installed()
}

/// The fewest elements a delegated ufunc is handed to NumPy at.
#[pyfunction]
fn ufunc_threshold() -> usize {
    MIN_UFUNC_LEN.load(Ordering::Relaxed)
}

/// Move the ufunc boundary, returning where it was; see
/// [`set_gemm_thresholds`] for why this is reachable at all. `0` delegates
/// every delegated ufunc at any size, and a very large value none.
#[pyfunction]
#[pyo3(signature = (min_len))]
fn set_ufunc_threshold(min_len: usize) -> usize {
    MIN_UFUNC_LEN.swap(min_len, Ordering::Relaxed)
}

/// The element-wise functions this build hands to NumPy, as `(name, dtype)`.
#[pyfunction]
fn delegated_ufuncs() -> Vec<(&'static str, &'static str)> {
    if !cfg!(not(feature = "blas")) {
        return Vec::new();
    }
    UFUNC_ORDER
        .into_iter()
        .flat_map(|op| {
            [("float32", false), ("float64", true)]
                .into_iter()
                .filter(move |&(_, wide)| delegated(op, wide))
                .map(move |(dtype, _)| (op.numpy_name(), dtype))
        })
        .collect()
}

pub fn register_dispatch_module(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "dispatch")?;
    module.setattr(
        "__doc__",
        "What is handed to NumPy -- dense products and a few element-wise \
         functions -- and the size boundaries that decide when it is worth \
         the crossing.",
    )?;
    module.add_function(wrap_pyfunction!(gemm_thresholds, &module)?)?;
    module.add_function(wrap_pyfunction!(set_gemm_thresholds, &module)?)?;
    module.add_function(wrap_pyfunction!(provider_installed, &module)?)?;
    module.add_function(wrap_pyfunction!(ufunc_threshold, &module)?)?;
    module.add_function(wrap_pyfunction!(set_ufunc_threshold, &module)?)?;
    module.add_function(wrap_pyfunction!(delegated_ufuncs, &module)?)?;
    module.add("DEFAULT_MIN_UFUNC_LEN", DEFAULT_MIN_UFUNC_LEN)?;
    module.add("DEFAULT_MIN_FLOPS", DEFAULT_MIN_FLOPS)?;
    module.add("DEFAULT_MIN_K", DEFAULT_MIN_K)?;
    // Whether this build installs a provider at all, which is a property of how
    // the extension was compiled and not otherwise visible from Python. A
    // `--features blas` build deliberately installs none: the engine has its
    // own BLAS and there is nothing to gain by crossing into the interpreter to
    // reach another. Without this, "no provider" and "the provider failed to
    // install" look the same from here, and only one of them is a bug.
    module.add("PROVIDER_EXPECTED", cfg!(not(feature = "blas")))?;
    parent.add_submodule(&module)?;
    Ok(())
}
