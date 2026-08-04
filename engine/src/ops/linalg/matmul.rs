// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;

use crate::{
    autograd::{DotBackward, MatMulBackward, SolveBackward, add_to_graph},
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_binary_operands},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "blas")]
use cblas::{Layout, Transpose};

pub(crate) use crate::ops::map::PAR_THRESHOLD;

#[derive(Debug, Clone)]
pub(crate) struct DiagonalSpec {
    pub diag_len: usize,
    pub base_offset: usize,
    pub diag_stride: usize,
    pub kept_dims: Vec<usize>,
    pub output_dims: Vec<usize>,
}

pub(crate) use crate::ops::util::{normalize_dim, normalize_dim_named};

pub(crate) fn compute_diagonal_spec(
    dims: &[usize],
    strides: &[usize],
    dim1: usize,
    dim2: usize,
    offset: isize,
) -> Result<DiagonalSpec> {
    debug_assert!(dim1 != dim2);

    let dim1_size = dims
        .get(dim1)
        .ok_or_else(|| MinitensorError::index_error(dim1 as isize, 0, dims.len()))?;
    let dim2_size = dims
        .get(dim2)
        .ok_or_else(|| MinitensorError::index_error(dim2 as isize, 0, dims.len()))?;
    let stride1 = strides
        .get(dim1)
        .ok_or_else(|| MinitensorError::index_error(dim1 as isize, 0, strides.len()))?;
    let stride2 = strides
        .get(dim2)
        .ok_or_else(|| MinitensorError::index_error(dim2 as isize, 0, strides.len()))?;

    let diag_stride = stride1.saturating_add(*stride2);

    let (diag_len, base_offset) = if offset >= 0 {
        let offset = offset as usize;
        if offset >= *dim2_size {
            (0, 0)
        } else {
            (
                (*dim1_size).min(dim2_size - offset),
                offset.saturating_mul(*stride2),
            )
        }
    } else {
        let neg = (-offset) as usize;
        if neg >= *dim1_size {
            (0, 0)
        } else {
            (
                (dim1_size - neg).min(*dim2_size),
                neg.saturating_mul(*stride1),
            )
        }
    };

    let mut kept_dims = Vec::with_capacity(dims.len().saturating_sub(2));
    let mut output_dims = Vec::with_capacity(kept_dims.capacity() + 1);
    for (idx, &size) in dims.iter().enumerate() {
        if idx == dim1 || idx == dim2 {
            continue;
        }
        kept_dims.push(idx);
        output_dims.push(size);
    }
    output_dims.push(diag_len);

    Ok(DiagonalSpec {
        diag_len,
        base_offset,
        diag_stride,
        kept_dims,
        output_dims,
    })
}

pub(crate) fn diagonal_copy<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [T],
    dims: &[usize],
    strides: &[usize],
    spec: &DiagonalSpec,
) {
    if output.is_empty() {
        return;
    }

    let mut axis_sizes: Vec<usize> = spec.kept_dims.iter().map(|&dim| dims[dim]).collect();
    axis_sizes.push(spec.diag_len);

    let mut axis_strides: Vec<usize> = spec.kept_dims.iter().map(|&dim| strides[dim]).collect();
    axis_strides.push(spec.diag_stride);

    let axes = axis_sizes.len();
    let mut indices = vec![0usize; axes];
    let mut out_idx = 0usize;

    loop {
        let mut input_offset = spec.base_offset;
        for axis in 0..axes {
            input_offset += indices[axis] * axis_strides[axis];
        }
        output[out_idx] = input[input_offset];
        out_idx += 1;

        let mut done = true;
        for axis in (0..axes).rev() {
            indices[axis] += 1;
            if indices[axis] < axis_sizes[axis] {
                done = false;
                break;
            }
            indices[axis] = 0;
        }
        if done {
            break;
        }
    }
}

pub(crate) fn diagonal_scatter<T>(
    grad_output: &[T],
    grad_input: &mut [T],
    dims: &[usize],
    strides: &[usize],
    spec: &DiagonalSpec,
) where
    T: Copy + Send + Sync + std::ops::AddAssign,
{
    if grad_output.is_empty() {
        return;
    }

    let mut axis_sizes: Vec<usize> = spec.kept_dims.iter().map(|&dim| dims[dim]).collect();
    axis_sizes.push(spec.diag_len);

    let mut axis_strides: Vec<usize> = spec.kept_dims.iter().map(|&dim| strides[dim]).collect();
    axis_strides.push(spec.diag_stride);

    let axes = axis_sizes.len();
    let mut indices = vec![0usize; axes];
    let mut out_idx = 0usize;

    loop {
        let mut input_offset = spec.base_offset;
        for axis in 0..axes {
            input_offset += indices[axis] * axis_strides[axis];
        }
        grad_input[input_offset] += grad_output[out_idx];
        out_idx += 1;

        let mut done = true;
        for axis in (0..axes).rev() {
            indices[axis] += 1;
            if indices[axis] < axis_sizes[axis] {
                done = false;
                break;
            }
            indices[axis] = 0;
        }
        if done {
            break;
        }
    }
}

/// # Safety
///
/// `a`, `b` and `c` must point to at least `m * k`, `k * n` and `m * n`
/// readable (writable, for `c`) elements, the same contract the
/// `matrixmultiply` path below documents.
#[cfg(feature = "blas")]
#[inline]
pub(crate) unsafe fn gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) {
    // `cblas` takes slices, not the raw pointers this signature carries for the
    // `matrixmultiply` path. Passing the pointers straight through did not
    // compile, so `--features blas` has never built; the lengths come from the
    // documented contract above.
    let (a, b, c) = unsafe {
        (
            std::slice::from_raw_parts(a, m * k),
            std::slice::from_raw_parts(b, k * n),
            std::slice::from_raw_parts_mut(c, m * n),
        )
    };
    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            n as i32,
            0.0,
            c,
            n as i32,
        );
    }
}

/// # Safety
///
/// `a`, `b` and `c` must point to at least `m * k`, `k * n` and `m * n`
/// readable (writable, for `c`) elements, the same contract the
/// `matrixmultiply` path below documents.
#[cfg(feature = "blas")]
#[inline]
pub(crate) unsafe fn gemm_f64(
    m: usize,
    k: usize,
    n: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    // `cblas` takes slices, not the raw pointers this signature carries for the
    // `matrixmultiply` path. Passing the pointers straight through did not
    // compile, so `--features blas` has never built; the lengths come from the
    // documented contract above.
    let (a, b, c) = unsafe {
        (
            std::slice::from_raw_parts(a, m * k),
            std::slice::from_raw_parts(b, k * n),
            std::slice::from_raw_parts_mut(c, m * n),
        )
    };
    unsafe {
        cblas::dgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            n as i32,
            0.0,
            c,
            n as i32,
        );
    }
}

/// A raw pointer that may be captured by a rayon task.
///
/// The tasks a split GEMM spawns each write a disjoint part of `c` and only
/// read `a` and `b`, so the aliasing rules hold; a bare `*const T` is simply
/// not `Send` on its own.
#[cfg(not(feature = "blas"))]
#[derive(Clone, Copy)]
struct SendPtr<T>(T);

#[cfg(not(feature = "blas"))]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(not(feature = "blas"))]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(not(feature = "blas"))]
impl<T: Copy> SendPtr<T> {
    /// Read the pointer back out.
    ///
    /// Taking `self` by value matters: a closure that named the field directly
    /// would capture the bare pointer under disjoint capture, and a bare
    /// pointer is exactly what is not `Send`.
    #[inline(always)]
    fn get(self) -> T {
        self.0
    }
}

/// How to spread one GEMM across rayon's pool.
#[cfg(not(feature = "blas"))]
enum GemmSplit {
    /// Run it on the calling thread.
    Whole,
    /// Give each task a band of output rows.
    Rows(usize),
    /// Give each task a band of output columns.
    Cols(usize),
}

/// Below this many multiply-accumulates a split costs more than it saves:
/// rayon's fork/join measures about 9 us here, against a serial GEMM that is
/// still around 15 us at this size.
#[cfg(not(feature = "blas"))]
const GEMM_MIN_MACS_TO_SPLIT: usize = 1 << 20;

/// A task thinner than this stops giving the kernel enough to work on.
#[cfg(not(feature = "blas"))]
const GEMM_MIN_SLICE: usize = 32;

/// Decide how to divide an `m x k` by `k x n` product.
///
/// `matrixmultiply` is built without its own `threading` feature, so this is
/// where a large product gets parallelised. Splitting the output rather than
/// the reduction is what keeps the result bit-identical to the serial call:
/// every output element still accumulates over the whole of `k` in the same
/// order, and no task touches an element another task writes.
#[cfg(not(feature = "blas"))]
fn plan_gemm(m: usize, k: usize, n: usize) -> GemmSplit {
    let threads = rayon::current_num_threads();
    if threads < 2 || m.saturating_mul(k).saturating_mul(n) < GEMM_MIN_MACS_TO_SPLIT {
        return GemmSplit::Whole;
    }

    // Whichever output axis is longer. A column split makes every task read all
    // of `a`, a row split makes every task read all of `b`, and dividing the
    // longer axis is what keeps that duplicated read small next to the work the
    // task actually does. Measured over shapes from (16,1024,1024) to
    // (65536,64,8), picking the longer axis wins in every case, by up to 4.5x
    // over picking the shorter one.
    if n >= m {
        match threads.min(n / GEMM_MIN_SLICE) {
            0 | 1 => GemmSplit::Whole,
            tasks => GemmSplit::Cols(tasks),
        }
    } else {
        match threads.min(m / GEMM_MIN_SLICE) {
            0 | 1 => GemmSplit::Whole,
            tasks => GemmSplit::Rows(tasks),
        }
    }
}

/// Define `gemm_f32` / `gemm_f64` over the corresponding `matrixmultiply` entry
/// point. Both take row-major operands with unit column stride; the row strides
/// stay at the *full* `k` and `n` even for a slice, which is what lets a task
/// address a band of the original matrices without copying anything.
#[cfg(not(feature = "blas"))]
macro_rules! split_gemm {
    ($name:ident, $serial:ident, $nt:ident, $tn:ident, $ty:ty, $kernel:path) => {
        /// `c = a * b` with explicit operand strides, on the calling thread.
        ///
        /// Row/column strides are given per operand so that an operand already
        /// stored transposed can be consumed in place: a logical `(p, q)` matrix
        /// held as `(q, p)` row-major is just `rs = 1, cs = p`. `matrixmultiply`
        /// packs its operands regardless, so the transpose costs nothing extra
        /// there -- whereas materialising it costs a full copy of the matrix.
        ///
        /// # Safety
        ///
        /// The strides must address only readable elements of `a` and `b`, and
        /// `c` must be writable at `rsc` per row with unit column stride. A
        /// column band of a wider output keeps the *full* width as `rsc`, which
        /// is why it is passed rather than derived from `n`.
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $serial(
            m: usize,
            k: usize,
            n: usize,
            a: *const $ty,
            rsa: usize,
            csa: usize,
            b: *const $ty,
            rsb: usize,
            csb: usize,
            c: *mut $ty,
            rsc: usize,
        ) {
            unsafe {
                $kernel(
                    m,
                    k,
                    n,
                    1.0,
                    a,
                    rsa as isize,
                    csa as isize,
                    b,
                    rsb as isize,
                    csb as isize,
                    0.0,
                    c,
                    rsc as isize,
                    1,
                )
            }
        }

        /// As [`$serial`], but spread across rayon when the product is large
        /// enough to be worth dividing.
        ///
        /// # Safety
        ///
        /// As [`$serial`].
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            m: usize,
            k: usize,
            n: usize,
            a: *const $ty,
            rsa: usize,
            csa: usize,
            b: *const $ty,
            rsb: usize,
            csb: usize,
            c: *mut $ty,
        ) {
            match plan_gemm(m, k, n) {
                GemmSplit::Whole => unsafe { $serial(m, k, n, a, rsa, csa, b, rsb, csb, c, n) },
                GemmSplit::Cols(tasks) => {
                    let width = n.div_ceil(tasks);
                    let (a, b, c) = (SendPtr(a), SendPtr(b), SendPtr(c));
                    (0..tasks).into_par_iter().for_each(|t| {
                        let start = t * width;
                        if start >= n {
                            return;
                        }
                        let cols = width.min(n - start);
                        // A column band of `b` starts one column stride in;
                        // `c` is contiguous output, so it starts `start` in.
                        unsafe {
                            $serial(
                                m,
                                k,
                                cols,
                                a.get(),
                                rsa,
                                csa,
                                b.get().add(start * csb),
                                rsb,
                                csb,
                                c.get().add(start),
                                // The band is part of a wider output: its rows
                                // are still `n` apart.
                                n,
                            )
                        }
                    });
                }
                GemmSplit::Rows(tasks) => {
                    let rows = m.div_ceil(tasks);
                    let (a, b, c) = (SendPtr(a), SendPtr(b), SendPtr(c));
                    (0..tasks).into_par_iter().for_each(|t| {
                        let start = t * rows;
                        if start >= m {
                            return;
                        }
                        let band = rows.min(m - start);
                        // A row band of `a` starts one row stride in; output
                        // rows are contiguous.
                        unsafe {
                            $serial(
                                band,
                                k,
                                n,
                                a.get().add(start * rsa),
                                rsa,
                                csa,
                                b.get(),
                                rsb,
                                csb,
                                c.get().add(start * n),
                                n,
                            )
                        }
                    });
                }
            }
        }

        /// `c = a * b`, both operands row-major and contiguous.
        ///
        /// # Safety
        ///
        /// `a`, `b` and `c` must point to at least `m * k`, `k * n` and `m * n`
        /// readable (writable, for `c`) elements.
        #[inline]
        pub(crate) unsafe fn $nt(
            m: usize,
            k: usize,
            n: usize,
            a: *const $ty,
            b: *const $ty,
            c: *mut $ty,
        ) {
            // `b` holds the logical `(k, n)` operand as `(n, k)` row-major.
            unsafe { $name(m, k, n, a, k, 1, b, 1, k, c) }
        }

        /// `c = a^T * b`, where `a` holds the logical `(m, k)` operand as
        /// `(k, m)` row-major.
        ///
        /// # Safety
        ///
        /// As [`$nt`], with `a` read as `k * m` elements.
        #[inline]
        pub(crate) unsafe fn $tn(
            m: usize,
            k: usize,
            n: usize,
            a: *const $ty,
            b: *const $ty,
            c: *mut $ty,
        ) {
            unsafe { $name(m, k, n, a, 1, m, b, n, 1, c) }
        }
    };
}

#[cfg(not(feature = "blas"))]
split_gemm!(
    gemm_strided_f32,
    gemm_strided_serial_f32,
    gemm_nt_f32,
    gemm_tn_f32,
    f32,
    matrixmultiply::sgemm
);
#[cfg(not(feature = "blas"))]
split_gemm!(
    gemm_strided_f64,
    gemm_strided_serial_f64,
    gemm_nt_f64,
    gemm_tn_f64,
    f64,
    matrixmultiply::dgemm
);

/// `c = a * b`, everything row-major and contiguous. The common case.
///
/// # Safety
///
/// `a`, `b` and `c` must point to at least `m * k`, `k * n` and `m * n`
/// readable (writable, for `c`) elements.
#[cfg(not(feature = "blas"))]
#[inline]
pub(crate) unsafe fn gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) {
    unsafe { gemm_strided_f32(m, k, n, a, k, 1, b, n, 1, c) }
}

/// See [`gemm_f32`].
///
/// # Safety
///
/// As [`gemm_f32`].
#[cfg(not(feature = "blas"))]
#[inline]
pub(crate) unsafe fn gemm_f64(
    m: usize,
    k: usize,
    n: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    unsafe { gemm_strided_f64(m, k, n, a, k, 1, b, n, 1, c) }
}

/// The whole product on the calling thread; see the module note on why a
/// batched matmul that already fills the pool uses this.
///
/// # Safety
///
/// As [`gemm_f32`].
#[cfg(not(feature = "blas"))]
#[inline]
pub(crate) unsafe fn gemm_serial_f32(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) {
    unsafe { gemm_strided_serial_f32(m, k, n, a, k, 1, b, n, 1, c, n) }
}

/// See [`gemm_serial_f32`].
///
/// # Safety
///
/// As [`gemm_f64`].
#[cfg(not(feature = "blas"))]
#[inline]
pub(crate) unsafe fn gemm_serial_f64(
    m: usize,
    k: usize,
    n: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    unsafe { gemm_strided_serial_f64(m, k, n, a, k, 1, b, n, 1, c, n) }
}

/// With a BLAS underneath there is no splitting to opt out of: a BLAS threads
/// its own GEMM, and dividing the call first would only fight it. So the
/// "serial" entry point is the same call.
///
/// # Safety
///
/// As [`gemm_f32`].
#[cfg(feature = "blas")]
#[inline]
pub(crate) unsafe fn gemm_serial_f32(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) {
    unsafe { gemm_f32(m, k, n, a, b, c) }
}

/// See [`gemm_serial_f32`].
///
/// # Safety
///
/// As [`gemm_f64`].
#[cfg(feature = "blas")]
#[inline]
pub(crate) unsafe fn gemm_serial_f64(
    m: usize,
    k: usize,
    n: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    unsafe { gemm_f64(m, k, n, a, b, c) }
}

/// Reject a product whose operands cannot be multiplied, describing both of
/// them as the caller wrote them.
///
/// Run before the 1-D promotion and batch folding below, because those rewrite
/// the shapes before anything checks them. `[3, 4] @ [7]` used to be reported
/// as "Shape mismatch: expected [4, 1], got [7, 1]" -- three of those four
/// numbers appear nowhere in the call, and the `[4, 1]` labelled "expected" is
/// not a shape either operand could have had.
fn validate_matmul_shapes(lhs: &[usize], rhs: &[usize]) -> Result<()> {
    if lhs.is_empty() || rhs.is_empty() {
        // Scalar operands are rejected further down, with their own message.
        return Ok(());
    }

    // A 1-D operand contributes its whole length; a matrix contributes the
    // axis that faces the other operand.
    let (inner_lhs, lhs_clause) = if lhs.len() == 1 {
        (
            lhs[0],
            format!("the length of the first operand ({})", lhs[0]),
        )
    } else {
        let k = lhs[lhs.len() - 1];
        (k, format!("the last dimension of the first operand ({k})"))
    };
    let (inner_rhs, rhs_clause) = if rhs.len() == 1 {
        (rhs[0], format!("the length of the second ({})", rhs[0]))
    } else {
        let k = rhs[rhs.len() - 2];
        (
            k,
            format!("the second-to-last dimension of the second ({k})"),
        )
    };

    if inner_lhs != inner_rhs {
        return Err(MinitensorError::invalid_argument_with_suggestion(
            format!(
                "matmul: shapes {lhs:?} and {rhs:?} cannot be multiplied -- \
                 {lhs_clause} must equal {rhs_clause}"
            ),
            matmul_transpose_hint(lhs, rhs),
        ));
    }

    // Everything but the last two axes is batch, and batches broadcast.
    if lhs.len() > 2 || rhs.len() > 2 {
        let lhs_batch = Shape::new(lhs[..lhs.len().saturating_sub(2)].to_vec());
        let rhs_batch = Shape::new(rhs[..rhs.len().saturating_sub(2)].to_vec());
        if lhs_batch.broadcast_with(&rhs_batch).is_err() {
            return Err(MinitensorError::invalid_argument_with_suggestion(
                format!(
                    "matmul: shapes {lhs:?} and {rhs:?} cannot be multiplied -- the \
                     matrix dimensions agree, but the batch dimensions {:?} and {:?} \
                     do not broadcast",
                    lhs_batch.dims(),
                    rhs_batch.dims()
                ),
                "Batch dimensions broadcast like elementwise operands: each pair must \
                 be equal, or one of them 1",
            ));
        }
    }

    Ok(())
}

/// The usual cause of a mismatched inner dimension is one operand stored the
/// other way round, and the shapes say which one.
fn matmul_transpose_hint(lhs: &[usize], rhs: &[usize]) -> String {
    if lhs.len() >= 2 && rhs.len() >= 2 {
        let cols_lhs = lhs[lhs.len() - 1];
        let rows_lhs = lhs[lhs.len() - 2];
        let cols_rhs = rhs[rhs.len() - 1];
        let rows_rhs = rhs[rhs.len() - 2];
        if cols_lhs == cols_rhs {
            return format!(
                "Both operands end in {cols_lhs}, so the second is likely stored \
                 transposed: try b.transpose(-1, -2)"
            );
        }
        if rows_lhs == rows_rhs {
            return format!(
                "Both operands have {rows_lhs} rows, so the first is likely stored \
                 transposed: try a.transpose(-1, -2)"
            );
        }
    }
    "Check that the operands are the right way round, and that any transpose you \
     meant to apply was applied"
        .to_string()
}

/// Matrix multiplication with gradient support
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    // Check data type compatibility
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }

    validate_matmul_shapes(lhs.shape().dims(), rhs.shape().dims())?;

    // For 1-D vectors, `lhs` is promoted by prepending a 1 and `rhs` by appending
    // a 1; the added axes are removed from the result
    // (so mat@vec -> vec, vec@mat -> vec, vec@vec -> scalar). Reshapes are
    // grad-aware, so the gradient flows through the promotion.
    let lhs_1d = lhs.ndim() == 1;
    let rhs_1d = rhs.ndim() == 1;
    if lhs_1d || rhs_1d {
        use crate::ops::shape_ops::reshape;
        let lhs2 = if lhs_1d {
            reshape(lhs, Shape::new(vec![1, lhs.shape().dims()[0]]))?
        } else {
            lhs.clone()
        };
        let rhs2 = if rhs_1d {
            reshape(rhs, Shape::new(vec![rhs.shape().dims()[0], 1]))?
        } else {
            rhs.clone()
        };
        let promoted = matmul(&lhs2, &rhs2)?;
        // Drop the promoted axes (remove the trailing column before the leading
        // row so the earlier index stays valid).
        let mut dims = promoted.shape().dims().to_vec();
        let len = dims.len();
        if rhs_1d {
            dims.remove(len - 1);
        }
        if lhs_1d {
            dims.remove(len - 2);
        }
        return reshape(&promoted, Shape::new(dims));
    }

    // Validate matrix multiplication dimensions
    if lhs.ndim() < 2 || rhs.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "Matrix multiplication requires tensors with at least 1 dimension (scalars are not valid operands)",
        ));
    }

    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    // A 2-D rhs multiplies every batch of the lhs by the same matrix -- the
    // shape any linear layer over batched sequences produces. Broadcasting it
    // below would materialize one copy of the weights per batch and then issue
    // that many small GEMMs. Folding the batch axes into the row dimension
    // instead costs no copy and issues a single large GEMM, which packs the
    // operands once rather than per batch: measured 2-3x faster on typical
    // sizes. Reshapes are grad-aware, so the gradient flows back unchanged.
    if rhs_shape.len() == 2 && lhs_shape.len() > 2 && lhs_shape[lhs_shape.len() - 1] == rhs_shape[0]
    {
        use crate::ops::shape_ops::reshape;
        let k = rhs_shape[0];
        let n = rhs_shape[1];
        let rows: usize = lhs_shape[..lhs_shape.len() - 1].iter().product();

        let folded = reshape(lhs, Shape::new(vec![rows, k]))?;
        let product = matmul(&folded, rhs)?;

        let mut out_dims = lhs_shape[..lhs_shape.len() - 1].to_vec();
        out_dims.push(n);
        return reshape(&product, Shape::new(out_dims));
    }

    // Broadcast batch dimensions when they differ, e.g.
    // [2, 3, 4] @ [4, 5] or [1, 3, 4] @ [7, 4, 5]. The expanded operands are
    // materialized contiguously; expand/contiguous are grad-aware so the
    // gradient reduces back over the broadcast batch dimensions.
    if lhs_shape[..lhs_shape.len() - 2] != rhs_shape[..rhs_shape.len() - 2] {
        let lhs_batch = Shape::new(lhs_shape[..lhs_shape.len() - 2].to_vec());
        let rhs_batch = Shape::new(rhs_shape[..rhs_shape.len() - 2].to_vec());
        let batch = lhs_batch
            .broadcast_with(&rhs_batch)
            .map_err(|_| MinitensorError::shape_mismatch(lhs_shape.to_vec(), rhs_shape.to_vec()))?;

        let mut lhs_target: Vec<isize> = batch.dims().iter().map(|&d| d as isize).collect();
        lhs_target.extend_from_slice(&[
            lhs_shape[lhs_shape.len() - 2] as isize,
            lhs_shape[lhs_shape.len() - 1] as isize,
        ]);
        let mut rhs_target: Vec<isize> = batch.dims().iter().map(|&d| d as isize).collect();
        rhs_target.extend_from_slice(&[
            rhs_shape[rhs_shape.len() - 2] as isize,
            rhs_shape[rhs_shape.len() - 1] as isize,
        ]);

        let lhs_b = lhs.expand(lhs_target)?;
        let rhs_b = rhs.expand(rhs_target)?;
        return matmul(&lhs_b, &rhs_b);
    }

    // Get the last two dimensions for matrix multiplication
    let lhs_rows = lhs_shape[lhs_shape.len() - 2];
    let lhs_cols = lhs_shape[lhs_shape.len() - 1];
    let rhs_rows = rhs_shape[rhs_shape.len() - 2];
    let rhs_cols = rhs_shape[rhs_shape.len() - 1];

    if lhs_cols != rhs_rows {
        // `validate_matmul_shapes` has already run on the operands as the
        // caller passed them, so reaching here means the promotion or folding
        // above produced something it did not. Re-run it rather than
        // hand-rolling a second message: one of them would go stale.
        validate_matmul_shapes(lhs_shape, rhs_shape)?;
        return Err(MinitensorError::internal_error(format!(
            "matmul: inner dimensions {lhs_cols} and {rhs_rows} disagree after \
             reshaping {lhs_shape:?} and {rhs_shape:?}, which the shape check accepted"
        )));
    }

    // Compute output shape
    let mut output_shape = lhs_shape[..lhs_shape.len() - 2].to_vec();
    output_shape.push(lhs_rows);
    output_shape.push(rhs_cols);
    let output_shape_obj = Shape::new(output_shape);

    if lhs.dtype() == DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "Matrix multiplication not supported for boolean tensors",
        ));
    }

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), lhs.dtype(), lhs.device());

    if output_shape_obj.numel() != 0 && lhs_cols != 0 {
        // Perform matrix multiplication based on data type
        match lhs.dtype() {
            DataType::Float32 => matmul_f32(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Float64 => matmul_f64(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Int32 => matmul_i32(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Int64 => matmul_i64(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Bool => unreachable!("bool dtype checked above"),
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape_obj,
        lhs.dtype(),
        lhs.device(),
        lhs.requires_grad() || rhs.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(MatMulBackward {
            lhs: lhs.detach(),
            rhs: rhs.detach(),
            input_ids: [lhs.id(), rhs.id()],
            lhs_requires_grad: lhs.requires_grad(),
            rhs_requires_grad: rhs.requires_grad(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Solve a linear system of equations `AX = B` for `X`.
///
/// Both `lhs` (`A`) and `rhs` (`B`) must be float tensors that live on the CPU.
/// `lhs` must have shape `[..., n, n]` (square matrices) and `rhs` can either have
/// shape `[..., n]` (a collection of vectors) or `[..., n, k]` (multiple right
/// hand sides). Batch dimensions need to match exactly across the operands.
pub fn solve(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }

    let lhs_ndim = lhs.ndim();
    if lhs_ndim < 2 {
        return Err(MinitensorError::invalid_operation(
            "solve expects lhs to have at least 2 dimensions",
        ));
    }

    let lhs_shape = lhs.shape().dims();
    let n = lhs_shape[lhs_ndim - 1];
    let m = lhs_shape[lhs_ndim - 2];
    if n != m {
        return Err(MinitensorError::invalid_operation(
            "solve expects lhs matrices to be square",
        ));
    }

    let rhs_ndim = rhs.ndim();
    if rhs_ndim < 1 {
        return Err(MinitensorError::invalid_operation(
            "solve expects rhs to have at least 1 dimension",
        ));
    }

    let rhs_shape = rhs.shape().dims();
    let (rhs_cols, rhs_batch_dims) = if rhs_ndim == lhs_ndim {
        if rhs_shape[rhs_ndim - 2] != n {
            return Err(MinitensorError::shape_mismatch(
                vec![n],
                vec![rhs_shape[rhs_ndim - 2]],
            ));
        }
        (rhs_shape[rhs_ndim - 1], &rhs_shape[..rhs_ndim - 2])
    } else if rhs_ndim + 1 == lhs_ndim {
        if rhs_shape[rhs_ndim - 1] != n {
            return Err(MinitensorError::shape_mismatch(
                vec![n],
                vec![rhs_shape[rhs_ndim - 1]],
            ));
        }
        (1usize, &rhs_shape[..rhs_ndim - 1])
    } else {
        return Err(MinitensorError::invalid_operation(
            "solve expects rhs to have either the same rank as lhs or one less",
        ));
    };

    if &lhs_shape[..lhs_ndim - 2] != rhs_batch_dims {
        return Err(MinitensorError::shape_mismatch(
            lhs_shape[..lhs_ndim - 2].to_vec(),
            rhs_batch_dims.to_vec(),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    let output_shape = rhs_shape.to_vec();
    let output_shape = Shape::new(output_shape);

    let mut output_data =
        TensorData::zeros_on_device(output_shape.numel(), lhs.dtype(), lhs.device());

    match lhs.dtype() {
        DataType::Float32 => solve_f32(lhs, rhs, &mut output_data, rhs_cols)?,
        DataType::Float64 => solve_f64(lhs, rhs, &mut output_data, rhs_cols)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "solve currently supports only Float32 and Float64 tensors",
            ));
        }
    }

    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        lhs.dtype(),
        lhs.device(),
        requires_grad,
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(SolveBackward {
            lhs: lhs.detach(),
            solution: output.detach(),
            input_ids: [lhs.id(), rhs.id()],
            lhs_requires_grad: lhs.requires_grad(),
            rhs_requires_grad: rhs.requires_grad(),
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

fn solve_f32(lhs: &Tensor, rhs: &Tensor, output: &mut TensorData, rhs_cols: usize) -> Result<()> {
    use std::borrow::Cow;

    let lhs_view = if lhs.is_contiguous() && lhs.data().is_contiguous() {
        Cow::Borrowed(lhs)
    } else {
        Cow::Owned(lhs.contiguous()?)
    };
    let rhs_view = if rhs.is_contiguous() && rhs.data().is_contiguous() {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.contiguous()?)
    };

    let lhs_slice = lhs_view
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f32 data for lhs"))?;
    let rhs_slice = rhs_view
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f32 data for rhs"))?;
    let out_slice = output
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f32 output slice"))?;

    solve_batched(
        lhs.shape().dims(),
        rhs_cols,
        lhs_slice,
        rhs_slice,
        out_slice,
    )
}

fn solve_f64(lhs: &Tensor, rhs: &Tensor, output: &mut TensorData, rhs_cols: usize) -> Result<()> {
    use std::borrow::Cow;

    let lhs_view = if lhs.is_contiguous() && lhs.data().is_contiguous() {
        Cow::Borrowed(lhs)
    } else {
        Cow::Owned(lhs.contiguous()?)
    };
    let rhs_view = if rhs.is_contiguous() && rhs.data().is_contiguous() {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.contiguous()?)
    };

    let lhs_slice = lhs_view
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f64 data for lhs"))?;
    let rhs_slice = rhs_view
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f64 data for rhs"))?;
    let out_slice = output
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f64 output slice"))?;

    solve_batched(
        lhs.shape().dims(),
        rhs_cols,
        lhs_slice,
        rhs_slice,
        out_slice,
    )
}

/// Solve every matrix in the batch independently.
///
/// Each system needs its own scratch copy of the matrix (elimination is
/// destructive), so the batches share nothing and run in parallel. Each task
/// allocates its scratch once and reuses it across the batches it is handed,
/// rather than once per system.
fn solve_batched<T>(
    lhs_shape: &[usize],
    rhs_cols: usize,
    lhs_slice: &[T],
    rhs_slice: &[T],
    out_slice: &mut [T],
) -> Result<()>
where
    T: Copy
        + Send
        + Sync
        + std::ops::SubAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Default
        + PartialEq,
{
    let n = *lhs_shape.last().expect("lhs has at least 2 dims");
    let batch = lhs_shape[..lhs_shape.len() - 2]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);
    let rhs_stride = n * rhs_cols;
    let matrix_stride = n * n;

    // Solve batches `first..first + out_group.len() / rhs_stride`, writing each
    // solution into `out_group`. The scratch buffers are allocated once per
    // call rather than once per system.
    let solve_group = |first: usize, count: usize, out_group: &mut [T]| -> Result<()> {
        let mut matrix = vec![T::default(); matrix_stride];
        let mut rhs_buf = vec![T::default(); rhs_stride];
        for local in 0..count {
            let batch_idx = first + local;
            let lhs_offset = batch_idx * matrix_stride;
            let rhs_offset = batch_idx * rhs_stride;

            matrix.copy_from_slice(&lhs_slice[lhs_offset..lhs_offset + matrix_stride]);
            rhs_buf.copy_from_slice(&rhs_slice[rhs_offset..rhs_offset + rhs_stride]);

            // Runs even with no right-hand-side columns: elimination is what
            // detects a singular `lhs`, and that must still be reported.
            gaussian_elimination(&mut matrix, &mut rhs_buf, n, rhs_cols)?;

            out_group[local * rhs_stride..(local + 1) * rhs_stride].copy_from_slice(&rhs_buf);
        }
        Ok(())
    };

    // With no right-hand-side columns there is no output to split over, so the
    // parallel path has nothing to chunk; a single batch is not worth a task.
    if batch == 1 || rhs_stride == 0 {
        return solve_group(0, batch, out_slice);
    }

    // One system per task is too fine-grained for small `n`; group them so each
    // rayon task carries a comparable amount of arithmetic (~n^3 per system).
    let per_task = (PAR_THRESHOLD / (n * n * n).max(1)).clamp(1, batch);

    // A singular matrix in any batch fails the whole call, as before. Which
    // singular batch is reported is unspecified, but the error is identical for
    // all of them ("solve received a singular matrix"), so the message does not
    // depend on the scheduling.
    out_slice
        .par_chunks_mut(per_task * rhs_stride)
        .enumerate()
        .map(|(group_idx, out_group)| {
            solve_group(
                group_idx * per_task,
                out_group.len() / rhs_stride,
                out_group,
            )
        })
        .collect::<Result<()>>()
}

fn gaussian_elimination<T>(matrix: &mut [T], rhs: &mut [T], n: usize, rhs_cols: usize) -> Result<()>
where
    T: Copy
        + Send
        + Sync
        + std::ops::SubAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Default
        + PartialEq,
{
    for k in 0..n {
        // Pivot selection
        let mut pivot_row = k;
        let mut pivot_val = abs(matrix[k * n + k]);
        for i in (k + 1)..n {
            let candidate = abs(matrix[i * n + k]);
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = i;
            }
        }

        if pivot_val == T::default() {
            return Err(MinitensorError::invalid_operation(
                "solve received a singular matrix",
            ));
        }

        if pivot_row != k {
            for col in 0..n {
                matrix.swap(k * n + col, pivot_row * n + col);
            }
            for col in 0..rhs_cols {
                rhs.swap(k * rhs_cols + col, pivot_row * rhs_cols + col);
            }
        }

        let pivot = matrix[k * n + k];

        for i in (k + 1)..n {
            let factor = matrix[i * n + k] / pivot;
            matrix[i * n + k] = T::default();
            for j in (k + 1)..n {
                let idx = i * n + j;
                matrix[idx] -= factor * matrix[k * n + j];
            }
            for col in 0..rhs_cols {
                let idx = i * rhs_cols + col;
                rhs[idx] -= factor * rhs[k * rhs_cols + col];
            }
        }
    }

    for i in (0..n).rev() {
        let pivot = matrix[i * n + i];
        if abs(pivot) == T::default() {
            return Err(MinitensorError::invalid_operation(
                "solve received a singular matrix",
            ));
        }
        for col in 0..rhs_cols {
            let mut value = rhs[i * rhs_cols + col];
            for j in (i + 1)..n {
                value -= matrix[i * n + j] * rhs[j * rhs_cols + col];
            }
            rhs[i * rhs_cols + col] = value / pivot;
        }
    }

    Ok(())
}

fn abs<T>(value: T) -> T
where
    T: Copy + PartialOrd + std::ops::Neg<Output = T> + Default,
{
    if value < T::default() { -value } else { value }
}

/// Batched matrix multiplication specialized for 3D tensors.
///
/// This is a thin convenience wrapper around [`matmul`] that enforces the
/// traditional batch matrix multiply constraints: both operands must be
/// rank-3 tensors with matching batch dimensions. The actual computation is
/// still delegated to the highly optimised [`matmul`] implementation so all
/// execution happens inside the Rust backend.
pub fn bmm(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.ndim() != 3 || rhs.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "bmm expects both inputs to be 3D tensors".to_string(),
        ));
    }

    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    if lhs_shape[0] != rhs_shape[0] {
        return Err(MinitensorError::shape_mismatch(
            lhs_shape.to_vec(),
            rhs_shape.to_vec(),
        ));
    }

    if lhs_shape[2] != rhs_shape[1] {
        return Err(MinitensorError::shape_mismatch(
            vec![lhs_shape[2]],
            vec![rhs_shape[1]],
        ));
    }

    matmul(lhs, rhs)
}

/// Dot product of two 1D tensors with gradient support
pub fn dot(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let lhs_dims = lhs.ndim();
    let rhs_dims = rhs.ndim();
    if lhs_dims != 1 || rhs_dims != 1 {
        return Err(MinitensorError::invalid_operation(format!(
            "dot: expected 1D tensors but got {}D and {}D tensors",
            lhs_dims, rhs_dims
        )));
    }

    if lhs.numel() != rhs.numel() {
        return Err(MinitensorError::shape_mismatch(
            lhs.shape().dims().to_vec(),
            rhs.shape().dims().to_vec(),
        ));
    }

    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Mul)?;

    if result_dtype == DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "dot does not support bool tensors",
        ));
    }

    let lhs_view = lhs_cast.as_ref();
    let rhs_view = rhs_cast.as_ref();

    let numel = lhs_view.numel();
    let device = lhs.device();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    let output_data = match result_dtype {
        DataType::Float32 => {
            let lhs_slice = lhs_view.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice for dot input")
            })?;

            let dot = if numel >= PAR_THRESHOLD {
                // Index-ordered partials: see `deterministic_par_sum`. A dot
                // product that changes in its last bits between runs is the
                // same reproducibility problem as a non-deterministic `sum`.
                // Both operands are the same length here, so chunking them at
                // the same size keeps the pairs aligned.
                let partials: Vec<f32> = lhs_slice
                    .par_chunks(8192)
                    .zip(rhs_slice.par_chunks(8192))
                    .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>())
                    .collect();
                crate::ops::util::pairwise_fold(partials, 0.0_f32, |a, b| a + b)
            } else {
                lhs_slice
                    .iter()
                    .zip(rhs_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
            };

            TensorData::from_vec_f32(vec![dot], device)
        }
        DataType::Float64 => {
            let lhs_slice = lhs_view.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice for dot input")
            })?;

            let dot = if numel >= PAR_THRESHOLD {
                // Index-ordered partials: see `deterministic_par_sum`. A dot
                // product that changes in its last bits between runs is the
                // same reproducibility problem as a non-deterministic `sum`.
                // Both operands are the same length here, so chunking them at
                // the same size keeps the pairs aligned.
                let partials: Vec<f64> = lhs_slice
                    .par_chunks(8192)
                    .zip(rhs_slice.par_chunks(8192))
                    .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>())
                    .collect();
                crate::ops::util::pairwise_fold(partials, 0.0_f64, |a, b| a + b)
            } else {
                lhs_slice
                    .iter()
                    .zip(rhs_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
            };

            TensorData::from_vec_f64(vec![dot], device)
        }
        DataType::Int32 => {
            let lhs_slice = lhs_view.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice for dot input")
            })?;

            let mut dot: i32 = 0;
            for (&a, &b) in lhs_slice.iter().zip(rhs_slice.iter()) {
                dot = dot.wrapping_add(a.wrapping_mul(b));
            }

            TensorData::from_vec_i32(vec![dot], device)
        }
        DataType::Int64 => {
            let lhs_slice = lhs_view.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice for dot input")
            })?;

            let mut dot: i64 = 0;
            for (&a, &b) in lhs_slice.iter().zip(rhs_slice.iter()) {
                dot = dot.wrapping_add(a.wrapping_mul(b));
            }

            TensorData::from_vec_i64(vec![dot], device)
        }
        DataType::Bool => unreachable!("Bool dtype handled earlier"),
    };

    let output_shape = Shape::new(Vec::new());
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        result_dtype,
        device,
        requires_grad,
    );

    if output.requires_grad() {
        let lhs_requires_grad = lhs.requires_grad();
        let rhs_requires_grad = rhs.requires_grad();
        let grad_fn = Arc::new(DotBackward {
            lhs: lhs_cast.into_owned().detach(),
            rhs: rhs_cast.into_owned().detach(),
            input_ids: [lhs.id(), rhs.id()],
            lhs_requires_grad,
            rhs_requires_grad,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

#[cfg(all(test, not(feature = "blas")))]
mod split_gemm_tests {
    use super::*;

    /// Deterministic values with full mantissas, so that reassociating the
    /// arithmetic anywhere would show up as a differing bit rather than being
    /// hidden by exactly representable inputs.
    fn fill(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map into [-1, 1) using the high mantissa bits.
                (((state >> 11) as f64) / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// The shapes below, and which branch of `plan_gemm` each one exercises.
    fn cases() -> Vec<(usize, usize, usize)> {
        vec![
            // Too small to divide: stays whole.
            (1, 1, 1),
            (8, 32, 32),
            (16, 128, 128),
            // Column split (n >= m).
            (16, 256, 1024),
            (64, 1024, 1024),
            (1, 4096, 4096),
            // Row split (m > n).
            (4096, 1024, 16),
            (1024, 1024, 64),
            (4096, 4096, 1),
            // Sizes that do not divide evenly by the task count.
            (333, 777, 555),
            (37, 1031, 1033),
            (1033, 1031, 37),
        ]
    }

    #[test]
    fn every_case_is_covered_by_the_branch_it_claims() {
        let mut whole = 0;
        let mut rows = 0;
        let mut cols = 0;
        for (m, k, n) in cases() {
            match plan_gemm(m, k, n) {
                GemmSplit::Whole => whole += 1,
                GemmSplit::Rows(tasks) => {
                    assert!(tasks >= 2, "a split into {tasks} task(s) is not a split");
                    rows += 1;
                }
                GemmSplit::Cols(tasks) => {
                    assert!(tasks >= 2, "a split into {tasks} task(s) is not a split");
                    cols += 1;
                }
            }
        }
        // Without this the bit-exactness test below could pass by never
        // splitting anything. Single-threaded machines legitimately take the
        // whole-product branch for everything, so only demand coverage where
        // there is a pool to spread across.
        assert!(whole >= 3, "expected the small shapes to stay whole");
        if rayon::current_num_threads() >= 2 {
            assert!(rows >= 3, "expected the tall shapes to split by row");
            assert!(cols >= 3, "expected the wide shapes to split by column");
        }
    }

    #[test]
    fn splitting_does_not_change_a_single_bit_f32() {
        for (m, k, n) in cases() {
            let a: Vec<f32> = fill(m * k, 0x51ed).iter().map(|&x| x as f32).collect();
            let b: Vec<f32> = fill(k * n, 0xc0ffee).iter().map(|&x| x as f32).collect();
            let mut split = vec![0f32; m * n];
            let mut whole = vec![0f32; m * n];
            unsafe {
                gemm_f32(m, k, n, a.as_ptr(), b.as_ptr(), split.as_mut_ptr());
                gemm_serial_f32(m, k, n, a.as_ptr(), b.as_ptr(), whole.as_mut_ptr());
            }
            let differing = split
                .iter()
                .zip(&whole)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "({m}, {k}, {n}) differs in {differing} element(s)"
            );
        }
    }

    #[test]
    fn splitting_does_not_change_a_single_bit_f64() {
        for (m, k, n) in cases() {
            let a = fill(m * k, 0x51ed);
            let b = fill(k * n, 0xc0ffee);
            let mut split = vec![0f64; m * n];
            let mut whole = vec![0f64; m * n];
            unsafe {
                gemm_f64(m, k, n, a.as_ptr(), b.as_ptr(), split.as_mut_ptr());
                gemm_serial_f64(m, k, n, a.as_ptr(), b.as_ptr(), whole.as_mut_ptr());
            }
            let differing = split
                .iter()
                .zip(&whole)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "({m}, {k}, {n}) differs in {differing} element(s)"
            );
        }
    }

    /// Consuming an operand that is already stored transposed must give the
    /// same answer as materialising the transpose and multiplying normally --
    /// that equivalence is the whole point of the stride form, and it is what
    /// lets `linear` skip a full copy of its weight on every call.
    #[test]
    fn transposed_operands_match_a_materialised_transpose() {
        fn transpose_of(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let mut out = vec![0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = src[i * cols + j];
                }
            }
            out
        }

        for (m, k, n) in cases() {
            let a: Vec<f32> = fill(m * k, 0x1234).iter().map(|&x| x as f32).collect();
            let b: Vec<f32> = fill(k * n, 0x9abc).iter().map(|&x| x as f32).collect();

            // `a * b` with `b` supplied as its (n, k) transpose.
            let bt = transpose_of(&b, k, n);
            let mut viaic_strides = vec![0f32; m * n];
            let mut materialised = vec![0f32; m * n];
            unsafe {
                gemm_nt_f32(m, k, n, a.as_ptr(), bt.as_ptr(), viaic_strides.as_mut_ptr());
                gemm_f32(m, k, n, a.as_ptr(), b.as_ptr(), materialised.as_mut_ptr());
            }
            assert_eq!(
                viaic_strides, materialised,
                "nt disagrees at ({m}, {k}, {n})"
            );

            // `a * b` with `a` supplied as its (k, m) transpose.
            let at = transpose_of(&a, m, k);
            let mut via_strides = vec![0f32; m * n];
            unsafe {
                gemm_tn_f32(m, k, n, at.as_ptr(), b.as_ptr(), via_strides.as_mut_ptr());
            }
            assert_eq!(via_strides, materialised, "tn disagrees at ({m}, {k}, {n})");
        }
    }

    /// A split writes through raw pointers into disjoint bands of the output.
    /// If a band were mis-addressed the result would still be *shaped* right,
    /// so check that every element was actually written.
    #[test]
    fn every_output_element_is_written() {
        for (m, k, n) in cases() {
            let a: Vec<f32> = vec![1.0; m * k];
            let b: Vec<f32> = vec![1.0; k * n];
            let mut out = vec![f32::NAN; m * n];
            unsafe { gemm_f32(m, k, n, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
            assert!(
                out.iter().all(|v| *v == k as f32),
                "({m}, {k}, {n}) left {} element(s) unwritten",
                out.iter().filter(|v| v.is_nan()).count()
            );
        }
    }
}
