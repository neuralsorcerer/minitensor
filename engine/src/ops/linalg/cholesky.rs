// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The Cholesky factorisation, `A = L L^T` for symmetric positive-definite `A`.
//!
//! `solve`, `det`, `slogdet` and `inv` all run a pivoted LU, which is the right
//! answer for a matrix with no structure and the wrong one for a covariance.
//! Nothing in the library could produce the factor itself, and the factor is
//! what a whole class of work is made of: `L @ z` turns standard normal noise
//! into a sample from `N(0, A)`, `2 * sum(log(diag(L)))` is the log-determinant
//! of a covariance without an intermediate that overflows, and whitening is a
//! triangular solve against `L`. None of those can be assembled out of `solve`,
//! which throws its factorisation away.
//!
//! It is also half the arithmetic of LU and needs no pivoting at all -- for a
//! positive-definite matrix the diagonal is always the largest available pivot,
//! which is why this is the routine every implementation reaches for when it
//! knows the matrix is one.
//!
//! Only the lower triangle of the input is read, as LAPACK does. The upper is
//! assumed to mirror it, and a matrix that does not is not symmetric and had no
//! Cholesky factor to begin with. `upper = true` is the same factorisation
//! transposed, composed rather than reimplemented, so the two spellings cannot
//! disagree.

use crate::{
    autograd::{CholeskyBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        linalg::{
            gemm_f32, gemm_f64, gemm_nt_f32, gemm_nt_f64, gemm_tn_f32, gemm_tn_f64, square_layout,
            transpose,
        },
        map::{PAR_THRESHOLD, try_par_out_chunks},
        simd::{simd_dot_f32_wide, simd_dot_f64},
        util::{Accumulate, accurate_pair_sum},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::{Float, Zero};
use std::sync::Arc;

/// The floating-point types a factorisation runs in, with the two vectorised
/// kernels its inner loops are made of.
///
/// The alternative is a macro that emits the whole routine twice, which is how
/// the determinant kernel is written; here the arithmetic is long enough that
/// two copies would be two places for the substitution order to be wrong.
pub(crate) trait Factorable: Float + Send + Sync + 'static {
    /// What a panel's arithmetic is carried out in: `f64` for both element
    /// types, so a single-precision factorisation still sums, subtracts and
    /// divides in double.
    ///
    /// This is worth 1.2-1.4x on the float32 residual and lands it on LAPACK's,
    /// which was not obvious -- panelling caps every sum at [`PANEL`] terms, and
    /// a first attempt at measuring the effect narrowed only [`Self::chunk_dot`]
    /// and concluded it bought nothing. It had left the surrounding subtract and
    /// divide in `f64`, which is where most of it turned out to live. Narrowing
    /// the whole panel moves float32 from 1.00-1.79x LAPACK's residual to
    /// 1.10-1.98x.
    ///
    /// It costs nothing at all in the double case, where the wide type is the
    /// narrow one and every one of these is the identity.
    type Acc: Float + Accumulate + Send + Sync;

    fn widen(self) -> Self::Acc;
    fn narrow(value: Self::Acc) -> Self;

    /// A dot product over at most [`crate::ops::util::RUN_SUM_CHUNK`] elements,
    /// lane-split so the error divides across the lanes.
    fn chunk_dot(a: &[Self], b: &[Self]) -> Self::Acc;

    /// `c = a^T * b`, with `a` holding the logical `(m, k)` operand as `(k, m)`.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` must point to at least `k * m`, `k * n` and `m * n`
    /// elements, writable for `c`.
    unsafe fn gemm_tn(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self);

    /// `c = a * b^T`, with `b` holding the logical `(k, n)` operand as `(n, k)`.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` must point to at least `m * k`, `n * k` and `m * n`
    /// elements, writable for `c`.
    unsafe fn gemm_nt(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self);

    /// `c = a * b`, everything row-major and contiguous.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` must point to at least `m * k`, `k * n` and `m * n`
    /// elements, writable for `c`.
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self);

    /// The full dot product, blocked and folded pairwise above one chunk.
    ///
    /// Every entry of `L` is one of these, so an inner product that grows its
    /// error like `n` would put that growth into every element of the factor and
    /// then square it in the reconstruction. In practice the panel bounds the
    /// length long before that matters -- see [`PANEL`].
    #[inline]
    fn dot(a: &[Self], b: &[Self]) -> Self::Acc {
        accurate_pair_sum(a, b, Self::Acc::zero(), Self::chunk_dot)
    }
}

impl Factorable for f32 {
    type Acc = f64;

    #[inline]
    fn widen(self) -> f64 {
        self as f64
    }

    #[inline]
    fn narrow(value: f64) -> f32 {
        value as f32
    }

    #[inline]
    fn chunk_dot(a: &[f32], b: &[f32]) -> f64 {
        simd_dot_f32_wide(a, b)
    }

    #[inline]
    unsafe fn gemm_tn(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
        unsafe { gemm_tn_f32(m, k, n, a, b, c) }
    }

    #[inline]
    unsafe fn gemm_nt(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
        unsafe { gemm_nt_f32(m, k, n, a, b, c) }
    }

    #[inline]
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
        unsafe { gemm_f32(m, k, n, a, b, c) }
    }
}

impl Factorable for f64 {
    type Acc = f64;

    #[inline]
    fn widen(self) -> f64 {
        self
    }

    #[inline]
    fn narrow(value: f64) -> f64 {
        value
    }

    #[inline]
    fn chunk_dot(a: &[f64], b: &[f64]) -> f64 {
        simd_dot_f64(a, b)
    }

    #[inline]
    unsafe fn gemm_tn(m: usize, k: usize, n: usize, a: *const f64, b: *const f64, c: *mut f64) {
        unsafe { gemm_tn_f64(m, k, n, a, b, c) }
    }

    #[inline]
    unsafe fn gemm_nt(m: usize, k: usize, n: usize, a: *const f64, b: *const f64, c: *mut f64) {
        unsafe { gemm_nt_f64(m, k, n, a, b, c) }
    }

    #[inline]
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const f64, b: *const f64, c: *mut f64) {
        unsafe { gemm_f64(m, k, n, a, b, c) }
    }
}

/// The error a non-positive-definite matrix produces, naming where it was
/// found.
///
/// The order matters to a caller: a factorisation that failed at row 40 of 50
/// says the leading 40x40 block *was* positive definite, which is usually the
/// difference between "my covariance needs more jitter" and "my covariance is
/// the wrong shape entirely".
fn not_positive_definite(order: usize) -> MinitensorError {
    MinitensorError::invalid_operation(format!(
        "cholesky: the matrix is not positive definite (the leading minor of order {} is not)",
        order + 1
    ))
}

/// How many columns are factored at a time before the finished part of the
/// matrix is folded into the next panel in one GEMM.
///
/// The unblocked routine reads every finished row of `L` while completing the
/// next one, so it streams the whole triangle per row and stops fitting in
/// cache the moment the triangle does. Measured against LAPACK it was 0.6-0.9x
/// the time up to `n = 512` and 2.7x at `n = 1024`, which is the cache falling
/// out and nothing else. Panelling turns that streaming into one pass per panel
/// over an operand the GEMM can keep resident.
///
/// It bounds the accuracy too, but not enough to carry it alone: capping every
/// remaining sum at `PANEL` terms is most of what single precision needed, and
/// [`Factorable::Acc`] is the rest.
///
/// A matrix this size or smaller is one panel, never takes the `first > 0`
/// branch, and is exactly the unblocked routine -- there is no second path.
const PANEL: usize = 64;

/// Copy the lower triangle of an `n x n` matrix. The strict upper triangle of
/// the destination is left as it was, which for a fresh output is zero.
fn copy_lower<T: Copy>(src: &[T], dst: &mut [T], n: usize) {
    for i in 0..n {
        dst[i * n..i * n + i + 1].copy_from_slice(&src[i * n..i * n + i + 1]);
    }
}

/// Grow `buffer` to hold at least `need` elements, keeping what is already
/// allocated. Scratch only ever grows, so a task that factors many matrices
/// pays for the largest one once.
fn ensure<T: Factorable>(buffer: &mut Vec<T>, need: usize) {
    if buffer.len() < need {
        buffer.resize(need, T::zero());
    }
}

/// Factor `nb` columns of a matrix held with row stride `stride`, in place.
///
/// The entries arrive holding the values `A` has after every earlier panel has
/// been subtracted out, and leave holding `L`. `first` is the column this panel
/// starts at in the whole matrix, so a failure can name the leading minor
/// rather than an offset into a panel nobody asked about.
///
/// Cholesky-Banachiewicz order -- one row completed before the next starts --
/// because every term it needs is then a contiguous prefix of a row that is
/// already finished, which is what lets the inner product be the vectorised
/// kernel rather than a strided walk.
fn factor_panel<T: Factorable>(
    work: &mut [T],
    stride: usize,
    rows: usize,
    nb: usize,
    first: usize,
) -> Result<()> {
    for i in 0..rows {
        // Rows `0..i` are finished and are read; row `i` is being written. Only
        // the panel's own `nb` columns are ever touched, which is also what
        // keeps the last row inside the buffer when the panel is a corner of a
        // larger matrix.
        let (done, rest) = work.split_at_mut(i * stride);
        let row = &mut rest[..nb];

        for j in 0..nb.min(i) {
            // L[i][j] = (A[i][j] - <L[i][..j], L[j][..j]>) / L[j][j]
            let correction = T::dot(&row[..j], &done[j * stride..j * stride + j]);
            row[j] = T::narrow((row[j].widen() - correction) / done[j * stride + j].widen());
        }

        if i < nb {
            // L[i][i] = sqrt(A[i][i] - <L[i][..i], L[i][..i]>)
            let residual = row[i].widen() - T::dot(&row[..i], &row[..i]);
            // NaN is named alongside the sign test rather than left to fall out
            // of one: an input carrying a NaN would otherwise pass a `> 0` check
            // by failing it and go on through the sqrt into the whole factor.
            if residual.is_nan() || residual <= T::Acc::zero() {
                return Err(not_positive_definite(first + i));
            }
            row[i] = T::narrow(residual.sqrt());
        }
    }
    Ok(())
}

/// Scratch a single task reuses across every matrix it is handed.
///
/// Both buffers stay empty for anything a single panel covers, which is the
/// common case and the one where an allocation per matrix would be most of the
/// cost.
struct Scratch<T> {
    /// The finished columns of the rows this panel spans, packed contiguously
    /// so the GEMM can read them. `L` itself has them `n` apart.
    packed: Vec<T>,
    /// The correction the GEMM produces, before it is subtracted from `A`.
    update: Vec<T>,
}

impl<T: Factorable> Scratch<T> {
    fn new() -> Self {
        Self {
            packed: Vec::new(),
            update: Vec::new(),
        }
    }
}

/// Factor one matrix: `a` is read row-major, `l` is written lower-triangular.
///
/// Left-looking, one panel at a time: before a panel is factored, everything
/// already known is subtracted from it in a single GEMM. That is the same
/// arithmetic a right-looking factorisation does and the same `n^3 / 6`
/// multiply-adds, but it rounds the running correction once per panel rather
/// than once per panel *per later panel*. In single precision the difference is
/// visible -- a right-looking version of this measured 2.7 times LAPACK's
/// reconstruction error at `n = 1024`, purely from those repeated stores.
///
/// It also means the output is the working buffer. There is no copy of `A` to
/// keep and no compaction step; the only scratch is the packed operand.
///
/// `l` arrives zeroed and nothing above the diagonal is ever written, so the
/// zeros there are the ones the allocation came with.
fn factor_one<T: Factorable>(
    a: &[T],
    l: &mut [T],
    n: usize,
    scratch: &mut Scratch<T>,
) -> Result<()> {
    copy_lower(a, l, n);

    let mut first = 0usize;
    while first < n {
        let nb = PANEL.min(n - first);
        let rows = n - first;

        if first > 0 {
            // Pack columns `0..first` of rows `first..n`. The first `nb` rows of
            // the result are also the second operand -- the panel's own rows --
            // so one packed buffer serves both sides of `L21 @ L21^T`.
            ensure(&mut scratch.packed, rows * first);
            for i in 0..rows {
                let src = (first + i) * n;
                scratch.packed[i * first..(i + 1) * first].copy_from_slice(&l[src..src + first]);
            }
            ensure(&mut scratch.update, rows * nb);

            // SAFETY: `packed` holds `rows * first` elements, which covers both
            // the `(rows, first)` operand and the `(nb, first)` one that is its
            // prefix; `update` holds `rows * nb`.
            unsafe {
                T::gemm_nt(
                    rows,
                    first,
                    nb,
                    scratch.packed.as_ptr(),
                    scratch.packed.as_ptr(),
                    scratch.update.as_mut_ptr(),
                );
            }

            for i in 0..rows {
                let width = nb.min(i + 1);
                let dst = (first + i) * n + first;
                let target = &mut l[dst..dst + width];
                let source = &scratch.update[i * nb..i * nb + width];
                for (slot, &correction) in target.iter_mut().zip(source) {
                    *slot = *slot - correction;
                }
            }
        }

        factor_panel(&mut l[first * n + first..], n, rows, nb, first)?;
        first += nb;
    }
    Ok(())
}

/// Factor every matrix in the batch, writing each into its own output block.
///
/// The batches share nothing, so they run in parallel; grouping them by the
/// cost of one factorisation (`n^3 / 3`) keeps a task worth scheduling when `n`
/// is small, exactly as `solve` groups its systems.
fn factor_batched<T: Factorable>(input: &[T], out: &mut [T], n: usize, batch: usize) -> Result<()> {
    let stride = n * n;
    let run = |first: usize, out_group: &mut [T]| -> Result<()> {
        let mut scratch = Scratch::new();
        for (local, block) in out_group.chunks_mut(stride).enumerate() {
            let offset = (first + local) * stride;
            factor_one(&input[offset..offset + stride], block, n, &mut scratch)?;
        }
        Ok(())
    };

    // A zero-order matrix has no factorisation to perform, and asking for
    // chunks of nothing panics rather than yielding none.
    if stride == 0 {
        return Ok(());
    }
    if batch <= 1 {
        return run(0, out);
    }

    let per_task = (PAR_THRESHOLD / (n * n * n / 3).max(1)).clamp(1, batch);
    let span = per_task * stride;
    // Which matrix is named when several are indefinite is unspecified, as it
    // is for `solve`'s singular matrices; the message identifies the row within
    // whichever one was reached first.
    try_par_out_chunks(out, span, &|start, out_group| {
        run((start / span) * per_task, out_group)
    })
}

/// Cholesky factor of every matrix in a stack of symmetric positive-definite
/// matrices.
///
/// Returns lower-triangular `L` with `A = L @ L.T`, or its transpose `U` with
/// `A = U.T @ U` when `upper` is set. Only the lower triangle of the input is
/// read in both cases.
///
/// A matrix that is not positive definite is an error, not a NaN: it means the
/// caller's assumption about the matrix was wrong, and the row where the
/// assumption broke is worth reporting.
pub fn cholesky(tensor: &Tensor, upper: bool) -> Result<Tensor> {
    let (batch_dims, n) = square_layout(tensor, "cholesky")?;
    let batch = batch_dims.iter().product::<usize>().max(1);

    let contiguous = tensor.contiguous()?;
    let shape = tensor.shape().clone();
    let mut output_data =
        TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

    macro_rules! factor {
        ($accessor:ident, $accessor_mut:ident) => {{
            let src = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("cholesky: dtype does not match the input slice")
            })?;
            let dst = output_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("cholesky: dtype does not match the output slice")
            })?;
            factor_batched(src, dst, n, batch)?;
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => factor!(as_f32_slice, as_f32_slice_mut),
        _ => factor!(as_f64_slice, as_f64_slice_mut),
    }

    let mut factor = Tensor::new(
        Arc::new(output_data),
        shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if factor.requires_grad() {
        let grad_fn = Arc::new(CholeskyBackward {
            factor: factor.detach(),
            input_id: tensor.id(),
            ids: [tensor.id()],
        });
        factor = with_grad_fn(factor, grad_fn)?;
    }

    if upper {
        // The upper factor *is* the transpose, and saying so is what stops the
        // two spellings from ever disagreeing -- the gradient comes along for
        // free through `transpose`'s own backward.
        let ndim = factor.ndim() as isize;
        return transpose(&factor, ndim - 2, ndim - 1);
    }
    Ok(factor)
}

/// Solve `L^T X = B` for `X`, in place.
///
/// `L` is lower triangular, so `L^T` is upper and this is back substitution:
/// the last rows of `X` depend on nothing, and each block above them on the
/// blocks below.
///
/// Blocked for the reason the forward is. Substituting one row at a time reads
/// every row below it, so it streams the whole of `X` per row -- `n^3 / 2`
/// element reads, and at `n = 512` that was 40ms of the backward's 48. Taking a
/// panel at a time turns the rows below into one GEMM against a packed operand
/// and leaves only the `PANEL x PANEL` diagonal block to substitute by hand.
fn back_substitute_lt<T: Factorable>(
    l: &[T],
    b: &mut [T],
    n: usize,
    cols: usize,
    scratch: &mut BackwardScratch<T>,
) {
    let mut i0 = n;
    while i0 > 0 {
        let nb = PANEL.min(i0);
        i0 -= nb;
        let below = n - i0 - nb;

        if below > 0 && cols > 0 {
            // `L[i0 + nb.., i0..i0 + nb]` is `n` apart in `L`; the GEMM needs it
            // contiguous, and it is only `below * nb` elements.
            ensure(&mut scratch.packed, below * nb);
            for r in 0..below {
                let src = (i0 + nb + r) * n + i0;
                scratch.packed[r * nb..r * nb + nb].copy_from_slice(&l[src..src + nb]);
            }
            ensure(&mut scratch.update, nb * cols);

            // SAFETY: `packed` holds the `(below, nb)` operand, the rows of `b`
            // from `i0 + nb` are `below * cols` elements, and `update` is
            // `nb * cols`.
            unsafe {
                T::gemm_tn(
                    nb,
                    below,
                    cols,
                    scratch.packed.as_ptr(),
                    b.as_ptr().add((i0 + nb) * cols),
                    scratch.update.as_mut_ptr(),
                );
            }

            for r in 0..nb {
                let dst = (i0 + r) * cols;
                let target = &mut b[dst..dst + cols];
                let source = &scratch.update[r * cols..r * cols + cols];
                for (slot, &correction) in target.iter_mut().zip(source) {
                    *slot = *slot - correction;
                }
            }
        }

        // What is left is the diagonal block, small enough to substitute one
        // row at a time.
        for i in (i0..i0 + nb).rev() {
            for k in (i + 1)..(i0 + nb) {
                // (L^T)[i][k] is L[k][i], and it is zero above the diagonal.
                let coefficient = l[k * n + i];
                if coefficient == T::zero() {
                    continue;
                }
                let (above, rest) = b.split_at_mut(k * cols);
                let source = &rest[..cols];
                let target = &mut above[i * cols..i * cols + cols];
                for (slot, &value) in target.iter_mut().zip(source) {
                    *slot = *slot - coefficient * value;
                }
            }
            let pivot = l[i * n + i];
            for slot in &mut b[i * cols..i * cols + cols] {
                *slot = *slot / pivot;
            }
        }
    }
}

/// Transpose an `n x n` block in place, by swapping across the diagonal.
fn transpose_square<T: Copy>(block: &mut [T], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            block.swap(i * n + j, j * n + i);
        }
    }
}

/// Scratch the backward reuses across every matrix a task is handed.
///
/// `work` carries the gradient through its three stages -- `Φ(L^T L̄)`, then
/// `L^-T` applied twice -- and the other two are the blocked solve's operands.
pub(crate) struct BackwardScratch<T> {
    work: Vec<T>,
    packed: Vec<T>,
    update: Vec<T>,
}

impl<T: Factorable> BackwardScratch<T> {
    fn new() -> Self {
        Self {
            work: Vec::new(),
            packed: Vec::new(),
            update: Vec::new(),
        }
    }
}

/// The gradient of one Cholesky factorisation.
///
/// With `Φ(X)` the lower triangle of `X` with its diagonal halved,
///
/// ```text
/// Ā = L^-T Φ(L^T L̄) L^-1,   then symmetrised as (Ā + Ā^T) / 2
/// ```
///
/// which is the standard reverse-mode form. The symmetrisation is not
/// cosmetic. The forward reads only the lower triangle, so the derivative *of
/// this routine* is lower triangular; but `A` is symmetric by assumption, and a
/// caller who built it as `X @ X.T` needs the sensitivity of both triangles or
/// their gradient comes out half the size. Splitting the total evenly across
/// the pair is what every implementation returns, and it is the gradient of
/// `L(sym(A))`, which is the function the caller believes they called.
///
/// The two inverses are triangular solves, not inversions: `L^-T P` is back
/// substitution, and `M L^-1` is the same substitution applied to `M^T`. That
/// second one leaves `Ā^T` rather than `Ā` in the buffer -- which costs
/// nothing, because the next step symmetrises it.
pub(crate) fn cholesky_backward_block<T: Factorable>(
    l: &[T],
    grad: &[T],
    out: &mut [T],
    scratch: &mut BackwardScratch<T>,
    n: usize,
) {
    let half = T::one() / (T::one() + T::one());
    ensure(&mut scratch.work, n * n);

    // Φ(L^T L̄). The full product is a GEMM rather than a triangular loop: it
    // does twice the necessary arithmetic in a blocked, vectorised kernel that
    // was already there, which beats a strided triple loop well before the
    // sizes anyone factorises.
    // SAFETY: `l`, `grad` and `scratch` are each `n * n` elements, checked by
    // the caller that allocated them.
    unsafe {
        T::gemm_tn(
            n,
            n,
            n,
            l.as_ptr(),
            grad.as_ptr(),
            scratch.work.as_mut_ptr(),
        )
    };
    for i in 0..n {
        scratch.work[i * n + i] = scratch.work[i * n + i] * half;
        for j in (i + 1)..n {
            scratch.work[i * n + j] = T::zero();
        }
    }

    // `work` is moved out so the solve can borrow the rest of the scratch
    // alongside it; it goes back before the function returns.
    let mut work = std::mem::take(&mut scratch.work);
    back_substitute_lt(l, &mut work, n, n, scratch);
    transpose_square(&mut work, n);
    back_substitute_lt(l, &mut work, n, n, scratch);

    for i in 0..n {
        for j in 0..=i {
            let averaged = (work[i * n + j] + work[j * n + i]) * half;
            out[i * n + j] = averaged;
            out[j * n + i] = averaged;
        }
    }
    scratch.work = work;
}

/// [`cholesky_backward_block`] over a whole batch, with one scratch buffer per
/// task rather than per matrix.
pub(crate) fn cholesky_backward_batched<T: Factorable>(
    factor: &[T],
    grad: &[T],
    out: &mut [T],
    n: usize,
    batch: usize,
) {
    let stride = n * n;
    let run = |first: usize, out_group: &mut [T]| {
        let mut scratch = BackwardScratch::new();
        for (local, block) in out_group.chunks_mut(stride).enumerate() {
            let offset = (first + local) * stride;
            cholesky_backward_block(
                &factor[offset..offset + stride],
                &grad[offset..offset + stride],
                block,
                &mut scratch,
                n,
            );
        }
    };

    if stride == 0 {
        return;
    }
    if batch <= 1 {
        run(0, out);
        return;
    }

    let per_task = (PAR_THRESHOLD / (n * n * n).max(1)).clamp(1, batch);
    let span = per_task * stride;
    crate::ops::map::par_out_chunks(out, span, &|start, out_group| {
        run((start / span) * per_task, out_group)
    });
}

/// The batch count and matrix order of an already-validated square stack.
pub(crate) fn square_extent(shape: &Shape) -> (usize, usize) {
    let dims = shape.dims();
    let n = dims[dims.len() - 1];
    let batch = dims[..dims.len() - 2].iter().product::<usize>().max(1);
    (batch, n)
}
