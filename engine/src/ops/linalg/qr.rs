// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The QR factorisation, `A = Q R` with `Q` orthonormal and `R` upper
//! triangular.
//!
//! `cholesky` needs a positive-definite matrix and `solve` needs a square one.
//! A rectangular matrix had nothing at all -- no least squares, no way to
//! orthonormalise a set of vectors, no stable basis for a subspace. Those are
//! not compositions of anything the library has: `Q` comes out of a sequence of
//! reflections, each one built from the column the previous ones left behind,
//! and no arrangement of `matmul` and `solve` performs that sequence.
//!
//! Householder reflections rather than Gram-Schmidt. Both produce the same `Q`
//! in exact arithmetic and they are not close in floating point: Gram-Schmidt
//! loses orthogonality in proportion to the condition number of `A`, while a
//! product of reflections is orthogonal to working precision whatever `A` was,
//! because each reflection is orthogonal *individually* and the error cannot
//! accumulate into the property. That is the whole reason to prefer it, and it
//! is what the `Q^T Q = I` test measures.
//!
//! The factorisation is never rejected -- a rank-deficient matrix has a QR,
//! with a zero on `R`'s diagonal. It is the *gradient* that needs `R`
//! invertible, and that is where the failure is reported.

use crate::{
    autograd::{QrBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        linalg::Factorable,
        map::{PAR_THRESHOLD, try_par_out_chunks_pair},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::{Float, One, Zero};
use std::sync::Arc;

/// How much of `Q` to return.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QrMode {
    /// `Q` is `[m, k]` and `R` is `[k, n]`, with `k = min(m, n)`. The columns of
    /// `Q` span the column space of `A` and nothing else.
    Reduced,
    /// `Q` is `[m, m]` and `R` is `[m, n]`. The extra columns complete `Q` to a
    /// full orthonormal basis -- and are an arbitrary choice among the bases
    /// that complete it, which is why the gradient refuses this shape when
    /// there are extra columns to choose.
    Complete,
}

impl QrMode {
    /// Parse the name the Python layer passes through.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "reduced" => Ok(QrMode::Reduced),
            "complete" => Ok(QrMode::Complete),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown qr mode {other:?}; expected \"reduced\" or \"complete\""
            ))),
        }
    }
}

/// The batch shape and the two matrix extents of a stack of matrices.
fn matrix_layout(tensor: &Tensor, op: &str) -> Result<(Vec<usize>, usize, usize)> {
    let ndim = tensor.ndim();
    if ndim < 2 {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects at least 2 dimensions"
        )));
    }
    if !matches!(tensor.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} supports only Float32 and Float64 tensors"
        )));
    }
    let dims = tensor.shape().dims();
    Ok((dims[..ndim - 2].to_vec(), dims[ndim - 2], dims[ndim - 1]))
}

/// Apply `H = I - tau w w^T` to a window of `target`, where `w[0] = 1` and the
/// rest of `w` is the stored reflector.
///
/// Written as a matrix-vector product followed by a rank-1 update rather than
/// as a per-column reflection, because both of those walk `target` one whole
/// row at a time. The per-column form reads down a column, which in row-major
/// storage is a stride of `n` per element and was the obvious way to make this
/// several times slower than it needs to be.
///
/// `z` accumulates in the wide type. It is one running sum per column over as
/// many rows as the matrix is tall, which is exactly the sum that grows, and
/// the buffer is one row long so carrying it wide costs nothing worth counting.
fn apply_reflector<T: Factorable>(
    target: &mut [T],
    stride: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    w: &[T],
    tau: T,
    z: &mut [T::Acc],
) {
    let width = cols.len();
    if width == 0 || tau == T::zero() {
        return;
    }
    let z = &mut z[..width];
    for slot in z.iter_mut() {
        *slot = T::Acc::zero();
    }

    // z = w^T * target[rows, cols]
    for (t, i) in rows.clone().enumerate() {
        let weight = w[t];
        if weight == T::zero() {
            continue;
        }
        let weight = weight.widen();
        let row = &target[i * stride + cols.start..i * stride + cols.end];
        for (acc, &value) in z.iter_mut().zip(row) {
            *acc = *acc + weight * value.widen();
        }
    }

    // target[rows, cols] -= tau * w z^T
    for (t, i) in rows.enumerate() {
        let scale = tau * w[t];
        if scale == T::zero() {
            continue;
        }
        let scale = scale.widen();
        let row = &mut target[i * stride + cols.start..i * stride + cols.end];
        for (slot, &value) in row.iter_mut().zip(z.iter()) {
            *slot = T::narrow(slot.widen() - scale * value);
        }
    }
}

/// How many reflectors are combined before the columns beyond them are brought
/// up to date.
///
/// Applying reflectors one at a time is two passes over the trailing block per
/// column: one to form `w^T C` and one to subtract `tau w (w^T C)`. That is a
/// single multiply-add per element loaded, and it is why the straightforward
/// version ran at 7.5 GFLOP/s against a GEMM's 43 on this machine -- 0.4-0.9x
/// LAPACK's time up to `n = 128` and 2.2x at `n = 512`.
///
/// It is not a cache problem, which was the first guess and was wrong: `L2` here
/// is 2MB a core, so a 512x512 double matrix is resident throughout, and tiling
/// the columns to keep them resident made it 25% *slower* by adding loop
/// structure to fix a miss that was not happening. The fix is arithmetic
/// intensity, which means combining the panel's reflectors into one operator and
/// applying it as a matrix product.
///
/// `H_start ... H_{stop-1} = I - V T V^T` is that operator -- the compact WY
/// form. Building `T` costs `O(PANEL^2 m)` per panel, a few percent of the
/// factorisation, and turns the other 90% into three GEMMs.
const PANEL: usize = 32;

/// Build the reflector that sends column `j` below the diagonal to zero,
/// leaving `beta` on the diagonal and the reflector beneath it.
fn make_reflector<T: Factorable>(work: &mut [T], m: usize, n: usize, j: usize) -> T {
    let mut below = T::Acc::zero();
    for i in (j + 1)..m {
        let value = work[i * n + j].widen();
        below = below + value * value;
    }
    if below == T::Acc::zero() {
        // Already zero underneath: the identity is the reflector, and the
        // existing entry stands as the diagonal whatever its sign.
        return T::zero();
    }

    let alpha = work[j * n + j].widen();
    let norm = (alpha * alpha + below).sqrt();
    // Away from `alpha`, so that `alpha - beta` is a sum of magnitudes and
    // never the cancellation that would wreck the reflector.
    let beta = if alpha > T::Acc::zero() { -norm } else { norm };
    let tau = T::narrow((beta - alpha) / beta);
    let scale = T::Acc::one() / (alpha - beta);
    for i in (j + 1)..m {
        work[i * n + j] = T::narrow(work[i * n + j].widen() * scale);
    }
    work[j * n + j] = T::narrow(beta);
    tau
}

/// Gather reflector `j` out of its column into a contiguous vector, with the
/// implicit leading one made explicit.
fn gather_reflector<T: Factorable>(work: &[T], m: usize, n: usize, j: usize, w: &mut [T]) {
    w[0] = T::one();
    for (t, i) in ((j + 1)..m).enumerate() {
        w[t + 1] = work[i * n + j];
    }
}

/// The buffers the block update works in, reused across panels and matrices.
struct Blocks<T: Factorable> {
    /// `V`, the panel's reflectors as an `(m - start) x nb` matrix with the
    /// implicit ones on its diagonal made explicit and zeros above.
    v: Vec<T>,
    /// `T`, upper triangular `nb x nb`.
    t: Vec<T>,
    /// `V^T C`, then `T^T V^T C` -- `nb` rows by as many columns as the block
    /// being updated.
    first: Vec<T>,
    second: Vec<T>,
    /// The block being updated, packed contiguously so the GEMMs can read it,
    /// and then overwritten with `V T^T V^T C`.
    block: Vec<T>,
}

impl<T: Factorable> Blocks<T> {
    fn new() -> Self {
        Self {
            v: Vec::new(),
            t: Vec::new(),
            first: Vec::new(),
            second: Vec::new(),
            block: Vec::new(),
        }
    }
}

/// Whether combining a panel into one operator pays for itself against applying
/// its reflectors one at a time.
///
/// The block form does three matrix products plus two copies of the block; the
/// direct form does `nb` pairs of sweeps over it and no copies at all. The
/// crossover is where the products are large enough to reach GEMM speed --
/// a narrow or short block spends its time in the packing instead, and the
/// first version of this measured 8x8 at three times its previous cost and
/// `1000x50` at a third again for exactly that reason.
fn worth_blocking(rows: usize, width: usize, nb: usize) -> bool {
    nb >= 8 && rows >= 64 && width >= 64
}

/// Build `V` and `T` for the panel at `start`, so that
/// `H_start ... H_{stop-1} = I - V T V^T`.
///
/// This is LAPACK's `larft`. Column `p` of `T` is
/// `-tau_p * T[..p, ..p] (V[.., ..p]^T V[.., p])`, which is the statement that
/// adding one more reflector to a product of reflectors is a rank-one update of
/// the operator that represents them.
fn block_reflector<T: Factorable>(
    work: &[T],
    m: usize,
    n: usize,
    start: usize,
    nb: usize,
    tau: &[T],
    blocks: &mut Blocks<T>,
) {
    let rows = m - start;
    blocks.v.clear();
    blocks.v.resize(rows * nb, T::zero());
    for p in 0..nb {
        blocks.v[p * nb + p] = T::one();
        for r in (p + 1)..rows {
            blocks.v[r * nb + p] = work[(start + r) * n + (start + p)];
        }
    }

    blocks.t.clear();
    blocks.t.resize(nb * nb, T::zero());
    let mut column = vec![T::Acc::zero(); nb];
    blocks.t[0] = tau[0];
    for p in 1..nb {
        if tau[p] == T::zero() {
            continue;
        }
        // column = -tau_p * V[p.., ..p]^T V[p.., p]
        for (q, slot) in column[..p].iter_mut().enumerate() {
            let mut sum = T::Acc::zero();
            for r in p..rows {
                sum = sum + blocks.v[r * nb + q].widen() * blocks.v[r * nb + p].widen();
            }
            *slot = -tau[p].widen() * sum;
        }
        // T[..p, p] = T[..p, ..p] * column, upper triangular so the sum starts
        // on the diagonal.
        for q in 0..p {
            let mut sum = T::Acc::zero();
            for u in q..p {
                sum = sum + blocks.t[q * nb + u].widen() * column[u];
            }
            blocks.t[q * nb + p] = T::narrow(sum);
        }
        blocks.t[p * nb + p] = tau[p];
    }
}

/// Apply `I - V T V^T` (or its transpose) to `target[start.., cols]`.
///
/// Three products: `V^T C`, then `T^T` (or `T`) against that, then `V` against
/// the result, subtracted from `C`. Each element of `C` is read once and
/// written once, where applying the reflectors one at a time read and wrote it
/// `nb` times -- that ratio is the whole point.
fn apply_block<T: Factorable>(
    target: &mut [T],
    stride: usize,
    m: usize,
    start: usize,
    nb: usize,
    cols: std::ops::Range<usize>,
    transposed: bool,
    blocks: &mut Blocks<T>,
) {
    let rows = m - start;
    let width = cols.len();
    if width == 0 || rows == 0 {
        return;
    }

    blocks.block.clear();
    blocks.block.resize(rows * width, T::zero());
    for r in 0..rows {
        let src = (start + r) * stride + cols.start;
        blocks.block[r * width..(r + 1) * width].copy_from_slice(&target[src..src + width]);
    }
    blocks.first.clear();
    blocks.first.resize(nb * width, T::zero());
    blocks.second.clear();
    blocks.second.resize(nb * width, T::zero());

    // SAFETY: `v` is `rows * nb`, `block` is `rows * width`, `t` is `nb * nb`
    // and both `first` and `second` are `nb * width`; every extent below is one
    // of those, and the three calls are the shapes documented on the trait.
    unsafe {
        // first = V^T C
        T::gemm_tn(
            nb,
            rows,
            width,
            blocks.v.as_ptr(),
            blocks.block.as_ptr(),
            blocks.first.as_mut_ptr(),
        );
        // second = T^T first, or T first
        if transposed {
            T::gemm_tn(
                nb,
                nb,
                width,
                blocks.t.as_ptr(),
                blocks.first.as_ptr(),
                blocks.second.as_mut_ptr(),
            );
        } else {
            T::gemm(
                nb,
                nb,
                width,
                blocks.t.as_ptr(),
                blocks.first.as_ptr(),
                blocks.second.as_mut_ptr(),
            );
        }
        // block = V second, overwriting the copy of `C` it no longer needs
        T::gemm(
            rows,
            nb,
            width,
            blocks.v.as_ptr(),
            blocks.second.as_ptr(),
            blocks.block.as_mut_ptr(),
        );
    }

    for r in 0..rows {
        let dst = (start + r) * stride + cols.start;
        let row = &mut target[dst..dst + width];
        let update = &blocks.block[r * width..(r + 1) * width];
        for (slot, &value) in row.iter_mut().zip(update) {
            *slot = *slot - value;
        }
    }
}

/// Reduce `work` to upper triangular form in place, leaving the reflector that
/// produced each column below that column's diagonal.
///
/// This is LAPACK's convention and it is worth keeping: the reflectors are
/// exactly as many numbers as the space `R`'s zeros leave free, so the whole
/// factorisation lives in one `m x n` buffer plus `k` scalars, and `Q` is built
/// from it afterwards rather than carried alongside.
fn householder<T: Factorable>(
    work: &mut [T],
    m: usize,
    n: usize,
    tau: &mut [T],
    reflectors: &mut [T],
    z: &mut [T::Acc],
    blocks: &mut Blocks<T>,
) {
    let k = m.min(n);
    let mut start = 0;
    while start < k {
        let nb = PANEL.min(k - start);
        let stop = start + nb;

        // The panel factors itself: each reflector reaches only the columns
        // still inside the panel, which is all the next reflector needs.
        for j in start..stop {
            tau[j] = make_reflector(work, m, n, j);
            let w = &mut reflectors[(j - start) * m..(j - start) * m + m];
            gather_reflector(work, m, n, j, w);
            apply_reflector(work, n, j..m, (j + 1)..stop, &w[..m - j], tau[j], z);
        }

        // Then the columns beyond it, all at once. The panel's reflectors
        // compose into `I - V T V^T`, so what was `nb` sweeps of the trailing
        // block becomes three matrix products over it.
        if stop < n {
            if worth_blocking(m - start, n - stop, nb) {
                block_reflector(work, m, n, start, nb, &tau[start..stop], blocks);
                apply_block(work, n, m, start, nb, stop..n, true, blocks);
            } else {
                // `reflectors` was filled while the panel factored itself, so
                // no column of `work` is gathered twice.
                for j in start..stop {
                    let w = &reflectors[(j - start) * m..(j - start) * m + m];
                    apply_reflector(work, n, j..m, stop..n, &w[..m - j], tau[j], z);
                }
            }
        }
        start = stop;
    }
}

/// Accumulate `Q = H_0 H_1 ... H_{k-1}` into an identity, applying the
/// reflectors in reverse.
///
/// Reverse order is what lets each step touch only columns `j..`: applying
/// `H_j` to `H_{j+1} ... H_{k-1}` leaves the earlier columns as the basis
/// vectors they started as, because `w` is supported on rows `j..m` and those
/// columns are zero there. Going forwards instead would touch the whole matrix
/// every time, for the same answer and twice the work.
///
/// Panelled like [`householder`] and for the same reason, and the block form is
/// the one without the transpose: `Q = H_0 ... H_{k-1}` multiplies the
/// already-accumulated tail from the left in panel order, which is
/// `I - V T V^T` exactly.
///
/// A blocked panel covers columns from its own first one rather than from each
/// reflector's. The columns between are basis vectors the reflector leaves
/// alone -- `w` is supported on rows the column is zero in -- so including them
/// is at most `PANEL` no-op columns and saves a ragged loop.
fn build_q<T: Factorable>(
    work: &[T],
    m: usize,
    n: usize,
    tau: &[T],
    q: &mut [T],
    q_cols: usize,
    reflectors: &mut [T],
    z: &mut [T::Acc],
    blocks: &mut Blocks<T>,
) {
    for i in 0..m.min(q_cols) {
        q[i * q_cols + i] = T::one();
    }
    let k = m.min(n).min(q_cols);
    let mut stop = k;
    while stop > 0 {
        let nb = PANEL.min(stop);
        let start = stop - nb;

        if worth_blocking(m - start, q_cols - start, nb) {
            block_reflector(work, m, n, start, nb, &tau[start..stop], blocks);
            // `Q = H_0 ... H_{k-1}` multiplies the accumulated tail from the
            // left in panel order, so this is `I - V T V^T` and not its
            // transpose.
            apply_block(q, q_cols, m, start, nb, start..q_cols, false, blocks);
        } else {
            for j in (start..stop).rev() {
                let w = &mut reflectors[..m];
                gather_reflector(work, m, n, j, w);
                apply_reflector(q, q_cols, j..m, j..q_cols, &w[..m - j], tau[j], z);
            }
        }
        stop = start;
    }
}

/// Copy the upper triangle out of the reduced `work` buffer.
///
/// Everything on or above the diagonal is `R`; everything below is the stored
/// reflectors, and the output was allocated zeroed, so only the triangle is
/// written.
fn extract_r<T: Copy>(work: &[T], n: usize, r: &mut [T], r_rows: usize) {
    for i in 0..r_rows {
        if i >= n {
            break;
        }
        r[i * n + i..i * n + n].copy_from_slice(&work[i * n + i..i * n + n]);
    }
}

/// Factor one matrix into `q` and `r`.
fn qr_one<T: Factorable>(
    a: &[T],
    m: usize,
    n: usize,
    q: &mut [T],
    q_cols: usize,
    r: &mut [T],
    r_rows: usize,
    scratch: &mut Scratch<T>,
) {
    let k = m.min(n);
    scratch.work.clear();
    scratch.work.extend_from_slice(a);
    scratch.tau.clear();
    scratch.tau.resize(k, T::zero());
    scratch.reflectors.clear();
    scratch.reflectors.resize(PANEL * m.max(1), T::zero());
    scratch.z.clear();
    scratch.z.resize(n.max(q_cols).max(1), T::Acc::zero());

    householder(
        &mut scratch.work,
        m,
        n,
        &mut scratch.tau,
        &mut scratch.reflectors,
        &mut scratch.z,
        &mut scratch.blocks,
    );
    extract_r(&scratch.work, n, r, r_rows);
    build_q(
        &scratch.work,
        m,
        n,
        &scratch.tau,
        q,
        q_cols,
        &mut scratch.reflectors,
        &mut scratch.z,
        &mut scratch.blocks,
    );
}

/// Scratch a single task reuses across every matrix it is handed.
struct Scratch<T: Factorable> {
    /// The matrix being reduced: `R` above the diagonal, reflectors below.
    work: Vec<T>,
    tau: Vec<T>,
    /// One panel's worth of gathered reflectors, `m` apart. Gathering them once
    /// per panel rather than once per tile is what keeps the strided read out
    /// of the inner loop.
    reflectors: Vec<T>,
    z: Vec<T::Acc>,
    blocks: Blocks<T>,
}

impl<T: Factorable> Scratch<T> {
    fn new() -> Self {
        Self {
            work: Vec::new(),
            tau: Vec::new(),
            reflectors: Vec::new(),
            z: Vec::new(),
            blocks: Blocks::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn qr_batched<T: Factorable>(
    input: &[T],
    q: &mut [T],
    r: &mut [T],
    m: usize,
    n: usize,
    q_cols: usize,
    r_rows: usize,
    batch: usize,
) {
    let a_stride = m * n;
    let q_stride = m * q_cols;
    let r_stride = r_rows * n;

    // `Result` only so the pair helper can be shared with routines that fail;
    // a factorisation has nothing to reject.
    // Every extent is non-zero here -- the caller checks -- so the group length
    // divides by the stride and there is no empty-group case to special-case.
    let _: Result<()> = try_par_out_chunks_pair(
        q,
        q_stride,
        r,
        r_stride,
        batch,
        (PAR_THRESHOLD / (m * n * n).max(1)).clamp(1, batch),
        &|first, q_group, r_group| {
            let mut scratch = Scratch::new();
            for local in 0..q_group.len() / q_stride {
                let offset = (first + local) * a_stride;
                qr_one(
                    &input[offset..offset + a_stride],
                    m,
                    n,
                    &mut q_group[local * q_stride..(local + 1) * q_stride],
                    q_cols,
                    &mut r_group[local * r_stride..(local + 1) * r_stride],
                    r_rows,
                    &mut scratch,
                );
            }
            Ok(())
        },
    );
}

/// `(Q, R)` for every matrix in a stack, with `A = Q @ R`.
///
/// `Q` has orthonormal columns and `R` is upper triangular. See [`QrMode`] for
/// the two shapes.
///
/// The factorisation itself never fails; a rank-deficient matrix simply has a
/// zero on `R`'s diagonal. The gradient is the part that needs `R` invertible.
pub fn qr(tensor: &Tensor, mode: QrMode) -> Result<(Tensor, Tensor)> {
    let (batch_dims, m, n) = matrix_layout(tensor, "qr")?;
    // A tensor with no batch dimensions holds one matrix; a batch dimension of
    // zero holds none. `product().max(1)` cannot tell those apart and answers
    // one for both, which is right for the first and a read past the end of an
    // empty input for the second.
    let batch = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product::<usize>()
    };
    let k = m.min(n);
    let (q_cols, r_rows) = match mode {
        QrMode::Reduced => (k, k),
        QrMode::Complete => (m, m),
    };

    let mut q_dims = batch_dims.clone();
    q_dims.extend_from_slice(&[m, q_cols]);
    let mut r_dims = batch_dims.clone();
    r_dims.extend_from_slice(&[r_rows, n]);
    let (q_shape, r_shape) = (Shape::new(q_dims), Shape::new(r_dims));

    let contiguous = tensor.contiguous()?;
    let mut q_data = TensorData::zeros_on_device(q_shape.numel(), tensor.dtype(), tensor.device());
    let mut r_data = TensorData::zeros_on_device(r_shape.numel(), tensor.dtype(), tensor.device());

    if m > 0 && n > 0 && batch > 0 {
        macro_rules! factor {
            ($accessor:ident, $accessor_mut:ident) => {{
                let src = contiguous.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("qr: dtype does not match the input slice")
                })?;
                let q_out = q_data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("qr: dtype does not match the Q slice")
                })?;
                let r_out = r_data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("qr: dtype does not match the R slice")
                })?;
                qr_batched(src, q_out, r_out, m, n, q_cols, r_rows, batch);
            }};
        }
        match tensor.dtype() {
            DataType::Float32 => factor!(as_f32_slice, as_f32_slice_mut),
            _ => factor!(as_f64_slice, as_f64_slice_mut),
        }
    }

    let requires_grad = tensor.requires_grad();
    let mut q = Tensor::new(
        Arc::new(q_data),
        q_shape,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );
    let mut r = Tensor::new(
        Arc::new(r_data),
        r_shape,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );

    if requires_grad {
        // The extra columns of a complete `Q` are an arbitrary completion of the
        // basis, so `Q` is not a function of `A` there and there is nothing to
        // differentiate. Saying so at the factorisation is better than handing
        // back a gradient for a choice the caller never made.
        if mode == QrMode::Complete && m > n {
            return Err(MinitensorError::invalid_operation(
                "qr is not differentiable in complete mode when there are more rows than columns; \
                 use mode=\"reduced\"",
            ));
        }
        let (detached_q, detached_r) = (q.detach(), r.detach());
        let node = |from_q: bool| {
            Arc::new(QrBackward {
                input: tensor.detach(),
                q: detached_q.clone(),
                r: detached_r.clone(),
                from_q,
                input_id: tensor.id(),
                ids: [tensor.id()],
            })
        };
        q = with_grad_fn(q, node(true))?;
        r = with_grad_fn(r, node(false))?;
    }
    Ok((q, r))
}
