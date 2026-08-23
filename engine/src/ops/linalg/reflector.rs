// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Householder reflectors, and the blocked form that makes them fast.
//!
//! `qr` reduces a matrix to triangular form by reflecting one column at a time;
//! `svd` reduces it to bidiagonal form by reflecting a column and then a row.
//! Those are the same reflector -- `I - tau w w^T`, built to send a vector onto
//! a multiple of a basis vector -- differing only in which direction it is
//! gathered from and which side it is applied to. Written twice they would be
//! two places for the sign of `beta` to be chosen wrong, and the choice is the
//! entire numerical content of the thing.
//!
//! So the reflector is described once, by where it starts, how far apart its
//! elements are, and how many there are. A column of a row-major matrix is a
//! stride of `n`; a row is a stride of one; nothing else about the two cases
//! differs.
//!
//! The blocked form is here for the same reason. `H_start ... H_{stop-1} =
//! I - V T V^T` turns a sequence of rank-one updates into three matrix products,
//! and both callers want it.

use crate::ops::linalg::Factorable;
use num_traits::{Float, One, Zero};

/// Build the reflector that sends everything after the head of a strided run to
/// zero, leaving `beta` in the head and the reflector in the rest.
///
/// The run is `work[head]`, `work[head + step]`, ... `count` elements in all.
/// `step` is the matrix's row length for a column and one for a row, and that is
/// the only difference between the two.
///
/// Returns `tau`, which is zero when the run was already reduced -- the identity
/// is the reflector then, and the existing head stands whatever its sign.
///
/// The run is divided by its largest magnitude before anything is squared. That
/// is not a refinement: the sum of squares is the only place this squares
/// anything, and it overflows above about `1e154` in double and `1e19` in
/// single -- magnitudes a real matrix reaches. Above that, `norm` came out
/// infinite and the whole factorisation was `NaN`; below about `1e-154` the sum
/// underflowed to zero, the run was declared already reduced, and `qr` returned
/// a perfectly orthogonal `Q` with a residual of order one and no error at all.
/// The quiet failure was the worse of the two.
///
/// Only `beta` has to be scaled back. Every other quantity here is invariant:
/// `tau` is a ratio of two lengths, and each stored component is divided by a
/// third, so the factor cancels in all of them.
pub(crate) fn make<T: Factorable>(work: &mut [T], head: usize, step: usize, count: usize) -> T {
    let mut largest = T::Acc::zero();
    for i in 0..count {
        largest = largest.max(work[head + i * step].widen().abs());
    }
    if largest == T::Acc::zero() || !largest.is_finite() {
        return T::zero();
    }

    // Every ratio is at most one, so no term can overflow, and a term too small
    // to matter after scaling was too small to matter unscaled.
    let mut below = T::Acc::zero();
    for i in 1..count {
        let value = work[head + i * step].widen() / largest;
        below = below + value * value;
    }
    if below == T::Acc::zero() {
        return T::zero();
    }

    let alpha = work[head].widen() / largest;
    let norm = (alpha * alpha + below).sqrt();
    // Away from `alpha`, so that `alpha - beta` is a sum of magnitudes and
    // never the cancellation that would wreck the reflector.
    let beta = if alpha > T::Acc::zero() { -norm } else { norm };
    let tau = T::narrow((beta - alpha) / beta);
    let scale = T::Acc::one() / ((alpha - beta) * largest);
    for i in 1..count {
        work[head + i * step] = T::narrow(work[head + i * step].widen() * scale);
    }
    work[head] = T::narrow(beta * largest);
    tau
}

/// Gather a stored reflector into a contiguous vector, with the implicit leading
/// one made explicit.
///
/// `head` is the element the reflector was built from -- the one now holding
/// `beta` -- so the gather starts one step past it and `w[0]` is the one.
pub(crate) fn gather<T: Copy + One>(
    work: &[T],
    head: usize,
    step: usize,
    count: usize,
    w: &mut [T],
) {
    w[0] = T::one();
    for i in 1..count {
        w[i] = work[head + i * step];
    }
}

/// Apply `H = I - tau w w^T` to a window of `target` from the left, where
/// `w[0] = 1` and the rest of `w` is the stored reflector.
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
pub(crate) fn apply<T: Factorable>(
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

/// Apply `H = I - tau w w^T` to a window of `target` from the right.
///
/// The mirror of [`apply`] and not a transposed call to it: reflecting from the
/// right contracts `w` along a row, so each row's whole update -- the inner
/// product and the rank-one subtraction that follows -- happens while that row
/// is in registers, and no accumulator has to be carried across rows at all.
/// The left version needs one per column precisely because it cannot do that.
///
/// The inner product goes through [`Factorable::dot`] rather than a running sum
/// written out here, and that is not a tidiness choice. A running sum makes each
/// addition wait for the previous one, so the loop runs at the latency of a
/// floating-point add -- four cycles an element -- where the lane-split version
/// runs at its throughput. It is the more accurate spelling as well, for the
/// same reason: the error divides across the lanes instead of accumulating down
/// one chain.
pub(crate) fn apply_right<T: Factorable>(
    target: &mut [T],
    stride: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    w: &[T],
    tau: T,
) {
    if cols.is_empty() || tau == T::zero() {
        return;
    }
    let tau = tau.widen();
    let w = &w[..cols.len()];
    for i in rows {
        let row = &mut target[i * stride + cols.start..i * stride + cols.end];
        let scale = tau * T::dot(row, w);
        if scale == T::Acc::zero() {
            continue;
        }
        for (slot, &weight) in row.iter_mut().zip(w) {
            *slot = T::narrow(slot.widen() - scale * weight.widen());
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
pub(crate) const PANEL: usize = 32;

/// The buffers the block update works in, reused across panels and matrices.
pub(crate) struct Blocks<T: Factorable> {
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
    pub(crate) fn new() -> Self {
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
pub(crate) fn worth_blocking(rows: usize, width: usize, nb: usize) -> bool {
    nb >= 8 && rows >= 64 && width >= 64
}

/// Build `V` and `T` for the panel of column reflectors at `start`, so that
/// `H_start ... H_{stop-1} = I - V T V^T`.
///
/// This is LAPACK's `larft`. Column `p` of `T` is
/// `-tau_p * T[..p, ..p] (V[.., ..p]^T V[.., p])`, which is the statement that
/// adding one more reflector to a product of reflectors is a rank-one update of
/// the operator that represents them.
///
/// A zero `tau` is not a special case: it zeroes that reflector's row of `T`,
/// and the operator comes out as though the identity had been folded in, which
/// is what a zero `tau` means.
pub(crate) fn block<T: Factorable>(
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
pub(crate) fn apply_block<T: Factorable>(
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

/// Accumulate `H_0 H_1 ... H_{k-1}` into an identity, applying the reflectors in
/// reverse.
///
/// Reverse order is what lets each step touch only columns `j..`: applying
/// `H_j` to `H_{j+1} ... H_{k-1}` leaves the earlier columns as the basis
/// vectors they started as, because `w` is supported on rows `j..m` and those
/// columns are zero there. Going forwards instead would touch the whole matrix
/// every time, for the same answer and twice the work.
///
/// Panelled like the reduction that produced the reflectors, and the block form
/// is the one without the transpose: the product multiplies the
/// already-accumulated tail from the left in panel order, which is
/// `I - V T V^T` exactly.
///
/// A blocked panel covers columns from its own first one rather than from each
/// reflector's. The columns between are basis vectors the reflector leaves
/// alone -- `w` is supported on rows the column is zero in -- so including them
/// is at most [`PANEL`] no-op columns and saves a ragged loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate<T: Factorable>(
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
            block(work, m, n, start, nb, &tau[start..stop], blocks);
            apply_block(q, q_cols, m, start, nb, start..q_cols, false, blocks);
        } else {
            for j in (start..stop).rev() {
                let w = &mut reflectors[..m - j];
                gather(work, j * n + j, n, m - j, w);
                apply(q, q_cols, j..m, j..q_cols, w, tau[j], z);
            }
        }
        stop = start;
    }
}
