// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The singular value decomposition, `A = U diag(s) V^T`.
//!
//! Every other factorisation in this module needs something of its matrix.
//! `cholesky` needs positive definiteness, `eigh` symmetry, `solve` and `det`
//! squareness; `qr` takes any matrix but only tells you about its columns in the
//! order they arrived. The singular value decomposition asks for nothing at all
//! and answers the questions that do not depend on the matrix being any of
//! those: how far it is from a lower rank, which directions it stretches and by
//! how much, what its rank is when the entries are inexact, and what the
//! least-squares solution is when the columns are dependent. `pinv`,
//! `matrix_rank`, `cond`, principal component analysis on a data matrix rather
//! than its covariance, and low-rank truncation are all one call away from it
//! and unreachable without it.
//!
//! It is the other factorisation that iterates, and it is not `eigh` in
//! disguise. The eigenvalues of `A^T A` are the squares of the singular values,
//! which is a proof and not an algorithm: forming `A^T A` squares the condition
//! number, so a singular value at `1e-9` times the largest -- perfectly
//! representable, and exactly the one a rank test is asking about -- comes back
//! from the squared problem with no correct digits at all. The work here is
//! arranged so that `A` is never squared.
//!
//! Two phases, as in `eigh`. Householder reflections from the left and the right
//! reduce `A` to bidiagonal form in a fixed number of steps, using the same
//! reflector `qr` is built from -- see [`reflector`]. Then implicitly shifted QR
//! sweeps chase the superdiagonal to zero, each sweep a chain of plane rotations
//! applied to the band and accumulated into `U` and `V`. Because every step is
//! an exact orthogonal transformation of the current band, the singular values
//! that come out are the exact singular values of a matrix within rounding of
//! `A`, whatever the shift did -- the shift buys convergence speed, not
//! accuracy, which is why an approximate one is not a compromise.
//!
//! Singular values come back in descending order, as LAPACK's `gesdd` and
//! NumPy's `svd` give them. `U` and `V` are determined up to the sign of each
//! column -- and up to a rotation within any repeated singular value's subspace
//! -- so nothing here imposes a convention and nothing should rely on one.

use crate::{
    autograd::{SvdBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        linalg::{Factorable, matrix_layout, reflector, rotation},
        map::{PAR_THRESHOLD, try_par_out_chunks_triple},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::{Float, Zero};
use std::sync::Arc;

/// Reduce `work` to upper bidiagonal form, leaving both families of reflectors
/// behind in the space the zeros freed.
///
/// One reflector per column sends the entries below the diagonal to zero, and
/// one per row sends the entries right of the superdiagonal to zero. They are
/// the same reflector applied from opposite sides, which is why both come out of
/// [`reflector::make`] with only the stride between elements differing: a column
/// of a row-major matrix steps by its row length, a row steps by one.
///
/// The column reflectors are left in the columns they cleared, which is exactly
/// the layout [`reflector::accumulate`] reads. The row reflectors are left in
/// their rows as well -- there is nowhere else for them to go -- but a copy is
/// scattered into `right` transposed, and shifted one place along, so that the
/// same accumulation builds `V` without a second version of it existing. The
/// shift is because a row reflector starts one column later than a column
/// reflector of the same index; storing it as reflector `j + 1` makes the two
/// layouts identical, and reflector `0` of `right` is then an identity that
/// costs nothing because its `tau` is zero.
fn bidiagonalize<T: Factorable>(scratch: &mut Scratch<T>, rows: usize, cols: usize) {
    for j in 0..cols {
        let tau = reflector::make(&mut scratch.work, j * cols + j, cols, rows - j);
        scratch.tau_left[j] = tau;
        let w = &mut scratch.reflectors[..rows - j];
        reflector::gather(&scratch.work, j * cols + j, cols, rows - j, w);
        reflector::apply(
            &mut scratch.work,
            cols,
            j..rows,
            (j + 1)..cols,
            w,
            tau,
            &mut scratch.z,
        );

        // Nothing lies right of the superdiagonal in the last two rows, so the
        // reduction from the right stops two columns early.
        if j + 2 >= cols {
            continue;
        }
        let tau = reflector::make(&mut scratch.work, j * cols + j + 1, 1, cols - j - 1);
        scratch.tau_right[j + 1] = tau;
        let w = &mut scratch.reflectors[..cols - j - 1];
        reflector::gather(&scratch.work, j * cols + j + 1, 1, cols - j - 1, w);
        reflector::apply_right(
            &mut scratch.work,
            cols,
            (j + 1)..rows,
            (j + 1)..cols,
            w,
            tau,
        );
        for i in (j + 2)..cols {
            scratch.right[i * cols + j + 1] = scratch.work[j * cols + i];
        }
    }
}

/// The Wilkinson shift for the band `l..=bottom`.
///
/// It is the eigenvalue of the trailing two-by-two of `B^T B` nearer to that
/// matrix's corner, which is the shift that makes the sweep converge cubically
/// once it is close. `B^T B` is formed only for those four entries, and only to
/// choose the shift: the band itself is never squared, and the caller has
/// already scaled it so that the four squares cannot overflow. The subtraction
/// is written so that it never cancels.
fn wilkinson_shift<T: Float>(d: &[T], e: &[T], l: usize, bottom: usize) -> T {
    let two = T::one() + T::one();
    let above = if bottom > l + 1 {
        e[bottom - 2]
    } else {
        T::zero()
    };
    let corner = d[bottom - 1] * d[bottom - 1] + above * above;
    let cross = d[bottom - 1] * e[bottom - 1];
    let last = d[bottom] * d[bottom] + e[bottom - 1] * e[bottom - 1];
    if cross == T::zero() {
        return last;
    }
    let half = (corner - last) / two;
    let root = rotation::hypotenuse(half, cross);
    let denominator = if half < T::zero() {
        half - root
    } else {
        half + root
    };
    last - cross * cross / denominator
}

/// One implicitly shifted QR sweep over the band `l..=bottom`.
///
/// Implicit means the shift is never subtracted from anything: it is folded
/// into the first rotation of a chain, and the rest of the chain chases the
/// bulge that rotation creates off the end of the band. Alternating sides -- a
/// rotation on the right, then one on the left -- is what keeps the band
/// bidiagonal throughout, with exactly one entry out of place at any moment.
///
/// The first pair is `(d_l^2 - mu, d_l e_l)` divided through by `d_l`, which a
/// rotation does not notice and which keeps the pair away from the squares. The
/// split loop guarantees `d_l` is not negligible whenever there is a sweep to
/// do, so the division is safe exactly when it is reached.
///
/// A shift below the rounding of `d_l^2` cannot survive that division:
/// `d_l - mu/d_l` *is* `d_l`, so the sweep that would actually run is the
/// unshifted one, and the unshifted one is worth writing differently. See
/// [`zero_shift_chase`].
#[allow(clippy::too_many_arguments)]
fn sweep<T: Float + Send + Sync>(
    d: &mut [T],
    e: &mut [T],
    l: usize,
    bottom: usize,
    u: &mut [T],
    u_rows: usize,
    u_cols: usize,
    v: &mut [T],
    n: usize,
    left: &mut rotation::Chain<T>,
    right: &mut rotation::Chain<T>,
) {
    left.start(l, rotation::Order::Rising);
    right.start(l, rotation::Order::Rising);
    let shift = wilkinson_shift(d, e, l, bottom);
    if shift.abs() <= T::epsilon() * d[l] * d[l] {
        zero_shift_chase(d, e, l, bottom, left, right);
    } else {
        shifted_chase(d, e, l, bottom, shift, left, right);
    }
    left.apply(u, u_rows, u_cols);
    right.apply(v, n, n);
}

/// The chase with a shift, carrying the bulge from one rotation to the next.
fn shifted_chase<T: Float + Send + Sync>(
    d: &mut [T],
    e: &mut [T],
    l: usize,
    bottom: usize,
    shift: T,
    left: &mut rotation::Chain<T>,
    right: &mut rotation::Chain<T>,
) {
    let (mut f, mut g) = (d[l] - shift / d[l], e[l]);
    for i in l..bottom {
        // From the right, against columns i and i+1: clears the bulge left
        // above the band and pushes a new one below it.
        let (c, s, r) = rotation::plane(f, g);
        if i > l {
            e[i - 1] = r;
        }
        f = c * d[i] - s * e[i];
        e[i] = s * d[i] + c * e[i];
        g = -s * d[i + 1];
        d[i + 1] = c * d[i + 1];
        right.push(c, s);

        // From the left, against rows i and i+1: clears that bulge and pushes
        // the next one back above the band, one column further along.
        let (c, s, r) = rotation::plane(f, g);
        d[i] = r;
        f = c * e[i] - s * d[i + 1];
        d[i + 1] = s * e[i] + c * d[i + 1];
        if i + 1 < bottom {
            g = -s * e[i + 1];
            e[i + 1] = c * e[i + 1];
        }
        left.push(c, s);
    }
    e[bottom - 1] = f;
}

/// The same sweep with no shift, in the arrangement that does the work.
///
/// [`shifted_chase`] carries the bulge along as an explicit value, and with no
/// shift to seed it that value starts at `e_l` and is multiplied by a sine at
/// every step. Down a band whose entries span twelve orders of magnitude it
/// arrives at the far end many orders below the rounding of what is there, so
/// the rotations at the bottom are the identity to working precision and the
/// sweep moves nothing at all: measured on a 128-wide band with a condition
/// number of `1e12`, sixty-five consecutive sweeps left the last superdiagonal
/// entry unchanged in every digit it has. The band came apart in the end, but
/// because some *other* entry of it eventually went negligible -- the sweeps
/// aimed at the bottom were spent and bought nothing.
///
/// Demmel and Kahan's arrangement never forms that value. Each rotation is
/// generated from the band entry it acts on times a running cosine, so it has
/// the magnitude the entries *there* deserve however small the ones above it
/// became, and the sweep does the same work at the bottom of the band as at the
/// top. These are the same rotations the explicit chase describes -- on a band
/// with nothing small in it the two agree to a part in `1e15` -- so this is not
/// a different algorithm, only the spelling of it that a graded band does not
/// flatten. It is worth the second spelling for what it saves: across graded
/// bands from 64 to 512 wide it never costs more sweeps and often costs half as
/// many, 272 against 579 on the worst measured.
///
/// `d[i]` and `e[i]` are read before either is written, which is what lets each
/// rotation come from the entries the *previous* sweep left rather than from
/// the ones this sweep has already touched.
fn zero_shift_chase<T: Float + Send + Sync>(
    d: &mut [T],
    e: &mut [T],
    l: usize,
    bottom: usize,
    left: &mut rotation::Chain<T>,
    right: &mut rotation::Chain<T>,
) {
    // The running cosine of the rotation applied from the right, and the pair
    // of the one applied from the left, both carried into the next step.
    let mut carried = T::one();
    let mut held = (T::one(), T::zero());
    for i in l..bottom {
        let (c, s, r) = rotation::plane(d[i] * carried, e[i]);
        if i > l {
            e[i - 1] = -held.1 * r;
        }
        let (hc, hs, hr) = rotation::plane(held.0 * r, -s * d[i + 1]);
        d[i] = hr;
        right.push(c, s);
        left.push(hc, hs);
        carried = c;
        held = (hc, hs);
    }
    let tail = d[bottom] * carried;
    d[bottom] = tail * held.0;
    e[bottom - 1] = -held.1 * tail;
}

/// Rotate a negligible diagonal entry's row out of the band.
///
/// A zero on the diagonal is where the shifted sweep breaks down -- the shift is
/// derived by dividing by `d[l]`, and more fundamentally the band no longer
/// couples the way the chase assumes. The row is decoupled instead: a chain of
/// rotations against the rows below carries the one entry left in row `z`
/// along the superdiagonal and off the end, after which `e[z]` is zero, the
/// problem splits there, and `d[z]` is a singular value of its own.
///
/// `d[z]` is set to zero first rather than left at its measured size. It is
/// already below the rounding of the matrix, and each rotation would otherwise
/// smear a multiple of it into the subdiagonal, which is a fill-in the band has
/// no room for. Dropping it is a perturbation of the same order as the arithmetic
/// that produced it.
#[allow(clippy::too_many_arguments)]
fn decouple<T: Float + Send + Sync>(
    d: &mut [T],
    e: &mut [T],
    z: usize,
    bottom: usize,
    u: &mut [T],
    u_rows: usize,
    u_cols: usize,
    left: &mut rotation::Chain<T>,
) {
    left.start(z + 1, rotation::Order::Fanned(z));
    let mut f = e[z];
    d[z] = T::zero();
    e[z] = T::zero();
    for i in (z + 1)..=bottom {
        // The rotation that clears `f` and keeps `d[i]`. `plane` clears its
        // second argument, so the pair goes in reversed, and negated so that
        // the surviving entry comes out with `d[i]`'s own sign rather than an
        // imposed one.
        let (c, s, r) = rotation::plane(d[i], -f);
        d[i] = r;
        left.push(c, s);
        if i == bottom {
            break;
        }
        f = -s * e[i];
        e[i] = c * e[i];
        if f == T::zero() {
            break;
        }
    }
    left.apply(u, u_rows, u_cols);
}

/// Drive the bidiagonal band to diagonal form, accumulating into `u` and `v`.
///
/// The band is scaled to a power of two first. Only the shift squares anything,
/// and only four entries of it, but a matrix whose entries reach `1e20` in
/// single precision would overflow those squares while every singular value it
/// has is perfectly representable. A power of two makes the scaling exact in
/// both directions, so this costs no accuracy at all -- and where it underflows
/// instead, the shift comes out zero, which is the sweep that is *more* accurate
/// and merely slower.
///
/// Two things end a band. A negligible superdiagonal entry splits it, and the
/// test for that is relative to the diagonal entries either side rather than to
/// the norm of the whole matrix, which is what lets a singular value many orders
/// below the largest still be computed to its own precision. A negligible
/// *diagonal* entry is the other, and it is handled by [`decouple`].
#[allow(clippy::too_many_arguments)]
fn diagonalize<T: Float + Send + Sync>(
    d: &mut [T],
    e: &mut [T],
    n: usize,
    u: &mut [T],
    u_rows: usize,
    u_cols: usize,
    v: &mut [T],
    left: &mut rotation::Chain<T>,
    right: &mut rotation::Chain<T>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let mut norm = T::zero();
    for j in 0..n {
        norm = norm.max(d[j].abs() + e[j].abs());
    }
    if norm == T::zero() {
        return Ok(());
    }
    let scale = norm.log2().floor().exp2();
    let scale = if scale.is_normal() { scale } else { T::one() };
    for j in 0..n {
        d[j] = d[j] / scale;
        e[j] = e[j] / scale;
    }
    let norm = norm / scale;
    let epsilon = T::epsilon();

    // One budget for the whole band rather than one per value: the sweeps a
    // graded band needs land almost entirely on the first value it deflates,
    // so a per-value cap has to be set for that one and is then far looser
    // than it needs to be for all the rest. See `rotation::check_sweeps`.
    let mut sweeps = 0usize;
    let mut k = n;
    while k > 0 {
        let bottom = k - 1;
        loop {
            // Walk up from the bottom for the first entry that ends the band.
            let mut l = bottom;
            let mut negligible = None;
            while l > 0 {
                if e[l - 1].abs() <= epsilon * (d[l - 1].abs() + d[l].abs()) {
                    e[l - 1] = T::zero();
                    break;
                }
                if d[l - 1].abs() <= epsilon * norm {
                    negligible = Some(l - 1);
                    break;
                }
                l -= 1;
            }
            if let Some(z) = negligible {
                decouple(d, e, z, bottom, u, u_rows, u_cols, left);
                l = z + 1;
            }
            if l == bottom {
                // A one-entry band is a converged singular value. It is defined
                // non-negative and the iteration does not promise that, so the
                // sign moves onto `V`'s column, where it cancels exactly.
                if d[bottom] < T::zero() {
                    d[bottom] = -d[bottom];
                    rotation::negate_column(v, n, n, bottom);
                }
                break;
            }
            sweeps += 1;
            rotation::check_sweeps(sweeps, n, "svd")?;
            sweep(d, e, l, bottom, u, u_rows, u_cols, v, n, left, right);
        }
        k = bottom;
    }

    for value in d[..n].iter_mut() {
        *value = *value * scale;
    }
    Ok(())
}

/// Everything one task reuses across the matrices it is handed.
struct Scratch<T: Factorable> {
    /// The matrix being reduced, `rows x cols`, holding the bidiagonal and both
    /// families of reflectors when the reduction is done.
    work: Vec<T>,
    /// The row reflectors again, transposed and shifted into the layout
    /// [`reflector::accumulate`] reads. `cols x cols`.
    right: Vec<T>,
    u: Vec<T>,
    v: Vec<T>,
    d: Vec<T>,
    e: Vec<T>,
    tau_left: Vec<T>,
    tau_right: Vec<T>,
    reflectors: Vec<T>,
    z: Vec<T::Acc>,
    blocks: reflector::Blocks<T>,
    /// The sweeps' rotations, held here rather than built per matrix so a batch
    /// of small ones reuses the capacity like everything else in this struct.
    left_chain: rotation::Chain<T>,
    right_chain: rotation::Chain<T>,
}

impl<T: Factorable> Scratch<T> {
    fn new() -> Self {
        Self {
            work: Vec::new(),
            right: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            d: Vec::new(),
            e: Vec::new(),
            tau_left: Vec::new(),
            tau_right: Vec::new(),
            reflectors: Vec::new(),
            z: Vec::new(),
            blocks: reflector::Blocks::new(),
            left_chain: rotation::Chain::new(),
            right_chain: rotation::Chain::new(),
        }
    }

    fn resize(&mut self, rows: usize, cols: usize, u_cols: usize) {
        let fill = |buffer: &mut Vec<T>, len: usize| {
            buffer.clear();
            buffer.resize(len, T::zero());
        };
        fill(&mut self.work, rows * cols);
        fill(&mut self.right, cols * cols);
        fill(&mut self.u, rows * u_cols);
        fill(&mut self.v, cols * cols);
        fill(&mut self.d, cols);
        fill(&mut self.e, cols);
        fill(&mut self.tau_left, cols);
        fill(&mut self.tau_right, cols);
        fill(&mut self.reflectors, rows.max(cols));
        self.z.clear();
        self.z.resize(rows.max(u_cols).max(cols), T::Acc::zero());
    }
}

/// Factor one matrix, leaving `U` in `scratch.u`, the singular values in
/// `scratch.d` and `V` in `scratch.v`.
///
/// The reduction wants at least as many rows as columns -- the row reflectors
/// would otherwise run out of band before the column ones do. A wider matrix is
/// transposed on the way in, which costs nothing because it is being copied
/// anyway, and its factorisation is read back out with `U` and `V` exchanged:
/// `A^T = U s V^T` is exactly `A = V s U^T`. That is the whole handling of the
/// wide case, and it is why nothing below has a second branch for it.
fn factor_one<T: Factorable>(
    a: &[T],
    m: usize,
    n: usize,
    full: bool,
    scratch: &mut Scratch<T>,
) -> Result<()> {
    let (rows, cols) = if m >= n { (m, n) } else { (n, m) };
    let u_cols = if full { rows } else { cols };
    scratch.resize(rows, cols, u_cols);

    if m >= n {
        scratch.work.copy_from_slice(a);
    } else {
        for i in 0..rows {
            for j in 0..cols {
                scratch.work[i * cols + j] = a[j * n + i];
            }
        }
    }

    bidiagonalize(scratch, rows, cols);
    for j in 0..cols {
        scratch.d[j] = scratch.work[j * cols + j];
        if j + 1 < cols {
            scratch.e[j] = scratch.work[j * cols + j + 1];
        }
    }

    reflector::accumulate(
        &scratch.work,
        rows,
        cols,
        &scratch.tau_left,
        &mut scratch.u,
        u_cols,
        &mut scratch.reflectors,
        &mut scratch.z,
        &mut scratch.blocks,
    );
    reflector::accumulate(
        &scratch.right,
        cols,
        cols,
        &scratch.tau_right,
        &mut scratch.v,
        cols,
        &mut scratch.reflectors,
        &mut scratch.z,
        &mut scratch.blocks,
    );

    diagonalize(
        &mut scratch.d,
        &mut scratch.e,
        cols,
        &mut scratch.u,
        rows,
        u_cols,
        &mut scratch.v,
        &mut scratch.left_chain,
        &mut scratch.right_chain,
    )?;

    let mut carried = [(&mut scratch.u[..], u_cols), (&mut scratch.v[..], cols)];
    rotation::sort_carrying_columns(&mut scratch.d, true, &mut carried);
    Ok(())
}

/// Copy `source`, which is `rows x cols`, into `target`, transposing if asked.
///
/// The four ways a factorisation's two matrices can land in its two outputs are
/// this one function with different arguments, because transposing `A` on the
/// way in is undone by transposing both matrices on the way out and exchanging
/// them.
fn place<T: Copy>(source: &[T], rows: usize, cols: usize, target: &mut [T], transposed: bool) {
    if transposed {
        for i in 0..rows {
            for j in 0..cols {
                target[j * rows + i] = source[i * cols + j];
            }
        }
    } else {
        target[..rows * cols].copy_from_slice(&source[..rows * cols]);
    }
}

macro_rules! svd_kernel {
    ($name:ident, $ty:ty, $accessor:ident, $accessor_mut:ident) => {
        /// Factor every matrix in the batch into the three output buffers.
        #[allow(clippy::too_many_arguments)]
        fn $name(
            input: &Tensor,
            u_data: &mut TensorData,
            s_data: &mut TensorData,
            vt_data: &mut TensorData,
            batch: usize,
            m: usize,
            n: usize,
            full: bool,
        ) -> Result<()> {
            let mismatch =
                || MinitensorError::internal_error("svd: dtype does not match the buffer");
            let a = input.data().$accessor().ok_or_else(mismatch)?;
            let k = m.min(n);
            let (u_cols, vt_rows) = if full { (m, n) } else { (k, k) };
            let (a_stride, u_stride, vt_stride) = (m * n, m * u_cols, vt_rows * n);
            let u = u_data.$accessor_mut().ok_or_else(mismatch)?;
            let s = s_data.$accessor_mut().ok_or_else(mismatch)?;
            let vt = vt_data.$accessor_mut().ok_or_else(mismatch)?;

            try_par_out_chunks_triple(
                u,
                u_stride,
                s,
                k,
                vt,
                vt_stride,
                batch,
                (PAR_THRESHOLD / (m * n * k).max(1)).clamp(1, batch),
                &|first, u_group, s_group, vt_group| {
                    let mut scratch = Scratch::new();
                    for local in 0..u_group.len() / u_stride {
                        let offset = (first + local) * a_stride;
                        factor_one(&a[offset..offset + a_stride], m, n, full, &mut scratch)?;
                        s_group[local * k..(local + 1) * k].copy_from_slice(&scratch.d[..k]);
                        let u_out = &mut u_group[local * u_stride..(local + 1) * u_stride];
                        let vt_out = &mut vt_group[local * vt_stride..(local + 1) * vt_stride];
                        // `factor_one` worked on `A` when it is tall and on
                        // `A^T` when it is wide, so the tall case reads `U`
                        // straight across and transposes `V`, and the wide case
                        // does the opposite with the two exchanged.
                        if m >= n {
                            place(&scratch.u, m, u_cols, u_out, false);
                            place(&scratch.v, n, n, vt_out, true);
                        } else {
                            place(&scratch.v, m, m, u_out, false);
                            place(&scratch.u, n, vt_rows, vt_out, true);
                        }
                    }
                    Ok(())
                },
            )
        }
    };
}

svd_kernel!(svd_f32, f32, as_f32_slice, as_f32_slice_mut);
svd_kernel!(svd_f64, f64, as_f64_slice, as_f64_slice_mut);

/// Write a stack of identity matrices into `data`.
///
/// Only [`decompose`] wants this, and only for a matrix with no rows or no
/// columns: the factorisation has nothing to do, but its two orthogonal factors
/// still have a shape and still have to be orthogonal when they get there.
fn fill_identity(
    data: &mut TensorData,
    dtype: DataType,
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    macro_rules! fill {
        ($accessor:ident, $ty:ty) => {{
            let slice = data.$accessor().ok_or_else(|| {
                MinitensorError::internal_error("svd: dtype does not match the buffer")
            })?;
            for b in 0..batch {
                for i in 0..rows.min(cols) {
                    slice[b * rows * cols + i * cols + i] = 1 as $ty;
                }
            }
        }};
    }
    match dtype {
        DataType::Float32 => fill!(as_f32_slice_mut, f32),
        _ => fill!(as_f64_slice_mut, f64),
    }
    Ok(())
}

/// Shared body for the two public entry points.
fn decompose(
    tensor: &Tensor,
    full: bool,
    requires_grad: bool,
    op: &str,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch_dims, m, n) = matrix_layout(tensor, op)?;
    // A tensor with no batch dimensions holds one matrix; a batch dimension of
    // zero holds none, and `product().max(1)` cannot tell those apart.
    let batch = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product::<usize>()
    };
    let k = m.min(n);
    let (u_cols, vt_rows) = if full { (m, n) } else { (k, k) };

    let mut u_dims = batch_dims.clone();
    u_dims.extend_from_slice(&[m, u_cols]);
    let mut s_dims = batch_dims.clone();
    s_dims.push(k);
    let mut vt_dims = batch_dims;
    vt_dims.extend_from_slice(&[vt_rows, n]);
    let (u_shape, s_shape, vt_shape) =
        (Shape::new(u_dims), Shape::new(s_dims), Shape::new(vt_dims));

    let contiguous = tensor.contiguous()?;
    let mut u_data = TensorData::zeros_on_device(u_shape.numel(), tensor.dtype(), tensor.device());
    let mut s_data = TensorData::zeros_on_device(s_shape.numel(), tensor.dtype(), tensor.device());
    let mut vt_data =
        TensorData::zeros_on_device(vt_shape.numel(), tensor.dtype(), tensor.device());

    if batch > 0 && (m == 0 || n == 0) {
        // Nothing to factor -- but `U` and `V^T` are still orthogonal matrices
        // of the size that was asked for, and a caller is entitled to use them
        // as such. Zero is not one of those. NumPy answers the same way.
        fill_identity(&mut u_data, tensor.dtype(), batch, m, u_cols)?;
        fill_identity(&mut vt_data, tensor.dtype(), batch, vt_rows, n)?;
    } else if batch > 0 {
        match tensor.dtype() {
            DataType::Float32 => svd_f32(
                &contiguous,
                &mut u_data,
                &mut s_data,
                &mut vt_data,
                batch,
                m,
                n,
                full,
            )?,
            _ => svd_f64(
                &contiguous,
                &mut u_data,
                &mut s_data,
                &mut vt_data,
                batch,
                m,
                n,
                full,
            )?,
        }
    }

    let build = |data: TensorData, shape: Shape| {
        Tensor::new(
            Arc::new(data),
            shape,
            tensor.dtype(),
            tensor.device(),
            requires_grad,
        )
    };
    Ok((
        build(u_data, u_shape),
        build(s_data, s_shape),
        build(vt_data, vt_shape),
    ))
}

/// `(U, s, V^T)` for every matrix in a stack, with `A = U @ diag(s) @ V^T`.
///
/// `s` is non-negative and descending. With `full_matrices` the two orthogonal
/// factors are square -- `[m, m]` and `[n, n]` -- and without it they are cut to
/// the `min(m, n)` columns that carry a singular value, which is the shape that
/// reconstructs `A` and the one almost every use wants.
///
/// The columns of `U` and `V` are determined only up to sign, and within a
/// repeated singular value's subspace only up to a rotation. Compare what they
/// do rather than what they are.
pub fn svd(tensor: &Tensor, full_matrices: bool) -> Result<(Tensor, Tensor, Tensor)> {
    let requires_grad = tensor.requires_grad();
    let (mut u, mut s, mut vt) = decompose(tensor, full_matrices, requires_grad, "svd")?;

    if requires_grad {
        // `decompose` already rejected anything with fewer than two dimensions.
        let dims = tensor.shape().dims();
        let (m, n) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        // The columns of a square `U` beyond the `n`th, and of a square `V^T`
        // beyond the `m`th, are an arbitrary completion of an orthonormal basis:
        // nothing about `A` chooses them, so there is no derivative to report.
        // Saying so here beats handing back a gradient for a choice the caller
        // never made -- the same reason `qr` refuses its complete mode.
        if full_matrices && m != n {
            return Err(MinitensorError::invalid_operation(
                "svd is not differentiable with full_matrices=True unless the matrix is square; \
                 use full_matrices=False",
            ));
        }
        // Three outputs and one gradient at a time, so each output gets its own
        // node and the engine adds what they produce. See `SvdBackward`.
        let (du, ds, dvt) = (u.detach(), s.detach(), vt.detach());
        let node = |from| {
            Arc::new(SvdBackward {
                u: du.clone(),
                values: ds.clone(),
                vt: dvt.clone(),
                from,
                input_id: tensor.id(),
                ids: [tensor.id()],
            })
        };
        u = with_grad_fn(u, node(crate::autograd::SvdOutput::U))?;
        s = with_grad_fn(s, node(crate::autograd::SvdOutput::Values))?;
        vt = with_grad_fn(vt, node(crate::autograd::SvdOutput::Vt))?;
    }
    Ok((u, s, vt))
}

/// The singular values alone, descending.
///
/// The orthogonal factors are still accumulated -- unlike `eigvalsh`, whose
/// iteration touches three diagonals and can genuinely skip the `n^3` part,
/// the reduction to bidiagonal form here is `n^3` whether or not anyone wants
/// the vectors. What this saves is the two output copies and the gradient
/// machinery, and what it offers is a name that says what is being asked.
pub fn svdvals(tensor: &Tensor) -> Result<Tensor> {
    if tensor.requires_grad() {
        return Ok(svd(tensor, false)?.1);
    }
    Ok(decompose(tensor, false, false, "svdvals")?.1)
}

#[cfg(test)]
mod band_tests {
    use super::*;

    /// A dense `n x n` copy of the bidiagonal band `(d, e)`.
    fn dense(d: &[f64], e: &[f64], n: usize) -> Vec<f64> {
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            b[i * n + i] = d[i];
            if i + 1 < n {
                b[i * n + i + 1] = e[i];
            }
        }
        b
    }

    fn identity(n: usize) -> Vec<f64> {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    fn multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let left = a[i * n + k];
                if left == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[i * n + j] += left * b[k * n + j];
                }
            }
        }
        out
    }

    fn transposed(a: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[j * n + i] = a[i * n + j];
            }
        }
        out
    }

    fn largest(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |acc, x| acc.max(x.abs()))
    }

    fn difference(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0f64, |acc, (x, y)| acc.max((x - y).abs()))
    }

    /// Run the iteration over a whole band, returning `(values, U, V)`.
    fn factor(d: &[f64], e: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (mut d, mut e) = (d.to_vec(), e.to_vec());
        let (mut u, mut v) = (identity(n), identity(n));
        let mut left = rotation::Chain::new();
        let mut right = rotation::Chain::new();
        diagonalize(
            &mut d, &mut e, n, &mut u, n, n, &mut v, &mut left, &mut right,
        )?;
        Ok((d, u, v))
    }

    /// A band graded geometrically from one down to `smallest`.
    fn graded(n: usize, smallest: f64, ratio: f64) -> (Vec<f64>, Vec<f64>) {
        let step = smallest.ln() / (n - 1) as f64;
        let d: Vec<f64> = (0..n).map(|i| (step * i as f64).exp()).collect();
        let mut e: Vec<f64> = d.iter().map(|x| x * ratio).collect();
        e[n - 1] = 0.0;
        (d, e)
    }

    /// `U diag(s) V^T == B`, with `U` and `V` orthogonal.
    ///
    /// This is the whole correctness statement and not an approximation of one:
    /// an orthogonal `U` and `V` that reconstruct `B` from a non-negative
    /// diagonal *is* a singular value decomposition, so `s` are the singular
    /// values of `B` to whatever the reconstruction holds to. It needs no
    /// reference implementation to compare against, which matters here because
    /// the interesting bands are exactly the ones where a reference is hard to
    /// come by.
    fn assert_is_a_decomposition(b: &[f64], values: &[f64], u: &[f64], v: &[f64], n: usize) {
        let scale = largest(b);
        let tolerance = 64.0 * f64::EPSILON * scale * (n as f64).sqrt();

        for (i, s) in values.iter().enumerate() {
            assert!(s.is_finite() && *s >= 0.0, "value {i} is {s}");
        }
        let eye = identity(n);
        assert!(
            difference(&multiply(&transposed(u, n), u, n), &eye) < 1e-12,
            "U is not orthogonal"
        );
        assert!(
            difference(&multiply(&transposed(v, n), v, n), &eye) < 1e-12,
            "V is not orthogonal"
        );

        let mut diagonal = vec![0.0; n * n];
        for i in 0..n {
            diagonal[i * n + i] = values[i];
        }
        let rebuilt = multiply(&multiply(u, &diagonal, n), &transposed(v, n), n);
        let residual = difference(&rebuilt, b);
        assert!(
            residual < tolerance,
            "U diag(s) V^T is {residual} away from B, tolerance {tolerance}"
        );
    }

    #[test]
    fn a_graded_band_converges() {
        // The regression, and it is the budget that fixes it: this band needs
        // more than fifty sweeps to deflate its first value, so a cap of fifty
        // per value made the factorisation report a matrix that "may contain
        // NaN or infinity" instead of the answer. Twelve orders of magnitude
        // over 128 values is an ordinary ill-conditioned matrix, and the values
        // wanted from one are the small ones.
        let n = 128;
        let (d, e) = graded(n, 1e-12, 0.3);
        let b = dense(&d, &e, n);
        let (values, u, v) = factor(&d, &e, n).expect("a finite band must converge");
        assert_is_a_decomposition(&b, &values, &u, &v, n);
    }

    #[test]
    fn a_graded_band_keeps_its_small_values() {
        // Reconstruction is an absolute statement, so it would be satisfied by
        // a factorisation that got the small end wrong. The product of the
        // singular values is `|det B|`, which for a bidiagonal band is the
        // product of its diagonal exactly -- a quantity dominated by the
        // *small* values, and one the iteration has no way to fake.
        let n = 96;
        let (d, e) = graded(n, 1e-10, 0.25);
        let (values, _, _) = factor(&d, &e, n).expect("a finite band must converge");

        let logarithm = |v: &[f64]| v.iter().map(|x| x.abs().ln()).sum::<f64>();
        let expected = logarithm(&d);
        let got = logarithm(&values);
        assert!(
            (got - expected).abs() < 1e-9 * expected.abs(),
            "log|det| is {got}, expected {expected}"
        );
    }

    #[test]
    fn the_unshifted_sweep_moves_the_bottom_of_a_graded_band() {
        // What the second spelling is for. A sweep is supposed to shrink the
        // superdiagonal entry above the value it is trying to deflate; on a
        // band graded over twelve orders of magnitude the explicit chase does
        // not move it at all, because the bulge it carries has been multiplied
        // by a sine at every one of 127 steps before it gets there. Thirty
        // sweeps leave that entry bit for bit where it started. The same thirty
        // sweeps of this form take four orders of magnitude off it.
        let n = 128;
        let (d, e) = graded(n, 1e-12, 0.3);

        let after_thirty = |zero_shift: bool| {
            let (mut d, mut e) = (d.clone(), e.clone());
            let (mut left, mut right) = (rotation::Chain::new(), rotation::Chain::new());
            let (mut u, mut v) = (identity(n), identity(n));
            for _ in 0..30 {
                left.start(0, rotation::Order::Rising);
                right.start(0, rotation::Order::Rising);
                if zero_shift {
                    zero_shift_chase(&mut d, &mut e, 0, n - 1, &mut left, &mut right);
                } else {
                    shifted_chase(&mut d, &mut e, 0, n - 1, 0.0, &mut left, &mut right);
                }
                left.apply(&mut u, n, n);
                right.apply(&mut v, n, n);
            }
            e[n - 2].abs()
        };

        let start = e[n - 2].abs();
        let explicit = after_thirty(false);
        let kahan = after_thirty(true);
        assert!(
            explicit > 0.99 * start,
            "the explicit chase moved it after all: {start} -> {explicit}"
        );
        assert!(
            kahan < 1e-4 * start,
            "this form did not converge the bottom: {start} -> {kahan}"
        );
    }

    #[test]
    fn the_two_chases_agree_where_both_work() {
        // The unshifted chase is written a second way because the first way
        // stops working on a graded band, not because it is a different
        // iteration. On a band with nothing small in it -- where the explicit
        // bulge still carries -- the two forms must produce the same band, and
        // they do to a part in `1e15`. They are not required to produce it
        // bit for bit: they are two spellings of the same rotations, and their
        // rounding differs. The superdiagonal comes out with opposite signs,
        // which no singular value can see and the next sweep does not care
        // about either.
        let n = 128;
        let d: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (i as f64 * 0.7).sin()).collect();
        let mut e: Vec<f64> = (0..n).map(|i| 0.3 + 0.2 * (i as f64 * 1.1).cos()).collect();
        e[n - 1] = 0.0;

        let run = |zero_shift: bool| {
            let (mut d, mut e) = (d.clone(), e.clone());
            let (mut left, mut right) = (rotation::Chain::new(), rotation::Chain::new());
            let (mut u, mut v) = (identity(n), identity(n));
            left.start(0, rotation::Order::Rising);
            right.start(0, rotation::Order::Rising);
            if zero_shift {
                zero_shift_chase(&mut d, &mut e, 0, n - 1, &mut left, &mut right);
            } else {
                shifted_chase(&mut d, &mut e, 0, n - 1, 0.0, &mut left, &mut right);
            }
            left.apply(&mut u, n, n);
            right.apply(&mut v, n, n);
            (d, e, u, v)
        };

        let b = dense(&d, &e, n);
        let (explicit_d, explicit_e, explicit_u, explicit_v) = run(false);
        let (kahan_d, kahan_e, kahan_u, kahan_v) = run(true);

        let tolerance = 1e-13 * largest(&b);
        assert!(
            difference(&explicit_d, &kahan_d) < tolerance,
            "diagonals differ by {}",
            difference(&explicit_d, &kahan_d)
        );
        let magnitude = |v: &[f64]| v.iter().map(|x| x.abs()).collect::<Vec<_>>();
        assert!(
            difference(&magnitude(&explicit_e), &magnitude(&kahan_e)) < tolerance,
            "superdiagonals differ in magnitude"
        );

        // And each on its own is exactly what a sweep is supposed to be: one
        // orthogonal change of basis carrying `B` to the band it reports. That
        // part *is* held to rounding, and it is what makes either form usable.
        for (u, v, d, e) in [
            (&explicit_u, &explicit_v, &explicit_d, &explicit_e),
            (&kahan_u, &kahan_v, &kahan_d, &kahan_e),
        ] {
            let swept = multiply(&multiply(&transposed(u, n), &b, n), v, n);
            let residual = difference(&swept, &dense(d, e, n));
            assert!(
                residual < 1e-14 * largest(&b),
                "U^T B V is {residual} away from the swept band"
            );
        }
    }
}
