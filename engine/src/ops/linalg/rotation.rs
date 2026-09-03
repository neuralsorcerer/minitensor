// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The orthogonal transformations the two converging factorisations are made
//! of.
//!
//! `eigh` chases a tridiagonal matrix's off-diagonal to zero and `svd` chases a
//! bidiagonal one's superdiagonal, and both do it with plane rotations: a
//! two-by-two turn that leaves every other coordinate alone. Both then finish by
//! permuting the result into order, carrying the vectors along -- which is the
//! same kind of transformation, an orthogonal one applied to both sides, and is
//! why it lives here too.
//!
//! There is one sign convention in this file and everything obeys it.
//! [`plane`] returns the `(c, s)` of
//!
//! ```text
//! G = [ c  s ]      with   [f  g] G = [r  0]
//!     [-s  c ]
//! ```
//!
//! and [`Chain`] applies that same `G` from the right. Getting the sign of
//! `s` wrong in one of the two produces a rotation by the correct angle in the
//! wrong direction, which is a factorisation that never converges rather than
//! one that is visibly wrong, so the two are defined together.

use crate::{
    error::{MinitensorError, Result},
    ops::map::PAR_THRESHOLD,
};
use num_traits::Float;
use rayon::prelude::*;

/// `sqrt(a^2 + b^2)` without the overflow the obvious spelling has.
///
/// Squaring is what overflows, and it overflows for arguments a long way inside
/// the representable range -- around `1e19` for `f32`, which a covariance
/// entry can reach without anything being wrong. Factoring the larger argument
/// out first means the only square formed is of a ratio no greater than one.
pub(crate) fn hypotenuse<T: Float>(a: T, b: T) -> T {
    let (a, b) = (a.abs(), b.abs());
    if a > b {
        let ratio = b / a;
        a * (T::one() + ratio * ratio).sqrt()
    } else if b > T::zero() {
        let ratio = a / b;
        b * (T::one() + ratio * ratio).sqrt()
    } else {
        T::zero()
    }
}

/// The rotation that sends `(f, g)` to `(r, 0)`, as `(c, s, r)`.
///
/// `r` is non-negative whenever a rotation was needed. When `g` is already zero
/// the identity is returned along with `f` unchanged, sign and all -- there is
/// nothing to rotate, and manufacturing a sign flip here would be a change to
/// the factorisation that no later step accounts for.
pub(crate) fn plane<T: Float>(f: T, g: T) -> (T, T, T) {
    if g == T::zero() {
        return (T::one(), T::zero(), f);
    }
    let r = hypotenuse(f, g);
    (f / r, -g / r, r)
}

/// Which columns rotation `k` of a chain turns.
///
/// Storing the shape rather than each rotation's own pair of indices is not
/// bookkeeping for its own sake -- it is the entire performance of the replay.
/// A pair read out of memory per rotation cannot be proved in bounds, so every
/// access is checked and nothing vectorises; worse, it hides that consecutive
/// rotations *share a column*, which is what lets the replay carry one value in
/// a register instead of storing it and loading it straight back. The first
/// version of this file stored the pairs and ran the whole factorisation twice
/// as slow as applying the rotations one at a time had been.
#[derive(Clone, Copy)]
pub(crate) enum Order {
    /// Rotation `k` turns `(first + k, first + k + 1)`: a band walked upward.
    Rising,
    /// Rotation `k` turns `(first - k, first - k + 1)`: the same band downward.
    Falling,
    /// Rotation `k` turns `(pivot, first + k)`: a fan out of one fixed column.
    Fanned(usize),
}

/// A chain of plane rotations, to be applied to one matrix in a single pass.
///
/// Each rotation is `M <- M G` for the `G` that [`plane`] describes, over one
/// pair of columns. A factorisation that rotates its band by `G` on the right
/// rotates the matrix accumulating its right factor the same way, and one that
/// rotates its band by `G^T` on the left rotates the left factor by `G` -- so
/// both sides of both routines come through here, and only the operand differs.
///
/// They are collected rather than applied as they are produced because a sweep
/// makes as many rotations as the band is wide, and applying each one on its own
/// is a separate pass over a matrix that left cache several rotations ago -- the
/// two columns a rotation touches are a row length apart, so each of them costs
/// a cache line per element.
///
/// That argument is true and it is not, on its own, worth anything, which is
/// the part worth writing down. Applying rotations one at a time has an inner
/// loop over *rows*, which are independent and vectorise; replaying a chain has
/// an inner loop over *rotations*, which are a dependency chain and do not.
/// Trading the one for the other bought `svd` at `n = 400` a move from 257ms to
/// 344ms -- a fifth slower, for a strictly better memory access pattern.
/// It only pays once [`Chain::replay`] puts independent work back into the
/// inner loop, and then it pays properly: 148ms, against 257ms for one rotation
/// at a time and 284ms for the same single-pass chain spread over four cores.
pub(crate) struct Chain<T> {
    first: usize,
    order: Order,
    turns: Vec<(T, T)>,
}

impl<T: Float + Send + Sync> Chain<T> {
    pub(crate) fn new() -> Self {
        Self {
            first: 0,
            order: Order::Rising,
            turns: Vec::new(),
        }
    }

    /// Begin a chain whose first rotation turns the columns `first` names.
    pub(crate) fn start(&mut self, first: usize, order: Order) {
        self.first = first;
        self.order = order;
        self.turns.clear();
    }

    /// Add the next rotation. Its columns follow from the [`Order`], so an
    /// identity is still pushed -- dropping it would shift every rotation after
    /// it onto the wrong pair.
    pub(crate) fn push(&mut self, c: T, s: T) {
        self.turns.push((c, s));
    }

    /// Replay the chain against every row of a row-major matrix.
    ///
    /// Sweeps are strictly ordered and rows are not: a rotation mixes columns
    /// and never reaches across rows, so this is the one place in either
    /// factorisation where there is parallelism to take without an argument
    /// about ordering. It is worth much less than it looks -- see
    /// [`Chain::replay`] for why -- but it composes with what is.
    pub(crate) fn apply(&self, matrix: &mut [T], rows: usize, stride: usize) {
        if self.turns.is_empty() || rows == 0 {
            return;
        }
        let matrix = &mut matrix[..rows * stride];
        if rows * self.turns.len() < PAR_THRESHOLD {
            self.replay(matrix, stride);
            return;
        }
        let per_task = rows.div_ceil(rayon::current_num_threads().max(1)).max(1);
        matrix
            .par_chunks_mut(per_task * stride)
            .for_each(|block| self.replay(block, stride));
    }

    /// The chain against one contiguous band of rows, four rows at a time.
    ///
    /// One row's replay is a dependency chain and nothing else: rotation `k`
    /// writes the column rotation `k + 1` reads, so every step waits a
    /// multiply-add -- about four cycles -- on the step before it, whatever the
    /// machine's throughput is. That is the whole cost. Measured on a 400x400
    /// matrix, the sweeps were 298ms of `svd`'s 344, and spreading the same rows
    /// across four cores bought only 1.2x, because four cores each waiting on a
    /// latency are still waiting on a latency.
    ///
    /// Rows do not depend on each other at all, so replaying four at once hands
    /// the machine four independent chains to interleave in the units that were
    /// idle. Identical arithmetic, identical order, 1.9x -- more than the
    /// parallelism was worth, and it composes with it.
    fn replay(&self, block: &mut [T], stride: usize) {
        let mut rest = block;
        while rest.len() >= LANES * stride {
            let (group, tail) = rest.split_at_mut(LANES * stride);
            self.replay_lanes::<LANES>(group, stride);
            rest = tail;
        }
        while rest.len() >= stride {
            let (group, tail) = rest.split_at_mut(stride);
            self.replay_lanes::<1>(group, stride);
            rest = tail;
        }
    }

    /// `L` rows of `group` at once, `L` known at compile time so the lane loop
    /// unrolls into `L` independent accumulators rather than a counted loop.
    fn replay_lanes<const L: usize>(&self, group: &mut [T], stride: usize) {
        let len = self.turns.len();
        // One base offset per row, hoisted so the inner loop adds a constant.
        let mut base = [0usize; L];
        for (r, slot) in base.iter_mut().enumerate() {
            *slot = r * stride;
        }
        let mut carry = [T::zero(); L];

        match self.order {
            Order::Rising => {
                let start = self.first;
                for (r, slot) in carry.iter_mut().enumerate() {
                    *slot = group[base[r] + start];
                }
                for (k, &(c, s)) in self.turns.iter().enumerate() {
                    for (r, held) in carry.iter_mut().enumerate() {
                        let index = base[r] + start + k;
                        let b = group[index + 1];
                        group[index] = c * *held - s * b;
                        *held = s * *held + c * b;
                    }
                }
                for (r, &value) in carry.iter().enumerate() {
                    group[base[r] + start + len] = value;
                }
            }
            Order::Falling => {
                let start = self.first + 1 - len;
                for (r, slot) in carry.iter_mut().enumerate() {
                    *slot = group[base[r] + start + len];
                }
                for (k, &(c, s)) in self.turns.iter().enumerate() {
                    let j = len - 1 - k;
                    for (r, held) in carry.iter_mut().enumerate() {
                        let index = base[r] + start + j;
                        let a = group[index];
                        group[index + 1] = s * a + c * *held;
                        *held = c * a - s * *held;
                    }
                }
                for (r, &value) in carry.iter().enumerate() {
                    group[base[r] + start] = value;
                }
            }
            Order::Fanned(pivot) => {
                for (r, slot) in carry.iter_mut().enumerate() {
                    *slot = group[base[r] + pivot];
                }
                for (k, &(c, s)) in self.turns.iter().enumerate() {
                    for (r, held) in carry.iter_mut().enumerate() {
                        let index = base[r] + self.first + k;
                        let b = group[index];
                        group[index] = s * *held + c * b;
                        *held = c * *held - s * b;
                    }
                }
                for (r, &value) in carry.iter().enumerate() {
                    group[base[r] + pivot] = value;
                }
            }
        }
    }
}

/// How many rows the replay interleaves.
///
/// Four independent chains against a multiply-add latency of about four cycles,
/// which is where the units stop waiting on themselves. More lanes would need
/// more of the register file for carries with nothing left to hide.
const LANES: usize = 4;

/// Negate column `p`, which is the one-dimensional rotation.
///
/// `svd` needs it because a singular value is defined non-negative and the
/// iteration does not promise one: flipping the sign of the value and of the
/// matching column of `V` leaves `U diag(s) V^T` exactly unchanged, since both
/// flips multiply the same rank-one term.
pub(crate) fn negate_column<T: Float>(matrix: &mut [T], rows: usize, stride: usize, p: usize) {
    for row in 0..rows {
        matrix[row * stride + p] = -matrix[row * stride + p];
    }
}

/// Reorder `keys`, carrying the columns of every matrix in `carried` with them.
///
/// Both factorisations produce their values in whatever order the iteration
/// converged, and both have a conventional order callers rely on -- ascending
/// for eigenvalues, matching LAPACK's `syev` and NumPy's `eigh`; descending for
/// singular values, matching `gesdd` and NumPy's `svd`. The permutation is
/// orthogonal, so applying it to the values and to the matching columns of the
/// vectors leaves the factorisation exactly intact.
///
/// Selection sort rather than anything cleverer: the key count is the matrix
/// order, so the comparisons are `n^2 / 2` against the `n^3` that produced the
/// keys, and moving a pair of columns is the expensive half either way.
///
/// `carried` is a slice so a caller that does not want its vectors can pass an
/// empty one and skip the column traffic entirely -- which is the whole saving
/// `eigvalsh` exists for.
pub(crate) fn sort_carrying_columns<T: Copy + PartialOrd>(
    keys: &mut [T],
    descending: bool,
    carried: &mut [(&mut [T], usize)],
) {
    for i in 0..keys.len() {
        let mut best = i;
        for j in (i + 1)..keys.len() {
            let better = if descending {
                keys[j] > keys[best]
            } else {
                keys[j] < keys[best]
            };
            if better {
                best = j;
            }
        }
        if best == i {
            continue;
        }
        keys.swap(i, best);
        for (data, stride) in carried.iter_mut() {
            for row in 0..data.len() / *stride {
                data.swap(row * *stride + i, row * *stride + best);
            }
        }
    }
}

/// How many sweeps each value in the band is allowed to cost, on average.
///
/// The budget is a total for the whole band rather than a cap on any one value,
/// because the work is not spread evenly across the values and a per-value cap
/// has to be set for the worst one. A band graded over many orders of magnitude
/// spends most of its sweeps deflating the *first* value and then takes two or
/// three for each of the rest: on a 400-wide band the first cost 157 sweeps of
/// a total of 205. A cap that lets the first value have what it needs is far
/// looser than it has to be for everything after it, while a total is tight
/// against the thing that actually matters, which is how much work the whole
/// factorisation is allowed to do.
///
/// Thirty is a shade over ten times the worst total measured across graded,
/// clustered, repeated and random bands from 32 to 400 wide, all of which
/// finished inside `3n`. It also bounds the pathological case: a band that
/// never converges costs `30n` sweeps of `O(n^2)` rotation replay, which is ten
/// times the reduction that produced it rather than the `6n^2` sweeps -- `n`
/// times more again -- that LAPACK allows itself.
const SWEEP_BUDGET_PER_VALUE: usize = 30;

/// Stop a band that will not converge, rather than letting it spin.
///
/// The guard exists for a matrix carrying a NaN, where every comparison is
/// false and no off-diagonal ever looks negligible. A finite band converges:
/// cubically once the shift is close, and linearly at a rate set by the ratio
/// of neighbouring values when there is no usable shift. That rate is what ties
/// the budget to `order` -- the linear phase needs a number of sweeps that
/// grows with how finely the band is graded, which is to say with how many
/// values it holds, so a constant cap is a limit on matrix size wearing a
/// convergence failure's error message.
pub(crate) fn check_sweeps(count: usize, order: usize, op: &str) -> Result<()> {
    if count > SWEEP_BUDGET_PER_VALUE * order.max(1) {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} did not converge; the matrix may contain NaN or infinity"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rotation applied the obvious way: `M <- M G` over two columns.
    ///
    /// This is the definition the whole file is an optimisation of, so every
    /// test below builds its expected answer out of it rather than out of
    /// another arrangement of the same clever indexing.
    fn turn(matrix: &mut [f64], rows: usize, stride: usize, p: usize, q: usize, c: f64, s: f64) {
        for row in 0..rows {
            let base = row * stride;
            let (a, b) = (matrix[base + p], matrix[base + q]);
            matrix[base + p] = c * a - s * b;
            matrix[base + q] = s * a + c * b;
        }
    }

    fn filled(rows: usize, stride: usize) -> Vec<f64> {
        (0..rows * stride).map(|i| (i as f64) * 0.5 - 3.0).collect()
    }

    /// Angles that are not special cases of each other.
    fn turns(count: usize) -> Vec<(f64, f64)> {
        (0..count)
            .map(|k| {
                let angle = 0.3 + 0.4 * k as f64;
                (angle.cos(), angle.sin())
            })
            .collect()
    }

    fn assert_close(got: &[f64], expected: &[f64]) {
        for (a, b) in got.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn plane_sends_the_pair_onto_its_first_axis() {
        for &(f, g) in &[
            (3.0, 4.0),
            (-3.0, 4.0),
            (3.0, -4.0),
            (1e-8, 2.0),
            (2.0, 1e-8),
        ] {
            let (c, s, r) = plane(f, g);
            assert!((c * f - s * g - r).abs() < 1e-12);
            assert!((s * f + c * g).abs() < 1e-12);
            assert!((c * c + s * s - 1.0).abs() < 1e-14);
            assert!(r > 0.0);
        }
    }

    #[test]
    fn plane_leaves_an_already_reduced_pair_alone() {
        // Including its sign: inventing a flip here would be a change to the
        // factorisation that no later step accounts for.
        assert_eq!(plane(-5.0, 0.0), (1.0, 0.0, -5.0));
    }

    #[test]
    fn hypotenuse_survives_arguments_whose_squares_do_not() {
        assert!((hypotenuse(3e200, 4e200) - 5e200).abs() < 1e188);
        assert!((hypotenuse(3e-200, 4e-200) - 5e-200).abs() < 1e-212);
        assert_eq!(hypotenuse(0.0, 0.0), 0.0);
    }

    #[test]
    fn rising_chain_matches_one_rotation_at_a_time() {
        let (rows, stride, first, count) = (7, 9, 2, 5);
        let angles = turns(count);
        let mut expected = filled(rows, stride);
        for (k, &(c, s)) in angles.iter().enumerate() {
            turn(&mut expected, rows, stride, first + k, first + k + 1, c, s);
        }

        let mut chain = Chain::new();
        chain.start(first, Order::Rising);
        for &(c, s) in &angles {
            chain.push(c, s);
        }
        let mut got = filled(rows, stride);
        chain.apply(&mut got, rows, stride);
        assert_close(&got, &expected);
    }

    #[test]
    fn falling_chain_matches_one_rotation_at_a_time() {
        let (rows, stride, first, count) = (7, 9, 6, 5);
        let angles = turns(count);
        let mut expected = filled(rows, stride);
        for (k, &(c, s)) in angles.iter().enumerate() {
            turn(&mut expected, rows, stride, first - k, first - k + 1, c, s);
        }

        let mut chain = Chain::new();
        chain.start(first, Order::Falling);
        for &(c, s) in &angles {
            chain.push(c, s);
        }
        let mut got = filled(rows, stride);
        chain.apply(&mut got, rows, stride);
        assert_close(&got, &expected);
    }

    #[test]
    fn fanned_chain_matches_one_rotation_at_a_time() {
        let (rows, stride, pivot, first, count) = (7, 9, 1, 2, 5);
        let angles = turns(count);
        let mut expected = filled(rows, stride);
        for (k, &(c, s)) in angles.iter().enumerate() {
            turn(&mut expected, rows, stride, pivot, first + k, c, s);
        }

        let mut chain = Chain::new();
        chain.start(first, Order::Fanned(pivot));
        for &(c, s) in &angles {
            chain.push(c, s);
        }
        let mut got = filled(rows, stride);
        chain.apply(&mut got, rows, stride);
        assert_close(&got, &expected);
    }

    #[test]
    fn a_chain_preserves_length() {
        // Every rotation is orthogonal, so replaying a chain cannot change the
        // norm of any row -- the property the factorisations rely on.
        let (rows, stride) = (5, 8);
        let mut chain = Chain::new();
        chain.start(1, Order::Rising);
        for &(c, s) in &turns(6) {
            chain.push(c, s);
        }
        let before = filled(rows, stride);
        let mut after = before.clone();
        chain.apply(&mut after, rows, stride);
        for row in 0..rows {
            let norm = |v: &[f64]| {
                v[row * stride..(row + 1) * stride]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
            };
            assert!((norm(&before) - norm(&after)).abs() < 1e-10);
        }
    }

    #[test]
    fn the_parallel_and_serial_replays_agree() {
        // Wide enough to cross the threshold that splits the rows across
        // threads, so the two paths are actually different code.
        let (rows, stride) = (600, 600);
        let mut chain = Chain::new();
        chain.start(0, Order::Rising);
        for &(c, s) in &turns(400) {
            chain.push(c, s);
        }
        let mut parallel = filled(rows, stride);
        chain.apply(&mut parallel, rows, stride);

        let mut serial = filled(rows, stride);
        chain.replay(&mut serial, stride);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn an_empty_chain_changes_nothing() {
        let mut chain: Chain<f64> = Chain::new();
        chain.start(3, Order::Rising);
        let before = filled(4, 6);
        let mut after = before.clone();
        chain.apply(&mut after, 4, 6);
        assert_eq!(before, after);
    }

    #[test]
    fn sorting_carries_the_columns_it_is_given() {
        let mut keys = vec![2.0, 5.0, 1.0, 4.0];
        let mut left = vec![
            0.0, 1.0, 2.0, 3.0, //
            10.0, 11.0, 12.0, 13.0,
        ];
        let mut right = vec![100.0, 101.0, 102.0, 103.0];
        {
            let mut carried = [(&mut left[..], 4), (&mut right[..], 4)];
            sort_carrying_columns(&mut keys, true, &mut carried);
        }
        assert_eq!(keys, vec![5.0, 4.0, 2.0, 1.0]);
        assert_eq!(left, vec![1.0, 3.0, 0.0, 2.0, 11.0, 13.0, 10.0, 12.0]);
        assert_eq!(right, vec![101.0, 103.0, 100.0, 102.0]);
    }

    #[test]
    fn sorting_ascending_is_the_other_direction() {
        let mut keys = vec![2.0, 5.0, 1.0, 4.0];
        sort_carrying_columns(&mut keys, false, &mut []);
        assert_eq!(keys, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn negating_a_column_touches_only_that_column() {
        let mut matrix = filled(3, 4);
        let before = matrix.clone();
        negate_column(&mut matrix, 3, 4, 2);
        for row in 0..3 {
            for col in 0..4 {
                let (got, want) = (matrix[row * 4 + col], before[row * 4 + col]);
                assert_eq!(got, if col == 2 { -want } else { want });
            }
        }
    }

    #[test]
    fn the_sweep_budget_reports_rather_than_spinning() {
        assert!(check_sweeps(300, 10, "svd").is_ok());
        let message = check_sweeps(301, 10, "svd").unwrap_err().to_string();
        assert!(message.contains("svd") && message.contains("converge"));
    }

    #[test]
    fn the_sweep_budget_grows_with_the_band() {
        // The point of the change: a band twice as wide is allowed twice the
        // work, so the guard is not a limit on matrix size.
        assert!(check_sweeps(600, 20, "svd").is_ok());
        assert!(check_sweeps(600, 10, "svd").is_err());
        // And a one-value band still gets a budget rather than zero.
        assert!(check_sweeps(1, 0, "eigh").is_ok());
    }
}
