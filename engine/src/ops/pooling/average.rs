// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Average pools, each mean and each gradient rounded once.
//!
//! A window's mean is the exact sum of what it reads over what it divides
//! by, rounded once. Added in the element type and then divided, it took a
//! rounding for each value past the first and one more. Here the windows of
//! a plane are summed side by side, a tap of every window at a time, each
//! in a two-sum that keeps what its additions round away, and each mean is
//! divided from that pair where its bound places the quotient; a window it
//! does not place is read whole and summed exactly.
//!
//! The gradient is the same sum turned round: a cell of the input takes,
//! from each window that read it, that window's gradient over its divisor,
//! and the total of those shares is rounded once. Where every window
//! divides by the same count the shares' numerators are summed and divided
//! once, as a mean is. Where the counts differ -- padding left out of the
//! divisor, or windows sized by the ratio of two extents -- each share is
//! taken as its quotient and the exact remainder's quotient, a pair that
//! misses the share by less than the pair's last place, and the pairs are
//! summed; a cell that leaves open is one exact fraction over the product
//! of its windows' divisors.

use super::adaptive_window;
use crate::ops::arithmetic::exact_sum::{
    Element, Output, Tallies, mean_to, sum_of_quotients, sum_over,
};
use crate::ops::conv::in_bounds_range;
use crate::ops::map::par_out_chunks;
use std::ops::Range;

/// At least how many elements the planes of one task come to: enough to be
/// worth a task, which makes its scratch once for all of them.
const BAND: usize = 4096;

/// The divisors a share can be divided by in float64 and narrowed without a
/// second rounding showing: under `2^24`, a float32's quotient by one is
/// never within a float64's rounding of halfway between two float32s
/// unless it is exactly there.
const DIRECT: usize = 1 << 24;

/// The planes an average pool reads and the planes it writes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extents {
    /// How many planes: the batch times the channels.
    pub(crate) planes: usize,
    pub(crate) in_h: usize,
    pub(crate) in_w: usize,
    pub(crate) out_h: usize,
    pub(crate) out_w: usize,
}

impl Extents {
    /// The fixed window that pools these extents, where the output's divide
    /// the input's: the windows of an adaptive pool are then all of one
    /// size and none overlaps another.
    fn even_window(&self) -> Option<FixedWindow> {
        let even = self.out_h > 0
            && self.out_w > 0
            && self.in_h.is_multiple_of(self.out_h)
            && self.in_w.is_multiple_of(self.out_w);
        let size = (self.in_h / self.out_h.max(1), self.in_w / self.out_w.max(1));
        (even && size.0 > 0 && size.1 > 0).then_some(FixedWindow {
            kernel: size,
            stride: size,
            padding: (0, 0),
            count_include_pad: true,
        })
    }
}

/// A window of one size, stepped across a padded plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedWindow {
    pub(crate) kernel: (usize, usize),
    pub(crate) stride: (usize, usize),
    pub(crate) padding: (usize, usize),
    /// Whether the padding a window lies over counts towards its divisor.
    pub(crate) count_include_pad: bool,
}

/// Where the windows of a plane lie, and what each divides by.
struct Layout {
    extents: Extents,
    /// The input rows that each row of windows reads, and the input columns
    /// that each column of them does.
    rows: Vec<Range<usize>>,
    cols: Vec<Range<usize>>,
    /// What every window divides by, where that is not the cells it reads.
    fixed: Option<usize>,
}

impl Layout {
    fn fixed(extents: Extents, window: &FixedWindow) -> Self {
        // A window's taps from `index * stride - padding`, kept to the
        // cells there are.
        let reach = |out: usize, dim: usize, kernel: usize, stride: usize, padding: usize| {
            (0..out)
                .map(|index| {
                    let first = index * stride;
                    let end = (first + kernel).saturating_sub(padding).min(dim);
                    first.saturating_sub(padding).min(end)..end
                })
                .collect()
        };
        let FixedWindow {
            kernel,
            stride,
            padding,
            count_include_pad,
        } = *window;
        Self {
            extents,
            rows: reach(extents.out_h, extents.in_h, kernel.0, stride.0, padding.0),
            cols: reach(extents.out_w, extents.in_w, kernel.1, stride.1, padding.1),
            fixed: count_include_pad.then_some(kernel.0 * kernel.1),
        }
    }

    fn adaptive(extents: Extents) -> Self {
        let reach = |out: usize, dim: usize| {
            (0..out)
                .map(|index| adaptive_window(index, dim, out))
                .collect()
        };
        Self {
            extents,
            rows: reach(extents.out_h, extents.in_h),
            cols: reach(extents.out_w, extents.in_w),
            fixed: None,
        }
    }

    /// What the window at `(oh, ow)` divides its sum by.
    fn divisor(&self, oh: usize, ow: usize) -> usize {
        self.fixed
            .unwrap_or(self.rows[oh].len() * self.cols[ow].len())
    }

    /// The one count every window divides by, where there is one.
    fn shared_divisor(&self) -> Option<usize> {
        if self.fixed.is_some() {
            return self.fixed;
        }
        let even =
            |ranges: &[Range<usize>]| ranges.iter().all(|range| range.len() == ranges[0].len());
        (even(&self.rows) && even(&self.cols)).then(|| self.divisor(0, 0))
    }

    /// Each window that reads cell `slot` of an input plane: its place in
    /// the output plane, and what it divides by.
    fn readers(&self, slot: usize, mut each: impl FnMut(usize, usize)) {
        let (ih, iw) = (slot / self.extents.in_w, slot % self.extents.in_w);
        for (oh, rows) in self.rows.iter().enumerate() {
            if !rows.contains(&ih) {
                continue;
            }
            for (ow, cols) in self.cols.iter().enumerate() {
                if cols.contains(&iw) {
                    each(oh * self.extents.out_w + ow, self.divisor(oh, ow));
                }
            }
        }
    }

    /// The mean of the window at `slot` of an output plane, read whole from
    /// `plane`: where it lies in one piece and divides by its own size it
    /// is the mean of that piece, and otherwise its values are brought
    /// together, with a zero for each cell of padding its divisor counts.
    fn mean_of<T: Element + Output + Default>(
        &self,
        plane: &[T],
        slot: usize,
        window: &mut Vec<T>,
    ) -> T {
        let Extents { in_w, out_w, .. } = self.extents;
        let (oh, ow) = (slot / out_w, slot % out_w);
        let (rows, cols) = (&self.rows[oh], &self.cols[ow]);
        let inside = rows.len() * cols.len();
        if inside == 0 {
            return T::default();
        }
        let divisor = self.divisor(oh, ow);
        if divisor == inside && (cols.len() == in_w || rows.len() == 1) {
            return mean_to(&plane[rows.start * in_w + cols.start..][..inside]);
        }
        window.clear();
        for ih in rows.clone() {
            window.extend_from_slice(&plane[ih * in_w..][cols.clone()]);
        }
        window.resize(divisor, T::default());
        mean_to(window)
    }

    /// The mean of every window, each read whole.
    fn means<T: Element + Output + Default>(&self, data: &[T]) -> Vec<T> {
        let Extents {
            planes,
            in_h,
            in_w,
            out_h,
            out_w,
        } = self.extents;
        let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
        let mut means = vec![T::default(); planes * plane_out];
        if plane_in == 0 || plane_out == 0 {
            return means;
        }
        let pooled = |plane: &[T], means: &mut [T], window: &mut Vec<T>| {
            for (slot, mean) in means.iter_mut().enumerate() {
                *mean = self.mean_of(plane, slot, window);
            }
        };
        for_each_plane(&mut means, plane_out, data, plane_in, &Vec::new, &pooled);
        means
    }

    /// The gradient where the windows may divide by counts of their own.
    ///
    /// A share `g / d` is taken as `q + r`: `q` the quotient rounded, and
    /// `r` the quotient of what `q` leaves of `g`, which a fused multiply
    /// finds exactly. The pair misses the share by the rounding of `r`
    /// alone, far below the last place of `q`, and that is counted with
    /// what the two-sum the pairs are added into has dropped.
    fn shares_of_quotients<T: Element + Output + Default>(&self, flowing: &[T]) -> Vec<T> {
        let Extents {
            planes,
            in_h,
            in_w,
            out_h,
            out_w,
        } = self.extents;
        let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
        let mut gradient = vec![T::default(); planes * plane_in];
        if plane_in == 0 || plane_out == 0 {
            return gradient;
        }
        type Scratch = (Tallies, Vec<usize>, Vec<(f64, u64)>);
        let scratch = || (Tallies::new(plane_in), Vec::new(), Vec::new());
        let spread = |flowing: &[T], cells: &mut [T], (tallies, open, terms): &mut Scratch| {
            tallies.clear(0.0);
            for (window, &share) in flowing.iter().enumerate() {
                let (rows, cols) = (&self.rows[window / out_w], &self.cols[window % out_w]);
                let share = share.wide();
                if share == 0.0 || rows.is_empty() || cols.is_empty() {
                    continue;
                }
                let divisor = self.divisor(window / out_w, window % out_w) as f64;
                let quotient = share / divisor;
                let remainder = (-quotient).mul_add(divisor, share);
                let rest = remainder / divisor;
                // The rounding of `rest`: half its last place, or half the
                // least float64 where it is that small; none where the
                // quotient is the share.
                let rounding = if remainder == 0.0 {
                    0.0
                } else {
                    rest.abs() * 2f64.powi(-52) + f64::from_bits(1)
                };
                for ih in rows.clone() {
                    tallies.add_pair(
                        ih * in_w + cols.start,
                        cols.len(),
                        (quotient, rest),
                        rounding,
                    );
                }
            }
            open.clear();
            tallies.totals_into(0, cells, open);
            for &slot in open.iter() {
                terms.clear();
                self.readers(slot, |window, divisor| {
                    terms.push((flowing[window].wide(), divisor as u64));
                });
                cells[slot] = sum_of_quotients(terms);
            }
        };
        for_each_plane(
            &mut gradient,
            plane_in,
            flowing,
            plane_out,
            &scratch,
            &spread,
        );
        gradient
    }

    /// The gradient where no cell is read twice: each takes the one share
    /// there is for it, a quotient rounded once.
    ///
    /// For float32 the quotient is taken in float64 and narrowed, which is
    /// a second rounding only in name: a float32 over a whole number under
    /// [`DIRECT`] is either a halfway point between two float32s, which
    /// float64 holds, or further from one than float64 rounds.
    fn shares_apart<T: Element + Output + Default>(&self, flowing: &[T]) -> Vec<T> {
        let Extents {
            planes,
            in_h,
            in_w,
            out_h,
            out_w,
        } = self.extents;
        let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
        let mut gradient = vec![T::default(); planes * plane_in];
        if plane_in == 0 || plane_out == 0 {
            return gradient;
        }
        let share_out = |flowing: &[T], cells: &mut [T], _: &mut ()| {
            for (slot, &share) in flowing.iter().enumerate() {
                let (oh, ow) = (slot / out_w, slot % out_w);
                let divisor = self.divisor(oh, ow);
                if share.wide() == 0.0 || divisor == 0 {
                    continue;
                }
                let share = T::narrow(share.wide() / divisor as f64);
                for ih in self.rows[oh].clone() {
                    cells[ih * in_w..][self.cols[ow].clone()].fill(share);
                }
            }
        };
        for_each_plane(
            &mut gradient,
            plane_in,
            flowing,
            plane_out,
            &|| (),
            &share_out,
        );
        gradient
    }

    /// The gradient where every window of `window` divides by `divisor`:
    /// the numerators are summed side by side, a tap of every window at a
    /// time, and each cell's sum divided once.
    fn shares_over<T: Element + Output + Default>(
        &self,
        flowing: &[T],
        window: &FixedWindow,
        divisor: usize,
    ) -> Vec<T> {
        let Extents {
            planes,
            in_h,
            in_w,
            out_h,
            out_w,
        } = self.extents;
        let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
        let mut gradient = vec![T::default(); planes * plane_in];
        if plane_in == 0 || plane_out == 0 || divisor == 0 {
            return gradient;
        }
        let step = window.stride.1;
        let counts = vec![divisor as f64; plane_in];
        type Scratch<T> = (Tallies, Vec<usize>, Vec<T>);
        let scratch = || (Tallies::new(plane_in), Vec::new(), Vec::new());
        let spread = |flowing: &[T], cells: &mut [T], (tallies, open, kept): &mut Scratch<T>| {
            tallies.clear(0.0);
            for_each_tap_row(self.extents, window, |place, windows| {
                if step == 1 {
                    tallies.add(place, &flowing[windows]);
                } else {
                    tallies.add_spread(place, step, &flowing[windows]);
                }
            });
            open.clear();
            tallies.means_into(0, &counts, cells, open);
            for &slot in open.iter() {
                kept.clear();
                self.readers(slot, |window, _| kept.push(flowing[window]));
                cells[slot] = sum_over(kept, divisor as u64);
            }
        };
        for_each_plane(
            &mut gradient,
            plane_in,
            flowing,
            plane_out,
            &scratch,
            &spread,
        );
        gradient
    }

    /// The mean of every window of `window`, the windows summed side by
    /// side, a tap of every window at a time; a window whose mean the bound
    /// leaves open is read whole.
    fn summed_side_by_side<T: Element + Output + Default>(
        &self,
        data: &[T],
        window: &FixedWindow,
    ) -> Vec<T> {
        let Extents {
            planes,
            in_h,
            in_w,
            out_h,
            out_w,
        } = self.extents;
        let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
        let mut means = vec![T::default(); planes * plane_out];
        if plane_in == 0 || plane_out == 0 {
            return means;
        }
        let step = window.stride.1;
        let counts: Vec<f64> = (0..plane_out)
            .map(|slot| self.divisor(slot / out_w, slot % out_w) as f64)
            .collect();
        type Scratch<T> = (Tallies, Vec<usize>, Vec<T>);
        let scratch = || (Tallies::new(plane_out), Vec::new(), Vec::new());
        let pool = |plane: &[T], means: &mut [T], (tallies, open, kept): &mut Scratch<T>| {
            tallies.clear(0.0);
            for_each_tap_row(self.extents, window, |place, windows| {
                if step == 1 {
                    tallies.add(windows.start, &plane[place..place + windows.len()]);
                } else {
                    tallies.add_spaced(windows.start, &plane[place..], step, windows.len());
                }
            });
            open.clear();
            tallies.means_into(0, &counts, means, open);
            for &slot in open.iter() {
                means[slot] = self.mean_of(plane, slot, kept);
            }
        };
        for_each_plane(&mut means, plane_out, data, plane_in, &scratch, &pool);
        means
    }
}

/// `work` for each plane of `out` beside its plane of `data`, with scratch
/// that a task makes once for the planes it takes.
fn for_each_plane<T, S, Make, Work>(
    out: &mut [T],
    out_plane: usize,
    data: &[T],
    data_plane: usize,
    scratch: &Make,
    work: &Work,
) where
    T: Send + Sync,
    Make: Fn() -> S + Sync,
    Work: Fn(&[T], &mut [T], &mut S) + Sync,
{
    let band = (BAND / out_plane).max(1) * out_plane;
    par_out_chunks(out, band, &|first, planes| {
        let mut kept = scratch();
        for (index, plane) in planes.chunks_mut(out_plane).enumerate() {
            let at = first / out_plane + index;
            work(&data[at * data_plane..][..data_plane], plane, &mut kept);
        }
    });
}

/// Whether the windows are better summed side by side, a tap of every
/// window at a time, than each read whole.
///
/// Side by side, each tap of each window is an addition into a two-sum kept
/// in memory, and each run of windows a tap reads costs a little more; read
/// whole, each window is gathered a row at a time and summed in registers,
/// at a cost for the window and less for each value. The weights are what
/// each of these was measured to take, in tenths of a nanosecond.
fn side_by_side(extents: &Extents, window: &FixedWindow) -> bool {
    let (rows, cols) = window.kernel;
    let taps = rows * cols;
    let per_run = 80 / extents.out_w.max(1);
    let per_tap = if window.stride.1 == 1 { 20 } else { 25 };
    let apart = taps * (per_run + per_tap) + 35;
    // A window in one piece of the plane is read where it lies.
    let gathered = if rows == 1 || cols >= extents.in_w {
        0
    } else {
        50 * rows
    };
    let whole = 250 + gathered + 6 * taps;
    apart < whole
}

/// `row(place, values)` for each run of windows that one tap of the kernel
/// reads from one row of the input: `place` is where in the input plane the
/// run's first window reads, and `values` where in the output plane the
/// run's windows lie.
fn for_each_tap_row(
    extents: Extents,
    window: &FixedWindow,
    mut row: impl FnMut(usize, Range<usize>),
) {
    let Extents {
        in_h,
        in_w,
        out_h,
        out_w,
        ..
    } = extents;
    let FixedWindow {
        kernel,
        stride,
        padding,
        ..
    } = *window;
    for ky in 0..kernel.0 {
        let (oh_lo, oh_hi) = in_bounds_range(ky, padding.0, in_h, stride.0, out_h);
        for kx in 0..kernel.1 {
            let (ow_lo, ow_hi) = in_bounds_range(kx, padding.1, in_w, stride.1, out_w);
            if oh_lo >= oh_hi || ow_lo >= ow_hi {
                continue;
            }
            let from = ow_lo * stride.1 + kx - padding.1;
            for oh in oh_lo..oh_hi {
                let place = (oh * stride.0 + ky - padding.0) * in_w + from;
                row(place, oh * out_w + ow_lo..oh * out_w + ow_hi);
            }
        }
    }
}

/// The mean of each window of each plane of `data`: the value nearest the
/// exact sum of what the window reads over what it divides by.
pub(crate) fn pooled<T: Element + Output + Default>(
    data: &[T],
    extents: Extents,
    window: &FixedWindow,
) -> Vec<T> {
    let layout = Layout::fixed(extents, window);
    if side_by_side(&extents, window) {
        layout.summed_side_by_side(data, window)
    } else {
        layout.means(data)
    }
}

/// The mean of each adaptive window of each plane of `data`, as [`pooled`]
/// rounds a fixed window's.
pub(crate) fn pooled_adaptively<T: Element + Output + Default>(
    data: &[T],
    extents: Extents,
) -> Vec<T> {
    match extents.even_window() {
        Some(window) => pooled(data, extents, &window),
        None => Layout::adaptive(extents).means(data),
    }
}

/// The gradient an average pool's input takes from `flowing`, the gradient
/// of its output: for each cell the value nearest the exact sum, over the
/// windows that read it, of each window's gradient over its divisor.
pub(crate) fn gradient<T: Element + Output + Default>(
    flowing: &[T],
    extents: Extents,
    window: &FixedWindow,
) -> Vec<T> {
    let layout = Layout::fixed(extents, window);
    let FixedWindow { kernel, stride, .. } = *window;
    if stride.0 >= kernel.0 && stride.1 >= kernel.1 && kernel.0 * kernel.1 < DIRECT {
        return layout.shares_apart(flowing);
    }
    match layout.shared_divisor() {
        Some(divisor) => layout.shares_over(flowing, window, divisor),
        None => layout.shares_of_quotients(flowing),
    }
}

/// The gradient an adaptive average pool's input takes from `flowing`, as
/// [`gradient`] rounds a fixed window's.
pub(crate) fn adaptive_gradient<T: Element + Output + Default>(
    flowing: &[T],
    extents: Extents,
) -> Vec<T> {
    match extents.even_window() {
        Some(window) => gradient(flowing, extents, &window),
        None => Layout::adaptive(extents).shares_of_quotients(flowing),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values of these tests are whole numbers over `2^SCALE`.
    const SCALE: u32 = 24;

    /// The float of `bits` significant bits nearest `numerator / denominator`,
    /// a tie going to the even one, by long division.
    fn nearest(numerator: i128, denominator: i128, bits: u32) -> f64 {
        if numerator == 0 {
            return 0.0;
        }
        let (mut whole, divisor) = (numerator.unsigned_abs(), denominator.unsigned_abs());
        let mut doubled = 0i32;
        while whole / divisor < 1u128 << (bits + 1) {
            whole <<= 1;
            doubled += 1;
        }
        let (quotient, left) = (whole / divisor, whole % divisor);
        let mut dropped = 0u32;
        while quotient >> dropped >= 1u128 << bits {
            dropped += 1;
        }
        let (kept, lost) = (quotient >> dropped, quotient & ((1u128 << dropped) - 1));
        let half = 1u128 << (dropped - 1);
        let up = lost > half || (lost == half && (left != 0 || kept & 1 == 1));
        let magnitude = (kept + u128::from(up)) as f64 * 2f64.powi(dropped as i32 - doubled);
        if numerator < 0 { -magnitude } else { magnitude }
    }

    /// Whole numbers that over `2^SCALE` are floats of `bits` bits: that
    /// many bits at a place of their own, of either sign, zeros among them.
    fn wholes(seed: u64, len: usize, bits: u32) -> Vec<i128> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state.is_multiple_of(11) {
                    return 0;
                }
                let digits = i128::from(state >> (64 - bits)) << (state % 12);
                if state & (1 << 20) == 0 {
                    digits
                } else {
                    -digits
                }
            })
            .collect()
    }

    fn floats<T: Output>(wholes: &[i128]) -> Vec<T> {
        wholes
            .iter()
            .map(|&whole| T::narrow(whole as f64 * 2f64.powi(-(SCALE as i32))))
            .collect()
    }

    fn lcm(a: i128, b: i128) -> i128 {
        let (mut x, mut y) = (a, b);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        a / x * b
    }

    /// The cells of the input that the window at `(oh, ow)` reads, taken tap
    /// by tap, and what it divides by.
    fn fixed_reads(
        extents: Extents,
        window: &FixedWindow,
        oh: usize,
        ow: usize,
    ) -> (Vec<usize>, i128) {
        let mut cells = Vec::new();
        for ky in 0..window.kernel.0 {
            for kx in 0..window.kernel.1 {
                let (ih, iw) = (oh * window.stride.0 + ky, ow * window.stride.1 + kx);
                let inside = (window.padding.0..extents.in_h + window.padding.0).contains(&ih)
                    && (window.padding.1..extents.in_w + window.padding.1).contains(&iw);
                if inside {
                    cells.push((ih - window.padding.0) * extents.in_w + iw - window.padding.1);
                }
            }
        }
        let divisor = if window.count_include_pad {
            window.kernel.0 * window.kernel.1
        } else {
            cells.len()
        };
        (cells, divisor as i128)
    }

    fn adaptive_reads(extents: Extents, oh: usize, ow: usize) -> (Vec<usize>, i128) {
        let mut cells = Vec::new();
        for ih in adaptive_window(oh, extents.in_h, extents.out_h) {
            for iw in adaptive_window(ow, extents.in_w, extents.out_w) {
                cells.push(ih * extents.in_w + iw);
            }
        }
        let divisor = cells.len() as i128;
        (cells, divisor)
    }

    /// Checks the means and the gradients of one pool against the exact
    /// quotients, for an element type of `bits` significant bits.
    fn check<T: Element + Output + Default + std::fmt::Debug>(
        extents: Extents,
        window: Option<&FixedWindow>,
        bits: u32,
        seed: u64,
    ) {
        let (plane_in, plane_out) = (extents.in_h * extents.in_w, extents.out_h * extents.out_w);
        let reads = |oh: usize, ow: usize| match window {
            Some(window) => fixed_reads(extents, window, oh, ow),
            None => adaptive_reads(extents, oh, ow),
        };
        let label = format!("{extents:?} {window:?} at {bits} bits");

        let input = wholes(seed, extents.planes * plane_in, bits);
        let data: Vec<T> = floats(&input);
        let mut exact_means = Vec::with_capacity(extents.planes * plane_out);
        for plane in 0..extents.planes {
            for slot in 0..plane_out {
                let (cells, divisor) = reads(slot / extents.out_w, slot % extents.out_w);
                let sum: i128 = cells
                    .iter()
                    .map(|&cell| input[plane * plane_in + cell])
                    .sum();
                exact_means.push(if divisor == 0 {
                    0.0
                } else {
                    nearest(sum, divisor << SCALE, bits)
                });
            }
        }
        // The pool as it chooses, and each way it can take.
        let layout = match window {
            Some(window) => Layout::fixed(extents, window),
            None => Layout::adaptive(extents),
        };
        let mut ways: Vec<(&str, Vec<T>)> = vec![("whole", layout.means(&data))];
        match window {
            Some(window) => {
                ways.push(("chosen", pooled(&data, extents, window)));
                ways.push(("side by side", layout.summed_side_by_side(&data, window)));
            }
            None => ways.push(("chosen", pooled_adaptively(&data, extents))),
        }
        for (way, means) in &ways {
            assert_eq!(means.len(), exact_means.len(), "{label} {way}");
            for (slot, (mean, exact)) in means.iter().zip(&exact_means).enumerate() {
                assert_eq!(
                    mean.widen().to_bits(),
                    exact.to_bits(),
                    "{label}: mean {slot} {way}"
                );
            }
        }

        let flowed = wholes(seed + 1, extents.planes * plane_out, bits);
        let flowing: Vec<T> = floats(&flowed);
        let mut exact_cells = Vec::with_capacity(extents.planes * plane_in);
        for plane in 0..extents.planes {
            // Each cell's shares, as numerators over their divisors.
            let mut shares: Vec<Vec<(i128, i128)>> = vec![Vec::new(); plane_in];
            for slot in 0..plane_out {
                let (cells, divisor) = reads(slot / extents.out_w, slot % extents.out_w);
                for cell in cells {
                    shares[cell].push((flowed[plane * plane_out + slot], divisor));
                }
            }
            for shares in &shares {
                let common = shares
                    .iter()
                    .fold(1, |common, &(_, divisor)| lcm(common, divisor));
                let sum: i128 = shares
                    .iter()
                    .map(|&(whole, divisor)| whole * (common / divisor))
                    .sum();
                exact_cells.push(nearest(sum, common << SCALE, bits));
            }
        }
        let mut ways: Vec<(&str, Vec<T>)> =
            vec![("quotients", layout.shares_of_quotients(&flowing))];
        match window {
            Some(window) => {
                ways.push(("chosen", gradient(&flowing, extents, window)));
                if let Some(divisor) = layout.shared_divisor() {
                    ways.push(("one divisor", layout.shares_over(&flowing, window, divisor)));
                }
                if window.stride.0 >= window.kernel.0 && window.stride.1 >= window.kernel.1 {
                    ways.push(("apart", layout.shares_apart(&flowing)));
                }
            }
            None => ways.push(("chosen", adaptive_gradient(&flowing, extents))),
        }
        for (way, cells) in &ways {
            assert_eq!(cells.len(), exact_cells.len(), "{label} {way}");
            for (cell, (taken, exact)) in cells.iter().zip(&exact_cells).enumerate() {
                assert_eq!(
                    taken.widen().to_bits(),
                    exact.to_bits(),
                    "{label}: cell {cell} {way}"
                );
            }
        }
    }

    fn extents_of(planes: usize, input: (usize, usize), window: &FixedWindow) -> Extents {
        let out = |dim: usize, kernel: usize, stride: usize, padding: usize| {
            (dim + 2 * padding - kernel) / stride + 1
        };
        Extents {
            planes,
            in_h: input.0,
            in_w: input.1,
            out_h: out(input.0, window.kernel.0, window.stride.0, window.padding.0),
            out_w: out(input.1, window.kernel.1, window.stride.1, window.padding.1),
        }
    }

    #[test]
    fn fixed_windows_round_once() {
        // Windows side by side and read whole, strided and not, with the
        // padding counted and left out, apart, overlapping and with gaps.
        let cases = [
            ((9, 20), (3, 3), (1, 1), (1, 1), true),
            ((9, 20), (3, 3), (1, 1), (1, 1), false),
            ((8, 21), (2, 3), (2, 2), (1, 1), false),
            ((8, 21), (2, 3), (2, 2), (1, 1), true),
            ((7, 7), (3, 3), (2, 2), (1, 1), true),
            ((7, 7), (3, 3), (2, 2), (1, 1), false),
            ((6, 6), (2, 2), (2, 2), (0, 0), true),
            ((12, 32), (2, 2), (2, 2), (0, 0), true),
            ((5, 16), (5, 4), (1, 1), (2, 2), false),
            ((4, 12), (2, 2), (3, 1), (0, 1), true),
            ((2, 2), (4, 4), (1, 1), (2, 2), false),
            ((2, 9), (4, 4), (1, 1), (2, 2), false),
            ((3, 40), (3, 1), (1, 1), (0, 0), true),
            ((6, 9), (6, 9), (1, 1), (0, 0), true),
            ((1, 30), (1, 7), (1, 3), (0, 3), false),
        ];
        for (index, &(input, kernel, stride, padding, count_include_pad)) in
            cases.iter().enumerate()
        {
            let window = FixedWindow {
                kernel,
                stride,
                padding,
                count_include_pad,
            };
            let extents = extents_of(3, input, &window);
            check::<f32>(extents, Some(&window), 24, 100 + index as u64);
            check::<f64>(extents, Some(&window), 53, 200 + index as u64);
        }
    }

    #[test]
    fn adaptive_windows_round_once() {
        // Uneven windows, more windows than cells, windows that divide the
        // plane evenly, and the one window that is the whole plane.
        let cases = [
            ((7, 10), (3, 4)),
            ((5, 5), (7, 9)),
            ((8, 12), (4, 3)),
            ((6, 6), (1, 1)),
            ((9, 16), (3, 16)),
            ((10, 7), (4, 7)),
            ((3, 17), (2, 11)),
            ((16, 16), (8, 8)),
        ];
        for (index, &(input, output)) in cases.iter().enumerate() {
            let extents = Extents {
                planes: 3,
                in_h: input.0,
                in_w: input.1,
                out_h: output.0,
                out_w: output.1,
            };
            check::<f32>(extents, None, 24, 300 + index as u64);
            check::<f64>(extents, None, 53, 400 + index as u64);
        }
    }

    #[test]
    fn many_small_planes_are_each_their_own() {
        // More planes than one task takes.
        let window = FixedWindow {
            kernel: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            count_include_pad: false,
        };
        let extents = extents_of(700, (3, 9), &window);
        check::<f32>(extents, Some(&window), 24, 500);
        let adaptive = Extents {
            planes: 900,
            in_h: 3,
            in_w: 5,
            out_h: 2,
            out_w: 2,
        };
        check::<f64>(adaptive, None, 53, 501);
    }

    #[test]
    fn halfway_means_and_shares_go_to_the_even_value() {
        let step = f32::EPSILON;
        // Two values a last place apart: their mean is halfway between them.
        let window = FixedWindow {
            kernel: (1, 2),
            stride: (1, 1),
            padding: (0, 0),
            count_include_pad: true,
        };
        let data: Vec<f32> = (0..10).map(|i| 1.0 + step * i as f32).collect();
        let extents = extents_of(1, (1, 10), &window);
        let means = pooled(&data, extents, &window);
        for (i, mean) in means.iter().enumerate() {
            // Between `1 + i` and `1 + (i + 1)` last places: the even one.
            let even = if i % 2 == 0 { i } else { i + 1 };
            assert_eq!(*mean, 1.0 + step * even as f32, "mean {i}");
        }
        // The same halves as shares of a gradient.
        let cells = gradient(&data[..9], extents, &window);
        for (i, cell) in cells.iter().enumerate().skip(1).take(8) {
            let even = if i % 2 == 1 { i - 1 } else { i };
            assert_eq!(*cell, 1.0 + step * even as f32, "cell {i}");
        }
    }

    #[test]
    fn zeros_are_positive_and_special_values_stay_where_they_are_read() {
        let window = FixedWindow {
            kernel: (2, 2),
            stride: (1, 1),
            padding: (1, 1),
            count_include_pad: false,
        };
        let extents = extents_of(1, (3, 9), &window);
        let positive = 0.0f64.to_bits();

        // Negative zeros, and values that cancel, make positive zeros.
        let mut data = vec![-0.0f64; 27];
        data[4] = 1.5;
        data[5] = -1.5;
        for window in [
            window,
            FixedWindow {
                count_include_pad: true,
                ..window
            },
        ] {
            let means = pooled(&data, extents, &window);
            assert!(
                means
                    .iter()
                    .all(|mean| *mean != 0.0 || mean.to_bits() == positive)
            );
            assert_eq!(means[extents.out_w + 5].to_bits(), positive);
            let mut flowing = vec![-0.0f64; extents.out_h * extents.out_w];
            flowing[extents.out_w + 4] = 3.0;
            flowing[extents.out_w + 5] = -3.0;
            let cells = gradient(&flowing, extents, &window);
            assert!(
                cells
                    .iter()
                    .all(|cell| *cell != 0.0 || cell.to_bits() == positive)
            );
            assert_eq!(cells[4].to_bits(), positive);
            assert_eq!(cells[3], 0.75);
        }
        let adaptive = Extents {
            planes: 1,
            in_h: 3,
            in_w: 9,
            out_h: 2,
            out_w: 4,
        };
        let means = pooled_adaptively(&data, adaptive);
        assert!(
            means
                .iter()
                .all(|mean| *mean != 0.0 || mean.to_bits() == positive)
        );
        let cells = adaptive_gradient(&[-0.0f64; 8], adaptive);
        assert!(cells.iter().all(|cell| cell.to_bits() == positive));

        // An infinity reaches the windows that read it and no others; two
        // of opposite signs in one window make a NaN.
        let mut data = vec![1.0f32; 27];
        data[9 + 2] = f32::INFINITY;
        data[9 + 7] = f32::NEG_INFINITY;
        data[9 + 8] = f32::INFINITY;
        let means = pooled(&data, extents, &window);
        for (slot, mean) in means.iter().enumerate() {
            let (oh, ow) = (slot / extents.out_w, slot % extents.out_w);
            let (cells, _) = fixed_reads(extents, &window, oh, ow);
            let reads = |cell: usize| cells.contains(&cell);
            if reads(16) && reads(17) {
                assert!(mean.is_nan(), "mean {slot}");
            } else if reads(11) || reads(17) {
                assert_eq!(*mean, f32::INFINITY, "mean {slot}");
            } else if reads(16) {
                assert_eq!(*mean, f32::NEG_INFINITY, "mean {slot}");
            } else {
                assert_eq!(*mean, 1.0, "mean {slot}");
            }
        }
        let mut flowing = vec![1.0f32; extents.out_h * extents.out_w];
        flowing[0] = f32::INFINITY;
        flowing[extents.out_w + 6] = f32::NAN;
        for window in [
            window,
            FixedWindow {
                count_include_pad: true,
                ..window
            },
        ] {
            let cells = gradient(&flowing, extents, &window);
            assert_eq!(cells[0], f32::INFINITY);
            assert!(cells[5].is_nan() && cells[6].is_nan() && cells[9 + 5].is_nan());
            assert!(cells[1].is_finite() && cells[4].is_finite() && cells[18].is_finite());
        }
    }

    #[test]
    fn shares_past_the_float_range_still_round_once() {
        // Gradients whose sum overflows before it is divided, and ones so
        // small that their quotients fall below the least float64.
        let window = FixedWindow {
            kernel: (1, 3),
            stride: (1, 1),
            padding: (0, 1),
            count_include_pad: true,
        };
        let extents = extents_of(1, (1, 12), &window);
        let cells = gradient(&[f64::MAX; 12], extents, &window);
        assert_eq!(cells[5], f64::MAX);
        assert_eq!(cells[0], f64::MAX / 3.0 * 2.0);
        let least = f64::from_bits(1);
        let cells = gradient(&[least; 12], extents, &window);
        assert_eq!(cells[5], least);
        // Two thirds of the least float64 is nearer it than zero.
        assert_eq!(cells[0], least);
        let uneven = FixedWindow {
            count_include_pad: false,
            ..window
        };
        // An end cell: a half and a third of the least float64, under it.
        let cells = gradient(&[least; 12], extents, &uneven);
        assert_eq!(cells[0], least);
        assert_eq!(cells[5], least);
        let cells = gradient(&[-least; 12], extents, &uneven);
        assert_eq!(cells[0], -least);
        let means = pooled(&[f64::MAX; 12], extents, &window);
        assert_eq!(means[5], f64::MAX);
        assert_eq!(means[0], f64::MAX / 3.0 * 2.0);

        // A negative mean or share too small to be anything but zero is a
        // negative zero, which is the nearest float of its sign, whichever
        // way it is taken.
        let negative_zero = (-0.0f64).to_bits();
        let apart = FixedWindow {
            kernel: (1, 3),
            stride: (1, 3),
            padding: (0, 0),
            count_include_pad: true,
        };
        let extents = extents_of(1, (1, 12), &apart);
        let layout = Layout::fixed(extents, &apart);
        let flowing = [-least; 4];
        for cells in [
            layout.shares_apart(&flowing),
            layout.shares_over(&flowing, &apart, 3),
            layout.shares_of_quotients(&flowing),
        ] {
            assert!(cells.iter().all(|cell| cell.to_bits() == negative_zero));
        }
        let data: Vec<f64> = (0..12)
            .map(|i| if i % 3 == 0 { -least } else { 0.0 })
            .collect();
        for means in [
            layout.means(&data),
            layout.summed_side_by_side(&data, &apart),
        ] {
            assert!(means.iter().all(|mean| mean.to_bits() == negative_zero));
        }
    }
}
