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
//! does not place is read whole and summed exactly. A volume is a stack of
//! planes and a window of one a stack of a plane's windows, so a slice of
//! windows takes the taps of each input slice they reach into the same
//! two-sums.
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

/// At least how many elements the slices of one task come to: enough to be
/// worth a task, which makes its scratch once for all of them.
const BAND: usize = 4096;

/// The divisors a share can be divided by in float64 and narrowed without a
/// second rounding showing: under `2^24`, a float32's quotient by one is
/// never within a float64's rounding of halfway between two float32s
/// unless it is exactly there.
const DIRECT: usize = 1 << 24;

/// The volumes an average pool reads and the volumes it writes, each a
/// stack of planes; a 2-D pool's are a plane deep.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extents {
    /// How many volumes: the batch times the channels.
    pub(crate) volumes: usize,
    pub(crate) in_d: usize,
    pub(crate) in_h: usize,
    pub(crate) in_w: usize,
    pub(crate) out_d: usize,
    pub(crate) out_h: usize,
    pub(crate) out_w: usize,
}

impl Extents {
    /// `volumes` planes of `input`, pooled to planes of `output`.
    pub(crate) fn planar(volumes: usize, input: (usize, usize), output: (usize, usize)) -> Self {
        Self {
            volumes,
            in_d: 1,
            in_h: input.0,
            in_w: input.1,
            out_d: 1,
            out_h: output.0,
            out_w: output.1,
        }
    }

    fn plane_in(&self) -> usize {
        self.in_h * self.in_w
    }

    fn plane_out(&self) -> usize {
        self.out_h * self.out_w
    }

    fn volume_in(&self) -> usize {
        self.in_d * self.plane_in()
    }

    fn volume_out(&self) -> usize {
        self.out_d * self.plane_out()
    }

    /// Whether a volume has nothing to read or nothing to write.
    fn empty(&self) -> bool {
        self.volume_in() == 0 || self.volume_out() == 0
    }

    /// The fixed window that pools these extents, where the output's divide
    /// the input's: the windows of an adaptive pool are then all of one
    /// size and none overlaps another.
    fn even_window(&self) -> Option<FixedWindow> {
        let size = |input: usize, output: usize| {
            (output > 0 && input >= output && input.is_multiple_of(output)).then(|| input / output)
        };
        Some(FixedWindow::tiling([
            size(self.in_d, self.out_d)?,
            size(self.in_h, self.out_h)?,
            size(self.in_w, self.out_w)?,
        ]))
    }
}

/// A window of one size, stepped across a padded volume: its extent, step
/// and padding along the depth, the height and the width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedWindow {
    pub(crate) kernel: [usize; 3],
    pub(crate) stride: [usize; 3],
    pub(crate) padding: [usize; 3],
    /// Whether the padding a window lies over counts towards its divisor.
    pub(crate) count_include_pad: bool,
}

impl FixedWindow {
    /// A window of a plane, a slice deep.
    pub(crate) fn planar(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Self {
        Self {
            kernel: [1, kernel.0, kernel.1],
            stride: [1, stride.0, stride.1],
            padding: [0, padding.0, padding.1],
            count_include_pad,
        }
    }

    /// Windows as large as their steps and over no padding, which tile the
    /// volume.
    fn tiling(size: [usize; 3]) -> Self {
        Self {
            kernel: size,
            stride: size,
            padding: [0; 3],
            count_include_pad: true,
        }
    }

    /// How many cells the window covers, padding included.
    fn taps(&self) -> usize {
        self.kernel.iter().product()
    }

    /// Whether no two windows read one cell.
    fn apart(&self) -> bool {
        (0..3).all(|axis| self.stride[axis] >= self.kernel[axis])
    }
}

/// Where the windows of a volume lie, and what each divides by.
struct Layout {
    extents: Extents,
    /// The input slices that each slice of windows reads, the input rows
    /// that each row of them does, and the input columns each column does.
    depths: Vec<Range<usize>>,
    rows: Vec<Range<usize>>,
    cols: Vec<Range<usize>>,
    /// What every window divides by, where that is not the cells it reads.
    fixed: Option<usize>,
}

impl Layout {
    fn fixed(extents: Extents, window: &FixedWindow) -> Self {
        // A window's taps from `index * stride - padding`, kept to the
        // cells there are.
        let reach = |axis: usize, out: usize, dim: usize| {
            let (kernel, stride, padding) = (
                window.kernel[axis],
                window.stride[axis],
                window.padding[axis],
            );
            (0..out)
                .map(|index| {
                    let first = index * stride;
                    let end = (first + kernel).saturating_sub(padding).min(dim);
                    first.saturating_sub(padding).min(end)..end
                })
                .collect()
        };
        Self {
            extents,
            depths: reach(0, extents.out_d, extents.in_d),
            rows: reach(1, extents.out_h, extents.in_h),
            cols: reach(2, extents.out_w, extents.in_w),
            fixed: window.count_include_pad.then_some(window.taps()),
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
            depths: reach(extents.out_d, extents.in_d),
            rows: reach(extents.out_h, extents.in_h),
            cols: reach(extents.out_w, extents.in_w),
            fixed: None,
        }
    }

    /// The slice, row and column of the window at `slot` of an output
    /// volume.
    fn place(&self, slot: usize) -> (usize, usize, usize) {
        let (plane, out_w) = (self.extents.plane_out(), self.extents.out_w);
        (slot / plane, slot % plane / out_w, slot % out_w)
    }

    /// What the window at `slot` of an output volume divides its sum by.
    fn divisor(&self, slot: usize) -> usize {
        let (od, oh, ow) = self.place(slot);
        self.fixed
            .unwrap_or(self.depths[od].len() * self.rows[oh].len() * self.cols[ow].len())
    }

    /// The one count every window divides by, where there is one.
    fn shared_divisor(&self) -> Option<usize> {
        if self.fixed.is_some() {
            return self.fixed;
        }
        let even =
            |ranges: &[Range<usize>]| ranges.windows(2).all(|pair| pair[0].len() == pair[1].len());
        let even = even(&self.depths) && even(&self.rows) && even(&self.cols);
        (even && !self.extents.empty()).then(|| self.divisor(0))
    }

    /// The slices of an output volume whose windows read slice `id` of the
    /// input.
    fn reading(&self, id: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.depths.len()).filter(move |&od| self.depths[od].contains(&id))
    }

    /// Each window that reads cell `slot` of slice `id` of an input volume:
    /// its place in the output volume, and what it divides by.
    fn readers(&self, id: usize, slot: usize, mut each: impl FnMut(usize, usize)) {
        let Extents { in_w, out_w, .. } = self.extents;
        let plane_out = self.extents.plane_out();
        let (ih, iw) = (slot / in_w, slot % in_w);
        for od in self.reading(id) {
            for (oh, rows) in self.rows.iter().enumerate() {
                if !rows.contains(&ih) {
                    continue;
                }
                for (ow, cols) in self.cols.iter().enumerate() {
                    if cols.contains(&iw) {
                        let window = od * plane_out + oh * out_w + ow;
                        each(window, self.divisor(window));
                    }
                }
            }
        }
    }

    /// The mean of the window at `slot` of an output volume, read whole from
    /// `volume`: where it lies in one piece and divides by its own size it
    /// is the mean of that piece, and otherwise its values are brought
    /// together, with a zero for each cell of padding its divisor counts.
    fn mean_of<T: Element + Output + Default>(
        &self,
        volume: &[T],
        slot: usize,
        window: &mut Vec<T>,
    ) -> T {
        let (in_w, plane_in) = (self.extents.in_w, self.extents.plane_in());
        let (od, oh, ow) = self.place(slot);
        let (depths, rows, cols) = (&self.depths[od], &self.rows[oh], &self.cols[ow]);
        let inside = depths.len() * rows.len() * cols.len();
        if inside == 0 {
            return T::default();
        }
        let divisor = self.divisor(slot);
        // A run of one row, whole rows of one slice, or whole slices.
        let flat = (rows.len() == 1 || cols.len() == in_w)
            && (depths.len() == 1 || rows.len() * cols.len() == plane_in);
        if divisor == inside && flat {
            let start = depths.start * plane_in + rows.start * in_w + cols.start;
            return mean_to(&volume[start..][..inside]);
        }
        window.clear();
        for id in depths.clone() {
            let slice = &volume[id * plane_in..][..plane_in];
            for ih in rows.clone() {
                window.extend_from_slice(&slice[ih * in_w..][cols.clone()]);
            }
        }
        window.resize(divisor, T::default());
        mean_to(window)
    }

    /// The mean of every window, each read whole.
    fn means<T: Element + Output + Default>(&self, data: &[T]) -> Vec<T> {
        let e = self.extents;
        let mut means = vec![T::default(); e.volumes * e.volume_out()];
        if e.empty() {
            return means;
        }
        let (volume_in, plane_out) = (e.volume_in(), e.plane_out());
        let pool = |at: usize, means: &mut [T], window: &mut Vec<T>| {
            let volume = &data[at / e.out_d * volume_in..][..volume_in];
            let first = at % e.out_d * plane_out;
            for (k, mean) in means.iter_mut().enumerate() {
                *mean = self.mean_of(volume, first + k, window);
            }
        };
        for_each_slice(&mut means, plane_out, &Vec::new, &pool);
        means
    }

    /// The mean of every window of `window`, the windows of a slice summed
    /// side by side, a tap of every window at a time; a window whose mean
    /// the bound leaves open is read whole.
    fn summed_side_by_side<T: Element + Output + Default>(
        &self,
        data: &[T],
        window: &FixedWindow,
    ) -> Vec<T> {
        let e = self.extents;
        let mut means = vec![T::default(); e.volumes * e.volume_out()];
        if e.empty() {
            return means;
        }
        let (plane_in, volume_in, plane_out) = (e.plane_in(), e.volume_in(), e.plane_out());
        let step = window.stride[2];
        let counts: Vec<f64> = (0..e.volume_out())
            .map(|slot| self.divisor(slot) as f64)
            .collect();
        type Scratch<T> = (Tallies, Vec<usize>, Vec<T>);
        let scratch = || (Tallies::new(plane_out), Vec::new(), Vec::new());
        let pool = |at: usize, means: &mut [T], (tallies, open, kept): &mut Scratch<T>| {
            let volume = &data[at / e.out_d * volume_in..][..volume_in];
            let od = at % e.out_d;
            tallies.clear(0.0);
            for id in self.depths[od].clone() {
                let plane = &volume[id * plane_in..][..plane_in];
                for_each_tap_row(e, window, |place, windows| {
                    if step == 1 {
                        tallies.add(windows.start, &plane[place..place + windows.len()]);
                    } else {
                        tallies.add_spaced(windows.start, &plane[place..], step, windows.len());
                    }
                });
            }
            let first = od * plane_out;
            open.clear();
            tallies.means_into(0, &counts[first..first + plane_out], means, open);
            for &k in open.iter() {
                means[k] = self.mean_of(volume, first + k, kept);
            }
        };
        for_each_slice(&mut means, plane_out, &scratch, &pool);
        means
    }

    /// The gradient where no cell is read twice: each takes the one share
    /// there is for it, a quotient rounded once.
    ///
    /// For float32 the quotient is taken in float64 and narrowed, which is
    /// a second rounding only in name: a float32 over a whole number under
    /// [`DIRECT`] is either a halfway point between two float32s, which
    /// float64 holds, or further from one than float64 rounds.
    fn shares_apart<T: Element + Output + Default>(&self, flowing: &[T]) -> Vec<T> {
        let e = self.extents;
        let mut gradient = vec![T::default(); e.volumes * e.volume_in()];
        if e.empty() {
            return gradient;
        }
        let (plane_in, plane_out, volume_out) = (e.plane_in(), e.plane_out(), e.volume_out());
        let share_out = |at: usize, cells: &mut [T], _: &mut ()| {
            let flowing = &flowing[at / e.in_d * volume_out..][..volume_out];
            for od in self.reading(at % e.in_d) {
                for k in 0..plane_out {
                    let slot = od * plane_out + k;
                    let (share, divisor) = (flowing[slot].wide(), self.divisor(slot));
                    if share == 0.0 || divisor == 0 {
                        continue;
                    }
                    let share = T::narrow(share / divisor as f64);
                    let (rows, cols) = (&self.rows[k / e.out_w], &self.cols[k % e.out_w]);
                    for ih in rows.clone() {
                        cells[ih * e.in_w..][cols.clone()].fill(share);
                    }
                }
            }
        };
        for_each_slice(&mut gradient, plane_in, &|| (), &share_out);
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
        let e = self.extents;
        let mut gradient = vec![T::default(); e.volumes * e.volume_in()];
        if e.empty() || divisor == 0 {
            return gradient;
        }
        let (plane_in, plane_out, volume_out) = (e.plane_in(), e.plane_out(), e.volume_out());
        let step = window.stride[2];
        let counts = vec![divisor as f64; plane_in];
        type Scratch<T> = (Tallies, Vec<usize>, Vec<T>);
        let scratch = || (Tallies::new(plane_in), Vec::new(), Vec::new());
        let spread = |at: usize, cells: &mut [T], (tallies, open, kept): &mut Scratch<T>| {
            let flowing = &flowing[at / e.in_d * volume_out..][..volume_out];
            let id = at % e.in_d;
            tallies.clear(0.0);
            for od in self.reading(id) {
                let slice = &flowing[od * plane_out..][..plane_out];
                for_each_tap_row(e, window, |place, windows| {
                    if step == 1 {
                        tallies.add(place, &slice[windows]);
                    } else {
                        tallies.add_spread(place, step, &slice[windows]);
                    }
                });
            }
            open.clear();
            tallies.means_into(0, &counts, cells, open);
            for &slot in open.iter() {
                kept.clear();
                self.readers(id, slot, |window, _| kept.push(flowing[window]));
                cells[slot] = sum_over(kept, divisor as u64);
            }
        };
        for_each_slice(&mut gradient, plane_in, &scratch, &spread);
        gradient
    }

    /// The gradient where the windows may divide by counts of their own.
    ///
    /// A share `g / d` is taken as `q + r`: `q` the quotient rounded, and
    /// `r` the quotient of what `q` leaves of `g`, which a fused multiply
    /// finds exactly. The pair misses the share by the rounding of `r`
    /// alone, far below the last place of `q`, and that is counted with
    /// what the two-sum the pairs are added into has dropped.
    fn shares_of_quotients<T: Element + Output + Default>(&self, flowing: &[T]) -> Vec<T> {
        let e = self.extents;
        let mut gradient = vec![T::default(); e.volumes * e.volume_in()];
        if e.empty() {
            return gradient;
        }
        let (plane_in, plane_out, volume_out) = (e.plane_in(), e.plane_out(), e.volume_out());
        type Scratch = (Tallies, Vec<usize>, Vec<(f64, u64)>);
        let scratch = || (Tallies::new(plane_in), Vec::new(), Vec::new());
        let spread = |at: usize, cells: &mut [T], (tallies, open, terms): &mut Scratch| {
            let flowing = &flowing[at / e.in_d * volume_out..][..volume_out];
            let id = at % e.in_d;
            tallies.clear(0.0);
            for od in self.reading(id) {
                for k in 0..plane_out {
                    let slot = od * plane_out + k;
                    let share = flowing[slot].wide();
                    let (rows, cols) = (&self.rows[k / e.out_w], &self.cols[k % e.out_w]);
                    if share == 0.0 || rows.is_empty() || cols.is_empty() {
                        continue;
                    }
                    let divisor = self.divisor(slot) as f64;
                    let quotient = share / divisor;
                    let remainder = (-quotient).mul_add(divisor, share);
                    let rest = remainder / divisor;
                    // The rounding of `rest`: half its last place, or half
                    // the least float64 where it is that small; none where
                    // the quotient is the share.
                    let rounding = if remainder == 0.0 {
                        0.0
                    } else {
                        rest.abs() * 2f64.powi(-52) + f64::from_bits(1)
                    };
                    for ih in rows.clone() {
                        let at = ih * e.in_w + cols.start;
                        tallies.add_pair(at, cols.len(), (quotient, rest), rounding);
                    }
                }
            }
            open.clear();
            tallies.totals_into(0, cells, open);
            for &slot in open.iter() {
                terms.clear();
                self.readers(id, slot, |window, divisor| {
                    terms.push((flowing[window].wide(), divisor as u64));
                });
                cells[slot] = sum_of_quotients(terms);
            }
        };
        for_each_slice(&mut gradient, plane_in, &scratch, &spread);
        gradient
    }
}

/// `work(index, slice, scratch)` for each slice of `out`, `len` long and
/// numbered from the first, with scratch that a task makes once for the
/// slices it takes.
fn for_each_slice<T, S, Make, Work>(out: &mut [T], len: usize, scratch: &Make, work: &Work)
where
    T: Send,
    Make: Fn() -> S + Sync,
    Work: Fn(usize, &mut [T], &mut S) + Sync,
{
    let band = (BAND / len).max(1) * len;
    par_out_chunks(out, band, &|first, slices| {
        let mut kept = scratch();
        for (index, slice) in slices.chunks_mut(len).enumerate() {
            work(first / len + index, slice, &mut kept);
        }
    });
}

/// Whether the windows are better summed side by side, a tap of every
/// window at a time, than each read whole.
///
/// Side by side, each tap of each window is an addition into a two-sum kept
/// in memory, and each run of windows a tap reads costs a little more; read
/// whole, each window is gathered a run of a row at a time and summed in
/// registers, at a cost for the window and less for each value. The weights
/// are what each of these was measured to take, in tenths of a nanosecond.
fn side_by_side(extents: &Extents, window: &FixedWindow) -> bool {
    let [depth, rows, cols] = window.kernel;
    let taps = window.taps();
    let per_run = 80 / extents.out_w.max(1);
    let per_tap = if window.stride[2] == 1 { 20 } else { 25 };
    let apart = taps * (per_run + per_tap) + 35;
    // A window in one piece of the volume is read where it lies.
    let runs = if rows == 1 || cols >= extents.in_w {
        depth
    } else {
        depth * rows
    };
    let gathered = if runs == 1 { 0 } else { 50 * runs };
    let whole = 250 + gathered + 6 * taps;
    apart < whole
}

/// `row(place, windows)` for each run of windows of a slice that one tap of
/// the kernel within a plane reads from one row of an input plane: `place`
/// is where in the input plane the run's first window reads, and `windows`
/// where in the slice of the output the run's windows lie.
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
    for ky in 0..kernel[1] {
        let (oh_lo, oh_hi) = in_bounds_range(ky, padding[1], in_h, stride[1], out_h);
        for kx in 0..kernel[2] {
            let (ow_lo, ow_hi) = in_bounds_range(kx, padding[2], in_w, stride[2], out_w);
            if oh_lo >= oh_hi || ow_lo >= ow_hi {
                continue;
            }
            let from = ow_lo * stride[2] + kx - padding[2];
            for oh in oh_lo..oh_hi {
                let place = (oh * stride[1] + ky - padding[1]) * in_w + from;
                row(place, oh * out_w + ow_lo..oh * out_w + ow_hi);
            }
        }
    }
}

/// The mean of each window of each volume of `data`: the value nearest the
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

/// The mean of each adaptive window of each volume of `data`, as [`pooled`]
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
    if window.apart() && window.taps() < DIRECT {
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

    /// The cells of an input volume that the window at `slot` of the output
    /// reads, taken tap by tap, and what it divides by.
    fn fixed_reads(extents: Extents, window: &FixedWindow, slot: usize) -> (Vec<usize>, i128) {
        let plane = extents.out_h * extents.out_w;
        let at = [
            slot / plane,
            slot % plane / extents.out_w,
            slot % extents.out_w,
        ];
        let input = [extents.in_d, extents.in_h, extents.in_w];
        let mut cells = Vec::new();
        for kd in 0..window.kernel[0] {
            for ky in 0..window.kernel[1] {
                for kx in 0..window.kernel[2] {
                    let place = [kd, ky, kx];
                    let read: Option<Vec<usize>> = (0..3)
                        .map(|axis| {
                            let padded = at[axis] * window.stride[axis] + place[axis];
                            let padding = window.padding[axis];
                            (padding..input[axis] + padding)
                                .contains(&padded)
                                .then(|| padded - padding)
                        })
                        .collect();
                    if let Some(read) = read {
                        cells.push((read[0] * input[1] + read[1]) * input[2] + read[2]);
                    }
                }
            }
        }
        let divisor = if window.count_include_pad {
            window.taps()
        } else {
            cells.len()
        };
        (cells, divisor as i128)
    }

    fn adaptive_reads(extents: Extents, slot: usize) -> (Vec<usize>, i128) {
        let plane = extents.out_h * extents.out_w;
        let (od, oh, ow) = (
            slot / plane,
            slot % plane / extents.out_w,
            slot % extents.out_w,
        );
        let mut cells = Vec::new();
        for id in adaptive_window(od, extents.in_d, extents.out_d) {
            for ih in adaptive_window(oh, extents.in_h, extents.out_h) {
                for iw in adaptive_window(ow, extents.in_w, extents.out_w) {
                    cells.push((id * extents.in_h + ih) * extents.in_w + iw);
                }
            }
        }
        let divisor = cells.len() as i128;
        (cells, divisor)
    }

    /// Checks the means and the gradients of one pool, the way it chooses
    /// and every way it can take, against the exact quotients, for an
    /// element type of `bits` significant bits.
    fn check<T: Element + Output + Default + std::fmt::Debug>(
        extents: Extents,
        window: Option<&FixedWindow>,
        bits: u32,
        seed: u64,
    ) {
        let (volume_in, volume_out) = (extents.volume_in(), extents.volume_out());
        let reads = |slot: usize| match window {
            Some(window) => fixed_reads(extents, window, slot),
            None => adaptive_reads(extents, slot),
        };
        let label = format!("{extents:?} {window:?} at {bits} bits");

        let input = wholes(seed, extents.volumes * volume_in, bits);
        let data: Vec<T> = floats(&input);
        let mut exact_means = Vec::with_capacity(extents.volumes * volume_out);
        for volume in 0..extents.volumes {
            for slot in 0..volume_out {
                let (cells, divisor) = reads(slot);
                let sum: i128 = cells
                    .iter()
                    .map(|&cell| input[volume * volume_in + cell])
                    .sum();
                exact_means.push(if divisor == 0 {
                    0.0
                } else {
                    nearest(sum, divisor << SCALE, bits)
                });
            }
        }
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

        let flowed = wholes(seed + 1, extents.volumes * volume_out, bits);
        let flowing: Vec<T> = floats(&flowed);
        let mut exact_cells = Vec::with_capacity(extents.volumes * volume_in);
        for volume in 0..extents.volumes {
            // Each cell's shares, as numerators over their divisors.
            let mut shares: Vec<Vec<(i128, i128)>> = vec![Vec::new(); volume_in];
            for slot in 0..volume_out {
                let (cells, divisor) = reads(slot);
                for cell in cells {
                    shares[cell].push((flowed[volume * volume_out + slot], divisor));
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
                if window.apart() {
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

    /// The extents `window` pools `volumes` volumes of `input` to.
    fn extents_of(volumes: usize, input: [usize; 3], window: &FixedWindow) -> Extents {
        let out = |axis: usize| {
            (input[axis] + 2 * window.padding[axis] - window.kernel[axis]) / window.stride[axis] + 1
        };
        Extents {
            volumes,
            in_d: input[0],
            in_h: input[1],
            in_w: input[2],
            out_d: out(0),
            out_h: out(1),
            out_w: out(2),
        }
    }

    /// A window of a plane, and the extents it pools `volumes` planes of
    /// `input` to.
    fn planar(
        volumes: usize,
        input: (usize, usize),
        geometry: ((usize, usize), (usize, usize), (usize, usize)),
        count_include_pad: bool,
    ) -> (Extents, FixedWindow) {
        let (kernel, stride, padding) = geometry;
        let window = FixedWindow::planar(kernel, stride, padding, count_include_pad);
        (extents_of(volumes, [1, input.0, input.1], &window), window)
    }

    #[test]
    fn fixed_windows_round_once() {
        // Windows side by side and read whole, strided and not, with the
        // padding counted and left out, apart, overlapping and with gaps.
        let cases = [
            ((9, 20), ((3, 3), (1, 1), (1, 1)), true),
            ((9, 20), ((3, 3), (1, 1), (1, 1)), false),
            ((8, 21), ((2, 3), (2, 2), (1, 1)), false),
            ((8, 21), ((2, 3), (2, 2), (1, 1)), true),
            ((7, 7), ((3, 3), (2, 2), (1, 1)), true),
            ((7, 7), ((3, 3), (2, 2), (1, 1)), false),
            ((6, 6), ((2, 2), (2, 2), (0, 0)), true),
            ((12, 32), ((2, 2), (2, 2), (0, 0)), true),
            ((5, 16), ((5, 4), (1, 1), (2, 2)), false),
            ((4, 12), ((2, 2), (3, 1), (0, 1)), true),
            ((2, 2), ((4, 4), (1, 1), (2, 2)), false),
            ((2, 9), ((4, 4), (1, 1), (2, 2)), false),
            ((3, 40), ((3, 1), (1, 1), (0, 0)), true),
            ((6, 9), ((6, 9), (1, 1), (0, 0)), true),
            ((1, 30), ((1, 7), (1, 3), (0, 3)), false),
        ];
        for (index, &(input, geometry, count_include_pad)) in cases.iter().enumerate() {
            let (extents, window) = planar(3, input, geometry, count_include_pad);
            check::<f32>(extents, Some(&window), 24, 100 + index as u64);
            check::<f64>(extents, Some(&window), 53, 200 + index as u64);
        }
    }

    #[test]
    fn windows_of_volumes_round_once() {
        // Overlapping along the depth and apart along the width, apart
        // everywhere, overlapping everywhere with padding counted and not,
        // windows only along the depth, and a wide volume summed side by
        // side.
        let cases = [
            ([4, 5, 6], [2, 3, 3], [1, 1, 2], [1, 1, 1], true),
            ([4, 5, 6], [2, 3, 3], [1, 1, 2], [1, 1, 1], false),
            ([5, 4, 7], [3, 2, 2], [2, 1, 1], [1, 0, 1], false),
            ([4, 4, 4], [2, 2, 2], [2, 2, 2], [0, 0, 0], true),
            ([2, 3, 12], [2, 1, 3], [1, 1, 3], [0, 0, 0], true),
            ([3, 5, 5], [3, 3, 3], [1, 2, 2], [1, 1, 1], false),
            ([3, 5, 5], [3, 3, 3], [1, 2, 2], [1, 1, 1], true),
            ([7, 3, 3], [5, 1, 1], [1, 1, 1], [0, 0, 0], true),
            ([7, 3, 3], [5, 1, 1], [1, 1, 1], [2, 0, 0], false),
            ([3, 4, 40], [2, 2, 2], [1, 1, 1], [1, 1, 1], false),
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
            let extents = extents_of(2, input, &window);
            check::<f32>(extents, Some(&window), 24, 600 + index as u64);
            check::<f64>(extents, Some(&window), 53, 700 + index as u64);
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
            let extents = Extents::planar(3, input, output);
            check::<f32>(extents, None, 24, 300 + index as u64);
            check::<f64>(extents, None, 53, 400 + index as u64);
        }
    }

    #[test]
    fn many_small_planes_are_each_their_own() {
        // More planes than one task takes.
        let (extents, window) = planar(700, (3, 9), ((3, 3), (1, 1), (1, 1)), false);
        check::<f32>(extents, Some(&window), 24, 500);
        check::<f64>(Extents::planar(900, (3, 5), (2, 2)), None, 53, 501);
        let window = FixedWindow {
            kernel: [2, 2, 2],
            stride: [1, 1, 1],
            padding: [1, 0, 1],
            count_include_pad: false,
        };
        check::<f32>(extents_of(300, [3, 2, 3], &window), Some(&window), 24, 502);
    }

    #[test]
    fn halfway_means_and_shares_go_to_the_even_value() {
        let step = f32::EPSILON;
        // Two values a last place apart: their mean is halfway between them.
        let (extents, window) = planar(1, (1, 10), ((1, 2), (1, 1), (0, 0)), true);
        let data: Vec<f32> = (0..10).map(|i| 1.0 + step * i as f32).collect();
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
        let (extents, window) = planar(1, (3, 9), ((2, 2), (1, 1), (1, 1)), false);
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
        let adaptive = Extents::planar(1, (3, 9), (2, 4));
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
            let (cells, _) = fixed_reads(extents, &window, slot);
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
        let (extents, window) = planar(1, (1, 12), ((1, 3), (1, 1), (0, 1)), true);
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
        let (extents, apart) = planar(1, (1, 12), ((1, 3), (1, 3), (0, 0)), true);
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
