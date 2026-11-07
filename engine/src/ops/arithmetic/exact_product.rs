// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Products rounded once.
//!
//! A product multiplied out in floats rounds at every factor: of two hundred
//! float64 products of a few factors near one, more than half were not the
//! float nearest the exact product, and a running product drifted further
//! with every step. Its partial products also leave the range where the
//! answer does not.
//!
//! Here a product is carried as a pair, `(hi + lo) · 2^exponent`, `hi` near
//! one and the exponent kept apart, so that no partial product overflows or
//! underflows. Each factor multiplies `hi` with the product's exact error
//! found by a fused multiply-add, and `lo` with one rounding; what a factor
//! leaves out is a few `u²` of the product, so a bound by the count of
//! factors settles the answer unless it lies within that of a halfway point.
//! Then the product is formed in whole numbers 128 bits wide, each cut below
//! them counted, and twice as wide again for as long as the cuts leave the
//! answer open -- which, once the whole product fits, they cannot.
//!
//! A factor between `2^-62` and `2^62` is taken as it stands and the pair
//! renormalized every eight, so that it costs three operations; another is
//! split into its mantissa and exponent first. Lanes of one block, runs or
//! columns side by side, and the pieces of one long running product are each
//! advanced together in vector registers, a running product settled by its
//! bound, or past either end of the range by its exponent, without a branch.

use super::exact_sum::{Element, Output, fused, product_of};
use super::rounded_once::{product_error, rounded};
use crate::parallel::Parts;
use std::sync::OnceLock;

/// Running products a block keeps side by side, so that a factor need not
/// wait on the last.
const LANES: usize = 8;
/// Factors a block takes; the blocks of a long run go across the pool.
const BLOCK: usize = 8192;
/// Factors a lane takes between renormalizations. Its leading part grows by
/// under a factor of two a factor, so it stays far inside the range.
const RENORM: usize = 16;
/// The relative error a factor, or a third of a merge, may add: `3u²`, a
/// hair over, with room for the roundings in the subnormal range a tiny `lo`
/// can take, each under the least subnormal against a `hi` of at least one.
const PER_STEP: f64 = 3.01 * (1.0 / (1u128 << 106) as f64);
/// [`PER_STEP`] with room for the roundings of a bound worked out from it.
const PER_STEP_BOUND: f64 = PER_STEP * (1.0 + 1.0 / (1u64 << 40) as f64);
/// The least and greatest magnitudes of a factor a fast lane takes as it
/// stands, unsplit: [`GROUP`] of them leave a pair that started in `[1, 2)`
/// far inside the normal range, and each product's error exact.
const SMALL: f64 = 1.0 / (1u64 << 62) as f64;
const LARGE: f64 = (1u64 << 62) as f64;
/// Factors a fast lane takes between renormalizations.
const GROUP: usize = 8;
/// What a factor counts for in a fast lane, in [`PER_STEP`]s. Between
/// renormalizations `lo` grows by up to `u` of the product a factor, so the
/// `k`-th factor of a group rounds by under `(k + 1)u²` with a fused
/// multiply-add and `(2k + 1)u²` without: under `17u²` either way.
const FAST_STEPS: u64 = 6;
/// Runs a band advances side by side.
const BAND: usize = 16;
/// Runs at least this long are reduced one at a time across a block's
/// lanes, shorter ones side by side in a band.
const LONG: usize = 64;
/// Runs at least this long have their running products cut into pieces
/// advanced side by side, when there are too few runs to fill the pool.
const LONG_RUN: usize = 2048;

/// A product of finite, nonzero factors carried as `±(hi + lo) · 2^exponent`
/// -- `hi` positive and normal, `lo` under half its last place -- with the
/// count of roundings that went into it, each under [`PER_STEP`] of it.
#[derive(Clone, Copy, Debug)]
struct Product {
    hi: f64,
    lo: f64,
    exponent: i64,
    negative: bool,
    steps: u64,
}

impl Product {
    const ONE: Self = Self {
        hi: 1.0,
        lo: 0.0,
        exponent: 0,
        negative: false,
        steps: 0,
    };

    /// `hi` brought into `[1, 2)`, `lo` and the exponent with it.
    #[inline(always)]
    fn normalized(mut self) -> Self {
        renormalize(&mut self.hi, &mut self.lo, &mut self.exponent);
        self
    }

    /// The product of two: the leading parts' product split exactly, the
    /// cross terms beside it with three roundings and the product of the
    /// trailing parts left out, under three steps' worth together.
    #[inline(always)]
    fn times<const FMA: bool>(self, other: Self) -> Self {
        let p = self.hi * other.hi;
        let p_error = if FMA {
            self.hi.mul_add(other.hi, -p)
        } else {
            product_error(self.hi, other.hi, p)
        };
        let s = (self.hi * other.lo + self.lo * other.hi) + p_error;
        let hi = p + s;
        Self {
            hi,
            lo: s - (hi - p),
            exponent: self.exponent + other.exponent,
            negative: self.negative ^ other.negative,
            steps: self.steps + other.steps + 3,
        }
        .normalized()
    }

    /// The value nearest the exact product, when the bound says which it is.
    fn settled<T: Output>(self) -> Option<T> {
        let product = self.normalized();
        let sign = if product.negative { -1.0 } else { 1.0 };
        let exponent = product.exponent;
        // `hi + lo` is in `[1, 2)`, so the value is under `2^(exponent + 1)`.
        if exponent > i64::from(T::GREATEST) {
            return Some(T::narrow(sign * f64::INFINITY));
        }
        if exponent < i64::from(T::LEAST) - 2 {
            // Under half the least subnormal.
            return Some(T::narrow(sign * 0.0));
        }
        let relative = (product.steps as f64 + 1.0) * PER_STEP_BOUND;
        let bound = product.hi * relative;
        if exponent >= -1022 {
            // Scaled exactly, `lo` short of the least subnormal at worst.
            let scale = power_of_two(exponent);
            let value = sign * (product.hi * scale);
            let rest = sign * (product.lo * scale);
            return T::settle(value, rest, bound * scale + f64::from_bits(2));
        }
        // A float64 result among the subnormals, rounded on their grid:
        // `k = (hi + lo) · 2^(exponent + 1074)` to the nearest whole number.
        let scale = power_of_two(exponent + 1074);
        let (whole, rest) = (product.hi * scale, product.lo * scale);
        let nearest = whole.round();
        let fraction = (whole - nearest) + rest;
        let reach = bound * scale + 2f64.powi(-50);
        let count = if fraction.abs() + reach < 0.5 {
            nearest
        } else if fraction - reach > 0.5 {
            nearest + 1.0
        } else if fraction + reach < -0.5 {
            nearest - 1.0
        } else {
            return None;
        };
        Some(T::narrow(sign * f64::from_bits(count as u64)))
    }
}

/// `2^exponent` for an exponent in the normal range.
#[inline(always)]
fn power_of_two(exponent: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// `hi`, a positive normal number, brought into `[1, 2)`: `lo` scaled with
/// it, exactly unless it falls among the subnormals, and the exponent taking
/// the difference.
#[inline(always)]
fn renormalize(hi: &mut f64, lo: &mut f64, exponent: &mut i64) {
    let shift = ((hi.to_bits() >> 52) as i64) - 1023;
    let scale = f64::from_bits(((1023 - shift) as u64) << 52);
    *hi *= scale;
    *lo *= scale;
    *exponent += shift;
}

/// One factor into a lane: its sign into `sign`, whether it is a zero, an
/// infinity or a NaN into `special`, and its magnitude `m · 2^e`, `m` in
/// `[1, 2)` -- a subnormal scaled into the normal range first -- into the
/// pair and the exponent. Garbage for a special factor, whose lane the
/// caller then sets aside; never a NaN or an overflow, since `m` is in
/// `[1, 2)` whatever the factor.
#[inline(always)]
fn step<const FMA: bool>(
    hi: &mut f64,
    lo: &mut f64,
    exponent: &mut i64,
    sign: &mut u64,
    special: &mut bool,
    value: f64,
) {
    const MANTISSA: u64 = (1 << 52) - 1;
    const ONE: u64 = 1023 << 52;
    let bits = value.to_bits();
    *sign ^= bits >> 63;
    let magnitude = value.abs();
    *special |= !((magnitude > 0.0) & (magnitude < f64::INFINITY));
    let tiny = magnitude < f64::MIN_POSITIVE;
    let scaled = if tiny {
        magnitude * 2f64.powi(64)
    } else {
        magnitude
    };
    let scaled_bits = scaled.to_bits();
    let mantissa = f64::from_bits((scaled_bits & MANTISSA) | ONE);
    let shift = ((scaled_bits >> 52) as i64) - 1023 - if tiny { 64 } else { 0 };
    let p = *hi * mantissa;
    let p_error = if FMA {
        hi.mul_add(mantissa, -p)
    } else {
        product_error(*hi, mantissa, p)
    };
    let s = *lo * mantissa + p_error;
    let next = p + s;
    *lo = s - (next - p);
    *hi = next;
    *exponent += shift;
}

/// One factor's magnitude, unsplit, into a fast lane's pair: `hi · value`
/// split exactly, and `lo · value` with that error beside it.
#[inline(always)]
fn fast_step<const FMA: bool>(hi: &mut f64, lo: &mut f64, value: f64) {
    let p = *hi * value;
    if FMA {
        let error = hi.mul_add(value, -p);
        *lo = lo.mul_add(value, error);
    } else {
        let error = product_error(*hi, value, p);
        *lo = *lo * value + error;
    }
    *hi = p;
}

/// A fast lane's pair renormalized: `hi` the float64 nearest `hi + lo` and
/// `lo` the exact rest, both scaled to bring `|hi|` into `[1, 2)`; the sign
/// rides in `hi`. Garbage, but no overflow, for a pair that is not finite.
#[inline(always)]
fn settle_pair(hi: &mut f64, lo: &mut f64, exponent: &mut i64) {
    let sum = *hi + *lo;
    let rest = *lo - (sum - *hi);
    let shift = ((sum.to_bits() >> 52) & 0x7ff) as i64 - 1023;
    let scale = f64::from_bits((((1023 - shift) as u64) & 0x7ff) << 52);
    *hi = sum * scale;
    *lo = rest * scale;
    *exponent += shift;
}

/// The least magnitude of `low` and `value`, and the greatest of `high` and
/// it, without the care a NaN would need: a NaN factor shows in the pair.
#[inline(always)]
fn spread(low: &mut f64, high: &mut f64, value: f64) {
    let magnitude = value.abs();
    *low = if magnitude < *low { magnitude } else { *low };
    *high = if magnitude > *high { magnitude } else { *high };
}

/// Whether a fast lane's factors all lay between [`SMALL`] and [`LARGE`],
/// so that its pair holds their product: none was a NaN either.
#[inline(always)]
fn took_all(low: f64, high: f64, hi: f64) -> bool {
    (low >= SMALL) & (high <= LARGE) & !hi.is_nan()
}

/// The product of factors among which a zero, an infinity or a NaN came:
/// NaN for a NaN or a zero and an infinity together, otherwise a zero or an
/// infinity, signed by the parity of the negative factors.
#[inline(always)]
fn special_value(negative: bool, zero: bool, infinite: bool, nan: bool) -> f64 {
    let sign = if negative { -1.0 } else { 1.0 };
    if nan | (zero & infinite) {
        f64::NAN
    } else if zero {
        sign * 0.0
    } else {
        sign * f64::INFINITY
    }
}

/// A block's [`LANES`] products: pairs, exponents, signs and whether a
/// special factor came.
struct Lanes {
    hi: [f64; LANES],
    lo: [f64; LANES],
    exponent: [i64; LANES],
    sign: [u64; LANES],
    special: [bool; LANES],
}

impl Lanes {
    const ONE: Self = Self {
        hi: [1.0; LANES],
        lo: [0.0; LANES],
        exponent: [0; LANES],
        sign: [0; LANES],
        special: [false; LANES],
    };

    /// The lanes, `rest` multiplied into the first by [`step`], multiplied
    /// together pairwise -- or the first alone where no factor reached the
    /// others -- each counted `steps` and `rest` more; with whether any
    /// factor was special, which leaves the product meaningless.
    #[inline(always)]
    fn joined<I: Element, const FMA: bool>(
        mut self,
        rest: &[I],
        steps: u64,
        filled: bool,
    ) -> (Product, bool) {
        for value in rest {
            step::<FMA>(
                &mut self.hi[0],
                &mut self.lo[0],
                &mut self.exponent[0],
                &mut self.sign[0],
                &mut self.special[0],
                value.wide(),
            );
        }
        let steps = steps + rest.len() as u64;
        let lane = |lane: usize| {
            Product {
                hi: self.hi[lane],
                lo: self.lo[lane],
                exponent: self.exponent[lane],
                negative: self.sign[lane] == 1,
                steps,
            }
            .normalized()
        };
        if !filled {
            return (lane(0), self.special[0]);
        }
        let mut lanes: [Product; LANES] = std::array::from_fn(lane);
        let mut width = LANES;
        while width > 1 {
            width /= 2;
            for k in 0..width {
                lanes[k] = lanes[2 * k].times::<FMA>(lanes[2 * k + 1]);
            }
        }
        (lanes[0], self.special.iter().any(|&special| special))
    }
}

/// One block's factors in [`LANES`] lanes, each split by [`step`], the
/// factors left over in the first: for a block a fast lane cannot take.
#[inline(always)]
fn block_lanes<I: Element, const FMA: bool>(values: &[I]) -> (Product, bool) {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let mut lanes = Lanes::ONE;
    for group in chunks.chunks(RENORM) {
        for chunk in group {
            for (lane, value) in chunk.iter().enumerate() {
                step::<FMA>(
                    &mut lanes.hi[lane],
                    &mut lanes.lo[lane],
                    &mut lanes.exponent[lane],
                    &mut lanes.sign[lane],
                    &mut lanes.special[lane],
                    value.wide(),
                );
            }
        }
        for lane in 0..LANES {
            renormalize(
                &mut lanes.hi[lane],
                &mut lanes.lo[lane],
                &mut lanes.exponent[lane],
            );
        }
    }
    lanes.joined::<I, FMA>(rest, chunks.len() as u64, !chunks.is_empty())
}

/// [`block_lanes`] for a block whose factors' magnitudes lie between
/// [`SMALL`] and [`LARGE`]: each multiplies its lane's pair unsplit, the
/// pair renormalized every [`GROUP`] factors, so that most factors take
/// three operations. `None` where a factor does not lie there.
#[inline(always)]
fn fast_lanes<I: Element, const FMA: bool>(values: &[I]) -> Option<(Product, bool)> {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let (mut hi, mut lo) = ([1f64; LANES], [0f64; LANES]);
    let (mut low, mut high) = ([1f64; LANES], [1f64; LANES]);
    let mut exponent = [0i64; LANES];
    for group in chunks.chunks(GROUP) {
        for chunk in group {
            for lane in 0..LANES {
                let value = chunk[lane].wide();
                spread(&mut low[lane], &mut high[lane], value);
                fast_step::<FMA>(&mut hi[lane], &mut lo[lane], value);
            }
        }
        for lane in 0..LANES {
            settle_pair(&mut hi[lane], &mut lo[lane], &mut exponent[lane]);
        }
    }
    if !(0..LANES).all(|lane| took_all(low[lane], high[lane], hi[lane])) {
        return None;
    }
    let mut lanes = Lanes::ONE;
    for lane in 0..LANES {
        let negative = hi[lane] < 0.0;
        lanes.sign[lane] = u64::from(negative);
        lanes.hi[lane] = hi[lane].abs();
        lanes.lo[lane] = if negative { -lo[lane] } else { lo[lane] };
        lanes.exponent[lane] = exponent[lane];
    }
    let steps = chunks.len() as u64 * FAST_STEPS;
    Some(lanes.joined::<I, FMA>(rest, steps, !chunks.is_empty()))
}

/// The carried product of one block, fast where its factors allow, with
/// whether any factor was special; in the widest build the CPU runs.
fn block_product<I: Element>(values: &[I]) -> (Product, bool) {
    #[inline(always)]
    fn either<I: Element, const FMA: bool>(values: &[I]) -> (Product, bool) {
        fast_lanes::<I, FMA>(values).unwrap_or_else(|| block_lanes::<I, FMA>(values))
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn with_fma<I: Element>(values: &[I]) -> (Product, bool) {
        either::<I, true>(values)
    }
    #[cfg(target_arch = "x86_64")]
    if fused() {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { with_fma(values) };
    }
    either::<I, false>(values)
}

/// The product of `values`, as [`block_product`] carries it, a block at a
/// time across the pool for a long run, the blocks multiplied pairwise.
fn carried_product<I: Element>(values: &[I]) -> (Product, bool) {
    if values.len() <= BLOCK {
        return block_product(values);
    }
    let span = |nth: usize| &values[nth * BLOCK..((nth + 1) * BLOCK).min(values.len())];
    let blocks = values.len().div_ceil(BLOCK);
    let parts = if values.len() >= crate::ops::map::PAR_THRESHOLD {
        crate::parallel::map_tasks(blocks, &|nth| block_product(span(nth)))
    } else {
        (0..blocks).map(|nth| block_product(span(nth))).collect()
    };
    let special = parts.iter().any(|&(_, special)| special);
    let fma = fused();
    let product = crate::ops::util::pairwise_fold(
        parts.into_iter().map(|(product, _)| product).collect(),
        Product::ONE,
        |a, b| {
            if fma {
                a.times::<true>(b)
            } else {
                a.times::<false>(b)
            }
        },
    );
    (product, special)
}

/// The value nearest the exact product of `values`, one for none.
///
/// A NaN factor, or a zero and an infinity together, makes NaN; otherwise a
/// zero makes a zero and an infinity an infinity, each signed by the parity
/// of the negative factors, as multiplying them out does. Finite, nonzero
/// factors make the value nearest their exact product, however far past the
/// range its partial products would have gone.
pub(crate) fn product_to<I: Element, T: Output>(values: &[I]) -> T {
    let (product, special) = carried_product(values);
    if special {
        return special_product(values.iter().map(|value| value.wide()));
    }
    product.settled().unwrap_or_else(|| match values.len() {
        0 => T::narrow(1.0),
        len => Prefix::empty().rounded(len - 1, &|index| values[index].wide()),
    })
}

/// The product of factors at least one of which is a zero, an infinity or a
/// NaN.
#[cold]
fn special_product<T: Output>(values: impl Iterator<Item = f64>) -> T {
    let (mut negative, mut zero, mut infinite, mut nan) = (false, false, false, false);
    for value in values {
        negative ^= value.is_sign_negative();
        zero |= value == 0.0;
        infinite |= value.is_infinite();
        nan |= value.is_nan();
    }
    T::narrow(special_value(negative, zero, infinite, nan))
}

/// The cheap pool helper: `task` for each of `count` tasks, across the pool
/// only when the `work` they share is worth it.
fn tasks(count: usize, work: usize, task: &(dyn Fn(usize) + Sync)) {
    if work < crate::ops::map::PAR_THRESHOLD {
        (0..count).for_each(task);
    } else {
        crate::parallel::for_each_task(count, task);
    }
}

/// How the lanes of a band lie: lane `k`'s value at step `s` sits at
/// `base + k·lane + s·step`.
#[derive(Clone, Copy)]
struct Geometry {
    base: usize,
    lane: usize,
    step: usize,
    width: usize,
    /// Where a reduction's first lane's output goes; the rest follow it.
    out: usize,
}

impl Geometry {
    /// The count of bands of the `[slabs, len, inner]` runs along the
    /// middle axis -- runs side by side when `inner` is one, else a slab's
    /// columns -- and each band by its index.
    fn bands(slabs: usize, len: usize, inner: usize) -> (usize, impl Fn(usize) -> Self) {
        let (lanes, groups, lane, step) = if inner == 1 {
            (slabs, 1, len, 1)
        } else {
            (inner, slabs, 1, inner)
        };
        let per_group = lanes.div_ceil(BAND);
        let band = move |flat: usize| {
            let (group, first) = (flat / per_group, flat % per_group * BAND);
            Self {
                base: group * len * inner + first * lane,
                lane,
                step,
                width: BAND.min(lanes - first),
                out: group * lanes + first,
            }
        };
        (groups * per_group, band)
    }

    /// Where lane `k`'s value at `step` is.
    #[inline(always)]
    fn at(&self, k: usize, step: usize) -> usize {
        self.base + k * self.lane + step * self.step
    }

    /// Each lane's values for `count` steps from `step`, read from `input`:
    /// a band's row at a time where the lanes lie side by side, else a
    /// lane's run of steps at a time, so that each line of memory is read
    /// once a tile.
    #[inline(always)]
    fn load<I: Element>(&self, input: &[I], step: usize, count: usize, tile: &mut Tile<f64>) {
        if self.lane == 1 {
            for (offset, row) in tile.iter_mut().enumerate().take(count) {
                let at = self.at(0, step + offset);
                if self.width == BAND {
                    // A whole band: a copy of known length, not a call.
                    let input: &[I; BAND] = input[at..at + BAND].try_into().expect("a band");
                    for (value, input) in row.iter_mut().zip(input) {
                        *value = input.wide();
                    }
                } else {
                    for (value, input) in row.iter_mut().zip(&input[at..at + self.width]) {
                        *value = input.wide();
                    }
                }
            }
            return;
        }
        for k in 0..self.width {
            let at = self.at(k, step);
            for (offset, row) in tile.iter_mut().enumerate().take(count) {
                row[k] = input[at + offset * self.step].wide();
            }
        }
    }

    /// Each lane's products for `count` steps from `step` into `output`,
    /// laid out as the values are from `offset` on.
    #[inline(always)]
    fn store<T: Copy>(
        &self,
        output: &Parts<T>,
        offset: usize,
        step: usize,
        count: usize,
        tile: &Tile<T>,
    ) {
        if self.lane == 1 {
            for (row_offset, row) in tile.iter().enumerate().take(count) {
                let at = offset + self.at(0, step + row_offset);
                // SAFETY: a band's places are its own, side by side inside
                // `output`, which is laid out as the values are.
                let start = unsafe { output.start().add(at) };
                if self.width == BAND {
                    // A whole band: a copy of known length, not a call.
                    unsafe { start.cast::<[T; BAND]>().write_unaligned(*row) };
                } else {
                    unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), start, self.width) };
                }
            }
            return;
        }
        for k in 0..self.width {
            let at = offset + self.at(k, step);
            for (row_offset, row) in tile.iter().enumerate().take(count) {
                // SAFETY: as above, a lane's run of places at a time.
                unsafe { *output.start().add(at + row_offset * self.step) = row[k] };
            }
        }
    }
}

/// The products along the middle axis of `input`, laid out `[outer, len,
/// inner]`, into `output`, laid out `[outer, inner]`: each the value nearest
/// the exact product of its slice, as [`product_to`] gives it.
pub(crate) fn axis_products<I: Element, T: Output>(
    input: &[I],
    output: &mut [T],
    len: usize,
    inner: usize,
) {
    if output.is_empty() {
        return;
    }
    if len == 0 {
        output.fill(T::narrow(1.0));
        return;
    }
    if inner == 1 && len >= LONG {
        crate::ops::map::par_out_chunks(output, ((1 << 14) / len).max(1), &|start, out| {
            for (offset, slot) in out.iter_mut().enumerate() {
                let first = (start + offset) * len;
                *slot = product_to(&input[first..first + len]);
            }
        });
        return;
    }
    let slabs = output.len() / inner;
    let (bands, band) = Geometry::bands(slabs, len, inner);
    let per_task = ((1 << 14) / (BAND * len)).max(1);
    let fma = fused();
    let parts = Parts::of(output);
    tasks(bands.div_ceil(per_task), input.len(), &|task| {
        for flat in task * per_task..((task + 1) * per_task).min(bands) {
            let geometry = band(flat);
            // SAFETY: a band's outputs lie side by side inside `output`, and
            // are no other band's.
            let out = unsafe {
                std::slice::from_raw_parts_mut(parts.start().add(geometry.out), geometry.width)
            };
            reduce(fma, input, geometry, len, out);
        }
    });
}

/// [`reduce_band`] in the widest build the CPU runs.
fn reduce<I: Element, T: Output>(
    fma: bool,
    input: &[I],
    geometry: Geometry,
    len: usize,
    out: &mut [T],
) {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn with_fma<I: Element, T: Output>(input: &[I], geometry: Geometry, len: usize, out: &mut [T]) {
        reduce_band::<I, T, true>(input, geometry, len, out);
    }
    #[cfg(target_arch = "x86_64")]
    if fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { with_fma(input, geometry, len, out) };
    }
    let _ = fma;
    reduce_band::<I, T, false>(input, geometry, len, out);
}

/// The products of a band's runs of `len` factors, its lanes advanced side
/// by side as [`fast_lanes`] advances a block's, into `out`. A run with a
/// factor a fast lane does not take, or whose bound does not settle it, is
/// gathered and taken by [`product_to`], or formed exactly.
#[inline(always)]
fn reduce_band<I: Element, T: Output, const FMA: bool>(
    input: &[I],
    geometry: Geometry,
    len: usize,
    out: &mut [T],
) {
    let width = geometry.width;
    let (mut hi, mut lo) = ([1f64; BAND], [0f64; BAND]);
    let (mut low, mut high) = ([1f64; BAND], [1f64; BAND]);
    let mut exponent = [0i64; BAND];
    let mut tile = [[1f64; BAND]; GROUP];
    for first in (0..len).step_by(GROUP) {
        let count = GROUP.min(len - first);
        geometry.load(input, first, count, &mut tile);
        for values in &tile[..count] {
            for k in 0..BAND {
                spread(&mut low[k], &mut high[k], values[k]);
                fast_step::<FMA>(&mut hi[k], &mut lo[k], values[k]);
            }
        }
        for k in 0..BAND {
            settle_pair(&mut hi[k], &mut lo[k], &mut exponent[k]);
        }
    }
    let steps = len as u64 * FAST_STEPS;
    let mut run: Vec<I> = Vec::new();
    for (k, slot) in out.iter_mut().enumerate().take(width) {
        if !took_all(low[k], high[k], hi[k]) {
            run.clear();
            run.extend((0..len).map(|step| input[geometry.at(k, step)]));
            *slot = product_to(&run);
            continue;
        }
        let negative = hi[k] < 0.0;
        let product = Product {
            hi: hi[k].abs(),
            lo: if negative { -lo[k] } else { lo[k] },
            exponent: exponent[k],
            negative,
            steps,
        };
        *slot = product.settled().unwrap_or_else(|| match len {
            0 => T::narrow(1.0),
            len => Prefix::empty().rounded(len - 1, &|step| input[geometry.at(k, step)].wide()),
        });
    }
}

/// Limbs a [`Wide`] product starts with: a cut then drops under `2^-127` of
/// the product, so that only a product within a few cuts of that of a
/// halfway point needs more.
const START_WIDTH: usize = 2;

/// A product of finite, nonzero factors in whole numbers cut to a fixed
/// width: `±limbs · 2^exponent`, the limbs little-endian with the top one's
/// top bit set, and the count of cuts that went into it. A cut drops under
/// `2^(1 - 64·width)` of what it keeps, so the exact product is at least the
/// value held and at most that times `1 + 2^(1 - 64·width)` to the power of
/// the cuts.
#[derive(Clone, Debug)]
struct Wide {
    limbs: Vec<u64>,
    exponent: i64,
    negative: bool,
    cuts: u64,
}

impl Wide {
    /// One, `width` limbs wide.
    fn one(width: usize) -> Self {
        let mut limbs = vec![0u64; width];
        limbs[width - 1] = 1 << 63;
        Self {
            limbs,
            exponent: 1 - 64 * width as i64,
            negative: false,
            cuts: 0,
        }
    }

    /// This product times a finite, nonzero factor: the factor's odd part
    /// times the limbs, the limb that then falls off the bottom a cut unless
    /// it is zero, and the power of two into the exponent.
    fn times_factor(&mut self, value: f64) {
        let bits = value.to_bits();
        self.negative ^= bits >> 63 == 1;
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1 << 52) - 1);
        let (whole, shift) = if biased == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), biased - 1075)
        };
        let zeros = whole.trailing_zeros();
        let odd = whole >> zeros;
        self.exponent += shift + i64::from(zeros);
        if odd == 1 {
            return;
        }
        let mut carry = 0u64;
        for limb in &mut self.limbs {
            let full = u128::from(*limb) * u128::from(odd) + u128::from(carry);
            *limb = full as u64;
            carry = (full >> 64) as u64;
        }
        // The top limb is at least 2^63 and `odd` at least three, so some of
        // it carries out, and under 2^53 does: the limbs and the carry
        // shifted up until the top bit is set, the bottom limb dropped.
        let lead = carry.leading_zeros();
        let dropped = self.limbs[0] << lead;
        let width = self.limbs.len();
        for index in 0..width {
            let upper = if index + 1 < width {
                self.limbs[index + 1]
            } else {
                carry
            };
            self.limbs[index] = if lead == 0 {
                upper
            } else {
                (upper << lead) | (self.limbs[index] >> (64 - lead))
            };
        }
        self.cuts += u64::from(dropped != 0);
        self.exponent += 64 - i64::from(lead);
    }

    /// The product of two as wide as each other, cut to that width.
    fn times(&self, other: &Self) -> Self {
        let width = self.limbs.len();
        debug_assert_eq!(width, other.limbs.len());
        let full = product_of(&self.limbs, &other.limbs);
        // At least 2^(128·width - 2), so longer than the width.
        let length = bit_length(&full);
        let shift = length - 64 * width;
        let (word, bit) = (shift / 64, shift % 64);
        let limbs = (0..width)
            .map(|index| window(&full, word + index, bit))
            .collect();
        let cut = full[..word].iter().any(|&limb| limb != 0)
            || (bit != 0 && full[word] << (64 - bit) != 0);
        Self {
            limbs,
            exponent: self.exponent + other.exponent + shift as i64,
            negative: self.negative ^ other.negative,
            cuts: self.cuts + other.cuts + u64::from(cut),
        }
    }

    /// The value nearest the exact product, when the cuts leave one value
    /// for it to be; `scratch` is room to work in.
    fn rounded<T: Output>(&self, scratch: &mut Vec<u64>) -> Option<T> {
        let low = nearest::<T>(self.negative, &self.limbs, self.exponent);
        if self.cuts == 0 {
            return Some(low);
        }
        // `(1 + 2^(1 - 64·width))^cuts` is under `1 + 1.01·cuts·2^(1 -
        // 64·width)`, and the limbs are under `2^(64·width)`, so the exact
        // product is under the limbs plus `3·cuts` in their last place.
        // Rounding is monotonic: the two ends rounding alike settle it.
        scratch.clear();
        scratch.extend_from_slice(&self.limbs);
        scratch.push(0);
        let mut carry = self.cuts.saturating_mul(3);
        for limb in scratch.iter_mut() {
            let (sum, over) = limb.overflowing_add(carry);
            *limb = sum;
            carry = u64::from(over);
            if carry == 0 {
                break;
            }
        }
        let high = nearest::<T>(self.negative, scratch, self.exponent);
        (high == low).then_some(low)
    }
}

/// The number of bits up to the highest one set in `limbs`, nonzero.
fn bit_length(limbs: &[u64]) -> usize {
    let top = limbs
        .iter()
        .rposition(|&limb| limb != 0)
        .expect("a nonzero product");
    64 * top + 64 - limbs[top].leading_zeros() as usize
}

/// The 64 bits of `limbs` from bit `bit` of limb `word` up, zeros past the
/// top.
#[inline(always)]
fn window(limbs: &[u64], word: usize, bit: usize) -> u64 {
    let limb = |index: usize| limbs.get(index).copied().unwrap_or(0);
    if bit == 0 {
        limb(word)
    } else {
        (limb(word) >> bit) | (limb(word + 1) << (64 - bit))
    }
}

/// `±limbs · 2^exponent`, nonzero, rounded to the nearest `T`, ties to
/// even: its leading 127 bits, with whether any below them are set marked
/// in a 128th.
fn nearest<T: Output>(negative: bool, limbs: &[u64], exponent: i64) -> T {
    let length = bit_length(limbs);
    let dropped = length.saturating_sub(127);
    let (word, bit) = (dropped / 64, dropped % 64);
    let mut leading =
        u128::from(window(limbs, word, bit)) | (u128::from(window(limbs, word + 1, bit)) << 64);
    let mut least = exponent + dropped as i64;
    if dropped > 0 {
        let below = limbs[..word].iter().any(|&limb| limb != 0)
            || (bit != 0 && limbs[word] << (64 - bit) != 0);
        leading = (leading << 1) | u128::from(below);
        least -= 1;
    }
    // Far past either end, the answer is plain without the exponent fitting.
    let sign = if negative { -1.0 } else { 1.0 };
    if least + 127 > i64::from(T::GREATEST) + 130 {
        return T::narrow(sign * f64::INFINITY);
    }
    if least + 127 < i64::from(T::LEAST) - 130 {
        return T::narrow(sign * 0.0);
    }
    rounded(negative, leading, least as i32)
}

/// The exact product of a run's first `done` factors, as a [`Wide`], for
/// the running products a bound cannot settle: extended from where it was
/// last asked for, so that a run whose products stay at a halfway point --
/// one followed by ones -- takes a step for each, not a pass over the run.
struct Prefix {
    wide: Wide,
    done: usize,
    scratch: Vec<u64>,
    /// The last answer, as a float64, and the index it was for.
    last: Option<(usize, f64)>,
}

impl Prefix {
    /// The prefix of no factors.
    fn empty() -> Self {
        Self::after(Wide::one(START_WIDTH), 0)
    }

    /// The prefix of the first `done` factors, whose product is `wide`.
    fn after(wide: Wide, done: usize) -> Self {
        Self {
            wide,
            done,
            scratch: Vec::new(),
            last: None,
        }
    }

    /// The value nearest the exact product of the first `upto + 1` factors,
    /// all finite and nonzero, which `read` gives by index. Where the cuts
    /// leave it open, the product is formed again from the first factor,
    /// twice as wide, until they do not -- as they cannot once the whole
    /// product fits, with no cut at all.
    #[cold]
    #[inline(never)]
    fn rounded<T: Output>(&mut self, upto: usize, read: &dyn Fn(usize) -> f64) -> T {
        if let Some(value) = self.rescaled::<T>(upto, read) {
            self.last = Some((upto, value));
            return T::narrow(value);
        }
        let value: T = self.formed(upto, read);
        self.last = Some((upto, value.widen()));
        value
    }

    /// The last answer, where every factor since is a power of two, signed:
    /// scaling by a power of two changes neither which value is nearest nor
    /// how a tie goes while the products stay on the normal grid, so the
    /// product times them rounds to the rounding times them. The answer is
    /// kept at least twice the least normal `T` throughout, so that the
    /// products it stands for are normal too, and short of the range's end.
    /// A run that sits at a halfway point, followed by ones, takes this.
    fn rescaled<T: Output>(&self, upto: usize, read: &dyn Fn(usize) -> f64) -> Option<f64> {
        let (at, mut value) = self.last?;
        let least = f64::from_bits(((i64::from(T::LEAST_NORMAL) + 1024) as u64) << 52);
        let beyond = f64::from_bits(((i64::from(T::GREATEST) + 1024) as u64) << 52);
        let inside = |value: f64| (least..beyond).contains(&value.abs());
        if !inside(value) {
            return None;
        }
        for index in at + 1..=upto {
            let factor = read(index);
            let bits = factor.to_bits();
            let biased = (bits >> 52) & 0x7ff;
            if bits & ((1 << 52) - 1) != 0 || biased == 0 || biased == 0x7ff {
                return None;
            }
            value *= factor;
            if !inside(value) {
                return None;
            }
        }
        Some(value)
    }

    /// [`Prefix::rounded`] by forming the product.
    fn formed<T: Output>(&mut self, upto: usize, read: &dyn Fn(usize) -> f64) -> T {
        for index in self.done..=upto {
            self.wide.times_factor(read(index));
        }
        self.done = upto + 1;
        loop {
            if let Some(value) = self.wide.rounded(&mut self.scratch) {
                return value;
            }
            let mut wide = Wide::one(2 * self.wide.limbs.len());
            for index in 0..=upto {
                wide.times_factor(read(index));
            }
            self.wide = wide;
        }
    }
}

/// A running product, a factor at a time: the carried product of the finite,
/// nonzero factors so far, and whether a zero, an infinity or a NaN has come.
#[derive(Clone, Copy, Debug)]
struct Running {
    product: Product,
    sign: u64,
    special: bool,
    zero: bool,
    infinite: bool,
    nan: bool,
}

impl Running {
    const START: Self = Self {
        product: Product::ONE,
        sign: 0,
        special: false,
        zero: false,
        infinite: false,
        nan: false,
    };

    /// The running product after `factor`, rounded once; `None` where the
    /// bound cannot say, for the caller to form exactly.
    #[inline(always)]
    fn next<T: Output, const FMA: bool>(&mut self, factor: f64) -> Option<T> {
        self.zero |= factor == 0.0;
        self.infinite |= factor.is_infinite();
        self.nan |= factor.is_nan();
        if self.special | self.zero | self.infinite | self.nan {
            // Settled by what came, whatever comes of the rest but its sign.
            self.special = true;
            self.sign ^= factor.to_bits() >> 63;
            self.product.negative = self.sign == 1;
            let negative = self.product.negative;
            return Some(T::narrow(special_value(
                negative,
                self.zero,
                self.infinite,
                self.nan,
            )));
        }
        let product = &mut self.product;
        let (hi, lo, exponent) = (&mut product.hi, &mut product.lo, &mut product.exponent);
        step::<FMA>(hi, lo, exponent, &mut self.sign, &mut self.special, factor);
        renormalize(&mut product.hi, &mut product.lo, &mut product.exponent);
        product.steps += 1;
        product.negative = self.sign == 1;
        product.settled()
    }

    /// This running product followed by `block`'s factors, whose carried
    /// product is `product`.
    fn then<const FMA: bool>(mut self, block: &[f64], product: Product, special: bool) -> Self {
        if special {
            for &factor in block {
                self.zero |= factor == 0.0;
                self.infinite |= factor.is_infinite();
                self.nan |= factor.is_nan();
                self.sign ^= factor.to_bits() >> 63;
            }
            self.special = true;
        } else {
            self.sign ^= u64::from(product.negative);
        }
        self.product = self.product.times::<FMA>(product);
        self.product.negative = self.sign == 1;
        self
    }
}

/// A band's [`BAND`] running products, each lane a [`Product`]'s fields
/// kept apart -- the sign riding in `hi`, which takes each factor as it
/// stands and is renormalized only every [`GROUP`] factors -- with what
/// rounding it needs worked out at each renormalization, so that a step
/// over the lanes is one loop the compiler can carry out in vector
/// registers. A lane that met a special factor goes on as a [`Running`]
/// apart.
struct Band {
    hi: [f64; BAND],
    lo: [f64; BAND],
    exponent: [i64; BAND],
    steps: [f64; BAND],
    /// `2^exponent`, or NaN where that is not a normal float64.
    scale: [f64; BAND],
    /// The least `|hi + lo|` whose product is past `T`'s range, and the
    /// least that is not short of a quarter of its least subnormal.
    over: [f64; BAND],
    under: [f64; BAND],
    slow: [u64; BAND],
}

impl Band {
    /// A band whose lanes start where `start` says; a lane that starts
    /// special starts slow, in `slow`.
    fn new<T: Output>(
        width: usize,
        start: impl Fn(usize) -> Running,
        slow: &mut [Running],
    ) -> Self {
        let mut band = Self {
            hi: [1.0; BAND],
            lo: [0.0; BAND],
            exponent: [0; BAND],
            steps: [0.0; BAND],
            scale: [1.0; BAND],
            over: [f64::INFINITY; BAND],
            under: [0.0; BAND],
            slow: [0; BAND],
        };
        for k in 0..BAND {
            band.limits::<T>(k);
        }
        for (k, slot) in slow.iter_mut().enumerate().take(width) {
            let state = start(k);
            if state.special {
                band.slow[k] = 1;
                *slot = state;
            } else {
                band.set::<T>(k, state.product);
            }
        }
        band
    }

    /// Lane `k` set to `product`.
    fn set<T: Output>(&mut self, k: usize, product: Product) {
        let product = product.normalized();
        let sign = if product.negative { -1.0 } else { 1.0 };
        self.hi[k] = sign * product.hi;
        self.lo[k] = sign * product.lo;
        self.exponent[k] = product.exponent;
        self.steps[k] = product.steps as f64;
        self.limits::<T>(k);
    }

    /// Lane `k` as a [`Product`].
    fn product(&self, k: usize) -> Product {
        let (mut hi, mut lo, mut exponent) = (self.hi[k], self.lo[k], self.exponent[k]);
        settle_pair(&mut hi, &mut lo, &mut exponent);
        let negative = hi < 0.0;
        Product {
            hi: hi.abs(),
            lo: if negative { -lo } else { lo },
            exponent,
            negative,
            steps: self.steps[k] as u64,
        }
    }

    /// Lane `k`'s scale and limits for its exponent. Until the next
    /// renormalization `|hi|` lies within `2^±(GROUP · 62)` of one, so a
    /// limit past the normal range is passed by every value or by none.
    #[inline(always)]
    fn limits<T: Output>(&mut self, k: usize) {
        /// `2^exponent`, infinite above the normal range and zero below.
        #[inline(always)]
        fn power(exponent: i64) -> f64 {
            let bits = ((exponent.clamp(-1022, 1023) + 1023) as u64) << 52;
            if exponent > 1023 {
                f64::INFINITY
            } else if exponent < -1022 {
                0.0
            } else {
                f64::from_bits(bits)
            }
        }
        let exponent = self.exponent[k];
        let scale = power(exponent);
        self.scale[k] = if scale == 0.0 || scale == f64::INFINITY {
            f64::NAN
        } else {
            scale
        };
        self.over[k] = power(i64::from(T::GREATEST) + 1 - exponent);
        self.under[k] = power(i64::from(T::LEAST) - 2 - exponent);
    }

    /// Every lane's pair renormalized, as each `GROUP`-th factor leaves it.
    #[inline(always)]
    fn renormalize<T: Output>(&mut self) {
        for k in 0..BAND {
            settle_pair(&mut self.hi[k], &mut self.lo[k], &mut self.exponent[k]);
            self.limits::<T>(k);
        }
    }
}

/// One factor into each of a band's first `width` lanes, with each running
/// product rounded once into `out` and whether that was settled into
/// `settled`, without a branch: by the bound, or by a limit for a product
/// past either end of `T`'s range. A lane is not settled where it is slow;
/// where the factor lies past [`SMALL`] or [`LARGE`], which the lane does
/// not take; where the product is among float64's subnormals; or where the
/// bound cannot say.
#[inline(always)]
fn advance<T: Output, const FMA: bool>(
    band: &mut Band,
    values: &[f64; BAND],
    width: usize,
    out: &mut [T; BAND],
    settled: &mut [u64; BAND],
) {
    const SIGN: u64 = 1 << 63;
    for k in 0..width.min(BAND) {
        let value = values[k];
        let fits = (SMALL..=LARGE).contains(&value.abs()) & (band.slow[k] == 0);
        let (mut hi, mut lo) = (band.hi[k], band.lo[k]);
        fast_step::<FMA>(&mut hi, &mut lo, value);
        let steps = band.steps[k] + FAST_STEPS as f64;
        let sum = hi + lo;
        let rest = lo - (sum - hi);
        // `(sum + rest) · 2^exponent`, scaled exactly where it is normal;
        // `rest` short of the least subnormal at worst.
        let whole = sum * band.scale[k];
        let bound = (steps + 1.0) * PER_STEP_BOUND * whole.abs() + f64::from_bits(2);
        let (rounded, clear) = T::settle_lanes(whole, rest * band.scale[k], bound);
        let size = sum.abs();
        let over = size >= band.over[k];
        let under = size < band.under[k];
        let edge = if over { f64::INFINITY } else { 0.0 };
        let edge = f64::from_bits(edge.to_bits() | (sum.to_bits() & SIGN));
        out[k] = if over | under {
            T::narrow(edge)
        } else {
            rounded
        };
        let normal = whole.abs() >= f64::MIN_POSITIVE;
        settled[k] = u64::from(fits & (over | under | (clear & normal)));
        band.hi[k] = if fits { hi } else { band.hi[k] };
        band.lo[k] = if fits { lo } else { band.lo[k] };
        band.steps[k] = if fits { steps } else { band.steps[k] };
    }
}

/// A band's values or products for [`GROUP`] steps, a row a step.
type Tile<T> = [[T; BAND]; GROUP];

/// Runs laid out as a [`Geometry`] says, each running product where its
/// factor is, from `at` on in `output`; each starting at one, or, for the
/// pieces of one long run, as `pieces` says.
struct Strided<'a, I, T> {
    input: &'a [I],
    output: &'a Parts<T>,
    at: usize,
    geometry: Geometry,
    pieces: Option<Pieces<'a>>,
}

/// The pieces of one long run in a band: lane `k` the `first + k`-th, each
/// `piece` factors long and starting from `starts`; `exact` keeps each
/// piece's exact product once a later piece has needed it.
#[derive(Clone, Copy)]
struct Pieces<'a> {
    piece: usize,
    first: usize,
    starts: &'a [Running],
    exact: &'a [OnceLock<Wide>],
}

impl<I: Element, T: Output> Strided<'_, I, T> {
    /// Lane `k`'s running product before its first step.
    fn start(&self, k: usize) -> Running {
        self.pieces
            .map_or(Running::START, |pieces| pieces.starts[pieces.first + k])
    }

    /// The value nearest lane `k`'s exact running product at `step`, from
    /// its exact prefix, which is made when first asked for and kept.
    fn exactly(&self, k: usize, step: usize, prefix: &mut Option<Box<Prefix>>) -> T {
        let Some(pieces) = self.pieces else {
            let prefix = prefix.get_or_insert_with(|| Box::new(Prefix::empty()));
            return prefix.rounded(step, &|index| self.input[self.geometry.at(k, index)].wide());
        };
        let nth = pieces.first + k;
        let prefix = prefix.get_or_insert_with(|| {
            // Only asked for while every factor so far is finite and nonzero.
            let mut wide = Wide::one(START_WIDTH);
            for earlier in 0..nth {
                let exact = pieces.exact[earlier].get_or_init(|| {
                    let mut exact = Wide::one(START_WIDTH);
                    let span = earlier * pieces.piece..(earlier + 1) * pieces.piece;
                    for value in &self.input[span] {
                        exact.times_factor(value.wide());
                    }
                    exact
                });
                wide = wide.times(exact);
            }
            Box::new(Prefix::after(wide, nth * pieces.piece))
        });
        prefix.rounded(nth * pieces.piece + step, &|index| self.input[index].wide())
    }
}

/// `steps` factors into each of a band's `width` lanes, every running
/// product rounded once: by [`advance`] where it settles one, and otherwise
/// by the lane's [`Product`] -- a factor past [`SMALL`] or [`LARGE`] split
/// into it by [`step`] -- or its [`Running`] once it is special, and exactly
/// where neither can say.
#[inline(always)]
fn scan_band<I: Element, T: Output, const FMA: bool>(
    runs: &Strided<'_, I, T>,
    width: usize,
    steps: usize,
) {
    let mut slow = [Running::START; BAND];
    let mut band = Band::new::<T>(width, |k| runs.start(k), &mut slow);
    let mut values = [[1f64; BAND]; GROUP];
    let mut out = [[T::narrow(1.0); BAND]; GROUP];
    let mut settled = [0u64; BAND];
    let mut prefixes: Vec<Option<Box<Prefix>>> = Vec::new();
    for first in (0..steps).step_by(GROUP) {
        let count = GROUP.min(steps - first);
        runs.geometry.load(runs.input, first, count, &mut values);
        for (offset, (values, out)) in values.iter().zip(&mut out).enumerate().take(count) {
            advance::<T, FMA>(&mut band, values, width, out, &mut settled);
            let open = settled[..width]
                .iter()
                .fold(0, |open, &done| open | (done ^ 1));
            if open == 0 {
                continue;
            }
            if prefixes.is_empty() {
                prefixes.resize_with(BAND, || None);
            }
            let step = first + offset;
            for k in 0..width {
                if settled[k] == 0 {
                    let exactly = |prefix: &mut Option<Box<Prefix>>| runs.exactly(k, step, prefix);
                    out[k] = fix::<T, FMA>(&mut band, &mut slow[k], k, values[k])
                        .unwrap_or_else(|| exactly(&mut prefixes[k]));
                }
            }
        }
        band.renormalize::<T>();
        runs.geometry
            .store(runs.output, runs.at, first, count, &out);
    }
}

/// Lane `k`'s running product after `value`, where [`advance`] did not
/// settle it; `None` where only the exact product can say.
#[cold]
#[inline(never)]
fn fix<T: Output, const FMA: bool>(
    band: &mut Band,
    slow: &mut Running,
    k: usize,
    value: f64,
) -> Option<T> {
    if band.slow[k] == 0 {
        let magnitude = value.abs();
        if (SMALL..=LARGE).contains(&magnitude) {
            // Taken, but the bound or the range left it open: where the
            // product was scaled to a normal float64, the bound alone, which
            // only the exact product can settle.
            let whole = (band.hi[k] + band.lo[k]) * band.scale[k];
            if whole.abs() >= f64::MIN_POSITIVE && whole.is_finite() {
                return None;
            }
            return band.product(k).settled();
        }
        if magnitude > 0.0 && magnitude < f64::INFINITY {
            // Too small or too large to take unsplit.
            let mut product = band.product(k);
            let (mut sign, mut special) = (u64::from(product.negative), false);
            let (hi, lo, exponent) = (&mut product.hi, &mut product.lo, &mut product.exponent);
            step::<FMA>(hi, lo, exponent, &mut sign, &mut special, value);
            product.negative = sign == 1;
            product.steps += 1;
            band.set::<T>(k, product);
            return product.settled();
        }
        // Special: the lane goes on as a running product apart.
        let product = band.product(k);
        *slow = Running {
            product,
            sign: u64::from(product.negative),
            ..Running::START
        };
        band.slow[k] = 1;
    }
    slow.next::<T, FMA>(value)
}

/// [`scan_band`] in the widest build the CPU runs.
fn scan<I: Element, T: Output>(fma: bool, runs: &Strided<'_, I, T>, width: usize, steps: usize) {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn with_fma<I: Element, T: Output>(runs: &Strided<'_, I, T>, width: usize, steps: usize) {
        scan_band::<I, T, true>(runs, width, steps);
    }
    #[cfg(target_arch = "x86_64")]
    if fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { with_fma(runs, width, steps) };
    }
    let _ = fma;
    scan_band::<I, T, false>(runs, width, steps);
}

/// The running products along the middle axis of `input`, laid out
/// `[outer, len, inner]`, into `output`, laid out the same: each the value
/// nearest the exact product of the values so far along its slice.
///
/// Runs and columns are advanced side by side in bands, a few bands to a
/// task, each running product settled by its bound or formed exactly where
/// the bound cannot say. A long run, among too few to fill the pool, is cut
/// into pieces whose products, carried and multiplied in order, start each
/// piece's own running products, the pieces then advanced side by side.
pub(crate) fn running_products<I: Element, T: Output>(
    input: &[I],
    output: &mut [T],
    len: usize,
    inner: usize,
) {
    debug_assert_eq!(input.len(), output.len());
    let slab = len * inner;
    if slab == 0 {
        return;
    }
    let fma = fused();
    let slabs = input.len() / slab;
    let parts = Parts::of(output);
    if inner == 1 && len >= LONG_RUN && slabs < 4 * BAND {
        for nth in 0..slabs {
            long_run(&input[nth * len..(nth + 1) * len], &parts, nth * len, fma);
        }
        return;
    }
    let (bands, band) = Geometry::bands(slabs, len, inner);
    let per_task = ((1 << 14) / (BAND * len)).max(1);
    tasks(bands.div_ceil(per_task), input.len(), &|task| {
        for flat in task * per_task..((task + 1) * per_task).min(bands) {
            let geometry = band(flat);
            let runs = Strided {
                input,
                output: &parts,
                at: 0,
                geometry,
                pieces: None,
            };
            scan(fma, &runs, geometry.width, len);
        }
    });
}

/// [`running_products`] for one long run, whose output starts at `at`: its
/// pieces' carried products across the pool, multiplied in order to start
/// each piece, and the pieces advanced side by side from there, a band to a
/// task. A piece that needs the exact product of what comes before it
/// multiplies the exact products of the pieces before it, each formed once
/// and kept.
fn long_run<I: Element, T: Output>(run: &[I], output: &Parts<T>, at: usize, fma: bool) {
    let len = run.len();
    // Pieces enough for a few bands, and long enough to repay starting one.
    let piece = (len / (4 * BAND)).clamp(LONG_RUN / BAND, BLOCK);
    let whole = len / piece;
    let count = len.div_ceil(piece);
    let span = |nth: usize| nth * piece..((nth + 1) * piece).min(len);
    let products = crate::parallel::map_tasks(count, &|nth| block_product(&run[span(nth)]));
    let mut starts = Vec::with_capacity(count);
    let mut state = Running::START;
    for (nth, &(product, special)) in products.iter().enumerate() {
        starts.push(state);
        let block: Vec<f64> = if special {
            run[span(nth)].iter().map(|value| value.wide()).collect()
        } else {
            Vec::new()
        };
        state = if fma {
            state.then::<true>(&block, product, special)
        } else {
            state.then::<false>(&block, product, special)
        };
    }
    let exact: Vec<OnceLock<Wide>> = (0..count).map(|_| OnceLock::new()).collect();
    let bands = whole.div_ceil(BAND);
    tasks(count.div_ceil(BAND).max(bands), len, &|task| {
        let pieces = |first: usize, width: usize| Strided {
            input: run,
            output,
            at,
            geometry: Geometry {
                base: first * piece,
                lane: piece,
                step: 1,
                width,
                out: 0,
            },
            pieces: Some(Pieces {
                piece,
                first,
                starts: &starts,
                exact: &exact,
            }),
        };
        if task < bands {
            let first = task * BAND;
            let width = BAND.min(whole - first);
            scan(fma, &pieces(first, width), width, piece);
        }
        if task + 1 == count.div_ceil(BAND).max(bands) && whole < count {
            // The last piece, shorter than the rest, on its own.
            scan(fma, &pieces(whole, 1), 1, len - whole * piece);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A float64 from a seeded generator: a mantissa of 53 random bits, the
    /// exponent within `spread` of zero, either sign.
    fn draw(state: &mut u64, spread: i32) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = 1.0 + (*state >> 11) as f64 / (1u64 << 53) as f64;
        let exponent = (*state % (2 * spread as u64 + 1)) as i32 - spread;
        let sign = if *state & (1 << 7) == 0 { 1.0 } else { -1.0 };
        sign * mantissa * 2f64.powi(exponent)
    }

    /// The exact product, wide enough that nothing is ever cut.
    fn exact<T: Output>(values: &[f64]) -> T {
        *exact_running::<T>(values).last().expect("a factor")
    }

    /// Each exact running product, as [`exact`] makes the whole one.
    fn exact_running<T: Output>(values: &[f64]) -> Vec<T> {
        let mut wide = Wide::one(values.len() + 2);
        let mut out = Vec::with_capacity(values.len());
        for &value in values {
            wide.times_factor(value);
            assert_eq!(wide.cuts, 0);
            out.push(nearest(wide.negative, &wide.limbs, wide.exponent));
        }
        out
    }

    #[test]
    fn a_narrow_prefix_widens_until_it_can_say() {
        let mut state = 7;
        for trial in 0..400 {
            let len = 1 + trial % 40;
            let values: Vec<f64> = (0..len).map(|_| draw(&mut state, 30)).collect();
            let read = |index: usize| values[index];
            let mut prefix = Prefix::after(Wide::one(1), 0);
            let got: f64 = prefix.rounded(len - 1, &read);
            assert_eq!(got.to_bits(), exact::<f64>(&values).to_bits(), "{values:?}");
            let mut prefix = Prefix::after(Wide::one(1), 0);
            let got: f32 = prefix.rounded(len - 1, &read);
            assert_eq!(got.to_bits(), exact::<f32>(&values).to_bits(), "{values:?}");
        }
    }

    #[test]
    fn products_and_running_products_match_the_exact_ones() {
        let mut state = 11;
        for trial in 0..60 {
            let len = 1 + trial * 7;
            // Factors near one, and factors spread far enough that partial
            // products leave the range and a few are split.
            let spread = if trial % 2 == 0 { 0 } else { 80 };
            let values: Vec<f64> = (0..len).map(|_| draw(&mut state, spread)).collect();
            let got: f64 = product_to(&values);
            assert_eq!(got.to_bits(), exact::<f64>(&values).to_bits(), "{trial}");
            let mut running = vec![0f64; len];
            running_products(&values, &mut running, len, 1);
            let want = exact_running::<f64>(&values);
            for (index, (&got, &want)) in running.iter().zip(&want).enumerate() {
                assert_eq!(got.to_bits(), want.to_bits(), "{trial} at {index}");
            }
        }
    }

    #[test]
    fn the_builds_without_a_fused_multiply_add_agree() {
        let mut state = 13;
        for trial in 0..24 {
            let len = 8 + trial * 23;
            let spread = if trial % 2 == 0 { 0 } else { 70 };
            let values: Vec<f64> = (0..len).map(|_| draw(&mut state, spread)).collect();
            for (product, _) in [
                block_lanes::<f64, false>(&values),
                fast_lanes::<f64, false>(&values)
                    .unwrap_or_else(|| block_lanes::<f64, false>(&values)),
            ] {
                let got: f64 = product.settled().expect("settled");
                assert_eq!(got.to_bits(), exact::<f64>(&values).to_bits(), "{trial}");
            }
            // Two runs side by side, with no fused multiply-add.
            let mut out = vec![0f64; 2 * len];
            let twice: Vec<f64> = values.iter().chain(&values).copied().collect();
            let parts = Parts::of(&mut out);
            let geometry = Geometry {
                base: 0,
                lane: len,
                step: 1,
                width: 2,
                out: 0,
            };
            let runs = Strided {
                input: &twice,
                output: &parts,
                at: 0,
                geometry,
                pieces: None,
            };
            scan_band::<f64, f64, false>(&runs, 2, len);
            let want = exact_running::<f64>(&values);
            for (index, want) in want.iter().enumerate() {
                assert_eq!(out[index].to_bits(), want.to_bits(), "{trial} at {index}");
                assert_eq!(
                    out[len + index].to_bits(),
                    want.to_bits(),
                    "{trial} at {index}"
                );
            }
        }
    }

    #[test]
    fn a_tie_followed_by_ones_is_rounded_to_even() {
        // 4097 squared is 2^24 + 2^13 + 1, halfway between two float32s.
        let mut values = vec![1f32; 3000];
        values[0] = 4097.0;
        values[1] = 4097.0;
        let mut out = vec![0f32; values.len()];
        running_products(&values, &mut out, values.len(), 1);
        assert_eq!(out[0], 4097.0);
        assert!(out[1..].iter().all(|&value| value == 16_785_408.0));
    }
}
