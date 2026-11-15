// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Sums side by side, and sums of quotients, rounded once.
//!
//! A window's mean, a column of rows added into a row, the gradient an
//! average pool hands back: each is a sum of a few values at every one of
//! a run of places, the places independent of one another. [`Tallies`] keeps
//! a two-sum for each place -- the sum so far, the exact errors of its
//! additions added up beside it by two-sums of their own, and the magnitude
//! of what those drop, which is nothing while the errors fit a float64 --
//! so that a row of values is added into a row of places in one pass the
//! compiler takes in vector lanes, and each place is then rounded once from
//! its pair: exactly while nothing was dropped, a halfway point included,
//! and otherwise where the bound places it.
//!
//! Where it does not, the place's values are gathered and summed exactly:
//! [`sum_over`] for a sum over a count, and [`sum_of_quotients`] for values
//! each over a divisor of its own, which is one exact fraction over the
//! product of the divisors.

use super::{Carried, Element, LEAST, Output, Total, ratio_settled, two_sum};

/// How many places are taken at a time where they are not taken a row at a
/// time: a stretch the stack holds.
const STRETCH: usize = 64;

/// A two-sum for each of a run of places.
pub(crate) struct Tallies {
    sums: Vec<f64>,
    errors: Vec<f64>,
    /// The magnitudes of what adding up the errors dropped.
    dropped: Vec<f64>,
}

impl Tallies {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            sums: vec![0.0; len],
            errors: vec![0.0; len],
            dropped: vec![0.0; len],
        }
    }

    /// Nothing added yet, every sum the zero `from`: a negative zero, which
    /// any first value replaces, keeps a total of negative zeros one; a
    /// positive zero makes every total that is zero a positive one.
    pub(crate) fn clear(&mut self, from: f64) {
        self.sums.fill(from);
        self.errors.fill(0.0);
        self.dropped.fill(0.0);
    }

    /// `row` added into the places from `start`, one value each.
    pub(crate) fn add<T: Element>(&mut self, start: usize, row: &[T]) {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { self.add_avx2(start, row) };
        }
        self.add_body(start, row.len(), |j| row[j].wide())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn add_avx2<T: Element>(&mut self, start: usize, row: &[T]) {
        self.add_body(start, row.len(), |j| row[j].wide())
    }

    /// `value(j)` added into place `start + j`, for each of `len` places.
    #[inline(always)]
    fn add_body(&mut self, start: usize, len: usize, value: impl Fn(usize) -> f64) {
        let sums = &mut self.sums[start..start + len];
        let errors = &mut self.errors[start..start + len];
        let dropped = &mut self.dropped[start..start + len];
        for j in 0..len {
            let (total, rounding) = two_sum(sums[j], value(j));
            sums[j] = total;
            let (error, lost) = two_sum(errors[j], rounding);
            errors[j] = error;
            dropped[j] += lost.abs();
        }
    }

    /// Every `step`-th value of `row`, `count` of them, added into the
    /// places from `start`: brought side by side a stretch at a time, and
    /// each stretch added as a row is.
    pub(crate) fn add_spaced<T: Element>(
        &mut self,
        start: usize,
        row: &[T],
        step: usize,
        count: usize,
    ) {
        let mut close = [0.0f64; STRETCH];
        for first in (0..count).step_by(STRETCH) {
            let len = STRETCH.min(count - first);
            for (k, value) in close[..len].iter_mut().enumerate() {
                *value = row[(first + k) * step].wide();
            }
            self.add(start + first, &close[..len]);
        }
    }

    /// `values` added into every `step`-th place from `start`.
    pub(crate) fn add_spread<T: Element>(&mut self, start: usize, step: usize, values: &[T]) {
        for (i, &value) in values.iter().enumerate() {
            let at = start + i * step;
            let (total, rounding) = two_sum(self.sums[at], value.wide());
            self.sums[at] = total;
            let (error, lost) = two_sum(self.errors[at], rounding);
            self.errors[at] = error;
            self.dropped[at] += lost.abs();
        }
    }

    /// `value + rest` added into each of `len` places from `start`: a pair
    /// that stands for something within `missing` of it, `rest` far below
    /// `value`, as a quotient and its remainder's quotient are.
    pub(crate) fn add_pair(&mut self, start: usize, len: usize, pair: (f64, f64), missing: f64) {
        let sums = &mut self.sums[start..start + len];
        let errors = &mut self.errors[start..start + len];
        let dropped = &mut self.dropped[start..start + len];
        for j in 0..len {
            let (total, rounding) = two_sum(sums[j], pair.0);
            sums[j] = total;
            let (error, lost) = two_sum(errors[j], rounding);
            let (error, less) = two_sum(error, pair.1);
            errors[j] = error;
            dropped[j] += lost.abs() + less.abs() + missing;
        }
    }

    /// The pair place `at` has come to, and the bound on what the pair
    /// leaves out: twice what was dropped, which was itself added up in
    /// float64.
    fn carried(&self, at: usize) -> Carried {
        Carried {
            sum: self.sums[at],
            error: self.errors[at],
            slack: 2.0 * self.dropped[at],
        }
    }

    /// The value nearest the exact total of place `at`, or nearest it over
    /// `mean_over`, where the pair says which that is.
    pub(crate) fn settled<T: Output>(&self, at: usize, mean_over: Option<usize>) -> Option<T> {
        let carried = self.carried(at);
        let exact = carried.slack == 0.0;
        if exact && carried.error == 0.0 && mean_over.is_none() {
            // No addition rounded, so the sum is the exact one, and a zero
            // has the sign its additions gave it.
            return carried.sum.is_finite().then(|| T::narrow(carried.sum));
        }
        match mean_over {
            // A zero from a bound that is not one may have the sign of what
            // the bound covers rather than the total's, so it does not settle.
            None => carried
                .rounded()
                .filter(|value: &T| exact || value.widen() != 0.0),
            Some(count) => carried.divided(count),
        }
    }

    /// The total of each place from `start` into `totals`, and the places
    /// among them that are left to be summed exactly into `open`, numbered
    /// from `start`.
    pub(crate) fn totals_into<T: Output>(
        &self,
        start: usize,
        totals: &mut [T],
        open: &mut Vec<usize>,
    ) {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { self.totals_avx2(start, totals, open) };
        }
        self.totals_body(start, totals, open)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn totals_avx2<T: Output>(&self, start: usize, totals: &mut [T], open: &mut Vec<usize>) {
        self.totals_body(start, totals, open)
    }

    #[inline(always)]
    fn totals_body<T: Output>(&self, start: usize, totals: &mut [T], open: &mut Vec<usize>) {
        let len = totals.len();
        let sums = &self.sums[start..start + len];
        let errors = &self.errors[start..start + len];
        let dropped = &self.dropped[start..start + len];
        let mut settled = [true; STRETCH];
        for (stretch, totals) in totals.chunks_mut(STRETCH).enumerate() {
            let first = stretch * STRETCH;
            let mut every = true;
            for (k, total) in totals.iter_mut().enumerate() {
                let at = first + k;
                let slack = 2.0 * dropped[at];
                let (value, rest) = two_sum(sums[at], errors[at]);
                let (rounded, clear) = T::settle_lanes(value, rest, slack);
                *total = rounded;
                // A zero is the total's own only where nothing was dropped.
                settled[k] = clear & ((slack == 0.0) | (rounded.widen() != 0.0));
                every &= settled[k];
            }
            if every {
                continue;
            }
            for (k, total) in totals.iter_mut().enumerate() {
                if settled[k] {
                    continue;
                }
                // An exact pair is rounded once wherever it lies, a halfway
                // point included.
                match self.settled::<T>(start + first + k, None) {
                    Some(value) => *total = value,
                    None => open.push(first + k),
                }
            }
        }
    }

    /// The mean of each place from `start` over its count in `counts`, into
    /// `means`, and the places among them that are left to be summed
    /// exactly into `open`, numbered from `start`. A total that is an exact
    /// zero makes a positive one.
    pub(crate) fn means_into<T: Output>(
        &self,
        start: usize,
        counts: &[f64],
        means: &mut [T],
        open: &mut Vec<usize>,
    ) {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { self.means_avx2(start, counts, means, open) };
        }
        self.means_body::<T, false>(start, counts, means, open)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn means_avx2<T: Output>(
        &self,
        start: usize,
        counts: &[f64],
        means: &mut [T],
        open: &mut Vec<usize>,
    ) {
        self.means_body::<T, true>(start, counts, means, open)
    }

    #[inline(always)]
    fn means_body<T: Output, const FMA: bool>(
        &self,
        start: usize,
        counts: &[f64],
        means: &mut [T],
        open: &mut Vec<usize>,
    ) {
        let len = means.len();
        let sums = &self.sums[start..start + len];
        let errors = &self.errors[start..start + len];
        let dropped = &self.dropped[start..start + len];
        let counts = &counts[..len];
        let mut settled = [true; STRETCH];
        for (stretch, means) in means.chunks_mut(STRETCH).enumerate() {
            let first = stretch * STRETCH;
            let mut every = true;
            for (k, mean) in means.iter_mut().enumerate() {
                let at = first + k;
                let (sum, error, count) = (sums[at], errors[at], counts[at]);
                let slack = 2.0 * dropped[at];
                // The values cancel exactly, or are all zeros, or are none.
                let nothing = (sum == 0.0) & (error == 0.0) & (slack == 0.0);
                // A power of two divides an exact pair exactly: the pair
                // rounded once and scaled, a halfway point included -- and
                // few values over a small count land on one often.
                let (whole, _) = two_sum(sum, error);
                let scaled = whole / count;
                let halving = count.to_bits() & ((1 << 52) - 1) == 0;
                let within = (scaled.abs() >= 2f64.powi(-900)) & (scaled.abs() <= 2f64.powi(900));
                let direct = (slack == 0.0) & T::exact_lane(error) & halving & within;
                let (value, clear) = Carried { sum, error, slack }.divided_lane::<T, FMA>(count);
                *mean = if nothing {
                    T::narrow(0.0)
                } else if direct {
                    T::narrow(scaled)
                } else {
                    value
                };
                settled[k] = nothing | direct | clear;
                every &= settled[k];
            }
            if every {
                continue;
            }
            for (k, mean) in means.iter_mut().enumerate() {
                if settled[k] {
                    continue;
                }
                // The quotient with its bound once more, and where the pair
                // is exact, the side of a halfway point it lies on.
                let at = first + k;
                match ratio_settled::<T, FMA>(self.carried(start + at), (counts[at], 0.0)) {
                    Some(value) => *mean = value,
                    None => open.push(at),
                }
            }
        }
    }
}

/// The value nearest the exact sum of `values` over `count`; an infinity or
/// a NaN where the values make one.
#[cold]
#[inline(never)]
pub(crate) fn sum_over<I: Element, T: Output>(values: &[I], count: u64) -> T {
    match Total::of(values) {
        Err(special) => T::narrow(special),
        Ok(total) => total.divided(count),
    }
}

/// `limbs *= factor`, a magnitude in little-endian 64-bit limbs.
fn times(limbs: &mut Vec<u64>, factor: u64) {
    let mut carry = 0u128;
    for limb in limbs.iter_mut() {
        let product = u128::from(*limb) * u128::from(factor) + carry;
        *limb = product as u64;
        carry = product >> 64;
    }
    if carry != 0 {
        limbs.push(carry as u64);
    }
}

/// `sum += limbs`, two magnitudes in little-endian 64-bit limbs.
fn add_limbs(sum: &mut Vec<u64>, limbs: &[u64]) {
    if sum.len() < limbs.len() {
        sum.resize(limbs.len(), 0);
    }
    let mut carry = false;
    for (index, place) in sum.iter_mut().enumerate() {
        let other = limbs.get(index).copied().unwrap_or(0);
        let (value, first) = place.overflowing_add(other);
        let (value, second) = value.overflowing_add(u64::from(carry));
        *place = value;
        carry = first || second;
    }
    if carry {
        sum.push(1);
    }
}

/// The value nearest the exact sum of each of `terms`' values over its own
/// divisor, none of them zero; what the quotients make added one by one
/// where a value is not finite.
///
/// The values sharing a divisor are summed exactly, and the sums brought
/// over the product of the divisors: one whole fraction, divided in limbs.
#[cold]
#[inline(never)]
pub(crate) fn sum_of_quotients<T: Output>(terms: &mut [(f64, u64)]) -> T {
    if terms.iter().any(|&(value, _)| !value.is_finite()) {
        let quotients = terms.iter().map(|&(value, divisor)| value / divisor as f64);
        return T::narrow(quotients.sum::<f64>());
    }
    terms.sort_unstable_by_key(|&(_, divisor)| divisor);
    let groups: Vec<(u64, Total)> = terms
        .chunk_by(|a, b| a.1 == b.1)
        .map(|group| {
            let values = group.iter().map(|&(value, _)| value);
            (group[0].1, Total::of_wide(values).expect("finite values"))
        })
        .collect();
    if let [(divisor, total)] = &groups[..] {
        return total.clone().divided(*divisor);
    }
    // Each group's sum times every other group's divisor, over them all.
    let (mut positive, mut negative) = (Vec::new(), Vec::new());
    for (nth, (_, total)) in groups.iter().enumerate() {
        let (mut plus, mut minus) = (total.positive.clone(), total.negative.clone());
        for (other, &(divisor, _)) in groups.iter().enumerate() {
            if other != nth {
                times(&mut plus, divisor);
                times(&mut minus, divisor);
            }
        }
        add_limbs(&mut positive, &plus);
        add_limbs(&mut negative, &minus);
    }
    let limbs = positive.len().max(negative.len());
    positive.resize(limbs, 0);
    negative.resize(limbs, 0);
    let mut denominator = vec![1u64];
    for &(divisor, _) in &groups {
        times(&mut denominator, divisor);
    }
    let numerator = Total {
        positive,
        negative,
        least: LEAST,
    };
    numerator.quotient(Total::of_magnitude(denominator, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::arithmetic::exact_sum::{mean_to, sum_to};

    fn values(seed: u64, len: usize) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                unit * 2f64.powi((state % 30) as i32 - 15)
            })
            .collect()
    }

    #[test]
    fn places_settle_to_their_exact_totals() {
        // Twelve rows of forty added into forty places, with a strided row,
        // a spread one and a constant among them.
        let rows: Vec<Vec<f32>> = (0..12)
            .map(|row| values(61 + row, 40).into_iter().map(|v| v as f32).collect())
            .collect();
        let mut tallies = Tallies::new(40);
        tallies.clear(-0.0);
        let mut gathered: Vec<Vec<f32>> = vec![Vec::new(); 40];
        for row in &rows[..9] {
            tallies.add(0, row);
            for (place, &value) in gathered.iter_mut().zip(row) {
                place.push(value);
            }
        }
        tallies.add_spaced(5, &rows[9], 3, 10);
        for i in 0..10 {
            gathered[5 + i].push(rows[9][3 * i]);
        }
        tallies.add_spread(2, 4, &rows[10][..8]);
        for (i, &value) in rows[10][..8].iter().enumerate() {
            gathered[2 + 4 * i].push(value);
        }
        tallies.add_pair(30, 10, (f64::from(rows[11][0]), 0.0), 0.0);
        for place in &mut gathered[30..] {
            place.push(rows[11][0]);
        }
        for (at, kept) in gathered.iter().enumerate() {
            let sum: f32 = tallies.settled(at, None).unwrap_or_else(|| sum_to(kept));
            assert_eq!(sum, sum_to::<f32, f32>(kept), "sum of place {at}");
            let mean: f32 = tallies
                .settled(at, Some(kept.len()))
                .unwrap_or_else(|| mean_to(kept));
            assert_eq!(mean, mean_to::<f32, f32>(kept), "mean of place {at}");
        }
    }

    #[test]
    fn a_sum_over_a_count_is_the_exact_quotient() {
        let kept: Vec<f64> = values(71, 9);
        // Over its own count it is the mean; over another, that mean scaled.
        assert_eq!(sum_over::<f64, f64>(&kept, 9), mean_to::<f64, f64>(&kept));
        assert_eq!(sum_over::<f32, f32>(&[1.0, 2.0], 3), 1.0);
        assert_eq!(sum_over::<f32, f32>(&[1.0], 3), 1.0f32 / 3.0);
        assert_eq!(
            sum_over::<f64, f64>(&[f64::INFINITY, 1.0], 4),
            f64::INFINITY
        );
        assert!(sum_over::<f64, f64>(&[f64::NAN, 1.0], 4).is_nan());
    }

    #[test]
    fn quotients_sum_as_one_fraction() {
        // 1/3 + 1/6 is a half exactly, where each quotient rounded first.
        assert_eq!(sum_of_quotients::<f64>(&mut [(1.0, 3), (1.0, 6)]), 0.5);
        assert_eq!(sum_of_quotients::<f32>(&mut [(1.0, 3), (1.0, 6)]), 0.5);
        // 1/3 - 2/6 cancels to a positive zero.
        let zero = sum_of_quotients::<f64>(&mut [(1.0, 3), (-2.0, 6)]);
        assert_eq!(zero.to_bits(), 0.0f64.to_bits());
        // One divisor is a sum over a count.
        let kept = values(73, 7);
        let mut terms: Vec<(f64, u64)> = kept.iter().map(|&value| (value, 7)).collect();
        assert_eq!(
            sum_of_quotients::<f64>(&mut terms),
            mean_to::<f64, f64>(&kept)
        );
        // Divisors whose product is far past 2^64, against the same sum
        // taken in two halves that are each exact.
        let divisors = [
            (1u64 << 40) + 1,
            (1u64 << 41) + 3,
            (1u64 << 42) + 5,
            (1u64 << 43) + 7,
        ];
        let mut terms: Vec<(f64, u64)> = divisors
            .iter()
            .flat_map(|&d| [(d as f64, d), (-(d as f64), d), (3.0 * d as f64, d)])
            .collect();
        assert_eq!(sum_of_quotients::<f64>(&mut terms), 12.0);
        // Special values make what the quotients would added one by one.
        assert_eq!(
            sum_of_quotients::<f32>(&mut [(f64::INFINITY, 2), (1.0, 3)]),
            f32::INFINITY
        );
        assert!(
            sum_of_quotients::<f32>(&mut [(f64::INFINITY, 2), (f64::NEG_INFINITY, 3)]).is_nan()
        );
    }
}
