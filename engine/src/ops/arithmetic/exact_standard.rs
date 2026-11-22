// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Standardized float64 values rounded once.
//!
//! A normalization layer takes each value of a slice less the slice's mean,
//! over the root of the slice's variance plus a floor `eps` -- or, not
//! centred, over the root of its mean square plus the floor -- and may then
//! scale it by a weight and shift it by a bias. Computed in float64 step by
//! step, most outputs of a float64 layer came out an ulp or more from the
//! exact value. Here each is the float64 nearest it.
//!
//! The mean, the variance and the inverse root are carried as pairs of
//! float64s with bounds on how far each is from the exact value, so that
//! each output is a pair and a bound, and settled where the bound places it.
//! An output the bound leaves open, and every output of a slice whose values
//! lie too far out for the pairs to be bounded, is decided in whole numbers:
//! with `n` values, sum `S` and sum of squares `Q`, the standardized value is
//! the root of `U / D` with its sign, where centred `U = (n x - S)^2` and
//! `D = n Q - S^2 + eps n^2`, and not centred `U = n x^2` and
//! `D = Q + eps n`. Without an affine part that root is rounded directly;
//! with one, `w y + b` is placed against the halfway points around a guess
//! by comparing squares.

use std::cmp::Ordering;

use super::{Carried, Element, Output, Total, at_least, shifted, subtract, two_sum};

#[path = "exact_standard_slope.rs"]
mod slope;
pub(crate) use self::slope::standardized_slopes;

/// A slice's statistics as pairs of float64s, with bounds.
pub(crate) struct Standard {
    /// The mean, within `centre_bound` of `centre + centre_rest`; zero when
    /// not centred.
    centre: f64,
    centre_rest: f64,
    centre_bound: f64,
    /// The inverse root, within `scale_relative` of `scale + scale_rest`
    /// relative to it.
    scale: f64,
    scale_rest: f64,
    scale_relative: f64,
}

/// Whether a value lies far enough inside the range that its squares,
/// products and their errors stay ordinary float64s.
#[inline(always)]
fn comfortable(value: f64) -> bool {
    let size = value.abs();
    (value == 0.0) | ((size >= 2f64.powi(-300)) & (size <= 2f64.powi(300)))
}

/// `x · y` and its exact error, by a fused multiply-add where `FMA`.
#[inline(always)]
fn split<const FMA: bool>(x: f64, y: f64) -> (f64, f64) {
    let product = x * y;
    let error = if FMA {
        x.mul_add(y, -product)
    } else {
        super::product_error(x, y, product)
    };
    (product, error)
}

/// A bound grown a little past itself, so that the roundings of its own
/// arithmetic cannot leave it short.
#[inline(always)]
fn generous(bound: f64) -> f64 {
    bound * (1.0 + 2f64.powi(-40))
}

impl Standard {
    /// The statistics of `values`, all finite, or None where a value or a
    /// statistic lies too far out for the pairs to be bounded.
    ///
    /// Inlined, so that its products are fused where the caller is built
    /// with fused multiply-adds: a call of its own is built without them,
    /// and each of its `mul_add`s was a call into a software one.
    #[inline(always)]
    pub(crate) fn of<const FMA: bool>(values: &[f64], eps: f64, centred: bool) -> Option<Self> {
        let count = values.len();
        if count == 0 || count >= 1 << 40 || !comfortable(eps) {
            return None;
        }
        if !values.iter().all(|&value| comfortable(value)) {
            return None;
        }
        let n = count as f64;
        let (centre, centre_rest, centre_bound) = if centred {
            let sum = <f64 as Element>::block(values);
            let (value, rest, bound, ordinary) = sum.quotient_parts::<FMA>(n);
            if !ordinary && value != 0.0 {
                return None;
            }
            (value, rest, bound)
        } else {
            (0.0, 0.0, 0.0)
        };
        // The sum of the squared deviations: each deviation a pair within
        // `error` of the exact one, its square split, the large parts added
        // by two-sums and the small ones in float64 with their magnitudes
        // kept for the bound.
        let (mut large, mut small, mut magnitude, mut drift) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &x in values {
            let (high, low, error) = deviation(x, centre, centre_rest, centre_bound);
            let (square, square_error) = split::<FMA>(high, high);
            let tail = square_error + (2.0 * high + low) * low;
            let (sum, carry) = two_sum(large, square);
            large = sum;
            small += carry + tail;
            magnitude += carry.abs() + tail.abs() + square_error.abs();
            // A deviation within `error` of the pair moves its square by at
            // most `error · (2 |pair| + error)`.
            drift += error * (2.0 * (high.abs() + low.abs()) + error);
        }
        // Each float64 sum of `n` terms that are not negative falls short of
        // the exact sum by at most `n + 4` units of its last place in all,
        // the terms' own roundings included: the roundings of the small
        // parts are bounded by twice that, and the drift grown by it.
        let slack = generous(
            magnitude * (n + 4.0) * 2f64.powi(-52) + drift * (1.0 + (n + 4.0) * 2f64.powi(-52)),
        );
        let total = Carried {
            sum: large,
            error: small,
            slack,
        };
        // Over the count, and the floor added.
        let (mean, mean_rest, mean_bound, ordinary) = total.quotient_parts::<FMA>(n);
        if !ordinary && mean != 0.0 {
            return None;
        }
        let (high, carry) = two_sum(mean, eps);
        let low = carry + mean_rest;
        let bound = generous(mean_bound + low.abs() * 2f64.powi(-52));
        if !(2f64.powi(-600)..=2f64.powi(600)).contains(&high) || bound >= high * 2f64.powi(-60) {
            return None;
        }
        // The inverse root by one Newton step from the float64 one: the
        // step's own error is below 2^-100 of it, the pair's error and the
        // bound on the variance move it by half their part of it.
        let first = 1.0 / high.sqrt();
        let (square, square_error) = split::<FMA>(first, first);
        let (product, product_error) = split::<FMA>(high, square);
        let residual = (product - 1.0) + (product_error + high * square_error + low * square);
        let correction = -0.5 * first * residual;
        let (scale, scale_rest) = two_sum(first, correction);
        let scale_relative = generous(2f64.powi(-96) + (bound / high) * 0.5);
        Some(Self {
            centre,
            centre_rest,
            centre_bound,
            scale,
            scale_rest,
            scale_relative,
        })
    }

    /// The float64 nearest the standardized `x` -- times `weight` plus
    /// `bias` with `affine` -- where the bound settles it, as the pair's
    /// float64 nearest and whether it did.
    #[inline(always)]
    pub(crate) fn settled<const FMA: bool>(
        &self,
        x: f64,
        affine: Option<(f64, f64)>,
    ) -> (f64, bool) {
        let (high, low, error) = deviation(x, self.centre, self.centre_rest, self.centre_bound);
        let (product, product_error) = split::<FMA>(high, self.scale);
        let parts = [
            high * self.scale_rest,
            low * self.scale,
            low * self.scale_rest,
        ];
        let cross = parts[0] + parts[1] + parts[2];
        let (value, rest) = two_sum(product, product_error + cross);
        // The pair is zero only where the deviation from the pair's mean
        // is, the values lying well inside the range; and a zero is the
        // formula's: a zero input keeps its sign, as `x - 0` and `x · r` do,
        // and a value equal to a mean that is not zero gives a positive zero.
        let value = if (value == 0.0) & (x == 0.0) {
            x
        } else {
            value
        };
        let size = value.abs() + rest.abs();
        // The cross terms' roundings are of each part, however they cancel.
        let spread = parts.iter().map(|part| part.abs()).sum::<f64>();
        let bound = generous(
            size * self.scale_relative
                + error * (self.scale.abs() + self.scale_rest.abs()) * (1.0 + self.scale_relative)
                + (2.0 * spread + (product_error + cross).abs()) * 2f64.powi(-52),
        );
        // Whether the standardized value's sign is certain: the pair's,
        // where the bound cannot carry it across zero, or an exact zero's.
        let signed = (value.abs() > rest.abs() + bound) | (bound == 0.0);
        match affine {
            None => {
                let (settled, clear) = f64::settle_lanes(value, rest, bound);
                (settled, clear & comfortable(value))
            }
            Some((weight, bias)) => {
                let (scaled, scaled_error) = split::<FMA>(value, weight);
                let carried = rest * weight;
                let (head, head_error) = two_sum(scaled, bias);
                let tail = head_error + (scaled_error + carried);
                let (sum, sum_rest) = two_sum(head, tail);
                let bound = generous(
                    bound * weight.abs()
                        + (carried.abs() + (scaled_error + carried).abs() + tail.abs())
                            * 2f64.powi(-52),
                );
                // Where the product is a zero, `head` is `w y + b` exactly,
                // signed as the formula signs it given the sign of `y`, which
                // adding the zero `tail` could turn.
                let direct = (value == 0.0) | (weight == 0.0);
                let sum = if direct { head } else { sum };
                let (settled, clear) = f64::settle_lanes(sum, sum_rest, bound);
                let ordinary = comfortable(value)
                    & comfortable(sum)
                    & comfortable(weight)
                    & comfortable(bias)
                    & (!direct | signed);
                (settled, clear & ordinary)
            }
        }
    }
}

/// `x` less the mean `centre + rest`, which is within `bound` of the exact
/// one: a pair and how far the exact deviation can be from it.
#[inline(always)]
fn deviation(x: f64, centre: f64, rest: f64, bound: f64) -> (f64, f64, f64) {
    let (head, head_error) = two_sum(x, -centre);
    let tail = head_error - rest;
    let (high, low) = two_sum(head, tail);
    (high, low, tail.abs() * 2f64.powi(-52) + bound)
}

/// A whole number of `2^least`, with its sign.
#[derive(Clone)]
struct Whole {
    negative: bool,
    limbs: Vec<u64>,
    least: i32,
}

impl Whole {
    /// A total's value, its zero limbs at either end trimmed away.
    fn of(total: Total) -> Self {
        let (negative, limbs, least) = total.difference();
        Self {
            negative,
            limbs,
            least,
        }
        .trimmed()
    }

    fn trimmed(mut self) -> Self {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
        let low = self.limbs.iter().take_while(|&&limb| limb == 0).count();
        if low > 0 {
            self.limbs.drain(..low);
            self.least += 64 * low as i32;
        }
        if self.limbs.is_empty() {
            self.negative = false;
        }
        self
    }

    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn negated(mut self) -> Self {
        if !self.is_zero() {
            self.negative = !self.negative;
        }
        self
    }

    fn times(&self, other: &Self) -> Self {
        let mut limbs = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &x) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in other.limbs.iter().enumerate() {
                let sum = u128::from(x) * u128::from(y) + u128::from(limbs[i + j]) + carry;
                limbs[i + j] = sum as u64;
                carry = sum >> 64;
            }
            limbs[i + other.limbs.len()] = carry as u64;
        }
        Self {
            negative: self.negative != other.negative,
            limbs,
            least: self.least + other.least,
        }
        .trimmed()
    }

    fn plus(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let least = self.least.min(other.least);
        let mut a = shifted(&self.limbs, (self.least - least) as usize);
        let mut b = shifted(&other.limbs, (other.least - least) as usize);
        let negative = if self.negative == other.negative {
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            a.push(0);
            let mut carry = false;
            for (index, limb) in a.iter_mut().enumerate() {
                let other = b.get(index).copied().unwrap_or(0);
                let (value, first) = limb.overflowing_add(other);
                let (value, second) = value.overflowing_add(u64::from(carry));
                *limb = value;
                carry = first || second;
            }
            self.negative
        } else if at_least(&a, &b) {
            subtract(&mut a, &b);
            self.negative
        } else {
            subtract(&mut b, &a);
            a = b;
            other.negative
        };
        Self {
            negative,
            limbs: a,
            least,
        }
        .trimmed()
    }

    fn sign(&self) -> Ordering {
        match (self.is_zero(), self.negative) {
            (true, _) => Ordering::Equal,
            (false, true) => Ordering::Less,
            (false, false) => Ordering::Greater,
        }
    }

    fn total(self) -> Total {
        let empty = vec![0u64; self.limbs.len()];
        if self.negative {
            Total {
                positive: empty,
                negative: self.limbs,
                least: self.least,
            }
        } else {
            Total {
                positive: self.limbs,
                negative: empty,
                least: self.least,
            }
        }
    }

    /// A float64's exact value.
    fn float(value: f64) -> Self {
        Self::of(Total::of_wide(std::iter::once(value)).expect("finite"))
    }
}

/// A slice's statistics in whole numbers, for its outputs the bound leaves
/// open.
pub(crate) struct Exactly {
    count: f64,
    centred: bool,
    sum: Whole,
    divisor: Whole,
}

impl Exactly {
    /// The statistics of `values`, all finite.
    pub(crate) fn of(values: &[f64], eps: f64, centred: bool) -> Self {
        let n = values.len() as f64;
        let sum = Whole::of(Total::of_wide(values.iter().copied()).expect("finite"));
        let factor = if centred { n } else { 1.0 };
        let mut terms: Vec<(f64, f64, f64)> =
            values.iter().map(|&value| (factor, value, value)).collect();
        terms.push((eps, n, factor));
        let mut divisor = Whole::of(Total::of_terms(&terms));
        if centred {
            divisor = divisor.plus(&sum.times(&sum).negated());
        }
        Self {
            count: n,
            centred,
            sum,
            divisor,
        }
    }

    /// The standardized `x` as the root of `U / D` and its sign: `U` and
    /// whether the value is negative.
    fn numerator(&self, x: f64) -> (Whole, bool) {
        if self.centred {
            let scaled = Whole::of(Total::of_terms(&[(self.count, x, 1.0)]));
            let deviation = scaled.plus(&self.sum.clone().negated());
            let negative = deviation.negative;
            (deviation.times(&deviation), negative)
        } else {
            (
                Whole::of(Total::of_terms(&[(self.count, x, x)])),
                x.is_sign_negative() && x != 0.0,
            )
        }
    }

    /// The float64 nearest the standardized `x`, times `weight` plus `bias`
    /// with `affine`; `guess` is a float64 near it.
    pub(crate) fn value(&self, x: f64, affine: Option<(f64, f64)>, guess: f64) -> f64 {
        let (square, negative) = self.numerator(x);
        // The standardized value: the root of `U / D` with its sign, and
        // where `U` is zero the formula's zero -- a zero input keeps its
        // sign, a value equal to a mean that is not zero gives a positive
        // zero.
        let standardized = |square: Whole| -> f64 {
            if square.is_zero() && !self.divisor.is_zero() {
                return if x == 0.0 { x } else { 0.0 };
            }
            let magnitude = square
                .total()
                .quotient_root::<f64>(self.divisor.clone().total());
            if negative { -magnitude } else { magnitude }
        };
        let Some((weight, bias)) = affine else {
            return standardized(square);
        };
        // `w y + b` with `y = s sqrt(U / D)`: the gain `g = w s` and the
        // value `g sqrt(U / D) + b`.
        let gain = if negative { -weight } else { weight };
        if gain == 0.0 || square.is_zero() || self.divisor.is_zero() {
            // A zero product, whose sum with the bias float64 holds exactly
            // and signs as the formula does, or a value that is not finite.
            return standardized(square) * weight + bias;
        }
        let weighed = square.times(&Whole::float(gain).times(&Whole::float(gain)));
        let bias_exact = Whole::float(bias);
        // How `g sqrt(U / D) + b` compares with `m`, a float64 or a halfway
        // point between two.
        let compare = |m: &Whole| -> Ordering {
            let v = m.plus(&bias_exact.clone().negated());
            let positive = gain > 0.0;
            match (positive, v.sign()) {
                (true, Ordering::Less | Ordering::Equal) => Ordering::Greater,
                (false, Ordering::Greater | Ordering::Equal) => Ordering::Less,
                _ => {
                    let order = weighed
                        .plus(&v.times(&v).times(&self.divisor).negated())
                        .sign();
                    if positive { order } else { order.reverse() }
                }
            }
        };
        // The float64s in order, as whole numbers: both zeros at zero, the
        // negatives below it.
        let key = |value: f64| -> i64 {
            let bits = value.to_bits() as i64;
            if bits < 0 { -(bits & i64::MAX) } else { bits }
        };
        let float = |key: i64| -> f64 {
            if key < 0 {
                -f64::from_bits(key.unsigned_abs())
            } else {
                f64::from_bits(key as u64)
            }
        };
        let top = key(f64::MAX);
        // The halfway point between the float64 at `k` and the next one up,
        // and whether the value lies at or below it.
        let halfway = |k: i64| -> Whole {
            let low = float(k);
            let mut half = if k == top {
                // Past the largest float64, half its gap: the overflow line.
                Whole::float(low).plus(&Whole::float(2f64.powi(970)))
            } else {
                Whole::float(low).plus(&Whole::float(float(k + 1)))
            };
            half.least -= 1;
            half
        };
        let not_above = |k: i64| compare(&halfway(k)) != Ordering::Greater;
        // The least `k` whose halfway point up is not below the value: from
        // the guess, by steps that double until the value is passed, then by
        // halving.
        let start = key(if guess.is_finite() { guess } else { 0.0 }).clamp(-top, top);
        let (mut low, mut high) = if not_above(start) {
            let mut step = 1i64;
            let mut high = start;
            loop {
                let next = high.saturating_sub(step).max(-top - 1);
                if next == -top - 1 || !not_above(next) {
                    break (next, high);
                }
                high = next;
                step = step.saturating_mul(2);
            }
        } else {
            let mut step = 1i64;
            let mut low = start;
            loop {
                let next = low.saturating_add(step).min(top);
                if not_above(next) {
                    break (low, next);
                }
                if next == top {
                    // Past the overflow line: the value rounds to infinity.
                    return f64::INFINITY;
                }
                low = next;
                step = step.saturating_mul(2);
            }
        };
        // `low` is below the answer (or past the bottom), `high` is it or above.
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if not_above(middle) {
                high = middle;
            } else {
                low = middle;
            }
        }
        if high == -top {
            // At or past the overflow line below the lowest float64, a tie
            // going out as it does at the top.
            let line = Whole::float(-f64::MAX).plus(&Whole::float(-(2f64.powi(970))));
            if compare(&line) != Ordering::Greater {
                return f64::NEG_INFINITY;
            }
        }
        let mut answer = float(high);
        if answer.to_bits() & 1 == 1 && compare(&halfway(high)) == Ordering::Equal {
            // A tie with the next one up goes to the even of the two.
            answer = if high == top {
                f64::INFINITY
            } else {
                float(high + 1)
            };
        }
        if answer == 0.0 && compare(&Whole::float(0.0)) == Ordering::Less {
            // A negative value too small for any float64 rounds to the zero
            // below; an exact zero, the product cancelling the bias, is the
            // positive one.
            answer = -0.0;
        }
        answer
    }
}

/// Each row of `input`, `norm` wide, standardized into `out` -- times
/// `weight` plus `bias` where given -- each the float64 nearest the exact
/// value; `normalized`, where given, takes the standardized values alone.
/// A row holding a value that is not finite is left as `out` has it, and so
/// is every row under an `eps` that is negative or not finite, which the
/// layers refuse.
#[allow(clippy::too_many_arguments)]
pub(crate) fn standardized_rows(
    input: &[f64],
    norm: usize,
    weight: Option<&[f64]>,
    bias: Option<&[f64]>,
    eps: f64,
    centred: bool,
    out: &mut [f64],
    normalized: Option<&mut [f64]>,
) {
    if norm == 0 || !(eps.is_finite() && eps >= 0.0) {
        return;
    }
    let shape = Row {
        norm,
        weight,
        bias,
        constant: None,
        eps,
        centred,
    };
    let per_task = crate::ops::map::outputs_per_task(norm * 8).max(1);
    let run = |first: usize, out: &mut [f64], normalized: Option<&mut [f64]>| {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { rows_fma(input, &shape, first, out, normalized) };
        }
        rows::<false>(input, &shape, first, out, normalized);
    };
    match normalized {
        Some(normalized) => crate::ops::map::par_out_chunks2(
            out,
            normalized,
            per_task * norm,
            &|start, out, normalized| run(start / norm, out, Some(normalized)),
        ),
        None => crate::ops::map::par_out_chunks(out, per_task * norm, &|start, out| {
            run(start / norm, out, None)
        }),
    }
}

/// What every row of a [`standardized_rows`] call shares.
struct Row<'a> {
    norm: usize,
    weight: Option<&'a [f64]>,
    bias: Option<&'a [f64]>,
    /// One weight and bias for every value, in place of the two slices.
    constant: Option<(f64, f64)>,
    eps: f64,
    centred: bool,
}

/// Each of `values` less their mean, over the root of their variance plus
/// `eps`, times `weight` plus `bias`, into `out`: each the float64 nearest
/// its exact value. Values that are not all finite, or an `eps` that is
/// negative or not finite, leave `out` as it is.
pub(crate) fn standardized_slice(
    values: &[f64],
    eps: f64,
    weight: f64,
    bias: f64,
    out: &mut [f64],
) {
    // A weight of one and a bias of negative zero change no value at all;
    // a bias of positive zero still turns a negative zero positive.
    let identity = weight == 1.0 && bias == 0.0 && bias.is_sign_negative();
    let shape = Row {
        norm: values.len(),
        weight: None,
        bias: None,
        constant: (!identity).then_some((weight, bias)),
        eps,
        centred: true,
    };
    if values.is_empty() || !(eps.is_finite() && eps >= 0.0) {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { rows_fma(values, &shape, 0, out, None) };
    }
    rows::<false>(values, &shape, 0, out, None);
}

/// [`rows`] built with AVX2 and fused multiply-adds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn rows_fma(
    input: &[f64],
    shape: &Row<'_>,
    first: usize,
    out: &mut [f64],
    normalized: Option<&mut [f64]>,
) {
    rows::<true>(input, shape, first, out, normalized);
}

/// The rows from `first` that `out` holds, `normalized` beside it.
#[inline(always)]
fn rows<const FMA: bool>(
    input: &[f64],
    shape: &Row<'_>,
    first: usize,
    out: &mut [f64],
    mut normalized: Option<&mut [f64]>,
) {
    let norm = shape.norm;
    for (offset, out) in out.chunks_mut(norm).enumerate() {
        let row = &input[(first + offset) * norm..][..norm];
        let normalized = normalized
            .as_deref_mut()
            .map(|normalized| &mut normalized[offset * norm..][..norm]);
        one_row::<FMA>(row, shape, out, normalized);
    }
}

impl Row<'_> {
    /// Whether the outputs are scaled or shifted.
    fn affine(&self) -> bool {
        self.weight.is_some() || self.bias.is_some() || self.constant.is_some()
    }

    /// The weight and bias of value `i` of a row: a missing one is the value
    /// that changes nothing, a weight of one or a bias of negative zero.
    #[inline(always)]
    fn parameters(&self, i: usize) -> (f64, f64) {
        match self.constant {
            Some(constant) => constant,
            None => (
                self.weight.map_or(1.0, |weight| weight[i]),
                self.bias.map_or(-0.0, |bias| bias[i]),
            ),
        }
    }
}

/// One row: each output from its pair where the bound settles it, in loops
/// with nothing to branch on, and then, only where one did not, the row's
/// outputs again by [`exact_row`].
#[inline(always)]
fn one_row<const FMA: bool>(
    row: &[f64],
    shape: &Row<'_>,
    out: &mut [f64],
    mut normalized: Option<&mut [f64]>,
) {
    if !row.iter().all(|value| value.is_finite()) {
        return;
    }
    let Some(standard) = Standard::of::<FMA>(row, shape.eps, shape.centred) else {
        return exact_row(row, shape, None, out, normalized);
    };
    let mut open = false;
    if let Some(normalized) = normalized.as_deref_mut() {
        for (cell, &x) in normalized.iter_mut().zip(row) {
            let (value, clear) = standard.settled::<FMA>(x, None);
            *cell = value;
            open |= !clear;
        }
    }
    if shape.affine() {
        for (i, (cell, &x)) in out.iter_mut().zip(row).enumerate() {
            let (value, clear) = standard.settled::<FMA>(x, Some(shape.parameters(i)));
            *cell = value;
            open |= !clear;
        }
    } else if let Some(normalized) = normalized.as_deref() {
        out.copy_from_slice(normalized);
    } else {
        for (cell, &x) in out.iter_mut().zip(row) {
            let (value, clear) = standard.settled::<FMA>(x, None);
            *cell = value;
            open |= !clear;
        }
    }
    if open {
        exact_row(row, shape, Some(&standard), out, normalized);
    }
}

/// A row's outputs where the quick ones did not all settle, or where its
/// statistics could not be bounded: each output its bound settles from the
/// pair, and the rest in whole numbers.
#[cold]
#[inline(never)]
fn exact_row(
    row: &[f64],
    shape: &Row<'_>,
    standard: Option<&Standard>,
    out: &mut [f64],
    mut normalized: Option<&mut [f64]>,
) {
    let exactly = Exactly::of(row, shape.eps, shape.centred);
    for (i, &x) in row.iter().enumerate() {
        let take = |affine: Option<(f64, f64)>, fallback: f64| -> f64 {
            match standard.map(|standard| standard.settled::<false>(x, affine)) {
                Some((value, true)) => value,
                Some((value, false)) => exactly.value(x, affine, value),
                None => exactly.value(x, affine, fallback),
            }
        };
        let plain = take(None, 0.0);
        if let Some(normalized) = normalized.as_deref_mut() {
            normalized[i] = plain;
        }
        out[i] = if shape.affine() {
            let (weight, bias) = shape.parameters(i);
            take(Some((weight, bias)), plain * weight + bias)
        } else {
            plain
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator(mut state: u64) -> impl FnMut() -> f64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let exponent = ((state >> 3) % 17) as i32 - 8;
            (unit - 0.5) * 2f64.powi(exponent)
        }
    }

    /// Every output the bound settles agrees with the one taken in whole
    /// numbers, which is the definition.
    #[test]
    fn the_quick_outputs_are_the_exact_ones() {
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        for trial in 0..60 {
            let width = 1 + trial % 11;
            let mut row: Vec<f64> = (0..width).map(|_| next()).collect();
            if trial % 5 == 0 {
                // A mean that dominates the spread.
                for value in &mut row {
                    *value += 1e6;
                }
            }
            if trial % 7 == 3 {
                // Zeros of both signs, about a mean of exactly zero.
                let half = row.clone();
                row.extend(half.iter().map(|value| -value));
                row.extend([-0.0, 0.0]);
            }
            for &centred in &[true, false] {
                let eps = if trial % 3 == 0 { 0.0 } else { 1e-5 };
                let exactly = Exactly::of(&row, eps, centred);
                let quick = Standard::of::<false>(&row, eps, centred);
                for &x in &row {
                    let weight = next();
                    let bias = next();
                    for affine in [
                        None,
                        Some((weight, bias)),
                        Some((weight, 0.0)),
                        Some((weight, -0.0)),
                        Some((0.0, -0.0)),
                        Some((-0.0, -0.0)),
                    ] {
                        let Some(standard) = &quick else {
                            continue;
                        };
                        let (value, clear) = standard.settled::<false>(x, affine);
                        // From the quick value and from far off alike.
                        let want = exactly.value(x, affine, value);
                        assert_eq!(
                            exactly.value(x, affine, -value * 3.0 + 1.0).to_bits(),
                            want.to_bits()
                        );
                        if clear {
                            assert_eq!(
                                value.to_bits(),
                                want.to_bits(),
                                "{row:?} {x} {affine:?} {centred}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The standardized row, times `weight` plus `bias` where given.
    fn standardize(
        row: &[f64],
        centred: bool,
        eps: f64,
        weight: Option<f64>,
        bias: Option<f64>,
    ) -> Vec<f64> {
        let n = row.len();
        let weights = weight.map(|weight| vec![weight; n]);
        let biases = bias.map(|bias| vec![bias; n]);
        let mut out = vec![f64::NAN; n];
        standardized_rows(
            row,
            n,
            weights.as_deref(),
            biases.as_deref(),
            eps,
            centred,
            &mut out,
            None,
        );
        out
    }

    /// A zero output is signed as the formula signs it: a zero input keeps
    /// its sign, a value equal to a mean that is not zero gives a positive
    /// zero, `w · y + b` signs a zero product as float64 does, and a value
    /// too small for any float64 rounds to the zero on its side.
    #[test]
    fn zeros_are_signed_as_the_formula_signs_them() {
        let same = |got: f64, want: f64| {
            assert_eq!(got.to_bits(), want.to_bits(), "{got} where {want}");
        };
        // A zero input keeps its sign, not centred and about a mean of zero.
        let zero_first = [-0.0, 3.0, 4.0];
        same(standardize(&zero_first, false, 0.0, None, None)[0], -0.0);
        same(
            standardize(&[0.0, 3.0, 4.0], false, 0.0, None, None)[0],
            0.0,
        );
        same(
            standardize(&zero_first, false, 0.0, Some(-2.0), None)[0],
            0.0,
        );
        same(
            standardize(&zero_first, false, 0.0, Some(2.0), Some(0.0))[0],
            0.0,
        );
        same(
            standardize(&[-0.0, 1.0, -1.0], true, 1e-5, None, None)[0],
            -0.0,
        );
        same(
            standardize(&[0.0, 1.0, -1.0], true, 1e-5, None, None)[0],
            0.0,
        );
        // A value equal to a mean that is not zero gives a positive zero, and
        // the weight and bias sign it as `w · y + b` does.
        let middle = [2.0, 1.0, 3.0];
        same(standardize(&middle, true, 1e-5, None, None)[0], 0.0);
        same(standardize(&middle, true, 1e-5, Some(-1.0), None)[0], -0.0);
        same(
            standardize(&middle, true, 1e-5, Some(-1.0), Some(0.0))[0],
            0.0,
        );
        // A zero weight times a negative value, plus a bias.
        same(
            standardize(&middle, true, 1e-5, Some(0.0), Some(-0.0))[1],
            -0.0,
        );
        same(
            standardize(&middle, true, 1e-5, Some(-0.0), Some(-0.0))[1],
            0.0,
        );
        same(
            standardize(&middle, true, 1e-5, Some(0.0), Some(1.5))[1],
            1.5,
        );
        // -1/3 times the smallest float64 rounds to the zero below.
        let mut tenth = [0.0; 10];
        tenth[9] = 1.0;
        let smallest = f64::from_bits(1);
        same(
            standardize(&tenth, true, 0.0, Some(smallest), None)[0],
            -0.0,
        );
        same(
            standardize(&tenth, true, 0.0, Some(smallest), Some(0.0))[0],
            -0.0,
        );
    }
}
