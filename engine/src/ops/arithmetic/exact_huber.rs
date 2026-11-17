// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The terms of `huber_loss` and `smooth_l1_loss`, and their sums and
//! means, each the float nearest its exact value.
//!
//! A Huber term of the exact difference `d` of a prediction and its target
//! is `d² / 2` where `|d| ≤ δ` and `δ (|d| - δ / 2)` past it, and a
//! smooth-L1 term with `β` is that, with `δ = β`, over `β`: `d² / 2D` inside
//! and `k (|d| - δ / 2)` past it, for `D` and `k` one and `δ` for Huber and
//! `β` and one for smooth-L1.
//!
//! The difference is a two-sum's pair `h + t`. Its square is `h² + 2ht +
//! t²`, and with each product split into the float64 nearest it and that
//! rounding's error it is a few floats exactly; so is `|d| - δ / 2`, a
//! two-sum of `|h|` and `δ / 2` with `t` beside it. Each is carried as a
//! leading pair and a bound on what the pair leaves -- none where nothing
//! was rounded on the way, so that a term on a halfway point is still told
//! -- and settled as a quotient by `2D`, or as itself times `k`. A sum or a
//! mean carries the numerators `2D` times the terms, `d²` and `2δ (|d| - δ
//! / 2)`, and divides once, by `2D` and, for a mean, the count. Where a
//! bound cannot say, or values are so small or so large that a product may
//! not split exactly, the numerators are summed in whole numbers and
//! divided there -- as every term is on an x86_64 CPU without fused
//! multiply-adds, where splitting a product exactly takes as long.

use super::{
    Carried, Output, Total, dot_blocks, is_negative, pairwise_fold, product_error, ratio_settled,
    two_sum,
};
use crate::ops::map::par_out_chunks;

/// A Huber-shaped loss's settings, at the width of its operands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Huber {
    /// Where the quadratic region ends: Huber's `δ`, or smooth-L1's `β`.
    pub(crate) delta: f64,
    /// Whether the terms are over `δ`, as smooth-L1's are.
    pub(crate) smooth: bool,
}

impl Huber {
    /// `D`, what each term's numerator is over twice.
    fn divisor(self) -> f64 {
        if self.smooth { self.delta } else { 1.0 }
    }
}

/// Whether `value` is a zero or lies within `[2^-400, 2^400]`: where it is,
/// and so are the operands of a difference and `δ`, every product a term is
/// made of lies above `2^-905`, and splits into two float64s exactly.
#[inline(always)]
fn in_range(value: f64) -> bool {
    value == 0.0 || (2f64.powi(-400)..=2f64.powi(400)).contains(&value.abs())
}

/// `x · y` as the float64 nearest it and that rounding's error.
#[inline(always)]
fn split<const FMA: bool>(x: f64, y: f64) -> (f64, f64) {
    let product = x * y;
    (
        product,
        crate::ops::simd::product_error::<FMA>(x, y, product),
    )
}

/// Whether the exact difference `h + t` lies within `δ`: where its head is
/// below `δ`, and not where it is above -- rounding is monotonic, and `δ` a
/// float -- and at `δ` by the tail's sign; there the two forms agree.
#[inline(always)]
fn inside(head: f64, tail: f64, delta: f64) -> bool {
    head.abs() < delta || (head.abs() == delta && head * tail <= 0.0)
}

/// `(h + t)²`, carried: `h² + 2ht` split exactly, and `t²`, below `2^-104`
/// of `h²`, in the bound with what adding the small parts dropped -- each
/// addition's error found by a two-sum, so that the bound is nothing where
/// nothing was.
#[inline(always)]
fn square<const FMA: bool>(head: f64, tail: f64) -> Carried {
    let (square, square_error) = split::<FMA>(head, head);
    let (cross, cross_error) = split::<FMA>(2.0 * head, tail);
    let (error, dropped) = two_sum(square_error, cross);
    Carried {
        sum: square,
        error,
        slack: dropped.abs() + cross_error.abs() + tail * tail * (1.0 + 2f64.powi(-50)),
    }
}

/// `|h + t| - δ / 2` past `δ`, carried: the head's magnitude less `δ / 2`
/// by a two-sum, and the tail, signed with the head, beside what that drops.
#[inline(always)]
fn excess(head: f64, tail: f64, delta: f64) -> Carried {
    let (head, tail) = if head < 0.0 {
        (-head, -tail)
    } else {
        (head, tail)
    };
    let (sum, dropped) = two_sum(head, -0.5 * delta);
    let (error, lost) = two_sum(dropped, tail);
    Carried {
        sum,
        error,
        slack: lost.abs(),
    }
}

/// `value` times `scale`, carried: both parts' products split exactly, the
/// error's error in the bound; exact for a scale of one.
#[inline(always)]
fn scaled<const FMA: bool>(value: Carried, scale: f64) -> Carried {
    let (sum, sum_error) = split::<FMA>(scale, value.sum);
    let (error, error_error) = split::<FMA>(scale, value.error);
    let (total, dropped) = two_sum(sum_error, error);
    Carried {
        sum,
        error: total,
        slack: scale * value.slack * (1.0 + 2f64.powi(-50)) + error_error.abs() + dropped.abs(),
    }
}

/// The numerator of the term of `a - b`, `2D` times it, carried: `d²`, or
/// `2δ (|d| - δ / 2)`. `a`, `b` and `delta` are in range.
#[inline(always)]
fn numerator<const FMA: bool>(a: f64, b: f64, delta: f64) -> Carried {
    let (head, tail) = two_sum(a, -b);
    if inside(head, tail, delta) {
        square::<FMA>(head, tail)
    } else {
        scaled::<FMA>(excess(head, tail, delta), 2.0 * delta)
    }
}

/// The term of `a - b` from its carried parts, when the bound says which
/// value it is nearest. `a`, `b` and `delta` are in range.
#[inline(always)]
fn settled_term<T: Output, const FMA: bool>(a: f64, b: f64, huber: Huber) -> Option<T> {
    let (head, tail) = two_sum(a, -b);
    if inside(head, tail, huber.delta) {
        ratio_settled::<T, FMA>(square::<FMA>(head, tail), (2.0 * huber.divisor(), 0.0))
    } else {
        let scale = if huber.smooth { 1.0 } else { huber.delta };
        scaled::<FMA>(excess(head, tail, huber.delta), scale).rounded()
    }
}

/// The products whose sum is exactly the numerator of the term of the
/// difference `Σ parts`, of finite floats: the parts' products with one
/// another in the quadratic region, and past it each part, signed as the
/// difference is, with `δ` twice and `δ` with `-δ`. The sign and the region
/// are found by a two-sum where it holds the difference, and otherwise in
/// whole numbers; an infinite `δ` leaves every difference inside it.
fn numerator_products(parts: [f64; 3], delta: f64) -> [(f64, f64); 9] {
    let (head, tail) = two_sum(parts[0], parts[1]);
    let (negative, beyond) = if parts[2] == 0.0 && head.is_finite() {
        let negative = head < 0.0 || (head == 0.0 && tail < 0.0);
        (negative, !inside(head, tail, delta))
    } else {
        let negative = is_negative(&parts);
        let sign = if negative { -1.0 } else { 1.0 };
        let beyond = delta.is_finite()
            && is_negative(&[delta, -sign * parts[0], -sign * parts[1], -sign * parts[2]]);
        (negative, beyond)
    };
    let sign = if negative { -1.0 } else { 1.0 };
    let mut products = [(0.0, 0.0); 9];
    if beyond {
        for (index, &part) in parts.iter().enumerate() {
            products[2 * index] = (delta, sign * part);
            products[2 * index + 1] = (delta, sign * part);
        }
        products[6] = (delta, -delta);
    } else {
        for (index, slot) in products.iter_mut().enumerate() {
            *slot = (parts[index / 3], parts[index % 3]);
        }
    }
    products
}

/// The term of the difference `d` taken plainly in float64: what the
/// formula gives for a NaN or an infinity.
fn plain(d: f64, huber: Huber) -> f64 {
    let magnitude = d.abs();
    let term = if magnitude <= huber.delta {
        0.5 * d * d
    } else {
        huber.delta * (magnitude - 0.5 * huber.delta)
    };
    term / huber.divisor()
}

/// The numerators of the terms of the differences `x - y - lows` summed
/// exactly, and `2D` times `count` beside them: or, with a NaN or an
/// infinity among the operands, the total over the count as the formula
/// gives it -- every term at least zero, an infinity or NaN.
#[cold]
#[inline(never)]
fn exact_parts(
    (x, y, lows): (&[f64], &[f64], Option<&[f64]>),
    huber: Huber,
    count: usize,
) -> Result<(Total, Total), f64> {
    let low = |index: usize| lows.map_or(0.0, |lows| lows[index]);
    let finite = x.iter().chain(y).all(|value| value.is_finite());
    if !(finite && huber.divisor().is_finite()) {
        let total: f64 = (0..x.len())
            .map(|index| plain((x[index] - y[index]) - low(index), huber))
            .sum();
        return Err(total / count as f64);
    }
    let products = (0..x.len())
        .flat_map(|index| numerator_products([x[index], -y[index], -low(index)], huber.delta));
    let divisor = Total::of_products([(huber.divisor(), count as f64); 2].into_iter())?;
    Ok((Total::of_products(products)?, divisor))
}

/// The sum of the terms of the differences `x - y - lows` over `count`, in
/// whole numbers: [`exact_parts`] divided.
fn exact_total<T: Output>(
    operands: (&[f64], &[f64], Option<&[f64]>),
    huber: Huber,
    count: usize,
) -> T {
    match exact_parts(operands, huber, count) {
        Ok((numerator, divisor)) => numerator.quotient(divisor),
        Err(value) => T::narrow(value),
    }
}

/// Whether the products are split by a fused multiply-add here: always
/// where it is one instruction of the target, and on x86_64 where the CPU
/// has it. Without one the terms are taken in whole numbers, as a dot's
/// third parts are.
fn fused() -> bool {
    !cfg!(target_arch = "x86_64") || super::fused()
}

/// The term of `a - b - low` in whole numbers.
#[cold]
#[inline(never)]
fn exact_term<T: Output>(a: f64, b: f64, low: f64, huber: Huber) -> T {
    exact_total((&[a], &[b], Some(&[low])), huber, 1)
}

/// Terms of `x - y`, less `lows` where an integer target lies past 2^53,
/// into `out`: each settled from its carried parts where its operands are
/// in range, and otherwise in whole numbers. Compiled for fused
/// multiply-adds, and reached only where [`fused`]; `δ` is in range.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
fn terms_run<T: Output>(
    (x, y, lows): (&[f64], &[f64], Option<&[f64]>),
    huber: Huber,
    out: &mut [T],
) {
    for (index, slot) in out.iter_mut().enumerate() {
        let (a, b) = (x[index], y[index]);
        let low = lows.map_or(0.0, |lows| lows[index]);
        let settled = if low == 0.0 && in_range(a) && in_range(b) {
            settled_term::<T, true>(a, b, huber)
        } else {
            None
        };
        *slot = settled.unwrap_or_else(|| exact_term(a, b, low, huber));
    }
}

/// Each term of `x - y` -- less `lows` too, where an integer target lies
/// past 2^53 -- the float nearest its exact value.
pub(crate) fn huber_terms<T: Output>(
    x: &[f64],
    y: &[f64],
    lows: Option<&[f64]>,
    huber: Huber,
) -> Vec<T> {
    let fast = in_range(huber.delta) && fused();
    let mut out = vec![T::narrow(0.0); x.len()];
    par_out_chunks(&mut out, 1 << 14, &|start, part| {
        let end = start + part.len();
        let operands = (
            &x[start..end],
            &y[start..end],
            lows.map(|lows| &lows[start..end]),
        );
        if fast {
            // SAFETY: where `fast` the CPU has avx2 and fma, or the target
            // a fused multiply-add of its own.
            #[allow(unused_unsafe)]
            return unsafe { terms_run(operands, huber, part) };
        }
        let (x, y, lows) = operands;
        for (index, slot) in part.iter_mut().enumerate() {
            let low = lows.map_or(0.0, |lows| lows[index]);
            *slot = exact_term(x[index], y[index], low, huber);
        }
    });
    out
}

/// A block's numerators of the terms of `xs - ys` added up, carried, in
/// four lanes; None where an operand is out of range.
#[inline(always)]
fn block_total<const FMA: bool>(xs: &[f64], ys: &[f64], delta: f64) -> Option<Carried> {
    let mut lanes = [Carried::default(); 4];
    for (index, (&a, &b)) in xs.iter().zip(ys).enumerate() {
        if !(in_range(a) && in_range(b)) {
            return None;
        }
        let lane = &mut lanes[index % 4];
        *lane = lane.merge_tracked(numerator::<FMA>(a, b, delta));
    }
    let [first, second, third, fourth] = lanes;
    Some(
        first
            .merge_tracked(second)
            .merge_tracked(third.merge_tracked(fourth)),
    )
}

/// [`block_total`] compiled for fused multiply-adds.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
fn block_total_fma(xs: &[f64], ys: &[f64], delta: f64) -> Option<Carried> {
    block_total::<true>(xs, ys, delta)
}

/// The numerators of the terms of `x - y` added up, carried, a block at a
/// time across the pool; None where an operand is out of range. Only where
/// [`fused`].
fn carried_numerators(x: &[f64], y: &[f64], delta: f64) -> Option<Carried> {
    // SAFETY: the caller has found a fused multiply-add: avx2 and fma on
    // x86_64.
    #[allow(unused_unsafe)]
    let block = |xs: &[f64], ys: &[f64]| unsafe { block_total_fma(xs, ys, delta) };
    let parts: Option<Vec<Carried>> = dot_blocks(x, y, block).into_iter().collect();
    Some(pairwise_fold(
        parts?,
        Carried::default(),
        Carried::merge_tracked,
    ))
}

/// The value nearest the exact sum of the terms of `x - y`, less `lows`
/// where given, over `count`: a loss's sum with a count of one, and its
/// mean with the number of terms; NaN for none over none.
pub(crate) fn huber_total<T: Output>(
    x: &[f64],
    y: &[f64],
    lows: Option<&[f64]>,
    huber: Huber,
    count: usize,
) -> T {
    if count == 0 {
        return T::narrow(f64::NAN);
    }
    let settled = (lows.is_none() && in_range(huber.delta) && fused())
        .then(|| carried_numerators(x, y, huber.delta))
        .flatten()
        .and_then(|total| {
            // `2D` times the count, as a product's exact pair.
            let (divisor, count) = (2.0 * huber.divisor(), count as f64);
            let product = divisor * count;
            ratio_settled::<T, false>(total, (product, product_error(divisor, count, product)))
        });
    settled.unwrap_or_else(|| exact_total((x, y, lows), huber, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// Operands spread over many binades, some whole, some equal, some a
    /// rounding apart, and some out of range.
    fn operands(seed: u64, len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut next = generator(seed);
        let mut value = move || {
            let bits = next();
            let unit = (bits >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            match bits % 13 {
                0 => (unit * 16.0).round(),
                1 => unit * 2f64.powi(-440),
                2 => unit * 2f64.powi(420),
                _ => unit * 2f64.powi((bits % 41) as i32 - 20),
            }
        };
        let x: Vec<f64> = (0..len).map(|_| value()).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(index, &a)| match index % 7 {
                0 => a,
                1 => a * (1.0 + 2f64.powi(-40)),
                _ => value(),
            })
            .collect();
        (x, y)
    }

    /// The exact term, or total, in whole numbers.
    fn exact<T: Output>(x: &[f64], y: &[f64], huber: Huber, count: usize) -> T {
        exact_total((x, y, None), huber, count)
    }

    #[test]
    fn every_term_and_total_is_its_exact_value() {
        let settings = [(1.0, false), (0.37, true), (3.0, false), (0.7, true)];
        for (index, &(delta, smooth)) in settings.iter().enumerate() {
            let huber = Huber { delta, smooth };
            let (x, y) = operands(11 + index as u64, 3000);
            let got: Vec<f64> = huber_terms(&x, &y, None, huber);
            for (i, value) in got.iter().enumerate() {
                let want: f64 = exact(&x[i..=i], &y[i..=i], huber, 1);
                assert_eq!(value.to_bits(), want.to_bits(), "term {i}, {huber:?}");
            }
            let narrow: Vec<f64> = x.iter().map(|&v| f64::from(v as f32)).collect();
            let narrow_y: Vec<f64> = y.iter().map(|&v| f64::from(v as f32)).collect();
            let narrow_huber = Huber {
                delta: f64::from(delta as f32),
                smooth,
            };
            let got: Vec<f32> = huber_terms(&narrow, &narrow_y, None, narrow_huber);
            for (i, value) in got.iter().enumerate() {
                let want: f32 = exact(&narrow[i..=i], &narrow_y[i..=i], narrow_huber, 1);
                assert_eq!(value.to_bits(), want.to_bits(), "float32 term {i}");
            }
            // Totals of operands in range, which are carried, and of all of
            // them, which are not.
            let (xi, yi): (Vec<f64>, Vec<f64>) = x
                .iter()
                .zip(&y)
                .filter(|&(&a, &b)| in_range(a) && in_range(b))
                .map(|(&a, &b)| (a, b))
                .unzip();
            for (xs, ys) in [(&xi, &yi), (&x, &y)] {
                for count in [1, xs.len()] {
                    let want: f64 = exact(xs, ys, huber, count);
                    let got: f64 = huber_total(xs, ys, None, huber, count);
                    assert_eq!(got.to_bits(), want.to_bits(), "total over {count}");
                }
            }
        }
    }

    #[test]
    fn halfway_points_zeros_and_specials() {
        let huber = Huber {
            delta: 1.0,
            smooth: false,
        };
        // Whole numbers: halves, exact as they stand, and a zero difference
        // a positive zero.
        let x = [3.0, -2.0, 0.5, 7.0, -0.0];
        let y = [3.0, 0.0, 0.0, -1.0, 0.0];
        let got: Vec<f64> = huber_terms(&x, &y, None, huber);
        assert_eq!(got, [0.0, 1.5, 0.125, 7.5, 0.0]);
        assert!(got.iter().all(|value| value.is_sign_positive()));
        let total: f64 = huber_total(&x, &y, None, huber, 5);
        assert_eq!(total, 9.125 / 5.0);
        // A term on a float32 halfway point: `2^24 + 1`, past `δ`, to the
        // even float32 below.
        let narrow: Vec<f32> = huber_terms(&[16_777_216.0], &[-1.5], None, huber);
        assert_eq!(narrow, [16_777_216.0]);
        // A smooth-L1 term past `β` on a float64 halfway point, which a
        // bound above zero cannot tell: `4 - 2^-52`, to the even 4.
        let smooth = Huber {
            delta: 2f64.powi(-51),
            smooth: true,
        };
        let got: Vec<f64> = huber_terms(&[4.0], &[0.0], None, smooth);
        assert_eq!(got, [4.0]);
        let mut carried = super::excess(4.0, 0.0, smooth.delta);
        assert_eq!(carried.slack, 0.0);
        carried = super::scaled::<false>(carried, 1.0);
        assert_eq!(carried.rounded::<f64>(), Some(4.0));
        // A difference past the largest float64 whose term is in range.
        let near = Huber {
            delta: 2f64.powi(-60),
            smooth: false,
        };
        let got: Vec<f64> = huber_terms(&[f64::MAX], &[-f64::MAX], None, near);
        assert_eq!(got[0], f64::MAX * 2f64.powi(-59));
        // A NaN, and infinities.
        let got: Vec<f64> = huber_terms(
            &[f64::NAN, f64::INFINITY, f64::INFINITY],
            &[0.0, 1.0, f64::INFINITY],
            None,
            huber,
        );
        assert!(got[0].is_nan() && got[1] == f64::INFINITY && got[2].is_nan());
        let total: f64 = huber_total(&[f64::INFINITY, 1.0], &[0.0, 0.0], None, huber, 2);
        assert_eq!(total, f64::INFINITY);
        // An integer target past 2^53, given as its float64 and what that
        // leaves.
        let high = 2f64.powi(60);
        let got: Vec<f64> = huber_terms(&[high], &[high], Some(&[-3.0]), huber);
        assert_eq!(got, [2.5]);
        let total: f64 = huber_total(&[high, 1.0], &[high, 0.5], Some(&[-3.0, 0.0]), huber, 1);
        assert_eq!(total, 2.625);
    }
}
