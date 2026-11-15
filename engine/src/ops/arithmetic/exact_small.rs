// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Determinants of order two and three, and Cramer's quotients of them,
//! rounded once.
//!
//! A determinant of order two is the sum of two products of its entries and
//! one of order three the sum of six, and by Cramer's rule each entry of the
//! solution of a system of that order -- and so of an inverse, the solution
//! against the identity -- is the quotient of two of them. Taken by
//! elimination, each pivot, product and update rounded, and a determinant of
//! whole numbers came out -0.9999999999999998 where it is -1.
//!
//! Here each is the float nearest its exact value. The products are split
//! exactly and added with a bound on what the additions rounded, and a
//! quotient takes its correction from the exact remainder of its leading
//! division; where the bound places the value it is taken from the pair, and
//! where it does not, or an entry lies too far out for the products to split
//! exactly, the sums are taken in whole numbers.

use super::{Output, Total, exact_terms, exact_terms_over, product_error, two_sum};

/// A unit in the last place of float64, relative: what one rounding to
/// nearest can move a value by.
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// A value within `bound` of the exact one, as a two-sum's pair.
#[derive(Clone, Copy, Debug)]
struct Estimate {
    value: f64,
    rest: f64,
    bound: f64,
}

/// A bound grown a little past itself, so that the roundings of its own
/// arithmetic cannot leave it short.
#[inline(always)]
fn generous(bound: f64) -> f64 {
    bound * (1.0 + 2f64.powi(-40))
}

/// Whether every entry lies far enough inside the range that the products the
/// estimates split, their errors and the products of those stay ordinary
/// float64s: each entry zero, or of a magnitude within `2^250` of one.
fn comfortable(entries: &[f64]) -> bool {
    entries
        .iter()
        .all(|&value| value == 0.0 || (2f64.powi(-250)..=2f64.powi(250)).contains(&value.abs()))
}

/// `x · y` and its exact error.
#[inline(always)]
fn split(x: f64, y: f64) -> (f64, f64) {
    let product = x * y;
    (product, product_error(x, y, product))
}

/// `x · y - z · w`.
#[inline(always)]
fn minor(x: f64, y: f64, z: f64, w: f64) -> Estimate {
    let (left, left_error) = split(x, y);
    let (right, right_error) = split(z, w);
    let (head, carry) = two_sum(left, -right);
    // Two roundings: of the errors' difference, and of its sum with what
    // the two-sum left.
    let errors = left_error - right_error;
    let tail = carry + errors;
    let (value, rest) = two_sum(head, tail);
    Estimate {
        value,
        rest,
        bound: (errors.abs() + tail.abs()) * UNIT,
    }
}

/// The cofactor of entry `(r, i)` of the 3x3 matrix `m`, rows in order: the
/// minor without row `r` and column `i`, its sign taken by swapping the two
/// products, which negates exactly.
#[inline(always)]
fn cofactor(m: &[f64], r: usize, i: usize) -> Estimate {
    let (top, bottom) = ([1, 0, 0][r], [2, 2, 1][r]);
    let (left, right) = ([1, 0, 0][i], [2, 2, 1][i]);
    let at = |row: usize, column: usize| m[row * 3 + column];
    let (x, y) = (at(top, left), at(bottom, right));
    let (z, w) = (at(top, right), at(bottom, left));
    if (r + i) % 2 == 1 {
        minor(z, w, x, y)
    } else {
        minor(x, y, z, w)
    }
}

/// `Σ x · e` over entries `xs` and estimates `es`, two or three of each.
#[inline(never)]
fn combine(xs: &[f64], es: &[Estimate]) -> Estimate {
    let mut heads = [0.0f64; 3];
    let (mut small, mut magnitude, mut bound) = (0.0f64, 0.0f64, 0.0f64);
    for ((head, &x), e) in heads.iter_mut().zip(xs).zip(es) {
        if x == 0.0 {
            continue;
        }
        let (product, product_error) = split(x, e.value);
        let carried = x * e.rest;
        *head = product;
        small += product_error + carried;
        magnitude += product_error.abs() + carried.abs();
        bound += x.abs() * e.bound + carried.abs() * UNIT;
    }
    let (first, first_error) = two_sum(heads[0], heads[1]);
    let (total, total_error) = two_sum(first, heads[2]);
    small += first_error + total_error;
    magnitude += first_error.abs() + total_error.abs();
    let (value, rest) = two_sum(total, small);
    // `small` took eight roundings at most, each of at most a unit of the
    // magnitude of everything it had added.
    Estimate {
        value,
        rest,
        bound: generous(bound + magnitude * 10.0 * UNIT),
    }
}

/// The cofactors of the `n`x`n` matrix `m`, rows in order, and its
/// determinant, its entries comfortable.
struct Adjugate {
    cofactors: [Estimate; 9],
    determinant: Estimate,
}

impl Adjugate {
    #[inline(never)]
    fn of(m: &[f64], n: usize) -> Self {
        let exact = |value: f64| Estimate {
            value,
            rest: 0.0,
            bound: 0.0,
        };
        let mut cofactors = [exact(0.0); 9];
        if n == 2 {
            for (slot, value) in cofactors.iter_mut().zip([m[3], -m[2], -m[1], m[0]]) {
                *slot = exact(value);
            }
            let determinant = minor(m[0], m[3], m[1], m[2]);
            return Self {
                cofactors,
                determinant,
            };
        }
        for (index, slot) in cofactors.iter_mut().enumerate() {
            *slot = cofactor(m, index / 3, index % 3);
        }
        let determinant = combine(&m[..3], &cofactors[..3]);
        Self {
            cofactors,
            determinant,
        }
    }

    /// Entry `i` of `adj(m) b`, `b`'s column `column` of `columns`: the
    /// determinant of `m` with its column `i` taken from `b`.
    fn numerator(&self, n: usize, b: &[f64], columns: usize, column: usize, i: usize) -> Estimate {
        let xs: [f64; 3] =
            std::array::from_fn(|r| if r < n { b[r * columns + column] } else { 0.0 });
        let es: [Estimate; 3] = std::array::from_fn(|r| {
            if r < n {
                self.cofactors[r * n + i]
            } else {
                self.cofactors[0]
            }
        });
        combine(&xs[..n], &es[..n])
    }
}

/// The determinant of the `n`x`n` matrix `m`, rows in order, `n` two or
/// three, its entries comfortable.
fn estimate(m: &[f64], n: usize) -> Estimate {
    if n == 2 {
        return minor(m[0], m[3], m[1], m[2]);
    }
    let first = [cofactor(m, 0, 0), cofactor(m, 0, 1), cofactor(m, 0, 2)];
    combine(&m[..3], &first)
}

/// The exact determinant as the sum of three-factor products: two for order
/// two, six for order three.
fn terms(m: &[f64], n: usize) -> Vec<(f64, f64, f64)> {
    if n == 2 {
        return vec![(m[0], m[3], 1.0), (-m[1], m[2], 1.0)];
    }
    let [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]];
    vec![
        (a, e, i),
        (-a, f, h),
        (b, f, g),
        (-b, d, i),
        (c, d, h),
        (-c, e, g),
    ]
}

/// `N / D` of two estimates, or None where the quotient leaves the ordinary
/// range or `D`'s bound reaches near its value.
#[inline(never)]
fn quotient(numerator: Estimate, divisor: Estimate) -> Option<Estimate> {
    if numerator.value == 0.0 && numerator.bound == 0.0 {
        // An exact zero over anything not zero.
        return Some(Estimate {
            value: 0.0,
            rest: 0.0,
            bound: 0.0,
        });
    }
    let scale = divisor.value.abs();
    // Written so that a NaN fails it too.
    let placed = divisor.bound < scale * 2f64.powi(-50);
    if !placed {
        return None;
    }
    let leading = numerator.value / divisor.value;
    if !(2f64.powi(-900)..=2f64.powi(900)).contains(&leading.abs()) {
        return None;
    }
    // The leading division's remainder, which float64 holds exactly.
    let (product, product_error) = split(leading, divisor.value);
    let remainder = (numerator.value - product) - product_error;
    // What the pairs' low parts add to it, in two roundings and a third for
    // the product.
    let carried = leading * divisor.rest;
    let low = numerator.rest - carried;
    let residual = remainder + low;
    let correction = residual / divisor.value;
    // The exact divisor is within its bound and its low part of the leading
    // one, so that dividing by the leading one moves the correction by those
    // over it; the estimates' bounds move the residual.
    let moved = numerator.bound
        + leading.abs() * divisor.bound
        + (carried.abs() + low.abs() + residual.abs()) * UNIT
        + residual.abs() * (divisor.rest.abs() + divisor.bound) / scale;
    let bound = moved / (scale * (1.0 - 2f64.powi(-49))) + correction.abs() * UNIT;
    let (value, rest) = two_sum(leading, correction);
    Some(Estimate {
        value,
        rest,
        bound: generous(bound) + f64::from_bits(2),
    })
}

/// Whether the exact determinant of `m` is zero, its entries finite.
fn singular(m: &[f64], n: usize) -> bool {
    let total = Total::of_terms(&terms(m, n));
    total.positive == total.negative
}

/// The float nearest the exact determinant of the `n`x`n` matrix `m`, rows in
/// order, `n` two or three and every entry finite; a zero determinant is
/// positive zero.
pub(crate) fn small_determinant<T: Output>(m: &[f64], n: usize) -> T {
    settled(comfortable(m).then(|| estimate(m, n)))
        .unwrap_or_else(|| exact_terms(&terms(m, n), 0.0))
}

/// The value an estimate settles to, where it does: an estimate of zero
/// settles only with no bound, as an exact zero, which is positive.
fn settled<T: Output>(estimate: Option<Estimate>) -> Option<T> {
    let estimate = estimate?;
    let value = T::settle(estimate.value, estimate.rest, estimate.bound)?;
    Some(if estimate.value == 0.0 {
        T::narrow(0.0)
    } else {
        value
    })
}

/// The solution of `a x = b` into `out`, each entry the float nearest its
/// exact value: `a` is `n`x`n` and `b` and `out` are `n`x`columns`, rows in
/// order, `n` two or three and every entry finite. False, with `out` as it
/// was, where `a` is singular.
pub(crate) fn small_solve<T: Output>(
    a: &[f64],
    n: usize,
    b: &[f64],
    columns: usize,
    out: &mut [T],
) -> bool {
    // Cramer's rule through the adjugate: entry `i` is the determinant of
    // `a` with its column `i` taken from `b`, which is `b` against the
    // cofactors of that column, over the determinant of `a`.
    let quick = (comfortable(a) && comfortable(b)).then(|| Adjugate::of(a, n));
    let nonzero = quick.as_ref().is_some_and(|quick| {
        let d = quick.determinant;
        d.value.abs() > d.rest.abs() + d.bound
    });
    if !nonzero && singular(a, n) {
        return false;
    }
    for column in 0..columns {
        for i in 0..n {
            let estimate = quick.as_ref().and_then(|quick| {
                quotient(quick.numerator(n, b, columns, column, i), quick.determinant)
            });
            out[i * columns + column] = settled(estimate).unwrap_or_else(|| {
                exact_terms_over(
                    &replaced_terms(a, n, b, columns, column, i),
                    &terms(a, n),
                    0.0,
                )
            });
        }
    }
    true
}

/// The terms of the determinant of `a` with its column `i` taken from
/// column `column` of `b`.
#[inline(never)]
fn replaced_terms(
    a: &[f64],
    n: usize,
    b: &[f64],
    columns: usize,
    column: usize,
    i: usize,
) -> Vec<(f64, f64, f64)> {
    let mut replaced = a.to_vec();
    for row in 0..n {
        replaced[row * n + i] = b[row * columns + column];
    }
    terms(&replaced, n)
}

/// The sign of the exact determinant of `m`, `n` two or three and every
/// entry finite, and the logarithm of its magnitude: `(0, -inf)` for a
/// singular matrix. None where the determinant lies outside float64's range
/// even with the entries scaled by a power of two.
pub(crate) fn small_log_determinant(m: &[f64], n: usize) -> Option<(f64, f64)> {
    let quick = comfortable(m).then(|| estimate(m, n));
    let determinant: f64 = settled(quick).unwrap_or_else(|| exact_terms(&terms(m, n), 0.0));
    if determinant.is_normal() {
        let sign = determinant.signum();
        let magnitude = determinant.abs();
        let log = if (0.5..=2.0).contains(&magnitude) {
            // Near one the logarithm is small, and `ln` of the rounded value
            // would keep only its rounding: `log1p` of the magnitude less
            // one, rounded once.
            let less_one = quick.and_then(|estimate| {
                let (head, carry) = two_sum(sign * estimate.value, -1.0);
                let tail = carry + sign * estimate.rest;
                let (value, rest) = two_sum(head, tail);
                f64::settle(value, rest, generous(estimate.bound + tail.abs() * UNIT))
            });
            less_one
                .unwrap_or_else(|| {
                    let mut shifted = terms(m, n);
                    for term in &mut shifted {
                        term.0 *= sign;
                    }
                    shifted.push((-1.0, 1.0, 1.0));
                    exact_terms(&shifted, 0.0)
                })
                .ln_1p()
        } else {
            magnitude.ln()
        };
        return Some((sign, log));
    }
    if determinant == 0.0 && singular(m, n) {
        return Some((0.0, f64::NEG_INFINITY));
    }
    // Past the range: the entries scaled by a power of two that brings the
    // largest to one, where that is exact, scale the determinant by its
    // `n`-th power.
    let top = m.iter().fold(0.0f64, |top, value| top.max(value.abs()));
    let shift = -(top.log2().floor() as i32);
    let factor = 2f64.powi(shift);
    let back = 2f64.powi(-shift);
    let scaled: Vec<f64> = m.iter().map(|&value| value * factor).collect();
    if !scaled
        .iter()
        .zip(m)
        .all(|(&value, &original)| (value * back == original) & value.is_finite())
    {
        return None;
    }
    let determinant = small_determinant::<f64>(&scaled, n);
    if !determinant.is_normal() {
        return None;
    }
    // `ln 2` in two parts, the leading one short enough that its product
    // with any power here is exact.
    let powers = f64::from(shift) * n as f64;
    let log = (determinant.abs().ln() - powers * 0.693_147_180_369_123_8)
        - powers * 1.908_214_929_270_587_7e-10;
    Some((determinant.signum(), log))
}

/// `scale` times each cofactor of the `n`x`n` matrix `m` into `out`, rows in
/// order, each the float nearest its exact value: the derivative of the
/// determinant, scaled. `n` is two or three and every value finite.
pub(crate) fn small_cofactors<T: Output>(m: &[f64], n: usize, scale: f64, out: &mut [T]) {
    if n == 2 {
        // Each cofactor is one entry, so its product with the scale rounds
        // once as it is: exact in float64 for float32 entries.
        let cofactors = [m[3], -m[2], -m[1], m[0]];
        for (slot, cofactor) in out.iter_mut().zip(cofactors) {
            let product = if scale == 0.0 || cofactor == 0.0 {
                0.0
            } else {
                scale * cofactor
            };
            *slot = T::narrow(product);
        }
        return;
    }
    let estimates = scaled_cofactors(m, scale);
    for (index, (slot, estimate)) in out.iter_mut().zip(estimates).enumerate() {
        *slot =
            settled(estimate).unwrap_or_else(|| exact_terms(&cofactor_terms(m, scale, index), 0.0));
    }
}

/// `scale` times cofactor `index` of the 3x3 matrix `m`, rows in order: two
/// three-factor products, the cofactor's sign on the scale, which negates
/// exactly.
fn cofactor_terms(m: &[f64], scale: f64, index: usize) -> [(f64, f64, f64); 2] {
    let (i, j) = (index / 3, index % 3);
    let (top, bottom) = ([1, 0, 0][i], [2, 2, 1][i]);
    let (left, right) = ([1, 0, 0][j], [2, 2, 1][j]);
    let at = |row: usize, column: usize| m[row * 3 + column];
    let signed = if (i + j) % 2 == 1 { -scale } else { scale };
    [
        (signed, at(top, left), at(bottom, right)),
        (-signed, at(top, right), at(bottom, left)),
    ]
}

/// The estimates of [`small_cofactors`] for order three, the same for every
/// dtype, or None for each where a value lies too far out.
#[inline(never)]
fn scaled_cofactors(m: &[f64], scale: f64) -> [Option<Estimate>; 9] {
    let easy = comfortable(m) && comfortable(&[scale]);
    std::array::from_fn(|index| {
        if !easy {
            return None;
        }
        let [(signed, x, y), (_, z, w)] = cofactor_terms(m, scale, index);
        let minor = minor(x, y, z, w);
        let (product, product_error) = split(signed, minor.value);
        let carried = signed * minor.rest;
        let tail = product_error + carried;
        let (value, rest) = two_sum(product, tail);
        let bound = signed.abs() * minor.bound + (carried.abs() + tail.abs()) * UNIT;
        Some(Estimate {
            value,
            rest,
            bound: generous(bound),
        })
    })
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
            let exponent = ((state >> 3) % 9) as i32 - 4;
            (unit - 0.5) * 2f64.powi(exponent)
        }
    }

    /// Every quick estimate settles to what the whole numbers give.
    #[test]
    fn the_quick_values_are_the_exact_ones() {
        let mut next = generator(0x2545_f491_4f6c_dd1d);
        for trial in 0..400 {
            let n = 2 + trial % 2;
            let mut m: Vec<f64> = (0..n * n).map(|_| next()).collect();
            if trial % 4 == 0 {
                // Whole numbers, whose determinants are whole numbers.
                for value in &mut m {
                    *value = (*value * 16.0).round();
                }
            }
            let want: f64 = exact_terms(&terms(&m, n), 0.0);
            let got: f64 = small_determinant(&m, n);
            assert_eq!(got.to_bits(), want.to_bits(), "{m:?}");
            let narrow: f32 = exact_terms(&terms(&m, n), 0.0);
            assert_eq!(small_determinant::<f32>(&m, n).to_bits(), narrow.to_bits());

            let b: Vec<f64> = (0..n * 2).map(|_| next()).collect();
            let mut out = vec![0.0f64; n * 2];
            if !small_solve(&m, n, &b, 2, &mut out) {
                assert!(singular(&m, n));
                continue;
            }
            for column in 0..2 {
                for i in 0..n {
                    let mut replaced = m.clone();
                    for row in 0..n {
                        replaced[row * n + i] = b[row * 2 + column];
                    }
                    let want: f64 = exact_terms_over(&terms(&replaced, n), &terms(&m, n), 0.0);
                    assert_eq!(out[i * 2 + column].to_bits(), want.to_bits(), "{m:?} {b:?}");
                }
            }
        }
    }

    #[test]
    fn whole_numbers_give_whole_numbers() {
        let m = [2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        assert_eq!(small_determinant::<f64>(&m, 3), -1.0);
        let mut inverse = [0.0f64; 9];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(small_solve(&m, 3, &identity, 3, &mut inverse));
        assert_eq!(inverse, [0.0, 0.0, 1.0, -2.0, 1.0, 3.0, 3.0, -1.0, -5.0]);
        // Singular, though elimination leaves a pivot of 6.7e-16.
        let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(small_determinant::<f64>(&m, 3).to_bits(), 0.0f64.to_bits());
        assert!(!small_solve(&m, 3, &[1.0, 2.0, 3.0], 1, &mut [0.0f64; 3]));
        assert_eq!(small_log_determinant(&m, 3), Some((0.0, f64::NEG_INFINITY)));
    }

    #[test]
    fn far_entries_are_taken_exactly() {
        // Past the range the estimates split, and a determinant past
        // float64's: the logarithm comes from the entries scaled.
        let m = [1e200, 3e199, 2e199, 1e200];
        assert!(small_determinant::<f64>(&m, 2).is_infinite());
        let (sign, log) = small_log_determinant(&m, 2).expect("in range scaled");
        assert_eq!(sign, 1.0);
        let want = (1.0f64 - 0.06).ln() + 400.0 * 10f64.ln();
        assert!(
            (log - want).abs() <= 4.0 * f64::EPSILON * want,
            "{log} {want}"
        );
    }

    #[test]
    fn cofactors_are_each_rounded_once() {
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        for _ in 0..50 {
            let m: Vec<f64> = (0..9).map(|_| next()).collect();
            let scale = next();
            let mut out = [0.0f64; 9];
            small_cofactors(&m, 3, scale, &mut out);
            for i in 0..3 {
                for j in 0..3 {
                    let rows: Vec<usize> = (0..3).filter(|&r| r != i).collect();
                    let columns: Vec<usize> = (0..3).filter(|&c| c != j).collect();
                    let at = |r: usize, c: usize| m[rows[r] * 3 + columns[c]];
                    let sign = if (i + j) % 2 == 1 { -scale } else { scale };
                    let want: f64 = exact_terms(
                        &[(sign, at(0, 0), at(1, 1)), (-sign, at(0, 1), at(1, 0))],
                        0.0,
                    );
                    assert_eq!(out[i * 3 + j].to_bits(), want.to_bits());
                }
            }
        }
    }
}
