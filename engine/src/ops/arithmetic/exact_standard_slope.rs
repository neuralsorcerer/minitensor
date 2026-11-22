// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The gradients of standardized values, rounded once.
//!
//! A standardized value is `y = d r`, with `d = x - mean` and
//! `r = 1 / sqrt(var + eps)`, or, not centred, `d = x` and
//! `r = 1 / sqrt(mean(x^2) + eps)`. With `h = g w`, the upstream gradient
//! times the weight, the input's gradient is
//!
//! ```text
//! r (u - r^2 C d),    u = h - mean(h) (h, not centred),    C = mean(u d).
//! ```
//!
//! `u` in place of `h` changes nothing where the deviations sum to zero, and
//! leaves an upstream gradient that is the same everywhere with nothing at
//! all. Formed in the dtype from the rounded standardized values and inverse
//! root, the two terms all but cancel wherever the upstream gradient lies
//! near the span of one and `y`, and kept only their roundings: a float32
//! gradient was 13 ulps out on ordinary data and two million near that span,
//! a float64 one 74 and fifteen million.
//!
//! A float32 row is taken first in plain float64, each gradient with a bound
//! on what its roundings can have moved it, from carried sums: there are 29
//! bits to spare, and nearly every row settles. Otherwise, and for float64,
//! each piece is a pair of float64s with a bound on how far it is from the
//! exact value, from the statistics [`Standard`] carries for the forward, so
//! that each gradient is a pair and a bound, settled where the bound places
//! it. One the bound leaves open, and every one of a row whose values
//! lie too far out for the pairs, is decided in whole numbers: with `n`
//! values, `S = Σx`, `G = Σh`, `H = Σhx` and the forward's divisor `D`, the
//! gradient is the root of `N^2 / D^3` with the sign of `N`, where centred
//! `N = (n h - G) D - (n H - S G)(n x - S)`, and, not centred, the root of
//! `n N^2 / D^3` with `N = h D - x H`.
//!
//! The weight's gradient sums `g y` over the values each weight multiplies:
//! down a column, or over a channel's positions in every group. A sum of
//! roots has no whole number to round from, so it is the sum of the pairs
//! `g y`, rounded once.

use super::super::{Carried, Element, Output, Total, two_sum};
use super::{Exactly, Standard, Whole, comfortable, deviation, generous, split};
use crate::ops::arithmetic::extended::Pair;

/// One row's operands, in float64.
struct Set<'a> {
    x: &'a [f64],
    g: &'a [f64],
    weight: Option<&'a [f64]>,
    /// The weight of every value, where `weight` is None.
    gain: f64,
    eps: f64,
    centred: bool,
}

/// What a row's pass leaves: the upstream times the weight as a product and
/// its exact error, and each input gradient as a pair and a bound.
struct Parts<'a> {
    head: &'a mut [f64],
    tail: &'a mut [f64],
    value: &'a mut [f64],
    rest: &'a mut [f64],
    bound: &'a mut [f64],
}

/// How a row's input gradients were found.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Each a pair and a bound, in the parts.
    Pairs,
    /// Each a float64, in the parts' values: a row holding a value that is
    /// not finite, which the formula carries as it comes.
    Plain,
    /// None: the values lie too far out for the pairs, so every one is
    /// decided in whole numbers.
    Exact,
}

/// Where a row's weight terms `g y` go: each to its column's pair, all of
/// them to one pair, each in place -- its head and tail side by side, for
/// the caller to sum -- or nowhere. Every pass that takes a row writes each
/// of its terms once.
enum Columns<'a> {
    Each(&'a mut [(f64, f64)]),
    Total(&'a mut (f64, f64)),
    Row(&'a mut [f64]),
    Unwanted,
}

impl Columns<'_> {
    fn wanted(&self) -> bool {
        !matches!(self, Columns::Unwanted)
    }

    /// The term `head + tail` of value `k` added to where it goes, by a
    /// two-sum.
    #[inline(always)]
    fn add(&mut self, k: usize, head: f64, tail: f64) {
        let slot = match self {
            Columns::Each(columns) => &mut columns[k],
            Columns::Total(total) => &mut **total,
            Columns::Row(row) => {
                row[2 * k] = head;
                row[2 * k + 1] = tail;
                return;
            }
            Columns::Unwanted => return,
        };
        let (sum, carry) = two_sum(slot.0, head);
        *slot = (sum, slot.1 + (carry + tail));
    }
}

/// A total over `n` as a pair and its bound: zeros for a total that is
/// exactly zero, and None where the quotient leaves the range the pairs
/// keep.
#[inline(always)]
fn over<const FMA: bool>(total: Carried, n: f64) -> Option<(f64, f64, f64)> {
    if total.sum == 0.0 && total.error == 0.0 && total.slack == 0.0 {
        return Some((0.0, 0.0, 0.0));
    }
    let (value, rest, bound, ordinary) = total.quotient_parts::<FMA>(n);
    (ordinary || value == 0.0).then_some((value, rest, bound))
}

/// `h` less the mean `mean + rest`, which is within `bound` of the exact
/// one, as a pair and how far the exact difference can be from it; `h`
/// itself, exactly, where not centred.
#[inline(always)]
fn upstream(
    head: f64,
    tail: f64,
    (mean, rest, bound): (f64, f64, f64),
    centred: bool,
) -> (f64, f64, f64) {
    if !centred {
        return (head, tail, 0.0);
    }
    let (lead, lead_error) = two_sum(head, -mean);
    let first = lead_error + tail;
    let second = first - rest;
    let (high, low) = two_sum(lead, second);
    (
        high,
        low,
        bound + (first.abs() + second.abs()) * 2f64.powi(-52),
    )
}

/// One row's input gradients into `parts` where `input`, and `g y` added to
/// `columns`.
#[inline(always)]
fn row_slopes<const FMA: bool>(
    set: &Set<'_>,
    input: bool,
    parts: &mut Parts<'_>,
    columns: &mut Columns<'_>,
) -> Outcome {
    let len = set.x.len();
    for k in 0..len {
        let (head, tail) = split::<FMA>(set.g[k], set.weight.map_or(set.gain, |w| w[k]));
        parts.head[k] = head;
        parts.tail[k] = tail;
    }
    let finite = |values: &[f64]| values.iter().all(|value| value.is_finite());
    if !(finite(set.x) && finite(set.g) && finite(parts.head)) {
        plain_row(set, parts, columns);
        return Outcome::Plain;
    }
    let ordinary = |values: &[f64]| values.iter().all(|&value| comfortable(value));
    let standard = if ordinary(set.g) && ordinary(parts.head) {
        Standard::of::<FMA>(set.x, set.eps, set.centred)
    } else {
        None
    };
    let Some(standard) = standard else {
        return Outcome::Exact;
    };
    let n = len as f64;
    let r = Pair(standard.scale, standard.scale_rest);
    let r_size = standard.scale.abs() + standard.scale_rest.abs();
    let relative = standard.scale_relative;
    let centre = (standard.centre, standard.centre_rest, standard.centre_bound);
    if input {
        // The upstream's mean, and whether every value of it is that mean.
        let mean = if set.centred {
            let (mut large, mut small, mut magnitude) = (0.0f64, 0.0f64, 0.0f64);
            for (&head, &tail) in parts.head.iter().zip(parts.tail.iter()) {
                let (sum, carry) = two_sum(large, head);
                large = sum;
                small += carry + tail;
                magnitude += carry.abs() + tail.abs();
            }
            let slack = generous(magnitude * (n + 4.0) * 2f64.powi(-52));
            let total = Carried {
                sum: large,
                error: small,
                slack,
            };
            let Some(mean) = over::<FMA>(total, n) else {
                return Outcome::Exact;
            };
            mean
        } else {
            (0.0, 0.0, 0.0)
        };
        let constant = set.centred
            && parts.head.iter().all(|&head| head == parts.head[0])
            && parts.tail.iter().all(|&tail| tail == parts.tail[0]);
        if constant {
            // `u` is zero throughout, and so, exactly, is every gradient.
            parts.value.fill(0.0);
            parts.rest.fill(0.0);
            parts.bound.fill(0.0);
        } else {
            // `C`: each product of pairs split, the large parts added by
            // two-sums and the small ones in float64 with their magnitudes
            // kept for the bound, as the forward's sum of squares is.
            let (mut large, mut small, mut magnitude) = (0.0f64, 0.0f64, 0.0f64);
            let (mut spread, mut drift) = (0.0f64, 0.0f64);
            for k in 0..len {
                let (d, d_low, d_error) = deviation(set.x[k], centre.0, centre.1, centre.2);
                let (u, u_low, u_error) = upstream(parts.head[k], parts.tail[k], mean, set.centred);
                let (product, product_error) = split::<FMA>(u, d);
                let cross = [u * d_low, u_low * d, u_low * d_low];
                let tail = product_error + (cross[0] + cross[1] + cross[2]);
                let (sum, carry) = two_sum(large, product);
                large = sum;
                small += carry + tail;
                magnitude += carry.abs() + tail.abs() + product_error.abs();
                spread += cross[0].abs() + cross[1].abs() + cross[2].abs();
                // Within `u_error` and `d_error` of the pairs, the product
                // moves by at most this.
                drift += (u.abs() + u_low.abs()) * d_error
                    + (d.abs() + d_low.abs()) * u_error
                    + u_error * d_error;
            }
            // The roundings of the small parts are bounded as the forward's
            // are, the three cross products' by twice their sizes, and the
            // drift grown by the roundings of its own sum.
            let slack = generous(
                magnitude * (n + 4.0) * 2f64.powi(-52)
                    + spread * 2f64.powi(-51)
                    + drift * (1.0 + (n + 4.0) * 2f64.powi(-52)),
            );
            let total = Carried {
                sum: large,
                error: small,
                slack,
            };
            let Some((c, c_rest, c_bound)) = over::<FMA>(total, n) else {
                return Outcome::Exact;
            };
            // `t = r^2 C`. A product of pairs is within 2^-100 of the
            // product of what it multiplies, and `r` within `relative` of
            // the exact inverse root, so its square within about twice that.
            let square = r.mul::<FMA>(r);
            let square_size = square.0.abs() + square.1.abs();
            let square_relative = generous(2.0 * relative + relative * relative + 2f64.powi(-100));
            let t = square.mul::<FMA>(Pair(c, c_rest));
            let t_size = t.0.abs() + t.1.abs();
            let c_size = c.abs() + c_rest.abs();
            let t_bound = generous(
                square_size * c_bound
                    + (c_size + c_bound)
                        * square_size
                        * square_relative
                        * (1.0 + 2.0 * square_relative)
                    + t_size * 2f64.powi(-100),
            );
            for k in 0..len {
                let (d, d_low, d_error) = deviation(set.x[k], centre.0, centre.1, centre.2);
                let (u, u_low, u_error) = upstream(parts.head[k], parts.tail[k], mean, set.centred);
                let along = t.mul::<FMA>(Pair(d, d_low));
                let along_size = along.0.abs() + along.1.abs();
                let along_bound = along_size * 2f64.powi(-100)
                    + t_size * d_error
                    + (d.abs() + d_low.abs()) * t_bound
                    + t_bound * d_error;
                // A sum of pairs is within 2^-100 of their sizes.
                let left = Pair(u, u_low).add(along.neg());
                let left_size = left.0.abs() + left.1.abs();
                let left_bound =
                    u_error + along_bound + (u.abs() + u_low.abs() + along_size) * 2f64.powi(-100);
                let slope = r.mul::<FMA>(left);
                let slope_size = slope.0.abs() + slope.1.abs();
                let mut bound = generous(
                    slope_size * 2f64.powi(-100)
                        + r_size * (1.0 + relative) * left_bound
                        + left_size * r_size * relative,
                );
                // Below this the pairs' trailing parts can be subnormal, and
                // their products lose what the bound counts on.
                if slope.0 != 0.0 && slope.0.abs() < 2f64.powi(-900) {
                    bound = f64::INFINITY;
                }
                parts.value[k] = slope.0;
                parts.rest[k] = slope.1;
                parts.bound[k] = bound;
            }
        }
    }
    if columns.wanted() {
        for k in 0..len {
            let (d, d_low, _) = deviation(set.x[k], centre.0, centre.1, centre.2);
            let y = Pair(d, d_low).mul::<FMA>(r);
            let term = y.mul::<FMA>(Pair(set.g[k], 0.0));
            columns.add(k, term.0, term.1);
        }
    }
    Outcome::Pairs
}

/// [`row_slopes`] in the widest build the CPU runs.
fn row_slopes_fused(
    set: &Set<'_>,
    input: bool,
    parts: &mut Parts<'_>,
    columns: &mut Columns<'_>,
) -> Outcome {
    use crate::ops::arithmetic::extended::fused;
    fused!(
        row_slopes(
            set: &Set<'_>,
            input: bool,
            parts: &mut Parts<'_>,
            columns: &mut Columns<'_>,
        ) -> Outcome
    )
}

/// [`row_slopes`] for a row holding a value that is not finite: the formula
/// in float64, which carries its infinities and NaNs as they come.
#[cold]
#[inline(never)]
fn plain_row(set: &Set<'_>, parts: &mut Parts<'_>, columns: &mut Columns<'_>) {
    let len = set.x.len();
    let n = len as f64;
    let h = |k: usize| set.g[k] * set.weight.map_or(set.gain, |w| w[k]);
    let mean = if set.centred {
        set.x.iter().sum::<f64>() / n
    } else {
        0.0
    };
    let var = set.x.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let r = 1.0 / (var + set.eps).sqrt();
    let upstream_mean = if set.centred {
        (0..len).map(h).sum::<f64>() / n
    } else {
        0.0
    };
    let c = (0..len)
        .map(|k| (h(k) - upstream_mean) * (set.x[k] - mean))
        .sum::<f64>()
        / n;
    let t = r * r * c;
    for k in 0..len {
        let d = set.x[k] - mean;
        parts.value[k] = r * ((h(k) - upstream_mean) - t * d);
        parts.rest[k] = 0.0;
        parts.bound[k] = 0.0;
        columns.add(k, set.g[k] * (d * r), 0.0);
    }
}

/// The input gradients at `which` -- at every index where it is None -- in
/// whole numbers; see the module's doc.
#[cold]
#[inline(never)]
fn exact_slopes<T: Output>(
    set: &Set<'_>,
    parts: &Parts<'_>,
    which: Option<&[usize]>,
    out: &mut [T],
) {
    let len = set.x.len();
    let exactly = Exactly::of(set.x, set.eps, set.centred);
    let n = Whole::float(exactly.count);
    let upstream = |k: usize| {
        Whole::of(Total::of_terms(&[
            (parts.head[k], 1.0, 1.0),
            (parts.tail[k], 1.0, 1.0),
        ]))
    };
    let mut terms = Vec::with_capacity(2 * len);
    let mut moments = Vec::with_capacity(2 * len);
    for k in 0..len {
        terms.extend([(parts.head[k], 1.0, 1.0), (parts.tail[k], 1.0, 1.0)]);
        moments.extend([
            (parts.head[k], set.x[k], 1.0),
            (parts.tail[k], set.x[k], 1.0),
        ]);
    }
    let total = Whole::of(Total::of_terms(&terms));
    let moment = Whole::of(Total::of_terms(&moments));
    let divisor = &exactly.divisor;
    let cube = divisor.times(divisor).times(divisor);
    // `n H - S G` centred, and `H` not.
    let across = if set.centred {
        moment.times(&n).plus(&exactly.sum.times(&total).negated())
    } else {
        moment
    };
    let one = |k: usize| -> T {
        if divisor.is_zero() {
            return T::narrow(f64::NAN);
        }
        let x = Whole::float(set.x[k]);
        let numerator = if set.centred {
            let first = upstream(k)
                .times(&n)
                .plus(&total.clone().negated())
                .times(divisor);
            let deviation = x.times(&n).plus(&exactly.sum.clone().negated());
            first.plus(&across.times(&deviation).negated())
        } else {
            upstream(k).times(divisor).plus(&x.times(&across).negated())
        };
        if numerator.is_zero() {
            return T::narrow(0.0);
        }
        let negative = numerator.negative;
        let mut square = numerator.times(&numerator);
        if !set.centred {
            square = square.times(&n);
        }
        let magnitude = square
            .total()
            .quotient_root::<T>(cube.clone().total())
            .widen();
        T::narrow(if negative { -magnitude } else { magnitude })
    };
    match which {
        Some(which) => {
            for &k in which {
                out[k] = one(k);
            }
        }
        None => {
            for (k, cell) in out.iter_mut().enumerate() {
                *cell = one(k);
            }
        }
    }
}

/// `g y` added to `columns` for a row whose values lie too far out for the
/// pairs: each `y` the float64 nearest it, its product with `g` exact.
#[cold]
#[inline(never)]
fn exact_columns(set: &Set<'_>, columns: &mut Columns<'_>) {
    let exactly = Exactly::of(set.x, set.eps, set.centred);
    for k in 0..set.x.len() {
        let y = exactly.value(set.x[k], None, 0.0);
        let (product, error) = split::<false>(y, set.g[k]);
        columns.add(k, product, error);
    }
}

/// A row's input gradients in plain float64 for a float32 output, each with
/// a bound on how far its roundings can have moved it, into `out` where it
/// is given: whether every bound places its value clear of the halfway
/// points around it and the statistics are close enough for the weight's
/// terms, which are then added to `columns`. The upstream times the weight
/// is exact, a product of two float32s.
///
/// The means come from carried sums, each within a few units of its own last
/// place, and `C` from the carried moment `Σ h x`: `Σ (h - A)(x - m)` with
/// float64 means `A` and `m` is the exact one but for `n (A - a)(m - μ)`, a
/// product of their two errors, since deviations off the exact means sum to
/// nothing. Each deviation `d` is then within `mean_error + |d| 2^-53` of the
/// exact one and each `u` likewise, so that every bound below is a row's
/// constants against `|d|` and `|v|`. The margins cover the bounds' own
/// roundings.
fn plain_settled<T: Output>(
    set: &Set<'_>,
    scratch: &mut [f64],
    out: Option<&mut [T]>,
    columns: &mut Columns<'_>,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { plain_settled_avx2(set, scratch, out, columns) };
    }
    plain_settled_body(set, scratch, out, columns)
}

/// [`plain_settled`] built with AVX2, four float64s to a register. Rust
/// neither fuses nor reorders the arithmetic, so every value and bound is
/// the same to the bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn plain_settled_avx2<T: Output>(
    set: &Set<'_>,
    scratch: &mut [f64],
    out: Option<&mut [T]>,
    columns: &mut Columns<'_>,
) -> bool {
    plain_settled_body(set, scratch, out, columns)
}

#[inline(always)]
fn plain_settled_body<T: Output>(
    set: &Set<'_>,
    scratch: &mut [f64],
    out: Option<&mut [T]>,
    columns: &mut Columns<'_>,
) -> bool {
    let len = set.x.len();
    let (h, squares) = scratch[..2 * len].split_at_mut(len);
    let (x, g) = (&set.x[..len], &set.g[..len]);
    match set.weight {
        Some(weight) => {
            for ((h, &g), &w) in h.iter_mut().zip(g).zip(&weight[..len]) {
                *h = g * w;
            }
        }
        None => {
            for (h, &g) in h.iter_mut().zip(g) {
                *h = g * set.gain;
            }
        }
    }
    let h = &*h;
    let n = len as f64;
    let unit = 2f64.powi(-53);
    let margin = 1.0 + 2f64.powi(-20);
    let (sum_x, sum_h) = (<f64 as Element>::block(x), <f64 as Element>::block(h));
    // A carried sum over the count, and how far its leading part can be from
    // the exact quotient.
    let mean_of = |total: Carried| -> Option<(f64, f64)> {
        if !set.centred {
            return Some((0.0, 0.0));
        }
        let (value, rest, bound, ordinary) = total.quotient_parts::<false>(n);
        (ordinary || value == 0.0).then_some((value, (rest.abs() + bound) * margin))
    };
    let (Some((mean, mean_error)), Some((upstream, upstream_error))) =
        (mean_of(sum_x), mean_of(sum_h))
    else {
        return false;
    };
    // The squares of the deviations off the float64 mean, each rounded twice
    // -- within `3 2^-53` of itself -- and their carried sum; off the exact
    // mean they sum to `n (m - μ)^2` less.
    for (square, &x) in squares.iter_mut().zip(x) {
        let d = x - mean;
        *square = d * d;
    }
    let carried = <f64 as Element>::block(squares);
    let (squares, _) = two_sum(carried.sum, carried.error);
    let q = squares / n + set.eps;
    let q_error = (4.0 * unit * squares + carried.slack + n * mean_error * mean_error) / n * margin
        + 2.0 * unit * q;
    if !(q > 0.0 && q.is_finite() && q_error <= q * 2f64.powi(-30)) {
        return false;
    }
    // `r` relative to the exact inverse root: half the variance's, and the
    // root's and the quotient's roundings.
    let r = 1.0 / q.sqrt();
    let relative = (0.5 * (q_error / q) * (1.0 + q_error / q) + 2.0 * unit) * margin;
    // The weight's terms want each `y` well inside a float32's last place.
    let close = mean_error * r <= 2f64.powi(-40);
    if columns.wanted() && !close {
        return false;
    }
    let mut clear = true;
    if let Some(out) = out {
        let out = &mut out[..len];
        // `n C = Σ h x - m Σ h - a Σ x + n a m`, in pairs from the carried
        // totals, each product of pairs within 2^-104 of itself.
        let moment = <f64 as Element>::carried_dot(h, x);
        let pair = |total: Carried| {
            let (sum, error) = two_sum(total.sum, total.error);
            Pair(sum, error)
        };
        let (first, second) = split::<false>(n, upstream);
        let terms = [
            pair(moment),
            pair(sum_h).mul::<false>(Pair(-mean, 0.0)),
            pair(sum_x).mul::<false>(Pair(-upstream, 0.0)),
            Pair(first, second).mul::<false>(Pair(mean, 0.0)),
        ];
        let size: f64 = terms.iter().map(|term| term.0.abs() + term.1.abs()).sum();
        let total = terms[0].add(terms[1]).add(terms[2]).add(terms[3]);
        let slack = moment.slack + mean.abs() * sum_h.slack + upstream.abs() * sum_x.slack;
        let c = total.0 / n;
        let c_error = ((slack + size * 2f64.powi(-98) + total.1.abs()) / n
            + unit * c.abs()
            + upstream_error * mean_error)
            * margin;
        let square = r * r;
        let square_relative = 2.0 * relative + relative * relative + unit;
        let t = square * c;
        let t_error =
            (square * c_error + (c.abs() + c_error) * square * square_relative + unit * t.abs())
                * margin;
        // `v = u - t d` is within `A0 + A1 |d| + A2 |v|` of the exact
        // value, and the slope `r v` within `B0 + B1 |d| + B2 |v|`.
        let a0 = upstream_error + t.abs() * mean_error + mean_error * t_error;
        let a1 = 3.0 * unit * t.abs() + t_error;
        let a2 = 2.0 * unit;
        let scaled = r * (1.0 + relative) * margin;
        let (b0, b1) = (scaled * a0, scaled * a1);
        let b2 = r * ((1.0 + relative) * a2 + relative + 2.0 * unit) * margin;
        if set.centred && h.iter().all(|&value| value == h[0]) {
            // `u` is zero throughout, and so, exactly, is every gradient.
            out.fill(T::narrow(0.0));
        } else {
            // Settled where both ends of the interval round to the same
            // value, so that every number between them does: rounding is
            // monotonic. The ends, rounded themselves, are pushed out by
            // twice what that can take from them, and a zero's sign counts.
            for ((cell, &x), &h) in out.iter_mut().zip(x).zip(h) {
                let d = x - mean;
                let v = (h - upstream) - t * d;
                let slope = r * v;
                let bound = b0 + b1 * d.abs() + b2 * v.abs() + 2f64.powi(-51) * slope.abs();
                let (low, high) = (T::narrow(slope - bound), T::narrow(slope + bound));
                *cell = high;
                clear &= low.widen().to_bits() == high.widen().to_bits();
            }
        }
    }
    if clear && columns.wanted() {
        for (k, (&x, &g)) in x.iter().zip(g).enumerate() {
            columns.add(k, g * ((x - mean) * r), 0.0);
        }
    }
    clear
}

/// The row's input gradients rounded into `out`: each the pass settled, and
/// the rest in whole numbers.
fn round_row<T: Output>(
    set: &Set<'_>,
    parts: &Parts<'_>,
    outcome: Outcome,
    out: &mut [T],
    open: &mut Vec<usize>,
) {
    match outcome {
        Outcome::Plain => {
            for (cell, &value) in out.iter_mut().zip(parts.value.iter()) {
                *cell = T::narrow(value);
            }
        }
        Outcome::Exact => exact_slopes(set, parts, None, out),
        Outcome::Pairs => {
            open.clear();
            for (k, cell) in out.iter_mut().enumerate() {
                let (value, clear) = T::settle_lanes(parts.value[k], parts.rest[k], parts.bound[k]);
                *cell = value;
                if !clear {
                    open.push(k);
                }
            }
            if !open.is_empty() {
                exact_slopes(set, parts, Some(open), out);
            }
        }
    }
}

/// The gradients of `values` standardized over themselves, then times
/// `gain`, under `grad`: the input's into `out`, each the value nearest its
/// exact one, and the weight's, `Σ g y` where `want_weight`, returned as the
/// pair the terms' pairs sum to. A slice holding a value that is not finite
/// takes the formula's float64 values. For a float32 output the values, the
/// upstream gradient and the gain are float32s. `scratch` is the caller's,
/// kept from one slice to the next.
pub(crate) fn standardized_slice_slopes<T: Output>(
    values: &[f64],
    grad: &[f64],
    (gain, eps): (f64, f64),
    out: &mut [T],
    (want_weight, scratch): (bool, &mut Vec<f64>),
) -> (f64, f64) {
    let len = values.len();
    let set = Set {
        x: values,
        g: grad,
        weight: None,
        gain,
        eps,
        centred: true,
    };
    let mut total = (0.0f64, 0.0f64);
    let mut columns = if want_weight {
        Columns::Total(&mut total)
    } else {
        Columns::Unwanted
    };
    // The plain pass's two rows, and the pairs' five where it does not settle.
    let rows = |scratch: &mut Vec<f64>, count: usize| {
        if scratch.len() < count * len {
            scratch.resize(count * len, 0.0);
        }
    };
    rows(scratch, if T::PLAIN { 2 } else { 5 });
    if !(T::PLAIN && plain_settled(&set, scratch, Some(&mut *out), &mut columns)) {
        rows(scratch, 5);
        let (head, scratch_rest) = scratch.split_at_mut(len);
        let (tail, scratch_rest) = scratch_rest.split_at_mut(len);
        let (value, scratch_rest) = scratch_rest.split_at_mut(len);
        let (rest, bound) = scratch_rest[..2 * len].split_at_mut(len);
        let mut parts = Parts {
            head,
            tail,
            value,
            rest,
            bound,
        };
        let outcome = row_slopes_fused(&set, true, &mut parts, &mut columns);
        if outcome == Outcome::Exact {
            exact_columns(&set, &mut columns);
        }
        round_row::<T>(&set, &parts, outcome, out, &mut Vec::new());
    }
    two_sum(total.0, total.1)
}

/// A pair `value + rest`, a two-sum's, rounded once to `T`.
pub(crate) fn round_pair<T: Output>((value, rest): (f64, f64)) -> T {
    T::settle(value, rest, 0.0).unwrap_or_else(|| T::narrow(value))
}

/// The gradients of each `norm`-wide row of `input` standardized -- less
/// its mean where `centred` -- under the upstream gradient `grad` and the
/// weight: the input's into `input_grad` where given, each the value nearest
/// its exact one, and, returned where `want_weight`, the weight's, as
/// two-sums' pairs. The weight is `period` rows of a value for each run of
/// `run` columns -- which divides `norm` -- row `r` taking row `r % period`,
/// and each of its gradients sums the terms `g y` of every value it weighs.
pub(crate) fn standardized_slopes<I: Element, T: Output>(
    input: &[I],
    grad: &[I],
    weight: Option<&[I]>,
    (norm, period, run): (usize, usize, usize),
    (eps, centred): (f64, bool),
    input_grad: Option<&mut [T]>,
    want_weight: bool,
) -> Vec<(f64, f64)> {
    if norm == 0 {
        return Vec::new();
    }
    let (period, run) = (period.max(1), run.max(1));
    let (rows, width) = (input.len() / norm, norm / run);
    let widened: Option<Vec<f64>> = weight.map(|w| {
        w.iter()
            .flat_map(|&v| std::iter::repeat_n(v.wide(), run))
            .collect()
    });
    // Rows a task takes, from the shape alone, as the column sums' grouping
    // decides their rounding; and few enough tasks that their partial sums
    // stay small beside the gradient.
    let per_task = crate::ops::map::outputs_per_task(norm * 16)
        .max(rows.div_ceil(256))
        .max(1);
    let task = |first: usize, count: usize, mut out: Option<&mut [T]>| -> Vec<(f64, f64)> {
        let mut sums = vec![(0.0f64, 0.0f64); if want_weight { period * width } else { 0 }];
        // A row's terms in place, where a run is wider than one, each run
        // summed into its pair once the row is done.
        let mut terms = vec![0.0f64; if want_weight && run > 1 { 2 * norm } else { 0 }];
        let mut scratch = vec![0.0f64; 4 * norm];
        let (x, scratch_rest) = scratch.split_at_mut(norm);
        let (g, work) = scratch_rest.split_at_mut(norm);
        // The pairs' buffers, made for the first row that needs them.
        let mut pairs: Vec<f64> = Vec::new();
        let mut open = Vec::new();
        for row in first..first + count {
            let at = row * norm;
            for ((x, g), (&input, &grad)) in x
                .iter_mut()
                .zip(g.iter_mut())
                .zip(input[at..at + norm].iter().zip(&grad[at..at + norm]))
            {
                *x = input.wide();
                *g = grad.wide();
            }
            let slot = row % period;
            let set = Set {
                x: &*x,
                g: &*g,
                weight: widened.as_deref().map(|w| &w[slot * norm..][..norm]),
                gain: 1.0,
                eps,
                centred,
            };
            let mut columns = match (want_weight, run) {
                (false, _) => Columns::Unwanted,
                (true, 1) => Columns::Each(&mut sums[slot * norm..][..norm]),
                (true, _) => Columns::Row(&mut terms),
            };
            'row: {
                if T::PLAIN {
                    let cells = out
                        .as_deref_mut()
                        .map(|out| &mut out[(row - first) * norm..][..norm]);
                    if plain_settled(&set, work, cells, &mut columns) {
                        break 'row;
                    }
                }
                let head = &mut work[..norm];
                if pairs.is_empty() {
                    pairs = vec![0.0f64; 4 * norm];
                }
                let (tail, pairs_rest) = pairs.split_at_mut(norm);
                let (value, pairs_rest) = pairs_rest.split_at_mut(norm);
                let (rest, bound) = pairs_rest.split_at_mut(norm);
                let mut parts = Parts {
                    head: &mut *head,
                    tail,
                    value,
                    rest,
                    bound,
                };
                let outcome = row_slopes_fused(&set, out.is_some(), &mut parts, &mut columns);
                if outcome == Outcome::Exact {
                    exact_columns(&set, &mut columns);
                }
                if let Some(out) = out.as_deref_mut() {
                    let cells = &mut out[(row - first) * norm..][..norm];
                    round_row(&set, &parts, outcome, cells, &mut open);
                }
            }
            if !terms.is_empty() {
                let runs = &mut sums[slot * width..][..width];
                for (sum, terms) in runs.iter_mut().zip(terms.chunks(2 * run)) {
                    if run < 32 {
                        // Term by term: a carried pass over a run this short
                        // costs more to start and merge than it saves.
                        for &[head, tail] in terms.as_chunks::<2>().0 {
                            let (lead, carry) = two_sum(sum.0, head);
                            *sum = (lead, sum.1 + (carry + tail));
                        }
                        continue;
                    }
                    let carried = <f64 as Element>::block(terms);
                    let (lead, carry) = two_sum(sum.0, carried.sum);
                    *sum = (lead, sum.1 + (carry + carried.error));
                }
            }
        }
        sums
    };
    let partials = match input_grad {
        Some(out) => {
            crate::ops::map::par_out_chunks_mapped(out, per_task * norm, &|start, chunk| {
                task(start / norm, chunk.len() / norm, Some(chunk))
            })
        }
        None => {
            let mut marks = vec![(); rows];
            crate::ops::map::par_out_chunks_mapped(&mut marks, per_task, &|start, chunk| {
                task(start, chunk.len(), None)
            })
        }
    };
    if !want_weight {
        return Vec::new();
    }
    let mut total = vec![(0.0f64, 0.0f64); period * width];
    for part in partials {
        for (sum, (head, tail)) in total.iter_mut().zip(part) {
            let (lead, carry) = two_sum(sum.0, head);
            *sum = (lead, sum.1 + (carry + tail));
        }
    }
    total
        .into_iter()
        .map(|(head, tail)| two_sum(head, tail))
        .collect()
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

    /// Every gradient the bound settles agrees with the one taken in whole
    /// numbers, which is the definition, whether the row is ordinary, sits
    /// far from zero against its spread, or has an upstream gradient lying
    /// all but in the span of one and the standardized values.
    #[test]
    fn the_settled_gradients_are_the_exact_ones() {
        let mut next = generator(0x2545_f491_4f6c_dd1d);
        for trial in 0..48 {
            let width = 2 + trial % 9;
            let mut x: Vec<f64> = (0..width).map(|_| next()).collect();
            if trial % 4 == 1 {
                for value in &mut x {
                    *value += 1e6;
                }
            }
            let mut g: Vec<f64> = (0..width).map(|_| next()).collect();
            if trial % 3 == 2 {
                let mean = x.iter().sum::<f64>() / width as f64;
                for (gv, &xv) in g.iter_mut().zip(&x) {
                    *gv = 0.75 + 1.25 * (xv - mean) + *gv * 1e-9;
                }
            }
            let weight: Vec<f64> = (0..width).map(|_| 1.0 + next()).collect();
            for &centred in &[true, false] {
                let eps = if trial % 5 == 0 { 0.0 } else { 1e-5 };
                let set = Set {
                    x: &x,
                    g: &g,
                    weight: (trial % 2 == 0).then_some(weight.as_slice()),
                    gain: 1.0,
                    eps,
                    centred,
                };
                let mut scratch = vec![0.0f64; 5 * width];
                let (head, scratch_rest) = scratch.split_at_mut(width);
                let (tail, scratch_rest) = scratch_rest.split_at_mut(width);
                let (value, scratch_rest) = scratch_rest.split_at_mut(width);
                let (rest, bound) = scratch_rest.split_at_mut(width);
                let mut parts = Parts {
                    head,
                    tail,
                    value,
                    rest,
                    bound,
                };
                let outcome = row_slopes::<false>(&set, true, &mut parts, &mut Columns::Unwanted);
                let mut exact = vec![0.0f64; width];
                exact_slopes::<f64>(&set, &parts, None, &mut exact);
                if outcome != Outcome::Pairs {
                    continue;
                }
                for k in 0..width {
                    let (value, clear) =
                        f64::settle_lanes(parts.value[k], parts.rest[k], parts.bound[k]);
                    if clear {
                        assert_eq!(
                            value.to_bits(),
                            exact[k].to_bits(),
                            "{x:?} {g:?} {centred} {k}"
                        );
                    }
                }
            }
        }
    }

    /// An upstream gradient that is the same everywhere leaves a centred
    /// row with exactly nothing, whatever the weight is.
    #[test]
    fn a_constant_upstream_gradient_leaves_nothing() {
        let x = [0.25, -1.5, 3.0, 0.125, 7.0];
        let g = [0.1; 5];
        let weight = [1.0 / 3.0; 5];
        let mut out = vec![1.0f64; 5];
        let gw = standardized_slopes::<f64, f64>(
            &x,
            &g,
            Some(&weight),
            (5, 1, 1),
            (1e-5, true),
            Some(&mut out),
            true,
        );
        assert!(
            out.iter().all(|&v| v == 0.0 && v.is_sign_positive()),
            "{out:?}"
        );
        assert_eq!(gw.len(), 5);
    }
}
