// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Piecewise linear interpolation, each value the float64 nearest the exact
//! one; see `arithmetic::exact_sum::interpolated`.

use std::cmp::Ordering;
use std::sync::Arc;

use crate::{
    error::{MinitensorError, Result},
    ops::arithmetic::exact_sum::{exact_ratio, interpolated},
    ops::arithmetic::rounded_once::two_sum,
    tensor::{DataType, Tensor, TensorData},
};

/// Queries a task takes.
const CHUNK: usize = 1024;

/// The piecewise linear interpolation of the samples `(points, values)` at
/// each of `query`, all float64, `points` increasing.
///
/// Between two samples the value is the float64 nearest
/// `(y_lo (x_hi - x) + y_hi (x - x_lo)) / (x_hi - x_lo)`, taken exactly; on
/// a sample it is that sample's value, the last of several at one position.
/// Below the first sample it is `left` and past the last `right`, each the
/// end sample's value where not given. With a `period` the queries and the
/// samples are wrapped onto `[0, period)` exactly, and the first sample
/// follows the last a period on: `left` and `right` do not apply. A NaN
/// query, or an infinite one with a period, gives NaN, and a sample value
/// that is not finite gives what the weighted form gives in float64.
pub fn interp(
    query: &Tensor,
    points: &Tensor,
    values: &Tensor,
    left: Option<f64>,
    right: Option<f64>,
    period: Option<f64>,
) -> Result<Tensor> {
    for tensor in [query, points, values] {
        if tensor.dtype() != DataType::Float64 {
            return Err(MinitensorError::invalid_operation(format!(
                "interp takes float64 queries and samples, got {:?}",
                tensor.dtype()
            )));
        }
    }
    let count = points.numel();
    if points.ndim() != 1 || values.ndim() != 1 || values.numel() != count || count == 0 {
        return Err(MinitensorError::invalid_operation(format!(
            "interp takes one value for each of at least one sample point, got {:?} and {:?}",
            points.shape().dims(),
            values.shape().dims()
        )));
    }
    if period.is_some_and(|period| !period.is_finite() || period == 0.0) {
        return Err(MinitensorError::invalid_operation(
            "interp takes a finite, non-zero period",
        ));
    }
    let query = query.contiguous()?;
    let (points, values) = (points.contiguous()?, values.contiguous()?);
    let missing = || MinitensorError::internal_error("interp: a float64 slice");
    let at = query.data().as_f64_slice().ok_or_else(missing)?;
    let xp = points.data().as_f64_slice().ok_or_else(missing)?;
    let fp = values.data().as_f64_slice().ok_or_else(missing)?;
    let mut data = TensorData::uninitialized_on_device(at.len(), DataType::Float64, query.device());
    let out = data.as_f64_slice_mut().ok_or_else(missing)?;
    match period {
        None => {
            let line = Line {
                xp,
                fp,
                left: left.unwrap_or(fp[0]),
                right: right.unwrap_or(fp[count - 1]),
            };
            fill(out, at, &line);
        }
        Some(period) => fill(out, at, &Circle::new(xp, fp, period.abs())),
    }
    Ok(Tensor::new(
        Arc::new(data),
        query.shape().clone(),
        DataType::Float64,
        query.device(),
        false,
    ))
}

/// Samples a query is interpolated between, with a product's error one
/// fused multiply-add where `FMA`.
trait Samples: Sync {
    fn at<const FMA: bool>(&self, x: f64) -> f64;
}

/// `out[i]` the samples' interpolation at `query[i]`, across the pool for
/// many queries, a chunk at a time in the widest build the CPU runs.
fn fill<S: Samples>(out: &mut [f64], query: &[f64], samples: &S) {
    #[cfg(target_arch = "x86_64")]
    let fused = crate::ops::simd::simd_capabilities().avx2_fma;
    crate::ops::map::par_out_chunks_gated(out, CHUNK, 16 * CHUNK, &|first, chunk| {
        let query = &query[first..first + chunk.len()];
        #[cfg(target_arch = "x86_64")]
        if fused {
            // SAFETY: fma was detected on this CPU.
            unsafe { fill_fused(chunk, query, samples) };
            return;
        }
        fill_chunk::<S, false>(chunk, query, samples);
    });
}

/// [`fill_chunk`] compiled with fused multiply-adds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn fill_fused<S: Samples>(out: &mut [f64], query: &[f64], samples: &S) {
    fill_chunk::<S, true>(out, query, samples);
}

/// `out[i]` the samples' interpolation at `query[i]`, for one chunk.
#[inline(always)]
fn fill_chunk<S: Samples, const FMA: bool>(out: &mut [f64], query: &[f64], samples: &S) {
    for (slot, &x) in out.iter_mut().zip(query) {
        *slot = samples.at::<FMA>(x);
    }
}

/// Samples along a line, held at `left` below the first and `right` past
/// the last.
struct Line<'a> {
    xp: &'a [f64],
    fp: &'a [f64],
    left: f64,
    right: f64,
}

impl Samples for Line<'_> {
    #[inline(always)]
    fn at<const FMA: bool>(&self, x: f64) -> f64 {
        on_line::<FMA>(x, self.xp, self.fp, self.left, self.right)
    }
}

/// The interpolation at `x` of samples along a line.
#[inline(always)]
fn on_line<const FMA: bool>(x: f64, xp: &[f64], fp: &[f64], left: f64, right: f64) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return f64::NAN;
    }
    if x < xp[0] {
        return left;
    }
    if x > xp[n - 1] {
        return right;
    }
    // The samples at or before `x`, which the last of lies on or past.
    let upper = xp.partition_point(|&p| p <= x).max(1);
    if upper == n {
        return fp[n - 1];
    }
    let lower = upper - 1;
    if x == xp[lower] {
        return fp[lower];
    }
    let (x_lo, x_hi, y_lo, y_hi) = (xp[lower], xp[upper], fp[lower], fp[upper]);
    let finite = [x, x_lo, x_hi, y_lo, y_hi]
        .iter()
        .all(|value| value.is_finite());
    if !finite || !(x_lo < x && x < x_hi) {
        return weighted(y_lo, y_hi, x_hi - x, x - x_lo, x_hi - x_lo);
    }
    let (a, b, d) = (two_sum(x_hi, -x), two_sum(x, -x_lo), two_sum(x_hi, -x_lo));
    let split = a.0.is_finite() && b.0.is_finite() && d.0.is_finite();
    split
        .then(|| interpolated::<FMA>(y_lo, y_hi, a, b, d))
        .flatten()
        .unwrap_or_else(|| {
            exact_ratio(
                &[y_lo, y_lo, y_hi, y_hi],
                &[x_hi, -x, x, -x_lo],
                &[x_hi, -x_lo],
            )
        })
}

/// `(y_lo · a + y_hi · b) / d` in float64, for what is not finite: an
/// infinite sample outweighs a finite one, and opposite infinities or an
/// infinite width make NaN.
fn weighted(y_lo: f64, y_hi: f64, a: f64, b: f64, d: f64) -> f64 {
    (y_lo * a + y_hi * b) / d
}

/// A place on a circle of circumference `span`, `at + turns · span`
/// exactly: `at` a remainder and `turns` the whole periods past it, and
/// `key + rest` the same place as a two-sum's pair, the float64 nearest it
/// and what is left. Every place is in range but the first sample's a
/// period on, which for a span past half the largest float may have an
/// infinite key; its `at` and `turns` still hold it.
#[derive(Clone, Copy, Debug)]
struct Place {
    key: f64,
    rest: f64,
    at: f64,
    turns: i32,
}

impl Place {
    fn new(at: f64, turns: i32, span: f64) -> Self {
        debug_assert!((-1..=2).contains(&turns));
        let (key, rest) = two_sum(at, f64::from(turns) * span);
        Self {
            key,
            rest,
            at,
            turns,
        }
    }

    /// The same place `turns` periods on.
    fn turned(self, turns: i32, span: f64) -> Self {
        Self::new(self.at, self.turns + turns, span)
    }

    /// The place as the exact sum of at most three floats.
    fn parts(self, span: f64) -> ([f64; 3], usize) {
        if self.key.is_finite() {
            return (
                [self.key, self.rest, 0.0],
                1 + usize::from(self.rest != 0.0),
            );
        }
        let whole = if self.turns < 0 { -span } else { span };
        let mut parts = [self.at, whole, whole];
        let len = 1 + self.turns.unsigned_abs().min(2) as usize;
        parts[len..].fill(0.0);
        (parts, len)
    }
}

/// `value` wrapped onto `[0, span)`: its remainder, exact, and a turn more
/// for a negative one.
fn wrap(value: f64, span: f64) -> Place {
    let at = value % span;
    Place::new(at, i32::from(at < 0.0), span)
}

/// How `x` compares with `y`, exactly: by their keys, which rounding keeps
/// in order, and between equal keys by what is left of each.
fn compare(x: Place, y: Place) -> Ordering {
    match x.key.partial_cmp(&y.key) {
        Some(Ordering::Equal) => x.rest.partial_cmp(&y.rest).unwrap_or(Ordering::Equal),
        order => order.unwrap_or(Ordering::Equal),
    }
}

/// `x - y` for two places, as the exact sum of at most six floats.
struct Terms {
    terms: [f64; 6],
    len: usize,
}

impl Terms {
    fn between(x: Place, y: Place, span: f64) -> Self {
        let mut terms = [0f64; 6];
        let (high, high_len) = x.parts(span);
        let (low, low_len) = y.parts(span);
        terms[..high_len].copy_from_slice(&high[..high_len]);
        for (slot, &part) in terms[high_len..].iter_mut().zip(&low[..low_len]) {
            *slot = -part;
        }
        Self {
            terms,
            len: high_len + low_len,
        }
    }

    fn terms(&self) -> &[f64] {
        &self.terms[..self.len]
    }

    /// The sum as float64 adds it.
    fn rounded(&self) -> f64 {
        self.terms().iter().sum()
    }
}

/// `x - y` as a two-sum's pair, where two floats hold it: of the keys alone
/// for places whose keys are the whole of them, and otherwise from the
/// terms grown into an expansion of parts that do not overlap, of which no
/// more than two may be nonzero.
fn difference(x: Place, y: Place, span: f64) -> Option<(f64, f64)> {
    if x.rest == 0.0 && y.rest == 0.0 {
        let pair = two_sum(x.key, -y.key);
        return pair.0.is_finite().then_some(pair);
    }
    let terms = Terms::between(x, y, span);
    let mut parts = [0f64; 6];
    for (count, &term) in terms.terms().iter().enumerate() {
        let mut carry = term;
        for part in &mut parts[..count] {
            let (sum, error) = two_sum(carry, *part);
            *part = error;
            carry = sum;
        }
        parts[count] = carry;
    }
    let parts = &parts[..terms.len];
    if !parts.iter().all(|part| part.is_finite()) {
        return None;
    }
    let mut nonzero = parts.iter().copied().filter(|&part| part != 0.0);
    match (nonzero.next(), nonzero.next(), nonzero.next()) {
        (None, _, _) => Some((0.0, 0.0)),
        (Some(only), None, _) => Some((only, 0.0)),
        // Smaller first, as the expansion grows.
        (Some(small), Some(large), None) => Some(two_sum(large, small)),
        _ => None,
    }
}

/// Samples wrapped onto a circle, in order, with the last a period before
/// the first and the first a period past the last.
struct Circle {
    span: f64,
    places: Vec<Place>,
    /// The places' keys, side by side for the search.
    keys: Vec<f64>,
    values: Vec<f64>,
    /// A sample position that is not finite has no place on the circle.
    undefined: bool,
}

impl Samples for Circle {
    #[inline(always)]
    fn at<const FMA: bool>(&self, x: f64) -> f64 {
        self.wrapped::<FMA>(x)
    }
}

impl Circle {
    fn new(xp: &[f64], fp: &[f64], span: f64) -> Self {
        let n = xp.len();
        let undefined = xp.iter().any(|point| !point.is_finite());
        let wrapped: Vec<Place> = xp.iter().map(|&point| wrap(point, span)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        if !undefined {
            // Stable, so samples at one place keep their order.
            order.sort_by(|&i, &j| compare(wrapped[i], wrapped[j]));
        }
        let (first, last) = (order[0], order[n - 1]);
        let mut places = Vec::with_capacity(n + 2);
        places.push(wrapped[last].turned(-1, span));
        places.extend(order.iter().map(|&i| wrapped[i]));
        places.push(wrapped[first].turned(1, span));
        let mut values = Vec::with_capacity(n + 2);
        values.push(fp[last]);
        values.extend(order.iter().map(|&i| fp[i]));
        values.push(fp[first]);
        let keys = places.iter().map(|place| place.key).collect();
        Self {
            span,
            places,
            keys,
            values,
            undefined,
        }
    }

    /// The places at or before `x`: those of a smaller key, and of an equal
    /// one those whose rest is no larger.
    fn before(&self, x: Place) -> usize {
        let below = self.keys.partition_point(|&key| key < x.key);
        if self.keys.get(below) != Some(&x.key) {
            return below;
        }
        let equal = below + self.keys[below..].partition_point(|&key| key == x.key);
        below + self.places[below..equal].partition_point(|place| place.rest <= x.rest)
    }

    /// The interpolation at `x`, wrapped.
    #[inline(always)]
    fn wrapped<const FMA: bool>(&self, x: f64) -> f64 {
        if self.undefined || !x.is_finite() {
            return f64::NAN;
        }
        let span = self.span;
        let x = wrap(x, span);
        // The first place lies below zero and the last at a period or past,
        // so `x` lies between two.
        let upper = self.before(x).clamp(1, self.places.len() - 1);
        let lower = upper - 1;
        let (x_lo, x_hi) = (self.places[lower], self.places[upper]);
        let (y_lo, y_hi) = (self.values[lower], self.values[upper]);
        if compare(x_lo, x) == Ordering::Equal {
            return y_lo;
        }
        if !(y_lo.is_finite() && y_hi.is_finite()) {
            let rounded = |high: Place, low: Place| Terms::between(high, low, span).rounded();
            return weighted(
                y_lo,
                y_hi,
                rounded(x_hi, x),
                rounded(x, x_lo),
                rounded(x_hi, x_lo),
            );
        }
        let settled = match (
            difference(x_hi, x, span),
            difference(x, x_lo, span),
            difference(x_hi, x_lo, span),
        ) {
            (Some(a), Some(b), Some(d)) => interpolated::<FMA>(y_lo, y_hi, a, b, d),
            _ => None,
        };
        settled.unwrap_or_else(|| {
            let (mut samples, mut parts) = (Vec::with_capacity(12), Vec::with_capacity(12));
            for (sample, high, low) in [(y_lo, x_hi, x), (y_hi, x, x_lo)] {
                for &term in Terms::between(high, low, span).terms() {
                    samples.push(sample);
                    parts.push(term);
                }
            }
            exact_ratio(&samples, &parts, Terms::between(x_hi, x_lo, span).terms())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::Shape;

    fn tensor(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn run(x: Vec<f64>, xp: Vec<f64>, fp: Vec<f64>, period: Option<f64>) -> Vec<f64> {
        let got = interp(&tensor(x), &tensor(xp), &tensor(fp), None, None, period).unwrap();
        got.data().as_f64_slice().unwrap().to_vec()
    }

    #[test]
    fn values_between_and_on_the_samples() {
        let (xp, fp) = (vec![0.0, 1.0, 3.0], vec![0.0, 10.0, -2.0]);
        let got = run(vec![-1.0, 0.0, 0.25, 1.0, 2.0, 3.0, 4.0], xp, fp, None);
        assert_eq!(got, vec![0.0, 0.0, 2.5, 10.0, 4.0, -2.0, -2.0]);
        // `1/3` of the way from 0 to 1, the float64 nearest a third.
        let got = run(vec![1.0], vec![0.0, 3.0], vec![0.0, 1.0], None);
        assert_eq!(got, vec![1.0 / 3.0]);
        // Several samples at one position: the last of them.
        let got = run(
            vec![1.0, 2.0],
            vec![0.0, 1.0, 1.0, 2.0, 2.0],
            vec![0.0, 1.0, 5.0, 6.0, 9.0],
            None,
        );
        assert_eq!(got, vec![5.0, 9.0]);
    }

    #[test]
    fn far_out_of_range_and_specials() {
        // Positions whose products with the values pass the largest float.
        let big = f64::MAX / 2.0;
        let got = run(vec![0.0], vec![-big, big], vec![-big, big], None);
        assert_eq!(got, vec![0.0]);
        let got = run(vec![0.5], vec![0.0, 1.0], vec![f64::INFINITY, 1.0], None);
        assert_eq!(got, vec![f64::INFINITY]);
        assert!(run(vec![f64::NAN], vec![0.0, 1.0], vec![0.0, 1.0], None)[0].is_nan());
        let got = interp(
            &tensor(vec![-1.0, 2.0]),
            &tensor(vec![0.0, 1.0]),
            &tensor(vec![3.0, 4.0]),
            Some(-7.0),
            Some(8.0),
            None,
        )
        .unwrap();
        assert_eq!(got.data().as_f64_slice().unwrap(), &[-7.0, 8.0]);
    }

    #[test]
    fn a_period_wraps_exactly() {
        let (xp, fp) = (vec![0.5, 2.0, 3.5], vec![1.0, 4.0, -2.0]);
        let got = run(vec![2.0, 6.0, -1.0, 4.0], xp.clone(), fp.clone(), Some(4.0));
        // 6 wraps to 2, a sample; -1 to 3, two thirds of the way from 2 to
        // 3.5; 4 to 0, halfway from 3.5 - 4 to 0.5.
        assert_eq!(got, vec![4.0, 4.0, 0.0, -0.5]);
        // A query far out wraps by its exact remainder, 2 here.
        let far = 2f64.powi(60) + 2560.0;
        let got = run(vec![far], vec![0.0, 1.0], vec![0.0, 1.0], Some(3.0));
        assert_eq!(got, vec![0.5]);
        // A negative query a hair below a whole period.
        let got = run(vec![-1e-20], vec![0.0, 0.5], vec![0.0, 1.0], Some(1.0));
        assert_eq!(got, vec![2e-20]);
    }

    #[test]
    fn refuses_what_it_cannot_interpolate() {
        let x = tensor(vec![0.5]);
        assert!(interp(&x, &tensor(vec![]), &tensor(vec![]), None, None, None).is_err());
        assert!(
            interp(
                &x,
                &tensor(vec![0.0]),
                &tensor(vec![1.0, 2.0]),
                None,
                None,
                None
            )
            .is_err()
        );
        let one = (tensor(vec![0.0]), tensor(vec![1.0]));
        assert!(interp(&x, &one.0, &one.1, None, None, Some(0.0)).is_err());
        assert!(interp(&x, &one.0, &one.1, None, None, Some(f64::NAN)).is_err());
    }
}
