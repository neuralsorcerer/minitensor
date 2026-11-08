// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Covariance and correlation matrices rounded once.
//!
//! With each observation weighted by `w` -- one, a count, a reliability `a`,
//! or the product of the two -- and the weights totalling `W`, the
//! covariance of rows `x` and `y` is `(W Σ w x y - Σ w x · Σ w y) / D`: a
//! numerator `n_xy` over `D = W (W - c)` for a correction `c`, or over
//! `D = W² - c Σ w a` where reliabilities shrink the sample. Their
//! correlation is `n_xy / sqrt(n_xx n_yy)`. Each entry here is the float
//! nearest that exact value.
//!
//! The numerators do not move when a row is shifted, so each row is taken
//! less a centre -- a float near its weighted mean where every value lies
//! close enough to it for the difference to be exact, and zero otherwise --
//! and scaled by a power of two into `[1, 2)`, as the weights are, which
//! moves an entry by a power of two and a correlation not at all. The sums
//! are carried with bounds, each weighted value split into an exact pair,
//! the numerator formed as a pair, and the entry settled by a bound. Where
//! it cannot be -- a row of integers too far apart for float64, weights
//! too far apart for the pairs, or an entry on a halfway point -- the entry
//! is formed in whole numbers.

use super::{
    Carried, Element, Output, Scaled, Total, at_least, binary_exponent, carried_with,
    product_error, product_of, shifted, subtract, trimmed, two_sum,
};

/// Work a task takes at least, in products.
const TASK: usize = 1 << 14;

/// Rows to a side of a tile, and columns to a chunk of one: eight rows'
/// chunks, 128 KB, stay in the cache while every pair of them is taken.
const TILE: usize = 4;
const CHUNK: usize = 2048;

/// What small values may leave out of a carried total over `count` of them:
/// their bits, or their products' errors, lost among the subnormals.
fn lost(count: usize) -> f64 {
    count as f64 * f64::from_bits(4)
}

/// A sample's rows and weights, each value exact in float64.
pub(crate) struct Sample<'a> {
    /// `variables` rows of `observations` values each.
    pub(crate) values: &'a [f64],
    /// Where given, each value is `values + lows`, the two exact: integers
    /// whose spread is past `2^53`.
    pub(crate) lows: Option<&'a [f64]>,
    pub(crate) variables: usize,
    pub(crate) observations: usize,
    /// How many times each observation counts.
    pub(crate) frequencies: Option<&'a [f64]>,
    /// How reliable each observation is.
    pub(crate) reliabilities: Option<&'a [f64]>,
    /// `c`, taken from the weights' total in the divisor.
    pub(crate) correction: f64,
}

/// Why a sample has no covariance.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The weights total zero or less.
    NoWeight,
    /// The correction leaves a divisor of zero or less.
    NoDivisor,
}

/// The covariance matrix of `sample`'s rows, or with `correlation` their
/// correlation matrix, into `out`, `variables` by `variables`, each entry
/// the float nearest the exact one. A row with a NaN or an infinity has NaN
/// entries, as does every entry when a weight is not finite; a row whose
/// values are all equal covaries by zero and correlates as NaN.
pub(crate) fn covariances<T: Output>(
    sample: &Sample,
    correlation: bool,
    out: &mut [T],
) -> Result<(), Refusal> {
    let variables = sample.variables;
    debug_assert_eq!(out.len(), variables * variables);
    let weights = match Weights::of(sample) {
        Some(Ok(weights)) => Some(weights),
        Some(Err(())) => {
            out.fill(T::narrow(f64::NAN));
            return Ok(());
        }
        None => None,
    };
    let total = match &weights {
        Some(weights) => weights.total,
        None => (sample.observations as f64, 0.0, 0.0),
    };
    // The divisor, scaled as the weights are; only its sign matters to a
    // correlation, and a sample with none is refused either way.
    let divisor = divisor(sample, weights.as_ref(), total)?;
    let rows = prepared_rows(sample, weights.as_ref());
    let context = Context {
        sample,
        weights: weights.as_ref(),
        total,
        divisor,
        rows: &rows,
        correlation,
    };
    // The diagonal numerators first: a correlation needs them all. A
    // float32 answer has its plain ones, and forms a carried one only for
    // an entry they do not settle.
    let diagonal: Vec<Option<(f64, f64, f64)>> = if T::PLAIN {
        Vec::new()
    } else {
        (0..variables)
            .map(|row| context.numerator(row, row))
            .collect()
    };
    let plain: Vec<Option<(f64, f64)>> = if T::PLAIN {
        (0..variables)
            .map(|row| context.plain_numerator(row, row))
            .collect()
    } else {
        Vec::new()
    };
    // Tiles of `TILE` rows against `TILE`, each pair once, so that the rows
    // of a tile are read from the cache a chunk of columns at a time.
    let blocks = variables.div_ceil(TILE);
    let tiles: Vec<(usize, usize)> = (0..blocks)
        .flat_map(|first| (first..blocks).map(move |second| (first, second)))
        .collect();
    let per_task = (TASK / (TILE * TILE * sample.observations).max(1)).max(1);
    let parts = crate::parallel::Parts::of(out);
    let task = |index: usize| {
        for &(first, second) in &tiles[index * per_task..((index + 1) * per_task).min(tiles.len())]
        {
            let rows = |block: usize| block * TILE..((block + 1) * TILE).min(variables);
            context.tile(
                rows(first),
                rows(second),
                &diagonal,
                &plain,
                &|i, j, value: T| {
                    // SAFETY: each pair writes its own two places, inside `out`.
                    unsafe {
                        *parts.start().add(i * variables + j) = value;
                        *parts.start().add(j * variables + i) = value;
                    }
                },
            );
        }
    };
    let tasks = tiles.len().div_ceil(per_task);
    let work = variables * variables * sample.observations / 2;
    if work < crate::ops::map::PAR_THRESHOLD / 8 {
        (0..tasks).for_each(task);
    } else {
        crate::parallel::for_each_task(tasks, &task);
    }
    Ok(())
}

/// Each observation's weight times a power of two that brings the largest
/// into `[1, 2)`, as `high + low` exactly -- a count times a reliability
/// split by a two-product -- and their total, carried, as a pair and a bound.
struct Weights {
    high: Vec<f64>,
    low: Vec<f64>,
    /// The power of two, as the two in the normal range it is applied as.
    scale: (f64, f64),
    total: (f64, f64, f64),
    /// Whether the weights lie too far apart for the pairs, which leaves
    /// every entry to whole numbers.
    spread: bool,
}

impl Weights {
    /// The weights of `sample`, none for an unweighted one, and an error
    /// where one is not finite.
    fn of(sample: &Sample) -> Option<Result<Self, ()>> {
        let count = sample.observations;
        let weight = |k: usize| -> (f64, f64) {
            match (sample.frequencies, sample.reliabilities) {
                (Some(f), Some(a)) => {
                    let product = f[k] * a[k];
                    (product, product_error(f[k], a[k], product))
                }
                (Some(w), None) | (None, Some(w)) => (w[k], 0.0),
                (None, None) => unreachable!("weighted"),
            }
        };
        if sample.frequencies.is_none() && sample.reliabilities.is_none() {
            return None;
        }
        let (mut high, mut low) = (Vec::with_capacity(count), Vec::with_capacity(count));
        let (mut largest, mut smallest, mut finite) = (0f64, f64::INFINITY, true);
        for k in 0..count {
            let (h, l) = weight(k);
            finite &= h.is_finite() && l.is_finite();
            if h != 0.0 {
                largest = largest.max(h.abs());
                smallest = smallest.min(h.abs());
            }
            high.push(h);
            low.push(l);
        }
        if !finite {
            return Some(Err(()));
        }
        let scale = if largest > 0.0 {
            powers(binary_exponent(largest))
        } else {
            (1.0, 1.0)
        };
        for (h, l) in high.iter_mut().zip(low.iter_mut()) {
            *h = *h * scale.0 * scale.1;
            *l = *l * scale.0 * scale.1;
        }
        let total = pair(carried_with(&high, false).merge(carried_with(&low, false)));
        Some(Ok(Self {
            high,
            low,
            scale,
            total,
            spread: largest > 0.0 && smallest < largest * 2f64.powi(-900),
        }))
    }
}

/// `2^-shift` as two powers of two, each in the normal range.
fn powers(shift: i32) -> (f64, f64) {
    let power = |shift: i32| f64::from_bits(((shift + 1023) as u64) << 52);
    (power(-shift / 2), power(-shift - (-shift / 2)))
}

/// The divisor `D`, scaled as the weights are: `N (N - c)` unweighted;
/// `W (W - c)` with the weights scaled by `2^-s`, and so `c` by it too; or
/// `W² - c 2^-s Σ w a` with reliabilities, as a pair and a bound. An error
/// where the weights total zero or less, or the divisor is.
fn divisor(
    sample: &Sample,
    weights: Option<&Weights>,
    total: (f64, f64, f64),
) -> Result<(f64, f64, f64), Refusal> {
    let correction = sample.correction;
    let Some(weights) = weights else {
        let count = sample.observations as f64;
        let rest = count - correction;
        let product = count * rest;
        let pair = two_sum(product, product_error(count, rest, product));
        return if product > 0.0 {
            Ok((pair.0, pair.1, 0.0))
        } else {
            Err(Refusal::NoDivisor)
        };
    };
    // A divisor the pairs cannot use, which leaves every entry to whole
    // numbers.
    let unusable = (0.0, 0.0, f64::INFINITY);
    if weights.spread {
        // Decided in whole numbers, and the entries left to them.
        let (total, divisor) = exact_divisor(sample);
        return if total.not_positive() {
            Err(Refusal::NoWeight)
        } else if divisor.not_positive() {
            Err(Refusal::NoDivisor)
        } else {
            Ok(unusable)
        };
    }
    // Where the bound leaves the total's sign open, whole numbers decide it.
    let (w1, _, bound) = total;
    if w1 - bound <= 0.0 && (w1 + bound < 0.0 || exact_divisor(sample).0.not_positive()) {
        return Err(Refusal::NoWeight);
    }
    let shifted_correction = correction * weights.scale.0 * weights.scale.1;
    let divisor = match sample.reliabilities {
        None => times(total, minus(total, (shifted_correction, 0.0, 0.0))),
        Some(reliabilities) => {
            let reliable = f64::carried_dot(&weights.high, reliabilities)
                .merge(f64::carried_dot(&weights.low, reliabilities));
            let reliable = Carried {
                slack: reliable.slack + lost(sample.observations),
                ..reliable
            };
            minus(
                times(total, total),
                times((shifted_correction, 0.0, 0.0), pair(reliable)),
            )
        }
    };
    let (d1, _, bound) = divisor;
    if d1 - bound > 0.0 && d1.is_finite() {
        Ok(divisor)
    } else if d1 + bound < 0.0 || exact_divisor(sample).1.not_positive() {
        Err(Refusal::NoDivisor)
    } else {
        // Positive, but too near zero for the pairs.
        Ok(unusable)
    }
}

/// A row ready for the products.
#[derive(Default)]
struct Row {
    /// A NaN or an infinity among the values.
    special: bool,
    /// Every value the same.
    constant: bool,
    /// Whether its entries are left to whole numbers.
    exact: bool,
    /// `(x - centre) · 2^-shift`, each exact but for bits lost among the
    /// subnormals.
    scaled: Vec<f64>,
    /// The weighted values `w · scaled` as `high + low`, within `loose` in
    /// total, for a weighted sample.
    high: Vec<f64>,
    low: Vec<f64>,
    loose: f64,
    shift: i32,
    /// `Σ w · scaled`, carried.
    total: Carried,
}

/// Each of `sample`'s rows, ready for the products.
fn prepared_rows(sample: &Sample, weights: Option<&Weights>) -> Vec<Row> {
    let count = sample.observations;
    let prepare = |row: usize| -> Row {
        let values = &sample.values[row * count..(row + 1) * count];
        let lows = sample
            .lows
            .map(|lows| &lows[row * count..(row + 1) * count]);
        let special = values.iter().any(|value| !value.is_finite());
        let constant = !special
            && values.iter().all(|&value| value == values[0])
            && lows.is_none_or(|lows| lows.iter().all(|&low| low == lows[0]));
        let exact = lows.is_some_and(|lows| lows.iter().any(|&low| low != 0.0))
            || weights.is_some_and(|weights| weights.spread);
        if special || constant || exact {
            return Row {
                special,
                constant,
                exact,
                ..Row::default()
            };
        }
        // A centre near the mean where every value is within half of it of
        // the centre, so that each difference is exact.
        let mean = match weights {
            Some(weights) => plain_dot(values, &weights.high).0 / f64::plain_block(&weights.high).0,
            None => f64::plain_block(values).0 / count as f64,
        };
        let close = mean.is_finite()
            && mean != 0.0
            && values
                .iter()
                .all(|&value| (value - mean).abs() <= mean.abs() * 0.5);
        let centre = if close { mean } else { 0.0 };
        let largest = largest_distance(values, centre);
        let shift = binary_exponent(largest);
        let scale = powers(shift);
        let scaled: Vec<f64> = values
            .iter()
            .map(|&value| (value - centre) * scale.0 * scale.1)
            .collect();
        let (high, low, loose, total) = match weights {
            None => (Vec::new(), Vec::new(), 0.0, carried_with(&scaled, false)),
            Some(weights) => {
                let mut high = Vec::with_capacity(count);
                let mut low = Vec::with_capacity(count);
                let mut magnitude = 0f64;
                for ((&value, &h), &l) in scaled.iter().zip(&weights.high).zip(&weights.low) {
                    let product = h * value;
                    high.push(product);
                    low.push(product_error(h, value, product) + l * value);
                    magnitude += product.abs();
                }
                // The low product and its sum each within a rounding of a
                // value under `2^-52` of the high one.
                let loose = magnitude * 2f64.powi(-103) + lost(count);
                let total = carried_with(&high, false).merge(carried_with(&low, false));
                (high, low, loose, total)
            }
        };
        let total = Carried {
            slack: total.slack + loose + lost(count),
            ..total
        };
        Row {
            scaled,
            high,
            low,
            loose,
            shift,
            total,
            ..Row::default()
        }
    };
    if sample.variables * count < crate::ops::map::PAR_THRESHOLD / 8 {
        (0..sample.variables).map(prepare).collect()
    } else {
        crate::parallel::map_tasks(sample.variables, &prepare)
    }
}

/// What every entry needs.
struct Context<'a> {
    sample: &'a Sample<'a>,
    weights: Option<&'a Weights>,
    total: (f64, f64, f64),
    divisor: (f64, f64, f64),
    rows: &'a [Row],
    correlation: bool,
}

impl Context<'_> {
    /// Whether rows `i` and `j` are ones the pairs can take.
    fn usable(&self, i: usize, j: usize) -> bool {
        let (x, y) = (&self.rows[i], &self.rows[j]);
        !(x.special || x.constant || x.exact || y.special || y.constant || y.exact)
    }

    /// `Σ w x y` of rows `i` and `j` as scaled, over the columns `range`,
    /// carried.
    fn products(&self, i: usize, j: usize, range: std::ops::Range<usize>) -> Carried {
        let (x, y) = (&self.rows[i], &self.rows[j]);
        let y = &y.scaled[range.clone()];
        match self.weights {
            None => f64::carried_dot(&x.scaled[range], y),
            Some(_) => f64::carried_dot(&x.high[range.clone()], y)
                .merge(f64::carried_dot(&x.low[range], y)),
        }
    }

    /// The same in plain float64 lanes, the low parts left out, and the sum
    /// of the magnitudes.
    fn plain_products(&self, i: usize, j: usize, range: std::ops::Range<usize>) -> (f64, f64) {
        let (x, y) = (&self.rows[i], &self.rows[j]);
        let weighted = if self.weights.is_some() {
            &x.high
        } else {
            &x.scaled
        };
        plain_dot(&weighted[range.clone()], &y.scaled[range])
    }

    /// `W Σ w x y - Σ w x Σ w y` of rows `i` and `j` as scaled, a pair and a
    /// bound, for rows the pairs can take.
    fn numerator(&self, i: usize, j: usize) -> Option<(f64, f64, f64)> {
        self.usable(i, j)
            .then(|| self.numerator_of(i, j, self.products(i, j, 0..self.sample.observations)))
    }

    /// [`Context::numerator`] of the products of rows `i` and `j`.
    fn numerator_of(&self, i: usize, j: usize, products: Carried) -> (f64, f64, f64) {
        let (x, y) = (&self.rows[i], &self.rows[j]);
        // `y.scaled` is under 2 in magnitude, so what `x`'s pairs leave out
        // moves the products by at most twice that.
        let products = Carried {
            slack: products.slack + 2.0 * x.loose + lost(self.sample.observations),
            ..products
        };
        minus(
            times(self.total, pair(products)),
            times(pair(x.total), pair(y.total)),
        )
    }

    /// The entries of the rows in `first` against those in `second`, each
    /// pair once, handed to `write`: the products of the rows the pairs can
    /// take formed a chunk of columns at a time, plainly for a float32
    /// answer, and each entry finished from them.
    fn tile<T: Output>(
        &self,
        first: std::ops::Range<usize>,
        second: std::ops::Range<usize>,
        diagonal: &[Option<(f64, f64, f64)>],
        plain: &[Option<(f64, f64)>],
        write: &dyn Fn(usize, usize, T),
    ) {
        let count = self.sample.observations;
        // At most `TILE²` pairs, each with whether its products are formed.
        let mut pairs = [(0usize, 0usize, false); TILE * TILE];
        let mut taken = 0;
        for i in first {
            for j in second.clone().filter(|&j| j >= i) {
                pairs[taken] = (i, j, i != j && self.usable(i, j));
                taken += 1;
            }
        }
        let pairs = &pairs[..taken];
        let mut sums = [(0f64, 0f64); TILE * TILE];
        let mut carried = [Carried::default(); TILE * TILE];
        for start in (0..count).step_by(CHUNK) {
            let range = start..(start + CHUNK).min(count);
            for (slot, &(i, j, formed)) in pairs.iter().enumerate() {
                if !formed {
                    continue;
                }
                if T::PLAIN {
                    let (sum, magnitude) = self.plain_products(i, j, range.clone());
                    sums[slot] = (sums[slot].0 + sum, sums[slot].1 + magnitude);
                } else {
                    carried[slot] = carried[slot].merge(self.products(i, j, range.clone()));
                }
            }
        }
        for (slot, &(i, j, formed)) in pairs.iter().enumerate() {
            let products = formed.then_some((sums[slot], carried[slot]));
            write(i, j, self.entry(i, j, products, diagonal, plain));
        }
    }

    /// The entry at rows `i` and `j`, given the plain and the carried
    /// products of a pair off the diagonal that the pairs can take: from the
    /// plain ones for a float32 answer where they settle it -- the carried
    /// ones are not formed then, and are where they do not -- then from the
    /// carried ones, then in whole numbers.
    fn entry<T: Output>(
        &self,
        i: usize,
        j: usize,
        products: Option<((f64, f64), Carried)>,
        diagonal: &[Option<(f64, f64, f64)>],
        plain: &[Option<(f64, f64)>],
    ) -> T {
        let (x, y) = (&self.rows[i], &self.rows[j]);
        if x.special || y.special {
            return T::narrow(f64::NAN);
        }
        if x.constant || y.constant {
            // A zero numerator: no covariance, and no correlation either.
            return T::narrow(if self.correlation { f64::NAN } else { 0.0 });
        }
        if T::PLAIN {
            let numerator = if i == j {
                plain[i]
            } else {
                products.and_then(|(sums, _)| self.plain_numerator_of(i, j, sums))
            };
            if let Some(value) =
                numerator.and_then(|numerator| self.plain_entry(i, j, numerator, plain))
            {
                return value;
            }
        }
        let own = |row: usize| {
            if T::PLAIN {
                self.numerator(row, row)
            } else {
                diagonal[row]
            }
        };
        let numerator = if i == j {
            own(i)
        } else {
            products.map(|(_, carried)| {
                if T::PLAIN {
                    self.numerator_of(i, j, self.products(i, j, 0..self.sample.observations))
                } else {
                    self.numerator_of(i, j, carried)
                }
            })
        };
        let settled = numerator.and_then(|numerator| {
            if self.correlation {
                self.correlation_of(numerator, own(i)?, own(j)?, i == j)
            } else {
                self.covariance_of(numerator, x.shift + y.shift)
            }
        });
        settled.unwrap_or_else(|| exact_entry(self.sample, i, j, self.correlation))
    }

    /// `W Σ w x y - Σ w x Σ w y` of rows `i` and `j` as scaled, from plain
    /// float64 lanes, and a bound on it, for rows the pairs can take.
    ///
    /// The products' sum and each row's are within `(n + 1)u` of their
    /// magnitudes' sums -- a rounding for each product and the depth of the
    /// lanes -- and the low parts a weighted row leaves out within `2.01u`
    /// of its own; `(n + 4)u` takes all of it, and the three roundings of
    /// the numerator come on top.
    fn plain_numerator(&self, i: usize, j: usize) -> Option<(f64, f64)> {
        let usable = self.usable(i, j);
        usable
            .then(|| self.plain_products(i, j, 0..self.sample.observations))
            .and_then(|sums| self.plain_numerator_of(i, j, sums))
    }

    /// [`Context::plain_numerator`] of the plain products of rows `i` and
    /// `j` and the sum of their magnitudes.
    fn plain_numerator_of(
        &self,
        i: usize,
        j: usize,
        (products, magnitude): (f64, f64),
    ) -> Option<(f64, f64)> {
        const U: f64 = 1.0 / (1u64 << 53) as f64;
        let (x, y) = (&self.rows[i], &self.rows[j]);
        let spread = (self.sample.observations as f64 + 4.0) * U * 1.01;
        if spread > 2f64.powi(-20) {
            return None;
        }
        let (w, w_rest, w_bound) = self.total;
        // The rows' sums as carried, each within its bound.
        let ((a, a_rest, a_bound), (b, b_rest, b_bound)) = (pair(x.total), pair(y.total));
        let (a_off, b_off) = (a_rest.abs() + a_bound, b_rest.abs() + b_bound);
        let (p, q) = (w * products, a * b);
        let numerator = p - q;
        let bound = (w.abs() * spread * magnitude
            + (w_rest.abs() + w_bound) * magnitude
            + a.abs() * b_off
            + b.abs() * a_off
            + a_off * b_off
            + (p.abs() + q.abs() + numerator.abs()) * U)
            * 1.01;
        Some((numerator, bound))
    }

    /// [`Context::entry`] from plain sums, for a float32 answer: the
    /// quotient within its bound and a float64 rounding more at each end,
    /// settled where both ends narrow alike and lie on one side of zero.
    fn plain_entry<T: Output>(
        &self,
        i: usize,
        j: usize,
        (numerator, bound): (f64, f64),
        diagonal: &[Option<(f64, f64)>],
    ) -> Option<T> {
        const U: f64 = 1.0 / (1u64 << 53) as f64;
        let relative = |(value, bound): (f64, f64)| bound / value.abs();
        let own = relative((numerator, bound));
        let known = own <= 2f64.powi(-20);
        if !known {
            return None;
        }
        let (value, relative) = if self.correlation {
            if i == j {
                return Some(T::narrow(1.0));
            }
            let ((first, first_bound), (second, second_bound)) = (diagonal[i]?, diagonal[j]?);
            let (first_relative, second_relative) = (
                relative((first, first_bound)),
                relative((second, second_bound)),
            );
            let known = first_relative <= 2f64.powi(-20) && second_relative <= 2f64.powi(-20);
            if !known {
                return None;
            }
            let value = numerator / (first.sqrt() * second.sqrt());
            let relative = own + 0.5 * (first_relative + second_relative) + 4.0 * U;
            (value, relative)
        } else {
            let (d1, d2, d_bound) = self.divisor;
            let rows = &self.rows;
            let scale = powers(-(rows[i].shift + rows[j].shift));
            let value = numerator / d1 * scale.0 * scale.1;
            let relative = own + (d2.abs() + d_bound) / d1.abs() + 2.0 * U;
            (value, relative)
        };
        let bound = value.abs() * (relative * 1.01 + 2.0 * U);
        let rounded = T::narrow(value);
        let settled = value.is_finite()
            && bound < value.abs()
            && T::narrow(value - bound) == rounded
            && T::narrow(value + bound) == rounded;
        settled.then_some(rounded)
    }

    /// The covariance of a numerator over the divisor, scaled back by
    /// `2^shift`, when the bound settles it.
    fn covariance_of<T: Output>(&self, numerator: (f64, f64, f64), shift: i32) -> Option<T> {
        let (value, rest, bound) = over(numerator, self.divisor)?;
        // Both powers move the same way, so a part that underflows on the
        // way leaves a value too small to be taken here.
        let scale = powers(-shift);
        let back = |part: f64| part * scale.0 * scale.1;
        let (value, rest) = (back(value), back(rest));
        let ordinary = value.abs() >= 2f64.powi(-960) && value.abs() <= 2f64.powi(1000);
        if !ordinary {
            return None;
        }
        // And a few least subnormals for the bits of `rest` that did.
        T::settle(value, rest, back(bound) + f64::from_bits(8))
    }

    /// The correlation of a numerator with the rows' own, when the bounds
    /// settle it: one on the diagonal where the numerator is surely positive.
    fn correlation_of<T: Output>(
        &self,
        numerator: (f64, f64, f64),
        first: (f64, f64, f64),
        second: (f64, f64, f64),
        diagonal: bool,
    ) -> Option<T> {
        if diagonal {
            let (value, _, bound) = numerator;
            return (value - bound > 0.0).then(|| T::narrow(1.0));
        }
        let carried = |(sum, error, slack): (f64, f64, f64)| Carried { sum, error, slack };
        Scaled {
            dot: carried(numerator),
            squares: [carried(first), carried(second)],
            scales: [(1.0, 1.0); 2],
        }
        .settled::<T, false>(0.0, 0.0, |_| false)
    }
}

/// The largest `|x - centre|` of finite `values`, in lanes.
fn largest_distance(values: &[f64], centre: f64) -> f64 {
    const LANES: usize = 8;
    let mut lanes = [0f64; LANES];
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let distance = (chunk[lane] - centre).abs();
            lanes[lane] = if distance > lanes[lane] {
                distance
            } else {
                lanes[lane]
            };
        }
    }
    let largest = rest
        .iter()
        .fold(0f64, |largest, &value| largest.max((value - centre).abs()));
    lanes
        .iter()
        .fold(largest, |largest, &lane| largest.max(lane))
}

/// `Σ a b` in plain float64 lanes, and `Σ |a b|`, in the widest build the
/// CPU runs.
fn plain_dot(a: &[f64], b: &[f64]) -> (f64, f64) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx {
        // SAFETY: avx was detected on this CPU.
        return unsafe { plain_dot_avx(a, b) };
    }
    plain_dot_lanes(a, b)
}

/// [`plain_dot`] in four 256-bit registers of sums and four of magnitudes,
/// sixteen values a step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_dot_avx(a: &[f64], b: &[f64]) -> (f64, f64) {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_andnot_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (mut sums, mut magnitudes) = ([_mm256_setzero_pd(); 4], [_mm256_setzero_pd(); 4]);
    let (left, left_rest) = a.as_chunks::<16>();
    let (right, right_rest) = b.as_chunks::<16>();
    for (x, y) in left.iter().zip(right) {
        for quarter in 0..4 {
            // SAFETY: four values from `4 * quarter` lie inside each chunk.
            let (x, y) = unsafe {
                (
                    _mm256_loadu_pd(x.as_ptr().add(4 * quarter)),
                    _mm256_loadu_pd(y.as_ptr().add(4 * quarter)),
                )
            };
            let product = _mm256_mul_pd(x, y);
            sums[quarter] = _mm256_add_pd(sums[quarter], product);
            magnitudes[quarter] =
                _mm256_add_pd(magnitudes[quarter], _mm256_andnot_pd(sign, product));
        }
    }
    let unpack = |vectors: [__m256d; 4]| {
        let mut lanes = [0f64; 16];
        for (quarter, vector) in vectors.iter().enumerate() {
            // SAFETY: four places from `4 * quarter` lie inside `lanes`.
            unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * quarter), *vector) };
        }
        lanes
    };
    let (mut sums, mut magnitudes) = (unpack(sums), unpack(magnitudes));
    for (x, y) in left_rest.iter().zip(right_rest) {
        sums[0] += x * y;
        magnitudes[0] += (x * y).abs();
    }
    (sums.iter().sum(), magnitudes.iter().sum())
}

/// [`plain_dot`] in lanes.
#[inline(always)]
fn plain_dot_lanes(a: &[f64], b: &[f64]) -> (f64, f64) {
    const LANES: usize = 8;
    let (mut sums, mut magnitudes) = ([0f64; LANES], [0f64; LANES]);
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..LANES {
            let product = x[lane] * y[lane];
            sums[lane] += product;
            magnitudes[lane] += product.abs();
        }
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        sums[0] += x * y;
        magnitudes[0] += (x * y).abs();
    }
    (sums.iter().sum(), magnitudes.iter().sum())
}

/// A carried total as a pair and a bound.
fn pair(carried: Carried) -> (f64, f64, f64) {
    let (value, rest) = two_sum(carried.sum, carried.error);
    (value, rest, carried.slack)
}

/// The product of two pairs, each within its bound of what it carries, as
/// a pair within its bound: what the pair leaves out of the pairs' product
/// -- the product of the second parts, and the roundings of the cross terms
/// and their sums -- and what the bounds move it by.
fn times(
    (x1, x2, x_bound): (f64, f64, f64),
    (y1, y2, y_bound): (f64, f64, f64),
) -> (f64, f64, f64) {
    let p1 = x1 * y1;
    let e1 = product_error(x1, y1, p1);
    let (a, b) = (x1 * y2, x2 * y1);
    let cross = a + b;
    let p2 = e1 + cross;
    let (value, rest) = two_sum(p1, p2);
    let left = (x2 * y2).abs() + (a.abs() + b.abs() + cross.abs() + p2.abs()) * 2f64.powi(-53);
    let moved =
        (x1.abs() + x2.abs()) * y_bound + (y1.abs() + y2.abs()) * x_bound + x_bound * y_bound;
    // A few least subnormals for products whose errors fell below them.
    let bound = (left + moved) * (1.0 + 2f64.powi(-50)) + f64::from_bits(16);
    (value, rest, bound)
}

/// The difference of two pairs, each within its bound, as a pair within its
/// bound: the leading parts' difference exact, and the rest rounded twice.
fn minus(
    (x1, x2, x_bound): (f64, f64, f64),
    (y1, y2, y_bound): (f64, f64, f64),
) -> (f64, f64, f64) {
    let (head, rounding) = two_sum(x1, -y1);
    let low = x2 - y2;
    let tail = rounding + low;
    let (value, rest) = two_sum(head, tail);
    let bound =
        (x_bound + y_bound + (low.abs() + tail.abs()) * 2f64.powi(-53)) * (1.0 + 2f64.powi(-50));
    (value, rest, bound)
}

/// The quotient of two pairs, each within its bound, as a pair and a bound,
/// when both lie far enough inside the range for the remainders and are
/// known to `2^-40` of themselves; formed as [`Scaled::settled`] forms its
/// quotient, within `2^-99` of the pairs' quotient.
fn over(
    (n1, n2, n_bound): (f64, f64, f64),
    (d1, d2, d_bound): (f64, f64, f64),
) -> Option<(f64, f64, f64)> {
    let inside = |value: f64| (2f64.powi(-900)..=2f64.powi(900)).contains(&value.abs());
    if !(inside(n1) && inside(d1)) {
        return None;
    }
    let relative = |bound: f64, value: f64| bound / value.abs() * (1.0 + 2f64.powi(-40));
    let parts = relative(n_bound, n1) + relative(d_bound, d1);
    let known = parts <= 2f64.powi(-40);
    if !known {
        return None;
    }
    let q1 = n1 / d1;
    let product = q1 * d1;
    let remainder = (n1 - product) - product_error(q1, d1, product);
    let q2 = ((remainder + n2) - q1 * d2) / d1;
    let (value, rest) = two_sum(q1, q2);
    let bound =
        value.abs() * (parts * (1.0 + 2f64.powi(-37)) + 2f64.powi(-99)) * (1.0 + 2f64.powi(-40));
    Some((value, rest, bound))
}

/// An exact signed number, `±limbs · 2^least`.
struct Signed {
    negative: bool,
    limbs: Vec<u64>,
    least: i32,
}

impl Signed {
    /// What `total` sums to.
    fn of(total: Total) -> Self {
        let (negative, limbs, least) = total.difference();
        Self::trimmed(negative, &limbs, least)
    }

    fn trimmed(negative: bool, limbs: &[u64], least: i32) -> Self {
        let (limbs, least) = trimmed(limbs, least);
        Self {
            negative: negative && !limbs.is_empty(),
            limbs: limbs.to_vec(),
            least,
        }
    }

    /// The exact sum of the products of `terms`, three factors each.
    fn of_terms(terms: &[(f64, f64, f64)]) -> Self {
        Self::of(Total::of_terms(terms))
    }

    fn not_positive(&self) -> bool {
        self.negative || self.limbs.is_empty()
    }

    fn times(&self, other: &Self) -> Self {
        let limbs = product_of(&self.limbs, &other.limbs);
        Self::trimmed(
            self.negative != other.negative,
            &limbs,
            self.least + other.least,
        )
    }

    fn minus(&self, other: &Self) -> Self {
        let least = self.least.min(other.least);
        let a = shifted(&self.limbs, (self.least - least) as usize);
        let b = shifted(&other.limbs, (other.least - least) as usize);
        // `self - other` is `self + (-other)`.
        let other_negative = !other.negative;
        if self.negative == other_negative {
            let length = a.len().max(b.len()) + 1;
            let (mut sum, mut carry) = (vec![0u64; length], 0u128);
            for (index, limb) in sum.iter_mut().enumerate() {
                let total = u128::from(a.get(index).copied().unwrap_or(0))
                    + u128::from(b.get(index).copied().unwrap_or(0))
                    + carry;
                *limb = total as u64;
                carry = total >> 64;
            }
            return Self::trimmed(self.negative, &sum, least);
        }
        let (mut larger, smaller, negative) = if at_least(&a, &b) {
            (a, b, self.negative)
        } else {
            (b, a, other_negative)
        };
        subtract(&mut larger, &smaller);
        Self::trimmed(negative, &larger, least)
    }

    fn total(self) -> Total {
        let zeros = vec![0u64; self.limbs.len()];
        let (positive, negative) = if self.negative {
            (zeros, self.limbs)
        } else {
            (self.limbs, zeros)
        };
        Total {
            positive,
            negative,
            least: self.least,
        }
    }
}

/// Each observation's weight as the parts that sum to it exactly.
fn weight_parts(sample: &Sample, k: usize) -> ([f64; 2], usize) {
    match (sample.frequencies, sample.reliabilities) {
        (Some(f), Some(a)) => {
            let product = f[k] * a[k];
            ([product, product_error(f[k], a[k], product)], 2)
        }
        (Some(w), None) | (None, Some(w)) => ([w[k], 0.0], 1),
        (None, None) => ([1.0, 0.0], 1),
    }
}

/// Row `row`'s value `k` as the parts that sum to it exactly.
fn value_parts(sample: &Sample, row: usize, k: usize) -> ([f64; 2], usize) {
    let at = row * sample.observations + k;
    match sample.lows {
        Some(lows) if lows[at] != 0.0 => ([sample.values[at], lows[at]], 2),
        _ => ([sample.values[at], 0.0], 1),
    }
}

/// The exact sum over the observations of `w`, `w x` or `w x y` for the
/// rows given.
fn exact_sum(sample: &Sample, x: Option<usize>, y: Option<usize>) -> Signed {
    let mut terms = Vec::new();
    for k in 0..sample.observations {
        let (weights, weight_count) = weight_parts(sample, k);
        let parts = |row: Option<usize>| match row {
            Some(row) => value_parts(sample, row, k),
            None => ([1.0, 0.0], 1),
        };
        let ((xs, x_count), (ys, y_count)) = (parts(x), parts(y));
        for &w in &weights[..weight_count] {
            for &a in &xs[..x_count] {
                for &b in &ys[..y_count] {
                    terms.push((w, a, b));
                }
            }
        }
    }
    Signed::of_terms(&terms)
}

/// The exact weight total and divisor of `sample`.
fn exact_divisor(sample: &Sample) -> (Signed, Signed) {
    let total = exact_sum(sample, None, None);
    let correction = Signed::of(
        Total::of_wide(std::iter::once(sample.correction)).expect("a finite correction"),
    );
    let divisor = match sample.reliabilities {
        None => total.times(&total.minus(&correction)),
        Some(reliabilities) => {
            let mut terms = Vec::new();
            for (k, &a) in reliabilities.iter().enumerate() {
                let (weights, count) = weight_parts(sample, k);
                for &w in &weights[..count] {
                    terms.push((w, a, 1.0));
                }
            }
            let reliable = Signed::of_terms(&terms);
            total.times(&total).minus(&correction.times(&reliable))
        }
    };
    (total, divisor)
}

/// The exact numerator `W Σ w x y - Σ w x Σ w y` of rows `i` and `j`.
fn exact_numerator(sample: &Sample, total: &Signed, i: usize, j: usize) -> Signed {
    let products = exact_sum(sample, Some(i), Some(j));
    let (x, y) = (
        exact_sum(sample, Some(i), None),
        exact_sum(sample, Some(j), None),
    );
    total.times(&products).minus(&x.times(&y))
}

/// The entry at rows `i` and `j`, formed in whole numbers: the numerator
/// over the divisor, or the root of its square over the product of the
/// rows' own, signed.
#[cold]
#[inline(never)]
fn exact_entry<T: Output>(sample: &Sample, i: usize, j: usize, correlation: bool) -> T {
    let (total, divisor) = exact_divisor(sample);
    let numerator = exact_numerator(sample, &total, i, j);
    if !correlation {
        return numerator.total().quotient(divisor.total());
    }
    let first = exact_numerator(sample, &total, i, i);
    if i == j {
        return T::narrow(if first.limbs.is_empty() {
            f64::NAN
        } else {
            1.0
        });
    }
    let second = exact_numerator(sample, &total, j, j);
    let negative = numerator.negative;
    let square = numerator.times(&numerator);
    let magnitude: T = square.total().quotient_root(first.times(&second).total());
    if negative {
        T::narrow(-magnitude.widen())
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<'a>(values: &'a [f64], variables: usize, correction: f64) -> Sample<'a> {
        Sample {
            values,
            lows: None,
            variables,
            observations: values.len() / variables,
            frequencies: None,
            reliabilities: None,
            correction,
        }
    }

    #[test]
    fn small_rows_are_exact() {
        let values = [1.0, 2.0, 4.0, 3.0, 1.0, 2.0];
        let mut out = [0f64; 4];
        covariances(&sample(&values, 2, 1.0), false, &mut out).unwrap();
        assert_eq!(out, [7.0 / 3.0, -0.5, -0.5, 1.0]);
        let mut narrow = [0f32; 4];
        covariances(&sample(&values, 2, 1.0), false, &mut narrow).unwrap();
        assert_eq!(narrow, [7.0 / 3.0, -0.5, -0.5, 1.0]);
        covariances(&sample(&values, 2, 1.0), true, &mut out).unwrap();
        assert_eq!((out[0], out[3]), (1.0, 1.0));
        assert_eq!(out[1], out[2]);
        // The float nearest `-sqrt(3 / 28)`; `-0.5 / sqrt(7 / 3)` in floats
        // rounds three times and lands on its neighbour.
        assert_eq!(out[1], -0.3273268353539886);
        assert_ne!(out[1], -0.5 / (7.0f64 / 3.0).sqrt());
    }

    #[test]
    fn constant_and_special_rows() {
        let values = [2.0, 2.0, 2.0, 1.0, f64::NAN, 3.0, 1.0, 5.0, 2.0];
        let mut out = [0f64; 9];
        covariances(&sample(&values, 3, 1.0), false, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_nan() && out[4].is_nan() && out[5].is_nan());
        assert_eq!(out[2], 0.0);
        covariances(&sample(&values, 3, 1.0), true, &mut out).unwrap();
        assert!(out[0].is_nan() && out[2].is_nan() && out[4].is_nan());
        assert_eq!(out[8], 1.0);
    }

    #[test]
    fn refusals() {
        let values = [1.0, 2.0, 4.0];
        let mut out = [0f64; 1];
        let refused = covariances(&sample(&values, 1, 3.0), false, &mut out);
        assert_eq!(refused, Err(Refusal::NoDivisor));
        let zeros = [0.0; 3];
        let weighted = Sample {
            frequencies: Some(&zeros),
            ..sample(&values, 1, 1.0)
        };
        assert_eq!(
            covariances(&weighted, false, &mut out),
            Err(Refusal::NoWeight)
        );
    }
}
