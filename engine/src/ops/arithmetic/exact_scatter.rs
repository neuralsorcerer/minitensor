// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Scattered sums rounded once.
//!
//! A scatter-add leaves at each destination the sum of what was there and of
//! every value addressed to it. Added one at a time in the element type,
//! that is a rounding for each value past the first, in whatever order they
//! arrive. Here a destination several values arrive at is the value nearest
//! the exact sum of them all: its values are added by two-sums, the errors
//! beside, and rounded from that total where its bound places it, or summed
//! exactly where it cannot. A destination one value arrives at takes one
//! addition, which rounds once as it is, and one nothing arrives at keeps
//! its value.
//!
//! A zero keeps the sign additions give it: negative only when every value
//! added, the destination's own among them, is a negative zero.
//!
//! A scattered product is grouped the same way, a block of destinations at
//! a time: a destination two factors or fewer reach is one multiplication,
//! which rounds once as it is, and one more reach is the value nearest the
//! exact product of them, as `prod` gives it.

use super::{BinTotal, Carried, Element, LANES, Output, Tallies, mean_to, sum_to};
use crate::ops::arithmetic::exact_product::product_to;
use crate::ops::map::par_out_chunks;

/// Destinations a block keeps totals for at once, 32 bytes each: a block's
/// totals stay in the second-level cache however far apart its sources lie.
const SLOTS: usize = 1 << 12;

/// Destinations a task takes at least, so that runs of a few destinations
/// each do not cost a task apiece.
const TASK: usize = 1 << 12;

/// The shape of a scatter: `out` is runs of `chunk` destinations, and run
/// `o` has `rows * inner` sources of its own, which reach no other run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scatter {
    pub(crate) chunk: usize,
    pub(crate) rows: usize,
    pub(crate) inner: usize,
    /// Whether a destination's own value joins the sum of what arrives; one
    /// that does not is replaced by it.
    pub(crate) own: bool,
    /// Whether a destination is the mean of its values, not their sum.
    pub(crate) mean: bool,
}

impl BinTotal {
    /// No values yet: a negative zero, which every first value replaces, so
    /// that a total of negative zeros stays one.
    const EMPTY: Self = Self {
        sum: -0.0,
        error: 0.0,
        magnitude: 0.0,
        additions: 0,
    };

    /// The value nearest the exact total, or nearest it over `mean_over`,
    /// where the bound says which that is.
    fn settled<T: Output>(&self, mean_over: Option<usize>) -> Option<T> {
        if self.magnitude == 0.0 && mean_over.is_none() {
            // No addition rounded, so the sum is the exact one.
            return self.sum.is_finite().then(|| T::narrow(self.sum));
        }
        let carried = Carried {
            sum: self.sum,
            error: self.error,
            // The errors pass through no more additions than were made.
            slack: self.magnitude * (self.additions + LANES as u64) as f64 * 2f64.powi(-52),
        };
        match mean_over {
            None => carried.rounded(),
            Some(count) => carried.divided(count),
        }
    }
}

/// The sources that reach a block of destinations, each a place in the block
/// and a value, in the order they arrive.
trait Arrivals<T> {
    fn none(&self) -> bool;
    fn each(&self, take: impl FnMut(usize, T));
}

/// A block's sources set out beside their places in it.
struct Paired<'a, T>(&'a [(u32, T)]);

impl<T: Copy> Arrivals<T> for Paired<'_, T> {
    fn none(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    fn each(&self, mut take: impl FnMut(usize, T)) {
        for &(slot, value) in self.0 {
            take(slot as usize, value);
        }
    }
}

/// The sources of a run that is one block, read where they lie.
struct Whole<'a, S> {
    source: &'a S,
    o: usize,
    rows: usize,
    inner: usize,
}

impl<T, S: Fn(usize, usize, usize) -> (usize, T)> Arrivals<T> for Whole<'_, S> {
    fn none(&self) -> bool {
        false
    }

    #[inline(always)]
    fn each(&self, mut take: impl FnMut(usize, T)) {
        for i in 0..self.rows {
            for j in 0..self.inner {
                let (slot, value) = (self.source)(self.o, i, j);
                take(slot, value);
            }
        }
    }
}

/// What a task keeps from one run to the next.
struct Scratch<T> {
    /// One bit for each destination of a run: whether a source has reached
    /// it.
    seen: Vec<u64>,
    /// Where each block's sources start among a run's.
    offsets: Vec<usize>,
    pairs: Vec<(u32, T)>,
    totals: Vec<BinTotal>,
}

/// What a destination becomes of the values that reach it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Sum,
    Mean,
    /// Their product; with `nonzero`, of those that are not zeros, and one
    /// for none.
    Product {
        nonzero: bool,
    },
}

/// Add each run's sources into its destinations in `out`: `source(o, i, j)`
/// is the destination within run `o` that source `(i, j)` of it adds to, and
/// the value it adds. Each destination is rounded once from the exact sum,
/// or mean, of its values.
pub(crate) fn add_scattered<T, S>(out: &mut [T], shape: Scatter, source: S)
where
    T: Element + Output + Default + std::ops::Add<Output = T>,
    S: Fn(usize, usize, usize) -> (usize, T) + Sync,
{
    let kind = if shape.mean { Kind::Mean } else { Kind::Sum };
    scattered(out, shape, kind, source);
}

/// Multiply each run's sources into its destinations in `out`, as
/// [`add_scattered`] adds them -- `shape.mean` aside: each destination
/// something reaches is rounded once from the exact product of its values,
/// and of its own where `shape.own`; with `nonzero`, of those that are not
/// zeros, one for none.
pub(crate) fn multiply_scattered<T, S>(out: &mut [T], shape: Scatter, nonzero: bool, source: S)
where
    T: Element + Output + Default + std::ops::Add<Output = T>,
    S: Fn(usize, usize, usize) -> (usize, T) + Sync,
{
    scattered(out, shape, Kind::Product { nonzero }, source);
}

/// [`add_scattered`] and [`multiply_scattered`], a destination of `kind`.
fn scattered<T, S>(out: &mut [T], shape: Scatter, kind: Kind, source: S)
where
    T: Element + Output + Default + std::ops::Add<Output = T>,
    S: Fn(usize, usize, usize) -> (usize, T) + Sync,
{
    let Scatter {
        chunk, rows, inner, ..
    } = shape;
    if chunk == 0 || rows == 0 || inner == 0 || out.is_empty() {
        return;
    }
    let runs_per_task = (TASK / chunk).max(1);
    par_out_chunks(out, chunk * runs_per_task, &|start, part| {
        let mut scratch = Scratch {
            seen: vec![0u64; chunk.div_ceil(64)],
            offsets: Vec::new(),
            pairs: Vec::new(),
            totals: Vec::new(),
        };
        for (offset, run) in part.chunks_mut(chunk).enumerate() {
            let o = start / chunk + offset;
            add_run(run, o, &shape, kind, &source, &mut scratch);
        }
    });
}

/// One run of [`scattered`].
fn add_run<T, S>(
    run: &mut [T],
    o: usize,
    shape: &Scatter,
    kind: Kind,
    source: &S,
    scratch: &mut Scratch<T>,
) where
    T: Element + Output + Default + std::ops::Add<Output = T>,
    S: Fn(usize, usize, usize) -> (usize, T) + Sync,
{
    let Scatter {
        chunk,
        rows,
        inner,
        own,
        ..
    } = *shape;
    // How many sources each block of destinations takes, and whether any
    // destination takes more than one.
    let blocks = chunk.div_ceil(SLOTS);
    let Scratch {
        seen,
        offsets,
        pairs,
        totals,
    } = scratch;
    offsets.clear();
    offsets.resize(blocks + 1, 0);
    seen.fill(0);
    let mut repeated = false;
    for i in 0..rows {
        for j in 0..inner {
            let (slot, _) = source(o, i, j);
            let (word, bit) = (slot / 64, 1u64 << (slot % 64));
            repeated |= seen[word] & bit != 0;
            seen[word] |= bit;
            offsets[slot / SLOTS + 1] += 1;
        }
    }
    if !repeated && kind == Kind::Sum {
        // One value at each destination it reaches: one addition.
        for i in 0..rows {
            for j in 0..inner {
                let (slot, value) = source(o, i, j);
                let start = if own { run[slot] } else { T::default() };
                run[slot] = start + value;
            }
        }
        return;
    }
    if blocks == 1 {
        let arrivals = Whole {
            source,
            o,
            rows,
            inner,
        };
        return settle_block(run, &arrivals, own, kind, totals);
    }

    // The sources a block at a time, each beside its place in the block, in
    // the order they arrive.
    for block in 0..blocks {
        offsets[block + 1] += offsets[block];
    }
    let mut cursor = offsets[..blocks].to_vec();
    pairs.clear();
    pairs.resize(rows * inner, (0u32, T::default()));
    for i in 0..rows {
        for j in 0..inner {
            let (slot, value) = source(o, i, j);
            let block = slot / SLOTS;
            pairs[cursor[block]] = ((slot % SLOTS) as u32, value);
            cursor[block] += 1;
        }
    }
    let (pairs, offsets) = (&*pairs, &*offsets);
    par_out_chunks(run, SLOTS, &|start, part| {
        let block = start / SLOTS;
        let arrivals = Paired(&pairs[offsets[block]..offsets[block + 1]]);
        settle_block(part, &arrivals, own, kind, &mut Vec::new());
    });
}

/// One block of destinations from the sources that reach it.
fn settle_block<T: Element + Output>(
    part: &mut [T],
    arrivals: &impl Arrivals<T>,
    own: bool,
    kind: Kind,
    totals: &mut Vec<BinTotal>,
) {
    if arrivals.none() {
        return;
    }
    let mean = match kind {
        Kind::Product { nonzero } => return multiply_block(part, arrivals, own, nonzero),
        kind => kind == Kind::Mean,
    };
    totals.clear();
    totals.resize(part.len(), BinTotal::EMPTY);
    arrivals.each(|slot, value| totals[slot].add(value.wide()));
    // Destinations whose bound does not place them, to be summed exactly.
    let mut open: Vec<u32> = Vec::new();
    for (slot, (place, total)) in part.iter_mut().zip(totals.iter_mut()).enumerate() {
        if total.additions == 0 {
            continue;
        }
        let count = total.additions as usize + usize::from(own);
        if own {
            total.add(place.wide());
        }
        match total.settled::<T>(mean.then_some(count)) {
            Some(value) => *place = value,
            None => open.push(slot as u32),
        }
    }
    if open.is_empty() {
        return;
    }
    let mut nth_open = vec![u32::MAX; part.len()];
    for (nth, &slot) in open.iter().enumerate() {
        nth_open[slot as usize] = nth as u32;
    }
    let mut gathered: Vec<Vec<T>> = vec![Vec::new(); open.len()];
    arrivals.each(|slot, value| {
        if let Some(kept) = gathered.get_mut(nth_open[slot] as usize) {
            kept.push(value);
        }
    });
    for (&slot, mut kept) in open.iter().zip(gathered) {
        let place = &mut part[slot as usize];
        if own {
            kept.push(*place);
        }
        *place = if mean { mean_to(&kept) } else { sum_to(&kept) };
    }
}

/// One block of destinations as the products of the sources that reach
/// them; see [`multiply_scattered`].
fn multiply_block<T: Element + Output>(
    part: &mut [T],
    arrivals: &impl Arrivals<T>,
    own: bool,
    nonzero: bool,
) {
    let len = part.len();
    let kept = |value: T| !nonzero || value.wide() != 0.0;
    // How many factors each destination takes, its own value first where it
    // counts.
    let (mut arrived, mut counts) = (vec![false; len], vec![0u32; len]);
    arrivals.each(|slot, value| {
        arrived[slot] = true;
        counts[slot] += u32::from(kept(value));
    });
    let first = |at: usize, value: T| own && arrived[at] && kept(value);
    // More than two side by side from `starts[at]`, to be multiplied out
    // exactly; two or fewer one multiplication at most, from the first.
    let mut starts = vec![0usize; len + 1];
    for (at, &value) in part.iter().enumerate() {
        counts[at] += u32::from(first(at, value));
        let many = if counts[at] > 2 {
            counts[at] as usize
        } else {
            0
        };
        starts[at + 1] = starts[at] + many;
    }
    let mut factors = vec![T::narrow(1.0); starts[len]];
    let mut cursor = starts[..len].to_vec();
    for (at, place) in part.iter_mut().enumerate() {
        let counted = first(at, *place);
        if counts[at] > 2 && counted {
            factors[cursor[at]] = *place;
            cursor[at] += 1;
        } else if arrived[at] && !counted {
            *place = T::narrow(1.0);
        }
    }
    arrivals.each(|slot, value| {
        if !kept(value) {
        } else if counts[slot] > 2 {
            factors[cursor[slot]] = value;
            cursor[slot] += 1;
        } else {
            part[slot] = T::narrow(part[slot].widen() * value.wide());
        }
    });
    for (at, place) in part.iter_mut().enumerate() {
        if counts[at] > 2 {
            *place = product_to(&factors[starts[at]..starts[at + 1]]);
        }
    }
}

/// Rows a scatter of whole rows takes one element at a time: narrower than
/// this, a row is too short for adding it a column beside a column to pay.
const NARROW: usize = 16;

/// Add whole rows into rows: `out` is runs of `rows` rows of `inner` values,
/// and source row `i` of a run -- every run has `row_of.len()` of them, laid
/// out after one another in `sources` -- adds to the run's row `row_of[i]`.
/// Each value of `out` is rounded once from the exact sum of what it held
/// and what arrives.
///
/// A row one source reaches takes one addition a value. A row several reach
/// has them added a row at a time into [`Tallies`], a two-sum for each
/// column, and each column settled as a destination of [`add_scattered`]
/// is.
pub(crate) fn add_scattered_rows<T>(
    out: &mut [T],
    sources: &[T],
    inner: usize,
    rows: usize,
    row_of: &[usize],
) where
    T: Element + Output + Default + std::ops::Add<Output = T>,
{
    let per_run = row_of.len();
    if inner == 0 || rows == 0 || per_run == 0 || out.is_empty() {
        return;
    }
    if inner < NARROW {
        let shape = Scatter {
            chunk: rows * inner,
            rows: per_run,
            inner,
            own: true,
            mean: false,
        };
        return add_scattered(out, shape, |o, i, j| {
            (
                row_of[i] * inner + j,
                sources[(o * per_run + i) * inner + j],
            )
        });
    }
    // The source rows of each destination row, in the order they arrive.
    let mut starts = vec![0usize; rows + 1];
    for &row in row_of {
        starts[row + 1] += 1;
    }
    for row in 0..rows {
        starts[row + 1] += starts[row];
    }
    let mut cursor = starts[..rows].to_vec();
    let mut order = vec![0usize; per_run];
    for (i, &row) in row_of.iter().enumerate() {
        order[cursor[row]] = i;
        cursor[row] += 1;
    }
    let (starts, order) = (&starts, &order);
    let rows_per_task = (TASK / inner).max(1);
    par_out_chunks(out, inner * rows_per_task, &|start, part| {
        let mut columns = Tallies::new(inner);
        for (offset, place) in part.chunks_mut(inner).enumerate() {
            let index = start / inner + offset;
            let (o, row) = (index / rows, index % rows);
            let source = |i: usize| &sources[(o * per_run + i) * inner..][..inner];
            match &order[starts[row]..starts[row + 1]] {
                [] => {}
                [only] => {
                    for (slot, &value) in place.iter_mut().zip(source(*only)) {
                        *slot = *slot + value;
                    }
                }
                several => {
                    columns.clear(-0.0);
                    columns.add(0, place);
                    for &i in several {
                        columns.add(0, source(i));
                    }
                    for (j, slot) in place.iter_mut().enumerate() {
                        *slot = columns.settled::<T>(j, None).unwrap_or_else(|| {
                            let mut kept: Vec<T> = several.iter().map(|&i| source(i)[j]).collect();
                            kept.push(*slot);
                            sum_to(&kept)
                        });
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(seed: u64, len: usize) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                unit * 2f64.powi((state % 40) as i32 - 20)
            })
            .collect()
    }

    /// Destinations by gathering each one's values and summing them exactly.
    fn reference<T: Element + Output>(
        base: &[T],
        slots: &[usize],
        sources: &[T],
        own: bool,
        mean: bool,
    ) -> Vec<T> {
        let mut out = base.to_vec();
        for (slot, place) in out.iter_mut().enumerate() {
            let mut kept: Vec<T> = slots
                .iter()
                .zip(sources)
                .filter(|(at, _)| **at == slot)
                .map(|(_, value)| *value)
                .collect();
            if kept.is_empty() {
                continue;
            }
            if own {
                kept.push(*place);
            }
            *place = if mean { mean_to(&kept) } else { sum_to(&kept) };
        }
        out
    }

    fn check<T>(narrow: impl Fn(f64) -> T, destinations: usize, sources: usize, seed: u64)
    where
        T: Element + Output + Default + std::ops::Add<Output = T> + std::fmt::Debug,
    {
        let base: Vec<T> = values(seed, destinations)
            .into_iter()
            .map(&narrow)
            .collect();
        let added: Vec<T> = values(seed + 1, sources).into_iter().map(&narrow).collect();
        // Most sources at a few destinations, the rest spread over them all.
        let slots: Vec<usize> = values(seed + 2, sources)
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let spread = (value.to_bits() >> 7) as usize;
                if index.is_multiple_of(3) {
                    spread % destinations
                } else {
                    spread % destinations.min(17)
                }
            })
            .collect();
        for (own, mean) in [(true, false), (false, false), (true, true), (false, true)] {
            let mut out = base.clone();
            let shape = Scatter {
                chunk: destinations,
                rows: sources,
                inner: 1,
                own,
                mean,
            };
            add_scattered(&mut out, shape, |_, i, _| (slots[i], added[i]));
            let want = reference(&base, &slots, &added, own, mean);
            assert_eq!(
                format!("{out:?}"),
                format!("{want:?}"),
                "own {own}, mean {mean}"
            );
        }
    }

    #[test]
    fn every_destination_is_its_exact_total() {
        for (destinations, sources) in [(5, 40), (300, 2000), (3 * SLOTS + 11, 9000)] {
            check(|value| value as f32, destinations, sources, 31);
            check(|value| value, destinations, sources, 37);
        }
    }

    #[test]
    fn every_destination_is_its_exact_product() {
        // Factors near one, so that a product of many stays in range, with
        // zeros among them; most at a few destinations, the rest spread.
        for (destinations, sources) in [(5, 40), (300, 2000), (3 * SLOTS + 11, 9000)] {
            let near_one = |seed: u64, len: usize| -> Vec<f64> {
                values(seed, len)
                    .iter()
                    .map(|value| {
                        let zero = (value.to_bits() >> 3).is_multiple_of(29);
                        if zero { 0.0 } else { 1.0 + value }
                    })
                    .collect()
            };
            let base = near_one(47, destinations);
            let added = near_one(53, sources);
            let slots: Vec<usize> = values(59, sources)
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let spread = (value.to_bits() >> 7) as usize;
                    let reach = if index.is_multiple_of(3) {
                        destinations
                    } else {
                        17
                    };
                    spread % destinations.min(reach)
                })
                .collect();
            for (own, nonzero) in [(true, false), (false, false), (true, true), (false, true)] {
                let mut out = base.clone();
                let shape = Scatter {
                    chunk: destinations,
                    rows: sources,
                    inner: 1,
                    own,
                    mean: false,
                };
                multiply_scattered(&mut out, shape, nonzero, |_, i, _| (slots[i], added[i]));
                for (slot, &got) in out.iter().enumerate() {
                    let arriving = slots.iter().zip(&added).filter(|(at, _)| **at == slot);
                    let mut kept: Vec<f64> = arriving.map(|(_, value)| *value).collect();
                    if kept.is_empty() {
                        assert_eq!(got.to_bits(), base[slot].to_bits());
                        continue;
                    }
                    if own {
                        kept.push(base[slot]);
                    }
                    kept.retain(|value| !nonzero || *value != 0.0);
                    let want: f64 = product_to(&kept);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "own {own}, nonzero {nonzero}"
                    );
                }
            }
        }
    }

    #[test]
    fn rows_added_into_rows_are_exact_totals() {
        for (inner, rows, per_run, runs) in [(40, 6, 25, 2), (3, 5, 30, 3), (64, 1, 9, 1)] {
            let base: Vec<f32> = values(41, runs * rows * inner)
                .into_iter()
                .map(|value| value as f32)
                .collect();
            let added: Vec<f32> = values(43, runs * per_run * inner)
                .into_iter()
                .map(|value| value as f32)
                .collect();
            // Row 0 of every run is reached by nothing when there are others.
            let row_of: Vec<usize> = (0..per_run)
                .map(|i| {
                    if rows == 1 {
                        0
                    } else {
                        1 + (i * 7 + i / 3) % (rows - 1)
                    }
                })
                .collect();
            let mut out = base.clone();
            add_scattered_rows(&mut out, &added, inner, rows, &row_of);
            for o in 0..runs {
                let at = |row: usize| (o * rows + row) * inner;
                let slots: Vec<usize> = (0..per_run * inner)
                    .map(|pos| row_of[pos / inner] * inner + pos % inner)
                    .collect();
                let want = reference(
                    &base[at(0)..at(rows)],
                    &slots,
                    &added[o * per_run * inner..(o + 1) * per_run * inner],
                    true,
                    false,
                );
                assert_eq!(
                    format!("{:?}", &out[at(0)..at(rows)]),
                    format!("{want:?}"),
                    "inner {inner}, run {o}"
                );
            }
        }
    }

    #[test]
    fn runs_of_a_few_destinations_each_are_exact_totals() {
        // Sixty runs of twenty destinations, five rows of four: every third
        // run reaches each destination at most once.
        let (runs, dim, inner, rows) = (60, 5, 4, 7);
        let base: Vec<f32> = values(47, runs * dim * inner)
            .into_iter()
            .map(|value| value as f32)
            .collect();
        let added: Vec<f32> = values(53, runs * rows * inner)
            .into_iter()
            .map(|value| value as f32)
            .collect();
        let row_at = |o: usize, i: usize, j: usize| {
            if o.is_multiple_of(3) {
                i % dim
            } else {
                (o * 31 + i * 7 + j * 3 + (i * j) % 5) % dim
            }
        };
        for (own, mean) in [(true, false), (false, false), (true, true)] {
            let mut out = base.clone();
            let shape = Scatter {
                chunk: dim * inner,
                rows: if own { rows } else { dim },
                inner,
                own,
                mean,
            };
            let rows = shape.rows;
            add_scattered(&mut out, shape, |o, i, j| {
                (row_at(o, i, j) * inner + j, added[(o * 7 + i) * inner + j])
            });
            for o in 0..runs {
                let slots: Vec<usize> = (0..rows * inner)
                    .map(|pos| row_at(o, pos / inner, pos % inner) * inner + pos % inner)
                    .collect();
                let sources: Vec<f32> = (0..rows * inner)
                    .map(|pos| added[(o * 7 + pos / inner) * inner + pos % inner])
                    .collect();
                let run = o * dim * inner..(o + 1) * dim * inner;
                let want = reference(&base[run.clone()], &slots, &sources, own, mean);
                assert_eq!(
                    format!("{:?}", &out[run]),
                    format!("{want:?}"),
                    "run {o}, own {own}, mean {mean}"
                );
            }
        }
    }

    #[test]
    fn columns_on_halfway_points_are_summed_exactly() {
        let inner = NARROW;
        let tiny = 2f32.powi(-60);
        let half = 2f32.powi(-24);
        let mut added = vec![0.0f32; 3 * inner];
        // Each column 1 + 2^-24 and a third value: past the halfway point,
        // short of it, on it, and on the one above an odd neighbour.
        for (column, (first, third)) in [
            (1.0, tiny),
            (1.0, -tiny),
            (1.0, 0.0),
            (1.0 + 2.0 * half, 0.0),
        ]
        .into_iter()
        .enumerate()
        {
            added[column] = first;
            added[inner + column] = half;
            added[2 * inner + column] = third;
        }
        for narrow in [false, true] {
            let mut out = vec![0.0f32; inner];
            if narrow {
                // The same sums one element at a time.
                let shape = Scatter {
                    chunk: inner,
                    rows: 3,
                    inner,
                    own: true,
                    mean: false,
                };
                add_scattered(&mut out, shape, |_, i, j| (j, added[i * inner + j]));
            } else {
                add_scattered_rows(&mut out, &added, inner, 1, &[0, 0, 0]);
            }
            assert_eq!(
                out[..4],
                [1.0 + 2.0 * half, 1.0, 1.0, 1.0 + 4.0 * half],
                "one element at a time: {narrow}"
            );
        }
    }

    #[test]
    fn rows_of_zeros_keep_their_signs() {
        let inner = NARROW;
        let mut out = vec![-0.0f32; 2 * inner];
        let mut added = vec![-0.0f32; 3 * inner];
        // Column 1 has a positive zero among its values, column 2 cancels.
        added[inner + 1] = 0.0;
        added[2] = 1.5;
        added[inner + 2] = -1.5;
        add_scattered_rows(&mut out, &added, inner, 2, &[1, 1, 1]);
        let bits = |value: f32| value.to_bits();
        assert!(out[..inner].iter().all(|&value| bits(value) == bits(-0.0)));
        assert_eq!(bits(out[inner]), bits(-0.0));
        assert_eq!(bits(out[inner + 1]), bits(0.0));
        assert_eq!(bits(out[inner + 2]), bits(0.0));
        assert!(
            out[inner + 3..]
                .iter()
                .all(|&value| bits(value) == bits(-0.0))
        );
    }

    #[test]
    fn one_value_at_each_destination_is_one_addition() {
        let base = [1.5f32, -0.0, 2.0, 0.25];
        let mut out = base;
        let shape = Scatter {
            chunk: 4,
            rows: 3,
            inner: 1,
            own: true,
            mean: false,
        };
        let added = [0.1f32, -0.0, 1e-9];
        add_scattered(&mut out, shape, |_, i, _| ([0, 1, 3][i], added[i]));
        assert_eq!(out[0], 1.5 + 0.1);
        assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 0.25 + 1e-9);
    }

    #[test]
    fn zeros_keep_the_sign_additions_give_them() {
        let shape = Scatter {
            chunk: 3,
            rows: 6,
            inner: 1,
            own: true,
            mean: false,
        };
        let slots = [0, 0, 1, 1, 2, 2];
        let added = [-0.0f64, -0.0, 1.0, -1.0, -0.0, 0.0];
        let mut out = [-0.0f64, 0.0, -0.0];
        add_scattered(&mut out, shape, |_, i, _| (slots[i], added[i]));
        // Every value a negative zero; values that cancel; a positive zero
        // among negative ones.
        assert_eq!(out[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(out[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(out[2].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn special_values_sum_as_they_would_one_by_one() {
        let shape = Scatter {
            chunk: 3,
            rows: 6,
            inner: 1,
            own: true,
            mean: false,
        };
        let slots = [0, 0, 1, 1, 2, 2];
        let added = [
            f32::INFINITY,
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1.0,
        ];
        let mut out = [1.0f32, 2.0, 3.0];
        add_scattered(&mut out, shape, |_, i, _| (slots[i], added[i]));
        assert_eq!(out[0], f32::INFINITY);
        assert!(out[1].is_nan() && out[2].is_nan());
    }
}
