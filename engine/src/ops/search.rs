// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Asking a sorted sequence where a value belongs.
//!
//! Everything here is one binary search wearing four hats. `searchsorted` is the
//! search itself; `bucketize` is the same call with the arguments the other way
//! round, the spelling that reads correctly when the sequence is a fixed set
//! of boundaries; `histogram` is the search
//! followed by a count; and `histc` is `histogram` with the edges chosen for you.
//!
//! `bincount` is the degenerate case of the same thing -- the value *is* the
//! bin, so there is no search left, only the count -- and it shares the
//! counting.
//!
//! It cannot be composed out of what the library has. A comparison against every
//! boundary would be `O(values * boundaries)` and would still leave the counting
//! to do; the whole point of a sorted sequence is that the answer is `log`
//! rather than linear in it, and nothing else here knows how to exploit
//! sortedness.
//!
//! None of it is differentiable, and not because it was easier that way: the
//! result is an index or a count, an integer that moves in jumps as a value
//! crosses a boundary. There is no derivative to hand back, so these return
//! `int64` and detach.

use crate::{
    error::{MinitensorError, Result},
    ops::map::{par_all_chunk, par_any_chunk, par_fold_chunks, par_map_indexed, par_out_chunks},
    ops::reduction::MINMAX_CHUNK,
    ops::util::pairwise_fold_vectors,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::hint::select_unpredictable;
use std::sync::Arc;

/// Where `value` belongs in an already-sorted `sequence`.
///
/// `right` picks which end of a run of equals to answer with: `false` gives the
/// first index whose element is greater than or equal to `value`, so the value
/// would be inserted *before* its equals; `true` gives the first strictly
/// greater, inserting *after* them. On a sequence with no duplicates the two
/// agree everywhere except exactly on an element.
///
/// NaN takes the place `sort` gives it: after every number, and equal to
/// itself. A NaN in the sequence already compared as larger than any value,
/// since every comparison with it is false. A NaN *value* did not: being less
/// than nothing it landed at index 0, before everything it sorts after, so
/// `sequence.searchsorted(nan)` disagreed with where the sort had put the NaNs.
/// It now lands at the first NaN, or past the last one with `right`.
fn locate<T: PartialOrd>(sequence: &[T], value: &T, right: bool) -> usize {
    // `x != x` holds only for NaN, and never for an integer or a bool, where
    // it folds away.
    #[allow(clippy::eq_op)]
    let value_is_nan = value != value;
    if value_is_nan {
        if right {
            return sequence.len();
        }
        // The NaNs are the run at the end of a sorted sequence; find its start.
        #[allow(clippy::eq_op)]
        return sequence.partition_point(|element| element == element);
    }
    let (mut low, mut high) = (0usize, sequence.len());
    while low < high {
        let middle = low + (high - low) / 2;
        let before = if right {
            sequence[middle] <= *value
        } else {
            sequence[middle] < *value
        };
        if before {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

/// The trailing extent of a tensor, and the number of rows in front of it.
fn rows_and_width(tensor: &Tensor) -> (usize, usize) {
    let dims = tensor.shape().dims();
    match dims.split_last() {
        Some((&width, leading)) => (leading.iter().product(), width),
        None => (1, 1),
    }
}

/// Run `body` over the two slices, which the caller has already checked hold
/// the same element type.
///
/// One `match` binding both, rather than two nested ones: the arms of two
/// independent matches are typed independently, so the compiler has no way to
/// know the needle and the haystack are the same kind of number even when they
/// certainly are. The comparison is the only thing that varies with the element
/// type, and it varies not at all in shape, so the five arms differ by one word.
macro_rules! with_pair {
    ($left:expr, $right:expr, $haystack:ident, $needles:ident, $body:expr) => {
        match $left.dtype() {
            DataType::Float32 => {
                let $haystack = $left.data().as_f32_slice().ok_or_else(dtype_mismatch)?;
                let $needles = $right.data().as_f32_slice().ok_or_else(dtype_mismatch)?;
                $body
            }
            DataType::Float64 => {
                let $haystack = $left.data().as_f64_slice().ok_or_else(dtype_mismatch)?;
                let $needles = $right.data().as_f64_slice().ok_or_else(dtype_mismatch)?;
                $body
            }
            DataType::Int32 => {
                let $haystack = $left.data().as_i32_slice().ok_or_else(dtype_mismatch)?;
                let $needles = $right.data().as_i32_slice().ok_or_else(dtype_mismatch)?;
                $body
            }
            DataType::Int64 => {
                let $haystack = $left.data().as_i64_slice().ok_or_else(dtype_mismatch)?;
                let $needles = $right.data().as_i64_slice().ok_or_else(dtype_mismatch)?;
                $body
            }
            DataType::Bool => {
                let $haystack = $left.data().as_bool_slice().ok_or_else(dtype_mismatch)?;
                let $needles = $right.data().as_bool_slice().ok_or_else(dtype_mismatch)?;
                $body
            }
        }
    };
}

fn dtype_mismatch() -> MinitensorError {
    MinitensorError::internal_error("searchsorted: dtype does not match the slice")
}

/// Where each element of `values` would be inserted into `sequence` to keep it
/// sorted.
///
/// `sequence` is searched along its last axis, and everything in front of that
/// axis is a batch: a one-dimensional sequence is searched by every value, and a
/// stack of sequences is matched row for row against a stack of values. The
/// sequence is *assumed* sorted and never checked, because checking would cost
/// the linear scan the binary search exists to avoid -- an unsorted sequence
/// gives a meaningless answer rather than an error.
///
/// The result is `int64` positions in `0..=width`, so a value past the end of
/// the sequence answers with the length rather than being clamped into it.
pub fn searchsorted(sequence: &Tensor, values: &Tensor, right: bool) -> Result<Tensor> {
    // A search is a comparison, and a comparison is answered exactly whatever
    // the two dtypes: refusing the pair turned away float32 values against a
    // float64 sequence, and `1.5` against integers, where every answer is
    // plain. Two floats or two integers meet at the wider, which holds both
    // exactly, as does a bool beside anything; an integer against a float has
    // no dtype that holds both and is searched across the two.
    let (from, to) = (sequence.dtype(), values.dtype());
    if from != to
        && from != DataType::Bool
        && to != DataType::Bool
        && from.is_float() != to.is_float()
    {
        return mixed_search(sequence, values, right);
    }
    if from != to {
        let common = crate::ops::binary::promote_arithmetic_dtype(from, to);
        return searchsorted(&sequence.astype(common)?, &values.astype(common)?, right);
    }
    search_rows(sequence, values, |layout, out, ordered, queried| {
        with_pair!(ordered, queried, haystack, needles, {
            layout.fill(out, haystack, needles, &|row, value| {
                locate(row, value, right)
            });
        });
        Ok(())
    })
}

/// The shape checks and the output of a search, with `body` filling the
/// positions from the contiguous sequence and values.
fn search_rows(
    sequence: &Tensor,
    values: &Tensor,
    body: impl FnOnce(&SearchLayout, &mut [i64], &Tensor, &Tensor) -> Result<()>,
) -> Result<Tensor> {
    if sequence.ndim() == 0 {
        return Err(MinitensorError::invalid_operation(
            "searchsorted: the sequence must have at least one dimension",
        ));
    }
    let (sequence_rows, width) = rows_and_width(sequence);
    // A one-dimensional sequence serves every value; anything else is matched
    // row for row, and the rows have to line up.
    let value_rows = if sequence.ndim() == 1 {
        1
    } else {
        rows_and_width(values).0
    };
    if sequence.ndim() > 1 && sequence_rows != value_rows {
        return Err(MinitensorError::invalid_operation(format!(
            "searchsorted: the sequence has {sequence_rows} rows and the values have \
             {value_rows}; a batched sequence must match its values"
        )));
    }

    let per_row = if sequence.ndim() == 1 {
        values.numel()
    } else {
        // No rows means no values, so there is nothing per row.
        values.numel().checked_div(value_rows).unwrap_or(0)
    };

    let ordered = sequence.contiguous()?;
    let queried = values.contiguous()?;
    let mut data = TensorData::zeros_on_device(values.numel(), DataType::Int64, values.device());
    if values.numel() > 0 {
        let out = data
            .as_i64_slice_mut()
            .ok_or_else(|| MinitensorError::internal_error("searchsorted: output is not int64"))?;
        body(
            &SearchLayout::of(sequence, width, per_row),
            out,
            &ordered,
            &queried,
        )?;
    }

    Ok(Tensor::new(
        Arc::new(data),
        values.shape().clone(),
        DataType::Int64,
        values.device(),
        false,
    ))
}

/// How the values of a search line up with the rows of its sequence.
struct SearchLayout {
    batch: bool,
    width: usize,
    per_row: usize,
}

impl SearchLayout {
    fn of(sequence: &Tensor, width: usize, per_row: usize) -> Self {
        Self {
            batch: sequence.ndim() > 1,
            width,
            per_row,
        }
    }

    /// `out[i]` set to `find` of the value's row and the value.
    fn fill<H: Sync, N: Sync>(
        &self,
        out: &mut [i64],
        haystack: &[H],
        needles: &[N],
        find: &(dyn Fn(&[H], &N) -> usize + Sync),
    ) {
        // Every value's search is independent of every other's, which is what
        // makes this parallel at all -- and it was not, which cost more than
        // the search: a million values against a million-element sequence took
        // 400 ms on one core.
        //
        // One search is about `log2(width)` probes and, past the cache, each is
        // a miss. A task wants a few thousand probes in it before the split
        // pays, so the chunk is that budget divided by the depth of one search:
        // a short sequence gets wide chunks because each search in it is cheap,
        // a long one gets narrow chunks because each is not.
        let depth = (usize::BITS - self.width.leading_zeros()).max(1) as usize;
        let chunk = ((1 << 12) / depth).max(1);
        par_out_chunks(out, chunk, &|start, block| {
            for (offset, slot) in block.iter_mut().enumerate() {
                let index = start + offset;
                let row = if self.batch {
                    index / self.per_row.max(1)
                } else {
                    0
                };
                let begin = row * self.width;
                *slot = find(&haystack[begin..begin + self.width], &needles[index]) as i64;
            }
        });
    }
}

/// Where a float belongs among sorted integers.
enum AmongIntegers {
    /// Where this integer does, searched on the same side.
    At(i64),
    /// Before every integer.
    Start,
    /// After every integer.
    End,
}

/// The integer search that places a float `value` where it belongs among
/// sorted integers, or the end it belongs at when no integer can name it.
///
/// An integer is below `value` exactly when it is below `ceil(value)`, and at
/// most `value` exactly when it is at most `floor(value)`, so the float's
/// place is an integer's. NaN sorts after every number.
fn float_among_integers(value: f64, right: bool) -> AmongIntegers {
    // 2^63, the first float past every int64.
    const BOUND: f64 = 9_223_372_036_854_775_808.0;
    if value.is_nan() {
        return AmongIntegers::End;
    }
    let key = if right { value.floor() } else { value.ceil() };
    if key >= BOUND {
        AmongIntegers::End
    } else if key < -BOUND {
        AmongIntegers::Start
    } else {
        AmongIntegers::At(key as i64)
    }
}

/// The float search that places an integer `value` where it belongs among
/// sorted floats: the largest float at most `value`, and the side to search
/// it on.
///
/// When that float is `value` itself the search is the ordinary one. When it
/// is not, `value` lies strictly between it and the next float, so the
/// elements below `value` are exactly those at most the float below it, and a
/// left search becomes a right one.
fn integer_among_floats(value: i64, right: bool) -> (f64, bool) {
    let mut below = value as f64;
    // Rounded to nearest, which may be above. One step down is then below, as
    // the gap between two floats this large is far wider than one.
    if below as i128 > i128::from(value) {
        below = if below > 0.0 {
            f64::from_bits(below.to_bits() - 1)
        } else {
            f64::from_bits(below.to_bits() + 1)
        };
    }
    let exact = below as i128 == i128::from(value);
    (below, right || !exact)
}

/// The largest float at or below `value`.
fn float_at_or_below(value: i64) -> f64 {
    integer_among_floats(value, false).0
}

/// The smallest float at or above `value`.
fn float_at_or_above(value: i64) -> f64 {
    let nearest = value as f64;
    if (nearest as i128) < i128::from(value) {
        nearest.next_up()
    } else {
        nearest
    }
}

/// [`searchsorted`] for an integer sequence searched by floats, or a float
/// sequence searched by integers. No dtype holds both exactly -- int64 is not
/// a subset of float64 past 2^53 -- so each value is turned into the search,
/// on the sequence's own dtype, that gives its exact place.
fn mixed_search(sequence: &Tensor, values: &Tensor, right: bool) -> Result<Tensor> {
    let integers = !sequence.dtype().is_float();
    // Widened within their own kind, which is exact: the sequence to int64 or
    // float64, the values to float64 or int64.
    let (wide_sequence, wide_values) = if integers {
        (
            sequence.astype(DataType::Int64)?,
            values.astype(DataType::Float64)?,
        )
    } else {
        (
            sequence.astype(DataType::Float64)?,
            values.astype(DataType::Int64)?,
        )
    };
    search_rows(
        &wide_sequence,
        &wide_values,
        |layout, out, ordered, queried| {
            if integers {
                let haystack = ordered.data().as_i64_slice().ok_or_else(dtype_mismatch)?;
                let needles = queried.data().as_f64_slice().ok_or_else(dtype_mismatch)?;
                layout.fill(
                    out,
                    haystack,
                    needles,
                    &|row, &value| match float_among_integers(value, right) {
                        AmongIntegers::At(key) => locate(row, &key, right),
                        AmongIntegers::Start => 0,
                        AmongIntegers::End => row.len(),
                    },
                );
            } else {
                let haystack = ordered.data().as_f64_slice().ok_or_else(dtype_mismatch)?;
                let needles = queried.data().as_i64_slice().ok_or_else(dtype_mismatch)?;
                layout.fill(out, haystack, needles, &|row, &value| {
                    let (key, side) = integer_among_floats(value, right);
                    locate(row, &key, side)
                });
            }
            Ok(())
        },
    )
}

/// Which bucket each element of `input` falls in, given the bucket
/// `boundaries`.
///
/// [`searchsorted`] with the arguments the other way round. Both spellings exist
/// because both readings are natural -- one asks where a value goes in a
/// sequence, the other asks which bucket a value is in.
pub fn bucketize(input: &Tensor, boundaries: &Tensor, right: bool) -> Result<Tensor> {
    if boundaries.ndim() != 1 {
        return Err(MinitensorError::invalid_operation(
            "bucketize: the boundaries must be one-dimensional",
        ));
    }
    searchsorted(boundaries, input, right)
}

/// How a histogram's bins were asked for.
pub enum Bins<'a> {
    /// This many equal-width bins, spanning the range.
    Count(usize),
    /// These edges exactly, which must be increasing. `n` edges make `n - 1`
    /// bins.
    Edges(&'a Tensor),
}

/// Every element of `tensor` as `f64`, for the arithmetic a histogram does.
fn as_doubles(tensor: &Tensor) -> Result<Vec<f64>> {
    let contiguous = tensor.contiguous()?;
    Ok(match tensor.dtype() {
        DataType::Float32 => contiguous
            .data()
            .as_f32_slice()
            .ok_or_else(dtype_mismatch)?
            .iter()
            .map(|value| *value as f64)
            .collect(),
        DataType::Float64 => contiguous
            .data()
            .as_f64_slice()
            .ok_or_else(dtype_mismatch)?
            .to_vec(),
        DataType::Int32 => contiguous
            .data()
            .as_i32_slice()
            .ok_or_else(dtype_mismatch)?
            .iter()
            .map(|value| *value as f64)
            .collect(),
        DataType::Int64 => contiguous
            .data()
            .as_i64_slice()
            .ok_or_else(dtype_mismatch)?
            .iter()
            .map(|value| *value as f64)
            .collect(),
        DataType::Bool => contiguous
            .data()
            .as_bool_slice()
            .ok_or_else(dtype_mismatch)?
            .iter()
            .map(|value| if *value { 1.0 } else { 0.0 })
            .collect(),
    })
}

/// Build a `float64` tensor from a vector.
fn doubles_to_tensor(values: Vec<f64>, device: crate::device::Device) -> Result<Tensor> {
    let shape = Shape::new(vec![values.len()]);
    let mut data = TensorData::zeros_on_device(values.len(), DataType::Float64, device);
    if !values.is_empty() {
        let slice = data
            .as_f64_slice_mut()
            .ok_or_else(|| MinitensorError::internal_error("histogram: output is not float64"))?;
        slice.copy_from_slice(&values);
    }
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        DataType::Float64,
        device,
        false,
    ))
}

/// The counts falling in each bin, and the edges that defined them.
///
/// The input is flattened first: a histogram is a question about a collection of
/// numbers, not about their arrangement.
///
/// Values outside the outermost edges are dropped rather than clamped into the
/// end bins, which is what makes a histogram over an explicit range mean what it
/// says. The last bin is closed on the right, so a value exactly at the top edge
/// lands in it rather than falling off -- an asymmetry every implementation of
/// this shares, because the alternative loses the maximum of the data.
///
/// `density` divides each count by the total and by its bin's width, so the
/// result integrates to one and is comparable across binnings.
/// Bins beyond which a private tally per task costs more than the contention it
/// saves: each task allocates one, and a wide one is a page fault per task.
const TALLY_PRIVATE_BINS: usize = 1 << 16;

/// The smallest run worth handing a task, and how many to aim for.
///
/// Both come from the value count alone -- never from the thread pool --
/// because a weighted tally adds floats, and a sum whose grouping follows the
/// pool answers differently on different machines.
///
/// 131,072 values, from 16,384: a band that short is tens of microseconds,
/// less than the pool costs to wake from Python, and `bincount` of 65,536
/// labels took 170-220us split four ways.
const TALLY_MIN_BAND: usize = 1 << 17;
const TALLY_BANDS: usize = 64;

/// Add one contribution per value into `bins` slots.
///
/// `contribution(index)` says which slot a value lands in and what it adds
/// there, or `None` for a value that lands nowhere. That closure is the only
/// difference between a histogram, a weighted histogram and a `bincount`, so
/// the counting itself is written once.
///
/// Each band keeps a private tally and they merge pairwise afterwards, which is
/// what makes this parallel at all: the slots a band touches are scattered, so
/// bands cannot be given disjoint pieces of one output.
pub(crate) fn tally<T, F>(count: usize, bins: usize, zero: T, contribution: F) -> Vec<T>
where
    T: Copy + Send + Sync + std::ops::Add<Output = T>,
    F: Fn(usize) -> Option<(usize, T)> + Sync,
{
    tally_bands(count, bins, zero, |first, last, slots: &mut [T]| {
        for position in first..last {
            if let Some((slot, value)) = contribution(position) {
                slots[slot] = slots[slot] + value;
            }
        }
    })
}

/// [`tally`] with the band's loop supplied whole: `fill(first, last, slots)`
/// adds everything in `first..last` into `slots`. For a caller whose loop is
/// worth writing itself, as counting labels is.
pub(crate) fn tally_bands<T, F>(count: usize, bins: usize, zero: T, fill: F) -> Vec<T>
where
    T: Copy + Send + Sync + std::ops::Add<Output = T>,
    F: Fn(usize, usize, &mut [T]) + Sync,
{
    if bins == 0 || count == 0 {
        return vec![zero; bins];
    }
    let band = count.div_ceil(TALLY_BANDS).max(TALLY_MIN_BAND);
    let bands = count.div_ceil(band);
    if bands < 2 || bins > TALLY_PRIVATE_BINS {
        let mut slots = vec![zero; bins];
        fill(0, count, &mut slots);
        return slots;
    }
    let partials = par_map_indexed(bands, &|index| {
        let first = index * band;
        let last = (first + band).min(count);
        let mut slots = vec![zero; bins];
        fill(first, last, &mut slots);
        slots
    });
    pairwise_fold_vectors(partials, |a, b| a + b)
}

/// Count each of `labels` into `slots`, every label below `slots.len()`.
fn count_labels<L: Copy>(labels: &[L], slots: &mut [i64], slot: impl Fn(L) -> usize) {
    let mut counters = Counters::new(slots.len());
    counters.add(labels, slots, slot);
    counters.finish(slots);
}

/// A count of labels into slots, kept so that an increment need not wait.
///
/// With few bins the same counter comes round again before its last
/// increment has landed, and each increment waits for the one before it:
/// sixteen thousand labels over seven bins took 37us. So a
/// short tally is kept four times over, label `i` going to copy `i % 4`, and
/// the copies added at the end -- exact, since the counts are integers.
struct Counters {
    /// Empty when there are enough bins to count into `slots` directly.
    copies: Vec<i64>,
}

impl Counters {
    const SPLIT_BELOW: usize = 1 << 12;

    fn new(bins: usize) -> Self {
        let copies = if bins > Self::SPLIT_BELOW {
            Vec::new()
        } else {
            vec![0; 4 * bins]
        };
        Self { copies }
    }

    /// Count `labels`, each naming a slot below `slots.len()`.
    fn add<L: Copy>(&mut self, labels: &[L], slots: &mut [i64], slot: impl Fn(L) -> usize) {
        let bins = slots.len();
        let copies = &mut self.copies;
        if copies.is_empty() {
            for &label in labels {
                slots[slot(label)] += 1;
            }
            return;
        }
        let (quads, rest) = labels.as_chunks::<4>();
        for quad in quads {
            copies[slot(quad[0])] += 1;
            copies[bins + slot(quad[1])] += 1;
            copies[2 * bins + slot(quad[2])] += 1;
            copies[3 * bins + slot(quad[3])] += 1;
        }
        for &label in rest {
            copies[slot(label)] += 1;
        }
    }

    fn finish(self, slots: &mut [i64]) {
        let copies = self.copies;
        if copies.is_empty() {
            return;
        }
        let bins = slots.len();
        for (bin, total) in slots.iter_mut().enumerate() {
            *total +=
                copies[bin] + copies[bins + bin] + copies[2 * bins + bin] + copies[3 * bins + bin];
        }
    }
}

/// Count `values` into `slots` by the slot `slot` names for each, a block at
/// a time: every slot in the block first, then every increment.
///
/// One pass doing both made each increment's address wait on the arithmetic
/// that names it, and the loads of the counters ran ahead of it on a guess.
/// With few bins the guess is often wrong -- the next value is in the same
/// bin -- and each wrong one is replayed: four million values in a mixed
/// range cost 6-9ns each. Named first, the increments run as fast as counting
/// labels does, 4ns each whatever the data.
fn count_binned<T: Copy>(values: &[T], slots: &mut [i64], slot: impl Fn(T) -> usize) {
    const BLOCK: usize = 1024;
    let mut counters = Counters::new(slots.len());
    let mut named = [0usize; BLOCK];
    for block in values.chunks(BLOCK) {
        let named = &mut named[..block.len()];
        for (bin, &value) in named.iter_mut().zip(block) {
            *bin = slot(value);
        }
        counters.add(named, slots, |bin| bin);
    }
    counters.finish(slots);
}

/// The labels of a [`bincount`], borrowed rather than copied.
///
/// This op used to collect its input into a `Vec<i64>` and then map that into a
/// `Vec<usize>` -- two copies of the whole tensor, 32MB for two million labels,
/// before any counting started. A three-way match per element is cheaper than
/// either of them.
enum Labels<'a> {
    I64(&'a [i64]),
    I32(&'a [i32]),
    Bool(&'a [bool]),
}

impl Labels<'_> {
    fn len(&self) -> usize {
        match self {
            Labels::I64(values) => values.len(),
            Labels::I32(values) => values.len(),
            Labels::Bool(values) => values.len(),
        }
    }

    #[inline]
    fn at(&self, index: usize) -> i64 {
        match self {
            Labels::I64(values) => values[index],
            Labels::I32(values) => i64::from(values[index]),
            Labels::Bool(values) => i64::from(values[index]),
        }
    }

    /// The smallest and largest label, or `(0, -1)` when there are none.
    ///
    /// One chunked pass for both, on the calling thread below the fold
    /// threshold. This was a per-element `par_iter` for the minimum and
    /// another for the maximum, two wakes of the pool per call: most of the
    /// 180us sixteen thousand labels took to count.
    fn bounds(&self) -> (i64, i64) {
        fn extremes<T: Copy + Into<i64> + Sync>(values: &[T]) -> (i64, i64) {
            let (low, high) = crate::ops::map::par_fold_chunks(
                values,
                crate::ops::map::PAR_CHUNK,
                (i64::MAX, i64::MIN),
                &|_, chunk| {
                    chunk.iter().fold((i64::MAX, i64::MIN), |(low, high), &v| {
                        let v: i64 = v.into();
                        (low.min(v), high.max(v))
                    })
                },
                &|(a, b), (c, d)| (a.min(c), b.max(d)),
            );
            if values.is_empty() {
                (0, -1)
            } else {
                (low, high)
            }
        }
        match self {
            Labels::I64(values) => extremes(values),
            Labels::I32(values) => extremes(values),
            Labels::Bool(values) => {
                let any =
                    crate::ops::map::par_any_chunk(values, crate::ops::map::PAR_CHUNK, &|c| {
                        c.contains(&true)
                    });
                (0, i64::from(any))
            }
        }
    }
}

/// How many times each non-negative integer occurs in `labels`, or -- with
/// `weights` -- the total weight sitting on it.
///
/// The output is as long as the largest label needs, or `minlength`, whichever
/// is more. Weighted counts take the weights' dtype, unweighted ones are
/// `int64`. Neither is differentiable: a count moves in jumps as a label
/// changes, so both detach.
pub fn bincount(labels: &Tensor, weights: Option<&Tensor>, minlength: usize) -> Result<Tensor> {
    if labels.ndim() != 1 {
        return Err(MinitensorError::invalid_operation(
            "bincount input must be 1-D",
        ));
    }
    let labels_contiguous = labels.contiguous()?;
    let data = labels_contiguous.data();
    let borrowed = match labels.dtype() {
        DataType::Int64 => Labels::I64(
            data.as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("bincount: input is not int64"))?,
        ),
        DataType::Int32 => Labels::I32(
            data.as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("bincount: input is not int32"))?,
        ),
        DataType::Bool => Labels::Bool(
            data.as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("bincount: input is not bool"))?,
        ),
        dtype => {
            return Err(MinitensorError::type_mismatch(
                "an integer or bool dtype",
                format!("{dtype:?}"),
            ));
        }
    };

    let count = borrowed.len();
    let (lowest, highest) = borrowed.bounds();
    if lowest < 0 {
        return Err(MinitensorError::invalid_operation(format!(
            "bincount input values must be non-negative, got {lowest}"
        )));
    }
    let reached = if count == 0 {
        0
    } else {
        usize::try_from(highest)
            .ok()
            .and_then(|top| top.checked_add(1))
            .ok_or_else(|| MinitensorError::invalid_operation("bincount output size overflow"))?
    };
    let bins = reached.max(minlength);
    let device = labels.device();

    let Some(weights) = weights else {
        // The dtype is settled once per band rather than once per label.
        let counts = tally_bands(count, bins, 0i64, |first, last, slots| match &borrowed {
            Labels::I64(values) => count_labels(&values[first..last], slots, |v| v as usize),
            Labels::I32(values) => count_labels(&values[first..last], slots, |v| v as usize),
            Labels::Bool(values) => count_labels(&values[first..last], slots, usize::from),
        });
        return Ok(counted_tensor(
            TensorData::from_vec_i64(counts, device),
            DataType::Int64,
            bins,
            device,
        ));
    };

    if weights.shape().dims() != labels.shape().dims() {
        return Err(MinitensorError::invalid_operation(
            "weights must have the same shape as input",
        ));
    }
    let weights_contiguous = weights.contiguous()?;
    match weights.dtype() {
        DataType::Float32 => {
            let values = weights_contiguous.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("bincount: weights are not float32")
            })?;
            let totals = tally(count, bins, 0f32, |index| {
                Some((borrowed.at(index) as usize, values[index]))
            });
            Ok(counted_tensor(
                TensorData::from_vec_f32(totals, device),
                DataType::Float32,
                bins,
                device,
            ))
        }
        DataType::Float64 => {
            let values = weights_contiguous.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("bincount: weights are not float64")
            })?;
            let totals = tally(count, bins, 0f64, |index| {
                Some((borrowed.at(index) as usize, values[index]))
            });
            Ok(counted_tensor(
                TensorData::from_vec_f64(totals, device),
                DataType::Float64,
                bins,
                device,
            ))
        }
        dtype => Err(MinitensorError::type_mismatch(
            "a floating-point dtype",
            format!("{dtype:?}"),
        )),
    }
}

/// A one-dimensional, non-differentiable result of `bins` counts.
fn counted_tensor(
    data: TensorData,
    dtype: DataType,
    bins: usize,
    device: crate::device::Device,
) -> Tensor {
    Tensor::new(Arc::new(data), Shape::new(vec![bins]), dtype, device, false)
}

/// The edges [`histogram`] would use, without counting anything into them.
///
/// The counting is the expensive half -- a binary search per value against the
/// edges -- and a caller choosing one set of edges to share between several
/// tensors does not want it. `histogram_bin_edges` used to run the whole
/// histogram and drop the counts: 12.4ms for a million values where the edges
/// alone take 0.7.
pub fn histogram_edges(
    input: &Tensor,
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
) -> Result<Tensor> {
    let edges = bin_edges(input, bins, range)?;
    doubles_to_tensor(edges, input.device())
}

/// One value type's part in [`finite_extremes`].
///
/// The fold runs in the values' own width rather than in `f64`. Widening each
/// element first cost 0.477ms over a million float32 against 0.431 for the
/// same count of float64 -- the same work on half the bytes, which is the
/// shape of a loop paying for a conversion rather than for memory.
trait Extremes: Copy + Send + Sync {
    /// Accumulators to run in parallel. A single running extremum makes the
    /// compare-and-replace a serial dependency across the whole slice, which
    /// is the thing the vectorizer cannot break; several independent ones let
    /// it fill a register instead. The counts are measured rather than
    /// derived -- eight for `f32`, four for the rest -- and on this toolchain
    /// picking four for `f32` instead cost it about a third.
    const LANES: usize;
    /// Whether to find the two extremes in separate loops over each chunk
    /// rather than in one loop carrying both.
    ///
    /// One loop is the better shape when it vectorizes: it reads the data
    /// once, and for `f64` it emits `minpd`/`maxpd` and runs at memory speed.
    /// But two accumulator arrays are twice the live chains, and for `f32`
    /// that was over whatever budget makes the vectorizer try -- it emitted
    /// scalar `minss` and cost 0.30ns an element, against 0.12 for `f64`
    /// doing the same work on twice the bytes. Split into two loops, `f32`
    /// takes the single-accumulator shape `min_all_f32` already proves
    /// vectorizes, and the second read is nearly free because a chunk is 32KB
    /// and stays in cache: 0.11ns an element. `f64` splits the other way,
    /// 0.12 fused against 0.21 split, because its chunk is 64KB and the
    /// second read leaves cache. Each takes the one it measured faster.
    ///
    /// Both are the same reduction and must stay so; only the loop nest
    /// differs. Nothing here is portable -- it is one toolchain's
    /// vectorizer -- but a measured constant beats picking the shape that
    /// happens to be slower for the library's default dtype.
    const SPLIT: bool;
    /// Below every value, so folding it in changes nothing.
    fn low_identity() -> Self;
    /// Above every value.
    fn high_identity() -> Self;
    /// Whether this value belongs in the answer. Only the floats can say no.
    fn keep(self) -> bool;
    /// Strictly past the incumbent, which is what decides whether it is
    /// replaced. A comparison against a NaN is false either way, so writing
    /// the test in this direction discards a NaN rather than adopting it --
    /// that is what lets the cheap pass ignore the question entirely.
    fn below(self, other: Self) -> bool;
    fn above(self, other: Self) -> bool;
    /// The largest float at or below the value, which is the value itself for
    /// every type a float holds exactly.
    fn widen_down(self) -> f64;
    /// The smallest float at or above the value.
    fn widen_up(self) -> f64;
}

/// Declares [`Extremes`] for a type that is always finite, so `keep` is a
/// constant the loop compiles away.
macro_rules! exact_extremes {
    ($ty:ty, $low:expr, $high:expr, $lanes:expr, $split:expr) => {
        impl Extremes for $ty {
            const LANES: usize = $lanes;
            const SPLIT: bool = $split;
            fn low_identity() -> Self {
                $high
            }
            fn high_identity() -> Self {
                $low
            }
            fn keep(self) -> bool {
                true
            }
            fn below(self, other: Self) -> bool {
                self < other
            }
            fn above(self, other: Self) -> bool {
                self > other
            }
            // Rounded outward, so a range taken from the data spans all of
            // it: past 2^53 the nearest float to the maximum can be below it,
            // and the exact count would then leave the maximum out.
            fn widen_down(self) -> f64 {
                float_at_or_below(self as i64)
            }
            fn widen_up(self) -> f64 {
                float_at_or_above(self as i64)
            }
        }
    };
}

/// Declares [`Extremes`] for a float, where `keep` is the finiteness test and
/// the identities are the infinities.
macro_rules! float_extremes {
    ($ty:ty, $lanes:expr, $split:expr) => {
        impl Extremes for $ty {
            const LANES: usize = $lanes;
            const SPLIT: bool = $split;
            fn low_identity() -> Self {
                <$ty>::INFINITY
            }
            fn high_identity() -> Self {
                <$ty>::NEG_INFINITY
            }
            fn keep(self) -> bool {
                self.is_finite()
            }
            fn below(self, other: Self) -> bool {
                self < other
            }
            fn above(self, other: Self) -> bool {
                self > other
            }
            fn widen_down(self) -> f64 {
                self as f64
            }
            fn widen_up(self) -> f64 {
                self as f64
            }
        }
    };
}

float_extremes!(f32, 8, true);
float_extremes!(f64, 4, false);
exact_extremes!(i32, i32::MIN, i32::MAX, 4, false);
exact_extremes!(i64, i64::MIN, i64::MAX, 4, false);

/// The finite minimum and maximum in one pass, in parallel.
///
/// Non-finite values are skipped rather than propagated: the edges have to be
/// finite and increasing, and a NaN in the data is not a reason to refuse to
/// bin the rest of it: a plain min and max would make one NaN the whole
/// range's end, and nothing could be binned. A tensor
/// with no finite value at all comes back as an empty range for the caller to
/// substitute, which is the one case where there is nothing better to do.
///
/// Read from the tensor's own buffer rather than from a `float64` copy of it.
/// The copy is what the counting pass needs and it is not free -- 32MB for
/// four million values, which cost more than the scan it was feeding.
///
/// Chunked and lane-blocked, for the two reasons the value reductions in
/// `ops::reduction` are. A `par_iter().map().filter().fold()` hands rayon one
/// work item per element and a closure chain it cannot see through, and a
/// single running pair makes the compare-and-select a serial dependency that
/// cannot vectorize either. Together they cost 0.846ms over a million float32
/// where `min` and `max` through those reductions cost 0.134 between them --
/// which was most of `histogram_bin_edges`, an op that is otherwise a
/// `linspace` over two numbers.
fn finite_extremes(tensor: &Tensor) -> Result<(f64, f64)> {
    /// The widest [`Extremes::LANES`], so one array type serves every dtype.
    /// Only the first `T::LANES` of it are ever touched.
    const MAX_LANES: usize = 8;

    /// One pass of the lane-blocked fold. `SKIP` decides whether a value is
    /// tested before it is folded in; both instances are monomorphised, so the
    /// cheap one carries no trace of the test.
    fn pass<T: Extremes, const SKIP: bool>(values: &[T]) -> (T, T) {
        par_fold_chunks(
            values,
            MINMAX_CHUNK,
            (T::low_identity(), T::high_identity()),
            &|_, chunk| {
                let lanes = T::LANES;
                let mut lows = [T::low_identity(); MAX_LANES];
                let mut highs = [T::high_identity(); MAX_LANES];
                // See `Extremes::SPLIT` for why this is two shapes and not
                // one. Both leave `lows` and `highs` holding the same values;
                // only the order the elements are visited in differs, and an
                // extremum does not care.
                if T::SPLIT {
                    for block in chunk.chunks_exact(lanes) {
                        for lane in 0..lanes {
                            let value = block[lane];
                            if SKIP && !value.keep() {
                                continue;
                            }
                            if value.below(lows[lane]) {
                                lows[lane] = value;
                            }
                        }
                    }
                    for block in chunk.chunks_exact(lanes) {
                        for lane in 0..lanes {
                            let value = block[lane];
                            if SKIP && !value.keep() {
                                continue;
                            }
                            if value.above(highs[lane]) {
                                highs[lane] = value;
                            }
                        }
                    }
                } else {
                    for block in chunk.chunks_exact(lanes) {
                        for lane in 0..lanes {
                            let value = block[lane];
                            if SKIP && !value.keep() {
                                continue;
                            }
                            if value.below(lows[lane]) {
                                lows[lane] = value;
                            }
                            if value.above(highs[lane]) {
                                highs[lane] = value;
                            }
                        }
                    }
                }
                let blocks = chunk.chunks_exact(lanes);
                let mut low = T::low_identity();
                let mut high = T::high_identity();
                for lane in 0..lanes {
                    if lows[lane].below(low) {
                        low = lows[lane];
                    }
                    if highs[lane].above(high) {
                        high = highs[lane];
                    }
                }
                for &value in blocks.remainder() {
                    if SKIP && !value.keep() {
                        continue;
                    }
                    if value.below(low) {
                        low = value;
                    }
                    if value.above(high) {
                        high = value;
                    }
                }
                (low, high)
            },
            &|a, b| {
                (
                    if b.0.below(a.0) { b.0 } else { a.0 },
                    if b.1.above(a.1) { b.1 } else { a.1 },
                )
            },
        )
    }

    /// The pair for one dtype, taking the skipping pass only when it can
    /// change the answer.
    ///
    /// The first pass asks nothing about the values, and for the dtypes that
    /// are always finite there is nothing to ask. For the floats it is still
    /// almost always enough: a comparison against a NaN is false either way,
    /// so a NaN never displaces a real value and is skipped at no cost, and
    /// only an actual infinity can reach the result. Testing every element for
    /// that cost 0.385ns each against a plain extremum's 0.062 -- six passes'
    /// worth of arithmetic to find something that is usually not there. So it
    /// is asked once, of the answer, instead of a billion times of the input.
    fn fold<T: Extremes>(values: &[T]) -> (f64, f64) {
        let (low, high) = pass::<T, false>(values);
        if low.keep() && high.keep() {
            return (low.widen_down(), high.widen_up());
        }
        // An infinity got through, or there was no value to find at all. The
        // second pass settles which: it returns the identities for an input
        // with nothing finite in it, and those widen to an empty range that
        // `bin_edges` reads as `low > high` and substitutes.
        let (low, high) = pass::<T, true>(values);
        (low.widen_down(), high.widen_up())
    }

    let contiguous = tensor.contiguous()?;
    let data = contiguous.data();
    Ok(match tensor.dtype() {
        DataType::Float32 => fold(data.as_f32_slice().ok_or_else(dtype_mismatch)?),
        DataType::Float64 => fold(data.as_f64_slice().ok_or_else(dtype_mismatch)?),
        DataType::Int32 => fold(data.as_i32_slice().ok_or_else(dtype_mismatch)?),
        DataType::Int64 => fold(data.as_i64_slice().ok_or_else(dtype_mismatch)?),
        // A boolean's extremes are decidable without folding at all: the
        // maximum is `true` exactly when some value is, and the minimum
        // `false` exactly when some value is not. Both scans short-circuit, so
        // a mixed tensor usually answers inside its first chunk where the fold
        // read every element of it -- 0.23ns each, the worst of any dtype,
        // because eight one-byte lanes do not fill a register.
        DataType::Bool => {
            let values = data.as_bool_slice().ok_or_else(dtype_mismatch)?;
            let any = par_any_chunk(values, MINMAX_CHUNK, &|chunk| chunk.contains(&true));
            let all = par_all_chunk(values, MINMAX_CHUNK, &|chunk| !chunk.contains(&false));
            // An empty tensor is `all` and not `any`, which comes out as the
            // empty range the callers already substitute for.
            (if all { 1.0 } else { 0.0 }, if any { 1.0 } else { 0.0 })
        }
    })
}

fn bin_edges(input: &Tensor, bins: Bins<'_>, range: Option<(f64, f64)>) -> Result<Vec<f64>> {
    match bins {
        Bins::Edges(tensor) => {
            if tensor.ndim() != 1 {
                return Err(MinitensorError::invalid_operation(
                    "histogram: the bin edges must be one-dimensional",
                ));
            }
            let edges = as_doubles(tensor)?;
            if edges.len() < 2 {
                return Err(MinitensorError::invalid_operation(
                    "histogram: at least two bin edges are needed to make a bin",
                ));
            }
            if edges.windows(2).any(|pair| pair[1] <= pair[0]) {
                return Err(MinitensorError::invalid_operation(
                    "histogram: the bin edges must increase",
                ));
            }
            Ok(edges)
        }
        Bins::Count(count) => {
            if count == 0 {
                return Err(MinitensorError::invalid_operation(
                    "histogram: at least one bin is needed",
                ));
            }
            // `count + 1` edges and `count` counts are about to be built, so a
            // count no memory could hold is an error here rather than a
            // capacity-overflow panic inside the allocation.
            let edge_count = count
                .checked_add(1)
                .ok_or_else(|| MinitensorError::invalid_operation("histogram: too many bins"))?;
            crate::tensor::TensorData::ensure_allocatable(edge_count, DataType::Float64)?;
            let (low, high) = match range {
                // Given a range, the data is not read at all.
                Some(pair) => pair,
                None => {
                    let (low, high) = finite_extremes(input)?;
                    if low > high { (0.0, 1.0) } else { (low, high) }
                }
            };
            // A range of no width still has to produce bins, so it is opened
            // by half a unit either side, which puts a constant sample in the
            // middle bin.
            let (low, high) = if low == high {
                (low - 0.5, high + 0.5)
            } else {
                (low, high)
            };
            if !(low.is_finite() && high.is_finite() && low < high) {
                return Err(MinitensorError::invalid_operation(
                    "histogram: the range must be finite and increasing",
                ));
            }
            let width = (high - low) / count as f64;
            let mut edges: Vec<f64> = if width.is_finite() {
                (0..=count)
                    .map(|index| low + index as f64 * width)
                    .collect()
            } else {
                // Both ends finite and the span between them not: halved, the
                // span fits, and doubling an edge back is exact. Spaced the
                // ordinary way the step was infinite and the first edge
                // `low + 0 * inf`, a NaN.
                let half_width = (high / 2.0 - low / 2.0) / count as f64;
                (0..=count)
                    .map(|index| 2.0 * (low / 2.0 + index as f64 * half_width))
                    .collect()
            };
            // The arithmetic above can miss the top edge by a rounding, and the
            // top edge is the one the closed last bin depends on.
            edges[count] = high;
            Ok(edges)
        }
    }
}

/// Which of a histogram's bins a value falls in.
///
/// A value's bin is the last edge at or below it, so finding it is a search,
/// and written with a branch on each comparison it cost 46ms for four million
/// values into a hundred bins: on unsorted data each branch goes either way,
/// and every wrong guess flushes the pipeline. Evenly spaced edges need no
/// search -- the scaled offset names the bin to within a rounding, and a step
/// to a neighbour settles it against the edges themselves, so the answer is
/// the one the search gives. Edges of any other spacing are still searched,
/// but without branching: the comparison picks the next probe rather than a
/// jump, and the loop runs the same length for every value.
struct BinFinder<'a> {
    edges: &'a [f64],
    /// Half the first edge, and the bins per half unit, when the edges are
    /// evenly spaced. Halved because the span between two finite edges can
    /// exceed the float range, and half of it cannot.
    spacing: Option<(f64, f64)>,
}

impl<'a> BinFinder<'a> {
    fn new(edges: &'a [f64], evenly_spaced: bool) -> Self {
        let bins = (edges.len() - 1) as f64;
        let first = edges[0] / 2.0;
        let spacing = evenly_spaced.then(|| (first, bins / (edges[edges.len() - 1] / 2.0 - first)));
        Self { edges, spacing }
    }

    /// The bin holding `value`, or one past the last bin outside the edges
    /// and for a NaN. The last bin is closed, so the top edge itself is in it.
    ///
    /// Branch-free but for a step that is almost never taken, so a mix of
    /// values inside and outside the edges costs what either does alone.
    #[inline]
    fn slot(&self, value: f64) -> usize {
        let edges = self.edges;
        let last = edges.len() - 2;
        // False for a NaN, and for an infinity, since the edges are finite.
        let inside = (value >= edges[0]) & (value <= edges[last + 1]);
        let slot = match self.spacing {
            Some((first, scale)) => {
                // Truncation saturates, and takes a NaN to zero. A branch for
                // the clamp mispredicted on every value in the top bin.
                let guess = ((value / 2.0 - first) * scale) as usize;
                let mut slot = select_unpredictable(guess > last, last, guess);
                // Within a rounding of an edge the guess can be a bin off. The
                // test does not short-circuit, so it is one branch, and that
                // one almost never taken.
                if inside & ((value < edges[slot]) | ((value >= edges[slot + 1]) & (slot < last))) {
                    while slot > 0 && value < edges[slot] {
                        slot -= 1;
                    }
                    while slot < last && value >= edges[slot + 1] {
                        slot += 1;
                    }
                }
                slot
            }
            None => {
                // The last edge at or below `value`, which the first is.
                let (mut base, mut size) = (0, edges.len());
                while size > 1 {
                    let half = size / 2;
                    let probe = base + half;
                    base = select_unpredictable(edges[probe] <= value, probe, base);
                    size -= half;
                }
                select_unpredictable(base > last, last, base)
            }
        };
        select_unpredictable(inside, slot, last + 1)
    }

    /// [`slot`](Self::slot) for an int64, which a float holds exactly only up
    /// to 2^53. Past it the nearest float can be an edge the integer is not
    /// at, so which side of each edge it is on is asked exactly: rounded,
    /// `2^60 - 1` counted in the bin that starts at `2^60`.
    #[inline]
    fn integer_slot(&self, value: i64) -> usize {
        if value.unsigned_abs() <= 1 << 53 {
            return self.slot(value as f64);
        }
        let count = |right| {
            let (key, side) = integer_among_floats(value, right);
            locate(self.edges, &key, side)
        };
        let (at_or_below, below) = (count(true), count(false));
        let bins = self.edges.len() - 1;
        if at_or_below == 0 || below == self.edges.len() {
            return bins;
        }
        (at_or_below - 1).min(bins - 1)
    }
}

/// The weight falling in each bin, read from the tensor in its own dtype.
///
/// Read in place rather than from a float64 copy, which for four million
/// float32 values was 32MB written and read again to feed a loop that only
/// compares. An unweighted count is kept in integers, by [`count_binned`]:
/// added one at a time into a float, a run of values in one bin made each
/// increment wait for the last, and a single bin took 4.5ms for four million.
fn bin_counts(input: &Tensor, finder: &BinFinder<'_>, weights: Option<&[f64]>) -> Result<Vec<f64>> {
    fn count<T: Copy + Sync>(
        values: &[T],
        bins: usize,
        weights: Option<&[f64]>,
        slot: impl Fn(T) -> usize + Sync,
    ) -> Vec<f64> {
        match weights {
            Some(weights) => tally(values.len(), bins, 0.0, |index| {
                let bin = slot(values[index]);
                (bin < bins).then(|| (bin, weights[index]))
            }),
            // One slot more than there are bins, for the values that fall in
            // none, and dropped once they are counted.
            None => {
                let mut counts = tally_bands(values.len(), bins + 1, 0i64, |first, last, slots| {
                    count_binned(&values[first..last], slots, &slot)
                });
                counts.pop();
                counts.into_iter().map(|count| count as f64).collect()
            }
        }
    }
    let bins = finder.edges.len() - 1;
    let contiguous = input.contiguous()?;
    let data = contiguous.data();
    Ok(match input.dtype() {
        DataType::Float32 => count(
            data.as_f32_slice().ok_or_else(dtype_mismatch)?,
            bins,
            weights,
            |v| finder.slot(f64::from(v)),
        ),
        DataType::Float64 => count(
            data.as_f64_slice().ok_or_else(dtype_mismatch)?,
            bins,
            weights,
            |v| finder.slot(v),
        ),
        DataType::Int32 => count(
            data.as_i32_slice().ok_or_else(dtype_mismatch)?,
            bins,
            weights,
            |v| finder.slot(f64::from(v)),
        ),
        DataType::Int64 => count(
            data.as_i64_slice().ok_or_else(dtype_mismatch)?,
            bins,
            weights,
            |v| finder.integer_slot(v),
        ),
        DataType::Bool => count(
            data.as_bool_slice().ok_or_else(dtype_mismatch)?,
            bins,
            weights,
            |v| finder.slot(if v { 1.0 } else { 0.0 }),
        ),
    })
}

pub fn histogram(
    input: &Tensor,
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
    weights: Option<&Tensor>,
    density: bool,
) -> Result<(Tensor, Tensor)> {
    let evenly_spaced = matches!(bins, Bins::Count(_));
    let edges = bin_edges(input, bins, range)?;
    let weights = match weights {
        Some(tensor) => {
            if tensor.numel() != input.numel() {
                return Err(MinitensorError::invalid_operation(format!(
                    "histogram: {} weights for {} values",
                    tensor.numel(),
                    input.numel()
                )));
            }
            Some(as_doubles(tensor)?)
        }
        None => None,
    };

    let mut counts = bin_counts(
        input,
        &BinFinder::new(&edges, evenly_spaced),
        weights.as_deref(),
    )?;

    if density {
        let total: f64 = counts.iter().sum();
        if total > 0.0 {
            for (slot, count) in counts.iter_mut().enumerate() {
                *count /= total * (edges[slot + 1] - edges[slot]);
            }
        }
    }

    Ok((
        doubles_to_tensor(counts, input.device())?,
        doubles_to_tensor(edges, input.device())?,
    ))
}

/// Counts over `bins` equal-width bins spanning `[min, max]`.
///
/// A second spelling of [`histogram`], with two differences it is worth being
/// exact about: the edges are not returned, and `min == max` means "span the
/// data" rather than "an empty range".
pub fn histc(input: &Tensor, bins: usize, min: f64, max: f64) -> Result<Tensor> {
    let range = if min == max { None } else { Some((min, max)) };
    Ok(histogram(input, Bins::Count(bins), range, None, false)?.0)
}

#[cfg(test)]
mod mixed_search_tests {
    use super::*;
    use crate::Device;

    fn place(sequence: &[i64], value: f64, right: bool) -> usize {
        match float_among_integers(value, right) {
            AmongIntegers::At(key) => locate(sequence, &key, right),
            AmongIntegers::Start => 0,
            AmongIntegers::End => sequence.len(),
        }
    }

    #[test]
    fn a_float_among_integers_lands_where_exact_arithmetic_puts_it() {
        let sequence = [-3_i64, 1, 2, 2, 5];
        for (value, left, right) in [
            (1.5, 2, 2),
            (2.0, 2, 4),
            (-3.5, 0, 0),
            (-3.0, 0, 1),
            (5.0, 4, 5),
            (5.5, 5, 5),
            (f64::INFINITY, 5, 5),
            (f64::NEG_INFINITY, 0, 0),
            (f64::NAN, 5, 5),
            (1e300, 5, 5),
            (-1e300, 0, 0),
        ] {
            assert_eq!(place(&sequence, value, false), left, "{value} left");
            assert_eq!(place(&sequence, value, true), right, "{value} right");
        }
        // The int64 extremes, which no float past them can name.
        let extremes = [i64::MIN, 0, i64::MAX];
        assert_eq!(place(&extremes, -(2.0_f64.powi(63)), false), 0);
        assert_eq!(place(&extremes, -(2.0_f64.powi(63)), true), 1);
        assert_eq!(place(&extremes, 2.0_f64.powi(63), false), 3);
    }

    #[test]
    fn an_integer_among_floats_lands_where_exact_arithmetic_puts_it() {
        let place = |sequence: &[f64], value: i64, right: bool| {
            let (key, side) = integer_among_floats(value, right);
            locate(sequence, &key, side)
        };
        let big = 2_f64.powi(53);
        // 2^53 + 1 is not a float64: it sits strictly between 2^53 and 2^53 + 2.
        let sequence = [0.5, big, big + 2.0, f64::NAN];
        let odd = (1_i64 << 53) + 1;
        assert_eq!(place(&sequence, odd, false), 2);
        assert_eq!(place(&sequence, odd, true), 2);
        assert_eq!(place(&sequence, 1 << 53, false), 1);
        assert_eq!(place(&sequence, 1 << 53, true), 2);
        assert_eq!(place(&sequence, 0, false), 0);
        assert_eq!(place(&sequence, 1, false), 1);
        // i64::MAX rounds up to 2^63, which is above it.
        assert_eq!(place(&[2_f64.powi(63)], i64::MAX, false), 0);
        assert_eq!(place(&[2_f64.powi(63)], i64::MAX, true), 0);
        assert_eq!(place(&[-(2_f64.powi(63))], i64::MIN, false), 0);
        assert_eq!(place(&[-(2_f64.powi(63))], i64::MIN, true), 1);
        // Below zero the step is toward more negative, still below.
        let negative = -((1_i64 << 53) + 1);
        assert_eq!(place(&[-big - 2.0, -big], negative, false), 1);
    }

    #[test]
    fn mixed_and_widened_pairs_search_rather_than_refuse() {
        let device = Device::cpu();
        let integers = Tensor::new(
            Arc::new(TensorData::from_vec_i32(vec![1, 2, 3], device)),
            Shape::new(vec![3]),
            DataType::Int32,
            device,
            false,
        );
        let halves = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.5, 3.0], device)),
            Shape::new(vec![2]),
            DataType::Float32,
            device,
            false,
        );
        let found = searchsorted(&integers, &halves, false).unwrap();
        assert_eq!(found.data().as_i64_slice().unwrap(), &[1, 2]);
        let doubles = halves.astype(DataType::Float64).unwrap();
        let widened = searchsorted(&doubles, &halves, true).unwrap();
        assert_eq!(widened.data().as_i64_slice().unwrap(), &[1, 2]);
    }
}
