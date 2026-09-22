// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Asking a sorted sequence where a value belongs.
//!
//! Everything here is one binary search wearing four hats. `searchsorted` is the
//! search itself; `bucketize` is the same call with the arguments the other way
//! round, which is the spelling PyTorch uses and the one that reads correctly
//! when the sequence is a fixed set of boundaries; `histogram` is the search
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
use rayon::prelude::*;
use std::sync::Arc;

/// Where `value` belongs in an already-sorted `sequence`.
///
/// `right` picks which end of a run of equals to answer with: `false` gives the
/// first index whose element is greater than or equal to `value`, so the value
/// would be inserted *before* its equals; `true` gives the first strictly
/// greater, inserting *after* them. On a sequence with no duplicates the two
/// agree everywhere except exactly on an element.
fn locate<T: PartialOrd>(sequence: &[T], value: &T, right: bool) -> usize {
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
/// gives a meaningless answer rather than an error, which is also what NumPy and
/// PyTorch do.
///
/// The result is `int64` positions in `0..=width`, so a value past the end of
/// the sequence answers with the length rather than being clamped into it.
pub fn searchsorted(sequence: &Tensor, values: &Tensor, right: bool) -> Result<Tensor> {
    if sequence.ndim() == 0 {
        return Err(MinitensorError::invalid_operation(
            "searchsorted: the sequence must have at least one dimension",
        ));
    }
    if sequence.dtype() != values.dtype() {
        return Err(MinitensorError::invalid_operation(format!(
            "searchsorted: the sequence is {:?} and the values are {:?}; they must match",
            sequence.dtype(),
            values.dtype()
        )));
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
        // Every value's search is independent of every other's, which is what
        // makes this parallel at all -- and it was not, which cost more than
        // the search: a million values against a million-element sequence took
        // 400 ms on one core while NumPy's `isin`, which is this search plus a
        // sort, took 102 ms for the whole thing.
        //
        // One search is about `log2(width)` probes and, past the cache, each is
        // a miss. A task wants a few thousand probes in it before the split
        // pays, so the chunk is that budget divided by the depth of one search:
        // a short sequence gets wide chunks because each search in it is cheap,
        // a long one gets narrow chunks because each is not.
        let batch = sequence.ndim() > 1;
        let depth = (usize::BITS - width.leading_zeros()).max(1) as usize;
        let chunk = ((1 << 12) / depth).max(1);
        with_pair!(ordered, queried, haystack, needles, {
            par_out_chunks(out, chunk, &|start, block| {
                for (offset, slot) in block.iter_mut().enumerate() {
                    let index = start + offset;
                    let row = if batch { index / per_row.max(1) } else { 0 };
                    let begin = row * width;
                    *slot = locate(&haystack[begin..begin + width], &needles[index], right) as i64;
                }
            });
        });
    }

    Ok(Tensor::new(
        Arc::new(data),
        values.shape().clone(),
        DataType::Int64,
        values.device(),
        false,
    ))
}

/// Which bucket each element of `input` falls in, given the bucket
/// `boundaries`.
///
/// [`searchsorted`] with the arguments the other way round. Both spellings exist
/// because both readings are natural -- one asks where a value goes in a
/// sequence, the other asks which bucket a value is in -- and PyTorch ships
/// both for the same reason.
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
const TALLY_MIN_BAND: usize = 1 << 14;
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
    if bins == 0 || count == 0 {
        return vec![zero; bins];
    }
    let band = count.div_ceil(TALLY_BANDS).max(TALLY_MIN_BAND);
    let bands = count.div_ceil(band);
    if bands < 2 || bins > TALLY_PRIVATE_BINS {
        let mut slots = vec![zero; bins];
        for index in 0..count {
            if let Some((slot, value)) = contribution(index) {
                slots[slot] = slots[slot] + value;
            }
        }
        return slots;
    }
    let partials = par_map_indexed(bands, &|index| {
        let first = index * band;
        let last = (first + band).min(count);
        let mut slots = vec![zero; bins];
        for position in first..last {
            if let Some((slot, value)) = contribution(position) {
                slots[slot] = slots[slot] + value;
            }
        }
        slots
    });
    pairwise_fold_vectors(partials, |a, b| a + b)
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
    fn bounds(&self) -> (i64, i64) {
        match self {
            Labels::I64(values) => (
                values.par_iter().copied().min().unwrap_or(0),
                values.par_iter().copied().max().unwrap_or(-1),
            ),
            Labels::I32(values) => (
                values.par_iter().copied().min().unwrap_or(0).into(),
                values.par_iter().copied().max().unwrap_or(-1).into(),
            ),
            Labels::Bool(values) => (
                0,
                if values.par_iter().copied().any(|flag| flag) {
                    1
                } else {
                    0
                },
            ),
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
        let counts = tally(count, bins, 0i64, |index| {
            Some((borrowed.at(index) as usize, 1i64))
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
    fn widen(self) -> f64;
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
            fn widen(self) -> f64 {
                self as f64
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
            fn widen(self) -> f64 {
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
/// bin the rest of it. This is a deliberate divergence -- NumPy takes a plain
/// min and max, so one NaN makes its range `[nan, nan]` and it raises
/// "autodetected range is not finite" rather than binning anything. A tensor
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
            return (low.widen(), high.widen());
        }
        // An infinity got through, or there was no value to find at all. The
        // second pass settles which: it returns the identities for an input
        // with nothing finite in it, and those widen to an empty range that
        // `bin_edges` reads as `low > high` and substitutes.
        let (low, high) = pass::<T, true>(values);
        (low.widen(), high.widen())
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
            let (low, high) = match range {
                // Given a range, the data is not read at all.
                Some(pair) => pair,
                None => {
                    let (low, high) = finite_extremes(input)?;
                    if low > high { (0.0, 1.0) } else { (low, high) }
                }
            };
            // A range of no width still has to produce bins, so it is opened
            // by half a unit either side -- NumPy's rule, and the only one that
            // puts a constant sample somewhere sensible.
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
            let mut edges: Vec<f64> = (0..=count)
                .map(|index| low + index as f64 * width)
                .collect();
            // The arithmetic above can miss the top edge by a rounding, and the
            // top edge is the one the closed last bin depends on.
            edges[count] = high;
            Ok(edges)
        }
    }
}

pub fn histogram(
    input: &Tensor,
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
    weights: Option<&Tensor>,
    density: bool,
) -> Result<(Tensor, Tensor)> {
    let edges = bin_edges(input, bins, range)?;
    let values = as_doubles(input)?;
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

    let bin_count = edges.len() - 1;
    let mut counts = tally(values.len(), bin_count, 0.0f64, |index| {
        let value = values[index];
        if !value.is_finite() || value < edges[0] || value > edges[bin_count] {
            return None;
        }
        let slot = locate(&edges, &value, true)
            .saturating_sub(1)
            .min(bin_count - 1);
        Some((
            slot,
            match &weights {
                Some(weights) => weights[index],
                None => 1.0,
            },
        ))
    });

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
/// PyTorch's spelling of [`histogram`], with two differences it is worth being
/// exact about: the edges are not returned, and `min == max` means "span the
/// data" rather than "an empty range".
pub fn histc(input: &Tensor, bins: usize, min: f64, max: f64) -> Result<Tensor> {
    let range = if min == max { None } else { Some((min, max)) };
    Ok(histogram(input, Bins::Count(bins), range, None, false)?.0)
}
