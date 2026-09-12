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
    ops::map::par_map_indexed,
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
        with_pair!(ordered, queried, haystack, needles, {
            for (index, needle) in needles.iter().enumerate() {
                let row = if sequence.ndim() == 1 {
                    0
                } else {
                    index / per_row.max(1)
                };
                let start = row * width;
                out[index] = locate(&haystack[start..start + width], needle, right) as i64;
            }
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

pub fn histogram(
    input: &Tensor,
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
    weights: Option<&Tensor>,
    density: bool,
) -> Result<(Tensor, Tensor)> {
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

    let edges = match bins {
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
            edges
        }
        Bins::Count(count) => {
            if count == 0 {
                return Err(MinitensorError::invalid_operation(
                    "histogram: at least one bin is needed",
                ));
            }
            let (low, high) = match range {
                Some(pair) => pair,
                None => {
                    let finite = values.iter().copied().filter(|value| value.is_finite());
                    let low = finite.clone().fold(f64::INFINITY, f64::min);
                    let high = finite.fold(f64::NEG_INFINITY, f64::max);
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
            edges
        }
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
