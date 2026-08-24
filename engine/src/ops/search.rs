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
    tensor::{DataType, Shape, Tensor, TensorData},
};
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
    } else if value_rows == 0 {
        0
    } else {
        values.numel() / value_rows
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
    let mut counts = vec![0.0f64; bin_count];
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < edges[0] || *value > edges[bin_count] {
            continue;
        }
        let slot = locate(&edges, value, true)
            .saturating_sub(1)
            .min(bin_count - 1);
        counts[slot] += match &weights {
            Some(weights) => weights[index],
            None => 1.0,
        };
    }

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
