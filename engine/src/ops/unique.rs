// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Asking which values are there, and which is there most.
//!
//! `unique` collapses a tensor to the distinct values in it, `unique_consecutive`
//! collapses only adjacent runs, and `mode` reports the value that occurs most
//! often along an axis. All three are the same walk over runs of equal elements;
//! `unique` sorts first so that equal values become adjacent, `mode` sorts and
//! keeps the longest run instead of all of them, and `unique_consecutive` does
//! not sort at all.
//!
//! None of it composes. Counting distinct values needs either a sort or a hash,
//! and no arrangement of the arithmetic and reduction operations here performs
//! either.
//!
//! ## What equality means here
//!
//! `PartialOrd` is not enough, and this is the whole numerical content of the
//! module. `NaN` is not ordered against anything, so a comparison sort over raw
//! `partial_cmp` has no defined result; and `NaN != NaN`, so a run detector over
//! `==` would emit every `NaN` in the input as its own distinct value. Both are
//! fixed by one comparison that puts `NaN` after every number and calls it equal
//! to itself -- which is what makes `unique([nan, 1.0, nan])` answer
//! `[1.0, nan]`, matching NumPy, rather than `[1.0, nan, nan]`.
//!
//! None of these is differentiable. `unique` returns a subset of its input, and
//! which subset changes discontinuously as values collide; `mode` returns a
//! value that jumps to another as counts cross. There is no derivative to hand
//! back, so they detach.

use crate::ops::order::{
    PAR_SORT_MIN_LEN, bool_key, float_key32, float_key64, int_key32, int_key64,
};
use crate::{
    error::{MinitensorError, Result},
    ops::map::{outputs_per_task, par_out_chunks2},
    ops::util::normalize_dim,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

/// An element that can be put in a total order even when its type cannot.
trait Orderable: Copy + PartialOrd {
    /// Whether this value is outside the ordering its own type provides.
    fn unordered(self) -> bool {
        false
    }
}

impl Orderable for f32 {
    fn unordered(self) -> bool {
        self.is_nan()
    }
}
impl Orderable for f64 {
    fn unordered(self) -> bool {
        self.is_nan()
    }
}
impl Orderable for i32 {}
impl Orderable for i64 {}
impl Orderable for bool {}
// The keys themselves, so [`walk_runs`] can find runs of equal values in a
// lane that was ordered by key rather than by value -- equal keys and equal
// values are the same runs, since the key folds `NaN` together and both zeros
// together exactly as [`compare`] does.
impl Orderable for u32 {}
impl Orderable for u64 {}

/// The total order the sorting and the run detection both use.
///
/// `NaN` sorts after every number and equals itself. Neither is what floating
/// point says; both are what a caller asking "which values are in here" means.
fn compare<T: Orderable>(left: T, right: T) -> Ordering {
    match (left.unordered(), right.unordered()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
    }
}

/// Walk the runs of equal values in `values`, handing each to `visit` as the
/// half-open range it occupies.
///
/// The one piece of machinery all three operations are made of. On a sorted
/// slice a run is every occurrence of a value; on an unsorted one it is only
/// the adjacent occurrences, which is exactly the difference between `unique`
/// and `unique_consecutive`.
fn walk_runs<T: Orderable, F: FnMut(usize, usize)>(values: &[T], mut visit: F) {
    let mut start = 0;
    while start < values.len() {
        let mut stop = start + 1;
        while stop < values.len() && compare(values[start], values[stop]) == Ordering::Equal {
            stop += 1;
        }
        visit(start, stop);
        start = stop;
    }
}

/// The values in ascending order, and -- only if `positions` is asked for --
/// where each of them came from.
///
/// Both halves matter. This used to sort a permutation of indices, which reads
/// the value it is comparing out of the original buffer -- two dependent loads
/// from anywhere in the tensor, per comparison -- where sorting the values
/// themselves reads what it compares. And when nobody wants the positions,
/// building them is work for no reader: `unique` over two million floats spent
/// 397ms sorting a permutation it then discarded, against NumPy's 15 for the
/// same question. It is 37ms now.
///
/// Ties need no rule here: equal values are equal by the comparison this module
/// uses -- `NaN` included -- so which of them comes out first cannot change an
/// answer built from runs of them.
fn in_order<T: Orderable + Send>(values: &[T], positions: bool) -> (Vec<T>, Vec<usize>) {
    if positions {
        let mut pairs: Vec<(T, usize)> = values.iter().copied().zip(0..).collect();
        pairs.par_sort_unstable_by(|left, right| compare(left.0, right.0));
        pairs.into_iter().unzip()
    } else {
        let mut sorted = values.to_vec();
        sorted.par_sort_unstable_by(|&left, &right| compare(left, right));
        (sorted, Vec::new())
    }
}

/// Build a tensor of `int64` from a vector.
fn indices_tensor(values: Vec<i64>, shape: Shape, device: crate::device::Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(values.len(), DataType::Int64, device);
    if !values.is_empty() {
        let slice = data
            .as_i64_slice_mut()
            .ok_or_else(|| MinitensorError::internal_error("unique: output is not int64"))?;
        slice.copy_from_slice(&values);
    }
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        DataType::Int64,
        device,
        false,
    ))
}

/// The distinct values, and whichever of the three extras was asked for: where
/// each first occurred, which distinct value each input element was, and how
/// many times each occurred -- in that order, which is NumPy's.
pub type UniqueParts = (Tensor, Option<Tensor>, Option<Tensor>, Option<Tensor>);

/// What a caller asked to be told, beyond the values themselves.
#[derive(Clone, Copy, Default)]
pub struct UniqueWanted {
    /// Where each distinct value first occurred in the input.
    pub index: bool,
    pub inverse: bool,
    pub counts: bool,
}

/// The distinct values of `tensor`, ascending, with `NaN` last and collapsed.
///
/// The input is flattened: this is a question about which values occur, not
/// about where. `inverse` gives, for every element of the input and in its
/// shape, the position of its value in the output -- so indexing the output by
/// it rebuilds the input. `counts` gives how many times each distinct value
/// occurred.
pub fn unique(tensor: &Tensor, wanted: UniqueWanted) -> Result<UniqueParts> {
    run_lengths(tensor, wanted, true)
}

/// The distinct values of *adjacent* runs, in the order they appear.
///
/// Nothing is sorted, so a value that recurs after something else appears
/// again. This is the operation you want after a sort you did yourself, or on a
/// sequence where the order carries meaning -- run-length encoding a label
/// sequence, for instance, where `unique` would destroy the very thing being
/// encoded.
pub fn unique_consecutive(tensor: &Tensor, wanted: UniqueWanted) -> Result<UniqueParts> {
    run_lengths(tensor, wanted, false)
}

/// Both flavours of `unique`, which differ only in whether the walk is over the
/// input or over its sorted order.
fn run_lengths(tensor: &Tensor, wanted: UniqueWanted, sorted: bool) -> Result<UniqueParts> {
    let contiguous = tensor.contiguous()?;
    let count = tensor.numel();
    let device = tensor.device();

    macro_rules! collect {
        ($accessor:ident, $accessor_mut:ident, $ty:ty) => {{
            let values = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("unique: dtype does not match the input")
            })?;
            // Sorting puts equal values next to each other, which is the only
            // thing separating this from the consecutive form. The positions
            // come back only when the inverse map is wanted, since that is the
            // only thing that reads them.
            let positions = wanted.inverse || wanted.index;
            let (arranged, order): (Vec<$ty>, Vec<usize>) = if sorted {
                in_order(values, positions)
            } else if positions {
                (values.to_vec(), (0..count).collect())
            } else {
                (values.to_vec(), Vec::new())
            };

            let mut distinct: Vec<$ty> = Vec::new();
            let mut counts: Vec<i64> = Vec::new();
            let mut first: Vec<i64> = Vec::new();
            let mut inverse = vec![0i64; if wanted.inverse { count } else { 0 }];
            walk_runs(&arranged, |start, stop| {
                if wanted.inverse {
                    for position in &order[start..stop] {
                        inverse[*position] = distinct.len() as i64;
                    }
                }
                if wanted.index {
                    // The sort does not keep equal values in their original
                    // order, so "where it first occurred" is the smallest
                    // position in the run rather than the one that landed
                    // first in it.
                    first.push(order[start..stop].iter().copied().min().unwrap_or(start) as i64);
                }
                distinct.push(arranged[start]);
                // A tensor of all-distinct values has one count per element,
                // which is a buffer the size of the input to build and throw
                // away when nobody asked for the counts.
                if wanted.counts {
                    counts.push((stop - start) as i64);
                }
            });

            let shape = Shape::new(vec![distinct.len()]);
            let mut data = TensorData::zeros_on_device(distinct.len(), tensor.dtype(), device);
            if !distinct.is_empty() {
                let slice = data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("unique: dtype does not match the output")
                })?;
                slice.copy_from_slice(&distinct);
            }
            let values_out = Tensor::new(Arc::new(data), shape, tensor.dtype(), device, false);

            let index_out = if wanted.index {
                let shape = Shape::new(vec![first.len()]);
                Some(indices_tensor(first, shape, device)?)
            } else {
                None
            };
            let inverse_out = if wanted.inverse {
                Some(indices_tensor(inverse, tensor.shape().clone(), device)?)
            } else {
                None
            };
            let counts_out = if wanted.counts {
                let shape = Shape::new(vec![counts.len()]);
                Some(indices_tensor(counts, shape, device)?)
            } else {
                None
            };
            (values_out, index_out, inverse_out, counts_out)
        }};
    }

    Ok(match tensor.dtype() {
        DataType::Float32 => collect!(as_f32_slice, as_f32_slice_mut, f32),
        DataType::Float64 => collect!(as_f64_slice, as_f64_slice_mut, f64),
        DataType::Int32 => collect!(as_i32_slice, as_i32_slice_mut, i32),
        DataType::Int64 => collect!(as_i64_slice, as_i64_slice_mut, i64),
        DataType::Bool => collect!(as_bool_slice, as_bool_slice_mut, bool),
    })
}

/// The value occurring most often along `dim`, and where it is.
///
/// Ties go to the smaller value, and the index reported is the *first* position
/// along `dim` holding it. Both are choices rather than consequences -- a tie
/// has no natural winner and a repeated value has no natural occurrence -- so
/// both are fixed here and tested, which is better than leaving them to fall out
/// of whatever the sort happened to do.
pub fn mode(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if tensor.ndim() == 0 {
        return Err(MinitensorError::invalid_operation(
            "mode requires a tensor with at least one dimension",
        ));
    }
    let axis = normalize_dim(dim, tensor.ndim())?;
    let dims = tensor.shape().dims().to_vec();
    let width = dims[axis];
    if width == 0 {
        return Err(MinitensorError::invalid_operation(
            "mode: the axis being reduced is empty, so there is no value to report",
        ));
    }

    // The axis being reduced goes last, which leaves the others in their
    // original order and makes each lane contiguous. An index along the axis is
    // unchanged by that move, so the answer needs no translating back.
    let mut order: Vec<isize> = (0..tensor.ndim() as isize)
        .filter(|&d| d != axis as isize)
        .collect();
    order.push(axis as isize);
    let arranged = crate::ops::shape_ops::permute(tensor, order)?.contiguous()?;
    let lanes = tensor.numel() / width;

    let mut reduced: Vec<usize> = dims.clone();
    reduced.remove(axis);
    let value_shape = Shape::new(if keepdim {
        let mut kept = dims.clone();
        kept[axis] = 1;
        kept
    } else {
        reduced.clone()
    });

    let device = tensor.device();
    let mut values = TensorData::zeros_on_device(lanes, tensor.dtype(), device);
    let mut positions = vec![0i64; lanes];

    // One long lane has nothing to spread across lanes, so it is sorted inside
    // itself instead -- the same choice the tensor sort makes, and for the same
    // reason: `mode` of a two-million-element vector is one output, so the band
    // split below hands the whole sort to a single core. That took 248ms.
    let within = lanes < rayon::current_num_threads() && width >= PAR_SORT_MIN_LEN;

    macro_rules! reduce {
        ($accessor:ident, $accessor_mut:ident, $key:expr) => {{
            let source = arranged.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("mode: dtype does not match the input")
            })?;
            let out = values.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("mode: dtype does not match the output")
            })?;
            let key = $key;
            // One lane is one output, and each sorts a copy of its own row:
            // the lanes never touch, so they are handed out a band at a time
            // sized by what one of them costs.
            par_out_chunks2(
                out,
                &mut positions,
                outputs_per_task(width),
                &|start, values_chunk, positions_chunk| {
                    let mut lane = Vec::with_capacity(width);
                    for (offset, (value_out, position_out)) in values_chunk
                        .iter_mut()
                        .zip(positions_chunk.iter_mut())
                        .enumerate()
                    {
                        let index = start + offset;
                        let row = &source[index * width..(index + 1) * width];
                        // Ordered by key rather than by value: the runs are the
                        // same ones, and an integer comparison has none of the
                        // branches the value comparison needs.
                        lane.clear();
                        lane.extend(row.iter().copied().map(key));
                        if within {
                            lane.par_sort_unstable();
                        } else {
                            lane.sort_unstable();
                        }

                        // Ascending, so the first run of maximal length is the
                        // one holding the smallest of the tied values.
                        let (mut best, mut longest) = (lane[0], 0usize);
                        walk_runs(&lane, |from, to| {
                            if to - from > longest {
                                longest = to - from;
                                best = lane[from];
                            }
                        });
                        // Read the value back out of the row rather than
                        // inverting the key: a key folds every `NaN` together
                        // and both zeros together, and the caller's own bits
                        // are the answer.
                        let at = row
                            .iter()
                            .position(|value| key(*value) == best)
                            .unwrap_or(0);
                        *value_out = row[at];
                        *position_out = at as i64;
                    }
                },
            );
        }};
    }
    match tensor.dtype() {
        DataType::Float32 => reduce!(as_f32_slice, as_f32_slice_mut, float_key32),
        DataType::Float64 => reduce!(as_f64_slice, as_f64_slice_mut, float_key64),
        DataType::Int32 => reduce!(as_i32_slice, as_i32_slice_mut, int_key32),
        DataType::Int64 => reduce!(as_i64_slice, as_i64_slice_mut, int_key64),
        DataType::Bool => reduce!(as_bool_slice, as_bool_slice_mut, bool_key),
    }

    Ok((
        Tensor::new(
            Arc::new(values),
            value_shape.clone(),
            tensor.dtype(),
            device,
            false,
        ),
        indices_tensor(positions, value_shape, device)?,
    ))
}
