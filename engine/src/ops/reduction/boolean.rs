// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::GatherBackward;
use crate::autograd::MinMaxBackward;
use crate::ops::map::{par_all_chunk, par_any_chunk, par_out_chunks, par_out_chunks2};
use crate::{
    autograd::with_grad_fn,
    error::{MinitensorError, Result},
    ops::map::{PAR_CHUNK, PAR_THRESHOLD},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::cmp::Ordering;
use std::sync::Arc;

/// Which of the two boolean folds to run.
///
/// `any` and `all` are the same reduction with two things exchanged: what the
/// answer starts as, and what ends the scan. `any` starts `false` and stops at
/// the first truthy element; `all` starts `true` and stops at the first falsy
/// one. Everything else -- how each dtype decides truthiness, the output shape,
/// the walk over the reduced axis -- is common, so it is written once below and
/// the entry points pass this.
///
/// They used to be four functions in two files, three hundred lines between
/// them, `any_along_dim` here and `all_along_dim` over in `logsumexp.rs`, each
/// with the same five dtype arms spelled out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoolFold {
    Any,
    All,
}

impl BoolFold {
    /// The answer for a slice with nothing in it, and the value a scan holds
    /// until something contradicts it. `any` of nothing is false; `all` of
    /// nothing is true.
    #[inline(always)]
    fn identity(self) -> bool {
        self == BoolFold::All
    }
}

/// Expand `$body` once per dtype, with `$input` bound to the element slice and
/// `$truthy` to the predicate that decides whether one element counts.
///
/// "Counts" is nonzero for the numeric dtypes and the value itself for `Bool`,
/// which is what makes `any`/`all` work on any tensor rather than only on masks.
macro_rules! with_truthy_slice {
    ($tensor:expr, |$input:ident, $truthy:ident| $body:expr) => {{
        macro_rules! arm {
            ($accessor:ident, $ty:ty, $tyname:literal, $pred:expr) => {{
                let $input = $tensor.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
                })?;
                let $truthy = $pred;
                $body
            }};
        }
        match $tensor.dtype() {
            DataType::Float32 => arm!(as_f32_slice, f32, "f32", |v: f32| v != 0.0),
            DataType::Float64 => arm!(as_f64_slice, f64, "f64", |v: f64| v != 0.0),
            DataType::Int32 => arm!(as_i32_slice, i32, "i32", |v: i32| v != 0),
            DataType::Int64 => arm!(as_i64_slice, i64, "i64", |v: i64| v != 0),
            DataType::Bool => arm!(as_bool_slice, bool, "bool", |v: bool| v),
        }
    }};
}

/// Wrap a boolean result as a 0-d (or all-ones, under `keepdim`) tensor.
fn bool_scalar_tensor(value: bool, tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };
    let mut data = TensorData::zeros_on_device(1, DataType::Bool, tensor.device());
    data.as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?[0] =
        value;
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

/// `any`/`all` over every element.
fn bool_fold_all(tensor: &Tensor, keepdim: bool, fold: BoolFold) -> Result<Tensor> {
    let value = with_truthy_slice!(tensor, |input, truthy| match fold {
        BoolFold::Any => par_any_chunk(input, PAR_CHUNK, &|chunk| chunk.iter().any(|&v| truthy(v))),
        BoolFold::All => par_all_chunk(input, PAR_CHUNK, &|chunk| chunk.iter().all(|&v| truthy(v))),
    });
    bool_scalar_tensor(value, tensor, keepdim)
}

/// Longest column band one task takes when the reduced axis is not the last
/// one. Wide enough that the per-task overhead disappears against a band's
/// `dim_size` passes over it, short enough that the band's accumulator and the
/// slab it is reading stay in cache together.
const BOOL_FOLD_BAND: usize = 4096;

/// How much of a contiguous run one branchless block covers, when the reduced
/// axis is the last one. Long enough to fill a couple of vector registers,
/// short enough that a scan which settles immediately reads little more than
/// it has to.
const RUN_BLOCK: usize = 64;

/// `any`/`all` along one axis.
///
/// Two layouts, because the reduced axis being last or not decides which way
/// the reads run.
///
/// **`inner == 1`** -- the reduced axis is the last one, so each output owns a
/// contiguous run of `dim_size` elements. Scanning that run directly is already
/// sequential in memory, and it can stop at the first element that settles the
/// answer.
///
/// **`inner > 1`** -- each output's elements are `inner` apart, so scanning one
/// output at a time walks the input in strides and touches a fresh cache line
/// every step. Accumulating whole slabs instead (`out[r] op= input[k][r]` for
/// every `r` at once) reads in memory order, at the cost of visiting every
/// element rather than stopping early. The early exit comes back at slab
/// granularity: after each slab, if nothing is left at the identity there is
/// nothing more to learn.
///
/// The parallel split is over the output — bands of columns within a slab, so
/// that a reduction over dimension 0 (one outer position, all the work in one
/// slab) is split at all — and it is gated on the *input* length, because that
/// is where the work is. Gating on the output would leave a 1024x1024 reduction
/// to a single core on the strength of its 1024 outputs.
fn bool_fold_along_dim(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
    fold: BoolFold,
) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::dim_out_of_range(
            dim as isize,
            tensor.ndim(),
        ));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Bool, tensor.device());

    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    let identity = fold.identity();
    let numel = tensor.numel();

    // `inner == 0` is the only case with nothing to do: it means some axis
    // inside the reduced one is empty, so the output is empty too, and it is
    // also the one chunk length `chunks_mut` refuses to take. A `dim_size` of
    // zero is *not* this case -- the output is a full set of identities, which
    // both branches below produce by scanning nothing.
    if inner != 0 {
        with_truthy_slice!(tensor, |input, truthy| {
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            if inner == 1 {
                // Contiguous runs, one per output, with a true early exit.
                let run = |first: usize, chunk: &mut [bool]| {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let base = (first + i) * dim_size;
                        let mut val = identity;
                        // Blocked rather than a bare `for .. { if .. break }`:
                        // the branch out of the middle of a loop is what stops
                        // it vectorizing, and these runs are usually scanned
                        // most of the way through. Within a block the combine
                        // is branchless, and the block boundary is where the
                        // scan is allowed to give up -- so an `any` that
                        // settles on the first element still reads only one
                        // block, not the whole run.
                        for block in input[base..base + dim_size].chunks(RUN_BLOCK) {
                            let mut acc = identity;
                            for &v in block {
                                if identity {
                                    acc &= truthy(v);
                                } else {
                                    acc |= truthy(v);
                                }
                            }
                            if acc != identity {
                                val = !identity;
                                break;
                            }
                        }
                        *slot = val;
                    }
                };
                if numel < PAR_THRESHOLD {
                    run(0, output);
                } else {
                    par_out_chunks(output, BOOL_FOLD_BAND, &run);
                }
            } else {
                // Slab accumulation. `band` divides `inner` so every task stays
                // inside one slab and can derive its outer position from the
                // chunk index alone.
                let band = column_band(inner);
                let bands_per_slab = inner / band;
                let run = |first: usize, chunk: &mut [bool]| {
                    let chunk_index = first / band;
                    let o = chunk_index / bands_per_slab;
                    let col0 = (chunk_index % bands_per_slab) * band;
                    let width = chunk.len();
                    chunk.fill(identity);
                    let mut base = o * outer_stride + col0;
                    for _ in 0..dim_size {
                        let slab = &input[base..base + width];
                        for (slot, &v) in chunk.iter_mut().zip(slab) {
                            // `identity` is the same for every element of the
                            // call, so this unswitches into a plain `&=` for
                            // `all` and `|=` for `any`.
                            if identity {
                                *slot &= truthy(v);
                            } else {
                                *slot |= truthy(v);
                            }
                        }
                        // Everything has settled; the remaining slabs cannot
                        // change an answer. `all` stops here on the usual
                        // rejection, `any` on the usual acceptance.
                        if chunk.iter().all(|&v| v != identity) {
                            break;
                        }
                        base += inner;
                    }
                };
                if numel < PAR_THRESHOLD {
                    output
                        .chunks_mut(band)
                        .enumerate()
                        .for_each(|(c, chunk)| run(c * band, chunk));
                } else {
                    par_out_chunks(output, band, &run);
                }
            }
        });
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

/// The widest band no longer than [`BOOL_FOLD_BAND`] that divides `inner`
/// exactly, so a band never straddles two slabs.
///
/// Falling back to `inner` itself costs nothing when the outer dimension is
/// large -- whole slabs still go to separate tasks -- and only leaves a single
/// task when `inner` is both large and awkwardly factored.
fn column_band(inner: usize) -> usize {
    if inner <= BOOL_FOLD_BAND {
        return inner;
    }
    for parts in 2..=(inner / BOOL_FOLD_BAND + 1) {
        if inner.is_multiple_of(parts) && inner / parts <= BOOL_FOLD_BAND {
            return inner / parts;
        }
    }
    inner
}

/// Logical `all` reduction, over one dimension or the whole tensor.
pub fn all(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => bool_fold_all(tensor, keepdim, BoolFold::All),
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            bool_fold_along_dim(tensor, d, keepdim, BoolFold::All)
        }
    }
}

/// Logical `any` reduction, over one dimension or the whole tensor.
pub fn any(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => bool_fold_all(tensor, keepdim, BoolFold::Any),
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            bool_fold_along_dim(tensor, d, keepdim, BoolFold::Any)
        }
    }
}

/// Reject an extremum reduction that has no elements to choose between.
///
/// Only the *reduced* axis matters. `max(dim=1)` on a `(0, 3)` tensor is fine --
/// every slice it reduces has three elements, there just are not any slices, so
/// the empty output is the honest answer. `max(dim=1)` on a
/// `(3, 0)` tensor is not: it would have to invent three values out of nothing.
///
/// Without this the kernels returned their fold identity, which for floats is
/// `-inf` (visibly wrong, at least) and for integers is `i64::MIN` -- a value a
/// real tensor can hold, making an empty reduction silently indistinguishable
/// from a legitimate result. `argmax` was worse still: index `0` into an axis
/// that has no element `0`, which then reads out of bounds in a later `gather`.
/// Normalizes `dim` and applies the check in one step, so each caller resolves
/// the dimension exactly once.
fn checked_reduction_dim(tensor: &Tensor, dim: Option<isize>, op: &str) -> Result<Option<usize>> {
    let norm = match dim {
        Some(d) => Some(normalize_dim(d, tensor.ndim())?),
        None => None,
    };
    let empty = match norm {
        None => tensor.numel() == 0,
        Some(d) => tensor.shape().dims().get(d) == Some(&0),
    };
    if empty {
        return Err(MinitensorError::invalid_argument(format!(
            "{op}() does not support empty tensors"
        )));
    }
    Ok(norm)
}

/// Maximum value along specified dimension
pub fn max(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    let dim = checked_reduction_dim(tensor, dim, "max")?;
    let (output, norm_dim) = match dim {
        None => {
            // Find global maximum
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => max_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => max_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => max_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => max_all_i64(tensor, &mut result_data)?,
                DataType::Bool => max_all_bool(tensor, &mut result_data)?,
            }

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => (max_along_dim(tensor, d, keepdim)?, Some(d)),
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, true, false)
}

/// Attach a [`GatherBackward`] gradient to a value tensor that was formed by
/// gathering the input along `dim` at `indices` (`sort`/`topk`). The forward is
/// `values = gather(input, dim, indices)`, so the backward scatters the gradient
/// straight back to the selected source positions.
pub(crate) fn attach_gather_like_grad(
    values: Tensor,
    input: &Tensor,
    dim: usize,
    indices: &Tensor,
) -> Result<Tensor> {
    if !input.requires_grad() || !input.dtype().is_float() {
        return Ok(values);
    }
    let index = indices
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("selection indices must be int64"))?
        .to_vec();
    let grad_fn = Arc::new(GatherBackward {
        input_id: input.id(),
        input_shape: input.shape().dims().to_vec(),
        dim,
        index,
    });
    with_grad_fn(values, grad_fn)
}

/// Attach a [`MinMaxBackward`] gradient to a `min`/`max`/`nanmax`/`nanmin` value
/// reduction (`nan_aware` selects the NaN-ignoring recompute in the backward).
fn attach_minmax_grad(
    output: Tensor,
    input: &Tensor,
    dim: Option<usize>,
    keepdim: bool,
    is_max: bool,
    nan_aware: bool,
) -> Result<Tensor> {
    if !input.requires_grad() || !input.dtype().is_float() {
        return Ok(output);
    }
    let grad_fn = Arc::new(MinMaxBackward {
        input_id: input.id(),
        input: input.detach(),
        dim,
        keepdim,
        is_max,
        nan_aware,
    });
    with_grad_fn(output, grad_fn)
}

/// Minimum value along specified dimension
pub fn min(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    let dim = checked_reduction_dim(tensor, dim, "min")?;
    let (output, norm_dim) = match dim {
        None => {
            // Find global minimum
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => min_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => min_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => min_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => min_all_i64(tensor, &mut result_data)?,
                DataType::Bool => min_all_bool(tensor, &mut result_data)?,
            }

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => (min_along_dim(tensor, d, keepdim)?, Some(d)),
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, false, false)
}

/// NaN-aware maximum value along specified dimension
pub fn nanmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // Delegate before normalizing: integers have no NaN, so `nanmax` is just
    // `max`, and `max` applies the same empty-input check itself.
    if !tensor.dtype().is_float() {
        return max(tensor, dim, keepdim);
    }
    let dim = checked_reduction_dim(tensor, dim, "nanmax")?;

    let (output, norm_dim) = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => nanmax_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => nanmax_all_f64(tensor, &mut result_data)?,
                _ => unreachable!("nanmax only supports floating point tensors"),
            }

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => {
            let (values, _) = nanmax_along_dim_with_indices(tensor, d, keepdim)?;
            (values, Some(d))
        }
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, true, true)
}

/// NaN-aware minimum value along specified dimension
pub fn nanmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // Delegate before normalizing: integers have no NaN, so `nanmin` is just
    // `min`, and `min` applies the same empty-input check itself.
    if !tensor.dtype().is_float() {
        return min(tensor, dim, keepdim);
    }
    let dim = checked_reduction_dim(tensor, dim, "nanmin")?;

    let (output, norm_dim) = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => nanmin_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => nanmin_all_f64(tensor, &mut result_data)?,
                _ => unreachable!("nanmin only supports floating point tensors"),
            }

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => {
            let (values, _) = nanmin_along_dim_with_indices(tensor, d, keepdim)?;
            (values, Some(d))
        }
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, false, true)
}

/// Maximum values and their indices along specified dimension
pub fn max_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let d = checked_reduction_dim(tensor, Some(dim), "max")?.expect("dim was Some");
    let (values, indices) = max_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, true, false)?;
    Ok((values, indices))
}

/// NaN-aware maximum values and their indices along specified dimension
pub fn nanmax_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if !tensor.dtype().is_float() {
        return max_with_indices(tensor, dim, keepdim);
    }

    let d = checked_reduction_dim(tensor, Some(dim), "nanmax")?.expect("dim was Some");
    let (values, indices) = nanmax_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, true, true)?;
    Ok((values, indices))
}

/// Minimum values and their indices along specified dimension
pub fn min_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let d = checked_reduction_dim(tensor, Some(dim), "min")?.expect("dim was Some");
    let (values, indices) = min_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, false, false)?;
    Ok((values, indices))
}

/// NaN-aware minimum values and their indices along specified dimension
pub fn nanmin_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if !tensor.dtype().is_float() {
        return min_with_indices(tensor, dim, keepdim);
    }

    let d = checked_reduction_dim(tensor, Some(dim), "nanmin")?.expect("dim was Some");
    let (values, indices) = nanmin_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, false, true)?;
    Ok((values, indices))
}

/// Argument of maximum value along specified dimension
pub fn argmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    let dim = checked_reduction_dim(tensor, dim, "argmax")?;
    match dim {
        None => {
            // Find global argmax
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

            match tensor.dtype() {
                DataType::Float32 => argmax_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => argmax_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => argmax_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => argmax_all_i64(tensor, &mut result_data)?,
                DataType::Bool => argmax_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                DataType::Int64,
                tensor.device(),
                false, // argmax doesn't require gradients
            ))
        }
        Some(d) => argmax_along_dim(tensor, d, keepdim),
    }
}

/// Argument of minimum value along specified dimension
pub fn argmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    let dim = checked_reduction_dim(tensor, dim, "argmin")?;
    match dim {
        None => {
            // Find global argmin
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

            match tensor.dtype() {
                DataType::Float32 => argmin_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => argmin_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => argmin_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => argmin_all_i64(tensor, &mut result_data)?,
                DataType::Bool => argmin_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                DataType::Int64,
                tensor.device(),
                false, // argmin doesn't require gradients
            ))
        }
        Some(d) => argmin_along_dim(tensor, d, keepdim),
    }
}

#[inline]
fn select_topk_entries<T>(
    entries: &mut [(usize, T)],
    k: usize,
    sorted: bool,
    compare: fn(&(usize, T), &(usize, T)) -> Ordering,
) {
    if k == 0 || entries.is_empty() {
        return;
    }

    if k < entries.len() {
        entries.select_nth_unstable_by(k - 1, compare);
        if sorted {
            entries[..k].sort_by(compare);
        }
    } else if sorted {
        entries.sort_by(compare);
    }
}

/// Select the top-`k` entries of every 1-D slice along a dimension,
/// parallelizing over the outer index.
///
/// Output is laid out `(outer, k, inner)`, so each outer position owns a
/// disjoint `k * inner` span of both destinations and `par_chunks_mut` can hand
/// them out without overlap — the same partitioning `sort_along_dim_par` uses.
/// The scratch buffer of `(index, value)` pairs is allocated once per chunk
/// rather than once per slice.
#[allow(clippy::too_many_arguments)]
fn topk_along_dim_par<T>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    k: usize,
    sorted: bool,
    compare: fn(&(usize, T), &(usize, T)) -> Ordering,
) where
    T: Copy + Send + Sync,
{
    let span = k * inner;
    par_out_chunks2(values, indices, span, &|start, vchunk, ichunk| {
        let o = start / span;
        let mut entries: Vec<(usize, T)> = Vec::with_capacity(dim_size);
        for r in 0..inner {
            entries.clear();
            let base = o * outer_stride + r;
            for d in 0..dim_size {
                entries.push((d, input[base + d * inner]));
            }

            select_topk_entries(&mut entries, k, sorted, compare);

            // Output shape is (outer, k, inner); write row-major so a
            // non-trailing reduction axis (inner > 1) lands correctly.
            for (j, &(index, value)) in entries.iter().take(k).enumerate() {
                let off = j * inner + r;
                vchunk[off] = value;
                ichunk[off] = index as i64;
            }
        }
    });
}

/// Return the top-``k`` values and their indices along ``dim``
pub fn topk(
    tensor: &Tensor,
    k: usize,
    dim: Option<isize>,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor, Tensor)> {
    let ndim = tensor.ndim();

    let axis = if ndim == 0 {
        match dim {
            Some(d) if d == 0 || d == -1 => 0,
            Some(d) => return Err(MinitensorError::dim_out_of_range(d, 1)),
            None => 0,
        }
    } else {
        let dim_value = dim.unwrap_or(-1);
        normalize_dim(dim_value, ndim)?
    };

    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[axis] };

    if k > dim_size {
        return Err(MinitensorError::invalid_argument(format!(
            "selected index k out of range for dimension {axis} with size {dim_size}"
        )));
    }

    let output_dims = if dims.is_empty() {
        vec![k]
    } else {
        let mut dims_vec = dims.to_vec();
        dims_vec[axis] = k;
        dims_vec
    };

    let values_shape = Shape::new(output_dims.clone());
    let indices_shape = Shape::new(output_dims);

    let num_out = values_shape.numel();
    let mut values_data = TensorData::zeros_on_device(num_out, tensor.dtype(), tensor.device());
    let mut indices_data = TensorData::zeros_on_device(num_out, DataType::Int64, tensor.device());

    if k == 0 || num_out == 0 {
        let values = Tensor::new(
            Arc::new(values_data),
            values_shape,
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        );
        let indices = Tensor::new(
            Arc::new(indices_data),
            indices_shape,
            DataType::Int64,
            tensor.device(),
            false,
        );
        return Ok((values, indices));
    }

    let inner = if dims.is_empty() || axis + 1 >= dims.len() {
        1
    } else {
        dims[axis + 1..].iter().product()
    };
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let compare = if largest { cmp_f32_desc } else { cmp_f32_asc };
            topk_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                k,
                sorted,
                compare,
            );
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let compare = if largest { cmp_f64_desc } else { cmp_f64_asc };
            topk_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                k,
                sorted,
                compare,
            );
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let values = values_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let compare = if largest { cmp_i32_desc } else { cmp_i32_asc };
            topk_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                k,
                sorted,
                compare,
            );
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let values = values_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let compare = if largest { cmp_i64_desc } else { cmp_i64_asc };
            topk_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                k,
                sorted,
                compare,
            );
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let values = values_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let compare = if largest { cmp_bool_desc } else { cmp_bool_asc };
            topk_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                k,
                sorted,
                compare,
            );
        }
    }

    let values = Tensor::new(
        Arc::new(values_data),
        values_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );
    let indices = Tensor::new(
        Arc::new(indices_data),
        indices_shape,
        DataType::Int64,
        tensor.device(),
        false,
    );

    // `values = gather(input, axis, indices)`; scatter the gradient back.
    let values = attach_gather_like_grad(values, tensor, axis, &indices)?;

    Ok((values, indices))
}

#[cfg(test)]
mod bool_fold_tests {
    use super::*;
    use crate::device::Device;

    fn f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        Tensor::new(
            Arc::new(TensorData::from_vec::<f32>(
                data,
                DataType::Float32,
                Device::cpu(),
            )),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn bool_tensor(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        Tensor::new(
            Arc::new(TensorData::from_vec::<bool>(
                data,
                DataType::Bool,
                Device::cpu(),
            )),
            shape,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    fn out(t: &Tensor) -> Vec<bool> {
        t.data().as_bool_slice().unwrap().to_vec()
    }

    /// Both folds along each axis of the same tensor, against the answer read
    /// off by hand. The two used to be separate implementations in separate
    /// files, so nothing forced them to agree about layout; now they are one
    /// walk and this says what that walk produces.
    #[test]
    fn any_and_all_agree_with_a_hand_reduction_on_every_axis() {
        // 2x3, with a row that is all-nonzero, a row that is all-zero in one
        // column and mixed elsewhere.
        let t = f32_tensor(vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0], vec![2, 3]);

        assert_eq!(
            out(&any(&t, Some(0), false).unwrap()),
            vec![true, true, true]
        );
        assert_eq!(
            out(&all(&t, Some(0), false).unwrap()),
            vec![false, false, true]
        );
        assert_eq!(out(&any(&t, Some(1), false).unwrap()), vec![true, true]);
        assert_eq!(out(&all(&t, Some(1), false).unwrap()), vec![true, false]);

        // negative axis resolves the same way
        assert_eq!(
            out(&all(&t, Some(-1), false).unwrap()),
            out(&all(&t, Some(1), false).unwrap())
        );

        // keepdim only changes the shape
        let kept = all(&t, Some(1), true).unwrap();
        assert_eq!(kept.shape().dims(), &[2, 1]);
        assert_eq!(out(&kept), vec![true, false]);

        // whole-tensor
        assert_eq!(out(&any(&t, None, false).unwrap()), vec![true]);
        assert_eq!(out(&all(&t, None, false).unwrap()), vec![false]);
        assert_eq!(all(&t, None, true).unwrap().shape().dims(), &[1, 1]);
    }

    /// An axis of length zero reduces to each fold's identity: `any` of nothing
    /// is false, `all` of nothing is vacuously true. This is the case a scan
    /// written as "stop at the first disagreement" gets right only because it
    /// starts from the identity.
    #[test]
    fn an_empty_axis_reduces_to_the_identity() {
        let t = f32_tensor(Vec::new(), vec![2, 0]);
        assert_eq!(out(&any(&t, Some(1), false).unwrap()), vec![false, false]);
        assert_eq!(out(&all(&t, Some(1), false).unwrap()), vec![true, true]);

        // Reducing the *other* axis leaves an empty output, not an identity.
        assert!(out(&any(&t, Some(0), false).unwrap()).is_empty());
        assert!(out(&all(&t, Some(0), false).unwrap()).is_empty());

        // ...and so does the whole-tensor form over no elements at all.
        assert_eq!(out(&any(&t, None, false).unwrap()), vec![false]);
        assert_eq!(out(&all(&t, None, false).unwrap()), vec![true]);
    }

    /// Truthiness is "nonzero" for the numeric dtypes and the value itself for
    /// `Bool`. Negative and NaN both count as truthy, being nonzero.
    #[test]
    fn truthiness_is_nonzero_including_negatives_and_nan() {
        let t = f32_tensor(vec![-1.0, f32::NAN, -0.0, 0.0], vec![4]);
        assert_eq!(out(&any(&t, None, false).unwrap()), vec![true]);
        assert_eq!(out(&all(&t, None, false).unwrap()), vec![false]);

        // -0.0 is zero, so a slice of only signed zeros is entirely falsy.
        let zeros = f32_tensor(vec![0.0, -0.0], vec![2]);
        assert_eq!(out(&any(&zeros, None, false).unwrap()), vec![false]);

        let b = bool_tensor(vec![false, true, false, false], vec![2, 2]);
        assert_eq!(out(&any(&b, Some(1), false).unwrap()), vec![true, false]);
        assert_eq!(out(&all(&b, Some(1), false).unwrap()), vec![false, false]);
    }

    /// The along-axis fold runs sequentially below the threshold and in
    /// parallel above it. The split is over whole output slabs, so it cannot
    /// change an answer -- which this checks at a size that crosses it, with
    /// the one disagreeing element placed at each end of the scanned axis.
    #[test]
    fn the_parallel_split_does_not_change_the_answer() {
        let rows = PAR_THRESHOLD + 3;
        let cols = 4;
        for offender in [0usize, cols - 1] {
            let mut data = vec![1.0f32; rows * cols];
            // one zero per row, at a fixed position along the reduced axis
            for r in 0..rows {
                data[r * cols + offender] = 0.0;
            }
            let t = f32_tensor(data, vec![rows, cols]);

            let alls = out(&all(&t, Some(1), false).unwrap());
            let anys = out(&any(&t, Some(1), false).unwrap());
            assert_eq!(alls.len(), rows);
            assert!(alls.iter().all(|&v| !v), "offender at {offender}");
            assert!(anys.iter().all(|&v| v), "offender at {offender}");
        }
    }
}
