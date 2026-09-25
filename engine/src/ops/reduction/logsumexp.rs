// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::CumprodBackward;
use crate::autograd::CumsumBackward;
use crate::autograd::NanMeanBackward;
use crate::autograd::ProdBackward;
use crate::ops::activation::ShiftedExp;
use crate::ops::map::{par_map_indexed, par_out_chunks};
use crate::ops::util::check_dim;
use crate::ops::util::{RUN_SUM_CHUNK, accumulating_dtype, pairwise_fold, pairwise_fold_vectors};
use crate::ops::{activation, arithmetic, shape_ops};
use crate::{
    autograd::with_grad_fn,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Numerically stable log-sum-exp reduction along specified dimensions
pub fn logsumexp(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // An integer argument widens rather than being refused, the rule `mean`
    // already follows: a variance, a norm and a log-sum-exp of integers are
    // all real numbers. See `ops::util::widen_integer_input`.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return logsumexp(&widened, dim, keepdim);
    }
    match tensor.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logsumexp only supported for floating point tensors",
            ));
        }
    }

    let dims = match normalize_reduction_dims(dim, tensor.ndim())? {
        Some(dims) => dims,
        None => (0..tensor.ndim()).collect(),
    };

    if dims.is_empty() {
        return Ok(tensor.clone());
    }

    // Fused fast path: single-axis log-sum-exp for tensors that don't require
    // gradients (mirrors the var/std fast path). Avoids materializing the
    // full-size shift/exp intermediates; the gradient path keeps the autograd
    // composition below.
    if !tensor.requires_grad() && dims.len() == 1 && tensor.shape().dims()[dims[0]] >= 1 {
        return logsumexp_fused_single_axis(tensor, dims[0], keepdim);
    }

    let mut max_tensor = tensor.clone();
    for &d in &dims {
        max_tensor = max_along_dim(&max_tensor, d, true)?;
    }
    let max_tensor = max_tensor.detach();

    let shifted = arithmetic::sub(tensor, &max_tensor)?;
    let exp_shifted = activation::exp(&shifted)?;
    let dims_isize: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    let sum_exp = sum(&exp_shifted, Some(dims_isize), true)?;
    let log_sum = activation::log(&sum_exp)?;
    let mut result = arithmetic::add(&max_tensor, &log_sum)?;

    // A non-finite row max poisons the shifted sum with `inf - inf = NaN`.
    // The correct limit for those rows is the max itself: +inf rows reduce to
    // +inf, all--inf rows to -inf, and NaN propagates as NaN.
    match tensor.dtype() {
        DataType::Float32 => {
            let max_slice = max_tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from max tensor")
            })?;
            if max_slice.iter().any(|v| !v.is_finite()) {
                let non_finite: Vec<(usize, f32)> = max_slice
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_finite())
                    .map(|(i, &v)| (i, v))
                    .collect();
                let out = result.data_mut().as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f32 slice from result")
                })?;
                for (i, m) in non_finite {
                    out[i] = m;
                }
            }
        }
        DataType::Float64 => {
            let max_slice = max_tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from max tensor")
            })?;
            if max_slice.iter().any(|v| !v.is_finite()) {
                let non_finite: Vec<(usize, f64)> = max_slice
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_finite())
                    .map(|(i, &v)| (i, v))
                    .collect();
                let out = result.data_mut().as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f64 slice from result")
                })?;
                for (i, m) in non_finite {
                    out[i] = m;
                }
            }
        }
        _ => unreachable!("logsumexp dtype checked above"),
    }

    if !keepdim {
        let mut new_dims = Vec::with_capacity(result.ndim() - dims.len());
        for (idx, &size) in result.shape().dims().iter().enumerate() {
            if dims.binary_search(&idx).is_err() {
                new_dims.push(size);
            }
        }

        let target_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        result = shape_ops::reshape(&result, target_shape)?;
    }

    Ok(result)
}

/// Fused single-axis log-sum-exp for tensors that do not require gradients.
///
/// The input is `outer` slabs of `(len, inner)` rows, and each column of each
/// slab reduces to `max + ln(sum(exp(x - max)))`, with the maximum propagating
/// NaN and a non-finite maximum being the answer itself: `+inf` columns reduce
/// to `+inf`, all-`-inf` columns to `-inf`. See [`lse_columns`] for how one
/// slab is done.
///
/// With at least `LSE_MIN_SPREAD` slabs, each is done whole on one thread;
/// with fewer, each is spread across the pool in bands of rows. Both compute
/// the same bands folded the same way, so which one runs changes the time and
/// not the answer.
fn logsumexp_fused_single_axis(tensor: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let len = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let slab = len * inner;

    let mut result_data =
        TensorData::zeros_on_device(outer * inner, tensor.dtype(), tensor.device());

    macro_rules! fill {
        ($accessor:ident, $accessor_mut:ident) => {{
            let input = tensor
                .data()
                .$accessor()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let out = result_data
                .$accessor_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            if inner != 0 {
                if outer < LSE_MIN_SPREAD {
                    for (source, target) in
                        input.chunks_exact(slab).zip(out.chunks_exact_mut(inner))
                    {
                        target.copy_from_slice(&lse_columns(source, inner, true));
                    }
                } else {
                    par_out_chunks(out, inner, &|start, target| {
                        let first = start / inner * slab;
                        target.copy_from_slice(&lse_columns(
                            &input[first..first + slab],
                            inner,
                            false,
                        ));
                    });
                }
            }
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut),
        DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut),
        _ => unreachable!("logsumexp dtype checked above"),
    }

    let out_shape = if keepdim {
        let mut d = dims.to_vec();
        d[axis] = 1;
        Shape::new(d)
    } else {
        let d: Vec<usize> = dims
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
            .collect();
        if d.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(d)
        }
    };

    Ok(Tensor::new(
        Arc::new(result_data),
        out_shape,
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Fewer slabs than this are each spread across the pool rather than handed
/// one to a thread.
const LSE_MIN_SPREAD: usize = 4;
/// Bands a slab's rows are cut into, and the fewest elements a band may hold,
/// so what a band costs to set up is spread over enough work to vanish. Both
/// are fixed: band boundaries follow from the shape alone.
const LSE_TARGET_BANDS: usize = 64;
const LSE_MIN_BAND: usize = 1 << 16;
/// The rounding chain each accumulator lane is held to.
const LSE_BLOCK_ROWS: usize = 128;
/// The fewest elements one call of the exponential kernel is given: narrow
/// rows are batched until a group is at least this wide.
const LSE_MIN_GROUP: usize = 512;

/// `max(a, b)` that lets a NaN through from either side, which is what makes
/// a column holding a NaN reduce to NaN.
#[inline(always)]
fn nan_max<T: num_traits::Float>(a: T, b: T) -> T {
    if a.is_nan() {
        a
    } else if b.is_nan() || b > a {
        b
    } else {
        a
    }
}

/// Log-sum-exp of one contiguous run: every last-axis reduction, one output
/// at a time.
///
/// [`lse_columns`] would do this as a one-column slab, but it sizes its
/// buffers to a group of rows and folds them back to columns, which a row of
/// a few thousand elements cannot pay for: 16384 rows of 256 took 67ms on one
/// thread. Here the lanes live on the stack and the exponentials land in a
/// stack buffer straight from the kernel, so a run of up to `RUN_SUM_CHUNK`
/// allocates nothing. Longer runs split into chunks of that length -- fixed,
/// so the split is the same on any pool -- whose partials fold pairwise.
fn lse_run<T: ShiftedExp + Send + Sync>(run: &[T], spread: bool) -> T {
    const LANES: usize = 16;
    const PIECE: usize = 1024;

    fn chunk_max<T: num_traits::Float>(chunk: &[T]) -> T {
        let mut top = [T::neg_infinity(); LANES];
        let mut nan = [false; LANES];
        let (blocks, rest) = chunk.as_chunks::<LANES>();
        for block in blocks {
            for lane in 0..LANES {
                let v = block[lane];
                nan[lane] |= v.is_nan();
                if v > top[lane] {
                    top[lane] = v;
                }
            }
        }
        let mut best = T::neg_infinity();
        for (lane, &t) in top.iter().enumerate() {
            if nan[lane] {
                return T::nan();
            }
            best = nan_max(best, t);
        }
        rest.iter().fold(best, |m, &v| nan_max(m, v))
    }

    fn chunk_sum<T: ShiftedExp>(chunk: &[T], top: T) -> T {
        let mut buffer = [T::zero(); PIECE];
        let mut sums = [T::zero(); LANES];
        for piece in chunk.chunks(PIECE) {
            let terms = &mut buffer[..piece.len()];
            T::exp_shifted_into(piece, top, terms);
            let (blocks, rest) = terms.as_chunks::<LANES>();
            for block in blocks {
                for lane in 0..LANES {
                    sums[lane] = sums[lane] + block[lane];
                }
            }
            for (lane, &v) in rest.iter().enumerate() {
                sums[lane] = sums[lane] + v;
            }
        }
        let mut width = LANES;
        while width > 1 {
            width /= 2;
            for lane in 0..width {
                sums[lane] = sums[lane] + sums[lane + width];
            }
        }
        sums[0]
    }

    let chunks = run.len().div_ceil(RUN_SUM_CHUNK).max(1);
    let chunk =
        |index: usize| &run[index * RUN_SUM_CHUNK..((index + 1) * RUN_SUM_CHUNK).min(run.len())];
    let over_chunks = |work: &(dyn Fn(usize) -> T + Sync)| -> Vec<T> {
        if spread && chunks > 1 {
            par_map_indexed(chunks, work)
        } else {
            (0..chunks).map(work).collect()
        }
    };
    let top = over_chunks(&|index| chunk_max(chunk(index)))
        .into_iter()
        .fold(T::neg_infinity(), nan_max);
    if !top.is_finite() {
        // `+inf` runs reduce to `+inf`, all-`-inf` runs to `-inf`, and a NaN
        // propagates.
        return top;
    }
    let total = pairwise_fold(
        over_chunks(&|index| chunk_sum(chunk(index), top)),
        T::zero(),
        |a, b| a + b,
    );
    top + total.ln()
}

/// Fold lanes laid out `cols` to a row back down to one value per column.
#[inline(always)]
fn lanes_to_columns<T: Copy>(lanes: &[T], cols: usize, merge: impl Fn(T, T) -> T) -> Vec<T> {
    let mut columns = lanes[..cols].to_vec();
    for tile in lanes[cols..].chunks_exact(cols) {
        for (c, &v) in columns.iter_mut().zip(tile) {
            *c = merge(*c, v);
        }
    }
    columns
}

/// Log-sum-exp down each column of a row-major `(rows, cols)` slab; a single
/// column is a contiguous run and goes to [`lse_run`].
///
/// Two passes over bands of rows: the column maxima, then the column sums of
/// `exp(x - max)`. The exponentials go through the vectorized kernel a group
/// of rows at a time -- a narrow row is batched with the ones after it until
/// the group is `LSE_MIN_GROUP` wide, against a shift vector tiled to match --
/// and never land in memory beyond that group. Each accumulator lane runs
/// `LSE_BLOCK_ROWS` additions; blocks and then bands fold pairwise.
///
/// Together with [`lse_run`] this replaced a loop calling scalar `exp` per
/// element that, for a last-axis reduction, walked the whole axis on one
/// thread. float32 on four cores, before -> after, against NumPy's
/// `log(exp(x - max).sum()) + max`:
///
/// ```text
///   (4194304,)        13.6ms -> 1.42     NumPy 7.6
///   (1024, 4096) d1    4.30  -> 1.27           6.4
///   (4096, 1024) d0    4.38  -> 2.50           6.2
///   (16, 100000, 2) d1 3.64  -> 1.70          65.6
/// ```
///
/// `spread` runs the bands on the pool; it does not change how they are cut
/// or folded, so it cannot change the result.
fn lse_columns<T: ShiftedExp + Send + Sync>(slab: &[T], cols: usize, spread: bool) -> Vec<T> {
    if cols == 1 {
        return vec![lse_run(slab, spread)];
    }
    let rows = slab.len() / cols;
    let band_rows = rows
        .div_ceil(LSE_TARGET_BANDS)
        .max(LSE_MIN_BAND.div_ceil(cols));
    let bands = rows.div_ceil(band_rows).max(1);
    let band =
        |index: usize| &slab[index * band_rows * cols..((index + 1) * band_rows).min(rows) * cols];
    let over_bands = |work: &(dyn Fn(usize) -> Vec<T> + Sync)| -> Vec<Vec<T>> {
        if spread && bands > 1 {
            par_map_indexed(bands, work)
        } else {
            (0..bands).map(work).collect()
        }
    };
    // Both passes take rows a group at a time, `per_group` of them, into
    // `group` lanes; lane `j` belongs to column `j % cols`.
    let per_group = LSE_MIN_GROUP.div_ceil(cols);
    let group = per_group * cols;

    // A NaN is recorded beside the lane rather than branched on, which is what
    // lets the comparison loop vectorize.
    let maxima = pairwise_fold_vectors(
        over_bands(&|index| {
            let mut top = vec![T::neg_infinity(); group];
            let mut nan = vec![false; group];
            for part in band(index).chunks(group) {
                for ((t, seen), &v) in top.iter_mut().zip(nan.iter_mut()).zip(part) {
                    *seen |= v.is_nan();
                    if v > *t {
                        *t = v;
                    }
                }
            }
            for (t, &seen) in top.iter_mut().zip(&nan) {
                if seen {
                    *t = T::nan();
                }
            }
            lanes_to_columns(&top, cols, nan_max)
        }),
        nan_max,
    );

    // The maxima tiled across a group's rows.
    let shift: Vec<T> = maxima.iter().copied().cycle().take(group).collect();
    let sums = pairwise_fold_vectors(
        over_bands(&|index| {
            let rows_here = band(index);
            let mut scratch = vec![T::zero(); group];
            let mut terms = vec![T::zero(); group];
            let blocks: Vec<Vec<T>> = rows_here
                .chunks(group * LSE_BLOCK_ROWS)
                .map(|block| {
                    let mut lanes = vec![T::zero(); group];
                    for part in block.chunks(group) {
                        let n = part.len();
                        T::exp_diff_into(part, &shift[..n], &mut scratch[..n], &mut terms[..n]);
                        for (lane, &term) in lanes.iter_mut().zip(&terms[..n]) {
                            *lane = *lane + term;
                        }
                    }
                    lanes_to_columns(&lanes, cols, |a, b| a + b)
                })
                .collect();
            if blocks.is_empty() {
                vec![T::zero(); cols]
            } else {
                pairwise_fold_vectors(blocks, |a, b| a + b)
            }
        }),
        |a, b| a + b,
    );

    maxima
        .iter()
        .zip(&sums)
        .map(|(&m, &s)| if m.is_finite() { m + s.ln() } else { m })
        .collect()
}

/// Product reduction along specified dimensions
pub fn prod(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // As `sum` does: `bool` has no multiplication to accumulate in, so the
    // running product goes to `Int64` and the integer path takes it from there.
    if tensor.dtype() == DataType::Bool {
        return prod(&tensor.astype(DataType::Int64)?, dim, keepdim);
    }

    // Normalise negative dimensions and deduplicate
    let dim = normalize_reduction_dims(dim, tensor.ndim())?;
    let dims_clone = dim.clone();

    let result = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let out_dtype = accumulating_dtype(tensor.dtype());
            let mut result_data = TensorData::zeros_on_device(1, out_dtype, tensor.device());
            match tensor.dtype() {
                DataType::Float32 => prod_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => prod_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => prod_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => prod_all_i64(tensor, &mut result_data)?,
                DataType::Bool => unreachable!("bool was promoted above"),
            }

            Tensor::new(
                Arc::new(result_data),
                result_shape,
                out_dtype,
                tensor.device(),
                tensor.requires_grad(),
            )
        }
        Some(dims) => {
            if dims.is_empty() {
                tensor.clone()
            } else {
                let mut result = tensor.clone();
                if keepdim {
                    for &d in &dims {
                        result = prod_along_dim(&result, d, true)?;
                    }
                } else {
                    for &d in dims.iter().rev() {
                        result = prod_along_dim(&result, d, false)?;
                    }
                }
                result
            }
        }
    };

    if result.requires_grad() {
        let grad_fn = Arc::new(ProdBackward {
            input: tensor.detach(),
            result: result.clone(),
            input_id: tensor.id(),
            dims: dims_clone,
            keepdim,
        });
        with_grad_fn(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Cumulative sum along a specified dimension
pub fn cumsum(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    // A running count of a mask, for the same reason `sum` accepts one: the
    // accumulator has to be wider than `bool`, and `int64` is where it goes.
    if tensor.dtype() == DataType::Bool {
        return cumsum(&tensor.astype(DataType::Int64)?, dim);
    }

    let dim = normalize_dim(dim, tensor.ndim())?;

    let out_dtype = accumulating_dtype(tensor.dtype());
    let mut result_data =
        TensorData::uninitialized_on_device(tensor.numel(), out_dtype, tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cumsum_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => cumsum_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => cumsum_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => cumsum_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Cumsum not supported for boolean tensors",
            ));
        }
    }

    let result = Tensor::new(
        Arc::new(result_data),
        tensor.shape().clone(),
        out_dtype,
        tensor.device(),
        tensor.requires_grad(),
    );

    if result.requires_grad() {
        let grad_fn = Arc::new(CumsumBackward {
            input_id: tensor.id(),
            dim,
        });
        with_grad_fn(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Backward helper for cumulative sum
pub fn cumsum_backward(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    check_dim(dim, tensor.ndim())?;

    let mut result_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cumsum_backward_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => cumsum_backward_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => cumsum_backward_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => cumsum_backward_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Cumsum not supported for boolean tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Cumulative product along a specified dimension
pub fn cumprod(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    // As in `cumsum`: `bool` has no multiplication to accumulate in, so the
    // running product goes to `int64` and the integer path takes it from there.
    if tensor.dtype() == DataType::Bool {
        return cumprod(&tensor.astype(DataType::Int64)?, dim);
    }

    let dim = normalize_dim(dim, tensor.ndim())?;

    let out_dtype = accumulating_dtype(tensor.dtype());
    let mut result_data =
        TensorData::uninitialized_on_device(tensor.numel(), out_dtype, tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cumprod_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => cumprod_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => cumprod_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => cumprod_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => unreachable!("bool was promoted above"),
    }

    let requires_grad =
        tensor.requires_grad() && matches!(tensor.dtype(), DataType::Float32 | DataType::Float64);

    let result = Tensor::new(
        Arc::new(result_data),
        tensor.shape().clone(),
        out_dtype,
        tensor.device(),
        requires_grad,
    );

    if result.requires_grad() {
        let grad_fn = Arc::new(CumprodBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
            output: result.clone(),
            dim,
        });
        with_grad_fn(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Backward helper for cumulative product
pub fn cumprod_backward(
    input: &Tensor,
    output: &Tensor,
    grad: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    if dim >= input.ndim() {
        return Err(MinitensorError::dim_out_of_range(
            dim as isize,
            input.ndim(),
        ));
    }

    let mut result_data = TensorData::zeros_on_device(input.numel(), input.dtype(), input.device());

    match input.dtype() {
        DataType::Float32 => cumprod_backward_f32(input, output, grad, &mut result_data, dim)?,
        DataType::Float64 => cumprod_backward_f64(input, output, grad, &mut result_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Cumprod backward only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    ))
}

/// Mean reduction along specified dimensions
pub fn mean(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // Normalise negative dimensions and deduplicate
    let normalized = normalize_reduction_dims(dim, tensor.ndim())?;

    let sum_result = sum(
        tensor,
        normalized
            .clone()
            .map(|d| d.iter().map(|&x| x as isize).collect()),
        keepdim,
    )?;

    // Compute the number of elements being averaged
    let num_elements = match &normalized {
        None => tensor.numel() as f64,
        Some(dims) => {
            if dims.is_empty() {
                return Ok(tensor.clone());
            }

            let mut count = 1.0;
            for &d in dims {
                count *= tensor.shape().dims()[d] as f64;
            }
            count
        }
    };

    // Prepare sum tensor and divisor for division
    let (sum_tensor, divisor) = match tensor.dtype() {
        DataType::Float32 => (
            sum_result,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements as f32],
                    DataType::Float32,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float32,
                tensor.device(),
                false,
            ),
        ),
        DataType::Float64 => (
            sum_result,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements],
                    DataType::Float64,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float64,
                tensor.device(),
                false,
            ),
        ),
        DataType::Int32 => (
            sum_result.astype(DataType::Float32)?,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements as f32],
                    DataType::Float32,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float32,
                tensor.device(),
                false,
            ),
        ),
        DataType::Int64 => (
            sum_result.astype(DataType::Float64)?,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements],
                    DataType::Float64,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float64,
                tensor.device(),
                false,
            ),
        ),
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Mean not supported for boolean tensors",
            ));
        }
    };

    crate::ops::arithmetic::div(&sum_tensor, &divisor)
}

/// NaN-aware mean reduction along specified dimensions
pub fn nanmean(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return mean(tensor, dim, keepdim);
    }

    let dim = normalize_reduction_dims(dim, tensor.ndim())?;
    let dims_clone = dim.clone();
    let needs_mask =
        tensor.requires_grad() || dim.as_ref().map(|dims| !dims.is_empty()).unwrap_or(false);
    let mask = if needs_mask {
        Some(non_nan_mask(tensor)?)
    } else {
        None
    };

    if let Some(dims) = &dim
        && dims.is_empty()
    {
        return Ok(tensor.clone());
    }

    let (sum, count) = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };
            let mut sum_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());
            let mut count_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => nanmean_all_f32(tensor, &mut sum_data, &mut count_data)?,
                DataType::Float64 => nanmean_all_f64(tensor, &mut sum_data, &mut count_data)?,
                _ => unreachable!("nanmean only supports floating point tensors"),
            }

            (
                Tensor::new(
                    Arc::new(sum_data),
                    result_shape.clone(),
                    tensor.dtype(),
                    tensor.device(),
                    false,
                ),
                Tensor::new(
                    Arc::new(count_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    false,
                ),
            )
        }
        Some(dims) => {
            let mask = mask.as_ref().ok_or_else(|| {
                MinitensorError::internal_error("nanmean expected mask for count computation")
            })?;
            let mut sum = tensor.clone();
            let mut count = mask.astype(tensor.dtype())?;

            if keepdim {
                for &d in &dims {
                    sum = nansum_along_dim(&sum, d, true)?;
                    count = sum_along_dim(&count, d, true)?;
                }
            } else {
                for &d in dims.iter().rev() {
                    sum = nansum_along_dim(&sum, d, false)?;
                    count = sum_along_dim(&count, d, false)?;
                }
            }
            (sum, count)
        }
    };

    let result = nanmean_from_sum_count(&sum, &count, tensor.requires_grad())?;

    if result.requires_grad() {
        let mask = mask.ok_or_else(|| {
            MinitensorError::internal_error("nanmean expected mask for gradient computation")
        })?;
        let grad_fn = Arc::new(NanMeanBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dims: dims_clone,
            keepdim,
            mask,
            count,
        });
        with_grad_fn(result, grad_fn)
    } else {
        Ok(result)
    }
}
