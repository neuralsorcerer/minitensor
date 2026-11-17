// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::CumprodBackward;
use crate::autograd::CumsumBackward;
use crate::autograd::LogSumExpBackward;
use crate::autograd::NanMeanBackward;
use crate::autograd::ProdBackward;
use crate::ops::activation::{EXP_GROUP, ShiftedExp, exp_columns_into, lanes_to_columns};
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
        return crate::ops::util::narrowed(tensor, logsumexp(&widened, dim, keepdim));
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

    // The value is the fused one whatever the axes and whether or not a
    // gradient is wanted, and the gradient its own: composed from `sub`,
    // `exp`, `sum` and `log1p` in the input's dtype, the value rounded every
    // term to float32 for a float32 input, took its exponents of rounded
    // differences, and kept nothing of a sum that cancelled -- a float64
    // `logsumexp` of normalised log-probabilities, with a gradient wanted,
    // was 1e16 ulps out. Only an empty reduction, `log 0`, is left to the
    // composition below.
    let sizes = tensor.shape().dims();
    if dims.iter().all(|&d| sizes[d] >= 1) {
        let value = fused_logsumexp(tensor, &dims, keepdim)?;
        if !(tensor.requires_grad() && crate::autograd::is_grad_enabled()) {
            return Ok(value);
        }
        let grad_fn = Arc::new(LogSumExpBackward {
            input_id: tensor.id(),
            input: tensor.detach(),
            dims,
            keepdim,
        });
        return with_grad_fn(value.requires_grad_(true), grad_fn);
    }

    let mut max_tensor = tensor.clone();
    for &d in &dims {
        max_tensor = max_along_dim(&max_tensor, d, true)?;
    }
    let max_tensor = max_tensor.detach();

    // A row that is `-inf` throughout has a maximum of `-inf`, and shifting by
    // it is `-inf - -inf = NaN` -- which the value is patched past below, but
    // which reached the gradient: `logsumexp` of a fully masked row
    // differentiated to NaN, where `softmax`, its derivative, answers zero for
    // that row. Such rows are computed on zeros instead, through a select, so
    // the entries pass no gradient back; their value is restored below.
    let (input, shift) = if let Some(empty) = negative_infinity_mask(&max_tensor) {
        (
            crate::ops::selection::masked_fill_scalar(tensor, &empty, 0.0)?,
            crate::ops::selection::masked_fill_scalar(&max_tensor, &empty, 0.0)?,
        )
    } else {
        (tensor.clone(), max_tensor.clone())
    };

    let shifted = arithmetic::sub(&input, &shift)?;
    let exp_shifted = activation::exp(&shifted)?;
    let dims_isize: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    // The terms at the maximum, each `exp(0) = 1`, are summed apart from the
    // rest and come back through `log1p`, as the fused path takes them; see
    // `ExpTotal`. `units - 1` is exact, and keeps the maximum's own term in
    // the graph, so its gradient is the softmax as before.
    let at_max = crate::ops::comparison::eq(&input, &shift)?;
    let zero = crate::ops::util::create_scalar_tensor(0.0, tensor.dtype(), tensor.device())?;
    let one = crate::ops::util::create_scalar_tensor(1.0, tensor.dtype(), tensor.device())?;
    let units = sum(
        &crate::ops::selection::where_op(&at_max, &exp_shifted, &zero)?,
        Some(dims_isize.clone()),
        true,
    )?;
    let rest = sum(
        &crate::ops::selection::where_op(&at_max, &zero, &exp_shifted)?,
        Some(dims_isize),
        true,
    )?;
    let log_sum = activation::log1p(&arithmetic::add(&arithmetic::sub(&units, &one)?, &rest)?)?;
    let mut result = arithmetic::add(&shift, &log_sum)?;

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

/// A bool tensor marking the `-inf` elements of a float tensor, or `None`
/// when there are none -- the common case, which then costs one scan.
fn negative_infinity_mask(tensor: &Tensor) -> Option<Tensor> {
    let flags: Vec<bool> = match tensor.dtype() {
        DataType::Float32 => {
            let values = tensor.data().as_f32_slice()?;
            if !values.contains(&f32::NEG_INFINITY) {
                return None;
            }
            values.iter().map(|&v| v == f32::NEG_INFINITY).collect()
        }
        DataType::Float64 => {
            let values = tensor.data().as_f64_slice()?;
            if !values.contains(&f64::NEG_INFINITY) {
                return None;
            }
            values.iter().map(|&v| v == f64::NEG_INFINITY).collect()
        }
        _ => return None,
    };
    Some(Tensor::new(
        std::sync::Arc::new(TensorData::from_vec(flags, DataType::Bool, tensor.device())),
        tensor.shape().clone(),
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

/// Fused single-axis log-sum-exp, the value of every `logsumexp` but an empty
/// one.
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

/// [`logsumexp_fused_single_axis`] over any set of axes: several are moved
/// behind the rest and flattened into one, which holds each slice's terms in
/// one contiguous run.
fn fused_logsumexp(tensor: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor> {
    if let [axis] = dims {
        return logsumexp_fused_single_axis(tensor, *axis, keepdim);
    }
    let (flat, _) = moved_behind(tensor, dims)?;
    let value = logsumexp_fused_single_axis(&flat, flat.ndim() - 1, false)?;
    if !keepdim {
        return Ok(value);
    }
    let kept: Vec<usize> = tensor
        .shape()
        .dims()
        .iter()
        .enumerate()
        .map(|(d, &size)| if dims.contains(&d) { 1 } else { size })
        .collect();
    shape_ops::reshape(&value, Shape::new(kept))
}

/// `tensor` with the axes in `dims` moved behind the rest, in order, and
/// flattened into one, and the permutation that moved them.
fn moved_behind(tensor: &Tensor, dims: &[usize]) -> Result<(Tensor, Vec<usize>)> {
    let sizes = tensor.shape().dims();
    let perm: Vec<usize> = (0..tensor.ndim())
        .filter(|d| !dims.contains(d))
        .chain(dims.iter().copied())
        .collect();
    let moved =
        shape_ops::permute(tensor, perm.iter().map(|&d| d as isize).collect())?.contiguous()?;
    let mut flat: Vec<usize> = perm[..perm.len() - dims.len()]
        .iter()
        .map(|&d| sizes[d])
        .collect();
    flat.push(dims.iter().map(|&d| sizes[d]).product());
    Ok((shape_ops::reshape(&moved, Shape::new(flat))?, perm))
}

/// The softmax of `input` over the axes in `dims` together, the weights of
/// `logsumexp`'s gradient: `softmax` itself for one axis, and for several,
/// over them moved behind the rest and flattened, and put back.
pub(crate) fn softmax_over(input: &Tensor, dims: &[usize]) -> Result<Tensor> {
    if let [axis] = dims {
        return activation::softmax(input, Some(*axis));
    }
    let sizes = input.shape().dims().to_vec();
    let (flat, perm) = moved_behind(input, dims)?;
    let weights = activation::softmax(&flat, Some(flat.ndim() - 1))?;
    let moved: Vec<usize> = perm.iter().map(|&d| sizes[d]).collect();
    let weights = shape_ops::reshape(&weights, Shape::new(moved))?;
    let mut inverse = vec![0isize; perm.len()];
    for (at, &d) in perm.iter().enumerate() {
        inverse[d] = at as isize;
    }
    shape_ops::permute(&weights, inverse)?.contiguous()
}

/// Fewer slabs than this are each spread across the pool rather than handed
/// one to a thread.
const LSE_MIN_SPREAD: usize = 4;
/// Bands a slab's rows are cut into, and the fewest elements a band may hold,
/// so what a band costs to set up is spread over enough work to vanish. Both
/// are fixed: band boundaries follow from the shape alone.
const LSE_TARGET_BANDS: usize = 64;
const LSE_MIN_BAND: usize = 1 << 16;

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

    // The sum of `exp(x - top)` over the chunk, less the term at `skip`, in
    // float64; see `ShiftedExp::wide_sum`.
    fn chunk_sum<T: ShiftedExp>(chunk: &[T], top: T, skip: Option<usize>) -> f64 {
        // Built with AVX2 where the CPU has it: the float64 lanes are half the
        // float32 ones' width on the baseline.
        //
        // A short run takes a short buffer: zeroing the full piece for a row
        // of 64 cost more than the row's arithmetic.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        fn avx2<T: ShiftedExp, const P: usize>(chunk: &[T], top: T, skip: Option<usize>) -> f64 {
            chunk_body::<T, P>(chunk, top, skip)
        }
        let short = chunk.len() <= SHORT_PIECE;
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe {
                if short {
                    avx2::<T, SHORT_PIECE>(chunk, top, skip)
                } else {
                    avx2::<T, PIECE>(chunk, top, skip)
                }
            };
        }
        if short {
            chunk_body::<T, SHORT_PIECE>(chunk, top, skip)
        } else {
            chunk_body::<T, PIECE>(chunk, top, skip)
        }
    }

    const SHORT_PIECE: usize = 128;

    #[inline(always)]
    fn chunk_body<T: ShiftedExp, const P: usize>(chunk: &[T], top: T, skip: Option<usize>) -> f64 {
        let mut buffer = [0.0f64; P];
        let mut sum = crate::ops::activation::CarriedSum::default();
        for (index, piece) in chunk.chunks(P).enumerate() {
            let terms = &mut buffer[..piece.len()];
            T::exp_shifted_into(piece, top, terms);
            if let Some(at) = skip.and_then(|at| at.checked_sub(index * P))
                && at < terms.len()
            {
                terms[at] = 0.0;
            }
            sum.add(T::total(terms));
        }
        sum.value()
    }

    let chunks = run.len().div_ceil(RUN_SUM_CHUNK).max(1);
    let chunk =
        |index: usize| &run[index * RUN_SUM_CHUNK..((index + 1) * RUN_SUM_CHUNK).min(run.len())];
    fn over_chunks<U: Send>(
        chunks: usize,
        spread: bool,
        work: &(dyn Fn(usize) -> U + Sync),
    ) -> Vec<U> {
        if spread && chunks > 1 {
            par_map_indexed(chunks, work)
        } else {
            (0..chunks).map(work).collect()
        }
    }
    let top = over_chunks(chunks, spread, &|index| chunk_max(chunk(index)))
        .into_iter()
        .fold(T::neg_infinity(), nan_max);
    if !top.is_finite() {
        // `+inf` runs reduce to `+inf`, all-`-inf` runs to `-inf`, and a NaN
        // propagates.
        return top;
    }
    // The maximum's own term, `exp(0) = 1` exactly, is left out of the sum
    // and comes back through `log1p`: `logsumexp([0, -20])` is
    // `log1p(2e-9)`, and the logarithm of the whole total gave 0 in float32.
    // See `ExpTotal`.
    let at = crate::ops::activation::first_position(run, top);
    let rest = pairwise_fold(
        over_chunks(chunks, spread, &|index| {
            let skip = at.and_then(|at| at.checked_sub(index * RUN_SUM_CHUNK));
            chunk_sum(chunk(index), top, skip.filter(|&s| s < RUN_SUM_CHUNK))
        }),
        0.0,
        |a, b| a + b,
    );
    let log_total = crate::ops::activation::log1p_of_rest(rest);
    let value = top.wide() + log_total;
    if lse_settled::<T>(value, log_total) {
        T::narrow(value)
    } else {
        let values: Vec<f64> = run.iter().map(|v| v.wide()).collect();
        T::narrow(exact_logsumexp(&values, top.wide()))
    }
}

/// Whether a log-sum-exp's float64 `value` -- the maximum plus the logarithm
/// of the total, `log_total` -- stands: for float64, where it has not
/// cancelled, the sum at least the logarithm it was formed from; for float32,
/// where its rounding is settled, the value moved either way by a bound on its
/// error rounding the same way.
///
/// A float32 total is a blocked float64 sum, a few hundred additions in a
/// chain at most, so within 2^-44 of itself; the bound is twice that, and the
/// logarithm's and the sum's own roundings. Where the sum is many times
/// smaller than the logarithm the bound is a larger share of it, but a float32
/// value is in doubt only near a point where its rounding changes.
#[inline]
fn lse_settled<T: ShiftedExp>(value: f64, log_total: f64) -> bool {
    if T::CARRIES {
        value.is_nan() || value.abs() >= log_total
    } else {
        const TOTAL: f64 = 1.0 / 8_796_093_022_208.0; // 2^-43
        const VALUE: f64 = 1.0 / 4_503_599_627_370_496.0; // 2^-52
        let bound = log_total * TOTAL + value.abs() * VALUE;
        value.is_nan() || (value - bound) as f32 == (value + bound) as f32
    }
}

/// `log(sum(exp(x)))` of a slice whose float64 value did not stand -- see
/// [`lse_settled`] -- formed again with every term `exp(x - top)` carried as
/// a pair of floats, the logarithm of their total too, and its sum with `top`
/// rounded once; see `arithmetic::extended`. `top` is the slice's maximum,
/// finite.
///
/// The sum of log-probabilities' exponentials is one, less what rounding them
/// left, and its logarithm the size of that rounding: float64 terms put an ulp
/// of the logarithm of the largest into it, and a float64 `logsumexp` of
/// normalised log-probabilities was hundreds of millions of ulps out.
#[cold]
fn exact_logsumexp(values: &[f64], top: f64) -> f64 {
    use crate::ops::arithmetic::extended::{self, Pair};
    use crate::ops::arithmetic::rounded_once::two_sum;
    #[inline(always)]
    fn body<const FMA: bool>(values: &[f64], top: f64) -> f64 {
        // The total less one, so a total near one keeps its small part; the
        // maximum's own term is one exactly. A term below `exp(-700)` is
        // below anything the sum with `top` can hold.
        let mut excess = Pair(-1.0, 0.0);
        for &v in values {
            let (d, e) = two_sum(v, -top);
            if d >= -700.0 {
                excess = excess.add(extended::exp::<FMA>(Pair(d, e)));
            }
        }
        let l = extended::ln_1p::<FMA>(excess);
        let (sum, error) = two_sum(top, l.0);
        sum + (error + l.1)
    }
    extended::fused!(body(values: &[f64], top: f64) -> f64)
}

/// Log-sum-exp down each column of a row-major `(rows, cols)` slab; a single
/// column is a contiguous run and goes to [`lse_run`].
///
/// Two passes over bands of rows: the column maxima, taken `EXP_GROUP` lanes
/// at a time with narrow rows batched together, then the column sums of
/// `exp(x - max)` from [`exp_columns_into`], which never stores the
/// exponentials beyond one group. Bands fold pairwise.
///
/// Together with [`lse_run`] this replaced a loop calling scalar `exp` per
/// element that, for a last-axis reduction, walked the whole axis on one
/// thread. float32 on four cores, before -> after:
///
/// ```text
///   (4194304,)        13.6ms -> 1.42
///   (1024, 4096) d1    4.30  -> 1.27
///   (4096, 1024) d0    4.38  -> 2.50
///   (16, 100000, 2) d1 3.64  -> 1.70
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
    fn over_bands<U: Send>(
        bands: usize,
        spread: bool,
        work: &(dyn Fn(usize) -> Vec<U> + Sync),
    ) -> Vec<Vec<U>> {
        if spread && bands > 1 {
            par_map_indexed(bands, work)
        } else {
            (0..bands).map(work).collect()
        }
    }
    // Both passes take rows a group at a time, `per_group` of them, into
    // `group` lanes; lane `j` belongs to column `j % cols`.
    let per_group = EXP_GROUP.div_ceil(cols);
    let group = per_group * cols;

    // A NaN is recorded beside the lane rather than branched on, which is what
    // lets the comparison loop vectorize.
    let maxima = pairwise_fold_vectors(
        over_bands(bands, spread, &|index| {
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

    let totals = pairwise_fold_vectors(
        over_bands(bands, spread, &|index| {
            exp_columns_into::<T, true>(band(index), None, &maxima, cols)
        }),
        |a, b| a + b,
    );

    // The logarithm of each total split as `ExpTotal` splits it, so a total
    // near one keeps its small part; a column whose value does not stand is
    // formed again from its terms as pairs.
    maxima
        .iter()
        .zip(&totals)
        .enumerate()
        .map(|(c, (&m, &t))| {
            if !m.is_finite() {
                return m;
            }
            let log_total = t.ln();
            let value = m.wide() + log_total;
            if lse_settled::<T>(value, log_total) {
                T::narrow(value)
            } else {
                let column: Vec<f64> = slab
                    .iter()
                    .skip(c)
                    .step_by(cols)
                    .map(|v| v.wide())
                    .collect();
                T::narrow(exact_logsumexp(&column, m.wide()))
            }
        })
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
        // A float product over any axes, each the float nearest the exact
        // one.
        Some(ref dims) if dims.is_empty() => return Ok(tensor.clone()),
        _ if tensor.dtype().is_float() => {
            let axes = dim.unwrap_or_else(|| (0..tensor.ndim()).collect());
            let tracked = tensor.requires_grad() && crate::autograd::is_grad_enabled();
            super::sort_impl::rounded_product(tensor, &axes, keepdim)?.requires_grad_(tracked)
        }
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let out_dtype = accumulating_dtype(tensor.dtype());
            let mut result_data = TensorData::zeros_on_device(1, out_dtype, tensor.device());
            match tensor.dtype() {
                DataType::Int32 => prod_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => prod_all_i64(tensor, &mut result_data)?,
                _ => unreachable!("a float is taken above, and bool was promoted"),
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
                // Reducing over no axes is the tensor itself, which is already
                // in the graph. Recording a node for it made the input its own
                // output -- a cycle backward refused.
                return Ok(tensor.clone());
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
        DataType::Float32 => exact_cumprod(
            tensor,
            &mut result_data,
            dim,
            TensorData::as_f32_slice,
            TensorData::as_f32_slice_mut,
        )?,
        DataType::Float64 => exact_cumprod(
            tensor,
            &mut result_data,
            dim,
            TensorData::as_f64_slice,
            TensorData::as_f64_slice_mut,
        )?,
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

/// The running products along `dim` of a float tensor, each the float
/// nearest the exact product of the values so far; see
/// `arithmetic::exact_product::running_products`. A running product rounded
/// at every step drifted from the exact one, and a float32 one carried in
/// float64 was rounded again on the way back.
fn exact_cumprod<T>(
    tensor: &Tensor,
    result_data: &mut TensorData,
    dim: usize,
    read: fn(&TensorData) -> Option<&[T]>,
    write: fn(&mut TensorData) -> Option<&mut [T]>,
) -> Result<()>
where
    T: crate::ops::arithmetic::exact_sum::Element + crate::ops::arithmetic::exact_sum::Output,
{
    let contiguous = tensor.contiguous()?;
    let input = read(contiguous.data())
        .ok_or_else(|| MinitensorError::internal_error("cumprod: float slice"))?;
    let output = write(result_data)
        .ok_or_else(|| MinitensorError::internal_error("cumprod: float output"))?;
    let shape = tensor.shape().dims();
    let inner = shape[dim + 1..].iter().product::<usize>();
    crate::ops::arithmetic::exact_product::running_products(input, output, shape[dim], inner);
    Ok(())
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
///
/// A float mean sums before it divides, and where the sum leaves the float
/// range the mean need not: `mean([1e308, 1e308])` is `1e308`, and was
/// infinite. Such a mean is taken again over each slice divided by a power of
/// two near its largest magnitude, which is exact, and scaled back.
pub fn mean(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    let result = mean_direct(tensor, dim.clone(), keepdim)?;
    rescue_mean(tensor, dim, result, |scaled, dim| {
        mean_direct(scaled, dim, keepdim)
    })
}

/// `result`, a mean of `tensor` over `dim`, with each slice whose sum may
/// have overflowed taken again by `compute` over the slice scaled into range
/// and scaled back; see [`super::rescue_scales`].
fn rescue_mean(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    result: Tensor,
    compute: impl FnOnce(&Tensor, Option<Vec<isize>>) -> Result<Tensor>,
) -> Result<Tensor> {
    // A float32 mean is summed in float64, which no float32 values overflow.
    if tensor.dtype() != DataType::Float64 || tensor.numel() == 0 {
        return Ok(result);
    }
    let dims = super::sort_impl::statistic_dims(tensor, &dim)?;
    if dims.is_empty() {
        return Ok(result);
    }
    let Some(scales) = super::rescue_scales(tensor, &dims, &result)? else {
        return Ok(result);
    };
    let (averaged, scales) =
        super::sort_impl::rescaled(tensor, &scales, |scaled| compute(scaled, dim))?;
    crate::ops::arithmetic::mul(&averaged, &scales)
}

/// The mean as computed directly: the sum over the count.
fn mean_direct(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // Normalise negative dimensions and deduplicate
    let normalized = normalize_reduction_dims(dim, tensor.ndim())?;
    if matches!(&normalized, Some(dims) if dims.is_empty()) {
        return Ok(tensor.clone());
    }
    let num_elements = slice_length(tensor, normalized.as_deref()) as f64;
    let as_isize = |dims: &Option<Vec<usize>>| {
        dims.as_ref()
            .map(|d| d.iter().map(|&x| x as isize).collect::<Vec<_>>())
    };

    match tensor.dtype() {
        // The exact sum over the count rounded once to the tensor's own
        // dtype: over every axis from the whole run, along one axis in place,
        // over several with those axes moved into one run. Summed and then
        // divided it rounded twice, and a float32 mean carried in float64
        // rounded a third time where the float64 mean fell on a float32
        // halfway point. The gradient is the mean's own, the output's over
        // the count.
        DataType::Float32 | DataType::Float64 => {
            let value = if tensor.dtype() == DataType::Float32 {
                exact_mean::<f32>(tensor, normalized.as_deref(), keepdim)?
            } else {
                exact_mean::<f64>(tensor, normalized.as_deref(), keepdim)?
            };
            with_mean_gradient(tensor, value, normalized, keepdim, num_elements)
        }
        // Over every axis, the exact 128-bit sum divided once.
        DataType::Int64
            if normalized
                .as_ref()
                .is_none_or(|dims| dims.len() == tensor.ndim()) =>
        {
            let value = crate::ops::arithmetic::exact_sum::mean_i64(
                tensor.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("mean: failed to get the i64 slice")
                })?,
            );
            let value = scalar_result::<f64>(tensor, value, keepdim)?;
            with_mean_gradient(tensor, value, normalized, keepdim, num_elements)
        }
        // An int64 sum wraps past 2^63 though the mean it feeds is always in
        // range: the mean of [2^62, 2^62] came back as -2^62. The result is
        // float64 either way, so the values are summed there.
        DataType::Int64 => {
            let total = sum(
                &tensor.astype(DataType::Float64)?,
                as_isize(&normalized),
                keepdim,
            )?;
            super::sort_impl::divide_by_count(&total, num_elements)
        }
        // An int32 sum accumulates in int64, which holds any count of int32
        // values that fits in memory, so it is the exact integer sum -- and
        // converting it to float32 before dividing rounded it first: the mean
        // of 2^24 + 1 threes came out 3.0000002. It is divided in float64,
        // where it and the count are exact, and rounds once on its way to
        // float32.
        DataType::Int32 => {
            let total = sum(tensor, as_isize(&normalized), keepdim)?;
            super::sort_impl::divide_by_count(&total.astype(DataType::Float64)?, num_elements)?
                .astype(DataType::Float32)
        }
        DataType::Bool => Err(MinitensorError::invalid_operation(
            "Mean not supported for boolean tensors",
        )),
    }
}

/// The mean of a float tensor over `dims` (every axis for `None`), each output
/// the exact sum over the count rounded once to the tensor's dtype.
fn exact_mean<T: super::sum_prod_impl::ExactFloat>(
    tensor: &Tensor,
    dims: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor> {
    let reduction = crate::ops::arithmetic::exact_sum::Reduction::Mean;
    match dims {
        Some(dims) if dims.len() < tensor.ndim() => {
            if let [dim] = dims[..] {
                let shape = tensor.shape().dims();
                let kept = |d: &usize| *d != dim;
                let outputs: usize = (0..tensor.ndim()).filter(kept).map(|d| shape[d]).product();
                let mut data = TensorData::zeros_on_device(outputs, T::DTYPE, tensor.device());
                super::sum_prod_impl::reduce_along_dim_exactly::<T>(
                    tensor, &mut data, dim, reduction,
                )?;
                let result_shape: Vec<usize> = if keepdim {
                    (0..tensor.ndim())
                        .map(|d| if d == dim { 1 } else { shape[d] })
                        .collect()
                } else {
                    (0..tensor.ndim()).filter(kept).map(|d| shape[d]).collect()
                };
                Ok(Tensor::new(
                    Arc::new(data),
                    Shape::new(result_shape),
                    T::DTYPE,
                    tensor.device(),
                    false,
                ))
            } else {
                super::sum_prod_impl::reduce_dims_exactly::<T>(
                    tensor, dims, keepdim, reduction, false,
                )
            }
        }
        _ => {
            let values = T::values(tensor.data())
                .ok_or_else(|| MinitensorError::internal_error("mean: failed to get the slice"))?;
            let value: T = crate::ops::arithmetic::exact_sum::mean_to(values);
            scalar_result(tensor, value, keepdim)
        }
    }
}

/// A one-element result of a reduction over every axis of `tensor`.
fn scalar_result<T: super::sum_prod_impl::ExactFloat>(
    tensor: &Tensor,
    value: T,
    keepdim: bool,
) -> Result<Tensor> {
    let shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };
    let mut data = TensorData::zeros_on_device(1, T::DTYPE, tensor.device());
    T::values_mut(&mut data)
        .ok_or_else(|| MinitensorError::internal_error("mean: no output slice"))?[0] = value;
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        T::DTYPE,
        tensor.device(),
        false,
    ))
}

/// `value`, the mean of `tensor`, with the mean's gradient when one is
/// wanted.
fn with_mean_gradient(
    tensor: &Tensor,
    value: Tensor,
    dims: Option<Vec<usize>>,
    keepdim: bool,
    count: f64,
) -> Result<Tensor> {
    if !(tensor.requires_grad() && crate::autograd::is_grad_enabled()) {
        return Ok(value);
    }
    let grad_fn = Arc::new(crate::autograd::MeanBackward {
        input_id: tensor.id(),
        input_shape: tensor.shape().dims().to_vec(),
        dims,
        keepdim,
        count,
    });
    with_grad_fn(value.requires_grad_(true), grad_fn)
}

/// How many elements a reduction over `dims` takes into each result, every
/// element for `None`.
pub(crate) fn slice_length(tensor: &Tensor, dims: Option<&[usize]>) -> usize {
    match dims {
        None => tensor.numel(),
        Some(dims) => dims.iter().map(|&d| tensor.shape().dims()[d]).product(),
    }
}

/// NaN-aware mean reduction along specified dimensions: each the float
/// nearest the exact mean of its slice's values that are not NaN, NaN for a
/// slice of nothing but NaN. The gradient is one over that count at each value
/// kept and zero at each NaN skipped.
pub fn nanmean(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return mean(tensor, dim, keepdim);
    }
    let dims = normalize_reduction_dims(dim.clone(), tensor.ndim())?;
    if dims.as_ref().is_some_and(|dims| dims.is_empty()) {
        return Ok(tensor.clone());
    }
    let axes = dims.clone().unwrap_or_else(|| (0..tensor.ndim()).collect());
    let result = super::sort_impl::rounded_nan_total(tensor, &axes, keepdim, true)?;
    if !(tensor.requires_grad() && crate::autograd::is_grad_enabled()) {
        return Ok(result);
    }
    let grad_fn = Arc::new(NanMeanBackward {
        input_id: tensor.id(),
        input_shape: tensor.shape().dims().to_vec(),
        dims,
        keepdim,
        mask: non_nan_mask(tensor)?,
        count: super::nan_stats_impl::non_nan_counts(tensor, dim, keepdim)?,
    });
    with_grad_fn(result.requires_grad_(true), grad_fn)
}
