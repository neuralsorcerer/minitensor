// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Running totals that are not sums.
//!
//! `cumsum` and `cumprod` already exist, built on [`scan_along_dim`] with `+`
//! and `*` as the step. These are the other three worth having: the running
//! maximum and minimum, which also report *where* they came from, and the
//! running log-sum-exp, which is the only way to accumulate probabilities over
//! a long axis without underflowing.
//!
//! None of them composes. A scan is a recurrence -- each step needs the
//! previous step's result -- and no arrangement of the elementwise, reduction
//! and contraction operations in this library runs one.
//!
//! ## Why `logcumsumexp` is not `cumsum` of `exp`
//!
//! That is what it means, and it is unusable. `exp` of a log-probability
//! underflows to zero after a few hundred steps, so the running total stops
//! moving and the answer becomes `-inf` for every position past that. Staying
//! in the log domain and combining with [`log_add_exp`] keeps every step
//! representable, which is the whole point of the operation existing separately.
//!
//! ## Why the extremum scan is not `scan_along_dim`
//!
//! It writes two outputs rather than one, and the second is a position.
//! `scan_along_dim` seeds with `out[i] = widen(inp[i])` -- a closure that never
//! sees the index it is seeding -- so a position cannot be introduced. Same
//! walk, different shape, which is a different computation rather than a
//! duplicated one.

use crate::{
    autograd::{CummaxBackward, LogcumsumexpBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        map::par_out_chunks2,
        reduction::scan_along_dim,
        util::{log_add_exp, normalize_dim},
    },
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// How a tensor is laid out for a scan along one axis: `outer` slabs, each of
/// `dim_size` rows of `inner` contiguous elements.
fn slab_of(tensor: &Tensor, dim: usize) -> (usize, usize) {
    let dims = tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    (dims[dim], inner)
}

/// The running extremum along one axis, and the position it came from.
///
/// `better(candidate, running)` decides whether the candidate replaces what is
/// there. It must be strict, so that a tie leaves the earlier position standing.
fn scan_extreme<T, F>(
    values: &mut [T],
    indices: &mut [i64],
    input: &[T],
    dim_size: usize,
    inner: usize,
    better: F,
) where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> bool + Send + Sync,
{
    let slab = dim_size * inner;
    if slab == 0 {
        return;
    }
    par_out_chunks2(values, indices, slab, &|start, value_slab, index_slab| {
        let source = &input[start..start + value_slab.len()];
        for i in 0..inner {
            value_slab[i] = source[i];
            index_slab[i] = 0;
        }
        for d in 1..dim_size {
            let (previous, current) =
                value_slab[(d - 1) * inner..(d + 1) * inner].split_at_mut(inner);
            let (before, now) = index_slab[(d - 1) * inner..(d + 1) * inner].split_at_mut(inner);
            for i in 0..inner {
                let candidate = source[d * inner + i];
                if better(candidate, previous[i]) {
                    current[i] = candidate;
                    now[i] = d as i64;
                } else {
                    current[i] = previous[i];
                    now[i] = before[i];
                }
            }
        }
    });
}

/// Both extremum scans, which differ only in which way the comparison runs.
fn cumextreme(tensor: &Tensor, dim: isize, greater: bool) -> Result<(Tensor, Tensor)> {
    let dim = normalize_dim(dim, tensor.ndim())?;
    if tensor.dtype() == DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "cummax and cummin are not supported for boolean tensors",
        ));
    }
    let contiguous = tensor.contiguous()?;
    let (dim_size, inner) = slab_of(tensor, dim);
    let numel = tensor.numel();
    let device = tensor.device();

    let mut value_data = TensorData::zeros_on_device(numel, tensor.dtype(), device);
    let mut index_data = TensorData::zeros_on_device(numel, DataType::Int64, device);
    let positions = index_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("cummax: indices are not int64"))?;

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident, $ty:ty, $nan:expr) => {{
            let source = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("cummax: dtype does not match the input")
            })?;
            let out = value_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("cummax: dtype does not match the output")
            })?;
            // A NaN swallows the running extremum and keeps it -- the first one
            // wins, and nothing after it displaces it. Without the guard on
            // `running` a later NaN would move the index while changing nothing
            // about the value.
            let unordered: fn($ty) -> bool = $nan;
            let better = move |candidate: $ty, running: $ty| {
                !unordered(running)
                    && (unordered(candidate)
                        || if greater {
                            candidate > running
                        } else {
                            candidate < running
                        })
            };
            scan_extreme(out, positions, source, dim_size, inner, better);
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut, f32, |v| v.is_nan()),
        DataType::Float64 => run!(as_f64_slice, as_f64_slice_mut, f64, |v| v.is_nan()),
        DataType::Int32 => run!(as_i32_slice, as_i32_slice_mut, i32, |_| false),
        DataType::Int64 => run!(as_i64_slice, as_i64_slice_mut, i64, |_| false),
        DataType::Bool => unreachable!("rejected above"),
    }

    let shape = tensor.shape().clone();
    let indices = Tensor::new(
        Arc::new(index_data),
        shape.clone(),
        DataType::Int64,
        device,
        false,
    );
    let mut values = Tensor::new(
        Arc::new(value_data),
        shape,
        tensor.dtype(),
        device,
        tensor.requires_grad() && tensor.dtype().is_float(),
    );

    if values.requires_grad() {
        // Each output took its value from exactly one input position, so the
        // gradient goes there and nowhere else -- the same rule `max` follows,
        // applied once per prefix.
        let grad_fn = Arc::new(CummaxBackward {
            input_id: tensor.id(),
            indices: indices.detach(),
            dim,
        });
        values = with_grad_fn(values, grad_fn)?;
    }
    Ok((values, indices))
}

/// The running maximum along `dim`, and where each one came from.
///
/// Ties go to the earliest position holding the value, which is a choice rather
/// than a consequence -- and the same one `mode` makes here. A `NaN` takes over
/// the running maximum and keeps it, as it does for `max`.
pub fn cummax(tensor: &Tensor, dim: isize) -> Result<(Tensor, Tensor)> {
    cumextreme(tensor, dim, true)
}

/// The running minimum along `dim`, and where each one came from.
pub fn cummin(tensor: &Tensor, dim: isize) -> Result<(Tensor, Tensor)> {
    cumextreme(tensor, dim, false)
}

/// Scan `input` into `output` with [`log_add_exp`], forwards or backwards.
pub(crate) fn logcumsumexp_raw(
    output: &mut [f64],
    input: &[f64],
    dim_size: usize,
    inner: usize,
    reverse: bool,
) {
    scan_along_dim(
        output,
        input,
        dim_size,
        inner,
        reverse,
        |v: f64| v,
        log_add_exp::<f64>,
    );
}

/// `log(sum(exp(x)))` accumulated along `dim`.
///
/// The running total of probabilities held as logarithms. Written as `cumsum`
/// of `exp` it underflows to zero after a few hundred steps and reports `-inf`
/// for everything after; combining in the log domain keeps every step
/// representable, which is why it exists as its own operation.
pub fn logcumsumexp(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "logcumsumexp is only supported for floating point tensors",
        ));
    }
    let contiguous = tensor.contiguous()?;
    let (dim_size, inner) = slab_of(tensor, dim);
    let numel = tensor.numel();
    let device = tensor.device();
    let mut data = TensorData::zeros_on_device(numel, tensor.dtype(), device);

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident, $ty:ty) => {{
            let source = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("logcumsumexp: dtype does not match the input")
            })?;
            let out = data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("logcumsumexp: dtype does not match the output")
            })?;
            scan_along_dim(
                out,
                source,
                dim_size,
                inner,
                false,
                |v: $ty| v,
                log_add_exp::<$ty>,
            );
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut, f32),
        _ => run!(as_f64_slice, as_f64_slice_mut, f64),
    }

    let mut result = Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        device,
        tensor.requires_grad(),
    );

    if result.requires_grad() {
        let grad_fn = Arc::new(LogcumsumexpBackward {
            input_id: tensor.id(),
            input: tensor.detach(),
            output: result.detach(),
            dim,
        });
        result = with_grad_fn(result, grad_fn)?;
    }
    Ok(result)
}

/// Route each output's gradient back to the input position that supplied it.
pub(crate) fn cumextreme_backward(
    indices: &Tensor,
    grad_output: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    let positions = indices.contiguous()?;
    let seeds = grad_output.contiguous()?;
    let (dim_size, inner) = slab_of(indices, dim);
    let numel = indices.numel();
    let device = grad_output.device();
    let mut data = TensorData::zeros_on_device(numel, grad_output.dtype(), device);
    let source = positions
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("cummax backward: indices are not int64"))?;

    macro_rules! route {
        ($accessor:ident, $accessor_mut:ident) => {{
            let grad = seeds.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("cummax backward: dtype does not match")
            })?;
            let out = data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("cummax backward: dtype does not match")
            })?;
            let slab = dim_size * inner;
            if slab > 0 {
                // Serial within a slab: several prefixes can name the same input
                // position, and they accumulate there. Slabs are disjoint, so
                // the outer walk could be parallel -- it is not, because the
                // scatter is a handful of adds per element and the parallel
                // split would cost more than it saved.
                for slab_start in (0..numel).step_by(slab) {
                    for d in 0..dim_size {
                        for i in 0..inner {
                            let at = slab_start + d * inner + i;
                            let target = slab_start + source[at] as usize * inner + i;
                            out[target] += grad[at];
                        }
                    }
                }
            }
        }};
    }

    match grad_output.dtype() {
        DataType::Float32 => route!(as_f32_slice, as_f32_slice_mut),
        _ => route!(as_f64_slice, as_f64_slice_mut),
    }

    Ok(Tensor::new(
        Arc::new(data),
        indices.shape().clone(),
        grad_output.dtype(),
        device,
        false,
    ))
}

/// The gradient of [`logcumsumexp`], kept in the log domain throughout.
///
/// `dL/dx_i = sum over k >= i of g_k * exp(x_i - y_k)`. Factoring `exp(x_i)`
/// out leaves a reverse cumulative sum of `g_k * exp(-y_k)` -- which overflows
/// for the very negative `y_k` that log-probabilities produce, and that is the
/// case this operation exists to serve. So the scan runs on `log|g_k| - y_k`
/// instead, reusing the forward's own combine, and the sign of `g` is carried
/// by splitting it into two scans and subtracting them at the end.
pub(crate) fn logcumsumexp_backward(
    input: &Tensor,
    output: &Tensor,
    grad_output: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    let x = input.contiguous()?;
    let y = output.contiguous()?;
    let g = grad_output.contiguous()?;
    let (dim_size, inner) = slab_of(input, dim);
    let numel = input.numel();
    let device = input.device();
    let mut data = TensorData::zeros_on_device(numel, input.dtype(), device);

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident) => {{
            let xs = x.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("logcumsumexp backward: dtype does not match")
            })?;
            let ys = y.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("logcumsumexp backward: dtype does not match")
            })?;
            let gs = g.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("logcumsumexp backward: dtype does not match")
            })?;
            let out = data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("logcumsumexp backward: dtype does not match")
            })?;

            let mut rising = vec![f64::NEG_INFINITY; numel];
            let mut falling = vec![f64::NEG_INFINITY; numel];
            for k in 0..numel {
                let weight = gs[k] as f64;
                let shifted = -(ys[k] as f64);
                if weight > 0.0 {
                    rising[k] = weight.ln() + shifted;
                } else if weight < 0.0 {
                    falling[k] = (-weight).ln() + shifted;
                }
            }
            let mut up = vec![0.0f64; numel];
            let mut down = vec![0.0f64; numel];
            logcumsumexp_raw(&mut up, &rising, dim_size, inner, true);
            logcumsumexp_raw(&mut down, &falling, dim_size, inner, true);
            for i in 0..numel {
                let base = xs[i] as f64;
                let positive = (base + up[i]).exp();
                let negative = (base + down[i]).exp();
                out[i] = (positive - negative) as _;
            }
        }};
    }

    match input.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut),
        _ => run!(as_f64_slice, as_f64_slice_mut),
    }

    Ok(Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        input.dtype(),
        device,
        false,
    ))
}
