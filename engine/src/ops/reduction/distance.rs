// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Distances between rows, each rounded once.
//!
//! A distance was the norm of the difference, and the difference was rounded
//! before the norm saw it: between rows of a hundred normal values, one
//! float64 2-distance in five and nearly half the 1-distances were not the
//! float nearest the exact one. Each 1- and 2-distance is now the float
//! nearest the exact value of the exact differences; see
//! `arithmetic::exact_sum::distance`.

use std::sync::Arc;

use crate::{
    error::{MinitensorError, Result},
    ops::arithmetic::exact_sum::{Element, Output, cosine, distance, normalized},
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// Elements of work a task takes at least.
const TASK: usize = 1 << 14;

/// The `p`-distance, `p` 1 or 2, between each pair of corresponding rows of
/// `a` and `b` -- their last axis -- with every difference shifted by
/// `shift`, read at the operands' width. `a` and `b` share a shape and a float
/// dtype; the result drops the last axis.
pub fn row_distances(a: &Tensor, b: &Tensor, p: f64, shift: f64) -> Result<Tensor> {
    let euclidean = check(a, b, p, "row_distances")?;
    if a.shape() != b.shape() || a.ndim() == 0 {
        return Err(MinitensorError::invalid_operation(format!(
            "row_distances takes two tensors of one shape with rows, got {:?} and {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let dims = a.shape().dims();
    let width = dims[dims.len() - 1];
    let shape = Shape::new(dims[..dims.len() - 1].to_vec());
    let rows = shape.numel();
    let data = match a.dtype() {
        DataType::Float32 => {
            let shift = f64::from(shift as f32);
            let read = TensorData::as_f32_slice;
            let out = pairs(read, &a, &b, rows, width, euclidean, shift, same_rows)?;
            TensorData::from_vec_f32(out, a.device())
        }
        _ => {
            let read = TensorData::as_f64_slice;
            let out = pairs(read, &a, &b, rows, width, euclidean, shift, same_rows)?;
            TensorData::from_vec_f64(out, a.device())
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        a.dtype(),
        a.device(),
        false,
    ))
}

/// Every `p`-distance, `p` 1 or 2, between the rows of `a`, laid out
/// `[batch, n, d]`, and the rows of `b`, `[batch, m, d]`, of one float dtype:
/// `[batch, n, m]`.
pub fn cross_distances(a: &Tensor, b: &Tensor, p: f64) -> Result<Tensor> {
    let euclidean = check(a, b, p, "cross_distances")?;
    let (ad, bd) = (a.shape().dims(), b.shape().dims());
    if ad.len() != 3 || bd.len() != 3 || ad[0] != bd[0] || ad[2] != bd[2] {
        return Err(MinitensorError::invalid_operation(format!(
            "cross_distances takes [batch, n, d] and [batch, m, d], got {ad:?} and {bd:?}"
        )));
    }
    let (batch, n, m, width) = (ad[0], ad[1], bd[1], ad[2]);
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let shape = Shape::new(vec![batch, n, m]);
    // Output `(k, i, j)` pairs row `k · n + i` of `a` with row `k · m + j`
    // of `b`; the indices are found once a block and then counted on.
    let rows = move |start: usize| {
        let (k, rest) = (start / (n * m).max(1), start % (n * m).max(1));
        let (mut i, mut j) = (k * n + rest / m.max(1), rest % m.max(1));
        let mut k = k;
        std::iter::from_fn(move || {
            let pair = (i, k * m + j);
            j += 1;
            if j == m {
                j = 0;
                i += 1;
                if i == (k + 1) * n {
                    k += 1;
                }
            }
            Some(pair)
        })
    };
    let data = match a.dtype() {
        DataType::Float32 => {
            let read = TensorData::as_f32_slice;
            let out = pairs(read, &a, &b, batch * n * m, width, euclidean, 0.0, rows)?;
            TensorData::from_vec_f32(out, a.device())
        }
        _ => {
            let read = TensorData::as_f64_slice;
            let out = pairs(read, &a, &b, batch * n * m, width, euclidean, 0.0, rows)?;
            TensorData::from_vec_f64(out, a.device())
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        a.dtype(),
        a.device(),
        false,
    ))
}

/// Each slice of float `tensor` along `dim` over the greater of its
/// `p`-norm, `p` 1 or 2, and `eps` -- `eps` as the dtype holds it -- each
/// element the float nearest the exact quotient; see
/// `arithmetic::exact_sum::normalized`. No gradient: `normalize` gives it.
pub fn normalized_slices(tensor: &Tensor, p: f64, dim: usize, eps: f64) -> Result<Tensor> {
    let taxicab = match p {
        1.0 => true,
        2.0 => false,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "normalized_slices takes p = 1 or p = 2, got {p}"
            )));
        }
    };
    if dim >= tensor.ndim() {
        return Err(MinitensorError::invalid_operation(format!(
            "normalized_slices: dim {dim} is past the last of {} axes",
            tensor.ndim()
        )));
    }
    let tensor = tensor.contiguous()?;
    let dims = tensor.shape().dims();
    let (len, inner) = (dims[dim], dims[dim + 1..].iter().product::<usize>());
    let missing = || MinitensorError::internal_error("normalized_slices: a float slice");
    let mut data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
    match tensor.dtype() {
        DataType::Float32 => {
            let values = tensor.data().as_f32_slice().ok_or_else(missing)?;
            let out = data.as_f32_slice_mut().ok_or_else(missing)?;
            over_norms(values, out, len, inner, taxicab, eps);
        }
        DataType::Float64 => {
            let values = tensor.data().as_f64_slice().ok_or_else(missing)?;
            let out = data.as_f64_slice_mut().ok_or_else(missing)?;
            over_norms(values, out, len, inner, taxicab, eps);
        }
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "normalized_slices takes a float32 or float64 tensor, got {other:?}"
            )));
        }
    }
    Ok(Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// [`normalized_slices`] of `values` laid out `[outer, len, inner]`, along
/// the middle axis, into `out`, laid out the same: whole slices to a task,
/// each gathered where it is not side by side.
fn over_norms<T: Element + Output>(
    values: &[T],
    out: &mut [T],
    len: usize,
    inner: usize,
    taxicab: bool,
    eps: f64,
) {
    if values.is_empty() || len == 0 {
        return;
    }
    let slab = len * inner;
    let slices = values.len() / len;
    let per_task = (TASK / len).max(1);
    let parts = crate::parallel::Parts::of(out);
    let task = |index: usize| {
        let (mut scratch, mut row, mut normal) = (Vec::new(), Vec::new(), Vec::new());
        for slice in index * per_task..((index + 1) * per_task).min(slices) {
            let base = slice / inner * slab + slice % inner;
            if inner == 1 {
                // SAFETY: the slice's places are its own, inside `out`.
                let out = unsafe { std::slice::from_raw_parts_mut(parts.start().add(base), len) };
                normalized(&values[base..base + len], out, taxicab, eps, &mut scratch);
                continue;
            }
            row.clear();
            row.extend((0..len).map(|k| values[base + k * inner]));
            normal.clear();
            normal.resize(len, T::narrow(0.0));
            normalized(&row, &mut normal, taxicab, eps, &mut scratch);
            for (k, &value) in normal.iter().enumerate() {
                // SAFETY: as above, a place at a time.
                unsafe { *parts.start().add(base + k * inner) = value };
            }
        }
    };
    let tasks = slices.div_ceil(per_task);
    // A few nanoseconds an element: worth a thread at an eighth of what a
    // plain map waits for.
    if values.len() < crate::ops::map::PAR_THRESHOLD / 8 {
        (0..tasks).for_each(task);
    } else {
        crate::parallel::for_each_task(tasks, &task);
    }
}

/// The cosine of the angle between each pair of corresponding slices of `a`
/// and `b` along `dim`, each norm floored at `eps` -- as the dtype holds it,
/// and no floor where that is not positive -- each the float nearest the
/// exact cosine; see `arithmetic::exact_sum::cosine`. `a` and `b` share a
/// shape and a float dtype; the result drops `dim`. No gradient:
/// `cosine_similarity` gives it.
pub fn cosine_slices(a: &Tensor, b: &Tensor, dim: usize, eps: f64) -> Result<Tensor> {
    check(a, b, 2.0, "cosine_slices")?;
    if a.shape() != b.shape() || dim >= a.ndim() {
        return Err(MinitensorError::invalid_operation(format!(
            "cosine_slices takes two tensors of one shape and an axis of it, got {:?}, {:?} \
             and axis {dim}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let dims = a.shape().dims();
    let (len, inner) = (dims[dim], dims[dim + 1..].iter().product::<usize>());
    let mut kept = dims.to_vec();
    kept.remove(dim);
    let count = kept.iter().product::<usize>();
    let missing = || MinitensorError::internal_error("cosine_slices: a float slice");
    let mut data = TensorData::uninitialized_on_device(count, a.dtype(), a.device());
    if a.dtype() == DataType::Float32 {
        let (x, y) = (a.data().as_f32_slice(), b.data().as_f32_slice());
        let out = data.as_f32_slice_mut().ok_or_else(missing)?;
        over_pairs(
            x.ok_or_else(missing)?,
            y.ok_or_else(missing)?,
            out,
            len,
            inner,
            eps,
        );
    } else {
        let (x, y) = (a.data().as_f64_slice(), b.data().as_f64_slice());
        let out = data.as_f64_slice_mut().ok_or_else(missing)?;
        over_pairs(
            x.ok_or_else(missing)?,
            y.ok_or_else(missing)?,
            out,
            len,
            inner,
            eps,
        );
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(kept),
        a.dtype(),
        a.device(),
        false,
    ))
}

/// [`cosine_slices`] of `a` and `b` laid out `[outer, len, inner]`, along
/// the middle axis, into `out`, laid out `[outer, inner]`: whole pairs of
/// slices to a task, each gathered where it is not side by side.
fn over_pairs<T: Element + Output>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    len: usize,
    inner: usize,
    eps: f64,
) {
    if out.is_empty() {
        return;
    }
    let mut scratch = (Vec::new(), Vec::new());
    if len == 0 {
        // Empty slices, every cosine the same one.
        out.fill(cosine::<T, T>(&[], &[], eps, &mut scratch));
        return;
    }
    let slices = out.len();
    let per_task = (TASK / len).max(1);
    let parts = crate::parallel::Parts::of(out);
    let task = |index: usize| {
        let mut scratch = (Vec::new(), Vec::new());
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for slice in index * per_task..((index + 1) * per_task).min(slices) {
            let base = slice / inner * len * inner + slice % inner;
            let value = if inner == 1 {
                let span = base..base + len;
                cosine(&a[span.clone()], &b[span], eps, &mut scratch)
            } else {
                left.clear();
                left.extend((0..len).map(|k| a[base + k * inner]));
                right.clear();
                right.extend((0..len).map(|k| b[base + k * inner]));
                cosine(&left, &right, eps, &mut scratch)
            };
            // SAFETY: each slice's cosine has its own place inside `out`.
            unsafe { *parts.start().add(slice) = value };
        }
    };
    let tasks = slices.div_ceil(per_task);
    // A few nanoseconds an element: worth a thread at an eighth of what a
    // plain map waits for.
    if a.len() < crate::ops::map::PAR_THRESHOLD / 8 {
        (0..tasks).for_each(task);
    } else {
        crate::parallel::for_each_task(tasks, &task);
    }
}

/// Whether `p` asks for the 2-distance, after checking the operands are two
/// tensors of one float dtype and `p` is 1 or 2.
fn check(a: &Tensor, b: &Tensor, p: f64, name: &str) -> Result<bool> {
    if a.dtype() != b.dtype() || !matches!(a.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} takes two float32 or two float64 tensors, got {:?} and {:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    if a.device() != b.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", a.device()),
            format!("{:?}", b.device()),
        ));
    }
    match p {
        1.0 => Ok(false),
        2.0 => Ok(true),
        _ => Err(MinitensorError::invalid_operation(format!(
            "{name} takes p = 1 or p = 2, got {p}"
        ))),
    }
}

/// The rows `row_distances` pairs from output `start` on: each with itself.
fn same_rows(start: usize) -> impl Iterator<Item = (usize, usize)> {
    (start..).map(|row| (row, row))
}

/// `count` distances, the `index`-th between the rows of `a` and `b` that
/// `rows(start)` names in turn from `start` on, each `width` long and read
/// by `read`, a block of them to a task.
#[allow(clippy::too_many_arguments)]
fn pairs<T: Element + Output, R: Iterator<Item = (usize, usize)>>(
    read: fn(&TensorData) -> Option<&[T]>,
    a: &Tensor,
    b: &Tensor,
    count: usize,
    width: usize,
    euclidean: bool,
    shift: f64,
    rows: impl Fn(usize) -> R + Sync,
) -> Result<Vec<T>> {
    let missing = || MinitensorError::internal_error("a distance operand without its float slice");
    let (x, y) = (
        read(a.data()).ok_or_else(missing)?,
        read(b.data()).ok_or_else(missing)?,
    );
    let mut out = vec![T::narrow(0.0); count];
    let per_task = (TASK / width.max(1)).max(1);
    crate::ops::map::par_out_chunks(&mut out, per_task, &|start, block| {
        for (slot, (i, j)) in block.iter_mut().zip(rows(start)) {
            let (row_x, row_y) = (&x[i * width..][..width], &y[j * width..][..width]);
            *slot = distance(row_x, row_y, euclidean, shift);
        }
    });
    Ok(out)
}
