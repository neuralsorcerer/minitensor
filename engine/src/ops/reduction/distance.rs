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
    ops::activation::CarriedSum,
    ops::arithmetic::exact_sum::{Element, Output, cosine, distance, normalized},
    ops::arithmetic::extended::{error_of, fused},
    ops::arithmetic::rounded_once::two_sum,
    ops::map::{PAR_THRESHOLD, par_out_chunks_gated},
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
/// element the float nearest the exact quotient; or with `capped`, each
/// slice whose norm exceeds `eps`, which is not negative, times `eps` over
/// its norm, and any other as it is. See
/// `arithmetic::exact_sum::normalized`. With it, whether each slice was
/// taken over its norm, `tensor`'s shape with one place along `dim`. No
/// gradient: `normalize` and `renorm` give it.
pub fn normalized_slices(
    tensor: &Tensor,
    p: f64,
    dim: usize,
    eps: f64,
    capped: bool,
) -> Result<(Tensor, Tensor)> {
    let taxicab = match p {
        1.0 => true,
        2.0 => false,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "normalized_slices takes p = 1 or p = 2, got {p}"
            )));
        }
    };
    if dim >= tensor.ndim() || (capped && (eps.is_nan() || eps < 0.0)) {
        return Err(MinitensorError::invalid_operation(format!(
            "normalized_slices takes one of the {} axes, and a cap that is a number not below \
             zero; got axis {dim} and {eps}",
            tensor.ndim()
        )));
    }
    let tensor = tensor.contiguous()?;
    let dims = tensor.shape().dims();
    let (len, inner) = (dims[dim], dims[dim + 1..].iter().product::<usize>());
    let missing = || MinitensorError::internal_error("normalized_slices: a float slice");
    let mut data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
    // One flag a slice, empty slices included.
    let mut kept = dims.to_vec();
    kept[dim] = 1;
    let mut over = vec![false; kept.iter().product()];
    let setting = (len, inner, taxicab, eps, capped);
    match tensor.dtype() {
        DataType::Float32 => {
            let values = tensor.data().as_f32_slice().ok_or_else(missing)?;
            let out = data.as_f32_slice_mut().ok_or_else(missing)?;
            over_norms(values, out, &mut over, setting);
        }
        DataType::Float64 => {
            let values = tensor.data().as_f64_slice().ok_or_else(missing)?;
            let out = data.as_f64_slice_mut().ok_or_else(missing)?;
            over_norms(values, out, &mut over, setting);
        }
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "normalized_slices takes a float32 or float64 tensor, got {other:?}"
            )));
        }
    }
    let over = Tensor::new(
        Arc::new(TensorData::from_vec_bool(over, tensor.device())),
        Shape::new(kept),
        DataType::Bool,
        tensor.device(),
        false,
    );
    let quotients = Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    );
    Ok((quotients, over))
}

/// [`normalized_slices`] of `values` laid out `[outer, len, inner]`, along
/// the middle axis, into `out`, laid out the same, and whether each slice
/// was taken over its norm into `over`: whole slices to a task, each
/// gathered where it is not side by side.
fn over_norms<T: Element + Output>(
    values: &[T],
    out: &mut [T],
    over: &mut [bool],
    (len, inner, taxicab, eps, capped): (usize, usize, bool, f64, bool),
) {
    if values.is_empty() || len == 0 {
        return;
    }
    let slab = len * inner;
    let slices = values.len() / len;
    let per_task = (TASK / len).max(1);
    let (parts, flags) = (
        crate::parallel::Parts::of(out),
        crate::parallel::Parts::of(over),
    );
    let task = |index: usize| {
        let (mut scratch, mut row, mut normal) = (Vec::new(), Vec::new(), Vec::new());
        for slice in index * per_task..((index + 1) * per_task).min(slices) {
            let base = slice / inner * slab + slice % inner;
            let taken = if inner == 1 {
                // SAFETY: the slice's places are its own, inside `out`.
                let out = unsafe { std::slice::from_raw_parts_mut(parts.start().add(base), len) };
                normalized(
                    &values[base..base + len],
                    out,
                    taxicab,
                    eps,
                    capped,
                    &mut scratch,
                )
            } else {
                row.clear();
                row.extend((0..len).map(|k| values[base + k * inner]));
                normal.clear();
                normal.resize(len, T::narrow(0.0));
                let taken = normalized(&row, &mut normal, taxicab, eps, capped, &mut scratch);
                for (k, &value) in normal.iter().enumerate() {
                    // SAFETY: as above, a place at a time.
                    unsafe { *parts.start().add(base + k * inner) = value };
                }
                taken
            };
            // SAFETY: the slice's flag is its own, inside `over`.
            unsafe { *flags.start().add(slice) = taken };
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
    let (a, b, len, inner, kept) = cosine_operands(a, b, dim, "cosine_slices")?;
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

/// Each pair's term of `cosine_embedding_loss` along `dim`, and the factor
/// its cosine's gradient takes in it: where its target is one, `1 - cos` and
/// -1; elsewhere `max(0, cos - margin)`, and 1 where that is not flat --
/// zero where it is, NaN where the cosine is. Each norm is floored at `eps`
/// as the dtype holds it, as [`cosine_slices`] floors it.
///
/// `1 - cos` is formed so that where the two lie almost along each other it
/// cancels nothing, as `1 - cos` of the rounded cosine did -- all of it, in
/// float32, once the cosine was within `2^-25` of one: where the cosine `c`
/// is positive it is `|r|^2 / (|b|^2 (1 + c))` for `r` the part of `b` at
/// right angles to `a`, which [`across`] forms whole; where it is
/// not, there is nothing to cancel. Where a floor holds a norm up, the cosine
/// is `s c` for `s` the floors' share, and its complement `(1 - s) + s (1 -
/// c)`. `cos - margin` is the cosine of the pair's sums less the margin. In
/// float64 and rounded once. `a` and `b` share a shape and a float dtype,
/// and `target` has the result's, which drops `dim`.
pub(crate) fn cosine_embedding_slices(
    a: &Tensor,
    b: &Tensor,
    target: &Tensor,
    dim: usize,
    (eps, margin): (f64, f64),
) -> Result<(Tensor, Vec<f64>)> {
    let (a, b, len, inner, kept) = cosine_operands(a, b, dim, "cosine_embedding_slices")?;
    let count = kept.iter().product::<usize>();
    let missing = || MinitensorError::internal_error("cosine_embedding_slices: a float slice");
    let target = target.contiguous()?;
    let target: Vec<f64> = match target.dtype() {
        DataType::Float32 => target
            .data()
            .as_f32_slice()
            .map(|t| t.iter().map(|&v| f64::from(v)).collect()),
        _ => target.data().as_f64_slice().map(<[f64]>::to_vec),
    }
    .ok_or_else(missing)?;
    if target.len() != count {
        return Err(missing());
    }
    let mut terms = vec![0.0f64; count];
    if a.dtype() == DataType::Float32 {
        let (x, y) = (a.data().as_f32_slice(), b.data().as_f32_slice());
        let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
        embedding_pairs(x, y, &target, &mut terms, (len, inner), (eps, margin));
    } else {
        let (x, y) = (a.data().as_f64_slice(), b.data().as_f64_slice());
        let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
        embedding_pairs(x, y, &target, &mut terms, (len, inner), (eps, margin));
    }
    let weights = terms
        .iter()
        .zip(&target)
        .map(|(&term, &t)| {
            if t == 1.0 {
                -1.0
            } else if term > 0.0 {
                1.0
            } else if term.is_nan() {
                f64::NAN
            } else {
                0.0
            }
        })
        .collect();
    let data = if a.dtype() == DataType::Float32 {
        let narrow = terms.iter().map(|&v| v as f32).collect();
        TensorData::from_vec::<f32>(narrow, a.dtype(), a.device())
    } else {
        TensorData::from_vec::<f64>(terms, a.dtype(), a.device())
    };
    let terms = Tensor::new(
        Arc::new(data),
        Shape::new(kept),
        a.dtype(),
        a.device(),
        false,
    );
    Ok((terms, weights))
}

/// The gradient of the cosine of each pair of slices of `a` and `b` along
/// `dim` -- in `a` where `which` is 0 and in `b` where it is 1 -- times
/// `grad`, one value a pair, laid out as the cosines are: for `a` it is
/// `g (b - (a . b / |a|^2) a) / (A B)`, `A` and `B` the norms floored at
/// `eps`, and `b` alone where `a`'s norm is under its floor and so does not
/// move the cosine's denominator. The difference is [`across`]'s,
/// formed whole, where as `b / (A B) - c a / A^2` it was two terms that all
/// but cancel for slices that lie almost along each other. In float64 and
/// rounded once; the result has `a`'s shape.
pub(crate) fn cosine_gradient_slices(
    a: &Tensor,
    b: &Tensor,
    grad: &[f64],
    dim: usize,
    eps: f64,
    which: usize,
) -> Result<Tensor> {
    let (a, b, len, inner, kept) = cosine_operands(a, b, dim, "cosine_gradient_slices")?;
    if grad.len() != kept.iter().product::<usize>() {
        return Err(MinitensorError::internal_error(
            "cosine_gradient_slices: one incoming gradient a pair of slices",
        ));
    }
    let missing = || MinitensorError::internal_error("cosine_gradient_slices: a float slice");
    let mut data = TensorData::uninitialized_on_device(a.numel(), a.dtype(), a.device());
    if a.dtype() == DataType::Float32 {
        let (x, y) = (a.data().as_f32_slice(), b.data().as_f32_slice());
        let out = data.as_f32_slice_mut().ok_or_else(missing)?;
        let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
        gradient_pairs(x, y, grad, out, (len, inner), eps, which);
    } else {
        let (x, y) = (a.data().as_f64_slice(), b.data().as_f64_slice());
        let out = data.as_f64_slice_mut().ok_or_else(missing)?;
        let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
        gradient_pairs(x, y, grad, out, (len, inner), eps, which);
    }
    Ok(Tensor::new(
        Arc::new(data),
        a.shape().clone(),
        a.dtype(),
        a.device(),
        false,
    ))
}

/// The gradient of [`normalized_slices`]' quotients in their input `x` at an
/// incoming `grad` of its shape, each slice along `dim` taken by its flag in
/// `over`, laid out as the forward returned them: a slice taken over its
/// `p`-norm moves by `scale (g - d (x . g) / |x|) / |x|`, for `d` its
/// direction for the 2-norm, `x / |x|`, and its signs for the 1-norm, and any
/// other by `g / under` -- `scale` one and `under` the floor for `normalize`,
/// and the cap and one for `renorm`, each as the dtype holds it. A zero
/// slice taken over its norm, with no floor, moves by NaN, as its quotients
/// are NaN.
///
/// The difference is formed whole: for the 2-norm it is `g`'s part at right
/// angles to `x`, by `across`, and for the 1-norm `g` less the
/// signs times `x . g / |x|_1`, a pair. As `g - d (x . g) / |x|` in the dtype
/// it was two terms that all but cancel for an incoming gradient almost
/// along `d`. In float64, each slice scaled into range by a power of two,
/// and rounded once; an infinite incoming gradient is taken as
/// `unbounded_slope` says.
pub fn normalized_gradient_slices(
    x: &Tensor,
    grad: &Tensor,
    over: &Tensor,
    p: f64,
    dim: usize,
    (scale, under): (f64, f64),
) -> Result<Tensor> {
    let taxicab = match p {
        1.0 => true,
        2.0 => false,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "normalized_gradient_slices takes p = 1 or p = 2, got {p}"
            )));
        }
    };
    let (x, grad, len, inner, kept) = cosine_operands(x, grad, dim, "normalized_gradient_slices")?;
    let over = over.contiguous()?;
    let flags = over.data().as_bool_slice();
    let missing =
        || MinitensorError::internal_error("normalized_gradient_slices: one flag a slice");
    let flags = flags
        .filter(|f| f.len() == kept.iter().product::<usize>())
        .ok_or_else(missing)?;
    let mut data = TensorData::uninitialized_on_device(x.numel(), x.dtype(), x.device());
    let setting = (len, inner, taxicab, scale, under);
    if x.dtype() == DataType::Float32 {
        let (values, g) = (x.data().as_f32_slice(), grad.data().as_f32_slice());
        let out = data.as_f32_slice_mut().ok_or_else(missing)?;
        normalized_gradient_pairs(
            values.ok_or_else(missing)?,
            g.ok_or_else(missing)?,
            flags,
            out,
            setting,
        );
    } else {
        let (values, g) = (x.data().as_f64_slice(), grad.data().as_f64_slice());
        let out = data.as_f64_slice_mut().ok_or_else(missing)?;
        normalized_gradient_pairs(
            values.ok_or_else(missing)?,
            g.ok_or_else(missing)?,
            flags,
            out,
            setting,
        );
    }
    Ok(Tensor::new(
        Arc::new(data),
        x.shape().clone(),
        x.dtype(),
        x.device(),
        false,
    ))
}

/// [`normalized_gradient_slices`] of `x` and `grad` laid out
/// `[outer, len, inner]` into `out`, laid out as they are, `over` laid out
/// `[outer, inner]`.
fn normalized_gradient_pairs<T: Element + Output>(
    x: &[T],
    grad: &[T],
    over: &[bool],
    out: &mut [T],
    (len, inner, taxicab, scale, under): (usize, usize, bool, f64, f64),
) {
    let (scale, under) = (T::narrow(scale).widen(), T::narrow(under).widen());
    let block = len * inner;
    if block == 0 {
        return;
    }
    // Whole `[len, inner]` blocks to a task, enough of them to be worth one.
    let batch = (TASK / block).max(1);
    par_out_chunks_gated(out, batch * block, PAR_THRESHOLD / 8, &|start, o| {
        let mut room = vec![0.0f64; 3 * len];
        let (xs, rest) = room.split_at_mut(len);
        let (gs, slope) = rest.split_at_mut(len);
        for (j, o) in o.chunks_mut(block).enumerate() {
            let outer = start / block + j;
            let at = outer * block;
            for i in 0..inner {
                for k in 0..len {
                    xs[k] = x[at + i + k * inner].wide();
                    gs[k] = grad[at + i + k * inner].wide();
                }
                let over = over[outer * inner + i];
                normalized_slope(xs, gs, over, (taxicab, scale, under), slope);
                for (k, &value) in slope.iter().enumerate() {
                    o[i + k * inner] = T::narrow(value);
                }
            }
        }
    });
}

/// One slice of [`normalized_gradient_slices`], `x` and `g` widened to
/// float64 and scaled in place, into `out`.
#[inline(never)]
fn normalized_slope(
    x: &mut [f64],
    g: &mut [f64],
    over: bool,
    (taxicab, scale, under): (bool, f64, f64),
    out: &mut [f64],
) {
    if !over {
        for (slot, &value) in out.iter_mut().zip(g.iter()) {
            *slot = value / under;
        }
        return;
    }
    let Some(pair) = CosinePair::new(x, g, 0.0) else {
        unbounded_slope(x, g, scale, out);
        return;
    };
    let (x, g) = (&*x, &*g);
    let length = if taxicab {
        let dot = pair.dot;
        fused!(taxicab_part(x: &[f64], g: &[f64], dot: (f64, f64), out: &mut [f64]) -> f64)
    } else {
        across(x, g, pair.squares[0], pair.dot, out);
        pair.norms[0]
    };
    if length == 0.0 {
        // A zero slice with no floor: its quotients are zero over zero.
        out.fill(f64::NAN);
        return;
    }
    // The scaled slices give the gradient times `2^(shift_g - shift_x)`, and
    // the cap is taken as `m 2^e`, `m` in `[1, 2)`, so that no product on the
    // way to an entry leaves the range where the entry does not.
    let e = if scale.is_finite() && scale > 0.0 {
        binary_exponent(scale)
    } else {
        0
    };
    let factor = libm::scalbn(scale, -e) / length;
    times_scaled(out, factor, pair.shifts[0] - pair.shifts[1] + e);
}

/// Each of `out` times `factor 2^shift`, rounded once: by their product
/// where that is a normal float, and otherwise by `factor` and then the
/// power, so that nothing on the way leaves the range where the entry does
/// not.
fn times_scaled(out: &mut [f64], factor: f64, shift: i32) {
    let fold = libm::scalbn(factor, shift);
    if fold.is_normal() {
        out.iter_mut().for_each(|slot| *slot *= fold);
    } else {
        for slot in out.iter_mut() {
            *slot = libm::scalbn(*slot * factor, shift);
        }
    }
}

/// [`normalized_slope`] of a slice taken over its norm where `x` or `g` is
/// not finite: NaN where either holds a NaN, `x` an infinity or the cap is
/// zero, and otherwise -- `g` holding infinities -- each entry the sum of its
/// row of the slope times `g` as float sums take it: an infinity where the
/// infinite entries' terms share a sign, NaN where they do not or one meets
/// a zero of the slope. For either norm the slope is `-sign(x_j) sign(x_k)`
/// times a positive number off its diagonal, and positive on it while the
/// slice holds an entry that is not zero beside `x_j`.
#[inline(never)]
fn unbounded_slope(x: &[f64], g: &[f64], scale: f64, out: &mut [f64]) {
    let unknown = scale.is_nan() || scale <= 0.0;
    if unknown || x.iter().any(|v| !v.is_finite()) || g.iter().any(|v| v.is_nan()) {
        out.fill(f64::NAN);
        return;
    }
    let sign = |v: f64| i32::from(v > 0.0) - i32::from(v < 0.0);
    let nonzero = x.iter().filter(|&&v| v != 0.0).count();
    // The infinite entries of `g` by the sign of `x` times theirs: against,
    // beside a zero, along.
    let mut infinite = [0usize; 3];
    for (&v, &w) in x.iter().zip(g) {
        if w.is_infinite() {
            infinite[(sign(v) * sign(w) + 1) as usize] += 1;
        }
    }
    for ((slot, &v), &w) in out.iter_mut().zip(x).zip(g) {
        let s = sign(v);
        let mut others = infinite;
        if w.is_infinite() {
            others[(s * sign(w) + 1) as usize] -= 1;
        }
        let [against, beside, along] = others;
        // Each other infinite entry's term has the sign `-s` along `x` and
        // `s` against it, and meets a zero beside a zero of `x`.
        let mut nan = beside > 0 || (s == 0 && against + along > 0);
        let (mut up, mut down) = (
            (along > 0 && s < 0) || (against > 0 && s > 0),
            (along > 0 && s > 0) || (against > 0 && s < 0),
        );
        if w.is_infinite() {
            if nonzero > usize::from(v != 0.0) {
                up |= w > 0.0;
                down |= w < 0.0;
            } else {
                nan = true;
            }
        }
        *slot = if nan || (up && down) {
            f64::NAN
        } else if up {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
}

/// `g - sign(x) (x . g) / |x|_1` into `out`, for `dot` the pair `x . g`, and
/// the 1-norm `|x|_1`: the quotient a pair, so that where `g` lies almost
/// along the signs each entry is formed whole.
#[inline(always)]
fn taxicab_part<const FMA: bool>(
    x: &[f64],
    g: &[f64],
    (dot, dot_lo): (f64, f64),
    out: &mut [f64],
) -> f64 {
    let (mut sum, mut sum_lo) = (0.0f64, 0.0f64);
    for &value in x {
        let (total, error) = two_sum(sum, value.abs());
        (sum, sum_lo) = (total, sum_lo + error);
    }
    let (sum, sum_lo) = two_sum(sum, sum_lo);
    let (multiple, multiple_lo) = if sum > 0.0 {
        let q = dot / sum;
        let p = q * sum;
        let remainder = (dot - p) - error_of::<FMA>(q, sum, p);
        (q, (remainder + dot_lo - q * sum_lo) / sum)
    } else {
        (0.0, 0.0)
    };
    for ((slot, &value), &w) in out.iter_mut().zip(x).zip(g) {
        let sign = if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        };
        *slot = (w - sign * multiple) - sign * multiple_lo;
    }
    sum
}

/// Two tensors of one shape and float dtype made contiguous, and the
/// length of `dim`, the extent after it and the shape without it.
fn cosine_operands(
    a: &Tensor,
    b: &Tensor,
    dim: usize,
    name: &str,
) -> Result<(Tensor, Tensor, usize, usize, Vec<usize>)> {
    check(a, b, 2.0, name)?;
    if a.shape() != b.shape() || dim >= a.ndim() {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} takes two tensors of one shape and an axis of it, got {:?}, {:?} and \
             axis {dim}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let dims = a.shape().dims();
    let (len, inner) = (dims[dim], dims[dim + 1..].iter().product::<usize>());
    let mut kept = dims.to_vec();
    kept.remove(dim);
    Ok((a, b, len, inner, kept))
}

/// [`cosine_embedding_slices`]' terms of `a` and `b` laid out
/// `[outer, len, inner]`, against `target`, into `out`, both laid out
/// `[outer, inner]`.
fn embedding_pairs<T: Element + Output>(
    a: &[T],
    b: &[T],
    target: &[f64],
    out: &mut [f64],
    (len, inner): (usize, usize),
    (eps, margin): (f64, f64),
) {
    let floor = T::narrow(eps).widen();
    let per_task = (TASK / len.max(1)).max(1);
    // Gated on the entries each pair reads, not the one value it writes.
    let gate = (PAR_THRESHOLD / 8).div_ceil(len.max(1));
    par_out_chunks_gated(out, per_task, gate, &|start, o| {
        let mut room = vec![0.0f64; 3 * len];
        let (x, rest) = room.split_at_mut(len);
        let (y, scratch) = rest.split_at_mut(len);
        for (i, slot) in o.iter_mut().enumerate() {
            let slice = start + i;
            let base = slice / inner * len * inner + slice % inner;
            for k in 0..len {
                x[k] = a[base + k * inner].wide();
                y[k] = b[base + k * inner].wide();
            }
            *slot = match CosinePair::new(x, y, floor) {
                Some(pair) if target[slice] == 1.0 => pair.complement(x, y, scratch),
                Some(pair) => {
                    let over = pair.cosine() - margin;
                    if over > 0.0 || over.is_nan() {
                        over
                    } else {
                        0.0
                    }
                }
                None => f64::NAN,
            };
        }
    });
}

/// [`cosine_gradient_slices`] of `a` and `b` laid out `[outer, len, inner]`
/// into `out`, laid out as they are, `grad` laid out `[outer, inner]`.
fn gradient_pairs<T: Element + Output>(
    a: &[T],
    b: &[T],
    grad: &[f64],
    out: &mut [T],
    (len, inner): (usize, usize),
    eps: f64,
    which: usize,
) {
    let floor = T::narrow(eps).widen();
    let block = len * inner;
    if block == 0 {
        return;
    }
    // Whole `[len, inner]` blocks to a task, enough of them to be worth one.
    let batch = (TASK / block).max(1);
    par_out_chunks_gated(out, batch * block, PAR_THRESHOLD / 8, &|start, o| {
        let mut room = vec![0.0f64; 3 * len];
        let (x, rest) = room.split_at_mut(len);
        let (y, slope) = rest.split_at_mut(len);
        for (j, o) in o.chunks_mut(block).enumerate() {
            let outer = start / block + j;
            let at = outer * block;
            for i in 0..inner {
                let g = grad[outer * inner + i];
                if g == 0.0 {
                    // Nothing comes back through this pair: a flat term's.
                    for k in 0..len {
                        o[i + k * inner] = T::narrow(0.0);
                    }
                    continue;
                }
                for k in 0..len {
                    x[k] = a[at + i + k * inner].wide();
                    y[k] = b[at + i + k * inner].wide();
                }
                match CosinePair::new(x, y, floor) {
                    Some(pair) => pair.slope(which, x, y, g, slope),
                    None => slope.fill(f64::NAN),
                }
                for (k, &value) in slope.iter().enumerate() {
                    o[i + k * inner] = T::narrow(value);
                }
            }
        }
    });
}

/// A pair of slices' sums for their cosine, in float64, each slice scaled
/// by a power of two that keeps its sum of squares inside the range: each
/// norm, and each floored at the floor as it is scaled, the dot and the
/// sums of squares. Their cosine, complement and gradients are formed from
/// the scaled slices and these.
struct CosinePair {
    shifts: [i32; 2],
    /// The sums of squares and the dot, each as a pair `hi + lo`.
    squares: [(f64, f64); 2],
    dot: (f64, f64),
    norms: [f64; 2],
    lengths: [f64; 2],
    /// Whether each norm is above its floor, and so moves the cosine's
    /// denominator.
    moving: [bool; 2],
}

impl CosinePair {
    /// The pair's sums, `a` and `b` scaled in place; `None` where either
    /// holds a NaN or an infinity, whose cosine is NaN.
    #[inline(never)]
    fn new(a: &mut [f64], b: &mut [f64], floor: f64) -> Option<Self> {
        let shifts = [into_range(a)?, into_range(b)?];
        let (a, b) = (&*a, &*b);
        let [squares_a, squares_b, dot] =
            fused!(pair_sums(a: &[f64], b: &[f64]) -> [(f64, f64); 3]);
        let squares = [squares_a, squares_b];
        let norms = [squares[0].0.sqrt(), squares[1].0.sqrt()];
        let floors = [
            times_power_of_two(floor, shifts[0]),
            times_power_of_two(floor, shifts[1]),
        ];
        Some(Self {
            shifts,
            squares,
            dot,
            norms,
            lengths: [norms[0].max(floors[0]), norms[1].max(floors[1])],
            moving: [norms[0] > floors[0], norms[1] > floors[1]],
        })
    }

    /// The cosine, of the pair's sums: within an ulp or two of float64.
    fn cosine(&self) -> f64 {
        let ([na, nb], [la, lb]) = (self.norms, self.lengths);
        if na == 0.0 || nb == 0.0 {
            return if la > 0.0 && lb > 0.0 { 0.0 } else { f64::NAN };
        }
        (self.dot.0 + self.dot.1) / (la * lb)
    }

    /// `1 - cos`; see [`cosine_embedding_slices`].
    #[inline(never)]
    fn complement(&self, a: &[f64], b: &[f64], scratch: &mut [f64]) -> f64 {
        let ([na, nb], [la, lb]) = (self.norms, self.lengths);
        if na == 0.0 || nb == 0.0 {
            // A zero dot: the cosine is zero over a floored norm, and NaN
            // over none.
            return if la > 0.0 && lb > 0.0 { 1.0 } else { f64::NAN };
        }
        let cosine = self.dot.0 / (na * nb);
        let share = (na / la) * (nb / lb);
        if cosine <= 0.0 {
            return 1.0 - share * cosine;
        }
        let across = across(a, b, self.squares[0], self.dot, scratch);
        let rest = across / (self.squares[1].0 * (1.0 + cosine));
        if share == 1.0 {
            rest
        } else {
            (1.0 - share) + share * rest
        }
    }

    /// `g` times the cosine's gradient in slice `which` of `(a, b)`, into
    /// `out`; see [`cosine_gradient_slices`]. The scaled slices give it as
    /// the other's part at right angles to this one over the scaled
    /// lengths, times the power of two this one was scaled by. `g` is taken
    /// as `m 2^e`, `m` in `[1, 2)`, so that no product on the way to an
    /// entry leaves the range where the entry does not.
    #[inline(never)]
    fn slope(&self, which: usize, a: &[f64], b: &[f64], g: f64, out: &mut [f64]) {
        let (this, other) = if which == 0 { (a, b) } else { (b, a) };
        if self.moving[which] {
            across(this, other, self.squares[which], self.dot, out);
        } else {
            out.copy_from_slice(other);
        }
        let e = if g.is_finite() && g != 0.0 {
            binary_exponent(g.abs())
        } else {
            0
        };
        let factor = libm::scalbn(g, -e) / (self.lengths[0] * self.lengths[1]);
        times_scaled(out, factor, self.shifts[which] + e);
    }
}

/// `other - (dot / squares) this`, the part of `other` at right angles to
/// `this` for `dot` their dot and `squares` `this`'s sum of squares, each a
/// pair `hi + lo`, into `out`, and the sum of its squares.
///
/// The multiple is a pair too, and each entry is the difference with the
/// product's rounding and the multiple's low part put back, so that where
/// `other` lies almost along `this` its small difference is formed whole --
/// a rounding of its own size -- rather than as what two roundings the size
/// of `other` leave of it. What the multiple's error leaves along `this` is
/// some `2^-104` of `other`, and moves the sum of squares, at right angles to
/// it, by the square of that.
#[inline(never)]
fn across(
    this: &[f64],
    other: &[f64],
    squares: (f64, f64),
    dot: (f64, f64),
    out: &mut [f64],
) -> f64 {
    fused!(
        orthogonal_part(
            this: &[f64],
            other: &[f64],
            squares: (f64, f64),
            dot: (f64, f64),
            out: &mut [f64],
        ) -> f64
    )
}

/// [`across`], the products' errors by a fused multiply-add where `FMA`.
#[inline(always)]
fn orthogonal_part<const FMA: bool>(
    this: &[f64],
    other: &[f64],
    (squares, squares_lo): (f64, f64),
    (dot, dot_lo): (f64, f64),
    out: &mut [f64],
) -> f64 {
    let (multiple, multiple_lo) = if squares > 0.0 {
        // The quotient's float, and what it leaves of the whole over the
        // squares: the remainder of a float division is a float.
        let q = dot / squares;
        let p = q * squares;
        let remainder = (dot - p) - error_of::<FMA>(q, squares, p);
        (q, (remainder + dot_lo - q * squares_lo) / squares)
    } else {
        (0.0, 0.0)
    };
    let mut total = CarriedSum::default();
    for ((slot, &x), &y) in out.iter_mut().zip(this).zip(other) {
        let p = multiple * x;
        *slot = (y - p) - (error_of::<FMA>(multiple, x, p) + multiple_lo * x);
        total.add(*slot * *slot);
    }
    total.value()
}

/// The sums of `a`'s squares, `b`'s and their products, in one pass and each
/// as a pair `hi + lo`: every addition's rounding and every product's carried
/// beside it.
#[inline(always)]
fn pair_sums<const FMA: bool>(a: &[f64], b: &[f64]) -> [(f64, f64); 3] {
    let mut sums = [(0.0f64, 0.0f64); 3];
    for (&x, &y) in a.iter().zip(b) {
        for (sum, (p, q)) in sums.iter_mut().zip([(x, x), (y, y), (x, y)]) {
            let product = p * q;
            let (hi, error) = two_sum(sum.0, product);
            *sum = (hi, sum.1 + (error + error_of::<FMA>(p, q, product)));
        }
    }
    sums.map(|(hi, lo)| two_sum(hi, lo))
}

/// Scales `values` in place by the power of two that brings its largest
/// magnitude into `[1, 2)` where that lies outside `[2^-400, 2^400]` -- where
/// its sum of squares could overflow or fall below the normal range -- and
/// returns the power; `None` where it holds a NaN or an infinity.
fn into_range(values: &mut [f64]) -> Option<i32> {
    let mut largest = 0.0f64;
    for &value in values.iter() {
        if !value.is_finite() {
            return None;
        }
        largest = largest.max(value.abs());
    }
    if largest == 0.0 || (2f64.powi(-400)..=2f64.powi(400)).contains(&largest) {
        return Some(0);
    }
    let shift = -binary_exponent(largest);
    let [low, high] = powers_of_two(shift);
    for value in values.iter_mut() {
        *value = *value * low * high;
    }
    Some(shift)
}

/// The exponent `e` of a positive, finite `x`, `2^e <= x < 2^(e + 1)`.
fn binary_exponent(x: f64) -> i32 {
    if x < f64::MIN_POSITIVE {
        return binary_exponent(x * 2f64.powi(64)) - 64;
    }
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// Two powers of two whose product is `2^k`, each a float for any `k` a
/// scaling here asks for: multiplying by both is exact wherever the result
/// is a normal float.
fn powers_of_two(k: i32) -> [f64; 2] {
    let half = k / 2;
    [2f64.powi(half), 2f64.powi(k - half)]
}

/// `x 2^k`; see [`powers_of_two`].
fn times_power_of_two(x: f64, k: i32) -> f64 {
    let [low, high] = powers_of_two(k);
    x * low * high
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
