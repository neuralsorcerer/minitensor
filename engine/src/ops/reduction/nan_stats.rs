// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The NaN-skipping reductions that are arrangements of the others.
//!
//! `nansum`, `nanmean`, `nanmax` and the rest each carry a kernel that walks
//! the buffer once, testing as it goes. These need no kernel of their own: a
//! variance is a mean of squared deviations, and a product with NaN skipped is
//! a product with NaN replaced by the identity. Writing them as those
//! arrangements is one definition rather than two, and it is what makes their
//! gradients the gradients of the ops underneath.
//!
//! `nanargmax` and `nanargmin` used to be here on the same grounds -- an index
//! of the largest non-NaN read as an index of the largest once NaN had been
//! pushed to the bottom -- and they are the counter-example. The substitution
//! was six passes of setup around a one-pass reduction, and it could not tell
//! a NaN pushed to `-inf` from an `-inf` that was there all along. They now
//! ask the NaN-skipping extremum directly, which is both the faster and the
//! only correct way to spell it.

use super::core_impl::normalize_reduction_dims;
use super::sum_prod_impl::{fold_lanes, fold_slab_with};
use crate::ops::map::{outputs_per_task, par_map_indexed};
use crate::{
    error::{MinitensorError, Result},
    ops::{
        activation::sqrt,
        arithmetic::{div, mul, sub},
        comparison::eq,
        minmax::maximum,
        reduction::{
            any, argmax, argmin, checked_reduction_dim, count_nonzero, nanargmax_all,
            nanargmin_all, nanmax_along_dim_with_indices, nanmean, nanmin_along_dim_with_indices,
            prod, sum,
        },
        selection::where_op,
        util::create_scalar_tensor,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::Float;
use std::sync::Arc;

/// How many entries along `dim` are not NaN, as a float ready to divide by.
///
/// `count_nonzero` counts the mask, which is what `nanmean` divides by too;
/// this is the same count in the dtype the division needs.
fn non_nan_count(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    let finite = eq(&tensor.isnan()?, &boolean_false(tensor)?)?;
    count_nonzero(&finite, dim, keepdim)?.astype(tensor.dtype())
}

/// A `false` to compare a NaN mask against, so "is not NaN" needs no operator
/// of its own.
fn boolean_false(like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::zeros(
        Shape::scalar(),
        DataType::Bool,
        like.device(),
        false,
    ))
}

/// What [`nan_moments_fused`] is asked for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NanMoment {
    Mean,
    /// The variance, divided by the count less this correction.
    Var {
        correction: usize,
    },
}

/// `nanmean` or `nanvar` over one axis, or over everything, for a tensor that
/// does not require gradients; `None` when the call is not one of those, and
/// the composition answers instead.
///
/// The compositions are the definitions -- and what gradients go through --
/// but as kernels they are a dozen full-size passes: `nanvar` built a NaN
/// mask, a boolean negation of it, a float copy of that to count, the
/// NaN-skipping mean (which built its own mask and float copy again), a
/// difference, a masked select and a square, each a new tensor the size of
/// the input. Here it is three reads of the input and no full-size
/// intermediate: the non-NaN sums, the non-NaN counts, and for the variance
/// the squared deviations from their quotient, each through the blocked
/// column fold `sum` uses, so the accuracy is `sum`'s as well.
///
/// The arithmetic after the passes is the composition's: the mean is
/// `sum / count` in the tensor's dtype, and the variance divides by
/// `max(count - correction, 0)`, so an all-NaN slice is `0 / 0` and a slice
/// with one value and a correction of one is `0 / 0` too. A NaN is skipped
/// wherever it is; an infinity is not, and its deviation `inf - inf` makes
/// the variance NaN, again as before.
pub(crate) fn nan_moments_fused(
    tensor: &Tensor,
    dims: Option<&[usize]>,
    keepdim: bool,
    moment: NanMoment,
) -> Result<Option<Tensor>> {
    if tensor.requires_grad() || !tensor.dtype().is_float() {
        return Ok(None);
    }
    let shape = tensor.shape().dims();
    let (outer, len, inner, out_shape) = match dims {
        None => (
            1,
            tensor.numel(),
            1,
            if keepdim {
                Shape::new(vec![1; shape.len()])
            } else {
                Shape::scalar()
            },
        ),
        Some(&[axis]) => {
            let mut kept = shape.to_vec();
            if keepdim {
                kept[axis] = 1;
            } else {
                kept.remove(axis);
            }
            (
                shape[..axis].iter().product(),
                shape[axis],
                shape[axis + 1..].iter().product(),
                if kept.is_empty() {
                    Shape::scalar()
                } else {
                    Shape::new(kept)
                },
            )
        }
        Some(_) => return Ok(None),
    };
    // The counts are `u32` lanes; an axis that long falls back rather than
    // risk them wrapping.
    if len > u32::MAX as usize {
        return Ok(None);
    }

    let data = match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            TensorData::from_vec_f32(
                nan_moments(input, outer, len, inner, moment),
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            TensorData::from_vec_f64(
                nan_moments(input, outer, len, inner, moment),
                tensor.device(),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(Tensor::new(
        Arc::new(data),
        out_shape,
        tensor.dtype(),
        tensor.device(),
        false,
    )))
}

/// [`NanMoment`] of every column of `outer` row-major `(len, inner)` slabs.
///
/// Few slabs are each spread across the pool and many are handed out whole,
/// the split `sum` makes for the same shape and for the same reason: which
/// one happens follows from the shape alone, so the thread count cannot
/// reach the answer.
fn nan_moments<T>(input: &[T], outer: usize, len: usize, inner: usize, moment: NanMoment) -> Vec<T>
where
    T: Float + Send + Sync,
{
    if outer == 0 || inner == 0 {
        return Vec::new();
    }
    let slab = len * inner;
    let one = |index: usize, spread: bool| {
        slab_moments(
            &input[index * slab..(index + 1) * slab],
            inner,
            spread,
            moment,
        )
    };
    if outer < NAN_MOMENT_MIN_SLABS {
        return (0..outer).flat_map(|index| one(index, true)).collect();
    }
    let per_task = outputs_per_task(slab.max(1)).div_ceil(inner).max(1);
    par_map_indexed(outer.div_ceil(per_task), &|task| {
        let first = task * per_task;
        (first..(first + per_task).min(outer))
            .flat_map(|index| one(index, false))
            .collect::<Vec<T>>()
    })
    .concat()
}

/// Fewer slabs than this are each spread across the pool; `sum`'s threshold.
const NAN_MOMENT_MIN_SLABS: usize = 4;

/// [`NanMoment`] of each column of one row-major `(rows, cols)` slab.
fn slab_moments<T>(slab: &[T], cols: usize, spread: bool, moment: NanMoment) -> Vec<T>
where
    T: Float + Send + Sync,
{
    let add = |x: T, y: T| x + y;
    let sums = fold_slab_with(
        slab,
        cols,
        T::zero(),
        |acc: &mut [T], values: &[T], _| {
            for (a, &v) in acc.iter_mut().zip(values) {
                *a = *a + if v.is_nan() { T::zero() } else { v };
            }
        },
        add,
        spread,
    );
    let counts = fold_slab_with(
        slab,
        cols,
        0u32,
        |acc: &mut [u32], values: &[T], _| {
            for (a, &v) in acc.iter_mut().zip(values) {
                *a += u32::from(!v.is_nan());
            }
        },
        |x, y| x + y,
        spread,
    );
    let count = |c: u32| T::from(c).unwrap_or_else(T::nan);
    let means: Vec<T> = sums
        .iter()
        .zip(&counts)
        .map(|(&s, &c)| s / count(c))
        .collect();
    let NanMoment::Var { correction } = moment else {
        return means;
    };

    let tiled: Vec<T> = means
        .iter()
        .copied()
        .cycle()
        .take(fold_lanes(cols).max(cols))
        .collect();
    let squares = fold_slab_with(
        slab,
        cols,
        T::zero(),
        |acc: &mut [T], values: &[T], first| {
            let means = &tiled[first..first + acc.len()];
            for ((a, &v), &m) in acc.iter_mut().zip(values).zip(means) {
                let d = v - m;
                *a = *a + if v.is_nan() { T::zero() } else { d * d };
            }
        },
        add,
        spread,
    );
    squares
        .iter()
        .zip(&counts)
        .map(|(&q, &c)| q / count(c.saturating_sub(correction.min(u32::MAX as usize) as u32)))
        .collect()
}

/// Variance over the non-NaN entries along `dim`.
///
/// A slice with fewer non-NaN entries than the correction demands has no
/// variance to report, and the division says so: `0 / 0` is NaN and `x / 0` is
/// infinity, which is what NumPy gives for the same slices.
pub fn nanvar(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    // An integer holds no NaN, so this is `var` -- the same delegation
    // `nanmax` and `nanprod` make, rather than a refusal.
    if !tensor.dtype().is_float() {
        return crate::ops::reduction::var(tensor, dim, keepdim, unbiased);
    }
    // An empty `dim` reduces nothing; the composition already answers that.
    let dims = normalize_reduction_dims(dim.clone(), tensor.ndim())?;
    if dims.as_ref().is_none_or(|dims| !dims.is_empty())
        && let Some(fused) = nan_moments_fused(
            tensor,
            dims.as_deref(),
            keepdim,
            NanMoment::Var {
                correction: usize::from(unbiased),
            },
        )?
    {
        return Ok(fused);
    }

    // Centred on the NaN-skipping mean, kept broadcastable.
    //
    // The deviation is zeroed at the NaN positions *before* it is squared,
    // rather than the NaN being carried through and dropped by `nansum`
    // afterwards. Both give the same total, but only this one has a gradient:
    // `nansum` hands back a zero for a skipped entry, and the square's chain
    // rule then computes `0 * 2 * NaN`, which is NaN and spreads through the
    // shared mean to every finite entry in the slice.
    let centre = nanmean(tensor, dim.clone(), true)?;
    let deviation = sub(tensor, &centre)?;
    let zero = create_scalar_tensor(0.0, tensor.dtype(), tensor.device())?;
    let finite_deviation = where_op(&tensor.isnan()?, &zero, &deviation)?;
    let squared = mul(&finite_deviation, &finite_deviation)?;

    let total = sum(&squared, dim.clone(), keepdim)?;
    let count = non_nan_count(tensor, dim, keepdim)?;
    let divisor = if unbiased {
        // Clamped at zero, because a slice with nothing in it has no variance
        // to report and the division is how this says so. `count - 1` is -1
        // for a slice that is all NaN, and `0 / -1` is a clean -0.0 -- an
        // answer claiming the data does not vary, from data that is not
        // there. At zero the division is `0 / 0`, which is NaN: what `var`
        // answers for an empty slice, and what NumPy answers for this one.
        let corrected = sub(
            &count,
            &create_scalar_tensor(1.0, tensor.dtype(), tensor.device())?,
        )?;
        maximum(
            &corrected,
            &create_scalar_tensor(0.0, tensor.dtype(), tensor.device())?,
        )?
    } else {
        count
    };

    div(&total, &divisor)
}

/// Standard deviation over the non-NaN entries: the square root of [`nanvar`].
pub fn nanstd(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    sqrt(&nanvar(tensor, dim, keepdim, unbiased)?)
}

/// Product over the non-NaN entries, taking an all-NaN slice as 1.
///
/// Replacing NaN with the multiplicative identity is the same thing as
/// skipping it, and it is what makes an empty product 1 rather than NaN --
/// which is the convention `prod` already follows for a genuinely empty slice.
pub fn nanprod(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return prod(tensor, dim, keepdim);
    }
    // Only the NaN is replaced. `nan_to_num` with its defaults also replaces
    // the infinities -- with the dtype's finite extremes, which is what that
    // function is for -- so `nanprod([nan, inf])` came back as 1.8e308 rather
    // than `inf`, a finite number standing where an infinite one was.
    prod(&without_nan_replaced_by(tensor, 1.0)?, dim, keepdim)
}

/// The tensor with every NaN replaced by `replacement` and everything else,
/// infinities included, left alone.
fn without_nan_replaced_by(tensor: &Tensor, replacement: f64) -> Result<Tensor> {
    let filler = create_scalar_tensor(replacement, tensor.dtype(), tensor.device())?;
    where_op(&tensor.isnan()?, &filler, tensor)
}

/// Rejects a reduction that would have to name an index among NaN alone.
///
/// `values` is what the NaN-skipping extremum reported for each slice, which
/// is NaN exactly when the slice held nothing else -- the reduction has
/// already looked, so this only has to read its answer. That answer is
/// `dim_size` times smaller than the input a separate count would re-walk.
fn reject_all_nan_slices(values: &Tensor, name: &str) -> Result<()> {
    let empty = any(&values.isnan()?, None, false)?;
    let is_empty = empty
        .data()
        .as_bool_slice()
        .and_then(|slice| slice.first().copied())
        .unwrap_or(false);
    if is_empty {
        return Err(MinitensorError::invalid_operation(format!(
            "{name}: a slice of all-NaN values has no index to report"
        )));
    }
    Ok(())
}

/// Index of the largest non-NaN entry along `dim`.
///
/// Both halves ask a NaN-skipping extremum that was already there. Over the
/// whole tensor the lane fold skips NaN by itself -- no comparison against one
/// is true -- and along a `dim` the indexed reduction seeds with NaN and only
/// moves off it for a real value, which is the same rule.
///
/// This was `argmax(where(isnan(x), -inf, x))` behind an all-NaN check built
/// from a second full-size count: seven passes and two full-size temporaries
/// where the reduction itself is one pass. It also answered differently. The
/// substitution cannot tell a NaN that became `-inf` from an `-inf` that was
/// always there, so `nanargmax([nan, -inf])` named index 0 -- a NaN, from the
/// reduction whose whole job is to skip them. NumPy does the same thing for
/// the same reason. Skipping instead of substituting answers 1, and is the
/// only one of the two that honours the name.
pub fn nanargmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return argmax(tensor, dim, keepdim);
    }
    match checked_reduction_dim(tensor, dim, "nanargmax")? {
        None => nanargmax_all(tensor, keepdim),
        Some(d) => {
            let (values, indices) = nanmax_along_dim_with_indices(tensor, d, keepdim)?;
            reject_all_nan_slices(&values, "nanargmax")?;
            Ok(indices)
        }
    }
}

/// Index of the smallest non-NaN entry along `dim`. See [`nanargmax`].
pub fn nanargmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return argmin(tensor, dim, keepdim);
    }
    match checked_reduction_dim(tensor, dim, "nanargmin")? {
        None => nanargmin_all(tensor, keepdim),
        Some(d) => {
            let (values, indices) = nanmin_along_dim_with_indices(tensor, d, keepdim)?;
            reject_all_nan_slices(&values, "nanargmin")?;
            Ok(indices)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autograd::backward_collect, device::Device, tensor::Shape, tensor::TensorData};
    use std::sync::Arc;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(shape),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn wide(t: &Tensor) -> Vec<f64> {
        t.contiguous()
            .unwrap()
            .data()
            .as_f64_slice()
            .unwrap()
            .to_vec()
    }

    fn indices(t: &Tensor) -> Vec<i64> {
        t.contiguous()
            .unwrap()
            .data()
            .as_i64_slice()
            .unwrap()
            .to_vec()
    }

    const NAN: f64 = f64::NAN;

    /// Mean and variance of the finite entries, computed the long way.
    fn reference_var(values: &[f64], unbiased: bool) -> f64 {
        let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        let count = finite.len() as f64;
        let mean = finite.iter().sum::<f64>() / count;
        let total: f64 = finite.iter().map(|v| (v - mean) * (v - mean)).sum();
        total / (count - if unbiased { 1.0 } else { 0.0 })
    }

    #[test]
    fn nanvar_and_nanstd_ignore_nan() {
        let values = vec![1.0, NAN, 3.0, 5.0, NAN, 9.0];
        let t = tensor(values.clone(), vec![6]);

        for unbiased in [true, false] {
            let want = reference_var(&values, unbiased);
            let got = wide(&nanvar(&t, None, false, unbiased).unwrap())[0];
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");

            let got = wide(&nanstd(&t, None, false, unbiased).unwrap())[0];
            assert!((got - want.sqrt()).abs() < 1e-12);
        }
    }

    #[test]
    fn nanvar_along_a_dim_matches_the_row_by_row_answer() {
        let rows = [[1.0, NAN, 3.0], [2.0, 4.0, 6.0]];
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let t = tensor(flat, vec![2, 3]);

        let got = wide(&nanvar(&t, Some(vec![1]), false, true).unwrap());
        for (index, row) in rows.iter().enumerate() {
            let want = reference_var(row, true);
            assert!((got[index] - want).abs() < 1e-12, "row {index}");
        }

        // `keepdim` keeps the axis rather than dropping it.
        let kept = nanvar(&t, Some(vec![1]), true, true).unwrap();
        assert_eq!(kept.shape().dims(), &[2, 1]);
    }

    #[test]
    fn a_slice_with_too_few_finite_entries_reports_no_variance() {
        // One finite value and an unbiased correction leaves `0 / 0`; none at
        // all leaves it whatever the numerator is over zero. Both are the
        // answers NumPy gives, and neither is a number.
        let one = tensor(vec![NAN, 4.0, NAN], vec![3]);
        assert!(wide(&nanvar(&one, None, false, true).unwrap())[0].is_nan());
        assert_eq!(wide(&nanvar(&one, None, false, false).unwrap())[0], 0.0);

        let none = tensor(vec![NAN, NAN], vec![2]);
        assert!(wide(&nanvar(&none, None, false, false).unwrap())[0].is_nan());
    }

    #[test]
    fn nanprod_skips_nan_and_takes_an_all_nan_slice_as_one() {
        let t = tensor(vec![2.0, NAN, 3.0, NAN, 4.0], vec![5]);
        assert_eq!(wide(&nanprod(&t, None, false).unwrap())[0], 24.0);

        let all_nan = tensor(vec![NAN, NAN], vec![2]);
        assert_eq!(wide(&nanprod(&all_nan, None, false).unwrap())[0], 1.0);

        let rows = tensor(vec![2.0, NAN, 3.0, 4.0], vec![2, 2]);
        assert_eq!(
            wide(&nanprod(&rows, Some(vec![1]), false).unwrap()),
            vec![2.0, 12.0]
        );
    }

    #[test]
    fn nanargmax_and_nanargmin_skip_nan() {
        // The extreme sits next to a NaN in both directions, so a reduction
        // that let NaN win or lose would be visible.
        let t = tensor(vec![1.0, NAN, 9.0, NAN, -4.0], vec![5]);
        assert_eq!(indices(&nanargmax(&t, None, false).unwrap()), vec![2]);
        assert_eq!(indices(&nanargmin(&t, None, false).unwrap()), vec![4]);

        let rows = tensor(vec![NAN, 1.0, 2.0, 7.0, NAN, 3.0], vec![2, 3]);
        assert_eq!(
            indices(&nanargmax(&rows, Some(1), false).unwrap()),
            vec![2, 0]
        );
        assert_eq!(
            indices(&nanargmin(&rows, Some(1), false).unwrap()),
            vec![1, 2]
        );
    }

    #[test]
    fn nanargmax_still_reports_an_infinity_that_was_really_there() {
        // NaN is pushed to negative infinity to get it out of the way, so an
        // actual -inf in the data must still be findable.
        let t = tensor(vec![f64::NEG_INFINITY, NAN], vec![2]);
        assert_eq!(indices(&nanargmax(&t, None, false).unwrap()), vec![0]);

        let t = tensor(vec![f64::INFINITY, 1.0, NAN], vec![3]);
        assert_eq!(indices(&nanargmax(&t, None, false).unwrap()), vec![0]);
        assert_eq!(indices(&nanargmin(&t, None, false).unwrap()), vec![1]);
    }

    #[test]
    fn a_nan_is_never_the_index_reported() {
        // The substitution this used to do pushed NaN to -inf, which made it
        // indistinguishable from an -inf that was there: `nanargmax` then
        // named index 0, a NaN. NumPy still does. Skipping answers 1, the
        // first index that holds a number.
        let t = tensor(
            vec![NAN, f64::NEG_INFINITY, NAN, f64::NEG_INFINITY],
            vec![4],
        );
        assert_eq!(indices(&nanargmax(&t, None, false).unwrap()), vec![1]);

        let t = tensor(vec![NAN, f64::INFINITY, NAN, f64::INFINITY], vec![4]);
        assert_eq!(indices(&nanargmin(&t, None, false).unwrap()), vec![1]);

        // The same data reduced along a dim has to agree with it: one library
        // giving two answers for one question is worse than either answer.
        let rows = tensor(
            vec![NAN, f64::NEG_INFINITY, NAN, f64::NEG_INFINITY],
            vec![2, 2],
        );
        assert_eq!(
            indices(&nanargmax(&rows, Some(1), false).unwrap()),
            vec![1, 1]
        );
        let rows = tensor(vec![NAN, f64::INFINITY, NAN, f64::INFINITY], vec![2, 2]);
        assert_eq!(
            indices(&nanargmin(&rows, Some(1), false).unwrap()),
            vec![1, 1]
        );
    }

    #[test]
    fn an_index_among_nothing_but_nan_is_refused() {
        let all_nan = tensor(vec![NAN, NAN], vec![2]);
        assert!(nanargmax(&all_nan, None, false).is_err());
        assert!(nanargmin(&all_nan, None, false).is_err());

        // One all-NaN row among good ones is enough to refuse.
        let rows = tensor(vec![1.0, 2.0, NAN, NAN], vec![2, 2]);
        assert!(nanargmax(&rows, Some(1), false).is_err());
        // ...but the same data reduced the other way has a finite entry in
        // every slice, so it answers.
        assert_eq!(
            indices(&nanargmax(&rows, Some(0), false).unwrap()),
            vec![0, 0]
        );
    }

    #[test]
    fn gradients_reach_the_finite_entries_and_stop_at_the_nan() {
        let values = vec![1.0, NAN, 3.0, 5.0];
        let t = tensor(values, vec![4]).requires_grad_(true);
        let out = nanvar(&t, None, false, true).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grad = wide(
            backward_collect(&out, Some(seed))
                .unwrap()
                .get(&t.id())
                .unwrap(),
        );

        // The finite entries carry the gradient of a variance: 2(x - mean)/(n-1).
        let mean = (1.0 + 3.0 + 5.0) / 3.0;
        for (index, value) in [(0usize, 1.0), (2, 3.0), (3, 5.0)] {
            let want = 2.0 * (value - mean) / 2.0;
            assert!(
                (grad[index] - want).abs() < 1e-9,
                "index {index}: {} vs {want}",
                grad[index]
            );
        }
        assert_eq!(grad[1], 0.0, "the NaN entry must not pull the fit");
    }

    #[test]
    fn nanprod_carries_a_gradient_past_its_nan() {
        let t = tensor(vec![2.0, NAN, 3.0], vec![3]).requires_grad_(true);
        let out = nanprod(&t, None, false).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grad = wide(
            backward_collect(&out, Some(seed))
                .unwrap()
                .get(&t.id())
                .unwrap(),
        );
        // d(2 * 1 * 3)/d2 = 3 and /d3 = 2; the NaN contributed the constant 1.
        assert_eq!(grad[0], 3.0);
        assert_eq!(grad[1], 0.0);
        assert_eq!(grad[2], 2.0);
    }

    #[test]
    fn integer_tensors_fall_through_to_the_plain_reductions() {
        let ints = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![2, 5, 3], Device::cpu())),
            Shape::new(vec![3]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        // An integer cannot be NaN, so there is nothing to skip.
        assert_eq!(indices(&nanargmax(&ints, None, false).unwrap()), vec![1]);
        assert_eq!(indices(&nanargmin(&ints, None, false).unwrap()), vec![0]);
        assert_eq!(
            nanprod(&ints, None, false)
                .unwrap()
                .data()
                .as_i64_slice()
                .unwrap(),
            &[30]
        );
        // `nanvar` falls through too: it is `var`, which widens an integer
        // to a float rather than refusing it. It used to be the one member of
        // this family that raised.
        let variance = nanvar(&ints, None, false, true).unwrap();
        assert_eq!(variance.dtype(), DataType::Float64);
        let got = variance.data().as_f64_slice().unwrap()[0];
        assert!((got - 2.333_333_333_333_333_5).abs() < 1e-12, "got {got}");
    }

    /// These refused a list of axes while `var` and `std` accepted one, and
    /// the reason was the divisor rather than the statistic: the non-NaN count
    /// came from a `count_nonzero` that reduced a single axis. It counts over
    /// as many as it is given now, so the only thing left to check is that the
    /// divisor is still the number of values that were actually summed.
    #[test]
    fn several_reduction_dims_at_once_divide_by_the_non_nan_count() {
        let t = tensor(vec![1.0, 2.0, f64::NAN, 4.0], vec![2, 2]);
        let mean: f64 = (1.0 + 2.0 + 4.0) / 3.0;
        let biased: f64 =
            ((1.0 - mean).powi(2) + (2.0 - mean).powi(2) + (4.0 - mean).powi(2)) / 3.0;

        let got = nanvar(&t, Some(vec![0, 1]), false, false).unwrap();
        assert!((got.data().as_f64_slice().unwrap()[0] - biased).abs() < 1e-12);

        let unbiased = biased * 3.0 / 2.0;
        let got = nanvar(&t, Some(vec![0, 1]), false, true).unwrap();
        assert!((got.data().as_f64_slice().unwrap()[0] - unbiased).abs() < 1e-12);

        let got = nanstd(&t, Some(vec![0, 1]), false, true).unwrap();
        assert!((got.data().as_f64_slice().unwrap()[0] - unbiased.sqrt()).abs() < 1e-12);
    }
}
