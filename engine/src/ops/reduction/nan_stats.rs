// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The NaN-skipping reductions that are arrangements of the others.
//!
//! `nansum`, `nanmean`, `nanmax` and the rest each carry a kernel that walks
//! the buffer once, testing as it goes. These five need no kernel: a variance
//! is a mean of squared deviations, a product with NaN skipped is a product
//! with NaN replaced by the identity, and an index of the largest non-NaN is an
//! index of the largest once NaN has been pushed to the bottom. Writing them as
//! those arrangements is one definition rather than two, and it is what makes
//! their gradients the gradients of the ops underneath.

use crate::{
    error::{MinitensorError, Result},
    ops::{
        activation::{nan_to_num, sqrt},
        arithmetic::{div, mul, sub},
        comparison::eq,
        reduction::{any, argmax, argmin, count_nonzero, nanmean, prod, sum},
        selection::where_op,
        util::create_scalar_tensor,
    },
    tensor::{DataType, Shape, Tensor},
};

/// How many entries along `dim` are not NaN, as a float ready to divide by.
///
/// `count_nonzero` counts the mask, which is what `nanmean` divides by too;
/// this is the same count in the dtype the division needs.
fn non_nan_count(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
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

/// The single reduction axis these take, since the count they divide by comes
/// from `count_nonzero`, which reduces one axis at a time.
fn single_dim(dim: Option<Vec<isize>>, name: &str) -> Result<Option<isize>> {
    match dim {
        None => Ok(None),
        Some(dims) if dims.len() == 1 => Ok(Some(dims[0])),
        Some(dims) => Err(MinitensorError::invalid_operation(format!(
            "{name} reduces one dimension at a time, got {} of them",
            dims.len()
        ))),
    }
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
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "nanvar is only supported for floating point tensors",
        ));
    }

    let axis = single_dim(dim.clone(), "nanvar")?;

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

    let total = sum(&squared, dim, keepdim)?;
    let count = non_nan_count(tensor, axis, keepdim)?;
    let divisor = if unbiased {
        sub(
            &count,
            &create_scalar_tensor(1.0, tensor.dtype(), tensor.device())?,
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
    prod(&nan_to_num(tensor, 1.0, None, None)?, dim, keepdim)
}

/// Pushes every NaN to one end so an index reduction skips it, and reports
/// whether a slice was left with nothing but pushed-aside values.
fn without_nan(tensor: &Tensor, replacement: f64) -> Result<Tensor> {
    let filler = create_scalar_tensor(replacement, tensor.dtype(), tensor.device())?;
    where_op(&tensor.isnan()?, &filler, tensor)
}

/// Rejects a reduction that would have to name an index among NaN alone.
///
/// `nanamax` answers NaN for such a slice, which an index reduction cannot do:
/// every index it could return points at a NaN. NumPy raises here, and so does
/// this.
fn reject_all_nan(tensor: &Tensor, dim: Option<isize>, name: &str) -> Result<()> {
    let count = non_nan_count(tensor, dim, false)?;
    let zero = create_scalar_tensor(0.0, tensor.dtype(), tensor.device())?;
    let empty = eq(&count, &zero)?;
    if any(&empty, None, false)?
        .data()
        .as_bool_slice()
        .map(|slice| slice.first().copied().unwrap_or(false))
        .unwrap_or(false)
    {
        return Err(MinitensorError::invalid_operation(format!(
            "{name}: a slice of all-NaN values has no index to report"
        )));
    }
    Ok(())
}

/// Index of the largest non-NaN entry along `dim`.
pub fn nanargmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return argmax(tensor, dim, keepdim);
    }
    reject_all_nan(tensor, dim, "nanargmax")?;
    argmax(&without_nan(tensor, f64::NEG_INFINITY)?, dim, keepdim)
}

/// Index of the smallest non-NaN entry along `dim`.
pub fn nanargmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return argmin(tensor, dim, keepdim);
    }
    reject_all_nan(tensor, dim, "nanargmin")?;
    argmin(&without_nan(tensor, f64::INFINITY)?, dim, keepdim)
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
        assert!(nanvar(&ints, None, false, true).is_err());
    }

    #[test]
    fn several_reduction_dims_at_once_are_refused() {
        let t = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(nanvar(&t, Some(vec![0, 1]), false, true).is_err());
        assert!(nanstd(&t, Some(vec![0, 1]), false, true).is_err());
    }
}
