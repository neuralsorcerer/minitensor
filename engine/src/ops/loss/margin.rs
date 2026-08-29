// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Similarity, margin and count losses.
//!
//! Unlike the losses in `regression.rs`, these are written as compositions of
//! ordinary tensor ops and let autograd record the whole computation. Each one
//! is a handful of element-wise steps around an existing primitive, so a
//! hand-written backward node would be a second derivation of arithmetic that
//! is already differentiated -- two places to be wrong instead of one, for a
//! forward that costs a few passes over a batch-sized tensor.

use crate::{
    error::{MinitensorError, Result},
    ops::{
        activation::{relu, softplus},
        arithmetic::{add, div, mul, neg, sub},
        loss::reduce_loss,
        reduction::{norm, sum},
        selection::where_op,
        util::create_scalar_tensor,
    },
    tensor::{Shape, Tensor},
};

/// A scalar of the same dtype and device as `like`, ready to broadcast.
fn scalar(value: f64, like: &Tensor) -> Result<Tensor> {
    create_scalar_tensor(value, like.dtype(), like.device())
}

/// Rejects the dtypes and device pairings none of these losses can work on.
fn check_operands(name: &str, operands: &[&Tensor]) -> Result<()> {
    let first = operands[0];
    for other in &operands[1..] {
        if other.device() != first.device() {
            return Err(MinitensorError::device_mismatch(
                format!("{:?}", first.device()),
                format!("{:?}", other.device()),
            ));
        }
    }
    if !first.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} requires floating point tensors, got {}",
            first.dtype()
        )));
    }
    Ok(())
}

/// Broadcasts two operands to their common shape, so a reduction over one of
/// them counts the same elements the product does.
fn align(lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor, Shape)> {
    let shape = lhs.shape().broadcast_with(rhs.shape())?;
    let dims: Vec<isize> = shape.dims().iter().map(|&d| d as isize).collect();
    Ok((lhs.expand(dims.clone())?, rhs.expand(dims)?, shape))
}

/// Cosine of the angle between `x1` and `x2` along `dim`.
///
/// Each norm is floored at `eps` on its own rather than their product being
/// floored once: that is what keeps a zero vector paired with a long one from
/// reporting a similarity far outside `[-1, 1]`, and it is PyTorch's
/// definition.
pub fn cosine_similarity(x1: &Tensor, x2: &Tensor, dim: isize, eps: f64) -> Result<Tensor> {
    check_operands("cosine_similarity", &[x1, x2])?;
    // Spelled out rather than as `!(eps > 0.0)`: NaN compares false either way
    // round, and it is no more a floor than a negative is.
    if eps.is_nan() || eps <= 0.0 {
        return Err(MinitensorError::invalid_argument(format!(
            "cosine_similarity requires a positive eps, got {eps}"
        )));
    }

    let (a, b, shape) = align(x1, x2)?;
    let axis = crate::ops::normalize_dim(dim, shape.ndim())?;
    let reduce = Some(vec![axis as isize]);

    let dot = sum(&mul(&a, &b)?, reduce.clone(), false)?;
    let floor = scalar(eps, &a)?;
    let norm_a = crate::ops::minmax::maximum(&norm(&a, 2.0, reduce.clone(), false)?, &floor)?;
    let norm_b = crate::ops::minmax::maximum(&norm(&b, 2.0, reduce, false)?, &floor)?;

    div(&dot, &mul(&norm_a, &norm_b)?)
}

/// `||x1 - x2||_p` along the last dimension, the distance the triplet loss
/// measures with.
///
/// `eps` is added before the norm, as PyTorch's `PairwiseDistance` does: at
/// two identical points the p-norm's derivative is undefined, and the shift
/// moves the evaluation off that point instead of returning NaN for a
/// perfectly good pair.
fn pairwise_distance(x1: &Tensor, x2: &Tensor, p: f64, eps: f64) -> Result<Tensor> {
    let shifted = add(&sub(x1, x2)?, &scalar(eps, x1)?)?;
    let axis = shifted.ndim().saturating_sub(1) as isize;
    norm(&shifted, p, Some(vec![axis]), false)
}

/// `max(0, -target * (x1 - x2) + margin)`.
///
/// `target` is `+1` where `x1` should rank higher and `-1` where `x2` should,
/// so the loss is zero exactly when the ranking is right by at least `margin`.
pub fn margin_ranking_loss(
    x1: &Tensor,
    x2: &Tensor,
    target: &Tensor,
    margin: f64,
    reduction: &str,
) -> Result<Tensor> {
    check_operands("margin_ranking_loss", &[x1, x2, target])?;
    let signed = mul(&neg(target)?, &sub(x1, x2)?)?;
    let values = relu(&add(&signed, &scalar(margin, &signed)?)?)?;
    reduce_loss(values, reduction)
}

/// `x` where `target` is `+1`, `max(0, margin - x)` where it is `-1`.
///
/// `x` is a distance, so a similar pair is penalised by how far apart it is
/// and a dissimilar one by how much closer than `margin` it has come.
pub fn hinge_embedding_loss(
    input: &Tensor,
    target: &Tensor,
    margin: f64,
    reduction: &str,
) -> Result<Tensor> {
    check_operands("hinge_embedding_loss", &[input, target])?;
    let similar = target_is_positive(target)?;
    let apart = relu(&sub(&scalar(margin, input)?, input)?)?;
    reduce_loss(where_op(&similar, input, &apart)?, reduction)
}

/// `1 - cos(x1, x2)` where `target` is `+1`, `max(0, cos(x1, x2) - margin)`
/// where it is `-1`.
pub fn cosine_embedding_loss(
    x1: &Tensor,
    x2: &Tensor,
    target: &Tensor,
    margin: f64,
    reduction: &str,
) -> Result<Tensor> {
    check_operands("cosine_embedding_loss", &[x1, x2, target])?;
    if !(-1.0..=1.0).contains(&margin) {
        return Err(MinitensorError::invalid_argument(format!(
            "cosine_embedding_loss requires margin in [-1, 1], got {margin}"
        )));
    }

    // The similarity runs along the feature axis, which is the last one for a
    // 1-D input and the second for the usual `(batch, features)`.
    let dim = if x1.ndim() <= 1 { 0 } else { 1 };
    let cosine = cosine_similarity(x1, x2, dim, 1e-8)?;

    let similar = target_is_positive(target)?;
    let pull = sub(&scalar(1.0, &cosine)?, &cosine)?;
    let push = relu(&sub(&cosine, &scalar(margin, &cosine)?)?)?;
    reduce_loss(where_op(&similar, &pull, &push)?, reduction)
}

/// `max(0, d(anchor, positive) - d(anchor, negative) + margin)`.
///
/// With `swap`, the negative distance becomes the smaller of
/// `d(anchor, negative)` and `d(positive, negative)`: if the positive is the
/// one closer to the negative, that is the violation worth penalising, and
/// ignoring it lets a triplet look satisfied while the classes still overlap.
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f64,
    p: f64,
    eps: f64,
    swap: bool,
    reduction: &str,
) -> Result<Tensor> {
    check_operands("triplet_margin_loss", &[anchor, positive, negative])?;
    if p.is_nan() || p <= 0.0 {
        return Err(MinitensorError::invalid_argument(format!(
            "triplet_margin_loss requires a positive norm order p, got {p}"
        )));
    }

    let to_positive = pairwise_distance(anchor, positive, p, eps)?;
    let mut to_negative = pairwise_distance(anchor, negative, p, eps)?;
    if swap {
        let positive_to_negative = pairwise_distance(positive, negative, p, eps)?;
        to_negative = crate::ops::minmax::minimum(&to_negative, &positive_to_negative)?;
    }

    let gap = add(
        &sub(&to_positive, &to_negative)?,
        &scalar(margin, &to_positive)?,
    )?;
    reduce_loss(relu(&gap)?, reduction)
}

/// `log(1 + exp(-target * input))`, for a `target` of `+1` or `-1`.
pub fn soft_margin_loss(input: &Tensor, target: &Tensor, reduction: &str) -> Result<Tensor> {
    check_operands("soft_margin_loss", &[input, target])?;
    // `softplus` rather than `log1p(exp(...))`: it takes the linear tail above
    // its threshold, where the exponential would overflow to infinity and the
    // logarithm would hand back that infinity instead of the value it converges
    // on.
    let values = softplus(&neg(&mul(target, input)?)?, 1.0, 20.0)?;
    reduce_loss(values, reduction)
}

/// The negative log-likelihood of a Poisson observation.
///
/// `log_input` says whether `input` is the log of the rate (the usual case,
/// and the numerically kind one) or the rate itself. `full` adds the Stirling
/// term that makes the result an actual log-likelihood rather than one short
/// of a constant; it changes no gradient, since it depends only on `target`.
pub fn poisson_nll_loss(
    input: &Tensor,
    target: &Tensor,
    log_input: bool,
    full: bool,
    eps: f64,
    reduction: &str,
) -> Result<Tensor> {
    check_operands("poisson_nll_loss", &[input, target])?;
    if eps < 0.0 || eps.is_nan() {
        return Err(MinitensorError::invalid_argument(format!(
            "poisson_nll_loss requires a non-negative eps, got {eps}"
        )));
    }

    let mut values = if log_input {
        // exp(input) - target * input
        sub(&crate::ops::activation::exp(input)?, &mul(target, input)?)?
    } else {
        // input - target * log(input + eps)
        let shifted = add(input, &scalar(eps, input)?)?;
        sub(
            input,
            &mul(target, &crate::ops::activation::log(&shifted)?)?,
        )?
    };

    if full {
        values = add(&values, &stirling_correction(target)?)?;
    }

    reduce_loss(values, reduction)
}

/// `target * log(target) - target + 0.5 * log(2 pi target)`, and zero where
/// `target` is 0 or 1.
///
/// Stirling's approximation to `log(target!)`, which is the constant the short
/// form of the Poisson likelihood drops. It is only an approximation above 1;
/// below that the exact value of `log(target!)` is zero, which is also what
/// the formula would give at `target = 1` and what it cannot give at 0.
fn stirling_correction(target: &Tensor) -> Result<Tensor> {
    let one = scalar(1.0, target)?;
    let large = crate::ops::comparison::gt(target, &one)?;

    // Evaluated on a floored copy so `log(0)` never runs on the branch that is
    // about to be discarded: `where` picks between two finished tensors, and a
    // `-inf * 0` in the unused one would still be NaN.
    let safe = crate::ops::minmax::maximum(target, &one)?;
    let log_target = crate::ops::activation::log(&safe)?;
    let two_pi = scalar(std::f64::consts::TAU, target)?;
    let stirling = add(
        &sub(&mul(&safe, &log_target)?, &safe)?,
        &mul(
            &scalar(0.5, target)?,
            &crate::ops::activation::log(&mul(&two_pi, &safe)?)?,
        )?,
    )?;

    where_op(&large, &stirling, &scalar(0.0, target)?)
}

/// `target == 1`, as the boolean mask the two-sided margin losses select with.
///
/// `target` carries `+1` and `-1`, so testing against `+1` is enough and
/// avoids a second comparison; a value that is neither lands on the `-1`
/// branch, which is the conservative reading.
fn target_is_positive(target: &Tensor) -> Result<Tensor> {
    let one = create_scalar_tensor(1.0, target.dtype(), target.device())?;
    crate::ops::comparison::eq(target, &one)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        autograd::backward_collect,
        device::Device,
        tensor::{DataType, TensorData},
    };
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

    fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                "{got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn cosine_similarity_matches_the_definition_and_stays_in_range() {
        let a = tensor(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor(vec![0.0, 1.0, 6.0, 8.0], vec![2, 2]);
        // Orthogonal, then parallel.
        close(
            &wide(&cosine_similarity(&a, &b, 1, 1e-8).unwrap()),
            &[0.0, 1.0],
        );

        // A zero vector has no direction; flooring each norm on its own keeps
        // the answer at zero rather than sending it to 1/eps.
        let zero = tensor(vec![0.0, 0.0], vec![1, 2]);
        let long = tensor(vec![3.0, 4.0], vec![1, 2]);
        let got = wide(&cosine_similarity(&zero, &long, 1, 1e-8).unwrap());
        assert_eq!(got, vec![0.0]);
    }

    #[test]
    fn cosine_similarity_broadcasts_one_vector_against_a_batch() {
        let batch = tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
        let query = tensor(vec![1.0, 0.0], vec![1, 2]);
        let got = wide(&cosine_similarity(&batch, &query, 1, 1e-8).unwrap());
        close(&got, &[1.0, 0.0, 1.0 / std::f64::consts::SQRT_2]);
    }

    #[test]
    fn margin_ranking_loss_is_zero_once_the_order_is_right_by_the_margin() {
        let x1 = tensor(vec![3.0, 0.0, 1.0], vec![3]);
        let x2 = tensor(vec![1.0, 1.0, 1.0], vec![3]);
        let target = tensor(vec![1.0, 1.0, -1.0], vec![3]);

        // margin 0: only the pair ranked the wrong way costs anything.
        close(
            &wide(&margin_ranking_loss(&x1, &x2, &target, 0.0, "none").unwrap()),
            &[0.0, 1.0, 0.0],
        );
        // margin 1.5: the first pair is ahead by 2, so it is still free; the
        // third ties and now costs the margin.
        close(
            &wide(&margin_ranking_loss(&x1, &x2, &target, 1.5, "none").unwrap()),
            &[0.0, 2.5, 1.5],
        );
    }

    #[test]
    fn hinge_embedding_loss_penalises_each_label_from_its_own_side() {
        let distances = tensor(vec![0.2, 0.2, 2.0, 2.0], vec![4]);
        let target = tensor(vec![1.0, -1.0, 1.0, -1.0], vec![4]);
        // margin 1: a similar pair pays its distance; a dissimilar one pays
        // what it lacks of the margin, and nothing once it is past it.
        close(
            &wide(&hinge_embedding_loss(&distances, &target, 1.0, "none").unwrap()),
            &[0.2, 0.8, 2.0, 0.0],
        );
    }

    #[test]
    fn cosine_embedding_loss_pulls_and_pushes() {
        let x1 = tensor(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]);
        let x2 = tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        // Identical pair labelled similar: no loss. Orthogonal pair labelled
        // dissimilar with margin 0: also none, since the cosine is already 0.
        let target = tensor(vec![1.0, -1.0], vec![2]);
        close(
            &wide(&cosine_embedding_loss(&x1, &x2, &target, 0.0, "none").unwrap()),
            &[0.0, 0.0],
        );

        // Swap the labels and both are wrong: 1 - cos = 1 for the orthogonal
        // pair, cos - margin = 1 for the identical one.
        let target = tensor(vec![-1.0, 1.0], vec![2]);
        close(
            &wide(&cosine_embedding_loss(&x1, &x2, &target, 0.0, "none").unwrap()),
            &[1.0, 1.0],
        );
    }

    #[test]
    fn triplet_margin_loss_measures_the_gap_and_swaps_when_asked() {
        // anchor at the origin, positive at distance 1, negative at distance 2.
        let anchor = tensor(vec![0.0, 0.0], vec![1, 2]);
        let positive = tensor(vec![1.0, 0.0], vec![1, 2]);
        let negative = tensor(vec![2.0, 0.0], vec![1, 2]);

        // Gap is 1 - 2 = -1, so margin 1 leaves exactly zero and margin 2 costs 1.
        let loss = |margin, swap| {
            wide(
                &triplet_margin_loss(
                    &anchor, &positive, &negative, margin, 2.0, 0.0, swap, "none",
                )
                .unwrap(),
            )[0]
        };
        assert!(loss(1.0, false).abs() < 1e-12);
        close(&[loss(2.0, false)], &[1.0]);

        // The positive is only 1 away from the negative, closer than the
        // anchor is. With `swap` that is the distance used, and the triplet
        // that looked satisfied now costs the margin.
        close(&[loss(1.0, true)], &[1.0]);
    }

    #[test]
    fn soft_margin_loss_is_the_smooth_hinge_and_survives_large_margins() {
        let input = tensor(vec![0.0, 2.0, -2.0], vec![3]);
        let target = tensor(vec![1.0, 1.0, 1.0], vec![3]);
        let got = wide(&soft_margin_loss(&input, &target, "none").unwrap());
        for (i, &x) in [0.0f64, 2.0, -2.0].iter().enumerate() {
            let want = (1.0 + (-x).exp()).ln();
            assert!((got[i] - want).abs() < 1e-12);
        }

        // A confidently wrong sample: the exponential in the definition
        // overflows, and the loss converges on the linear tail.
        let input = tensor(vec![-800.0], vec![1]);
        let target = tensor(vec![1.0], vec![1]);
        close(
            &wide(&soft_margin_loss(&input, &target, "none").unwrap()),
            &[800.0],
        );
    }

    #[test]
    fn poisson_nll_loss_takes_the_rate_either_way_round() {
        let rate = tensor(vec![1.0, 2.0, 4.0], vec![3]);
        let target = tensor(vec![0.0, 1.0, 3.0], vec![3]);

        // Given the rate directly.
        let got = wide(&poisson_nll_loss(&rate, &target, false, false, 1e-8, "none").unwrap());
        for (i, (&r, &t)) in [1.0f64, 2.0, 4.0].iter().zip(&[0.0, 1.0, 3.0]).enumerate() {
            let want = r - t * (r + 1e-8).ln();
            assert!((got[i] - want).abs() < 1e-9, "{} vs {want}", got[i]);
        }

        // Given its logarithm, which is the same likelihood.
        let log_rate = tensor(vec![0.0, 2.0_f64.ln(), 4.0_f64.ln()], vec![3]);
        let got = wide(&poisson_nll_loss(&log_rate, &target, true, false, 1e-8, "none").unwrap());
        for (i, (&r, &t)) in [1.0f64, 2.0, 4.0].iter().zip(&[0.0, 1.0, 3.0]).enumerate() {
            let want = r - t * r.ln();
            assert!((got[i] - want).abs() < 1e-12, "{} vs {want}", got[i]);
        }
    }

    #[test]
    fn the_stirling_term_is_added_only_above_one_and_changes_no_gradient() {
        let rate = tensor(vec![1.0, 1.0, 1.0], vec![3]).requires_grad_(true);
        let target = tensor(vec![0.0, 1.0, 5.0], vec![3]);

        let short = wide(&poisson_nll_loss(&rate, &target, true, false, 0.0, "none").unwrap());
        let full = wide(&poisson_nll_loss(&rate, &target, true, true, 0.0, "none").unwrap());
        // log(0!) and log(1!) are both zero, so only the last sample moves.
        assert_eq!(short[0], full[0]);
        assert_eq!(short[1], full[1]);
        let want = 5.0 * 5.0_f64.ln() - 5.0 + 0.5 * (std::f64::consts::TAU * 5.0).ln();
        assert!((full[2] - short[2] - want).abs() < 1e-12);

        // It depends only on the target, so the gradient is untouched.
        //
        // Each pass gets its own leaf: the graph is global and lives until it
        // is cleared, so two backward passes from one leaf would accumulate
        // into the same slot and report twice the gradient.
        let grad_of = |full_term| {
            let rate = tensor(vec![1.0, 1.0, 1.0], vec![3]).requires_grad_(true);
            let out = poisson_nll_loss(&rate, &target, true, full_term, 0.0, "sum").unwrap();
            let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
            wide(
                backward_collect(&out, Some(seed))
                    .unwrap()
                    .get(&rate.id())
                    .unwrap(),
            )
        };
        assert_eq!(grad_of(false), grad_of(true));
    }

    #[test]
    fn gradients_match_central_differences() {
        let eps = 1e-6;
        let base = vec![0.3, -0.7, 1.4, 0.9];
        let other = tensor(vec![0.5, 0.2, -1.1, 0.4], vec![2, 2]);
        let target = tensor(vec![1.0, -1.0], vec![2]);
        let flat_target = tensor(vec![1.0, -1.0, 1.0, -1.0], vec![2, 2]);

        /// One loss under test: its name, and the call with every operand but
        /// the one being differentiated already bound.
        type BoundLoss<'a> = (&'a str, Box<dyn Fn(&Tensor) -> Result<Tensor> + 'a>);

        let cases: Vec<BoundLoss> = vec![
            (
                "margin_ranking",
                Box::new(|t: &Tensor| margin_ranking_loss(t, &other, &flat_target, 0.5, "sum")),
            ),
            (
                "hinge_embedding",
                Box::new(|t: &Tensor| hinge_embedding_loss(t, &flat_target, 1.0, "sum")),
            ),
            (
                "cosine_embedding",
                Box::new(|t: &Tensor| cosine_embedding_loss(t, &other, &target, 0.2, "sum")),
            ),
            (
                "triplet",
                Box::new(|t: &Tensor| {
                    triplet_margin_loss(t, &other, &flat_target, 1.0, 2.0, 1e-6, false, "sum")
                }),
            ),
            (
                "soft_margin",
                Box::new(|t: &Tensor| soft_margin_loss(t, &flat_target, "sum")),
            ),
            (
                "poisson",
                Box::new(|t: &Tensor| poisson_nll_loss(t, &other, true, false, 0.0, "sum")),
            ),
        ];

        for (name, build) in cases {
            let input = tensor(base.clone(), vec![2, 2]).requires_grad_(true);
            let out = build(&input).unwrap();
            let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
            let analytic = wide(
                backward_collect(&out, Some(seed))
                    .unwrap()
                    .get(&input.id())
                    .unwrap(),
            );

            for i in 0..base.len() {
                let mut up = base.clone();
                let mut down = base.clone();
                up[i] += eps;
                down[i] -= eps;
                let f = |v: Vec<f64>| wide(&build(&tensor(v, vec![2, 2])).unwrap())[0];
                let numeric = (f(up) - f(down)) / (2.0 * eps);
                assert!(
                    (analytic[i] - numeric).abs() <= 1e-4 * (1.0 + numeric.abs()),
                    "{name} d/dx[{i}]: analytic {}, numeric {numeric}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn reductions_agree_with_each_other() {
        let x1 = tensor(vec![3.0, 0.0, 1.0, 2.0], vec![4]);
        let x2 = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let target = tensor(vec![1.0, 1.0, -1.0, -1.0], vec![4]);

        let none = wide(&margin_ranking_loss(&x1, &x2, &target, 0.5, "none").unwrap());
        let summed = wide(&margin_ranking_loss(&x1, &x2, &target, 0.5, "sum").unwrap())[0];
        let averaged = wide(&margin_ranking_loss(&x1, &x2, &target, 0.5, "mean").unwrap())[0];

        assert!((summed - none.iter().sum::<f64>()).abs() < 1e-12);
        assert!((averaged - summed / none.len() as f64).abs() < 1e-12);
        assert!(margin_ranking_loss(&x1, &x2, &target, 0.5, "batchmean").is_err());
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let a = tensor(vec![1.0, 2.0], vec![1, 2]);
        assert!(cosine_similarity(&a, &a, 0, 0.0).is_err());
        assert!(cosine_similarity(&a, &a, 5, 1e-8).is_err());
        assert!(cosine_embedding_loss(&a, &a, &a, 2.0, "mean").is_err());
        assert!(triplet_margin_loss(&a, &a, &a, 1.0, 0.0, 1e-6, false, "mean").is_err());
        assert!(poisson_nll_loss(&a, &a, true, false, -1.0, "mean").is_err());

        let ints = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![1, 2], Device::cpu())),
            Shape::new(vec![1, 2]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        assert!(cosine_similarity(&ints, &ints, 1, 1e-8).is_err());
        assert!(soft_margin_loss(&ints, &ints, "mean").is_err());
    }
}
