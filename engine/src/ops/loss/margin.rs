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
//!
//! The cosine is the exception, and the Poisson terms: rounded once, each is
//! one kernel rather than a composition, and records its own backward.

use std::sync::Arc;

use super::regression_impl::manual_backward_needed;
use crate::{
    autograd::{CosineSimilarityBackward, NoGradGuard, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        activation::{relu, softplus},
        arithmetic::{add, mul, neg, sub},
        loss::{check_finite_parameter, reduce_loss},
        reduction::norm,
        selection::where_op,
        util::create_scalar_tensor,
    },
    tensor::{DataType, Shape, Tensor},
};

/// A scalar of the same dtype and device as `like`, ready to broadcast.
fn scalar(value: f64, like: &Tensor) -> Result<Tensor> {
    create_scalar_tensor(value, like.dtype(), like.device())
}

/// Rejects the dtypes and device pairings none of these losses can work on.
/// The operands of one of these, on one device and at the float dtype they
/// promote to. The first `inputs` are a model's activations and must be
/// floats, as they must for every layer-shaped op; the rest are labels or
/// counts of any dtype. They meet as arithmetic meets them -- the wider float
/// decides, and an integer label takes its width -- where an integer label of
/// `1` or `-1` used to fail further in, and a float32 input against a float64
/// target kept the narrower width or was refused.
fn float_operands<const N: usize>(
    name: &str,
    inputs: usize,
    operands: [&Tensor; N],
) -> Result<[Tensor; N]> {
    let first = operands[0];
    for other in &operands[1..] {
        if other.device() != first.device() {
            return Err(MinitensorError::device_mismatch(
                format!("{:?}", first.device()),
                format!("{:?}", other.device()),
            ));
        }
    }
    if let Some(integer) = operands[..inputs].iter().find(|t| !t.dtype().is_float()) {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} requires floating point tensors, got {}",
            integer.dtype()
        )));
    }
    // Promoted as arithmetic first and made a float last: widening the first
    // operand on its own would take an int64 to float64 before it met the
    // float32 that decides the width.
    let joint = operands[1..].iter().fold(first.dtype(), |dtype, other| {
        crate::ops::binary::promote_arithmetic_dtype(dtype, other.dtype())
    });
    let common = crate::ops::binary::promote_division_dtype(joint, joint);
    let mut converted = Vec::with_capacity(N);
    for operand in operands {
        converted.push(if operand.dtype() == common {
            operand.clone()
        } else {
            operand.astype(common)?
        });
    }
    Ok(converted
        .try_into()
        .unwrap_or_else(|_| unreachable!("one tensor per operand")))
}

/// Rejects operands of a loss whose shapes differ.
///
/// Each of these losses pairs its operands element by element, and they were
/// broadcast into one another instead: an input of shape `(n, 1)` against a
/// target of `(n,)` became `(n, n)`, every prediction scored against every
/// target, and the mean of that came back as the loss -- a plausible number,
/// and a wrong one, with nothing to say so.
fn check_same_shape(name: &str, operands: &[(&str, &Tensor)]) -> Result<()> {
    let (first_name, first) = operands[0];
    for &(other_name, other) in &operands[1..] {
        if other.shape() != first.shape() {
            return Err(MinitensorError::invalid_argument(format!(
                "{name} pairs `{first_name}` and `{other_name}` element by element, \
                 so they must have the same shape, but `{first_name}` is {:?} and \
                 `{other_name}` is {:?}",
                first.shape().dims(),
                other.shape().dims()
            )));
        }
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
/// reporting a similarity far outside `[-1, 1]`.
///
/// Each cosine is the float nearest the exact dot over the exact floored
/// norms, `eps` as the dtype holds it; see
/// `arithmetic::exact_sum::cosine`. Formed from ordinary operations -- each
/// operand over its own norm, and their products summed -- every step
/// rounded on its own, and one cosine in four or so of a few hundred random
/// pairs was an ulp or more out. The gradient is the quotient's, taken in
/// [`CosineSimilarityBackward`].
pub fn cosine_similarity(x1: &Tensor, x2: &Tensor, dim: isize, eps: f64) -> Result<Tensor> {
    let [x1, x2] = float_operands("cosine_similarity", 2, [x1, x2])?;
    let (x1, x2) = (&x1, &x2);
    // Spelled out rather than as `!(eps > 0.0)`: NaN compares false either way
    // round, and it is no more a floor than a negative is.
    if !(eps.is_finite() && eps > 0.0) {
        return Err(MinitensorError::invalid_argument(format!(
            "cosine_similarity requires a finite, positive eps, got {eps}"
        )));
    }

    let needs_grad = manual_backward_needed(&[x1, x2]);
    let (a, b, axis, cosine) = {
        let _guard = NoGradGuard::new();
        let (a, b, shape) = align(x1, x2)?;
        let axis = crate::ops::normalize_dim(dim, shape.ndim())?;
        let cosine = crate::ops::reduction::cosine_slices(&a, &b, axis, eps)?;
        (a, b, axis, cosine)
    };
    if !needs_grad {
        return Ok(cosine);
    }
    let grad_fn = Arc::new(CosineSimilarityBackward {
        a: a.detach(),
        b: b.detach(),
        weights: None,
        axis,
        floor: eps,
        shapes: [x1.shape().clone(), x2.shape().clone()],
        input_ids: [x1.id(), x2.id()],
        input_requires_grad: [x1.requires_grad(), x2.requires_grad()],
    });
    with_grad_fn(cosine.requires_grad_(true), grad_fn)
}

/// `||x1 - x2||_p` along the last dimension, the distance the triplet loss
/// measures with.
///
/// `eps` is added before the norm: at two identical points the p-norm's derivative is undefined, and the shift
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
    let [x1, x2, target] = float_operands("margin_ranking_loss", 2, [x1, x2, target])?;
    let (x1, x2, target) = (&x1, &x2, &target);
    check_finite_parameter("margin_ranking_loss", "margin", margin)?;
    check_same_shape(
        "margin_ranking_loss",
        &[("input1", x1), ("input2", x2), ("target", target)],
    )?;
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
    let [input, target] = float_operands("hinge_embedding_loss", 1, [input, target])?;
    let (input, target) = (&input, &target);
    check_finite_parameter("hinge_embedding_loss", "margin", margin)?;
    check_same_shape(
        "hinge_embedding_loss",
        &[("input", input), ("target", target)],
    )?;
    let similar = target_is_positive(target)?;
    let apart = relu(&sub(&scalar(margin, input)?, input)?)?;
    reduce_loss(where_op(&similar, input, &apart)?, reduction)
}

/// `1 - cos(x1, x2)` where `target` is `+1`, `max(0, cos(x1, x2) - margin)`
/// where it is `-1`. The first is formed from the part of one input at right
/// angles to the other, and cancels nothing where they lie almost along each
/// other.
pub fn cosine_embedding_loss(
    x1: &Tensor,
    x2: &Tensor,
    target: &Tensor,
    margin: f64,
    reduction: &str,
) -> Result<Tensor> {
    let [x1, x2, target] = float_operands("cosine_embedding_loss", 2, [x1, x2, target])?;
    let (x1, x2, target) = (&x1, &x2, &target);
    if !(-1.0..=1.0).contains(&margin) {
        return Err(MinitensorError::invalid_argument(format!(
            "cosine_embedding_loss requires margin in [-1, 1], got {margin}"
        )));
    }

    check_same_shape("cosine_embedding_loss", &[("input1", x1), ("input2", x2)])?;
    // A cosine is an angle between vectors, and a 0-d input has no feature
    // axis to hold one; asked for its axis, the shape had none to remove.
    if x1.ndim() == 0 {
        return Err(MinitensorError::invalid_argument(
            "cosine_embedding_loss compares vectors of features, and its inputs are 0-d",
        ));
    }

    // The similarity runs along the feature axis, which is the last one for a
    // 1-D input and the second for the usual `(batch, features)`.
    let dim: usize = if x1.ndim() <= 1 { 0 } else { 1 };
    // One target per pair: the inputs' shape without that axis. A target
    // shaped like anything else was broadcast against the similarities.
    let mut pairs = x1.shape().dims().to_vec();
    pairs.remove(dim);
    if target.shape().dims() != pairs.as_slice() {
        return Err(MinitensorError::invalid_argument(format!(
            "cosine_embedding_loss takes one target per pair of inputs, so for \
             inputs of shape {:?} the target must have shape {pairs:?}, but it is {:?}",
            x1.shape().dims(),
            target.shape().dims()
        )));
    }
    // Each pair's term and the factor its cosine's gradient takes in it, from
    // the cosine and its complement, with one gradient for them all.
    let floor = 1e-8;
    let needs_grad = manual_backward_needed(&[x1, x2]);
    let (a, b, terms, weights) = {
        let _guard = NoGradGuard::new();
        let (a, b, _) = align(x1, x2)?;
        let (terms, weights) =
            crate::ops::reduction::cosine_embedding_slices(&a, &b, target, dim, (floor, margin))?;
        (a, b, terms, weights)
    };
    let terms = if needs_grad {
        let grad_fn = Arc::new(CosineSimilarityBackward {
            a: a.detach(),
            b: b.detach(),
            weights: Some(weights),
            axis: dim,
            floor,
            shapes: [x1.shape().clone(), x2.shape().clone()],
            input_ids: [x1.id(), x2.id()],
            input_requires_grad: [x1.requires_grad(), x2.requires_grad()],
        });
        with_grad_fn(terms.requires_grad_(true), grad_fn)?
    } else {
        terms
    };
    reduce_loss(terms, reduction)
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
    let [anchor, positive, negative] =
        float_operands("triplet_margin_loss", 3, [anchor, positive, negative])?;
    let (anchor, positive, negative) = (&anchor, &positive, &negative);
    check_same_shape(
        "triplet_margin_loss",
        &[
            ("anchor", anchor),
            ("positive", positive),
            ("negative", negative),
        ],
    )?;
    // An infinite `p` is the max-norm, a distance like any other; NaN is not.
    if p.is_nan() || p <= 0.0 {
        return Err(MinitensorError::invalid_argument(format!(
            "triplet_margin_loss requires a positive norm order p, got {p}"
        )));
    }
    check_finite_parameter("triplet_margin_loss", "margin", margin)?;
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(MinitensorError::invalid_argument(format!(
            "triplet_margin_loss requires a finite, non-negative eps, got {eps}"
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
    let [input, target] = float_operands("soft_margin_loss", 1, [input, target])?;
    let (input, target) = (&input, &target);
    check_same_shape("soft_margin_loss", &[("input", input), ("target", target)])?;
    // `softplus` rather than `log1p(exp(...))`, whose exponential overflows to
    // infinity where the loss is an ordinary number: softplus takes it as
    // `max(z, 0) + log1p(exp(-|z|))`, which cannot. With no threshold, which
    // would take the loss past it as `z` and its slope as one: past 20 that
    // left `log1p(e^-z)` out, 5.7e5 float64 ulps of the loss there and 1.8e7
    // of the slope.
    let values = softplus(&neg(&mul(target, input)?)?, 1.0, f64::INFINITY)?;
    reduce_loss(values, reduction)
}

/// The negative log-likelihood of a Poisson observation.
///
/// `log_input` says whether `input` is the log of the rate (the usual case,
/// and the numerically kind one) or the rate itself. `full` adds the Stirling
/// term that makes the result an actual log-likelihood rather than one short
/// of a constant; it changes no gradient in `input`, since it depends only on
/// `target`. Each term and its gradients are rounded once; see
/// [`PoissonBackward`].
pub fn poisson_nll_loss(
    input: &Tensor,
    target: &Tensor,
    log_input: bool,
    full: bool,
    eps: f64,
    reduction: &str,
) -> Result<Tensor> {
    let [input, target] = float_operands("poisson_nll_loss", 1, [input, target])?;
    let (input, target) = (&input, &target);
    check_same_shape("poisson_nll_loss", &[("input", input), ("target", target)])?;
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(MinitensorError::invalid_argument(format!(
            "poisson_nll_loss requires a finite, non-negative eps, got {eps}"
        )));
    }

    ensure_counts(target)?;

    let mut values = poisson_terms(input, target, log_input, eps)?;

    if full {
        values = add(&values, &stirling_correction(target)?)?;
    }

    reduce_loss(values, reduction)
}

/// Each Poisson term, `exp(x) - t x` for a log-rate and `x - t log(x + eps)`
/// for a rate, rounded once from float64 pairs; with its gradient,
/// [`PoissonBackward`].
fn poisson_terms(input: &Tensor, target: &Tensor, log_input: bool, eps: f64) -> Result<Tensor> {
    let needs_grad = manual_backward_needed(&[input, target]);
    let (x, t) = (widened(input)?, widened(target)?);
    let (values, counts) = (f64_values(&x)?, f64_values(&t)?);
    let mut out = vec![0.0f64; values.len()];
    let wide = input.dtype() == DataType::Float64;
    poisson_pass(values, counts, None, (log_input, eps, wide), &mut out);
    let terms = narrowed(out, input)?;
    if !needs_grad {
        return Ok(terms);
    }
    let grad_fn = Arc::new(PoissonBackward {
        input: x.detach(),
        target: t.detach(),
        log_input,
        eps,
        input_ids: [input.id(), target.id()],
        input_requires_grad: [input.requires_grad(), target.requires_grad()],
    });
    with_grad_fn(terms.requires_grad_(true), grad_fn)
}

/// The gradient of [`poisson_terms`]: in the input `g` times `exp(x) - t`, or
/// `1 - t / (x + eps)`, each formed where its two terms cancel -- near the
/// fit, `exp(x)` or `x` beside `t` -- without taking either rounded first,
/// and in the target `-g x` or `-g log(x + eps)`, the logarithm a pair;
/// each rounded once.
pub struct PoissonBackward {
    /// The operands, widened to float64.
    input: Tensor,
    target: Tensor,
    log_input: bool,
    eps: f64,
    input_ids: [crate::autograd::TensorId; 2],
    input_requires_grad: [bool; 2],
}

impl crate::autograd::GradientFunction for PoissonBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
    ) -> Result<rustc_hash::FxHashMap<crate::autograd::TensorId, Tensor>> {
        let mut gradients = rustc_hash::FxHashMap::default();
        let flowing = widened(grad_output)?;
        let (x, t, g) = (
            f64_values(&self.input)?,
            f64_values(&self.target)?,
            f64_values(&flowing)?,
        );
        let like = grad_output;
        let wide = like.dtype() == DataType::Float64;
        if self.input_requires_grad[0] {
            let mut out = vec![0.0f64; x.len()];
            poisson_pass(
                x,
                t,
                Some((g, false)),
                (self.log_input, self.eps, wide),
                &mut out,
            );
            let grad = narrowed(out, like)?;
            crate::autograd::accumulate_grad(&mut gradients, self.input_ids[0], grad)?;
        }
        if self.input_requires_grad[1] {
            let mut out = vec![0.0f64; x.len()];
            poisson_pass(
                x,
                t,
                Some((g, true)),
                (self.log_input, self.eps, wide),
                &mut out,
            );
            let grad = narrowed(out, like)?;
            crate::autograd::accumulate_grad(&mut gradients, self.input_ids[1], grad)?;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[crate::autograd::TensorId] {
        &self.input_ids
    }
}

/// `tensor` in float64, contiguous.
fn widened(tensor: &Tensor) -> Result<Tensor> {
    tensor.astype(DataType::Float64)?.contiguous()
}

fn f64_values(tensor: &Tensor) -> Result<&[f64]> {
    tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("poisson_nll_loss: a float64 operand"))
}

/// `values` as a tensor of `like`'s shape and dtype, rounded once.
fn narrowed(values: Vec<f64>, like: &Tensor) -> Result<Tensor> {
    let data = crate::tensor::TensorData::from_vec_f64(values, like.device());
    let wide = Tensor::new(
        Arc::new(data),
        like.shape().clone(),
        DataType::Float64,
        like.device(),
        false,
    );
    wide.astype(like.dtype())
}

/// The Poisson terms of `x` and `t` into `out`, or with `g` their slopes
/// times it: in `x`, or with the flag in `t`. `wide` asks for float64's
/// digits, where float32's are wanted otherwise: float64 arithmetic gives
/// either everywhere but where two terms all but cancel, and there a pair
/// takes over, for float32 only once the cancellation reaches 2^-20.
fn poisson_pass(
    x: &[f64],
    t: &[f64],
    g: Option<(&[f64], bool)>,
    setting: (bool, f64, bool),
    out: &mut [f64],
) {
    use crate::ops::arithmetic::extended::fused;
    crate::ops::map::par_out_chunks_gated(
        out,
        4096,
        crate::ops::map::PAR_THRESHOLD / 8,
        &|start, part| {
            let end = start + part.len();
            let (x, t) = (&x[start..end], &t[start..end]);
            let g = g.map(|(g, of_t)| (&g[start..end], of_t));
            fused!(
                poisson_part(
                    x: &[f64],
                    t: &[f64],
                    g: Option<(&[f64], bool)>,
                    setting: (bool, f64, bool),
                    part: &mut [f64],
                ) -> ()
            )
        },
    );
}

#[inline(always)]
fn poisson_part<const FMA: bool>(
    x: &[f64],
    t: &[f64],
    g: Option<(&[f64], bool)>,
    (log_input, eps, wide): (bool, f64, bool),
    out: &mut [f64],
) {
    use crate::ops::arithmetic::extended::Pair;
    // The logarithms of the counts below 64 as pairs, found once a task.
    let mut logs = [Pair(f64::NAN, 0.0); 64];
    for (k, slot) in out.iter_mut().enumerate() {
        let (x, t) = (x[k], t[k]);
        *slot = match (g, log_input) {
            (None, true) => log_rate_term::<FMA>(x, t, wide),
            (None, false) => rate_term::<FMA>(x, t, eps, wide),
            (Some((g, false)), true) => {
                rate_less_count::<FMA>(x, t, wide, &mut logs).times_rounded::<FMA>(g[k])
            }
            (Some((g, false)), false) => {
                Pair(rate_slope(x, t, eps), 0.0).times_rounded::<FMA>(g[k])
            }
            (Some((g, true)), true) => -(g[k] * x),
            (Some((g, true)), false) => -log_shifted(x, eps).times_rounded::<FMA>(g[k]),
        };
    }
}

/// How far below its terms' magnitudes a difference taken in float64 may
/// fall before a pair takes it instead: a quarter of them for a float64
/// answer, which keeps it within two and a half ulps above that, and `2^-20`
/// of them for a float32 one.
fn cancels(difference: f64, magnitudes: f64, wide: bool) -> bool {
    let share = if wide { 0.25 } else { 2f64.powi(-20) };
    difference.abs() < share * magnitudes
}

/// `exp(x) - t x`: in float64 where the two are apart, and where they all
/// but cancel with the exponential a pair, the product exact, rounded once.
#[inline(always)]
fn log_rate_term<const FMA: bool>(x: f64, t: f64, wide: bool) -> f64 {
    use crate::ops::arithmetic::extended::{Pair, exp_pair};
    let (e, p) = (x.exp(), t * x);
    let plain = e - p;
    if !(cancels(plain, e + p.abs(), wide) && x.abs() <= 708.0 && t.is_finite()) {
        return plain;
    }
    let product = Pair(t, 0.0).mul::<FMA>(Pair(x, 0.0));
    exp_pair::<FMA>(Pair(x, 0.0)).add(product.neg()).0
}

/// `x - t log(x + eps)`: in float64 where the two are apart, the logarithm
/// as [`log_shifted`] takes it, and where they all but cancel with the
/// logarithm a pair, rounded once.
#[inline(always)]
fn rate_term<const FMA: bool>(x: f64, t: f64, eps: f64, wide: bool) -> f64 {
    use crate::ops::arithmetic::extended::Pair;
    let log = log_shifted(x, eps);
    let taken = t * log.0;
    let plain = x - (taken + t * log.1);
    let finite = x.is_finite() && t.is_finite() && log.0.is_finite();
    if !(finite && cancels(plain, x.abs() + taken.abs(), wide)) {
        return plain;
    }
    let log = log_pair::<FMA>(x, eps);
    Pair(x, 0.0).add(log.mul::<FMA>(Pair(t, 0.0)).neg()).0
}

/// `log(x + eps)`, with a trailing correction: near one, where it is small,
/// `log1p` of the exact `x + eps - 1` and the share of what that leaves out,
/// where the rounded sum's logarithm would keep few of its digits; the
/// float64 logarithm elsewhere. Within an ulp or two either way.
#[inline(always)]
fn log_shifted(x: f64, eps: f64) -> crate::ops::arithmetic::extended::Pair {
    use crate::ops::arithmetic::extended::Pair;
    use crate::ops::arithmetic::rounded_once::two_sum;
    let (s, s_lo) = two_sum(x, eps);
    let (w, w_lo) = two_sum(s, -1.0);
    if w.is_nan() || w.abs() >= 0.5 {
        return Pair((x + eps).ln(), 0.0);
    }
    Pair(w.ln_1p(), (w_lo + s_lo) / (1.0 + w))
}

/// `log(x + eps)` as a pair, of the sum a pair, where that is positive and in
/// range; the float64 logarithm of the rounded sum elsewhere.
#[inline(always)]
fn log_pair<const FMA: bool>(x: f64, eps: f64) -> crate::ops::arithmetic::extended::Pair {
    use crate::ops::arithmetic::extended::{Pair, ln_1p_pair};
    use crate::ops::arithmetic::rounded_once::two_sum;
    let (s, s_lo) = two_sum(x, eps);
    if !(s > 0.0 && s < 2f64.powi(1000)) {
        return Pair((x + eps).ln(), 0.0);
    }
    let (w, w_lo) = two_sum(s, -1.0);
    ln_1p_pair::<FMA>(Pair(w, w_lo + s_lo))
}

/// `exp(x) - t` as a pair: the float64 difference where the two are apart,
/// and where they all but cancel `t expm1(x - log t)`, the logarithm a pair
/// and so the difference exact, its `expm1` the float64 one beside the
/// trailing part's share.
#[inline(always)]
fn rate_less_count<const FMA: bool>(
    x: f64,
    t: f64,
    wide: bool,
    logs: &mut [crate::ops::arithmetic::extended::Pair; 64],
) -> crate::ops::arithmetic::extended::Pair {
    use crate::ops::arithmetic::extended::Pair;
    let e = x.exp();
    let plain = e - t;
    let near = t > 0.0 && t < 2f64.powi(1000) && x.abs() <= 708.0;
    if !(near && cancels(plain, e + t, wide)) {
        return Pair(plain, 0.0);
    }
    let log = match logs.get_mut(t as usize) {
        Some(log) if t.fract() == 0.0 => {
            if log.0.is_nan() {
                *log = log_pair::<FMA>(t, 0.0);
            }
            *log
        }
        _ => log_pair::<FMA>(t, 0.0),
    };
    let u = Pair(x, 0.0).add(log.neg());
    let rest = u.0.exp_m1();
    Pair(rest, u.1 * (1.0 + rest)).mul::<FMA>(Pair(t, 0.0))
}

/// `1 - t / (x + eps)`, as `((x - t) + eps) / (x + eps)`: `x - t` exact,
/// where the two cancel.
#[inline(always)]
fn rate_slope(x: f64, t: f64, eps: f64) -> f64 {
    use crate::ops::arithmetic::rounded_once::two_sum;
    if !(x.is_finite() && t.is_finite()) {
        return 1.0 - t / (x + eps);
    }
    let (d, d_lo) = two_sum(x, -t);
    let (n, n_lo) = two_sum(d, eps);
    (n + (n_lo + d_lo)) / (x + eps)
}

/// Refuse a negative Poisson target.
///
/// A count below zero has no likelihood, yet the loss is finite there -- the
/// log-rate form is linear in `target` -- so it trained toward an optimum that
/// means nothing. A NaN passes through, as it does in every loss.
fn ensure_counts(target: &Tensor) -> Result<()> {
    let target = target.contiguous()?;
    let negative = match target.dtype() {
        DataType::Float32 => target
            .data()
            .as_f32_slice()
            .is_some_and(|values| values.iter().any(|&v| v < 0.0)),
        DataType::Float64 => target
            .data()
            .as_f64_slice()
            .is_some_and(|values| values.iter().any(|&v| v < 0.0)),
        _ => false,
    };
    if negative {
        Err(MinitensorError::invalid_argument(
            "poisson_nll_loss requires every target to be a count, at least 0",
        ))
    } else {
        Ok(())
    }
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
        let counts = tensor(vec![0.0, 2.0, 1.0, 3.0], vec![2, 2]);

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
                Box::new(|t: &Tensor| poisson_nll_loss(t, &counts, true, false, 0.0, "sum")),
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
