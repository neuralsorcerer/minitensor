// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::ops::arithmetic::mul;
use crate::ops::map::{VECTOR_F32_PAR_THRESHOLD, outputs_per_task, par_out_chunks, unary_map};
use crate::{
    error::{MinitensorError, Result},
    ops::{comparison, selection::masked_fill_scalar},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

pub(crate) fn fill_one_hot_f64<T, F>(
    indices: &[T],
    out: &mut [f64],
    num_classes: usize,
    to_index: F,
) -> Result<()>
where
    F: Fn(&T) -> Result<usize>,
{
    for (i, value) in indices.iter().enumerate() {
        let class = to_index(value)?;
        out[i * num_classes + class] = 1.0;
    }
    Ok(())
}

/// Compute the sign of each tensor element (-1.0, 0.0, or 1.0)
pub(crate) fn sign_tensor(tensor: &Tensor) -> Result<Tensor> {
    // Producer kernel (no memset + raw-pointer loop). NaN propagates, matching
    // the public `sign` op and the L1 gradient.
    let output_data = match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            TensorData::from_vec::<f32>(
                unary_map(input_data, |v: f32| {
                    if v.is_nan() {
                        v
                    } else if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }),
                DataType::Float32,
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            TensorData::from_vec::<f64>(
                unary_map(input_data, |v: f64| {
                    if v.is_nan() {
                        v
                    } else if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }),
                DataType::Float64,
                tensor.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sign operation only supported for floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Sum all elements in a tensor to produce a scalar.
///
/// Through the whole-tensor `sum`'s own kernels: vectorised, folded pairwise
/// across 8192-element chunks, and so the same number `sum()` gives. Summed
/// here one element after another below the parallel threshold, it was a
/// chain of dependent adds -- 35us of a 50us `mse_loss` over 32K float32
/// values -- whose error grew with the length rather than its logarithm.
pub(crate) fn sum_all_elements(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());
    match tensor.dtype() {
        DataType::Float32 => crate::ops::reduction::sum_all_f32(tensor, &mut output_data)?,
        DataType::Float64 => crate::ops::reduction::sum_all_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sum only supported for floating point tensors",
            ));
        }
    }

    // Reduced losses are 0-dim scalars; a shape-[1] result breaks float(loss).
    Ok(Tensor::new(
        Arc::new(output_data),
        Shape::scalar(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Divide tensor by a scalar value
pub(crate) fn divide_by_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let scalar_f32 = scalar as f32;
            TensorData::from_vec::<f32>(
                unary_map(input_data, |v: f32| v / scalar_f32),
                DataType::Float32,
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            TensorData::from_vec::<f64>(
                unary_map(input_data, |v: f64| v / scalar),
                DataType::Float64,
                tensor.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Division only supported for floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Elementwise binary cross entropy from logits, `w` the positive weight
/// (one without it):
///
/// ```text
/// loss = (1 - t) x + c softplus(-x),   c = (1 - t) + w t
///      = k |x| + c log(1 + exp(-|x|)), k = 1 - t for x >= 0, w t below
/// ```
///
/// The second form is the same number, and for a target in `[0, 1]` and a
/// weight not negative its two terms are not negative: nothing cancels. The
/// first cancelled where the logit is negative and the target near zero --
/// `(1 - t) x` near `x` against `softplus(-x)` near `-x` -- and at a logit of
/// -20 against a target of 0 the loss of 2.06e-9 came out with the error of a
/// number near 20, some 10^9 ulps. `exp` is only ever taken of a number that
/// is not positive, so nothing overflows, and no sigmoid is formed to
/// saturate: the obvious `-t log(sigmoid(x)) - (1 - t) log(1 - sigmoid(x))`
/// reached `log(0)` by `|x| = 89` in float32 and had to be clamped, losing the
/// gradient.
///
/// A float32 loss is taken in float64 and narrowed once, through a
/// vectorized kernel where there is no weight. At an infinite logit the loss
/// is `k |x|`, and nothing where `k` is: a target of one against `+inf` is a
/// perfect prediction.
pub(crate) fn compute_bce_with_logits_elementwise(
    logits: &Tensor,
    targets: &Tensor,
    pos_weight: Option<&Tensor>,
) -> Result<Tensor> {
    let missing = || MinitensorError::internal_error("binary_cross_entropy_with_logits: a slice");
    let data = match logits.dtype() {
        DataType::Float32 => {
            let x = logits.data().as_f32_slice().ok_or_else(missing)?;
            let t = targets.data().as_f32_slice().ok_or_else(missing)?;
            let values = match pos_weight {
                None => {
                    let kernel = crate::ops::simd::F32Kernel::select();
                    // SAFETY: the block closure is one `F32Kernel` method
                    // call, which writes every element of the block.
                    unsafe {
                        crate::ops::map::binary_map_blocks_threshold(
                            x,
                            t,
                            VECTOR_F32_PAR_THRESHOLD,
                            crate::ops::map::PAR_CHUNK,
                            |x, t, out| kernel.bce_with_logits(x, t, out),
                        )
                    }
                }
                Some(w) => {
                    let w = w.data().as_f32_slice().ok_or_else(missing)?;
                    let mut values = vec![0f32; x.len()];
                    par_out_chunks(&mut values, outputs_per_task(3), &|offset, block| {
                        for (i, o) in block.iter_mut().enumerate() {
                            let at = offset + i;
                            let term = bce_logit_term(x[at].into(), t[at].into(), w[at].into());
                            *o = term as f32;
                        }
                    });
                    values
                }
            };
            TensorData::from_vec::<f32>(values, DataType::Float32, logits.device())
        }
        DataType::Float64 => {
            let x = logits.data().as_f64_slice().ok_or_else(missing)?;
            let t = targets.data().as_f64_slice().ok_or_else(missing)?;
            let w = pos_weight
                .map(|w| w.data().as_f64_slice().ok_or_else(missing))
                .transpose()?;
            let mut values = vec![0f64; x.len()];
            par_out_chunks(&mut values, outputs_per_task(3), &|offset, block| {
                for (i, o) in block.iter_mut().enumerate() {
                    let at = offset + i;
                    *o = bce_logit_term(x[at], t[at], w.map_or(1.0, |w| w[at]));
                }
            });
            TensorData::from_vec::<f64>(values, DataType::Float64, logits.device())
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "binary_cross_entropy_with_logits requires floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        logits.shape().clone(),
        logits.dtype(),
        logits.device(),
        false,
    ))
}

/// One term of [`compute_bce_with_logits_elementwise`] in float64:
/// `k |x| + c log(1 + exp(-|x|))`. A NaN logit is NaN whatever the
/// coefficients.
#[inline(always)]
fn bce_logit_term(x: f64, t: f64, w: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let magnitude = x.abs();
    let tail = (-magnitude).exp().ln_1p();
    let k = if x >= 0.0 { 1.0 - t } else { w * t };
    let c = (1.0 - t) + w * t;
    let linear = if k == 0.0 { 0.0 } else { k * magnitude };
    linear + if c == 0.0 { 0.0 } else { c * tail }
}

/// Negate tensor elements
fn negate(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            TensorData::from_vec::<f32>(
                unary_map(input_data, |value: f32| -value),
                DataType::Float32,
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            TensorData::from_vec::<f64>(
                unary_map(input_data, |value: f64| -value),
                DataType::Float64,
                tensor.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Negation only supported for floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

pub(crate) fn negative_log_likelihood(
    log_predictions: &Tensor,
    targets: &Tensor,
) -> Result<Tensor> {
    // Multiplying a zero target by a non-finite log-probability would produce
    // NaN (`0 * -inf`). Zero those positions before multiplication so only
    // classes with target mass contribute. This preserves +inf when the
    // target class itself has log-probability -inf and also supports soft
    // targets, where more than one class may carry non-zero mass.
    let zeros = Tensor::zeros(
        targets.shape().clone(),
        targets.dtype(),
        targets.device(),
        false,
    );
    let zero_targets = comparison::eq(targets, &zeros)?;
    let contributing_log_predictions = masked_fill_scalar(log_predictions, &zero_targets, 0.0)?;
    let likelihood = mul(&contributing_log_predictions, targets)?;
    negate(&likelihood)
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::test_support::tensor_of;

    #[test]
    fn test_mse_loss_mean() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5, 2.5], vec![3], false);

        let loss = mse_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: ((1.0-1.5)² + (2.0-2.5)² + (3.0-2.5)²) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert!((loss_data[0] - 0.25).abs() < 1e-6);
        // A reduced loss is a 0-dim scalar.
        assert_eq!(loss.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_mse_loss_sum() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![2.0, 3.0], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "sum").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (1.0-2.0)² + (2.0-3.0)² = 1.0 + 1.0 = 2.0
        assert!((loss_data[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_none() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![2.0, 3.0], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [(1.0-2.0)², (2.0-3.0)²] = [1.0, 1.0]
        assert!((loss_data[0] - 1.0).abs() < 1e-6);
        assert!((loss_data[1] - 1.0).abs() < 1e-6);
        assert_eq!(loss.shape().dims(), &[2]);
    }

    #[test]
    fn test_mae_loss_mean() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5, 2.0], vec![3], false);

        let loss = mae_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (|1.0-1.5| + |2.0-2.5| + |3.0-2.0|) / 3 = (0.5 + 0.5 + 1.0) / 3 = 2.0/3 ≈ 0.667
        assert!((loss_data[0] - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![1.2, 2.3], vec![2], false);

        // Delta = 1.0, differences are 0.2 and 0.3, both <= 1.0, so quadratic
        let loss = huber_loss(&predictions, &targets, 1.0, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [0.5 * 0.2², 0.5 * 0.3²] = [0.02, 0.045]
        assert!((loss_data[0] - 0.02).abs() < 1e-6);
        assert!((loss_data[1] - 0.045).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![3.0, 0.0], vec![2], false);

        // Delta = 1.0, differences are 2.0 and 2.0, both > 1.0, so linear
        let loss = huber_loss(&predictions, &targets, 1.0, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [1.0 * (2.0 - 0.5 * 1.0), 1.0 * (2.0 - 0.5 * 1.0)] = [1.5, 1.5]
        assert!((loss_data[0] - 1.5).abs() < 1e-6);
        assert!((loss_data[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss_mean_and_backward() {
        let predictions = tensor_of::<f32>(vec![0.8, 0.2], vec![2], true);
        let targets = tensor_of::<f32>(vec![1.0, 0.0], vec![2], false);

        let loss = binary_cross_entropy_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        let expected = -((0.8f32).ln() + (0.8f32).ln()) / 2.0;
        assert!((loss_val - expected).abs() < 1e-6);

        let grads = crate::autograd::backward_collect(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        let expected_grad = [-(1.0 / 0.8) / 2.0, (1.0 / 0.8) / 2.0];
        assert!((grad_slice[0] - expected_grad[0]).abs() < 1e-6);
        assert!((grad_slice[1] - expected_grad[1]).abs() < 1e-6);
    }

    /// `(1-t)*x + (1+(w-1)*t) * -log(sigmoid(x))`, in f64 so it can serve as
    /// the reference for the f32 kernel.
    fn bce_logits_ref(x: f64, t: f64, w: f64) -> f64 {
        (1.0 - t) * x + (1.0 + (w - 1.0) * t) * ((-x.abs()).exp().ln_1p() + (-x).max(0.0))
    }

    #[test]
    fn test_bce_with_logits_matches_reference_across_logit_range() {
        for &x in &[
            -100.0f32, -50.0, -30.0, -8.0, -1.0, 0.0, 1.0, 8.0, 30.0, 100.0,
        ] {
            for &t in &[0.0f32, 0.3, 1.0] {
                let logits = tensor_of::<f32>(vec![x], vec![1], true);
                let targets = tensor_of::<f32>(vec![t], vec![1], false);
                let loss =
                    binary_cross_entropy_with_logits_loss(&logits, &targets, None, "sum").unwrap();

                let got = loss.data().as_f32_slice().unwrap()[0];
                let want = bce_logits_ref(x as f64, t as f64, 1.0) as f32;
                assert!(got.is_finite(), "x={x} t={t} loss={got}");
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "x={x} t={t}: loss {got} != {want}"
                );

                // d/dx = sigmoid(x) - t, exact at every magnitude.
                let grads = crate::autograd::backward_collect(&loss, None).unwrap();
                let grad = grads
                    .get(&logits.id())
                    .unwrap()
                    .data()
                    .as_f32_slice()
                    .unwrap()[0];
                let sigmoid = (1.0f64 / (1.0 + (-(x as f64)).exp())) as f32;
                assert!(
                    (grad - (sigmoid - t)).abs() <= 1e-6,
                    "x={x} t={t}: grad {grad} != {}",
                    sigmoid - t
                );
            }
        }
    }

    #[test]
    fn test_bce_with_logits_keeps_gradient_where_sigmoid_bce_loses_it() {
        // A logit of -30 against a target of 1 is a confident and completely
        // wrong prediction: the gradient should be -1, the strongest signal the
        // loss can produce. Routing through sigmoid first lets it round to
        // ~9.4e-14 in f32, and the BCE backward's clamped denominator then
        // returns roughly -0.09 instead -- the example all but stops training.
        for &x in &[-30.0f32, -50.0, -100.0] {
            let targets = tensor_of::<f32>(vec![1.0], vec![1], false);

            let logits = tensor_of::<f32>(vec![x], vec![1], true);
            let fused =
                binary_cross_entropy_with_logits_loss(&logits, &targets, None, "sum").unwrap();
            let fused_grad = crate::autograd::backward_collect(&fused, None).unwrap()[&logits.id()]
                .data()
                .as_f32_slice()
                .unwrap()[0];
            assert!(
                (fused_grad + 1.0).abs() <= 1e-6,
                "x={x}: fused grad {fused_grad} should be -1"
            );

            let probs_input = tensor_of::<f32>(vec![x], vec![1], true);
            let probs = crate::ops::activation::sigmoid(&probs_input).unwrap();
            let split = binary_cross_entropy_loss(&probs, &targets, "sum").unwrap();
            let split_grad = crate::autograd::backward_collect(&split, None).unwrap()
                [&probs_input.id()]
                .data()
                .as_f32_slice()
                .unwrap()[0];
            assert!(
                split_grad.abs() < 0.5,
                "x={x}: sigmoid+BCE unexpectedly kept its gradient ({split_grad}); if that \
                 path was fixed this test no longer demonstrates anything"
            );
        }
    }

    #[test]
    fn test_bce_with_logits_pos_weight_broadcasts_and_reduces() {
        let logits_v = vec![-2.0f32, 0.5, 3.0, 1.0, -1.5, 0.0];
        let targets_v = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0];
        let weights_v = vec![0.5f32, 2.0, 3.0]; // per column, broadcast over rows

        let expected: Vec<f64> = (0..6)
            .map(|i| {
                bce_logits_ref(
                    logits_v[i] as f64,
                    targets_v[i] as f64,
                    weights_v[i % 3] as f64,
                )
            })
            .collect();

        let logits = tensor_of::<f32>(logits_v, vec![2, 3], false);
        let targets = tensor_of::<f32>(targets_v, vec![2, 3], false);
        let weights = tensor_of::<f32>(weights_v, vec![3], false);

        let none = binary_cross_entropy_with_logits_loss(&logits, &targets, Some(&weights), "none")
            .unwrap();
        for (got, want) in none
            .data()
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip(expected.iter())
        {
            assert!((got - *want as f32).abs() < 1e-5, "{got} != {want}");
        }

        let sum = binary_cross_entropy_with_logits_loss(&logits, &targets, Some(&weights), "sum")
            .unwrap();
        let want_sum: f64 = expected.iter().sum();
        assert!((sum.data().as_f32_slice().unwrap()[0] - want_sum as f32).abs() < 1e-4);

        let mean = binary_cross_entropy_with_logits_loss(&logits, &targets, Some(&weights), "mean")
            .unwrap();
        let want_mean = want_sum / 6.0;
        assert!((mean.data().as_f32_slice().unwrap()[0] - want_mean as f32).abs() < 1e-5);
    }

    #[test]
    fn test_bce_with_logits_rejects_non_broadcastable_pos_weight() {
        let logits = tensor_of::<f32>(vec![0.4, -0.7], vec![2, 1], false);
        let targets = tensor_of::<f32>(vec![1.0, 0.0], vec![2, 1], false);
        let weights = tensor_of::<f32>(vec![1.0, 2.0, 3.0], vec![3], false);
        assert!(
            binary_cross_entropy_with_logits_loss(&logits, &targets, Some(&weights), "mean")
                .is_err()
        );
    }

    #[test]
    fn test_bce_saturated_prediction_stays_finite() {
        // BCE clamps its log outputs to >= -100 (forward) and its backward
        // denominator to >= 1e-12, so a saturated prediction (exactly 0 or 1)
        // yields a finite loss and gradient instead of inf/nan.
        let predictions = tensor_of::<f32>(vec![0.0, 1.0], vec![2], true);
        let targets = tensor_of::<f32>(vec![1.0, 0.0], vec![2], false);

        let loss = binary_cross_entropy_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        // Each element contributes -log(0) -> clamped to 100; mean = 100.
        assert!(loss_val.is_finite(), "loss={loss_val}");
        assert!((loss_val - 100.0).abs() < 1e-3, "loss={loss_val}");

        let grads = crate::autograd::backward_collect(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        assert!(
            grad_slice.iter().all(|g| g.is_finite()),
            "grads={grad_slice:?}"
        );
    }

    #[test]
    fn test_kl_div_loss_mean_and_backward() {
        // This test used to assert an undivided forward against a gradient
        // divided by 2, which is exactly the forward/backward disagreement
        // `mean` had: the forward divided by the batch dimension (1, for a 1-D
        // tensor) while the backward divided by the element count.
        let predictions = tensor_of::<f32>(vec![0.4, 0.6], vec![2], true);
        let targets = tensor_of::<f32>(vec![0.5, 0.5], vec![2], false);

        let elementwise = 0.5 * (0.5f32.ln() - 0.4f32.ln()) + 0.5 * (0.5f32.ln() - 0.6f32.ln());

        let loss = kl_div_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        assert!((loss_val - elementwise / 2.0).abs() < 1e-6, "{loss_val}");

        let grads = crate::autograd::backward_collect(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        let expected_grad = [-(0.5 / 0.4) / 2.0, -(0.5 / 0.6) / 2.0];
        assert!((grad_slice[0] - expected_grad[0]).abs() < 1e-6);
        assert!((grad_slice[1] - expected_grad[1]).abs() < 1e-6);
    }

    #[test]
    fn kl_div_treats_a_zero_target_as_contributing_nothing() {
        // `target * (log target - log prediction)` is `0 * -inf` where the
        // target is zero, and one NaN term takes the whole reduction with it.
        // A one-hot target is nothing but zeros and a one, so this made the
        // most common target a classifier has return NaN.
        let predictions = tensor_of::<f32>(vec![0.1, 0.2, 0.3, 0.4], vec![4], true);
        let targets = tensor_of::<f32>(vec![0.0, 0.0, 1.0, 0.0], vec![4], false);

        let loss = kl_div_loss(&predictions, &targets, "sum").unwrap();
        let value = loss.data().as_f32_slice().unwrap()[0];
        assert!((value - -0.3f32.ln()).abs() < 1e-6, "{value}");

        // A zero target opposite a zero prediction is `-inf - -inf`, which is
        // NaN before the mask rather than an infinity.
        let both_zero = tensor_of::<f32>(vec![0.0, 1.0], vec![2], false);
        let loss = kl_div_loss(&both_zero, &both_zero, "sum").unwrap();
        assert_eq!(loss.data().as_f32_slice().unwrap()[0], 0.0);

        // But a live target against a zero prediction still diverges.
        let live = tensor_of::<f32>(vec![0.5, 0.5], vec![2], false);
        let loss = kl_div_loss(&both_zero, &live, "sum").unwrap();
        assert!(loss.data().as_f32_slice().unwrap()[0].is_infinite());
    }

    #[test]
    fn kl_div_batchmean_divides_by_the_leading_dimension() {
        let predictions = tensor_of::<f32>(vec![0.4, 0.6, 0.3, 0.7], vec![2, 2], true);
        let targets = tensor_of::<f32>(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2], false);

        let sum = kl_div_loss(&predictions, &targets, "sum").unwrap();
        let sum_val = sum.data().as_f32_slice().unwrap()[0];

        let batchmean = kl_div_loss(&predictions, &targets, "batchmean").unwrap();
        assert!((batchmean.data().as_f32_slice().unwrap()[0] - sum_val / 2.0).abs() < 1e-6);

        let mean = kl_div_loss(&predictions, &targets, "mean").unwrap();
        assert!((mean.data().as_f32_slice().unwrap()[0] - sum_val / 4.0).abs() < 1e-6);

        // The gradient must follow whichever divisor the forward used. Each
        // reduction gets its own input: gradients accumulate per tensor id, so
        // reusing one would sum the three backward passes.
        for (reduction, divisor) in [("mean", 4.0f32), ("batchmean", 2.0), ("sum", 1.0)] {
            let inputs = tensor_of::<f32>(vec![0.4, 0.6, 0.3, 0.7], vec![2, 2], true);
            let loss = kl_div_loss(&inputs, &targets, reduction).unwrap();
            let grads = crate::autograd::backward_collect(&loss, None).unwrap();
            let grad = grads.get(&inputs.id()).unwrap();
            let first = grad.data().as_f32_slice().unwrap()[0];
            assert!(
                (first - -(0.5 / 0.4) / divisor).abs() < 1e-6,
                "{reduction}: {first}"
            );
        }
    }

    #[test]
    fn test_loss_gradient_tracking() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], true);
        let targets = tensor_of::<f32>(vec![1.5, 2.5], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "mean").unwrap();

        assert!(loss.requires_grad());
        assert!(loss.grad_fn().is_some());
    }

    #[test]
    fn test_loss_input_validation() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5, 3.5], vec![3], false);

        // Shape mismatch should fail
        let result = mse_loss(&predictions, &targets, "mean");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_reduction_mode() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5], vec![2], false);

        let result = mse_loss(&predictions, &targets, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_huber_loss_invalid_delta() {
        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5], vec![2], false);

        let result = huber_loss(&predictions, &targets, -1.0, "mean");
        assert!(result.is_err());
    }

    #[test]
    fn test_smooth_l1_loss_matches_huber() {
        let predictions = tensor_of::<f32>(vec![0.5, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![0.0, 0.0], vec![2], false);

        let smooth = smooth_l1_loss(&predictions, &targets, 1.0, "none").unwrap();
        let huber = huber_loss(&predictions, &targets, 1.0, "none").unwrap();

        let smooth_data = smooth.data().as_f32_slice().unwrap();
        let huber_data = huber.data().as_f32_slice().unwrap();
        assert!((smooth_data[0] - huber_data[0]).abs() < 1e-6);
        assert!((smooth_data[1] - huber_data[1]).abs() < 1e-6);
    }

    #[test]
    fn smooth_l1_is_huber_scaled_by_beta_away_from_one() {
        // The two agree only at 1.0, so implementing smooth-l1 as a bare
        // `huber_loss(.., beta, ..)` is right for the default and wrong
        // everywhere else: huber(x, d) == d * smooth_l1(x, beta = d).
        let predictions = tensor_of::<f32>(vec![0.25, 1.0, 4.0], vec![3], false);
        let targets = tensor_of::<f32>(vec![0.0, 0.0, 0.0], vec![3], false);

        for beta in [0.5f32, 1.0, 2.0, 5.0] {
            let smooth = smooth_l1_loss(&predictions, &targets, beta as f64, "none").unwrap();
            let huber = huber_loss(&predictions, &targets, beta as f64, "none").unwrap();
            let smooth_data = smooth.data().as_f32_slice().unwrap();
            let huber_data = huber.data().as_f32_slice().unwrap();

            for (i, x) in [0.25f32, 1.0, 4.0].into_iter().enumerate() {
                let expected = if x < beta {
                    0.5 * x * x / beta
                } else {
                    x - 0.5 * beta
                };
                assert!(
                    (smooth_data[i] - expected).abs() < 1e-6,
                    "beta={beta} x={x}: {} != {expected}",
                    smooth_data[i]
                );
                assert!(
                    (huber_data[i] - beta * smooth_data[i]).abs() < 1e-5,
                    "beta={beta}"
                );
            }
        }

        // At zero there is no quadratic region left: the L1 loss exactly.
        let at_zero = smooth_l1_loss(&predictions, &targets, 0.0, "none").unwrap();
        let l1 = crate::ops::loss::mae_loss(&predictions, &targets, "none").unwrap();
        assert_eq!(
            at_zero.data().as_f32_slice().unwrap(),
            l1.data().as_f32_slice().unwrap()
        );
        assert!(smooth_l1_loss(&predictions, &targets, -1.0, "none").is_err());
        assert!(smooth_l1_loss(&predictions, &targets, f64::NAN, "none").is_err());
        assert!(smooth_l1_loss(&predictions, &targets, f64::INFINITY, "none").is_err());
    }

    #[test]
    fn test_log_cosh_loss_mean() {
        let predictions = tensor_of::<f32>(vec![0.0, 1.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![0.0, 0.0], vec![2], false);

        let loss = log_cosh_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        let expected = (0.0f32.cosh().ln() + 1.0f32.cosh().ln()) / 2.0;
        assert!((loss_data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_log_cosh_loss_invalid_reduction() {
        let predictions = tensor_of::<f32>(vec![0.0], vec![1], false);
        let targets = tensor_of::<f32>(vec![0.0], vec![1], false);

        let result = log_cosh_loss(&predictions, &targets, "invalid");
        assert!(result.is_err());
    }
}
