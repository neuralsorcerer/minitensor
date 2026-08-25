// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Connectionist temporal classification.
//!
//! The loss for a model that emits one distribution per input step but is
//! trained against a target that is shorter and unaligned: speech against text,
//! handwriting against characters, a spectrogram against a phoneme sequence.
//! Nothing says which input step produced which symbol, so the loss is the
//! total probability of *every* alignment that collapses to the target.
//!
//! It is the one loss here that cannot be written as an expression over
//! tensors. There are exponentially many alignments, and summing them is a
//! dynamic program over the time axis -- a recurrence whose step depends on the
//! previous step's result. No arrangement of the elementwise, reduction and
//! contraction operations in this library runs a recurrence, so this is a loop
//! or it is nothing.
//!
//! ## The alignment
//!
//! A path is one class per input step. Collapsing a path merges adjacent equal
//! classes and then deletes the blank, so `a a _ a b` collapses to `a a b`: the
//! blank is what lets a target repeat a symbol. Writing the target with a blank
//! between every symbol and at both ends gives the *extended* sequence, of
//! length `2S + 1`, and every collapsing path is a monotone walk through it
//! that may stay put, step once, or skip a blank between two different symbols.
//! The forward recursion sums those walks.
//!
//! ## Why it is all in the log domain
//!
//! A path probability is a product of `T` numbers below one, so it underflows
//! `f32` after a few dozen steps and `f64` after a few hundred -- and a real
//! utterance is thousands of steps. The recursion therefore runs on log
//! probabilities throughout, adding where the definition multiplies and using
//! [`log_add_exp`] where it adds -- the same one the elementwise `logaddexp`
//! and the cumulative form use, rather than a third copy. It never forms either
//! exponential and never subtracts, so nothing cancels.
//!
//! The whole computation is in `f64` regardless of the input's dtype. It is
//! `O(T x S)` per sample against the `O(T x C)` of the layer that produced the
//! input, so a wider accumulator costs nothing measurable and removes the one
//! place where `f32` would show.
//!
//! ## Both directions, once
//!
//! The gradient of `-log p` with respect to the log probabilities is
//! `-alpha_t(s) beta_t(s) / p` summed over the extended positions holding each
//! class, where `beta` is the same recursion run backwards. So the backward
//! pass is not a second traversal of a recorded graph: it is one more sweep of
//! the same shape as the forward, done here, and the result is stored. That is
//! why this loss carries a hand-written gradient node like the others in this
//! directory rather than being differentiated by the engine.

use super::regression_impl::manual_backward_needed;
use crate::{
    autograd::{CtcLossBackward, NoGradGuard, with_grad_fn},
    error::{MinitensorError, Result},
    ops::util::log_add_exp,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

const NEG_INF: f64 = f64::NEG_INFINITY;

/// One sample's forward-backward over the extended label sequence.
///
/// `logs` holds `steps` rows of `classes` log probabilities. `extended` is the
/// target with blanks interleaved. Returns `-log p` and, when asked, the
/// gradient of that with respect to `logs`, in the same layout.
///
/// An unreachable target -- one needing more steps than the input has -- has
/// probability zero, so the loss is infinite and the gradient does not exist.
/// Zeros are returned for it rather than the `inf - inf` the recursion would
/// otherwise produce.
fn align(
    logs: &[f64],
    extended: &[usize],
    steps: usize,
    classes: usize,
    want_gradient: bool,
) -> (f64, Vec<f64>) {
    let width = extended.len();
    let empty = Vec::new();
    if steps == 0 {
        // No input steps: only the empty target is reachable, by the empty path.
        return (if width == 1 { 0.0 } else { f64::INFINITY }, empty);
    }

    // Whether position `s` can be reached from `s - 2` directly. Only a symbol
    // may skip, and only over a blank separating it from a *different* symbol --
    // otherwise the two would collapse into one.
    //
    // The usual statement of this also asks that `extended[s]` is not the
    // blank, and that test can never fail here: the sequence alternates, so an
    // even `s` has a blank at `s - 2` as well and is stopped by the inequality,
    // while an odd one holds a target symbol, which is never the blank because
    // a target containing it is rejected above.
    let skippable: Vec<bool> = (0..width)
        .map(|s| s >= 2 && extended[s] != extended[s - 2])
        .collect();

    let mut alpha = vec![NEG_INF; steps * width];
    alpha[0] = logs[extended[0]];
    if width > 1 {
        alpha[1] = logs[extended[1]];
    }
    for step in 1..steps {
        let (earlier, current) = alpha.split_at_mut(step * width);
        let previous = &earlier[(step - 1) * width..];
        let row = &logs[step * classes..(step + 1) * classes];
        for s in 0..width {
            let mut total = previous[s];
            if s >= 1 {
                total = log_add_exp(total, previous[s - 1]);
            }
            if skippable[s] {
                total = log_add_exp(total, previous[s - 2]);
            }
            current[s] = total + row[extended[s]];
        }
    }

    // A path ends on the last symbol or on the blank after it.
    let last = (steps - 1) * width;
    let mut evidence = alpha[last + width - 1];
    if width > 1 {
        evidence = log_add_exp(evidence, alpha[last + width - 2]);
    }
    if !evidence.is_finite() || !want_gradient {
        return (-evidence, empty);
    }

    let mut beta = vec![NEG_INF; steps * width];
    beta[last + width - 1] = 0.0;
    if width > 1 {
        beta[last + width - 2] = 0.0;
    }
    for step in (0..steps - 1).rev() {
        let (current, later) = beta.split_at_mut((step + 1) * width);
        let current = &mut current[step * width..];
        let row = &logs[(step + 1) * classes..(step + 2) * classes];
        for s in 0..width {
            let mut total = later[s] + row[extended[s]];
            if s + 1 < width {
                total = log_add_exp(total, later[s + 1] + row[extended[s + 1]]);
            }
            if s + 2 < width && skippable[s + 2] {
                total = log_add_exp(total, later[s + 2] + row[extended[s + 2]]);
            }
            current[s] = total;
        }
    }

    // d(-log p) / d(log y_t^k) = -sum over the extended positions holding k of
    // alpha_t(s) beta_t(s) / p. Every term is positive, so accumulating the
    // ratios directly cannot cancel; and they sum to one across each step,
    // which is what the tests check.
    let mut gradient = vec![0.0; steps * classes];
    for step in 0..steps {
        let base = step * width;
        let row = &mut gradient[step * classes..(step + 1) * classes];
        for s in 0..width {
            let share = alpha[base + s] + beta[base + s] - evidence;
            if share > NEG_INF {
                row[extended[s]] -= share.exp();
            }
        }
    }
    (-evidence, gradient)
}

/// Read a length vector as `usize`, whatever integer dtype it arrived in.
fn lengths_of(tensor: &Tensor, batch: usize, name: &str) -> Result<Vec<usize>> {
    if tensor.ndim() != 1 || tensor.shape().dims()[0] != batch {
        return Err(MinitensorError::invalid_operation(format!(
            "ctc_loss: {name} must be a vector of one length per batch element, {} of them",
            batch
        )));
    }
    let contiguous = tensor.contiguous()?;
    let read = |value: i64| -> Result<usize> {
        usize::try_from(value).map_err(|_| {
            MinitensorError::invalid_operation(format!("ctc_loss: {name} cannot be negative"))
        })
    };
    match tensor.dtype() {
        DataType::Int32 => contiguous
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("ctc_loss: length dtype mismatch"))?
            .iter()
            .map(|&value| read(value as i64))
            .collect(),
        DataType::Int64 => contiguous
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("ctc_loss: length dtype mismatch"))?
            .iter()
            .map(|&value| read(value))
            .collect(),
        _ => Err(MinitensorError::invalid_operation(format!(
            "ctc_loss: {name} must be an integer tensor"
        ))),
    }
}

/// Read the targets as one row per batch element, from either accepted layout.
///
/// `(N, S)` is padded -- each row is read up to its own target length and the
/// rest ignored -- and a flat vector is the rows concatenated, which is what a
/// caller with wildly uneven targets wants rather than padding to the longest.
fn targets_of(tensor: &Tensor, lengths: &[usize]) -> Result<Vec<Vec<usize>>> {
    let contiguous = tensor.contiguous()?;
    let flat: Vec<usize> = match tensor.dtype() {
        DataType::Int32 => contiguous
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("ctc_loss: target dtype mismatch"))?
            .iter()
            .map(|&value| value.max(0) as usize)
            .collect(),
        DataType::Int64 => contiguous
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("ctc_loss: target dtype mismatch"))?
            .iter()
            .map(|&value| value.max(0) as usize)
            .collect(),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "ctc_loss: targets must be an integer tensor",
            ));
        }
    };

    let dims = tensor.shape().dims();
    match dims.len() {
        2 => {
            let (batch, padded) = (dims[0], dims[1]);
            if batch != lengths.len() {
                return Err(MinitensorError::invalid_operation(
                    "ctc_loss: a two-dimensional targets tensor needs one row per batch element",
                ));
            }
            lengths
                .iter()
                .enumerate()
                .map(|(row, &length)| {
                    if length > padded {
                        return Err(MinitensorError::invalid_operation(format!(
                            "ctc_loss: target length {length} exceeds the {padded} columns of the targets tensor"
                        )));
                    }
                    Ok(flat[row * padded..row * padded + length].to_vec())
                })
                .collect()
        }
        1 => {
            let total: usize = lengths.iter().sum();
            if total != flat.len() {
                return Err(MinitensorError::invalid_operation(format!(
                    "ctc_loss: the concatenated targets hold {} entries but the lengths add up to {total}",
                    flat.len()
                )));
            }
            let mut cut = 0;
            Ok(lengths
                .iter()
                .map(|&length| {
                    let row = flat[cut..cut + length].to_vec();
                    cut += length;
                    row
                })
                .collect())
        }
        _ => Err(MinitensorError::invalid_operation(
            "ctc_loss: targets must be a padded (batch, length) tensor or the rows concatenated into a vector",
        )),
    }
}

/// The connectionist temporal classification loss.
///
/// `log_probs` is `(steps, batch, classes)` and is expected to be log
/// probabilities already -- the output of a `log_softmax` over the class axis.
/// Nothing normalises it here, deliberately: a caller who has a numerically
/// careful log-softmax should not have it undone and redone, and a caller who
/// passes raw scores would get a plausible-looking wrong answer either way.
///
/// `blank` is the class standing for "emit nothing", which is why a target may
/// not contain it.
///
/// `reduction` is `"none"` for one loss per batch element, `"sum"` for their
/// total, or `"mean"` for the average of each divided by its own target length.
/// That division is what makes the mean comparable across batches of unequal
/// targets, and it is why `"mean"` is not the mean of what `"none"` returns.
///
/// `zero_infinity` replaces an unreachable target's infinite loss, and its
/// gradient, with zero. Such a target is a data problem rather than a modelling
/// one -- the input is too short to spell it -- and left alone a single one
/// poisons the whole batch's gradient.
#[allow(clippy::too_many_arguments)]
pub fn ctc_loss(
    log_probs: &Tensor,
    targets: &Tensor,
    input_lengths: &Tensor,
    target_lengths: &Tensor,
    blank: usize,
    reduction: &str,
    zero_infinity: bool,
) -> Result<Tensor> {
    if !matches!(reduction, "none" | "sum" | "mean") {
        // Checked here rather than after the fact: the forward-backward pass
        // below is the expensive part, and there is no point running it for a
        // mode that cannot be applied to the result.
        return Err(crate::ops::loss::invalid_reduction(reduction, false));
    }
    if log_probs.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "ctc_loss: log_probs must be (steps, batch, classes)",
        ));
    }
    if !matches!(log_probs.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "ctc_loss: log_probs must be a floating point tensor",
        ));
    }
    let dims = log_probs.shape().dims().to_vec();
    let (steps, batch, classes) = (dims[0], dims[1], dims[2]);
    if blank >= classes {
        return Err(MinitensorError::invalid_operation(format!(
            "ctc_loss: blank index {blank} is outside the {classes} classes"
        )));
    }

    let input_len = lengths_of(input_lengths, batch, "input_lengths")?;
    let target_len = lengths_of(target_lengths, batch, "target_lengths")?;
    if let Some(&long) = input_len.iter().find(|&&length| length > steps) {
        return Err(MinitensorError::invalid_operation(format!(
            "ctc_loss: an input length of {long} exceeds the {steps} steps provided"
        )));
    }
    let rows = targets_of(targets, &target_len)?;
    for row in &rows {
        if let Some(&symbol) = row.iter().find(|&&symbol| symbol >= classes) {
            return Err(MinitensorError::invalid_operation(format!(
                "ctc_loss: target class {symbol} is outside the {classes} classes"
            )));
        }
        if row.contains(&blank) {
            return Err(MinitensorError::invalid_operation(format!(
                "ctc_loss: a target may not contain the blank class {blank}, which stands for emitting nothing"
            )));
        }
    }

    // The recursion reads one contiguous (steps, classes) plane per batch
    // element, so the time-major input is transposed once here rather than
    // strided through steps x width times.
    let contiguous = log_probs.contiguous()?;
    let mut planes = vec![0.0f64; batch * steps * classes];
    match log_probs.dtype() {
        DataType::Float32 => {
            let source = contiguous
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("ctc_loss: dtype mismatch"))?;
            transpose_planes(source, &mut planes, steps, batch, classes, |v| v as f64);
        }
        _ => {
            let source = contiguous
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("ctc_loss: dtype mismatch"))?;
            transpose_planes(source, &mut planes, steps, batch, classes, |v| v);
        }
    }

    let want_gradient = manual_backward_needed(&[log_probs]);
    let plane = steps * classes;
    let found: Vec<(f64, Vec<f64>)> = planes
        .par_chunks(plane)
        .enumerate()
        .map(|(index, logs)| {
            let mut extended = Vec::with_capacity(2 * rows[index].len() + 1);
            extended.push(blank);
            for &symbol in &rows[index] {
                extended.push(symbol);
                extended.push(blank);
            }
            let (loss, gradient) = align(
                &logs[..input_len[index] * classes],
                &extended,
                input_len[index],
                classes,
                want_gradient,
            );
            if zero_infinity && !loss.is_finite() {
                (0.0, Vec::new())
            } else {
                (loss, gradient)
            }
        })
        .collect();

    let device = log_probs.device();
    let dtype = log_probs.dtype();
    let losses: Vec<f64> = found.iter().map(|(loss, _)| *loss).collect();

    let scales: Vec<f64> = match reduction {
        "none" => vec![1.0; batch],
        "sum" => vec![1.0; batch],
        // Dividing by the target length is what makes the average comparable
        // across batches; an empty target divides by one rather than by zero.
        _ => target_len
            .iter()
            .map(|&length| 1.0 / (length.max(1) * batch) as f64)
            .collect(),
    };

    let value = match reduction {
        "none" => losses.clone(),
        "sum" => vec![losses.iter().sum()],
        _ => vec![
            losses
                .iter()
                .zip(&scales)
                .map(|(loss, scale)| loss * scale)
                .sum(),
        ],
    };
    let shape = if reduction == "none" {
        Shape::new(vec![batch])
    } else {
        Shape::new(vec![])
    };

    let loss = {
        let _guard = NoGradGuard::new();
        let mut data = TensorData::zeros_on_device(value.len(), dtype, device);
        match dtype {
            DataType::Float32 => {
                let out = data
                    .as_f32_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("ctc_loss: dtype mismatch"))?;
                for (slot, item) in out.iter_mut().zip(&value) {
                    *slot = *item as f32;
                }
            }
            _ => {
                let out = data
                    .as_f64_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("ctc_loss: dtype mismatch"))?;
                out.copy_from_slice(&value);
            }
        }
        Tensor::new(Arc::new(data), shape, dtype, device, false)
    };

    if !want_gradient {
        return Ok(loss);
    }

    // Back to time-major, with the per-sample scaling of the reduction already
    // applied. Steps beyond a sample's own input length contributed nothing to
    // the loss and stay zero -- and so does a sample with no gradient at all,
    // which is an unreachable target, or one `zero_infinity` has cleared.
    let mut spread = vec![0.0f64; steps * batch * classes];
    for (index, (_, gradient)) in found.iter().enumerate() {
        if gradient.is_empty() {
            continue;
        }
        let scale = scales[index];
        for step in 0..input_len[index] {
            let from = &gradient[step * classes..(step + 1) * classes];
            let into = &mut spread
                [(step * batch + index) * classes..(step * batch + index) * classes + classes];
            for (slot, item) in into.iter_mut().zip(from) {
                *slot = item * scale;
            }
        }
    }

    let mut data = TensorData::zeros_on_device(spread.len(), dtype, device);
    match dtype {
        DataType::Float32 => {
            let out = data
                .as_f32_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("ctc_loss: dtype mismatch"))?;
            for (slot, item) in out.iter_mut().zip(&spread) {
                *slot = *item as f32;
            }
        }
        _ => {
            let out = data
                .as_f64_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("ctc_loss: dtype mismatch"))?;
            out.copy_from_slice(&spread);
        }
    }
    let gradient = Tensor::new(Arc::new(data), Shape::new(dims), dtype, device, false);

    let grad_fn = Arc::new(CtcLossBackward {
        input_id: log_probs.id(),
        reduction: reduction.to_string(),
        gradient,
    });
    with_grad_fn(loss.requires_grad_(true), grad_fn)
}

/// `(steps, batch, classes)` to `batch` contiguous `(steps, classes)` planes.
fn transpose_planes<T: Copy, F: Fn(T) -> f64>(
    source: &[T],
    into: &mut [f64],
    steps: usize,
    batch: usize,
    classes: usize,
    widen: F,
) {
    for step in 0..steps {
        for index in 0..batch {
            let from = (step * batch + index) * classes;
            let to = (index * steps + step) * classes;
            for offset in 0..classes {
                into[to + offset] = widen(source[from + offset]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `log_add` would be if underflow were not a concern.
    fn naively(a: f64, b: f64) -> f64 {
        (a.exp() + b.exp()).ln()
    }

    #[test]
    fn log_add_agrees_with_the_naive_form_where_that_form_works() {
        for &(a, b) in &[
            (0.0, 0.0),
            (-1.0, -2.0),
            (-0.5, -30.0),
            (-12.25, -12.25),
            (-100.0, -101.5),
        ] {
            assert!((log_add_exp(a, b) - naively(a, b)).abs() < 1e-12);
        }
    }

    #[test]
    fn log_add_is_symmetric() {
        for &(a, b) in &[(-1.0, -900.0), (-900.0, -1.0), (-3.5, -3.5)] {
            assert_eq!(log_add_exp(a, b), log_add_exp(b, a));
        }
    }

    #[test]
    fn log_add_survives_where_the_naive_form_underflows() {
        // Both terms are far below the smallest positive double, so `exp` gives
        // zero for each and the naive form answers negative infinity. The whole
        // reason the recursion is in the log domain.
        let (a, b) = (-5000.0, -5001.0);
        assert!(naively(a, b).is_infinite());
        let got = log_add_exp(a, b);
        assert!((got - (a + (1.0f64 + (-1.0f64).exp()).ln())).abs() < 1e-12);
    }

    #[test]
    fn an_impossible_term_contributes_nothing() {
        assert_eq!(log_add_exp(NEG_INF, -3.0), -3.0);
        assert_eq!(log_add_exp(-3.0, NEG_INF), -3.0);
        assert_eq!(log_add_exp(NEG_INF, NEG_INF), NEG_INF);
    }

    #[test]
    fn an_empty_target_is_the_all_blank_path() {
        // Two steps, two classes, blank 0: the only path is blank, blank.
        let logs = [(0.25f64).ln(), (0.75f64).ln(), (0.5f64).ln(), (0.5f64).ln()];
        let (loss, _) = align(&logs, &[0], 2, 2, false);
        assert!((loss - -((0.25f64 * 0.5).ln())).abs() < 1e-12);
    }

    #[test]
    fn a_repeat_needs_a_blank_between_the_two() {
        // `[1, 1]` cannot be spelled in two steps -- `1 1` collapses to `1`.
        let logs = [(0.5f64).ln(); 4];
        let (loss, _) = align(&logs, &[0, 1, 0, 1, 0], 2, 2, false);
        assert!(loss.is_infinite());
    }

    #[test]
    fn the_gradient_of_one_step_sums_to_minus_one() {
        let logs: Vec<f64> = [0.2f64, 0.3, 0.5, 0.4, 0.4, 0.2, 0.1, 0.6, 0.3]
            .iter()
            .map(|p| p.ln())
            .collect();
        let (_, gradient) = align(&logs, &[0, 1, 0, 2, 0], 3, 3, true);
        for step in 0..3 {
            let total: f64 = gradient[step * 3..(step + 1) * 3].iter().sum();
            assert!((total + 1.0).abs() < 1e-12, "step {step} summed to {total}");
        }
    }
}
