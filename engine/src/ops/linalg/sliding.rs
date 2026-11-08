// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The sliding products `correlate` and `convolve` take, each overlap the
//! float nearest its exact dot.

use std::borrow::Cow;
use std::sync::Arc;

use crate::{
    error::{MinitensorError, Result},
    ops::arithmetic::exact_sum::{dot_f32, dot_f64},
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// A float dtype an overlap is rounded to, by its rounded-once dot.
trait Overlap: Copy + Send + Sync + 'static {
    fn dot(a: &[Self], b: &[Self]) -> Self;
}

impl Overlap for f32 {
    fn dot(a: &[Self], b: &[Self]) -> Self {
        dot_f32(a, b)
    }
}

impl Overlap for f64 {
    fn dot(a: &[Self], b: &[Self]) -> Self {
        dot_f64(a, b)
    }
}

/// The overlaps `start..start + count` of the full correlation of two 1-D
/// float sequences of one dtype, `kernel` reversed first where `flip`, which
/// makes it a convolution.
///
/// Overlap `k` of `n` samples and `m` taps is `Σ signal[k + j - (m - 1)] ·
/// kernel[j]` over the taps that land on the signal, `n + m - 1` of them in
/// all; each is the float nearest that exact sum.
pub fn sliding_products(
    signal: &Tensor,
    kernel: &Tensor,
    flip: bool,
    start: usize,
    count: usize,
) -> Result<Tensor> {
    let dtype = signal.dtype();
    if !matches!(dtype, DataType::Float32 | DataType::Float64)
        || kernel.dtype() != dtype
        || signal.ndim() != 1
        || kernel.ndim() != 1
        || signal.numel() == 0
        || kernel.numel() == 0
    {
        return Err(MinitensorError::invalid_operation(format!(
            "sliding products take two non-empty 1-D float sequences of one dtype, got {:?} \
             of {:?} and {:?} of {:?}",
            dtype,
            signal.shape().dims(),
            kernel.dtype(),
            kernel.shape().dims()
        )));
    }
    let full = signal.numel() + kernel.numel() - 1;
    if start.checked_add(count).is_none_or(|end| end > full) {
        return Err(MinitensorError::invalid_operation(format!(
            "sliding products: overlaps {start}..{} of {full}",
            start.saturating_add(count)
        )));
    }
    let (signal, kernel) = (signal.contiguous()?, kernel.contiguous()?);
    let missing = || MinitensorError::internal_error("sliding products: a float slice");
    let mut data = TensorData::uninitialized_on_device(count, dtype, signal.device());
    if dtype == DataType::Float32 {
        let (x, w) = (signal.data().as_f32_slice(), kernel.data().as_f32_slice());
        let out = data.as_f32_slice_mut().ok_or_else(missing)?;
        overlaps(
            x.ok_or_else(missing)?,
            w.ok_or_else(missing)?,
            flip,
            start,
            out,
        );
    } else {
        let (x, w) = (signal.data().as_f64_slice(), kernel.data().as_f64_slice());
        let out = data.as_f64_slice_mut().ok_or_else(missing)?;
        overlaps(
            x.ok_or_else(missing)?,
            w.ok_or_else(missing)?,
            flip,
            start,
            out,
        );
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![count]),
        dtype,
        signal.device(),
        false,
    ))
}

/// The overlaps from `start` into `out`, a run of them to a task; a long
/// overlap's dot spreads across the pool itself.
fn overlaps<T: Overlap>(signal: &[T], kernel: &[T], flip: bool, start: usize, out: &mut [T]) {
    let weights: Cow<[T]> = if flip {
        Cow::Owned(kernel.iter().rev().copied().collect())
    } else {
        Cow::Borrowed(kernel)
    };
    let (n, m) = (signal.len(), weights.len());
    let overlap = |k: usize| {
        // The taps `j` with `0 <= k + j - (m - 1) < n`.
        let first = (m - 1).saturating_sub(k);
        let last = m.min(n + m - 1 - k);
        let offset = k + first - (m - 1);
        T::dot(
            &signal[offset..offset + (last - first)],
            &weights[first..last],
        )
    };
    let longest = n.min(m);
    let chunk = (8192 / longest).max(1);
    let serial_below = (1 << 15) / longest + 1;
    crate::ops::map::par_out_chunks_gated(out, chunk, serial_below, &|first, part| {
        for (offset, slot) in part.iter_mut().enumerate() {
            *slot = overlap(start + first + offset);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn tensor(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    #[test]
    fn full_correlations_and_convolutions() {
        let (x, w) = (tensor(vec![1.0, 2.0, 3.0]), tensor(vec![1.0, 10.0]));
        let got = sliding_products(&x, &w, false, 0, 4).unwrap();
        assert_eq!(values(&got), vec![10.0, 21.0, 32.0, 3.0]);
        let got = sliding_products(&x, &w, true, 0, 4).unwrap();
        assert_eq!(values(&got), vec![1.0, 12.0, 23.0, 30.0]);
        let got = sliding_products(&x, &w, true, 1, 2).unwrap();
        assert_eq!(values(&got), vec![12.0, 23.0]);
    }

    #[test]
    fn each_overlap_rounds_once() {
        // `1 + 2^-53 + 2^-53`: a plain sum drops both small products.
        let tiny = 2f64.powi(-53);
        let x = tensor(vec![1.0, tiny, tiny]);
        let w = tensor(vec![1.0, 1.0, 1.0]);
        let got = sliding_products(&x, &w, false, 2, 1).unwrap();
        assert_eq!(values(&got), vec![1.0 + 2f64.powi(-52)]);
    }

    #[test]
    fn refuses_what_it_cannot_slide() {
        let x = tensor(vec![1.0, 2.0]);
        assert!(sliding_products(&x, &tensor(vec![]), false, 0, 1).is_err());
        assert!(sliding_products(&x, &x, false, 2, 2).is_err());
    }
}
