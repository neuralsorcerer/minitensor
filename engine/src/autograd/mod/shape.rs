// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    operations::reduction,
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// Map an output tap `(oh, ow, kh, kw)` to the input coordinate it reads, or
/// `None` when it lands in the zero padding. The padding bound already forces
/// `0 <= ih < in_h` (and likewise for `iw`), so no second range check is needed.
#[inline(always)]
fn conv_input_coord(
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    in_h: usize,
    in_w: usize,
) -> Option<(usize, usize)> {
    let h_in = oh * stride.0 + kh;
    let w_in = ow * stride.1 + kw;
    if h_in < padding.0 || w_in < padding.1 || h_in >= in_h + padding.0 || w_in >= in_w + padding.1
    {
        None
    } else {
        Some((h_in - padding.0, w_in - padding.1))
    }
}

/// Gradient function for 2D convolution (`operations::conv2d`).
///
/// Given `grad_output` of shape `[N, C_out, OH, OW]`, produces:
/// * `grad_input[n, ic, ih, iw]  = Σ grad_output[n, oc, oh, ow] · weight[oc, ic, kh, kw]`
/// * `grad_weight[oc, ic, kh, kw] = Σ grad_output[n, oc, oh, ow] · input[n, ic, ih, iw]`
/// * `grad_bias[oc]               = Σ grad_output[n, oc, oh, ow]`
///
/// with the same padding/stride index mapping as the forward pass. Each gradient
/// is only computed when its operand requires it, and each is parallelised over a
/// race-free axis: `grad_input` over the batch (disjoint output slices),
/// `grad_weight`/`grad_bias` over the output channel (disjoint kernel/bias
/// slices). The padding/stride coordinate is hoisted out of the input-channel
/// loop since it does not depend on it.
pub struct Conv2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub input_id: TensorId,
    pub weight_id: TensorId,
    pub bias_id: Option<TensorId>,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
    pub bias_requires_grad: bool,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub deps: SmallVec<[TensorId; 3]>,
}

impl GradientFunction for Conv2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let in_dims = self.input.shape().dims();
        let w_dims = self.weight.shape().dims();
        let (batch, in_channels, in_h, in_w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        let (out_channels, kernel_h, kernel_w) = (w_dims[0], w_dims[2], w_dims[3]);
        let go_dims = grad_output.shape().dims();
        let (out_h, out_w) = (go_dims[2], go_dims[3]);
        let stride = self.stride;
        let padding = self.padding;

        let input =
            self.input.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("conv2d backward expects f32 input")
            })?;
        let weight =
            self.weight.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("conv2d backward expects f32 weight")
            })?;
        let go = grad_output.data().as_f32_slice().ok_or_else(|| {
            MinitensorError::internal_error("conv2d backward expects f32 grad_output")
        })?;

        let device = self.input.device();
        let mut gradients = FxHashMap::default();

        // grad_input: parallel over batches, which own disjoint output regions.
        if self.input_requires_grad {
            let in_stride = in_channels * in_h * in_w;
            let mut grad_input = vec![0f32; batch * in_stride];
            grad_input
                .par_chunks_mut(in_stride)
                .enumerate()
                .for_each(|(n, gi)| {
                    for oc in 0..out_channels {
                        let w_base = oc * in_channels * kernel_h * kernel_w;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let g = go[((n * out_channels + oc) * out_h + oh) * out_w + ow];
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        if let Some((ih, iw)) = conv_input_coord(
                                            oh, ow, kh, kw, stride, padding, in_h, in_w,
                                        ) {
                                            let spatial = ih * in_w + iw;
                                            for ic in 0..in_channels {
                                                let w_idx =
                                                    w_base + (ic * kernel_h + kh) * kernel_w + kw;
                                                gi[ic * in_h * in_w + spatial] += g * weight[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            let grad = Tensor::new(
                Arc::new(TensorData::from_vec_f32(grad_input, device)),
                self.input.shape().clone(),
                DataType::Float32,
                device,
                false,
            );
            accumulate_grad(&mut gradients, self.input_id, grad)?;
        }

        // grad_weight: parallel over output channels, which own disjoint slices.
        if self.weight_requires_grad {
            let w_stride = in_channels * kernel_h * kernel_w;
            let mut grad_weight = vec![0f32; out_channels * w_stride];
            grad_weight
                .par_chunks_mut(w_stride)
                .enumerate()
                .for_each(|(oc, gw)| {
                    for n in 0..batch {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let g = go[((n * out_channels + oc) * out_h + oh) * out_w + ow];
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        if let Some((ih, iw)) = conv_input_coord(
                                            oh, ow, kh, kw, stride, padding, in_h, in_w,
                                        ) {
                                            let spatial = ih * in_w + iw;
                                            for ic in 0..in_channels {
                                                let in_idx =
                                                    (n * in_channels + ic) * in_h * in_w + spatial;
                                                gw[(ic * kernel_h + kh) * kernel_w + kw] +=
                                                    g * input[in_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            let grad = Tensor::new(
                Arc::new(TensorData::from_vec_f32(grad_weight, device)),
                self.weight.shape().clone(),
                DataType::Float32,
                device,
                false,
            );
            accumulate_grad(&mut gradients, self.weight_id, grad)?;
        }

        // grad_bias: parallel over output channels.
        if self.bias_requires_grad
            && let Some(bias_id) = self.bias_id
        {
            let mut grad_bias = vec![0f32; out_channels];
            grad_bias.par_iter_mut().enumerate().for_each(|(oc, gb)| {
                let mut sum = 0f32;
                for n in 0..batch {
                    let base = (n * out_channels + oc) * out_h * out_w;
                    for k in 0..out_h * out_w {
                        sum += go[base + k];
                    }
                }
                *gb = sum;
            });
            let grad = Tensor::new(
                Arc::new(TensorData::from_vec_f32(grad_bias, device)),
                Shape::new(vec![out_channels]),
                DataType::Float32,
                device,
                false,
            );
            accumulate_grad(&mut gradients, bias_id, grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.deps
    }
}

impl GradientFunction for PowBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        match self.output.dtype() {
            DataType::Float32 => {
                let base_slice = self.base.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from base tensor")
                })?;
                let exp_slice = self.exponent.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from exponent tensor")
                })?;
                let out_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;

                if self.base_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.base.numel(),
                        self.base.dtype(),
                        self.base.device(),
                    );
                    let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f32 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = exp_slice[i]
                                        * base_slice[i].powf(exp_slice[i] - 1.0)
                                        * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let exp_ptr = exp_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f32;
                                    let exp_ptr = exp_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) = *exp_ptr.add(i)
                                        * (*base_ptr.add(i)).powf(*exp_ptr.add(i) - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] =
                                        exp_val * base_slice[i].powf(exp_val - 1.0) * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) = exp_val
                                        * (*base_ptr.add(i)).powf(exp_val - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[0], grad_tensor)?;
                }

                if self.exp_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.exponent.numel(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );
                    let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f32 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = exp_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = out_slice[i] * base_slice[i].ln() * grad_out[i];
                                }
                            } else {
                                let out_ptr = out_slice.as_ptr() as usize;
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let out_ptr = out_ptr as *const f32;
                                    let base_ptr = base_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) =
                                        *out_ptr.add(i) * (*base_ptr.add(i)).ln() * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            for i in 0..grad_out.len() {
                                grad_slice[i] = out_slice[i] * base_val.ln() * grad_out[i];
                            }
                        }
                        PowBroadcast::ExponentScalar => {
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[1], grad_tensor)?;
                }
            }
            DataType::Float64 => {
                let base_slice = self.base.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from base tensor")
                })?;
                let exp_slice = self.exponent.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from exponent tensor")
                })?;
                let out_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;

                if self.base_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.base.numel(),
                        self.base.dtype(),
                        self.base.device(),
                    );
                    let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f64 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = exp_slice[i]
                                        * base_slice[i].powf(exp_slice[i] - 1.0)
                                        * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let exp_ptr = exp_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f64;
                                    let exp_ptr = exp_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) = *exp_ptr.add(i)
                                        * (*base_ptr.add(i)).powf(*exp_ptr.add(i) - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] =
                                        exp_val * base_slice[i].powf(exp_val - 1.0) * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) = exp_val
                                        * (*base_ptr.add(i)).powf(exp_val - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[0], grad_tensor)?;
                }

                if self.exp_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.exponent.numel(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );
                    let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f64 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = exp_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = out_slice[i] * base_slice[i].ln() * grad_out[i];
                                }
                            } else {
                                let out_ptr = out_slice.as_ptr() as usize;
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let out_ptr = out_ptr as *const f64;
                                    let base_ptr = base_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) =
                                        *out_ptr.add(i) * (*base_ptr.add(i)).ln() * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            for i in 0..grad_out.len() {
                                grad_slice[i] = out_slice[i] * base_val.ln() * grad_out[i];
                            }
                        }
                        PowBroadcast::ExponentScalar => {
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[1], grad_tensor)?;
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Power backward only supported for floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for Hardshrink
pub struct HardshrinkBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for HardshrinkBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        if *mask.get_unchecked(i) {
                            *grad_ptr.add(i) = *go_ptr.add(i);
                        } else {
                            *grad_ptr.add(i) = 0.0;
                        }
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        if *mask.get_unchecked(i) {
                            *grad_ptr.add(i) = *go_ptr.add(i);
                        } else {
                            *grad_ptr.add(i) = 0.0;
                        }
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "hardshrink backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for nan_to_num.
pub struct NanToNumBackward {
    pub input_id: TensorId,
    pub finite_mask: Vec<bool>,
}

impl GradientFunction for NanToNumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        if self.finite_mask.len() != grad_output.numel() {
            return Err(MinitensorError::gradient_error(
                "nan_to_num backward mask length does not match gradient size",
            ));
        }

        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let grad = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let out = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                apply_finite_mask(grad, out, &self.finite_mask);
            }
            DataType::Float64 => {
                let grad = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let out = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                apply_finite_mask(grad, out, &self.finite_mask);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "nan_to_num backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

#[inline(always)]
fn apply_finite_mask<T>(grad: &[T], output: &mut [T], finite_mask: &[bool])
where
    T: Copy + Default + Send + Sync,
{
    debug_assert_eq!(grad.len(), output.len());
    debug_assert_eq!(grad.len(), finite_mask.len());

    let len = grad.len();
    if len < PAR_THRESHOLD {
        for i in 0..len {
            output[i] = if finite_mask[i] {
                grad[i]
            } else {
                T::default()
            };
        }
    } else {
        grad.par_iter()
            .zip(output.par_iter_mut())
            .zip(finite_mask.par_iter())
            .for_each(|((g, out), is_finite)| {
                *out = if *is_finite { *g } else { T::default() };
            });
    }
}

/// Gradient function for ReLU
pub struct ReluBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = go[i] * if self.mask[i] { 1.0 } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let m = if *mask.get_unchecked(i) { 1.0 } else { 0.0 };
                        *grad_ptr.add(i) = *go_ptr.add(i) * m;
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = go[i] * if self.mask[i] { 1.0 } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let m = if *mask.get_unchecked(i) { 1.0 } else { 0.0 };
                        *grad_ptr.add(i) = *go_ptr.add(i) * m;
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "ReLU backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for LeakyReLU
pub struct LeakyReluBackward {
    pub input_id: TensorId,
    pub negative_slope: f64,
    pub mask: Vec<bool>,
}

impl GradientFunction for LeakyReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                let slope = self.negative_slope as f32;
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { go[i] * slope };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let val = if *mask.get_unchecked(i) {
                            *go_ptr.add(i)
                        } else {
                            *go_ptr.add(i) * slope
                        };
                        *grad_ptr.add(i) = val;
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                let slope = self.negative_slope;
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { go[i] * slope };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let val = if *mask.get_unchecked(i) {
                            *go_ptr.add(i)
                        } else {
                            *go_ptr.add(i) * slope
                        };
                        *grad_ptr.add(i) = val;
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LeakyReLU backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for the element-wise absolute value.
///
/// `d/dx |x| = sign(x)` with the sub-gradient at `x == 0` taken as `0`.
/// The stored input shares storage with the forward input (a detached
/// clone), so no data is copied.
pub struct AbsBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AbsBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        macro_rules! abs_grad {
            ($slice:ident, $mut_slice:ident, $ty:ty) => {{
                let x = self.input.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read input for abs backward")
                })?;
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for abs backward")
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for abs backward")
                })?;
                let sign = |v: $ty| -> $ty {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                };
                if gi.len() < PAR_THRESHOLD {
                    for i in 0..gi.len() {
                        gi[i] = go[i] * sign(x[i]);
                    }
                } else {
                    gi.par_iter_mut()
                        .zip(go.par_iter())
                        .zip(x.par_iter())
                        .for_each(|((g, &o), &v)| *g = o * sign(v));
                }
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => abs_grad!(as_f32_slice, as_f32_slice_mut, f32),
            DataType::Float64 => abs_grad!(as_f64_slice, as_f64_slice_mut, f64),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "abs backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `clamp`/`clip`.
///
/// The gradient is passed through where the input lies inside the (inclusive)
/// clamp bounds and zeroed where it was saturated. Either
/// bound may be absent (`clamp_min`/`clamp_max`).
pub struct ClampBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl GradientFunction for ClampBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        macro_rules! clamp_grad {
            ($slice:ident, $mut_slice:ident, $ty:ty) => {{
                let x = self.input.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read input for clamp backward")
                })?;
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for clamp backward")
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for clamp backward")
                })?;
                let min = self.min.map(|m| m as $ty);
                let max = self.max.map(|m| m as $ty);
                let passes = move |v: $ty| -> bool {
                    min.map_or(true, |m| v >= m) && max.map_or(true, |m| v <= m)
                };
                if gi.len() < PAR_THRESHOLD {
                    for i in 0..gi.len() {
                        gi[i] = if passes(x[i]) { go[i] } else { 0.0 };
                    }
                } else {
                    gi.par_iter_mut()
                        .zip(go.par_iter())
                        .zip(x.par_iter())
                        .for_each(|((g, &o), &v)| *g = if passes(v) { o } else { 0.0 });
                }
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => clamp_grad!(as_f32_slice, as_f32_slice_mut, f32),
            DataType::Float64 => clamp_grad!(as_f64_slice, as_f64_slice_mut, f64),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "clamp backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Scatter-add `grad_output` back to the source positions selected along `dim`.
///
/// `indices[i]` is the source position (along `dim`) that produced output row `i`
/// for every outer/inner coordinate. This is the shared backward for
/// `index_select` and `slice` (and, transitively, `narrow`/`flip`/`roll`).
/// Duplicated source indices accumulate, matching the forward gather semantics.
fn index_select_backward_grad(
    grad_output: &Tensor,
    input_shape: &[usize],
    dim: usize,
    indices: &[usize],
) -> Result<Tensor> {
    let numel: usize = input_shape.iter().product();
    let mut grad_data =
        TensorData::zeros_on_device(numel, grad_output.dtype(), grad_output.device());

    let dim_size = input_shape[dim];
    let inner: usize = input_shape[dim + 1..].iter().product();
    let out_dim = indices.len();

    if numel != 0 && out_dim != 0 && inner != 0 {
        let in_chunk = dim_size * inner;
        let out_chunk = out_dim * inner;

        macro_rules! fill {
            ($slice:ident, $mut_slice:ident) => {{
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for index backward")
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for index backward")
                })?;
                gi.par_chunks_mut(in_chunk)
                    .enumerate()
                    .for_each(|(o, gi_chunk)| {
                        let go_chunk = &go[o * out_chunk..(o + 1) * out_chunk];
                        for (i, &idx) in indices.iter().enumerate() {
                            let dst = idx * inner;
                            let src = i * inner;
                            for j in 0..inner {
                                gi_chunk[dst + j] += go_chunk[src + j];
                            }
                        }
                    });
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "index/slice backward only supported for floating point tensors",
                ));
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(grad_data),
        Shape::new(input_shape.to_vec()),
        grad_output.dtype(),
        grad_output.device(),
        false,
    ))
}

/// Scatter-add `grad_output` back to the input positions named by a full `index`
/// tensor (`gather` backward, also reused by min/max/sort/topk along a dim). The
/// `index` slice is laid out identically to `grad_output`; entry `index[..]` is
/// the source coordinate along `dim`. Colliding indices accumulate.
fn gather_backward_grad(
    grad_output: &Tensor,
    input_shape: &[usize],
    dim: usize,
    index: &[i64],
) -> Result<Tensor> {
    let numel: usize = input_shape.iter().product();
    let mut grad_data =
        TensorData::zeros_on_device(numel, grad_output.dtype(), grad_output.device());

    let dim_size = input_shape[dim];
    let inner: usize = input_shape[dim + 1..].iter().product();
    // The index tensor shares `grad_output`'s shape, so the output extent along
    // `dim` is read directly from it.
    let out_dim = grad_output.shape().dims()[dim];

    if numel != 0 && !index.is_empty() && inner != 0 {
        let in_chunk = dim_size * inner;
        let out_chunk = out_dim * inner;

        macro_rules! fill {
            ($slice:ident, $mut_slice:ident) => {{
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to read grad_output for gather backward",
                    )
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for gather backward")
                })?;
                gi.par_chunks_mut(in_chunk)
                    .enumerate()
                    .for_each(|(o, gi_chunk)| {
                        let go_chunk = &go[o * out_chunk..(o + 1) * out_chunk];
                        let idx_chunk = &index[o * out_chunk..(o + 1) * out_chunk];
                        for i in 0..out_dim {
                            for j in 0..inner {
                                let pos = i * inner + j;
                                let src_idx = idx_chunk[pos] as usize;
                                gi_chunk[src_idx * inner + j] += go_chunk[pos];
                            }
                        }
                    });
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "gather backward only supported for floating point tensors",
                ));
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(grad_data),
        Shape::new(input_shape.to_vec()),
        grad_output.dtype(),
        grad_output.device(),
        false,
    ))
}

/// Gradient function for `index_select` and `slice` (source indices along `dim`).
pub struct IndexSelectBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub indices: Vec<usize>,
}

impl GradientFunction for IndexSelectBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input =
            index_select_backward_grad(grad_output, &self.input_shape, self.dim, &self.indices)?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `gather` (and, reused, min/max/sort/topk along a dim).
pub struct GatherBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub index: Vec<i64>,
}

impl GradientFunction for GatherBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input =
            gather_backward_grad(grad_output, &self.input_shape, self.dim, &self.index)?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `concatenate` (and, transitively, `cat`/`stack`/`roll`).
pub struct ConcatBackward {
    pub input_ids: SmallVec<[TensorId; 4]>,
    pub sizes: SmallVec<[usize; 4]>,
    pub dim: usize,
    /// Which inputs actually need a gradient; frozen inputs skip their
    /// slice extraction.
    pub input_requires_grad: SmallVec<[bool; 4]>,
}

impl GradientFunction for ConcatBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let mut offset = 0usize;
        for ((&id, &size), &needs_grad) in self
            .input_ids
            .iter()
            .zip(self.sizes.iter())
            .zip(self.input_requires_grad.iter())
        {
            if needs_grad {
                let grad_slice = crate::operations::shape_ops::narrow(
                    grad_output,
                    self.dim as isize,
                    offset,
                    size,
                )?;
                accumulate_grad(&mut gradients, id, grad_slice)?;
            }
            offset += size;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for `roll`: rolling is a bijection, so the gradient is the
/// input rolled back by the negated shifts. Computed with a dedicated node rather
/// than by composing `slice`/`concatenate`, because `roll`'s flatten path builds
/// a storage-sharing view whose gradient edges cannot be composed safely.
pub struct RollBackward {
    pub input_id: TensorId,
    pub shifts: Vec<isize>,
    pub dims: Option<Vec<isize>>,
}

impl GradientFunction for RollBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let neg: Vec<isize> = self.shifts.iter().map(|s| -s).collect();
        let grad_input =
            crate::operations::shape_ops::roll(grad_output, &neg, self.dims.as_deref())?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `repeat` (tiling): sum the gradient over the tiled copies.
pub struct RepeatBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub repeats: Vec<usize>,
}

impl GradientFunction for RepeatBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // `repeat` may prepend leading singleton axes; align the input rank to the
        // repeat/output rank, tile every axis, then sum the tiled copies back down.
        let out_ndim = self.repeats.len();
        let pad = out_ndim - self.input_shape.len();
        let mut aligned = vec![1usize; pad];
        aligned.extend_from_slice(&self.input_shape);

        // View grad_output as (rep_0, in_0, rep_1, in_1, ...) then sum the rep axes.
        let mut split_shape = Vec::with_capacity(2 * out_ndim);
        for axis in 0..out_ndim {
            split_shape.push(self.repeats[axis]);
            split_shape.push(aligned[axis]);
        }
        let reshaped = crate::operations::shape_ops::reshape(grad_output, Shape::new(split_shape))?;
        let rep_axes: Vec<isize> = (0..out_ndim).map(|axis| (2 * axis) as isize).collect();
        let summed = reduction::sum(&reshaped, Some(rep_axes), false)?;
        let grad_input =
            crate::operations::shape_ops::reshape(&summed, Shape::new(self.input_shape.clone()))?;

        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for basic indexing (`tensor[...]` via [`Tensor::index`]).
///
/// The forward gathers input element `offset + Σ_j (start_j + coord_j·step_j)·
/// input_stride_{dim_j}` for each output coordinate; the backward scatters the
/// gradient straight back to those positions. Assumes contiguous input storage,
/// which always holds at the Python boundary where indexing is applied.
pub struct IndexBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub input_strides: Vec<usize>,
    pub offset: usize,
    pub out_dims: Vec<usize>,
    pub orig_dim_map: Vec<usize>,
    pub starts: Vec<usize>,
    pub steps: Vec<usize>,
}

impl GradientFunction for IndexBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let numel: usize = self.input_shape.iter().product();
        let mut grad_data =
            TensorData::zeros_on_device(numel, grad_output.dtype(), grad_output.device());
        let out_strides = Strides::from_shape(&Shape::new(self.out_dims.clone()));
        let out_strides = out_strides.as_slice();

        macro_rules! scatter {
            ($slice:ident, $mut_slice:ident) => {{
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for index backward")
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for index backward")
                })?;
                if self.out_dims.is_empty() {
                    // Scalar result: a single collapsed element.
                    gi[self.offset] += go[0];
                } else {
                    for (idx, &g) in go.iter().enumerate() {
                        let mut rem = idx;
                        let mut src = self.offset;
                        for (j, &ostride) in out_strides.iter().enumerate() {
                            let coord = rem / ostride;
                            rem %= ostride;
                            src += (self.starts[j] + coord * self.steps[j])
                                * self.input_strides[self.orig_dim_map[j]];
                        }
                        gi[src] += g;
                    }
                }
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => scatter!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => scatter!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "index backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            Shape::new(self.input_shape.clone()),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for softmax
pub struct SoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Allocate gradient buffer
        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                softmax_backward_f32(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                softmax_backward_f64(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for log-softmax
pub struct LogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                log_softmax_backward_f32(
                    go,
                    log_y,
                    grad_slice,
                    self.output.shape().dims(),
                    self.dim,
                );
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                log_softmax_backward_f64(
                    go,
                    log_y,
                    grad_slice,
                    self.output.shape().dims(),
                    self.dim,
                );
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LogSoftmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for masked log-softmax
pub struct MaskedLogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub mask: Tensor,
    pub dim: usize,
}
