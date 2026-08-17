// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Backward passes for the 2-D spatial operations: convolution and pooling.
//!
//! All of them scatter into the input plane, and several output windows can
//! overlap the same input element once `stride < kernel`, so the accumulation
//! has to be per-plane rather than per-output: each task owns one `[H, W]`
//! plane of the gradient and no two tasks touch the same element.

use super::*;
use crate::ops::conv::ConvScalar;
use crate::ops::map::par_out_chunks;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// Route each output gradient back to the input element that won its window.
///
/// The winning offsets were recorded during the forward pass, so this does not
/// need the input values at all — only where the maxima came from.
pub struct MaxPool2dBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    /// Flat offset within the `[H, W]` plane that supplied each output element,
    /// or `-1` for a window that selected nothing.
    pub indices: Vec<i64>,
}

/// Spread each output gradient evenly over the window it averaged.
pub struct AvgPool2dBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub count_include_pad: bool,
}

macro_rules! max_pool_backward {
    ($name:ident, $ty:ty, $accessor:ident, $from_vec:ident) => {
        fn $name(
            grad_output: &Tensor,
            indices: &[i64],
            input_shape: &[usize],
        ) -> Result<TensorData> {
            let go = grad_output.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("max_pool2d backward received a mismatched dtype")
            })?;
            let plane_in = input_shape[2] * input_shape[3];
            let planes = input_shape[0] * input_shape[1];
            let plane_out = if planes == 0 { 0 } else { go.len() / planes };

            let mut grad = vec![0 as $ty; planes * plane_in];
            par_out_chunks(&mut grad, plane_in, &|first, grad_plane| {
                let start = (first / plane_in) * plane_out;
                for slot in 0..plane_out {
                    let offset = indices[start + slot];
                    if offset >= 0 {
                        grad_plane[offset as usize] += go[start + slot];
                    }
                }
            });

            Ok(TensorData::$from_vec(grad, grad_output.device()))
        }
    };
}

max_pool_backward!(max_pool_backward_f32, f32, as_f32_slice, from_vec_f32);
max_pool_backward!(max_pool_backward_f64, f64, as_f64_slice, from_vec_f64);

impl GradientFunction for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let data = match grad_output.dtype() {
            DataType::Float32 => {
                max_pool_backward_f32(grad_output, &self.indices, &self.input_shape)?
            }
            DataType::Float64 => {
                max_pool_backward_f64(grad_output, &self.indices, &self.input_shape)?
            }
            _ => {
                return Err(MinitensorError::internal_error(
                    "max_pool2d backward expects a floating point gradient",
                ));
            }
        };

        let mut gradients = FxHashMap::default();
        gradients.insert(
            self.input_id,
            Tensor::new(
                Arc::new(data),
                Shape::new(self.input_shape.clone()),
                grad_output.dtype(),
                grad_output.device(),
                false,
            ),
        );
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

macro_rules! avg_pool_backward {
    ($name:ident, $ty:ty, $accessor:ident, $from_vec:ident) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            grad_output: &Tensor,
            input_shape: &[usize],
            out_h: usize,
            out_w: usize,
            kernel: (usize, usize),
            stride: (usize, usize),
            padding: (usize, usize),
            count_include_pad: bool,
        ) -> Result<TensorData> {
            let go = grad_output.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("avg_pool2d backward received a mismatched dtype")
            })?;
            let (in_h, in_w) = (input_shape[2], input_shape[3]);
            let plane_in = in_h * in_w;
            let plane_out = out_h * out_w;
            let planes = input_shape[0] * input_shape[1];

            let mut grad = vec![0 as $ty; planes * plane_in];
            par_out_chunks(&mut grad, plane_in, &|first, grad_plane| {
                let start = (first / plane_in) * plane_out;
                {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            // Recompute the divisor the forward pass used, so an
                            // edge window that excluded padding spreads its
                            // gradient over the same cells it averaged.
                            let mut counted = 0usize;
                            for ky in 0..kernel.0 {
                                let ih = oh * stride.0 + ky;
                                if ih < padding.0 || ih >= in_h + padding.0 {
                                    continue;
                                }
                                for kx in 0..kernel.1 {
                                    let iw = ow * stride.1 + kx;
                                    if iw >= padding.1 && iw < in_w + padding.1 {
                                        counted += 1;
                                    }
                                }
                            }
                            let divisor = if count_include_pad {
                                kernel.0 * kernel.1
                            } else {
                                counted
                            };
                            if divisor == 0 {
                                continue;
                            }
                            let share = go[start + oh * out_w + ow] / divisor as $ty;

                            for ky in 0..kernel.0 {
                                let ih = oh * stride.0 + ky;
                                if ih < padding.0 || ih >= in_h + padding.0 {
                                    continue;
                                }
                                let ih = ih - padding.0;
                                for kx in 0..kernel.1 {
                                    let iw = ow * stride.1 + kx;
                                    if iw < padding.1 || iw >= in_w + padding.1 {
                                        continue;
                                    }
                                    let iw = iw - padding.1;
                                    grad_plane[ih * in_w + iw] += share;
                                }
                            }
                        }
                    }
                }
            });

            Ok(TensorData::$from_vec(grad, grad_output.device()))
        }
    };
}

avg_pool_backward!(avg_pool_backward_f32, f32, as_f32_slice, from_vec_f32);
avg_pool_backward!(avg_pool_backward_f64, f64, as_f64_slice, from_vec_f64);

impl GradientFunction for AvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let go_dims = grad_output.shape().dims();
        let (out_h, out_w) = (go_dims[2], go_dims[3]);

        let data = match grad_output.dtype() {
            DataType::Float32 => avg_pool_backward_f32(
                grad_output,
                &self.input_shape,
                out_h,
                out_w,
                self.kernel,
                self.stride,
                self.padding,
                self.count_include_pad,
            )?,
            DataType::Float64 => avg_pool_backward_f64(
                grad_output,
                &self.input_shape,
                out_h,
                out_w,
                self.kernel,
                self.stride,
                self.padding,
                self.count_include_pad,
            )?,
            _ => {
                return Err(MinitensorError::internal_error(
                    "avg_pool2d backward expects a floating point gradient",
                ));
            }
        };

        let mut gradients = FxHashMap::default();
        gradients.insert(
            self.input_id,
            Tensor::new(
                Arc::new(data),
                Shape::new(self.input_shape.clone()),
                grad_output.dtype(),
                grad_output.device(),
                false,
            ),
        );
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for 2D convolution (`ops::conv2d`).
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
impl Conv2dBackward {
    fn backward_typed<T: ConvScalar>(
        &self,
        grad_output: &Tensor,
        gradients: &mut FxHashMap<TensorId, Tensor>,
    ) -> Result<()> {
        let in_dims = self.input.shape().dims();
        let w_dims = self.weight.shape().dims();
        let (batch, in_channels, in_h, in_w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        let (out_channels, kernel_h, kernel_w) = (w_dims[0], w_dims[2], w_dims[3]);
        let go_dims = grad_output.shape().dims();
        let (out_h, out_w) = (go_dims[2], go_dims[3]);
        let stride = self.stride;
        let padding = self.padding;

        let input = T::slice(self.input.data()).ok_or_else(|| {
            MinitensorError::internal_error("conv2d backward: input dtype does not match gradient")
        })?;
        let weight = T::slice(self.weight.data()).ok_or_else(|| {
            MinitensorError::internal_error("conv2d backward: weight dtype does not match gradient")
        })?;
        let go = T::slice(grad_output.data()).ok_or_else(|| {
            MinitensorError::internal_error("conv2d backward: failed to read grad_output")
        })?;

        let device = self.input.device();
        let dtype = T::DTYPE;

        let ohw = out_h * out_w;
        let n_ohw = batch * ohw;
        let k_dim = in_channels * kernel_h * kernel_w;
        let kh_kw = kernel_h * kernel_w;

        // Transpose grad_output [N, C_out, OH*OW] into `go_mat` [C_out, N*OH*OW],
        // the layout both weight- and input-gradient GEMMs contract against.
        let go_mat = if n_ohw > 0 && (self.input_requires_grad || self.weight_requires_grad) {
            let mut gm = vec![T::default(); out_channels * n_ohw];
            par_out_chunks(&mut gm, n_ohw, &|start, row| {
                let oc = start / n_ohw;
                for n in 0..batch {
                    let src = (n * out_channels + oc) * ohw;
                    row[n * ohw..n * ohw + ohw].copy_from_slice(&go[src..src + ohw]);
                }
            });
            gm
        } else {
            Vec::new()
        };

        // grad_input = col2im(weightᵀ @ go_mat). The GEMM yields grad_cols
        // [K, N*OH*OW]; col2im scatters each column back to the input positions it
        // was gathered from. Parallel over the batch (disjoint grad_input regions),
        // with a serial scatter-add within each batch, so there are no races.
        if self.input_requires_grad {
            let in_stride = in_channels * in_h * in_w;
            let mut grad_input = vec![T::default(); batch * in_stride];
            if n_ohw > 0 {
                let mut grad_cols = vec![T::default(); k_dim * n_ohw];
                // SAFETY: the logical `[K, C_out]` operand is `weight`, stored
                // `[C_out, K]`, so the GEMM reads it transposed by stride --
                // this used to materialise it with a serial strided-write copy,
                // which for a 512-channel layer is 9 MB moved per backward.
                // `go_mat` is [C_out, N*OH*OW] and `grad_cols` [K, N*OH*OW],
                // both contiguous row-major.
                unsafe {
                    T::gemm_tn(
                        k_dim,
                        out_channels,
                        n_ohw,
                        weight.as_ptr(),
                        go_mat.as_ptr(),
                        grad_cols.as_mut_ptr(),
                    );
                }
                // Output positions are walked as nested loops and their
                // in-bounds range is hoisted, for the reason the forward's
                // im2col does the same: recovering `(oh, ow)` from a flat `p`
                // cost two runtime-divisor divisions per element, over 4.7M
                // elements for a 16x32x32x32 conv.
                par_out_chunks(&mut grad_input, in_stride, &|start, gi| {
                    let n = start / in_stride;
                    {
                        for k in 0..k_dim {
                            let ic = k / kh_kw;
                            let rem = k % kh_kw;
                            let ky = rem / kernel_w;
                            let kx = rem % kernel_w;
                            let row_base = k * n_ohw + n * ohw;
                            let ic_base = ic * in_h * in_w;
                            let (oh_lo, oh_hi) = crate::ops::conv::in_bounds_range(
                                ky, padding.0, in_h, stride.0, out_h,
                            );
                            let (ow_lo, ow_hi) = crate::ops::conv::in_bounds_range(
                                kx, padding.1, in_w, stride.1, out_w,
                            );
                            for oh in oh_lo..oh_hi {
                                let ih = oh * stride.0 + ky - padding.0;
                                let dst = ic_base + ih * in_w;
                                let src = row_base + oh * out_w;
                                for ow in ow_lo..ow_hi {
                                    let iw = ow * stride.1 + kx - padding.1;
                                    gi[dst + iw] += grad_cols[src + ow];
                                }
                            }
                        }
                    }
                });
            }
            let grad = Tensor::new(
                Arc::new(T::into_tensor_data(grad_input, device)),
                self.input.shape().clone(),
                dtype,
                device,
                false,
            );
            accumulate_grad(gradients, self.input_id, grad)?;
        }

        // grad_weight = go_mat @ cols, where `cols` [N*OH*OW, K] is the im2col of
        // the input (the same lowering as the forward). One GEMM replaces the
        // naive per-element accumulation.
        if self.weight_requires_grad {
            let mut grad_weight = vec![T::default(); out_channels * k_dim];
            if n_ohw > 0 {
                let mut cols = vec![T::default(); n_ohw * k_dim];
                // `k` is walked as nested (ic, ky, kx) loops rather than
                // decomposed: three divisions per element, and this buffer has
                // the same 4.7M elements as the one above.
                par_out_chunks(&mut cols, k_dim, &|start, prow| {
                    let r = start / k_dim;
                    {
                        let n = r / ohw;
                        let p = r % ohw;
                        let oh = p / out_w;
                        let ow = p % out_w;
                        let mut k = 0usize;
                        for ic in 0..in_channels {
                            let plane = (n * in_channels + ic) * in_h * in_w;
                            for ky in 0..kernel_h {
                                let ih = oh * stride.0 + ky;
                                let row_ok = ih >= padding.0 && ih < in_h + padding.0;
                                let ih = ih.wrapping_sub(padding.0);
                                for kx in 0..kernel_w {
                                    let iw = ow * stride.1 + kx;
                                    if row_ok && iw >= padding.1 && iw < in_w + padding.1 {
                                        prow[k] = input[plane + ih * in_w + (iw - padding.1)];
                                    }
                                    k += 1;
                                }
                            }
                        }
                    }
                });
                // SAFETY: go_mat is [C_out, N*OH*OW], cols is [N*OH*OW, K], and
                // grad_weight is [C_out, K]; all contiguous row-major, dims match.
                unsafe {
                    T::gemm(
                        out_channels,
                        n_ohw,
                        k_dim,
                        go_mat.as_ptr(),
                        cols.as_ptr(),
                        grad_weight.as_mut_ptr(),
                    );
                }
            }
            let grad = Tensor::new(
                Arc::new(T::into_tensor_data(grad_weight, device)),
                self.weight.shape().clone(),
                dtype,
                device,
                false,
            );
            accumulate_grad(gradients, self.weight_id, grad)?;
        }

        // grad_bias: parallel over output channels.
        if self.bias_requires_grad
            && let Some(bias_id) = self.bias_id
        {
            let mut grad_bias = vec![T::default(); out_channels];
            par_out_chunks(&mut grad_bias, 1, &|oc, gb| {
                let mut sum = T::default();
                for n in 0..batch {
                    let base = (n * out_channels + oc) * out_h * out_w;
                    for k in 0..out_h * out_w {
                        sum += go[base + k];
                    }
                }
                gb[0] = sum;
            });
            let grad = Tensor::new(
                Arc::new(T::into_tensor_data(grad_bias, device)),
                Shape::new(vec![out_channels]),
                dtype,
                device,
                false,
            );
            accumulate_grad(gradients, bias_id, grad)?;
        }

        Ok(())
    }
}

impl GradientFunction for Conv2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        match grad_output.dtype() {
            DataType::Float32 => self.backward_typed::<f32>(grad_output, &mut gradients)?,
            DataType::Float64 => self.backward_typed::<f64>(grad_output, &mut gradients)?,
            _ => {
                return Err(MinitensorError::internal_error(
                    "conv2d backward expects a floating point gradient",
                ));
            }
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.deps
    }
}
