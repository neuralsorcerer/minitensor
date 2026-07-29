// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Backward passes for the 2-D poolers.
//!
//! Both scatter into the input plane, and several output windows can overlap
//! the same input element once `stride < kernel`, so the accumulation has to be
//! per-plane rather than per-output: each task owns one `[H, W]` plane of the
//! gradient and no two tasks touch the same element.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
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
            grad.par_chunks_mut(plane_in)
                .enumerate()
                .for_each(|(plane, grad_plane)| {
                    let start = plane * plane_out;
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
            grad.par_chunks_mut(plane_in)
                .enumerate()
                .for_each(|(plane, grad_plane)| {
                    let start = plane * plane_out;
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
