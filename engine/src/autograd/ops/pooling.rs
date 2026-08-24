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

/// Gradient of [`crate::ops::interpolate::interpolate`].
///
/// Interpolation is linear in its input, so the gradient is the transpose of
/// the forward: the same source indices and the same weights, scattered where
/// the forward gathered. Nothing else is stored -- the axis maps are `O(out)`
/// numbers and cheaper to rebuild than to carry, and rebuilding them from the
/// same function is what stops the two directions from ever disagreeing.
pub struct InterpolateBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub output_size: Vec<usize>,
    pub mode: crate::ops::interpolate::InterpolateMode,
    pub align_corners: bool,
}

/// Gradient of [`crate::ops::grid_sample::grid_sample`], for both of its
/// inputs.
///
/// The taps are recomputed rather than carried. Finding them again costs
/// exactly what the forward pass paid, which is less than storing a
/// `2^axes`-wide weight table for every output position -- and much less when
/// only one of the two gradients is wanted, which is the usual case.
pub struct GridSampleBackward {
    pub input_ids: [TensorId; 2],
    /// Which of [input, grid] actually need a gradient. A warp against a fixed
    /// grid needs only the first, and a spatial transformer reading a frozen
    /// feature map needs only the second.
    pub input_requires_grad: [bool; 2],
    pub input: Tensor,
    pub grid: Tensor,
    pub mode: crate::ops::grid_sample::SampleMode,
    pub padding: crate::ops::grid_sample::Padding,
    pub align_corners: bool,
}

impl GradientFunction for GridSampleBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let (for_input, for_grid) = crate::ops::grid_sample::grid_sample_backward(
            &self.input,
            &self.grid,
            grad_output,
            self.input_requires_grad,
            self.mode,
            self.padding,
            self.align_corners,
        )?;
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);
        if let Some(gradient) = for_input {
            accumulate_grad(&mut gradients, self.input_ids[0], gradient)?;
        }
        if let Some(gradient) = for_grid {
            accumulate_grad(&mut gradients, self.input_ids[1], gradient)?;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

macro_rules! interpolate_backward {
    ($name:ident, $ty:ty, $accessor:ident, $from_vec:ident) => {
        fn $name(
            grad_output: &Tensor,
            planes: usize,
            in_h: usize,
            in_w: usize,
            rows: &crate::ops::interpolate::AxisMap,
            cols: &crate::ops::interpolate::AxisMap,
        ) -> Result<TensorData> {
            let grad = grad_output.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("interpolate backward: mismatched dtype")
            })?;
            let (out_h, out_w) = (rows.len(), cols.len());
            let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
            let mut values = vec![0 as $ty; planes * plane_in];
            if plane_in == 0 || plane_out == 0 {
                return Ok(TensorData::$from_vec(values, grad_output.device()));
            }

            // One task per plane: several output positions read the same source,
            // so the scatter inside a plane is serial, but planes are disjoint.
            par_out_chunks(&mut values, plane_in, &|first, image| {
                let base = (first / plane_in) * plane_out;
                for oh in 0..out_h {
                    let (top, bottom) = (rows.lower(oh) * in_w, rows.upper(oh) * in_w);
                    let row_weight = rows.weight(oh) as $ty;
                    for ow in 0..out_w {
                        let (left, right) = (cols.lower(ow), cols.upper(ow));
                        let column_weight = cols.weight(ow) as $ty;
                        let share = grad[base + oh * out_w + ow];
                        // The four coefficients the forward multiplied by, in
                        // the same order it formed them.
                        let upper_share = share * row_weight;
                        let lower_share = share - upper_share;
                        image[top + left] += lower_share * (1.0 as $ty - column_weight);
                        image[top + right] += lower_share * column_weight;
                        image[bottom + left] += upper_share * (1.0 as $ty - column_weight);
                        image[bottom + right] += upper_share * column_weight;
                    }
                }
            });
            Ok(TensorData::$from_vec(values, grad_output.device()))
        }
    };
}

interpolate_backward!(interpolate_backward_f32, f32, as_f32_slice, from_vec_f32);
interpolate_backward!(interpolate_backward_f64, f64, as_f64_slice, from_vec_f64);

impl GradientFunction for InterpolateBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let dims = &self.input_shape;
        let spatial = &dims[2..];
        let (in_h, in_w) = if spatial.len() == 2 {
            (spatial[0], spatial[1])
        } else {
            (1, spatial[0])
        };
        let (out_h, out_w) = if self.output_size.len() == 2 {
            (self.output_size[0], self.output_size[1])
        } else {
            (1, self.output_size[0])
        };
        let planes = dims[0] * dims[1];
        let rows = crate::ops::interpolate::axis_map(in_h, out_h, self.mode, self.align_corners);
        let cols = crate::ops::interpolate::axis_map(in_w, out_w, self.mode, self.align_corners);

        let grad = grad_output.contiguous()?;
        let data = match grad.dtype() {
            DataType::Float32 => interpolate_backward_f32(&grad, planes, in_h, in_w, &rows, &cols)?,
            DataType::Float64 => interpolate_backward_f64(&grad, planes, in_h, in_w, &rows, &cols)?,
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "interpolate backward only supports floating point tensors",
                ));
            }
        };

        let mut gradients = FxHashMap::default();
        accumulate_grad(
            &mut gradients,
            self.input_id,
            Tensor::new(
                Arc::new(data),
                Shape::new(dims.clone()),
                grad.dtype(),
                grad.device(),
                false,
            ),
        )?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Spread each output gradient evenly over the window it averaged, where the
/// windows came from the ratio of the extents rather than a fixed kernel.
///
/// Separate from [`AvgPool2dBackward`] because the windows are not uniform:
/// each one has its own size, so each carries its own divisor, and they overlap
/// by however much the ratio demands. A caller who reached for the fixed-window
/// backward with an averaged kernel size would be wrong by a little almost
/// everywhere and by a lot at the edges.
pub struct AdaptiveAvgPool2dBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub output_size: (usize, usize),
}

macro_rules! adaptive_avg_pool_backward {
    ($name:ident, $ty:ty, $accessor:ident, $from_vec:ident) => {
        fn $name(
            grad_output: &Tensor,
            input_shape: &[usize],
            output_size: (usize, usize),
        ) -> Result<TensorData> {
            let grad = grad_output.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(
                    "adaptive_avg_pool2d backward received a mismatched dtype",
                )
            })?;
            let (batch, channels) = (input_shape[0], input_shape[1]);
            let (in_h, in_w) = (input_shape[2], input_shape[3]);
            let (out_h, out_w) = output_size;
            let plane_in = in_h * in_w;
            let plane_out = out_h * out_w;

            let mut values = vec![0 as $ty; batch * channels * plane_in];
            if plane_out == 0 || plane_in == 0 {
                return Ok(TensorData::$from_vec(values, grad_output.device()));
            }

            // One task per `[H, W]` plane: the windows within a plane overlap,
            // so the scatter-add inside is serial, but planes never touch.
            par_out_chunks(&mut values, plane_in, &|first, image| {
                let base = (first / plane_in) * plane_out;
                for oh in 0..out_h {
                    let rows = crate::ops::pooling::adaptive_window(oh, in_h, out_h);
                    for ow in 0..out_w {
                        let cols = crate::ops::pooling::adaptive_window(ow, in_w, out_w);
                        let share = grad[base + oh * out_w + ow] / (rows.len() * cols.len()) as $ty;
                        for ih in rows.clone() {
                            let row = ih * in_w;
                            for slot in &mut image[row + cols.start..row + cols.end] {
                                *slot += share;
                            }
                        }
                    }
                }
            });
            Ok(TensorData::$from_vec(values, grad_output.device()))
        }
    };
}

adaptive_avg_pool_backward!(
    adaptive_avg_pool_backward_f32,
    f32,
    as_f32_slice,
    from_vec_f32
);
adaptive_avg_pool_backward!(
    adaptive_avg_pool_backward_f64,
    f64,
    as_f64_slice,
    from_vec_f64
);

impl GradientFunction for AdaptiveAvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let data = match grad_output.dtype() {
            DataType::Float32 => {
                adaptive_avg_pool_backward_f32(grad_output, &self.input_shape, self.output_size)?
            }
            DataType::Float64 => {
                adaptive_avg_pool_backward_f64(grad_output, &self.input_shape, self.output_size)?
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "adaptive_avg_pool2d backward only supports floating point tensors",
                ));
            }
        };
        let mut gradients = FxHashMap::default();
        accumulate_grad(
            &mut gradients,
            self.input_id,
            Tensor::new(
                Arc::new(data),
                Shape::new(self.input_shape.clone()),
                grad_output.dtype(),
                grad_output.device(),
                false,
            ),
        )?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
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
/// Gradient of [`crate::ops::conv::conv_transpose2d`].
///
/// Every piece of this already existed, because a transposed convolution is a
/// convolution read from the other side and so is each of its gradients:
///
/// * the input gradient *gathers* where the forward scattered, which is a
///   convolution -- the same `conv2d` a caller would write, on the same weight
/// * the weight gradient is the same sum a convolution's is, with the two
///   operands swapping which one plays the image
/// * the bias gradient is a sum over positions, as always
///
/// So this holds no arithmetic of its own. If it did, there would be two
/// implementations of the same index mapping and one of them would eventually
/// be wrong.
pub struct ConvTranspose2dBackward {
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
    pub dilation: (usize, usize),
    pub groups: usize,
    pub deps: SmallVec<[TensorId; 3]>,
}

impl ConvTranspose2dBackward {
    fn backward_typed<T: ConvScalar>(
        &self,
        grad_output: &Tensor,
        gradients: &mut FxHashMap<TensorId, Tensor>,
    ) -> Result<()> {
        let in_dims = self.input.shape().dims();
        let w_dims = self.weight.shape().dims();
        let go_dims = grad_output.shape().dims();
        // The forward's geometry, unchanged: the grid it wrote is `grad_output`
        // and the signal it read is `input`.
        let geometry = crate::ops::conv::ConvGeometry {
            batch_size: in_dims[0],
            in_channels: go_dims[1],
            input_height: go_dims[2],
            input_width: go_dims[3],
            out_channels: in_dims[1],
            kernel_h: w_dims[2],
            kernel_w: w_dims[3],
            output_height: in_dims[2],
            output_width: in_dims[3],
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        };

        let device = self.input.device();
        let dtype = T::DTYPE;

        if self.weight_requires_grad || self.bias_requires_grad {
            let go = T::slice(grad_output.data()).ok_or_else(|| {
                MinitensorError::internal_error(
                    "conv_transpose2d backward: failed to read grad_output",
                )
            })?;

            if self.weight_requires_grad {
                let input = T::slice(self.input.data()).ok_or_else(|| {
                    MinitensorError::internal_error(
                        "conv_transpose2d backward: input dtype does not match gradient",
                    )
                })?;
                // The image is the gradient of the grid that was written, and
                // the signal is the input that was scattered -- the convolution
                // weight gradient with the roles the forward gave them.
                let signal = crate::ops::conv::to_channel_major(
                    input,
                    geometry.batch_size,
                    geometry.out_channels,
                    geometry.output_height * geometry.output_width,
                );
                let grad_weight =
                    crate::ops::conv::column_weight_gradient::<T>(go, &signal, &geometry);
                let grad = Tensor::new(
                    Arc::new(T::into_tensor_data(grad_weight, device)),
                    self.weight.shape().clone(),
                    dtype,
                    device,
                    false,
                );
                accumulate_grad(gradients, self.weight_id, grad)?;
            }

            if self.bias_requires_grad
                && let Some(bias_id) = self.bias_id
            {
                let grad_bias = crate::ops::conv::channel_sums(
                    go,
                    geometry.batch_size,
                    geometry.in_channels,
                    geometry.input_height * geometry.input_width,
                );
                let grad = Tensor::new(
                    Arc::new(T::into_tensor_data(grad_bias, device)),
                    Shape::new(vec![geometry.in_channels]),
                    dtype,
                    device,
                    false,
                );
                accumulate_grad(gradients, bias_id, grad)?;
            }
        }
        Ok(())
    }
}

impl GradientFunction for ConvTranspose2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        // Gathering where the forward scattered is exactly a convolution, so
        // this is the public op rather than a second index mapping. The
        // operands are detached, so nothing is recorded.
        if self.input_requires_grad {
            let grad_input = crate::ops::conv::conv2d(
                grad_output,
                &self.weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )?;
            accumulate_grad(&mut gradients, self.input_id, grad_input.detach())?;
        }

        match grad_output.dtype() {
            DataType::Float32 => self.backward_typed::<f32>(grad_output, &mut gradients)?,
            DataType::Float64 => self.backward_typed::<f64>(grad_output, &mut gradients)?,
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "conv_transpose2d backward supports only floating point tensors",
                ));
            }
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.deps
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
    pub dilation: (usize, usize),
    pub groups: usize,
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
        let go_dims = grad_output.shape().dims();
        let geometry = crate::ops::conv::ConvGeometry {
            batch_size: in_dims[0],
            in_channels: in_dims[1],
            input_height: in_dims[2],
            input_width: in_dims[3],
            out_channels: w_dims[0],
            kernel_h: w_dims[2],
            kernel_w: w_dims[3],
            output_height: go_dims[2],
            output_width: go_dims[3],
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        };

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
        let plane = geometry.output_height * geometry.output_width;

        // Both gradient GEMMs contract against `grad_output` with the channel
        // outermost, so it is rearranged once rather than once each.
        let needs_gemm = self.input_requires_grad || self.weight_requires_grad;
        let go_mat = if needs_gemm {
            crate::ops::conv::to_channel_major(
                go,
                geometry.batch_size,
                geometry.out_channels,
                plane,
            )
        } else {
            Vec::new()
        };

        // The input gradient scatters the signal back through the kernel, which
        // is the operation `conv_transpose2d` performs forwards.
        if self.input_requires_grad {
            let grad_input = crate::ops::conv::scatter_columns::<T>(&go_mat, weight, &geometry);
            let grad = Tensor::new(
                Arc::new(T::into_tensor_data(grad_input, device)),
                self.input.shape().clone(),
                dtype,
                device,
                false,
            );
            accumulate_grad(gradients, self.input_id, grad)?;
        }

        if self.weight_requires_grad {
            let grad_weight =
                crate::ops::conv::column_weight_gradient::<T>(input, &go_mat, &geometry);
            let grad = Tensor::new(
                Arc::new(T::into_tensor_data(grad_weight, device)),
                self.weight.shape().clone(),
                dtype,
                device,
                false,
            );
            accumulate_grad(gradients, self.weight_id, grad)?;
        }

        if self.bias_requires_grad
            && let Some(bias_id) = self.bias_id
        {
            let grad_bias = crate::ops::conv::channel_sums(
                go,
                geometry.batch_size,
                geometry.out_channels,
                plane,
            );
            let grad = Tensor::new(
                Arc::new(T::into_tensor_data(grad_bias, device)),
                Shape::new(vec![geometry.out_channels]),
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
