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
use crate::ops::arithmetic::exact_sum::{Element, Output, Tallies, sum_to};
use crate::ops::conv::ConvScalar;
use crate::ops::map::par_out_chunks;
use crate::ops::pooling::{Extents, FixedWindow, average};
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

/// Spread each output gradient over the 3-D window it averaged.
pub struct AvgPool3dBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub(crate) window: FixedWindow,
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

impl GradientFunction for AdaptiveAvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let shape = &self.input_shape;
        let extents = Extents::planar(shape[0] * shape[1], (shape[2], shape[3]), self.output_size);
        averaged_gradient(
            grad_output,
            self.input_id,
            shape,
            extents,
            None,
            "adaptive_avg_pool2d",
        )
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// The gradient an average pool's input of `input_shape` takes from
/// `grad_output`: of `window`, or of adaptive windows without one.
fn averaged_gradient(
    grad_output: &Tensor,
    input_id: TensorId,
    input_shape: &[usize],
    extents: Extents,
    window: Option<&FixedWindow>,
    name: &str,
) -> Result<FxHashMap<TensorId, Tensor>> {
    let mismatched =
        || MinitensorError::internal_error(format!("{name} backward received a mismatched dtype"));
    let data = match grad_output.dtype() {
        DataType::Float32 => {
            let flowing = grad_output.data().as_f32_slice().ok_or_else(mismatched)?;
            let cells = match window {
                Some(window) => average::gradient(flowing, extents, window),
                None => average::adaptive_gradient(flowing, extents),
            };
            TensorData::from_vec_f32(cells, grad_output.device())
        }
        DataType::Float64 => {
            let flowing = grad_output.data().as_f64_slice().ok_or_else(mismatched)?;
            let cells = match window {
                Some(window) => average::gradient(flowing, extents, window),
                None => average::adaptive_gradient(flowing, extents),
            };
            TensorData::from_vec_f64(cells, grad_output.device())
        }
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "{name} backward only supports floating point tensors"
            )));
        }
    };
    let mut gradients = FxHashMap::default();
    gradients.insert(
        input_id,
        Tensor::new(
            Arc::new(data),
            Shape::new(input_shape.to_vec()),
            grad_output.dtype(),
            grad_output.device(),
            false,
        ),
    );
    Ok(gradients)
}

/// The gradient a max pool's input takes from `flowing`: each window's
/// gradient sent to the cell that won it, `indices` naming that cell within
/// its plane, or `-1` where a window chose none.
///
/// A plane in which no cell won twice takes one addition a cell. In one
/// where windows overlap and a cell won several, each cell takes the value
/// nearest the exact sum of the gradients of the windows it won, rounded
/// once: they are added into a two-sum for each cell, and a cell its bound
/// leaves open has its gradients gathered and summed exactly. Added one at
/// a time in the element type, they took a rounding for each past the
/// first.
fn routed_to_winners<T>(flowing: &[T], indices: &[i64], planes: usize, plane_in: usize) -> Vec<T>
where
    T: Element + Output + Default + std::ops::Add<Output = T>,
{
    let mut grad = vec![T::default(); planes * plane_in];
    if plane_in == 0 || planes == 0 {
        return grad;
    }
    let plane_out = flowing.len() / planes;
    let band = average::slices_per_task(planes, plane_in) * plane_in;
    par_out_chunks(&mut grad, band, &|first, part| {
        let (mut seen, mut tallies) = (vec![false; plane_in], None);
        for (index, cells) in part.chunks_mut(plane_in).enumerate() {
            let start = (first / plane_in + index) * plane_out;
            let (flowing, won) = (
                &flowing[start..start + plane_out],
                &indices[start..start + plane_out],
            );
            seen.fill(false);
            let mut twice = false;
            for &cell in won.iter().filter(|&&cell| cell >= 0) {
                twice |= std::mem::replace(&mut seen[cell as usize], true);
            }
            if !twice {
                for (&cell, &share) in won.iter().zip(flowing) {
                    if cell >= 0 {
                        cells[cell as usize] = cells[cell as usize] + share;
                    }
                }
                continue;
            }
            // Only the cells some window chose are emptied, added into and
            // rounded; the rest stay zero.
            let tallies = tallies.get_or_insert_with(|| Tallies::new(plane_in));
            let chosen = || {
                won.iter()
                    .filter(|&&cell| cell >= 0)
                    .map(|&cell| cell as usize)
            };
            for cell in chosen() {
                tallies.clear_at(cell, 0.0);
            }
            for (&cell, &share) in won.iter().zip(flowing) {
                if cell >= 0 {
                    tallies.add_at(cell as usize, share.wide());
                }
            }
            for cell in chosen() {
                // Each chosen cell once: its mark is taken as it is rounded.
                if std::mem::take(&mut seen[cell]) {
                    cells[cell] = tallies.settled::<T>(cell, None).unwrap_or_else(|| {
                        let kept: Vec<T> = won
                            .iter()
                            .zip(flowing)
                            .filter(|&(&winner, _)| winner == cell as i64)
                            .map(|(_, &share)| share)
                            .collect();
                        sum_to(&kept)
                    });
                }
            }
        }
    });
    grad
}

impl GradientFunction for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mismatched =
            || MinitensorError::internal_error("max_pool2d backward received a mismatched dtype");
        let planes = self.input_shape[0] * self.input_shape[1];
        let plane_in = self.input_shape[2] * self.input_shape[3];
        let device = grad_output.device();
        let data = match grad_output.dtype() {
            DataType::Float32 => {
                let flowing = grad_output.data().as_f32_slice().ok_or_else(mismatched)?;
                let cells = routed_to_winners(flowing, &self.indices, planes, plane_in);
                TensorData::from_vec_f32(cells, device)
            }
            DataType::Float64 => {
                let flowing = grad_output.data().as_f64_slice().ok_or_else(mismatched)?;
                let cells = routed_to_winners(flowing, &self.indices, planes, plane_in);
                TensorData::from_vec_f64(cells, device)
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

impl GradientFunction for AvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let (shape, out) = (&self.input_shape, grad_output.shape().dims());
        let extents = Extents::planar(shape[0] * shape[1], (shape[2], shape[3]), (out[2], out[3]));
        let window = FixedWindow::planar(
            self.kernel,
            self.stride,
            self.padding,
            self.count_include_pad,
        );
        averaged_gradient(
            grad_output,
            self.input_id,
            shape,
            extents,
            Some(&window),
            "avg_pool2d",
        )
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

impl GradientFunction for AvgPool3dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let (shape, out) = (&self.input_shape, grad_output.shape().dims());
        let extents = Extents {
            volumes: shape[0] * shape[1],
            in_d: shape[2],
            in_h: shape[3],
            in_w: shape[4],
            out_d: out[2],
            out_h: out[3],
            out_w: out[4],
        };
        averaged_gradient(
            grad_output,
            self.input_id,
            shape,
            extents,
            Some(&self.window),
            "avg_pool3d",
        )
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
                let grad_weight = T::weight_gradient(go, input, &geometry);
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
                accumulate_grad(gradients, bias_id, channel_sums(grad_output)?)?;
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

/// A bias's gradient: the incoming `[N, C, H, W]` gradient summed down to one
/// value per channel, each the value nearest its exact sum.
fn channel_sums(grad_output: &Tensor) -> Result<Tensor> {
    Ok(crate::ops::reduction::sum(&grad_output.detach(), Some(vec![0, 2, 3]), false)?.detach())
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

        // The input gradient scatters the signal back through the kernel, which
        // is the operation `conv_transpose2d` performs forwards.
        if self.input_requires_grad {
            let grad_input = T::scatter(go, weight, None, &geometry);
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
            let grad_weight = T::weight_gradient(input, go, &geometry);
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
            accumulate_grad(gradients, bias_id, channel_sums(grad_output)?)?;
        }
        Ok(())
    }
}

/// Gradient function for 3-D convolution (`ops::conv3d`), the gradients of
/// the convolution of planes it is.
///
/// The kernel's gradient is that plane's, over the input set side by side
/// as the forward pass read it, and the bias's each channel of the incoming
/// gradient summed. The input's scatters the incoming gradient, set side by
/// side by input slice, a channel for each output channel at each depth
/// tap, back through the kernel a depth tap at a time.
pub struct Conv3dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub input_id: TensorId,
    pub weight_id: TensorId,
    pub bias_id: Option<TensorId>,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
    pub bias_requires_grad: bool,
    pub(crate) shape: crate::ops::conv::Volumes,
    pub deps: SmallVec<[TensorId; 3]>,
}

impl Conv3dBackward {
    fn backward_typed<T: ConvScalar>(
        &self,
        grad_output: &Tensor,
        gradients: &mut FxHashMap<TensorId, Tensor>,
    ) -> Result<()> {
        let mismatched =
            || MinitensorError::internal_error("conv3d backward received a mismatched dtype");
        let flowing = T::slice(grad_output.data()).ok_or_else(mismatched)?;
        let shape = &self.shape;
        let (device, dtype) = (self.input.device(), T::DTYPE);
        let gradient = |values: Vec<T>, like: &Tensor| {
            Tensor::new(
                Arc::new(T::into_tensor_data(values, device)),
                like.shape().clone(),
                dtype,
                device,
                false,
            )
        };

        if self.input_requires_grad {
            let weight = T::slice(self.weight.data()).ok_or_else(mismatched)?;
            let spread = shape.stack_gradient(flowing);
            let planes = T::scatter(&spread, &shape.kernel_by_tap(weight), None, &shape.spread());
            let plane = shape.input[1] * shape.input[2];
            let cells =
                crate::ops::conv::slices_inside(&planes, shape.input[0], shape.in_channels, plane);
            accumulate_grad(gradients, self.input_id, gradient(cells, &self.input))?;
        }

        if self.weight_requires_grad {
            let input = T::slice(self.input.data()).ok_or_else(mismatched)?;
            let plane = shape.output[1] * shape.output[2];
            let planes = crate::ops::conv::slices_outside(
                flowing,
                shape.output[0],
                shape.out_channels,
                plane,
            );
            let taps = T::weight_gradient(&shape.stack_input(input), &planes, &shape.stacked());
            accumulate_grad(gradients, self.weight_id, gradient(taps, &self.weight))?;
        }

        if self.bias_requires_grad
            && let Some(bias_id) = self.bias_id
        {
            let sums =
                crate::ops::reduction::sum(&grad_output.detach(), Some(vec![0, 2, 3, 4]), false)?;
            accumulate_grad(gradients, bias_id, sums.detach())?;
        }
        Ok(())
    }
}

impl GradientFunction for Conv3dBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        match grad_output.dtype() {
            DataType::Float32 => self.backward_typed::<f32>(grad_output, &mut gradients)?,
            DataType::Float64 => self.backward_typed::<f64>(grad_output, &mut gradients)?,
            _ => {
                return Err(MinitensorError::internal_error(
                    "conv3d backward expects a floating point gradient",
                ));
            }
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.deps
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_that_wins_several_windows_takes_their_sum_rounded_once() {
        let tiny = 2f32.powi(-24);
        // Two planes of four cells and six windows each. In the first, cell
        // 0 wins three windows, whose gradients come to one and a float32's
        // last place exactly, where adding them one at a time stays at one;
        // cell 1 wins two that cancel, and a window chose nothing.
        let flowing = [
            1.0, tiny, tiny, 3.0, -3.0, 7.0, //
            0.5, 0.25, -0.0, 2.0, tiny, 9.0,
        ];
        let indices = [0, 0, 0, 1, 1, -1, 3, 2, 1, 0, 0, 3];
        let cells = routed_to_winners(&flowing, &indices, 2, 4);
        assert_eq!(cells[0], 1.0 + 2.0 * tiny);
        assert_eq!(cells[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(cells[2..4], [0.0, 0.0]);
        assert_eq!(cells[4], 2.0 + tiny);
        assert_eq!(cells[5].to_bits(), 0.0f32.to_bits());
        assert_eq!(cells[6], 0.25);
        assert_eq!(cells[7], 9.5);

        // A plane in which no cell won twice takes each gradient as it is,
        // a negative zero as a positive one.
        let cells = routed_to_winners(&[1.5f64, -0.0, 2.5], &[2, 0, -1], 1, 3);
        assert_eq!(cells, vec![0.0, 0.0, 1.5]);
        assert_eq!(cells[0].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn infinities_and_halfway_sums_are_rounded_as_their_exact_sums_are() {
        let step = f32::EPSILON;
        // One and a half last places past one: halfway, so the even one.
        let cells = routed_to_winners(&[1.0f32, step, step / 2.0], &[0, 0, 0], 1, 1);
        assert_eq!(cells, vec![1.0 + 2.0 * step]);
        let cells = routed_to_winners(&[1.0f32, step / 2.0], &[0, 0], 1, 1);
        assert_eq!(cells, vec![1.0]);
        let cells = routed_to_winners(
            &[f32::INFINITY, 1.0, f32::MAX, f32::MAX],
            &[0, 0, 1, 1],
            1,
            2,
        );
        assert_eq!(cells, vec![f32::INFINITY, f32::INFINITY]);
        let cells = routed_to_winners(&[f64::INFINITY, f64::NEG_INFINITY], &[0, 0], 1, 1);
        assert!(cells[0].is_nan());
    }
}
