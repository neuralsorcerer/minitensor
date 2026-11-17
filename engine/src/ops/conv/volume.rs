// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Convolutions of volumes, rounded as a plane's are.
//!
//! A 3-D window is a stack of 2-D ones, one for each depth tap of the
//! kernel, so the convolution of a volume is the convolution of a plane
//! whose channels are the input's at each depth tap: for each slice of the
//! output, the input slices its windows read, set side by side as channels.
//! A kernel `[C_out, C_in / groups, kD, kH, kW]` is already that plane's
//! kernel, `[C_out, C_in / groups * kD, kH, kW]`, with each group's channels
//! together. Each entry is then one sum of the products of every tap of its
//! window, the bias in it -- the float nearest its exact value, as a plane's
//! are -- where each depth tap was convolved on its own and the planes
//! added, a rounding for each and one for the bias.
//!
//! The gradients follow. The kernel's is that plane's, over the same signal.
//! The input's sets the output's gradient side by side the same way, a
//! channel for each output channel at each depth tap -- the slice of the
//! output that tap reaches the input slice from, or zeros -- and scatters it
//! back through the kernel taken a depth tap at a time.

use super::{ConvGeometry, ConvScalar};
use crate::autograd::{Conv3dBackward, with_grad_fn};
use crate::error::{MinitensorError, Result};
use crate::ops::map::par_out_chunks;
use crate::tensor::{DataType, Shape, Tensor, TensorData};
use smallvec::SmallVec;
use std::sync::Arc;

/// The shape of a convolution of volumes: the extents along the depth, the
/// height and the width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Volumes {
    pub(crate) batch: usize,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) input: [usize; 3],
    pub(crate) kernel: [usize; 3],
    pub(crate) output: [usize; 3],
    pub(crate) stride: [usize; 3],
    pub(crate) padding: [usize; 3],
    pub(crate) dilation: [usize; 3],
    pub(crate) groups: usize,
}

impl Volumes {
    /// The input slice that depth tap `tap` of output slice `od` reads,
    /// where that is not padding.
    fn read_slice(&self, od: usize, tap: usize) -> Option<usize> {
        (od * self.stride[0] + tap * self.dilation[0])
            .checked_sub(self.padding[0])
            .filter(|&id| id < self.input[0])
    }

    /// The output slice whose depth tap `tap` reads input slice `id`, where
    /// one does.
    fn reading_slice(&self, id: usize, tap: usize) -> Option<usize> {
        let at = (id + self.padding[0]).checked_sub(tap * self.dilation[0])?;
        let od = at / self.stride[0];
        (at % self.stride[0] == 0 && od < self.output[0]).then_some(od)
    }

    /// The convolution of planes the forward pass and the kernel's gradient
    /// take: each output slice's windows, their depth taps as channels.
    pub(crate) fn stacked(&self) -> ConvGeometry {
        ConvGeometry {
            batch_size: self.batch * self.output[0],
            in_channels: self.in_channels * self.kernel[0],
            input_height: self.input[1],
            input_width: self.input[2],
            out_channels: self.out_channels,
            kernel_h: self.kernel[1],
            kernel_w: self.kernel[2],
            output_height: self.output[1],
            output_width: self.output[2],
            stride: (self.stride[1], self.stride[2]),
            padding: (self.padding[1], self.padding[2]),
            dilation: (self.dilation[1], self.dilation[2]),
            groups: self.groups,
        }
    }

    /// The convolution of planes whose input gradient is the volume's: each
    /// input slice to a channel for each output channel at each depth tap.
    pub(crate) fn spread(&self) -> ConvGeometry {
        ConvGeometry {
            batch_size: self.batch * self.input[0],
            in_channels: self.in_channels,
            out_channels: self.out_channels * self.kernel[0],
            ..self.stacked()
        }
    }

    /// The input, `[N, C_in, D, H, W]`, as `[N * OD, C_in * kD, H, W]`: for
    /// each output slice, the input slice each depth tap reads, or zeros.
    pub(crate) fn stack_input<T: ConvScalar>(&self, input: &[T]) -> Vec<T> {
        let plane = self.input[1] * self.input[2];
        let (channels, taps) = (self.in_channels, self.kernel[0]);
        let block = channels * taps * plane;
        let mut stacked = vec![T::default(); self.batch * self.output[0] * block];
        par_out_chunks(&mut stacked, block, &|first, out| {
            let (n, od) = (
                first / block / self.output[0],
                first / block % self.output[0],
            );
            for c in 0..channels {
                for tap in 0..taps {
                    if let Some(id) = self.read_slice(od, tap) {
                        let from = ((n * channels + c) * self.input[0] + id) * plane;
                        out[(c * taps + tap) * plane..][..plane]
                            .copy_from_slice(&input[from..from + plane]);
                    }
                }
            }
        });
        stacked
    }

    /// The output's gradient, `[N, C_out, OD, OH, OW]`, as
    /// `[N * D, C_out * kD, OH, OW]`: for each input slice, the slice of the
    /// output each depth tap of each output channel reaches it from, or
    /// zeros.
    pub(crate) fn stack_gradient<T: ConvScalar>(&self, flowing: &[T]) -> Vec<T> {
        let plane = self.output[1] * self.output[2];
        let (channels, taps) = (self.out_channels, self.kernel[0]);
        let block = channels * taps * plane;
        let mut stacked = vec![T::default(); self.batch * self.input[0] * block];
        par_out_chunks(&mut stacked, block, &|first, out| {
            let (n, id) = (first / block / self.input[0], first / block % self.input[0]);
            for oc in 0..channels {
                for tap in 0..taps {
                    if let Some(od) = self.reading_slice(id, tap) {
                        let from = ((n * channels + oc) * self.output[0] + od) * plane;
                        out[(oc * taps + tap) * plane..][..plane]
                            .copy_from_slice(&flowing[from..from + plane]);
                    }
                }
            }
        });
        stacked
    }

    /// The kernel, `[C_out, C_in / groups, kD, kH, kW]`, as
    /// `[C_out * kD, C_in / groups, kH, kW]`: a plane's kernel for each
    /// output channel at each depth tap.
    pub(crate) fn kernel_by_tap<T: ConvScalar>(&self, weight: &[T]) -> Vec<T> {
        let (taps, per_group) = (self.kernel[0], self.in_channels / self.groups);
        let plane = self.kernel[1] * self.kernel[2];
        let mut out = vec![T::default(); weight.len()];
        for oc in 0..self.out_channels {
            for c in 0..per_group {
                for tap in 0..taps {
                    let from = ((oc * per_group + c) * taps + tap) * plane;
                    let to = ((oc * taps + tap) * per_group + c) * plane;
                    out[to..to + plane].copy_from_slice(&weight[from..from + plane]);
                }
            }
        }
        out
    }
}

/// `[N * S, C, P]` as `[N, C, S, P]`: `slices` planes of `plane` values for
/// each of `channels` channels.
pub(crate) fn slices_inside<T: ConvScalar>(
    stacked: &[T],
    slices: usize,
    channels: usize,
    plane: usize,
) -> Vec<T> {
    let block = slices * plane;
    let mut out = vec![T::default(); stacked.len()];
    par_out_chunks(&mut out, block, &|first, volume| {
        let (n, c) = (first / block / channels, first / block % channels);
        for s in 0..slices {
            let from = ((n * slices + s) * channels + c) * plane;
            volume[s * plane..][..plane].copy_from_slice(&stacked[from..from + plane]);
        }
    });
    out
}

/// `[N, C, S, P]` as `[N * S, C, P]`, as [`slices_inside`] undoes.
pub(crate) fn slices_outside<T: ConvScalar>(
    volume: &[T],
    slices: usize,
    channels: usize,
    plane: usize,
) -> Vec<T> {
    let block = channels * plane;
    let mut out = vec![T::default(); volume.len()];
    par_out_chunks(&mut out, block, &|first, planes| {
        let (n, s) = (first / block / slices, first / block % slices);
        for c in 0..channels {
            let from = ((n * channels + c) * slices + s) * plane;
            planes[c * plane..][..plane].copy_from_slice(&volume[from..from + plane]);
        }
    });
    out
}

/// 3-D cross-correlation of `[N, C_in, D, H, W]` with a
/// `[C_out, C_in / groups, kD, kH, kW]` kernel: the convolution of the plane
/// whose channels are each output slice's depth taps.
///
/// `stride`, `padding` and `dilation` are along the depth, the height and
/// the width. Each entry of the output, and of each gradient, is the float
/// nearest its exact value, the bias added before the one rounding.
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
) -> Result<Tensor> {
    if input.ndim() != 5 {
        return Err(MinitensorError::invalid_operation(
            "conv3d expects 5D input tensor [N, C_in, D, H, W]",
        ));
    }
    if weight.ndim() != 5 {
        return Err(MinitensorError::invalid_operation(
            "conv3d expects 5D weight tensor [C_out, C_in / groups, kD, kH, kW]",
        ));
    }
    let (dims, taps) = (input.shape().dims(), weight.shape().dims());
    let (in_channels, out_channels) = (dims[1], taps[0]);
    let kernel = [taps[2], taps[3], taps[4]];
    if kernel.contains(&0) {
        return Err(MinitensorError::invalid_operation(format!(
            "kernel size must be greater than zero, got {}x{}x{}",
            kernel[0], kernel[1], kernel[2]
        )));
    }
    if groups == 0 || !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(MinitensorError::invalid_operation(format!(
            "conv3d groups={groups} must be positive and divide both \
             in_channels={in_channels} and out_channels={out_channels}"
        )));
    }
    if in_channels / groups != taps[1] {
        return Err(MinitensorError::shape_mismatch_explained(
            vec![taps[1]],
            vec![in_channels / groups],
            format!(
                "conv3d: the input has {in_channels} channels (shape {dims:?}), but the \
                 weight {taps:?} with groups={groups} reads {} -- check the input's \
                 channel axis, dimension 1",
                taps[1] * groups
            ),
        ));
    }
    if let Some(b) = bias
        && (b.ndim() != 1 || b.size(0)? != out_channels)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![out_channels],
            b.shape().dims().to_vec(),
        ));
    }
    if stride.contains(&0) || dilation.contains(&0) {
        return Err(MinitensorError::invalid_operation(
            "stride and dilation values must be greater than zero",
        ));
    }
    let mut output = [0; 3];
    for axis in 0..3 {
        let span = dilation[axis] * (kernel[axis] - 1) + 1;
        let padded = dims[2 + axis] + 2 * padding[axis];
        if span > padded {
            return Err(MinitensorError::invalid_operation(
                "kernel size cannot be larger than padded input",
            ));
        }
        output[axis] = (padded - span) / stride[axis] + 1;
    }
    if !input.device().is_cpu() || !weight.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "conv3d is implemented only for CPU tensors",
        ));
    }
    if weight.dtype() != input.dtype() || bias.is_some_and(|b| b.dtype() != input.dtype()) {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", input.dtype()),
            format!("{:?}", weight.dtype()),
        ));
    }
    let shape = Volumes {
        batch: dims[0],
        in_channels,
        out_channels,
        input: [dims[2], dims[3], dims[4]],
        kernel,
        output,
        stride,
        padding,
        dilation,
        groups,
    };
    let output_shape = Shape::new(vec![dims[0], out_channels, output[0], output[1], output[2]]);
    TensorData::ensure_allocatable(output_shape.try_numel()?, input.dtype())?;
    // The signal set side by side, which the forward pass reads.
    let stacked = Shape::new(vec![
        dims[0] * output[0],
        in_channels * kernel[0],
        dims[3],
        dims[4],
    ]);
    TensorData::ensure_allocatable(stacked.try_numel()?, input.dtype())?;

    let data = match input.dtype() {
        DataType::Float32 => forward::<f32>(input, weight, bias, &shape)?,
        DataType::Float64 => forward::<f64>(input, weight, bias, &shape)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "conv3d is implemented only for floating point tensors",
            ));
        }
    };
    let bias_requires_grad = bias.is_some_and(|b| b.requires_grad());
    let requires_grad = input.requires_grad() || weight.requires_grad() || bias_requires_grad;
    let mut result = Tensor::new(
        Arc::new(data),
        output_shape,
        input.dtype(),
        input.device(),
        requires_grad,
    );
    if requires_grad {
        let mut deps: SmallVec<[_; 3]> = SmallVec::new();
        if input.requires_grad() {
            deps.push(input.id());
        }
        if weight.requires_grad() {
            deps.push(weight.id());
        }
        if bias_requires_grad && let Some(b) = bias {
            deps.push(b.id());
        }
        let grad_fn = Arc::new(Conv3dBackward {
            input: input.detach(),
            weight: weight.detach(),
            input_id: input.id(),
            weight_id: weight.id(),
            bias_id: bias.map(|b| b.id()),
            input_requires_grad: input.requires_grad(),
            weight_requires_grad: weight.requires_grad(),
            bias_requires_grad,
            shape,
            deps,
        });
        result = with_grad_fn(result, grad_fn)?;
    }
    Ok(result)
}

/// The output of [`conv3d`], of `shape`, in the element type `T`.
fn forward<T: ConvScalar>(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    shape: &Volumes,
) -> Result<TensorData> {
    let mismatched = || MinitensorError::internal_error("conv3d received a mismatched dtype");
    let values = T::slice(input.data()).ok_or_else(mismatched)?;
    let kernel = T::slice(weight.data()).ok_or_else(mismatched)?;
    let added = match bias {
        Some(bias) => Some(T::slice(bias.data()).ok_or_else(mismatched)?),
        None => None,
    };
    let planes = T::convolve(&shape.stack_input(values), kernel, added, &shape.stacked())?;
    let plane = shape.output[1] * shape.output[2];
    let volume = slices_inside(&planes, shape.output[0], shape.out_channels, plane);
    Ok(T::into_tensor_data(volume, input.device()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    /// Values `m * 2^e`, `m` of seven bits and `e` in `[-6, 6]`, zeros among
    /// them: every product of two is exact in float64, and so is any sum of
    /// fewer than `2^10` of them, while few such sums fit a float32.
    fn values(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let digits = (state >> 57) as i32 - 64;
                if state.is_multiple_of(13) {
                    return 0.0;
                }
                digits as f32 * 2f32.powi((state % 13) as i32 - 6)
            })
            .collect()
    }

    fn tensor(values: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(values, Device::cpu())),
            Shape::new(shape.to_vec()),
            DataType::Float32,
            Device::cpu(),
            true,
        )
    }

    /// The output and the gradients of the input, the kernel and the bias,
    /// from the definition: each an exact sum in float64.
    fn reference(x: &[f32], w: &[f32], b: &[f32], g: &[f32], s: &Volumes) -> [Vec<f64>; 4] {
        let [d, h, wd] = s.input;
        let [kd, kh, kw] = s.kernel;
        let [od, oh, ow] = s.output;
        let (per_in, per_out) = (s.in_channels / s.groups, s.out_channels / s.groups);
        let mut out = vec![0.0f64; s.batch * s.out_channels * od * oh * ow];
        let mut grad_x = vec![0.0f64; x.len()];
        let mut grad_w = vec![0.0f64; w.len()];
        let mut grad_b = vec![0.0f64; s.out_channels];
        let at = |o: usize, k: usize, axis: usize, dim: usize| {
            (o * s.stride[axis] + k * s.dilation[axis])
                .checked_sub(s.padding[axis])
                .filter(|&i| i < dim)
        };
        for n in 0..s.batch {
            for oc in 0..s.out_channels {
                let group = oc / per_out;
                for (slot, total) in out
                    .iter_mut()
                    .skip((n * s.out_channels + oc) * od * oh * ow)
                    .take(od * oh * ow)
                    .enumerate()
                {
                    let (o0, o1, o2) = (slot / (oh * ow), slot / ow % oh, slot % ow);
                    let flowing = f64::from(g[(n * s.out_channels + oc) * od * oh * ow + slot]);
                    grad_b[oc] += flowing;
                    let mut sum = f64::from(b[oc]);
                    for cc in 0..per_in {
                        let c = group * per_in + cc;
                        for t0 in 0..kd {
                            let Some(i0) = at(o0, t0, 0, d) else { continue };
                            for t1 in 0..kh {
                                let Some(i1) = at(o1, t1, 1, h) else { continue };
                                for t2 in 0..kw {
                                    let Some(i2) = at(o2, t2, 2, wd) else {
                                        continue;
                                    };
                                    let xi =
                                        (((n * s.in_channels + c) * d + i0) * h + i1) * wd + i2;
                                    let wi = (((oc * per_in + cc) * kd + t0) * kh + t1) * kw + t2;
                                    sum += f64::from(x[xi]) * f64::from(w[wi]);
                                    grad_x[xi] += flowing * f64::from(w[wi]);
                                    grad_w[wi] += flowing * f64::from(x[xi]);
                                }
                            }
                        }
                    }
                    *total = sum;
                }
            }
        }
        [out, grad_x, grad_w, grad_b]
    }

    fn check(volumes: Volumes, seed: u64) {
        let s = volumes;
        let [d, h, wd] = s.input;
        let x = values(seed, s.batch * s.in_channels * d * h * wd);
        let w = values(
            seed + 1,
            s.out_channels * (s.in_channels / s.groups) * s.kernel.iter().product::<usize>(),
        );
        let b = values(seed + 2, s.out_channels);
        let shape_x = [s.batch, s.in_channels, d, h, wd];
        let shape_w = [
            s.out_channels,
            s.in_channels / s.groups,
            s.kernel[0],
            s.kernel[1],
            s.kernel[2],
        ];
        let (xt, wt, bt) = (
            tensor(x.clone(), &shape_x),
            tensor(w.clone(), &shape_w),
            tensor(b.clone(), &[s.out_channels]),
        );
        let out = conv3d(
            &xt,
            &wt,
            Some(&bt),
            s.stride,
            s.padding,
            s.dilation,
            s.groups,
        )
        .unwrap();
        let g = values(seed + 3, out.numel());
        let [want, want_x, want_w, want_b] = reference(&x, &w, &b, &g, &s);
        let label = format!("{s:?}");
        let narrow = |exact: &[f64]| exact.iter().map(|&v| v as f32).collect::<Vec<_>>();
        assert_eq!(
            out.data().as_f32_slice().unwrap(),
            &narrow(&want)[..],
            "{label} output"
        );

        let flowing = tensor(g, out.shape().dims());
        let gradients = out.grad_fn().unwrap().backward(&flowing).unwrap();
        for (operand, want, what) in [
            (&xt, &want_x, "input"),
            (&wt, &want_w, "weight"),
            (&bt, &want_b, "bias"),
        ] {
            let got = gradients[&operand.id()].data().as_f32_slice().unwrap();
            assert_eq!(got, &narrow(want)[..], "{label} {what} gradient");
        }
    }

    fn volumes(
        batch: usize,
        channels: (usize, usize),
        input: [usize; 3],
        kernel: [usize; 3],
        steps: ([usize; 3], [usize; 3], [usize; 3]),
        groups: usize,
    ) -> Volumes {
        let (stride, padding, dilation) = steps;
        let output = std::array::from_fn(|axis| {
            (input[axis] + 2 * padding[axis] - dilation[axis] * (kernel[axis] - 1) - 1)
                / stride[axis]
                + 1
        });
        Volumes {
            batch,
            in_channels: channels.0,
            out_channels: channels.1,
            input,
            kernel,
            output,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    #[test]
    fn volumes_convolve_and_take_their_gradients_rounded_once() {
        let ones = [1, 1, 1];
        let cases = [
            volumes(2, (3, 4), [5, 6, 7], [3, 3, 3], (ones, ones, ones), 1),
            volumes(
                1,
                (4, 4),
                [6, 5, 5],
                [2, 3, 2],
                ([2, 1, 2], [1, 0, 1], [2, 1, 1]),
                2,
            ),
            volumes(2, (4, 4), [4, 5, 6], [3, 1, 3], (ones, [1, 0, 1], ones), 4),
            volumes(1, (8, 16), [4, 10, 12], [3, 3, 3], (ones, ones, ones), 1),
            volumes(
                1,
                (2, 3),
                [7, 4, 4],
                [5, 2, 2],
                ([3, 2, 1], [2, 1, 0], ones),
                1,
            ),
            volumes(2, (3, 2), [3, 3, 3], [1, 2, 2], (ones, [0, 1, 1], ones), 1),
        ];
        for (index, case) in cases.into_iter().enumerate() {
            check(case, 900 + 10 * index as u64);
        }
    }
}
