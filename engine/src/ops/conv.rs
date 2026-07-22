// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{Conv2dBackward, add_to_graph},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;

/// Perform 2D convolution on the input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape `[N, C_in, H, W]`
/// * `weight` - Convolution kernel of shape `[C_out, C_in, kH, kW]`
/// * `bias` - Optional bias tensor of shape `[C_out]`
/// * `stride` - Stride of the convolution `(sH, sW)`
/// * `padding` - Zero padding added to both sides of the input `(pH, pW)`
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    // Validate dimensions
    if input.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(
            "conv2d expects 4D input tensor [N, C_in, H, W]",
        ));
    }
    if weight.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(
            "conv2d expects 4D weight tensor [C_out, C_in, kH, kW]",
        ));
    }

    let batch_size = input.size(0)?;
    let in_channels = input.size(1)?;
    let input_height = input.size(2)?;
    let input_width = input.size(3)?;

    let out_channels = weight.size(0)?;
    let weight_in_channels = weight.size(1)?;
    let kernel_h = weight.size(2)?;
    let kernel_w = weight.size(3)?;

    if in_channels != weight_in_channels {
        return Err(MinitensorError::shape_mismatch(
            vec![weight_in_channels],
            vec![in_channels],
        ));
    }

    if let Some(b) = bias
        && (b.ndim() != 1 || b.size(0)? != out_channels)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![out_channels],
            vec![b.size(0)?],
        ));
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(MinitensorError::invalid_operation(
            "stride values must be greater than zero",
        ));
    }

    if kernel_h > input_height + 2 * padding.0 || kernel_w > input_width + 2 * padding.1 {
        return Err(MinitensorError::invalid_operation(
            "kernel size cannot be larger than padded input",
        ));
    }

    let output_height = (input_height + 2 * padding.0 - kernel_h) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - kernel_w) / stride.1 + 1;
    let output_shape = Shape::new(vec![batch_size, out_channels, output_height, output_width]);

    match (
        input.dtype(),
        weight.dtype(),
        bias.map(|b| b.dtype()),
        input.device().is_cpu(),
        weight.device().is_cpu(),
    ) {
        (DataType::Float32, DataType::Float32, Some(DataType::Float32), true, true)
        | (DataType::Float32, DataType::Float32, None, true, true) => {
            let input_data = input
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::invalid_operation("Expected f32 input data"))?;
            let weight_data = weight
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::invalid_operation("Expected f32 weight data"))?;
            let bias_data =
                if let Some(bias) = bias {
                    Some(bias.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::invalid_operation("Expected f32 bias data")
                    })?)
                } else {
                    None
                };

            // im2col + GEMM. Lower each output position's receptive field into a
            // column of `cols` ([K, N*out_h*out_w], K = C_in*kH*kW), then a single
            // matrix multiply `weight[C_out, K] @ cols` produces `[C_out,
            // N*out_h*out_w]`, which is scattered (with bias) into the
            // `[N, C_out, out_h, out_w]` output. `weight` is already laid out as
            // `[C_out, K]`, so it needs no repacking. This routes the arithmetic
            // through the tuned GEMM instead of a naive per-output accumulation,
            // and produces the same cross-correlation result as before.
            let ohw = output_height * output_width;
            let k_dim = in_channels * kernel_h * kernel_w;
            let n_cols = batch_size * ohw;
            let kh_kw = kernel_h * kernel_w;

            let mut output_vec = vec![0f32; batch_size * out_channels * ohw];

            if !output_vec.is_empty() {
                // Build cols row by row (one row per kernel-input index `k`), so
                // each row is written contiguously.
                let mut cols = vec![0f32; k_dim * n_cols];
                cols.par_chunks_mut(n_cols)
                    .enumerate()
                    .for_each(|(k, row)| {
                        let ic = k / kh_kw;
                        let rem = k % kh_kw;
                        let ky = rem / kernel_w;
                        let kx = rem % kernel_w;
                        for (c, slot) in row.iter_mut().enumerate() {
                            let n = c / ohw;
                            let p = c % ohw;
                            let oh = p / output_width;
                            let ow = p % output_width;
                            let ih = oh * stride.0 + ky;
                            let iw = ow * stride.1 + kx;
                            // Padded coordinate; the valid (unpadded) region is
                            // [padding, dim + padding); everything else is zero pad.
                            if ih >= padding.0
                                && iw >= padding.1
                                && ih < input_height + padding.0
                                && iw < input_width + padding.1
                            {
                                let ih = ih - padding.0;
                                let iw = iw - padding.1;
                                let idx =
                                    ((n * in_channels + ic) * input_height + ih) * input_width + iw;
                                *slot = input_data[idx];
                            }
                        }
                    });

                let mut gemm_out = vec![0f32; out_channels * n_cols];
                // SAFETY: `weight_data` is [C_out, k_dim], `cols` is [k_dim,
                // n_cols], and `gemm_out` is [C_out, n_cols]; all are contiguous
                // row-major and the dimensions match the GEMM signature.
                unsafe {
                    crate::ops::linalg::gemm_f32(
                        out_channels,
                        k_dim,
                        n_cols,
                        weight_data.as_ptr(),
                        cols.as_ptr(),
                        gemm_out.as_mut_ptr(),
                    );
                }

                // Scatter [C_out, N*ohw] into [N, C_out, ohw], adding bias. For a
                // given (n, oc) the source and destination are contiguous `ohw`
                // slabs.
                output_vec
                    .par_chunks_mut(ohw)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let n = chunk_idx / out_channels;
                        let oc = chunk_idx % out_channels;
                        let b = bias_data.map(|bd| bd[oc]).unwrap_or(0.0);
                        let base = oc * n_cols + n * ohw;
                        for (o, &v) in out_chunk.iter_mut().zip(&gemm_out[base..base + ohw]) {
                            *o = v + b;
                        }
                    });
            }

            let requires_grad = input.requires_grad()
                || weight.requires_grad()
                || bias.is_some_and(|b| b.requires_grad());
            let output_data = TensorData::from_vec_f32(output_vec, input.device());
            let mut output = Tensor::new(
                Arc::new(output_data),
                output_shape,
                DataType::Float32,
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
                let bias_requires_grad = bias.is_some_and(|b| b.requires_grad());
                if bias_requires_grad {
                    deps.push(bias.unwrap().id());
                }
                let grad_fn = Arc::new(Conv2dBackward {
                    input: input.detach(),
                    weight: weight.detach(),
                    input_id: input.id(),
                    weight_id: weight.id(),
                    bias_id: bias.map(|b| b.id()),
                    input_requires_grad: input.requires_grad(),
                    weight_requires_grad: weight.requires_grad(),
                    bias_requires_grad,
                    stride,
                    padding,
                    deps,
                });
                output.set_grad_fn(Some(grad_fn.clone()));
                add_to_graph(&output, Some(grad_fn))?;
            }

            Ok(output)
        }
        _ => Err(MinitensorError::invalid_operation(
            "conv2d is implemented only for Float32 CPU tensors",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor, TensorData},
    };

    #[test]
    fn test_conv2d_basic() {
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1., 2., 3., 4.],
                Device::cpu(),
            )),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.], Device::cpu())),
            Shape::new(vec![1, 1, 1, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let bias = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.], Device::cpu())),
            Shape::new(vec![1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let out = conv2d(&input, &weight, Some(&bias), (1, 1), (0, 0)).unwrap();
        let data = out.data().as_f32_slice().unwrap();
        assert_eq!(data, &[2., 3., 4., 5.]);
    }

    #[test]
    fn test_conv2d_padding_and_stride() {
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(input_data, Device::cpu())),
            Shape::new(vec![1, 1, 4, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1., 0., 0., 1.],
                Device::cpu(),
            )),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let out = conv2d(&input, &weight, None, (2, 2), (1, 1)).unwrap();
        assert_eq!(out.shape(), &Shape::new(vec![1, 1, 3, 3]));
        let data = out.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1., 3., 0., 9., 17., 8., 0., 14., 16.]);
    }

    #[test]
    fn test_conv2d_invalid_kernel() {
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![0.; 4], Device::cpu())),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![0.; 25], Device::cpu())),
            Shape::new(vec![1, 1, 5, 5]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert!(result.is_err());
    }
}
