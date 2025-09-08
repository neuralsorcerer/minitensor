// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
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

    if let Some(b) = bias {
        if b.ndim() != 1 || b.size(0)? != out_channels {
            return Err(MinitensorError::shape_mismatch(
                vec![out_channels],
                vec![b.size(0)?],
            ));
        }
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

            let mut output_vec =
                vec![0f32; batch_size * out_channels * output_height * output_width];

            output_vec
                .par_chunks_mut(output_height * output_width)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let n = chunk_idx / out_channels;
                    let oc = chunk_idx % out_channels;
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let mut sum = 0f32;
                            for ic in 0..in_channels {
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let h_in = oh * stride.0 + kh;
                                        let w_in = ow * stride.1 + kw;
                                        if h_in < padding.0
                                            || w_in < padding.1
                                            || h_in >= input_height + padding.0
                                            || w_in >= input_width + padding.1
                                        {
                                            continue;
                                        }
                                        let ih = h_in - padding.0;
                                        let iw = w_in - padding.1;
                                        if ih < input_height && iw < input_width {
                                            let input_idx = ((n * in_channels + ic) * input_height
                                                + ih)
                                                * input_width
                                                + iw;
                                            let weight_idx = ((oc * in_channels + ic) * kernel_h
                                                + kh)
                                                * kernel_w
                                                + kw;
                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                            if let Some(bias) = bias_data {
                                sum += bias[oc];
                            }
                            out_chunk[oh * output_width + ow] = sum;
                        }
                    }
                });

            let requires_grad = input.requires_grad()
                || weight.requires_grad()
                || bias.map_or(false, |b| b.requires_grad());
            let output_data = TensorData::from_vec_f32(output_vec, input.device());
            Ok(Tensor::new(
                Arc::new(output_data),
                output_shape,
                DataType::Float32,
                input.device(),
                requires_grad,
            ))
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
