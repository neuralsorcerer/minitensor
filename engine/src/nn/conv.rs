// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    init::{init_bias, init_parameter, InitMethod},
    Layer,
};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::{collections::HashMap, sync::Arc};

/// 2D Convolutional layer
///
/// Applies a 2D convolution over an input signal composed of several input planes.
/// Input shape: (N, C_in, H_in, W_in)
/// Output shape: (N, C_out, H_out, W_out)
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2d {
    /// Create a new 2D convolutional layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of channels in the input image
    /// * `out_channels` - Number of channels produced by the convolution
    /// * `kernel_size` - Size of the convolving kernel (height, width)
    /// * `stride` - Stride of the convolution. Default: (1, 1)
    /// * `padding` - Zero-padding added to both sides of the input. Default: (0, 0)
    /// * `bias` - If true, adds a learnable bias to the output. Default: true
    /// * `device` - Device to place the layer parameters on
    /// * `dtype` - Data type for the layer parameters
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));

        // Initialize weight tensor with shape [out_channels, in_channels, kernel_height, kernel_width]
        let weight_shape = Shape::new(vec![
            out_channels,
            in_channels,
            kernel_size.0,
            kernel_size.1,
        ]);
        let weight = init_parameter(weight_shape, InitMethod::HeUniform, dtype, device)?;

        // Initialize bias tensor if requested
        let bias_tensor = if bias {
            let bias_shape = Shape::new(vec![out_channels]);
            Some(init_bias(bias_shape, dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Create a new Conv2d layer with custom initialization
    pub fn new_with_init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: bool,
        weight_init: InitMethod,
        bias_init: Option<InitMethod>,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));

        // Initialize weight tensor
        let weight_shape = Shape::new(vec![
            out_channels,
            in_channels,
            kernel_size.0,
            kernel_size.1,
        ]);
        let weight = init_parameter(weight_shape, weight_init, dtype, device)?;

        // Initialize bias tensor if requested
        let bias_tensor = if bias {
            let bias_shape = Shape::new(vec![out_channels]);
            let init_method = bias_init.unwrap_or(InitMethod::Zeros);
            Some(init_parameter(bias_shape, init_method, dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Get input channels count
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get output channels count
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get stride
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Get padding
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// Get the weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor (if it exists)
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Calculate output dimensions for given input dimensions
    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let output_height =
            (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width =
            (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (output_height, output_width)
    }

    /// Get named parameters for this layer
    pub fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &self.weight);
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    /// Get named mutable parameters for this layer
    pub fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }
}

impl Layer for Conv2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Validate input dimensions - expect 4D tensor [N, C, H, W]
        if input.ndim() != 4 {
            return Err(MinitensorError::invalid_operation(
                "Conv2d expects 4D input tensor with shape [batch_size, channels, height, width]",
            ));
        }

        let batch_size = input.size(0)?;
        let input_channels = input.size(1)?;
        let input_height = input.size(2)?;
        let input_width = input.size(3)?;

        // Validate input channels
        if input_channels != self.in_channels {
            return Err(MinitensorError::shape_mismatch(
                vec![self.in_channels],
                vec![input_channels],
            ));
        }

        // Calculate output dimensions
        let (output_height, output_width) = self.output_size(input_height, input_width);

        let output_shape = Shape::new(vec![
            batch_size,
            self.out_channels,
            output_height,
            output_width,
        ]);

        match (input.dtype(), input.device().is_cpu()) {
            (DataType::Float32, true) => {
                let input_data = input
                    .data()
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::invalid_operation("Expected f32 input data"))?;
                let weight_data = self.weight.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::invalid_operation("Expected f32 weight data")
                })?;
                let bias_data = if let Some(ref bias) = self.bias {
                    Some(bias.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::invalid_operation("Expected f32 bias data")
                    })?)
                } else {
                    None
                };

                let mut output_vec =
                    vec![0f32; batch_size * self.out_channels * output_height * output_width];

                for n in 0..batch_size {
                    for oc in 0..self.out_channels {
                        for oh in 0..output_height {
                            for ow in 0..output_width {
                                let mut sum = 0f32;
                                for ic in 0..self.in_channels {
                                    for kh in 0..self.kernel_size.0 {
                                        for kw in 0..self.kernel_size.1 {
                                            let h_in = oh * self.stride.0 + kh;
                                            let w_in = ow * self.stride.1 + kw;
                                            if h_in < self.padding.0
                                                || w_in < self.padding.1
                                                || h_in >= input_height + self.padding.0
                                                || w_in >= input_width + self.padding.1
                                            {
                                                continue;
                                            }
                                            let ih = h_in - self.padding.0;
                                            let iw = w_in - self.padding.1;
                                            if ih < input_height && iw < input_width {
                                                let input_idx = ((n * self.in_channels + ic)
                                                    * input_height
                                                    + ih)
                                                    * input_width
                                                    + iw;
                                                let weight_idx = ((oc * self.in_channels + ic)
                                                    * self.kernel_size.0
                                                    + kh)
                                                    * self.kernel_size.1
                                                    + kw;
                                                sum +=
                                                    input_data[input_idx] * weight_data[weight_idx];
                                            }
                                        }
                                    }
                                }
                                if let Some(bias) = bias_data {
                                    sum += bias[oc];
                                }
                                let out_idx = ((n * self.out_channels + oc) * output_height + oh)
                                    * output_width
                                    + ow;
                                output_vec[out_idx] = sum;
                            }
                        }
                    }
                }

                let requires_grad = input.requires_grad()
                    || self.weight.requires_grad()
                    || self.bias.as_ref().map_or(false, |b| b.requires_grad());
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
                "Conv2d forward is implemented only for Float32 CPU tensors",
            )),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{DataType, Shape, Tensor, TensorData};
    use std::sync::Arc;

    #[test]
    fn test_conv2d_creation() {
        let layer = Conv2d::new(
            3,
            16,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        assert_eq!(layer.in_channels(), 3);
        assert_eq!(layer.out_channels(), 16);
        assert_eq!(layer.kernel_size(), (3, 3));
        assert_eq!(layer.stride(), (1, 1));
        assert_eq!(layer.padding(), (1, 1));
        assert_eq!(layer.weight().shape(), &Shape::new(vec![16, 3, 3, 3]));
        assert!(layer.bias().is_some());
        assert_eq!(layer.bias().unwrap().shape(), &Shape::new(vec![16]));
    }

    #[test]
    fn test_conv2d_without_bias() {
        let layer = Conv2d::new(
            3,
            16,
            (3, 3),
            None,
            None,
            false,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        assert_eq!(layer.stride(), (1, 1)); // Default stride
        assert_eq!(layer.padding(), (0, 0)); // Default padding
        assert!(layer.bias().is_none());
    }

    #[test]
    fn test_conv2d_output_size_calculation() {
        let layer = Conv2d::new(
            3,
            16,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        // With padding=1, stride=1, kernel=3x3, input 32x32 should give output 32x32
        let (out_h, out_w) = layer.output_size(32, 32);
        assert_eq!(out_h, 32);
        assert_eq!(out_w, 32);

        // Without padding, stride=1, kernel=3x3, input 32x32 should give output 30x30
        let layer_no_pad = Conv2d::new(
            3,
            16,
            (3, 3),
            Some((1, 1)),
            Some((0, 0)),
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();
        let (out_h, out_w) = layer_no_pad.output_size(32, 32);
        assert_eq!(out_h, 30);
        assert_eq!(out_w, 30);
    }

    #[test]
    fn test_conv2d_parameters() {
        let mut layer = Conv2d::new(
            3,
            16,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let mut_params = layer.parameters_mut();
        assert_eq!(mut_params.len(), 2);
    }

    #[test]
    fn test_conv2d_forward_shape_validation() {
        let mut layer = Conv2d::new(
            3,
            16,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        // Test with correct input shape [batch=2, channels=3, height=32, width=32]
        let input = Tensor::zeros(
            Shape::new(vec![2, 3, 32, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &Shape::new(vec![2, 16, 32, 32]));

        // Test with incorrect number of channels
        let wrong_input = Tensor::zeros(
            Shape::new(vec![2, 5, 32, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong number of dimensions
        let wrong_dim_input = Tensor::zeros(
            Shape::new(vec![2, 3, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_named_parameters() {
        let mut layer = Conv2d::new(
            3,
            16,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        let named_params = layer.named_parameters();
        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));

        let named_params_mut = layer.named_parameters_mut();
        assert_eq!(named_params_mut.len(), 2);
        assert!(named_params_mut.contains_key("weight"));
        assert!(named_params_mut.contains_key("bias"));
    }

    #[test]
    fn test_conv2d_forward_computation() {
        let mut layer = Conv2d::new(
            1,
            1,
            (1, 1),
            None,
            None,
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        // Set weight to 1 and bias to 1 for easy verification
        layer.weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0], Device::cpu())),
            Shape::new(vec![1, 1, 1, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        if let Some(ref mut b) = layer.bias {
            *b = Tensor::new(
                Arc::new(TensorData::from_vec_f32(vec![1.0], Device::cpu())),
                Shape::new(vec![1]),
                DataType::Float32,
                Device::cpu(),
                false,
            );
        }

        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0],
                Device::cpu(),
            )),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &Shape::new(vec![1, 1, 2, 2]));
        let data = output.data().as_f32_slice().unwrap();
        assert_eq!(data, &[2.0, 3.0, 4.0, 5.0]);
    }
}
