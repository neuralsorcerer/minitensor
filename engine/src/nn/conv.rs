// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    Layer,
    init::{InitMethod, init_bias, init_parameter},
};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor},
};

/// Both convolution layers reject a grouping the channels cannot be split
/// into, at construction rather than at the first forward pass -- the weight
/// shape depends on it, so a layer built with a bad `groups` would otherwise
/// allocate parameters that no input could ever match.
fn check_groups(in_channels: usize, out_channels: usize, groups: usize) -> Result<()> {
    if groups == 0 {
        return Err(MinitensorError::invalid_argument(
            "groups must be greater than zero",
        ));
    }
    if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(MinitensorError::invalid_argument(format!(
            "groups={groups} must divide both in_channels={in_channels} and out_channels={out_channels}"
        )));
    }
    Ok(())
}

/// 2D Convolutional layer
///
/// Applies a 2D convolution over an input signal composed of several input planes.
/// Input shape: (N, C_in, H_in, W_in)
/// Output shape: (N, C_out, H_out, W_out)
#[derive(Clone)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
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
    /// * `dilation` - Spacing between kernel taps. Default: (1, 1)
    /// * `groups` - How many independent convolutions to split the channels
    ///   into; must divide both channel counts. `groups == in_channels` is a
    ///   depthwise convolution. Default: 1
    /// * `bias` - If true, adds a learnable bias to the output. Default: true
    /// * `device` - Device to place the layer parameters on
    /// * `dtype` - Data type for the layer parameters
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);
        check_groups(in_channels, out_channels, groups)?;

        // Initialize weight tensor with shape [out_channels, in_channels, kernel_height, kernel_width]
        let weight_shape = Shape::new(vec![
            out_channels,
            in_channels / groups,
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
            dilation,
            groups,
        })
    }

    /// Create a new Conv2d layer with custom initialization
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: bool,
        weight_init: InitMethod,
        bias_init: Option<InitMethod>,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);
        check_groups(in_channels, out_channels, groups)?;

        // Initialize weight tensor
        let weight_shape = Shape::new(vec![
            out_channels,
            in_channels / groups,
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
            dilation,
            groups,
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

    /// Get dilation
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    /// Get the number of groups
    pub fn groups(&self) -> usize {
        self.groups
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
        // The kernel's *span* once its taps are spread by the dilation, which
        // is what the padded input has to accommodate.
        let span_h = self.dilation.0 * (self.kernel_size.0 - 1) + 1;
        let span_w = self.dilation.1 * (self.kernel_size.1 - 1) + 1;
        let output_height = (input_height + 2 * self.padding.0 - span_h) / self.stride.0 + 1;
        let output_width = (input_width + 2 * self.padding.1 - span_w) / self.stride.1 + 1;
        (output_height, output_width)
    }
}

impl Layer for Conv2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Delegate actual computation to ops::conv::conv2d
        crate::ops::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }

    crate::weight_and_optional_bias_parameters!();
}

/// 1-D convolutional layer over `[N, C_in, L]` signals.
///
/// Wraps [`crate::ops::conv1d`], which in turn defers to the 2-D kernel with a
/// singleton height, so there is one implementation of the windowing and one
/// backward pass rather than two to keep in step.
#[derive(Clone)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv1d {
    /// Create a new 1D convolutional layer.
    ///
    /// `stride` defaults to 1, `padding` to 0, `dilation` to 1 and `groups` to
    /// 1, matching the 2-D layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 {
            return Err(MinitensorError::invalid_argument(
                "Conv1d requires non-zero in_channels, out_channels and kernel_size",
            ));
        }
        if !dtype.is_float() {
            return Err(MinitensorError::invalid_argument(
                "Conv1d parameters must have a floating point dtype",
            ));
        }

        let groups = groups.unwrap_or(1);
        check_groups(in_channels, out_channels, groups)?;

        let weight_shape = Shape::new(vec![out_channels, in_channels / groups, kernel_size]);
        let weight = init_parameter(weight_shape, InitMethod::HeUniform, dtype, device)?;
        let bias_tensor = if bias {
            Some(init_bias(Shape::new(vec![out_channels]), dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_size,
            stride: stride.unwrap_or(1),
            padding: padding.unwrap_or(0),
            dilation: dilation.unwrap_or(1),
            groups,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn dilation(&self) -> usize {
        self.dilation
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Layer for Conv1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::conv1d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }

    crate::weight_and_optional_bias_parameters!();
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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

    #[test]
    fn test_conv2d_stride_padding_output() {
        let mut layer = Conv2d::new(
            1,
            1,
            (3, 3),
            Some((2, 2)),
            Some((1, 1)),
            None,
            None,
            true,
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        let input = Tensor::zeros(
            Shape::new(vec![1, 1, 7, 7]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &Shape::new(vec![1, 1, 4, 4]));
    }
}
