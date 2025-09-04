// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    init::{init_bias, init_parameter, InitMethod},
    Layer,
};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    operations::linalg::matmul,
    tensor::{DataType, Shape, Tensor},
};
use std::collections::HashMap;

/// DenseLayer (fully connected) layer
///
/// Applies a dense layer transformation: y = xW^T + b
/// where W is the weight matrix and b is the bias vector
pub struct DenseLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl DenseLayer {
    /// Create a new dense layer
    ///
    /// # Arguments
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample  
    /// * `bias` - If set to false, the layer will not learn an additive bias
    /// * `device` - Device to place the layer parameters on
    /// * `dtype` - Data type for the layer parameters
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        // Initialize weight matrix with Xavier uniform initialization
        let weight_shape = Shape::new(vec![out_features, in_features]);
        let weight = init_parameter(weight_shape, InitMethod::XavierUniform, dtype, device)?;

        // Initialize bias vector if requested
        let bias_tensor = if bias {
            let bias_shape = Shape::new(vec![out_features]);
            Some(init_bias(bias_shape, dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        })
    }

    /// Create a new dense layer with custom initialization
    pub fn new_with_init(
        in_features: usize,
        out_features: usize,
        bias: bool,
        weight_init: InitMethod,
        bias_init: Option<InitMethod>,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        // Initialize weight matrix
        let weight_shape = Shape::new(vec![out_features, in_features]);
        let weight = init_parameter(weight_shape, weight_init, dtype, device)?;

        // Initialize bias vector if requested
        let bias_tensor = if bias {
            let bias_shape = Shape::new(vec![out_features]);
            let init_method = bias_init.unwrap_or(InitMethod::Zeros);
            Some(init_parameter(bias_shape, init_method, dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        })
    }

    /// Get input features count
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features count
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get the weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor (if it exists)
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
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

impl Layer for DenseLayer {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Validate input dimensions
        if input.ndim() < 2 {
            return Err(MinitensorError::invalid_operation(
                "DenseLayer expects input with at least 2 dimensions (batch_size, features)",
            ));
        }

        let input_features = input.size(input.ndim() - 1)?;
        if input_features != self.in_features {
            return Err(MinitensorError::shape_mismatch(
                vec![self.in_features],
                vec![input_features],
            ));
        }

        // Perform matrix multiplication: input @ weight.T
        // Note: We need to transpose the weight matrix since it's stored as [out_features, in_features]
        // but we need [in_features, out_features] for the multiplication
        let weight_t = self.weight.transpose(0, 1)?;
        let mut output = matmul(input, &weight_t)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Broadcast bias across all batch dimensions
            output = output.add(bias)?;
        }

        Ok(output)
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

// Note: Module is blanket-implemented for all Layers; named_parameters helpers are available as inherent methods.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{DataType, Shape};

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();

        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert_eq!(layer.weight().shape(), &Shape::new(vec![5, 10]));
        assert!(layer.bias().is_some());
        assert_eq!(layer.bias().unwrap().shape(), &Shape::new(vec![5]));
    }

    #[test]
    fn test_dense_layer_without_bias() {
        let layer = DenseLayer::new(10, 5, false, Device::cpu(), DataType::Float32).unwrap();

        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(layer.bias().is_none());
    }

    #[test]
    fn test_dense_layer_parameters() {
        let mut layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let mut_params = layer.parameters_mut();
        assert_eq!(mut_params.len(), 2);
    }

    #[test]
    fn test_dense_layer_parameters_no_bias() {
        let mut layer = DenseLayer::new(10, 5, false, Device::cpu(), DataType::Float32).unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 1); // weight only

        let mut_params = layer.parameters_mut();
        assert_eq!(mut_params.len(), 1);
    }

    #[test]
    fn test_dense_layer_named_parameters() {
        let mut layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();

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
    fn test_dense_layer_custom_init() {
        let layer = DenseLayer::new_with_init(
            10,
            5,
            true,
            InitMethod::HeUniform,
            Some(InitMethod::Constant(0.1)),
            Device::cpu(),
            DataType::Float32,
        )
        .unwrap();

        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_dense_layer_forward_shape_validation() {
        let mut layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();

        // Test with correct input shape
        let input = Tensor::zeros(
            Shape::new(vec![2, 10]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&input).unwrap();
        assert_eq!(result.shape(), &Shape::new(vec![2, 5]));

        // Test with incorrect input shape
        let wrong_input = Tensor::zeros(
            Shape::new(vec![2, 8]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dense_layer_forward_dimension_validation() {
        let mut layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();

        // Test with 1D input (should fail)
        let input_1d = Tensor::zeros(
            Shape::new(vec![10]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&input_1d);
        assert!(result.is_err());
    }
}
