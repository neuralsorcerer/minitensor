// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{error::Result, nn::layer::Layer, tensor::Tensor};
use std::collections::HashMap;

/// Sequential container for neural network layers
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    training: bool,
}

impl Sequential {
    /// Create a new empty sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Create a sequential model from a vector of layers
    pub fn from_layers(layers: Vec<Box<dyn Layer>>) -> Self {
        Self {
            layers,
            training: true,
        }
    }

    /// Add a layer to the sequential model
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    /// Get the number of layers in the model
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get a reference to a layer by index
    pub fn get_layer(&self, index: usize) -> Option<&dyn Layer> {
        self.layers.get(index).map(|layer| layer.as_ref())
    }

    // Note: get_layer_mut is complex due to lifetime issues with trait objects
    // For now, we'll access layers through the sequential interface

    /// Apply a function to all layers
    pub fn apply_to_layers<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&mut dyn Layer) -> Result<()>,
    {
        for layer in &mut self.layers {
            f(layer.as_mut())?;
        }
        Ok(())
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sequential {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();

        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }
}

impl Sequential {
    /// Get named parameters of the sequential model
    pub fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut named_params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (j, param) in layer_params.iter().enumerate() {
                let name = format!("layer_{}.param_{}", i, j);
                named_params.insert(name, *param);
            }
        }

        named_params
    }

    /// Get named mutable parameters of the sequential model
    pub fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut named_params = HashMap::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_params = layer.parameters_mut();
            for (j, param) in layer_params.into_iter().enumerate() {
                let name = format!("layer_{}.param_{}", i, j);
                named_params.insert(name, param);
            }
        }

        named_params
    }
}

/// Builder pattern for creating sequential models
pub struct SequentialBuilder {
    layers: Vec<Box<dyn Layer>>,
}

impl SequentialBuilder {
    /// Create a new sequential builder
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer to the builder
    pub fn add(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Build the sequential model
    pub fn build(self) -> Sequential {
        Sequential::from_layers(self.layers)
    }
}

impl Default for SequentialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        nn::layer::Layer,
        tensor::{DataType, Shape, Tensor},
    };

    // Mock layer for testing
    struct MockLayer {
        weight: Tensor,
    }

    impl MockLayer {
        fn new(input_size: usize, output_size: usize) -> Self {
            let shape = Shape::new(vec![output_size, input_size]);
            let weight = Tensor::zeros(shape, DataType::Float32, Device::cpu(), true);
            Self { weight }
        }
    }

    impl Layer for MockLayer {
        fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
            // Simple identity forward pass for testing
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weight]
        }
    }

    #[test]
    fn test_sequential_creation() {
        let mut seq = Sequential::new();
        assert_eq!(seq.len(), 0);
        assert!(seq.is_empty());

        let layer = Box::new(MockLayer::new(10, 5));
        seq.add_layer(layer);
        assert_eq!(seq.len(), 1);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_sequential_forward() {
        let mut seq = Sequential::new();
        seq.add_layer(Box::new(MockLayer::new(10, 8)));
        seq.add_layer(Box::new(MockLayer::new(8, 5)));

        let input_shape = Shape::new(vec![1, 10]);
        let input = Tensor::zeros(input_shape, DataType::Float32, Device::cpu(), false);

        let output = seq.forward(&input).unwrap();
        // With our mock layers, output should be the same as input
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_sequential_parameters() {
        let mut seq = Sequential::new();
        seq.add_layer(Box::new(MockLayer::new(10, 8)));
        seq.add_layer(Box::new(MockLayer::new(8, 5)));

        let params = seq.parameters();
        assert_eq!(params.len(), 2); // Two layers, each with one parameter

        let mut_params = seq.parameters_mut();
        assert_eq!(mut_params.len(), 2);
    }

    #[test]
    fn test_sequential_builder() {
        let seq = SequentialBuilder::new()
            .add(Box::new(MockLayer::new(10, 8)))
            .add(Box::new(MockLayer::new(8, 5)))
            .build();

        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_parameters(), 80 + 40); // 10*8 + 8*5
    }

    #[test]
    fn test_sequential_training_mode() {
        let mut seq = Sequential::new();
        seq.add_layer(Box::new(MockLayer::new(10, 5)));

        // Test training mode switching
        seq.train();
        seq.eval();
        // No assertions here since our mock layer doesn't implement mode switching
        // but this tests that the methods can be called without errors
    }

    #[test]
    fn test_named_parameters() {
        let mut seq = Sequential::new();
        seq.add_layer(Box::new(MockLayer::new(10, 8)));
        seq.add_layer(Box::new(MockLayer::new(8, 5)));

        let named_params = seq.named_parameters();
        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("layer_0.param_0"));
        assert!(named_params.contains_key("layer_1.param_0"));

        let named_params_mut = seq.named_parameters_mut();
        assert_eq!(named_params_mut.len(), 2);
    }
}
