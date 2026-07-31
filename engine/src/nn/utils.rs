// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{Layer, Sequential};
use crate::tensor::DataType;
use std::collections::HashMap;

/// Utility functions for layer and model inspection
pub struct LayerUtils;

impl LayerUtils {
    /// Count the total number of parameters in a layer
    pub fn count_parameters(layer: &dyn Layer) -> usize {
        layer.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Count the total number of trainable parameters in a layer
    pub fn count_trainable_parameters(layer: &dyn Layer) -> usize {
        layer
            .parameters()
            .iter()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel())
            .sum()
    }

    /// Get parameter statistics for a layer
    pub fn parameter_stats(layer: &dyn Layer) -> ParameterStats {
        let params = layer.parameters();
        let mut total_params = 0usize;
        let mut trainable_params = 0usize;
        let mut counts = Vec::with_capacity(params.len());

        for p in &params {
            let n = p.numel();
            total_params += n;
            if p.requires_grad() {
                trainable_params += n;
            }
            counts.push(n);
        }

        ParameterStats {
            total_parameters: total_params,
            trainable_parameters: trainable_params,
            non_trainable_parameters: total_params - trainable_params,
            parameter_count_by_tensor: counts,
        }
    }

    /// Get memory usage statistics for a layer's parameters
    pub fn memory_usage(layer: &dyn Layer) -> MemoryUsage {
        let params = layer.parameters();
        let mut total_bytes = 0usize;
        let mut bytes_by_dtype = HashMap::with_capacity(params.len());

        for param in params {
            let dtype = param.dtype();
            let param_bytes = param.numel() * dtype.size_bytes();
            total_bytes += param_bytes;
            *bytes_by_dtype.entry(dtype).or_insert(0) += param_bytes;
        }

        MemoryUsage {
            total_bytes,
            bytes_by_dtype,
        }
    }

    /// Generate a summary string for a layer
    pub fn layer_summary(layer: &dyn Layer, layer_name: &str) -> String {
        let stats = Self::parameter_stats(layer);
        let memory = Self::memory_usage(layer);

        format!(
            "Layer: {}\n\
             Total Parameters: {}\n\
             Trainable Parameters: {}\n\
             Non-trainable Parameters: {}\n\
             Memory Usage: {:.2} KB ({} bytes)",
            layer_name,
            stats.total_parameters,
            stats.trainable_parameters,
            stats.non_trainable_parameters,
            memory.total_bytes as f64 / 1024.0,
            memory.total_bytes
        )
    }
}

/// Utility functions specifically for Sequential models
pub struct SequentialUtils;

impl SequentialUtils {
    /// Count total parameters in a sequential model
    pub fn count_parameters(model: &Sequential) -> usize {
        model.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Count trainable parameters in a sequential model
    pub fn count_trainable_parameters(model: &Sequential) -> usize {
        model
            .parameters()
            .iter()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel())
            .sum()
    }

    /// Get detailed parameter statistics for a sequential model
    pub fn parameter_stats(model: &Sequential) -> SequentialStats {
        let params = model.parameters();
        let mut total_params = 0usize;
        let mut trainable_params = 0usize;
        for p in &params {
            let n = p.numel();
            total_params += n;
            if p.requires_grad() {
                trainable_params += n;
            }
        }
        let non_trainable_params = total_params - trainable_params;

        let num_layers = model.len();
        let mut layer_stats = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if let Some(layer) = model.get_layer(i) {
                let stats = LayerUtils::parameter_stats(layer);
                layer_stats.push((i, stats));
            }
        }

        SequentialStats {
            total_parameters: total_params,
            trainable_parameters: trainable_params,
            non_trainable_parameters: non_trainable_params,
            num_layers,
            layer_stats,
        }
    }

    /// Generate a detailed summary of a sequential model
    pub fn model_summary(model: &Sequential, model_name: Option<&str>) -> String {
        let stats = Self::parameter_stats(model);
        let memory = LayerUtils::memory_usage(model);

        let mut summary = String::new();

        // Model header
        if let Some(name) = model_name {
            summary.push_str(&format!("Model: {}\n", name));
        } else {
            summary.push_str("Sequential Model\n");
        }
        summary.push_str(&"=".repeat(50));
        summary.push('\n');

        // Layer-by-layer breakdown
        for (layer_idx, layer_stats) in &stats.layer_stats {
            summary.push_str(&format!(
                "Layer {}: {} parameters ({} trainable)\n",
                layer_idx, layer_stats.total_parameters, layer_stats.trainable_parameters
            ));
        }

        summary.push_str(&"=".repeat(50));
        summary.push('\n');

        // Overall statistics
        summary.push_str(&format!(
            "Total Parameters: {}\n\
             Trainable Parameters: {}\n\
             Non-trainable Parameters: {}\n\
             Number of Layers: {}\n\
             Memory Usage: {:.2} KB ({} bytes)\n",
            stats.total_parameters,
            stats.trainable_parameters,
            stats.non_trainable_parameters,
            stats.num_layers,
            memory.total_bytes as f64 / 1024.0,
            memory.total_bytes
        ));

        summary
    }

    /// Rough upper-bound sketch of the memory a forward pass touches.
    ///
    /// `parameter_memory` is exact. The activation figures are not: [`Layer`]
    /// has no way to report an output shape, so this cannot trace widths
    /// through the model and instead assumes **every layer produces an
    /// activation the same size as the input**. A model that widens (a
    /// classifier's hidden layers) is under-counted; one that narrows is
    /// over-counted. Use it for order-of-magnitude budgeting, not for deciding
    /// whether a model fits.
    ///
    /// The element width comes from the model's own parameters rather than
    /// being assumed to be 4 bytes, so an f64 model is not reported at half
    /// its size. An empty model has no parameters to read a dtype from and
    /// falls back to `Float32`.
    ///
    /// [`Layer`]: crate::nn::Layer
    pub fn estimate_forward_memory(
        model: &Sequential,
        input_shape: &[usize],
        batch_size: usize,
    ) -> ForwardMemoryEstimate {
        let input_elements = input_shape.iter().product::<usize>() * batch_size;
        let parameter_memory = LayerUtils::memory_usage(model).total_bytes;

        let element_bytes = model
            .parameters()
            .first()
            .map_or(DataType::Float32, |param| param.dtype())
            .size_bytes();

        let input_memory = input_elements * element_bytes;
        let estimated_activation_memory = input_memory * model.len();

        ForwardMemoryEstimate {
            parameter_memory,
            estimated_activation_memory,
            estimated_total_memory: parameter_memory + estimated_activation_memory,
            input_memory,
        }
    }
}

/// Statistics about layer parameters
#[derive(Debug, Clone)]
pub struct ParameterStats {
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub non_trainable_parameters: usize,
    pub parameter_count_by_tensor: Vec<usize>,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub total_bytes: usize,
    pub bytes_by_dtype: HashMap<crate::tensor::DataType, usize>,
}

/// Statistics for sequential models
#[derive(Debug, Clone)]
pub struct SequentialStats {
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub non_trainable_parameters: usize,
    pub num_layers: usize,
    pub layer_stats: Vec<(usize, ParameterStats)>,
}

/// Forward pass memory estimation
#[derive(Debug, Clone)]
pub struct ForwardMemoryEstimate {
    pub parameter_memory: usize,
    pub estimated_activation_memory: usize,
    pub estimated_total_memory: usize,
    pub input_memory: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        nn::{DenseLayer, ReLU, SequentialBuilder},
        tensor::DataType,
    };

    #[test]
    fn test_layer_utils_parameter_counting() {
        // Create a DenseLayer with known parameter count
        let layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();

        let total_params = LayerUtils::count_parameters(&layer);
        assert_eq!(total_params, 10 * 5 + 5); // weight + bias

        let trainable_params = LayerUtils::count_trainable_parameters(&layer);
        assert_eq!(trainable_params, total_params); // All parameters are trainable
    }

    #[test]
    fn test_layer_utils_parameter_stats() {
        let layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();
        let stats = LayerUtils::parameter_stats(&layer);

        assert_eq!(stats.total_parameters, 55); // 10*5 + 5
        assert_eq!(stats.trainable_parameters, 55);
        assert_eq!(stats.non_trainable_parameters, 0);
        assert_eq!(stats.parameter_count_by_tensor.len(), 2); // weight + bias
    }

    #[test]
    fn test_layer_utils_memory_usage() {
        let layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();
        let memory = LayerUtils::memory_usage(&layer);

        assert_eq!(memory.total_bytes, 55 * 4); // 55 parameters * 4 bytes each (f32)
        assert!(memory.bytes_by_dtype.contains_key(&DataType::Float32));
    }

    #[test]
    fn test_layer_utils_summary() {
        let layer = DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap();
        let summary = LayerUtils::layer_summary(&layer, "DenseLayer");

        assert!(summary.contains("Layer: DenseLayer"));
        assert!(summary.contains("Total Parameters: 55"));
        assert!(summary.contains("Trainable Parameters: 55"));
    }

    #[test]
    fn test_sequential_utils_parameter_counting() {
        let model = SequentialBuilder::new()
            .add(Box::new(
                DenseLayer::new(10, 8, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .add(Box::new(ReLU::new()))
            .add(Box::new(
                DenseLayer::new(8, 5, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .build();

        let total_params = SequentialUtils::count_parameters(&model);
        assert_eq!(total_params, (10 * 8 + 8) + (8 * 5 + 5)); // Two dense layers

        let trainable_params = SequentialUtils::count_trainable_parameters(&model);
        assert_eq!(trainable_params, total_params);
    }

    #[test]
    fn test_sequential_utils_stats() {
        let model = SequentialBuilder::new()
            .add(Box::new(
                DenseLayer::new(10, 8, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .add(Box::new(ReLU::new()))
            .add(Box::new(
                DenseLayer::new(8, 5, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .build();

        let stats = SequentialUtils::parameter_stats(&model);

        assert_eq!(stats.num_layers, 3);
        assert_eq!(stats.total_parameters, (10 * 8 + 8) + (8 * 5 + 5));
        assert_eq!(stats.layer_stats.len(), 3);
    }

    #[test]
    fn test_sequential_utils_summary() {
        let model = SequentialBuilder::new()
            .add(Box::new(
                DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .build();

        let summary = SequentialUtils::model_summary(&model, Some("TestModel"));

        assert!(summary.contains("Model: TestModel"));
        assert!(summary.contains("Total Parameters:"));
        assert!(summary.contains("Number of Layers: 1"));
    }

    #[test]
    fn test_forward_memory_estimation() {
        let model = SequentialBuilder::new()
            .add(Box::new(
                DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .build();

        let estimate = SequentialUtils::estimate_forward_memory(&model, &[10], 32);

        // 10 * 5 weights + 5 biases, f32.
        assert_eq!(estimate.parameter_memory, (10 * 5 + 5) * 4);
        assert_eq!(estimate.input_memory, 10 * 32 * 4);
        // One layer, and the estimate charges each layer an input-sized
        // activation -- see the doc comment for why that is a sketch.
        assert_eq!(estimate.estimated_activation_memory, 10 * 32 * 4);
        assert_eq!(
            estimate.estimated_total_memory,
            estimate.parameter_memory + estimate.estimated_activation_memory
        );
    }

    #[test]
    fn forward_memory_estimate_follows_the_model_dtype() {
        let build = |dtype| {
            SequentialBuilder::new()
                .add(Box::new(
                    DenseLayer::new(10, 5, true, Device::cpu(), dtype).unwrap(),
                ))
                .build()
        };

        let f32_estimate =
            SequentialUtils::estimate_forward_memory(&build(DataType::Float32), &[10], 32);
        let f64_estimate =
            SequentialUtils::estimate_forward_memory(&build(DataType::Float64), &[10], 32);

        // The activation figures are a sketch, but the element width is not
        // guesswork: an f64 model must not be reported at f32 sizes.
        assert_eq!(f64_estimate.input_memory, 2 * f32_estimate.input_memory);
        assert_eq!(
            f64_estimate.parameter_memory,
            2 * f32_estimate.parameter_memory
        );
        assert_eq!(
            f64_estimate.estimated_activation_memory,
            2 * f32_estimate.estimated_activation_memory
        );
    }

    #[test]
    fn forward_memory_estimate_handles_a_model_with_no_parameters() {
        let model = SequentialBuilder::new().build();
        let estimate = SequentialUtils::estimate_forward_memory(&model, &[10], 32);

        assert_eq!(estimate.parameter_memory, 0);
        assert_eq!(estimate.input_memory, 10 * 32 * 4);
        assert_eq!(estimate.estimated_activation_memory, 0);
        assert_eq!(estimate.estimated_total_memory, 0);
    }
}
