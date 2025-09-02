// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{Layer, Sequential};
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
        let total_params = params.iter().map(|p| p.numel()).sum();
        let trainable_params = params
            .iter()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel())
            .sum();
        let non_trainable_params = total_params - trainable_params;

        ParameterStats {
            total_parameters: total_params,
            trainable_parameters: trainable_params,
            non_trainable_parameters: non_trainable_params,
            parameter_count_by_tensor: params.iter().map(|p| p.numel()).collect(),
        }
    }

    /// Get memory usage statistics for a layer's parameters
    pub fn memory_usage(layer: &dyn Layer) -> MemoryUsage {
        let params = layer.parameters();
        let mut total_bytes = 0;
        let mut bytes_by_dtype = HashMap::new();

        for param in params {
            let param_bytes = param.numel() * param.dtype().size_in_bytes();
            total_bytes += param_bytes;

            let dtype_name = format!("{:?}", param.dtype());
            *bytes_by_dtype.entry(dtype_name).or_insert(0) += param_bytes;
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
        let total_params = Self::count_parameters(model);
        let trainable_params = Self::count_trainable_parameters(model);
        let non_trainable_params = total_params - trainable_params;

        let mut layer_stats = Vec::new();
        for i in 0..model.len() {
            if let Some(layer) = model.get_layer(i) {
                let stats = LayerUtils::parameter_stats(layer);
                layer_stats.push((i, stats));
            }
        }

        SequentialStats {
            total_parameters: total_params,
            trainable_parameters: trainable_params,
            non_trainable_parameters: non_trainable_params,
            num_layers: model.len(),
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

    /// Calculate the theoretical memory usage for forward pass
    pub fn estimate_forward_memory(
        model: &Sequential,
        input_shape: &[usize],
        batch_size: usize,
    ) -> ForwardMemoryEstimate {
        // This is a simplified estimation
        // In practice, this would need to trace through each layer to get accurate intermediate tensor sizes

        let input_elements = input_shape.iter().product::<usize>() * batch_size;
        let parameter_memory = LayerUtils::memory_usage(model).total_bytes;

        // Rough estimate: assume each layer doubles the memory requirement for activations
        let estimated_activation_memory = input_elements * 4 * model.len(); // Assuming f32

        ForwardMemoryEstimate {
            parameter_memory,
            estimated_activation_memory,
            estimated_total_memory: parameter_memory + estimated_activation_memory,
            input_memory: input_elements * 4, // Assuming f32
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
    pub bytes_by_dtype: HashMap<String, usize>,
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

/// Extension trait to add utility methods to DataType
pub trait DataTypeExt {
    /// Get the size in bytes for this data type
    fn size_in_bytes(&self) -> usize;
}

impl DataTypeExt for crate::tensor::DataType {
    fn size_in_bytes(&self) -> usize {
        use crate::tensor::DataType;
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }
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
        assert!(memory.bytes_by_dtype.contains_key("Float32"));
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
    fn test_datatype_size_in_bytes() {
        use crate::tensor::DataType;

        assert_eq!(DataType::Float32.size_in_bytes(), 4);
        assert_eq!(DataType::Float64.size_in_bytes(), 8);
        assert_eq!(DataType::Int32.size_in_bytes(), 4);
        assert_eq!(DataType::Int64.size_in_bytes(), 8);
        assert_eq!(DataType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn test_forward_memory_estimation() {
        let model = SequentialBuilder::new()
            .add(Box::new(
                DenseLayer::new(10, 5, true, Device::cpu(), DataType::Float32).unwrap(),
            ))
            .build();

        let estimate = SequentialUtils::estimate_forward_memory(&model, &[10], 32);

        assert!(estimate.parameter_memory > 0);
        assert!(estimate.estimated_activation_memory > 0);
        assert!(estimate.estimated_total_memory > estimate.parameter_memory);
        assert_eq!(estimate.input_memory, 10 * 32 * 4); // 10 features * 32 batch * 4 bytes
    }
}
