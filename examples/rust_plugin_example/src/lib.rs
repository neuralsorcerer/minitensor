// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    Plugin, PluginInfo, VersionInfo, CustomOp, CustomOpRegistry, CustomOpBuilder,
    Result, MinitensorError, Tensor, DataType,
    tensor::Shape, device::Device,
    autograd::{GradientFunction, TensorId},
};
use std::sync::Arc;
use std::collections::HashMap;

/// Example plugin implementation
pub struct RustExamplePlugin {
    info: PluginInfo,
}

impl RustExamplePlugin {
    /// Create a new instance of the plugin
    pub fn new() -> Self {
        Self {
            info: PluginInfo {
                name: "rust_example_plugin".to_string(),
                version: VersionInfo::new(1, 0, 0),
                description: "Example Rust plugin demonstrating custom operations".to_string(),
                author: "Rust Plugin Developer".to_string(),
                min_minitensor_version: VersionInfo::new(0, 1, 0),
                max_minitensor_version: Some(VersionInfo::new(1, 0, 0)),
            },
        }
    }
}

impl Plugin for RustExamplePlugin {
    fn info(&self) -> &PluginInfo {
        &self.info
    }

    fn initialize(&self, registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust example plugin initialized!");

        // Register our custom operations
        for op in self.custom_operations() {
            registry.register(op)?;
        }

        Ok(())
    }

    fn cleanup(&self, registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust example plugin cleaned up!");

        // Unregister our custom operations
        for op in self.custom_operations() {
            let _ = registry.unregister(op.name()); // Ignore errors during cleanup
        }

        Ok(())
    }

    fn custom_operations(&self) -> Vec<Arc<dyn CustomOp>> {
        vec![
            create_abs_operation(),
            create_clamp_operation(),
            create_gelu_operation(),
        ]
    }
}

/// Create an absolute value operation
fn create_abs_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_abs", 1)
        .forward(|inputs| {
            let x = inputs[0];
            // Implement element-wise absolute value
            // This is a simplified implementation - real version would use optimized kernels
            let data = x.data();
            let shape = x.shape().clone();
            let dtype = x.dtype();
            let device = x.device();

            // Create output tensor with same properties
            let mut output_data = Vec::new();
            match dtype {
                DataType::F32 => {
                    let input_slice = data.as_f32_slice()?;
                    output_data = input_slice.iter().map(|&v| v.abs()).collect::<Vec<f32>>();
                }
                DataType::F64 => {
                    let input_slice = data.as_f64_slice()?;
                    output_data = input_slice.iter().map(|&v| v.abs()).collect::<Vec<f64>>();
                }
                _ => return Err(MinitensorError::invalid_operation(
                    "Absolute value operation only supports f32 and f64 types"
                )),
            }

            Tensor::from_data(output_data, shape, device)
        })
        .backward(|grad_output, input_ids, input_shapes, input_dtypes, input_devices| {
            // Gradient of abs(x) is sign(x)
            let mut gradients = HashMap::new();

            if !input_ids.is_empty() {
                // For simplicity, we'll return the gradient as-is
                // Real implementation would compute sign(x) * grad_output
                gradients.insert(input_ids[0], grad_output.clone());
            }

            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs.len() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "Absolute value operation requires exactly one input"
                ));
            }
            Ok(())
        })
        .output_shape(|input_shapes| Ok(input_shapes[0].clone()))
        .output_dtype(|input_dtypes| Ok(input_dtypes[0]))
        .output_device(|input_devices| Ok(input_devices[0].clone()))
        .build()
        .unwrap()
}

/// Create a clamp operation (clamp values between min and max)
fn create_clamp_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_clamp", 3) // input, min, max
        .forward(|inputs| {
            let x = inputs[0];
            let min_val = inputs[1];
            let max_val = inputs[2];

            // Simplified implementation - real version would be more efficient
            // clamp(x, min, max) = max(min(x, max), min)

            // For demo purposes, assume scalar min/max values
            let min_scalar = min_val.item::<f32>()?;
            let max_scalar = max_val.item::<f32>()?;

            let data = x.data();
            let shape = x.shape().clone();
            let dtype = x.dtype();
            let device = x.device();

            let mut output_data = Vec::new();
            match dtype {
                DataType::F32 => {
                    let input_slice = data.as_f32_slice()?;
                    output_data = input_slice.iter()
                        .map(|&v| v.clamp(min_scalar, max_scalar))
                        .collect::<Vec<f32>>();
                }
                _ => return Err(MinitensorError::invalid_operation(
                    "Clamp operation currently only supports f32 type"
                )),
            }

            Tensor::from_data(output_data, shape, device)
        })
        .backward(|grad_output, input_ids, input_shapes, input_dtypes, input_devices| {
            // Gradient of clamp is 1 where min <= x <= max, 0 elsewhere
            let mut gradients = HashMap::new();

            if !input_ids.is_empty() {
                // Simplified: return gradient for input tensor only
                gradients.insert(input_ids[0], grad_output.clone());
            }

            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs.len() != 3 {
                return Err(MinitensorError::invalid_argument(
                    "Clamp operation requires exactly three inputs: tensor, min, max"
                ));
            }

            // Check that min and max are scalars
            if inputs[1].shape().numel() != 1 || inputs[2].shape().numel() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "Min and max values must be scalars"
                ));
            }

            Ok(())
        })
        .output_shape(|input_shapes| Ok(input_shapes[0].clone()))
        .output_dtype(|input_dtypes| Ok(input_dtypes[0]))
        .output_device(|input_devices| Ok(input_devices[0].clone()))
        .build()
        .unwrap()
}

/// Create a GELU (Gaussian Error Linear Unit) activation function
fn create_gelu_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_gelu", 1)
        .forward(|inputs| {
            let x = inputs[0];

            // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
            // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

            let data = x.data();
            let shape = x.shape().clone();
            let dtype = x.dtype();
            let device = x.device();

            let mut output_data = Vec::new();
            match dtype {
                DataType::F32 => {
                    let input_slice = data.as_f32_slice()?;
                    output_data = input_slice.iter().map(|&v| {
                        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
                        let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                        0.5 * v * (1.0 + inner.tanh())
                    }).collect::<Vec<f32>>();
                }
                DataType::F64 => {
                    let input_slice = data.as_f64_slice()?;
                    output_data = input_slice.iter().map(|&v| {
                        let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
                        let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                        0.5 * v * (1.0 + inner.tanh())
                    }).collect::<Vec<f64>>();
                }
                _ => return Err(MinitensorError::invalid_operation(
                    "GELU operation only supports f32 and f64 types"
                )),
            }

            Tensor::from_data(output_data, shape, device)
        })
        .backward(|grad_output, input_ids, input_shapes, input_dtypes, input_devices| {
            // Gradient of GELU is more complex - simplified implementation
            let mut gradients = HashMap::new();

            if !input_ids.is_empty() {
                // For demo purposes, return the gradient as-is
                // Real implementation would compute the actual GELU derivative
                gradients.insert(input_ids[0], grad_output.clone());
            }

            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs.len() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "GELU operation requires exactly one input"
                ));
            }
            Ok(())
        })
        .output_shape(|input_shapes| Ok(input_shapes[0].clone()))
        .output_dtype(|input_dtypes| Ok(input_dtypes[0]))
        .output_device(|input_devices| Ok(input_devices[0].clone()))
        .build()
        .unwrap()
}

/// Export function for dynamic loading
/// This is the entry point that the plugin system will call
#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    let plugin = RustExamplePlugin::new();
    Box::into_raw(Box::new(plugin))
}

/// Export function for plugin cleanup
#[no_mangle]
pub extern "C" fn destroy_plugin(plugin: *mut dyn Plugin) {
    if !plugin.is_null() {
        unsafe {
            let _ = Box::from_raw(plugin);
        }
    }
}

/// Get plugin information without creating the plugin
#[no_mangle]
pub extern "C" fn get_plugin_info() -> PluginInfo {
    RustExamplePlugin::new().info().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_creation() {
        let plugin = RustExamplePlugin::new();
        let info = plugin.info();

        assert_eq!(info.name, "rust_example_plugin");
        assert_eq!(info.version.major, 1);
        assert_eq!(info.version.minor, 0);
        assert_eq!(info.version.patch, 0);
    }

    #[test]
    fn test_custom_operations() {
        let plugin = RustExamplePlugin::new();
        let operations = plugin.custom_operations();

        assert_eq!(operations.len(), 3);

        let op_names: Vec<&str> = operations.iter().map(|op| op.name()).collect();
        assert!(op_names.contains(&"rust_abs"));
        assert!(op_names.contains(&"rust_clamp"));
        assert!(op_names.contains(&"rust_gelu"));
    }
}