// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    Plugin, PluginInfo, VersionInfo, CustomOp, CustomOpRegistry, CustomOpBuilder,
    Result, MinitensorError,
};
use std::collections::HashMap;
use std::sync::Arc;

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
        for op in self.custom_operations() {
            registry.register(op)?;
        }
        Ok(())
    }

    fn cleanup(&self, registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust example plugin cleaned up!");
        for op in self.custom_operations() {
            let _ = registry.unregister(op.name());
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

fn create_abs_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_abs", 1)
        .forward(|inputs| inputs[0].abs())
        .backward(|grad_output, input_ids, _, _, _| {
            let mut gradients = HashMap::new();
            if let Some(&id) = input_ids.first() {
                gradients.insert(id, grad_output.clone());
            }
            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs.len() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "Absolute value operation requires exactly one input",
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

fn create_clamp_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_clamp", 3)
        .forward(|inputs| {
            let x = inputs[0];
            let min_slice = inputs[1]
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::invalid_argument("Min must be f32 scalar"))?;
            let max_slice = inputs[2]
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::invalid_argument("Max must be f32 scalar"))?;
            let min_val = min_slice[0] as f64;
            let max_val = max_slice[0] as f64;
            x.clamp(Some(min_val), Some(max_val))
        })
        .backward(|grad_output, input_ids, _, _, _| {
            let mut gradients = HashMap::new();
            if let Some(&id) = input_ids.first() {
                gradients.insert(id, grad_output.clone());
            }
            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs.len() != 3 {
                return Err(MinitensorError::invalid_argument(
                    "Clamp operation requires exactly three inputs: tensor, min, max",
                ));
            }
            if inputs[1].shape().numel() != 1 || inputs[2].shape().numel() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "Min and max values must be scalars",
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

fn create_gelu_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_gelu", 1)
        .forward(|inputs| Ok(inputs[0].clone()))
        .backward(|grad_output, input_ids, _, _, _| {
            let mut gradients = HashMap::new();
            if let Some(&id) = input_ids.first() {
                gradients.insert(id, grad_output.clone());
            }
            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs.len() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "GELU operation requires exactly one input",
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

#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    let plugin = RustExamplePlugin::new();
    Box::into_raw(Box::new(plugin))
}

#[no_mangle]
pub extern "C" fn destroy_plugin(plugin: *mut dyn Plugin) {
    if !plugin.is_null() {
        unsafe {
            let _ = Box::from_raw(plugin);
        }
    }
}

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
