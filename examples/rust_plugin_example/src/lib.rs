// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    CustomOp, CustomOpBuilder, CustomOpRegistry, MinitensorError, Plugin, PluginInfo, Result,
    VersionInfo,
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Example plugin implementation
pub struct RustExamplePlugin {
    info: PluginInfo,
}

impl Default for RustExamplePlugin {
    fn default() -> Self {
        Self::new()
    }
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

    // The host registers and unregisters everything `custom_operations`
    // returns, so these hooks only cover this plugin's own setup and teardown —
    // registering the declared ops here would be a double registration and the
    // load would fail.
    fn initialize(&self, _registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust example plugin initialized!");
        Ok(())
    }

    fn cleanup(&self, _registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust example plugin cleaned up!");
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
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();
            if let Some(&id) = ctx.input_ids.first() {
                gradients.insert(id, ctx.grad_output.clone());
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
        .output_device(|input_devices| Ok(*input_devices[0]))
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
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();
            if let Some(&id) = ctx.input_ids.first() {
                gradients.insert(id, ctx.grad_output.clone());
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
        .output_device(|input_devices| Ok(*input_devices[0]))
        .build()
        .unwrap()
}

fn create_gelu_operation() -> Arc<dyn CustomOp> {
    CustomOpBuilder::new("rust_gelu", 1)
        .forward(|inputs| Ok(inputs[0].clone()))
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();
            if let Some(&id) = ctx.input_ids.first() {
                gradients.insert(id, ctx.grad_output.clone());
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
        .output_device(|input_devices| Ok(*input_devices[0]))
        .build()
        .unwrap()
}

// `create_plugin` is the whole ABI the host looks for, and it passes a
// Rust-native type across the `extern "C"` boundary: a `*mut dyn Plugin` is a fat
// pointer carrying a vtable address, which has no stable layout, so clippy's
// `improper_ctypes_definitions` fires — correctly.
//
// This is inherent to the host's plugin ABI, not a mistake here: the loader in
// `engine::plugins` resolves `create_plugin` as
// `unsafe extern "C" fn() -> *mut dyn Plugin`, the identical signature. Host and
// plugin therefore agree, but only as long as both are built by the same rustc
// against the same `engine` version and with the same global allocator — the
// host reclaims this `Box`, and frees the `Arc<dyn CustomOp>` values handed to
// it. A plugin compiled against a different engine build, or by a different
// compiler, will produce a mismatched vtable and corrupt at the first call.
// Anyone adapting this example must ship plugins alongside the exact host build;
// a `repr(C)` shim ABI would be needed to relax that, and it would be a redesign
// of the plugin system rather than a change here.
#[allow(improper_ctypes_definitions)]
#[unsafe(no_mangle)]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    let plugin = RustExamplePlugin::new();
    Box::into_raw(Box::new(plugin))
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
