// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#![allow(clippy::all)]

pub mod autograd;
pub mod backends;
pub mod custom_ops;
pub mod debug;
pub mod device;
pub mod error;
pub mod hardware;
pub mod memory;
pub mod nn;
pub mod operations;
pub mod optim;
pub mod plugins;
pub mod serialization;
pub mod tensor;

// Re-export core types
pub use autograd::{ComputationGraph, GradientFunction};
pub use custom_ops::{
    CustomOp, CustomOpBuilder, CustomOpRegistry, execute_custom_op, is_custom_op_registered,
    list_custom_ops, register_custom_op, unregister_custom_op,
};
pub use debug::{
    GraphEdge, GraphNode, GraphVisualizer, MemoryTracker, OperationProfiler, TensorDebugger,
    TensorInfo,
};
pub use device::{Device, DeviceType};
pub use error::{MinitensorError, Result};
pub use hardware::{
    AllocationStrategy, ExecutionPlan, HardwareProfile, HardwareProfiler, ResourceOptimizer,
    SystemHardware, WorkloadAnalysis,
};
pub use plugins::{
    Plugin, PluginInfo, PluginManager, VersionInfo, get_plugin_info, is_plugin_loaded,
    list_plugins, register_plugin, unload_plugin,
};
pub use serialization::{
    DeploymentModel, ModelMetadata, ModelSerializer, ModelVersion, SerializationFormat,
    SerializedModel, SerializedTensor, StateDict,
};
pub use tensor::{DataType, Tensor, TensorData, TensorIndex};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
