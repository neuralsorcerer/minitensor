// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod tensor;
pub mod operations;
pub mod backends;
pub mod autograd;
pub mod nn;
pub mod optim;
pub mod error;
pub mod memory;
pub mod device;
pub mod hardware;
pub mod debug;
pub mod custom_ops;
pub mod plugins;
pub mod serialization;

// Re-export core types
pub use tensor::{Tensor, TensorData, DataType, TensorIndex};
pub use device::{Device, DeviceType};
pub use error::{MinitensorError, Result};
pub use autograd::{GradientFunction, ComputationGraph};
pub use hardware::{
    HardwareProfiler, HardwareProfile, SystemHardware, ResourceOptimizer,
    WorkloadAnalysis, ExecutionPlan, AllocationStrategy
};
pub use debug::{
    TensorInfo, TensorDebugger, MemoryTracker, GraphVisualizer, 
    OperationProfiler, GraphNode, GraphEdge
};
pub use custom_ops::{
    CustomOp, CustomOpRegistry, CustomOpBuilder, register_custom_op,
    unregister_custom_op, execute_custom_op, list_custom_ops, is_custom_op_registered
};
pub use plugins::{
    Plugin, PluginInfo, PluginManager, VersionInfo, register_plugin, unload_plugin,
    list_plugins, get_plugin_info, is_plugin_loaded
};
pub use serialization::{
    ModelVersion, ModelMetadata, SerializedTensor, StateDict, SerializedModel,
    SerializationFormat, ModelSerializer, DeploymentModel
};

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