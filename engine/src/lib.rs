// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

// Three style lints are allowed crate-wide. `needless_range_loop` and
// `too_many_arguments` fight numeric-kernel idioms: stencil loops whose index
// feeds stride arithmetic (not just slice access), and kernel helpers that
// take many scalar parameters. `items_after_test_module` is a false positive
// of the current `include!`-based module layout (multiple files share one
// module, so items legitimately follow an included test module); it can be
// re-enabled once the layout migration in docs/architecture_review.md lands.
// Every other clippy lint is enforced; CI runs `cargo clippy -- -D warnings`.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::items_after_test_module)]

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
pub mod random;
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
pub use random::manual_seed;
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
