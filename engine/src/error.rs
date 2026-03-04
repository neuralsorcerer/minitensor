// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use thiserror::Error;

/// Result type alias for minitensor operations
pub type Result<T> = std::result::Result<T, MinitensorError>;

/// Comprehensive error types for minitensor operations with actionable suggestions
#[derive(Error, Debug, Clone)]
pub enum MinitensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeError {
        expected: Vec<usize>,
        actual: Vec<usize>,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Data type mismatch: expected {expected}, got {actual}")]
    TypeError {
        expected: String,
        actual: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Device mismatch: expected {expected}, got {actual}")]
    DeviceError {
        expected: String,
        actual: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Gradient computation error: {message}")]
    GradientError {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Memory allocation error: {message}")]
    MemoryError {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Invalid operation: {message}")]
    InvalidOperation {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Backend error ({backend}): {message}")]
    BackendError {
        backend: String,
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error(
        "Index out of bounds: index {index} is out of bounds for dimension {dim} with size {size}"
    )]
    IndexError {
        index: isize,
        dim: usize,
        size: usize,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Internal error: {message}")]
    InternalError {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Not implemented: {message}")]
    NotImplemented {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Invalid argument: {message}")]
    InvalidArgument {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Broadcasting error: cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Dimension error: {message}")]
    DimensionError {
        message: String,
        expected_dims: Option<usize>,
        actual_dims: Option<usize>,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Computation graph error: {message}")]
    ComputationGraphError {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Serialization error: {message}")]
    SerializationError {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Plugin error: {message}")]
    PluginError {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },

    #[error("Version mismatch: {message}")]
    VersionMismatch {
        message: String,
        suggestion: Option<String>,
        context: Option<String>,
    },
}

impl MinitensorError {
    /// Create a new shape error with suggestion
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        let suggestion = Self::generate_shape_suggestion(&expected, &actual);
        Self::ShapeError {
            expected,
            actual,
            suggestion: Some(suggestion),
            context: None,
        }
    }

    /// Create a shape error with custom context
    pub fn shape_mismatch_with_context(
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: impl Into<String>,
    ) -> Self {
        let suggestion = Self::generate_shape_suggestion(&expected, &actual);
        Self::ShapeError {
            expected,
            actual,
            suggestion: Some(suggestion),
            context: Some(context.into()),
        }
    }

    /// Create a simple shape error with message
    pub fn shape_error(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            message: message.into(),
            suggestion: None,
            context: None,
        }
    }

    /// Create a new type error with suggestion
    pub fn type_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        let expected_str = expected.into();
        let actual_str = actual.into();
        let suggestion = format!(
            "Use .to_dtype({}) to convert the tensor to the expected type",
            expected_str
        );

        Self::TypeError {
            expected: expected_str,
            actual: actual_str,
            suggestion: Some(suggestion),
            context: None,
        }
    }

    /// Create a type error with custom context
    pub fn type_mismatch_with_context(
        expected: impl Into<String>,
        actual: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        let expected_str = expected.into();
        let actual_str = actual.into();
        let suggestion = format!(
            "Use .to_dtype({}) to convert the tensor to the expected type",
            expected_str
        );

        Self::TypeError {
            expected: expected_str,
            actual: actual_str,
            suggestion: Some(suggestion),
            context: Some(context.into()),
        }
    }

    /// Create a new device error with suggestion
    pub fn device_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        let expected_str = expected.into();
        let actual_str = actual.into();
        let suggestion = format!(
            "Use .to({}) to move the tensor to the expected device",
            expected_str
        );

        Self::DeviceError {
            expected: expected_str,
            actual: actual_str,
            suggestion: Some(suggestion),
            context: None,
        }
    }

    /// Create a device error with custom context
    pub fn device_mismatch_with_context(
        expected: impl Into<String>,
        actual: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        let expected_str = expected.into();
        let actual_str = actual.into();
        let suggestion = format!(
            "Use .to({}) to move the tensor to the expected device",
            expected_str
        );

        Self::DeviceError {
            expected: expected_str,
            actual: actual_str,
            suggestion: Some(suggestion),
            context: Some(context.into()),
        }
    }

    /// Create a new gradient error
    pub fn gradient_error(message: impl Into<String>) -> Self {
        Self::GradientError {
            message: message.into(),
            suggestion: Some(
                "Ensure tensors have requires_grad=True and are part of a computation graph"
                    .to_string(),
            ),
            context: None,
        }
    }

    /// Create a gradient error with custom suggestion and context
    pub fn gradient_error_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
        context: Option<String>,
    ) -> Self {
        Self::GradientError {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context,
        }
    }

    /// Create a new memory error
    pub fn memory_error(message: impl Into<String>) -> Self {
        Self::MemoryError {
            message: message.into(),
            suggestion: Some("Try reducing batch size or using a smaller model".to_string()),
            context: None,
        }
    }

    /// Create a memory error with custom suggestion
    pub fn memory_error_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::MemoryError {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Create a new invalid operation error
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            message: message.into(),
            suggestion: None,
            context: None,
        }
    }

    /// Create an invalid operation error with suggestion
    pub fn invalid_operation_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::InvalidOperation {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Create a new backend error
    pub fn backend_error(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
            suggestion: Some(
                "Check if the backend is properly installed and configured".to_string(),
            ),
            context: None,
        }
    }

    /// Create a backend error with custom suggestion
    pub fn backend_error_with_suggestion(
        backend: impl Into<String>,
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Create a new index error with suggestion
    pub fn index_error(index: isize, dim: usize, size: usize) -> Self {
        Self::IndexError {
            index,
            dim,
            size,
            suggestion: Some(Self::generate_index_suggestion(index, size)),
            context: None,
        }
    }

    /// Create an index error with custom context
    pub fn index_error_with_context(
        index: isize,
        dim: usize,
        size: usize,
        context: impl Into<String>,
    ) -> Self {
        Self::IndexError {
            index,
            dim,
            size,
            suggestion: Some(Self::generate_index_suggestion(index, size)),
            context: Some(context.into()),
        }
    }

    /// Create a new internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
            suggestion: Some(
                "This is likely a bug. Please report it with a minimal reproduction case"
                    .to_string(),
            ),
            context: None,
        }
    }

    /// Create a new not implemented error
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::NotImplemented {
            message: message.into(),
            suggestion: Some("This feature is planned for a future release".to_string()),
            context: None,
        }
    }

    /// Create a not implemented error with custom suggestion
    pub fn not_implemented_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::NotImplemented {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Create a new invalid argument error
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
            suggestion: None,
            context: None,
        }
    }

    /// Create an invalid argument error with suggestion
    pub fn invalid_argument_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::InvalidArgument {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Create a new broadcasting error
    pub fn broadcast_error(shape1: Vec<usize>, shape2: Vec<usize>) -> Self {
        let suggestion = Self::generate_broadcast_suggestion(&shape1, &shape2);
        Self::BroadcastError {
            shape1,
            shape2,
            suggestion: Some(suggestion),
            context: None,
        }
    }

    /// Create a new dimension error
    pub fn dimension_error(
        message: impl Into<String>,
        expected_dims: Option<usize>,
        actual_dims: Option<usize>,
    ) -> Self {
        let suggestion = match (expected_dims, actual_dims) {
            (Some(expected), Some(actual)) => {
                if expected > actual {
                    Some(format!(
                        "Use .unsqueeze() to add dimensions or .view() to reshape"
                    ))
                } else {
                    Some(format!(
                        "Use .squeeze() to remove dimensions or .view() to reshape"
                    ))
                }
            }
            _ => Some("Check tensor dimensions and use reshape operations if needed".to_string()),
        };

        Self::DimensionError {
            message: message.into(),
            expected_dims,
            actual_dims,
            suggestion,
            context: None,
        }
    }

    /// Create a new computation graph error
    pub fn computation_graph_error(message: impl Into<String>) -> Self {
        Self::ComputationGraphError {
            message: message.into(),
            suggestion: Some(
                "Ensure all tensors are properly connected in the computation graph".to_string(),
            ),
            context: None,
        }
    }

    /// Create a new serialization error
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
            suggestion: Some("Check file permissions and available disk space".to_string()),
            context: None,
        }
    }

    /// Create a new plugin error
    pub fn plugin_error(message: impl Into<String>) -> Self {
        Self::PluginError {
            message: message.into(),
            suggestion: Some("Check plugin compatibility and installation".to_string()),
            context: None,
        }
    }

    /// Create a plugin error with custom suggestion
    pub fn plugin_error_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::PluginError {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Create a new version mismatch error
    pub fn version_mismatch(message: impl Into<String>) -> Self {
        Self::VersionMismatch {
            message: message.into(),
            suggestion: Some("Update the plugin or minitensor to compatible versions".to_string()),
            context: None,
        }
    }

    /// Create a version mismatch error with custom suggestion
    pub fn version_mismatch_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::VersionMismatch {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            context: None,
        }
    }

    /// Generate helpful suggestion for indexing errors
    fn generate_index_suggestion(index: isize, size: usize) -> String {
        if index < 0 {
            if size == 0 {
                "Cannot index into an empty dimension; ensure the tensor has elements before indexing"
                    .to_string()
            } else {
                format!(
                    "Use positive indices (0 to {}) or negative indices (-{} to -1)",
                    size - 1,
                    size
                )
            }
        } else {
            format!("Index must be in range [0, {})", size)
        }
    }

    /// Generate helpful suggestion for shape mismatches
    fn generate_shape_suggestion(expected: &[usize], actual: &[usize]) -> String {
        if expected.len() != actual.len() {
            format!(
                "Expected {} dimensions but got {}. Use .view() or .reshape() to change shape, or .unsqueeze()/.squeeze() to add/remove dimensions",
                expected.len(),
                actual.len()
            )
        } else {
            let mismatched_dims: Vec<_> = expected
                .iter()
                .zip(actual.iter())
                .enumerate()
                .filter(|(_, (e, a))| e != a)
                .map(|(i, (e, a))| format!("dim {}: expected {}, got {}", i, e, a))
                .collect();

            format!(
                "Shape mismatch in dimensions: {}. Use .view() or .reshape() to change the tensor shape",
                mismatched_dims.join(", ")
            )
        }
    }

    /// Generate helpful suggestion for broadcasting errors
    fn generate_broadcast_suggestion(shape1: &[usize], shape2: &[usize]) -> String {
        format!(
            "Shapes {:?} and {:?} cannot be broadcast together. Use .expand(), .unsqueeze(), or .view() to make shapes compatible",
            shape1, shape2
        )
    }

    /// Get the suggestion for this error, if any
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            Self::ShapeError { suggestion, .. } => suggestion.as_deref(),
            Self::TypeError { suggestion, .. } => suggestion.as_deref(),
            Self::DeviceError { suggestion, .. } => suggestion.as_deref(),
            Self::GradientError { suggestion, .. } => suggestion.as_deref(),
            Self::MemoryError { suggestion, .. } => suggestion.as_deref(),
            Self::InvalidOperation { suggestion, .. } => suggestion.as_deref(),
            Self::BackendError { suggestion, .. } => suggestion.as_deref(),
            Self::IndexError { suggestion, .. } => suggestion.as_deref(),
            Self::InternalError { suggestion, .. } => suggestion.as_deref(),
            Self::NotImplemented { suggestion, .. } => suggestion.as_deref(),
            Self::InvalidArgument { suggestion, .. } => suggestion.as_deref(),
            Self::BroadcastError { suggestion, .. } => suggestion.as_deref(),
            Self::DimensionError { suggestion, .. } => suggestion.as_deref(),
            Self::ComputationGraphError { suggestion, .. } => suggestion.as_deref(),
            Self::SerializationError { suggestion, .. } => suggestion.as_deref(),
            Self::PluginError { suggestion, .. } => suggestion.as_deref(),
            Self::VersionMismatch { suggestion, .. } => suggestion.as_deref(),
        }
    }

    /// Get the context for this error, if any
    pub fn context(&self) -> Option<&str> {
        match self {
            Self::ShapeError { context, .. } => context.as_deref(),
            Self::TypeError { context, .. } => context.as_deref(),
            Self::DeviceError { context, .. } => context.as_deref(),
            Self::GradientError { context, .. } => context.as_deref(),
            Self::MemoryError { context, .. } => context.as_deref(),
            Self::InvalidOperation { context, .. } => context.as_deref(),
            Self::BackendError { context, .. } => context.as_deref(),
            Self::IndexError { context, .. } => context.as_deref(),
            Self::InternalError { context, .. } => context.as_deref(),
            Self::NotImplemented { context, .. } => context.as_deref(),
            Self::InvalidArgument { context, .. } => context.as_deref(),
            Self::BroadcastError { context, .. } => context.as_deref(),
            Self::DimensionError { context, .. } => context.as_deref(),
            Self::ComputationGraphError { context, .. } => context.as_deref(),
            Self::SerializationError { context, .. } => context.as_deref(),
            Self::PluginError { context, .. } => context.as_deref(),
            Self::VersionMismatch { context, .. } => context.as_deref(),
        }
    }

    /// Create a formatted error message with suggestion and context
    pub fn detailed_message(&self) -> String {
        let mut message = self.to_string();

        if let Some(suggestion) = self.suggestion() {
            message.push_str(&format!("\n💡 Suggestion: {}", suggestion));
        }

        if let Some(context) = self.context() {
            message.push_str(&format!("\n📍 Context: {}", context));
        }

        message
    }
}

#[cfg(test)]
mod tests {
    use super::MinitensorError;

    #[test]
    fn test_shape_and_dimension_suggestion_branches() {
        let rank = MinitensorError::shape_mismatch(vec![2, 3], vec![6]);
        assert!(
            rank.suggestion()
                .unwrap()
                .contains("Expected 2 dimensions but got 1")
        );

        let dims = MinitensorError::shape_mismatch(vec![2, 3], vec![2, 4]);
        assert!(
            dims.suggestion()
                .unwrap()
                .contains("dim 1: expected 3, got 4")
        );

        let add_dims = MinitensorError::dimension_error("expand", Some(4), Some(2));
        assert!(add_dims.suggestion().unwrap().contains(".unsqueeze()"));

        let remove_dims = MinitensorError::dimension_error("reduce", Some(2), Some(4));
        assert!(remove_dims.suggestion().unwrap().contains(".squeeze()"));

        let equal_dims = MinitensorError::dimension_error("same-rank", Some(3), Some(3));
        assert!(equal_dims.suggestion().unwrap().contains(".squeeze()"));

        let fallback = MinitensorError::dimension_error("unknown", None, None);
        assert!(
            fallback
                .suggestion()
                .unwrap()
                .contains("Check tensor dimensions")
        );
    }

    #[test]
    fn test_builder_specific_suggestions_and_context() {
        let dtype = MinitensorError::type_mismatch_with_context("f32", "i64", "cast input");
        assert_eq!(dtype.context(), Some("cast input"));
        assert!(dtype.suggestion().unwrap().contains(".to_dtype(f32)"));

        let device = MinitensorError::device_mismatch_with_context("cpu", "cuda", "device copy");
        assert_eq!(device.context(), Some("device copy"));
        assert!(device.suggestion().unwrap().contains(".to(cpu)"));

        let neg_index = MinitensorError::index_error(-1, 0, 5);
        assert!(neg_index.suggestion().unwrap().contains("negative indices"));

        let empty_neg_index = MinitensorError::index_error(-1, 0, 0);
        assert!(
            empty_neg_index
                .suggestion()
                .unwrap()
                .contains("empty dimension")
        );

        let pos_index = MinitensorError::index_error_with_context(7, 1, 4, "slice op");
        assert_eq!(pos_index.context(), Some("slice op"));
        assert!(pos_index.suggestion().unwrap().contains("range [0, 4)"));

        let zero_len_pos_index = MinitensorError::index_error(0, 0, 0);
        assert_eq!(
            zero_len_pos_index.suggestion(),
            Some("Index must be in range [0, 0)")
        );

        let empty_neg_index_with_ctx = MinitensorError::index_error_with_context(-1, 0, 0, "take");
        assert_eq!(empty_neg_index_with_ctx.context(), Some("take"));
        assert!(
            empty_neg_index_with_ctx
                .suggestion()
                .unwrap()
                .contains("empty dimension")
        );

        let broadcast = MinitensorError::broadcast_error(vec![2, 3], vec![4, 5]);
        assert!(
            broadcast
                .suggestion()
                .unwrap()
                .contains("cannot be broadcast together")
        );

        let gradient = MinitensorError::gradient_error("missing graph node");
        assert!(
            gradient
                .suggestion()
                .unwrap()
                .contains("requires_grad=True")
        );
    }

    #[test]
    fn test_detailed_message_formatting_with_and_without_optional_fields() {
        let with_all = MinitensorError::gradient_error_with_suggestion(
            "missing grad",
            "set requires_grad",
            Some("backward pass".to_string()),
        );
        let rendered = with_all.detailed_message();
        assert!(rendered.contains("Gradient computation error: missing grad"));
        assert!(rendered.contains("💡 Suggestion: set requires_grad"));
        assert!(rendered.contains("📍 Context: backward pass"));

        let base_only = MinitensorError::invalid_argument("bad argument").detailed_message();
        assert_eq!(base_only, "Invalid argument: bad argument");

        let suggestion_only =
            MinitensorError::invalid_argument_with_suggestion("arg", "fix").detailed_message();
        assert!(suggestion_only.contains("Invalid argument: arg"));
        assert!(suggestion_only.contains("💡 Suggestion: fix"));
        assert!(!suggestion_only.contains("📍 Context:"));

        let context_only = MinitensorError::ShapeError {
            expected: vec![1],
            actual: vec![2],
            suggestion: None,
            context: Some("shape op".into()),
        }
        .detailed_message();
        assert!(context_only.contains("Shape mismatch: expected [1], got [2]"));
        assert!(!context_only.contains("💡 Suggestion:"));
        assert!(context_only.contains("📍 Context: shape op"));
    }

    #[test]
    fn test_all_constructor_helpers_and_accessor_match_arms() {
        let constructed = vec![
            MinitensorError::shape_mismatch_with_context(vec![1, 2], vec![2, 1], "matmul"),
            MinitensorError::shape_error("bad shape"),
            MinitensorError::type_mismatch("f32", "i32"),
            MinitensorError::device_mismatch("cpu", "cuda"),
            MinitensorError::memory_error("oom"),
            MinitensorError::memory_error_with_suggestion("oom", "free cache"),
            MinitensorError::invalid_operation("bad op"),
            MinitensorError::invalid_operation_with_suggestion("bad op", "use add"),
            MinitensorError::backend_error("cuda", "driver"),
            MinitensorError::backend_error_with_suggestion("cuda", "driver", "reinstall"),
            MinitensorError::internal_error("panic"),
            MinitensorError::not_implemented("todo"),
            MinitensorError::not_implemented_with_suggestion("todo", "track issue"),
            MinitensorError::invalid_argument("arg"),
            MinitensorError::invalid_argument_with_suggestion("arg", "fix input"),
            MinitensorError::computation_graph_error("cycle"),
            MinitensorError::serialization_error("io"),
            MinitensorError::plugin_error("missing"),
            MinitensorError::plugin_error_with_suggestion("missing", "install plugin"),
            MinitensorError::version_mismatch("abi"),
            MinitensorError::version_mismatch_with_suggestion("abi", "upgrade"),
        ];

        for err in constructed {
            let _ = err.to_string();
            let _ = err.detailed_message();
        }

        // Explicit per-variant values ensure every accessor match arm is hit in one place,
        // without duplicative assertions across many tests.
        let accessor_arms = [
            MinitensorError::ShapeError {
                expected: vec![1],
                actual: vec![2],
                suggestion: None,
                context: Some("ctx".into()),
            },
            MinitensorError::TypeError {
                expected: "f32".into(),
                actual: "i32".into(),
                suggestion: Some("s".into()),
                context: None,
            },
            MinitensorError::DeviceError {
                expected: "cpu".into(),
                actual: "cuda".into(),
                suggestion: None,
                context: None,
            },
            MinitensorError::GradientError {
                message: "m".into(),
                suggestion: Some("s".into()),
                context: Some("c".into()),
            },
            MinitensorError::MemoryError {
                message: "m".into(),
                suggestion: None,
                context: None,
            },
            MinitensorError::InvalidOperation {
                message: "m".into(),
                suggestion: Some("s".into()),
                context: Some("c".into()),
            },
            MinitensorError::BackendError {
                backend: "b".into(),
                message: "m".into(),
                suggestion: None,
                context: None,
            },
            MinitensorError::IndexError {
                index: -1,
                dim: 0,
                size: 4,
                suggestion: Some("s".into()),
                context: None,
            },
            MinitensorError::InternalError {
                message: "m".into(),
                suggestion: Some("s".into()),
                context: None,
            },
            MinitensorError::NotImplemented {
                message: "m".into(),
                suggestion: None,
                context: Some("c".into()),
            },
            MinitensorError::InvalidArgument {
                message: "m".into(),
                suggestion: Some("s".into()),
                context: None,
            },
            MinitensorError::BroadcastError {
                shape1: vec![1],
                shape2: vec![1],
                suggestion: None,
                context: Some("c".into()),
            },
            MinitensorError::DimensionError {
                message: "m".into(),
                expected_dims: Some(1),
                actual_dims: Some(2),
                suggestion: Some("s".into()),
                context: None,
            },
            MinitensorError::ComputationGraphError {
                message: "m".into(),
                suggestion: None,
                context: Some("c".into()),
            },
            MinitensorError::SerializationError {
                message: "m".into(),
                suggestion: Some("s".into()),
                context: None,
            },
            MinitensorError::PluginError {
                message: "m".into(),
                suggestion: None,
                context: None,
            },
            MinitensorError::VersionMismatch {
                message: "m".into(),
                suggestion: Some("s".into()),
                context: Some("c".into()),
            },
        ];

        for err in accessor_arms {
            let _ = err.suggestion();
            let _ = err.context();
        }
    }

    #[test]
    fn test_type_and_dimension_contextual_paths() {
        let ty = MinitensorError::type_mismatch_with_context("f16", "i8", "cast kernel");
        assert_eq!(ty.context(), Some("cast kernel"));
        assert!(
            ty.suggestion()
                .unwrap()
                .contains("Use .to_dtype(f16) to convert")
        );

        let grow_rank = MinitensorError::dimension_error("rank too small", Some(4), Some(2));
        assert!(grow_rank.suggestion().unwrap().contains(".unsqueeze()"));

        let shrink_rank = MinitensorError::dimension_error("rank too large", Some(2), Some(4));
        assert!(shrink_rank.suggestion().unwrap().contains(".squeeze()"));

        let unknown_rank = MinitensorError::dimension_error("unknown", None, Some(4));
        assert!(
            unknown_rank
                .suggestion()
                .unwrap()
                .contains("Check tensor dimensions")
        );
    }
}
