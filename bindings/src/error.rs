// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::MinitensorError;
use pyo3::prelude::*;

/// Convert Rust errors to Python exceptions with detailed messages
pub fn _convert_error(err: MinitensorError) -> PyErr {
    // Use the detailed message that includes suggestions and context
    let detailed_msg = err.detailed_message();

    match err {
        MinitensorError::ShapeError { .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(detailed_msg)
        }
        MinitensorError::TypeError { .. } => {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(detailed_msg)
        }
        MinitensorError::DeviceError { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
        MinitensorError::GradientError { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
        MinitensorError::MemoryError { .. } => {
            PyErr::new::<pyo3::exceptions::PyMemoryError, _>(detailed_msg)
        }
        MinitensorError::InvalidOperation { .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(detailed_msg)
        }
        MinitensorError::BackendError { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
        MinitensorError::IndexError { .. } => {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(detailed_msg)
        }
        MinitensorError::InternalError { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
        MinitensorError::NotImplemented { .. } => {
            PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(detailed_msg)
        }
        MinitensorError::InvalidArgument { .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(detailed_msg)
        }
        MinitensorError::BroadcastError { .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(detailed_msg)
        }
        MinitensorError::DimensionError { .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(detailed_msg)
        }
        MinitensorError::ComputationGraphError { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
        MinitensorError::SerializationError { .. } => {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(detailed_msg)
        }
        MinitensorError::PluginError { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
        MinitensorError::VersionMismatch { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(detailed_msg)
        }
    }
}

/// Convert Rust errors to Python exceptions with enhanced error messages
/// This is a simplified version that focuses on clear error messages
pub fn _convert_error_detailed(err: MinitensorError) -> PyErr {
    // For now, just use the standard conversion with detailed messages
    // Custom exception classes can be added later if needed
    _convert_error(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_errors() -> Vec<(MinitensorError, &'static str)> {
        vec![
            (
                MinitensorError::ShapeError {
                    expected: vec![2, 2],
                    actual: vec![3, 1],
                    suggestion: None,
                    context: None,
                },
                "ValueError",
            ),
            (
                MinitensorError::TypeError {
                    expected: "float32".to_string(),
                    actual: "int32".to_string(),
                    suggestion: None,
                    context: None,
                },
                "TypeError",
            ),
            (
                MinitensorError::DeviceError {
                    expected: "cpu".to_string(),
                    actual: "cuda:0".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
            (
                MinitensorError::GradientError {
                    message: "bad grad".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
            (
                MinitensorError::MemoryError {
                    message: "oom".to_string(),
                    suggestion: None,
                    context: None,
                },
                "MemoryError",
            ),
            (
                MinitensorError::InvalidOperation {
                    message: "invalid".to_string(),
                    suggestion: None,
                    context: None,
                },
                "ValueError",
            ),
            (
                MinitensorError::BackendError {
                    backend: "cpu".to_string(),
                    message: "backend failed".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
            (
                MinitensorError::IndexError {
                    index: -1,
                    dim: 0,
                    size: 2,
                    suggestion: None,
                    context: None,
                },
                "IndexError",
            ),
            (
                MinitensorError::InternalError {
                    message: "internal".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
            (
                MinitensorError::NotImplemented {
                    message: "nyi".to_string(),
                    suggestion: None,
                    context: None,
                },
                "NotImplementedError",
            ),
            (
                MinitensorError::InvalidArgument {
                    message: "bad arg".to_string(),
                    suggestion: None,
                    context: None,
                },
                "ValueError",
            ),
            (
                MinitensorError::BroadcastError {
                    shape1: vec![2, 3],
                    shape2: vec![4],
                    suggestion: None,
                    context: None,
                },
                "ValueError",
            ),
            (
                MinitensorError::DimensionError {
                    message: "bad dim".to_string(),
                    expected_dims: Some(2),
                    actual_dims: Some(3),
                    suggestion: None,
                    context: None,
                },
                "ValueError",
            ),
            (
                MinitensorError::ComputationGraphError {
                    message: "graph".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
            (
                MinitensorError::SerializationError {
                    message: "io".to_string(),
                    suggestion: None,
                    context: None,
                },
                "OSError",
            ),
            (
                MinitensorError::PluginError {
                    message: "plugin".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
            (
                MinitensorError::VersionMismatch {
                    message: "version".to_string(),
                    suggestion: None,
                    context: None,
                },
                "RuntimeError",
            ),
        ]
    }

    #[test]
    fn convert_error_maps_all_variants_to_expected_python_types() {
        for (err, expected_name) in sample_errors() {
            Python::attach(|py| {
                let py_err = _convert_error(err.clone());
                let type_name = py_err
                    .get_type(py)
                    .name()
                    .expect("python exception type should have a name");
                assert_eq!(type_name, expected_name);
                assert!(!py_err.to_string().is_empty());
            });
        }
    }

    #[test]
    fn convert_error_detailed_delegates_to_standard_converter() {
        let err = MinitensorError::InvalidArgument {
            message: "delegation".to_string(),
            suggestion: None,
            context: None,
        };

        Python::attach(|py| {
            let standard = _convert_error(err.clone());
            let detailed = _convert_error_detailed(err.clone());
            let standard_name = standard.get_type(py).name().unwrap().to_string();
            let detailed_name = detailed.get_type(py).name().unwrap().to_string();
            assert_eq!(standard_name, detailed_name);
            assert_eq!(standard.to_string(), detailed.to_string());
        });
    }
}
