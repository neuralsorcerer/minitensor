// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use pyo3::prelude::*;
use engine::MinitensorError;

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