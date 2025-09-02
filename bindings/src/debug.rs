// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::tensor::PyTensor;
use engine::debug::{MemoryTracker, OperationProfiler, TensorDebugger, TensorInfo};
use pyo3::prelude::*;

/// Python wrapper for TensorInfo
#[pyclass(name = "TensorInfo")]
pub struct PyTensorInfo {
    inner: TensorInfo,
}

#[pymethods]
impl PyTensorInfo {
    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.dims().to_vec()
    }

    /// Get tensor data type
    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype.clone()
    }

    /// Get tensor device
    #[getter]
    fn device(&self) -> String {
        format!("{:?}", self.inner.device)
    }

    /// Get number of elements
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel
    }

    /// Check if requires gradients
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    /// Check if is leaf node
    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf
    }

    /// Get memory usage in bytes
    #[getter]
    fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes
    }

    /// Get memory usage in MB
    #[getter]
    fn memory_usage_mb(&self) -> f64 {
        self.inner.memory_usage_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get stride information
    #[getter]
    fn stride(&self) -> Vec<usize> {
        self.inner.stride.clone()
    }

    /// Get summary string
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Get detailed information string
    fn detailed(&self) -> String {
        self.inner.detailed()
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!("TensorInfo({})", self.summary())
    }
}

/// Python wrapper for TensorDebugger
#[pyclass(name = "TensorDebugger")]
pub struct PyTensorDebugger;

#[pymethods]
impl PyTensorDebugger {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Inspect a tensor and return detailed information
    fn inspect(&self, tensor: &PyTensor) -> String {
        TensorDebugger::inspect(tensor.tensor())
    }

    /// Compare two tensors and highlight differences
    fn compare(&self, tensor1: &PyTensor, tensor2: &PyTensor) -> String {
        TensorDebugger::compare(tensor1.tensor(), tensor2.tensor())
    }

    /// Perform health check on a tensor
    fn health_check(&self, tensor: &PyTensor) -> Vec<String> {
        TensorDebugger::health_check(tensor.tensor())
    }

    /// Get tensor info object
    fn get_info(&self, tensor: &PyTensor) -> PyTensorInfo {
        PyTensorInfo {
            inner: TensorInfo::from_tensor(tensor.tensor()),
        }
    }
}

/// Python wrapper for MemoryTracker
#[pyclass(name = "MemoryTracker")]
pub struct PyMemoryTracker {
    inner: MemoryTracker,
}

#[pymethods]
impl PyMemoryTracker {
    #[new]
    fn new() -> Self {
        Self {
            inner: MemoryTracker::new(),
        }
    }

    /// Record a memory allocation
    fn allocate(&mut self, name: String, size: usize) {
        self.inner.allocate(name, size);
    }

    /// Record a memory deallocation
    fn deallocate(&mut self, name: &str) {
        self.inner.deallocate(name);
    }

    /// Get current memory usage in bytes
    #[getter]
    fn current_usage(&self) -> usize {
        self.inner.current_usage()
    }

    /// Get peak memory usage in bytes
    #[getter]
    fn peak_usage(&self) -> usize {
        self.inner.peak_usage()
    }

    /// Get current memory usage in MB
    #[getter]
    fn current_usage_mb(&self) -> f64 {
        self.inner.current_usage_mb()
    }

    /// Get peak memory usage in MB
    #[getter]
    fn peak_usage_mb(&self) -> f64 {
        self.inner.peak_usage_mb()
    }

    /// Get memory usage summary
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Get detailed allocation information
    fn detailed_allocations(&self) -> String {
        self.inner.detailed_allocations()
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryTracker(current={:.2}MB, peak={:.2}MB)",
            self.inner.current_usage_mb(),
            self.inner.peak_usage_mb()
        )
    }
}

/// Python wrapper for OperationProfiler
#[pyclass(name = "OperationProfiler")]
pub struct PyOperationProfiler {
    inner: OperationProfiler,
}

#[pymethods]
impl PyOperationProfiler {
    #[new]
    fn new() -> Self {
        Self {
            inner: OperationProfiler::new(),
        }
    }

    /// Record timing for an operation
    fn record_timing(&mut self, operation: String, duration_ms: f64) {
        self.inner.record_timing(operation, duration_ms);
    }

    /// Record memory usage for an operation
    fn record_memory(&mut self, operation: String, memory_bytes: usize) {
        self.inner.record_memory(operation, memory_bytes);
    }

    /// Get average timing for an operation
    fn average_timing(&self, operation: &str) -> Option<f64> {
        self.inner.average_timing(operation)
    }

    /// Get average memory usage for an operation
    fn average_memory(&self, operation: &str) -> Option<f64> {
        self.inner.average_memory(operation)
    }

    /// Generate performance report
    fn report(&self) -> String {
        self.inner.report()
    }

    fn __str__(&self) -> String {
        self.report()
    }

    fn __repr__(&self) -> String {
        "OperationProfiler()".to_string()
    }
}

/// Context manager for timing operations
#[pyclass(name = "Timer")]
pub struct PyTimer {
    start_time: std::time::Instant,
    operation: String,
    profiler: Option<Py<PyOperationProfiler>>,
}

#[pymethods]
impl PyTimer {
    #[new]
    fn new(operation: String, profiler: Option<Py<PyOperationProfiler>>) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            operation,
            profiler,
        }
    }

    /// Enter the context manager
    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Exit the context manager and record timing
    fn __exit__(
        &mut self,
        py: Python,
        _exc_type: Option<&PyAny>,
        _exc_value: Option<&PyAny>,
        _traceback: Option<&PyAny>,
    ) -> PyResult<bool> {
        let duration = self.start_time.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;

        if let Some(ref profiler) = self.profiler {
            profiler
                .borrow_mut(py)
                .record_timing(self.operation.clone(), duration_ms);
        }

        Ok(false) // Don't suppress exceptions
    }

    /// Get elapsed time in milliseconds
    fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }
}

/// Helper function to create a timer context manager
#[pyfunction]
fn timer(operation: String, profiler: Option<Py<PyOperationProfiler>>) -> PyTimer {
    PyTimer::new(operation, profiler)
}

/// Module initialization
pub fn init_debug_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensorInfo>()?;
    m.add_class::<PyTensorDebugger>()?;
    m.add_class::<PyMemoryTracker>()?;
    m.add_class::<PyOperationProfiler>()?;
    m.add_class::<PyTimer>()?;
    m.add_function(wrap_pyfunction!(timer, m)?)?;
    Ok(())
}
