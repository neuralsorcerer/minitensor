// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::Device;
use pyo3::prelude::*;

/// Python wrapper for Device
#[pyclass(name = "Device")]
#[derive(Clone)]
pub struct PyDevice {
    inner: Device,
}

#[pymethods]
impl PyDevice {
    /// Create a new device
    #[new]
    fn new(device_str: &str) -> PyResult<Self> {
        let device = Device::from_str(device_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Ok(Self { inner: device })
    }

    /// Create a CPU device
    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: Device::cpu(),
        }
    }

    /// Create a CUDA device
    #[staticmethod]
    fn cuda(device_id: Option<usize>) -> Self {
        Self {
            inner: Device::cuda(device_id),
        }
    }

    /// Create a Metal device
    #[staticmethod]
    fn metal() -> Self {
        Self {
            inner: Device::metal(),
        }
    }

    /// Create an OpenCL device
    #[staticmethod]
    fn opencl(device_id: Option<usize>) -> Self {
        Self {
            inner: Device::opencl(device_id),
        }
    }

    /// Get device type as string
    #[getter]
    fn device_type(&self) -> String {
        format!("{:?}", self.inner.device_type())
    }

    /// Get device ID
    #[getter]
    fn device_id(&self) -> Option<usize> {
        self.inner.device_id()
    }

    /// Check if this is a CPU device
    fn is_cpu(&self) -> bool {
        self.inner.is_cpu()
    }

    /// Check if this is a GPU device
    fn is_gpu(&self) -> bool {
        self.inner.is_gpu()
    }

    /// String representation
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    /// String representation
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

impl PyDevice {
    /// Get the inner device
    pub fn device(&self) -> Device {
        self.inner
    }
}
