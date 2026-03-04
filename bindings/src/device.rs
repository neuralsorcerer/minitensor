// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::Device;
use pyo3::prelude::*;

/// Python wrapper for Device
#[pyclass(name = "Device", from_py_object)]
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
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

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

    pub(crate) fn from_device(device: Device) -> Self {
        Self { inner: device }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_and_properties_cover_cpu_and_gpu_paths() {
        let cpu = PyDevice::cpu();
        assert_eq!(cpu.device_type(), "Cpu");
        assert_eq!(cpu.device_id(), None);
        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());
        assert_eq!(cpu.__str__(), cpu.__repr__());

        let cuda = PyDevice::cuda(Some(1));
        assert_eq!(cuda.device_type(), "Cuda");
        assert_eq!(cuda.device_id(), Some(1));
        assert!(!cuda.is_cpu());
        assert!(cuda.is_gpu());

        let metal = PyDevice::metal();
        assert_eq!(metal.device_type(), "Metal");
        assert!(metal.is_gpu());

        let opencl = PyDevice::opencl(Some(2));
        assert_eq!(opencl.device_type(), "OpenCL");
        assert_eq!(opencl.device_id(), Some(2));
        assert!(opencl.is_gpu());
    }

    #[test]
    fn new_accepts_valid_strings_and_rejects_invalid_ones() {
        let cpu = PyDevice::new("cpu").expect("cpu should parse");
        assert!(cpu.is_cpu());

        let cuda = PyDevice::new("cuda:3").expect("cuda should parse");
        assert_eq!(cuda.device_type(), "Cuda");
        assert_eq!(cuda.device_id(), Some(3));

        let opencl = PyDevice::new("opencl:4").expect("opencl should parse");
        assert_eq!(opencl.device_type(), "OpenCL");
        assert_eq!(opencl.device_id(), Some(4));

        let err = match PyDevice::new("definitely-not-a-device") {
            Ok(_) => panic!("invalid device string should fail"),
            Err(err) => err,
        };
        Python::attach(|py| {
            assert_eq!(err.get_type(py).name().unwrap(), "ValueError");
            let message = err.to_string();
            assert!(!message.is_empty());
            assert!(message.contains("device") || message.contains("Device"));
        });
    }

    #[test]
    fn device_roundtrip_helpers_cover_conversion_paths() {
        let from_device = PyDevice::from_device(Device::cuda(Some(7)));
        assert_eq!(from_device.device_type(), "Cuda");
        assert_eq!(from_device.device().device_id(), Some(7));

        let cpu = PyDevice::from_device(Device::cpu());
        assert!(cpu.device().is_cpu());
    }
}
