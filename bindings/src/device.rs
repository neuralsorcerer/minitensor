// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::Device;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Reject a device that tensors cannot actually live on.
///
/// Every placement argument funnels through here so an unusable device fails
/// where the user named it. Without this the tensor is built, reports the
/// device it was asked for, and then fails inside the first operation applied
/// to it -- with an internal error telling the user to file a bug.
pub(crate) fn ensure_available(device: Device) -> PyResult<Device> {
    if device.is_available() {
        return Ok(device);
    }
    Err(PyRuntimeError::new_err(format!(
        "device '{device}' is not available: minitensor executes on the CPU only, \
         so tensors cannot be placed on {}. Use device='cpu' (the default).",
        device.device_type()
    )))
}

/// Resolve an optional `device=` argument, defaulting to `fallback`.
///
/// `fallback` is the device of whatever the result is modelled on (`*_like`
/// constructors) and is not re-checked: it came from a tensor that already
/// exists, so it passed this test when that tensor was built.
pub(crate) fn resolve_device_or(device: Option<&PyDevice>, fallback: Device) -> PyResult<Device> {
    match device {
        Some(device) => ensure_available(device.device()),
        None => Ok(fallback),
    }
}

/// Resolve an optional `device=` argument, defaulting to CPU.
pub(crate) fn resolve_device(device: Option<&PyDevice>) -> PyResult<Device> {
    resolve_device_or(device, Device::cpu())
}

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
    #[pyo3(signature = (device_id=None))]
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
    #[pyo3(signature = (device_id=None))]
    fn opencl(device_id: Option<usize>) -> Self {
        Self {
            inner: Device::opencl(device_id),
        }
    }

    /// Get device type as string
    #[getter]
    fn device_type(&self) -> String {
        self.inner.device_type().to_string()
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

    /// Check whether tensors can be placed on this device.
    ///
    /// True for CPU only: minitensor's kernels all run on the host, so tensor
    /// creation refuses any other device rather than returning a tensor that
    /// no operation accepts.
    fn is_available(&self) -> bool {
        self.inner.is_available()
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

    // `device_type` reports the same lower-case spelling the rest of the API
    // uses -- `str(device)`, `Device(...)` parsing, and `Tensor.dtype`. It used
    // to be Rust's `Debug` output ("Cpu", "OpenCL"), which matched nothing else
    // and could not be fed back to `Device(...)` without knowing to case-fold.
    #[test]
    fn constructors_and_properties_cover_cpu_and_gpu_paths() {
        let cpu = PyDevice::cpu();
        assert_eq!(cpu.device_type(), "cpu");
        assert_eq!(cpu.device_id(), None);
        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());
        assert_eq!(cpu.__str__(), cpu.__repr__());

        let cuda = PyDevice::cuda(Some(1));
        assert_eq!(cuda.device_type(), "cuda");
        assert_eq!(cuda.device_id(), Some(1));
        assert!(!cuda.is_cpu());
        assert!(cuda.is_gpu());

        let metal = PyDevice::metal();
        assert_eq!(metal.device_type(), "metal");
        assert!(metal.is_gpu());

        let opencl = PyDevice::opencl(Some(2));
        assert_eq!(opencl.device_type(), "opencl");
        assert_eq!(opencl.device_id(), Some(2));
        assert!(opencl.is_gpu());
    }

    #[test]
    fn new_accepts_valid_strings_and_rejects_invalid_ones() {
        let cpu = PyDevice::new("cpu").expect("cpu should parse");
        assert!(cpu.is_cpu());

        let cuda = PyDevice::new("cuda:3").expect("cuda should parse");
        assert_eq!(cuda.device_type(), "cuda");
        assert_eq!(cuda.device_id(), Some(3));

        let opencl = PyDevice::new("opencl:4").expect("opencl should parse");
        assert_eq!(opencl.device_type(), "opencl");
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
        assert_eq!(from_device.device_type(), "cuda");
        assert_eq!(from_device.device().device_id(), Some(7));

        let cpu = PyDevice::from_device(Device::cpu());
        assert!(cpu.device().is_cpu());
    }
}
