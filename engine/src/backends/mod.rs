// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Device backends: contexts, allocators, and per-device kernels.
//!
//! Nothing in [`crate::ops`] dispatches here, and [`get_backend`] has no
//! callers outside this module's tests. Tensor execution is CPU-only and goes
//! straight through [`crate::tensor::TensorData`]'s host slices;
//! [`crate::device::Device::is_available`] is the authority on where a tensor
//! may be placed, and it answers `true` for CPU alone.
//!
//! This module is therefore groundwork rather than a live execution path. The
//! GPU submodules each carry the same small kernel set (add, mul, matmul,
//! relu, sigmoid) and are compiled only with their feature enabled, so nothing
//! here affects a default build.

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

// The `metal` crate is only declared for Apple targets (see engine/Cargo.toml),
// so every site guarded by the `metal` feature must also check the target --
// otherwise `--features gpu`, which turns the feature on unconditionally, would
// compile this module with no `metal` crate to compile it against.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;

#[cfg(feature = "opencl")]
pub mod opencl;

use crate::{device::Device, error::Result};

/// Trait for backend implementations
pub trait Backend: Send + Sync {
    /// Get the device this backend operates on
    fn device(&self) -> Device;

    /// Check if this backend is available
    fn is_available() -> bool
    where
        Self: Sized;

    /// Initialize the backend
    fn initialize() -> Result<Self>
    where
        Self: Sized;

    /// Allocate memory on this backend
    fn allocate(&self, size_bytes: usize) -> Result<*mut u8>;

    /// Deallocate memory on this backend
    fn deallocate(&self, ptr: *mut u8, size_bytes: usize) -> Result<()>;

    /// Copy data to this backend
    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()>;

    /// Copy data from this backend
    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()>;
}

/// Get the appropriate backend for a device
#[inline(always)]
pub fn get_backend(device: Device) -> Result<Box<dyn Backend>> {
    match device.device_type() {
        crate::device::DeviceType::Cpu => Ok(Box::new(cpu::CpuBackend::initialize()?)),
        #[cfg(feature = "cuda")]
        crate::device::DeviceType::Cuda => Ok(Box::new(cuda::CudaBackend::initialize()?)),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        crate::device::DeviceType::Metal => Ok(Box::new(metal::MetalBackend::initialize()?)),
        #[cfg(feature = "opencl")]
        crate::device::DeviceType::OpenCL => Ok(Box::new(opencl::OpenCLBackend::initialize()?)),
        _ => Err(crate::error::MinitensorError::backend_error(
            "Unknown",
            format!("Backend not available for device: {}", device),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_backend_cpu() {
        let backend = get_backend(Device::cpu()).unwrap();
        assert!(backend.device().is_cpu());
    }

    #[test]
    fn test_get_backend_unavailable() {
        #[cfg(not(feature = "cuda"))]
        assert!(get_backend(Device::cuda(Some(0))).is_err());
    }
}
