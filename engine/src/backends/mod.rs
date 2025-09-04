// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
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
pub fn get_backend(device: Device) -> Result<Box<dyn Backend>> {
    match device.device_type() {
        crate::device::DeviceType::Cpu => Ok(Box::new(cpu::CpuBackend::initialize()?)),
        #[cfg(feature = "cuda")]
        crate::device::DeviceType::CUDA => Ok(Box::new(cuda::CudaBackend::initialize()?)),
        #[cfg(feature = "metal")]
        crate::device::DeviceType::Metal => Ok(Box::new(metal::MetalBackend::initialize()?)),
        #[cfg(feature = "opencl")]
        crate::device::DeviceType::OpenCL => Ok(Box::new(opencl::OpenCLBackend::initialize()?)),
        _ => Err(crate::error::MinitensorError::backend_error(
            "Unknown",
            format!("Backend not available for device: {}", device),
        )),
    }
}
