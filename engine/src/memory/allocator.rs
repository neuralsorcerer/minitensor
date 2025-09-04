// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{device::Device, error::Result};

/// Trait for memory allocators
pub trait Allocator: Send + Sync {
    /// Allocate memory of the given size
    fn allocate(&mut self, _size: usize) -> Result<*mut u8>;
    
    /// Deallocate previously allocated memory
    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()>;
    
    /// Get the device this allocator operates on
    fn device(&self) -> Device;
}

/// CPU memory allocator
pub struct CpuAllocator {
    device: Device,
}

/// CUDA memory allocator
#[cfg(feature = "cuda")]
pub struct CudaAllocator {
    device: Device,
}

/// Metal memory allocator
#[cfg(feature = "metal")]
pub struct MetalAllocator {
    device: Device,
}

/// OpenCL memory allocator
#[cfg(feature = "opencl")]
pub struct OpenCLAllocator {
    device: Device,
}

impl CpuAllocator {
    /// Create a new CPU allocator
    pub fn new() -> Self {
        Self {
            device: Device::cpu(),
        }
    }
}

impl Default for CpuAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl CudaAllocator {
    /// Create a new CUDA allocator
    pub fn new(device_id: Option<usize>) -> Self {
        Self {
            device: Device::cuda(device_id),
        }
    }
}

#[cfg(feature = "cuda")]
impl Allocator for CudaAllocator {
    fn allocate(&mut self, _size: usize) -> Result<*mut u8> {
                Err(crate::error::MinitensorError::backend_error(
            "CUDA allocator not yet implemented"
        ))
    }

    fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> Result<()> {
                Err(crate::error::MinitensorError::backend_error(
            "CUDA deallocator not yet implemented"
        ))
    }

    fn device(&self) -> Device {
        self.device
    }
}

#[cfg(feature = "metal")]
impl MetalAllocator {
    /// Create a new Metal allocator
    pub fn new(device_id: Option<usize>) -> Self {
        Self {
            device: Device::metal(device_id),
        }
    }
}

#[cfg(feature = "metal")]
impl Allocator for MetalAllocator {
    fn allocate(&mut self, _size: usize) -> Result<*mut u8> {
                Err(crate::error::MinitensorError::backend_error(
            "Metal allocator not yet implemented"
        ))
    }

    fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> Result<()> {
                Err(crate::error::MinitensorError::backend_error(
            "Metal deallocator not yet implemented"
        ))
    }

    fn device(&self) -> Device {
        self.device
    }
}

#[cfg(feature = "opencl")]
impl OpenCLAllocator {
    /// Create a new OpenCL allocator
    pub fn new(device_id: Option<usize>) -> Self {
        Self {
            device: Device::opencl(device_id),
        }
    }
}

#[cfg(feature = "opencl")]
impl Allocator for OpenCLAllocator {
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
                Err(crate::error::MinitensorError::backend_error(
            "OpenCL allocator not yet implemented"
        ))
    }

    fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> Result<()> {
                Err(crate::error::MinitensorError::backend_error(
            "OpenCL deallocator not yet implemented"
        ))
    }

    fn device(&self) -> Device {
        self.device
    }
}

impl Allocator for CpuAllocator {
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Ok(std::ptr::null_mut());
        }

        // Use system allocator for CPU memory
        let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|_| crate::error::MinitensorError::memory_error(
                format!("Invalid memory layout for size {}", size)
            ))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        
        if ptr.is_null() {
            Err(crate::error::MinitensorError::memory_error(
                format!("Failed to allocate {} bytes", size)
            ))
        } else {
            Ok(ptr)
        }
    }

    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        if ptr.is_null() || size == 0 {
            return Ok(());
        }

        let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|_| crate::error::MinitensorError::memory_error(
                format!("Invalid memory layout for size {}", size)
            ))?;

        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
        
        Ok(())
    }

    fn device(&self) -> Device {
        self.device
    }
}