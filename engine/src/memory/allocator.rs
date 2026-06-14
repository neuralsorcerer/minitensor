// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{device::Device, error::Result};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, DevicePtr, UnifiedSlice};
#[cfg(feature = "cuda")]
use rustc_hash::FxHashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

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
    context: Option<Arc<CudaContext>>,
    allocations: FxHashMap<usize, UnifiedSlice<u8>>,
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
    #[inline]
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
        let device = Device::cuda(device_id);
        let context = CudaContext::new(device.id()).ok();
        Self {
            device,
            context,
            allocations: FxHashMap::default(),
        }
    }
}

#[cfg(feature = "cuda")]
impl Allocator for CudaAllocator {
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Ok(std::ptr::null_mut());
        }
        let context = self.context.as_ref().ok_or_else(|| {
            crate::error::MinitensorError::backend_error(
                "CUDA",
                format!("CUDA device {} is not available", self.device.id()),
            )
        })?;
        let slice = unsafe { context.alloc_unified::<u8>(size, true) }.map_err(|e| {
            crate::error::MinitensorError::memory_error(format!(
                "CUDA unified-memory allocation failed: {e}"
            ))
        })?;
        let stream = context.default_stream();
        let (raw, sync) = DevicePtr::device_ptr(&slice, &stream);
        let ptr = raw as usize as *mut u8;
        drop(sync);
        drop(stream);
        self.allocations.insert(ptr as usize, slice);
        Ok(ptr)
    }

    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let allocation_len = self
            .allocations
            .get(&(ptr as usize))
            .ok_or_else(|| crate::error::MinitensorError::memory_error("Unknown CUDA pointer"))?
            .len();
        if size != 0 && size != allocation_len {
            return Err(crate::error::MinitensorError::memory_error(format!(
                "CUDA deallocation size mismatch: got {} bytes for allocation of {} bytes",
                size, allocation_len
            )));
        }
        self.allocations.remove(&(ptr as usize));
        Ok(())
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
            "Metal",
            "Metal allocator not yet implemented",
        ))
    }

    fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> Result<()> {
        Err(crate::error::MinitensorError::backend_error(
            "Metal",
            "Metal deallocator not yet implemented",
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
            "OpenCL",
            "OpenCL allocator not yet implemented",
        ))
    }

    fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> Result<()> {
        Err(crate::error::MinitensorError::backend_error(
            "OpenCL",
            "OpenCL deallocator not yet implemented",
        ))
    }

    fn device(&self) -> Device {
        self.device
    }
}

impl Allocator for CpuAllocator {
    #[inline(always)]
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Ok(std::ptr::null_mut());
        }

        if size > isize::MAX as usize {
            return Err(crate::error::MinitensorError::memory_error(format!(
                "Invalid memory layout for size {}",
                size
            )));
        }

        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(size, 1) };
        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            Err(crate::error::MinitensorError::memory_error(format!(
                "Failed to allocate {} bytes",
                size
            )))
        } else {
            Ok(ptr)
        }
    }

    #[inline(always)]
    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        if ptr.is_null() || size == 0 {
            return Ok(());
        }

        if size > isize::MAX as usize {
            return Err(crate::error::MinitensorError::memory_error(format!(
                "Invalid memory layout for size {}",
                size
            )));
        }

        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(size, 1) };
        unsafe { std::alloc::dealloc(ptr, layout) };

        Ok(())
    }

    #[inline(always)]
    fn device(&self) -> Device {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allocator_zero() {
        let mut alloc = CpuAllocator::new();
        let ptr = alloc.allocate(0).unwrap();
        assert!(ptr.is_null());
        alloc.deallocate(ptr, 0).unwrap();
    }

    #[test]
    fn test_cpu_allocator_large_allocation_error() {
        let mut alloc = CpuAllocator::new();
        let res = alloc.allocate(usize::MAX);
        assert!(res.is_err());
    }
}
