// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Backend;
use crate::{device::Device, error::Result};
use std::alloc::{alloc, dealloc, Layout};

/// CPU backend for tensor operations
pub struct CpuBackend {
    device: Device,
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        self.device
    }

    fn is_available() -> bool {
        true // CPU is always available
    }

    fn initialize() -> Result<Self> {
        Ok(Self {
            device: Device::cpu(),
        })
    }

    fn allocate(&self, size_bytes: usize) -> Result<*mut u8> {
        if size_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let layout = Layout::from_size_align(size_bytes, 64) // 64-byte alignment for SIMD
            .map_err(|e| crate::error::MinitensorError::memory_error(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Failed to allocate memory"
            ));
        }

        Ok(ptr)
    }

    fn deallocate(&self, ptr: *mut u8, size_bytes: usize) -> Result<()> {
        if ptr.is_null() || size_bytes == 0 {
            return Ok(());
        }

        let layout = Layout::from_size_align(size_bytes, 64)
            .map_err(|e| crate::error::MinitensorError::memory_error(format!("Invalid layout: {}", e)))?;

        unsafe { dealloc(ptr, layout) };
        Ok(())
    }

    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        if dst.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null destination pointer"
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        if src.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null source pointer"
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        assert!(CpuBackend::is_available());
        
        let backend = CpuBackend::initialize().unwrap();
        assert_eq!(backend.device(), Device::cpu());
    }

    #[test]
    fn test_memory_allocation() {
        let backend = CpuBackend::initialize().unwrap();
        
        let ptr = backend.allocate(1024).unwrap();
        assert!(!ptr.is_null());
        
        backend.deallocate(ptr, 1024).unwrap();
    }

    #[test]
    fn test_memory_copy() {
        let backend = CpuBackend::initialize().unwrap();
        
        let src_data = vec![1u8, 2, 3, 4, 5];
        let ptr = backend.allocate(5).unwrap();
        
        backend.copy_from_host(ptr, &src_data).unwrap();
        
        let mut dst_data = vec![0u8; 5];
        backend.copy_to_host(&mut dst_data, ptr).unwrap();
        
        assert_eq!(src_data, dst_data);
        
        backend.deallocate(ptr, 5).unwrap();
    }
}